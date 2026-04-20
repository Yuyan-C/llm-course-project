"""Run text-to-image reranking on INQUIRE-Rerank with CLIP-like models."""

from contextlib import nullcontext
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import argparse
import time

from src.inquire.utils import load_clip
from src.inquire.metrics import MetricAverage, compute_retrieval_metrics
from src.inquire.web_image_quality import extract_image_embedding_tensor


def patch_transformers_tokenizer_compat() -> None:
    # all_clip expects some transformers tokenizers to expose batch_encode_plus.
    try:
        from transformers import T5Tokenizer, T5TokenizerFast  # type: ignore
    except Exception:
        return

    for cls in (T5Tokenizer, T5TokenizerFast):
        if cls is None:
            continue
        if hasattr(cls, "batch_encode_plus"):
            continue

        def _batch_encode_plus(self, *args, **kwargs):
            return self(*args, **kwargs)

        setattr(cls, "batch_encode_plus", _batch_encode_plus)


def _extract_text_embedding_tensor(model_output):
    if torch.is_tensor(model_output):
        return model_output

    for attr in ("text_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(model_output, attr, None)
        if torch.is_tensor(value):
            if attr == "last_hidden_state" and value.ndim >= 3:
                return value[:, 0, :]
            return value

    if isinstance(model_output, (tuple, list)) and len(model_output) > 0:
        first = model_output[0]
        if torch.is_tensor(first):
            if first.ndim >= 3:
                return first[:, 0, :]
            return first

    raise TypeError(f"Unsupported text embedding output type: {type(model_output).__name__}")


def _tokenize_for_model(tokenizer, text: str, device: str):
    tokens = tokenizer(text)
    if hasattr(tokens, "to"):
        return tokens.to(device)
    if isinstance(tokens, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in tokens.items()}
    raise TypeError(f"Unsupported tokenizer output type: {type(tokens).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T2I retrieval evaluation.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model keys to evaluate (e.g. vit-b-32 bioclip).",
    )
    parser.add_argument("--save-results-path", type=str, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-load-retries", type=int, default=3)
    parser.add_argument("--model-load-retry-sleep", type=float, default=5.0)
    return parser.parse_args()


args = parse_args()
patch_transformers_tokenizer_compat()

split = args.split
save_results_path = args.save_results_path or f"results_rerank_with_clip_{split}.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load INQUIRE-Rerank from HuggingFace
dataset = load_dataset("evendrow/INQUIRE-Rerank", split=('validation' if split == 'val' else 'test'))
queries = np.unique(dataset['query']).tolist()
if args.max_queries is not None:
    queries = queries[: args.max_queries]

batch_size = args.batch_size
num_workers = args.num_workers

all_models_available = {
    "vit-b-32": "hf_clip:openai/clip-vit-base-patch32",
    "bioclip": "bioclip",
    "biocap": "biocap",
    "siglip-vit-b-16": "open_clip:ViT-B-16-SigLIP-256/webli",
}

if args.models:
    unknown = [name for name in args.models if name not in all_models_available]
    if unknown:
        raise ValueError(f"Unknown model key(s): {unknown}. Valid keys: {list(all_models_available.keys())}")
    all_models = {name: all_models_available[name] for name in args.models}
else:
    all_models = all_models_available

results = []
for title, clip_name in all_models.items():
    model = preprocess = tokenizer = None
    last_exc = None
    for attempt in range(1, args.model_load_retries + 2):
        try:
            model, preprocess, tokenizer = load_clip(clip_name, use_jit=False, device=device)
            break
        except Exception as exc:  # noqa: PERF203
            last_exc = exc
            if attempt > args.model_load_retries:
                break
            print(
                f"[{title}] load failed on attempt {attempt}/{args.model_load_retries + 1}: "
                f"{exc.__class__.__name__}: {exc}. Retrying in {args.model_load_retry_sleep:.1f}s..."
            )
            time.sleep(args.model_load_retry_sleep)

    if model is None or preprocess is None or tokenizer is None:
        reason = f"{last_exc.__class__.__name__}: {last_exc}" if last_exc is not None else "unknown load error"
        raise RuntimeError(f"[{title}] failed to load after retries: {reason}") from last_exc

    # Efficiently compute image embeddings in batches
    def collate_transform(examples):
        pixel_values = torch.cat([preprocess(ex["image"]).unsqueeze(0) for ex in examples])
        ids = [ex['inat24_image_id'] for ex in examples]
        return pixel_values, ids
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_transform, num_workers=num_workers)

    image_emb_cache = {}
    for images, ids in tqdm(dataloader, total=len(dataset)//batch_size):
        with torch.no_grad(), torch.autocast(device):
            image_embs = extract_image_embedding_tensor(model.encode_image(images.to(device))).float().cpu()
            image_embs /= image_embs.norm(dim=-1, keepdim=True)
        image_emb_cache.update(dict(zip(ids, image_embs)))


    # Score images for each query by embedding similarity
    metrics_avg = MetricAverage()
    for query in queries:
        query_ds = dataset.select(np.argwhere(np.asarray(dataset['query']) == query).squeeze())

        text = _tokenize_for_model(tokenizer, query, device)
        amp_ctx = torch.autocast(device_type="cuda") if device == "cuda" else nullcontext()
        with torch.no_grad(), amp_ctx:
            if isinstance(text, dict):
                text_out = model.encode_text(**text)
            else:
                text_out = model.encode_text(text)
            text_emb = _extract_text_embedding_tensor(text_out).squeeze().float().cpu()
            text_emb /= text_emb.norm(dim=-1, keepdim=True)

        image_embs = torch.stack([image_emb_cache[image_id] for image_id in query_ds['inat24_image_id']])
        y_pred = (image_embs.float() @ text_emb.float()).numpy()
        y_true = np.asarray(query_ds['relevant'])

        pr, rec, ap, ndcg, mrr = compute_retrieval_metrics(y_true, y_pred, count_pos=sum(y_true))
        metrics_avg.update([ap*100, ndcg*100, mrr])
        results.append(dict(model=title, query=query, ap=ap*100, ndcg=ndcg*100, mrr=mrr))

    ap, ndcg, mrr = metrics_avg.avg
    print(f'{title:30s}\t{ap:.1f}\t{ndcg:.1f}\t{mrr:.2f}')

results_df = pd.DataFrame.from_dict(results)
pd.options.display.float_format = ' {:,.2f}'.format
if len(results_df) > 0:
    print(results_df.groupby('model').agg({'ap': 'mean', 'ndcg': 'mean', 'mrr': 'mean'}).sort_values('ap'))
else:
    print("No evaluation rows were produced.")
    results_df = pd.DataFrame(columns=["model", "query", "ap", "ndcg", "mrr"])

results_df.to_csv(save_results_path, index=False)
print("All done! Saved results to", save_results_path)
