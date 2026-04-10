import re
import json
import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.tools.utils import logger, pipeline
from src.inquire.utils import load_clip
from src.inquire.metrics import MetricAverage, compute_retrieval_metrics


def to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj


def load_config(config_path="config.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return to_namespace(config)


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token_id
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


REWRITE_INIT_PROMPT = """
You are a prompt rewriting assistant for an image retrieval system.

You may call the tool `run_web_search` to look up common names, scientific names,
habitats, and visual descriptors that help retrieve images. Use web search only
when it improves clarity or disambiguates the query.

Rewrite the user query into a concise, visually descriptive prompt that keeps the
original intent. Avoid jargons.

Output format (JSON only, no extra text):
{"rewritten_prompt": "...", "used_search": true, "notes": "..."}
""".strip()


def build_rewrite_config(base_config):
    return SimpleNamespace(
        init_prompt=REWRITE_INIT_PROMPT,
        max_tool_calls=base_config.max_tool_calls,
        chat_template=base_config.chat_template,
        generate=base_config.generate,
    )


def _extract_first_json_block(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:idx + 1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_rewritten_prompt(text: str) -> str | None:
    payload = _extract_first_json_block(text)
    if isinstance(payload, dict):
        for key in ("rewritten_prompt", "rewrite", "prompt", "final_prompt"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    match = re.search(r"(?:Rewritten|Rewrite|Final) Prompt:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().splitlines()[0]

    return None


def rewrite_prompt(query_text: str, model, tokenizer, rewrite_config):
    response = pipeline(model, tokenizer, query_text, config=rewrite_config)
    rewritten = extract_rewritten_prompt(response)
    if not rewritten:
        rewritten = query_text
    return rewritten, response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite prompts with web search and rerank with CLIP models.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--debug", action="store_true", help="Run on a small subset of queries")
    args = parser.parse_args()

    config = load_config("configs/eval.yml")
    rewrite_config = build_rewrite_config(config)
    model, tokenizer = load_model(config.model.name)

    split = args.split
    dataset = load_dataset("evendrow/INQUIRE-Rerank", split=("validation" if split == "val" else "test"))
    queries = np.unique(dataset["query"]).tolist()
    if args.debug:
        queries = queries[:5]

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    rewrite_cache = {}
    rewrite_responses = {}
    for i, query_text in enumerate(queries):
        logger.info("[%s/%s] Rewriting query: %s", i + 1, len(queries), query_text)
        rewritten, response = rewrite_prompt(query_text, model, tokenizer, rewrite_config)
        rewrite_cache[query_text] = rewritten
        rewrite_responses[query_text] = response

    rewrite_path = output_dir / f"rewrite_prompts_{split}.json"
    with rewrite_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"rewritten": rewrite_cache, "responses": rewrite_responses},
            f,
            indent=2,
            ensure_ascii=True,
        )
    logger.info("Saved rewritten prompts to %s", rewrite_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 256
    num_workers = 4

    all_models = {
        "siglip-so400m-14-384": "open_clip:ViT-SO400M-14-SigLIP-384/webli",
    }

    results = []
    for title, clip_name in all_models.items():
        model_clip, preprocess, clip_tokenizer = load_clip(clip_name, use_jit=False, device=device)

        def collate_transform(examples):
            pixel_values = torch.cat([preprocess(ex["image"]).unsqueeze(0) for ex in examples])
            ids = [ex["inat24_image_id"] for ex in examples]
            return pixel_values, ids

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_transform,
            num_workers=num_workers,
        )

        image_emb_cache = {}
        for images, ids in tqdm(dataloader, total=len(dataset) // batch_size):
            with torch.no_grad(), torch.autocast(device):
                image_embs = model_clip.encode_image(images.to(device)).cpu()
                image_embs /= image_embs.norm(dim=-1, keepdim=True)
            image_emb_cache.update(dict(zip(ids, image_embs)))

        metrics_avg = MetricAverage()
        for query in queries:
            query_ds = dataset.select(np.argwhere(np.asarray(dataset["query"]) == query).squeeze())
            rewritten_query = rewrite_cache.get(query, query)

            text = clip_tokenizer(rewritten_query).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_emb = model_clip.encode_text(text).squeeze().cpu()
                text_emb /= text_emb.norm(dim=-1, keepdim=True)

            image_embs = torch.stack([image_emb_cache[image_id] for image_id in query_ds["inat24_image_id"]])
            y_pred = (image_embs.float() @ text_emb.float()).numpy()
            y_true = np.asarray(query_ds["relevant"])

            pr, rec, ap, ndcg, mrr = compute_retrieval_metrics(y_true, y_pred, count_pos=sum(y_true))
            metrics_avg.update([ap * 100, ndcg * 100, mrr])
            results.append(
                dict(
                    model=title,
                    query=query,
                    rewritten_query=rewritten_query,
                    ap=ap * 100,
                    ndcg=ndcg * 100,
                    mrr=mrr,
                )
            )

        ap, ndcg, mrr = metrics_avg.avg
        print(f"{title:30s}\t{ap:.1f}\t{ndcg:.1f}\t{mrr:.2f}")

    results_df = pd.DataFrame.from_dict(results)
    pd.options.display.float_format = " {:,.2f}".format
    print(results_df.groupby("model").agg({"ap": "mean", "ndcg": "mean", "mrr": "mean"}).sort_values("ap"))

    save_results_path = f"results_rerank_with_clip_rewrite_{split}.csv"
    results_df.to_csv(save_results_path, index=False)
    print("All done! Saved results to", save_results_path)