"""This script runs evals for CLIP models on INQUIRE-Rerank, the reranking task.
The data is automatically loaded from HuggingFace Hub, so you don't need to download 
anything yourself to run this evaluation."""
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from collections import defaultdict 
import argparse
from PIL import Image

from src.inquire.utils import load_clip
from src.inquire.metrics import MetricAverage, compute_retrieval_metrics


# Command line argument parser
parser = argparse.ArgumentParser(description='Run retrieval evaluation.')
parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                    help="Dataset split to evaluate on. Options: 'val', 'test'. Default is 'test'.")
parser.add_argument('--metadata', type=str, default='/network/scratch/y/yuyan.chen/inquire/web_images/image_search_metadata_test.json',
                    help='Path to the downloaded image metadata JSON.')
args = parser.parse_args()

split = args.split
save_results_path = f'results_rerank_with_clip_{split}.csv'

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load INQUIRE-Rerank from HuggingFace
dataset = load_dataset("evendrow/INQUIRE-Rerank", split=('validation' if split == 'val' else 'test'))
queries = np.unique(dataset['query']).tolist()

metadata_path = Path(args.metadata)
with metadata_path.open('r', encoding='utf-8') as f:
    metadata_records = json.load(f)

query_to_downloads = {}
for record in metadata_records:
    if not isinstance(record, dict):
        continue
    query_text = record.get('query')
    image_paths = [p for p in record.get('image_paths', []) if isinstance(p, str)]
    if not query_text:
        continue
    query_to_downloads.setdefault(query_text, []).extend(image_paths)

batch_size = 256
num_workers = 4

all_models = {
    'vit-b-32': 'hf_clip:openai/clip-vit-base-patch32',
    # 'wildclip-t1': 'wildclip_vitb16_t1',
    # 'wildclip-t1t7-lwf': 'wildclip_vitb16_t1t7_lwf',
   
    # 'rn50': 'open_clip:RN50/openai',
    # 'rn50x16': 'open_clip:RN50x16/openai',
    # 'vit-b-16': 'hf_clip:openai/clip-vit-base-patch16',
    # 'vit-l-14': 'hf_clip:openai/clip-vit-large-patch14',
    # 'vit-b-16-dfn': 'open_clip:ViT-B-16/dfn2b',
    # 'vit-l-14-dfn': 'open_clip:ViT-L-14-quickgelu/dfn2b',
    # 'vit-h-14-378': 'open_clip:ViT-H-14-378-quickgelu/dfn5b',
    # 'siglip-vit-l-16-384': 'open_clip:ViT-L-16-SigLIP-384/webli',
   
     'bioclip': 'bioclip',
    'biocap': 'biocap',
     'siglip-so400m-14-384': 'open_clip:ViT-SO400M-14-SigLIP-384/webli',
}

results = []
for title, clip_name in all_models.items():
    model, preprocess, _ = load_clip(clip_name, use_jit=False, device=device)

    # Efficiently compute image embeddings in batches
    def collate_transform(examples):
        pixel_values = torch.cat([preprocess(ex["image"]).unsqueeze(0) for ex in examples])
        ids = [ex['inat24_image_id'] for ex in examples]
        return pixel_values, ids
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_transform, num_workers=num_workers)

    image_emb_cache = {}
    for images, ids in tqdm(dataloader, total=len(dataset)//batch_size):
        with torch.no_grad(), torch.autocast(device):
            image_embs = model.encode_image(images.to(device)).cpu()
            image_embs /= image_embs.norm(dim=-1, keepdim=True)
        image_emb_cache.update(dict(zip(ids, image_embs)))


    def embed_downloaded_images(paths: list[str]) -> torch.Tensor:
        images = []
        for path in paths:
            try:
                with Image.open(path) as img:
                    images.append(preprocess(img.convert("RGB")).unsqueeze(0))
            except Exception:
                continue
        if not images:
            return torch.empty((0, image_emb_cache[next(iter(image_emb_cache))].shape[-1]))
        batch = torch.cat(images)
        with torch.no_grad(), torch.autocast(device):
            emb = model.encode_image(batch.to(device)).cpu()
            emb /= emb.norm(dim=-1, keepdim=True)
        return emb

    # Score images for each query by average image-to-image similarity
    metrics_avg = MetricAverage()
    for query in queries:
        query_ds = dataset.select(np.argwhere(np.asarray(dataset['query']) == query).squeeze())

        download_paths = [p for p in query_to_downloads.get(query, []) if Path(p).exists()]
        download_embs = embed_downloaded_images(download_paths)
        if download_embs.numel() == 0:
            continue

        image_embs = torch.stack([image_emb_cache[image_id] for image_id in query_ds['inat24_image_id']])
        sim_matrix = image_embs.float() @ download_embs.float().T
        y_pred = sim_matrix.mean(dim=1).numpy()
        y_true = np.asarray(query_ds['relevant'])

        pr, rec, ap, ndcg, mrr = compute_retrieval_metrics(y_true, y_pred, count_pos=sum(y_true))
        metrics_avg.update([ap*100, ndcg*100, mrr])
        results.append(dict(model=title, query=query, ap=ap*100, ndcg=ndcg*100, mrr=mrr))

    ap, ndcg, mrr = metrics_avg.avg
    print(f'{title:30s}\t{ap:.1f}\t{ndcg:.1f}\t{mrr:.2f}')

results_df = pd.DataFrame.from_dict(results)
pd.options.display.float_format = ' {:,.2f}'.format
print(results_df.groupby('model').agg({'ap': 'mean', 'ndcg': 'mean', 'mrr': 'mean'}).sort_values('ap'))

results_df.to_csv(save_results_path)
print("All done! Saved results to", save_results_path)
