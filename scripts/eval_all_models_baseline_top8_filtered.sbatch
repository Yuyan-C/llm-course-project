#!/bin/bash
#SBATCH --job-name=inquire_all_cmp
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --exclude=cn-c[026,038]
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /home/mila/e/echchabo/projects/llm-course-project
source .venv/bin/activate

mkdir -p output/web_images/compare

MODELS=(vit-b-32 bioclip biocap siglip-vit-b-16 siglip-so400m-14-384 dinov3-b16)
RAW_WEB_DIR=/network/scratch/y/yuyan.chen/inquire/web_images_raw_top20
RAW_METADATA=${RAW_WEB_DIR}/image_search_metadata_test.json
TOP8_FILTERED_METADATA=/network/scratch/y/yuyan.chen/inquire/web_images_top8_min2_metadata_only/image_search_metadata_test.json

mkdir -p "${RAW_WEB_DIR}"
mkdir -p "$(dirname "${TOP8_FILTERED_METADATA}")"

python scripts/use_image_search.py \
  --split test \
  --output "${RAW_WEB_DIR}" \
  --max-results 20 \
  --keep-k 20 \
  --min-keep 0 \
  --disable-quality-filter

# Prime cache for network-sensitive models before full evaluation.
.venv/bin/python - <<'PY'
import time
from src.inquire.utils import load_clip
from transformers import AutoImageProcessor, AutoModel
from transformers import T5Tokenizer, T5TokenizerFast

for cls in (T5Tokenizer, T5TokenizerFast):
    if cls is not None and not hasattr(cls, "batch_encode_plus"):
        def _batch_encode_plus(self, *args, **kwargs):
            return self(*args, **kwargs)
        setattr(cls, "batch_encode_plus", _batch_encode_plus)

def retry(fn, name, retries=6, sleep_s=10):
    for i in range(1, retries + 1):
        try:
            fn()
            print(f"[prefetch] {name} ok")
            return
        except Exception as exc:
            if i == retries:
                raise
            print(f"[prefetch] {name} failed attempt {i}/{retries}: {exc}. retry in {sleep_s}s")
            time.sleep(sleep_s)

retry(lambda: load_clip("open_clip:ViT-SO400M-14-SigLIP-384/webli", device="cuda", use_jit=False), "siglip")
retry(lambda: load_clip("open_clip:ViT-B-16-SigLIP-256/webli", device="cuda", use_jit=False), "siglip-b16")
retry(lambda: AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m"), "dinov3-processor")
retry(lambda: AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m"), "dinov3")
PY

python scripts/eval_rerank_image_similarity.py \
  --split test \
  --metadata "${RAW_METADATA}" \
  --models "${MODELS[@]}" \
  --max-web-images-per-query 3 \
  --min-web-images-per-query 0 \
  --aggregation mean \
  --batch-size 64 \
  --num-workers 0 \
  --model-load-retries 6 \
  --model-load-retry-sleep 10 \
  --save-results-path output/web_images/compare/results_baseline_top3_all_models_test.csv

python scripts/eval_rerank_image_similarity.py \
  --split test \
  --metadata "${RAW_METADATA}" \
  --models "${MODELS[@]}" \
  --max-web-images-per-query 8 \
  --min-web-images-per-query 2 \
  --apply-web-image-filters \
  --metadata-relevance-threshold 0.25 \
  --filtered-metadata-out "${TOP8_FILTERED_METADATA}" \
  --aggregation topk_mean \
  --top-k-sims 3 \
  --batch-size 64 \
  --num-workers 0 \
  --model-load-retries 6 \
  --model-load-retry-sleep 10 \
  --save-results-path output/web_images/compare/results_top8_filtered_topk_all_models_test.csv

python scripts/summarize_baseline_vs_filtered.py \
  --baseline-results output/web_images/compare/results_baseline_top3_all_models_test.csv \
  --comparison-results output/web_images/compare/results_top8_filtered_topk_all_models_test.csv \
  --queries-csv /network/scratch/y/yuyan.chen/inquire/inquire/inquire_queries_test.csv \
  --summary-out output/web_images/compare/results_all_models_baseline_vs_filtered_summary.csv \
  --supercategory-out output/web_images/compare/results_all_models_baseline_vs_filtered_by_supercategory.csv
