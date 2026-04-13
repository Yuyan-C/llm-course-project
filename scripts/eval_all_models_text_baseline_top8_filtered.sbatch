#!/bin/bash
#SBATCH --job-name=inquire_text_cmp
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --exclude=cn-c[026,038]
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /home/mila/e/echchabo/projects/llm-course-project
source .venv/bin/activate

mkdir -p output/web_images/compare

MODELS=(vit-b-32 bioclip biocap siglip-so400m-14-384 dinov3-b16)
RAW_WEB_DIR=/network/scratch/y/yuyan.chen/inquire/web_images_raw_top20
RAW_METADATA=${RAW_WEB_DIR}/image_search_metadata_test.json
TOP8_FILTERED_METADATA=/network/scratch/y/yuyan.chen/inquire/web_images_top8_min2_metadata_only_text/image_search_metadata_test.json
CAPTION_CACHE=output/web_images/compare/caption_cache_test.json
CAPTION_VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct

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
from transformers import T5Tokenizer, T5TokenizerFast
from src.inquire.utils import load_clip
from src.inquire.lmm_utils_new import ModelWrapper

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

retry(lambda: ModelWrapper("Qwen/Qwen3-VL-4B-Instruct", "cuda"), "caption-vlm")
retry(lambda: load_clip("hf_clip:openai/clip-vit-base-patch32", device="cuda", use_jit=False), "vit-b-32")
retry(lambda: load_clip("bioclip", device="cuda", use_jit=False), "bioclip")
retry(lambda: load_clip("biocap", device="cuda", use_jit=False), "biocap")
retry(lambda: load_clip("open_clip:ViT-SO400M-14-SigLIP-384/webli", device="cuda", use_jit=False), "siglip")
PY

python scripts/eval_rerank_text_similarity.py \
  --split test \
  --metadata "${RAW_METADATA}" \
  --models "${MODELS[@]}" \
  --caption-vlm-model "${CAPTION_VLM_MODEL}" \
  --caption-cache-path "${CAPTION_CACHE}" \
  --max-web-images-per-query 3 \
  --min-web-images-per-query 0 \
  --aggregation mean \
  --batch-size 64 \
  --model-load-retries 6 \
  --model-load-retry-sleep 10 \
  --save-results-path output/web_images/compare/results_text_baseline_top3_all_models_test.csv

python scripts/eval_rerank_text_similarity.py \
  --split test \
  --metadata "${RAW_METADATA}" \
  --models "${MODELS[@]}" \
  --caption-vlm-model "${CAPTION_VLM_MODEL}" \
  --caption-cache-path "${CAPTION_CACHE}" \
  --max-web-images-per-query 8 \
  --min-web-images-per-query 2 \
  --apply-web-image-filters \
  --metadata-relevance-threshold 0.25 \
  --filtered-metadata-out "${TOP8_FILTERED_METADATA}" \
  --aggregation topk_mean \
  --top-k-sims 3 \
  --batch-size 64 \
  --model-load-retries 6 \
  --model-load-retry-sleep 10 \
  --save-results-path output/web_images/compare/results_text_top8_filtered_topk_all_models_test.csv

python scripts/summarize_baseline_vs_top8.py \
  --baseline-results output/web_images/compare/results_text_baseline_top3_all_models_test.csv \
  --top8-results output/web_images/compare/results_text_top8_filtered_topk_all_models_test.csv \
  --queries-csv /network/scratch/y/yuyan.chen/inquire/inquire/inquire_queries_test.csv \
  --summary-out output/web_images/compare/results_text_all_models_baseline_vs_top8_summary.csv \
  --supercategory-out output/web_images/compare/results_text_all_models_baseline_vs_top8_by_supercategory.csv
