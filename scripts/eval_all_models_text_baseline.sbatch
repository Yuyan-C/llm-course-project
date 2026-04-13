#!/bin/bash
#SBATCH --job-name=inquire_text_prompt
#SBATCH --partition=main
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /home/mila/e/echchabo/projects/llm-course-project
source .venv/bin/activate

mkdir -p output/web_images/compare

MODELS=(vit-b-32 bioclip biocap siglip-vit-b-16 siglip-so400m-14-384 dinov3-b16)
CAPTION_CACHE=output/web_images/compare/caption_cache_test.json
CAPTION_VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct
TEXT_RESULTS=output/web_images/compare/results_text_prompt_similarity_all_models_test.csv
TEXT_SUMMARY=output/web_images/compare/results_text_prompt_similarity_all_models_summary.csv
TEXT_SUPERCAT=output/web_images/compare/results_text_prompt_similarity_all_models_by_supercategory.csv

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
retry(lambda: load_clip("open_clip:ViT-B-16-SigLIP-256/webli", device="cuda", use_jit=False), "siglip-b16")
retry(lambda: load_clip("open_clip:ViT-SO400M-14-SigLIP-384/webli", device="cuda", use_jit=False), "siglip")
PY

python scripts/eval_rerank_text_similarity.py \
  --split test \
  --models "${MODELS[@]}" \
  --caption-vlm-model "${CAPTION_VLM_MODEL}" \
  --caption-cache-path "${CAPTION_CACHE}" \
  --batch-size 64 \
  --model-load-retries 6 \
  --model-load-retry-sleep 10 \
  --save-results-path "${TEXT_RESULTS}"

python scripts/summarize_single_results.py \
  --results "${TEXT_RESULTS}" \
  --queries-csv /network/scratch/y/yuyan.chen/inquire/inquire/inquire_queries_test.csv \
  --summary-out "${TEXT_SUMMARY}" \
  --supercategory-out "${TEXT_SUPERCAT}"
