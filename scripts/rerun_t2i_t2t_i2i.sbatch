#!/bin/bash
#SBATCH --job-name=inquire_methods_rerun
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

# Use cached HF assets only to avoid transient hub/network issues during model loads.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

RESULTS_DIR=output/retrieval/method_benchmarks
SCRATCH_WEB_DIR=/network/scratch/y/yuyan.chen/inquire/web_images
METADATA=${SCRATCH_WEB_DIR}/image_search_metadata_test.json
MODELS=(vit-b-32 bioclip biocap siglip-vit-b-16)
MODEL_LOAD_RETRIES=6
MODEL_LOAD_RETRY_SLEEP=10
SIGLIP_PREFLIGHT_ATTEMPTS=8
SIGLIP_PREFLIGHT_SLEEP=15

mkdir -p "${RESULTS_DIR}"

if [[ ! -s "${METADATA}" ]]; then
  echo "Missing scratch metadata: ${METADATA}"
  exit 1
fi
echo "[1/4] Reusing scratch web metadata at ${METADATA}"

echo "[preflight] Ensuring siglip-vit-b-16 loads successfully..."
siglip_ok=0
for attempt in $(seq 1 "${SIGLIP_PREFLIGHT_ATTEMPTS}"); do
  echo "[preflight] siglip load attempt ${attempt}/${SIGLIP_PREFLIGHT_ATTEMPTS}"
  if .venv/bin/python - <<'PY'
import torch
from src.inquire.utils import load_clip

try:
    from transformers import T5Tokenizer, T5TokenizerFast  # type: ignore
    for cls in (T5Tokenizer, T5TokenizerFast):
        if cls is None:
            continue
        if hasattr(cls, "batch_encode_plus"):
            continue

        def _batch_encode_plus(self, *args, **kwargs):
            return self(*args, **kwargs)

        setattr(cls, "batch_encode_plus", _batch_encode_plus)
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, tokenizer = load_clip("open_clip:ViT-B-16-SigLIP-256/webli", use_jit=False, device=device)
print("siglip-vit-b-16 preflight load: OK")
PY
  then
    siglip_ok=1
    break
  fi
  sleep "${SIGLIP_PREFLIGHT_SLEEP}"
done

if [[ "${siglip_ok}" -ne 1 ]]; then
  echo "siglip-vit-b-16 failed preflight loading after retries; exiting."
  exit 1
fi

echo "[2/4] Running T2I (repo baseline script)..."
python scripts/eval_rerank_with_clip.py \
  --split test \
  --models "${MODELS[@]}" \
  --model-load-retries "${MODEL_LOAD_RETRIES}" \
  --model-load-retry-sleep "${MODEL_LOAD_RETRY_SLEEP}" \
  --save-results-path "${RESULTS_DIR}/results_t2i_all4_test.csv"

echo "[3/4] Running I2I (web-image exemplars)..."
python scripts/eval_rerank_image_similarity.py \
  --split test \
  --metadata "${METADATA}" \
  --models "${MODELS[@]}" \
  --model-load-retries "${MODEL_LOAD_RETRIES}" \
  --model-load-retry-sleep "${MODEL_LOAD_RETRY_SLEEP}" \
  --save-results-path "${RESULTS_DIR}/results_i2i_all4_test.csv"

echo "[4/4] Running T2T (query text vs generated caption text)..."
python scripts/eval_rerank_text_similarity.py \
  --split test \
  --models "${MODELS[@]}" \
  --model-load-retries "${MODEL_LOAD_RETRIES}" \
  --model-load-retry-sleep "${MODEL_LOAD_RETRY_SLEEP}" \
  --caption-vlm-model Qwen/Qwen3-VL-4B-Instruct \
  --caption-cache-path "${RESULTS_DIR}/caption_cache_test.json" \
  --save-results-path "${RESULTS_DIR}/results_t2t_all4_test.csv"

echo "Building method summary..."
.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

compare_dir = Path("output/retrieval/method_benchmarks")
paths = {
    "t2i": compare_dir / "results_t2i_all4_test.csv",
    "i2i": compare_dir / "results_i2i_all4_test.csv",
    "t2t": compare_dir / "results_t2t_all4_test.csv",
}

combined_rows = []
for method, path in paths.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing expected results file: {path}")
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    df["method"] = method
    df["source_path"] = str(path)
    combined_rows.append(df)

combined = pd.concat(combined_rows, ignore_index=True)
combined_path = compare_dir / "results_methods_all4_combined_test.csv"
combined.to_csv(combined_path, index=False)

summary_method_model = (
    combined.groupby(["method", "model"], as_index=False)
    .agg(ap=("ap", "mean"), ndcg=("ndcg", "mean"), mrr=("mrr", "mean"), n_queries=("query", "count"))
    .sort_values(["method", "ap"], ascending=[True, False])
)
summary_method_model_path = compare_dir / "results_methods_all4_summary_by_model.csv"
summary_method_model.to_csv(summary_method_model_path, index=False)

summary_method = (
    combined.groupby("method", as_index=False)
    .agg(ap=("ap", "mean"), ndcg=("ndcg", "mean"), mrr=("mrr", "mean"), n_queries=("query", "count"))
    .sort_values("ap", ascending=False)
)
summary_method_path = compare_dir / "results_methods_all4_summary.csv"
summary_method.to_csv(summary_method_path, index=False)

print("Summary by method and model:")
print(summary_method_model.to_string(index=False))
print(f"\nSaved combined rows to {combined_path}")
print(f"Saved per-method/model summary to {summary_method_model_path}")
print(f"Saved per-method summary to {summary_method_path}")
PY

echo "Done."
