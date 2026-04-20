#!/bin/bash
#SBATCH --job-name=inquire_orch
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

MODEL_KEY="${1:-vit-b-32}"  # vit-b-32 | bioclip | biocap | siglip-vit-b-16
SPLIT="${2:-test}"
MODEL_LOAD_RETRIES=6
MODEL_LOAD_RETRY_SLEEP=10
ORCH_LOAD_RETRIES=4
ORCH_LOAD_RETRY_SLEEP=10

SCRATCH_WEB_DIR="/network/scratch/y/yuyan.chen/inquire/web_images"
ORCH_DIR="output/retrieval/method_router"
METADATA="${SCRATCH_WEB_DIR}/image_search_metadata_${SPLIT}.json"

mkdir -p "${ORCH_DIR}"

# Force cache-only model/dataset loading to avoid transient HF network failures.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [[ ! -s "${METADATA}" ]]; then
  echo "Missing scratch metadata: ${METADATA}"
  exit 1
fi

echo "Running orchestrator for model=${MODEL_KEY}, split=${SPLIT}"
python scripts/eval_orchestrator_method_router.py \
  --config configs/eval.yml \
  --split "${SPLIT}" \
  --model-key "${MODEL_KEY}" \
  --force-reroute \
  --metadata "${METADATA}" \
  --max-web-images-per-query 3 \
  --min-web-images-per-query 1 \
  --disable-web-image-filters \
  --aggregation mean \
  --model-load-retries "${MODEL_LOAD_RETRIES}" \
  --model-load-retry-sleep "${MODEL_LOAD_RETRY_SLEEP}" \
  --orchestrator-load-retries "${ORCH_LOAD_RETRIES}" \
  --orchestrator-load-retry-sleep "${ORCH_LOAD_RETRY_SLEEP}" \
  --caption-vlm-model Qwen/Qwen3-VL-4B-Instruct \
  --caption-cache-path "${ORCH_DIR}/caption_cache_${SPLIT}.json" \
  --decision-cache-path "${ORCH_DIR}/decision_cache_${MODEL_KEY}_${SPLIT}.json" \
  --save-results-path "${ORCH_DIR}/results_orchestrator_${MODEL_KEY}_${SPLIT}.csv" \
  --save-summary-path "${ORCH_DIR}/results_orchestrator_${MODEL_KEY}_${SPLIT}_summary.csv" \
  --save-usage-path "${ORCH_DIR}/results_orchestrator_${MODEL_KEY}_${SPLIT}_usage.csv"

echo "Done."
