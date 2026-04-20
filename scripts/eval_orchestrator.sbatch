#!/bin/bash
#SBATCH --job-name=inquire_orchestrator
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
mkdir -p output/orchestrator/test

RESULTS_CSV=output/web_images/compare/results_orchestrator_4modes_test.csv
SUMMARY_CSV=output/web_images/compare/results_orchestrator_4modes_test_summary.csv
SUPERCAT_CSV=output/web_images/compare/results_orchestrator_4modes_test_by_supercategory.csv

python scripts/eval_orchestrator.py \
  --config configs/eval_orchestrator.yml \
  --split test \
  --retrieval-mode auto \
  --save-results-path "${RESULTS_CSV}" \
  --save-query-details-dir output/orchestrator/test

python scripts/summarize_single_results.py \
  --results "${RESULTS_CSV}" \
  --queries-csv /network/scratch/y/yuyan.chen/inquire/inquire/inquire_queries_test.csv \
  --summary-out "${SUMMARY_CSV}" \
  --supercategory-out "${SUPERCAT_CSV}"
