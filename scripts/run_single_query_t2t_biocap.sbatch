#!/bin/bash
#SBATCH --job-name=t2t_biocap_q
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /home/mila/e/echchabo/projects/llm-course-project
source .venv/bin/activate

python -u scripts/query_top3_biocap.py \
  --query "A Eurasian Red Squirrel gathering material for its nest" \
  --cache output/web_images/compare/caption_cache_test.json \
  --model biocap \
  --split test \
  --topk 3
