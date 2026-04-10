#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G


source .venv/bin/activate

python scripts/use_caption.py --save_results_path results_rerank_with_llm_test_qwen4b_thinking_cap.csv