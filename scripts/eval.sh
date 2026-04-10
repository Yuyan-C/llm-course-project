#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G


# module load anaconda/3
# conda create -p "$SCRATCH/transformers4571" python=3.11
# conda activate "$SCRATCH/transformers4571"
# pip install transformers
# pip install torch pillow einops torchvision accelerate decord2 molmo_utils
# pip install all_clip datasets numpy pandas pyarrow scikit-learn tqdm

# # source .venv/bin/activate
# export PROJECT_ROOT=/home/mila/y/yuyan.chen/projects/llm-course-project
# export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
source .venv/bin/activate
# python scripts/eval_rerank_with_llm.py 

python scripts/use_image_search.py --output /network/scratch/y/yuyan.chen/inquire/web_images