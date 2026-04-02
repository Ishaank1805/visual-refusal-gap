#!/bin/bash
#SBATCH --job-name=2_3
#SBATCH --output=2_3.out
#SBATCH --error=2_3.err
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=20G
#SBATCH --time=96:00:00
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -w gnode080


# --- Environment setup ---
source ~/.bashrc
conda activate pyg
ROOT_DIR="${VRG_ROOT:-$(pwd)}"
OUTPUT_DIR="${VRG_OUTPUT_DIR:-$ROOT_DIR/outputs}"

python -u phase2/phase2_03_behavioral_validation.py \
    --model_id llava-hf/llava-1.5-7b-hf \
    --hazards_dir ./data/visual_hazards_v2 \
    --output_dir "$OUTPUT_DIR/gap_analysis" \
    --use_4bit
