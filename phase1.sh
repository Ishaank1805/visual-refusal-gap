#!/bin/bash
#SBATCH --job-name=phase1
#SBATCH --output=phase1.out
#SBATCH --error=phase1.err
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

bash run_phase1.sh
