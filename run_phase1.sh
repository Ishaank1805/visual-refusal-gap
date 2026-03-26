#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_CACHE:-/scratch/ishaan.karan/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
MODEL_ID="${MODEL_ID:-llava-hf/llava-1.5-7b-hf}"

echo "=== PHASE 1: The Safety Switch ==="

echo "[1/4] Preparing data..."
python -u phase1/phase1_01_prepare_data.py --output_dir /scratch/ishaan.karan/data/prompts

echo "[2/4] Extracting refusal direction..."
python -u phase1/phase1_02_extract_refusal_vector.py \
    --model_id "$MODEL_ID" --data_dir /scratch/ishaan.karan/data/prompts \
    --output_dir /scratch/ishaan.karan/outputs/vectors --use_4bit

echo "[3/4] Validating..."
if [ -n "$GEMINI_KEY_1" ] || [ -n "$GEMINI_API_KEY" ]; then
    python -u phase1/phase1_03_validate_refusal_vector.py \
        --model_id "$MODEL_ID" --data_dir /scratch/ishaan.karan/data/prompts \
        --vector_dir /scratch/ishaan.karan/outputs/vectors \
        --output_dir /scratch/ishaan.karan/outputs/logs \
        --use_4bit --alpha_values 0.1 0.3 0.5 1.0 2.0
else
    python -u phase1/phase1_03_validate_refusal_vector.py \
        --model_id "$MODEL_ID" --data_dir /scratch/ishaan.karan/data/prompts \
        --vector_dir /scratch/ishaan.karan/outputs/vectors \
        --output_dir /scratch/ishaan.karan/outputs/logs \
        --use_4bit --alpha_values 0.1 0.3 0.5 1.0 2.0 --skip_judge
fi

echo "[4/4] Plots..."
python -u phase1/phase1_04_visualize.py \
    --model_id "$MODEL_ID" --vector_dir /scratch/ishaan.karan/outputs/vectors \
    --log_dir /scratch/ishaan.karan/outputs/logs --plot_dir /scratch/ishaan.karan/outputs/plots

echo "Phase 1 COMPLETE"
