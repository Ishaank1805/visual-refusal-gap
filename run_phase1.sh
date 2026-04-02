#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
ROOT_DIR="${VRG_ROOT:-$(pwd)}"
DATA_DIR="${VRG_DATA_DIR:-$ROOT_DIR/data}"
OUTPUT_DIR="${VRG_OUTPUT_DIR:-$ROOT_DIR/outputs}"
export HF_HOME="${HF_CACHE:-$ROOT_DIR/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
MODEL_ID="${MODEL_ID:-llava-hf/llava-1.5-7b-hf}"

echo "=== PHASE 1: The Safety Switch ==="

echo "[1/4] Preparing data..."
python -u phase1/phase1_01_prepare_data.py --output_dir "$DATA_DIR/prompts"

echo "[2/4] Extracting refusal direction..."
python -u phase1/phase1_02_extract_refusal_vector.py \
    --model_id "$MODEL_ID" --data_dir "$DATA_DIR/prompts" \
    --output_dir "$OUTPUT_DIR/vectors" --use_4bit

echo "[3/4] Validating..."
if [ -n "$GEMINI_KEY_1" ] || [ -n "$GEMINI_API_KEY" ]; then
    python -u phase1/phase1_03_validate_refusal_vector.py \
        --model_id "$MODEL_ID" --data_dir "$DATA_DIR/prompts" \
        --vector_dir "$OUTPUT_DIR/vectors" \
        --output_dir "$OUTPUT_DIR/logs" \
        --use_4bit --alpha_values 0.1 0.3 0.5 1.0 2.0
else
    python -u phase1/phase1_03_validate_refusal_vector.py \
        --model_id "$MODEL_ID" --data_dir "$DATA_DIR/prompts" \
        --vector_dir "$OUTPUT_DIR/vectors" \
        --output_dir "$OUTPUT_DIR/logs" \
        --use_4bit --alpha_values 0.1 0.3 0.5 1.0 2.0 --skip_judge
fi

echo "[4/4] Plots..."
python -u phase1/phase1_04_visualize.py \
    --model_id "$MODEL_ID" --vector_dir "$OUTPUT_DIR/vectors" \
    --log_dir "$OUTPUT_DIR/logs" --plot_dir "$OUTPUT_DIR/plots"

echo "Phase 1 COMPLETE"
