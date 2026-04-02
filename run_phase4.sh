#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
ROOT_DIR="${VRG_ROOT:-$(pwd)}"
DATA_DIR="${VRG_DATA_DIR:-$ROOT_DIR/data}"
OUTPUT_DIR="${VRG_OUTPUT_DIR:-$ROOT_DIR/outputs}"
export HF_HOME="${HF_CACHE:-$ROOT_DIR/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
MODEL_ID="${MODEL_ID:-llava-hf/llava-1.5-7b-hf}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen2-VL-2B-Instruct}"

echo "=== PHASE 4: Generalization ==="

echo "[1/3] Direction selectivity..."
python -u phase4/phase4_01_extract_directions.py \
    --model_id "$MODEL_ID" \
    --direction_file "$DATA_DIR/visual_hazards_v2/direction_prompts.json" \
    --vector_dir "$OUTPUT_DIR/vectors" \
    --output_dir "$OUTPUT_DIR/generalization" --use_4bit

echo "[2/3] Cross-architecture (Qwen2-VL)..."
python -u phase4/phase4_02_cross_architecture.py \
    --model_id "$QWEN_ID" --data_dir "$DATA_DIR/prompts" \
    --hazards_dir "$DATA_DIR/visual_hazards_v2" \
    --output_dir "$OUTPUT_DIR/generalization" --use_4bit

echo "[3/3] Plots..."
python -u phase4/phase4_03_visualize.py \
    --gen_dir "$OUTPUT_DIR/generalization" \
    --vector_dir "$OUTPUT_DIR/vectors" \
    --plot_dir "$OUTPUT_DIR/plots"

echo "Phase 4 COMPLETE"
