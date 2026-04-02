#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_CACHE:-~/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
MODEL_ID="${MODEL_ID:-llava-hf/llava-1.5-7b-hf}"
ALPHA="${ALPHA:-0.5}"
MAX_PAIRS="${MAX_PAIRS:-100}"

echo "=== PHASE 5: The Fix ==="

echo "[1/4] Steering defense..."
python -u phase5/phase5_01_steering_defense.py \
    --model_id "$MODEL_ID" --vector_dir ./outputs/vectors \
    --hazards_dir ./data/visual_hazards_v2 \
    --output_dir ./outputs/defense \
    --use_4bit --alpha "$ALPHA" --max_pairs "$MAX_PAIRS"

echo "[2/4] Gated steering..."
python -u phase5/phase5_02_gated_steering.py \
    --model_id "$MODEL_ID" --vector_dir ./outputs/vectors \
    --hazards_dir ./data/visual_hazards_v2 \
    --output_dir ./outputs/defense \
    --use_4bit --alpha "$ALPHA" --max_pairs "$MAX_PAIRS"

echo "[3/4] Safety adapter (CPU)..."
python -u phase5/phase5_03_safety_adapter.py \
    --model_id "$MODEL_ID" --vector_dir ./outputs/vectors \
    --gap_dir ./outputs/gap_analysis \
    --output_dir ./outputs/defense

echo "[4/4] Plots..."
python -u phase5/phase5_04_visualize.py \
    --defense_dir ./outputs/defense \
    --plot_dir ./outputs/plots

echo "Phase 5 COMPLETE"
