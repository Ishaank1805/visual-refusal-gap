#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
ROOT_DIR="${VRG_ROOT:-$(pwd)}"
DATA_DIR="${VRG_DATA_DIR:-$ROOT_DIR/data}"
OUTPUT_DIR="${VRG_OUTPUT_DIR:-$ROOT_DIR/outputs}"
export HF_HOME="${HF_CACHE:-$ROOT_DIR/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
MODEL_ID="${MODEL_ID:-llava-hf/llava-1.5-7b-hf}"
HAZARDS="${HAZARDS_DIR:-$DATA_DIR/visual_hazards_v2}"
OUTDIR="${GAP_DIR:-$OUTPUT_DIR/gap_analysis}"
MAX_PAIRS="${MAX_PAIRS:-0}"

echo "=== PHASE 2: The Gap ==="

if [ ! -f "$HAZARDS/visual_hazards_metadata.json" ]; then
    echo "[1/5] Generating dataset..."
    python -u phase2/phase2_01_generate_dataset.py --output_dir "$HAZARDS"
else
    echo "[1/5] Dataset exists, skipping"
fi

echo "[2/5] Measuring visual gap..."
python -u phase2/phase2_02_measure_visual_gap.py \
    --model_id "$MODEL_ID" --vector_dir "$OUTPUT_DIR/vectors" \
    --hazards_dir "$HAZARDS" --output_dir "$OUTDIR" --use_4bit --max_pairs "$MAX_PAIRS"

echo "[3/5] Behavioral validation..."
python -u phase2/phase2_03_behavioral_validation.py \
    --model_id "$MODEL_ID" --hazards_dir "$HAZARDS" \
    --output_dir "$OUTDIR" --use_4bit --max_pairs "$MAX_PAIRS"

if [ -n "$GEMINI_KEY_1" ] || [ -n "$GEMINI_API_KEY" ]; then
    echo "[4/5] Gemini judge..."
    python -u phase2/phase2_04_gemini_judge.py --output_dir "$OUTDIR"
else
    echo "[4/5] Skipping judge (no keys)"
fi

echo "[5/5] Plots..."
python -u phase2/phase2_05_visualize.py \
    --model_id "$MODEL_ID" --output_dir "$OUTDIR" \
    --plot_dir "$OUTPUT_DIR/plots"

echo "Phase 2 COMPLETE"
