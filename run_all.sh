#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
echo "=== THE VISUAL REFUSAL GAP — Full Pipeline ==="
bash run_phase1.sh
bash run_phase2.sh
bash run_phase3.sh
bash run_phase4.sh
bash run_phase5.sh
echo ""
echo "ALL PHASES COMPLETE"
ROOT_DIR="${VRG_ROOT:-$(pwd)}"
DATA_DIR="${VRG_DATA_DIR:-$ROOT_DIR/data}"
OUTPUT_DIR="${VRG_OUTPUT_DIR:-$ROOT_DIR/outputs}"
echo "  $DATA_DIR           — prompts + images"
echo "  $OUTPUT_DIR        — all results + plots"
