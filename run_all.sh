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
echo "  /scratch/ishaan.karan/data/           — prompts + images"
echo "  /scratch/ishaan.karan/outputs/        — all results + plots"
