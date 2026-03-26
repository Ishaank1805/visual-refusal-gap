#!/bin/bash
# ============================================================
# Phase 3: The Mechanism (8 experiments)
#
# Step 1: Alignment Geometry (CPU — PCA, SVD, surgical dissection, categories)
# Step 2: Linear Probe (CPU — train text, test image)
# Step 3: Interpolation + Boundary (GPU — text↔image blend + binary search)
# Step 4: Projector Ablation Grid (GPU — zero out SVs, circuit map)
# Step 5: Projector Surgery (GPU — gradient-edit to restore refusal)
# Step 6: Representation Swap (GPU — swap refusal component only)
# Step 7: Layer Freezing Sweep (GPU — steer each layer, propagation)
# Step 8: Decoding Dynamics (GPU — refusal token probability per step)
# ============================================================

set -e
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_CACHE:-/scratch/ishaan.karan/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
MODEL_ID="${MODEL_ID:-llava-hf/llava-1.5-7b-hf}"
HAZARDS="${HAZARDS_DIR:-/scratch/ishaan.karan/data/visual_hazards_v2}"

echo "=============================================="
echo "  PHASE 3: The Mechanism (8 experiments)"
echo "=============================================="

echo "[1/8] Alignment geometry..."
python -u phase3/phase3_01_alignment_geometry.py \
    --model_id "$MODEL_ID" \
    --vector_dir /scratch/ishaan.karan/outputs/vectors \
    --gap_dir /scratch/ishaan.karan/outputs/gap_analysis \
    --output_dir /scratch/ishaan.karan/outputs/mechanism \
    --plot_dir /scratch/ishaan.karan/outputs/plots

echo "[2/8] Linear probe..."
python -u phase3/phase3_02_linear_probe.py \
    --model_id "$MODEL_ID" \
    --gap_dir /scratch/ishaan.karan/outputs/gap_analysis \
    --vector_dir /scratch/ishaan.karan/outputs/vectors \
    --output_dir /scratch/ishaan.karan/outputs/mechanism \
    --plot_dir /scratch/ishaan.karan/outputs/plots

echo "[3/8] Interpolation + boundary..."
python -u phase3/phase3_03_interpolation.py \
    --model_id "$MODEL_ID" \
    --vector_dir /scratch/ishaan.karan/outputs/vectors \
    --hazards_dir "$HAZARDS" \
    --output_dir /scratch/ishaan.karan/outputs/mechanism \
    --plot_dir /scratch/ishaan.karan/outputs/plots \
    --use_4bit --num_pairs 50

echo "[4/8] Projector ablation grid..."
python -u phase3/phase3_04_projector_ablation.py \
    --model_id "$MODEL_ID" \
    --vector_dir /scratch/ishaan.karan/outputs/vectors \
    --hazards_dir "$HAZARDS" \
    --output_dir /scratch/ishaan.karan/outputs/mechanism \
    --plot_dir /scratch/ishaan.karan/outputs/plots \
    --use_4bit --num_pairs 50

echo "[5/8] Projector surgery..."
python -u phase3/phase3_05_projector_surgery.py \
    --model_id "$MODEL_ID" \
    --vector_dir /scratch/ishaan.karan/outputs/vectors \
    --hazards_dir "$HAZARDS" \
    --output_dir /scratch/ishaan.karan/outputs/mechanism \
    --plot_dir /scratch/ishaan.karan/outputs/plots \
    --use_4bit --num_pairs 50 --steps 100

echo "[6/8] Representation swap..."
python -u phase3/phase3_06_representation_swap.py \
    --model_id "$MODEL_ID" \
    --vector_dir /scratch/ishaan.karan/outputs/vectors \
    --hazards_dir "$HAZARDS" \
    --output_dir /scratch/ishaan.karan/outputs/mechanism \
    --plot_dir /scratch/ishaan.karan/outputs/plots \
    --use_4bit --num_pairs 30

echo "[7/8] Layer freezing sweep..."
python -u phase3/phase3_07_layer_freezing.py \
    --model_id "$MODEL_ID" \
    --vector_dir /scratch/ishaan.karan/outputs/vectors \
    --hazards_dir "$HAZARDS" \
    --output_dir /scratch/ishaan.karan/outputs/mechanism \
    --plot_dir /scratch/ishaan.karan/outputs/plots \
    --use_4bit --num_pairs 30

echo "[8/8] Decoding dynamics..."
python -u phase3/phase3_08_decoding_dynamics.py \
    --model_id "$MODEL_ID" \
    --hazards_dir "$HAZARDS" \
    --output_dir /scratch/ishaan.karan/outputs/mechanism \
    --plot_dir /scratch/ishaan.karan/outputs/plots \
    --use_4bit --num_pairs 30

echo ""
echo "=============================================="
echo "  Phase 3 COMPLETE (8 experiments)"
echo "=============================================="
echo ""
echo "Key outputs:"
echo "  geometry_*.json         — PCA rank, SVD alignment, surgical dissection"
echo "  linear_probe_*.json     — signal destroyed vs hidden"
echo "  interpolation_*.json    — smooth curve + boundary distribution"
echo "  ablation_grid_*.json    — circuit map: which SVs carry safety"
echo "  projector_surgery_*.json — minimal edit restores safety"
echo "  representation_swap_*.json — swap refusal component → behavior change"
echo "  layer_freezing_*.json   — minimum controllable layer + propagation"
echo "  decoding_dynamics_*.json — refusal token probability text vs image"
