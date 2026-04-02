#!/usr/bin/env python3
"""
Phase 4 - Step 3: Visualization

Plots:
  1. Direction selectivity (refusal/privacy lost, honesty/sycophancy preserved)
  2. Cross-architecture comparison (LLaVA vs Qwen2-VL)

Usage:
    python phase4_03_visualize.py
"""

import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_direction_selectivity(gen_dir, plot_dir):
    """Bar chart: visual gap per direction."""
    candidates = list(Path(gen_dir).glob("direction_selectivity_*.json"))
    if not candidates:
        print("  No direction selectivity data"); return

    with open(candidates[0]) as f:
        data = json.load(f)

    names = list(data.keys())
    text_gaps = [data[n].get("best_gap", 0) for n in names]
    visual_gaps = [data[n].get("visual_gap", 0) for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    w = 0.35

    colors_text = ["#e74c3c" if "refusal" in n else "#3498db" for n in names]
    colors_vis = ["#c0392b" if "refusal" in n else "#2980b9" for n in names]

    bars1 = ax.bar(x - w/2, text_gaps, w, color=colors_text, alpha=0.85, label="Text separation gap")
    bars2 = ax.bar(x + w/2, visual_gaps, w, color=colors_vis, alpha=0.6, label="Visual gap (text − image)")

    for bar, v in zip(bars1, text_gaps):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=9)
    for bar, v in zip(bars2, visual_gaps):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("Gap Score", fontsize=12)
    ax.set_title("Direction Selectivity: Which Alignment Signals Does the Projector Lose?",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", lw=0.5)

    plt.tight_layout()
    path = Path(plot_dir) / "direction_selectivity.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cross_architecture(gen_dir, vector_dir, plot_dir):
    """Compare layer-wise gaps across architectures."""
    # Find all cross-arch results
    arch_data = {}

    # LLaVA (from Phase 1)
    for p in Path(vector_dir).glob("metadata_*.json"):
        with open(p) as f:
            m = json.load(f)
        name = m["model_id"].split("/")[-1]
        arch_data[name] = {
            "scores": m["scores"],
            "best_layer": m["best_layer"],
            "best_gap": m["best_gap"],
        }

    # Other models
    for p in Path(gen_dir).glob("cross_arch_*.json"):
        with open(p) as f:
            m = json.load(f)
        name = m["model_id"].split("/")[-1]
        arch_data[name] = {
            "scores": m["scores"],
            "best_layer": m["best_layer"],
            "best_gap": m["best_gap"],
        }

    if len(arch_data) < 2:
        print("  Need 2+ architectures for comparison plot"); return

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.tab10

    for i, (name, data) in enumerate(sorted(arch_data.items())):
        scores = data["scores"]
        layers = sorted(int(l) for l in scores.keys())
        gaps = [scores[str(l)]["gap"] for l in layers]
        ax.plot(layers, gaps, "-o", ms=4, lw=2, color=cmap(i),
                label=f"{name} (best={data['best_gap']:.3f} @ L{data['best_layer']})")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Refusal Direction Gap", fontsize=12)
    ax.set_title("Cross-Architecture: Refusal Direction Exists in All Models",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", ls="--", alpha=0.4)

    plt.tight_layout()
    path = Path(plot_dir) / "cross_architecture.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default="outputs/generalization")
    ap.add_argument("--vector_dir", default="outputs/vectors")
    ap.add_argument("--plot_dir", default="outputs/plots")
    args = ap.parse_args()

    Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
    print("Phase 4 visualization...")
    plot_direction_selectivity(args.gen_dir, args.plot_dir)
    plot_cross_architecture(args.gen_dir, args.vector_dir, args.plot_dir)
    print(f"\nAll plots: {args.plot_dir}")


if __name__ == "__main__":
    main()
