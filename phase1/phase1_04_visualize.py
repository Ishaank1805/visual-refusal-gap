#!/usr/bin/env python3
"""
Phase 1 - Step 4: Visualize Refusal Direction Results

Generates:
  1. Layer-wise refusal scores (harmful vs benign separation)
  2. Score distributions at key layers
  3. Steering validation curves (alpha vs refusal rate)

Usage:
    python phase1_04_visualize.py
    python phase1_04_visualize.py --vector_dir ./outputs/vectors --plot_dir ./outputs/plots
"""

import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_layer_scores(metadata, plot_dir, model_name):
    """Harmful vs benign cosine similarity across layers."""
    scores = metadata["scores"]
    layers = sorted(int(l) for l in scores.keys())

    harmful = [scores[str(l)]["harmful_mean"] for l in layers]
    benign = [scores[str(l)]["benign_mean"] for l in layers]
    gap = [scores[str(l)]["gap"] for l in layers]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(layers, harmful, "r-o", markersize=4, linewidth=2.5,
            label="Harmful prompts (should trigger refusal)")
    ax.plot(layers, benign, "b-s", markersize=4, linewidth=2.5,
            label="Benign prompts (should not trigger)")
    ax.fill_between(layers, harmful, benign, alpha=0.15, color="purple",
                     label="Separation gap")

    best = metadata["best_layer"]
    ax.axvline(x=best, color="green", linestyle="--", alpha=0.7,
               label=f"Best layer: {best} (gap={metadata['best_gap']:.4f})")

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Cosine Similarity with Refusal Direction", fontsize=13)
    ax.set_title("Refusal Direction Separation Across Layers", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)

    ax2 = axes[1]
    ax2.bar(layers, gap, color="purple", alpha=0.6)
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Gap", fontsize=12)
    ax2.set_title("Harmful − Benign Gap", fontsize=12)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(plot_dir) / f"refusal_scores_by_layer_{model_name}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_steering_validation(log_path, plot_dir, model_name):
    """Alpha vs refusal rate for add/subtract experiments."""
    with open(log_path) as f:
        results = json.load(f)

    experiments = results["experiments"]
    baseline_h = results["baseline_harmful_refusal_rate"]
    baseline_b = results["baseline_benign_refusal_rate"]

    additive = [e for e in experiments if e["method"] == "additive"]
    if not additive:
        additive = experiments  # fallback

    alphas = [e["alpha"] for e in additive]
    add_rates = [e["add_to_benign_refusal_rate"] for e in additive]
    sub_rates = [e["subtract_from_harmful_refusal_rate"] for e in additive]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Add to benign
    ax = axes[0]
    ax.plot(alphas, add_rates, "r-o", markersize=8, linewidth=2.5, label="Steered benign")
    ax.axhline(y=baseline_b, color="blue", linestyle="--", alpha=0.7,
               label=f"Baseline benign: {baseline_b:.1%}")
    ax.set_xlabel("Alpha (steering strength)", fontsize=12)
    ax.set_ylabel("Refusal Rate", fontsize=12)
    ax.set_title("Add Refusal → Benign Prompts\n(should induce refusal)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    for i, (a, r) in enumerate(zip(alphas, add_rates)):
        ax.annotate(f"{r:.0%}", (a, r), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    # Subtract from harmful
    ax = axes[1]
    ax.plot(alphas, sub_rates, "b-o", markersize=8, linewidth=2.5, label="Steered harmful")
    ax.axhline(y=baseline_h, color="red", linestyle="--", alpha=0.7,
               label=f"Baseline harmful: {baseline_h:.1%}")
    ax.set_xlabel("Alpha (steering strength)", fontsize=12)
    ax.set_ylabel("Refusal Rate", fontsize=12)
    ax.set_title("Subtract Refusal → Harmful Prompts\n(should remove refusal)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    for i, (a, r) in enumerate(zip(alphas, sub_rates)):
        ax.annotate(f"{r:.0%}", (a, r), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    path = Path(plot_dir) / f"steering_validation_{model_name}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description="Phase 1 visualization")
    ap.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--vector_dir", default="/scratch/ishaan.karan/outputs/vectors")
    ap.add_argument("--log_dir", default="/scratch/ishaan.karan/outputs/logs")
    ap.add_argument("--plot_dir", default="/scratch/ishaan.karan/outputs/plots")
    args = ap.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_id.replace("/", "_").replace("-", "_")

    print("Generating Phase 1 plots...")

    # Layer scores
    meta_path = Path(args.vector_dir) / f"metadata_{model_name}.json"
    if not meta_path.exists():
        candidates = list(Path(args.vector_dir).glob("metadata_*.json"))
        meta_path = candidates[0] if candidates else None
    if meta_path and meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        plot_layer_scores(metadata, args.plot_dir, model_name)
    else:
        print(f"  Skipping layer scores (no metadata found)")

    # Steering validation
    log_path = Path(args.log_dir) / f"validation_results_{model_name}.json"
    if not log_path.exists():
        candidates = list(Path(args.log_dir).glob("validation_results_*.json"))
        log_path = candidates[0] if candidates else None
    if log_path and log_path.exists():
        plot_steering_validation(log_path, args.plot_dir, model_name)
    else:
        print(f"  Skipping steering plot (no validation results)")

    print(f"\nAll plots: {args.plot_dir}")


if __name__ == "__main__":
    main()
