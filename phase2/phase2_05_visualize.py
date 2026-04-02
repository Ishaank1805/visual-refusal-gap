#!/usr/bin/env python3
"""
Phase 2 - Step 5: Visualization

Generates publication-quality plots:
  1. Layer-wise refusal gap (text vs image)
  2. Category heatmap
  3. Behavioral 3-way bar chart

Usage:
    python phase2_05_visualize.py
"""

import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_visual_gap(output_dir, plot_dir, model_name):
    """THE key figure: text vs image refusal alignment per layer."""
    arr_path = Path(output_dir) / f"visual_gap_arrays_{model_name}.npz"
    if not arr_path.exists():
        candidates = list(Path(output_dir).glob("visual_gap_arrays_*.npz"))
        arr_path = candidates[0] if candidates else None
    if not arr_path or not arr_path.exists():
        print("  Skipping gap plot (no arrays)"); return

    data = np.load(arr_path)
    layers = np.arange(len(data["harmful_text_means"]))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(layers, data["harmful_text_means"], "r-o", ms=4, lw=2.5,
            label="Harmful TEXT (should trigger refusal)", zorder=5)
    ax.plot(layers, data["harmful_img_means"], "r--s", ms=4, lw=2.5, alpha=0.7,
            label="Harmful IMAGE (fails to trigger)", zorder=5)
    ax.fill_between(layers, data["harmful_text_means"], data["harmful_img_means"],
                    alpha=0.15, color="red", label="Visual Refusal Gap")
    ax.plot(layers, data["benign_text_means"], "b-^", ms=3, lw=1.5, alpha=0.6, label="Benign TEXT")
    ax.plot(layers, data["benign_img_means"], "b--d", ms=3, lw=1.5, alpha=0.6, label="Benign IMAGE")

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Cosine Similarity with Refusal Direction", fontsize=13)
    ax.set_title("The Visual Refusal Gap", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", ls="--", alpha=0.4)

    ax2 = axes[1]
    ax2.bar(layers - 0.2, data["harmful_gap_means"], width=0.4, color="red", alpha=0.7, label="Harmful gap")
    ax2.bar(layers + 0.2, data["benign_gap_means"], width=0.4, color="blue", alpha=0.5, label="Benign gap")
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Gap (text − image)", fontsize=12)
    ax2.axhline(y=0, color="black", lw=0.5)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(plot_dir) / f"visual_refusal_gap_{model_name}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_category_heatmap(output_dir, plot_dir, model_name):
    """Gap by category and layer."""
    res_path = Path(output_dir) / f"visual_gap_results_{model_name}.json"
    if not res_path.exists():
        candidates = list(Path(output_dir).glob("visual_gap_results_*.json"))
        res_path = candidates[0] if candidates else None
    if not res_path or not res_path.exists():
        print("  Skipping heatmap (no results)"); return

    with open(res_path) as f:
        results = json.load(f)

    pairs = results["harmful_per_pair"]
    num_layers = results["num_layers"]

    cats = {}
    for p in pairs:
        c = p.get("category", "unknown")
        cats.setdefault(c, []).append(p)

    if not cats: return

    layer_idx = list(range(0, num_layers + 1, max(1, num_layers // 16)))
    cat_names = sorted(cats.keys())

    heatmap = np.zeros((len(cat_names), len(layer_idx)))
    for i, c in enumerate(cat_names):
        for j, l in enumerate(layer_idx):
            gaps = [p["gap"].get(str(l), 0) for p in cats[c] if str(l) in p.get("gap", {})]
            if gaps:
                heatmap[i, j] = np.mean(gaps)

    fig, ax = plt.subplots(figsize=(14, max(4, len(cat_names) * 0.6 + 2)))
    sns.heatmap(heatmap, ax=ax, xticklabels=[str(l) for l in layer_idx],
                yticklabels=cat_names, cmap="RdYlBu_r", center=0,
                annot=len(cat_names) <= 15, fmt=".3f",
                cbar_kws={"label": "Gap (text − image)"})
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Category", fontsize=12)
    ax.set_title("Visual Refusal Gap by Category", fontsize=14)
    plt.tight_layout()

    path = Path(plot_dir) / f"category_heatmap_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_behavioral(output_dir, plot_dir, model_name):
    """3-way behavioral bar chart."""
    behav_path = Path(output_dir) / f"behavioral_judged_{model_name}.json"
    if not behav_path.exists():
        candidates = list(Path(output_dir).glob("behavioral_judged_*.json"))
        behav_path = candidates[0] if candidates else None
    if not behav_path or not behav_path.exists():
        print("  Skipping behavioral plot (no judged results)"); return

    with open(behav_path) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: 3-way breakdown
    ax = axes[0]
    categories = ["Text\n(harmful)", "Image\n(harmful)"]
    refused = [results.get("harmful_text_refusal_rate", 0),
               results.get("harmful_image_refusal_rate", 0)]
    complied = [results.get("harmful_text_compliance_rate", 0),
                results.get("harmful_image_compliance_rate", 0)]
    evasion = [1 - refused[0] - complied[0], 1 - refused[1] - complied[1]]

    x = np.arange(len(categories))
    w = 0.25
    ax.bar(x - w, refused, w, color="#e74c3c", alpha=0.85, label="REFUSED")
    ax.bar(x, complied, w, color="#27ae60", alpha=0.85, label="COMPLIED")
    ax.bar(x + w, evasion, w, color="#95a5a6", alpha=0.85, label="EVASION")

    for i in range(2):
        ax.text(x[i] - w, refused[i] + 0.02, f"{refused[i]:.0%}", ha="center", fontsize=9, fontweight="bold")
        ax.text(x[i], complied[i] + 0.02, f"{complied[i]:.0%}", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title("Model Response: Text vs Image\n(3-way classification)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: gap and jailbreak
    ax2 = axes[1]
    gap = results.get("behavioral_gap", 0)
    jb = results.get("jailbreak_rate", 0)
    bars = ax2.bar(["Behavioral\nGap", "Jailbreak\nRate"],
                    [gap, jb], color=["#e67e22", "#c0392b"], alpha=0.85, width=0.5)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                 f"{bar.get_height():.1%}", ha="center", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Rate", fontsize=12)
    ax2.set_title("Visual Jailbreak Effectiveness", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = Path(plot_dir) / f"behavioral_results_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description="Phase 2 visualization")
    ap.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--output_dir", default="outputs/gap_analysis")
    ap.add_argument("--plot_dir", default="outputs/plots")
    args = ap.parse_args()

    Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
    model_name = args.model_id.replace("/", "_").replace("-", "_")

    print("Generating Phase 2 plots...")
    plot_visual_gap(args.output_dir, args.plot_dir, model_name)
    plot_category_heatmap(args.output_dir, args.plot_dir, model_name)
    plot_behavioral(args.output_dir, args.plot_dir, model_name)
    print(f"\nAll plots: {args.plot_dir}")


if __name__ == "__main__":
    main()
