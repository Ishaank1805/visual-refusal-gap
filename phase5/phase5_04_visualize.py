#!/usr/bin/env python3
"""
Phase 5 - Step 4: Defense Visualization

Plots defense comparison: baseline vs always-on vs gated vs oracle vs adapter.

Usage:
    python phase5_04_visualize.py
"""

import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense_dir", default="./outputs/defense")
    ap.add_argument("--plot_dir", default="./outputs/plots")
    args = ap.parse_args()

    Path(args.plot_dir).mkdir(parents=True, exist_ok=True)

    # Collect all defense results
    modes = {}

    for p in Path(args.defense_dir).glob("defense_results_*.json"):
        with open(p) as f:
            d = json.load(f)
        modes["baseline"] = {"asr": d["baseline"]["asr"], "fpr": d["baseline"]["fpr"]}
        modes["always-on"] = {"asr": d["always_on"]["asr"], "fpr": d["always_on"]["fpr"]}
        modes["oracle"] = {"asr": d["oracle"]["asr"], "fpr": d["oracle"]["fpr"]}

    for p in Path(args.defense_dir).glob("gated_steering_*.json"):
        with open(p) as f:
            d = json.load(f)
        modes["gated"] = {"asr": d["gated_asr"], "fpr": d["gated_fpr"]}

    for p in Path(args.defense_dir).glob("adapter_training_*.json"):
        with open(p) as f:
            d = json.load(f)
        # Adapter doesn't have ASR/FPR directly — use gap closure
        modes["adapter"] = {"gap_closed": d.get("gap_closed_pct", 0)}

    if not modes:
        print("No defense results found"); return

    # Plot: ASR vs FPR comparison
    asr_modes = {k: v for k, v in modes.items() if "asr" in v}
    if asr_modes:
        names = list(asr_modes.keys())
        asrs = [asr_modes[n]["asr"] for n in names]
        fprs = [asr_modes[n]["fpr"] for n in names]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ASR comparison
        ax = axes[0]
        colors = {"baseline": "#e74c3c", "always-on": "#f39c12", "gated": "#27ae60", "oracle": "#3498db"}
        c = [colors.get(n, "#95a5a6") for n in names]
        bars = ax.bar(names, asrs, color=c, alpha=0.85)
        for bar, v in zip(bars, asrs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f"{v:.0%}", ha="center", fontsize=12, fontweight="bold")
        ax.set_ylabel("Attack Success Rate ↓", fontsize=12)
        ax.set_title("ASR Comparison (lower is better)", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")

        # FPR comparison
        ax = axes[1]
        bars = ax.bar(names, fprs, color=c, alpha=0.85)
        for bar, v in zip(bars, fprs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f"{v:.0%}", ha="center", fontsize=12, fontweight="bold")
        ax.set_ylabel("False Positive Rate ↓", fontsize=12)
        ax.set_title("FPR Comparison (lower is better)", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = Path(args.plot_dir) / "defense_comparison.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

        # Scatter: ASR vs FPR tradeoff
        fig, ax = plt.subplots(figsize=(8, 6))
        for n in names:
            ax.scatter(asr_modes[n]["fpr"], asr_modes[n]["asr"],
                       s=150, color=colors.get(n, "#95a5a6"), zorder=5)
            ax.annotate(n, (asr_modes[n]["fpr"], asr_modes[n]["asr"]),
                        textcoords="offset points", xytext=(10, 5), fontsize=11)
        ax.set_xlabel("False Positive Rate (benign images refused) →", fontsize=12)
        ax.set_ylabel("Attack Success Rate (harmful images pass) →", fontsize=12)
        ax.set_title("Safety–Utility Tradeoff", fontsize=14, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.plot([0, 1], [1, 0], "k--", alpha=0.2)  # ideal frontier
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = Path(args.plot_dir) / "safety_utility_tradeoff.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    print(f"\nAll plots: {args.plot_dir}")


if __name__ == "__main__":
    main()
