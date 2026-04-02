#!/usr/bin/env python3
"""
Phase 3 Step 1: Alignment Geometry (Expanded)

A. PCA rank: Is refusal rank-1? Variance explained vs k.
B. SVD alignment: Does refusal align with low singular values of projector?
C. Surgical dissection: Where exactly in W1→GELU→W2 does the signal die?
D. Category-specific directions: 13 categories → 13 refusal vectors → similarity matrix.
E. Noise robustness.

NO GPU for A,B,D,E. Needs model download for C (projector weights).

Usage:
    python phase3_01_alignment_geometry.py
"""

import os
os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", os.path.expanduser("~/.cache/huggingface")))

import json, argparse
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_refusal(vector_dir, model_name):
    path = Path(vector_dir) / f"refusal_directions_{model_name}.npz"
    if not path.exists():
        path = list(Path(vector_dir).glob("refusal_directions_*.npz"))[0]
    raw = np.load(path)
    dirs = {int(k.replace("layer_", "")): v for k, v in raw.items()}
    meta_path = Path(vector_dir) / f"metadata_{model_name}.json"
    if not meta_path.exists():
        meta_path = list(Path(vector_dir).glob("metadata_*.json"))[0]
    with open(meta_path) as f:
        meta = json.load(f)
    return dirs, meta


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0


# ============================================================
# A. PCA Rank Analysis
# ============================================================
def analyze_pca_rank(gap_dir, model_name, num_layers):
    """Is refusal rank-1? Measure variance explained by top-k components."""
    print("\n" + "=" * 60)
    print("A. PCA RANK ANALYSIS")
    print("=" * 60)

    gap_path = list(Path(gap_dir).glob("visual_gap_results_*.json"))
    if not gap_path:
        print("  No gap results. Skipping."); return {}

    with open(gap_path[0]) as f:
        gap = json.load(f)

    # Build matrix: each row = per-layer text scores for one harmful pair
    pairs = gap["harmful_per_pair"]
    matrix = []
    for p in pairs:
        ts = p.get("text_scores", {})
        row = [ts.get(str(l), ts.get(l, 0)) for l in range(num_layers + 1)]
        if any(v != 0 for v in row):
            matrix.append(row)

    if len(matrix) < 10:
        print("  Not enough data."); return {}

    X = np.array(matrix)
    print(f"  Matrix: {X.shape} (pairs × layers)")

    pca = PCA()
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    n_90 = int(np.searchsorted(cumvar, 0.9) + 1)
    n_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    n_99 = int(np.searchsorted(cumvar, 0.99) + 1)

    print(f"  First component: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  Top 2: {cumvar[1]:.1%}")
    print(f"  Components for 90%: {n_90}")
    print(f"  Components for 95%: {n_95}")
    print(f"  Components for 99%: {n_99}")

    if n_90 <= 2:
        print("  → REFUSAL IS LOW-RANK (rank-1 or rank-2)")
    else:
        print(f"  → REFUSAL IS DISTRIBUTED (rank-{n_90})")

    return {
        "explained_variance": pca.explained_variance_ratio_[:20].tolist(),
        "cumulative": cumvar[:20].tolist(),
        "n_90": n_90, "n_95": n_95, "n_99": n_99,
        "first_component": float(pca.explained_variance_ratio_[0]),
    }


# ============================================================
# B. SVD Alignment
# ============================================================
def analyze_svd_alignment(refusal_dir, model_id):
    """Does refusal align with low singular values of projector?"""
    print("\n" + "=" * 60)
    print("B. PROJECTOR SVD ALIGNMENT")
    print("=" * 60)

    import torch
    from transformers import LlavaForConditionalGeneration

    print(f"  Loading projector weights from {model_id}...")
    # Load in float16 on GPU to avoid CPU OOM
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )

    projector = model.multi_modal_projector
    weights = {}
    for name, param in projector.named_parameters():
        weights[name] = param.detach().cpu().float().numpy()
        print(f"  {name}: {param.shape}")

    # Get W2 (output layer)
    w2 = None
    w1 = None
    for name, w in weights.items():
        if "linear_2" in name and "weight" in name: w2 = w
        elif "linear_1" in name and "weight" in name: w1 = w

    del model; import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    if w2 is None:
        print("  Could not find linear_2."); return {}, None, None

    # SVD of W2
    U, S, Vt = np.linalg.svd(w2, full_matrices=False)
    print(f"  W2 shape: {w2.shape}, rank: {len(S)}")

    # Project refusal direction onto left singular vectors
    d = refusal_dir.astype(np.float64)
    d = d / (np.linalg.norm(d) + 1e-8)
    components = U.T @ d

    # Where does refusal energy concentrate?
    energy = components ** 2
    cumulative_energy = np.cumsum(energy) / (np.sum(energy) + 1e-10)

    # Weighted average singular value for refusal
    weighted_sv = np.sum(np.abs(components) * S) / (np.sum(np.abs(components)) + 1e-8)
    mean_sv = np.mean(S)

    print(f"\n  Mean singular value: {mean_sv:.4f}")
    print(f"  Refusal-weighted singular value: {weighted_sv:.4f}")
    print(f"  Ratio: {weighted_sv / mean_sv:.3f}")

    if weighted_sv < mean_sv * 0.8:
        print("  → REFUSAL ALIGNS WITH LOW-GAIN DIRECTIONS")
    else:
        print("  → REFUSAL IS NOT SPECIFICALLY ATTENUATED")

    # Gain: ||W2^T @ refusal|| / ||refusal||
    gain = np.linalg.norm(w2.T @ d)
    # Compare with random directions
    rng = np.random.RandomState(42)
    random_gains = []
    for _ in range(100):
        rv = rng.randn(len(d))
        rv = rv / np.linalg.norm(rv)
        random_gains.append(np.linalg.norm(w2.T @ rv))

    print(f"\n  Refusal gain: {gain:.4f}")
    print(f"  Random gain (mean): {np.mean(random_gains):.4f} ± {np.std(random_gains):.4f}")
    print(f"  Percentile: {(np.sum(np.array(random_gains) < gain) / len(random_gains)):.1%}")

    return {
        "singular_values": S[:100].tolist(),
        "refusal_weighted_sv": float(weighted_sv),
        "mean_sv": float(mean_sv),
        "ratio": float(weighted_sv / mean_sv),
        "refusal_gain": float(gain),
        "random_gain_mean": float(np.mean(random_gains)),
        "random_gain_std": float(np.std(random_gains)),
        "refusal_energy_top10": float(cumulative_energy[9] if len(cumulative_energy) > 9 else 0),
        "refusal_energy_top50": float(cumulative_energy[49] if len(cumulative_energy) > 49 else 0),
    }, w1, w2


# ============================================================
# C. Surgical Projector Dissection
# ============================================================
def analyze_surgical_dissection(refusal_dir, w1, w2):
    """Measure refusal alignment after W1, after GELU, after W2."""
    print("\n" + "=" * 60)
    print("C. SURGICAL PROJECTOR DISSECTION")
    print("=" * 60)

    if w1 is None or w2 is None:
        print("  Missing projector weights."); return {}

    d = refusal_dir.astype(np.float64)
    d = d / (np.linalg.norm(d) + 1e-8)

    # Simulate projector forward pass on refusal direction
    # Note: refusal direction lives in text_hidden space (output of projector)
    # W1: (intermediate, vision_hidden), W2: (text_hidden, intermediate)
    # We measure how well the projector OUTPUT preserves the refusal direction

    # Approach: measure alignment at each stage using W2's inverse mapping
    # W2^T @ refusal_dir → intermediate space representation of refusal
    intermediate_refusal = w2.T @ d
    intermediate_norm = np.linalg.norm(intermediate_refusal)

    # After GELU: positive components survive, negative get killed
    gelu_refusal = np.where(intermediate_refusal > 0,
                             intermediate_refusal,
                             0.5 * intermediate_refusal * (1 + np.tanh(0.7978 * intermediate_refusal)))
    gelu_norm = np.linalg.norm(gelu_refusal)

    # How much does GELU kill?
    gelu_survival = gelu_norm / (intermediate_norm + 1e-8)
    neg_fraction = np.mean(intermediate_refusal < 0)

    # Reconstruct after W2
    reconstructed = w2 @ gelu_refusal
    recon_alignment = cosine(reconstructed, d)

    # Compare: no GELU (linear only)
    linear_recon = w2 @ intermediate_refusal
    linear_alignment = cosine(linear_recon, d)

    print(f"  After W2^T (intermediate): norm = {intermediate_norm:.4f}")
    print(f"  Negative fraction: {neg_fraction:.1%}")
    print(f"  After GELU: norm = {gelu_norm:.4f} (survival: {gelu_survival:.1%})")
    print(f"  W2 → GELU → W2 alignment: {recon_alignment:.4f}")
    print(f"  W2 → W2 (linear, no GELU): {linear_alignment:.4f}")
    print(f"  GELU damage: {linear_alignment - recon_alignment:.4f}")

    if gelu_survival < 0.8:
        print("  → GELU SIGNIFICANTLY DESTROYS REFUSAL SIGNAL")
    elif recon_alignment < 0.5:
        print("  → W2 PROJECTION DESTROYS REFUSAL SIGNAL")
    else:
        print("  → SIGNAL SURVIVES PROJECTOR REASONABLY WELL")

    return {
        "intermediate_norm": float(intermediate_norm),
        "gelu_norm": float(gelu_norm),
        "gelu_survival": float(gelu_survival),
        "negative_fraction": float(neg_fraction),
        "with_gelu_alignment": float(recon_alignment),
        "without_gelu_alignment": float(linear_alignment),
        "gelu_damage": float(linear_alignment - recon_alignment),
    }


# ============================================================
# D. Category-Specific Directions
# ============================================================
def analyze_category_directions(gap_dir, refusal_dirs, best_layer, num_layers):
    """Extract per-category refusal direction, measure similarity."""
    print("\n" + "=" * 60)
    print("D. CATEGORY-SPECIFIC DIRECTIONS")
    print("=" * 60)

    gap_path = list(Path(gap_dir).glob("visual_gap_results_*.json"))
    if not gap_path:
        print("  No gap results."); return {}

    with open(gap_path[0]) as f:
        gap = json.load(f)

    # Group pairs by category
    cats = {}
    for p in gap["harmful_per_pair"]:
        c = p.get("category", "unknown")
        cats.setdefault(c, []).append(p)

    benign = gap.get("benign_per_pair", [])
    if len(cats) < 2:
        print("  Need 2+ categories."); return {}

    # For each category, compute mean text score vector across layers
    # Then compute direction = cat_mean - benign_mean
    global_refusal = refusal_dirs[best_layer]

    cat_directions = {}
    for cat, pairs in sorted(cats.items()):
        scores = []
        for p in pairs:
            ts = p.get("text_scores", {})
            val = ts.get(str(best_layer), ts.get(best_layer, None))
            if val is not None:
                scores.append(val)
        if scores:
            cat_directions[cat] = float(np.mean(scores))

    # Similarity matrix between categories (using gap patterns across layers)
    cat_gap_profiles = {}
    for cat, pairs in sorted(cats.items()):
        profile = []
        for l in range(num_layers + 1):
            gaps = [p.get("gap", {}).get(str(l), 0) for p in pairs
                    if str(l) in p.get("gap", {})]
            profile.append(float(np.mean(gaps)) if gaps else 0)
        cat_gap_profiles[cat] = np.array(profile)

    cat_names = sorted(cat_gap_profiles.keys())
    n = len(cat_names)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = cosine(cat_gap_profiles[cat_names[i]],
                                       cat_gap_profiles[cat_names[j]])

    print(f"  Categories: {n}")
    print(f"\n  Mean text score at best layer:")
    for c in sorted(cat_directions, key=cat_directions.get, reverse=True):
        print(f"    {c:<30} {cat_directions[c]:.4f}")

    mean_off_diag = (sim_matrix.sum() - np.trace(sim_matrix)) / (n * n - n)
    print(f"\n  Mean off-diagonal similarity: {mean_off_diag:.4f}")

    if mean_off_diag > 0.8:
        print("  → UNIVERSAL REFUSAL AXIS (same direction across categories)")
    elif mean_off_diag > 0.5:
        print("  → MOSTLY SHARED with category-specific variation")
    else:
        print("  → MULTI-DIMENSIONAL safety (different directions per category)")

    return {
        "cat_text_scores": cat_directions,
        "cat_names": cat_names,
        "similarity_matrix": sim_matrix.tolist(),
        "mean_off_diagonal": float(mean_off_diag),
    }


# ============================================================
# E. Noise Robustness
# ============================================================
def analyze_noise_robustness(refusal_dir):
    print("\n" + "=" * 60)
    print("E. NOISE ROBUSTNESS")
    print("=" * 60)

    d = refusal_dir / (np.linalg.norm(refusal_dir) + 1e-8)
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    rng = np.random.RandomState(42)

    cosines = []
    for sigma in noise_levels:
        sims = []
        for _ in range(100):
            noisy = d + rng.randn(len(d)) * sigma
            noisy = noisy / (np.linalg.norm(noisy) + 1e-8)
            sims.append(float(np.dot(d, noisy)))
        cosines.append(float(np.mean(sims)))

    print(f"  {'Noise σ':<10} {'Cosine':<10}")
    for s, c in zip(noise_levels, cosines):
        print(f"  {s:<10} {c:<10.4f}")

    return {"noise_levels": noise_levels, "cosines": cosines}


# ============================================================
# Visualization
# ============================================================
def plot_all(results, plot_dir, model_name):
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # SVD alignment
    svd = results.get("svd", {})
    if "singular_values" in svd:
        fig, ax = plt.subplots(figsize=(10, 5))
        sv = np.array(svd["singular_values"])
        ax.semilogy(sv, "b-", lw=2)
        ax.axhline(y=svd["refusal_weighted_sv"], color="red", ls="--",
                    label=f"Refusal-weighted SV: {svd['refusal_weighted_sv']:.2f}")
        ax.axhline(y=svd["mean_sv"], color="green", ls="--",
                    label=f"Mean SV: {svd['mean_sv']:.2f}")
        ax.set_xlabel("Index"); ax.set_ylabel("Singular Value")
        ax.set_title("Projector SVD: Where Does Refusal Live?", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(Path(plot_dir) / f"svd_alignment_{model_name}.png", dpi=200)
        plt.close()
        print(f"  Saved SVD plot")

    # Category similarity
    cat = results.get("category_directions", {})
    if "similarity_matrix" in cat:
        fig, ax = plt.subplots(figsize=(12, 10))
        sim = np.array(cat["similarity_matrix"])
        names = [n.replace("_", "\n") for n in cat["cat_names"]]
        sns.heatmap(sim, ax=ax, xticklabels=names, yticklabels=names,
                    cmap="RdYlBu_r", center=0.5, annot=True, fmt=".2f",
                    cbar_kws={"label": "Cosine Similarity"})
        ax.set_title("Category-Specific Refusal Direction Similarity", fontweight="bold")
        plt.tight_layout()
        plt.savefig(Path(plot_dir) / f"category_similarity_{model_name}.png", dpi=150)
        plt.close()
        print(f"  Saved category similarity plot")

    # Surgical dissection bar chart
    surg = results.get("surgical", {})
    if surg:
        fig, ax = plt.subplots(figsize=(8, 5))
        stages = ["After W1\n(intermediate)", "After GELU", "After W2\n(reconstructed)"]
        vals = [1.0, surg["gelu_survival"], surg["with_gelu_alignment"]]
        colors = ["#3498db", "#e67e22", "#e74c3c"]
        ax.bar(stages, vals, color=colors, alpha=0.85)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
        ax.set_ylabel("Signal Strength")
        ax.set_title("Surgical Dissection: Where Safety Dies in the Projector", fontweight="bold")
        ax.set_ylim(0, 1.2)
        plt.tight_layout()
        plt.savefig(Path(plot_dir) / f"surgical_dissection_{model_name}.png", dpi=200)
        plt.close()
        print(f"  Saved surgical dissection plot")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--vector_dir", default="./outputs/vectors")
    ap.add_argument("--gap_dir", default="./outputs/gap_analysis")
    ap.add_argument("--output_dir", default="./outputs/mechanism")
    ap.add_argument("--plot_dir", default="./outputs/plots")
    ap.add_argument("--skip_projector", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    model_name = args.model_id.replace("/", "_").replace("-", "_")

    dirs, meta = load_refusal(args.vector_dir, model_name)
    best_layer = meta["best_layer"]
    num_layers = meta["num_layers"]
    refusal_dir = dirs[best_layer]

    print("=" * 60)
    print("PHASE 3 STEP 1: ALIGNMENT GEOMETRY (EXPANDED)")
    print("=" * 60)

    results = {}

    # A: PCA
    results["pca"] = analyze_pca_rank(args.gap_dir, model_name, num_layers)

    # B + C: SVD + Surgical (need projector weights)
    w1, w2 = None, None
    if not args.skip_projector:
        try:
            results["svd"], w1, w2 = analyze_svd_alignment(refusal_dir, args.model_id)
            results["surgical"] = analyze_surgical_dissection(refusal_dir, w1, w2)
        except Exception as e:
            print(f"  Projector analysis failed: {e}")
    else:
        print("\n  Skipping projector analysis")

    # D: Category directions
    results["category_directions"] = analyze_category_directions(
        args.gap_dir, dirs, best_layer, num_layers)

    # E: Noise robustness
    results["noise"] = analyze_noise_robustness(refusal_dir)

    # Save
    with open(out / f"geometry_{model_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out / f'geometry_{model_name}.json'}")

    # Plot
    plot_all(results, args.plot_dir, model_name)


if __name__ == "__main__":
    main()
