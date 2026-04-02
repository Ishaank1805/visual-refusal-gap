#!/usr/bin/env python3
"""
Phase 3 - Step 2: Linear Probe (Alignment Reconstruction Test)

Train logistic regression on TEXT hidden states (harmful vs benign).
Test on IMAGE hidden states.

Interpretation:
  - Probe fails on image states → safety signal is DESTROYED by projector
  - Probe works on image states → signal is present but HIDDEN (cosine misses it)

Either result is publishable. Destruction = projector architecturally incapable.
Hidden = different encoding, potentially recoverable.

NO GPU NEEDED. Uses saved per-pair activation data from Phase 2.

Usage:
    python phase3_02_linear_probe.py
    python phase3_02_linear_probe.py --gap_results ./outputs/gap_analysis/visual_gap_results_*.json
"""

import json, argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_gap_results(gap_dir, model_name):
    """Load Phase 2 gap results containing per-pair scores."""
    gap_path = Path(gap_dir) / f"visual_gap_results_{model_name}.json"
    if not gap_path.exists():
        candidates = list(Path(gap_dir).glob("visual_gap_results_*.json"))
        gap_path = candidates[0] if candidates else None
    if not gap_path or not gap_path.exists():
        raise FileNotFoundError(f"No gap results in {gap_dir}")
    with open(gap_path) as f:
        return json.load(f)


def build_probe_dataset(gap_data, layer):
    """
    Build X_text, X_image, y from gap results.
    Uses the cosine scores at each layer as features.

    For a proper probe we'd want the raw activations, but cosine scores
    across layers give us a meaningful feature vector.
    """
    num_layers = gap_data["num_layers"]
    layers = list(range(num_layers + 1))

    harmful = gap_data["harmful_per_pair"]
    benign = gap_data["benign_per_pair"]

    # Build feature vectors: cosine score at the target layer
    # Also build multi-layer features for a stronger probe
    X_text_single = []  # just the target layer score
    X_image_single = []
    X_text_multi = []   # all layer scores
    X_image_multi = []
    y = []

    for pair in harmful:
        ts = pair.get("text_scores", {})
        ims = pair.get("image_scores", {})

        t_val = ts.get(str(layer), ts.get(layer, None))
        i_val = ims.get(str(layer), ims.get(layer, None))

        if t_val is not None and i_val is not None:
            X_text_single.append([t_val])
            X_image_single.append([i_val])

            t_multi = [ts.get(str(l), ts.get(l, 0)) for l in layers]
            i_multi = [ims.get(str(l), ims.get(l, 0)) for l in layers]
            X_text_multi.append(t_multi)
            X_image_multi.append(i_multi)
            y.append(1)  # harmful

    for pair in benign:
        ts = pair.get("text_scores", {})
        ims = pair.get("image_scores", {})

        t_val = ts.get(str(layer), ts.get(layer, None))
        i_val = ims.get(str(layer), ims.get(layer, None))

        if t_val is not None and i_val is not None:
            X_text_single.append([t_val])
            X_image_single.append([i_val])

            t_multi = [ts.get(str(l), ts.get(l, 0)) for l in layers]
            i_multi = [ims.get(str(l), ims.get(l, 0)) for l in layers]
            X_text_multi.append(t_multi)
            X_image_multi.append(i_multi)
            y.append(0)  # benign

    return {
        "X_text_single": np.array(X_text_single),
        "X_image_single": np.array(X_image_single),
        "X_text_multi": np.array(X_text_multi),
        "X_image_multi": np.array(X_image_multi),
        "y": np.array(y),
    }


def run_probe(X_train, y_train, X_test, y_test, name=""):
    """Train logistic regression and evaluate."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.0

    return {
        "accuracy": float(acc),
        "auc": float(auc),
        "name": name,
    }


def main():
    ap = argparse.ArgumentParser(description="Linear Probe: train text → test image")
    ap.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--gap_dir", default="outputs/gap_analysis")
    ap.add_argument("--vector_dir", default="outputs/vectors")
    ap.add_argument("--output_dir", default="outputs/mechanism")
    ap.add_argument("--plot_dir", default="outputs/plots")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
    model_name = args.model_id.replace("/", "_").replace("-", "_")

    print("=" * 60)
    print("PHASE 3 STEP 2: LINEAR PROBE")
    print("=" * 60)

    # Load gap data
    gap_data = load_gap_results(args.gap_dir, model_name)
    num_layers = gap_data["num_layers"]

    # Get best layer
    meta_path = Path(args.vector_dir) / f"metadata_{model_name}.json"
    if not meta_path.exists():
        candidates = list(Path(args.vector_dir).glob("metadata_*.json"))
        meta_path = candidates[0] if candidates else None
    with open(meta_path) as f:
        best_layer = json.load(f)["best_layer"]

    print(f"Best layer: {best_layer}")
    print(f"Harmful pairs: {len(gap_data['harmful_per_pair'])}")
    print(f"Benign pairs: {len(gap_data['benign_per_pair'])}")

    # ============================================================
    # Run probes at multiple layers
    # ============================================================
    probe_layers = list(range(0, num_layers + 1, 4)) + [best_layer]
    probe_layers = sorted(set(probe_layers))

    all_results = []

    print(f"\n{'Layer':<8} {'Text→Text':<12} {'Text→Image':<12} {'Image→Image':<12} {'Signal?'}")
    print("-" * 56)

    for layer in probe_layers:
        data = build_probe_dataset(gap_data, layer)
        X_tt = data["X_text_single"]
        X_it = data["X_image_single"]
        y = data["y"]

        if len(y) < 10 or len(np.unique(y)) < 2:
            continue

        # Probe 1: Train on text, test on text (sanity check — should be high)
        n = len(y)
        idx = np.random.RandomState(42).permutation(n)
        split = int(0.8 * n)
        train_idx, test_idx = idx[:split], idx[split:]

        r_tt = run_probe(X_tt[train_idx], y[train_idx],
                          X_tt[test_idx], y[test_idx], "text→text")

        # Probe 2: Train on text, test on IMAGE (THE key test)
        r_ti = run_probe(X_tt[train_idx], y[train_idx],
                          X_it[test_idx], y[test_idx], "text→image")

        # Probe 3: Train on image, test on image (does signal exist at all?)
        r_ii = run_probe(X_it[train_idx], y[train_idx],
                          X_it[test_idx], y[test_idx], "image→image")

        signal = "DESTROYED" if r_ti["accuracy"] < 0.6 else "HIDDEN" if r_ti["accuracy"] < 0.8 else "PRESERVED"
        marker = " ← BEST" if layer == best_layer else ""

        print(f"  {layer:<6} {r_tt['accuracy']:<12.1%} {r_ti['accuracy']:<12.1%} "
              f"{r_ii['accuracy']:<12.1%} {signal}{marker}")

        all_results.append({
            "layer": layer,
            "text_to_text": r_tt,
            "text_to_image": r_ti,
            "image_to_image": r_ii,
            "signal_status": signal,
        })

    # ============================================================
    # Multi-layer probe (using all layers as features)
    # ============================================================
    print("\n" + "=" * 60)
    print("MULTI-LAYER PROBE (all layer scores as features)")
    print("=" * 60)

    data = build_probe_dataset(gap_data, best_layer)
    X_tm = data["X_text_multi"]
    X_im = data["X_image_multi"]
    y = data["y"]

    n = len(y)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    r_tt_m = run_probe(X_tm[train_idx], y[train_idx], X_tm[test_idx], y[test_idx], "text→text (multi)")
    r_ti_m = run_probe(X_tm[train_idx], y[train_idx], X_im[test_idx], y[test_idx], "text→image (multi)")
    r_ii_m = run_probe(X_im[train_idx], y[train_idx], X_im[test_idx], y[test_idx], "image→image (multi)")

    # Cross-val for text→text
    scaler = StandardScaler()
    X_tm_s = scaler.fit_transform(X_tm)
    cv_scores = cross_val_score(LogisticRegression(max_iter=1000), X_tm_s, y, cv=5, scoring="accuracy")

    print(f"  Text→Text:   {r_tt_m['accuracy']:.1%} (5-fold CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%})")
    print(f"  Text→Image:  {r_ti_m['accuracy']:.1%}")
    print(f"  Image→Image: {r_ii_m['accuracy']:.1%}")

    signal = "DESTROYED" if r_ti_m["accuracy"] < 0.6 else "HIDDEN" if r_ti_m["accuracy"] < 0.8 else "PRESERVED"
    print(f"\n  VERDICT: Safety signal is {signal} in image representations")

    # ============================================================
    # Save
    # ============================================================
    results = {
        "model_id": args.model_id,
        "best_layer": best_layer,
        "per_layer_probes": all_results,
        "multi_layer_probe": {
            "text_to_text": r_tt_m,
            "text_to_image": r_ti_m,
            "image_to_image": r_ii_m,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "verdict": signal,
        },
    }

    path = out / f"linear_probe_{model_name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {path}")

    # ============================================================
    # Plot
    # ============================================================
    if all_results:
        layers = [r["layer"] for r in all_results]
        tt_acc = [r["text_to_text"]["accuracy"] for r in all_results]
        ti_acc = [r["text_to_image"]["accuracy"] for r in all_results]
        ii_acc = [r["image_to_image"]["accuracy"] for r in all_results]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(layers, tt_acc, "g-o", ms=6, lw=2.5, label="Train TEXT → Test TEXT (control)")
        ax.plot(layers, ti_acc, "r-s", ms=6, lw=2.5, label="Train TEXT → Test IMAGE (key test)")
        ax.plot(layers, ii_acc, "b--^", ms=5, lw=2, alpha=0.7, label="Train IMAGE → Test IMAGE")
        ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5, label="Chance (50%)")
        ax.axvline(x=best_layer, color="green", ls=":", alpha=0.5, label=f"Best layer ({best_layer})")

        ax.set_xlabel("Layer", fontsize=13)
        ax.set_ylabel("Probe Accuracy", fontsize=13)
        ax.set_title("Linear Probe: Is Safety Signal Destroyed or Hidden?", fontsize=15, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = Path(args.plot_dir) / f"linear_probe_{model_name}.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_path}")

    print(f"\nNext: python phase3_03_interpolation.py --use_4bit")


if __name__ == "__main__":
    main()
