#!/usr/bin/env python3
"""
Phase 3 Step 3: Interpolation + Refusal Boundary Estimation

A. Interpolation: h(t) = (1-t)*h_image + t*h_text, measure refusal at each t.
B. Binary search: find exact threshold α* where refusal activates per example.

GPU NEEDED.

Usage:
    python phase3_03_interpolation.py --use_4bit
    python phase3_03_interpolation.py --use_4bit --num_pairs 50
"""

import os
os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", "/scratch/ishaan.karan/hf_cache"))

import gc, json, argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def clear_vram():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.synchronize()

def load_model(model_id, use_4bit=True):
    print(f"Loading {model_id}...")
    clear_vram()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True) if use_4bit else None
    model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16,
        quantization_config=quant, device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor.tokenizer, processor

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0

def get_text_hidden(model, tokenizer, text, layer):
    prompt = f"USER: {text}\nASSISTANT:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.language_model(**inputs, output_hidden_states=True)
    h = out.hidden_states[layer][0, -1, :].detach().cpu()
    del out, inputs; clear_vram()
    return h

def get_image_hidden(model, processor, text, img_path, layer):
    image = Image.open(img_path).convert("RGB")
    prompt = f"USER: <image>\n{text}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = out.hidden_states[layer][0, -1, :].detach().cpu()
    del out, inputs, image; clear_vram()
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--vector_dir", default="/scratch/ishaan.karan/outputs/vectors")
    ap.add_argument("--hazards_dir", default="/scratch/ishaan.karan/data/visual_hazards_v2")
    ap.add_argument("--output_dir", default="/scratch/ishaan.karan/outputs/mechanism")
    ap.add_argument("--plot_dir", default="/scratch/ishaan.karan/outputs/plots")
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--num_pairs", type=int, default=50)
    ap.add_argument("--num_steps", type=int, default=21)
    ap.add_argument("--binary_search_steps", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    model_name = args.model_id.replace("/", "_").replace("-", "_")

    # Load refusal direction
    ref_path = list(Path(args.vector_dir).glob("refusal_directions_*.npz"))[0]
    raw = np.load(ref_path)
    refusal_dirs = {int(k.replace("layer_", "")): v for k, v in raw.items()}
    meta_path = list(Path(args.vector_dir).glob("metadata_*.json"))[0]
    with open(meta_path) as f:
        meta = json.load(f)
    best_layer = meta["best_layer"]
    ref_dir = refusal_dirs[best_layer]

    # Load dataset
    with open(Path(args.hazards_dir) / "visual_hazards_metadata.json") as f:
        dataset = json.load(f)
    pairs = [p for p in dataset["harmful_pairs"]
             if "typographic_clean" in p.get("images", {})][:args.num_pairs]

    print("=" * 60)
    print("PHASE 3 STEP 3: INTERPOLATION + BOUNDARY")
    print("=" * 60)
    print(f"  Pairs: {len(pairs)}, Steps: {args.num_steps}, Layer: {best_layer}")

    model, tokenizer, processor = load_model(args.model_id, args.use_4bit)
    t_values = np.linspace(0, 1, args.num_steps)

    # A. Interpolation curves
    all_curves = []
    for pair in tqdm(pairs, desc="Interpolating"):
        img_path = pair["images"]["typographic_clean"]
        if not Path(img_path).exists(): continue
        try:
            h_text = get_text_hidden(model, tokenizer, pair["text_prompt"], best_layer)
            h_image = get_image_hidden(model, processor, pair["neutral_carrier"], img_path, best_layer)

            curve = []
            for t in t_values:
                h = (1 - t) * h_image + t * h_text
                curve.append(cosine_sim(h.float().numpy(), ref_dir))

            all_curves.append({
                "id": pair["id"], "category": pair.get("category", ""),
                "scores": curve,
                "text_score": float(curve[-1]),
                "image_score": float(curve[0]),
            })
        except RuntimeError:
            clear_vram()

    # B. Binary search for boundary
    print(f"\nBinary search for refusal boundary...")
    boundaries = []
    for pair in tqdm(pairs[:30], desc="Binary search"):
        img_path = pair["images"]["typographic_clean"]
        if not Path(img_path).exists(): continue
        try:
            h_text = get_text_hidden(model, tokenizer, pair["text_prompt"], best_layer)
            h_image = get_image_hidden(model, processor, pair["neutral_carrier"], img_path, best_layer)

            lo, hi = 0.0, 1.0
            midpoint = (cosine_sim(h_image.float().numpy(), ref_dir) +
                        cosine_sim(h_text.float().numpy(), ref_dir)) / 2

            for _ in range(args.binary_search_steps):
                mid = (lo + hi) / 2
                h = (1 - mid) * h_image + mid * h_text
                score = cosine_sim(h.float().numpy(), ref_dir)
                if score < midpoint:
                    lo = mid
                else:
                    hi = mid

            boundaries.append({"id": pair["id"], "threshold": float((lo + hi) / 2)})
        except RuntimeError:
            clear_vram()

    # Aggregate
    if all_curves:
        matrix = np.array([c["scores"] for c in all_curves])
        mean_curve = matrix.mean(axis=0)
        std_curve = matrix.std(axis=0)

        coeffs = np.polyfit(t_values, mean_curve, 1)
        predicted = np.polyval(coeffs, t_values)
        r2 = 1 - np.sum((mean_curve - predicted)**2) / np.sum((mean_curve - mean_curve.mean())**2)

        print(f"\n  Mean curve: image={mean_curve[0]:.4f} → text={mean_curve[-1]:.4f}")
        print(f"  R² linearity: {r2:.4f}")
        print(f"  Slope: {coeffs[0]:.4f}")

    if boundaries:
        thresholds = [b["threshold"] for b in boundaries]
        print(f"\n  Boundary: mean={np.mean(thresholds):.3f} ± {np.std(thresholds):.3f}")
        print(f"  Requires {np.mean(thresholds):.0%} text-likeness to trigger refusal")

    # Save
    results = {
        "model_id": args.model_id, "best_layer": best_layer,
        "num_pairs": len(all_curves), "num_steps": args.num_steps,
        "t_values": t_values.tolist(),
        "mean_curve": mean_curve.tolist() if all_curves else [],
        "std_curve": std_curve.tolist() if all_curves else [],
        "r2": float(r2) if all_curves else 0,
        "slope": float(coeffs[0]) if all_curves else 0,
        "boundaries": boundaries,
        "mean_boundary": float(np.mean(thresholds)) if boundaries else 0,
        "per_pair": all_curves,
    }
    with open(out / f"interpolation_{model_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    if all_curves:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax = axes[0]
        for c in all_curves[:30]:
            ax.plot(t_values, c["scores"], color="gray", alpha=0.15, lw=0.5)
        ax.fill_between(t_values, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color="red")
        ax.plot(t_values, mean_curve, "r-o", ms=4, lw=2.5, label=f"Mean (n={len(all_curves)})")
        ax.plot(t_values, predicted, "b--", lw=1.5, label=f"Linear fit (R²={r2:.3f})")
        ax.set_xlabel("t (0=image, 1=text)", fontsize=12)
        ax.set_ylabel("Refusal Direction Activation", fontsize=12)
        ax.set_title("Interpolation: Image → Text", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if boundaries:
            ax.hist(thresholds, bins=15, color="#e74c3c", alpha=0.7, edgecolor="black")
            ax.axvline(np.mean(thresholds), color="blue", ls="--", lw=2,
                       label=f"Mean: {np.mean(thresholds):.3f}")
            ax.set_xlabel("Threshold t*", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title("Refusal Boundary Distribution", fontsize=14, fontweight="bold")
            ax.legend()

        plt.tight_layout()
        plt.savefig(Path(args.plot_dir) / f"interpolation_{model_name}.png", dpi=200)
        plt.close()
        print(f"  Saved plot")

    print(f"\nSaved: {out / f'interpolation_{model_name}.json'}")


if __name__ == "__main__":
    main()
