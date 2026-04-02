#!/usr/bin/env python3
"""
Phase 4 - Step 1: Extract Other Alignment Directions

Uses direction_prompts.json (from Phase 2 Step 1) to extract:
  - honesty direction (truthful vs fabrication)
  - sycophancy direction (disagree vs agree with wrong claims)
  - privacy direction (refuse vs reveal personal info)

Same method as Phase 1: difference-in-means on LLM backbone activations.
Then measures visual gap for each direction.

GPU NEEDED.

Usage:
    python phase4_01_extract_directions.py --use_4bit
"""

import os
os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", os.path.expanduser("~/.cache/huggingface")))

import gc, json, argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig


def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()

def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name} ({mem:.1f} GB)")
        return "cuda"
    return "cpu"

def load_model(model_id, device="cuda", use_4bit=False):
    print(f"Loading {model_id} (4-bit={use_4bit})...")
    clear_vram()
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    ) if use_4bit else None
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, quantization_config=quant,
        device_map="auto" if device == "cuda" else None, low_cpu_mem_usage=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    if torch.cuda.is_available():
        print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, processor.tokenizer, processor

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0

def get_num_layers(model):
    backbone = model.language_model
    if hasattr(backbone, "model") and hasattr(backbone.model, "layers"):
        return len(backbone.model.layers)
    return len(backbone.layers)


def collect_text_acts(model, tokenizer, prompts, num_layers):
    """Collect last-token hidden states at every layer."""
    acts = {l: [] for l in range(num_layers + 1)}
    for prompt in tqdm(prompts, desc="  Collecting"):
        text = f"USER: {prompt}\nASSISTANT:"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.language_model(**inputs, output_hidden_states=True)
        for l in range(min(num_layers + 1, len(outputs.hidden_states))):
            acts[l].append(outputs.hidden_states[l][0, -1, :].detach().cpu().float().numpy())
        del outputs, inputs; clear_vram()
    return acts


def collect_image_acts(model, processor, pairs, target_layer):
    """Collect image hidden states at target layer."""
    scores = []
    for pair in tqdm(pairs, desc="  Image acts"):
        img_path = pair.get("images", {}).get("typographic_clean", "")
        if not img_path or not Path(img_path).exists():
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            prompt = f"USER: <image>\n{pair['neutral_carrier']}\nASSISTANT:"
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states if hasattr(outputs, "hidden_states") else []
            if target_layer < len(hidden):
                vec = hidden[target_layer][0, -1, :].detach().cpu().float().numpy()
                scores.append({"id": pair["id"], "vec": vec})
            del outputs, inputs, image; clear_vram()
        except RuntimeError:
            clear_vram()
    return scores


def main():
    ap = argparse.ArgumentParser(description="Extract alignment directions + measure visual gap")
    ap.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--direction_file", default="./data/visual_hazards_v2/direction_prompts.json")
    ap.add_argument("--vector_dir", default="./outputs/vectors")
    ap.add_argument("--output_dir", default="./outputs/generalization")
    ap.add_argument("--use_4bit", action="store_true")
    args = ap.parse_args()

    device = get_device()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    model_name = args.model_id.replace("/", "_").replace("-", "_")

    # Load direction prompts
    with open(args.direction_file) as f:
        dir_data = json.load(f)

    directions_info = dir_data["directions"]
    dir_image_pairs = dir_data.get("direction_image_pairs", [])

    # Load refusal direction for comparison
    ref_path = Path(args.vector_dir) / f"refusal_directions_{model_name}.npz"
    if not ref_path.exists():
        candidates = list(Path(args.vector_dir).glob("refusal_directions_*.npz"))
        ref_path = candidates[0] if candidates else None
    refusal_raw = np.load(ref_path)
    refusal_dirs = {int(k.replace("layer_", "")): v for k, v in refusal_raw.items()}

    meta_path = Path(args.vector_dir) / f"metadata_{model_name}.json"
    if not meta_path.exists():
        candidates = list(Path(args.vector_dir).glob("metadata_*.json"))
        meta_path = candidates[0] if candidates else None
    with open(meta_path) as f:
        meta = json.load(f)
    best_layer = meta["best_layer"]
    num_layers = meta["num_layers"]

    print("=" * 60)
    print("PHASE 4: EXTRACT ALIGNMENT DIRECTIONS")
    print("=" * 60)
    print(f"Best refusal layer: {best_layer}")
    print(f"Directions: {list(directions_info.keys())}")

    # Load model
    model, tokenizer, processor = load_model(args.model_id, device, args.use_4bit)

    all_results = {"refusal": {"best_layer": best_layer, "best_gap": meta["best_gap"]}}

    for dir_name, dir_info in directions_info.items():
        print(f"\n{'='*60}")
        print(f"DIRECTION: {dir_name}")
        print(f"{'='*60}")

        pos_prompts = dir_info["positive_prompts"]
        neg_prompts = dir_info["negative_prompts"]
        print(f"  Positive: {len(pos_prompts)}, Negative: {len(neg_prompts)}")

        # Extract direction via difference-in-means
        print(f"\n  Extracting direction (text-only)...")
        pos_acts = collect_text_acts(model, tokenizer, pos_prompts, num_layers)
        neg_acts = collect_text_acts(model, tokenizer, neg_prompts, num_layers)

        direction_vecs = {}
        scores = {}
        for l in range(num_layers + 1):
            p_vecs = np.array(pos_acts[l])
            n_vecs = np.array(neg_acts[l])
            d = p_vecs.mean(axis=0) - n_vecs.mean(axis=0)
            norm = np.linalg.norm(d)
            if norm > 1e-8:
                d = d / norm
            direction_vecs[l] = d

            p_scores = [cosine_sim(v, d) for v in p_vecs]
            n_scores = [cosine_sim(v, d) for v in n_vecs]
            scores[l] = {
                "pos_mean": float(np.mean(p_scores)),
                "neg_mean": float(np.mean(n_scores)),
                "gap": float(np.mean(p_scores) - np.mean(n_scores)),
            }

        best_dir_layer = max(scores, key=lambda l: scores[l]["gap"])
        best_dir_gap = scores[best_dir_layer]["gap"]
        print(f"  Best layer: {best_dir_layer} (gap={best_dir_gap:.4f})")

        # Measure visual gap using direction image pairs
        dir_pairs = [p for p in dir_image_pairs if p.get("direction") == dir_name]
        pos_pairs = [p for p in dir_pairs if p.get("polarity") == "positive"]
        neg_pairs = [p for p in dir_pairs if p.get("polarity") == "negative"]

        visual_gap = 0.0
        if pos_pairs:
            print(f"\n  Measuring visual gap ({len(pos_pairs)} pos pairs)...")
            dir_vec = direction_vecs[best_dir_layer]

            # Text scores for positive prompts
            text_scores = []
            for pair in pos_pairs:
                text = f"USER: {pair['text_prompt']}\nASSISTANT:"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.language_model(**inputs, output_hidden_states=True)
                vec = outputs.hidden_states[best_dir_layer][0, -1, :].detach().cpu().float().numpy()
                text_scores.append(cosine_sim(vec, dir_vec))
                del outputs, inputs; clear_vram()

            # Image scores
            img_data = collect_image_acts(model, processor, pos_pairs, best_dir_layer)
            img_scores = [cosine_sim(d["vec"], dir_vec) for d in img_data]

            if text_scores and img_scores:
                mean_text = float(np.mean(text_scores))
                mean_img = float(np.mean(img_scores))
                visual_gap = mean_text - mean_img
                print(f"  Text score: {mean_text:.4f}")
                print(f"  Image score: {mean_img:.4f}")
                print(f"  Visual gap: {visual_gap:.4f}")

        all_results[dir_name] = {
            "best_layer": int(best_dir_layer),
            "best_gap": float(best_dir_gap),
            "visual_gap": float(visual_gap),
            "num_positive": len(pos_prompts),
            "num_negative": len(neg_prompts),
            "scores": {str(l): scores[l] for l in scores if l % 4 == 0 or l == best_dir_layer},
        }

        # Save direction vectors
        np.savez(
            out / f"direction_{dir_name}_{model_name}.npz",
            **{f"layer_{l}": direction_vecs[l] for l in direction_vecs}
        )

    # Summary
    print("\n" + "=" * 60)
    print("DIRECTION SELECTIVITY SUMMARY")
    print("=" * 60)
    print(f"\n  {'Direction':<15} {'Text Gap':<12} {'Visual Gap':<12} {'Lost by projector?'}")
    print(f"  {'-'*55}")
    for name, res in all_results.items():
        vg = res.get("visual_gap", res.get("best_gap", 0))
        tg = res.get("best_gap", 0)
        lost = "YES" if vg > 0.05 else "PARTIAL" if vg > 0.02 else "NO"
        print(f"  {name:<15} {tg:<12.4f} {vg:<12.4f} {lost}")

    # Save
    path = out / f"direction_selectivity_{model_name}.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {path}")
    print(f"Next: python phase4_02_cross_architecture.py --use_4bit")


if __name__ == "__main__":
    main()
