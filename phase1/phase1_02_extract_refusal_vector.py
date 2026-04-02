#!/usr/bin/env python3
"""
Phase 1 - Step 2: Extract Refusal Direction

Loads LLaVA backbone in text-only mode, runs 400 harmful + 400 benign prompts,
collects hidden states at every layer (last token), computes difference-in-means.

Fixes baked in:
  - Chat template: "USER: {text}\nASSISTANT:"
  - model.language_model() for text-only path
  - total_memory not total_mem
  - HF cache configurable

Usage:
    python phase1_02_extract_refusal_vector.py
    python phase1_02_extract_refusal_vector.py --use_4bit --model_id llava-hf/llava-1.5-7b-hf
"""

import os
os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", "hf_cache"))

import gc, json, argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig


def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name} ({mem:.1f} GB)")
        return "cuda"
    print("WARNING: No GPU")
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


def get_num_layers(model):
    backbone = model.language_model
    if hasattr(backbone, "model") and hasattr(backbone.model, "layers"):
        return len(backbone.model.layers)
    if hasattr(backbone, "layers"):
        return len(backbone.layers)
    raise AttributeError("Cannot find layers")


def collect_activations(model, tokenizer, prompts, num_layers):
    """Collect last-token hidden states at every layer for all prompts."""
    # activations[layer] = list of numpy vectors
    activations = {l: [] for l in range(num_layers + 1)}

    for prompt in tqdm(prompts, desc="Collecting"):
        text = f"USER: {prompt}\nASSISTANT:"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.language_model(**inputs, output_hidden_states=True)

        for l in range(min(num_layers + 1, len(outputs.hidden_states))):
            vec = outputs.hidden_states[l][0, -1, :].detach().cpu().float().numpy()
            activations[l].append(vec)

        del outputs, inputs
        clear_vram()

    return activations


def extract_refusal_direction(
    model_id="llava-hf/llava-1.5-7b-hf",
    data_dir="data/prompts",
    output_dir="outputs/vectors",
    use_4bit=True,
):
    device = get_device()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_name = model_id.replace("/", "_").replace("-", "_")

    # Load prompts
    data_path = Path(data_dir) / "prompt_data.json"
    with open(data_path) as f:
        data = json.load(f)

    harmful = data["harmful_train"]
    benign = data["benign_train"]
    print(f"harmful_train: {len(harmful)}")
    print(f"benign_train: {len(benign)}")

    # Load model
    model, tokenizer, processor = load_model(model_id, device, use_4bit)
    num_layers = get_num_layers(model)
    hidden_size = model.config.text_config.hidden_size
    print(f"Layers: {num_layers}, Hidden: {hidden_size}")

    # Collect activations
    print("\nCollecting harmful activations...")
    harmful_acts = collect_activations(model, tokenizer, harmful, num_layers)

    print("\nCollecting benign activations...")
    benign_acts = collect_activations(model, tokenizer, benign, num_layers)

    # Compute refusal directions (difference-in-means, normalized)
    refusal_directions = {}
    scores = {}

    for l in range(num_layers + 1):
        h_vecs = np.array(harmful_acts[l])
        b_vecs = np.array(benign_acts[l])

        mean_h = h_vecs.mean(axis=0)
        mean_b = b_vecs.mean(axis=0)

        direction = mean_h - mean_b
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm

        refusal_directions[l] = direction

        # Measure separation
        h_scores = [float(np.dot(v, direction) / (np.linalg.norm(v) + 1e-8)) for v in h_vecs]
        b_scores = [float(np.dot(v, direction) / (np.linalg.norm(v) + 1e-8)) for v in b_vecs]

        scores[l] = {
            "harmful_mean": float(np.mean(h_scores)),
            "benign_mean": float(np.mean(b_scores)),
            "gap": float(np.mean(h_scores) - np.mean(b_scores)),
        }

    # Find best layer
    best_layer = max(scores, key=lambda l: scores[l]["gap"])
    best_gap = scores[best_layer]["gap"]

    # Print results
    print(f"\n{'Layer':<6} {'Harmful':<10} {'Benign':<10} {'Gap':<10}")
    print("-" * 36)
    for l in range(num_layers + 1):
        s = scores[l]
        if l % 4 == 0 or l == num_layers:
            print(f"  {l:<6} {s['harmful_mean']:<10.4f} {s['benign_mean']:<10.4f} {s['gap']:<10.4f}")

    print(f"\nBest: layer {best_layer} (gap={best_gap:.4f})")

    # Save
    np.savez(
        out / f"refusal_directions_{model_name}.npz",
        **{f"layer_{l}": refusal_directions[l] for l in refusal_directions}
    )

    metadata = {
        "model_id": model_id,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "best_layer": best_layer,
        "best_gap": best_gap,
        "num_harmful": len(harmful),
        "num_benign": len(benign),
        "scores": {str(l): scores[l] for l in scores},
    }
    with open(out / f"metadata_{model_name}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved to {out}")
    return best_layer, best_gap


def main():
    ap = argparse.ArgumentParser(description="Extract refusal direction")
    ap.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--data_dir", default="data/prompts")
    ap.add_argument("--output_dir", default="outputs/vectors")
    ap.add_argument("--use_4bit", action="store_true")
    args = ap.parse_args()
    extract_refusal_direction(**vars(args))
    print("\nNext: python phase1_03_validate_refusal_vector.py --use_4bit")


if __name__ == "__main__":
    main()
