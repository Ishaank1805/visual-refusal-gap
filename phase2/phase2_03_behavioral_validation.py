#!/usr/bin/env python3
"""
Phase 2 - Step 3: Behavioral Validation

Generate actual responses for text vs image inputs.
Saves all prompts + responses for Gemini judging in Step 4.

Fixes:
  - model.generate() not model.language_model.generate()
  - USER: {text}\nASSISTANT: for text
  - USER: <image>\n{text}\nASSISTANT: for image
  - total_memory not total_mem

Usage:
    python phase2_03_behavioral_validation.py --use_4bit
    python phase2_03_behavioral_validation.py --use_4bit --max_pairs 200
"""

import os
os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", "hf_cache"))

import gc, json, argparse
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


def gen_text_response(model, tokenizer, text, max_new_tokens=150):
    """Text-only with chat template. Uses model.generate()."""
    prompt = f"USER: {text}\nASSISTANT:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    del outputs, inputs; clear_vram()
    return response.strip()


def gen_image_response(model, processor, text, image_path, max_new_tokens=150):
    """Image+text with chat template and <image> token. Uses model.generate()."""
    image = Image.open(image_path).convert("RGB")
    prompt = f"USER: <image>\n{text}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    del outputs, inputs, image; clear_vram()
    return response.strip()


def run_behavioral(
    model_id="llava-hf/llava-1.5-7b-hf",
    hazards_dir="data/visual_hazards_v2",
    output_dir="outputs/gap_analysis",
    use_4bit=True,
    image_style="clean",
    max_new_tokens=150,
    max_pairs=0,
):
    device = get_device()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    model_name = model_id.replace("/", "_").replace("-", "_")

    print("\n" + "=" * 60)
    print("PHASE 2: BEHAVIORAL VALIDATION")
    print("=" * 60)

    # Load dataset
    meta_path = Path(hazards_dir) / "visual_hazards_metadata.json"
    with open(meta_path) as f:
        dataset = json.load(f)

    harmful = dataset["harmful_pairs"]
    benign = dataset["benign_controls"]
    image_key = f"typographic_{image_style}"

    harmful = [p for p in harmful if image_key in p.get("images", {})]
    benign = [b for b in benign if image_key in b.get("images", {})]

    if max_pairs > 0:
        harmful = harmful[:max_pairs]
        benign = benign[:max(max_pairs // 10, 10)]

    print(f"Pairs: {len(harmful)} harmful, {len(benign)} benign")

    # Load model
    model, tokenizer, processor = load_model(model_id, device, use_4bit)

    # ---- Harmful ----
    print(f"\nTesting {len(harmful)} harmful pairs...")
    harmful_results = []

    for pair in tqdm(harmful, desc="Harmful"):
        r = {"id": pair["id"], "category": pair.get("category", "unknown"),
             "text_prompt": pair["text_prompt"]}
        img_path = pair["images"][image_key]

        # Text
        try:
            resp = gen_text_response(model, tokenizer, pair["text_prompt"], max_new_tokens)
            r["text_response"] = resp[:500]
        except RuntimeError:
            clear_vram(); r["text_response"] = "[OOM]"

        # Image
        if Path(img_path).exists():
            try:
                resp = gen_image_response(model, processor, pair["neutral_carrier"], img_path, max_new_tokens)
                r["image_response"] = resp[:500]
            except RuntimeError:
                clear_vram(); r["image_response"] = "[OOM]"
        else:
            r["image_response"] = "[MISSING]"

        harmful_results.append(r)

    # ---- Benign ----
    print(f"\nTesting {len(benign)} benign controls...")
    benign_results = []

    for ctrl in tqdm(benign, desc="Benign"):
        r = {"id": ctrl["id"], "text_prompt": ctrl["text_prompt"]}
        img_path = ctrl["images"][image_key]

        try:
            resp = gen_text_response(model, tokenizer, ctrl["text_prompt"], max_new_tokens)
            r["text_response"] = resp[:500]
        except RuntimeError:
            clear_vram(); r["text_response"] = "[OOM]"

        if Path(img_path).exists():
            try:
                resp = gen_image_response(model, processor, ctrl["neutral_carrier"], img_path, max_new_tokens)
                r["image_response"] = resp[:500]
            except RuntimeError:
                clear_vram(); r["image_response"] = "[OOM]"
        else:
            r["image_response"] = "[MISSING]"

        benign_results.append(r)

    # Save raw responses (judging happens in step 4)
    all_results = {
        "model_id": model_id,
        "image_style": image_style,
        "num_harmful": len(harmful_results),
        "num_benign": len(benign_results),
        "harmful_results": harmful_results,
        "benign_results": benign_results,
    }

    fname = out / f"behavioral_responses_{model_name}.json"
    with open(fname, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResponses saved: {fname}")
    print(f"Next: python phase2_04_gemini_judge.py")


def main():
    ap = argparse.ArgumentParser(description="Behavioral validation (generate responses)")
    ap.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--hazards_dir", default="data/visual_hazards_v2")
    ap.add_argument("--output_dir", default="outputs/gap_analysis")
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--image_style", default="clean")
    ap.add_argument("--max_new_tokens", type=int, default=150)
    ap.add_argument("--max_pairs", type=int, default=0)
    args = ap.parse_args()
    run_behavioral(**vars(args))

if __name__ == "__main__":
    main()
