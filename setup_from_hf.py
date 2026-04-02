#!/usr/bin/env python3
"""
Download project data from HuggingFace into current directory.

Creates:
    ./data/prompts/
    ./data/visual_hazards_v2/
    ./outputs/vectors/
    ./outputs/logs/
    ./outputs/plots/
    ./outputs/gap_analysis/
    ./outputs/mechanism/

Usage:
    cd ~/pipeline
    python setup_from_hf.py
"""

from huggingface_hub import snapshot_download, login
import os

REPO = "Ishaank18/visual-refusal-gap-new"

token = os.environ.get("HF_TOKEN", None)
if token:
    login(token=token)

print(f"Downloading {REPO} into current directory...")
snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    local_dir=".",
)
print("Done.")
print("\nFiles:")
for root, dirs, files in os.walk("."):
    # skip hidden dirs and hf cache
    dirs[:] = [d for d in dirs if not d.startswith(".")]
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path) / 1e6
        if size > 0.01:
            print(f"  {path} ({size:.1f} MB)")
