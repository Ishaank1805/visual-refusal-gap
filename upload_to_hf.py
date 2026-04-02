#!/usr/bin/env python3
"""
Upload project data to HuggingFace: Ishaank18/visual-refusal-gap-new

Checks both ./data, ./outputs AND /scratch/ishaan.karan/ paths.
Uploads whatever exists.

Usage:
    cd ~/pipeline
    python upload_to_hf.py
"""

from huggingface_hub import HfApi, login
import os

REPO = "Ishaank18/visual-refusal-gap-new"

token = os.environ.get("HF_TOKEN", None)
if token:
    login(token=token)

api = HfApi()
try:
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
except:
    pass

# Check both relative and absolute paths, prefer relative
folders = [
    # (local_path, repo_path)
    ("./data", "data"),
    ("./outputs", "outputs"),
    ("/scratch/ishaan.karan/data", "data"),
    ("/scratch/ishaan.karan/outputs", "outputs"),
]

uploaded = set()
for local, repo_path in folders:
    if not os.path.exists(local):
        continue
    if repo_path in uploaded:
        print(f"SKIP: {local} (already uploaded {repo_path}/)")
        continue
    n = sum(1 for _, _, fs in os.walk(local) for f in fs)
    print(f"Uploading {local}/ → {repo_path}/ ({n} files)...")
    api.upload_folder(
        folder_path=local,
        path_in_repo=repo_path,
        repo_id=REPO,
        repo_type="dataset",
    )
    uploaded.add(repo_path)
    print(f"  Done.")

print(f"\nhttps://huggingface.co/datasets/{REPO}")
