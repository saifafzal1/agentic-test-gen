"""
Download microsoft/Phi-3-mini-4k-instruct to fine_tuning/phi3-base-model/
Run: python -u fine_tuning/download_phi3.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from huggingface_hub import login as hf_login, snapshot_download
import torch

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    hf_login(token=HF_TOKEN, add_to_git_credential=False)
    print("✅ HuggingFace login successful")
else:
    print("⚠️  HF_TOKEN not set — may fail on gated models")

MODEL_ID   = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = Path(__file__).parent / "phi3-base-model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📥 Downloading {MODEL_ID} → {OUTPUT_DIR}")
print("   This is ~7.6 GB — may take 15–25 min depending on connection...\n")

local_dir = snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(OUTPUT_DIR),
    token=HF_TOKEN,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*", "onnx/*"],
)

print(f"\n✅ Download complete → {local_dir}")

# Quick sanity check
import json
config_path = OUTPUT_DIR / "config.json"
if config_path.exists():
    with open(config_path) as f:
        cfg = json.load(f)
    print(f"   Architecture : {cfg.get('architectures', ['?'])[0]}")
    print(f"   Model type   : {cfg.get('model_type', '?')}")
    print(f"   Hidden size  : {cfg.get('hidden_size', '?')}")
    print(f"   Num layers   : {cfg.get('num_hidden_layers', '?')}")
    print(f"   Max position : {cfg.get('max_position_embeddings', '?')}")
