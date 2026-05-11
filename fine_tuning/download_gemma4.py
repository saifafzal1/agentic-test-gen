"""
Download google/gemma-4-E4B-it to fine_tuning/gemma4-base-model/
Run: python -u fine_tuning/download_gemma4.py
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

MODEL_ID   = "google/gemma-4-E4B-it"
OUTPUT_DIR = Path(__file__).parent / "gemma4-base-model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📥 Downloading {MODEL_ID} → {OUTPUT_DIR}")
print("   This is ~8 GB — may take 15–30 min depending on connection...\n")

local_dir = snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(OUTPUT_DIR),
    token=HF_TOKEN,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
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
    print(f"   Hidden size  : {cfg.get('text_config', {}).get('hidden_size', cfg.get('hidden_size', '?'))}")
    print(f"   Vocab size   : {cfg.get('text_config', {}).get('vocab_size', cfg.get('vocab_size', '?'))}")
