"""
Phase 5 — QLoRA Fine-Tuning: Phi-3-mini-4k-instruct (Framework-Isolated)
=========================================================================
Fine-tunes microsoft/Phi-3-mini-4k-instruct on framework-isolated datasets
using QLoRA (LoRA adapters) via PEFT + TRL on Apple Silicon (MPS, bf16).

Two separate training runs — one per framework:
  Run 3 (Cypress)    : python fine_tuning/finetune_phi3.py --framework cypress
  Run 4 (Playwright) : python fine_tuning/finetune_phi3.py --framework playwright

Hardware  : Mac A — Apple M4 Pro, 48 GB RAM (MPS backend, bf16)
Dataset   : data/cypress_dataset.jsonl  OR  data/playwright_dataset.jsonl
Output    : fine_tuning/phi3-cypress/  OR  fine_tuning/phi3-playwright/

Phi-3 chat template (ChatML-style):
  <|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n{response}<|end|>

Lessons from Gemma runs:
  - bf16=True (fp16 GradScaler is CUDA-only — silently skips all optimizer steps on MPS)
  - DataCollatorForLanguageModeling(mlm=False) — avoids TRL completion-only mask bug
  - python -u for unbuffered stdout
  - load_best_model_at_end=False (eval_loss can be NaN on MPS)

Run:
  cd ~/Documents/Dissertation/agentic-test-gen
  source .venv/bin/activate
  python -u fine_tuning/finetune_phi3.py --framework cypress
  python -u fine_tuning/finetune_phi3.py --framework playwright
"""

import os, json, math, argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Args ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--framework", choices=["cypress", "playwright"], required=True,
                    help="Framework to fine-tune on (cypress or playwright)")
args = parser.parse_args()
FRAMEWORK = args.framework

# ── HuggingFace auth ────────────────────────────────────────────────────────
from huggingface_hub import login as hf_login
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    hf_login(token=HF_TOKEN, add_to_git_credential=False)
    print("✅ HuggingFace login successful")
else:
    print("⚠️  HF_TOKEN not set — may fail on gated models")

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
import wandb

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODEL_ID   = str(Path(__file__).parent / "phi3-base-model")   # local path — avoids re-download via HF hub
DATA_FILE  = BASE_DIR / "data" / f"{FRAMEWORK}_dataset.jsonl"
OUTPUT_DIR = Path(__file__).parent / f"phi3-{FRAMEWORK}"
WANDB_KEY  = os.environ.get("WANDB_API_KEY", "")

# Training hyperparameters
EPOCHS        = 3
BATCH_SIZE    = 1       # MPS: keep at 1
GRAD_ACCUM    = 8       # effective batch = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LEN   = 2048    # Phi-3-mini supports up to 4k; 2k fits our dataset
WARMUP_STEPS  = 10
SAVE_STEPS    = 50
EVAL_STEPS    = 50
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05

# ── Device ───────────────────────────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️  Device    : {device}")
print(f"🎯 Framework  : {FRAMEWORK}")
print(f"📦 Model      : {MODEL_ID}")

# ── WandB ────────────────────────────────────────────────────────────────────
if WANDB_KEY:
    wandb.login(key=WANDB_KEY)
    wandb.init(
        project="dissertation-phi3-finetune",
        name=f"phi3-mini-{FRAMEWORK}-qlora"
    )
else:
    os.environ["WANDB_DISABLED"] = "true"
    print("⚠️  WANDB_API_KEY not set — metrics won't be tracked online")

# ── Load & split dataset ─────────────────────────────────────────────────────
print(f"\n📂 Loading {DATA_FILE.name}...")
records = []
with open(DATA_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

n         = len(records)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)
train_raw = records[:train_end]
val_raw   = records[train_end:val_end]
print(f"   Total : {n} | Train : {len(train_raw)} | Val : {len(val_raw)} | Test : {n - val_end}")

# ── Format as instruction-tuning pairs ──────────────────────────────────────
# Phi-3 ChatML-style template:
#   <|system|>\n...<|end|>\n<|user|>\n...<|end|>\n<|assistant|>\n...<|end|>

SYSTEM_CY = (
    "You are a senior QA automation engineer specialising in Cypress. "
    "Given a Jira user story, generate a complete, production-quality Cypress test script. "
    "Return ONLY the raw Cypress JavaScript — no markdown fences, no explanations. "
    "Include describe(), beforeEach(), and at least 2 it() blocks (happy path + edge case)."
)

SYSTEM_PW = (
    "You are a senior QA automation engineer specialising in Playwright. "
    "Given a Jira user story, generate a complete, production-quality Playwright TypeScript test script. "
    "Return ONLY the raw Playwright TypeScript — no markdown fences, no explanations. "
    "Use test.describe(), test.beforeEach(), at least 2 test() blocks. "
    "Use getByRole(), getByLabel(), or getByText() locators — never page.locator()."
)

SYSTEM = SYSTEM_CY if FRAMEWORK == "cypress" else SYSTEM_PW

def format_sample(rec: dict) -> dict:
    user = (
        f'User story: "{rec["user_story"]}"\n'
        f'Category: {rec["category"]}\n'
        f'Complexity: {rec.get("complexity", "medium")}\n'
        f'Return the {FRAMEWORK.capitalize()} script only.'
    )
    script = rec["script"]
    # Phi-3 ChatML template
    text = (
        f"<|system|>\n{SYSTEM}<|end|>\n"
        f"<|user|>\n{user}<|end|>\n"
        f"<|assistant|>\n{script}<|end|>"
    )
    return {"text": text}

train_data = Dataset.from_list([format_sample(r) for r in train_raw])
val_data   = Dataset.from_list([format_sample(r) for r in val_raw])
print(f"   Sample text length: {len(train_data[0]['text'])} chars")

# ── Tokeniser ────────────────────────────────────────────────────────────────
print("\n📥 Loading tokeniser...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=False,   # native Phi3 in transformers 5.x — no custom code needed
    token=HF_TOKEN,
)
# Phi-3 uses eos_token as pad; padding_side right for causal LM
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Model — bf16 on MPS ──────────────────────────────────────────────────────
# Bulk-load to CPU first (zero-copy move to unified MPS memory), same pattern as Gemma4.
# trust_remote_code=False: transformers 5.x has native Phi3ForCausalLM.
# attn_implementation='eager': flash_attn not available on MPS.
print("📥 Loading Phi-3-mini-4k-instruct in bf16 → bulk CPU then move to MPS...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=False,
    token=HF_TOKEN,
    attn_implementation="eager",   # flash_attn not available on MPS
)
print(f"   Moving model to {device}...")
model = model.to(device)
print("   ✅ Loaded and on MPS")
model.config.use_cache = False

# ── LoRA config ──────────────────────────────────────────────────────────────
# Native transformers 5.x Phi3 uses qkv_proj (combined) and gate_up_proj (combined).
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Training config ──────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sft_config = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    eval_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=False,   # avoid NaN eval_loss blocking best-model selection
    bf16=True,                       # bf16 on MPS — no GradScaler, stable gradients
    fp16=False,
    optim="adamw_torch",
    report_to="wandb" if WANDB_KEY else "none",
    run_name=f"phi3-mini-{FRAMEWORK}-qlora",
    max_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    dataloader_pin_memory=False,    # required for MPS
)

# Full LM collator: loss on ALL tokens (avoids TRL completion-only mask bug)
lm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ── Trainer ──────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_data,
    eval_dataset=val_data,
    processing_class=tokenizer,
    data_collator=lm_collator,
)

# ── Train ────────────────────────────────────────────────────────────────────
train_steps = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM) * EPOCHS
print(f"\n🚀 Starting Phi-3-mini fine-tuning ({FRAMEWORK.upper()})...")
print(f"   Epochs          : {EPOCHS}")
print(f"   Effective batch : {BATCH_SIZE * GRAD_ACCUM}")
print(f"   Learning rate   : {LEARNING_RATE}")
print(f"   LoRA rank       : {LORA_R}")
print(f"   Max seq length  : {MAX_SEQ_LEN}")
print(f"   Train steps     : {train_steps}\n")

train_result = trainer.train()

# ── Save ─────────────────────────────────────────────────────────────────────
print("\n💾 Saving model & tokeniser...")
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

metrics = train_result.metrics
metrics["train_samples"] = len(train_data)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

print(f"\n✅ Fine-tuning complete! ({FRAMEWORK.upper()})")
print(f"   Model saved → {OUTPUT_DIR}")
print(f"   Train loss  : {metrics.get('train_loss', 'N/A'):.4f}")
print(f"   Train time  : {metrics.get('train_runtime', 0)/3600:.2f} hrs")

if WANDB_KEY:
    wandb.finish()
