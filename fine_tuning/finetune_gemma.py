"""
Phase 4 — QLoRA Fine-Tuning
==============================
Fine-tunes Gemma 3:4B (google/gemma-3-4b-it) on the 237-pair dataset
using QLoRA (4-bit quantisation + LoRA adapters) via PEFT + TRL.

Hardware  : Mac A — Apple M4 Pro, 48 GB RAM (MPS backend)
Dataset   : data/dataset_280.json  (train 70% / val 15% / test 15%)
Output    : fine_tuning/gemma-finetuned/

Run:
  cd ~/Documents/Dissertation/agentic-test-gen
  source .venv/bin/activate
  huggingface-cli login          # needed to download Gemma
  python fine_tuning/finetune_gemma.py
"""

import os, json, math
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Authenticate with HuggingFace (required for gated Gemma model)
from huggingface_hub import login as hf_login
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    hf_login(token=HF_TOKEN, add_to_git_credential=False)
    print(f"✅ HuggingFace login successful")
else:
    print("⚠️  HF_TOKEN not set — may fail on gated models")

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import wandb

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_ID    = "google/gemma-3-4b-it"
DATA_FILE   = Path(__file__).parent.parent / "data" / "dataset_280.json"
OUTPUT_DIR  = Path(__file__).parent / "gemma-finetuned"
WANDB_KEY   = os.environ.get("WANDB_API_KEY", "")

# Training hyperparameters
EPOCHS          = 3
BATCH_SIZE      = 1       # keep low for MPS memory
GRAD_ACCUM      = 8       # effective batch = 8
LEARNING_RATE   = 2e-4
MAX_SEQ_LEN     = 1024
WARMUP_STEPS    = 10
SAVE_STEPS      = 50
EVAL_STEPS      = 50
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05

# ── Device ─────────────────────────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️  Device : {device}")
print(f"💾  RAM    : {torch.mps.current_allocated_memory()/1e9:.1f} GB allocated" if device=="mps" else "")

# ── WandB ──────────────────────────────────────────────────────────────────
if WANDB_KEY:
    wandb.login(key=WANDB_KEY)
    wandb.init(project="dissertation-gemma-finetune", name="gemma3-4b-qlora-v1")
else:
    os.environ["WANDB_DISABLED"] = "true"
    print("⚠️  WANDB_API_KEY not set — training metrics won't be tracked online")

# ── Load & split dataset ────────────────────────────────────────────────────
print("\n📂 Loading dataset...")
with open(DATA_FILE) as f:
    raw = json.load(f)

n         = len(raw)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)
train_raw = raw[:train_end]
val_raw   = raw[train_end:val_end]
print(f"   Train : {len(train_raw)} | Val : {len(val_raw)} | Total : {n}")

# ── Format as instruction-tuning pairs ─────────────────────────────────────
SYSTEM = """You are a senior QA automation engineer. Given a Jira user story, \
generate test scripts for Cypress AND Playwright. \
Return ONLY a raw JSON object with keys cypress_script and playwright_script."""

def format_sample(sample: dict) -> dict:
    user = f'User story: "{sample["user_story"]}"\nCategory: {sample["category"]}\nReturn JSON only.'
    assistant = json.dumps({
        "cypress_script":    sample["cypress_script"],
        "playwright_script": sample["playwright_script"]
    })
    # Gemma chat template format
    text = (
        f"<start_of_turn>system\n{SYSTEM}<end_of_turn>\n"
        f"<start_of_turn>user\n{user}<end_of_turn>\n"
        f"<start_of_turn>model\n{assistant}<end_of_turn>"
    )
    return {"text": text}

train_data = Dataset.from_list([format_sample(s) for s in train_raw])
val_data   = Dataset.from_list([format_sample(s) for s in val_raw])
print(f"   Sample text length: {len(train_data[0]['text'])} chars")

# ── Tokeniser ───────────────────────────────────────────────────────────────
print("\n📥 Loading tokeniser...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Model (4-bit QLoRA via bitsandbytes — fallback to fp16 on MPS) ──────────
print("📥 Loading base model...")

# Note: bitsandbytes 4-bit quant not fully supported on MPS yet
# We use fp16 on MPS which still fits in 48GB RAM for 4B model
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    print("   ✅ Loaded in 4-bit (QLoRA)")
except Exception as e:
    print(f"   ⚠️  4-bit failed ({e}) — loading in fp16")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )

model.config.use_cache = False

# ── LoRA config ─────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Training args ────────────────────────────────────────────────────────────
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
    evaluation_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,
    optim="adamw_torch",
    report_to="wandb" if WANDB_KEY else "none",
    run_name="gemma3-4b-qlora-v1",
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    dataloader_pin_memory=False,   # required for MPS
)

# ── Trainer ──────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_data,
    eval_dataset=val_data,
    processing_class=tokenizer,
)

# ── Train ────────────────────────────────────────────────────────────────────
print(f"\n🚀 Starting fine-tuning...")
print(f"   Epochs          : {EPOCHS}")
print(f"   Effective batch : {BATCH_SIZE * GRAD_ACCUM}")
print(f"   Learning rate   : {LEARNING_RATE}")
print(f"   LoRA rank       : {LORA_R}")
print(f"   Max seq length  : {MAX_SEQ_LEN}")
print(f"   Train steps     : {math.ceil(len(train_data)/BATCH_SIZE/GRAD_ACCUM)*EPOCHS}\n")

train_result = trainer.train()

# ── Save ─────────────────────────────────────────────────────────────────────
print("\n💾 Saving model & tokeniser...")
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

# Save training metrics
metrics = train_result.metrics
metrics["train_samples"] = len(train_data)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

print(f"\n✅ Fine-tuning complete!")
print(f"   Model saved → {OUTPUT_DIR}")
print(f"   Train loss  : {metrics.get('train_loss', 'N/A'):.4f}")
print(f"   Train time  : {metrics.get('train_runtime', 0)/60:.1f} min")

if WANDB_KEY:
    wandb.finish()
