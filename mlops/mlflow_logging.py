"""
MLOps Week 1 — Retroactive MLflow Logging
==========================================
Logs two experiments to a local MLflow tracking server:

  1. Phase5-FineTuning  — 4 QLoRA runs (Gemma4 + Phi-3 × Cypress + Playwright)
  2. Phase3-Baselines   — Baseline generation runs (Claude Haiku, GPT-4o-mini,
                          Gemini 3.1 Flash Lite, Gemini Flash) × 2 frameworks

Run:
  cd ~/dissertation-project
  python3 mlops/mlflow_logging.py
  mlflow ui --backend-store-uri mlruns/   # open http://127.0.0.1:5000
"""

import json
import os
from pathlib import Path

import mlflow

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent          # ~/dissertation-project
MLRUNS_DIR   = BASE_DIR / "mlruns"
RESULTS_DIR  = BASE_DIR / "baselines" / "results"

mlflow.set_tracking_uri(str(MLRUNS_DIR))
print(f"📊 MLflow tracking URI : {MLRUNS_DIR}")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Phase5-FineTuning
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🏋️  Logging Phase5-FineTuning experiment...")
mlflow.set_experiment("Phase5-FineTuning")

FINETUNING_RUNS = [
    {
        "run_name"          : "gemma4-cypress-qlora",
        "model"             : "google/gemma-3-4b-it",
        "framework"         : "cypress",
        "train_loss"        : 0.4925,
        "train_runtime_s"   : 13594.4,
        "train_runtime_hrs" : 13594.4 / 3600,
        "train_samples"     : 195,
        "total_flos"        : 1.290e+16,
        "lora_r"            : 16,
        "lora_alpha"        : 32,
        "lora_dropout"      : 0.05,
        "target_modules"    : "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "epochs"            : 3,
        "batch_size"        : 1,
        "grad_accum"        : 8,
        "effective_batch"   : 8,
        "learning_rate"     : 2e-4,
        "max_seq_len"       : 2048,
        "hardware"          : "Apple M4 Pro MPS bf16",
        "hf_adapter_repo"   : "saifafzal1/gemma4-E4B-cypress-qlora",
    },
    {
        "run_name"          : "gemma4-playwright-qlora",
        "model"             : "google/gemma-3-4b-it",
        "framework"         : "playwright",
        "train_loss"        : 0.5380,
        "train_runtime_s"   : 20568.1,
        "train_runtime_hrs" : 20568.1 / 3600,
        "train_samples"     : 195,
        "total_flos"        : 1.465e+16,
        "lora_r"            : 16,
        "lora_alpha"        : 32,
        "lora_dropout"      : 0.05,
        "target_modules"    : "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "epochs"            : 3,
        "batch_size"        : 1,
        "grad_accum"        : 8,
        "effective_batch"   : 8,
        "learning_rate"     : 2e-4,
        "max_seq_len"       : 2048,
        "hardware"          : "Apple M4 Pro MPS bf16",
        "hf_adapter_repo"   : "saifafzal1/gemma4-E4B-playwright-qlora",
    },
    {
        "run_name"          : "phi3-cypress-qlora",
        "model"             : "microsoft/Phi-3-mini-4k-instruct",
        "framework"         : "cypress",
        "train_loss"        : 0.3808,
        "train_runtime_s"   : 2872.4,
        "train_runtime_hrs" : 2872.4 / 3600,
        "train_samples"     : 195,
        "total_flos"        : 1.170e+16,
        "lora_r"            : 16,
        "lora_alpha"        : 32,
        "lora_dropout"      : 0.05,
        "target_modules"    : "qkv_proj,o_proj,gate_up_proj,down_proj",
        "epochs"            : 3,
        "batch_size"        : 1,
        "grad_accum"        : 8,
        "effective_batch"   : 8,
        "learning_rate"     : 2e-4,
        "max_seq_len"       : 2048,
        "hardware"          : "Apple M4 Pro MPS bf16",
        "hf_adapter_repo"   : "saifafzal1/phi3-mini-cypress-qlora",
    },
    {
        "run_name"          : "phi3-playwright-qlora",
        "model"             : "microsoft/Phi-3-mini-4k-instruct",
        "framework"         : "playwright",
        "train_loss"        : 0.4295,
        "train_runtime_s"   : 3123.2,
        "train_runtime_hrs" : 3123.2 / 3600,
        "train_samples"     : 195,
        "total_flos"        : 1.303e+16,
        "lora_r"            : 16,
        "lora_alpha"        : 32,
        "lora_dropout"      : 0.05,
        "target_modules"    : "qkv_proj,o_proj,gate_up_proj,down_proj",
        "epochs"            : 3,
        "batch_size"        : 1,
        "grad_accum"        : 8,
        "effective_batch"   : 8,
        "learning_rate"     : 2e-4,
        "max_seq_len"       : 2048,
        "hardware"          : "Apple M4 Pro MPS bf16",
        "hf_adapter_repo"   : "saifafzal1/phi3-mini-playwright-qlora",
    },
]

for run_cfg in FINETUNING_RUNS:
    run_name = run_cfg.pop("run_name")
    with mlflow.start_run(run_name=run_name):
        # Params (hyperparameters — fixed before training)
        mlflow.log_params({
            "model"           : run_cfg["model"],
            "framework"       : run_cfg["framework"],
            "lora_r"          : run_cfg["lora_r"],
            "lora_alpha"      : run_cfg["lora_alpha"],
            "lora_dropout"    : run_cfg["lora_dropout"],
            "target_modules"  : run_cfg["target_modules"],
            "epochs"          : run_cfg["epochs"],
            "batch_size"      : run_cfg["batch_size"],
            "grad_accum"      : run_cfg["grad_accum"],
            "effective_batch" : run_cfg["effective_batch"],
            "learning_rate"   : run_cfg["learning_rate"],
            "max_seq_len"     : run_cfg["max_seq_len"],
            "hardware"        : run_cfg["hardware"],
            "hf_adapter_repo" : run_cfg["hf_adapter_repo"],
            "train_samples"   : run_cfg["train_samples"],
        })
        # Metrics (outcomes)
        mlflow.log_metrics({
            "train_loss"        : run_cfg["train_loss"],
            "train_runtime_s"   : run_cfg["train_runtime_s"],
            "train_runtime_hrs" : round(run_cfg["train_runtime_hrs"], 4),
            "total_flos"        : run_cfg["total_flos"],
        })
        # Tags
        mlflow.set_tags({
            "phase"     : "Phase5-FineTuning",
            "method"    : "QLoRA",
            "framework" : run_cfg["framework"],
        })
    print(f"   ✅ Logged: {run_name}  (loss={run_cfg['train_loss']:.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Phase3-Baselines
# ═══════════════════════════════════════════════════════════════════════════════
print("\n📋 Logging Phase3-Baselines experiment...")
mlflow.set_experiment("Phase3-Baselines")

FRAMEWORKS = ["cypress", "playwright"]
MODELS     = ["claude-haiku", "gpt4o-mini", "gemini3-flash-lite", "gemini-flash"]

MODEL_DISPLAY = {
    "claude-haiku"      : "claude-3-haiku-20240307",
    "gpt4o-mini"        : "gpt-4o-mini",
    "gemini3-flash-lite": "gemini-3.1-flash-lite",
    "gemini-flash"      : "gemini-2.5-flash",
}

for framework in FRAMEWORKS:
    for model_key in MODELS:
        summary_path = RESULTS_DIR / framework / model_key / "summary.json"
        if not summary_path.exists():
            print(f"   ⚠️  Skipping {model_key}/{framework} — summary.json not found")
            continue

        with open(summary_path) as f:
            s = json.load(f)

        run_name = f"{model_key}-{framework}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model"         : MODEL_DISPLAY.get(model_key, model_key),
                "model_key"     : model_key,
                "framework"     : framework,
                "total_records" : s["total_records"],
            })
            mlflow.log_metrics({
                "generated"              : s["generated"],
                "skipped"                : s.get("skipped", 0),
                "completion_rate_pct"    : round(s["generated"] / s["total_records"] * 100, 2),
                "avg_latency_s"          : s["avg_latency_s"],
                "total_prompt_tokens"    : s.get("total_prompt_tokens", 0),
                "total_completion_tokens": s.get("total_completion_tokens", 0),
            })
            mlflow.set_tags({
                "phase"     : "Phase3-Baselines",
                "method"    : "zero-shot",
                "framework" : framework,
                "complete"  : str(s["generated"] == s["total_records"]),
            })
        pct = s["generated"] / s["total_records"] * 100
        print(f"   ✅ Logged: {run_name:<35}  {s['generated']}/{s['total_records']} ({pct:.0f}%)  latency={s['avg_latency_s']}s")

print("\n✅ All MLflow runs logged!")
print(f"   Run:  mlflow ui --backend-store-uri {MLRUNS_DIR}")
print(f"   Open: http://127.0.0.1:5000")
