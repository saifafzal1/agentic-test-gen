#!/usr/bin/env bash
# Entrypoint for the GPU inference container.
#   1. Authenticates to HuggingFace (HF_TOKEN env var).
#   2. Downloads base models + the four private QLoRA adapters into the
#      exact local dirs agentic_loop/generator.py expects (skips any
#      already present, so a mounted cache makes restarts instant).
#   3. Runs the BMAD loop for the requested model/framework/data.
#
# Env vars:
#   HF_TOKEN        (required) HF token with read access to the private adapters
#   RUN_MODEL       phi3 | gemma4          (default phi3)
#   RUN_FRAMEWORK   cypress | playwright   (default cypress)
#   RUN_DATA        dataset path           (default data/dataset_final.json)
#   RUN_OUT_DIR     results base dir       (default results/agentic_loop)
#   SKIP_GEMMA      set to 1 to skip the ~15 GB Gemma base-model download
set -euo pipefail

: "${RUN_MODEL:=phi3}"
: "${RUN_FRAMEWORK:=cypress}"
: "${RUN_DATA:=data/dataset_final.json}"
: "${RUN_OUT_DIR:=results/agentic_loop}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is required (read access to the private adapters)." >&2
  exit 1
fi
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential false

dl() {  # dl <hf-repo> <local-dir>
  if [[ -f "$2/adapter_model.safetensors" || -f "$2/config.json" ]]; then
    echo "✓ $2 already present — skipping download"
  else
    echo "↓ downloading $1 → $2"
    huggingface-cli download "$1" --local-dir "$2" --quiet
  fi
}

# Base models
dl microsoft/Phi-3-mini-4k-instruct fine_tuning/phi3-base-model
if [[ "${SKIP_GEMMA:-0}" != "1" ]]; then
  dl google/gemma-3-4b-it fine_tuning/gemma4-base-model
fi

# Private adapters → dirs the loader reads
dl saifafzal1/phi3-mini-cypress-qlora     fine_tuning/phi3-cypress
dl saifafzal1/phi3-mini-playwright-qlora  fine_tuning/phi3-playwright
dl saifafzal1/gemma4-E4B-cypress-qlora    fine_tuning/gemma4-cypress
dl saifafzal1/gemma4-E4B-playwright-qlora fine_tuning/gemma4-playwright

LIMIT_ARG=()
[[ -n "${RUN_LIMIT:-}" ]] && LIMIT_ARG=(--limit "$RUN_LIMIT")   # e.g. RUN_LIMIT=2 for a CPU smoke test

echo "=== Running BMAD loop: model=$RUN_MODEL framework=$RUN_FRAMEWORK data=$RUN_DATA ${LIMIT_ARG[*]:-} ==="
exec python -m agentic_loop.run_loop \
  --model "$RUN_MODEL" --framework "$RUN_FRAMEWORK" \
  --data "$RUN_DATA" --out-dir "$RUN_OUT_DIR" "${LIMIT_ARG[@]}"
