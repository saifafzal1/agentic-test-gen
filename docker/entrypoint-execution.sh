#!/usr/bin/env bash
# Entrypoint for the execution + metrics container.
# Executes generated scripts (mounted under results/) against the live
# demo applications and writes per-spec pass/fail outcomes.
#
# Env vars:
#   EXEC_MODEL      phi3 | gemma4 | gpt4o-mini | claude-haiku | phi3-grounded
#                   (default: run all fine-tuned via --source finetuned)
#   EXEC_FRAMEWORK  cypress | playwright   (default: both)
set -euo pipefail

cd execution_harness

ARGS=()
[[ -n "${EXEC_MODEL:-}" ]]     && ARGS+=(--model "$EXEC_MODEL")
[[ -n "${EXEC_FRAMEWORK:-}" ]] && ARGS+=(--framework "$EXEC_FRAMEWORK")

echo "=== Execution harness: python run_execution.py ${ARGS[*]:-(all finetuned)} ==="
exec python run_execution.py "${ARGS[@]}"
