#!/usr/bin/env bash
# ============================================================
# run_grounded.sh — Execution-validation generation run
# ============================================================
# Runs the BMAD loop over the 24 grounded user stories
# (data/execution_validation/grounded_stories.jsonl) for all 4
# model/framework combinations. Outputs are kept fully separate from the
# dissertation results, under results/execution_validation/.
#
# Same reliability pattern as run_pipeline.sh: sequential combos,
# checkpoint/resume (a restart skips completed TC_G* records), and up to
# 3 retries per combo on a crash.
#
# Usage:
#   caffeinate -dims ./run_grounded.sh              # all 4 combos
#   caffeinate -dims ./run_grounded.sh phi3 cypress # one combo
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv/bin"
DATA="$SCRIPT_DIR/data/execution_validation/grounded_stories.jsonl"
OUT_DIR="$SCRIPT_DIR/results/execution_validation"
LOG_DIR="$SCRIPT_DIR/logs"
MAX_TOKENS=1024
MAX_RETRIES=3

mkdir -p "$LOG_DIR" "$OUT_DIR"

COMBOS=("phi3/cypress" "phi3/playwright" "gemma4/cypress" "gemma4/playwright")
if [[ $# -eq 2 ]]; then
    COMBOS=("$1/$2")
fi

EXPECTED=$(wc -l < "$DATA" | tr -d ' ')
echo "============================================================"
echo "  Grounded execution-validation generation run"
echo "  Stories: $EXPECTED   Combos: ${COMBOS[*]}"
echo "============================================================"

overall_rc=0
for combo in "${COMBOS[@]}"; do
    model="${combo%%/*}"
    framework="${combo##*/}"
    log="$LOG_DIR/grounded_${model}_${framework}.log"

    attempt=1
    while (( attempt <= MAX_RETRIES )); do
        echo ""
        echo ">>> [$combo] attempt $attempt/$MAX_RETRIES  ($(date '+%H:%M:%S'))"
        "$VENV/python" -u -m agentic_loop.run_loop \
            --model "$model" \
            --framework "$framework" \
            --data "$DATA" \
            --out-dir "$OUT_DIR" \
            --max-tokens "$MAX_TOKENS" 2>&1 | tee -a "$log"
        rc=${PIPESTATUS[0]}

        done_count=$(ls "$OUT_DIR/$model/$framework"/TC_G*.json 2>/dev/null | wc -l | tr -d ' ')
        if [[ $rc -eq 0 && "$done_count" -eq "$EXPECTED" ]]; then
            echo ">>> [$combo] complete: $done_count/$EXPECTED records"
            break
        fi
        echo ">>> [$combo] incomplete (rc=$rc, $done_count/$EXPECTED) — retrying"
        (( attempt++ ))
    done

    if (( attempt > MAX_RETRIES )); then
        echo "!!! [$combo] FAILED after $MAX_RETRIES attempts — continuing to next combo"
        overall_rc=1
    fi
done

echo ""
echo "============================================================"
echo "  Generation finished ($(date '+%H:%M:%S')). Per-combo record counts:"
for combo in "${COMBOS[@]}"; do
    model="${combo%%/*}"; framework="${combo##*/}"
    n=$(ls "$OUT_DIR/$model/$framework"/TC_G*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "    $combo: $n/$EXPECTED"
done
echo "============================================================"
exit $overall_rc
