#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh  —  Full BMAD Agentic Loop Pipeline
# ============================================================
# Runs all 4 model/framework combinations in sequence.
# For each combo it:
#   1. Deletes previously generated scripts that are syntactically
#      invalid (token-truncated), keeping the valid ones.
#   2. Starts the FastAPI inference server with the correct adapter.
#   3. Runs the BMAD loop (checkpoint/resume skips valid scripts).
#   4. Shuts the server down cleanly before the next combo.
#
# Prerequisites:
#   - .venv with transformers, fastapi, uvicorn installed
#   - Fine-tuned adapters present under fine_tuning/
#   - dataset_final.json present under data/
#   - node installed (used by syntax validator)
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh                     # run all 4 combos
#   ./run_pipeline.sh phi3 cypress        # run one specific combo
#   ./run_pipeline.sh --skip-validate     # skip initial syntax scan
#   ./run_pipeline.sh --fresh-start       # delete ALL old results first
# ============================================================
set -uo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv/bin"
PORT=8000
API_URL="http://localhost:8000"
API_WAIT_TIMEOUT=360    # seconds to wait for API to become healthy
MAX_TOKENS=1024
LOG_DIR="$SCRIPT_DIR/logs"

# Execution order — each entry is "model/framework"
COMBOS=("phi3/cypress" "phi3/playwright" "gemma4/cypress" "gemma4/playwright")

# ── Parse args ────────────────────────────────────────────────────────────────
SKIP_VALIDATE=false
FRESH_START=false
SINGLE_MODEL=""
SINGLE_FRAMEWORK=""

for arg in "$@"; do
    case "$arg" in
        --skip-validate)  SKIP_VALIDATE=true  ;;
        --fresh-start)    FRESH_START=true    ;;
        --help|-h)
            grep '^#' "$0" | head -30 | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

# Single-combo override: ./run_pipeline.sh phi3 cypress
if [ $# -ge 2 ] && [[ "$1" != --* ]]; then
    SINGLE_MODEL="$1"
    SINGLE_FRAMEWORK="$2"
    COMBOS=("$SINGLE_MODEL/$SINGLE_FRAMEWORK")
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
API_PID=""

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_api() {
    local elapsed=0
    log "  Waiting for API to become healthy (timeout ${API_WAIT_TIMEOUT}s)..."
    until curl -sf "$API_URL/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ "$elapsed" -ge "$API_WAIT_TIMEOUT" ]; then
            log "  ERROR: API did not start within ${API_WAIT_TIMEOUT}s"
            log "  Check log: $1"
            tail -20 "$1" 2>/dev/null || true
            stop_api
            exit 1
        fi
    done
    log "  API healthy after ${elapsed}s"
}

start_api() {
    local model="$1" framework="$2"
    local logfile="$LOG_DIR/api_${model}_${framework}.log"

    log ""
    log "========================================================"
    log "  Starting API  —  WARMUP_MODEL=$model/$framework"
    log "  Log: $logfile"
    log "========================================================"

    WARMUP_MODEL="$model/$framework" \
        "$VENV/uvicorn" api.app:app \
            --host 0.0.0.0 \
            --port "$PORT" \
            --log-level warning \
        > "$logfile" 2>&1 &
    API_PID=$!
    log "  API PID: $API_PID"
    wait_for_api "$logfile"
}

stop_api() {
    if [ -n "${API_PID:-}" ] && kill -0 "$API_PID" 2>/dev/null; then
        log "  Stopping API (PID $API_PID)..."
        kill "$API_PID" 2>/dev/null || true
        # Wait up to 15s for graceful shutdown
        local i=0
        while kill -0 "$API_PID" 2>/dev/null && [ $i -lt 15 ]; do
            sleep 1; i=$((i+1))
        done
        kill -9 "$API_PID" 2>/dev/null || true
        API_PID=""
        log "  API stopped."
    fi
}

purge_invalid_results() {
    local model="$1" framework="$2"
    local report="$SCRIPT_DIR/evaluation/syntax_validation_report.json"

    if [ ! -f "$report" ]; then
        log "  No syntax validation report found — will run full loop (no purge needed)."
        return 0
    fi

    log "  Purging invalid (truncated) results for $model/$framework..."
    "$VENV/python" -m evaluation.rerun_invalid \
        --model "$model" \
        --framework "$framework" \
        --dry-run 2>&1 | grep -E '(would delete|invalid scripts|nothing to re-run)' || true
    # Actual delete (not dry-run) — rerun_invalid also calls run_loop,
    # but we call run_loop separately so we pass just the delete step here.
    "$VENV/python" -c "
import json, sys
from pathlib import Path
report = Path('evaluation/syntax_validation_report.json')
results_dir = Path('results/agentic_loop') / '$model' / '$framework'
data = json.loads(report.read_text())
detail = data.get('detail', {}).get('$model/$framework', {})
records = detail.get('records', [])
deleted = 0
for r in records:
    if not r['syntax_valid']:
        p = results_dir / (r['tc_id'] + '.json')
        if p.exists():
            p.unlink()
            deleted += 1
print(f'  Deleted {deleted} invalid result files for $model/$framework')
"
}

run_loop() {
    local model="$1" framework="$2"
    log ""
    log "  Running BMAD loop: $model/$framework  (max_tokens=$MAX_TOKENS)"
    "$VENV/python" -m agentic_loop.run_loop \
        --model     "$model" \
        --framework "$framework" \
        --max-tokens "$MAX_TOKENS"
}

# ── Cleanup on unexpected exit ────────────────────────────────────────────────
trap 'log "Pipeline interrupted."; stop_api' EXIT INT TERM

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

if [ "$FRESH_START" = true ]; then
    log "FRESH START — deleting all existing results..."
    rm -rf "$SCRIPT_DIR/results/agentic_loop"
    rm -f  "$SCRIPT_DIR/evaluation/syntax_validation_report.json"
fi

# ── Optionally refresh syntax validation report ───────────────────────────────
if [ "$SKIP_VALIDATE" = false ] && [ "$FRESH_START" = false ]; then
    RESULTS_EXIST=$(find "$SCRIPT_DIR/results/agentic_loop" -name "TC_*.json" 2>/dev/null | head -1)
    if [ -n "$RESULTS_EXIST" ]; then
        log ""
        log "Scanning existing results for truncated scripts..."
        "$VENV/python" -m evaluation.validate_syntax 2>&1 | tail -20
    fi
fi

# ── Main loop ─────────────────────────────────────────────────────────────────
OVERALL_START=$(date +%s)
COMPLETED=()

for combo in "${COMBOS[@]}"; do
    model="${combo%/*}"
    framework="${combo#*/}"
    COMBO_START=$(date +%s)

    log ""
    log "################################################################"
    log "  COMBO: $model / $framework"
    log "################################################################"

    # Remove invalid old results so run_loop regenerates them
    purge_invalid_results "$model" "$framework"

    # Start API with the right adapter warm
    start_api "$model" "$framework"

    # Run the BMAD loop (checkpoint skips already-valid results)
    run_loop "$model" "$framework"

    stop_api

    COMBO_END=$(date +%s)
    ELAPSED=$(( (COMBO_END - COMBO_START) / 60 ))
    COMPLETED+=("$combo (${ELAPSED}m)")
    log "  $model/$framework done in ${ELAPSED} min."
done

# ── Final syntax validation ───────────────────────────────────────────────────
log ""
log "Running final syntax validation..."
"$VENV/python" -m evaluation.validate_syntax 2>&1 | tail -30

# ── Summary ───────────────────────────────────────────────────────────────────
OVERALL_END=$(date +%s)
TOTAL_MIN=$(( (OVERALL_END - OVERALL_START) / 60 ))

log ""
log "========================================================"
log "  PIPELINE COMPLETE — total time: ${TOTAL_MIN} min"
log "========================================================"
for c in "${COMPLETED[@]}"; do
    log "  ✅  $c"
done
log "========================================================"
log "  Results : $SCRIPT_DIR/results/agentic_loop/"
log "  Syntax  : $SCRIPT_DIR/evaluation/syntax_validation_report.json"
log "  API logs: $LOG_DIR/"
log "========================================================"

# Remove the EXIT trap so it doesn't print "Pipeline interrupted"
trap - EXIT
