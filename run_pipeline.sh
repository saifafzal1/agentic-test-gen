#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh  —  Full BMAD Agentic Loop Pipeline (Phase 6 rebuild)
# ============================================================
# Runs all 4 model/framework combinations in sequence. Unlike the previous
# (phase6-eval) implementation, there is no separate inference server to
# start/stop/wait-on — agentic_loop.run_loop loads the model in-process and
# calls it directly. Every crash hit on phase6-eval (port conflicts, dropped
# connections, read timeouts, a retry-logic bug) came from that HTTP layer;
# removing it removes that whole failure class.
#
# For each combo it:
#   1. Deletes previously generated scripts that are syntactically invalid
#      (token-truncated), keeping the valid ones.
#   2. Runs the BMAD loop (checkpoint/resume skips valid scripts already done).
#   3. Retries a crashed combo up to 3x (relies on checkpoint/resume so a
#      restart picks up where it left off) before giving up.
#
# Prerequisites:
#   - .venv with transformers, peft, torch installed
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
log() { echo "[$(date '+%H:%M:%S')] $*"; }

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

    # Run the BMAD loop, retrying (checkpoint/resume picks up where it left
    # off) on crash instead of aborting the whole multi-hour pipeline.
    COMBO_OK=false
    for attempt in 1 2 3; do
        if run_loop "$model" "$framework"; then
            COMBO_OK=true
            break
        fi
        log "  ⚠️  $model/$framework failed on attempt $attempt/3."
        [ "$attempt" -lt 3 ] && log "     Resuming from checkpoint..."
    done

    if [ "$COMBO_OK" != true ]; then
        log ""
        log "  ❌ ERROR: $model/$framework FAILED after 3 attempts."
        log "     Aborting pipeline — fix the issue and re-run rather than silently continuing."
        exit 1
    fi

    # Sanity check: did we actually produce a result for every dataset record?
    EXPECTED=$("$VENV/python" -c "import json; print(len(json.load(open('$SCRIPT_DIR/data/dataset_final.json'))))")
    ACTUAL=$(find "$SCRIPT_DIR/results/agentic_loop/$model/$framework" -maxdepth 1 -name "TC_*.json" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$ACTUAL" -ne "$EXPECTED" ]; then
        log "  ⚠️  WARNING: $model/$framework produced $ACTUAL/$EXPECTED records despite exit 0 — treat as incomplete."
    fi

    COMBO_END=$(date +%s)
    ELAPSED=$(( (COMBO_END - COMBO_START) / 60 ))
    COMPLETED+=("$combo (${ELAPSED}m, ${ACTUAL}/${EXPECTED} records)")
    log "  $model/$framework done in ${ELAPSED} min. (${ACTUAL}/${EXPECTED} records)"
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
log "========================================================"
