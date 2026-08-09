#!/usr/bin/env bash
# ============================================================
# run_full_sweep.sh — bake the full BMAD sweep into the demo notebook
# ============================================================
# Runs AgentDemoOnLocal.ipynb end to end with the REAL fine-tuned models,
# executing the full 4-stories x 2-frameworks sweep (FULL_SWEEP=1) and
# saving every cell's output back into the .ipynb, so you can present the
# results at the viva without re-running anything.
#
# caffeinate wraps the whole run: sleep is prevented ONLY for the lifetime
# of this process and released automatically the moment it finishes — no
# fixed timer to guess, nothing left holding the machine awake afterwards.
#
# Usage (from the repo root):
#   ./notebooks/run_full_sweep.sh
#
# After it finishes, AgentDemoOnLocal.ipynb contains the baked outputs.
# For the live viva, leave FULL_SWEEP unset so the notebook runs the safe
# single-story cell instead.
# ============================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV_PY="$REPO/.venv/bin/python"
NB="$REPO/notebooks/AgentDemoOnLocal.ipynb"
LOG="$REPO/logs/full_sweep_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$REPO/logs"

# Register the venv as the "diss-venv" kernel the notebook declares (idempotent).
"$VENV_PY" -m ipykernel install --user --name diss-venv >/dev/null 2>&1 || true

# Run the FULL 4x2 sweep (the notebook's cell reads this env var).
export FULL_SWEEP=1

echo "Repo        : $REPO"
echo "Notebook    : $NB"
echo "Log         : $LOG"
echo "Model       : phi3 (default) | FULL_SWEEP=1 (4x2) | caffeinated for the run only"
echo "Starting at : $(date '+%H:%M:%S')  — expect ~8-15 min on phi3"
echo "============================================================"

# caffeinate holds sleep for exactly as long as the nbconvert run takes,
# then releases automatically. -dimsu = display/idle/disk/system + user-active.
caffeinate -dimsu "$VENV_PY" -m nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=diss-venv \
    "$NB" 2>&1 | tee "$LOG"

echo "============================================================"
echo "Done at     : $(date '+%H:%M:%S')"
echo "Outputs baked into: $NB"
echo "caffeinate released automatically."
