# Agentic AI System for Autonomous Test Script Generation & Optimization
**Dissertation | M.Tech AI & ML | BITS Pilani WILP | Saif Afzal (2024AA05546)**

## Project Structure
| Folder | Machine | Purpose |
|---|---|---|
| `data/` | Mac B | Synthetic Jira → test script dataset (279 pairs) |
| `baselines/` | Mac B | Zero-shot API inference (GPT-4o-mini, Claude Haiku) |
| `fine_tuning/` | Mac A | QLoRA fine-tuning of Phi-3 Mini & Gemma 4 E4B |
| `agentic_loop/` | Mac A | BMAD (Build–Measure–Assess–Decide) feedback loop |
| `evaluation/` | Mac A | Scoring, ANOVA, Wilcoxon, chi-square, error matrix |
| `results/` | Mac A | Per-record outputs and run summaries |
| `writing/` | — | Dissertation report (LaTeX source in `writing/latex_export/`) |

## Reproduce on a fresh machine — **without retraining**

The trained LoRA adapters live on the HuggingFace Hub, not in this repo
(weights are too large for git). A new machine only needs to *download*
them — no fine-tuning required. Total download ≈ 30 GB (base models) +
~3 GB (adapters).

### 1. Clone and create the environment
```bash
git clone https://github.com/saifafzal1/agentic-test-gen.git
cd agentic-test-gen
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Authenticate to HuggingFace
The base models are gated and the adapters are private, so a token with
read access to both is required:
```bash
huggingface-cli login        # paste a token from https://huggingface.co/settings/tokens
```

### 3. Download the base models (public, gated)
```bash
python fine_tuning/download_phi3.py     # → fine_tuning/phi3-base-model/
python fine_tuning/download_gemma4.py   # → fine_tuning/gemma4-base-model/
```

### 4. Download the fine-tuned adapters (private) — **this replaces training**
The loader (`agentic_loop/generator.py`) reads each adapter from a fixed
local directory, so download each HF repo into the matching folder:

```bash
huggingface-cli download saifafzal1/phi3-mini-cypress-qlora     --local-dir fine_tuning/phi3-cypress
huggingface-cli download saifafzal1/phi3-mini-playwright-qlora  --local-dir fine_tuning/phi3-playwright
huggingface-cli download saifafzal1/gemma4-E4B-cypress-qlora    --local-dir fine_tuning/gemma4-cypress
huggingface-cli download saifafzal1/gemma4-E4B-playwright-qlora --local-dir fine_tuning/gemma4-playwright
```

| HuggingFace repo | Local dir the code expects |
|---|---|
| `saifafzal1/phi3-mini-cypress-qlora` | `fine_tuning/phi3-cypress/` |
| `saifafzal1/phi3-mini-playwright-qlora` | `fine_tuning/phi3-playwright/` |
| `saifafzal1/gemma4-E4B-cypress-qlora` | `fine_tuning/gemma4-cypress/` |
| `saifafzal1/gemma4-E4B-playwright-qlora` | `fine_tuning/gemma4-playwright/` |

### 5. Run the BMAD agentic loop (single combo, or all four)
```bash
# one model/framework combination
python -m agentic_loop.run_loop --model phi3 --framework cypress

# all four combinations, with checkpoint/resume + sleep-prevention
caffeinate -dims ./run_pipeline.sh
```
Outputs land in `results/agentic_loop/<model>/<framework>/`.

### 6. Reproduce the evaluation
```bash
python evaluation/validate_syntax.py
python evaluation/statistical_analysis.py    # ANOVA, Wilcoxon, chi-square
```

## Only if you *want* to retrain from scratch
Training is **not** required to run the system (use the download steps
above). To reproduce training itself:
```bash
python -u fine_tuning/finetune_phi3.py   --framework cypress
python -u fine_tuning/finetune_phi3.py   --framework playwright
python -u fine_tuning/finetune_gemma4.py --framework cypress
python -u fine_tuning/finetune_gemma4.py --framework playwright
```
Requires an Apple-Silicon Mac (MPS, bf16) or a CUDA GPU; ~1 h per Phi-3
adapter and 3–5 h per Gemma adapter on an M4 Pro (48 GB).

## Run via Docker (no local setup)

Two containers reproduce the pipeline end to end; they share the
`./results` volume (generation output feeds the executor). Definitions
are in `docker/`.

- **`inference`** — downloads base models + the four private adapters
  from HuggingFace at start (no training, no baked-in weights) and runs
  the BMAD loop. **Needs a CUDA GPU host + the NVIDIA Container Toolkit.**
- **`execution`** — GPU-free; built on the official Playwright image
  (browsers preinstalled) plus Cypress, it runs the generated scripts
  against the live demo apps and captures pass/fail metrics on both the
  dummy (synthetic 279) and practical (grounded, live-app) data.

Configure once via an env file (works identically on every OS — inline
`VAR=value` before the command does **not** work in Windows PowerShell):
```bash
cp docker/.env.example docker/.env      # then edit docker/.env, set HF_TOKEN
```
```bash
# 1. Generate (GPU host recommended)
docker compose --env-file docker/.env -f docker/docker-compose.yml run --rm inference
# 2. Execute the generated scripts + capture metrics (any host)
docker compose --env-file docker/.env -f docker/docker-compose.yml run --rm execution
```

### Windows (Docker Desktop + WSL2)
Windows is a **first-class GPU host** for this — better than a Mac, which
cannot do GPU-in-Docker at all.

- **Prerequisites:** Docker Desktop with the **WSL2 backend** enabled. For
  GPU: an NVIDIA GPU with a current driver — CUDA works through WSL2 with
  **no extra toolkit install** (Docker Desktop wires it up). Verify with
  `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- **Clone cleanly:** `git clone` normally — the repo's `.gitattributes`
  forces LF on the shell scripts, and the Dockerfiles strip any stray CR as
  a second safety net, so the classic `bash\r` container error can't occur.
- **Run the exact commands above** in PowerShell or WSL — they are
  OS-neutral because settings come from `docker/.env`, not inline vars, and
  the compose volume paths are relative (no `$(pwd)` needed).
- **No NVIDIA GPU (CPU-only Windows box):** the recommended split is —
  - **`execution` container: use it fully.** GPU-free, fast, runs the
    Cypress/Playwright metrics capture on dummy + practical data. This is
    the main value on a CPU machine.
  - **`inference` container: smoke test only.** It runs on CPU but a
    3.8–4B model is minutes per story. The code now loads **float32 on CPU**
    (bf16 is emulated/slow there), and `docker/.env` ships with
    `RUN_LIMIT=2` + `RUN_MODEL=phi3` so a first run generates just 2 stories
    with the *smaller* model to prove the pipeline end-to-end. Do **not**
    attempt full runs or Gemma on CPU — Gemma (4B) likely won't fit in a
    typical laptop's RAM, and a full 279-story CPU run could take days.
  - **Delete the `deploy:` block** in `docker/docker-compose.yml` so compose
    doesn't demand a GPU that isn't there.
  - Needs ~8 GB free RAM for the Phi-3 smoke test (float32) plus ~8 GB disk
    for its base model.
- **Disk:** ~35 GB of model downloads land in the persistent `hf_cache`
  volume on first run; ensure WSL2 has the space.

**GPU vs Mac vs Windows:** macOS Docker **cannot reach Apple Silicon MPS**
(CPU-only in a container — use the native steps above there). A Linux or
**Windows+WSL2 CUDA host** gets real GPU acceleration, where
`agentic_loop/generator.py` now auto-selects `cuda`. The `execution`
container is GPU-free everywhere.

> These Docker files are provided as reproducibility scaffolding and
> should be validated on their first build on an actual CUDA host (they
> were authored, not yet built, in the environment that produced them).

## Notes
- The four adapter repos are **private**; the token in step 2 must have
  access. Base models (`microsoft/Phi-3-mini-4k-instruct`,
  `google/gemma-3-4b-it`) are re-downloadable by anyone with gated access.
- The dissertation report is in `writing/latex_export/` (compile
  `main.tex` with **XeLaTeX**).
