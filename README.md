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

## Notes
- The four adapter repos are **private**; the token in step 2 must have
  access. Base models (`microsoft/Phi-3-mini-4k-instruct`,
  `google/gemma-3-4b-it`) are re-downloadable by anyone with gated access.
- The dissertation report is in `writing/latex_export/` (compile
  `main.tex` with **XeLaTeX**).
