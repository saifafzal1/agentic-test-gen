# Agentic AI System for Autonomous Test Script Generation & Optimization
**Dissertation | M.Tech AI & ML | BITS Pilani WILP | Saif Afzal (2024AA05546)**

## Project Structure
| Folder | Machine | Purpose |
|---|---|---|
| `data/` | Mac B | Synthetic Jira → test script dataset (280 pairs) |
| `baselines/` | Mac B | Zero-shot API inference (Claude Haiku, Gemini Flash) |
| `fine_tuning/` | Mac A | QLoRA fine-tuning of Gemma 4 E4B |
| `agentic_loop/` | Mac A | BMAD feedback loop implementation |
| `evaluation/` | Mac A | Scoring, ANOVA, Wilcoxon, error matrix |
| `notebooks/` | Both | Experiments & EDA |
| `results/` | Mac A | Outputs, plots, model checkpoints |
| `writing/` | Mac B | Dissertation chapters |

## Setup
```bash
# Mac A (M4 Pro — fine-tuning & agentic loop)
cd agentic-test-gen && source .venv/bin/activate

# Mac B (Intel — data gen & baselines)
cd dissertation-project && source .venv/bin/activate
```
