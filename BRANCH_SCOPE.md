# Branch: `final-submission`

## Scope — Final Dissertation Submission
**Deadline: 02 Aug 2026** | Turnitin final check (<20%) | Upload to VIVA portal | 20-min PPT + live demo | Final VIVA

---

## All Phases Included in This Branch

| Phase | Dates | Work |
|-------|-------|------|
| Phase 1 — Outline | 05 May – 11 May 2026 | Literature review; dataset scoping; Outline document submission |
| Phase 2 — Dataset Prep | 12 May – 25 May 2026 | Collect & anonymize 280 Cypress + 280 Playwright Jira-to-script pairs (560 total); JSONL formatting |
| Phase 3 — Baseline Runs | 26 May – 08 Jun 2026 | Zero-shot LLM baselines (GPT-4o-mini, Claude Haiku, Gemini) across both frameworks; baseline F1 established |
| Phase 4 — Fine-tune Phi-3-mini | 09 Jun – 15 Jun 2026 | QLoRA fine-tuning of Phi-3-mini on Cypress (280 pairs); then Playwright (280 pairs); F1 recorded per framework |
| Phase 7 — Mid-Sem Report | 16 Jun – 21 Jun 2026 | Mid-Semester report drafted; Turnitin check (<25%); submitted to VIVA portal; Mid-Sem VIVA |
| Phase 5 — Fine-tune Gemma 4 E4B | 22 Jun – 05 Jul 2026 | QLoRA fine-tuning of Gemma 4 E4B on Cypress; then Playwright; cross-model comparison table populated |
| Phase 6 — Agentic Loop | 06 Jul – 14 Jul 2026 | BMAD lifecycle agent implementation; iterative script validation & correction layer built and tested |
| Phase 8 — Evaluation | 15 Jul – 21 Jul 2026 | Full Precision/Recall/F1 matrix; error taxonomy; ANOVA/Wilcoxon statistical significance tests |
| Phase 9 — Writing | 22 Jul – 27 Jul 2026 | Full dissertation drafted; supervisor & examiner review; revisions incorporated |
| Phase 10 — Final Submission | 28 Jul – 02 Aug 2026 | Turnitin final check (<20%); upload to VIVA portal; 20-min PPT + live demo prepared; Final VIVA |

---

## Merge Strategy
- Day-to-day development happens on `main`
- Continuously merge `main` → `final-submission` as phases complete
- Final merge + tag as `v2.0-final` before 28 Jul 2026
