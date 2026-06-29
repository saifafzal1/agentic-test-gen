# Branch: `midsem`

## Scope — Mid-Semester Submission
**Deadline: 21 Jun 2026** | Turnitin check (<25%) | VIVA portal submission | Mid-Sem VIVA

---

## Phases Included in This Branch

| Phase | Dates | Work |
|-------|-------|------|
| Phase 1 — Outline | 05 May – 11 May 2026 | Literature review; dataset scoping; Outline document submission |
| Phase 2 — Dataset Prep | 12 May – 25 May 2026 | Collect & anonymize 280 Cypress + 280 Playwright Jira-to-script pairs (560 total); JSONL formatting |
| Phase 3 — Baseline Runs | 26 May – 08 Jun 2026 | Zero-shot LLM baselines (GPT-4o-mini, Claude Haiku, Gemini) across both frameworks; baseline F1 established |
| Phase 4 — Fine-tune Phi-3-mini | 09 Jun – 15 Jun 2026 | QLoRA fine-tuning of Phi-3-mini on Cypress (280 pairs); then Playwright (280 pairs); F1 recorded per framework |
| Phase 7 — Mid-Sem Report | 16 Jun – 21 Jun 2026 | Mid-Semester report drafted; Turnitin check (<25%); submitted to VIVA portal; Mid-Sem VIVA |

## Phases NOT in this branch (post-midsem)
- Phase 5 — Fine-tune Gemma 4 E4B (22 Jun – 05 Jul 2026)
- Phase 6 — Agentic Loop (06 Jul – 14 Jul 2026)
- Phase 8 — Evaluation (15 Jul – 21 Jul 2026)
- Phase 9 — Writing (22 Jul – 27 Jul 2026)
- Phase 10 — Final Submission (28 Jul – 02 Aug 2026)

---

## Merge Strategy
- Day-to-day development happens on `main`
- Merge `main` → `midsem` when Phase 4 is complete (by 15 Jun 2026)
- Tag as `v1.0-midsem` before VIVA submission
