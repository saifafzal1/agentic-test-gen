# Inference-Time Application Grounding — Experiment Design

## Motivation

The dissertation's execution pilot showed a **ranking inversion**: on live applications,
fine-tuned models passed only **2/96** specs vs the commercial baselines' **42/96**, despite
winning every static metric. Root cause = **application grounding**: the fine-tuned adapters
impose the training corpus's `data-testid` selector convention over the *real* selectors stated
in each story.

The grounded-corpus retrain (train on execution-verified grounded pairs, hold out apps)
**eliminated the convention bias (47/48 → 0/48 on unseen apps)** but only *partially* recovered
execution. Conclusion: **training-time grounding fixes the learned bias; full execution on an
unseen app additionally needs INFERENCE-time grounding** — giving the generator the target app's
real DOM at generation time so it uses selectors that actually exist.

This experiment measures the size of that inference-time grounding effect.

## Hypothesis

Injecting the target application's real selector inventory into the generation prompt (grounding
**ON**) raises the live-execution pass rate substantially over the story-only prompt (grounding
**OFF**) — most visibly for the fine-tuned models, which scored ~0 in the pilot.

## Design

Controlled A/B on the same evaluation set, changing only the grounding condition.

- **Evaluation set:** the 24 grounded pilot stories (`data/execution_validation/grounded_stories.jsonl`),
  spanning saucedemo + the-internet — apps with a live DOM the harness can execute against.
- **Factors:**
  - Grounding: **OFF** (story only, = the pilot baseline) vs **ON** (story + real selector inventory).
  - Model: `phi3`, `gemma4` (fine-tuned); optionally `gpt4o-mini`, `claude-haiku` (baselines, for the ceiling).
  - Framework: `cypress`, `playwright`.
- **Primary metric:** live-execution pass rate — specs fully passing and individual tests passing
  (from `execution_harness/run_execution.py`).
- **Secondary metrics:**
  - **Selector-adherence rate** — fraction of generated selectors that exist in the real DOM
    inventory (does grounding actually change what the model emits?).
  - **BMAD composite / acceptance** — to confirm static quality is not sacrificed.
  - Failure taxonomy on the residual failures.

## What "grounding ON" means (Task 2–3)

1. **DOM-context builder (Task 2):** for a story's target URL, return the real selector inventory
   (ids, `data-test`/`data-testid`, roles, inputs, buttons, links, selects). Source of truth =
   the crawled `grounded_corpus/profiles/*.json`; fall back to a live crawl via
   `grounded_corpus/crawl_dom.mjs` if a page isn't already profiled. Cached per (app, page).
2. **Grounding-aware generation (Task 3):** `agentic_loop.generator.generate()` gains an optional
   `dom_context` string; when present it injects an "Available real selectors on the target page:
   …" block into the prompt. When absent, behaviour is byte-identical to the current pilot path
   (clean OFF baseline).

## Comparison table this produces

| Model | Framework | Grounding OFF (pilot) | Grounding ON | Lift |
|-------|-----------|-----------------------|--------------|------|
| Phi-3 | Cypress | 0/24 (from pilot) | ? | ? |
| Phi-3 | Playwright | 1/24 | ? | ? |
| Gemma 4 | Cypress | 0/24 | ? | ? |
| Gemma 4 | Playwright | 1/24 | ? | ? |
| (baselines optional) | | 42/96 aggregate | ? | ? |

## Success criterion

Grounding ON lifts fine-tuned execution well above the ~2/96 floor. Any material, consistent
lift confirms the hypothesis and completes the paper's arc: **static wins → execution inverts →
grounding recovers.** A null/negative result is also publishable (it would imply the deficit is
deeper than selector knowledge).

## Reused assets

- `grounded_corpus/crawl_dom.mjs` — live-DOM selector crawler
- `grounded_corpus/profiles/*.json` — 8 apps already profiled
- `agentic_loop/generator.py`, `run_loop.py` — generation (to be extended with `dom_context`)
- `execution_harness/run_execution.py` — live-app executor + pass/fail parser
- `data/execution_validation/grounded_stories.jsonl` — 24-story eval set

## Compute notes (Apple M4 Pro, 48 GB, MPS)

Real models: run one model per process/kernel, mind MPS memory (no mid-process release; Phi-3
uses eager attention). Overnight, chunked, `caffeinate`. OFF and ON are separate generation
passes over 24 stories each — small, ~1–2 evenings total including execution.

## Status

- [x] Task 1 — branch `inference-grounding` + this design doc
- [ ] Task 2 — DOM-context builder (`grounding_experiment/dom_context.py`)
- [ ] Task 3 — `generate()` gains `dom_context` injection
- [ ] Task 4 — OFF vs ON generation over the 24 stories
- [ ] Task 5 — execute against live apps
- [ ] Task 6 — analysis (lift table + selector-adherence)
