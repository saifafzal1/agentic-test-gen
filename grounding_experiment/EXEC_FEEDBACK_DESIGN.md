# Execution-Feedback BMAD Loop — Design

## Why

Static grounding failed (OFF/ON/REPAIRED all 0/24, both models): the fine-tuned models hallucinate
a whole fictional app contract (selectors, routes, login APIs, redirects), and neither suggesting
nor enforcing real selectors recovers execution. The remaining lever is to close the loop with the
**ground truth itself** — run the script against the live app and feed the concrete runtime errors
back as the correction signal.

## The loop (variant of BMAD, execution-in-the-loop)

For each story, up to `max_iters`:
1. **Build** — generate a candidate (optionally with `dom_context` grounding ON).
2. **Measure = EXECUTE** — run the candidate against the live app via the harness; capture
   pass/fail + per-test failure messages.
3. **Assess** — accept if every test passes (status == passed).
4. **Decide** — else build corrective feedback from the *actual* runtime errors and regenerate.

Keep the best attempt (max individual tests passed) if none fully passes.

## Conditions

- Baseline for comparison = the pilot's static loop (already 0/24).
- `execfb` (this): execution feedback, grounding OFF.
- `execfb+ground`: execution feedback + real-selector injection — gives the model both *what
  failed* (runtime errors) and *what exists* (real selectors). Expected strongest.

## Metric

Live-execution pass rate (specs fully passing, individual tests) vs the static-loop pilot floor,
and iterations-to-pass. Eval set: the 24 grounded stories (saucedemo + the-internet).

## Honest expectation

Execution feedback is the mechanism most likely to help, but a 4B model may still struggle to
repair invented routes / redirects / network waits within a few iterations. Any lift over 0/24 is
a positive signal; the iteration trajectory (does test-pass count climb across attempts?) is itself
informative.

## Cost

Each iteration now includes a live execution (~30-60 s), so runs are slower than pure generation.
24 stories x ~3 iters x (gen + exec) ~ 2-3 h per condition on Phi-3. Overnight, caffeinated.
