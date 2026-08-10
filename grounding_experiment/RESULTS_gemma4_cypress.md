# Grounding Experiment — Gemma 4 / Cypress (Task 8, generality check)

## Result: same as Phi-3 — grounding gives no meaningful execution lift

```
specs fully passing:   OFF 0/24  ->  ON 0/24  ->  REPAIRED 0/24
individual tests:      OFF 0/48  ->  ON 1/49  ->  REPAIRED 2/50   (242 selectors repaired)
```

- ON (prompt-injection grounding): 23/24 scripts still used the trained `data-testid`
  convention despite the real selectors being injected — same override behaviour as Phi-3.
- REPAIRED (enforced selectors): the data-testid failures dropped (22 -> 13) but were replaced by
  the residual contract-hallucination failures (invented redirects/behaviour, invented routes,
  invented cy.wait, dynamic selectors not on the static page). Only 2/50 individual tests pass;
  still 0/24 full specs.

## Conclusion — the finding is robust across BOTH fine-tuned models

| Model | OFF | ON (suggested) | REPAIRED (enforced) |
|-------|-----|----------------|---------------------|
| Phi-3 Mini | 0/24 | 0/24 | 0/24 |
| Gemma 4 E4B | 0/24 | 0/24 | 0/24 |

Neither fine-tuned model produces executable saucedemo/the-internet scripts, and neither
prompt-injection nor enforced selector-repair recovers execution. The deficit is a whole-app
contract hallucination (selectors + routes + API calls + redirects), not a selector-only or a
single-model artefact. Training-time grounding (grounded-corpus retrain) remains the only
intervention that addressed the root cause.
