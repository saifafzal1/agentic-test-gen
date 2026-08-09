# Grounding Experiment — Phi-3 / Cypress (first slice)

## Result: inference-time prompt-injection grounding gives ZERO execution lift

| Condition | Specs fully passing | Individual tests passing | ON scripts adopting real `data-test` selectors |
|-----------|--------------------|--------------------------|-----------------------------------------------|
| OFF (ungrounded, = pilot) | 0/24 | 0/40 | — |
| ON (real selector inventory injected) | 0/24 | 0/44 | 0/24 |

OFF reproduces the dissertation pilot's 0/24 for phi3/cypress exactly (harness sanity check passes).

## Why ON failed — selector-adherence analysis
- 0/24 ON scripts used the injected real `data-test` attributes.
- 20/24 reverted to the trained `data-testid` convention.
- Failure taxonomy (ON): 16 assert on non-existent `data-testid` selectors; 6 invented
  network waits / missing elements; 1 invented `/login` route (404); 1 `cy.viewport` preset.

The model used the correct selector *values* (username, password) but forced its habitual
*attribute name* (`data-testid`) instead of the real `data-test` it was handed.

## Interpretation (the finding)
Training-time grounding (grounded-corpus retrain) ELIMINATED the convention bias (0/48 scripts
used data-testid). Inference-time prompt injection did NOT (20/24 still used it). **Merely
supplying the real selectors in the prompt is insufficient — a domain-fine-tuned model overrides
explicit contrary instructions with its trained convention.** Grounding must be *enforced*, not
suggested.

## Directly motivates Task 7 (enforced grounding)
Deterministic selector-repair: post-generation, rewrite `data-testid="X"` -> `data-test="X"`
(and nearest-match) using the same real DOM inventory, then re-execute. Expected to flip many of
the 16 selector-name failures to passing — testing whether ENFORCED grounding recovers execution
where SUGGESTED grounding did not. This is the compelling next experiment.

---

# Task 7 — Enforced selector-repair result: ALSO zero lift (the deeper finding)

```
specs fully passing:   OFF 0/24  ->  ON 0/24  ->  REPAIRED 0/24
individual tests:      OFF 0/40  ->  ON 0/44  ->  REPAIRED 0/47   (237 selectors repaired)
```

Even after deterministically rewriting 237 invented selectors to the REAL ones from the DOM
inventory, not one spec or test passes. Residual failure taxonomy (24 specs):

| Count | Residual blocker |
|-------|------------------|
| 8 | Invented network wait — `cy.wait('@loginRequest')` for a login REST API saucedemo has no such call |
| 8 | Invented app behaviour — asserts redirect to `/dashboard` (real: `/inventory.html`); invented counts/text |
| 5 | Dynamic selector absent from the static crawl (`data-test="error"` appears only after a failed login) |
| 2 | Invented route — `cy.visit('/inventory')` (missing `.html`) -> 404 |
| 1 | Framework API misuse — `cy.viewport('375x812')` is not a valid Cypress preset |

## Synthesis — the paper's argument

1. Static metrics (dissertation Ch.5): fine-tuning wins on every measure.
2. Execution (pilot): ranking inverts — fine-tuned 2/96 vs baselines 42/96.
3. **Diagnosis (this experiment):** the deficit is NOT merely selector convention. Fine-tuning on
   synthetic data taught the model to hallucinate an entire fictional application contract —
   selectors *and* API routes *and* network calls *and* redirect targets *and* framework-API forms.
   - Prompt-injection grounding (suggest real selectors): **0 lift** — model overrides them.
   - Enforced selector-repair (force real selectors): **0 lift** — scripts still fail on the *other*
     hallucinated dimensions.
4. The only intervention that helped was **training-time grounding** (grounded-corpus retrain:
   convention bias eliminated 47/48 -> 0/48), because it addresses the root — the model's tendency
   to invent — rather than patching one symptom at inference time.

**Conclusion:** inference-time selector grounding is a dead end for these models; deployable
generation requires grounding the *whole* contract, best achieved by training on
execution-validated (grounded) data and/or an execution-feedback signal inside the correction
loop — not by post-hoc selector patching.
