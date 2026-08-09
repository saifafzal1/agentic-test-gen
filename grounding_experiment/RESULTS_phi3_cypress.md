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
