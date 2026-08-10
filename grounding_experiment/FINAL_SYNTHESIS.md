# Application-Grounding Investigation — Final Synthesis

## The question
Can inference-time intervention make the fine-tuned SLMs' generated scripts actually execute
against a live application, given they scored ~0 in the dissertation's execution pilot?

## Four interventions tested (Phi-3 / Cypress, 24 grounded stories, live saucedemo + the-internet)

| Intervention | What it does | Specs fully passing | Individual tests |
|--------------|--------------|--------------------|------------------|
| Static loop (pilot, OFF) | ungrounded generation | 0/24 | 0/40 |
| Grounding ON | inject real selectors into the prompt (suggested) | 0/24 | 0/44 |
| Selector-repair | rewrite invented selectors to real ones (enforced) | 0/24 | 0/47 |
| Execution-feedback loop | run live each iteration, feed real runtime errors back | 0/24 | 0/44 |

Gemma 4 / Cypress generality check (grounding OFF/ON/repaired): 0/24 as well.

Execution-feedback diagnostic: in **0/24** stories did the number of passing tests increase across
the up-to-4 feedback iterations — the concrete runtime errors ("data-testid never found",
"loginRequest never occurred", "expected /dashboard got /inventory.html") did not steer the model
toward repair.

## Conclusion
No inference-time intervention — suggesting real selectors, enforcing them, or closing the loop
with live execution feedback — recovers execution for these 4B fine-tuned SLMs. The deficit is not
selector-convention alone: fine-tuning on synthetic reference scripts taught the models to generate
an entire self-consistent but fictional application contract (selectors + routes + network calls +
redirects), and they reproduce it even when handed the real DOM and the exact runtime errors.

The only intervention that moved the metric was **training-time grounding** (the grounded-corpus
retrain, which eliminated the data-testid convention bias 47/48 -> 0/48 on unseen apps). This points
the productive path squarely at the training data / objective, not at inference-time repair.

## Contrast (what DOES execute on these apps)
- Commercial baselines: GPT-4o-mini 15/24, Claude Haiku 13/24 specs fully passing.
- Execution-verified grounded corpus: 173 scripts (all admitted only if they passed live).
So the harness and the task are sound; the gap is specific to the synthetic-fine-tuned SLMs.

## Recommended next direction
Train on execution-validated (grounded) data at scale — i.e. build the corpus (pipeline already
exists) large enough to fine-tune on, then re-run this same evaluation. This addresses the root
(the models' tendency to invent) rather than patching symptoms at inference.
