# Base Gemma 4 E4B — saucedemo control result

Zero-shot base Gemma 4 (no adapter), same ungrounded prompt as the fine-tuned pilot,
12 saucedemo stories, Cypress, executed live.

| Metric | Base Gemma 4 (zero-shot) | Fine-tuned Gemma 4 | GPT-4o-mini | Claude Haiku |
|--------|--------------------------|--------------------|-------------|--------------|
| Specs passing (of 12) | 0/12 | 0/12 | 9/12 | 8/12 |
| Status | 12/12 load_error (0 tests ran) | tests run then fail | pass | pass |
| Avg static score | 0.18 | 0.79 | high | high |
| describe()+it() structure | 0/12 | ~all | yes | yes |
| Output | echoed the prompt, not a script | valid structured Cypress | valid | valid |

## Refined finding (corrects "fine-tuning hurt execution")
- Base Gemma 4 E4B zero-shot CANNOT do the task: it echoes the prompt, produces no valid
  runnable Cypress (0.18 static, 0 structure, all load errors).
- Fine-tuning was NECESSARY to make a 4B model emit valid, well-structured, runnable scripts.
- But fine-tuning on the SYNTHETIC corpus grounded those scripts in a fictional app
  (data-testid, invented routes/APIs), so they run but fail on the real app.
- Commercial models (far larger) have both structure and real grounding -> 8-9/12.

Conclusion: the execution deficit is not misconfiguration and not simply "fine-tuning hurt it."
A 4B model needs fine-tuning to produce test scripts at all; the gap is that the fine-tuning
data was synthetic. The fix is fine-tuning on execution-validated (grounded) data, which would
give both structure AND real-app grounding.

Caveat: base instruct models are prompt-sensitive; a different prompt/decoding might coax more
from base Gemma. This run used the IDENTICAL prompt/decoding as the fine-tuned model (the fair
apples-to-apples comparison).
