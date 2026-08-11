# Qwen2.5-Coder-14B — saucedemo result (the capstone)

Zero-shot, via Ollama, 12 saucedemo stories, executed live. No fine-tuning.

    specs fully passing:  10/12   (1 partial)
    individual tests:     15/17
    scripts using wrong data-testid convention: 0/12  (all used the REAL selectors)

## Full model comparison on saucedemo (12 stories, Cypress)

| Model | Size | Fine-tuned? | Valid scripts? | Real selectors? | Specs passing |
|-------|------|-------------|----------------|-----------------|---------------|
| Base Gemma 4 E4B | 4B | no | no (echoes prompt) | - | 0/12 |
| Fine-tuned Gemma 4 / Phi-3 | 4B | yes (synthetic) | yes | no (fictional) | 0/12 |
| Claude Haiku | large | no | yes | yes | 8/12 |
| GPT-4o-mini | large | no | yes | yes | 9/12 |
| **Qwen2.5-Coder-14B** | **14B** | **no** | **yes** | **yes** | **10/12** |

## Conclusion — the whole story resolves
The execution deficit was never misconfiguration and never "open-weight vs closed". Two factors
decide real-app execution:
1. **Model capability** — a 4B base model can't produce valid test scripts at all; a 14B
   code-specialised model does so zero-shot and beats the commercial baselines.
2. **Grounding of the training signal** — fine-tuning a 4B model on SYNTHETIC references made it
   produce valid-but-fictional scripts (0/12); a capable model with no such fine-tuning uses the
   real selectors and passes.

A sufficiently capable open code model (Qwen2.5-Coder-14B), zero-shot, is the strongest performer
on real execution — better than GPT-4o-mini and Claude Haiku here. The dissertation's fine-tuning
optimised static similarity to synthetic references at the expense of real-world execution; for
deployable, executable test generation the better lever is a larger code model (and, if fine-
tuning, grounded/executable training data), not a small model tuned on synthetic scripts.
