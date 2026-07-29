"""
In-process model loading + generation for the BMAD agentic loop.
==================================================================
Phase 6 (rebuild): loads the fine-tuned Phi-3/Gemma4 QLoRA adapters and calls
model.generate() directly in the same process as the BMAD loop — no FastAPI
server, no HTTP, no port. The previous implementation (phase6-eval branch) ran
a separate uvicorn server the loop talked to over localhost:8000; every crash
hit there (port conflicts, dropped connections, read timeouts, a retry-logic
bug) came from that network layer, not from the model itself. Removing it
removes that entire class of failure.

Hardware: Mac A — Apple M4 Pro, 48 GB RAM (MPS, bf16).
          Set MOCK_MODEL=true for CPU-only smoke testing (returns stub output).
"""

import gc
import os
import time
from pathlib import Path

import torch

BASE_DIR = Path(__file__).parent.parent   # repo root

MOCK_MODEL = os.environ.get("MOCK_MODEL", "false").lower() == "true"

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "phi3": {
        "display"   : "microsoft/Phi-3-mini-4k-instruct",
        "base_path" : str(BASE_DIR / "fine_tuning" / "phi3-base-model"),
        "adapters"  : {
            "cypress"   : str(BASE_DIR / "fine_tuning" / "phi3-cypress"),
            "playwright": str(BASE_DIR / "fine_tuning" / "phi3-playwright"),
        },
        "hf_adapters": {
            "cypress"   : "saifafzal1/phi3-mini-cypress-qlora",
            "playwright": "saifafzal1/phi3-mini-playwright-qlora",
        },
    },
    "gemma4": {
        "display"   : "google/gemma-3-4b-it",
        "base_path" : str(BASE_DIR / "fine_tuning" / "gemma4-base-model"),
        "adapters"  : {
            "cypress"   : str(BASE_DIR / "fine_tuning" / "gemma4-cypress"),
            "playwright": str(BASE_DIR / "fine_tuning" / "gemma4-playwright"),
        },
        "hf_adapters": {
            "cypress"   : "saifafzal1/gemma4-E4B-cypress-qlora",
            "playwright": "saifafzal1/gemma4-E4B-playwright-qlora",
        },
    },
}

SYSTEM_PROMPTS = {
    "cypress": (
        "You are a senior QA automation engineer specialising in Cypress. "
        "Given a Jira user story, generate a complete, production-quality Cypress test script. "
        "Return ONLY the raw Cypress JavaScript — no markdown fences, no explanations. "
        "Include describe(), beforeEach(), and at least 2 it() blocks (happy path + edge case)."
    ),
    "playwright": (
        "You are a senior QA automation engineer specialising in Playwright. "
        "Given a Jira user story, generate a complete, production-quality Playwright TypeScript test script. "
        "Return ONLY the raw Playwright TypeScript — no markdown fences, no explanations. "
        "Use test.describe(), test.beforeEach(), at least 2 test() blocks. "
        "Use getByRole(), getByLabel(), or getByText() locators — never page.locator()."
    ),
}

# Chat-template turn markers that must never appear in the emitted script. If
# generation doesn't stop cleanly at EOS, it can run on into a hallucinated
# next turn — truncate at the first sign of that instead of just stripping
# the marker text (which would leave the hallucinated continuation in place).
STOP_MARKERS = ["<start_of_turn>", "<end_of_turn>", "<|user|>", "<|system|>", "<|assistant|>", "<|end|>"]

def _select_device() -> str:
    """Prefer CUDA (Linux/Docker GPU host) → MPS (Apple Silicon) → CPU."""
    if MOCK_MODEL:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = _select_device()

_model_cache: dict = {}


def _truncate_at_stop_marker(script: str) -> str:
    cut = len(script)
    for marker in STOP_MARKERS:
        idx = script.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return script[:cut].strip()


def _stop_token_ids(model_key: str, tokenizer) -> list:
    """EOS ids to pass to generate() so it actually halts at the model's turn boundary."""
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    extra_token = "<end_of_turn>" if model_key == "gemma4" else "<|end|>"
    extra_id = tokenizer.convert_tokens_to_ids(extra_token)
    if isinstance(extra_id, int) and extra_id >= 0 and extra_id != tokenizer.unk_token_id:
        ids.add(extra_id)
    return list(ids)


def _load_phi3(base_path: str, adapter_path: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="eager",
    )
    base = base.to(device)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tokenizer, model


def _load_gemma4(base_path: str, adapter_path: str):
    """Load Gemma4 + LoRA adapter — unwraps ClippableLinear first."""
    from transformers import AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = Gemma4ForConditionalGeneration.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base = base.to(device)

    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
        replaced = 0
        for mod_name, module in list(base.named_modules()):
            if isinstance(module, Gemma4ClippableLinear) and "language_model" in mod_name:
                parts = mod_name.split(".")
                parent = base
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], module.linear)
                replaced += 1
        print(f"   Unwrapped {replaced} Gemma4ClippableLinear → nn.Linear")
    except (ImportError, AttributeError) as e:
        print(f"   ⚠️  ClippableLinear unwrap skipped: {e}")

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tokenizer, model


def load_model(model_key: str, framework: str):
    """Load and cache a model+adapter pair for this process's lifetime."""
    cache_key = (model_key, framework)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if MOCK_MODEL:
        _model_cache[cache_key] = (None, None)
        return None, None

    cfg = MODEL_REGISTRY[model_key]
    adapter_path = cfg["adapters"][framework]
    print(f"⏳ Loading {model_key}/{framework} on {device}...")

    if model_key == "phi3":
        result = _load_phi3(cfg["base_path"], adapter_path)
    else:
        result = _load_gemma4(cfg["base_path"], adapter_path)

    _model_cache[cache_key] = result
    print(f"✅ {model_key}/{framework} ready")
    return result


def _build_prompt(model_key: str, framework: str,
                   user_story: str, category: str, complexity: str) -> str:
    """Build the inference prompt matching the fine-tuning template exactly."""
    system = SYSTEM_PROMPTS[framework]
    user = (
        f'User story: "{user_story}"\n'
        f'Category: {category}\n'
        f'Complexity: {complexity}\n'
        f'Return the {framework.capitalize()} script only.'
    )
    if model_key == "phi3":
        return (
            f"<|system|>\n{system}<|end|>\n"
            f"<|user|>\n{user}<|end|>\n"
            f"<|assistant|>\n"
        )
    else:
        return (
            f"<start_of_turn>system\n{system}<end_of_turn>\n"
            f"<start_of_turn>user\n{user}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )


def _mock_generate(framework: str) -> str:
    """Stub script for MOCK_MODEL mode (smoke testing without loading real weights)."""
    if framework == "cypress":
        return (
            "describe('Login', () => {\n"
            "  beforeEach(() => { cy.visit('/login'); });\n"
            "  it('logs in with valid credentials', () => {\n"
            "    cy.get('[data-testid=email]').type('user@example.com');\n"
            "    cy.get('[data-testid=submit]').click();\n"
            "    cy.url().should('include', '/dashboard');\n"
            "  });\n"
            "  it('shows error for invalid password', () => {\n"
            "    cy.get('[data-testid=submit]').click();\n"
            "    cy.contains('Invalid credentials').should('be.visible');\n"
            "  });\n"
            "});\n"
        )
    return (
        "import { test, expect } from '@playwright/test';\n\n"
        "test.describe('Login', () => {\n"
        "  test.beforeEach(async ({ page }) => { await page.goto('/login'); });\n"
        "  test('logs in with valid credentials', async ({ page }) => {\n"
        "    await page.getByLabel('Email').fill('user@example.com');\n"
        "    await expect(page).toHaveURL('/dashboard');\n"
        "  });\n"
        "  test('shows error for invalid password', async ({ page }) => {\n"
        "    await expect(page.getByText('Invalid credentials')).toBeVisible();\n"
        "  });\n"
        "});\n"
    )


def generate(model_key: str, framework: str, user_story: str,
             category: str, complexity: str, max_new_tokens: int) -> tuple:
    """
    Generate a test script in-process (no HTTP). Returns (script, latency_s).
    """
    if MOCK_MODEL:
        load_model(model_key, framework)
        t0 = time.time()
        script = _mock_generate(framework)
        return script, round(time.time() - t0, 3)

    tokenizer, model = load_model(model_key, framework)
    prompt = _build_prompt(model_key, framework, user_story, category, complexity)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=_stop_token_ids(model_key, tokenizer),
        )
    latency = time.time() - t0

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    script = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Safety net: truncate at the first leaked turn marker even if EOS was missed
    script = _truncate_at_stop_marker(script)

    # MPS doesn't release intermediate allocations between calls as eagerly as
    # CUDA; over hundreds of sequential generate() calls in one long-lived
    # process this fragments/grows memory until the OS kills the process.
    del inputs, output_ids, new_tokens
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return script, round(latency, 3)
