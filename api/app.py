"""
MLOps Week 2 — FastAPI Inference Service
=========================================
Serves QLoRA fine-tuned models (Phi-3 + Gemma4) for Cypress/Playwright
test script generation.

Endpoints:
  GET  /health            → service liveness + loaded model info
  GET  /models            → registry of available model/framework combos
  POST /generate-test     → generate a test script from a user story

Hardware: Mac A — Apple M4 Pro, 48 GB RAM (MPS, bf16)
          Set MOCK_MODEL=true for CPU-only smoke testing (returns stub output).

Run (Mac A):
  cd ~/Documents/Dissertation/agentic-test-gen
  source .venv/bin/activate
  WARMUP_MODEL=phi3/cypress uvicorn api.app:app --host 0.0.0.0 --port 8000

Or via Docker (CPU, stub mode):
  docker run -e MOCK_MODEL=true -p 8000:8000 agentic-test-gen:latest
"""

import os
import re
import time
import torch
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────
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
        "model_class": "AutoModelForCausalLM",
        "trust_remote_code": False,
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
        "model_class": "Gemma4ForConditionalGeneration",
        "trust_remote_code": True,
    },
}

# ── System prompts (identical to fine-tuning) ─────────────────────────────────
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

# ── Device ────────────────────────────────────────────────────────────────────
device = "mps" if (not MOCK_MODEL and torch.backends.mps.is_available()) else "cpu"

# ── Model cache: (model_key, framework) → (tokenizer, model) ─────────────────
_model_cache: dict = {}


def _load_phi3(base_path: str, adapter_path: str):
    """Load Phi-3-mini + LoRA adapter on MPS/CPU."""
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

    # Unwrap Gemma4ClippableLinear → nn.Linear (required for PEFT inference)
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
    """Load and cache a model+adapter pair."""
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
        # Phi-3 ChatML template
        return (
            f"<|system|>\n{system}<|end|>\n"
            f"<|user|>\n{user}<|end|>\n"
            f"<|assistant|>\n"
        )
    else:
        # Gemma 4 chat template
        return (
            f"<start_of_turn>system\n{system}<end_of_turn>\n"
            f"<start_of_turn>user\n{user}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )


def _mock_generate(framework: str) -> tuple[str, float, int]:
    """Return a stub script for MOCK_MODEL mode (smoke/CI testing)."""
    if framework == "cypress":
        script = (
            "describe('Login', () => {\n"
            "  beforeEach(() => { cy.visit('/login'); });\n"
            "  it('logs in with valid credentials', () => {\n"
            "    cy.get('[data-testid=email]').type('user@example.com');\n"
            "    cy.get('[data-testid=password]').type('password123');\n"
            "    cy.get('[data-testid=submit]').click();\n"
            "    cy.url().should('include', '/dashboard');\n"
            "  });\n"
            "  it('shows error for invalid password', () => {\n"
            "    cy.get('[data-testid=email]').type('user@example.com');\n"
            "    cy.get('[data-testid=password]').type('wrong');\n"
            "    cy.get('[data-testid=submit]').click();\n"
            "    cy.contains('Invalid credentials').should('be.visible');\n"
            "  });\n"
            "});\n"
        )
    else:
        script = (
            "import { test, expect } from '@playwright/test';\n\n"
            "test.describe('Login', () => {\n"
            "  test.beforeEach(async ({ page }) => { await page.goto('/login'); });\n"
            "  test('logs in with valid credentials', async ({ page }) => {\n"
            "    await page.getByLabel('Email').fill('user@example.com');\n"
            "    await page.getByLabel('Password').fill('password123');\n"
            "    await page.getByRole('button', { name: 'Sign in' }).click();\n"
            "    await expect(page).toHaveURL('/dashboard');\n"
            "  });\n"
            "  test('shows error for invalid password', async ({ page }) => {\n"
            "    await page.getByLabel('Email').fill('user@example.com');\n"
            "    await page.getByLabel('Password').fill('wrong');\n"
            "    await page.getByRole('button', { name: 'Sign in' }).click();\n"
            "    await expect(page.getByText('Invalid credentials')).toBeVisible();\n"
            "  });\n"
            "});\n"
        )
    return script, 0.042, len(script.split())


# ── Schemas ───────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    user_story    : str = Field(..., min_length=10, description="Jira user story text")
    framework     : str = Field("cypress", pattern="^(cypress|playwright)$")
    model_key     : str = Field("phi3",    pattern="^(phi3|gemma4)$")
    category      : str = Field("functional")
    complexity    : str = Field("medium")
    max_new_tokens: int = Field(512, ge=64, le=1024)

class GenerateResponse(BaseModel):
    script          : str
    model           : str
    framework       : str
    latency_s       : float
    tokens_generated: int
    mock            : bool = False

class HealthResponse(BaseModel):
    status       : str
    device       : str
    mock_mode    : bool
    cached_models: list[str]

class ModelInfo(BaseModel):
    key        : str
    model_key  : str
    framework  : str
    display    : str
    hf_adapter : str
    loaded     : bool


# ── Lifespan: optional warm-up ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup = os.environ.get("WARMUP_MODEL", "")   # e.g. "phi3/cypress"
    if warmup and not MOCK_MODEL:
        parts = warmup.split("/")
        mk, fw = (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "cypress")
        if mk in MODEL_REGISTRY and fw in ("cypress", "playwright"):
            load_model(mk, fw)
    yield
    _model_cache.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       ="Agentic Test Generator API",
    description ="QLoRA fine-tuned Cypress & Playwright test script generation",
    version     ="1.0.0",
    lifespan    =lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(
        status        ="ok",
        device        =device,
        mock_mode     =MOCK_MODEL,
        cached_models =[f"{mk}/{fw}" for mk, fw in _model_cache],
    )


@app.get("/models", tags=["ops"])
def list_models() -> dict:
    return {
        "models": [
            ModelInfo(
                key        =f"{mk}-{fw}",
                model_key  =mk,
                framework  =fw,
                display    =MODEL_REGISTRY[mk]["display"],
                hf_adapter =MODEL_REGISTRY[mk]["hf_adapters"][fw],
                loaded     =(mk, fw) in _model_cache,
            )
            for mk in MODEL_REGISTRY
            for fw in ["cypress", "playwright"]
        ]
    }


@app.post("/generate-test", response_model=GenerateResponse, tags=["inference"])
def generate_test(req: GenerateRequest):
    # MOCK mode — return stub for Docker/CI smoke testing
    if MOCK_MODEL:
        load_model(req.model_key, req.framework)   # registers in cache
        script, latency, tokens = _mock_generate(req.framework)
        return GenerateResponse(
            script=script,
            model=MODEL_REGISTRY[req.model_key]["display"],
            framework=req.framework,
            latency_s=latency,
            tokens_generated=tokens,
            mock=True,
        )

    tokenizer, model = load_model(req.model_key, req.framework)
    prompt = _build_prompt(req.model_key, req.framework,
                           req.user_story, req.category, req.complexity)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens   =req.max_new_tokens,
            do_sample        =False,
            temperature      =1.0,
            pad_token_id     =tokenizer.eos_token_id,
        )
    latency = time.time() - t0

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    script = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Strip any trailing stop tokens that slipped through
    script = re.sub(r"<\|end\|>|<end_of_turn>", "", script).strip()

    return GenerateResponse(
        script          =script,
        model           =MODEL_REGISTRY[req.model_key]["display"],
        framework       =req.framework,
        latency_s       =round(latency, 3),
        tokens_generated=len(new_tokens),
        mock            =False,
    )
