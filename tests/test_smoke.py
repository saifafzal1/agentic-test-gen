"""
Week 3 Smoke Tests — FastAPI Inference Service
================================================
Tests all three API endpoints against a running service.

Target:
  - Docker container in MOCK_MODEL=true mode (default, Mac B CI)
  - Native FastAPI on Mac A over LAN (real model inference)

Usage:
  # Docker (Mac B) — start container first:
  #   docker run -d -e MOCK_MODEL=true -p 8000:8000 agentic-test-gen:latest
  pytest tests/test_smoke.py -v

  # Native Mac A over LAN:
  API_URL=http://192.168.x.x:8000 pytest tests/test_smoke.py -v

  # Specific test only:
  pytest tests/test_smoke.py::test_health -v
"""

import os
import pytest
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TIMEOUT_HEALTH = 10
TIMEOUT_GEN    = 120   # real model inference can take up to ~60s cold


# ── /health ───────────────────────────────────────────────────────────────────
def test_health_returns_200():
    resp = requests.get(f"{API_URL}/health", timeout=TIMEOUT_HEALTH)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


def test_health_schema():
    resp = requests.get(f"{API_URL}/health", timeout=TIMEOUT_HEALTH)
    data = resp.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert "mock_mode" in data
    assert isinstance(data["cached_models"], list)


# ── /models ───────────────────────────────────────────────────────────────────
def test_models_returns_200():
    resp = requests.get(f"{API_URL}/models", timeout=TIMEOUT_HEALTH)
    assert resp.status_code == 200


def test_models_schema():
    resp = requests.get(f"{API_URL}/models", timeout=TIMEOUT_HEALTH)
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) == 4


def test_models_contains_all_variants():
    resp = requests.get(f"{API_URL}/models", timeout=TIMEOUT_HEALTH)
    keys = {m["key"] for m in resp.json()["models"]}
    assert keys == {"phi3-cypress", "phi3-playwright", "gemma4-cypress", "gemma4-playwright"}


def test_models_have_hf_adapter_links():
    resp = requests.get(f"{API_URL}/models", timeout=TIMEOUT_HEALTH)
    for m in resp.json()["models"]:
        assert m["hf_adapter"].startswith("saifafzal1/"), \
            f"Unexpected adapter URL: {m['hf_adapter']}"


# ── /generate-test (valid requests) ──────────────────────────────────────────
@pytest.mark.parametrize("framework,model_key", [
    ("cypress",    "phi3"),
    ("playwright", "phi3"),
])
def test_generate_returns_script(framework, model_key):
    payload = {
        "user_story"    : "As a user, I want to log in with valid credentials so I can access my dashboard.",
        "framework"     : framework,
        "model_key"     : model_key,
        "category"      : "authentication",
        "complexity"    : "medium",
        "max_new_tokens": 256,
    }
    resp = requests.post(f"{API_URL}/generate-test", json=payload, timeout=TIMEOUT_GEN)
    assert resp.status_code == 200, f"POST /generate-test failed: {resp.text}"
    data = resp.json()
    assert "script" in data
    assert len(data["script"]) > 50, "Script too short — likely empty generation"
    assert data["framework"] == framework
    assert data["tokens_generated"] > 0
    assert data["latency_s"] >= 0


def test_generate_cypress_script_looks_like_cypress():
    payload = {
        "user_story"    : "As a user, I want to search for a product by name.",
        "framework"     : "cypress",
        "model_key"     : "phi3",
        "max_new_tokens": 256,
    }
    resp = requests.post(f"{API_URL}/generate-test", json=payload, timeout=TIMEOUT_GEN)
    script = resp.json()["script"]
    # In mock mode the stub always contains cypress keywords
    assert any(kw in script for kw in ["cy.", "describe(", "it("]), \
        f"Cypress script missing expected keywords:\n{script[:300]}"


def test_generate_playwright_script_looks_like_playwright():
    payload = {
        "user_story"    : "As a user, I want to add an item to my shopping cart.",
        "framework"     : "playwright",
        "model_key"     : "phi3",
        "max_new_tokens": 256,
    }
    resp = requests.post(f"{API_URL}/generate-test", json=payload, timeout=TIMEOUT_GEN)
    script = resp.json()["script"]
    assert any(kw in script for kw in ["page.", "test(", "expect("]), \
        f"Playwright script missing expected keywords:\n{script[:300]}"


# ── /generate-test (validation errors) ───────────────────────────────────────
def test_generate_rejects_invalid_framework():
    payload = {
        "user_story": "As a user, I want to reset my password.",
        "framework" : "selenium",   # not allowed
        "model_key" : "phi3",
    }
    resp = requests.post(f"{API_URL}/generate-test", json=payload, timeout=TIMEOUT_HEALTH)
    assert resp.status_code == 422


def test_generate_rejects_invalid_model_key():
    payload = {
        "user_story": "As a user, I want to view my order history.",
        "framework" : "cypress",
        "model_key" : "gpt4",   # not allowed
    }
    resp = requests.post(f"{API_URL}/generate-test", json=payload, timeout=TIMEOUT_HEALTH)
    assert resp.status_code == 422


def test_generate_rejects_short_user_story():
    payload = {
        "user_story": "login",   # < 10 chars
        "framework" : "cypress",
        "model_key" : "phi3",
    }
    resp = requests.post(f"{API_URL}/generate-test", json=payload, timeout=TIMEOUT_HEALTH)
    assert resp.status_code == 422


def test_generate_rejects_missing_user_story():
    payload = {
        "framework": "cypress",
        "model_key": "phi3",
    }
    resp = requests.post(f"{API_URL}/generate-test", json=payload, timeout=TIMEOUT_HEALTH)
    assert resp.status_code == 422
