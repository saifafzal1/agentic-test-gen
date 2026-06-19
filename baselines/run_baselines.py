"""
Phase 3 — Baseline Evaluation
==============================
Generates test scripts from all 279 user stories using 3 LLM APIs × 2 frameworks = 6 runs.

Models:
  - GPT-4o-mini      (OpenAI)
  - Claude Haiku     (Anthropic claude-haiku-4-5)
  - Gemini 1.5 Flash (Google)

Frameworks:
  - Cypress    → baselines/results/cypress/{model}/
  - Playwright → baselines/results/playwright/{model}/

Usage:
  cd ~/dissertation-project          # or agentic-test-gen on Mac A
  source .venv/bin/activate
  python -u baselines/run_baselines.py --model gpt4o-mini --framework cypress
  python -u baselines/run_baselines.py --model claude-haiku --framework cypress
  python -u baselines/run_baselines.py --model gemini-flash --framework cypress
  python -u baselines/run_baselines.py --model gpt4o-mini --framework playwright
  python -u baselines/run_baselines.py --model claude-haiku --framework playwright
  python -u baselines/run_baselines.py --model gemini-flash --framework playwright

  # Or run all 6 at once (sequential):
  python -u baselines/run_baselines.py --all

Output per record:
  baselines/results/{framework}/{model}/{id}.json
  {
    "id": "TC_001",
    "category": "authentication",
    "complexity": "medium",
    "user_story": "...",
    "framework": "cypress",
    "model": "gpt4o-mini",
    "generated_script": "...",
    "prompt_tokens": 123,
    "completion_tokens": 456,
    "latency_s": 2.1,
    "timestamp": "2026-05-10T..."
  }

Summary saved to: baselines/results/{framework}/{model}/summary.json
"""

import os, json, time, argparse, traceback
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Args ─────────────────────────────────────────────────────────────────────
MODELS     = ["gpt4o-mini", "claude-haiku", "gemini-flash"]
FRAMEWORKS = ["cypress", "playwright"]

parser = argparse.ArgumentParser()
parser.add_argument("--model",     choices=MODELS,     help="Model to run")
parser.add_argument("--framework", choices=FRAMEWORKS, help="Framework to generate")
parser.add_argument("--all",       action="store_true", help="Run all 6 combinations sequentially")
parser.add_argument("--limit",     type=int, default=None, help="Limit to first N records (for testing)")
args = parser.parse_args()

if not args.all and (not args.model or not args.framework):
    parser.error("Provide --model and --framework, or --all")

RUNS = [(m, f) for m in MODELS for f in FRAMEWORKS] if args.all else [(args.model, args.framework)]

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# ── System prompts ────────────────────────────────────────────────────────────
SYSTEM_CY = (
    "You are a senior QA automation engineer specialising in Cypress. "
    "Given a Jira user story, generate a complete, production-quality Cypress test script. "
    "Return ONLY the raw Cypress JavaScript — no markdown fences, no explanations. "
    "Include describe(), beforeEach(), and at least 2 it() blocks (happy path + edge case)."
)

SYSTEM_PW = (
    "You are a senior QA automation engineer specialising in Playwright. "
    "Given a Jira user story, generate a complete, production-quality Playwright TypeScript test script. "
    "Return ONLY the raw Playwright TypeScript — no markdown fences, no explanations. "
    "Use test.describe(), test.beforeEach(), at least 2 test() blocks. "
    "Use getByRole(), getByLabel(), or getByText() locators — never page.locator()."
)

def build_user_prompt(rec: dict, framework: str) -> str:
    return (
        f'User story: "{rec["user_story"]}"\n'
        f'Category: {rec["category"]}\n'
        f'Complexity: {rec.get("complexity", "medium")}\n'
        f'Return the {framework.capitalize()} script only.'
    )

# ── API clients (lazy init per run) ──────────────────────────────────────────
def make_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def make_anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def make_gemini_client():
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    return genai

# ── Generation functions ──────────────────────────────────────────────────────
def generate_openai(client, system: str, user: str) -> dict:
    t0 = time.time()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    return {
        "generated_script":    resp.choices[0].message.content.strip(),
        "prompt_tokens":       resp.usage.prompt_tokens,
        "completion_tokens":   resp.usage.completion_tokens,
        "latency_s":           round(time.time() - t0, 2),
    }

def generate_anthropic(client, system: str, user: str) -> dict:
    t0 = time.time()
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=0.2,
    )
    return {
        "generated_script":  resp.content[0].text.strip(),
        "prompt_tokens":     resp.usage.input_tokens,
        "completion_tokens": resp.usage.output_tokens,
        "latency_s":         round(time.time() - t0, 2),
    }

def generate_gemini(genai_module, system: str, user: str) -> dict:
    t0 = time.time()
    model = genai_module.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system,
        generation_config={"temperature": 0.2, "max_output_tokens": 2048},
    )
    resp = model.generate_content(user)
    # Gemini doesn't always expose token counts in the same way
    pt = getattr(resp.usage_metadata, "prompt_token_count",    0) if hasattr(resp, "usage_metadata") else 0
    ct = getattr(resp.usage_metadata, "candidates_token_count", 0) if hasattr(resp, "usage_metadata") else 0
    return {
        "generated_script":  resp.text.strip(),
        "prompt_tokens":     pt,
        "completion_tokens": ct,
        "latency_s":         round(time.time() - t0, 2),
    }

# ── Main runner ───────────────────────────────────────────────────────────────
def run_single(model_key: str, framework: str):
    print(f"\n{'='*60}")
    print(f"🚀 Run: {model_key.upper()} × {framework.upper()}")
    print(f"{'='*60}")

    # Load dataset
    data_file = DATA_DIR / f"{framework}_dataset.jsonl"
    records = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[:args.limit]
        print(f"   ⚠️  Limited to first {args.limit} records")

    print(f"   Dataset : {len(records)} records from {data_file.name}")

    # Output dir
    out_dir = RESULTS_DIR / framework / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    # System prompt
    system = SYSTEM_CY if framework == "cypress" else SYSTEM_PW

    # Init client
    if model_key == "gpt4o-mini":
        client = make_openai_client()
        gen_fn = lambda s, u: generate_openai(client, s, u)
    elif model_key == "claude-haiku":
        client = make_anthropic_client()
        gen_fn = lambda s, u: generate_anthropic(client, s, u)
    else:  # gemini-flash
        genai_mod = make_gemini_client()
        gen_fn = lambda s, u: generate_gemini(genai_mod, s, u)

    # Check for existing results (resume support)
    existing = {p.stem for p in out_dir.glob("*.json") if p.stem != "summary"}
    print(f"   Resuming: {len(existing)} already done, {len(records) - len(existing)} remaining")

    results  = []
    skipped  = 0
    errors   = []
    total_pt = 0
    total_ct = 0
    total_lt = 0.0

    for i, rec in enumerate(records, 1):
        rec_id = rec["id"]

        # Load existing result if already done
        existing_path = out_dir / f"{rec_id}.json"
        if rec_id in existing:
            with open(existing_path) as f:
                results.append(json.load(f))
            continue

        user_prompt = build_user_prompt(rec, framework)

        # Retry up to 3× with exponential backoff
        last_error = None
        for attempt in range(1, 4):
            try:
                gen = gen_fn(system, user_prompt)
                break
            except Exception as e:
                last_error = str(e)
                wait = 2 ** attempt
                print(f"   ⚠️  {rec_id} attempt {attempt} failed: {e} — retry in {wait}s")
                time.sleep(wait)
        else:
            print(f"   ❌ Skipping {rec_id} after 3 attempts: {last_error}")
            skipped += 1
            errors.append({"id": rec_id, "error": last_error})
            continue

        result = {
            "id":               rec_id,
            "category":         rec["category"],
            "complexity":       rec.get("complexity", "medium"),
            "user_story":       rec["user_story"],
            "framework":        framework,
            "model":            model_key,
            "generated_script": gen["generated_script"],
            "prompt_tokens":    gen["prompt_tokens"],
            "completion_tokens":gen["completion_tokens"],
            "latency_s":        gen["latency_s"],
            "timestamp":        datetime.utcnow().isoformat(),
        }

        with open(existing_path, "w") as f:
            json.dump(result, f, indent=2)

        results.append(result)
        total_pt += gen["prompt_tokens"]
        total_ct += gen["completion_tokens"]
        total_lt += gen["latency_s"]

        # Progress every 10
        if i % 10 == 0:
            print(f"   [{i:3d}/{len(records)}] {rec_id} — {gen['latency_s']:.1f}s  "
                  f"(tokens: {gen['prompt_tokens']}+{gen['completion_tokens']})")

        # Rate limit buffer
        time.sleep(0.3)

    # Summary
    generated_count = len(records) - skipped - len(existing)
    summary = {
        "model":            model_key,
        "framework":        framework,
        "total_records":    len(records),
        "generated":        len(results),
        "skipped":          skipped,
        "resumed":          len(existing),
        "total_prompt_tokens":     total_pt,
        "total_completion_tokens": total_ct,
        "avg_latency_s":           round(total_lt / max(generated_count, 1), 2),
        "errors":           errors,
        "timestamp":        datetime.utcnow().isoformat(),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Done: {model_key.upper()} × {framework.upper()}")
    print(f"   Generated : {len(results)} | Skipped: {skipped}")
    print(f"   Avg latency : {summary['avg_latency_s']:.2f}s")
    print(f"   Total tokens: {total_pt:,} prompt + {total_ct:,} completion")
    print(f"   Results → {out_dir}")

    return summary

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_summaries = []
    for model_key, framework in RUNS:
        summary = run_single(model_key, framework)
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        print(f"\n{'='*60}")
        print("📊 All runs complete:")
        for s in all_summaries:
            print(f"   {s['model']:15s} × {s['framework']:10s} → "
                  f"{s['generated']}/{s['total_records']} generated, "
                  f"avg {s['avg_latency_s']:.1f}s/call")
