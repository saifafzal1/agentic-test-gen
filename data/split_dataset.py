"""
Dataset Splitter — Framework Isolation
=======================================
Splits dataset_final.json (279 combined pairs) into two framework-isolated
JSONL files as required by the dissertation outline:

  data/cypress_dataset.jsonl    — 279 records, cypress_script only
  data/playwright_dataset.jsonl — 279 records, playwright_script only

Each JSONL record contains:
  id, category, complexity, user_story, framework, script

A separate `text` field is included with the full instruction-tuning
conversation using a model-agnostic template (adapted per model at training).

Run:
  cd ~/Documents/Dissertation/agentic-test-gen
  python data/split_dataset.py
"""

import json
from pathlib import Path
from collections import Counter

SRC  = Path(__file__).parent / "dataset_final.json"
CY   = Path(__file__).parent / "cypress_dataset.jsonl"
PW   = Path(__file__).parent / "playwright_dataset.jsonl"

# ── System prompts per framework ───────────────────────────────────────────
SYSTEM_CY = (
    "You are a senior QA automation engineer specialising in Cypress. "
    "Given a Jira user story, generate a complete, production-quality Cypress test script. "
    "Return ONLY the raw Cypress JavaScript — no markdown fences, no explanations. "
    "Include describe(), beforeEach(), and at least 2 it() blocks (happy path + edge case)."
)

SYSTEM_PW = (
    "You are a senior QA automation engineer specialising in Playwright. "
    "Given a Jira user story, generate a complete, production-quality Playwright test script. "
    "Return ONLY the raw Playwright TypeScript — no markdown fences, no explanations. "
    "Use test.describe(), test.beforeEach(), at least 2 test() blocks. "
    "Use getByRole(), getByLabel(), or getByText() locators — never page.locator()."
)

USER_TEMPLATE = 'User story: "{story}"\nCategory: {category}\nComplexity: {complexity}\nReturn the script only.'

def make_text(system: str, story: str, category: str, complexity: str, script: str) -> str:
    """Gemma 4 / Phi-3 agnostic conversation format (placeholder tokens replaced at training time)."""
    user = USER_TEMPLATE.format(story=story, category=category, complexity=complexity)
    # Generic format — finetune scripts apply model-specific chat template on top
    return f"<SYSTEM>{system}</SYSTEM>\n<USER>{user}</USER>\n<ASSISTANT>{script}</ASSISTANT>"

# ── Load source ─────────────────────────────────────────────────────────────
with open(SRC) as f:
    pairs = json.load(f)

print(f"\n📂 Loaded {len(pairs)} pairs from {SRC.name}")

cy_records, pw_records = [], []

for p in pairs:
    base = {
        "id":         p["id"],
        "category":   p["category"],
        "complexity":  p.get("complexity", "medium"),
        "user_story": p["user_story"],
    }

    # ── Cypress record ──────────────────────────────────────────────────────
    cy_rec = {
        **base,
        "framework": "cypress",
        "script":    p["cypress_script"],
        "text":      make_text(SYSTEM_CY, p["user_story"], p["category"],
                               p.get("complexity","medium"), p["cypress_script"]),
    }
    cy_records.append(cy_rec)

    # ── Playwright record ───────────────────────────────────────────────────
    pw_rec = {
        **base,
        "framework": "playwright",
        "script":    p["playwright_script"],
        "text":      make_text(SYSTEM_PW, p["user_story"], p["category"],
                               p.get("complexity","medium"), p["playwright_script"]),
    }
    pw_records.append(pw_rec)

# ── Write JSONL ─────────────────────────────────────────────────────────────
def write_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

write_jsonl(cy_records, CY)
write_jsonl(pw_records, PW)

# ── Validation report ────────────────────────────────────────────────────────
def report(name, records, script_key="script"):
    scripts = [r[script_key] for r in records]
    lens    = [len(s) for s in scripts]
    cats    = Counter(r["category"] for r in records)
    cmpx    = Counter(r["complexity"] for r in records)

    print(f"\n{'─'*55}")
    print(f"  {name}  ({len(records)} records)")
    print(f"{'─'*55}")
    print(f"  Script length  avg={sum(lens)/len(lens):.0f}  min={min(lens)}  max={max(lens)}")
    print(f"  Complexity:  {dict(sorted(cmpx.items()))}")
    print(f"  Category distribution (min={min(cats.values())} max={max(cats.values())}):")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        bar = "█" * cnt + "░" * (21 - cnt)
        print(f"    {cat:<28} {cnt:>3}  {bar}")

report("cypress_dataset.jsonl", cy_records)
report("playwright_dataset.jsonl", pw_records)

print(f"\n{'='*55}")
print(f"  ✅ Cypress    → {CY}")
print(f"  ✅ Playwright → {PW}")
print(f"\n  Note: 279 pairs (outline target: 280).")
print(f"  Generate 1 additional pair per framework to reach 280 if required.")
print(f"{'='*55}\n")
