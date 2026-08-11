"""
Run Qwen2.5-Coder (via Ollama) zero-shot on the 12 saucedemo stories and execute
against the live site. Additional open-weight point: larger (7B/14B), code-
specialised, NOT fine-tuned on the synthetic corpus -> no data-testid bias, and
capable enough to produce valid scripts (unlike base Gemma 4).

Prereq: `ollama serve` running with the model registered (default qwen-coder-14b).

Usage: python -m grounding_experiment.qwen_saucedemo --model qwen-coder-14b
"""
import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "execution_harness"))
from run_execution import run_cypress, CY_SPEC_DIR   # noqa: E402

STORIES = REPO / "data" / "execution_validation" / "grounded_stories.jsonl"
OLLAMA = "http://localhost:11434/api/chat"

SYSTEM_CY = (
    "You are a senior QA automation engineer specialising in Cypress. "
    "Given a Jira user story, generate a complete, production-quality Cypress test script. "
    "Return ONLY the raw Cypress JavaScript — no markdown fences, no explanations. "
    "Include describe(), beforeEach(), and at least 2 it() blocks (happy path + edge case)."
)


def is_saucedemo(tc):
    return int(tc.split("_G")[1]) <= 12


def extract_code(text):
    m = re.search(r"```(?:javascript|js|typescript|ts)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.replace("```javascript", "").replace("```js", "").replace("```", "").strip()


def ollama_generate(model, story):
    user = (f'User story: "{story["user_story"]}"\n'
            f'Category: {story.get("category","functional")}\n'
            f'Complexity: {story.get("complexity","medium")}\n'
            f'Return the Cypress script only.')
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_CY},
                     {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 2048, "stop": ["<|im_end|>"]},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    return extract_code(resp.get("message", {}).get("content", ""))


def execute(script, tc, base_url):
    CY_SPEC_DIR.mkdir(parents=True, exist_ok=True)
    for old in CY_SPEC_DIR.glob("*.cy.js"):
        old.unlink()
    spec = CY_SPEC_DIR / f"{tc}.cy.js"; spec.write_text(script)
    try:
        return run_cypress(spec, base_url)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "tests": 0, "passed": 0, "failed": 0, "failures": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-coder-14b")
    args = ap.parse_args()

    stories = [json.loads(l) for l in open(STORIES) if l.strip() and
               is_saucedemo(json.loads(l)["id"])]
    out_dir = REPO / "results" / "grounding_experiment" / "qwen" / args.model / "cypress"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}\n  QWEN CONTROL: {args.model} / cypress / saucedemo "
          f"({len(stories)} stories, zero-shot via Ollama)\n{'='*62}")
    outcomes = []
    for s in stories:
        tc = s["id"]
        try:
            script = ollama_generate(args.model, s)
        except Exception as e:
            print(f"  {tc}  GEN ERROR: {str(e)[:80]}"); continue
        res = execute(script, tc, s["base_url"])
        dtid = "data-testid" in script
        rec = {"tc_id": tc, "model": args.model, "status": res["status"],
               "tests": res["tests"], "passed": res["passed"], "failed": res["failed"],
               "uses_data_testid": dtid, "final_script": script}
        (out_dir / f"{tc}.json").write_text(json.dumps(rec, indent=2))
        outcomes.append(rec)
        print(f"  {tc}  {res['status']:<11} tests={res['tests']} passed={res['passed']} "
              f"failed={res['failed']}  data-testid={dtid}")

    full = sum(1 for r in outcomes if r["status"] == "passed")
    part = sum(1 for r in outcomes if r["status"] == "failed" and r["passed"] > 0)
    tp = sum(r["passed"] for r in outcomes); tt = sum(r["tests"] for r in outcomes)
    dtid = sum(1 for r in outcomes if r["uses_data_testid"])
    print(f"\n{'='*62}\n  RESULT — {args.model} on saucedemo (zero-shot):")
    print(f"    specs fully passing:   {full}/{len(outcomes)}   (partial: {part})")
    print(f"    individual tests:      {tp}/{tt}")
    print(f"    scripts using data-testid (wrong convention): {dtid}/{len(outcomes)}")
    print(f"    CONTEXT:  base Gemma4 0/12 (echoed prompt) | fine-tuned Gemma4 0/12 | "
          f"GPT-4o-mini 9/12 | Claude 8/12")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
