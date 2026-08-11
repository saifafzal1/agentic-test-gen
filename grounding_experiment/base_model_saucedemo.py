"""
Control experiment: run the BASE Gemma 4 E4B (no QLoRA adapter), zero-shot, on
the 12 saucedemo stories, and execute against the live site.

Apples-to-apples with the fine-tuned pilot (same ungrounded story prompt; the
only difference is the absence of the adapter) — isolates whether fine-tuning
HURT execution (base > fine-tuned) or the 4B model simply can't do it
(base ~= fine-tuned ~= 0).

Usage: python -m grounding_experiment.base_model_saucedemo --model gemma4-base
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "execution_harness"))
from agentic_loop.generator import generate as real_generate    # noqa: E402
from agentic_loop.scorer import score as real_score             # noqa: E402
from run_execution import run_cypress, CY_SPEC_DIR              # noqa: E402

STORIES = REPO / "data" / "execution_validation" / "grounded_stories.jsonl"


def is_saucedemo(tc):  # TC_G01..12 are saucedemo
    return int(tc.split("_G")[1]) <= 12


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
    ap.add_argument("--model", default="gemma4-base")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    stories = [json.loads(l) for l in open(STORIES) if l.strip() and
               is_saucedemo(json.loads(l)["id"])]
    out_dir = REPO / "results" / "grounding_experiment" / "base" / args.model / "cypress"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}\n  BASE-MODEL CONTROL: {args.model} / cypress / saucedemo "
          f"({len(stories)} stories, zero-shot)\n{'='*62}")
    outcomes = []
    for s in stories:
        tc = s["id"]
        script, lat = real_generate(args.model, "cypress", s["user_story"],
                                    s.get("category", "functional"),
                                    s.get("complexity", "medium"), args.max_tokens)
        res = execute(script, tc, s["base_url"])
        q = real_score(script, "cypress")
        rec = {"tc_id": tc, "model": args.model, "status": res["status"],
               "tests": res["tests"], "passed": res["passed"], "failed": res["failed"],
               "static_score": q.total, "latency_s": lat, "final_script": script}
        (out_dir / f"{tc}.json").write_text(json.dumps(rec, indent=2))
        outcomes.append(rec)
        print(f"  {tc}  {res['status']:<11} tests={res['tests']} passed={res['passed']} "
              f"failed={res['failed']}  static={q.total:.2f}  ({lat:.0f}s)")

    full = sum(1 for r in outcomes if r["status"] == "passed")
    part = sum(1 for r in outcomes if r["status"] == "failed" and r["passed"] > 0)
    tp = sum(r["passed"] for r in outcomes); tt = sum(r["tests"] for r in outcomes)
    print(f"\n{'='*62}\n  RESULT — {args.model} on saucedemo (zero-shot, ungrounded):")
    print(f"    specs fully passing:   {full}/{len(outcomes)}   (partial: {part})")
    print(f"    individual tests:      {tp}/{tt}")
    print(f"    fine-tuned Gemma 4 (same condition):  0/12  specs")
    print(f"    commercial baselines (context):       GPT-4o-mini 9/12, Claude 8/12")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
