"""
Task 5 — execute the OFF and ON grounding generations against live apps.

Reuses the execution-harness runners (run_cypress / run_playwright) to run
each generated script from results/grounding_experiment/<off|on>/<model>/
<framework>/ against its target application, and writes per-condition outcome
files to results/grounding_experiment/outcomes/.

Usage (from repo root):
  python -m grounding_experiment.execute_grounding --model phi3 --framework cypress
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "execution_harness"))
from run_execution import (run_cypress, run_playwright, load_base_urls,   # noqa: E402
                           CY_SPEC_DIR, PW_SPEC_DIR)

GEN_ROOT = REPO / "results" / "grounding_experiment"
OUT_DIR = GEN_ROOT / "outcomes"


def execute(framework, script, tc_id, base_url):
    if framework == "cypress":
        CY_SPEC_DIR.mkdir(parents=True, exist_ok=True)
        for old in CY_SPEC_DIR.glob("*.cy.js"):
            old.unlink()
        spec = CY_SPEC_DIR / f"{tc_id}.cy.js"
        spec.write_text(script)
        return run_cypress(spec, base_url)
    PW_SPEC_DIR.mkdir(parents=True, exist_ok=True)
    for old in PW_SPEC_DIR.glob("*.spec.*"):
        old.unlink()
    spec = PW_SPEC_DIR / f"{tc_id}.spec.ts"
    spec.write_text(script)
    return run_playwright(spec, base_url)


def run_condition(model, framework, condition, base_urls):
    gen_dir = GEN_ROOT / condition / model / framework
    records = sorted(gen_dir.glob("TC_G*.json"))
    print(f"\n{'='*60}\n  EXECUTE grounding {condition.upper()}: {model}/{framework} "
          f"({len(records)} scripts)\n{'='*60}")
    outcomes = []
    for rp in records:
        rec = json.loads(rp.read_text())
        tc = rec["tc_id"]
        base = base_urls[tc]
        try:
            res = execute(framework, rec["final_script"], tc, base)
        except subprocess.TimeoutExpired:
            res = {"status": "timeout", "tests": 0, "passed": 0, "failed": 0}
        res.update(tc_id=tc, condition=condition, model=model, framework=framework,
                   bmad_accepted=rec.get("accepted"), dom_context_used=rec.get("dom_context_used"))
        outcomes.append(res)
        print(f"  {tc}  {condition}  {res['status']:<11} "
              f"tests={res['tests']} passed={res['passed']} failed={res['failed']}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{model}_{framework}_{condition}.json").write_text(json.dumps(outcomes, indent=2))
    full = sum(1 for o in outcomes if o["status"] == "passed")
    tp = sum(o["passed"] for o in outcomes); tt = sum(o["tests"] for o in outcomes)
    print(f"  SUMMARY {condition}: {full}/{len(outcomes)} specs fully passing; "
          f"{tp}/{tt} individual tests")
    return full, len(outcomes), tp, tt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phi3")
    ap.add_argument("--framework", choices=["cypress", "playwright"], default="cypress")
    args = ap.parse_args()
    base_urls = load_base_urls()
    res = {}
    for cond in ["off", "on"]:
        res[cond] = run_condition(args.model, args.framework, cond, base_urls)
    print(f"\n{'='*60}\n  LIFT ({args.model}/{args.framework}):")
    print(f"    specs fully passing:  OFF {res['off'][0]}/{res['off'][1]}  ->  "
          f"ON {res['on'][0]}/{res['on'][1]}")
    print(f"    individual tests:     OFF {res['off'][2]}/{res['off'][3]}  ->  "
          f"ON {res['on'][2]}/{res['on'][3]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
