"""
Task 7 driver — repair the ON generations with enforced selector-grounding,
then execute against live apps. Prints the OFF -> ON -> REPAIRED progression.

Usage: python -m grounding_experiment.repair_execute --model phi3 --framework cypress
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
from grounding_experiment.selector_repair import inventory, repair          # noqa: E402

GEN_ROOT = REPO / "results" / "grounding_experiment"
STORIES = REPO / "data" / "execution_validation" / "grounded_stories.jsonl"


def execute(framework, script, tc_id, base_url):
    if framework == "cypress":
        CY_SPEC_DIR.mkdir(parents=True, exist_ok=True)
        for old in CY_SPEC_DIR.glob("*.cy.js"):
            old.unlink()
        spec = CY_SPEC_DIR / f"{tc_id}.cy.js"; spec.write_text(script)
        return run_cypress(spec, base_url)
    PW_SPEC_DIR.mkdir(parents=True, exist_ok=True)
    for old in PW_SPEC_DIR.glob("*.spec.*"):
        old.unlink()
    spec = PW_SPEC_DIR / f"{tc_id}.spec.ts"; spec.write_text(script)
    return run_playwright(spec, base_url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phi3")
    ap.add_argument("--framework", choices=["cypress", "playwright"], default="cypress")
    args = ap.parse_args()
    m, fw = args.model, args.framework

    stories = {json.loads(l)["id"]: json.loads(l) for l in open(STORIES) if l.strip()}
    base_urls = load_base_urls()
    on_dir = GEN_ROOT / "on" / m / fw
    rep_gen = GEN_ROOT / "repaired" / m / fw
    rep_gen.mkdir(parents=True, exist_ok=True)

    outcomes = []
    total_changes = 0
    print(f"\n{'='*62}\n  REPAIR + EXECUTE (enforced grounding): {m}/{fw}\n{'='*62}")
    for rp in sorted(on_dir.glob("TC_G*.json")):
        rec = json.loads(rp.read_text())
        tc = rec["tc_id"]
        s = stories[tc]
        dts, ids = inventory(s)
        repaired, nchg = repair(rec["final_script"], dts, ids)
        total_changes += nchg
        # persist the repaired script
        (rep_gen / f"{tc}.json").write_text(json.dumps(
            {**rec, "condition": "repaired", "selectors_repaired": nchg,
             "final_script": repaired}, indent=2))
        try:
            res = execute(fw, repaired, tc, base_urls[tc])
        except subprocess.TimeoutExpired:
            res = {"status": "timeout", "tests": 0, "passed": 0, "failed": 0}
        res.update(tc_id=tc, condition="repaired", selectors_repaired=nchg)
        outcomes.append(res)
        print(f"  {tc}  repaired={nchg:<2}  {res['status']:<11} "
              f"tests={res['tests']} passed={res['passed']} failed={res['failed']}")

    (GEN_ROOT / "outcomes" / f"{m}_{fw}_repaired.json").write_text(json.dumps(outcomes, indent=2))

    def load(cond):
        p = GEN_ROOT / "outcomes" / f"{m}_{fw}_{cond}.json"
        o = json.loads(p.read_text()) if p.exists() else []
        full = sum(1 for x in o if x["status"] == "passed")
        tp = sum(x["passed"] for x in o); tt = sum(x["tests"] for x in o)
        return full, len(o), tp, tt

    off, on = load("off"), load("on")
    rep = (sum(1 for x in outcomes if x["status"] == "passed"), len(outcomes),
           sum(x["passed"] for x in outcomes), sum(x["tests"] for x in outcomes))
    print(f"\n{'='*62}\n  PROGRESSION ({m}/{fw})   (total selectors repaired: {total_changes})")
    print(f"    specs fully passing:  OFF {off[0]}/{off[1]}  ->  ON {on[0]}/{on[1]}  ->  REPAIRED {rep[0]}/{rep[1]}")
    print(f"    individual tests:     OFF {off[2]}/{off[3]}  ->  ON {on[2]}/{on[3]}  ->  REPAIRED {rep[2]}/{rep[3]}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
