"""
Execution-feedback BMAD loop — run the generated script against the live app
each iteration and feed the real runtime errors back into the next attempt.

Usage:
  python -m grounding_experiment.exec_feedback_loop --model phi3 --framework cypress
  python -m grounding_experiment.exec_feedback_loop --model phi3 --framework cypress --grounding on
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "execution_harness"))

from agentic_loop.generator import generate as real_generate            # noqa: E402
from agentic_loop.scorer import score as real_score                     # noqa: E402
from run_execution import (run_cypress, run_playwright, load_base_urls,  # noqa: E402
                           CY_SPEC_DIR, PW_SPEC_DIR)
from grounding_experiment.dom_context import build_dom_context          # noqa: E402

STORIES = REPO / "data" / "execution_validation" / "grounded_stories.jsonl"
OUT_ROOT = REPO / "results" / "grounding_experiment" / "execfb"


def _execute(framework, script, tc_id, base_url):
    if framework == "cypress":
        CY_SPEC_DIR.mkdir(parents=True, exist_ok=True)
        for old in CY_SPEC_DIR.glob("*.cy.js"):
            old.unlink()
        spec = CY_SPEC_DIR / f"{tc_id}.cy.js"; spec.write_text(script)
        try:
            return run_cypress(spec, base_url)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "tests": 0, "passed": 0, "failed": 0, "failures": []}
    PW_SPEC_DIR.mkdir(parents=True, exist_ok=True)
    for old in PW_SPEC_DIR.glob("*.spec.*"):
        old.unlink()
    spec = PW_SPEC_DIR / f"{tc_id}.spec.ts"; spec.write_text(script)
    try:
        return run_playwright(spec, base_url)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "tests": 0, "passed": 0, "failed": 0, "failures": []}


def _exec_feedback(res):
    msgs = [f.get("message", "") for f in res.get("failures", []) if f.get("message")]
    if not msgs and res.get("detail"):
        msgs = [res["detail"]]
    joined = " | ".join(m.replace("\n", " ")[:200] for m in msgs[:3]) or "the tests did not pass"
    return ("The previous script FAILED when executed against the LIVE application. "
            "Fix these ACTUAL runtime errors — use only selectors, routes, and behaviour that "
            f"truly exist in the app; do not invent them: {joined}")


def exec_feedback_run(story, model, framework, grounding, max_iters, max_tokens):
    tc = story["id"]
    base_url = story["base_url"]
    dom = build_dom_context(story) if grounding else ""
    feedback = ""
    history = []
    best = None  # (tests_passed, status, script, res)

    for i in range(1, max_iters + 1):
        story_text = story["user_story"] if not feedback else (
            f'{story["user_story"]}\n\n[CORRECTION REQUIRED — {feedback}]')
        script, _lat = real_generate(model, framework, story_text,
                                     story.get("category", "functional"),
                                     story.get("complexity", "medium"),
                                     max_tokens, dom_context=dom)
        res = _execute(framework, script, tc, base_url)
        q = real_score(script, framework)
        history.append({"iter": i, "status": res["status"], "tests": res["tests"],
                        "passed": res["passed"], "failed": res["failed"],
                        "static_score": q.total})
        rank = (res["passed"], res["status"] == "passed", q.total)
        if best is None or rank > best[0]:
            best = (rank, res, script)
        print(f"    iter {i}: {res['status']:<11} tests={res['tests']} passed={res['passed']} "
              f"failed={res['failed']} static={q.total:.2f}")
        if res["status"] == "passed" and res["tests"] >= 1:
            break
        feedback = _exec_feedback(res)

    _, best_res, best_script = best
    return {
        "tc_id": tc, "model": model, "framework": framework,
        "grounding": grounding, "max_iters": max_iters,
        "accepted": best_res["status"] == "passed",
        "best_status": best_res["status"], "best_tests": best_res["tests"],
        "best_passed": best_res["passed"], "iterations": len(history),
        "history": history, "final_script": best_script,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["phi3", "gemma4", "phi3-grounded"], default="phi3")
    ap.add_argument("--framework", choices=["cypress", "playwright"], default="cypress")
    ap.add_argument("--grounding", choices=["on", "off"], default="on")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    stories = [json.loads(l) for l in open(STORIES) if l.strip()]
    tag = f"{args.grounding}"
    out_dir = OUT_ROOT / f"{args.model}_{args.framework}_ground-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in out_dir.glob("TC_G*.json")}
    print(f"\n{'='*66}\n  EXEC-FEEDBACK LOOP  |  {args.model}/{args.framework}  "
          f"|  grounding={args.grounding}  |  max_iters={args.max_iters}  "
          f"|  {len(done)} done\n{'='*66}")

    outcomes = []
    for s in stories:
        if s["id"] in done:
            outcomes.append(json.loads((out_dir / f"{s['id']}.json").read_text())); continue
        print(f"  {s['id']}:")
        rec = exec_feedback_run(s, args.model, args.framework,
                                args.grounding == "on", args.max_iters, args.max_tokens)
        (out_dir / f"{s['id']}.json").write_text(json.dumps(rec, indent=2))
        outcomes.append(rec)

    full = sum(1 for r in outcomes if r["accepted"])
    tp = sum(r["best_passed"] for r in outcomes)
    tt = sum(r["best_tests"] for r in outcomes)
    conv = sum(1 for r in outcomes if r["accepted"])
    print(f"\n{'='*66}\n  RESULT ({args.model}/{args.framework}, grounding={args.grounding}, "
          f"max_iters={args.max_iters}):")
    print(f"    specs fully passing (executed):  {full}/{len(outcomes)}")
    print(f"    individual tests passing:        {tp}/{tt}")
    print(f"    vs static-loop pilot floor:      0/24")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
