"""
Task 4 — grounding OFF vs ON generation over the 24 grounded stories.

For a given model/framework, runs the real BMAD loop on every grounded story
TWICE: once with grounding OFF (empty dom_context = the ungrounded pilot path)
and once with grounding ON (real-selector inventory from build_dom_context).
Outputs go to results/grounding_experiment/<condition>/<model>/<framework>/.

Checkpoint/resume: skips a story whose TC_G*.json already exists for that
(condition, model, framework).

Usage (from repo root, .venv active):
  python -m grounding_experiment.run_grounding --model phi3 --framework cypress
  python -m grounding_experiment.run_grounding --model phi3 --framework cypress --condition on
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from agentic_loop.loop import run as bmad_run          # noqa: E402
from grounding_experiment.dom_context import build_dom_context  # noqa: E402

STORIES = REPO / "data" / "execution_validation" / "grounded_stories.jsonl"
OUT_ROOT = REPO / "results" / "grounding_experiment"


def load_stories():
    return [json.loads(l) for l in open(STORIES) if l.strip()]


def run_condition(model, framework, condition, stories, max_tokens):
    out_dir = OUT_ROOT / condition / model / framework
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in out_dir.glob("TC_G*.json")}
    print(f"\n{'='*66}\n  GROUNDING {condition.upper()}  |  {model}/{framework}  "
          f"|  {len(done)} already done\n{'='*66}")

    for s in stories:
        tc = s["id"]
        if tc in done:
            print(f"  {tc}  (skip, done)")
            continue
        dom = build_dom_context(s) if condition == "on" else ""
        if condition == "on" and not dom:
            print(f"  {tc}  ⚠️  no DOM context matched — running ungrounded")
        r = bmad_run(
            tc_id=tc, user_story=s["user_story"], framework=framework,
            model_key=model, category=s.get("category", "functional"),
            complexity=s.get("complexity", "medium"),
            exemplar="", max_tokens=max_tokens, dom_context=dom,
        )
        rec = {
            "tc_id": r.tc_id, "condition": condition, "model": model,
            "framework": framework, "accepted": r.accepted,
            "iterations": r.iterations, "best_score": r.best_score,
            "total_latency_s": r.total_latency_s,
            "dom_context_used": bool(dom), "dom_context_chars": len(dom),
            "final_script": r.final_script,
        }
        (out_dir / f"{tc}.json").write_text(json.dumps(rec, indent=2))
        print(f"  {tc}  {condition}  accepted={r.accepted!s:<5} "
              f"score={r.best_score:.3f}  iters={r.iterations}  "
              f"({r.total_latency_s:.1f}s)  domctx={len(dom)}c")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["phi3", "gemma4", "phi3-grounded"], default="phi3")
    ap.add_argument("--framework", choices=["cypress", "playwright"], default="cypress")
    ap.add_argument("--condition", choices=["off", "on", "both"], default="both")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    stories = load_stories()
    conditions = ["off", "on"] if args.condition == "both" else [args.condition]
    for cond in conditions:
        run_condition(args.model, args.framework, cond, stories, args.max_tokens)
    print("\nDONE.")


if __name__ == "__main__":
    main()
