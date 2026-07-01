"""
Batch BMAD loop runner — processes all records in the dataset.

Usage:
  python -m agentic_loop.run_loop --framework cypress --model phi3
  python -m agentic_loop.run_loop --framework playwright --model gemma4 --max-iters 3
  python -m agentic_loop.run_loop --framework cypress --model phi3 --limit 10  # quick test

Results are saved to:
  results/agentic_loop/{model}/{framework}/TC_XXX.json

Start the inference API first:
  source .venv/bin/activate
  WARMUP_MODEL=phi3/cypress uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

from .loop import run as bmad_run, DEFAULT_THRESHOLD, DEFAULT_MAX_ITERS

BASE_DIR    = Path(__file__).parent.parent
DATASET     = BASE_DIR / "data" / "dataset_final.json"
RESULTS_DIR = BASE_DIR / "results" / "agentic_loop"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _api_alive(base: str) -> bool:
    try:
        return requests.get(f"{base}/health", timeout=5).status_code == 200
    except Exception:
        return False


def _exemplar_index(records: list, framework: str) -> dict:
    key = f"{framework}_script"
    return {r["id"]: r.get(key, "") for r in records}


def _summary_row(results: list) -> dict:
    if not results:
        return {}
    accepted    = sum(1 for r in results if r["accepted"])
    total       = len(results)
    scores      = [r["best_score"]  for r in results]
    iters       = [r["iterations"]  for r in results]
    latencies   = [r["total_latency_s"] for r in results]
    return {
        "total_records":    total,
        "accepted":         accepted,
        "accept_rate_pct":  round(100 * accepted / total, 2),
        "avg_score":        round(sum(scores)   / total, 4),
        "avg_iterations":   round(sum(iters)    / total, 2),
        "avg_latency_s":    round(sum(latencies)/ total, 2),
        "total_runtime_s":  round(sum(latencies), 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run BMAD loop over dataset")
    parser.add_argument("--framework",  choices=["cypress", "playwright"], default="cypress")
    parser.add_argument("--model",      choices=["phi3", "gemma4"],        default="phi3")
    parser.add_argument("--max-iters",  type=int,   default=DEFAULT_MAX_ITERS,
                        help=f"Max correction iterations (default {DEFAULT_MAX_ITERS})")
    parser.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD,
                        help=f"Accept threshold 0–1 (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--max-tokens", type=int,   default=512)
    parser.add_argument("--limit",      type=int,   default=None,
                        help="Process only first N records (for smoke testing)")
    parser.add_argument("--api-url",    default="http://localhost:8000")
    args = parser.parse_args()

    # ── Pre-flight: API reachability ──────────────────────────────────────────
    if not _api_alive(args.api_url):
        print(f"\nERROR: Inference API not reachable at {args.api_url}\n")
        print("Start it with:")
        print("  source .venv/bin/activate")
        print(f"  WARMUP_MODEL={args.model}/{args.framework} uvicorn api.app:app --host 0.0.0.0 --port 8000\n")
        sys.exit(1)

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(DATASET) as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]

    exemplars = _exemplar_index(records, args.framework)

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = RESULTS_DIR / args.model / args.framework
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume: skip already-completed records ────────────────────────────────
    completed = {p.stem for p in out_dir.glob("TC_*.json")}
    if completed:
        print(f"\n  Resuming — {len(completed)} records already done, skipping.")

    print(f"\n{'='*60}")
    print(f"  BMAD Loop  |  model={args.model}  framework={args.framework}")
    print(f"  Records={len(records)}  threshold={args.threshold}  max_iters={args.max_iters}")
    print(f"{'='*60}")

    # Reload existing results for summary
    all_results = []
    for p in sorted(out_dir.glob("TC_*.json")):
        with open(p) as f:
            all_results.append(json.load(f))

    t_wall = time.time()

    for idx, record in enumerate(records, 1):
        tc_id  = record["id"]

        # Skip if already done
        if tc_id in completed:
            print(f"  [{idx:3d}/{len(records)}] {tc_id}  ⏭️   (already done)")
            continue

        result = bmad_run(
            tc_id      = tc_id,
            user_story = record["user_story"],
            framework  = args.framework,
            model_key  = args.model,
            category   = record.get("category",   "functional"),
            complexity  = record.get("complexity", "medium"),
            exemplar   = exemplars.get(tc_id, ""),
            threshold  = args.threshold,
            max_iters  = args.max_iters,
            max_tokens = args.max_tokens,
            api_base   = args.api_url,
        )

        # Serialise and save per-record result
        record_out = {
            "tc_id":           result.tc_id,
            "framework":       result.framework,
            "model_key":       result.model_key,
            "accepted":        result.accepted,
            "iterations":      result.iterations,
            "best_score":      result.best_score,
            "total_latency_s": result.total_latency_s,
            "final_script":    result.final_script,
            "history": [
                {
                    "iteration":     h.iteration,
                    "score":         vars(h.score),
                    "latency_s":     h.latency_s,
                    "feedback_used": h.feedback_used,
                }
                for h in result.history
            ],
        }
        with open(out_dir / f"{tc_id}.json", "w") as f:
            json.dump(record_out, f, indent=2)

        all_results.append(record_out)

        status = "✅" if result.accepted else "⚠️ "
        print(
            f"  [{idx:3d}/{len(records)}] {tc_id}  {status}  "
            f"score={result.best_score:.3f}  "
            f"iters={result.iterations}  "
            f"lat={result.total_latency_s:.1f}s"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    wall = round(time.time() - t_wall, 1)
    summary = _summary_row(all_results)
    summary["wall_clock_s"] = wall

    summary_path = out_dir / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Accepted   : {summary['accepted']}/{summary['total_records']} "
          f"({summary['accept_rate_pct']:.1f}%)")
    print(f"  Avg score  : {summary['avg_score']:.4f}")
    print(f"  Avg iters  : {summary['avg_iterations']:.2f}")
    print(f"  Wall clock : {wall/60:.1f} min")
    print(f"  Results    : {out_dir}")
    print(f"  Summary    : {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
