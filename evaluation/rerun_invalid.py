"""
Targeted re-run: regenerate only syntactically invalid scripts.

Reads evaluation/syntax_validation_report.json (produced by validate_syntax.py),
deletes the stale TC_*.json for each invalid record, then invokes run_loop
so checkpoint/resume re-generates only those records.

Usage:
  # Re-run invalid scripts for one combination
  python -m evaluation.rerun_invalid --model phi3 --framework cypress

  # Re-run all combinations with invalid results
  python -m evaluation.rerun_invalid --all

  # Dry-run: list what would be deleted without touching anything
  python -m evaluation.rerun_invalid --model phi3 --framework cypress --dry-run
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "agentic_loop"
REPORT_PATH = BASE_DIR / "evaluation" / "syntax_validation_report.json"

MODELS     = ["phi3", "gemma4"]
FRAMEWORKS = ["cypress", "playwright"]


def load_invalid_ids(model: str, framework: str) -> list:
    """Return list of tc_ids whose scripts are syntactically invalid."""
    if not REPORT_PATH.exists():
        print(f"ERROR: {REPORT_PATH} not found. Run validate_syntax.py first.")
        sys.exit(1)

    with open(REPORT_PATH) as f:
        report = json.load(f)

    key = f"{model}/{framework}"
    detail = report.get("detail", {}).get(key)
    if not detail:
        print(f"  [{key}] No validation data found — skipping.")
        return []

    invalid = [r["tc_id"] for r in detail["records"] if not r["syntax_valid"]]
    return invalid


def rerun(model: str, framework: str, dry_run: bool = False):
    invalid_ids = load_invalid_ids(model, framework)
    if not invalid_ids:
        print(f"  [{model}/{framework}] All scripts are valid — nothing to re-run.")
        return

    run_dir = RESULTS_DIR / model / framework
    print(f"\n  [{model}/{framework}] {len(invalid_ids)} invalid scripts to re-generate.")

    deleted = 0
    for tc_id in invalid_ids:
        path = run_dir / f"{tc_id}.json"
        if path.exists():
            if dry_run:
                print(f"    DRY-RUN: would delete {path.name}")
            else:
                path.unlink()
                deleted += 1

    if dry_run:
        print(f"    (dry-run) would delete {len(invalid_ids)} files and re-run.")
        return

    print(f"    Deleted {deleted} stale files. Starting targeted re-run with max-tokens=1024...")

    cmd = [
        sys.executable, "-m", "agentic_loop.run_loop",
        "--model",      model,
        "--framework",  framework,
        "--max-tokens", "1024",
    ]
    print(f"    Command: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=BASE_DIR)


def main():
    parser = argparse.ArgumentParser(description="Re-run syntactically invalid BMAD results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",       action="store_true",
                       help="Re-run all model/framework combinations with invalid scripts")
    group.add_argument("--model",     choices=MODELS)
    parser.add_argument("--framework", choices=FRAMEWORKS)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show what would be deleted without deleting anything")
    args = parser.parse_args()

    if args.all:
        for model in MODELS:
            for fw in FRAMEWORKS:
                rerun(model, fw, dry_run=args.dry_run)
    else:
        if not args.framework:
            parser.error("--framework is required when --model is specified")
        rerun(args.model, args.framework, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
