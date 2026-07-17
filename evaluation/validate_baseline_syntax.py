"""
Syntax Validity Validator for Zero-Shot Baseline Results
==========================================================
Runs `node --check` on every baseline-generated script, mirroring
validate_syntax.py's check so baseline and fine-tuned syntax rates are
measured on the exact same yardstick.

Outputs:
  - Console summary per model/framework
  - evaluation/baseline_syntax_validation_report.json   (full per-record detail)

Usage:
  python -m evaluation.validate_baseline_syntax
  python -m evaluation.validate_baseline_syntax --model claude-haiku --framework cypress
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "baselines" / "results"
REPORT_OUT  = BASE_DIR / "evaluation" / "baseline_syntax_validation_report.json"

MODELS     = ["gpt4o-mini", "claude-haiku"]
FRAMEWORKS = ["cypress", "playwright"]


def _strip_markdown_fences(script: str) -> str:
    """
    Commercial APIs (unlike the fine-tuned models, whose system prompt forbids
    it) sometimes wrap the script in a ```language ... ``` fence. Left in place,
    the stray backtick characters turn most of the file into one giant
    accidental template literal, so a single unstripped fence can invalidate
    the whole script regardless of whether the actual code is fine.
    """
    s = script.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        if s.rstrip().endswith("```"):
            s = re.sub(r"\n?```\s*$", "", s.rstrip())
    return s.strip()


def check_syntax(script: str, framework: str) -> tuple:
    """
    Run node --check on the script.
    For Playwright (TypeScript), strip TS-specific type annotations first.
    Returns (is_valid: bool, error_msg: str)
    """
    code = _strip_ts_annotations(script) if framework == "playwright" else script
    with tempfile.NamedTemporaryFile(suffix=".js", mode="w",
                                     delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            ["node", "--check", tmp],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return True, ""
        err = (result.stderr or result.stdout).strip().splitlines()
        return False, err[0] if err else "unknown syntax error"
    except subprocess.TimeoutExpired:
        return False, "node --check timed out"
    finally:
        os.unlink(tmp)


def _strip_ts_annotations(script: str) -> str:
    """Remove TypeScript-only syntax so node --check can parse Playwright scripts."""
    script = re.sub(r"import type\s+\{[^}]+\}\s+from\s+'[^']+';?\n?", "", script)
    script = re.sub(r":\s*(string|number|boolean|void|any|Page|BrowserContext)\b", "", script)
    script = re.sub(r"<[A-Z][a-zA-Z]+>", "", script)
    return script


def validate_run(model: str, framework: str) -> dict:
    """Validate all result files for one model/framework combination."""
    run_dir = RESULTS_DIR / framework / model
    if not run_dir.exists():
        return {"error": f"No results found at {run_dir}"}

    result_files = sorted(p for p in run_dir.glob("*.json") if p.stem != "summary")
    if not result_files:
        return {"error": "No TC_*.json files found"}

    valid = invalid = 0
    records = []

    for path in result_files:
        with open(path) as f:
            data = json.load(f)

        script = _strip_markdown_fences(data.get("generated_script", ""))
        tc_id  = data["id"]

        is_valid, err = check_syntax(script, framework)

        if is_valid:
            valid += 1
        else:
            invalid += 1

        records.append({
            "tc_id":        tc_id,
            "syntax_valid": is_valid,
            "error":        err,
        })

    total = valid + invalid
    return {
        "model":            model,
        "framework":        framework,
        "total":            total,
        "syntax_valid":     valid,
        "syntax_invalid":   invalid,
        "syntax_rate_pct":  round(100 * valid / total, 2) if total else 0,
        "records":          records,
    }


def main():
    parser = argparse.ArgumentParser(description="Syntax-validate zero-shot baseline results")
    parser.add_argument("--model",     choices=MODELS + ["all"],     default="all")
    parser.add_argument("--framework", choices=FRAMEWORKS + ["all"], default="all")
    args = parser.parse_args()

    models     = MODELS     if args.model     == "all" else [args.model]
    frameworks = FRAMEWORKS if args.framework == "all" else [args.framework]

    all_reports = {}
    print(f"\n{'='*60}")
    print("  Syntax Validity Check — node --check on all baseline scripts")
    print(f"{'='*60}\n")

    for framework in frameworks:
        for model in models:
            run_dir = RESULTS_DIR / framework / model
            if not run_dir.exists():
                print(f"  [{model}/{framework}] — no results, skipping")
                continue

            print(f"  Checking {model}/{framework}...", end="", flush=True)
            report = validate_run(model, framework)

            if "error" in report:
                print(f" ERROR: {report['error']}")
                continue

            key = f"{model}/{framework}"
            all_reports[key] = report

            invalid_cases = [r for r in report["records"] if not r["syntax_valid"]]
            print(f"\n  {key}:")
            print(f"    Syntax valid  : {report['syntax_valid']}/{report['total']} "
                  f"({report['syntax_rate_pct']}%)")
            if invalid_cases:
                print(f"    Invalid scripts ({len(invalid_cases)}):")
                for r in invalid_cases[:10]:
                    print(f"      {r['tc_id']}  error: {r['error'][:80]}")
                if len(invalid_cases) > 10:
                    print(f"      ... and {len(invalid_cases)-10} more")
            print()

    if all_reports:
        print(f"\n{'='*60}")
        print(f"  {'Model/Framework':<22} {'Valid':>8} {'Invalid':>9} {'Rate':>8}")
        print(f"  {'-'*22} {'-'*8} {'-'*9} {'-'*8}")
        for key, r in all_reports.items():
            print(f"  {key:<22} {r['syntax_valid']:>8} {r['syntax_invalid']:>9} "
                  f"{r['syntax_rate_pct']:>7.1f}%")
        print(f"{'='*60}\n")

        REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
        summary = {k: {kk: vv for kk, vv in v.items() if kk != "records"}
                   for k, v in all_reports.items()}
        with open(REPORT_OUT, "w") as f:
            json.dump({"summary": summary, "detail": all_reports}, f, indent=2)
        print(f"  Full report saved to: {REPORT_OUT}")


if __name__ == "__main__":
    main()
