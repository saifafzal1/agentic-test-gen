"""
Tier-1 runtime-load validation over the full dissertation script corpus.

For all 2,232 scripts (279 stories x 8 systems: 4 fine-tuned BMAD combos +
4 zero-shot baseline combos), checks whether each generated script LOADS AND
REGISTERS as a valid test suite in its target framework -- a strictly
stronger property than `node --check` syntax validity:

  - Playwright: `npx playwright test --list` collects every spec in the
    batch without opening a browser; per-file collection errors are parsed
    from the JSON report.
  - Cypress: cypress_load_check.mjs emulates the collection phase (suite
    bodies execute, test/hook callbacks register) in a single Node process.

Usage (from execution_harness/):  python load_check.py
Outputs: results/load_validation/load_check_results.json + summary table.
"""
import json
import re
import subprocess
from pathlib import Path

HARNESS_DIR = Path(__file__).parent
BASE_DIR    = HARNESS_DIR.parent
OUT_DIR     = BASE_DIR / "results" / "load_validation"
PW_SPEC_DIR = HARNESS_DIR / "pw_specs"
CY_LOAD_DIR = HARNESS_DIR / "cy_load_specs"

# (system, framework, generation dir, script field, strip fences)
SOURCES = [
    ("phi3",         "cypress",    BASE_DIR / "results/agentic_loop/phi3/cypress",             "final_script",     False),
    ("phi3",         "playwright", BASE_DIR / "results/agentic_loop/phi3/playwright",          "final_script",     False),
    ("gemma4",       "cypress",    BASE_DIR / "results/agentic_loop/gemma4/cypress",           "final_script",     False),
    ("gemma4",       "playwright", BASE_DIR / "results/agentic_loop/gemma4/playwright",        "final_script",     False),
    ("gpt4o-mini",   "cypress",    BASE_DIR / "baselines/results/cypress/gpt4o-mini",          "generated_script", True),
    ("gpt4o-mini",   "playwright", BASE_DIR / "baselines/results/playwright/gpt4o-mini",       "generated_script", True),
    ("claude-haiku", "cypress",    BASE_DIR / "baselines/results/cypress/claude-haiku",        "generated_script", True),
    ("claude-haiku", "playwright", BASE_DIR / "baselines/results/playwright/claude-haiku",     "generated_script", True),
]


def strip_fences(script: str) -> str:
    s = script.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*\n", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s


def load_scripts(gen_dir: Path, field: str, fences: bool) -> dict:
    scripts = {}
    for p in sorted(gen_dir.glob("TC_*.json")):
        if p.stem in ("summary", "_summary"):
            continue
        rec = json.loads(p.read_text())
        s = rec.get(field) or ""
        scripts[p.stem] = strip_fences(s) if fences else s
    return scripts


def check_playwright(scripts: dict) -> dict:
    """Write batch as .spec.ts, run --list once, parse per-file errors."""
    PW_SPEC_DIR.mkdir(parents=True, exist_ok=True)
    for old in PW_SPEC_DIR.glob("*.spec.*"):
        old.unlink()
    for tc_id, s in scripts.items():
        (PW_SPEC_DIR / f"{tc_id}.spec.ts").write_text(s)

    proc = subprocess.run(
        ["npx", "playwright", "test", "--list", "--reporter=json"],
        cwd=HARNESS_DIR, capture_output=True, text=True, timeout=900,
    )
    out = proc.stdout
    start, end = out.find("{"), out.rfind("}")
    rep = json.loads(out[start:end + 1])

    bad = {}
    for err in rep.get("errors", []):
        loc = (err.get("location") or {}).get("file", "")
        m = re.search(r"(TC_[A-Za-z0-9_]+)\.spec\.ts", loc or err.get("message", ""))
        if m:
            bad.setdefault(m.group(1), (err.get("message") or "")[:250])
    return {tc: {"ok": tc not in bad, "error": bad.get(tc)} for tc in scripts}


def check_cypress(scripts: dict) -> dict:
    """Write batch as .cjs/.mjs by module style, load via the Node emulator."""
    CY_LOAD_DIR.mkdir(parents=True, exist_ok=True)
    for old in CY_LOAD_DIR.glob("*"):
        old.unlink()
    paths, id_of = [], {}
    for tc_id, s in scripts.items():
        ext = ".cjs" if re.search(r"\brequire\s*\(", s) else ".mjs"
        p = CY_LOAD_DIR / f"{tc_id}{ext}"
        p.write_text(s)
        paths.append(str(p))
        id_of[str(p)] = tc_id

    results = {}
    CHUNK = 200
    for i in range(0, len(paths), CHUNK):
        proc = subprocess.run(
            ["node", str(HARNESS_DIR / "cypress_load_check.mjs"), *paths[i:i + CHUNK]],
            capture_output=True, text=True, timeout=600,
        )
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            results[id_of[r["file"]]] = {"ok": r["ok"], "error": r.get("error")}
    for tc in scripts:
        results.setdefault(tc, {"ok": False, "error": "no result from loader"})
    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows, summary = [], []
    for system, framework, gen_dir, field, fences in SOURCES:
        scripts = load_scripts(gen_dir, field, fences)
        checker = check_playwright if framework == "playwright" else check_cypress
        res = checker(scripts)
        ok = sum(1 for r in res.values() if r["ok"])
        n = len(res)
        kind = "fine-tuned" if system in ("phi3", "gemma4") else "baseline"
        summary.append(dict(system=system, kind=kind, framework=framework,
                            loads_ok=ok, total=n,
                            load_valid_pct=round(100 * ok / n, 1) if n else 0))
        print(f"{system:>13}/{framework:<10} {ok}/{n} load-valid "
              f"({100*ok/n:.1f}%)" if n else f"{system}/{framework}: EMPTY")
        for tc, r in sorted(res.items()):
            if not r["ok"]:
                all_rows.append(dict(system=system, framework=framework,
                                     tc_id=tc, error=r["error"]))

    out = dict(
        study="Tier-1 runtime-load validation of all dissertation scripts",
        method=("Playwright: `playwright test --list` collection; Cypress: "
                "emulated collection phase (suite bodies executed, test/hook "
                "callbacks registered) in Node"),
        per_system=summary,
        failures=all_rows,
    )
    path = OUT_DIR / "load_check_results.json"
    path.write_text(json.dumps(out, indent=2))
    total = sum(s["total"] for s in summary)
    ok = sum(s["loads_ok"] for s in summary)
    print(f"\nOVERALL: {ok}/{total} ({100*ok/total:.1f}%) load-valid; "
          f"{len(all_rows)} failures recorded")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
