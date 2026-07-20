"""
Execution-admission filter: a candidate story/script pair enters the
grounded corpus ONLY if its script executes and passes against the live
application. Reuses the execution harness runners.

Usage: python3 grounded_corpus/filter_corpus.py
Output: grounded_corpus/corpus/grounded_corpus.jsonl (one record per
        admitted framework script) + admission_report.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "execution_harness"))
from run_execution import run_cypress, run_playwright, CY_SPEC_DIR, PW_SPEC_DIR  # noqa: E402

CANDIDATES = HERE / "candidates"
CORPUS_DIR = HERE / "corpus"


def execute(framework: str, script: str, tc_id: str, base_url: str) -> dict:
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


def main():
    CORPUS_DIR.mkdir(exist_ok=True)
    corpus_path = CORPUS_DIR / "grounded_corpus.jsonl"
    report_path = CORPUS_DIR / "admission_report.json"
    admitted = 0
    # Resume support: skip anything already admitted OR already adjudicated
    # in a previous filter run (rejects stay rejected; regenerate under a
    # new candidate id to retry).
    report = json.loads(report_path.read_text()) if report_path.exists() else []
    existing = {r["id"] + "/" + r["framework"] for r in report}
    if corpus_path.exists():
        existing |= {json.loads(l)["id"] + "/" + json.loads(l)["framework"]
                     for l in corpus_path.open() if l.strip()}

    with corpus_path.open("a") as corpus:
        for cand_path in sorted(CANDIDATES.glob("GC_*.json")):
            cand = json.loads(cand_path.read_text())
            for framework, field in (("cypress", "cypress_script"),
                                     ("playwright", "playwright_script")):
                key = cand["id"] + "/" + framework
                if key in existing:
                    continue
                script = cand.get(field) or ""
                try:
                    res = execute(framework, script, cand["id"], cand["base_url"])
                except Exception as e:
                    res = {"status": "harness_error", "tests": 0, "passed": 0,
                           "failed": 0, "detail": str(e)[:200]}
                ok = res["status"] == "passed" and res["tests"] >= 2
                report.append(dict(id=cand["id"], framework=framework,
                                   status=res["status"], tests=res["tests"],
                                   passed=res["passed"], admitted=ok))
                print(f"  {cand['id']:<28} {framework:<10} {res['status']:<11} "
                      f"({res['passed']}/{res['tests']}) "
                      f"{'ADMITTED' if ok else 'rejected'}")
                if ok:
                    corpus.write(json.dumps(dict(
                        id=cand["id"], app=cand["app"], page=cand["page"],
                        category=cand["category"],
                        complexity=cand.get("complexity", "medium"),
                        user_story=cand["user_story"], framework=framework,
                        script=script,
                        execution=dict(tests=res["tests"], passed=res["passed"]),
                    )) + "\n")
                    corpus.flush()
                    admitted += 1

    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nAdmitted {admitted} new scripts this run; report now covers "
          f"{len(report)} adjudications; corpus at {corpus_path}")


if __name__ == "__main__":
    main()
