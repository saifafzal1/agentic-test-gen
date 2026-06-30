"""
Phase 3 — Baseline Evaluation Scoring
======================================
Reads generated scripts from baselines/results/ and computes:

  - Syntax heuristics    : keyword presence, structural patterns
  - ROUGE-L              : against ground-truth script from dataset
  - Token-level F1       : precision / recall / F1 on whitespace tokens
  - Quality metrics      : framework-specific keyword counts

Per-record results  → evaluation/scores/{framework}/{model}/{id}_score.json
Aggregated summary  → evaluation/scores/{framework}/{model}/aggregate.json
Cross-model matrix  → evaluation/scores/comparison_matrix.json

Usage:
  python -u evaluation/evaluate_baselines.py --framework cypress
  python -u evaluation/evaluate_baselines.py --framework playwright
  python -u evaluation/evaluate_baselines.py --all
"""

import os, json, re, argparse
from pathlib import Path
from collections import defaultdict

# ── Args ──────────────────────────────────────────────────────────────────────
MODELS     = ["gpt4o-mini", "claude-haiku", "gemini3-flash-lite"]
FRAMEWORKS = ["cypress", "playwright"]

parser = argparse.ArgumentParser()
parser.add_argument("--framework", choices=FRAMEWORKS, help="Framework to evaluate")
parser.add_argument("--all",       action="store_true", help="Evaluate all frameworks")
args = parser.parse_args()

if not args.all and not args.framework:
    parser.error("Provide --framework or --all")

EVAL_FRAMEWORKS = FRAMEWORKS if args.all else [args.framework]

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
DATA_DIR     = BASE_DIR / "data"
RESULTS_DIR  = BASE_DIR / "baselines" / "results"
SCORES_DIR   = Path(__file__).parent / "scores"

# ── ROUGE-L ───────────────────────────────────────────────────────────────────
def lcs_length(a: list, b: list) -> int:
    """Compute LCS length using DP."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-optimised 1D DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    return prev[n]

def rouge_l(hypothesis: str, reference: str) -> dict:
    hyp_tokens = hypothesis.split()
    ref_tokens = reference.split()
    if not hyp_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = lcs_length(hyp_tokens, ref_tokens)
    p = lcs / len(hyp_tokens)
    r = lcs / len(ref_tokens)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

# ── Token-level F1 ────────────────────────────────────────────────────────────
def token_f1(hypothesis: str, reference: str) -> dict:
    hyp = set(hypothesis.split())
    ref = set(reference.split())
    if not hyp or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    common = hyp & ref
    p = len(common) / len(hyp)
    r = len(common) / len(ref)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

# ── Framework-specific quality checks ────────────────────────────────────────
CYPRESS_KEYWORDS = ["describe(", "it(", "cy.visit(", "cy.get(", "cy.click(",
                    "cy.type(", "cy.should(", "beforeEach(", "cy.intercept("]

PLAYWRIGHT_KEYWORDS = ["test.describe(", "test(", "page.goto(", "getByRole(",
                       "getByLabel(", "getByText(", "expect(", "test.beforeEach(",
                       "await "]

def quality_checks(script: str, framework: str) -> dict:
    keywords = CYPRESS_KEYWORDS if framework == "cypress" else PLAYWRIGHT_KEYWORDS
    hits = {kw: (kw in script) for kw in keywords}
    score = sum(hits.values()) / len(keywords)
    return {
        "keyword_hits":  hits,
        "keyword_score": round(score, 4),
        "line_count":    len(script.splitlines()),
        "char_count":    len(script),
        # No markdown fences (good sign)
        "no_fences":     "```" not in script,
    }

# ── Per-record scorer ─────────────────────────────────────────────────────────
def score_record(result: dict, ground_truth: str, framework: str) -> dict:
    gen = result.get("generated_script", "")
    rl  = rouge_l(gen, ground_truth)
    tf1 = token_f1(gen, ground_truth)
    qc  = quality_checks(gen, framework)
    return {
        "id":          result["id"],
        "model":       result["model"],
        "framework":   framework,
        "category":    result["category"],
        "complexity":  result["complexity"],
        "rouge_l":     rl,
        "token_f1":    tf1,
        "quality":     qc,
        "latency_s":   result.get("latency_s", 0),
        "prompt_tokens":     result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
    }

# ── Main evaluation loop ──────────────────────────────────────────────────────
comparison = {}   # framework → model → aggregate metrics

for framework in EVAL_FRAMEWORKS:
    print(f"\n{'='*60}")
    print(f"📊 Evaluating: {framework.upper()}")
    print(f"{'='*60}")

    # Load ground-truth scripts
    data_file = DATA_DIR / f"{framework}_dataset.jsonl"
    ground_truth = {}
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                ground_truth[rec["id"]] = rec["script"]

    print(f"   Ground-truth records: {len(ground_truth)}")

    comparison[framework] = {}

    for model in MODELS:
        results_dir = RESULTS_DIR / framework / model
        if not results_dir.exists():
            print(f"   ⚠️  No results for {model} — skipping (run baselines first)")
            continue

        result_files = [p for p in results_dir.glob("*.json") if p.stem != "summary"]
        if not result_files:
            print(f"   ⚠️  Empty results dir for {model} — skipping")
            continue

        scores_dir = SCORES_DIR / framework / model
        scores_dir.mkdir(parents=True, exist_ok=True)

        all_scores = []
        for rp in sorted(result_files):
            with open(rp) as f:
                result = json.load(f)

            rec_id = result["id"]
            gt     = ground_truth.get(rec_id, "")
            if not gt:
                print(f"      ⚠️  No ground-truth for {rec_id}")
                continue

            sc = score_record(result, gt, framework)
            all_scores.append(sc)

            with open(scores_dir / f"{rec_id}_score.json", "w") as f:
                json.dump(sc, f, indent=2)

        if not all_scores:
            continue

        # Aggregate
        def mean(vals): return round(sum(vals) / len(vals), 4) if vals else 0.0

        agg = {
            "model":     model,
            "framework": framework,
            "n":         len(all_scores),
            "rouge_l": {
                "precision": mean([s["rouge_l"]["precision"] for s in all_scores]),
                "recall":    mean([s["rouge_l"]["recall"]    for s in all_scores]),
                "f1":        mean([s["rouge_l"]["f1"]        for s in all_scores]),
            },
            "token_f1": {
                "precision": mean([s["token_f1"]["precision"] for s in all_scores]),
                "recall":    mean([s["token_f1"]["recall"]    for s in all_scores]),
                "f1":        mean([s["token_f1"]["f1"]        for s in all_scores]),
            },
            "quality": {
                "avg_keyword_score": mean([s["quality"]["keyword_score"] for s in all_scores]),
                "avg_line_count":    mean([s["quality"]["line_count"]    for s in all_scores]),
                "pct_no_fences":     mean([float(s["quality"]["no_fences"]) for s in all_scores]),
            },
            "avg_latency_s":           mean([s["latency_s"]           for s in all_scores]),
            "avg_prompt_tokens":       mean([s["prompt_tokens"]       for s in all_scores]),
            "avg_completion_tokens":   mean([s["completion_tokens"]   for s in all_scores]),
            # Per-category breakdown
            "by_category": {},
        }

        cats = defaultdict(list)
        for s in all_scores:
            cats[s["category"]].append(s["rouge_l"]["f1"])
        agg["by_category"] = {cat: mean(vals) for cat, vals in sorted(cats.items())}

        with open(scores_dir / "aggregate.json", "w") as f:
            json.dump(agg, f, indent=2)

        comparison[framework][model] = {
            "rouge_l_f1":         agg["rouge_l"]["f1"],
            "token_f1":           agg["token_f1"]["f1"],
            "keyword_score":      agg["quality"]["avg_keyword_score"],
            "avg_latency_s":      agg["avg_latency_s"],
        }

        print(f"\n   ✅ {model:15s} | n={len(all_scores):3d} | "
              f"ROUGE-L F1={agg['rouge_l']['f1']:.3f} | "
              f"Token-F1={agg['token_f1']['f1']:.3f} | "
              f"Keywords={agg['quality']['avg_keyword_score']:.3f} | "
              f"Latency={agg['avg_latency_s']:.2f}s")

# ── Cross-model comparison matrix ────────────────────────────────────────────
matrix_path = SCORES_DIR / "comparison_matrix.json"
SCORES_DIR.mkdir(parents=True, exist_ok=True)

# Merge with existing if present
if matrix_path.exists():
    with open(matrix_path) as f:
        existing = json.load(f)
    for fw, models in comparison.items():
        existing.setdefault(fw, {}).update(models)
    comparison = existing

with open(matrix_path, "w") as f:
    json.dump(comparison, f, indent=2)

print(f"\n📋 Comparison matrix saved → {matrix_path}")
print("\n📊 Summary:")
for fw, models in comparison.items():
    print(f"\n  {fw.upper()}:")
    print(f"  {'Model':<18} {'ROUGE-L F1':>10} {'Token-F1':>10} {'Keywords':>10} {'Latency':>10}")
    print(f"  {'-'*58}")
    for mdl, metrics in models.items():
        print(f"  {mdl:<18} {metrics['rouge_l_f1']:>10.3f} {metrics['token_f1']:>10.3f} "
              f"{metrics['keyword_score']:>10.3f} {metrics['avg_latency_s']:>9.2f}s")
