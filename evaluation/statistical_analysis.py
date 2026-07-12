"""
Phase 8 — Statistical Evaluation
==================================
Builds one unified per-record table across all 8 model/framework combos
(2 baseline models + 2 fine-tuned models, x 2 frameworks) and runs:

  - One-way ANOVA        : do mean ROUGE-L F1 scores differ across combos?
  - Wilcoxon signed-rank  : paired, per-record comparison (fine-tuned vs. best
                            baseline, and phi3 vs. gemma4), matched by tc_id
  - Chi-square            : is syntax validity independent of group
                            (baseline vs. fine-tuned)?
  - Error matrix          : per-category syntax-valid rate, every combo

ROUGE-L F1 is the only quality metric computed on both sides (evaluate_baselines.py
for baselines, agentic_loop/scorer.py inside the BMAD loop for fine-tuned) so it's
the metric used for the quantitative comparison. Syntax validity comes from the
same node --check pass on both sides (validate_syntax.py / validate_baseline_syntax.py).

Usage:
  python -m evaluation.statistical_analysis
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

BASE_DIR     = Path(__file__).parent.parent
DATA_FILE    = BASE_DIR / "data" / "dataset_final.json"
BASELINE_DIR = BASE_DIR / "baselines" / "results"
SCORES_DIR   = Path(__file__).parent / "scores"
FT_DIR       = BASE_DIR / "results" / "agentic_loop"
REPORT_OUT   = Path(__file__).parent / "statistical_analysis_report.json"

BASELINE_MODELS  = ["gpt4o-mini", "claude-haiku"]
FINETUNED_MODELS = ["phi3", "gemma4"]
FRAMEWORKS       = ["cypress", "playwright"]


# ── Load metadata (category / complexity per tc_id) ─────────────────────────
def load_metadata():
    records = json.load(open(DATA_FILE))
    return {r["id"]: {"category": r["category"], "complexity": r["complexity"]} for r in records}


# ── Load per-record rouge_l F1 + syntax validity for one combo ─────────────
def load_baseline_combo(model, framework):
    scores_dir = SCORES_DIR / framework / model
    syntax_report = json.load(open(BASE_DIR / "evaluation" / "baseline_syntax_validation_report.json"))
    syntax_detail = {r["tc_id"]: r["syntax_valid"]
                      for r in syntax_report["detail"][f"{model}/{framework}"]["records"]}

    rows = []
    for f in sorted(scores_dir.glob("TC_*_score.json")):
        s = json.load(open(f))
        tc_id = s["id"]
        rows.append({
            "tc_id": tc_id, "group": "baseline", "model": model, "framework": framework,
            "rouge_l_f1": s["rouge_l"]["f1"], "syntax_valid": syntax_detail.get(tc_id),
        })
    return rows


def load_finetuned_combo(model, framework):
    run_dir = FT_DIR / model / framework
    syntax_report = json.load(open(BASE_DIR / "evaluation" / "syntax_validation_report.json"))
    syntax_detail = {r["tc_id"]: r["syntax_valid"]
                      for r in syntax_report["detail"][f"{model}/{framework}"]["records"]}

    rows = []
    for f in sorted(run_dir.glob("TC_*.json")):
        r = json.load(open(f))
        tc_id = r["tc_id"]
        best_iter = max(r["history"], key=lambda h: h["score"]["total"])
        rows.append({
            "tc_id": tc_id, "group": "finetuned", "model": model, "framework": framework,
            "rouge_l_f1": best_iter["score"]["rouge_l"], "syntax_valid": syntax_detail.get(tc_id),
            "bmad_accepted": r["accepted"],
        })
    return rows


def build_table():
    meta = load_metadata()
    rows = []
    for fw in FRAMEWORKS:
        for m in BASELINE_MODELS:
            rows.extend(load_baseline_combo(m, fw))
        for m in FINETUNED_MODELS:
            rows.extend(load_finetuned_combo(m, fw))
    for row in rows:
        row["category"]   = meta[row["tc_id"]]["category"]
        row["complexity"] = meta[row["tc_id"]]["complexity"]
        row["combo"] = f"{row['model']}/{row['framework']}"
    return rows


# ── Stats helpers ────────────────────────────────────────────────────────────
def group_values(rows, key_fn, value_key="rouge_l_f1"):
    buckets = defaultdict(list)
    for r in rows:
        buckets[key_fn(r)].append(r[value_key])
    return buckets


def paired_values(rows, tc_ids, filt_a, filt_b, value_key="rouge_l_f1"):
    """Return (a_vals, b_vals) aligned by tc_id for two filters over the same framework."""
    a_map = {r["tc_id"]: r[value_key] for r in rows if filt_a(r)}
    b_map = {r["tc_id"]: r[value_key] for r in rows if filt_b(r)}
    common = [tid for tid in tc_ids if tid in a_map and tid in b_map]
    return [a_map[t] for t in common], [b_map[t] for t in common], common


def main():
    rows = build_table()
    tc_ids = sorted({r["tc_id"] for r in rows})
    report = {}

    print(f"\n{'='*70}")
    print("  STATISTICAL EVALUATION — 8 combos x 279 records")
    print(f"{'='*70}\n")

    # ── Descriptive means per combo ─────────────────────────────────────────
    combo_vals = group_values(rows, lambda r: r["combo"])
    print(f"  {'Combo':<24}{'n':>5}{'Mean ROUGE-L F1':>18}{'Std':>10}")
    print(f"  {'-'*24}{'-'*5}{'-'*18}{'-'*10}")
    descriptive = {}
    for combo, vals in combo_vals.items():
        arr = np.array(vals)
        descriptive[combo] = {"n": len(arr), "mean": round(arr.mean(), 4), "std": round(arr.std(ddof=1), 4)}
        print(f"  {combo:<24}{len(arr):>5}{arr.mean():>18.4f}{arr.std(ddof=1):>10.4f}")
    report["descriptive"] = descriptive

    # ── One-way ANOVA across all 8 combos ───────────────────────────────────
    f_stat, p_val = stats.f_oneway(*combo_vals.values())
    print(f"\n  ANOVA (all 8 combos, ROUGE-L F1):  F={f_stat:.4f}  p={p_val:.6g}"
          f"  {'(significant, p<0.05)' if p_val < 0.05 else '(not significant)'}")
    report["anova_all_combos"] = {"F": round(f_stat, 4), "p": p_val, "significant": bool(p_val < 0.05)}

    # ── ANOVA within each framework (4 models each) ─────────────────────────
    report["anova_by_framework"] = {}
    for fw in FRAMEWORKS:
        fw_vals = group_values([r for r in rows if r["framework"] == fw], lambda r: r["model"])
        f_stat, p_val = stats.f_oneway(*fw_vals.values())
        print(f"  ANOVA ({fw} only, 4 models):        F={f_stat:.4f}  p={p_val:.6g}"
              f"  {'(significant)' if p_val < 0.05 else '(not significant)'}")
        report["anova_by_framework"][fw] = {"F": round(f_stat, 4), "p": p_val, "significant": bool(p_val < 0.05)}

    # ── ANOVA baseline vs fine-tuned (2 groups, collapsed across model/framework) ──
    group_vals = group_values(rows, lambda r: r["group"])
    f_stat, p_val = stats.f_oneway(*group_vals.values())
    print(f"\n  ANOVA (baseline vs fine-tuned, collapsed): F={f_stat:.4f}  p={p_val:.6g}"
          f"  {'(significant)' if p_val < 0.05 else '(not significant)'}")
    report["anova_group_collapsed"] = {"F": round(f_stat, 4), "p": p_val, "significant": bool(p_val < 0.05)}

    # ── Wilcoxon signed-rank: paired, per-framework, key comparisons ────────
    print(f"\n  {'-'*70}")
    print("  WILCOXON SIGNED-RANK (paired by tc_id, per framework)")
    print(f"  {'-'*70}")
    report["wilcoxon"] = {}
    comparisons = [
        ("gemma4 vs claude-haiku", lambda r: r["model"] == "gemma4", lambda r: r["model"] == "claude-haiku"),
        ("gemma4 vs gpt4o-mini",   lambda r: r["model"] == "gemma4", lambda r: r["model"] == "gpt4o-mini"),
        ("phi3 vs claude-haiku",   lambda r: r["model"] == "phi3",   lambda r: r["model"] == "claude-haiku"),
        ("phi3 vs gpt4o-mini",     lambda r: r["model"] == "phi3",   lambda r: r["model"] == "gpt4o-mini"),
        ("gemma4 vs phi3",         lambda r: r["model"] == "gemma4", lambda r: r["model"] == "phi3"),
    ]
    for label, fa, fb in comparisons:
        for fw in FRAMEWORKS:
            fw_rows = [r for r in rows if r["framework"] == fw]
            a, b, common = paired_values(fw_rows, tc_ids,
                                          lambda r, fa=fa: fa(r) and True,
                                          lambda r, fb=fb: fb(r) and True)
            if len(common) < 10:
                continue
            try:
                w_stat, p_val = stats.wilcoxon(a, b)
            except ValueError:
                continue
            key = f"{label} [{fw}]"
            mean_diff = round(np.mean(a) - np.mean(b), 4)
            print(f"  {key:<32} n={len(common):<5} mean_diff={mean_diff:+.4f}  "
                  f"W={w_stat:.1f}  p={p_val:.6g}  {'*' if p_val < 0.05 else ''}")
            report["wilcoxon"][key] = {
                "n": len(common), "mean_diff": mean_diff,
                "W": round(w_stat, 2), "p": p_val, "significant": bool(p_val < 0.05),
            }

    # ── Chi-square: syntax validity independent of group? ───────────────────
    print(f"\n  {'-'*70}")
    print("  CHI-SQUARE — syntax validity vs. group (baseline / fine-tuned)")
    print(f"  {'-'*70}")
    contingency = np.array([
        [sum(1 for r in rows if r["group"] == g and r["syntax_valid"] is True) for g in ("baseline", "finetuned")],
        [sum(1 for r in rows if r["group"] == g and r["syntax_valid"] is False) for g in ("baseline", "finetuned")],
    ])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    print(f"  Contingency [valid/invalid x baseline/finetuned]:\n{contingency}")
    print(f"  chi2={chi2:.4f}  dof={dof}  p={p_val:.6g}  {'(significant)' if p_val < 0.05 else '(not significant)'}")
    report["chi_square_syntax_vs_group"] = {
        "contingency": contingency.tolist(), "chi2": round(chi2, 4), "dof": dof,
        "p": p_val, "significant": bool(p_val < 0.05),
    }

    # ── Error matrix: category x combo syntax-valid rate ────────────────────
    print(f"\n  {'-'*70}")
    print("  ERROR MATRIX — syntax-valid rate (%) by category x combo")
    print(f"  {'-'*70}")
    categories = sorted({r["category"] for r in rows})
    combos = sorted({r["combo"] for r in rows})
    col_w = max(len(c) for c in combos) + 3
    matrix = {}
    header = f"  {'Category':<24}" + "".join(f"{c:>{col_w}}" for c in combos)
    print(header)
    for cat in categories:
        line = f"  {cat:<24}"
        matrix[cat] = {}
        for combo in combos:
            subset = [r["syntax_valid"] for r in rows if r["category"] == cat and r["combo"] == combo]
            rate = round(100 * sum(subset) / len(subset), 1) if subset else None
            matrix[cat][combo] = rate
            cell = f"{rate:.1f}%" if rate is not None else "--"
            line += f"{cell:>{col_w}}"
        print(line)
    report["error_matrix_syntax_by_category"] = matrix

    # ── Error matrix: category x combo mean ROUGE-L F1 ──────────────────────
    print(f"\n  {'-'*70}")
    print("  ERROR MATRIX — mean ROUGE-L F1 by category x combo")
    print(f"  {'-'*70}")
    header = f"  {'Category':<24}" + "".join(f"{c:>{col_w}}" for c in combos)
    print(header)
    rouge_matrix = {}
    for cat in categories:
        line = f"  {cat:<24}"
        rouge_matrix[cat] = {}
        for combo in combos:
            subset = [r["rouge_l_f1"] for r in rows if r["category"] == cat and r["combo"] == combo]
            val = round(float(np.mean(subset)), 4) if subset else None
            rouge_matrix[cat][combo] = val
            cell = f"{val:.3f}" if val is not None else "--"
            line += f"{cell:>{col_w}}"
        print(line)
    report["error_matrix_rouge_by_category"] = rouge_matrix

    # Per-category ANOVA: does mean ROUGE-L differ across the 8 combos within this category?
    print(f"\n  Per-category ANOVA (ROUGE-L F1 across 8 combos):")
    report["anova_by_category"] = {}
    for cat in categories:
        cat_vals = group_values([r for r in rows if r["category"] == cat], lambda r: r["combo"])
        f_stat, p_val = stats.f_oneway(*cat_vals.values())
        sig = "*" if p_val < 0.05 else ""
        print(f"    {cat:<24} F={f_stat:>8.3f}  p={p_val:.4g}  {sig}")
        report["anova_by_category"][cat] = {"F": round(f_stat, 4), "p": p_val, "significant": bool(p_val < 0.05)}

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved to: {REPORT_OUT}\n")


if __name__ == "__main__":
    main()
