"""
Task 7 — deterministic selector-repair (ENFORCED grounding).

Prompt-injection grounding (Task 4-6) suggested the real selectors but the
fine-tuned model overrode them with its trained `data-testid` convention.
This module ENFORCES grounding: it rewrites the model's invented selectors to
the real ones present in the crawled DOM inventory, deterministically.

Rules (applied to a generated script, using that story's real inventory):
  A. data-testid="X"  ->  data-test="X"      when X is a real data-test value
                                              (saucedemo: same value, wrong attr name)
  B. [data-testid="X"] ->  #<id>             when X maps to a real element id
                                              (the-internet: id-based selectors)

Repair uses ONLY selectors that exist in the real DOM — it cannot invent.
"""
import re

from grounding_experiment.dom_context import _match_pages


def inventory(story: dict):
    """Return (data_test_values, id_values) sets from the story's matched pages."""
    dts, ids = set(), set()
    for dom in _match_pages(story):
        for el in dom.get("data_test_elements", []):
            v = el.get("data-test") or el.get("data-testid")
            if v:
                dts.add(v)
        for el in dom.get("ids", []):
            i = el.get("id") if isinstance(el, dict) else None
            if i:
                ids.add(i)
    return dts, ids


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def repair(script: str, dts: set, ids: set) -> tuple:
    """Return (repaired_script, n_changes)."""
    changes = 0

    # Rule A: rename data-testid -> data-test when the value is a real data-test
    def rule_a(m):
        nonlocal changes
        q, x = m.group(1), m.group(2)
        if x in dts:
            changes += 1
            return f"data-test={q}{x}{q}"
        return m.group(0)
    script = re.sub(r"data-testid=([\"'])([^\"']+)\1", rule_a, script)

    # Rule B: [data-testid="X"] -> #id when X maps (normalised) to a real id
    id_by_norm = {_norm(i): i for i in ids}
    def rule_b(m):
        nonlocal changes
        x = m.group(2)
        key = _norm(x)
        if key in id_by_norm:
            changes += 1
            return f"#{id_by_norm[key]}"
        return m.group(0)
    script = re.sub(r"\[data-testid=([\"'])([^\"']+)\1\]", rule_b, script)

    return script, changes


if __name__ == "__main__":
    import json, glob
    stories = {json.loads(l)["id"]: json.loads(l) for l in
               open("data/execution_validation/grounded_stories.jsonl")}
    for f in sorted(glob.glob("results/grounding_experiment/on/phi3/cypress/TC_G0[1-3].json")):
        rec = json.load(open(f))
        s = stories[rec["tc_id"]]
        dts, ids = inventory(s)
        rep, n = repair(rec["final_script"], dts, ids)
        print(f"\n===== {rec['tc_id']} | {n} selector(s) repaired | dts={sorted(dts)[:4]} ids~{len(ids)} =====")
        # show the selector lines before/after
        for line in rep.splitlines():
            if "data-test" in line or "#" in line and "cy.get" in line:
                print("  ", line.strip()[:90])
