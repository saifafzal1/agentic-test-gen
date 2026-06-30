"""
Quality scorer for generated test scripts.

Computes a composite score (0.0 – 1.0) from three dimensions:
  1. Syntax completeness  (0.40 weight) — required keywords present
  2. Assertion density    (0.30 weight) — assertions per test block
  3. ROUGE-L similarity   (0.30 weight) — overlap with ground-truth exemplar
"""
import re
from dataclasses import dataclass

# ── Per-framework keyword lists ───────────────────────────────────────────────

CYPRESS_REQUIRED    = ["describe(", "beforeEach(", "it(", ".should("]
CYPRESS_RECOMMENDED = ["cy.visit(", "cy.get(", "cy.click(", "cy.type("]

PLAYWRIGHT_REQUIRED    = ["expect(", "test("]
PLAYWRIGHT_RECOMMENDED = ["test.describe(", "test.beforeEach(", "getByRole(", "getByLabel("]

_CY_ASSERT  = re.compile(r'\.should\(')
_CY_BLOCK   = re.compile(r'\bit\(')
_PW_ASSERT  = re.compile(r'\bexpect\(')
_PW_BLOCK   = re.compile(r'\btest\(')

TARGET_DENSITY = 2.0   # minimum assertions per test block


@dataclass
class QualityScore:
    total:            float        # composite 0–1
    syntax:           float        # keyword completeness
    assertion:        float        # assertion density
    rouge_l:          float        # ROUGE-L vs exemplar
    missing_keywords: list
    assertion_count:  int
    feedback:         str          # specific issues for retry prompt


# ── ROUGE-L (no external dependencies) ───────────────────────────────────────

def _rouge_l(hyp: str, ref: str) -> float:
    """Token-level ROUGE-L F1 via LCS dynamic programming."""
    h = hyp.lower().split()
    r = ref.lower().split()
    if not h or not r:
        return 0.0
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if r[i-1] == h[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec   = lcs / n if n else 0.0
    recall = lcs / m if m else 0.0
    return 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0


# ── Main scorer ───────────────────────────────────────────────────────────────

def score(script: str, framework: str, exemplar: str = "") -> QualityScore:
    """Score a generated test script and return a QualityScore."""
    fw = framework.lower()

    if fw == "cypress":
        required    = CYPRESS_REQUIRED
        recommended = CYPRESS_RECOMMENDED
        assert_re   = _CY_ASSERT
        block_re    = _CY_BLOCK
    else:
        required    = PLAYWRIGHT_REQUIRED
        recommended = PLAYWRIGHT_RECOMMENDED
        assert_re   = _PW_ASSERT
        block_re    = _PW_BLOCK

    # 1. Syntax score
    missing          = [kw for kw in required if kw not in script]
    present_required = len(required) - len(missing)
    present_rec      = sum(1 for kw in recommended if kw in script)
    syntax_score = (
        0.7 * (present_required / len(required)) +
        0.3 * (present_rec      / len(recommended))
    )

    # 2. Assertion density score
    assertions    = len(assert_re.findall(script))
    blocks        = max(len(block_re.findall(script)), 1)
    density       = assertions / blocks
    assert_score  = min(density / TARGET_DENSITY, 1.0)

    # 3. ROUGE-L score (neutral 0.5 when no exemplar provided)
    rouge = _rouge_l(script, exemplar) if exemplar else 0.5

    # Composite
    total = round(0.4 * syntax_score + 0.3 * assert_score + 0.3 * rouge, 4)

    # Build human-readable feedback for the retry prompt
    issues = []
    if missing:
        issues.append(f"Missing required elements: {', '.join(missing)}")
    if assert_score < 0.7:
        need = int(TARGET_DENSITY * blocks)
        issues.append(
            f"Insufficient assertions — found {assertions} in {blocks} block(s), need at least {need}"
        )
    if syntax_score < 0.6:
        issues.append("Incomplete script structure — ensure describe/beforeEach/it blocks are present")
    feedback = "; ".join(issues) if issues else "Script meets quality threshold."

    return QualityScore(
        total            = total,
        syntax           = round(syntax_score, 4),
        assertion        = round(assert_score, 4),
        rouge_l          = round(rouge, 4),
        missing_keywords = missing,
        assertion_count  = assertions,
        feedback         = feedback,
    )
