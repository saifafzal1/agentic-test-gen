"""
BMAD Agentic Correction Loop
=============================
Build → Measure → Assess → Decide

For each user story:
  Build  — call the FastAPI inference service to generate a test script
  Measure — score the script (syntax + assertion density + ROUGE-L)
  Assess  — is the score >= threshold?
  Decide  — accept and return  OR  inject targeted feedback and regenerate

Returns the best-scoring script seen across all iterations.
"""
import time
import requests
from dataclasses import dataclass, field
from typing import Optional

from .scorer import score, QualityScore

API_BASE           = "http://localhost:8000"
DEFAULT_THRESHOLD  = 0.60
DEFAULT_MAX_ITERS  = 3
DEFAULT_MAX_TOKENS = 512    # MPS inference is ~15s/record at this length — acceptable for batch
API_TIMEOUT        = 600    # 10 min — MPS inference is slow (~2–4 min per request)


@dataclass
class IterationResult:
    iteration:    int
    script:       str
    score:        QualityScore
    latency_s:    float
    feedback_used: str = ""


@dataclass
class LoopResult:
    tc_id:           str
    framework:       str
    model_key:       str
    accepted:        bool
    iterations:      int
    best_score:      float
    final_script:    str
    history:         list = field(default_factory=list)
    total_latency_s: float = 0.0


# ── Inference call ────────────────────────────────────────────────────────────

def _generate(
    user_story:    str,
    framework:     str,
    model_key:     str,
    category:      str,
    complexity:    str,
    feedback:      str = "",
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    api_base:      str = API_BASE,
) -> tuple:
    """
    Call POST /generate-test.
    When feedback is non-empty the previous issues are appended to the
    user story so the model has explicit correction guidance.
    """
    story = user_story if not feedback else (
        f"{user_story}\n\n"
        f"[CORRECTION REQUIRED — previous attempt was rejected. "
        f"Please fix the following issues: {feedback}]"
    )
    resp = requests.post(
        f"{api_base}/generate-test",
        json={
            "user_story":     story,
            "framework":      framework,
            "model_key":      model_key,
            "category":       category,
            "complexity":     complexity,
            "max_new_tokens": max_new_tokens,
        },
        timeout=API_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["script"], data["latency_s"]


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(
    tc_id:      str,
    user_story: str,
    framework:  str,
    model_key:  str   = "phi3",
    category:   str   = "functional",
    complexity: str   = "medium",
    exemplar:   str   = "",
    threshold:  float = DEFAULT_THRESHOLD,
    max_iters:  int   = DEFAULT_MAX_ITERS,
    max_tokens: int   = DEFAULT_MAX_TOKENS,
    api_base:   str   = API_BASE,
) -> LoopResult:
    """
    Run the BMAD loop for a single user story.

    Args:
        tc_id:      Record identifier (e.g. 'TC_001')
        user_story: Natural-language requirement
        framework:  'cypress' or 'playwright'
        model_key:  'phi3' or 'gemma4'
        exemplar:   Ground-truth script for ROUGE-L scoring (optional)
        threshold:  Minimum composite score to accept (default 0.60)
        max_iters:  Maximum generation attempts (default 3)

    Returns:
        LoopResult with accepted flag, best script, and full iteration history
    """
    best:          Optional[IterationResult] = None
    history:       list[IterationResult]    = []
    feedback:      str                      = ""
    total_latency: float                    = 0.0

    for i in range(1, max_iters + 1):
        t0 = time.time()
        script, _ = _generate(
            user_story, framework, model_key,
            category, complexity, feedback, max_tokens, api_base,
        )
        elapsed = round(time.time() - t0, 3)
        total_latency += elapsed

        q = score(script, framework, exemplar)

        iteration = IterationResult(
            iteration    = i,
            script       = script,
            score        = q,
            latency_s    = elapsed,
            feedback_used = feedback,
        )
        history.append(iteration)

        # Track best seen
        if best is None or q.total > best.score.total:
            best = iteration

        # Assess — accept if threshold met
        if q.total >= threshold:
            return LoopResult(
                tc_id          = tc_id,
                framework      = framework,
                model_key      = model_key,
                accepted       = True,
                iterations     = i,
                best_score     = best.score.total,
                final_script   = best.script,
                history        = history,
                total_latency_s = round(total_latency, 3),
            )

        # Decide — prepare targeted feedback for next Build step
        feedback = q.feedback

    # Exhausted iterations — return best candidate found
    return LoopResult(
        tc_id          = tc_id,
        framework      = framework,
        model_key      = model_key,
        accepted       = False,
        iterations     = max_iters,
        best_score     = best.score.total if best else 0.0,
        final_script   = best.script      if best else "",
        history        = history,
        total_latency_s = round(total_latency, 3),
    )
