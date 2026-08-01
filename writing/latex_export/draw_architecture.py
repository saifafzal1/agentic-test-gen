"""
Figure 3.1 -- End-to-end system architecture.
Rebuilt per faculty feedback: larger, readable text; arrows routed so no
connector crosses a box (the baseline path runs down the left margin,
clear of the BMAD box). High-DPI for print.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/Users/saif.afzal/Documents/Dissertation/agentic-test-gen/writing/latex_export/images/architecture.png"

FILL = "#EAF0F7"       # light blue-gray
FILL2 = "#F3ECF7"      # light lavender for the two generation paths
EDGE = "#2A3B4D"
TXT = "#1A2430"

fig, ax = plt.subplots(figsize=(8.2, 11.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# ---- boxes: (cx, cy, w, h, title, detail, fill) ----
boxes = {
    "data":  (50, 91, 60, 8.5, "1. Data Layer",
              "279 aligned user stories | framework-isolated\nCypress / Playwright JSONL splits", FILL),
    "base":  (26, 73, 40, 9.5, "2a. Baseline Inference Engine",
              "GPT-4o-mini, Claude Haiku\n(zero-shot, commercial APIs)", FILL2),
    "ft":    (74, 73, 40, 9.5, "2b. Fine-Tuning Module",
              "QLoRA adapters: Phi-3 Mini,\nGemma 4 E4B (per framework)", FILL2),
    "bmad":  (60, 52, 46, 11, "3. BMAD Agentic Correction Loop",
              "Build -> Measure -> Assess -> Decide\nautonomous generate + iterative revise\n(bounded at 3 correction iterations)", FILL),
    "eval":  (50, 30, 60, 11, "4. Evaluation & Statistical Analysis",
              "syntax validity (node --check), ROUGE-L / token-F1,\nkeyword coverage; ANOVA, Wilcoxon, chi-square", FILL),
    "res":   (50, 11, 48, 8.5, "5. Comparative Results",
              "8 systems x 279 stories x 2 frameworks\nquality, validity, execution pilot (Chapter 5-6)", FILL),
}

coords = {}
for key, (cx, cy, w, h, title, detail, fill) in boxes.items():
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
        linewidth=1.8, edgecolor=EDGE, facecolor=fill, zorder=2))
    ax.text(cx, cy + h * 0.20, title, ha="center", va="center",
            fontsize=13.5, fontweight="bold", color=TXT, zorder=3)
    ax.text(cx, cy - h * 0.20, detail, ha="center", va="center",
            fontsize=9.5, color=TXT, zorder=3, linespacing=1.35)
    coords[key] = dict(cx=cx, cy=cy, w=w, h=h, top=cy + h / 2,
                       bot=cy - h / 2, left=cx - w / 2, right=cx + w / 2)


def arrow(p0, p1, style="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=style, arrowstyle="-|>",
        mutation_scale=22, linewidth=1.9, color=EDGE, zorder=1))


d, b, f, m, e, r = (coords[k] for k in ("data", "base", "ft", "bmad", "eval", "res"))

# Data Layer -> both generation paths (straight diagonals, no box between)
arrow((d["cx"] - 14, d["bot"]), (b["cx"] + 4, b["top"]))
arrow((d["cx"] + 14, d["bot"]), (f["cx"] - 4, f["top"]))
# Fine-Tuning -> BMAD
arrow((f["cx"], f["bot"]), (m["cx"] + 6, m["top"]))
# BMAD -> Evaluation
arrow((m["cx"] - 6, m["bot"]), (e["cx"] + 6, e["top"]))
# Baseline path -> Evaluation, routed down the LEFT margin then right into
# the box's left side (clean L-shape, clear of the BMAD box). Start x and
# end x differ so the elbow is well-defined.
arrow((b["cx"] - 8, b["bot"]), (e["left"], e["cy"]),
      style="angle,angleA=-90,angleB=0,rad=8")
# Evaluation -> Results
arrow((e["cx"], e["bot"]), (r["cx"], r["top"]))

ax.text(50, 98.5, "Figure 3.1: End-to-end System Architecture",
        ha="center", va="center", fontsize=16, fontweight="bold", color=TXT)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("saved", OUT)
