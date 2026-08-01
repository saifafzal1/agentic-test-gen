"""
Figure 4.2 -- Dataset generation flowchart.
Per faculty feedback: readable text, and drawn as a proper flowchart
(process boxes, a decision diamond, and the regenerate-on-failure loop).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D

OUT = "/Users/saif.afzal/Documents/Dissertation/agentic-test-gen/writing/latex_export/images/dataset_generation.png"
PROC = "#EAF0F7"     # process box
IO = "#EAF6EC"       # input/output
DEC = "#FBF1DA"      # decision
EDGE = "#2A3B4D"
TXT = "#1A2430"

fig, ax = plt.subplots(figsize=(8.4, 10.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

CX = 56  # main column centre (left margin kept free for the retry loop)


def box(cx, cy, w, h, title, detail, fill):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.5,rounding_size=1.1", linewidth=1.8,
                 edgecolor=EDGE, facecolor=fill, zorder=2))
    ax.text(cx, cy + h * 0.24, title, ha="center", va="center",
            fontsize=13, fontweight="bold", color=TXT, zorder=3)
    if detail:
        ax.text(cx, cy - h * 0.18, detail, ha="center", va="center",
                fontsize=9.2, color=TXT, zorder=3, linespacing=1.3)


def diamond(cx, cy, w, h, label):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=1.8, edgecolor=EDGE,
                 facecolor=DEC, zorder=2))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=12,
            fontweight="bold", color=TXT, zorder=3)


def arrow(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=20,
                 linewidth=1.9, color=EDGE, zorder=1))


# ---- nodes ----
box(CX, 93, 60, 8, "Input Specification",
    "16 functional categories · 3 complexity tiers · 2 frameworks\n"
    "(authentication, CRUD, forms, navigation, search, upload, ...)", IO)
box(CX, 78, 60, 9, "Claude Haiku Generation  [16]",
    "one user story + Cypress script + Playwright script\n"
    "temperature 0.7, max_tokens 2048, structured output", PROC)
# validator with two rule sub-columns
box(CX, 56, 74, 16, "Heuristic Validator", "", PROC)
ax.text(CX - 18, 57.5, "Cypress rules", ha="center", fontsize=10,
        fontweight="bold", color=TXT, zorder=3)
ax.text(CX - 18, 53.6, "describe(), cy.visit(), cy.get()\n"
        "it(), type(), should(), click()\n≥ 2 interaction methods",
        ha="center", va="center", fontsize=8.6, color=TXT, zorder=3, linespacing=1.3)
ax.text(CX + 18, 57.5, "Playwright rules", ha="center", fontsize=10,
        fontweight="bold", color=TXT, zorder=3)
ax.text(CX + 18, 53.6, "test(), page.goto(), page.click()\n"
        "getByRole(), getByLabel(), expect()\n≥ 2 modern locators",
        ha="center", va="center", fontsize=8.6, color=TXT, zorder=3, linespacing=1.3)
diamond(CX, 36, 22, 12, "Valid?")
box(CX, 15, 64, 9, "dataset_final.json",
    "279 aligned records · 16 categories × 3 tiers\n"
    "cypress_dataset.jsonl + playwright_dataset.jsonl · 70/15/15 split", IO)

# ---- straight-through flow ----
arrow((CX, 89), (CX, 82.5))
arrow((CX, 73.5), (CX, 64))
arrow((CX, 48), (CX, 42))
arrow((CX, 30), (CX, 19.5))
ax.text(CX + 2.5, 24.5, "Yes", ha="left", va="center", fontsize=11,
        fontweight="bold", color="#1B7A3D", zorder=3)

# ---- regenerate-on-failure loop (down the left margin, clear of all boxes) ----
lane = 12
ax.plot([CX - 11, lane], [36, 36], linewidth=1.9, color=EDGE, zorder=1)
ax.plot([lane, lane], [36, 78], linewidth=1.9, color=EDGE, zorder=1)
arrow((lane, 78), (CX - 30, 78))
ax.text(lane + 1.0, 41, "No — regenerate\n(max 3 attempts)", ha="left",
        va="center", fontsize=10.5, fontweight="bold", color="#B23A2E",
        zorder=3, linespacing=1.3)

ax.text(50, 99, "Figure 4.2: Dataset Generation Flowchart", ha="center",
        va="center", fontsize=16, fontweight="bold", color=TXT)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("saved", OUT)
