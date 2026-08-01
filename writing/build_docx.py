"""
Build a single Word .docx from the finalized LaTeX sources via pandoc.
Preprocessing: flatten \\input order, resolve \\IfFileExists{path}{A}{B} -> A
(files exist), drop LaTeX-only preamble/titlepage plumbing pandoc can't use.
"""
import re
import subprocess
from pathlib import Path

SRC = Path("/Users/saif.afzal/Documents/Dissertation/agentic-test-gen/writing/latex_export")
OUT_TEX = Path("/private/tmp/claude-1670252721/-Users-saif-afzal-Documents-Dissertation/651df370-dcc6-41dd-90d7-b12eb01592c6/scratchpad/combined.tex")
OUT_DOCX = Path("/Users/saif.afzal/Documents/Dissertation/agentic-test-gen/writing/Agentic_AI_Final_Report_Saif_Afzal_2024AA05546.docx")

# Document order (matches main.tex).
ORDER = [
    "CoverPage", "TitlePage", "CertificatePage", "FrontMatter",
    "Intro_LitReview_Chapters", "Architecture_Chapter", "Implementation_Chapter",
    "Evaluation_Chapter", "Conclusion_Chapter", "Appendices", "BackMatter",
    "ChecklistPage",
]


def resolve_iffileexists(s: str) -> str:
    # \IfFileExists{path}{THEN}{ELSE} -> THEN  (brace-balanced, may span lines)
    out = []
    i = 0
    key = r"\IfFileExists"
    while i < len(s):
        j = s.find(key, i)
        if j == -1:
            out.append(s[i:])
            break
        out.append(s[i:j])
        k = j + len(key)
        args = []
        while len(args) < 3:
            while k < len(s) and s[k] in " \n\t%":
                k += 1
            if k >= len(s) or s[k] != "{":
                break
            depth, start = 0, k
            while k < len(s):
                if s[k] == "{":
                    depth += 1
                elif s[k] == "}":
                    depth -= 1
                    if depth == 0:
                        args.append(s[start + 1:k]); k += 1; break
                k += 1
        out.append(args[1] if len(args) >= 2 else "")  # THEN branch
        i = k
    return "".join(out)


parts = []
for name in ORDER:
    raw = (SRC / f"{name}.tex").read_text()
    raw = resolve_iffileexists(raw)
    # CoverPage/TitlePage are centred title plumbing; keep text, drop the env.
    raw = raw.replace("\\begin{titlepage}", "").replace("\\end{titlepage}", "")
    raw = raw.replace("\\centering", "")
    # Chapter breaks between files
    parts.append(raw)

body = "\n\n\\clearpage\n\n".join(parts)

preamble = r"""\documentclass[12pt]{report}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{booktabs}
\graphicspath{{%s/}}
\begin{document}
""" % SRC

OUT_TEX.write_text(preamble + body + "\n\\end{document}\n")

subprocess.run([
    "pandoc", str(OUT_TEX), "-f", "latex", "-o", str(OUT_DOCX),
    "--toc", "--toc-depth=3", "--reference-doc=/private/tmp/claude-1670252721/-Users-saif-afzal-Documents-Dissertation/651df370-dcc6-41dd-90d7-b12eb01592c6/scratchpad/ref_quarto.docx",
    f"--resource-path={SRC}",
], check=True)
print("wrote", OUT_DOCX)
