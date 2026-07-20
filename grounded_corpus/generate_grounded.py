"""
Grounded story+script generation.

For every crawled page profile, asks GPT-4o-mini to author one Jira-style
user story plus a Cypress and a Playwright script, constrained to the REAL
selectors extracted from the live DOM. Candidates go to
grounded_corpus/candidates/ for the execution-admission filter
(filter_corpus.py) -- nothing enters the corpus until it passes live.

Usage: .venv/bin/python grounded_corpus/generate_grounded.py [--per-page N]
"""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE.parent / ".env")

PROFILES = HERE / "profiles"
CANDIDATES = HERE / "candidates"

SYSTEM = (
    "You are a senior QA automation engineer. You write end-to-end tests "
    "grounded in a REAL application whose actual DOM elements are provided. "
    "You must use ONLY selectors, URLs, credentials and texts that appear in "
    "the provided element inventory. Never invent selectors, routes, network "
    "intercepts, or custom commands."
)

PROMPT = """Application: {app_key} at {base_url}
Page under test: {url} ("{feature}")
Functional category: {category}

REAL element inventory extracted from the live page (use ONLY these):
{dom}

{auth_note}

Write ONE Jira-style user story for this page's "{feature}" functionality and
two matching test scripts. Requirements:
- The user story must include acceptance criteria that cite the exact
  selectors from the inventory (quote them literally).
- cypress_script: complete Cypress JavaScript spec; cy.visit the full URL
  {url}; ONE describe() containing exactly 2 it() tests (one happy path,
  one edge/negative case that this page genuinely supports).
- playwright_script: the equivalent complete Playwright TypeScript spec
  (import from '@playwright/test'; ONE test.describe() containing exactly
  2 test() blocks; page.goto the full URL).
- ASSERTION RULES (critical): assert ONLY on (a) elements present in the
  inventory, (b) URL changes, or (c) text strings that appear verbatim in
  the inventory. The inventory is a snapshot BEFORE any interaction --
  you do NOT know what messages appear after clicks, so never assert on
  guessed message wording; assert visibility/count/URL instead.
- The edge/negative test must also be verifiable from the inventory
  alone: e.g. submit an empty form and assert the URL did not change or
  an inventory element is still visible. Do NOT assert specific error
  text unless that exact text appears in the inventory.
- Keep each test to the simplest behaviour the page genuinely supports;
  do not script multi-page flows beyond the page under test. Rely on
  auto-waiting assertions (cy.get(...).should(...), await expect(...))
  rather than fixed sleeps.
- FORMATTING (critical): write the code multi-line with real newline
  characters (\\n) inside the JSON strings -- normal indented code, one
  statement per line, never the whole script on a single line. Balance
  all braces; end each spec with the describe's closing marker only.
- No markdown fences. Return strict JSON:
{{"user_story": "...", "complexity": "simple|medium|complex",
  "cypress_script": "...", "playwright_script": "..."}}"""

AUTH_NOTES = {
    "saucedemo": ("If the page requires login first, log in inside the test "
                  "using username 'standard_user' and password 'secret_sauce' "
                  "via [data-test=\"username\"], [data-test=\"password\"], "
                  "[data-test=\"login-button\"] at https://www.saucedemo.com/."),
    "the-internet": ("The /login page accepts username 'tomsmith' and password "
                     "'SuperSecretPassword!' (documented demo credentials)."),
}


def dom_summary(dom: dict) -> str:
    keep = {k: dom.get(k) for k in
            ("title", "headings", "inputs", "buttons", "links", "selects",
             "data_test_elements", "ids")}
    return json.dumps(keep, indent=1)[:6000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-page", type=int, default=1)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    CANDIDATES.mkdir(exist_ok=True)
    n = 0
    for prof_path in sorted(PROFILES.glob("*.json")):
        prof = json.loads(prof_path.read_text())
        if prof.get("skip"):
            print(f"  skip app {prof['key']} (marked skip in profile)")
            continue
        for pi, page in enumerate(prof["pages"]):
            if page.get("error"):
                continue
            dom = page.get("dom", {})
            n_elements = sum(len(dom.get(k, [])) for k in
                             ("inputs", "buttons", "links", "selects",
                              "data_test_elements", "ids"))
            if n_elements < 4:
                print(f"  skip {prof['key']}{page['path']} (sparse DOM: "
                      f"{n_elements} elements)")
                continue
            for k in range(args.per_page):
                cid = f"GC_{prof['key']}_{pi:02d}_{k}"
                out = CANDIDATES / f"{cid}.json"
                if out.exists():
                    continue
                prompt = PROMPT.format(
                    app_key=prof["key"], base_url=prof["base_url"],
                    url=page["url"], feature=page["feature"],
                    category=page["category"], dom=dom_summary(page["dom"]),
                    auth_note=AUTH_NOTES.get(prof["key"], ""))
                t = time.time()
                data = None
                for attempt in range(3):
                    try:
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            response_format={"type": "json_object"},
                            messages=[{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": prompt}],
                            # v2 evidence: 0.7-temp admitted at 38%,
                            # 1.0-temp at 17% -- cap at 0.7.
                            max_tokens=4000,
                            temperature=min(0.4 + 0.15 * k, 0.7))
                        data = json.loads(resp.choices[0].message.content)
                        break
                    except (json.JSONDecodeError, Exception) as e:  # noqa: BLE001
                        print(f"  {cid} attempt {attempt+1} failed: "
                              f"{str(e)[:100]}")
                        time.sleep(2 * (attempt + 1))
                if data is None:
                    print(f"  {cid} SKIPPED after 3 attempts")
                    continue
                data.update(id=cid, app=prof["key"], base_url=prof["base_url"],
                            page=page["path"], category=page["category"],
                            latency_s=round(time.time() - t, 1))
                out.write_text(json.dumps(data, indent=2))
                n += 1
                print(f"  {cid} generated ({data['latency_s']}s)")
    print(f"\n{n} new candidates in {CANDIDATES}")


if __name__ == "__main__":
    main()
