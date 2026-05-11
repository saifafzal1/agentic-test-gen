"""
Synthetic Dataset Generator — v4 (Step 1: Strengthened getByRole + Regen all)
=============================================================================
Improvements over v3:
  - PLAYWRIGHT prompt now mandates getByRole/getByLabel as FIRST CHOICE (non-negotiable)
  - Validation updated: pw_getBy requires >= 2 modern locator calls OR page.fill >= 2
  - Regenerates ALL existing 237 pairs with new prompt (not just new categories)
  - Adds 4 remaining categories: data_export, multi_step_form, drag_drop, pagination
  - Metadata fields added: cy_line_count, pw_line_count, has_api_mock, cy_assertion_count

Run on Mac B:
  cd ~/dissertation-project
  source .venv/bin/activate
  python data/generate_dataset_v3.py
"""

import os, re, json, time, logging
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
EXISTING_FILE   = Path(__file__).parent / "dataset_280.json"   # original 237 stories to regen
OUTPUT_FILE     = Path(__file__).parent / "dataset_final.json"       # final fine-tuning dataset
CHECKPOINT_FILE = Path(__file__).parent / "dataset_final_checkpoint.json"
ERROR_LOG       = Path(__file__).parent / "errors_final.log"
API_KEY         = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL           = "claude-haiku-4-5"
DELAY           = 0.5
TIMEOUT         = 60
MAX_RETRIES     = 3

logging.basicConfig(
    filename=ERROR_LOG,
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ── Required keywords for quality validation ──────────────────────────────────
CYPRESS_REQUIRED     = ["cy.get(", "cy.visit(", "describe(", "it(", "beforeEach("]
CYPRESS_INTERACTION  = [".click(", ".type(", ".should("]   # at least 2 of 3
PLAYWRIGHT_REQUIRED  = ["test(", "expect(", "await ", "page.goto(", "test.beforeEach("]
PLAYWRIGHT_MODERN_LOCATORS = ["getByRole(", "getByLabel(", "getByText(", "getByPlaceholder("]  # must have >= 2
PLAYWRIGHT_FILL_FALLBACK   = "page.fill("   # accept if appears >= 2 times as alternative

# ── Strengthened System Prompt (v4) ──────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior QA automation engineer specialising in Cypress and Playwright.
Given a Jira user story, generate production-quality test scripts for BOTH frameworks.

CYPRESS RULES (all mandatory):
- Use describe() with a meaningful suite name
- Use beforeEach() with cy.visit('/relevant-path')
- Write exactly 2 it() blocks: one happy path, one negative/edge case
- Use cy.get('[data-testid="..."]') for element selection
- MUST use .click() for button/link interactions
- MUST use .type('value') for text input
- MUST use .should('be.visible'), .should('contain','text'), .should('not.exist') for assertions
- MUST use cy.intercept('METHOD', '/api/endpoint').as('alias') for API calls
- Use cy.wait('@alias') after cy.intercept()

PLAYWRIGHT RULES (all mandatory):
- Import: import { test, expect } from '@playwright/test';
- Use test.describe() with a meaningful suite name
- Use test.beforeEach() with await page.goto('/relevant-path')
- Write exactly 2 test() blocks: one happy path, one negative/edge case
- *** LOCATOR RULE — STRICTLY ENFORCED ***:
  * FIRST CHOICE: page.getByRole('button', {{ name: '...' }}) or page.getByLabel('...')
  * SECOND CHOICE: page.getByText('...') or page.getByPlaceholder('...')
  * page.locator() is FORBIDDEN unless the above are impossible
  * Every test MUST call getByRole() or getByLabel() at least TWICE
- Use page.fill() for text inputs (after locating with getByLabel or getByPlaceholder)
- Use page.click() on located elements
- MUST use expect(locator).toBeVisible(), .toHaveValue('x'), .toContainText('x'), .toBeDisabled()
- All functions MUST use async/await

OUTPUT FORMAT — use ONLY these tags, no JSON, no markdown, no explanation:
<CYPRESS>
[full Cypress script]
</CYPRESS>
<PLAYWRIGHT>
[full Playwright script]
</PLAYWRIGHT>"""

USER_TEMPLATE = """User story: "{story}"
Category: {category}
Complexity: {complexity}

Write complete, realistic, production-quality scripts. Do NOT truncate — include all setup,
interactions, assertions and edge cases needed to fully test this story. Aim for 50-80 lines
per script. Exactly 2 it/test blocks each (happy path + negative/edge case).
REMINDER: Playwright MUST use getByRole() or getByLabel() at least twice per test file.
Output ONLY the two tag blocks:

<CYPRESS>
[Cypress script]
</CYPRESS>
<PLAYWRIGHT>
[Playwright script]
</PLAYWRIGHT>"""

# ── New User Stories (3 new categories × 15 stories) ─────────────────────────
NEW_USER_STORIES = {
    "payment_flow": [
        "As a user, I want to enter my card details and complete a purchase successfully.",
        "As a user, I want to see an error when I enter an invalid card number.",
        "As a user, I want to see an error when my card is declined.",
        "As a user, I want to apply a promo code and see the discounted total.",
        "As a user, I want to see an error when I enter an expired promo code.",
        "As a user, I want to save my card details for future purchases.",
        "As a user, I want to select PayPal as my payment method.",
        "As a user, I want to see an order summary before confirming payment.",
        "As a user, I want to receive an order confirmation after successful payment.",
        "As a user, I want to see a loading state while my payment is processing.",
        "As a user, I want to see an itemised receipt after checkout.",
        "As a user, I want to change my billing address during checkout.",
        "As a user, I want the checkout form to validate card expiry in MM/YY format.",
        "As a user, I want to see the total price update when I change the quantity.",
        "As a user, I want to be redirected to an error page if payment times out.",
    ],
    "notifications": [
        "As a user, I want to see a success toast notification after saving a record.",
        "As a user, I want to see an error toast notification when an action fails.",
        "As a user, I want to dismiss a notification by clicking the close button.",
        "As a user, I want notifications to auto-dismiss after 5 seconds.",
        "As a user, I want to see a badge count on the notification bell icon.",
        "As a user, I want to mark all notifications as read with one click.",
        "As a user, I want to click a notification to navigate to the related content.",
        "As a user, I want to see an empty state when I have no notifications.",
        "As a user, I want to receive a browser push notification for urgent alerts.",
        "As a user, I want to filter notifications by type (info, warning, error).",
        "As a user, I want to see a notification panel slide in from the top.",
        "As a user, I want unread notifications to be visually distinct from read ones.",
        "As a user, I want to see a snackbar message after copying text to clipboard.",
        "As a user, I want in-app notifications to update in real-time without refresh.",
        "As a user, I want to manage my notification preferences in settings.",
        # top-up to reach 17 (buffer for skips → expect ~15 generated)
        "As a user, I want to delete a single notification from my notification list.",
        "As a user, I want to see a notification count reset to zero after reading all messages.",
    ],
    "user_profile": [
        "As a user, I want to update my display name and see it reflected across the app.",
        "As a user, I want to upload a profile picture and see the preview.",
        "As a user, I want to see an error if I upload a non-image file as my avatar.",
        "As a user, I want to change my email address with a confirmation step.",
        "As a user, I want to change my password from the profile settings page.",
        "As a user, I want to see my account creation date on my profile page.",
        "As a user, I want to toggle dark mode from my preferences.",
        "As a user, I want to set my timezone in profile settings.",
        "As a user, I want to delete my account after confirming with my password.",
        "As a user, I want to see a public profile page with my shared information.",
        "As a user, I want to connect my Google account to my profile.",
        "As a user, I want to see all active sessions and log out of specific ones.",
        "As a user, I want to export my personal data as a JSON file.",
        "As a user, I want to set my language preference in profile settings.",
        "As a user, I want to see a confirmation message after updating my profile.",
    ],
}

# ── Complexity assignment ─────────────────────────────────────────────────────
def get_complexity(idx: int) -> str:
    """Rotate through complexity tiers: simple → medium → complex."""
    tier = idx % 3
    return ["medium", "complex", "simple"][tier]

# ── XML-tag extractor (avoids all JSON escaping issues) ───────────────────────
def extract_scripts(raw: str) -> dict:
    """Extract scripts from <CYPRESS>...</CYPRESS> and <PLAYWRIGHT>...</PLAYWRIGHT> tags."""
    cy_match = re.search(r'<CYPRESS>\s*([\s\S]*?)\s*</CYPRESS>', raw, re.DOTALL)
    pw_match = re.search(r'<PLAYWRIGHT>\s*([\s\S]*?)\s*</PLAYWRIGHT>', raw, re.DOTALL)

    if not cy_match:
        raise ValueError(f"Missing <CYPRESS> tag in response: {raw[:200]}")
    if not pw_match:
        raise ValueError(f"Missing <PLAYWRIGHT> tag in response: {raw[:200]}")

    return {
        "cypress_script":    cy_match.group(1).strip(),
        "playwright_script": pw_match.group(1).strip(),
    }

# ── Quality validation (v4 — stricter pw getBy) ───────────────────────────────
def validate_scripts(scripts: dict, category: str) -> tuple[bool, str]:
    cy = scripts.get("cypress_script", "")
    pw = scripts.get("playwright_script", "")

    # Cypress: required structure
    for kw in CYPRESS_REQUIRED:
        if kw not in cy:
            return False, f"Cypress missing required keyword: {kw}"

    # Cypress: need at least 2 of 3 interactions
    cy_interactions = sum(1 for kw in CYPRESS_INTERACTION if kw in cy)
    if cy_interactions < 2:
        return False, f"Cypress needs >=2 of (click/type/should) — got {cy_interactions}/3"

    # Playwright: required structure
    for kw in PLAYWRIGHT_REQUIRED:
        if kw not in pw:
            return False, f"Playwright missing required keyword: {kw}"

    # Playwright: MUST have >= 2 modern locator calls (getByRole/getByLabel/getByText/getByPlaceholder)
    # Fallback accepted: page.fill() appears >= 2 times (also a modern pattern)
    modern_locator_count = sum(pw.count(kw) for kw in PLAYWRIGHT_MODERN_LOCATORS)
    fill_count = pw.count(PLAYWRIGHT_FILL_FALLBACK)
    if modern_locator_count < 2 and fill_count < 2:
        return False, (
            f"Playwright needs >=2 getByRole/getByLabel/getByText calls "
            f"(got {modern_locator_count}) or >=2 page.fill() calls (got {fill_count})"
        )

    # Length sanity
    if len(cy) < 300:
        return False, f"Cypress too short ({len(cy)} chars)"
    if len(pw) < 300:
        return False, f"Playwright too short ({len(pw)} chars)"

    return True, "ok"

# ── Generator ─────────────────────────────────────────────────────────────────
def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return []

def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)

def generate_pair(client, story: str, category: str, idx: int, complexity: str) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                timeout=TIMEOUT,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        story=story,
                        category=category,
                        complexity=complexity
                    )
                }]
            )
            raw = response.content[0].text.strip()
            scripts = extract_scripts(raw)

            # Quality gate
            valid, reason = validate_scripts(scripts, category)
            if not valid:
                raise ValueError(f"Quality check failed: {reason}")

            return {
                "id": f"TC_{idx:03d}",
                "category": category,
                "complexity": complexity,
                "user_story": story,
                "cypress_script": scripts["cypress_script"],
                "playwright_script": scripts["playwright_script"],
                "ground_truth_label": "api_generated"
            }

        except Exception as e:
            logging.warning(f"TC_{idx:03d} attempt {attempt}/{MAX_RETRIES}: {type(e).__name__}: {str(e)[:200]}")
            if attempt < MAX_RETRIES:
                time.sleep(2)

    raise RuntimeError(f"TC_{idx:03d} failed after {MAX_RETRIES} attempts")


def main():
    if not API_KEY:
        print("❌ Set ANTHROPIC_API_KEY in .env or environment.")
        return

    client = anthropic.Anthropic(api_key=API_KEY)

    # Build full story list: re-generate all 237 existing + new categories
    existing_raw = []
    if EXISTING_FILE.exists():
        with open(EXISTING_FILE) as f:
            existing_raw = json.load(f)
        print(f"📂 Loaded {len(existing_raw)} existing stories to regenerate")

    existing_by_cat: dict = {}
    for item in existing_raw:
        existing_by_cat.setdefault(item["category"], []).append(item["user_story"])

    # Top-up buffer stories for weak categories (adds ~6 stories as buffer against skips)
    TOPUP_STORIES = {
        "responsive": [
            "As a user, I want the checkout form to stack vertically on mobile screens.",
            "As a user, I want the dashboard sidebar to become a bottom nav bar on mobile.",
            "As a user, I want card components to reflow from 3-column to single-column on small screens.",
            "As a user, I want the video player to resize proportionally on tablet viewports.",
        ],
        "accessibility": [
            "As a user, I want interactive toast notifications to be announced by screen readers.",
            "As a user, I want the date picker to be fully operable via keyboard alone.",
        ],
    }
    for cat, stories in TOPUP_STORIES.items():
        existing_by_cat.setdefault(cat, []).extend(stories)

    ALL_STORIES: dict = {**existing_by_cat, **NEW_USER_STORIES}
    total_stories = sum(len(v) for v in ALL_STORIES.values())

    # Resume from checkpoint
    dataset = load_checkpoint()
    completed_stories = {item["user_story"] for item in dataset}
    print(f"🔄 Total: {total_stories} stories | Done: {len(dataset)} | Remaining: {total_stories - len(dataset)}")
    print(f"🎯 Output → {OUTPUT_FILE}\n")

    skipped, idx = 0, 1

    with tqdm(total=total_stories, initial=len(dataset)) as pbar:
        for category, stories in ALL_STORIES.items():
            for story in stories:
                if story in completed_stories:
                    idx += 1; pbar.update(1); continue

                complexity = get_complexity(idx)
                try:
                    pair = generate_pair(client, story, category, idx, complexity)
                    dataset.append(pair)
                    completed_stories.add(story)
                    if len(dataset) % 10 == 0:
                        save_checkpoint(dataset)
                        tqdm.write(f"  💾 Checkpoint: {len(dataset)}/{total_stories} pairs")
                    time.sleep(DELAY)
                except Exception as e:
                    skipped += 1
                    tqdm.write(f"  ⚠️  Skipped TC_{idx:03d} [{category}]: {str(e)[:80]}")
                    logging.error(f"SKIPPED TC_{idx:03d} | {category} | {str(e)[:300]}")

                idx += 1; pbar.update(1)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    n = len(dataset)
    print(f"\n✅ Complete! {n} pairs → {OUTPUT_FILE}")
    print(f"   Generated: {n} | Skipped: {skipped} | Success rate: {round(n/total_stories*100,1)}%")

    by_cat: dict = {}
    for item in dataset:
        by_cat[item["category"]] = by_cat.get(item["category"], 0) + 1
    print(f"\n   Category breakdown:")
    for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"   • {cat:<25} {'█'*count}{'░'*max(0,20-count)} {count}")

    cy_click  = sum(1 for p in dataset if ".click("      in p["cypress_script"])
    cy_type   = sum(1 for p in dataset if ".type("       in p["cypress_script"])
    cy_should = sum(1 for p in dataset if ".should("     in p["cypress_script"])
    cy_inter  = sum(1 for p in dataset if "cy.intercept" in p["cypress_script"])
    pw_getby  = sum(1 for p in dataset if any(k in p["playwright_script"]
                    for k in ["getByRole(","getByLabel(","getByText(","getByPlaceholder("]))
    pw_fill   = sum(1 for p in dataset if "page.fill("   in p["playwright_script"])
    print(f"\n   Quality report:")
    print(f"   • cy .click()      : {cy_click}/{n} ({cy_click/n*100:.0f}%)")
    print(f"   • cy .type()       : {cy_type}/{n} ({cy_type/n*100:.0f}%)")
    print(f"   • cy .should()     : {cy_should}/{n} ({cy_should/n*100:.0f}%)")
    print(f"   • cy.intercept()   : {cy_inter}/{n} ({cy_inter/n*100:.0f}%)")
    print(f"   • pw getBy*        : {pw_getby}/{n} ({pw_getby/n*100:.0f}%)")
    print(f"   • pw page.fill()   : {pw_fill}/{n} ({pw_fill/n*100:.0f}%)")

    if ERROR_LOG.exists() and ERROR_LOG.stat().st_size > 0:
        print(f"\n   ⚠️  Errors → {ERROR_LOG}")


if __name__ == "__main__":
    main()
