"""
DOM-context builder for inference-time application grounding (Task 2).

Given a grounded story, returns a compact inventory of the REAL selectors that
exist on the target page(s), formatted for injection into the generation prompt
(grounding "ON"). Source of truth: the crawled grounded_corpus/profiles/*.json.

Matching strategy:
  1. Extract any explicit page URL(s) from the story text.
  2. Match each URL to a profiled page (exact, else path-suffix).
  3. If no page URL is found, fall back to the app landing page(s) matched by
     the story's base_url.

Usage:
    from grounding_experiment.dom_context import build_dom_context
    block = build_dom_context(story_dict)   # -> str ("" if nothing found)
"""
import json
import re
from pathlib import Path
from functools import lru_cache

HERE = Path(__file__).parent
PROFILES_DIR = HERE.parent / "grounded_corpus" / "profiles"

URL_RE = re.compile(r"https?://[^\s\"'()]+")


@lru_cache(maxsize=1)
def _load_pages():
    """Return list of (base_url, url, path, dom) across every profiled page."""
    pages = []
    for pf in sorted(PROFILES_DIR.glob("*.json")):
        prof = json.loads(pf.read_text())
        base = prof.get("base_url", "").rstrip("/")
        for pg in prof.get("pages", []):
            if pg.get("error") or not pg.get("dom"):
                continue
            pages.append((base, (pg.get("url") or "").rstrip("/"),
                          pg.get("path", ""), pg["dom"]))
    return pages


def _match_pages(story: dict):
    """Return the DOM dicts of pages relevant to this story.

    Base-anchored so a bare "/login" path never matches a different app.
    A URL that is just an app's base (e.g. https://www.saucedemo.com) is
    treated as "the whole app" — include every profiled page of that app,
    since such stories act across login -> inventory -> cart after signing in.
    A URL with a specific path matches only that page (within the same app).
    """
    pages = _load_pages()
    text = story.get("user_story", "")
    base = (story.get("base_url") or "").rstrip("/")
    urls = [u.rstrip("/.,") for u in URL_RE.findall(text)]

    matched, seen = [], set()

    def add(dom, key):
        if key not in seen:
            seen.add(key); matched.append(dom)

    for u in urls:
        # the app whose base_url is a prefix of this URL (base-anchored)
        app_base = next((b for b, purl, path, dom in pages if b and u.startswith(b)), None)
        if app_base is None:
            continue
        remainder = u[len(app_base):]
        if remainder in ("", "/"):                      # bare base -> whole app
            for b, purl, path, dom in pages:
                if b == app_base:
                    add(dom, purl)
        else:                                            # specific page
            for b, purl, path, dom in pages:
                if b == app_base and (u == purl or (path not in ("", "/") and u.endswith(path))):
                    add(dom, purl)

    # fall back to same-app pages if nothing matched
    if not matched and base:
        for b, purl, path, dom in pages:
            if b == base:
                add(dom, purl)

    return matched


def _collect(dom_list, key, limit):
    """Merge + de-dup a field across matched pages, capped at `limit`."""
    out, seen = [], set()
    for dom in dom_list:
        for item in dom.get(key, []):
            k = json.dumps(item, sort_keys=True) if isinstance(item, dict) else item
            if k not in seen:
                seen.add(k); out.append(item)
            if len(out) >= limit:
                return out
    return out


def build_dom_context(story: dict, max_chars: int = 2500) -> str:
    """Build the injectable 'real selectors' block for a story ('' if none)."""
    dom_list = _match_pages(story)
    if not dom_list:
        return ""

    lines = ["Available REAL selectors on the target page(s) — use ONLY these; "
             "do NOT invent data-testid values, routes, or custom commands:"]

    # data-test / data-testid attributes (the highest-signal selectors)
    dts = []
    for el in _collect(dom_list, "data_test_elements", 40):
        v = el.get("data-test") or el.get("data-testid")
        if v and v not in dts:
            dts.append(v)
    if dts:
        lines.append("  data-test attributes: " + ", ".join(f'"{d}"' for d in dts[:30]))

    # element ids
    ids = []
    for el in _collect(dom_list, "ids", 40):
        i = el.get("id") if isinstance(el, dict) else None
        if i and i not in ids:
            ids.append(i)
    if ids:
        lines.append("  ids: " + ", ".join(f"#{i}" for i in ids[:30]))

    # inputs (name / type / placeholder)
    inp = []
    for el in _collect(dom_list, "inputs", 25):
        desc = []
        for a in ("name", "type", "placeholder"):
            if el.get(a):
                desc.append(f'{a}="{el[a]}"')
        if desc:
            inp.append("[" + " ".join(desc) + "]")
    if inp:
        lines.append("  inputs: " + "; ".join(inp[:15]))

    # buttons (text)
    btns = [el.get("text") for el in _collect(dom_list, "buttons", 25)
            if isinstance(el, dict) and el.get("text")]
    if btns:
        lines.append("  buttons: " + ", ".join(f'"{b}"' for b in btns[:15]))

    # links (text)
    links = [el.get("text") for el in _collect(dom_list, "links", 40)
             if isinstance(el, dict) and el.get("text")]
    if links:
        lines.append("  links: " + ", ".join(f'"{l}"' for l in links[:15]))

    # selects + options
    for sel in _collect(dom_list, "selects", 8):
        if isinstance(sel, dict) and sel.get("options"):
            opts = ", ".join(f'{o.get("value")}={o.get("text")}' for o in sel["options"][:8])
            ident = sel.get("data-test") or sel.get("id") or sel.get("name") or "select"
            lines.append(f'  select [{ident}] options: {opts}')

    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[:max_chars].rsplit("\n", 1)[0]
    return block


if __name__ == "__main__":
    import sys
    stories = [json.loads(l) for l in
               open(HERE.parent / "data/execution_validation/grounded_stories.jsonl")]
    ids = sys.argv[1:] or ["TC_G01", "TC_G13", "TC_G08"]
    for s in stories:
        if s["id"] in ids:
            print(f"\n===== {s['id']} ({s['base_url']}) =====")
            ctx = build_dom_context(s)
            print(ctx if ctx else "(no DOM context matched)")
