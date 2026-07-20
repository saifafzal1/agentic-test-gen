// DOM crawler for the grounded-corpus pipeline.
//
// For each app/page in apps.json, loads the live page with Playwright
// (running scripted auth steps first where required) and extracts the
// selectors that actually exist: ids, data-test/data-testid attributes,
// input names/types/placeholders, buttons, links, selects with options,
// and headings. Output: grounded_corpus/profiles/<app>.json
//
// Run from execution_harness/ (where playwright is installed):
//   node ../grounded_corpus/crawl_dom.mjs
import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { createRequire } from "module";

const HERE = dirname(fileURLToPath(import.meta.url));
// Playwright is installed in execution_harness/, not here.
const requireFromHarness = createRequire(join(HERE, "../execution_harness/package.json"));
const { chromium } = requireFromHarness("@playwright/test");
const CONFIG = JSON.parse(readFileSync(join(HERE, "apps.json"), "utf8"));
const OUT_DIR = join(HERE, "profiles");
mkdirSync(OUT_DIR, { recursive: true });

const EXTRACT = () => {
  const cap = (arr, n) => arr.slice(0, n);
  const txt = (el) => (el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 60);
  const attrs = (el) => {
    const out = { tag: el.tagName.toLowerCase() };
    for (const a of ["id", "data-test", "data-testid", "name", "type", "placeholder", "aria-label", "href", "value", "class"]) {
      const v = el.getAttribute && el.getAttribute(a);
      if (v) out[a] = v.slice(0, 80);
    }
    const t = txt(el);
    if (t) out.text = t;
    return out;
  };
  return {
    title: document.title,
    headings: cap([...document.querySelectorAll("h1,h2,h3")].map(txt).filter(Boolean), 12),
    inputs: cap([...document.querySelectorAll("input,textarea")].map(attrs), 25),
    buttons: cap([...document.querySelectorAll("button,input[type=submit],a.button,[role=button]")].map(attrs), 25),
    links: cap([...document.querySelectorAll("a")].map(attrs).filter((a) => a.text), 40),
    selects: cap([...document.querySelectorAll("select")].map((s) => ({
      ...attrs(s),
      options: [...s.options].map((o) => ({ value: o.value, text: (o.textContent || "").trim().slice(0, 40) })).slice(0, 12),
    })), 8),
    data_test_elements: cap([...document.querySelectorAll("[data-test],[data-testid]")].map(attrs), 40),
    ids: cap([...document.querySelectorAll("[id]")].map(attrs), 40),
  };
};

const browser = await chromium.launch();
for (const app of CONFIG.apps) {
  const profile = { key: app.key, base_url: app.base_url, auth: app.auth, pages: [] };
  const context = await browser.newContext();
  const page = await context.newPage();

  if (app.auth) {
    await page.goto(app.base_url + app.auth.path, { waitUntil: "domcontentloaded", timeout: 45000 });
    for (const [op, sel, val] of app.auth.steps) {
      if (op === "fill") await page.fill(sel, val);
      if (op === "click") await page.click(sel);
    }
    await page.waitForLoadState("domcontentloaded");
  }

  for (const p of app.pages) {
    try {
      await page.goto(app.base_url + p.path, { waitUntil: "domcontentloaded", timeout: 45000 });
      await page.waitForTimeout(1500); // let SPAs settle
      const dom = await page.evaluate(EXTRACT);
      profile.pages.push({ ...p, url: app.base_url + p.path, dom });
      console.log(`[${app.key}] ${p.path} ok (${dom.ids.length} ids, ${dom.data_test_elements.length} data-test)`);
    } catch (e) {
      profile.pages.push({ ...p, url: app.base_url + p.path, error: String(e).slice(0, 200) });
      console.log(`[${app.key}] ${p.path} FAILED: ${String(e).slice(0, 120)}`);
    }
  }
  await context.close();
  writeFileSync(join(OUT_DIR, `${app.key}.json`), JSON.stringify(profile, null, 2));
}
await browser.close();
console.log("profiles written to", OUT_DIR);
