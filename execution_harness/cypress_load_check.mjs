// Emulates Cypress's spec-collection phase: loads each spec file with
// mocha-style globals stubbed the way collection behaves (describe/context
// bodies EXECUTE; it/hook callbacks only REGISTER), plus chainable cy/Cypress
// proxies for any top-level references. A spec "loads" if the file evaluates
// without throwing. Usage: node cypress_load_check.mjs <file...>
import { pathToFileURL } from "url";

const chainable = () => {
  const p = new Proxy(function () {}, {
    get: (_, prop) => (prop === Symbol.toPrimitive ? () => "" : p),
    apply: () => p,
  });
  return p;
};

const suite = (_title, fn) => { if (typeof fn === "function") fn(); };
suite.only = suite;
suite.skip = suite;
const testcase = (_title, _fn) => {};
testcase.only = testcase;
testcase.skip = testcase;
const hook = (_fn) => {};

globalThis.describe = suite;
globalThis.context = suite;
globalThis.it = testcase;
globalThis.specify = testcase;
globalThis.before = hook;
globalThis.beforeEach = hook;
globalThis.after = hook;
globalThis.afterEach = hook;
globalThis.cy = chainable();
globalThis.Cypress = chainable();
globalThis.expect = chainable();
globalThis.assert = chainable();
globalThis.chai = chainable();

// Full-parse first (mirrors the eager AST parse Cypress's bundler performs;
// V8's dynamic import alone lazy-parses function bodies and misses errors
// like duplicate lexical declarations inside test callbacks).
import { readFileSync } from "fs";
import { parse } from "@babel/parser";

for (const f of process.argv.slice(2)) {
  try {
    parse(readFileSync(f, "utf8"), {
      sourceType: "unambiguous",
      errorRecovery: false,
    });
    await import(pathToFileURL(f).href);
    console.log(JSON.stringify({ file: f, ok: true }));
  } catch (e) {
    console.log(JSON.stringify({ file: f, ok: false, error: String(e && e.message ? e.message : e).slice(0, 250) }));
  }
}
