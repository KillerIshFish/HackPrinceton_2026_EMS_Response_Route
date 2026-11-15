#!/usr/bin/env node
// Wording check (docs/BUILD_PLAN.md, ground rule 1 and step D4).
//
// The console advises a dispatcher. It must never call a place or a person safe, all clear,
// empty, unsurvivable or already lost, never say "no one there", and never show a missing
// hazard record as "no known hazard". This script scans what the console can display: string
// literals, template-literal text, JSX text and JSX attribute strings in web/src. Identifiers,
// type names, literal types, comments and import paths are not scanned, so a key such as
// `cleared_room_ids` or a type `"empty"` used only as a tag never triggers it.
//
// A legitimate use (for example the word in a sentence that tells the dispatcher what the
// console will NOT say) is allowed with a comment on the same line or the line above:
//     // wording-ok: <reason>
//     {/* wording-ok: <reason> */}
// The reason is required.
//
// Usage: node scripts/check-wording.mjs [dir ...]   (default: src). Exits 1 on any hit.

import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "@babel/parser";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const EXTENSIONS = new Set([".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"]);

/** What the console must not say about a place or a person. Matched whole-word, any case. */
export const FORBIDDEN = [
  { phrase: "safe", re: /\bsafe\b/i },
  { phrase: "all clear", re: /\ball[\s-]+clear\b/i },
  { phrase: "empty", re: /\bempty\b/i },
  { phrase: "no known hazard", re: /\bno\s+known\s+hazards?\b/i },
  { phrase: "unsurvivable", re: /\bun-?survivable\b/i },
  { phrase: "already lost", re: /\balready\s+lost\b/i },
  { phrase: "no one there", re: /\b(?:no[\s-]?one|nobody)\s+(?:is\s+)?there\b/i },
];

const ALLOW = /wording-ok:\s*\S/;

function* walk(dir) {
  for (const name of readdirSync(dir)) {
    if (name === "node_modules" || name.startsWith(".")) continue;
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) yield* walk(full);
    else if (EXTENSIONS.has(extname(name)) && !name.endsWith(".d.ts")) yield full;
  }
}

/** Parent-aware AST walk: calls visit(node, parent, key) for every node. */
function traverse(node, visit, parent = null, key = null) {
  if (!node || typeof node.type !== "string") return;
  if (visit(node, parent, key) === false) return;
  for (const k of Object.keys(node)) {
    if (k === "loc" || k === "start" || k === "end" || k === "extra" || k.endsWith("Comments")) continue;
    const v = node[k];
    if (Array.isArray(v)) for (const c of v) traverse(c, visit, node, k);
    else if (v && typeof v === "object" && typeof v.type === "string") traverse(v, visit, node, k);
  }
}

/** String nodes that are not display text: module paths, literal types, object keys, directives. */
function isNonDisplay(node, parent, key) {
  if (!parent) return false;
  if (key === "source" && /Import|Export/.test(parent.type)) return true; // import ... from "x"
  if (parent.type === "TSLiteralType" || parent.type === "TSEnumMember" || parent.type === "TSExternalModuleReference") return true;
  if (key === "key" && (parent.type === "ObjectProperty" || parent.type === "ClassProperty" || parent.type === "TSPropertySignature")) return true;
  if (parent.type === "TSImportType") return true;
  return false;
}

/** Lines (1-based) that carry an allow comment. A comment allows its own line and the next one. */
function allowedLines(comments) {
  const lines = new Set();
  for (const c of comments ?? []) {
    if (!ALLOW.test(c.value)) continue;
    for (let l = c.loc.start.line; l <= c.loc.end.line + 1; l++) lines.add(l);
  }
  return lines;
}

function check(file) {
  const code = readFileSync(file, "utf8");
  const ext = extname(file);
  let ast;
  try {
    ast = parse(code, {
      sourceType: "module",
      errorRecovery: false,
      plugins: ext === ".ts" ? ["typescript"] : ext === ".tsx" ? ["typescript", "jsx"] : ["jsx"],
    });
  } catch (e) {
    return { parseError: `${e.message}` };
  }
  const allow = allowedLines(ast.comments);
  const hits = [];
  const test = (text, loc) => {
    if (!text) return;
    for (const f of FORBIDDEN) {
      const m = f.re.exec(text);
      if (!m) continue;
      // the line the matched word sits on (multi-line JSX text / template text)
      const line = loc.start.line + (text.slice(0, m.index).match(/\n/g)?.length ?? 0);
      // an allow comment on the word's own line, or on the line where its (multi-line) text node starts
      if (allow.has(line) || allow.has(loc.start.line)) continue;
      hits.push({ line, phrase: f.phrase, text: text.replace(/\s+/g, " ").trim().slice(0, 90) });
    }
  };
  traverse(ast.program, (node, parent, key) => {
    switch (node.type) {
      case "StringLiteral":
        if (!isNonDisplay(node, parent, key)) test(node.value, node.loc);
        break;
      case "TemplateElement":
        if (parent?.type !== "TSLiteralType") test(node.value.cooked ?? node.value.raw, node.loc);
        break;
      case "JSXText":
        test(node.value, node.loc);
        break;
      case "TSLiteralType":
      case "TSTemplateLiteralType":
        return false; // type-level strings are tags, never displayed
      default:
        break;
    }
    return true;
  });
  return { hits };
}

const dirs = process.argv.slice(2);
const targets = (dirs.length ? dirs : ["src"]).map((d) => resolve(ROOT, d));
let total = 0;
let files = 0;
let failed = false;
for (const dir of targets) {
  for (const file of walk(dir)) {
    files += 1;
    const rel = relative(ROOT, file).replaceAll("\\", "/");
    const { hits, parseError } = check(file);
    if (parseError) {
      console.error(`${rel}: could not parse (${parseError})`);
      failed = true;
      continue;
    }
    for (const h of hits) {
      total += 1;
      console.error(`${rel}:${h.line}: "${h.phrase}" in: ${h.text}`);
    }
  }
}
if (total > 0 || failed) {
  console.error(
    `\nWording check failed: ${total} phrase${total === 1 ? "" : "s"} the console must not say about a place or a person.` +
      `\nRephrase, or mark a legitimate use with a "// wording-ok: <reason>" comment on that line or the line above.`,
  );
  process.exit(1);
}
console.log(`Wording check passed: ${files} files scanned.`);
