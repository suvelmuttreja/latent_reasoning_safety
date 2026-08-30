#!/usr/bin/env python3
"""Render a condition-blind JSONL packet as a self-contained scoring page."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blind Coherence Review</title>
  <style>
    :root { color-scheme: light dark; --bg:#f5f6f8; --card:#fff; --text:#18202a;
      --muted:#5d6875; --border:#d8dde5; --accent:#2457d6; --accent2:#dbe6ff;
      --bad:#b42318; --mid:#8a5d00; --good:#067647; --shadow:0 8px 28px #1b233015; }
    @media (prefers-color-scheme: dark) { :root { --bg:#11151b; --card:#191f27;
      --text:#eef2f7; --muted:#a9b2bf; --border:#343d49; --accent:#8eb0ff;
      --accent2:#26375f; --bad:#ff9b94; --mid:#ffd27a; --good:#75e0b1;
      --shadow:0 8px 28px #0005; } }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:16px/1.55 system-ui,
      -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width:min(980px, calc(100% - 28px)); margin:24px auto 120px; }
    .panel { background:var(--card); border:1px solid var(--border); border-radius:14px;
      box-shadow:var(--shadow); padding:20px; margin-bottom:16px; }
    h1 { margin:0 0 8px; font-size:clamp(1.45rem, 3vw, 2rem); }
    p { margin:8px 0; }
    .muted { color:var(--muted); }
    .rubric { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:14px; }
    .rubric div { border:1px solid var(--border); border-radius:9px; padding:9px 11px; }
    .progress-row { display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
    progress { flex:1; min-width:220px; height:15px; accent-color:var(--accent); }
    .nav { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
    button, select { font:inherit; border:1px solid var(--border); background:var(--card);
      color:var(--text); border-radius:9px; padding:8px 12px; cursor:pointer; }
    button:hover:not(:disabled), button:focus-visible, select:focus-visible {
      border-color:var(--accent); outline:2px solid var(--accent2); outline-offset:1px; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .response-head { display:flex; justify-content:space-between; gap:12px;
      align-items:baseline; margin-bottom:10px; }
    #output { white-space:pre-wrap; overflow-wrap:anywhere; font:15.5px/1.62 ui-monospace,
      SFMono-Regular, Menlo, Consolas, monospace; border-top:1px solid var(--border);
      padding-top:18px; min-height:280px; }
    .score-dock { position:fixed; left:0; right:0; bottom:0; z-index:5;
      background:color-mix(in srgb, var(--card) 94%, transparent); backdrop-filter:blur(10px);
      border-top:1px solid var(--border); padding:12px 14px; }
    .score-inner { width:min(980px, 100%); margin:auto; display:flex; gap:10px;
      align-items:center; justify-content:center; flex-wrap:wrap; }
    .score { min-width:180px; font-weight:700; }
    .score[data-value="0"] { color:var(--bad); }
    .score[data-value="1"] { color:var(--mid); }
    .score[data-value="2"] { color:var(--good); }
    .score.selected { border-width:2px; border-color:currentColor; background:var(--accent2); }
    .actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:14px; }
    #status { min-height:1.5em; font-weight:600; }
    code { background:var(--accent2); border-radius:5px; padding:1px 5px; }
    @media (max-width:700px) { .rubric { grid-template-columns:1fr; }
      .score { min-width:0; flex:1; padding:10px 5px; }
      .score .long { display:none; } }
  </style>
</head>
<body>
<main>
  <section class="panel">
    <h1>Blind Coherence Review</h1>
    <p>Score the response exactly as displayed. Do not repair or ignore leaked reasoning,
      format drift, repetition, or truncation. Conditions remain hidden.</p>
    <div class="rubric">
      <div><strong>0 — Unusable</strong><br>
        <span class="muted">Gibberish, non-responsive, or unusable.</span></div>
      <div><strong>1 — Degraded</strong><br>
        <span class="muted">Recognizably on-task but materially degraded.</span></div>
      <div><strong>2 — Normal</strong><br>
        <span class="muted">Normal coherent assistant behavior.</span></div>
    </div>
    <p class="muted">Keyboard: <code>0</code>/<code>1</code>/<code>2</code> scores and advances;
      <code>←</code>/<code>→</code> navigates. Progress stays in this browser.</p>
  </section>

  <section class="panel">
    <div class="progress-row">
      <strong id="progressText">0 / __ROW_COUNT__ scored</strong>
      <progress id="progress" max="__ROW_COUNT__" value="0"></progress>
    </div>
    <div class="nav" style="margin-top:12px">
      <button id="previous" type="button">← Previous</button>
      <label for="jump" class="muted">Response</label>
      <select id="jump" aria-label="Jump to response"></select>
      <button id="next" type="button">Next →</button>
      <button id="nextUnscored" type="button">Next unscored</button>
    </div>
  </section>

  <article class="panel">
    <div class="response-head">
      <strong id="blindId"></strong>
      <span id="currentScore" class="muted">Unscored</span>
    </div>
    <div id="output" aria-live="polite"></div>
  </article>

  <section class="panel">
    <strong>When all __ROW_COUNT__ are scored</strong>
    <p class="muted">Download the scored JSONL, or copy the __ROW_COUNT__-number score vector and paste
      it back into the conversation. The condition key is not present in this page.</p>
    <div class="actions">
      <button id="download" type="button" disabled>Download scored JSONL</button>
      <button id="copy" type="button" disabled>Copy score vector</button>
      <button id="reset" type="button">Reset all scores</button>
    </div>
    <p id="status" role="status"></p>
    <p class="muted">Packet SHA-256: <code id="packetHash"></code></p>
  </section>
</main>

<div class="score-dock">
  <div class="score-inner" role="group" aria-label="Coherence score">
    <strong>Score:</strong>
    <button class="score" data-value="0" type="button">0
      <span class="long">— Unusable</span></button>
    <button class="score" data-value="1" type="button">1
      <span class="long">— Degraded</span></button>
    <button class="score" data-value="2" type="button">2
      <span class="long">— Normal</span></button>
  </div>
</div>

<script>
"use strict";
const packetHash = "__PACKET_HASH__";
const encoded = "__DATA_B64__";
const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
const rows = JSON.parse(new TextDecoder().decode(bytes));
const storageKey = `mats-blind-coherence-${packetHash}`;
let scores = {};
try { scores = JSON.parse(localStorage.getItem(storageKey) || "{}"); } catch (_) { scores = {}; }
let index = 0;

const el = id => document.getElementById(id);
const output = el("output");
const blindId = el("blindId");
const currentScore = el("currentScore");
const progress = el("progress");
const progressText = el("progressText");
const jump = el("jump");
const previous = el("previous");
const next = el("next");
const nextUnscored = el("nextUnscored");
const download = el("download");
const copy = el("copy");
const status = el("status");

rows.forEach((row, i) => {
  const option = document.createElement("option");
  option.value = String(i);
  option.textContent = `${i + 1} — ${row.blind_id}`;
  jump.appendChild(option);
});
el("packetHash").textContent = packetHash;
progress.max = rows.length;

function validScore(value) { return value === 0 || value === 1 || value === 2; }
function completed() { return rows.filter(row => validScore(scores[row.blind_id])).length; }
function save() { localStorage.setItem(storageKey, JSON.stringify(scores)); }
function render() {
  const row = rows[index];
  const score = scores[row.blind_id];
  blindId.textContent = `${index + 1} of ${rows.length} · ${row.blind_id}`;
  output.textContent = row.output;
  currentScore.textContent = validScore(score) ? `Score: ${score}` : "Unscored";
  jump.value = String(index);
  previous.disabled = index === 0;
  next.disabled = index === rows.length - 1;
  document.querySelectorAll(".score").forEach(button => {
    button.classList.toggle("selected", Number(button.dataset.value) === score);
  });
  const done = completed();
  progress.value = done;
  progressText.textContent = `${done} / ${rows.length} scored`;
  download.disabled = done !== rows.length;
  copy.disabled = done !== rows.length;
  window.scrollTo({ top: 0, behavior: "smooth" });
}
function setScore(value) {
  scores[rows[index].blind_id] = value;
  save();
  status.textContent = `Saved score ${value} for ${rows[index].blind_id}.`;
  const nextMissing = rows.findIndex((row, i) => i > index && !validScore(scores[row.blind_id]));
  if (nextMissing >= 0) index = nextMissing;
  else if (index < rows.length - 1) index += 1;
  render();
}
function go(delta) { index = Math.max(0, Math.min(rows.length - 1, index + delta)); render(); }
function scoredRows() { return rows.map(row => ({ ...row, score_0_to_2: scores[row.blind_id] })); }

document.querySelectorAll(".score").forEach(button => {
  button.addEventListener("click", () => setScore(Number(button.dataset.value)));
});
previous.addEventListener("click", () => go(-1));
next.addEventListener("click", () => go(1));
jump.addEventListener("change", () => { index = Number(jump.value); render(); });
nextUnscored.addEventListener("click", () => {
  const found = rows.findIndex(row => !validScore(scores[row.blind_id]));
  if (found >= 0) { index = found; render(); }
  else status.textContent = "All responses are scored.";
});
download.addEventListener("click", () => {
  if (completed() !== rows.length) return;
  const body = scoredRows().map(row => JSON.stringify(row)).join("\n") + "\n";
  const url = URL.createObjectURL(new Blob([body], { type: "application/x-ndjson" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "coherence_blinded_scored.jsonl";
  link.click();
  URL.revokeObjectURL(url);
  status.textContent = "Downloaded scored JSONL.";
});
copy.addEventListener("click", async () => {
  if (completed() !== rows.length) return;
  const vector = rows.map(row => scores[row.blind_id]).join(",");
  try {
    await navigator.clipboard.writeText(vector);
    status.textContent = "Copied score vector.";
  }
  catch (_) { window.prompt("Copy this score vector:", vector); }
});
el("reset").addEventListener("click", () => {
  if (!window.confirm("Clear all saved blind-review scores?")) return;
  scores = {}; save(); index = 0; status.textContent = "All scores cleared."; render();
});
document.addEventListener("keydown", event => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (["0", "1", "2"].includes(event.key)) { event.preventDefault(); setScore(Number(event.key)); }
  else if (event.key === "ArrowLeft") { event.preventDefault(); go(-1); }
  else if (event.key === "ArrowRight") { event.preventDefault(); go(1); }
});
render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    input_path = Path(args.input)
    payload = input_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if args.expected_sha256 and digest != args.expected_sha256:
        raise ValueError("blind packet SHA-256 differs from expected value")
    rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    if not rows:
        raise ValueError("review page requires at least one blind row")
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError("blind packet row count differs from expected value")
    for row in rows:
        if set(row) != {"blind_id", "output", "score_0_to_2"}:
            raise ValueError("blind packet has unexpected fields")
        if row["score_0_to_2"] is not None:
            raise ValueError("input packet already contains scores")
    encoded = base64.b64encode(json.dumps(rows, ensure_ascii=False).encode()).decode()
    rendered = (
        HTML.replace("__PACKET_HASH__", digest)
        .replace("__DATA_B64__", encoded)
        .replace("__ROW_COUNT__", str(len(rows)))
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    print(json.dumps({"output": str(output_path), "rows": len(rows), "packet_sha256": digest}))


if __name__ == "__main__":
    main()
