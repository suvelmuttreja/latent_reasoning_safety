"""Post-hoc descriptive count of CJK-heavy endpoint safety fields.

Rule: >30% of all characters are in the Unicode CJK Unified Ideographs block.
This mechanical heuristic is not a language identifier. All 60 prompts are
English. Reads committed artifacts only.
"""
import json, re, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "artifacts/discovery/results/official_safety"
cjk = re.compile(r"[一-鿿]")
def frac(s): s = s or ""; return len(cjk.findall(s)) / max(1, len(s))
out = {"rule": "share of CJK Unified Ideograph characters > 0.30", "conditions": {}}
for c in ["m0", "cot_u3", "coco_u3_k6"]:
    rows = [json.loads(l) for l in open(R / c / "generations.jsonl")]
    th = [r["prompt_id"] for r in rows if frac(r["parsed_thinking"]) > 0.3]
    an = [r["prompt_id"] for r in rows if frac(r["parsed_final_answer"]) > 0.3]
    out["conditions"][c] = {"rows": len(rows), "thinking_above_threshold": len(th), "final_answer_above_threshold": len(an),
                            "thinking_ids": th, "final_answer_ids": an}
    print(c, len(th), len(an))
(ROOT / "writeup/figures/language_switch_counts.json").write_text(json.dumps(out, indent=1))
