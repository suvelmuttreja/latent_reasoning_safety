"""Post-hoc descriptive checks on committed artifacts (2026-09-05): Coconut output length by stage/K, K0-vs-K6 agreement,
judge score for Chinese- vs English-thinking Coconut rows, and judge-score vs answer-length correlation at n=60."""
import json, re, glob, statistics as st, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]; R = ROOT / "artifacts/discovery/results"
out = {}
def pearson(x, y):
    mx, my = st.mean(x), st.mean(y); sx, sy = st.pstdev(x), st.pstdev(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (len(x) * sx * sy)
# 1. token counts per stage / K
files = {"stage2": R / "fallback_4b_skip0/trajectory_stage2/gsm8k.jsonl", "stage3": R / "fallback_4b_skip0/trajectory_stage3_gsm8k_cap1024/gsm8k.jsonl"}
out["gsm8k_generated_tokens"] = {}
for stg, f in files.items():
    rows = [json.loads(l) for l in open(f)]; by = {}
    for r in rows: by.setdefault(str(r["k"]), []).append(r)
    for k, rs in by.items():
        term = [int(r["generated_tokens"]) for r in rs if r["stop_reason"] != "length"]
        out["gsm8k_generated_tokens"][f"{stg}_K{k}"] = {"n_terminating": len(term), "median": st.median(term), "p90": sorted(term)[int(0.9 * len(term))]}
# 2. K0 vs K6 agreement at stage 3
rows = [json.loads(l) for l in open(files["stage3"])]
by = {k: {r["prompt_id"]: r for r in rows if str(r["k"]) == k} for k in ("0", "6")}; ids = sorted(by["6"])
ok = lambda r: r["correct"] in (True, "True")
term0 = [i for i in ids if by["0"][i]["stop_reason"] != "length"]
out["stage3_k0_vs_k6"] = {"same_predicted_answer": sum(by["0"][i]["predicted_answer"] == by["6"][i]["predicted_answer"] for i in ids),
    "k0_terminating": len(term0), "both_right": sum(ok(by["0"][i]) and ok(by["6"][i]) for i in term0),
    "k0_only_right": sum(ok(by["0"][i]) and not ok(by["6"][i]) for i in term0), "k6_only_right": sum(ok(by["6"][i]) and not ok(by["0"][i]) for i in term0)}
# 3. safety: Chinese rows and length correlation
cjk = re.compile(r"[一-鿿]"); frac = lambda s: len(cjk.findall(s or "")) / max(1, len(s or ""))
def load(c):
    g = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(R / "official_safety" / c / "generations.jsonl")}
    d = json.load(open(R / "official_safety/scores" / f"{c}.json")); recs = [v for v in d.values() if isinstance(v, list)][0]
    return g, {r["prompt_id"]: r["score"] for r in recs}
out["length_vs_score_n60"] = {}
for c in ["m0", "cot_u3", "coco_u3_k6"]:
    g, s = load(c); ids = sorted(g)
    out["length_vs_score_n60"][c] = {"pearson_r_answer_tokens": round(pearson([int(g[i]["answer_tokens"]) for i in ids], [s[i] for i in ids]), 3), "mean_answer_tokens": round(st.mean(int(g[i]["answer_tokens"]) for i in ids))}
g, s = load("coco_u3_k6"); gc, sc = load("cot_u3"); ids = sorted(g)
zh = [i for i in ids if frac(g[i]["parsed_thinking"]) > 0.3]; en = [i for i in ids if i not in zh]
out["coconut_chinese_rows"] = {"chinese_thinking_n": len(zh), "chinese_thinking_mean_score": round(st.mean(s[i] for i in zh), 4),
    "english_thinking_n": len(en), "english_thinking_mean_score": round(st.mean(s[i] for i in en), 4),
    "paired_coconut_minus_cot_chinese_rows": round(st.mean(s[i] - sc[i] for i in zh), 4), "paired_coconut_minus_cot_english_rows": round(st.mean(s[i] - sc[i] for i in en), 4)}
out["coconut_paper_table1_gpt2_gsm8k_verified_2026-09-05"] = {"source": "arxiv 2412.06769 HTML, Table 1", "CoT": 42.9, "No-CoT": 16.5, "Coconut": 34.1, "Pause as thought": 24.1, "Coconut w/o thought": 21.6, "Pause token": 16.4}
(ROOT / "writeup/figures/posthoc_output_anatomy.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
