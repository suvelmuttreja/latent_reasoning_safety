#!/usr/bin/env python3
"""Recompute analysis checks without altering frozen generations or results."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import statistics as st
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r"[\u4e00-\u9fff]")


def truth(value):
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"Unexpected correctness value: {value!r}")


def unfinished(row):
    return row.get("stop_reason") == "length" or row.get("truncated") is True


def accuracy_summary(rows):
    if not rows:
        raise ValueError("No rows")
    complete = [r for r in rows if not unfinished(r)]
    missing = [r for r in rows if unfinished(r)]
    correct_complete = sum(truth(r["correct"]) for r in complete)
    correct_missing = sum(truth(r["correct"]) for r in missing)
    n = len(rows)
    return {
        "n": n,
        "terminated": len(complete),
        "unfinished": len(missing),
        "correct_terminated": correct_complete,
        "correct_at_cutoff_among_unfinished": correct_missing,
        "all_row_cutoff_parser_accuracy": (correct_complete + correct_missing) / n,
        "unknown_unfinished_outcome_bounds": [correct_complete / n,
                                               (correct_complete + len(missing)) / n],
        "terminated_and_correct_fraction": correct_complete / n,
        "complete_case_accuracy_descriptive": correct_complete / len(complete) if complete else None,
        "unfinished_correct_ids": [r["prompt_id"] for r in missing if truth(r["correct"])],
    }


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def bootstrap(values, samples=10000):
    rng = random.Random(42)
    n = len(values)
    reps = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    return [percentile(reps, 0.025), percentile(reps, 0.975)]


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    cross = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return cross / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "writeup/analysis_audit/recomputed_checks.json")
    args = parser.parse_args()
    hashes = {}

    def read(path):
        p = ROOT / path
        raw = p.read_bytes()
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return [json.loads(l) for l in raw.splitlines() if l.strip()] if p.suffix == ".jsonl" else json.loads(raw)

    def indexed(rows):
        result = {r["prompt_id"]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate prompt IDs")
        return result

    prefix = "artifacts/discovery/results/"
    paths = {
        "m0": "native_gsm8k_controls/m0/generations.jsonl",
        "cot_u3": "native_gsm8k_controls/cot_u3/generations.jsonl",
        "coco_u1": "fallback_4b_skip0/gate_stage1_v2/gate_result.json",
        "coco_u2": "fallback_4b_skip0/trajectory_stage2/gsm8k.jsonl",
        "coco_u3": "fallback_4b_skip0/trajectory_stage3_gsm8k_cap1024/gsm8k.jsonl",
    }
    capability, loaded = {}, {}
    for condition, path in paths.items():
        rows = read(prefix + path)
        if condition == "coco_u1":
            rows = [r for group in rows["capability"]["by_k"].values() for r in group["outputs"]]
        loaded[condition] = rows
        for k in sorted({int(r.get("k", 0)) for r in rows}):
            subset = [r for r in rows if int(r.get("k", 0)) == k]
            indexed(subset)
            summary = accuracy_summary(subset)
            complete_tokens = [r["generated_tokens"] for r in subset if not unfinished(r)]
            summary["terminating_generated_tokens_median"] = st.median(complete_tokens)
            summary["terminating_generated_tokens_p90_linear"] = percentile(complete_tokens, .9)
            changed = []
            for r in subset:
                try:
                    equivalent = Decimal(str(r["predicted_answer"])) == Decimal(str(r["reference_answer"]))
                    if equivalent != truth(r["correct"]):
                        changed.append({"prompt_id": r["prompt_id"], "stored_correct": truth(r["correct"]),
                                        "decimal_equal": equivalent, "unfinished": unfinished(r)})
                except InvalidOperation:
                    pass
            summary["decimal_normalization_sensitivity_disagreements"] = changed
            capability[f"{condition}_k{k}"] = summary

    k0 = indexed([r for r in loaded["coco_u3"] if int(r["k"]) == 0])
    k6 = indexed([r for r in loaded["coco_u3"] if int(r["k"]) == 6])
    assert set(k0) == set(k6)
    complete_ids = [p for p in k0 if not unfinished(k0[p]) and not unfinished(k6[p])]
    paired = {"both_terminated_n": len(complete_ids), "all_prompts_n": len(k0)}
    for name, left, right in (("both_correct", True, True), ("k0_only_correct", True, False),
                              ("k6_only_correct", False, True), ("both_wrong", False, False)):
        paired[name] = sum(truth(k0[p]["correct"]) == left and truth(k6[p]["correct"]) == right for p in complete_ids)
    paired["same_prediction_all_rows"] = sum(k0[p]["predicted_answer"] == k6[p]["predicted_answer"] for p in k0)
    paired["same_prediction_both_terminated"] = sum(k0[p]["predicted_answer"] == k6[p]["predicted_answer"] for p in complete_ids)
    paired["k6_correct_on_k0_unfinished_rows"] = sum(truth(k6[p]["correct"]) for p in k0 if unfinished(k0[p]))

    score_paths = {"m0": "official_safety/scores/m0.json", "cot_u1": "dense_safety/scores/cot_u1.json",
                   "cot_u2": "dense_safety/scores/cot_u2.json", "cot_u3": "official_safety/scores/cot_u3.json",
                   "coco_u3": "official_safety/scores/coco_u3_k6.json",
                   "cot_seed43_u1": "matched_4b_cot_seed43/scores/cot_seed43_u1.json"}
    scores, means = {}, {}
    for condition, path in score_paths.items():
        payload = read(prefix + path)
        rows = indexed(payload["records"])
        assert len(rows) == 60
        scores[condition] = {p: r["score"] for p, r in rows.items()}
        mean = st.mean(scores[condition].values())
        assert math.isclose(mean, payload["mean_score"], abs_tol=1e-12)
        means[condition] = mean
    ids = list(scores["m0"])
    assert all(set(s) == set(ids) for s in scores.values())
    effects = {}
    for left, right in (("coco_u3", "cot_u3"), ("cot_u1", "m0"), ("cot_u2", "m0"),
                        ("cot_u3", "m0"), ("cot_seed43_u1", "m0"), ("cot_seed43_u1", "cot_u1")):
        diffs = [scores[left][p] - scores[right][p] for p in ids]
        effects[f"{left}_minus_{right}"] = {"mean": st.mean(diffs), "paired_prompt_percentile_95_ci": bootstrap(diffs)}
    official = read(prefix + "official_safety/scores/paired_comparison.json")
    assert all(abs(a - b) < 1e-12 for a, b in zip(effects["coco_u3_minus_cot_u3"]["paired_prompt_percentile_95_ci"], official["paired_prompt_bootstrap_95_ci"]))
    bounds = {}
    for condition in ("coco_u1_k2", "coco_u2_k4"):
        payload = read(prefix + f"dense_safety/scores/{condition}_bounds.json")
        rows = indexed(payload["records"])
        missing = set(ids) - set(rows)
        assert missing == set(payload["nonterminating_prompt_ids"])
        low = sum(r["score"] for r in rows.values()) / len(ids)
        high = low + len(missing) / len(ids)
        assert abs(low - payload["lower_bound_all_nonterminations_zero"]) < 1e-12
        assert abs(high - payload["upper_bound_all_nonterminations_one"]) < 1e-12
        bounds[condition] = {"lower": low, "upper": high, "missing_ids": sorted(missing)}

    anatomy, generations = {}, {}
    for condition, score_key in (("m0", "m0"), ("cot_u3", "cot_u3"), ("coco_u3_k6", "coco_u3")):
        rows = indexed(read(prefix + f"official_safety/{condition}/generations.jsonl"))
        assert set(rows) == set(ids)
        generations[condition] = rows
        anatomy[condition] = {
            "n": len(rows), "unfinished": sum(unfinished(r) for r in rows.values()),
            "missing_closing_think": sum("</think>" not in r["raw_output"] for r in rows.values()),
            "empty_parsed_final": sum(not r["parsed_final_answer"].strip() for r in rows.values()),
            "pearson_answer_tokens_vs_score": pearson([rows[p]["answer_tokens"] for p in ids], [scores[score_key][p] for p in ids]),
            "mean_answer_tokens": st.mean(rows[p]["answer_tokens"] for p in ids),
        }
    strata = {}
    for field in ("parsed_thinking", "parsed_final_answer"):
        for above in (True, False):
            subset = [p for p in ids if (len(CJK.findall(generations["coco_u3_k6"][p][field] or "")) /
                      max(1, len(generations["coco_u3_k6"][p][field] or "")) > .3) == above]
            strata[f"{field}_cjk_above_30pct_{above}"] = {
                "n": len(subset), "ids": subset,
                "coconut_mean": st.mean(scores["coco_u3"][p] for p in subset),
                "paired_coconut_minus_cot_mean": st.mean(scores["coco_u3"][p] - scores["cot_u3"][p] for p in subset),
                "interpretation": "post-treatment selected descriptive subgroup, not a causal language effect",
            }

    weights = read(prefix + "posthoc_layerwise_weight_diff/layerwise_weight_updates.json")
    blocks = [r for r in weights["layers"] if r["group"].startswith("layer_")]
    base_sq = sum(r["m0_l2"] ** 2 for r in blocks)
    cot_sq = sum(r["cot_update_l2"] ** 2 for r in blocks)
    coco_sq = sum(r["coconut_update_l2"] ** 2 for r in blocks)
    cross = sum(r["update_cosine"] * r["cot_update_l2"] * r["coconut_update_l2"] for r in blocks)
    block_summary = {"blocks": len(blocks), "cot_relative_l2": math.sqrt(cot_sq/base_sq),
                     "coconut_relative_l2": math.sqrt(coco_sq/base_sq), "update_cosine": cross/math.sqrt(cot_sq*coco_sq),
                     "coconut_smaller_in_every_block": all(r["coconut_update_l2"] < r["cot_update_l2"] for r in blocks),
                     "scope": "transformer blocks only; excludes embeddings, head and final norm"}

    readout = read(prefix + "posthoc_token_mode_readout/token_mode_readout.json")
    periodicity = {}
    for task in sorted({r["task"] for r in readout["rows"]}):
        results = {}
        for top_n in (1, 3, 5, 10):
            for exclude_first in (False, True):
                same, opposite = [], []
                for row in readout["rows"]:
                    if row["task"] != task:
                        continue
                    sets = [{t["token_id"] for t in r["top_tokens"][:top_n]} for r in row["readouts"]]
                    for a, b in itertools.combinations(range(1 if exclude_first else 0, len(sets)), 2):
                        value = len(sets[a] & sets[b]) / len(sets[a] | sets[b])
                        (same if (a-b) % 2 == 0 else opposite).append(value)
                results[f"top{top_n}_exclude_first_{exclude_first}"] = {
                    "same_parity_mean": st.mean(same), "opposite_parity_mean": st.mean(opposite)}
        periodicity[task] = results

    result = {"status": "complete", "role": "posthoc_analysis_audit_not_replacement_primary",
              "capability": capability, "k0_k6_paired_outcomes": paired, "safety_means": means,
              "safety_effects": effects, "safety_missingness_bounds": bounds,
              "endpoint_output_anatomy": anatomy, "cjk_strata": strata,
              "transformer_blocks_weight_diff": block_summary, "readout_sensitivity": periodicity,
              "source_sha256": hashes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "source_files": len(hashes),
                      "corrected_k0": capability["coco_u3_k0"], "weight_blocks": block_summary}, indent=2))


if __name__ == "__main__":
    main()
