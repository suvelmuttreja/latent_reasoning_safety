"""Validation and summaries for the matched Coconut stage trajectory."""

from __future__ import annotations

import statistics


def stage_k_values(stage: int, c_thought: int, max_stage: int) -> list[int]:
    if stage < 1 or stage > max_stage:
        raise ValueError(f"stage must be in [1, {max_stage}]")
    return [0, min(stage, max_stage) * c_thought]


def validate_trajectory_config(config: dict, evaluation: dict) -> None:
    capability = config["capability"]
    coherence = config["coherence"]
    if capability["max_new_tokens"] != evaluation["coconut_generation"][
        "answer_max_new_tokens"
    ]:
        raise ValueError("capability cap differs from frozen Coconut answer cap")
    if coherence["max_new_tokens"] != evaluation["explicit_generation"][
        "frozen_max_new_tokens"
    ]:
        raise ValueError("coherence cap differs from frozen explicit-thinking cap")
    if capability["stop_tokens"] != evaluation["coconut_generation"]["stop_tokens"]:
        raise ValueError("capability stop tokens differ from frozen Coconut config")
    if coherence["stop_tokens"] != evaluation["explicit_generation"]["stop_tokens"]:
        raise ValueError("coherence stop tokens differ from frozen explicit config")
    for key in ("do_sample", "temperature", "top_p", "top_k"):
        if coherence[key] != evaluation["sampling"][key]:
            raise ValueError(f"coherence {key} differs from frozen sampler")


def summarize_generation_rows(rows: list[dict], max_new_tokens: int) -> dict:
    if not rows:
        raise ValueError("cannot summarize empty generations")

    def one(group: list[dict]) -> dict:
        return {
            "outputs": len(group),
            "cap_hits": sum(
                row["stop_reason"] == "length"
                or row["generated_tokens"] >= max_new_tokens
                for row in group
            ),
            "eos_stops": sum(row["stop_reason"] == "eos_token" for row in group),
            "missing_closing_think": sum(
                "</think>" not in row["raw_output"] for row in group
            ),
            "mean_generated_tokens": statistics.mean(
                row["generated_tokens"] for row in group
            ),
            "mean_seconds": statistics.mean(row["seconds"] for row in group),
        }

    return {
        "overall": one(rows),
        "by_k": {
            str(k): one([row for row in rows if int(row["k"]) == k])
            for k in sorted({int(row["k"]) for row in rows})
        },
    }
