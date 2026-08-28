"""Pure validation and summary helpers for coherence harness audits."""

from __future__ import annotations

import statistics


def summarize_coherence_rows(rows: list[dict], max_new_tokens: int) -> dict:
    def one(group: list[dict]) -> dict:
        return {
            "outputs": len(group),
            "cap_hits": sum(
                row["stop_reason"] == "length"
                or row["generated_tokens"] >= max_new_tokens
                for row in group
            ),
            "missing_closing_think": sum(not row["has_closing_think"] for row in group),
            "cap_and_missing_closing_think": sum(
                (
                    row["stop_reason"] == "length"
                    or row["generated_tokens"] >= max_new_tokens
                )
                and not row["has_closing_think"]
                for row in group
            ),
            "eos_stops": sum(row["stop_reason"] == "eos_token" for row in group),
            "mean_generated_tokens": statistics.mean(
                row["generated_tokens"] for row in group
            ),
        }

    replicates = sorted({int(row["replicate"]) for row in rows})
    return {
        "overall": one(rows),
        "by_replicate": {
            str(replicate): one(
                [row for row in rows if int(row["replicate"]) == replicate]
            )
            for replicate in replicates
        },
    }


def validate_control_against_frozen_evaluation(config: dict, evaluation: dict) -> None:
    sampling = config["sampling"]
    frozen = evaluation["sampling"]
    for key in ("do_sample", "temperature", "top_p", "top_k"):
        if sampling[key] != frozen[key]:
            raise ValueError(f"control {key} differs from frozen evaluation config")
    if sampling["stop_tokens"] != evaluation["explicit_generation"]["stop_tokens"]:
        raise ValueError("control stop tokens differ from frozen explicit generation config")
    if sampling["max_new_tokens"] != evaluation["coconut_generation"]["answer_max_new_tokens"]:
        raise ValueError("control does not reproduce the shared 512-token gate cap")
