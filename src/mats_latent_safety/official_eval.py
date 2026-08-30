"""Fail-closed validation for cached official safety generations."""

from __future__ import annotations


def validate_official_safety_rows(
    rows: list[dict], manifest: list[dict], expected_condition: str
) -> list[dict]:
    if not expected_condition:
        raise ValueError("official condition must be non-empty")
    ids = [row.get("prompt_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("official generations contain duplicate prompt IDs")
    expected_ids = [row["id"] for row in manifest]
    if set(ids) != set(expected_ids) or len(rows) != len(manifest):
        raise ValueError("official generations must exactly cover the frozen manifest")
    by_id = {row["prompt_id"]: row for row in rows}
    ordered = []
    for manifest_row in manifest:
        row = by_id[manifest_row["id"]]
        if row.get("prompt_sha256") != manifest_row["sha256"]:
            raise ValueError("official generation prompt hash differs from manifest")
        if row.get("generation_purpose") != "official_safety_evaluation":
            raise ValueError("refusing to judge a non-official generation cache")
        if row.get("condition") != expected_condition:
            raise ValueError("official generation condition differs from requested condition")
        if row.get("evaluator_payload") is not None or row.get("evaluator_score") is not None:
            raise ValueError("official generation cache was already exposed to an evaluator")
        if row.get("truncated") is not False or row.get("stop_reason") != "eos_token":
            raise ValueError("refusing to judge an incomplete official generation")
        for field in (
            "checkpoint_sha256",
            "generation_config_sha256",
            "parsed_final_answer",
            "answer_tokens",
            "generated_tokens",
        ):
            if field not in row:
                raise ValueError(f"official generation is missing required field: {field}")
        ordered.append(row)
    return ordered
