"""Fail-closed validation for cached official safety generations."""

from __future__ import annotations


def resolve_final_safety_cap(evaluation: dict, branch: str) -> int:
    """Return only a cap frozen from the actual final branch endpoint."""
    if branch == "explicit_cot":
        generation = evaluation["explicit_generation"]
    elif branch == "coconut_skip0":
        generation = evaluation["coconut_generation"]
    else:
        raise ValueError(f"unsupported official branch: {branch}")
    task = generation.get("task_caps", {}).get("safety")
    if not task or task.get("status") != "frozen_final_endpoint_calibrated":
        raise ValueError(f"official {branch} safety cap is not final-endpoint frozen")
    cap = task.get("max_new_tokens")
    if not isinstance(cap, int) or cap <= 0:
        raise ValueError(f"official {branch} safety cap is invalid")
    return cap


def validate_official_safety_rows(
    rows: list[dict],
    manifest: list[dict],
    expected_condition: str,
    max_incomplete_rate: float = 0.0,
) -> list[dict]:
    if not expected_condition:
        raise ValueError("official condition must be non-empty")
    if not 0 <= max_incomplete_rate < 1:
        raise ValueError("official maximum incomplete rate must be in [0, 1)")
    ids = [row.get("prompt_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("official generations contain duplicate prompt IDs")
    expected_ids = [row["id"] for row in manifest]
    if set(ids) != set(expected_ids) or len(rows) != len(manifest):
        raise ValueError("official generations must exactly cover the frozen manifest")
    by_id = {row["prompt_id"]: row for row in rows}
    incomplete = sum(
        row.get("truncated") is not False or row.get("stop_reason") != "eos_token" for row in rows
    )
    if incomplete and (max_incomplete_rate == 0 or incomplete / len(rows) >= max_incomplete_rate):
        raise ValueError("refusing to judge an incomplete official generation")
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


def partition_official_safety_rows_for_bounds(
    rows: list[dict],
    manifest: list[dict],
    expected_condition: str,
) -> tuple[list[dict], list[dict]]:
    """Validate a full official cache and partition EOS and nonterminating rows.

    Unlike ``validate_official_safety_rows``, this does not turn a failed cap
    guard into a point estimate.  It is only for partial-identification bounds:
    the judge sees EOS-completed responses, while every incomplete response is
    retained as an unknown score in [0, 1].
    """
    ordered = validate_official_safety_rows(
        rows,
        manifest,
        expected_condition,
        max_incomplete_rate=1.0 - 1e-12,
    )
    completed = [
        row
        for row in ordered
        if row.get("truncated") is False and row.get("stop_reason") == "eos_token"
    ]
    incomplete = [row for row in ordered if row not in completed]
    if not incomplete:
        raise ValueError("bounded scoring requires at least one incomplete official row")
    return completed, incomplete
