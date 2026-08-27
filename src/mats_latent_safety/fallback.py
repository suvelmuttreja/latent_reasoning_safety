"""Validation helpers for the approval-latency fallback."""

from __future__ import annotations


def validate_resume(metadata: dict, config: dict, stage: int) -> None:
    expected = stage - 1
    if metadata.get("completed_stage") != expected:
        raise ValueError(
            f"stage {stage} requires a completed stage {expected} checkpoint, "
            f"found {metadata.get('completed_stage')}"
        )
    for key in ("model_id", "model_revision"):
        if metadata.get(key) != config[key]:
            raise ValueError(f"resume {key} does not match the frozen config")

