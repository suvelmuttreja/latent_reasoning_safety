"""Validation helpers for the approval-latency fallback."""

from __future__ import annotations


TRIGGER_ACK = "persistent-403-fallback-authorized"
INLINE_GATE_ACK = "coco-u1-inline-gate-passed"


def validate_authorization(config: dict, stage: int, fallback_ack: str, gate_ack: str) -> None:
    branch = config.get("branch")
    status = config.get("submission_status")
    if branch == "coconut_skip0" and stage == 1:
        if status != "fallback_trigger_acknowledged" or fallback_ack != TRIGGER_ACK:
            raise ValueError("the registered approval-latency fallback trigger is not acknowledged")
        return
    if branch in {"coconut_skip0", "explicit_cot"} and stage >= 1:
        if status != "inline_gate_passed" or gate_ack != INLINE_GATE_ACK:
            raise ValueError("the coco_u1 in-line method gate has not authorized matched training")
        return
    raise ValueError(f"unsupported training branch: {branch}")


def validate_resume(metadata: dict, config: dict, stage: int) -> None:
    expected = stage - 1
    if metadata.get("completed_stage") != expected:
        raise ValueError(
            f"stage {stage} requires a completed stage {expected} checkpoint, "
            f"found {metadata.get('completed_stage')}"
        )
    for key in ("model_id", "model_revision", "branch"):
        if key not in config:
            continue
        if metadata.get(key) != config[key]:
            raise ValueError(f"resume {key} does not match the frozen config")
