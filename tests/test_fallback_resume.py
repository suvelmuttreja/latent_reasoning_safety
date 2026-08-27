import pytest

from mats_latent_safety.fallback import (
    INLINE_GATE_ACK,
    TRIGGER_ACK,
    validate_authorization,
    validate_resume,
)


CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Thinking-2507",
    "model_revision": "revision",
    "branch": "coconut_skip0",
}


def test_validate_resume_accepts_immediately_preceding_stage():
    validate_resume(
        {
            "completed_stage": 1,
            "model_id": CONFIG["model_id"],
            "model_revision": "revision",
            "branch": "coconut_skip0",
        },
        CONFIG,
        2,
    )


def test_validate_resume_rejects_wrong_stage_or_model():
    with pytest.raises(ValueError, match="requires a completed stage"):
        validate_resume(
            {
                "completed_stage": 0,
                "model_id": CONFIG["model_id"],
                "model_revision": "revision",
                "branch": "coconut_skip0",
            },
            CONFIG,
            2,
        )
    with pytest.raises(ValueError, match="does not match"):
        validate_resume(
            {
                "completed_stage": 1,
                "model_id": "wrong",
                "model_revision": "revision",
                "branch": "coconut_skip0",
            },
            CONFIG,
            2,
        )


def test_validate_resume_rejects_cross_branch_checkpoint():
    with pytest.raises(ValueError, match="resume branch"):
        validate_resume(
            {
                "completed_stage": 1,
                "model_id": CONFIG["model_id"],
                "model_revision": "revision",
                "branch": "explicit_cot",
            },
            CONFIG,
            2,
        )


def test_stage1_coconut_requires_fallback_acknowledgement():
    config = {"branch": "coconut_skip0", "submission_status": "fallback_trigger_acknowledged"}
    validate_authorization(config, 1, TRIGGER_ACK, "")
    with pytest.raises(ValueError, match="fallback trigger"):
        validate_authorization(config, 1, "", "")


def test_later_coconut_and_explicit_cot_require_inline_gate():
    for branch, stage in (("coconut_skip0", 2), ("explicit_cot", 1)):
        config = {"branch": branch, "submission_status": "inline_gate_passed"}
        validate_authorization(config, stage, "", INLINE_GATE_ACK)
        with pytest.raises(ValueError, match="in-line method gate"):
            validate_authorization(config, stage, "", "")
