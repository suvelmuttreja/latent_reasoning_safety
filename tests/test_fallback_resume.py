import pytest

from mats_latent_safety.fallback import validate_resume


CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Thinking-2507",
    "model_revision": "revision",
}


def test_validate_resume_accepts_immediately_preceding_stage():
    validate_resume(
        {"completed_stage": 1, "model_id": CONFIG["model_id"], "model_revision": "revision"},
        CONFIG,
        2,
    )


def test_validate_resume_rejects_wrong_stage_or_model():
    with pytest.raises(ValueError, match="requires a completed stage"):
        validate_resume(
            {"completed_stage": 0, "model_id": CONFIG["model_id"], "model_revision": "revision"},
            CONFIG,
            2,
        )
    with pytest.raises(ValueError, match="does not match"):
        validate_resume(
            {"completed_stage": 1, "model_id": "wrong", "model_revision": "revision"},
            CONFIG,
            2,
        )
