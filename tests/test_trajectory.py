import pytest

from mats_latent_safety.trajectory import (
    stage_k_values,
    summarize_generation_rows,
    validate_trajectory_config,
)


def frozen_settings():
    config = {
        "capability": {"max_new_tokens": 512, "stop_tokens": ["eos_token"]},
        "coherence": {
            "max_new_tokens": 5120,
            "stop_tokens": ["eos_token"],
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
    }
    evaluation = {
        "sampling": {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
        "explicit_generation": {
            "frozen_max_new_tokens": 5120,
            "stop_tokens": ["eos_token"],
        },
        "coconut_generation": {
            "answer_max_new_tokens": 512,
            "stop_tokens": ["eos_token"],
        },
    }
    return config, evaluation


def test_stage_k_values_track_curriculum_endpoint():
    assert stage_k_values(2, 2, 3) == [0, 4]
    assert stage_k_values(3, 2, 3) == [0, 6]
    with pytest.raises(ValueError, match="stage must"):
        stage_k_values(4, 2, 3)


def test_trajectory_enforces_per_branch_caps():
    config, evaluation = frozen_settings()
    validate_trajectory_config(config, evaluation)
    config["coherence"]["max_new_tokens"] = 512
    with pytest.raises(ValueError, match="explicit-thinking cap"):
        validate_trajectory_config(config, evaluation)


def test_trajectory_summary_is_split_by_k():
    rows = [
        {
            "k": 0,
            "stop_reason": "eos_token",
            "generated_tokens": 10,
            "raw_output": "</think>ok",
            "seconds": 1.0,
        },
        {
            "k": 4,
            "stop_reason": "length",
            "generated_tokens": 5120,
            "raw_output": "unfinished",
            "seconds": 2.0,
        },
    ]
    summary = summarize_generation_rows(rows, 5120)
    assert summary["by_k"]["0"]["eos_stops"] == 1
    assert summary["by_k"]["4"]["cap_hits"] == 1
    assert summary["by_k"]["4"]["missing_closing_think"] == 1
