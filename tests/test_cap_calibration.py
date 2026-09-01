import pytest

from mats_latent_safety.cap_calibration import (
    cap_projection,
    select_smallest_cap_by_k_or_none,
    select_smallest_cap,
    select_smallest_cap_or_none,
    validate_partial_calibration_prefix,
)


def rows(lengths_by_k):
    return [
        {"k": k, "generated_tokens": length, "stop_reason": "eos_token"}
        for k, lengths in lengths_by_k.items()
        for length in lengths
    ]


def test_projection_is_per_condition():
    result = cap_projection(rows({0: [100, 600], 2: [200, 300]}), [512])
    assert result["512"]["0"]["projected_truncation_rate"] == 0.5
    assert result["512"]["2"]["projected_truncation_rate"] == 0.0


def test_selection_requires_strictly_less_than_five_percent_for_every_k():
    sample = rows({0: [100] * 19 + [600], 2: [100] * 20})
    selected, _ = select_smallest_cap(sample, [512, 1024], 0.05)
    assert selected == 1024


def test_selection_accepts_four_percent_and_rejects_unordered_candidates():
    sample = rows({0: [100] * 24 + [600], 2: [100] * 25})
    selected, _ = select_smallest_cap(sample, [512, 1024], 0.05)
    assert selected == 512
    with pytest.raises(ValueError, match="unique and increasing"):
        select_smallest_cap(sample, [1024, 512], 0.05)


def test_nonpassing_calibration_preserves_projection():
    sample = rows({0: [100] * 18 + [600] * 2, 2: [100] * 20})
    selected, projection = select_smallest_cap_or_none(sample, [512], 0.05)
    assert selected is None
    assert projection["512"]["0"]["projected_truncation_rate"] == 0.1


def test_per_k_selection_does_not_let_ood_k0_block_natural_condition():
    sample = rows({0: [100] * 18 + [600] * 2, 4: [100] * 20})
    selected, projection = select_smallest_cap_by_k_or_none(sample, [512], 0.05)
    assert selected == {"0": None, "4": 512}
    assert projection["512"]["0"]["projected_truncation_rate"] == 0.1


def partial_row(prompt_id="a", prompt_hash="hash-a"):
    return {
        "k": 0,
        "prompt_id": prompt_id,
        "prompt_sha256": prompt_hash,
        "model_id": "model",
        "model_revision": "revision",
        "checkpoint_sha256": "checkpoint",
        "generation_config_sha256": "generation",
        "partial_run_config_sha256": "generation",
        "evaluator_payload": None,
        "evaluator_score": None,
    }


def test_partial_calibration_requires_exact_unjudged_grid_prefix():
    grid = [
        {"k": 0, "prompt_id": "a", "prompt_sha256": "hash-a"},
        {"k": 0, "prompt_id": "b", "prompt_sha256": "hash-b"},
    ]
    validate_partial_calibration_prefix(
        [partial_row()],
        grid,
        model_id="model",
        model_revision="revision",
        checkpoint_sha256="checkpoint",
        partial_run_config_sha256="generation",
    )
    with pytest.raises(ValueError, match="prefix"):
        validate_partial_calibration_prefix(
            [partial_row("b", "hash-b")],
            grid,
            model_id="model",
            model_revision="revision",
            checkpoint_sha256="checkpoint",
            partial_run_config_sha256="generation",
        )


def test_partial_calibration_rejects_provenance_or_evaluator_data():
    grid = [{"k": 0, "prompt_id": "a", "prompt_sha256": "hash-a"}]
    row = partial_row()
    row["evaluator_score"] = 0.0
    with pytest.raises(ValueError, match="provenance"):
        validate_partial_calibration_prefix(
            [row],
            grid,
            model_id="model",
            model_revision="revision",
            checkpoint_sha256="checkpoint",
            partial_run_config_sha256="generation",
        )
