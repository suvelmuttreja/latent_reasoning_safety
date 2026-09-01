import pytest

from mats_latent_safety.cap_calibration import (
    cap_projection,
    select_smallest_cap_by_k_or_none,
    select_smallest_cap,
    select_smallest_cap_or_none,
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
