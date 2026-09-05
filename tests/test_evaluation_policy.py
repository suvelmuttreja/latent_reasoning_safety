from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_endpoint_cap_guard_is_symmetric_and_official_outputs_are_regenerated():
    evaluation = yaml.safe_load((ROOT / "configs/evaluation.yaml").read_text())
    guard = evaluation["endpoint_truncation_guard"]
    assert guard["scope"] == ("both_explicit_cot_and_coconut_at_every_official_stage_endpoint")
    assert guard["threshold"] == 0.05
    assert guard["stage3_budget_assumption"] == ("allow_one_pause_and_full_regeneration_cycle")
    generation = evaluation["official_generation_policy"]
    assert generation["calibration_generations_are_never_judged"] is True
    assert generation["source"] == (
        "regenerate_all_official_conditions_from_scratch_after_cap_commit"
    )
    assert generation["deterministic_prefix_derivation_from_calibration"] == ("prohibited")


def test_pinned_judge_truncation_is_explicit_and_has_full_output_companions():
    evaluation = yaml.safe_load((ROOT / "configs/evaluation.yaml").read_text())
    judge = evaluation["evaluator"]
    assert judge["max_response_length"] == 512
    assert judge["truncation_side"] == "left_keep_rightmost_tokens"
    assert "frozen_12_prompt_human_audit_on_full_outputs" in judge["required_companion_evidence"]
