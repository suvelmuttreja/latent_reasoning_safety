import pytest

from mats_latent_safety.official_eval import resolve_final_safety_cap
from mats_latent_safety.official_eval import validate_official_safety_rows


def manifest():
    return [
        {"id": "a", "sha256": "hash-a", "forbidden_prompt": "prompt a"},
        {"id": "b", "sha256": "hash-b", "forbidden_prompt": "prompt b"},
    ]


def generation(prompt_id, prompt_hash):
    return {
        "prompt_id": prompt_id,
        "prompt_sha256": prompt_hash,
        "generation_purpose": "official_safety_evaluation",
        "condition": "coco_u3_k6",
        "evaluator_payload": None,
        "evaluator_score": None,
        "truncated": False,
        "stop_reason": "eos_token",
        "checkpoint_sha256": "checkpoint",
        "generation_config_sha256": "generation",
        "parsed_final_answer": "answer",
        "answer_tokens": 1,
        "generated_tokens": 2,
    }


def test_official_rows_are_returned_in_manifest_order():
    rows = [generation("b", "hash-b"), generation("a", "hash-a")]
    ordered = validate_official_safety_rows(rows, manifest(), "coco_u3_k6")
    assert [row["prompt_id"] for row in ordered] == ["a", "b"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.append(dict(rows[0])), "duplicate"),
        (lambda rows: rows[0].update(prompt_sha256="wrong"), "prompt hash"),
        (lambda rows: rows[0].update(condition="coco_u3_k0"), "condition"),
        (lambda rows: rows[0].update(generation_purpose="calibration"), "non-official"),
        (lambda rows: rows[0].update(evaluator_score=0.0), "already exposed"),
        (lambda rows: rows[0].update(truncated=True), "incomplete"),
    ],
)
def test_official_rows_fail_closed(mutation, message):
    rows = [generation("a", "hash-a"), generation("b", "hash-b")]
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        validate_official_safety_rows(rows, manifest(), "coco_u3_k6")


def test_official_rows_allow_incomplete_below_registered_aggregate_threshold():
    rows = [generation("a", "hash-a"), generation("b", "hash-b")]
    rows[0].update(truncated=True, stop_reason="length")
    ordered = validate_official_safety_rows(
        rows,
        manifest(),
        "coco_u3_k6",
        max_incomplete_rate=0.6,
    )
    assert [row["prompt_id"] for row in ordered] == ["a", "b"]


def test_official_rows_reject_incomplete_at_registered_aggregate_threshold():
    rows = [generation("a", "hash-a"), generation("b", "hash-b")]
    rows[0].update(truncated=True, stop_reason="length")
    with pytest.raises(ValueError, match="incomplete"):
        validate_official_safety_rows(
            rows,
            manifest(),
            "coco_u3_k6",
            max_incomplete_rate=0.5,
        )


def test_final_safety_cap_requires_endpoint_freeze_for_each_branch():
    evaluation = {
        "explicit_generation": {
            "task_caps": {
                "safety": {
                    "status": "frozen_final_endpoint_calibrated",
                    "max_new_tokens": 5120,
                }
            }
        },
        "coconut_generation": {
            "task_caps": {"safety": {"status": "blocked", "max_new_tokens": None}}
        },
    }
    assert resolve_final_safety_cap(evaluation, "explicit_cot") == 5120
    with pytest.raises(ValueError, match="not final-endpoint frozen"):
        resolve_final_safety_cap(evaluation, "coconut_skip0")


def test_final_safety_cap_rejects_nonpositive_value():
    evaluation = {
        "coconut_generation": {
            "task_caps": {
                "safety": {
                    "status": "frozen_final_endpoint_calibrated",
                    "max_new_tokens": 0,
                }
            }
        }
    }
    with pytest.raises(ValueError, match="invalid"):
        resolve_final_safety_cap(evaluation, "coconut_skip0")
