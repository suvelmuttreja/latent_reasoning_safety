import pytest

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
