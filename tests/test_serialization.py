from mats_latent_safety.constants import IGNORE_INDEX, k_for_stage, optimizer_updates
from mats_latent_safety.serialization import (
    TokenizedReasoningExample,
    build_coconut_question,
    build_training_record,
    evaluator_payload,
)


MARKERS = {"start": 90, "end": 91, "latent": 92}
EXAMPLE = TokenizedReasoningExample([1, 2], ([3, 4], [5]), [6, 7])


def test_registered_stage_to_k_mapping():
    assert [k_for_stage(stage) for stage in range(4)] == [0, 2, 4, 6]


def test_update_accounting_includes_partial_batch():
    assert optimizer_updates(7473, 2, 32) == 468


def test_stage_one_replaces_first_step_and_masks_prefix():
    record = build_training_record(EXAMPLE, stage=1, marker_ids=MARKERS)
    assert record["k"] == 2
    assert record["skipped_steps"] == 1
    assert record["input_ids"] == [1, 2, 90, 92, 92, 91, 5, 6, 7]
    assert record["labels"] == [IGNORE_INDEX] * 6 + [5, 6, 7]


def test_explicit_cot_keeps_all_steps_and_no_markers():
    record = build_training_record(EXAMPLE, stage=3, marker_ids=MARKERS, explicit_cot=True)
    assert record["k"] == 0
    assert record["input_ids"] == [1, 2, 3, 4, 5, 6, 7]
    assert record["labels"] == [IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5, 6, 7]


def test_k_zero_still_uses_scaffold_boundaries():
    record = build_coconut_question([1, 2], MARKERS, 0)
    assert record["input_ids"] == [1, 2, 90, 91]
    assert record["k"] == 0


def test_evaluator_payload_uses_final_answer_only():
    payload = evaluator_payload("forbidden", "<think>hidden material</think> refusal")
    assert payload == {"forbidden_prompt": "forbidden", "response": "refusal"}

