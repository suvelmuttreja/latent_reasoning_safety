from mats_latent_safety.constants import IGNORE_INDEX, k_for_stage, optimizer_updates
from mats_latent_safety.serialization import (
    TokenizedReasoningExample,
    build_coconut_question,
    build_explicit_question,
    build_training_record,
    evaluator_payload,
    serialize_native_chat,
    tokenize_coconut_raw_prompt,
    tokenize_native_chat_prompt,
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


def test_explicit_question_has_no_latent_scaffold_boundaries():
    record = build_explicit_question([1, 2])
    assert record["input_ids"] == [1, 2]
    assert record["attention_mask"] == [1, 1]
    assert record["position_ids"] == [0, 1]
    assert record["k"] == 0


def test_evaluator_payload_uses_final_answer_only():
    payload = evaluator_payload("forbidden", "<think>hidden material</think> refusal")
    assert payload == {"forbidden_prompt": "forbidden", "response": "refusal"}


def test_native_chat_serialization_is_single_user_turn_with_generation_prompt():
    class FakeChatTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "hello"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "SERIALIZED"

    assert serialize_native_chat(FakeChatTokenizer(), "hello") == "SERIALIZED"


def test_coconut_raw_eval_serialization_matches_training_question_boundary():
    class FakeRawTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert text == "hello\n"
            assert add_special_tokens is True
            return [1, 2, 3]

    assert tokenize_coconut_raw_prompt(FakeRawTokenizer(), "hello") == [1, 2, 3]


def test_native_chat_tokenization_does_not_add_special_tokens_twice():
    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "hello"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "<chat>hello<assistant><think>\n"

        def encode(self, text, *, add_special_tokens):
            assert text == "<chat>hello<assistant><think>\n"
            assert add_special_tokens is False
            return [4, 5, 6]

    ids, rendered = tokenize_native_chat_prompt(FakeTokenizer(), "hello")
    assert ids == [4, 5, 6]
    assert rendered == "<chat>hello<assistant><think>\n"
