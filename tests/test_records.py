import yaml

from mats_latent_safety.records import GenerationRecord


def test_generation_record_covers_every_frozen_required_field():
    record = GenerationRecord(
        prompt_id="p",
        prompt_sha256="h",
        model_id="m",
        model_revision="r",
        code_revision="c",
        raw_serialized_input="i",
        raw_output="o",
        parsed_thinking=None,
        parsed_final_answer="a",
        k=0,
        input_tokens=1,
        thinking_tokens=None,
        answer_tokens=1,
        stop_reason="eos_token",
        truncated=False,
        generation_config_sha256="g",
        evaluator_payload=None,
        evaluator_score=None,
    )
    config = yaml.safe_load(open("configs/evaluation.yaml"))
    assert set(config["records"]["required_fields"]) == set(record.to_dict())
