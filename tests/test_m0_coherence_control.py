from mats_latent_safety.coherence_audit import summarize_coherence_rows
from mats_latent_safety.coherence_audit import validate_control_against_frozen_evaluation


def test_control_settings_match_frozen_sampler_and_shared_cap():
    control = {
        "sampling": {
            "cap_role": "coconut_answer_diagnostic",
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "stop_tokens": ["eos_token"],
        }
    }
    evaluation = {
        "sampling": {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
        "explicit_generation": {
            "stop_tokens": ["eos_token"],
            "frozen_max_new_tokens": 5120,
        },
        "coconut_generation": {"answer_max_new_tokens": 512},
    }
    validate_control_against_frozen_evaluation(control, evaluation)


def test_adequate_explicit_thinking_cap_matches_frozen_config():
    control = {
        "sampling": {
            "cap_role": "explicit_thinking",
            "max_new_tokens": 5120,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "stop_tokens": ["eos_token"],
        }
    }
    evaluation = {
        "sampling": {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        },
        "explicit_generation": {
            "stop_tokens": ["eos_token"],
            "frozen_max_new_tokens": 5120,
        },
        "coconut_generation": {"answer_max_new_tokens": 512},
    }
    validate_control_against_frozen_evaluation(control, evaluation)


def test_summary_counts_cap_and_missing_marker_separately():
    rows = [
        {
            "replicate": 0,
            "stop_reason": "length",
            "generated_tokens": 512,
            "has_closing_think": False,
        },
        {
            "replicate": 0,
            "stop_reason": "length",
            "generated_tokens": 512,
            "has_closing_think": True,
        },
        {
            "replicate": 1,
            "stop_reason": "eos_token",
            "generated_tokens": 10,
            "has_closing_think": True,
        },
    ]
    result = summarize_coherence_rows(rows, 512)
    assert result["overall"]["cap_hits"] == 2
    assert result["overall"]["missing_closing_think"] == 1
    assert result["overall"]["cap_and_missing_closing_think"] == 1
    assert result["by_replicate"]["1"]["eos_stops"] == 1
