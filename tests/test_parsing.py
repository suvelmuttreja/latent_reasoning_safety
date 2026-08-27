from mats_latent_safety.parsing import (
    extract_gsm8k_answer,
    is_incomplete_generation,
    parse_thinking_response,
)


def test_parse_normal_thinking_response():
    parsed = parse_thinking_response("<think>work\nmore</think> 42")
    assert parsed.thinking == "work\nmore"
    assert parsed.final_answer == "42"
    assert parsed.closing_delimiters == 1
    assert not parsed.missing_closing_delimiter


def test_missing_delimiter_is_all_final_answer():
    parsed = parse_thinking_response("unfinished but visible")
    assert parsed.thinking is None
    assert parsed.final_answer == "unfinished but visible"
    assert parsed.missing_closing_delimiter


def test_multiple_delimiters_split_on_last():
    parsed = parse_thinking_response("<think>a</think> accidental </think> final")
    assert parsed.thinking == "a</think> accidental"
    assert parsed.final_answer == "final"
    assert parsed.closing_delimiters == 2


def test_gsm8k_answer_extraction():
    assert extract_gsm8k_answer("reason\n#### 1,234") == "1234"
    assert extract_gsm8k_answer("The answer is -2.5.") == "-2.5"
    assert extract_gsm8k_answer("no number") is None


def test_incomplete_generation_requires_cap_and_missing_close():
    assert is_incomplete_generation("<think>unfinished", stop_reason="length", generated_tokens=10, max_new_tokens=10)
    assert not is_incomplete_generation("<think>x</think> y", stop_reason="length", generated_tokens=10, max_new_tokens=10)
    assert not is_incomplete_generation("<think>unfinished", stop_reason="eos_token", generated_tokens=9, max_new_tokens=10)

