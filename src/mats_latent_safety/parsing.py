"""Frozen parsers for Qwen thinking output and GSM8K answers."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedResponse:
    thinking: str | None
    final_answer: str
    closing_delimiters: int
    missing_closing_delimiter: bool


def parse_thinking_response(text: str) -> ParsedResponse:
    """Split on the last ``</think>``; missing delimiters leave all text final."""
    delimiter = "</think>"
    count = text.count(delimiter)
    if count == 0:
        return ParsedResponse(None, text.strip(), 0, True)
    thinking, final = text.rsplit(delimiter, 1)
    thinking = thinking.removeprefix("<think>").strip()
    return ParsedResponse(thinking, final.strip(), count, False)


_HASH_ANSWER = re.compile(r"####\s*([^\n]+)")
_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def normalize_numeric_answer(answer: str) -> str:
    value = answer.strip().replace(",", "")
    if value.endswith("."):
        value = value[:-1]
    return value


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the canonical GSM8K answer, preferring a ``####`` marker."""
    marked = _HASH_ANSWER.findall(text)
    if marked:
        candidates = _NUMBER.findall(marked[-1])
        return normalize_numeric_answer(candidates[-1]) if candidates else marked[-1].strip()
    candidates = _NUMBER.findall(text)
    return normalize_numeric_answer(candidates[-1]) if candidates else None


def is_incomplete_generation(
    raw_output: str,
    *,
    stop_reason: str,
    generated_tokens: int,
    max_new_tokens: int,
) -> bool:
    """Flag cap hits and missing completed Qwen reasoning as truncations."""
    hit_cap = stop_reason == "length" or generated_tokens >= max_new_tokens
    missing_close = "</think>" not in raw_output
    return hit_cap and missing_close

