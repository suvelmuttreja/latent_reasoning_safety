"""Canonical clean-GSM8K conversion used by the access-independent smoke."""

from __future__ import annotations

import re


_FINAL = re.compile(r"\s*####\s*([^\n]+)\s*$")


def clean_gsm8k_row(question: str, answer: str) -> dict[str, object]:
    """Convert an OpenAI GSM8K row to the Meta Coconut question/steps/answer schema."""
    match = _FINAL.search(answer)
    if match is None:
        raise ValueError("GSM8K answer is missing the final #### marker")
    rationale = answer[: match.start()].strip()
    steps = [line.strip() for line in rationale.splitlines() if line.strip()]
    if not steps:
        raise ValueError("GSM8K rationale has no non-empty steps")
    return {"question": question.strip(), "steps": steps, "answer": match.group(1).strip()}

