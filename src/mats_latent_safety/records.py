"""Strict generation-record schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationRecord:
    prompt_id: str
    prompt_sha256: str
    model_id: str
    model_revision: str
    code_revision: str
    raw_serialized_input: str
    raw_output: str
    parsed_thinking: str | None
    parsed_final_answer: str
    k: int | None
    input_tokens: int
    thinking_tokens: int | None
    answer_tokens: int
    stop_reason: str
    truncated: bool
    generation_config_sha256: str
    evaluator_payload: dict[str, Any] | None = None
    evaluator_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
