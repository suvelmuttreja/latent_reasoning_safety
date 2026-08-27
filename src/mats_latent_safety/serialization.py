"""Training and evaluation serialization with explicit, testable boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .constants import END_LATENT, IGNORE_INDEX, LATENT, START_LATENT, k_for_stage


class TokenizerLike(Protocol):
    eos_token_id: int

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...


@dataclass(frozen=True)
class TokenizedReasoningExample:
    question: list[int]
    steps: tuple[list[int], ...]
    answer: list[int]


def ensure_latent_tokens(tokenizer) -> dict[str, int]:
    existing = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    to_add = [token for token in (START_LATENT, END_LATENT, LATENT) if token not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    ids = {
        "start": tokenizer.convert_tokens_to_ids(START_LATENT),
        "end": tokenizer.convert_tokens_to_ids(END_LATENT),
        "latent": tokenizer.convert_tokens_to_ids(LATENT),
    }
    if len(set(ids.values())) != 3:
        raise ValueError(f"latent marker token IDs are not unique: {ids}")
    return ids


def tokenize_reasoning_example(tokenizer: TokenizerLike, example: dict) -> TokenizedReasoningExample:
    """Use the canonical Meta Coconut raw GSM serialization."""
    question = tokenizer.encode(example["question"] + "\n", add_special_tokens=True)
    steps = tuple(
        tokenizer.encode(step + "\n", add_special_tokens=False) for step in example["steps"]
    )
    answer = tokenizer.encode("### " + example["answer"], add_special_tokens=False)
    answer = answer + [tokenizer.eos_token_id]
    return TokenizedReasoningExample(question, steps, answer)


def build_training_record(
    tokenized: TokenizedReasoningExample,
    *,
    stage: int,
    marker_ids: dict[str, int],
    c_thought: int = 2,
    max_stage: int = 3,
    explicit_cot: bool = False,
) -> dict[str, list[int] | int]:
    """Build one registered CoT or standard Coconut-SFT training record."""
    if explicit_cot:
        k = 0
        skipped_steps = 0
        prefix = list(tokenized.question)
    else:
        k = k_for_stage(stage, c_thought, max_stage)
        skipped_steps = min(stage, len(tokenized.steps))
        prefix = (
            list(tokenized.question)
            + [marker_ids["start"]]
            + [marker_ids["latent"]] * k
            + [marker_ids["end"]]
        )
    supervised = [token for step in tokenized.steps[skipped_steps:] for token in step]
    supervised += list(tokenized.answer)
    input_ids = prefix + supervised
    labels = [IGNORE_INDEX] * len(prefix) + supervised
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "position_ids": list(range(len(input_ids))),
        "k": k,
        "skipped_steps": skipped_steps,
        "supervised_tokens": len(supervised),
    }


def build_coconut_question(
    question_ids: list[int], marker_ids: dict[str, int], k: int
) -> dict[str, list[int] | int]:
    if k < 0:
        raise ValueError("k must be non-negative")
    ids = (
        list(question_ids)
        + [marker_ids["start"]]
        + [marker_ids["latent"]] * k
        + [marker_ids["end"]]
    )
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "position_ids": list(range(len(ids))),
        "k": k,
    }


def serialize_native_chat(tokenizer, prompt: str, *, add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def evaluator_payload(forbidden_prompt: str, raw_output: str) -> dict[str, str]:
    """Primary S always judges only the parsed final answer."""
    from .parsing import parse_thinking_response

    return {
        "forbidden_prompt": forbidden_prompt,
        "response": parse_thinking_response(raw_output).final_answer,
    }

