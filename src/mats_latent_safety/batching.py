"""Small deterministic causal-LM batching helpers."""

from __future__ import annotations

import torch

from .constants import IGNORE_INDEX


def pad_causal_records(
    records: list[dict[str, list[int] | int]],
    *,
    pad_token_id: int,
    device: torch.device,
    pad_to_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """Right-pad causal records while preserving loss and attention boundaries."""
    if not records:
        raise ValueError("at least one record is required")
    observed = max(len(record["input_ids"]) for record in records)
    target = observed if pad_to_length is None else pad_to_length
    if target < observed:
        raise ValueError(f"pad_to_length={target} is below observed length {observed}")

    input_ids, attention_mask, labels, position_ids = [], [], [], []
    for record in records:
        length = len(record["input_ids"])
        padding = target - length
        input_ids.append(list(record["input_ids"]) + [pad_token_id] * padding)
        attention_mask.append(list(record["attention_mask"]) + [0] * padding)
        labels.append(list(record["labels"]) + [IGNORE_INDEX] * padding)
        positions = record.get("position_ids", list(range(length)))
        position_ids.append(list(positions) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "position_ids": torch.tensor(position_ids, dtype=torch.long, device=device),
    }


def pad_coconut_records(
    records: list[dict[str, list[int] | int]],
    *,
    latent_token_id: int,
    pad_token_id: int,
    device: torch.device,
    pad_to_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """Left-align first latent positions, then right-pad as in Meta Coconut."""
    if not records:
        raise ValueError("at least one record is required")
    starts = []
    for record in records:
        try:
            starts.append(list(record["input_ids"]).index(latent_token_id))
        except ValueError as error:
            raise ValueError("every Coconut record must contain a latent token") from error
    target_start = max(starts)
    aligned = []
    for record, start in zip(records, starts):
        left_padding = target_start - start
        aligned.append(
            {
                "input_ids": [pad_token_id] * left_padding + list(record["input_ids"]),
                "attention_mask": [0] * left_padding + list(record["attention_mask"]),
                "labels": [IGNORE_INDEX] * left_padding + list(record["labels"]),
                "position_ids": [0] * left_padding + list(record["position_ids"]),
            }
        )
    return pad_causal_records(
        aligned,
        pad_token_id=pad_token_id,
        device=device,
        pad_to_length=pad_to_length,
    )
