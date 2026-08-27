import pytest
import torch

from mats_latent_safety.batching import pad_causal_records
from mats_latent_safety.constants import IGNORE_INDEX


def test_pad_causal_records_preserves_boundaries():
    records = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [IGNORE_INDEX, 2, 3],
        },
        {
            "input_ids": [4, 5],
            "attention_mask": [1, 1],
            "labels": [IGNORE_INDEX, 5],
        },
    ]
    batch = pad_causal_records(
        records, pad_token_id=0, device=torch.device("cpu"), pad_to_length=4
    )
    assert batch["input_ids"].tolist() == [[1, 2, 3, 0], [4, 5, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]]
    assert batch["labels"].tolist() == [
        [IGNORE_INDEX, 2, 3, IGNORE_INDEX],
        [IGNORE_INDEX, 5, IGNORE_INDEX, IGNORE_INDEX],
    ]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0], [0, 1, 0, 0]]


def test_pad_causal_records_rejects_short_target():
    records = [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]}]
    with pytest.raises(ValueError, match="below observed"):
        pad_causal_records(
            records, pad_token_id=0, device=torch.device("cpu"), pad_to_length=1
        )

