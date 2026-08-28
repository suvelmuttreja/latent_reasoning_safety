from types import SimpleNamespace

import pytest
import torch

from mats_latent_safety.batching import pad_causal_records
from mats_latent_safety.constants import IGNORE_INDEX
from mats_latent_safety.training import run_update


class ToyTokenMeanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, input_ids, labels, **_kwargs):
        features = input_ids[..., 1:].float()
        targets = labels[..., 1:].clamp_min(0).float()
        mask = labels[..., 1:] != IGNORE_INDEX
        losses = (self.weight * features - targets).square()
        return SimpleNamespace(loss=losses[mask].mean())


RECORDS = [
    {
        "input_ids": list(range(1, length + 1)),
        "attention_mask": [1] * length,
        "labels": [IGNORE_INDEX, *range(2, length + 1)],
        "position_ids": list(range(length)),
    }
    for length in (2, 5, 3, 6)
]


def make_batches(micro_batch_size: int):
    return [
        pad_causal_records(
            RECORDS[offset : offset + micro_batch_size],
            pad_token_id=0,
            device=torch.device("cpu"),
        )
        for offset in range(0, len(RECORDS), micro_batch_size)
    ]


def run_partition(micro_batch_size: int):
    model = ToyTokenMeanModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = run_update(
        model,
        optimizer,
        make_batches(micro_batch_size),
        accumulation=len(RECORDS) // micro_batch_size,
        check_gradients=True,
    )
    return result, float(model.weight.detach())


def test_token_weighted_update_is_microbatch_partition_invariant():
    micro1, weight1 = run_partition(1)
    micro2, weight2 = run_partition(2)
    assert micro1["loss_tokens"] == micro2["loss_tokens"] == 12
    assert micro1["token_weighted_loss"] == pytest.approx(
        micro2["token_weighted_loss"], rel=1e-7
    )
    assert weight1 == pytest.approx(weight2, rel=1e-7)
    assert micro1["mean_microbatch_loss"] != pytest.approx(
        micro2["mean_microbatch_loss"], rel=1e-3
    )
    assert (
        micro1["loss_normalization"]
        == "shifted_nonignored_token_count_over_effective_batch"
    )


def test_update_rejects_microbatch_without_supervised_loss_tokens():
    empty = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "labels": torch.tensor([[IGNORE_INDEX, IGNORE_INDEX]]),
        "position_ids": torch.tensor([[0, 1]]),
    }
    model = ToyTokenMeanModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="supervised loss token"):
        run_update(model, optimizer, [empty], accumulation=1, check_gradients=True)

