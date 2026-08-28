"""Shared matched-training update semantics and accounting."""

from __future__ import annotations

import time

import numpy as np
import torch

from .constants import IGNORE_INDEX


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


def gradients_finite(model: torch.nn.Module) -> bool:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return bool(gradients) and all(torch.isfinite(gradient).all() for gradient in gradients)


def shifted_supervised_tokens(batch: dict[str, torch.Tensor]) -> int:
    """Count exactly the labels included by next-token causal cross-entropy."""
    return int((batch["labels"][..., 1:] != IGNORE_INDEX).sum().item())


def aggregate_token_weighted_loss(rows: list[dict[str, float | int | bool]]) -> float:
    total = sum(int(row["loss_tokens"]) for row in rows)
    if total <= 0:
        raise ValueError("cannot aggregate updates without supervised loss tokens")
    return sum(float(row["token_weighted_loss"]) * int(row["loss_tokens"]) for row in rows) / total


def run_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[dict[str, torch.Tensor]],
    accumulation: int,
    *,
    check_gradients: bool,
) -> dict[str, float | int | bool | str]:
    """Run one effective-batch update with token-correct loss normalization.

    Each model loss is a mean over that micro-batch's shifted, non-ignored
    labels. Weighting by the corresponding token count makes the accumulated
    gradient the mean over all supervised tokens in the effective batch,
    independent of how examples are partitioned into micro-batches (up to
    floating-point accumulation order).
    """
    if not 1 <= len(batches) <= accumulation:
        raise ValueError("an optimizer update requires between 1 and accumulation micro-batches")
    loss_tokens_by_batch = [shifted_supervised_tokens(batch) for batch in batches]
    if any(count <= 0 for count in loss_tokens_by_batch):
        raise ValueError("every micro-batch must contain at least one supervised loss token")
    total_loss_tokens = sum(loss_tokens_by_batch)
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    nonpadding_tokens = 0
    supervised_tokens = 0
    examples = 0
    started = time.perf_counter()
    for batch, loss_tokens in zip(batches, loss_tokens_by_batch, strict=True):
        output = model(**batch)
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError(f"non-finite loss: {output.loss}")
        (output.loss * (loss_tokens / total_loss_tokens)).backward()
        losses.append(float(output.loss.detach().float().cpu()))
        nonpadding_tokens += int(batch["attention_mask"].sum().item())
        supervised_tokens += int((batch["labels"] != IGNORE_INDEX).sum().item())
        examples += int(batch["input_ids"].shape[0])
    finite = gradients_finite(model) if check_gradients else True
    if not finite:
        raise RuntimeError("missing or non-finite gradients")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(parameter.is_cuda for parameter in model.parameters()):
        torch.cuda.synchronize()
    return {
        "seconds": time.perf_counter() - started,
        "token_weighted_loss": sum(
            loss * count for loss, count in zip(losses, loss_tokens_by_batch, strict=True)
        )
        / total_loss_tokens,
        "mean_microbatch_loss": sum(losses) / len(losses),
        "loss_tokens": total_loss_tokens,
        "loss_normalization": "shifted_nonignored_token_count_over_effective_batch",
        "examples": examples,
        "nonpadding_tokens": nonpadding_tokens,
        "supervised_tokens": supervised_tokens,
        "gradients_finite": finite,
    }

