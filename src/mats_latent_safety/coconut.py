"""Standard Coconut wrapper with minimal Qwen cache compatibility.

The scientific behavior is inherited from facebookresearch/coconut at
27273cb8cca4bb763c041a63b036d0c3b7cbbb48: every latent position receives the
preceding token's final-layer hidden state. This module intentionally contains
no hidden-state projection, suppressed-activation extraction, special latent
position IDs, auxiliary loss, or alternate layer selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CoconutOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    inputs_embeds: torch.Tensor
    past_key_values: Any


@dataclass(frozen=True)
class CoconutGeneration:
    token_ids: list[int]
    stop_reason: str
    generated_tokens: int


def _cache_to_legacy(cache: Any) -> Any:
    if cache is None:
        return None
    converter = getattr(cache, "to_legacy_cache", None)
    return converter() if converter is not None else cache


def _cache_from_legacy(legacy: Any, template: Any) -> Any:
    if legacy is None or template is None or isinstance(template, (tuple, list)):
        return legacy
    cls = type(template)
    converter = getattr(cls, "from_legacy_cache", None)
    return converter(legacy) if converter is not None else legacy


def _crop_cache(cache: Any, length: int) -> Any:
    """Return a differentiable cache containing positions ``[0, length)``."""
    if cache is None:
        return None
    legacy = _cache_to_legacy(cache)
    cropped = []
    for layer in legacy:
        if len(layer) < 2:
            raise ValueError("unexpected cache layer without key/value tensors")
        key, value, *rest = layer
        cropped.append((key[..., :length, :], value[..., :length, :], *rest))
    return _cache_from_legacy(tuple(cropped), cache)


def _sample_next_token(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if not do_sample:
        return logits.argmax(dim=-1)
    if temperature <= 0:
        raise ValueError("temperature must be positive when sampling")
    scores = logits / temperature
    if top_k > 0:
        cutoff = torch.topk(scores, min(top_k, scores.shape[-1]), dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)
    if top_p < 1:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        probs = torch.softmax(sorted_scores, dim=-1)
        remove = probs.cumsum(dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
        scores = torch.full_like(scores, -torch.inf).scatter(-1, sorted_indices, sorted_scores)
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


class StandardCoconut(nn.Module):
    """Final-layer recurrent-latent wrapper compatible with Qwen3 caches."""

    def __init__(
        self,
        base_causallm: nn.Module,
        *,
        latent_token_id: int,
        start_latent_id: int,
        end_latent_id: int,
        eos_token_id: int,
    ) -> None:
        super().__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = int(latent_token_id)
        self.start_latent_id = int(start_latent_id)
        self.end_latent_id = int(end_latent_id)
        self.eos_token_id = int(eos_token_id)
        self.embedding = self.base_causallm.get_input_embeddings()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_causallm.train(mode)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> CoconutOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        latent_indices = (input_ids == self.latent_token_id).nonzero(as_tuple=False)
        latent_lists = [
            [int(index[1]) for index in latent_indices if int(index[0]) == batch_index]
            for batch_index in range(input_ids.shape[0])
        ]
        max_latents = max((len(indices) for indices in latent_lists), default=0)
        inputs_embeds = self.embedding(input_ids)
        next_range = (0, input_ids.shape[1])
        if max_latents:
            next_range = (0, int(latent_indices[:, 1].min()))

        cache = None
        logits_parts: list[torch.Tensor] = []
        for latent_pass in range(max_latents):
            start, end = next_range
            cropped_cache = _crop_cache(cache, start)
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, start:end, :],
                attention_mask=attention_mask[:, :end],
                position_ids=position_ids[:, start:end],
                past_key_values=cropped_cache,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
                **kwargs,
            )
            logits_parts.append(outputs.logits)
            hidden = outputs.hidden_states[-1]
            offset = start
            cache = outputs.past_key_values
            replacements: dict[tuple[int, int], torch.Tensor] = {}
            for batch_index, positions in enumerate(latent_lists):
                if latent_pass < len(positions):
                    position = positions[latent_pass]
                    replacements[(batch_index, position)] = hidden[
                        batch_index, position - 1 - offset, :
                    ]
            rows = []
            for batch_index in range(inputs_embeds.shape[0]):
                row = [
                    replacements.get((batch_index, position), inputs_embeds[batch_index, position])
                    for position in range(inputs_embeds.shape[1])
                ]
                rows.append(torch.stack(row))
            inputs_embeds = torch.stack(rows)
            next_range = (
                end,
                input_ids.shape[1] if latent_pass + 1 >= max_latents else end + 1,
            )

        start, end = next_range
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, start:end, :],
            attention_mask=attention_mask[:, :end],
            position_ids=position_ids[:, start:end],
            past_key_values=_crop_cache(cache, start),
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
            **kwargs,
        )
        logits_parts.append(outputs.logits)
        logits = torch.cat(logits_parts, dim=-2)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CoconutOutput(loss, logits, inputs_embeds, outputs.past_key_values)

    @torch.no_grad()
    def generate_from_scaffold(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        seed: int = 42,
    ) -> CoconutGeneration:
        if input_ids.shape[0] != 1:
            raise ValueError("latent generation currently requires batch size 1")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        initial = self.forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        cache = initial.past_key_values
        logits = initial.logits[:, -1, :]
        generated: list[int] = []
        rng = torch.Generator(device=input_ids.device)
        rng.manual_seed(seed)
        stop_reason = "length"
        current_length = input_ids.shape[1]
        for _ in range(max_new_tokens):
            token = _sample_next_token(
                logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generator=rng,
            )
            token_id = int(token.item())
            generated.append(token_id)
            if token_id == self.eos_token_id:
                stop_reason = "eos_token"
                break
            current_length += 1
            step_mask = torch.ones((1, current_length), dtype=attention_mask.dtype, device=input_ids.device)
            step_position = torch.tensor([[current_length - 1]], device=input_ids.device)
            step = self.base_causallm(
                input_ids=token.view(1, 1),
                attention_mask=step_mask,
                position_ids=step_position,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = step.past_key_values
            logits = step.logits[:, -1, :]
        return CoconutGeneration(generated, stop_reason, len(generated))


def initialize_latent_embeddings(model: nn.Module, marker_ids: dict[str, int], anchor_id: int) -> None:
    """Copy one known embedding into all three new marker rows, as Meta does."""
    with torch.no_grad():
        input_weight = model.get_input_embeddings().weight
        output = model.get_output_embeddings()
        for marker_id in marker_ids.values():
            input_weight[marker_id].copy_(input_weight[anchor_id])
            if output is not None and output.weight.shape[0] > marker_id:
                output.weight[marker_id].copy_(output.weight[anchor_id])

