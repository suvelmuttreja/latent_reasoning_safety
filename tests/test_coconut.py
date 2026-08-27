from types import SimpleNamespace

import torch
from torch import nn

from mats_latent_safety.coconut import StandardCoconut


class TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 16, hidden: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, **kwargs):
        hidden = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        hidden = hidden + 0.25
        batch, length, width = hidden.shape
        if past_key_values is None:
            previous = hidden.new_zeros((batch, 1, 0, width))
        else:
            previous = past_key_values[0][0]
        current = hidden.unsqueeze(1)
        cache_tensor = torch.cat([previous, current], dim=2)
        cache = ((cache_tensor, cache_tensor),)
        return SimpleNamespace(
            logits=self.head(hidden),
            hidden_states=(hidden,),
            past_key_values=cache,
        )


def wrapper():
    torch.manual_seed(0)
    return StandardCoconut(
        TinyCausalLM(),
        latent_token_id=12,
        start_latent_id=13,
        end_latent_id=14,
        eos_token_id=15,
    )


def test_forward_reinjects_preceding_final_layer_hidden_state():
    model = wrapper()
    ids = torch.tensor([[1, 13, 12, 12, 14, 2]])
    output = model(ids, labels=ids, attention_mask=torch.ones_like(ids))
    original = model.embedding(ids)
    assert torch.allclose(output.inputs_embeds[0, 2], original[0, 1] + 0.25)
    assert torch.allclose(output.inputs_embeds[0, 3], output.inputs_embeds[0, 2] + 0.25)
    assert torch.isfinite(output.loss)


def test_k_selection_changes_number_of_recurrent_positions():
    model = wrapper()
    zero = torch.tensor([[1, 13, 14]])
    two = torch.tensor([[1, 13, 12, 12, 14]])
    assert model(zero).inputs_embeds.shape[1] == 3
    assert model(two).inputs_embeds.shape[1] == 5


def test_greedy_generation_is_deterministic():
    model = wrapper().eval()
    ids = torch.tensor([[1, 13, 12, 12, 14]])
    first = model.generate_from_scaffold(ids, max_new_tokens=3, seed=42)
    second = model.generate_from_scaffold(ids, max_new_tokens=3, seed=42)
    assert first.token_ids == second.token_ids

