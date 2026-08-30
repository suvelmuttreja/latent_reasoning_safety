from __future__ import annotations

import runpy
import sys
from pathlib import Path


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert isinstance(text, str)
        assert add_special_tokens is False
        return list(range(len(text.split())))


def test_enrichment_counts_absent_coconut_thinking_as_zero(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.setattr(sys, "path", [str(scripts), *sys.path])
    namespace = runpy.run_path(
        str(scripts / "evaluate_coconut_stage_trajectory.py"), run_name="import_test"
    )
    rows = [
        {
            "prompt_id": "coherence-1",
            "k": 4,
            "raw_serialized_prefix": "prompt",
            "parsed_thinking": None,
            "parsed_final_answer": "clear answer",
        }
    ]
    namespace["enrich_rows"](
        rows,
        FakeTokenizer(),
        {"max_new_tokens": 512},
        "revision",
        "model",
        "model-revision",
        {"coherence-1": "prompt-hash"},
        "pending_blind_human_coherence_0_to_2",
    )
    assert rows[0]["thinking_tokens"] == 0
    assert rows[0]["answer_tokens"] == 2
