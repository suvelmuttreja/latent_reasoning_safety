"""Catch stale cross-script imports before queued Slurm jobs start."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script",
    sorted(path.name for path in (Path(__file__).parents[1] / "scripts").glob("*.py")),
)
def test_entry_points_import_without_starting_main(monkeypatch, script):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.setattr(sys, "path", [str(scripts), *sys.path])
    namespace = runpy.run_path(str(scripts / script), run_name="import_test")
    assert callable(namespace["main"])


def test_official_coconut_serialization_uses_native_prefix_before_scaffolding(monkeypatch):
    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "hello"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "<chat>hello<assistant><think>\n"

        def encode(self, text, *, add_special_tokens):
            assert text == "<chat>hello<assistant><think>\n"
            assert add_special_tokens is False
            return [4, 5, 6]

    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.setattr(sys, "path", [str(scripts), *sys.path])
    namespace = runpy.run_path(str(scripts / "gate_4b_coco_u1.py"), run_name="import_test")
    ids, rendered = namespace["serialize_question"](
        FakeTokenizer(), " hello ", "native_qwen_chat_with_latent_scaffold"
    )
    assert ids == [4, 5, 6]
    assert rendered == "<chat>hello<assistant><think>\n"
