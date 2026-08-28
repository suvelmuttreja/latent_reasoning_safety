"""Catch stale cross-script imports before queued Slurm jobs start."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script",
    [
        "preflight_4b_cot.py",
        "preflight_4b_coconut.py",
        "train_4b_skip0_stage.py",
        "gate_4b_coco_u1.py",
        "claim_stage_race.py",
    ],
)
def test_gpu_scripts_import_without_starting_main(monkeypatch, script):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.setattr(sys, "path", [str(scripts), *sys.path])
    namespace = runpy.run_path(str(scripts / script), run_name="import_test")
    assert callable(namespace["main"])
