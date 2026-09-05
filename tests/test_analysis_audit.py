"""Regression checks for censored outcomes, including correct cutoff parses."""

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "analysis_audit", Path(__file__).resolve().parents[1] / "scripts/audit_existing_analyses.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(i, correct, stop):
    return {"prompt_id": str(i), "correct": correct, "stop_reason": stop}


def test_correct_cutoff_parse_does_not_count_as_a_completed_outcome():
    rows = [row(0, True, "eos_token"), row(1, False, "eos_token"),
            row(2, True, "length"), row(3, False, "length")]
    result = audit.accuracy_summary(rows)
    assert result["all_row_cutoff_parser_accuracy"] == .5
    assert result["unknown_unfinished_outcome_bounds"] == [.25, .75]
    assert result["correct_at_cutoff_among_unfinished"] == 1
    assert result["complete_case_accuracy_descriptive"] == .5


def test_all_unfinished_correct_parses_still_have_unknown_completed_outcomes():
    result = audit.accuracy_summary([row(0, "True", "length"), row(1, "True", "length")])
    assert result["unknown_unfinished_outcome_bounds"] == [0, 1]
    assert result["complete_case_accuracy_descriptive"] is None


def test_no_missingness_collapses_bounds_to_exact_accuracy():
    result = audit.accuracy_summary([row(0, "True", "eos_token"), row(1, "False", "eos_token")])
    assert result["unknown_unfinished_outcome_bounds"] == [.5, .5]


def test_unrecognized_correctness_is_rejected():
    with pytest.raises(ValueError, match="Unexpected correctness"):
        audit.accuracy_summary([row(0, "false", "eos_token")])
