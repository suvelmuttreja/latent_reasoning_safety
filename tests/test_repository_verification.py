"""Evidence checks must reject changed, missing, or unindexed scientific files."""

import hashlib

import pytest

from verify_repository import compare_analysis, verify_checksums


@pytest.fixture
def evidence(tmp_path):
    target = tmp_path / "artifacts/discovery/results/run/result.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"score": 0.25}\n')
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    (tmp_path / "artifacts/checksums.sha256").write_text(
        f"{checksum}  artifacts/discovery/results/run/result.json\n"
    )
    return tmp_path, target


def test_saved_evidence_verifies_without_git_metadata(evidence):
    root, target = evidence
    tokenizer = target.parent / "tokenizer/tokenizer.json"
    tokenizer.parent.mkdir()
    tokenizer.write_text("{}")  # ignored model cache is not research evidence
    assert verify_checksums(root) == 1


def test_changed_evidence_is_rejected(evidence):
    root, target = evidence
    target.write_text('{"score": 0.75}\n')
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksums(root)


def test_missing_evidence_is_rejected(evidence):
    root, target = evidence
    target.unlink()
    with pytest.raises(ValueError, match="inventory differs"):
        verify_checksums(root)


def test_unindexed_result_is_rejected(evidence):
    root, target = evidence
    target.with_name("another.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory differs"):
        verify_checksums(root)


def test_analysis_comparison_accepts_roundoff_but_rejects_changed_findings():
    compare_analysis({"correlation": 0.39183561028318714}, {"correlation": 0.3918356102831872})
    with pytest.raises(ValueError, match="audit.correlation"):
        compare_analysis({"correlation": 0.3918}, {"correlation": 0.3918356102831872})
    with pytest.raises(ValueError, match="audit.n"):
        compare_analysis({"n": 59}, {"n": 60})
