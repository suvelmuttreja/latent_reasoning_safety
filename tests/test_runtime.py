"""Provenance must identify the project even in exported or dirty working trees."""

import subprocess

import pytest

from mats_latent_safety import runtime


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    (tmp_path / "code.py").write_text("x = 1\n")
    git(tmp_path, "add", "code.py")
    git(tmp_path, "commit", "-qm", "initial")
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    return tmp_path


def test_revision_uses_project_root_instead_of_current_directory(checkout, monkeypatch):
    expected = git(checkout, "rev-parse", "HEAD")
    monkeypatch.chdir(checkout.parent)
    assert runtime.git_revision() == expected


def test_dirty_tracked_code_is_not_reported_as_a_clean_commit(checkout):
    expected = git(checkout, "rev-parse", "HEAD")
    (checkout / "code.py").write_text("x = 2\n")
    assert runtime.git_revision() == expected + "-dirty"


def test_export_marker_overrides_stale_remote_git_metadata(checkout):
    revision = "a" * 40
    (checkout / ".source-revision").write_text(revision + "\n")
    assert runtime.git_revision() == revision


def test_invalid_export_marker_is_rejected(checkout):
    (checkout / ".source-revision").write_text("unknown\n")
    with pytest.raises(ValueError, match="Invalid source revision"):
        runtime.git_revision()
