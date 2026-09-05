"""Exercise source-export boundaries without contacting an HPC server."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sync_checkout(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/sync.sh", repo / "scripts/sync.sh")
    (repo / ".gitignore").write_text(".env\n")
    (repo / "tracked.txt").write_text("committed science\n")
    for args in [
        ["init", "-q"],
        ["config", "user.name", "Test"],
        ["config", "user.email", "test@example.invalid"],
        ["add", "."],
        ["commit", "-qm", "initial"],
    ]:
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    (repo / ".env").write_text("PRIVATE_LOCAL_SETTING=do-not-copy\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(
        "#!/usr/bin/env python3\nimport os,sys,json\n"
        'with open(os.environ["SSH_LOG"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
        'if sys.argv[-1] == "id -un": print("hpcuser")\n'
    )
    rsync = bin_dir / "rsync"
    rsync.write_text(
        "#!/usr/bin/env python3\nimport os,sys,json,pathlib\n"
        "source=pathlib.Path(sys.argv[-2])\n"
        'data={"args":sys.argv[1:],"files":sorted(str(p.relative_to(source)) '
        'for p in source.rglob("*") if p.is_file())}\n'
        'data["revision"]=(source/".source-revision").read_text().strip()\n'
        'pathlib.Path(os.environ["RSYNC_LOG"]).write_text(json.dumps(data))\n'
    )
    ssh.chmod(0o755)
    rsync.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("REMOTE_")}
    env.update(
        PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        SSH_LOG=str(tmp_path / "ssh.jsonl"),
        RSYNC_LOG=str(tmp_path / "rsync.json"),
    )
    return repo, env


def run_sync(fixture, *args):
    repo, env = fixture
    return subprocess.run(
        ["bash", str(repo / "scripts/sync.sh"), *args], env=env, capture_output=True, text=True
    )


def test_push_exports_only_committed_files_and_identifies_remote_account(sync_checkout):
    repo, env = sync_checkout
    result = run_sync(sync_checkout, "--push")
    assert result.returncode == 0, result.stderr
    data = json.loads(Path(env["RSYNC_LOG"]).read_text())
    assert data["files"] == [".gitignore", ".source-revision", "scripts/sync.sh", "tracked.txt"]
    assert (
        data["revision"]
        == subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    )
    assert data["args"][-1] == "discovery:/home1/hpcuser/mats_latent_safety/"
    assert ".git/" in data["args"]  # protect existing remote Git metadata from deletion
    assert "results/" in data["args"]


def test_push_refuses_uncommitted_code(sync_checkout):
    repo, env = sync_checkout
    (repo / "tracked.txt").write_text("uncommitted change\n")
    result = run_sync(sync_checkout, "--push")
    assert result.returncode == 2
    assert "commit the working tree" in result.stderr
    assert not Path(env["RSYNC_LOG"]).exists()


def test_dry_run_does_not_create_remote_directories(sync_checkout):
    _, env = sync_checkout
    result = run_sync(sync_checkout, "--push", "--dry-run")
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["SSH_LOG"]).read_text().splitlines()]
    assert calls == [["discovery", "id -un"]]
    assert "--dry-run" in json.loads(Path(env["RSYNC_LOG"]).read_text())["args"]


@pytest.mark.parametrize(
    "path", ["/", "/home1/user/../..", "/home1/user/.", "/home1/u/x;echo bad", "/project2/u/x"]
)
def test_rejects_unsafe_or_prohibited_remote_paths(sync_checkout, path):
    _, env = sync_checkout
    env["REMOTE_CODE"] = path
    result = run_sync(sync_checkout, "--push")
    assert result.returncode == 2
    assert "unsupported remote project path" in result.stderr
    assert not Path(env["RSYNC_LOG"]).exists()
