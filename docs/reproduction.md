# Reproducing and inspecting the project

Run commands from the repository root. The environment requires Python 3.11
and [uv](https://docs.astral.sh/uv/getting-started/installation/). The experiment's
training and evaluation dependencies remain pinned in [uv.lock](../uv.lock).
The optional `report` group adds figure and Word-formatting dependencies.

## Verify the saved experiment without models

The evidence and numerical checks use only the Python standard library:

```bash
python3 scripts/verify_repository.py
```

This checks the complete [evidence inventory](../artifacts/checksums.sha256),
parses saved JSON and JSONL, validates documentation links and shell syntax,
rebuilds the Markdown in a temporary directory, and independently recomputes
[the numerical audit](../writeup/analysis_audit/recomputed_checks.json).
It requires Bash for syntax checks. It does not download weights or alter the
committed results. Use `--skip-recompute` for file checks alone.

To run the library tests and all Python style checks:

```bash
uv sync --locked --group report
uv run --no-sync pytest
uv run --no-sync ruff check src tests scripts writeup
uv run --no-sync ruff format --check src tests scripts writeup
```

The tests use tiny CPU models and fixtures, with no checkpoint access or GPU
required. Installing the full locked environment still downloads PyTorch and
its dependencies. CI runs these checks with Hugging Face downloads disabled
after dependency installation.

## Rebuild figures and the write-up

```bash
uv run --no-sync python writeup/figures/plot_capability_revised.py
uv run --no-sync python writeup/figures/plot_safety_endpoint.py
uv run --no-sync python writeup/figures/plot_training_loss.py
uv run --no-sync python writeup/figures/plot_k0_vs_trained.py
uv run --no-sync python writeup/figures/plot_safety_trajectory_writeup.py
uv run --no-sync python writeup/build_full_writeup.py
```

These commands overwrite the corresponding derived files under `writeup/`.
The numerical reproduction check compares JSON values (floating-point tolerance
of 1e-12 for platform/Python roundoff) and the exact Markdown bytes;
PNG and DOCX bytes can differ across rendering-library and font versions.
The extra descriptive scripts and their inputs are listed in the
[artifact guide](../artifacts/README.md).

To build Word, also install [Pandoc](https://pandoc.org/installing.html):

```bash
pandoc writeup/FULL_WRITEUP.built.md --resource-path=writeup -o /tmp/writeup_raw.docx
uv run --no-sync python writeup/format_full_writeup.py /tmp/writeup_raw.docx /tmp/FULL_WRITEUP.docx
```

The formatted [Word report](../writeup/FULL_WRITEUP.docx) is already included.

## Regenerate source data and manifests

Upstream reference checkouts are optional for inspecting saved results:

```bash
./scripts/clone_vendor.sh
```

The script clones pinned references into this checkout's ignored `vendor/`
directory. `CODE_ROOT` overrides the destination. Sources and licensing are
listed in [third-party notices](../THIRD_PARTY_NOTICES.md).

To download public data and reconstruct manifests into a separate directory:

```bash
export WORK_ROOT="$(mktemp -d)"
./scripts/prepare_discovery_data.sh
uv run --no-sync python scripts/build_manifests.py \
  --gsm-train "$WORK_ROOT/data/sources/gsm8k-train.parquet" \
  --gsm-test "$WORK_ROOT/data/sources/gsm8k-test.parquet" \
  --strongreject "$WORK_ROOT/data/sources/strongreject-small.csv" \
  --output "$WORK_ROOT/manifests"
diff -r manifests "$WORK_ROOT/manifests"
```

The downloader verifies the three frozen source-file hashes. The manifest
builder records the fixed selections and content hashes. Keep the committed
manifests as the canonical inputs for inspecting the completed experiment.

## Training, generation, and scoring on Discovery

The saved records support numerical reanalysis by any reader. Regenerating
model outputs additionally requires GPU resources, the training data, the
pinned base model and judge, and trained checkpoints. The trained checkpoints
are in the private Hugging Face repository named in
[storage.json](../configs/storage.json); they are not anonymously downloadable.
A reader without access must train their own checkpoints before generating
new outputs. This repository does not claim a publicly downloadable checkpoint
release.

[scripts/README.md](../scripts/README.md) maps every historical SLURM submission
to its entry point and configuration. The main training entry point is
[train_4b_skip0_stage.py](../scripts/train_4b_skip0_stage.py), which handles both
matched branches. Read the [configuration guide](../configs/README.md) and
[execution protocol](../project_plan/MATS_execution_protocol_v6_3.md) before
launching a new run. Entry points expose their options with `--help`.

The archived configs and SBATCH directives retain the paths, account-specific
log locations, gates, and job dependencies used in the experiment. They are
provenance records, not portable job templates. Copy and adapt them for a new
run, recording the new config hash; do not overwrite historical results.
In particular, SLURM does not expand shell variables in `#SBATCH` directives.
Override log destinations on the `sbatch` command line when adapting jobs.
GPU types, modules, partitions, time limits, remote checkpoint destination, and
paths inside configs must also match the new environment.

The environment helpers accept `CODE_ROOT`, `WORK_ROOT`, and `HF_TOKEN_PATH`.
On Discovery their defaults are `/home1/$USER/mats_latent_safety`,
`/scratch1/$USER/mats_latent_safety`, and the home1 Hugging Face token cache.
Code and environments belong on quota-limited home1; large working data belongs
on scratch1. Unique checkpoints and results need a durable copy before scratch
can be considered expendable. `project2` is not used.

## Syncing an authorized new run

```bash
./scripts/sync.sh --push --dry-run
./scripts/sync.sh --push
./scripts/sync.sh --pull --dry-run
./scripts/sync.sh --pull
```

The host defaults to the SSH alias `discovery`. The helper queries the remote
username; override `REMOTE_HOST`, `REMOTE_USER`, `REMOTE_CODE`, or `REMOTE_WORK`
as needed. Push requires a clean, committed tree and exports that commit,
excluding Git metadata, local credentials, caches, and saved artifacts. It
writes `.source-revision` so job records identify the exported code even when
the destination contains old Git metadata. Treat that destination as an export:
make and commit edits in the canonical checkout, then sync again.

Push removes obsolete source files within the selected remote code directory
while preserving excluded environment and result directories. Pull copies
compact results and logs without deleting local evidence; model weights,
checkpoint directories, tokenizers, and partial generation dumps are excluded.
Review and commit newly pulled evidence before treating scratch as disposable.

The frozen checksum inventory is intentionally strict: a new or changed run
requires reviewing and updating `artifacts/checksums.sha256` along with the
corresponding evidence. An integrity failure must not be fixed by silently
regenerating the inventory over unexplained changes.
