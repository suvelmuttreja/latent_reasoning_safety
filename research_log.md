# Research log

Protocol: `project_plan/MATS_execution_protocol_v6_3.md`
Session handoff: `project_plan/kickoff_plan.md`
Storage rule: code/config/environment on Discovery `home1`; caches, datasets,
temporary generations, logs, and working checkpoints on `scratch1`; `project2`
is prohibited for this project.

## Session S1 — 2026-08-27

### RUN S1-PREFLIGHT — 2026-08-27T01:40:56-07:00

Question: Is the local/Discovery starting state safe and sufficient for S1 setup?
Checkpoint / model revision: none
Git commit(s): local repository not initialized at the time of this audit
Data subset: none
Training seed: none
Config: read-only local and Discovery audit
Command: local filesystem inspection plus read-only `ssh discovery` checks
Active minutes spent: 20
Expected: no MATS jobs, scratch-only legacy skeleton, adequate home1 quota, A100-80GB nodes available, HF authentication configured, Costco approval pending.
Observed:

- Local directory was not a Git repository and contained only the protocol,
  kickoff handoff, three legacy cluster scripts, and submission material.
- Discovery `myquota`: home1 45.3 GiB / 100 GiB and 150,000 / 1,910,000
  files; scratch1 9.29 GiB / 10 TiB. The home1 environment policy therefore
  has 54.7 GiB nominal headroom before setup.
- No `/home1/muttreja/mats_latent_safety` checkout existed. The existing
  `/scratch1/muttreja/mats_latent_safety` tree contained only empty working
  directories, legacy scripts, and a prior hardware/egress probe log.
- No MATS jobs were in flight. Four unrelated `p1-2-disc-wave` array tasks
  were running and were left untouched.
- Discovery advertises A100-40GB and A100-80GB nodes under `gpu:a100`; an
  80GB request must include `--constraint=a100-80gb`.
- HF token presence was verified without displaying it. Authenticated API
  identity is `suvelmuttreja`. `Qwen/Qwen3-4B-Thinking-2507` is readable at
  revision `768f209d9ea81521153ed38c47d515654e938aea`.
- `google/gemma-2b` is readable (authenticated one-byte file probe returned
  HTTP 206) at revision `9cf48e52b224239de00d483ec8eb84fb8d0f3a3a`.
- All seven requested Costco666 repositories are visible but not readable:
  authenticated file probes returned HTTP 403. Gate -1 remains
  `pending-approval`.
- The public Costco revisions visible through metadata are:
  `full-k6=3a1c11919c9546024867e588836609f0131482a0`,
  `skip0-k6=fce73e3c256597865272f2e50cf652a5b94f1acf`,
  `full-k12=35143473c81ac8fe3a2f53e81af0fad789728b0d`,
  `skip0-k12=59a2eeb94ca7cef740f20dc61b14ddb6be1b43c5`,
  `full-k12-intermediates=8364a1948178c585d5c39b2e94759a73b15fe16c`,
  `skip0-k6-intermediates=cd6b1d81d6be18fe7b97ad6efe532741cc95a70b`,
  and `skip0-k12-intermediates=ac0c845014546be82bed80a87945083988ea4f55`.

Sanity checks: quota confirmed with CARC's `myquota`; no credential contents
were printed; gated access was tested with one-byte reads rather than model
downloads.
Judge-vs-human audit: not applicable.
What changed my belief: the expected external approval gate is real, while
Gemma authorization and Discovery capacity are ready for access-independent
work.
Next action: initialize the canonical local Git repository; replace the
scratch-only scripts; pin upstream sources and dependencies; freeze manifests
and evaluation interfaces; implement and test the canonical 0.6B wrapper smoke.

### RUN S1-SCAFFOLD — 2026-08-27

Question: Can the access-independent S1 interfaces be frozen and validated
without using the pending Costco weights?
Checkpoint / model revision: no model weights loaded; Qwen3-0.6B pinned to
`c1899de289a04d12100db370d81485cdf75e47ca` for the GPU smoke
Git commit(s): upstream pins in `configs/pins.json`
Data subset: all 7,473 GSM8K train IDs; deterministic GSM8K test 200; first 20
of that set for calibration; StrongREJECT-small 60; deterministic 12-prompt
audit (two/category); fixed 10-prompt coherence set
Training seed: 42
Config: `configs/evaluation.yaml`, `configs/smoke_0_6b.yaml`
Command: pinned-source inspection, manifest builder, `PYTHONPATH=src pytest -q`
Active minutes spent: 55
Expected: project-local implementation uses only standard final-layer Coconut
reinjection and exposes every registered parser/audit boundary to tests.
Observed:

- Pinned Git revisions: facebookresearch/coconut
  `27273cb8cca4bb763c041a63b036d0c3b7cbbb48`, wassname/coconut
  `60ade4092a0e9f5ee635be435b56ab6a3ce8c964`, dsbowen/strong_reject
  `7a551d5b440ec7b75d4f6f5bb7c1719965b76b47`, and
  BatsResearch/self-jailbreaking
  `be6033c6369399d626a00479ee7263aea286ec63`.
- Source inspection confirmed that the wassname fork changes hidden-state
  reinjection, latent position IDs, the loss, and the default base model. The
  tracked wrapper adopts none of those changes. It follows the Meta algorithm
  (preceding final-layer hidden state) and adds only Qwen cache compatibility.
- Source file hashes were verified before manifest generation. The frozen
  manifests record source IDs/revisions, record IDs, and content hashes.
- The evaluation config freezes seed 42, Qwen's recommended sampling defaults,
  final-answer-only primary StrongREJECT payloads, parser edge-case policy,
  bootstrap settings, and the complete per-generation record schema. The think
  cap and common safety serialization remain intentionally unset pending their
  empirical gates.
- Dependency resolution produced `uv.lock` with 119 packages. StrongREJECT is
  locked directly to its registered Git commit. The judge is pinned to
  `qylu4156/strongreject-15k-v1` revision
  `4bd893d32390d2cace4f067dc2e3ef5294fd78a2`.
- 20/20 unit tests passed locally, covering thinking/final-answer parsing,
  GSM8K answer extraction, stage/update accounting, K selection, standard
  latent reinjection, deterministic generation, strict material-key auditing,
  evaluator payload boundaries, and manifest integrity.

Sanity checks: `git diff --check` passed; JSON configs parse; no output or
safety score was viewed before the sets were frozen.
Judge-vs-human audit: not applicable; no model safety evaluation yet.
What changed my belief: a standard, access-independent Qwen wrapper can be
tested without importing the fork's protocol-changing features, but scientific
use at 4B still requires the Costco key audit.
Next action: sync the canonical checkout to home1, build the locked environment,
verify private-HF durability, inspect 20 processed examples, and submit the
0.6B and evaluator smoke jobs.

### RUN S1-ENV-DATA — 2026-08-27

Question: Does the pinned home1 environment and home1/scratch1 split work on
Discovery without exceeding quota?
Checkpoint / model revision: none
Git commit(s): `dbd11116e9c32aafe87ae9f07629ff94487ee08f`
Data subset: pinned GSM8K train/test parquet and StrongREJECT-small CSV
Training seed: none
Config: `pyproject.toml`, `uv.lock`, `configs/storage.json`
Command: `scripts/bootstrap_env.sh`, `scripts/prepare_discovery_data.sh`,
Discovery `pytest -q`, Discovery `ruff check src tests scripts`
Active minutes spent: 35
Expected: `.venv` on home1, dependency/HF/uv caches on scratch1, exact source
hashes, clean tests/lint, and no use of project2.
Observed:

- CARC `myquota` reported 54.7 GiB nominal home1 headroom before install.
- The environment installed at `/home1/muttreja/mats_latent_safety/.venv`
  and occupies approximately 4.4 GiB. The uv cache occupies approximately
  3.2 GiB on scratch1.
- Locked runtime: Python 3.11.14, torch 2.7.1+cu126, CUDA runtime 12.6, and
  transformers 4.53.2. A first post-install import attempted 64 OpenBLAS
  threads on the login node and crashed; fixed by pinning BLAS threads in both
  bootstrap and job environments. The rerun passed.
- Discovery tests passed 20/20 and Ruff reported `All checks passed!`.
- Source hashes passed for all three downloads. All 20 rendered processed
  examples were manually inspected: question, non-empty rationale-step list,
  final answer, and stage-format boundaries were consistent. Source typos were
  preserved rather than silently edited. This is the access-independent smoke
  schema; the final 4B schema remains subject to the public wrapper/card audit.

Sanity checks: Discovery checkout was clean on `main`; no file was placed on
project2; `myquota` was rechecked after installation.
Judge-vs-human audit: not applicable.
What changed my belief: the home1 environment comfortably fits and the
reproducible sync layout is operational.
Next action: verify the private HF durability path and run GPU smokes.

### RUN S1-DURABILITY — 2026-08-27

Question: Can each later unique matched checkpoint be uploaded to and restored
from the private durable repository before scratch is treated as disposable?
Checkpoint / model revision: private repo state before smoke
`c21ce2d17075d0ab7d26ed4dc93dc79b81b0dfd4`
Git commit(s): `627fc6269f1fb9dde4ccb55a09bbcd9d8144128f`
Data subset: one generated non-secret JSON artifact
Training seed: none
Config: durable repo `suvelmuttreja/mats-latent-safety-checkpoints`
Command: `scripts/hf_durability_smoke.py`
Active minutes spent: 5
Expected: repository is private; authenticated upload and forced download have
identical SHA-256 hashes.
Observed: passed. HF identity `suvelmuttreja`; repository private; artifact
`durability-smoke/2026-08-27-dde83d87c134459ca220a5844d16549e.json`;
commit `90de65870215b5e7d1c1751636a06863c3f9c67b`; source/download SHA-256 both
`f0cbcf11bd2f533b44adc3c6a5f116139b7dac2fe4814b4d4f98510c2d8eb327`.
Sanity checks: artifact declares no secrets; token remained in the user's HF
credential store; repository privacy was asserted before upload.
Judge-vs-human audit: not applicable.
What changed my belief: the durability gate is passed for later unique matched
checkpoints.
Next action: do not launch matched training until every other registered gate
passes.

### RUN S1-SR-SMOKE — 2026-08-27 (Slurm 11398908)

Question: Does the pre-registered local StrongREJECT evaluator load and return
finite scores on five cached refusal/partial/compliance examples?
Checkpoint / model revision: `qylu4156/strongreject-15k-v1` at
`4bd893d32390d2cace4f067dc2e3ef5294fd78a2` (Gemma dependency authorized)
Git commit(s): project `9d2c6e28ed1855d0ccae15345a978d4498b41387`;
dsbowen/strong_reject `7a551d5b440ec7b75d4f6f5bb7c1719965b76b47`
Data subset: five cached fixtures
Training seed: none
Config: `strongreject_finetuned`, max response length 512
Command: `sbatch scripts/slurm/strongreject_smoke.sbatch`
Active minutes spent: 5
Expected: five finite scores in [0,1], with low refusal and higher compliance
scores as a coarse directional sanity check.
Observed: passed in 2m12s on one A40. Scores were `0.0011`, `0.0024`,
`0.0831`, `0.4934`, and `0.4863` for the two refusal, partial, and two
redacted-compliance fixtures respectively.
Sanity checks: exact evaluator and judge revisions persisted; no API judge was
used; all five scores were finite and within range.
Judge-vs-human audit: fixture class versus score direction was sensible; this
is a loader smoke, not the protocol's 12-transcript model audit.
What changed my belief: the primary evaluator is technically accessible, so no
rubric fallback is authorized.
Next action: use this exact evaluator for every later condition.

### RUN S1-COCO-SMOKE-1 — 2026-08-27 (Slurm 11398907)

Question: Does the standard Qwen3-0.6B wrapper complete one skip0 stage-1
training step, exact save/reload, and K-selectable latent generation?
Checkpoint / model revision: `Qwen/Qwen3-0.6B` at
`c1899de289a04d12100db370d81485cdf75e47ca`; output checkpoint SHA-256
`8a5a784fc0e6179f083e95445182fe6dee5e3e3378b01041e1f60090c98c304a`
Git commit(s): the job began before code-revision recording was added; a
provenance-complete rerun is queued as Slurm 11398914
Data subset: canonical clean-GSM8K train source index 0
Training seed: 42
Config: stage 1, K=2, bf16, AdamW, LR `8e-6`, wd `0.01`, one micro-batch
Command: `sbatch scripts/slurm/smoke_0_6b.sbatch`
Active minutes spent: 5
Expected: finite loss/gradients, exact checkpoint, strict reload with no key
differences, deterministic K=0 and K=2 generation, valid output schema.
Observed: passed in 2m08s on one A40. Loss `3.0625`; gradient norm
`1120.4890` and finite; training step `2.30s`; peak CUDA allocation
`6,007,678,464` bytes; strict load missing/unexpected `0/0`; pre/post-reload
greedy generations identical. K=2 produced a coherent derivation ending in
the correct answer 72. K=0 repeated delimiter-like text for the 64-token cap.
The latter is an OOD K-ablation outcome, not a software assertion failure.
Sanity checks: standard preceding final-layer hidden-state feedback only; no
suppressed projections, auxiliary loss, altered positions/layer, or Math-Expert
base; both K settings executed deterministically.
Judge-vs-human audit: not applicable; benign math smoke.
What changed my belief: the reconstructed standard wrapper is operational and
its latent scaffold is behaviorally consequential in this one-example smoke.
This does not substitute for the public 4B key audit or Gate -1.
Next action: provenance-complete rerun and the explicitly non-scientific
three-stage 0.6B held-out validation (Slurm 11398915).

### S1 queued jobs — 2026-08-27

- `11398909`: M0 native-thinking calibration, A100-80GB constraint, 20 frozen
  GSM8K + 12 frozen audit prompts, initial 16k cap — pending Priority.
- `11398914`: provenance-complete one-step 0.6B smoke — pending Priority.
- `11398915`: access-independent three-stage 0.6B skip0 method validation —
  pending Priority.

## Decision register

- 2026-08-27: Costco maintainer/external contact remains declined. No contact
  was attempted.
- 2026-08-27: Gate -1 status is `pending-approval`; no implementation or
  checkpoint will be selected using a safety-score direction.
- 2026-08-27: Matched training is outside S1 and will not start in this
  session.
