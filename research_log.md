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

## Decision register

- 2026-08-27: Costco maintainer/external contact remains declined. No contact
  was attempted.
- 2026-08-27: Gate -1 status is `pending-approval`; no implementation or
  checkpoint will be selected using a safety-score direction.
- 2026-08-27: Matched training is outside S1 and will not start in this
  session.
