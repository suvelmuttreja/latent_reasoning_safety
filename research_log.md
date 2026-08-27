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

## Decision register

- 2026-08-27: Costco maintainer/external contact remains declined. No contact
  was attempted.
- 2026-08-27: Gate -1 status is `pending-approval`; no implementation or
  checkpoint will be selected using a safety-score direction.
- 2026-08-27: Matched training is outside S1 and will not start in this
  session.

