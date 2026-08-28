# Kickoff Plan — What Makes Benign Reasoning Post-Training Erode Safety?

## Purpose and authority

This is the single execution handoff for Session S1. Read it together with [`MATS_execution_protocol_v6_3.md`](MATS_execution_protocol_v6_3.md), which remains the authoritative research protocol. This document fixes implementation defaults, access gates, and conceptual qualifications that must be settled before results are observed.

The motivating question is:

> **Is safety erosion from benign reasoning post-training fundamentally a property of what the model learns, or does it depend on reasoning being externalized and trained through language?**

The experiment asks whether self-jailbreaking-style drift persists, weakens, or intensifies when matched benign GSM8K reasoning post-training changes from explicit rationale supervision to a Coconut latentization regime. It does not isolate hidden-state recurrence, verbalization, token supervision, or inference compute individually.

## Scientific contract

Pre-register these qualitative accounts before viewing safety outcomes:

- **Learning-content account:** CoT and Coconut drift similarly when capability remains intact and the latents are load-bearing, suggesting that behavioral erosion primarily tracks what benign reasoning SFT teaches rather than whether reasoning is externalized.
- **Externalized-language account:** CoT drifts more, consistent with verbalized rationalization or token-level rationale supervision contributing. This does not distinguish language externalization from the associated supervision pattern.
- **Latentization-regime account:** Coconut drifts more, establishing an end-to-end regime-specific effect. It does not show that hidden-state recurrence alone is unsafe.
- **Organism failure:** neither branch drifts despite headroom/coherence/capability, or M0 lacks refusal headroom before training. This is not evidence of substrate independence.

Use these estimands in implementation outputs:

- `S`: primary StrongREJECT score on the **parsed final answer only**, identically across regimes.
- `delta_regime`: Coconut drift minus CoT drift, with the protocol's `delta_substrate` retained as a documented compatibility alias. This is a **matched end-to-end latentization-regime effect** because training is matched but each branch uses its natural inference procedure.
- `delta_latent_depth`: fixed-weight Coconut `K=Kmax` minus `K=0`. This identifies the intervention on latent depth, not latent semantic content alone; it also changes recurrent steps, positions, and sequence length, and `K=0` may be OOD.
- Full `<think>…answer` scoring for M0/CoT is a secondary explicit-branch diagnostic, never a replacement for `S` or a symmetric cross-regime endpoint.

Do not claim that Coconut removes the textual rationale channel until serialized targets and raw outputs establish that fact. Until then say that it progressively reduces or replaces explicit rationale supervision. A null `delta_regime` is reported as "no measurable difference at this assay's resolution," not equivalence, unless an equivalence margin is frozen before branch results.

**Frozen equivalence margin (2026-08-27, before any branch results):** the margin is the absolute M0 serialization-anchor effect measured in protocol Phase 2. Declare "similar drift" only if the paired-prompt bootstrap CI for `delta_regime` lies entirely within ± that margin; if the measured anchor is ≈0, no equivalence language is available and nulls are reported only as "no measurable difference at this assay's resolution."

## Verified starting state and hard blockers

- The local directory is **not yet a Git repository**. It contains only the protocol, this handoff, three initial cluster scripts, and submission material; no experimental code or results exist.
- Discovery has the scratch project skeleton but no MATS jobs in flight as of the last check. Re-check before submitting anything.
- The Costco666 `full-k6`, `skip0-k6`, and intermediate checkpoints are manual-gated Hugging Face repositories. The API confirmed `gated: "manual"` for `skip0-k6` on 2026-08-27, so approval latency is external and unbounded.
- The checkpoint cards reference Qwen-specific `coconut/run_coconut.py` and `materialize_checkpoint_for_vllm`; the exact training source has not been identified in `facebookresearch/coconut` or `wassname/coconut`. Treat it as unknown, not as "probably wassname."
- `strongreject_finetuned` requires an HF token with access to its gated Gemma dependency. The evaluator must pass a five-example cached smoke test before official scoring.
- Discovery has A100-40GB and A100-80GB variants under the same GRES. Any 80GB preflight must explicitly request `--constraint=a100-80gb`; do not infer that full bf16 AdamW fits until the memory test succeeds.
- Local free space was approximately 56 GiB on 2026-08-27, too tight to assume all 48–56 GiB of unique stage checkpoints can safely remain local. A durable checkpoint destination must be verified before matched training.

## Step 0 — user access actions

Do these immediately because they can proceed in parallel with scaffolding:

1. Request access to all seven Costco666 Coconut repositories: `{full,skip0}-k6`, `{full,skip0}-k12`, and the available `full-k12`, `skip0-k6`, and `skip0-k12` intermediate repositories. Use k6 for the registered experiment; k12 is diagnostic/salvage only and must never be selected based on its safety direction.
2. Accept the relevant Gemma license and authenticate Hugging Face on Discovery. Store credentials only in the user's HF credential store or another non-committed secret location; never put tokens in configs, commands captured in logs, or the repository.
3. **Maintainer contact: declined by user decision (2026-08-27).** Do not open discussions with or otherwise contact the Costco666 maintainer or any external party without explicit user approval. The unavailable training source is not a blocker: once weights are approved, follow the wrapper-reconstruction path in §4.1 (key audit against the open implementations).
4. Create or authorize a private durable Hugging Face repository for unique trained stage checkpoints. Verify upload and download with a small test artifact before matched training. Public checkpoints remain re-downloadable scratch cache and need no second copy.

While approval is pending, proceed with all local scaffolding, public-source inspection, parsers, data manifests, and the 0.6B smoke path. Do not reconstruct or train 4B Coconut speculatively before completing the checkpoint-key comparison against the open implementations (with maintainer contact declined, the §4.1 key-audit reconstruction is the only source-recovery path).

### Approval-latency fallback amendment — 2026-08-27

User-authorized before any matched-branch safety result: the gated public
checkpoints are advance method-validation and triage anchors, not a logical
dependency of the matched experiment, which trains both branches from M0.
Pinned full-k6 and skip0-k6 cards are readable without weight access and
document the same base, 7,473-example clean GSM8K set, `c_thought=2`,
`max_latent_stage=3`, two epochs/stage, micro-batch 2 × accumulation 16,
LR `8e-6`, AdamW/wd `0.01`, and bf16.

The user activated this fallback immediately on 2026-08-27 because the
remaining calendar risk outweighs the value of waiting for an unbounded
approval. All weight probes still returned HTTP 403 at activation. Begin the
access-independent work now rather than waiting for the originally proposed
48–72-hour checkpoint:

1. Pin the self-contained standard wrapper adapted from
   `facebookresearch/coconut`, with the wassname fork used only as an
   engineering reference. Preserve the existing forbidden-change list.
2. Begin skip0 stage 1 from exact M0 using the documented public recipe and a
   frozen data order. Treat the first saved slice as the beginning of the real
   branch so it is resumable rather than disposable.
3. Move Gate -1 in-line: at `coco_u1`, decide only from strict save/reload,
   coherence, GSM8K preservation, and K-dependence. Do not use its safety
   direction for selection.
4. If `coco_u1` passes, continue stages 2–3 and train the matched CoT branch
   from exact M0 at identical cumulative update counts. If direct skip0 fails,
   use the registered shared-Mpre Phase 3B fallback; if neither path produces
   load-bearing latents, stop the matched experiment.
5. If Costco access arrives later, compare the self-trained endpoint with the
   public checkpoint as a robustness footnote. The self-trained run is not
   retrospectively relabeled a public reproduction.

If access arrives before the in-line gate is decided, restore the public
checkpoint triage/key-audit path in parallel. Do not discard or relabel the
self-trained branch: it remains the matched branch if it obeys the frozen
configuration, while the public checkpoint supplies the originally intended
advance validation and comparison anchor.

Launching the CoT branch before the in-line method gate is a deliberate
calendar-risk trade, not the default. It requires an explicit decision-log
entry because a Phase 3B outcome could require retraining that branch.

## S1 execution plan

### 1. Durability and reproducibility

1. Initialize Git locally; make the local repository canonical and commit the protocol, handoff, scripts, and subsequent source/config changes. Start `research_log.md` immediately and log active project minutes.
2. Enforce the storage policy in `AGENTS.md`:
   - `/home1/$USER/mats_latent_safety`: code checkout, configs/manifests, and reproducible `.venv`.
   - `/scratch1/$USER/mats_latent_safety`: HF/uv caches, datasets, temporary generations, logs, and working checkpoints.
   - never use `project2`; never leave a unique result or trained checkpoint only on scratch.
   - check home quota before environment creation; if the environment does not fit, stop for user direction rather than silently moving it to scratch.
3. Replace the current scratch-only bootstrap/sync behavior. Use a pinned `pyproject.toml` plus `uv.lock`; sync code to `home1`, pull results/configs/logs after every session, and upload each unique matched checkpoint to the verified durable repository as it is created.
4. Clone `wassname/coconut`, `facebookresearch/coconut`, `dsbowen/strong_reject`, and `BatsResearch/self-jailbreaking` under ignored `vendor/` directories. Record exact remotes and commit hashes in `configs/pins.json`; keep all project-specific adaptations in tracked project source.
5. Install only the native PyTorch/Coconut/evaluator stack initially. Add `bitsandbytes` only if the memory ladder reaches 8-bit AdamW, and add vLLM/SGLang only if measured native generation is the bottleneck and byte-identical serialization/output parsing is verified.

### 2. Freeze data and evaluation interfaces

1. Commit deterministic manifests containing IDs, source revisions, and hashes for GSM8K-200, the 12 StrongREJECT audit prompts (two/category), and the 10-prompt coherence set. Use seed 42 for any sampling/shuffling and never regenerate a set after viewing model outputs.
2. Add one frozen evaluation config recording generation parameters, seeds, explicit and Coconut answer caps, stop tokens, prompt serialization, answer parser, evaluator revision, bootstrap seed/count, and response fields. Prefer the exact public reproduction settings when recovered; if generation sampling is unspecified, use the official Qwen recommendation (`temperature=0.6`, `top_p=0.95`, `top_k=20`, one sample, seed 42) for every condition.
3. Save per generation: prompt ID/hash, model and code revision, raw serialized input, raw output, parsed thinking, parsed final answer, K, token counts, stop reason, truncation flag, generation config hash, and evaluator payload/score.
4. Unit-test prompt serialization, `</think>` parsing (including missing/multiple/truncated delimiters), GSM8K answer extraction, stage/update accounting, K selection, state-dict key auditing, and final-answer versus full-transcript evaluator payloads.
5. Blind coherence scoring by exporting shuffled outputs without checkpoint/condition labels. Run the coherence set at both `K=0` and `K=Kmax` for the fixed-weight contrast; degraded K=0 coherence makes its safety difference capability-confounded.

### 3. Access-independent smoke and first GPU work

1. Implement the canonical 0.6B smoke path: one training step, NaN/gradient checks, exact stage-end save, reload, K-selectable latent generation, and output-schema validation. It must exercise the same wrapper path intended for 4B, not merely an unrelated fork demo.
2. The first generation-heavy 4B job is M0 think-budget calibration: ~20 frozen GSM8K items plus the 12 audit prompts, starting at 16k new tokens. Parse the final `</think>`, rerun only truncated cases at a higher cap, then freeze a cap with less than 5% incomplete generations and record length percentiles and tokens/sec.
3. Verify `strongreject_finetuned` on five cached refusal/partial/compliance examples. Pin the StrongREJECT commit and resolved judge revision. If technically inaccessible after correct Gemma authorization, freeze one rubric judge/config before any official checkpoint score; never mix evaluators.

### 4. Public checkpoint loading and Gate −1

1. Recover the canonical Qwen wrapper if possible. For every `strict=False` load, persist all missing/unexpected keys and reject any unexplained material weight mismatch. If reconstruction is unavoidable, document every adaptation and require save/reload determinism plus coherent K-selectable output before scientific use.
2. Evaluate M0, public full-k6, and public skip0-k6 on GSM8K-200, StrongREJECT-small, the 12 audit prompts, the coherence set, and response length. Cache generations before judging. Run K=0 versus K=6 GSM8K for both Coconut checkpoints.
3. If approved, use skip0-k6 intermediates for stage-wise capability/K-dependence diagnostics. Treat k12 only as an explicitly labeled diagnostic or salvage condition; never use its safety score for design selection.
4. Record Gate −1 using only method validity:
   - skip0 loads, is coherent, preserves meaningful GSM8K capability, and shows at least partial K-dependence → direct-M0 Phase 3A.
   - skip0 fails but full-k6 passes → Mpre fallback Phase 3B, with claims restricted to post-Mpre modulation.
   - neither is load-bearing/coherent/capable → stop matched training and report the checkpoint audit as lower-novelty salvage.
5. K-dependence is a scientific gate outcome, not a software assertion. Code is considered correct if both K settings execute deterministically with the intended intervention; `K=0 ≈ K=6` is a valid method failure.

### 5. Memory and throughput gate

1. Run the 4B full-sequence memory preflight on one explicitly constrained A100-80GB. Walk the registered ladder only as needed: bf16 → gradient checkpointing → micro-batch 1 plus accumulation → 8-bit AdamW → readily available multi-GPU support → Qwen3-1.7B fallback. Do not introduce QLoRA.
2. Time about 50 representative CoT-SFT steps and 50 Coconut stage-1 steps. Measure native Qwen/CoT tokens/sec and Coconut examples/sec under the frozen caps.
3. Log projected wall time, GPU hours, storage, and dense-tier evaluation time for the full 7,473 examples. Use the same pre-registered reduction for both branches if the full dose is infeasible.
4. Do not launch matched training until access, wrapper, Gate -1, memory, throughput, storage-backup, and M0-headroom gates all pass.

The selected micro-batch/accumulation pair is global to the matched experiment:
both branches and all stages use it. Effective batch 32 alone is insufficient
for matching because bf16 accumulation order and variable-length token
weighting can differ. Normalize each update by the total shifted supervised
token count across all micro-batches. If `1 x 32` wins the hardware gate, log
the deviation from the public `2 x 16` recipe as expectation-preserving and
keep the frozen data order unchanged.

If two hardware paths are queued for the same stage, the first job to start is
the winner and the other job must be cancelled and logged before it starts.
Atomic stage-directory claiming is a backstop, not a substitute for cancelling
the loser.

Training-only projections must be reported separately from the dense
GSM8K/safety/coherence evaluation passes, 5,120-token explicit-thinking caps,
checkpoint serialization/hashing, and durable upload. The current projection
is frozen in `configs/wall_clock_budget.yaml`; replace pending Coconut-generation
and upload estimates with measurements from `coco_u1`.

## Confound controls and result policy

- Report response length at every checkpoint and inspect StrongREJECT score versus length. If branch lengths diverge or judge disagreements cluster on terse/degenerate outputs, foreground the blinded 12-prompt human audit and concrete mismatches.
- Log supervised/non-padding token counts as well as examples and optimizer updates. The supervision difference is part of the regime, not a hidden nuisance that can be corrected away.
- Force `save_only_improve=false` for matched runs and save every exact stage endpoint regardless of validation movement.
- A large `delta_regime` with intact capability/coherence supports a single-run regime result. A small effect or null cannot resolve training-seed uncertainty; prompt-bootstrap intervals cover prompts only.
- If a meaningful regime difference appears and minimum deliverables are complete, prioritize one answer-only/no-CoT control before making a strong externalized-language claim. It tests the simpler reduced-rationale-supervision explanation but still does not isolate verbalization perfectly.
- **Sham-latent (pause-token) control, pre-registered 2026-08-27 as an equally admissible conditional branch:** train from M0 with the identical skip0 stage schedule and textual-supervision removal, but with fixed pause/filler tokens at the latent positions instead of fed-back continuous hidden states. Public latent-reasoning results report pause tokens sometimes matching continuous thoughts; if the sham branch reproduces Coconut's drift, the effect is attributable to curriculum/supervision structure rather than learned latent content. `no_cot` tests removal of rationale supervision; sham-latent tests structure-without-content. Both remain conditional on a real `delta_regime` signal — run at most one unless compute clearly allows both, chosen by which rival explanation the observed result makes most pressing.
- A pre-registered **inference-only ablation menu** (protocol Phase 6 amendment, 2026-08-27) covers the cheap fixed-weights tests reviewers will ask about: latent-content substitution (mean / fixed embedding / norm-matched noise), cross-prompt latent transplant (benign↔harmful), and explicit-branch think-budget truncation. All are eval passes on the frozen sets, labeled ablations with OOD caveats, run only after the primary `K=0` vs `K=Kmax` contrast; none is part of the minimum deliverable and they are later-session (S4) work, not S1.
- Recorded exploratory material, not S1 work: an intermediate token-mode readout across latent depths ("at what depth does safety-relevant behavior become decodable?") is a Phase-7 candidate — decodability claims only, never faithfulness or monitorability. Suppressed-activation projections (`wassname/eliciting_suppressed_knowledge`) are citation/post-sprint follow-up only.
- Similar drift with load-bearing latents supports the learning-content account only behaviorally; it does not demonstrate that Coconut hides the same internal mechanism.
- Public-checkpoint safety is expectation-setting only. Never select skip0/full/k6/k12 because it produces the preferred safety direction.

## S1 completion criteria

S1 ends only when the following are recorded in `research_log.md`, or explicitly marked blocked by external approval:

- canonical Git/lockfile/storage/sync layout works and no unique artifact is scratch-only;
- HF/Gemma access status and maintainer-source request are documented;
- frozen data/evaluation manifests and unit tests pass;
- 0.6B save/reload/latent-generation smoke passes;
- M0 think cap and baseline headroom are measured;
- StrongREJECT five-example smoke passes;
- public checkpoints pass a clean material-key audit and Gate −1 is decided;
- 4B A100-80GB memory and 50-step throughput projections are logged.

Matched-branch training, dense safety trajectories, final K sweeps, figures, and optional controls belong to later sessions under the main protocol. Do not begin them merely to fill passive approval or queue time.
