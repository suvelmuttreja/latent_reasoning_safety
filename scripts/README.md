# Scripts index

Every entry point in this directory, grouped by pipeline stage. The one-line
descriptions are the scripts' own docstrings. `slurm/` holds the SLURM
submissions that actually ran on Discovery; the table at the end maps each one
to its job name, entry point, and config so a log in `artifacts/discovery/logs/`
can be traced back to the exact command.

For executable setup and reanalysis commands, see [reproduction](../docs/reproduction.md).
The historical job scripts retain the original cluster paths and run settings.

## Repository verification

- [`verify_repository.py`](verify_repository.py): Check frozen evidence hashes, structured files, documentation links, shell syntax, and numerical reproduction without model access.

## Environment and data setup

- `audit_public_checkpoint.py`: Audit a gated public wrapper state dict before any strict=False load.
- `bootstrap_env.sh`: Reproducible Discovery environment setup. Run on a login node after --push.
- `build_manifests.py`: Build deterministic, content-hashed S1 manifests from pinned source files.
- `capture_hf_cache_metadata.py`: Persist resolved HF cache revisions and byte counts without reading tokens.
- `capture_public_cards.py`: Capture pinned Costco model cards after manual approval lands.
- `check_hf_access.py`: Non-secret HF identity, revision, and one-byte read-access audit.
- `clone_vendor.sh`: Clone/fetch exact ignored upstream references into the current checkout.
- `copy_compact_stage_results.sh`: Copy a finished stage's compact JSON/metrics records from scratch1 to home1.
- `discovery_env.sh`: Shared environment for Discovery jobs. Source; do not execute directly.
- `hf_durability_smoke.py`: Verify private HF checkpoint-repository upload and clean download.
- `inspect_processed_data.py`: Render the first 20 canonical clean-GSM8K records for required inspection.
- `prepare_discovery_data.sh`: Download only public, regenerable source data to scratch1 and verify hashes.
- `sync.sh`: Export committed code without local credentials or Git metadata, or pull compact evidence from scratch1.

## Session S1: smoke tests, validation, preflights, M0 calibration

- `calibrate_think_budget.py`: Calibrate native Qwen thinking length on the frozen 20+12 prompt set.
- `preflight_4b_coconut.py`: Time the self-contained standard 4B Coconut stage-1 fallback path.
- `preflight_4b_cot.py`: Measure access-independent 4B explicit-CoT training memory and throughput.
- `smoke_0_6b.py`: One-step Qwen3-0.6B standard-Coconut save/reload/generation smoke.
- `smoke_strongreject.py`: Load and score five cached examples with the registered local evaluator.
- `validate_0_6b.py`: Small self-contained skip0 validation while public 4B approval is pending.

## Training the two matched branches

- `claim_stage_race.py`: Atomically select one queued hardware path and cancel its rivals.
- `repair_4b_stage_finalization.py`: Finalize and durably upload a completed stage whose metadata step failed.
- `train_4b_skip0_stage.py`: Train one resumable stage of either frozen matched 4B branch.

## Capability and coherence evaluation

- `evaluate_coconut_stage_trajectory.py`: Evaluate Coconut capability/coherence at K=0 and the stage endpoint K.
- `evaluate_native_gsm8k_control.py`: Evaluate M0 or CoT-u3 on GSM8K-200 with the same native-chat harness.
- `export_blind_coherence.py`: Export unlabeled coherence records with a reproducible shuffle key.
- `export_coherence_harness_blind.py`: Combine stage and M0 coherence outputs into one condition-blind packet.
- `gate_4b_coco_u1.py`: Evaluate the self-trained 4B stage-1 endpoint for in-line method validity.
- `render_blind_review_html.py`: Render a condition-blind JSONL packet as a self-contained scoring page.
- `run_m0_coherence_control.py`: Generate an exact-M0 native-chat control for the stage-1 coherence harness.
- `run_stage_coherence_adequate_cap.py`: Run only stage-1 K=0/K=2 coherence at the frozen explicit-thinking cap.

## Safety generation and response-cap calibration

- `analyze_calibration_nontermination.py`: Compute content-blind structural diagnostics for safety calibration rows.
- `calibrate_coconut_safety_cap.py`: Calibrate a branch safety-response cap from token lengths without judging.
- `generate_m0_safety_baseline.py`: Generate the frozen full-60 M0 safety baseline without loading a judge.
- `generate_official_safety.py`: Generate one fail-closed official safety condition without judging it.
- `run_m0_format_anchor.py`: Generate the paired M0 dual-serialization StrongREJECT anchor.

## StrongREJECT scoring and the preregistered comparison

- `build_dense_safety_trajectory.py`: Build the M0-anchored matched safety trajectory table from frozen evidence.
- `compare_bounded_u1_safety.py`: Bootstrap the paired CoT-versus-Coconut u1 partially identified contrast.
- `compare_cot_u1_seeds.py`: Report paired prompt uncertainty and observed spread for two CoT-u1 seeds.
- `compare_official_safety_scores.py`: Compute the preregistered paired-prompt endpoint comparison.
- `plot_dense_safety_trajectory.py`: Render point estimates and partial-identification safety bounds.
- `score_bounded_strongreject.py`: Score terminating official responses and bound nonterminating outcomes.
- `score_cached_strongreject.py`: Score cached final answers without regenerating model outputs.
- `score_m0_format_anchor.py`: Score and summarize the paired M0 dual-serialization anchor.

## Human audits

- `audit_judge_visibility.py`: Measure the configured judge's input visibility without loading model weights.
- `build_judge_human_table.py`: Merge the sealed blind human audit with continuous judge scores.
- `export_official_safety_audit.py`: Export the frozen endpoint audit subset without condition labels or judge scores.
- `render_safety_audit_html.py`: Render the condition-blind official endpoint human audit as a local page.

## Post-hoc analyses and independent audit

- `analyze_readout_periodicity.py`: Quantify period-two structure in the post-hoc latent token readout.
- `audit_existing_analyses.py`: Recompute analysis checks without altering frozen generations or results.
- `compare_layerwise_weight_updates.py`: Describe layer-wise endpoint weight updates relative to the pinned M0.
- `plot_layerwise_weight_updates.py`: Plot the post-hoc descriptive endpoint weight-update comparison.
- `readout_coconut_latent_tokens.py`: Decode native token logits at each recurrent Coconut latent step.

## SLURM submissions (`slurm/`)

| Script | Job name | Entry point | Config |
| --- | --- | --- | --- |
| `analyze_coconut_gsm8k_nontermination.sbatch` | `mats-coco-gsm-nonterm` | `analyze_calibration_nontermination.py` | `coconut_gsm8k_nontermination_diagnostic.yaml` |
| `analyze_coconut_safety_nontermination.sbatch` | `mats-coco-nonterm` | `analyze_calibration_nontermination.py` | `coconut_safety_nontermination_diagnostic.yaml` |
| `calibrate_coconut_safety_cap.sbatch` | `mats-coco-safety-cap` | `calibrate_coconut_safety_cap.py` | (inline arguments) |
| `calibrate_coconut_safety_cap_extension1.sbatch` | `mats-coco-cap-ext1` | `calibrate_coconut_safety_cap.py` | `coconut_safety_cap_calibration_extension1.yaml` |
| `calibrate_coconut_safety_cap_stage3.sbatch` | `mats-coco-s3-cap` | `calibrate_coconut_safety_cap.py` | `coconut_safety_cap_calibration_stage3.yaml` |
| `calibrate_coconut_safety_cap_stage3_extension1.sbatch` | `mats-coco-s3-cap-ext1` | `calibrate_coconut_safety_cap.py` | `coconut_safety_cap_calibration_stage3_extension1.yaml` |
| `calibrate_dense_safety_condition.sbatch` | `mats-dense-safety-cap` | `calibrate_coconut_safety_cap.py` | `dense_coco_u2_safety_cap.yaml`, `dense_cot_u1_safety_cap.yaml`, `dense_cot_u2_safety_cap.yaml`, `seed43_cot_u1_safety_cap.yaml` |
| `calibrate_explicit_safety_cap_stage3.sbatch` | `mats-cot-s3-cap` | `calibrate_coconut_safety_cap.py` | `explicit_cot_safety_cap_calibration_stage3.yaml` |
| `calibrate_m0.sbatch` | `mats-s1-m0-cal` | `calibrate_think_budget.py` | (inline arguments) |
| `coherence_adequate_cap.sbatch` | `mats-coh-adequate-cap` | `export_coherence_harness_blind.py`, `run_m0_coherence_control.py`, `run_stage_coherence_adequate_cap.py` | `gate_4b_coherence_adequate_cap.yaml` |
| `compare_layerwise_weight_updates.sbatch` | `mats-layer-weight-diff` | `compare_layerwise_weight_updates.py` | (inline arguments) |
| `compare_official_safety.sbatch` | `mats-compare-official-safety` | `compare_official_safety_scores.py` | `official_safety_scoring.yaml` |
| `copy_4b_coconut_stage3.sbatch` | `mats-copy-coco-s3` | (shell only) | (inline arguments) |
| `evaluate_coconut_stage2_trajectory.sbatch` | `mats-coco-s2-trajectory` | `evaluate_coconut_stage_trajectory.py` | (inline arguments) |
| `evaluate_coconut_stage3_trajectory.sbatch` | `mats-coco-s3-trajectory` | `evaluate_coconut_stage_trajectory.py` | (inline arguments) |
| `evaluate_native_gsm8k_control.sbatch` | `mats-native-gsm-control` | `evaluate_native_gsm8k_control.py` | `native_gsm8k_endpoint_controls.yaml` |
| `evaluate_native_gsm8k_early_cot.sbatch` | `mats-early-cot-gsm` | `evaluate_native_gsm8k_control.py` | `native_gsm8k_early_cot_controls.yaml` |
| `export_official_safety_audit.sbatch` | `mats-export-official-audit` | `export_official_safety_audit.py`, `render_safety_audit_html.py` | `official_safety_human_audit.yaml` |
| `finalize_bounded_safety_trajectory.sbatch` | `mats-finalize-safety-fig` | `build_dense_safety_trajectory.py`, `plot_dense_safety_trajectory.py` | (inline arguments) |
| `gate_4b_coco_u1.sbatch` | `mats-gate-4b-coco-u1` | `export_blind_coherence.py`, `gate_4b_coco_u1.py` | (inline arguments) |
| `gate_4b_coco_u1_v2.sbatch` | `mats-gate-4b-coco-u1-v2` | `export_blind_coherence.py`, `gate_4b_coco_u1.py` | (inline arguments) |
| `generate_dense_official_safety_condition.sbatch` | `mats-dense-official-safety` | `generate_official_safety.py` | `dense_official_safety_trajectory.yaml` |
| `generate_m0_full60_safety.sbatch` | `mats-m0-full60-safety` | `generate_m0_safety_baseline.py` | `m0_full60_safety.yaml` |
| `generate_official_coco_u3_k6.sbatch` | `mats-official-coco-u3-k6` | `generate_official_safety.py` | `official_safety_endpoints.yaml` |
| `generate_official_coco_u3_k6_a40_rescue.sbatch` | `mats-official-coco-a40-rescue` | `generate_official_safety.py` | `official_safety_endpoints.yaml` |
| `generate_official_cot_u3.sbatch` | `mats-official-cot-u3` | `generate_official_safety.py` | `official_safety_endpoints.yaml` |
| `generate_seed43_cot_u1_official_safety.sbatch` | `mats-seed43-official-u1` | `generate_official_safety.py` | `seed43_cot_u1_official_safety.yaml` |
| `generate_seed43_cot_u1_official_safety_chunk1.sbatch` | `mats-seed43-official-u1a` | `generate_official_safety.py` | `seed43_cot_u1_official_safety.yaml` |
| `m0_coherence_control.sbatch` | `mats-m0-coh-control` | `claim_stage_race.py`, `export_coherence_harness_blind.py`, `run_m0_coherence_control.py` | `m0_coherence_hardware_race.yaml` |
| `m0_format_anchor.sbatch` | `mats-m0-format-anchor` | `run_m0_format_anchor.py` | (inline arguments) |
| `preflight_4b_coconut.sbatch` | `mats-s1-4b-coco` | `preflight_4b_coconut.py` | (inline arguments) |
| `preflight_4b_coconut_a40.sbatch` | `mats-s1-4b-coco-a40` | `preflight_4b_coconut.py` | `preflight_4b_coconut_a40.yaml` |
| `preflight_4b_cot.sbatch` | `mats-s1-4b-cot` | `preflight_4b_cot.py` | (inline arguments) |
| `readout_coconut_latent_tokens.sbatch` | `mats-latent-readout` | `readout_coconut_latent_tokens.py` | `posthoc_token_mode_readout.yaml` |
| `regenerate_coconut_stage3_capability.sbatch` | `mats-coco-s3-gsm1k` | `evaluate_coconut_stage_trajectory.py` | `coconut_stage3_capability_regeneration.yaml` |
| `repair_4b_coco_stage1.sbatch` | `mats-repair-coco-s1` | `repair_4b_stage_finalization.py` | `fallback_4b_skip0.yaml` |
| `repair_4b_cot_stage1.sbatch` | `mats-repair-cot-s1` | `repair_4b_stage_finalization.py` | `matched_4b_cot.yaml` |
| `score_bounded_dense_safety_condition.sbatch` | `mats-score-bounded-safety` | `score_bounded_strongreject.py` | `dense_safety_scoring.yaml`, `evaluation.yaml` |
| `score_dense_safety_condition.sbatch` | `mats-score-dense-safety` | `score_cached_strongreject.py` | `dense_safety_scoring.yaml`, `evaluation.yaml` |
| `score_m0_audit.sbatch` | `mats-s1-m0-score` | `score_cached_strongreject.py` | (inline arguments) |
| `score_m0_format_anchor.sbatch` | `mats-score-format-anchor` | `score_m0_format_anchor.py` | (inline arguments) |
| `score_official_coco_u3_k6.sbatch` | `mats-score-official-coco-u3-k6` | `score_cached_strongreject.py` | `evaluation.yaml`, `official_safety_scoring.yaml` |
| `score_official_cot_u3.sbatch` | `mats-score-official-cot-u3` | `score_cached_strongreject.py` | `evaluation.yaml`, `official_safety_scoring.yaml` |
| `score_official_m0.sbatch` | `mats-score-official-m0` | `score_cached_strongreject.py` | `evaluation.yaml`, `m0_safety_scoring.yaml` |
| `score_seed43_cot_u1.sbatch` | `mats-score-seed43-u1` | `score_cached_strongreject.py` | `evaluation.yaml`, `seed43_cot_u1_safety_scoring.yaml` |
| `smoke_0_6b.sbatch` | `mats-s1-coco-smoke` | `smoke_0_6b.py` | (inline arguments) |
| `strongreject_smoke.sbatch` | `mats-s1-sr-smoke` | `smoke_strongreject.py` | (inline arguments) |
| `train_4b_cot_seed43_stage1.sbatch` | `mats-cot-seed43-u1` | `train_4b_skip0_stage.py` | `matched_4b_cot_seed43_u1.yaml` |
| `train_4b_cot_stage1.sbatch` | `mats-matched-4b-cot-s1` | `train_4b_skip0_stage.py` | `matched_4b_cot.yaml` |
| `train_4b_cot_stage1_a40_early.sbatch` | `mats-matched-4b-cot-s1-a40` | `train_4b_skip0_stage.py` | `matched_4b_cot.yaml` |
| `train_4b_cot_stage2.sbatch` | `mats-matched-4b-cot-s2` | `train_4b_skip0_stage.py` | `matched_4b_cot.yaml` |
| `train_4b_cot_stage3.sbatch` | `mats-matched-4b-cot-s3` | `train_4b_skip0_stage.py` | `matched_4b_cot.yaml` |
| `train_4b_skip0_stage1.sbatch` | `mats-fallback-4b-coco-s1` | `claim_stage_race.py`, `train_4b_skip0_stage.py` | (inline arguments) |
| `train_4b_skip0_stage1_a40.sbatch` | `mats-fallback-4b-coco-s1-a40` | `claim_stage_race.py`, `train_4b_skip0_stage.py` | (inline arguments) |
| `train_4b_skip0_stage2.sbatch` | `mats-matched-4b-coco-s2` | `train_4b_skip0_stage.py` | (inline arguments) |
| `train_4b_skip0_stage3.sbatch` | `mats-matched-4b-coco-s3` | `train_4b_skip0_stage.py` | (inline arguments) |
| `validate_0_6b.sbatch` | `mats-s1-coco-validate` | `validate_0_6b.py` | (inline arguments) |
| `validate_0_6b_reduced.sbatch` | `mats-s1-coco06-reduced` | `validate_0_6b.py` | `validation_0_6b_reduced.yaml` |
