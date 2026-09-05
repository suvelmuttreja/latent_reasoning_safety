# Configuration guide

These files are the frozen inputs and recorded decisions for the completed
experiment. Their status/authorization fields describe the original execution;
they do not imply that a new run has already passed the same gates. Exact bytes
are covered by [the evidence checksums](../artifacts/checksums.sha256).

| Files or family | Role |
| --- | --- |
| `pins.json`, `evaluation.yaml`, `format_anchor.yaml`, `format_examples.txt` | Upstream revisions, judge and decoding policy, and serialization controls. |
| `storage.json`, `storage_policy.yaml`, `wall_clock_budget.yaml`, `deadline_scope_freeze_2026-08-31.yaml` | Original storage/durability rules, compute budget, and frozen scope. |
| `smoke_0_6b.yaml`, `validation_0_6b*.yaml`, `strongreject_smoke_cases.json`, `preflight_4b*.yaml` | Small-model validation, evaluator smoke cases, and GPU feasibility checks. |
| `fallback_4b_skip0.yaml`, `matched_4b_cot.yaml`, `matched_batching.yaml`, `matched_stage23_launch.yaml` | Main Coconut and explicit-CoT branches and matched training schedule. |
| `stage1_*.yaml`, `m0_coherence_hardware_race.yaml` | Historical recovery and mutually exclusive hardware job selection. |
| `gate_4b*.yaml` | Stage-1 validity checks, coherence review, and base-model controls. |
| `coconut_safety_cap*.yaml`, `explicit_cot_safety_cap_calibration_stage3.yaml`, `dense_*_safety_cap.yaml` | Content-blind calibration of response length limits. |
| `coconut_*nontermination_diagnostic.yaml` | Structural analysis of outputs that hit their generation cap. |
| `coconut_stage_trajectory.yaml`, `coconut_stage3_capability_regeneration.yaml`, `native_gsm8k_*controls.yaml` | Capability across stages, latent counts, and native-chat controls. |
| `official_safety_endpoints.yaml`, `official_safety_scoring.yaml`, `official_safety_human_audit.yaml`, `official_coco_a40_deadline_rescue.yaml` | Main endpoint generation, scoring, blind audit, and the historical rescue job. |
| `m0_full60_safety.yaml`, `m0_safety_scoring.yaml` | Base-model safety generation and scoring. |
| `dense_official_safety_trajectory.yaml`, `dense_safety_scoring.yaml`, `dense_safety_trajectory.yaml` | Intermediate-stage safety generation, scoring, and combined trajectory. |
| `matched_4b_cot_seed43_u1.yaml`, `seed43_cot_u1_*.yaml` | Additional CoT stage-1 seed and its calibrated safety evaluation. |
| `posthoc_layerwise_weight_diff.yaml`, `posthoc_token_mode_readout.yaml` | Descriptive weight-change and latent-readout diagnostics. |

The [script index](../scripts/README.md) maps jobs to these files. Several configs
contain original scratch paths, checkpoint IDs, job IDs, or hashes of other
configs. Preserve those references when inspecting the old run. To execute a
new experiment, copy the relevant configuration, adapt its paths and checkpoint
repository, and record its new hash. See [reproduction](../docs/reproduction.md).
