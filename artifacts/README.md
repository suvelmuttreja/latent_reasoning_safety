# Saved experimental evidence

The repository includes compact results and execution logs, approximately
50 MB in total. These are sufficient to inspect the model outputs and reproduce
the reported numerical analyses. Model weights, optimizer states, source-data
caches, and incomplete generation dumps are excluded.

[checksums.sha256](checksums.sha256) inventories every saved result and log,
plus the frozen configs, manifests, and protocol. Run
`python3 scripts/verify_repository.py` from the repository root to verify them.

## Where to start

| Results directory | Contents and interpretation |
| --- | --- |
| [official_safety](discovery/results/official_safety) | M0, CoT-u3, and Coconut-u3 K=6 generations; official scores; paired endpoint comparison; sealed human audit and judge/human table. The primary endpoint safety evidence. |
| [native_gsm8k_controls](discovery/results/native_gsm8k_controls) | Native-chat base and CoT capability controls, including the early-stage CoT checks. |
| [fallback_4b_skip0](discovery/results/fallback_4b_skip0) | Self-trained Coconut branch: training receipts, gates, cap calibration, stage capability/coherence, and nontermination diagnostics. “Fallback” is its historical run name. |
| [matched_4b_cot](discovery/results/matched_4b_cot) | Matched explicit-CoT branch training metadata and response-cap calibration. |
| [dense_safety](discovery/results/dense_safety) | Intermediate-stage safety generations, calibrations, scores, and the combined trajectory with missingness bounds. |
| [matched_4b_cot_seed43](discovery/results/matched_4b_cot_seed43) | Additional CoT stage-1 seed and associated safety evidence. |
| [m0_format_anchor](discovery/results/m0_format_anchor) | Base-model serialization control used to inspect format effects. |
| [posthoc_layerwise_weight_diff](discovery/results/posthoc_layerwise_weight_diff) | Descriptive endpoint weight changes by layer; not a causal localization experiment. |
| [posthoc_token_mode_readout](discovery/results/posthoc_token_mode_readout) | Token readout of latent states and period-two diagnostics; readouts are not a translation of hidden reasoning. |
| [s1](discovery/results/s1) | Initial validation, preprocessing, model/evaluator checks, and access/durability evidence. |

[discovery/logs](discovery/logs) contains historical SLURM stdout and stderr.
Failed attempts and retries are retained where they explain execution choices;
an error log does not by itself invalidate a later successful run. Use the
[script/job mapping](../scripts/README.md) and
[research log](../research_log.md) to trace a particular job.

## Manual grading interfaces

The condition-blind human reviews were performed with self-contained HTML
interfaces generated from the frozen review packets:

- [Stage-1 coherence review](discovery/results/fallback_4b_skip0/gate_stage1_v2_coherence_adequate_cap/coherence_adequate_cap_review.html)
- [Stage-2 coherence review](discovery/results/fallback_4b_skip0/trajectory_stage2/coherence_blind_review.html)
- [Stage-3 coherence review](discovery/results/fallback_4b_skip0/trajectory_stage3/coherence_blind_review.html)
- [Endpoint safety audit](discovery/results/official_safety/human_audit/audit_blind_review.html)

Each page hides condition labels and automatic scores while presenting the full
responses for manual grading. The exported labels, blind keys, and summaries
are stored in the same result directories. Download and open the HTML locally;
GitHub displays repository HTML as source rather than running it.

## Reading a record

Generation JSONL files generally identify the prompt, raw serialized input,
raw model output, parsed reasoning/final answer, token counts, stopping reason,
model/config/code revisions, and run condition. Schemas differ between early
validation and later official evaluation; consult the generating script before
combining records. `length` means generation exhausted its cap. A parsed answer
at that cutoff is not automatically a completed outcome.

Training metadata and durability receipts link a stage to its code/config and
private checkpoint revision. Score files record the evaluator and the payload
it received. The write-up's
[claim-to-artifact map](../writeup/claims_and_numbers.md) identifies the exact
files supporting each claim. The independent
[analysis review](../writeup/analysis_audit/REVIEW.md) explains corrections and
limits, including censored capability and judge visibility.

The files contain verbatim safety-evaluation prompts and model responses,
including harmful or offensive text. They are experimental stimuli and
observations. Upstream prompts retain their source provenance; see
[third-party notices](../THIRD_PARTY_NOTICES.md).

## Derived presentation files

[writeup/figures](../writeup/figures) contains plotting scripts and figures,
plus language-switch and output-anatomy summaries. The audit's corrected
capability plot lives in [writeup/analysis_audit](../writeup/analysis_audit).
[Reproduction instructions](../docs/reproduction.md) distinguish reanalysis
of these saved files from GPU regeneration requiring checkpoint access.
