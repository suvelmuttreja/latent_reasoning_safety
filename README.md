# Safety drift under explicit and latent reasoning fine-tuning

Fine-tuning an aligned model on benign reasoning data can weaken its refusals.
I asked whether that changes when written chain-of-thought is progressively
replaced by [Coconut](https://arxiv.org/abs/2412.06769)-style continuous latent
states.

Starting from `Qwen/Qwen3-4B-Thinking-2507`, I trained explicit-CoT and Coconut
branches on the same 7,473 GSM8K examples, with matched data order, seed,
optimizer, batch size, and update count. The intended comparison required both
branches to retain similar capability. That is where the experiment failed.

## What happened

1. **My Coconut branch lost most of its math ability.** Explicit CoT ended at
   92% on GSM8K-200. My implementation of the public skip-stage-0 Coconut
   configuration fell from 65.5% to 46.0% to 31.0% across its three stages.
   Without an accessible reference checkpoint, I could not distinguish an
   implementation problem from a recipe mismatch.
2. **The latent states mattered, but did not rescue capability.** At the final
   weights, removing the latents raised the GSM8K range to 48–58%, but 20/200
   generations hit the token cap and 19 entered repeating cycles. With all six
   trained latent positions, every generation terminated but accuracy was 31%.
3. **The safety result was inconclusive.** Mean StrongREJECT scores were 0.093
   for CoT and 0.105 for Coconut, versus 0.088 for the base model. The paired
   Coconut-minus-CoT difference was +0.012 with a 95% prompt-bootstrap interval
   of [-0.023, +0.046]. It established neither a difference nor equivalence,
   and the 61-point capability gap confounded the intended interpretation.

![GSM8K capability across the matched branches](writeup/figures/fig1_capability.png)

The central lesson was methodological: capability preservation should have
been a hard gate before continuing to the downstream safety comparison.

## Read the project

- [Full report](writeup/FULL_WRITEUP.built.md) and
  [formatted Word version](writeup/FULL_WRITEUP.docx)
- [Claim-to-evidence map](writeup/claims_and_numbers.md)
- [Independent analysis audit](writeup/analysis_audit/REVIEW.md)
- [Seeded qualitative examples](writeup/random_examples.md)
- [Experimental artifacts](artifacts/README.md)
- [Condition-blind manual grading interfaces](#manual-grading-interfaces)

## Manual grading interfaces

I used four self-contained HTML interfaces to grade model responses without
seeing condition labels or automatic scores:

- [Stage-1 coherence review](artifacts/discovery/results/fallback_4b_skip0/gate_stage1_v2_coherence_adequate_cap/coherence_adequate_cap_review.html)
- [Stage-2 coherence review](artifacts/discovery/results/fallback_4b_skip0/trajectory_stage2/coherence_blind_review.html)
- [Stage-3 coherence review](artifacts/discovery/results/fallback_4b_skip0/trajectory_stage3/coherence_blind_review.html)
- [Endpoint safety audit](artifacts/discovery/results/official_safety/human_audit/audit_blind_review.html)

The pages contain the shuffled review packets and the controls used to record
human labels. The sealed labels, condition keys, and audit summaries are stored
beside them. GitHub does not execute repository HTML, so download a page and
open it locally to use the interface.

## Repository map

| Path | Contents |
| --- | --- |
| [src/mats_latent_safety](src/mats_latent_safety) | Coconut training and inference, batching, evaluation, and record helpers |
| [scripts](scripts/README.md) | Training, generation, scoring, analysis, and SLURM entry points |
| [configs](configs/README.md) | Frozen experiment and evaluation configurations |
| [manifests](manifests) | Fixed GSM8K, StrongREJECT, and coherence examples |
| [artifacts](artifacts/README.md) | Saved generations, scores, metrics, audits, and logs |
| [project_plan](project_plan) | Preregistered execution protocol |
| [writeup](writeup) | Report source, figures, examples, and analysis audit |
| [tests](tests) | CPU unit and regression tests |

## Running the repository

The project uses Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --group report
uv run --no-sync pytest
```

To check the saved results and rebuild the numerical analysis without model
weights or external services:

```bash
python3 scripts/verify_repository.py
```

See [reproduction.md](docs/reproduction.md) for figure generation, source-data
reconstruction, checkpoint requirements, and adapting the historical Discovery
jobs. The trained checkpoints are not publicly downloadable; the saved outputs
are sufficient to inspect the analyses reported here.

Safety artifacts contain verbatim harmful or offensive benchmark prompts and
model responses.

## License and citation

Original code and writing are released under the [MIT License](LICENSE).
Benchmark data and upstream-derived code retain the notices in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Citation metadata is provided
in [CITATION.cff](CITATION.cff).
