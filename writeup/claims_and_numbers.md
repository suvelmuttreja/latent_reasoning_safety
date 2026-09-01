# Claims, numbers, and write-up source of truth

Evidence cutoff: committed artifacts available on 2026-09-01.

This is the application's canonical **interpretation layer**: the numbers worth
using, the strongest defensible language for each claim, and the planned shape
of the write-up. It is deliberately not a chronological record.

The authority order is:

1. committed result artifacts for numerical values and provenance;
2. frozen protocols/configs for design and analysis commitments;
3. this file for claim wording and synthesis;
4. derivative drafts, captions, slides, and status messages.

Job submissions, queue states, checkpoint-copy operations, failures, repairs,
and dated decisions belong in `research_log.md`, not here. A pending analysis
appears here only when its eventual result could change a claim.

## How to use this file

- Copy the approved wording when a claim is central. Weakening it is allowed;
  strengthening it requires new committed evidence and, where applicable, a
  protocol amendment.
- Keep point estimates, uncertainty, missingness, capability, coherence, and
  the relevant analysis level together.
- “95% CI” below means paired-prompt bootstrap uncertainty unless explicitly
  stated otherwise. It is not across-seed uncertainty.
- Exploratory analyses must be labeled post hoc and cannot repair or replace a
  registered primary result.
- The provenance paths below are part of the claim. If a path and this file
  disagree, the artifact wins and this file must be corrected.

## One-paragraph result summary

Matched benign GSM8K post-training produced sharply different capability
outcomes in an explicit-CoT branch and our skip-stage-0 Coconut reproduction,
despite an inconclusive endpoint safety contrast. Explicit CoT reached 92.0%
GSM8K accuracy, while Coconut at its trained K=6 reached 31.0%, a 61-point
matched-branch gap. At the Coconut endpoint, removing latents caused structural
nontermination, yet retaining them still underperformed: the latents were
load-bearing for termination but deleterious for capability in this run. The
endpoint Coconut-minus-CoT harmfulness estimate was +0.01172 (paired-prompt
bootstrap 95% CI [-0.02278, +0.04594]); this is an inconclusive null, not
evidence of equivalence. Mid-training results suggest substrate-dependent
trajectory structure, but their conservative uncertainty region crosses zero
and the branches lack matched capability/coherence controls at that update.
Post-hoc model diffing and token-mode readout add descriptive mechanism clues,
not causal or faithful explanations.

## Claim map

| Claim | Analysis role | Supported conclusion |
|---|---|---|
| Endpoint Coconut − CoT safety | registered primary substrate comparison | inconclusive null; equivalence not established |
| Matched endpoint capability | required registered co-report | 61-point CoT advantage over our skip0 reproduction |
| Coconut endpoint K=0 vs K=6 | registered fixed-weights primary | safety scalar undefined at K=0; latents load-bearing for termination but harmful to capability |
| CoT safety trajectory | registered trajectory | transient single-seed u1 elevation |
| CoT − Coconut at u1 | post-hoc bounded contrast on registered trajectory data | suggestive structure; not a detected 95% interaction |
| Human audit | registered qualitative validation | coherence asymmetry; judge-deflation concern not demonstrated |
| Weight diff | post hoc descriptive | update magnitude and alignment differ; no causal localization |
| Token-mode readout | post hoc descriptive | period-two decodability; no faithfulness claim |
| Self-jailbreaking comparison | literature synthesis | no substantial drift replicated in this setup only |

## Design facts

- Base M0: `Qwen/Qwen3-4B-Thinking-2507`, frozen revision
  `768f209d9ea81521153ed38c47d515654e938aea`.
- Primary branches start from the same M0 and use the same 7,473 GSM8K
  examples, frozen order, seed 42, optimizer, and effective batch size 32.
- Both branches use micro-batch 1 × gradient accumulation 32 at every stage.
  Loss normalization is token-count-correct. This is a documented,
  expectation-preserving deviation from the public recipe's 2 × 16; it is not
  bit-identical to that arithmetic.
- Training dose: three stages × two epochs. Coconut uses `c_thought=2` and
  trained depths K=2, 4, 6.
- Safety assay: frozen StrongREJECT-60 with the pinned judge.
- Frozen endpoint equivalence margin: ±0.0199470123.
- Generation caps are selected mechanically and score-blind, separately by
  task, branch, and checkpoint; the adequacy rule is strictly `<5%` truncation.
  Official judging uses fresh generations after the cap is frozen.

Primary protocol: `project_plan/MATS_execution_protocol_v6_3.md`, its dated
amendments, and the frozen configs. Scope decisions are recorded in
`configs/deadline_scope_freeze_2026-08-31.yaml` and `research_log.md`.

## Frozen number inventory

### Endpoint safety, StrongREJECT-60

| Condition | Mean judge score |
|---|---:|
| M0 | 0.0880096555 |
| explicit CoT u3 | 0.0934414268 |
| Coconut u3, K=6 | 0.1051624656 |
| Coconut − CoT | +0.0117210388 |
| paired-prompt 95% CI | [-0.0227828934, +0.0459375448] |

Provenance:
`artifacts/discovery/results/official_safety/scores/paired_comparison.json`,
the three files in `artifacts/discovery/results/official_safety/scores/`, and
`configs/official_safety_scoring.yaml`.

### Endpoint capability, native-format GSM8K-200

| Condition | Observed | Nontermination/truncation | No-imputation range |
|---|---:|---:|---:|
| M0 | 175/200 = 87.5% | 4/200 | [87.5%, 89.5%] |
| explicit CoT u3 | 184/200 = 92.0% | 0/200 | 92.0% |
| Coconut u3, K=6 | 62/200 = 31.0% | 0/200 | 31.0% |

Derived contrasts:

- CoT − M0: +2.5 to +4.5 percentage points.
- Coconut K=6 − M0: -58.5 to -56.5 percentage points.
- Coconut K=6 − CoT: -61.0 percentage points.

Provenance: `artifacts/discovery/results/native_gsm8k_controls/*/summary.json`
and the stage-3 Coconut capability summary recorded in
`artifacts/discovery/results/fallback_4b_skip0/k_trajectory_consolidated.json`.

### Safety trajectory

| Update | explicit CoT | Coconut identified set | Coconut nontermination basis |
|---|---:|---:|---|
| M0 | 0.0880096555 | same M0 | none |
| u1 | 0.1267296394 | [0.0509530584, 0.0842863917] | fresh bounds run: 2/60; score-blind calibration: 3/60 through 16k |
| u2 | 0.0948836009 | [0.0836456140, 0.1336456140] | 3/60 at 5,120 |
| u3 | 0.0934414268 | 0.1051624656 | 0/60 |

The Coconut bounds assign each missing outcome the theoretical assay extremes
0.0 and 1.0, not the observed maximum. A later fresh u1 run producing 2/60
nonterminations does not retroactively make the original 3/60 calibration pass
the strict `<5%` rule and does not authorize a point estimate.

CoT u1 − M0: +0.0387199839, paired-prompt bootstrap 95% CI
[+0.0006073858, +0.0772292611]. Three endpoints were inspected; no
multiplicity adjustment was applied.

At u1, CoT − Coconut:

- pointwise identified set: [+0.0424432476, +0.0757765810];
- lower-endpoint 95% CI: [-0.0214918616, +0.0997356698];
- upper-endpoint 95% CI: [+0.0331138500, +0.1200593761];
- conservative confidence region: [-0.0214918616, +0.1200593761].

Provenance:
`artifacts/discovery/results/dense_safety/trajectory/trajectory.json` and
`artifacts/discovery/results/dense_safety/trajectory/cot_minus_coconut_u1_bounded_bootstrap.json`.

### Coconut capability/coherence trajectory

| Stage | K=0 GSM8K | trained-K GSM8K | coherence K=0 / trained K |
|---|---:|---:|---:|
| u1, K=2 | 76.0% (8/200 length stops; 4.0%) | 65.5% | 1.9 / 1.3 |
| u2, K=4 | 69.5% (5/200 length stops; 2.5%) | 46.0% | 2.0 / 2.0 |
| u3, K=6 | [49.5%, 59.5%] (20/200 structural length stops) | 31.0% | 2.0 / 2.0 |

The stage-1 trained-depth cost is 10.5 points. The trajectory dates the loss
of robust K=0 viability; it does not show that K=0 was in-distribution at u3.

Provenance:
`artifacts/discovery/results/fallback_4b_skip0/k_trajectory_consolidated.json`
and the stage-specific blind-coherence artifacts under the same directory.

## Claim cards: approved wording and ceilings

### Claim 1 — endpoint safety is an inconclusive null, not equivalence

Approved wording:

> The endpoint Coconut-minus-CoT harmfulness estimate was +0.01172
> (paired-prompt bootstrap 95% CI [-0.02278, +0.04594]). The interval includes
> zero and is not contained within the frozen ±0.01995 equivalence margin. We
> therefore detected no measurable endpoint difference at this assay's
> resolution; equivalence was not established.

Do not write “no effect,” “the methods are equally safe,” “equivalent,” or
“the null was proven.” The interval is prompt-level uncertainty from one
training seed per branch.

### Claim 2 — matched benign training sharply diverged in capability

Approved wording:

> Under matched benign post-training, explicit CoT reached 92.0% GSM8K
> accuracy, whereas our skip-stage-0 Coconut reproduction reached 31.0% at
> K=6, a 61-point matched-branch gap. Relative to M0's no-imputation range
> [87.5%, 89.5%], CoT changed by +2.5 to +4.5 points and Coconut by -58.5 to
> -56.5 points.

Use “our reproduction of the published skip0 recipe,” not an unqualified
claim about Coconut generally. Public checkpoint access was unavailable, so
we did not verify weight-level agreement with the reference implementation.
The public full and skip0 model cards document the recipes and describe their
uploaded endpoints as converged, but report no GSM8K metric against which to
validate our 31.0% result. Do not say where the “damage lives”; the completed
weight diff does not identify a causal locus.

### Claim 3 — latents are load-bearing for termination, not beneficial overall

Approved wording:

> At the final Coconut checkpoint, K=0 caused structural nontermination on
> 13/60 safety prompts even with a 16,000-token ceiling, so its safety mean and
> the K=6-minus-K=0 safety scalar are undefined. On GSM8K, K=0 accuracy was
> bounded at [49.5%, 59.5%], versus 31.0% at K=6, giving a K=6-minus-K=0
> capability effect of [-28.5, -18.5] points. The latents were therefore
> load-bearing for termination but deleterious for capability at this endpoint.

K=0 is an inference-only necessity ablation and is out-of-distribution for a
skip0-trained endpoint. Do not complete-case score it, impute a safety scalar,
or describe nontermination as a missing experiment.

### Claim 4 — explicit-CoT safety elevation was transient and single-seed

Approved wording:

> The explicit-CoT trajectory rose from 0.08801 at M0 to 0.12673 at u1, then
> returned to 0.09488 at u2 and 0.09344 at u3. The paired-prompt u1-minus-M0
> interval was [+0.00061, +0.07723]. This is a transient within-run elevation;
> it comes from one training seed, three checkpoints were examined without a
> multiplicity correction, and the interval captures prompt rather than seed
> variation.

“Early format shock” may appear only as a labeled hypothesis, not an
explanation. Update this card only after the seed-43 pipeline is complete.

### Claim 5 — u1 has suggestive bounded substrate structure, not a detected interaction

Approved wording:

> At u1, the CoT-minus-Coconut contrast had a strictly positive pointwise
> identified set, [+0.04244, +0.07578], after assigning Coconut's
> nonterminating outcomes the theoretical score extremes 0 and 1. However,
> its conservative confidence region, [-0.02149, +0.12006], crossed zero.
> This is suggestive transient substrate-dependent trajectory structure, not
> a detected 95% interaction.

Mandatory context: Coconut-u1 K=2 GSM8K was 65.5%, and its blind coherence
mean was 1.3/2 with 4/10 fully coherent outputs. There is no matched native
CoT-u1 capability or human-coherence control, so the safety contrast cannot
be presented as unconfounded. Coconut u1/u2 are bound bars, not points.

### Claim 6 — the human audit reveals asymmetry, but not judge-score deflation

Frozen audit counts:

- CoT: 12/12 coherent; 11 refusal, 1 partial compliance, 0 harmful.
- Coconut: 7/12 coherent; 9 refusal, 1 harmful compliance, 2 incoherent.
- The harmful compliance received judge score 0.332275.
- Mean judge score was 0.160983 for incoherent rows and 0.155220 for coherent
  rows in this n=12 Coconut audit.
- At M0, response length and judge score had r=0.708 (n=12), descriptive only.

Approved wording:

> Endpoint human review found a coherence asymmetry and one Coconut harmful
> compliance, but the small audit did not support the concern that incoherence
> mechanically deflated judge scores: incoherent and coherent Coconut rows
> averaged 0.161 and 0.155, respectively. This concern remains a limitation,
> not a demonstrated explanation of the endpoint null.

A coherent-only estimate is post-treatment conditioning and may be shown only
as a descriptive sensitivity check, never as a corrected primary estimate.

Provenance: `artifacts/discovery/results/official_safety/judge_human/` and
`artifacts/discovery/results/official_safety/human_audit/`.

### Claim 7 — post-hoc weight diffing shows distributed, differently aligned updates

Frozen results:

- Overall relative update norm: CoT 0.004727457 (0.473%); Coconut 0.004376299
  (0.438%); Coconut/CoT ratio 0.9257.
- Whole-update cosine similarity: 0.426879.
- Coconut's update norm is smaller in every transformer block (ratios
  approximately 0.756–0.969).
- Cross-regime alignment generally rises with depth (layer 1 cosine 0.145;
  layer 35 cosine 0.528), but not monotonically.
- Shared embedding rows are the exception: CoT 0.669%, Coconut 0.716%, ratio
  1.07, cosine 0.866.
- M0 had 151,936 padded embedding rows versus 151,672 at the endpoints; the
  comparison uses only the shared token prefix. A tied `lm_head` alias was
  absent and excluded.

Approved wording:

> Post-hoc weight diffing found similarly sized but only moderately aligned
> global updates (0.473% for CoT, 0.438% for Coconut; cosine 0.427). Coconut
> moved less in every transformer block, while shared embeddings were a small
> exception. This describes where parameter changes differ; it does not
> localize the cause of capability loss.

Provenance:
`artifacts/discovery/results/posthoc_layerwise_weight_diff/layerwise_weight_updates.json`.

### Claim 8 — token projections show period-two decodability only

Protocol: first five frozen GSM8K prompts and first five frozen StrongREJECT
prompts; six latent positions; six depths; native output head; top-10 overlap;
no generation and no judge access.

| Task | same parity | opposite parity | curriculum-stage pairs | all-pairs baseline | lag 2 |
|---|---:|---:|---:|---:|---:|
| GSM8K | 0.4472 | 0.0000 | 0.0000 | 0.1789 | 0.4738 |
| StrongREJECT | 0.1535 | 0.0361 | 0.0366 | 0.0831 | 0.2221 |

On StrongREJECT, the same-parity advantage was positive on 4/5 prompts.

Approved wording:

> In the native unembedding basis, top-token projections were more similar
> across same-parity latent positions than across opposite-parity positions
> on both small prompt samples. This period-two pattern is consistent with
> alternating slot roles under `c_thought=2`, but the design contains no
> alternate-`c_thought` control and does not identify the curriculum as its
> cause.

Claim ceiling, verbatim:

> These are decodability results under a logit lens, not evidence of
> faithfulness, causal use, monitorability, or absence of distributed content.

No obvious readable harmful plan or refusal trajectory appeared in the top
projections. That absence is not evidence that the latent states lack such
distributed information. The analysis is post hoc, n=5 prompts per task, and
top-10 overlap is coarse.

Provenance:
`artifacts/discovery/results/posthoc_token_mode_readout/period_two_analysis.json`
and `token_mode_readout.json` in the same directory.

### Claim 9 — relation to self-jailbreaking literature is bounded

Approved wording:

> We did not replicate substantial benign-training safety drift in this
> setup: M0 scored 0.08801, versus 0.09344 and 0.10516 at the two endpoints.
> The setup differs materially from prior self-jailbreaking demonstrations,
> including the already-aligned thinking base, terse training rationales, and
> small training dose.

Do not write that self-jailbreaking “does not replicate” in general.

Related-work citation rule: any specific number quoted from prior work (e.g.
the Coconut paper's GPT-2 GSM8K accuracies for the latent-below-explicit
direction claim) must be verified against the source paper or vendored repo
before it enters a draft — never cited from memory. Directional consistency
with the Coconut paper may be claimed only after that check.

## Pending evidence not yet claimable

The seed-43 explicit-CoT u1 checkpoint exists, but no completed, committed
official safety result is included in the evidence cutoff. Therefore:

- Claims 4 and 5 remain based on the primary seed only.
- Do not say that the u1 elevation replicated or failed to replicate.
- When the frozen evaluation completes, add its mean, prompt-level interval,
  and comparison with seed 42 here regardless of direction.
- Treat the result as a second-seed noise-floor diagnostic. Two CoT seeds do
  not yield a seed confidence interval, and there is no second Coconut seed.

Checkpoint provenance is in
`artifacts/discovery/results/matched_4b_cot_seed43/stage1/`. Operational status
and the evaluation chain remain solely in `research_log.md`.

## Power and frozen scope

To put the endpoint CI inside the equivalence margin around the observed
estimate requires a half-width near 0.0082 rather than the current ~0.0344,
or roughly `(0.0344 / 0.0082)^2 ≈ 17×` the information under simple scaling.
StrongREJECT-313 supplies only about 5.2× as many prompts. The no-313 decision
is therefore a frozen deadline cut, not an invitation to expand and peek.

The completed trajectory, endpoint K test, human audit, layer diff, and token
readout remain reportable. New training branches and deeper intervention work
are future work unless explicitly reopened in a dated amendment.

## Methods and reporting commitments

- Pre-registration froze the estimands, margin, caps, gates, and conditional
  branches before official scores.
- Per-task/per-branch cap calibration caught a real harness artifact: M0 at a
  512-token ceiling had 24.5% truncation. Never quote that run as M0 ability.
- Nontermination is treated with identified-set bounds or an undefined scalar,
  according to the estimand, rather than convenience imputation or
  completed-cases-only scoring.
- Departures from the intended blind-review sequence must be disclosed briefly
  in methods or limitations. Their operational chronology stays in
  `research_log.md`.
- Post-freeze analyses are labeled exploratory and cannot replace a registered
  primary result.

## Prohibited wording

- Never quote 24.5% as M0 capability.
- Never give the Coconut-u3 K=0 safety scalar a numeric value; it is undefined.
- Never call the endpoint result equivalent, safe, “no effect,” or a proved null.
- Never call the u1 substrate interaction detected at 95% confidence.
- Never show Coconut u1/u2 safety as point estimates; use bound bars.
- Never treat completed outputs as if termination were pre-treatment.
- Never present coherent-only scoring as a corrected estimate.
- Never turn prompt-bootstrap intervals into across-seed uncertainty.
- Never call the seed-43 run a seed CI.
- Never claim the readout is faithful, causal, monitorable, or content-free.
- Never claim the layer diff localizes where capability was damaged.
- Never generalize from “our skip0 reproduction” to Coconut as a method class.
- Never select a checkpoint, cap, or implementation by safety-score direction.

## Write-up outline

### 1. Question and motivation

Ask whether changing the reasoning substrate during matched benign training
changes safety drift, while treating capability, coherence, and termination as
necessary context rather than disposable side metrics.

### 2. Pre-registered matched design

Specify M0, data, frozen order, seed, optimizer, effective batch, the shared
1×32 micro-batch configuration, explicit-CoT versus skip0 Coconut, K schedule,
StrongREJECT-60, GSM8K-200, human review, cap policy, and ±0.01995 margin.

### 3. Capability and termination result

Show M0, CoT-u3, Coconut-u3 K=6, and the K=0 capability range. State the
61-point matched-branch gap and the load-bearing/deleterious distinction. Name
the inaccessible-reference limitation and avoid a universal Coconut claim.

### 4. Endpoint registered primary result

Give the estimate and CI against the frozen margin immediately after the
capability result. Use Claim 1 exactly: inconclusive null, not equivalence.
Show M0 in the table/figure so branch means are interpretable as drift. Leading
the results narrative with the larger capability finding does not change the
endpoint safety comparison's registered-primary status.

### 5. Safety trajectory

Plot all updates. CoT uses points; Coconut u1/u2 use identified-set bound bars.
Give the CoT-u1 interval, the u1 bounded substrate contrast, multiplicity and
single-seed cautions, and mandatory u1 capability/coherence context. Incorporate
seed 43 only after the full pending chain finishes.

### 6. Human validation and measurement limitations

Report the blind-audit counts, harmful example and judge score, coherence
asymmetry, non-demonstrated judge-deflation concern, length correlation, cap
calibration, and contamination disclosures. Label coherent-only analysis as
post-treatment and descriptive.

### 7. Post-hoc mechanistic observations

Put layer-wise diffing and token-mode readout in a clearly exploratory section.
Use Claim 7's nonlocalizing language and Claim 8's decodability-only ceiling.
The period-two/`c_thought=2` match is a consistency observation, not causation.

### 8. Interpretation and related work

Center the capability/termination finding, then the inconclusive endpoint
safety result and suggestive transient structure. Compare cautiously with
self-jailbreaking work using explicit setup differences. Distinguish absence of
detection from evidence of absence.

### 9. Limitations and next experiments

List one primary seed per branch, n=60 safety prompts, prompt-level CIs,
nontermination, missing matched u1 capability/coherence controls, inaccessible
public checkpoints, and post-hoc interpretability analyses. Highest-value future
tests: slot-aware latent substitution (mean/noise/A↔B swap), alternate
`c_thought`, nontermination mechanism analysis, more seeds, and a sufficiently
powered safety assay.

## Figure inventory

1. Capability plot: M0 range, CoT-u3 point, Coconut-u3 K=6 point, Coconut K=0
   range and nontermination annotation.
2. Endpoint safety forest/equivalence plot: M0 context; Coconut−CoT estimate
   and CI; frozen ±0.01995 margin.
3. Full safety trajectory: CoT dots/line; Coconut u1/u2 vertical bound bars;
   u1 contrast uncertainty; nontermination counts.
4. Human audit table: behavior and coherence counts plus judge-vs-human check.
5. Appendix weight-diff plot: relative norms and cosine alignment, explicitly
   descriptive.
6. Appendix readout heatmap/table: pairwise top-10 overlaps, with the
   decodability-only caption.

## Final drafting checklist

- [ ] Every headline number matches a cited committed artifact.
- [ ] M0 appears wherever endpoint branch safety means appear.
- [ ] Endpoint wording says inconclusive null and not equivalence.
- [ ] Coconut u1/u2 appear as bounds, not points.
- [ ] K=0 safety remains undefined.
- [ ] Prompt uncertainty is not described as seed uncertainty.
- [ ] Capability/coherence/termination context accompanies safety comparisons.
- [ ] Seed-43 text is updated only after its complete frozen pipeline.
- [ ] Exploratory analyses are labeled post hoc.
- [ ] Readout text says decodability only.
- [ ] Layer-diff text makes no causal localization claim.
- [ ] Prior-work numbers verified against sources, not memory.
- [ ] The final draft has been checked against this file, not chat memory.
