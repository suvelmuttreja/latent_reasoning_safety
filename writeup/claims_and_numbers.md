# Claims, numbers, and evidence map

> **Analysis correction (2026-09-05):** The independent [analysis audit](analysis_audit/REVIEW.md) supersedes older K=0 bounds, language-gap interpretations, embedding-inclusive training-update claims, and evaluator visibility assumptions. Frozen raw results remain unchanged. Final K=0 bounds are 48–58%; 49.5% is cutoff-parser accuracy, including three unfinished correct parses.


Evidence cutoff: committed artifacts available on 2026-09-01.

This is the project's canonical **interpretation layer**: the numbers worth
using and the strongest defensible language for each claim. It is deliberately
not a chronological record.

The result summary below is followed by supporting evidence, claim boundaries,
and provenance rather than a second report draft.

The authority order is:

1. committed result artifacts for numerical values and provenance;
2. frozen protocols/configs for design and analysis commitments;
3. this file for claim wording and synthesis;
4. derived reports, figures, captions, or summaries.

Job submissions, queue states, checkpoint-copy operations, failures, repairs,
and dated decisions belong in `research_log.md`, not here.

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

## Result summary

I set out to test whether replacing explicit reasoning with recurrent latent
reasoning during matched benign post-training changes safety drift. The
experiment did not produce a capability-matched latent model, so it could not
cleanly answer that question. Explicit CoT reached 92.0% GSM8K accuracy, while
our independent Meta-style implementation of the public skip-stage-0
configuration reached 31.0% at its trained K=6, a 61-point gap. This result
cannot be generalized to Coconut: the exact Qwen training wrapper, processed
training file, and reference weights were inaccessible, so implementation or
recipe divergence remains a live explanation.

The failure had structure. Across the Coconut trajectory, trained-depth GSM8K
accuracy declined from 65.5% to 46.0% to 31.0%. At the endpoint, removing the
latents caused structural nontermination, yet retaining them reduced capability
further: the latent states were load-bearing for termination but deleterious for
GSM8K performance in this run. A second explicit-CoT seed repeated the direction
of an early safety elevation, but not its nominal prompt-level significance;
both bounded CoT-minus-Coconut regions crossed zero and lacked matched
capability/coherence controls.

The registered endpoint Coconut-minus-CoT harmfulness estimate was +0.01172
(paired-prompt bootstrap 95% CI [-0.02278, +0.04594]). The interval includes
zero and exceeds the frozen equivalence margin. More importantly, the 61-point
capability divergence makes the safety contrast capability-confounded. It is an
underpowered, inconclusive registered result, not evidence of substrate safety,
harm, equivalence, or no effect. Post-hoc model diffing and token-mode readout
are descriptive clues only and do not validate the implementation or explain
the failure causally.

## Claim map

| Claim | Analysis role | Supported conclusion |
|---|---|---|
| Matched endpoint capability | model-organism validity check and required co-report | 61-point CoT advantage; intended substrate comparison became capability-confounded |
| Coconut endpoint K=0 vs K=6 | registered fixed-weights primary | safety scalar undefined at K=0; latents load-bearing for termination but harmful to capability |
| Endpoint Coconut − CoT safety | registered primary estimand, narratively demoted after validity failure | underpowered and capability-confounded inconclusive result; equivalence not established |
| CoT safety trajectory | registered trajectory plus post-hoc second-seed diagnostic | directional u1 elevation in both point estimates; only seed 42 excludes zero |
| CoT − Coconut at u1 | post-hoc bounded contrast on registered trajectory data | both CoT-seed pointwise sets are positive; neither conservative region excludes zero |
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
  Loss is weighted by supervised token count across accumulation. This is a
  documented deviation from the public recipe's 2 × 16 microbatch-loss averaging;
  different example lengths can change the weighting.
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
| u1 | seed 42: 0.1267296394; seed 43: 0.1135835012 | [0.0509530584, 0.0842863917] | bounds artifact: 2/60; cap-adequacy evidence: 3/60 at 16,000 |
| u2 | 0.0948836009 | [0.0836456140, 0.1336456140] | 3/60 at 5,120 |
| u3 | 0.0934414268 | 0.1051624656 | 0/60 |

The Coconut bounds assign each missing outcome the theoretical assay extremes
0.0 and 1.0, not the observed maximum. The u1 bounds artifact and cap-adequacy
artifact serve different registered roles: 2/60 determines the displayed
bounds, while 3/60 exactly fails the strict `<5%` adequacy rule. Neither role
authorizes a scalar point estimate.

CoT u1 − M0:

- seed 42: +0.0387199839, paired-prompt bootstrap 95% CI
  [+0.0006073858, +0.0772292611];
- seed 43: +0.0255738457, paired-prompt bootstrap 95% CI
  [-0.0092438617, +0.0612922352].

The observed seed-43-minus-seed-42 difference is -0.0131461382, paired-prompt
95% CI [-0.0454932489, +0.0190783025]. This is not a seed confidence interval.
Three seed-42 endpoints were inspected; no multiplicity adjustment was applied.

At u1, CoT − Coconut:

- seed-42 pointwise identified set: [+0.0424432476, +0.0757765810];
  conservative confidence region: [-0.0214918616, +0.1200593761];
- seed-43 pointwise identified set: [+0.0292971094, +0.0626304428];
  conservative confidence region: [-0.0339444330, +0.1008435236].

Provenance:
`artifacts/discovery/results/dense_safety/trajectory/trajectory.json` and
`artifacts/discovery/results/dense_safety/trajectory/cot_minus_coconut_u1_bounded_bootstrap.json`,
plus
`artifacts/discovery/results/matched_4b_cot_seed43/cot_u1_seed_comparison.json`.

### Coconut capability/coherence trajectory

| Stage | K=0 GSM8K | trained-K GSM8K | coherence K=0 / trained K |
|---|---:|---:|---:|
| u1, K=2 | 76.0% (8/200 length stops; 4.0%) | 65.5% | 1.9 / 1.3 |
| u2, K=4 | 69.5% (5/200 length stops; 2.5%) | 46.0% | 2.0 / 2.0 |
| u3, K=6 | [48%, 58%] (20/200 structural length stops) | 31.0% | 2.0 / 2.0 |

The stage-1 trained-depth cost is 10.5 points. The trajectory dates the loss
of robust K=0 viability; it does not show that K=0 was in-distribution at u3.

Provenance:
`artifacts/discovery/results/fallback_4b_skip0/k_trajectory_consolidated.json`
and the stage-specific blind-coherence artifacts under the same directory.

## Claim cards: approved wording and ceilings

### Claim 1 — the intended substrate comparison failed its capability-validity condition

Approved wording:

> Matched benign post-training did not produce a capability-matched latent
> model. Explicit CoT reached 92.0% GSM8K accuracy, whereas our independent
> Meta-style implementation of the public skip-stage-0 configuration reached
> 31.0% at K=6, a 61-point gap. The registered safety contrast is therefore
> capability-confounded and cannot cleanly answer whether reasoning substrate
> changes safety.

The endpoint safety contrast remains the registered primary estimand and must
still be reported. Narrative demotion records a model-organism validity failure;
it does not relabel the capability result as a pre-registered safety endpoint.

### Claim 2 — endpoint safety is underpowered, inconclusive, and not equivalence

Approved wording:

> The endpoint Coconut-minus-CoT harmfulness estimate was +0.01172
> (paired-prompt bootstrap 95% CI [-0.02278, +0.04594]). The interval includes
> zero and is not contained within the frozen ±0.01995 equivalence margin. We
> therefore detected no measurable endpoint difference at this assay's
> resolution; equivalence was not established.

Do not write “no effect,” “the methods are equally safe,” “equivalent,” or
“the null was proven.” The interval is prompt-level uncertainty from one
endpoint training seed per branch.

### Claim 3 — matched benign training sharply diverged in capability

Approved wording:

> Under matched benign post-training, explicit CoT reached 92.0% GSM8K
> accuracy, whereas our independent Meta-style implementation of the public
> skip-stage-0 configuration reached 31.0% at
> K=6, a 61-point matched-branch gap. Relative to M0's no-imputation range
> [87.5%, 89.5%], CoT changed by +2.5 to +4.5 points and Coconut by -58.5 to
> -56.5 points.

Use “our independent Meta-style implementation of the public skip-stage-0
configuration,” not “a faithful reproduction” or an unqualified claim about
Coconut generally. Public checkpoint access was unavailable, so we did not
verify weight-level agreement with the reference implementation. The public
cards expose headline hyperparameters but not the exact Qwen wrapper, processed
training file, or a GSM8K result against which to validate our 31.0% outcome.
Implementation or recipe divergence therefore remains a live explanation and
must accompany the capability claim. Do not say where the “damage lives”; the
completed weight diff does not identify a causal locus.

### Claim 4 — latents are load-bearing for termination, not beneficial overall

Approved wording:

> At the final Coconut checkpoint, K=0 caused structural nontermination on
> 13/60 safety prompts even with a 16,000-token ceiling, so its safety mean and
> the K=6-minus-K=0 safety scalar are undefined. On GSM8K, K=0 accuracy was
> bounded at [48%, 58%], versus 31.0% at K=6, giving a K=6-minus-K=0
> capability effect of [-27, -17] points. The latents were therefore
> load-bearing for termination but deleterious for capability at this endpoint.

K=0 is an inference-only necessity ablation and is out-of-distribution for a
skip0-trained endpoint. Do not complete-case score it, impute a safety scalar,
or describe nontermination as a missing experiment.

### Claim 5 — the directional explicit-CoT u1 elevation appears under two seeds

Approved wording:

> The explicit-CoT u1 mean exceeded M0 under both training seeds: 0.12673 for
> seed 42 and 0.11358 for seed 43, versus 0.08801 at M0. Their paired-prompt
> contrasts were +0.03872 [0.00061, 0.07723] and +0.02557 [-0.00924, 0.06129],
> respectively. Thus the directional elevation recurred in the second point
> estimate, but only seed 42's prompt-level interval excluded zero. The observed
> seed-43-minus-seed-42 difference was -0.01315 [-0.04549, 0.01908].

“Early format shock” may appear only as a labeled hypothesis, not an
explanation. Seed 42 returned near M0 at u2/u3; seed 43 was evaluated only at
u1. These intervals represent prompt uncertainty, not training-seed
uncertainty, and three seed-42 checkpoints were inspected without multiplicity
correction.

Interpret the observed 0.01315 seed difference only as a one-observation
noise-floor calibration: it is of the same order as the 0.01172 endpoint
substrate difference, while both u1 pointwise substrate sets start above it.
The comparisons involve different checkpoints and do not estimate seed
variance.

Evidence-quality note: seed 43 used the frozen 60 prompts at a mechanically,
score-blind selected 4,096-token ceiling and had zero truncations. Two CoT
seeds do not yield a seed confidence interval, and the shared single Coconut
seed prevents claims about Coconut seed variability.

### Claim 6 — u1 has suggestive bounded substrate structure, not a detected interaction

Approved wording:

> At u1, the CoT-minus-Coconut pointwise identified set was strictly positive
> under both CoT seeds: [+0.04244, +0.07578] for seed 42 and
> [+0.02930, +0.06263] for seed 43, assigning Coconut's two nonterminating
> outcomes the theoretical score extremes 0 and 1. Both conservative regions
> crossed zero (seed 42: [-0.02149, +0.12006]; seed 43:
> [-0.03394, +0.10084]). The directional substrate pattern therefore repeats
> across the two CoT point estimates, but it is not a detected 95% interaction.

Mandatory context: Coconut-u1 K=2 GSM8K was 65.5%, and its blind coherence
mean was 1.3/2 with 4/10 fully coherent outputs. There is no matched native
CoT-u1 capability or human-coherence control, so the safety contrast cannot
be presented as unconfounded. The same single Coconut seed anchors both
comparisons, so this does not estimate Coconut seed variability. Coconut u1/u2
are bound bars, not points.

### Claim 7 — the human audit reveals asymmetry, but not judge-score deflation

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

Post-hoc descriptive addendum (2026-09-01, derived from committed endpoint
generations by a character-class rule, share of CJK Unified Ideographs among
all characters > 0.30, all 60 prompts English): 12/60 Coconut u3 K=6 reasoning
fields and 7/60 final answers crossed the threshold; M0 and explicit CoT were
0/60 on both. Approved wording: "Twelve Coconut reasoning fields and seven
final answers crossed a >30%-CJK character threshold on English prompts;
neither comparison condition did." The rule is not a language identifier. Do
not call this a cause of the capability loss or of the safety scores; the judge
was not validated on these CJK-heavy answers, making the seven final-answer
scores a condition-specific measurement gap. Provenance:
`writeup/figures/language_switch_counts.json` and the script beside it.

Provenance: `artifacts/discovery/results/official_safety/judge_human/` and
`artifacts/discovery/results/official_safety/human_audit/`.

### Claim 8 — post-hoc weight diffing shows distributed, differently aligned updates

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

### Claim 9 — token projections show period-two decodability only

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

### Claim 10 — relation to self-jailbreaking literature is bounded

Approved wording:

> We did not replicate substantial benign-training safety drift in this
> setup: M0 scored 0.08801, versus 0.09344 and 0.10516 at the two endpoints.
> The setup differs materially from prior self-jailbreaking demonstrations,
> including the already-aligned thinking base, terse training rationales, and
> small training dose.

Do not write that self-jailbreaking “does not replicate” in general.

Verified 2026-09-05 from arXiv 2412.06769 (HTML, Table 1, GPT-2 GSM8k): CoT 42.9±0.2, No-CoT 16.5±0.5, Coconut 34.1±1.5, Pause as thought 24.1±0.7, Coconut w/o thought 21.6±0.5, Pause token 16.4±1.8. Directional consistency (latent below explicit CoT) may now be claimed; magnitude differs (9 vs 61 points).

Post-hoc descriptive addendum (2026-09-05, `writeup/figures/posthoc_output_anatomy.json`): Coconut stage-3 K=6 GSM8K outputs have median 6 generated tokens (bare answer); stage-2 K=4 median 41; stage-3 K=0 terminating rows median 65 (terse text steps). K=0 vs K=6 identical predicted answer on 42/200. Coconut endpoint Chinese-thinking rows (12) mean judge 0.044 vs English-thinking rows (48) 0.120; paired Coconut−CoT +0.002 vs +0.014. Answer-length vs judge Pearson r at n=60: M0 0.39, CoT 0.44, Coconut 0.34 (supersedes the n=12 r=0.708 as the quoted figure).

Related-work citation rule: any specific number quoted from prior work (e.g.
the Coconut paper's GPT-2 GSM8K accuracies for the latent-below-explicit
direction claim) must be verified against the source paper or vendored repo
before it enters the project synthesis — never cited from memory. Directional consistency
with the Coconut paper may be claimed only after that check.

## Power limitation and analysis scope

The project did not conduct and act on a prospective equivalence-power analysis
once the frozen margin and baseline variability were available. That was a
design mistake, not a virtue of pre-registration. Under simple scaling and a
zero-centered effect, the observed endpoint half-width of ~0.0344 would need to
fall below 0.01995, requiring roughly 3× the information. The exact 17× figure
below additionally depends on the observed nonzero endpoint estimate and could
not have been known prospectively.

To put the endpoint CI inside the equivalence margin around the observed
estimate requires a half-width near 0.0082 rather than the current ~0.0344,
or roughly `(0.0344 / 0.0082)^2 ≈ 17×` the information under simple scaling.
StrongREJECT-313 supplies only about 5.2× as many prompts. The no-313 decision
is therefore a frozen scope choice, not an invitation to expand and peek after
observing the registered result.

The completed trajectory, endpoint K test, human audit, layer diff, and token
readout remain reportable. New training branches and deeper intervention
analyses are outside this evidence set and belong in future work.

## Methods and reporting commitments

- Pre-registration froze the estimands, margin, caps, gates, and conditional
  branches before official scores.
- Per-task/per-branch cap calibration caught a real harness artifact: M0 at a
  512-token ceiling had 24.5% truncation. Never quote that run as M0 ability.
- Nontermination is treated with identified-set bounds or an undefined scalar,
  according to the estimand, rather than convenience imputation or
  completed-cases-only scoring.
- Any departure from the intended blind-review sequence is disclosed in methods
  or limitations; its operational chronology remains in `research_log.md`.
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
- Never call the seed-43 run a seed CI or an independent 95% replication.
- Never claim the readout is faithful, causal, monitorable, or content-free.
- Never claim the layer diff localizes where capability was damaged.
- Never generalize from “our independent skip0 implementation” to Coconut as a
  method class.
- Never select a checkpoint, cap, or implementation by safety-score direction.

## Figure inventory

1. **Capability:** M0 range, CoT-u3 point, Coconut-u3 K=6 point, and Coconut
   K=0 range with a nontermination annotation.
2. **Endpoint safety:** M0 context, Coconut−CoT estimate and interval, and the
   frozen ±0.01995 equivalence margin.
3. **Safety trajectory:** seed-42 CoT dots/line;
   seed-43 u1-only marker;
   Coconut u1/u2 vertical bound bars; both u1 contrast regions;
   nontermination counts.
4. **Qualitative examples:** seeded random examples plus the targeted
   harmful-compliance case.
5. **Weight differences:** relative norms and cosine alignment, explicitly
   descriptive.
6. **Token readout:** pairwise top-10 overlaps with a decodability-only caption.
