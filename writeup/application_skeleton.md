# Application skeleton (reframed: capability collapse first)

Purpose: a number-verified bullet skeleton for the MATS form answers, the executive
summary, and the doc outline. Every number below was re-read from the committed
artifact on 2026-09-01 and matches `claims_and_numbers.md`. Claim ceilings from that
file still apply; this file only reorders the story.

**You write the prose.** Bullets marked `[YOU]` need your own words or facts I do
not have. Do not paste these bullets into the form. Neel rejects LLM-sounding
answers and reads the form first.

Reframe in one sentence: matched benign GSM8K post-training in our independent
Meta-style implementation of the public skip-stage-0 Coconut configuration
collapsed capability so badly that the registered safety comparison became
uninterpretable; the interesting findings are about the collapse and about what a
transient CoT-side safety bump looked like along the way.

---

## Form Q1. What question did you try to answer?

- Does the *reasoning substrate* used during benign post-training (explicit
  chain-of-thought vs. Coconut-style continuous latent thoughts) change how much
  safety behaviour drifts, holding data, seed, optimizer, and dose fixed?
- Pre-registered primary: endpoint Coconut-minus-CoT StrongREJECT harmfulness,
  with a frozen equivalence margin of ±0.020.
- Secondary, also registered: matched capability, coherence, and termination as
  mandatory context, and a K=0 "remove the latents at inference" ablation.
- `[YOU]` One sentence on the original motivation in your own framing (CoT
  monitorability worry: if reasoning moves into latents, does benign training
  quietly erode safety in a way we cannot read?).

## Form Q2. Why is this interesting / why did you choose it?

- Benign fine-tuning is known to degrade safety in aligned models. Latent
  reasoning methods are being proposed to replace visible CoT. Nobody had put the
  two together with a matched design on a modern thinking model.
- Science-of-post-training angle: same data, order, seed, optimizer, and update
  count, but different reasoning/training regimes. This isolates the two implemented
  regimes better than an unmatched comparison, but it does **not** isolate substrate
  alone: the supervision format and latent wrapper differ, and endpoint capability
  diverged sharply.
- `[YOU]` Why you personally picked it. Keep to 2 to 3 sentences.

## Form Q3. What conclusions have you reached?

Order matters. Lead with what you actually learned.

1. **Capability collapsed in the latent-training branch, not the explicit-CoT branch.**
   Same 7,473 GSM8K examples, same order, seed 42, same optimizer and effective
   batch 32, three stages × two epochs. Explicit CoT: 92.0% on GSM8K-200 (M0 base:
   87.5 to 89.5%). Our independent Meta-style implementation of the public skip0
   configuration at its trained depth K=6: 31.0%. A 61-point matched-branch gap.
   Stage by stage at trained K: 65.5 → 46.0 → 31.0.
   Figure 1.
2. **The latents are load-bearing for termination but harmful for capability.**
   Removing them at inference (K=0) at the final checkpoint: 20/200 GSM8K rows never
   terminate (19 are exact repeating token cycles), accuracy bounded [49.5%, 59.5%],
   still above K=6's 31.0%. On the 60 safety prompts, K=0 fails to terminate on
   13/60 even with a 16,000-token ceiling, so its safety score is undefined.
   Nontermination at K=0 on safety prompts was already 12/60 after stage 1.
3. **Because of (1), the registered safety comparison is an inconclusive null,
   and it is confounded.** Coconut−CoT = +0.012, paired-prompt bootstrap 95% CI
   [−0.023, +0.046], not inside the ±0.020 margin. M0 = 0.088, CoT = 0.093,
   Coconut = 0.105. A 31%-accuracy model with 7/12 audited outputs coherent is not
   a clean substrate comparison. Say this plainly: the primary question could not be
   answered with this data.
4. **A transient CoT-side safety bump after stage 1, seen under two seeds.**
   CoT u1 mean 0.127 (seed 42) and 0.114 (seed 43) vs M0 0.088; paired contrasts
   +0.039 [+0.001, +0.077] and +0.026 [−0.009, +0.061]. Only seed 42 excludes zero.
   Seed 42 returned to near M0 at u2/u3 (0.095, 0.093). Coconut at u1 sits *below*
   M0 in its bounds [0.051, 0.084]. The CoT-minus-Coconut u1 identified set is
   strictly positive under both seeds, but every conservative region crosses zero.
   Suggestive, not detected. Figure 2 (existing trajectory plot).
5. **Two descriptive clues, both post hoc.** Weight diffs: similar size
   (0.47% vs 0.44% relative norm), moderately aligned (cosine 0.43), Coconut moved
   less in every transformer block. Logit-lens over the six latent slots: same-parity
   positions share top tokens far more than opposite-parity ones (GSM8K overlap 0.45
   vs 0.00), matching the c_thought=2 curriculum. Decodability only.
6. **CJK-heavy output is a visible signature of the Coconut endpoint.** On the 60
   English safety prompts, 12/60 Coconut u3 K=6 reasoning fields and 7/60 final
   answers crossed an explicit threshold of >30% CJK Unified Ideographs among all
   characters; M0 and CoT were 0/60 on both fields. Found by reading the random
   examples, then counted mechanically. Two threshold-positive rows were the blind
   audit's "all Chinese" observations. Post hoc and descriptive; the heuristic is
   not a language identifier. Figure-free; one table row.
7. **Not replicated: substantial benign-training safety drift in this setup.**
   M0 0.088 → 0.093 (CoT) and 0.105 (Coconut). Setup differs from prior
   self-jailbreaking work (already-aligned thinking base, terse rationales, small
   dose).

## Form Q4. Technical setup

- Base model M0: `Qwen/Qwen3-4B-Thinking-2507`, pinned revision.
- Training: GSM8K train (7,473 examples), 3 stages × 2 epochs, micro-batch 1 ×
  grad-accum 32, one NVIDIA A40 per matched training job on a university Slurm
  cluster. Separate inference and calibration jobs also used available A40/A100
  nodes; hardware was not an experimental variable.
  - Explicit-CoT branch: SFT on the natural-language rationale + answer.
  - Coconut branch: skip-stage-0 curriculum from the public recipe, `c_thought=2`,
    latent depth K=2 / 4 / 6 at stages 1 / 2 / 3. The implementation was built from
    the pinned `facebookresearch/coconut` code and public recipe; the reference
    checkpoints needed for an implementation positive control remained gated.
- Safety metric: StrongREJECT-small, 60 frozen prompts, pinned fine-tuned judge
  scoring the parsed final answer only, 0 = harmless, 1 = fully harmful. Fresh
  sampled generations (`temperature=0.6`, `top_p=0.95`, `top_k=20`, one fixed
  per-row seed) after a score-blind per-task, per-branch cap is frozen (rule: <5%
  truncation, ladder 512 → 16,000).
- Capability: GSM8K-200 fixed test items, native format per branch, greedy.
- Uncertainty: 10,000 paired-prompt bootstraps. Nonterminating rows are never
  imputed: reported as [all-wrong, all-right] bounds or as undefined.
- Human checks: condition-blind coherence ratings (0 to 2) on 10 to 20 outputs per
  condition at each stage, and a sealed 24-row condition-blind behaviour audit at
  the endpoint (refusal / partial / harmful / incoherent).
- Post hoc: layer-wise weight diff vs M0; logit-lens top-10 overlap across latent
  slots on 5 + 5 prompts.
- Two extra things a reader will ask about: the ±0.020 equivalence margin was
  frozen before any branch score; the M0-at-512-tokens run (24.5%) is a harness
  artifact caught by the cap policy and is never quoted as ability.

## Form Q5. Strongest evidence against your hypotheses

State the hypothesis, then the evidence against it.

- *Hypothesis: reasoning/training regime changes safety drift.* Against: endpoint CI [−0.023,
  +0.046] straddles zero; M0 appears in the same range as both endpoints; and the
  comparison is confounded by capability collapse, so even a detected difference
  would not have been attributable to substrate alone.
- *Hypothesis: latents carry harmful or safety-relevant content we cannot see.*
  Against: no readable harmful plan or refusal trajectory appeared in the top
  logit-lens projections (n=5 prompts, top-10, coarse). Absence of detection, not
  evidence of absence.
- *Hypothesis: the u1 bump is a real regime interaction.* Against: seed 43's
  interval includes zero; the conservative regions of both u1 contrasts include
  zero; three seed-42 checkpoints were inspected without multiplicity correction;
  there is no matched u1 capability or coherence control.
- *Hypothesis: incoherence deflates judge scores and hides Coconut harm.* Against:
  in the 12-row Coconut audit, incoherent rows averaged 0.161 and coherent rows
  0.155. The one human-labelled harmful compliance scored 0.332.
- *Hypothesis: the safety null means the two endpoints behave alike.* Against:
  12/60 Coconut reasoning fields crossed the >30%-CJK character threshold versus
  0/60 CoT fields; the scalar hides a qualitative divergence.
- *Hypothesis: the 31% is a parsing artifact.* Against: all 200 K=6 rows produced
  a parsed numeric answer (checked 2026-09-01 from the stage-3 GSM8K rows).

## Form Q6. Biggest limitations, and could you have addressed them?

Be blunt. Neel says he checks.

- **No positive control on the Coconut implementation.** Public checkpoints were
  gated (403 throughout), so agreement with a reference model's capability was never
  verified. K-dependence (82/200 predictions change between K=0 and K=2 at stage 1)
  shows the latents are functional, not that the implementation is correct. Could it
  have been addressed? Only with checkpoint access or a from-scratch GPT-2
  replication of the published numbers; the latter was cut for time. Until then,
  call this an independent implementation, not a faithful reproduction.
- **The primary test was underpowered by construction, and this was not caught
  prospectively.** At zero offset the margin needed roughly 3× the information of
  n=60; after the observed +0.012 offset it needed about 17×. The mistake was not
  running a prospective power analysis and acting on it (StrongREJECT-313 was
  available, 5.2× the prompts). `[YOU]` own this in first person.
- **The intended substrate comparison is confounded by capability and coherence.** There is
  no matched native CoT-u1 capability or coherence control, and the sham
  pause-token branch (which tests whether latent *content* matters versus just extra
  compute slots) was cut in the scope freeze.
- **One training seed per branch on the primary comparison.** All CIs are
  prompt-level, not seed-level. Seed 43 is a single extra CoT point at u1 only, and
  Coconut has one seed everywhere. The seed 43 minus seed 42 difference (−0.013) is
  the same size as the endpoint substrate difference (+0.012).
- **n=60 safety prompts, one judge.** Judge score correlates with response length
  at M0 (r=0.71, n=12). Seven Coconut endpoint answers cross the >30%-CJK
  character threshold; the judge's behaviour on these CJK-heavy answers was not
  validated, so those 7 scores are a measurement-validity gap on the Coconut side
  specifically.
- **Blind-review contamination, disclosed.** At stage 1 an agent pre-scored and
  reported unblinded aggregates before the human pass; at the endpoint a session
  message identified the condition of two anomalous outputs before labelling
  finished. Logged in `research_log.md`.
- **Post-hoc analyses are post hoc.** Weight diff and logit lens were not
  registered and have no alternate-`c_thought` control.

## Form Q7. How did you use LLMs, and how did you avoid slop?

Facts I can source from the repo. `[YOU]` must add the surprise levels and anything
about your own process I cannot see.

**Division of labour (as you described it):**
- You: research question, experimental design and protocol decisions, monitoring
  and go/no-go calls, all human blind reviews.
- Agents (Codex 5.6 and Claude Code / Fable): all code, Slurm submission, artifact
  hashing, the research log, and the drafting reference `claims_and_numbers.md`.

**Decisions the log records as yours (use the ones you actually remember making):**
- 2026-08-27: corrected the agent's treatment of the 4B memory/throughput preflight
  as access-blocked; authorized the access-independent timing run.
- 2026-08-30: activated the fallback (self-train skip0 from M0) after 48+ hours of
  403s on the gated checkpoints, rather than waiting or contacting the maintainer.
- 2026-08-31: scope freeze; cut sham-latent branch and StrongREJECT-313.
- Score-blind cap policy and the strict <5% rule; the K=0 necessity ablation; the
  decision to bound nonterminating cells rather than impute.
- `[YOU]` any others.

**What you checked with your own eyes:**
- Stage-1 blind coherence: 40 outputs (K=0, K=2, and an M0 control), 0 to 2
  rubric, packet hash frozen before rating. Result 2.0 / 1.9 / 1.3.
- Stage-2 and stage-3 blind coherence: 20 outputs each.
- Endpoint 24-row sealed blind behaviour audit: labels committed before the key
  was opened. Found the one harmful compliance, two all-Chinese outputs, and one
  CoT partial compliance (assists in evading detection).
- `[YOU]` This session: caught the reviewing agent conflating the sham control with
  an implementation-correctness control, and misattributing the power shortfall.
  Say it in one sentence; it is direct evidence of staying in control.
- `[YOU]` Recompute two numbers yourself before submitting (commands at the end of
  this file) and say which ones.

**What you did not check, and how surprised you would be by a major error:**
`[YOU]` fill the surprise column honestly. My risk ranking, highest first:
1. Coconut wrapper correctness (latent insertion, loss masking, curriculum).
   Highest risk. No reference agreement. A bug here changes conclusion 1 from
   "the method collapsed" to "my implementation may be flawed."
2. Cap / truncation bookkeeping and nontermination classification across the
   512 / 1,024 / 5,120 / 16,000 ladder. Medium. Many moving parts, but the M0-at-512
   artifact shows the guard does fire.
3. GSM8K answer extraction. Low after today's check (0 unparsed rows at K=6).
4. Paired bootstrap and bound arithmetic. Low. Simple code, and the bounds can be
   recomputed by hand: 2 nonterminating rows × (1/60) = 0.033 width, matches
   [0.051, 0.084].
5. Judge pipeline. Low. Pinned upstream StrongREJECT evaluator, final answer only.

**Anti-slop process:** `[YOU]` describe what you actually did. If it was "I did not
read most of the code," say so and say what you read instead.

## Form Q8 to Q11. Personal

`[YOU]` entirely. Prior mech interp experience; 1 to 3 pieces of evidence (not this
project, ~100 words); why Neel's stream; likelihood of joining Sept 28 to Oct 30.

---

## Executive summary outline (max 600 words, 1 to 3 pages, graphs included)

1. One-paragraph setup: matched benign GSM8K training in two implemented regimes,
   one 4B thinking model, pre-registered safety comparison at n=60.
2. **Figure 1** (capability) with two-sentence caption.
3. Headline bullets, in this order: collapse (61 points); K=0 termination/accuracy
   ablation; safety comparison inconclusive and confounded; transient u1 bump under
   two seeds. Give the CJK-heavy-output finding one short descriptive sentence.
   Keep weight-diff and logit-lens analyses out of the executive summary unless
   space remains after the core validity story.
4. **Figure 2** (existing safety trajectory plot).
5. Three-bullet limitations: no implementation positive control; underpowered
   primary not caught prospectively; one seed, prompt-level CIs.
6. One line on hours and on the division of labour with agents.
7. In the long research document, immediately after the executive summary: the
   random examples section (`random_examples.md`, seed 2026, 5 prompts × 3
   conditions, plus the one harmful compliance labelled as non-random). Do not
   attach its 3,631 words to a form field with a 600-word limit.

## Doc outline (reframed; maps to claims file sections)

1. Question and why. 2. Design (claims §2). 3. Capability and termination result
(claims §3, lead). 4. Safety: registered primary, inconclusive and confounded
(claims §4). 5. Trajectory and the u1 bump (claims §5). 6. Human audit and
measurement limits (claims §6). 7. Post-hoc clues (claims §7). 8. What this does and
does not say about latent reasoning; relation to self-jailbreaking (claims §8). 9.
Limitations and the experiments that would actually settle it: reference-checkpoint
positive control, sham pause-token branch, alternate c_thought, more seeds, a
powered safety assay (claims §9).

## Figures and tables

- Figure 1: `writeup/figures/fig1_capability.png` (new; script alongside it).
- Figure 2: `artifacts/discovery/results/dense_safety/trajectory/trajectory.png`
  (existing).
- Appendix figure: `artifacts/discovery/results/posthoc_layerwise_weight_diff/layerwise_weight_updates.png` (existing).
- Table: endpoint safety (M0 / CoT / Coconut / delta / CI / margin).
- Table: audit counts (from `audit_unblinded_summary.json`).
- Table: logit-lens overlaps (claims §Claim 8 table).

## Hand-verification commands for you to run and cite

Endpoint safety means (should print 0.0880, 0.0934, 0.1052):

```bash
cd ~/Downloads/mats_latent_safety
for c in m0 cot_u3 coco_u3_k6; do python3 -c "
import json; d=json.load(open('artifacts/discovery/results/official_safety/scores/$c.json'))
r=[v for v in d.values() if isinstance(v,list)][0]; print('$c', len(r), sum(x['score'] for x in r)/len(r))"; done
```

GSM8K correct counts (should print 184/200 CoT and 62/200 Coconut K=6):

```bash
grep -c '"correct": true' artifacts/discovery/results/native_gsm8k_controls/cot_u3/generations.jsonl
python3 -c "
import json; r=[json.loads(l) for l in open('artifacts/discovery/results/fallback_4b_skip0/trajectory_stage3_gsm8k_cap1024/gsm8k.jsonl')]
print({k: sum(1 for x in r if str(x['k'])==k and x['correct'] in (True,'True')) for k in ('0','6')})"
```

Then open two Coconut K=6 GSM8K rows and two safety rows in the raw JSONL and read
them. Say in Q7 that you did.
