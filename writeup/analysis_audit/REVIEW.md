# Audit of existing analyses — 5 September 2026

The core arithmetic survives: explicit CoT scores 184/200 (92%) and Coconut K=6 scores 62/200 (31%). The endpoint safety difference and its paired bootstrap reproduce. Several secondary claims need correction, and the safety measurement has a substantial visibility limitation. These data support a documented, capability-degraded reproduction with useful diagnostics; they do not establish a substrate effect on safety or validate the implementation.

This review covers the capability trajectory and controls, K ablations, termination diagnostics, safety endpoints and trajectory, missingness bounds, seed follow-up, equivalence calibration, human audits, output anatomy, weight differences, and token readouts. It is an analysis/source audit, not an independent training reproduction or exhaustive model-code certification.

## Corrections that matter before submission

### 1. K=0's missing-outcome bounds were calculated incorrectly

At u3 K=0, 99 answers parse as correct at the token cutoff, but **three of those are among the 20 unfinished generations**. There are 96 correct terminated answers. If all unfinished completed outcomes are unknown, the bounds are **96/200 to 116/200 = 48–58%**, not 49.5–59.5%. The old calculation counted those three as known correct and then counted them again as potentially correct missing outcomes.

The cutoff-parser accuracy of 49.5% is a legitimate separate quantity. It must not be called the all-unfinished-wrong bound. The corrected K=6 minus K=0 range is **−27 to −17 percentage points**. The substantive result survives.

| Condition | Correct at cutoff | Correct among unfinished | Unfinished | Unknown-outcome bounds |
|---|---:|---:|---:|---:|
| Coconut u1 K=0 | 152 | 2 | 8 | 75–79% |
| Coconut u2 K=0 | 139 | 1 | 5 | 69–71.5% |
| Coconut u3 K=0 | 99 | 3 | 20 | 48–58% |
| M0 | 175 | 0 | 4 | 87.5–89.5% |

The u3 exceptions are `gsm8k-test-0471`, `0507`, and `1292`. The same bug exists in `scripts/evaluate_native_gsm8k_control.py`'s summary construction. The independent audit script corrects the derived results without changing the generation script whose hash is used by the ongoing early-CoT evaluation. Check the new CoT summaries for this issue when they arrive.

Paired outcomes strengthen the descriptive ablation result: among 180 jointly terminating items, K=0 alone is correct on 62 and K=6 alone on 26; 34 are both correct and 58 both wrong. This subset is selected by termination, so retain the full-set bounds as the primary description.

### 2. The judge usually sees only a suffix of the answer

The configured evaluator truncates answers to **512 judge-tokenizer tokens, retaining the rightmost tokens**. A read-only CPU check using the configured tokenizer snapshot found:

| Endpoint | Answers longer than 512 judge tokens | Empty parsed final answers |
|---|---:|---:|
| M0 | 48/60 | 0/60 |
| CoT u3 | 41/60 | 0/60 |
| Coconut u3 K=6 | 38/60 | 5/60 |

The five Coconut empties terminate at EOS without a closing thinking delimiter and have no parsed final answer. They were still scored. Thus generation termination and usable scored-answer availability are different checks. Keep the registered scores, but describe them as the output of this configured evaluator; they are not a validated assessment of every full response. Empty answers should be co-reported, not silently dropped or assigned invented harmfulness values.

The [pinned upstream implementation](https://raw.githubusercontent.com/dsbowen/strong_reject/7a551d5b440ec7b75d4f6f5bb7c1719965b76b47/strong_reject/evaluate.py) also loads model and tokenizer without a revision argument. The local scorer checks that the configured revision exists **after** scoring; that does not prove it loaded that revision. Kickoff cache metadata provides supporting evidence of the intended snapshot, but strict model-load provenance is weaker than the word “pinned” implies. There is no evidence here that a different judge was actually used.

**Best remaining safety analysis:** score these same cached endpoint answers with an explicitly pinned judge and a window large enough for every answer, preserving the original scoring rule otherwise. Report it as a post-hoc window sensitivity with paired contrasts and leave the registered result intact. Inspect the five empty cases separately. No new target-model generation is needed. A larger window is still not human validation of multilingual or corrupted answers.

### 3. The language subgroup interpretation used the wrong field

The output-anatomy analysis splits on CJK characters in **reasoning**, whereas the judge receives the **final answer**. Its 12-versus-48 groups are valid descriptive reasoning-field groups, but the complement is not a validated “English” category.

Using the actual scored field gives seven final answers above the 30% CJK threshold. Their Coconut mean is 0.0411 and their paired Coconut-minus-CoT difference is **+0.0323**; the other 53 have mean 0.1136 and paired difference **+0.0090**. The earlier reasoning-field split gave +0.0024 versus +0.0140. Consequently, “Coconut's excess harmfulness comes from its English outputs” is not supported by that analysis.

Neither split estimates a causal language effect: language is an output of the treatment, groups are small, and the threshold is a heuristic. Keep the 12/60 reasoning-field and 7/60 final-answer counts; drop the explanation of the safety gap. Text-language observations also do not show the language of latent computation.

### 4. The overall weight update mixes training with vocabulary initialization

The comparison keeps the common token-ID prefix of vocabulary matrices. Added marker tokens can occupy rows that were previously padding inside the public model's matrix; those rows are reinitialized during model construction. Excluding only the extra matrix tail does not establish that all initialization differences were removed. Do not present the overall embedding/head-inclusive delta as pure training movement without constructing the actual initialized starting model or excluding the marker rows.

A conservative recomputation using the **36 transformer blocks only** gives relative update norms **0.4563% for CoT and 0.4105% for Coconut**, and update cosine **0.3416**. Coconut's update remains smaller in every block. The descriptive claim of distributed, similarly sized updates survives. Weight norms neither locate a causal failure nor exonerate the implementation; comparisons across differently parameterized modules are particularly weak for this purpose.

### 5. Training-format and latent-computation descriptions need tightening

Both branches' actual training pipeline tokenizes raw question, rationale, and `###` answer text. The explicit branch was **not trained in native thinking-chat format**. M0/CoT capability evaluation uses native chat; Coconut capability evaluation uses its raw scaffold. This evaluates their intended inference regimes but adds a format difference to the comparison. Matching data, order, update count and optimizer remains useful; it does not match supervised tokens or computation.

The local accumulation objective weights by supervised token count. That is a coherent choice, but the [pinned public training loop](https://raw.githubusercontent.com/facebookresearch/coconut/27273cb8cca4bb763c041a63b036d0c3b7cbbb48/run.py) averages microbatch losses across accumulation steps. Different example lengths can change weighting. “Expectation-preserving deviation” is too strong.

The curriculum removes up to three written reasoning steps at the final stage; examples with more steps retain written supervision. A median six-token K=6 output supports “short, often answer-only outputs.” It does not prove that all rationale supervision was replaced or that the model performs its arithmetic in latent space. All 200 numeric parses rule out missing extraction, not every extraction error or a guessing strategy.

## Analyses whose main choices hold up

| Analysis | Audit verdict and interpretation ceiling |
|---|---|
| Endpoint safety means and paired bootstrap | All six scored-condition means reproduce. Endpoint difference is +0.011721, 95% interval [−0.022783, +0.045938]. Pairing by prompt is appropriate. This is prompt uncertainty conditional on these trained models and outputs. |
| Early Coconut safety missingness bounds | Recomputed lower/upper score bounds agree. Treating missing scores as ranging over [0,1] is transparent; averaging only completed prompts would change the target. Bootstrap regions remain approximate, not finite-sample guarantees. |
| CoT trajectory and second seed | Seed-42 u1 minus M0 is +0.03872 [0.00061, 0.07723]; seed-43 is +0.02557 [−0.00924, 0.06129]. Direction repeats; a statistically established effect across training seeds does not. Three timepoints add multiplicity. Later scores near baseline do not establish equivalence or a true reversal. |
| Equivalence decision | Correctly inconclusive: the endpoint interval crosses zero and both margin boundaries. The frozen margin avoids moving the goalposts, but its 12-prompt format anchor is noisy and includes capped generations. It is weakly calibrated as a substantively meaningful safety tolerance. |
| Sample-size discussion | Roughly 3×/17× information calculations are retrospective interval-width scaling under assumptions, not prospective sample sizes for specified statistical power. Do not label them a completed power analysis. |
| Human coherence and safety reviews | Separate behavior and coherence labels explain the apparently non-summing endpoint columns. The benign stage packets do not validate all GSM8K outputs. Small samples and disclosed blinding leaks limit inference. The incoherence subgroup scores do not validate the judge. |
| Length/score correlation | Reproduces at n=60: M0 r=0.392, CoT 0.440, Coconut 0.344. Prefer these to the old M0 n=12 r=0.71. Correlation alone cannot distinguish judge length bias from real differences in answer content. |
| Termination diagnostics | Repeated suffixes are useful evidence of decoding degeneration. The cycle test retokenizes decoded strings; say “repeating patterns in retokenized output,” not exact original generated-token cycles or proven infinite loops. K=0 changes positions and available computation and is out of distribution. |
| Token-mode readout | The native-head projections and Jaccard arithmetic are reasonable descriptive diagnostics. Top-1/3/5/10 and exclusion of the first boundary slot all preserve greater same-parity overlap in both tasks. This supports a robust projection pattern, not faithful latent thoughts, identical hidden states, or absence of information. |

The readout sample is five prompts per task from the beginning of each frozen manifest, not a broad random sample; task comparisons also differ in serialization. For GSM8K, top-10 same-parity overlap changes from 0.447 to 0.665 when excluding the first slot, while opposite-parity overlap remains zero. For safety it changes from 0.154 versus 0.036 to 0.222 versus 0.049. Pairwise comparisons are dependent within a prompt and should not be treated as independent observations for significance.

The numeric parser has an additional sensitivity: four M0 answers marked incorrect differ from the reference only by decimal formatting (e.g. `1.00` versus `1`). Numeric equality would raise M0 cutoff accuracy from 87.5% to 89.5%; no such disagreements occur in the audited CoT/Coconut conditions. Retain the registered parser as primary and disclose this sensitivity. It does not change the 61-point endpoint gap.

## Literature and submission priorities

The [original Coconut paper](https://arxiv.org/html/2412.06769) reports GPT-2 GSM8K CoT 42.9%, Coconut 34.1%, and no-CoT 16.5%; its pause-as-thought result is 24.1%. Near-equality between pause and Coconut on another benchmark is not evidence of equality on GSM8K. The paper's curriculum, training dose, model and selection differ from this run. Its smaller capability deficit neither explains this deficit nor validates the local implementation.

The [Self-Jailbreaking paper](https://arxiv.org/abs/2510.20956) is by **Zheng-Xin Yong and Stephen H. Bach**, not Betley et al. Cite the StrongREJECT repository as a repository, or its paper as Souly et al.

Before submission: correct the bounds and language inference, state the judge window/empties, state the training-format distinction, and use the block-only weight numbers if retaining that section. Incorporate early-CoT capability only once the raw outputs are complete; current placeholders and past-tense “were evaluated” are not results. Targeted corrections have been applied to `writeup/FULL_WRITEUP.md`, `writeup/FULL_WRITEUP_revised.md`, `writeup/claims_and_numbers.md`, the application answer draft, and the current filled/form/summary drafts, but any already exported Word/PDF copy and older drafts must be regenerated or checked separately.

If there is time for one further experiment, the cached-answer judge-window sensitivity has the best direct connection to the safety claim. A raw-format CoT capability control would isolate part of the inference-format difference, but is secondary to the already running earlier-stage controls. A sham branch, activation intervention, or full positive-control reproduction could be valuable later; none is a reliable last-minute substitute for implementation validation. Avoid adding a large exploratory analysis whose failure to finish would consume submission time.

## Reproduction and artifact boundaries

- Run `python3 scripts/audit_existing_analyses.py` to recreate `recomputed_checks.json` from 19 saved source files. It records their SHA-256 hashes and checks prompt pairing and the registered endpoint interval.
- `corrected_capability.png` and its plotting script show corrected K=0 ranges at all three stages.
- Run `python3 -m pytest tests/test_analysis_audit.py` for four regression checks, including correct parses among unfinished generations.
- `judge_visibility.json` records every endpoint's per-prompt token count using the configured judge tokenizer snapshot on Discovery. These are tokenizer visibility measurements, not a new judging run. `scripts/audit_judge_visibility.py` provides the corresponding read-only computation.
- Frozen generations and original score files remain the evidence; these outputs are a dated post-hoc audit. Ongoing generation/scoring code has not been changed by this audit.
