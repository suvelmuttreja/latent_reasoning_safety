# Executive summary

## Question and design

Does progressively replacing explicit chain-of-thought supervision with
recurrent latent-state reasoning change safety drift under otherwise matched
benign reasoning training? I trained explicit-CoT and skip-stage-0 Coconut
branches from the same pinned Qwen3-4B-Thinking base, with the same data order,
optimizer, effective batch, three-stage curriculum dose, and checkpoint/eval
schedule. The primary endpoint estimand was the paired StrongREJECT-small
difference between branches; M0 was retained as a descriptive shared-pretraining
anchor, not a correction constant.

## Main results

The endpoint safety comparison was inconclusive rather than equivalent.
Mean harmfulness was 0.09344 for explicit CoT and 0.10516 for Coconut
(`Coconut - CoT = +0.01172`; paired-prompt bootstrap 95% CI
`[-0.02278, 0.04594]`). The interval includes zero and is not contained in the
pre-registered ±0.01995 equivalence margin. M0 scored 0.08801 on the identical
60-prompt harness, so endpoint shifts from base were small: +0.00543 for CoT
and +0.01715 for Coconut. The supported conclusion is therefore no detectable
endpoint substrate difference at this assay's resolution, with equivalence not
established—not evidence that the branches are the same.

The explicit-CoT safety trajectory contained a transient stage-1 elevation:
M0 0.08801 → u1 0.12673 → u2 0.09488 → u3 0.09344. Thus endpoint-only
reporting would have hidden a +0.03872 stage-1 shift (paired-prompt bootstrap
95% CI [+0.00061, +0.07723]) that largely resolved by stage 2. I describe this
as transient drift detected in this run, not a seed-stable effect: the interval
captures prompt uncertainty only, three registered checkpoints were inspected
without a multiplicity adjustment, and each branch has one training seed. An
early format-shock explanation is a hypothesis, not an established mechanism.

Coconut-u1 is unavailable as a full-set scalar point estimate because
3/60 generations remained nonterminating at the maximum registered cap,
exactly failing the strict <5% guard. Coconut-u2 K=4 independently produced
3/60 nonterminations at its maximum registered 5,120-token cap (with K=0 at
16/60), again exactly failing the guard. These cells are retained as
partial-identification intervals: terminating rows receive the pinned judge
score, while every nontermination is assigned 0 for the lower bound and 1 for
the upper bound. They remain intervals with no point estimate—not completed-case
means or imputations. Fresh official generation yielded u1 `[0.05095, 0.08429]`
with 2/60 nonterminations and u2 `[0.08365, 0.13365]` with 3/60. U1's official
count fell below 5%, but this did not retroactively select a cap after its
score-blind calibration failure or authorize a point estimate. Coconut-u3 is
0.10516 and had no nontermination. The
appearance of termination failures at intermediate trained K but not at K=6 is
K-selective format dependence, although one run cannot identify its mechanism.

The largest effect was capability, not safety. On the same frozen 200 GSM8K
prompts under native chat, M0 accuracy was bounded at 87.5–89.5% because four
outputs truncated; explicit CoT-u3 reached 92.0% with zero truncations; natural
Coconut-u3 K=6 reached 31.0%. Coconut was therefore 56.5–58.5 percentage points
below M0 and 61 points below its matched CoT endpoint. This anchors the
capability collapse to the latent branch rather than to ordinary benign
reasoning SFT.

The registered fixed-weight K=0 versus K=6 test showed that latents became
load-bearing but deleterious. At the Coconut-u3 weights, K=0 had structural
nontermination on 13/60 safety prompts through every registered cap, so the
full-set safety scalar contrast is undefined without conditioning on
termination. On GSM8K, K=0 accuracy bounds were 49.5–59.5%, versus 31.0% at K=6,
and all 200 paired outputs changed. The valid result is the nontermination and
capability contrast; complete-case or imputed safety scoring is prohibited.

## Qualitative audit and interpretation

In the fixed condition-blind 12-prompt-per-branch audit, explicit CoT was
12/12 coherent; Coconut was 7/12 coherent, including one substantive harmful
compliance and two outputs unclassifiable for incoherence/language mismatch.
The automatic judge assigned the harmful output 0.332. Coconut's incoherent
responses did not have lower mean judge scores than coherent responses
(0.16098 versus 0.15522), so coherence-driven score deflation was a concern but
was not demonstrated in this small audit. The audit is qualitative (`n=12`),
and coherent-only scoring would condition on a post-treatment variable.

Taken together, this experiment did not instantiate a robust differential
safety-drift phenotype at the endpoint. It did find that this public skip0
schedule, as reproduced in our implementation, can make recurrent latents
strictly necessary while
severely degrading capability and coherence on a strong 4B thinking model, and
that safety trajectories can contain transient effects hidden by endpoint
means. These results do not establish that hidden-state recurrence itself is
causal, that latent reasoning is generally less monitorable, or that no safety
difference exists. Training-level uncertainty is also unresolved because each
branch has one seed; the paired-prompt interval is not a training-seed interval.
The public full and skip0 model cards report their recipes and describe uploaded
endpoints as converged but provide no GSM8K metric; their weights were
inaccessible. We therefore could not verify endpoint agreement with the public
run or distinguish skip0-specific collapse from full-Coconut collapse without
training a new full-stage0 branch.

## Research-process evidence

The project used pre-registered caps, an equivalence margin fixed before branch
results, score-blind cap selection, a blind human audit sealed before automatic
judging, immutable checkpoint/data hashes, progressive GitHub commits, and
scratch-to-home durability checks. Two apparent findings were rejected after
controls: a terse coherence harness artifact and the hypothesis that incoherent
outputs mechanically depressed judge means. Scheduler timeouts were handled by
exact manifest-prefix continuation; when coco-u2 exposed that its calibration
script lacked incremental persistence, the next job was held, a fail-closed
fsynced prefix cache was tested and committed, and only then was it released.
The negative results and nontermination bounds are reported rather than
salvage-scored.

## Positioning

Self-Jailbreaking establishes safety drift from benign reasoning post-training;
this project tests whether that phenotype changes under a matched latentization
regime. Ulterior Motives studies detection of deliberately backdoored or
misaligned Coconut states; this experiment instead measures emergent behavior
under benign training. Coconut/CODI and latent-interpretability work motivate
the fixed-weight necessity test, but the present evidence does not support a
general monitorability claim.
