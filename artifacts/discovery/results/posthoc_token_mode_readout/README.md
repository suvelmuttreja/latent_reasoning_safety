# Post-hoc token-mode latent readout

This exploratory appendix applies the model's native output head at each of the
six recurrent latent steps for the first five prompts in each pre-existing
frozen manifest. It makes decodability claims only—not faithfulness,
monitorability, or causal-mechanism claims.

The GSM8K readouts are strongly depth- and format-structured. The same top-1
token appears on all five prompts at depths 2 (`这张`), 3 (` the`), 4 (`这张`),
5 (` the`), and 6 (`这张`). Mean pairwise top-10 Jaccard similarity is 0.873,
0.686, 1.000, 0.542, and 0.873 at those depths, respectively. Depth 1 instead
projects generic reasoning-boundary tokens such as `thinking`, `Image`, and
`problem`.

StrongREJECT is less invariant: mean pairwise top-10 Jaccard ranges from 0.092
to 0.522 across depths. Its projections are still dominated by generic
reasoning, format, punctuation, and syntactic continuations; this ten-prompt
slice shows no clear natural-language harmful plan or refusal trajectory.

The appropriate conclusion is narrow: these native token projections expose a
strong recurrent phase/scaffold signature, especially under the raw GSM8K
serialization, but do not provide an obvious textual rationale. Distributed
task information may still be present outside the top-token projection. The
complete table is retained in `token_mode_readout.md`; no prompt or depth was
selected based on interpretability.

## Period-two check

A post-hoc top-10 token-set analysis quantifies the visible odd/even cycle. For
GSM8K, mean within-prompt Jaccard overlap is 0.447 for same-parity depths and
0.000 for opposite-parity depths; lag-2 overlap is 0.474 versus 0.000 at lag 1.
For StrongREJECT, the corresponding values are 0.154 versus 0.036 and 0.222
versus 0.038; the same-parity advantage is positive on four of five individual
prompts. Thus both task slices have an aggregate descriptive period-two
signature, much stronger and more prompt-consistent for raw-format GSM8K.

This period matches `c_thought=2`, the curriculum setting that adds two latent
positions per stage, but the match is not causal evidence. Only one
`c_thought` value was trained, and GSM8K and StrongREJECT use different frozen
serializations. Establishing curriculum causation would require another
`c_thought` condition, which is outside this quick appendix.
