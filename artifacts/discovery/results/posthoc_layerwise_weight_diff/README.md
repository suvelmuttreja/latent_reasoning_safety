# Post-hoc endpoint weight-update comparison

This exploratory descriptive analysis compares each stage-3 endpoint with the
exact pinned M0 checkpoint. It is not a registered primary result and does not
localize the cause of any behavioral change.

Across all compared parameters, the relative L2 update is 0.473% for explicit
CoT and 0.438% for Coconut; the two full update vectors have cosine 0.427.
Coconut's update norm is smaller in every transformer block (0.756–0.969 times
the CoT norm). Update-direction cosine generally rises with depth, from 0.145
in layer 1 to 0.528 in layer 35. Shared embedding rows are the exception to the
norm ordering: 0.669% for CoT versus 0.716% for Coconut, with cosine 0.866.

This does not support an early-layer concentration account. It also does not
show that capability loss resides in the embeddings: norm structure alone is
non-causal and scale-dependent. The public M0 vocabulary matrix is padded to
151,936 rows, while training resized endpoints to 151,672; embedding and tied
head comparisons therefore use only the shared token-ID prefix. The public M0
index omits a separate tied `lm_head.weight`, so that alias is excluded.
