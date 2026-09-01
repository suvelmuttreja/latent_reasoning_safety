| Condition | Branch | Stage | Status | Mean | Delta from M0 |
|---|---|---:|---|---:|---:|
| m0 | base | 0 | scored_complete | 0.088010 | +0.000000 |
| cot_u1 | explicit_cot | 1 | scored_complete | 0.126730 | +0.038720 |
| cot_u2 | explicit_cot | 2 | scored_complete | 0.094884 | +0.006874 |
| cot_u3 | explicit_cot | 3 | scored_complete | 0.093441 | +0.005432 |
| coco_u1_k2 | coconut_skip0 | 1 | partial_identification_bounds_no_point_estimate | [0.050953, 0.084286] | — |
| coco_u2_k4 | coconut_skip0 | 2 | partial_identification_bounds_no_point_estimate | [0.083646, 0.133646] | — |
| coco_u3_k6 | coconut_skip0 | 3 | scored_complete | 0.105162 | +0.017153 |

Failed-cap Coconut cells have no point estimate. Intervals assign each
nontermination the theoretical StrongREJECT minimum (0.0) or maximum (1.0),
not the maximum observed in this sample.

At stage 1, the CoT-minus-Coconut identified set is `[+0.042443, +0.075777]`.
The paired-prompt bootstrap 95% CIs are `[-0.021492, +0.099736]` for its lower
endpoint and `[+0.033114, +0.120059]` for its upper endpoint; the conservative
identified-set confidence region is `[-0.021492, +0.120059]`. These intervals
represent prompt uncertainty only, not training-seed uncertainty.
