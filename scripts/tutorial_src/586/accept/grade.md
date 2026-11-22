Score: 12/12

All 12 LOs correct against design.md reference answers; no partial or wrong, so no page fixes required.

Notes (minor, not scored down):
- LO1: omits "collapse in earlier layers" but the question only asks the stable range (4-16); page line 550 states the collapse.
- LO3: lists 4 conditions (adds "J_i,h_j evaluated at pretrained weights"); superset of reference (single token, differentiable, small dW). Correct.
- LO9: answer omits the "only constant-factor (sqrt2) equivalence, ordering can flip" limit, though the question does not ask it and the novice raised it under stuck points; page states it (lines 443, 527).
- LO10, LO12: all numbers match (0.891/0.050, 0.000/0.226; 85.7/71.2/79.9/69.7; 68.0 vs 55.0, Dowser 69.7/62.7; 77.7 vs 79.9, CIDEr 116.0->123.6). LO12's two readings of the question are both covered.
- LO4: 1.118/0.355 and 3.640/3.597 correct.

Real "where a beginner gets stuck" points from novice:
1. M=1 means update, not freeze; Fig. 2 plus sign vs Eq. 8 minus sign (page S8 common errors).
2. Sum of S is not an upper bound on true displacement (needs first order and |dW|<=|W|).
3. First-order failure depends on where pre-activation sits on GELU, not just |dW|.
4. Paper inconsistencies: squared (body) vs absolute (App. D) vs plain (Fig. 2) estimator; SPIDER L>20 statements. Real.
5. Unspecified details (which tensor f, token aggregation, per-layer vs global ranking, probe-time memory). Real; page already lists them.
