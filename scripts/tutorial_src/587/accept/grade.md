Score: 12/13

Correct: LO1-LO12 (numbers and claims match reference; LO6 omits "only at boundary equals the cap" but that is a side clause, and the core Eq.10/11/δ-2 answer is right).

LO13 - partial
- Right: B ratio > 1.2 -> zero gradient, small rise, policy concentrates on A; not attributable to shape; no-clip 0.395 vs RIC 0.370; DAPO-style bound 1.0 gives 0.505.
- Missing: the RIC half of the reference, "RIC makes the upper bound larger for low-probability tokens (p_B=0.01, delta=0.05 -> about 3.24)".
- Page states it: line 342 "RIC 讓低機率 token 的上界變大（pB=0.0..." (3.24 follows). Gap is in the answer.

Real beginner stuck points (from the novice list):
1. Factor 2 between Eq. 9 (cap delta/2) and Eq. 11 (delta absorbs 2) - real, page handles it in one sentence only.
2. "Distance" vs "ratio range" are different quantities (d_clip, d_geom vs |r-1|<eps) - real.
3. min picks upper bound for A>0, lower for A<0, only in prose - real (page line 172).
4. KL derivation Eq. 4-7 needs Fisher / second-order expansion background - real.
5. Mixed "paper" vs "本頁推導" labels, easy to misread derived results as paper claims - real. Unexplained meaning of double-clip upper bound 10 is a paper gap, not a page defect.
