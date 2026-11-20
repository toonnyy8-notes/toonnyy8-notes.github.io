# Grade: 119 LOPD acceptance

**Score: 12/12** (12 correct, 0 partial, 0 wrong)

## Per-LO verdict
- LO1 correct: student state, teacher with c_fix, prefix-level supervision, student-only inference all match.
- LO2 correct: frozen θ̄ + c_φ, updates θ and φ=(ψ,χ), no sg in Eq. 10, encoder LoRA off in teacher forward.
- LO3 correct: retrieval (J=3), LoRA encoder, 8-layer QFormer K=32, 96 latent tokens, torch.cat; also caught the Eq. 4 vs 7 mismatch.
- LO4 correct: reverse KL, teacher top-M=20 support, merged tail, truncated KL <= full KL, tail interior unsupervised.
- LO5 correct: no sg lets composer move teacher toward student; q=p minimum; marked as page derivation.
- LO6 correct: δ, A=2r-1, Δ, E[δ]=-KL<=0 with sampling condition; batch can be positive.
- LO7 correct: Lagrangian, anchor, β update, defaults.
- LO8 correct: q*, β=0 and constant-Q cases, flagged as idealized.
- LO9 correct: seven numbers, measured vs author claim vs unmeasured separated.
- LO10 correct: aggregate-column meaning, sub-column counterexample, problem counts (141/140 etc.), missing stats.
- LO11 correct: text-only points, 480 = 30% of 1,600, true for Skill-SD only, Table 1 vs Fig. 4 mismatch.
- LO12 correct: no text-retrieval baseline, Frozen φ₀ is not retrieval-only, top-M provenance contradiction.

## Stuck points
1. **Flag (minor).** Page line 37 (S1): "自蒸餾：同一個語言模型扮演兩個角色" is unqualified. Lines 109/114 (S3) say "同一個模型" but scope it to OPSD. The "not same weights" correction appears only in the summary (line 51) and S6 (line 276/278). A reader hits two unqualified "same model" statements before S6. Not an LO failure (reader answered LO2 correctly), but it is confusing.
2. **Flag.** Eq. 4 (K) vs Eq. 7 (J·K) and J vs n_ret are already noted on the page (S14); add a one-line pointer where Eq. 4 first appears.
3. **Flag (small).** sg placement (Eq. 3 yes, Eq. 10 no, Eq. 11 student term yes) is the most confusable item; add a three-row mini table in S6.
4. Already handled (page line 173 separates M=20 from top-k=20); no fix.
5. Already stated as unknown by the page (m=0, Eq. 12 vs Alg. 1); no fix.

## Fix list
1. S1 line 37: change to "同一個語言模型扮演兩個角色（這是 OPSD 的做法；LOPD 的老師權重是凍結的，見 06）".
2. S3 line 109 and 114: append "（OPSD 的情形；LOPD 的老師不同，見 06）" after each "同一個模型／同一模型".
3. S6: add a table "Eq. 3 老師 sg｜Eq. 10 老師無 sg｜Eq. 11 學生項 sg" with one-line reason for each.
4. S5, at Eq. 4: add "此處 K 為單筆軌跡的向量數；串接後共 J·K=96（Eq. 7）；J 即 n_ret".
