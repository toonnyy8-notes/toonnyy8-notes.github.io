# Fidelity review (Goldfish) for tutorial 127

Verification: `python3 -I verify/check.py` runs. Its C1, C2, C3, C4 and F1 outputs match the section 5 acceptance table. I spot-checked Tables 1, 2, 4-7, Eq. 1-8, Alg. 1 and Sec. 4.1-4.3, 5.1-5.3 against paper.txt, and the numbers copied from them are correct. The issues below are what the scripts do not cover.

1. **P0, C2 observation text.** design.md: "右式會把機率推向被拒絕的輸出 5、6". check.py's own REINFORCE output has right-hand components for outputs 5 and 6 of -0.00945 and -0.00519. Here ΣA = +0.04, so the bias term is -(ΣA)π0(j), which pushes those outputs DOWN. The direction depends on the sign of ΣA. Fix: write "右式在被拒絕的輸出上多出 −(ΣA)π₀(j) 的偏差項；ΣA>0 時壓低、ΣA<0 時抬高".

2. **P1, LO1 and step 2 ("獎勵全為 0", "約 94% 的題目…沒有梯度").** App. B.3 gives the student reward as 120 (verified correct), 20 (boxed, wrong, answer appears in y_ans), 10 (boxed, wrong), 0 (no boxed). An all-wrong group therefore usually has unequal rewards, and the gradient is not 0. Sec. 1 and 4.1 only say "sparse, binary rewards". Fix: say that under the binary-reward idealisation the gradient is 0, and that in practice format shaping (R5) breaks the tie. Label the 94% figure as a binary-reward model. Change the LO1 reference answer to match.

3. **P1, C3b σ_MA = σ/√(m·g·r).** This treats all g·r student rewards in a step as independent. They share one Q_R (64 questions) and one baseline term Acc(π_S(Q_R)). That common noise is not reduced by r or g, only by m. Students trained on the same X_k also share a dataset effect. Greedy Acc on 64 questions is quantised in steps of 1/64 = 0.0156, which is above τ = 0.01. Fix: state these assumptions and note that 2.66e-4 is a lower bound. Add an optional shared-noise term.

4. **P1, C4 group "真實整理資料（上界參考）" and the recovery ratio.** Table 4 has HARP train(128) at 23.0, above MATH Full at 22.0. Table 6 has MATH train(128) at 14.6, above Full at 13.1. Full is therefore not an upper bound. The Olympiad recovery of 0.82 is the page's own extension, and the paper only gives 75% (MATH) and 50% (HARP). Fix: relabel the group "真實題目參照"; show the Olympiad ratio only with a note that its denominator is not a maximum.

5. **P1, step 10 and C.5 ("結構重於正確").** Correct-only has 82 questions and Well-posed has 144 (App. C.5). The improvement is confounded with 1.75× more synthetic data. It uses 3 seeds, on HARP only. The paper's own wording on why Intrinsic-T is worse is "potentially because of greater diversity" (Table 7 caption). Fix: add the size confound and the single-dataset caveat, and mark the diversity explanation as the author's speculation.

6. **P1, Prop. 1 and step 5.** The design omits the stated assumption π0(S) > 0. It also omits that Prop. 1 is for a reward function R(z) of z. In SOAR every rollout in X_k receives the shared dataset reward R_k. The proof only needs ΣA = 0, but the group over which the RLOO baseline is taken is not specified in the paper. C3a's "A_k = R_k − ⅓Σ_{l≠k}R_l" (a group of datasets) is therefore a design assumption and is not tagged. Fix: add π0(S) > 0; tag the A_k computation 〔本頁約定〕.

7. **P1, 1.1 and Hero card 3.** "用老師累積的題目訓練的學生 pass@32 為 18.9%，直接訓練難題為 9.6%". PQ trains on PQ plus the fail@128 train set (Sec. 4.3), so it gets 128-256 extra questions. Table 4 PQ pass@1 is 1.7 ± 1.0, so the 3.4× ratio is dominated by noise. Fix: write "PQ＋fail@128 訓練集" and attach the SD wherever a pass@1 ratio is quoted.

8. **P2, C4 "k 越大差距越大".** This holds only in percentage points (Sec. 5.1: "larger gains at higher k"). The ratio falls from 3.40 at k = 1 to 1.97 at k = 32 (check.py). Fix: say "百分點差距".

9. **P2, F1 and the shared p axis.** In C1, p belongs to the base student on hard problems. In F1 and the "可學習性峰值" marker, p belongs to the student on synthetic questions. Putting both on one axis implies they are the same quantity. Eq. 8 is itself monotone in s̄ for s̄ > 0. The peak at 0.106 comes only from the zero cut-off with 32 samples. Fix: use separate axes or captions.

10. **P2, step 1 "與 Table 4 的 7.5% 數量級相符".** This is nearly circular, because Table 4 base pass@1 = 0.3% already implies mean p ≈ 0.003. R3 says "不說論文支持單題 p 的模型", but the sentence reads as support. Fix: say "由 pass@1 = 0.3% 校準的示範值".

11. **P2, C3a demo values.** Per-student rewards of 0.008 and 0.023 are not multiples of 1/64, so a single student cannot produce them. Fix: use multiples of 1/64 (0.0156, 0.0313, 0.0469, ...) or note the 示範值 label explicitly.

12. **P2, header "v3，2026-06-30".** paper.txt prints "Date: July 1, 2026". Fix: cite the date printed on the paper, or confirm against the arXiv listing.

13. **P2, step 6 and R6 (no MA reset after promotion).** The design chooses not to reset the moving average. Pre-promotion high rewards then stay in the window and can trigger immediate re-promotion. This is exactly the dynamic that R1's toy found, and it conflicts with Fig. 19. Fix: say explicitly in S6 that this is a page convention that may differ from the paper's implementation.

14. **P2, step 8 "回收 HARP 0.51".** The paper's sentence "≈50%" refers to HARP subsets and PQ-HARP against full MATH, and the denominator is MATH Full (16.2). The text does not say this. Fix: write the denominator explicitly.

No wrong numbers found in the Tables 1, 2, 4-7 transcriptions or in the Eq. 1-8 references.
