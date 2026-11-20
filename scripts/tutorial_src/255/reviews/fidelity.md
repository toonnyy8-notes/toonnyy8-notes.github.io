# Fidelity review (255, Goldfish)

Scripts: check_closed.py and check_toy.py run and match section 6 (C1, C2, C4, S8 shares, C3 TV rows). The single-step table (chainSeed 3, L=6) is not printed by the script's main block, which uses B=2 and chainSeed 1. I re-ran `single_steps(3,6,6,1,4.0,0.25,3,8)` and it matches the table. No numeric errors found. Issues below are about claims, not arithmetic.

1. **P1 (wrong definition).** Hero card 1: 「base model 是沒有經過 RL 後訓練的語言模型」. Paper Sec. 5.1 uses Phi-3.5-mini-instruct as a "base model". An instruct model is already post-trained (Abdin 2024), and the paper does not say how. Fix: 「作為取樣對象的預訓練或既有模型（Phi-3.5-mini-instruct 為 instruct 版）」. Table 1 "Base" row: note it for Phi.

2. **P1 (missing assumption, MH).** Step 6: 「p_prop 可為任意 LLM 與取樣法」 and 「任兩序列可互達（不可約、非週期）」. Paper Sec. 4.2 asserts this with Def. 1 only. Irreducibility needs p_prop support to cover the support of p^α, and q(x|x′)>0 iff q(x′|x)>0, otherwise A in Eq. 9 is undefined. Greedy or top-k proposals break it. Aperiodicity comes from the possible self-loop, not from "restart from the beginning". Fix: state "p_prop 須對 p^α 的支撐有正機率（τ=1/α 的全詞彙取樣滿足）", mark it 〔本頁推導〕, and add a note to LO6.

3. **P1 (overstated causal wording).** Step 10, 「增益多半來自低溫取樣」 and LO10 「8 格的增益有 62%–77% 已由低溫取樣達成」. The ratio is computed from single runs with no error bars. Several denominators are tiny: Qwen2.5-7B GPQA power−base = 0.040 (8 of 198), low-temp−base = 0.025 (about 5 questions). Power sampling's proposal is itself a τ=1/α sampler, so this is a decomposition of the table, not an ablation. Fix: 「表中差值的占比」, say it is not a causal attribution, and show the question counts for GPQA.

4. **P1 (wrong ratio in reference answer).** LO9 and C4 line: 「K=⌈T/B⌉=4；近似用 K²/2=6.26」. 4²/2=8. 6.26 is (679/192)²/2, using the non-integer T/B=3.537. Fix: 「近似相當於 (T/B)²/2=6.26，精確和用 K(K+1)/2=10（K=4）」.

5. **P1 (weak support for the result).** Step 7 / C3 observation 「N=50 時接近 0」 and the B-effect teaching. The toy shows V=2 N=50 TV=0.0144 and B=6 N=50 TV=0.0081. At N=10, B=3 (0.0601) is below B=2 (0.0624), so TV is not monotone in B. The paper only says (Sec. 4.3) that larger B needs larger N_MCMC. Fix: say only "TV 隨 N 下降，下降速度依設定而異", and give no ordering of B.

6. **P1 (Fig. 5 / E5 incomplete).** The caption says 「strictly better than both GRPO and the base model」 and, in the same caption, 「our pass rate at high k matches the base model」. These contradict each other. The text says power "supersedes that of the base model until finally converging". Design E5 only flags the k=1 vs GRPO point. Fix: add that "strictly better than base" also fails at large k by the paper's own wording. R5 also assumes Fig. 5's k=1 equals Table 1, but the paper does not state this. Keep 「依 Table 1」 and say the figure's values are unread.

7. **P2 (C2 constraint).** C2 input: 「ε + ε′ ≤ 1」. Def. 2 defines ε as Σ_{x>t} p(x_0..x_T) for a fixed prefix x_{0:t−1}, a joint weight, not a conditional. The bound is ε+ε′ ≤ p(prefix). Fix: label as "前綴為空（Example 1）的情形".

8. **P2 (C2 readout).** 「冪分佈：選 x_t 的機率 0.561」 is the pairwise-normalized sigmoid, which assumes only two tokens. Prop. 3 compares weights only. Fix: label 「兩者之間選 x_t 的機率」.

9. **P2 (Hero card 3).** 「重新生成後半段」. In Alg. 1 the resample runs from m to the end of the current block (k+1)B, not to the end of the sequence. Fix: 「到目前區塊結尾」.

10. **P2 (N_MCMC sweep).** Step 10, 「α 與 N_MCMC 是在 MATH500 上掃出來的」. Sec. 5.3 says MATH500 for the α sweep. The Fig. 6 right panel and its caption name only "Qwen models", with no benchmark. Fix: 「N 掃描的基準論文未明寫（推測 MATH500）」, and add it to the S9 list.

11. **P2 (Fig. 4 confidence).** Step 10 folds confidence into the likelihood description. Paper Sec. 5.3 says power sampling and GRPO sit in "similarly high confidence regions" (right panel), while for likelihood GRPO is "heavily concentrated" and power sampling keeps "noticeable spread". Fix: split the two claims.

12. **P2 (A.2).** Design records only the HumanEval remark. A.2 also says power sampling is "near universally better" than GRPO and base for k>1 in all three domains, and α=1.67 "exhibits slightly better results at earlier k". Fix: record both and flag the sentence as the author's judgment, since no values are given. "直到 k=16" in the paper may mean "up to but not including 16"; quote it verbatim.

13. **P2 (labels).** Step 6 〔本頁推導〕 for (b) 「與 m 落在共同前綴內的哪裡無關」 is trivially true and could be read as a property of the algorithm. Add 「因共同前綴的 Z_t 相消」. Also add 「Z_t 化簡要求提議為未截斷（無 top-k、top-p）的 τ=1/α 取樣」; the paper does not say whether truncation is used.

14. **P2 (E3 wording).** The paper's Prop. 1 is stated for general p. Equality at t=T shows the proof's step is wrong, and the proposition is also false for special p (α=1, or a deterministic p). The claim holds only for generic p. Fix: add 「對一般 p 而言」.

Verified faithful: Eq. 1–15 forms; Example 1 values and E2; Prop. 3 and Eq. 14–15 (including α≥1 in the paper vs the design's need for N>1 and α>1); Alg. 1 lines, E4, E1; Table 1 transcription; 679, 671, 600, 8.84; Sec. 5.1 settings; Fig. 6 statements.
