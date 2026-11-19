# 447 Fidelity review (Goldfish)

Scripts: check.py, c3_alpha.py and mc.py were run on a copy. All outputs are byte-identical to the stored *_out.txt and to the section 5 tables. Table 1-3 numbers, Appendix C values, Alg. 1 line numbers (8, 10), Eq. 1-3 and the DDPM formulas all match the paper. The issues below are what the scripts do not cover.

1. P0. Section 1.4 "常見錯誤 2": 「只在最後替換會得到不從目前狀態出發的計畫」. A final replacement at i=0 forces τ_{s0}=s, so the plan does start at s. What it loses is consistency: the remaining columns were denoised without the constraint, so the plan jumps at column 0. The paper (Sec. 3.3) only says to replace "after all diffusion timesteps i∈{0,…,N}" and gives no failure mode. Fix: say "起點與其餘計畫不連貫", mark it 〔本頁推論〕, or drop it.

2. P1. C3 and LO10: 「αβ̃ᵢ>2 的步驟會放大偏離」, 「這是 C3 發散現象的來源」. The page's own figures contradict this. At α=10 (N=20) seven steps have αβ̃>2 (up to 9.93) and it does not diverge (end 1.30). At α=15.8 ten steps exceed 2, and it diverges at step 13. Divergence comes from the cumulative gain, not from a single step above 2. Fix: say 「αβ̃>2 是單步放大的必要條件；是否發散取決於累積增益（本頁玩具 α≥10^1.2）」, and drop 「來源」.

3. P1. Section 1.4 common mistake 3 and C3 observation (2): 「把 α 當成溫度」, 「α 不是溫度」. The paper says nothing about temperature. Its target is p(τ)exp(J), which is α=1 (Sec. 3.2); α is an added scale in Alg. 1. The tilt p(τ)exp(αJ) is the usual classifier-guidance reading (Dhariwal & Nichol, which the paper cites). The toy's mismatch at α=1 (1.826 vs 1.007) is Eq. 3's Gaussian approximation failing, not proof that α is not a temperature. Fix: say 「Eq. 3 在大 α 下近似變差」, and drop the "not a temperature" claim.

4. P1. C3 uses the true J evaluated at μ, but the paper's Eq. 3 needs p(O|τ^i) to be smooth in the noisy τ^i, which is why J_φ is trained on noisy samples. The overshoot and divergence are therefore partly caused by the simplification. R7 says the version with E[J(τ⁰)|τ^i] was "數值不穩定". For a quadratic J and a Gaussian prior that quantity is closed-form and smoother, so the claim looks wrong or untested. Fix: either report that variant, or state that α-overshoot may be a toy artifact and not a property of Diffuser.

5. P1. C3 default 「α=0.1，論文值」 and 「α≥15.8 發散」. The paper's α scales a learned, normalized J_φ (App. C 5, 7). The toy's J is −½Σ‖s−c‖² summed over H=32 steps, so the units are different. Fix: do not call α=0.1 "論文值" in the toy. Add 「α 的數值與 J 的尺度綁定，不可跨模型比較」.

6. P1. Section 1.4 step 6 〔本頁推導〕: 「N 增加時差距縮小但不消失」. The data cover only N≤100 (jump 0.599, 0.168, 0.073). At N=100 the jump is already below the prior's typical step (0.113). "Does not vanish" is an extrapolation. For the replacement scheme at N→∞ the paper gives nothing. Fix: say 「在 N≤100 內縮小，未見歸零」. In C2, report the jump relative to 0.113.

7. P1. C1 observation (2): 「換 H…同一個先驗直接產生不同長度」. The toy defines a different analytic distribution for each H (a new DCT basis, and step_sd 0.084 at H=16 versus 0.113 at H=64). This is not the same trained fully-convolutional model (Sec. 4, Fig. 3c). Fix: say 「示範的是公式對任意 H 有定義；不是同一組權重」.

8. P1. Section 4.0 and S3: Q is pentadiagonal "與時間局部呼應". The ideal denoiser used (ε̂_k=√(1−ᾱ)τ̂_k/(ᾱλ_k+1−ᾱ), dense in time) has a global receptive field at high noise. The toy therefore does not show Sec. 3.1 "each denoising step only enforces local consistency". Fix: either state that the toy denoiser is non-local, or remove the pairing with locality.

9. P1. Section 1.4 step 5 〔本頁推導〕 for g=∇J(μ): it names two missing assumptions (conditional independence of O_t, r≤0). It omits that the paper's chain "∇log p(O|τ)|_{τ=μ} = Σ∇r|_{μ_t}" replaces the noisy τ^i by the denoised mean μ and p(O|τ^i) by the clean-trajectory reward. That is a further approximation, not an identity. Also, a shift in r does not change g, and a scale does. Fix: add 「p(O|τ^i)≈p(O|μ)」 as a third condition.

10. P2. LO10 reference answer puts the toy's 「α≥10^{1.2} 發散」 in the same sentence as the paper's 「α=0.1、0.0001」. This violates the page's own rule in 6.5. Fix: split into two sentences, and tag the toy numbers 「本頁玩具」.

11. P2. LO11 and S7: 「每次只執行計畫第一個動作再重新規劃」. Sec. 5.1 executes the sampled Maze2D plan "as an open-loop plan". Sec. 5.4 also says "execute plans open loop" while describing per-step replanning, so the paper is inconsistent. Fix: LO11 should say "locomotion (Alg. 1)" and note that Maze2D is open loop.

12. P2. Section 1.3 「貢獻（Sec. 1）」: the paper has no numbered contribution list (only "The core contribution…" and bold property headings). Contribution 4 (experiments) is the designer's synthesis. Fix: relabel as 「本頁整理」.

13. P2. mc.py: the design says the difference is 「在 Monte Carlo 標準誤內」, but the output shows 0.0076 vs s.e. 0.0064 and 0.0070 vs 0.0066. These are maxima over 128 coordinates, so a max of ~2 s.e. is expected. The statement is literally false. Fix: say 「與 128 個座標最大值的預期範圍一致」.

14. P2. S9 Table 2 excerpt shows only the Medium-Expert rows, where Diffuser is strongest. It omits e.g. Medium-Replay HalfCheetah 37.7 (below CQL 45.5, TT 41.9, MOPO 53.1) and Medium-Replay Walker2d 70.6 (below TT 82.6). Fix: add one sentence about the weaker rows or the "within 5% of max" bold rule.

15. P2. Section 1.4 common mistake 4: asserts 「用無噪軌跡訓練 J_φ」 is wrong, while R7 admits the paper's target is unspecified (τ^i vs the corresponding τ⁰ return). The paper only says "cumulative rewards of trajectory samples τ^i". Fix: soften to 「論文以帶噪樣本 τ^i 為輸入」. Also note that the paper's h(τ)=+∞ for c_t=s_t is a notational convenience, not a literal density.
