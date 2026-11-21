# Round 1 reviews (fidelity + pedagogy, merged)

Fidelity: verify/check.py output matches §5 (C1 sweep, C2 τ, C3 fit, C4 step counts); Eq. 4 = Eq. 10 re-derived; Tables 1–8, 11, 12, App. A, Eq. 9–16, Fig. 1d multipliers spot-checked correct. No wrong numbers.

## P0

1. **C2 「校準 τ」 contradicts its expected output (pedagogy; fidelity #3).** Spec: bisection finds the smallest τ with f_h ≥ ρ, but expected "τ≈0.7423, f_h≈0.496" (< 0.5). Slider step 0.005 cannot represent 0.7423; §5 calibration rows are computed at τ=0.74; selfTest range [0.742,0.743] ambiguous. Fix: define which bisection end is returned, whether the slider snaps, align §5. Also tag the algorithm 推定: App. A.3 calibrates sequentially from outer to inner stage on 1,024 noisy samples "to match the target aggregate compression ratios"; the single-stage bisection is the designer's own.
2. **C2 cross-language PRNG (pedagogy).** K_b must match Python exactly; last-digit Math.cos/log differences near τ=0.80 can change counts; Box-Muller has no guard for u1=0. Fix: embed the 4×255 scores directly (~6 KB) and drop the PRNG.

## P1

3. **Simple vs complex images (fidelity).** 「簡單影像可以用很少群組、複雜影像用較多（Sec. 3.3）」 and C2 "簡單樣本只保留 14%、複雜樣本保留 89%": Sec. 3.3 only says counts "vary with image content and diffusion timestep"; no per-image count vs complexity is reported. The 14%/89% spread comes from the invented c_b. Fix: cite Sec. 1/3.3 only for "counts vary"; tag the reading 推定/illustration; label the C2 spread as an artifact of c_b.
4. **C1 q_i = k_i, S_ii = 1 (fidelity).** Eq. 2 symmetrizes because q ≠ k; real S_ii = q_iᵀk_i; e_ig in Eq. 9 includes the self term for singleton groups. Fix: state S_ii = 1 is a toy artifact; one sentence on why the paper symmetrizes.
5. **C1 uses S5 concepts early (pedagogy).** Eq. 11 軟群組數, T_b, sigmoid, observation 5 appear before S5 teaches the compression loss. Fix: hide K̃ and T_b under an "advanced" disclosure in C1, reveal in S5.
6. **LO4 non-local grouping has no widget (pedagogy).** C1 has angles only, no spatial position. Fix: add a 2×3 patch-grid view where same-group tokens are non-adjacent.
7. **Untaught prerequisites / density (pedagogy).**
   - S3 packs Eq. 2, Eq. 8, tree, τ, monotonicity and two comparisons; cosine similarity, unit vectors, dendrogram, hierarchical clustering, softmax temperature, sigmoid untaught. Fix: start with a 3-token worked merge example; move monotonicity and ToMe/H-Net contrast into collapsibles.
   - S1: x_t, t direction, x-prediction, Eq. 12 with max(1−t, a_min)² unexplained. Fix: Eq. 12 in a collapsible background block.
   - S2 uses "router", "W_1, W_2", "分組", "跳接" before definition. Fix: label the figure only "跳接"; drop "router" or define it.
   - S4 overloaded (Eq. 9/3/6/7, gradient path, condition tokens, Table 3, C1). Fix: split S4/S4b or expand; keep one writing of Eq. 3 vs 6/7.
8. **C4 time direction and LO11 (pedagogy).** t=1 is clean, so "t ≥ 0.85" is the low-noise end; LO11 is about r_c/r_u changing group counts but C4 only plots the time grid. Fix: label axis ends 「噪聲多／近乾淨」; add a group-count-multiplier vs t step plot for Eq. 15.
9. **C1 interaction spec (pedagogy).** No setPointerCapture, touch-action:none, keyboard arrows, 0/359 wraparound and rounding. Fix: spec these; disc endpoint role="slider" with arrows (±1°, Shift ±10°).
10. **390px layout (pedagogy).** C1 has four panels (disc, 6×6 heatmap, tree, 6×K matrix). Fix: single column below 600px with minimum panel sizes; add Eq. 3 and Eq. 9 to the equation-split list.
11. **Dynamic KaTeX (pedagogy).** Don't re-run renderMathInElement on drag input. Render TeX labels once; update numbers as text; live region carries one summary line.
12. **Color / dark mode (pedagogy).** Grayscale heatmap over S ∈ [−1,1] has no dark mapping, negatives hard to read, numbers off by default ⇒ color-only. C2 histogram with four samples described only by the τ line. Fix: diverging scale, numbers on by default; line styles or small multiples; define the dark-mode scale.

## P2

13. 「JiT 有 32 個 in-context class token」 is not in the paper (App. A.4 only says "without the input bottleneck used in JiT"; Table 6: adaLN variants add class-context tokens after the fourth block; JiT-B in-context 18 blocks = 39.3 GFLOPs). Delete "32" or cite JiT.
14. C3 「11 點最大誤差 0.25」 is the in-sample residual of a 3-parameter fit; write 「擬合殘差」, never 驗證/預測; say the model is not an independent prediction (the JiT-like 25.08 vs 25.2 match is a coincidence).
15. R4 「摘要寫…約一半更新次數」: the abstract says only "approximately 40% of its cumulative training FLOPs"; "approximately half as many optimizer updates" is Sec. 1; Fig. 1d gives 2.3×.
16. Step 8 fixed-grouping control: per Sec. 4.5 it "retains learned query/key projections and input-dependent soft pooling and ungrouping"; say so (not "無樹、無壓縮損失" only).
17. LO2 / symbol table 256→128→64, N=256 hold for ImageNet only; T2I is HiT-B/16 D27 at 512 px (32×32 = 1024 image tokens + caption tokens; targets not stated). Scope LO2 and C3 to ImageNet D18; add the T2I setting to step 11.
18. LO3 answer 「合併分數不增，所以等於最高分低於 τ 時停止」: Sec. 3.2 only says "We retain merges with affinity at least τ". Tag 〔本頁推導〕 inside the answer, "assuming exact average linkage".
19. W_ℓ caveat: Table 10 shows plain addition R+H (11.65) beating the default projected skip (12.27); authors keep projection "as a simple fusion rule". One sentence so the page doesn't imply W_ℓ helps.
20. Common error 3 title 「每張影像固定群組數」 vs body (cross-device summation, gradient-accumulation averaging): split or retitle.
21. Hero card 2 「平均保留一半的 token」: ρ=1/2 is per stage; overall 1/4 (256→64). Say 「每一階段」.
22. LO11: k=1.7 applies only to the 600-epoch endpoint; add 「（僅 600 epoch 端點）」.
23. Style: 「token 數沙漏」(metaphor) → 「token 數變化圖」; 「鼓勵合併」 → 「降低 f_s（使合併分數上升）」; 「讓每個群組讀取所有 token」 → 「每個群組 token 由所有輸入 token 加權平均而成」.
24. Smaller gaps: collapse Eq. 10 beside Eq. 4 in C2; define C3 stage count → segment mapping, default block counts, data structure (layout:[…], ratios:[…]); §6.8 reduced-motion: state "no animation" or list animations with fallbacks.
