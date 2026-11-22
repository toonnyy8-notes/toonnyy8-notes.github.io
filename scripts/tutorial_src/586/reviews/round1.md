# Round 1 reviews (fidelity + pedagogy, merged)

Fidelity: c1–c4 rerun under `python3 -I`, byte-identical to c*_out.txt. C4 TABLE vs Tables 8/9, Tables 1–4, 6, 7, 12, 13 and Eqs. 1–31 checked. Confirmed correct: H/Avg examples (IconQA, NVILA COCO 62.65/99.1/85.7), Tailor Avg 66.9 vs 56.8, Table 3 memory + Grafting 100%, Eq. 7 bottom-ρ vs Fig. 2 sign, Tables 7/12, ablation +2.2, ρ=0.5 vs 0.75 max gap, BWT 1.25 factor. No fidelity P0.

## P0 — pedagogy

1. **H-score semantics (S1, Eq. 9, C4).** A_down can be CIDEr up to 200, so "harmonic mean of two accuracies" is undefined for a beginner. In S1 prose: define the harmonic mean in one sentence with a 2-number example (90 and 10 → Avg 50, H 18) before the formula, and state that A_down may exceed 100 and H is used anyway.
2. **S2 overload / Jacobians untaught.** S2 (350–450 字) must teach Jacobian, F, chain rule, first-order Taylor, the working hypothesis and a symbol table. Split S2a (z = Wh, 2-2-3 SVG, a literal scalar-derivative analogy) / S2b (vector derivative, Taylor), ~600 字. Use one term for J_i (「第 i 個欄向量」, not 行/列 mixed); say ∂f/∂z_i is the vector of d_final partial derivatives.

## P1 — fidelity

3. **LO6 / C2 homogeneity example doesn't show non-homogeneity.** W₂₂ scores low because GELU′(z₂) ≈ 0 — the same as a dead ReLU unit (J = 0). It shows magnitude ignores ‖J_i‖, not that non-homogeneity breaks magnitude. Reword; keep the homogeneity link as the paper's claim (§2.2/§3.1) without demonstration.
4. **LO10 「約等於 75% 重疊」.** Paper's random Hamming 0.226 exceeds the 0.2 maximum of a plain fraction, so its normalisation is unknown; also masks compare *top* 10% while training updates the *bottom* 10% (App. I, Tables 10/11). Drop 75% from the reference answer; at most a flagged aside 「若 Hamming 為一般比例」.
5. **LO7 / step 6 「同一輸出節點內排序相同」** holds only for a single input: Eq. 5 averages ‖J_{i,n}‖·|h_{j,n}| over N probes and App. K's Wanda uses ‖h_j‖₂; Eq. 7 ranks all rows of a layer together. Add 「單一輸入時」. (Pedagogy: also state in S4 that i indexes the output node/row and ‖J_i‖ is constant along a row before S5's title relies on it.)
6. **LO12 mixes settings.** Tables 1–2 = last 20 layers; Table 4/8 = all 28 layers (App. H). NVILA ImageNet-R Dowser is 71.2 (20 layers) vs 69.7 (28 layers); 69.7 coinciding with LLaVA's main-table value is a coincidence. Write 「20 層」「全 28 層」 in the answer and C4 caption.
7. **§1.1 「由一階 Taylor 展開導出（Theorem 3.1）」** contradicts step 5 and App. B ("not aim to predict the exact output shift"): write 「由 Theorem 3.1 把 |ΔW| 代換為 |W| 定義」.

## P1 — pedagogy / implementation

8. **Toy architecture undefined before C1.** S2 defines z = Wh only; C1 introduces V, b, GELU in the plate. Define f = V·GELU(Wh + b) and GELU (formula, small plot, why non-homogeneous) before C1; add a, V, b, GELU, Φ to the symbol table.
9. **S4 homogeneity paragraph uses C2's layer numbers (W₂₂ = 2.0, z₂ = −2.03, GELU′ = −0.082, ‖J₂‖ = 0.109) before the layer appears (S5).** Show the 4×4 toy layer in S4 or move the paragraph to S5 right before C2; say it is a custom toy layer.
10. **S5 density / forward refs.** Keep magnitude and Wanda in the body; SNIP/OBD/OBS/MAS into a collapsible aside tagged 〔本頁推導〕 (SNIP/OBD are not in the paper). Eq. 3 equality S² = E[(W ∂(ξᵀf)/∂W)²] uses ξ and Eq. 4 from S6 — move it to S6 or drop it.
11. **C2 compute on main thread.** 2¹⁶ subsets × forward on slider release: make chunked compute (requestIdleCallback / setTimeout slices) or a Worker the default, keep the 「計算中」 state, cancel stale jobs via a run token when b₃ changes; consider Gray-code incremental enumeration.
12. **C2 ties.** S₂₁ = S₂₂ = 0.044 and near-zero row scores at b₃ = −4: sort on values rounded to 1e−9 (state ε), display the same rounded values; define Spearman when variance is 0 ("—", already) and the m = 15 near-full-mask case.
13. **C1 formatting/axes.** Specify percent formatting for relative error > 100% (「+215.1%」) and sign convention ((預測−真實)/真實); specify axis max so the predicted V stays visible at the slider edges for the (1,2) chip.
14. **390px.** Wrap every `.eq` in overflow-x:auto with a scroll cue (Eq. 5, 7, 8 not just 2/26); every `.tscroll` table gets a sticky first column (S8, S9 8-setting table); specify the C2 readout table at 390px with the three heatmaps stacked.

## P2

15. **Theorem assumptions (S3, LO3):** Eq. 12 "exact" holds for one token position (same W_ij acts at every position), F differentiable at z (ReLU kinks excluded), J at pretrained weights, small ΔW. Add to S3 or tag LO3 「單一位置、可微、小 ΔW 的理想化」.
16. **C3 / step 7:** the paper gives only Eq. 4 and Eq. 5; how R samples are combined (√(mean X²), Jensen 4.000) is a design choice — tag 推定.
17. **C2 plate:** add a footer line that the toy does not test the hypothesis linking Δf to A_up (R2); 「full 的真實位移最小」 must not read as evidence that minimising Δf is the goal.
18. **p-values:** 「本法只有單次執行」 is not stated; write 「表中只有 Random 附標準差（三個 seed）；論文未說明本法次數」.
19. **Step 9 Table 11 (0.081) vs Table 5 (0.050):** different reference pairs (text-only vs image+text probes; synthetic vs real), Spearman 0.882 vs 0.891 — not like-for-like.
20. **Table 13:** upstream set differs from the main six (Zeroshot A_up 63.9 vs 61.0); say so.
21. **Step 8:** sharp √½ for p = 1 is Szarek (1976); paper cites Khintchine 1923 / Haagerup 1981 — mention Szarek or write 「由引用的 Khintchine 型不等式」.
22. **R2:** tag C1's V and x as 推定 on the page too.
23. **Dynamic KaTeX:** no $…$ inside re-rendered nodes (chip labels, 「ΔW = −W_ij（歸零）」 button, aria-labels) or re-run renderMathInElement on them; JetBrains Mono lacks ᵢⱼ subscript glyphs — specify fallback (tspan / sub).
24. **Accessibility:** one debounced (500 ms) live region per plate announcing 2–3 key numbers; C2 heatmap cells get row/column aria-labels with one tab stop per map + arrow keys (roving tabindex); define C3 tab order; C4 ρ columns need Enter/Space and arrow keys.
25. **Contrast / colour:** C2 in-cell numbers switch text colour at high fill (≥ 4.5:1 light and dark); factor bar adds text 「用／未用」; hatch stroke token for dark mode; C3 number-line labels stagger with leader lines when estimates are within a few percent (e.g. (3,4) at R = 64).
26. **Motion:** state widget updates are instantaneous (no transitions); timeline/TOC smooth scroll disabled under reduced motion.
27. **Style / density:** S0 subtitle and fact card 2 use 分數, 神經元, 敏感度 before definition — use plain words; S1 heading states the authors' finding as fact → 「作者觀察到…」; split S6 into 6a (Eq. 4, VJP, C3 with d = 2) / 6b (L1 bound, caveats) and raise S6/S9 length limits; add a small static 3×3 mask example (M = 1 = updated) before S8's formula, since that is the most common misreading; LO10/LO12 are pure recall — acceptable, but say so.
