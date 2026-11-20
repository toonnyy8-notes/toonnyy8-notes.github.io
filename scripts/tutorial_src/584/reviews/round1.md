# Round 1 reviews (fidelity + pedagogy, merged)

Fidelity: check.py output identical to check_out.txt; §5 tables, Table 1–3 numbers, Eq. numbering spot-checked correct; 0.786 limit independently confirmed (0.78647). No P0.

## P1

1. **LO10 / step 10 / C4 (both reviewers).** Paper Sec. 4.2 computes G "from the returned LSE values and the query-key product for the first token"; kernel LSE includes token 0, so LSE¬0 needs LSE+log(1−e^{z0−LSE}), which has the same cancellation as the failing path. C4's σ path computes LSE¬0 directly from z1..11, which implies a method the paper does not give; 1.9e-174 is not reachable via the paper's route. Fix: LO10 says the kernel LSE includes token 0 and the source of LSE¬0 is unspecified; label C4's σ column 「需另行取得 LSE¬0（論文未說明）」. C4's real lessons (chunked online LSE, overflow, cancellation) map to no LO: add an LO13 (numerical stability) tied to C4, or fold into LO10.
2. **C1 Gated mode internal implicit gate 1−A0 (fidelity).** Sec. 6.2: implicit gates come from tokens with high weight + low value norm. At ρ=1 the first token is not drained, so 1−A0=0.8902 is not a gate, contradicting point 3 (gate form needs v0≈0). Fix: show 1−A0 as a gate only when ρ≈0 (or default Gated to ρ=0, labelled).
3. **Step 7 claim 「G∈(0,1)，所以 Imp∈(0,1)，CV 分母恆正」 (fidelity).** False for Vanilla at t=0: A_{0,0}=1 ⇒ G=0, LSE¬0 over empty set; Eq. 5 averages over all (i,t), paper silent. Fix: Imp∈[0,1], positive if any t≥1 exists; t=0 convention unstated.
4. **Step 8–9 / C3 causal claims (fidelity).** App. D: dominant heads' gates "resist change", others "increase their own gating factors to align". The toy freezes heads, so pinning is built in; the scale-invariance second gradient term isn't needed for it. Routed limit 0.786 exceeds two of three shared heads (0.739, 0.578), so 「逼近主導 head」 is loose; 「最便宜」 unproven. Fix: say only that the toy shows a CV objective with fixed heads raises the rest; drop the causal claim about the second term. Also label 0.442 / 0.161 「本頁玩具模型，非論文結果」.
5. **Symbols before definition (pedagogy).** (a) Hero uses sink, value 向量, head, 隱式門控, MoE, 負載平衡損失 before S1: use known words and say terms are defined from S1. (b) Q/K/V, multi-head, causal mask, logit never taught: add a short attention primer; define LSE (log Σ e^a) in S1 or start of S4. (c) S3 uses A_{i,sink}, α_sink, τ (Eq. 9) before S4 defines the sink component: define first, or move Eq. 9 after S4.
6. **Density (pedagogy).** S4 + C1 (Eq. 3–4, Eq. 12–16, B.3, ln((1−G)/G), 3 mechanisms, 5 sliders, 3 plots, 7 readouts): split S4 into 4a (Sink exact) / 4b (Vanilla approximate, ρ); default C1 to Vanilla/Sink, Gated mode later (S5). S8: put Eq. 7 gradient/N·CV² identity in `<details>`, prose states only that CV is scale-invariant so the loss does not constrain the mean. S10: move common mistakes #4–#6 (Eq. 8, std version, λ) to S8/S9 or split S10.
7. **Literal style (pedagogy).** Remove personification: 「抗拒改變」 (C3, LO9), 「搶權重」, 「吸收/傾倒冗餘」, 「吸引不成比例的更新」. Limit the 「開關」 metaphor to hero card 2 (card 3 and C1 「像被關掉」 rewrite literally). Note: if quoting App. D's "resist change", mark it as the paper's wording.
8. **C2 guards (pedagogy).** Arrows normalized by max|g| ⇒ 0/0 at β=0: draw no arrows when max|g|<1e-12. On N_h change clamp m=min(m,N−2).
9. **390px equations (pedagogy).** Eq. 2, 7, 8 and O_t sums overflow: `.katex-display{overflow-x:auto;max-width:100%}` with tabindex=0.
10. **Touch targets (pedagogy).** C2 bars ≥16px and 16 tab stops: one role="group" with roving tabindex (arrow keys) plus a head table as accessible alternative; drop "tap = focus".

## P2

11. App. C model count: Fig. 10 has 6, Fig. 11 has 10 ⇒ 16, not 10.
12. Fig. 3 percentages: labels "PC1 44.7%, λ=0.303" are unexplained in the paper; call them 「圖中標注（論文未說明）」 and drop the 「特別大」 interpretation; (c)(f) are schematics.
13. Step 4 「步驟 6 幾何限制的代數形式」 → 「與…相容的必要條件」 (Sec. 3.2 geometry is qualitative PCA).
14. C1 Sink row z0=4, sink 4 (G=0.5545): per Sec. 6.2 (DeepSeek-V4) A_sink should include first-token weight; here 1−A_sink−A0=0.109. Annotate the row / flag that the gate understates shrinkage.
15. Fig. 5: only Gated plateaus; Vanilla and Sink "display a gradual increase" (Sec. 5.2).
16. Step 14 item 5: N/(N−1) holds for CV² (the loss); std/CV differ by sqrt(N/(N−1)). State which.
17. Step 12 「Qwen3-8B 的增幅大於 Qwen3-4B」: paper only says "larger models benefit more"; tag 「本頁對照」.
18. Hero card 1: value ≈ 0 is shown only for Vanilla models (Fig. 2); add 「（Vanilla 模型）」.
19. C3 附註 「腳本以格點搜尋得到」 0.786: check.py has no such search; add it or drop the claim.
20. Unicode subscripts (ₛᵢₙₖ) in SVG/readouts may be tofu in JetBrains Mono: use `<tspan baseline-shift>`; render KaTeX row labels once, update numbers as text.
21. C3 plots: 13 routed curves coincide in Eq. 8 ⇒ plot min/mean/max band + 3 shared lines; record hist every 10 steps (not 3000×16); fix loss axis range.
22. Reduced motion / dark mode: specify matchMedia per component (C1 transitions, C4 stepping, hero); give light/dark values for --c-par/--c-perp; hatch pattern stroke uses a token with ≥3:1 contrast.
23. Static SVGs: specify S2 Fig. 1 three columns (boxes, labels) and S6 two-panel schematic (polar coordinates, angle ranges, point counts, seed).
