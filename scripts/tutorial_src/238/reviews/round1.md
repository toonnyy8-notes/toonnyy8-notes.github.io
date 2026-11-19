# Goldfish review round 1 — consolidated action list

Three stateless reviewers read design.md: (G1) Haiku comprehension check, (G2) Sonnet fidelity vs paper, (G3) Sonnet pedagogy + build readiness. No P0. G1 restated the argument chain correctly; G2 independently reproduced κ, Eq. 54c/d, Δ* 22/11/10/9.2, C1 residual (1−k)²/8, C3 tables (A/B/C, 8-bin), C4 a=0.448 b=−0.300, and all Fig.1/Table 1–2/Sec.5 numbers.

Resolve every item below in design.md (or state explicitly why not, in 風險與未決事項).

## A. Ambiguities / contradictions (G1, G2)
1. Loss space of Eq. 15: the doc mixes ‖u_θ−u‖² with D̃α, Alg. 1/Table 3 v-loss, and §1.4 step 9 "D̃α 含 v-loss 的 κ²". Decide from the paper and state one rule for the page and for C3/C4.
2. Clamp rule: state exactly per Table 3 — pixel 0.05; latent 0.05 only for x-prediction, 0 otherwise. Fix pitfall 2, step 9 and LO11 consistently. Say whether C3/C4 clamp κ² and why.
3. Sampler: Alg. 2 is Euler, but all reported results use Heun 50 steps (NFE 99). Say both; pitfall 6 must mention it.
4. Define before use: interval I (step 5), Σ (§1.3 vs step 7), Alg. 1's `e` vs text n, and "v" used both as velocity target and loss name.
5. Step 10 self-contradiction: k(t) ≈0.56 at t≈0 vs "endpoints stay at init 0.5" — reconcile with Fig. 5a wording.
6. 0.66: label as "Fig. 3b long-run converged value (the 800 epochs belong to the Fig. 3a FID curve)"; keep 0.522 as Table 2 epoch-64 footnote.
7. LO10 answer mixes panels: name the Fig. 1 panel for each number (0.212 / 0.659 are bottom-right: time conditioning + logit-normal + v-MSE).
8. Table 1 pixel rows: the paper doesn't state their epoch count — don't imply 600.
9. Table 3 transcription: add pixel warmup 5 epochs (latent 0), EMA decay {0.9996, 0.9998, 0.9999}, bottleneck 128, in-context tokens 32, class-token drop 0.1; pitfall 4 mentions EMA.
10. External references the builder can't see (AGENTS.md 6-3, DESIGN.md, `#pm-bar`, Paper Master, timeline.json format, slug): inline the facts the builder needs (tokens, nav bar markup, the timeline.json schema). Use slug `prediction-targets-k-diff`.

## B. Fidelity nuances (G2)
11. C3 observation 2 ("noise-dependent → always 0.5 because perpendicular residual vanishes"): the k*=0.5 result is your derivation (k↔1−k, t↔1−t symmetry + convexity of Eq. 77c); tag it 〔本頁推導〕; the remaining parallel term is (φσ−ψα)²/(α²+σ²); add that this is a linear-model population result that does not predict trained networks (Sec. 3.4, Sec. 6; time-conditioned DiT/JiT learn k→1.0 in pixel space).
12. Step 5: zero perpendicular residual iff ψ/σ constant in α (Eq. 78b); within the k family that is only k=1.
13. C1 / LO7: the shared-vs-dependent gap is not purely perpendicular — Eq. 78a also lowers the parallel term.
14. Wording "可重現 … 0.212" for the inferred logit-normal (μ=−0.8, s=0.8): use "與…一致"; state on-page this is a calibration to the paper's numbers, not the paper's stated setting.
15. "ε、v、x 都不是最小值": qualify with 0.5<k*<1 (Sec. 3.3 "neither conventional target is optimal").

## C. Pedagogy (G3)
16. C1 appears in S1 but needs k, u, D/d, perpendicular residual taught in S2–S4. Split: C1a in S1 (t, seed, z = t x + (1−t) n only) + 2–3 sentences on why a network predicts a target (sampling); C1b (k slider, chips, gain/residual plot) in S4.
17. Density: defer ξ/η/κ/D̃α and the κ table to where first used (S7/S8). Add to S3 a short explanation of PPᵀ as a projection and gradient descent as an ODE. Split S4 into two sections; put the two mode equations before Eq. 12; LO4's answer (E[zzᵀ], Eq. 32/34) must be taught in the outline.
18. C3 load: add a teaching paragraph for W(α) (noise-dependent) and logit-normal before C3; step the widget A → N bins → B → C; weight sub-plot only for B/C; explain the "population optimum ≠ trained value (0.212 vs 0.659)" point explicitly.
19. S0 hero shows 0.66, 1.0, k*, "two modes" undefined → plain-language fact cards; k-axis labels must not overlap at 358 px.
20. S3 figure (d=2 in D=3) vs C1 (d=1 in D=2): use one consistent setup.
21. S6 (eigenvalues/trace/SNR) has no LO or widget and uses untaught terms: shrink to the Eq. 72 sentence or add an LO.
22. Wording rules: replace "拉鋸" → "權衡", "讓 k 自己學" → "k 作為可訓練參數", "k 尺" → "k 軸", "陷阱" → "常見錯誤"; use declarative section titles (no question-like titles).
23. 5.5 rule vs widget readouts: widget readouts carry one "本頁推導" caption instead of per-number citations.

## D. Build readiness / edge cases (G3)
24. C4: require analytic gradients (loss is quadratic in a,b for A/B; precompute moments once; recompute per step only for C). Specify lr schedule (decay) or state expected jitter; acceptance tolerances must be reproducible with your verify scripts.
25. C1b plot: g(t)=−(1−k)/(1−t) is 0/0 or −∞ at t=1 → special-case; plot e(t)=b*σ−ψ (or shade (b*−g)²σ²) so the shaded area equals the (1−k)²/8 readout.
26. C3 `modeParts`: catastrophic cancellation near k=1 (∫wψ² − (∫wψσ)²/∫wσ²), empty tail bins with N=64 → guard denominators (den<1e-300 → 0), clamp Δ≥0; M=1024 or floor(t·N) binning so bins get integer points.
27. Math rendering: KaTeX doesn't render inside SVG text/tooltips/live readouts → Unicode in SVG, call renderMathInElement on dynamic DOM, `.katex-display{overflow-x:auto}`; tables ≥4 columns use scroll wrapper or stacked cards at 390 px.
28. Accessibility: live regions update on `change` or debounce 500 ms; C4 announces only final state; par/perp distinguished by dash/marker + direct labels, not color alone; C3 tooltip needs tap-to-pin or a d slider on touch.
29. C4 reduced-motion path must not block the main thread (fast with item 24, otherwise chunk with setTimeout).
30. C2: fix y-axis range [0, 0.35]; ρ axis down to 1e-5; print k* with 4 decimals when >0.999; thousands separators.
31. Fonts: display=swap, system fallbacks; mention page-weight check on mobile.
