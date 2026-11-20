NO-GO

**Explain-back**

The page argues that flow-matching networks regress the velocity u = ε − x (Eq. 1–2). AnisoFlow instead regresses u_W = Wε − x with W = B·diag(w)·Bᵀ, w_i ≥ 0 (Eq. 3–4), and recovers velocity exactly through û = (αW + σI)⁻¹((I − W)z + û_W) (Eq. 5, 10). Because recovery is exact, the minimum reachable loss does not depend on w (Eq. 11–12); only the excess error, divided by (αw + σ)², depends on w. Under a frozen-error model the per-direction optimum is w* = (Γσ + αC)/(Aσ + αΓ) ≈ αC/(σA) (Eq. 18–19). If A(t) = αq·Ā and C(t) = σq·C̄, then w* = C̄/Ā at every t, so the time average does not depend on p(t) (Eq. 20–23). The no-information limit does not cancel: its per-time optimum is (α/σ)λ (Eq. 25–26). Two extra assumptions give w* ∝ λ (Eq. 24), supported by measurement (w ≈ 0.15λ on JiT-H/16; FID 1.90 → 1.55).

Components: C1 scalar recovery (sliders t, w, δ, optional σ_min clamp) shows the error δ/den and amplification 1/den. C2 linear-Gaussian scatter (λ, t, w) shows the regression line moving with w while the recovery coefficient stays −1.2 at λ = 4, t = 0.5. C3 frozen-error curve f(w) (A, C, ρ, t) shows w* and whether the minimum is interior or at a boundary. C4 time-averaged loss (separable or no-information mode, uniform or logit-normal p(t)) shows whether the argmin moves. C5 batch-PCA scatter (λ2, θ, n) shows the first spectral update's angle error shrinking with n.

**Blocking issues**

1. C1 right plot range cannot contain the specified curve. The spec says "縱軸對數 $[10^{-2},10^{2}]$" while "取 $t_j=j/400$，$j=1..400$", so the w = 0 curve reaches 1/t = 400 at t = 0.0025. No clipping or arrow rule is given for out-of-range values (C3 and C4 give such rules; C1 does not). Fix: widen the axis to 10³ or specify clipping with arrows, and align the t grid with the slider's 0.01 lower bound.

2. C4 main plot range cannot contain the specified curves. The spec says "縱軸 $f_t(w)/\min_wf_t(w)\in[1,3]$", but the default separable case (r = 0.1) gives a ratio of about 90 at large w, and the no-information case (λ = 1, t = 0.9) gives about 23 at w = 10. No out-of-range rule is given. Fix: use a log y-axis or clip with arrows.

3. Symbols used before definition in S7. The LO9 answer in S7 uses "$\bar C^{\rm eff}\propto\lambda^{1.00}$", "$\bar A^{\rm eff}\propto\lambda^{-0.09}$" and the ratio exponent 1.09, but the S7 content list ("A.3 一致性檢查（設定、18 方向、指數、0.97、門檻放寬）") does not define A^eff or C^eff. The definitions (A^eff = A + (α/σ)Γ, C^eff = C + (σ/α)Γ, Eq. 35) appear only in the design's step 9, not in the page plan. Fix: add one defining sentence to S7.

**Non-blocking notes**

1. Table 5.3, t = 0.8, column f(0): the table shows 0.1562 with a note allowing 0.1563. The page's toFixed(4) gives 0.1563. Write 0.1563 (or 0.15625) and drop the hedge.
2. LO12's answer cites the quantization evidence (W8A8, Fig. 1b/6b), but that evidence is taught in S9, not S10. Add a cross-reference in S10.
3. S5 mentions "ρ" (C3 default ρ = 0) without defining it in the section text. Add "ρ = Γ/√(AC)" to S5.
4. The hero axis note uses λ and "w ≈ 0.15λ" before S2 defines λ_i. Card 3 glosses it in words, so this is minor.
5. The hero and S2 say "非負" (non-negative), but step 12 says the implementation is strictly positive (exp parameterization). Keep the distinction visible in S2 or the hero.
6. Related slugs in the seven notes all resolve. `prediction-targets-k-diff` is at tutorials_raw/prediction-targets-k-diff.html. Frontmatter (slug, title, cluster, kind, tags, arxiv, sources, related) is complete in all seven aniso-*.md notes.
7. Spot checks of §5 by hand (5.1 with truncation, 5.2, 5.3 rows, 5.4 EDM rows, 5.5 first rows) match verify/check_out.txt and the closed forms. Tolerances are stated in every table.
8. LO and section mapping is complete: LO1–LO12 map to S1–S10, and each has a reference answer in its section.
