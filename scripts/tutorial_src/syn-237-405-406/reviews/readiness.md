NO-GO

## Explain-back

The page teaches that score errors must be read in the right units. For VP diffusion, score error = ε error / σ_t = √ᾱ_t · (x0 error) / σ_t² (D1). Theorem 4.1 of 237 gives sufficient conditions: score error o(σ⁻²) concentrates on the manifold, and o(1) recovers p_data. Two sampling-time fixes act on the score. TS (237) multiplies the unconditional corrector score by σ^α, giving a uniform distribution on the manifold in the limit. ES (406) divides ε by λ in the predictor, with crossover σ* = λ^(−1/α); constant ES is not a special case of TS. IP (405) trains on noisy inputs; its Bayes-optimal network is a √(1+γ²)-scaled score, which does not explain its gain because DDPM-y has nearly the same optimum and performs worse. γ, ν_t, e_t, and δ_t are linked by a Gaussian model, and a matched γ holds only over a range of t.

Components: W0 converts an x0 or ε error into a score error (σ slider, error input, unit toggle). W1 plots TS, ES, and IP multipliers against σ, with α, λ, σ, and γ (IP only after S6) controls and readouts of σ* and α_eq. W2 plots stationary densities on the unit circle with σ, λ, α, γ, and overlay controls, reporting TV and max/min ratios. W3 plots γ_1 and γ_M against t with T′, e, schedule, and t controls, reporting the matched range.

## Blocking

1. Symbols used before definition or never defined (page-facing):
   - S3 table, "關鍵數字": "γ = 0.1；$\nu_t$ 平均 0.20／0.19". γ and ν_t are defined only in S6/S7.
   - S3 table: "ε ÷ $\lambda_t$" and "$\delta_t$（Sec. 3.3）". λ is defined in S4, δ_t in S7.
   - S3 table: "$\max\{-\beta,0\}<\alpha<2$". β is never defined on the page.
   - S3 table: "$L^\infty(K)$". K is never defined.
   - S5 body: "$T'=20$". T′ (sampling steps) is not defined in S0–S5; it first appears explained in the W3 spec.
   - S5 body: "$p_{vol}^{1/\lambda}$". p_vol is defined only inside the collapsible advanced box, so the body uses it before readers can see it.
   - S3 table: "取樣 $\hat x_t$". x̂_t is never defined; S0 defines only x̂0.
2. Acceptance tables have no tolerances. §5.1–§5.6 give values with no ± column and no rounding rule. verify/check_out.txt has no tolerance either. §6.3 gives tolerances for only some items. §5.3 (TV columns), §5.4 (γ_M rows for T′=100 and 1000), §5.5 (λ table), and the W1–W3 readouts have none. The S5 claim "0.0046 等於…0.0045" needs a stated tolerance or "約".

## Checks that passed

- All 8 LOs have reference answers and a teaching section (S1–S8).
- Linked slugs exist: score-geometry-lineage-hub, input-perturbation-lineage-overview, exposure-bias-elucidation-overview, score-geometry-tempered-langevin, input-perturbation-ddpm-ip, epsilon-scaling-method, diffusion-exposure-bias. The hub note score-error-exposure-bias-relations.md has all 8 required frontmatter fields.
- Spot-checked against verify/check_out.txt and matched: mES(1.017)=0.98328; mIPman(0.1)=0.99010; σ* values; α_eq 0.00732/0.00366; first t=883; T′=20 δ values 8.711e−3 / 6.054e−4 / 8.292e−6; 0.18467 / 0.64902 / 0.96751; 0.23261 / 0.24172 / 0.21686; max/min 2603.7 / 2976.1; W0 values 19.90 / 2000 / 2.00 / 20.0. §5.4 T′=20 γ_1 and γ_M values are exactly half of check_out's e=0.1 values, consistent with linearity in e.
- Not checked: §5.4 T′=100/1000 rows, §5.2 Monte Carlo, and paper-number claims (outside the requested inputs).

## Non-blocking notes

1. notes/timeline.json does not exist. §6.1 says no timeline is shown, so this is likely harmless, but the requested input was missing.
2. W1 label "ES：predictor 步，整個 ε" conflicts with S3 "predictor 平均中的 ε；噪聲項不變". Use one wording.
3. W2 spec calls refCurve('pdata') and c2Marginal({a,b,sigma,alpha}). The actual refCurve signature in 237/verify/core.js is (kind, a, b, lambda, nt). Give the arguments.
4. W2 and selfTest depend on 237/verify/core.js (c2Marginal, refCurve, circleScorePolar), which exist. The synthesis core.js exports do not include them, so the build must paste both files as §6.2 says.
5. W3 e default 0.2 and T′ default 100 are consistent with §5.4, but the "推定" tag on cosine should also appear on the W3 readout line, not only the figure title.
6. Table §5.3 "ES 型 TV 0.0046" is described as equal to the limit's 0.0045. Say "約等於".
7. Symbol V_t appears in D4/D5 (developer section) without definition. Fine for the page only if D4/D5 stay collapsed.
8. The 406 table "CIFAR-10 FID 10.36 → 5.15 at T′=20" is cited in D3 and K4 but was not checked against the source paper (outside scope).
