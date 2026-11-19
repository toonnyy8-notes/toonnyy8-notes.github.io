NO-GO

Prior blocking items (readiness.md)
1. Alg. 3 line references: partly resolved. Step 4 (design.md line 92) and the pseudo-code note (line 93) now agree: line 3 draws ξ, line 4 computes y_t, line 5 is the gradient step. The page spec still disagrees (B1).
2. ν_t dataset mapping: resolved. Tagged 〔本頁判讀〕 in step 6, LO9, S7; R8 closed; notes/diffusion-exposure-bias.md says the mapping is inferred from legend order.
3. Sections 1.1–1.3 with no page section: resolved. Line 15 declares them design background and maps each 1.4 step to a page section.

Explain-back
DDPM trains on x_t built from x0 (Eq. 4) but samples from its own predictions x̂_t (Alg. 2), so errors accumulate (Tab. 1: ADM FID 0.983 at 100 steps, 3.544 at 1000). Fewer steps can win (Tab. 3 grid best is 300). DDPM-IP trains on y_t = √ᾱ x0 + √(1−ᾱ)(ε+γξ) with target ε, γ = 0.1, and unchanged inference (Eq. 6, Alg. 3). Eq. 7–8 show y_t has the same law as DDPM-y's input, but DDPM-y's target ε′ differs and it scores worse (Tab. 7). γ comes from a grid search over [0, 0.2] around ν_t ≈ 0.2. The slope toy shows IP's slope is at most 1% smaller, which cannot explain the gains.
- C1: sliders ρ, s, t0; FD/L1 toggle. Reader sees the std ratio drift, FD rise with t0, and an L1 floor at ρ = 0.
- C2: sliders t, γ; resample. Reader sees the y_t cloud, and ε versus ε′ target arrows with their distance.
- C3: sliders γ, s, t. Reader sees extra noise in x_t and x0 space and the slope ratios.
- C4: tabs over Tab. 1, 3, 4, 5, 7. Reader sees best T′, gains, speedups.
All components have inputs, ranges, defaults, computation and expected observations.

Blocking
B1. Alg. 3 is described with two different line counts. Step 4 says the change is lines 3, 4, 5 and the pseudo-code has three lines. But S5 (line 243) says "Alg. 3 與兩行虛擬碼", and 6.2 (line 437) says "Alg. 3 中與 Alg. 1 不同的兩行". Fix: state three lines (3 draws ξ, 4 computes y_t, 5 gradient on y_t) in S5 and 6.2.
B2. sFID is used before it is defined. Section 3 requires each symbol to be defined at first use, and sFID is defined in S10 (line 248). But Tab. 7 (S6, line 109) and Tab. 2 (S9, line 148) are specified with "FID / sFID" columns, and nothing says to drop sFID there. Fix: define sFID in S6 at first use, or specify FID-only columns for S6 and S9 and move sFID columns to S10.

Non-blocking
1. ADM is never defined on the page. Its first use is the S3 Tab. 1 row. Add a definition at first use.
2. Step 4 writes Eq. 6 with γ_t, while the page uses a constant γ. S5 should state the constant form.
3. τ (C1 left-plot axis, r_τ table) is never defined. Say it is the step index.
4. The C1 zero-mean paragraph and the R4 global-generation numbers have no stated tolerance. Add one if they feed selfTest.
5. Hero link ../index.html#paper-405: index.template.html:485 only writes '#paper-' + id. A paper-405 entry was not verified in this pass.
6. The 8 notes exist only under scripts/tutorial_src/405/notes/. The build must publish them to tutorials/<slug>.html or the page links 404. All 8 notes have the required frontmatter fields. timeline.json has 14 entries matching the TL_GROUP table. The three existing tutorials/ slugs (generative-eval-metrics, eps-prediction-ddpm-score, prediction-targets-k-diff) exist.
7. The hero subtitle uses "DDPM" before any definition on the page.
