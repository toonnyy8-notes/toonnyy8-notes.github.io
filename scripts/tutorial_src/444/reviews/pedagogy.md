# Pedagogy + build-readiness review of 444/design.md (Goldfish)

1. **P0** — C1 quantile display: "μ·|τ−1(a<V)|/max(|a−V|,h) … 裁到縱軸頂端". `h` is never defined (value, units). Pick e.g. h = 0.02 (one grid step 0.0025 x 8), state it, and state how the "→ ∞" marker is placed. Without it the quantile plot cannot be built.

2. **P0** — C4 indexing. Eq. 5 runs t = T..1 (1-based); the VP formula uses (2t−1) (1-based); the C4 loop says "t = T−1..0 (0 起算)" with `ᾱ[t]`. A builder will be off by one in β/ᾱ and break every number in the section-5 C4 table. Fix: give `betas(name,T)` as an explicit 1-based array `beta[1..T]`, `abar[0..T]` (abar[0]=1), and write the loop as for k=T..1 using beta[k], abar[k]; the "t>0 adds noise" becomes "k>1".

3. **P1** — S5 uses μ_φ, Eq. 4 (Alg. 1 line `φ ← … Eq. 4`), "擴散" before S6 defines them; the symbol table even says μ_φ first appears in S5, while R11 says the μ/μ_φ clarification is at the start of S6. S4 also mentions "AWR 加權 DDPM". Fix: one sentence in S4/S5 ("μ_φ：用擴散模型學到的 μ；訓練與取樣細節在第 06 章；本章只需把它當成可抽樣的黑盒"); Alg. 1 shows "φ ← 擴散行為複製（第 06 章）" without Eq. 4.

4. **P1** — C2 (S4) uses Alg. 2 resampling, Eq. 9 weights, N and argmax (Eq. 13), but Alg. 2/Eq. 13 are taught in S5. S3 also states the Sec. 6.3 conclusion "argmax 較好" (LO6) before argmax exists. Fix: move the C2 widget to the end of S5 (keep the S4 text for LO7), or add one sentence + mini-definition of "抽 N 個、依 w 選一個／選 Q 最大者" at the start of S4; defer the Sec. 6.3 experiment results to S9.

5. **P1** — S3 is too dense: 3 loss formulas, 2 本頁推導 (quantile non-differentiable, divergent integral), exponential naming note, π_awr vs π_imp, KL, toy setup, C1, Fig. 2, Sec. 6.3, LO4–LO6 in "600–700 字". Fix: split into S3 (expectile + exponential, C1 with expectile default) and a collapsible `details.more` for the quantile divergence argument; move Fig. 2/Sec. 6.3 text into S9. Same for S6 (DDPM + Eq. 4/5 + three schedule formulas + F.3 + C4 in 500–600 字): put the three schedule formulas and cosine clip in `details.more`.

6. **P1** — Missing prerequisite for the S1 question "Q 是哪一個策略的值". S1 never says that V^π(s)=E_{a~π}Q(s,a) and that Q^π depends on the policy; Eq. 7 then calls this "慣用定義" with only a link. Add two sentences in S1 and in the 本章符號 table (Q^π, V^π).

7. **P1** — Eq. 4/5/8 "兩種版本都寫、media query 切換": both copies get rendered by KaTeX and read twice by screen readers; `display:none` fixes the latter only if the hidden copy is not rendered at all in the accessibility tree. Specify `aria-hidden` plus `display:none` on the inactive one, and that selfTest counts only visible equations. Also formula-heavy tables (損失表 4 columns with Eq. 9–12 in cells; Table 1 with 9 columns) at 358px: `.tscroll` hides the weight column off-screen; use a stacked-card layout for the loss table and a sticky first column for Table 1.

8. **P1** — "ᾱ (U+1FB1)" in dynamic text/SVG: it is in the Greek Extended block and not in IBM Plex Sans / Noto Sans TC / JetBrains Mono, so it falls back to an arbitrary font (wrong size/baseline). Use "abar_T" / "ᾱ" via `<tspan>` with `text-decoration: overline`, or write "ᾱ" with an explicit font stack that has it. Also the selfTest regex is shown with literal combining characters; write `/[̀-ͯ]/` and scan only dynamic nodes (readouts, SVG `<text>`, aria-labels), not all `body.innerText`.

9. **P1** — C2/C4 chunked Monte Carlo: no cancel/stale-result rule. If the user changes N or τ mid-computation, old chunks continue and overwrite the readout. Add a run token; discard chunks whose token is not current; disable all controls (not only 重新抽樣) or restart. Also set a fallback when cumulative weight never exceeds u by rounding (choose last index). Add `announce()` once at completion, not on "計算中…".

10. **P1** — LO9/C4 contradiction: the LO says "T = 5、VP", the widget shows T larger is better (R9 acknowledges it). A beginner sees T=50 beating T=5 right after reading "T=5 is best". Put R9's sentence in the body next to C4 (not only under the plate title), and retitle S6 to avoid "五步" as a general claim ("D4RL 設定 T = 5").

11. **P2** — Controls: C3 ε slider with 9 non-uniform stops needs an index slider (0–8) with `aria-valuetext` "ε = 0.01"; C1 log10α slider needs `aria-valuetext` "α = 1" not the log value. 9 N buttons for C3 at 390px must wrap (specify 2 rows of ≤5). C3 second curve "1−(1−ε)^N" "幾乎重合": hidden under the first; use offset marker shapes or draw the formula as dots only.

12. **P2** — Wording: S2 title "任意凸損失的 critic 都對應一個隱含 actor" overclaims (needs f'(0)=0, and quantile fails per R2); use "凸且 f'(0) = 0 的損失". S9 title "差距最大" is true only for total score (locomotion -1 DQL 741.2 > IDQL 711.6); say "總分差距最大". Figurative verbs to rewrite literally: "把質量放在群與群之間", "推向高群的尾端", "落在", "擴散行為模型以…去噪產生" is fine.

13. **P2** — Perf/numerics: "C1 < 5ms" while computing a 300-point V curve with 200-step bisections (≈180k erfc per recompute) on every p_h change; precompute or cache by (loss, p_h) and measure. Document that exponential `w_exp` uses `expm1` and `V_exp` log-sum-exp (done) and that expectile bisection brackets [LO,HI] hold for τ in [0.5,0.99] (fine). selfTest MC assertions with ±0.01 while the design admits Math.cos/log last-digit flips: use ±0.02.

14. **P2** — Reduced motion and dark mode only inherited from the IQL shell; new elements are not covered: C2 hatch pattern (low-density region) and C4 bars need token colors under dark mode; any histogram transition/hero bar animation must be disabled under `prefers-reduced-motion`. State it in 6.2.

15. **P2** — Links/notes: `diffusion-ql.html` slug is unconfirmed (R10; no 443 design.md exists yet). `diffusion-policy-class` and `gaussian-policy-multimodality-limits` exist in 443/notes (verified); `diffusion-ql-lineage-hub` exists but the S11/S12 lists do not link it; add it to S12 and to the 擴散 × RL timeline card. Check 444 `diffusion-behavior-model-policy-extraction` against 443 `diffusion-in-rl-concurrent-work`/`diffusion-policy-inference-cost` for duplication (AGENTS 步驟 6 merge rule); I did not read their bodies.
