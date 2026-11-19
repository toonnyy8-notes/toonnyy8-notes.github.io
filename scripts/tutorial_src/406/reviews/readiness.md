NO-GO

Explain-back
The page teaches that DDPM sampling feeds the network its own x̂_t while training feeds it true-noised x_t. Under the Eq. 7 model (x_θ = x_0 + e_t ε_0), the one-step sampling variance exceeds the training variance by (c_{1,t+1} e_{t+1})² with the same mean (Eq. 9, Table 1; DDIM in Table 8). Multi-step errors accumulate via V_{t−1} = c_{2,t}² V_t + (c_{1,t} e_t)² + β̃_t, measured by δ_t = (√V_t − √(1−ᾱ_t))². Noise-schedule tweaks are argued not to cancel the gap (author's argument). Epsilon Scaling divides ε by λ at sampling time; λ(t) is shaped by Eq. 10 (second differences of ΔN−1), and the practical choice is constant b (ADM CIFAR-10 b = 1.017, FID 3.37 to 2.17 at 100 steps).
Components:
- C1: sampler, schedule, T', e, t controls; plots training vs sampling variance and the gap (log axis), with readouts.
- C2: model A (exact Eq. 7 recursion) or B (perfect Gaussian denoiser); T', e, Var(x_0), t; plots single- vs multi-step variance error and δ_t.
- C3: h and shape w; plots λ(t) from a second-difference inversion with a back-substitution check.
- C4: browser for FID tables (Tables 2–5, 17–18) as bars and sensitivity lines.

Blocking
1. C4 has no defaults. "輸入：分段按鈕「表格」…第二層 chip 選子集…勾選框「以表格顯示」" never says which table, which chips, or the checkbox state on load.
2. Tolerances too loose for the values they test. §5 gives "絕對 ±5e-4（線性軸數值）" for C2, whose values are 1e-3 to 1e-6 (e.g., Δ多步 at t=5 = 1.935e-3 accepts ±26%; δ at t=5 = 8.292e-6 passes any value below 5e-4). Need relative tolerance on every readout, with a small absolute floor.
3. ΔN is not operationally defined. S7 says ΔN = ‖ε^s‖₂/‖ε^t‖₂, but the expected-value formulas in §1.4 step 8 give squared norms. The acceptance values (1.4096, 1.1308) match the square root of the ratio of expected squared norms; without the sqrt the t=1 value is about 1.99. State the sqrt and both expectations explicitly.
4. V_t is used before it is defined. The S5 advanced box (LO7 answer, recursion V_{t−1} = c_{2,t}²V_t + …) uses V_t, but S5's 本章符號 list does not include it, and S6 is where it is defined ("定義（頁面寫 V_t）"). Define it in S5 or move the recursion to S6.

Non-blocking notes
1. C1 t=1 shows 1−ᾱ₁ as "0.00004"; the true cosine value is 4.130e-5, and the stated ratio 240 only works with that. Display 4.130e-5.
2. In C2 mode B, s is the loop index (c_{2,s}, k_s, a_s) and s² is Var(x_0). The state already uses s2, but the formulas read ambiguously. Consider naming the variance v or σ₀².
3. §1.4 step 6 writes β′_t while the page uses V_t. Unify the notation.
4. Timeline lists 1511.06732 as 2015, but §1.1 cites "Ranzato et al., 2016". Pick one year convention.
5. The LO7 answer sits inside a collapsed `<details>` 進階 box. Confirm it counts as "寫在內文" for the acceptance check.
6. The hero link `../index.html#paper-406` was not verified. Static index.html has no "paper-406" string, so confirm the anchor is script-generated.
7. Checked and consistent: all 11 slugs linked from design.md exist (diffusion-forcing and eps-prediction-ddpm-score in notes_split; diffusion-exposure-bias and input-perturbation-ddpm-ip in tutorial_src/405/notes; the other seven in tutorial_src/406/notes). All 7 notes in 406/notes have slug, title, cluster, kind, tags, arxiv, sources, related. timeline.json has 13 entries, and every arXiv ID maps to a TL_GROUP. C3 closed-form values for h=0.02, w=0/0.5/1 re-derive correctly; the FFHQ and CelebA improvement percentages recompute; the A.10 b − 9.5k check holds for LSUN/FFHQ/IN64/IN128 (CIFAR 0.99325).
