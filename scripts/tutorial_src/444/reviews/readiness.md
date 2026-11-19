NO-GO

## Explain-back

The page argues that IQL's critic, with its loss generalized to a convex f (Eq. 6), implicitly defines an actor pi_imp proportional to mu times |f'|/|Q-V*| (Thm 4.1, Eq. 8), so IQL is actor-critic. Expectile gives E_pi_imp[Q] = V*. Quantile gives a point mass; exponential gives pi_awr with mean above V_exp. Because pi_imp is multimodal, single-mode Gaussian AWR fails, so IDQL trains a diffusion behavior model separately (T = 5, VP, LN_Resnet) and at evaluation samples N candidates and picks by critic weight or argmax Q. C1 (loss, tau, alpha, p_h) shows V*, E_pi_imp[Q] and densities. C2 (AWR alpha, tau, N, mode) compares histograms with pi_imp. C3 (eps, N) shows E[r] and outlier probability. C4 (schedule, T) shows sampling cluster shares. I checked the stated numbers against verify/*_out.txt and tables_out.txt, and they match.

## Blocking

1. Diffuser is mapped to the wrong note. design.md line 424 maps 2205.09991 to `diffusion-behavior-model-policy-extraction`, whose arxiv list omits 2205.09991. `trajectory-diffusion-planning` lists it. Fix: map it there.

2. SAC is mapped to a note that does not cite it. design.md line 422 maps 1801.01290 to `awr-advantage-weighted-policy-extraction`, whose arxiv list omits it. `in-sample-implicit-value-regularization-eql-ivr` lists it. Fix: map it there.

3. The C2 tau slider has no step. design.md line 291 gives "τ ∈ [0.5, 0.99]，預設 0.9" with no step. An HTML range without a step defaults to 1, which breaks the slider. Fix: add "步長 0.01".

## Non-blocking

1. Eq. 7 (line 68) uses pi_imp before Eq. 8 defines it. Add a forward pointer.
2. Alg. 1 (line 114) uses `L_V^f`, which is undefined. Define it as E_mu[f(Q-V)].
3. The hero link `../index.html#paper-444` (line 233) has no anchor in the worktree index.html yet. The batch must add it.
4. `diffusion-ql` is not in the worktree tutorials/ folder. I accepted it per your note, but confirm at build time.
5. Tolerances: §5 says ±0.01 for MC proportions, selfTest uses ±0.02. State the selfTest tolerance once.
6. Copy the shell from tut/569/build/index.html (exists). Frontmatter of all six 444 notes is complete. The 443 and notes_split slugs exist. timeline.json has 14 entries.
