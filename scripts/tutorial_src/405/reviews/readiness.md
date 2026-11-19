NO-GO

Explain-back
DDPM trains on x_t = sqrt(abar) x0 + sqrt(1-abar) eps (Eq. 4) but samples from its own predictions, so errors compound (Tab. 1: ADM FID 0.983 at 100 steps, 3.544 at 1000). The page also covers why fewer steps can win (Tab. 3). DDPM-IP trains on y_t = sqrt(abar) x0 + sqrt(1-abar)(eps + gamma xi) (Eq. 6), target still eps, gamma = 0.1, sampling unchanged. Supporting sections: DDPM-y vs IP (Eq. 7-8, Tab. 7), gamma from nu_t about 0.2, IP as data noising, GP/WD comparison (Tab. 2), results (Tab. 3-5).
- C1: rho-biased toy. Sliders rho, s, t0; metric toggle. Reader sees the std ratio drift from 1, FD grow with t0, and the L1 floor rise even at rho = 0.
- C2: 2D x_t, y_t, eps, eps' diagram; gamma and t sliders. Same input, different targets.
- C3: noise scale in x_t and x0 space, and slope ratio. Reader sees the IP slope change (about 1%) cannot explain the gains.
- C4: tabbed browser over Tabs 1, 3, 4, 5, 7 with derived gains.

Blocking items
1. Alg. 3 line references contradict. design.md 1.4 step 4 says "第 3 行多抽 xi" and "第 4–5 行用 y_t"; the pseudo-code note says "对应 Alg. 3 第 3–4 行". Paper Alg. 3 (src/paper.txt ~line 234): line 3 samples xi, line 4 computes y_t, line 5 is the gradient step. Use one statement everywhere.
2. nu_t dataset mapping unverified. S7, step 6 and notes/diffusion-exposure-bias.md assign ImageNet 32 = 0.20, CIFAR10 = 0.19 "by legend order". Paper text (src/paper.txt line 322) says only "blue and orange curve ... 0.20 and 0.19". R8 is still open. Check the Fig. 2 legend in src/paper.pdf, or tag the mapping 〔本頁判讀〕.
3. Sections 1.1–1.3 map to no page section in Section 3, yet use FID, DDIM, gamma, Tab. numbers and Lipschitz before S3, S8, S10 define them. State whether they are page copy.

Checks run
- All 12 LO answers sit in their designated sections; every component has inputs, ranges, defaults, computation, observations.
- verify/check.py and check_r4.py reproduce the Section 5 numbers I compared.
- The 8 notes have all required frontmatter fields; timeline.json has 14 entries with required fields.
- generative-eval-metrics and eps-prediction-ddpm-score exist as published pages; prediction-targets-k-diff.html template exists.

Non-blocking
1. C2 draw order for x0, eps, xi unspecified; the 200-point cloud is not reproducible.
2. C4 "精確值" has no tolerance; state one for selfTest.
3. "Mixup" used in S6 without definition.
4. The 8 notes exist only under tutorial_src/405/notes/; the build must publish them to tutorials/<slug>.html.
5. Hero link ../index.html#paper-405 depends on JS (scripts/index.template.html:485); confirm a paper-405 entry exists.
6. C1 rho = 0, t0 = 10 is 3.100e-8 with absolute tolerance 1e-8 (about 30% relative); add relative tolerance.
