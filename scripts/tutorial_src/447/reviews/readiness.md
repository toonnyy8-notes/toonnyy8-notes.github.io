NO-GO

**Blocking (1)**

- Symbol used before definition, and the doc contradicts itself on where the diffusion schedule is defined. design.md line 45-49 (symbol table): "$\bar\alpha_i,\beta_i,\tilde\beta_i$ 為擴散排程（S2 定義）". Line 67 (step 2): "$\tilde\beta_i=\ldots$ 與 cosine 排程延到 S3b（玩具說明）才定義". S2 uses $\bar\alpha_i$ in the closed form (line 65) and $\beta_i$ in the reverse mean (line 66), but no page section gives $\bar\alpha_i$ or $\beta_i$ as a formula. The only definition is in §4.0 (line 260, cosine $f(i)$, $\beta_i=\min(1-f(i)/f(i-1),0.999)$, $\bar\alpha_i=\prod(1-\beta_j)$), which is developer spec, not page text. Fix: put the schedule definition in S2 or S3b, and make the symbol-table row match.

**Explain-back**

The page argues that learned-model planners get exploited by their own model errors, and proposes Diffuser: a diffusion model over the whole trajectory array (2 rows by H columns, Eq. 2) with a 1-D temporal U-Net. Planning is sampling from $\tilde p\propto p\,h$ (Eq. 1). Reward maximisation is classifier guidance, $\mu+\alpha\Sigma\nabla\mathcal J(\mu)$ (Eq. 3, Alg. 1). Goal conditioning is inpainting: constrained entries are overwritten after every denoising step. Results: Maze2D 119.5 vs IQL 47.0; blocks 54.4 vs CQL 8.1; D4RL 77.5 vs CQL 77.6.

C1 (sliders N, H, step i, time t; play; resample) shows the reverse chain, the array strip and the variance ratio. C2 (H, N, goal B sliders; toggles; resample) shows the replacement-method mean against the exact conditional mean, plus endpoint jump and rms. C3 (guide mode, log10 alpha slider, target c, H, N; toggles) shows the guided mean against the exact tilted mean, J readings and divergence messaging.

I could explain every part. Checks run: I recomputed the C3 exact tilted means and both program-mean guide modes from the §4.0 formulas in a scratch script. All listed endpoint and J values reproduce to four decimals (for example clean alpha=0.1: endpoint 1.0789 exact, J -34.138; noise-level alpha=0.1: endpoint 0.8862, J -13.095). The 7 notes all carry the 8 required frontmatter keys. All 15 timeline IDs map to TL_GROUP. All related and linked slugs exist, including the 568 notes and notes_split.

**Non-blocking (8)**

1. C3 J readings: the spec does not say J is evaluated on the mean path. Recomputation confirms §5 uses $\mathcal J=-\tfrac12\sum_t\|m_t-c\|^2$ on the mean path, summed over x and y. Say so in C3.
2. Inverse transform: the guide $g$ is computed in DCT space and must be mapped back with $\Phi^\top$. Step 3 of 4.0 does not say so. Recomputation needed $\hat g\Phi^\top$ to match.
3. Sample index: "seed 1000s+j" needs j = 0..5 stated. The §5 values (seed 1000 and 1001) imply 0-based.
4. Table 2: the full table in `<details>` needs all rows, since the bold rule is per task (≥0.95·max). Only the average and a few rows are in the design. Full rows are at paper.txt line 456.
5. PRNG: mulberry32 constants and the uint-to-float mapping are not written. Cite the canonical code so the §5 value (0.627073941…) is reproducible.
6. Colour mapping for the diverging array-strip scale (clamp range, interpolation) and the hero's "固定 seed" are unspecified.
7. C1 sampling: §4.0 step 4 says "若取樣", but C1 should state explicitly that the chain samples (noise on), while C2/C3 program means use z=0.
8. Not verified in this read: `../index.html#paper-447`, the shell template `tutorials/prediction-targets-k-diff.html` (§6.1), and the S7 SVG layout details.

Report file: /tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/wt-paper-master/scripts/tutorial_src/447/reviews/readiness.md
