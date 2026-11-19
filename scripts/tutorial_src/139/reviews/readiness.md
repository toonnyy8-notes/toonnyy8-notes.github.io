NO-GO

## Explain-back

The page teaches why distributional RL should enforce Bellman consistency at every flow noise level tau, not only at tau = 1. The argument: Eq. 1 is the distributional Bellman equation; Lemma 1 turns flow matching into velocity regression. Prior methods apply consistency only at the endpoint (FlowCritic), at every tau but wrongly at tau = 0 (Value Flows), or along shared-noise paths (PCBF). Value Forcing's flow Bellman operator (Eq. 5), Z(tau) = tau r + gamma Z'(tau) + (1 - tau) sqrt(1 - gamma^2) N, keeps N(0,1) at tau = 0, is a gamma-contraction per tau (Remark 1), and has fixed point tau Z + (1 - tau) G. Splitting into V and Q velocity losses (Eq. 11-13) gives VF (K-candidate argmax, Eq. 14) and IVF (expectile weights from scalar Q, V, Eq. 15-16).

Components: C1 (one-step backup per tau; gamma, tau, r, m', sigma' inputs; W2 readout). C2 (iterated operators on a 2-state toy with 400-atom particles; gamma, tau, k inputs; W1 and contraction-ratio readouts). C3 (Q-loss regression target; mode radio; slope and intercept readouts). C4 (expectile weights vs mu; gap between envelope and max-velocity flow). I recomputed the C3 defaults independently: correct slope -1.0969, intercept -4.3357; +eps slope -0.5085; r-times-tau intercept -3.8357. All match. Every component is explainable.

## Blocking

1. **S12 main data is not in the design.** S12 specifies "OGBench 總平均表（4 資料量 × 13 方法）" and an OCBench table. Design section 5 gives only VF, IVF and the best two baselines per budget, and defers the rest to `numbers_out.txt`. The 13-method values, method names, and OCBench values are missing. Abbreviations such as "QC" are never expanded. Fix: paste the full tables with method names into design section 5.

2. **LO12 reference answer is not assigned to the body.** The LO12 reference includes "拿掉 reward scaling：VF 約 86% → 11%、IVF 76% → 11%". Section 2's preamble requires every reference answer to be written in its chapter body. S12's body list only says "消融段落（44% 照原句）" and does not mention the reward-scaling figures. Fix: add them to the S12 row, and note that the 86% comes from Fig. 5 under a different protocol from Table 5 (design section 1.4, step 10).

## Pass checks

- All 12 linked note slugs resolve. Six are in `tutorial_src/139/notes/`; `rl-mdp-q-learning-basics`, `offline-rl-distribution-shift`, `expectile-regression-value-learning` and `iql-lineage-overview` are in `notes_split/`; `flow-matching-velocity-target` is in `notes_split/` and `tutorial_src/238/notes/`; `diffusion-policy-class` is in `tutorial_src/443/notes/`. Hub `index.html#paper-139` is exempt.
- All six `139/notes/*.md` files have the eight required frontmatter keys (slug, title, cluster, kind, tags, arxiv, sources, related).
- Every learning objective (LO1-LO12) has a reference answer and a teaching section.
- Tolerances are stated in section 5. Component inputs, ranges, defaults and computations are specified.

## Non-blocking notes

1. invNorm: the design says "係數照 common.py" but does not include the coefficients. Any inverse-normal with error below 1e-8 reproduces the C4 values within tolerance, so state that or paste the coefficients.
2. The hero card 3 and the S0 subtitle use VF, IVF, "explicit" and "implicit" before S8 defines them. Add a one-line gloss.
3. S11 cites the 81% and 89% PCBF reductions through "讀圖注意（1.4 步驟 10）", which is a pointer to design notes. Write the sentence into the S11 body.
4. VF training uses a one-step Q estimate while evaluation uses 10 steps (step 9). The design states this as an interpretation; mark it in the page as a local reading.
5. The C2 noise table and the mulberry32 seeds are fully specified; no action needed.
