NO-GO

Scope: design.md (486 lines), notes/timeline.json, and the linked note slugs. Checked against the readiness criteria in /tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/tp/568_readiness.md.

## Explain-back

The page teaches Conservative Q-Learning (arXiv 2006.04779) in 17 sections (S0-S16). Argument chain: offline actor-critic overestimates Q on OOD actions, because the Bellman target uses the learned policy's actions while Q is trained only on data actions (S2). Eq. 1 adds a push-down term alpha*E_mu[Q], giving the per-entry fixed point Q = B Q - alpha*mu/pi_b (Eq. 11) and a pointwise lower bound (Thm. 3.1). Eq. 2 adds a push-up term on the data policy; the tabular update B Q - alpha(mu/pi_b - 1) (Eq. 13) only bounds the expectation under pi, via D_CQL >= 0 (chi-square identity, S5). Sampling error sets an alpha threshold that shrinks with 1/sqrt(N) (S7). CQL(H) is the log-sum-exp form (S8, S9). Later sections cover the slow-policy condition (Thm. 3.3), gap expansion (3.4), penalized objective (3.5), safe improvement with zeta (3.6), implementation, and results.

Components: C1 single-state, per-action Q axes with alpha and pi presets. C2 three-state MDP with alpha, gamma, k sliders; shows the constant -alpha/(1-gamma) gap and iteration curves. C3 closed-form P(V_hat > V) and alpha_delta versus N, sigma, delta. C4 four-action softmax with alpha or tau modes; shows mu, the gap, and the alpha*chi^2 identity.

I recomputed C2's default values under both readings of the s1 transition; the table matches the reading "s1 right goes to s2 with 0.8, to s0 with 0.2". I also spot-checked C3 (N=10, 100, 1000 values, the N=10 boundary case) and C4 (alpha=1 point: stationarity, E_pi_b, gap, and increase = alpha*chi^2). All match.

## Blocking items

B1. Undefined symbol B* (optimal Bellman operator). It is used in running text but never defined. The symbol table in 1.4 defines only B^pi.
- Line 117 (step 6, S8/S9): "CQL 可用 $\mathcal B^*$ 寫成 Q-learning"
- Line 157 (step 12, S13): "（Q-learning 用 $\mathcal B^*$，actor-critic 用 $\mathcal B^{\pi_{\phi_t}}$）"
Fix: add B* to the 1.4 symbol table and to the S8 or S13 "本節先定義" list.

B2. Table 3 Asterix (1%) cell is not settled. The page table needs a single CQL value.
- Line 179: "Asterix 592.4（REM 386.5；PDF 文字層錯位，正確讀法 QR-DQN 166.3、REM 386.5）"
- R4 (line 442): "Table 3 用正文的 setting (2) 與正確對齊值"
The design says the PDF is misaligned and gives the corrected QR-DQN and REM values, but never states the aligned CQL value. It is unclear whether 592.4 is the CQL value or itself misaligned. Fix: state the exact CQL, QR-DQN and REM cells for Asterix 1%, with the source (paper.txt line or table location).

## Non-blocking notes

1. Symbol metadata: the 1.4 table lists d^pi_M first appearing in S12, but d^{pi_b} is used in S2 and S3 (step 1 and step 2). The definition is present in the table, so this is metadata only.
2. R5 says "9 列混合資料中 3 列是輸或接近持平". Step 13 lists 1.2x for halfcheetah-medium-expert (62.4 vs 51.7), which is a clear win, not a near tie. Reword R5 to "1 loss, 1 near tie, 1 modest 1.2x win".
3. The C2 MDP line "s1 左→s0、右→s2（0.8）或 s0（0.2）" is parseable but ambiguous. Write it as "s1, right: to s2 w.p. 0.8, to s0 w.p. 0.2".
4. C3 erfc is specified only as "Numerical Recipes erfcc, 係數同 check.py". Inline the coefficients or say where to get them, since a front-end build cannot read check.py.
5. S14 Table 1 "9 列混合資料全列" gives only a few cells. The column set (all baselines) and the full 9-row values are not enumerated; the builder must copy them from src/paper.txt. Name the columns explicitly.
6. Undefined or unspecified terms: log-sum-exp is used in S8 without being in its "本節先定義" list; G_Q and G_pi appear in step 12 (only "未寫明" is stated later); the C2 k slider has no step size.
7. C4 tau mode: the Q-axis behavior for a finite alpha* is not stated. Only the alpha = infinity case (draw alpha = 1e4) is specified.
8. Build dependency: section 6.1 depends on tut/406/build/index.html as the shell source. The file exists, but the handoff should say which parts are copied versus new.
