NO-GO

## Explain-back

The page teaches Implicit Q-Learning (Kostrikov, Nair, Levine, arXiv 2110.06169). Offline Q-learning bootstraps through max over out-of-data actions (Eq. 1), which overestimates. IQL avoids querying those actions: V is fit by asymmetric L2 (expectile tau) over dataset actions only (Eq. 5), then Q by MSE on r + gamma V(s') (Eq. 6). tau = 0.5 gives SARSA; tau -> 1 approaches the in-support max (Lemma 1, Lemma 2, Theorem 3). The policy is extracted by advantage-weighted regression, weight exp(beta(Q - V)) truncated at 100 (Eq. 7). Reported results: Ant Maze 378.0 vs CQL 303.6; locomotion 692.4 vs 698.5; fine-tuning 408.2 to 597.7.

Components: C1 expectile explorer (Q-value and mu inputs, tau slider, toggle for unseen action a6, reads m_tau). C2 single-loss vs IQL (p, q, tau sliders; Q(a2) 7.9412 vs 3, V 7.6471 vs 4.8). C3 toy maze heatmap (tau, sigma, seed, finite vs population mode; reads V_tau(S) against V* = 1.1186). C4 AWR bars and density (beta, tau, truncation; reads mean, ESS, truncated fraction). I can explain every component's mechanics. The blockers below concern definitions and links, not unexplained behavior.

## Blocking issues

1. Linked slugs fail the stated check. `rl-mdp-q-learning-basics` and `offline-rl-distribution-shift` are linked in S1, S12 and 6.4, and appear in `related` of several 569 notes. Neither is in `tutorial_src/569/notes/` or `scripts/notes_split/`. Both exist only as `tutorial_src/568/notes/*.md`, and design R9 says the names may change: "批次去重後可能改名". Fix: confirm the 568 notes are merged into notes_split under these slugs, or pin the names.

2. Symbols used before definition.
   - `V*` first appears in S6 and LO8 ("V* = 1.1186"). The symbol table and S5 define only `Q*`. `V*` is defined only in the implementer spec for C3 (section 4), not in reader-facing text. Fix: add `V*(s) = max_a Q*(s,a)` to S5 or the symbol table.
   - `τ` appears in the S0 hero ("τ 軸（0.5 標「SARSA」、1 標「資料內 Q-learning」") before S3 defines it. Fix: gloss τ in the hero or a footnote.

All 12 learning objectives have a reference answer and a teaching section. The C1-C4 acceptance values were spot-checked against `verify/*_out.txt` and match. The Table 1 and Table 2 numbers match `src/paper.txt`.

## Non-blocking notes

1. Paper Table 2 prints CQL offline total 182.8, but its rows sum to 151.5 + 31.5 = 183.0 (paper.txt lines 606, 610, 611). The design uses 182.8 in LO12 and S10 without comment. Add an "as printed" note.
2. Fine-tuning offline steps: "25000 offline steps" for IQL/AWAC is ambiguous (paper.txt lines 806, 811). The design flags this, but S10 needs a rendering rule (for example, an ambiguity pill).
3. Alg. 1 uses undefined learning rates lambda_V, lambda_Q, lambda_pi. State "Adam, lr 3e-4" next to the code.
4. Section 5 tolerance covers only 4-decimal and 2-3-decimal values. C1 loss values such as 0.04826 and 0.03231 (5 decimals) have no stated tolerance.
5. C2 does not specify the sample count for the tau-axis curves (only the range [0.5, 0.999]). State it so curves match across builds.
6. The C3 21-point curve timing (section 4, C3) is still unmeasured. The design says to record it in section 9, which does not exist yet.
