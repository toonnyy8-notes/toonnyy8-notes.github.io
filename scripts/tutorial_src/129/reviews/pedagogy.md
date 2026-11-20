# Pedagogy review — No.129 (no P0)
1. P1 S1/S4/S7 prerequisites missing: no log-derivative trick (∇E[R]=E[R∇log p]), advantage, value/critic (LO9 compares), covariance, entropy, CE/KL. Fix: one-line derivation in S1; define covariance at S4 start; entropy+CE in S5; one-sentence value before S7 baseline paragraph.
2. P1 S8 too dense (608KB/20TB, Eq.11,7,8,6, code, eps, Eq.9,10,12, two checks, C4; order 11→7→8→6 tangled). Fix: split S8a (Eq.11+sg, LO10) / S8b (top-k+stacking, LO11, C4); raise budget to 900+ or move Eq.9–12 into <details>.
3. P1 S4 too dense (symbol table, fixed prefix, E_p[s]=0, s̄, covariance, Eq.3, s_v=e_v−p, toy, C1a). E_q[R|prefix] unexplained in plain words. Fix: S4a (E_p[s]=0, LO4) / S4b (Eq.3+toy, LO5); walk a 3-token numeric example before C1a.
4. P1 Forward references: SC curves in C3 (S5) and C2 row "SC: c=0" (S6) before SC defined in S7; group-centering/advantage used in S3 LO3 answer. Fix: define group centering+advantage in one sentence in S3; hide SC curves by default or label "explained in S7".
5. P1 C3 linear x-axis 0–3000 hides early collapse (t½ ≈ 12–29 steps). Fix: symlog x-axis or 0–100 inset; spec t½ "not reached" for SC/fixed teacher.
6. P1 C4 relative residual 0/0 at σ=0 → NaN. Fix: if ‖q−p‖<1e-12 show "— (no drift)".
7. P1 Small errors display as 0.0000 (C1 decomposition error, C4 identity error). Fix: toExponential(1) + "< 1e-9 ✓".
8. P1 Symbol collisions: Q_T,P_T vs Q_v; tokens A–D / prefixes A,B vs A_i; a,b vs α; H set vs H(p); T tail vs T=3000; ε slider vs eps; L loss vs L_i. Fix: m_q^T/m_p^T; Ent; N steps; tokens 1–4; 路徑甲／乙; floor δ.
9. P1 390px: C4 20 bars+labels; Table 1 6 cols; .pg two; Eq.8/12; long code lines. Fix: narrow → label only head/tail boundary, values in readout; .katex-display overflow-x:auto; .pg two single column ≤600px with scrollable pre; selfTest checks scrollWidth per component.
10. P1 Keyboard: C4 hover/touch only. Fix: roving tabindex + arrow keys, readout on focus, "show as table" toggle; same for C1, C2.
11. P1 KaTeX in SVG unspecified. Fix: plain Unicode in SVG text, KaTeX only in HTML; lazy per-section render via IntersectionObserver.
12. P2 Literal style: S3 title personification, "兩條平行的軌道", "teacher 跟著 student 動", "拉向／推離"; S9 title overgeneralizes (FP8 KV plain PG also stable 58%); S5 title states argument as fact. Fix: S3 "SFT 在離線資料下穩定；線上 RL 在 TIM 下不穩定"; S5 "論文論證：drift 是朝 sampler 的蒸餾，並在回饋迴路中累積"; S9 "實驗：量化越嚴重，SC 與其他方法的差距越大（0.6B 與 30B）"; motif "trainer 與 sampler 對照圖".
13. P2 Hero cards use undefined terms (SC, TIS, 30B, 量化); 推論／訓練引擎 vs sampler/trainer. Fix: card 3 "減去平均值後，drift 為 0；30B 嚴重量化設定下訓練準確率 30%，次佳方法 12%"; first use "推論引擎（sampler，q）" then sampler/trainer.
14. P2 Full Table 1 in S2 too early. Fix: only TIS/MIS rows in S2; full table in <details>.
15. P2 C3 ‖z‖<10 claim only checked at η=1; no reduced-motion rule; SVG pattern stroke not tokenized. Fix: drop ‖z‖ guarantee, state softmax max-subtraction; no transitions under reduced-motion; pattern stroke currentColor/var(--ink-2).
