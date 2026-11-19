# 570 Fidelity review (Goldfish)

Scripts: toy.py, c3.py, c4.py rerun and match the design.md acceptance tables. c2.py has an import problem when run with -I (needs sys.path); its output matches c2_out.txt. All 15 per-row Table 2/7/9 values I spot-checked match paper.txt. Problems are below.

1. P0. Location: sec 4.0 "〔本頁推導〕跳躍：… J''=2λc₂/s_a²−2 … 內部局部最大值消失 … π* 由內部跳到 a=1"; also R11 "π* 不連續", C1 觀察 (3) "跳到 a=1", C2 "c≥1 跳到 a=1", "b=−2 跳到 1". Paper: nothing (toy derivation). I ran toy.py seed 7: π* = 0.699 / 0.783 / 0.874 / 0.977 / 1.0 at α = 3 / 3.1 / 3.2 / 3.3 / 3.4. J'' stays negative until α≈5.1 (λ≈1.15; J''=−0.62 at α=3.5). π* is continuous. The interior maximum slides past a=1 and is clamped by the action bound at α≈3.3. J is not convex there. Fix: replace the explanation with "the concave interior optimum π−ā = c₁λ/(s_a(2−2c₂λ/s_a²))·… exceeds the boundary, so π* is clamped at 1". Drop "不連續/跳躍" and "目標變凸". Convexity appears only for λ>s_a²/c₂≈1.15.

2. P1. Location: 步驟 5 "尺度不變 … 最佳策略不變", LO8 answer, C2 "正規化時 π* 對所有 c 都是 0.381, g 恆為 0.6576". Paper (Sec. 5): Eq. 5 "balance … susceptible to the scale of Q", "normalizing the learning rate across tasks". It claims no invariance. The widget uses the closed form Q̂_{c,b}=c(Q̂+b), so exactness is built in. A real TD3 critic (initialization, Adam, bootstrapping, γ, termination masks) is not exactly c·Q. Fix: state "在理想擬合下恰好不變；實際網路只近似", and label C2 as an idealization.

3. P1. Location: LO8 and C2 平移 "獎勵加上常數 … b". In a multi-step MDP a reward shift b shifts Q by about b/(1−γ) (≈100b at γ=0.99), plus termination effects. The one-step toy has no γ. Fix: add that caveat. Do not let the b slider read as a reward offset of the real task.

4. P1. Location: C4 "標準差較小", LO10 "979.3±33.4", step 8 "± 為 seed 間標準差", "Fisher-BRC 標準差 108.3 主要來自 Hopper-me ±43.3 與 Walker2d-e ±32.4". Paper (Table 2): "± captures the standard deviation over seeds". I summed the per-row ±. TD3+BC = 33.4, Fisher-BRC = 108.3, CQL = 61.5, BC = 91.5. Every Total ± is exactly the sum of the 15 per-row ±, not the std of the summed score. The "主要來自" decomposition is therefore arithmetic, not a variance share. Fix: write "Total 的 ± 等於各列 ± 之和（本頁核算）", and do not call it seed std. Same for Table 9 totals.

5. P1. Location: C3 "梯度下降的收斂 … Eq. 4 改善條件數" with η=1/λmax. Paper (Table 4): Adam optimizer. Adam's per-parameter scaling removes much of the pure scale mismatch that κ measures. The mean offset and correlations remain. Plain GD overstates the effect (κ 3182 vs 3, 6987 vs 3 steps). R6 mentions deep nets but not Adam. Fix: say explicitly "實際用 Adam；此處為 GD 的玩具，量級不可外推", and drop the step counts as evidence.

6. P1. Location: C1 觀察 (4) "200 個 seed 中 192 … 177 個 α=2.5 勝過純 BC，正文寫『多數 seed』並引用這兩個數字". These are statistics of constants calibrated "為了讓 α=2.5 落在好的區域" (R1). Quoting them in prose reads as evidence for α=2.5. Paper evidence: App. B and Fig. 6 only. Fix: if kept, mark them "推定常數下", or drop the counts.

7. P1. Location: LO6 answer "TD3+BC：a=0.381，4.83" as a graded answer. It is seed-7 and calibration specific. In the same table α=2 gives 4.99, and at α=3 TD3+BC (4.25) is below pure BC (4.30). Fix: add "(α=2.5, seed 7, 推定玩具)" to the answer. Mention that the benefit is non-monotone in α.

8. P2. Location: LO3 "TD3 的 critic 目標用了哪三個元件？ … 每 2 步更新一次 actor". Delayed actor updates are not part of the critic target (the design's own step 2 correctly says (a)(b) target, (c) update schedule). The paper lists only the hyperparameters in Table 4. Fix: "TD3 的三個設計：clipped double-Q 目標、目標平滑、延遲更新".

9. P2. Location: 步驟 2 "〔本頁推導〕…這是 Eq. 3 只改 actor 也能約束 critic 目標的理由". The paper makes no such claim. BC only constrains π′ at dataset states s′, and only up to network generalization. "約束" is overstated. Fix: "傾向使目標動作靠近資料，並無保證".

10. P2. Location: 步驟 7 pseudocode line "λ ← 2.5 / mean|Q(s, ·)|（不微分）  # 新增（Eq. 5）". Eq. 5 is mean|Q(s_i,a_i)|. The pseudocode silently uses the official-code variant. Fix: write mean|Q(s,a)| per Eq. 5 with a comment "官方程式碼用 Q(s,π(s))（論文外）".

11. P2. Location: 步驟 1 "離線 RL（歷史上稱 batch RL）". The paper says "fixed batch of data" and does not equate the names. This is unmarked outside knowledge. Fix: tag 背景（論文外）.

12. P2. Location: 步驟 10 "限制（Sec. 7、App. A）". Appendix A is "Broader Impact", and the "gain is minimal" sentence is there. Sec. 7 is the Conclusion. Fix: cite "App. A（Broader Impact）".

13. P2. Location: hero/1.1 "差 4.7" and the C4 note "Fisher-BRC 975.2 vs 974.6". The comparison mixes the paper Total with the recomputed row sum. Fix: cite one basis. Row sums give a gap of 4.1.

14. P2. Location: 4.0 "最大值 … 外插誤差來源只有迴歸外推" and C1 "高估約 3.2". The toy overestimation comes from an upward-convex quadratic that is chosen because the data are biased left of the optimum. The paper's mechanism is bootstrapped OOD value error (Sec. 1, 3: "generalization error in the approximate value function"). This is disclosed, but the page should not call it "外插誤差的再現".

15. P2. Location: 步驟 8 "Fig. 8: DT 只訓練 100k 步 1h 38m, DT(Fig.8)". Table 9's DT numbers use 3 seeds vs TD3+BC 5 seeds (paper footnote to Table 9). The 677.4 vs 672.6 comparison omits this. Fix: add "DT 3 seeds".
