# Fidelity review, tutorial 119 (LOPD)

verify/check.py was re-run and its output is identical to check.out. The tables, SE values and gradient/fixed-point checks reproduce. Spot checks of Tables 1-3, the 8B-gap arithmetic, Eq./Alg./App. numbering and the Fig. 3/5 values against paper.txt found no mismatch. The problems below are in what the scripts do not test.

1. P0, coverage claim. design.md 1.3: "沒有任何 backbone 跑滿七個 benchmark（...只有 Qwen3-4B 兩類都有）". This contradicts itself. Qwen3-4B has Env, BFCL, ACE (Table 1) plus LCB v5, v6, HE+, MBPP+ (Table 2), which is all 7 under the paper's own count (§4.1). Paper: "three model backbones and seven benchmarks". The paper's claim is a union, not a full grid. Fix: "不是 3×7 全格：4B 全部 7 個，8B 只有 3 個工具類，Olmo3 只有 4 個程式類". Also change the S11 row "三 backbone 七 benchmark" to match.

2. P1, LO6 and S6. "為什麼不加權的平均 δ 不可能為正" and "Δ 要達到 m 必須靠第二項". E[δ]=−KL≤0 is an expectation, not a bound on the sample mean. The design's own decomposition has first term −Q̄·KL, which is positive when Q̄<0. Table 3 gives Ā=−0.028 for vanilla, so the second-term-only claim is not general. Paper: Eq. 11-12 only. Fix: "期望上不為正（取樣平均可為正）", and "當 Q̄≥0 時 Δ 為正必須靠共變項".

3. P1, E[δ]=−KL assumption. The derivation needs actions sampled from exactly p^S. App. B.2/Table 4 use temperature 0.7, top-p 0.95, top-k 20. design.md's "未說明" list admits the temperature used for p^S, p^T is unknown. Fix: state "僅當取樣分布＝δ 中的 p^S 時成立" in S6, LO6 and C2's assumption line.

4. P1, C3 narrative (4.C3). "r̄ 全部相同時 q* = p 且 Δ 恆為 0，對偶模式下 β 每步增加 0.5·m". For constant Q, Δ=−Q·KL(p‖q), which is nonzero until q=p. It is exactly 0 only for Q=0 (preset B). With Q<0, Δ=+|Q|·KL, so margin is satisfied early by a context carrying no reward information. Fix: "q*=p；Δ=−Q·KL，Q=0 時恆為 0，β 每步 +0.5m 僅在 Δ=0 時".

5. P1, β bound. 步驟 8: "β 至多約 1.25" (50 steps × 0.025). That holds only if Δ≥0. Δ can be negative (C2 itself shows c=−0.0842), and then the increment is 0.5(m−Δ)>0.025. Also, 50 steps assumes one dual update per 32 rollouts and Fig. 4's x-axis counts 32 per step. Both are inferences (paper: Alg. 1 line 14, B.1). Fix: tag it as the Δ=0 case, not an upper bound.

6. P1, m=0 (S12 mistake 7 and LO9). "m=0 仍是約束 Δ≥0" can read as a nontrivial constraint. At collapse π^T=π^S, Δ=0≥0, so the collapse point is feasible and the paper's exclusion ("Δ→0<m") needs m>0. Fix: add this sentence. It also explains why C3 with m=0 collapses to q=p.

7. P2, Ā conversion. 步驟 7 "以 Table 3 換算整體平均 Ā: 2(0.486)−1=−0.028、2(0.637)−1=0.274". These are test-set rewards of the final policies, not training-rollout means, and Q̄ is per-prefix. "成功率過半" is also wrong for EnvScaler, where r is a continuous subtask fraction (App. B.2). Fix: say "平均獎勵過半"; label the numbers as test-set proxies.

8. P2, "凍結 composer" wording. 1.1/S8 title: "學生（0.551）低於凍結 composer（0.573）". 0.573 is a student trained with the frozen φ0 (App. B.4: "trains with a frozen composer"), not the composer or a teacher. Fix: "以凍結 composer 訓練出的學生".

9. P2, LO11. "預算不含建經驗庫與冷啟動". The paper does not say. A.2/A.3 only show that these use extra base-model rollouts. Fix: tag as inference ("論文未說是否計入").

10. P2, Fig. 4 provenance. B.4: "Mean reward is periodically evaluated on the EnvScaler test set". The design's Fig. 4 text says only "圖說 Mean reward over 1,600 rollouts". This bears on the Table 1 vs Fig. 4 mismatch (R3) and on the "30%" claim. Fix: note that Fig. 4 is a test-set evaluation curve, so checkpoint/evaluation-protocol differences cannot be excluded.

11. P2, hyperparameter selection. 步驟 12 "K、n_ret、m 依 EnvScaler 測試集上的敏感度挑選". The paper says this for K and n_ret (§4.3). For m it gives no selection statement; Fig. 3 only shows 0.05 is best. Fix: "m 的選法未說明，Fig. 3 在測試集上".

12. P2, C4 yardstick. "差距÷標準誤" and the "落在 ±1.96 SE 區間之內" text. Both models run on the same questions, so the single-score binomial SE over questions overstates the paired-difference uncertainty. The assumption line mentions pairing but the readout still invites "not significant". Fix: label it "單一分數的題目取樣 SE，不是配對檢定"; avoid the "之內/之外" verdict.

13. P2, Fig. 6. The case-study task ids are "env_174_rl-task_29" and "taco_00713", which look like training-corpus ids. Rank 2 is shown in both. This bears on the R5 self-retrieval question (rank 1 may be the task itself), which the design does not connect. Fix: add a note, tagged as 疑問.

14. P2, step-6 gradient. "這個方向與脈絡有沒有資訊無關" holds only for the logit-space gradient in the fixed-student, free-logit model. Elsewhere the student also moves toward the teacher, and the real gradient passes through ∂u/∂φ. S5 does caveat this, but the sentence is stronger than the caveat. Fix: "在此理想化模型中，損失梯度本身不含脈絡資訊".
