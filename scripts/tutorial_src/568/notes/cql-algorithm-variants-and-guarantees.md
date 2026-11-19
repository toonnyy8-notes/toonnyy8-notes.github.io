---
slug: cql-algorithm-variants-and-guarantees
title: CQL(R) 的 min-max 形式、CQL(H) 與 CQL(ρ)，以及定理 3.3 至 3.6
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [CQL, min-max, CQL-H, gap-expanding, safe-policy-improvement]
arxiv: ["2006.04779", "1801.01290", "1312.5602", "1710.10044"]
sources: ["arXiv:2006.04779 §3.2, §3.3, §4", "arXiv abstract 1801.01290", "arXiv abstract 1710.10044"]
related: ["cql-conservative-policy-evaluation", "rl-mdp-q-learning-basics", "cql-vs-policy-constraint-methods"]
---
# CQL 的 min-max 形式、變體與保證

**從 evaluation 到演算法。** [式 2](cql-conservative-policy-evaluation.html) 在 `μ = π` 時給出下界。對每個策略做完整 evaluation 再改進一步太耗時，所以 CQL §3.2 讓 `μ` 逼近能最大化目前 Q 的策略，得到對 `μ` 的 min-max 問題，稱為 CQL(R)：

`min_Q max_μ α(E_{s~D,a~μ}[Q] − E_{s~D,a~π̂_β}[Q]) + ½ E[(Q − B̂^{π^k} Q̂^k)²] + R(μ)`（式 3）

**CQL(H)。** `R(μ) = −D_KL(μ, ρ)` 時，最佳 `μ(a|s) ∝ ρ(a|s)·exp(Q(s,a))`。`ρ = Unif(a)` 時第一項成為 Q 值的 log-sum-exp（soft maximum）：

`min_Q α E_{s~D}[log Σ_a exp(Q(s,a)) − E_{a~π̂_β}[Q(s,a)]] + ½ E[(Q − B̂^{π^k}Q̂^k)²]`（式 4）

**CQL(ρ)。** `ρ` 取前一個策略 `π̂^{k−1}` 時，式 4 的第一項改為從 `π̂^{k−1}` 取樣動作的指數加權平均。論文觀察到動作維度高時（如 Adroit）估計 `log Σ exp` 的取樣變異大，CQL(ρ) 較穩定。Appendix A 另有一個與 distributionally robust optimization 相關的變體。CQL 可用 `B*`（Q-learning）或 `B^π`（actor-critic）實作。

**定理 3.3。** 設 `π_{Q̂^k} ∝ exp(Q̂^k)` 且 `D_TV(π̂^{k+1}, π_{Q̂^k}) ≤ ε`（策略變動慢）。若 `E_{π_{Q̂^k}}[π_{Q̂^k}/π̂_β − 1] ≥ max_{a: π̂_β>0}(π_{Q̂^k}/π̂_β)·ε`，則 `V̂^{k+1}(s) ≤ V^{k+1}(s)`。左式是保守程度，右式是策略與 soft-optimal 策略不同造成的最大高估量。論文因此把策略學習率設為 3e-5，低於 Q 函數的 3e-4 或 1e-4。

**定理 3.4（gap-expanding）。** `α_k` 夠大時，CQL 使 `E_{π_β}[Q̂^k] − E_{μ_k}[Q̂^k]` 大於真實 Q 的對應差值，所以策略 `∝ exp(Q̂^k)` 被推向資料分布 `π̂_β`，隱含地避開 OOD 動作。

**定理 3.5、3.6（§3.3）。** 設 `Q̂^π` 為式 2 的不動點，則 `π* = argmax_π J(π, M̂) − α/(1−γ) · E_{s~d^π_{M̂}}[D_CQL(π, π̂_β)(s)]`，其中 `D_CQL(π,π_β)(s) = Σ_a π(a|s)(π(a|s)/π_β(a|s) − 1)`，`M̂` 為資料集誘導的經驗 MDP。定理 3.6 指出 `π*` 對 `π̂_β` 是 `ζ`-safe policy improvement：`J(π*, M) ≥ J(π̂_β, M) − ζ`，機率至少 `1−δ`。論文將其與 Laroche 等人的 SPIBB 結果類比：對 horizon 為二次依賴、對計數為平方根倒數依賴。

**實作（§4、Algorithm 1）。** 每步用 `G_Q` 次梯度更新訓練式 4 的目標；actor-critic 版再以 SAC 風格的熵正則化更新策略。論文稱只需在 SAC（連續控制，arXiv:1801.01290）或 QR-DQN（離散控制，arXiv:1710.10044）上加約 20 行程式；連續控制的 `α` 以 Lagrangian dual gradient descent 自動調整，離散控制固定。因為懲罰直接作用在 Q 函數，CQL 不需要策略限制，也不需要另外估計行為策略。
