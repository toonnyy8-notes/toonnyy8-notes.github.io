---
slug: expectile-regression-value-learning
title: Expectile regression 與 value learning：以非對稱 L2 損失近似資料支撐內的最大值
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [expectile-regression, quantile-regression, distributional-rl, iql, value-learning]
arxiv: ["2110.06169", "1710.10044", "1806.06923", "2005.04269"]
sources: ["arXiv:2110.06169 §2, §4.1, §4.2, §4.4, Appendix A", "Dabney 等、Kuznetsov 等的 arXiv 摘要", "Koenker 與 Hallock 2001（非 arXiv，僅依 IQL 論文的引用）"]
related: ["rl-mdp-q-learning-basics", "offline-rl-distribution-shift", "iql-lineage-overview", "one-step-offline-rl-and-sequence-modeling", "iql-algorithm-and-d4rl-evaluation"]
---
# Expectile regression 與 value learning

## 統計工具

隨機變數 X 的 τ ∈ (0, 1) expectile 是下列非對稱最小平方問題的解（IQL 論文 §4.1）：

`m_τ = argmin_m E[L_2^τ(x − m)]`，`L_2^τ(u) = |τ − 1(u < 0)| u²`

τ = 0.5 時等於一般的 MSE，結果是平均值；τ > 0.5 時，小於 m_τ 的樣本權重較低，大於 m_τ 的樣本權重較高。條件版本為 `argmin_{m_τ(x)} E[L_2^τ(y − m_τ(x))]`，可用 SGD 優化，梯度無偏。它與 quantile regression（Koenker 與 Hallock，2001）相近，差別在 quantile regression 使用非對稱 ℓ1 損失。論文腳註指出，方法也可用 quantile 推導，但只估計單一 expectile 時，對 MSE 作小修改較方便，且實驗上略優於 ℓ1 的 quantile regression。

## 與 distributional RL 的差別

quantile regression 已用於 RL，以估計狀態動作價值的 quantile function：QR-DQN（Dabney、Rowland、Bellemare、Munos，arXiv:1710.10044，2017）、IQN（Dabney 等，arXiv:1806.06923，2018）、TQC（Kuznetsov 等，arXiv:2005.04269，2020）。論文說明差異：這些方法估計的是隨機轉移造成的價值分布；IQL 估計的是狀態價值對隨機動作的 expectile，並對轉移的隨機性取平均。論文也指出 distributional RL 用於 offline RL 時，仍有相同的 action extrapolation 問題（見 [distribution shift](offline-rl-distribution-shift.html)）。

## 用於 value learning

目標是式 (4)：以資料集支撐內的最大值取代 `max_a' Q(s', a')`，亦即 `max_{a' : π_β(a'|s') > 0} Q(s', a')`。若直接對 TD 目標做 expectile regression，目標會同時包含動態的隨機性，大目標值可能只是「幸運」轉移到好狀態。論文改用兩個損失（式 5、6）：

- `L_V(ψ) = E_{(s,a)~D}[L_2^τ(Q_θ̂(s,a) − V_ψ(s))]`：V 只對動作分布取 expectile。
- `L_Q(θ) = E_{(s,a,s')~D}[(r + γ V_ψ(s') − Q_θ(s,a))²]`：Q 以 MSE 對轉移取平均，目標中沒有未見動作。Q-learning 的 Bellman backup 見 [MDP 與 Q-learning](rl-mdp-q-learning-basics.html)。

## 理論（§4.4、Appendix A）

- Lemma 1：有界支撐的隨機變數，支撐上確界為 x*，則 `lim_{τ→1} m_τ = x*`。
- Lemma 2：τ1 < τ2 時，V_{τ1}(s) ≤ V_{τ2}(s)；證明依循 policy improvement 的證明（Sutton 與 Barto，2018）。
- Corollary 2.1：V_τ(s) ≤ 受資料支撐限制的 max_a Q*(s, a)。
- Theorem 3：`lim_{τ→1} V_τ(s) = max_{a : π_β(a|s) > 0} Q*(s, a)`。

論文因此將方法稱為 implicit Q-learning。τ 越接近 1，越接近最大值，但優化更困難，所以 τ 當作超參數；τ = 0.5 對應 SARSA，τ → 1 對應 Q-learning。

與 [單步方法](one-step-offline-rl-and-sequence-modeling.html) 比較：τ = 0.5 時等價於學行為策略的價值，提高 τ 才使價值函數能傳遞多步資訊。完整演算法見 [IQL 演算法與 D4RL 評估](iql-algorithm-and-d4rl-evaluation.html)，脈絡見 [總覽](iql-lineage-overview.html)。
