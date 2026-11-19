---
slug: rl-mdp-q-learning-basics
title: MDP、Q 函數與 Bellman backup：value-based RL 的基本記號
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [MDP, Q-learning, Bellman-operator, actor-critic, policy-evaluation]
arxiv: ["2006.04779", "1312.5602", "1802.09477", "1801.01290"]
sources: ["arXiv:2006.04779 §2 Preliminaries", "arXiv abstract 1312.5602", "arXiv abstract 1802.09477", "arXiv abstract 1801.01290"]
related: ["offline-rl-distribution-shift"]
---
# MDP、Q 函數與 Bellman backup

本篇整理 Conservative Q-Learning（CQL，arXiv:2006.04779）§2 使用的記號，供其他 offline RL 筆記引用。

**MDP。** 環境以 `(S, A, T, r, γ)` 描述：狀態空間 `S`、動作空間 `A`、轉移機率 `T(s'|s,a)`、獎勵 `r(s,a)`、折扣因子 `γ ∈ (0,1)`。目標是最大化期望折扣報酬。CQL 另假設 `|r(s,a)| ≤ R_max`。

**Q 函數。** `Q(s,a)` 表示在狀態 `s` 採取動作 `a` 之後的期望折扣報酬。off-policy、以動態規劃為基礎的演算法維護參數化的 `Q_θ(s,a)`，部分方法另外維護參數化策略 `π_φ(a|s)`。

**Bellman 運算子。** CQL 使用兩種：

- Bellman optimality operator：`B*Q(s,a) = r(s,a) + γ E_{s'~P(s'|s,a)}[max_{a'} Q(s',a')]`。Q-learning 反覆套用它，並以精確或近似的最大化（論文舉 CEM 為例）取得 greedy 策略。
- 策略 `π` 的 Bellman operator：`B^π Q = r + γ P^π Q`，其中 `P^π Q(s,a) = E_{s'~T, a'~π(·|s')}[Q(s',a')]`。

**Actor-critic 的兩個步驟。** 交替進行 policy evaluation（反覆套用 `B^π` 求 `Q^π`）與 policy improvement（把 `π` 往最大化期望 Q 值的動作更新）。

**經驗 Bellman operator。** 資料集 `D` 通常不含所有 `(s,a,s')` 轉移，所以 policy evaluation 實際使用的 `B̂^π` 對每個轉移只做單一樣本的 backup。論文給出的更新（`k` 為迭代次數）為：

- policy evaluation：`Q̂^{k+1} ← argmin_Q E_{s,a,s'~D}[(r(s,a) + γ E_{a'~π̂^k(a'|s')}[Q̂^k(s',a')] − Q(s,a))²]`
- policy improvement：`π̂^{k+1} ← argmax_π E_{s~D, a~π^k(a|s)}[Q̂^{k+1}(s,a)]`

用來產生資料的策略稱為行為策略 `π_β`，`D` 取自 `d^{π_β}(s) π_β(a|s)`。

**與本論文相關的 online 演算法。** CQL 的實驗以 SAC（arXiv:1801.01290，off-policy actor-critic）為底；TD3（arXiv:1802.09477）處理 actor-critic 中的函數近似誤差。DQN（arXiv:1312.5602）是以深度網路實作 Q-learning 的早期工作。這三項的內容僅依其 abstract 與 CQL 的描述。

這套記號在 offline 設定下遇到的問題見 [Offline RL 的分布偏移](offline-rl-distribution-shift.html)。
