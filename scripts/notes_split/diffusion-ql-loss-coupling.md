---
slug: diffusion-ql-loss-coupling
title: Diffusion-QL：行為複製項與 Q 值項的損失耦合
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [diffusion-ql, q-learning, actor-critic, td3, policy-improvement, offline-rl]
arxiv: ["2208.06193", "2106.06860", "1509.02971", "1812.02900", "2006.04779"]
sources: ["paper.txt §3.2, Algorithm 1, §5.2", "arXiv API 驗證 2106.06860、1509.02971、1812.02900、2006.04779 的題名與作者；演算法細節取自 paper.txt"]
related: ["diffusion-policy-class", "rl-mdp-q-learning-basics", "ddpg-td3-actor-critic", "td3bc-algorithm-and-d4rl", "cql-lineage-hub", "gaussian-policy-multimodality-limits", "diffusion-policy-experiments-d4rl"]
---
# Diffusion-QL 的損失耦合

只用 `L_d`（見 [擴散模型作為策略類別](diffusion-policy-class.html)）只會複製行為策略，無法超越產生資料的策略。論文（arXiv 2208.06193）把 Q 值引導放進同一個訓練損失：

`π = argmin_θ L(θ) = L_d(θ) + L_q(θ) = L_d(θ) − α · E_{s~D, a⁰~π_θ}[Q_φ(s, a⁰)]`

## 設計要點

- `a⁰` 由反向鏈式 (1) 以重參數化取得，所以 Q 對動作的梯度會反向傳遞穿過整條擴散鏈。這是「訓練期」引導，不同於 Diffuser 在取樣階段加入回報引導。
- `α = η / E_{(s,a)~D}[|Q_φ(s,a)|]`，沿用 TD3+BC（Fujimoto & Gu, 2021，arXiv 2106.06860）的正規化；分母只用於縮放，不求導。`η` 平衡兩個損失項。見 [TD3+BC 演算法與 D4RL 評估](td3bc-algorithm-and-d4rl.html)。
- Critic 以一般方式學習：兩個 Q 網路 `Q_φ1, Q_φ2` 加目標網路，目標為 `r + γ min_i Q_φ'i(s', a'⁰)`，其中 `a'⁰ ~ π_θ'`（式 4）。論文引用 Lillicrap et al.（2015，arXiv 1509.02971）與 Fujimoto et al.（2019，arXiv 1812.02900）作為 Bellman 目標的來源，雙 Q 技巧引自 Hasselt（2010，本筆記未驗證其 arXiv ID）。見 [DDPG 到 TD3：連續動作的 actor-critic](ddpg-td3-actor-critic.html) 與 [MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)。
- Algorithm 1 可選加入 Kumar et al.（2020，arXiv 2006.04779）的 max Q backup。CQL 的背景見 [Conservative Q-Learning 的研究脈絡總覽](cql-lineage-hub.html)。
- 目標網路以 `θ' = ρθ' + (1−ρ)θ` 更新，策略、兩個 critic 皆同。

## 為什麼稱為耦合

論文主張 `L_d` 是軟性的、基於樣本的策略正則化，`L_q` 在正則化範圍內往高值動作移動；兩者同時作用於同一個擴散鏈。論文的 bandit 與消融結果（見 [多峰限制](gaussian-policy-multimodality-limits.html)、[D4RL 實驗與消融](diffusion-policy-experiments-d4rl.html)）顯示：擴散步數 `N` 較大時正則化較強；加入 Q 引導後，較小的 `N` 也能得到接近最佳的策略。
