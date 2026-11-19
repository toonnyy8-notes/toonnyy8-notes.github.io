---
slug: diffusion-policy-experiments-d4rl
title: Diffusion-QL 的 D4RL 實驗結果與消融
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [d4rl, ablation, cvae, bcq, antmaze, adroit, kitchen]
arxiv: ["2208.06193", "2004.07219", "1812.02900"]
sources: ["paper.txt §5, Table 1, Table 2, Figure 3", "arXiv API 驗證 2004.07219、1812.02900"]
related: ["offline-rl-benchmarks-and-baselines", "diffusion-ql-loss-coupling", "diffusion-policy-class", "gaussian-policy-multimodality-limits", "diffusion-policy-inference-cost"]
---
# D4RL 實驗與消融

評估基準為 D4RL（Fu et al., 2020，arXiv 2004.07219），涵蓋 Gym-MuJoCo、AntMaze、Adroit、Kitchen；基準與比較方法的共通背景見 [本論文使用的離線 RL 基準與比較方法](offline-rl-benchmarks-and-baselines.html)。

## 主要結果（論文 Table 1 的平均）

| 領域 | Diffusion-QL | 對照（同表） |
|---|---|---|
| Gym | 88.0 | CQL 77.6、IQL 77.0、Onestep RL 76.1 |
| AntMaze | 69.6 | IQL 63.0、CQL 50.6 |
| Adroit | 65.1 | IQL 54.4 |
| Kitchen | 69.0 | IQL 53.3 |

結果為 Gym 50 次 rollout（5 個模型各 10 條軌跡）、其他任務 500 次 rollout 的平均。論文指出 Adroit 需要較強的策略正則化（較小 `η`），AntMaze 需要穩定的 Q-learning。

## 消融（Table 2，Gym medium-expert 三任務平均）

| BC-CVAE | BC-Diffusion | BCQ-CVAE | BCQ-Diffusion | CVAE-QL | Diffusion-QL |
|---|---|---|---|---|---|
| 79.9 | 85.9 | 105.1 | 106.0 | 76.6 | 106.9 |

論文的解讀：BC-Diffusion 優於 BC-CVAE，顯示擴散策略較能擬合資料分布；在 BCQ 框架內擴散仍略優；BCQ 的硬性偏差限制會壓低策略改善；CVAE 與自由 Q 引導搭配時表現差（CVAE-QL）。兩個成分（擴散策略與 Q 引導）需一起使用。損失設計見 [Diffusion-QL 的損失耦合](diffusion-ql-loss-coupling.html)，策略類別見 [擴散模型作為策略類別](diffusion-policy-class.html)。

## 擴散步數 N

Figure 3 在 `N∈[2,5,10,20]` 下比較，`N` 增大時收斂較快、表現較穩；全論文取 `N=5`。成本討論見 [擴散策略的推論成本](diffusion-policy-inference-cost.html)。bandit 示例見 [多峰限制](gaussian-policy-multimodality-limits.html)。
