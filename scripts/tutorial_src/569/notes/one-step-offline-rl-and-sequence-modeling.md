---
slug: one-step-offline-rl-and-sequence-modeling
title: 單步 offline RL 與序列建模：One-step RL、Decision Transformer
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [offline-rl, one-step-rl, decision-transformer, stitching]
arxiv: ["2110.06169", "2106.08909", "2106.01345", "2109.00137", "2004.07219"]
sources: ["arXiv:2110.06169 §2, §4, §5.1, §5.2", "各論文 arXiv 摘要"]
related: ["iql-lineage-overview", "expectile-regression-value-learning", "iql-algorithm-and-d4rl-evaluation", "offline-rl-distribution-shift"]
---
# 單步 offline RL 與序列建模

IQL 論文 §2 把一類方法稱為「single-step」：只做一次 policy iteration 的改進，或完全不用價值函數。這些方法不需要查詢未見動作，因為它們不用價值函數，或只學行為策略的價值。

## 代表方法

- One-step RL（Brandfonbrener、Whitney、Ranganath、Bruna，*Offline RL Without Off-Policy Evaluation*，arXiv:2106.08909，2021）：擬合行為策略的 value 或 Q 函數，再取出對應的 greedy 策略。
- Decision Transformer（Chen 等，arXiv:2106.01345，2021）：以序列建模與 behavioral cloning 式目標取代價值函數。
- Implicit Behavioral Cloning（Florence 等，arXiv:2109.00137，2021）：IQL 論文用它支持「Adroit 與 Kitchen 上多數 RL 方法表現與模仿學習相近」的說法。

## 論文的對照

IQL 論文 §4 用 SARSA 式損失 `r + γ Q(s', a') − Q(s, a)`（式 2）說明單步方法：該損失只用資料集動作，學到的是行為策略的價值。論文 §5.1 的 U 形迷宮實驗（資料集為 1 條最佳軌跡與 99 條隨機軌跡，γ = 0.9）顯示，one-step 方法的價值隨離目標變遠而衰減過快，策略因此偏離最佳。

論文 §5.2 指出，locomotion 資料集含相當比例的近最佳軌跡，單步方法表現良好；Ant Maze 的 medium 與 large 資料集幾乎沒有近最佳軌跡，需要「stitching」不同的次佳軌跡（stitching 的資料集設計來自 D4RL，Fu 等，arXiv:2004.07219，2020）。Table 1 中 One-step RL 在 antmaze-v0 的總分為 125.3，IQL 為 378.0。

IQL 保留 SARSA 式損失只用資料集動作的優點，並以 [expectile regression](expectile-regression-value-learning.html) 使它能做多步動態規劃。

相關：[總覽](iql-lineage-overview.html)、[IQL 演算法與 D4RL 評估](iql-algorithm-and-d4rl-evaluation.html)、[offline RL 的 distribution shift](offline-rl-distribution-shift.html)。
