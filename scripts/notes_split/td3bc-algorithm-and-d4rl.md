---
slug: td3bc-algorithm-and-d4rl
title: TD3+BC 演算法與 D4RL 評估
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [td3bc, d4rl, offline-rl, benchmark, state-normalization]
arxiv: ["2106.06860", "1802.09477", "2004.07219", "2006.04779", "2103.08050", "1910.01708"]
sources: [2106.06860 section 1, 5, 6, 2004.07219 abstract, 1910.01708 title]
related: ["ddpg-td3-actor-critic", "offline-rl-distribution-shift", "batch-constrained-policy-constraint-methods", "advantage-weighted-bc-methods"]
---
# TD3+BC 演算法與 D4RL 評估

前置：[DDPG 到 TD3](ddpg-td3-actor-critic.html) 與 [離線強化學習與分布偏移](offline-rl-distribution-shift.html)。

## 演算法（論文式 3–5）

1. 策略更新：TD3 的 π = argmax E[Q(s, π(s))] 改為 π = argmax E_(s,a)~D [λQ(s, π(s)) − (π(s) − a)²]。第二項為 BC 的均方誤差。
2. 狀態特徵標準化：s_i = (s_i − μ_i) / (σ_i + ε)，ε = 10⁻³，統計量取自整個資料集。
3. λ 的設定：λ = α / ((1/N) Σ |Q(s_i, a_i)|)，α = 2.5。分母只用來縮放，不對其微分；實作上以 mini-batch 估計。動作範圍為 [−1, 1] 時，BC 項最大為 4，而 Q 的尺度取決於獎勵，因此需要正規化。

論文表示以上三式涵蓋對 TD3 的全部修改。附錄說明 α 僅在 Hopper-medium-v0 與 Hopper-expert-v0 上以單一 seed 於 (1, 2, 2.5, 3, 4) 中調整。

## 評估

資料集為 D4RL（Fu 等人，arXiv 2004.07219，2020）的 OpenAI gym MuJoCo 任務；摘要說明該基準是為離線設定設計，因為既有基準只含部分訓練代理產生的資料。比較對象為 BC、BRAC-p、AWAC、CQL 與 Fisher-BRC。

表 2（最後 10 次評估、5 個 seed 的平均標準化分數）總分：BC 595.3、CQL 764.3、Fisher-BRC 974.6、TD3+BC 979.3。論文結論是 TD3+BC 與 Fisher-BRC 效能相當，且計算成本不到一半。

論文也提到 Fujimoto 等人（Benchmarking Batch Deep RL Algorithms，arXiv 1910.01708，2019a）作為在離線設定下比較演算法的先前工作之一。

## 相關背景

各比較方法所屬類別見 [批次約束與策略約束方法](batch-constrained-policy-constraint-methods.html) 與 [優勢加權的行為複製](advantage-weighted-bc-methods.html)。CQL 的悲觀價值估計見 [離線強化學習與分布偏移](offline-rl-distribution-shift.html)。
