---
slug: batch-constrained-policy-constraint-methods
title: 批次約束與策略約束方法：BCQ、BEAR、BRAC 與其後續
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [offline-rl, policy-constraint, bcq, bear, brac, fisher-brc]
arxiv: ["1812.02900", "1906.00949", "1911.11361", "1907.00456", "2103.08050", "2007.11091", "2106.06860"]
sources: [1812.02900 abstract, 1911.11361 abstract, 1906.00949 abstract, 2103.08050 abstract, 2007.11091 abstract, 2106.06860 section 1-2]
related: ["offline-rl-distribution-shift", "td3bc-algorithm-and-d4rl", "advantage-weighted-bc-methods"]
---
# 批次約束與策略約束方法

前置：[離線強化學習與分布偏移](offline-rl-distribution-shift.html)。

## 共同想法

本論文第 1 節把這一類解法統稱為「學到的策略應與資料產生者（行為策略）保持接近」，依「接近」的實作方式有不同名稱：batch-constrained（Fujimoto 等人，2019b）、KL-control（Jaques 等人，2019）、behavior-regularized（Wu 等人，2019）、policy constraint（Levine 等人，2020）。

## 代表工作

- **BCQ**（Off-Policy Deep Reinforcement Learning without Exploration，arXiv 1812.02900，2018）：摘要指出 DQN 與 DDPG 這類標準 off-policy 演算法會因外插誤差，無法從與當前策略分布不同的資料學習；提出 batch-constrained RL，限制動作空間使行為接近 on-policy。本論文據此以「extrapolation error」描述離線 RL 的核心困難。
- **BEAR**（Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction，arXiv 1906.00949，2019）：摘要指出 bootstrapping error 是現有方法不穩定的主因。本論文把它歸入 divergence regularization。
- **KL-control**（Jaques 等人，arXiv 1907.00456，2019）：對話情境中的 batch RL，本論文歸入 divergence regularization。
- **BRAC**（Behavior Regularized Offline Reinforcement Learning，arXiv 1911.11361，2019）：摘要描述其為評估近期方法與簡單基準的通用框架，並指出許多複雜設計在實驗中並非必要。本論文表 2 以其報告數字作為基準之一。
- **Fisher-BRC**（Kostrikov 等人，arXiv 2103.08050，2021）：將 critic 參數化為 log 行為策略加上一個偏移項，並對偏移項做梯度懲罰。為本論文的主要比較對象；論文指出它在 SAC 上加了固定的獎勵加成並調整網路架構。
- **EMaQ**（Ghasemipour 等人，arXiv 2007.11091，2020）：摘要稱其為 BCQ 的簡化。本論文歸入「估計行為策略後定義顯式策略參數化」。

## 與 TD3+BC 的關係

TD3+BC 屬於這個類別中最簡單的形式：不需要生成模型，也不估計 KL 或其他散度，只在 actor 目標加入 MSE 形式的 BC 項。本論文表示，沒有根本理由說明為何最小化某一種散度或距離會優於另一種。方法細節見 [TD3+BC 演算法與 D4RL 評估](td3bc-algorithm-and-d4rl.html)。

另見 [優勢加權的行為複製](advantage-weighted-bc-methods.html)，那一類以加權 BC 達成類似的靠近資料效果。
