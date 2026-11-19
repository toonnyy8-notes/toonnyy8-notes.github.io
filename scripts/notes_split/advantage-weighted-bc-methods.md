---
slug: advantage-weighted-bc-methods
title: 優勢加權的行為複製：AWR、CRR、AWAC、BAIL
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [offline-rl, advantage-weighted, behavior-cloning, awr, crr, awac]
arxiv: ["1910.00177", "2006.15134", "2006.09359", "1910.12179", "2106.06860"]
sources: [1910.00177 abstract, 2006.15134 abstract, 2006.09359 abstract, 1910.12179 abstract, 2106.06860 section 2]
related: ["td3bc-algorithm-and-d4rl", "offline-rl-distribution-shift"]
---
# 優勢加權的行為複製

本論文相關工作寫道：另一類方法「使用加權版本的 BC，偏好高優勢（advantage）的動作」（Wang 等人 2018、Peng 等人 2019、Siegel 等人 2020、Wang 等人 2020、Nair 等人 2020），或「只對資料集的一個明確子集做 BC」（Chen 等人 2020）。

## 代表工作

- **AWR**（Advantage-Weighted Regression，arXiv 1910.00177，2019）：摘要描述為兩個標準監督式學習步驟，一是回歸價值函數目標，二是以加權的目標動作回歸策略。
- **CRR**（Critic Regularized Regression，arXiv 2006.15134，2020）：摘要描述為離線 RL 演算法，以 critic 正則化的回歸學習策略。
- **AWAC**（arXiv 2006.09359，2020）：摘要的重點是利用先前資料加速線上學習。本論文表 2 以其報告數字為基準，並說明最相似的前例之一是 Nair 等人（2020）的 SAC+BC 基準。
- **BAIL**（Best-Action Imitation Learning，arXiv 1910.12179，2019）：摘要描述為學習 V 函數，選出被認為高表現的動作，再以模仿學習訓練策略。對應本論文所說「對資料子集做 BC」。

## 與 TD3+BC 的差別

這類方法的 BC 損失有權重，權重由優勢或價值估計決定；TD3+BC 的 BC 項無權重，價值資訊以另一項 λQ 進入同一個目標，並以 Q 的平均絕對值正規化 λ。兩者都不需要明確建模行為策略的機率密度。細節見 [TD3+BC 演算法與 D4RL 評估](td3bc-algorithm-and-d4rl.html)。

前置概念：[離線強化學習與分布偏移](offline-rl-distribution-shift.html)。
