---
slug: rl-implementation-details-reproducibility
title: 深度 RL 的實作細節與可重現性：本論文的「極簡」動機
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [reproducibility, implementation-details, meta-analysis, hyperparameters]
arxiv: ["1709.06560", "2005.12729", "2006.05990", "2103.17258", "2007.09055", "2106.06860"]
sources: [2106.06860 section 1, 2, 4, 1709.06560 abstract, 2005.12729 abstract, 2006.05990 abstract, 2103.17258 abstract, 2007.09055 title]
related: ["td3bc-algorithm-and-d4rl"]
---
# 深度 RL 的實作細節與可重現性

本論文第 1 節指出：許多離線 RL 演算法需要大量實作層級調整、未明說的超參數或生成模型等次要元件，使其難以重現與調校，也使效能提升難以歸因。這個觀察延續線上 RL 的 meta analysis。

## 線上 RL 的前例

- Henderson 等人（Deep Reinforcement Learning that Matters，arXiv 1709.06560，2017）：摘要指出環境的非決定性與方法本身的變異使結果難以詮釋，需要顯著性指標與更嚴謹的報告。
- Engstrom 等人（Implementation Matters in Deep Policy Gradients，arXiv 2005.12729，2020）：摘要指出 code-level optimization 是 PPO 相對 TRPO 大部分獎勵增益的來源。
- Andrychowicz 等人（What Matters In On-Policy RL?，arXiv 2006.05990，2020）：實作 50 項以上的選擇並做大規模實驗。
- Furuta 等人（arXiv 2103.17258，2021）：以 MPO、AWR、SAC 為對象，分離演算法創新與實作決策。本論文指出少數細微的實作細節與特定演算法強烈共適應。

## 離線設定的額外困難

本論文指出離線 RL 中評估實作與超參數的改變，與「不與環境互動」的目標相衝突（Paine 等人，Hyperparameter Selection for Offline RL，arXiv 2007.09055，2020；Yang 等人 2020）。

## 論文對 CQL 與 Fisher-BRC 的檢視

論文表 1 檢視兩者在演算法之外的修改：CQL 加入只做模仿學習的 actor 預訓練，並在評估時於取樣動作中取最大者；Fisher-BRC 加入固定獎勵加成；兩者都在 SAC 上移除 target 中的熵項並修改網路架構。結論是：離線時不能互動驗證，額外調整的成本更高，所以演算法應盡量少修改底層演算法。

後續： [TD3+BC 演算法與 D4RL 評估](td3bc-algorithm-and-d4rl.html)。
