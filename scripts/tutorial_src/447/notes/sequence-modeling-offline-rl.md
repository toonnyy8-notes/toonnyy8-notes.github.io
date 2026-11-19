---
slug: sequence-modeling-offline-rl
title: 以序列模型處理離線 RL：Decision Transformer 與 Trajectory Transformer
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [offline-rl, sequence-modeling, transformer, decision-transformer, trajectory-transformer]
arxiv: ["2106.01345", "2106.02039", "2205.09991"]
sources: ["paper.txt §5.3, §6, Appendix A.3", "arXiv API metadata (2106.01345, 2106.02039)"]
related: ["offline-rl-distribution-shift", "model-based-planning-abstraction-barrier", "trajectory-diffusion-planning"]
---
# 以序列模型處理離線 RL

## 兩篇前作

- Chen, Lu, Rajeswaran, Lee et al. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling*（2106.01345）。本論文（§5.3）把它歸為 return-conditioning 方法，簡稱 DT。
- Janner, Li, Levine (2021). *Offline Reinforcement Learning as One Big Sequence Modeling Problem*（2106.02039）。本論文歸為 model-based 方法，簡稱 TT，並在 §6 把 Transformer 列為被用於動態模型的一類生成模型。

Diffuser 的第一作者同時是 TT 的第一作者；論文把軌跡視為整體建模的出發點與 TT 相同，差別在 §1 所說：Diffuser 同時預測所有時間步，不依時間順序自迴歸。

## 在本論文中的角色

兩者都是 Table 2（D4RL locomotion）的比較對象。論文引用的分數來源：DT 取自 Chen et al. (2021b) Table 2，TT 取自 Janner et al. (2021) Table 1（Appendix A.3）。論文表中的九項平均：DT 74.7、TT 78.9、Diffuser 77.5。

## 與其他筆記的關係

離線設定下的分佈偏移問題見 [離線 RL 的分佈偏移](offline-rl-distribution-shift.html)。論文 §6 另提到 Lambert et al. (2020) 研究過非自迴歸的軌跡層級動態模型；該文 arXiv 編號（2012.09156）未能在本次以 API 逐一核對標題，因此不列入時間軸。Diffuser 的設計見 [軌跡層級擴散規劃（Diffuser）](trajectory-diffusion-planning.html)；規劃器與模型的分工背景見 [學習式動態模型與軌跡最佳化的分工](model-based-planning-abstraction-barrier.html)。
