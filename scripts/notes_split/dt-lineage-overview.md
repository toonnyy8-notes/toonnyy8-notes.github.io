---
slug: dt-lineage-overview
title: Decision Transformer 的研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [decision-transformer, offline-rl, sequence-modeling, lineage]
arxiv: ["2106.01345", "1706.03762", "2005.14165", "1912.02877", "1912.13465", "2006.04779", "2106.02039"]
sources: ["2106.01345 §1, §6", "arXiv abstract 1706.03762", "arXiv abstract 2005.14165", "arXiv abstract 1912.02877", "arXiv abstract 1912.13465", "arXiv abstract 2006.04779", "arXiv abstract 2106.02039"]
related: [rl-mdp-q-learning-basics, offline-rl-distribution-shift, dt-transformer-lm-background, return-conditioned-sequence-modeling, dt-offline-rl-baselines-datasets, dt-credit-assignment-lineage, sequence-modeling-offline-rl, dt-method-and-results]
---
# Decision Transformer 的研究脈絡總覽

Decision Transformer（arXiv 2106.01345）把強化學習改寫成條件式序列建模：以 GPT 架構在離線軌跡上預測動作，輸入包含 returns-to-go、狀態與動作。論文的動機有三條來源：Transformer 與語言模型的成功、離線 RL 中 TD 學習的困難，以及把 RL 轉成監督式學習的既有嘗試。

建議閱讀順序如下。

1. 先讀 MDP、Q-learning 與離線 RL 的共用背景：[MDP 與 Q-learning 基礎](rl-mdp-q-learning-basics.html)、[離線 RL 的分布位移](offline-rl-distribution-shift.html)。
2. [Transformer 與語言模型背景](dt-transformer-lm-background.html)：Attention、GPT 與其他被論文引為動機的序列模型。
3. [離線 RL 基準方法與資料集](dt-offline-rl-baselines-datasets.html)：論文比較的 TD 方法（BCQ、BEAR、BRAC、REM、CQL）與 D4RL、Atari 資料。
4. [以報酬為條件的監督式學習](return-conditioned-sequence-modeling.html)：上下顛倒 RL、Reward-Conditioned Policies 與目標條件模仿，是 Decision Transformer 最直接的前身。
5. [信用分配的相關研究](dt-credit-assignment-lineage.html)：RUDDER、HCA 等方法，以及論文 Key-to-Door 實驗的來源。
6. [RL 中的 Transformer 與 Trajectory Transformer](sequence-modeling-offline-rl.html)：把 Transformer 當成元件的先前工作，與同期提出的 Trajectory Transformer。
7. [Decision Transformer 的方法與實驗結果](dt-method-and-results.html)：returns-to-go 表示法、訓練與推論流程、主要結果。

時間軸見 timeline.json。
