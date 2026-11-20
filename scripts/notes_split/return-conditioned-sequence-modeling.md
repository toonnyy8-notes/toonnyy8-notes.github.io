---
slug: return-conditioned-sequence-modeling
title: 以報酬為條件的監督式學習與序列建模
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [upside-down-rl, reward-conditioned-policies, behavior-cloning, hindsight, goal-conditioned]
arxiv: ["2106.01345", "1912.02877", "1912.13465", "1912.06088", "2012.02419", "2106.02039"]
sources: ["2106.01345 §6.2, §5.1", "arXiv abstract 1912.02877", "arXiv abstract 1912.13465", "arXiv abstract 1912.06088", "arXiv abstract 2012.02419"]
related: [dt-lineage-overview, dt-method-and-results, sequence-modeling-offline-rl, rl-mdp-q-learning-basics]
---
# 以報酬為條件的監督式學習與序列建模

核心想法：把軌跡的實際報酬（或目標）當成條件，用監督式損失學習「在這個條件下會採取的動作」，不使用 Bellman 備份。Decision Transformer（arXiv 2106.01345，§6.2）把這類方法稱為與自己最接近的前身。

## 前身工作

- **Upside-Down RL（UDRL）**，Srivastava 等，2019，arXiv 1912.02877。abstract：只用監督式學習讓代理學習行動，不做報酬預測也不搜尋最佳策略，而是訓練代理遵循「在多少時間內取得多少總報酬」這類指令。
- **Reward-Conditioned Policies（RCP）**，Kumar、Peng、Levine，2019，arXiv 1912.13465。abstract：次佳策略收集的非專家軌跡，可視為「匹配該軌跡報酬」的最佳監督；以報酬數值為條件即可得到能泛化到更高報酬的策略。DT 論文指出 RCP 與 K = 1 的 Decision Transformer 最相似，且 §5.3 的結果顯示較長的序列脈絡表現較佳。
- **目標條件的迭代監督式學習**，Ghosh 等，2019，arXiv 1912.06088。abstract：不用示範或價值函數，將每條軌跡視為到達其最終狀態的成功示範，反覆重標記並模仿。DT 論文說明這類方法以狀態目標取代報酬作為條件。
- **Planning from Pixels using Inverse Dynamics Models**，Paster 等，2020，arXiv 2012.02419。abstract：學習以任務完成為條件、預測未來動作序列的模型。DT 論文描述它使用 LSTM 與狀態目標條件，用於目標條件的線上 RL。

## Decision Transformer 的區別

論文 §6.2 的說法：差別主要在動機轉向序列建模而非單純監督式學習；實作上主要是脈絡長度與架構不同。序列建模即使沒有報酬也能建模行為，並且與語言、影像的做法一致。

與行為複製的關係見 §5.1：論文提出 Percentile BC（%BC），只複製依 episode 報酬排序的前 X% 時間步。資料充足的 D4RL 上 %BC 可與其他離線 RL 方法持平或更好；Atari 的 1% 資料量下 %BC 較弱，DT 較佳。

與同期 Trajectory Transformer 的比較見 [RL 中的 Transformer 與 Trajectory Transformer](sequence-modeling-offline-rl.html)。方法細節見 [Decision Transformer 的方法與實驗結果](dt-method-and-results.html)。
