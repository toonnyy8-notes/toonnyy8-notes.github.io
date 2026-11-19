---
slug: dt-offline-rl-baselines-datasets
title: Decision Transformer 比較的離線 RL 基準方法與資料集
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [offline-rl, CQL, BEAR, BRAC, REM, D4RL, Atari, OpenAI-Gym]
arxiv: ["2106.01345", "2005.01643", "1812.02900", "1906.00949", "1911.11361", "1907.04543", "2006.04779", "2004.07219", "1910.00177", "1606.01540", "1312.5602", "2010.02193"]
sources: ["2106.01345 §1, §4, §6.1", "arXiv abstract 2005.01643", "arXiv abstract 1812.02900", "arXiv abstract 1906.00949", "arXiv abstract 1911.11361", "arXiv abstract 1907.04543", "arXiv abstract 2006.04779", "arXiv abstract 2004.07219"]
related: ["dt-lineage-overview", "dt-method-and-results", "rl-mdp-q-learning-basics", "offline-rl-distribution-shift"]
---
# Decision Transformer 比較的離線 RL 基準方法與資料集

MDP、Q-learning 與離線 RL 的分布位移問題見 [MDP 與 Q-learning 基礎](rl-mdp-q-learning-basics.html) 與 [離線 RL 的分布位移](offline-rl-distribution-shift.html)。本篇只列 Decision Transformer（arXiv 2106.01345）實際引用的方法與資料。

## 離線 RL 方法

論文 §6.1 將既有方法分為三類：限制策略的動作空間、加入價值悲觀（pessimism）、在學得的動力學模型中加入悲觀。

| 方法 | arXiv | 作者（年） | abstract 的重點 |
| --- | --- | --- | --- |
| 離線 RL 教學綜述 | 2005.01643 | Levine 等（2020） | 論文 §1 引用為「誤差傳播與價值高估」使離線 RL 困難的來源 |
| BCQ | 1812.02900 | Fujimoto、Meger、Precup（2018） | 外推誤差使標準 off-policy 方法在固定資料上失敗；提出限制動作空間的 batch-constrained RL |
| BEAR | 1906.00949 | Kumar 等（2019） | 指出 bootstrapping error 是不穩定來源，限制動作選擇可降低它 |
| BRAC | 1911.11361 | Wu、Tucker、Nachum（2019） | 行為正則化的 actor-critic 框架；發現許多近期方法的複雜設計並非必要 |
| REM | 1907.04543 | Agarwal 等（2020） | 以 DQN replay dataset 做離線 RL；提出 Random Ensemble Mixture |
| CQL | 2006.04779 | Kumar 等（2020） | 學保守的 Q 函數，使策略價值的估計為真實價值的下界 |

DT 論文以 CQL 作為主要比較對象，另比較 BEAR、BRAC、REM、QR-DQN，以及 D4RL 上的 AWR（1910.00177，Peng 等，2019，advantage-weighted regression）。

## 資料集與環境

- **D4RL**（2004.07219，Fu 等，2020）：專為離線設定設計的基準；DT 用其中的 Gym 連續控制資料（medium、medium-replay、medium-expert 等）。
- **OpenAI Gym**（1606.01540，Brockman 等，2016）：連續控制環境。
- **Atari**：論文使用 DQN replay dataset 的 1%（50 百萬筆轉移中的 50 萬筆），遵循 REM 論文的設定；分數依 Hafner 等（2010.02193）的做法以專業玩家為 100、隨機策略為 0 正規化。DQN 見 1312.5602（Mnih 等，2013）。

## 與 Decision Transformer 的關係

論文 §5.7 的推測：TD 方法會優化學得的價值函數，因而放大其誤差；DT 不以學得的函數為優化目標，所以不需要悲觀或行為正則化。這是論文的推測，不是證明。結果見 [Decision Transformer 的方法與實驗結果](dt-method-and-results.html)；整體脈絡見 [研究脈絡總覽](dt-lineage-overview.html)。
