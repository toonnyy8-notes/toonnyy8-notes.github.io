---
slug: cql-experiments-d4rl-atari
title: CQL 的實驗：D4RL 連續控制、AntMaze、Adroit、Kitchen 與離線 Atari
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [CQL, D4RL, Atari, benchmark, ablation]
arxiv: ["2006.04779", "2004.07219", "1906.00949", "1911.11361", "1710.10044"]
sources: ["arXiv:2006.04779 §6, Tables 1–3", "arXiv abstract 2004.07219", "arXiv abstract 1710.10044"]
related: ["cql-vs-policy-constraint-methods", "cql-algorithm-variants-and-guarantees"]
---
# CQL 的實驗

CQL（arXiv:2006.04779）§6 涵蓋連續與離散動作、不同維度的狀態，以及影像輸入。連續控制使用 D4RL（Fu 等，arXiv:2004.07219）。基線為 BEAR（arXiv:1906.00949）、BRAC（arXiv:1911.11361）、離線化的 SAC，以及 behavioral cloning（BC）。CQL 的實作與 CQL(H)、CQL(ρ) 的差別見 [CQL 的演算法變體](cql-algorithm-variants-and-guarantees.html)，基線的設計見 [CQL 與策略限制方法](cql-vs-policy-constraint-methods.html)。

**Gym（Table 1，4 個種子，normalized return，BEAR、BRAC、SAC、BC 的數字取自 D4RL 論文）。** 單一策略資料（random、medium、expert），CQL 約略持平或小幅超過最佳基線。混合多個策略的資料（mixed、medium-expert、random-expert），CQL 大幅領先，論文稱最高 2 至 3 倍。例：walker2d-medium-expert，CQL(H) 98.7，BEAR 10.8；hopper-random-expert，CQL(H) 110.5，BEAR 10.1。

**Adroit、AntMaze、Kitchen（Table 2）。** Adroit 為 24 自由度機械手、資料量少。論文稱 CQL 各變體是唯一勝過 BC 的方法，分數為次佳 offline 方法的 2 至 9 倍。動作維度較高時 CQL(ρ) 在多項任務上優於 CQL(H)，論文歸因於 CQL(H) 的重要性權重變異較大。AntMaze 較難的 medium、large 迷宮中只有 CQL(H) 取得非零回報（例：antmaze-medium-play 61.2，其他方法皆 0.0）。Kitchen 各任務 CQL 超過 BC，成功率超過 40%。

**離線 Atari（Figure 1、Table 3）。** 離散動作的 Q-learning 版本在 Pong、Breakout、Qbert、Seaquest、Asterix 上與 REM、QR-DQN（arXiv:1710.10044）比較，資料集來自 Agarwal 等人釋出的 DQN replay：(1) 線上 DQN 前 20% 的樣本；(2) 全部樣本的 1% 與 10%。設定 (1) 中 CQL 表現相近或較佳；設定 (2) 的 1% 條件，Q*bert 為 14012.0，Breakout 為 61.1，論文稱分別為最佳先前方法的 36 倍與 6 倍。

**驗證下界。** 論文估計 `E_{s~D}[V̂^k(s)]` 與實際折扣報酬的差，驗證 CQL 低估價值；BEAR 的對應差值一般為正（過高估計）。另有以雙 Q 取最小的 ensemble（大小 2、4、10、20）與 BEAR 的比較（Appendix）。
