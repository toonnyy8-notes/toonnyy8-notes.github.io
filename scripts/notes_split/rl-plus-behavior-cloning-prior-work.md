---
slug: rl-plus-behavior-cloning-prior-work
title: RL 與行為複製的結合：從示範學習到 SAC+BC 基準
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [behavior-cloning, imitation-learning, learning-from-demonstrations, ddpgfd]
arxiv: ["1509.02971", "1709.10089", "1709.10087", "1704.03732", "1707.08817", "2006.09359", "2106.06860"]
sources: [2106.06860 section 2, 1709.10089 abstract, 1704.03732 abstract, 1707.08817 abstract, 1709.10087 abstract]
related: ["ddpg-td3-actor-critic", "offline-rl-distribution-shift"]
---
# RL 與行為複製的結合

本論文第 2 節先聲明：就作者所知，將 TD3 與 BC 用於離線 RL 是首例，但 RL 加 BC 或其他模仿學習已被許多作者探討。

## RL + BC

目的是加速「從示範學習」（Atkeson 與 Schaal，1997）。BC 被用作策略最佳化的正則化，搭配 DDPG（Lillicrap 等人 2015、Nair 等人 2018、Goecks 等人 2020）或 natural policy gradient（Kakade 2001、Rajeswaran 等人 2017），但需要修改 replay buffer 與預訓練等額外設計。

- Nair 等人（Overcoming Exploration in Reinforcement Learning with Demonstrations，arXiv 1709.10089，2017）：摘要指出以示範克服稀疏獎勵的探索問題，任務為機械手臂堆疊積木等長時序連續控制。
- Rajeswaran 等人（arXiv 1709.10087，2017）：以深度 RL 與示範學習靈巧操作。

最相近的是 Nair 等人（2020）的 SAC+BC 基準，以及一份未發表的課程專案（PPO+BC）。

## RL + Imitation

本論文列舉的其他結合方式：對抗式方法、預訓練、修改 replay buffer（Večerík 等人，arXiv 1707.08817，2017）、調整價值函數（Hester 等人，Deep Q-learning from Demonstrations，arXiv 1704.03732，2017）、獎勵塑形。本論文評論：這些方法都把示範用於克服探索困難或提升學習速度。

## 與本論文的差異

前例的資料是示範，用途是線上學習的輔助；本論文的資料集是固定的，且完全不與環境互動，BC 項的作用是限制 actor 遠離資料集動作，以降低外插誤差。底層演算法見 [DDPG 到 TD3](ddpg-td3-actor-critic.html)，離線設定見 [離線強化學習與分布偏移](offline-rl-distribution-shift.html)。
