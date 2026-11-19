---
slug: ddpg-td3-actor-critic
title: DDPG 到 TD3：連續動作的 actor-critic
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [actor-critic, ddpg, td3, deterministic-policy-gradient, overestimation]
arxiv: ["1509.02971", "1802.09477", "1801.01290", "2106.06860"]
sources: [1509.02971 abstract, 1802.09477 abstract, 2106.06860 section 1 and 5, 1801.01290 abstract]
related: ["rl-mdp-q-learning-basics", "offline-rl-distribution-shift", "td3bc-algorithm-and-d4rl"]
---
# DDPG 到 TD3：連續動作的 actor-critic

本筆記假設讀者已讀過 [強化學習與 MDP、Q-learning 基礎](rl-mdp-q-learning-basics.html)。

## Actor-critic 的分工

Actor-critic 同時學兩個函數：critic 估計 Q(s, a)，actor 輸出動作的策略 π。連續動作空間無法對所有動作取 max，所以 actor 負責提供使 Q 較大的動作。

## 確定性策略梯度與 DDPG

Silver 等人（2014）提出確定性策略梯度：策略 π(s) 直接輸出動作，更新目標為最大化 Q(s, π(s))。本論文的式 (1) 即是此目標在資料集 D 上的形式：π = argmax E[Q(s, π(s))]。

Lillicrap 等人（Continuous control with deep reinforcement learning，arXiv 1509.02971，2015）的 DDPG 摘要描述為「以確定性策略梯度為基礎、可在連續動作空間運作的 actor-critic、model-free 演算法」，把 Deep Q-Learning 的想法移到連續動作領域。

## TD3 的三項修改

Fujimoto、van Hoof 與 Meger（Addressing Function Approximation Error in Actor-Critic Methods，arXiv 1802.09477，2018）的摘要指出：函數近似誤差會造成價值高估，這個問題在 actor-critic 中同樣存在。TD3 的機制如摘要所述：

- 以兩個 critic 取最小值，限制高估（建立在 Double Q-learning 上）。
- 延遲策略更新，以降低每次更新的誤差。
- 摘要也說明了 target network 與高估偏差的關聯。

TD3 是 TD3+BC 的底層演算法。論文附錄說明其 TD3 實作取自作者 GitHub 的 TD3 原始碼。

## 與 SAC 的關係

Soft Actor-Critic（Haarnoja 等人，arXiv 1801.01290，2018）是以最大熵框架、隨機策略為基礎的 off-policy actor-critic。本論文指出離線 RL 演算法多半建立在 TD3 或 SAC 之上（例如 CQL 與 Fisher-BRC 修改的是 SAC），而 TD3+BC 選擇 TD3，其策略是確定性的。

## 在離線設定的問題

TD3 本身不處理資料分布偏移。[離線強化學習與分布偏移](offline-rl-distribution-shift.html) 說明 critic 在資料集外的動作上會高估，actor 的 argmax 會偏向這些動作。TD3+BC 對 actor 目標的修改見 [TD3+BC 演算法與 D4RL 評估](td3bc-algorithm-and-d4rl.html)。
