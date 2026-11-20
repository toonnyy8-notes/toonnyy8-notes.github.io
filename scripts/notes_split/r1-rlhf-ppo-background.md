---
slug: r1-rlhf-ppo-background
title: LLM 的 RL 前史：RLHF、PPO 與 GAE
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [RLHF, PPO, GAE, DPO, reward-overoptimization]
arxiv: ["2203.02155", "2204.05862", "1707.06347", "1506.02438", "2305.18290", "2210.10760", "2501.12948"]
sources: ["arXiv 摘要 2203.02155、1707.06347、1506.02438、2210.10760", "2501.12948 §2.1、附錄 A.3、H.3"]
related: ["r1-lineage-hub", "r1-grpo-outcome-rewards", "r1-process-supervision-prm", "rl-mdp-q-learning-basics", "rlvr-grpo-credit-assignment"]
---
# LLM 的 RL 前史：RLHF、PPO 與 GAE

**RLHF。** 本文 H.3：RL 在對齊 LLM 與人類偏好上扮演關鍵角色（Bai 等 2022，arXiv 2204.05862；Ouyang 等 2022，arXiv 2203.02155），但少有研究把 RL 用於提升推理。傳統流程依序是：以高品質人類示範做 SFT 作為初始化、以人類偏好訓練獎勵模型、再以 PPO 或 DPO（Rafailov 等 2023，arXiv 2305.18290）最佳化語言模型。Ouyang 等的摘要：以人類回饋微調語言模型，使其符合使用者意圖。

**PPO。** Schulman 等（2017，arXiv 1707.06347）摘要：一族 policy gradient 方法，交替進行取樣與以 surrogate 目標做多個 epoch 的 minibatch 隨機梯度上升。本文 §2.1 稱 PPO 廣泛用於 LLM 的 RL 階段。

**GAE 與價值模型。** PPO 的優勢值通常以 GAE（Schulman 等 2015，arXiv 1506.02438）計算，需要一個學習出來的價值模型。本文附錄 A.3 指出價值模型通常與策略模型同尺寸，造成大量記憶體與運算負擔；在只有最終結果獎勵、輸出很長時，由部分回應預測最終獎勵更困難。GAE 的價值函數與優勢值概念見 [MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)。

**獎勵過度最佳化。** Gao 等（2022，arXiv 2210.10760）摘要：獎勵模型只是代理，過度最佳化會損害真實表現。本文 G.2 以此說明引入模型式獎勵會導致 reward hacking，見 [過程監督與 PRM](r1-process-supervision-prm.html)。

**接續。** 去掉價值模型的做法見 [GRPO 與結果導向獎勵](r1-grpo-outcome-rewards.html)；GRPO 與 PPO 在後續論文中的比較見 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)。總覽：[總覽](r1-lineage-hub.html)。
