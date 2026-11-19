---
slug: diffusion-ql-lineage-hub
title: Diffusion-QL 的研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [diffusion-ql, diffusion-policy, offline-rl, lineage]
arxiv: ["2208.06193", "1503.03585", "2006.11239", "1907.05600", "2011.13456", "1812.02900", "2106.06860", "2205.09991"]
sources: ["paper.txt 全文與參考文獻", "arXiv API 驗證各引用的題名、作者、年份與 ID"]
related: ["rl-mdp-q-learning-basics", "offline-rl-distribution-shift", "batch-constrained-policy-constraint-methods", "ddpg-td3-actor-critic", "ddpm-training-and-sampling-recap", "gaussian-policy-multimodality-limits", "diffusion-policy-class", "diffusion-ql-loss-coupling", "diffusion-in-rl-concurrent-work", "diffusion-policy-experiments-d4rl", "diffusion-policy-inference-cost"]
---
# Diffusion-QL 的研究脈絡

*Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning*（arXiv 2208.06193）把策略表示為條件擴散模型，並把 Q 值最大化項加進擴散損失。建議的閱讀順序如下。

1. 離線 RL 的問題：資料集固定，OOD 動作的 Q 值被高估。見 [MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html) 與 [Offline RL 的分布偏移](offline-rl-distribution-shift.html)。
2. 策略正則化的既有方法：BCQ、BEAR、BRAC、TD3+BC、IQL，以及它們的策略類別限制。見 [批次約束與策略約束方法](batch-constrained-policy-constraint-methods.html)、[DDPG 到 TD3](ddpg-td3-actor-critic.html)、[多峰限制與 bandit 示例](gaussian-policy-multimodality-limits.html)。
3. 擴散模型基礎：Sohl-Dickstein et al.（2015）、Song & Ermon（2019）、Ho et al.（2020）、Song et al.（2021）。見 [DDPM 的訓練與取樣流程](ddpm-training-and-sampling-recap.html)。
4. 擴散模型作為策略：[擴散模型作為策略類別](diffusion-policy-class.html)。
5. 加入 Q 學習：[Diffusion-QL 的損失耦合](diffusion-ql-loss-coupling.html)。
6. 與軌跡層級方法的對照：[擴散模型在 RL 中的同期與前驅工作](diffusion-in-rl-concurrent-work.html)。
7. 結果與限制：[D4RL 實驗與消融](diffusion-policy-experiments-d4rl.html)、[推論成本](diffusion-policy-inference-cost.html)。

時間線見同資料夾的 timeline.json。
