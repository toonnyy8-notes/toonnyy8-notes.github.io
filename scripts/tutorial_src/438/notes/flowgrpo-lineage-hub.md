---
slug: flowgrpo-lineage-hub
title: Flow-GRPO 的研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [flow-grpo, online-rl, flow-matching, text-to-image, lineage]
arxiv: ["2505.05470", "2209.03003", "2210.02747", "2011.13456", "2303.08797", "2305.13301", "2402.03300", "2305.18290"]
sources: ["paper.txt §1, §2, §3", "arXiv API metadata"]
related: ["flowgrpo-flow-matching-backbone", "flowgrpo-diffusion-sde-background", "flowgrpo-denoising-mdp-and-grpo", "rl-finetuning-text-to-image-diffusion", "flow-ode-to-sde-for-rl", "flowgrpo-design-choices-and-results", "flow-matching-velocity-target", "rlvr-grpo-credit-assignment", "rl-mdp-q-learning-basics"]
---
# Flow-GRPO 的研究脈絡總覽

Flow-GRPO（2505.05470，NeurIPS 2025）把線上策略梯度 RL（GRPO）用在 flow matching 文字生成圖像模型上。論文指出兩個衝突：flow 模型以確定性 ODE 取樣，RL 需要隨機探索；flow 模型每個樣本需要多步迭代，線上 RL 取樣成本高。論文用兩個做法回應：ODE-to-SDE 轉換，以及 Denoising Reduction。

閱讀順序如下。

1. [Flow matching 與 rectified flow 作為生成骨幹](flowgrpo-flow-matching-backbone.html)：論文要訓練的模型是什麼，速度場與確定性取樣從哪來。
2. [從 DDPM 到 score SDE 與 stochastic interpolants](flowgrpo-diffusion-sde-background.html)：同一組邊際分布可由 ODE 或 SDE 取得。
3. [Denoising 作為 MDP，以及 GRPO 目標](flowgrpo-denoising-mdp-and-grpo.html)：為什麼取樣過程可寫成 RL 問題，以及優勢值如何計算。
4. [文字生成圖像擴散模型的 RL 微調脈絡](rl-finetuning-text-to-image-diffusion.html)：RWR、DPO、可微獎勵、PPO 式方法，Flow-GRPO 屬第四類。
5. [ODE-to-SDE 轉換](flow-ode-to-sde-for-rl.html)：論文的第一個核心做法，含更新式。
6. [Denoising Reduction、KL 與實驗結果](flowgrpo-design-choices-and-results.html)：第二個核心做法與消融。

已有的相關筆記：[Flow matching 的速度目標](flow-matching-velocity-target.html)、[RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)、[MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)。時間軸見 `timeline.json`。
