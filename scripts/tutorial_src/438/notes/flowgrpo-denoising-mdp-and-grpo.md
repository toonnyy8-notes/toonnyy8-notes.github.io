---
slug: flowgrpo-denoising-mdp-and-grpo
title: Denoising 作為 MDP，以及 Flow-GRPO 的目標函數
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [mdp, grpo, ppo, advantage, ddpo]
arxiv: ["2505.05470", "2305.13301", "1707.06347", "2402.03300", "2501.12948"]
sources: ["paper.txt §2, §3, §4.1", "arXiv abstract 2305.13301", "arXiv abstract 1707.06347", "arXiv abstract 2402.03300", "arXiv abstract 2501.12948"]
related: ["flowgrpo-lineage-hub", "rl-mdp-q-learning-basics", "rlvr-grpo-credit-assignment", "rl-post-training-dllm-grpo", "flowgrpo-design-choices-and-results"]
---
# Denoising 作為 MDP，以及 Flow-GRPO 的目標函數

屬於 [Flow-GRPO 的研究脈絡總覽](flowgrpo-lineage-hub.html)。MDP 記號見 [MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)，GRPO 與 PPO 的比較見 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)，GRPO 用於擴散語言模型見 [擴散語言模型的 GRPO 後訓練](rl-post-training-dllm-grpo.html)。

## 把去噪寫成 MDP

論文 §3 依 Black 等的 DDPO（2305.13301，2023）把反向去噪寫成多步 MDP $(S,A,\rho_0,P,R)$：

- 狀態 $s_t=(c,t,x_t)$，動作 $a_t=x_{t-1}$，策略 $\pi(a_t\mid s_t)=p(x_{t-1}\mid x_t,c)$。
- 轉移對 $x_{t-1}$ 是確定的，初始狀態為提示與 $x_T\sim\mathcal N(0,I)$。
- 獎勵只在最後一步給出：$R=r(x_0,c)$。

## GRPO

GRPO 出自 DeepSeekMath（Shao 等，2402.03300，2024），後用於 DeepSeek-R1（2501.12948，2025）。論文稱它去掉價值網路，比 PPO（Schulman 等，1707.06347，2017）省記憶體。Flow-GRPO 對每個提示取樣 $G$ 張圖，優勢為組內標準化（式 4）：

$\hat A_t^i=\dfrac{R(x_0^i,c)-\mathrm{mean}(\{R\})}{\mathrm{std}(\{R\})}$

目標（式 5）為逐步 clipped 機率比乘優勢，減去 $\beta D_{KL}(\pi_\theta\|\pi_{ref})$。論文註明 PPO 也能以類似方式用於 flow matching，並選 GRPO。群組大小的影響見 [Denoising Reduction、KL 與實驗結果](flowgrpo-design-choices-and-results.html)。
