---
slug: traj-diffusion-lineage-hub
title: Diffuser 研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [diffuser, lineage, model-based-rl, diffusion, offline-rl]
arxiv: ["2205.09991"]
sources: ["paper.txt §1, §6 and reference list"]
related: ["rl-mdp-q-learning-basics", "offline-rl-distribution-shift", "model-based-planning-abstraction-barrier", "control-as-inference-guided-sampling", "ddpm-training-and-sampling-recap", "eps-prediction-ddpm-score", "diffusion-classifier-guidance-background", "offline-rl-benchmarks-and-baselines", "sequence-modeling-offline-rl", "trajectory-diffusion-planning"]
---
# Diffuser 研究脈絡總覽

*Planning with Diffusion for Flexible Behavior Synthesis*（2205.09991）結合三條脈絡：軌跡最佳化與 model-based RL、控制即推論、擴散模型。依閱讀順序如下。

1. **RL 基礎**：[MDP 與 Q-learning 基礎](rl-mdp-q-learning-basics.html)、[離線 RL 的分佈偏移](offline-rl-distribution-shift.html)。
2. **學習模型加規劃器的做法與缺點**：[學習式動態模型與軌跡最佳化的分工](model-based-planning-abstraction-barrier.html)。
3. **把 RL 寫成條件取樣**：[控制即推論](control-as-inference-guided-sampling.html)。
4. **擴散模型基礎**：[ε-prediction 與 DDPM、score-based 模型](eps-prediction-ddpm-score.html)、[DDPM 的訓練與取樣流程](ddpm-training-and-sampling-recap.html)，再接 [分類器引導與餘弦排程](diffusion-classifier-guidance-background.html)。
5. **評估環境與比較對象**：[離線 RL 基準與比較方法](offline-rl-benchmarks-and-baselines.html)、[以序列模型處理離線 RL](sequence-modeling-offline-rl.html)。
6. **本論文方法**：[軌跡層級擴散規劃（Diffuser）](trajectory-diffusion-planning.html)。

時間軸見同資料夾的 timeline.json。
