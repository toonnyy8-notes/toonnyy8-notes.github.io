---
slug: decision-diffuser-lineage-hub
title: Decision Diffuser 的研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [decision-diffuser, conditional-diffusion, offline-rl, lineage]
arxiv: ["2211.15657", "2205.09991", "2207.12598", "2206.01714", "2106.01345", "1606.07419", "1804.08606"]
sources: ["paper.txt §1-§5, Appendix A-M", "arXiv API metadata"]
related: ["rl-mdp-q-learning-basics", "offline-rl-distribution-shift", "sequence-modeling-offline-rl", "return-conditioned-sequence-modeling", "trajectory-diffusion-planning", "diffusion-classifier-guidance-background", "ddpm-training-and-sampling-recap", "inverse-dynamics-state-diffusion", "classifier-free-guidance-return-conditioning", "implicit-dp-trajectory-stitching", "composing-constraints-and-skills", "dd-experiments-and-ablations", "dd-limitations-stochastic-dynamics"]
---
# Decision Diffuser 的研究脈絡總覽

論文 "Is Conditional Generative Modeling all you need for Decision-Making?"（Ajay 等，ICLR 2023，arXiv 2211.15657）把離線決策寫成條件生成建模：對狀態序列做條件擴散，再用逆動力學模型取出動作。論文稱此方法為 Decision Diffuser（DD），不估計 Q 函數，也不使用 TD 學習。

建議閱讀順序如下。

1. 背景：MDP 與 Q-learning 見[MDP 與 Q-learning 基礎](rl-mdp-q-learning-basics.html)；TD 學習在離線設定下的問題見[分布偏移與 OOD 動作](offline-rl-distribution-shift.html)。
2. 前人的離線決策做法：以序列模型建模見[Decision Transformer 與 Trajectory Transformer](sequence-modeling-offline-rl.html)，以報酬為條件的監督式學習見[以報酬為條件的監督式學習](return-conditioned-sequence-modeling.html)。
3. 前人的擴散規劃：[Diffuser](trajectory-diffusion-planning.html) 對狀態與動作的聯合序列做擴散，並以分類器引導取得高報酬軌跡；擴散模型本身見 [DDPM 訓練與取樣](ddpm-training-and-sampling-recap.html)，分類器引導見[分類器引導背景](diffusion-classifier-guidance-background.html)。
4. 本論文的設計，依序為：
   - [只擴散狀態、以逆動力學取動作](inverse-dynamics-state-diffusion.html)
   - [無分類器引導與報酬條件](classifier-free-guidance-return-conditioning.html)
   - [不做動態規劃的軌跡拼接](implicit-dp-trajectory-stitching.html)
   - [組合限制與技能](composing-constraints-and-skills.html)
5. 實驗與限制：[實驗與消融](dd-experiments-and-ablations.html)、[限制與隨機動力學](dd-limitations-stochastic-dynamics.html)。

時間軸見同資料夾的 timeline.json。
