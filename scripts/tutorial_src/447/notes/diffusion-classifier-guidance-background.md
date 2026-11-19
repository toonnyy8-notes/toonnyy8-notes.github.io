---
slug: diffusion-classifier-guidance-background
title: 擴散模型的分類器引導與餘弦排程
cluster: Diffusion & Generative Models
kind: topic
tags: [diffusion, classifier-guidance, cosine-schedule, ddpm]
arxiv: ["1503.03585", "2006.11239", "2102.09672", "2105.05233", "2205.09991"]
sources: ["paper.txt §2.2, §3.1, §3.2, §6", "arXiv API metadata (2102.09672, 2105.05233)"]
related: ["eps-prediction-ddpm-score", "ddpm-training-and-sampling-recap", "control-as-inference-guided-sampling", "trajectory-diffusion-planning"]
---
# 擴散模型的分類器引導與餘弦排程

Diffuser 直接沿用影像擴散模型的三項既有技術。DDPM 的前向與反向過程、ε-prediction 訓練目標已有現成筆記：[ε-prediction 與 DDPM、score-based 模型](eps-prediction-ddpm-score.html)、[DDPM 的訓練與取樣流程](ddpm-training-and-sampling-recap.html)，此處不重寫。

## 論文引用的三項技術

| 技術 | 出處（已核對 arXiv） | 在本論文的用途 |
| --- | --- | --- |
| 擴散機率模型 | Sohl-Dickstein et al., 2015（1503.03585）；Ho, Jain, Abbeel, 2020（2006.11239） | 軌跡的生成模型 p_θ(τ)；訓練用簡化目標 E‖ε − ε_θ(τ^i, i)‖² |
| 餘弦噪聲排程 | Nichol & Dhariwal, 2021，*Improved Denoising Diffusion Probabilistic Models*（2102.09672） | 反向過程的共變異數 Σ^i 依此排程 |
| 分類器引導 | Dhariwal & Nichol, 2021，*Diffusion Models Beat GANs on Image Synthesis*（2105.05233） | 用分類器梯度做類別條件取樣；論文將其對應到 RL 的回報引導 |

## 引導的形式

論文 §3.2 以公式 (3) 近似引導後的反向轉移：p_θ(τ^{i-1} | τ^i, O_{1:T}) ≈ N(τ^{i-1}; μ + Σg, Σ)，其中 g = ∇_τ log p(O_{1:T} | τ) 在 τ = μ 取值，等於 ∇J(μ)。論文說明此近似需要 p(O_{1:T} | τ^i) 夠平滑，並引用 Sohl-Dickstein et al. 2015。這與影像中以分類器梯度修改反向轉移平均值的做法相同，只是分類器換成預測累積獎勵的 J_φ。

## 修補式條件（inpainting）

論文 §3.3 把目標條件寫成類似影像修補：已知的狀態或動作視為已觀測像素，在每個擴散步之後把取樣值換成條件值。論文以 Sohl-Dickstein et al. 2015 作為此類條件的出處。

回報引導的設計與演算法見 [軌跡層級擴散規劃（Diffuser）](trajectory-diffusion-planning.html)；RL 目標如何寫成條件取樣見 [控制即推論](control-as-inference-guided-sampling.html)。
