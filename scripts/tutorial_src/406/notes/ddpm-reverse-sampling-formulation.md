---
slug: ddpm-reverse-sampling-formulation
title: DDPM 反向取樣公式：後驗、ε 預測與 DDIM
cluster: Diffusion & Generative Models
kind: topic
tags: [ddpm, ddim, score-based, posterior, epsilon-prediction]
arxiv: ["1503.03585", "2006.11239", "1907.05600", "2011.13456", "2010.02502", "2308.15321"]
sources: [2308.15321]
related: ["exposure-bias-elucidation-overview", "eps-prediction-ddpm-score", "diffusion-sampling-variance-schedule", "prediction-error-variance-analysis"]
---
# DDPM 反向取樣公式：後驗、ε 預測與 DDIM

回到 [總覽](exposure-bias-elucidation-overview.html)。ε 預測與 score-based 模型的一般介紹見 [ε-prediction 與 DDPM、score-based 模型](eps-prediction-ddpm-score.html)。

## 本論文引言與第 2 節對這些工作的描述
- Sohl-Dickstein 等人 2015，*Deep Unsupervised Learning using Nonequilibrium Thermodynamics*（arXiv:1503.03585）：擴散模型由其提出，後由 Song & Ermon（2019）與 Nichol & Dhariwal（2021）改進。
- Song & Ermon 2019，*Generative Modeling by Estimating Gradients of the Data Distribution*（arXiv:1907.05600）：score-based 路線。
- Ho、Jain、Abbeel 2020，*Denoising Diffusion Probabilistic Models*（arXiv:2006.11239）：本論文第 2 節採用其記號。前向過程 q(x_t | x_0) = N(√ᾱ_t x_0, (1−ᾱ_t)I)；反向以 p_θ(x_{t−1}|x_t) = N(μ_θ(x_t,t), σ_t I) 近似，目標是與後驗 q(x_{t−1}|x_t,x_0) 的 KL。Ho 等人發現以網路預測 ε 的效果在實務上優於預測 x_0，此時 μ_θ = (1/√α_t)(x_t − β_t/√(1−ᾱ_t) · ε_θ(x_t,t))。
- Song 等人 2021，*Score-Based Generative Modeling through Stochastic Differential Equations*（arXiv:2011.13456）：本論文稱其以 SDE 統一 score-based 模型與 DDPM。
- Song、Meng、Ermon 2021，*Denoising Diffusion Implicit Models*（arXiv:2010.02502）：本論文附錄 A.3 推導 DDIM 的取樣分佈 q_θ(x̂_t|x_{t+1})，並在 5.3 節以 η=0 為 DDIM、η=1 為 DDPM 進行實驗。

## 與本論文分析的連結
取樣分佈 q(x_{t−1}|x_t, x_θ^t) 與訓練時的後驗只有在 x_θ^t = x_0 時才相同。因此預測誤差 x_θ^t − x_0 成為分析起點，見 [預測誤差與取樣變異數分析](prediction-error-variance-analysis.html)。取樣變異數的選擇見 [取樣變異數與 noise schedule](diffusion-sampling-variance-schedule.html)。
