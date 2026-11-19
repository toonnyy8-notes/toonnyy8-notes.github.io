---
slug: ddpm-training-and-sampling-recap
title: DDPM 的訓練與取樣流程
cluster: Diffusion & Generative Models
kind: topic
tags: [ddpm, ddim, forward-process, reverse-process, epsilon-prediction, score-based, posterior]
arxiv: ["1503.03585", "2006.11239", "2301.11706", "1907.05600", "2011.13456", "2010.02502", "2308.15321"]
sources: ["paper.txt §1, §3", "arXiv API metadata", "2308.15321"]
related: ["eps-prediction-ddpm-score", "diffusion-exposure-bias", "exposure-bias-elucidation-overview", "diffusion-sampling-variance-schedule", "prediction-error-variance-analysis"]
---
# DDPM 的訓練與取樣流程

屬於[總覽](exposure-bias-elucidation-overview.html)。

## 來源

- Sohl-Dickstein 等人（2015），*Deep Unsupervised Learning using Nonequilibrium Thermodynamics*（1503.03585）：Ning 等人將擴散模型的提出歸於此文；2308.15321 的引言同樣如此描述，並說後由 Song & Ermon（2019）與 Nichol & Dhariwal（2021）改進。
- Ho 等人（2020），*Denoising Diffusion Probabilistic Models*（2006.11239）：2301.11706 以此文的設定作為 DDPM 標準演算法；2308.15321 第 2 節採用其記號。
- Song & Ermon 2019，*Generative Modeling by Estimating Gradients of the Data Distribution*（1907.05600）：score-based 路線。
- Song 等人 2021，*Score-Based Generative Modeling through Stochastic Differential Equations*（2011.13456）：2308.15321 稱其以 SDE 統一 score-based 模型與 DDPM。
- Song、Meng、Ermon 2021，*Denoising Diffusion Implicit Models*（2010.02502）：2308.15321 附錄 A.3 推導 DDIM 的取樣分佈 q_θ(x̂_t|x_{t+1})，並在 5.3 節以 η=0 為 DDIM、η=1 為 DDPM 進行實驗。

ε 預測與 score-based 的對應已有專文：[ε-prediction 與 DDPM、score-based 模型](eps-prediction-ddpm-score.html)。本篇只整理理解 DDPM-IP（No.405，2301.11706）與 2308.15321 所需的部分。

## 前向過程

給定資料 x_0 與固定的雜訊排程 (β_1, ..., β_T)，前向過程是 Markov chain，每一步加入高斯雜訊，共 T 步。2301.11706 的 Eq. 4 寫成

x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε，ε ~ N(0, I)。

所以任一步的 x_t 可由 x_0 與一個 ε 直接算出，不需逐步模擬。2308.15321 的記號為 q(x_t | x_0) = N(√ᾱ_t x_0, (1−ᾱ_t)I)。

## 反向過程與訓練

反向過程由 θ 參數化的轉移機率 p_θ(x_{t−1} | x_t) 定義，Eq. 5 的訓練目標是

L(θ) = E[ ‖ε − ε_θ(x_t, t)‖² ]。

Algorithm 1 的步驟：取 x_0、均勻取 t、取 ε，用 Eq. 4 算 x_t，對 ‖ε − ε_θ(x_t, t)‖² 做梯度下降。

在 2308.15321 的描述中，反向以 p_θ(x_{t−1}|x_t) = N(μ_θ(x_t,t), σ_t I) 近似，目標是與後驗 q(x_{t−1}|x_t,x_0) 的 KL。Ho 等人發現以網路預測 ε 的效果在實務上優於預測 x_0，此時 μ_θ = (1/√α_t)(x_t − β_t/√(1−ᾱ_t) · ε_θ(x_t,t))。

## 取樣

Algorithm 2 從 x̂_T ~ N(0, I) 開始，t 由 T 到 1 反覆用 ε_θ(x̂_t, t) 算出 x̂_{t−1}（t > 1 時再加 σ_t z），最後輸出 x̂_0。

## 與兩篇論文的關係

對照 Algorithm 1 與 2 的第 4 行，ε_θ 的輸入在訓練時是 x_t（由真實 x_0 算出），在取樣時是 x̂_t（由前一步輸出算出）。這個差異是 [擴散模型的 exposure bias](diffusion-exposure-bias.html) 的出發點。

論文（2301.11706）另指出（引用 Xiao 等人 2022）：T 必須很大，是因為每一步被假設為高斯，而這個假設只在步長小時成立。

在 2308.15321 中，取樣分佈 q(x_{t−1}|x_t, x_θ^t) 與訓練時的後驗只有在 x_θ^t = x_0 時才相同。因此預測誤差 x_θ^t − x_0 成為分析起點，見 [預測誤差與取樣變異數分析](prediction-error-variance-analysis.html)。取樣變異數的選擇見 [取樣變異數與 noise schedule](diffusion-sampling-variance-schedule.html)。
