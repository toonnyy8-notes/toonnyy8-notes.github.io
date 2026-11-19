---
slug: ddpm-training-and-sampling-recap
title: DDPM 的訓練與取樣流程
cluster: Diffusion & Generative Models
kind: topic
tags: [ddpm, forward-process, reverse-process, epsilon-prediction]
arxiv: ["1503.03585", "2006.11239", "2301.11706"]
sources: ["paper.txt §1, §3", "arXiv API metadata"]
related: ["eps-prediction-ddpm-score", "diffusion-exposure-bias"]
---
# DDPM 的訓練與取樣流程

## 來源

- Sohl-Dickstein 等人（2015），*Deep Unsupervised Learning using Nonequilibrium Thermodynamics*（1503.03585）：Ning 等人將擴散模型的提出歸於此文。
- Ho 等人（2020），*Denoising Diffusion Probabilistic Models*（2006.11239）：本論文以此文的設定作為 DDPM 標準演算法。

ε 預測與 score-based 的對應已有專文：[ε-prediction 與 DDPM、score-based 模型](eps-prediction-ddpm-score.html)。本篇只整理理解 DDPM-IP 所需的部分。

## 前向過程

給定資料 x_0 與固定的雜訊排程 (β_1, ..., β_T)，前向過程是 Markov chain，每一步加入高斯雜訊，共 T 步。論文的 Eq. 4 寫成

x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε，ε ~ N(0, I)。

所以任一步的 x_t 可由 x_0 與一個 ε 直接算出，不需逐步模擬。

## 反向過程與訓練

反向過程由 θ 參數化的轉移機率 p_θ(x_{t−1} | x_t) 定義，Eq. 5 的訓練目標是

L(θ) = E[ ‖ε − ε_θ(x_t, t)‖² ]。

Algorithm 1 的步驟：取 x_0、均勻取 t、取 ε，用 Eq. 4 算 x_t，對 ‖ε − ε_θ(x_t, t)‖² 做梯度下降。

## 取樣

Algorithm 2 從 x̂_T ~ N(0, I) 開始，t 由 T 到 1 反覆用 ε_θ(x̂_t, t) 算出 x̂_{t−1}（t > 1 時再加 σ_t z），最後輸出 x̂_0。

## 與本論文的關係

對照 Algorithm 1 與 2 的第 4 行，ε_θ 的輸入在訓練時是 x_t（由真實 x_0 算出），在取樣時是 x̂_t（由前一步輸出算出）。這個差異是 [擴散模型的 exposure bias](diffusion-exposure-bias.html) 的出發點。

論文另指出（引用 Xiao 等人 2022）：T 必須很大，是因為每一步被假設為高斯，而這個假設只在步長小時成立。
