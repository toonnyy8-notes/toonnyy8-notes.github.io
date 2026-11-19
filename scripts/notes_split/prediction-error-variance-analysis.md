---
slug: prediction-error-variance-analysis
title: 預測誤差與取樣變異數：曝光偏差的解析分析與度量 δ_t
cluster: Diffusion & Generative Models
kind: topic
tags: [exposure-bias, variance-error, metric, analytic-dpm]
arxiv: ["2308.15321", "2201.06503", "2006.11239"]
sources: [2308.15321]
related: [exposure-bias-elucidation-overview, diffusion-exposure-bias, ddpm-training-and-sampling-recap, diffusion-sampling-variance-schedule, epsilon-scaling-method]
---
# 預測誤差與取樣變異數：解析分析與度量 δ_t

回到 [總覽](exposure-bias-elucidation-overview.html)。本篇為 2308.15321 第 3 節的內容。

## 建模
依 [Analytic-DPM 的高斯近似](diffusion-sampling-variance-schedule.html)，令 x_θ^t = x_0 + e_t ε_0（ε_0 ~ N(0, I)）。代入 [DDPM 後驗](ddpm-training-and-sampling-recap.html) 後，取樣時網路實際收到的 x̂_t 之分佈 q_θ(x̂_t | x_{t+1}) 為：平均 √ᾱ_t x_0，變異數 (1 − ᾱ_t + (√ᾱ_t β_{t+1} / (1−ᾱ_{t+1}) · e_{t+1})²) I。訓練時的 q(x_t|x_0) 變異數為 (1−ᾱ_t) I。

## 結論
- 取樣分佈的變異數在每一步都大於訓練分佈，差距為 (√ᾱ_t β_{t+1} / (1−ᾱ_{t+1}) · e_{t+1})²，且此差距在單步即出現。
- 多步取樣中該誤差累積；CIFAR-10 的 20 步實驗顯示越接近 t=1（取樣終點）變異數誤差越大。本論文將預測誤差 x_θ^t − x_0 視為曝光偏差的根本原因。
- 取樣噪聲變異數不能低於 β̃_t，故無法靠調整 noise schedule 消除此誤差。

## 度量
本論文以多步取樣下 q_θ(x̂_t|x_T) 的變異數誤差量化曝光偏差：δ_t = (√β'_t − √β̄_t)²，其中 β̄_t = 1−ᾱ_t 為訓練分佈變異數，β'_t 為一般取樣過程的變異數。其依據是 Fréchet distance，並在附錄 A.6 以經驗方式指出 δ_t 與給定模型的 FID 有強相關。度量的量測步驟為附錄 Algorithm 3。

接下來的處理方法見 [Epsilon Scaling 方法與結果](epsilon-scaling-method.html)；一般概念見 [擴散模型的曝光偏差](diffusion-exposure-bias.html)。
