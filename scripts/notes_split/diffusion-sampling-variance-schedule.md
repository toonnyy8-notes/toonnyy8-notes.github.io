---
slug: diffusion-sampling-variance-schedule
title: 取樣變異數與 noise schedule：Improved DDPM、Analytic-DPM、EDM
cluster: Diffusion & Generative Models
kind: topic
tags: [variance-schedule, analytic-dpm, edm, sampling]
arxiv: ["2102.09672", "2105.05233", "2201.06503", "2206.00364", "2308.15321"]
sources: [2308.15321]
related: ["exposure-bias-elucidation-overview", "ddpm-reverse-sampling-formulation", "prediction-error-variance-analysis", "epsilon-scaling-method"]
---
# 取樣變異數與 noise schedule

回到 [總覽](exposure-bias-elucidation-overview.html)。

## 文獻與本論文的用法
- Nichol & Dhariwal 2021，*Improved Denoising Diffusion Probabilistic Models*（arXiv:2102.09672）：本論文引用其結論：取樣噪聲變異數 β̇_t 的範圍為 [β̃_t, β_t]，下界對應 q(x_0) 為 delta 函數，上界對應各向同性高斯。本論文也引用其 respacing 技術以進行快速隨機取樣，以及 t 接近 0 時 ε_θ 預測很差、loss 比其他時間步大數個數量級的觀察。
- Dhariwal & Nichol 2021，*Diffusion Models Beat GANs on Image Synthesis*（arXiv:2105.05233）：即 ADM，是本論文主要實驗的預訓練基線（ADM-ES 即 ADM 加 Epsilon Scaling），FID 計算方式也與其一致。
- Bao 等人 2022，*Analytic-DPM*（arXiv:2201.06503）：本論文沿用其以高斯分佈近似 x_θ^t 的作法（x_θ^t = x_0 + e_t ε_0），並指出其最優取樣變異數 β_t* 大於下界 β̃_t，在 1000 步取樣下 FID 反而比 β̃_t 差。本論文認為曝光偏差可解釋此現象，但屬本論文自己的推測（"We believe"）。
- Karras 等人 2022，*Elucidating the Design Space of Diffusion-Based Generative Models*（arXiv:2206.00364）：EDM。本論文將其列為降低預測誤差的方式之一（CIFAR-10 FID 由 3.01 降至 2.51），並在表 3 以其隨機取樣器作為比較對象。

## 結論
本論文由 β̃_t 為取樣變異數下界推得：無法靠調整取樣噪聲 schedule 緩解曝光偏差，因此轉而處理網路輸出（見 [Epsilon Scaling 方法與結果](epsilon-scaling-method.html)）。分析見 [預測誤差與取樣變異數分析](prediction-error-variance-analysis.html)。
