---
slug: input-perturbation-ddpm-ip
title: DDPM-IP：訓練時的輸入擾動
cluster: Diffusion & Generative Models
kind: topic
tags: [ddpm-ip, input-perturbation, regularization, exposure-bias]
arxiv: ["2301.11706", "2105.05233", "2010.02502", "2206.00364", "2308.15321"]
sources: ["paper.txt Abstract, §5.1–5.3, Alg. 3, Tab. 5", "arXiv API metadata", "2308.15321 §3.4"]
related: ["diffusion-exposure-bias", "vicinal-risk-and-smoothness-regularization", "ddpm-training-and-sampling-recap", "ddpm-quality-and-ddim-sampling"]
---
# DDPM-IP：訓練時的輸入擾動

Ning、Sangineto、Porrello、Calderara、Cucchiara，*Input Perturbation Reduces Exposure Bias in Diffusion Models*（ICML 2023，arXiv 2301.11706）。

## 演算法

Algorithm 3 與 Algorithm 1 的差別只有輸入。取額外的 ξ ~ N(0, I)，建立擾動輸入

y_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · (ε + γ_t ξ)（Eq. 6），

然後對 ‖ε − ε_θ(y_t, t)‖² 做梯度下降。取樣仍使用不變的 Algorithm 2。論文說明只需兩行程式碼，不改網路與損失函數。

## 不對稱用法

ξ 只加在網路輸入，不進入預測目標 ε，所以不等於在 Algorithm 1 中換一個 ε。論文推出 y_t ~ N(√ᾱ_t x_0, (1−ᾱ_t)(1+γ²) I)（Eq. 7）。若在標準演算法中直接用此分佈（稱 DDPM-y，Eq. 8），輸入相同但目標改為 ε′，論文的附錄 A.3 顯示 DDPM-y 比標準 DDPM 更差。

## γ 的選擇

γ_t 取常數 γ，不另設第二個排程。論文以 CIFAR10 與 ImageNet 32×32 的網格搜尋，範圍為 ν_t 的區間 [0, 0.2]（參考 EDM 指出推論軌跡後段影響最大），最後取 γ = 0.1，所有資料集與基準都用同一值。論文承認針對各模型調 γ 可能更好。誤差分佈的量測見 [擴散模型的 exposure bias](diffusion-exposure-bias.html)。

## 結果（依論文）

- CelebA 64×64：FID 1.27，節省 37.5% 訓練時間（摘要）。
- 在 CIFAR10、ImageNet 32×32、LSUN tower 64×64、CelebA 64×64、FFHQ 128×128 上，ADM-IP 以較少取樣步數達到不輸 1,000 步 ADM 的 FID（例如 CIFAR10 為 80 步 2.93 對 1,000 步 2.99）；論文稱加速超過 12.5 倍。
- 與 DDIM 併用（Tab. 5）：見 [生成品質改進與 DDIM 取樣](ddpm-quality-and-ddim-sampling.html)。

## 2308.15321 的描述

2308.15321 稱 Ning 等人首次指出擴散模型的曝光偏差，並引入輸入擾動來模擬取樣時的預測誤差，其依據是 Lipschitz 連續性；在第 3.4 節歸類為資料增強：以高斯項擴增每個訓練樣本 x_t，FID 有大幅改善。該方法需要重新訓練。與其他緩解方法的比較見 [曝光偏差的既有緩解方法](exposure-bias-prior-mitigations.html)。

## 侷限（論文自述）

只用小解析度資料集；只在影像領域驗證。為何有效的理論解釋見 [Vicinal risk 與平滑性正則化](vicinal-risk-and-smoothness-regularization.html)；訓練流程回顧見 [DDPM 的訓練與取樣流程](ddpm-training-and-sampling-recap.html)。
