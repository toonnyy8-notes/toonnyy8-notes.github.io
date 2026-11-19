---
slug: ddpm-quality-and-ddim-sampling
title: 生成品質改進與 DDIM 取樣
cluster: Diffusion & Generative Models
kind: topic
tags: [ddpm, ddim, adm, score-based, sampling-steps]
arxiv: ["1907.05600", "2011.13456", "2102.09672", "2105.05233", "2010.02502", "2301.11706"]
sources: ["paper.txt §1, §2, §6", "arXiv API metadata"]
related: ["ddpm-training-and-sampling-recap", "diffusion-exposure-bias", "input-perturbation-ddpm-ip"]
---
# 生成品質改進與 DDIM 取樣

DDPM 的基本流程見 [DDPM 的訓練與取樣流程](ddpm-training-and-sampling-recap.html)。本論文在 Related Work 寫道，擴散模型由 Sohl-Dickstein 等人（2015）提出，之後由 Song & Ermon（2019）、Ho 等人（2020）、Song 等人（2021b）、Nichol & Dhariwal（2021）改進。以下是與實驗直接相關的幾篇。

| 論文 | arXiv | 在本論文中的角色 |
| --- | --- | --- |
| Song & Ermon 2019，*Generative Modeling by Estimating Gradients of the Data Distribution* | 1907.05600 | 列為擴散模型的改進之一 |
| Song 等人 2021b，*Score-Based Generative Modeling through Stochastic Differential Equations* | 2011.13456 | 同上 |
| Nichol & Dhariwal 2021，*Improved Denoising Diffusion Probabilistic Models* | 2102.09672 | 觀察到取樣步數少於訓練步數時結果可能更好，但未提供解釋 |
| Dhariwal & Nichol 2021，*Diffusion Models Beat GANs on Image Synthesis* | 2105.05233 | 提出 ADM，本論文的基準模型 |
| Song 等人 2021a，*Denoising Diffusion Implicit Models* | 2010.02502 | 非 Markov 擴散過程，可用比訓練時少的取樣步數而不需重新訓練；本論文也測試 DDIM-IP |

## 與本論文的關係

- ADM 是本論文主要的基準：ADM-IP 指在 ADM 訓練時加入 input perturbation。
- 本論文的 Sec. 6.2 顯示，以 T 步訓練的標準 DDPM，用 T′ < T 步取樣可得到更好的結果；論文說 Nichol & Dhariwal（2021）也觀察到類似現象。作者的解釋是：長取樣鏈較符合高斯假設，但也累積較多誤差，兩者在約 100 至 300 步之間取得折衷。這個解釋見 [擴散模型的 exposure bias](diffusion-exposure-bias.html)。
- DDIM 對照（論文 Table 5，CIFAR10，T = 1,000 訓練）：η = 0 時，10 步 FID 由 14.21（DDIM）降為 10.54（DDIM-IP），1,000 步為 4.29 與 4.27。

方法本身見 [DDPM-IP：訓練時的輸入擾動](input-perturbation-ddpm-ip.html)。
