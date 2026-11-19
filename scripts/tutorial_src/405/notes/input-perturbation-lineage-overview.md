---
slug: input-perturbation-lineage-overview
title: Input Perturbation（DDPM-IP）的研究脈絡總覽
cluster: Diffusion & Generative Models
kind: hub
tags: [exposure-bias, ddpm, input-perturbation, regularization]
arxiv: ["2301.11706", "1503.03585", "2006.11239", "1506.03099", "1511.06732", "1910.00292", "1612.00563"]
sources: ["paper.txt §1, §2, §4", "arXiv API metadata"]
related: ["ddpm-training-and-sampling-recap", "ddpm-quality-and-ddim-sampling", "fewer-steps-sampling-acceleration", "teacher-forcing-and-scheduled-sampling", "diffusion-exposure-bias", "vicinal-risk-and-smoothness-regularization", "input-perturbation-ddpm-ip"]
---
# Input Perturbation（DDPM-IP）的研究脈絡總覽

Ning 等人（2301.11706）指出 DDPM 的訓練與取樣之間有輸入不一致：訓練時網路收到真實的 x_t，取樣時網路收到前一步自己產生的 x̂_t。論文將這個現象類比自回歸文字生成中的 exposure bias，並提出在訓練時對網路輸入加額外高斯雜訊（input perturbation），不改變預測目標。

閱讀順序如下，每篇只處理一個主題。

1. [DDPM 的訓練與取樣流程](ddpm-training-and-sampling-recap.html)：Sohl-Dickstein 等人（2015）與 Ho 等人（2020）定義的前向與反向過程，以及 Algorithm 1、2。
2. [生成品質與 DDIM 取樣](ddpm-quality-and-ddim-sampling.html)：Nichol & Dhariwal、Dhariwal & Nichol 與 DDIM，包含「取樣步數少於訓練步數反而較好」的既有觀察。
3. [減少取樣步數的方法](fewer-steps-sampling-acceleration.html)：論文在 Related Work 列出的加速方法，本論文宣稱與它們正交。
4. [Teacher forcing 與 scheduled sampling](teacher-forcing-and-scheduled-sampling.html)：exposure bias 一詞在序列生成中的來源。
5. [擴散模型的 exposure bias](diffusion-exposure-bias.html)：把 4 的概念對應到擴散模型，說明誤差如何沿取樣鏈累積與如何量測。
6. [Vicinal risk 與平滑性正則化](vicinal-risk-and-smoothness-regularization.html)：論文用來解釋方法的 VRM、Mixup 與 Lipschitz 連續性。
7. [DDPM-IP：訓練時的輸入擾動](input-perturbation-ddpm-ip.html)：Algorithm 3、γ 的選擇、與其他設計的差異。

時間線見 `timeline.json`。
