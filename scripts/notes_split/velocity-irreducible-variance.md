---
slug: velocity-irreducible-variance
title: Velocity Irreducible Variance（VIV）：以速度歧義衡量 diffusability
cluster: Diffusion & Generative Models
kind: topic
tags: [viv, velocity-ambiguity, flow-matching, diffusability, anisotropy]
sources: [latent-diffusion-rae.md]
arxiv: ["2606.03578"]
related: ["latent-diffusion-rae-overview", "diffusability-systematic-study", "representation-autoencoder-rae", "rae-distillation-drifting", "latent-diffusion-open-problems"]
---
# Velocity Irreducible Variance（VIV）：以速度歧義衡量 diffusability

Zhong et al.（Diffusing in the Right Space，arXiv:2606.03578）的核心方法論創新。屬於 [Latent Diffusion 總覽](latent-diffusion-rae-overview.html)；整體實驗見 [系統性比較研究](diffusability-systematic-study.html)。

- Flow Matching 目標可分解為 **Reducible Error**（v* 與 vθ 的差）+ **Irreducible Variance**（v 與 Bayes 最優 v* 的差）。後者反映 trajectory crossing 造成的速度歧義。 [arxiv](https://arxiv.org/pdf/2606.03578)
- 把 class-conditional latent 建模為 anisotropic Gaussian（整體為 L-component GMM）。因 d≫M，採 **Kronecker Flip-Flop 協方差分解** Σ≈Σs⊗Σc（channel 與 spatial 可分離），大幅降低參數量、提升每個因子的有效樣本數。
- 解析形式：Iₖ(t)=Σλₖ,ᵢ/[(1−t)²+t²λₖ,ᵢ]；對 t∼U(0,1) 積分得 **Iₖ=(π/2)Σ√λₖ,ᵢ**，可再改寫為 Iₖ=(π/2)√(d(τₖ−d·Aₖ))，其中 τₖ=tr(Σₖ) 為總變異、Aₖ 為標準差譜的各向異性。 [arxiv](https://arxiv.org/pdf/2606.03578)
- **兩個 insight**：(1) **Intra-class compactness**：固定譜形，減小 τₖ 降低 Iₖ；(2) **Spectral anisotropy**：固定總變異，更各向異性（Aₖ 大）降低 Iₖ。∂Iₖ/∂τₖ>0、∂Iₖ/∂Aₖ<0。 [arxiv](https://arxiv.org/pdf/2606.03578)
- 對 T2I 等非 class-conditional 場景，target 不能化簡為單一 class Gaussian，改用 GMM + Monte Carlo 估計 I。
- 結果：與 gFID **Pearson 0.87**，最穩定。

Insight 2 對 RAE 的意涵（高各向異性有利訓練、卻不利蒸餾）見 [RAE](representation-autoencoder-rae.html) 與 [RAE 蒸餾](rae-distillation-drifting.html)。

## 假設限制
VIV 的解析形式建立在 anisotropic Gaussian / GMM 與 Kronecker 可分離協方差假設上，在高度非高斯、極端各向異性的 RAE latent 上是否精確成立，論文未直接驗證（列為 [開放問題](latent-diffusion-open-problems.html)）。
