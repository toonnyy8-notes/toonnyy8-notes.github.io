---
slug: psi-ctmc-discrete-diffusion-samplers
title: 連續時間離散擴散（CTMC）與其採樣器
cluster: Diffusion Language Models
kind: topic
tags: [ctmc, discrete-diffusion, sampler, discrete-flow-matching, sedd]
arxiv: ["2602.21185", "2205.14987", "2310.16834", "2407.15595", "2402.04997", "2502.00234"]
sources: ["arxiv.org/abs 摘要（2205.14987）", "paper:2602.21185 第 1、6 節對其餘各工作的描述", "arXiv API metadata（標題、作者、年份）"]
related: [psi-samplers-lineage-hub, duo-self-correction-and-correctors, psi-superposition-posteriors-method, discrete-diffusion-foundations-d3pm]
---
# 連續時間離散擴散（CTMC）與其採樣器

屬於 [Ψ-Samplers 脈絡總覽](psi-samplers-lineage-hub.html)。前置：[離散擴散的基礎](discrete-diffusion-foundations-d3pm.html)。

## Campbell 等人（2022）
*A Continuous Time Framework for Discrete Denoising Models*（arXiv 2205.14987）。摘要：以連續時間馬可夫鏈（CTMC）表述前向加噪與反向生成過程，訓練使用連續時間 ELBO，並推導出可優於離散時間方法的採樣器。論文第 1 節說明：該工作的 PC 方法利用 CTMC 的 rate matrix，但這類採樣器已知表現不如 ancestral sampler（引 Lou et al., 2024；Schiff et al., 2025）。

## 後續以 CTMC 為基礎的工作
- Lou 等人（2023，arXiv 2310.16834）：*Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*。
- Gat 等人（2024，arXiv 2407.15595）：*Discrete Flow Matching*。論文把它與 Campbell 2022 並列為 ReMDM 所推廣的既有 PC 方法。
- Campbell 等人（2024，arXiv 2402.04997）：*Generative Flows on Discrete State-Spaces*，論文第 6 節同樣列為既有 PC 方法。

## 其他離散擴散採樣器（論文第 6 節）
Ren 等人（2025，arXiv 2502.00234），*Fast Solvers for Discrete Diffusion Models*，發展高階求解器。論文指出自身只用一階資訊，兩者互補。

## 與本論文的關係
Ψ-sampler 不經過 CTMC rate matrix，而是直接修改離散時間 posterior（見 [Ψ-posterior](psi-superposition-posteriors-method.html)）。
