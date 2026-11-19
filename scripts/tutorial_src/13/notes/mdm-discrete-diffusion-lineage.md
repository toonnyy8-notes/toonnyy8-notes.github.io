---
slug: mdm-discrete-diffusion-lineage
title: Masked diffusion 的離散擴散脈絡
cluster: Diffusion Language Models
kind: topic
tags: [discrete-diffusion, masked-diffusion, mdlm, d3pm, elbo]
arxiv: ["2107.03006", "2310.16834", "2406.07524", "2406.04329", "2406.03736", "2202.04200", "2512.09106"]
sources: ["paper.txt §1, §2.1", "arXiv API metadata"]
related: [unmasking-policies-dllm-overview, dllm-llada-dream-models, unmasking-mdp-policy-design]
---
# Masked diffusion 的離散擴散脈絡

屬於[總覽](unmasking-policies-dllm-overview.html)。

## 脈絡

論文第 1 節把下列工作列為離散擴散（discrete diffusion）的起點：Austin 等人（2021）的 D3PM「Structured Denoising Diffusion Models in Discrete State-Spaces」（2107.03006）與 Lou 等人（2024）的「Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution」（2310.16834）。

Masked diffusion model（MDM）是離散擴散的子類，前向過程以隨時間變化的 BERT 式遮罩破壞序列。論文引用 Sahoo 等人「Simple and Effective Masked Diffusion Language Models」（2406.07524）、Shi 等人「Simplified and Generalized Masked Diffusion for Discrete Data」（2406.04329）與 Ou 等人「Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data」（2406.03736）作為 MDM 的訓練方法。

## 論文使用的形式

前向過程對每個位置獨立：位置 k 以機率 1 − t 保持 x0 的 token，以機率 t 變成遮罩 token M。訓練目標是遮罩位置的 cross-entropy 加權和，論文式 (2.1) 寫成負 ELBO 的上界。

取樣從全遮罩序列開始，每步由 dLLM 預測所有位置的 token 分佈，再選出一個仍被遮罩的位置子集 U_t 並從預測分佈取樣填入（式 2.2）。選 U_t 的規則就是 unmasking 策略，訓練目標不決定這個規則。

## 與影像生成的關係

論文把 MaskGIT（Chang 等人，2202.04200，「MaskGIT: Masked Generative Image Transformer」）列為 confidence 式 unmasking 的先例：它依 token 信心決定每步解開哪些位置，並非固定隨機取樣。細節以 MaskGIT 原文為準，本筆記只記錄論文的引用關係。
