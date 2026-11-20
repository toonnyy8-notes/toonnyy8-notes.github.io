---
slug: psi-uniform-state-guidance-and-self-correction
title: Uniform-state 擴散、guidance 與自我修正
cluster: Diffusion Language Models
kind: topic
tags: [uniform-state-diffusion, guidance, classifier-free-guidance, self-correction]
arxiv: ["2602.21185", "2412.10193", "2406.01572", "2207.12598", "2503.04482", "2506.10892"]
sources: ["arxiv.org/abs 摘要（2412.10193、2506.10892）", "paper:2602.21185 第 1、2.1.2、2.2 節", "arXiv API metadata（標題、作者、年份）"]
related: [psi-samplers-lineage-hub, psi-duo-curriculum-background, psi-superposition-posteriors-method, uniform-noise-gidd-clean-supervision]
---
# Uniform-state 擴散、guidance 與自我修正

屬於 [Ψ-Samplers 脈絡總覽](psi-samplers-lineage-hub.html)。

## 為何是 uniform 噪聲
論文第 1 節：MDM 每個 token 只更新一次；USDM 允許 token 多次修改，因此可自我修正，在少步生成（Sahoo et al., 2025a，即 Duo）與 guided 生成（Schiff et al., 2025）特別有效。缺點是高步數下品質不及 MDM，且以 likelihood 衡量的建模能力較差。

## Guidance 相關工作
- Ho 與 Salimans（2022，arXiv 2207.12598）：*Classifier-Free Diffusion Guidance*，連續資料的 CFG。
- Nisonoff 等人（2024，arXiv 2406.01572）：*Unlocking Guidance for Discrete State-Space Diffusion and Flow Models*。論文式 (10) 的離散 CFG posterior 引用此工作與 Schiff 等人。
- Schiff 等人（2024，arXiv 2412.10193）：*Simple Guidance Mechanisms for Discrete Diffusion Models*。摘要：推導離散擴散的 classifier-free 與 classifier-based guidance，並提出以 uniform 噪聲、可持續編輯輸出而更易 guide 的模型類別。

## 相關的 uniform 類模型
von Rütte 等人（2025，arXiv 2503.04482），*Generalized Interpolating Discrete Diffusion*，混合 masked 與 uniform 噪聲；論文第 6 節指出 Ψ-sampler 可延伸到這類混合過程。既有筆記：[均勻噪聲與 GIDD](uniform-noise-gidd-clean-supervision.html)。

Duo（arXiv 2506.10892）見 [Duo 與課程學習背景](psi-duo-curriculum-background.html)。
