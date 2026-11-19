---
slug: rae-distillation-drifting
title: RAE latent 的蒸餾困難與 Drifting Models
cluster: Diffusion & Generative Models
kind: topic
tags: [distillation, drifting-models, anisotropy, rae, one-step-generation]
sources: [latent-diffusion-rae.md]
arxiv: ["2606.15553"]
related: ["latent-diffusion-rae-overview", "representation-autoencoder-rae", "velocity-irreducible-variance", "latent-diffusion-open-problems"]
---
# RAE latent 的蒸餾困難與 Drifting Models

Distilling Drifting Transformers，arXiv:2606.15553。屬於 [Latent Diffusion 總覽](latent-diffusion-rae-overview.html)。

- **診斷**：[RAE](representation-autoencoder-rae.html) latent 因 DINO 特徵強烈 label-wise 聚類而**各向異性極高、ODE 軌跡曲率比 SD-VAE 大約兩個數量級**（isotropy 統計：participation ratio PR 0.2068→0.0630、spectral entropy SE 0.3781→0.2110；dispersion NN-d 4.3083→1.0664、S-MMD 0.00576→0.07913）。
- 傳統 trajectory-based distillation 假設 latent 近似 isotropic、軌跡平滑，故在 RAE 上不穩定。
- **解法**：Drifting Models（Deng et al. 2026）直接用兩批樣本計算 drifting field 而非匹配 ODE 軌跡，避開高曲率問題； [arxiv](https://arxiv.org/pdf/2606.15553) 並理論證明 Drifting Model 在極度分散的 reconstruction-based VAE latent 上反而可能失敗——RAE 與 drifting 是互相契合的組合。
- 限制：需專用蒸餾範式。

## 與 diffusability 的張力
依 [VIV](velocity-irreducible-variance.html) 的 Insight 2，高各向異性降低 irreducible variance、有利訓練；但同一性質使軌跡高曲率、不利蒸餾。

## 建議（蒸餾/一步生成）
RAE latent 上**避免 trajectory-based distillation**，改用 Drifting Models；或先評估 latent 的 PR/SE/曲率再決定。

## 開放問題：訓練–蒸餾張力的統一解
能否設計一個 latent space（或 reparameterization）同時滿足「訓練時高各向異性（低 VIV）」與「蒸餾時低曲率（近 isotropic）」？Drifting Models 是一種繞道，但是否有 latent 側的根本解？（亦列於 [開放問題](latent-diffusion-open-problems.html)）
