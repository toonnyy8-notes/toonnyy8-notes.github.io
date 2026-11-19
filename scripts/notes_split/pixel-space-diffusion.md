---
slug: pixel-space-diffusion
title: Pixel-space diffusion：latent 層是否必要的第三路線
cluster: Diffusion & Generative Models
kind: topic
tags: [pixel-space-diffusion, jit, pixeldit, pixelgen, epg]
sources: [latent-diffusion-rae.md]
arxiv: ["2511.13720", "2511.20645", "2602.02493", "2510.12586", "2606.24888"]
related: ["latent-diffusion-rae-overview", "representation-autoencoder-rae", "generative-eval-metrics", "optimization-dilemma-semantic-alignment"]
---
# Pixel-space diffusion：latent 層是否必要的第三路線

屬於 [Latent Diffusion 總覽](latent-diffusion-rae-overview.html)。

## pixel-space 陣營
- JiT（Li & He 2025，Back to Basics / arXiv:2511.13720 相關）：x-prediction 取代 velocity。
- PixelDiT（arXiv:2511.20645）：單階段全 transformer，dual-level DiT 分離全域語意與 pixel 紋理，直接在 1K 解析度訓練；主打免 VAE artifact、編輯時背景一致性佳。
- PixelGen（arXiv:2602.02493）：perceptual loss 讓 pixel diffusion 聚焦感知流形；ImageNet-256 無 CFG FID 5.11（80 epoch）、T2I GenEval 0.79。
- There is No VAE / EPG（arXiv:2510.12586）：兩階段自監督預訓練 + end-to-end；ImageNet-256 FID 1.58、512 FID 2.35（75 NFE），僅用 DiT 約 30% compute；consistency model 直接在高解析度訓練達 FID 8.82。

## 核心爭論
若 latent 的價值主要來自語意先驗（[RAE](representation-autoencoder-rae.html)）而非壓縮，pixel-space + 表徵預訓練是否就能取消 latent 層？

## 關鍵反面證據
DiffusionBench 在統一 NanoGen 框架重訓 JiT/PixNerd/PixelGen [arXiv](https://arxiv.org/html/2606.24888v1) 時，**在 80 epoch 未觀察到加速收斂**（如同傳統 VAE），且 pixel-space 方法在 ImageNet FID 與 T2I 指標上都明顯落後 latent-space 方法（作者主分析甚至排除 pixel-space 以免人為抬高相關性；DiffusionBench 見 [生成評估指標](generative-eval-metrics.html)）。iREPA 亦顯示其空間結構技巧能加速 pixel-space JiT（見 [語意/空間對齊](optimization-dilemma-semantic-alignment.html)）——暗示「表徵/空間先驗」才是共通關鍵，而非 latent 與否。

## 判斷
目前實證天平仍偏向「需要某種 encoder 承載語意/空間先驗」；pixel-space 尚未在統一框架下複現 RAE/latent 的收斂優勢。latent 層的必要性未被證偽，但其角色正從「壓縮」重新定義為「表徵基座」。

門檻：若 pixel-space 方法在統一框架（NanoGen 類）下開始展現與 latent 相當的收斂加速，且 T2I 指標追平 → 重新評估是否還需 latent 層。
