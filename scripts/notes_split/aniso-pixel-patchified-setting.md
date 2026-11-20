---
slug: aniso-pixel-patchified-setting
title: Pixel 空間的 patchify 設定：JiT、PixelFlow 與 DiT
cluster: Diffusion & Generative Models
kind: topic
tags: [pixel-space, patchify, jit, dit, x-prediction]
arxiv: ["2611.17767", "2511.13720", "2504.07963", "2212.09748", "2601.21419", "2605.12964"]
sources: [2611.17767 introduction, 2611.17767 section 4, 2511.13720 abstract, 2504.07963 title, 2212.09748 title]
related: ["aniso-lineage-overview", "aniso-interpolated-targets-edm-kdiff-asymflow", "x-prediction-manifold-view", "pixel-space-diffusion", "aniso-target-normal-form-and-recovery"]
---
# Pixel 空間的 patchify 設定

回到[總覽](aniso-lineage-overview.html)。

## 本論文的設定
本論文的主要實驗在 pixel 空間，網路為 patchified transformer：每個 token 包含一個 patch 內的多個像素座標。W 作用在每個 patch 上並在空間位置間共用，因此資料維度 D 是 patch 維度；以 JiT 為例，D = 16 × 16 × 3 = 768（本論文 Sec. 4）。本論文指出，這種設定使 token 內出現資料方差差異很大的方向（引用 Li & He 2026 與 Chen et al. 2026a），因此不同方向可能適合不同的預測目標。

## 前驅工作
- **DiT**（Peebles & Xie，arXiv:2212.09748）：以 transformer 作為擴散模型的去噪網路。本論文在引言將其列為「可擴展的去噪網路」。
- **PixelFlow**（Chen 等，arXiv:2504.07963）：標題為 pixel 空間的 flow 生成模型；本論文僅列於參考文獻，本筆記只依標題引用。
- **JiT / Back to Basics**（Li & He，arXiv:2511.13720）：摘要主張直接預測乾淨資料，依流形假設，帶噪量不在低維流形上；使用大 patch（16、32）在 ImageNet 256 與 512 取得有競爭力的結果。本論文以 JiT-H/16 為主要基準，JiT-B/16 用於較小規模實驗。細節見 [x-prediction 與流形觀點](x-prediction-manifold-view.html) 與 [Pixel-space diffusion](pixel-space-diffusion.html)。
- 本論文引言將「pixel 模型使用乾淨資料目標」歸於 Li & He 2026 與 Guo et al. 2026，將「latent 模型使用速度目標」歸於 SD3、FLUX.2 等（描述來自本論文，未逐項查證）。

## 與本論文的關係
AnisoFlow 保留網路架構、flow-matching 損失與取樣器，只更換預測目標。Table 1 在固定 JiT-H/16 骨幹上比較：JiT（權重全為 0）在 600 epoch 的 FID 為 1.90，AnisoFlow 為 1.55。
