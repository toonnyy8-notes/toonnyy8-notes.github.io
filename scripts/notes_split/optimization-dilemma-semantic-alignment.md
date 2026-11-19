---
slug: optimization-dilemma-semantic-alignment
title: Optimization dilemma 與語意／空間對齊：VA-VAE、REPA、MAETok、iREPA
cluster: Diffusion & Generative Models
kind: topic
tags: [optimization-dilemma, repa, va-vae, maetok, irepa, semantic-separability, spatial-structure]
sources: [latent-diffusion-rae.md]
arxiv: ["2501.01423", "2410.06940", "2502.03444", "2512.10794", "2606.03578"]
related: ["latent-diffusion-rae-overview", "cdlm-architecture-recipes", "pixel-space-diffusion", "diffusability-systematic-study"]
---
# Optimization dilemma 與語意／空間對齊：VA-VAE、REPA、MAETok、iREPA

屬於 [Latent Diffusion 總覽](latent-diffusion-rae-overview.html)。

## 錯配的來源（optimization dilemma）
VA-VAE/LightningDiT（CVPR 2025 Oral，arXiv:2501.01423）首次系統化命名「optimization dilemma」——tokenizer 維度（d）上升會改善 rFID 但惡化 gFID，因為高維無約束 latent 難以被 diffusion 學習。VA-VAE 的 Table 2 顯示：固定 compute budget 下，隨 tokenizer 維度上升 rFID 下降但 gFID 上升。作者論證這源自「學習無約束高維 latent space 的內在困難」。

解法是在訓練 tokenizer 時就把 latent 對齊到 pre-trained vision foundation model。VF Loss 由兩項組成：marginal cosine similarity（L_mcos）與 marginal distance matrix similarity（L_mdms），並用 adaptive weighting 平衡與 reconstruction/KL loss。整合 LightningDiT 後 ImageNet-256 FID 1.35（rFID 0.28），64 epoch 達 2.11（>21× 加速，官方 repo 稱 21.8×）。限制：VF Loss 過強會壓縮容量。

## (a) 語意可分性 (semantic separability)
- **REPA（arXiv:2410.06940，ICLR 2025 Oral）**：對齊 noisy hidden state 投影與 clean DINOv2 表徵，正則化係數 λ≈0.5 後飽和；SiT-XL/2 FID 1.42（w/ CFG + guidance interval），訓練加速 >17.5×（<400K steps 匹配 7M steps 表現）。限制：需選 target encoder；不改 latent 本身。（REPA 也被用於語言模型 ELF-REG，見 [連續擴散 LM 架構配方](cdlm-architecture-recipes.html)。）
- **VA-VAE**：把語意先驗「烘焙」進 tokenizer 而非 diffusion 中間層。
- **MAETok（arXiv:2502.03444，ICML 2025 Spotlight）**：以 mask autoencoding 學到「較少 GMM modes、較有判別性」的 latent；理論與實證都指向「latent 結構（而非 variational 約束）」才是關鍵；僅用 128 tokens 即達 SOTA、512² gFID 1.69、76× 訓練加速、31× 推論吞吐。限制：AE 無 variational 保證。

## (b) 空間結構 (spatial structure) — iREPA（arXiv:2512.10794，ICLR 2026）
- 在 27 個 vision encoder 上做大規模分析，發現驅動生成品質的是**空間結構（patch token 間的 pairwise cosine similarity）而非全域語意（ImageNet 準確率）**。
- 三行程式碼改動：spatial regularization layer + 用 conv 取代 MLP projection（MLP 會丟失空間資訊）。 [End2end-diffusion](https://end2end-diffusion.github.io/irepa/)
- 指標：LDS（近鄰是否比遠處更相似）、CDS（相似度隨距離衰減率）、SRSS（前景遮罩下 intra-foreground 一致性）、RMSC（空間多樣性，被 Zhong et al. 排除）。 [arxiv](https://arxiv.org/pdf/2606.03578)
- 加速隨 encoder 增大而增大：PE-B 22.2%、PE-L 38.8%、PE-G 39.6%。 [arxiv](https://arxiv.org/pdf/2512.10794)
- iREPA 的空間結構技巧也能加速 pixel-space JiT（見 [pixel-space 第三路線](pixel-space-diffusion.html)）。

這兩類觀點在 [系統性比較研究](diffusability-systematic-study.html) 中被判定為穩定預測子（LNC、SRSS）。
