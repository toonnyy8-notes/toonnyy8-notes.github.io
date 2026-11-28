---
slug: revflow-generation-target-space
title: 生成目標空間：像素空間與重建式 latent
cluster: Diffusion & Generative Models
kind: topic
tags: [latent-diffusion, vae, vq-vae, pixel-space-diffusion, tokenizer]
arxiv: ["2611.34782", "2112.10752", "2511.13720", "2504.07963", "2507.23268", "2602.11401", "2610.12978"]
sources: [arxiv-abstracts, paper-2611.34782-related-work]
related: ["pixel-space-diffusion", "latent-diffusion-rae-overview", "revflow-rae-decoder-burden", "revflow-lineage-overview"]
---
# 生成目標空間

論文 Related Work 的第一段把既有做法分成兩類：在像素空間訓練（論文引用 Simple Diffusion、Back to Basics 等），或在以重建為目標訓練的 tokenizer latent 中訓練（VQGAN、VAE、Latent Diffusion、VQ-VAE）。

- 2021，High-Resolution Image Synthesis with Latent Diffusion Models（Rombach 等，2112.10752）：在預訓練自編碼器的 latent 空間做擴散（依摘要）。
- 2025，Back to Basics: Let Denoising Generative Models Denoise（Li、He，2511.13720）：論文將其歸為像素空間生成的近期工作。
- 2025，PixelFlow（2504.07963）、PixNerd（2507.23268）：論文在比較表中列為像素空間生成模型。
- 2026，Latent Forcing（Baade 等，2602.11401）：論文描述為在像素之前先對特徵去噪。
- 2026，Learning Representations to Guide Pixel Diffusion Transformers（Pan 等，2610.12978）：論文與 RevFlow 作者群有重疊，歸入像素空間且利用表示的做法。

RevFlow 的前提是：重建式 latent 不含語意結構；而改用預訓練表示時，目標越抽象，從表示回到像素的解碼越難。這個問題在 [RAE 與解碼器負擔](revflow-rae-decoder-burden.html) 說明。

像素空間與 latent 空間之爭的完整整理見 [Pixel-space diffusion](pixel-space-diffusion.html) 與 [Latent Diffusion 總覽](latent-diffusion-rae-overview.html)。
