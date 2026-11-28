---
slug: revflow-lineage-overview
title: RevFlow（Generation as Reverse Encoding）的研究脈絡總覽
cluster: Diffusion & Generative Models
kind: hub
tags: [revflow, reverse-encoding, rae, flow-matching, dit, few-step]
sources: [paper-2611.34782-related-work, arxiv-abstracts]
arxiv: ["2611.34782", "2210.02747", "2212.09748", "2112.10752", "2410.06940", "2510.11690", "2505.13447"]
related: ["latent-diffusion-rae-overview", "representation-autoencoder-rae", "flow-matching-velocity-target", "revflow-flow-matching-dit-backbone", "revflow-generation-target-space", "revflow-representation-alignment-lineage", "revflow-rae-decoder-burden", "revflow-feature-inversion-multistage", "revflow-few-step-native-matching"]
---
# RevFlow 的研究脈絡總覽

RevFlow（arXiv 2611.34782）把影像生成寫成「反向編碼」：凍結的視覺編碼器（本文主模型為 DINOv3-L）逐層把影像轉成越來越抽象的表示；RevFlow 先由類別標籤生成深層表示，再生成淺層表示，最後解碼成像素。理解這篇論文需要六條脈絡，依下列順序閱讀。

1. [Flow matching 與 DiT 骨幹](revflow-flow-matching-dit-backbone.html)：RevFlow 用的訓練目標（`z_t = (1 - t)z + tε` 的 flow matching）與 DiT 架構。
2. [生成目標空間：像素與 latent](revflow-generation-target-space.html)：模型在哪個空間生成，以及這個選擇如何決定解碼器的負擔。
3. [以預訓練表示輔助生成：VA-VAE、REPA、REPA-E](revflow-representation-alignment-lineage.html)：把預訓練編碼器的語意帶進生成模型的第一批做法。
4. [RAE 與解碼器負擔](revflow-rae-decoder-burden.html)：直接在凍結編碼器的表示上生成，以及表示越抽象解碼器越難還原的問題。
5. [特徵反演與多階段生成](revflow-feature-inversion-multistage.html)：深層特徵損失外觀與空間細節的觀察，以及先生成緊湊表示再生成影像的既有方法。
6. [少步生成與原生分佈匹配](revflow-few-step-native-matching.html)：蒸餾、consistency、MeanFlow，以及以表示空間做分佈匹配的 FD-loss 等。

本論文與這些工作的差別（依論文 Related Work 所述）：既有方法只生成單一目標，其餘交給獨立訓練的解碼器；RevFlow 沿編碼器層級分段生成，各階段共用一個 DiT 骨幹。完整年表見同資料夾的 timeline.json。

已有的相關筆記：[Latent Diffusion 兩大主線總覽](latent-diffusion-rae-overview.html)、[RAE](representation-autoencoder-rae.html)、[Flow matching 的速度目標](flow-matching-velocity-target.html)。
