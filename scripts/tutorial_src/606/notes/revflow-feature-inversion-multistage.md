---
slug: revflow-feature-inversion-multistage
title: 特徵反演與多階段、表示條件式生成
cluster: Diffusion & Generative Models
kind: topic
tags: [feature-inversion, unclip, multi-stage-generation, latent-forcing, coarse-to-fine]
arxiv: ["2611.34782", "1412.0035", "2204.06125", "2602.11401", "2403.05135"]
sources: [arxiv-abstracts, paper-2611.34782-related-work]
related: ["revflow-rae-decoder-burden", "revflow-generation-target-space", "revflow-lineage-overview"]
---
# 特徵反演與多階段生成

特徵反演。Mahendran 與 Vedaldi，Understanding Deep Image Representations by Inverting Them（1412.0035，2014）從表示重建影像以分析表示保留的資訊。論文引用此類工作（連同 Achille 與 Soatto 2018）說明：越深的層越不變，保留語意而損失外觀與空間細節；預訓練 ViT 也呈現類似趨勢。論文的 ImageNet 檢索例子顯示淺層特徵對應顏色與紋理，深層特徵較常檢索到同類別影像。編碼是多對一，所以單純的確定性回歸（自編碼器、RAE 解碼器）無法還原生成所需的多樣性；RevFlow 對每個反演步驟做生成式建模，只在保留細節的淺層使用確定性像素解碼。

多階段與表示條件式生成（依論文描述）：
- 粗到細（解析度分解）：Matryoshka diffusion、Cascaded diffusion、Progressive GAN、VQ-VAE-2、VAR。
- 先生成緊湊表示：unCLIP（Ramesh 等，2204.06125）取樣 CLIP 嵌入；RCG 生成自監督表示；Würstchen 生成壓縮語意 latent；Latent Forcing（2602.11401）在像素前先對特徵去噪。

論文對差異的說明：上述方法中表示是另一空間的條件；RevFlow 的各階段目標是同一個凍結編碼器的成對特徵，每階段反向一段編碼路徑，且共用同一骨幹。空間條件採用 SPADE 式的逐 token adaLN，而非 ControlNet 式的附加分支。
