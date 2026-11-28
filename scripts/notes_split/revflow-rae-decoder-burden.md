---
slug: revflow-rae-decoder-burden
title: RAE 與解碼器負擔：RevFlow 要解決的問題
cluster: Diffusion & Generative Models
kind: topic
tags: [rae, raev2, dinov2, dinov3, decoder, representation-autoencoder]
arxiv: ["2611.34782", "2510.11690", "2605.18324", "2304.07193", "2508.10104", "2606.24888"]
sources: [arxiv-abstracts, paper-2611.34782-abstract, paper-2611.34782-related-work]
related: ["representation-autoencoder-rae", "rae-followups", "revflow-generation-target-space", "revflow-lineage-overview"]
---
# RAE 與解碼器負擔

- 2023，DINOv2（Oquab 等，2304.07193）、2025，DINOv3（Siméoni 等，2508.10104）：自監督視覺編碼器，是 RAE 與 RevFlow 的凍結編碼器。
- 2025，Diffusion Transformers with Representation Autoencoders（Zheng 等，2510.11690）：RAE。依論文描述，RAE 直接生成編碼器最後一層特徵，由獨立訓練的解碼器還原像素。
- 2026，Improved Baselines with Representation Autoencoders（Singh 等，2605.18324）：RAEv2，依論文描述把多層特徵融合成一個 latent。
- 2026，DiffusionBench（Leng 等，2606.24888）：論文引用的評估基準。

RevFlow 摘要指出的問題：預訓練表示抽象掉外觀與空間細節，使「表示到像素」的路徑更難，把更多負擔推給較大的解碼器。論文以數字對照：RevFlow 以 89M 解碼器得到 30.25 dB 重建 PSNR，RAE 的 415M 解碼器為 18.09 dB（皆為論文所報）。

RevFlow 的回應：既然編碼器已提供像素到語意的路徑，就沿該路徑反向分階段生成（類別 → 深層表示 → 淺層表示 → 像素），而不是獨立學習單一的表示到像素映射。詳見 [RAE](representation-autoencoder-rae.html) 與 [RAE 後續改進](rae-followups.html)。
