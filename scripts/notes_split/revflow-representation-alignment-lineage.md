---
slug: revflow-representation-alignment-lineage
title: 以預訓練表示輔助生成：VA-VAE、REPA、REPA-E
cluster: Diffusion & Generative Models
kind: topic
tags: [repa, repa-e, va-vae, representation-alignment, optimization-dilemma]
arxiv: ["2611.34782", "2501.01423", "2410.06940", "2504.10483", "2304.07193", "2508.10104"]
sources: [arxiv-abstracts, paper-2611.34782-related-work]
related: ["optimization-dilemma-semantic-alignment", "revflow-rae-decoder-burden", "revflow-lineage-overview"]
---
# 以預訓練表示輔助生成

論文 Related Work 描述：VA-VAE 與 REPA 把 VAE latent 或擴散模型的隱藏狀態對齊到凍結編碼器的特徵，REPA-E 端到端微調 VAE，其他 tokenizer 則為生成而設計。這些方法都只生成單一目標空間。

| 年份 | 工作 | arXiv | 依摘要的內容 |
| --- | --- | --- | --- |
| 2024 | REPA（Yu 等） | 2410.06940 | 把擴散 Transformer 的隱藏狀態對齊到外部預訓練視覺表示，加速訓練。 |
| 2025 | VA-VAE：Reconstruction vs. Generation（Yao 等） | 2501.01423 | 指出 latent 擴散中重建與生成的最佳化兩難，以視覺基礎模型對齊 latent。 |
| 2025 | REPA-E（Leng 等） | 2504.10483 | 以表示對齊損失端到端訓練 VAE 與擴散 Transformer。 |

編碼器 DINOv2（Oquab 等，2304.07193，2023）是這些對齊目標常用的特徵來源；DINOv3 見 [RAE 與解碼器負擔](revflow-rae-decoder-burden.html)。

與 RevFlow 的關係：REPA 類方法只把表示當作輔助訊號，生成目標仍是 latent；RevFlow 與 RAE 則直接把編碼器表示當作生成目標。詳細說明見 [Optimization dilemma 與語意／空間對齊](optimization-dilemma-semantic-alignment.html)。
