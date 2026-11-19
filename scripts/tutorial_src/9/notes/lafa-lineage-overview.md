---
slug: lafa-lineage-overview
title: Lafa（Large Flow Language Models）的研究脈絡總覽
cluster: Diffusion Language Models
kind: hub
tags: [flow-language-model, block-autoregressive, self-speculative-decoding, lineage]
arxiv: ["2610.28559", "2211.15089", "2602.16813", "2503.09573", "2404.19737"]
sources: ["paper.txt §1, §4", "arXiv abstracts"]
related: ["lafa-parallel-decoding-background", "lafa-ar-to-block-adaptation", "lafa-embedding-flow-denoiser", "lafa-asynchronous-tokenwise-time", "lafa-self-conditioning-and-flow-map", "lafa-self-speculative-decoding", "lafa-scale-and-evaluation", "continuous-flow-lm-overview", "fmlm-lineage-overview"]
---
# Lafa 的研究脈絡總覽

Lafa（arXiv 2610.28559）是擴展到 8B 參數的連續 flow 語言模型。它在每個 block 內做雙向、各位置時間不同步的 flow，block 之間維持因果注意力，並由預訓練的自回歸（AR）模型調整而來。

論文把相關工作分成三類：block-autoregressive 模型、離散擴散模型、連續擴散與 flow 模型。理解 Lafa 需要依序掌握下列主題，每個主題一篇筆記。

1. [AR 生成的延遲與平行解碼方法](lafa-parallel-decoding-background.html)：問題的來源，以及 blockwise parallel decoding、multi-token prediction 等不改變模型型態的加速法。
2. [從預訓練 AR 模型調整為 block 模型](lafa-ar-to-block-adaptation.html)：block diffusion 與 AR 轉 diffusion 的調整路線。
3. [以詞嵌入為空間的 flow 與 clean-token 去噪器](lafa-embedding-flow-denoiser.html)：flow matching、CDCD、Flow Map Language Models、ELF 等連續語言模型。
4. [block 內各位置的非同步時間](lafa-asynchronous-tokenwise-time.html)：tokenwise time 與學到的時間排程。
5. [self-conditioning 與 flow map 蒸餾](lafa-self-conditioning-and-flow-map.html)：提升取樣品質與減少步數的兩個技術。
6. [自我推測解碼](lafa-self-speculative-decoding.html)：以同一模型的 flow 草稿與 AR 驗證取得吞吐量。
7. [規模、資料量與評估](lafa-scale-and-evaluation.html)：與其他模型的比較方式與評估上的限制。

資料庫中已有的相關筆記：[2026 年連續 flow 語言模型總覽](continuous-flow-lm-overview.html) 整理同期連續語言模型的比較；[Flow Map Language Models 的研究脈絡總覽](fmlm-lineage-overview.html) 說明 Lafa 所延伸的前作。
