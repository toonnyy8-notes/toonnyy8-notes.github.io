---
slug: lafa-ar-to-block-adaptation
title: 從預訓練 AR 模型調整為 block 模型（Block Diffusion 與 AR 轉 diffusion）
cluster: Diffusion Language Models
kind: topic
tags: [block-diffusion, ar-adaptation, prefix-cache, block-bidirectional-attention]
arxiv: ["2610.28559", "2503.09573", "2508.15487", "2512.14067", "2512.06776", "2512.22737"]
sources: ["arXiv abstracts", "paper.txt §2, §4"]
related: ["lafa-lineage-overview", "lafa-embedding-flow-denoiser", "lafa-parallel-decoding-background", "semi-ar-block-generation", "dllm-llada-dream-models"]
---
# 從預訓練 AR 模型調整為 block 模型

屬於 [Lafa 研究脈絡總覽](lafa-lineage-overview.html)。

## Block 分解
Lafa 採用標準的 block-autoregressive 分解：長度 M 的序列切成大小 B 的 block，每個 block 的機率以前面所有 block 為條件。論文把這個分解的來源追溯到 semi-autoregressive 機器翻譯，並以 Block Diffusion（Arriola et al., 2025，arXiv 2503.09573）作為把 block 內去噪與跨 block 因果生成結合的代表。其摘要說明該方法在 AR 與 diffusion 之間插值，並支援可變長度生成。Semi-AR 區塊生成的細節見 [Semi-AR 區塊生成](semi-ar-block-generation.html)。

## 從 AR 模型調整
訓練大型擴散模型的成本高，因此一組工作改由預訓練 AR 權重出發：

- Dream 7B（2508.15487）：論文將其列為擴展到數十億參數的離散擴散模型（見 [LLaDA 與 Dream](dllm-llada-dream-models.html)）。
- Efficient-DLM（2512.14067）：摘要稱研究 AR 轉 dLM 的轉換，因為從頭訓練的學習效率落後於 AR。
- From Next-Token to Next-Block（2512.06776）：摘要稱提出由 AR 權重調整為 block 擴散模型的方法。
- WeDLM（2512.22737）：摘要稱使其與標準因果注意力相容以利快速推論。

## Lafa 的選擇（paper.txt §2.2）
- 區塊內雙向注意力，對乾淨前綴使用因果注意力，前綴的 key / value 可在去噪步驟間重用。
- 時間條件以 adaLN 加入，並初始化為恆等，使調整初期不改變預訓練模型。
- 論文的 Table 3 顯示 block 內雙向注意力優於因果注意力。
