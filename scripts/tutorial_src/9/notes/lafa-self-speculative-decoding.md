---
slug: lafa-self-speculative-decoding
title: 自我推測解碼：flow 草稿與 AR 驗證
cluster: Diffusion Language Models
kind: topic
tags: [speculative-decoding, multi-token-prediction, ar-verification, throughput]
arxiv: ["2610.28559", "1811.03115", "2401.10774", "2404.19737", "2604.07023"]
sources: ["paper.txt §1, §2.3, §4", "arXiv abstracts"]
related: ["lafa-lineage-overview", "lafa-parallel-decoding-background", "lafa-asynchronous-tokenwise-time", "lafa-scale-and-evaluation"]
---
# 自我推測解碼

屬於 [Lafa 研究脈絡總覽](lafa-lineage-overview.html)。

## 構成
1. 訓練時，block 的去噪預測與前綴的 AR 預測由同一組 Transformer 參數產生，損失同時包含 AR 的 cross entropy 與去噪 cross entropy（paper.txt 式 6）。
2. 論文報告這種聯合訓練保留甚至提升預訓練 AR 的能力，並稱其與 multi-token prediction 的效果一致（引用 [6, 8, 34]，對應 [Better & Faster LLMs via Multi-token Prediction](lafa-parallel-decoding-background.html) 等）。
3. 推論時，flow 產生整個 block 的草稿，同一模型的 AR 預測由左到右驗證，被接受的 token 才寫入 KV cache。
4. 非同步時間使左側 token 先成形，與由左到右的驗證順序一致（見 [非同步時間](lafa-asynchronous-tokenwise-time.html)）。

## 與先前方法的差別
Blockwise Parallel Decoding、Medusa、MARS 等方法需要額外的 proposal 模組或 head，或直接把 AR 模型微調為預測多個 token（依各摘要與論文 Related work）。Lafa 的草稿與驗證由同一個模型完成，不增加獨立模組。

## 結果
論文摘要報告 7B 模型取得 AR 的 2.54 倍吞吐量，品質相當；Figure 1 標示範圍為 2.02–2.54 倍。結果的比較條件見 [規模、資料量與評估](lafa-scale-and-evaluation.html)。
