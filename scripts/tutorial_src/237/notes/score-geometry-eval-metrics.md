---
slug: score-geometry-eval-metrics
title: Stable Diffusion 實驗的評估指標：P-sim 與 I-sim
cluster: Diffusion & Generative Models
kind: topic
tags: [clip, clipscore, evaluation, diversity]
arxiv: ["2104.08718", "2509.24912"]
sources: [2509.24912 section 7.2, 2104.08718 abstract, user_terms_note.md]
related: ["score-geometry-lineage-hub", "score-geometry-inverse-problems-guidance"]
---
# 評估指標

回到 [總覽](score-geometry-lineage-hub.html)。

兩個指標都由 CLIP 嵌入的餘弦相似度算出。CLIP 是把圖像與文字映射到同一嵌入空間的對比式模型。CLIPScore（Hessel et al. 2021，arXiv 2104.08718）是無參考的圖像描述評估指標，以圖像嵌入與文字嵌入的餘弦相似度為基礎。

- **P-sim**（CLIP Prompt Similarity）：生成圖與提示詞的平均 CLIP 分數；越高越好（品質）。
- **I-sim**（CLIP Inter-Image Similarity）：同一提示詞下所有生成圖兩兩的平均 CLIP 相似度；越低越多樣。

## 實驗設定

引導強度 w = 7.5、30 步推論、每組 512 張圖、縮到 256 × 256 再算分數；提示詞為 Creative furniture、An innovative car design、A creative architecture。

差異幅度不大（例如家具類 P-sim 29.40 → 30.20），論文未提供重複實驗的誤差範圍，作者自稱為初步實驗。

方法與引導的對應見 [貝氏逆問題與 guidance](score-geometry-inverse-problems-guidance.html)。
