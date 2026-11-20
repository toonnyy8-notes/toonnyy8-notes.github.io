---
slug: psi-duo-curriculum-background
title: Duo 與 Gaussian 鬆弛課程學習：論文第 4 節的前置
cluster: Diffusion Language Models
kind: topic
tags: [duo, curriculum-learning, gaussian-relaxation, uniform-state-diffusion, top-k]
arxiv: ["2602.21185", "2506.10892", "2602.15014"]
sources: ["arxiv.org/abs 摘要（2506.10892）", "paper:2602.21185 第 1、2.1.2、4 節", "arXiv API metadata（標題、作者、年份）"]
related: [psi-samplers-lineage-hub, psi-uniform-state-guidance-and-self-correction, psi-superposition-posteriors-method]
---
# Duo 與 Gaussian 鬆弛課程學習

屬於 [Ψ-Samplers 脈絡總覽](psi-samplers-lineage-hub.html)。Duo 的完整介紹見 [The Diffusion Duality](diffusion-duality.html)，本篇只列第二章需要的部分。

## Duo（Sahoo 等人，2025；arXiv 2506.10892）
摘要：uniform-state 擴散過程由底層 Gaussian 擴散自然產生；Duo 把 Gaussian 擴散的技巧移植過來，包含以 Gaussian 過程引導的課程學習（訓練速度加倍），以及 Discrete Consistency Distillation（少步生成，取樣加速約兩個數量級）。

## 課程學習的成本（論文第 1、4 節）
課程以 softmax 後的 Gaussian 潛變數取代離散 token 作為 transformer 輸入，溫度為 $\tau=10^{-3}$。直接實作需要為每個 token 在每個訓練步驟建立 $K$ 維權重向量，詞表 $K>100{,}000$ 時不可行。論文的觀察是低溫下 softmax 幾乎只集中在少數項，因此只用 top-$k$ 個 embedding 近似，並以均勻亂數的 order statistics 直接取樣 top-$k$ 值，而不建立完整向量。論文報告相對 Duo 訓練時間減少 25%、記憶體減少 33%（摘要）。

## 同一作者群的相關後續
Sahoo 等人（2026，arXiv 2602.15014），*Scaling Beyond Masked Diffusion Language Models*，為論文引用的同系列工作；本筆記未讀其內容。
