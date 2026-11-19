---
slug: ttt-e2e
title: TTT-E2E：以 next-token prediction 做測試時學習、以 meta-learning 學初始化
cluster: Architectures & Efficiency
kind: topic
tags: [ttt-e2e, test-time-training, meta-learning, long-context, sliding-window-attention, continual-learning]
sources: [ttt-sequence-modeling.md]
arxiv: ["2512.23675"]
related: ["ttt-sequence-modeling-overview"]
---
# TTT-E2E：以 next-token prediction 做測試時學習、以 meta-learning 學初始化

arXiv:2512.23675，原 TTT 團隊的直系新作。屬於 [TTT 序列建模總覽](ttt-sequence-modeling-overview.html)。

**作者與狀態。** Arnuv Tandon、Karan Dalal、Xinhao Li、Daniel Koceja、Marcel Rød、Sam Buchanan、Xiaolong Wang、Jure Leskovec、Sanmi Koyejo、Tatsunori Hashimoto、Carlos Guestrin、Jed McCaleb、Yejin Choi、Yu Sun（Astera Institute、NVIDIA、Stanford、UC Berkeley、UC San Diego），2025-12-29 preprint；Yu Sun 為 project lead。code：github.com/test-time-training/e2e。

## 設計理念與「End-to-End」的兩層含義
把長上下文建模從「架構設計」重構為「持續學習」。
- **測試時 E2E**：內層直接優化網路尾端的 next-token prediction（交叉熵）損失 `ℓ_t(W)=CE(f(x_{t-1};W), x_t)`，而非層內重建代理損失。
- **訓練時 E2E**：用 meta-learning 學 fast weight 的初始化 `W_0`——把每條訓練序列當測試序列，對它做 TTT，再對「TTT 之後的最終損失」用 gradients-of-gradients 優化 `W_0`。

這與主流 dynamic evaluation 的 TTT-naive（`L=⅟T Σ ℓ_t(W_0)`）形成對比；TTT-naive 僅略勝無 attention baseline，TTT-E2E 幾乎追平 full attention。

## 架構與更新
標準 Transformer + sliding-window attention 當「工作記憶/短期記憶」，被更新的權重當「長期記憶」。三個實作要點：（1）只 TTT MLP 層（凍結 embedding/norm/attention，否則外層不穩）；（2）只更新最後 1/4 的 block；（3）更新 block 內加一個靜態第二 MLP 當「安全」儲存以抗遺忘。mini-batch 更新：`W_i = W_{i-1} − η·(1/b)·Σ_t ∇ℓ_t(W_{i-1})`。主結果用 window k=8K、mini-batch b=1K（需 k≥b，使模型在 TTT 更新前仍能在 mini-batch 內記住 context）。

## 實驗結果
- 3B 模型、164B token：context 擴展行為與 full attention 一致，而 Mamba 2、Gated DeltaNet 不行。
- 推理延遲恆定，abstract 載明「TTT-E2E has constant inference latency regardless of context length, making it 2.7× faster than full attention for 128K context」（H100）。
- hidden state「5x larger hidden state (88M vs. 18M for the 760M model)」。
- 但在 Needle-in-Haystack（RULER 128K）上 full attention 大幅領先——因 TTT-E2E 的機制是壓縮而非無損 recall。
- 訓練延遲是已知弱點：128K 時比 full attention 快 1.2×，但 8K 時慢 3.4×（雙重反傳未被 cuDNN 最佳化）。

## 注意事項
「TTT-E2E 在 2M context 快 35×」出現在第三方部落格，arXiv v2 正文只證實「128K 快 2.7×、延遲恆定」；此類超大倍率宣稱應視為未經原文證實。

## 使用門檻
若資料是 1D 語言且追求 SOTA recall，優先考慮 full attention 或 hybrid；若 context >128K 且延遲/吞吐是瓶頸，TTT-E2E 的恆定延遲（2.7× @128K）開始划算。
