---
slug: lafa-embedding-flow-denoiser
title: 以詞嵌入為空間的 flow 與 clean-token 去噪器（CDCD 到 Lafa）
cluster: Diffusion Language Models
kind: topic
tags: [flow-matching, embedding-flow, clean-token-denoiser, continuous-diffusion-lm]
arxiv: ["2610.28559", "2210.02747", "2209.15571", "2211.15089", "2602.16813", "2605.10938", "2604.11748", "2605.07013"]
sources: ["arXiv abstracts", "paper.txt §2.1, §4"]
related: [lafa-lineage-overview, lafa-asynchronous-tokenwise-time, lafa-self-conditioning-and-flow-map, flow-matching-velocity-target, cdlm-training-objective, cdlm-representation-space, fmlm-continuous-text-diffusion]
---
# 以詞嵌入為空間的 flow 與 clean-token 去噪器

屬於 [Lafa 研究脈絡總覽](lafa-lineage-overview.html)。

## 基礎
Lafa 以 Flow Matching（Lipman et al., 2022，2210.02747）與 Stochastic Interpolants（Albergo et al.，論文引用）為基礎，內插 `I_t = (1-t)x0 + t x1` 屬此類。基礎與速度目標見 [Flow matching、interpolant 與速度場／denoiser 的關係](fmlm-flow-matching-interpolant.html) 與 [Flow matching 的速度目標](flow-matching-velocity-target.html)。

## 連續語言模型
- CDCD（Dieleman et al., 2022，2211.15089）：以連續擴散處理類別資料，Lafa 沿用其 time warping。背景見 [連續空間的文字擴散](fmlm-continuous-text-diffusion.html)。
- Flow Map Language Models（Lee et al., 2026，2602.16813）：摘要稱以連續 flow 處理 one-hot 表示，可做一步生成；Lafa 將其兩時間去噪器推廣到非同步 flow。
- ELF（2605.10938）、LangFlow（2604.11748）、CoBit（2605.07013）：論文列為同期連續語言模型，後者以位元碼取代整個詞彙表的預測（依 CoBit 摘要）。

## Lafa 的設計（paper.txt §2.1）
- 使用詞彙嵌入矩陣 E（維度 d）把 token 嵌入，避免 one-hot 的 |V| 維度。
- 模型輸出是 clean token 的機率（tokenwise classification），速度場由去噪器與目前狀態的差推得（論文式 5，證明在 Appendix A.1）。
- 訓練目標與嵌入規範化的細節見 [連續擴散語言模型的訓練目標與參數化](cdlm-training-objective.html) 與 [表示空間與 rounding 問題](cdlm-representation-space.html)。
