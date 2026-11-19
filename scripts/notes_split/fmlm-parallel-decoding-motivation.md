---
slug: fmlm-parallel-decoding-motivation
title: 自回歸的序列瓶頸與平行解碼的動機
cluster: Diffusion Language Models
kind: topic
tags: [autoregressive, non-autoregressive, parallel-decoding]
arxiv: ["2602.16813", "1711.02281", "2410.21035", "2505.22618"]
sources: [paper-2602.16813-intro-background, arxiv-abstracts]
related: ["fmlm-lineage-overview", "fmlm-discrete-diffusion-factorization"]
---
# 自回歸的序列瓶頸與平行解碼的動機

屬於 [Flow Map Language Models 的研究脈絡總覽](fmlm-lineage-overview.html)。

## 自回歸分解

本文背景一節把自回歸語言模型寫成 p(y) = p(y1) p(y2|y1) ... p(yL|y<L)，推論時每個 token 需要前面所有 token，生成過程因此是序列化的。訓練時可以用 teacher forcing 與 Transformer 平行處理，瓶頸只在取樣。本文引用的早期自回歸模型包括 Jordan（1986）、Elman（1990）、Bengio 等人（2003）。

## 非自回歸生成

Gu 等人的 *Non-Autoregressive Neural Machine Translation*（1711.02281，2017）是本文引用的非自回歸做法之一，目標是一次輸出整個序列以避免序列化。本文把這類方法視為「一次建模整個序列」的替代方案，動機是降低生成延遲。（說明依本文描述。）

## 擴散語言模型的承諾與問題

本文指出，離散擴散語言模型每步可平行輸出多個 token，因此有機會加速，但少步時品質快速下降。引用的證據是 Deschenaux 與 Gulcehre 的 *Beyond Autoregression: Fast LLMs via Self-Distillation Through Time*（2410.21035，2024）。Fast-dLLM（2505.22618，2025）與 ParallelBench 被本文引用為說明平行解碼取捨的工作。（以上依本文描述與標題；Fast-dLLM 標題為 Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding。）

本文要解決的就是這個少步品質問題，詳見 [離散擴散與因子化誤差](fmlm-discrete-diffusion-factorization.html)。
