---
slug: fmlm-continuous-text-diffusion
title: 連續空間的文字擴散：從 Diffusion-LM 到 CDCD
cluster: Diffusion Language Models
kind: topic
tags: [continuous-diffusion-lm, embedding-diffusion, self-conditioning]
arxiv: ["2602.16813", "2205.14217", "2211.15089", "2208.04202", "2211.04236", "2305.18619", "2502.11564"]
sources: [paper-2602.16813-intro-background, arxiv-abstracts]
related: ["fmlm-lineage-overview", "fmlm-flow-matching-interpolant", "fmlm-onehot-denoiser-two-time", "cdlm-representation-space", "self-conditioning-flow-lm"]
---
# 連續空間的文字擴散：從 Diffusion-LM 到 CDCD

屬於 [Flow Map Language Models 的研究脈絡總覽](fmlm-lineage-overview.html)。

## 把 token 放進連續空間

本文第 3.1 節說明連續語言模型需要一個編碼 f 與解碼 g，使 g(f(y)) = y。既有選擇有兩類：

- **學習得到的 embedding**：Diffusion-LM（Li 等，2205.14217，2022）、CDCD（Dieleman 等，*Continuous diffusion for categorical data*，2211.15089，2022）、Gulrajani 與 Hashimoto 的 *Likelihood-Based Diffusion Language Models*（2305.18619，2023）。本文指出這類 embedding 需要正則化以避免崩塌或發散。
- **預訓練 embedding**：Strudel 等人的 *Self-conditioned Embedding Diffusion for Text Generation*（2211.04236，2022）。本文指出預訓練 embedding 對 flow 未必最佳。

Chen 等人的 *Analog Bits*（2208.04202，2022）把離散資料轉成類比位元後用擴散模型生成並加入 self-conditioning；本文把它視為類似 one-hot 的連續表示先例。

## 本文的選擇

本文使用逐 token one-hot 編碼加 argmax 解碼，無損、不需要正則化或輔助訓練。與 SSD-LM、TESS 等先前做法不同，本文不把擴散過程投影到 simplex，而在無約束的歐氏空間運作。

## 為何連續方法曾被認為較弱

本文指出普遍看法是連續擴散語言模型表現不如離散方法，並引用 The Diffusion Duality（2506.10892）與 CANDI（2510.22510）。後續的 Jo 與 Hwang *Continuous Diffusion Model for Language Modeling*（2502.11564，2025）屬於縮小差距的連續方法之一。本文的主張是這個看法可以被推翻。

表示空間的比較見 [連續擴散語言模型的表示空間與 rounding / decoder 問題](cdlm-representation-space.html)；self-conditioning 見 [self-conditioning 在連續 flow 語言模型中的作用](self-conditioning-flow-lm.html)。
