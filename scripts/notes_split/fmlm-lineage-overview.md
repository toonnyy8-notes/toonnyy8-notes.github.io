---
slug: fmlm-lineage-overview
title: Flow Map Language Models 的研究脈絡總覽
cluster: Diffusion Language Models
kind: hub
tags: [flow-map, language-model, lineage, overview]
arxiv: ["2602.16813", "1711.02281", "2107.03006", "2406.07524", "2506.10892", "2205.14217", "2211.15089", "2210.02747", "2303.01469", "2406.07507", "2505.18825"]
sources: [paper-2602.16813-intro-background, arxiv-abstracts]
related: [lafa-parallel-decoding-background, fmlm-discrete-diffusion-factorization, fmlm-continuous-text-diffusion, fmlm-flow-matching-interpolant, fmlm-flow-map-distillation-lineage, fmlm-onehot-denoiser-two-time, continuous-flow-lm-overview, flow-map-few-step-distillation]
---
# Flow Map Language Models 的研究脈絡總覽

Flow Map Language Models（FMLM，Lee 等人，2602.16813）主張：在 one-hot token 表示上做連續 flow，再蒸餾成 flow map，可以一步生成文字，品質超過近期少步離散擴散語言模型的 8 步結果。理解這個主張需要依序認識六個主題，以下依閱讀順序列出。

1. [自回歸的序列瓶頸與平行解碼](lafa-parallel-decoding-background.html)：為什麼要找自回歸以外的生成方式，以及非自回歸方法早期的品質問題。
2. [離散擴散與因子化誤差](fmlm-discrete-diffusion-factorization.html)：D3PM、masked diffusion 等方法如何平行生成，以及它們在少步時品質下降的原因。
3. [連續空間的文字擴散](fmlm-continuous-text-diffusion.html)：Diffusion-LM、CDCD 等把 token 映射到連續空間的做法，以及它們為何被認為不及離散方法。
4. [Flow matching 與 interpolant](fmlm-flow-matching-interpolant.html)：本文使用的連續生成框架，速度場與 denoiser 的關係。
5. [Flow map 與少步蒸餾](fmlm-flow-map-distillation-lineage.html)：consistency model、shortcut model、MeanFlow、flow map matching 的關係。
6. [FMLM 的設計：one-hot、CE denoiser 與 two-time denoiser](fmlm-onehot-denoiser-two-time.html)：把前五條脈絡接在一起的本文方法。

資料庫中已有的相關筆記：[2026 年連續 flow 語言模型總覽](continuous-flow-lm-overview.html) 整理同期各論文；[連續 flow 語言模型的少步生成](flow-map-few-step-distillation.html) 比較各種蒸餾方法。本組筆記只補足 FMLM 的前置脈絡，不重複那兩篇。

年代順序見 timeline.json。
