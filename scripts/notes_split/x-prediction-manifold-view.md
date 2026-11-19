---
slug: x-prediction-manifold-view
title: x-prediction 與流形觀點：JiT 與 pixel-space 的近期結果
cluster: Diffusion & Generative Models
kind: topic
tags: [x-prediction, manifold-hypothesis, jit, pixel-space]
arxiv: ["2511.13720", "2410.19324", "2509.24527", "2601.21419"]
sources: [2601.21419, abstract]
related: ["prediction-targets-overview", "loss-weighting-and-elbo", "generalized-target-k-parameterization", "pixel-space-diffusion"]
---
# x-prediction 與流形觀點

回到 [總覽](prediction-targets-overview.html)。

## Back to Basics（Li & He 2025，arXiv:2511.13720）
摘要：現今擴散模型的網路多預測噪聲或帶噪量，並非預測乾淨影像；依流形假設，自然資料位於低維流形，帶噪量則不然；因此主張直接預測乾淨資料，使「看似容量不足」的網路能在高維空間運作；提出 JiT，在 ImageNet 256 與 512 用大 patch（16、32）取得有競爭力的結果，並指出此時預測高維帶噪量可能「災難性失敗」。

## 其他 x-prediction 結果
本論文引言以 [7, 8, 9]（SiD2、JiT、Hafner 等人 2025 *Training Agents Inside of Scalable World Models*，arXiv:2509.24527）作為 x-prediction 在高解析度表現較佳的例證；本筆記僅依本論文的描述引用第三者，未逐項查證其內容。

## 本論文補上的部分
本論文說明，先前以流形假設為 x-prediction 辯護的說法屬「定性」，缺少定量公式；它在[廣義目標](generalized-target-k-parameterization.html)中給出 d、D 與最佳 k 的關係。pixel-space 各方法的比較見[Pixel-space diffusion](pixel-space-diffusion.html)；SiD2 的加權見[Loss weighting 與 ELBO](loss-weighting-and-elbo.html)。
