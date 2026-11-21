---
slug: hit-latent-hierarchy-and-pixel-space
title: HiT 的出發點：潛空間的階層與 pixel-space 的固定 token 格點
cluster: Diffusion & Generative Models
kind: topic
tags: ["latent-diffusion", "pixel-space", "jit", "visual-hierarchy"]
arxiv: ["2112.10752", "2212.09748", "2410.06940", "2510.11690", "2301.11093", "2410.19324", "2504.07963", "2511.13720", "2611.22962"]
sources: ["2611.22962 Introduction、Related Work", "2112.10752 abstract", "2212.09748 abstract", "2410.06940 abstract", "2510.11690 abstract", "2511.13720 abstract"]
related: ["pixel-space-diffusion", "x-prediction-manifold-view", "latent-diffusion-rae-overview", "tokc-hourglass-and-latent-interfaces", "hit-hnet-blt-to-image-groups"]
---
# HiT 的出發點：潛空間的階層與 pixel-space 的固定 token 格點

## 論文如何描述兩條路線

HiT（2611.22962）的引言把潛空間擴散描述為「像素與精簡視覺表示之間有一層階層」：另外訓練的編碼器先把影像資訊整理成特徵空間，擴散模型再學習該空間的分布。論文引用的代表工作有 Latent Diffusion Models（Rombach 等，2112.10752）、DiT（Peebles 與 Xie，2212.09748）、REPA（Yu 等，2410.06940）與 Diffusion Transformers with Representation Autoencoders（Zheng 等，2510.11690）。論文指出這層階層的組織方式由編碼器架構與預訓練目標決定，發生在生成訓練之外。

Pixel-space 擴散則在單一階段端到端學習生成，不需要另外的視覺編碼器。論文列出的代表工作包括 simple diffusion（2301.11093）、Simpler Diffusion（2410.19324）、PixelFlow（2504.07963）與 PixelDiT，並以 JiT（Li 與 He，2511.13720）作為 ImageNet 的基線、以 MiniT2I 作為文字到影像的基線。JiT 的設定是預測乾淨影像（clean-image prediction）；MiniT2I 把同一設定與影像文字聯合注意力結合。

## 論文指出的缺口

論文的假設是：pixel-space Transformer 在整個骨幹中每個 patch 保留一個 token，沒有明確鼓勵「由細到粗」的視覺階層，這是它落後於雙階段潛空間模型的原因之一。論文在 Related Work 中描述這些 pixel-space 模型的共同點：架構在每一階段固定 image token 的數量與空間排列。HiT 的做法是在骨幹內部學出階層，因此不再需要外部預訓練的編碼器（Table 1 的 HiT 列在「Pre-training」欄為空）。

## 本站已有的背景筆記

- [Pixel-space diffusion：latent 層是否必要的第三路線](pixel-space-diffusion.html)：pixel-space 路線的整體介紹。
- [x-prediction 與流形觀點：JiT 與 pixel-space 的近期結果](x-prediction-manifold-view.html)：JiT 的 x-prediction 設定。
- [Latent Diffusion 兩大主線總覽：Diffusability 與 RAE](latent-diffusion-rae-overview.html)：REPA、RAE 一側的背景。
- [Hourglass 與 latent 介面：階層式 Transformer 的縮減與還原](tokc-hourglass-and-latent-interfaces.html)：先前在 Transformer 內部做階層的作法，本筆記不重複。

## 與本論文的關係

這條脈絡說明 HiT 要解決的問題（pixel-space 缺少階層），下一步是看階層如何在序列模型中形成：見 [H-Net 與 BLT：從動態切塊到影像分組](hit-hnet-blt-to-image-groups.html)。
