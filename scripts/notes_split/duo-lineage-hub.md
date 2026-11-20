---
slug: duo-lineage-hub
title: Duo（The Diffusion Duality）的研究脈絡總覽
cluster: Diffusion Language Models
kind: hub
tags: [duo, uniform-state-diffusion, gaussian-diffusion, hub]
arxiv: ["2506.10892", "1503.03585", "2006.11239", "2107.03006", "2310.16834", "2406.07524", "2412.10193", "2303.01469", "2410.21035", "2205.14987", "2407.21243", "2503.00307", "1611.01144", "1611.00712"]
sources: ["arXiv:2506.10892 摘要、§1、§2、§3、§4、§6", "各引用論文的 arXiv 摘要（見各子筆記）"]
related: ["duo-gaussian-diffusion-foundations", "duo-discrete-diffusion-masked-and-uniform", "duo-argmax-duality-theory", "duo-curriculum-softmax-relaxation", "duo-consistency-distillation-to-discrete", "duo-self-correction-and-correctors", "dllm-three-corruption-paradigms"]
---
# Duo（The Diffusion Duality）的研究脈絡總覽

The Diffusion Duality（arXiv 2506.10892，方法名 Duo）處理 Uniform-state diffusion model（USDM）：每個 token 在前向過程中以某機率被替換成詞表內任一 token。論文指出 USDM 在反向取樣時可以持續更新 token，具有自我修正的性質，但歷來的困惑度（perplexity）落後自迴歸（AR）模型與遮罩式擴散模型（MDM）。論文的核心觀察是：對 one-hot 向量上的 Gaussian diffusion 施加 arg max，邊際分佈恰好是 USDM 的邊際分佈。因此 Gaussian diffusion 的訓練與取樣技巧可以移植到 USDM。

建議閱讀順序：

1. [Gaussian diffusion 的基礎：從 DDPM 到 PF-ODE](duo-gaussian-diffusion-foundations.html)：本文要借用技巧的來源。
2. [離散擴散：遮罩式與均勻狀態](duo-discrete-diffusion-masked-and-uniform.html)：D3PM、SEDD、MDLM、UDLM 的設定與本文的記號。
3. [arg max 對偶的理論](duo-argmax-duality-theory.html)：Diffusion Transformation 運算子與兩個定理。
4. [課程學習與 softmax 溫度鬆弛](duo-curriculum-softmax-relaxation.html)：降低訓練變異的做法與其前人。
5. [從 consistency distillation 到 Discrete Consistency Distillation](duo-consistency-distillation-to-discrete.html)：少步取樣。
6. [自我修正與 predictor-corrector](duo-self-correction-and-correctors.html)：為何選 USDM 而非 MDM。

站內已有的相關筆記：[三種 corruption 與 dLLM 的發展](dllm-three-corruption-paradigms.html)（整理 masked、uniform、Gaussian 三類，並引用本文）。

年表見 timeline.json。本文的實驗設定為 170M 參數的 DiT，於 LM1B 與 OpenWebText 訓練 1M 步；論文報告課程學習使訓練速度約為 2 倍，並將取樣步數由 1024 降至 8 步（以 Greedy-Tail 取樣器）。
