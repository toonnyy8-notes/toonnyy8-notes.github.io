---
slug: psi-samplers-lineage-hub
title: Ψ-Samplers 的研究脈絡總覽：從離散擴散到 predictor-corrector 採樣器
cluster: Diffusion Language Models
kind: hub
tags: [psi-samplers, duo, predictor-corrector, uniform-state-diffusion, lineage]
arxiv: ["2602.21185", "2506.10892", "2503.00307", "2412.10193", "2205.14987"]
sources: ["paper:2602.21185 第 1、2、6 節與參考文獻", "arXiv API metadata（標題、作者、年份）"]
related: [psi-discrete-diffusion-interpolating-process, psi-ctmc-discrete-diffusion-samplers, psi-predictor-corrector-gaussian-to-discrete, psi-uniform-state-guidance-and-self-correction, psi-duo-curriculum-background, psi-superposition-posteriors-method, mdlm-masked-diffusion-objective, discrete-diffusion-foundations-d3pm]
---
# Ψ-Samplers 的研究脈絡總覽

*The Diffusion Duality, Chapter II: Ψ-Samplers*（arXiv 2602.21185；Deschenaux、Gulcehre、Sahoo）處理的問題：uniform-state 離散擴散模型（USDM）在少步生成與 guidance 表現良好，但用 ancestral sampler 增加取樣步數時品質趨於持平（paper 摘要）。論文提出適用於任意噪聲過程的 predictor-corrector（PC）採樣器，並改良 Duo 的 Gaussian 鬆弛課程學習。

Duo 本身的內容見 [The Diffusion Duality](diffusion-duality.html)，本脈絡不重複。

## 閱讀順序
1. [離散擴散的 interpolating 前向過程與兩種 prior](psi-discrete-diffusion-interpolating-process.html)：論文第 2.1 節的記號，以及 masked 與 uniform 兩種 posterior。既有筆記：[離散擴散的基礎（D3PM）](discrete-diffusion-foundations-d3pm.html)、[MDLM](mdlm-masked-diffusion-objective.html)。
2. [連續時間離散擴散與 CTMC 採樣器](psi-ctmc-discrete-diffusion-samplers.html)：Campbell 等人的 CTMC 框架、SEDD、Discrete Flow Matching。
3. [從 Gaussian 到離散的 predictor-corrector](psi-predictor-corrector-gaussian-to-discrete.html)：Song 等人的 SDE 校正器、CTMC 校正器、ReMDM。
4. [USDM、guidance 與自我修正](psi-uniform-state-guidance-and-self-correction.html)：為何論文關注 uniform 噪聲。
5. [Duo 與課程學習背景](psi-duo-curriculum-background.html)：論文第 4 節 top-k 課程的前置。
6. [Ψ-posterior 與 Ψ-sampler](psi-superposition-posteriors-method.html)：論文本身的方法。

年代順序見 timeline.json。
