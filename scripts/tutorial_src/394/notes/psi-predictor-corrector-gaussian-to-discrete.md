---
slug: psi-predictor-corrector-gaussian-to-discrete
title: Predictor-corrector 採樣器：從 Gaussian SDE 到 ReMDM
cluster: Diffusion Language Models
kind: topic
tags: [predictor-corrector, remdm, remasking, sampler, inference-time-scaling]
arxiv: ["2602.21185", "2011.13456", "2205.14987", "2407.15595", "2503.00307"]
sources: ["arxiv.org/abs 摘要（2011.13456、2503.00307）", "paper:2602.21185 第 1、2.1.1、3、6 節", "arXiv API metadata（標題、作者、年份）"]
related: [psi-samplers-lineage-hub, psi-ctmc-discrete-diffusion-samplers, psi-superposition-posteriors-method, dlm-remasking-self-correction, mdlm-masked-diffusion-objective]
---
# Predictor-corrector 採樣器：從 Gaussian SDE 到 ReMDM

屬於 [Ψ-Samplers 脈絡總覽](psi-samplers-lineage-hub.html)。

## Gaussian 擴散中的 PC
Song 等人（2020，arXiv 2011.13456）的摘要指出，其 SDE 框架引入 predictor-corrector 架構，用來修正離散化反向 SDE 演化中的誤差。論文第 3 節以此類比：corrector 對應加入額外 Gaussian 噪聲，在 Ψ-sampler 中 $q_t$ 扮演 corrector，$q_{s|t}$ 扮演 predictor。

## 離散擴散中的 PC
- Campbell 等人（2022）與 Gat 等人（2024）的 PC 方法以 CTMC 為基礎，見 [CTMC 採樣器](psi-ctmc-discrete-diffusion-samplers.html)。
- Wang 等人（2025，arXiv 2503.00307）*Remasking Discrete Diffusion Models with Inference-Time Scaling*（ReMDM）。摘要：masked diffusion 的 token 一旦生成就不能更新；ReMDM 是可套用於預訓練 masked diffusion 的採樣器，源自帶自訂 remasking 反向過程的離散擴散模型，並使推論時增加取樣步數能提升品質。論文第 2.1.1 節補充：ReMDM 的 posterior 與式 (2) 保持相同邊際，並推廣先前的 PC 方法。

remask 式修正的其他工作見 [DLM 的 remask 式迭代修正](dlm-remasking-self-correction.html)；MDM 的 posterior 形式見 [MDLM](mdlm-masked-diffusion-objective.html)。

## 論文的定位
ReMDM 限於 masked 噪聲。論文第 6 節說明 Ψ-sampler 把它推廣到任意擴散過程，且不像 Lezama 等人（2023）等工作另外訓練 corrector 模組。細節見 [Ψ-posterior](psi-superposition-posteriors-method.html)。
