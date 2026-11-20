---
slug: duo-self-correction-and-correctors
title: 自我修正與 predictor-corrector：Duo 選擇 USDM 的理由
cluster: Diffusion Language Models
kind: topic
tags: [self-correction, predictor-corrector, remasking, usdm, mdm, remdm, sde]
arxiv: ["2506.10892", "2205.14987", "2407.21243", "2503.00307", "2407.15595", "2602.21185", "2011.13456"]
sources: ["arXiv:2506.10892 §1、§2.1、§5.2", "各論文 arXiv 摘要", "paper:2602.21185 第 2.1.1、3、6 節", "arxiv.org/abs 摘要（2011.13456、2503.00307）"]
related: ["duo-lineage-hub", "psi-samplers-lineage-hub", "psi-superposition-posteriors-method", "psi-ctmc-discrete-diffusion-samplers", "duo-discrete-diffusion-masked-and-uniform", "duo-consistency-distillation-to-discrete", "dlm-remasking-self-correction", "corrective-dlm-overview"]
---
# 自我修正與 predictor-corrector

屬於 [Duo 研究脈絡總覽](duo-lineage-hub.html)與 [Ψ-Samplers 脈絡總覽](psi-samplers-lineage-hub.html)。

## 問題
MDM 的 token 一旦解除遮罩就固定，早期錯誤無法在後續步驟修正；論文並指出 MDM 在少步時同時生成許多彼此獨立的 token，導致不連貫（§5.2）。

## 針對 MDM 的補救（Duo 引用）
- Campbell 等人 2022（arXiv:2205.14987；其 CTMC 框架見 [CTMC 採樣器](psi-ctmc-discrete-diffusion-samplers.html)）與 Gat 等人 2024，*Discrete Flow Matching*（arXiv:2407.15595）：Duo 列為 MDM 的 predictor-corrector 方法。
- Zhao 等人 2024，*Informed Correctors for Discrete Diffusion Models*（arXiv:2407.21243）。
- Wang 等人 2025，*Remasking Discrete Diffusion Models with Inference-Time Scaling*（arXiv:2503.00307，ReMDM）：Duo 的作者群之一為此文第一作者。摘要：masked diffusion 的 token 一旦生成就不能更新；ReMDM 是可套用於預訓練 masked diffusion 的採樣器，源自帶自訂 remasking 反向過程的離散擴散模型，並使推論時增加取樣步數能提升品質。Ψ-Samplers 論文第 2.1.1 節補充：ReMDM 的 posterior 與式 (2) 保持相同邊際，並推廣先前的 PC 方法。
Duo 的評語是這些方法需要額外的函數評估（NFE）。

## Duo 的取捨
USDM 的反向過程每步都可更新 token，Duo 稱其為「自我修正」，不需額外 corrector 步驟。論文結果（§5.2）：T ≤ 32 時蒸餾後 Duo 優於蒸餾後 MDLM，論文歸因於 USDM 能在後續去噪步驟修正前面的錯誤；NFE 較高時 MDLM 的 Gen PPL 較好。論文並說明 MDLM 以 SDTT 蒸餾後 Gen PPL 可比 AR，但熵較低（5.4 對 5.6），表示多樣性下降。

站內對重新遮罩與自我修正的整理見 [dLLM 的重新遮罩與自我修正](dlm-remasking-self-correction.html) 與 [Corrective dLM 總覽](corrective-dlm-overview.html)。

## Gaussian 擴散中的 PC 與 Ψ-Samplers 的定位
Song 等人（2020，arXiv 2011.13456）的摘要指出，其 SDE 框架引入 predictor-corrector 架構，用來修正離散化反向 SDE 演化中的誤差。Ψ-Samplers 論文第 3 節以此類比：corrector 對應加入額外 Gaussian 噪聲，在 Ψ-sampler 中 $q_t$ 扮演 corrector，$q_{s|t}$ 扮演 predictor。

Campbell 等人（2022）與 Gat 等人（2024）的 PC 方法以 CTMC 為基礎，見 [CTMC 採樣器](psi-ctmc-discrete-diffusion-samplers.html)。ReMDM 限於 masked 噪聲；Ψ-Samplers 論文第 6 節說明 Ψ-sampler 把它推廣到任意擴散過程，且不像 Lezama 等人（2023）等工作另外訓練 corrector 模組。細節見 [Ψ-posterior](psi-superposition-posteriors-method.html)；MDM 的 posterior 形式見 [MDLM](mdlm-masked-diffusion-objective.html)。
