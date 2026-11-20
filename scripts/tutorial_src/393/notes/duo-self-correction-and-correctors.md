---
slug: duo-self-correction-and-correctors
title: 自我修正與 predictor-corrector：Duo 選擇 USDM 的理由
cluster: Diffusion Language Models
kind: topic
tags: [self-correction, predictor-corrector, remasking, usdm, mdm]
arxiv: ["2506.10892", "2205.14987", "2407.21243", "2503.00307", "2407.15595"]
sources: ["arXiv:2506.10892 §1、§2.1、§5.2", "各論文 arXiv 摘要"]
related: ["duo-lineage-hub", "duo-discrete-diffusion-masked-and-uniform", "duo-consistency-distillation-to-discrete", "dlm-remasking-self-correction", "corrective-dlm-overview"]
---
# 自我修正與 predictor-corrector

屬於 [Duo 研究脈絡總覽](duo-lineage-hub.html)。

## 問題
MDM 的 token 一旦解除遮罩就固定，早期錯誤無法在後續步驟修正；論文並指出 MDM 在少步時同時生成許多彼此獨立的 token，導致不連貫（§5.2）。

## 針對 MDM 的補救（Duo 引用）
- Campbell 等人 2022（arXiv:2205.14987）與 Gat 等人 2024，*Discrete Flow Matching*（arXiv:2407.15595）：Duo 列為 MDM 的 predictor-corrector 方法。
- Zhao 等人 2024，*Informed Correctors for Discrete Diffusion Models*（arXiv:2407.21243）。
- Wang 等人 2025，*Remasking Discrete Diffusion Models with Inference-Time Scaling*（arXiv:2503.00307）：Duo 的作者群之一為此文第一作者。
Duo 的評語是這些方法需要額外的函數評估（NFE）。

## Duo 的取捨
USDM 的反向過程每步都可更新 token，Duo 稱其為「自我修正」，不需額外 corrector 步驟。論文結果（§5.2）：T ≤ 32 時蒸餾後 Duo 優於蒸餾後 MDLM，論文歸因於 USDM 能在後續去噪步驟修正前面的錯誤；NFE 較高時 MDLM 的 Gen PPL 較好。論文並說明 MDLM 以 SDTT 蒸餾後 Gen PPL 可比 AR，但熵較低（5.4 對 5.6），表示多樣性下降。

站內對重新遮罩與自我修正的整理見 [dLLM 的重新遮罩與自我修正](dlm-remasking-self-correction.html) 與 [Corrective dLM 總覽](corrective-dlm-overview.html)。
