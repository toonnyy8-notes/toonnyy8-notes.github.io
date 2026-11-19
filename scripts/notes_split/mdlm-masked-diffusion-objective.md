---
slug: mdlm-masked-diffusion-objective
title: MDLM：遮罩式擴散的訓練目標
cluster: Diffusion Language Models
kind: topic
tags: [masked-diffusion, elbo, mlm, discrete-diffusion, mdlm, d3pm]
arxiv: ["2406.07524", "2512.15596", "2107.03006", "2310.16834", "2406.04329", "2406.03736", "2202.04200", "2512.09106"]
sources: ["paper:2512.15596 附錄 B 與第 2 節對該工作的描述", "arxiv.org/abs 頁面（標題、作者、年份）", "paper.txt §1, §2.1", "arXiv API metadata"]
related: [corrective-dlm-overview, discrete-diffusion-foundations-d3pm, large-masked-dlms-llada-dream, corrective-dlm-mixture-noise-posttraining, unmasking-policies-dllm-overview, unmasking-mdp-policy-design]
---
# MDLM：遮罩式擴散的訓練目標

屬於 [CDLM 研究脈絡總覽](corrective-dlm-overview.html)與[unmasking policy 總覽](unmasking-policies-dllm-overview.html)。前置：[離散擴散的基礎](discrete-diffusion-foundations-d3pm.html)。

## 來源
Sahoo et al.（2024；arXiv 2406.07524），*Simple and Effective Masked Diffusion Language Models*，作者共 8 人。CDLM 論文附錄 B.1 描述它把遮罩式擴散詮釋為「由加權的 MLM 式重建項組成的變分目標」，並在離散 DLM 中表現強。

## CDLM 論文使用的形式（第 2.1 節）
- 乾淨序列 =(x_1,\dots,x_n)$，吸收遮罩 token  \notin V$。
- 每個樣本抽 $\lambda\sim\mathrm{Uniform}[0,1]$，每個 token 以機率 $\lambda$ 獨立換成 $，得到 $。
- 損失 {absorb}(\theta)=\mathbb{E}_{x,z}\big[-\sum_{i:z_i=m}\log p_\theta(x_i\mid z)\big]$：只有被遮罩的位置有損失，可見位置的 logits 被忽略。

## 與 CDLM 的關係
論文的核心診斷來自這個目標的監督範圍：可見 token 沒有學習訊號，所以模型的信心不一定反映該 token 是否正確。CDLM 保留此項，再加上對受損可見 token 的項，見 [CDLM 的混合噪聲後訓練](corrective-dlm-mixture-noise-posttraining.html)。

論文指出 MDLM 這類模型是一整類 DLM 的基礎，列舉 LLaDA、Dream、Mercury、Seed Diffusion 等，見 [大型遮罩式 DLM](large-masked-dlms-llada-dream.html)。

## 離散擴散脈絡（Learning Unmasking Policies…，arXiv 2512.09106，第 1、2.1 節）
論文第 1 節把下列工作列為離散擴散（discrete diffusion）的起點：Austin 等人（2021）的 D3PM「Structured Denoising Diffusion Models in Discrete State-Spaces」（2107.03006）與 Lou 等人（2024）的「Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution」（2310.16834）。

Masked diffusion model（MDM）是離散擴散的子類，前向過程以隨時間變化的 BERT 式遮罩破壞序列。論文引用 Sahoo 等人「Simple and Effective Masked Diffusion Language Models」（2406.07524）、Shi 等人「Simplified and Generalized Masked Diffusion for Discrete Data」（2406.04329）與 Ou 等人「Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data」（2406.03736）作為 MDM 的訓練方法。

### 該論文使用的形式
前向過程對每個位置獨立：位置 k 以機率 1 − t 保持 x0 的 token，以機率 t 變成遮罩 token M。訓練目標是遮罩位置的 cross-entropy 加權和，論文式 (2.1) 寫成負 ELBO 的上界。

取樣從全遮罩序列開始，每步由 dLLM 預測所有位置的 token 分佈，再選出一個仍被遮罩的位置子集 U_t 並從預測分佈取樣填入（式 2.2）。選 U_t 的規則就是 unmasking 策略，訓練目標不決定這個規則。

### 與影像生成的關係
論文把 MaskGIT（Chang 等人，2202.04200，「MaskGIT: Masked Generative Image Transformer」）列為 confidence 式 unmasking 的先例：它依 token 信心決定每步解開哪些位置，並非固定隨機取樣。細節以 MaskGIT 原文為準，本筆記只記錄論文的引用關係。
