---
slug: corrective-dlm-overview
title: 修正式擴散語言模型（CDLM）的研究脈絡總覽
cluster: Diffusion Language Models
kind: hub
tags: [masked-diffusion, self-correction, remasking, code-benchmark]
arxiv: ["2512.15596", "2107.03006", "2406.07524", "2502.09992", "2503.00307", "2503.04482", "2510.01384"]
sources: ["paper:2512.15596", "arxiv.org/abs 頁面（標題、作者、年份）"]
related: ["discrete-diffusion-foundations-d3pm", "mdlm-masked-diffusion-objective", "large-masked-dlms-llada-dream", "dlm-remasking-self-correction", "uniform-noise-gidd-clean-supervision", "code-benchmarks-for-revision", "corrective-dlm-mixture-noise-posttraining"]
---
# 修正式擴散語言模型（CDLM）的研究脈絡總覽

論文《Corrective Diffusion Language Models》（arXiv 2512.15596）指出：現成的遮罩式擴散語言模型（MDLM）訓練時只監督被遮罩的位置，因此不能分辨可見序列中哪些 token 是錯的。論文提出在後訓練加入「均勻替換的可見 token 也要監督」的目標，並建立 Code Revision Benchmark（CRB）評估錯誤定位與就地修正。

理解這篇論文需要依序掌握下列主題，建議按此順序閱讀：

1. [離散擴散的基礎：D3PM、Diffusion-LM 與 DiffusionBERT](discrete-diffusion-foundations-d3pm.html)：吸收（遮罩）與均勻兩種離散噪聲的來源。
2. [MDLM：遮罩式擴散的訓練目標](mdlm-masked-diffusion-objective.html)：論文第 2.1 節的 {absorb}$ 從何而來。
3. [大型遮罩式 DLM：LLaDA、Dream 與相關系統](large-masked-dlms-llada-dream.html)：論文評估的基底模型與其訓練來源。
4. [DLM 的 remask 式迭代修正與自我修正](dlm-remasking-self-correction.html)：論文第 2.2 節「信心驅動的 remask-then-repredict」的前人工作。
5. [均勻噪聲、GIDD 與是否監督乾淨 token](uniform-noise-gidd-clean-supervision.html)：與論文方法最接近的 GIDD 以及兩者的差異。
6. [程式碼評測基準：HumanEval、MBPP 與 CRB 的定位](code-benchmarks-for-revision.html)：CRB 沿用的資料集與它補上的評測缺口。
7. [CDLM 的混合噪聲後訓練](corrective-dlm-mixture-noise-posttraining.html)：把前面各項合在一起的論文方法。

年代順序見同資料夾的 timeline：2021 D3PM，2022 Diffusion-LM／DiffusionBERT，2024 MDLM，2025 LLaDA、Dream、ReMDM、GIDD、PRISM，最後是 2025 年 12 月提出的本論文。
