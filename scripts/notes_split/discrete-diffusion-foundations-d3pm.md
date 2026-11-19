---
slug: discrete-diffusion-foundations-d3pm
title: 離散擴散的基礎：D3PM、Diffusion-LM 與 DiffusionBERT
cluster: Diffusion Language Models
kind: topic
tags: [discrete-diffusion, absorbing-state, uniform-noise]
arxiv: ["2107.03006", "2205.14217", "2211.15029", "2512.15596"]
sources: ["paper:2512.15596 附錄 B 與第 2 節對該工作的描述", "arxiv.org/abs 頁面（標題、作者、年份）"]
related: ["corrective-dlm-overview", "mdlm-masked-diffusion-objective", "continuous-flow-lm-overview"]
---
# 離散擴散的基礎：D3PM、Diffusion-LM 與 DiffusionBERT

屬於 [CDLM 研究脈絡總覽](corrective-dlm-overview.html)。連續空間的語言擴散見 [2026 年連續 flow 語言模型總覽](continuous-flow-lm-overview.html)。

## 為什麼需要離散擴散
CDLM 論文附錄 B.1 的描述：Diffusion-LM（Li et al., 2022；arXiv 2205.14217）在 token embedding 的連續空間加入高斯噪聲，但論文認為這與文字的離散性質不一致。

## D3PM
- Austin et al.（2021；arXiv 2107.03006），*Structured Denoising Diffusion Models in Discrete State-Spaces*。
- 論文描述：D3PM 以可調的轉移核建立離散狀態擴散的統一框架，涵蓋均勻、吸收與結構化三種破壞過程。
- 與 CDLM 的關係：論文用到其中兩種。吸收過程把 token 換成遮罩 token $（標準 MDLM），均勻過程把 token 換成隨機 token（CDLM 加入的第二種噪聲）。

## DiffusionBERT
- He et al.（2023；arXiv 2211.15029），*DiffusionBERT*，作者共 5 人。
- 論文描述：以預訓練 BERT 作為吸收狀態（[MASK]）擴散的去噪器，在困惑度與 BLEU 上優於 D3PM 與 Diffusion-LM。
- 這是「MLM 式預訓練與擴散結合」的早期例子，也是論文討論 BERT 式訓練（監督所有位置）時的背景。

## 小結
兩種離散噪聲在 D3PM 就已並列；後續主流語言模型選擇吸收噪聲，見 [MDLM](mdlm-masked-diffusion-objective.html)。
