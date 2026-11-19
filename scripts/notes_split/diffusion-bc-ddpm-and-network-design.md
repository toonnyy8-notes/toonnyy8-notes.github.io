---
slug: diffusion-bc-ddpm-and-network-design
title: 擴散行為複製、DDPM 目標與 LN_Resnet 網路設計
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [diffusion, ddpm, behavior-cloning, score-network, layer-norm, resnet]
arxiv: ["2304.10573", "1503.03585", "2006.11239", "2011.13456", "2301.10677", "2304.02532", "2205.09991", "2211.15657"]
sources: ["arXiv:2304.10573 §2, §3, §5", "arXiv 摘要 1503.03585、2006.11239、2011.13456、2301.10677、2304.02532"]
related: ["diffusion-behavior-model-policy-extraction", "ddpm-training-and-sampling-recap", "trajectory-diffusion-planning", "idql-evaluation-and-online-finetuning"]
---
# 擴散行為複製、DDPM 目標與 LN_Resnet

## 擴散模型的來源

IDQL §3 以三篇作為擴散模型的來源：Sohl-Dickstein et al.（arXiv 1503.03585，2015，以非平衡熱力學為基礎的深度無監督學習）、Ho et al. 的 DDPM（arXiv 2006.11239，2020）、Song et al. 以 SDE 表述的 score-based 生成模型（arXiv 2011.13456，2020）。DDPM 的完整訓練與取樣見 [DDPM 的訓練與取樣流程](ddpm-training-and-sampling-recap.html)。

## IDQL 使用的形式

前向過程：q(a_t | a_{t-1}) = N(sqrt(1−β_t) a_{t-1}, β_t I)。訓練以 DDPM 目標（式 4）預測雜訊 ε，條件為狀態 s 與步數 t；取樣為反向擴散，a_T ~ N(0, I)，每步重新取樣 ε（式 5）。此模型在 IDQL 中表示行為策略 μ_φ(a|s)。

## 擴散用於行為複製與決策

- Pearce et al.（arXiv 2301.10677，2023）：以擴散模型模仿人類行為（行為複製）。
- Florence et al.：Implicit behavioral cloning，用 energy-based 模型（IDQL 引用，未取得 arXiv 條目，未驗證）。
- Reuss et al.（arXiv 2304.02532，2023）：以 score-based 擴散策略做目標條件模仿學習。
- Janner et al.（arXiv 2205.09991，2022）與 Ajay et al.（arXiv 2211.15657，2022）：以擴散建模整條軌跡，分別用梯度引導與獎勵條件取樣。見 [軌跡層級擴散規劃（Diffuser）](trajectory-diffusion-planning.html)。

## 連續動作空間的問題與網路設計（§5）

簡單 MLP 的 DDPM 在 2D 連續資料集上無法擬合分布並產生許多分布外樣本。這些離群值在訓練過的 Q 函數上可能得到錯誤的高值。增加批次大小與容量可改善但離群值仍在。IDQL 改用帶 layer normalization（Ba et al. 2016, arXiv 1607.06450）的殘差網路（LN_Resnet）作為 score 網路。§6.4 的消融：LN_Resnet 使結果對 N 較不敏感；MLP 增加 N 反而降低表現，尤其在 antmaze。

策略抽取如何使用此模型見 [以擴散行為模型做策略抽取](diffusion-behavior-model-policy-extraction.html)，結果見 [IDQL 的實驗與線上微調](idql-evaluation-and-online-finetuning.html)。
