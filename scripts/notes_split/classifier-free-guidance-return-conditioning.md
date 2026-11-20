---
slug: classifier-free-guidance-return-conditioning
title: 無分類器引導、報酬條件與低溫取樣
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [classifier-free-guidance, return-conditioning, low-temperature-sampling, diffusion, decision-diffuser]
arxiv: ["2211.15657", "2207.12598", "2102.09672", "2011.13456", "2205.09991"]
sources: ["paper.txt §2.2, §3.2, §3.3, §3.4, Appendix C, K", "2207.12598 與 2102.09672 依本論文描述", "arXiv API metadata"]
related: ["decision-diffuser-lineage-hub", "diffusion-classifier-guidance-background", "ddpm-training-and-sampling-recap", "eps-prediction-ddpm-score", "trajectory-diffusion-planning", "implicit-dp-trajectory-stitching", "dd-experiments-and-ablations"]
---
# 無分類器引導、報酬條件與低溫取樣

屬於[總覽](decision-diffuser-lineage-hub.html)。

## 兩種條件化方式
論文 §2.2 把條件取樣分成兩類（噪聲預測與分數的對應見[ε 預測與分數](eps-prediction-ddpm-score.html)；分類器引導的細節見[分類器引導背景](diffusion-classifier-guidance-background.html)）：
- 分類器引導（Nichol & Dhariwal，arXiv 2102.09672，"Improved Denoising Diffusion Probabilistic Models"）：另外在帶噪資料上訓練 p_φ(y|x_k)，取樣時使用 ε_θ(x_k,k) − ω√(1−ᾱ_k)∇log p(y|x_k)。
- 無分類器引導（Ho & Salimans，arXiv 2207.12598，"Classifier-Free Diffusion Guidance"）：同一個網路同時學條件與無條件噪聲，無條件以空值 Ø 取代 y，取樣時使用 ε_θ(x_k,k) + ω(ε_θ(x_k,y,k) − ε_θ(x_k,k))。

Diffuser（[軌跡層級擴散規劃](trajectory-diffusion-planning.html)）以訓練好的報酬函數引導；對報酬而言這需要估計 Q 函數，也就是另一個動態規劃程序。DD 改為直接以回報 R(τ)（正規化到 [0,1]）作為 y，推論時設 R(τ)=1。

## 訓練
訓練時以機率 p 丟棄條件（β ~ Bern(p)，條件換成 Ø）。條件 y 為純量或 one-hot，經 MLP 投影成隱變數 z；y=Ø 時 z 全為零。

## 低溫取樣
去噪步驟改為 x_{k-1} ~ N(μ_{k-1}, αΣ_{k-1})，α ∈ [0,1)。附錄 C 的 Hopper-Medium-Expert 結果：α=0 為 104.3±0.7，α=0.5 為 111.8±1.6，α=1.0 為 107.1±3.5。論文的解讀是 α=0 去除取樣的熵而降低回報，α=1.0 的回報變異較大。

## 與 Q 函數引導的比較（附錄 K）
論文主張兩者理論上等價，但實務上無分類器引導較好，理由有二：條件模型在訓練時就專注於條件生成；離線資料訓練的 Q 函數可能對 OOD 動作給出過高估計（見[分布偏移](offline-rl-distribution-shift.html)），把擴散模型導向 OOD 動作。
