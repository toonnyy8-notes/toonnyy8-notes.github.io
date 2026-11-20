---
slug: aniso-target-normal-form-and-recovery
title: AnisoFlow 目標的標準形式與速度還原
cluster: Diffusion & Generative Models
kind: topic
tags: [anisoflow, velocity-recovery, psd, rectified-flow, target-family]
arxiv: ["2611.17767", "2210.02747", "2209.03003", "2209.15571", "2601.21419", "2605.12964"]
sources: [2611.17767 sections 3-4, 2611.17767 appendix A.1, 2210.02747 title, 2209.03003 title, 2209.15571 title]
related: ["aniso-lineage-overview", "prediction-targets-k-diff", "aniso-interpolated-targets-edm-kdiff-asymflow", "aniso-spectral-weight-analysis", "flow-matching-velocity-target", "prediction-targets-overview"]
---
# AnisoFlow 目標的標準形式與速度還原

回到[總覽](aniso-lineage-overview.html)。

## 前置：線性路徑與速度
本論文採用 z_t = α_t x + σ_t ε，α_t = 1 − t，σ_t = t，速度 u = ε − x，訓練目標是對 u 做平方誤差回歸（flow matching：Lipman 等 arXiv:2210.02747、Liu 等 arXiv:2209.03003、Albergo & Vanden-Eijnden arXiv:2209.15571）。這與 [No.238 教學](prediction-targets-k-diff.html)的慣例（t = 1 為資料）方向相反。背景見 [Flow matching 的速度目標](flow-matching-velocity-target.html)。

## 目標的標準形式（Eq. 3–4）
任意線性目標 y = Kx + Lε 乘上可逆矩陣 T = −K⁻¹ 後，可寫成 u_W = Wε − x，W = −K⁻¹L。整體尺度可被網路輸出吸收，所以資料係數固定為 −1，只學噪聲權重。純噪聲預測（K 奇異）在 w_i → ∞ 的極限取得。

## 速度還原（Eq. 5）
û = (α_t W + σ_t I)⁻¹ ((I − W) z_t + û_W)；在 W 的特徵基底中逐座標為 û_i = ((1 − w_i) z_t,i + û_W,i) / (α_t w_i + σ_t)。因此損失仍是還原後速度的平方誤差，網路、取樣器不變。

## 為何限制 W 為半正定
還原在 −t/(1 − t) 為 W 的特徵值時失效，此量涵蓋整條負實軸；為使每個內部時間步都可還原，W 不能有負實特徵值。限制 W 對稱後，條件化為 W ⪰ 0（Appendix A.1）。

## 特例
w_i = 0：資料預測；w_i = 1：速度；w_i ∈ {0,1} 且基底為 PCA：AsymFlow；w_i ≡ w：k-Diff 型純量內插；w_i → ∞：噪聲預測。預設初始化為 B = I、w_i = 1，即從速度預測起步。
