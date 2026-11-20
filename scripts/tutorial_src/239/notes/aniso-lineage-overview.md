---
slug: aniso-lineage-overview
title: AnisoFlow 的研究脈絡總覽：從固定預測目標到可學習的各向異性目標
cluster: Diffusion & Generative Models
kind: hub
tags: [anisoflow, prediction-target, flow-matching, pixel-space, lineage]
arxiv: ["2611.17767"]
sources: [2611.17767 introduction, 2611.17767 related work]
related: ["prediction-targets-k-diff", "aniso-target-normal-form-and-recovery", "aniso-interpolated-targets-edm-kdiff-asymflow", "aniso-pixel-patchified-setting", "aniso-spectral-weight-analysis", "aniso-quantization-conditioning", "aniso-latent-models-check", "prediction-targets-overview", "flow-matching-velocity-target", "pixel-space-diffusion"]
---
# AnisoFlow 的研究脈絡總覽

*Learning Anisotropic Prediction Targets for Flow Models*（Zirvi、Chen 等，arXiv:2611.17767）把 flow model 的預測目標從手選改成可學習：目標寫成 u_W = Wε − x，W 為半正定矩陣，每個方向各有一個非負的噪聲權重。本文依閱讀順序列出理解這篇論文所需的前置脈絡。

## 閱讀順序
1. 共同背景：x/ε/v 預測、flow matching 與 loss 加權已在站內另有筆記，見[預測目標演進總覽](prediction-targets-overview.html)與[Flow matching 的速度目標](flow-matching-velocity-target.html)；對應的互動教學為[預測目標與 k-Diff（No.238）](prediction-targets-k-diff.html)。本論文的符號與該教學不同：本論文 z_t = (1−t)x + tε、u = ε − x。
2. [Pixel 空間的 patchify 設定](aniso-pixel-patchified-setting.html)：為什麼目標問題在 pixel 空間最明顯（JiT、PixelFlow、DiT）。
3. [內插目標：EDM、k-Diff、AsymFlow](aniso-interpolated-targets-edm-kdiff-asymflow.html)：AnisoFlow 直接泛化的三條前驅。
4. [AnisoFlow 目標的標準形式與速度還原](aniso-target-normal-form-and-recovery.html)：Eq. 3–5 與特例。
5. [學到的權重分析：誤差放大與 w ∝ λ](aniso-spectral-weight-analysis.html)：PCA 基底下的理論，及相關的譜偏差與條件數研究。
6. [低精度推論與旋轉量化](aniso-quantization-conditioning.html)：x-prediction 的 1/σ_t 放大與 QuaRot、SpinQuant 的對照。
7. [Latent 模型上的對照實驗](aniso-latent-models-check.html)：SiT-XL/2-REPA 與 RAEv2。

## 時間線
各前置工作的年份、角色與出處見同資料夾的 timeline.json。
