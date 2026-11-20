---
slug: aniso-quantization-conditioning
title: 低精度推論與旋轉量化：x-prediction 的 1/σ_t 放大
cluster: Diffusion & Generative Models
kind: topic
tags: [quantization, fp8, conditioning, quarot, spinquant]
arxiv: ["2611.17767", "2404.00456", "2405.16406", "2412.19437"]
sources: [2611.17767 related work, 2611.17767 section 6.2, 2404.00456 abstract, 2405.16406 abstract, 2412.19437 title]
related: ["aniso-lineage-overview", "aniso-target-normal-form-and-recovery", "aniso-spectral-weight-analysis"]
---
# 低精度推論與旋轉量化

回到[總覽](aniso-lineage-overview.html)。

## 問題
資料預測（w_i = 0）還原速度時，預測誤差被乘上與 1/σ_t 成正比的係數，低噪聲時病態（本論文 Sec. 2、Sec. 4）。本論文 Sec. 6.2 報告：在 INT8、FP8、blockwise FP8 推論下，x-prediction 基準出現 patch 接縫或強烈噪聲，AnisoFlow 則維持穩定；本論文歸因於學到的目標使還原後速度誤差最小，因而條件較好。
還原式的分母 α_t w_i + σ_t 在 w_i > 0 時不趨近 0，這是條件改善的來源（Eq. 5）。

## 旋轉量化
本論文指出 AnisoFlow 的「旋轉後縮放」形式類似旋轉量化：
- **QuaRot**（Ashkboos 等，arXiv:2404.00456）：摘要說明以旋轉移除隱藏狀態中的離群值而不改變輸出，使 LLM 端到端 4 位元量化。
- **SpinQuant**（Liu 等，arXiv:2405.16406）：摘要說明以學到的旋轉矩陣改善量化。
- 區別（本論文）：這些方法逐層作用於已訓練的網路；AnisoFlow 透過預測目標與網路一起學習全網路的 preconditioning。

## FP8 blockwise
blockwise FP8 推論引用 DeepSeek-V3 技術報告（arXiv:2412.19437）；本筆記只引用其作為量化格式的出處。
