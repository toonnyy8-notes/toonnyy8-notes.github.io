---
slug: rolling-diffusion
title: Rolling Diffusion：滑動窗口漸進噪聲排程
cluster: Diffusion & Generative Models
kind: topic
tags: [rolling-diffusion, sliding-window, noise-schedule, video-prediction, streaming]
sources: [blockwise-smc-separation.md, causal-attention-diffusion.md]
arxiv: ["2402.09470", "2506.20024", "2503.08307", "2503.10488"]
related: ["long-sequence-consistency-overview", "causal-attention-diffusion-overview", "blockwise-smc-time-axis"]
---
# Rolling Diffusion：滑動窗口漸進噪聲排程

屬於 [長序列一致性總覽](long-sequence-consistency-overview.html) 與 [Causal Attention × Diffusion 總覽](causal-attention-diffusion-overview.html)。

- **論文**：Ruhe, Heek, Salimans, Hoogeboom，"Rolling Diffusion Models"，arXiv 2402.09470（ICML 2024）。

## 核心機制
sliding window 去噪——將全域擴散時間重參數化為「frame-dependent local time」，對窗內較晚（較遠未來）的 frame 指派較高噪聲，反映「未來不確定性遞增」的 arrow-of-time inductive bias（SNR(τ_{w+1}) < SNR(τ_w)）。這隱含了時間因果結構（過去比未來更確定／更乾淨）。用線性 reparameterization 使同一 inductive bias 可在任意窗位置重用，支援無限展開。

- **為何不需 AR**：透過漸進噪聲排程而非逐步解碼來實現時間方向性；但推理時以滑動窗口前進，本質上是一種「軟自回歸」。
- 是否重訓：**是**。
- 能否 posterior sampling：未見，但 sliding window + 遞增噪聲的結構與音訊時間軸天然契合。
- 計算成本：與窗大小成正比，比 full-sequence 便宜。

## 應用與結果
- 在 Kinetics-600 長影片預測與混沌流體動力學預測上，當時間動態複雜時優於標準擴散。
- **Elucidated Rolling Diffusion Models（ERDM，NVIDIA，arXiv 2506.20024）**：將 EDM 的預條件/取樣器原則移植到 rolling 設定，用於 2D Navier-Stokes 與 ERA5 全球天氣預報（1.5°），優於 conditional autoregressive EDM baseline。
- 已被延伸到音訊-影片聯合生成（RFLAV, arXiv:2503.08307 的 Rolling Flow Matching），但未用於 posterior sampling。
- **Streaming Co-Speech Gestures via Accelerated Rolling Diffusion**（arXiv 2503.10488）：把 Rolling Diffusion 用於串流手勢生成（連續運動序列），以結構化漸進噪聲排程實現連續長序列生成。

## 局限
當時間動態簡單時優勢不明顯；推理仍帶滾動性質。

## 轉移到音訊
sliding window 遞增噪聲適合 streaming 分離；可作為 [block-wise SMC](blockwise-smc-time-axis.html) 階段 3 的恆定記憶體 streaming 結構之一。
