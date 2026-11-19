---
slug: blockwise-smc-time-axis
title: 沿時間軸的 block-wise SMC + diffusion prior：研究缺口與分階段建議
cluster: Diffusion & Generative Models
kind: topic
tags: [smc, block-wise, source-separation, research-gap, zero-shot, identity-drift]
sources: [blockwise-smc-separation.md]
arxiv: ["2101.12072", "2507.15716", "2311.07345", "2405.11473", "2502.05908"]
related: ["long-sequence-consistency-overview", "smc-diffusion-noise-axis", "duet-posterior-sampling-separation", "diffusion-forcing", "rolling-diffusion", "training-free-long-sequence-diffusion", "causvid-self-forcing", "streaming-flow-matching-speech"]
---
# 沿時間軸的 block-wise SMC + diffusion prior：研究缺口與分階段建議

屬於 [長序列一致性總覽](long-sequence-consistency-overview.html)。背景見 [沿噪聲軸的 SMC](smc-diffusion-noise-axis.html) 與 [Duet 式分段 posterior sampling](duet-posterior-sampling-separation.html)。

## block-wise / 沿時間軸 SMC 是否有前例
經專門調查，**沒有**。所有 SMC-diffusion 沿噪聲軸；TimeGrad（Rasul et al., ICML 2021, arXiv:2101.12072）沿時間軸自回歸但每步只採單一值向前 rollout（無 SMC、易累積誤差）；FIFO/Rolling/block video diffusion（如 BlockVid、SANA-Video、Rolling Forcing）沿時間軸但用 KV-cache/噪聲排程/蒸餾，無粒子重採樣。DiffPF（arXiv:2507.15716）則是相反關係。這是明確的研究缺口。

## video 技術轉移到音訊分離的可行性與缺口
- [Diffusion Forcing](diffusion-forcing.html) → 音訊：per-frame 獨立噪聲天然適合把「已分離的前段」當零噪聲 context，但需重訓 prior，與 zero-shot 目標衝突。
- [Rolling Diffusion](rolling-diffusion.html) → 音訊：sliding window 遞增噪聲適合 streaming 分離；已有 RFLAV 用於音訊-影片，但未用於 posterior sampling。
- FIFO diagonal denoising → 音訊 streaming inverse problem：training-free，最可能直接轉移（見 [training-free 長序列機制](training-free-long-sequence-diffusion.html)）。
- 缺口：以上都是生成設定，尚無人把它們與 posterior sampling / likelihood guidance 結合做長音訊分離。

## 分階段建議
**階段 1（低風險，立即可做）**：以 Duet（arXiv:2311.07345）的 overlapping-segment autoregressive posterior sampling 為 baseline，套在現有的 UnDiff/MSDM 式 prior 上。用重疊區 inpainting 條件傳遞 identity。這是 training-free、已驗證有效的起點。benchmark：跨段 SI-SDR、identity switch rate（可用 speaker/instrument verification embedding 的段間相似度量化 drift）。

**階段 2（核心創新，佔據空白）**：把階段 1 的單軌跡改為**沿時間軸的 block-wise SMC**。具體：
- 粒子 = 對「到目前為止已分離序列」的多個假設；
- 沿 audio 時間塊（非噪聲軸）傳播：每個新塊用 diffusion prior 做 posterior sampling 作為 proposal；
- twisting/reweight potential = 前段重疊區一致性 + （可選）speaker/instrument embedding 相似度 + 混音重建 likelihood；
- resample 淘汰漂移的粒子。

這同時解決 identity drift（重加權偏好一致粒子）與 error accumulation（多假設避免早期錯誤鎖死），且理論上粒子數→∞ 漸近正確（延續 FPS 的 Feynman-Kac 框架，但改到時間軸）。這是文獻空白，具原創性。可將 TDS 的 twisting 思想從噪聲軸「借」到時間軸作為理論骨架。

**階段 3（若需即時/長音訊）**：引入 FIFO-Diffusion（arXiv:2405.11473）式 diagonal denoising 佇列或 Rolling Diffusion 式遞增噪聲，做恆定記憶體的 streaming block-wise SMC。若願意重訓 prior，考慮 Diffusion Forcing / Self-Forcing 式訓練讓 prior 原生支援乾淨 context 條件化（見 [CausVid 與 Self-Forcing](causvid-self-forcing.html)）；若需降低每步成本，考慮 MeanFlow 式一步 proposal（見 [串流 flow matching](streaming-flow-matching-speech.html)）。

## 與 Duet AR posterior sampling 相比的優劣
- 優：block-wise SMC 保留多假設與回溯能力（resample），對 monotimbral（同音色）源的 identity drift 更 robust；有漸近正確性保證。
- 劣：計算成本 = 粒子數 × 段數 × 每段 sampling，比 Duet 單軌跡貴 N_particles 倍；需設計低方差 proposal 與有效 potential，否則粒子退化（weight collapse）。緩解：少粒子 + GPU 平行（如 LD-SMC, arXiv:2502.05908 所述）。

## 會改變建議的門檻
若階段 1 的 identity switch rate 已足夠低（如 < 5%），則 SMC 的邊際效益可能不值其成本，應改投資 better likelihood/guidance；若 drift 嚴重且 embedding 條件化無效，則 SMC 的多假設優勢最大化，值得全力投入階段 2。

## 注意事項
「沿時間軸 SMC + diffusion prior 無前例」是基於廣泛但非窮盡的搜尋（子代理已交叉檢查 TDS/FPS/MCGdiff/SMCDiff/TimeGrad/FIFO/block-video 等）；建議投稿前再做一次針對性檢索確認，尤其 2025–2026 的 inference-time-scaling 文獻。
