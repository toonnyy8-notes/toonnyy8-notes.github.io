---
slug: long-sequence-consistency-overview
title: 時序 Diffusion / Flow Matching 的長序列一致性與 zero-shot 音源分離總覽
cluster: Diffusion & Generative Models
kind: hub
tags: [long-sequence, temporal-consistency, source-separation, smc, posterior-sampling, overview]
sources: [blockwise-smc-separation.md]
arxiv: ["2407.01392", "2506.08009", "2412.07772", "2311.07345", "2302.08113", "2405.11473", "2312.00752"]
related: ["diffusion-forcing", "causvid-self-forcing", "duet-posterior-sampling-separation", "training-free-long-sequence-diffusion", "smc-diffusion-noise-axis", "blockwise-smc-time-axis", "streaming-flow-matching-speech", "rolling-diffusion"]
---
# 時序 Diffusion / Flow Matching 的長序列一致性與 zero-shot 音源分離總覽

主題：時序 diffusion / flow matching 的長序列上下文一致性——機制、理論，以及對 zero-shot diffusion source separation + SMC 的建議。

## TL;DR
- **訓練時最強的一致性機制是 [Diffusion Forcing](diffusion-forcing.html)（arXiv:2407.01392，per-token 獨立噪聲）與其視覺後繼 [Self-Forcing（arXiv:2506.08009）/ CausVid（arXiv:2412.07772）](causvid-self-forcing.html)；推論時（training-free，最適合 zero-shot inverse problem）最可轉移的是 [Duet Singing](duet-posterior-sampling-separation.html)（arXiv:2311.07345）的 overlapping-segment autoregressive posterior sampling、MultiDiffusion（arXiv:2302.08113）的重疊區平均、以及 FIFO-Diffusion（arXiv:2405.11473）的 diagonal denoising（見 [training-free 長序列機制](training-free-long-sequence-diffusion.html)）。**
- **關鍵研究缺口：所有已發表的 SMC + diffusion 方法都沿「diffusion 噪聲軸」做粒子濾波（FPS、MCGdiff、TDS、SMCDiff，見 [沿噪聲軸的 SMC](smc-diffusion-noise-axis.html)）；所有沿「時間軸」的分塊/自回歸 diffusion 生成都用單軌跡 rollout，不做粒子重採樣。沿 audio 時間軸做 block-wise SMC + diffusion prior 目前無前例——這正是使用者可以佔據的原創空間。**
- **對使用者的建議：以 Duet 式 overlapping-segment autoregressive posterior sampling 為 backbone，將「跨段一致性」重寫為沿時間軸的 Feynman-Kac / 粒子濾波問題（block-wise SMC），用 speaker/instrument embedding 或前段重疊區作為 twisting potential 重加權粒子，可同時解決 identity drift 與 error accumulation，且完全 training-free、與現有 UnDiff/MSDM/ArrayDPS/DGMO prior 相容。** 詳見 [沿時間軸的 block-wise SMC](blockwise-smc-time-axis.html)。

## Key Findings
1. **兩種「序貫性」必須區分**：diffusion 有兩個時間軸——(a) 去噪軸（noise level t: T→0），(b) 資料/序列軸（audio time n）。既有 SMC-diffusion 文獻（FPS、MCGdiff、TDS、SMCDiff）全部沿 (a) 做濾波；長序列生成問題本質上是沿 (b) 的外推。使用者提的 block-wise SMC 若沿 (b)，屬於文獻空白。
2. **訓練時機制**：Diffusion Forcing、Rolling Diffusion、CausVid、Self-Forcing 皆需重訓，但能穩定外推超過訓練長度；核心共通點是「乾淨的過去 context 以零噪聲餵入，未來以較高噪聲」，避免 autoregressive rollout 的分布偏移。
3. **推論時機制（training-free）**：MultiDiffusion / Gen-L-Video（重疊窗平均）、FIFO-Diffusion（對角去噪佇列）、outpainting/inpainting continuation、Duet 的 autoregressive posterior sampling，都不需重訓，直接可套在既有 prior 上；這是 zero-shot inverse problem 設定唯一可用的一類。
4. **flow matching 端**：CosyVoice 2 的 chunk-aware causal flow matching、F5-TTS/E2-TTS 的 infilling 條件、MeanFlow 的一步生成，是 flow 端對應技術；但多屬條件生成，尚未見用於 posterior sampling（見 [串流 flow matching](streaming-flow-matching-speech.html)）。
5. **與分離的連結**：僅 Duet Singing（arXiv:2311.07345）明確在 posterior sampling 設定下處理跨段 identity 一致性；MSDM、ArrayDPS、UnDiff、DGMO 皆未系統性處理長序列跨段漂移，多在固定片段上運作。

## 長上下文架構層面
- causal attention + KV cache（CausVid、Self-Forcing）。
- SSM/Mamba 作為 backbone：Mamba（arXiv:2312.00752）在音訊波形建模上優於 SaShiMi/Transformer——原文於 4 小時 YouTubeMix 16kHz 鋼琴資料上（訓練長度 8,192→約 10⁶ samples）指出「Mamba 與 SaShiMi(S4+MLP) baseline 皆隨 context 增長而穩定改善；Mamba 全程更佳，且差距在較長長度時擴大」，線性複雜度適合長序列；已有 DiffuMamba、Video Diffusion State Space Models 等把 Mamba 當 diffusion backbone。對 16kHz 長音訊特別有吸引力。

## 一般注意事項
- **時效與版本**：多篇為 2024–2026 的快速演進領域（video streaming diffusion 尤甚），部分後繼工作（Causal Forcing、Rolling Forcing、LongLive 等）仍在 arXiv 階段，結論可能隨新工作變動。
- **posterior sampling 可轉移性**：多數 video 一致性技術是生成導向，轉到 inverse-problem posterior sampling 的可行性多為推論，缺乏直接實驗證據，屬研究假設而非已證事實。

## 建議閱讀順序
1. [Diffusion Forcing](diffusion-forcing.html)
2. [Rolling Diffusion](rolling-diffusion.html)
3. [CausVid 與 Self-Forcing](causvid-self-forcing.html)
4. [串流 flow matching（CosyVoice 2、StreamFlow、MeanFlow 等）](streaming-flow-matching-speech.html)
5. [Training-free 長序列 diffusion 機制與外推失敗的理論](training-free-long-sequence-diffusion.html)
6. [Duet 式分段 posterior sampling 與 diffusion 音源分離](duet-posterior-sampling-separation.html)
7. [沿 diffusion 噪聲軸的 SMC（FPS、MCGdiff、TDS、SMCDiff）](smc-diffusion-noise-axis.html)
8. [沿時間軸的 block-wise SMC：研究缺口與分階段建議](blockwise-smc-time-axis.html)
