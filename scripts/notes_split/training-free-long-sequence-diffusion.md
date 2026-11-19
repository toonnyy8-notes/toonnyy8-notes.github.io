---
slug: training-free-long-sequence-diffusion
title: Training-free 長序列 diffusion 機制與固定長度 prior 外推失敗的理論
cluster: Diffusion & Generative Models
kind: topic
tags: [multidiffusion, fifo-diffusion, gen-l-video, stable-audio, training-free, length-extrapolation]
sources: [blockwise-smc-separation.md]
arxiv: ["2302.08113", "2305.18264", "2405.11473", "2402.04825", "2404.10301", "2507.08557"]
related: ["long-sequence-consistency-overview", "duet-posterior-sampling-separation", "causvid-self-forcing"]
---
# Training-free 長序列 diffusion 機制與固定長度 prior 外推失敗的理論

屬於 [長序列一致性總覽](long-sequence-consistency-overview.html)。這是 zero-shot inverse problem 設定唯一可直接使用的一類（不需重訓）。

## 推論時機制
- **Overlap-add / crossfade**：最簡單，但在分離的相位不連續處會產生 artifact，且無法傳遞 source identity。
- **MultiDiffusion（Bar-Tal et al., ICML 2023, arXiv:2302.08113）**：把多個重疊 crop 的 diffusion 路徑用最小平方（重疊區平均）綁定成單一一致生成——`Ψ(J_t,z)=argmin Σ‖F_i(J)−Φ(F_i(J),z)‖²`。原用於 panorama（H×4W），從同一初始噪聲融合原本不相關的路徑成連貫全景。可直接轉移到音訊時間軸（把 crop 換成時間窗）。
  - 是否重訓：**否**；能否 posterior sampling：可與 guidance 結合；計算成本：與窗數線性，重疊增加成本。
- **Gen-L-Video（arXiv:2305.18264）/ FreeNoise / FreeLong**：training-free 用 sliding-window 合併重疊子段（temporal co-denoising）擴展短影片模型到數百 frame，保持一致性；FreeNoise 用 window-based temporal attention + noise rescheduling。
- **FIFO-Diffusion（Kim et al., NeurIPS 2024, arXiv:2405.11473）**：diagonal denoising——維持一個佇列，frame 噪聲等級沿佇列遞增；每步 dequeue 頭部全乾淨 frame、enqueue 尾部新噪聲。加 latent partitioning 減少訓練-推論 gap、lookahead denoising 利用前向參考。training-free，理論上無限長，記憶體恆定。
  - 是否重訓：**否**；能否 posterior sampling：未見，但佇列結構與 streaming inverse problem 契合；計算成本：恆定記憶體，可多 GPU 平行。
  - 轉移到音訊 streaming inverse problem：training-free，最可能直接轉移。
- **outpainting/inpainting continuation** 與 Duet 的 autoregressive posterior sampling（見 [Duet 式分段 posterior sampling](duet-posterior-sampling-separation.html)）也屬此類。

## Latent 空間長序列（音訊）
Stable Audio（arXiv:2402.04825）用 timing conditioning（duration + start time embedding）讓 latent diffusion 生成到訓練窗長度的變長音訊，解決「隨機 crop 導致樂句中間開始/結束」問題；「Stable Audio is capable of rendering stereo signals of up to 95 sec at 44.1kHz in 8 sec on an A100 GPU (40GB VRAM)」。Long-form music generation（arXiv:2404.10301）進一步延伸。InfiniteAudio、FreeAudio（arXiv:2507.08557）做 training-free 長音訊。

## 為何固定長度訓練的 prior 外推失敗
- 訓練分布 vs 測試長度 mismatch：模型只見過固定窗內的統計，長序列的全局結構（如整首歌的 identity）不在訓練分布中。
- positional encoding 外推：LongDiff（CVPR 2025）分析指出，序列變長時模型「無法準確區分 frame 相對位置」（positional ambiguity）導致時序不連貫，以及「temporal correlation entropy 隨長度下降」（information dilution）。
- FIFO 的訓練-推論 gap：模型訓練時所有 frame 同噪聲等級，diagonal denoising 卻讓 frame 有不同噪聲，故需 latent partitioning 緩解。

autoregressive error accumulation / exposure bias 的訓練時解法譜系見 [CausVid 與 Self-Forcing](causvid-self-forcing.html)。
