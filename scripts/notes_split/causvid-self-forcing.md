---
slug: causvid-self-forcing
title: CausVid 與 Self-Forcing：雙向擴散蒸餾為因果自回歸串流生成
cluster: Diffusion & Generative Models
kind: topic
tags: [causvid, self-forcing, causal-attention, kv-cache, distillation, exposure-bias, streaming-video]
sources: [blockwise-smc-separation.md, causal-attention-diffusion.md]
arxiv: ["2412.07772", "2506.08009", "2509.25161", "2603.11647", "2604.23632"]
related: ["long-sequence-consistency-overview", "causal-attention-diffusion-overview", "diffusion-forcing", "noncausal-diffusion-baselines"]
---
# CausVid 與 Self-Forcing：雙向擴散蒸餾為因果自回歸串流生成

屬於 [長序列一致性總覽](long-sequence-consistency-overview.html) 與 [Causal Attention × Diffusion 總覽](causal-attention-diffusion-overview.html)。前身見 [Diffusion Forcing](diffusion-forcing.html)。

## CausVid（Yin et al., CVPR 2025, arXiv:2412.07772）
- 核心機制：把雙向 diffusion teacher 蒸餾成 causal（block-wise causal attention）autoregressive student，用 asymmetric distillation + DMD 把 50 步蒸成 4 步。
- 相較雙向 baseline，初始延遲從 219.2 秒降到 1.3 秒（160× 加速）、吞吐量從 0.6 提升到 9.4 FPS（16×），達成串流生成。在全部 946 條 VBench-Long prompt、16 項指標上取得「最高總分 84.27」，為當時所有已驗證 VBench 提交中第一。
- 關鍵是「asymmetric distillation」緩解 autoregressive error accumulation，使短 clip 訓練能長時生成。
- 是否重訓：**是**（需蒸餾）。能否 posterior sampling：可 zero-shot 做 image-to-video（因 causal 設計），但非 inverse-problem posterior sampling。計算成本：訓練貴（蒸餾），推論極快（few-step + KV cache）。

## Self-Forcing（Huang et al., NeurIPS 2025 Spotlight, arXiv:2506.08009）
- 核心機制：解決 exposure bias——訓練時就用「自己生成的前綴」做 autoregressive rollout（帶 KV cache），而非用 ground-truth context，並用 video-level holistic loss 監督整段。
- 指出 CausVid 的缺陷：其 student 在 diffusion forcing 下用 ground-truth context，導致 concat 後不成有效影片。
- 將雙向 DiT 蒸餾為**因果注意力遮罩** + rolling KV cache 的自回歸擴散，達到次秒延遲、單 GPU 即時串流，品質匹配甚至超越較慢的非因果擴散。基於 Wan2.1-T2V-1.3B + Flow Matching；初始化沿用 CausVid 的 causal attention masking 微調，訓練時 rollout 自生成 context 以縮小 train-test gap。
- 是否重訓：**是**。能否 posterior sampling：否（生成導向）。計算成本：訓練用 few-step + stochastic gradient truncation 平衡；推論即時。
- 後繼：Causal Forcing、Rolling Forcing（arXiv:2509.25161）、LongLive 等一系列（多仍在 arXiv 階段）。

## exposure bias 解法譜系
問題：exposure bias——訓練用 ground-truth context，推論用自己的不完美輸出，誤差逐段放大。解法譜系：Diffusion Forcing（乾淨 context 零噪聲餵入避免偏移）→ CausVid（asymmetric distillation）→ Self-Forcing（訓練時就自 rollout + video-level loss）→ rollout/self-resampling training。

## 音視訊串流延伸
- **OmniForcing**（arXiv 2603.11647）、**Hallo-Live**（arXiv 2604.23632）：音視訊串流，使用 asymmetric block-causal alignment、Audio Sink Token、Future-Expanding Attention（讓 video block 存取同步音訊 + 短程未來音素線索以解決因果生成的發音延遲）。

## 建議
若研究動機是即時/串流，因果是剛需，採 Self-Forcing 式的「雙向教師蒸餾為因果學生 + rolling KV cache」是當前最強範式，可遷移到音訊 latent。若動機僅是離線生成品質，**不要**引入因果注意力（見 [非因果擴散對照組](noncausal-diffusion-baselines.html)）。
