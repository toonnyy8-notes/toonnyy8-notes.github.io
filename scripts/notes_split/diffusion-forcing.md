---
slug: diffusion-forcing
title: Diffusion Forcing：per-token 獨立噪聲的因果序列擴散
cluster: Diffusion & Generative Models
kind: topic
tags: [diffusion-forcing, causal, noise-as-masking, time-series, long-rollout, timegrad]
sources: [blockwise-smc-separation.md, causal-attention-diffusion.md]
arxiv: ["2407.01392", "2101.12072"]
related: ["long-sequence-consistency-overview", "causal-attention-diffusion-overview", "blockwise-smc-time-axis", "causvid-self-forcing"]
---
# Diffusion Forcing：per-token 獨立噪聲的因果序列擴散

屬於 [長序列一致性總覽](long-sequence-consistency-overview.html) 與 [Causal Attention × Diffusion 總覽](causal-attention-diffusion-overview.html)。

- **論文**：Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, Vincent Sitzmann，"Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion"，NeurIPS 2024，arXiv 2407.01392。

## 核心機制
- 訓練時對序列中每個 token（frame）指派「獨立、隨機的噪聲等級」，使模型能接受乾淨（零噪聲）的過去 frame 作為 context 而不產生分布偏移，達成「stable, arbitrarily long rollouts」。它統一了 next-token prediction（可變長度）與 full-sequence diffusion（可全局引導）。論文證明其優化的是「所有子序列聯合分布 likelihood 的變分下界」。實驗顯示能 rollout 超過訓練長度而 baseline 發散。
- **為何不需 AR 逐步解碼**：關鍵創新是「noise as masking」——每個 token 有獨立隨機噪聲等級，零噪聲=乾淨（已知），高噪聲=被遮罩（待生成）。因此模型可以同時是全序列擴散與下一 token 預測，不必逐 token 順序解碼，同時又能做長程 guidance（這是純 next-token 模型做不到的）。

## 架構
主要實作用的是**因果 RNN（GRU）骨幹**，非因果 Transformer。論文原文：「We implement DF for sequence generation as Causal Diffusion Forcing (CDF), in which future tokens depend on past ones via a causal architecture. We train the model to denoise all tokens of a sequence at once, with an independent noise level」，並註明「Our current causal implementation is based on a small RNN」。Transformer 版本（含 causal attention mask）僅在 Appendix B.1 作為未來工作討論。架構細節：U-Net 的輸出餵入 GRU，GRU 隱藏狀態 z_{t-1} 攜帶過去資訊，以馬可夫方式更新；非空間資料則以 residual MLP 取代 U-Net。

## 屬性
- 是否重訓：**是**（新訓練範式）。
- 能否 posterior sampling：架構原生支援 guidance，理論上可導入 likelihood 引導，但原文聚焦生成與 planning，未做 inverse problem。
- 計算成本：訓練成本與標準 diffusion 相當；推論可用 causal 架構 + KV cache。

## 實驗結果
- **時間序列（CRPS_sum，越低越好，五次運行均值±標準差）**：Exchange 0.003±0.001、Solar 0.289±0.002、Electricity 0.023±0.001、Traffic 0.040±0.004、Taxi 0.075±0.002、Wikipedia 0.085±0.007。
- **與 AR baseline 比較**：主要對手 TimeGrad（Rasul et al., ICML 2021, arXiv:2101.12072；AR 擴散，以 RNN（LSTM/GRU）逐時步自回歸條件化，teacher forcing）：Exchange 0.006、Solar 0.287、Electricity 0.021、Traffic 0.044、Taxi 0.114、Wikipedia 0.049。Diffusion Forcing 在多數資料集勝出，但與 ScoreGrad sub-VP SDE「整體打平」，且在 Wikipedia 上僅排第四。
- **規劃與機器人**：規劃任務（D4RL Maze2D/Multi2D）Diffusion Forcing 單任務平均 reward 141.7 對 Diffuser 119.5、多任務 146.2 對 129.4；真實 Franka 機器人模仿學習成功率 80%（遮擋/噪聲下降至 76%，對照 next-frame diffusion 48%）。

## 局限
- 主實作仍是 RNN（Transformer 因果版僅為未來工作）；視訊實驗無量化指標（僅定性宣稱可穩定 rollout 至 1000+ frames，訓練 72 frames 可外推至 180 甚至 2000 frames）；CSDI 並非其 baseline；作者自承時間序列非其核心應用，episode reward 指標可能對 Diffuser 有利而不完全反映規劃品質。
- 在 token/frame 軸上仍有滾動推理，學界常歸為 AR-diffusion 混合（見 [Causal Attention × Diffusion 總覽](causal-attention-diffusion-overview.html) 的界定）。

## 應用建議
- **時間序列方向（明確的研究空白）**：若需因果，把 Diffusion Forcing 的因果 RNN 骨幹替換為 causal-masked Transformer（其 Appendix B.1 已勾勒方向但未實作）是一個明確的貢獻點。Benchmark 用 CRPS_sum，對照 TimeGrad（AR）、ScoreGrad、CSDI。
- **轉移到音訊分離**：per-frame 獨立噪聲天然適合把「已分離的前段」當零噪聲 context，但需重訓 prior，與 zero-shot 目標衝突（見 [沿時間軸的 block-wise SMC](blockwise-smc-time-axis.html)）。
- 在 exposure bias 解法譜系中的位置：Diffusion Forcing（乾淨 context 零噪聲餵入避免偏移）→ CausVid → Self-Forcing（見 [CausVid 與 Self-Forcing](causvid-self-forcing.html)）。
