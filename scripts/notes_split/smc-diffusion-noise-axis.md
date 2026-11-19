---
slug: smc-diffusion-noise-axis
title: 沿 diffusion 噪聲軸的 SMC 後驗取樣：FPS、MCGdiff、TDS、SMCDiff
cluster: Diffusion & Generative Models
kind: topic
tags: [smc, particle-filtering, posterior-sampling, feynman-kac, twisted-diffusion, inverse-problems]
sources: [blockwise-smc-separation.md]
arxiv: ["2308.14409", "2308.07983", "2306.17775", "2206.04119", "2502.05908", "2507.15716"]
related: ["long-sequence-consistency-overview", "blockwise-smc-time-axis"]
---
# 沿 diffusion 噪聲軸的 SMC 後驗取樣：FPS、MCGdiff、TDS、SMCDiff

屬於 [長序列一致性總覽](long-sequence-consistency-overview.html)。

## 兩種序貫性
diffusion 有兩個時間軸——(a) 去噪軸（noise level t: T→0），(b) 資料/序列軸（audio time n）。本筆記的方法全部沿 (a)；沿 (b) 的序貫（autoregressive 生成、chunk-by-chunk、FIFO 佇列）**都是單軌跡確定性 rollout，不做粒子重採樣**。

## 沿噪聲軸的 SMC 方法
- **FPS**（Dou & Song, ICLR 2024, arXiv:2308.14409）：證明「Bayesian posterior sampling 是 Bayesian filtering 的特例」，把去噪鏈 t=N→0 當 state-space 濾波問題，粒子數→∞ 時漸近精確。注意：不同來源對 FPS 編號略有出入（此處採 2308.14409 / OpenReview id=tplXNcHZs1），引用前建議再核對。
- **MCGdiff**（Cardoso et al., ICLR 2024 oral, arXiv:2308.07983）：把恢復問題寫成「從前向 diffusion 改編的 Feynman-Kac model」，SMC 沿噪聲等級求解。
- **TDS**（Wu, Trippe et al., NeurIPS 2023, arXiv:2306.17775）：用 twisting function 做 reconstruction-guidance proposal，SMC 沿去噪步驟。
- **SMCDiff**（Trippe et al., ICLR 2023, arXiv:2206.04119）：同理。
- **LD-SMC**（arXiv:2502.05908）：提到少粒子 + GPU 平行以緩解粒子退化。

## 相反關係的工作
DiffPF（arXiv:2507.15716）是把 diffusion 放進傳統粒子濾波的狀態更新，非「用 SMC 沿時間重採樣 diffusion 生成的訊號塊」。

## 關鍵洞見
這兩條線（沿噪聲軸的 SMC、沿時間軸的序貫生成）從未被結合。沿時間軸做 SMC（粒子=不同的分離假設，沿 audio 時間塊傳播並依與觀測/前段一致性重加權），用 diffusion prior 作 proposal，是空白。具體提案見 [沿時間軸的 block-wise SMC](blockwise-smc-time-axis.html)。
