---
slug: dlm-remasking-self-correction
title: DLM 的 remask 式迭代修正與自我修正
cluster: Diffusion Language Models
kind: topic
tags: [remasking, self-correction, parallel-decoding, inference-time-scaling]
arxiv: ["2502.03540", "2503.00307", "2510.01384", "2511.05562", "2512.15596"]
sources: ["paper:2512.15596 附錄 B 與第 2 節對該工作的描述", "arxiv.org/abs 頁面（標題、作者、年份）"]
related: ["corrective-dlm-overview"]
---
# DLM 的 remask 式迭代修正與自我修正

屬於 [CDLM 研究脈絡總覽](corrective-dlm-overview.html)。

## 論文第 2.2 節的流程
每步前向一次，遮罩位置填入 $\arg\max_v p_\theta(v\mid z^{(t)})$；位置 $ 的信心 $ 是模型對目前該位置 token 的機率；\le\tau$ 的位置被重新遮罩。可見 token 不會被直接改寫，須先 remask 再重新預測，稱為一個 refinement step。

## 前人工作（依論文附錄 B.2 的描述）
- **Path Planning（P2）**：Peng et al.（2025a；arXiv 2502.03540），*Path Planning for Masked Diffusion Model Sampling*，8 位作者。以外部模型（如 BERT）或去噪器本身挑選要重新取樣的 token。
- **ReMDM**：Wang et al.（2025；arXiv 2503.00307），*Remasking Discrete Diffusion Models with Inference-Time Scaling*，4 位作者。以 token 層級信心引導推論時 remask，不需額外訓練。CDLM 論文用它作為解碼變體做對照。
- **Lee et al.**（2025；arXiv 2511.05562），*Effective Test-Time Scaling of Discrete Diffusion through Iterative Refinement*，5 位作者。論文描述為以 reward model 引導加噪與去噪轉移。
- **PRISM**：Kim et al.（2026，論文引用；arXiv 2510.01384 為同題預印本，標題 *Fine-Tuning Masked Diffusion for Provable Self-Correction*，7 位作者）。論文描述為加入內部預測頭，估計每個 token 應被 remask 的機率，訓練資料由 MDM 生成的 token 與真值比較而得。

## 與 CDLM 的差異
上述方法在推論端或以額外模組處理「該改哪裡」。CDLM 論文的說法是：它直接修改訓練目標，不改架構、不另建資料，與 PRISM 處理的面向互補，原則上可以併用。論文附錄 M.1 在相同基底模型、資料與訓練步數下比較兩者。
