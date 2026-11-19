---
slug: dllm-masked-schedule-dtc-theory
title: masked diffusion 的 inference schedule 與 dual total correlation 縮放
cluster: Diffusion Language Models
kind: topic
tags: [masked-diffusion, dual-total-correlation, total-correlation, inference-schedule, query-complexity]
arxiv: ["2609.20539", "2511.04647", "2510.25544", "2602.15008", "2602.20126", "2608.13520"]
sources: [2609.20539 introduction, 2609.20539 related work, arXiv titles]
related: ["dllm-parallelism-lineage-hub", "dllm-discrete-sampling-theory", "dllm-parallelism-separation-result"]
---
# masked diffusion 的 inference schedule 與 DTC 縮放

屬於 [dLLM 平行生成的理論脈絡總覽](dllm-parallelism-lineage-hub.html)。

## 量
- 分布 q 在 Σ^d 上的 **dual total correlation** DTC(q)：粗略而言，是分布總熵中不能由各座標的局部熵 H(Xi | X−i) 解釋的部分（本論文 §1.1）。DTC 可遠小於序列長度 d。
- 取樣誤差來自每步忽略座標間的條件相關；本論文以 conditional total correlation 描述。

## 前人結果（本論文 §1.3 的描述）
- Li & Cai（NeurIPS 2025 版，arXiv ID 未能驗證）：開始以資料分布的資訊論性質研究 masked diffusion 的 query 數。
- Chen, Cong & Li, *Optimal Inference Schedules for Masked Diffusion Models*（arXiv:2511.04647）與 Lavenant & Zanella, *Error Bounds and Optimal Schedules for Masked Diffusions with Factorized Approximations*（arXiv:2510.25544）：取樣誤差有以資料分布 information curve 表示的精確刻劃，可得 query 數隨 total correlation 與 dual total correlation 線性成長的界。Chen–Cong–Li 另給出以 MDS code 構造的 Ω(d) 下界。
- Dmitriev et al.（arXiv:2602.15008）、Zhao & Cai, *Adaptation to Intrinsic Dependence in Diffusion Language Models*（arXiv:2602.20126）：masking τ-leaping 可適應 total 與 dual total correlation。
- Wainwright 2026, *The data geometry of masking diffusion: Certified-optimal schedules via unmasking growth complexity*（arXiv:2608.13520）：本論文將其與上述工作並列為 masked diffusion 的 few-step 理論。

## 缺口
上述結果只針對 masked diffusion；uniform 與 Gaussian 沒有對應的 DTC 縮放。本論文 Theorem 1.1、1.2 補上，見 [本論文的分離結果](dllm-parallelism-separation-result.html)。
