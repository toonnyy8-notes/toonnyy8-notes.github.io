---
slug: rswp-lineage-hub
title: Reasoning with Sampling 的研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [power-sampling, MCMC, RLVR, distribution-sharpening]
arxiv: ["2510.14901", "2501.12948", "2402.03300", "2506.02355", "2506.10947", "2504.13837", "2509.06941", "2505.22660", "2505.19590", "2404.17546", "2503.02819", "2502.00921"]
sources: ["arXiv:2510.14901 §1, §2, §4", "arXiv abstracts of cited works"]
related: ["rswp-rlvr-grpo-background", "rswp-distribution-sharpening-debate", "rswp-label-free-likelihood-rewards", "rswp-power-vs-low-temperature", "rswp-metropolis-hastings-autoregressive", "rswp-annealed-tempered-sampling", "rswp-power-sampling-algorithm-results"]
---
# Reasoning with Sampling 的研究脈絡總覽

arXiv:2510.14901（Karan 與 Du，2025）提出以 base model 自身的概似度做 MCMC 取樣，不經訓練就取得接近 GRPO 的單次作答推理表現。本文依閱讀順序列出理解該論文需要的脈絡。

1. 動機來自 RLVR 與 GRPO：[RLVR 與 GRPO 背景](rswp-rlvr-grpo-background.html)。
2. RL 後訓練是否只是把 base model 的分布「變尖」，以及 pass@k 的證據：[分布銳化之爭](rswp-distribution-sharpening-debate.html)。
3. 同樣使用模型內部概似度或信心的訓練方法，與本文的差別：[無外部標籤的概似度獎勵](rswp-label-free-likelihood-rewards.html)。
4. 本文的目標分布 p^α 與低溫取樣的差異：[冪次分布與低溫取樣](rswp-power-vs-low-temperature.html)。
5. 取樣 p^α 所用的 Metropolis-Hastings 與自回歸 LLM 上的 MCMC 前作：[Metropolis-Hastings 與自回歸 MCMC](rswp-metropolis-hastings-autoregressive.html)。
6. 退火／回火取樣在統計物理與擴散模型中的來源：[退火與回火取樣](rswp-annealed-tempered-sampling.html)。
7. 演算法 1 的逐步細節、實驗數字與成本：[冪次取樣演算法與結果](rswp-power-sampling-algorithm-results.html)。

已有的相關筆記：[GRPO 與規則式結果獎勵](r1-grpo-outcome-rewards.html)、[思維鏈、驗證器與推論時運算](r1-cot-and-test-time-compute.html)。年代順序見同資料夾的 timeline.json。
