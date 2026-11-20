---
slug: rswp-power-sampling-algorithm-results
title: 冪次取樣演算法 1、超參數、成本與實驗結果
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [power-sampling, algorithm, MATH500, HumanEval, GPQA, AlpacaEval]
arxiv: ["2510.14901", "2402.03300"]
sources: ["arXiv:2510.14901 §4.3, §5, Table 1, Eq. 12"]
related: ["rswp-metropolis-hastings-autoregressive", "rswp-power-vs-low-temperature", "rswp-distribution-sharpening-debate", "rswp-rlvr-grpo-background"]
---
# 冪次取樣演算法 1、超參數、成本與實驗結果

演算法 1（2510.14901 §4.3）：序列長度 T 切成大小 B 的區塊，定義前綴目標 π_k(x0:kB) ∝ p(x0:kB)^α。對每個 k，先以 p_prop 把前綴延伸 B 個 token 作為初值，再執行 N_MCMC 步 MH（隨機位置重取樣、依式 9 接受或拒絕），完成後固定前綴進入下一段。整個過程為單次作答，接受與否只依 base model 概似度。預期生成 token 數 E_tokens ≈ N_MCMC T² / (4B)（式 12）。

設定：T_max = 3072，B = 192，α = 4.0，p_prop 為 base model 以溫度 1/α 取樣；AlpacaEval 2.0 的提案溫度改為 0.5。模型為 Qwen2.5-Math-7B、Qwen2.5-7B、Phi-3.5-mini-instruct；GRPO 基準以 MATH 訓練集後訓練（實作見 [RLVR 與 GRPO](rswp-rlvr-grpo-background.html)）。

Table 1（Qwen2.5-Math-7B；Base / 低溫 / 冪次取樣 / GRPO）：MATH500 0.496 / 0.690 / 0.748 / 0.785；HumanEval 0.329 / 0.512 / 0.573 / 0.537；GPQA 0.278 / 0.353 / 0.389 / 0.399；AlpacaEval 2.0 1.61 / 2.09 / 2.88 / 2.38。

其他觀察：N_MCMC 到 10 前準確率穩定上升；α ≥ 2.0 後表現穩定；N_MCMC = 10、T = 679、B = 192 時成本為標準推論的 8.84 倍；pass@k 在 k > 1 優於 GRPO（見 [分布銳化之爭](rswp-distribution-sharpening-debate.html)）。MH 的理論部分見 [Metropolis-Hastings 與自回歸 MCMC](rswp-metropolis-hastings-autoregressive.html)，目標分布的動機見 [冪次分布與低溫取樣](rswp-power-vs-low-temperature.html)。
