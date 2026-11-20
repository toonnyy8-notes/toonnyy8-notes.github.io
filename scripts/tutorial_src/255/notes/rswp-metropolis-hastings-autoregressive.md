---
slug: rswp-metropolis-hastings-autoregressive
title: Metropolis-Hastings 與自回歸 LLM 上的 MCMC 前作
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [MCMC, Metropolis-Hastings, sequential-Monte-Carlo, autoregressive]
arxiv: ["2510.14901", "2404.17546"]
sources: ["arXiv:2510.14901 §2, §4.2", "arXiv abstract 2404.17546"]
related: ["rswp-power-vs-low-temperature", "rswp-power-sampling-algorithm-results", "control-as-inference-guided-sampling"]
---
# Metropolis-Hastings 與自回歸 LLM 上的 MCMC 前作

Metropolis-Hastings（MH）出自 Metropolis 等人（1953，Journal of Chemical Physics，無 arXiv 版本）。它只需要未正規化的目標密度：以提案分布 q(x|x^i) 產生候選 x，以機率 A = min(1, p^α(x) q(x^i|x) / (p^α(x^i) q(x|x^i))) 接受，否則保留原狀態。正規化常數在比值中相消。論文引用 Neal（1993）：只要 q 不可約且非週期，n 夠大時鏈收斂到目標分布。

2510.14901 §4.2 的提案是隨機重取樣：以均勻機率選一個位置 t，從 t 起以 p_prop 重新生成後半段。因每次都有機率從序列開頭重取樣，此提案不可約且非週期；反向機率 q(x^i|x) 可由對稱性計算。

LLM 上結合 MCMC 與自回歸取樣的前作（論文 §2）：

- Zhao 等人（2404.17546，Twisted SMC）：學習中間價值函數，在 Sequential Monte Carlo 框架下維持多個候選序列，依預期未來獎勵更新。
- Faria 等人（2024，QUEST，NeurIPS，arXiv ID 未查證）：以 MH 迭代重取樣並依預期獎勵更新。論文表示本文方法與此最相近，差別是目標分布完全由 base LLM 指定，不需外部獎勵。

論文也承認 MCMC 的主要缺點是可能有指數級混合時間，高維序列空間更嚴重，所以演算法 1 採分段前進，見 [冪次取樣演算法與結果](rswp-power-sampling-algorithm-results.html)。把獎勵最大化視為條件取樣的另一種觀點見 [控制即推論](control-as-inference-guided-sampling.html)。
