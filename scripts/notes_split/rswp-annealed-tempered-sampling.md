---
slug: rswp-annealed-tempered-sampling
title: 退火與回火取樣：從統計物理到擴散模型的推論時取樣
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [annealing, tempering, diffusion, inference-time-sampling]
arxiv: ["2510.14901", "2503.02819"]
sources: ["arXiv:2510.14901 §2", "arXiv abstract 2503.02819"]
related: ["rswp-power-vs-low-temperature", "rswp-metropolis-hastings-autoregressive", "ddpm-training-and-sampling-recap"]
---
# 退火與回火取樣：從統計物理到擴散模型的推論時取樣

2510.14901 §2 指出，從 p^α 取樣在統計物理與 Monte Carlo 文獻中稱為從退火（annealed）或回火（tempered）分布取樣，引用 Neal（1998，Annealed importance sampling）。傳統 MCMC 以退火避免取樣時的 mode collapse，較準確地從複雜多峰分布取樣（論文引用 Łatuszyński 等人，2025）。物理科學中的多峰取樣需求另引 Sambridge（2014，parallel tempering）。

在擴散模型方面，論文列出多項把預訓練模型導向「tilted distribution」的推論時方法（Du 等人 2023、Kim 等人 2025、Karan 等人 2025、Wang 等人 2025、Kong 等人 2025、Zhang 等人 2025），其中 Du、Wang、Kim 等人建構一串退火分布，使分布從擴散的 base 分布平順過渡到 tilted 分布。另有工作刻意以 α > 1 取樣 p^α 取得較高品質樣本，包括 Skreta 等人（2503.02819，Feynman-Kac Correctors）與 Xu 等人（2025）；論文指出此作法在生成較可設計（designable）的蛋白質上特別受歡迎（Geffner 等人，2025）。

本文把同一想法移到 LLM 的序列空間：目標同為 p^α，但以自回歸結構切成前綴分布序列逐段取樣。差異與細節見 [冪次分布與低溫取樣](rswp-power-vs-low-temperature.html)、[Metropolis-Hastings 與自回歸 MCMC](rswp-metropolis-hastings-autoregressive.html)。擴散模型本身的取樣背景見 [DDPM 訓練與取樣回顧](ddpm-training-and-sampling-recap.html)。

本筆記中除 2503.02819 外，其餘引用只依本文 §2 的描述，未逐一查證 arXiv。
