---
slug: rswp-power-vs-low-temperature
title: 冪次分布 p^α 與低溫取樣為何不同
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [power-distribution, low-temperature, sharpening, critical-windows]
arxiv: ["2510.14901", "2502.00921"]
sources: ["arXiv:2510.14901 §4.1, Proposition 1, Example 1", "arXiv abstract 2502.00921"]
related: ["rswp-distribution-sharpening-debate", "rswp-metropolis-hastings-autoregressive", "rswp-annealed-tempered-sampling"]
---
# 冪次分布 p^α 與低溫取樣為何不同

LLM 以 p(x0:T) = Π p(xt | x<t) 定義序列分布。2510.14901 §4.1 以 p^α（α ≥ 1）作為銳化的目標：若 p(x) > p(x')，則 p(x)^α / p(x')^α > p(x)/p(x')，高概似度序列被相對加重。

低溫取樣（τ = 1/α）對每一步的條件分布取冪次再正規化。論文的 Proposition 1 證明它不等於 p^α，差別在於：

- p^α 在第 t 步的相對權重正比於「未來路徑概似度的 α 次方之和」，即 Σ p(x0,…,xT)^α（式 7，exponent 在 sum 內）。
- 低溫取樣正比於「未來路徑概似度之和的 α 次方」，即 (Σ p(x0,…,xT))^α（式 8，exponent 在 sum 外）。

Example 1：序列 aa、ab、ba、bb 的概似度為 0.00、0.40、0.25、0.25，α = 2。p^α 偏好 x0 = a（0.160 對 0.125），低溫取樣偏好 x0 = b（0.250 對 0.160）。論文的觀察是：p^α 加重「未來路徑少但概似度高」的 token，低溫取樣加重「未來路徑多但概似度低」的 token。

論文以此連到 critical windows 或 pivotal tokens：少數 token 對答案正確性影響極大，Li、Karan、Chen（2502.00921，Blink of an eye）顯示尖銳的 critical window 與推理失敗高度相關。論文稱 p^α 隱含偏向為未來高概似度 token 做規劃。

因 p^α 沒有逐 token 的封閉形式，需要 [Metropolis-Hastings](rswp-metropolis-hastings-autoregressive.html) 近似取樣。冪次與回火的來源見 [退火與回火取樣](rswp-annealed-tempered-sampling.html)。
