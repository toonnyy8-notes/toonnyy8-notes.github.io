---
slug: idql-implicit-policy-and-critic-choice
title: 廣義 IQL 的隱含 actor 與 critic 損失選擇
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [iql, implicit-policy, expectile, quantile, convex-loss, actor-critic]
arxiv: ["2304.10573", "2110.06169", "2301.02328", "2303.15810"]
sources: ["arXiv:2304.10573 §3, §4.1, 附錄 B、C", "arXiv:2110.06169 摘要（IQL 的 expectile 價值學習）"]
related: ["expectile-regression-value-learning", "awr-advantage-weighted-policy-extraction", "in-sample-implicit-value-regularization-eql-ivr", "diffusion-behavior-model-policy-extraction", "idql-evaluation-and-online-finetuning"]
---
# 廣義 IQL 的隱含 actor 與 critic 損失選擇

## 出發點

IQL 的價值損失為 L_V(ψ) = E[L_τ²(Q(s,a) − V(s))]，其中 L_τ²(u) = |τ − 1(u<0)|u²（expectile）。IQL 原論文指出 τ 趨近 1 時回到 Q-learning，但 0.5 < τ < 1 時 Q 函數對應哪個策略沒有說明（IDQL §4）。細節見 [Expectile regression 與價值學習](expectile-regression-value-learning.html)。

## 推廣：任意凸損失 f

IDQL 把價值目標寫成

V*(s) = argmin_V E_{a~μ(a|s)}[ f(Q(s,a) − V(s)) ]，μ 為資料集的行為策略。

當 f(u) = L_τ²(u) 時即為 IQL。再依 actor-critic 慣例定義 V(s) = E_{a~π_imp}[Q(s,a)]。

**定理 4.1**：對每個狀態 s，若 f 為凸且 f'(0)=0，上式的解也是 argmin_V E_{a~π_imp}[(Q(s,a) − V(s))²] 的解，其中

π_imp(a|s) ∝ μ(a|s) · w(s,a)，w(s,a) = |f'(Q(s,a) − V*(s))| / |Q(s,a) − V*(s)|。

證明（附錄 B）：對 V 微分並令一階條件為 0，把 f' 寫成 |f'|·sign，即可把期望改寫成對 π_imp 的期望。結論是隱含 actor 等於行為策略乘上 critic 算出的重要性權重。f 的形式決定 π_imp 偏離 μ 的方式。

## 三種損失

| 損失 f | 權重 w(s,a) | 超參數的作用 |
| --- | --- | --- |
| Expectile：L_τ²(u) | |τ − 1(Q < V_τ)| | τ 增加，偏離行為策略的程度增加 |
| Quantile：|τ − 1(u<0)|·|u| | |τ − 1(Q < V_τ)| / |Q − V_τ| | τ 增加，外推程度增加；樣本聚集在分位數附近 |
| Exponential（linex）：α·exp(u) − αu | α·|exp(α(Q − V)) − 1| / |Q − V| | α 平衡匹配行為策略與最佳化 critic |

指數型的解為 V_exp(s) = (1/α) log Σ_a exp(αQ(s,a) + log μ(a|s))，是 AWR 策略 π_awr ∝ exp(αQ + log μ) 的正規化項，等同於施加 KL 散度限制（見 [Advantage-weighted regression 與策略抽取](awr-advantage-weighted-policy-extraction.html)）。此目標與 Garg et al. 及 Xu et al. 的 critic 目標相同，見 [EQL 與 IVR](in-sample-implicit-value-regularization-eql-ivr.html)。

## 三種隱含 actor 的差異

論文在三叢集 bandit 上的比較（圖 2）：expectile actor 涵蓋較寬的結果範圍，quantile actor 緊密集中，exponential actor 在最大值附近有部分涵蓋。D4RL 消融（§6.3）顯示 expectile 與 quantile 搭配 argmax 抽取表現良好，exponential 不穩定、表現較差；整體以 expectile 搭配 greedy 抽取最強。

## 對策略抽取的意涵

π_imp 是複雜且可能多峰的重新加權分布，單峰高斯加 AWR 難以近似（圖 1）。critic 不知道 actor 的近似限制，因此需要表達力夠強的抽取方式，見 [以擴散行為模型做策略抽取](diffusion-behavior-model-policy-extraction.html)。實驗結果見 [IDQL 的實驗與線上微調](idql-evaluation-and-online-finetuning.html)。
