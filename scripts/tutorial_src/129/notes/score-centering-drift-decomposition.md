---
slug: score-centering-drift-decomposition
title: Score centering：drift 與 signal 的分解與 top-k 實作
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [score-centering, drift, covariance, top-k, implementation]
arxiv: ["2609.20807"]
sources: ["arXiv:2609.20807 §4, §4.1, §5, Appendix A"]
related: ["policy-gradient-baselines-score-function", "importance-sampling-corrections-llm-rl", "sft-vs-rl-offline-negative-rewards", "training-inference-mismatch-sources", "ppo-grpo-dapo-llm-rl-objectives"]
---
# Score centering：drift 與 signal 的分解

**記號。** 在固定 prefix 上，sampler 機率 $q_v$，trainer 機率 $p_v$，score $s_v=\nabla_\theta\log p_v$，sampler 下的期望 score $\bar s=\sum_v q_v s_v$。

**分解（Eq. 3）。** 利用共變異恆等式 $E[AB]=E[A]E[B]+\mathrm{Cov}(A,B)$：

$E_q[R\,s_{y_t}] = E_q[R]\,\bar s + \mathrm{Cov}_q(R, s_{y_t})$

第一項為 drift，只透過平均獎勵依賴獎勵，不含哪些 rollout 成功的資訊；第二項為 signal。on-policy 時 $\bar s=0$，drift 消失。本文指出 $\bar s$ 是以 sampler 為老師的交叉熵損失的負梯度，所以 drift 等於每個 prefix 上把 trainer 往 sampler 拉近，強度正比於該 prefix 的期望獎勵。因 sampler 是 trainer 的有偏拷貝且每步同步，誤差在回饋迴路中累積。

**Group centering 不能消除 drift。** 優勢在同一 prompt 的 rollout 間總和為 0，但 drift 發生在個別 prefix，而 prefix 的期望優勢不為 0。

**Score centering（Eq. 4、5）。** 令 $\tilde s_{y_t}=s_{y_t}-\bar s$。兩個期望在同一分佈下，故 drift 在每個 prefix 精確抵消，期望更新為 $\mathrm{Cov}_q(R,s_{y_t})$，與 on-policy 更新的差別只在共變異是在 sampler 分佈下量測。無 TIM 時修正項為 0。

**與 IS 的比較。** IS 也能消除 drift，但乘上隨機的乘法項而增加變異，實務上的 clip 又重新帶入 drift。Score centering 是加法修正，給定 prefix 下確定，可精確計算，因此可與 IS 組合（Appendix A.2：減去加權 score 的期望，$\alpha=\rho f(1/\rho)$）。

**實作（§4.1，Appendix A.1、A.3）。** 儲存完整詞彙分佈不可行（Qwen3 詞彙約 152K，1024 條 32K 長度 rollout 約需 20TB）。僅記錄 sampler 的 top-k logprob（k = 128），尾端以 trainer 分佈乘上尾端質量比 $\rho$ 補足，損失為 $L=-R\log p_{y_t}-\sum_{v\in H}\mathrm{sg}[q_v-\rho p_v]\log p_v$。k = 128 與 k = 32 皆與完整詞彙的結果相當，k = 128 的耗時在基準方法的 1% 內。

**結果。** Qwen3 0.6B 至 30B 的實驗中，單獨使用 SC 在量化下與 IS 方法相當或較佳，差距隨 TIM 增大而擴大；staleness 設定下 TIS 或 MIS 加 SC 最佳。
