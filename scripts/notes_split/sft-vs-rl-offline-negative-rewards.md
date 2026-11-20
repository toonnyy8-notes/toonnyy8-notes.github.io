---
slug: sft-vs-rl-offline-negative-rewards
title: SFT 與 RL 的差異：離線負獎勵不穩定，線上全正獎勵不穩定
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [SFT, distillation, negative-gradient, reward-modes, offline-vs-online]
arxiv: ["2609.20807", "1503.02531", "2501.12948", "2505.18830", "2506.01347"]
sources: ["arXiv:2609.20807 §3, Figure 2", "arXiv abstract 1503.02531", "arXiv abstract 2505.18830", "arXiv abstract 2506.01347"]
related: ["score-centering-drift-decomposition", "knowledge-distillation-foundations", "on-policy-distillation-imitation", "rlvr-grpo-credit-assignment"]
---
# SFT 與 RL 的差異

本文（§3）的出發點：SFT 在遠比 TIM 嚴重的不一致下仍然穩定，例如以另一個模型離線產生的資料訓練，不需要 importance sampling 修正（Hinton et al., 2015，arXiv 1503.02531；DeepSeek-AI, 2025，arXiv 2501.12948）。Policy gradient 在二元獎勵、優勢為 +1/0、且完全離線時等同 SFT。因此獎勵模式與 staleness 是 SFT 與 RL 的兩個關鍵差異。

**實驗（Figure 2，Qwen3-1.7B，Countdown）。** 離線訓練只有 +1/0 獎勵穩定；線上且有 TIM 時，group-centered 獎勵最穩定，+1/0 最不穩定。本文據此假設存在兩個獨立機制。

**離線負獎勵。** 本文不深入探討，引用 Ren & Sutherland（2025）：負獎勵無界，加上分佈變尖（sharpening），log 機率可發散至 $-\infty$。本文另引用 Deng et al.（arXiv 2505.18830，其摘要指出 GRPO 中負梯度造成 Lazy Likelihood Displacement，即正確回應的似然幾乎不增或下降）以及 Zhu et al.（arXiv 2506.01347，其摘要指出僅用負樣本訓練也可以很有效，on-policy 時負梯度攜帶有用訊號）。

**線上全正獎勵。** 本文指出此設定下的訓練本身是一種蒸餾：trainer 擬合來自 sampler（自身的有偏拷貝）的成功 rollout，而 sampler 每步由 trainer 更新權重，與離線蒸餾的差異只在老師隨學生移動。這是第 4 節 drift 分析的動機，見 [Score centering：drift 與 signal 的分解](score-centering-drift-decomposition.html)。蒸餾的基本形式見 [知識蒸餾基礎](knowledge-distillation-foundations.html) 與 [On-policy distillation 與模仿學習](on-policy-distillation-imitation.html)。
