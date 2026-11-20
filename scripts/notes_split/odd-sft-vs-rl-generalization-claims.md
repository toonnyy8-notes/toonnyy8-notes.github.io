---
slug: odd-sft-vs-rl-generalization-claims
title: SFT 與 RL 的泛化比較，以及 on-policy 資料對泛化的說法
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [generalization, SFT-vs-RL, pass@k, on-policy-data]
arxiv: ["2609.35259", "2501.17161", "2509.25123", "2509.26313", "2602.12222", "2504.13837"]
sources: ["2609.35259 §1, §4.2", "arXiv abstract 2501.17161", "arXiv abstract 2509.25123", "arXiv abstract 2509.26313", "arXiv abstract 2602.12222", "arXiv abstract 2504.13837"]
related: ["odd-lineage-hub", "odd-learning-rate-adam-clipping-rlvr", "odd-on-policy-forgetting-sparsity-claims", "rlvr-grpo-credit-assignment", "sft-vs-rl-offline-negative-rewards"]
---
# SFT 與 RL 的泛化比較

屬於 [蒸餾動態研究的研究脈絡總覽](odd-lineage-hub.html)。論文（arXiv:2609.35259）引這些工作作為「on-policy 資料改善泛化」的論據（Chu et al., 2025；Zhang et al., 2026b；Yuan et al., 2026；Ming et al., 2025）。

- Chu et al.（arXiv:2501.17161，SFT Memorizes, RL Generalizes）：摘要指出以結果獎勵訓練的 RL 能泛化到規則與視覺的新變體，SFT 傾向記憶訓練資料。
- Yuan et al.（arXiv:2509.25123）：摘要指出以合成的字串轉換任務為框架，提供 LLM 在 RL 中透過組合既有技能習得新技能的證據。
- Ming et al.（arXiv:2509.26313，One-Token Rollout）：摘要主張 SFT 與 RL 的泛化差距來自 SFT 使用固定資料、RL 使用 on-policy 資料，並提出每個 token 取樣多個候選、以監督資料的真實 token 引導的 SFT 演算法。
- Zhang et al.（arXiv:2602.12222）：摘要主張 SFT 與 RL 的差距主要由 RL 的 on-policy 資料造成，提出分布判別理論與兩種讓 SFT 更接近 on-policy 的技術。
- Yue et al.（arXiv:2504.13837）：摘要指出以大 k 的 pass@k 檢視時，RLVR 訓練的模型未產生基礎模型之外的根本新推理模式。本論文引它說明 pass@1 的進步不必然對應較高 pass@k。

## 本論文的檢驗
論文發現 on-policy 資料使 Countdown 較難變體上的表現在兩種 KL 方向下都較好，但此優勢在後續 RLVR 後不穩定地持續（見 [學習率、Adam、梯度裁剪與 RLVR](odd-learning-rate-adam-clipping-rlvr.html)）。RLVR 本身的背景見 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)。

遺忘面的說法見 [On-policy 學習的遺忘與更新稀疏說法](odd-on-policy-forgetting-sparsity-claims.html)；SFT 與 RL 的目標差異見 [SFT 與 RL 的差異](sft-vs-rl-offline-negative-rewards.html)。
