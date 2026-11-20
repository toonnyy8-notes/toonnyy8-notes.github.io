---
slug: mk-verifiers-and-reference-based-benchmarks
title: 驗證器與參考式基準（GSM8K 驗證器、VerifyBench、規則式驗證器的限制）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [verifier, GSM8K, VerifyBench, rule-based-verifier, reward-hacking]
arxiv: ["2507.08794", "2110.14168", "2505.22203", "2505.15801", "2410.07985", "2408.15240"]
sources: ["arXiv 摘要 2110.14168、2505.22203、2505.15801", "arXiv:2507.08794 §2、§3.1、§3.3.2、附錄 A"]
related: ["genrm-best-of-n-discriminative-verifiers", "mk-generative-rm-for-rlvr", "mk-lineage-hub"]
---
# 驗證器與參考式基準

**Cobbe 等 2021（arXiv 2110.14168）。** 提出 GSM8K 與驗證器；本文以 GSM8K 作為評估集之一。內容見 [判別式驗證器與 Best-of-N](genrm-best-of-n-discriminative-verifiers.html)。

**本文使用的評估資料（§3.1）。** Multi-subject RLVR 與 NaturalReasoning 子集（一般推理），GSM8K、MATH、AIME 1983–2024（數學）。NaturalReasoning 對應 arXiv 2502.13124（Yuan 等 2025）。

**Yan 等 2025（arXiv 2505.15801，VerifyBench）。** 摘要：現有獎勵基準多為回應間的偏好比較，缺少針對「對照真實參考答案做驗證」的評測；提出 VerifyBench 與較難的 VerifyBench-Hard，經人工標註。本文 §3.3.2 用其比較評審模型與規則式 math-verify（Table 2b：Master-RM-32B 在 VerifyBench 準確率 95.15%）。

**Huang 等 2025c（arXiv 2505.22203）。** 本文參考文獻表的標題為「Pitfalls of rule- and model-based verifiers」，arXiv 目前標題為 From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in Mathematical Reasoning。摘要：規則式驗證器無法辨認不同格式的等價答案，造成大量假陰性，且隨政策變強而阻礙 RL；模型式驗證器靜態準確率較高，但在 RL 期間容易被 reward hacking，把某些回應模式誤判為正確。這與本文的假陽性發現同期且方向一致（附錄 A 稱為 concurrent work）。

**Omni-MATH（arXiv 2410.07985）。** 本文 §2 以此作為 Omni-Judge 的出處；Omni-MATH 摘要描述的是 4428 題奧林匹亞等級數學基準，Omni-Judge 的細節不在摘要中，此處不補充。

下一步：本文攻擊的生成式驗證器，見 [RLVR 中的生成式獎勵模型](mk-generative-rm-for-rlvr.html)；回到 [總覽](mk-lineage-hub.html)。
