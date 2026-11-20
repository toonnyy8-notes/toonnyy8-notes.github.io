---
slug: odd-on-policy-forgetting-sparsity-claims
title: On-policy 學習較不遺忘、更新較稀疏的既有說法
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [catastrophic-forgetting, update-sparsity, on-policy, RL-vs-SFT]
arxiv: ["2609.35259", "2509.04259", "2510.18874", "2505.11711", "2601.19897", "2606.13657"]
sources: ["2609.35259 §1-2, §4.1", "arXiv abstract 2509.04259", "arXiv abstract 2510.18874", "arXiv abstract 2505.11711", "arXiv abstract 2601.19897", "arXiv abstract 2606.13657"]
related: ["odd-lineage-hub", "odd-sft-vs-rl-generalization-claims", "odd-learning-rate-adam-clipping-rlvr", "context-distillation-self-distillation", "sft-vs-rl-offline-negative-rewards"]
---
# On-policy 學習的遺忘與更新稀疏說法

屬於 [蒸餾動態研究的研究脈絡總覽](odd-lineage-hub.html)。論文引言（arXiv:2609.35259）把這些結果列為「on-policy 較佳」的論據，並指出多數比較同時改變了目標、獎勵與超參數，因此視為待檢驗的假說。

## 遺忘
- Shenfeld et al.（arXiv:2509.04259，RL's Razor）：摘要指出 RL 與 SFT 在新任務表現相近時，RL 保留舊能力較好；遺忘程度由微調後與基礎策略在新任務上的 KL 決定，on-policy RL 隱含偏向 KL 最小的解。
- Chen et al.（arXiv:2510.18874，Retaining by Doing）：摘要指出在 Llama、Qwen 與多種任務上，RL 的遺忘少於 SFT 且目標任務表現相當或更高，並以兩分布混合的簡化模型分析原因。
- Shenfeld et al.（arXiv:2601.19897）：摘要提出 SDFT，以示範條件化的模型作為自己的教師，從示範直接做 on-policy 學習並保留舊能力。方法形式見 [context distillation 與自蒸餾](context-distillation-self-distillation.html)。

## 更新稀疏
- Mukherjee et al.（arXiv:2505.11711）：摘要指出 RL 只更新 5% 至 30% 的參數，在 7 種 RL 演算法與 10 個 LLM 中皆觀察到。本論文採其稀疏度量：參數相對初始檢查點的絕對變化小於 1e-6 的比例。附錄 E 另列出該文各檢查點的學習率，顯示 SFT 檢查點的學習率均值高於 RL 與偏好最佳化檢查點。
- Yu et al.（arXiv:2606.13657）：摘要指出 on-policy 蒸餾的更新小且座標稀疏，在子網路上遮罩訓練幾乎可還原完整訓練的表現。

## 與本論文的關係
論文在固定學習率與 KL 方向下比較 OnPD 與 OffPD，結論是遺忘與更新稀疏主要由學習率決定，rollout policy 的影響很小（見 [學習率、Adam、梯度裁剪與 RLVR](odd-learning-rate-adam-clipping-rlvr.html)）。SFT 與 RL 的目標差異另見 [SFT 與 RL 的差異](sft-vs-rl-offline-negative-rewards.html)。

泛化面的說法見 [SFT 與 RL 的泛化比較](odd-sft-vs-rl-generalization-claims.html)。
