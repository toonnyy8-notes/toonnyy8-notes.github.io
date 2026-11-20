---
slug: odd-lineage-hub
title: 蒸餾動態研究（On-policy 與 off-policy）的研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [on-policy-distillation, off-policy, KL-direction, forgetting, hub]
arxiv: ["2609.35259", "1503.02531", "1606.07947", "2306.13649", "2506.09477", "2509.04259", "2505.11711", "2501.17161"]
sources: ["2609.35259 §1-3", "arXiv abstract 1503.02531", "arXiv abstract 1606.07947", "arXiv abstract 2306.13649"]
related: ["knowledge-distillation-foundations", "on-policy-distillation-imitation", "context-distillation-self-distillation", "odd-onpd-vs-offpd-setup", "odd-token-kl-direction-gradients", "odd-on-policy-forgetting-sparsity-claims", "odd-sft-vs-rl-generalization-claims", "odd-recent-opd-studies-2026", "odd-strong-to-weak-distillation-practice", "odd-learning-rate-adam-clipping-rlvr", "rlvr-grpo-credit-assignment", "ppo-grpo-dapo-llm-rl-objectives"]
---
# 蒸餾動態研究的研究脈絡總覽

論文《On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics》（arXiv:2609.35259）在強到弱蒸餾的設定中，分別改變 rollout policy（訓練資料由誰產生）、token 層級 KL 方向與學習率，檢驗「on-policy 學習較不遺忘、更新較稀疏、泛化較好」這組說法。以下依閱讀順序列出理解該論文所需的背景。

## 閱讀順序
1. 蒸餾的基本形式：[知識蒸餾基礎](knowledge-distillation-foundations.html) 說明 logit 蒸餾與序列層級蒸餾。
2. 線上模仿與 on-policy 蒸餾：[On-policy 蒸餾與線上模仿學習](on-policy-distillation-imitation.html)。
3. 本論文使用的名詞與設定（OnPD、OffPD、混合式）：[OnPD 與 OffPD 的定義與設定](odd-onpd-vs-offpd-setup.html)。
4. 為何要拆開 KL 方向與 rollout policy：[Token 層級 KL 方向與梯度](odd-token-kl-direction-gradients.html)。
5. 被檢驗的第一組說法（遺忘、更新稀疏）：[On-policy 學習的遺忘與更新稀疏說法](odd-on-policy-forgetting-sparsity-claims.html)。
6. 被檢驗的第二組說法（泛化）：[SFT 與 RL 的泛化比較](odd-sft-vs-rl-generalization-claims.html)。
7. 同期的 on-policy 蒸餾研究：[2026 年的 on-policy 蒸餾研究](odd-recent-opd-studies-2026.html)。
8. 強到弱蒸餾的實務背景：[強到弱蒸餾在後訓練中的使用](odd-strong-to-weak-distillation-practice.html)。
9. 控制變因與後續 RLVR：[學習率、Adam、梯度裁剪與 RLVR](odd-learning-rate-adam-clipping-rlvr.html)。

## 相關的既有筆記
- [context distillation 與自蒸餾](context-distillation-self-distillation.html)：本論文附錄 D.7 的西班牙文風格轉移實驗以教師條件化（帶示範與語言指令）進行，形式與自蒸餾相近。
- [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)、[PPO、GRPO、DAPO 與 LLM 的 RL 目標](ppo-grpo-dapo-llm-rl-objectives.html)：蒸餾後接 RLVR 的實驗所用的 RL 目標。

## 論文結論（摘要所述）
rollout policy 不一定是主因；token 層級 KL 方向較能決定任務表現與輸出覆蓋，學習率則主導遺忘與更新稀疏。正向 KL 對 rollout policy 穩健，反向 KL 較敏感且偏好學生產生的 rollout。
