---
slug: mk-lineage-hub
title: One Token to Fool LLM-as-a-Judge 的研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [LLM-as-a-judge, reward-hacking, RLVR, generative-reward-model, master-key]
arxiv: ["2507.08794", "1811.07871", "2212.08073", "2309.00267", "2305.17926", "2402.14016", "2410.07137", "2401.08967", "2501.12948", "2503.23829", "2505.22203", "2505.15801"]
sources: ["arXiv:2507.08794 §1、§2、附錄 A", "各引用論文的 arXiv 摘要"]
related: ["mk-ai-feedback-reward-modeling-origins", "genrm-llm-as-a-judge", "mk-judge-biases-and-adversarial-attacks", "mk-rlvr-and-rule-based-rewards", "r1-grpo-outcome-rewards", "rlvr-grpo-credit-assignment", "mk-verifiers-and-reference-based-benchmarks", "mk-generative-rm-for-rlvr", "genrm-yes-token-method", "mk-master-key-attack-and-master-rm", "mk-inference-time-and-prompt-defenses", "genrm-lineage-hub", "r1-lineage-hub"]
---
# One Token to Fool LLM-as-a-Judge 的研究脈絡總覽

本文（arXiv 2507.08794）的問題：RLVR（以可驗證獎勵做強化學習）在參考答案已知的設定下，仍可能用 LLM 當獎勵模型。作者在一次 RLVR 訓練崩潰中發現，政策模型退化成輸出「Solution」「Thought process:」這類短開頭，而 LLM 評審（Qwen2.5-72B-Instruct）仍給正向獎勵（§1）。作者把這類無實質內容的輸入稱為 master keys，並提出以截斷回答作負例的資料擴增，訓練出 Master-RM。

閱讀順序如下，每一步補上下一步需要的背景。

1. [AI 回饋與獎勵模型的起源](mk-ai-feedback-reward-modeling-origins.html)：以模型評估取代人類標註的想法，從 reward modeling 研究方向、Constitutional AI 到 RLAIF。LLM-as-a-Judge 本身見 [LLM-as-a-Judge 與自我獎勵、自學評估器](genrm-llm-as-a-judge.html)。
2. [LLM 評審的偏誤與對抗攻擊](mk-judge-biases-and-adversarial-attacks.html)：本文之前已知的評審弱點，集中在偏好比較設定。
3. [RLVR 與規則式獎勵](mk-rlvr-and-rule-based-rewards.html)：獎勵由參考答案決定的訓練設定。RLVR、GRPO 的詳細推導見 [GRPO 與規則式結果獎勵（DeepSeek-R1-Zero）](r1-grpo-outcome-rewards.html) 與 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)。
4. [驗證器與參考式基準](mk-verifiers-and-reference-based-benchmarks.html)：從 GSM8K 驗證器到 VerifyBench，以及規則式驗證器的限制。
5. [RLVR 中的生成式獎勵模型](mk-generative-rm-for-rlvr.html)：本文攻擊與修補的直接對象（Multi-sub RM、General-Verifier、Omni-Judge）。生成式驗證器的一般方法見 [GenRM 方法](genrm-yes-token-method.html)。
6. [Master key 攻擊與 Master-RM](mk-master-key-attack-and-master-rm.html)：本文的攻擊定義、訓練資料擴增與主要結果。
7. [推論時策略與提示設計的影響](mk-inference-time-and-prompt-defenses.html)：CoT、多數決與移除問題對假陽性率的影響。

相關脈絡總覽：[Generative Verifiers（GenRM）的研究脈絡總覽](genrm-lineage-hub.html)、[DeepSeek-R1 的研究脈絡總覽](r1-lineage-hub.html)。年表見 `timeline.json`。
