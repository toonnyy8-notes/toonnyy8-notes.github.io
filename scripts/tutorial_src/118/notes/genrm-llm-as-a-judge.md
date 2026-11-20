---
slug: genrm-llm-as-a-judge
title: LLM-as-a-Judge 與自我獎勵、自學評估器
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [LLM-as-a-judge, self-rewarding, self-taught-evaluators, verification]
arxiv: ["2408.15240", "2306.05685", "2401.10020", "2408.02666"]
sources: ["arXiv 摘要 2306.05685、2401.10020、2408.02666", "arXiv:2408.15240 §2、§5"]
related: ["genrm-lineage-hub", "genrm-critiques-and-cot-reward-models", "genrm-yes-token-method"]
---
# LLM-as-a-Judge 與自我獎勵、自學評估器

**Zheng 等 2023（arXiv 2306.05685，Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena）。** 摘要：探討以強 LLM 作為評審評估開放式問答的聊天助理，檢視其限制，包括位置、冗長與自我偏好偏誤，以及有限的推理能力，並提出部分緩解方法；以 MT-bench 與 Chatbot Arena 驗證 LLM 評審與人類偏好的一致性。

**GenRM 的定位。** GenRM §2：LLM-as-a-Judge 不微調驗證器，只提示 LLM 做驗證或自我批評，有時附參考解。§5 把它視為「同樣以 next-token prediction 做驗證」的另一條路線，差別在沒有針對驗證訓練。實驗結果（§5）：較強的 Gemini 1.0 Pro 當評審，表現不如用較弱的 Gemma 訓練的 GenRM。

**以 LLM-as-a-Judge 取得獎勵的後續工作。** GenRM §5 列舉 Yuan 等 2024（arXiv 2401.10020）與 Wang 等 2024（arXiv 2408.02666）。
- Self-Rewarding Language Models 摘要：由語言模型自己以 LLM-as-a-Judge 提示給出訓練期間的獎勵；Iterative DPO 訓練下，指令遵循與自我給獎勵的能力同時提升。
- Self-Taught Evaluators 摘要：只用合成訓練資料改進評估器；從未標註指令出發，迭代產生對比輸出並訓練 LLM-as-a-Judge 產生推理軌跡與最終判斷。

GenRM 與這些工作的差別（§5）：GenRM 直接以模型產生的 CoT 訓練驗證器。相關的批評式方法見 [批評與 CoT 用於獎勵模型](genrm-critiques-and-cot-reward-models.html)。回到 [總覽](genrm-lineage-hub.html)。
