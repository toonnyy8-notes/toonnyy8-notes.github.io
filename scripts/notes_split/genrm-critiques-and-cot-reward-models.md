---
slug: genrm-critiques-and-cot-reward-models
title: 批評與 CoT 用於獎勵模型（Saunders、McAleese、Ye、Ankner）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [critique, chain-of-thought, reward-model, synthetic-rationale]
arxiv: ["2408.15240", "2206.05802", "2407.00215", "2405.20850", "2408.11791"]
sources: ["arXiv 摘要 2206.05802、2407.00215", "arXiv:2408.15240 §5 對 2405.20850、2408.11791 的描述"]
related: ["genrm-lineage-hub", "genrm-llm-as-a-judge", "genrm-yes-token-method", "genrm-inference-time-compute-majority-voting"]
---
# 批評與 CoT 用於獎勵模型

**Saunders 等 2022（arXiv 2206.05802，Self-critiquing models for assisting human evaluators）。** 摘要：以行為複製微調 LLM 撰寫自然語言批評；在主題式摘要任務中，模型批評幫助人類找到原本會漏掉的缺陷。GenRM §5：這類工作以高品質人類資料訓練批評模型；GenRM 則可只用合成、模型產生的批評訓練。

**McAleese 等 2024（arXiv 2407.00215，LLM Critics Help Catch LLM Bugs）。** 摘要：以 RLHF 訓練 critic 模型，撰寫指出程式碼問題的自然語言回饋，協助人類評估模型寫的程式碼。GenRM §5：該工作訓練判別式 RM 來產生程式碼批評。

**Ye 等 2024（arXiv 2405.20850，Improving Reward Models with Synthetic Critiques）。** GenRM §5：此工作只使用另一個高能力 LLM 產生的 CoT；GenRM-CoT 在推論時使用自己產生的 CoT 判斷正確性。

**Ankner 等 2024（arXiv 2408.11791，Critique-out-Loud Reward Models）。** GenRM §5 稱其為同期工作：訓練 RM 對偏好對產生批評，批評由強得多的 LLM 產生，再輸入與底層 LLM 分離的 RM head。該 head 以判別式訓練，因此無法統一生成與驗證；其合成批評未經正確性過濾。

**以 LLM-as-a-Judge 取得 CoT 訊號。** 見 [LLM-as-a-Judge 與自我獎勵、自學評估器](genrm-llm-as-a-judge.html)。

**GenRM 的差異。** CoT 理由直接作為 next-token prediction 的訓練目標（[GenRM 的 Yes token 方法](genrm-yes-token-method.html)），結論稱含錯誤的合成理由已足以讓 GenRM 學會挑出數學題中的細微錯誤。多條 CoT 的整合見 [推論時運算與多數決](genrm-inference-time-compute-majority-voting.html)。回到 [總覽](genrm-lineage-hub.html)。
