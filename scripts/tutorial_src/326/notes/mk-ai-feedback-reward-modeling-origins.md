---
slug: mk-ai-feedback-reward-modeling-origins
title: AI 回饋與獎勵模型的起源（Reward Modeling、Constitutional AI、RLAIF）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [reward-modeling, RLAIF, constitutional-AI, LLM-as-a-judge]
arxiv: ["2507.08794", "1811.07871", "2212.08073", "2309.00267", "2306.05685"]
sources: ["arXiv 摘要 1811.07871、2212.08073、2309.00267", "arXiv:2507.08794 §1（引用方式）"]
related: ["genrm-llm-as-a-judge", "mk-judge-biases-and-adversarial-attacks", "r1-rlhf-ppo-background", "mk-lineage-hub"]
---
# AI 回饋與獎勵模型的起源

本文 §1 的開場論點是：評估一個回答通常比從頭生成一個回答容易（引用 Leike 等 2018），這個想法促成以 LLM 作自動評審（引用 Bai 等 2022、Kim 等 2023b、Lee 等 2023、Zheng 等 2023、Zhang 等 2024a）。

**Leike 等 2018（arXiv 1811.07871，Scalable agent alignment via reward modeling）。** 摘要：RL 應用於真實問題的障礙之一是缺少合適的獎勵函數；提出以 reward modeling 為核心的研究方向，即從與使用者的互動學習獎勵函數，再用 RL 最佳化該函數，並討論擴大規模時預期的挑戰。

**Bai 等 2022（arXiv 2212.08073，Constitutional AI）。** 摘要：不用人類標註有害輸出，只靠一份規則或原則清單，經由自我批評與修訂的監督階段，以及由模型評估哪個回應較好的 RL 階段，訓練無害的助理。

**Lee 等 2023（arXiv 2309.00267，RLAIF vs. RLHF）。** 摘要：以現成 LLM 產生偏好標籤訓練獎勵模型；在摘要、有益對話與無害對話任務上，RLAIF 的表現與 RLHF 相當。

**Zheng 等 2023（arXiv 2306.05685）。** MT-Bench 與 Chatbot Arena 的 LLM 評審研究，內容見 [LLM-as-a-Judge 與自我獎勵、自學評估器](genrm-llm-as-a-judge.html)，此處不重述。

這些工作的共同點是獎勵或偏好由模型產生，因此模型評審的可靠性直接決定訓練訊號的品質。本文針對的是其中「與參考答案比較」的子情境。下一步是這些評審已知的弱點：[LLM 評審的偏誤與對抗攻擊](mk-judge-biases-and-adversarial-attacks.html)。RLHF、PPO 的背景見 [LLM 的 RL 前史](r1-rlhf-ppo-background.html)；回到 [總覽](mk-lineage-hub.html)。
