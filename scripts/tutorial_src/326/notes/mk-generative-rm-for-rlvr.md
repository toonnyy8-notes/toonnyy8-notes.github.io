---
slug: mk-generative-rm-for-rlvr
title: RLVR 中的生成式獎勵模型（Multi-sub RM、General-Verifier、Omni-Judge）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [generative-reward-model, RLVR, Multi-sub-RM, General-Reasoner, LLM-as-a-judge]
arxiv: ["2507.08794", "2503.23829", "2505.14652", "2410.07985", "2408.15240", "2306.05685"]
sources: ["arXiv 摘要 2503.23829、2505.14652", "arXiv:2507.08794 §2、§3.1、§3.2、附錄 A（Generative Reward Model）"]
related: ["genrm-yes-token-method", "genrm-llm-as-a-judge", "mk-master-key-attack-and-master-rm", "mk-lineage-hub"]
---
# RLVR 中的生成式獎勵模型

**動機（附錄 A）。** 規則式獎勵計算成本低，但難以辨認以不同形式表達的等價答案，也無法評估一般推理場景的開放式回答。因此有工作提示 LLM 評估答案以產生獎勵，引用 Zheng 等 2023、Lee 等 2023、Tian 等 2024、Zhang 等 2024a、Su 等 2025、Ma 等 2025a 等。可搭配 CoT 與多數決提升評估準確度（引用 Zhang 等 2024a）。一般的生成式驗證器方法見 [GenRM 方法](genrm-yes-token-method.html) 與 [LLM-as-a-Judge](genrm-llm-as-a-judge.html)。

**本文攻擊與修補的直接前作。**

- **Multi-sub RM（Su 等 2025，arXiv 2503.23829，Crossing the Reward Bridge）。** 摘要：RLVR 擴展到醫學、化學、心理學、經濟、教育等沒有結構化參考答案的領域；提供專家撰寫的參考答案時，各 LLM 的二元驗證判斷一致性高；使用生成式評分產生軟性的模型式獎勵，並可用 7B 模型訓練跨領域生成式獎勵模型。本文 §3.2 說明 Su 等釋出 160k 筆 (q, a*, o, y) 資料，標籤由 Qwen2.5-72B-Instruct 擔任教師評分，並以 SFT 得到 Multi-sub RM（本文簡稱）。
- **General-Verifier（Ma 等 2025a，arXiv 2505.14652，General-Reasoner）。** 摘要：建立可驗證答案的跨學科大規模問題資料集，並發展具 CoT 與情境感知能力的生成式答案驗證器，取代規則式驗證。
- **Omni-Judge（Gao 等 2024，arXiv 2410.07985）。** 本文列為 RLVR 框架下微調的專用生成式 RM。

**本文如何使用它們（§3.1）。** 本文把受測評審分兩組：專用生成式 RM（Multi-sub RM、General-Verifier、Omni-Judge）與未針對獎勵建模微調的通用 LLM（Qwen2.5-72B-Instruct、LLaMA3-70B-Instruct、GPT-4o、GPT-o1、Claude-4）。Master-RM 的訓練起點是 Su 等的 160k 資料與 Qwen2.5-7B-Instruct（Multi-sub RM 同一基底）。結果見 [Master key 攻擊與 Master-RM](mk-master-key-attack-and-master-rm.html)；回到 [總覽](mk-lineage-hub.html)。
