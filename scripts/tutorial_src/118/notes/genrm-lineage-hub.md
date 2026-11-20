---
slug: genrm-lineage-hub
title: Generative Verifiers（GenRM）的研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [GenRM, verifier, reward-model, best-of-n, hub]
arxiv: ["2408.15240", "2009.01325", "2110.14168", "2211.14275", "2305.20050", "2312.08935", "2306.05685", "2206.05802", "2201.11903", "2203.11171", "2407.21787", "2305.18290", "2402.06457", "2203.14465", "2401.10020", "2408.02666"]
sources: ["arXiv:2408.15240 摘要、§1、§2、§3、§5", "各引用論文的 arXiv 摘要（見各子筆記）"]
related: ["genrm-best-of-n-discriminative-verifiers", "genrm-step-level-supervision-verifiers", "genrm-llm-as-a-judge", "genrm-critiques-and-cot-reward-models", "genrm-dpo-implicit-verifiers-unification", "genrm-inference-time-compute-majority-voting", "genrm-yes-token-method", "r1-rlhf-ppo-background", "r1-process-supervision-prm", "r1-cot-and-test-time-compute", "r1-star-self-generated-reasoning"]
---
# Generative Verifiers（GenRM）的研究脈絡總覽

GenRM（arXiv 2408.15240）把驗證器（verifier）訓練成一般的 next-token prediction：以 "Yes" 或 "No" 這個 token 的機率作為正確性分數，並可在判定前先生成 CoT 理由。本文的 Related Work（§5）把前人工作分成四類：判別式 RM 與驗證器、LLM-as-a-Judge、以 CoT 或批評輔助 RM、以及生成與驗證的統一（DPO 一系）。

建議閱讀順序：

1. [Best-of-N 與判別式驗證器](genrm-best-of-n-discriminative-verifiers.html)：問題設定與本文要取代的基線。
2. [步驟級與結果級監督的驗證器](genrm-step-level-supervision-verifiers.html)：判別式驗證器的監督粒度。
3. [LLM-as-a-Judge](genrm-llm-as-a-judge.html)：不訓練、只提示的生成式驗證。
4. [批評與 CoT 用於獎勵模型](genrm-critiques-and-cot-reward-models.html)：與本文最接近的工作與差異。
5. [DPO 與隱式驗證器：生成與驗證的統一](genrm-dpo-implicit-verifiers-unification.html)：本文 unified loss 的對照。
6. [推論時運算與多數決](genrm-inference-time-compute-majority-voting.html)：GenRM-CoT 的 majority voting 來源。
7. [GenRM 的 Yes token 方法](genrm-yes-token-method.html)：本文方法本身，彙整上述脈絡。

已有的相關筆記：LLM 的 RL 前史見 [LLM 的 RL 前史：RLHF、PPO 與 GAE](r1-rlhf-ppo-background.html)；過程監督見 [過程監督與 PRM](r1-process-supervision-prm.html)；CoT 與推論時運算見 [思維鏈、驗證器與推論時運算](r1-cot-and-test-time-compute.html)；STaR 見 [STaR](r1-star-self-generated-reasoning.html)。時間線見 `timeline.json`。
