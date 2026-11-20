---
slug: genrm-yes-token-method
title: GenRM 方法：以 Yes token 機率作驗證分數
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [GenRM, GenRM-CoT, next-token-prediction, SFT, verifier]
arxiv: ["2408.15240", "2110.14168", "2201.11903", "2203.11171"]
sources: ["arXiv:2408.15240 §2、§3、§5、§6"]
related: ["genrm-lineage-hub", "genrm-best-of-n-discriminative-verifiers", "genrm-dpo-implicit-verifiers-unification", "genrm-inference-time-compute-majority-voting", "genrm-critiques-and-cot-reward-models"]
---
# GenRM 方法：以 Yes token 機率作驗證分數

本筆記整理 GenRM（arXiv 2408.15240）的方法本身，並標出每一部分承接的前人工作。

1. **Direct verifier（§3.1）。** 以 SFT 最大化 log p(Yes | x, y+, I) 與 log p(No | x, y−, I)，I = "Is the answer correct (Yes/No)?"。推論時分數 r_Direct = p_θ(Yes | x, y, I)。對照基線是判別式 RM（[Best-of-N 與判別式驗證器](genrm-best-of-n-discriminative-verifiers.html)，式 (3)）。
2. **統一生成與驗證（§3.2）。** L_GenRM = L_SFT(θ, D_verify) + λ L_SFT(θ, D_correct)，λ 為混合比例；論文預設使用此損失。對照 [DPO 與隱式驗證器](genrm-dpo-implicit-verifiers-unification.html)。
3. **GenRM-CoT（§3.3）。** 先生成理由 v_CoT（提示 I_CoT = "Let's verify step by step."），再以 p(Yes | x, y, I_CoT, v_CoT, I) 評分。理由可由人或 LLM 產生。CoT 來源見 [推論時運算與多數決](genrm-inference-time-compute-majority-voting.html)。
4. **多數決（式 (7)）。** 取樣 K 條理由並平均 Yes 機率，使驗證能使用推論時運算。
5. **合成理由。** 與依賴人類資料或強模型批評的工作比較，見 [批評與 CoT 用於獎勵模型](genrm-critiques-and-cot-reward-models.html)。

**摘要中的數字（原文）。** Best-of-N 在演算法任務由 5% 到 45.3%，GSM8K 由 73% 到 93.4%；易到難泛化在 MATH 由 28% 到 44.6%，MMLU abstract algebra 由 37.9% 到 53.5%。

**未來方向（§6）。** 擴展到程式、對齊、文字到圖像與開放式生成；利用過程級監督與以 RL 訓練 CoT 驗證器；把生成式驗證器放入 LLM 的 RL 流程。回到 [總覽](genrm-lineage-hub.html)。
