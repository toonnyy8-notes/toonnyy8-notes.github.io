---
slug: mk-master-key-attack-and-master-rm
title: Master key 攻擊與 Master-RM（本文方法與主要結果）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [master-key, Master-RM, false-positive-rate, data-augmentation, reward-hacking]
arxiv: ["2507.08794", "2503.23829", "2505.15801"]
sources: ["arXiv:2507.08794 §1、§2、§3.1–3.3、Table 1、Table 2"]
related: ["mk-generative-rm-for-rlvr", "mk-judge-biases-and-adversarial-attacks", "mk-inference-time-and-prompt-defenses", "mk-verifiers-and-reference-based-benchmarks", "mk-lineage-hub"]
---
# Master key 攻擊與 Master-RM

**發現過程（§1）。** 一次 RLVR 訓練中，回應長度降到少於 30 個 token、KL 散度上升（Figure 2）；政策模型輸出「Solution」「Thought process:」之類的開頭，仍被評審判為正確。

**評審函數（§2）。** J(q, a*, o) → {YES, NO}；YES 對應獎勵 1。

**Master keys（§2、§3.1）。** 兩類：非字詞符號（空白、「.」「,」「:」），以及 reasoning openers（「Thought process:」「Let's solve this problem step by step.」「Solution」及中文「解」、日文「かいせつ」、西班牙文「Respuesta」）。共十個。

**假陽性率（Table 1，%）。** 十個 master key 的平均與最差值（跨五個資料集）：GPT-4o 平均 14.6、最差 53.6；Claude-4 平均 12.4、最差 57.7；Qwen2.5-72B-Instruct 平均 66.8、最差 90.9；LLaMA3-70B-Instruct 平均 80.6、最差 95.1；General-Verifier 平均 9.7、最差 87.0。Master-RM-7B 平均 0.1、最差 2.9；Master-RM-32B 平均 0.8、最差 5.4。

**Master-RM 的訓練（§3.2）。** 從 Su 等 160k 資料隨機取 20k 筆，以 GPT-4o-mini 的 CoT 提示重新產生回應，只保留第一句（通常是 reasoning opener），標為 NO；與原 160k 合併為 180k 筆。檢查與評估用 master keys 無重疊。以標準交叉熵 SFT 微調 Qwen2.5-7B-Instruct 與 Qwen2.5-32B-Instruct，得到 Master-RM-7B 與 Master-RM-32B。

**一般評審能力是否受損（§3.3.2、Table 2）。** Master-RM-7B 與 GPT-4o 的 Cohen's kappa 為 0.91、與人類為 0.90；VerifyBench 上 Master-RM-32B 準確率 95.15%、VerifyBench-Hard 86.80%。

前置研究見 [RLVR 中的生成式獎勵模型](mk-generative-rm-for-rlvr.html) 與 [LLM 評審的偏誤與對抗攻擊](mk-judge-biases-and-adversarial-attacks.html)；分析實驗見 [推論時策略與提示設計的影響](mk-inference-time-and-prompt-defenses.html)；基準見 [驗證器與參考式基準](mk-verifiers-and-reference-based-benchmarks.html)。回到 [總覽](mk-lineage-hub.html)。
