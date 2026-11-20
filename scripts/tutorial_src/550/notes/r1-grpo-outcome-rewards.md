---
slug: r1-grpo-outcome-rewards
title: GRPO 與規則式結果獎勵（DeepSeek-R1-Zero）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [GRPO, DeepSeekMath, rule-based-reward, R1-Zero, outcome-reward]
arxiv: ["2402.03300", "1707.06347", "1506.02438", "2412.19437", "2501.12948"]
sources: ["arXiv 摘要 2402.03300", "2501.12948 §1、§2.1、§2.2、附錄 A.3"]
related: ["r1-lineage-hub", "r1-rlhf-ppo-background", "r1-process-supervision-prm", "rlvr-grpo-credit-assignment"]
---
# GRPO 與規則式結果獎勵（DeepSeek-R1-Zero）

**GRPO 的來源。** 本文 §2.1：GRPO 由 Shao 等（2024，arXiv 2402.03300，DeepSeekMath）提出，原意是簡化 PPO 的訓練並降低資源消耗。DeepSeekMath 摘要的主題是以 120B 數學相關 token 繼續預訓練 7B 模型，GRPO 作為其 RL 方法在該文中提出。

**演算法（依本文 §2.1、A.3）。** 對每個問題 q，從舊策略取樣一組 G 個輸出，最佳化帶 clip 與對參考策略 KL 懲罰的目標。優勢值為組內標準化：`A_i = (r_i − mean) / std`，不需要與策略同尺寸的價值模型；KL 以無偏估計式直接加進損失，而非 PPO 的逐 token 獎勵。與 PPO 的差異見 [LLM 的 RL 前史](r1-rlhf-ppo-background.html)。

**DeepSeek-R1-Zero 的設計。** 以 DeepSeek-V3-Base（arXiv 2412.19437）為基底，不做 SFT 即進行 RL。獎勵由準確度獎勵與格式獎勵組成，兩者等權重相加；本文表示不對推理任務使用神經獎勵模型（結果式或過程式皆不用），理由見 [過程監督與 PRM](r1-process-supervision-prm.html)。

**後續定位。** GRPO 在 RLVR 文獻中的位置與信用分配問題見 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)。總覽：[總覽](r1-lineage-hub.html)。
