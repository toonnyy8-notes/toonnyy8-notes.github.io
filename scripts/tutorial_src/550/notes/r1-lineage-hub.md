---
slug: r1-lineage-hub
title: DeepSeek-R1 的研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [DeepSeek-R1, reasoning, RL, lineage]
arxiv: ["2501.12948", "2201.11903", "2203.02155", "1707.06347", "2203.14465", "2305.20050", "2402.03300", "1503.02531"]
sources: ["2501.12948 摘要、§1、§2.1、附錄 F、G.2、H.2、H.3", "各前作的 arXiv 摘要（見各子筆記）"]
related: ["r1-cot-and-test-time-compute", "r1-rlhf-ppo-background", "r1-star-self-generated-reasoning", "r1-process-supervision-prm", "r1-grpo-outcome-rewards", "r1-distillation-to-small-models", "rlvr-grpo-credit-assignment", "rl-mdp-q-learning-basics", "knowledge-distillation-foundations"]
---
# DeepSeek-R1 的研究脈絡總覽

DeepSeek-R1（DeepSeek-AI，arXiv 2501.12948）主張不使用人工標註的推理軌跡，僅以強化學習（RL）提升 LLM 的推理能力。論文 §1 指出，既有做法依賴人工標註的推理軌跡，表現受人類範例限制。理解本文所需的前置工作可依下列順序閱讀。

1. [思維鏈與推論時運算](r1-cot-and-test-time-compute.html)：LLM 先產生中間推理步驟（Wei 等 2022），再以多次取樣、驗證器與投票換取效能（Cobbe 2021、Wang 2022、Snell 2024）。本文把這種「多花推論運算」改為由 RL 讓模型自行產生更長的推理。
2. [RLHF 與 PPO](r1-rlhf-ppo-background.html)：RL 在 LLM 的既有用途是對齊人類偏好（Ouyang 2022），演算法為 PPO（Schulman 2017）。本文 H.3 指出這類流程先 SFT 再訓練獎勵模型，可能限制模型只模仿人類推理。基本符號見 [MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)。
3. [STaR：以自產推理微調](r1-star-self-generated-reasoning.html)：以模型自己產生、答案正確的思維鏈再微調（Zelikman 2022），是最接近的前作；本文與之不同處是直接對 base model 做結果導向 RL，且不先做 SFT。
4. [過程監督與 PRM](r1-process-supervision-prm.html)：Uesato 2022、Lightman 2023 比較結果監督與過程監督；本文 G.2 記錄 PRM 嘗試的三項限制。
5. [GRPO 與結果導向獎勵](r1-grpo-outcome-rewards.html)：GRPO 源自 DeepSeekMath（Shao 2024），本文以規則式的準確度與格式獎勵訓練 DeepSeek-R1-Zero。已有的 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html) 說明 GRPO 在後續工作中的位置。
6. [蒸餾到小模型](r1-distillation-to-small-models.html)：以 DeepSeek-R1 產生的 800,000 筆樣本對 Qwen、LLaMA 做 SFT（本文附錄 F），背景為 Hinton 2015。通用背景見 [知識蒸餾基礎](knowledge-distillation-foundations.html)。

年代順序見 timeline.json。
