---
slug: mk-rlvr-and-rule-based-rewards
title: RLVR 與規則式獎勵（本文的訓練設定）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [RLVR, rule-based-reward, verifiable-reward, reasoning]
arxiv: ["2507.08794", "2411.01111", "2401.08967", "2411.15124", "2501.12948", "2502.14768"]
sources: ["arXiv:2507.08794 §2、附錄 A（Rule-Based Reward in RLVR）", "arXiv 摘要 2411.01111、2401.08967、2411.15124、2502.14768"]
related: ["r1-grpo-outcome-rewards", "rlvr-grpo-credit-assignment", "mk-generative-rm-for-rlvr", "mk-verifiers-and-reference-based-benchmarks", "mk-lineage-hub"]
---
# RLVR 與規則式獎勵

**本文的設定（§2）。** RLVR 是參考答案已知的設定：獎勵訊號由規則函數或生成式 LLM 評審提供。每一步獎勵模型收到問題 q、政策回應 o、參考答案 a*，輸出 YES 或 NO，對應獎勵 1 或 0。引用的 RLVR 工作為 Luong 等 2024、Lambert 等 2024、Guo 等 2025、Su 等 2025。

**規則式獎勵的來源（附錄 A）。** 本文說規則式獎勵最早用於安全，後來在推理任務上表現顯著，並進一步擴展到 [0, 1] 的連續值。

- Mu 等 2024（arXiv 2411.01111，Rule Based Rewards for Language Model Safety）摘要：提出以 AI 回饋、僅需少量人類資料的偏好建模方法 RBR，以一組規則為基礎。
- Luong 等 2024（arXiv 2401.08967，ReFT）摘要：先以 SFT 暖身，再用線上 RL（本文用 PPO），獎勵直接由真實答案得到；在 GSM8K、MathQA、SVAMP 上顯著優於 SFT。
- Lambert 等 2024（arXiv 2411.15124，Tulu 3）：本文引為 RLVR 工作之一；摘要描述的是完全開放的後訓練模型、資料與配方。
- Xie 等 2025（arXiv 2502.14768，Logic-RL）：本文列為連續值獎勵的例子；摘要提到以規則式 RL 訓練，7B 模型僅用 5K 邏輯題即可推廣到 AIME 與 AMC。
- Guo 等 2025（arXiv 2501.12948，DeepSeek-R1）：RLVR 的代表性工作，方法見 [GRPO 與規則式結果獎勵（DeepSeek-R1-Zero）](r1-grpo-outcome-rewards.html)；GRPO 與信用分配的討論見 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)。

**對本文的意義。** 規則式驗證器難以辨認數學上等價但寫法不同的答案，也無法評估開放式問答，因此改用生成式獎勵模型（見 [RLVR 中的生成式獎勵模型](mk-generative-rm-for-rlvr.html)）。一旦獎勵由模型給出，該模型的假陽性就會直接進入訓練訊號；本文 §2 稱這會誤導學習軌跡。規則式驗證器的限制見 [驗證器與參考式基準](mk-verifiers-and-reference-based-benchmarks.html)。回到 [總覽](mk-lineage-hub.html)。
