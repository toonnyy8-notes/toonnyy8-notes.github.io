---
slug: rzero-lineage-hub
title: R-Zero 的研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [R-Zero, self-evolving, RLVR, self-play, label-free-RL]
arxiv: ["2508.05004", "2402.03300", "2501.12948", "2504.16084", "2505.03335"]
sources: ["arXiv 2508.05004 §1、§2、§4、§6、附錄 E–F", "arXiv abstract 2402.03300", "arXiv abstract 2501.12948", "arXiv abstract 2504.16084", "arXiv abstract 2505.03335"]
related: ["rzero-grpo-rlvr-foundation", "rzero-label-free-intrinsic-rewards", "rzero-majority-vote-pseudo-labels", "rzero-self-play-task-generation", "rzero-difficulty-curriculum", "rzero-challenger-solver-pipeline", "rzero-self-training-collapse", "r1-grpo-outcome-rewards", "rlvr-grpo-credit-assignment", "test-time-training-for-rl", "ppo-grpo-dapo-llm-rl-objectives"]
---
# R-Zero 的研究脈絡總覽

R-Zero（arXiv 2508.05004）從同一個 base 模型初始化 Challenger 與 Solver 兩個角色。Challenger 以 GRPO 訓練，產生讓 Solver 約有一半機率答對的題目；Solver 以 GRPO 訓練，標籤來自自己多次作答的多數決。整個流程不使用外部題目與標籤。

理解這篇論文需要六條脈絡，建議依下列順序閱讀：

1. [GRPO 與可驗證獎勵的 RL（RLVR）](rzero-grpo-rlvr-foundation.html)：Challenger 與 Solver 共用的訓練演算法。GRPO 與 R1 的細節見既有筆記 [GRPO 與規則式結果獎勵（DeepSeek-R1-Zero）](r1-grpo-outcome-rewards.html) 與 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)。
2. [無標籤 RL：以模型自身訊號當獎勵](rzero-label-free-intrinsic-rewards.html)：熵最小化、信心與 self-certainty。這類方法去掉標籤，但仍需要題目。
3. [多數決偽標籤](rzero-majority-vote-pseudo-labels.html)：TTRL 等方法以多次取樣的多數決取代標準答案，R-Zero 的 Solver 獎勵屬於這一類。
4. [自我對弈與自產任務](rzero-self-play-task-generation.html)：SPIN、Absolute Zero、Self-Challenging 等讓模型自己出題，多數依賴程式執行器驗證。
5. [難度課程：在能力邊緣出題](rzero-difficulty-curriculum.html)：成功率接近 50% 的題目學習訊號最大，是 Challenger 獎勵的理論依據。
6. [R-Zero 的 Challenger–Solver 流程](rzero-challenger-solver-pipeline.html)：把上述元件組合成一個迭代迴圈。
7. [自我訓練的崩壞風險](rzero-self-training-collapse.html)：論文觀察到數次迭代後效能下降，並連結到 model collapse 文獻。

相關的既有筆記：[測試時訓練與 test-time RL](test-time-training-for-rl.html)、[PPO、GRPO、DAPO 與 LLM 的 RL 目標](ppo-grpo-dapo-llm-rl-objectives.html)。
