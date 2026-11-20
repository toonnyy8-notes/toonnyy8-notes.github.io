---
slug: rzero-self-play-task-generation
title: 自我對弈與模型自產任務（SPIN、Absolute Zero、Self-Challenging）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [self-play, self-challenging, Absolute-Zero, co-evolution, task-generation]
arxiv: ["2508.05004", "2401.01335", "2505.03335", "2506.01716", "2506.03136", "2502.14948", "2408.01072", "2404.14387"]
sources: ["arXiv 2508.05004 §1、表 1–2", "arXiv abstract 2401.01335", "arXiv abstract 2505.03335", "arXiv abstract 2506.01716", "arXiv abstract 2506.03136", "arXiv abstract 2502.14948", "arXiv abstract 2408.01072", "arXiv abstract 2404.14387"]
related: ["rzero-lineage-hub", "rzero-label-free-intrinsic-rewards", "rzero-challenger-solver-pipeline", "r1-star-self-generated-reasoning", "rzero-majority-vote-pseudo-labels"]
---
# 自我對弈與模型自產任務

R-Zero 引言（§1）把「self-challenging」方法定義為在模型自己產生的任務上訓練（Zhou et al., 2025a；Wang et al., 2025a；Zhao et al., 2025a），並指出其中許多方法依賴外部程式執行器確認任務可行且可驗證。在沒有驗證 oracle 的領域，例如開放式推理，自產資料的品質與正確性是未解的問題。

按摘要內容整理如下：

- SPIN（Chen et al., 2024，arXiv 2401.01335）：從 SFT 模型出發，讓 LLM 與自己前幾輪的版本對弈，區分自己產生的回答與人工標註回答；不需要額外人工標註資料。
- Absolute Zero（Zhao et al., 2025a，arXiv 2505.03335）：單一模型學習提出能最大化自身學習進展的任務並解題，提出「零」外部資料的 RLVR 範式。R-Zero 在表 1、表 2 以它為比較基準。
- Self-Challenging（Zhou et al., 2025a，arXiv 2506.01716）：agent 先扮演 challenger 產生 Code-as-Task（含指令、驗證函式、成功與失敗案例），再扮演 executor 以 RL 訓練；任務以驗證函式過濾。
- CURE（Wang et al., 2025a，arXiv 2506.03136）：以互動結果讓程式產生與單元測試產生兩種能力共同演化，不需要標準程式碼。
- Lin et al.（arXiv 2502.14948）：程式與測試產生的 self-play 框架。
- 綜述：自我演化 LLM（Tao et al., 2024，arXiv 2404.14387）與 RL 中的 self-play（Zhang et al., 2024，arXiv 2408.01072）。

這些工作中，Absolute Zero 與 CURE 依賴程式執行來驗證。R-Zero 把角色分成獨立訓練的 Challenger 與 Solver，並以多數決取代執行器（見 [多數決偽標籤](rzero-majority-vote-pseudo-labels.html)），因此限定在答案可客觀比對的數學題。模型自產推理軌跡再微調的較早做法見 [STaR：以模型自產的推理軌跡微調](r1-star-self-generated-reasoning.html)。
