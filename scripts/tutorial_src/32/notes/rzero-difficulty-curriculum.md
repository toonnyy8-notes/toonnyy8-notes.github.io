---
slug: rzero-difficulty-curriculum
title: 難度課程：成功率接近 50% 的題目
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [curriculum-learning, difficulty-filtering, AdaRFT, GRPO, reward-variance]
arxiv: ["2508.05004", "2504.05520", "2504.03380", "2504.09696", "2505.00949"]
sources: ["arXiv 2508.05004 §2.2、§2.3、§3.1、附錄 F", "arXiv abstract 2504.05520", "arXiv abstract 2504.03380", "arXiv abstract 2504.09696"]
related: ["rzero-lineage-hub", "rzero-challenger-solver-pipeline", "rzero-grpo-rlvr-foundation"]
---
# 難度課程：成功率接近 50% 的題目

二元獎勵下，若一題的成功率 p̂ 是 0 或 1，組內回答的獎勵全相同，GRPO 的優勢為零，沒有學習訊號。R-Zero 因此要求題目難度落在 Solver 能力的邊緣。

R-Zero 附錄 F 引用的理論依據來自 Shi et al.（2025a）與 Bae et al.（2025）：學習潛力以與最優策略的 KL 散度衡量，其下界正比於獎勵變異數；二元獎勵下為 p̂(1 − p̂)/(2β²)，在 p̂ = 0.5 最大。對應的設計：

- Challenger 的不確定性獎勵 1 − 2|p̂ − 1/2|，在 p̂ = 0.5 時最大（§2.2）。
- Solver 資料集只保留 |p̂ − 1/2| ≤ δ 的題目；m = 10、δ = 0.25 時等於保留 3 至 7 個答案與多數一致的題目。論文指出這個範圍與 Zhang & Zuo（2025）、Li et al.（2025b）、Bercovich et al.（2025）的做法一致（§3.1）。

來源論文摘要的內容：

- AdaRFT（Shi et al.，arXiv 2504.05520）：依模型近期獎勵動態調整訓練題目難度，讓模型持續訓練在有挑戰但可解的題目。
- Bae et al.（arXiv 2504.03380）：對線上難度過濾做形式化分析，預期的策略改善下界由任務成功機率的變異數決定，中等難度的題目學習效率最高。
- GRPO-LEAD（Zhang & Zuo，arXiv 2504.09696）：對 GRPO 加入長度正則化獎勵、錯誤解答懲罰與依難度重新加權的優勢。
- Llama-Nemotron（arXiv 2505.00949）：R-Zero 引用其過濾範圍作為依據；本筆記未摘錄其內容。

這些方法的題目來自固定資料集；R-Zero 的差別是由 Challenger 以 RL 學習產生這個難度的題目。消融實驗中移除過濾使一般領域平均從 31.15 降到 26.69（表 3）。GRPO 背景見 [R-Zero 使用的訓練基礎：GRPO 與 RLVR](rzero-grpo-rlvr-foundation.html)。
