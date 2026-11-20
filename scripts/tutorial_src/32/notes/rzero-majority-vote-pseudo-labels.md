---
slug: rzero-majority-vote-pseudo-labels
title: 多數決偽標籤：TTRL 與 self-consistency 訓練
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [majority-vote, pseudo-label, TTRL, self-consistency, self-training]
arxiv: ["2508.05004", "2504.16084", "2411.04109", "2505.21444", "2506.08745", "2509.15194", "2505.20347"]
sources: ["arXiv 2508.05004 §2.3、§2.4、§1", "arXiv abstract 2504.16084", "arXiv abstract 2411.04109", "arXiv abstract 2505.21444", "arXiv abstract 2506.08745", "arXiv abstract 2509.15194", "arXiv abstract 2505.20347"]
related: ["rzero-lineage-hub", "rzero-label-free-intrinsic-rewards", "rzero-challenger-solver-pipeline", "rzero-self-training-collapse", "test-time-training-for-rl"]
---
# 多數決偽標籤：TTRL 與 self-consistency 訓練

R-Zero 的 Solver 在每題取樣 m = 10 個回答，出現最多次的答案當偽標籤 ỹ，與偽標籤一致的回答獲得獎勵 1（§2.3、§2.4）。這個做法延續下列研究：

- TTRL（Zuo et al., 2025，arXiv 2504.16084）：摘要指出 test-time scaling 常用的多數決可以當 RL 的獎勵，在沒有標籤的資料上訓練。TTRL 的整體脈絡見既有筆記 [測試時訓練與 test-time RL](test-time-training-for-rl.html)。
- ScPO（Prasad et al., 2024，arXiv 2411.04109）：把推論時的 self-consistency 用在訓練，讓一致的答案在偏好學習中優於不一致的答案。
- Shafayat et al.（arXiv 2505.21444）：以多數決作自我回饋研究 RL 能否自我訓練；摘要同時報告長時間訓練會出現 reward hacking 並導致效能驟降。
- SeRL（Fang et al., 2025，arXiv 2505.20347）：以 self-instruction 產生額外指令，並用多數決估計回答的獎勵。
- CoVo（Zhang et al., 2025a，arXiv 2506.08745）：以推理軌跡中間狀態的一致性與波動作為內在獎勵。
- EVOL-RL（Zhou et al., 2025b，arXiv 2509.15194）：以多數決作為穩定的選擇訊號，並加入基於 embedding 的新穎度以避免多樣性崩壞。

R-Zero 與 TTRL 的差別在於題目來源：TTRL 使用既有的未標註題目，R-Zero 的題目由 Challenger 產生。多數決的可靠度隨題目難度下降；R-Zero 因此只保留 3 至 7 個答案與多數答案一致的題目（m = 10，δ = 0.25，§3.1），並說明這個過濾同時是品質控制（§2.3）。偽標籤品質退化與崩壞的關係見 [自我訓練的崩壞風險](rzero-self-training-collapse.html)。
