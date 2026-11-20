---
slug: soar-intrinsic-vs-grounded-reward
title: 內在獎勵與接地獎勵
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [intrinsic-reward, curiosity, grounded-reward, proxy-reward]
arxiv: ["2601.18778", "1705.05363", "2505.21444", "2508.03682"]
sources: ["arXiv:2601.18778 §2, §A.3（本文對各工作的描述）", "arXiv API（題名、作者、年份、ID 核對）"]
related: ["soar-llm-data-free-self-play", "soar-bilevel-meta-learning-dataset-distillation", "test-time-training-for-rl"]
---
# 內在獎勵與接地獎勵

本文 §2、§A.3 列出內在獎勵在機器人、模擬與任務無關設定中，用於課程生成與探索的研究：Schmidhuber（1991）、Pathak et al.（arXiv 1705.05363，2017，Curiosity-driven Exploration by Self-supervised Prediction）、Colas et al.（2019、2022）、Blaes et al.（2019）、Sancaktar et al.（2023）。所最大化的量包括估計的學習進度、規律性與預測誤差。

在近期 LLM 自我改進工作中，本文列出的代理獎勵包括：模型信心（Intuitor）、預測熵（DP、RENT）、多數答案（TTRL；Shafayat et al.，arXiv 2505.21444；SQLM，arXiv 2508.03682）。本文指出幾乎所有近期「資料自由」self-play 方法都以內在或代理獎勵訓練 teacher 或 proposer，沒有錨定到真實的 student 表現；唯一例外是 Zweiger et al.（2025）的自我適應工作，其內外層分別使用 ReSTEM 與 SFT。

SOAR 的獎勵（本文自述）只在合成問答使 student 在真實題目上進步時才給予，此黑盒訊號將產題綁定到實際學習進度，間接懲罰退化題目與 reward hacking。teacher 訓練時看不到困難題目，只從 student 進步量發現有用的墊腳石。

代理獎勵常是務實的簡化，因為可避開明確的雙層最佳化，見 [雙層最佳化](soar-bilevel-meta-learning-dataset-distillation.html)。以代理獎勵的 self-play 工作與其崩潰現象見 [LLM 資料自由 self-play 與自我獎勵的崩潰](soar-llm-data-free-self-play.html)，多數決型 test-time RL 見 [測試時訓練與 test-time RL](test-time-training-for-rl.html)。
