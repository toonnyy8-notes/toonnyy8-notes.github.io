---
slug: soar-lineage-hub
title: SOAR 的研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [SOAR, hub, self-play, curriculum, meta-RL]
arxiv: ["2601.18778", "2501.12948", "1703.05407", "2101.04882", "2012.02096", "2508.05004", "2505.03335", "1703.03400", "1811.10959", "2402.14740"]
sources: ["arXiv:2601.18778 §1–§3, §A（以本文參考文獻為主幹）", "arXiv API（題名、作者、年份、ID 核對）"]
related: ["soar-rlvr-sparse-reward-plateau", "soar-asymmetric-self-play-origins", "soar-curriculum-and-environment-design", "soar-llm-data-free-self-play", "soar-bilevel-meta-learning-dataset-distillation", "soar-intrinsic-vs-grounded-reward", "soar-teacher-student-rloo-method", "r1-lineage-hub"]
---
# SOAR 的研究脈絡總覽

SOAR（arXiv 2601.18778，Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability）處理的問題是：RLVR 在初始成功率為 0 的困難題上沒有訊號，模型能否自行產生墊腳石課程。以下依閱讀順序列出所需背景。

1. [RLVR 在低成功率資料上的停滯](soar-rlvr-sparse-reward-plateau.html)：問題設定，連到 DeepSeek-R1 與 GRPO 相關筆記（總覽見 [DeepSeek-R1 的研究脈絡總覽](r1-lineage-hub.html)）。
2. [非對稱 self-play 的來源](soar-asymmetric-self-play-origins.html)：AlphaZero、Alice–Bob、機器人操作中的非對稱 self-play。
3. [課程學習與非監督環境設計](soar-curriculum-and-environment-design.html)：課程學習、PAIRED、PLR 與 LLM 課程方法。
4. [LLM 資料自由 self-play 與自我獎勵的崩潰](soar-llm-data-free-self-play.html)：Absolute Zero、R-Zero 等近期工作及其失效模式。
5. [雙層最佳化](soar-bilevel-meta-learning-dataset-distillation.html)：MAML、Reptile、dataset distillation 與 BPTT 的困難。
6. [內在獎勵與接地獎勵](soar-intrinsic-vs-grounded-reward.html)：SOAR 與既有代理獎勵設計的差別。
7. [SOAR：以 RLOO 訓練的 teacher–student 迴圈](soar-teacher-student-rloo-method.html)：本文方法的外層、內層與晉升機制。

主線：非對稱 self-play 提供 teacher–student 結構（2、3）；既有 LLM self-play 使用代理獎勵而易崩潰（4、6）；SOAR 以雙層目標（5）和 RLOO 的外層更新（7）把獎勵綁定到 student 的真實進步。年代對照見 timeline.json。
