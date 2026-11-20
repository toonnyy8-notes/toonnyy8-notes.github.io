---
slug: soar-llm-data-free-self-play
title: LLM 資料自由 self-play 與自我獎勵的崩潰
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [self-play, R-Zero, Absolute-Zero, SQLM, reward-hacking, collapse]
arxiv: ["2601.18778", "2505.03335", "2508.05004", "2509.07414", "2505.20347", "2508.03682", "2505.21444"]
sources: ["arXiv:2601.18778 §2, §A.2, §A.3（本文對各工作的描述）", "arXiv API（題名、作者、年份、ID 核對）"]
related: ["soar-intrinsic-vs-grounded-reward", "r1-star-self-generated-reasoning", "test-time-training-for-rl"]
---
# LLM 資料自由 self-play 與自我獎勵的崩潰

本文 §2 描述一系列與本文接近同期的工作，以預訓練 LLM 作為問題產生器，建立「完全不需資料」的共同演化系統：

- Absolute Zero（Zhao et al.，arXiv 2505.03335，2025）
- R-Zero（Huang et al.，arXiv 2508.05004，2025；ICLR 2026）
- Language Self-Play, LSP（Kuba et al.，arXiv 2509.07414，2025）
- SeRL（Fang et al.，arXiv 2505.20347，2025）
- Self-Questioning Language Models, SQLM（Chen et al.，arXiv 2508.03682，2025）

本文指出這些工作以多數決、learnability、reward model 偏好或梯度大小等內在或代理獎勵訓練產題者，因此可能漂向退化或不可學的任務、易受 reward hacking 影響，且缺乏進展保證（Chae et al., 2025 對 Absolute Zero 的分析）。長期以自我獎勵進行 RL 常導致突然且完全的表現崩潰（Shafayat et al.，arXiv 2505.21444，2025，Can Large Reasoning Models Self-Train?；Chae et al.，2025），尤其在近乎二元正確性訊號的離散符號領域。

較早的 LLM self-play 方法（SPIN、Triplet self-play、ReSTEM、eva）本文描述為以自洽或解答品質為目標，仍預設已有格式良好的題目。以模型自產推理軌跡做微調的較早例子見 [STaR：以模型自產的推理軌跡微調](r1-star-self-generated-reasoning.html)；以多數決為獎勵的 test-time RL 見 [測試時訓練與 test-time RL](test-time-training-for-rl.html)。

SOAR 的主張（本文自述）是此類方法中第一個把課程生成綁定到具體失敗情境，而非內部難度代理。代理獎勵與接地獎勵的對照見 [內在獎勵與接地獎勵](soar-intrinsic-vs-grounded-reward.html)。
