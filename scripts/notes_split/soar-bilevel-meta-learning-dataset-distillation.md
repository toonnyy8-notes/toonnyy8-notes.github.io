---
slug: soar-bilevel-meta-learning-dataset-distillation
title: 雙層最佳化：meta-learning 與 dataset distillation
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [bilevel, meta-learning, MAML, Reptile, dataset-distillation, BPTT]
arxiv: ["2601.18778", "1703.03400", "1803.02999", "1502.03492", "1611.02779", "1811.10959"]
sources: ["arXiv:2601.18778 §3.1, §A.3（本文對各工作的描述）", "arXiv API（題名、作者、年份、ID 核對）"]
related: ["soar-intrinsic-vs-grounded-reward", "soar-teacher-student-rloo-method"]
---
# 雙層最佳化：meta-learning 與 dataset distillation

本文 §3.1 把 teacher 的目標寫成雙層最佳化：外層最佳化 teacher 參數 φ，使其產生的合成資料 X 在經過 student 的 RL 更新後，能提升 student 在 D_train 上的表現。本文指出這類目標在以下領域有先例：

- **meta-learning**：MAML（Finn et al.，arXiv 1703.03400，2017）與 Reptile（Nichol et al.，arXiv 1803.02999，2018，On First-Order Meta-Learning Algorithms）以內外層結構訓練少樣本學習器；本文 §A.3 說明其承襲 RL²（Duan et al.，arXiv 1611.02779，2016）等較早研究。
- **超參數學習**：Maclaurin et al.（arXiv 1502.03492，2015）透過對整個訓練過程反向傳播學習超參數。
- **dataset distillation**：Wang et al.（arXiv 1811.10959，2018）以外層最佳化一個小資料集，使內層訓練在目標任務上表現良好。本文 §2 指出此為啟發本文的來源。

本文指出這類方法的困難：內層迴圈包含大量步驟，需要 BPTT（「backpropagation through gradient descent」），展開內層迴圈並計算 meta-gradient，通常不可行。SOAR 的做法（本文自述）是在外層使用 RLOO，將 student 的進步量當作獎勵來強化問答組，因此不需展開內層迴圈；本文稱此為其所知的第一個用於 LLM self-play 的「雙重 meta-RL 迴圈」。實作細節見 [SOAR：以 RLOO 訓練的 teacher–student 迴圈](soar-teacher-student-rloo-method.html)。

代理獎勵常被採用的原因之一，正是避開此雙層問題，見 [內在獎勵與接地獎勵](soar-intrinsic-vs-grounded-reward.html)。
