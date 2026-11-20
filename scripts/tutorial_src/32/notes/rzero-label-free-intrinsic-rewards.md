---
slug: rzero-label-free-intrinsic-rewards
title: 無標籤 RL：熵、信心與 self-certainty 獎勵
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [label-free-RL, entropy-minimization, self-certainty, RLIF]
arxiv: ["2508.05004", "2505.15134", "2505.22660", "2505.19590", "2504.05812", "2506.14758", "2506.06395", "2503.00031"]
sources: ["arXiv 2508.05004 §1", "arXiv abstract 2505.15134", "arXiv abstract 2505.22660", "arXiv abstract 2505.19590", "arXiv abstract 2504.05812", "arXiv 2508.05004 §1 對 2506.14758、2506.06395、2503.00031 的描述"]
related: ["rzero-lineage-hub", "rzero-majority-vote-pseudo-labels", "rzero-self-play-task-generation"]
---
# 無標籤 RL：熵、信心與 self-certainty 獎勵

R-Zero 引言（§1）把「label-free RL」定義為從模型自己的輸出取得獎勵訊號，例如序列層級的信心分數（Li et al., 2025a；Prabhudesai et al., 2025；Huang et al., 2025）與輸出熵（Agarwal et al., 2025；Cheng et al., 2025）。這些方法移除了標準答案，但論文指出它們仍依賴既有的題目集合，因此不能算完全自我演化。

各方法在摘要中的內容如下：

- Agarwal et al.（arXiv 2505.15134）：以熵最小化訓練，提出 EM-FT、EM-RL、EM-INF 三種做法；摘要報告 Qwen-7B 上的 EM-RL 不用標註資料即可接近或超過以 6 萬筆標註資料訓練的 GRPO 與 RLOO。
- Prabhudesai et al.（arXiv 2505.22660，RENT）：以模型輸出分佈的熵作為內在獎勵，強化讓模型信心高的 chain of thought。
- Zhao et al.（arXiv 2505.19590，Intuitor）：提出 RLIF，以模型自己的信心（self-certainty）取代 GRPO 的外部獎勵。
- Zhang et al.（arXiv 2504.05812，EMPO）：在語意空間最小化未標註題目上的預測熵。
- Cheng et al.（arXiv 2506.14758）、Li et al.（arXiv 2506.06395）、Huang et al.（arXiv 2503.00031）：R-Zero 把它們歸入信心或熵相關的做法；本筆記僅依 R-Zero 的描述記錄，未另行摘錄其摘要。

這些方法的共同前提是題目已存在；R-Zero 進一步讓題目也由模型產生，見 [自我對弈與自產任務](rzero-self-play-task-generation.html)。另一種不靠標準答案的做法是多數決，見 [多數決偽標籤](rzero-majority-vote-pseudo-labels.html)。
