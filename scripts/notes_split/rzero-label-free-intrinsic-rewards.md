---
slug: rzero-label-free-intrinsic-rewards
title: 無標籤 RL：熵、信心與 self-certainty 獎勵
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [label-free-RL, entropy-minimization, self-certainty, RLIF, intrinsic-reward, confidence, self-entropy, RLVR]
arxiv: ["2508.05004", "2505.15134", "2505.22660", "2505.19590", "2504.05812", "2506.14758", "2506.06395", "2503.00031", "2510.14901", "2506.10947"]
sources: ["arXiv 2508.05004 §1", "arXiv abstract 2505.15134", "arXiv abstract 2505.22660", "arXiv abstract 2505.19590", "arXiv abstract 2504.05812", "arXiv 2508.05004 §1 對 2506.14758、2506.06395、2503.00031 的描述", "arXiv 2510.14901 §1, §2, §5.3", "arXiv abstract 2506.10947"]
related: ["rzero-lineage-hub", "rzero-majority-vote-pseudo-labels", "rzero-self-play-task-generation", "rswp-distribution-sharpening-debate", "rswp-power-sampling-algorithm-results", "r1-grpo-outcome-rewards"]
---
# 無標籤 RL：熵、信心與 self-certainty 獎勵

R-Zero 引言（§1）把「label-free RL」定義為從模型自己的輸出取得獎勵訊號，例如序列層級的信心分數（Li et al., 2025a；Prabhudesai et al., 2025；Huang et al., 2025）與輸出熵（Agarwal et al., 2025；Cheng et al., 2025）。這些方法移除了標準答案，但論文指出它們仍依賴既有的題目集合，因此不能算完全自我演化。

各方法在摘要中的內容如下：

- Agarwal et al.（arXiv 2505.15134）：以熵最小化訓練，提出 EM-FT、EM-RL、EM-INF 三種做法；摘要報告 Qwen-7B 上的 EM-RL 不用標註資料即可接近或超過以 6 萬筆標註資料訓練的 GRPO 與 RLOO。
- Prabhudesai et al.（arXiv 2505.22660，RENT，標題 Maximizing Confidence Alone Improves Reasoning）：以模型輸出分佈的熵作為內在獎勵，強化讓模型信心高的 chain of thought。在 2510.14901（冪次取樣）§5.3 中，論文採用其信心定義：生成序列上各步 next-token 分布的平均負熵，Conf(x) = (1/(T+1)) Σ_t Σ_x p(x|x<t) log p(x|x<t)。
- Zhao et al.（arXiv 2505.19590，Intuitor，標題 Learning to Reason without External Rewards）：提出 RLIF，以模型自己的信心（self-certainty）取代 GRPO 的外部獎勵。2510.14901 將其歸類為使用 self-entropy 的訊號。
- Shao et al.（arXiv 2506.10947）：2510.14901 列為使用隨機獎勵的工作。
- Zhang et al.（arXiv 2504.05812，EMPO）：在語意空間最小化未標註題目上的預測熵。
- Cheng et al.（arXiv 2506.14758）、Li et al.（arXiv 2506.06395）、Huang et al.（arXiv 2503.00031）：R-Zero 把它們歸入信心或熵相關的做法；本筆記僅依 R-Zero 的描述記錄，未另行摘錄其摘要。

這些方法的共同前提是題目已存在；R-Zero 進一步讓題目也由模型產生，見 [自我對弈與自產任務](rzero-self-play-task-generation.html)。另一種不靠標準答案的做法是多數決，見 [多數決偽標籤](rzero-majority-vote-pseudo-labels.html)。

在 2510.14901（冪次取樣）§2 中，這批不依賴外部驗證器的工作被視為繼 RLVR 之後的一條脈絡；論文表示該文與它們相似之處在於利用 base model 的概似度提升推理，差別在於該文不訓練。§5.3 的 Figure 4 以上述信心指標顯示：GRPO 與冪次取樣的回答都落在 base model 的高信心區域，與較高的正確率相關。這說明為何「只用模型自身機率」就可能改善推理：與前述工作一致，改善可能來自把機率集中到模型原本就偏好的區域（見 [分布銳化之爭](rswp-distribution-sharpening-debate.html)）。數字與實驗見 [冪次取樣演算法與結果](rswp-power-sampling-algorithm-results.html)。相對的規則式結果獎勵見 [GRPO 與規則式結果獎勵](r1-grpo-outcome-rewards.html)。
