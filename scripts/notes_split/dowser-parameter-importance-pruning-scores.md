---
slug: dowser-parameter-importance-pruning-scores
title: 參數重要性與剪枝分數：OBS、magnitude、SynFlow、Wanda
cluster: Memory & Continual Learning
kind: topic
tags: [parameter-importance, pruning, Wanda, SynFlow, magnitude]
arxiv: ["1711.05769", "2006.05467", "2306.11695", "2402.17762", "2602.04509"]
sources: ["arXiv abstract 2006.05467", "arXiv abstract 2306.11695", "arXiv abstract 1711.09601", "2602.04509 §2.2, §3.1", "2602.04509 reference list (1711.05769, 2402.17762)"]
related: ["dowser-catastrophic-forgetting-mllm-fine-tuning", "dowser-functional-importance-theory-and-probing", "dowser-lineage-hub", "dowser-sparse-finetuning-hft-spider"]
---
# 參數重要性與剪枝分數

Model-Dowser 的分數沿用剪枝文獻中「為每個參數打分」的作法。論文 §2.2 依序討論下列方法，並說明各自的限制。

**二階方法：Optimal Brain Surgeon。** Hassibi 與 Stork（1992）。論文認為二階方法對基礎模型的計算成本過高。這篇不在 arXiv，本站未驗證其內容，只記錄論文的說法。

**Magnitude 方法。** 論文 §2.2 引用 PackNet（Mallya 與 Lazebnik，arXiv:1711.05769，"PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning"）與 SynFlow（Tanaka 等人，arXiv:2006.05467）。SynFlow 摘要指出：它在初始化時不看資料就找出稀疏的可訓練子網路，並以守恆定律解釋梯度式剪枝為何出現 layer-collapse。論文的批評是：magnitude 準則假設激活函數齊次，而 GELU、SiLU、GLU 變體等非齊次激活下，權重大小不再對應輸出層級的功能影響（§3.1）。

**Wanda。** Sun 等人，A Simple and Effective Pruning Approach for Large Language Models（arXiv:2306.11695）。摘要指出：它剪掉「權重大小乘以對應輸入激活」最小的權重（逐輸出節點），不需重訓或權重更新。論文的批評是：這類經驗準則（基於權重、激活或梯度統計）缺少對輸出層級功能穩定性影響的理論分析；並且在 massive activations（Sun 等人，arXiv:2402.17762）出現時可能不可靠（§2.2）。

**Memory Aware Synapses（MAS）。** 見 [MLLM 微調中的 catastrophic forgetting](dowser-catastrophic-forgetting-mllm-fine-tuning.html)：以輸出對參數的敏感度累積重要性。

**Model-Dowser 的位置。** 論文 §3.1 的說法：它的分數 S = ‖J_i‖₂ · |W_ij| · |h_j| 整合了「結構流方法（structural flow）的梯度敏感度」與「局部剪枝（如 Wanda）的激活感知」，並由 Theorem 3.1 導出。與 Wanda 相比多了輸出敏感度項 ‖J_i‖₂；與純 magnitude 相比多了輸入激活與輸出敏感度。推導見 [功能重要性分數、定理與合成探測](dowser-functional-importance-theory-and-probing.html)。

**用途的差異。** 剪枝用分數決定「刪除哪些權重」；Model-Dowser 用分數決定「微調時凍結哪些權重」。使用方式見 [Sparse fine-tuning：HFT 與 SPIDER](dowser-sparse-finetuning-hft-spider.html)。

總覽：[Model-Dowser 的研究脈絡總覽](dowser-lineage-hub.html)
