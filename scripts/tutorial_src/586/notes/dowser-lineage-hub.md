---
slug: dowser-lineage-hub
title: Model-Dowser 的研究脈絡總覽
cluster: Memory & Continual Learning
kind: hub
tags: [catastrophic-forgetting, sparse-fine-tuning, parameter-importance, MLLM]
arxiv: ["1312.6211", "1612.00796", "1711.05769", "1711.09601", "2006.05467", "2106.09685", "2302.06600", "2304.08485", "2306.11695", "2309.10313", "2310.03744", "2311.03099", "2402.12048", "2402.17762", "2403.03003", "2404.18466", "2411.10928", "2412.04468", "2602.04509"]
sources: ["2602.04509 §1", "2602.04509 §2 Related Works", "2602.04509 reference list"]
related: ["dowser-catastrophic-forgetting-mllm-fine-tuning", "dowser-experiments-depth-and-memory", "dowser-functional-importance-theory-and-probing", "dowser-llava-nvila-setting", "dowser-parameter-importance-pruning-scores", "dowser-post-merging-methods", "dowser-sparse-finetuning-hft-spider", "odd-on-policy-forgetting-sparsity-claims"]
---
# Model-Dowser 的研究脈絡總覽

Model-Dowser（arXiv:2602.04509）處理的問題是：把多模態大型語言模型（MLLM）拿去微調下游任務時，預訓練任務的表現下降（catastrophic forgetting）。論文指出現有方法在微調較深（較早）的語言解碼器層時失效，或記憶體用量隨模型變大而過高。Model-Dowser 在微調前用合成輸入估計每個參數的重要性（權重大小 × 輸入激活 × 輸出敏感度），凍結重要性高的參數，只更新其餘參數。

閱讀順序如下。

1. [MLLM 微調中的 catastrophic forgetting](dowser-catastrophic-forgetting-mllm-fine-tuning.html)：問題定義、早期持續學習的重要性方法（EWC、MAS）、MLLM 上的遺忘觀察。
2. [LLaVA 與 NVILA：被微調的模型與評估設定](dowser-llava-nvila-setting.html)：解碼器加視覺編碼器的架構，以及 A_up、A_down、H-score。
3. [Post-merging 方法](dowser-post-merging-methods.html)：Grafting、DARE、Model Tailor，微調之後再融合權重。
4. [參數重要性與剪枝分數](dowser-parameter-importance-pruning-scores.html)：OBS、magnitude、SynFlow、Wanda、massive activations。
5. [Sparse fine-tuning：HFT 與 SPIDER](dowser-sparse-finetuning-hft-spider.html)：只更新一部分參數的做法，以及 SPIDER 的記憶體成本。
6. [功能重要性分數、定理與合成探測](dowser-functional-importance-theory-and-probing.html)：Theorem 3.1、Hutchinson 估計、無資料探測。
7. [實驗：微調深度與記憶體](dowser-experiments-depth-and-memory.html)：LLaVA-1.5-7B 與 NVILA-Lite-2B 的結果。

遺忘與更新稀疏在 on-policy 與 SFT 間的比較，是本站另一條脈絡，見 [On-policy 學習的遺忘與更新稀疏說法](odd-on-policy-forgetting-sparsity-claims.html)；本論文不涉及 on-policy 學習。時間順序的文獻清單見 timeline.json。前人工作的內容來自該篇 arXiv 摘要或 Model-Dowser 論文的描述，各筆記的 `sources` 欄位逐一標示。
