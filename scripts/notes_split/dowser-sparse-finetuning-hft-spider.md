---
slug: dowser-sparse-finetuning-hft-spider
title: Sparse fine-tuning：HFT 與 SPIDER
cluster: Memory & Continual Learning
kind: topic
tags: [sparse-fine-tuning, HFT, SPIDER, update-mask, memory-overhead]
arxiv: ["2404.18466", "2411.10928", "2602.04509"]
sources: ["arXiv abstract 2404.18466", "arXiv abstract 2411.10928", "2602.04509 §1, §2.1, §2.2, §3.3, §4.2 Table 3, Appendix E.2"]
related: ["dowser-functional-importance-theory-and-probing", "dowser-lineage-hub", "dowser-parameter-importance-pruning-scores", "dowser-post-merging-methods"]
---
# Sparse fine-tuning：HFT 與 SPIDER

Sparse fine-tuning 限制每一步只更新一部分參數，其餘維持預訓練值。Model-Dowser 論文把 HFT、SPIDER 歸為這一類（Model-Dowser 也屬於此類）。

**HFT（Half Fine-Tuning）。** Hui 等人，arXiv:2404.18466（論文引用其 ACL 2025 版本）。摘要指出：LLM 定期重設部分參數可找回部分原有知識；因此選一半參數學習新任務、另一半凍結。摘要把參數選擇解釋為正則化項。論文 §2.1 對這一類的批評是缺少有原則的重要性準則，可能改到對泛化關鍵的參數。

**SPIDER。** Huang 等人，Learn from Downstream and Be Yourself in Multimodal Large Language Model Fine-Tuning（arXiv:2411.10928）。摘要指出：它衡量參數對預訓練分佈與微調分佈的重要性，依據是凍結的預訓練權重大小與累積的微調梯度值，並以重要性感知的權重配置策略選擇性更新相對重要的參數。Model-Dowser 論文把它列為當時最佳的 MLLM sparse 方法。

**SPIDER 的成本（論文 §4.2 Table 3、Appendix E.2）。** SPIDER 需要維護每個參數的累積梯度歷史與 soft mask，記憶體為 O(3|P|)，是標準微調的三倍；Grafting 為 O(2|P|)；Model-Dowser 為 O(|P|)。論文的實驗中 SPIDER 在 LLaVA-1.5-7B 上，L > 20 時因記憶體無法訓練。論文也指出 SPIDER 的實驗僅微調 LLaVA 少於最後 5 層。

**Model-Dowser 的差異。** 遮罩只在微調前算一次，是靜態二元遮罩，所以不需額外儲存梯度歷史。遮罩與更新公式見 [功能重要性分數、定理與合成探測](dowser-functional-importance-theory-and-probing.html)。評分來源的比較見 [參數重要性與剪枝分數](dowser-parameter-importance-pruning-scores.html)；與 post-merging 的對照見 [Post-merging 方法](dowser-post-merging-methods.html)。

總覽：[Model-Dowser 的研究脈絡總覽](dowser-lineage-hub.html)
