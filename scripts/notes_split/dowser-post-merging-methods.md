---
slug: dowser-post-merging-methods
title: Post-merging 方法：Grafting、DARE、Model Tailor
cluster: Memory & Continual Learning
kind: topic
tags: [post-merging, model-grafting, DARE, Model-Tailor, task-vector]
arxiv: ["2302.06600", "2311.03099", "2402.12048", "2602.04509"]
sources: ["arXiv abstract 2302.06600", "arXiv abstract 2402.12048", "2602.04509 §1, §2.1, §2.2, §4.2, Appendix E.2", "2602.04509 reference list (2311.03099)"]
related: ["dowser-experiments-depth-and-memory", "dowser-lineage-hub"]
---
# Post-merging 方法

Post-merging 的做法是先照常微調，再把預訓練權重與微調後權重融合，以找回預訓練任務的表現。Model-Dowser 論文把 Grafting、DARE、Model Tailor 當作這一類的基線。

**Model Grafting。** Panigrahi 等人，Task-Specific Skill Localization in Fine-tuned Language Models（arXiv:2302.06600）。摘要指出：給定下游任務與微調後模型，用簡單的最佳化找出極小的參數子集（約 0.01%），把這些微調後的值嫁接到預訓練模型上，就能達到近似微調模型的表現。Model-Dowser 論文 Appendix E.2 的描述：它對融合遮罩做下游損失最佳化，不保證預訓練任務的表現，且訓練需要標準微調兩倍的記憶體。

**DARE。** Yu 等人，Language Models are Super Mario（arXiv:2311.03099）。Model-Dowser 論文 Appendix E.2 的描述：不需額外訓練，先隨機丟棄一部分 delta weights，再重新縮放其餘部分，之後加回預訓練模型；套用到 MLLM 時結果次佳且不穩定。

**Model Tailor。** Zhu 等人（arXiv:2402.12048）。摘要指出：它針對 MLLM 的遺忘，保留預訓練參數，只替換少量（≤10%）微調後參數，遮罩由結合 salience 與 sensitivity 分析的融合策略得到，並有補償機制。Model-Dowser 論文稱它是專為 MLLM 探討此問題的早期方法，但會為了保留預訓練表徵而犧牲目標任務表現。

**論文對這一類方法的批評。**

- 在 §2.1：這類方法難以平衡通用知識與領域適應，對超參數敏感。
- 在 §1 與 §4.2：Model Tailor 只微調 LLaVA 最後 12 層；當微調延伸到較早的解碼器層，post-merging 方法快速退化。論文推測原因是大幅更新破壞了預訓練的潛在空間，無法由事後融合復原；§4.2 稱這類方法本質上是被動的。

相對地，Model-Dowser 在微調前決定要凍結哪些參數，微調過程中不再需要融合步驟。實驗結果見 [實驗：微調深度與記憶體](dowser-experiments-depth-and-memory.html)。

總覽：[Model-Dowser 的研究脈絡總覽](dowser-lineage-hub.html)
