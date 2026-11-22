---
slug: dowser-catastrophic-forgetting-mllm-fine-tuning
title: MLLM 微調中的 catastrophic forgetting
cluster: Memory & Continual Learning
kind: topic
tags: [catastrophic-forgetting, continual-learning, EWC, MAS, MLLM]
arxiv: ["1312.6211", "1612.00796", "1711.09601", "2106.09685", "2309.10313", "2602.04509"]
sources: ["arXiv abstract 1312.6211", "arXiv abstract 1612.00796", "arXiv abstract 1711.09601", "arXiv abstract 2309.10313", "2602.04509 §1, §2.1, Appendix A", "2602.04509 reference list"]
related: ["dowser-lineage-hub", "dowser-post-merging-methods", "dowser-sparse-finetuning-hft-spider", "odd-on-policy-forgetting-sparsity-claims"]
---
# MLLM 微調中的 catastrophic forgetting

**定義。** Catastrophic forgetting 指模型適應新任務之後，在先前學過或未見過的任務上失去能力。Model-Dowser 論文引用 Goodfellow 等人（arXiv:1312.6211，"An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks"）作為這個現象的出處之一。

**早期設定：持續學習。** 論文的 Appendix A 說明，早期研究多在單模態模型的持續學習設定下進行：任務依序訓練，來源任務的訓練資料在適應期間仍可取得。其中有兩個與本論文直接相關的方法：

- EWC（Kirkpatrick 等人，arXiv:1612.00796，"Overcoming catastrophic forgetting in neural networks"）。
- Memory Aware Synapses（Aljundi 等人，arXiv:1711.09601）。其摘要指出，它以無監督、線上的方式計算每個參數的重要性，依預測輸出函數對參數變動的敏感度累積重要性。這個「以輸出敏感度衡量參數重要性」的想法，與 Model-Dowser 的分數同一方向；差別在於 MAS 累積真實樣本，Model-Dowser 用合成探測。

**基礎模型的設定不同。** 論文 Appendix A 的說法是：基礎模型的預訓練資料分佈常常未知或取不到，遺忘表現為微調下游任務後的泛化能力下降。因此需要不依賴原始資料的方法。

**在 MLLM 上的觀察。** Zhai 等人（arXiv:2309.10313）的摘要指出，catastrophic forgetting 在 MLLM 中仍是固有問題；他們提出 EMT，把每個 MLLM 當成影像分類器來評估遺忘，並發現幾乎所有被評估的微調後 MLLM 在標準影像分類上都達不到其視覺編碼器的表現。Model-Dowser 論文的 §1 說明，下游指令資料通常量少且範圍窄，微調會覆蓋預訓練表徵。

**緩解方法的三類（論文 §2.1）。**

1. Additive 方法：新增少量參數、固定預訓練權重，例如 LoRA（arXiv:2106.09685）。論文認為架構修改使部署與持續微調變得複雜。
2. Post-merging 方法：見 [Post-merging 方法](dowser-post-merging-methods.html)。
3. Sparse fine-tuning：見 [Sparse fine-tuning：HFT 與 SPIDER](dowser-sparse-finetuning-hft-spider.html)。論文認為這類方法缺少有原則的參數重要性準則。

**與本站其他筆記的關係。** 另一條脈絡比較 on-policy 與 SFT 的遺忘，見 [On-policy 學習的遺忘與更新稀疏說法](odd-on-policy-forgetting-sparsity-claims.html)。Model-Dowser 只討論監督式微調（SFT 式的下游適應）。

總覽：[Model-Dowser 的研究脈絡總覽](dowser-lineage-hub.html)
