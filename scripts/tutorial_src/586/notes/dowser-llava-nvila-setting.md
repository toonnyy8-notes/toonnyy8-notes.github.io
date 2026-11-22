---
slug: dowser-llava-nvila-setting
title: LLaVA 與 NVILA：被微調的模型與評估設定
cluster: Memory & Continual Learning
kind: topic
tags: [LLaVA, NVILA, MLLM, evaluation, H-score]
arxiv: ["2304.08485", "2310.03744", "2403.03003", "2412.04468", "2602.04509"]
sources: ["arXiv abstract 2304.08485", "arXiv abstract 2310.03744", "arXiv abstract 2412.04468", "2602.04509 §1, §4.1, Appendix E"]
related: ["dowser-lineage-hub"]
---
# LLaVA 與 NVILA：被微調的模型與評估設定

**MLLM 的組成。** 論文 §1 描述 MLLM 由預訓練的大型語言解碼器（通常是 transformer）與視覺編碼器，經連接模組串接而成。下游微調時，通常最佳化語言解碼器，其他元件凍結。

**兩個實驗模型。**

- LLaVA：Liu 等人，Visual Instruction Tuning（arXiv:2304.08485）提出視覺指令微調；Improved Baselines with Visual Instruction Tuning（arXiv:2310.03744）是 LLaVA-1.5 的出處。Model-Dowser 使用 LLaVA-1.5-7B（解碼器 32 層）。
- NVILA：NVILA: Efficient Frontier Visual Language Models（arXiv:2412.04468）。Model-Dowser 使用 NVILA-Lite-2B。

**微調設定（論文 §4.1）。** 每個下游資料集隨機取 10k 筆訓練資料，學習率 2×10^-5，訓練 5 個 epoch；只更新語言解碼器的最後 L 層，L 以 4 為間隔，LLaVA 從 4 到 32，NVILA 從 4 到 28。

**資料集。** 下游：COCO-Caption、Flickr30k（影像描述，用 CIDEr）、ImageNet-R（影像分類）、IconQA（VQA）；後兩者用 Exact Match。上游（衡量遺忘）：TextVQA、OKVQA、OCRVQA、GQA、MMBench 英文與中文（MMB、MMB (CN)）。

**指標。** A_up 是上游任務平均準確率（依 Luo 等人，arXiv:2403.03003 的評估協議），A_down 是下游任務表現。論文同時報告算術平均 Avg = (A_up + A_down)/2 與調和平均 H-score = 2·A_up·A_down/(A_up + A_down)。H-score 在任一端偏低時會被拉低，用來衡量適應與保留的平衡。

總覽：[Model-Dowser 的研究脈絡總覽](dowser-lineage-hub.html)
