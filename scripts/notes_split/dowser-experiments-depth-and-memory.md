---
slug: dowser-experiments-depth-and-memory
title: Model-Dowser 實驗：微調深度與記憶體
cluster: Memory & Continual Learning
kind: topic
tags: [Model-Dowser, fine-tuning-depth, H-score, memory-complexity, LLaVA, NVILA]
arxiv: ["2602.04509"]
sources: ["2602.04509 §4.1-§4.2, Tables 1-3, Figures 1, 3", "2602.04509 Conclusion"]
related: ["depth-nonuniform-lora-rank", "dowser-lineage-hub", "dowser-llava-nvila-setting", "dowser-post-merging-methods", "dowser-sparse-finetuning-hft-spider"]
---
# Model-Dowser 實驗：微調深度與記憶體

設定見 [LLaVA 與 NVILA](dowser-llava-nvila-setting.html)。基線為 Full-FT、Grafting、DARE、Model Tailor、SPIDER（方法說明見 [Post-merging 方法](dowser-post-merging-methods.html)、[Sparse fine-tuning](dowser-sparse-finetuning-hft-spider.html)）。

**微調深度的發現。** 論文 §4.2 指出，當微調延伸到較早的解碼器層，遺忘最嚴重。Grafting、DARE、Tailor 在最後 4–16 層穩定，更深就崩潰；SPIDER 能維持較多層，但即使微調全部 32 層，結果仍低於 Model-Dowser。論文引用的動機是：早期解碼器層對多模態理解有重要作用（§1），所以淺層微調未必能發揮適應能力。

**H-score 結果（§4.2，更新最後 20 層、ρ = 0.1）。** NVILA-Lite-2B 在 COCO-Caption 與 ImageNet-R 上，Model-Dowser 的 H-score 分別為 85.7 與 71.2；LLaVA-1.5-7B 為 79.9 與 69.7。論文的解讀是：遺忘較少是因為保留了功能敏感的參數，而不是下游適應不足。

**記憶體（Table 3）。** Full FT、DARE、Tailor、Model-Dowser 的記憶體複雜度為 O(|P|)；Grafting 為 O(2|P|)；SPIDER 為 O(3|P|)。更新最後 20 層時，可學習參數在 NVILA / LLaVA 分別為：Full FT 1.4B / 4.5B；ρ = 10% 的方法（Grafting、DARE、Tailor、Model-Dowser）143M / 438M；SPIDER（ρ = 50%）714M / 2.3B。Grafting 與 SPIDER 在實驗中改用 NVIDIA H200（143 GB）訓練。

**結論欄位。** 論文結論是 Model-Dowser 不需額外記憶體，因此可擴展到數十億參數的 MLLM。論文僅在 LLaVA-1.5-7B 與 NVILA-Lite-2B 驗證（因運算資源限制），更大模型的結果不在本文中。

**與本站其他筆記的關係。** 「依層深度決定更新方式」的另一個例子見 [深度非均勻 LoRA rank 分配](depth-nonuniform-lora-rank.html)；那篇討論的是 rank 配置，不是遺忘。

總覽：[Model-Dowser 的研究脈絡總覽](dowser-lineage-hub.html)
