---
slug: rti-elastic-budget-lora-training
title: 彈性預算與 LoRA 訓練：一個 checkpoint 對應任意 token 數
cluster: Diffusion & Generative Models
kind: topic
tags: [rti, elastic, lora, dc-dit, elit, budget]
arxiv: ["2608.29281", "2603.12245", "2603.06351", "2106.09685", "2412.16822"]
sources: ["2608.29281 §2, §3.2, §3.3, §4.1, §4.3, §5", "2603.12245 abstract", "2603.06351 abstract", "2106.09685 abstract", "2412.16822 abstract"]
related: ["rti-lineage-hub", "rti-read-write-interface", "tokc-learned-routing-dynamic-dit", "tokc-hourglass-and-latent-interfaces", "tokc-dynamic-chunking-bytes-to-pixels"]
---
# 彈性預算與 LoRA 訓練

屬於[RTI 研究脈絡總覽](rti-lineage-hub.html)。

## 訓練配方（§3.2、§4.1）

- 骨幹凍結，只訓練注意力與 MLP 投影上的 LoRA（Hu 等，arXiv 2106.09685）與 [Read/Write 介面](rti-read-write-interface.html)；介面初始化為恆等，訓練從凍結模型的行為開始。
- 每個訓練步從 {64, 128, 256, 512} 抽一個區域數 R。
- 沿用 MiniT2I 的微調配方：40K 步，約 120K 張對齊資料，512²。B/16 的可訓練參數為 16.3M（5.9%）。
- 推論時預算 R 由使用者在部署時設定，不在訓練時決定（§5）。

## 彈性（§4.3）

評估訓練時未見過的預算 R ∈ {96, 192, 384}，結果落在與四個訓練預算同一條曲線上，在此範圍內預算是連續旋鈕。速度在約 2.5× 飽和，R 約 192 以下省不到時間，因為 prelude、coda 與文字流不縮減。

## 與相近的彈性或自適應方法比較（§2）

- ELIT（Haji-Ali 等，arXiv 2603.12245）：可變長度 latent 介面，從頭訓練；其摘要描述為讓單一模型支援多種計算預算。RTI 稱其為最接近的工作，差別在 latent 是否錨定於空間區域。
- DC-DiT（Haridas 等，arXiv 2603.06351）：同樣可由預訓練 Transformer upcycle，但 RTI 指出其切塊由 router 學得、控制平均比例而非精確預算，且在已吸收部分冗餘的 VAE latent 上運作。RTI 的切割無參數，每張圖的區域數精確為 R。
- DiffCR（arXiv 2412.16822）等學得路由方法見[學得路由與動態 DiT](tokc-learned-routing-dynamic-dit.html)。

另見 [Hourglass 與 latent 介面](tokc-hourglass-and-latent-interfaces.html)與[動態切塊](tokc-dynamic-chunking-bytes-to-pixels.html)。

## 與減少取樣步數的比較（§4.2）

區域 token 使每步便宜 2.6×，同樣的時間可跑 2.6× 長的軌跡。在 1.8 秒時，稠密骨幹只能跑 6 步而無法成像，RTI 跑 16 步已清晰，GenEval 差 16.8 點；約 5 秒以上則增加步數成為更好的選擇。
