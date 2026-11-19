---
slug: relaxed-recursive-transformer
title: Relaxed Recursive Transformer（RRT）與層級 LoRA 鬆綁
cluster: Architectures & Efficiency
kind: topic
tags: [recursive-transformer, lora, svd-init, uptraining, early-exit, gemma]
sources: [weight-sharing-lora.md]
arxiv: ["2410.20672"]
related: ["weight-sharing-recursive-overview", "lottery-ticket-redundancy-scaffold", "depth-nonuniform-lora-rank", "recursive-transformer-followups"]
---
# Relaxed Recursive Transformer（RRT）與層級 LoRA 鬆綁

Bae et al., arXiv:2410.20672, ICLR 2025；KAIST × Google DeepMind。屬於 [層間權重共享總覽](weight-sharing-recursive-overview.html)。

## 方法四要素
- **Looped layer tying（CYCLE 策略）**：Gemma 2B（18 層）只存前 9 層、迴圈兩次（ [arxiv](https://arxiv.org/html/2410.20672v3) B=2 blocks）。
- **層間初始化策略**：Stepwise（固定首尾、間隔取中間層）、Average（平均權重）、Lower（取前 K 層）。 [arxiv](https://arxiv.org/html/2410.20672v3) 非鬆綁遞迴模型 Stepwise 最佳（15B token、SlimPajama：PPL 12.85 / Avg 51.7，優於 Average 46.7、Lower 47.1、Random 37.6）。
- **層級 LoRA 鬆綁 + SVD 初始化**：對「原層權重 − 共享權重」的殘差矩陣做 truncated SVD 來初始化 LoRA， [arxiv](https://arxiv.org/html/2410.20672v3) 使 rank→0 時退化為純遞迴、rank→full 時退化為 vanilla。鬆綁模型改以 Average 初始化最佳（因平均後的殘差易被低秩逼近）。
- **Continuous Depth-wise Batching + early exiting**：利用「不同深度共享同一組權重」多開一個批次維度。 [arxiv](https://arxiv.org/html/2410.20672v3)

## 關鍵數字
皆 SlimPajama uptraining、7 項 few-shot 任務平均：LAMBADA/HellaSwag/PIQA/WinoGrande/ARC-e/ARC-c/OpenBookQA。
- **Recursive vs 從頭訓練**：Recursive Transformer 以 Stepwise 初始化在 15B token uptraining 下，recursive Gemma 1B 達 few-shot 平均 51.7%，比從頭訓練的同尺寸 Gemma 1B（38.2%）高 13.5 個百分點（原文：「up to 13.5 absolute accuracy improvement (22% error reduction) on few-shot tasks compared to a non-recursive Gemma 1B model (pretrained from scratch)」），且勝過完整預訓練的 Pythia 1B（48.8%）。
- **rank 消融**（Gemma, Average+SVD, 15B）：rank 0/64/128/256/512 → 非嵌入參數 0.99/1.07/1.15/1.30/1.60B，few-shot Avg 46.7/52.1/52.5/54.2/55.5，PPL 15.15/12.83/12.52/12.10/11.83。高 rank 單調變好但遞減；Stepwise 初始化下則呈 U 形（中 rank 反而低於純遞迴）。LoRA 參數開銷：rank 64 +0.08B、128 +0.16B、256 +0.31B、512 +0.61B。
- **SVD vs Zero LoRA 初始化**：rank 512 下 SVD 55.5 vs Zero 46.7（最多提升約 6.5 點）。
- **uptraining vs from-scratch**：pretrained-init（Stepwise 51.7）vs random-init 遞迴從頭訓（37.6），差約 14.1 個百分點；Pythia 410M 對照（Table F.3）顯示多數增益來自利用預訓練權重（pretrained-init 非遞迴 Pile PPL 11.48 vs from-scratch 14.11），遞迴再加成（pretrained+recursive PPL 11.03 / Avg 41.54）。這與 [樂透票假說](lottery-ticket-redundancy-scaffold.html) 的推論相關。
- **蒸餾 headline**：relaxed Gemma（rank 512）+ 60B token + 知識蒸餾達 few-shot 58.4%，與 Gemma 2B 於 60B token uptraining 的 58.6% 打平（原文：「uptrained on 60 billion tokens, achieves performance on par with the full-size Gemma model」）。注意：原始 Gemma 2B 未 uptraining 為 61.7%，因 SlimPajama 分布偏移而下降，故論文以 58.6% 為對照。
- **吞吐量 2–3×（甚至近 4×）為理論值**：基於 oracle-exiting 模擬（假設中間預測與最終相符即可退出），非端到端實測；論文明言實用信心早退演算法「超出本文範圍」。
- **component-wise 消融（Table G.1）**：僅比較 Q/KV/Out/FFN 模組別，發現砍 FFN rank 傷害最大、砍 KV 最小；未做深度別實驗（見 [深度非均勻 rank](depth-nonuniform-lora-rank.html)）。

## LoRA 鬆綁的代價
rank 從 0 → 512，Gemma 非嵌入參數由 0.99B 增至 1.60B（+0.61B，達 vanilla 的 80%），few-shot 從 46.7% 升至 55.5%。rank 越高越好但呈現遞減報酬，且會破壞「單一權重區塊常駐」的 serving 優勢——每個 loop 需切換不同 LoRA，serving 上需管理 multi-LoRA（論文將此列為未來工作）——與 continuous depth-wise batching 的效率。rank 敏感度高且與初始化耦合（Stepwise+SVD 在中 rank 出現 U 形退化）。

## 實用建議（資源效率目標）
1. 首選「轉換式」路線而非從頭訓：以 Stepwise 初始化把現成 2× 大小模型轉成遞迴模型，先做 15B token uptraining 驗證，few-shot 掉分可控（RRT 顯示可打平）。
2. 若準確率不足，加入層級 LoRA（SVD 初始化、Average 底層初始化），rank 從 256 起試；**基準線**：rank 每翻倍帶來的 few-shot 增益 < 1 點時即停（RRT 中 256→512 僅 +1.3 點，已近遞減拐點）。
3. 若 serving 吞吐是瓶頸，優先評估 MoR（實測 2.06×，見 [後續工作](recursive-transformer-followups.html)）而非 RRT 的理論早退。

## 注意事項
- **理論 vs 實測**：RRT 的 2–3×/近 4× 吞吐為 oracle-exiting 模擬值，非端到端實測；Continuous Depth-wise Batching 與信心早退的實用演算法作者列為未來工作。引用時務必標明「理論/假設性」。
- **對照基準的微妙處**：RRT 的 58.4% vs 58.6% 是與「SlimPajama uptraining 後的 Gemma 2B」相比；原始 Gemma 2B 為 61.7%，因分布偏移下降。故「打平」是相對於 uptraining 後基線，非原始上界——這強化了「資源效率而非上界提升」的結論。
- **因果歸因**：RRT 的增益中「利用預訓練權重」與「遞迴結構本身」的貢獻，Table F.3 顯示前者為主、後者加成，但拆分僅在 Pythia 410M 小規模驗證。
