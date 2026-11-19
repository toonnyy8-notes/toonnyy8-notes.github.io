---
slug: early-cross-layer-weight-sharing
title: 早期跨層權重共享：Universal Transformer、ALBERT、DEQ
cluster: Architectures & Efficiency
kind: topic
tags: [weight-sharing, universal-transformer, albert, deq]
sources: [weight-sharing-lora.md]
arxiv: ["1807.03819", "2310.07096", "2104.06022", "1909.11942", "2405.16039"]
related: ["weight-sharing-recursive-overview", "depth-redundancy-adjacent-methods"]
---
# 早期跨層權重共享：Universal Transformer、ALBERT、DEQ（2018–2021）

屬於 [層間權重共享總覽](weight-sharing-recursive-overview.html)。

- **Universal Transformer**（Dehghani et al., arXiv:1807.03819, ICLR 2019）：所有層共享同一組參數並沿深度遞迴更新， [arXiv](https://ar5iv.labs.arxiv.org/html/2310.07096) 配合 Adaptive Computation Time（ACT, Graves 2016）做 per-token 動態停止。原文報告 UT「achieve a 0.9 BLEU improvement over Transformers on the WMT14 En-De dataset」（並較 Weighted Transformer 高 0.5 BLEU），在 subject-verb agreement 上達約 99.2% 平均。但要達到相同表現需更寬的層、訓練時間約為 vanilla 的兩倍（Takase & Kiyono, arXiv:2104.06022 指出 UT 在 WMT En-De 需約兩倍訓練時間）。 [arxiv](https://arxiv.org/pdf/2104.06022)
- **ALBERT**（Lan et al., arXiv:1909.11942, ICLR 2020）：跨層全共享 + 分解式嵌入，ALBERT-large 相對 BERT-large 參數少 18×、訓練快約 1.7×。關鍵限制：ALBERT 是 encoder-only、以 MLM 目標訓練，共享在此設定下同時提供了正則化與梯度穩定。
- **Deep Equilibrium Models（DEQ）**（Bai et al. 2019）：把無限深度共享層視為求解不動點；ALBERT 論文觀察到其嵌入是振盪而非收斂到平衡點。 [arxiv](https://arxiv.org/pdf/1909.11942)

## 為何早期 encoder 可行、現代 decoder-only LLM 效果有限
（i）現代 LLM 是 parameter-dominated 任務（語言建模的 perplexity 強烈依賴參數量），共享直接砍掉記憶容量；（ii）decoder-only 自回歸 + KV cache 使得「不同深度異質參數」的早退與批次更難整合；（iii）MoEUT（Csordás et al., NeurIPS 2024, arXiv:2405.16039）明言：在 MoEUT 之前「沒有任何工作成功做出在語言建模上有競爭力的共享層 Transformer」，因為要補回參數就得把層做得極寬，運算/記憶體成本高到不切實際。 [arXiv](https://arxiv.org/pdf/2405.16039) MoEUT 本身見 [相鄰路線](depth-redundancy-adjacent-methods.html)。
