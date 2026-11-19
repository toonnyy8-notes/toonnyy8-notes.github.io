---
slug: looped-transformers-reasoning
title: Looped transformers 的理論與推理優勢
cluster: Architectures & Efficiency
kind: topic
tags: [looped-transformer, reasoning, iso-flop, length-generalization]
sources: [weight-sharing-lora.md]
arxiv: ["2409.15647", "2502.17416", "2602.11451"]
related: ["weight-sharing-recursive-overview", "recursive-transformer-followups"]
---
# Looped transformers 的理論與推理優勢

屬於 [層間權重共享總覽](weight-sharing-recursive-overview.html)。

- **Looped Transformers as Programmable Computers**（Giannou et al., ICML 2023, PMLR 202:11398）：證明常數層（文獻歸納為 ≤13 層）的 looped transformer 可模擬 SUBLEQ、實作基本計算器、線性代數常式，達到圖靈完備。
- **Looped Transformers for Length Generalization**（Fan et al., arXiv:2409.15647, 2024）：decoder-only + input injection（每步注入原始輸入）， [arXiv](https://ar5iv.labs.arxiv.org/html/2409.15647) 在算術/copy 等任務展現超越訓練分布的長度外推。
- **Reasoning with Latent Thoughts: On the Power of Looped Transformers**（Saunshi, Dikkala, Li, Kumar, Reddi, arXiv:2502.17416, ICLR 2025, Google Research）：主張「許多推理問題需要大深度但不需大參數」。
  - 關鍵實驗設計與可信度——用 iso-param（同參數）與 iso-flop（同運算）雙基線做對照：k 層 looped L 次（k⊗L）在合成推理任務（加法、p-hop induction、i-GSM 數學）上接近 kL 層非 looped 模型。 [Liner](https://liner.com/review/reasoning-with-latent-thoughts-on-the-power-of-looped-transformers)
  - 在真實語言建模上，looped 模型 perplexity 較 iso-flop 差（只彌補 iso-param 與 iso-flop 間約 34–50% 的 perplexity gap），但在推理任務下游表現接近甚至偶爾超過 iso-flop 基線。
  - 論文明確結論：「Overall the looped models are worse on perplexity and closed book QA (memorization benchmarks), but the % Gap is much higher for task groups that require reasoning (open book QA, math word problems)」，並發現「accuracy on downstream tasks scale as logarithm of the effective depth」。
  - 具體例：數學字題上 (12⊗2) 達 34.3%，超越 24 層 iso-FLOP 基線的 29.3%（「despite having 50% fewer parameters and worse perplexity」）。
  - 此為「looped 在推理上有 inductive bias」的核心證據，設計嚴謹（同時控制參數與運算），可信度中高，但侷限在中小模型與特定任務組。

## 查證
Saunshi et al. 的雙基線設計嚴謹，結論可信但範圍有限；LoopFormer（arXiv:2602.11451）等後續一致確認「looped 在 perplexity 落後、在推理趨近 iso-FLOP」。 [arXiv](https://arxiv.org/html/2602.11451v1) Ouro 進一步用 controlled experiment 把「容量 vs 操作」拆開，是目前最強的機制性佐證（見 [後續工作](recursive-transformer-followups.html)）。
