---
slug: engram-sparsity-allocation-u-shape
title: 稀疏度配置與 U 形縮放律
cluster: Architectures & Efficiency
kind: topic
tags: [sparsity-allocation, scaling-law, moe, engram]
sources: ["arXiv 2601.07372 全文（§1、§7 Related Work）"]
arxiv: ["2601.07372"]
related: ["engram-lineage-overview", "moe-conditional-computation-lineage", "engram-module-design"]
---
# 稀疏度配置與 U 形縮放律

屬於 [Engram 條件式記憶的研究脈絡總覽](engram-lineage-overview.html)。論文 §3 把「在固定總參數下，容量如何分給 [MoE 專家](moe-conditional-computation-lineage.html) 與 [Engram 記憶](engram-module-design.html)」形式化為 Sparsity Allocation 問題。

- **定義**：配置比例 ρ∈[0,1] 是非啟用參數預算中分給 MoE 專家的比例；P(MoE)=ρ·P(sparse)，P(Engram)=(1−ρ)·P(sparse)。ρ=1 為純 MoE。
- **實驗設定**：維持總參數與啟用參數比 P(tot)/P(act)≈10。C=2×10^20 FLOPs 時 P(tot)≈5.7B、P(act)=568M；C=6×10^20 FLOPs 時 P(tot)≈9.9B、P(act)=993M。
- **結果**：驗證損失對 ρ 呈 U 形。ρ 降到約 40% 時仍與純 MoE 相當；最佳點在 ρ≈75%–80%，即把約 20%–25% 的稀疏參數預算移給 Engram。10B 設定下驗證損失由 1.7248（ρ=100%）降至 1.7109，差 0.0139。
- **後續**：論文依此配置擴大到 27B，與嚴格同參數、同 FLOPs 的 MoE-27B 比較，報告 MMLU +3.0（Table 1；摘要寫 +3.4）、BBH +5.0、HumanEval +3.0、MATH +2.4 等差距，以及 Multi-Query NIAH 由 84.2 升至 97.0。
