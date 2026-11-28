---
slug: revflow-flow-matching-dit-backbone
title: Flow matching 與 DiT／SiT：RevFlow 的訓練目標與骨幹
cluster: Diffusion & Generative Models
kind: topic
tags: [flow-matching, rectified-flow, stochastic-interpolant, dit, sit]
arxiv: ["2611.34782", "2006.11239", "2011.13456", "2210.02747", "2209.03003", "2209.15571", "2212.09748", "2401.08740"]
sources: [arxiv-abstracts, paper-2611.34782-related-work]
related: ["flow-matching-velocity-target", "revflow-lineage-overview"]
---
# Flow matching 與 DiT／SiT

論文 Related Work 把 DDPM、score SDE、flow matching、rectified flow、stochastic interpolant、DiT 並列為「擴散與流模型」的基礎。

| 年份 | 工作 | arXiv | 內容（依摘要） |
| --- | --- | --- | --- |
| 2020 | Denoising Diffusion Probabilistic Models（Ho 等） | 2006.11239 | 以逐步去噪訓練生成模型。 |
| 2020 | Score-Based Generative Modeling through SDEs（Song 等） | 2011.13456 | 以 SDE 統一分數式生成。 |
| 2022 | Flow Matching for Generative Modeling（Lipman 等） | 2210.02747 | 以回歸向量場訓練連續正規化流。 |
| 2022 | Flow Straight and Fast（Liu 等） | 2209.03003 | rectified flow，學習資料與雜訊間的傳輸。 |
| 2022 | Building Normalizing Flows with Stochastic Interpolants（Albergo、Vanden-Eijnden） | 2209.15571 | 以 interpolant 建立流模型。 |
| 2022 | Scalable Diffusion Models with Transformers（Peebles、Xie） | 2212.09748 | DiT：以 Transformer 作為擴散骨幹。 |
| 2024 | SiT（Ma 等） | 2401.08740 | 在 DiT 骨幹上探索 flow 與擴散的 interpolant 設定。 |

RevFlow 使用的形式（論文 Sec. 3.2）：目標與雜訊線性內插 `z_t = (1 - t)z + tε`，網路預測乾淨目標 `ẑ = f_θ(z_t, t, c)`（clean-target prediction），而非速度。論文報告在跨階段共用權重時，速度預測會退化，乾淨目標預測可用約一半參數達到與各階段獨立模型相當的結果。

預測目標的差異（速度、乾淨目標）另見 [Flow matching 的速度目標](flow-matching-velocity-target.html)。
