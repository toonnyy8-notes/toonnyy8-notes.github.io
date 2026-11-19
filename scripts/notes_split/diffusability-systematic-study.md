---
slug: diffusability-systematic-study
title: Diffusability 系統性比較研究（Zhong et al., arXiv:2606.03578）
cluster: Diffusion & Generative Models
kind: topic
tags: [diffusability, tokenizer, gfid-prediction, benchmark-study]
sources: [latent-diffusion-rae.md]
arxiv: ["2606.03578", "2410.06940", "2501.01423", "2502.03444", "2502.09509", "2502.14831", "2603.05630"]
related: ["latent-diffusion-rae-overview", "optimization-dilemma-semantic-alignment", "diffusability-latent-regularizers", "velocity-irreducible-variance", "generative-eval-metrics"]
---
# Diffusability 系統性比較研究（Zhong et al., arXiv:2606.03578）

屬於 [Latent Diffusion 總覽](latent-diffusion-rae-overview.html)。被比較的各觀點見 [語意/空間對齊](optimization-dilemma-semantic-alignment.html)、[其他觀點](diffusability-latent-regularizers.html)、[VIV](velocity-irreducible-variance.html)。

## 實驗設計
三個 tokenizer 家族——43 個 conv-f16d32、22 個 conv-f16d64、21 個 trans-f16d32（合計 86 個）；每個 tokenizer 訓練 16 epochs；四個下游 backbone SiT-B（400k steps）、SiT-XL（80k）、LightningDiT-B/XL（100k）。控制同家族 reconstruction 品質可比。

## 結果
- **單一指標相關性排序**：VIV（Velocity Ambiguity）與 gFID 的 **Pearson=0.87**，為最穩定預測子；語意可分性（LNC）、空間結構（SRSS）緊隨其後「stand out」；流形連續性（iFID）、分布均勻性、仿射等變性相關性偏低，**仿射等變性最低且被剔除**（其兩個度量彼此不一致）。個別指標數值僅以散點圖呈現，原始數據於 GitHub（KlingAIResearch/diffusing-right-space）。
- **雙視角聯合迴歸**：**SRSS + LNC 可將 gFID 擬合到 R²=0.91**，說明在空間結構與語意可分性上位於 Pareto 前緣者生成品質較好。多數觀點近似正交，但頻譜平滑度、分布均勻性、VIV 三者呈現共線。
- **跨 backbone 泛化**：VIV 與空間結構最穩定；capacity 從 B→XL 時 SRSS 擬合更好、其他指標相關性下降或持平；LNC 在 SiT 上較好、SEC（Spectral Energy Concentration，本文提出的頻譜代理）在 LightningDiT 上較好。
- **跨家族泛化**：VIV、語意可分性、空間結構皆有效；iFID 在 conv-f16d64 家族相關性特別高（媲美 SRSS）但整體不穩定。
- **更好的 latent → 更直的 ODE 軌跡**：定義 Straightness（相鄰軌跡段 cosine 相似度均值）與 Efficiency（端點位移/累積路徑長），兩者皆與 gFID 高度相關（僅圖示、未給係數）；更好的 latent 讓 length ratio 更接近 1（更均衡的傳輸排程），且 irreducible variance 大的區域對應學到的步長較小。

## 綜合分析：正交還是共線？
- **可跨家族/backbone 泛化的結論**：VIV、語意可分性、空間結構是穩定預測子——這是目前最接近共識的部分。
- **各自 post-hoc 解釋、泛化性弱者**：仿射等變性（被裁決剔除）、分布均勻性、單獨的流形連續性（不穩定）。
- **正交 vs 共線**：多數觀點近似正交（故聯合迴歸能把 R² 提升到 0.91），但頻譜平滑度/分布均勻性/VIV 三者共線——暗示它們可能都是「velocity ambiguity/intra-class 結構」的不同投影。
- **未解問題**：VIV 的 Gaussian/GMM 假設在 RAE 這種極度各向異性、非高斯 latent 上是否成立；指標間的因果 vs 相關；T2I 場景（非 class-conditional）下這些指標是否仍預測有效（DiffusionBench 的負相關發現是警訊，見 [生成評估指標](generative-eval-metrics.html)）。

## 表：Diffusability 觀點與正則化方法比較

| 方法/指標 | 年份/arXiv | 核心想法 | 正則化/度量手段 | 關鍵實驗結果 | 限制 |
|---|---|---|---|---|---|
| REPA | 2024.10 / 2410.06940 | 語意可分性（對齊 diffusion 中間層） | noisy hidden ↔ DINOv2 clean feat. | SiT-XL >17.5× 加速、FID 1.42 | 需選 target encoder；不改 latent 本身 |
| VA-VAE / LightningDiT | 2025.01 / 2501.01423 | 語意先驗烘焙進 tokenizer + 均勻性 | VF Loss（mcos+mdms）+ adaptive weight | FID 1.35、64ep 達 2.11（>21×） | VF Loss 過強會壓縮容量 |
| MAETok | 2025.02 / 2502.03444 | 少 GMM modes、判別性 latent | mask autoencoding（非 variational） | 512² gFID 1.69、76× 訓練、128 tokens | AE 無 variational 保證 |
| EQ-VAE | 2025.02 / 2502.09509 | 仿射等變性 | scale/rotation 等變 fine-tune | DiT-XL ×7、REPA ×4 | Zhong et al. 判定相關性最低、被剔除 |
| Improving Diffusability | 2025.02 / 2502.14831 | 頻譜平滑（抑高頻） | decoder scale equivariance | FID −19%、FVD ≥−44%、≤20K steps | 大 channel bottleneck 才明顯 |
| iFID | 2026.03 / 2603.05630 | 流形連續性 | 最近鄰插值後解碼算 FID | 跨 VAE Pearson/Spearman ≈0.85 | 不可直接最小化 |
| VIV（Zhong et al.） | 2026.06 / 2606.03578 | 速度歧義（irreducible variance） | GMM + Kronecker flip-flop 解析式 | 與 gFID Pearson 0.87、最穩定 | Gaussian/可分離假設限制 |

## 建議（diffusability 診斷或指標選擇）
優先量測 **VIV、LNC（語意可分性 proxy）、SRSS（空間結構）** 三者；用 SRSS+LNC 聯合迴歸預測 gFID（R²≈0.91）。**不要**倚賴仿射等變性或單獨的分布均勻性做決策。

## 注意事項
- **「86」**：論文正文未寫單一「86」字樣，該總數為三家族 43+22+21 之和；家族規模不均。個別指標（除 VIV 的 Pearson 0.87 與聯合 R²=0.91 外）的相關係數僅以散點圖呈現，精確數值需查其 GitHub 原始數據。
- **指標 = 相關而非因果**：diffusability 各指標與 gFID 的高相關不等於因果；多數「解釋」是各方法的 post-hoc 敘事，Zhong et al. 的貢獻正是揭露此點。
