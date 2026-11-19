---
slug: score-geometry-diffusion-guarantees
title: 擴散模型的收斂保證：分數誤差如何決定分佈誤差
cluster: Diffusion & Generative Models
kind: topic
tags: [convergence, score-error, diffusion-theory, asymptotics]
arxiv: ["2209.11215", "2209.12381", "2208.05314", "2409.18804", "2509.24912"]
sources: [2509.24912 section 1.1, 2209.11215 abstract, 2209.12381 abstract, 2208.05314 abstract, 2409.18804 abstract, user_terms_note.md]
related: ["score-geometry-lineage-hub"]
---
# 擴散模型的收斂保證

回到 [總覽](score-geometry-lineage-hub.html)。

## 既有結果

本論文 1.1 節的描述：既有理論顯示，學到的分數夠準時，擴散／分數式採樣器收斂到目標分佈，誤差上界隨分數誤差直接增加。

- Chen et al. 2023（2209.11215）：最小資料假設下的擴散模型理論，標題為「sampling is as easy as learning the score」。
- Lee, Lu, Tan 2023（2209.12381）：一般資料分佈下的分數式生成模型收斂。
- De Bortoli 2022（2208.05314）：流形假設下的去噪擴散模型收斂。
- Azangulov et al. 2024（2409.18804）：高維流形假設下的收斂；與 Tang & Yang 2024 並列為研究維度依賴的工作（Tang & Yang 的 arXiv 編號未查證）。

本論文指出這些結果把幾何與密度放在同一個分數誤差裡，不區分兩者，所以不能得出尺度分離。

## 本論文的誤差量

分數誤差 E_σ = ‖s − s*‖_{L∞(K)}，K 是包含 T_M(ε) 的緊集。所有定理都用 E_σ 的階數表達。作者在限制中指出：DSM 訓練控制的是 L² 誤差，理論假設的是 L∞；L² 的對應結果列為未來工作（Fisher 散度 E‖s − ∇ log p‖²）。

## 漸近符號（σ → 0）

- a = O(b)：a/b 有界；a = o(b)：a/b → 0；a = Θ(b)：同階；a = Ω(b)：a/b 有下界。
- σ → 0 時 σ^(−2) → ∞，所以 E_σ = o(σ^(−2)) 比 o(1) 寬鬆很多。例：誤差 σ^(−1) 滿足 o(σ^(−2)) 但不滿足 o(1)。
- 本論文結論：幾何資訊在 Θ(σ^(−2)) 階，密度資訊在 Θ(1) 階。誤差 o(σ^(−2)) 足以讓分佈集中在流形（定理 4.1），還原 p_data 則需 o(1)。

## 假設 4.1

K 為緊集且「可求長路徑連通」：任兩點可用長度有一致上界的路徑連接。此條件讓「梯度誤差小」推出「函數誤差小」。
