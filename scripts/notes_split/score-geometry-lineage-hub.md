---
slug: score-geometry-lineage-hub
title: When Scores Learn Geometry 的研究脈絡總覽
cluster: Diffusion & Generative Models
kind: hub
tags: [manifold-hypothesis, score-learning, rate-separation, diffusion-theory]
arxiv: ["2509.24912", "1907.05600", "2006.11239", "2011.13456", "2209.11215", "2208.05314", "2206.01018", "2212.12611", "2301.07882", "2506.15385", "2103.04715", "2507.15397"]
sources: [2509.24912 introduction, 2509.24912 section 1.1, user_terms_note.md]
related: ["score-geometry-score-matching-sde", "eps-prediction-ddpm-score", "score-geometry-diffusion-guarantees", "score-geometry-manifold-detection", "x-prediction-manifold-view", "score-geometry-laplace-expansion", "score-geometry-tempered-langevin", "score-geometry-sde-pde-wkb", "score-geometry-inverse-problems-guidance", "score-geometry-eval-metrics"]
---
# When Scores Learn Geometry 的研究脈絡

論文：Li, Shen, Hsieh, He，*When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis*（arXiv 2509.24912，v2 2026-03-16）。主張：在資料落在低維流形的假設下，小噪音 σ → 0 時分數中的幾何資訊比分佈資訊強 Θ(σ^(−2)) 倍，所以擴散模型的成功可解釋為學到流形；還原流形上的均勻分佈只需 o(σ^(−2)) 的分數誤差，還原 p_data 需要 o(1)。年代順序見 timeline.json。

## 閱讀順序

1. [分數學習與擴散模型的基礎](score-geometry-score-matching-sde.html)：分數匹配、Langevin、VE／VP-SDE、PC 採樣器。DDPM 細節見 [ε-prediction 與 DDPM、score-based 模型](eps-prediction-ddpm-score.html)。
2. [擴散模型的收斂保證](score-geometry-diffusion-guarantees.html)：既有結果把幾何與密度合在一個誤差裡；漸近符號與 L∞ 誤差。
3. [流形假設與擴散模型偵測流形](score-geometry-manifold-detection.html)：Pidstrigach、Stanczuk、Lu–Wang–Bal 等先行研究，加微分幾何名詞。相關：[x-prediction 與流形觀點](x-prediction-manifold-view.html)。
4. [小噪音下的 log p_σ 展開](score-geometry-laplace-expansion.html)：定理 3.1 與 Laplace 方法。
5. [Tempered Score Langevin](score-geometry-tempered-langevin.html)：定理 5.1 與均勻採樣。
6. [Fokker–Planck 與 WKB 展開](score-geometry-sde-pde-wkb.html)：定理 5.2 的證明工具。
7. [貝氏逆問題與 guidance](score-geometry-inverse-problems-guidance.html)：定理 6.1、PnP、CFG。
8. [評估指標](score-geometry-eval-metrics.html)：P-sim、I-sim。

## 全文符號

d：外部空間維度；M、n：流形與維度（n < d）；σ：高斯噪音標準差；μ_data、p_data(u)：資料分佈與其在座標 u 上的密度；μ_σ、p_σ(x)：加噪後分佈與 R^d 上密度；s*(x, σ) = ∇ log p_σ：真實分數；s(x, σ)：模型分數；Φ、u、g(u)：座標圖、局部座標、黎曼度量；P_M(x)、d_M(x)：最近點與 ½dist²(x, M)；T_M(ε)：管狀鄰域；α、β：TS Langevin 縮放指數與分數誤差階數 o(σ^β)。

## 學習路線（取自 user_terms_note.md）

前三步足以讀懂正文第 1–7 節，第 4–6 步為附錄證明所需。括號內為自我檢查。

1. 先修：多變數微積分、線性代數、基礎機率（能寫多維高斯密度、算 Hessian、做含 Jacobian 行列式的變數變換積分）。
2. 分數式生成模型：先讀 Yang Song 的部落格與 Lilian Weng 的綜述，再讀 Song & Ermon 2019、Ho et al. 2020、Song et al. 2021（能寫 DSM 損失、說明 VE 與 VP 差別、寫 PC 採樣器假碼）。
3. 漸近分析與 Laplace 方法：Bender & Orszag 第 6 章（Laplace 方法）、第 10 章（WKB）（能對單位圓上的 p_data 推出 log p_σ 的主導項 −(‖x‖ − 1)²/(2σ²)）。
4. 曲面與流形：do Carmo 建立直覺，再讀 Lee 的 Smooth Manifolds 與 Riemannian Manifolds（能算橢圓的度量與弧長元素，並說明對角度均勻不等於對弧長均勻）。
5. SDE、Fokker–Planck 與混合時間：Pavliotis 第 3–4 章；進階 Bakry–Gentil–Ledoux（能證明梯度漂移的平穩分佈是 exp(−f/θ)，並說明非梯度漂移為何沒有通式）。
6. 橢圓型 PDE 與極大值原理：Evans 第 6.4 節；Gilbarg–Trudinger 第 3 章（能說明緊緻無邊界流形上 Δ_M c + b·∇c = 0 只有常數解）。
7. 相關理論與應用：Pidstrigach 2022、Stanczuk et al.、De Bortoli 2022、Loaiza-Ganem et al. 綜述，再讀 Laumont et al. 2022 與 De Santi et al. 2025（能說出本論文與這些工作各差在哪裡）。

動手練習：重做附錄 C.3。在單位圓取 von Mises 分佈（κ = 4、θ₀ = π），數值積分算出真實分數，加一個 O(1) 的誤差場，比較標準 Langevin 與 α = 1 的 TS Langevin 在角度上的直方圖；再換成橢圓，確認 TS 結果對弧長均勻、對角度不均勻。

初學建議：第一次讀論文先看「分數式生成模型」與「漸近符號」；讀附錄證明時再查微分幾何與 SDE／PDE。

## 教科書（章節編號依常見版本，可能不同）

- Bender & Orszag, *Advanced Mathematical Methods for Scientists and Engineers*：Laplace 方法、WKB（第 6、10 章）。
- Holmes, *Introduction to Perturbation Methods*：匹配漸近展開（論文引用）。
- do Carmo, *Differential Geometry of Curves and Surfaces*：第一／第二基本形式（第 2–3 章）。
- Lee, *Introduction to Smooth Manifolds*；*Introduction to Riemannian Manifolds*：座標圖、度量、體積、管狀鄰域。
- Pavliotis, *Stochastic Processes and Applications*：SDE、Fokker–Planck、可逆性、Poincaré 不等式（第 3–4 章）。
- Bakry, Gentil, Ledoux, *Analysis and Geometry of Markov Diffusion Operators*：函數不等式（進階）。
- Evans, *Partial Differential Equations*：橢圓型 PDE、極大值原理（第 6.4 節）。
- Gilbarg & Trudinger, *Elliptic Partial Differential Equations of Second Order*：強極大值原理（第 3 章，Thm 3.5）。

## 線上資源

- Yang Song 部落格：https://yang-song.net/blog/2021/score/
- Lilian Weng, What are Diffusion Models?：https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- Hyvärinen 2005：https://jmlr.org/papers/v6/hyvarinen05a.html
- Stable Diffusion v1.5：https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5

未附連結的文獻與教科書資訊來自論文參考文獻與一般常識。
