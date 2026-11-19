---
slug: score-error-exposure-bias-relations
title: 分數誤差、曝光偏差與三種縮放：No. 237、405、406 的關係
cluster: Diffusion & Generative Models
kind: hub
tags: [exposure-bias, score-learning, manifold-hypothesis, epsilon-scaling, input-perturbation, tempered-langevin]
arxiv: ["2509.24912", "2301.11706", "2308.15321"]
sources: ["2509.24912 Sec. 1, Thm 4.1, Eq. 8, Sec. 7.2, App. C.2, Sec. 8 Limitations", "2301.11706 Eq. 6–8, Sec. 5.2–5.4, Tab. 1, Tab. 3, Tab. 7", "2308.15321 Eq. 7, Eq. 9, Sec. 3.4, Sec. 4.1, Table 3, Table 14", "syn-237-405-406/verify/check_out.txt"]
related: ["score-geometry-lineage-hub", "input-perturbation-lineage-overview", "exposure-bias-elucidation-overview", "score-geometry-tempered-langevin", "input-perturbation-ddpm-ip", "epsilon-scaling-method", "diffusion-exposure-bias"]
---
# 分數誤差、曝光偏差與三種縮放

本篇只整理三篇論文之間的關係。各篇的內容與研究脈絡由各自的總覽負責：[When Scores Learn Geometry 的研究脈絡總覽](score-geometry-lineage-hub.html)（No. 237）、[Input Perturbation（DDPM-IP）的研究脈絡總覽](input-perturbation-lineage-overview.html)（No. 405）、[擴散模型曝光偏差分析與 Epsilon Scaling：研究脈絡總覽](exposure-bias-elucidation-overview.html)（No. 406）。互動版見 [綜合教學](score-error-exposure-bias-synthesis.html)。

標記：「論文」＝該篇原文寫出的內容；〔本頁推導〕＝綜合教學自行推導；〔本頁詮釋〕＝解讀，不是推導。

## 論文自己寫出的關係

只有 No. 406 引用 No. 405：Sec. 3.4 把 DDPM-IP 列為以資料增強降低曝光偏差的方法，Table 3 把「ADM-IP (Ning et al., 2023)」100 步的 FID 2.38 列為 best FID，但 2.38 不在 No. 405 的任何表中（No. 405 Tab. 3 CIFAR10 ADM-IP 在 1,000／300／100／80 步為 2.76／2.67／2.70／2.93），γ 與權重來源未寫。No. 237 沒有提到曝光偏差；No. 405、406 沒有提到流形。No. 237 在 Limitations (a) 寫明不追蹤取樣軌跡上的誤差累積，只假設知道最終生成分佈的誤差。

## 共同的問題：小 σ 的誤差〔本頁詮釋〕

三篇都把重點放在噪音小的末段：No. 237 稱低溫分數最有價值也最難估計；No. 405 用「取樣軌跡後半段」說明 γ 的搜尋範圍 [0, 0.2]（取自實測 $\nu_t$ 的平均，No. 405 本身沒有分析小 σ）；No. 406 指出 t 接近 0 時 ε 預測最差，多步變異數誤差也在那裡最大。三者量誤差的位置不同：No. 237 量最終生成分佈的分數（$L^\infty$，σ → 0），No. 405、406 量取樣過程中 $\hat x_t$ 偏離 $x_t$ 的程度（No. 405 Tab. 1 是 ImageNet 32×32 上的 FID；No. 406 用 $\delta_t$、ε 範數比與 CIFAR-10 等的 FID）。曝光偏差可以看成產生 No. 237 所說「最終分佈誤差」的一種機制，但 No. 237 的定理沒有模型化這個機制。

〔本頁推導〕VP 下分數誤差 = ε 誤差 / σ_t = √ᾱ_t·($x_0$ 誤差) / σ_t²。No. 237 Thm 4.1 的兩個充分條件換成 $x_0$ 誤差：o(1) 足以集中到流形，o(σ²) 足以還原 $p_{data}$。這只是充分條件（前提為 Assumptions 2.1、2.2、4.1 與 $L^\infty(K)$ 範數），且 No. 237 的誤差是生成分佈的分數誤差，等於網路誤差只在特定設定成立。

## 兩種取樣時的縮放：TS 與 ES

No. 237 的 Tempered Score 把 corrector（Langevin）步中的無條件分數乘上 σ^α（0 < α < 2；SD 實驗中 CFG 項 w = 7.5 不變）；No. 406 的 Epsilon Scaling 把 predictor 平均中的 ε 除以 λ（ADM CIFAR-10 為 1.017）。分數 = −ε/σ，所以兩者都是分數乘數，但作用的步驟不同。〔本頁推導〕α = 1、λ = 1.017 時兩個乘數在 σ = 0.9833 相等，σ = 0.1 時 TS 為 0.1、ES 為 0.9833；常數 ES 寫成 σ^α 時的等效 α 隨 σ → 0 趨近 0，所以常數 ES 不是 TS 的特例（No. 406 另有線性 λ(t) = kt + b 的 ADM-ES*）。

〔本頁推導〕在 No. 237 的設定（Assumptions 2.1、2.2、4.1，精確分數、梯度場、連續時間 Langevin 跑到平穩）下，常數乘數 c 的平穩分佈 ∝ p_σ^c，σ → 0 時在流形上 ∝ $p_{vol}^c$（c < 1 時較平）；c = σ^α 時回到 No. 237 的均勻分佈。ES 實際是在有限步反向過程中只縮放平均裡的 ε 項，不是這個設定。單位圓、κ = 4 的例子中 c = 1/1.017 使 TV 偏離 $p_{data}$ 0.0046。〔本頁玩具〕高斯資料、完美網路的反向擴散中，λ = 1.017 在 100 步使輸出變異數增加 3.9%；但在 20 步（No. 406 改善最大處，CIFAR-10 FID 10.36 → 5.15）λ > 1 與 λ < 1 都使輸出大幅變寬，玩具失效。λ = 1.017 是 CIFAR-10 的值，其他資料集為 1.004–1.011，No. 406 稱最佳 b 隨步數增加而下降。No. 406 的理由是 $\hat x_t$ 上的 ε 過長、需要縮小，與精確分數下的方向相反；ES 的效益是實驗結果，本篇的推導不解釋它。

## 訓練端的修正：IP

〔本頁推導〕DDPM-IP 的貝氏最佳網路經標準取樣器換算後，等於噪聲標準差放大 √(1+γ²) 倍（γ = 0.1 時 1.005 倍）的資料分佈的精確分數；近流形時法向分量乘上 1/(1+γ²) = 0.9901、切向不變；Langevin 極限仍是 $p_{data}$。這個等式解釋不了 IP 的改善：DDPM-y（同一輸入、目標改為 ε′）的最佳網路只差 √(1+γ²) 倍，No. 405 Tab. 7 中 DDPM-y（CIFAR10 100 步 3.90）卻比 DDPM（3.37）差、IP 為 2.70；No. 405 把改善歸因於平滑預測函數的正則化。

〔本頁詮釋〕No. 405 的 $\nu_t$ 與 No. 406 的 $e_t$ 定義相同（$\hat x_0-x_0$ 的標準差），來源不同：前者沿取樣軌跡實測，後者是 No. 406 Eq. 7 的建模參數、未報數值。〔本頁推導〕在 No. 406 的模型下，使 IP 訓練輸入與單步取樣分佈一致的 γ 是 $c_{1,t+1}e_{t+1}/\sigma_t$，不是 No. 405 所說的 γ_t = ν_t，且隨 t 變化數個數量級。以推定值（常數 e = 0.2、cosine 排程、100 步）計算，常數 γ = 0.1 只在 t = 16..39 與模型一致；t 很小時的大 γ 是「e 不隨 t 變」的假設造成的。

## 流形觀點的邊界〔本頁詮釋〕

能說明：小 σ 的法向誤差被 σ⁻² 放大，三種修正的 σ → 0 極限不同（皆在精確分數的 Langevin 設定下：IP 的最佳網路：$p_{data}$；常數乘數 1/λ：$p_{vol}^{1/\lambda}$；TS：均勻）。不能說明：有限步與離散化、軌跡上的誤差累積、$L^2$ 訓練誤差、FID 的改善，以及 λ = 1.017、γ = 0.1 這些值。三篇的評估也不能互比（No. 237 用 CLIP 分數於 Stable Diffusion 1.5，No. 405、406 用 FID）。

## 延伸

- 各方法本身：[Tempered Score Langevin](score-geometry-tempered-langevin.html)、[DDPM-IP](input-perturbation-ddpm-ip.html)、[Epsilon Scaling](epsilon-scaling-method.html)。
- 曝光偏差的一般定義：[擴散模型的 exposure bias](diffusion-exposure-bias.html)。
