NO-GO

**Explain-back**

頁面講 Ψ-sampler。離散擴散的前向過程只規定邊際（式 (1)），先驗有兩種：MDM 用 [MASK]，USDM 用均勻分佈。式 (11) 把反向 posterior $q_{s|t}$ 與前向邊際 $q_s$ 以 $\kappa_t$ 疊加，對任意先驗都保持邊際（式 (16)）。把 $x$ 換成去噪器 $x_\theta$ 得式 (12)，邊際保證失效。在 mask 先驗且 $\kappa_t\in[0,1]$ 時，$\kappa_t=1-\sigma_t/(1-\alpha_s)$ 把它對應到 ReMDM。語言實驗：rescale $\eta=0.05$、$p=0.9$ 下 Gen. PPL 在 4096 步降到 19.21；影像 FID 依 $\kappa$ 與時間窗而定。課程學習以 order statistics 抽 top-k 值，normalizer 以截斷高斯期望近似。

元件：W1 熱圖與長條，可調 K、α_t、α_s、κ 與開關，讀邊際最大差與保留率（κ=0.5 時 0.505）。W2 單步三組長條，可調先驗、z_t、去噪器與 κ，讀 P(A)。W2b 讀 σ 到 κ 的直線。W3 toy 的錯誤率對步數曲線與軌跡，可調 schedule、T、ε_max。W4 top-k 抽樣與三個 normalizer 的比值，可調 K、k、α̃、τ。

**Blocking**

1. 符號先用後定義。§1.1 在符號表（§1.4）與 S1 之前使用：「把標準反向 posterior $q_{s|t}$ 與前向邊際 $q_s$ 以係數 $\kappa_t$ 做凸組合（式 (11)）」。§1.1 也寫「Gen. PPL 在 1024 步之後仍下降」與「nucleus」，但設計自己說指標在 S7 才定義（「指標（S7 先定義）」）。修法：把定義移到 §1.1 之前，或在 §1.1 第一次出現時附定義。

2. W4a 的驗收值無法重現。§5.4 的「單次抽樣（seed 3）：𝒦 = [4.0994, 3.5772, 3.5308]；w̃ = 0.3703；case = 1」沒有容許誤差。§5 的 ±0.02 只適用統計值。此值依賴抽樣順序：`verify/topk.py` 的 `topk_zero_mean` 先抽 k 個均勻數（ℓ = N, N−1, …, N−k+1），再由 `gauss(r)` 抽 w̃，但設計文字沒有寫出這個順序。`efficient()` 只有 Python 版，沒有 JS 移植（`chain.js` 只涵蓋 W3）；`prng.js` 也只驗證了均勻數，沒有驗證 gauss 與 `core.py` 一致。修法：在 §4 W4a 寫出抽樣順序並提供 JS 版 `efficient`／`logMu`／`normPpf`，再給這列明確容許誤差；或刪除這列。

**Non-blocking**

1. §4 W3 沒有寫出 c 與 window 的公式。`chain.js` 定義 c 為全部 L 個位置中 z = x* 的比例（[MASK] 算錯），window 為 t_off ≤ t ≤ t_on 時 κ = kap，否則 κ = 1。請抄進設計。
2. S10 說索引以 Floyd 演算法無放回抽出，但 `topk.py` 不抽索引，W4a 也只顯示值。請刪掉或補規格。
3. §7 R4 與 B1 說均勻先驗只開放 rescale 與 window，但 §4 W3 的 schedule 選單包含 ancestral，§5.3 也有「均勻 / ancestral」列。請釐清 ancestral 是否可選。
4. §1.4 步驟 5 使用 $q_{0|t}$，但符號表沒有這一列。請補上（定義為 $s=0$ 時的 $q_{s|t}$）。
5. LO7 參考答案寫「rescale 全程 κ_t ≥ 1−2η」，S5 寫的是「約在 [1−2η, 1−η]」。精確地說，分支條件迫使 α_t 約大於 0.5，下界只在該近似下成立。請與 S5 統一用語。
6. Table 4 的時間比：以前後半各半、吞吐 81.8／122.4 與 121.9／122.4 計算，得 0.803（省 19.7%），不是 0.804／19.6%。差異是捨入，但應寫明「前後半各半」的假設。

**Checks passed**

- 所有連結 slug 都存在：`notes_split/` 的 discrete-diffusion-foundations-d3pm、mdlm-masked-diffusion-objective、dlm-remasking-self-correction、uniform-noise-gidd-clean-supervision；`394/notes/` 的 7 篇 psi-* 筆記。diffusion-duality 依指示不計。
- 所有 notes 的 frontmatter 都有 slug、title、cluster、kind、tags、arxiv、sources、related。
- `timeline.json` 有 13 筆，§6.7 的 8 組 arXiv 編號全部涵蓋。
- LO1–LO12 每題都有參考答案，且各有指定的教學段落。
- 手算抽查通過：§5.1 的 q_{s|t}（0.9211、0.5000、0.3571）與保留率（0.625／0.505／0.385）；§5.3 的 rescale κ（0.9000、0.9496）、cap 最小值（t = 0.0547 時 −0.0579，即 −t/(1−t)）、cap η=0.01 的 −0.0159。
