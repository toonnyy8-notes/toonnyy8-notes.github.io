# Round 1 合併審查意見（No. 5，arXiv 2609.20539）

來源：F = Fidelity review，P = Pedagogy/build review。已去重，依嚴重度排序。數值本身兩位審查者都未發現錯誤（verify 腳本重現驗收表）；以下皆為措辭、教學順序與建置規格。

| # | 級 | 來源 | 位置 | 問題 | 處理 |
| --- | --- | --- | --- | --- | --- |
| 1 | P0 | F1 | LO11 | 把 worst-case oracle 的「真實區」寫成 $\lvert m-k\rvert\le\ell$；Def 8.3 是單側：$m\le k-\ell$ 全回均勻，$m>k-\ell$ 的 consistent assignment 回真實 marginal、inconsistent 回均勻。$\lvert m-k\rvert\le\ell$ 只是 Prop 8.5 的「命中」事件，$m>k+\ell$ 由 $Q2^{-\ell}$ 項處理 | 改寫 LO11；補 Eq. 100 三項各自對應的事件 |
| 2 | P1 | F2 | LO12、Hero 卡 3、1.3 | 「masked 比 uniform 高 $\tilde\Theta(\sqrt d)$ 倍」過度：只有 masked $\tilde\Omega(d)$ 下界對 uniform $\tilde O(\sqrt d)$ 上界，能說「至少 $\tilde\Omega(\sqrt d)$ 倍」；兩邊 oracle 類別不同（masked：worst-case $\eta=d^{-\Theta(1)}$；uniform：誤差 $\le e^{-B\Lambda}$、$1/\varepsilon^2$）；Thm 8.1 是「以機率 $\ge1-\rho$（對 $k,C$）」的敘述 | 改寫 LO12、Hero 卡 3 |
| 3 | P1 | F3 | LO1、1.1、步驟 0 | masked 只給未揭露座標 $i\notin I$ 的 marginal；$\alpha_t/(1-\alpha_t)$ 依賴 $t$，只有 $q_{i\mid I}$ 與 $t$ 無關，且對估計 score 是 Sec. 2.3 的假設 | 改寫 LO1 與步驟 0 |
| 4 | P1 | P1 | S1 | 一章塞入 6 個符號、3 個 kernel、oracle、query complexity、上下界 | 拆成 S1a（三種腐蝕＋C1）、S1b（oracle＋query） |
| 5 | P1 | P2 | C1 | C1 讀 $I_{\rm unif},I_{\rm gauss}$，但 $I$ 到 S5 才定義、無對應 LO | C1 只留觸及比例與改變比例；$I$ 讀數移到 C4 小圖 |
| 6 | P1 | P3 | S2 | 一段複習 6 個前置概念，DTC 被擠壓 | 前置只留 $H$、KL、條件熵；TC、DTC 各附 $d=3$ 算式 |
| 7 | P1 | P4 | S2、符號表 | $h$、$s$、$t$ 的方向未在 LO4–LO5 之間定義 | S2 product reverse kernel 處寫「從 $t$ 退到 $s=t-h$」；符號表補 |
| 8 | P1 | P5 | S4 | Gaussian 段塞 Lemma 4.3、$c_u(v)$、$F_G$，無元件無 LO | 正文一句；細節進 `<details>` |
| 9 | P1 | P6 | 標題與用語 | 比喻用語「望遠鏡相加」「穿窗」「盲找窗」「藏窗」等違反字面風格 | 全文替換為 telescoping、窗內取樣、搜尋窗的位置、worst-case oracle、選窗位置 |
| 10 | P1 | P7 | C5 | $n_M=601$ 時 250ms/段掃描約 150 秒；查詢滑桿無 step | 滑桿改整數段號 $1..n$、`aria-valuetext`「第 k/n 段」；掃描固定 32 tick（≤8 秒），每 tick 查 $\lceil n/32\rceil$ 段 |
| 11 | P1 | P8 | C5 | 預設與範圍落在 $d\le256$ 時 $n_M\le n_U$、$Q_{\max}<1$，互動結果與主張相反 | $p\in\{10,\dots,20\}$、預設 16；小 $d$ 改 callout；$Q_{\max}<1$ 標「保證無意義」；驗收表延伸到 $2^{20}$ |
| 12 | P1 | P9 | C4 | 窗寬二分端點可能包不住 0.2/0.8 | 先檢查端點，包不住回傳 `NaN` 並顯示「窗超出範圍」；擬合略過 NaN、至少 3 點 |
| 13 | P1 | P10 | C2 | 絕對門檻 $10^{-12}$ 在 $10^{-12}$–$10^{-9}$ 區間比值為雜訊 | 相對門檻：$\lvert x\rvert<10^{-9}H_{\max}$ 視為 0；比值只在上界 $>10^{-9}$ 時顯示 |
| 14 | P1 | P11 | 390px 版面 | 表格、演算法格線、C2 兩圖、時間線無窄螢幕規格 | 表格包 `overflow-x:auto`；C2 <600px 上下堆疊；SVG `viewBox` 縮放＋最小高度 |
| 15 | P1 | P12 | KaTeX／SVG | SVG 標籤用 KaTeX 或 Unicode 未定 | SVG 內一律 Unicode＋`<tspan>`；KaTeX 只在 HTML 圖例，動態更新後 `renderMathInElement` |
| 16 | P2 | F4 | LO8、1.1 | 窗寬比例的條件未寫：uniform 需 $\delta=d^{-\Theta(1)}$、$\varepsilon$ 常數、$\log d\le\Lambda\le cd$；masked 需 $\eta=d^{-\Theta(1)}$ | 補條件 |
| 17 | P2 | F5 | LO4 | 把「實際誤差 = ΣTC + Σscore」與 Lemma 3.8 的上界（DTC 落差 + $\varepsilon_{\rm unif}$）混寫 | 分開兩句 |
| 18 | P2 | F6 | LO9、步驟 7 | 漲落的是 $\mathbb E[Z\mid d_H]$ 不是 $Z$；「轉變 ≈ $\mathbb E[Z\mid d_H]\approx1$」是啟發式；Prop 6.4 只給充分窗寬 | LO9 改「窗寬上界 $w\propto\sqrt{\Lambda/d}$，Fig. 4(d)／求積約 $1/\sqrt d$」；推導保留〔本頁推導〕並標啟發式 |
| 19 | P2 | F7 | S9、1.1、1.3、1.5 | 「下界論證沒有用到 commitment」是推論 | 改為「論文主張分離來自窗寬（abstract、Sec. 9）」 |
| 20 | P2 | F8a | C4 | masked 在 $I=\kappa$ 恆為 0.632 只在連續理想化（忽略 $\lceil\cdot\rceil$、$m$ 取實數）成立 | 註明 |
| 21 | P2 | F8b | C1 | 「只有一半真的改變」只對 binary；S 元為 $(S-1)/S$ | 改寫 |
| 22 | P2 | F8c | LO11 | Lemma 8.2 是 $\mathbb E_C$ 的敘述，good $C$ 由 Markov 得到（Eq. 98–99） | 補上 |
| 23 | P2 | F8d | 步驟 8 | Gaussian 查詢數寫成證明中的 $O((\sqrt{d\Lambda}+\Lambda)/\varepsilon^2+1)$，Thm 7.1 敘述為 $O(\sqrt{d\Lambda}/\varepsilon^2)$ | 兩者並列並註明 $\Lambda\le c_0d$ 時相同 |
| 24 | P2 | P13 | 無障礙 | C1 128 格各帶 `aria-label` 過重；C2、C3 長條只靠實心／外框 | C1 每列一句摘要、格子 `aria-hidden`、附可收合文字表；長條加文字標籤 |
| 25 | P2 | P14 | 動態 | 只有 C5 提 reduced motion | 全域 `prefers-reduced-motion` 關 transition；滑桿數值標籤 `input` 即時更新，重算只在 `change` |
| 26 | P2 | P15 | 暗色 | `--ink-2` 點線與 `--accent` 暗色對比未驗證 | Gaussian 點線改 `--ink-1`；建置時每個元件暗色截圖＋圖形對比 ≥3:1 自檢 |
