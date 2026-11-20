# 設計文件：〈The Diffusion Duality〉互動教學

- 論文：Subham Sekhar Sahoo、Justin Deschenaux、Aaron Gokaslan、Guanghan Wang、Justin Chiu、Volodymyr Kuleshov，arXiv 2506.10892（v3，ICML 2025，PMLR 267）。方法名 **Duo**。資料庫編號 No. 393。
- 原文：`tut/393/src/paper.txt`（PDF 同目錄）。本文件中的「(n)」指論文式號，「§n」「A.n」「B.n」「C.n」「D.n」指章節與附錄，「Fig. n」「Table n」「Alg. n」照原文編號。顧問筆記：`tut/393/advisor.md`。
- 產出：`tut/393/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/diffusion-duality.html`（slug `diffusion-duality`），所有站內相對連結以上線位置為準。
- 驗證腳本（非頁面程式）：`tut/393/verify/dmath.py`（與頁面相同的數值演算法）、`check_T.py`、`check_widgets.py`、`check_seq.py`、`check_soft_f.py`、`check_sign.py`、`check_bayes_ode.py`。第 5 節所有數字由這些腳本產生。
- 研究筆記：`tut/393/notes/`（總覽 `duo-lineage-hub` + 6 篇主題筆記 + `timeline.json`，15 筆）。
- 標記慣例：〔本頁推導〕＝本設計自行推導或數值驗證、論文沒有寫出；**推定**＝為了對上論文數字而校準、論文沒有寫明的值。頁面用相同標籤（6.5）。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

離散擴散模型分兩類：遮罩式（MDM，token 一旦揭露就不再改）與均勻狀態（USDM，每個 token 可以被替換成詞表內任一 token，反向取樣時能反覆修改，論文稱為自我修正）。USDM 的困惑度一直落後 AR 與 MDM。本文的觀察是：在 one-hot 向量 $x$ 上做高斯擴散 $w_t\sim\mathcal N(\tilde\alpha_t x,(1-\tilde\alpha_t^2)I_K)$，再取 $z_t=\arg\max(w_t)$，$z_t$ 的條件邊際恰好是 USDM 的邊際 $\mathrm{Cat}(T(\tilde\alpha_t)x+(1-T(\tilde\alpha_t))\mathbf 1/K)$（(10)），其中 $T$ 是一維積分定義的 Diffusion Transformation 運算子（(11)）；這個邊際也滿足 USDM 的線性 ODE（(12)）。作者稱之為 Diffusion Duality（(13)）。這個對偶**只在每個時刻的邊際上成立**，把高斯軌跡逐點取 arg max 並不是 USDM 的 Markov 鏈（A.3，(40)）。作者用它做兩件事：(1) 把離散 NELBO 改寫成對高斯潛變數取期望（(19)），再把網路輸入的 arg max 換成溫度 softmax，形成課程學習（(21)），論文報告訓練速度約 2 倍；(2) 用最優 denoiser 下的高斯 PF-ODE 直線軌跡取 arg max，得到確定性離散軌跡 $P_{DDT}$（(22)），據此做 Discrete Consistency Distillation（DCD，Alg. 1），取樣步數由 1024 降到 16（ancestral）或 8（Greedy-Tail）。另外提出與 UDLM 損失 (49) 代數等價、但不需建構 $\bar x$ 的 $f_{Duo}$（(17)）。LM1B 困惑度 33.7、OWT 25.2（Table 1），在 7 個 zero-shot 資料集中 3 個勝過 AR（Table 2）。

### 1.2 研究動機（§1）

1. 高斯擴散有成熟的技巧（參數化、快速訓練、PF-ODE 取樣器、一步蒸餾），離散擴散仍以 mean parameterization 與慢速 ancestral 取樣為主。
2. MDM 的蒸餾（SDTT）在少步時品質大幅下降；MDM 沒有「implicit」性質（雜訊到資料的確定性映射），因為它的先驗是確定的（全部是 mask），取樣必須帶隨機性。
3. USDM 能在反向過程更新 token，不需額外 predictor-corrector 步驟，但表現落後。作者問：USDM 能否與 MDM 競爭？能否借用底層高斯擴散的 implicit 性質做少步生成？

### 1.3 貢獻（§1 末段）

1. 連續與離散擴散的理論連結：離散（均勻狀態）擴散由底層的高斯擴散經 arg max 產生（§3）。
2. 低變異的課程學習使 USDM 訓練速度加倍（§4.1、§5.1）。
3. DCD 把取樣加速兩個數量級（§4.2、§5.2）。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S1 末尾以可收合表格提供）

| 符號 | 意義 |
| --- | --- |
| $K$ | 詞彙大小（類別數）。LM1B 用 bert-base-uncased（$K=30522$），OWT 用 GPT-2（$K=50257$）〔詞彙數為 tokenizer 的公開值，論文未寫出，頁面標「tokenizer 詞彙數」〕 |
| $\mathcal V$ | 所有 one-hot 向量的集合；$x\in\mathcal V$ 是乾淨 token，$x_m=1$ 的索引記為 $m$（A.1 用 $k$） |
| $\mathrm{Cat}(\cdot;\pi)$、$\mathbf 1$ | 類別分布、全 1 向量；$\pi=\mathbf 1/K$ 為均勻先驗 |
| $t\in[0,1]$ | 擴散時間；$t=0$ 接近資料、$t=1$ 為先驗（與 k-Diff 頁相反） |
| $z_t\in\mathcal V$、$\alpha_t$ | 離散潛變數與離散擴散參數，$\alpha_t$ 嚴格遞減，$\alpha_0\approx1$、$\alpha_1\approx0$ |
| $w_t\in\mathbb R^K$、$\tilde\alpha_t$、$\tilde\sigma_t$ | 高斯潛變數、高斯擴散參數、$\tilde\sigma_t=\sqrt{1-\tilde\alpha_t^2}$ |
| $\phi,\Phi$ | 標準常態 pdf $e^{-z^2/2}/\sqrt{2\pi}$ 與 cdf（正文 (11) 下方印成 $e^{-z^2}$，見 1.4 步驟 3 注意） |
| $T(\cdot)$ | Diffusion Transformation 運算子 (11)，把 $\tilde\alpha_t$ 換成 $\alpha_t$ |
| $[\arg\max]_\star$ | 推前（pushforward）：隨機變數經 arg max 映射後的分布 |
| $x_\theta$ | denoiser，輸出 $\Delta^K$（機率單體）上的向量 |
| $\bar x,\bar x_\theta,\kappa_t$ | $\bar x=K\alpha_tx+(1-\alpha_t)\mathbf 1$，$\bar x_\theta$ 同式以 $x_\theta$ 代入，$\kappa_t=(1-\alpha_t)/(K\alpha_t+1-\alpha_t)$ |
| $\tau$、$[\beta,\gamma]$ | softmax 溫度、課程期間 $t$ 的取樣區間 |
| $\delta$、$N$、$M$ | DCD 的時間步長、輪數、每輪步數 |
| $L$、$\ell$ | 序列長度、位置索引 |

符號重複（頁面在用到處各提醒一次）：$\beta,\gamma$ 也是 A.1 的機率與 (44)(45) 的量；$T$ 也指取樣步數（Table 5–7）；Alg. 2 的 $K$ 是輪數。頁面中取樣步數一律寫「步數 $T_{\text{steps}}$」或「T = 16 步」以免混淆。

以下依論文推導順序。每步寫「要回答什麼／假設／得到什麼／下一步為何需要」。

#### 步驟 1：離散擴散的插值式寫法（§2.1，(1)–(6)）

- 前向邊際（Sahoo et al. 2024a 的插值框架）：$q_t(\cdot\mid x;\alpha_t)=\mathrm{Cat}(\cdot;\alpha_tx+(1-\alpha_t)\pi)$ (1)。MDM 取 $\pi=m$（mask token），USDM 取 $\pi=\mathbf 1/K$（Schiff et al. 2025）。
- 邊際隨時間的線性 ODE：$\frac{d}{dt}q_t=Q_tq_t$ (2)。USDM 的 $Q_t$ 在 (3) 印成 $+\frac{\alpha_t'}{K\alpha_t}[\mathbf 1\mathbf 1^\top-KI_K]$。〔本頁推導〕直接對 (1) 微分得 $\frac{d}{dt}q_t=\alpha_t'(x-\mathbf 1/K)=-\frac{\alpha_t'}{K\alpha_t}[\mathbf 1\mathbf 1^\top-KI]q_t$，與 (12)/(36) 一致；非對角速率 $-\alpha_t'/(K\alpha_t)>0$。頁面採 (12)/(36) 的負號，並在 S2 以一行註明 (3) 的正負號與 (12) 不同（見 1.5-2）。
- 真實後驗 (4)：$q_{s|t}(\cdot\mid z_t,x)=\mathrm{Cat}\Big(\cdot;\dfrac{K\alpha_tz_t\odot x+(\alpha_{t|s}-\alpha_t)z_t+(\alpha_s-\alpha_t)x+(1-\alpha_{t|s})(1-\alpha_s)\mathbf 1/K}{K\alpha_t\langle z_t,x\rangle+1-\alpha_t}\Big)$，$\alpha_{t|s}=\alpha_t/\alpha_s$。〔本頁推導〕即 $q(z_t|z_s)q(z_s|x)$ 以貝氏定理正規化（`check_bayes_ode.py` 2000 組隨機數相等到 1e-12）。
- 推論時以 $x_\theta(z_t,t)$ 取代 $x$：$p^\theta_{s|t}=q_{s|t}(\cdot\mid z_t,x=x_\theta)$ (5)。取樣從 $z_1\sim\mathbf 1/K$ 出發，逐步抽 $z_s\sim p^\theta_{s|t}$（ancestral sampling）。
- 目標：NELBO (6) $=\mathbb E_{t\sim U[0,1],q_t}f(z_t,x_\theta,\alpha_t;x)$，$f$ 見 (49)（出自 Schiff et al. 2025）。
- 為什麼需要：之後所有結果都以「(1) 的邊際 + (2) 的 ODE」作為 USDM 的判準。

#### 步驟 2：直接作用在 one-hot 上的高斯擴散（§2.2，(7)(8)）

- $\tilde q_t(\cdot\mid x;\tilde\alpha_t)=\mathcal N(\tilde\alpha_tx,(1-\tilde\alpha_t^2)I_K)$ (7)，$\tilde\alpha_t$ 單調遞減。注意是 one-hot 向量本身加雜訊，不是 embedding（§6 與 Diffusion-LM、Plaid 的差別）。
- (8) 是 Kingma et al. 2021 的 NELBO，之後沒有再用到；頁面只提一句。
- PF-ODE 與 consistency distillation（§2.3，(9)）：教師 $x_{\theta^-}$ 用 PF-ODE 從 $w_t$ 走一步到 $w_s$，學生在 $w_t$ 的輸出逼近教師在 $w_s$ 的輸出。頁面在 S9 才講。

#### 步驟 3：arg max 下的離散邊際（§3.1，A.1，(10)(11)(24)–(32)）

- 要回答：$z_t=\arg\max(w_t)$ 的分布是什麼？
- 假設：$x_m=1$；$w=\tilde\alpha x+\tilde\sigma\epsilon$，$\epsilon\sim\mathcal N(0,I_K)$。
- 推導：$w_m\sim\mathcal N(\tilde\alpha,\tilde\sigma^2)$、$w_{i\ne m}\sim\mathcal N(0,\tilde\sigma^2)$ 且獨立 (24)(25)；$Z=\max_{i\ne m}w_i$ 的 cdf 為 $\Phi^{K-1}(l/\tilde\sigma)$ (27)；
  $$P(\arg\max=m)=\int\Phi^{K-1}(l/\tilde\sigma)\tfrac1{\tilde\sigma}\phi\big(\tfrac{l-\tilde\alpha}{\tilde\sigma}\big)dl=\int_{-\infty}^{\infty}\phi\Big(z-\tfrac{\tilde\alpha}{\sqrt{1-\tilde\alpha^2}}\Big)\Phi^{K-1}(z)\,dz\tag{28}$$
  其餘 $K-1$ 個索引對稱，各為 $\beta_t=(1-P_m)/(K-1)$ (29)(30)，向量形式 $\mathrm{Cat}(\beta_t\mathbf 1+(1-K\beta_t)x)$ (32)。令 $\alpha_t=1-K\beta_t$：
  $$z_t\sim\mathrm{Cat}\big(T(\tilde\alpha_t)x+(1-T(\tilde\alpha_t))\mathbf 1/K\big),\quad T(\tilde\alpha)=\frac{K}{K-1}\Big[\int\phi\Big(z-\tfrac{\tilde\alpha}{\sqrt{1-\tilde\alpha^2}}\Big)\Phi^{K-1}(z)dz-\frac1K\Big]\tag{10, 11, 34}$$
- 性質〔本頁推導〕：$T(0)=0$（積分為 $1/K$）、$\tilde\alpha\to1$ 時 $T\to1$；$T$ 只依賴訊雜比 $\tilde\alpha/\tilde\sigma$ 與 $K$；$K=2$ 時 $T=2\Phi(\tilde\alpha/(\sqrt2\tilde\sigma))-1=\mathrm{erf}(\tilde\alpha/(2\tilde\sigma))$（$\tilde\alpha=0.6$：兩式皆 0.404117）。
- 注意：(11) 下方與 A.1 Part 3 把 $\phi$ 印成 $\exp(-z^2)/\sqrt{2\pi}$，不是機率密度；A.1 Part 2 與 A.3 寫成 $\exp(-z^2/2)/\sqrt{2\pi}$。所有數值以後者為準（K=2 封閉式、蒙地卡羅都只與後者相符）。
- 下一步為何需要：邊際的形式對了，還要確認它隨時間的演化是 USDM 的演化。

#### 步驟 4：邊際的 ODE 與對偶陳述（§3.1，A.2，(12)(13)(33)–(36)）

- 由 $p_t=\beta_t\mathbf 1+(1-K\beta_t)x$ 得 $\frac{d}{dt}p_t=\beta_t'(\mathbf 1-Kx)$；關鍵恆等式 $[\mathbf 1\mathbf 1^\top-KI]p_t=(1-K\beta_t)(\mathbf 1-Kx)$，所以 $\frac{d}{dt}p_t=\frac{\beta_t'}{1-K\beta_t}[\mathbf 1\mathbf 1^\top-KI]p_t$ (33)；代入 $\alpha_t=1-K\beta_t$：$\frac{d}{dt}p_t=-\frac{\alpha_t'}{K\alpha_t}[\mathbf 1\mathbf 1^\top-KI]p_t$ (36)，即
  $$\frac{d}{dt}P_t=-\frac{T'(\tilde\alpha_t)}{KT(\tilde\alpha_t)}[\mathbf 1\mathbf 1^\top-KI_K]P_t\tag{12}$$
  （$T'$ 為對 $t$ 的導數）。〔本頁推導〕`check_bayes_ode.py` 以 $K=5$、$\tilde\alpha_t=1-t$ 有限差分確認 (12) 兩邊相等到 6 位。
- 對偶 (13)：$q_t(z_t\mid x;T(\tilde\alpha_t))=[\arg\max]_\star\tilde q_t(w_t\mid x;\tilde\alpha_t)$。論文稱 (35)(36) 為 USDM 的「unique signatures」。〔本頁推導〕(36) 可由 (35) 直接微分得到，不是獨立條件；頁面照錄作者說法，加一句說明。
- 為什麼重要：高斯擴散的參數 $\tilde\alpha_t$ 經 $T$ 對應到離散參數 $\alpha_t$；凡是只依賴「每個時刻的邊際」的量（例如 NELBO 中對 $z_t$ 的期望），都可以改用高斯潛變數計算。

#### 步驟 5：邊際相同，軌跡不同（A.3，(37)–(40)；§3.1 末句、Fig. 1 圖說）

- 高斯轉移核 $w_t\mid w_s\sim\mathcal N(\tilde\alpha_{t|s}w_s,(1-\tilde\alpha_{t|s}^2)I)$，$\tilde\alpha_{t|s}=\tilde\alpha_t/\tilde\alpha_s$ (37)；離散轉移核 $z_t\mid z_s\sim\mathrm{Cat}(\alpha_{t|s}z_s+(1-\alpha_{t|s})\mathbf 1/K)$ (38)，所有「非 $z_s$」索引機率相等。
- 推前 (39)：$P(\arg\max=i)=\int\prod_{j\ne i}\Phi\big(\frac{l-\tilde\alpha_{t|s}(w_s)_j}{\sqrt{1-\tilde\alpha_{t|s}^2}}\big)\frac{1}{\sqrt{1-\tilde\alpha_{t|s}^2}}\phi\big(\frac{l-\tilde\alpha_{t|s}(w_s)_i}{\sqrt{1-\tilde\alpha_{t|s}^2}}\big)dl$。它依賴整個 $w_s$，不只 $\arg\max(w_s)$；非最大索引機率相等若且唯若對應的 $(w_s)_j$ 相等，機率為 0。所以 (40) $q_{t|s}(\cdot\mid z_s)\ne[\arg\max]_\star\tilde q_{t|s}(\cdot\mid w_s)$。
- 〔本頁推導〕即使非最大座標相等（機率 0 的情形），推前分布一般也不等於 (38)：$K=3$、$\tilde\alpha_s=0.9$、$\tilde\alpha_t=0.6$、$w_s=(1.2,0.3,0.3)$ 時推前為 (0.576, 0.212, 0.212)，(38) 為 (0.611, 0.195, 0.195)。論文的論證只需要「不等」，這個數字只用來說明差異不只在對稱性。
- 〔本頁推導〕對 $w_s\sim\tilde q_s(\cdot\mid x)$ 取平均後，推前分布回到時刻 $t$ 的邊際（1500 個樣本：(0.559, 0.222, 0.219) 對理論 (0.559, 0.220, 0.220)）。也就是：個別路徑不是 USDM，但每個時刻的分布是。
- 頁面用語（全頁一致）：「對偶在**邊際**層級成立」；不寫「離散擴散等於高斯擴散取 arg max」而不加限定。

#### 步驟 6：兩個定理（§3.2，A.4–A.5，Thm 3.1–3.2，(14)–(16)(41)–(47)）

- Thm 3.1：給定任一高斯反向過程 $\bar p^\theta$，存在離散反向核 (15) 使所有 $t$ 有 $p^\theta_t=[\arg\max]_\star\bar p^\theta_t$ (14)；A.4 另解出對應的離散 denoiser (44)。實驗沒有實作，§7.2 列為未來工作（guidance、ε／v 參數化）。
- Thm 3.2：$\mathbb E_{q_{data}}\log p^\theta_{data}\ge\mathbb E_{q_{data}}\log\bar p^\theta_{data}$ (16)。證明用 data-processing inequality (46)，以及 $q_{data}$ 支撐在 one-hot 上所以 $[\arg\max]_\star q_{data}=q_{data}$。作者據此選擇在離散潛變數上定義模型，並以離散 NELBO (6) 訓練與評估（§3.2 末段）。
- 頁面必須寫明的限定：(a) 比較的是 Thm 3.1 構造出的一對取樣器（共用同一個高斯空間 denoiser）的**邊際概似**，不是 ELBO（Fig. 1 圖說寫成 ELBO）；(b) 實際訓練的 Duo 直接用 (6)/(17) 訓練離散 denoiser，不是該構造；(c)〔本頁推導〕證明中把 one-hot 上的 pmf 與 $\mathbb R^K$ 上的密度放進同一個 $\sum_x q\log(q/\bar p)$，嚴格說兩者的對數不可直接比較，頁面稱之為「動機性的論證」。

#### 步驟 7：Duo 的訓練目標與取樣器（§3.3，A.6，(17)(18)(48)–(51)）

- Greedy-Tail：最後一步不抽樣，改取 $\arg\max p^\theta_{0|\delta}$，類似 nucleus sampling 降低熵。
- $f$ (49)：$f=-\frac{\alpha_t'}{K\alpha_t}\Big[\frac K{\bar x_i}-\frac K{(\bar x_\theta)_i}-\sum_j\frac{\bar x_j}{\bar x_i}\log\frac{(\bar x_\theta)_i\bar x_j}{(\bar x_\theta)_j\bar x_i}\Big]$，$i$ 為 $z_t$ 的索引。需要建構 $\bar x$。
- $f_{Duo}$ (17)：
  $f_{Duo}=-\frac{\alpha_t'}{K\alpha_t}\Big[\frac K{\bar x_i}-\frac K{(\bar x_\theta)_i}-(\kappa_t\mathbb 1_{z_t=x}+\mathbb 1_{z_t\ne x})\sum_j\log\frac{(\bar x_\theta)_i}{(\bar x_\theta)_j}-K\frac{\alpha_t}{1-\alpha_t}\log\frac{(\bar x_\theta)_i}{(\bar x_\theta)_m}\mathbb 1_{z_t\ne x}-\big((K-1)\kappa_t\mathbb 1_{z_t=x}-\tfrac1{\kappa_t}\mathbb 1_{z_t\ne x}\big)\log\kappa_t\Big]$。
  - 〔本頁推導〕$\bar x$ 只取兩個值（$\bar x_m=K\alpha_t+1-\alpha_t$，其餘 $1-\alpha_t$），比值 $\bar x_j/\bar x_i\in\{1,\kappa_t,1/\kappa_t\}$，把 (49) 的和分成兩種情形（$z_t=x$、$z_t\ne x$）即得 (17)。這是對固定 $(z_t,x)$ 的**代數恆等式**：$K=7$、$\alpha=0.37$、$\alpha'=-1$、隨機 $x_\theta$，三種 $(z_t,x)$ 的兩式差 ≤ 4.4e-16（`check_soft_f.py`）。
  - 論文的說法：(17) 省記憶體、不必建 one-hot $\bar x$，並「applies Rao-Blackwellization… reducing variance」（A.6）；論文自己報告兩式評估得到同一 PPL 33.7（A.6 末段）。推導中沒有任何期望被解析積掉。頁面寫：「論文稱之為 Rao-Blackwellized；從推導看，它是 (49) 的逐點等價改寫。Table 3 中去掉它 PPL 由 35.0 變 36.7（約 1.7），論文沒有說明這個差距的機制。」
  - 〔本頁推導〕正負號：若 $\alpha_t'$ 取 $d\alpha_t/dt$（$<0$），(49) 如印出來恆 $\le0$（`check_sign.py`：20000 組隨機情形最大值 $-7\times10^{-8}$），而連續時間 KL 速率應 $\ge0$；若 $\alpha_t'$ 理解為 $-d\alpha_t/dt$，(3)、(49)、(17) 三式就都合理，但 (12)(36) 又多一個負號。頁面在 S7 以一行說明「(3)(17)(49) 與 (12)(36) 的 $\alpha_t'$ 正負號慣例不一致，不影響 (17) 與 (49) 的等價」。
- 序列：前向與反向都對 token 分解，序列 NELBO 是 token 損失的和 (18)。

#### 步驟 8：以高斯潛變數改寫 NELBO，再做課程學習（§4.1，B.1，(19)–(21)，C.2，C.7）

- (19)：由步驟 3 的 (10)，對 $z_t\sim q_t$ 的期望可換成對 $w_t\sim\tilde q_t$ 的期望，$z_t:=\arg\max(w_t)$、$\alpha_t:=T(\tilde\alpha_t)$。數值相等；作者以 1000 個 antithetic 取樣的 $t$ 驗證兩種算法 PPL 皆 33.7（B.1）。這一步只用到每個時刻的邊際（步驟 5 的限定在這裡不造成問題）。
- 鬆弛：$\arg\max(w)=\lim_{\tau\to0^+}\mathrm{softmax}(w/\tau)$ (20)。訓練時 denoiser 的**輸入**改成 $\mathrm{softmax}(w_t^\ell/\tau)$，損失仍是離散 $f_{Duo}$（$z_t:=\arg\max(w_t)$）：
  $$\mathcal L_{train}=\mathbb E_{x,t\sim U[\beta,\gamma],\tilde q_t}\sum_\ell f_{Duo}\big(z^\ell_t:=\arg\max(w^\ell_t),\,x^\ell_\theta([\mathrm{softmax}(w^{\ell'}_t/\tau)]_{\ell'=1}^L,t),\,\alpha_t:=T(\tilde\alpha_t);x^\ell\big)\tag{21}$$
  $\tau>0$ 時 (21) 不是合法的 NELBO；只有 $\tau\to0^+$ 且 $\beta=0,\gamma=1$ 時才是（§4.1.2）。評估一律用 (18)。
- denoiser 同時接受 one-hot 與單體上的向量：第一層 embedding 以 $y^\top E$（$E\in\mathbb R^{K\times m}$）計算，連續輸入等於 embedding 的凸組合（C.2）。
- 作者的理由（§4.1.2）：arg max 對微小擾動極敏感，大部分訊號在離散化時遺失；讓網路看到 softmax 保留更多訊號，$\tau$ 控制難度。$T$ 的性質：$K$ 大時，$\tilde\alpha\in[a,1]$ 的窄區間就足以讓 $T$ 涵蓋 $[0,1]$；C.7：$K=30K$、$\tilde\alpha_t\in[0.85,1]$ 時 $T$ 幾乎涵蓋 $[0,1]$。
- 訓練窗（C.7）：損失被 $\alpha_t'$ 加權，$\alpha_t'\approx0$ 的 $t$ 幾乎沒有訊號，所以把 $t$ 限制在使 $\alpha_t=T(\tilde\alpha_t)\in[0.05,0.95]$ 的區間；兩個 tokenizer 皆對應 $[\beta,\gamma]=[0.03,0.15]$，$\tilde\alpha_t=1-t$（§5 開頭）。
- **與式 (11) 不符**〔本頁推導，`check_T.py`〕：照 (11)，$K=30522$ 時 $T(0.85)=0.0077$、$T(0.97)=0.4523$；$T=0.05$ 與 $0.95$ 落在 $\tilde\alpha=0.9238$ 與 $0.9856$（$K=50257$：0.9299、0.9861）。這兩點的平方 0.8533、0.9715 接近論文的 0.85、0.97，與「$\tilde\alpha_t^2=1-t$」相符：此時 $t\in[0.03,0.15]$ 對應 $T\in[0.047,0.934]$（$K=30522$）。論文印出的是 $\tilde\alpha_t=1-t$。本頁無法判定哪一處是筆誤；W2 同時畫出兩種讀法，文字照錄論文並說明差異。§4.1.2 那句「$\tilde\alpha\in[0.85,1]$ 時 $T$ 幾乎涵蓋 $[0,1]$」在兩種讀法下都成立。
- 排程（§5.1）：前 500K 步 $\tau=0.001$、$\beta=0.03,\gamma=0.15$；之後到 1M 步 $\tau=0$（純離散）。後 500K 步的 $t$ 區間與 $\alpha_t$ 排程論文沒寫；評估 PPL 用 (18) 與 $\alpha_t=1-t$。
- $T$ 的計算：預先算好 100K 組 $(\tilde\alpha_t,T(\tilde\alpha_t))$ 快取（§5），積分方法未說明。
- 偏差–變異（§5.1、Fig. 8、D.1）：LM1B 150K 步、固定 $\tau\in\{0,0.001,0.01,0.1\}$；$\tau=0.1$ 損失急降（偏差過大），$\tau=0.001$ 接近 $\tau=0$ 曲線且變異明顯較低。Table 4（$\tau=0.001$）top-100 權重梯度變異：10K 步 0.30 對 11.7，100K 步 0.86 對 55.1，500K 步 1.15 對 1.92。〔本頁推導〕$\tau=0.001$ 相對於 $w$ 的座標尺度（$\tilde\sigma$ 約 0.17–0.53）極小，softmax 輸出幾乎總是 one-hot；論文沒有把「softmax 鬆弛」與「$t$ 限制在 $[\beta,\gamma]$」兩個因素分開消融。頁面以 W5 呈現前半句，後半句寫在 S8 文字。

#### 步驟 9：確定性離散軌跡與 DCD（§4.2，B.2，(22)(23)(57)–(60)，Alg. 1–2）

- 問題：consistency distillation 需要由 denoiser 參數化的 PF-ODE；訓練完的 Duo 只吃離散輸入（課程後 $\tau=0$），不能在高斯空間解 ODE。
- 做法：假設最優 denoiser $x_\theta=x$，DDIM 步 (57) 退化為 $w_s=\tilde\alpha_sx+\sqrt{1-\tilde\alpha_s^2}\epsilon$（(58)，原文左邊寫 $z_s$、右邊下標 $t$），整條 PF-ODE 軌跡是 $P_{ODE}=\{\tilde\alpha_tx+\sqrt{1-\tilde\alpha_t^2}\epsilon\}_t$ (59)；取 arg max 得
  $$P_{DDT}(x^{1:L},\epsilon^{1:L})=\big\{[\arg\max(\tilde\alpha_tx^\ell+\sqrt{1-\tilde\alpha_t^2}\epsilon^\ell)]_{\ell=1}^L\big\}_{t\in[0,1]}\tag{22}$$
- 〔本頁推導〕$(x,\epsilon)$ 固定時，非 $m$ 的座標都是 $\tilde\sigma_t\epsilon_i$，最大者恆為 $j^*=\arg\max_{i\ne m}\epsilon_i$，所以 $z_t\in\{x,e_{j^*}\}$；$z_t=x\iff\tilde\alpha_t/\tilde\sigma_t>\Delta:=\epsilon_{j^*}-\epsilon_m$。訊雜比隨 $t$ 單調，每個 token 至多翻轉一次，翻轉點 $\tilde\alpha=\Delta/\sqrt{1+\Delta^2}$；$\Delta\le0$ 時恆為 $x$。這就是 A.7 說的「只能取兩個值、類似 MDM 的 carry-over」。A.7 也強調：$P_{DDT}$ 只在蒸餾時用來產生教師與學生的輸入，Duo 本身不是以 carry-over 方式取樣。
- 蒸餾 (23)：在同一條 $P_{DDT}$ 上取相鄰兩點 $(z_s,z_t)$，$s=\max(t-\delta,0)$，最小化 $\sum_\ell D_{KL}(x^\ell_\theta(z_t,t),x^\ell_{\theta^-}(z_s,s))$（學生看較吵的 $z_t$）。Alg. 1：$N$ 輪，每輪 $M$ 步；每輪開始 $\theta^-\leftarrow$ stopgrad$(\theta)$（教師＝上一輪結束時的學生權重，不是 EMA）；每輪結束 $\delta\leftarrow2\delta$；回傳 $\theta_{ema}$。實驗 $N=5$、$M=10K$、初始 $\delta=1/512$（五輪的 $\delta$ 為 1/512、1/256、1/128、1/64、1/32〔本頁推導：連乘〕）。Alg. 2 改用 EMA 為教師，結果較差（Fig. 10、Table 6）。
- 頁面用語：DCD 沒有用教師解 ODE，也沒有 (9) 的 $\lambda(t)$；頁面寫「沿 $P_{DDT}$ 相鄰兩點的輸出分布做 KL 對齊，並逐輪加倍步長」，並註明結構上接近 SDTT（Deschenaux & Gulcehre 2024）。
- 未寫明：EMA 係數 $\mu$、Duo 蒸餾的 batch 與學習率（只說同 SDTT 設定；C.8 給的是 MDLM 的：batch 128、lr 6e-5、warmup 500 步）。

#### 步驟 10：實驗（§5，Table 1–7，Fig. 2–4、9–12）

- 設定（§5、C.5）：170M 參數 DiT（12 層、hidden 768、12 heads、時間嵌入 128、旋轉位置編碼、adaLN），AdamW、batch 512、lr 3e-4（warmup 2500 步）、dropout 0.1、1M 步、8×H100、bfloat16。LM1B 長度 128（bert-base-uncased，有／無 packing 兩種），OWT 長度 1024（GPT-2），保留最後 100K 篇為驗證集。取樣實驗一律 float64（§5.2，因 Zheng et al. 2024 指出低精度取樣的問題）。
- Table 1（PPL↓，LM1B 無 packing／LM1B／OWT）：AR 22.3／22.8／17.5；MDLM 27.0／31.8／23.2；SEDD Absorb 32.7／–／24.1；SEDD Uniform 40.3／–／29.7；UDLM 31.3／36.7／27.4；**Duo 29.9／33.7／25.2**。擴散模型為上界。與 MDLM 差距 2.9／1.9／2.0〔本頁推導：相減〕；論文寫「below 2 PPL points」，只有中間一欄嚴格小於 2，頁面照數字寫。
- 速度：510K 步（課程後 10K 步離散微調）PPL 35.2，比訓練 1M 步的 UDLM（36.7）好約 1.5，作者據此稱「至少 2×」。頁面註明比較對象是 UDLM；同表 Duo w/o CL 在 1M 步是 35.0。Plaid 在 LM1B 100K 步 PPL 89.9，Duo 同步數 43.0（未列入表，C.1）。
- Table 3：Duo 33.7；w/o CL 35.0；再 w/o (17)（即 UDLM）36.7。§7.1 建議無法調參時不用課程。
- Table 2（zero-shot，OWT 訓練 1M 步）：Duo 89.35／33.57／73.86／49.78／67.81／44.48／40.39（PTB／Wikitext／LM1B／Lambada／AG News／Pubmed／Arxiv）；AR 82.05／25.75／51.25／51.28／52.09／49.01／41.73。Duo 在 Lambada、Pubmed、Arxiv 勝過 AR（3/7）；對 SEDD Absorb 4/7、對 MDLM 1/7（PTB）。MDLM 在同樣三個資料集也勝過 AR。
- 取樣（Table 5，未蒸餾，Gen PPL／entropy）：Duo $T$=1024 為 77.69／5.55，$T$=8 為 198.27／5.57；MDLM 104.85／5.63 與 830.82／5.91。評估：GPT-2 Large 的 Gen PPL（↓）與平均序列熵（論文用語 entropy，太低代表多樣性下降）。
- 蒸餾（Fig. 4、Table 6–7，Round 5）：ancestral $T$=16 為 75.24／5.53（base 1024 步 77.69／5.55）→ 64×；Greedy-Tail $T$=8 為 69.58／5.30 → 128×，熵較低。與 MDLM+SDTT（Round 5，ancestral）：$T$=8 111.88 對 193.05、$T$=16 75.24 對 89.17、$T$=32 61.31 對 62.29、$T$=64 55.83 對 47.04、$T$=1024 50.55 對 36.89。即低步數 Duo 較好，$T\ge64$ MDLM 較好（§5.2、§7 都承認）。
- 熵隨蒸餾輪次：SDTT 蒸餾的 MDLM 下降，DCD 蒸餾的 Duo（Greedy-Tail）上升（Fig. 12）。

#### 步驟 11：實作細節與常見錯誤（頁面 S11 逐條列出）

1. $\phi$ 用 $e^{-z^2/2}/\sqrt{2\pi}$，不是 (11) 下方印的 $e^{-z^2}/\sqrt{2\pi}$。
2. $Q_t$ 的正負號以 (12)/(36) 為準；(3)(17)(49) 的 $\alpha'_t$ 慣例與之相反（步驟 1、7）。
3. 計算 $T$ 時避免下溢與相消：$\Phi^{K-1}$ 以 $\exp((K-1)\log\Phi)$ 計算；$T$ 的方括號直接算 $\int[\phi(z-\mu)-\phi(z)]\Phi^{K-1}(z)dz$（因 $\int\phi\Phi^{K-1}=1/K$），不要先算積分再減 $1/K$〔本頁推導〕。論文以 100K 組快取查表。
4. 訓練窗 $[\beta,\gamma]=[0.03,0.15]$ 與 $\tilde\alpha_t=1-t$ 照式 (11) 不給 $T\in[0.05,0.95]$（步驟 8）。實作時以「$T(\tilde\alpha_t)\in[0.05,0.95]$」這個目的為準，自行用 (11) 求區間。
5. (21) 只把 **denoiser 輸入** 換成 softmax；損失裡的 $z_t$ 仍是 arg max，$\alpha_t$ 用 $T(\tilde\alpha_t)$。$\tau>0$ 時不是 NELBO，評估用 (18)。
6. 後 500K 步 $\tau=0$；PPL 評估 $\alpha_t=1-t$，單一 $t$ 的低差異取樣（C.3、C.4）。
7. DCD 的教師是上一輪的學生權重（Alg. 1），用 EMA（Alg. 2）較差；每輪 $\delta$ 加倍；學生輸入是較吵的 $z_t$。
8. 不要把對偶用在轉移核：$\arg\max$ 後的高斯軌跡不是 USDM 的 Markov 鏈（步驟 5）。
9. 論文內部不一致（照錄並註明，不自行更正）：Table 3 圖說與 §5.1 把課程標成「Alg. 1」（Alg. 1 是 DCD）；Fig. 13–16 圖說的 Gen PPL／entropy 與 Table 6–7 不完全一致（Fig. 15–16 的 79.24／5.25、42.46／5.25 是 Table 6 的 Alg. 2 Greedy-Tail 數字）；C.8 寫蒸餾 MDLM 於 LM1B，§5.2 寫在 OWT；名稱 DCD／CDC（Fig. 12）混用；C.7 首句說畫的是 $\tilde\alpha_t$，Fig. 6 畫的是 $T(\tilde\alpha_t)$。

#### 步驟 12：限制與未來工作（§7）

課程有兩個超參數（$\tau$ 與課程長度），其他領域可能需要調整；USDM 的困惑度仍落後 MDM，但在少步生成中較好。未來工作：以 Thm 3.1 的 (15) 讓高斯潛變數累積分類器梯度來引導離散取樣器；以 (44) 把 ε／v 參數化帶進離散 denoiser。

### 1.5 頁面上必須避免的誤讀（Goldfish 審查逐條檢查）

1. 不寫「離散擴散就是高斯擴散取 arg max」而不加「在邊際上」。
2. 不把 Thm 3.2 寫成「Duo 的 ELBO 比高斯擴散緊」。
3. 不把 $f_{Duo}$ 的效果歸因於變異數下降（論文這樣說，但推導沒有對應步驟；頁面照錄論文說法並加註）。
4. 「2×」註明相對 UDLM；「兩個數量級」註明 Greedy-Tail 才到 128×，且熵下降。
5. 「蒸餾後勝過 MDLM」註明只在 $T\le32$。

---

## 2. 學習目標（驗收測試）

每章結尾以 `<details class="check">`（summary「檢查理解」）列出該章題目與參考答案（照搬）。參考答案的每個要素必須同時出現在該章**內文**段落中。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 區分 MDM 與 USDM 的前向過程與反向行為 | MDM 與 USDM 的先驗各是什麼？哪一種在反向取樣時能修改已產生的 token？ | MDM 先驗是 mask token $m$，揭露後不再改；USDM 先驗是均勻 $\mathbf 1/K$，token 可在反向過程中反覆更新（自我修正）（§2.1） | S1 |
| LO2 | 寫出 USDM 的邊際、後驗與取樣方式 | USDM 的前向邊際是什麼？取樣時如何從 $z_t$ 得到 $z_s$？ | $q_t=\mathrm{Cat}(\alpha_tx+(1-\alpha_t)\mathbf 1/K)$ (1)；以 $x_\theta(z_t,t)$ 代入真實後驗 (4) 得 $p^\theta_{s\mid t}$ (5)，從 $z_1\sim\mathbf 1/K$ 開始逐步抽樣（ancestral sampling） | S2 |
| LO3 | 推導 arg max 的邊際 | $w=\tilde\alpha x+\tilde\sigma\epsilon$ 時，正確類別成為 arg max 的機率怎麼寫？其餘類別呢？ | $\int\phi(z-\tilde\alpha/\tilde\sigma)\Phi^{K-1}(z)dz$ (28)；其餘 $K-1$ 類對稱，各為 $(1-P_m)/(K-1)$，合起來是 $\mathrm{Cat}(Tx+(1-T)\mathbf 1/K)$ (10) | S3 |
| LO4 | 說出 $T$ 的性質 | $T(0)$、$T(\tilde\alpha\to1)$ 是多少？$K=2$ 時的封閉式？$K$ 變大時曲線如何變化？ | $T(0)=0$、$T\to1$；$K=2$ 時 $T=\mathrm{erf}(\tilde\alpha/(2\tilde\sigma))$〔本頁推導〕；$K$ 越大，$T$ 越集中在 $\tilde\alpha$ 接近 1 處（$K=30522$：$T(0.85)=0.008$、$T(0.97)=0.45$） | S4 |
| LO5 | 說明對偶的完整陳述 | 除了邊際形式，作者還驗證了什麼才說 arg max 把高斯擴散變成 USDM？ | 邊際滿足 USDM 的線性 ODE (12)，擴散參數為 $T(\tilde\alpha_t)$；合起來寫成 (13) $q_t(\cdot\mid x;T(\tilde\alpha_t))=[\arg\max]_\star\tilde q_t(\cdot\mid x;\tilde\alpha_t)$ | S4 |
| LO6 | 區分邊際對偶與軌跡 | 把一條高斯軌跡逐點取 arg max，得到的是 USDM 的 Markov 鏈嗎？為什麼？ | 不是。推前的轉移機率 (39) 依賴整個 $w_s$，不只 $z_s=\arg\max(w_s)$；USDM 核 (38) 的非 $z_s$ 類機率相等，推前只在非最大座標相等時才可能如此（機率 0），所以 (40) 不等。對偶只在每個時刻的邊際成立（A.3） | S5 |
| LO7 | 正確陳述 Thm 3.2 | Thm 3.2 比較的是什麼？作者用它決定了什麼？ | 比較 Thm 3.1 構造的離散與高斯取樣器在真實資料下的邊際概似，離散的不低於高斯的 (16)（data-processing inequality）；作者因此在離散潛變數上定義模型，用離散 NELBO 訓練與評估。它比較的不是 ELBO（§3.2、A.5） | S6 |
| LO8 | 說明 $f_{Duo}$ 與 $f$ 的關係 | (17) 與 (49) 有什麼關係？論文怎麼確認？去掉 (17) 對 PPL 的影響？ | 對每個固定的 $(z_t,x)$ 兩式相等，(17) 是不需建構 $\bar x$ 的代數改寫（A.6）；以兩式評估同一模型都得 33.7；Table 3 去掉 (17)（成為 UDLM）PPL 由 35.0 變 36.7 | S7 |
| LO9 | 描述課程學習 | (21) 改了哪個地方？溫度與時間區間怎麼設？它是合法的 NELBO 嗎？ | 只把 denoiser 的輸入由 $\arg\max(w_t)$ 換成 $\mathrm{softmax}(w_t/\tau)$，損失仍是 $f_{Duo}$；前 500K 步 $\tau=0.001$、$t\sim U[0.03,0.15]$，之後 $\tau=0$；$\tau>0$ 時不是合法 NELBO，評估用 (18)（§4.1.2、§5.1） | S8 |
| LO10 | 解釋 DDT 與 DCD | $P_{DDT}$ 上一個 token 最多改變幾次？DCD 的教師、步長、輪數怎麼設？ | 至多一次，只在 $x$ 與 $\arg\max_{i\ne m}\epsilon_i$ 之間，翻轉點 $\tilde\alpha=\Delta/\sqrt{1+\Delta^2}$〔本頁推導〕；教師為上一輪的學生權重（非 EMA），學生在較吵的 $z_t$ 上以 KL 逼近教師在 $z_s$ 的輸出，$N=5$ 輪、每輪 10K 步、$\delta$ 由 1/512 起每輪加倍（Alg. 1、§5.2） | S9 |
| LO11 | 讀出主要實驗數字 | Duo 在 LM1B／OWT 的 PPL？蒸餾後幾步達到 base 1024 步的品質？與蒸餾 MDLM 比較如何？ | LM1B 33.7、OWT 25.2（Table 1）；ancestral 16 步 Gen PPL 75.24（base 1024 步 77.69），Greedy-Tail 8 步 69.58 但熵 5.30 較低；$T\le32$ 勝過 MDLM+SDTT，$T\ge64$ 落後（Table 7） | S10 |
| LO12 | 指出實作時的數值與一致性問題 | 照式 (11) 計算，$[\beta,\gamma]=[0.03,0.15]$ 與 $\tilde\alpha_t=1-t$ 能給出 $T\in[0.05,0.95]$ 嗎？$\phi$ 該用哪個式子？ | 不能：$K=30522$ 時 $T(0.85)=0.008$、$T(0.97)=0.45$；$T\in[0.05,0.95]$ 對應 $\tilde\alpha\in[0.924,0.986]$，平方約為 [0.85, 0.97]，與 $\tilde\alpha_t^2=1-t$ 相符〔本頁推導〕；$\phi=e^{-z^2/2}/\sqrt{2\pi}$ | S8、S11 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），互動元件放加寬圖版（≤1080px）。章節編號以 JetBrains Mono 顯示（`03 · arg max 的邊際`）。章節標題用陳述句。視覺記號：一條「訊雜比軸」（$\tilde\alpha$ 由 0 到 1）在 hero、W2、S8 重複出現，標出論文訓練窗。

| # | id | 標題 | 內容（依序；每個符號先定義再使用） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈擴散對偶：對高斯擴散取 arg max，每個時刻的分布就是均勻狀態離散擴散〉 | 副標「這篇論文證明兩種擴散在邊際上相連，並藉此把高斯擴散的課程學習與一致性蒸餾移植到離散文字模型。」三張事實卡（下方）；連結 arXiv、`../index.html#paper-393`、`duo-lineage-hub.html`；hero 圖：$K=3$ 的單一 token，左邊三根 $w$ 長條、右邊 one-hot，靜態 SVG | 1 螢幕 |
| S1 | `discrete` | 01 · 文字擴散有遮罩與均勻兩種雜訊 | one-hot、$\mathcal V$、$K$、Cat、$\mathbf 1$；前向過程把 token 逐步換成雜訊；MDM（mask，揭露後固定）vs USDM（均勻替換、可反覆更新＝自我修正）；Fig. 5 以四欄小表文字重述（AR／MDM／USDM／$P_{DDT}$，$P_{DDT}$ 欄註「S9 才解釋」）；USDM 落後的事實（Table 1 一句）；符號表（可收合）。延伸：`duo-discrete-diffusion-masked-and-uniform.html`、`duo-self-correction-and-correctors.html` | 450–550 字 |
| S2 | `usdm` | 02 · 均勻狀態擴散由邊際、ODE 與後驗定義 | $t$ 方向、$\alpha_t$；(1)；(2) 與速率矩陣（寫 (12)/(36) 的負號版本，一行註 (3) 印成正號）；(4) 與一句「這是貝氏定理」〔本頁推導〕；(5) 與 ancestral sampling；(6) 只寫存在（$f$ 到 S7 才展開） | 350–450 字 |
| S3 | `argmax` | 03 · 對 one-hot 加高斯雜訊再取 arg max，得到類別分布 | (7)、$\tilde\alpha,\tilde\sigma$、arg max 運算、推前；A.1 三步推導 (24)–(32)（主文寫結論，完整推導放 `<details>`）；$\phi$ 印刷錯誤一行；W1 | 450–550 字 + W1 |
| S4 | `T` | 04 · 運算子 T 把高斯參數換成離散參數，且邊際滿足 USDM 的 ODE | (11) 與性質（$T(0)=0$、$T\to1$、K=2 封閉式、只依賴訊雜比）；$K$ 大時壓縮到 $\tilde\alpha\approx1$；(33)→(36)→(12)；(13) 對偶陳述；一句「unique signatures」說明；W2 | 450–550 字 + W2 |
| S5 | `paths` | 05 · 對偶只在邊際成立，軌跡不是離散擴散 | (37)(38)(39)(40)；平均後回到邊際〔本頁推導〕；Fig. 1 圖說「separate Markov processes」；W3；callout「本頁用語：對偶在邊際層級成立」 | 400–500 字 + W3 |
| S6 | `theorems` | 06 · 兩個定理說明離散取樣器存在且概似不低於高斯 | Thm 3.1 (14)(15)、(44) 一句；Thm 3.2 (16)、data-processing (46)；三點限定（1.4 步驟 6）；作者結論：在離散潛變數上訓練與評估 | 300–400 字 |
| S7 | `loss` | 07 · Duo 的損失是 UDLM 損失的等價改寫 | $\bar x,\bar x_\theta,\kappa_t,m$；(49)；(17)；兩種情形的拆法（`<details>`）；本頁即時數值表（f 與 f_Duo 並列，6.3）；33.7 同值、Table 3 的 1.7；正負號慣例一行；(18)；Greedy-Tail | 400–500 字 + 小表 |
| S8 | `curriculum` | 08 · 課程學習先讓網路看 softmax，再換回 arg max | (19)（只用到邊際，所以成立）；(20)(21)；C.2 soft lookup；$[\beta,\gamma]$ 與 C.7 的理由；Fig. 6／(11) 不符的說明（回指 W2 的兩條帶）；排程；偏差–變異（Fig. 8、Table 4 三列）；W5；未分離兩個因素的說明。延伸：`duo-curriculum-softmax-relaxation.html` | 550–650 字 + W5 |
| S9 | `dcd` | 09 · 最優 denoiser 的直線軌跡取 arg max，成為蒸餾用的確定性離散軌跡 | consistency distillation 一段（(9)，PF-ODE、教師／學生）；為何 Duo 不能直接用；(57)→(59)→(22)；至多翻轉一次〔本頁推導〕；(23)；Alg. 1 逐行兩欄對照（同 k-Diff 頁格式）；Alg. 2；W4。延伸：`duo-consistency-distillation-to-discrete.html` | 550–650 字 + W4 |
| S10 | `results` | 10 · Duo 在均勻狀態模型中困惑度最低，蒸餾後在少步取樣勝過 MDLM | Table 1 摘錄（7 列×3 欄）；510K 步 35.2；Table 3；Table 2 摘錄（Duo、AR、MDLM、UDLM 四列）；Table 7 Round 5 摘錄（$T$=8,16,32,64,1024）；64×／128× 與熵；1.5 的限定 | 400–500 字 + 3 表 |
| S11 | `impl` | 11 · 實作設定與常見錯誤 | 訓練設定表（1.4 步驟 10 首條 + 課程排程 + DCD 參數）；9 條常見錯誤（步驟 11） | 350–450 字 + 表 |
| S12 | `history` | 12 · 研究脈絡 | 時間線（6.6）+ 100–150 字把脈絡串起來：擴散 → 高斯（DDPM、score SDE、VDM）→ 離散（D3PM、CTMC、SEDD、MDLM、UDLM）→ 鬆弛（Gumbel-Softmax、Concrete）→ 蒸餾（Progressive、Consistency、SDTT）→ Duo | 時間線 + 文字 |
| S13 | `next` | 13 · 限制與延伸閱讀 | §7.1、§7.2；研究筆記依總覽順序；站內既有 `dllm-three-corruption-paradigms.html`、`discrete-diffusion-foundations-d3pm.html`、`mdlm-masked-diffusion-objective.html`、`dlm-remasking-self-correction.html` | 200 字 + 清單 |

Hero 三張事實卡（不用未定義術語）：
1. 「把一個詞寫成長度 K 的 one-hot 向量，加上高斯雜訊再取最大座標，得到的詞服從『以某機率保留原詞、否則均勻換成任一詞』的分布。這正是均勻狀態離散擴散在每個時刻的分布。」
2. 「這個對應只在每個時刻的分布上成立；把一條連續的雜訊路徑逐點取最大座標，得到的不是離散擴散的路徑。」
3. 「論文用它把 USDM 的訓練加速約 2 倍（相對 UDLM），並把取樣步數由 1024 降到 16（品質相當）或 8（多樣性略降）。」

---

## 4. 互動元件規格

### 4.0 共用數學（W1–W5 共用，函式名見 6.3）

- `phi(z)=exp(-z²/2)/√(2π)`；`erfc(x)`：Numerical Recipes `erfcc`（相對誤差 < 1.2e-7；已與 `math.erfc` 比對，最大 1.04e-7）；`Phi(z)=0.5·erfc(−z/√2)`；`logPhi(z)`：$z>0$ 用 `log1p(−0.5·erfc(z/√2))`，否則 `log(0.5·erfc(−z/√2))`（避免 $\Phi\approx1$ 時 `log` 失去精度，$K=50257$ 需要）。
- **$T(\tilde\alpha,K)$**：$\mu=\tilde\alpha/\sqrt{1-\tilde\alpha^2}$；固定網格 $z_i=-10+0.005i$，$i=0..4400$（到 12）；`G_K[i]=exp((K−1)·logPhi(z_i))`，每個 $K$ 算一次並快取。
  $S=h\sum_i{}'[\phi(z_i-\mu)-\phi(z_i)]G_K[i]+[\Phi(12)-\Phi(12-\mu)]$（$\sum'$ 為梯形權重；尾項處理 $z>12$，該處 $G_K=1$ 的誤差 < $K\Phi(-12)\approx10^{-28}$）。$T=\mathrm{clamp}(\frac{K}{K-1}S,0,1)$。守門：$\tilde\alpha\le0\Rightarrow0$；$\tilde\alpha\ge0.999999\Rightarrow1$。直接計算 $\int[\phi(z-\mu)-\phi(z)]G$ 避免「積分 − 1/K」的相消〔本頁推導：$\int\phi\Phi^{K-1}=1/K$〕。
- 反解 `alphaForT(target,K)`：二分法 60 次於 $[0,0.999999]$。
- PRNG：`mulberry32(seed)`、`gauss(r)`（Box–Muller 只取 cos 分支），與 k-Diff 頁相同（直接重用其函式）。
- 所有讀數：機率 4 位小數，$T$ 4 位，$\tilde\alpha$ 4 位；JetBrains Mono。

### W1 · arg max 的邊際：蒙地卡羅 vs 式 (10)（S3）

- **教的概念**：$z=\arg\max(\tilde\alpha x+\tilde\sigma\epsilon)$ 服從 $\mathrm{Cat}(Tx+(1-T)\mathbf 1/K)$；正確類別機率為 $T+(1-T)/K$，其餘類別相等（對稱）。
- **輸入**：$K$ 滑桿 2–10（整數，預設 5）；$\tilde\alpha$ 滑桿 0–0.99（步長 0.01，預設 0.6）；「重新抽樣」按鈕（seed 由 7 起每次 +1）。樣本數固定 $N=20000$。
- **計算**：`r=mulberry32(seed)`；對 $n=1..N$：依序對 $k=0..K-1$ 抽 `e=gauss(r)`，`w_k=(k==0?α̃:0)+σ̃·e`；取最大者索引（並列取最小索引）計數。正確類別固定為索引 0。理論值用 4.0 的 $T$。標準誤 $SE_k=\sqrt{p_k(1-p_k)/N}$。
- **顯示**：
  - 主圖：$K$ 根長條（MC 頻率，實心；正確類別長條加斜線紋理並標「x（正確詞）」，其餘標 1…K−1）；每根上方畫理論值的水平粗短線（標記形狀：菱形端點）與 ±2SE 細線。
  - 小圖「一個樣本」：第一個樣本的 $w$（K 根長條，可為負，有 0 基線），最大者以 `--accent` 外框並標「arg max」。
  - 讀數：$\tilde\sigma$、$\tilde\alpha/\tilde\sigma$、$T$、理論 $p_x$、理論 $p_{\text{其他}}$、MC $p_x$、MC 其他類平均、$\max_k|MC_k-p_k|/SE_k$。
- **守門**：$\tilde\alpha$ 上限 0.99（$\tilde\sigma\ge0.141$）；重算以 `requestAnimationFrame` 合併連續 input（20000×10 次 gauss ≈ 10ms）。
- **應觀察到**：$\tilde\alpha=0$ 時所有類別 $1/K$；$\tilde\alpha$ 增加，正確類別上升、其餘一起下降且彼此相等；MC 落在 ±2SE 內（偶爾單一長條超出屬正常）。
- **無障礙**：圖 `role="img"` + `aria-label`（例「K = 5，α̃ = 0.60：理論 0.4124，模擬 0.4142」），於 `change` 或停止輸入 500ms 後更新；長條以紋理與文字標籤區分。

### W2 · T(α̃) 曲線與論文訓練窗（S4；S8 回指）

- **教的概念**：(11) 的形狀；$K$ 越大曲線越靠右；論文的訓練窗與 (11) 的關係（步驟 8 的不一致）。
- **輸入**：$K$ 分段按鈕 {2, 3, 10, 100, 1000, 30522「BERT」, 50257「GPT-2」}（預設 30522）；$\tilde\alpha$ 游標滑桿 0–1（步長 0.001，預設 0.95）；勾選框「顯示論文訓練窗」（預設開）；分段按鈕「橫軸」：$\tilde\alpha$ ／ $\tilde\alpha^2$（預設 $\tilde\alpha$）。
- **計算**：401 點 $\tilde\alpha_i=i/400$ 的 $T$；另以細灰線畫 $K=2$ 參考曲線（$K=2$ 時隱藏）。訓練窗兩條帶：
  - 帶 A「論文所寫：$\tilde\alpha_t=1-t$，$t\in[0.03,0.15]$」→ $\tilde\alpha\in[0.85,0.97]$（斜線填色）。
  - 帶 B「照 (11) 使 $T\in[0.05,0.95]$ 的範圍」→ `[alphaForT(0.05,K), alphaForT(0.95,K)]`（點狀填色）。
  - 橫軸為 $\tilde\alpha^2$ 時，兩條帶與曲線都換座標；並在帶 A 位置標「若讀作 $\tilde\alpha_t^2=1-t$」。
- **顯示**：橫軸 0–1，縱軸 $T\in[0,1]$；游標處垂直線與讀數框（固定在圖版右上角，不跟隨指標）：$\tilde\alpha$、$\tilde\alpha^2$、$\tilde\alpha/\tilde\sigma$、$T$、$P(z=x)=T+(1-T)/K$、帶 A 端點的 $T$、帶 B 端點。另加縮放鈕「放大 0.8–1.0」（橫軸範圍改為 [0.8, 1]），預設在 $K\ge1000$ 時自動開啟。
- **守門**：$K$ 改變時才重建 `G_K`（4401 次 erfc）；曲線 401×4401 次 exp ≈ 15ms，結果依 $K$ 快取；$\tilde\alpha=1$ 端點直接設 1。
- **應觀察到**：$K=2$ 曲線平緩；$K=30522$ 時 $T(0.85)\approx0.008$、$T(0.97)\approx0.45$，帶 A 與帶 B 不重合；切到 $\tilde\alpha^2$ 橫軸時兩帶大致重合。頁面文字在 S4 寫前兩點，在 S8 寫第三點。
- **無障礙**：$\tilde\alpha$ 游標為原生 range（鍵盤、觸控）；讀數框 `aria-live="polite"`（500ms debounce）；兩條帶以不同填色紋理 + 直接文字標籤區分。

### W3 · 邊際相同、轉移不同：式 (39) vs 式 (38)（S5）

- **教的概念**：推前的轉移機率依賴整個 $w_s$；同一個 $z_s$ 可以對應不同的下一步分布；USDM 核只看 $z_s$。
- **輸入**：$K=3$ 固定。$w_s$ 三個座標滑桿 $[-2,3]$（步長 0.05，預設 (1.2, 0.6, 0.0)）；預設按鈕「(1.2, 0.6, 0)」「(1.2, 0, 0.6)」「非最大座標相等 (1.2, 0.3, 0.3)」；$\tilde\alpha_s$ 滑桿 [0.5, 0.99]（預設 0.9）；$\tilde\alpha_t$ 滑桿 [0.05, $\tilde\alpha_s-0.05$]（預設 0.6，$\tilde\alpha_s$ 改變時夾住）。
- **計算**：
  - 高斯推前：$a=\tilde\alpha_t/\tilde\alpha_s$，$s=\sqrt{1-a^2}$，$c_j=a(w_s)_j$。以 $u=(l-c_i)/s$ 代換：$P_i=\int\phi(u)\prod_{j\ne i}\Phi(u+(c_i-c_j)/s)\,du$，梯形法 $u\in[-9,9]$、步長 0.01（1801 點）。顯示 $\sum_iP_i$（應為 1.0000）。
  - 離散核：$z_s=\arg\max(w_s)$（並列取最小索引並顯示「arg max 有並列」）；$\alpha_{t|s}=T(\tilde\alpha_t,3)/T(\tilde\alpha_s,3)$；$Q=\alpha_{t|s}e_{z_s}+(1-\alpha_{t|s})\mathbf 1/3$。
  - 總變異距離 $\frac12\sum_i|P_i-Q_i|$。
- **顯示**：三組並列長條（每組左：$P$「高斯路徑取 arg max」，實心；右：$Q$「USDM 轉移」，斜線），長條上標數值；$z_s$ 類別以「z_s」標籤標出；上方小圖畫 $w_s$ 三根長條。讀數：$\tilde\alpha_{t|s}$、$\alpha_{t|s}$、TV 距離、說明列（例「非 z_s 的兩類：高斯 0.2963 / 0.1399，USDM 都是 0.1946」）。
- **應觀察到**：(1.2, 0.6, 0) 與 (1.2, 0, 0.6) 的 $z_s$ 相同、$Q$ 相同，但 $P$ 的第 2、3 類互換；非最大座標相等時 $P$ 的兩類相等，但仍不等於 $Q$。
- **守門**：$\tilde\alpha_t<\tilde\alpha_s$ 保證 $a<1$、$s\ge\sqrt{1-(0.94/0.99)^2}>0$；每次重算 3×1801×2 次 Phi，同步完成。
- **無障礙**：滑桿 `aria-valuetext`；長條紋理 + 數字標籤；live region 宣告 TV 距離（500ms debounce）。

### W4 · 反向取樣：USDM 可反覆修改，P_DDT 每個 token 至多翻轉一次（S9）

- **教的概念**：USDM ancestral sampling（(4)(5)）中 token 可多次改變；$P_{DDT}$（(22)）中每個 token 只在 $x$ 與 $\arg\max_{i\ne m}\epsilon_i$ 間翻轉一次（Fig. 5 c、d）。
- **簡化（頁面寫出）**：兩欄都用「神諭」$x_\theta=x$（真實模型用學到的 $x_\theta$，會出錯）；詞彙 $K=17$：`Really, Through, Little, Good, Always, Number, Building, Question, Other, Think, Date, Want, Dual, Diffusion, For, Discrete, Data`；目標句 `Dual Diffusion For Discrete Data`（$L=5$，取自 Fig. 5）。USDM 用 $\alpha_t=1-t$（論文評估所用排程）；DDT 用 $\tilde\alpha_t=1-t$。
- **輸入**：步數 $T_{\text{steps}}$ 分段按鈕 {4, 8, 16, 32, 64}（預設 16）；seed（預設 7，「重新抽樣」+1）；時間滑桿 $n\in[0,T_{\text{steps}}]$（第 $n$ 步，$t=1-n/T_{\text{steps}}$，預設 $T_{\text{steps}}$）；「播放」按鈕（每步 250ms；reduced-motion 時直接跳到結尾）。
- **計算**：
  - USDM：`r=mulberry32(seed)`；初始 $z^\ell=\lfloor r()\cdot K\rfloor$（上限 $K-1$），$\ell=1..5$ 依序。對 $i=T_{\text{steps}}..1$：$t=i/T_{\text{steps}}$、$s=(i-1)/T_{\text{steps}}$、$\alpha_t=1-t$、$\alpha_s=1-s$、$\alpha_{t|s}=\alpha_t/\alpha_s$（$\alpha_s=0$ 不會發生，因 $s<t\le1$）；每個 $\ell$ 依 (4) 算 $K$ 維機率，用一個 `r()` 以累積和反查（落在最後則取 $K-1$）。
  - DDT：`r2=mulberry32(seed+1000)`；$\epsilon^\ell_k$ 依 $\ell$ 外、$k$ 內順序抽 `gauss(r2)`；對 $i=T_{\text{steps}}..0$：$\tilde\alpha=1-i/T_{\text{steps}}$，$z^\ell=\arg\max_k(\tilde\alpha[k=x^\ell]+\sqrt{\max(0,1-\tilde\alpha^2)}\epsilon^\ell_k)$。翻轉點：$j^*=\arg\max_{k\ne x}\epsilon_k$，$\Delta=\epsilon_{j^*}-\epsilon_x$，$\Delta>0$ 時 $\tilde\alpha^*=\Delta/\sqrt{1+\Delta^2}$，否則「不翻轉」。
- **顯示**：上下兩個面板（USDM、$P_{DDT}$），各為 5 欄：上方是第 $n$ 步的 5 個詞（pill；與目標相同者實心 `--c-sig` 底色並加 ✓，不同者外框），下方每欄一條「變動時間線」：$T_{\text{steps}}$ 格的細條，發生改變的步以實心方塊（USDM）或實心三角（DDT）標示，目前步以游標線標示；欄底顯示「改變 k 次」。DDT 面板另顯示每欄的 $\tilde\alpha^*$。兩面板右側合計「總改變次數」。
- **應觀察到**：USDM 的 token 常改變 2 次以上，最後一步一定回到目標（神諭）；DDT 每欄改變 0 或 1 次；DDT 的翻轉點與 $T_{\text{steps}}$ 無關（$\epsilon$ 固定）。
- **無障礙**：播放中不更新 live region；停止或滑桿 `change` 時宣告「第 n 步：Dual Diffusion …；USDM 已改變 a 次，DDT b 次」；改變標記用形狀區分，不只靠顏色。

### W5 · 課程的輸入：softmax(w/τ)（S8）

- **教的概念**：(20)：$\tau\to0$ 時 softmax 趨近 one-hot；$\tau=0.001$ 時輸入與 arg max 幾乎相同；$\tau$ 大時網路看到的是軟分布（保留 $w$ 的大小資訊）。
- **輸入**：$K=8$ 固定；$\tilde\alpha$ 滑桿 [0.5, 0.99]（預設 0.9）；$\log_{10}\tau$ 滑桿 [−3, 1]（步長 0.05，預設 −1，即 $\tau=0.1$）；chip「τ = 0.001（論文前 500K 步）」；「重新抽樣 ε」（seed 由 3 起 +1）。
- **計算**：`r=mulberry32(seed)`，$\epsilon_k$ 依序 8 個 `gauss(r)`；$w_k=\tilde\alpha[k=0]+\tilde\sigma\epsilon_k$；$p=\mathrm{softmax}(w/\tau)$ 以減去 $\max w$ 計算；熵 $H=-\sum p\log p$（$p=0$ 項略過，結果 $<10^{-12}$ 顯示為 0）；$1-p_{\max}<10^{-12}$ 時顯示「< 1e-12」。
- **顯示**：左圖 $w$ 長條（含 0 基線，arg max 加外框）；右圖 $p$ 長條（縱軸 0–1）與 one-hot 的虛線外框對照。讀數：$\tau$、$p_{\max}$、$p_x$、$H$（nats，最大 $\log8=2.0794$）、arg max 是否為正確詞、同 $\tilde\alpha$ 下 $T(\tilde\alpha,8)$。
- **應觀察到**：$\tau=10$ 時接近均勻；$\tau=1$ 時正確詞約 0.39；$\tau\le0.01$ 時已是 one-hot。頁面文字：論文 $K$ 為數萬，$\tau=0.001$ 時輸入更接近 one-hot，因此課程效果的來源（鬆弛 vs 時間窗）論文沒有分開驗證。
- **無障礙**：滑桿 `aria-valuetext`（「τ = 0.100」）；live region 500ms debounce。

---

## 5. 驗收數值表

全部由 `verify/` 腳本以與頁面相同的演算法產生；頁面 `?test` 的 `selfTest()` 比對其中標 ★ 的項目。

### 5.1 T(α̃)（`check_T.py`；容許 ±2e-4）

| K | 0.3 | 0.5 | 0.6 | 0.8 | 0.85 | 0.9 | 0.95 | 0.97 | 0.985 | 0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.1760 | 0.3169 | ★0.4041 | 0.6542 | 0.7461 | 0.8557 | 0.9685 | 0.9952 | 0.9999 | 1.0000 |
| 3 | 0.1387 | 0.2596 | 0.3390 | 0.5874 | 0.6874 | 0.8142 | 0.9565 | 0.9931 | 0.9999 | 1.0000 |
| 5 | 0.1009 | 0.1976 | ★0.2655 | 0.5018 | 0.6079 | 0.7534 | 0.9362 | 0.9892 | 0.9999 | 1.0000 |
| 100 | 0.0113 | 0.0275 | 0.0428 | 0.1319 | 0.1989 | 0.3384 | 0.6885 | 0.9124 | 0.9980 | 1.0000 |
| 30522 | 0.0001 | 0.0003 | 0.0005 | 0.0035 | ★0.0077 | 0.0237 | 0.1505 | ★0.4523 | 0.9364 | 0.9971 |
| 50257 | 0.0001 | 0.0002 | 0.0003 | 0.0025 | 0.0056 | 0.0180 | 0.1258 | 0.4090 | 0.9221 | 0.9961 |

- K=2 對照封閉式 $\mathrm{erf}(\tilde\alpha/(2\tilde\sigma))$：$\tilde\alpha$=0.1, 0.3, 0.6, 0.9 → 0.056655, 0.175978, 0.404117, 0.855708（★，差 < 1e-6）。$T(0,K)=0$（★）。
- 帶 B（`alphaForT`；±5e-4）：$K=30522$：0.9238、0.9856（★）；$K=50257$：0.9299、0.9861。若讀作 $\tilde\alpha^2=1-t$：$\tilde\alpha=\sqrt{0.85}=0.9220$、$\sqrt{0.97}=0.9849$，$K=30522$ 的 $T$ 為 0.0469、0.9337；$K=50257$ 為 0.0368、0.9190。

### 5.2 W1（`check_widgets.py`；N=20000，seed=7；與頁面同 PRNG 順序，應逐位相同，容許 ±5e-4）

| K | α̃ | T | 理論 p_x | 理論 p_其他 | MC p_x | MC 其他平均 | MC 其他 min / max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.6 | 0.2655 | 0.4124 | 0.1469 | ★0.4142 | 0.1465 | 0.1435 / 0.1527 |
| 5 | 0.0 | 0.0000 | 0.2000 | 0.2000 | 0.2001 | 0.2000 | 0.1948 / 0.2048 |
| 5 | 0.9 | 0.7534 | 0.8027 | 0.0493 | 0.7993 | 0.0502 | 0.0492 / 0.0516 |
| 2 | 0.6 | 0.4041 | 0.7021 | 0.2979 | 0.6997 | 0.3003 | — |
| 10 | 0.8 | 0.3906 | 0.4515 | 0.0609 | 0.4581 | 0.0602 | 0.0583 / 0.0636 |
| 3 | 0.3 | 0.1387 | 0.4258 | 0.2871 | 0.4341 | 0.2829 | 0.2803 / 0.2856 |

最後兩列偏離約 1.9、2.4 個標準誤；以 seed 7、8、9 各 10–20 萬樣本重跑皆落在 0.0015 內（$K=3$：0.4266、0.4269、0.4256 對 0.4258），確認無偏。頁面若 PRNG 實作不同導致 MC 數字不同，改以「$|MC-p|\le4SE$」驗收。

### 5.3 W3（`check_widgets.py`；容許 ±5e-4）

| $w_s$ | $\tilde\alpha_s,\tilde\alpha_t$ | $\tilde\alpha_{t|s}$ | 高斯推前 $P$ | $\alpha_{t|s}$ | USDM 核 $Q$ |
| --- | --- | --- | --- | --- | --- |
| (1.2, 0.6, 0.0) | 0.9, 0.6 | 0.6667 | ★(0.5638, 0.2963, 0.1399) | ★0.4163 | ★(0.6109, 0.1946, 0.1946) |
| (1.2, 0.0, 0.6) | 0.9, 0.6 | 0.6667 | (0.5638, 0.1399, 0.2963) | 0.4163 | (0.6109, 0.1946, 0.1946) |
| (1.2, 0.3, 0.3) | 0.9, 0.6 | 0.6667 | ★(0.5760, 0.2120, 0.2120) | 0.4163 | (0.6109, 0.1946, 0.1946) |
| (1.2, 0.6, 0.0) | 0.95, 0.3 | 0.3158 | (0.4197, 0.3279, 0.2524) | 0.1450 | (0.4300, 0.2850, 0.2850) |
| (0.5, 0.4, −1.0) | 0.9, 0.6 | 0.6667 | (0.4868, 0.4387, 0.0746) | 0.4163 | (0.6109, 0.1946, 0.1946) |
| (2.0, −0.5, 1.0) | 0.9, 0.6 | 0.6667 | (0.7172, 0.0298, 0.2530) | 0.4163 | (0.6109, 0.1946, 0.1946) |

所有 $\sum P=1.000000$（★）。預設值的 TV 距離 $=0.1018$（腳本值 0.101784；容許 ±5e-4）。平均後回到邊際（1500 樣本，seed 11）：(0.5590, 0.2215, 0.2194) 對 (0.5593, 0.2203, 0.2203)（只寫在 S5 文字，不在頁面計算）。

### 5.4 W4（`check_seq.py`；K=17、L=5；應逐位相同）

| $T_{\text{steps}}$ | seed | USDM 每欄改變次數 | DDT 每欄改變次數 | DDT 起點（t=1） |
| --- | --- | --- | --- | --- |
| 8 | 7 | [2, 3, 3, 3, 3] | [0, 1, 1, 1, 1] | Dual, Good, Dual, Question, Always |
| 16 | 7 | ★[2, 3, 4, 3, 3] | ★[0, 1, 1, 1, 1] | 同上 |
| 64 | 7 | [3, 8, 8, 3, 3] | [0, 1, 1, 1, 1] | 同上 |
| 16 | 8 | [3, 5, 3, 5, 3] | [1, 1, 1, 1, 1] | Data, Data, Through, Data, Question |

- seed 7 的 DDT 翻轉點 $\tilde\alpha^*$：Dual 不翻轉（$\Delta=-0.4574$）；Diffusion 0.7845（自 Good）；For 0.1217（自 Dual）；Discrete 0.9451（自 Question）；Data 0.3204（自 Always）（★ 檢查 Discrete 0.9451）。
- USDM 每次最後一列皆等於目標句（★）。$T_{\text{steps}}=16$、seed 7 的 USDM 第 0 列為 `Really Through Data Want Other`。
- 後驗 (4) 每列和為 1（★，隨機 1000 組，誤差 < 1e-12）。

### 5.5 W5（`check_soft_f.py`；ε 由 mulberry32(3) 產生 = [1.5492, 0.9837, −1.6850, 0.6078, −0.7394, 0.5697, −1.5649, 1.2935]；容許 ±1e-4）

| α̃ | τ = 10 | τ = 1 | τ = 0.1 | τ ≤ 0.01 |
| --- | --- | --- | --- | --- |
| 0.5 | p_max 0.1468, H 2.0733 | 0.3926, 1.6736 | 0.9992, 0.0066 | 1.0000, 0 |
| 0.9 | 0.1435, 2.0769 | ★0.3891, 1.7951 | ★0.9999, 0.0006 | 1.0000, 0 |
| 0.97 | 0.1407, 2.0780 | 0.3503, 1.8890 | 0.9999, 0.0007 | 1.0000, 0 |

所有情形 arg max 都是正確詞（索引 0）。$\tilde\sigma$：$\tilde\alpha$=0.85→0.5268、0.9238→0.3829〔即 $\sqrt{1-0.9238^2}$〕、0.97→0.2431、0.9856→0.1691。

### 5.6 S7 小表：f (49) 與 f_Duo (17)（`check_soft_f.py`；K=7、α=0.37、α′=−1；$x_\theta$ = mulberry32(5) 的 7 個 `r()` 正規化 = [0.1863, 0.2087, 0.0593, 0.1683, 0.0230, 0.1599, 0.1945]；x 的索引 m = 2）

| 情形 | f (49) | f_Duo (17) | 差 |
| --- | --- | --- | --- |
| $z_t=x$（i = 2） | ★−1.7499022162 | −1.7499022162 | 2.2e-16 |
| $z_t\ne x$（i = 4） | −1.6870330826 | −1.6870330826 | 4.4e-16 |
| $z_t\ne x$（i = 0） | −2.2488799387 | −2.2488799387 | 0 |

神諭 $x_\theta=x$ 時兩式皆為 0（★）。負值即 1.4 步驟 7 的正負號問題；小表下方註「依論文 (49)(17) 原式、$\alpha_t'=-1$ 計算；若以 $-d\alpha/dt$ 代入則全部變號」。

### 5.7 ODE 與後驗（`check_bayes_ode.py`，只用於文件，不在頁面計算）

$K=5$、$\tilde\alpha_t=1-t$：$t$=0.2、0.4、0.6 時 $dP/dt$ 與 (12) 右邊的正確類別分量分別為 −1.459504、−0.625977、−0.381713（兩邊相同）。(4) 與貝氏定理 2000 組相等到 1e-12。

---

## 6. 實作計畫

### 6.1 檔案與外殼（重用）

單一檔案 `tut/393/build/index.html`，複製 `tutorials/prediction-targets-k-diff.html` 的外殼並保留：`<head>`（字型、KaTeX 0.16.9 defer、auto-render）、CSS tokens（含深色模式三段式）、appbar 與主題切換（`pm-theme`）、`.toc` 章節目錄（`initToc`）、`.hero`、`.ch`、`.plate`／`.plate-h`／`.plate-lead`／`.plate-f`、callout、`details.check`、表格橫捲容器、`#pm-bar`、時間線 rail（`renderTimeline`）、`announce()` live region、`el/tx/lp/frame/arrow/widthOf` SVG 輔助、`mulberry32`、`gauss`、`renderMath`、`?test` → `selfTest()`。以下只列新增或改動。

### 6.2 新增／改動的樣式與文案常數

- 圖表色：k-Diff 頁的 `--c-par`、`--c-perp` 改名為 `--c-sig`（正確詞／訊號，沿用 #3F6E8C／#8DB4D3）與 `--c-noi`（雜訊／其他詞，沿用 #6B7A3A／#A9BC7C）。只用於圖中資料系列。
- 新樣式：`.tok`（W4 的詞 pill，`min-width` 依最長詞「Diffusion」，等寬字 13px；窄螢幕 5 欄改為可橫捲的單列）、`.tl-strip`（變動時間線，高 10px）、`.band-a`（斜線 pattern）、`.band-b`（點狀 pattern），SVG `<pattern>` 定義放在每張圖的 `<defs>`。
- 頁面標題 `<title>擴散對偶 Duo</title>`；appbar 短標「No. 393 · The Diffusion Duality」；`description`：「圖解 Sahoo 等人 ICML 2025 論文：高斯擴散取 arg max 後的邊際是均勻狀態離散擴散，並據此做課程學習與離散一致性蒸餾。」
- 連結：Hero「在關聯圖中查看」→ `../index.html#paper-393`；arXiv `https://arxiv.org/abs/2506.10892`；專案頁 `https://s-sahoo.com/duo`；筆記 `<slug>.html`。
- 演算法兩欄對照（S9 Alg. 1）沿用 k-Diff 頁 `.algo` grid 樣式。

### 6.3 新增 JS（單一 IIFE，函式名固定）

- 數學：`erfcNR(x)`、`Phi(z)`、`logPhi(z)`、`phi(z)`、`ZG`（Float64Array 4401）、`gTab(K)`（Map 快取）、`Top(at,K)`、`alphaForT(target,K)`。
- W1：`initW1()`、`w1Sample(K,at,N,seed)` → `{freq, first}`、`renderW1()`。
- W2：`initW2()`、`curveT(K)`（快取 401 點）、`renderW2()`；狀態 `{K, at, showBand, axis:'a'|'a2', zoom}`。
- W3：`initW3()`、`pushK(ws,ats,att)` → `P[]`、`usdmKernel(ws,ats,att)` → `{Q, a, zs, tie}`、`renderW3()`。
- W4：`VOCAB`、`TARGET`、`posteriorUS(zt,x,at,as,K)`（(4)）、`runUSDM(Tst,seed)` → 歷史陣列、`runDDT(Tst,seed)` → `{hist, flips}`、`initW4()`、`renderW4(n)`、播放用 `requestAnimationFrame` + 250ms 節流。
- W5：`initW5()`、`softmaxT(w,tau)`、`renderW5()`。
- S7 小表：`f49(i,m,xth,a,ad,K)`、`f17(i,m,xth,a,ad,K)`、`fillLossTable()`（載入時填入 5.6 的表格；JS 失敗時表格保留靜態數字）。
- `selfTest()` 新增斷言（全部 ★ 項）：`Top` 5.1 的 ★ 值與 K=2 封閉式、`Top(0,K)=0`、`alphaForT(0.05|0.95,30522)`；W1 seed 7 的 MC p_x；W3 預設 $P$、$Q$、$\alpha_{t|s}$、$\sum P$；W4 改變次數、最後一列＝目標、翻轉點；W5 兩格；f49=f17（1e-12）與神諭為 0；`scrollWidth <= innerWidth`。輸出一行 `selfTest: N passed, M failed`。
- 效能：W2 單一 $K$ 首次約 15ms；W1 約 10ms；W3 < 5ms；W4 < 1ms。全部同步，不需 worker。

### 6.4 數學排版

沿用 k-Diff 頁規則（`.katex-display` 橫捲、SVG 內用 Unicode、動態插入後 `renderMathInElement`）。(4)、(17)、(21)、(39) 在 < 600px 時用 `aligned` 拆行。公式旁右對齊等寬式號「(11)」。

### 6.5 引用與標籤

內文每個論文數字標出處（(n)／§／Table／Fig.）；元件圖版底部一行「讀數由本頁依 (n) 即時計算」；〔本頁推導〕以小 pill「本頁推導」（`--ink-3` 外框）+ 一行依據；論文內部不一致以 callout「論文原文與式 (11) 不一致」呈現，照錄兩邊數字，不替論文更正。區分：論文結果／作者解釋（「作者認為」）／本頁推導。

### 6.6 研究脈絡（notes/timeline.json）

- 原樣貼入 `<script type="application/json" id="timeline-data">`；15 筆，年份 2015–2025。
- `TL_GROUP`（以 arXiv ID 為 key）：

| 類別 | arXiv ID | 卡片連到的筆記 |
| --- | --- | --- |
| 高斯擴散 | 1503.03585、2006.11239、2011.13456、2107.00630 | `duo-gaussian-diffusion-foundations.html` |
| 離散擴散 | 2107.03006、2205.14987、2310.16834、2406.07524、2412.10193 | `duo-discrete-diffusion-masked-and-uniform.html` |
| 鬆弛 | 1611.01144、1611.00712 | `duo-curriculum-softmax-relaxation.html` |
| 蒸餾 | 2202.00512、2303.01469、2410.21035 | `duo-consistency-distillation-to-discrete.html` |
| 本文 | 2506.10892 | `duo-argmax-duality-theory.html` |

- 呈現與缺欄位處理沿用 k-Diff 頁（橫捲卡片、類別 chip、本文卡片 `--accent` 外框）。
- S13 延伸閱讀順序：`duo-lineage-hub`（總覽）、`duo-gaussian-diffusion-foundations`、`duo-discrete-diffusion-masked-and-uniform`、`duo-argmax-duality-theory`、`duo-curriculum-softmax-relaxation`、`duo-consistency-distillation-to-discrete`、`duo-self-correction-and-correctors`；每項附 frontmatter `title`。

### 6.7 文案規則

繁體中文、台灣用語、字面學術語氣；不用比喻、反問、宣傳形容詞。用詞：「推前」（pushforward，首次附英文）、「均勻狀態離散擴散（USDM）」、「遮罩式離散擴散（MDM）」、「自我修正」、「常見錯誤」。專有名詞首次出現附英文。「對偶」一律搭配「在邊際上」或「每個時刻的分布」。

### 6.8 建置後自我檢查

截圖 390／768／1024／1440 + 深色、鍵盤 focus、reduced-motion；依 AGENTS.md 步驟 6-3 逐項評分修正；`?test` 全數通過；逐條核對第 2 節 12 個學習目標的答案是否出現在內文；逐條核對 1.5 的五項誤讀。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 訓練窗與 (11) 不符 | 論文 $\tilde\alpha_t=1-t$、$[\beta,\gamma]=[0.03,0.15]$、$T\in[0.05,0.95]$ 三者照 (11) 互相矛盾；與 $\tilde\alpha_t^2=1-t$ 相符。Fig. 6 只有圖。 | W2 同時畫兩條帶與 $\tilde\alpha^2$ 橫軸；文字照錄論文，明寫「本頁無法判定哪一處是筆誤」。LO12 依此出題。 |
| R2 | $\phi$ 的印刷錯誤 | (11) 下方與 A.1 Part 3 的 $e^{-z^2}$。 | 全頁用 $e^{-z^2/2}$；S3 一行註明；常見錯誤第 1 條。 |
| R3 | $\alpha_t'$ 正負號慣例 | (3)(17)(49) 與 (12)(36) 相反；(49) 照印恆 ≤ 0（`check_sign.py`）。 | S2、S7 各一行說明；S7 小表照原式算出負值並註明。 |
| R4 | 「Rao-Blackwellized」的說法 | A.6 推導是逐點恆等，沒有積掉期望；論文稱降低變異。Table 3 的 1.7 機制未說明。 | 照錄論文說法，加「從推導看是等價改寫」與數值表；不替論文補理由。 |
| R5 | Thm 3.2 的嚴格性 | pmf 與密度的 KL；比較的是 Thm 3.1 構造，不是實際訓練的 Duo。 | S6 三點限定；稱「動機性論證」。 |
| R6 | 宣稱數字的限定 | 2×（相對 UDLM）、兩個數量級（Greedy-Tail、熵降）、「below 2 PPL」（只有一欄）、蒸餾勝過 MDLM（只在 $T\le32$）。 | 1.5 列為審查項；S10 照數字寫。 |
| R7 | 未寫明的設定 | 後 500K 步的 $t$ 區間；$T$ 快取的積分法；DCD 的 $\mu$、batch、lr；梯度變異估計方式；AR 的 Gen PPL 只在圖上。 | S11 表格中標「論文未寫明」；不推測數值。 |
| R8 | W4 的神諭簡化 | 真實 Duo 用學到的 $x_\theta$；神諭下 USDM 最後必然正確，可能讓讀者以為 USDM 不會出錯。 | 圖版副標與 plate-f 明寫「神諭 $x_\theta=x$，只用來展示 token 的變動模式」；文字補一句實際模型的 Gen PPL 數字。 |
| R9 | W1 的蒙地卡羅波動 | 個別長條可能超出 ±2SE。 | 讀數顯示最大偏離（單位 SE）；文字說明 ±2SE 約 95%。 |
| R10 | 詞彙數的出處 | 30522、50257 是 tokenizer 公開值，論文只寫「K = 30K」與 tokenizer 名稱。 | W2 按鈕標「BERT（30522）」「GPT-2（50257）」並在 plate-f 註明「詞彙數為 tokenizer 公開值」。 |
| R11 | 論文內部其他不一致 | Table 3 圖說「Alg. 1」、Fig. 13–16 圖說數字、C.8 LM1B/OWT、DCD/CDC、C.7 首句。 | 常見錯誤第 9 條照錄；S10 只引用 Table 5–7 的數字，不引用 Fig. 13–16 圖說。 |
| R12 | 初學者門檻 | 推前、CTMC 速率矩陣、ELBO 都是新概念。 | S1–S2 先用文字定義；(2)(12) 只要求讀懂「邊際隨時間的線性 ODE」；A.1、A.6 完整推導收在 `<details>`。 |
| R13 | 相對連結 | 建置位置與上線位置不同。 | 以上線位置撰寫；預覽只檢查外部連結與錨點。 |
| R14 | 筆記更新 | 建置前 notes 若增減，`TL_GROUP` 與 S13 清單需同步；未列 ID 歸「其他」並回報。 | 建置時重讀 `notes/`。 |
