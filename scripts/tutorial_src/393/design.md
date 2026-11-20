# 設計文件：〈The Diffusion Duality〉互動教學（round 1 修訂版）

- 論文：Subham Sekhar Sahoo、Justin Deschenaux、Aaron Gokaslan、Guanghan Wang、Justin Chiu、Volodymyr Kuleshov，arXiv 2506.10892（v3，ICML 2025，PMLR 267）。方法名 **Duo**。資料庫編號 No. 393。
- 原文：`tut/393/src/paper.txt`。「(n)」為論文式號；「§n」「A.n」「B.n」「C.n」「D.n」為章節與附錄；Fig.／Table／Alg. 照原文。顧問筆記：`tut/393/advisor.md`。審查：`tut/393/reviews/round1.md`。上一版：`design.round0.md`。
- 產出：`tut/393/build/index.html`（單檔）。上線 `tutorials/diffusion-duality.html`（slug `diffusion-duality`），站內相對連結以上線位置為準。
- 驗證：`tut/393/verify/`。`dmath.py` 是與頁面相同的演算法；`check_T.py`、`check_w1.py`、`check_w3.py`、`check_markov.py`、`check_w4.py`、`check_w5.py`、`check_loss.py`、`check_ode.py` 各自可用 `python3 -I <檔名>` 單獨執行（固定 seed）；`node check_prng.js` 確認 JS 與 Python 的 PRNG 輸出逐位相同。第 5 節數字全部來自這些腳本。
- 研究筆記：`tut/393/notes/`（總覽 `duo-lineage-hub` + 6 篇 + `timeline.json` 15 筆）。
- 標記：〔本頁推導〕＝本設計自行推導或數值驗證；**推定**＝為了對上論文數字而做的解讀，論文沒有寫明。頁面用相同標籤（6.5）。TeX 中的小於號一律寫 `\lt`、大於號 `\gt`。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

離散擴散模型分兩類：遮罩式（MDM，token 一旦揭露就不再改）與均勻狀態（USDM，每個 token 可被換成詞表內任一 token，反向取樣時可以反覆修改）。USDM 的困惑度一直落後 AR 與 MDM。本文的觀察：在 one-hot 向量 $x$ 上做高斯擴散 $w_t\sim\mathcal N(\tilde\alpha_tx,(1-\tilde\alpha_t^2)I_K)$ 再取 $z_t=\arg\max(w_t)$，$z_t$ 的條件邊際恰好是 USDM 的邊際 $\mathrm{Cat}(T(\tilde\alpha_t)x+(1-T(\tilde\alpha_t))\mathbf 1/K)$（(10)），$T$ 是一維積分定義的 Diffusion Transformation 運算子（(11)），且此邊際滿足 USDM 的線性 ODE（(12)）；作者稱之為 Diffusion Duality（(13)）。對偶**只在每個時刻的邊際上被證明**；論文另說明把高斯軌跡逐點取 arg max「不一定」是離散擴散過程（§3.1 末、A.3）。作者用它做兩件事：(1) 把離散 NELBO 改寫成對高斯潛變數的期望（(19)），再把網路輸入的 arg max 換成溫度 softmax（(21)，論文稱 curriculum learning）；(2) 以最優 denoiser 下的高斯 PF-ODE 直線軌跡取 arg max 得到確定性離散軌跡 $P_{DDT}$（(22)），做 Discrete Consistency Distillation（DCD，Alg. 1）。另提出與 UDLM 損失 (49) 代數等價、不需建構 $\bar x$ 的 $f_{Duo}$（(17)）。LM1B PPL 33.7、OWT 25.2（Table 1）；zero-shot 7 個資料集中 3 個勝過 AR（Table 2）；蒸餾後 ancestral 16 步、Greedy-Tail 8 步的 Gen PPL 低於未蒸餾模型 1024 步（Table 6）。

### 1.2 研究動機（§1）

1. 高斯擴散有成熟的參數化、快速訓練、PF-ODE 取樣與一步蒸餾技巧；離散擴散仍以 mean parameterization 與慢速 ancestral 取樣為主。
2. MDM 的蒸餾（SDTT）在少步時品質大幅下降；MDM 先驗是確定的（全為 mask），沒有「implicit」性質（雜訊到資料的確定性映射）。
3. USDM 可更新 token、不需 predictor-corrector，但落後。問題：USDM 能否與 MDM 競爭？能否借用底層高斯擴散的 implicit 性質做少步生成？

### 1.3 貢獻（§1 末段，照錄並加限定）

1. 連續與離散擴散的理論連結（§3）。
2. 「doubles the training speed of USDMs by introducing a low-variance curriculum」。限定（F2）：依據是 Duo（課程 + (17)）在 510K 步 PPL 35.2 勝過 UDLM 1M 步的 36.7（§5.1）；Duo 去掉課程在 1M 步為 35.0（Table 3），比 35.2 還低，所以「2×」沒有歸因到課程本身。
3. DCD 把取樣步數減少兩個數量級（§4.2、§5.2）。

### 1.4 從作者的角度逐步拆解

符號在頁面上於首次使用的章節定義（B6）；S1 末的可收合符號表只列 S1–S3 用到的：$K$、$\mathcal V$、$x$、$m$、$\mathrm{Cat}$、$\mathbf 1$、$t$、$\alpha_t$、$z_t$、$w_t$、$\tilde\alpha_t$、$\tilde\sigma_t=\sqrt{1-\tilde\alpha_t^2}$、$\phi$、$\Phi$。之後的符號（$T$、$\bar x$、$\kappa_t$、$\tau$、$[\beta,\gamma]$、$\delta$、$N$、$M$、$L$）在各章首次使用處以一行定義。$t=0$ 接近資料、$t=1$ 為先驗。符號重複：$\beta,\gamma$ 也是 A.1 的機率與 (44)(45) 的量；Table 5–7 的 $T$ 是取樣步數（頁面寫「步數 $T_{\text{steps}}$」）；Alg. 2 的 $K$ 是輪數。

#### 步驟 1：離散擴散的插值式寫法（§2.1，(1)–(6)）

- 邊際 (1)：$q_t(\cdot\mid x;\alpha_t)=\mathrm{Cat}(\cdot;\alpha_tx+(1-\alpha_t)\pi)$，MDM 取 $\pi=m$（mask），USDM 取 $\pi=\mathbf 1/K$。
- 邊際的線性 ODE (2) $\frac{d}{dt}q_t=Q_tq_t$；(3) 印成 $Q_t=+\frac{\alpha_t'}{K\alpha_t}[\mathbf 1\mathbf 1^\top-KI]$。〔本頁推導〕對 (1) 微分得 $\alpha_t'(x-\mathbf 1/K)=-\frac{\alpha_t'}{K\alpha_t}[\mathbf 1\mathbf 1^\top-KI]q_t$，與 (12)(36) 同號。頁面正文只寫「邊際隨時間的變化是一個線性 ODE」，速率矩陣與正負號放在 S2 的 `<details>`（B4）。
- 後驗 (4)：$q_{s|t}=\mathrm{Cat}\Big(\dfrac{K\alpha_tz_t\odot x+(\alpha_{t|s}-\alpha_t)z_t+(\alpha_s-\alpha_t)x+(1-\alpha_{t|s})(1-\alpha_s)\mathbf 1/K}{K\alpha_t\langle z_t,x\rangle+1-\alpha_t}\Big)$，$\alpha_{t|s}=\alpha_t/\alpha_s$。〔本頁推導〕即貝氏定理（`check_w4.py`，2000 組相等到 1e-12）。
- (5)：推論時以 $x_\theta(z_t,t)$ 代入 $x$；從 $z_1\sim\mathbf 1/K$ 逐步抽 $z_s\sim p^\theta_{s|t}$（ancestral sampling）。
- NELBO (6)：$\mathbb E_{t\sim U[0,1],q_t}f(z_t,x_\theta,\alpha_t;x)$，$f$ 在 (49)。頁面定義（S2 一句）：「NELBO 是 $-\log p_\theta(x)$ 的上界，由每個時刻『真實後驗與模型後驗的差異（KL 散度）』累加而成；訓練就是把它變小，報告的 PPL 是它的指數，所以是上界。」

#### 步驟 2：直接作用在 one-hot 上的高斯擴散（§2.2–2.3，(7)–(9)）

- (7) $\tilde q_t=\mathcal N(\tilde\alpha_tx,(1-\tilde\alpha_t^2)I_K)$，加在 one-hot 本身，不是 embedding（與 Diffusion-LM、Plaid 不同，§6）。(8) 之後未再使用。
- consistency distillation (9)：教師用 PF-ODE 從 $w_t$ 走一步到 $w_s$，學生在 $w_t$ 的輸出逼近教師在 $w_s$ 的輸出。頁面在 S11 才講。

#### 步驟 3：arg max 下的離散邊際（§3.1，A.1，(10)(11)(24)–(32)）

- 假設 $x_m=1$，$w=\tilde\alpha x+\tilde\sigma\epsilon$，$\epsilon\sim\mathcal N(0,I_K)$。$w_m\sim\mathcal N(\tilde\alpha,\tilde\sigma^2)$、$w_{i\ne m}\sim\mathcal N(0,\tilde\sigma^2)$ 獨立 (24)(25)；$Z=\max_{i\ne m}w_i$ 的 cdf $\Phi^{K-1}(l/\tilde\sigma)$ (27)；
  $$p_x:=P(\arg\max=m)=\int_{-\infty}^{\infty}\phi\Big(z-\tfrac{\tilde\alpha}{\sqrt{1-\tilde\alpha^2}}\Big)\Phi^{K-1}(z)\,dz\tag{28}$$
  其餘類別對稱，各 $(1-p_x)/(K-1)$ (30)；向量形式 $\mathrm{Cat}(\beta_t\mathbf 1+(1-K\beta_t)x)$ (32)。
- 頁面 S3 的寫法（B3）：先把結果寫成 $\mathrm{Cat}(a\,x+(1-a)\mathbf 1/K)$，$a:=\frac{Kp_x-1}{K-1}$ 稱為「保留係數」，說明這和 (1) 的 USDM 邊際同形；S4 才命名 $T(\tilde\alpha):=a$ 並寫出 (11)。
- $\phi$：(11) 下方與 A.1 Part 3 印成 $e^{-z^2}/\sqrt{2\pi}$，不是機率密度；A.1 Part 2、A.3 為 $e^{-z^2/2}/\sqrt{2\pi}$。頁面與所有數值用後者。

#### 步驟 4：運算子 T（(11)(34)）

- $T(\tilde\alpha)=\frac{K}{K-1}\Big[\int\phi(z-\tilde\alpha/\sqrt{1-\tilde\alpha^2})\Phi^{K-1}(z)dz-\frac1K\Big]$。
- 〔本頁推導〕$T(0)=0$；$\tilde\alpha\to1$ 時 $T\to1$；只依賴訊雜比 $\tilde\alpha/\tilde\sigma$ 與 $K$；$K=2$ 時 $T=\mathrm{erf}(\tilde\alpha/(2\tilde\sigma))$（$\tilde\alpha=0.6$：0.404117）。$K$ 越大曲線越靠近 $\tilde\alpha=1$：$K=30522$ 時 $T(0.85)=0.007730$、$T(0.97)=0.452299$。
- §4.1.2／C.7：「$K=30K$、$\tilde\alpha_t\in[0.85,1]$ 時 $T$ 幾乎涵蓋 $[0,1]$」——照 (11) 成立。

#### 步驟 5：ODE 與對偶陳述（A.2，(12)(13)(33)–(36)）

- $\frac{d}{dt}p_t=\beta_t'(\mathbf 1-Kx)$；$[\mathbf 1\mathbf 1^\top-KI]p_t=(1-K\beta_t)(\mathbf 1-Kx)$ ⇒ (33) ⇒ 以 $\alpha_t=1-K\beta_t$ 得 (36) $\frac{d}{dt}p_t=-\frac{\alpha_t'}{K\alpha_t}[\mathbf 1\mathbf 1^\top-KI]p_t$，即 (12) $\frac{d}{dt}P_t=-\frac{T'(\tilde\alpha_t)}{KT(\tilde\alpha_t)}[\mathbf 1\mathbf 1^\top-KI_K]P_t$（$T'$ 對 $t$ 微分）。`check_ode.py`：$K=5$、$\tilde\alpha_t=1-t$ 時兩邊在 $t=0.2,0.4,0.6$ 的正確類別分量為 −1.459504、−0.625977、−0.381713，相等。
- (13) $q_t(z_t\mid x;T(\tilde\alpha_t))=[\arg\max]_\star\tilde q_t(w_t\mid x;\tilde\alpha_t)$。作者稱 (35)(36) 為 USDM 的「unique signatures」；〔本頁推導〕(36) 由 (35) 直接微分可得，頁面加一句說明。
- 意義：凡只依賴「每個時刻的邊際」的量（例如 NELBO 中對 $z_t$ 的期望），可改用高斯潛變數計算。

#### 步驟 6：邊際相同，軌跡呢？（A.3，(37)–(40)；§3.1 末句；Fig. 1 圖說）

- 論文：高斯核 (37) $w_t\mid w_s\sim\mathcal N(\tilde\alpha_{t|s}w_s,(1-\tilde\alpha_{t|s}^2)I)$；USDM 核 (38) 的非 $z_s$ 類機率相等；**給定整個 $w_s$** 的推前 (39) 只在非最大座標相等時才有此對稱（機率 0），故 (40) $q_{t|s}(\cdot\mid z_s)\ne[\arg\max]_\star\tilde q_{t|s}(\cdot\mid w_s)$；結論寫「does not necessarily follow a discrete diffusion process」，§3.1 寫「might not」。
- 限定（F1）：(39)(40) 是以 $w_s$ 為條件；它說明 $z_t$ 不是只由 $z_s$ 決定，但沒有直接證明「$z$ 序列不是任何 USDM 鏈」。頁面照論文用「不一定」。
- 〔本頁推導〕以 $z$ 為條件的蒙地卡羅（`check_markov.py`；$K=3$、$\tilde\alpha_s,\tilde\alpha_t,\tilde\alpha_u=0.9,0.6,0.3$；高斯鏈 (37)；40 萬條，seed 2024）：$P(z_t=z_s\mid z_s=x)=0.5915$、$P(z_t=z_s\mid z_s\ne x)=0.4573$，USDM (38) 對兩者都給 0.6109；Markov 檢查：$P(z_u=z_t\mid z_t\ne x,z_s=z_t)=0.5104$ 而 $z_s\ne z_t$ 時 0.4792（USDM 0.6061）。所以離散化的高斯軌跡在這個例子中確實不是 USDM 鏈，只是理由不同於 A.3 的論證。
- 〔本頁推導〕W3 預設值的總變異距離 0.1018 是「給定 $w_s$」的差距；即使非最大座標相等，推前 (0.5760, 0.2120, 0.2120) 也不等於 (38) 的 (0.6109, 0.1946, 0.1946)。
- 頁面用語：「對偶在**邊際**層級成立」；不寫「離散擴散等於高斯擴散取 arg max」而不加限定。

#### 步驟 7：兩個定理（§3.2，A.4–A.5，(14)–(16)(41)–(47)）

- Thm 3.1：存在離散反向核 (15) 使 $p^\theta_t=[\arg\max]_\star\bar p^\theta_t$ (14)。A.4 的前提（F4）：高斯先驗 $\bar p_1=\mathcal N(0,I_K)$（base case，由對稱性推前為 $\mathbf 1/K$）；(15) 除以 $p^\theta_t(z_t)$；歸納步實質是一步邊際化 (43)。(44) 是令 (5)＝(15) 解出的 $x_\theta$，論文沒有證明它落在機率單體 $\Delta^K$ 內。實驗沒有實作，§7.2 列為未來工作。
- Thm 3.2：$\mathbb E_{q_{data}}\log p^\theta_{data}\ge\mathbb E_{q_{data}}\log\bar p^\theta_{data}$ (16)，用 data-processing inequality (46) 與 $[\arg\max]_\star q_{data}=q_{data}$。作者據此在離散潛變數上定義模型，以離散 NELBO 訓練與評估。
- 頁面限定：(a) 比較的是 Thm 3.1 構造出的一對取樣器的**邊際概似**，不是 ELBO（Fig. 1 圖說寫 ELBO）；(b) 實際訓練的 Duo 不是該構造；(c)（F5）〔本頁推導〕$q_{data}$ 是 one-hot 上的 pmf、$\bar p^\theta_{data}$ 是 $\mathbb R^K$ 上的密度，$q_{data}$ 對 $\bar p$ 不絕對連續，$D_{KL}(q_{data}\Vert\bar p^\theta_{data})=+\infty$，所以照原式這個不等式沒有提供資訊；頁面稱之為「動機性論證」。

#### 步驟 8：Duo 的損失與取樣器（§3.3，A.6，(17)(18)(48)–(51)）

- Greedy-Tail：最後一步改取 $\arg\max p^\theta_{0|\delta}$，類似 nucleus sampling 降低熵。
- 定義（S8 首次使用處）：$\bar x=K\alpha_tx+(1-\alpha_t)\mathbf 1$、$\bar x_\theta=K\alpha_tx_\theta+(1-\alpha_t)\mathbf 1$、$i$＝$z_t$ 的索引、$m$＝$x$ 的索引、$\kappa_t=(1-\alpha_t)/(K\alpha_t+1-\alpha_t)$。
- (49) $f=-\frac{\alpha_t'}{K\alpha_t}\Big[\frac K{\bar x_i}-\frac K{(\bar x_\theta)_i}-\sum_j\frac{\bar x_j}{\bar x_i}\log\frac{(\bar x_\theta)_i\bar x_j}{(\bar x_\theta)_j\bar x_i}\Big]$。
- (17) $f_{Duo}=-\frac{\alpha_t'}{K\alpha_t}\Big[\frac K{\bar x_i}-\frac K{(\bar x_\theta)_i}-(\kappa_t\mathbb 1_{z_t=x}+\mathbb 1_{z_t\ne x})\sum_j\log\frac{(\bar x_\theta)_i}{(\bar x_\theta)_j}-K\frac{\alpha_t}{1-\alpha_t}\log\frac{(\bar x_\theta)_i}{(\bar x_\theta)_m}\mathbb 1_{z_t\ne x}-\big((K-1)\kappa_t\mathbb 1_{z_t=x}-\tfrac1{\kappa_t}\mathbb 1_{z_t\ne x}\big)\log\kappa_t\Big]$。
- 〔本頁推導〕$\bar x$ 只有兩種值，比值 $\bar x_j/\bar x_i\in\{1,\kappa_t,1/\kappa_t\}$，分 $z_t=x$／$z_t\ne x$ 兩種情形即得 (17)。這是固定 $(z_t,x)$ 的代數恆等式（`check_loss.py`：差 ≤ 4.4e-16）。論文稱 Rao-Blackwellized 並說降低變異（A.6），但推導沒有積掉任何期望；論文自己報告兩式評估皆 33.7。Table 3 去掉 (17) 後 35.0→36.7，但最後一列是重新訓練的 UDLM（「Reduces to UDLM」），差距可能含其他實作差異（F6）；論文沒有說明機制。
- 〔本頁推導〕正負號：若 $\alpha_t'=d\alpha_t/dt\lt0$，(49) 照印恆 ≤ 0（`check_loss.py`：20000 組隨機情形全部 ≤ 0；seed 0）；KL 速率應 ≥ 0。若 $\alpha_t'$ 讀作 $-d\alpha_t/dt$，(3)(17)(49) 合理但 (12)(36) 多一個負號。不影響 (17)＝(49)。
- 序列 NELBO 為 token 損失之和 (18)。

#### 步驟 9：以高斯潛變數改寫 NELBO 與 softmax 鬆弛（§4.1，B.1，(19)–(21)，C.2，C.7）

- (19)：由 (10)，對 $z_t\sim q_t$ 的期望換成對 $w_t\sim\tilde q_t$，$z_t:=\arg\max(w_t)$、$\alpha_t:=T(\tilde\alpha_t)$；只用到每時刻的邊際，所以步驟 6 的限定不影響它。B.1：1000 個 antithetic $t$ 樣本，兩種算法 PPL 皆 33.7。
- (20) $\arg\max(w)=\lim_{\tau\to0^+}\mathrm{softmax}(w/\tau)$；(21) 只把 **denoiser 的輸入** 換成 $\mathrm{softmax}(w^\ell_t/\tau)$，損失仍是 $f_{Duo}$（$z_t:=\arg\max$、$\alpha_t:=T(\tilde\alpha_t)$），$t\sim U[\beta,\gamma]$。$\tau\gt0$ 時不是合法 NELBO；只有 $\tau\to0^+$、$\beta=0,\gamma=1$ 時才是。評估用 (18)。
- 「課程學習」首見定義：依難度由易到難安排訓練（Bengio et al. 2009）；此處「易」指 $\tau$ 大、網路看到的輸入保留較多 $w$ 的數值資訊。
- C.2：denoiser 第一層以 $y^\top E$（$E\in\mathbb R^{K\times m}$）計算 embedding，連續輸入是 embedding 的凸組合。
- 作者理由（§4.1.2）：arg max 對擾動極敏感，離散化丟失訊號；softmax 保留訊號，$\tau$ 控制難度。
- 時間窗（C.7）：損失被 $\alpha_t'$ 加權，$\alpha_t'\approx0$ 處訊號少；選 $t$ 使 $T(\tilde\alpha_t)\in[0.05,0.95]$，論文說兩個 tokenizer 相近，皆為 $[\beta,\gamma]=[0.03,0.15]$；§5 開頭 $\tilde\alpha_t=1-t$。
- 與 (11) 不符〔本頁推導，`check_T.py`〕：照 (11)，$\tilde\alpha\in[0.85,0.97]$ 給 $T\in[0.0077,0.4523]$（$K=30522$）；$T\in[0.05,0.95]$ 需要 $\tilde\alpha\in[0.9238,0.9856]$（$K=50257$：$[0.9299,0.9861]$）。**推定**（F3）：若讀作 $\tilde\alpha_t^2=1-t$，$t\in[0.03,0.15]$ 給 $\tilde\alpha\in[0.9220,0.9849]$、$T\in[0.0469,0.9337]$（$K=30522$），接近 [0.05, 0.95]；但 $K=50257$ 為 $[0.0368,0.9190]$，較不吻合。這只是數值上的巧合，論文沒有這樣寫。頁面照錄論文、列出兩種讀法，不判定筆誤所在。
- 排程：§4.1 文字說溫度「gradually decreased… initially fully relaxed」；§5.1 的實際設定是兩段：前 500K 步 $\tau=0.001$、$[\beta,\gamma]=[0.03,0.15]$，之後到 1M 步 $\tau=0$。兩處描述不同（F9），頁面照錄兩者。後 500K 步的 $t$ 區間論文沒寫；PPL 評估 $\alpha_t=1-t$。
- $T$ 預算 100K 組 $(\tilde\alpha_t,T(\tilde\alpha_t))$ 快取，積分法未說明（§5）。
- 偏差–變異（Fig. 8，LM1B 150K 步，$\tau\in\{0,0.001,0.01,0.1\}$）：$\tau=0.1$ 損失急降＝偏差過大；$\tau=0.001$ 接近 $\tau=0$ 且變異低。Table 4（$\tau=0.001$，top-100 權重梯度變異，有／無課程）：10K 步 0.30／11.7；100K 步 0.86／55.1；500K 步 1.15／1.92。論文沒有把「softmax 鬆弛」與「$t$ 限制在 $[\beta,\gamma]$」分開消融。

#### 步驟 10：P_DDT（§4.2，B.2.1，(22)(57)–(60)，A.7）

- 先備（頁面 S10 以 3–4 句教）：高斯擴散除了隨機的反向取樣，還有一條確定性的常微分方程（PF-ODE）：給定起點雜訊，軌跡唯一；DDIM (57) 是它的離散步。
- 問題：訓練完的 Duo 只吃離散輸入，不能在高斯空間解 ODE。
- 做法：假設最優 denoiser $x_\theta=x$，(57) 退化為 $w_s=\tilde\alpha_sx+\sqrt{1-\tilde\alpha_s^2}\epsilon$（(58)，原文左邊寫 $z_s$、右邊下標 $t$），軌跡 (59) 是 $x$ 與 $\epsilon$ 的固定內插；取 arg max 得 (22) $P_{DDT}$。
- 〔本頁推導〕$(x,\epsilon)$ 固定時非 $m$ 座標都是 $\tilde\sigma_t\epsilon_i$，最大者恆為 $j^*=\arg\max_{i\ne m}\epsilon_i$，故 $z_t\in\{x,e_{j^*}\}$，$z_t=x\iff\tilde\alpha_t/\tilde\sigma_t\gt\Delta:=\epsilon_{j^*}-\epsilon_m$；每個 token 至多翻轉一次，翻轉點 $\tilde\alpha=\Delta/\sqrt{1+\Delta^2}$；$\Delta\le0$ 時恆為 $x$。A.7：只能取兩個值，像 MDM 的「carry-over（揭露後沿用）」；$P_{DDT}$ 只在蒸餾時產生輸入，Duo 本身不以此方式取樣。

#### 步驟 11：DCD（§4.2，(23)，Alg. 1–2，B.2.2，C.8）

- 先備（頁面 S11 以 3–4 句教）：知識蒸餾＝讓學生模型模仿教師模型的輸出；consistency distillation（(9)）讓學生在較吵的輸入上直接輸出教師在較不吵輸入上的結果，因此能用較少步數生成。
- (23)：同一條 $P_{DDT}$ 上的相鄰兩點 $(z_s,z_t)$，$s=\max(t-\delta,0)$；最小化 $\sum_\ell D_{KL}(x^\ell_\theta(z_t,t),x^\ell_{\theta^-}(z_s,s))$（學生看較吵的 $z_t$）。
- Alg. 1：$N$ 輪、每輪 $M$ 步；每輪開始 $\theta^-\leftarrow$ stopgrad$(\theta)$（教師＝上一輪結束時的學生權重）；每輪結束 $\delta\leftarrow2\delta$；回傳 $\theta_{ema}$。實驗 $N=5$、$M=10K$、初始 $\delta=1/512$（五輪 1/512、1/256、1/128、1/64、1/32〔本頁推導：連乘〕）。
- Alg. 2（F8）：B.2.2 文字說用「pre-training 時得到的 EMA 權重」當教師，Alg. 2 偽碼則是每輪 $\theta^-\leftarrow$ stopgrad$(\theta_{ema})$（訓練中持續更新的 EMA）。頁面兩種寫法都列。結果較差（Fig. 10、Table 6）。
- 頁面用語：DCD 沒有用教師解 ODE，也沒有 (9) 的 $\lambda(t)$；是「沿 $P_{DDT}$ 相鄰兩點的輸出分布做 KL 對齊並逐輪加倍步長」，結構上接近 SDTT。
- 未寫明：EMA 係數 $\mu$、Duo 蒸餾的 batch 與 lr（只說同 SDTT；C.8 給 MDLM 的 batch 128、lr 6e-5、warmup 500 步；C.8 寫 LM1B，§5.2 寫 OWT）。

#### 步驟 12：實驗（§5，Table 1–7）

- 設定（§5、C.5）：170M DiT（12 層、hidden 768、12 heads、時間嵌入 128、旋轉位置編碼、adaLN）；AdamW、batch 512、lr 3e-4（warmup 2500）、dropout 0.1、1M 步、8×H100、bfloat16。LM1B 長度 128（bert-base-uncased，有／無 packing），OWT 長度 1024（GPT-2），保留最後 100K 篇驗證。取樣一律 float64。
- 指標（S12 首段定義，B8）：PPL＝NELBO 的指數（上界，越低越好）；Gen PPL＝用 GPT-2 Large 評估生成文字的困惑度（越低越流暢）；entropy＝生成序列的平均熵（論文用語，越低越缺乏多樣性）；兩者要一起看。
- Table 1（PPL↓；欄＝LM1B 無 packing、LM1B、OWT）：AR 22.3、22.8、17.5；MDLM 27.0、31.8†、23.2；SEDD Absorb 32.7、–、24.1；SEDD Uniform 40.3、–、29.7；UDLM‡ 31.3、36.7、27.4；Duo 29.9、33.7、25.2。†＝取自 Arriola et al.，‡＝重新訓練；與 MDLM 差距 2.9、1.9、2.0（中間一欄的 MDLM 數字來源不同，F12）。論文寫「below 2 PPL points」，只有中間一欄小於 2。
- 速度：510K 步 PPL 35.2，UDLM 1M 步 36.7（限定見 1.3）。Plaid LM1B 100K 步 89.9，Duo 同步數 43.0（C.1）。
- Table 3：Duo 33.7；去掉課程 35.0；再去掉 (17)（＝UDLM）36.7。§7.1：無法調參時建議不用課程。
- Table 2（PTB、Wikitext、LM1B、Lambada、AG News、Pubmed、Arxiv）：Duo 89.35、33.57、73.86、49.78、67.81、44.48、40.39；AR 82.05、25.75、51.25、51.28、52.09、49.01、41.73；MDLM 95.26、32.83、67.01、47.52、61.15、41.89、37.37；UDLM 112.82、39.42、77.59、53.57、80.96、50.98、44.08。Duo 勝 AR 3/7（Lambada、Pubmed、Arxiv），勝 SEDD Absorb 4/7，勝 MDLM 1/7（PTB）；MDLM 也在同樣三個勝 AR。
- Table 5（未蒸餾，Gen PPL、entropy）：Duo 1024 步 77.69、5.55；8 步 198.27、5.57；MDLM 1024 步 104.85、5.63；8 步 830.82、5.91。
- 蒸餾（Table 6–7，Round 5）：ancestral 16 步 75.24、5.53（對 base ancestral 1024 步 77.69、5.55 → 64×）；Greedy-Tail 8 步 69.58、5.30（論文對照 base ancestral 1024 步 → 128×）。同取樣器對照（F11）：base Greedy-Tail 1024 步為 71.72、5.22，所以 Greedy-Tail 本身就會降低熵。對 MDLM+SDTT（Round 5，ancestral）：8 步 111.88 對 193.05；16 步 75.24 對 89.17；32 步 61.31 對 62.29；64 步 55.83 對 47.04；1024 步 50.55 對 36.89。
- Fig. 12：SDTT 蒸餾的 MDLM 熵隨輪次下降，DCD 蒸餾的 Duo（Greedy-Tail）上升。

#### 步驟 13：實作細節與常見錯誤（頁面 S13 逐條列出）

1. $\phi=e^{-z^2/2}/\sqrt{2\pi}$，不是 (11) 下方的 $e^{-z^2}/\sqrt{2\pi}$。
2. $Q_t$ 正負號以 (12)(36) 為準；(3)(17)(49) 的 $\alpha_t'$ 慣例相反。
3. 算 $T$：$\Phi^{K-1}=\exp((K-1)\log\Phi)$；直接算 $\int[\phi(z-\mu)-\phi(z)]\Phi^{K-1}dz$（因 $\int\phi\Phi^{K-1}=1/K$），避免先積分再減 $1/K$ 的相消〔本頁推導〕。論文用 100K 組快取。
4. 時間窗照「$T(\tilde\alpha_t)\in[0.05,0.95]$」的目的以 (11) 自行求，不要直接套 $[0.03,0.15]$ 與 $\tilde\alpha_t=1-t$。詞彙數（F13）：論文只寫 tokenizer 名稱與「K = 30K」，沒給確切 $K$；實作若把詞表補齊（padding）到其他大小，$T$ 會改變（$T(0.97)$：$K=30522$ 為 0.4523、$K=50257$ 為 0.4090）。
5. (21) 只換 denoiser 輸入；損失的 $z_t$ 仍是 arg max；$\tau\gt0$ 時不是 NELBO。
6. 後 500K 步 $\tau=0$；PPL 用 $\alpha_t=1-t$、單一 $t$ 的低差異取樣（C.3、C.4）。
7. DCD 教師＝上一輪學生權重；每輪 $\delta$ 加倍；學生輸入是較吵的 $z_t$。
8. 對偶不要用在轉移核：邊際相同不代表路徑相同。
9. 論文其他不一致（照錄）：Table 3 圖說與 §5.1 把課程標成「Alg. 1」；Fig. 13–16 圖說數字與 Table 6–7 不完全一致（Fig. 15–16 的 79.24／5.25、42.46／5.25 是 Table 6 的 Alg. 2 Greedy-Tail）；DCD／CDC（Fig. 12）混用；C.7 首句說畫 $\tilde\alpha_t$，Fig. 6 畫的是 $T(\tilde\alpha_t)$。

#### 步驟 14：限制與未來工作（§7）

課程有 $\tau$ 與課程長度兩個超參數，其他領域可能需調整；USDM 困惑度仍落後 MDM，但少步生成較好。未來：以 (15) 讓高斯潛變數累積分類器梯度引導離散取樣；以 (44) 把 ε／v 參數化帶入離散 denoiser。

### 1.5 頁面上必須避免的誤讀（審查逐條檢查）

1. 不寫「離散擴散就是高斯擴散取 arg max」而不加「在邊際上」；軌跡用「不一定」。
2. 不把 Thm 3.2 寫成「Duo 的 ELBO 比高斯擴散緊」。
3. 不把 $f_{Duo}$ 的效果歸因於變異數下降；Table 3 的 36.7 是重訓的 UDLM。
4. 「2×」寫成「Duo（課程 + (17)）510K 步勝過 UDLM 1M 步」；「兩個數量級」註明 Greedy-Tail 8 步、熵較低，且 Greedy-Tail 本身降熵。
5. 「蒸餾後勝過 MDLM」只在 $T_{\text{steps}}\le32$。
6. W4 展示的是「可修改」，不是「自我修正」（已知真值的 denoiser 不會犯錯）。

---

## 2. 學習目標（驗收測試）

每章結尾以 `<details class="check">`（summary「檢查理解」）列題目與參考答案。**每題的參考答案以一個段落完整出現在該章內文中**（建置時把答案段落標 `data-lo="LOn"`，審查用）。

| # | 學習目標 | 檢查題 | 參考答案 | 章 |
| --- | --- | --- | --- | --- |
| LO1 | 區分 MDM 與 USDM | 兩者的先驗各是什麼？哪一種在反向取樣時能修改已產生的 token？ | MDM 的先驗是 mask token，token 揭露後不再改變；USDM 的先驗是 $K$ 個詞的均勻分布 $\mathbf 1/K$，每個 token 在反向過程中可以被反覆更新（§2.1）。 | S1 |
| LO2 | 理解邊際並寫出 USDM 的邊際與取樣 | 「邊際」與「路徑」差在哪？USDM 的邊際是什麼？如何由 $z_t$ 抽 $z_s$？ | 邊際是單一時刻 $z_t$ 的分布，路徑是多個時刻的聯合分布；USDM 的邊際是 $\mathrm{Cat}(\alpha_tx+(1-\alpha_t)\mathbf 1/K)$ (1)；取樣時把 $x_\theta(z_t,t)$ 代入真實後驗 (4) 得 (5)，從 $z_1\sim\mathbf 1/K$ 開始逐步抽樣（ancestral sampling）。 | S2 |
| LO3 | 推導 arg max 的邊際 | $w=\tilde\alpha x+\tilde\sigma\epsilon$ 時，正確類別成為 arg max 的機率是什麼？整個分布的形式？ | $p_x=\int\phi(z-\tilde\alpha/\tilde\sigma)\Phi^{K-1}(z)dz$ (28)；其餘 $K-1$ 類對稱，各 $(1-p_x)/(K-1)$；分布可寫成 $\mathrm{Cat}(a\,x+(1-a)\mathbf 1/K)$，$a=(Kp_x-1)/(K-1)$，與 USDM 邊際同形。 | S3 |
| LO4 | 說出 $T$ 的定義與性質 | $T$ 是什麼？$T(0)$、$T(\tilde\alpha\to1)$？$K=2$ 的封閉式？$K$ 大時？ | $T(\tilde\alpha)$ 就是 S3 的保留係數 $a$，寫成 (11)；$T(0)=0$，$\tilde\alpha\to1$ 時 $T\to1$；$K=2$ 時 $T=\mathrm{erf}(\tilde\alpha/(2\tilde\sigma))$（本頁推導）；$K$ 越大曲線越靠近 $\tilde\alpha=1$，$K=30522$ 時 $T(0.85)=0.0077$、$T(0.97)=0.4523$。 | S4 |
| LO5 | 說明對偶的完整陳述 | 除了邊際的形式，作者還驗證了什麼？結論寫成哪一式？ | 驗證 arg max 後的邊際隨時間滿足 USDM 的線性 ODE (12)，擴散參數為 $T(\tilde\alpha_t)$；結論為 (13)：$q_t(\cdot\mid x;T(\tilde\alpha_t))=[\arg\max]_\star\tilde q_t(\cdot\mid x;\tilde\alpha_t)$，即兩者在每個時刻的邊際相同。 | S5 |
| LO6 | 區分邊際對偶與路徑 | 逐點取 arg max 的高斯軌跡一定是 USDM 的 Markov 鏈嗎？論文怎麼論證？ | 不一定。論文說明給定整個 $w_s$ 時，下一步的機率 (39) 依賴 $w_s$ 的每個座標，而 USDM 核 (38) 只依賴 $z_s$，兩者一般不相等 (40)；這個論證以 $w_s$ 為條件，論文的結論措辭是「不一定」。本頁以 $z$ 為條件的模擬也得到不相等的轉移機率（0.5915、0.4573 對 0.6109）。 | S6 |
| LO7 | 正確陳述 Thm 3.2 | Thm 3.2 比較什麼？有什麼限制？作者用它決定什麼？ | 比較 Thm 3.1 構造的離散與高斯取樣器在真實資料下的邊際概似，離散的不低於高斯的 (16)；它不是 ELBO 的比較，而且照原式 pmf 對密度的 KL 為無限大，屬動機性論證；作者據此在離散潛變數上定義模型、用離散 NELBO 訓練與評估。 | S7 |
| LO8 | 說明 NELBO 與 $f_{Duo}$ | NELBO 是什麼？(17) 與 (49) 的關係？論文如何確認？ | NELBO 是 $-\log p_\theta(x)$ 的上界，由各時刻後驗的 KL 累加，PPL 是它的指數；(17) 對每個固定的 $(z_t,x)$ 都等於 (49)，是不需建構 $\bar x$ 的代數改寫；論文用兩式評估同一模型都得到 33.7。 | S8 |
| LO9 | 描述課程學習 | (21) 改了哪裡？$\tau$ 與時間區間怎麼設？它是合法的 NELBO 嗎？ | 只把 denoiser 的輸入由 $\arg\max(w_t)$ 換成 $\mathrm{softmax}(w_t/\tau)$，損失仍是 $f_{Duo}$；依 §5.1，前 500K 步 $\tau=0.001$、$t\sim U[0.03,0.15]$，之後 $\tau=0$；$\tau\gt0$ 時不是合法 NELBO，評估用 (18)；論文說 $[0.03,0.15]$ 是為了讓 $T\in[0.05,0.95]$，但照 (11) 與 $\tilde\alpha_t=1-t$ 計算並不成立。 | S9 |
| LO10 | 解釋 $P_{DDT}$ | $P_{DDT}$ 怎麼構造？其中一個 token 最多改變幾次？ | 假設 denoiser 完全正確（$x_\theta=x$），高斯 PF-ODE 軌跡變成 $\tilde\alpha_tx+\tilde\sigma_t\epsilon$，逐點取 arg max 得 (22)；每個 token 至多改變一次，只在 $x$ 與 $\arg\max_{i\ne m}\epsilon_i$ 之間，翻轉點 $\tilde\alpha=\Delta/\sqrt{1+\Delta^2}$（本頁推導）。 | S10 |
| LO11 | 說出 DCD 的設定 | DCD 的教師、損失、步長與輪數怎麼設？ | 依 Alg. 1，教師是上一輪結束時的學生權重，不是 EMA；B.2.2 與 Alg. 2 的對照組改用 EMA 權重當教師（B.2.2 文字說是預訓練得到的 EMA，Alg. 2 偽碼是訓練中更新的 θ_ema），結果較差；學生在較吵的 $z_t$ 上以 KL 逼近教師在 $z_s$（$s=t-\delta$）的輸出 (23)；$N=5$ 輪、每輪 10K 步，$\delta$ 由 1/512 起每輪加倍（Alg. 1、§5.2）。 | S11 |
| LO12 | 讀出主要實驗數字 | Duo 在 LM1B／OWT 的 PPL？蒸餾後幾步達到 base 1024 步的 Gen PPL？與蒸餾 MDLM 比較？ | LM1B 33.7、OWT 25.2（Table 1）；ancestral 16 步 Gen PPL 75.24（base 1024 步 77.69），Greedy-Tail 8 步 69.58 但熵 5.30 較低；步數 ≤ 32 時勝過 MDLM+SDTT，64 步以上落後（Table 7）。 | S12 |

---

## 3. 頁面大綱

單欄 680–720px；圖版 ≤1080px；章節編號 JetBrains Mono（`03 · arg max 的邊際`）；標題用陳述句。

| # | id | 標題 | 內容（依序；符號先定義再使用） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈擴散對偶：對高斯擴散取 arg max，每個時刻的分布就是均勻狀態離散擴散〉 | 副標「這篇論文證明兩種擴散在每個時刻的分布上相連，並據此把高斯擴散的技巧移植到離散文字模型。」三張事實卡；連結 arXiv、`../index.html#paper-393`、`duo-lineage-hub.html`；靜態 SVG：$K=3$、正確詞為第 1 類、$\tilde\alpha=0.9$ 時的 $w=(0.9+0.436\cdot0.4,\ 0.436\cdot(-0.6),\ 0.436\cdot0.8)=(1.074,-0.262,0.349)$ 三根長條 → one-hot (1,0,0) | 1 螢幕 |
| S1 | `discrete` | 01 · 文字擴散有遮罩與均勻兩種雜訊 | one-hot、$\mathcal V$、$K$、Cat、$\mathbf 1$；前向過程；MDM vs USDM；Fig. 5 四欄文字表（AR／MDM／USDM／$P_{DDT}$，最後一欄註「S10 解釋」）；USDM 落後（Table 1 一句）；符號表（S1–S3）。延伸：`duo-discrete-diffusion-masked-and-uniform.html`、`duo-self-correction-and-correctors.html` | 400–500 字 |
| S2 | `usdm` | 02 · 均勻狀態擴散由每個時刻的分布與反向後驗定義 | $t$、$\alpha_t$；「邊際 vs 路徑」定義段（S6 的前提）；(1)，並在 (1) 處定義 $\pi$＝先驗分布（時間 $t=1$ 時的分布；MDM 為 mask token $m$ 的 one-hot，USDM 為 $\mathbf 1/K$）；「邊際隨時間的變化是線性 ODE」一句 + `<details>`（(2)(3)(12)、速率矩陣、正負號）；(4)＝貝氏定理；(5) ancestral；NELBO 一句定義 | 400–500 字 |
| S3 | `argmax` | 03 · 對 one-hot 加高斯雜訊再取 arg max，得到「保留或均勻替換」的分布 | (7)、$\tilde\alpha,\tilde\sigma$、arg max、「推前（pushforward）」定義；(28) 與保留係數 $a$；`<details>` A.1 三步；$\phi$ 印刷錯誤；W1 | 450–550 字 + W1 |
| S4 | `T` | 04 · 保留係數就是運算子 T(α̃)，詞表越大越集中在 α̃≈1 | (11)；性質；W2a（無訓練窗） | 300–400 字 + W2a |
| S5 | `duality` | 05 · arg max 後的分布滿足均勻狀態擴散的 ODE | (33)→(36)→(12)（`<details>` 完整推導）；(13)；unique signatures 一句；「只依賴邊際的量可以改用高斯潛變數計算」 | 300–400 字 |
| S6 | `paths` | 06 · 對偶只在邊際上成立，軌跡不一定是離散擴散 | (37)(38)(39)(40)；論證以 $w_s$ 為條件；以 $z$ 為條件的模擬〔本頁推導〕；W3；callout「本頁用語」 | 400–500 字 + W3 |
| S7 | `theorems` | 07 · 兩個定理：構造出的離散取樣器存在，且其概似不低於對應的高斯取樣器 | Thm 3.1 與 A.4 前提、(44) 未證在單體內；Thm 3.2、(46)；三點限定 | 350–450 字 |
| S8 | `loss` | 08 · Duo 的損失是 UDLM 損失的等價改寫 | NELBO 再述（LO8 答案段）；$\bar x,\bar x_\theta,\kappa_t,i,m$；(49)；(17)；`<details>` 兩情形拆法；即時數值表（6.3）；33.7、Table 3 與重訓 UDLM 的限定；正負號一行；(18)；Greedy-Tail | 400–500 字 + 小表 |
| S9 | `curriculum` | 09 · 課程學習先讓網路看 softmax，再換回 arg max | 課程學習定義；(19)；(20)(21)；C.2；時間窗與 C.7；(11) 不符與推定讀法（W2b）；兩段排程與 §4.1 文字的差異；Fig. 8、Table 4；W5。延伸：`duo-curriculum-softmax-relaxation.html` | 550–650 字 + W2b + W5 |
| S10 | `ddt` | 10 · 已知真值的直線軌跡取 arg max，每個 token 至多翻轉一次 | PF-ODE 3–4 句；(57)→(59)→(22)；翻轉點〔本頁推導〕；A.7；W4 | 400–500 字 + W4 |
| S11 | `dcd` | 11 · 離散一致性蒸餾沿 P_DDT 讓學生模仿教師 | 蒸餾與 consistency distillation 3–4 句（(9)）；(23)；Alg. 1 逐行兩欄對照（k-Diff 頁 `.algo`）；Alg. 2 兩種讀法；與 SDTT 的關係。延伸：`duo-consistency-distillation-to-discrete.html` | 400–500 字 |
| S12 | `results` | 12 · Duo 在均勻狀態模型中困惑度最低；蒸餾後在 32 步以內勝過蒸餾的 MDLM | 首段定義 PPL、Gen PPL、entropy、Greedy-Tail；Table 1 摘錄；510K 步；Table 3；Table 2 摘錄（4 列）；Table 7 Round 5 摘錄（欄：步數、Duo+DCD ancestral Gen PPL、MDLM+SDTT ancestral Gen PPL）；64×／128× 與同取樣器對照 | 450–550 字 + 3 表 |
| S13 | `impl` | 13 · 實作設定與常見錯誤 | 設定表（步驟 12 首條 + 課程排程 + DCD）；9 條常見錯誤 | 350–450 字 |
| S14 | `history` | 14 · 研究脈絡 | 時間線（6.6）+ 100–150 字 | — |
| S15 | `next` | 15 · 限制與延伸閱讀 | §7.1、§7.2；筆記依總覽順序；站內 `dllm-three-corruption-paradigms.html`、`discrete-diffusion-foundations-d3pm.html`、`mdlm-masked-diffusion-objective.html`、`dlm-remasking-self-correction.html` | 200 字 |

Hero 事實卡（每張卡只用卡內自己定義的詞）：
1. 「把一個詞寫成長度 K 的向量：這個詞的位置是 1，其餘是 0。在每個位置加上獨立的高斯雜訊，再取數值最大的位置當作新的詞。新詞的分布是：以某個機率保留原詞，否則在 K 個詞中均勻挑一個。一種逐步把詞換成隨機詞的生成模型（均勻狀態離散擴散）在每個時刻的分布正是這個形式。」
2. 「這個對應保證的是每個時刻的分布相同。若把一條連續加雜訊的過程逐時刻取最大位置，得到的詞序列不一定符合那種模型逐步替換詞的規則。」
3. 「論文據此做了兩件事：訓練時先讓網路看到加雜訊向量的平滑版本，再換回離散的詞；以及把生成所需的步數由 1024 減到 16（品質相當）或 8（輸出的多樣性略降）。困惑度衡量模型對測試文字的預測，越低越好：用這些做法的模型在 LM1B 上訓練 51 萬步時為 35.2，低於前一個同類模型 UDLM 訓練 100 萬步的 36.7。」

桌機左側 sticky 章節目錄；< 1024px 改頂部橫列（沿用 k-Diff 頁）。

---

## 4. 互動元件規格

共同規則（B12、B15、建置教訓）：
- 每個圖版的 `.plate-lead` **第一句寫假設**（例：「假設：K = 3、正確詞為第 1 類、雜訊獨立。」），第二句才寫操作。
- 元件不寫死論文結果：所有讀數由輸入即時算出；動態說明句一律由模板與目前數值組成（例 `非 z_s 的兩類：高斯 ${P[j]} / ${P[k]}，USDM 都是 ${Q[j]}`），不寫「你會看到…」這種只在預設值成立的文字。「應觀察到」只放在本文件與章節正文（以預設值為例，寫明「預設值下」）。
- 標籤（含 $\tilde\alpha$ 等 TeX）在 HTML 中靜態寫好、載入時 KaTeX 渲染一次；互動只更新純數字 `<span>`。SVG 內文字與 `aria-label` 用純文字「alpha~」「sigma~」「tau」，不用組合字元 α̃。
- SVG `<pattern>` 的 id 加圖版前綴（`w1-hatch`、`w2-bandA` …）；pattern 內圖形以 `class` 上色（`.fill-sig{fill:var(--c-sig)}`），深色模式才會跟著換色。
- 長條與曲線的 transition 160ms；`prefers-reduced-motion: reduce` 時所有元件取消 transition、W4 播放改為直接跳到結尾。
- 每個圖版內含 `<noscript>` 與 JS 失敗後備：顯示一行「互動元件需要 JavaScript；預設值的結果見下表」與第 5 節對應的預設值表格（靜態 HTML）。
- 讀數表格「標籤在前、數值在後」（`<th>T</th><td>0.2655</td>`）。
- 滑桿：原生 `<input type="range">`；步長細的滑桿另附「−」「＋」步進鈕（大步），並讓 PageUp／PageDown 走大步。live region 只在 `change` 事件或停止輸入 500ms 後更新，且只播最關鍵的 1–2 個數字。

### 4.0 共用數學（函式名見 6.3）

- PRNG（照抄，與 k-Diff 頁相同；Python 移植在 `dmath.py`，`node check_prng.js` 已確認 seed 3 的前 8 個 `gauss` 值與 Python 逐位相同：1.5492425834, 0.9836970989, −1.6850406004, 0.6077804980, −0.7393693106, 0.5697372037, −1.5649041638, 1.2935265898）：
  ```js
  function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
  function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}
  ```
- `argmaxFirst(v)`：嚴格大於才更新，並列取最小索引。
- `phi(z)=exp(-z*z/2)/Math.sqrt(2*Math.PI)`；`erfcNR(x)`（Numerical Recipes `erfcc`，相對誤差 < 1.2e-7；實測最大 1.04e-7）：
  ```js
  function erfcNR(x){const z=Math.abs(x),t=1/(1+0.5*z);const r=t*Math.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))));return x>=0?r:2-r}
  ```
  `Phi(z)=0.5*erfcNR(-z/Math.SQRT2)`；`logPhi(z)= z>0 ? Math.log1p(-0.5*erfcNR(z/Math.SQRT2)) : Math.log(0.5*erfcNR(-z/Math.SQRT2))`。
- `Top(at,K)`：$\mu=\tilde\alpha/\sqrt{1-\tilde\alpha^2}$；網格 $z_i=-10+0.005i$，$i=0..4400$；`gTab(K)[i]=exp((K-1)*logPhi(z_i))`（每個 $K$ 快取）；$S=h\sum_i{}'[\phi(z_i-\mu)-\phi(z_i)]g_i+[\Phi(12)-\Phi(12-\mu)]$（$\sum'$ 梯形權重，首尾 0.5）；回傳 `clamp(K/(K-1)*S,0,1)`。守門：$\tilde\alpha\le0\to0$；$\tilde\alpha\ge0.999999\to1$。
- `alphaForT(target,K)`：在 $[0,0.999999]$ 二分 60 次，`Top(mid,K)<target` 則 `lo=mid`，回傳 `(lo+hi)/2`。

### W1 · arg max 的邊際：模擬 vs 式 (28)（S3）

- **教的概念**：正確類別機率 $p_x$ 由 (28) 給出；其餘類別相等；分布形如 $\mathrm{Cat}(ax+(1-a)\mathbf 1/K)$。
- **plate-lead**：「假設：正確詞為第 0 類；ε 的 K 個座標獨立、標準常態；樣本數 20000。拖動 α~ 與 K，比較模擬頻率與 (28) 的理論值。」
- **輸入**：$K$ 滑桿 2–10（整數，預設 5）；$\tilde\alpha$ 滑桿 0–0.99（步長 0.01，預設 0.60）；「重新抽樣」（seed 預設 7，每按 +1；改 $K$ 或 $\tilde\alpha$ 不改 seed）。
- **計算**：`r=mulberry32(seed)`；對 $n=1..20000$：`w=[...Array(K)].map((_,k)=>(k===0?at:0)+s*gauss(r))`（$s=\tilde\sigma$，依 $k$ 順序呼叫 `gauss`），`cnt[argmaxFirst(w)]++`；第一個樣本的 $w$ 另存。理論 $p_x=T+(1-T)/K$、$p_o=(1-T)/K$（$T$=`Top(at,K)`，此處文字稱「保留係數 a」）；$SE_k=\sqrt{p_k(1-p_k)/20000}$。
- **顯示**：主圖 $K$ 根長條（MC 頻率；第 0 類用 `w1-hatch` 斜線並標「x（正確詞）」，其餘標 1…K−1）；每根上方理論值以菱形端點的水平短線 + ±2SE 細線。小圖「第一個樣本的 w」（含 0 基線，最大者 `--accent` 外框並標「arg max」）。讀數表：sigma~、alpha~/sigma~、保留係數 a、理論 p_x、理論 p_其他、模擬 p_x、模擬 p_其他平均、最大偏離（以 SE 計）。
- **守門**：$\tilde\alpha\le0.99$；連續 input 以 `requestAnimationFrame` 合併（20000×10 次 gauss ≈ 10ms）。
- **無障礙**：`#w1-live`（polite）在 `change` 時播「理論 p_x = …，模擬 p_x = …」；圖 `role="img"` 的 `aria-label` 同步更新（純文字）。
- **預設值下應觀察到**：$K=5$、$\tilde\alpha=0.6$：理論 0.4124、模擬 0.4142。

### W2 · T(α~) 曲線（W2a 在 S4，W2b 在 S9；同一函式兩個實例）

- **教的概念**：W2a：(11) 的形狀與 $K$ 的影響。W2b：論文訓練窗與 (11) 的關係。
- **plate-lead**：W2a「假設：T 依式 (11) 以數值積分計算；φ 用 e^{−z²/2}/√(2π)。選 K、拖動 alpha~ 讀出 T。」W2b 加「三條帶的定義見圖下說明；詞彙數為 tokenizer 的公開值，論文沒有寫出。」
- **輸入**：$K$ 分段按鈕 {2, 3, 10, 100, 1000, 30522「BERT」, 50257「GPT-2」}（W2a 預設 10，W2b 預設 30522）；alpha~ 游標滑桿 0–1（步長 0.001；「−」「＋」鈕與 PageUp/Down 為 ±0.01；W2a 預設 0.600、W2b 預設 0.970）；「放大 0.8–1.0」開關（W2a 預設關，W2b 預設開）。W2a 沒有帶；W2b 永遠顯示三條帶，無橫軸切換（B1：改為單一 alpha~ 橫軸，見第 8 節）。
- **計算**：401 點 $\tilde\alpha_i=i/400$（`Top`，依 $K$ 快取）；細灰線為 $K=2$ 參考（$K=2$ 時隱藏）。放大時橫軸 [0.8, 1.0]、曲線改用 401 點 $\tilde\alpha_i=0.8+0.2i/400$。
- **W2b 的三條帶（全部是 alpha~ 的區間，畫在同一條 alpha~ 橫軸上）**：

  | 帶 | 圖例文字 | alpha~ 區間 | 填色 |
  | --- | --- | --- | --- |
  | A | 論文所寫：alpha~ = 1 − t，t ∈ [0.03, 0.15] | [0.8500, 0.9700]（與 K 無關） | `w2-bandA` 斜線 |
  | A′ | 推定讀法：alpha~² = 1 − t | [0.9220, 0.9849]（√0.85、√0.97，與 K 無關） | `w2-bandA2` 橫線 |
  | B | 照式 (11) 使 T ∈ [0.05, 0.95] | `[alphaForT(0.05,K), alphaForT(0.95,K)]`（依 K，見 5.1） | `w2-bandB` 點狀 |

  三條帶上下分三層（各高 10px，畫在曲線圖下方的橫條區），直接標字母 A／A′／B；放大關閉時三條帶照樣畫在 [0,1] 座標。讀數表另列「T 在帶 A 兩端」「T 在帶 A′ 兩端」「帶 B 端點」。
- **讀數**（標籤在前）：alpha~、sigma~、alpha~/sigma~、T、P(z=x)=T+(1−T)/K。讀數框固定在圖版右上角，不跟隨指標。
- **守門**：$K$ 改變才重建 `gTab`（4401 次 erfc）；曲線 401×4401 次 exp ≈ 15ms；端點 $\tilde\alpha=1$ 設 1；`alphaForT` 結果依 $K$ 快取。
- **無障礙**：live region 只在 `change` 時播「T = …，P(z=x) = …」；帶以紋理 + 字母區分。
- **預設值下應觀察到**（只寫在正文）：S4（K=10、0.6）T=0.1838；S9（K=30522、0.97）T=0.4523，帶 A 與帶 B 只部分重疊，帶 A′ 與帶 B 接近。

### W3 · 給定 w_s 時的下一步：式 (39) vs 式 (38)（S6）

- **教的概念**：給定整個 $w_s$，arg max 後的下一步機率依賴 $w_s$ 的每個座標；USDM 核只看 $z_s$。
- **plate-lead**：「假設：K = 3；條件是整個向量 w_s（不只是 z_s）；高斯轉移依 (37)，離散轉移依 (38)，α_{t|s} = T(α~_t)/T(α~_s)。」
- **輸入**：$w_s$ 三個座標滑桿 $[-2,3]$（步長 0.05，預設 1.20、0.60、0.00）；預設鈕「(1.2, 0.6, 0)」「(1.2, 0, 0.6)」「非最大座標相等 (1.2, 0.3, 0.3)」；alpha~_s 滑桿 [0.50, 0.99]（步長 0.01，預設 0.90）；alpha~_t 滑桿 [0.05, alpha~_s − 0.05]（步長 0.01，預設 0.60；alpha~_s 改變時夾住）。
- **計算**：$a=\tilde\alpha_t/\tilde\alpha_s$、$s=\sqrt{1-a^2}$、$c_j=a(w_s)_j$；$P_i=\sum_q{}'\,\phi(u_q)\prod_{j\ne i}\Phi(u_q+(c_i-c_j)/s)\cdot0.01$，$u_q=-9+0.01q$，$q=0..1800$（梯形）。$z_s$=`argmaxFirst(w_s)`（並列時讀數加「arg max 並列，取索引較小者」）；$\alpha_{t|s}$=`Top(att,3)/Top(ats,3)`；$Q_i=\alpha_{t|s}[i=z_s]+(1-\alpha_{t|s})/3$；給定 $w_s$ 的總變異距離 $\frac12\sum|P_i-Q_i|$。
- **顯示**：三組並列長條（左 $P$「高斯軌跡取 arg max（給定 w_s）」實心；右 $Q$「USDM 轉移（給定 z_s）」`w3-hatch`），長條上標 4 位數字；$z_s$ 類別下方標「z_s」；上方小圖畫 $w_s$。讀數表：alpha~_{t|s}、α_{t|s}、ΣP、TV（給定 w_s）。動態說明句（模板）：「非 z_s 的兩類：高斯 {P_j} 與 {P_k}，USDM 都是 {Q_j}。」
- **守門**：$\tilde\alpha_t\le\tilde\alpha_s-0.05$ 保證 $a\lt1$；每次 3×1801×2 次 `Phi`，同步。
- **無障礙**：live region 在 `change` 時播「TV = …」；長條紋理 + 數字。
- **預設值下應觀察到**（正文）：(1.2, 0.6, 0) 與 (1.2, 0, 0.6) 的 $z_s$、$Q$ 相同，$P$ 的第 2、3 類互換；非最大座標相等時 $P$ 兩類相等，但仍不等於 $Q$。正文另附 `check_markov.py` 的 $z$ 條件數字（步驟 6）。

### W4 · 反向取樣中 token 的變動：USDM vs P_DDT（S10）

- **教的概念**：USDM ancestral（(4)(5)）中 token 可多次改變（可修改性）；$P_{DDT}$（(22)）中每個 token 至多翻轉一次。**不**展示自我修正（F7）：已知真值的 denoiser 不會犯錯。
- **plate-lead**：「假設：兩欄都使用已知真值的 denoiser（x_θ = x），所以 USDM 最後一定得到目標句；真實模型的 x_θ 會出錯。USDM 用 α_t = 1 − t，P_DDT 用 alpha~_t = 1 − t；詞表 17 個詞。」
- **常數**：`VOCAB=['Really','Through','Little','Good','Always','Number','Building','Question','Other','Think','Date','Want','Dual','Diffusion','For','Discrete','Data']`（$K=17$，取自 Fig. 5）；`TARGET`＝`Dual Diffusion For Discrete Data` 的索引 [12, 13, 14, 15, 16]（$L=5$）。
- **輸入**：步數 $T_{\text{steps}}$ 分段按鈕 {4, 8, 16, 32, 64}（預設 16）；「重新抽樣」（seed 預設 7，+1）；時間滑桿 $n\in[0,T_{\text{steps}}]$（第 $n$ 步、$t=1-n/T_{\text{steps}}$，**預設 0**＝雜訊起點）；「播放／暫停」切換鈕（每 250ms 前進一步，到 $T_{\text{steps}}$ 停）；「到結尾」鈕。改 $T_{\text{steps}}$ 或 seed 時重算並把 $n$ 設回 0、停止播放。
- **計算**（`hist` 為長度 $T_{\text{steps}}+1$ 的陣列，`hist[n]` 是 5 個詞索引，對應 $t=1-n/T_{\text{steps}}$）：
  - USDM：`r=mulberry32(seed)`；`hist[0][l]=Math.min(K-1,Math.floor(r()*K))`，$l=0..4$。對 $n=1..T_{\text{steps}}$：$\alpha_t=(n-1)/T_{\text{steps}}$、$\alpha_s=n/T_{\text{steps}}$（$s\lt t$），對每個 $l$ 依 (4) 算 $K$ 維機率 $p$，取 `u=r()`，累積和第一個 `u<c` 的索引（皆不滿足取 $K-1$）。$\alpha_s\ge1/T_{\text{steps}}\gt0$，除法安全。
  - DDT（`hist[0]` 為 $a=0$ 時對純雜訊取 arg max；ε 各座標同分布，所以起點在 $K$ 個詞上均勻，與 USDM 的均勻起點一致）：`r2=mulberry32(seed+1000)`；`eps[l][k]=gauss(r2)`（$l$ 外 $k$ 內）；`hist[n][l]=argmaxFirst(k ↦ (k===TARGET[l]?a:0)+Math.sqrt(Math.max(0,1-a*a))*eps[l][k])`，$a=n/T_{\text{steps}}$。翻轉點：$j^*$＝$\epsilon$ 中除目標外最大者，$\Delta=\epsilon_{j^*}-\epsilon_{x}$，$\Delta\gt0$ 時 $\tilde\alpha^*=\Delta/\sqrt{1+\Delta^2}$，否則「不翻轉」。
  - 「改變次數」＝ $n\in1..T_{\text{steps}}$ 中 `hist[n][l]!==hist[n-1][l]` 的個數。
- **顯示**：USDM 與 $P_{DDT}$ 兩個面板放在**同一個**橫向捲動容器（窄螢幕一起捲動，右側漸層提示）；每個面板 5 欄：上方第 $n$ 步的詞 pill（與目標相同者 `--c-sig` 實心底並加 ✓，不同者外框），下方「變動時間線」（$T_{\text{steps}}$ 格；改變處 USDM 實心方塊、DDT 實心三角；目前步以細線標示），欄底「改變 {k} 次」；DDT 欄底另列 alpha~*。面板右側「總改變次數」。
- **無障礙**：播放中不更新 live region；暫停、到結尾或滑桿 `change` 時播「第 n 步：{5 個詞}；USDM 已改變 a 次，P_DDT 已改變 b 次」；標記用形狀區分。
- **預設值下應觀察到**（正文，seed 7、16 步）：USDM 各欄 2、3、4、3、3 次，DDT 各欄 0、1、1、1、1 次；DDT 的翻轉點與 $T_{\text{steps}}$ 無關。正文寫明：USDM 的改變次數隨步數增加（64 步時 3、8、8、3、3）。

### W5 · 課程的輸入：softmax(w/τ)（S9）

- **教的概念**：(20)：$\tau\to0$ 時輸入趨近 one-hot；輸入軟硬取決於「最大兩個座標的差除以 τ」；詞表大時，最大的雜訊座標變大，正確詞與第二名的差變小，甚至 arg max 不是正確詞。
- **plate-lead**：「假設：一個 token；正確詞為第 0 類；ε 由固定 seed 產生，K 個座標獨立。結果只代表這一組 ε。」
- **輸入**：$K$ 分段按鈕 {8, 1000, 30522}（預設 8）；alpha~ 滑桿 [0.50, 0.99]（步長 0.01，預設 0.90）；log10 τ 滑桿 [−3, 1]（步長 0.05，預設 −1）；chip「τ = 0.001」；「重新抽樣 ε」（seed 預設 3，+1）。
- **計算**：`r=mulberry32(seed)`；`w[k]=(k===0?at:0)+s*gauss(r)`，$k=0..K-1$ 依序（$K=30522$ 時約 3ms）；$p=\mathrm{softmax}(w/\tau)$ 以減去 $\max w$ 計算；$H=-\sum p\log p$（$p=0$ 略過；$|H|\lt10^{-12}$ 顯示 0）；$j^*$＝除第 0 類外 $w$ 最大者；差距 $g=w_0-w_{j^*}$；正規化差距 $g/\tau$；$1-p_{\max}\lt10^{-12}$ 時顯示「< 1e-12」。
- **顯示**：左圖：$K\le8$ 畫全部 $w$；$K\gt8$ 只畫第 0 類 + $w$ 最大的 7 個其他座標（依 $w$ 排序，軸標「第 j 類」），arg max 加外框。右圖：同一批座標的 $p$（縱軸 0–1），並以虛線外框畫 one-hot。讀數表：τ、arg max 是否為正確詞、g、g/τ、p_x、p_max、H（nats；上限 log K）。
- **無障礙**：live region 在 `change` 時播「p_x = …，g/τ = …」。
- **預設值下應觀察到**（正文）：K=8、alpha~=0.9：τ=1 時 p_x 0.3891，τ=0.1 時 0.9999；K=30522、alpha~=0.9 時 arg max 不是正確詞（g = −0.3832）；alpha~=0.97 時 g = 0.2543，τ=0.001 時 g/τ ≈ 254，輸入為 one-hot。正文：論文沒有把鬆弛與時間窗兩個因素分開消融，這個元件不能回答哪個因素造成變異下降。

---

## 5. 驗收數值表

腳本與頁面演算法相同；★ 為 `selfTest()` 斷言項。容許值針對跨引擎 `Math.exp/log/cos` 末位差。

### 5.1 T(α~)（`check_T.py`；容許 |Δ| ≤ max(1e-8, 1e-5×值)）

| K | α~=0.3 | 0.5 | 0.6 | 0.8 | 0.85 | 0.9 | 0.95 | 0.97 | 0.985 | 0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.175978 | 0.316909 | ★0.404117 | 0.654221 | 0.746116 | 0.855708 | 0.968550 | 0.995218 | 0.999946 | 0.999999 |
| 3 | 0.138707 | 0.259622 | 0.338977 | 0.587381 | 0.687384 | 0.814193 | 0.956461 | 0.993097 | 0.999919 | 0.999999 |
| 5 | 0.100851 | 0.197609 | ★0.265547 | 0.501812 | 0.607935 | 0.753363 | 0.936184 | 0.989222 | 0.999867 | 0.999998 |
| 10 | 0.063484 | 0.131753 | ★0.183770 | 0.390562 | 0.497175 | 0.659218 | 0.898303 | 0.980976 | 0.999742 | 0.999997 |
| 100 | 0.011252 | 0.027471 | 0.042776 | 0.131880 | 0.198927 | 0.338400 | 0.688532 | 0.912357 | 0.998013 | 0.999968 |
| 1000 | 0.001656 | 0.004626 | 0.007918 | 0.034157 | 0.060553 | 0.132233 | 0.426718 | 0.760837 | 0.989397 | 0.999758 |
| 30522 | 0.000082 | 0.000270 | 0.000523 | 0.003514 | ★0.007730 | 0.023729 | 0.150483 | ★0.452299 | 0.936404 | 0.997132 |
| 50257 | 0.000052 | 0.000177 | 0.000348 | 0.002478 | 0.005612 | 0.018046 | 0.125794 | 0.409002 | 0.922118 | 0.996110 |

- K=2 封閉式 $\mathrm{erf}(\tilde\alpha/(2\tilde\sigma))$：α~ = 0.1 → 0.056655；0.3 → 0.175978；0.6 → 0.404117；0.9 → 0.855708（★ 差 < 1e-6）。`Top(0,K)=0`（★）。
- W2b 帶與端點的 T（`alphaForT` 容許 ±1e-4；T 容許 ±1e-4）：

| K | 帶 B 下端 | 帶 B 上端 | T(0.85) | T(0.97) | T(0.9220) | T(0.9849) |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.0883 | 0.9407 | 0.7461 | 0.9952 | 0.9077 | 0.9999 |
| 3 | 0.1153 | 0.9474 | 0.6874 | 0.9931 | 0.8779 | 0.9999 |
| 10 | 0.2473 | 0.9611 | 0.4972 | 0.9810 | 0.7560 | 0.9997 |
| 100 | 0.6331 | 0.9741 | 0.1989 | 0.9124 | 0.4518 | 0.9979 |
| 1000 | 0.8347 | 0.9805 | 0.0606 | 0.7608 | 0.2078 | 0.9888 |
| 30522 | ★0.9238 | ★0.9856 | 0.0077 | 0.4523 | 0.0469 | 0.9337 |
| 50257 | 0.9299 | 0.9861 | 0.0056 | 0.4090 | 0.0368 | 0.9190 |

### 5.2 W1（`check_w1.py`；N=20000、seed 7；容許 ±0.0005，或 |模擬−理論| ≤ 4SE）

| K | α~ | 保留係數 a | 理論 p_x | 理論 p_其他 | 模擬 p_x | 模擬 p_其他平均 | 模擬 p_其他 最小／最大 | (模擬−理論)/SE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.6 | 0.2655 | 0.4124 | 0.1469 | ★0.4142 | 0.1465 | 0.1435／0.1527 | 0.49 |
| 5 | 0.0 | 0.0000 | 0.2000 | 0.2000 | 0.2001 | 0.2000 | 0.1948／0.2048 | 0.02 |
| 5 | 0.9 | 0.7534 | 0.8027 | 0.0493 | 0.7993 | 0.0502 | 0.0492／0.0516 | −1.20 |
| 2 | 0.6 | 0.4041 | 0.7021 | 0.2979 | 0.6997 | 0.3003 | — | −0.73 |
| 10 | 0.8 | 0.3906 | 0.4515 | 0.0609 | 0.4581 | 0.0602 | 0.0583／0.0636 | 1.87 |
| 3 | 0.3 | 0.1387 | 0.4258 | 0.2871 | 0.4341 | 0.2829 | 0.2803／0.2856 | 2.39 |

無偏檢查（K=3、α~=0.3、N=200000）：seed 7、8、9 的模擬 p_x 為 0.4266、0.4269、0.4256（理論 0.4258）。

### 5.3 W3（`check_w3.py`；容許 ±1e-4）

| w_s | α~_s、α~_t | α~_{t|s} | P（高斯，給定 w_s） | α_{t|s} | Q（USDM） | TV |
| --- | --- | --- | --- | --- | --- | --- |
| (1.2, 0.6, 0.0) | 0.90、0.60 | 0.6667 | ★(0.5638, 0.2963, 0.1399) | ★0.4163 | ★(0.6109, 0.1946, 0.1946) | ★0.1018 |
| (1.2, 0.0, 0.6) | 0.90、0.60 | 0.6667 | (0.5638, 0.1399, 0.2963) | 0.4163 | (0.6109, 0.1946, 0.1946) | 0.1018 |
| (1.2, 0.3, 0.3) | 0.90、0.60 | 0.6667 | ★(0.5760, 0.2120, 0.2120) | 0.4163 | (0.6109, 0.1946, 0.1946) | 0.0349 |
| (1.2, 0.6, 0.0) | 0.95、0.30 | 0.3158 | (0.4197, 0.3279, 0.2524) | 0.1450 | (0.4300, 0.2850, 0.2850) | 0.0429 |
| (0.5, 0.4, −1.0) | 0.90、0.60 | 0.6667 | (0.4868, 0.4387, 0.0746) | 0.4163 | (0.6109, 0.1946, 0.1946) | 0.2441 |
| (2.0, −0.5, 1.0) | 0.90、0.60 | 0.6667 | (0.7172, 0.0298, 0.2530) | 0.4163 | (0.6109, 0.1946, 0.1946) | 0.1647 |

所有 ΣP = 1.000000（★ 容許 1e-6）。`check_markov.py`（只用於正文）：見步驟 6 的四個機率（MC 標準誤約 0.001–0.003）。

### 5.4 W4（`check_w4.py`；整數，預期完全相同；若跨引擎不同，以下列不變式為準）

| T_steps | seed | USDM 改變次數 | USDM 總數 | DDT 改變次數 | DDT 總數 | USDM 第 0 列 | DDT 第 0 列 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 7 | [2, 3, 2, 2, 1] | 10 | [0, 1, 1, 1, 1] | 4 | Really Through Data Want Other | Dual Good Dual Question Always |
| 8 | 7 | [2, 3, 3, 3, 3] | 14 | [0, 1, 1, 1, 1] | 4 | 同上 | 同上 |
| 16 | 7 | ★[2, 3, 4, 3, 3] | 15 | ★[0, 1, 1, 1, 1] | 4 | 同上 | 同上 |
| 64 | 7 | [3, 8, 8, 3, 3] | 25 | [0, 1, 1, 1, 1] | 4 | 同上 | 同上 |
| 16 | 8 | [3, 5, 3, 5, 3] | 19 | [1, 1, 1, 1, 1] | 5 | Little Date Number Want Think | Data Data Through Data Question |

- seed 7 的 DDT 翻轉點 alpha~*：Dual「不翻轉」（Δ = −0.4574）；Diffusion 0.7845（自 Good，Δ = 1.2652）；For 0.1217（自 Dual，Δ = 0.1226）；Discrete ★0.9451（自 Question，Δ = 2.8917）；Data 0.3204（自 Always，Δ = 0.3382）（容許 ±1e-4）。
- 不變式（★）：USDM `hist[T_steps]` 等於 TARGET；DDT 每欄改變次數 ≤ 1；(4) 每列和為 1（誤差 < 1e-12）。

### 5.5 W5（`check_w5.py`；seed 3；容許 ±1e-4）

K=8 的 ε：1.5492、0.9837、−1.6850、0.6078、−0.7394、0.5697、−1.5649、1.2935。

| K | α~ | arg max | g = w_0 − max 其他 | τ=10：p_x、H | τ=1：p_x、H | τ=0.1：p_x、p_max、H | τ=0.01：p_x | τ=0.001：p_x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.90 | 0 | 1.0115 | 0.1435、2.0769 | ★0.3891、1.7951 | ★0.9999、0.9999、0.0006 | 1.0000 | 1.0000 |
| 8 | 0.97 | 0 | 1.0322 | 0.1407、2.0780 | 0.3503、1.8890 | 0.9999、0.9999、0.0007 | 1.0000 | 1.0000 |
| 1000 | 0.90 | 0 | 0.1004 | 0.0012、6.9068 | 0.0043、6.8130 | 0.5416、0.5416、1.3690 | 0.9999 | 1.0000 |
| 1000 | 0.97 | 0 | 0.5240 | 0.0011、6.9075 | 0.0037、6.8771 | 0.9729、0.9729、0.2324 | 1.0000 | 1.0000 |
| 30522 | 0.90 | 12147 | ★−0.3832 | 0.0000、10.3253 | 0.0001、10.2312 | 0.0128、0.5890、2.0479 | 0.0000 | 0.0000 |
| 30522 | 0.97 | 0 | ★0.2543 | 0.0000、10.3259 | 0.0001、10.2966 | 0.5347、0.5347、4.0056 | 1.0000 | 1.0000 |
| 30522 | 0.99 | 0 | 0.5747 | 0.0000、10.3261 | 0.0001、10.3162 | 0.6824、0.6824、3.5867 | 1.0000 | 1.0000 |

arg max 是否正確只對這一組 ε 成立。

### 5.6 S8 小表 f (49) 與 f_Duo (17)（`check_loss.py`；K=7、α=0.37、α′=−1；x_θ＝mulberry32(5) 前 7 個 `r()` 除以總和＝[0.1863, 0.2087, 0.0593, 0.1683, 0.0230, 0.1599, 0.1945]；m = 2；容許 1e-9）

| 情形 | f (49) | f_Duo (17) | 差 |
| --- | --- | --- | --- |
| z_t = x（i = 2） | ★−1.7499022162 | −1.7499022162 | 2.2e-16 |
| z_t ≠ x（i = 4） | −1.6870330826 | −1.6870330826 | 4.4e-16 |
| z_t ≠ x（i = 0） | −2.2488799387 | −2.2488799387 | 0 |

$x_\theta=x$ 時兩式皆 0（★）。表下註「依 (49)(17) 原式、α′ = −1；以 −dα/dt 代入則全部變號」。

---

## 6. 實作計畫

### 6.1 外殼（重用）

複製 `tutorials/prediction-targets-k-diff.html` 的外殼：`<head>`（字型、KaTeX 0.16.9 defer、auto-render）、CSS tokens 與深色三段式、appbar 與主題切換（`pm-theme`）、`initToc`、`.hero`、`.ch`、`.plate*`、callout、`details.check`、表格橫捲容器、`.algo`、`#pm-bar`、時間線 rail（`renderTimeline`）、`announce()`、SVG 輔助（`el/tx/lp/frame/arrow/widthOf`）、`mulberry32`、`gauss`、`renderMath`、`?test`→`selfTest()`。以下只列新增或改動。

### 6.2 新增樣式與文案常數

- 圖表色改名：`--c-sig`（正確詞／訊號；#3F6E8C／深色 #8DB4D3）、`--c-noi`（雜訊／其他詞；#6B7A3A／#A9BC7C）；只用於圖中資料。
- `.tok`（W4 詞 pill，等寬 13px，`min-width` 依「Diffusion」）；`.tl-strip`（高 10px）；`.scroll-x`（橫捲 + 右側 `mask-image` 漸層提示）用於 W4 兩面板共用容器、Table 2、Table 7 摘錄、S1 四欄表。
- `<title>擴散對偶 Duo</title>`；appbar「No. 393 · The Diffusion Duality」；description「圖解 Sahoo 等人 ICML 2025 論文：高斯擴散取 arg max 後每個時刻的分布是均勻狀態離散擴散，並據此做課程學習與離散一致性蒸餾。」
- 連結：`../index.html#paper-393`；`https://arxiv.org/abs/2506.10892`；`https://s-sahoo.com/duo`；筆記 `<slug>.html`。

### 6.3 新增 JS（單一 IIFE）

- 數學：`argmaxFirst`、`erfcNR`、`Phi`、`logPhi`、`phi`、`ZG`、`gTab(K)`、`Top(at,K)`、`alphaForT(target,K)`（4.0 照抄）。
- W1：`initW1()`、`w1Sample(K,at,seed)`→`{freq, first}`、`renderW1()`。
- W2：`initW2(rootId,{bands,defaultK,defaultAt,zoom})`（W2a、W2b 兩個實例）、`curveT(K,zoom)`（快取）、`renderW2(inst)`。
- W3：`initW3()`、`pushK(ws,ats,att)`、`usdmKernel(ws,ats,att)`→`{Q,a,zs,tie}`、`renderW3()`。
- W4：`VOCAB`、`TARGET`、`posteriorUS(zt,x,at,as,K)`、`runUSDM(Tst,seed)`、`runDDT(Tst,seed)`→`{hist,flips}`、`changes(hist)`、`initW4()`、`renderW4()`；播放用 `setInterval(250)`，暫停清除。
- W5：`initW5()`、`renderW5()`（softmax 內聯）。
- S8：`f49`、`f17`、`fillLossTable()`（載入時填表；失敗保留靜態數字）。
- `selfTest()`：所有 ★ 項，加 `document.documentElement.scrollWidth <= innerWidth`（在當前視窗寬執行；建置檢查時以 390px 與 1440px 各跑一次 `?test`）。輸出 `selfTest: N passed, M failed`。

### 6.4 數學排版

沿用 k-Diff 頁（`.katex-display` 橫捲、SVG 用純文字、動態插入後才 `renderMathInElement`，互動更新不再呼叫）。< 600px 以 `aligned` 拆行的式子：(4)、(11)、(12)、(13)、(17)、(21)、(28)、(39)、(49)。TeX 中 `<`、`>` 寫 `\lt`、`\gt`。

### 6.5 引用與標籤

論文數字標出處；元件圖版底部一行「讀數由本頁依 (n) 即時計算」；〔本頁推導〕用 pill「本頁推導」+ 一行依據；推定用 pill「推定」；論文內部不一致用 callout「論文原文與式 (11) 不一致」，照錄兩邊數字。區分論文結果／作者解釋（「作者認為」）／本頁推導。英文術語首見附中文：carry-over（揭露後沿用）、teacher／student（教師／學生模型）、curriculum learning（課程學習）。

### 6.6 研究脈絡（notes/timeline.json，15 筆，原樣貼入 `#timeline-data`）

| 類別 | arXiv ID | 卡片連到的筆記 |
| --- | --- | --- |
| 高斯擴散 | 1503.03585、2006.11239、2011.13456、2107.00630 | `duo-gaussian-diffusion-foundations.html` |
| 離散擴散 | 2107.03006、2205.14987、2310.16834、2406.07524、2412.10193 | `duo-discrete-diffusion-masked-and-uniform.html` |
| 鬆弛 | 1611.01144、1611.00712 | `duo-curriculum-softmax-relaxation.html` |
| 蒸餾 | 2202.00512、2303.01469、2410.21035 | `duo-consistency-distillation-to-discrete.html` |
| 本文 | 2506.10892 | `duo-argmax-duality-theory.html` |

呈現沿用 k-Diff 頁。S15 延伸閱讀順序：`duo-lineage-hub`、`duo-gaussian-diffusion-foundations`、`duo-discrete-diffusion-masked-and-uniform`、`duo-argmax-duality-theory`、`duo-curriculum-softmax-relaxation`、`duo-consistency-distillation-to-discrete`、`duo-self-correction-and-correctors`，各附 frontmatter `title`。

### 6.7 文案規則

繁體中文、台灣用語、字面學術語氣；不用比喻、反問、宣傳形容詞。不用「神諭」，寫「已知真值的 denoiser（$x_\theta=x$）」。「帶」首見寫「區間帶（圖中一段 alpha~ 範圍）」。「對偶」一律搭配「在邊際上」或「每個時刻的分布」。

### 6.8 建置後自我檢查

截圖 390／768／1024／1440 + 深色、鍵盤 focus、reduced-motion；依 AGENTS.md 步驟 6-3 逐項評分修正；`?test` 在 390 與 1440 全數通過；逐條核對 12 個 LO 的答案段落（`data-lo`）與 1.5 的六項。

---

## 7. 風險與未決事項

| # | 項目 | 處理 |
| --- | --- | --- |
| R1 | 訓練窗與 (11) 不符；「α~² = 1 − t」只是推定且只對 K=30522 較吻合 | W2b 畫 A、A′、B 三帶；正文照錄、標推定、不判定筆誤；LO9 答案只寫「不成立」 |
| R2 | φ 印刷錯誤 | 全頁用 e^{−z²/2}；S3 一行；常見錯誤 1 |
| R3 | α′ 正負號慣例不一致 | S2 `<details>`、S8 各一行；S8 小表照原式為負值並註明 |
| R4 | Rao-Blackwellized 說法；Table 3 最後一列是重訓 UDLM | 照錄論文說法，加「推導上是等價改寫」與數值表 |
| R5 | Thm 3.1 前提、(44) 未證在單體內；Thm 3.2 的 KL 為無限大 | S7 寫明前提與「動機性論證」 |
| R6 | 軌跡結論：A.3 的論證以 w_s 為條件 | 用「不一定」；以 z 為條件的模擬標〔本頁推導〕；W3 的 TV 標「給定 w_s」 |
| R7 | 宣稱數字的限定（2×、128×、below 2、勝 MDLM） | 1.3、1.5、S12 照數字寫並加限定 |
| R8 | 未寫明的設定（後 500K 步 t 區間、T 快取積分法、μ、蒸餾 batch/lr、確切 K、梯度變異估計法） | S13 表格標「論文未寫明」 |
| R9 | W4 用已知真值的 denoiser | plate-lead 第一句寫假設；正文說明只展示可修改性 |
| R10 | W1 模擬波動 | 讀數顯示偏離（SE）；驗收容許 4SE |
| R11 | 跨引擎浮點末位差 | 連續量用容許值；W4 整數以不變式為準 |
| R12 | 初學者門檻（推前、ODE、NELBO、PF-ODE、蒸餾） | S2 定義邊際／路徑與 NELBO；S3 定義推前；S10、S11 各以 3–4 句教 PF-ODE 與蒸餾；完整推導收 `<details>` |
| R13 | 相對連結與筆記更新 | 以上線位置撰寫；建置時重讀 `notes/`，未列 ID 歸「其他」並回報 |

---

## 8. 審查回覆（round 1）

全部 P0、P1 已處理；P2 除下列外已處理。

| 項目 | 處理 |
| --- | --- |
| F1 | 步驟 6、LO6、hero 卡 2、W3 改為「不一定」，新增 `check_markov.py`（z 條件），TV 標「給定 w_s」 |
| F2 | 1.3、hero 卡 3、1.5-4 改寫 |
| F3 | 標推定，加 K=50257 數字 |
| F4、F5 | 步驟 7 加前提與 KL = +∞ |
| F6 | 步驟 8、1.5-3 加重訓 UDLM 限定 |
| F7 | W4 改為「可修改性」，1.5-6 |
| F8 | 步驟 11 列兩種讀法 |
| F9 | 步驟 9 排程兩處描述並列 |
| F10 | 改為「20000 組全部 ≤ 0」，`check_loss.py` 加 seed |
| F11 | 步驟 12 加 base Greedy-Tail 71.72／5.22 |
| F12 | 步驟 12 標 † 與 ‡ |
| F13 | 步驟 13-4 |
| B1 | W2b 改為單一 alpha~ 橫軸、三條帶（A、A′、B）的端點表；取消 α~² 橫軸 |
| B2 | W2 拆成 W2a（S4，無帶）與 W2b（S9，有帶） |
| B3 | S3 用保留係數 a，S4 才命名 T |
| B4 | S2 加邊際／路徑與 NELBO 定義，速率矩陣移入 `<details>`；LO2、LO8 |
| B5 | S4/S5、S10/S11 拆章；S10、S11 先教 PF-ODE 與蒸餾 |
| B6 | 符號表只列 S1–S3 |
| B7 | S7、S12 標題加限定 |
| B8 | S12 首段定義指標 |
| B9 | 文案替換（6.5、6.7） |
| B10 | 刪除無依據的句子；W5 加 K 選項與 g/τ |
| B11 | pattern id 加前綴、class 上色 |
| B12 | 靜態標籤、只更新數字、SVG/aria 用純文字 |
| B13 | live region 精簡、步進鈕與 PageUp/Down、W4 暫停鍵、W1 live region |
| B14 | 6.4 補齊 aligned 清單；W4 兩面板共用捲動容器；`.scroll-x` 漸層 |
| B15 | W4 `hist` 形狀、改變次數定義、n 預設 0 與重置；全元件 reduced-motion；noscript 後備；selfTest 視窗寬；5.1 改相對容許值；跨引擎改容許值與不變式；驗證腳本可 `python3 -I` 單獨執行 |

**未採納（部分）**
- B10 建議以 σ√(2 ln K) 估計大 K 的最大雜訊座標。改為直接抽 K 個樣本（K = 30522 約 3ms），不需近似，結果也能在 5.5 精確驗收。
- B1 建議列出 {軸, 帶 A, 帶 B, zoom} 四種組合。改為取消 α~² 橫軸：三條帶都定義為 alpha~ 區間並畫在同一軸上，組合只剩「放大開／關」，兩者的座標都已在 W2 規格中寫明。

## 9. readiness 回覆

- 阻擋 1：S2 在 (1) 處定義 $\pi$。阻擋 2：LO11 改為「依 Alg. 1」並加 B.2.2／Alg. 2 的 EMA 讀法。
- 非阻擋 1（Table 7 欄名）、2（hero SVG 數值）、4（W4 DDT 均勻起點）、7（LO9「依 §5.1」）已寫入；3、5、6：S14／S15 內容依 6.6 清單；程式中不以數值使用 $Q_t$；`#paper-393` 由主站產生的 index 提供（其他教學頁同樣格式）。

## 10. 建置紀錄（build round）

- 原始碼：`build/src/{head.html,style.css,body.html,app.js}`，以 `build/assemble.py` 組成 `build/index.html`（內嵌 timeline.json，無 file:// 路徑）。`build/localtest.sh` 以本機 KaTeX 跑 `?test`；`build/caps.sh` 截取單一圖版；`build/m390.html`、`m390t.html` 在 390px iframe 內截圖與跑 `?test`。
- 結果：`?test` 58 passed、0 failed（預設視窗與 390px iframe）；impeccable 0 anti-patterns（只剩 numbered-section-labels advisory）；LO 1–12 的答案段落（`data-lo`）關鍵詞與數字皆出現在 `accept/page.txt`。
- 與設計的差異：W3 未畫 $w_s$ 小圖（三個滑桿的數值即為 $w_s$）；`selfTest()` 改在 `load` 且 KaTeX 渲染後執行（否則未渲染的 TeX 會造成 390px 的寬度誤判）；寬式子 (11)(16)(17)(21)(22)(23)(28)(37)(39)(49) 在所有寬度都用 `aligned` 拆行；(39) 以 $\varsigma=\sqrt{1-\tilde\alpha_{t|s}^2}$ 簡寫；圖例斜線改用單一斜線樣本（impeccable：repeating-stripes-gradient）；S8 數值表設 `min-width:560px` 橫捲。
- 未驗證：以 cdnjs KaTeX 與 Google Fonts 的原始 `index.html?test` 在本機無網路時無法於時限內完成；以本機 KaTeX 替換後通過。
