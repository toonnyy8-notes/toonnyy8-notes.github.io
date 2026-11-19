# 設計文件：〈Elucidating the Exposure Bias in Diffusion Models〉互動教學

- 論文：Mang Ning、Mingxiao Li、Jianlin Su、Albert Ali Salah、Itir Onal Ertugrul，arXiv 2308.15321（v6，ICLR 2024）。資料庫編號 No. 406。
- 原文：`tut/406/src/paper.txt`（PDF 同目錄；Fig. 1–5 的讀圖值取自 PDF 第 4–7 頁）。本文件中的「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」「Alg. n」皆指論文原文編號。`tut/` = `scripts/tutorial_src/`。
- 產出：`tut/406/build/index.html`（單一 HTML 檔）。上線位置：`tutorials/exposure-bias-epsilon-scaling.html`（slug `exposure-bias-epsilon-scaling`）。站內相對連結一律以上線位置為準。
- 驗算腳本（非頁面程式）：`tut/406/verify/check.py`（C1、C2 的論文模型、A.10、表格數字、Monte Carlo 交叉檢查）、`floor.py`（C2 的完美網路模式、λ 在高斯玩具中的效果）、`eq10.py`（C3）。輸出存於同目錄 `*_out.txt`。頁面演算法與腳本一致（4.0）。
- 研究筆記：`tut/406/notes/`（總覽 `exposure-bias-elucidation-overview` + 6 篇主題筆記 + `timeline.json`，13 筆）。
- 範圍：本頁只談 No. 406。DDPM-IP（No. 405）只在 No. 406 自己比較它的地方出現（Sec. 2、Sec. 3.4、Table 3）。
- 標記慣例：**〔本頁推導〕** = 本設計自行推導、論文沒有寫出；**推定** = 為了貼近論文數字而校準、論文沒有寫明的值（含讀圖值）；「作者認為」= 論文作者的解釋而非實驗結果。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

擴散模型訓練時，網路 $\epsilon_\theta$ 的輸入是由真實資料加噪得到的 $x_t$；取樣時，輸入是網路自己前幾步產生的 $\hat x_t$。兩者的差異稱為**曝光偏差（exposure bias）**，一詞沿用自序列生成（Ranzato et al., arXiv 2015、論文引為 2016；Schmidt, 2019），在擴散模型中由 Ning et al.（2023）首先指出。本文沿用 Analytic-DPM 的作法，把網路對 $x_0$ 的預測寫成 $x_\theta^t=x_0+e_t\epsilon_0$（Eq. 7），推得單步取樣分佈 $q_\theta(\hat x_t|x_{t+1})$ 的平均與訓練分佈 $q(x_t|x_0)$ 相同、變異數多出 $\big(\frac{\sqrt{\bar\alpha_t}\beta_{t+1}}{1-\bar\alpha_{t+1}}e_{t+1}\big)^2$（Eq. 9、Table 1；DDIM 見 Table 8）。作者把預測誤差 $x_\theta^t-x_0$ 視為曝光偏差的根本原因，提出度量 $\delta_t=(\sqrt{\beta'_t}-\sqrt{\bar\beta_t})^2$（Sec. 3.3），並觀察到取樣時 $\|\epsilon_\theta\|_2$ 在每個時間步都大於訓練時（Fig. 2）。據此提出 **Epsilon Scaling**：取樣時把網路輸出除以 $\lambda_t$（Sec. 4.1），不需重新訓練、不增加計算量。多數實驗用常數 $\lambda(t)=b$（ADM CIFAR-10 為 $b=1.017$），ADM 在 CIFAR-10 100 步無條件生成的 FID 由 3.37 降到 2.17（Table 2），並在 DDIM/DDPM、EDM、LDM、DiT、PFGM++ 上都有改善。

### 1.2 研究動機（Sec. 1–2）

- 擴散模型需要長的迭代取樣鏈，因為反向過程的高斯假設只在小步長成立（Xiao et al., 2022）；迭代鏈也帶來曝光偏差（Ning et al., 2023）。
- 作者認為曝光偏差「缺乏深入探討」：沒有適當的度量，也沒有明確的誤差分析（Sec. 1）。既有方法：Ning et al.（2023）以訓練正則項模擬預測誤差（需重新訓練）；Li et al.（2023a）調整取樣時間步（免訓練）；Li & van der Schaar（2023）估計並最佳化累積誤差上界（Sec. 2）。

### 1.3 貢獻（Sec. 1 列點）

1. 深入分析曝光偏差並提出度量 $\delta_t$。
2. 討論可能的解法，提出免訓練、可外掛（plug-in）的 Epsilon Scaling。
3. 在 ADM、DDIM、EDM、LDM、DiT、PFGM++ 上驗證通用性。

### 1.4 從作者的角度逐步拆解

#### 符號表

頁面不放一張總表，而是在每章開頭以可收合小表列出**該章新增**的符號（「本章符號」）。右欄「頁面寫法」是頁面實際使用的記號；論文記號在首次出現時並列一次。

| 符號 | 意義 | 頁面寫法／首次出現 |
| --- | --- | --- |
| $x_0$ | 乾淨資料；$\epsilon,\epsilon_0,\epsilon_1,\epsilon_3\sim\mathcal N(0,I)$ 為彼此獨立的標準高斯 | S1（文字）、S2 |
| $t$、$T$ | 時間步；$t$ 越大越吵，$t=T$ 為純噪聲（DDPM 慣例）。快速取樣的總步數記為 $T'$ | S1 |
| $\beta_t,\ \alpha_t=1-\beta_t,\ \bar\alpha_t=\prod_{s\le t}\alpha_s$ | 噪聲排程；$\bar\beta_t=1-\bar\alpha_t$ 為訓練分佈的變異數 | S2（S1 只用文字：ᾱₜ 是第 t 步保留原始資料的比例） |
| $x_t$、$\hat x_t$ | 訓練時由 Eq. 1 得到的帶噪樣本、取樣時由網路一步步產生的樣本 | S1 |
| $\epsilon_\theta(\cdot,t)$、$x_\theta^t$ | 噪聲預測網路、由它換算出的 $x_0$ 預測 $x_\theta(x_t,t)$ | S1（ε_θ）、S2（x_θ） |
| $\tilde\beta_t$、$\dot\beta_t$ | 後驗變異數（Eq. 4）、實際取樣時使用的噪聲變異數 | S2（β̃）、S4（β̇） |
| $e_t$ | Eq. 7 中 $x_0$ 預測誤差的標準差（每個維度） | S3 |
| $c_{1,t}=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}$、$c_{2,t}=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$ | Eq. 3/5 中 $x_0$（或 $x_\theta^t$）與 $x_t$ 的係數。本頁為了簡潔引入的記號，論文直接寫出分式 | S2 |
| $\beta'_t$、$\hat\beta_t$ | 多步取樣分佈 $q_\theta(\hat x_t\mid x_T)$ 的變異數、Alg. 1–3 量到的樣本變異數 | 頁面寫 $V_t$（多步取樣變異數，論文 $\beta'_t$）；$\hat\beta_t$ 只在 Alg. 對照框中出現，稱「樣本變異數」。S5 |
| $\Delta_t$、$\Delta'_t$、$\delta_t$ | 單步變異數誤差、多步變異數誤差（App. A.4）、曝光偏差度量（Sec. 3.3） | 頁面寫 Δ單步、Δ多步（論文 $\Delta_t$、$\Delta'_t$，S5）；$\delta_t$（S6） |
| $\epsilon_\theta^t$、$\epsilon_\theta^s$ | 網路在訓練情境（輸入 $x_t$）與取樣情境（輸入 $\hat x_t$）的輸出（Sec. 4.1）。上標 $t$ 代表 training，不是時間步 | 頁面寫 ε_train、ε_samp（論文 $\epsilon_\theta^t$、$\epsilon_\theta^s$，S7 並列一次） |
| $\Delta N(t)$ | $\|\epsilon_\theta^s\|_2/\|\epsilon_\theta^t\|_2$（Sec. 4.2） | S7（文字＋數字）、S8（C3） |
| $\lambda_t=\lambda(t)$、$k$、$b$ | 縮放係數、線性排程 $\lambda(t)=kt+b$ 的斜率與截距 | S7、S8 |

#### 步驟 1：DDPM 的取樣其實是「先預測 $x_0$，再代入後驗」（Sec. 3.1，Eq. 1–6）

$$q(x_t|x_0)=\mathcal N(x_t;\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I),\qquad x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon\tag{Eq. 1}$$

後驗 $q(x_{t-1}|x_t,x_0)=\mathcal N(\tilde\mu(x_t,x_0),\tilde\beta_tI)$（Eq. 2），$\tilde\mu=c_{1,t}x_0+c_{2,t}x_t$（Eq. 3），$\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$（Eq. 4）。網路參數化 $\mu_\theta=c_{1,t}x_\theta^t+c_{2,t}x_t$（Eq. 5）$=\frac1{\sqrt{\alpha_t}}\big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\big)$（Eq. 6）。

- Song et al.（2021a）強調：$p_\theta(x_{t-1}|x_t)$ 實際上是 $q(x_{t-1}|x_t,x_\theta^t)$。實務流程：預測 $\epsilon$ → 由 Eq. 1 換算 $x_\theta^t$ → 把後驗中的 $x_0$ 換成 $x_\theta^t$。
- 為什麼需要這一步：只有 $x_\theta^t=x_0$ 時，$q(x_{t-1}|x_t,x_\theta^t)$ 才等於 $q(x_{t-1}|x_t,x_0)$；預測誤差不為 0，所以要把誤差放進分析。

#### 步驟 2：把預測誤差寫成高斯（Eq. 7）

$$p_\theta(x_0|x_t)=\mathcal N(x_\theta^t;x_0,e_t^2I),\qquad x_\theta^t=x_0+e_t\epsilon_0\tag{Eq. 7}$$

- 沿用 Analytic-DPM（Bao et al., 2022）。論文註明：平均不是 $x_0$ 也不影響 Table 1 的變異數差距與 Sec. 3.2 的多步變異數誤差增大（論文用 explosion）。
- 隱含假設〔本頁推導：由 Eq. 12→13 的計算方式看出〕：$\epsilon_0$ 與 $x_{t+1}$ 的噪聲 $\epsilon$ 獨立，變異數才能逐項相加。頁面在 S3 寫出這一點。
- 〔本頁推導〕$e_t$ 包含不可約（Bayes）誤差：即使網路等於最佳去噪器 $\mathbb E[x_0|x_t]$，$x_t$ 也無法唯一決定 $x_0$，$e_t$ 不為 0；且最佳去噪器的誤差與 $x_\theta$ 正交，而不是與 $x_0$ 獨立。所以「誤差平均為 $x_0$、與噪聲獨立」是建模選擇，「預測誤差是根本原因」不代表更好的網路能把它降到 0。頁面 S3 以一句話寫出。

#### 步驟 3：單步取樣分佈（Eq. 8–9、Table 1，App. A.1）

一步取樣 $\hat x_t=\mu_\theta(x_{t+1},t+1)+\sqrt{\tilde\beta_{t+1}}\epsilon_1$（Eq. 8），代入 Eq. 7 與 Eq. 1（index $t+1$）：

$$\hat x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t+\Big(\frac{\sqrt{\bar\alpha_t}\beta_{t+1}}{1-\bar\alpha_{t+1}}e_{t+1}\Big)^2}\;\epsilon_3\tag{Eq. 9}$$

| | 平均 | 變異數 |
| --- | --- | --- |
| $q(x_t|x_0)$（訓練） | $\sqrt{\bar\alpha_t}x_0$ | $(1-\bar\alpha_t)I$ |
| $q_\theta(\hat x_t|x_{t+1})$（DDPM 取樣） | $\sqrt{\bar\alpha_t}x_0$ | $\big(1-\bar\alpha_t+(c_{1,t+1}e_{t+1})^2\big)I$ |

- App. A.1 的關鍵恆等式（Eq. 14→15）：$c_{2,t+1}^{\,2}(1-\bar\alpha_{t+1})+\tilde\beta_{t+1}=1-\bar\alpha_t$。也就是 $e_{t+1}=0$ 時，取樣變異數剛好等於訓練變異數。
- DDIM（App. A.3，Table 8）：變異數多出 $\big(1-\sqrt{\tfrac{\alpha_{t+1}-\bar\alpha_{t+1}}{1-\bar\alpha_{t+1}}}\big)^2\bar\alpha_te_{t+1}^2$（Eq. 31），因 $\alpha_{t+1}<1$ 恆為正。
- 這個差距**在單一步就出現**，即使網路拿到的是真實 $x_{t+1}$（Sec. 3.2）。

#### 步驟 4：調整取樣噪聲變異數不能消除差距（Sec. 3.2）

- 直覺解法：取樣噪聲變異數小於 $\tilde\beta_t$，抵消多出的項。但 Nichol & Dhariwal（2021）給出 $\dot\beta_t\in[\tilde\beta_t,\beta_t]$：下界對應 $q(x_0)$ 為 delta 函數、上界對應各向同性高斯。作者據此結論：無法靠調整 $\dot\beta_t$ 緩解曝光偏差。這是**作者的論證**：Nichol & Dhariwal 的範圍描述的是真實資料後驗的變異數，不是對取樣器的數學限制；頁面以「作者認為」呈現，不寫成定理。
- 〔本頁推導〕把 Eq. 8 的 $\tilde\beta_{t+1}$ 換成 $\dot\beta_{t+1}$，變異數變成 $1-\bar\alpha_t+(c_{1,t+1}e_{t+1})^2+(\dot\beta_{t+1}-\tilde\beta_{t+1})$；在作者採用的 $\dot\beta\ge\tilde\beta$ 範圍內，差距只會更大。此式只寫在 S4 內文，不做互動（互動只能探索 $\dot\beta\ge\tilde\beta$ 一側，無法檢驗作者論證的前提）。
- 作者對 Analytic-DPM 的解釋（「作者認為」）：Bao et al. 的最佳變異數 $\beta_t^\star>\tilde\beta_t$，1000 步時 FID 反而比 $\tilde\beta_t$ 差；作者認為長取樣鏈中曝光偏差的負面影響超過最佳變異數的增益。

#### 步驟 5：誤差在多步取樣中累積（Sec. 3.2、Fig. 1，App. A.2、A.4）

- App. A.2：兩步 $q_\theta(\hat x_{t-1}|x_{t+1})$ 平均仍為 $\sqrt{\bar\alpha_{t-1}}x_0$，變異數更大（Table 7）；同法可得 $q_\theta(\hat x_t|x_T)$ 變異數大於 $(1-\bar\alpha_t)I$。
- Fig. 1（CIFAR-10、ADM、20 步）：單步誤差 $\Delta_t$（Alg. 1：每步輸入真實 $x_t$）接近 0；多步誤差 $\Delta'_t$（Alg. 2：從 $x_T$ 連續取樣）越接近 $t=1$ 越大。讀圖值（推定，±0.02）：$\Delta'_t$ 在 $t=2$ 約 0.47、$t=10$ 約 0.265、$t=18$ 約 0.015；$\Delta_t$ 全程約 0.01 以下。
- 作者的結論：多步變異數誤差持續增大（論文用 explosion），使取樣分佈偏離訓練分佈（論文稱 sampling drift，即曝光偏差），根本原因是 $x_\theta^t-x_0$。論文只以實驗呈現 Fig. 1 的大小，沒有宣稱 Fig. 1 的曲線可由 Eq. 7 推得；Table 7／Eq. 23 只說明變異數差距會隨步數保留下來。
- 〔本頁推導 1：Table 7 的交叉項〕在 Eq. 7 模型下，兩步的精確變異數是 $1-\bar\alpha_{t-1}+(c_{1,t}e_t)^2+c_{2,t}^{\,2}f(t)$，$f(t)=(c_{1,t+1}e_{t+1})^2$（直接把 Table 1 的變異數乘上 $c_{2,t}^2$，不需泰勒展開）。Eq. 22 把同一個 $\epsilon_3$ 的兩個係數「平方和」而非「和的平方」，因此 Eq. 23／Table 7 少了一階項 $c_{2,t}^2f(t)$。Monte Carlo（$N=4\times10^5$，線性排程、$T'=20$、$e=0.3$、$t=5$）：0.25333；精確式 0.25347；Eq. 23 0.24990。結論「變異數大於 $1-\bar\alpha_{t-1}$」不受影響。頁面在 S5 以可收合附註呈現，不設學習目標。
- 〔本頁推導 2：論文模型的遞迴〕Eq. 7 模型下的多步變異數有精確遞迴 $V_{t-1}=c_{2,t}^2V_t+(c_{1,t}e_t)^2+\tilde\beta_t$，$V_{T'}=1-\bar\alpha_{T'}$（Alg. 2 的閉式版；Monte Carlo 交叉檢查一致，見第 5 節）。$c_{2,t}$ 在取樣末段趨近 0，前一步多出的變異數在下一步的權重變小，所以最後一步的多步誤差幾乎等於單步誤差；累積主要出現在取樣中段（$T'=20$ 時中點的 $\Delta'/\Delta=1.49$，$T'=100$ 時 4.55）。Eq. 7 模型本身不會產生 Fig. 1 那樣末段最大的差距（C2 呈現）；這不與論文矛盾，因為論文沒有用 Eq. 7 推導 Fig. 1。
- 〔本頁推導 3：完美網路下 Alg. 2／Alg. 3 的讀數（玩具估計，不是下限）〕Alg. 2、3 從 $x_T=\sqrt{\bar\alpha_T}x_0+\dots$ 出發，再減去同一個 $x_0$ 的 $\sqrt{\bar\alpha_{t-1}}x_0$。$\bar\alpha_T$ 很小（線性排程約 $4\times10^{-5}$），$\hat x$ 幾乎不含 $x_0$ 的資訊，所以即使網路完美，量到的 $\hat\beta_t\approx\mathrm{Var}(\hat x_t)+\bar\alpha_t\mathrm{Var}(x_0)$，變異數誤差約 $2\bar\alpha_t\mathrm{Var}(x_0)$。高斯資料、完美網路（$x_\theta=\mathbb E[x_0|x_t]$）、cosine 排程、$T'=20$、$\mathrm{Var}(x_0)=0.25$（推定：CIFAR-10 正規化到 [−1, 1] 的每像素變異數約 0.25）時，$\Delta'_t$ 在 $t=2,10,18$ 為 0.435、0.244、0.009，與 Fig. 1 讀圖值 0.47、0.265、0.015 的大小與形狀相近；但同一玩具的單步誤差在中段達 0.06，大於 Fig. 1 的單步曲線。這個值不是下限：它只對玩具成立（高斯、像素獨立、$\mathrm{Var}(x_0)=0.25$、最佳去噪器），變異數較小（多樣性較低）的取樣器可以低於它；Fig. 8（A.6，CIFAR-10、20 步）實測的 $\delta_1$ 約 0.39–0.465（讀圖），與玩具的 0.426 同一量級，部分 ADM-ES 設定低於它。頁面只陳述「Fig. 1 的多步曲線中有一部分即使網路完美也會出現」，不宣稱 Fig. 1 全部來自這裡（R2）。

#### 步驟 6：度量 $\delta_t$（Sec. 3.3、App. A.5–A.6）

$$\delta_t=\big(\sqrt{V_t}-\sqrt{1-\bar\alpha_t}\big)^2$$

（論文寫 $\delta_t=(\sqrt{\beta'_t}-\sqrt{\bar\beta_t})^2$，$\beta'_t=V_t$、$\bar\beta_t=1-\bar\alpha_t$；頁面只用 $V_t$。）

- 來源：$q(x_t|x_0)$ 與 $q_\theta(\hat x_t|x_T)$ 的 Fréchet distance $d^2=N(\sqrt{\beta'_t}-\sqrt{\bar\beta_t})^2$，$N$ 為 $x_t$ 的維度（Dowson & Landau, 1982）；兩者平均相同、共變異數皆為等向，所以只剩變異數項〔本頁推導：Fréchet 公式的套用〕。
- 量測（Alg. 3）：$N=50{,}000$ 個樣本，每個樣本從 $x_0$ 算出 $x_T$，依 Eq. 6 一路取樣，收集 $\hat x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0$，$\hat\beta_t=$ `numpy.var`，$\delta_t=(\sqrt{\hat\beta_t}-\sqrt{\bar\beta_t})^2$。
- App. A.6：給定已訓練模型，$\delta_1$ 與 FID 強相關（Fig. 8，CIFAR-10、20 步）；$\delta_t$ 也能指出中間步的輸入品質（Fig. 4 中接近 $t=1$ 時急遽上升）。
- 〔本頁推導〕由步驟 5 推導 3，完美網路的 $\delta_t$ 也不為 0（玩具估計 $\delta_1=0.426$，與 Fig. 8 實測的 0.39–0.465 同量級）。因此 $\delta_t$ 適合比較同一量測方式下的不同取樣器（論文 Fig. 4、8、9、10 正是如此使用），不宜讀成絕對偏差量。

#### 步驟 7：可能的解法（Sec. 3.4）

1. 更準確的 $\epsilon$／score 網路：EDM 以網路與超參數設計把 CIFAR-10 FID 由 3.01 降到 2.51。
2. 資料增強學到更密的向量場：幾何增強（Karras et al., 2020）；Ning et al.（2023）以高斯項擴增 $x_t$（即 DDPM-IP），FID 大幅改善。
3. Poisson Flow 對預測誤差較不敏感（Xu et al., 2022）；本文驗證 PFGM++ 的穩健性（App. A.7）。
- 以上都需要重新訓練與昂貴的參數搜尋 → 作者的問題：能否在取樣階段、不重新訓練就緩解曝光偏差？

#### 步驟 8：Epsilon Scaling（Sec. 4.1，Fig. 2）

- 觀察（Fig. 2，CIFAR-10、20 步、每步 50k 樣本）：固定已訓練網路，$\epsilon_\theta^t=\epsilon_\theta(x_t,t)$ 模擬訓練情境，$\epsilon_\theta^s=\epsilon_\theta(\hat x_t,t)$ 在真實取樣中收集；**每個時間步** $\|\epsilon_\theta^s\|_2>\|\epsilon_\theta^t\|_2$。作者推論：$\hat x_t$ 以較大變異數分布在 $x_t$ 周圍，網路在 $x_t$ 附近學到的向量場 $\epsilon_\theta(\hat x_t,t)$ 長度偏長。
- 依據：$t=T$ 時兩者輸入相同（$x_T\sim\mathcal N(0,I)$）；從 $T-1$ 起，$\hat x_t$ 因前一步的誤差開始偏離，沿取樣鏈反覆累積。所以把過長的 $\epsilon_\theta^s$ 縮小，可以縮小 $\hat x_t$ 與 $x_t$ 的差距（作者的推論）。
- 方法：
$$\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\frac{\epsilon_\theta(x_t,t)}{\lambda_t}\Big)$$
  只改平均中的 $\epsilon$ 項，噪聲項 $\sqrt{\tilde\beta_t}z$ 不變；不增加計算量（Sec. 4.1）。
- 〔本頁推導〕在 Eq. 7 模型下，Fig. 2 的方向可以重現。操作定義（頁面照錄）：$\Delta N(t)=\sqrt{\mathbb E\|\epsilon_{\text{samp}}\|^2/\mathbb E\|\epsilon_{\text{train}}\|^2}$（$N$ 維時 L2 範數集中在其均方根，所以範數比等於均方範數比的平方根）。兩個期望（每個維度）：訓練情境 $\mathbb E\|\epsilon_{\text{train}}\|^2/N=1+\frac{\bar\alpha_te_t^2}{1-\bar\alpha_t}$，取樣情境 $\mathbb E\|\epsilon_{\text{samp}}\|^2/N=\frac{V_t+\bar\alpha_te_t^2}{1-\bar\alpha_t}$；$V_t>1-\bar\alpha_t$ 所以 $\Delta N(t)>1$，在 $t=T'$ 等於 1。只重現方向、不重現量級：cosine、20 步、$e=0.1$ 時 $\Delta N(1)=1.41$、$\Delta N(2)=1.13$，Fig. 3 左端約 1.19（讀圖）；$\Delta N(1)\approx\sqrt2$ 幾乎與 $e$ 無關，是 $c_{1,1}=1$ 的數學結果（最後一步的多餘變異數約等於 $e^2$），不是資料。頁面在 S7 以一段文字與這兩個數字呈現，不放進 C2（$\Delta N$ 在 S7 才定義）。
- 〔本頁推導，不放進頁面元件〕在高斯資料＋完美網路的玩具中，常數 $\lambda>1$ 會讓 $\hat x$ 的變異數更偏離（cosine、$T'=20$、$\mathrm{Var}(x_0)=0.25$：$\lambda=1,1.017$ 時 $\delta_1=0.426,\ 0.887$）。所以 Epsilon Scaling 的效益是論文以真實網路量到的實驗結果（Fig. 2、4、5，Table 2–6），本頁不以解析玩具示範 FID 或 $\delta_t$ 的改善（R3）。

#### 步驟 9：縮放排程（Sec. 4.2，Eq. 10，Fig. 3，App. A.10–A.11）

$$\Delta N(t)-1\approx\int_t^T(\lambda_T-1)\,dt+\int_t^{T-1}(\lambda_{T-1}-1)\,dt+\dots+\int_t^{t+1}(\lambda_{t+1}-1)\,dt\tag{Eq. 10}$$

- 假設誤差沿取樣鏈線性傳播；第一項是 $T$（取樣起點）的誤差從 $T$ 傳到 $t$。量到 $\Delta N(t)$ 後解 Eq. 10 即得 $\lambda(t)$。
- 擬合時忽略 $t$ 接近 0 的區域：該處 $\epsilon_\theta$ 預測很差、loss 比其他時間步大數個數量級（Nichol & Dhariwal 2021；Benny & Wolf 2022），縮放有問題的預測沒有幫助。Fig. 3（CIFAR-10，20 步與 50 步）：$\Delta N(t)$ 在 $t\sim(5,T)$ 可用二次函數擬合；論文據此寫「Eq. 10 的解是線性函數 $\lambda(t)=kt+b$」。讀圖值（推定，±0.01）：兩條曲線大致重疊，最左端約 1.19，$t=5$ 時約 1.02–1.03，50 步曲線約在 $t=15$–20 就接近 1。
- 〔本頁推導〕Eq. 10 等價於離散和 $F(t):=\Delta N(t)-1=\sum_{s=t+1}^{T}(\lambda_s-1)(s-t)$，其二階差分為 $F(t-1)-2F(t)+F(t+1)=\lambda_t-1$（定義 $F(T+1)=0$）。所以 $\Delta N$ 是二次式時 $\lambda$ 為常數（邊界 $\lambda_T-1$ 為內部值的一半），三次式時 $\lambda$ 才是線性。一行證明（頁面正文照錄）：連續形式 $\Delta N(t)-1=\int_t^T(\lambda(s)-1)(s-t)\,ds$，對 $t$ 微分兩次得 $\Delta N''(t)=\lambda(t)-1$；二次式的二階導數是常數。這與論文「二次 → 線性」的寫法不同，但與論文最後採用的 $\lambda(t)=b$ 一致（C3）。
- 兩者量的不是同一件事（頁面 S8 開頭先寫）：Eq. 10 的 $\lambda_t$ 是在「誤差線性傳播」假設下，由範數比推得的每步範數高估比例；表中的 $b$ 是以 FID 搜尋出的取樣超參數。論文說明理想上應先模擬 $\Delta N$ 再解 Eq. 10，但實際上直接搜尋 $k,b$，Eq. 10 只提供 $\lambda$ 形狀的定性依據。〔本頁推導〕以 Fig. 3 的推定值 $F(5)=0.02$（±0.01，結果隨之變動約 1.5 倍）反解，$T'=20$ 的 $\lambda-1\approx1.8\times10^{-4}$，量級遠小於 $b-1=0.017$；此數字只放在 C3 底部一句，不放進學習目標（R4）。
- 作者的實際作法：$k,b$ 在不同網路間變化不大，所以直接搜尋，不必模擬（simulation-free）；$T'$ 約 50 時 $k$ 衰減到 0，因此多數實驗用 $\lambda(t)=b$（Sec. 4.2、5.1）。
- App. A.10 搜尋建議：(1) 常數 $b$ 由粗到細，步長 0.001、0.0005、0.0001；(2) 一般而言最佳 $b$ 隨 $T'$ 增加而下降；(3) 得到 $b^*$ 後，保持 $\sum\lambda(t)=\sum\lambda^*(t)$ 搜尋 $k$，$b$ 由此算出而不搜尋；(4) 用 10k 樣本的 FID 搜尋。Sec. 5.1：6 到 10 次試驗。
- 〔本頁推導，推定〕Table 14 的 ADM-ES\* 中，LSUN、FFHQ、ImageNet 64、ImageNet 128 四列都滿足 $b=b^*-9.5k$，即 $t$ 取 $0,\dots,19$（程式中的時間索引）時 $\sum\lambda$ 守恆；CIFAR-10 列（$k=0.0025,b=1.0,b^*=1.017$）不滿足（應為 0.99325）。
- 不敏感（App. A.11，作者的說法）：Table 17（ADM-ES，CIFAR-10）$b\in[1.015,1.019]$ 時 100 步 FID 2.17–2.31、50 步 2.49–2.55，基準 3.37、4.43；Table 18（EDM-ES，Heun）35 步 1.947–1.953（基準 1.97，只差約 0.02）、13 步 6.54–6.60（基準 7.16）。掃描範圍很窄（最佳值 ±0.002），「wide range」是相對於基準增益而言。

#### 步驟 10：實驗（Sec. 5，App. A.7–A.14）

- 評估：FID，50k 樣本、整個訓練集為參考，計算方式同 ADM；正文聚焦 $T'\le100$（Sec. 5）。
- ADM（Table 2；respacing；ImageNet 64 無 classifier guidance，ImageNet 128 guidance scale 0.5）：CIFAR-10 100/50/20 步 3.37→2.17、4.43→2.49、10.36→5.15（ES\* 4.31）。FFHQ 128 改善 7.75、16.65（100、50 步）；論文寫 20 步改善 34.52，此數字等於 59.35−24.83（ADM-ES\*，線性排程），不是 ADM-ES 的 59.35−26.14=33.21〔本頁推導：算術核對〕。
- Table 3：ADM-ES 100 步 2.17、50 步 2.49；EDM (VP) 511 步 2.27、EDM (VE) 2047 步 2.23、Improved SDE 1023 步 2.35、Restart 115 步 2.21、SA-Solver 95 步 2.63、ADM-IP 100 步 2.38。作者稱相對 EDM 與 Improved SDE 快 5 到 20 倍（511/100、2047/100）。
- DDIM/DDPM（Table 4，$\eta=0$ DDIM、$\eta=1$ DDPM）：CelebA DDPM 改善 47.7%、63.1%、60.7%（20、50、100 步）；DDIM 改善較小，作者認為原因是確定性取樣在短鏈的 FID 優勢，以及 DDPM 的噪聲項能主動修正先前誤差（Karras et al., 2022）。
- EDM（Table 5，CIFAR-10，VP，$T'$=NFE）：輸出不是 $\epsilon$，但每步可取出 $\epsilon$ 再縮放。35 步 Euler：無條件 3.81→2.80、有條件 3.74→2.59；Heun 增益較小（1.97→1.95）。作者的兩個解釋：Heun 截斷誤差較小；Heun 的校正步修正前一個 Euler 步的預測誤差，使取樣軌跡更接近訓練時的向量場（Fig. 6）。
- LDM（Table 6）：CelebA-HQ 20 步 29.62→15.68；DiT（Table 19，ImageNet 256）：12.95→10.00（20 步，$b=0.965$）。PFGM++（Tables 9–10）：$\lambda$ 的可用範圍比 EDM 寬。$T'>100$（Tables 11–12）：ADM CIFAR-10 1000 步 3.01→2.21。
- Precision/recall（Table 13）：ADM-ES recall 較高、precision 略低，差異小。
- $\delta_t$ 與範數（Sec. 5.2、App. A.12）：ADM-ES 在 $t=1$ 的 $\delta_t$ 較低（Fig. 4 為 LSUN tower 64，正文寫 CIFAR-10；CIFAR-10 見 Fig. 9、FFHQ 見 Fig. 10）；ES 後取樣 $\|\epsilon_\theta\|$ 接近訓練值（Fig. 5，LSUN tower 64）。頁面不混用不同資料集的讀值。前提是 $\|\epsilon_\theta(x_t)\|$ 在 $x_t$ 附近局部單調。ES 只修正大小誤差、不修正方向誤差，所以無法達到 $\delta_t=0$。
- 定性（Sec. 5.6）：相同 seed 下，ADM-ES 修正過曝、欠曝、粗糙背景與細節缺陷（Fig. 7），兩者到達相同或相近的 mode。

#### 步驟 11：實作細節與常見錯誤（頁面 S11 逐條列出）

1. 縮放方向：$\lambda>1$ 是**縮小** $\epsilon$；只改 Eq. 6 平均中的 $\epsilon$，不動噪聲項。
2. $b$ 不一定大於 1：EDM Heun 21 步 0.9985／0.9983（Table 16）、PFGM++ 21 步 0.970／0.996（Tables 9–10）、DiT 0.965–0.995（Table 19）。
3. $b$ 依資料集、步數、模型而定，且一般隨 $T'$ 增加而下降（A.10）；DDPM（$\eta=1$）用的 $b$（1.03–1.06）遠大於 DDIM（1.001–1.0052）（Table 15）。
4. 線性排程的 $t$ 索引：論文沒有寫明；Table 14 的算術與 $t=0,\dots,T'-1$ 一致（推定），且 CIFAR-10 列不符合 A.10 的 $\sum\lambda$ 規則。
5. EDM：先由網路輸出取出 $\epsilon$ 再縮放（Sec. 5.4）。〔本頁推導〕論文實驗用 EDM 的 VP 設定，$x=s(t)(y+\sigma n)$、$s(t)\ne1$，所以 $\hat\epsilon=\big(x/s(t)-D_\theta(x/s(t);\sigma)\big)/\sigma$（$D_\theta$ 作用在縮放後的 $x/s(t)$）。
6. 快速取樣用 respacing（Nichol & Dhariwal 2021）；擬合 $\Delta N$ 時忽略 $t$ 接近 0 的區域。
7. 搜尋 $\lambda$ 時用 10k 樣本算 FID，最終報告用 50k（A.10、Sec. 5）。

#### 步驟 12：限制

- ES 只修正 $\epsilon$ 的大小，不修正方向（Sec. 5.2）。
- 理論分析依賴 Eq. 7 的高斯、獨立誤差模型；$\lambda$ 的選擇靠 FID 搜尋，Eq. 10 只提供定性依據。
- 〔本頁推導〕即使網路完美，Alg. 3 量到的 $\delta_t$ 也不為 0（步驟 6，玩具估計）。

---

## 2. 學習目標（驗收測試）

每個參考答案都必須寫在指定章節的內文（不能只出現在元件、表格、圖或答案框）。每章末以 `<details>`「檢查理解」照錄題目與參考答案；LO9 的題目放在該章「進階」框內；LO7 的答案寫在 S5 正文（標「本頁推導」）。

「來源」欄：**論文** = 答案全取自論文；**論文＋本頁推導** = 核心答案取自論文，附帶的推導也寫在正文並標示；**進階（本頁推導）** = 章節中以「進階」框呈現的本頁推導，可略過，不屬核心驗收路徑。所有答案都寫在指定章節的正文。

| # | 學習目標 | 檢查題 | 參考答案 | 來源 | 章節 |
| --- | --- | --- | --- | --- | --- |
| LO1 | 定義擴散模型的曝光偏差 | 訓練與取樣時，$\epsilon_\theta$ 的輸入各是什麼？ | 訓練時是由真實 $x_0$ 加噪得到的 $x_t$；取樣時是網路前幾步產生的 $\hat x_t$。兩者不一致即曝光偏差（Sec. 1） | 論文 | S1 |
| LO2 | 說明 DDPM 取樣的實際流程 | $p_\theta(x_{t-1}|x_t)$ 實際上如何參數化？ | 先預測 $\epsilon$，由 Eq. 1 換算 $x_\theta^t$，再把後驗 $q(x_{t-1}|x_t,x_0)$ 中的 $x_0$ 換成 $x_\theta^t$，即 $q(x_{t-1}|x_t,x_\theta^t)$（Sec. 3.1） | 論文 | S2 |
| LO3 | 寫出單步變異數差距 | Eq. 7 模型下，$q_\theta(\hat x_t|x_{t+1})$ 比 $q(x_t|x_0)$ 多出多少變異數？平均呢？ | 多出 $\big(\frac{\sqrt{\bar\alpha_t}\beta_{t+1}}{1-\bar\alpha_{t+1}}e_{t+1}\big)^2$，平均同為 $\sqrt{\bar\alpha_t}x_0$（Eq. 9、Table 1）；網路拿到真實 $x_{t+1}$ 時就已出現 | 論文 | S3 |
| LO4 | 知道 DDIM 也有差距 | DDIM 的差距項是什麼？為什麼恆為正？ | $\big(1-\sqrt{\tfrac{\alpha_{t+1}-\bar\alpha_{t+1}}{1-\bar\alpha_{t+1}}}\big)^2\bar\alpha_te_{t+1}^2$；$\alpha_{t+1}<1$ 使根號內小於 1（Eq. 31、Table 8） | 論文 | S3 |
| LO5 | 說明作者為何認為噪聲排程無法解決 | 作者為什麼認為不能把取樣噪聲變異數調小來抵消差距？ | 作者引用 Nichol & Dhariwal（2021）：取樣噪聲變異數 $\dot\beta_t\in[\tilde\beta_t,\beta_t]$，$\tilde\beta_t$ 是下界（$q(x_0)$ 為 delta 函數時），所以不能再小；這是作者的論證（Sec. 3.2） | 論文 | S4 |
| LO6 | 區分單步與多步誤差的量測 | Alg. 1 與 Alg. 2 的差別？Fig. 1 顯示什麼？ | Alg. 1 每步輸入真實 $x_t$，Alg. 2 從 $x_T$ 連續取樣只看得到 $\hat x_t$；Fig. 1（CIFAR-10、20 步）多步誤差越接近 $t=1$ 越大，單步接近 0（App. A.4、Sec. 3.2） | 論文 | S5 |
| LO7 | 說明論文模型中誤差如何傳遞 | Eq. 7 模型下多步變異數的遞迴是什麼？為什麼最後一步幾乎不累積？ | $V_{t-1}=c_{2,t}^2V_t+(c_{1,t}e_t)^2+\tilde\beta_t$；$c_{2,t}$ 在末段趨近 0，前一步多出的變異數權重變小，最後一步的誤差主要來自該步自己的 $c_{1,t}e_t$ | 本頁推導（正文） | S5 |
| LO8 | 寫出並解讀 $\delta_t$ | $\delta_t$ 的定義、它與 Fréchet distance 的關係？ | $\delta_t=(\sqrt{V_t}-\sqrt{1-\bar\alpha_t})^2$（論文寫 $\beta'_t$、$\bar\beta_t$）；Fréchet distance $d^2=N\delta_t$，$N$ 為維度；$\delta_1$ 與 FID 強相關（Sec. 3.3、App. A.6） | 論文 | S6 |
| LO9 | 知道 $\delta_t$ 宜用於比較 | 網路完美時，Alg. 3 量到的 $\delta_t$ 是否為 0？為什麼？ | 不是。$x_T$ 幾乎不含 $x_0$ 的資訊，減去起點的 $\sqrt{\bar\alpha_t}x_0$ 後仍有約 $2\bar\alpha_t\mathrm{Var}(x_0)$ 的變異數誤差；所以 $\delta_t$ 宜用於比較取樣器，不是絕對偏差量 | 進階（本頁推導） | S6 |
| LO10 | 寫出 Epsilon Scaling 與其依據 | ES 改了取樣公式的哪一項？依據什麼觀察？ | $\mu_\theta=\frac1{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\frac{\epsilon_\theta}{\lambda_t})$，只把 $\epsilon$ 除以 $\lambda_t$，免訓練、不增加計算；依據 Fig. 2：每個時間步取樣時 ε_samp 的 L2 範數都大於 ε_train（Sec. 4.1） | 論文 | S7 |
| LO11 | 說明 $\lambda(t)$ 的設計與實際作法 | Eq. 10 的意義？論文最後用什麼排程、怎麼決定 $b$？ | $\Delta N(t)-1$ 是先前各步範數高估在線性傳播假設下的累積；論文寫二次 $\Delta N$ 對應線性 $\lambda=kt+b$，但 $T'\approx50$ 時 $k\to0$，多數實驗用常數 $\lambda=b$（ADM CIFAR-10 $b=1.017$），$b$ 以 FID 搜尋、6–10 次試驗（Sec. 4.2、5.1）。正文附〔本頁推導〕：$\Delta N''=\lambda-1$，二次 $\Delta N$ 對應常數 $\lambda$ | 論文＋本頁推導 | S8 |
| LO12 | 讀出實驗結果與限制 | ES 在 ADM CIFAR-10 100 步的 FID？哪些設定增益較大？ES 的限制？ | 3.37→2.17（Table 2）；步數少（FID 絕對改善較大）、隨機取樣（DDPM $\eta=1$）、Euler（相對 Heun）增益較大（Tables 2、4、5）；只修正 $\epsilon$ 的大小、不修正方向，無法使 $\delta_t=0$（Sec. 5.2） | 論文 | S9 |
---

## 3. 頁面大綱

單欄閱讀（680–720px），圖版加寬到最大 1080px。章節編號以 JetBrains Mono 顯示（`01 · …`），標題用陳述句。每個符號在首次使用的章節定義；每個元件放在它用到的概念之後。

| # | id | 標題 | 內容 | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈擴散模型的曝光偏差與 Epsilon Scaling〉 | 副標：「這篇論文分析訓練與取樣時網路輸入的變異數差距，並在取樣時把預測的噪聲除以一個常數來縮小差距。」三張事實卡（見下）；靜態 SVG「兩條管線」：上列訓練 $x_0\to x_t\to\epsilon_\theta$，下列取樣 $x_T\to\hat x_{T-1}\to\cdots\to\hat x_t\to\epsilon_\theta$，兩個 $\epsilon_\theta$ 方塊以虛線相連並標「輸入不同」。連結：arXiv、`../index.html#paper-406`（在關聯圖中查看）、`exposure-bias-elucidation-overview.html` | 1 螢幕 |
| S1 | `bias` | 01 · 曝光偏差是訓練與取樣時的輸入不一致 | **不寫公式**：擴散模型兩階段（訓練時把資料加噪、讓網路預測所加的噪聲；取樣時從純噪聲反覆去噪）；ᾱₜ 只用文字（「第 t 步保留原始資料的比例，t 越大越小」）；$x_t$ 與 $\hat x_t$；曝光偏差定義與序列生成出處；本文回答的三件事。答 LO1。延伸：`exposure-bias-seq2seq-origin.html`、`diffusion-exposure-bias.html` | 300–400 字 |
| S2 | `sampler` | 02 · DDPM 取樣先預測 x₀ 再代入後驗 | 本章符號（$\beta_t,\alpha_t,\bar\alpha_t,\tilde\beta_t,c_1,c_2,x_\theta$）→ Eq. 1–6，每式後一句白話 → respacing 一段（$T'$ 步時從 1000 步中均勻挑時間步，$\bar\alpha$ 沿用原值）。答 LO2。延伸：`ddpm-reverse-sampling-formulation.html`、`eps-prediction-ddpm-score.html` | 400–500 字 |
| S3 | `gap` | 03 · 預測誤差讓單步取樣的變異數變大 | Eq. 7 → 獨立性假設與不可約誤差（各一句）→ Eq. 8 → Eq. 9／Table 1（A.1 恆等式一行）→ Table 8（DDIM）→ **C1**。答 LO3、LO4。延伸：`prediction-error-variance-analysis.html` | 400–500 字 + C1 |
| S4 | `schedule-noise` | 04 · 作者認為調整噪聲排程無法抵消差距 | Nichol & Dhariwal 的範圍與作者的結論（「作者認為」框）→〔本頁推導〕把 β̃ 換成 β̇ 的變異數式 → Analytic-DPM 的「作者認為」。答 LO5。延伸：`diffusion-sampling-variance-schedule.html` | 200–300 字 |
| S5 | `accum` | 05 · 誤差在多步取樣中累積 | 本章符號：$V_t$（多步取樣得到的 $\hat x_t$ 在 Eq. 7 模型下的變異數，論文寫 $\beta'_t$）、Δ單步、Δ多步 → Alg. 1 vs Alg. 2（兩欄對照，≥768px 並排）→ Fig. 1 敘述與讀圖值（推定）→ 作者結論 → 正文一段〔本頁推導〕寫出遞迴 $V_{t-1}=c_{2,t}^2V_t+(c_{1,t}e_t)^2+\tilde\beta_t$ 與「$c_2$ 末段趨近 0，最後一步的誤差主要來自該步自己」（LO7 答案在正文，不在收合框內）→ 「進階」框：推導細節與 $T'$ 對中段比值的影響 → 可收合附註「Table 7 的交叉項」。答 LO6 | 400–500 字 |
| S6 | `metric` | 06 · δₜ 以變異數差距量化曝光偏差 | 定義（沿用 S5 的 $V_t$）、Fréchet 來源、Alg. 3 要點、Fig. 8 → **C2**（預設模式 A）→ 「進階」框：高斯資料的最佳去噪器（見下方 S6 進階框規格）與模式 B 的讀法、玩具估計 $\delta_1=0.426$ 與 Fig. 8 的 0.39–0.465（答 LO9）。答 LO8 | 400–500 字 + C2 + 進階框 200 字 |
| S7 | `es` | 07 · Epsilon Scaling 在取樣時把 ε 除以 λ | Sec. 3.4 三類解法（各一句，DDPM-IP 只提「以高斯項擴增 $x_t$，需重新訓練」）→ ε_train／ε_samp 與 Fig. 2 觀察 → 推論與依據（$t=T$ 輸入相同）→ 公式（與 Eq. 6 並列，差異以 `--accent` 標出）→ 取樣迴圈程式碼（見下）→ $\Delta N$ 定義＋〔本頁推導〕Eq. 7 模型只重現方向（1.41、1.13 對 Fig. 3 約 1.19）→ 一句：本頁不以玩具示範 FID 改善。答 LO10。延伸：`epsilon-scaling-method.html`、`exposure-bias-prior-mitigations.html` | 450–550 字 |
| S8 | `lambda` | 08 · λ(t) 的形狀來自範數比，數值由搜尋決定 | 開頭先寫「Eq. 10 的 λ 與表中的 b 量的不是同一件事」（1.4 步驟 9）→ Eq. 10 與線性傳播假設 → Fig. 3（讀圖值推定）→ 論文的線性解與 $k\to0$、常數 $b$ → **C3** →〔本頁推導〕一行證明 $\Delta N''=\lambda-1$ → Table 14 ADM-ES\* 與 $\sum\lambda$ 核對（小表）。答 LO11 | 400–500 字 + C3 + 1 小表 |
| S9 | `results` | 09 · ES 在六種框架上都降低 FID | 評估設定一句 → 正文只寫 Table 2、3、4、5 的關鍵數字（1.4 步驟 10）→ **C4**（其餘表格）→ Fig. 4／5 的 $\delta_t$ 與範數（標資料集）→ 限制（大小 vs 方向）。答 LO12 | 400–500 字 + C4 |
| S10 | `why` | 10 · 作者對增益差異的解釋 | 「作者認為」三段：DDIM 增益較小、Heun 校正步（Fig. 6）、定性樣本（Fig. 7） | 200–300 字 |
| S11 | `impl` | 11 · 實作設定與常見錯誤 | ES 一行修改的程式碼 → A.10 四條搜尋建議 → Table 17／18 不敏感（小表，含「範圍很窄」說明）→ 1.4 步驟 11 七條常見錯誤 | 350–450 字 + 1 小表 |
| S12 | `history` | 12 · 研究脈絡 | 時間線（6.4）＋一段文字：序列生成的曝光偏差 → DDPM／DDIM → 取樣變異數（IDDPM、Analytic-DPM、EDM）→ 需訓練（DDPM-IP、誤差傳播）與免訓練（shifted time steps）的緩解 → 本文 | 時間線 + 100–150 字 |
| S13 | `next` | 13 · 限制與延伸閱讀 | 1.4 步驟 12；筆記清單（6.4）；arXiv | 200 字 + 清單 |

S6 進階框規格（item：模式 B 的先備知識）：一段內文說明「若資料是平均 0、變異數 $\sigma_0^2$ 的高斯，$x_0$ 與 $x_t$ 為聯合高斯，最佳預測是線性的：$\mathbb E[x_0|x_t]=k_tx_t$，$k_t=\mathrm{Cov}(x_0,x_t)/\mathrm{Var}(x_t)=\sqrt{\bar\alpha_t}\sigma_0^2/(\bar\alpha_t\sigma_0^2+1-\bar\alpha_t)$。代入 Eq. 5 後每一步都是 $\hat x_{t-1}=a_t\hat x_t+\sqrt{\tilde\beta_t}z$，所以只要追蹤 $\hat x$ 的變異數與它和 $x_0$ 的共變異數兩個數，就能算出 Alg. 2 會量到的值。」接著兩句結論（$x_T$ 幾乎不含 $x_0$ 資訊；誤差約 $2\bar\alpha_t\sigma_0^2$）。共變異數遞迴的式子不寫在正文，放在元件底部的「計算方式」`<details>`。

文案用詞（item 12）：不用「爆炸」「漂移」「推回」「拉回」「阻尼」「壓小」作為敘述用語。改寫為「變異數誤差持續增大」「取樣分佈偏離訓練分佈」「縮小 $\hat x_t$ 與 $x_t$ 的差距」「$c_2$ 趨近 0，使前一步多出的變異數權重變小」。論文術語 explosion、sampling drift 只在引述時以括號附原文一次。
Hero 三張事實卡：
1. 「訓練時網路看到的是由真實資料加噪的 xₜ，取樣時看到的是自己前幾步產生的 x̂ₜ。兩者分佈不同，稱為曝光偏差。」
2. 「把每一步的 x₀ 預測誤差建模成高斯分佈後，取樣分佈的變異數在每一步都大於訓練分佈。」
3. 「Epsilon Scaling 在取樣時把網路預測的噪聲除以常數 λ（例如 1.017），不需重新訓練。ADM 在 CIFAR-10 以 100 步取樣，FID 由 3.37 降到 2.17。」

S7 程式碼（ADM 風格，`# ES` 為唯一改動；頁面以等寬字＋左側行號呈現）：
```
eps = model(x, t)
eps = eps / lam          # ES：lam = b（常數）或 k*t + b
mean = (x - beta[t] / sqrt(1 - abar[t]) * eps) / sqrt(alpha[t])
x = mean + sqrt(beta_tilde[t]) * randn_like(x)   if t > 0 else mean
```

桌機（≥1024px）左側 sticky 章節目錄，目前章節以 `--accent` 標示；<1024px 改為頂部橫向捲動章節列。

---

## 4. 互動元件規格

### 4.0 共用數學（C1、C2 共用，函式見 6.3）

- 原始排程 $T=1000$：
  - `linear`：$\beta_i=10^{-4}+(0.02-10^{-4})\,i/999$，$i=0..999$；$\bar\alpha^{(1000)}_i=\prod_{j\le i}(1-\beta_j)$。
  - `cosine`：$f(u)=\cos^2\!\big(\frac{u/1000+0.008}{1.008}\cdot\frac\pi2\big)$，$\beta_i=\min(1-f(i+1)/f(i),\,0.999)$，同樣累乘。
- Respacing 到 $T'$ 步：$\tau_t=\mathrm{round}\big((t-1)\cdot999/(T'-1)\big)$（先乘後除；$T'\in\{10,20,50,100\}$ 時無 .5 平手），$\bar\alpha_t=\bar\alpha^{(1000)}_{\tau_t}$（$t=1..T'$），$\bar\alpha_0=1$；$\alpha_t=\bar\alpha_t/\bar\alpha_{t-1}$，$\beta_t=1-\alpha_t$，$\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$，$c_{1,t},c_{2,t}$ 見符號表。$t=1$ 時 $\tilde\beta_1=0$、$c_{1,1}=1$、$c_{2,1}=0$。
- 預設排程 `cosine`（推定：論文未寫 ADM CIFAR-10 的排程；cosine 使 C2 模式 B 的曲線與 Fig. 1 相近，見 R2）。
- **索引對照（item 11）**：`coefs(A, s)` 描述「迴圈第 $s$ 步：$x_s\to x_{s-1}$」。頁面與驗收表一律以**顯示的 $t$**（樣本 $\hat x_t$ 的下標）為準：

  | 顯示量（在顯示的 $t$） | 由哪一步算出 | 實作 |
  | --- | --- | --- |
  | C1 差距 $g_t$、$\tilde\beta$、$\beta$ 讀數 | 產生 $\hat x_t$ 的那一步 $s=t+1$ | `coefs(A, t+1)` |
  | C2 Δ單步(t)、Δ多步(t)、$V_t$、$\delta_t$ | 迴圈第 $s=t+1$ 步結束時 | 迴圈內寫入 `arr[s-1]` |
  | C2 模式 A 讀數 $c_2$ | 下一步 $s=t$ 的權重 | `coefs(A, t).c2` |
  | 顯示範圍 | $t=1..T'-1$（論文 Alg. 的索引集合） | — |

- 數值保護：$1-\bar\alpha_t\ge4\times10^{-5}$（cosine $t=1$ 為最小值，$T'\le100$ 時無除零）；開根號前取 $\max(0,\cdot)$；對數軸見 C1。

### C1 · 單步：訓練分佈與取樣分佈的變異數（S3）

- **教的概念**：Eq. 9／Table 1（DDPM）與 Eq. 31／Table 8（DDIM）的差距項恆為正，在單一步就出現（LO3、LO4）。噪聲排程的論證（LO5）只在 S4 正文，不在本元件。
- **輸入**：分段按鈕「取樣器」DDPM／DDIM（預設 DDPM）；「排程」cosine／linear（預設 cosine）；「步數 $T'$」10／20／50／100（預設 20）；滑桿 $e\in[0,0.5]$ 步長 0.01（預設 0.1，標籤「e（示意值，論文未給）」）；滑桿 $t\in[1,T'-1]$ 整數（預設 2；改 $T'$ 時夾到範圍內）。
- **計算**：訓練 $\bar\beta_t=1-\bar\alpha_t$；DDPM 差距 $g_t=(c_{1,t+1}e)^2$；DDIM 差距 $g_t=\big(1-\sqrt{(\alpha_{t+1}-\bar\alpha_{t+1})/(1-\bar\alpha_{t+1})}\big)^2\bar\alpha_te^2$；取樣變異數 $\bar\beta_t+g_t$；相對差距 $g_t/\bar\beta_t$。
- **顯示**：
  - 左圖：橫軸 $t=1..T'-1$（軸上箭頭「取樣方向」由右到左），縱軸對數 $[10^{-8},1]$。三條曲線：訓練變異數（`--c-train` 實線＋圓點，標「訓練 1−ᾱₜ」）、差距（`--accent` 虛線＋方點，標「差距」）、相對差距（`--ink-3` 點線＋三角點，標「差距／訓練」）。目前 $t$ 以垂直細線標示。低於 $10^{-8}$ 的點畫在 $10^{-8}$ 並改為空心點，圖下註「空心點表示小於 10⁻⁸」；相對差距大於 1 的點畫在上緣並同樣改為空心（上緣標「≥ 1」）。$e=0$ 時「差距」與「差距／訓練」兩條都不畫，顯示「e = 0：沒有預測誤差，差距為 0」。
  - 右側長條（線性軸，0 到兩者較大值的 1.1 倍）：「訓練」「取樣」並排，取樣長條堆疊 $\bar\beta_t$＋差距（斜線填充，pattern 線色用 `--accent`），各段直接標籤。
  - 讀數（Unicode，4 位有效數字）：ᾱₜ、1−ᾱₜ、β̃ₜ₊₁、βₜ₊₁、差距、取樣變異數、差距／訓練。
  - 圖下兩行：「讀數由本頁依 Eq. 9、Eq. 31 即時計算。」「e 是示意值；論文的 eₜ 隨 t 改變（Sec. 4.2 指出 t 接近 0 時預測最差），這裡用常數，所以差距隨 t 的變化只來自係數 c₁。」
- **讀者應觀察到**：(1) $e>0$ 時差距恆為正；(2) 在常數 $e$ 下，差距在取樣末段（小 $t$）最大，$t=1$ 時是訓練變異數的 240 倍（cosine、20 步、$e=0.1$，示意）；這個形狀來自 $c_{1,t+1}$，不是論文的實測；(3) DDIM 差距較小但不為 0。

### C2 · 多步：變異數誤差的累積與 δₜ（S6）

- **教的概念**：Alg. 1／Alg. 2 的單步與多步誤差（LO6）；$\delta_t$（LO8）；進階：Eq. 7 模型的遞迴（LO7）與完美網路下的讀數（LO9）。$\Delta N$ 不在本元件（S7 才定義）。
- **輸入**：分段按鈕「網路模型」A＝「論文 Eq. 7」／B＝「完美網路（進階）」（預設 A；B 按鈕旁標「進階，見下方說明框」）；「排程」cosine／linear（預設 cosine）；「$T'$」10／20／50／100（預設 20）；滑桿 $e\in[0,0.5]$ 步長 0.01（預設 0.1，只在 A，標「示意值」）；滑桿 $\sigma_0^2=\mathrm{Var}(x_0)\in[0.05,1]$（state 變數 `v0`） 步長 0.05（預設 0.25，只在 B，標「推定：CIFAR-10 在 [−1,1] 約 0.25」）；滑桿 $t$（預設 1）；勾選框「顯示 Fig. 1 讀圖值（推定）」（預設勾選，只在 $T'=20$ 顯示）。
- **計算**（迴圈第 $s=T'..1$ 步，寫入顯示 $t=s-1$）：
  - A：$V_{T'}=1-\bar\alpha_{T'}$；$V_{s-1}=c_{2,s}^2V_s+(c_{1,s}e)^2+\tilde\beta_s$。Δ單步$(s-1)=(c_{1,s}e)^2$；Δ多步$(t)=V_t-(1-\bar\alpha_t)$；$\delta_t=(\sqrt{V_t}-\sqrt{1-\bar\alpha_t})^2$。
  - B：$k_s=\sqrt{\bar\alpha_s}\sigma_0^2/(\bar\alpha_s\sigma_0^2+1-\bar\alpha_s)$，$a_s=c_{2,s}+c_{1,s}k_s$。多步：$P_{01}=\sqrt{\bar\alpha_{T'}}\sigma_0^2$、$P_{11}=\bar\alpha_{T'}\sigma_0^2+1-\bar\alpha_{T'}$；每步 $P_{01}\leftarrow a_sP_{01}$、$P_{11}\leftarrow a_\sigma_0^2P_{11}+\tilde\beta_s$；$\hat\beta_{s-1}=P_{11}-2\sqrt{\bar\alpha_{s-1}}P_{01}+\bar\alpha_{s-1}\sigma_0^2$，Δ多步 $=\hat\beta-(1-\bar\alpha)$，$\delta$ 以 $\hat\beta$ 代 $V$。單步（Alg. 1）：$\hat\beta^{(1)}_{s-1}=(a_s\sqrt{\bar\alpha_s}-\sqrt{\bar\alpha_{s-1}})^2\sigma_0^2+a_\sigma_0^2(1-\bar\alpha_s)+\tilde\beta_s$。參考線 $2\bar\alpha_t\sigma_0^2$。這些式子放在元件底部 `<details>`「計算方式」。
- **顯示**：
  - 上圖「變異數誤差」：橫軸 $t$（取樣方向箭頭），縱軸線性；A 自動縮放（上限為兩曲線最大值的 1.15 倍），B 固定 $[0,1]$。Δ單步（`--c-train` 實線＋圓點）、Δ多步（`--accent` 實線＋方點），直接標籤；B 另有參考線（`--ink-3` 細虛線，標「2ᾱₜ·Var(x₀)」）。Fig. 1 讀圖值：空心菱形 $(2,0.47),(10,0.265),(18,0.015)$，標「Fig. 1 讀圖（推定）」。A 模式下讀圖值遠超出自動縮放範圍時，菱形畫在上緣並加「↑ 0.47」等文字。
  - 下圖：兩種模式都畫 $\delta_t$（縱軸線性、自動縮放）。
  - 讀數（選定 $t$）：Δ單步、Δ多步、Δ多步／Δ單步（Δ單步 < 1e-15 顯示「—」）、δₜ；A 另有 c₂（下一步的權重）。
  - 模式說明列：A「論文的解析模型：每步的 x₀ 預測以真實 x₀ 為中心，誤差彼此獨立。」B「網路等於高斯資料的最佳去噪器，用 Alg. 2 的方式量測：從 x_T 出發，減去起點 x₀ 的 √ᾱₜ x₀。」
  - 圖下：「讀數由本頁即時計算。A 是 Eq. 7 下的精確遞迴，B 是高斯玩具，兩者都是本頁推導；Fig. 1 讀圖值為推定。B 的值是玩具估計，不是下限。」
- **讀者應觀察到**：(A) Δ多步 ≥ Δ單步，中段比值隨 $T'$ 增大（20 步 1.49、100 步 4.55），但 $t=1$ 時兩者幾乎相等。(B) 多步誤差在 $t=2,10,18$ 為 0.435、0.244、0.009，與 Fig. 1 讀圖值形狀相近、接近 $2\bar\alpha_t\sigma_0^2$；$\delta_1=0.426\ne0$；單步誤差中段約 0.06，大於 Fig. 1 的單步曲線。

### C3 · Eq. 10：範數比的形狀決定 λ(t) 的形狀（S8）

- **教的概念**：Eq. 10 的累積結構；二階差分反解；二次 $\Delta N$ ↔ 常數 $\lambda$、三次 ↔ 線性（LO11 的本頁推導部分）。量級比較只在底部一句。
- **輸入**：$T'=20$ 固定（Fig. 3 的 20 步曲線；20 與 50 步曲線重疊，50 步不另設選項）；滑桿 $h=\Delta N(5)-1\in[0,0.1]$ 步長 0.001（預設 0.02，標「推定，讀自 Fig. 3（±0.01）」）；滑桿 $w\in[0,1]$ 步長 0.05（預設 0，標「形狀：0 = 二次，1 = 三次」）。
- **計算**（$t\in[5,20]$，對應 Sec. 4.2 的 $t\sim(5,T)$）：$u_t=(20-t)/15$，$F(t)=h[(1-w)u_t^2+wu_t^3]$，$F(21)=0$；$\lambda_t=1+F(t-1)-2F(t)+F(t+1)$，$t=6..20$；回代 $F'(t)=\sum_{s=\max(t+1,6)}^{20}(\lambda_s-1)(s-t)$，顯示 $\max|F'-F|$（< 1e-12）。
- **顯示**：左圖 $\Delta N(t)=1+F(t)$（`--c-train` 實線）；右圖 $\lambda_t-1$（`--accent` 圓點＋連線，縱軸科學記號），水平參考 $2h/15^2$（細虛線，標「二次式的常數解」）。讀數：$\lambda_6$、$\lambda_{13}$、$\lambda_{19}$、$\lambda_{20}$（6 位小數）、回代誤差。
- 圖下三行：「Eq. 10 的離散形式與二階差分為本頁推導；h 讀自 Fig. 3（推定）。」「這裡的 λ 是由範數比推得的每步高估比例；論文實際使用的 b 是以 FID 搜尋的取樣超參數，兩者不是同一個量。」「以預設值反解得 λ − 1 ≈ 1.8×10⁻⁴；ADM-ES 在 CIFAR-10 20 步搜尋到 b − 1 = 0.017。」
- **讀者應觀察到**：$w=0$ 時 $\lambda$ 在 $t=6..19$ 為常數（預設 1.000178），$\lambda_{20}$ 的增量為其一半；$w=1$ 時 $\lambda$ 隨 $t$ 線性變化。

### C4 · 實驗結果瀏覽器（S9）

- **教的概念**：ES 在各框架、步數、取樣器的 FID 改善；$b$ 的取值；敏感度（LO12）。
- **輸入**：分段按鈕「表格」：ADM（Table 2）／DDIM·DDPM（Table 4）／EDM（Table 5）／LDM（Table 6）／DiT（Table 19）／PFGM++（Tables 9–10）／長取樣鏈（Tables 11–12）／敏感度（Tables 17–18）；第二層 chip 選子集（ADM 5 個資料集；DDIM 的 CIFAR η=0、η=1、CelebA η=0、η=1；EDM 的無條件／有條件 × Heun／Euler；PFGM++ D=128／2048；長鏈 Analytic-DPM／ADM；敏感度 Table 17 100 步／50 步、Table 18 35 步／13 步）。勾選框「以表格顯示」。
- **預設（載入時）**：表格 = ADM（Table 2）、子集 chip = CIFAR-10 32×32、「以表格顯示」未勾選（顯示長條圖）。切換表格時子集重設為該表的第一個 chip。
- **資料結構**（`RESULTS`，item 10）：
  ```js
  // 一般表
  {id:'adm', table:'Table 2', subsets:[{name:'CIFAR-10 32×32', rows:[
    {T:100, base:3.37, es:2.17, b:1.017}, {T:50, base:4.43, es:2.49, b:1.017},
    {T:20, base:10.36, es:5.15, b:1.017, esStar:{fid:4.31, k:0.0025, b:1.0}} ]}, …]}
  // 敏感度
  {id:'sens', table:'Table 17', subsets:[{name:'ADM-ES CIFAR-10 100 步', base:3.37,
    points:[{b:1.015, fid:2.20}, …]}, …]}
  ```
  `b` 可為 `null`（Table 11、12 未列 b，顯示「論文未列」）；`esStar` 只出現在 ADM 20 步；橫軸標籤用 `T`（EDM 註明「NFE」）。
- **計算**：每列 Δ = base − es、改善率 100Δ/base（一位小數）；有 `esStar` 時另算一組。
- **顯示**：每個子集一張成對長條圖（基準：`--surface-2` 填色＋`--ink` 外框；ES：`--accent` 實心；ES\*：`--accent` 斜線填充），**縱軸從 0 起、依該子集最大值的 1.15 倍自動縮放**（FID 1.9 到 59 不共用軸），每條上方標數值、組下方標改善率與 b。敏感度：折線（點＋線），橫軸 b，縱軸自動縮放且**不從 0 起**（軸旁註明「縱軸不從 0 起」），基準以水平虛線標「b = 1（基準）」；若基準超出範圍，畫在上緣並標數值。「以表格顯示」改為 `<table>`（含 `<caption>`）。
- **資料**（照錄論文）：
  - Table 2／14（ADM；CIFAR-10、LSUN tower 64、FFHQ 128、ImageNet 64、ImageNet 128）：100 步 ADM 3.37/3.59/14.52/2.71/3.55，ES 2.17/2.91/6.77/2.39/3.37，b 1.017/1.006/1.005/1.006/1.004；50 步 ADM 4.43/7.28/26.15/3.75/5.15，ES 2.49/3.68/9.50/3.07/4.33，b 1.017/1.007/1.007/1.006/1.004；20 步 ADM 10.36/23.92/59.35/10.96/12.48，ES 5.15/8.22/26.14/7.52/9.95，b 1.017/1.011/1.008/1.006/1.005；ES\* 4.31/7.60/24.83/7.37/9.86，(k,b) (0.0025,1.0)/(0.0008,1.0034)/(0.0004,1.0042)/(0.0002,1.0041)/(0.00022,1.00291)。
  - Table 4／15（DDIM；CIFAR η=0、η=1、CelebA η=0、η=1）：100 步 4.06/6.73/5.67/11.33 → 3.38/4.01/5.05/4.45，b 1.0014/1.03/1.003/1.04；50 步 4.82/10.29/6.88/15.09 → 4.17/4.57/6.20/5.57，b 1.0030/1.04/1.004/1.05；20 步 8.21/20.15/10.43/22.61 → 6.54/7.78/10.38/11.83，b 1.0052/1.05/1.001/1.06。
  - Table 5／16（EDM，CIFAR-10；無條件 Heun/Euler、有條件 Heun/Euler）：35 步 1.97/3.81/1.82/3.74 → 1.95/2.80/1.80/2.59，b 1.0005/1.0034/1.0006/1.0035；21 步 2.33/6.29/2.17/5.91 → 2.24/4.32/2.08/3.74，b 0.9985/1.0043/0.9983/1.0045；13 步 7.16/12.28/6.69/10.66 → 6.54/8.39/6.16/6.59，b 1.0060/1.0048/1.0070/1.0051。
  - Table 6／15（LDM；FFHQ 256、CelebA-HQ 256）：100 步 10.90/9.31 → 9.83/7.36，b 1.00015/1.0009；50 步 14.34/13.95 → 11.57/9.16，b 1.0016/1.003；20 步 33.13/29.62 → 20.91/15.68，b 1.007/1.010。
  - Table 19（DiT，ImageNet 256）：20/50/100 步 12.95/3.71/2.57 → 10.00/3.30/2.52，b 0.965/0.989/0.995。
  - Table 9（PFGM++ D=128）：9/13/21/35 步 37.82/7.55/2.34/1.92 → 17.91/4.51/2.31/1.91，b 1.008/1.016/0.970/1.00045；Table 10（D=2048）：37.16/7.34/2.31/1.91 → 17.27/4.88/2.20/1.90，b 1.007/1.015/0.996/1.0007。
  - Table 11（Analytic-DDPM，CIFAR-10 線性排程）：20/50/100/200/400/1000 步 14.61/7.25/5.40/4.01/3.62/4.03 → 11.02/5.03/4.09/3.39/3.14/3.42；Table 12（ADM）：200/300/1000 步 3.04/2.95/3.01 → 2.15/2.14/2.21（b 皆 null）。
  - Table 17（ADM-ES，CIFAR-10）：b = 1.015/1.016/1.017/1.018/1.019；100 步 2.20/2.18/2.17/2.21/2.31（基準 3.37）；50 步 2.53/2.51/2.49/2.53/2.55（基準 4.43）。Table 18（EDM-ES Heun）：35 步 b = 1.0005/1.0006/1.0007/1.0008 → 1.948/1.947/1.949/1.953（基準 1.97）；13 步 b = 1.004/1.005/1.006/1.007 → 6.60/6.55/6.54/6.55（基準 7.16）。
- **讀者應觀察到**：ADM 的 FID 絕對改善隨步數減少而增加（FFHQ 100/50/20 步 7.75、16.65、33.21），改善率不一定單調（53.4%、63.7%、56.0%；ES\* 20 步 58.2%）；DDPM 欄改善大於 DDIM 欄；Euler 欄大於 Heun 欄；敏感度模式中所有掃描點都低於基準，Table 17 與 Table 18 的 13 步差距明顯，Table 18 的 35 步只差約 0.02；掃描範圍只有最佳值 ±0.002，「不敏感」是作者相對於基準增益的說法。

### 4.5 共同的無障礙、排版與互動規則（items 8、12）

- **KaTeX**：靜態正文與靜態標籤（`<label>`、表頭）用 KaTeX，頁面載入後對每個章節容器各呼叫一次 `renderMathInElement(container, opts)`；元件內所有**動態**文字（讀數、滑桿值、圖上標籤、模式說明、`aria-label`、`aria-valuetext`、live region）一律用 Unicode 純文字（例：「ᾱₜ = 0.99085」「Δ多步 = 0.4353」），不再呼叫 KaTeX。`<details>` 內的靜態 TeX 在初次載入時一起渲染。
- **窄螢幕（< 600px）**：圖版改為單欄：圖在上、控制項在中、讀數在下；每張 SVG 的寬度等於容器實際寬度（`ResizeObserver` 觸發重畫，比照 k-diff 頁的建置作法），圖上文字固定 12–13px（以 px 設定、不隨 viewBox 縮放）；窄寬度時減少刻度數量（橫軸只標首、尾與每 5 步）。C1 右側長條移到左圖下方。讀數改為兩欄 `<dl>`（名稱／數值）。≥ 600px 時讀數為表格、控制項在圖右側。
- 曲線以線型／點形＋直接標籤區分，不只靠顏色；堆疊與 ES\* 的 pattern 線色使用 tokens，建置時在 light／dark 兩種 `--surface` 上確認圖形對比 ≥ 3:1，不足則加粗線條或改色並回寫 6.2。
- 控制項：滑桿有 `<label>`、`aria-valuetext`；分段按鈕用 `role="radiogroup"`＋`role="radio"`／`aria-checked`，roving tabindex（只有選取項 `tabindex=0`），左右／上下方向鍵切換並立即套用；勾選框用原生元素。觸控區 ≥ 44px。
- 不在圖上做點擊或拖曳：所有選取都透過滑桿 $t$，避免與觸控捲動衝突；因此不需要 hover 或 `touch-action` 設定。
- 更新頻率：`input` 事件只記錄狀態，以 `requestAnimationFrame` 合併成每影格一次重畫；重畫只更新曲線 path、指標線與讀數節點，不重建座標軸（軸只在換 $T'$、排程、模式或尺寸時重建）。live region（`aria-live="polite"`）與 SVG 的 `aria-label` 只在 `change` 或停止輸入 500ms 後更新。
- `prefers-reduced-motion: reduce`：停用所有 CSS transition 與動畫；圖形更新直接切換、不做補間；`scroll-behavior: auto`。本頁沒有自動播放的動畫。
- 焦點樣式：2px `--accent` 外框、2px 位移。

---

## 5. 驗收數值表

全部由 `verify/` 腳本產生（同一演算法、雙精度）。**容許誤差（所有讀數）**：$|x-x_{ref}|\le10^{-3}\,|x_{ref}|+10^{-12}$（相對 1e-3 涵蓋表中 4 位有效數字的捨入；絕對下限 1e-12 只處理參考值為 0 的情形），另註明者除外。

所有表的 $t$ 為**顯示的 $t$**（4.0 索引對照）。

**C1**（cosine、$T'=20$、$e=0.1$，`c1_cos_out.txt`）

| t | ᾱₜ | 訓練 1−ᾱₜ | 差距 DDPM | 差距／訓練 | 差距 DDIM | β̃ₜ₊₁ | βₜ₊₁ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.99996 | 4.128e-5 | 9.910e-3 | 240 | 8.707e-3 | 0.00004 | 0.00911 |
| 2 | 0.99085 | 0.00915 | 5.024e-3 | 0.549 | 2.128e-3 | 0.00652 | 0.02213 |
| 5 | 0.88812 | 0.11188 | 1.287e-3 | 0.0115 | 4.031e-4 | 0.04259 | 0.06435 |
| 10 | 0.53434 | 0.46566 | 4.239e-4 | 9.1e-4 | 1.242e-4 | 0.13116 | 0.15440 |
| 18 | 0.02654 | 0.97346 | 1.487e-4 | 1.53e-4 | 6.598e-5 | 0.72874 | 0.74351 |

（表頭 β̃ₜ₊₁、βₜ₊₁ 指產生 $\hat x_t$ 那一步，腳本欄名 `bt`、`beta`。）linear、$T'=20$、$e=0.1$：$t=1$ 差距 9.941e-3、DDIM 8.951e-3；$t=10$ 差距 1.974e-4。$e=0$ 時差距為 0（對數軸不畫，顯示說明）。

**C2 模式 A**（Eq. 7 模型，`check_out.txt`；ΔN 欄不在元件中，只作 S7 正文數字 1.41、1.13 的依據）

| 設定 | t | Δ單步 | Δ多步 | δₜ | ΔN(t)（S7 正文） |
| --- | --- | --- | --- | --- | --- |
| cosine, 20, e=0.1 | 1 | 9.910e-3 | 9.910e-3 | 8.711e-3 | 1.4096 |
| | 2 | 5.024e-3 | 5.313e-3 | 6.054e-4 | 1.1308 |
| | 5 | 1.287e-3 | 1.935e-3 | 8.292e-6 | 1.0080 |
| | 10 | 4.239e-4 | 9.148e-4 | 4.489e-7 | 1.0010 |
| | 19 | 6.807e-5 | 6.807e-5 | 1.166e-9 | 1.0000 |
| cosine, 20, e=0.3 | 1 | 8.919e-2 | 8.919e-2 | 8.544e-2 | 1.4109 |
| | 5 | 1.158e-2 | 1.741e-2 | 6.294e-4 | 1.0444 |
| linear, 20, e=0.1 | 1 | 9.941e-3 | 9.941e-3 | 8.137e-3 | 1.4087 |
| | 10 | 1.974e-4 | 2.941e-4 | 2.406e-8 | 1.0002 |

linear、$e=0.1$ 中點 $\Delta'/\Delta$：$T'=10,20,50,100$ → 1.107、1.490、2.763、4.552；$t=1$ 比值 1.0000、1.0000、1.0001、1.0010。Monte Carlo（$N=4\times10^5$，控制變量）：$t=1,2,5,10$ 的 $\Delta'$ 為 9.904e-3、5.456e-3、1.753e-3、2.915e-4，與閉式 9.941e-3、5.461e-3、1.732e-3、2.941e-4 在抽樣誤差內一致。

**C2 模式 B**（完美網路，`floor_out.txt`）

| 設定 | t=1 Δ單步 / Δ多步 / δ | t=2 | t=5 | t=10 | t=18 |
| --- | --- | --- | --- | --- | --- |
| cosine, 20, s²=0.25 | 0.0089 / 0.4347 / 0.4263 | 0.0258 / 0.4353 / 0.3260 | 0.0614 / 0.4032 / 0.1469 | 0.0536 / 0.2444 / 0.0257 | 0.0062 / 0.0091 / 0.0000 |
| cosine, 20, s²=0.5 | 0.0091 / 0.8892 / 0.8772 | 0.0274 / 0.8863 / 0.7235 | 0.0792 / 0.8104 / 0.3917 | 0.0916 / 0.4886 / 0.0867 | 0.0124 / 0.0183 / 0.0001 |
| linear, 20, s²=0.25 | 0.0303 / 0.4005 / 0.3880 | 0.0765 / 0.3966 / 0.2238 | 0.0874 / 0.2735 / 0.0375 | 0.0172 / 0.0434 / 0.0005 | 0.0001 / 0.0001 / 0.0000 |

cosine、$\sigma_0^2=0.25$：$T'=50$ 時 $t=1$ 的 $\Delta'=0.4683$；$T'=100$ 時 0.4826。參考線 $2\bar\alpha_ts^2$ 在 $t=2,10,18$ 為 0.4954、0.2672、0.0133。

**C3**（`eq10_out.txt`；容許 ±1e-6）

| T′ | h | w | λ₆ | λ₁₃ | λ₁₉ | λ₂₀ | 回代誤差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.02 | 0 | 1.000178 | 1.000178 | 1.000178 | 1.000089 | <1e-14 |
| 20 | 0.02 | 1 | 1.000498 | 1.000249 | 1.000036 | 1.000006 | <1e-14 |
| 20 | 0.02 | 0.5 | 1.000338 | 1.000213 | 1.000107 | 1.000047 | <1e-14 |
| 20 | 0.1 | 0 | 1.000889 | 1.000889 | 1.000889 | 1.000444 | <1e-14 |

**C4 與內文算術**（`check_out.txt`）：FFHQ 改善 7.75、16.65、33.21（ES）、34.52（ES\*）；CelebA DDPM 改善率 47.7%、63.1%、60.7%；511/100 = 5.11、2047/100 = 20.47；A.10 規則 $b^*-9.5k$：LSUN 1.0034、FFHQ 1.0042、IN64 1.0041、IN128 1.00291（皆等於表值），CIFAR-10 0.99325（表值 1.0）。

**其他內文數字**：Table 7 交叉項（linear、$T'=20$、$e=0.3$）$t=5$：MC 0.25333、精確 0.25347、Eq. 23 0.24990；$t=10$：0.83822、0.84098、0.84002。高斯玩具中 λ 的效果（cosine、20 步、$\sigma_0^2=0.25$）：$\lambda=0.98,1,1.017,1.05$ → $\delta_1=0.757,0.426,0.887,6.82$（只寫在設計文件與 R3，不上頁面）。

---

## 6. 實作計畫

### 6.1 檔案與共用外殼

- 單一檔案 `tut/406/build/index.html`；CSS、JS、時間線資料全部內嵌，不 `fetch` 本機檔案。上線時複製為 `tutorials/exposure-bias-epsilon-scaling.html`。
- **直接沿用** `tutorials/prediction-targets-k-diff.html` 的外殼，不重寫：CSS tokens（含深色模式三段覆寫、`pm-theme`）、字型載入、appbar、`#pm-bar` 導覽、章節目錄（sticky 左欄／窄螢幕橫列）、`.plate` 圖版、callout、`<details>` 檢查題、「本頁推導」pill、表格橫捲容器、KaTeX 0.16.9 設定與 `renderMath()`、`?test` selfTest 框架（`selfTest()` 的 `ok()` 計數與 `#selftest` 輸出、`scrollWidth <= innerWidth` 檢查）、時間線卡片列元件與 SVG 輔助函式（`el`、`scaleLin`、`scaleLog`、`linePath`）。
- 以下只列新增或變更的部分。

### 6.2 新增 tokens 與連結

- 圖表色：沿用 `--accent` 作為「取樣／差距／ES」；新增 `--c-train`（light #3F6E8C／dark #8DB4D3，沿用 k-diff 頁 `--c-par` 的色值，改名以符合本頁語意），用於「訓練」系列。建置時確認在 `--surface` 上圖形對比 ≥ 3:1、文字 ≥ 4.5:1。
- 新增「推定」pill（與「本頁推導」同樣式，文字不同）、「作者認為」的引述樣式（左側 2px `--line` 線＋`--ink-2`）、「進階」框（可收合 `<details class="advanced">`，摘要列「進階：本頁推導，可略過」）。
- 「本章符號」小表：每章開頭一個可收合 `<details>`，列該章新增符號（第 1.4 節符號表的第三欄）。
- 站內連結：Hero「在關聯圖中查看」→ `../index.html#paper-406`；arXiv `https://arxiv.org/abs/2308.15321`；程式碼 `https://github.com/forever208/ADM-ES`；筆記 → `<slug>.html`。

### 6.3 程式結構（單一 IIFE）

- 共用數學：`abarFull(kind)`（快取兩種 1000 步排程）、`respace(kind, Tp)` → `Float64Array A`（`A[0]=1`）、`coefs(A, t)` → `{alpha, beta, q, c1, c2, bt}`。
- C1：`initC1()`，`state={sampler,kind,Tp,e,t,rho}`，`c1Calc(state)` 回傳整條曲線與選定點讀數。
- C2：`initC2()`，`state={mode,kind,Tp,e,s2,t,showFig1}`；`c2Paper(A,e)` → `{single,multi,delta}`；`c2Perfect(A,s2)` → `{single,multi,delta}`（陣列以顯示的 $t$ 為索引，4.0 對照表；與 `check.py`／`floor.py` 逐式相同）。`FIG1_READ=[[2,.47],[10,.265],[18,.015]]`。
- C3：`initC3()`，`eq10Invert(h,w)`、`eq10Forward(lam)`（$T'=20$ 固定）。
- C4：`initC4()`，`RESULTS` 常數（第 4 節 C4 的資料結構與資料），`renderBars()`／`renderSens()`／`renderTable()`。
- 共用：`schedule(fn)`（rAF 合併重畫）、`debounce(fn,500)`（live region／aria-label）、`radiogroup(el, onChange)`（roving tabindex＋方向鍵）。
- 時間線：沿用卡片列元件，新的 `TL_GROUP`（6.4）；`year` 在 JSON 中是字串，排序用 `Number(year)`。
- `selfTest()` 新增斷言（容許誤差見第 5 節）：
  - `coefs` 恆等式：對 cosine/linear、$T'=20$ 的每個 $t$，$c_{2,t}^2(1-\bar\alpha_t)+\tilde\beta_t=1-\bar\alpha_{t-1}$（1e-12）。
  - C1：cosine 20 步 $e=0.1$，$t=1$ 差距 9.910e-3、DDIM 8.707e-3；$t=10$ 差距 4.239e-4。
  - C2 A：cosine 20 步 $e=0.1$，顯示 $t=2$ 的 Δ多步 = 5.313e-3、$t=1$ 的 $\delta=8.711\text{e-}3$；B：$\sigma_0^2=0.25$ 的 Δ多步(2) = 0.4353、$\delta_1=0.4263$。
  - C3：$T'=20,h=0.02,w=0$ 的 $\lambda_6=1.000178$、回代誤差 < 1e-12。
  - C4：FFHQ 20 步 ES\* 改善 34.52；CelebA DDPM 20 步改善率 47.7。

### 6.4 研究筆記與時間線

- 來源 `tut/406/notes/timeline.json`（13 筆，欄位 `year`（字串）、`title`、`authors_short`、`arxiv`、`role`、`source`），原樣貼進 `<script type="application/json" id="timeline-data">`。
- `TL_GROUP`（以 arXiv ID 為 key；不在表中歸「其他」）：

  | 類別 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | 擴散模型基礎 | 1503.03585、1907.05600、2006.11239、2010.02502 | `ddpm-reverse-sampling-formulation.html` |
  | 曝光偏差的起源 | 1511.06732、1910.00292 | `exposure-bias-seq2seq-origin.html` |
  | 取樣變異數 | 2102.09672、2201.06503、2206.00364 | `diffusion-sampling-variance-schedule.html` |
  | 既有緩解方法 | 2301.11706、2305.15583、2308.05021 | `exposure-bias-prior-mitigations.html` |
  | 本文 | 2308.15321 | `epsilon-scaling-method.html` |

- S13 延伸閱讀（依總覽閱讀順序）：`exposure-bias-elucidation-overview`、`exposure-bias-seq2seq-origin`、`ddpm-reverse-sampling-formulation`、`diffusion-sampling-variance-schedule`、`exposure-bias-prior-mitigations`、`prediction-error-variance-analysis`、`epsilon-scaling-method`；另列站內既有 `eps-prediction-ddpm-score`、`diffusion-forcing`，以及 No. 405 的 `diffusion-exposure-bias`、`input-perturbation-ddpm-ip`（已確認位於 `tut/405/notes/`）。每項附 frontmatter 的 `title`。

### 6.5 文案與排版規則（新增部分）

- 用詞：「曝光偏差（exposure bias）」「取樣分佈偏離訓練分佈（論文稱 sampling drift，只在引述時附原文）」「縮放係數 λ」「變異數誤差」「常見錯誤」。第一次出現的專有名詞附英文。
- 三種陳述分開標示：論文結果（附出處）、「作者認為」、〔本頁推導〕；讀圖或校準值加「推定」pill。
- 公式旁標論文編號；Eq. 9、Eq. 10、ES 公式在 < 600px 時拆行。SVG 內文字用 Unicode（「1−ᾱₜ」「Δ′ₜ」「λ − 1」）。
- 第 1 節以外不得新增論文數字；頁面數字須與第 1、5 節一致。

### 6.6 建置後自我檢查

依 k-diff 頁的流程：390／768／1024／1440 截圖＋深色、鍵盤 focus、reduced-motion；逐項評分（字體、留白、視覺層級、色彩、動態、微互動、響應式、原創性）並修正；`?test` 全數通過；逐條核對 12 個學習目標的答案出現在指定章節內文。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 本頁推導的份量 | 數項〔本頁推導〕與論文的敘述不同：Table 7 少了一階項、Eq. 7 模型末段不累積、完美網路下 Alg. 2/3 的讀數不為 0、Eq. 10 二次 ↔ 常數、Table 14 CIFAR 列不符 A.10、FFHQ 34.52 來自 ES\*。頁面可能讀起來像批評。 | 每項標「本頁推導」並附依據；LO7、LO9 放進「進階」框、不屬核心驗收；主線依論文順序講；只陳述可驗證的計算並加限定語。 |
| R2 | C2 B 模式的校準 | cosine 排程與 $\mathrm{Var}(x_0)=0.25$ 是推定；高斯資料不是影像；玩具的單步誤差（中段 0.06）大於 Fig. 1。Fig. 1 讀圖值 ±0.02；Fig. 8 的 $\delta_1$（0.39–0.465）部分低於玩具的 0.426。 | 一律稱「玩具估計」，明寫不是下限；以 Fig. 8 佐證同一量級；頁面只說「Fig. 1 的多步曲線中有一部分即使網路完美也會出現」。 |
| R3 | 不以玩具示範 ES 的效益 | Eq. 7 模型中 λ>1 只是縮小最後一步的 $x_0$ 誤差（$\delta_1$ 隨 λ 單調下降，與 Table 17 的 U 形不符）；高斯完美網路中 λ>1 使 $\delta_1$ 變大（0.426 → 0.887）。 | C2 不加 λ；S7 一句說明 ES 的效益是真實網路上的實驗結果；ES 的數字只在 C4。 |
| R4 | Eq. 10 的量級 | 反解 λ−1 ≈ 1.8e-4（隨 h 的讀圖誤差變動約 1.5 倍）與搜尋值 0.017 相差很大；兩者不是同一個量。 | S8 開頭先說明兩者的差別；C3 只教形狀，量級只在圖下一句；不進 LO11 答案。 |
| R5 | 線性排程的 t 索引 | 論文未寫明；由 Table 14 四列的算術推定為 0..T′−1。 | 常見錯誤第 4 條寫「推定」；C3 不使用 k 的實際值。 |
| R6 | 讀圖值 | Fig. 1（2, 0.47）（10, 0.265）（18, 0.015）、Fig. 3（左端約 1.19、t = 5 約 1.02–1.03，±0.01）、Fig. 8（0.39–0.465）皆為讀圖。 | 一律標「推定」；其他圖只做文字描述。 |
| R7 | 跨教學的連結 | 已解決：`diffusion-exposure-bias`、`input-perturbation-ddpm-ip` 位於 `tut/405/notes/`。 | 建置時確認上線檔名一致。 |
| R8 | DDPM-IP 的提及範圍 | 本頁與 No. 405、No. 237 一起製作，之後另有綜合教學。 | DDPM-IP 只出現在 Sec. 2、3.4、Table 3 的脈絡（S1 延伸閱讀、S7 一句、S9 文字中 ADM-IP 2.38）。不比較兩者機制。 |
| R9 | 相對連結 | 建置位置與上線位置不同，預覽時站內連結失效。 | 接受；預覽只檢查外部連結與頁內錨點。 |
| R10 | 初學者門檻 | 讀者不熟 DDPM；S2 公式多。 | S1 不寫公式；S2 每式後一句白話；各章「本章符號」小表；頁面記號 ε_train／ε_samp、Δ單步／Δ多步、$V_t$ 取代相近的論文記號。 |
| R11 | Fig. 4 的資料集 | Fig. 4／5 圖說為 LSUN tower 64，正文寫 CIFAR-10。 | S9 依圖說寫 LSUN tower 64，CIFAR-10 指向 Fig. 9；不混用不同資料集的讀值。 |

---

## 8. 修訂紀錄（round 1）

| 審查項 | 變更 |
| --- | --- |
| 1 | 「量測下限」改為「完美網路下的玩具估計，不是下限」（步驟 5、6、12，LO9，C2，R2）；以 Fig. 8 的 0.39–0.465 佐證量級。 |
| 2 | C3 移除 $T'=50$（Fig. 3 兩曲線重疊）；Fig. 3 讀值改為 ±0.01；S8 開頭與步驟 9 寫明「Eq. 10 的 λ 與搜尋的 b 不是同一個量」；量級比較移到 C3 圖下一句，移出 LO11。 |
| 3 | S1 不寫公式（ᾱ 只用文字）；Eq. 1 移到 S2；ΔN 移出 C2，改在 S7 以正文數字呈現。 |
| 4 | C2 模式 B 標「進階」，預設 A；S6 進階框以一段內文教高斯資料的最佳去噪器 $k_tx_t$ 與「只需追蹤兩個數」；遞迴式放元件 `<details>`；LO9 只問「是否為 0、為什麼」。 |
| 5 | 頁面記號：ε_train／ε_samp、Δ單步／Δ多步、$V_t$（取代 β′、不顯示 β̂）；符號表加「頁面寫法／首次出現」欄，各章開頭「本章符號」小表；4.0 新增索引對照表，驗收表以顯示的 $t$ 為準。 |
| 6 | S3 拆成 S3（單步差距＋C1）與 S4（噪聲排程，作者論證）；A.10 建議與 Table 17／18 移到 S11；S9 正文只寫 Table 2–5，其餘交給 C4。章節重新編號為 S0–S13。 |
| 7 | LO 表加「來源」欄；LO7、LO9 改為「進階（本頁推導）」並放進進階框；LO11 核心答案改為論文內容，本頁推導附一行證明（$\Delta N''=\lambda-1$）寫在正文。 |
| 8 | 4.5：動態文字一律 Unicode、KaTeX 每容器渲染一次、`aria-label` 純文字；< 600px 單欄、SVG 寬度隨容器、文字固定 12–13px、讀數改 `<dl>`；C4 規定 `RESULTS` 結構（`b` 可為 null、`esStar` 選填）、每子集縱軸從 0 起自動縮放、敏感度圖註明不從 0 起。 |
| 9 | Fig. 1（10, 0.30）→（10, 0.265），（18, 0.01）→（18, 0.015）；Fig. 3 讀值加 ±0.01；Eq. 7 的 ΔN 改寫為「只重現方向，1.41 是 $c_{1,1}=1$ 的結果」；步驟 5 註明論文沒有宣稱 Fig. 1 可由 Eq. 7 推得。 |
| 10 | C4 觀察改為「都低於基準」，Table 18 35 步只差 0.02；註明掃描範圍 ±0.002、「不敏感」是作者說法（步驟 9、C4）。 |
| 11 | LO5 與步驟 4 改為「作者的論證」；移除 C1 的 ρ 滑桿（無法檢驗論證前提），β̇ 式只留在 S4 正文；步驟 2 加不可約誤差一句；C1 的 e 標「示意值」並說明 t 依賴來自 $c_1$；EDM VP 改為 $(x/s(t)-D_\theta(x/s(t);\sigma))/\sigma$。 |
| 12 | 文案改字面用語（第 3 節末）；4.5：圖上不做拖曳（只用滑桿）、radiogroup roving tabindex＋方向鍵、pattern 對比在深色模式驗證、reduced-motion 範圍明列；C1 對數軸夾值以空心點標示、$e=0$ 時兩條曲線都不畫；`input` 以 rAF 合併、只重畫變動元素。 |
| 其他 P2 | fidelity 11：S9 不混用 LSUN 與 CIFAR-10 讀值（步驟 10、R11）；fidelity 12：步驟 5 加註。 |

## 9. 修訂紀錄（readiness）

| 項 | 變更 |
| --- | --- |
| B1 | C4 預設：ADM（Table 2）、CIFAR-10、未勾選「以表格顯示」。 |
| B2 | §5 容許誤差改為相對 1e-3 ＋ 絕對下限 1e-12，適用所有讀數。 |
| B3 | ΔN 操作定義為均方範數比的平方根，寫出兩個期望（步驟 8）。 |
| B4 | $V_t$ 在 S5 的本章符號中定義；LO7 答案寫在 S5 正文。 |
| N1–N5 | C1 $t=1$ 的 1−ᾱ₁ 顯示 4.128e-5（readiness 寫 4.130e-5，實算 4.1284e-5）；C2 模式 B 的資料變異數改記為 $\sigma_0^2$（`v0`）；步驟 6 改用 $V_t$；Ranzato 標「arXiv 2015、論文引為 2016」；LO7 答案移出收合框。 |

## 10. 建置紀錄（build round）

- `build/index.html` 由 `scratchpad/tp406/build.py` 組合：CSS 外殼逐字取自 k-diff 頁，內容與元件為本頁新寫。`?test`：184 passed、0 failed（1440 與 390 寬皆同；含 §5 全部驗收表、容許誤差 rel 1e-3 + abs 1e-12、無水平捲動）。impeccable：0 anti-patterns（只剩 13 則 numbered-section-labels advisory）。
- 與設計的差異：讀數一律用兩欄 `<dl>`（不在 ≥600px 改為表格）；C1 的 1−ᾱ₁ 實算為 4.128e-5（readiness 的 4.130e-5 為近似，240 倍不變）；Hero 的兩條管線圖在窄螢幕以 `overflow-x:auto`、`min-width:640px` 保持字級；C4 窄螢幕長條圖省略 b 值，改在圖下說明並以表格模式提供；收合框分三類：`details.check`（檢查題）、`details.sym`（本章符號）、`details.adv`／`details.calc`（進階、計算方式），只有前者被驗收腳本剝除。
- 截圖：`build/caps/`（1440 的 C1–C4 與 hero、390 的 hero／C1／C2／C4／λ 章、深色 C2）。第一輪發現 CSS 開頭多一個 `<style>` 字串使 `:root` tokens 失效，已修正。
