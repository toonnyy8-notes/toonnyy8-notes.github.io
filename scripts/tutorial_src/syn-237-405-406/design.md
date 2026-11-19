# 設計文件：綜合教學〈分數誤差、曝光偏差與三種縮放〉（No. 237 × No. 405 × No. 406）

- 三篇論文（原文 `tut/<編號>/src/paper.txt`；`tut/` = `scripts/tutorial_src/`）：
  - No. 237：Li、Shen、Hsieh、He，*When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis*，arXiv 2509.24912v2。教學 `tutorials/scores-learn-geometry.html`。
  - No. 405：Ning、Sangineto、Porrello、Calderara、Cucchiara，*Input Perturbation Reduces Exposure Bias in Diffusion Models*（DDPM-IP），arXiv 2301.11706（ICML 2023）。教學 `tutorials/ddpm-input-perturbation.html`。
  - No. 406：Ning、Li、Su、Salah、Ertugrul，*Elucidating the Exposure Bias in Diffusion Models*（Epsilon Scaling），arXiv 2308.15321（ICLR 2024）。教學 `tutorials/exposure-bias-epsilon-scaling.html`。
- 「237 Eq. n」「405 Tab. n」「406 Sec. n」皆指各論文原文編號；L 後的數字是 `paper.txt` 行號。三篇的 `design.md` 只用來定位，每條論文陳述都已回原文核對。
- 產出：`tut/syn-237-405-406/build/index.html`，上線 `tutorials/score-error-exposure-bias-synthesis.html`（slug `score-error-exposure-bias-synthesis`）。
- 驗算：`verify/core.js`（頁面數值核心，建置時逐字複製；依賴 `tut/237/verify/core.js`，兩檔依序貼入）、`verify/check.js`（第 5 節所有數字，輸出 `check_out.txt`；`node check.js`，約 2 秒）。
- 標記（頁面同樣使用）：**論文**（附出處）；**〔本頁推導〕**＝本頁自行推導、三篇都沒寫；**〔本頁詮釋〕**＝解讀或對應，不是推導；**〔本頁玩具〕**＝教學用簡化模型；**推定**＝論文沒寫、本頁選定的值。後三者絕不寫成任何一篇論文的主張。
- 讀者假設：三篇教學都沒讀過，只有基本機率與神經網路知識（審查 #1）。

---

## 1. 介紹（步驟 6-1；本節是給設計者的背景，頁面文案依第 3 節的定義順序）

### 1.1 三篇各一段

**No. 237。** 在流形假設（資料支撐在 $\mathbb R^d$ 中的 $n$ 維緊緻流形 $M$）下，$\log p_\sigma(x)=-d_M(x)/\sigma^2+\log p_{data}(\Phi^{-1}(P_M(x)))-\tfrac{d-n}{2}\log2\pi\sigma^2+H(x)+o(1)$（237 Eq. 6，L292）：幾何在 $\sigma^{-2}$ 階，資料密度只在 Θ(1) 階。Thm 4.1（L347–357；Assumptions 2.1、2.2、4.1）：生成分佈 $\pi_\sigma$ 的分數誤差 $E_\sigma=\|s_{\pi_\sigma}-s^*\|_{L^\infty(K)}$ 為 $o(\sigma^{-2})$ **足以**集中到 $M$，$o(1)$ **足以**還原 $p_{data}$，並可構造 $\Omega(1)$ 誤差使極限為 $M$ 上任意分佈。把 Langevin 的分數乘上 $\sigma^\alpha$（Tempered Score，TS，Eq. 8，L385），在誤差 $o(\sigma^\beta)$、$\beta>-2$、$\max\{-\beta,0\}<\alpha<2$ 下平穩分佈收斂到流形上對黎曼體積的均勻分佈（Thm 5.1；Thm 5.2 另需 $p_{data}\in C^2$ 與 WKB 假設）。SD 1.5 實驗只改 Predictor–Corrector 的 corrector 步，把**無條件** ε 乘 $\sigma_t^\alpha$，CFG 項（w = 7.5）不變（Sec. 7.2 L549–552；App. C.2）。

**No. 405。** 訓練時網路輸入是真實的 $x_t$，取樣時是自己產生的 $\hat x_t$；作者稱為曝光偏差，並在 **ImageNet 32×32** 上以「從真實 $x_t$ 跑 $t$ 步」的 FID 量化（Tab. 1：100 步 0.983 → 1,000 步 3.544）。DDPM-IP：訓練輸入 $y_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}(\epsilon+\gamma\xi)$（Eq. 6），目標仍是 $\epsilon$，$\gamma=0.1$，推論不變（Alg. 3）。對照組 DDPM-y（Eq. 8）用同一個 $y_t$、目標換成 $\epsilon'$，結果比 DDPM 還差（Sec. 5.2 L281–288、App. A.3、Tab. 7：CIFAR10 100 步 DDPM 3.37、DDPM-y 3.90、DDPM-IP 2.70）。作者把 IP 的效果歸因於平滑預測函數的正則化（Sec. 5.2 L293–299、Sec. 5.4）。γ 的搜尋範圍取自沿取樣軌跡實測的 $x_0$ 預測誤差標準差 $\nu_t$：「the mean of the blue and the orange curve is 0.20 and 0.19」，範圍 $[0,\mathbb E_t\nu_t]=[0,0.2]$，作者稱此範圍「covering the last half of the sampling trajectory」（Sec. 5.3 L282–296、L320–347）。

**No. 406。** 沿用 Analytic-DPM，把 $x_0$ 預測**建模**為 $x_\theta^t=x_0+e_t\epsilon_0$（Eq. 7，未報 $e_t$ 數值），推得單步取樣分佈變異數多 $(c_{1,t+1}e_{t+1})^2$（Eq. 9、Table 1）；度量 $\delta_t$（Sec. 3.3）；取樣時 $\|\epsilon_\theta\|$ 每一步都比訓練情境大（Fig. 2，L380–399），於是取樣時把 ε 除以 $\lambda_t$（ES，Sec. 4.1，L402–406），噪聲項不變。多數實驗用常數 $\lambda=b$；也有線性 $\lambda(t)=kt+b$（ADM-ES*，Table 2、14）。ADM 的 $b$：CIFAR-10 三種步數都是 1.017，其他資料集 1.004–1.011（Table 14，L1515–1524）；作者稱最佳 $b$ 一般隨 $T'$ 增加而下降（App. A.10）。**CIFAR-10** 100 步 FID 3.37 → 2.17（Table 2）。Sec. 3.4 把 DDPM-IP 列為資料增強類解法（L340–342）；Table 3（標題「best FID achieved under T′ sampling steps」）列「ADM-IP (Ning et al., 2023) 100 2.38」（L492）。

### 1.2 關係地圖

| # | 關係 | 狀態 | 方程式與數字 |
| --- | --- | --- | --- |
| R0 | 三篇之間唯一由論文寫出的關係：406 引用 405 | 論文 | 406 Sec. 3.4（L340–342）、Table 3（L492）。237 沒有提到曝光偏差；405、406 沒有提到流形（`grep` 三份 paper.txt） |
| R1 | 三篇都在處理「取樣器需要的分數／ε」與「實際送進取樣器的值」的差，且都特別處理噪音小的末段 | 各自內容為論文；連結為〔本頁詮釋〕 | 237 Sec. 1 L39–46（低溫分數最有價值也最難估計）；405 Sec. 5.3（「last half of the sampling trajectory」用來界定 $\nu_t$ 的取值範圍 [0, 0.2]，405 沒有分析小 σ）；406 Sec. 4.2 L429–433（t 接近 0 時 ε 預測很差）、Fig. 1（多步誤差越近 $t=1$ 越大） |
| R1b | 誤差單位換算 $\Delta s=-\Delta\epsilon/\sigma_t=\sqrt{\bar\alpha_t}\,\Delta x_0/\sigma_t^2$ | 〔本頁推導〕 | 237 的充分條件換成 $x_0$ 單位：$x_0$ 誤差 $o(1)$ ⇒ 集中；$o(\sigma^2)$ ⇒ 還原 $p_{data}$（只在「生成分佈的分數＝網路分數」時適用，D1） |
| R1c | 237 的誤差是最終生成分佈的分數誤差；曝光偏差是產生這個誤差的一種機制，237 沒有模型化 | 論文（237 Limitations a，L603–605）＋〔本頁詮釋〕 | — |
| R2 | TS 與 ES 都是取樣時的分數乘數：TS 對無條件分數乘 $\sigma^\alpha$，常數 ES 對整個 ε 乘 $1/\lambda$ | 定義為論文；同軸比較為〔本頁推導〕 | 交點 $\sigma^*=\lambda^{-1/\alpha}$（α = 1、λ = 1.017：0.9833）；σ = 0.1：TS 0.1、ES 0.9833；常數 ES 的 $\alpha_{eq}=\ln\lambda/(-\ln\sigma)$，σ → 0 時 → 0（線性 λ(t) 另論） |
| R2b | 作用位置不同：TS 在 corrector（Langevin）步；ES 在 predictor（反向步）平均中的 ε | 論文 | 237 L549–552、App. C.2；406 L402–406 |
| R2c | 精確分數、梯度場、Langevin 跑到平穩時，乘常數 $c$ 的平穩分佈 $\propto p_\sigma^c$，σ → 0 時在 $M$ 上 $\propto p_{vol}^{\,c}$；$c=\sigma^\alpha$ 即 Thm 5.1 的均勻 | 〔本頁推導〕（237 Thm B.1，假設同 Thm 4.1(3)） | 單位圓、κ = 4、$c=1/1.017$：σ = 0.01 時 TV($p_{data}$) = 0.0046，最大／最小比 2976 → 2604 |
| R2d | 精確分數下，λ > 1 使輸出變寬（$p^c$ 較平）；406 的理由是修正 $\hat x_t$ 上過長的 ε。玩具在 406 效果最明顯的 $T'=20$ 不成立 | 〔本頁玩具〕＋〔本頁詮釋〕 | 高斯、完美網路、cosine、λ = 1.017：$\mathrm{Var}(\hat x_0)$ $T'=100$ 0.2326 → 0.2417；$T'=20$ 0.185 → 0.649（λ = 0.965 也變寬到 0.968） |
| R3 | DDPM-IP 的**貝氏最佳**網路等於噪聲大 $\sqrt{1+\gamma^2}$ 倍的精確分數；DDPM-y 的最佳網路只差一個 $\sqrt{1+\gamma^2}$ 倍 | 〔本頁推導〕（一般資料分佈；Monte Carlo 驗證） | $\epsilon^*_{IP}=-\sigma_t\nabla\log\tilde p_t$，$\sigma'=1.005\sigma$；$\epsilon^*_{IP}=\epsilon^*_{y}/\sqrt{1+\gamma^2}$（差 0.5%），但 Tab. 7 中 IP 2.70、DDPM-y 3.90：最佳網路的等式**解釋不了** IP 的改善 |
| R3b | 405 的 $\nu_t$ 與 406 的 $e_t$ 定義相同（$\hat x_0-x_0$ 的標準差），來源不同：前者沿取樣軌跡實測，後者是建模參數 | 〔本頁詮釋〕 | 405 Sec. 5.3 L334–339；406 Eq. 7 |
| R3c | 在 406 的模型下，使 IP 訓練輸入與取樣分佈一致的 γ 隨 t 變化數個數量級；單步版是 $c_{1,t+1}e_{t+1}/\sigma_t$，不是 405 所說的 $\gamma_t=\nu_t$ | 〔本頁推導〕；數值依推定的 e、排程、$T'$ | 推定 cosine、$T'=100$、常數 e = 0.2：多步 γ 在 [0.05, 0.2] 只有 $t=16..39$；$t=1$ 的 29.4 是「e 為常數」的假設造成 |
| R4 | 流形觀點能說明：法向誤差被 $\sigma^{-2}$ 放大、三種修正的極限不同。不能說明：有限步、軌跡累積、FID、λ 與 γ 的值 | 〔本頁詮釋〕，邊界依 237 Limitations a–e（L603–612） | — |

不相關之處（頁面明寫）：(1) TS 的目標是流形上的均勻分佈，IP、ES 的目標是 $p_{data}$；(2) 評估不可互比：237 用 CLIP P-sim／I-sim（SD 1.5），405／406 用 FID（405 Tab. 1 為 ImageNet 32×32，406 的 3.37 → 2.17 為 CIFAR-10）；(3) 237 沒有定理涉及有限步反向擴散，405／406 沒有漸近分析；(4) 406 Table 3 把 2.38 列為 Ning et al. 2023 的 best FID，但 2.38 不在 405 的任何表中（405 Tab. 3 CIFAR10 ADM-IP：1,000／300／100／80 步為 2.76／2.67／2.70／2.93），γ 與權重來源未寫。

### 1.3 推導細節（頁面以可收合區塊呈現）

**D1（R1b）。** VP：$x_t=\sqrt{\bar\alpha_t}x_0+\sigma_t\epsilon$，$s=-\epsilon/\sigma_t$，$\hat x_0=(x_t-\sigma_t\hat\epsilon)/\sqrt{\bar\alpha_t}$，得 $\Delta s=-\Delta\epsilon/\sigma_t=\sqrt{\bar\alpha_t}\Delta\hat x_0/\sigma_t^2$。σ → 0 時 $\bar\alpha_t\to1$：分數誤差 $o(\sigma^\beta)$ ⇔ ε 誤差 $o(\sigma^{\beta+1})$ ⇔ $x_0$ 誤差 $o(\sigma^{\beta+2})$。Thm 4.1 的 (1)(3) 是充分條件（「若…則…」），所以換算後也只能寫「⇒」；D2 的常數乘數例子就是誤差為 $\Theta(\sigma^{-2})$ 仍集中的情形。限定：$E_\sigma$ 是生成分佈 $\pi_\sigma$ 的分數，等於網路輸出只在「網路就是 Langevin 漂移的梯度場」這類設定；前提還有 Assumptions 2.1、2.2、4.1 與 $L^\infty(K)$ 範數。405 的 $\nu_t$ 平均 0.2 是對 t 平均，沒有 σ → 0 的值，頁面不據此判定階數。

**D2（R2c）。** 漂移 $c\nabla\log p_\sigma$、擴散 $\sqrt2$ 的 Langevin 平穩分佈 $\propto p_\sigma^c$。代入 237 Eq. 6，取 $\theta=\sigma^2/c$：$f_0=d_M$，$f_1=c(-\log p_{data}+\log\sqrt{|\hat H|})$。237 Thm B.1：$\pi(u)\propto e^{-f_1(u,0)}|\partial_r^2f_0|^{-1/2}\tfrac{dM}{du}=p_{data}^c(\tfrac{dM}{du})^{1-c}$，即對體積的密度 $p_{vol}=p_{data}/(dM/du)$ 變成 $p_{vol}^c$。$c=1$ 回到 Thm 4.1(3)；$c=\sigma^\alpha\to0$ 時回到 Thm 5.1。前提：237 的假設（Assumptions 2.1、2.2、4.1，管外質量估計同 Lemma B.4）、精確分數、梯度場、連續時間、跑到平穩。這**不是** ES 的設定：ES 在有限步反向過程中只縮放平均裡的 ε 項（406 Sec. 4.1）。常數乘數相當於誤差 $(c-1)s^*$，在離 $M$ 固定距離處是 $\Theta(\sigma^{-2})$，不在 Thm 4.1 的充分條件內；集中性來自梯度結構。

**D3（R2d）。** 一維高斯 $\mathcal N(0,s^2)$、精確 $\epsilon^*(x)=\sigma_tx/(\bar\alpha_ts^2+\sigma_t^2)$、DDPM 祖先取樣把 ε 除以 λ；每步線性，變異數遞迴 $v\leftarrow(c_{1,t}g_t+c_{2,t})^2v+\tilde\beta_t$。$T'=20$ 時，接近 $T'$ 的步驟 $\hat x_0=(x-\sigma\epsilon/\lambda)/\sqrt{\bar\alpha}$ 除以很小的 $\sqrt{\bar\alpha}$，λ 的任何偏離都被放大，λ > 1 與 λ < 1 都使輸出大幅變寬。406 在 $T'=20$ 量到最大的 FID 改善（CIFAR-10 10.36 → 5.15），所以這個玩具不能解釋 ES 的效果；ADM 是否截斷 $\hat x_0$ 論文未寫。406 design 步驟 8 的同類玩具（$\delta_1$：0.426 → 0.887）結論一致。

**D4（R3）。** 令 $n=\epsilon+\gamma\xi$、$\sigma'=\sigma_t\sqrt{1+\gamma^2}$。給定 $n$ 時 $\epsilon$ 與 $x_0$ 獨立，$\mathbb E[\epsilon\mid n]=n/(1+\gamma^2)$，所以 $\mathbb E[\epsilon\mid y]=\mathbb E[n\mid y]/(1+\gamma^2)$。$y=\sqrt{\bar\alpha_t}x_0+\sigma'z'$（$z'=n/\sqrt{1+\gamma^2}$），Tweedie：$\mathbb E[\sigma'z'\mid y]=-\sigma'^2\nabla\log\tilde p_t(y)$。合併得 $\mathbb E[\epsilon\mid y]=-\sigma_t\nabla\log\tilde p_t(y)$，標準取樣器換成分數為 $\nabla\log\tilde p_t$。DDPM-y 的目標 $\epsilon'=z'$，最佳網路 $\mathbb E[z'\mid y]=\sqrt{1+\gamma^2}\,\mathbb E[\epsilon\mid y]$。推論：(i) 若取樣時 $\hat x_t\mid x_0\sim\mathcal N(\sqrt{\bar\alpha_t}x_0,V_t)$（406 Eq. 7 模型的結論），取 $\gamma_t^2=V_t/\sigma_t^2-1$ 時 IP 的最佳網路輸出 $\hat x_t$ 所在分佈的精確分數；(ii) 近流形 $s^*\approx-(x-P_M(x))/\sigma^2$（237 Eq. 20 首項），法向分量乘 $1/(1+\gamma^2)$、切向約不變；(iii) $\tilde p_t\to p_{data}$，Langevin 極限不變。限制（頁面正文寫，不放可收合區）：這些都是關於**貝氏最佳**網路；IP 與 DDPM-y 的最佳網路只差 0.5%，Tab. 7 的 FID 卻差很多（2.70 對 3.90），所以 IP 的改善來自有限網路的訓練行為，405 歸因於平滑／正則化（Sec. 5.2、5.4），本頁的等式不解釋它。

**D5（R3c）。** 406 Eq. 9：單步取樣變異數 $\sigma_t^2+(c_{1,t+1}e_{t+1})^2$；IP 訓練輸入變異數 $\sigma_t^2(1+\gamma^2)$（405 Eq. 7）。相等 ⇔ $\gamma=c_{1,t+1}e_{t+1}/\sigma_t$。多步：406 App. A.1 的恆等式 $c_{2,t}^2\sigma_t^2+\tilde\beta_t=\sigma_{t-1}^2$ 讓多出的變異數 $D_t=V_t-\sigma_t^2$ 直接遞迴 $D_{t-1}=c_{2,t}^2D_t+(c_{1,t}e_t)^2$，$D_{T'}=0$；$\gamma_M=\sqrt{D_t}/\sigma_t$，$\delta_t=(\sqrt{V_t}-\sigma_t)^2=\big(D_t/(\sqrt{\sigma_t^2+D_t}+\sigma_t)\big)^2$（不做相近數相減）。405 L346–347「could … set $\gamma_t=\nu_t$」把 $x_t$ 空間的相對擾動與 $x_0$ 空間的誤差直接對應；兩者只在 $c_{1,t+1}\approx\sigma_t$ 處一致。頁面寫「單位不同」，不寫「論文錯誤」（405 自稱 imitate）。

---

## 2. 學習目標（驗收測試）

參考答案必須出現在指定章節的內文段落。每節末 `<details class="check">` 照搬題目與答案。8 項（審查 #16 合併原 LO3/LO4、LO5/LO6）。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 在分數、ε、$\hat x_0$ 間換算誤差 | VP 中 ε 誤差 η、$x_0$ 誤差 e 各對應多大的分數誤差？237 Thm 4.1 的兩個充分條件換成 $x_0$ 誤差是什麼？這是論文結論還是本頁推導？ | $\eta/\sigma_t$；$\sqrt{\bar\alpha_t}e/\sigma_t^2$。$x_0$ 誤差 $o(1)$ 足以集中到流形、$o(\sigma^2)$ 足以還原 $p_{data}$（只是充分條件；前提為 Assumptions 2.1、2.2、4.1、$L^\infty(K)$ 範數，且誤差指生成分佈的分數）〔本頁推導〕 | S2 |
| LO2 | 分辨三篇在哪裡量誤差 | 237 的 $E_\sigma$、405 Tab. 1、406 的 $\delta_t$ 各量什麼？237 為什麼不能直接描述曝光偏差？ | 237：生成分佈的分數與真實分數的 $L^\infty(K)$ 差，σ → 0；405：ImageNet 32×32 上從真實 $x_t$ 跑 $t$ 步後的 FID；406：多步取樣變異數與訓練變異數的差 $(\sqrt{V_t}-\sigma_t)^2$。237 Limitations a：不追蹤取樣軌跡上的誤差累積 | S3 |
| LO3 | 把 TS 與 ES 放在同一個乘數軸上，並說出作用位置 | σ = 0.1、α = 1、λ = 1.017 時兩者的乘數？在哪個 σ 相等？各改哪一步、哪一項？ | TS 0.1（只乘無條件分數）、ES 0.9833；$\sigma^*=\lambda^{-1/\alpha}=0.9833$〔本頁推導〕。TS：corrector 步，$\sigma_t^\alpha\epsilon(x,t,\varnothing)$，CFG 項不變（237 Sec. 7.2、App. C.2）；ES：predictor 平均中的 ε 除以 $\lambda_t$，噪聲項不變（406 Sec. 4.1） | S4 |
| LO4 | 用平穩分佈比較乘數的效果，並判斷其適用範圍 | 精確分數、梯度場、Langevin 跑到平穩、σ → 0 時，乘常數 $1/\lambda$、乘 $\sigma^\alpha$ 各收斂到什麼？這能說明 ES 的效果嗎？ | $p_{vol}^{1/\lambda}$；流形上均勻（Thm 5.1）〔本頁推導，前提同 237 Thm 4.1(3)〕。不能：ES 是有限步反向過程中的 ε 縮放；高斯玩具在 $T'=20$（406 改善最大處）λ 兩個方向都使輸出大幅變寬；406 的理由是修正 $\hat x_t$ 上過長的 ε，效果是實驗結果 | S5 |
| LO5 | 說出 IP 最佳網路的意義與限制 | DDPM-IP 的貝氏最佳網路經標準取樣器換算後是什麼分數？為什麼這解釋不了 IP 的改善？ | 噪聲標準差 $\sigma_t\sqrt{1+\gamma^2}$（1.005 倍）的精確分數〔本頁推導〕；DDPM-y 的最佳網路只差 $\sqrt{1+\gamma^2}$ 倍（0.5%），但 405 Tab. 7 中 DDPM-y（3.90）比 DDPM（3.37）差、IP 2.70 | S6 |
| LO6 | 對照 γ、$\nu_t$、$e_t$ | $\nu_t$ 與 $e_t$ 有何異同？使 IP 與 406 單步取樣變異數一致的 γ 是什麼？ | 定義相同（$\hat x_0-x_0$ 的標準差），405 沿取樣軌跡實測、406 是未給數值的建模參數〔本頁詮釋〕；$\gamma=c_{1,t+1}e_{t+1}/\sigma_t$〔本頁推導〕。推定常數 e = 0.2、cosine、$T'=100$ 時，多步版只在 $t=16..39$ 落在 [0.05, 0.2] | S7 |
| LO7 | 分辨論文陳述與本頁連結 | 三篇之間有哪些是論文自己寫的關係？ | 只有 406 引用 405：Sec. 3.4 列為資料增強、Table 3 列 ADM-IP 2.38（405 的表中沒有這個值）；237 沒提曝光偏差，405／406 沒提流形 | S1 |
| LO8 | 說出流形觀點的適用邊界 | 237 的理論不能解釋 405／406 的哪些部分？ | 有限步與離散化（Limitations d）、軌跡上的累積（a）、$L^2$ 訓練誤差（b）、FID；也不預測 λ = 1.017 或 γ = 0.1 | S8 |

---

## 3. 頁面大綱

- **Hero**：標題「分數誤差、曝光偏差與三種縮放」，副標「三篇論文的關係：哪裡相連、哪裡不相連」；三張論文卡，每張一句不含符號的主張＋「教學」「論文」連結。
- **S0 最少的定義**（審查 #1）：各兩句、不含證明。去噪擴散：把資料一步步加噪、訓練網路逐步去噪；$\bar\alpha_t$ 是第 t 步保留資料的比例，$\sigma_t=\sqrt{1-\bar\alpha_t}$ 是噪聲標準差（VP）。網路預測噪聲 ε；分數（score）＝ log 密度的梯度，指向密度較高處，與 ε 的關係 $s=-\epsilon/\sigma_t$；也可由 ε 換算出乾淨資料的估計 $\hat x_0$。訓練時網路看到的帶噪樣本記為 $x_t$；取樣時取樣器自己一步步產生的狀態記為 $\hat x_t$。取樣總步數記為 $T'$（訓練用 1,000 步，取樣可以只用 20 或 100 步）。取樣：從純噪聲出發，每步用網路輸出往回走（predictor）；有些取樣器在每步之間再跑幾步「沿分數走一小步並加一點噪聲」的修正（corrector，即 Langevin 動態）。流形：高維空間中的低維曲面，例如平面上的一個圓。
- **S1 三篇論文與關係地圖**：每篇三句白話（問題／修正／證據，無符號），例：「237：資料若落在低維曲面上，分數裡『離曲面多遠』的資訊比『曲面上哪裡比較多資料』強很多；把 corrector 的分數乘上隨噪聲變小的係數，就能得到曲面上的均勻分佈；證據是定理與初步實驗。」各附教學連結。關係地圖改為卡片（關係／狀態 pill／一行數字），< 600px 單欄。「不相關之處」四點。LO7 的答案在本節內文。
- **S2 同一個誤差的三種單位**：D1 的換算鏈；**W0 換算器**；門檻表（分數／ε／$x_0$ × 兩條充分條件，< 600px 轉置為兩張卡）；限定句。LO1。
- **S3 誤差在哪裡量：比較表**（必備）＋「曝光偏差與最終分佈誤差」段（R1c）。LO2。

  | 項目 | No. 237 TS | No. 405 IP | No. 406 ES |
  | --- | --- | --- | --- |
  | 問題 | 小 σ 時幾何與密度所需的分數精度 | 訓練 $x_t$ 與取樣 $\hat x_t$ 不一致 | 同左，量化為變異數差 |
  | 誤差在哪裡量 | 生成分佈的分數與真實分數的最大差距，在包住流形附近的一塊有界區域 K 上取最大（$L^\infty(K)$），σ → 0（Thm 4.1） | 從真實 $x_t$ 跑 $t$ 步的 FID（Tab. 1，ImageNet 32）；沿取樣軌跡實測的 $\hat x_0$ 誤差標準差（S7 記為 $\nu_t$） | 取樣狀態與訓練樣本的變異數差（S7 記為 $\delta_t$）、ε 範數比（Fig. 2–3）、FID |
  | 修正 | 無條件分數 × $\sigma^\alpha$（Eq. 8、App. C.2） | 訓練輸入多加一份噪聲（Eq. 6，S6 介紹），目標不變 | ε 除以略大於 1 的係數（Sec. 4.1，S4 記為 λ） |
  | 訓練或推論 | 推論（corrector） | 訓練（推論不變） | 推論（免重訓） |
  | 縮放的對象 | corrector 中的無條件分數；CFG 項不變 | 網路輸入的噪聲量 | predictor 平均中的 ε；噪聲項不變 |
  | 關鍵數字 | 若分數誤差為 $o(\sigma^\beta)$（β 是誤差階數，β > −2），需 $\max\{-\beta,0\}<\alpha<2$；SD 搜 α ∈ {0.1, 0.5, 1.0, 1.5} | 輸入擾動強度 γ = 0.1（S6 定義）；實測誤差標準差平均 0.20／0.19（S7） | ADM：CIFAR-10 係數 1.017，其他 1.004–1.011；另有隨 t 線性變化的版本 |
  | 目標分佈 | 流形上均勻（黎曼體積） | $p_{data}$ | $p_{data}$ |
  | 證據類型 | 定理（漸近、連續時間）＋初步實驗（CLIP） | 實驗（FID，5 資料集） | 高斯誤差模型分析＋實驗（FID，6 種框架） |

  < 600px 每篇一張卡（8 列）。
- **S4 兩種縮放放在同一軸上**：$m_{TS}=\sigma^\alpha$、$m_{ES}=1/\lambda$、交點、$\alpha_{eq}$（只對常數 ES；線性 λ(t) 的 ADM-ES* 不是單一常數）；作用位置。**W1（IP 線預設隱藏）**。LO3。
- **S5 乘數對平穩分佈與 $\hat x_0$ 變異數的影響**（字面標題，審查 #8）：開頭三個定義段（審查 #3）：(i) Langevin 動態：重複「沿分數走一小步＋加一點噪聲」；(ii) 平穩分佈：跑夠久後樣本分佈不再改變，分數精確時就是原分佈 p；漂移乘常數 c 時變成 $p^c$（一維例：$\mathcal N(0,v)$ 變成 $\mathcal N(0,v/c)$；c < 1 時 $p^c$ 較平，文獻稱 tempering，本頁不用「溫度」一詞）；(iii) TV 距離：兩個分佈差的絕對值積分的一半，0 表示相同、1 表示不重疊。正文先定義 $p_{vol}$：流形上對弧長（一般為黎曼體積）的資料密度，單位圓上即對角度的密度。正文結論：精確分數下三種乘數的 σ → 0 極限（常數 $1/\lambda$：$p_{vol}^{1/\lambda}$；$\sigma^\alpha$：均勻）＋前提一句；**W2**；D3 的玩具結論（含 $T'=20$ 失效）與 406 的相反理由；〔本頁詮釋〕精確分數在較遠的 $\hat x_t$ 上本來就較長，範數比本身不能區分網路誤差與分佈偏移。von Mises、$p_{vol}$、Thm B.1 代入（D2）放可收合「進階」區。LO4。
- **S6 訓練端：IP 的最佳網路與它解釋不了的部分**：先介紹 IP 與 DDPM-y（405 Eq. 6、8、Tab. 7）；D4 結論與限制（正文）、推導（可收合）；Monte Carlo 一行；**此處在 W1 打開 IP 線**（頁內按鈕「在 W1 顯示 IP」捲回 W1 並勾選；W2 的 IP 疊加按鈕也在本節之後才可用）。LO5。
- **S7 γ、$\nu_t$、$e_t$、$\delta_t$**：R3b；D5；**W3**；「數值依推定」「γ = 0.1 只在一段 t 內與模型一致」。LO6。
- **S8 流形觀點能與不能解釋的**：兩欄清單。LO8。
- **S9 常見誤讀**：(1) 把常數 ES 當成 TS 的特例（$\alpha_{eq}\to0$，極限不是均勻）；(2) 把 TS 的保證套到 predictor；(3) 把 γ = 0.1 與 $\nu_t\approx0.2$ 直接比大小；(4) 把 $o(\sigma^{-2})$ 讀成「ε 誤差可到 $\sigma^{-2}$」（ε 單位是 $o(\sigma^{-1})$），或把充分條件讀成必要條件；(5) 把 Table 3 的 2.38 當成 405 報告的數字；(6) 把 IP 的最佳網路等式當成 IP 有效的原因。
- **延伸閱讀**：三個教學頁、三個論文錨點、綜合筆記與三篇 hub 筆記。

---

## 4. 互動元件規格

共用（審查 #10、#12、#15）：SVG 寬＝容器寬，resize 重畫，**無任何動畫過渡**（`transition:none`，與 reduced-motion 無關）；SVG 顏色全用 CSS 變數（`--ink`、`--ink-3`、`--line`、`--c-par`、`--c-perp`、`--accent`、淡底色 `--accent-soft`），深色模式自動切換；圖上標籤 13px（最小 12px），Unicode 上下標；讀數區 JetBrains Mono，標籤是載入時渲染一次的靜態 KaTeX，更新只改 `textContent`；`aria-live="polite"` 在輸入停止 400 ms 後宣告一句摘要（debounce，方向鍵連按不重複宣告）；每個滑桿有 `<label>` 與 `aria-valuetext`，寫顯示值（「σ = 0.100」「λ = 1.017」「α = 1.00」「t = 20」），不是內部值（對數滑桿亦同）。每條曲線同時有顏色、線型與直接標籤（< 600px 改圖例），色彩不是唯一區分。語意色：`--c-par` = TS、`--c-perp` = ES、`--accent` = IP、`--ink` = 精確／參考。圖版底部一行「讀數由本頁依 ○ 即時計算」＋標籤 pill。

### W0 · 誤差換算器（S2）〔本頁推導〕

- 輸入：σ（對數滑桿 $\log_{10}\sigma\in[-2.3,-0.3]$，預設 0.1）；誤差大小（數字欄，0–1，預設 0.2）；單位選擇「$x_0$／ε」。
- 輸出：對應的分數誤差 $\sqrt{1-\sigma^2}\,e/\sigma^2$ 或 $\eta/\sigma$，三位有效數字；一行文字「σ 每小 10 倍，$x_0$ 誤差對應的分數誤差大 100 倍」。
- 應觀察：$x_0$ 誤差 0.2 在 σ = 0.1、0.01 對應約 19.9、2000（$\sqrt{\bar\alpha}$ 因子，σ = 0.1 時 0.995）。

### W1 · 分數乘數對 σ（S4；S6 加入 IP）〔本頁推導〕

- **輸入**：α ∈ [0.05, 1.95] 步長 0.05（預設 1）；λ ∈ [0.950, 1.060] 步長 0.001（預設 1.017）；目前 σ（對數滑桿 $\log_{10}\sigma\in[-2.3,-0.004]$，即 σ ∈ [0.005, 0.991]，步長 0.001，預設 0.1；上限 0.991 使 σ* = 0.9833 可顯示）；y 軸「全範圍 [0, 1.1]」／「放大 [0.95, 1.01]」。S6 之後才出現：「顯示 IP」勾選（預設關）與 γ ∈ [0, 0.30] 步長 0.01（預設 0.10）。
- **計算**：`mTS`、`mES`、`mIPman(γ)`、`mIPgauss(σ,γ,0.25)`（〔本頁玩具〕一維高斯 $s^2=0.25$，$\bar\alpha=1-\sigma^2$）、`sigmaCross`、`alphaEq`；200 個對數等距點。`alphaEq`、`mIPgauss` 在 σ ≥ 1 回傳 NaN（讀數「—」）。
- **顯示**：橫軸 σ（對數 [0.005, 1]）；上方第二軸 cosine T = 1000 的 t：寬 ≥ 600px 標 1、10、50、100、200、500，< 600px 標 1、10、100、500；t = 883 不上軸，寫在交點註記「σ* = 0.9833（cosine t ≈ 883）」。軸標「t（cosine 排程，推定）」。TS 實線、ES 虛線、IP 法向點線、IP 高斯玩具點虛線；交點圓點；目前 σ 直線。標籤條「TS：corrector 步，只乘無條件分數」「ES：predictor 平均中的 ε，噪聲項不變」「IP：訓練（貝氏最佳網路）」。IP 線旁固定註記：「IP 線是最佳網路分數的法向分量倍數，不是取樣時的乘數，與 TS、ES 不是同一種量」（審查 #14）。
- **讀數**：$m_{TS}$、$m_{ES}$、（IP 顯示時）$m_{IP}$ 法向與玩具，四位小數；$\sigma^*$；$\alpha_{eq}$ 五位小數。
- **防護**：λ = 1 → σ* 顯示「—（λ = 1 時 ES 不縮放）」；λ < 1 → 「無交點（ES 放大 ε）」；σ* > 0.991 時交點畫在右緣外並標「σ* = … → 」；放大模式下低於 0.95 的 TS 線裁切並標「↓ TS」。
- **應觀察到**：(1) TS 隨 σ 減小下降，ES 水平；(2) α = 1 交點 0.9833，σ = 0.1 時差 10 倍；(3) $\alpha_{eq}$ 隨 σ 減小趨近 0（σ = 0.01 為 0.00366）；(4) S6 打開後：流形資料的 IP 法向倍數 0.9901 是常數，高斯玩具的 IP 倍數在小 σ 趨近 1（高斯資料沒有法向方向）。

### W2 · 單位圓上的平穩分佈（S5；IP 疊加在 S6 之後才啟用）〔本頁推導〕

- **簡化**（圖版標題下第一句）：237 頁 C2 的設定：VE、單位圓、von Mises κ = 4、θ₀ = π（237 App. C.3）、分數精確且是梯度場、Langevin 跑到平穩。這不是 ES 或 IP 實際使用的有限步反向過程。
- **輸入**：σ 離散 {0.2, 0.1, 0.05, 0.02, 0.01}（預設 0.05）；λ ∈ [0.950, 1.060] 步長 0.001（預設 1.017）；α ∈ [0.05, 1.90] 步長 0.05（預設 1）；γ ∈ [0, 0.30]（預設 0.10，S6 後出現）；分段按鈕「疊加：ES 型／TS／IP」（預設 ES 型；IP 在 S6 前停用並顯示「S6 介紹」）。
- **計算**：237 `c2Marginal({a:1, b:1, sigma:σ, alpha:α})`（不傳 err、lambda，即無誤差、無似然項）：精確 `alpha:0`；ES 型 `alpha:alphaEq(λ,σ)`；TS `alpha:α`；IP `sigma:σ*Math.sqrt(1+γ*γ), alpha:0`。參考 `refCurve('pdata',1,1,0,180)`、`refCurve('unif',1,1,0,180)`、`vmPow(1/λ,180)`。λ < 1 時 $\alpha_{eq}<0$：λ = 0.95 時 σ = 0.2 為 −0.0319、σ = 0.01 為 −0.0111，網格寬 $w=\sigma^{1-\alpha/2}$ 仍有限，密度有限且積分為 1（5.3 實測，列入 selfTest）。
- **顯示**：橫軸改為 $\varphi=t-\theta_0\in[-\pi,\pi]$（峰在中央，審查 #10）；縱軸 [0, 1.05·峰值]；精確（`--ink` 細實線）、$p_{data}$（`--ink-3` 虛線）、均勻（`--ink-3` 點線）、ES 型（`--c-perp` 粗虛線，標「ES 型 c = 1/λ」）、TS（`--c-par` 粗實線＋每 20 點一個圓點）、IP（`--accent` 粗點虛線＋方形標記）；ES 型時加 `vmPow(1/λ)` 細點虛線（「σ → 0 極限 $p^{1/\lambda}$」）。所選修正與精確的 TV < 0.001 時加註「與精確幾乎重合（TV = …）」。放大小圖 $\varphi\in[-0.6,0.6]$（同一變數）顯示峰頂差異。
- **讀數表**：列＝精確／ES 型／TS／IP，欄＝TV($p_{data}$)、TV(均勻)、最大／最小比；< 600px 轉置為每個修正一張卡。
- **防護**：c2Marginal 在 log 域正規化；圓 < 3 ms／條，以 (σ, α) 為 key 快取 dens（最多 40 筆）。
- **應觀察到**（σ = 0.01、λ = 1.017、α = 1、γ = 0.1）：精確與 IP 的 TV($p_{data}$) 都是 0.0001；ES 型 0.0046，約等於極限 $p^{1/\lambda}$ 對 $p_{data}$ 的 0.0045，最大／最小比 2976 → 2604；TS 的 TV(均勻) 0.0127。λ = 1.06：極限 TV 0.0155；0.965：0.0094。

### W3 · 讓 IP 對上 406 模型所需的 γ（S7）〔本頁推導〕

- **簡化**：406 Eq. 7 的高斯誤差模型，$x_0$ 誤差標準差 e 對所有 t 相同（推定：405 只給 $\nu_t$ 的平均）。圖版標題下寫：「t 很小時 γ 很大，是『e 不隨 t 變』這個假設造成的；真實的 $\nu_t$ 隨 t 的形狀論文只以圖呈現」。
- **輸入**：排程 cosine／linear（預設 cosine，推定）；$T'$ ∈ {20, 100, 1000}（預設 100）；e ∈ [0.00, 0.40] 步長 0.01（預設 0.20）；參考 γ ∈ [0.01, 0.30]（預設 0.10）；目前 t（整數 1..T′−1，預設 20）。切換 $T'$ 時 t clamp 到 [1, T′−1] 並重畫讀數（審查 #11）。
- **計算**：`gammaMatch(kind,T',e,'x0')`（D5 的 $D_t$ 遞迴，不做相近數相減）回傳每個 t 的 σ_t、$c_{1,t+1}$、$\gamma_1$、$\gamma_M$、$\delta_t$。索引照 406 design 4.0（顯示的 t 用 `coefs(A,t+1)`）；t ≤ T′−1，$1-\bar\alpha_t\ge4\times10^{-5}$，$\gamma_1=c_{1,t+1}e/\sigma_t$ 在所有可選 t 都有限（最大在 t = 1，σ₁ = 0.0064）。
- **顯示**：橫軸 t（$T'=1000$ 時對數）；縱軸 γ（對數 [1e−4, 1e2]，超出裁切並標箭頭）；$\gamma_1$ 虛線、$\gamma_M$ 實線、參考 γ 水平點線；$\gamma_M\in[\gamma/2,2\gamma]$ 的 t 區段以淡底色標「γ 與模型一致的區段」。
- **讀數**：σ_t、$c_{1,t+1}$、$\gamma_1$、$\gamma_M$（三位有效數字）、$\delta_t$；一致區段「t = a..b」或「無」。讀數列首固定加「推定」pill 與一行「依推定：e 為常數、排程 ○」。$V_t$ 只出現在可收合推導（D5）；正文與讀數只用 $\gamma_M$ 與 $\delta_t$。e = 0 時全部為 0，顯示「e = 0：沒有多出的變異數」。
- **應觀察到**：(1) $\gamma_M$ 在 t = 1 最大，隨 t 增大下降數個數量級；(2) 一致區段只占一小段（預設 t = 16..39）；(3) $T'$ 越大同樣的 e 對應的 γ 越小；(4) 中段 $\gamma_M>\gamma_1$（累積），末段相等。

---

## 5. 驗收數值表（全部出自 `verify/check_out.txt`）

**容許誤差（所有表與 W0–W3 讀數）**：頁面值 v 與表列值 r 視為一致，若 $|v-r|\le\max(10^{-3}|r|,\ \tfrac12\text{表列最後一位})$；科學記號三位有效數字者用相對 $10^{-3}$ 再加最後一位的一半；整數（t 區段、t = 883）必須相等。Monte Carlo（5.2）以表列標準誤為準。selfTest（6.3）用同一規則，個別項目若寫更嚴的 ± 以該項為準。

### 5.1 W1

| 量 | 值 |
| --- | --- |
| $m_{ES}(1.017)$、$m_{IP}^{man}(0.1)$ | 0.98328、0.99010 |
| σ = 0.5：TS α = 0.5／1／1.5；IP 玩具；$\alpha_{eq}$ | 0.7071／0.5000／0.3536；0.99432；0.02432 |
| σ = 0.1：同上 | 0.3162／0.1000／0.0316；0.99961；0.00732 |
| σ = 0.01：同上 | 0.1000／0.0100／0.0010；1.00000；0.00366 |
| $\sigma^*(1.017,\alpha)$，α = 0.1／0.5／1／1.5 | 0.8449／0.9668／0.9833／0.9888 |
| cosine T = 1000 的 $\sigma_t$，t = 1／10／50／100／200／500 | 0.00643／0.02513／0.08940／0.16705／0.31827／0.71145 |
| 第一個 $\sigma_t\ge1/1.017$ 的 t | 883 |
| 防護：`alphaEq(1.017,1)`、`mIPgauss(1,…)`；`alphaEq(1.017,0.99)` | NaN、NaN；1.6773 |

### 5.2 IP 恆等式（D4）

Monte Carlo（$x_0\in\{-1,1\}$、σ = 0.6、γ = 0.1、$3\times10^7$ 樣本、箱寬 ±0.005、seed 405）：

| y | MC $\mathbb E[\epsilon\mid y]$ | $-\sigma\nabla\log\tilde p$（σ′） | $-\sigma\nabla\log p$（σ） |
| --- | --- | --- | --- |
| −1.2 | −0.6748 ± 0.0008 | −0.6734 | −0.6795 |
| −0.4 | 0.2742 ± 0.0031 | 0.2726 | 0.2811 |
| 0.5 | −0.2316 ± 0.0026 | −0.2317 | −0.2393 |
| 1.0 | 0.3605 ± 0.0010 | 0.3620 | 0.3643 |

四點都在 σ′ 預測的 2 個標準誤內，對 σ 預測偏離 2.2–5.9 個標準誤（殘差含箱寬偏差）。單位圓（κ = 4、φ = 90°）$s_{\sigma'}/s_\sigma$：σ = 0.1、ρ = 1.1 徑向 0.99065、切向 0.99994；σ = 0.01、ρ = 1.1 徑向 0.99010、切向 1.00000；σ = 0.01、ρ = 1.01 徑向 0.99015。

### 5.3 W2（TV = ½∫|p − q|dt；λ = 1.017、α = 1、γ = 0.1）

| σ | TV($p_{data}$)：精確／ES 型／TS／IP | TV(均勻)：ES 型／TS | TV(ES 型, $p^{1/\lambda}$) | 最大／最小：ES 型／精確 |
| --- | --- | --- | --- | --- |
| 0.2 | 0.0348／0.0391／0.4287／0.0351 | 0.5950／0.2297 | 0.0346 | 2209.1／2518.1 |
| 0.05 | 0.0023／0.0067／0.5673／0.0023 | 0.6140／0.0633 | 0.0022 | 2579.2／2947.7 |
| 0.01 | 0.0001／0.0046／0.6082／0.0001 | 0.6152／0.0127 | 0.0001 | 2603.7／2976.1 |

極限：$p^{1/1.017}$ 最大／最小 2604.7、$p_{data}$ 2977.3、TV($p^{1/1.017}$, $p_{data}$) = 0.0045（σ = 0.01 的 ES 型 0.0046 約等於此值，差 0.0001 來自有限 σ）；λ = 0.965：0.0094；λ = 1.06：0.0155。λ = 0.95 防護：σ = 0.2 時 $\alpha_{eq}=-0.03187$、w = 0.1949、密度有限、積分 1.000000000、TV($p_{data}$) = 0.0217；σ = 0.01 時 $\alpha_{eq}=-0.01114$、TV($p_{data}$) = 0.0135、TV(·, $p^{1/0.95}$) = 0.0001。

### 5.4 W3（cosine，e 為 $x_0$ 誤差；全部依推定的常數 e 與排程）

| 設定 | t = 1 的 $\gamma_M$ | 代表點（t：σ, $\gamma_1$, $\gamma_M$, $\delta_t$） | $\gamma_M\ge0.1$ | $\gamma_M\in[0.05,0.2]$ |
| --- | --- | --- | --- | --- |
| $T'=20$, e = 0.2 | 30.99 | t = 5：0.334, 2.145e−1, 2.630e−1, 1.294e−4；t = 10：0.682, 6.035e−2, 8.865e−2, 7.161e−6 | t ≤ 9（σ ≤ 0.621） | t = 6..14 |
| $T'=100$, e = 0.2 | 29.37 | t = 10：0.155, 2.265e−1, 3.674e−1, 1.023e−4；t = 50：0.706, 1.330e−2, 3.687e−2, 2.301e−7 | t ≤ 24（σ ≤ 0.367） | t = 16..39 |
| $T'=1000$, e = 0.2 | 17.64 | t = 100：0.167, 2.209e−2, 1.037e−1, 8.039e−7；t = 500：0.711, 1.228e−3, 1.141e−2, 2.144e−9 | t ≤ 102（σ ≤ 0.170） | t = 62..169 |
| $T'=100$, e = 0.1 | 14.68 | t = 10：0.155, 1.132e−1, 1.837e−1, 6.705e−6 | — | — |

交叉核對：cosine $T'=20$、e = 0.1 的 $\delta_t$（t = 1, 2, 5：8.711e−3、6.054e−4、8.292e−6）與 406 `check_out.txt` 逐位相同。$D_t$ 遞迴與舊的 $V_t$ 遞迴在 e = 0.01 時 $\delta_t$ 相對差 2e−14（t = 1）到 2.7e−9（t = 90），所有表列值不變。ε 空間誤差模式在 t 接近 $T'$ 時發散（$\bar\alpha\to0$），只留在 `check_out.txt`，不放進元件。

### 5.5 S5 玩具（D3，高斯 $s^2=0.25$、完美網路、cosine）

| $T'$ | λ = 1 | λ = 1.017 | λ = 0.965 |
| --- | --- | --- | --- |
| 20 | 0.18467 | 0.64902 | 0.96751 |
| 100 | 0.23261 | 0.24172 | 0.21686 |
| 1000 | 0.24800 | 0.25663（λ 值推定） | 0.23113 |

頁面正文三列都寫：$T'=100$（406 Table 14 的 CIFAR-10 值）λ > 1 變寬、λ < 1 變窄；$T'=20$ 兩個方向都大幅變寬，玩具在 406 改善最大處失效；$T'=1000$ 的 λ = 1.017 是推定（406 稱最佳 b 隨 $T'$ 增加而下降，未給 CIFAR-10 1000 步的 b）。

### 5.6 W0

$x_0$ 誤差 0.2：σ = 0.1 → 19.9，σ = 0.01 → 2000；ε 誤差 0.2：2.0、20。

---

## 6. 實作計畫

### 6.1 檔案與外殼

- 單一檔案 `tut/syn-237-405-406/build/index.html`，CSS／JS 內嵌；上線為 `tutorials/score-error-exposure-bias-synthesis.html`（`add_tutorial.py`，`tutorial_list.json` 的 `papers: [237, 405, 406]`）。
- 沿用 `tutorials/exposure-bias-epsilon-scaling.html` 的外殼：CSS tokens（含 `--c-par`、`--c-perp`、深色模式）、KaTeX 0.16.9（cdnjs）、appbar 與主題切換（`pm-theme`）、`#pm-bar`（`← Paper Master` → `../index.html`、`教學與筆記` → `index.html`）、`initToc`、plate／callout／`.eq`＋`.eq-n`、`<details class="check">`、表格橫捲容器、`announce()`、`el()`／`widthOf()`、`?test` 的 `selfTest()`。新增：hero 論文卡、關係卡、第三種 pill「本頁詮釋」（`--ink-3` 虛線邊框）。
- 連結：三個教學頁（各章首次提到時連到該頁對應章節錨點，id 建置時從三頁 HTML 取得）；論文 `../index.html#paper-237`、`#paper-405`、`#paper-406`；arXiv 三篇；筆記 `score-error-exposure-bias-relations.html`、`score-geometry-lineage-hub.html`、`input-perturbation-lineage-overview.html`、`exposure-bias-elucidation-overview.html`。
- 不放時間線（三個子頁各有一條）。

### 6.2 程式（一個 IIFE）

- 依序貼入 `tut/237/verify/core.js`、`tut/syn-237-405-406/verify/core.js`（去掉兩檔末行 `module.exports`）。
- `initW0()`；`initW1()`（`state={alpha,lambda,gamma,logSigma,showIP,zoom}`，`revealIP()` 由 S6 的按鈕與捲入 S6 的 IntersectionObserver 呼叫一次）；`initW2()`（`state={iSigma,lambda,alpha,gamma,overlay,ipEnabled}`）；`initW3()`（`state={kind,Tp,e,gRef,t}`）。共用 `debounceAnnounce(msg, 400)`。
- 關係地圖、比較表、門檻表、W2 讀數表的窄版用 CSS 重排同一份 DOM。

### 6.3 selfTest（`?test`）

1. `mES(1.017)` = 0.98328、`sigmaCross(1.017,1)` = 0.9833、`alphaEq(1.017,0.01)` = 0.00366（±1e−5）；`alphaEq(1.017,1)` 與 `mIPgauss(1,0.1,0.25)` 為 NaN。
2. `respace('cosine',1000)`：σ₁₀₀ = 0.16705；第一個 σ_t ≥ 1/1.017 的 t = 883。
3. `circleScorePolar` 徑向比（σ = 0.01、ρ = 1.1、γ = 0.1）= 0.99010（±1e−5）。
4. W2：σ = 0.01 時 TV($p_{data}$)：精確 0.0001、ES 型 0.0046、IP 0.0001；TS 的 TV(均勻) 0.0127（±0.0005）；λ = 0.95、σ = 0.2 的密度全部有限且積分 1（±1e−9）。
5. `gammaMatch('cosine',100,0.2,'x0')`：t = 1 的 γ_M = 29.37（±0.01）；[0.05, 0.2] 區段 = 16..39。
6. `gammaMatch('cosine',20,0.1,'x0')` 的 δ₁ = 8.711e−3（±1e−6）。
7. `esGaussVar('cosine',100,0.25,1.017)` = 0.24172、`esGaussVar('cosine',20,0.25,1.017)` = 0.64902（±1e−5）。
8. W3 把 $T'$ 從 100 切到 20 後 state.t ≤ 19。
9. `document.documentElement.scrollWidth <= innerWidth`。

### 6.4 文案與排版

- 繁體中文、台灣用語、字面學術語氣，不用比喻、反問或問句標題。首次出現附英文：曝光偏差 exposure bias、流形 manifold、平穩分佈 stationary distribution、校正步 corrector、預測步 predictor、tempering（只作為文獻用詞出現一次，並先給字面定義）。
- 每句關係陳述後接狀態 pill；論文陳述附出處。頁面數字只能取自第 1 節與第 5 節。
- 390px 需拆行的公式：D4、D2 的代入式、D1 的換算鏈（`aligned` 在 = 處斷行）。
- 截圖清單：390／768／1440、深色模式的 W1–W3、鍵盤 focus、reduced-motion。

### 6.5 研究筆記

- 一篇 hub：`notes/score-error-exposure-bias-relations.md`（frontmatter `kind: hub`、cluster「Diffusion & Generative Models」、`arxiv` 三篇、`related` 三篇 hub 與四篇主題筆記）。只寫關係地圖，不重複三篇 hub 的內容；`index.json` 中放在三篇 hub 之後。
- `scripts/news.json` 由 `add_tutorial.py` 自動加消息。

### 6.6 建置後自我檢查

`?test` 全過；6.4 的截圖；逐項比對第 2 節答案確實在內文；所有站內連結（含三頁錨點）存在。

---

## 7. 風險與未決事項

| # | 項目 | 處理 |
| --- | --- | --- |
| K1 | 三篇評估不可比，讀者可能想看排名 | 不排名；比較表「證據類型」列與 S1「不相關之處」 |
| K2 | R2c、R3 的極限在 237 的 Langevin 設定，ES／IP 實際用有限步反向過程 | W2 標題下第一句；LO4 答案；S9 (2) |
| K3 | 406 Table 3 的 ADM-IP 2.38 標為 Ning et al. 2023 的 best FID，但不在 405 的任何表中（405：2.76／2.67／2.70／2.93） | S1、S9 照錄；γ 與權重來源未寫，不推測 |
| K4 | D3 玩具在 $T'=20$ 失效，正是 406 改善最大處；ADM 是否截斷 $\hat x_0$ 未寫 | 寫進 S5 正文與 LO4 答案 |
| K5 | W3 假設 e 為常數；405 Fig. 2 的形狀無法從文字讀出 | 標推定；t 很小時的大 γ 明寫為假設造成 |
| K6 | 排程：405 的 ADM 用 cosine（Tab. 9），406 未寫 ADM CIFAR-10 的排程，237 未寫 SD 1.5 的排程 | W1 t 軸與 W3 預設標「cosine，推定」 |
| K7 | D2 的常數乘數誤差是 $\Theta(\sigma^{-2})$，讀者可能誤以為違反 Thm 4.1 | D1、D2 寫明 Thm 4.1 是充分條件 |
| K8 | 「λ > 1 使輸出變寬」與 406 的理由方向相反，容易被讀成否定 406 | 寫明 406 的改善是真實網路的實驗結果，玩具只說明精確分數下的方向 |
| K9 | 三頁錨點 id 未在本文件列出 | 建置時從三頁 HTML 取；取不到時連頁首 |
| K10 | MC 驗證的殘差含箱寬偏差 | 頁面只寫「Monte Carlo 與推導一致（差 < 0.002）」 |
| K11 | D4 只關於貝氏最佳網路，可能被讀成 IP 的機制 | S6 正文與 LO5 寫出 DDPM-y 對照；S9 (6) |

---

## 8. 修訂紀錄（round 1，對應 `reviews/round1.md`）

| # | 處理 |
| --- | --- |
| 1 | 新增 S0 最少的定義；S1 每篇改為三句無符號白話，細節移後 |
| 2 | W1 的 IP 線與 γ 滑桿、W2 的 IP 疊加都在 S6 之後才出現 |
| 3 | S5 開頭加 Langevin、平穩分佈、TV 的定義段；von Mises、$p_{vol}$、Thm B.1 收進進階區 |
| 4 | R1b、LO1、D1 改為「足以／⇒」，補前提與「$E_\sigma$ 是生成分佈的分數」 |
| 5 | R3b 改標〔本頁詮釋〕，寫明 $\nu_t$ 實測、$e_t$ 為建模參數且未報數值 |
| 6 | R3、D4、S6、LO5 加入 DDPM-y（Eq. 8、Tab. 7）與「最佳網路等式解釋不了 IP 的改善」；新增 K11 |
| 7 | LO6、R3c、W3、5.4 標推定，並寫明 t = 1 的大值是常數 e 假設造成 |
| 8 | 去掉「升溫／溫度」；S5 改字面標題；tempering 只作文獻用詞並先定義 |
| 9 | W1 σ 上限 0.991；`alphaEq`、`mIPgauss` 在 σ ≥ 1 回傳 NaN（core.js 已改、5.1 驗證） |
| 10 | W2 實測 λ = 0.95（5.3）；橫軸改 $t-\theta_0$；每條曲線加線型與標記、重合註記 |
| 11 | `gammaMatch` 改 $D_t$ 遞迴（core.js 已改、5.4 比對）；$T'$ 切換時 clamp t；γ₁ 有限性寫明 |
| 12 | 關係地圖、門檻表、W2 讀數表加窄版；W1 上軸窄版標籤與 883 改註記；400 ms debounce；aria-valuetext 規格 |
| 13 | D3、LO4 寫入 $T'=20$ 失效；λ = 1.017 在 $T'=1000$ 標推定、寫明 CIFAR-10 是其他資料集之外的高值；$\alpha_{eq}\to0$ 限定為常數 ES，提到 ADM-ES* |
| 14 | LO4 答案附前提；W1 IP 線加「不是同一種量」註記；TS 一律寫「無條件分數」、CFG w = 7.5 不變 |
| 15 | Tab. 1 標 ImageNet 32×32、3.37 → 2.17 標 CIFAR-10；K3 照審查改寫；405「last half」改述為 $\nu_t$ 範圍的依據 |
| 16 | LO 合併為 8 項；LO1 加 W0 換算器；規定無動畫過渡與 SVG 深色 token |

## 9. 修訂紀錄（readiness）

| # | 處理 |
| --- | --- |
| B1 | S0 定義 $x_t$、$\hat x_t$、$T'$；S3 表改為文字並註明符號在哪節定義，β（誤差階數）與 K（量誤差的有界區域）在表內定義；$p_{vol}$ 在 S5 正文定義 |
| B2 | 第 5 節開頭加容許誤差規則；0.0046 對 0.0045 改「約等於」 |
| N2 | W1 ES 標籤與 S3 一致：「predictor 平均中的 ε，噪聲項不變」 |
| N3 | W2 寫出 `c2Marginal`、`refCurve(kind,a,b,lambda,nt)`、`vmPow` 的實際參數 |
| N5 | W3 讀數列加「推定」pill |
| N6 | 同 B2 |
| N7 | $V_t$ 只在可收合推導中出現 |
| N1 | 不放時間線（協調者確認），故不建 `notes/timeline.json` |

## 10. 建置紀錄（build round）

- 產出 `build/index.html`（約 114 KB），由 `build/src/assemble.sh` 組合：406 頁外殼的 CSS（第 1–382 行）＋`src/extra.css`＋`src/body.html`＋兩份 core.js＋`src/app.js`。`build/src/local.py` 產生測試用 `.local.html`（KaTeX 改本機路徑、去掉 Google Fonts；上線檔保留 CDN）。
- `?test`：116 passed、0 failed（390、800、1440 寬）。`impeccable detect`：0 項（10 則 numbered-section-labels 提示，允許）。390px 無水平捲動；KaTeX 已渲染；無 console 錯誤。
- LO 自查：`accept/page.txt` 含 LO1–LO8 參考答案的關鍵詞與數字（`extract_page_text.py`）。
- 視覺一輪修正：W2 峰頂放大圖改用平移後的陣列並縮到 ±0.2；W1 上軸標題與 ES 標籤移位；窄版 `table.stack` 儲存格改 block（KaTeX 不再被 grid 拆開）。截圖在 `build/caps/`。
- 不放時間線（協調者確認）。

- 驗收第一輪 2/8（reader miss）：S1、S2、S4、S5、S6、S7 在檢查題前各加一段「本章結論」（`p.concl`），完整寫出 LO7、LO1、LO3、LO4、LO5、LO6 的參考答案、狀態（「這是本頁的推導，論文沒有寫」／「本頁的詮釋」）與前提；LO1 題目加問狀態。
