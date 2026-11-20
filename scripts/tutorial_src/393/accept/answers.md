## 答案（Goldfish 驗收，只用頁面文字）

**LO1**
- 答：MDM 的先驗是 mask token；USDM 的先驗是 K 個詞的均勻分布 $\mathbf 1/K$。能在反向過程修改已產生 token 的是 USDM（自我修正）。
- 引文：「USDM 的先驗是 K 個詞的均勻分布」
- 信心：high

**LO2**
- 答：邊際是單一時刻 $z_t$ 的分布；路徑是多個時刻的聯合分布。USDM 的邊際是 $\mathrm{Cat}(\alpha_t x+(1-\alpha_t)\mathbf 1/K)$。生成時 $x$ 未知，用 $x_\theta(z_t,t)$ 代入後驗 (4)，得反向核 $p^\theta_{s|t}(\cdot\mid z_t)$，再從中抽 $z_s$（ancestral sampling）。
- 引文：「逐步抽 zs∼ps∣tθ」
- 信心：high（抽樣公式 (4) 頁面只給結構，未逐項展開）

**LO3**
- 答：正確類別成為 arg max 的機率 $p_x=\int\phi(z-\tilde\alpha/\tilde\sigma)\Phi^{K-1}(z)\,dz$，式 (28)。其餘 $K-1$ 類對稱，各為 $(1-p_x)/(K-1)$。整體分布為 $\mathrm{Cat}(a x+(1-a)\mathbf 1/K)$，$a=(Kp_x-1)/(K-1)$，形式與 USDM 邊際相同。
- 引文：「其餘 K−1 類對稱，各為」
- 信心：high（頁面警告 $\phi$ 的印刷有誤，見第 5 點）

**LO4**
- 答：$T$ 是 Diffusion Transformation 運算子，把高斯參數 $\tilde\alpha_t$ 映到離散參數 $\alpha_t=T(\tilde\alpha_t)$，即式 (11)。$T(0)=0$；$\tilde\alpha\to1$ 時 $T\to1$。$K=2$ 時封閉式 $T=\mathrm{erf}(\tilde\alpha/(2\tilde\sigma))$（$\tilde\alpha=0.6$ 時為 0.4041）。$K$ 大時曲線右移：$K=30522$ 時 $T(0.85)=0.0077$、$T(0.97)=0.4523$。
- 引文：「K=2 時有封閉式」
- 信心：high

**LO5**
- 答：作者還驗證 arg max 後的邊際隨時間滿足 USDM 的線性 ODE (12)，擴散參數為 $T(\tilde\alpha_t)$。結論寫成式 (13)：$q_t(\cdot\mid x;T(\tilde\alpha_t))=[\arg\max]_\star\tilde q_t(\cdot\mid x;\tilde\alpha_t)$，即兩者每個時刻的邊際相同，稱為 Diffusion Duality。
- 引文：「作者還驗證了 arg max 後的邊際隨時間」
- 信心：high

**LO6**
- 答：不一定。給定整個 $w_s$ 時，下一步 arg max 的機率 (39) 依賴 $w_s$ 的每個座標，而 USDM 轉移 (38) 只依賴 $z_s$，兩者一般不等 (40)。論文措辭是「does not necessarily follow」。頁面的 z 條件模擬（K=3）也得到不等的轉移機率：0.5915、0.4573，對 USDM 的 0.6109。但對 $w_s$ 取平均後，邊際回到正確分布。
- 引文：「所以，逐點取 arg max 的高斯軌跡不一定是」
- 信心：high

**LO7**
- 答：比較的是 Theorem 3.1 構造出的離散取樣器與高斯取樣器在真實資料上的邊際概似（式 (16)），不是 ELBO 的比較。限制：$q_{data}$ 是 one-hot 上的質量函數，$\bar p$ 是密度，KL 為 $+\infty$，所以原式不提供資訊，頁面稱之為動機性論證。作者據此在離散潛變數上定義模型，並以離散 NELBO 訓練與評估。
- 引文：「屬動機性論證」
- 信心：high

**LO8**
- 答：NELBO 是 $-\log p_\theta(x)$ 的上界，由各時刻後驗的 KL 散度累加而成；PPL 是它的指數。(17) 是 (49) 對每個固定 $(z_t,x)$ 的代數改寫，不需要建構 $\bar x$。論文用兩式評估同一模型，都得 PPL 33.7（A.6）。頁面數值驗證兩式只差浮點誤差（約 $10^{-16}$）。
- 引文：「用兩式評估同一個模型，都得到 PPL 33.7」
- 信心：high

**LO9**
- 答：(21) 把 denoiser 的輸入由 $\arg\max(w_t)$ 換成 $\mathrm{softmax}(w_t/\tau)$；損失仍是 $f_{Duo}$，$z_t$ 仍用 arg max。時間 $t\sim U[\beta,\gamma]$。前 500K 步 $\tau=0.001$、$[\beta,\gamma]=[0.03,0.15]$；之後 $\tau=0$（後段 $t$ 區間未寫）。不是合法 NELBO：只有 $\tau\to0^+$ 且 $\beta=0,\gamma=1$ 時才是。評估一律用 (18)。
- 引文：「τ>0 時 (21) 不是合法的 NELBO」
- 信心：high

**LO10**
- 答：假設 denoiser 完全正確（$x_\theta=x$），PF-ODE 軌跡變成 $\tilde\alpha_t x+\tilde\sigma_t\epsilon$，逐點取 arg max 得式 (22)。每個 token 只可能是 $x$ 或 $j^*=\arg\max_{i\ne m}\epsilon_i$；翻轉點為 $\tilde\alpha=\Delta/\sqrt{1+\Delta^2}$，$\Delta=\epsilon_{j^*}-\epsilon_m$。$\Delta\le0$ 時 token 全程為 $x$。
- 引文：「每個 token 至多翻轉一次」
- 信心：high

**LO11**
- 答：教師是上一輪結束時的學生權重（Alg. 1，非 EMA）；損失為式 (23)，$L_{DCD}=\sum_\ell D_{KL}(x^\ell_\theta(z_t,t),\,x^\ell_{\theta^-}(z_s,s))$，學生看較吵的 $z_t$，$s=\max(t-\delta,0)$。設定：$N=5$ 輪、每輪 $M=10K$ 步，$\delta$ 從 $1/512$ 起每輪加倍。EMA 係數、batch、學習率頁面未寫。
- 引文：「N=5 輪、每輪 M=10K 步」
- 信心：high（Alg. 1 與 B.2.2 對教師的描述不一致，見第 4 點）

**LO12**
- 答：Duo 的 PPL：LM1B 為 33.7、OWT 為 25.2（Table 1）。蒸餾 5 輪後，ancestral 16 步的 Gen PPL 為 75.24，低於未蒸餾 1024 步的 77.69（Table 5、Table 7）。與蒸餾 MDLM+SDTT 比：32 步以內 Duo 較好（例如 16 步 75.24 對 89.17）；64 步以上落後（64 步 55.83 對 47.04）。
- 引文：「16 步的 Gen PPL 為 75.24」
- 信心：high（Table 1、7 的數字都直接取自頁面）

## 讀者容易卡住的地方

1. $\phi$ 的印刷有誤：(11) 下方與 A.1 Part 3 寫成 $e^{-z^2}/\sqrt{2\pi}$，正確應為 $e^{-z^2/2}/\sqrt{2\pi}$。頁面的計算都用後者。
2. 速率矩陣的正負號：(3)、(17)、(49) 的 $\alpha_t'$ 慣例與 (12)、(36) 相反。頁面以 (12)、(36) 為準。
3. 「邊際」與「路徑」：邊際相同不代表路徑相同（第 6 章）。讀者容易以為 arg max 後整條軌跡都是 USDM。
4. DCD 教師的說法不一致：Alg. 1 用上一輪學生權重，B.2.2 與 Alg. 2 的對照組用 EMA，頁面以 Alg. 1 為準。
5. 訓練時間窗與式 (11) 不一致：照 $\tilde\alpha_t=1-t$ 計算，$[0.03,0.15]$ 只對應 $T\in[0.0077,0.4523]$（K=30522），並不是頁面所說的 $[0.05,0.95]$。K 的確切值（30522 或 50257）論文未寫，頁面的 T 數字也隨 K 而變。
