## LO1
- 答：訓練時輸入的是由真實資料 $x_0$ 加噪得到的 $x_t$；取樣時輸入的是網路前幾步自己產生的 $\hat x_t$。網路的工作是看著 $x_t$ 預測當初加進去的噪聲。
- 段落：「訓練時是由真實 x0 加噪得到的 xt」
- 信心：high

## LO2
- 答：$p_\theta(x_{t-1}|x_t)$ 實際上就是 $q(x_{t-1}|x_t,x_\theta^t)$。流程是先預測 $\epsilon_\theta$，由 Eq. 1 換算 $x_\theta^t$，再把後驗中的 $x_0$ 換成 $x_\theta^t$。平均為 $\mu_\theta=c_{1,t}x_\theta^t+c_{2,t}x_t=\frac{1}{\sqrt{\alpha_t}}\big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta\big)$（Eq. 5–6）。
- 段落：「p_θ(x_{t-1}|x_t) 實際上就是 q(x_{t-1}|x_t,x_θ^t)」
- 信心：high

## LO3
- 答：多出的變異數是 $\big(\frac{\sqrt{\bar\alpha_t}\beta_{t+1}}{1-\bar\alpha_{t+1}}e_{t+1}\big)^2$，即 $(c_{1,t+1}e_{t+1})^2$。平均沒有多出：兩者都是 $\sqrt{\bar\alpha_t}x_0$。Table 1 中取樣變異數為 $1-\bar\alpha_t+(c_{1,t+1}e_{t+1})^2$。
- 段落：「比訓練分佈 q(xt|x0) 多出 (√ᾱt βt+1 …)²」
- 信心：high

## LO4
- 答：DDIM 的差距項是 $\big(1-\sqrt{\frac{\alpha_{t+1}-\bar\alpha_{t+1}}{1-\bar\alpha_{t+1}}}\big)^2\bar\alpha_t e_{t+1}^2$（Eq. 31）。它恆為正的理由，頁面只寫「$\alpha_{t+1}<1$，根號內小於 1」。因為這一項是平方，所以不會為負。
- 段落：「因為 αt+1<1，根號內小於 1，所以這一項恆為正」
- 信心：medium（頁面的理由很簡短，平方是關鍵）

## LO5
- 答：作者的論證是，Nichol & Dhariwal 給出取樣噪聲變異數範圍 $[\tilde\beta_t,\beta_t]$，而 $\tilde\beta_t$ 已是下界，不能再小。若把噪聲變異數換成更大的 $\dot\beta$，差距只會更大。頁面明確標示這是作者的論證，不是定理。
- 段落：「β̃t 已是下界，不能再小」
- 信心：high（作者的說法）；medium（論證是否成立）

## LO6
- 答：Alg. 1（單步）每步的輸入都是真實的 $x_t$，量的是一步的誤差。Alg. 2（多步）從 $x_T$ 連續取樣，每步只看得到自己產生的 $\hat x_t$。Fig. 1（CIFAR-10、20 步）顯示多步誤差越接近 $t=1$ 越大，單步誤差接近 0。頁面的讀圖值（推定）：$t=2$ 約 0.47、$t=10$ 約 0.265、$t=18$ 約 0.015；單步全程約 0.01 以下。
- 段落：「多步誤差越接近 t=1 越大，單步誤差接近 0」
- 信心：high（結構）；medium（數字是讀圖值）

## LO7
- 答：遞迴是 $V_{t-1}=c_{2,t}^2V_t+(c_{1,t}e_t)^2+\tilde\beta_t$，起點 $V_{T'}=1-\bar\alpha_{T'}$。最後一步幾乎不累積，因為 $c_{2,t}$ 在取樣末段趨近 0，前一步多出的變異數權重變小，最後一步的誤差主要來自它自己的 $c_{1,t}e_t$。累積主要在取樣中段。頁面也說明，Eq. 7 模型本身不會產生 Fig. 1 那樣末段最大的形狀。
- 段落：「c2,t 在取樣末段趨近 0」
- 信心：high（本頁推導）

## LO8
- 答：$\delta_t=(\sqrt{V_t}-\sqrt{1-\bar\alpha_t})^2$，其中 $V_t$ 是多步取樣 $\hat x_t$ 的變異數，$1-\bar\alpha_t$ 是訓練變異數。它來自 Fréchet distance：$d^2=N\delta_t$（$N$ 為維度）。因為兩分佈平均相同、共變異數都是單位矩陣的倍數，所以只剩變異數項。量測用 Alg. 3，取 50,000 個樣本。
- 段落：「Fréchet distance 為 d2=Nδt」
- 信心：high

## LO9
- 答：不是 0。原因是 $x_T$ 幾乎不含 $x_0$ 的資訊（$\bar\alpha_T$ 很小），減去起點的 $\sqrt{\bar\alpha_t}x_0$ 後，仍留下約 $2\bar\alpha_t\mathrm{Var}(x_0)$ 的變異數誤差。高斯玩具估計 $\delta_1=0.426$，與 Fig. 8 讀圖值 0.39–0.465 相近。頁面說這不是下限，且 $\delta_t$ 應用於比較，不是絕對偏差量。
- 段落：「網路完美時，Alg. 3 量到的 δt 也不是 0」
- 信心：high（本頁推導）

## LO10
- 答：ES 只改 Eq. 6 平均中的 $\epsilon$ 項，改為 $\epsilon_\theta/\lambda_t$；噪聲項 $\sqrt{\tilde\beta_t}z$ 不變。依據是 Fig. 2（CIFAR-10、20 步）：每個時間步，取樣時 $\epsilon_{\text{samp}}$ 的 L2 範數都大於 $\epsilon_{\text{train}}$。
- 段落：「每個時間步，取樣時 ε_samp 的 L2 範數都大於 ε_train」
- 信心：high

## LO11
- 答：Eq. 10 是 $\Delta N(t)-1\approx\sum\int(\lambda-1)\,dt$，表示先前各步的範數高估，在線性傳播假設下的累積。作者據此推得 $\lambda(t)=kt+b$。但頁面說明 $k,b$ 在不同網路間變化不大，實際上直接搜尋；$T'$ 約 50 時 $k\to0$，多數實驗用常數 $\lambda=b$（ADM CIFAR-10 的 $b=1.017$）。$b$ 以 FID 搜尋，約 6–10 次試驗；搜尋時用 10k 樣本，報告用 50k。App. A.10 的搜尋步長為 0.001、0.0005、0.0001，由粗到細。
- 段落：「實際作法：作者發現 k,b 在不同網路間變化不大」
- 信心：medium（線性排程的 $t$ 索引頁面未說明）

## LO12
- 答：ADM CIFAR-10 100 步的 FID 由 3.37 降到 2.17。增益較大的設定有：步數少（20 步由 10.36 降到 5.15）、隨機取樣（DDPM $\eta=1$）、Euler 相對 Heun。限制：只修正 $\epsilon$ 的大小、不修正方向，無法使 $\delta_t=0$；理論依賴 Eq. 7 的高斯、獨立誤差假設；$\lambda$ 由 FID 搜尋，Eq. 10 只給定性依據。
- 段落：「ADM 在 CIFAR-10 以 100 步取樣，FID 由 3.37 降到 2.17」
- 信心：high

## 讀者容易卡住的地方
1. 符號：$\epsilon_\theta^t$ 的上標 $t$ 代表 training，不是時間步；$\epsilon_{\text{train}}$ 與 $\epsilon_{\text{samp}}$ 容易混淆。
2. 「本頁推導」與論文結論的界線：Eq. 7 模型、Fig. 1 讀圖值、高斯玩具數字都不是論文結論，讀者容易把它們當成論文數字。
3. LO4 的「恆為正」：頁面只寫根號內小於 1，沒有說明為什麼平方後一定非負。
4. Eq. 10 與 $\lambda$ 的形狀：頁面說二次 $\Delta N$ 對應線性 $\lambda$，但多數實驗用常數 $\lambda$；兩者的關係需要對照 App. A.10 才看得懂。
5. 目標方向：$\delta_t=0$ 不是可達到的目標（完美網路也不為 0），$\delta_t$ 只用於比較取樣器。
