# 設計文件：〈Flow Map Language Models: One-step Language Modeling via Continuous Denoising〉互動教學

- 論文：Chanhyuk Lee、Jaehoon Yoo、Manan Agarwal、Sheel Shah、Jerry Huang、Aditi Raghunathan、Seunghoon Hong、Nicholas M. Boffi†、Jinwoo Kim†（KAIST、CMU），arXiv 2602.16813v3。資料庫編號 No. 4。
- 原文：`tut/4/src/paper.txt`（PDF 同目錄）；顧問備忘 `tut/4/advisor.md`。本文件中「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」「Alg. n」皆指論文原文編號（論文的演算法編號是 5–8、表格編號不連續，照原文）。
- 產出：`tut/4/build/index.html`（單一 HTML）。上線位置 `tutorials/flow-map-language-models.html`（slug `flow-map-language-models`），站內相對連結以上線位置為準。
- 驗算腳本：`tut/4/verify/toy.py`（C1a、C1b 與玩具模型）、`c2.py`（C2）、`c3.py`（C3）、`c4.py`（C4）。第 5 節的驗收數值全部由這四支腳本印出，演算法與頁面相同。
- 研究筆記：`tut/4/notes/`（總覽 `fmlm-lineage-overview` + 6 篇主題筆記 + `timeline.json`，14 筆）。
- 文件慣例：本設計自行推導、論文沒有寫出的內容標「〔本頁推導〕」；用論文數字校準而非論文寫明的值標「推定」（本頁沒有推定值）。頁面上用同樣的標籤（沿用 238 的「本頁推導」pill）。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

離散擴散語言模型每一步可以平行更新多個 token，但取樣步數少時品質崩落：反向轉移機率定義在 $V^L$ 上，只能用逐 token 的乘積近似（Eq. 2），這個近似忽略 token 之間的相關。本文改走連續路線：把每個 token 寫成 $|V|$ 維 one-hot 向量，在無約束的歐氏空間 $\mathbb R^{L\times|V|}$ 上做 flow matching（Sec. 3.1–3.2）。網路不學速度，而學 denoiser $D_t(x)=\mathbb E[x_1\mid I_t=x]$；在 one-hot 資料上它正好是逐 token 後驗機率（Lemma 3.1），因此可用 softmax 輸出、交叉熵訓練（Prop. 3.2）。這個模型稱為 FLM。為了少步生成，作者學 flow map $X_{s,t}$（ODE 的解算子），並提出 two-time denoiser $\delta_{s,t}(x)=x+(1-s)v_{s,t}(x)$：它永遠在 simplex 上、能精確還原 $X_{s,t}$、滿足半群條件（Prop. 3.3），所以 flow map 也能用 KL 目標蒸餾（Eq. 23）。蒸餾後的模型稱為 FMLM。作者另證明離散狀態空間不存在樣本層級的確定性 flow map（Prop. 3.5）。實驗中 FLM 在 1024 步的 Gen. PPL 追平或低於離散擴散（LM1B 96.91、OWT 62.23，Table 10），但 OWT 的 entropy 5.33 低於資料 5.44（Duo 5.55 則高於資料），PPL 要和 entropy 一起讀；FMLM 一步生成在 LM1B 得 119.34、OWT 得 168.30，優於所有少步蒸餾的離散基線（Table 13）。

### 1.2 研究動機（Sec. 1–2）

1. 自回歸模型（Eq. 1）取樣必須逐 token 進行，生成速度有序列瓶頸。
2. 離散擴散（masked、uniform）每步可平行輸出多個 token，但少步時品質急降 [7]；原因是反向轉移 $p_{t|s}(y_t|y_s)$ 定義在 $V^L$ 上，輸出維度不可行，只能用逐 token 乘積近似（Eq. 2）。條件獨立只在 $t\to s$ 的極限成立 [33]，步數一少就產生 new diego、san york 這類不存在的組合（Fig. 3）。作者強調這是「改善模型品質也無法解決」的根本問題（Sec. 2）。
3. 連續擴散語言模型以確定性 ODE 演化，不需要這個近似，還能接上 flow map 少步蒸餾；但普遍認為它表現不如離散方法 [19, 20]。本文要推翻這個看法。

### 1.3 貢獻（Sec. 1）

1. 證明 one-hot 上的連續 flow 定義唯一的 flow map，可直接學來做少步推論；並證明離散擴散沒有這個結構（Prop. 3.5）。
2. 把 flow 與 flow map 都重參數化成 simplex 上的量（$D_t$、$\delta_{s,t}$），提出交叉熵／KL 目標，並大幅優於平方誤差版本（Table 18）。
3. 以解碼錯誤率設計時間重參數化（Eq. 24–25），解決先前連續方法的訓練不穩。
4. LM1B、OWT 上 FLM 追平最先進離散擴散；FMLM 一步生成優於少步離散基線；並展示 autoguidance 與 flow map reward guidance（Sec. 5.3）。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S2 末尾以可收合表格提供）

| 符號 | 意義 |
| --- | --- |
| $V$、$\lvert V\rvert$、$L$ | 詞表、詞表大小、序列長度；$y=(y^l)_{l=1}^L\in V^L$ |
| $f$、$g$ | one-hot 編碼、逐位置 argmax 解碼（Eq. 4） |
| $x_0\sim\mathcal N(0,I)$、$x_1=f(y)$ | 噪聲、資料（$\mathbb R^{L\times\lvert V\rvert}$） |
| $t\in[0,1]$ | **$t=0$ 純噪聲，$t=1$ 資料**（與 DDPM 相反） |
| $I_t$ | interpolant $(1-t)x_0+tx_1$（Eq. 5） |
| $b_t$、$D_t$ | 速度場（Eq. 7）、denoiser（Eq. 9） |
| $p^l_{1\mid t}(\cdot\mid I_t=x)$ | 第 $l$ 個位置的 token 後驗（條件在整個序列的 $I_t$ 上） |
| $\Delta^{\lvert V\rvert-1}$ | 機率 simplex：分量非負、總和為 1 的 $\lvert V\rvert$ 維向量 |
| $X_{s,t}$、$v_{s,t}$ | flow map（Eq. 14）、平均速度（Eq. 15） |
| $\delta_{s,t}$、$\gamma$ | two-time denoiser（Eq. 18）、半群權重（Eq. 22） |
| $P_e(t)$、$\tau(t)$ | 解碼錯誤率（Eq. 24）、重參數化時間（Eq. 25） |
| $d$ | 主文：網路內部特徵維度；**App. B 起改記 $d=L\times\lvert V\rvert$**，頁面只用主文意義 |
| sg | stop-gradient |

#### 步驟 1：問題——因子化誤差（Sec. 2，Eq. 1–2，Fig. 3）

$$p(y)=p(y^1)p(y^2|y^1)\cdots p(y^L|y^{<L})\tag{Eq. 1}$$
$$\tilde p_{t|s}(y_t|y_s):=p^1_{t|s}(y^1_t|y_s)\cdots p^L_{t|s}(y^L_t|y_s)\tag{Eq. 2}$$

- 離散擴散以 ancestral sampling 在格點 $0=t_0<\dots<t_N=1$ 上反轉加噪過程；每個因子是「給定 $y_s$、對其他位置邊際化後，第 $l$ 個 token 的條件機率」。
- 假設去噪後的 token 在 $y_s$ 下條件獨立；此假設只在 $t\to s$ 成立，步數少時忽略相關。
- 玩具（Fig. 3）：資料只有 new york 與 san diego；少步時離散法產生四種組合，其中 new diego、san york 無效；連續 flow 在多步與少步都只產生有效資料。圖中少步四種組合各 25%。Fig. 3 沒有說明用哪一種離散模型與排程。
- 頁面 S1 用一句話各自定義：ancestral sampling（每一步從上一步的結果出發、依模型給的轉移機率抽下一個狀態）；$t\to s$ 極限（相鄰兩步的時間差趨近 0，一步只改變極少 token）；遮罩（masked diffusion 中把 token 換成「隱藏」符號，揭露即填回一個詞）。
- 〔本頁推導〕C1a 的離散玩具（本頁自選設定，不是 Fig. 3 的設定）：masked diffusion、線性遮罩排程（時間 $t$ 時每個 token 以機率 $t$ 已揭露）、等距 $N$ 步、**模型完美**（每個因子就是真條件機率）。每個 token 的揭露時間獨立均勻，兩者落在同一步的機率 $1/N$；同一步揭露時兩個位置各自 50/50 獨立抽樣，一半機率無效。所以無效率 $=1/(2N)$。只有 $N=1$ 的「四種組合各 25%」與 Fig. 3 一致；頁面不暗示 Fig. 3 使用這個模型或排程。這說明在這個設定下誤差來自因子化本身，不是模型不準。

#### 步驟 2：連續表示（Sec. 3.1，Eq. 3–4）

$$f:y\mapsto(\mathrm{onehot}(y^1),\dots,\mathrm{onehot}(y^L))^\top,\qquad g:x\mapsto(\mathrm{argmax}(x^1),\dots,\mathrm{argmax}(x^L))\tag{Eq. 4}$$

- $g(f(y))=y$，無損；不需要正則化或另外訓練 embedding。學習得到的 embedding 需要小心正則化以免崩塌或發散，預訓練 embedding 對 flow 未必最佳（Sec. 3.1）。
- 不投影回 simplex，狀態在無約束歐氏空間（與 simplex diffusion [38, 39] 不同）。

#### 步驟 3：flow matching（Sec. 3.2，Eq. 5–8）

$$I_t=(1-t)x_0+tx_1,\quad \dot x_t=b_t(x_t),\quad b_t(x)=\mathbb E[\dot I_t\mid I_t=x]=\mathbb E[x_1-x_0\mid I_t=x]\tag{Eq. 5–7}$$

- 速度可用 MSE 回歸學（Eq. 8，$t$ 均勻取樣），生成時在格點上數值積分 Eq. 6。
- 頁面 S2 在 Eq. 7 之前用一句話說明條件期望：$\mathbb E[x_1-x_0\mid I_t=x]$ 是「所有會經過點 $x$ 的 $(x_0,x_1)$ 配對，其 $x_1-x_0$ 的平均」，即站在 $x$ 時速度的最佳猜測。符號表收合，所以不能取代正文定義。
- 頁面 S2 用 2–3 句說明 Euler 法：$x_{t_{n+1}}=x_{t_n}+(t_{n+1}-t_n)\,b_{t_n}(x_{t_n})$，步長越大誤差越大。

#### 步驟 4：denoiser 與 Lemma 3.1（Eq. 9–11，App. C.1.1、C.2）

$$D_t(x):=\mathbb E[x_1\mid I_t=x],\qquad b_t(x)=\frac{D_t(x)-x}{1-t}\tag{Eq. 9}$$

- 推導（頁面可收合）：對 $I_t=x$ 取條件期望，$x=(1-t)\mathbb E[x_0|I_t=x]+tD_t(x)$，解出 $\mathbb E[x_0|\cdot]$ 代入 $b_t=D_t-\mathbb E[x_0|\cdot]$ 得 Eq. 9（App. C.2，Eq. 74–76）。
- 為何不學速度：目標 $x_1-x_0$ 繼承高斯噪聲的 full-rank 結構；$|V|$ 遠大於網路內部特徵維度 $d$ 時已知會欠擬合 [40]（Li & He，*Back to Basics*，arXiv 2511.13720，見筆記 `fmlm-flow-matching-interpolant`）。Table 18：Velocity + MSE 的 Gen. PPL 3801.36（不收斂）。
- **rank bottleneck（秩瓶頸）**，頁面正文要寫出的定義：網路最後一層把 $d$ 維內部特徵線性映射到 $|V|$ 維輸出，所以每個位置能輸出的向量最多只張成 $d$ 維子空間（秩 $\le d$）。速度目標 $x_1-x_0$ 含 $x_0$ 的高斯噪聲，噪聲在全部 $|V|$ 個方向都有分量（full-rank），$d\ll|V|$ 時大部分方向網路根本表示不出來，只能欠擬合。denoiser 目標 $D_t$ 是後驗機率向量，不帶這份噪聲，沒有這個問題。名稱出自 Sec. 5（Table 18 討論）：「Velocity prediction fails to converge, confirming the rank bottleneck induced by the Gaussian noise in high-dimensional one-hot spaces」。「秩 $\le d$」是本頁對 [40] 論點的白話說明，論文本身只寫 full-rank 與欠擬合。
- 正文定義（不只放在符號表）：$p^l_{1|t}(\cdot\mid I_t=x)$ 是「看到整個序列的含噪狀態 $x$ 後，第 $l$ 個位置的真 token 是各個詞的機率」。
- **Lemma 3.1**：$D_t(x)^l=p^l_{1|t}(\cdot\mid I_t=x)\in\Delta^{|V|-1}$（Eq. 11）。證明只用「$x_1^l$ 只取 $e_1,\dots,e_{|V|}$」：$\mathbb E[x_1^l|I_t=x]=\sum_ie_i\,p^l_{1|t}(x_1^l=e_i|I_t=x)$，就是後驗機率向量（Eq. 40–41）。
- 推論：輸出層用逐 token softmax，假設空間限制在有效的離散分布。
- 改寫 $D_t(x)=x+(1-t)b_t(x)$（Eq. 77）：denoiser 是用速度走一個步長 $1-t$ 的 Euler 步。它是後驗平均，會把多模態平均掉（App. C.2），不是一步 flow map。

#### 步驟 5：交叉熵訓練（Prop. 3.2，Eq. 12–13，App. C.1.2）

$$\mathcal L_{CE}(\hat D)=\int_0^1\mathbb E\Big[-\sum_{l=1}^L\log\hat D_t(I_t)^l\cdot x_1^l\Big]dt\tag{Eq. 12}$$

- 分解：$\mathbb E_p[-\log q]=H(p)+\mathrm{KL}(p\|q)$（Eq. 42）⇒ $\mathcal L_{CE}=\int\mathbb E[H(x_1|I_t)]dt+\int\mathbb E[\sum_l\mathrm{KL}(p^l_{1|t}\|\hat p^l_{1|t})]dt$（Eq. 45）；第一項不可約，KL ≥ 0 且等號 iff 相等 ⇒ 唯一最小值是真 denoiser。
- 誤差界：若超額風險 $\Delta_D(\hat D)\le\epsilon$，對任意 $\xi\in(0,1)$，$W_2^2(\hat p_{1-\xi},p_{1-\xi})\le C\epsilon$（Eq. 13）。比較在 $1-\xi$ 是為了避開 Eq. 9 的 $(1-t)^{-1}$ 奇異點。證明鏈：Pinsker（Eq. 52）⇒ $\int\mathbb E|\hat D-D|^2\le2\Delta_D$（Eq. 50）⇒ $\int_0^{1-\xi}\mathbb E|\hat b-b|^2\le2\xi^{-2}\Delta_D$（Eq. 53）⇒ Gronwall（Eq. 64）⇒ $W_2\le C_D(\xi)\sqrt{\Delta_D}$，$C_D(\xi)=\sqrt{2(1-\xi)}e^{L_{\hat b}(1-\xi)}/\xi$（Eq. 65）。假設：$b,\hat b$ Lipschitz（C.1）、二階動差有界（C.2）、輸出在 simplex 內部（C.3）。〔本頁推導〕Eq. 13 的 $C=C_D(\xi)^2$，$\xi\to0$ 時這個上界以 $\xi^{-2}$ 發散；發散的只是上界，不代表真誤差發散，屬定性保證。頁面 S3b 正文只寫一句結果（「交叉熵超額風險小，則在 $t=1-\xi$ 時生成分布與真分布的距離也小」），$W_2$、$\xi$、Gronwall 與證明鏈全部放進 `<details>`，不設元件。

#### 步驟 6：與離散擴散的關鍵差別（Sec. 3.2 末段）

- 兩者都學同一個逐 token 後驗 $p^l_{1|t}$（離散法也常用逐 token 交叉熵）。
- 離散法拿它做 ancestral sampling，需要聯合分布，所以有因子化誤差；連續法只拿它算速度（Eq. 11 + Eq. 9）。速度是條件期望，對 $x_1$ 線性，逐 token 邊際已經足以給出精確的 $b_t$。
- 〔本頁推導，依顧問備忘修正〕論文（Sec. 3.2）只說連續法不需要因子化近似。token 之間的相關由兩件事帶入：網路對每個位置的後驗都條件在**整個序列**的 $I_t$ 上（$p^l_{1|t}(\cdot\mid I_t=x)$），以及確定性 flow 把各位置耦合在一起演化；初始噪聲 $x_0$ 只提供隨機性。C1b 的突顯樣本展示這點：位置 2 一開始偏向 york，但位置 2 的後驗依賴兩個位置的座標（玩具中 $p$ 只依賴 $u=a_1+a_2$），最後兩個位置一起落在 san diego。
- 頁面必須避免「連續法沒有逐 token 分解」的說法：網路輸出同樣是逐 token softmax，差別在用途。

#### 步驟 7：時間重參數化（Sec. 4，Eq. 24–25，Fig. 9，App. C.6、E、F）

$$P_e(t):=\frac1L\sum_{l=1}^LP\big(g^l(x_t)\neq g^l(x_1)\big),\qquad \tau(t)=\frac{P_e(0)-P_e(t)}{P_e(0)}=1-\frac{|V|}{|V|-1}P_e(t)\tag{Eq. 24–25}$$

- $P_e(0)=1-1/|V|$，$P_e(1)=0$；$|\dot P_e(t)|$ 代表 flow 在 $t$ 決定 token 的進度。大 $|V|$ 時 token 在很窄的時間窗內被決定，均勻 $t$ 浪費訓練訊號、等距格點把步數花在幾乎不改變結果的區段。
- 做法：訓練取 $\tau(t)\sim U[0,1]$，取樣格點 $t_n=t(n/N)$。FLM 與 FMLM 用同一個 $\tau$。實作（App. E 描述）：以 Gauss–Hermite 求積近似「(25) 的 CDF」，在 1,000 個等距 $t$ 上得到 $(t,\tau)$，擬合 cubic spline 得 $\tau(t)$ 與 $t(\tau)$，訓練前算好；論文沒有寫出被積函數。**$\tau$ 不是學出來的**；被比較的 CDCD [13] 才是線上估計。
- 出處：$P_e$ 作為量化指標沿用自 Duo [19] 與 CANDI [20]（Sec. 4 第一句；筆記 `fmlm-onehot-denoiser-two-time`）。頁面 S4 一句帶過，本文的新東西是把它轉成時間重參數化。
- 〔本頁推導，interpolant 耦合讀法〕Eq. 24 用 $x_t$ 寫，論文沒有給公式。本頁把 $x_t$ 讀成 interpolant $I_t$、$x_1$ 為同一配對的資料（App. E 的 LUT 很可能算同一個量，但論文沒寫明）。在這個讀法下：單一位置、真 token 為 $k$，$I_t^l=(1-t)z+te_k$，$z\sim\mathcal N(0,I)$；解碼正確 ⇔ $z_k+t/(1-t)>\max_{j\neq k}z_j$，所以
  $$P_e(t)=1-\int\varphi(z)\,\Phi\!\big(z+\tfrac{t}{1-t}\big)^{|V|-1}dz,$$
  與資料分布無關，只依賴 $|V|$（$\varphi$、$\Phi$ 是標準常態的密度與 CDF）。$t=0$ 時積分為 $1/|V|$，與 Eq. 24 的端點一致。依此算出：$|V|=30{,}522$ 時 $\tau=0.05$ 與 $0.95$ 對應 $t\approx0.707$ 與 $0.854$，即 90% 的解碼進度在 $t\approx0.71$–$0.85$。窗的位置與 C3 的格點計數（2/17 對 15）都是這個讀法下的本頁推導，**不稱為重現 Fig. 9**。論文文字寫「concentrates acutely near $t=1$」；頁面寫「集中在很窄的時間窗（本頁計算約 0.71–0.85）」，不寫「貼近 $t=1$」（R2）。
- 頁面順序（S4）：先 Eq. 24–25 的文字定義 → C3，讓讀者看到「曲線在窄窗內落下」→ 正文寫出觀察 → 閉式放在 `<details>`「本頁怎麼算這條曲線」，第一句「Φ 是標準常態分布的累積分布函數」。
- 消融（Table 18「Time reparameterization」子表，300k 步）：無重參數化（$t$ 均勻取樣）149.18、learned entropic time [13] 130.42、Rank 121.28、解碼錯誤率（Eq. 25）106.98。四列都是時間重參數化 $\tau(t)$ 的不同選法。
  - **Rank 列的定義**：Table 18 標為「Rank [20]」，Sec. 5（Table 18 討論，p. 17）稱為「rank-based reparameterization [20]」，即 CANDI（Pynadath, Shi & Zhang，arXiv 2510.22510）提出的時間重參數化。論文沒有說明它怎麼算，頁面只寫「CANDI [20] 提出的另一種時間重參數化（rank-based）」，不解釋其機制、不臆測公式。頁面要寫的結論是論文的原句意思：解碼錯誤率版優於均勻取樣、learned entropic time 與 rank-based 三者，即把訓練信號集中在 token 被解出來的時段，比學一個排程更有效。
- App. C.6（Prop. C.12）：若每個位置的 token 在 $V$ 上均勻，由 Fano 不等式得 $\tau(t)\le\sigma(t)+O(1/\log|V|)$（Eq. 106），$\sigma$ 是 entropic time（Eq. 105）；作者自己說此假設「rather strong」。頁面一句帶過。
- 先驗消融（App. F，Table 28，200k 步）：Dir(1) 先驗 660.33 / 4.12，高斯 105.48 / 4.31。原因：Dir(1) 每維變異數 $\sim1/|V|^2$，集中在均勻向量附近，$P_e(t)$ 幾乎立刻歸零，$\tau$ 把整段時間擠到 $t=0$ 附近。

#### 步驟 8：flow map（Sec. 3.3，Eq. 14–17，App. B）

$$X_{s,t}(x_s)=x_t,\qquad X_{s,t}(x)=x+(t-s)v_{s,t}(x)\tag{Eq. 14–15}$$
$$X_{s,s}(x)=x,\quad \lim_{s\to t}\partial_tX_{s,t}(x)=b_t(x),\quad X_{u,t}(X_{s,u}(x))=X_{s,t}(x)\tag{Eq. 16}$$

- 一步取樣 $\hat x_1=X_{0,1}(x_0)$；多步 $\hat x_{t_{i+1}}=X_{t_i,t_{i+1}}(\hat x_{t_i})$，對任意格點都精確；實務上步數多通常較好。
- 半群條件可換成兩種微分刻畫：Lagrangian（Eq. 29，沿軌跡的時間導數）與 Eulerian（Eq. 30，速度的空間導數），對應 MeanFlow、Lagrangian self-distillation 等需要 JVP 的方法。本文主線用半群條件（相關於 progressive distillation、shortcut models）。
- 平方誤差蒸餾（Eq. 17）：離對角項強制半群、對角項 $v_{t,t}=\hat b_t$；邊界條件由 Eq. 15 自動滿足。把 $\hat b_t$ 換成 $\dot I_t$ 得到不需教師的直接訓練。
- 〔本頁推導〕C1b 的玩具：精確 flow map 的多步合成與一步結果相同（半群），端點與 $N$ 無關；Euler 的一步結果是 $D_0(x_0)$，在玩具中恆為 $(0.5,0.5,0.5,0.5)$，即兩個模態的平均。

#### 步驟 9：two-time denoiser（Prop. 3.3，Eq. 18–22，App. C.3–C.4，Fig. 4）

$$\delta_{s,t}(x):=x+(1-s)v_{s,t}(x)\tag{Eq. 18}$$

1. $X_{s,t}(x)=\frac{1-t}{1-s}x+\frac{t-s}{1-s}\delta_{s,t}(x)$（Eq. 19）：代 $v=(\delta-x)/(1-s)$ 入 Eq. 15（Eq. 78）。
2. $\delta_{s,t}(x)^l\in\Delta^{|V|-1}$（Eq. 20）：Eq. 79 乘積分因子 $1/(1-\tau)$ 得 Eq. 80，從 $s$ 積到 $t$ 得 Eq. 81，與 (i) 比對得
   $$\delta_{s,t}(x)=\frac{(1-s)(1-t)}{t-s}\int_s^t\frac{D_\tau(X_{s,\tau}(x))}{(1-\tau)^2}d\tau\tag{Eq. 82}$$
   權重非負，且 $\frac{(1-s)(1-t)}{t-s}\big[\frac1{1-t}-\frac1{1-s}\big]=1$（Eq. 83），所以 $\delta$ 是沿軌跡 denoiser 的凸組合。
3. $\delta_{s,s}=D_s$（Eq. 21、84）。
4. $\delta_{s,t}(x)=\gamma\,\delta_{s,u}(x)+(1-\gamma)\,\delta_{u,t}(X_{s,u}(x))$，$\gamma=\frac{(1-t)(u-s)}{(1-u)(t-s)}\in[0,1]$（Eq. 22）。推導：兩邊用 Eq. 19 展開（Eq. 94–96），消去 $x$ 項後乘 $(1-s)/(t-s)$（Eq. 97）；係數和 $(1-t)(u-s)+(t-u)(1-s)=(t-s)(1-u)$（Eq. 98）。中點 $u=(s+t)/2$ 時 $\gamma=(1-t)/(2-s-t)$（Prop. C.9）。

- 意義：$X_{s,t}$ 因為含高斯噪聲項 $\frac{1-t}{1-s}x$ 而離開 simplex，$\delta_{s,t}$ 永遠在 simplex 上（Fig. 4），所以 flow map 也能用 softmax 與 KL／CE 學。
- 條件：性質 (ii) 對真 flow map、真 denoiser 成立；學到的 $\hat\delta$ 在 simplex 上是因為 softmax 參數化。
- 〔本頁推導〕$t=1$ 時 Eq. 19 給 $X_{s,1}=\delta_{s,1}$；真資料是 one-hot，所以真 $\delta_{0,1}(x_0)$ 幾乎必然是頂點，一步生成是對輸出取 argmax（Alg. 8 回傳 $g(x_{t_N})$），不是從 softmax 抽樣。$t=1$ 時 $\gamma=0$，教師完全是 $\delta_{u,1}(X_{s,u})$。

#### 步驟 10：KL 半群蒸餾（Prop. 3.4，Eq. 23，Alg. 7–8，Sec. 4，App. C.5、E）

$$\mathcal L_{KL}(\delta)=\mathbb E_{t,s,u}\mathbb E\Big[\sum_l\mathrm{KL}(\bar\delta^l_{s,t}\,\|\,\delta^l_{s,t}(I_s))\Big]+\mathbb E_t\mathbb E\Big[\sum_l\mathrm{KL}(D^l_t(I_t)\,\|\,\delta^l_{t,t}(I_t))\Big],\quad\bar\delta_{s,t}:=\mathrm{sg}\big(\gamma\delta_{s,u}(I_s)+(1-\gamma)\delta_{u,t}(X_{s,u}(I_s))\big)\tag{Eq. 23}$$

- $(s,u,t)$ 的分布需在 $\{0\le s\le u\le t\le1\}$ 上 full support。實作把 $D_t$ 換成凍結的預訓練 $\hat D_t$（蒸餾）。用 KL 而非 CE 是因為最小值為 0、可直接量測重建品質；兩者只差常數，學習動態相同（Sec. 3.3）。Alg. 7 寫成 CE 形式。
- 把對角項的 $D_t$ 換成 one-hot $x_1$ 即得 Eq. 12 的 CE，成為不需教師的直接訓練（self-distillation，Eq. 104）。
- 對角項（Alg. 7）：$\tau(s)\sim U[0,1]$，輸入 $I_s$，$\mathcal L_{diag}=-\sum_l\hat D(I_s)^l\cdot\log\hat\delta^l_{s,s}(I_s)$；Eq. 23 寫成 $t$、$I_t$，兩者只是變數名不同，頁面兩欄對照時註明。
- 取樣三元組（離對角）：$h\sim U[0,1]$、$\tau(s)\sim U[0,1-h]$、$\tau(t)=\tau(s)+h$、$\tau(u)=(\tau(s)+\tau(t))/2$；**中點取在 $\tau$ 空間**，$\gamma$ 用換回 $t$ 後的 $(s,u,t)$ 以一般式計算（Alg. 7）。連續取 $(s,t)$，不同於 shortcut models 的二進格點 [47]。
- 邊界取樣：$\tau(t)$ 在 $t=0$ 附近平坦，$s$ 很少落在 0；以 $p=1/32$ 直接取 $(s,t)=(0,1)$。App. E：每個 batch 對半分成對角（$s=t$，flow matching）與離對角（蒸餾），1/32 是在蒸餾那一半之中。
- 取樣（Alg. 8）：$x_{t_{n+1}}=\frac{1-t_{n+1}}{1-t_n}x_{t_n}+\frac{t_{n+1}-t_n}{1-t_n}\hat\delta_{t_n,t_{n+1}}(x_{t_n})$，最後 $g(x_{t_N})$。
- 其他目標：Lagrangian（Eq. 85 → Eq. 99）、Eulerian（Eq. 86 → Eq. 100）的教師含導數項，可能暫時離開 simplex，用 CE。Table 18：半群 119.34、Eulerian 149.13、Lagrangian 193.08；蒸餾 119.34 優於 self-distillation 159.53。歸屬分開寫：主文（Sec. 5）只**推測** Eulerian 目標需要額外正則化才能讓預測留在 simplex 上；「Eulerian 因為對整個 $L\times|V|$ 狀態算空間 Jacobian 而極不穩定」出自 App. F（Fig. 31 的說明）。
- App. D：舊版的兩階段平方誤差蒸餾（Eq. 110–116）在 LM1B 一步得 102.49（第一階段）／104.37（第二階段），entropy 4.13／4.12，與 CE 版 119.34 / 4.16 相當；OWT 只有第一階段，一步 129.32 / **4.53**，對 CE 168.30 / 5.17，PPL 較低是以 entropy 大幅下降換來的（Table 24）。頁面在 S7 末尾一句說明：只有 LM1B 的比較 entropy 相當，該設定下 MSE 兩階段的 PPL 較低（R8）。

#### 步驟 11：離散沒有樣本層級的 flow map（Sec. 3.3，Prop. 3.5，App. C.1.3）

- 離散擴散的 CTMC 在分布層級有確定性演化 $\dot p_t=Q_tp_t$，$Q_t\in\mathbb R^{|V|^L\times|V|^L}$，所以分布層級的 flow map 存在，但無法計算或表示，只能用因子化近似。
- 頁面 S8 順序：先一句定義 pushforward（$f_\#\mu$ 是「把點 $x$ 上的機率整塊搬到 $f(x)$」得到的分布）→ C4 給直覺（只能整塊搬，搬不出比最小一塊更小的正機率）→ 命題 → 最後才命名 CTMC（連續時間馬可夫鏈：離散狀態間依速率跳躍的隨機過程）與 master equation（描述 CTMC 分布隨時間變化的線性 ODE $\dot p_t=Q_tp_t$）。
- **Prop. 3.5**：$S$ 有限且 $|S|\ge2$（$|S|=1$ 時唯一的分布就是 $\mu$），對任意 $\mu$，存在 $\nu$ 無法寫成 $f_\#\mu$（任意確定性 $f:S\to S$）。證明：$\nu(y)=\sum_{x\in f^{-1}(y)}\mu(x)$（Eq. 70），所以 $\nu$ 的正值至少是 $\mu_{\min}:=\min\{\mu(x)\mid\mu(x)>0\}$（最小正質量，App. C.1.3），取 $\nu(y_0)=\mu_{\min}/2$ 即不可達。取 $S=V^L$、$\mu=p_0$。
- 它沒有說的事（頁面明寫）：沒有說真實資料分布不可達，也沒有量化近似誤差。〔本頁推導〕若 $\mu$ 在 $S$ 上均勻，可達的 $\nu$ 是機率皆為 $1/|S|$ 整數倍的分布，近似可以任意細；若 $\mu$ 是 masked diffusion 全遮罩的點質量，$f_\#\mu$ 只能是點質量。實務上的障礙是步驟 1 的因子化。

#### 步驟 12：實驗（Sec. 5，App. E–G）

- 設定：LM1B（$L=128$、bert-base-uncased、$|V|=30{,}522$）、OWT（$L=1024$、gpt-2、$|V|=50{,}257$）；179M 參數 DiT，12 層、RoPE、AdaLN。FLM 1M 步、batch 512、Adam（β1 0.9、β2 0.999）、lr 3×10⁻⁴、2,500 warmup、attention softcapping；FMLM 蒸餾 100k 步、其餘相同。
- 評估：每個模型 1,024 個樣本，GPT-2 Large 算 Gen. PPL，另報逐樣本 unigram entropy（資料 LM1B 4.31、OWT 5.44）；entropy 低（例如 < 4）代表重複，所以高品質是「PPL 低且 entropy 接近資料」。沒有 validation perplexity（腳註 1：需要額外訓練）。
- Table 10（1024 步，Gen. PPL / Entropy）：LM1B：RDLM 268.21 / 4.33、CANDI 120.99 / 4.35、MDLM 109.21 / 4.32、Duo 98.14 / 4.31、FLM 96.91 / 4.29。OWT：CANDI 143.13 / 5.71、MDLM 105.15 / 5.63、Duo 77.69 / 5.55、FLM 62.23 / 5.33。Fig. 11：FLM 在步數少時退化（尚未蒸餾）。
- Table 13（FMLM，Gen. PPL / Ent.）：LM1B 1／2／4 步 119.34 / 4.16、110.19 / 4.21、98.76 / 4.21；OWT 168.30 / 5.17、133.29 / 5.25、111.31 / 5.26。一步基線 LM1B：MDLM+SDTT 1429.48 / 4.31、MDLM+Di4C 1217.10 / 4.38、Duo+DCD 1224.52 / 4.33、Duo+Di4C 292.94 / 3.79；OWT：1260.86 / 5.26、1298.80 / 5.29、5743.29 / 6.02、370.51 / 3.92。Sec. 5.2：LM1B 一步相當於基線 8–16 步，OWT 相當於 4–8 步；兩種失敗模式：PPL > 1,000（不連貫）與 entropy 崩塌（重複）。
- 條件生成（Table 15，OWT，50 token 前綴續寫 50 token，一步，finetune 25k 步）：FMLM 141.66 / 4.14，MDLM+SDTT 374.70 / 4.09（2.6× 低）。
- Self-BLEU（Table 25，一步，1,024 樣本；數值越低 n-gram 越多樣，但目標是**接近真實資料**，不是越低越好）：LM1B／OWT：真實資料 0.047 / 0.046、MDLM+SDTT 0.026 / 0.036、MDLM+Di4C 0.023 / 0.031、Duo+DCD 0.075 / 0.297、Duo+Di4C 0.054 / 0.272、FMLM 0.073 / 0.121、全部相同時 1.000。用途只是排除 mode collapse（FMLM 遠低於 1.0）；MDLM 基線低於資料是因為它們的樣本不連貫（PPL > 1,000），頁面不把 FMLM 的值說成勝過 MDLM。
- Autoguidance（Eq. 26，weak model = 推論時 dropout 0.1）：FLM 1024 步 96.91 → 51.62（η = 50，entropy 4.00）；Duo、MDLM 在 η ≥ 10 崩潰（PPL > 1900、> 1750）。作者假設離散 guidance 的乘法形式 $\hat p^{(guided)}\propto(\hat p^{(weak)})^{1-\eta}\hat p^\eta$ 放大因子化誤差。
- FMRG（Eq. 27）：flow map $X_{t,1}$ 提供可微分的終點預覽，獎勵模型只需在乾淨文字上訓練。Table 17 例：8 步 Sentiment，FMLM 102.0 / 0.948，MDLM+SDTT 117.0 / 0.477。
- 消融（Table 18，LM1B）：FLM：Denoiser + MSE 129.04 / 3.97、Denoiser + softmax + MSE 120.16 / 4.28、+ CE 96.91 / 4.29；表示子表（300k 步、learned entropic time [13]，與 96.91 不可比）：One-hot 130.42 / 4.27（即時間組的「Learned entropic time」列）優於所有 embedding（最好 Learned w/ L2Norm 243.42）；Riemannian 268.21 / 4.33、Simplex 85.07 / **3.76**、Euclidean 96.91 / 4.29。FMLM 一步：Average velocity + MSE 199.01 / 3.93、δ-denoiser + softmax + CE 119.34 / 4.16；時間取樣 Independent 142.38、Grid 159.01、h + random u 126.98、h + midpoint u 119.34；無邊界取樣 142.61 / 3.81。
- Sudoku（Table 22，%，1024 樣本，$|V|=10$，長度 81）：FMLM 無條件 1／2／4／1024 步 9.38 / 57.03 / 79.69 / 94.92；held-out 條件 39.84 / 85.15 / 92.19 / 98.83；Sudoku-Extreme 16.01 / 20.70 / 25.00 / 26.56；離散基線一步皆 0.00。論文稱一步配合 rejection sampling 相對 81 步自回歸約 7.6× 加速，未給算式。〔本頁推導，近似〕$81\times0.0938\approx7.6$ 吻合，但這個讀法含兩個假設：一次 FMLM 前向與一次自回歸步成本相同（自回歸有 KV cache，實際每步較便宜），以及拒絕取樣的期望嘗試次數為 $1/0.0938$。頁面只寫「可由 81 × 9.38% 近似得到」。
- 擴展（Table 27，OWT 150k 步）：179M / 424M / 870M 得 75.57 / 65.94 / 61.59。

#### 步驟 13：常見錯誤（頁面 S10 逐條列出）

1. 時間方向弄反：本文 $t=0$ 噪聲、$t=1$ 資料。
2. 一步生成從 softmax 抽樣：Alg. 8 是確定性合成後 argmax（$g$）。
3. 把 $\delta_{s,t}$ 當成 $x_t$ 處的 denoiser 或終點預測：它的輸入是 $x_s$，是 $s\to t$ 沿軌跡 denoiser 的加權平均（Eq. 82），只有 $s=t$ 時等於 $D_s$。
4. 把 denoiser 當成一步 flow map：$D_t$ 是步長 $1-t$ 的 Euler 步、後驗平均（Eq. 77）。
5. $\gamma$ 用 $t$ 空間中點公式：中點取在 $\tau$ 空間，$\gamma$ 必須用一般式。
6. 在 $t\to1$ 直接用 Eq. 9 換算速度：$(1-t)^{-1}$ 奇異；Alg. 6 只在 $t_n<1$ 評估 denoiser。
7. 省略邊界取樣：一步 142.61 / 3.81（Table 18）。
8. 只看 Gen. PPL：Simplex 85.07 但 entropy 3.76；Duo+Di4C 292.94 但 3.79。
9. 以為 $\tau(t)$ 是學出來的：它是預先算好的查表 + spline。
10. 取樣器：App. E 寫 FLM 用 Euler、FMLM 用 Kim et al. [91] 的「γ-sampling」（γ 值未給，與半群 γ 同符號不同義）；App. G 又稱兩者為 deterministic sampling，與 Alg. 6、8 一致。論文引用 [91, 92] 取最佳 γ 值，但沒有說明 γ-sampling 與 Alg. 8 的關係，兩者未調和。頁面以 Alg. 8（論文自己的虛擬碼）為準並註明此出入（R5）。

#### 步驟 14：限制（Sec. 6）

one-hot 表示每步要對完整的 $|V|\times d$ embedding 矩陣求值與反向傳播，比只更新相關 embedding 的 embedding diffusion 多約 30% 時間與記憶體（Sec. 6）；App. F 另寫比 MDLM 多 15–20% 記憶體但 wall-clock 較快（Table 32）。作者建議以稀疏梯度或結構化表示改善，並看好用於 RL finetuning 的 rollout。

---

## 2. 學習目標（驗收測試）

每題的參考答案必須寫在指定章節的**正文**（不能只在元件、表格、圖或答案框中）。每章結尾以 `<details>`「檢查理解」照搬題目與答案。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明離散擴散少步失敗的原因 | Eq. 2 做了什麼近似？為何步數少時出錯？new york / san diego 玩具一步取樣得到什麼？ | 把 $p_{t\mid s}(y_t\mid y_s)$ 近似成逐 token 條件機率的乘積，假設去噪後的 token 在 $y_s$ 下條件獨立；此假設只在 $t\to s$ 極限成立，步數少時忽略 token 相關；一步時四種組合各 25%，new diego 與 san york 合計 50% 無效（Fig. 3）。〔本頁推導〕完美模型、$N$ 步時無效率為 $1/(2N)$ | S1 |
| LO2 | 寫出 one-hot 表示與 interpolant | $f$、$g$ 是什麼？$I_t$ 怎麼寫？$t=0$、$t=1$ 各是什麼？狀態是否限制在 simplex？ | $f$ 把每個 token 換成 $\lvert V\rvert$ 維 one-hot，$g$ 逐位置取 argmax，$g(f(y))=y$（Eq. 4）；$I_t=(1-t)x_0+tx_1$，$x_0\sim\mathcal N(0,I)$（Eq. 5）；$t=0$ 純噪聲、$t=1$ 資料；狀態在無約束的 $\mathbb R^{L\times\lvert V\rvert}$，不投影到 simplex | S2 |
| LO3 | 說明 denoiser 與速度的關係及改學 denoiser 的理由 | $b_t$ 與 $D_t$ 的關係？為何不直接回歸速度？ | $b_t(x)=\mathbb E[x_1-x_0\mid I_t=x]=(D_t(x)-x)/(1-t)$（Eq. 7、9）；速度目標 $x_1-x_0$ 帶有高斯噪聲的 full-rank 結構，$\lvert V\rvert$ 遠大於網路內部維度 $d$ 時欠擬合；Table 18：Velocity + MSE 3801.36，Denoiser + softmax + CE 96.91 | S3a |
| LO4 | 解釋 Lemma 3.1 與交叉熵目標 | 為何最佳 denoiser 在 simplex 上？用什麼目標訓練？為何其唯一最小值是真 denoiser？ | $x_1^l$ 只取 one-hot 值 $e_i$，條件期望 $\sum_ie_i\,p(x_1^l=e_i\mid I_t=x)$ 就是後驗機率向量（Eq. 40–41）；所以用逐 token softmax、以交叉熵 Eq. 12 訓練；交叉熵 = 不可約條件熵 + 逐 token KL（Eq. 45），KL ≥ 0 且等號 iff 相等 | S3a |
| LO5 | 區分連續法與離散法如何使用同一個後驗 | 兩者都學逐 token 後驗 $p^l_{1\mid t}$，差別在哪？ | 離散法用它做 ancestral sampling，需要聯合分布，逐 token 獨立抽樣造成因子化誤差；連續法只用它算速度，速度是條件期望（對 $x_1$ 線性），逐 token 邊際就給出精確的 $b_t$；token 相關由兩件事帶入：每個位置的後驗都條件在整個序列的 $I_t$ 上，且確定性 flow 讓各位置耦合演化，$x_0$ 只提供隨機性（〔本頁推導〕，論文 Sec. 3.2 只說不需因子化近似） | S3b |
| LO6 | 說明時間重參數化 | $P_e(t)$ 的定義與端點？$\tau(t)$ 怎麼定義？訓練與取樣怎麼用？為何不用均勻 $t$？ | $P_e(t)=\frac1L\sum_lP(g^l(x_t)\ne g^l(x_1))$，$P_e(0)=1-1/\lvert V\rvert$、$P_e(1)=0$；$\tau(t)=1-\frac{\lvert V\rvert}{\lvert V\rvert-1}P_e(t)$（Eq. 24–25）；訓練取 $\tau(t)\sim U[0,1]$、取樣格點 $t_n=t(n/N)$；大 $\lvert V\rvert$ 時 token 在很窄的時間窗內決定，均勻 $t$ 浪費訓練訊號；消融（300k 步）106.98 對無重參數化 149.18 | S4 |
| LO7 | 定義 flow map 並區分它與 denoiser | flow map 的定義與三個刻畫條件？一步取樣怎麼做？denoiser 為何不是一步 flow map？ | $X_{s,t}(x_s)=x_t$，$X_{s,t}(x)=x+(t-s)v_{s,t}(x)$；$X_{s,s}=x$、$\lim_{s\to t}\partial_tX_{s,t}=b_t$、$X_{u,t}(X_{s,u}(x))=X_{s,t}(x)$（Eq. 14–16）；$\hat x_1=X_{0,1}(x_0)$；$D_t(x)=x+(1-t)b_t(x)$ 只是步長 $1-t$ 的一個 Euler 步，是後驗平均，會平均掉多模態（Eq. 77、App. C.2） | S5 |
| LO8 | 說出 two-time denoiser 的四個性質 | $\delta_{s,t}$ 的定義？Prop. 3.3 四性質？為何 $\delta$ 在 simplex 上而 $X_{s,t}$ 不在？ | $\delta_{s,t}(x)=x+(1-s)v_{s,t}(x)$；(i) $X_{s,t}=\frac{1-t}{1-s}x+\frac{t-s}{1-s}\delta_{s,t}$；(ii) $\delta^l_{s,t}\in\Delta^{\lvert V\rvert-1}$；(iii) $\delta_{s,s}=D_s$；(iv) $\delta_{s,t}=\gamma\delta_{s,u}+(1-\gamma)\delta_{u,t}(X_{s,u})$，$\gamma=\frac{(1-t)(u-s)}{(1-u)(t-s)}\in[0,1]$；$\delta$ 是沿軌跡 denoiser 的加權平均，權重非負且積分為 1（Eq. 82–83）；$X_{s,t}$ 含噪聲項 $\frac{1-t}{1-s}x$ | S6 |
| LO9 | 描述 FMLM 蒸餾 | Eq. 23 兩項各做什麼？Alg. 7 如何取 $(s,u,t)$？為何需要邊界取樣？ | 離對角項讓 $\delta_{s,t}(I_s)$ 逼近停止梯度的半群教師 $\mathrm{sg}(\gamma\delta_{s,u}+(1-\gamma)\delta_{u,t}(X_{s,u}))$，對角項讓 $\delta_{t,t}$ 逼近凍結的 FLM $\hat D_t$；$h\sim U[0,1]$、$\tau(s)\sim U[0,1-h]$、$\tau(t)=\tau(s)+h$、$\tau(u)$ 取 $\tau$ 空間中點；$\tau(t)$ 在 $t=0$ 附近平坦，$s$ 很少落在 0，所以以 $p=1/32$ 直接取 $(s,t)=(0,1)$；無邊界取樣一步 142.61 / 3.81，有則 119.34 / 4.16（Table 18） | S7 |
| LO10 | 讀懂 Prop. 3.5 與其適用範圍 | 命題說什麼？證明關鍵？它沒有說什麼？ | $S$ 有限時，對任意 $\mu$ 存在 $\nu$ 無法寫成確定性映射的 $f_\#\mu$；需 $\lvert S\rvert\ge2$；$\nu(y)$ 是 $\mu$ 值的子集和（Eq. 70），正值至少是最小正質量 $\mu_{\min}=\min\{\mu(x)\mid\mu(x)>0\}$，取 $\nu(y_0)=\mu_{\min}/2$ 即不可達；它沒有說真實資料分布不可達，也沒有量化誤差 | S8 |
| LO11 | 讀出主要實驗數字 | FLM 1024 步與 FMLM 一步在 LM1B、OWT 的 Gen. PPL / entropy？FMLM 一步相當於基線幾步？ | FLM：LM1B 96.91 / 4.29、OWT 62.23 / 5.33（Duo 98.14 / 4.31、77.69 / 5.55）；OWT 的 PPL 最低，但 entropy 5.33 低於資料 5.44（Duo 5.55 高於資料，兩者與資料各差 0.11，方向相反），所以 OWT 上 FLM 是「PPL 較低、entropy 略低」而非全面勝出；FMLM 一步：LM1B 119.34 / 4.16、OWT 168.30 / 5.17，最好的一步基線 Duo+Di4C 292.94 / 3.79、370.51 / 3.92；LM1B 相當於基線 8–16 步、OWT 4–8 步（Sec. 5.2） | S9 |
| LO12 | 說明為何 Gen. PPL 要與 entropy 一起看 | 為何不能只看 Gen. PPL？舉兩個例子。 | 重複 token 會讓 Gen. PPL 偏低，entropy 低（例如 < 4）代表重複；Simplex 擴散 PPL 85.07 但 entropy 3.76；Duo+Di4C 一步 292.94 但 entropy 3.79；FMLM 一步 entropy 也低於資料（LM1B 4.16 對 4.31）；FLM 1024 步在 OWT 的 entropy 只略低於資料（5.33 對 5.44，差 0.11），程度遠小於前兩例 | S9 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），圖版加寬至 1080px，章節編號用 JetBrains Mono（`01 · …`），標題為陳述句。桌機左側 sticky 章節目錄，窄螢幕頂部橫列（沿用 238）。

| # | id | 標題 | 內容（依序；每個符號先定義再使用） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈Flow Map Language Models：用連續去噪一步生成文字〉 | 副標「把 token 寫成 one-hot 向量、在連續空間學 flow，再蒸餾成 flow map，一次網路計算生成整段文字。」三張事實卡（見下）。連結：arXiv、`../index.html#paper-4`（在關聯圖中查看）、`fmlm-lineage-overview.html` | 1 螢幕 |
| S1 | `discrete` | 01 · 離散擴散一次更新多個 token，步數少時會產生不存在的組合 | token、$V$、$L$、Eq. 1 與序列瓶頸；離散擴散、遮罩、ancestral sampling 各一句定義（步驟 1）、Eq. 2；條件獨立只在 $t\to s$ 成立（一句解釋「$t\to s$」）；玩具定義（只有 new york、san diego）；C1a；正文寫出 LO1 答案。延伸：`fmlm-parallel-decoding-motivation.html`、`fmlm-discrete-diffusion-factorization.html` | 450–550 字 + C1a |
| S2 | `onehot` | 02 · 把 token 寫成 one-hot 向量，在歐氏空間做 flow matching | Eq. 3–4；simplex 定義（一句 + $\lvert V\rvert=3$ 靜態三角形 SVG，標出頂點 one-hot 與內部機率向量）；Eq. 5–8（Eq. 7 前一句說明條件期望）；時間方向；Euler 法 2–3 句；可收合符號表（只作查閱，所有符號在正文首次出現處定義）。LO2。延伸：`fmlm-continuous-text-diffusion.html`、`fmlm-flow-matching-interpolant.html` | 400–500 字 + 1 靜態 SVG |
| S3a | `denoiser` | 03 · 最佳 denoiser 是逐 token 後驗，所以輸出層用 softmax | Eq. 9（可收合推導 Eq. 74–76）；rank bottleneck（秩瓶頸，正文定義見步驟 4）與 3801.36；$p^l_{1\mid t}$ 正文定義；Lemma 3.1（Eq. 40–41）；Eq. 12、42、45。LO3、LO4。延伸：`x-prediction-manifold-view.html` | 400–500 字 |
| S3b | `crossent` | 04 · 交叉熵訓練與離散擴散用的是同一個後驗，用途不同 | Eq. 13 一句結果（$W_2$、$\xi$、Gronwall、證明鏈與〔本頁推導〕$C=C_D^2$ 放 `<details>`）；與離散法的差別（callout，步驟 6）；Alg. 5–6 兩欄對照。LO5。延伸：`cdlm-training-objective.html` | 300–400 字 |
| S4 | `time` | 05 · 時間重參數化把訓練與取樣集中在 token 被決定的時間窗 | Eq. 24–25 文字定義（$P_e$ 沿用自 Duo、CANDI）；C3 與觀察；〔本頁推導，interpolant 耦合讀法〕$P_e$ 閉式放 `<details>`；LUT + spline（App. E 描述）、$\tau$ 非學習；消融數字；Prop. C.12 一句；Dir(1) 先驗一段。LO6。延伸：`cdlm-noise-schedule.html` | 450–550 字 + C3 |
| S5 | `flowmap` | 06 · flow map 從時間 s 的狀態直接算出時間 t 的狀態 | Eq. 14–16、Eq. 17 一段、Lagrangian／Eulerian 一句；Eq. 77（denoiser = 一個 Euler 步）；C1b；正文寫明「Euler 一步 = 後驗平均」與「flow map 多步合成 = 一步」。LO7。延伸：`fmlm-flow-map-distillation-lineage.html`、`flow-map-few-step-distillation.html` | 450–550 字 + C1b |
| S6 | `delta` | 07 · two-time denoiser 把 flow map 放回 simplex 上 | Eq. 18；Prop. 3.3 四性質（Eq. 19–22）；可收合推導 Eq. 79–83 與 Eq. 94–98；中點 γ；〔本頁推導〕$t=1$ 時 $X_{s,1}=\delta_{s,1}$；C2。LO8。延伸：`fmlm-onehot-denoiser-two-time.html` | 500–600 字 + C2 |
| S7 | `distill` | 08 · 以 KL 半群目標把 FLM 蒸餾成 FMLM | Eq. 23；KL 與 CE 只差常數；Alg. 7–8 兩欄對照；三元組、$\tau$ 空間中點、邊界 1/32、batch 對半；self-distillation（Eq. 104）；Lagrangian／Eulerian 與 Table 18 數字；App. D 一句。LO9 | 450–550 字 |
| S8 | `nomap` | 09 · 離散狀態空間沒有樣本層級的 flow map | pushforward 一句定義 → C4 直覺 → Prop. 3.5（$\lvert S\rvert\ge2$、$\mu_{\min}$ 為最小正質量）與 Eq. 70 → CTMC、master equation 各一句定義；「它沒有說的事」callout。LO10 | 300–400 字 + C4 |
| S9 | `results` | 10 · FLM 的生成困惑度追平離散擴散，FMLM 一步勝過少步蒸餾基線 | 評估方法（Gen. PPL、entropy）；Table 10（正文寫 OWT entropy 5.33 對資料 5.44）、Table 13（全表，sticky 首欄）、Table 15；Table 18 摘錄（FLM 組：Velocity + MSE、Denoiser + MSE、Denoiser + softmax + MSE、+ CE；FMLM 組：Average velocity + MSE、δ-denoiser + softmax + CE、Eulerian、Lagrangian、self-distillation、無邊界取樣；300k 步子表另標：四種時間重參數化、Simplex、Riemannian）；Self-BLEU 一句（目標是接近資料）；callout「論文中『一步 vs 8 步』的四種說法，本頁採 Sec. 5.2」。收合 `<details>`「其他結果」（無 LO）：guidance（Eq. 26–27、96.91 → 51.62、Table 17 一列）、Sudoku 一段、Table 27 一句。LO11、LO12 | 400–500 字 + 3 表 + 收合段 |
| S10 | `impl` | 11 · 實作設定與常見錯誤 | 設定表（步驟 12 第一點 + App. E：LUT、batch 對半、取樣器）；10 條常見錯誤（步驟 13） | 300–400 字 + 表 |
| S11 | `history` | 12 · 研究脈絡（由 notes/timeline.json 提供） | 時間線 + 一段把六條脈絡接到本文 | 時間線 + 150 字 |
| S12 | `next` | 13 · 限制與延伸閱讀 | Sec. 6 限制（30% vs 15–20% 出入一句）；筆記清單；arXiv | 200 字 + 連結 |

Hero 三張事實卡（不使用未定義術語）：
1. 「離散擴散語言模型每步平行產生多個 token，但各 token 是獨立抽樣的；步數一少，就會拼出資料中不存在的組合。」
2. 「本文把每個 token 寫成 one-hot 向量，在連續空間中讓噪聲沿確定的路徑流向資料；網路只需要預測每個位置的 token 機率。」
3. 「蒸餾後的模型只做一次網路計算就生成整段文字。以 GPT-2 Large 計算的生成困惑度（越低越好）在 LM1B 上是 119.34；同樣只算一次的離散擴散基線在 LM1B 上最好是 292.94（Table 13）。」

---

## 4. 互動元件規格

### 4.0 共用數學：二位置二詞玩具（C1a、C1b、C2 共用一份函式）

- **標籤與數學排版（全部元件）**：SVG `<text>` 與動態產生的 DOM 一律用 Unicode 純文字，不放 TeX：例如「x¹ new」「x² york」「δ s,t」「δ s,u」「δ u,t(X s,u)」「X s,t」「γ」「τ」。`renderMath()` 只在初始化時對靜態節點執行一次，不對重建的 live region、讀數表或 SVG 執行；讀數一律純文字（KaTeX 會在 live region 內加一份 MathML，螢幕閱讀器重複朗讀）。本節下文以 TeX 寫的標籤只是設計文件記法，實作時照此規則轉成 Unicode。

- 設定（頁面寫出）：$L=2$；位置 1 詞表 {new, san}，位置 2 詞表 {york, diego}，$|V|=2$。資料只有 new york、san diego，各 50%。狀態 $x=(x^1_1,x^1_2,x^2_1,x^2_2)\in\mathbb R^4$；one-hot：new york $m_{NY}=(1,0,1,0)$、san diego $m_{SD}=(0,1,0,1)$。
- 〔本頁推導〕後驗：$p(x\mid k)\propto\exp(-|x-tm_k|^2/(2(1-t)^2))$，$|m_k|$ 相同，所以 $p(NY\mid I_t=x)=\mathrm{sigmoid}(t\,u/(1-t)^2)$，$u:=(x^1_1-x^1_2)+(x^2_1-x^2_2)$。denoiser $D_t(x)=(p,1-p,p,1-p)$。sigmoid 用穩定式（$z\ge0$：$1/(1+e^{-z})$；否則 $e^z/(1+e^z)$）。$t\ge1$ 時 $p=\mathbb 1[u>0]$（$u=0$ 取 0.5）。
- 〔本頁推導〕座標分解：$a_l=x^l_1-x^l_2$、$\sigma_l=x^l_1+x^l_2$、$u=a_1+a_2$、$w=a_1-a_2$。精確 flow：$w_t=w_s\frac{1-t}{1-s}$；$1-\sigma_{l,t}=(1-\sigma_{l,s})\frac{1-t}{1-s}$；只有 $u$ 需要數值積分：
  $$\frac{du}{dt}=\frac{2\tanh\!\big(k(t)\,u\big)-u}{1-t},\qquad k(t)=\frac{t}{2(1-t)^2}.$$
  $u=0$ 是不變點，$u$ 的正負號在 flow 下保持不變；$t\to1$ 時 $u\to2\,\mathrm{sgn}(u)$，終點恰為 one-hot。
- `flowU(u, s, t)`（$0\le s\le t\le1$）：改用 $r=-\ln(1-t)$（$dt=(1-t)dr$）消去 $1/(1-t)$，$du/dr=2\tanh(k u)-u$。迴圈：若 $|k(r)\,u|>20$（tanh 已飽和，誤差 < 1e-17），改用精確線性尾段 $u\leftarrow2\,\mathrm{sgn}(u)+(u-2\,\mathrm{sgn}(u))e^{-(r_t-r)}$ 並結束（$t=1$ 時 $e^{-\infty}=0$）；否則 RK4，步長 $h=\min(0.01,\ 0.5/(2k+1),\ r_t-r)$。保護：$u=0$ 直接回傳；RK4 後若正負號改變則設 $u=0$；超過 20,000 步回傳 `{u, failed:true}`（不再靜默回傳 0）；呼叫端把該樣本另計「數值不收斂」，讀數顯示「數值不收斂 k 個」（k = 0 時隱藏），不併入平手或有效。正常情況回傳 `{u, failed:false}`。`toy.py` 驗證：步長 0.01 與 0.001 在 $[0,0.9]$ 的最大差 9.8e-9。
- `X(x, s, t)`：轉座標、呼叫 `flowU`、轉回（$a_1=(u+w)/2$、$a_2=(u-w)/2$、$x^l_1=(\sigma_l+a_l)/2$、$x^l_2=(\sigma_l-a_l)/2$）。要求 $s<1$。
- `delta(x, s, t)`：$t-s<10^{-12}$ 時回傳 $D_s(x)$；否則 $((1-s)X_{s,t}(x)-(1-t)x)/(t-s)$（Eq. 19 反解）。`gamma(s,u,t)`：$t\ge1$ 時 0，否則 Eq. 22。
- 取樣器：Euler（Alg. 6，精確 $D$）：$x\leftarrow x+(t_{n+1}-t_n)(D_{t_n}(x)-x)/(1-t_n)$；flow map（Alg. 8，精確 $\delta$，等同 $x\leftarrow X_{t_n,t_{n+1}}(x)$）。玩具格點用等距 $t_n=n/N$（$|V|=2$，不做 $\tau$ 重參數化，頁面註明）。
- 分類：位置差 $|a_l|<10^{-12}$ 為平手；否則 (+,+) new york、(−,−) san diego、(+,−) new diego、(−,+) san york。距離指標 $\lVert\hat x_1-f(g(\hat x_1))\rVert_2$（平手時取第一個詞）。
- PRNG（與 238 相同）：`mulberry32(seed)`；`gauss()` 每次取兩個均勻數，回傳 $\sqrt{-2\ln(1-u_1)}\cos(2\pi u_2)$。每個樣本依序抽 $x^1_1,x^1_2,x^2_1,x^2_2$，共 200 個，預設 seed 7。

### C1a · 離散擴散的因子化誤差（S1）

- **教的概念**：逐 token 獨立抽樣在步數少時拼出無效組合；誤差來自 Eq. 2，與模型準不準無關。
- **簡化**（頁面寫出）：本頁自選的設定——masked diffusion（遮罩 = token 被隱藏）、線性遮罩排程、等距 $N$ 步、完美模型。圖版註明「只有 N = 1 的 25% 與論文 Fig. 3 一致；Fig. 3 未說明所用的離散模型與排程」。
- **讀者操作**：分段按鈕 $N\in\{1,2,4,8,16,32,64\}$，預設 1；按鈕「重新抽 16 個樣本」。
- **計算**：閉式 〔本頁推導〕：無效率 $1/(2N)$；new diego = san york $=1/(4N)$；new york = san diego $=(1-1/(2N))/2$。樣本列：16 次模擬，每步 $s\to t$ 中仍遮罩的位置以機率 $(t-s)/(1-s)$ 揭露；同一步揭露的位置各自依「給定 $y_s$ 的條件」抽樣（另一位置已揭露則跟隨它，否則 50/50）。PRNG 用 mulberry32（seed 101 起，每按一次加 1）。
- **顯示**：四根橫條（new york、san diego、new diego、san york），無效兩根加斜線紋理並標「無效」，條尾標百分比（一位小數）；下方 16 個樣本 chip（文字為兩個詞，無效 chip 加刪除線與「無效」字樣），chip 區上方標「16 個隨機樣本，與上方閉式值不必相符」。一行讀數「無效率 = 1/(2N) = 50.0%」。
- **讀者應觀察到**：$N=1$ 四種組合各 25%（Fig. 3）；$N$ 每加倍無效率減半，要很多步才接近 0。
- **無障礙**：橫條以紋理與文字區分有效／無效；chip 以刪除線 + 文字標示；讀數 `aria-live="polite"`，於按鈕切換後更新。

### C1b · Euler 與 flow map：同一批噪聲、兩種取樣器（S5）

- **教的概念**：連續 flow 的確定性映射保留 token 相關；Euler 一步只得到後驗平均；精確 flow map 一步就到 one-hot，多步合成結果與一步相同（Eq. 16 半群）。
- **讀者操作**：分段按鈕「取樣器」Euler（Alg. 6）／flow map（Alg. 8），預設 Euler；分段按鈕 $N\in\{1,2,4,8,16,32\}$，預設 1；按鈕「換一個突顯樣本」（循環索引）；在圖上點擊／輕觸最近的點也可突顯；「重新抽樣」（seed + 1）。
- **顯示**：
  - 正方形平面，橫軸 $a_1=x^1_{new}-x^1_{san}$（SVG 標籤「a₁ = x¹ new − x¹ san」），縱軸 $a_2=x^2_{york}-x^2_{diego}$（「a₂ = x² york − x² diego」），範圍 $[-4,4]^2$（超出者畫在邊界並改用三角形標記）。四個象限文字標籤；new diego、san york 兩象限加淡斜線紋理並標「無效」。終點 one-hot 位置 $(1,1)$、$(-1,-1)$ 以十字標出。
  - 200 個起點（淺色小圓）、所選取樣器的 200 個終點（實心）。突顯樣本：精確軌跡（細實線，200 個顯示時間點，逐段 `flowU`）；取樣器的迭代點（Euler：空心圓 + 直線段；flow map：實心方塊，落在精確軌跡上）。
  - 讀數表（純文字）：五類計數（new york、san diego、new diego、san york、平手；`flowU` 失敗時另列「數值不收斂」）、平均 $\lVert\hat x_1-\text{one-hot}\rVert$（三位小數）；同一 $N$ 下 C1a 的離散無效率（閉式）並列一行「離散（C1a，同 N）無效率」。
  - 控制列旁（不是圖版底部）一行說明：「本頁玩具，非論文數字。這個玩具要看的是：Euler 一步 = 兩個模態的平均；真實語言模型的少步退化見第 10 節（Fig. 11）。」不在圖版文字中強調「N ≥ 8 已近乎精確」，該事實只留在 R11。
- **預設值的正確輸出**：見 5.2。
- **讀者應觀察到**：
  0. 正文先寫結論：Euler 一步 = 兩個模態的平均；flow map 一步就到資料。
  1. Euler $N=1$：200 個終點全部疊在原點（每個位置都是 0.5／0.5，平手），因為 $D_0(x_0)$ 不含資訊，是兩個模態的平均。
  2. flow map $N=1$：終點只在 new york 與 san diego 兩點，無效 0；$N$ 改變時終點不動（半群）。
  3. 突顯樣本（seed 7 的索引 2，0 起算，即第 3 個樣本）一開始位置 1 偏 san、位置 2 偏 york，最後兩個位置一起落在 san diego：每個位置的後驗看的是兩個位置的座標（$u=a_1+a_2$），確定性 flow 讓兩個位置一起演化；$x_0$ 只決定起點。
  4. Euler 隨 $N$ 增加逐漸逼近 one-hot（$N=2$ 平均距離 0.441、$N=4$ 0.032）。
- **無障礙**：點的形狀區分起點／終點／迭代點；象限以紋理 + 文字；圖 `role="img"` 的 `aria-label` 摘要「Euler，N = 1：200 個樣本全部平手」；讀數 live region 於按鈕切換後更新；鍵盤以「換一個突顯樣本」按鈕取代點擊。

### C2 · two-time denoiser 在 simplex 上、滿足半群（S6）

- **教的概念**：Prop. 3.3：$X_{s,t}$ 離開 simplex、$\delta_{s,t}$ 留在 simplex；$\delta_{s,t}$ 等於兩段 δ 的凸組合（Eq. 22）；$\delta_{s,t}$ 是沿軌跡 denoiser 的加權平均（Eq. 82）。對應 Fig. 4。
- **分層顯示**：預設只顯示滑桿 $s$、$t$，兩個位置面板中的 $x_s$、$X_{s,t}$、$\delta_{s,t}$ 與 simplex 線段，讀數只有 $X_{s,t}$ 與 $\delta_{s,t}$ 的座標和。`<details>`「進階：半群與 Eq. 82」展開後才出現 $u$ 滑桿與中點勾選、$X_{s,u}$、$\delta_{s,u}$、$\delta_{u,t}(X_{s,u})$、教師圓環、Eq. 82 面板，以及讀數 γ、$X_{s,u}$ 座標和、教師差。兩個重設按鈕合併成一個「回到預設」（重設滑桿與 $x_0$）。
- **讀者操作**：滑桿 $s$、$t$（步長 0.01；預設 0.1、0.9）；進階中滑桿 $u$（預設 0.5）與勾選「u 取中點 (s+t)/2」（預設勾選，勾選時 $u$ 滑桿停用）；按鈕「重新抽樣 x₀」（seed 11 起，每次加 1）與「回到預設」。
- **約束與保護**：$0\le s\le0.97$；$t\in[s+0.02,\,1]$；$s\le u\le t$ 且 $u\le0.99$。移動任一滑桿後依序夾住其他滑桿。$t-s\ge0.02$ 讓 Eq. 19 反解的抵銷誤差 < 1e-7。$t=1$ 時 $\gamma=0$，Eq. 82 面板改顯示一行說明「t = 1 時 δ_{s,1} = X_{s,1}，權重集中在 τ → 1」。
- **狀態**：預設 $x_0=(0.9,-0.2,-0.4,0.3)$（位置 1 偏 new、位置 2 偏 diego，$u_0=0.4$）；$x=x_s=X_{0,s}(x_0)$。
- **計算**：$X_{s,u}(x)$、$X_{s,t}(x)$、$\delta_{s,u}(x)$、$\delta_{u,t}(X_{s,u}(x))$、$\delta_{s,t}(x)$、$\gamma$、教師 $\gamma\delta_{s,u}+(1-\gamma)\delta_{u,t}$、$D_s(x)$；軌跡 $X_{s,\tau}(x)$ 於 60 個等距 $\tau$；Eq. 82 求積：Simpson、400 等分，逐段以 `flowU` 推進軌跡，$q_{82}=\frac{(1-s)(1-t)}{t-s}\int_s^t p_\tau/(1-\tau)^2d\tau$（$p_\tau$ 為 $D_\tau(X_{s,\tau})$ 的 new 機率）。
- **顯示**：
  - 兩個正方形面板（位置 1：橫軸「x¹ new」、縱軸「x¹ san」；位置 2：「x² york」、「x² diego」），範圍 $[-1.5,2]^2$，超出者畫在邊界改用三角形。simplex $\Delta^1$ 為 $(1,0)$–$(0,1)$ 粗線段並標「simplex」。標記：$x_s$ 空心圓、$X_{s,u}$ 空心方塊、$X_{s,t}$ 空心菱形（三者以虛線軌跡連接）；$\delta_{s,u}$ 實心方塊、$\delta_{u,t}(X_{s,u})$ 實心三角、$\delta_{s,t}$ `--accent` 實心菱形、教師以外圈圓環套在 $\delta_{s,t}$ 上；每個標記有直接 Unicode 文字標籤（「δ s,t」等；窄螢幕改為圖例編號，預設 3 個、進階 6 個）。
  - 第三面板「Eq. 82：δ 是沿軌跡 denoiser 的加權平均」：橫軸 $\tau\in[s,t]$；$p_\tau$ 曲線（`--c-par` 實線）；權重 $w(\tau)=\frac{(1-s)(1-t)}{(t-s)(1-\tau)^2}$ 除以最大值後填色（`--surface-2`，加斜線紋理）；水平線 $q_{82}$（`--accent` 虛線），標「δ s,t 的 new 機率」。
  - 讀數（純文字，四位小數）：預設「X s,t 座標和」「δ s,t 座標和（= 1）」；進階加「γ」「X s,u 座標和」「|δ s,t − 教師| 最大值」（科學記號）。共 4 類（γ、三個座標和、教師差）。各 δ 的 new／york 機率與 $q_{82}$ 對反解值不放讀數，改在 SVG 標籤旁以小字顯示（進階），selfTest 仍檢查。
- **讀者應觀察到**：
  1. $X_{s,t}$ 不在線段上（預設座標和 0.97 與 0.89），$\delta_{s,t}$ 在線段上；進階中 $X_{s,u}$（0.85 與 0.45）同樣不在線段上，三個 δ 都在。
  2. （進階）教師與 $\delta_{s,t}$ 重合（差 < 1e-8）；預設 $\gamma=0.1$，教師幾乎全來自第二段 $\delta_{u,t}$。
  3. （進階）位置 2：$X_{s,u}$ 的 diego 座標 0.368 大於 york 座標 0.082；$\delta_{s,t}$ 的 york 機率是 0.939。兩者偏向不同的詞，因為 $\delta_{s,t}$ 是 $s$ 到 $t$ 整段軌跡上 denoiser 的加權平均，權重集中在 $\tau$ 接近 $t$ 的一端（Eq. 82）。
  4. 把 $t$ 拉到 1：$\delta_{s,1}=X_{s,1}=$ one-hot，$\gamma=0$。
  5. （進階）Eq. 82 面板：權重在 $\tau\to t$ 端最大，$q_{82}$ 與反解值一致。
- **無障礙**：標記以形狀 + 文字區分；滑桿 `aria-valuetext`（「s = 0.10」）；讀數 live region 於 `change` 或停止輸入 500ms 後更新；SVG `aria-label` 摘要 γ 與 δ 值。

### C3 · 解碼錯誤率與時間重參數化（S4）

- **教的概念**：$P_e(t)$ 在大 $|V|$ 時於窄時間窗內從接近 1 掉到 0；$\tau$ 格點把步數放進這個窗，等距 $t$ 格點大多落在窗外（Eq. 24–25、Fig. 9）。
- **讀者操作**：分段按鈕 $|V|\in\{2,10,100,1000,30{,}522\text{（LM1B）},50{,}257\text{（OWT）}\}$，預設 30,522；分段按鈕 $N\in\{4,8,16,32,64\}$，預設 16。
- **計算**（〔本頁推導〕閉式，見步驟 7）：
  - $P_e(t)=1-\int_{-8}^{8}\varphi(z)\exp\big((|V|-1)\log\Phi(z+\tfrac t{1-t})\big)dz$，梯形法 $h=0.01$（1,601 點）；$t\ge1$ 時回傳 0。
  - $\log\Phi(x)$：令 $z=-x/\sqrt2$；$z\ge0$：$\ln0.5+\mathrm{logErfc}(z)$；$z<0$：$\mathrm{log1p}(-0.5\,e^{\mathrm{logErfc}(-z)})$。`logErfc`（$z\ge0$）用 Numerical Recipes erfcc 的對數形式（$t=1/(1+z/2)$，$\ln t+(-z^2-1.26551223+t(1.00002368+\dots))$，相對誤差 < 1.2e-7）。這樣 $|V|=50{,}257$ 時不會下溢或在 $\Phi\approx1$ 時抵銷。`c3.py` 與 `math.erfc`、$h=0.002$ 比較差 < 1e-8。
  - LUT：$t_i=i/1000$（1,001 點），$\tau_i=(P_e(t_0)-P_e(t_i))/P_e(t_0)$ 夾在 [0,1]（用數值 $P_e(0)$ 使端點精確為 0；與 $1-1/|V|$ 差 < 1e-6）。$t(\tau)$ 以線性內插反查（$\tau$ 單調，已驗證）。對應論文的 1,000 點 LUT（論文用 cubic spline，本頁用線性內插，差 ≤ 0.001，頁面註明）。
  - 背景淡曲線：六個 $|V|$ 各 201 點，所選 $|V|$ 的 1,001 點 LUT；全部以 `requestIdleCallback`（不支援時用 `setTimeout(…,0)`）每批 50 個 $t$ 分段計算（冷啟動約 1,601 × 1,001 ≈ 1.6M 次 logErfc，手機未測，不承諾時間）。計算中主圖顯示「計算中…」與已完成的部分，分段按鈕保持可用；切換時取消未完成的批次。結果以 `Map` 依 $|V|$ 快取（含六條背景曲線），C3 進入視窗（`IntersectionObserver`）才開始計算。
- **顯示**：
  - 主圖：橫軸「t」，縱軸「Pₑ」（Unicode 標籤）；六條淡虛線（直接標「|V| = …」），所選者粗實線；色帶 $[t(0.05),t(0.95)]$ 標「90% 解碼進度」。軸下兩列刻度：「等距 t 格點 n/N」（空心短線）與「τ 格點 t(n/N)」（實心長線）。
  - 副圖：$\tau(t)$ 曲線，對角虛線 $\tau=t$ 作對照。
  - 讀數：$P_e(0)$、$P_e(0.5)$、$P_e(0.8)$；$t(0.05)$、$t(0.5)$、$t(0.95)$；「落在窗內的格點（共 N+1 個）：等距 k 個；τ 格點 m 個」。
- **讀者應觀察到**：
  1. $|V|$ 越大，曲線越晚、越陡地落下；$|V|=30{,}522$ 時窗約在 $t\approx0.71$–$0.85$。
  2. $N=16$ 時等距格點只有 2 個落在窗內，$\tau$ 格點有 15 個。
  3. $|V|=2$ 時曲線平緩，等距格點已足夠（10 / 17）：重參數化對大詞表才關鍵。
- **頁面文字**：圖版標「本頁推導」，註明「曲線由本頁閉式計算，把 Eq. 24 的 $x_t$ 讀成 interpolant $I_t$；論文未寫出公式（App. E 只描述以 Gauss–Hermite 求積建表），這不是 Fig. 9 的重現」；窗 0.71–0.85 與格點計數同樣標本頁推導；不寫「貼近 t = 1」。閉式在 C3 之後的 `<details>` 才出現（步驟 7）。
- **無障礙**：曲線直接標籤；兩列刻度以空心／實心與長短區分；live region 於按鈕切換後更新。

### C4 · 確定性映射能推出哪些分布（S8）

- **教的概念**：Prop. 3.5：有限集合上的確定性映射只能把 $\mu$ 的機率「整塊搬移」，$\nu(y)$ 只能是 $\mu$ 值的子集和（Eq. 70）。
- **讀者操作**：分段按鈕 $\mu$：A $(0.4,0.3,0.2,0.1)$（預設）、B 均勻 $(0.25\times4)$、C 點質量 $(1,0,0,0)$；滑桿目標 $\nu(y_0)\in[0,1]$，步長 0.005，預設 0.05。
- **計算**：列舉 $4^4=256$ 個 $f:S\to S$，對每個 $f$ 算 $\nu=f_\#\mu$（四捨五入到 1e-10 後去重）；可達的 $\nu(y_0)$ 集合；不同 $\nu$ 向量個數；$\mu_{\min}/2$；目標與最近可達值的差。
- **顯示**：$\mu$ 的四根直條；數線 [0,1] 上以實心點標出所有可達的 $\nu(y_0)$、以空心菱形標 $\mu_{\min}/2$、以 `--accent` 指標標目標；讀數：「可達／不可達」、最近可達值、差距、「256 個映射推出 N 種不同的 ν」。下方一行對照：「連續空間中，高斯經單調映射可推出任何一維連續分布」（文字，不計算）。
- **預設值的正確輸出**：見 5.5。
- **讀者應觀察到**：A 的可達值是 0, 0.1, …, 1.0，0.05 不可達；B 只有 0、0.25、0.5、0.75、1；C 只有 0 與 1。C 對應 masked diffusion 從全遮罩出發的情形。
- **無障礙**：可達點與目標以形狀 + 文字區分；滑桿 `aria-valuetext`；live region 於 `change` 後更新。

---

## 5. 驗收數值表

全部由 `tut/4/verify/` 腳本印出（`python3 toy.py`、`c2.py`、`c3.py`、`c4.py`）。頁面與腳本演算法相同，容許誤差只涵蓋浮點加總順序與 erfc 近似。

### 5.1 C1a（閉式，`toy.py`；蒙地卡羅 10 萬次交叉驗證，差 < 0.002；頁面閉式值容許 ±1e-9，顯示四捨五入到表中位數）

| N | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 無效率 | 50.0% | 25.0% | 12.5% | 6.25% | 3.13% | 1.56% | 0.78% |
| new york（= san diego） | 25.0% | 37.5% | 43.8% | 46.9% | 48.4% | 49.2% | 49.6% |

### 5.2 C1b（seed 7、200 個樣本，`toy.py`；計數須完全一致，距離 ±0.001）

| 取樣器 | N | new york | san diego | 無效 | 平手 | 平均距離 |
| --- | --- | --- | --- | --- | --- | --- |
| Euler | 1 | 0 | 0 | 0 | 200 | 1.000 |
| Euler | 2 | 98 | 102 | 0 | 0 | 0.441 |
| Euler | 4 | 98 | 102 | 0 | 0 | 0.032 |
| Euler | 8 / 16 / 32 | 98 | 102 | 0 | 0 | 0.000 |
| flow map | 1–32 | 98 | 102 | 0 | 0 | 0.000（與 N = 1 終點差 0） |

- 第一個樣本 $x_0=(0.1420,-0.8643,-1.0064,0.0709)$（PRNG 對拍用）。
- 突顯樣本規則：第一個滿足 $|u_0|>0.5$ 且 $a_1a_2<0$ 的樣本 → 索引 2（0 起算），$x_0=(-0.8134,0.8304,0.5425,0.4031)$，$u_0=-1.5044$；精確軌跡 $X_{0,0.5}=(-0.2586,0.7671,0.4194,0.5535)$、$X_{0,0.9}=(-0.0498,0.9515,0.0858,0.9087)$、$X_{0,1}=(0,1,0,1)$；Euler $N=2$ 終點 $(0.1818,0.8182,0.1818,0.8182)$。容許 ±1e-4。

### 5.3 C2（預設 $x_0=(0.9,-0.2,-0.4,0.3)$，`c2.py`，容許 ±1e-4；差值欄為上限）

| (s, u, t) | γ | $\delta_{s,u}$ new | $\delta_{u,t}(X_{s,u})$ new | $\delta_{s,t}$ new | $X_{s,u}$ 座標和 | $X_{s,t}$ 座標和 | 教師差 | Eq. 82 差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1, 0.5, 0.9 | 0.1000 | 0.5711 | 0.9795 | 0.9386 | 0.85 / 0.45 | 0.97 / 0.89 | < 1e-8 | < 1e-6 |
| 0.3, 0.6, 0.95 | 0.0577 | 0.6661 | 0.9978 | 0.9787 | 0.88 / 0.56 | 0.985 / 0.945 | < 1e-8 | < 1e-6 |
| 0, 0.25, 0.5 | 0.3333 | 0.5171 | 0.5872 | 0.5638 | 0.775 / 0.175 | 0.85 / 0.45 | < 1e-8 | < 1e-6 |
| 0, 0.5, 1 | 0 | 0.5638 | 1.0000 | 1.0000 | 0.85 / 0.45 | 1 / 1 | 0 | （t = 1 不顯示） |

- 預設的完整向量：$x_s=(0.8606,-0.1306,-0.3094,0.3194)$；$X_{s,u}=(0.7319,0.1181,0.0819,0.3681)$；$X_{s,t}=(0.9300,0.0400,0.8000,0.0900)$；$D_s(x)$ new 機率 0.5112。
- 註：座標和由閉式 $1-\sigma_t=(1-\sigma_s)\frac{1-t}{1-s}$ 決定，與 $u$ 無關；第 1 列與第 4 列的 $X_{s,u}$ 相同，因為兩者是同一條軌跡在 $u=0.5$ 的點：第 1 列 $X_{0.1,0.5}(X_{0,0.1}(x_0))=X_{0,0.5}(x_0)$（半群），第 4 列 $X_{0,0.5}(x_0)$。

### 5.4 C3（`c3.py`，容許 $P_e$ ±1e-4、$t(\tau)$ ±0.002）

| $\lvert V\rvert$ | $P_e(0)$ | $P_e(0.5)$ | $P_e(0.8)$ | $t(0.05)$ | $t(0.25)$ | $t(0.5)$ | $t(0.75)$ | $t(0.95)$ | 等距格點在窗內（N=16，共 17） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.5000 | 0.2398 | 0.0023 | 0.081 | 0.311 | 0.488 | 0.619 | 0.735 | 10 |
| 10 | 0.9000 | 0.6591 | 0.0168 | 0.203 | 0.487 | 0.618 | 0.702 | 0.777 | 9 |
| 100 | 0.9900 | 0.9175 | 0.0854 | 0.450 | 0.642 | 0.715 | 0.764 | 0.812 | 5 |
| 1000 | 0.9990 | 0.9834 | 0.2360 | 0.603 | 0.716 | 0.764 | 0.798 | 0.833 | 4 |
| 30,522 | 0.99997 | 0.9987 | 0.5439 | 0.707 | 0.773 | 0.804 | 0.828 | 0.854 | 2 |
| 50,257 | 0.99998 | 0.9991 | 0.5873 | 0.717 | 0.779 | 0.809 | 0.831 | 0.856 | 2 |

- τ 格點在窗內：$N=16$ 時 $\tau_n=n/16$，$n=1..15$ 落在 $[0.05,0.95]$，共 15 個（與 $|V|$ 無關）。
- $|V|=30{,}522$、$N=8$ 的 τ 格點：0.000、0.745、0.773、0.791、0.804、0.816、0.828、0.842、1.000。
- 與顧問備忘的獨立計算（200 點 Gauss–Hermite）一致，差 ≤ 0.001。

### 5.5 C4（`c4.py`，須完全一致）

| μ | $\mu_{\min}/2$ | 可達的 $\nu(y_0)$ | 不同 ν 向量數（256 個映射） |
| --- | --- | --- | --- |
| A (0.4, 0.3, 0.2, 0.1) | 0.05 | 0, 0.1, 0.2, …, 1.0（11 個） | 202 |
| B (0.25 × 4) | 0.125 | 0, 0.25, 0.5, 0.75, 1 | 35 |
| C (1, 0, 0, 0) | 0.5 | 0, 1 | 4 |

---

## 6. 實作計畫

單一檔案 `tut/4/build/index.html`，沿用 `tutorials/prediction-targets-k-diff.html` 的外殼：CSS tokens（含 `--c-par`、`--c-perp` 與已調整的深色 `--ink-3` #9C978C）、appbar + 主題切換（`pm-theme`）、`#pm-bar` 導覽、左側章節目錄、`plate`／`callout`／`fact`／`seg`／`chip`／`tscroll`／`check` 元件、Alg 兩欄對照格線、`?test` selfTest 框架、KaTeX 0.16.9 設定與 `renderMath()`、時間線元件 `renderTimeline()`、`el()`／比例尺輔助。以下只列新內容。

### 6.1 連結與中繼資料

- `<title>`：「Flow Map Language Models」。Hero「在關聯圖中查看」→ `../index.html#paper-4`；arXiv `https://arxiv.org/abs/2602.16813`；程式碼 `https://github.com/david3684/flm`（論文寫明）。
- 筆記連結（同在 `tutorials/`）：`fmlm-lineage-overview.html`、`fmlm-parallel-decoding-motivation.html`、`fmlm-discrete-diffusion-factorization.html`、`fmlm-continuous-text-diffusion.html`、`fmlm-flow-matching-interpolant.html`、`fmlm-flow-map-distillation-lineage.html`、`fmlm-onehot-denoiser-two-time.html`；站內既有：`continuous-flow-lm-overview.html`、`flow-map-few-step-distillation.html`、`hybrid-continuous-discrete-lm.html`、`self-conditioning-flow-lm.html`、`cdlm-training-objective.html`、`cdlm-noise-schedule.html`、`x-prediction-manifold-view.html`。

### 6.2 新的程式結構（單一 IIFE）

- 玩具：`mulberry32`、`gauss`（自 238 沿用）；`sigmoid`、`kfun`、`flowU`、`toCoords`、`fromCoords`、`X`、`D`、`delta`、`gamma`、`euler(x,N)`、`flowmap(x,N)`、`classify`、`distOnehot`、`discreteClosed(N)`、`discreteSample(N, rng)`。全部照 4.0 與 `toy.py`。
- C3：`logErfc`、`logPhi`、`Pe(t,V)`、`lutFor(V)`（`Map` 快取）、`tOfTau(lut, y)`。
- C4：`pushforwards(mu)` → `{nu1Set, nVectors}`。
- 元件：`initC1a()`、`initC1b()`、`initC2()`、`initC3()`、`initC4()`，各自只碰自己的容器；狀態物件與 render 函式同 238 模式。
- 靜態 SVG：S2 的 $|V|=3$ simplex 三角形（頂點標 one-hot、內部一點標「機率向量」、外部一點標「$I_t$ 可在 simplex 外」）。
- 寬式與寬表（390px）：CSS `.katex-display{overflow-x:auto;overflow-y:hidden}`；`renderMath()` 後給每個 `.katex-display` 加 `tabindex="0"` 與 `aria-label`（例如「Eq. 23，可水平捲動」），讓鍵盤可捲動。Eq. 23 用 `\begin{aligned}` 拆三行（離對角項／對角項／$\bar\delta$ 定義）；Eq. 82 與 Eq. 24–25 也各拆兩行。Table 13 全表放 `tscroll`，第一欄 `position:sticky;left:0` 並給背景色。
- 演算法兩欄對照：S3b（Alg. 5–6）、S7（Alg. 7–8）；≥ 768px 兩欄逐行對齊，< 768px 每行程式碼下接說明（同 238 S8）。

### 6.3 selfTest（`?test`）

- `discreteClosed(1)` 無效率 0.5、`discreteClosed(4)` 0.125。
- PRNG：seed 7 第一個樣本 $x^1_1=0.1420$（±1e-4）。
- C1b：Euler $N=1$ 200 個平手；flow map $N=1$ 計數 98／102；Euler $N=2$ 平均距離 0.441、$N=4$ 0.032（±0.001）；flow map $N=8$ 與 $N=1$ 終點最大差 < 1e-9。
- `flowU` 失敗旗標：`flowU(0.4,0,0.5).failed === false`；以步數上限 1 呼叫內部版本時回傳 `failed:true`（`flowU` 接受選用的 `maxSteps` 參數，只供 selfTest）。
- C2 預設：$\gamma=0.1$；$\delta_{s,t}$ new 0.9386（±1e-4）；教師差 < 1e-8；$|q_{82}-q|$ < 1e-6；$X_{s,u}$ 座標和 0.85 / 0.45。
- C3：$|V|=30{,}522$：$P_e(0.5)=0.9987$、$P_e(0.8)=0.5439$（±1e-4）、$t(0.5)=0.804$（±0.002）；$|V|=2$：$P_e(0)=0.5$。
- C4：A 可達 11 個、202 個向量；C 4 個向量。
- `scrollWidth <= innerWidth`。結果一行 `selfTest: N passed, M failed`。

### 6.4 時間線（S11）

- `notes/timeline.json` 原樣貼入 `<script type="application/json" id="timeline-data">`（14 筆，含本文 2602.16813）。
- `TL_GROUP`（以 arXiv ID 為 key；未列者歸「其他」）：

  | 類別 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | 平行解碼 | 1711.02281、2410.21035 | `fmlm-parallel-decoding-motivation.html` |
  | 離散擴散 | 2107.03006、2310.16834、2406.07524、2506.10892 | `fmlm-discrete-diffusion-factorization.html` |
  | 連續文字擴散 | 2205.14217、2211.15089 | `fmlm-continuous-text-diffusion.html` |
  | flow matching | 2210.02747 | `fmlm-flow-matching-interpolant.html` |
  | flow map | 2303.01469、2406.07507、2505.18825、2505.13447 | `fmlm-flow-map-distillation-lineage.html` |
  | 本文 | 2602.16813 | `fmlm-onehot-denoiser-two-time.html` |

- S11 文字：平行解碼的動機 → 離散擴散與因子化誤差 → 連續 embedding 擴散（Diffusion-LM、CDCD；Analog Bits 是類似 one-hot 的連續表示先例，筆記 `fmlm-continuous-text-diffusion`）→ flow matching（[40] Li & He 2511.13720 是改學 denoiser 的理由）→ flow map 少步蒸餾 → 本文把 one-hot、CE denoiser、two-time denoiser 接在一起；$P_e$ 指標沿用自 Duo、CANDI。Analog Bits（2208.04202）、Back to Basics（2511.13720）不在 timeline.json，只在文字與延伸閱讀提及，不加入時間線。
- S12 延伸閱讀依總覽順序列出 7 篇 fmlm 筆記，再列 6.1 的站內既有筆記。

### 6.5 文案與標籤

- 沿用 238 的 5.6（引用、「本頁推導」pill）、5.8（台灣用語、字面語氣、陳述句標題）、5.9（無障礙）規則。
- 用詞：「因子化誤差」（factorization error）、「two-time denoiser」保留英文、「flow map」保留英文、「半群條件」、「解碼錯誤率」、「時間重參數化」、「蒸餾」、「常見錯誤」。專有名詞第一次出現附英文。
- 動態與深色主題：`@media (prefers-reduced-motion: reduce)` 下關閉所有 transition／animation（chip、點、突顯、色帶）。斜線紋理（C1a 無效橫條、C1b 無效象限、C2 權重填色）用 SVG `<pattern>`，線條 `stroke="currentColor"`、`stroke-opacity` 0.25–0.35，由外層 `color` 決定，兩種主題自動跟隨；`--accent` 填色與紋理在淺色、深色主題都檢查對比（文字 ≥ 4.5:1、圖形 ≥ 3:1），截圖時兩種主題各看一次 C1a 與 C1b。
- 文案語氣：不用比喻或擬人（例如不寫「δ 預告方向」「一次跨越」），一律寫成可驗證的數值或操作敘述。
- 區分三種陳述：論文結果、作者的解釋（「作者認為」「作者假設」）、本頁推導。玩具數字一律標「本頁玩具，非論文數字」，不與論文表格並列成「重現」。

### 6.6 建置後自我檢查

- `?test` 全數通過；`npx impeccable detect` 0 項；截圖 390 與 1440 各一張，只看新元件（C1a–C4）與 Hero（截圖迴圈最多 1 輪，依 SKILL.md）。
- 執行 `python3 .claude/skills/paper-tutorial/scripts/extract_page_text.py tut/4`，逐條確認 12 個 LO 的參考答案出現在 `accept/page.txt` 的正文中。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 玩具不是論文實驗 | Fig. 3 是示意圖；C1a 的 $1/(2N)$、C1b、C2 的數字都是本頁玩具推導。 | 元件圖版底部標「本頁推導」與「本頁玩具，非論文數字」；只把 $N=1$ 的 25% 對到 Fig. 3。 |
| R2 | $P_e$ 閉式與「near $t=1$」 | 論文未寫公式（App. E 只描述以 Gauss–Hermite 近似 (25) 的 CDF）；Eq. 24 用 $x_t$，本頁閉式是把它讀成 interpolant $I_t$ 的結果；依此算出窗在 $t\approx0.71$–$0.85$，論文文字寫「concentrates acutely near $t=1$」。無法從 paper.txt 讀出 Fig. 9 曲線。 | 窗與格點計數一律標「本頁推導（interpolant 耦合讀法）」，不稱重現 Fig. 9；頁面寫「很窄的時間窗」並給本頁計算值；建置時以 PDF Fig. 9 目視對照曲線形狀，若明顯不符則在 R2 回報。 |
| R3 | 「一步 vs 8 步」四種說法 | 摘要「exceeds the 8-step quality」、貢獻「nearing their 8-step quality」、Fig. 1「matches the 8-step」、Sec. 5.2「LM1B 8–16 步、OWT 4–8 步」；8 步以上基線數字只在 Fig. 12，無表格。 | 正文與 LO11 採 Sec. 5.2；S9 callout 列出四種說法。Fig. 1 的 8.3× 加速未說明對象與硬體，頁面引用時註明。 |
| R4 | 其他內部不一致 | Fig. 16 圖例 entropy 門檻 4.0 vs 圖說 3.9；Sudoku 一步 9.38%（Table 22）vs App. F「5%」；Self-BLEU 主文說「slightly below real data」、App. F 說「slightly worse」，FMLM 的值較高（0.073 vs 0.047，多樣性略差），且 MDLM 基線低於資料；Table 26 標 % 但數值為比例；記憶體成本 30%（Sec. 6）vs 15–20%（App. F），Table 32 算得 1.16–1.24；Table 17 圖說稱「all step counts」但 1024 步 Grammaticality MDLM 0.255 > FLM 0.240。 | 以表格數字為準；S9、S12 用一句註明相關出入，不展開。 |
| R5 | FMLM 取樣器 | Alg. 8 是確定性合成；App. E 稱 FMLM 用 Kim et al. [91] 的 γ-sampling（γ 未給）；App. G 稱確定性取樣。一步時無差別。 | 頁面以 Alg. 8 說明，常見錯誤第 10 條照錄出入，並寫明論文未說明 γ-sampling 與 Alg. 8 的關係。 |
| R6 | 式子下標不一致 | Eq. 36（PMD）寫 $\hat X_{t,u}(\hat X_{s,t})$，Eq. 31 的 $(s,t,u)$ 順序也與 Eq. 16 不同；Sec. 3.2 說後驗「defined in (2)」但 Eq. 2 定義的是 $p^l_{t\mid s}$；Eq. 68 上方「12 dt」為「$\int1^2dt$」排版錯誤。 | 頁面統一用 Eq. 16、17、22 的寫法，不引用 Eq. 31、36。 |
| R7 | Prop. 3.4 的「unique critical point」 | App. C.5 的證明只是兩三行 sketch，且假設教師精確。 | 頁面寫「作者證明在教師精確時唯一最小值是真 δ」，不說成收斂保證。 |
| R8 | CE 不一定勝過 MSE | App. D 兩階段 MSE 蒸餾 LM1B 一步 102.49 / 104.37 低於 CE 119.34（entropy 4.13／4.12 對 4.16，相當）；OWT 一階段 129.32 但 entropy 4.53，對 CE 168.30 / 5.17，是以 entropy 換 PPL；Table 18 中 δ-denoiser + MSE 156.83 優於 + softmax + MSE 181.72（但 entropy 3.94）。 | S7 寫「在同一套單階段設定下 CE 最好」，並以一句提 App. D，註明只有 LM1B 的比較 entropy 相當。 |
| R9 | Table 18 星號列 | 時間重參數化與表示兩組是 300k 步，不能和 1M 步的 96.91 比；embedding 消融都用 learned entropic time，所以「One-hot 130.42」與「Learned entropic time 130.42」是同一模型；「Riemannian 268.21」即 Table 10 的 RDLM。 | S4、S9 引用時標明 300k 步，只在組內比較。 |
| R10 | Table 28 設定不明 | 「FMLM … 200k steps」高斯 105.48 / 4.31，比主結果好；步數與是否 FLM 不清楚。 | 只用於「Dir(1) 遠差於高斯」的組內比較，不與主表並列。 |
| R11 | C1b 玩具過於簡單 | 對稱兩模態下 Euler 在 $N\ge8$ 已精確、從不產生無效組合；無法展示真實語言模型的少步退化（Fig. 11）。 | 控制列旁一行說明（4 C1b），先寫「Euler 一步 = 兩模態平均」並把 Fig. 11 連到第 10 節；「N ≥ 8 已近乎精確」不寫進圖版；正文不以玩具推論 Euler 在語言上的表現。 |
| R12 | ODE 在 $t\to1$ 剛性 | $k(t)$ 以 $(1-t)^{-2}$ 增長，$u\approx0$ 附近剛性。 | `flowU` 改用 $r$ 變數、自適應步長、飽和後精確尾段、正負號保護、步數上限（4.0）；selfTest 檢查。 |
| R13 | 符號 $d$ 的兩種意義 | 主文 $d$ 是網路內部特徵維度，App. B 起 $d=L\times\lvert V\rvert$。 | 頁面只用主文意義，符號表註明。 |
| R14 | 相對連結 | 建置位置與上線位置不同，預覽時站內連結失效。 | 同 238 R7，接受。 |
| R15 | 筆記可能更新 | 建置前 notes 若增減，`TL_GROUP` 與延伸閱讀需同步。 | 建置時重新讀取 `notes/`；新 arXiv ID 歸「其他」並在建置回報列出。 |
| R16 | 章節數增為 14 | S3 拆成 S3a／S3b 後顯示編號 04 起整體後移一號；內部代號 S4–S12 不改。 | 目錄與交叉引用（例如「見第 10 節」）以顯示編號為準，建置時全文檢查。 |
| R17 | C2 進階內容不在預設畫面 | 半群（Eq. 22）與 Eq. 82 的觀察要展開「進階」才看得到，LO8 的答案仍必須在正文。 | LO8 答案寫在 S6 正文；進階只是互動驗證。 |
| R18 | C3 冷啟動時間未在手機測量 | 分段計算避免卡住主執行緒，但完成時間未知。 | 建置截圖時記錄 390px 模擬下完成時間；若 > 2 秒，背景曲線改為 101 點。 |

---

## 8. 修訂紀錄

- **Round 1（合併清單：`reviews/round1.md`，29 項）**：
  - 忠實度：$P_e$ 閉式改標「interpolant 耦合讀法」的本頁推導，不稱重現 Fig. 9；token 相關改歸於「後驗條件在整個 $I_t$ 上 + 確定性 flow 耦合」，$x_0$ 只提供隨機性（步驟 6、LO5、C1b）；Self-BLEU 以接近資料為目標並列全表；LO11、1.1 補 OWT entropy 5.33 對 5.44；Table 18 表示子表標 300k 步、learned entropic time；Prop. 3.5 補 $\lvert S\rvert\ge2$ 與最小正質量；Eulerian 解釋分開歸屬主文與 App. F；App. D 補 OWT 列；App. E 改為「描述」；γ-sampling 未調和；Alg. 7 對角項用 $I_s$；7.6× 標為近似並寫明兩個假設；Eq. 13 只是上界發散；C1a 只把 N = 1 對到 Fig. 3。
  - 教學與建置：SVG／動態 DOM 改 Unicode 標籤、`renderMath()` 只跑靜態節點（P0）；S3 拆成 S3a／S3b，Eq. 13 證明入 `<details>`；C2 分層顯示、讀數縮為 4 類；S1、S2、S8 補定義與順序；S4 先 C3 後閉式；寬式可捲動、Eq. 23 拆行、Table 13 sticky 首欄；C1a chip 標註；C3 分段計算；`flowU` 失敗旗標與 selfTest；C1b 免責移到控制列旁；reduced motion 與深色紋理；Hero 卡 3 去術語；S9 次要結果收合；去除比喻用語。
  - 研究筆記：$P_e$ 指標沿用自 Duo、CANDI（已核對 paper.txt）；[40] 標出 Li & He 2511.13720；S11 文字加入 Analog Bits 與 Back to Basics（不入時間線）。
  - 不需修改：F14（$C=C_D^2$ 正確，只把措辭改為上界發散）、F15（Table 28 不加步數）。

- **Readiness NO-GO 修正**：定義 S3a 的「rank bottleneck」（正文寫出秩 $\le d$ 對 full-rank 噪聲的白話說明，標明是本頁對 [40] 的說明）；S4 的 Table 18「Rank」列標為 CANDI [20] 的 rank-based 時間重參數化（Sec. 5，p. 17），論文未給機制，頁面不解釋其公式；大綱 S3a 列註明定義位置。

- **Readiness 非阻擋注意事項（建置前套用）**：S3a 交叉引用改為步驟 4；5.1 補容許誤差；LO12 的 FLM OWT 例子註明差 0.11、程度較小；Hero 卡 3 限定 LM1B；S9 的 Table 18 摘錄指定列；C1b 突顯樣本索引註明 0 起算；5.3 註改寫為同一條軌跡（半群）；外殼模板 `tutorials/prediction-targets-k-diff.html` 已在 repo 內，建置直接取用。

- **Elephant 驗收補強（無上下文讀者，答案框移除）**：三項原已在正文但讀者漏答，改為更顯眼的獨立敘述。LO1：S1 玩具段補一句「Eq. 2 的條件獨立只在 $t\to s$ 極限成立，一步取樣 $s=0$、$t=1$ 相距最遠，假設不成立」；LO3：S3a 的 Table 18 數字（Velocity + MSE 3801.36、Denoiser + softmax + CE 96.91）拆成獨立段落；LO5：S3b 段落改以「連續法裡 token 相關從哪裡進來？」開頭，回答後驗條件在整個 $I_t$ 上 + 確定性 flow 耦合、$x_0$ 只提供隨機性（〔本頁推導〕）。
