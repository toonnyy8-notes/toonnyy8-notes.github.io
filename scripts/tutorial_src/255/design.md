# 設計文件：〈Reasoning with Sampling: Your Base Model is Smarter Than You Think〉互動教學

- 論文：Aayush Karan、Yilun Du（Harvard），arXiv 2510.14901v1（2025-10-16）。資料庫編號 No. 255。
- 原文：`scripts/tutorial_src/255/src/paper.txt`（PDF 同目錄）。顧問筆記：`scripts/tutorial_src/255/advisor.md`。「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」「Alg. 1」皆指論文原文編號。
- 產出：`scripts/tutorial_src/255/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/reasoning-with-sampling.html`（slug `reasoning-with-sampling`）；站內相對連結以上線位置為準。
- 驗算腳本（非頁面程式）：`verify/check_closed.py`（C1、C2、C4、Table 1 差值）、`verify/check_toy.py`（C3 的玩具模型、Alg. 1 精確輸出分佈、單步軌跡、Z_t 恆等式）。第 5 節驗收表全部由這兩支腳本輸出，兩支都已執行完畢。
- 研究筆記：`notes/`（總覽 `rswp-lineage-hub` + 7 篇主題 + `timeline.json`，14 筆）。
- 標籤慣例（頁面沿用）：
  - 〔本頁推導〕：本頁或顧問自行推導、論文沒寫的結論（含 13.5–14.5 倍成本、Z_t 化簡）。
  - 〔示範值〕：本頁為了示範自訂的數值（玩具模型、示意圖參數）。
  - 「作者認為」：作者的解釋或歸因，與量測結果分開寫。
  - 論文內部不一致處以「論文勘誤」callout 呈現（見 1.6）。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

RL 後訓練（例如 GRPO）能提升 LLM 的單次作答推理準確率，但近期文獻觀察到 RL 後的輸出集中在 base model 的高 likelihood 區域，且大 k 的 pass@k 低於 base model（Sec. 1）。作者據此問：能否不訓練，只靠取樣，就從 base model 得到相近的推理能力？本文把取樣目標定為冪分佈 $p^\alpha$（Eq. 2），證明它不同於常用的低溫取樣（Prop. 1，Eq. 4–8），並說明 $p^\alpha$ 偏好「未來路徑少但 likelihood 高」的 token（Obs. 1、Example 1、App. A.1 Prop. 3）。因為 $p^\alpha$ 的正規化常數無法計算，作者用 Metropolis-Hastings（MH）搭配「隨機位置重取」提議分佈（Eq. 9、Fig. 3），再依自回歸結構把序列切成長 $B$ 的區塊逐段做 MH（Alg. 1，Eq. 10–11）。在 Qwen2.5-Math-7B、Qwen2.5-7B、Phi-3.5-mini-instruct 上，單次準確率接近 GRPO：MATH500 與 GPQA 上兩個 Qwen 模型仍是 GRPO 較高，HumanEval 與 AlpacaEval 2.0 上 power sampling 較高（Table 1）；pass@k 在 k>1 時高於 GRPO（Fig. 5）。論文報告的 token 成本為一般推論的 8.84 倍（Sec. 5.3）。

### 1.2 研究動機（Sec. 1）

1. RLVR 以可自動驗證的獎勵後訓練 LLM，在數學、程式、科學上有效（Guo 2025 等）。
2. 「分佈銳化」問題：RL 後的分佈是否只是 base 分佈變尖，而不是學到 base 不會產生的推理軌跡（He 2025、Shao 2025、Yue 2025）。
3. 證據：大 k 的 pass@k 是 base model 較高（He 2025、Song 2025）；RL 後軌跡集中在 base 的高 likelihood/confidence（Yue 2025）。
4. 作者的推論：若 RL 只是銳化，應能直接寫出一個「銳化後」的目標分佈並從中取樣。這是假說，不是本文證明的事。

### 1.3 貢獻（Sec. 1 i–iii）

1. 把冪分佈 $p^\alpha$ 定為推理任務的取樣目標；它可由 base model 直接寫出，不需訓練。
2. 用 MCMC（MH）近似取樣 $p^\alpha$，依 base model likelihood 反覆重取 token 子序列（Alg. 1）。
3. 在 3 個模型、4 個基準上與 GRPO 比較單次準確率、pass@k、likelihood 與 confidence 分佈、回應長度。

### 1.4 符號表（頁面 S1a 末尾以可收合 `<dl>` 提供）

| 符號 | 意義 |
| --- | --- |
| $\mathcal X$ | 有限 token 詞彙表 |
| $x_{0:T}=(x_0,\dots,x_T)$ | token 序列；$x_{<t},x_{>t},x_{\le t},x_{\ge t}$ 為前後綴（Sec. 3）。$T$ 是最後一個位置的索引；成本處寫「生成長度 T（平均 679）」。Example 1 的 $x_0$ 是第一個生成的 token；C3 的 $x_0$ 是不重取的提示 token |
| base model | 本文指「作為取樣對象、不再訓練的既有模型」。Qwen2.5-Math-7B、Qwen2.5-7B 是預訓練模型；Phi-3.5-mini-instruct 是 instruct 版，已經過後訓練（Abdin 2024），論文未說明細節 |
| $p$ | base model 定義的序列分佈；$p(x_t\mid x_{<t})$ 為 next-token 分佈 |
| $\alpha$ | 冪次（power），$\alpha>1$；論文主實驗 $\alpha=4.0$ |
| $\tau$ | 溫度；低溫取樣取 $\tau=1/\alpha$ |
| $p_{\text{temp}}$、$p_{\text{pow}}$ | 低溫取樣、冪分佈的 next-token 條件分佈（Eq. 3、4） |
| $F(x_{\le t})$ | 〔本頁推導〕未來修正項 $\sum_{x_{>t}}p(x_{>t}\mid x_{\le t})^\alpha$ |
| $\varepsilon,\varepsilon',N$ | Prop. 3：正向 pivotal token 的邊際權重、負向 pivotal token 的邊際權重、負向 token 的未來路徑數 |
| $q(x\mid x^i)$、$A$ | MH 提議分佈、接受率（Eq. 9）；上標 $x^i$ 是鏈的第 $i$ 個狀態，與 token 下標 $x_i$ 不同 |
| $p_{\text{prop}}$ | 提議用 LLM（推理任務：base model 配 $\tau=1/\alpha$） |
| $Z_t(x_{<t})$ | 〔本頁推導〕$\sum_v p(v\mid x_{<t})^\alpha$，低溫分佈的正規化常數 |
| $B$、$N_{\text{MCMC}}$ | 區塊長度、每區塊 MH 步數 |
| $\pi_k$ | 中間目標 $\pi_k(x_{0:kB})\propto p(x_{0:kB})^\alpha$（Eq. 11） |
| $m$ | Alg. 1 中均勻抽出的重取起點 |
| TV | 總變差距離 $\tfrac12\sum_x\lvert P(x)-Q(x)\rvert\in[0,1]$；S6 內文定義，C3 使用 |
| 偏好 | 頁面用詞：「某取樣法偏好 token a」＝ a 在該取樣法下得到較大的相對權重（S3a 首次定義） |

### 1.5 從作者的角度逐步拆解

#### 步驟 1：LLM 給出序列的機率（Sec. 3，Eq. 1）

$$p(x_{0:T})=\prod_{t=0}^{T}p(x_t\mid x_{<t})\tag{Eq. 1}$$

逐 token 取樣即從聯合分佈取樣。任何給定序列的 $p$ 值都可由一次前向計算得到。這是後面「未正規化權重可算」的基礎。

#### 步驟 2：目標分佈 $p^\alpha$（Sec. 4.1，Eq. 2，Fig. 2）

$$p(x)>p(x')\ \Rightarrow\ \frac{p(x)^\alpha}{p(x')^\alpha}>\frac{p(x)}{p(x')}\tag{Eq. 2}$$

- 指數放大使高 likelihood 序列的相對權重上升、低的下降（「銳化」）。Fig. 2 用高斯混合畫出 $p$ 與 $p^{4.0}$（混合參數論文未給）。
- **勘誤 E1**：論文寫 $\alpha\in[1,\infty]$。$\alpha=1$ 時兩邊相等，嚴格不等號不成立，須 $\alpha>1$。
- 相關文獻（Sec. 2）：統計物理稱之為退火（annealed）或回火（tempered）分佈（Neal 1998）；擴散模型以 $p^\alpha$（$\alpha>1$）提升樣本品質。筆記 `rswp-annealed-tempered-sampling`。

#### 步驟 3：低溫取樣不等於 $p^\alpha$（Eq. 3–8，Prop. 1）

低溫取樣逐步把 next-token 機率取 $\alpha$ 次方再正規化：

$$p_{\text{temp}}(x_t\mid x_{<t})=\frac{p(x_t\mid x_{<t})^\alpha}{\sum_{x'_t}p(x'_t\mid x_{<t})^\alpha},\quad \tau=1/\alpha\tag{Eq. 3}$$

冪分佈的條件分佈（Eq. 4）與低溫（用 Eq. 5 改寫成 Eq. 6）忽略正規化後的相對權重：

$$p_{\text{pow}}(x_t\mid x_{<t})\propto\sum_{x_{>t}}p(x_0,\dots,x_T)^\alpha\ \ \text{（次方的和，Eq. 7）}\qquad p_{\text{temp}}(x_t\mid x_{<t})\propto\Big(\sum_{x_{>t}}p(x_0,\dots,x_T)\Big)^\alpha\ \ \text{（和的次方，Eq. 8）}$$

- 〔本頁推導〕把 $p(x_{0:T})=p(x_{<t})\,p(x_t\mid x_{<t})\,p(x_{>t}\mid x_{\le t})$ 代入 Eq. 7，前綴項為常數：
  $$p_{\text{pow}}(x_t\mid x_{<t})\propto p(x_t\mid x_{<t})^\alpha\cdot F(x_{\le t}),\qquad F(x_{\le t})=\sum_{x_{>t}}p(x_{>t}\mid x_{\le t})^\alpha$$
  $\alpha>1$ 時 $0<F\le1$，$F=1$ 若且唯若選了 $x_t$ 之後未來只有一條路。低溫取樣等於把所有 $F$ 當成 1。
- 〔本頁推導〕兩者相等的情形：$\alpha=1$；所有候選 token 的 $F$ 相同；最後一個位置 $t=T$（沒有未來，$F\equiv1$）。所以對一般 $p$ 而言 Prop. 1 成立，但證明中「每個時間步都不同」在 $t=T$ 不成立；$\alpha=1$ 或確定性 $p$ 時命題本身也不成立（勘誤 E3）。
- Eq. 5 論文稱為 Bayes rule，實際是條件機率定義加邊際化。

#### 步驟 4：Obs. 1 與 Example 1（Sec. 4.1）

Obs. 1：冪分佈偏好「未來路徑少但 likelihood 高」的 token，低溫偏好「未來路徑多但各自 likelihood 低」的 token。

Example 1（$\mathcal X=\{a,b\}$、兩個 token、$\alpha=2.0$）：$p(aa)=0.00,\ p(ab)=0.40,\ p(ba)=0.25,\ p(bb)=0.25$。

- $p_{\text{pow}}(x_0=a)\propto0.00^2+0.40^2=0.160$，$p_{\text{pow}}(x_0=b)\propto0.25^2+0.25^2=0.125$ → 偏好 a。
- $p_{\text{temp}}(x_0=a)\propto(0.40)^2=0.160$，$p_{\text{temp}}(x_0=b)\propto(0.50)^2=0.250$ → 偏好 b。
- **勘誤 E2**：四個機率總和為 0.90，不是 1。排序結論不受影響，因為兩邊都只比相對權重。
- 〔本頁推導〕正規化：$p_{\text{pow}}(x_0=a)=0.160/0.285=0.561$，$p_{\text{temp}}(x_0=a)=0.160/0.410=0.390$。整條序列：冪分佈 ab 0.561、ba=bb 0.219；低溫 ab 0.390、ba=bb 0.305。用 $F$ 驗算：$F(a)=1$、$F(b)=0.5^2+0.5^2=0.5$，$0.4^2\cdot1=0.160$、$0.5^2\cdot0.5=0.125$。
- 〔本頁推導〕翻轉點：$0.4^\alpha>2\cdot0.25^\alpha\iff\alpha>\ln2/\ln1.6\approx1.475$。總和為 1 的版本 $p(ab)=0.45,\ p(ba)=p(bb)=0.275$ 結論相同（$\alpha=2$：0.2025 對 0.15125；翻轉點 1.407）。
- 作者認為：這種權重差異對推理有用，因為「pivotal token / critical window」（少數 token 決定答案對錯，Li 2025、Abdin 2024）中，選錯 token 會使所有後續序列的 likelihood 較低；作者稱 $p^\alpha$ 內含 "an implicit bias towards planning for future high likelihood tokens"（Sec. 4.1 末段，頁面照錄英文並註明是作者用語）。

#### 步驟 5：Prop. 3（App. A.1，Def. 2–4，Eq. 14–15）

- Def. 2：固定前綴，$x_t$ 的邊際權重 $\varepsilon=\sum_{x_{>t}}p(x_0,\dots,x_T)$。
- Def. 3（正向 pivotal token）：全部權重 $\varepsilon$ 在一條未來。Def. 4（負向）：權重 $\varepsilon'$ 均分到 $N$ 條未來，每條 $\varepsilon'/N$。
- 冪分佈權重：$\varepsilon^\alpha$ 對 $\varepsilon'^\alpha/N^{\alpha-1}$；低溫權重：$\varepsilon^\alpha$ 對 $\varepsilon'^\alpha$。

$$\frac{\varepsilon'}{N^{1-1/\alpha}}<\varepsilon<\varepsilon'\tag{Eq. 14}$$

在此區間，冪分佈偏好 $x_t$、低溫偏好 $x'_t$；並由 $\varepsilon'/N^{1-1/\alpha}>\varepsilon'/N$（Eq. 15）得 $x_t$ 的未來 likelihood 高於 $x'_t$ 任何一條未來。
- 〔本頁推導〕Eq. 15 的嚴格不等號需 $N>1$ 且 $\alpha>1$；$\alpha=1$ 或 $N=1$ 時 Eq. 14 區間為空。Example 1 驗算：$\varepsilon=0.4,\varepsilon'=0.5,N=2,\alpha=2$，下界 $0.5/\sqrt2=0.354<0.4<0.5$。
- 適用範圍：只在「單一未來 vs N 條均勻未來」的極端模型下成立，比較的是 likelihood 排序，不是答案正確性。

#### 步驟 6：為什麼用 MCMC；MH（Sec. 4.2，Eq. 9，Def. 1，Fig. 3）

- 直接取樣 $p^\alpha$ 需要對所有序列求和以正規化，不可行。
- MH 建立狀態鏈 $x^0,x^1,\dots$；以提議 $q(x\mid x^i)$ 產生候選 $x$，以機率

$$A(x,x^i)=\min\Big\{1,\ \frac{p^\alpha(x)\,q(x^i\mid x)}{p^\alpha(x^i)\,q(x\mid x^i)}\Big\}\tag{Eq. 9}$$

接受，否則 $x^{i+1}=x^i$。正規化常數在比值中相消。$n$ 夠大時收斂到 $p^\alpha$，條件為 $q$ 不可約、非週期（Def. 1，引用 Neal 1993，論文未證）。
- 〔本頁推導〕平穩性一行：$\pi(x)q(x'\mid x)A(x',x)=\min\{\pi(x)q(x'\mid x),\pi(x')q(x\mid x')\}$，對 $x,x'$ 對稱（detailed balance）。
- 隨機重取提議（Fig. 3）：以機率 $1/T$ 選 $t\in[1,T]$，從 $t$ 起用 $p_{\text{prop}}$ 重取到結尾。論文寫：可從開頭重取，所以提議不可約且非週期；$p_{\text{prop}}$ 可為任意 LLM 與取樣法。
- 〔本頁推導〕論文略過的條件：(1) 不可約需要 $p_{\text{prop}}$ 對 $p^\alpha$ 支撐內的每個 token 給正機率，且 $q(x'\mid x)>0\iff q(x\mid x')>0$，否則 Eq. 9 的比值沒有定義；τ = 1/α 的全詞彙取樣滿足，greedy、top-k、top-p 等截斷取樣不一定滿足。(2) 非週期來自拒絕或提議到自身時的自環，不是來自「可從開頭重取」。
- 〔本頁推導〕重取起點 $m$、區塊結尾 $L$，前綴與 $1/L$ 相消：
  $$\frac{p^\alpha(x')q(x\mid x')}{p^\alpha(x)q(x'\mid x)}=\prod_{t=m}^{L}\frac{p(x'_t\mid x'_{<t})^\alpha/p_{\text{prop}}(x'_t\mid x'_{<t})}{p(x_t\mid x_{<t})^\alpha/p_{\text{prop}}(x_t\mid x_{<t})}$$
  **Z_t 化簡〔本頁推導〕**：當 $p_{\text{prop}}$ 為同一 base model 配 $\tau=1/\alpha$，$p_{\text{prop}}(v\mid x_{<t})=p(v\mid x_{<t})^\alpha/Z_t(x_{<t})$，比值化為 $\prod_{t=m}^{L}Z_t(x'_{<t})/Z_t(x_{<t})$。每個 token 的 $p^\alpha$ 全部相消，只剩每一步低溫分佈的正規化常數；$Z_t$ 越大表示該步 next-token 分佈越尖。兩個推論：(a) 只重取區塊最後一個 token（$m=L$）時前綴相同，比值恆為 1，必定接受；(b) 因共同前綴上的 $Z_t$ 在比值中相消，同一對新舊序列不論 $m$ 落在共同前綴內的哪裡，比值都相同。這和步驟 3 的 $F$ 是同一件事：低溫提議 × $Z$ 修正 = 冪分佈。化簡要求提議是與目標同一模型、未截斷（無 top-k、top-p）的 τ = 1/α 取樣；論文未說明是否截斷。AlpacaEval 用 $\tau=0.5$（$\alpha$ 未說明），化簡不成立。`check_toy.py` 在 $V=2,T=4,\alpha=4$ 全部 256 對序列上驗證，最大誤差 $1.8\times10^{-15}$。
- 〔本頁推導〕同一個 $x'$ 可由多個 $m$ 產生；因 $m$ 的選取與狀態無關，每個固定 $m$ 的核各自滿足 detailed balance，混合後仍滿足。論文只寫「by symmetry」。

#### 步驟 7：Alg. 1 分區塊的自回歸 MCMC（Sec. 4.3，Eq. 10–11）

- 動機：直接在長度 $T$ 上做 MH 需要大量完整序列的推論呼叫；MCMC 可能有指數級 mixing time，壞的初始化與高維狀態空間會加重（Gheissari 2017 等）。
- 中間分佈序列 $\varnothing\to p(x_{0:B})^\alpha\to p(x_{0:2B})^\alpha\to\cdots\to p(x_{0:T})^\alpha$（Eq. 10），$\pi_k(x_{0:kB})\propto p(x_{0:kB})^\alpha$（Eq. 11）。
- Alg. 1：對 $k=0,\dots,\lceil T/B\rceil-1$：(3) 用 $p_{\text{prop}}$ 延伸 $B$ 個 token 當初始化；(4–8) 跑 $N_{\text{MCMC}}$ 步：均勻抽 $m\in\{1,\dots,(k+1)B\}$，從 $m$ 重取到 $(k+1)B$，算接受率，抽 $u\sim U(0,1)$，$u\le A$ 則接受；(10) 把結果當下一階段的前綴。
- **勘誤 E4**：第 3 行說要取樣 $\pi_{k+1}$，第 7 行接受率卻寫 $\pi_k$；應為 $\pi_{k+1}$。本頁 C3 依 $\pi_{k+1}$ 實作。
- 第 10 行「fix the new prefix」不代表凍結：第 5 行的 $m$ 從 1 起抽，較早的 token 在後面階段仍會被重取。
- 〔本頁推導〕$\pi_k$ 是「前綴邊際的 $\alpha$ 次方」，不是 $p^\alpha(x_{0:T})$ 在前綴上的邊際（Prop. 1 的同一差別，發生在區塊層級）；最後一階段的目標是完整的 $p^\alpha$，漸近不影響正確性，有限步數下是偏差來源。
- Output 寫 $\sim p^\alpha$，那是 $N_{\text{MCMC}}\to\infty$ 的極限；論文沒有 mixing 診斷，Sec. 5.3 自己說步數越少離目標越遠。
- x_0：§4.2 抽 $t\in[1,T]$，所以 $x_0$ 永不重取；論文未說明 $x_0$ 是否代表 prompt（本頁 C3 把 $x_0$ 當固定提示）。
- 作者稱 Alg. 1 為 single-shot：最後只交一個序列、接受與否只看 base likelihood，不用驗證器；生成過程中其實產生許多候選。

#### 步驟 8：成本（Sec. 4.3 Eq. 12；Sec. 5.3）

$$E_{\text{tokens}}=N_{\text{MCMC}}\sum_{k=1}^{\lceil T/B\rceil}\frac{kB}{2}\approx\frac{N_{\text{MCMC}}T^2}{4B}\tag{Eq. 12}$$

- 相對一般推論的倍數 $N_{\text{MCMC}}T/(4B)$；代入 $N_{\text{MCMC}}=10,\ T=679,\ B=192$ 得 **8.84 倍**（Sec. 5.3）。
- 〔本頁推導〕近似偏低：$K=\lceil679/192\rceil=4$，精確和用 $K(K+1)/2=10$；近似相當於 $(T/B)^2/2=(679/192)^2/2=6.26$。照 Eq. 12 的和精確算：$10\times96\times10=9600$ token，**14.14 倍**；把最後區塊截到 679：$10\times(192+384+576+679)/2=9155$，**13.48 倍**；再加初始化延伸的約 $T$ 個 token，14.48 倍。頁面寫「論文報告 8.84 倍；照它自己的求和式精確計算約 13.5–14.5 倍〔本頁推導〕」。成本 ∝ $T^2$，用平均長度代入會低估平均成本（$E[T^2]\ge E[T]^2$）。$T=T_{\max}=3072$ 時近似為 40 倍。
- 作者的比較：假設每題 8 個 rollout、資料集一樣大，成本約等於 GRPO 一個 epoch；通常 GRPO 一個 epoch 更貴（16 個 rollout、訓練集大於 MATH500）。頁面補一句：GRPO 是一次性成本，power sampling 每次查詢都要付；論文未報告延遲。

#### 步驟 9：實驗設定（Sec. 5.1）

- 基準：MATH500（OpenAI 從 MATH 測試集抽的 500 題）、HumanEval（164 題，平均 7.7 個單元測試）、GPQA Diamond（198 題）、AlpacaEval 2.0（805 個 prompt，GPT-4-turbo 評審，長度校正勝率）。全部以單一最終回應評分。
- 模型：Qwen2.5-Math-7B、Qwen2.5-7B、Phi-3.5-mini-instruct；GRPO 用 Shao 2025 的實作在 MATH 訓練集上後訓練（Qwen 用其預設超參數，Phi 用選自 Abdin 2024 的超參數）。
- Power sampling：$T_{\max}=3072$（可提早遇 EOS 結束），$B=3072/16=192$，$\alpha=4.0$，$p_{\text{prop}}$ = base model 配溫度 $1/\alpha$；AlpacaEval 改用提議溫度 $\tau=0.5$。$N_{\text{MCMC}}=10$ 只在 Sec. 5.3 成本處出現。

#### 步驟 10：結果（Sec. 5.2–5.3，Table 1，Fig. 4–6，App. A.2）

Table 1（照錄，頁面以靜態表呈現）：

| 模型 | 方法 | MATH500 | HumanEval | GPQA | AlpacaEval 2.0 |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-Math-7B | Base | 0.496 | 0.329 | 0.278 | 1.61 |
| | Low-temperature | 0.690 | 0.512 | 0.353 | 2.09 |
| | Power Sampling | 0.748 | 0.573 | 0.389 | 2.88 |
| | GRPO (MATH) | 0.785 | 0.537 | 0.399 | 2.38 |
| Qwen2.5-7B | Base | 0.498 | 0.329 | 0.278 | 7.05 |
| | Low-temperature | 0.628 | 0.524 | 0.303 | 5.29 |
| | Power Sampling | 0.706 | 0.622 | 0.318 | 8.59 |
| | GRPO (MATH) | 0.740 | 0.561 | 0.354 | 7.62 |
| Phi-3.5-mini-instruct | Base（instruct 版，已後訓練）| 0.400 | 0.213 | 0.273 | 14.82 |
| | Low-temperature | 0.478 | 0.585 | 0.293 | 18.15 |
| | Power Sampling | 0.508 | 0.732 | 0.364 | 17.65 |
| | GRPO (MATH) | 0.406 | 0.134 | 0.359 | 16.74 |

量測結果（由表直接讀出，`check_closed.py` 計算差值；皆為單次執行、無誤差棒）：
- **低於 GRPO 的格**：MATH500 上兩個 Qwen 都是 GRPO 較高（0.748 對 0.785、0.706 對 0.740）；GPQA 上兩個 Qwen 也是 GRPO 較高（0.389 對 0.399、0.318 對 0.354）。只有 Phi-3.5 在這兩欄較高。
- **高於 GRPO 的格**：HumanEval 三個模型；AlpacaEval 三個模型。
- **表中差值的占比**〔本頁推導〕：在 9 個推理格（3 模型 × MATH500/HumanEval/GPQA）中，8 格的「低溫 − base」占「power − base」的 62%–77%；例外是 Phi-3.5 GPQA（22%）。例：Qwen2.5-Math MATH500，base→低溫 +0.194，低溫→power 再 +0.058。這是對表中數字的分解，不是因果歸因：power sampling 的提議本身就是 τ = 1/α 的取樣器，論文沒有做消融，也沒有誤差棒。GPQA（198 題）的分母很小，換算題數：Qwen2.5-Math base 55、低溫 70、power 77、GRPO 79；Qwen2.5-7B 55、60、63、70（power − base 8 題，低溫 − base 5 題）；Phi-3.5 54、58、72、71。AlpacaEval 不適用此比例：Qwen2.5-7B 低溫（5.29）低於 base（7.05），Phi-3.5 低溫（18.15）高於 power（17.65）。Base 與 Low-temperature 列的溫度論文未寫（推測分別為 1 與 1/α）。
- 「+25.2%」「+51.9%」「+59.8%」是百分點差；+59.8 是對 Phi-3.5 的 GRPO（HumanEval 0.134，低於 base 0.213）算的。
- 作者認為：MATH500 上與 GRPO「on par」，域外任務可高於 GRPO，AlpacaEval 結果顯示可推廣到不可驗證的領域（Sec. 5.2）。

Fig. 4（Qwen2.5-Math-7B，MATH500；論文未給數值，頁面只以文字描述，兩項分開寫）：
- 左圖 log-likelihood（依長度平均）：GRPO "heavily concentrated at the highest likelihood peak"；power sampling 位於較高 likelihood 區但 "maintains noticeable spread"。
- 右圖 confidence（Eq. 13：$\mathrm{Conf}=\frac1{T+1}\sum_t\sum_x p(x\mid x_{<t})\log p(x\mid x_{<t})$，平均負熵，$\le0$，越接近 0 表示每步 next-token 分佈越集中）：power sampling 與 GRPO 位於 "similarly high confidence regions"。

回應長度（MATH500）：base 600、GRPO 671、power sampling 679 token。作者認為這「自然出現」。

pass@k（S8 首次定義）：每題取 k 個樣本，至少一個正確即算解出。

Fig. 5（MATH500 pass@k，k=1…16）：內文寫 power sampling 在 **k>1** 時高於 GRPO，"supersedes that of the base model until finally converging"。**勘誤 E5**：圖說寫「strictly better than both GRPO and the base model」，與兩件事不一致：(1) k=1 時依 Table 1 是 GRPO 較高（0.785 對 0.748；Fig. 5 的數值本頁未讀出，論文未說明 Fig. 5 的 k=1 與 Table 1 是否同一設定）；(2) 同一圖說又寫 "our pass rate at high k matches the base model"，大 k 時對 base 也不是嚴格較高。

App. A.2（作者判斷，無數值）：power sampling 在 k>1 時 "near universally better performance than both GRPO and the base model"；HumanEval 用 $\alpha=1.67$，"this temperature exhibits slightly better results at earlier k"；HumanEval 上 GRPO "exhibits better pass@k than the base model until k = 16"（照錄原文），作者猜測是程式題可有多個正解。曲線數值論文未給，頁面不重繪。

Fig. 6：$\alpha=4.0$ 最佳、$\alpha\ge2.0$ 後大致穩定（MATH500，Sec. 5.3）；$N_{\text{MCMC}}$ 增到 10 前穩定上升（超過 10 持平，not plotted）；2→10 步差「不超過 3–4%」，0→2 步「substantial (3–4%)」——同一數字一處說小一處說大。α 掃描在 MATH500（測試集）上；$N_{\text{MCMC}}$ 掃描的圖說只寫 "Qwen models"，基準未明寫（推測 MATH500）。數值不重繪。

#### 步驟 11：論文未說明的設定（頁面 S9 逐條列出）

1. Base 列的取樣溫度（推測 1）。
2. Low-temperature 列的溫度（推測 1/α = 0.25）。
3. Table 1 主結果的 $N_{\text{MCMC}}$（只在成本估計處出現 10）。
4. AlpacaEval 用的 $\alpha$（只說提議 $\tau=0.5$）。
5. HumanEval 在 Table 1 用哪個 $\alpha$（Sec. 5.1 說 4.0，A.2 的 pass@k 用 1.67）。
6. 區塊內遇到 EOS 如何處理、不同長度序列如何比較 likelihood。
7. GRPO 超參數只說沿用 Shao 2025 與 Abdin 2024。
8. pass@k 的 k 個樣本如何產生（是否 k 次獨立的 Alg. 1）。
9. 無誤差棒、無種子數（GPQA 0.389 對 0.399 是 198 題中的 77 對 79 題〔本頁推導〕）。
10. $x_0$ 是否代表 prompt。
11. Fig. 6 右圖 $N_{\text{MCMC}}$ 掃描用的基準。
12. 提議取樣是否截斷（top-k、top-p）。

#### 步驟 12：實作注意事項（頁面 S9 常見錯誤）

1. 把低溫取樣當成 $p^\alpha$（Prop. 1）。
2. 第 7 行用 $\pi_k$（應為 $\pi_{k+1}$，E4）。
3. 以為前面的區塊被凍結（第 5 行 $m$ 從 1 起抽）。
4. 只重取新區塊：不符合 Alg. 1。
5. 直接把機率相乘：長序列的機率連乘可能低於浮點下限，實作用 log 加總〔本頁推導〕。
6. 以為需要額外前向計算：生成新後綴時同一份 logits 同時給出 $p$ 與 $p_{\text{prop}}$，目前狀態的值可快取〔本頁推導，論文未說明〕。
7. 把 $\tau=0.5$ 的 AlpacaEval 設定或截斷提議套用 Z_t 化簡（不成立）。

### 1.6 論文勘誤一覽（頁面在對應章節以 callout「論文勘誤」呈現）

| # | 位置 | 內容 | 頁面章節 |
| --- | --- | --- | --- |
| E1 | Eq. 2 | 寫 $\alpha\in[1,\infty]$，嚴格不等號需 $\alpha>1$ | S2 |
| E2 | Example 1 | 四個機率總和 0.90 | S3a |
| E3 | Prop. 1 證明 | 對一般 $p$ 而言命題成立，但證明中「每個時間步都不同」在 $t=T$ 不成立 | S3b |
| E4 | Alg. 1 第 7 行 | $\pi_k$ 應為 $\pi_{k+1}$ | S6 |
| E5 | Fig. 5 圖說 | 「strictly better than both GRPO and the base model」過度宣稱：k=1 時依 Table 1 GRPO 0.785 > 0.748；同一圖說又寫大 k 時與 base 相同 | S8 |

---

## 2. 學習目標

每題的參考答案必須以**一段完整、可獨立閱讀的文字**寫在所列章節（只有一個章節）的內文 prose 中（不能只出現在元件、表格、圖或 `<details>` 答案框）。每章末尾 `<details>`「檢查理解」照搬題目與參考答案。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 寫出 LLM 的序列機率與低溫取樣 | 序列機率如何由 next-token 分佈得到？低溫取樣做什麼？ | $p(x_{0:T})=\prod_t p(x_t\mid x_{<t})$（Eq. 1）；低溫取樣每步把 $p(x_t\mid x_{<t})$ 取 $\alpha$ 次方再正規化，溫度 $\tau=1/\alpha$（Eq. 3） | S1a |
| LO2 | 說明 $p^\alpha$ 如何放大高機率序列的相對權重，以及 α 的範圍 | Eq. 2 說什麼？α 需要什麼條件？ | 若 $p(x)>p(x')$，則 $p(x)^\alpha/p(x')^\alpha>p(x)/p(x')$，高 likelihood 序列的相對權重上升（本文稱為銳化）；須 $\alpha>1$，論文寫的 $\alpha\in[1,\infty]$ 在 $\alpha=1$ 時兩邊相等 | S2 |
| LO3 | 手算 Example 1 | α=2 時兩種取樣給 a、b 的權重？例子有什麼瑕疵？ | 冪分佈 0.160 對 0.125（a 的權重較大），低溫 0.160 對 0.250（b 的權重較大）；四個機率總和 0.90 不是 1，但只比相對權重所以大小關係不變；冪分佈給 a 較大權重的條件是 $\alpha>\ln2/\ln1.6\approx1.475$〔本頁推導，利用 p(ba)=p(bb)〕 | S3a |
| LO4 | 區分冪分佈與低溫的 next-token 權重 | 兩者的 next-token 相對權重各是什麼？何時相同？ | 冪分佈 ∝ $\sum_{x_{>t}}p(x_{0:T})^\alpha$（次方的和，Eq. 7）；低溫 ∝ $(\sum_{x_{>t}}p(x_{0:T}))^\alpha$（和的次方，Eq. 8）；冪分佈 = 低溫權重 × $F(x_{\le t})=\sum_{x_{>t}}p(x_{>t}\mid x_{\le t})^\alpha$〔本頁推導〕；$\alpha=1$、各 token 的 $F$ 相同或 $t=T$ 時相同 | S3b |
| LO5 | 敘述 Prop. 3 的條件與適用範圍 | 在什麼條件下冪分佈給正向 pivotal token 較大權重，而低溫給負向 pivotal token 較大權重？這證明了什麼？ | $\varepsilon'/N^{1-1/\alpha}<\varepsilon<\varepsilon'$（Eq. 14），且由 Eq. 15 正向 token 的未來 likelihood 高於負向任一條未來；只在「一條未來 vs N 條均勻未來」的極端模型下比較權重大小，沒有證明答案較正確 | S4 |
| LO6 | 說明為何用 MH 與 MH 的條件 | 為何不能直接從 $p^\alpha$ 取樣？MH 為何可行？收斂需要什麼？ | 正規化要對所有序列求和，不可計算；MH 接受率 Eq. 9 只用 $p^\alpha$ 的比值，正規化常數相消；提議分佈須不可約且非週期（Def. 1，收斂結果引用 Neal 1993），實際上還需提議對 $p^\alpha$ 支撐內的序列給正機率、且雙向提議機率同時為正〔本頁推導〕 | S5a |
| LO7 | 推得接受率的 Z_t 化簡 | 提議為同一 base model、未截斷、$\tau=1/\alpha$ 時，接受率化成什麼？ | 〔本頁推導〕比值 $=\prod_{t=m}^{L}Z_t(x'_{<t})/Z_t(x_{<t})$，$Z_t=\sum_v p(v\mid x_{<t})^\alpha$；只重取區塊最後一個 token 時比值為 1 必接受；截斷提議（top-k、top-p）或 AlpacaEval 的 $\tau=0.5$ 不適用 | S5b |
| LO8 | 描述 Alg. 1 的區塊結構與兩個易誤讀處 | Alg. 1 每一階段做什麼？第 7 行與第 10 行要注意什麼？ | 每階段先用 $p_{\text{prop}}$ 延伸 $B$ 個 token，再跑 $N_{\text{MCMC}}$ 步隨機位置重取 MH（重取到目前區塊結尾），目標 $\pi_{k+1}\propto p(x_{0:(k+1)B})^\alpha$；第 7 行寫 $\pi_k$ 是索引錯誤，應為 $\pi_{k+1}$；第 10 行不凍結前綴，$m$ 從 1 起抽；只有 $N_{\text{MCMC}}\to\infty$ 時輸出才是 $p^\alpha$；B 決定區塊數，在本頁玩具中輸出與 $p^\alpha$ 的 TV 隨 $N_{\text{MCMC}}$ 下降，不同 B 之間沒有一致的大小順序 | S6 |
| LO9 | 計算 token 成本 | 論文報告的成本倍數是多少？精確照 Eq. 12 的和算呢？ | $N_{\text{MCMC}}T/(4B)=10\times679/(4\times192)=8.84$ 倍（Sec. 5.3）；精確求和 9600 token 為 14.14 倍，最後區塊截到 679 為 13.48 倍，約 13.5–14.5 倍〔本頁推導；含初始化為 14.48〕；差異來自近似相當於 $(T/B)^2/2=6.26$，而 $K=\lceil T/B\rceil=4$ 時精確和用 $K(K+1)/2=10$ | S7 |
| LO10 | 讀出 Table 1 的高低與差值分解 | Power sampling 在哪些格低於 GRPO？表中相對 base 的差值有多少比例低溫取樣列也有？ | 兩個 Qwen 模型在 MATH500（0.748 對 0.785、0.706 對 0.740）與 GPQA（0.389 對 0.399、0.318 對 0.354）都低於 GRPO；9 個推理格中 8 格的「低溫 − base」占「power − base」的 62%–77%（例外 Phi-3.5 GPQA 22%），這是表中數字的分解，不是因果歸因，且 GPQA 的差只有數題（Qwen2.5-7B：8 題對 5 題）；HumanEval 與 AlpacaEval 高於 GRPO | S8 |
| LO11 | 正確敘述 pass@k 結果 | Fig. 5 內文說什麼？圖說的哪句話過度宣稱？ | 內文：k>1 時 power sampling 高於 GRPO，大 k 時與 base 收斂；圖說的「strictly better than both GRPO and the base model」不成立：k=1 時依 Table 1 GRPO 0.785 高於 0.748，且同一圖說寫大 k 時與 base 相同 | S8 |
| LO12 | 列出論文未說明的設定 | 重現 Table 1 時缺哪些設定？ | Base 與 Low-temperature 列的溫度、Table 1 的 $N_{\text{MCMC}}$、AlpacaEval 的 α、HumanEval 在 Table 1 的 α、EOS 處理、GRPO 超參數細節、pass@k 樣本的產生方式、$N_{\text{MCMC}}$ 掃描的基準、提議是否截斷、誤差棒與種子數 | S9 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），互動元件放加寬「圖版」（最大 1080px）。章節編號 JetBrains Mono（`01 · …`，子節 `01a`）。章節標題用陳述句。每個符號在首次使用處（含 hero）以文字定義；概念一律先於使用它的元件。視覺記號：一對並排小長條「次方的和｜和的次方」，在 hero、C1、C3 重複出現。

| # | id | 標題 | 內容 | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈不訓練，只取樣：從 base model 的冪分佈取樣做推理〉 | 副標：「這篇論文只用既有模型自己給的機率挑選回答，不再訓練，單次作答的準確率接近經過強化學習後訓練的模型（後文說明）。」三張事實卡（下方）。連結：arXiv、`../index.html#paper-255`、`rswp-lineage-hub.html` | 1 螢幕 |
| S1a | `basics` | 01a · 語言模型逐 token 給出序列的機率 | token、next-token 分佈、Eq. 1、likelihood 與 log-likelihood、溫度 Eq. 3（τ 小時每步的分佈更集中）；base model 定義（含 Phi-3.5 為 instruct 版）。LO1 答案段。符號表（`<dl>` 堆疊、可收合） | 400–450 字 |
| S1b | `rl` | 01b · RL 後訓練與分佈銳化假說 | RL 後訓練（以驗證器給獎勵）、GRPO（一種 RL 演算法，不講數學）、「分佈銳化」假說與文獻觀察（Sec. 1）；pass@k 只說「多次作答指標，S8 定義」。延伸：`rswp-rlvr-grpo-background.html`、`rswp-distribution-sharpening-debate.html` | 300–350 字 |
| S2 | `power` | 02 · 冪分佈把高機率序列的相對權重放大 | Eq. 2；「銳化」以 Eq. 2 定義；E1 callout；靜態 SVG（見下）；為什麼不能直接取樣的預告。LO2 答案段。延伸：`rswp-annealed-tempered-sampling.html` | 300–400 字 + 靜態圖 |
| S3a | `example` | 03a · 兩個 token 的例子：低溫與冪分佈給出不同答案 | 先定義「偏好＝得到較大的相對權重」；一行說明「此處 x_0 是第一個生成的 token」；Example 1 逐步手算（只用四個數字與平方）；E2 callout；翻轉點；C1。LO3 答案段 | 400–450 字 + C1 |
| S3b | `vs-temp` | 03b · 一般情形：次方的和與和的次方 | Eq. 4–8（Eq. 7、Eq. 8 各一個 display）；$F$ 分解〔本頁推導〕並用 Example 1 回驗（F(a)=1、F(b)=0.5）；相等條件；E3 callout。LO4 答案段。延伸：`rswp-power-vs-low-temperature.html` | 350–400 字 |
| S4 | `pivotal` | 04 · 冪分佈給「未來路徑少但機率高」的 token 較大權重 | Obs. 1；Def. 2–4；Eq. 14 先以 Example 1 代入（0.354 < 0.4 < 0.5）再寫一般式；Eq. 15；適用範圍；「作者認為」段（pivotal token，planning 一詞照錄並標為作者用語）；C2。LO5 答案段 | 400–500 字 + C2 |
| S5a | `mh` | 05a · Metropolis-Hastings 只需要未正規化的權重 | 正規化不可行；Markov chain 與平穩分佈（2 句）；Eq. 9；detailed balance 一行〔本頁推導〕；Def. 1；支撐與雙向正機率條件、非週期來自自環〔本頁推導〕。LO6 答案段。延伸：`rswp-metropolis-hastings-autoregressive.html` | 350–400 字 |
| S5b | `resample` | 05b · 隨機位置重取與接受率的化簡 | 隨機重取提議（Fig. 3 的文字描述，重取到目前區塊結尾）；接受率展開；Z_t 化簡〔本頁推導〕、兩個推論（(b) 因共同前綴的 Z_t 相消）、適用條件（未截斷、τ = 1/α、同一模型）；C3a。LO7 答案段 | 350–400 字 + C3a |
| S6 | `alg1` | 06 · Algorithm 1 逐區塊執行 MH | mixing time 的直覺；Eq. 10–11；Alg. 1 逐行（≥768px 兩欄：左為行號與原文、右為中文說明；<768px 每行下方接說明）；E4 callout；第 10 行與 $\pi_k$ 邊際說明；TV 定義一句（「兩分佈差異的總量，0 表示相同、1 表示不重疊；C3 以精確分佈傳遞計算，不是抽樣」）；C3。LO8 答案段。延伸：`rswp-power-sampling-algorithm-results.html` | 500–600 字 + C3 |
| S7 | `cost` | 07 · 成本與生成長度的平方成正比 | Eq. 12 推導；8.84 倍；13.5–14.5 倍〔本頁推導〕；與 GRPO 成本的比較（作者的假設）；C4。LO9 答案段 | 300–400 字 + C4 |
| S8 | `results` | 08 · 單次準確率接近 GRPO，部分基準較低 | 設定（步驟 9）；pass@k 定義；Table 1 靜態表（Power vs GRPO 以 ▲ 較高／▼ 較低文字標示；Phi Base 列加註 instruct）；「量測結果」與「作者認為」分開的段落；差值分解表（9 格，附 GPQA 題數）；Fig. 4 兩項分開、長度、Fig. 5（E5 callout）、A.2（作者判斷）、Fig. 6 以文字描述；明寫「訓練與基準數值無法在瀏覽器重現，本頁不重繪曲線」。LO10、LO11 答案段 | 700–850 字 + 2 表 |
| S9 | `impl` | 09 · 論文未說明的設定與常見錯誤 | 步驟 11 十二條、步驟 12 七條。LO12 答案段 | 300–400 字 |
| S10 | `history` | 10 · 研究脈絡 | 時間線（5.4）+ 一段：RLHF → GRPO/RLVR → 銳化之爭 → 內部訊號獎勵 → MCMC 前作與退火 → 本文 | 時間線 + 120 字 |
| S11 | `next` | 11 · 限制與延伸閱讀 | 限制（likelihood 與推理的關係是相關不是因果；成本；未測 mixing）；筆記清單；arXiv | 200 字 |
| — | — | 頁尾 | 引用格式、arXiv；`#pm-bar` | — |

Hero 三張事實卡（只用卡內定義的詞；MH、GRPO、成本倍數不出現，以「後文說明」帶過）：
1. 「base model 指作為取樣對象、不再訓練的既有語言模型，它對每條回答序列 x 給出機率 p(x)。本文的取樣目標是 p(x)^α（α 是大於 1 的冪次，論文用 4），它使機率較高的回答得到更大的相對權重。」
2. 「常用的低溫取樣在每一步把下一個 token 的機率取 α 次方再重新正規化。這和對整條序列的機率取 α 次方得到的分佈不同（03a 節用一個兩 token 的例子說明）。」
3. 「直接從 p(x)^α 取樣需要對所有可能序列加總，無法計算。作者改為反覆執行：隨機選一個位置，從該位置重新生成到目前區塊結尾，再依新舊序列的機率比值決定是否接受。這個方法的名稱、準確率與計算成本在後文說明。」

S2 靜態圖：$p(y)=0.6\,\mathcal N(y;-1,0.6^2)+0.4\,\mathcal N(y;1.5,0.4^2)$〔示範值〕與正規化後的 $p^4$，橫軸 $[-3,3]$ 取 601 點以梯形法正規化。實線（p）／虛線（p⁴）＋直接標籤。圖說首句：「混合參數為本頁示範值；論文 Fig. 2 未給參數。」

桌機（≥1024px）左側 sticky 章節目錄；<1024px 頂部可橫捲章節列（沿用參考頁 `.toc`／`#tocbar`）。

---

## 4. 互動元件規格

共通規則：
- 每個元件**一個** `aria-live="polite"` 區（`.sr`），由 `announce(text)` 以 500ms debounce 更新（滑桿 `input` 時重設計時器，`change` 時立即更新；按鈕按下時呼叫一次）。SVG `role="img"`，其 `aria-label` 只在 `change` 或按鈕事件後更新。
- 分段按鈕：`role="radiogroup"`，子項 `role="radio"`＋`aria-checked`，左右方向鍵切換。
- 動態讀數一律 Unicode，不用 KaTeX、不用組合字元（不寫 ᾱ、x̄）；上標在 HTML 用 `<sup>`，在 SVG 用 `<tspan dy="-0.45em" font-size="0.75em">`。只有檢查題答案與勘誤 callout 內含 KaTeX，插入或展開後呼叫 `renderMath()`。
- 曲線與標記以線型／形狀＋直接文字標籤區分；標記最小 10px。觸控：讀數固定在讀數框，不依賴 hover。
- 390px：控制列在圖下方單欄堆疊，觸控區 ≥44px，SVG 寬度＝容器寬度（重畫，不縮字），圖上文字 12–13px；讀數表包 `overflow-x:auto`。`.katex-display` 與橫捲表格右緣加 16px 漸層陰影提示（以 `background-attachment: local` 的 scroll-shadow 實作，捲到底時消失）。
- 圖版底部一行來源說明（「讀數由本頁依 Eq. ○ 即時計算」＋必要的「本頁推導」標籤）。
- 「計算中」的 opacity 變化不加 transition（reduced-motion 下也一致）。

### C1 · 兩個 token 的例子：Example 1 計算器（S3a）

- **教的概念**：Example 1 的兩組權重；翻轉點；E2。C1 不顯示 $F$（$F$ 在 S3b 才定義）；S3b 內文以 Example 1 的 F(a) = 1、F(b) = 0.5 靜態回驗。
- **輸入**：分段「機率表」：「論文原例（總和 0.90）」（預設，$p=(0,0.40,0.25,0.25)$ 依序 aa、ab、ba、bb）／「總和為 1 的版本〔本頁推導〕」（$(0,0.45,0.275,0.275)$）。兩組都滿足 p(ba) = p(bb)，翻轉點公式依賴此條件，圖版註明。滑桿 α ∈ [1, 8]，步長 0.01，預設 2.00；chip「α = 2（論文）」「α = 4（主實驗）」。
- **計算**（唯一定義 `ex1(p, a)`，畫面與 selfTest 共用）：
  - `powA = p_aa^a + p_ab^a`，`powB = p_ba^a + p_bb^a`；`tmpA = (p_aa+p_ab)^a`，`tmpB = (p_ba+p_bb)^a`；冪次一律 `Math.pow`（`Math.pow(0,a)=0`，a ≥ 1）。
  - `ex1` 仍回傳 $F(a)=\sum_j (p_{aj}/r_a)^\alpha$（$r_a=p_{aa}+p_{ab}$；$r_a=0$ 時 F = 1）供 selfTest 驗證 $r^\alpha F$ = powA，但畫面不顯示。
  - 正規化：`PpowA = powA/(powA+powB)`，`PtmpA = tmpA/(tmpA+tmpB)`；序列分佈：冪分佈 $p_i^\alpha/\sum p^\alpha$；低溫 = PtmpA × 第二步低溫條件分佈。
  - 翻轉點 $\alpha^*=\ln2/\ln(p_{ab}/p_{ba})$（只在 p(aa)=0、p(ba)=p(bb) 時成立；原例 1.4748，總和 1 版本 1.4075）。
  - 顯示 4 位有效數字（<0.001 改為科學記號）。
- **顯示**：左：兩組並排長條「次方的和（冪分佈）」「和的次方（低溫）」，各含 a、b，長條內寫數值，a 實心、b 斜線填滿，權重較大的一方旁加「◀ 權重較大」。右：α 軸（1–8）上 $P_{\text{pow}}(x_0=a)$（實線）與 $P_{\text{temp}}(x_0=a)$（虛線），0.5 參考線，α* 處垂直細線標「α* = 1.475」，目前 α 用圓點。讀數表：powA、powB、tmpA、tmpB、序列機率（ab、ba、bb）。
- **390px**：長條在上、曲線在下；讀數表橫捲。
- **觀察**：α=1 時兩者相同；α 越過 α* 後冪分佈給 a 較大權重，低溫始終給 b 較大權重且差距擴大；$F(b)=2^{1-\alpha}$ 隨 α 變小。

### C2 · Prop. 3 的區間（S4）

- **教的概念**：Eq. 14 區間；兩種取樣各給哪個 token 較大權重；未來 likelihood 比較（Eq. 15）。
- **輸入**：預設只顯示 ε ∈ [0.01, 0.95]（步長 0.01，預設 0.40）與 N ∈ {1,…,64}（整數，預設 2）。`<details>`「進階」內：ε′ ∈ [0.05, 0.95]（預設 0.50）、α ∈ [1, 8]（步長 0.01，預設 2.00）。約束 ε + ε′ ≤ 1，標為「前綴為空（如 Example 1）的情形；一般前綴下上限是該前綴的機率」：拖動造成超過時把另一個值夾到 1 − 該值，並顯示「已調整使 ε + ε′ ≤ 1」。chip「Example 1（ε = 0.4、ε′ = 0.5、N = 2、α = 2）」。
- **計算**（唯一定義 `prop3(e, ep, N, a)`）：下界 `lo = ep / N^(1-1/a)`；`lPos = a*ln(e)`；`lNegPow = a*ln(ep) - (a-1)*ln(N)`；`lNegTmp = a*ln(ep)`；兩者之間選 $x_t$ 的機率 `sigmoid(lPos - lNeg)`（z ≥ 0 用 `1/(1+exp(-z))`，否則 `exp(z)/(1+exp(z))`）；未來比較 $\varepsilon$ 對 $\varepsilon'/N$；N = 1 或 α = 1 顯示「區間為空（需 N > 1 且 α > 1）」。
- **顯示**：水平 ε 軸 [0, 1]：區間 (lo, ε′) 塗淺色、括號標「Eq. 14 區間」，ε′/N 以小三角標，目前 ε 以菱形。下方兩列「冪分佈：只在 x_t 與 x′_t 兩者之間選時，選 x_t 的機率 0.561（x_t 權重較大）」「低溫：同上 0.390（x′_t 權重較大）」，各附長條；第三列「x_t 的未來 0.400 ＞ x′_t 每條未來 0.250」。
- **觀察**：預設值落在區間內；增大 N 下界左移；（進階）α→∞ 時下界趨近 ε′/N；ε > ε′ 時兩者都給 x_t 較大權重。

### C3a · 單步 Metropolis-Hastings（S5b）

- **教的概念**：Fig. 3 的一步：選位置、重取、算比值、抽 u；Z_t 化簡；m = L 必接受；推論 (b)。
- **玩具模型**：與 C3 共用（見 C3「玩具模型」；本元件固定 V = 3、seed = 1、α = 4、L = 6，即單一階段 B = 6 的單純 MH）。提議溫度分段「1/α」（預設）／「1」。
- **PRNG 與步驟**（逐字依此順序取亂數）：`r = mulberry32(chainSeed)`，chainSeed 預設 3，「重新開始」使 chainSeed + 1。初始化：從 $p_{\text{prop}}$ 逐 token 抽 6 個（反 CDF：`u=r()`，依 v=0..V−1 累加機率，第一個使累加 > u 的 v；都不滿足取 V−1）。「下一步」：`m = 1 + floor(r()*6)`；從 m 到 6 逐 token 以同法抽；算 $A=\min\{1,\exp(\Sigma)\}$，Σ 為 $\sum_{t\ge m}[\log w(x'_t)-\log w(x_t)]$（τ = 1/α 時等於 $\sum_{t\ge m}[\log Z_t(x'_{<t})-\log Z_t(x_{<t})]$）；`u = r()`；`u ≤ A` 接受。
- **顯示**：目前序列 6 格（m 之後加粗框）、候選序列；τ = 1/α 時列出每個 $t\ge m$ 的 $\log Z_t(x'_{<t})-\log Z_t(x_{<t})$ 與總和〔本頁推導〕；τ = 1 時改列 $\log w$ 差並註明「化簡不適用」；A、u、「接受 ✓／拒絕 ✕」；步驟歷史表（最近 8 步）。每按一次 `announce` 一次（「第 6 步：m = 4，A = 0.43，拒絕」）。
- **390px**：6 格每格 44px、排成 3×2；歷史表橫捲。
- **觀察**：m = 6 的提議一定被接受；第 6–8 步 m 不同但 A 相同（同一對新舊序列）。

### C3 · 小型語言模型上的 Algorithm 1 輸出分佈（S6）

圖版首行（必須原樣出現）：「本圖的 base model 是由亂數種子產生的小型語言模型〔示範值〕；它只檢驗 Algorithm 1 的輸出分佈是否趨近 p^α，不涉及答案正確性，也不代表哪一種取樣比較好。」

- **教的概念**：Alg. 1 的輸出分佈隨 $N_{\text{MCMC}}$ 從提議分佈移向 $p^\alpha$；$N_{\text{MCMC}}=0$ 且 $\tau=1/\alpha$ 時輸出正是低溫分佈；下降速度依 B、α、提議溫度、模型而異。
- **玩具模型**（〔示範值〕；頁面、C3a 與 `check_toy.py` 完全相同）：
  - 詞彙 V ∈ {2, 3}（token 顯示為 a、b、c），生成 6 個 token $x_1..x_6$；提示 token $x_0$ 固定、不重取，也不進入模型（next-token 只依生成前綴 $x_1..x_{t-1}$ 計算）。
  - PRNG（必須逐字使用）：
    ```js
    function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}}
    ```
  - 前綴編碼：生成前綴 $(x_1..x_\ell)$（token 以 0..V−1 表示）的 `code = Σ_{i<ℓ} V^i + Σ_i x_i·V^(ℓ−1−i)`（第一個 token 為最高位）。
  - next-token：`r = mulberry32(seed*100003 + code)`；依序取 V 個 `r()`，logit$_v$ = 3.0 · r()（BETA = 3.0〔示範值〕）；softmax。seed 預設 1，「換一個模型」使 seed + 1（1–999 循環）。
  - 錨點：seed 1、V 3：$p(\cdot\mid\varnothing)=(0.2731,0.4845,0.2424)$；`mulberry32(1)` 前三值 0.627074、0.002736、0.527447。
- **輸入**：V（分段 2 / 3，預設 3）；α ∈ [1, 6] 步長 0.5 預設 4；提議溫度（分段「1/α（論文推理任務）」預設／「1」／「0.5」）；B（分段 1 / 2 / 3 / 6，預設 2）；$N_{\text{MCMC}}$ 滑桿 0–50 預設 10；「換一個模型」。
- **精確計算**（不抽樣；唯一定義在 `toyCore`）：
  1. 列舉 $V^6$ 條序列；以 log 計算 base $p$、低溫 $p_{\text{temp}}(x)=\prod_t p(x_t\mid x_{<t})^\alpha/Z_t$、精確 $p^\alpha$（log-sum-exp 正規化）。
  2. 每個區塊 $k$，$L=\min((k+1)B,6)$：建立長度 $L$ 全部狀態上的稠密轉移矩陣（Float64Array，維度 ≤ 729²）。對狀態 x、每個 $m\in\{1..L\}$（機率 1/L）、每個後綴 s：$y=x_{<m}\Vert s$，$q=\prod_{t\ge m}p_{\text{prop}}$，log 比值 $=\sum_{t\ge m}\log w(y_t)-\sum_{t\ge m}\log w(x_t)$，$\log w=\alpha\log p-\log p_{\text{prop}}$（目標 $\pi_{k+1}$）；log 比值 ≥ 0 時 A = 1，否則 $e^{\text{log 比值}}$；$K[x,y]$ += $qA/L$，$K[x,x]$ += $q(1-A)/L$。以後綴累積和預先計算 $\sum_{t\ge m}$。
  3. 傳遞：$\mu$ 從空前綴開始；每區塊先以 $p_{\text{prop}}$ 延伸到 L，再乘 $K$ 共 N 次。
  4. TV 曲線：$N\in\{0,1,2,3,5,10,20,50\}$ 與目前滑桿值。
- **Worker 協定與快取**：
  - key = `${V}|${alpha}|${tau}|${B}|${seed}`。worker 內 `Map` 快取 `{matrices, base, temp, pow, tvCurve}`，最多 8 個 key（LRU）。
  - 設定改變：送 `{reqId, key, V, alpha, tau, B, seed, nList}`，worker 建矩陣、算 TV 曲線與目前 N 的分佈。
  - 只改 N：送 `{reqId, key, nList:[N]}`，worker 直接用快取矩陣傳遞（命中失敗時回 `{reqId, miss:true}`，主執行緒改送完整請求）。N 滑桿拖動以 150ms debounce 發送。
  - 回傳 `{reqId, key, top:[{s, base, temp, pow, alg}], other, tvBase, tvTemp, tvCurve, ms}`；主執行緒丟棄 `reqId` 小於最新請求的結果。
  - 耗時預算：V = 3、B = 6 建矩陣（約 80 萬項）＋ TV 曲線（共 91 次 729² 矩陣向量乘）目標 < 300ms；超過 500ms 時讀數框顯示「計算中…（區塊 k/3）」，worker 每完成一個區塊回報一次進度。
  - Worker 建立失敗：退回主執行緒，以 `setTimeout` 逐區塊分段執行同一份函式。
- **顯示**：上：「前 10 條序列」排行（依 $p^\alpha$ 排序，第 11 列「其他」合計），每列一條軌道上四個標記：base ○、低溫 △、精確 p^α ■（長條）、Alg. 1 輸出 ◆（皆 ≥10px）；序列字串在左（JetBrains Mono）。下：TV 對 $N_{\text{MCMC}}$（橫軸 0–50，平方根刻度標 0, 1, 2, 5, 10, 20, 50），折線＋圓點；水平虛線「TV(低溫, p^α)」「TV(base, p^α)」；目前 N 以菱形。讀數：三個 TV、耗時（ms）。
- **390px**：排行每列兩行（序列字串一行、軌道一行），TV 圖在下；控制列單欄。
- **觀察**（只寫這些）：$N=0$ 且 τ = 1/α 時 Alg. 1 輸出與低溫分佈重合；TV 隨 $N_{\text{MCMC}}$ 下降，下降速度依設定而異（例：預設 N = 50 時 0.0041，V = 2 時 0.0144）；提議溫度 1 時 N = 0 的起點是 base；α = 1 時所有分佈相同。不排序 B（N = 10 時 B = 3 的 0.0601 低於 B = 2 的 0.0624，B = 6 為 0.1056，沒有單調關係）。

### C4 · 成本計算器（S7）

- **教的概念**：Eq. 12 的近似與精確和；8.84 倍；13.5–14.5 倍〔本頁推導〕；成本 ∝ T²。
- **輸入**：$N_{\text{MCMC}}$ 0–50 整數預設 10；生成長度 T 1–3072 整數預設 679（滑桿＋數字輸入框，chip「T = 679（平均長度）」「T = 3072（上限）」）；B 1–3072 整數預設 192（chip「B = 192」）。
- **計算**（唯一定義 `cost(N, T, B)`）：`K = ceil(T/B)`；近似倍數 `N*T/(4*B)`；Eq. 12 精確和倍數 `N*B*K*(K+1)/4 / T`；截斷版 `N*Σ_{k=1..K} min(kB,T)/2 / T`；截斷＋初始化 `截斷版 + 1`。截斷和以 ≤ 3072 次迴圈計算。非整數或超範圍輸入以 `Math.round` 夾到範圍內；N = 0 時三者為 0、加初始化為 1。
- **顯示**：四列長條（近似＝實線外框「論文 8.84×」，其餘三列標「本頁推導」pill），長度以最大值正規化，數值寫在末端（兩位小數＋「×」）。下方小圖：橫軸 T（0–3072），近似（虛線）與截斷＋初始化（實線）兩條曲線，目前 T 標點。一行文字：「K = ⌈T/B⌉ = 4；精確和用 K(K+1)/2 = 10；近似相當於 (T/B)²/2 = 6.26。」
- **390px**：長條全寬，曲線在下。
- **觀察**：預設 8.84 對 14.14／13.48／14.48；T = 3072 時近似 40.00；B = T 時近似 2.5、精確 5。

---

## 5. 實作計畫

### 5.1 檔案與外殼

- 單一檔 `scripts/tutorial_src/255/build/index.html`，上線為 `tutorials/reasoning-with-sampling.html`。CSS、JS、時間線資料全部內嵌，不 `fetch` 本機檔。
- **沿用參考頁 `tutorials/prediction-targets-k-diff.html` 的外殼，不重寫**：CSS tokens（含深色模式與 `pm-theme`）、appbar、`.toc`／`#tocbar`、`.plate`／`.plate-h`／`.plate-f`／`.readout`／`.seg`／`.rng`／`.legend`／`.check` 元件、`.eq`＋`.eq-n` 方程式列、`#pm-bar`、KaTeX 0.16.9 設定與 `renderMath()`、`?test` 的 `selfTest()` 骨架、時間線元件。
- 連結：hero「在關聯圖中查看」→ `../index.html#paper-255`；arXiv `https://arxiv.org/abs/2510.14901`；筆記 `<slug>.html`。

### 5.2 本頁新增

- 色彩：參考頁的 `--c-par`、`--c-perp` 改名為 `--c-pow`（冪分佈系列，light #3F6E8C／dark #8DB4D3）與 `--c-temp`（低溫系列，light #6B7A3A／dark #A9BC7C）；`--c-base` 用 `--ink-3`；Alg. 1 輸出用 `--accent`。C1 的斜線填滿用 SVG `<pattern>`，stroke 取 `currentColor`（由所屬系列 token 設定），深淺色自動切換；虛線與參考線用 `--ink-3`。
- callout 類型 `.erratum`（標題「論文勘誤 E#」，左側 ⚠ 文字符號＋`--line` 邊框，不用彩色粗邊）與 `.derive`（「本頁推導」pill）、`.demo`（「示範值」pill）、`.claim`（「作者認為」段落前綴）。
- S6 Alg. 1 兩欄逐行格線（沿用參考頁 S8 的程式碼對照版面）。
- S8 兩張表：Table 1（Power vs GRPO 以 ▲ 較高／▼ 較低文字標示；Phi Base 列加註 instruct）與「表中差值分解」表（9 格，欄：模型、基準、power − base、低溫 − base、比例；GPQA 三列另附題數）。表題寫明「分解，不是因果歸因」。
- 符號表改為 `<dl>` 堆疊（窄螢幕不出現兩欄表）。

### 5.3 程式結構（單一 IIFE；每個函式只定義一次）

- 共用：`ex1`、`prop3`、`cost`（C1、C2、C4 與 selfTest 共用）。
- `toyCore`：一個自足函式（不引用外部變數），內含 `mulberry32`、`makeLM(V,T,seed)`、`dists(...)`、`stageKernels(...)`、`alg1Dist(...)`、`tv`、`singleStep(...)`、`logZ` 與 worker 的 key 快取。Worker 以 `new Worker(URL.createObjectURL(new Blob(['('+toyCore.toString()+')(self)'])))` 建立；`toyCore(self)` 在 worker 內註冊 `onmessage`。主執行緒以 `const TOY = toyCore(null)` 取得同一份函式（參數為 null 時只回傳函式表，不註冊），供 C3a、主執行緒退路與 selfTest 使用。只有一份定義。建置規則：不得壓縮或轉譯 `toyCore`（`toString()` 要能原樣在 worker 執行）。
- Worker 建立失敗（例如 CSP）：退回主執行緒，以 `setTimeout` 逐區塊分段計算，顯示「計算中…」。
- 訊息協定見 C3「Worker 協定與快取」。
- 元件 init：`initC1()`、`initC2()`、`initC3a()`、`initC3()`、`initC4()`；各有 `announce(text)`（500ms debounce 的唯一 live region）。
- `selfTest()`（`?test`）：
  - `ex1(原例, 2)`：powA 0.160、powB 0.125、tmpA 0.160、tmpB 0.250；PpowA 0.5614、PtmpA 0.3902（±1e-4）；α* 1.4748（±1e-4）。
  - `prop3(0.4,0.5,2,2)`：lo 0.3536、P_pow 0.5614、P_temp 0.3902（±1e-4）。
  - `cost(10,679,192)`：8.841、14.138、13.483、14.483（±1e-3）。
  - `ex1` 的 0^α 與列和 0：`ex1([0,0,.5,.5],2)` 的 F(a) = 1、powA = 0。
  - `TOY`：`mulberry32(1)` 前三值（±1e-6）；seed 1 V 3 的 $p(\cdot\mid\varnothing)$（±1e-4）；Z_t 恆等式最大誤差 < 1e-12（V 2、T 4、α 4）；V 3、α 4、τ 1/α、B 2：TV(低溫) 0.3150、N = 0 TV 0.3150、N = 10 TV 0.0624（±1e-3）；C3a 預設 8 步軌跡第 6 步 A = 0.4335、拒絕。
  - Worker 一致性（非同步，結果於 worker 回覆後計入）：同一設定（V 3、α 4、τ 1/α、B 2、seed 1、N = 10）worker 回傳的 TV 與主執行緒 `TOY` 計算值差 < 1e-12；只送 `{key,nList:[10]}` 的快取路徑回傳相同值。Worker 不可用時記為「略過」而非失敗。
  - `scrollWidth <= innerWidth`。輸出 `selfTest: N passed, M failed`。

### 5.4 研究脈絡（`notes/timeline.json`，14 筆）

- 建置時原樣貼進 `<script type="application/json" id="timeline-data">`，沿用參考頁時間線元件（卡片列、篩選 chip、缺欄位處理、解析失敗訊息）。
- `TL_GROUP`（以 arXiv ID 為 key；未列者歸「其他」）：

  | 類別 | arXiv | 卡片連到的筆記 |
  | --- | --- | --- |
  | RL 後訓練 | 2203.02155、2402.03300、2501.12948、2503.24290 | `rswp-rlvr-grpo-background.html` |
  | 銳化之爭 | 2504.13837、2506.02355、2509.06941 | `rswp-distribution-sharpening-debate.html` |
  | 內部訊號獎勵 | 2505.22660、2505.19590、2506.10947 | `rswp-label-free-likelihood-rewards.html` |
  | MCMC／SMC | 2404.17546 | `rswp-metropolis-hastings-autoregressive.html` |
  | 退火／pivotal token | 2503.02819、2502.00921 | `rswp-annealed-tempered-sampling.html`、`rswp-power-vs-low-temperature.html`（2502.00921） |
  | 本文 | 2510.14901 | `rswp-power-sampling-algorithm-results.html` |

- S11 延伸閱讀（總覽閱讀順序）：`rswp-lineage-hub`、`rswp-rlvr-grpo-background`、`rswp-distribution-sharpening-debate`、`rswp-label-free-likelihood-rewards`、`rswp-power-vs-low-temperature`、`rswp-metropolis-hastings-autoregressive`、`rswp-annealed-tempered-sampling`、`rswp-power-sampling-algorithm-results`，各附 frontmatter `title`。

### 5.5 文案與數學

- 繁體中文、台灣用語、字面學術語氣；不用反問、比喻、宣傳形容詞。專有名詞首次出現附英文（冪分佈 power distribution、低溫取樣 low-temperature sampling、提議分佈 proposal distribution、混合時間 mixing time）。
- 三種陳述分開：量測結果（直接引用表、圖）、作者認為、本頁推導。
- 頁面的方程式與數字必須出自第 1 節；第 1 節以外不得新增論文數字。訓練與基準曲線（Fig. 4–9）不重繪、不估讀座標。
- 公式旁等寬字標註 Eq. 編號；`.katex-display{overflow-x:auto}` 加右緣捲動陰影提示；Eq. 7 與 Eq. 8 各為一個 display（不並列）；Z_t 乘積與 Eq. 12 在 <600px 以 `\\` 斷成兩行（aligned 環境）。
- 用詞規則（字面敘述）：「偏好」只在 S3a 定義（得到較大的相對權重）後使用，元件改寫「權重較大」；「困在」不用；「planning」只在「作者認為」段照錄英文；「銳化」在 S2 以 Eq. 2 定義後使用，「尖」改「集中」；「勝出／輸給」改「較高／較低」。

### 5.6 建置後檢查

截圖 390／768／1024／1440、深色、鍵盤 focus、reduced-motion；依 AGENTS.md 步驟 6-3 八項逐項自評修正。`?test` 全部通過；逐條核對 12 個 LO 的參考答案確實寫在指定章節的內文段落；確認五個勘誤 callout 都在頁面上；C3 首行句子原樣出現；hero 卡沒有出現 MH、GRPO、8.84；390px 下所有 KaTeX display 與表格可橫捲且有陰影提示。

---

## 6. 驗收數值表（`verify/check_closed.py`、`verify/check_toy.py` 輸出，已完整執行）

### C1（容許 ±1e-4）

| 機率表 | α | powA / powB | tmpA / tmpB | F(a) / F(b) | P_pow(x₀=a) | P_temp(x₀=a) | 序列 冪分佈 ab, ba, bb | 序列 低溫 ab, ba, bb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 原例 | 1 | 0.4000 / 0.5000 | 0.4000 / 0.5000 | 1 / 1 | 0.4444 | 0.4444 | .4444 .2778 .2778 | .4444 .2778 .2778 |
| 原例 | 2 | 0.1600 / 0.1250 | 0.1600 / 0.2500 | 1 / 0.5 | 0.5614 | 0.3902 | .5614 .2193 .2193 | .3902 .3049 .3049 |
| 原例 | 4 | 0.0256 / 0.0078 | 0.0256 / 0.0625 | 1 / 0.125 | 0.7662 | 0.2906 | .7662 .1169 .1169 | .2906 .3547 .3547 |
| 原例 | 8 | 6.554e-4 / 3.052e-5 | 6.554e-4 / 3.906e-3 | 1 / 0.0078 | 0.9555 | 0.1437 | .9555 .0222 .0222 | .1437 .4282 .4282 |
| 總和 1 | 2 | 0.2025 / 0.1513 | 0.2025 / 0.3025 | 1 / 0.5 | 0.5724 | 0.4010 | .5724 .2138 .2138 | .4010 .2995 .2995 |
| 總和 1 | 4 | 0.0410 / 0.0114 | 0.0410 / 0.0915 | 1 / 0.125 | 0.7819 | 0.3095 | .7819 .1091 .1091 | .3095 .3453 .3453 |

翻轉點 α*：原例 1.4748，總和 1 版本 1.4075。

### C2（容許 ±1e-4）

| ε, ε′, N, α | 下界 | P_pow(選 x_t) | P_temp(選 x_t) | x_t 未來較高 |
| --- | --- | --- | --- | --- |
| 0.4, 0.5, 2, 2 | 0.3536 | 0.5614 | 0.3902 | 是 |
| 0.4, 0.5, 2, 1 | 0.5000（區間空） | 0.4444 | 0.4444 | 是 |
| 0.3, 0.5, 2, 2 | 0.3536 | 0.4186 | 0.2647 | 是 |
| 0.3, 0.5, 4, 2 | 0.2500 | 0.5902 | 0.2647 | 是 |
| 0.3, 0.6, 8, 4 | 0.1261 | 0.9697 | 0.0588 | 是 |
| 0.4, 0.5, 1, 4 | 0.5000（區間空） | 0.2906 | 0.2906 | 否 |
| 0.2, 0.7, 64, 8 | 0.0184 | 1.0000 | 0.0000 | 是 |

### C3（同一演算法、精確傳遞，容許 ±1e-3；只差浮點加總順序）

錨點：`mulberry32(1)` → 0.627074、0.002736、0.527447（±1e-6）；seed 1、V 3：$p(\cdot\mid\varnothing)=(0.2731,0.4845,0.2424)$、$p(\cdot\mid a)=(0.7624,0.1445,0.0931)$；Z_t 恆等式最大誤差 1.8e-15（< 1e-12）。

| V | seed | α | τ_prop | B | TV(base,p^α) | TV(低溫,p^α) | TV(Alg. 1, p^α)，N = 0, 1, 2, 3, 5, 10, 20, 50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 1 | 4 | 1/α | 2 | 0.6886 | 0.3150 | 0.3150, 0.2454, 0.1995, 0.1657, 0.1191, 0.0624, 0.0287, 0.0041 |
| 3 | 1 | 4 | 1/α | 1 | 0.6886 | 0.3150 | 0.3150, 0.2411, 0.1879, 0.1523, 0.1080, 0.0566, 0.0246, 0.0029 |
| 3 | 1 | 4 | 1/α | 3 | 0.6886 | 0.3150 | 0.3150, 0.2544, 0.2120, 0.1780, 0.1276, 0.0601, 0.0299, 0.0056 |
| 3 | 1 | 4 | 1/α | 6 | 0.6886 | 0.3150 | 0.3150, 0.2645, 0.2306, 0.2033, 0.1622, 0.1056, 0.0550, 0.0081 |
| 3 | 1 | 4 | 1 | 2 | 0.6886 | 0.3150 | 0.6886, 0.4603, 0.3304, 0.2409, 0.1350, 0.0379, 0.0150, 0.0066 |
| 3 | 1 | 2 | 1/α | 2 | 0.4189 | 0.1367 | 0.1367, 0.0704, 0.0400, 0.0248, 0.0107, 0.0015, 0.0004, 0.0000 |
| 2 | 1 | 4 | 1/α | 2 | 0.6596 | 0.3247 | 0.3247, 0.3018, 0.2789, 0.2568, 0.2169, 0.1439, 0.0693, 0.0144 |
| 3 | 2 | 4 | 1/α | 2 | 0.7992 | 0.4389 | 0.4389, 0.2824, 0.1888, 0.1300, 0.0663, 0.0218, 0.0139, 0.0051 |
| 3 | 1 | 1 | 1 | 2 | 0.0000 | 0.0000 | 全為 0.0000 |

預設設定前 3 條序列（base / 低溫 / p^α）：bacabb 0.0344 / 0.6569 / 0.4074；bacbca 0.0250 / 0.0376 / 0.1131；aacaba 0.0249 / 0.0409 / 0.1123。（此玩具中低溫把最高序列推得比 p^α 更集中；頁面不對此下一般性結論。）

C3a 單步（V 3、seed 1、α 4、τ 1/α、L = 6、chainSeed 3；初始化 bacabb；`check_toy.py` 主程式輸出；A、u 容許 ±1e-4）：

| 步 | m | 候選 | A | u | 結果 |
| --- | --- | --- | --- | --- | --- |
| 1 | 3 | bacabb | 1.0000 | 0.0836 | 接受 |
| 2 | 5 | bacabb | 1.0000 | 0.8798 | 接受 |
| 3 | 2 | bacabb | 1.0000 | 0.5399 | 接受 |
| 4 | 1 | bacabb | 1.0000 | 0.1543 | 接受 |
| 5 | 3 | bacaab | 1.0000 | 0.8631 | 接受 |
| 6 | 4 | bacabb | 0.4335 | 0.5818 | 拒絕 |
| 7 | 1 | bacabb | 0.4335 | 0.5709 | 拒絕 |
| 8 | 5 | bacabb | 0.4335 | 0.3723 | 接受 |

（第 6–8 步的 m 不同而 A 相同，對應 Z_t 化簡推論 (b)。）

### C4（容許 ±1e-3）

| N, T, B | K | 近似 | Eq. 12 精確和 | 截斷 | 截斷＋初始化 |
| --- | --- | --- | --- | --- | --- |
| 10, 679, 192 | 4 | 8.841 | 14.138 | 13.483 | 14.483 |
| 10, 3072, 192 | 16 | 40.000 | 42.500 | 42.500 | 43.500 |
| 2, 679, 192 | 4 | 1.768 | 2.828 | 2.697 | 3.697 |
| 10, 679, 679 | 1 | 2.500 | 5.000 | 5.000 | 6.000 |
| 0, 679, 192 | 4 | 0.000 | 0.000 | 0.000 | 1.000 |
| 10, 679, 1 | 679 | 1697.500 | 1700.000 | 1700.000 | 1701.000 |

### S8 表中差值分解（靜態表，容許 ±0.01；GPQA 題數 ±1 題）

| 模型 | MATH500 | HumanEval | GPQA |
| --- | --- | --- | --- |
| Qwen2.5-Math-7B | 0.77 | 0.75 | 0.68 |
| Qwen2.5-7B | 0.63 | 0.67 | 0.62 |
| Phi-3.5-mini | 0.72 | 0.72 | 0.22 |

比例 =（低溫 − base）／（power − base），由 Table 1 計算〔本頁推導〕；分解，不是因果歸因。GPQA 題數（×198，四捨五入）：Qwen2.5-Math base 55／低溫 70／power 77／GRPO 79；Qwen2.5-7B 55／60／63／70；Phi-3.5 54／58／72／71。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 13.5–14.5 倍與 8.84 倍並陳 | 讀者可能誤以為論文算錯；實際是近似式在 K 小時偏低 | S7 先寫論文 8.84 倍，再以〔本頁推導〕寫精確和與差異來源（K²/2 對 K(K+1)/2） |
| R2 | Z_t 化簡的範圍 | 只在 $p_{\text{prop}}$ 與目標同一模型、未截斷且 τ = 1/α 時成立；論文未說明是否截斷 | S5b、C3a 在 τ ≠ 1/α 時明示「化簡不適用」；截斷與 AlpacaEval 例外寫入常見錯誤與 S9 |
| R3 | 玩具模型被讀成論文證據 | 低溫在玩具中比 p^α 更集中，可能被解讀成一般規律 | C3 首行假設句；觀察只寫「TV 隨 N_MCMC 下降，速度依設定而異」；不排序 B；不對哪種分佈較好下結論 |
| R4 | 差值分解被讀成因果 | 單次執行、無誤差棒、GPQA 分母只有數題；依賴 Base 與 Low-temperature 列的溫度（論文未寫）；power 的提議本身就是 τ = 1/α 取樣 | S8 稱「表中差值的占比」，寫明非因果歸因，附 GPQA 題數與溫度未說明註記；AlpacaEval 明寫不適用 |
| R5 | Fig. 5 的 k=1 值 | 圖的數值未給，論文未說明 Fig. 5 的 k=1 與 Table 1 是否同一設定 | E5 寫「依 Table 1；Fig. 5 數值本頁未讀出」，並以圖說自身「high k matches the base model」指出對 base 的過度宣稱 |
| R6 | 不可重現的圖表 | Fig. 4–9 需要 7B 模型與完整基準 | 只文字描述，不估讀座標、不畫示意曲線 |
| R7 | C3 計算量 | V 3、B 6 時矩陣 729²、約 80 萬個非零項；N 清單合計 91 次矩陣向量乘 × 3 區塊 | Worker；`reqId` 丟棄過期結果；若單次 > 500ms 顯示進度文字 |
| R8 | Worker 不可用 | Blob worker 在少數環境被禁止 | 主執行緒分段退路（5.3） |
| R9 | 勘誤措辭 | 五項勘誤應寫成事實陳述，不評價作者 | callout 只寫「論文寫…；依…應為…」 |
| R10 | 相對連結 | 建置位置與上線位置不同 | 以上線位置撰寫；預覽只檢查外部連結與錨點 |
| R11 | 筆記更新 | notes 若增減，`TL_GROUP` 與延伸閱讀需同步 | 建置時重讀 `notes/`，未分類 ID 歸「其他」並回報 |
| R12 | base model 一詞 | Phi-3.5-mini-instruct 已經過後訓練，論文仍稱 base model | 符號表、S1a、Table 1 註記；hero 卡用「作為取樣對象、不再訓練的既有模型」 |

---

## 8. 修訂紀錄（round 1）

審查合併與逐條處理見 `reviews/round1.md`（P0 2、P1 16、P2 10；第 14 條部分接受）。

| round1 # | 變更位置 | 內容 |
| --- | --- | --- |
| 1 | 3、4、5.3、6 | S5 拆為 S5a／S5b；單步面板獨立為 C3a（S5b 末）；C3 只留 Alg. 1 輸出分佈；LO6→S5a、LO7→S5b |
| 2 | 3（hero） | 三張卡重寫，不出現 MH、GRPO、8.84；卡 3 寫「重新生成到目前區塊結尾」；副標改為不用術語 |
| 3 | 1.4、步驟 10 表、3 | base model 定義；Phi Base 列註「instruct 版，已後訓練」；R12 |
| 4 | 步驟 6、LO6、LO7、S9 | 支撐／雙向正機率、非週期來自自環、Z_t 化簡需未截斷提議，皆〔本頁推導〕；S9 加「提議是否截斷」 |
| 5 | 步驟 10、LO10、6、R4 | 「表中差值的占比」，非因果；GPQA 題數 |
| 6 | 步驟 8、LO9、C4 | 近似 = (T/B)²/2 = 6.26；精確 K(K+1)/2 = 10 |
| 7 | C3、LO8、R3 | TV 對 B 不單調；觀察只寫隨 N 下降、速度依設定而異 |
| 8 | 步驟 10、1.6 E5、LO11、R5 | E5 補「high k matches base」矛盾與「Fig. 5 數值未讀出」 |
| 9 | 3 | S1 拆 S1a／S1b；pass@k 移到 S8 定義 |
| 10 | 3、2 | S3 拆 S3a（Example 1 + C1）／S3b（Eq. 7/8、F、E3）；LO3、LO4 對調使順序一致 |
| 11 | 3、C2 | Eq. 14 先代入 Example 1；C2 預設只露 ε、N，ε′、α 收進「進階」 |
| 12 | 3（S6）、1.4 | TV 在 S6 內文定義 |
| 13 | 5.5、全文 | 用詞規則；文件與元件中「偏好／輸給／勝出／困在／規劃」改寫 |
| 14 | 1.4、3 | x_0／T 加註（部分接受，保留論文記號） |
| 15 | C1、5.3 | p(ba)=p(bb) 條件；列和 0 時 F=1；selfTest 檢查 |
| 16 | C3 | worker key 快取（LRU 8）、`{key,nList}`、150ms debounce、耗時預算與進度 |
| 17 | 5.3 | selfTest 比對 worker 與主執行緒 TV；不得壓縮 `toyCore` |
| 18 | 4 共通、5.2、5.5 | 動態讀數 Unicode、KaTeX 只在答案與勘誤；捲動陰影；Eq. 7/8 分開；`<dl>` 符號表 |
| 19–20 | C2 | 約束標為空前綴情形；讀數改「只在兩者之間選時」 |
| 21 | 步驟 10、步驟 11 | N_MCMC 掃描基準未明寫，加入 S9 |
| 22 | 步驟 10 | Fig. 4 likelihood 與 confidence 分開 |
| 23 | 步驟 10 | A.2 主張記為作者判斷；「until k = 16」照錄 |
| 24 | 步驟 6 | 推論 (b) 改「因共同前綴的 Z_t 相消」 |
| 25 | 步驟 3、1.6 | E3「對一般 p 而言」 |
| 26 | 4 共通、C3a | radiogroup＋方向鍵；標記 ≥10px；按鈕 announce；aria-label 只在 change 更新 |
| 27 | 4 共通、5.2 | 計算中 opacity 無 transition；深淺色 token 與 pattern stroke |
| 28 | verify | `check_toy.py` 主程式輸出 C3a 預設軌跡；`check_closed.py` 加 GPQA 題數與 0^α 檢查；兩支皆重跑完成 |

## 9. 修訂紀錄（readiness 後）

- 阻斷項：C1 移除 F(a)／F(b) 讀數（F 在 S3b 才定義），S3b 內文以 Example 1 靜態回驗；`ex1` 仍回傳 F 供 selfTest。
- C3：註明提示 token $x_0$ 不進入玩具模型。
- C3a：寫明 $A=\min\{1,\exp(\Sigma)\}$ 與 Σ 的定義。
- S7／LO9：成本統一寫「約 13.5–14.5 倍」（13.48 截斷、14.14 精確和、14.48 含初始化）。


## 10. 建置紀錄（build round）

- 原始碼在 `build/src/`（head、css_base（取自 No.131）、css_add、body1–3、toy.js、app.js、assemble.py、lo.py）；`python3 build/src/assemble.py` 產生 `build/index.html`。assemble 會檢查 `$…$` 內沒有裸露的 `<`（TeX 一律寫 `\lt`）。
- 與設計的差異：
  - `?test` 模式不建立 C3 的 Worker，C3 改在主執行緒以同一份 `toyCore` 計算（`--virtual-time-budget` 不等待 Worker）；selfTest 另外檢查 Blob Worker 可以建立，並比對 key 快取路徑與完整路徑的結果相同。原設計的「worker 與主執行緒 TV 比對」改為此兩項。
  - C3 排行只畫前 10 條序列（比例尺依前 10 條），「其他」合計改為圖下一行文字，避免 0.79 的 base 合計壓縮比例尺。
  - C2 的 ε 標籤與區間標籤分層、刻度下移，避免與 ε′/N 重疊。
  - `.ch-num`、`.plate-h .id` 的 letter-spacing 由 .06em 改為 .04em（impeccable wide-tracking）。
- 驗證：`?test` 57 passed / 0 failed（1440 寬；390 寬以 iframe 執行，含無水平捲動）；LO 自檢 12/12（extract_page_text.py + KATEX_DIR）；impeccable 0 anti-patterns（numbered-section-labels advisory 保留）；無 console 錯誤。截圖在 `build/caps/`。
