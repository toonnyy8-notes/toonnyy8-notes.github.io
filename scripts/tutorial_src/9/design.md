# 設計文件：〈Large Flow Language Models〉（Lafa）互動教學

- 論文：Arnav Chavan、Chanhyuk Lee、Udbhav Bamba、Jaehoon Yoo、Seunghoon Hong、Aditi Raghunathan、Jinwoo Kim，arXiv 2610.28559（v1，2026-10-20）。資料庫編號 No. 9。
- 原文：`scripts/tutorial_src/9/src/paper.txt`。本文件的「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」「Alg. n」「Prop. X」皆指論文原文編號。理解備忘：`advisor.md`（本文件已逐條核對）。
- 產出：`scripts/tutorial_src/9/build/index.html`（單檔）。上線位置 `tutorials/large-flow-language-models.html`（slug `large-flow-language-models`），所有站內相對連結以上線位置為準；論文連結 `../index.html#paper-9`。
- 驗算腳本：`verify/check.py`（stdlib，`python3 -I check.py`；輸出存 `verify/out.txt`）。演算法、常數、PRNG、抽樣順序與頁面相同。第 5 節的表全部由它產生。
- 研究筆記：`notes/`（總覽 `lafa-lineage-overview` + 7 篇主題筆記 + `timeline.json`，14 筆）。
- 標記：本設計自行推導、論文未寫出的結論標「〔本頁推導〕」；為了玩具示範而選的數值標「示意值」（不是論文數字）。本論文沒有需要用論文數字反推的參數，因此沒有「推定」值。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

AR 語言模型一次 forward 只產生一個 token。Lafa 把續寫切成長度 $B$ 的 block：block 之間照 AR 順序（Eq. 1），block 內的 $B$ 個 token 在詞嵌入空間中從高斯噪聲沿一條連續的 flow 一起生成。每個位置有自己的時間 $\tau^i(t)$（tokenwise time，Eq. 2），由訓練中的 denoising loss 學出（Eq. 22）。模型只學一個逐位置的 token 分類器（block denoiser，Eq. 4），速度由 Eq. 5 換算。作者把預訓練 AR 模型（Qwen2.5 0.5B/7B Instruct、Qwen3 8B Base）改成 block denoiser：prefix 維持 causal 並固定時間 1，所以 prefix 的 KV cache 可以重用，prefix 輸出仍是 AR 的下一個 token 預測。訓練同時含 AR cross entropy、denoising cross entropy 與嵌入平方誤差（Eq. 6），用少於 30B tokens。推論有三種：多步 flow decoding（Eq. 8）、蒸餾後的 flow map（Eq. 7、9，最少一步），以及自我推測解碼（self-speculative decoding）：flow 一步產生草稿（Eq. 10），同一模型的 greedy AR 預測驗證（Eq. 28），輸出與該 checkpoint 的 greedy AR 輸出相同（邏輯上；數值上取決於實作，論文原話為「matches greedy AR decoding」）。7B、$B=16$ 時吞吐量為 AR SFT 的 2.54 倍（100.2 vs 39.5 TPS，三項平均準確率 74.5 vs 73.9，Table 8）。

### 1.2 研究動機（Sec. 1）

1. AR 生成每次 forward 一個 token，推論延遲是瓶頸；既有的平行生成有 MTP、離散擴散、連續 flow 三類。
2. 連續 flow 可沿用影像／影片的控制與加速技術（例如 flow map 少步生成、guidance），離散方法用不上；但既有 flow 語言模型只在小規模與簡單任務上展示，先前的擴大嘗試遇到訓練效率問題 [18]。
3. 文字有由左到右的方向性 [19]，作者因此讓架構與歸納偏差接近 AR，以便從預訓練 AR 模型調整，而不是從零訓練。

### 1.3 貢獻（Sec. 1 列點）

1. Lafa：blockwise 非同步 flow；可從預訓練 AR 模型調整，並以 prefix caching 與平行 flow 加速推論。
2. 可擴展的設計選擇：架構與非同步 flow 公式、訓練與 flow map 蒸餾目標、推論方式（含 self-speculative decoding）。
3. 7–8B、<30B tokens；flow 品質達到或超過 AR，可蒸餾成 flow map；flow 訓練提升 AR 能力（六項平均的提升，Sec. 3.2；部分項目下降，見步驟 12）。
4. 結合 flow 與 AR 的自我推測解碼：2.54× AR 吞吐量、品質相當。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S1 末尾以可收合表格提供同一份內容）

| 符號 | 意義 |
| --- | --- |
| $V$、$\lvert V\rvert$ | 詞彙表與其大小 |
| $c\in V^N$ | prefix（已知文字，長度 $N$） |
| $y\in V^M$ | 續寫（長度 $M$）；$B$ 為 block 大小，假設 $B\mid M$ |
| $\mathbf E\in\mathbb R^{\lvert V\rvert\times d}$、$\mathbf E_v$ | 詞嵌入矩陣與 token $v$ 的嵌入。**論文把嵌入矩陣與期望值都寫成 $E$；頁面一律用粗體 $\mathbf E$ 表示嵌入、$\mathbb E[\cdot]$ 表示期望值** |
| $x_1\in\mathbb R^{B\times d}$ | 一個 block 的乾淨嵌入 $(\mathbf E_{y_1},\dots,\mathbf E_{y_B})^\top$ |
| $x_0\sim\mathcal N(0,\sigma^2I)$ | 噪聲；$\sigma>0$ 為 noise scale |
| $t\in[0,1]$ | base time；$t=0$ 為噪聲，$t=1$ 為資料 |
| $\tau^i(t)$ | 位置 $i$ 的 tokenwise time；**上標 $i$ 是 block 內位置，不是次方** |
| $I_t$、$b_t$ | 內插（interpolant）與速度場（velocity field） |
| $D_t(x;c)^i$、$\hat D_t$ | 理想 block denoiser（位置 $i$ 的後驗分布）與學到的 denoiser |
| $\Delta^{\lvert V\rvert-1}$ | 長度 $\lvert V\rvert$ 的機率向量集合 |
| $f$、$f_t(x;c)$、$f(c)_i$ | 預訓練 AR transformer、加上時間條件的模型、prefix 位置 $i$ 的 AR 預測（預測 $c_{i+1}$；$i=N$ 時預測 $y_1$） |
| $\ell^i(\tau)$、$F_i$ | 位置 $i$ 在 tokenwise time $\tau$ 的期望 denoising CE、其正規化累積分布 |
| $M_{\text{bin}}$ | Alg. 2 的時間格數（論文寫 $M=200$，與續寫長度 $M$ 同名；頁面改寫為 $M_{\text{bin}}$） |
| $K$、$t_k=k/K$、$p_k$ | Euler 步數、base time 網格、第 $k$ 步的 denoiser 輸出 |
| $X_{s,t}$、$\delta_{s,t}$ | flow map（由時間 $s$ 直接送到 $t$）與 two-time denoiser |
| $\gamma^i$、$w^i(r)$、$Z^i_{s,t}$ | semigroup 權重、Eq. 17 的權重函數與其積分 |
| $g$ | prefix guidance scale |
| $\bar L_{i,m}$ | Alg. 2 本輪四次無梯度 forward 中，落在位置 $i$、第 $m$ 格的 denoising CE 平均（Eq. 23） |
| $\mathrm{sg}[\cdot]$ | stop-gradient：前向照算、反向不傳梯度（Eq. 19、24） |
| $a$、TPF、TPS | 一次 fused forward 接受的 token 數、tokens per forward、tokens per second |

#### 步驟 1：Block-AR 分解（Sec. 2.1，Eq. 1）

$$p(y\mid c)=\prod_{r=0}^{M/B-1}p\big(y_{rB+1:(r+1)B}\mid c,\,y_{\le rB}\big)\tag{Eq. 1}$$

- 這是機率的連鎖律，沒有近似。沿用 block-autoregressive 分解 [22] 與 block discrete diffusion [23]。
- 作用：block 之間維持 causal，前面的 token 不會因後面的 block 改變，所以 prefix 的 KV cache 可以沿用；平行只發生在 block 內。
- 頁面 S1 先用 2–3 句說明 AR 基礎：next-token 分布、greedy decoding（每步取機率最大的 token）、一次 forward 一個 token、KV cache（已處理位置的 key/value 存起來，只要這些位置的表示不隨後續輸入改變就能重用）。

#### 步驟 2：在詞嵌入空間做 flow（Sec. 2.1 Embedded flows）

- $x_1:=(\mathbf E_{y_1},\dots,\mathbf E_{y_B})^\top\in\mathbb R^{B\times d}$；每個 token 的維度由 one-hot 的 $\lvert V\rvert$（[13] 的做法）降到 $d$。這是設計選擇，論文寫「making training more scalable」。
- 用 flow matching [24, 25] 建構條件機率路徑 $p_t(\cdot\mid c)$，$x_0\sim\mathcal N(0,\sigma^2I)$。標準內插 $I_t=(1-t)x_0+tx_1$，所有 token 共用 $t$。
- 頁面說明 flow matching 的一句話定義：速度 $b_t(x)=\mathbb E[\dot I_t\mid I_t=x]$；從 $x_0$ 沿 ODE $\dot x_t=b_t(x_t)$ 積分到 $t=1$，終點的分布就是資料分布。

#### 步驟 3：非同步（tokenwise）時間（Eq. 2–3）

- tokenwise time $\tau:[0,1]\to[0,1]^B$，每個分量可微且遞增，$\tau(0)=0$、$\tau(1)=1$（Sec. 2.1；App. A.2 的證明另要求 $\tau^i\in C^1$、$t<1$ 時 $\dot\tau^i>0$）。
- 動機（Sec. 2.1）：「To introduce a left-to-right bias within each block to match the inherent directionality of text [19], we use tokenwise time [26–30], allowing earlier positions to progress faster.」先例 [26–30] 含 Learning When to Denoise（2606.19662，影像的多表示擴散）、Token Time Continuous Diffusion、AR-Diffusion（依筆記 `lafa-asynchronous-tokenwise-time`；後兩者 arXiv ID 未驗證，頁面只寫名稱）。相近想法：Diffusion Forcing（站內 `diffusion-forcing.html`）。注意：動機是由左到右，但 $\tau$ 的實際形狀由 loss 學出（步驟 8），論文沒有把 $\tau^i>\tau^j$ 設成約束。

$$I_t^i:=(1-\tau^i(t))\,x_0^i+\tau^i(t)\,x_1^i,\quad i\in[B],\quad I_t\sim p_t(\cdot\mid c)\tag{Eq. 2}$$
$$\dot x_t=b_t(x_t;c),\quad x_0\sim p_0,\quad x_t\sim p_t(\cdot\mid c)\tag{Eq. 3}$$

- 速度場 $b_t(\cdot;c)$ 的存在引用 [27]，本文不證。若 $\tau^i(t)>\tau^j(t)$（$i<j$），較前的位置先被還原。實際上 $\tau$ 由資料學出（步驟 8）。

#### 步驟 4：Denoiser 與速度恆等式（Eq. 4–5，Prop. A.1，Eq. 11–12）

$$D_t(x;c)^i:=p(y_i=\cdot\mid c,\,I_t=x)\in\Delta^{\lvert V\rvert-1}\tag{Eq. 4}$$
$$b_t(x;c)^i=\frac{\dot\tau^i(t)}{1-\tau^i(t)}\big(D_t(x;c)^i\mathbf E-x^i\big),\quad t\in[0,1)\tag{Eq. 5}$$

推導（頁面完整寫出，Prop. A.1）：
1. $\dot I_t^i=\dot\tau^i(t)(x_1^i-x_0^i)$（Eq. 11）。
2. 由 Eq. 2 解出噪聲：$x_0^i=(I_t^i-\tau^ix_1^i)/(1-\tau^i)$，所以 $x_1^i-x_0^i=(x_1^i-I_t^i)/(1-\tau^i)$。
3. 取條件期望 $b_t(x;c)=\mathbb E[\dot I_t\mid c,I_t=x]$：$b^i=\frac{\dot\tau^i}{1-\tau^i}(\mathbb E[x_1^i\mid c,I_t=x]-x^i)$。
4. $x_1^i=\mathbf E_{y_i}$，所以 $\mathbb E[x_1^i\mid c,I_t=x]=\sum_v p(y_i=v\mid c,I_t=x)\mathbf E_v=D_t(x;c)^i\mathbf E$（Eq. 12）。

重點：條件是整個 block 的 $x$，所以位置 $i$ 的速度透過 $D$ 依賴其他位置；但只需要每個位置的**邊際**後驗，不需要聯合分布。係數只用該位置自己的 $\tau^i$。$t\to1$ 時係數 $1/(1-\tau^i)$ 發散，所以命題限定 $t\in[0,1)$。「機率向量乘 $\mathbf E$ = 期望嵌入」在頁面以一行例子說明。

#### 步驟 5：Euler 步、一步生成與邊際問題（Sec. 2.3，Eq. 8、10）

$$\hat x^i_{t_{k+1}}=\hat x^i_{t_k}+\frac{\tau^i(t_{k+1})-\tau^i(t_k)}{1-\tau^i(t_k)}\big(p^i_k\mathbf E-\hat x^i_{t_k}\big),\qquad \hat y_i=\arg\max(p^i_{K-1})\tag{Eq. 8}$$

$p_k:=\hat D_{t_k}(\hat x_{t_k};c)$，$t_k=k/K$。生成的 token 是最後一次 denoiser 預測的 argmax，**不是**把 $\hat x_1$ 找最近的嵌入。
- 〔本頁推導〕Eq. 8 是 tokenwise time 上的 Euler 步，等於「假設 denoiser 輸出在這一步內不變」時 ODE 的精確解：改寫為 $\hat x_{k+1}=\frac{1-\tau_{k+1}}{1-\tau_k}\hat x_k+\frac{\tau_{k+1}-\tau_k}{1-\tau_k}p_k\mathbf E$，與 Eq. 7 同形（論文在 Eq. 9 前也說 flow map 是把 $\hat D_{t_k}$ 換成 $\hat\delta_{t_k,t_{k+1}}$）。它不是 base time 的 Euler $x+\Delta t\cdot b$。後果：最後一步 $\tau=1$ 時 $\hat x=p\mathbf E$，不會因 $1/(1-\tau)$ 發散。
- $K=1$：因 $\tau^i(0)=0$、$\tau^i(1)=1$，退化成一次 denoiser 計算（Eq. 10）：$p^i=\hat D_0(\hat x_0;c)^i$，$\hat x_1^i=p^i\mathbf E$，$\hat y_i=\arg\max p^i$。
- 〔本頁推導〕$t=0$ 時 $I_0=x_0$ 與 $y$ 獨立，理想的 $D_0(x_0;c)^i=p(y_i\mid c)$：各位置獨立的邊際，噪聲不帶資訊。各位置分別取 argmax 可能組出聯合機率為 0 的序列。論文的說法：「One-step drafts can be inconsistent across positions」（Sec. 2.3）。C1 以玩具示範：$p(\text{hot dog})=0.4$、$p(\text{ice cream})=0.3$、$p(\text{sour cream})=0.3$，位置 1 的 argmax 是 hot（0.4），位置 2 的 argmax 是 cream（0.6），一步得到機率 0 的「hot cream」。

#### 步驟 6：把預訓練 AR 模型改成 block denoiser（Sec. 2.2）

1. **Tokenwise time conditioning**：輸入為嵌入後的 prefix $c$ 接 noisy block $x$；每個位置以 adaLN [31]（依時間縮放與平移每層的特徵）接收自己的時間；頁面只在 S4 出現 adaLN。prefix 恆為時間 1（無噪聲），block 位置 $i$ 為 $\tau^i(t)$。adaLN 初始化為恆等，加入時間條件時不改變預訓練計算。記為 $f_t(x;c)$。
2. **Block-bidirectional attention**（Fig. 3）：block 內雙向；block 位置可看整個 prefix；prefix 維持 causal 且看不到 noisy block [32, 33]。加上 prefix 時間固定為 1，prefix 的 key/value 與 block 狀態 $(x,t)$ 無關，可跨 denoising 步與後續 block 重用。prefix 輸出保留 AR 角色：$f_t(x;c)_i$ 預測 $c_{i+1}$（$i<N$），$i=N$ 預測 $y_1$；記為 $f(c)_i$。雙向優於 causal（Table 3：38.0 vs 32.7）。
3. **共享且受約束的嵌入**：$\mathbf E$ 由預訓練模型初始化，prefix 與 flow 共用，與 denoiser 一起訓練；但每個 $\mathbf E_v$ 的 L2 norm 固定為預訓練值，只學方向。理由：不受約束時嵌入 norm 會相對固定的 $\sigma$ 膨脹，降低有效 corruption，使 denoising 變容易 [12, 13]。Table 3：norm-frozen 38.0、unconstrained 36.9、frozen 34.4。Alg. 1 在每次更新後「Project each $\mathbf E_v$ to its pretrained norm」。
4. **平行訓練**（Fig. 4、Alg. 1、App. B.1）：每條乾淨序列配一份切成 noisy block 的複本。乾淨位置 causal；第 $r$ 個 noisy block 只看它之前的乾淨 token 與自己（block 內雙向），看不到後面的乾淨 token 或其他 noisy block。一次 forward 同時產生乾淨序列的 AR 預測與每個 block 的 denoising 預測。每個 block 各自抽 $t_r\sim U[0,1]$。

#### 步驟 7：訓練目標（Eq. 6）

$$\mathcal L(\hat D):=\mathbb E_{c,y,t,x_0}\Big[\underbrace{\sum_{i=1}^{N}-\log f(c)^i_{c_{i+1}}}_{\text{AR CE}}+\underbrace{\sum_{i=1}^{B}-\log\hat D_t(I_t;c)^i_{y_i}}_{\text{denoising CE}}+\underbrace{\sum_{i=1}^{B}\big\lVert\hat D_t(I_t;c)^i\mathbf E-\mathbf E_{y_i}\big\rVert^2}_{\text{嵌入平方誤差}}\Big]\tag{Eq. 6}$$

- $\hat D_t(x;c)^i:=f_t(x;c)_{N+i}$；$c_{N+1}:=y_1$；$t\sim U[0,1]$，$I_t$ 依 Eq. 2，$x_1^i=\mathbf E_{y_i}$。三項權重皆為 1，共用 transformer 參數。
- denoising CE 監督 Eq. 4 的機率；平方誤差監督決定速度的期望嵌入（Eq. 5）。這是經驗設計：Table 3（0.5B、$B=4$、32 步、三項平均）CE 33.6、MSE 29.7、CE & MSE 35.8、再加 AR 38.0。作者寫 AR 監督「unexpectedly」進一步改善 flow。

#### 步驟 8：學出 tokenwise time（Sec. 2.2，App. B.1，Eq. 22–23，Alg. 2）

$$\tau^i(t):=F_i^{-1}(t),\qquad F_i(\tau):=\frac{\int_0^\tau\ell^i(r)\,dr}{\int_0^1\ell^i(r)\,dr}\tag{Eq. 22}$$

- $\ell^i(\tau)>0$：位置 $i$ 在 tokenwise time $\tau$ 的期望 denoising CE。靈感來自 CDCD [12] 的 loss-aware time warping，逐位置分開做。
- 論文：「$\dot\tau^i\propto1/\ell^i(\tau^i)$ is small where the loss is high」，所以均勻的 $t$ 取樣（訓練）與均勻的 $t$ 網格（推論）都集中在最難的噪聲程度。〔本頁推導〕對 $F_i(\tau^i(t))=t$ 微分得 $\dot\tau^i(t)=\big(\int_0^1\ell^i\big)/\ell^i(\tau^i(t))$；等價地，$t\sim U[0,1]$ 時 $\tau^i(t)$ 的密度正比於 $\ell^i(\tau)$。$\ell^i>0$ 保證 $F_i$ 嚴格遞增、$\tau^i(0)=0$、$\tau^i(1)=1$。
- 估計（Alg. 2）：$L\in\mathbb R^{B\times M_{\text{bin}}}$，$M_{\text{bin}}=200$ 個等寬格，均勻初始化；每 $S=100$ 步做四次無梯度 forward，每個 noisy block 獨立抽均勻時間；$L_{i,m}\leftarrow\beta L_{i,m}+(1-\beta)\bar L_{i,m}$，$\beta=0.99$（Eq. 23），沒被訪問的格保留原值；$F_i$ 為正規化列 $L_{i,:}$ 的分段線性 CDF。每個 noisy block 各自獨立抽均勻時間；論文對這一點的理由是「This independent uniform estimation prevents the learned sampling distribution from starving low-probability regions」（App. B.1，Eq. 23 後）。另外，Alg. 2 的虛擬碼用 synced interpolant $(1-t)x_0+tx_1$（block 內共用 $t$）估 $\ell^i$；論文**沒有**說明為何不用訓練時的非同步內插（R7）。
- 結果：學到的時間「generally satisfy $\tau^i(t)>\tau^j(t)$ for $i<j$」（Fig. 5，Lafa 7B，16 個位置），由左到右的偏差不是人工設定的。論文沒有給機制解釋，也沒有 Fig. 5 的數值。Table 3：同步時間 36.2、非同步 38.0。

#### 步驟 9：Self-conditioning 與 prefix guidance（Sec. 2.2–2.3，App. B.2，Eq. 24–27，Alg. 3）

- Self-conditioning [37]：訓練時以機率 $p_{sc}=0.5$ 先做一次無梯度預測，再把期望嵌入回饋到第二次 forward：$\tilde I_t=I_t+I_tW_1^\top+\bar x_1W_2^\top$，$\bar x_1=\mathrm{sg}(D_t(I_t;c)\mathbf E)$，$W_1,W_2\in\mathbb R^{d\times d}$ 初始化為 0（Eq. 24，依 [15]）；loss 算在第二次。推論時每個 Euler 步沿用前一步的預測（Eq. 25：$\tilde x_{t_k}=\hat x_{t_k}+\hat x_{t_k}W_1^\top+(q_{k-1}\mathbf E)W_2^\top$，$q_k=D_{t_k}(\tilde x_{t_k};c)$，$q_0=D_0(\hat x_0;c)$），不增加 forward。Table 3：無 32.3、有 38.1。
- Prefix guidance（類比 classifier-free guidance [40] 與 history guidance [39]）：$D^g_t(x;c):=D_t(x;\varnothing)+g\big(D_t(x;c)-D_t(x;\varnothing)\big)$，$g\ge1$（Eq. 27），$g=8.0$（Table 4）。每步多一次不含 prefix 的 forward，只處理 $B$ 個 block token。Table 3：無 34.3、有 38.1。Alg. 3 在嵌入空間做 $\tilde x=s'\mathbf E+g(s\mathbf E-s'\mathbf E)$；最後的 token 取未 guided 的條件預測 $s$ 的 argmax——這只出現在 Alg. 3 的 $\hat y\leftarrow\arg\max(s_j)$，正文（Eq. 8、Sec. 2.3）沒有寫，頁面引用時只標 Alg. 3。〔本頁推導〕$g>1$ 時 $D^g$ 可能有負值，不是機率向量；這不影響 Alg. 3，因為只用到它的嵌入。
- 頁面不做元件；文字加兩個小式子。LO8 只由正文達成（沒有元件）。延伸：[36]（Self-conditioned Flow Map Language Models via Fixed-point Flows，2607.00714）為論文引用的 self-conditioning 理論，依筆記 `lafa-self-conditioning-and-flow-map` 只寫一句、連到站內 `self-conditioning-flow-lm.html`。

#### 步驟 10：Flow map 蒸餾（Sec. 2.2，App. A.2，Eq. 7、13–21，Prop. A.2–A.3）

$$X_{s,t}(x;c)^i:=\frac{1-\tau^i(t)}{1-\tau^i(s)}x^i+\frac{\tau^i(t)-\tau^i(s)}{1-\tau^i(s)}\,\delta_{s,t}(x;c)^i\mathbf E,\qquad 0\le s<t\le1\tag{Eq. 7}$$

- 動機：flow decoding 每個 block 要多次 denoiser 計算。flow map $X_{s,t}$ 直接把時間 $s$ 的狀態送到 $t$ [38]。作者把 simplex-valued two-time denoiser [13] 推廣到詞嵌入上的非同步 flow：$\delta_{s,t}$ 預測 token 機率，經 $\mathbf E$ 後與 $x$ 內插。
- 設定（App. A.2）：$c$、$\mathbf E$、$\tau$ 固定；$\tau^i\in C^1$、$\dot\tau^i>0$（$t<1$）；ODE 解唯一；denoiser 對時間連續可微；在 $0\le s\le t<1$ 上討論，$t=1$ 以極限理解。teacher 的速度與 flow map：Eq. 13（$\partial_tX_{s,t}=b_t(X_{s,t})$，$X_{s,s}=x$）。平均速度 $v_{s,t}:=(X_{s,t}-x)/(t-s)$（Eq. 14）；$\delta$ 由其期望嵌入定義：$\delta_{s,t}\mathbf E=x+\frac{(1-\tau_s)(t-s)}{\tau_t-\tau_s}v_{s,t}$（Eq. 15，代入 Eq. 14 即 Eq. 7）。因 $\lvert V\rvert\gg d$，Eq. 15 只固定期望嵌入，不固定 $\delta$ 本身。
- Prop. A.2 的構造（頁面以可收合推導呈現）：
  1. $\frac{d}{dr}\frac{X_{s,r}^i}{1-\tau^i_r}=w^i(r)\hat D_r(X_{s,r})^i\mathbf E$，$w^i(r):=\dot\tau^i(r)/(1-\tau^i_r)^2$（Eq. 17）。
  2. $Z^i_{s,t}:=\int_s^tw^i=\frac{\tau^i_t-\tau^i_s}{(1-\tau^i_t)(1-\tau^i_s)}$，$\delta^i_{s,t}:=\frac1{Z^i_{s,t}}\int_s^tw^i(r)\hat D_r(X_{s,r})^idr$（Eq. 18）：機率向量的加權平均，仍是機率向量。
  3. 對 Eq. 17 在 $[s,t]$ 積分再乘 $1-\tau^i_t$ 得 Eq. 7。（原文寫「which is (15)」，兩者等價。）
  4. Diagonal：$t\to s$ 時 $w/Z$ 集中在 $r=s$，$\delta_{s,t}\to\hat D_s$。
  5. Semigroup（$s<u<t$）：在 $u$ 切開積分，$r\ge u$ 時 $X_{s,r}=X_{u,r}(X_{s,u})$，

$$\delta_{s,t}^i=\gamma^i\delta^i_{s,u}+(1-\gamma^i)\,\delta_{u,t}(X_{s,u})^i,\qquad \gamma^i=\frac{Z^i_{s,u}}{Z^i_{s,t}}=\frac{(1-\tau^i_t)(\tau^i_u-\tau^i_s)}{(1-\tau^i_u)(\tau^i_t-\tau^i_s)}\tag{Eq. 16}$$

  6. 反方向：任何滿足 diagonal 與 semigroup 的 $\delta$，經 Eq. 7 得到的就是真正的 flow map（$X_{s,s}=\mathrm{Id}$、切線條件、semigroup [38]）。App. A.2 後段把 Prop. A.2 稱為「Theorem A.2」，是同一命題。
- 〔本頁推導〕$t\to1$ 時 $Z_{s,t}\to\infty$、$\gamma^i\to0$，權重集中在軌跡終點，所以 $\delta_{0,1}(x_0)$ 是 ODE 終點 token 的 one-hot，$X_{0,1}(x_0)$ 是一步得到的、與多步積分相同的聯合結果。對照步驟 5：$\hat D_0$ 是各位置的邊際。這是一步 flow map 能保留位置間關聯、一步 denoiser 不能的原因。以上是對理想的 teacher $\delta$；學到的 $\hat\delta$ 仍有差距（Table 5：7B 一步 flow map 49.0，多步 flow 75.9）。
- 〔本頁推導〕離散版：把 Eq. 8 的更新兩邊除以 $1-\tau_{k+1}$，得 $\zeta_{k+1}\hat x_{k+1}=\zeta_k\hat x_k+(\zeta_{k+1}-\zeta_k)p_k\mathbf E$，$\zeta:=1/(1-\tau)$。所以在 Eq. 8 的離散軌跡上，Eq. 7、Eq. 18 與 Eq. 16 在代數上恆等成立（$\delta$ 是以 $\Delta\zeta$ 加權的 $p_k$ 平均），殘差只剩浮點誤差。這**不是** Prop. A.2 的數值驗證：參考的「flow map」就是離散軌跡本身；$n_f$ 增大時離散軌跡收斂到 ODE 的解。C4 用這個形式計算，避免數值微分 $\dot\tau$。
- 蒸餾目標：student $\hat\delta_{s,t}$ 由凍結的 teacher $\hat D_s$ 初始化，加一個恆等初始化的 adaLN 接收目的時間 $t$。off-diagonal 目標組合兩段較短的 student 轉移：$\bar\delta_{s,t}(x_s)^i:=\mathrm{sg}[\gamma^i\hat\delta_{s,u}(x_s)^i+(1-\gamma^i)\hat\delta_{u,t}(x_u)^i]$，$x_u:=\hat X_{s,u}(x_s)$（Eq. 19）；$0\le\gamma^i\le1$，目標仍是類別分布。

$$\mathcal L_{\text{distill}}:=\mathbb E_c\,\mathrm{KL}(f(c)\Vert f^{\hat\delta}(c))+\mathbb E_{t,c,I_t}\mathrm{KL}(\hat D_t(I_t;c)\Vert\hat\delta_{t,t}(I_t;c))+\mathbb E_{s,u,t,c,I_s}\mathrm{KL}(\bar\delta_{s,t}(I_s;c)\Vert\hat\delta_{s,t}(I_s;c))\tag{Eq. 20}$$

  三項依序為 AR 蒸餾、diagonal 蒸餾、off-diagonal 蒸餾。Prop. A.3：在 $(s,u,t)$ 與狀態取樣 full support 下，Eq. 21（後兩項）的母體最小值為 0，且每個最小解經 Eq. 7 都還原 teacher 的 flow map；AR 項只依賴 prefix 讀出，不改變最小解。
- 推論（Eq. 9）：$\hat x_{t_{k+1}}=\hat X_{t_k,t_{k+1}}(\hat x_{t_k};c)$，$\hat y_i=\arg\max\hat\delta_{t_{K-1},1}(\hat x_{t_{K-1}};c)^i$；$K=1$ 即 $\hat X_{0,1}$ 一步生成。student 保留 prefix cache 與 AR 預測 $f^{\hat\delta}$。
- 論文未說明：$(s,u,t)$ 取樣分布、蒸餾步數與 token 數、學習率、KL 權重以外的超參數；只蒸餾 $B=4$。作者把 self-conditioning 與 prefix guidance 納入 flow map 列為未來工作（Sec. 3.1、Sec. 5）。

#### 步驟 11：自我推測解碼（Sec. 2.3，App. B.3，Eq. 10、28–29，Alg. 4，Fig. 9）

- 草稿：一步 Euler（Eq. 10），不用 prefix guidance。
- 驗證：commit 與 greedy AR 預測一致的最長草稿前綴；第一個 token 不一致時改 commit $\arg\max f(c)_N$。第一個 token 可免費驗證（與已有的 $q_N$ 比）。其餘：

$$a=\max\{\,j\in[B]:\ \hat y_{i+1}=\arg\max f(c\oplus\hat y)_{N+i}\ \ \forall\,1\le i<j\,\}\tag{Eq. 28}$$

  $j=1$ 的條件是空的，所以 $a\ge1$。
- 為什麼輸出等於 greedy AR〔本頁推導：補上歸納步驟，論文只陳述結論〕：被接受的 $\hat y_{i+1}$ 等於以 $c\oplus\hat y_{1:i}$ 為 prefix 的 greedy AR 預測；prefix 為 causal，$f(c\oplus\hat y)_{N+i}$ 只依賴 $c\oplus\hat y_{1:i}$；而 $\hat y_{1:i}$ 已歸納地等於 greedy AR 輸出。所以 2.54× 的品質就是該 checkpoint 的 greedy AR 品質，flow 只當草稿器（邏輯上；數值上取決於實作，例如 bfloat16 下 fused forward 與逐字 forward 的 batch 形狀不同，logits 可能有微小差異並影響同值處理）。
- 背景（Sec. 4，筆記 `lafa-parallel-decoding-background`）：「Many parallel decoders predict tokens independently within an update, which can degrade」品質；部分方法以 AR 驗證草稿緩解。Lafa 也用 AR 驗證，差別在草稿來自同一模型的 flow。筆記 `lafa-self-speculative-decoding` 另寫「非同步時間使左側 token 先成形，與由左到右的驗證順序一致」，論文沒有這樣連結（只寫「accepts flow-based drafts from left to right」，Sec. 1），頁面不採用。
- Fused forward（Alg. 4、Fig. 9）：輸入 = 乾淨 prefix $c\oplus\hat y$（時間 1）+ $B$ 個噪聲 block $z^{(1)},\dots,z^{(B)}\in\mathbb R^{B\times d}$（時間 0，各自 $\sim\mathcal N(0,\sigma^2I)$）。$c\oplus\hat y$ causal；$z^{(j)}$ 看 $c\oplus\hat y_{1:j}$、block 內雙向、block 之間互不可見。驗證後取 $z^{(a)}$ 的輸出當下一份草稿（計算上等同 $D_0(z^{(a)};c\oplus\hat y_{1:a})$），沿用 $f(c\oplus\hat y)_{1:N+a}$ 與 KV cache。第一個 token 被修正時 $A=\{1\}$，只需一個噪聲 block。
- 〔本頁推導〕每次 fused forward commit $1\le a\le B$ 個 token；新處理的位置數為 $B+B^2$（修正時 $1+B$），$B=16$ 時 272。設計者推測這是 TPF 4.9 只換到 2.54× TPS 的原因之一；論文沒有這個計數，只寫「forward-pass costs differ」。頁面以「本頁推測」標示。
- TPF $=\sum_jT_j/\sum_jF_j$（Eq. 29），$T_j$ 為 commit 的 token 數，$F_j$ 為 fused forward 次數加 1 次初始草稿（不含 prompt prefill）。AR 的 TPF 定義為 1。
- 替代方案（App. C.3）：confidence thresholding，接受信心 $\ge\lambda$ 的最長前綴，$a\ge1$；已知 $a$ 時只需 $z^{(a)}$。論文以 self-speculation 為預設，理由是它不需調門檻、低門檻時不退化。

#### 步驟 12：實驗（Sec. 3，App. C，Table 1–8，Fig. 6–8）

- 模型與資料：0.5B、7B 由 Qwen2.5 Instruct 調整；8B 用 Qwen3 8B Base（論文：其 Instruct 版再訓練會退步）。資料 Dolci-Instruct-SFT，Sec. 3 寫 5–15 epochs（9–27B tokens），Table 4 寫 5 epochs／約 9B（見 R2）。「at least 18× fewer」（Sec. 3）是相對 Dream、LLaDA、Efficient-DLM 的「reported budgets」；「up to 500B tokens」（Sec. 1）引用 [20, 21]，即 Dream 與 Efficient-DLM，論文沒有給 LLaDA 的預算。0.5B 做消融、7B 主結果、8B 對比更多基準。
- 基準：標 † 者同初始化、資料、預算；其餘用釋出的 checkpoint。AR SFT 是同資料微調的 AR，作為品質與速度參考。六項 benchmark：IFEval（0-shot，prompt-level strict）、BBH（3-shot）、MMLU-Pro、GPQA、GSM8K、HumanEval（pass@1），皆 0-shot 除 BBH；生成上限 1024 tokens（GSM8K 512、HumanEval 640）。single-token 準確率、self-speculative 準確率與 TPF 用六項平均；flow／flow map、吞吐量、消融用 IFEval、GSM8K、HumanEval 三項平均。
- Flow 與 flow map（Fig. 6、Table 5，三項平均）：

  | 設定 | 0.5B | 7B |
  | --- | --- | --- |
  | AR SFT | 38.6 | 73.9 |
  | Flow $B=4$：1 / 2 / 4 / 8 步 | 20.8 / 19.7 / 30.0 / 31.8 | 43.7 / 46.2 / 64.3 / 71.2 |
  | Flow $B=4$：32 / 64 / 128 步 | 38.1 / 38.5 / 37.1 | 75.2 / 75.9 / 75.7 |
  | Flow $B=16$：32 / 128 步 | 29.1 / 31.0 | 66.5 / 67.2 |
  | Flow map $B=4$：1 / 2 / 4 / 8 步 | 23.3 / 25.2 / 28.5 / 31.8 | 49.0 / 63.3 / 69.4 / 71.0 |

  論文結論：步數多時一般變好；$B=4$ 在 0.5B 接近、7B 超過 AR SFT；同步數下 block 越大越差；flow map 改善 1–2 步，7B 兩步 student（63.3）接近四步 teacher（64.3），四步 student 勝過 teacher（69.4 vs 64.3），八步相近（71.0 vs 71.2），但與多步 flow 仍有差距。步數–準確率不單調（0.5B：1 步 20.8 → 2 步 19.7；64 步 38.5 → 128 步 37.1）。
- AR 能力（Table 1，六項平均，single-token）：AR SFT† 28.7 / 58.4 / 56.8，Lafa† 31.6 / 60.9 / 63.1（0.5B / 7B / 8B）；DeepSeek-MTP† 30.2 / 60.0，MARS† 30.4 / 59.7；8B 的 Efficient-DLM 63.5。這是平均值：7B HumanEval 78.7 → 76.8 下降；8B BBH 71.6 → 70.7、HumanEval 82.3 → 79.9 下降，平均的提升主要來自 IFEval 38.1 → 62.3 與 GSM8K 62.9 → 77.8（Table 6）。作者的解釋：聯合訓練帶來「multi-token prediction effect」（Abstract、Sec. 1、Sec. 2.2，「consistent with the benefits of multi-token prediction [6, 8, 34]」）；頁面標為「作者認為」。
- Self-speculative（Table 2，六項平均）：7B $B=4/8/16$：準確率 60.9 / 59.3 / 58.0，TPF 2.6 / 4.3 / 4.9；0.5B：31.6 / 32.3 / 30.8，TPF 2.9 / 4.0 / 5.0；8B $B=4$：63.1、2.9。論文概括「approximately 2.5–5 TPF」。
- 吞吐量（Table 8，單張 H100、batch size 1、bfloat16、32-token warm-up，三項平均）：AR SFT 73.9 / 39.5 TPS；Lafa $B=4$ 76.5 / 79.9（2.02×）、$B=8$ 76.3 / 93.8、$B=16$ 74.5 / 100.2（2.54×）；DeepSeek-MTP 76.3 / 53.6；MARS 至多 49.0 TPS；Dream $k=4$ 為 113.7 TPS 但準確率 37.7。六項平均下 7B $B=16$ 為 58.0，低於 AR SFT 的 58.4（Table 2）。
- 消融（Table 3，0.5B、$B=4$、32 步、三項平均）：見步驟 6–9 各處引用的數字。

#### 步驟 13：實作細節（Table 4）與常見錯誤

| 項目 | 0.5B | 7B | 8B |
| --- | --- | --- | --- |
| 初始化 | Qwen2.5 0.5B Instruct | Qwen2.5 7B Instruct | Qwen3 8B Base |
| 資料／epochs／tokens | Dolci-Instruct-SFT／5／約 9B（三者同） | | |
| Block 大小 | 4, 8, 16 | 4, 8, 16 | 4 |
| 最大序列長度 | 512–1024 | 512–1024 | 512 |
| adaLN | 每層 | 跨層共用 | 跨層共用 |
| 學習率 | 1×10⁻⁴ | 1×10⁻⁵ | 1×10⁻⁵ |
| 分散式 | DDP | ZeRO-3 | ZeRO-3 |

三者共同：AdamW，(β₁, β₂) = (0.9, 0.999)，constant schedule、warm-up 前 1% 步；weight decay 0、gradient clipping 1.0、batch 384 sequences（$B=4$；較大 block 用 memory-matched microbatch，適用時由較小 block 的 checkpoint 接續訓練）；bfloat16；訓練 $\sigma=4.0$；$p_{sc}=0.5$；adaLN 學習率倍數 5.0；參數 EMA 0.999（step 500 起）；$M_{\text{bin}}=200$、$\beta=0.99$、$S=100$；flow decoding 推論 $\sigma=0.1$、$g=8.0$。

常見錯誤（頁面 S10 逐條列出）：
1. 把 2.54× 當成 flow 生成的速度：它來自自我推測解碼，輸出等於 greedy AR；純 flow decoding 在 $B=4$、32–64 步時，0.5B 接近（38.1、38.5 vs 38.6）、7B 超過（75.2、75.9 vs 73.9）AR SFT（Table 5；Sec. 3.1 用「approaches」與「surpasses」）。
2. 用「最近的嵌入」把 $\hat x_1$ 轉回 token：Eq. 8 取最後一次 denoiser 預測的 argmax。
3. 讓 prefix 看到 noisy block，或給 prefix 1 以外的時間：prefix 的 KV 會隨 $(x,t)$ 改變，cache 失效，AR 預測也不再是 $f(c)$。
4. 不固定嵌入 norm：norm 相對固定的 $\sigma$ 膨脹，有效 corruption 下降（Table 3）。
5. 把 Eq. 8 寫成 base time 的 Euler $x+\Delta t\,b_t$：Eq. 8 用 $\tau$ 的增量，最後一步落在 $p\mathbf E$，不會碰到 $1/(1-\tau)$ 的發散。
6. 推論沿用訓練的 $\sigma$：Table 4 訓練 4.0、flow decoding 推論 0.1，論文未解釋差異；self-speculative 草稿的 $\sigma$ 未寫明（Alg. 4 只寫「scale $\sigma$」）。
7. 在自我推測草稿加 prefix guidance：論文的草稿不用 guidance；flow decoding 才用 $g=8$。
8. 以為 tokenwise time 用訓練時的非同步內插估計：Alg. 2 的虛擬碼用 synced interpolant 估 $\ell^i$（論文未說明理由）。

#### 步驟 14：限制（Sec. 3、Sec. 5，加上論文範圍）

- 論文自述：flow decoding 要多步才達到 AR 品質；flow map 與多步 flow 仍有差距；如何把 self-conditioning 與 prefix guidance 納入 flow map 留待未來。
- 範圍（頁面照錄，不加評論）：吞吐量只在 7B（與 Efficient-DLM 8B）、batch size 1 量測；8B 只有 $B=4$ 的 single-token 與 self-speculative 結果；消融全為 0.5B、$B=4$、32 步；沒有 seed 或信賴區間；非 † 的 diffusion 基準不是 matched 比較；Table 8 的基準用各自的 native decoder（多為 confidence threshold），只有 Lafa 與 DeepSeek-MTP 以 greedy 驗證（AR SFT 為逐字 greedy；Efficient-DLM 門檻 0.7、MARS 門檻 0.95，App. C.2）。

---

## 2. 學習目標（驗收測試）

每個參考答案都必須寫在指定章節的正文段落中（不只在元件、表格、圖或答案框）。每章末以 `<details class="check">` 呈現該章檢查題（題目與參考答案照搬）。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明 block-AR 分解與它保留的性質 | Eq. 1 怎麼分解 $p(y\mid c)$？平行發生在哪裡？為什麼 prefix 的 KV cache 可以沿用？ | $p(y\mid c)=\prod_r p(y_{rB+1:(r+1)B}\mid c,y_{\le rB})$，是連鎖律、沒有近似；平行只在 block 內；block 之間維持 causal，前面的表示不隨後面的 block 改變 | S1 |
| LO2 | 寫出非同步內插 | Eq. 2 的 $I_t^i$ 是什麼？$\tau$ 要滿足哪些條件？$\tau^i(t)>\tau^j(t)$（$i<j$）代表什麼？ | $I_t^i=(1-\tau^i(t))x_0^i+\tau^i(t)x_1^i$；每個分量可微、遞增（App. A.2 另要求 $C^1$、$t<1$ 時 $\dot\tau^i>0$），$\tau(0)=0$、$\tau(1)=1$；較前的位置先被還原；論文的動機是由左到右，但 $\tau$ 是學出來的 | S2 |
| LO3 | 推導 denoiser 與速度的關係 | Eq. 5 的速度怎麼由 denoiser 得到？推導的關鍵兩步是什麼？ | $b^i=\frac{\dot\tau^i}{1-\tau^i}(D^i\mathbf E-x^i)$，$t<1$；由 Eq. 2 得 $x_1^i-x_0^i=(x_1^i-I_t^i)/(1-\tau^i)$；$x_1^i=\mathbf E_{y_i}$ 使 $\mathbb E[x_1^i\mid c,I_t=x]=D_t(x;c)^i\mathbf E$（Prop. A.1） | S3 |
| LO4 | 說明一步生成為何會不一致 | Eq. 8 在 $K=1$ 時變成什麼？理想的 $D_0$ 等於什麼？C1 玩具一步得到什麼？ | $p^i=\hat D_0(\hat x_0;c)^i$，$\hat y_i=\arg\max p^i$（Eq. 10）；$I_0=x_0$ 與 $y$ 獨立，$D_0^i=p(y_i\mid c)$，各位置獨立的邊際；玩具得到機率為 0 的「hot cream」（hot 0.4、cream 0.6）；步數增加後落在有支撐的組合 | S3 |
| LO5 | 列出把 AR 模型改成 block denoiser 的做法 | prefix 的時間與 attention 怎麼設定？為什麼？嵌入怎麼訓練？為什麼？ | prefix 以 adaLN 固定為時間 1、attention causal 且不看 noisy block，所以 prefix 的 K/V 與 $(x,t)$ 無關，可跨步驟與 block 重用，prefix 輸出仍是 AR 預測 $f(c)$；adaLN 恆等初始化；嵌入與 prefix 共用、可訓練但固定每個 $\mathbf E_v$ 的 L2 norm、只學方向，避免 norm 膨脹降低固定 $\sigma$ 的有效 corruption（Table 3：38.0 vs 36.9、34.4） | S4 |
| LO6 | 說明訓練目標三項 | Eq. 6 的三項各監督什麼？消融結果？ | AR CE 監督 prefix 的下一 token；denoising CE 監督 Eq. 4 的機率；嵌入平方誤差監督決定速度的期望嵌入（Eq. 5）。Table 3：CE 33.6、MSE 29.7、CE & MSE 35.8、+AR 38.0 | S4 |
| LO7 | 說明 tokenwise time 如何學出 | Eq. 22 怎麼定義 $\tau^i$？$\dot\tau^i$ 與 loss 的關係？如何估計 $\ell^i$？ | $\tau^i=F_i^{-1}$，$F_i$ 為 $\ell^i$ 的正規化累積；$\dot\tau^i\propto1/\ell^i(\tau^i)$，loss 高處 $\tau$ 的增加率小，均勻的 $t$ 集中在難的噪聲程度；$M_{\text{bin}}=200$ 格、EMA $\beta=0.99$、每 $S=100$ 步四次無梯度 forward、分段線性 CDF（Eq. 23、Alg. 2）；由左到右是學出來的；同步 36.2 vs 非同步 38.0 | S5 |
| LO8 | 說明兩個推論輔助技巧的成本 | self-conditioning 與 prefix guidance 在推論時各增加多少計算？各改善多少？ | self-conditioning 沿用前一步預測，不增加 forward，32.3 → 38.1；prefix guidance 每步多一次不含 prefix 的 forward（只處理 $B$ 個 block token），$g=8$，34.3 → 38.1（Table 3、Eq. 25、27） | S6 |
| LO9 | 說明 flow map 的兩個條件 | Eq. 7 的 $X_{s,t}$ 怎麼寫？學到的 $\delta$ 要滿足哪兩個條件才等於真正的 flow map？為什麼一步 flow map 能保留位置間關聯？ | 在 $\tau^i\in C^1$、$t<1$ 時 $\dot\tau^i>0$、ODE 解唯一的假設下（$t=1$ 取極限）：$X^i_{s,t}=\frac{1-\tau^i_t}{1-\tau^i_s}x^i+\frac{\tau^i_t-\tau^i_s}{1-\tau^i_s}\delta^i_{s,t}\mathbf E$；diagonal $\delta_{s,s}=\hat D_s$ 與 semigroup $\delta_{s,t}=\gamma\delta_{s,u}+(1-\gamma)\delta_{u,t}(X_{s,u})$，$\gamma^i=\frac{(1-\tau_t)(\tau_u-\tau_s)}{(1-\tau_u)(\tau_t-\tau_s)}$（Prop. A.2）；$\delta_{0,1}$ 是沿整條軌跡的加權平均、權重集中在終點，等於 ODE 終點的 token，而 $\hat D_0$ 只是邊際〔本頁推導；對理想的 δ，學到的 δ̂ 仍有差距〕 | S7 |
| LO10 | 說明自我推測解碼的規則 | 草稿與驗證怎麼做？為什麼輸出等於 greedy AR？TPF 怎麼算？ | 一步 flow 草稿（Eq. 10，無 guidance）；commit 與 greedy AR 一致的最長前綴（Eq. 28），第一個 token 不符則改 commit $\arg\max f(c)_N$，所以 $a\ge1$；被接受的 token 都等於以已接受前綴為條件的 greedy AR 預測；fused forward 用 $B$ 個噪聲 block 同時產生下一份草稿；TPF $=\sum T_j/\sum F_j$，$F_j$ 含 1 次初始草稿（Eq. 29） | S8 |
| LO11 | 正確解讀速度結果 | 2.54× 從哪個設定、哪個平均來？純 flow decoding 要多少步才達到 AR？ | 7B、$B=16$ 自我推測解碼，IFEval/GSM8K/HumanEval 三項平均 74.5 vs 73.9，100.2 vs 39.5 TPS（Table 8）；$B=4$ 為 2.02×、76.5；六項平均下 $B=16$ 為 58.0 < 58.4；純 flow decoding 在 $B=4$、32–64 步時 0.5B 接近、7B 超過 AR SFT（7B 75.2、75.9 vs 73.9；0.5B 38.1、38.5 vs 38.6，Table 5） | S9 |
| LO12 | 正確解讀 AR 能力與資料效率的結果 | flow 訓練對 AR 準確率的影響是多少？要注意什麼？訓練用多少 tokens？ | Table 1：0.5B 28.7 → 31.6、7B 58.4 → 60.9、8B 56.8 → 63.1；是六項平均，部分項目下降（7B HumanEval 78.7 → 76.8），8B 的提升主要來自 IFEval 與 GSM8K，8B 由 Base 起訓；作者把提升解釋為 multi-token prediction effect（作者的解釋）；訓練 9–27B tokens（<30B），比 Dream、LLaDA、Efficient-DLM 的報告預算少至少 18 倍 | S9 |

---

## 3. 頁面大綱

全頁單欄（文字欄 680–720px），元件放加寬圖版（最大 1080px），章節編號以 JetBrains Mono 顯示（`01 · …`）。視覺記號：「block 條」——一列 token 格，左側乾淨 prefix（實線框）、右側 $B$ 個 noisy 格（虛線框，內部以灰階深淺表示 $\tau^i$，每格下方另有一列 $\tau^i$ 數值文字，不只靠灰階），在 hero、S1、S2、S8 重複出現。章節標題一律用陳述句。

| # | id | 標題 | 內容 | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈Lafa：把預訓練語言模型改成 block 內平行生成的連續 flow〉 | 副標「這篇論文讓 block 之間照 AR 順序、block 內的 token 從噪聲經連續的逐步變換（flow）一起生成，並用同一個模型的 AR 預測驗證一步草稿。」block 條（prefix「a dog waits」+ 3 格 block，取自 Fig. 2）；三張事實卡（下方）；連結 arXiv、`../index.html#paper-9`、`lafa-lineage-overview.html` | 1 螢幕 |
| S1 | `blockar` | 01 · Block 之間自迴歸，block 內平行 | AR 基礎（2–3 句，見步驟 1）、Eq. 1、LO1 答案句；符號表（可收合）。延伸：`lafa-parallel-decoding-background.html` | 350–450 字 |
| S2 | `flow` | 02 · 每個 block 的嵌入從高斯噪聲經 ODE 積分到資料的嵌入 | $x_1$、$x_0$、同步內插、flow matching 速度與 ODE（Eq. 3）、tokenwise time（Eq. 2）與條件（每個分量 $C^1$、$\tau(0)=0$、$\tau(1)=1$、嚴格遞增，即 $t<1$ 時 $\dot\tau^i>0$）、論文的動機句與先例 [26–30]（一句）、「τ 是學出來的」；靜態 SVG：兩條 $\tau^i(t)$ 曲線（同步虛線 $\tau=t$）。延伸：`lafa-embedding-flow-denoiser.html`、`flow-matching-velocity-target.html`、`lafa-asynchronous-tokenwise-time.html` | 400–500 字 + 小圖 |
| S3 | `denoiser` | 03 · 預測 token 機率就能得到速度 | 開頭兩句：「$\hat D$ 是 transformer 的輸出，第 04 節說明怎麼由 AR 模型改成；本節先把它當成給定的函式。C1 用玩具的精確 $D$（由高斯似然算出的後驗），不是神經網路。」Eq. 4、5；正文寫出推導的兩個關鍵步驟（LO3 答案句），四步完整推導收在 `details`；「機率向量 × $\mathbf E$ = 期望嵌入」例子；Eq. 8；「τ 時間的精確步」〔本頁推導〕正文一句、推導收在 `details`；Eq. 10；$D_0$ = 邊際〔本頁推導〕；C1。延伸：`cdlm-representation-space.html` | 550–650 字 + C1 |
| S4 | `adapt` | 04 · 預訓練 AR 模型改成 block denoiser | 四個改動（步驟 6）、Eq. 6 與 Table 3 相關列、平行訓練；C2。延伸：`lafa-ar-to-block-adaptation.html` | 500–600 字 + C2 |
| S5 | `time` | 05 · Tokenwise time 是 loss 的反累積分布 | Eq. 22、〔本頁推導〕$\dot\tau$、Eq. 23 與 Alg. 2、Fig. 5 的文字描述、同步 vs 非同步 36.2/38.0；C3。延伸：`cdlm-noise-schedule.html` | 400–500 字 + C3 |
| S6 | `assist` | 06 · Self-conditioning 與 prefix guidance | Eq. 24–25、27，成本與消融，$g>1$ 註記。延伸：`self-conditioning-flow-lm.html` | 300–400 字 |
| S7 | `flowmap` | 07 · Flow map 把多步積分換成一次跳躍 | 順序：動機（多步的成本）→ Eq. 7 → R11 的兩句「δ 是沿路 denoiser 預測的加權平均」→ diagonal、semigroup 與 $\gamma$（Eq. 16）、$t\to1$〔本頁推導〕與其假設（LO9 答案句全在正文）→ C4 → Prop. A.2 構造（可收合）→ Eq. 9。Eq. 14–15、19–21 與 Prop. A.3（蒸餾）收在 `details`（不是學習目標）。延伸：`flow-map-few-step-distillation.html`、`lafa-self-conditioning-and-flow-map.html` | 850–950 字 + C4 |
| S8 | `spec` | 08 · 自我推測解碼用 AR 驗證一步草稿 | 步驟 11 全部；C5（含 fused 遮罩）。延伸：`lafa-self-speculative-decoding.html` | 500–600 字 + C5 |
| S9 | `results` | 09 · 多步 flow 追上 AR，速度來自自我推測 | F1 靜態圖（Table 5 重繪，見 4.6）；Table 1、Table 2（7B 列）、Table 8（Lafa、AR SFT、DeepSeek-MTP、MARS $B=16$、Dream $k=4$）摘錄；「解讀注意」callout（步驟 12 的平均與範圍說明）；正文寫出訓練量：Sec. 3「5–15 epochs (9–27B tokens)」、Table 4「5 epochs / approximately 9B」，兩處照錄，合起來 <30B，比 Dream、LLaDA、Efficient-DLM 報告的預算少至少 18 倍（LO12）。F1 讀者應觀察到：0.5B 與 7B 的 $B=4$ flow 在 32–64 步附近追上或超過 AR SFT 水平線；同步數下 $B=16$ 低於 $B=4$；flow map 在 1–2 步高於 flow，8 步兩者相近。延伸：`lafa-scale-and-evaluation.html` | 500–600 字 + 圖 + 3 表 |
| S10 | `impl` | 10 · 實作設定與常見錯誤 | Table 4（步驟 13）與 8 條常見錯誤 | 300–400 字 + 表 |
| S11 | `history` | 11 · 研究脈絡 | 時間線（6.4）+ 100–150 字：block 平行解碼 → flow matching / CDCD → MTP → block diffusion → AR 轉擴散（Dream、Efficient-DLM；筆記另列 From Next-Token to Next-Block 2512.06776、WeDLM 2512.22737，只在文字提及、不入時間線）→ flow map → 同期嵌入 flow → Lafa | 時間線 |
| S12 | `next` | 12 · 限制與延伸閱讀 | 步驟 14；筆記清單（6.4）；arXiv | 200 字 + 連結 |

Hero 三張事實卡（不使用未定義術語）：
1. 「文字切成長度固定的 block：block 之間照 AR 順序生成，block 內的 token 從高斯噪聲出發，經過連續的逐步變換（解一個常微分方程）一起變成詞嵌入。」
2. 「模型不從零訓練：把預訓練的 AR 模型（Qwen2.5 0.5B／7B、Qwen3 8B）改成同時做 AR 預測與 block 去噪，訓練資料少於 300 億 tokens。」
3. 「最快的模式是 flow 一步寫草稿、同一個模型的 AR 預測逐字驗證。輸出與 greedy AR（每步取機率最大的 token）相同，7B 每秒產生的 token 數（吞吐量）是 AR 的 2.54 倍。」

章節目錄與窄螢幕章節列沿用參考頁。

---

## 4. 互動元件規格

### 4.0 共用玩具（C1、C4 共用同一份函式與粒子）

- 詞彙 `VOCAB = [hot, ice, sour, dog, cream]`（$\lvert V\rvert=5$），兩個位置共用；$d=2$，$\mathbf E_v=(\cos\theta_v,\sin\theta_v)$，$\theta_v=\pi/2-2\pi v/5$（單位圓上等距，norm 皆 1，對應 norm-frozen 嵌入）。$B=2$。
- 聯合先驗（prefix 固定，示意值）：$p(\text{hot dog})=0.4$、$p(\text{ice cream})=0.3$、$p(\text{sour cream})=0.3$，其餘 0。邊際：位置 1 hot 0.4 / ice 0.3 / sour 0.3；位置 2 dog 0.4 / cream 0.6。
- tokenwise time（示意族，不是論文的學習結果）：$\tau^1(t)=1-(1-t)^{1+2\lambda}$，$\tau^2(t)=t$，$\lambda\in[0,1]$；$\lambda=0$ 為同步，$\lambda>0$ 時 $\tau^1$ 增加得比 $\tau^2$ 快。滿足 $\tau(0)=0$、$\tau(1)=1$、嚴格遞增。
- 精確 denoiser（〔本頁推導〕，由 Eq. 2、4 與高斯似然）：$p(y\mid I_t=x)\propto p(y)\prod_i\mathcal N\big(x^i;\tau^i\mathbf E_{y_i},(1-\tau^i)^2\sigma^2I\big)$；$D_t(x)^i$ 為對另一位置加總的邊際。實作：對三個有支撐的組合算 $\log p(y)-\sum_i\lVert x^i-\tau^i\mathbf E_{y_i}\rVert^2/(2\max((1-\tau^i)^2\sigma^2,10^{-24}))$，log-sum-exp 正規化。$\tau=0$ 時似然與 $y$ 無關，$D_0$ 精確等於邊際。
- 粒子：`mulberry32(seed)` + Box–Muller（每次取兩個均勻數 $u_1,u_2$，回傳 $\sqrt{-2\ln(1-u_1)}\cos(2\pi u_2)$，只用 cos 分支）。粒子 $n=0..499$ 依序抽 $g^1_x,g^1_y,g^2_x,g^2_y$；$\hat x_0=\sigma g$（改 $\sigma$ 只縮放同一批 $g$）。預設 seed 7；「重新抽樣」每按一次 seed +1。
- 推論（Eq. 8）：$t_k=k/K$；$p_k=D_{t_k}(\hat x_k)$；$\hat x^i_{k+1}=\hat x^i_k+\frac{\tau^i_{k+1}-\tau^i_k}{1-\tau^i_k}(p^i_k\mathbf E-\hat x^i_k)$；$\hat y_i=\arg\max p^i_{K-1}$（同值取索引小者）。$k\le K-1$ 時 $t_k<1$，故 $1-\tau^i_k>0$。

### C1 · 非同步 flow 玩具：一步的邊際與多步的聯合（S3）

- **教的概念**：Eq. 5、8 的運作；$K=1$ 時 $\hat D_0$ = 邊際，各位置分別取 argmax 得到不一致的組合；步數增加後樣本落在有支撐的組合、頻率接近先驗；改變 $\lambda$（位置 1 的 $\tau$ 增加較快）時少步結果怎麼變。
- **對應 LO**：LO4（主要）、LO3（Eq. 5、8 的運作）。
- **輸入**：分段按鈕 $K\in\{1,2,4,8,16,32,64\}$（預設 1）；滑桿 $\lambda\in[0,1]$ 步長 0.25（預設 0.5）；分段按鈕 $\sigma\in\{0.25,0.5,1,2,4\}$（預設 1）；按鈕「重新抽樣」。
- **計算**：500 個粒子各跑一次 Eq. 8（500 × 64 × 3 組合；`input` 以 rAF 合併、依 key 快取，見 6.2）。
- **顯示**：
  - 左：兩張並排平面圖（位置 1、位置 2；寬度 < 600px 時上下排），座標 $[-2.2,2.2]^2$，單位圓上標 5 個 token 點（文字標籤）。前 12 個粒子的軌跡 $\hat x_0\to\hat x_K$ 以折線畫出（位置 1 `--c-par` 實線、位置 2 `--c-perp` 虛線），終點標記形狀依生成組合：有支撐 ●、無支撐 ✕。其餘粒子只畫終點小點。
  - 右：長條圖，六個組合（hot/ice/sour × dog/cream）的生成頻率（實心條）與先驗（空心框）並排，無支撐的三個組合先驗為 0，條上直接標百分比。
  - 讀數：「有支撐比例」、「與先驗的總變差距離 TV $=\frac12\sum\lvert\hat p-p\rvert$」、每步 $\tau^1,\tau^2$ 網格（JetBrains Mono，三位小數）。附註：「500 個粒子的抽樣誤差使 TV 有約 0.025 量級的隨機起伏〔本頁推導：三類多項分布期望絕對偏差之和的一半〕，所以 K ≥ 16 之後 TV 的差異不代表方法差異。」
- **數值保護**：變異數下限 $10^{-24}$；log-sum-exp；$\hat y$ 同值取小索引。
- **讀者應觀察到**：(1) $K=1$：500 個粒子全部是「hot cream」，有支撐比例 0；(2) $K=2$ 時同步（$\lambda=0$）有支撐比例 0.898，$\lambda=0.5$ 為 0.984、$\lambda=1$ 為 1.000，但少步時頻率偏離先驗（$\lambda=1,K=2$：ice cream 0.936）；(3) $K\ge16$ 時三種 $\lambda$ 的 TV 都 ≤ 0.026。(4) 少步時 $\lambda$ 越大，有支撐比例越高，但 TV 也越大（$K=2$：TV 0.102 / 0.340 / 0.636，$\lambda=0/0.5/1$）。頁面明寫：這是 2 個位置、5 個 token 的玩具與精確 denoiser，不代表 Lafa 的實際步數需求（Table 5）。另加 callout：「玩具的 $\lambda$ 族是手選的，不是 Lafa 由 loss 學出的 $\tau$；它在 2 位置、少步時的分布誤差，既不支持也不反駁 Table 3 的同步 36.2 vs 非同步 38.0（不同的量、不同的規模）。」
- **無障礙**：組合以形狀（●/✕）+ 文字標籤區分；長條圖附 `<table class="sr">` 同內容；讀數 live region 在輸入停止 500ms 後更新；平面圖 `role="img"` + `aria-label`（「K = 1，有支撐比例 0%，全部為 hot cream」）。沒有動畫（每次輸入靜態重繪）。

### C2 · 注意力遮罩（S4；C5 重用同一個繪製函式）

- **教的概念**：Fig. 3（推論時 prefix + 一個 noisy block）與 Fig. 4b（平行訓練：乾淨序列 + noisy 複本）的規則；為什麼 prefix 的 K/V 與 block 狀態無關。
- **對應 LO**：LO5（attention 部分）；C5 的 fused 模式支援 LO10。
- **輸入**：分段按鈕「推論（Fig. 3）／平行訓練（Fig. 4b）」（預設推論）；$N\in\{2,\dots,6\}$（推論用，預設 3）；$B\in\{2,3,4\}$（預設 3）；訓練模式的 block 數 $R=2$ 固定（同 Fig. 4），此時 $N$ 控制項停用（`disabled` 並註明「訓練模式不用 N」）。
- **規則**（`maskAllowed(mode, q, k)`）：
  - 推論：prefix query $i$ 看 prefix key $j\le i$；block query 看所有 prefix 與所有 block key；prefix query 不看 block key。
  - 訓練：乾淨 query $p$ 看乾淨 key $q\le p$；第 $r$ 個 noisy block（0 起算）的 query 看乾淨 key $q<rB$ 與同一 block 的 noisy key。
  - Fused（C5 用）：$c\oplus\hat y$ 內 causal；$z^{(j)}$ 看全部 $c$、$\hat y_{1:j}$ 與自己的 block。
- **顯示**：方格（列 = query、欄 = key），允許的格畫實心點 ●，阻擋的格留空並以淡斜線底紋；列與欄以「c₁…、x₁…」標籤，區塊分界加粗線。點選（或 Tab 聚焦）一列：該列允許的 key 加框，並以一句話描述（例：「x₂ 看 c₁–c₃ 與 x₁–x₃」）。讀數：允許格數／總格數；推論與 fused 模式另顯示「prefix 列落在 block 欄的允許格數：0」（KV cache 可重用的量化依據）。方格放在 `.tscroll` 內，列標籤欄 sticky。
- **讀者應觀察到**：推論模式中 prefix 列沒有任何點落在 block 欄——prefix 的表示與 $x,t$ 無關；訓練模式第 1 個 noisy block 看不到任何乾淨 token（Fig. 4 中序列沒有 prompt），第 2 個看到前 $B$ 個乾淨 token。
- **無障礙**：方格為 `<table>`（格內 ● 或空白 + `aria-label`「允許／阻擋」），不用顏色區分；列標頭是 `<button>`，整組只佔一個 Tab 停點（roving tabindex），上下方向鍵換列、Home/End 跳首尾，觸控可用。

### C3 · Tokenwise time = loss 的反累積分布（S5）

- **對應 LO**：LO7。
- **教的概念**：Eq. 22；loss 高處 $\tau$ 的增加率小；均勻 $t$ 網格映到 $\tau$ 後集中在高 loss 區；用 Alg. 2 的 $M_{\text{bin}}=200$ 格分段線性 CDF 計算。
- **玩具 loss 族（示意值）**：$\ell^i(\tau)=\varepsilon+(1-\tau)^{q_i}$，$\varepsilon=0.05$（保證 $\ell>0$），$q_i=q_1+(i-1)\Delta q$，$B=4$，$q_i$ 夾在 $[0.1,7]$；被夾住時讀數表該列加註「q 已夾在 0.1」（或 7），順序讀數用夾住後的 $q$，相等者寫「相同」。$q$ 越大，loss 越集中在噪聲多（$\tau$ 小）的一側。頁面明寫：真實 $\ell^i$ 由模型估計，論文沒有給數值；這個族只用來示範 Eq. 22 的機制。
- **輸入**：滑桿 $q_1\in[0.2,4]$ 步長 0.1（預設 0.5）；滑桿 $\Delta q\in[-1,1]$ 步長 0.1（預設 0.5）；分段按鈕 $K\in\{4,8,16,32\}$（預設 8）。
- **計算**（`tauBinned(q, t)`）：$L_m=\ell((m+0.5)/200)$；$F_0=0$、$F_{m+1}=F_m+L_m/\sum L$；$t\ge1$ 回傳 1；否則二分搜尋最後一個 $F_m\le t$，$\tau=(m+(t-F_m)/(F_{m+1}-F_m))/200$（$F_{m+1}-F_m>0$ 由 $\varepsilon$ 保證）。每個 $q$ 的 $F$ 陣列快取。$\dot\tau^i(t)$ 讀數用〔本頁推導〕$\big(\int_0^1\ell\big)/\ell(\tau^i(t))$，$\int_0^1\ell=\varepsilon+1/(q+1)$（閉式）。
- **顯示**：
  - 左圖「$\ell^i(\tau)$」：四條曲線，各除以自己的積分（即密度），位置 1–4 以實線／長虛線／短虛線／點線 + 曲線端直接標「i = 1…4」。
  - 右圖「$\tau^i(t)$」：同樣四條 + 同步參考 $\tau=t$（`--ink-3` 細虛線，標「同步」），排版仿 Fig. 5。
  - 下方「網格」條：每個位置一條水平 $\tau$ 軸，畫出 $\tau^i(k/K)$，$k=0..K$ 的刻度——高 loss 區刻度較密。
  - 讀數表：位置、$q_i$、$\tau^i(0.5)$、$\dot\tau^i(0.5)$，以及「順序：τ¹ > τ² > τ³ > τ⁴」或實際順序。
- **讀者應觀察到**：(1) 預設下 $\tau^1(t)>\tau^2(t)>\tau^3(t)>\tau^4(t)$：loss 越集中在噪聲側的位置，$\tau$ 在噪聲側增加得越慢；(2) $\Delta q=0$ 時四條重合；$\Delta q<0$ 時順序反轉；(3) 網格在 $\tau$ 小處較密。頁面加一句：論文中由左到右的順序是訓練中學到的，不是人工設定（Fig. 5），論文沒有給出原因。
- **無障礙**：曲線以線型 + 直接標籤區分；讀數表為真實 `<table>`；live region 於 `change` 或 500ms 後更新。

### C4 · Two-time denoiser 與 flow map（S7）

- **對應 LO**：LO9。
- **教的概念**：Eq. 18 的 $\delta_{s,t}$ 是沿軌跡的加權平均；Eq. 7 以 $\delta_{s,t}$ 一次跳躍就到達真正的 $X_{s,t}$，用 $\hat D_s$ 跳（即一步 Euler）則偏離；semigroup 權重 $\gamma$（Eq. 16）；$t\to1$ 時 $\delta_{0,t}$ 趨向終點 token 的 one-hot，而 $\hat D_0$ 是邊際。
- **設定**：4.0 的玩具，teacher = 精確 denoiser；$\sigma$、$\lambda$、seed 與 C1 同步（讀取 C1 目前的值）。
- **輸入**：滑桿 粒子編號 $n\in[0,11]$（預設 0；即 C1 平面圖中的 12 條軌跡之一）；滑桿 $s\in[0,0.9]$ 步長 0.01（預設 0）；滑桿 $t\in[s+0.01,0.99]$ 步長 0.01（預設 0.5；$s$ 改變時若 $t\le s$ 則設 $t=s+0.01$）。
- **計算**（細網格 $n_f=2000$，$r_j=j/2000$；$s,t$ 對齊網格，$u=\lfloor(j_s+j_t)/2\rfloor/2000$）：
  1. 由 $\hat x_0$ 以 Eq. 8 的更新（細步）走到 $s$，得 $x_s$。
  2. 由 $s$ 走到 $t$，同時累積 $\mathrm{acc}^i\mathrel{+}=(\zeta^i_{j+1}-\zeta^i_j)\,p^i_j$，$\zeta=1/(1-\tau)$；$\delta^i_{s,t}=\mathrm{acc}^i/(\zeta^i_t-\zeta^i_s)$（Eq. 18 的離散形式，〔本頁推導〕見步驟 10）。軌跡終點為參考 $X_{s,t}$。
  3. $X^{(7)}=$ Eq. 7 以 $\delta_{s,t}$ 計算；$X^{\text{Euler}}=$ Eq. 7 以 $\hat D_s(x_s)$ 代替 $\delta$。
  4. $\gamma^i$ 依 Eq. 16；semigroup 殘差 $\max\lvert\delta_{s,t}-[\gamma\delta_{s,u}+(1-\gamma)\delta_{u,t}(X_{s,u})]\rvert$（分別以 $[s,u]$、$[u,t]$ 兩段累積）。
  每次輸入改變重算（3 段 × 至多 2000 步 × 3 組合，同步完成）。$t\le0.99$ 時 $\zeta$ 至多 $10^{6}$（$\lambda=1$、位置 1；$\lambda=0.5$ 為 $10^4$）。最壞情形實測（check.py「C4 stress」）：$\lambda=1$、$s=0.98$、$t=0.99$ 的 Eq. 7 殘差 1.6e-16、semigroup 殘差 0；$s=0$、$t=0.99$ 為 4.6e-16、1.1e-16。原因是兩邊用同一組 $\Delta\zeta$，相消誤差也一致。selfTest 門檻用 1e-9。
- **顯示**：
  - 左：兩個位置的平面圖，畫 $x_s$（空心圓）、參考軌跡 $s\to t$（細實線）、$X^{(7)}$（實心圓）、$X^{\text{Euler}}$（✕，虛線連到 $x_s$）、token 點。
  - 右：每個位置一組水平長條（只列有支撐的 token：位置 1 hot/ice/sour，位置 2 dog/cream），$\hat D_s$（空心）與 $\delta_{s,t}$（實心）並排，標數值。
  - 讀數：$\tau^i_s,\tau^i_t$、$\gamma^i$（$u$ 取中點）、「Eq. 7 與軌跡終點的差」（科學記號）、「一步 Euler 的誤差」、semigroup 殘差、兩種 argmax。
  - 附註：「Eq. 7 與 Eq. 16 的殘差只有浮點誤差，這是代數恆等：參考軌跡就是 Eq. 8 在細網格上的離散軌跡，每一步是一段常數 denoiser 的精確解〔本頁推導〕。它不是 Prop. A.2 的數值驗證；細網格越密，這條離散軌跡越接近 ODE 的解。」
  - 左右兩區在 < 600px 時上下排。沒有動畫。
- **讀者應觀察到**：(1) $s=0$：$\hat D_0$ 為邊際（hot 0.4 / ice 0.3 / sour 0.3；dog 0.4 / cream 0.6），$\delta_{0,t}$ 隨 $t$ 增大集中到該粒子的終點組合（粒子 0：sour cream），$t=0.99$ 時近似 one-hot；(2) 一步 Euler 的誤差隨 $t-s$ 增大；(3) $s$ 接近 $t$ 時 $\delta_{s,t}\approx\hat D_s$（diagonal 條件）。
- **無障礙**：長條附數值文字；點形狀區分（○ ● ✕）；live region 於 `change` 或 500ms 後更新。

### C5 · 自我推測解碼模擬（S8）

- **對應 LO**：LO10。
- **教的概念**：Eq. 28 的接受規則與第一個 token 的修正；$a\ge1$；TPF（Eq. 29）；每次 fused forward 的位置數 $B+B^2$；接受率隨位置遞減時，加大 $B$ 的 TPF 增益遞減（Table 2：7B 2.6 → 4.3 → 4.9）。
- **模型（示意，頁面明寫）**：不模擬語言；把「greedy AR 的輸出」當成固定的參考序列（長度 $T=1024$；$T=256$ 時 $B=16$ 的模擬 TPF 與 $\mathbb E[a]$ 差 5%，會被誤讀成錯誤），草稿在 block 位置 $i$ 正確的機率為 $r_i=r_1\rho^{i-1}$，各位置獨立。真實的一步草稿錯誤來自 C1 的邊際問題，不是獨立的；這裡只用來看規則與 TPF。
- **輸入**：分段按鈕 $B\in\{4,8,16\}$（預設 8）；滑桿 $r_1\in[0.5,1]$ 步長 0.05（預設 0.95）；滑桿 $\rho\in[0.5,1]$ 步長 0.05（預設 0.9）；按鈕「下一步」「自動執行／暫停」（同一鍵切換）「跑完」「重設」；seed 1（「重新抽樣」+1）。
- **演算法**（`c5Sim`，抽樣順序固定）：每次迴圈先抽 $B$ 個均勻數 $u_1..u_B$（不論是否用到），$\mathrm{ok}_i=(u_i<r_1\rho^{i-1})$；若 $\neg\mathrm{ok}_1$：修正，$a=1$，本次位置數 $1+B$；否則 $a=1+$（$\mathrm{ok}_2,\mathrm{ok}_3,\dots$ 連續為真的個數），位置數 $B+B^2$；$a\leftarrow\min(a,T-\text{done})$；迴圈數 +1。直到 done $=T$。$F=$ 迴圈數 $+1$（初始草稿），TPF $=T/F$。
- **閉式期望**〔本頁推導〕：$a\ge j$（$j\ge2$）若且唯若 $\hat y_1..\hat y_j$ 全對，故 $\mathbb E[a]=1+\sum_{j=2}^{B}\prod_{i=1}^{j}r_i$；長序列時 TPF → $\mathbb E[a]$。讀數旁註明「模擬有抽樣誤差，$T=1024$ 時可達約 3%」。
- **顯示**：
  - 上：本次迴圈——草稿列 $B$ 格（✓/✗ 文字標記）、接受長度 $a$（括號標出被 commit 的格）、是否修正；$B$ 個候選 block $z^{(1)}..z^{(B)}$ 的小列，被選中的 $z^{(a)}$ 以 `--accent` 外框 +「下一份草稿」標籤。
  - 中：已 commit 序列條（每格一個 token，依 commit 的 forward 編號交替深淺，並在每段起點標編號，非顏色編碼）。
  - 右：fused 遮罩（呼叫 C2 的繪製函式，mode fused，$N=3$，$B$ 取 $\min(B,4)$ 並註明「遮罩以 B ≤ 4 顯示」）；$c$ 的列以灰底標「KV cache，不重算」。
  - 讀數：已 commit $T$（最後一次被 $T-\text{done}$ 截斷時，該次 $a$ 旁加「（序列結尾截斷）」）、forward 數 $F$、目前 TPF、$\mathbb E[a]$ 閉式、平均每次 forward 的位置數、論文 Table 2 對照（7B：$B=4/8/16$ TPF 2.6 / 4.3 / 4.9，準確率 60.9 / 59.3 / 58.0）。
  - 一行說明：「每個被接受的 token 都等於 greedy AR 的預測，所以輸出與 greedy AR 相同；flow 只決定一次 forward 能 commit 幾個。」
- **數值保護**：$r_i\le1$；$T-\text{done}$ 截斷最後一次；自動執行每 400ms 一步（可暫停），reduced-motion 時「自動執行」改為一次跑完。
- **讀者應觀察到**：(1) $r_1=\rho=1$ 時每次 commit $B$ 個，TPF $=T/(T/B+1)$（$B=16$：15.75）；(2) 預設下 $\mathbb E[a]$ 由 $B=4$ 的 2.870 到 $B=8$ 的 3.402，再到 $B=16$ 只增到 3.424，但每次 forward 的位置數由 20 → 72 → 272；(3) 第一個 token 錯時只 commit 1 個（修正值）。
- **無障礙**：✓/✗ 文字；live region 只在「下一步」或自動執行停止時宣告一次（「第 n 次 forward，接受 a 個，TPF …」）。

### 4.6 F1 · Table 5 重繪（S9，靜態，非互動）

兩格（0.5B、7B），橫軸 Euler 步數（對數，1–128），縱軸三項平均準確率。$B=4/8/16$ 三條 flow 折線（實線／長虛線／點線 + 端點標籤）、$B=4$ flow map 折線（`--accent`，◆ 標記）、AR SFT 水平線（38.6、73.9）。資料直接取 Table 5，圖下標「數字取自論文 Table 5」。附 `<table class="sr">`。對應 LO11。線條以線型與端點標籤區分，不靠顏色。

---

## 5. 驗收數值表

全部由 `verify/check.py` 產生（輸出 `verify/out.txt`）。頁面以相同演算法即時計算，`?test` 的 `selfTest()` 逐項比對。

### 5.1 C1（seed 7，500 粒子）

| σ | λ | K | 有支撐比例 | TV | hot dog / ice cream / sour cream |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 1 | 0.000（全部 hot cream） | 1.000 | 0 / 0 / 0 |
| 1 | 0 | 2 | 0.898 | 0.102 | 0.342 / 0.294 / 0.262 |
| 1 | 0 | 4 | 0.998 | 0.058 | 0.390 / 0.356 / 0.252 |
| 1 | 0 | 16 | 1.000 | 0.018 | 0.418 / 0.298 / 0.284 |
| 1 | 0.5 | 1 | 0.000 | 1.000 | 0 / 0 / 0 |
| 1 | 0.5 | 2 | 0.984 | 0.340 | 0.242 / 0.624 / 0.118 |
| 1 | 0.5 | 4 | 1.000 | 0.132 | 0.344 / 0.432 / 0.224 |
| 1 | 0.5 | 8 | 1.000 | 0.056 | 0.374 / 0.356 / 0.270 |
| 1 | 0.5 | 16 | 1.000 | 0.026 | 0.380 / 0.326 / 0.294 |
| 1 | 0.5 | 64 | 1.000 | 0.008 | 0.396 / 0.296 / 0.308 |
| 1 | 1 | 2 | 1.000 | 0.636 | 0.058 / 0.936 / 0.006 |
| 1 | 1 | 64 | 1.000 | 0.010 | 0.398 / 0.292 / 0.310 |
| 0.25 | 0.5 | 2 | 1.000 | 0.696 | 0.004 / 0.996 / 0.000 |
| 0.25 | 0.5 | 64 | 1.000 | 0.000 | 0.400 / 0.300 / 0.300 |
| 4 | 0.5 | 2 | 0.820 | 0.180 | 0.308 / 0.216 / 0.296 |

容許：比例與頻率 ±0.004（2 個粒子；同 PRNG，只容許浮點近同值時的 argmax 翻轉）。另：粒子 0 的 $g=(0.14197,-0.86425;\,-1.00637,0.07088)$；$\sigma=1,\lambda=0.5$ 時 $K=1$ hot cream、$K=2$ ice cream、$K=8$ 與 64 sour cream。$D_0$（任意 $x$）＝位置 1 [0.4, 0.3, 0.3, 0, 0]、位置 2 [0, 0, 0, 0.4, 0.6]（誤差 1e-12）。

### 5.2 C2（允許格數／總格數）

推論 $N=3,B=3$：24/36；推論 $N=4,B=4$：42/64；訓練 $R=2,B=3$：48/144；fused $N=3,B=3$：93/225；fused $N=2,B=4$：157/484。必須完全相等。

### 5.3 C3（$\varepsilon=0.05$，$M_{\text{bin}}=200$）

| 設定 | $q_i$ | $\tau^i(0.25)$ | $\tau^i(0.5)$ | $\tau^i(0.75)$ | $\dot\tau^i(0.5)$ |
| --- | --- | --- | --- | --- | --- |
| 預設 $q_1=0.5,\Delta q=0.5$ | 0.5 / 1 / 1.5 / 2 | 0.1785 / 0.1403 / 0.1167 / 0.1006 | 0.3778 / 0.3067 / 0.2606 / 0.2285 | 0.6140 / 0.5232 / 0.4605 / 0.4154 | 0.8544 / 0.7399 / 0.6562 / 0.5941 |
| $q_1=1,\Delta q=0$ | 1 ×4 | 0.1403 | 0.3067 | 0.5232 | 0.7399 |
| $q_1=2,\Delta q=-0.5$ | 2 / 1.5 / 1 / 0.5 | 0.1006 / 0.1167 / 0.1403 / 0.1785 | 0.2285 / 0.2606 / 0.3067 / 0.3778 | 0.4154 / 0.4605 / 0.5232 / 0.6140 | 0.5941 / 0.6562 / 0.7399 / 0.8544 |

$K=8$ 網格：$q=0.5$：0, 0.0872, 0.1785, 0.2749, 0.3778, 0.4894, 0.6140, 0.7615, 1；$q=2$：0, 0.0478, 0.1006, 0.1600, 0.2285, 0.3105, 0.4154, 0.5704, 1。分段線性與閉式反函數（二分法）在 $q\in\{0.5,\dots,5.5\}$、$t$ 每 0.01 的最大差 $9.2\times10^{-5}$。容許 ±0.0005。

### 5.4 C4（seed 7 粒子 0，$\sigma=1$，$\lambda=0.5$，$n_f=2000$）

| $s,t$ | $\hat D_s$ 位置 1（hot/ice/sour）；位置 2（dog/cream） | $\delta_{s,t}$ 同順序 | 一步 Euler 誤差（位置 1, 2） | $\gamma$（$u$ 中點） |
| --- | --- | --- | --- | --- |
| 0, 0.5 | 0.4/0.3/0.3；0.4/0.6 | 0.086/0.181/0.733；0.109/0.891 | 0.536, 0.171 | 0.2593, 0.3333 |
| 0, 0.9 | 同上 | 0.003/0.006/0.992；0.012/0.988 | 1.045, 0.410 | 0.0233, 0.0909 |
| 0, 0.99 | 同上 | 0.000/0.000/1.000；0.001/0.999 | 1.066, 0.464 | 0.0003, 0.0099 |
| 0.3, 0.6 | 0.075/0.288/0.637；0.075/0.925 | 0.005/0.054/0.941；0.006/0.994 | 0.263, 0.035 | 0.3005, 0.3636 |

所有情形 Eq. 7 與軌跡終點的差、semigroup 殘差實測 < 1e-15（含壓力案例 $\lambda=1$，$s=0.98,t=0.99$ 與 $s=0,t=0.99$）；selfTest 門檻 1e-9。這是代數恆等的浮點檢查，不是 Prop. A.2 的驗證。容許：機率 ±0.002、誤差 ±0.002、$\gamma$ ±0.0002。

### 5.5 C5（$T=1024$，seed 1）

| $B$ | $r_1,\rho$ | $\mathbb E[a]$ | 模擬 $F$ | 模擬 TPF | 平均位置／forward |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.95, 0.9 | 2.8701 | 351 | 2.9174 | 19.2 |
| 8 | 0.95, 0.9 | 3.4024 | 300 | 3.4133 | 69.1 |
| 16 | 0.95, 0.9 | 3.4243 | 302 | 3.3907 | 261.0 |
| 4 / 8 / 16 | 1, 1 | 4 / 8 / 16 | 257 / 129 / 65 | 3.9844 / 7.9380 / 15.7538 | 20 / 72 / 272 |
| 8 | 0.9, 0.8 | 2.2806 | 452 | 2.2655 | 65.7 |
| 8 | 0.5, 0.9 | 1.3645 | 760 | 1.3474 | 40.6 |

抽樣造成的模擬 TPF 與 $\mathbb E[a]$ 差距在上表最大 1.7%（$B=4$ 預設），在 check.py 全部 12 組中最大 3.0%（$B=16$、0.9、0.8）；$(1,1)$ 列的差來自 $F$ 多算的 1 次初始草稿，不是抽樣誤差。對照：$T=256$ 時預設三列 TPF 為 2.8764 / 3.4133 / 3.2405（$B=16$ 差 5%），這是改用 1024 的理由。

預設（$B=8$）前 12 次的 $a$：3, 1, 3, 4, 2, 3, 1, 1, 2, 2, 4, 1（與 $T$ 無關）。$\mathbb E[a]$ 誤差 1e-9；$F$ 必須完全相等，TPF 由 $T/F$ 計算（$F$ 相等則 TPF 相等，比較時容許 1e-9）。論文比值核對：100.2/39.5 = 2.537、79.9/39.5 = 2.023、93.8/39.5 = 2.375；$B+B^2$ = 20 / 72 / 272。

---

## 6. 實作計畫（只列與參考頁不同之處）

### 6.1 沿用

單檔 `build/index.html`，從 `tutorials/prediction-targets-k-diff.html` 複製：CSS tokens（含 `--c-par`、`--c-perp`、深色模式 `--ink-3` #9C978C）、appbar、主題切換、`#pm-bar`、章節目錄（`initToc`）、plate／callout／`details.check`／`.tscroll` 表格、KaTeX 0.16.9 與 `renderMath`、`el`／`frame`／`announce`／`widthOf` SVG 輔助、`mulberry32`／`gauss`、`renderTimeline`、`selfTest` 外殼（結果寫入 `document.title` 與 `#selftest`）。不改版面。appbar 編號「No. 9」，eyebrow「No. 9 · 互動教學 · arXiv 2610.28559」。

### 6.2 新增

- 頁面中心視覺「block 條」：HTML + CSS 小元件 `blockStrip(el, {prefix:[…], block:[…], taus:[…]})`，hero、S1、S2、S8 共用。
- 共用玩具：`VOCAB`、`EMB`、`PAIRS`、`tauToy(i,t,lam)`、`denoiser(x,taus,sigma)`、`meanEmb(p)`、`argmax(p)`、`particles(seed,n)`、`flowDecode(g,sigma,lam,K)`；`toy = {seed, sigma, lam}` 由 C1 持有，C4 讀取（C1 改變時呼叫 `renderC4()`）。
- C1：`initC1`／`renderC1`（快取 key `${seed}|${sigma}|${lam}|${K}`）。
- C2：`maskAllowed(mode,N,B,q,k)`、`renderMask(container,mode,N,B)`（C5 也呼叫）、`initC2`。
- C3：`ellToy(q,t)`、`cdfCache`（Map，key = q）、`tauBinned(q,t)`、`initC3`／`renderC3`。
- C4：`c4Compute(g,sigma,lam,s,t)` → `{Ds, delta, X7, XE, Xref, gamma, semi, res7}`、`initC4`／`renderC4`。
- C5：`c5Expected(B,r1,rho)`、`c5Sim(B,r1,rho,T,seed)`（一次算完整個 `hist`，「下一步」只推進顯示指標）、`initC5`／`renderC5`。
- F1：`renderF1()`，資料常數 `TABLE5`（只含 4.6 用到的列）。
- 符號：KaTeX 巨集 `\E` → `\mathbf E`；`\Ex` → `\mathbb E`。
- 數學與動態 DOM：KaTeX 不能放進 SVG `<text>`，所有 SVG 標籤用純 Unicode（τ¹、δ_{s,t} 寫成 δₛ,ₜ 或「δ(s,t)」、𝐄），不用 foreignObject；讀數為純文字（JetBrains Mono）。動態產生且含 TeX 的 HTML（C3 讀數表的表頭、C5 的 Table 2 對照）在更新後呼叫 `renderMath(container)`；live region 只放純文字，不放 KaTeX。
- 事件與效能：所有滑桿與按鈕在 `input` 時以 `requestAnimationFrame` 合併（每幀最多一次計算與重繪）；C1 以 `${seed}|${sigma}|${lam}|${K}` 快取、C4 以 `${n}|${s}|${t}|${sigma}|${lam}|${seed}` 快取。500ms 延遲只用於 live region 的宣告。C1、C3、C4 沒有動畫；C5 的自動執行是唯一的計時器。
- 窄螢幕（390px）：C1、C4 的平面圖與長條在 < 600px 時上下排；C2／C5 遮罩放 `.tscroll`、列標籤欄 `position:sticky`；所有 display math 外包 `overflow-x:auto`；Eq. 6、16、20 一律寫成多行 `aligned`（每項一行）。
- 深色模式：除沿用的 `--ink-3` 外，新增 `--tau-lo`、`--tau-hi`（block 條的 τ 灰階兩端），淺色與深色各自定義（深色不是淺色的反轉），兩端與 `--paper` 的對比 ≥ 3:1；C3 四條線與 F1 三條線以線型區分，顏色只用 `--ink`／`--ink-2`；F1 的 ◆ 用 `--accent`，同時以形狀區分。

### 6.3 selfTest（`?test`）

逐項比對第 5 節：5.1 的 $\lambda=0.5$ 列（$K=1,2,4,64$）與 $\lambda=0,K=2$、$\sigma=0.25,K=2$ 的有支撐比例與 TV；$D_0$ 邊際；5.2 全部；5.3 預設列 $\tau^i(0.5)$ 與 $\dot\tau^i(0.5)$；5.4 第 1、4 列的 $\delta$、誤差與 $\gamma$，以及 res7、semi < 1e-9，另加壓力案例 $\lambda=1$、$s=0.98$、$t=0.99$（res7、semi < 1e-9）；5.5（$T=1024$）前三列與 $(1,1)$ 列的 $F$、TPF、$\mathbb E[a]$，與前 12 個 $a$；另檢查 `scrollWidth <= innerWidth`。建置驗收時以 390px 寬視窗與桌面寬度各跑一次 `?test`。輸出 `selfTest: N passed, M failed`。

### 6.4 研究脈絡與筆記

- `notes/timeline.json` 原樣貼進 `<script type="application/json" id="timeline-data">`（14 筆，1811.03115 … 2610.28559）。分組常數 `TL_GROUP`（不在表中者歸「其他」）：

  | 類別 | arXiv ID | 卡片連到 |
  | --- | --- | --- |
  | 平行解碼 | 1811.03115、2401.10774、2404.19737、2604.07023 | `lafa-parallel-decoding-background.html` |
  | 連續 flow | 2210.02747、2211.15089、2605.10938 | `lafa-embedding-flow-denoiser.html` |
  | Block 擴散與 AR 調整 | 2503.09573、2512.14067 | `lafa-ar-to-block-adaptation.html` |
  | Self-conditioning 與 flow map | 2208.04202、2505.18825、2602.16813、2607.00714 | `lafa-self-conditioning-and-flow-map.html` |
  | 本文 | 2610.28559 | `lafa-lineage-overview.html` |

- S12 延伸閱讀依總覽閱讀順序：`lafa-lineage-overview`、`lafa-parallel-decoding-background`、`lafa-ar-to-block-adaptation`、`lafa-embedding-flow-denoiser`、`lafa-asynchronous-tokenwise-time`、`lafa-self-conditioning-and-flow-map`、`lafa-self-speculative-decoding`、`lafa-scale-and-evaluation`；站內既有：`continuous-flow-lm-overview`、`flow-matching-velocity-target`、`self-conditioning-flow-lm`、`flow-map-few-step-distillation`、`cdlm-noise-schedule`、`cdlm-representation-space`。建置時重新讀 `notes/index` 順序與 frontmatter `title`。

### 6.5 文案

沿用參考頁 5.8 規則。本頁用詞：「block」「token」「prefix」「flow」「flow map」不翻譯；「tokenwise time」第一次出現寫「逐位置時間（tokenwise time）」；「自我推測解碼（self-speculative decoding）」；「草稿」「驗證」「接受長度」；「常見錯誤」（不用「陷阱」）。三種陳述分開：論文結果、作者的解釋（「作者認為」）、本頁推導。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 玩具與論文的距離 | C1、C4 用 2 個位置、5 個 token、精確 denoiser；C3 的 loss 族與 C5 的接受率都是示意值。讀者可能把玩具的步數、TV、TPF 當成 Lafa 的結果，或把 C1 少步時「λ 越大 TV 越大」當成非同步時間有害。 | 每個圖版底部寫「示意值，不是論文數字」與對應的論文數字（Table 5、Table 2）；C1 寫明 Lafa 用 32–64 步時 0.5B 接近、7B 超過 AR SFT，並加 λ 族 ≠ 學到的 τ 的 callout。 |
| R2 | 訓練量不一致 | Sec. 3 寫 5–15 epochs（9–27B），abstract 寫 <30B，Table 4 三個尺度皆「5 epochs / approximately 9B」，Table 6 寫 † 模型 five epochs。Table 4 caption 提到較大 block 由較小 block 接續訓練，15 epochs 可能是累計，論文未明說。 | 頁面照錄兩處，不推測；LO12 寫「9–27B（<30B）」。S9 正文措辭：「Sec. 3 寫訓練 5–15 epochs（9–27B tokens）；Table 4 對三個尺度都寫 5 epochs、約 9B tokens。論文沒有說明兩者的關係。」 |
| R3 | $\sigma$ 不一致 | 訓練 4.0、flow decoding 推論 0.1，self-speculative 草稿未寫；嵌入 norm 相對 $\sigma$ 的大小未給。 | 常見錯誤 6 照錄；C1 的 $\sigma$ 只是玩具尺度，頁面註明與論文的 σ 不可直接比較。 |
| R4 | 數值四捨五入 | Table 2/7 的 7B $B=8$ 平均 TPF 寫 4.3（分項平均 4.37）；Table 8 Lafa $B=16$ 寫 74.5（分項平均 74.57，論文是截斷而不是四捨五入）；Table 3 的完整設定在訓練區塊為 38.0、推論區塊為 38.1。 | 一律照錄論文的平均值，不自行重算。 |
| R5 | 原文筆誤 | App. A.2 證明寫「which is (15)」實為 Eq. 7（等價）；稱 Prop. A.2 為「Theorem A.2」。 | S7 可收合推導以腳註說明一次。 |
| R6 | Fig. 5、Fig. 7、Fig. 10 無數值 | Fig. 5（學到的 $\tau$）與 Fig. 10（confidence decoding）沒有對應表；Fig. 10 caption 與 C.3 正文對 7B/8B 的 block 大小敘述不一致。 | 只用文字描述 Fig. 5；Fig. 10 只寫 C.3 的定性結論；不重繪。 |
| R7 | 未說明的細節 | flow map 蒸餾的 $(s,u,t)$ 分布與訓練量；prefix-free $D_t(x;\varnothing)$ 如何訓練（沒有提 prefix dropout）；輸出 head 是否與 $\mathbf E$ tied；Alg. 2 虛擬碼用 synced interpolant 而訓練用非同步（App. B.1 的「independent uniform estimation」一句是在說明各 block 獨立抽時間，不是在說明這一點）。 | S7、S10 各列一句「論文未說明」；不猜測。 |
| R8 | 「2.54×」與「品質相當」的範圍 | 三項平均；六項平均 $B=16$ 為 58.0 < 58.4；吞吐量只在 batch size 1；Dream $k=4$ TPS 更高但準確率 37.7。 | S9 callout 與 LO11 明寫。 |
| R9 | 「flow 訓練改善 AR」的範圍 | 平均值；部分項目下降；8B 由 Base 起訓且 AR SFT 的 IFEval 只有 38.1；無 seed／信賴區間。 | S9 正文與 LO12 明寫。 |
| R10 | 筆記仍可能更新 | 筆記與 timeline 由平行流程產生，建置前可能增減。 | 建置時重讀 `notes/`；新 arXiv ID 未在 `TL_GROUP` 中歸「其他」並回報。 |
| R11 | C4 對初學者偏難 | two-time denoiser 與 semigroup 是全文最抽象的部分。 | S7 先用兩句字面說明（「δ 是沿路 denoiser 預測的加權平均，越接近終點權重越大」），推導收在可收合區；C4 預設 $s=0$ 只看 $t$ 的效果。 |
| R12 | 相對連結 | 建置位置與上線位置不同，站內連結預覽時失效。 | 接受；預覽只檢查頁內錨點與外部連結。 |
| R13 | 筆記與論文的出入 | `lafa-asynchronous-tokenwise-time` 把「i<j 時 τ^i>τ^j」寫得像約束，但論文是學出來的結果（「generally satisfy」「not manually imposed」）；`lafa-self-speculative-decoding` 第 4 點把非同步時間與由左到右驗證連在一起，論文沒有這樣寫；`lafa-asynchronous-tokenwise-time` 中兩篇先例的 arXiv ID 未驗證；`lafa-embedding-flow-denoiser` 的 2209.15571 是 [25] 的前作。 | 頁面依論文原文；筆記照連結，不引用這幾句；建議筆記流程修正。 |
| R14 | C4 數值精度（審查 P0） | 審查者擔心 ζ 達 10⁶ 時相消。實測最壞情形殘差 ≤ 4.6e-16。 | 不改用閉式 Δζ（頁面與 check.py 須同一演算法，改了也不會改變結論）；selfTest 加壓力案例、門檻 1e-9。若之後放寬 $t$ 上限超過 0.99 再重新評估。 |

---

## 8. 修訂紀錄

- **Round 1（2026-11-19）**：合併 Fidelity 與 Pedagogy 兩份審查，共 30 條（`reviews/round1.md`）。
  - 正確性修正：synced interpolant 與「independent uniform」的理由分開寫；「multi-token prediction effect」移到 AR 能力段並標為作者的解釋；500B 只歸給 Dream、Efficient-DLM；「0.5B 接近、7B 超過」；greedy 等價加上「邏輯上」；理想 δ 與學到的 δ̂ 分開；Alg. 3 的出處；Prop. A.2 的假設寫進 LO9；τ 的 C¹ 條件寫進步驟 3、LO2；Table 8 74.5 是截斷；B+B² 標為推測；Table 8 的解碼方式。
  - 元件：C1 加 λ 族 ≠ 學到的 τ 的 callout 與觀察 (4)；C4 附註改成「代數恆等，不是驗證定理」，並加壓力案例；C5 改用 $T=1024$（check.py 已改並重跑，`verify/out.txt` 已更新，5.5 表已換成新數字）、加入暫停與「跑完」、截斷時的註記；C2 的 KV 讀數、roving tabindex、N 停用；C3 夾住 q 時的註記；每個元件加「對應 LO」。
  - 頁面：S3 開頭說明 D̂ 的來源，推導收進 details；S7 改為 850–950 字並重排順序，蒸餾收進 details；S2 標題與事實卡改用字面描述；hero 補上 greedy、吞吐量、flow 的說明；6.2 新增 KaTeX／SVG、rAF、390px、深色 τ ramp 的規則。
  - 筆記：步驟 3 加上論文的動機句與先例 [26–30]；步驟 11 加上 Sec. 4 的平行解碼背景；S6 加上 [36]；S11 文字提及 Next-Block、WeDLM；筆記與論文的出入記在 R13。
  - 未完全照審查意見做的項目：P1（C4 相消）只加測試、不改演算法（理由見 R14）。
- **Readiness（2026-11-19，GO）非阻擋意見處理**：S9 內容加入訓練 token 量（LO12）；R2 寫出頁面措辭；5.3 寫出 $q_1=2,\Delta q=-0.5$ 的四組值；5.5 註明 $F$ 完全相等、TPF $=T/F$；符號表加 $\bar L$ 與 sg；S2 加入 τ 的條件；S9 加 F1 的預期讀法。Eq. 6、16、20 的多行 aligned 維持不變。
