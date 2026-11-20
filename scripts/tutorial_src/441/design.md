# 設計文件：〈Is Conditional Generative Modeling all you need for Decision-Making?〉互動教學

- 論文：Anurag Ajay*、Yilun Du*、Abhi Gupta*、Joshua Tenenbaum、Tommi Jaakkola、Pulkit Agrawal（* 同等貢獻），arXiv 2211.15657（v4），ICLR 2023。資料庫編號 No. 441。方法名稱 **Decision Diffuser**（以下 DD）。
- 原文：`tut/441/src/paper.txt`（PDF 同目錄）。`tut/` = `scripts/tutorial_src/`。「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」「Alg. 1」皆指原文編號。論文有編號的式子為 Eq. 1–9；Sec. 3.4 的訓練損失沒有編號，本文稱「Sec. 3.4 損失」。
- 產出：`tut/441/build/index.html`（單檔）。上線位置 `tutorials/decision-diffuser.html`（slug `decision-diffuser`）。站內連結以上線位置為準：論文 `../index.html#paper-441`，筆記 `<slug>.html`。
- 驗算腳本（與頁面同一演算法）：`tut/441/verify/core.py`（C1、C2 的數學）、`tables.py`（論文表格與 C3 的即時計算）、`check.py`（輸出 `check_out.txt`，第 5 節全部數字由此而來）。
- 研究筆記：`tut/441/notes/`（總覽 `decision-diffuser-lineage-hub` ＋ 6 篇主題筆記 ＋ `timeline.json` 15 筆），見 6.4。
- 標記慣例：**〔本頁推導〕** = 本設計自行推導、論文沒有寫出；**推定** = 為玩具可視化選定、論文沒有寫明的值；**本頁玩具** = 元件中的數字，不是論文模型的結果，也不重現論文的任何表格；「作者認為／作者假設」= 論文的解釋或假設，不是實驗結果。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

離線強化學習（offline RL）要從一份固定的、帶有獎勵標記的軌跡資料中得到高報酬策略。既有方法多以時序差分（TD）學習估計價值函數，再以動態規劃把次佳軌跡「拼接」成較好的軌跡；作者指出價值估計受「致命三元組」（函數近似、off-policy 學習、bootstrapping）影響而不穩定，離線設定還需要額外的限制避免策略離開資料分布（Sec. 1、Sec. 2.1）。本文改把決策寫成條件生成建模 $\max_\theta\mathbb E_{\tau\sim\mathcal D}[\log p_\theta(x_0(\tau)\mid y(\tau))]$（Eq. 4）：$x_0(\tau)$ 是一段**狀態**序列（Eq. 6），$y(\tau)$ 是關於軌跡的資訊（報酬、滿足的限制或展示的技能）。動作由逆動力學模型 $a_t=f_\phi(s_t,s_{t+1})$ 取得（Eq. 7）。取樣時用無分類器引導（classifier-free guidance，Eq. 8）與低溫取樣，作者假設這樣能「隱式地執行動態規劃」，取出資料中最好的行為（Sec. 1、App. A）。D4RL 運動控制 9 個資料集的平均正規化分數 DD 為 81.8，CQL 77.6、IQL 77、Diffuser 75.3、TT 78.9（Table 1）；9 列中 DD 有 4 列最高（`tables.py`）。條件改為限制或技能時，同一個噪聲模型可在測試時以 Eq. 9 組合多個條件：Kuka 積木多重限制平均成功率 63.8（Table 3），四足步態可組合出訓練資料中沒有的步態轉換（Fig. 6 的軌跡、Fig. 7 的分類器讀數）。

### 1.2 研究動機（Sec. 1、Sec. 2.1）

1. **價值估計不穩定**：TD 方法以 Eq. 1 估計 $Q$；函數近似、off-policy、bootstrapping 三者同時出現時不穩定（Sutton & Barto 的 deadly triad）。
2. **離線分布偏移**：直接用 TD 會使策略的狀態訪問分布 $d^{\pi_\phi}(s)$ 偏離資料的 $d^\mu(s)$；離線方法加入 $D(d^{\pi_\phi}\|d^\mu)$ 型限制，帶來額外超參數與實作技巧（Sec. 2.1）。
3. **條件生成模型已能組合資料**：文字到影像模型能組合訓練資料產生新樣本；作者因此問能否不估計價值就拼接出最佳軌跡（Sec. 1）。

### 1.3 貢獻（Sec. 1 末段原文列出三點）

(i) 說明條件生成建模是離線決策的有效工具；(ii) 以無分類器引導加低溫取樣取代動態規劃，得到報酬最大化的軌跡；(iii) 在推論時彈性地組合限制與技能。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S3 末尾以可收合表格提供，< 600px 為堆疊卡片；含論文的符號衝突）

| 符號 | 意義 | 衝突與頁面寫法 |
| --- | --- | --- |
| $s_t,a_t,r_t$ | 環境時間 $t$ 的狀態、動作、獎勵 | 下標 $t$ 一律是環境時間 |
| $\gamma\in[0,1)$ | 折扣因子 | — |
| $R(\tau)=\sum_{k\ge0}\gamma^kr_k$ | 軌跡的報酬（return） | 論文也用 $R$ 表示獎勵函數 $S\times A\times S\to\mathbb R$，以及 App. A.2 的大圓半徑 $R=1$；頁面在 S10 註明「此處 $R,r$ 是半徑」 |
| $\mathcal D$ | 離線資料集 | — |
| $k\in\{1,\dots,K\}$ | 擴散步（前向加噪的步數），$K=100$（App. B） | Sec. 2.1 的報酬求和也用 $k$；頁面在 S1 求和處改寫成 $\sum_{t\ge0}\gamma^tr_t$ 並註明原文用 $k$ |
| $x_k(\tau)$ | 擴散步 $k$ 的帶噪狀態序列；$x_0$ 是乾淨序列 | — |
| $\alpha_k,\ \bar\alpha_k=\prod_{i\le k}\alpha_i$ | 前向過程的排程（Sec. 2.2）；$\bar\alpha_k$ 是本頁的定義（DDPM 慣例） | 論文寫 $q(x_{k+1}\mid x_k)$ 用 $\alpha_k$，照字面 $x_k$ 對應 $\prod_{i<k}\alpha_i$，差一個索引，頁面 S3 加註。與低溫係數 $\alpha$（無下標）同字母；排程一律帶下標，低溫係數稱「低溫係數 α」；動態文字不出現排程符號 |
| $\epsilon,\ \epsilon_\theta$ | 標準高斯噪聲、噪聲預測網路 | 讀數與 SVG 用 Unicode「ε」；$\hat\epsilon$ 只出現在 KaTeX 中，動態文字寫「引導後的噪聲預測」。Alg. 1 第 7 行把無條件預測寫成 $\epsilon_\theta(x_k,k)$，Eq. 8 寫成 $\epsilon_\theta(x_k,\text{Ø},k)$，是同一個量 |
| $y(\tau)$、Ø | 條件；Ø 為「沒有條件」的空值 | — |
| $\omega$ | 引導強度（guidance scale） | App. B 寫成 $s\in\{1.2,1.4,1.6,1.8\}$；頁面全用 ω 並註明 |
| $\alpha\in[0,1)$ | 低溫係數，$x_{k-1}\sim\mathcal N(\mu_{k-1},\alpha\Sigma_{k-1})$（Sec. 3.4） | App. C 的 Table A1 含 α = 1.0，與 $[0,1)$ 不一致，照錄 |
| $H$、$C$ | 規劃視界（序列長度）、歷史長度 $C=20$ | $C_i$ 另指第 $i$ 個限制的集合 |
| $f_\phi$ | 逆動力學模型 | — |
| $\beta\sim\mathrm{Bern}(p)$ | 訓練時是否丟棄條件，$p=0.25$ | App. G 的 $p$ 是隨機動作機率；App. J 的 $\beta=1$ 是 Liu et al. 的係數；頁面在各處就地註明 |
| $h$ | Alg. 1 的歷史佇列 | Sec. 3.4 的 $z\in\mathbb R^h$ 中 $h$ 是維度 |

#### 步驟 1：問題設定（Sec. 2.1）

- MDP $\langle\rho_0,\mathcal S,\mathcal A,\mathcal T,\mathcal R,\gamma\rangle$；隨機策略 $\pi:\mathcal S\to\Delta_{\mathcal A}$；目標 $\pi^*=\arg\max_\pi\mathbb E_{\tau\sim p_\pi}[R(\tau)]$。
- $Q^*(s,a):=\mathbb E_{\tau\sim p_{\pi^*}}[R(\tau)\mid s_0=s,a_0=a]$；TD 損失
  $$\mathcal L_{TD}(\theta)=\mathbb E_{(s,a,r,s')\in\mathcal D}\big[(r+\gamma\max_{a'}Q_\theta(s',a')-Q_\theta(s,a))^2\big]\tag{Eq. 1}$$
  連續動作另需策略 $\pi_\phi$ 最大化 $\mathcal J(\phi)=\mathbb E_{s\in\mathcal D,a\sim\pi_\phi}[Q(s,a)]$（Eq. 2）。
- 離線：資料由未知行為策略 $\mu$ 收集；限制 $D(d^{\pi_\phi}\|d^\mu)$。作者主張 DD 不估計 Q，且生成模型以最大概似訓練，因此「不面臨分布偏移的風險」（Sec. 2.1，作者的主張）。
- 頁面字面解釋：「bootstrapping＝用模型自己的估計當作訓練目標（Eq. 1 的 $\max_{a'}Q_\theta(s',a')$）」；「off-policy＝用別的策略收集的資料學目前策略的價值」。

#### 步驟 2：擴散模型背景（Sec. 2.2）

- 前向 $q(x_{k+1}\mid x_k)=\mathcal N(\sqrt{\alpha_k}x_k,(1-\alpha_k)I)$；反向 $p_\theta(x_{k-1}\mid x_k)=\mathcal N(\mu_\theta(x_k,k),\Sigma_k)$；$x_K\sim\mathcal N(0,I)$。
- 簡化損失 $\mathcal L_{denoise}(\theta)=\mathbb E_{k,x_0,\epsilon}\|\epsilon-\epsilon_\theta(x_k,k)\|^2$（Eq. 3）；$\mu_\theta$ 可由 $\epsilon_\theta$ 算出。
- 〔本頁推導，標準結果〕閉式 $x_k=\sqrt{\bar\alpha_k}x_0+\sqrt{1-\bar\alpha_k}\,\epsilon$；分數（score）$\nabla_x\log p(x)$ 是讓機率密度增加最快的方向（一維 $\mathcal N(0,1)$ 的分數為 $-x$）；對最佳（訓練完美）的預測器，$\epsilon_\theta(x_k,k)=-\sqrt{1-\bar\alpha_k}\,\nabla_{x_k}\log p_k(x_k)$，其中 $p_k$ 是 $x_k$ 的邊際分布。Sec. 2.2 寫「$\epsilon_\theta\propto\nabla\log p$」省略負號；App. D 寫 $\nabla\log q\propto-\epsilon_\theta$。頁面採帶負號的版本並註明。
- 兩種條件化（Sec. 2.2）：分類器引導 $\epsilon_\theta(x_k,k)-\omega\sqrt{1-\bar\alpha_k}\nabla_{x_k}\log p(y\mid x_k)$，需另訓練帶噪分類器；無分類器引導 $\epsilon_\theta(x_k,k)+\omega(\epsilon_\theta(x_k,y,k)-\epsilon_\theta(x_k,k))$，無條件模型以 $\epsilon_\theta(x_k,\text{Ø},k)$ 表示。$\bar\alpha_k$ 在 Sec. 2.2 未定義，頁面補上定義。

#### 步驟 3：決策＝條件生成（Sec. 3，Eq. 4–5）

$$\max_\theta\ \mathbb E_{\tau\sim\mathcal D}[\log p_\theta(x_0(\tau)\mid y(\tau))]\tag{Eq. 4}$$
$$q(x_{k+1}(\tau)\mid x_k(\tau)),\qquad p_\theta(x_{k-1}(\tau)\mid x_k(\tau),y(\tau))\tag{Eq. 5}$$
$y$ 的例子：報酬、滿足的限制、展示的技能（Fig. 1 三列）。這是監督式學習，不含 Bellman 備份。

#### 步驟 4：只擴散狀態，動作由逆動力學取得（Sec. 3.1，Eq. 6–7，Table 2，App. F）

$$x_k(\tau):=(s_t,s_{t+1},\dots,s_{t+H-1})_k\tag{Eq. 6}\qquad a_t:=f_\phi(s_t,s_{t+1})\tag{Eq. 7}$$
- 兩個理由（Sec. 3.1）：(1) 狀態多為連續，動作型態多樣且常為離散；(2) 以關節力矩表示的動作序列頻率高、不平滑，較難預測。$x_k(\tau)$ 視為二維陣列，每欄一個時間步。
- $f_\phi$ 與 $p_\theta$ 用同一份離線資料訓練。
- Table 2（Hopper，正規化分數）：Diffuser／CondDiffuser／CondMLPDiffuser／DD：Med-Expert 107.6／111.3／105.6／111.8±1.6；Medium 58.5／66.3／54.1／79.3±3.6；Med-Replay 96.8／76.5／66.5／100±0.7。CondDiffuser（對狀態與動作一起擴散、使用無分類器引導）在 3 個中有 2 個優於 Diffuser（Med-Replay 低 20.3）；DD 在 3 個都優於 CondDiffuser（差 +0.5、+13.0、+23.5，`tables.py`）。CondMLPDiffuser（只對目前動作擴散）在擴散模型中最差。限定：Med-Expert 的 +0.5 在 DD 標準誤 ±1.6 之內（基準沒有標準誤）；DD 與 CondDiffuser 同時改變了擴散對象（只有狀態）與動作來源（逆動力學），Table 2 不能把差異單獨歸給其中一個；單獨對應「動作平滑度」解釋的是 App. F 的 Table A2。
- App. F Table A2（Block Push 成功率，BC／CondDiffuser／DD）：位置控制 57.3±1.2／87.3±3.1／87.8±2.8；力矩控制 55.2±1.5／71.8±3.4／84.7±2.2。動作平滑（位置控制）時兩者相近，動作含高頻成分（力矩控制）時 CondDiffuser 較差。

#### 步驟 5：無分類器引導與低溫取樣（Sec. 3.2、3.4，Eq. 8，App. C、K）

- 為什麼不用分類器引導：若 $y$ 是報酬，訓練 $p_\phi(y\mid x_k)$ 等於估計 Q 函數，需要另一個動態規劃程序（Sec. 3.2）。直接以報酬訓練條件模型則會混入資料中的次佳行為（作者用語 polluted）；作者因此用無分類器引導加低溫取樣，「取出資料中高概似的軌跡」，作者發現這些軌跡對應資料中最好的行為（Sec. 3.2）。
- 訓練（Sec. 3.4 損失）：
  $$\mathcal L(\theta,\phi)=\mathbb E_{k,\tau\in\mathcal D,\beta\sim\mathrm{Bern}(p)}\big[\|\epsilon-\epsilon_\theta(x_k(\tau),(1-\beta)y(\tau)+\beta\text{Ø},k)\|^2\big]+\mathbb E_{(s,a,s')\in\mathcal D}\big[\|a-f_\phi(s,s')\|^2\big]$$
  $k\sim\mathcal U\{1,\dots,K\}$；以機率 $p$（$=0.25$，App. B）丟棄條件，因此**同一個網路**同時學到條件與無條件噪聲；逆動力學以單一轉移訓練。$y$ 為純量或 one-hot，經 MLP 投影到 $z\in\mathbb R^h$；$y=\text{Ø}$ 時把 $z$ 的元素設為 0（Sec. 3.4）。
- 取樣：
  $$\hat\epsilon:=\epsilon_\theta(x_k,\text{Ø},k)+\omega\big(\epsilon_\theta(x_k,y,k)-\epsilon_\theta(x_k,\text{Ø},k)\big)\tag{Eq. 8}$$
  〔本頁推導：代入 Eq. 8〕$\omega=0$ 得無條件預測、$\omega=1$ 得純條件預測；App. B 用 1.2–1.8。
- 〔本頁推導〕把最佳噪聲預測代入（步驟 2 的關係）並用貝氏定理 $\nabla\log p_k(x\mid y)-\nabla\log p_k(x)=\nabla\log p_k(y\mid x)$：$\hat\epsilon=-\sqrt{1-\bar\alpha_k}\,\nabla_x\big[\log p_k(x)+\omega\log p_k(y\mid x)\big]$。在 $k=0$ 這是 $p(x)\,p(y\mid x)^\omega$ 的分數：$\omega>1$ 加強「隱式分類器」$p(y\mid x)$。但各個 $k>0$ 的 $p_k(x)p_k(y\mid x)^\omega$ 一般不是同一個分布加噪後的結果，所以取樣結果不必等於 $p(x)p(y\mid x)^\omega$（C1 以玩具數字呈現兩者的差距）。App. K 主張分類器引導（此處即 Q 函數引導）與無分類器引導「理論上等價」但沒有證明，本頁未驗證；上式只在分數層面把兩者對應起來。App. K 說明無分類器引導實務上較好的兩點理由：條件模型在訓練時就學條件生成；離線學到的 Q 函數可能對 OOD 動作給出過高估計（作者的解釋）。頁面把這段推導放進標「進階」的 `<details>`。
- 低溫取樣（Sec. 3.4、Alg. 1 第 4、9 行）：$x_K\sim\mathcal N(0,\alpha I)$，每步 $x_{k-1}\sim\mathcal N(\mu_{k-1},\alpha\Sigma_{k-1})$，App. B 取 $\alpha=0.5$。Table A1（Hopper-Medium-Expert）：α = 0：104.3±0.7；α = 0.5：111.8±1.6；α = 1.0：107.1±3.5。作者的解讀：α = 0 去除取樣的熵、報酬較低；α = 1.0 報酬變異較大（App. C）。

#### 步驟 6：規劃迴圈（Sec. 3.2，Alg. 1，Fig. 3，App. E）

Alg. 1 原文（頁面兩欄對照，左程式右中文）：輸入 $\epsilon_\theta,f_\phi,\omega,C,y$ → 初始化長度 $C$ 的佇列 $h$ → 迴圈：觀察 $s$ 並放入 $h$；$x_K\sim\mathcal N(0,\alpha I)$；對 $k=K\dots1$：$x_k[:\mathrm{length}(h)]\leftarrow h$（讓計畫與歷史一致）、Eq. 8、$(\mu_{k-1},\Sigma_{k-1})\leftarrow\mathrm{Denoise}(x_k,\hat\epsilon)$、$x_{k-1}\sim\mathcal N(\mu_{k-1},\alpha\Sigma_{k-1})$ → 由 $x_0$ 取 $(s_t,s_{t+1})$，執行 $a_t=f_\phi(s_t,s_{t+1})$。
- 這是滾動視界控制（receding-horizon control）：每個環境步重新規劃，只執行第一個動作。
- 歷史以替換（inpainting）寫入每個擴散步。論文沒有說明 $H$ 欄中前 $C$ 欄是否是歷史、$s_t$ 在第幾欄（R3）。
- App. E（作者報告，只在 Hopper-Medium-Expert 上量測；性能與時間的關係只有 Fig. A2，沒有數據表）：每個動作需 100 個反向步，平均 1.26 s；以上一步的計畫暖啟動（先前向加噪固定步數，再反向同樣步數）可降到 40 步（0.48 s，性能不變）、20 步（0.21 s，損失很小）、5 步（0.06 s，性能損失小於 20%）。

#### 步驟 7：報酬以外的條件與組合（Sec. 3.3，Eq. 9，App. A.2、D、I.2、J）

- 報酬：$y=R(\tau)$，正規化到 $[0,1]$，取樣時取 $R(\tau)=1$，不使用任何 Q 值。
- 限制：$y=\mathbf 1(\tau\in C_i)$（one-hot）；技能：$y=\mathbf 1(\tau\in B_i)$，$B_i$ 為技能 $i$ 的示範集合。訓練資料中每條軌跡只滿足一個限制／展示一個技能。
- 組合（Liu et al. 2022）：
  $$\hat\epsilon:=\epsilon_\theta(x_k,\text{Ø},k)+\omega\sum_{i=1}^n\big(\epsilon_\theta(x_k,y^i,k)-\epsilon_\theta(x_k,\text{Ø},k)\big)\tag{Eq. 9}$$
  條件獨立的寫法在論文中不一致：Sec. 3.3（Eq. 9 下方）寫「給定狀態軌跡 $x_0(\tau)$」，App. D 寫「給定 $x_k(\tau)$」；下面的推導對每個 $k$ 都需要給定 $x_k$ 的版本，而給定 $x_0$ 的條件獨立一般不蘊含給定 $x_k$ 的條件獨立〔本頁推論〕。App. D 推導：假設 $\{y^i\}$ 在給定 $x_k(\tau)$ 時條件獨立，$q(x_k\mid\{y^i\})\propto q(x_k)\prod_i\frac{q(x_k\mid y^i)}{q(x_k)}$，取對數梯度即得。作者觀察此假設不必嚴格成立，但組合必須**可行**（存在同時滿足所有條件的 $x_0$）；不可行時產生不連貫的行為（App. D，以影片呈現）。資料需包含滿足每個條件的多種方式；組合出的狀態序列中相鄰狀態對 $(s_t,s_{t+1})$ 須來自示範資料的分布，否則逆動力學可能出錯（App. D）。
- NOT（App. J，$\beta=1$）：$\hat\epsilon=\epsilon_\theta(\text{Ø})+\omega\big(\sum_{i\ne j}(\epsilon_\theta(y^i)-\epsilon_\theta(\text{Ø}))-(\epsilon_\theta(y^j)-\epsilon_\theta(\text{Ø}))\big)$（省略 $x_k,k$）。DD 沒有各條件的顯式密度，不支援 OR（App. J、M）。
- 簡單基準（App. I.2）：把兩個 one-hot 相加當成一個條件輸入 $\epsilon_\theta(x_k,\text{one-hot}(i)+\text{one-hot}(j),k)$「災難性地」無法產生任何步態，因為訓練時沒看過這個輸入。
- App. A.2（線性系統導航，$H=50$，1000 條專家軌跡，各滿足 $\|s_T\|\le R$（$R=1$）或 $\|s_T\|\ge r$（$r=0.7$）；$s_T$ 為最終二維位置）：單一限制成功率 100%；組合限制 $r\le\|s_T\|\le R$ 為 91.3%（±2.6%，5 個種子的標準誤）。

#### 步驟 8：架構與實作細節（Sec. 3.4，App. B）

| 項目 | 值（App. B） |
| --- | --- |
| $\epsilon_\theta$ | temporal U-Net（沿用 Diffuser 程式碼），6 個殘差塊；每塊兩個時間卷積，各接 group norm，最後 Mish。時間步與條件的嵌入各 128 維，由各自的 2 層 MLP（256 隱藏單元、Mish）產生，串接後加到每塊第一個卷積的激活 |
| $f_\phi$ | 2 層 MLP，512 隱藏單元，ReLU |
| 步態分類器 | 3 層 MLP，1024 隱藏單元，ReLU；Adam 2e−4、batch 64、1e6 步（App. I.1 另寫 2e6 步，R4） |
| 最佳化 | $\epsilon_\theta$、$f_\phi$：Adam，學習率 2e−4，batch 32，2e6 步 |
| 丟棄條件機率 $p$ | 0.25 |
| 擴散步 $K$ | 100 |
| 視界 $H$ | D4RL 運動控制 100、Kitchen 56、Kuka 128、Unitree 56、示意例 50、Block Push 60 |
| 引導強度 | $\{1.2,1.4,1.6,1.8\}$，依任務而定 |
| 低溫係數 α | 0.5 |
| 歷史長度 $C$ | 20 |

#### 步驟 9：實驗（Sec. 4，Table 1–3，Fig. 4、6、7，App. A.1、G、H、I）

- **離線 RL（Table 1，正規化分數，5 個種子的平均；DD 附標準誤）**：運動控制平均 BC 51.9、CQL 77.6、IQL 77、DT 74.7、TT 78.9、MOReL 72.9、Diffuser 75.3、DD 81.8；Kitchen 平均 BC 44.8、CQL 51.2、IQL 48.7、DD 61（Mixed 65±2.8、Partial 57±2.5；其餘方法無 Kitchen 數字）。逐列：DD 在 Hopper-ME、HalfCheetah-M、Walker2d-M、Hopper-MR 與兩個 Kitchen 共 6/11 列最高；不是最高的 5 列為 HalfCheetah-ME（TT 95、CQL 91.6 > 90.6）、Walker2d-ME（IQL 109.6 > 108.8）、Hopper-M（MOReL 95.4 > 79.3）、HalfCheetah-MR（39.3，排第 6；CQL 45.5）、Walker2d-MR（TT 82.6、CQL 77.2 > 75）（`tables.py`）。論文的文字是「either competitive or outperforms many of our offline RL baselines」，頁面採同樣強度，不寫「全面優於」。
- **隱式動態規劃（App. A.1）**：Maze2D-open，獎勵為到 C 點的負距離；資料為 500 條 A→B 與 500 條 B→C 軌跡，最大長度 50；測試從 A 出發要盡快到 C。論文稱 DD 能拼接訓練軌跡形成（近）直線軌跡（Fig. A1，只有圖）；原文該句寫「from point A to point B」，與任務描述不一致，照錄並註明（R5）。這是作者的假設與單一例子，沒有定量結果。
- **限制滿足（Sec. 4.2，Table 3，App. H）**：Kuka 積木 4 塊，限制形如 BlockHeight(i) > BlockHeight(j)；10,000 條專家示範，每條滿足一個限制；計畫長度 128，最大軌跡長度 384，以閉迴路控制器產生 7 個關節的動作。成功率（Diffuser／DD）：單一限制 Stacking 45.6±3.1／58.0±3.1，Rearrangement 58.9±3.4／62.7±3.1，平均 52.3／60.4（Rearrangement 62.7±3.1 對 58.9±3.4，區間重疊；論文寫「greater success rate」，沒有檢定。App. H 稱 Rearrangement 為「out-of-distribution challenge」，因為要求的配置可能不是單一塔）；多重限制（$i>j>k$ 三塊，資料中沒有示範）DD Stacking 60.3±3.1、Rearrangement 67.2±3.1，平均 63.8。BCQ 與 CQL（目標條件版本）成功率 0.0。
- **技能組合（Sec. 4.3，Fig. 6、7，App. I）**：Fig. 6 顯示一條從 bound 開始、以 pace 結束的軌跡（影格）。Unitree-go-running，步態 bound／pace／trot，每種 2500 條示範；狀態 56 維、動作 12 維、最大長度 250。步態分類器以長度 10 的狀態子序列（串接成 560 維）預測步態，以 MixUp 式增強訓練（70% 真實、30% 合成）。1000 條生成軌跡中各步態被分類的時間比例（Trot／Pace／Bound，%）：只條件 Bound 0.8／1.0／98.2；只 Pace 1.4／97.7／0.9；Bound + Pace 1.4／38.5／60.1（Fig. 7 表格）。正文寫成「bounding 38.5%、pacing 60.1%」，與表格欄位相反（R6），頁面照表格欄位並加註。App. I：Trot + Pace 55.6／43.6／0.8；Trot + Bound 51.3／0.9／47.8。
- **隨機動力學（App. G，Table A3）**：Block Push 力矩控制，每步以機率 $p$ 改執行均勻隨機動作；成功率（BC／DD／Diffuser／CQL）：$p=0$：55.2±1.5／84.7±2.2／72.4±1.4／73.2±2.3；0.05：49.3±3.6／77.3±3.1／63.2±2.9／61.8±3.7；0.10：25.8±3.8／53.2±4.1／52.3±4.6／51.2±4.3；0.15：15.1±4.3／41.3±4.9／41.6±5.1／42.2±5.5。$p\ge0.10$ 時 DD 與 Diffuser、CQL 相近（$p=0.15$ 時 DD 41.3 略低於 CQL 42.2 與 Diffuser 41.6，區間重疊）。作者的解釋：報酬條件策略無法區分「好動作帶來的高報酬」與「環境隨機性帶來的高報酬」。

#### 步驟 10：限制（App. M）

只處理完全可觀察的 MDP（POMDP 可能造成 self-delusion）；只離線、不能探索；只在狀態型環境實驗；只支援 AND 與 NOT；隨機動力學下優勢消失；資料量少時容易過擬合。

#### 常見錯誤（頁面 S13 逐條列出）

1. 只訓練條件模型：Eq. 8 需要 $\epsilon_\theta(x_k,\text{Ø},k)$；訓練時必須以 $p=0.25$ 丟棄條件。
2. 用 $y=0$ 代表 Ø：報酬正規化到 $[0,1]$，$y=0$ 是合法的報酬值；論文把投影後的 $z$ 設為 0 來表示 Ø（〔本頁推論〕：$y=0$ 經 MLP 一般不會得到零向量，兩者因此可區分）。
3. 以為 $\omega=1$ 已有引導：代入 Eq. 8，$\omega=1$ 是純條件模型〔本頁推導〕；論文用 1.2–1.8（App. B 記為 $s$）。
4. 低溫只套在每一步：Alg. 1 的起點 $x_K\sim\mathcal N(0,\alpha I)$ 也縮放；α = 0 時取樣完全確定（Table A1：104.3，低於 α = 0.5 的 111.8）。
5. 歷史只在最後寫入：Alg. 1 第 6 行在每個擴散步都替換。
6. 組合時把多個 one-hot 相加輸入：App. I.2 顯示會失敗；應依 Eq. 9 對每個條件分別算噪聲再相加差值。
7. 混淆兩個 α：排程 $\alpha_k$ 帶下標，低溫係數 α 不帶下標。

---

## 2. 學習目標（驗收測試）

每章末以 `<details>`（summary「檢查理解」）呈現該章的題目與參考答案（原文照搬）。**每題參考答案的內容必須寫在該章正文段落中**，不只出現在元件、表格、圖或答案框。< 600px 時本表在頁面附錄中改為堆疊卡片（題目、答案、章節三行）。

| # | 學習目標 | 檢查題 | 參考答案 | 章 |
| --- | --- | --- | --- | --- |
| LO1 | 說明 TD 式離線 RL 的兩個困難 | 作者指出以 TD 學習做離線 RL 有哪兩類困難？ | 價值估計受函數近似、off-policy、bootstrapping（致命三元組）影響而不穩定；策略的狀態分布 $d^{\pi}$ 偏離資料的 $d^{\mu}$，需要 $D(d^\pi\|d^\mu)$ 型限制與額外調參（Sec. 1、2.1） | S2 |
| LO2 | 寫出 DD 的訓練目標 | DD 最大化什麼？$x_0(\tau)$ 與 $y$ 是什麼？ | $\max_\theta\mathbb E_{\tau\sim\mathcal D}[\log p_\theta(x_0(\tau)\mid y(\tau))]$（Eq. 4）；$x_0(\tau)$ 是從軌跡取出的一段狀態序列（Eq. 6）；$y$ 可為報酬、滿足的限制或展示的技能 | S5 |
| LO3 | 噪聲預測與分數 | Eq. 3 訓練什麼？$\epsilon_\theta$ 與分數 $\nabla\log p_k$ 的關係在什麼條件下成立？ | 預測加到 $x_0$ 上的噪聲，損失 $\|\epsilon-\epsilon_\theta(x_k,k)\|^2$；對最佳（訓練完美）的預測器，$\epsilon_\theta(x_k,k)=-\sqrt{1-\bar\alpha_k}\nabla\log p_k(x_k)$，$p_k$ 是 $x_k$ 的邊際分布（論文 Sec. 2.2 只寫 ∝，App. D 帶負號） | S4 |
| LO4 | 為什麼只擴散狀態 | DD 的 $x_k(\tau)$ 包含什麼？動作怎麼來？作者的理由與 Table 2 的證據？ | 只含狀態 $(s_t,\dots,s_{t+H-1})$（Eq. 6）；$a_t=f_\phi(s_t,s_{t+1})$（Eq. 7）；兩個理由：動作型態多樣、常為離散；力矩序列高頻不平滑。Hopper 三個資料集 DD 111.8／79.3／100 對 CondDiffuser 111.3／66.3／76.5；Med-Expert 的 +0.5 在 DD 標準誤 ±1.6 之內 | S6 |
| LO5 | 無分類器引導的訓練與取樣 | 一個網路如何同時給出條件與無條件噪聲？Eq. 8 中 ω = 0、1 各代表什麼？ | 訓練時以機率 $p=0.25$ 把條件換成 Ø（$z$ 設為 0）；$\hat\epsilon=\epsilon_\theta(\text{Ø})+\omega(\epsilon_\theta(y)-\epsilon_\theta(\text{Ø}))$；代入 Eq. 8：ω = 0 得無條件預測、ω = 1 得純條件預測〔本頁推導〕；論文用 1.2–1.8 | S7 |
| LO6 | ω 的作用（本頁推導，進階） | 以分數表示，引導後的噪聲預測對應哪個量？為什麼取樣結果不一定等於 $p(x)p(y\mid x)^\omega$？ | $\hat\epsilon=-\sqrt{1-\bar\alpha_k}\nabla[\log p_k(x)+\omega\log p_k(y\mid x)]$，ω > 1 加強隱式分類器 $p_k(y\mid x)$；各擴散步的 $p_k(x)p_k(y\mid x)^\omega$ 一般不是同一個分布加噪後的結果 | S7 |
| LO7 | 低溫取樣 | 低溫取樣改了哪兩處？論文的 α 與 Table A1 結果？ | $x_K\sim\mathcal N(0,\alpha I)$ 與 $x_{k-1}\sim\mathcal N(\mu,\alpha\Sigma)$；α = 0.5；Hopper-ME：α = 0 為 104.3、0.5 為 111.8、1.0 為 107.1（變異較大） | S8 |
| LO8 | 規劃迴圈 | Alg. 1 每個環境步做什麼？$C$ 是多少？每個動作的計算時間？ | 觀察狀態放入長度 $C=20$ 的歷史；從 $\mathcal N(0,\alpha I)$ 出發，每個擴散步先把歷史寫入計畫、用 Eq. 8 去噪；由 $x_0$ 取 $(s_t,s_{t+1})$，執行 $f_\phi(s_t,s_{t+1})$，下一步重新規劃。作者報告（Hopper-Medium-Expert，App. E）：100 步約 1.26 s；暖啟動 40 步 0.48 s 性能不變 | S9 |
| LO9 | 條件組合 | Eq. 9 依賴什麼假設？論文兩處的寫法有何不同？組合失敗的條件？支援哪些邏輯組合？ | 各條件條件獨立：Sec. 3.3 寫「給定 $x_0(\tau)$」，App. D 寫「給定 $x_k(\tau)$」，推導需要後者；作者觀察假設不必嚴格成立；組合必須可行，不可行時行為不連貫；支援 AND（Eq. 9）與 NOT（App. J），不支援 OR；把 one-hot 相加輸入會失敗（App. I.2） | S10 |
| LO10 | 離線 RL 結果 | Table 1 運動控制與 Kitchen 平均？DD 是否每列最高？ | 運動控制 DD 81.8，CQL 77.6、IQL 77、TT 78.9、Diffuser 75.3；Kitchen DD 61，CQL 51.2；9 個運動控制資料集中 DD 只有 4 個最高，例如 Hopper-Medium MOReL 95.4 高於 DD 79.3 | S11 |
| LO11 | 限制與技能實驗 | Kuka 積木單一與多重限制的平均成功率？Bound + Pace 組合時各步態比例？ | 單一：DD 60.4、Diffuser 52.3（Rearrangement 62.7±3.1 對 58.9±3.4，區間重疊；App. H 稱 Rearrangement 為分布外的挑戰）；多重：DD 63.8（資料中沒有示範）；Bound + Pace：Bound 60.1%、Pace 38.5%、Trot 1.4%（Fig. 7 表格；正文把兩者寫反） | S12 |
| LO12 | 限制 | 在什麼環境中 DD 的優勢消失？作者如何解釋？另列兩項限制。 | 隨機動力學：$p=0.05$ 時 DD 77.3 高於 Diffuser 63.2、CQL 61.8，$p\ge0.10$ 時三者相近；報酬條件策略無法區分好動作與隨機性帶來的高報酬；另有只支援完全可觀察 MDP、只離線不探索、不支援 OR、資料少時過擬合等（App. G、M） | S15 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），元件放加寬圖版（最大 1080px）。章節編號 JetBrains Mono（`01 · …`），標題用陳述句。視覺記號：**條件 chip 列**「報酬 · 限制 · 技能」（Fig. 1 的三種 $y$），在 Hero、S5、S10 出現，當前章節談到的 chip 以 `--accent` 外框標示。

| # | id | 標題 | 內容（依序；每個符號在使用前定義） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈Decision Diffuser：把決策寫成條件生成〉 | 副標「這篇論文不估計價值函數，而是學一個以報酬、限制或技能為條件的狀態序列生成模型，再從相鄰狀態推出動作。」三張事實卡（見下）；條件 chip 列；連結：arXiv、`../index.html#paper-441`（在關聯圖中查看）、筆記總覽 `decision-diffuser-lineage-hub.html` | 1 螢幕 |
| S1 | `rl` | 01 · 強化學習以報酬衡量策略 | 狀態、動作、獎勵、策略、軌跡的白話定義；折扣因子 $\gamma\in[0,1)$ 在此就地定義（越遠的獎勵乘上越多次 γ）；MDP 元組（$\rho_0$ 初始狀態分布、$\mathcal T$ 轉移函數，就地定義）；報酬 $R(\tau)=\sum_t\gamma^tr_t$（註明原文以 $k$ 為求和指標）；$\pi^*$；$Q^*$。延伸閱讀：`rl-mdp-q-learning-basics` | 250–350 字 |
| S2 | `offline` | 02 · 離線 RL 的 TD 方法有兩個困難 | Eq. 1、Eq. 2；致命三元組三個詞的字面解釋；離線設定、$\mu$、$d^\pi$ 對 $d^\mu$、$D(\cdot\|\cdot)$；「軌跡拼接」的字面意思（把資料中不同軌跡的片段接成一條新軌跡）；作者假設 DD 能隱式拼接（App. A.1 照錄，見步驟 9）；LO1 答案。延伸閱讀：`offline-rl-distribution-shift`、`implicit-dp-trajectory-stitching` | 300–400 字 |
| S3 | `diffusion` | 03 · 擴散模型以逐步去噪產生資料 | 前向／反向式；$\alpha_k,\bar\alpha_k$ 定義（含索引差一的註，並聲明與低溫係數 α 的寫法區分）；閉式 $x_k$；Eq. 3；可收合符號表（1.4，< 600px 為堆疊卡片）。延伸閱讀：`ddpm-training-and-sampling-recap` | 250–350 字 |
| S4 | `score` | 04 · 噪聲預測對應分數，分數可以加上條件 | 分數的字面定義「$\nabla_x\log p(x)$：讓機率密度增加最快的方向」；靜態例：一維 $\mathcal N(0,1)$ 的分數 $-x$（SVG：密度曲線＋數個 $x$ 處指向 0 的箭頭，箭頭長度 $|x|$）；最佳噪聲預測與分數（含條件與負號）；條件分數與貝氏定理 $\nabla\log p(x\mid y)=\nabla\log p(x)+\nabla\log p(y\mid x)$，$p(y\mid x)$ 即「分類器」；分類器引導式（Sec. 2.2）與無分類器引導式（Sec. 2.2，以 Ø 表示無條件）；LO3 答案。延伸閱讀：`eps-prediction-ddpm-score`、`diffusion-classifier-guidance-background` | 350–450 字 + 靜態 SVG |
| S5 | `formulation` | 05 · 決策被寫成狀態序列的條件生成 | 先寫 $x_0(\tau)$ 的白話定義與 Eq. 6（從軌跡取出的 $H$ 個連續狀態）；Eq. 4、Eq. 5；$y$ 的三種內容（chip 列）；與報酬條件序列模型（DT）及 Diffuser 的差別各一句（Sec. 5）；LO2 答案。延伸閱讀：`sequence-modeling-offline-rl`、`return-conditioned-sequence-modeling`、`trajectory-diffusion-planning` | 250–350 字 |
| S6 | `states` | 06 · 只對狀態擴散，動作由逆動力學取得 | 狀態陣列（HTML 格線 4 列 × 8 欄）；Eq. 7；兩個理由；Table 2 全表與兩點限定（+0.5 在標準誤內；Table 2 同時改變擴散對象與動作來源）；App. F Table A2；LO4 答案。延伸閱讀：`inverse-dynamics-state-diffusion` | 350–450 字 + 2 小表 |
| S7 | `guidance` | 07 · 無分類器引導讓樣本集中在資料中報酬較高的部分 | (a) 為什麼不用分類器引導（Sec. 3.2；「混入次佳行為」，作者用語原為 polluted）；(b) Sec. 3.4 損失（< 600px 拆兩行）、丟棄條件、Ø 的編碼；(c) Eq. 8 與 ω = 0／1〔本頁推導〕；(d) 進階 `<details>`：隱式分類器式〔本頁推導〕與「取樣結果不必等於 $p(x)p(y\mid x)^\omega$」；(e) App. K 兩點理由（作者主張，本頁未驗證）；(f) 玩具說明（4.0）；(g) C1a；(h) 一段玩具觀察（標「本頁玩具」，不與論文數字同句）。LO5、LO6 答案。延伸閱讀：`classifier-free-guidance-return-conditioning` | 500–650 字 + C1a |
| S8 | `lowtemp` | 08 · 低溫取樣縮小樣本的散布 | Alg. 1 第 4、9 行；α = 0.5；Table A1 與作者解讀；C1b；一段玩具觀察（α = 0 時所有樣本相同；玩具不重現 Table A1）；LO7 答案 | 250–350 字 + C1b |
| S9 | `planning` | 09 · 每個環境步重新產生計畫，只執行第一個動作 | Alg. 1 兩欄對照（447 的 `.algo` 格線）；Alg. 1 第 7 行記號註；滾動視界；歷史寫入；流程圖 `#s9-loop`；App. E 的時間與暖啟動（字面定義：以上一步的計畫加噪後作為起點，少跑幾步去噪）；LO8 答案 | 350–450 字 + 流程圖 |
| S10 | `compose` | 10 · 限制與技能可以在測試時組合 | one-hot 條件；Eq. 9；條件獨立的兩處寫法；App. D 推導（`<details>`，aligned 放橫捲容器）；可行性；NOT 式；無 OR；App. I.2 的失敗基準；App. A.2 的設定與 100%／91.3%（註明「此處 R、r 是半徑」）；C2；C2 之後一段玩具觀察；LO9 答案。延伸閱讀：`composing-constraints-and-skills` | 550–650 字 + C2 |
| S11 | `results` | 11 · 離線 RL：平均分數最高，但不是每個資料集都最高 | D4RL 與「正規化分數」的字面定義（依 D4RL 定義：0 約為隨機策略、100 約為專家）；比較方法一句一個；Table 1 全表（橫捲容器）；逐列結果一段（LO10 答案）；Fig. 4 一句；C3；Table A3 的結論一段。延伸閱讀：`dd-experiments-and-ablations`、`offline-rl-benchmarks-and-baselines` | 500–600 字 + 表 + C3 |
| S12 | `skills` | 12 · 積木堆疊與四足步態：組合出資料中沒有的行為 | Kuka 設定與 Table 3（含區間重疊與 OOD 註）；BCQ／CQL 0.0；Unitree 設定、分類器、Fig. 6 的描述、Fig. 7 表格與 App. I 兩表（三張 3×3 小表，< 480px 堆疊）；正文與表格欄位相反的註腳；LO11 答案 | 350–450 字 + 表 |
| S13 | `impl` | 13 · 實作設定與常見錯誤 | 1.4 步驟 8 的表（< 480px 改為堆疊卡片）；7 條常見錯誤 | 300–400 字 + 表 |
| S14 | `history` | 14 · 研究脈絡 | 時間線（6.4）＋一段話：擴散模型 → 分數觀點 → 分類器／無分類器引導 → 報酬條件策略與 DT → Diffuser → 組合式擴散 → 本文 | 時間線 + 100–150 字 |
| S15 | `next` | 15 · 限制與延伸閱讀 | App. M 六項；隨機動力學數字與作者解釋（LO12 答案，數字重寫在正文）；筆記清單；arXiv、專案網站 anuragajay.github.io/decision-diffuser | 200–300 字 + 清單 |

Hero 三張事實卡（所有術語在卡內定義）：
1. 「Decision Diffuser 是一個生成模型：給定想要的報酬 y（一條軌跡的折扣獎勵總和），它產生一串未來的狀態，再用另一個小網路從相鄰兩個狀態推出要執行的動作。」
2. 「它不學價值函數（估計未來報酬的函數）。取樣時，它把同一個網路『有條件』與『沒有條件』的兩個預測混合，讓產生的軌跡集中在資料中報酬較高的部分；這種做法稱為無分類器引導。」
3. 「在 D4RL（離線強化學習的公開資料集）運動控制 9 個資料集上，平均正規化分數（0 約為隨機策略、100 約為專家）81.8；兩種既有的離線 RL 方法 CQL、IQL 為 77.6、77。但 9 個資料集中只有 4 個是最高分（Table 1）。」

桌機（≥ 1024px）左側 sticky 章節目錄；< 1024px 改頂部橫向捲動列（沿用外殼）。

---

## 4. 互動元件規格

### 4.0 共用：排程、PRNG、玩具的界線

- **為什麼用玩具**（S7 寫出）：瀏覽器中無法訓練 DD。本頁用答案已知的高斯混合分布當作「資料分布」，它的最佳噪聲預測有閉式解，可當作訓練完美的 $\epsilon_\theta$；條件與無條件的 $\epsilon$ 都由同一份公式算出。玩具不含時間序列、逆動力學與環境；C1 的資料是一維、C2 是二維。**玩具的數值是推定值，玩具結果不重現論文的任何數字**（C2 的成功率與 App. A.2 的 91.3% 接近純屬巧合，頁面明寫不可比較）。
- **排程**（推定）：Diffuser 使用 cosine 排程，推定 DD 相同（App. B 只說 temporal U-Net 的程式碼取自 Diffuser，未寫排程）。$s=0.008$，$K=100$：$f(i)=\cos^2\!\big(\frac{i/K+s}{1+s}\cdot\frac\pi2\big)$，$\beta_i=\min(1-f(i)/f(i-1),0.999)$，$\bar\alpha_0=1$，$\bar\alpha_i=\prod_{j\le i}(1-\beta_j)$，$\tilde\beta_i=\beta_i(1-\bar\alpha_{i-1})/(1-\bar\alpha_i)$，$\tilde\beta_1=0$；索引 $i=1,\dots,K$（$i=1$ 時用到 $f(0)$）。本頁的 $\alpha_i=1-\beta_i$ 是第 $i-1\to i$ 步（DDPM 慣例）；論文寫 $q(x_{k+1}\mid x_k)$ 用 $\alpha_k$，照字面 $x_k$ 對應 $\prod_{i<k}\alpha_i$，差一個索引。頁面在 S3 註明 $\bar\alpha_k$ 是本頁的定義。反向變異數 $\Sigma_k=\tilde\beta_k$（推定，與 447 頁相同）。
- **一個反向步**（Alg. 1 第 7–9 行）：$\mu=(x-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\hat\epsilon)/\sqrt{1-\beta_k}$；$k>1$ 時 $x\leftarrow\mu+\sqrt{\alpha\tilde\beta_k}\,z$，$k=1$ 時 $x\leftarrow\mu$。起點 $x_K=\sqrt\alpha\,z$。不裁切 $\hat x_0$。
- **PRNG**：`mulberry32(seed)`（447 頁同一寫法：`a=(a+0x6D2B79F5)|0; t=Math.imul(a^(a>>>15),1|a); t=(t+Math.imul(t^(t>>>7),61|t))^t; return ((t^(t>>>14))>>>0)/4294967296;`），`gauss()` 為 Box–Muller 只取 cos 分支：$\sqrt{-2\ln(1-u_1)}\cos(2\pi u_2)$。**抽樣為步驟優先**：先抽全部樣本的起點，再對 $k=K\dots2$ 每步依序抽全部樣本的噪聲；樣本彼此獨立，所以「前 $n$ 個樣本」的結果與只跑 $n$ 個樣本時完全相同（`sweep.py` 斷言），C1 拖動時利用這一點。實作時預先產生整張噪聲表（C1：100 × 2000，C2：100 × 500 × 2）並快取，參數改變不重抽。
- **頁面界線**：C1、C2 圖版標題旁加「本頁玩具」pill；圖版底部一行「讀數由本頁的高斯混合玩具依 4.0 的公式即時計算，不是論文的模型，也不重現論文數字」。

### C1 · 引導強度 ω 與低溫係數 α（C1a 在 S7、C1b 在 S8；同一份程式、兩個圖版）

- **教的概念**：C1a：Eq. 8 的 ω 讓樣本集中到高報酬區；引導結果與 $p(x)p(y\mid x)^\omega$ 不同。C1b：低溫取樣（Alg. 1 第 4、9 行）縮小樣本散布，α = 0 時樣本完全相同。
- **玩具資料**（推定，頁面寫出）：每條軌跡只用一個數 $x$ 表示（例如平均前進速度）。資料分布 $p(x)=\sum_{j=1}^3\pi_j\mathcal N(x;m_j,0.07^2)$，$m=(0.25,0.55,0.85)$、$\pi=(0.50,0.35,0.15)$：三種次佳程度不同的行為，最好的一種最少。報酬標籤 $y=x+\eta$，$\eta\sim\mathcal N(0,0.25^2)$（同一個 $x$ 的軌跡報酬略有不同），因此 $p(y\mid x)=\mathcal N(y;x,0.25^2)$，生成樣本的期望報酬等於樣本 $x$ 的平均。
- **閉式**〔本頁推導〕：$p(x)\,p(y\mid x)^\omega$（ω > 0）仍是三成分高斯混合：令 $s^2=0.07^2$、$e^2=0.25^2/\omega$，成分 $j$ 的權重 $\propto\pi_j\mathcal N(y;m_j,s^2+e^2)$、平均 $(m_je^2+ys^2)/(s^2+e^2)$、變異數 $s^2e^2/(s^2+e^2)$；ω = 1 即 $p(x\mid y)$，ω = 0 時取 $p(x)$。加噪：$p_k=\sum_jw_j\mathcal N(x;\sqrt{\bar\alpha_k}\mu_j,\bar\alpha_kv_j+1-\bar\alpha_k)$，$\epsilon(x)=-\sqrt{1-\bar\alpha_k}\sum_jr_j(\sqrt{\bar\alpha_k}\mu_j-x)/V_j$，$r_j$ 為後驗責任。$\epsilon_{\text{Ø}}$ 用 $p(x)$，$\epsilon_y$ 用 ω = 1 的 $p(x\mid y)$，再以 Eq. 8 組合。
- **數值保護**：$r_j$ 以 log-sum-exp 計算；權重 $<10^{-300}$ 的成分略過；$V_j\ge1-\bar\alpha_1\approx6.3\times10^{-4}$。完整滑桿網格（ω 51 值 × α 21 值 × y 21 值，`sweep.py`）結果見第 5 節：無非有限值、最大 |x| 有界。仍檢查非有限值，出現時顯示「數值溢位」並保留上次結果。
- **輸入**：C1a：滑桿 ω ∈ [0, 5] 步長 0.1（預設 1.4，論文範圍 1.2–1.8 標在滑桿下方）；滑桿「條件報酬 y」∈ [0, 1] 步長 0.05（預設 1.0；上限 1 是論文正規化報酬的上限，Sec. 3.3）；α 固定為 1（圖版註明「低溫取樣見第 08 章」）。C1b：同樣的 ω、y 滑桿（與 C1a 同步的狀態），加滑桿「低溫係數 α」∈ [0, 1] 步長 0.05（預設 0.5，App. B）。兩圖版各有「回到預設」按鈕。
- **計算**：$N=2000$ 個樣本、seed 1。`input` 事件以 `requestAnimationFrame` 合併，若上一次計算尚未完成就跳過該影格；**拖動中只算前 500 個樣本**（約 3×10⁵ 次 exp），`change`（放開或鍵盤操作後 250ms）才算完整 2000 個（約 1.2×10⁶ 次 exp）；拖動中讀數旁標「預覽：500 個樣本」。統計：平均、標準差（母體式）、$x>0.7$ 的比例（0.7 是第二與第三成分平均的中點）；閉式目標的三個相同統計（比例用常態 CDF，`erf` 以 Abramowitz–Stegun 7.1.26，誤差 < 1.5e−7）。
- **顯示**（≥ 768px：圖左、控制右；< 768px：圖、控制、讀數上下堆疊，圖寬 = 容器寬、高 200px）：
  - 直方圖：橫軸 $x\in[-0.2,1.4]$，64 格，樣本以 `--accent` 長條（密度尺度）；`--ink-3` 虛線 = 資料 $p(x)$，標「資料」；`--c-ref` 長虛線 = $p(x)p(y\mid x)^\omega$，標「p(x)·p(y|x)^ω」；垂直細線在 $x=y$ 標「y」；點線在 0.7 標「高報酬區」；點線在 0.85 標「資料中最好的行為」。縱軸上限取「最高長條與目標曲線最大值的較大者」向上取到 {2, 4, 8, 16, 32, 48} 中最小的可容納值（α = 1 時最高約 9 → 16；α = 0.05 時約 31 → 32；α = 0 時全部樣本在同一格，密度 40 → 48），換刻度時以 240ms 過渡（reduced-motion 時直接跳變）。
  - 讀數（`<dl>` 兩欄）：樣本平均（＝期望報酬）、樣本標準差、高報酬區比例；目標分布的三個值；C1b 在 α = 0 時加一行「α = 0：2000 個樣本完全相同」。
- **讀者應觀察到**（本頁玩具，y = 1）：C1a：(1) ω = 0 時樣本接近資料分布（平均 0.4445，目標 0.4450），高報酬區約 15%；(2) ω 增大時樣本移向高報酬區：ω = 1、1.4、2、4 的比例為 0.6085、0.8065、0.9515、1.0000；(3) ω = 1 時樣本近似 $p(x\mid y)$（平均 0.7407 對 0.7388；標準差 0.1605 對 0.1709 偏小，來自有限 $K$ 與 $\tilde\beta$ 的選擇，R7），ω > 1 時樣本比 $p(x)p(y\mid x)^\omega$ 更集中（ω = 2：標準差 0.0775 對 0.1171）；(4) ω = 4 時樣本平均 0.9173，越過資料中最好行為的中心 0.85：引導太強時樣本離開資料支撐。頁面寫明：論文只說引導「取出資料中最好的部分」（Sec. 3.2），沒有主張也沒有展示超出資料的外推，這是玩具中的現象。C1b：(5) α = 0.5 縮小散布（ω = 1.4：標準差 0.1229 → 0.0874），α = 0 時所有樣本相同；(6) 論文 Table A1 的 α 結果是 Hopper 上的實驗，玩具不重現它。

### C2 · 組合限制：AND、NOT 與不可行的組合（S10）

- **教的概念**：Eq. 9 在測試時組合兩個訓練時分開的條件；可行性（App. D）；NOT（App. J）。
- **玩具資料**（推定；設定仿 App. A.2，但只建模最終位置 $s_T\in\mathbb R^2$，不含軌跡與動力學）：資料集 1（條件 $y^1$：$\|s_T\|\le R$，$R=1$）200 點，在半徑 1 的圓盤內均勻：每點依序抽 $u_1,u_2$，$\rho=\sqrt{u_1}$、$\theta=2\pi u_2$，點 $=(\rho\cos\theta,\rho\sin\theta)$（seed 11）；資料集 2（$y^2$：$\|s_T\|\ge r$）200 點，在環 $r\le\rho\le1.5$ 均勻：$\rho=\sqrt{r^2+u_1(1.5^2-r^2)}$（seed 12；改 $r$ 時用同一批 $u$，點連續移動）。外徑 1.5 是推定值（論文只寫 $\ge r$）。每點視為 $\mathcal N(c_n,0.05^2I)$（核寬推定）。無條件分布 $p=\tfrac12p_1+\tfrac12p_2$（以 $p=0.25$ 丟棄條件訓練出的無條件模型對應全部資料，兩集合等量）。
- **閉式**：加噪後每點為 $\mathcal N(\sqrt{\bar\alpha_k}c_n,VI)$，$V=\bar\alpha_k\cdot0.05^2+1-\bar\alpha_k$，所有點共用 $V$。
  - 集合 $i$ 的 log 密度：$\ell_i(x)=\mathrm{logsumexp}_{n}\big(-\|x-\sqrt{\bar\alpha_k}c_n\|^2/(2V)\big)-\log200-\log(2\pi V)$（logsumexp 先減去最大項）。
  - 集合 $i$ 的分數：$\big(\sum_nr_n\sqrt{\bar\alpha_k}c_n-x\big)/V$，$r_n$ 為同一個 logsumexp 的 softmax 權重。
  - $g_1(x)=\exp(\ell_1)/(\exp(\ell_1)+\exp(\ell_2))$，以 $m=\max(\ell_1,\ell_2)$ 平移計算（兩集合點數與 $V$ 相同，常數項會相消，仍照式計算）；無條件分數 $=g_1\cdot$分數$_1+(1-g_1)\cdot$分數$_2$（混合分布的分數）。
  - $\epsilon_i=-\sqrt{1-\bar\alpha_k}\cdot$分數$_i$。模式：「只用 y¹」$\epsilon_{\text{Ø}}+\omega(\epsilon_1-\epsilon_{\text{Ø}})$；「只用 y²」同理；「AND（Eq. 9）」$\epsilon_{\text{Ø}}+\omega[(\epsilon_1-\epsilon_{\text{Ø}})+(\epsilon_2-\epsilon_{\text{Ø}})]$；「y¹ AND NOT y²（App. J）」$\epsilon_{\text{Ø}}+\omega[(\epsilon_1-\epsilon_{\text{Ø}})-(\epsilon_2-\epsilon_{\text{Ø}})]$。
  - g₁ 驗收值見第 5 節（`sweep.py`）。
- **取樣**：500 個樣本、seed 21、低溫係數 α = 0.5 固定（App. B 值）、$K=100$。PRNG 步驟優先：起點逐樣本抽 $(x,y)$，之後每步同樣順序。
- **計算與效能**：每次約 $500\times400\times100\times2=4\times10^7$ 次 exp。**以 `setTimeout(…,0)` 切片，每片 1 個擴散步**（約 4×10⁵ 次 exp，中階手機約 5–15ms），共 100 片；圖版顯示進度「計算中 k = …」，控制項維持可用；新請求以 token 取消舊切片；結果快取（key `${mode}|${omega}|${r}`）。首次計算在圖版第一次捲入視窗（IntersectionObserver，可見 25%）時開始。
- **拖動中**：只更新資料點與兩個圓；已有樣本與讀數降為 50% 不透明並標「放開後重算」；`change` 後啟動計算；鍵盤方向鍵每次觸發 `change`，以 250ms debounce 後才啟動。
- **數值保護**：logsumexp；$V\ge6.3\times10^{-4}$；非有限值時中止並顯示「數值溢位」。完整網格（4 模式 × ω 26 值 × r 11 值，`sweep.py`）結果見第 5 節。
- **輸入**：分段按鈕（`role="radiogroup"`）模式 4 種（預設 AND）；滑桿 ω ∈ [0.5, 3] 步長 0.1（預設 1.2，推定：取論文引導範圍的下端）；滑桿 $r$ ∈ [0.3, 1.3] 步長 0.1（預設 0.7）；勾選框「顯示資料點」（預設開）。
- **顯示**（≥ 768px：平面圖左、半徑直方圖與讀數右；< 768px：平面圖（正方形，寬 = 容器寬）→ 半徑直方圖（高 120px）→ 控制 → 讀數）：
  - 平面圖 $[-1.8,1.8]^2$：圓 $\|s\|=1$（實線，標「R = 1」）與 $\|s\|=r$（虛線，標「r」）；資料集 1 小圓點（`--c-ref`）、資料集 2 小叉號（`--ink-3`）；樣本 `--accent` 實心點；$r\le\|s\|\le1$ 的環以淺色填滿，標「兩個限制都滿足」（$r>1$ 時不畫，改標「沒有同時滿足兩個限制的位置」，此句同時送入 live region）。
  - 半徑直方圖：橫軸 $\rho\in[0,1.8]$，36 格；垂直線標 $r$ 與 1。
  - 讀數：滿足 $\|s\|\le1$、滿足 $\|s\|\ge r$、兩者都滿足、$\|s\|<r$ 的比例；本模式的「目標」列加粗（AND→兩者、NOT→$\|s\|<r$、單一→對應限制）；平均半徑。
- **讀者應觀察到**（本頁玩具）：(1) 只用 y¹ 時樣本在圓盤內，只有 21.0% 落在環中（ω = 1.2）；(2) AND 時 92.6% 落在環中（ω = 1.0：90.0%，ω = 2.0：95.0%），兩個資料集都沒有「只在環內」的資料；(3) NOT 時 100% 落在 $\|s\|<r$，但樣本擠在原點附近（平均半徑 0.08）：玩具的 $p_2$ 在圓盤內部趨近 0，使 $p\,p_1/p_2$ 在那裡變得極大，樣本被推向離環最遠的地方。這是玩具密度造成的現象，不代表 DD 的行為；論文 App. J 的 NOT 證據只有影片；(4) $r=1.1$、$1.3$ 時組合不可行，樣本停在兩個區域之間（平均半徑 1.036、1.131），兩個限制都不滿足（比例 0.000），對應 App. D「不可行時行為不連貫」；(5) $r=0.9$ 時可行區很窄，成功率降到 73.0%。頁面加一句：論文 App. A.2 的 91.3% 來自有軌跡與動力學的實驗，玩具數字不可與之比較。

### C3 · 論文結果瀏覽器（S11）

- **教的概念**：讀 Table 1 時區分「平均最高」與「每列最高」；Table A3 中 DD 的優勢隨隨機性消失；平均列可由各列重算。
- **資料**：常數陣列內嵌在頁面 JS（與 `verify/tables.py` 逐字相同，建置時從該檔複製）。Table 1 的 Hopper-ME Diffuser 顯示 Table 1 的 107.2（S6 的 Table 2 照錄 107.6，註腳說明兩表不同）。Table 1（11 列 × 8 方法，含 DD 標準誤；Kitchen 兩列只有 BC、CQL、IQL、DD 有數字）、Table A3（4 個 $p$ × 4 方法，含標準誤），逐字照錄（`tables.py` 同一份陣列）。
- **輸入**：分段按鈕「Table 1」「Table A3」（預設 Table 1）。Table 1：下拉選單 `<select>` 選列（11 列，預設 Medium · Hopper）＋按鈕「上一列／下一列」；勾選框「顯示平均列（由本頁重算）」。Table A3：滑桿 $p$ 索引 0–3（`aria-valuetext`「p = 0.05」等；預設索引 1）。
- **計算（即時）**：Table 1：該列最大值與方法（同值全列）；DD 名次 = 1 + 嚴格高於 DD 的方法數，**分母 = 該列非缺值的方法數**（運動控制 8、Kitchen 4）；高於 DD 的方法清單；全表「DD 為最高的列數」並註明「Kitchen 列只與 BC、CQL、IQL 比較」；平均列：運動控制 9 列與 Kitchen 2 列的算術平均，與論文印出的平均並列，差值 ≤ 0.05 標「與論文一致（四捨五入）」。Table A3：DD 減去 Diffuser、CQL 中較高者；區間 [平均 − 標準誤, 平均 + 標準誤] 是否重疊（頁面說明：重疊只是粗略的視覺判斷，不是檢定）。
- **顯示**：
  - Table 1：水平長條圖，8 列方法（固定順序 BC、CQL、IQL、DT、TT、MOReL、Diffuser、DD），左側 72px 方法名，長條右端寫數值；DD 長條 `--accent` 並畫 ±標準誤的誤差線；其他方法 `--ink-3`；最高值長條右側加「最高」文字標籤（不只靠顏色）；垂直虛線標 DD 的值。缺值方法顯示「—（未報告）」。橫軸 0–120。390px 時每列高 28px、總高約 240px，不需橫捲。讀數例：「最高：TT 95」「DD 名次：3／8」「高於 DD：CQL 91.6、TT 95」「DD 為最高的列：6／11（運動控制 4／9；Kitchen 列只比 3 個方法）」；Kitchen 列「DD 名次：1／4」。
  - Table A3：折線圖 4 系列，標記形狀區分（DD 圓、Diffuser 方、CQL 三角、BC 菱形）＋線型（DD 實線粗，其餘細）＋各系列右端直接標籤；誤差線；目前 $p$ 以垂直帶標示。< 600px 時右端標籤改為圖下方圖例（形狀樣本＋名稱），圖高 220px。讀數：「p = 0.05：DD 77.3，較高的對手 Diffuser 63.2，差 +14.1，區間不重疊」。
- **讀者應觀察到**：(1) 平均列重算與論文一致（最大差 0.044，四捨五入）；(2) DD 在 9 個運動控制資料集中 4 個最高，HalfCheetah-Medium-Replay 排第 6／8；(3) $p=0$、0.05 時 DD 領先 11.5、14.1，$p=0.10$、0.15 時差 +0.9、−0.9，區間重疊。
- **數值保護**：缺值（`null`）不參與平均與名次。

### 4.4 共同的無障礙、排版與互動規則

- **動態文字不用組合字元與 Greek Extended**：讀數、SVG 標籤、`aria-label`、`aria-valuetext`、live region 只用 ω、α、ε、Ø、ρ、≤、≥、·、上標 ¹／² 與下標 ₜ（U+209C）、₊、₁ 等常見預組字元；不寫 ε̂、x̃、ᾱ（U+1FB1，內文字型多半沒有 Greek Extended）。動態文字不需要累積排程符號；「引導後的噪聲預測」以文字表達。KaTeX 內的 $\bar\alpha_k$、$\hat\epsilon$ 不受此限。建置時在 390px 截圖確認無缺字方框，selfTest 以 `document.fonts.check('13px "JetBrains Mono"', 'ωαερØ')` 記錄結果（不計入失敗）。
- **非顏色編碼**：C1 樣本長條／資料虛線／目標長虛線＋直接標籤；C2 資料集 1 圓點、資料集 2 叉號、樣本實心點、兩個圓實線／虛線；C3 標記形狀＋「最高」文字。
- **觸控**：不使用 hover；所有查值以滑桿、按鈕、下拉選單完成；`input[type=range]` 與按鈕 `min-height:44px`。
- **live region**：`aria-valuetext` 隨滑桿更新（例「ω = 1.4」「α = 0.50」「r = 0.7」「p = 0.05」）；讀數區 `aria-live="polite"` 與 SVG `aria-label` 只在 `change` 或停止輸入 500ms 後更新（共用 debounce）；C2 只在計算完成時宣告一次（例「AND，兩個限制都滿足 92.6%」）。
- **分段按鈕**：`role="radiogroup"`／`role="radio"`＋`aria-checked`，roving tabindex，方向鍵切換（447 的 `segInit`）。
- **數學排版**：所有 display 數學放 `.katex-display{overflow-x:auto;overflow-y:hidden}`；Sec. 3.4 損失、Eq. 8、Eq. 9、NOT 式在 < 600px 拆成兩行：寫兩個版本（單行、`aligned` 兩行），CSS 依寬度切換 `display`，**目前隱藏的版本加 `aria-hidden="true"`**（以 `matchMedia('(max-width:599px)')` 監聽切換屬性），頁內搜尋重複可接受。App. D 推導在 `<details>` 內。Table 1 放 `overflow-x:auto` 容器並加左右漸層；LO 表與符號表在 < 600px 改為堆疊卡片。
- **390px 檢查清單**：無水平捲動（selfTest 檢查 `scrollWidth <= innerWidth`）；C1、C2 的圖與控制上下堆疊；C3 長條圖不橫捲；讀數 `<dl>` 不截斷。
- `prefers-reduced-motion: reduce`：取消 CSS transition 與 JS 動畫（C1 縱軸過渡、捲動）；C2 照常切片，但不顯示進度動畫。

---

## 5. 驗收數值表

全部取自 `verify/check_out.txt`（同一演算法、雙精度）。**容許誤差**：C1 平均與標準差 ±5e−4（拖動預覽的 500 個樣本不驗收），比例 ±0.002（4 個樣本），閉式目標統計 ±1e−4，`x[0]` ±1e−4；C2 比例 ±0.004（2 個樣本），平均半徑 ±0.002，`X0` ±1e−3；排程與 PRNG ±1e−6；C3 平均與差值 ±1e−3（名次、列數必須完全相同）。

**排程與 PRNG**：β₁ 0.000631、β₅₀ 0.030593、β₁₀₀ 0.999000；ᾱ₁ 0.999369、ᾱ₅₀ 0.493844、ᾱ₁₀₀ 2.4286e−07；β̃₂ 0.000403、β̃₅₀ 0.029651、β̃₁₀₀ 0.998758。`mulberry32(1)` 前三個 0.627073941、0.002735721、0.527447040；seed 1 前兩個 gauss 1.404339、1.215755。

**C1 條件分布**：$p(x\mid y=1)$ 權重 0.0362、0.3666、0.5972，平均 0.3045、0.5827、0.8609，變異數 0.004544；$y=0.5$：權重 0.4377、0.4782、0.0841。

**C1 樣本**（$N=2000$、seed 1、$y=1$；樣本 平均／標準差／比例 ‖ 目標 $p(x)p(y\mid x)^\omega$ 平均／標準差／比例 ‖ 第 0 個樣本）

| ω | α | 樣本 | 目標 | x[0] |
| --- | --- | --- | --- | --- |
| 0 | 1 | 0.4445／0.2202／0.1475 | 0.4450／0.2289／0.1532 | 0.30792 |
| 0 | 0.5 | 0.4194／0.1770／0.0665 | 同上 | 0.30316 |
| 0 | 0 | 0.3384／0.0000／0.0000 | 同上 | 0.33839 |
| 1 | 1 | 0.7407／0.1605／0.6085 | 0.7388／0.1709／0.6071 | 0.69304 |
| 1 | 0.5 | 0.7584／0.1267／0.6635 | 同上 | 0.76235 |
| 1 | 0 | 0.8210／0.0000／1.0000 | 同上 | 0.82098 |
| 1.4 | 1 | 0.8102／0.1229／0.8065 | 0.7857／0.1451／0.7281 | 0.83431 |
| 1.4 | 0.5 | 0.8306／0.0874／0.8920 | 同上 | 0.84555 |
| 1.4 | 0 | 0.8579／0.0000／1.0000 | 同上 | 0.85793 |
| 2 | 1 | 0.8683／0.0775／0.9515 | 0.8275／0.1171／0.8467 | 0.87414 |
| 2 | 0.5 | 0.8776／0.0443／0.9915 | 同上 | 0.87669 |
| 4 | 1 | 0.9173／0.0449／1.0000 | 0.8794／0.0719／0.9775 | 0.91686 |
| 4 | 0.5 | 0.9152／0.0319／1.0000 | 同上 | 0.91527 |

其他：$y=0.5$、ω = 1.4、α = 0.5：0.4481／0.1271／0.0100 ‖ 0.4526／0.1750／0.0693；$y=0.5$、ω = 1、α = 1：0.4476／0.1777／0.0760 ‖ 0.4480／0.1878／0.0868；$y=0.75$、ω = 1.4、α = 0.5：0.6489／0.1244／0.3105 ‖ 0.6328／0.1783／0.3397；$y=0$、ω = 2、α = 0.5：0.2073／0.0363／0.0000 ‖ 0.2225／0.0766／0.0000。

**穩定性掃描**（`verify/sweep_out.txt`，完整滑桿網格）：C1 的 ω 0–5（步長 0.1）× α 0–1（0.05）× y 0–1（0.05）共 22,491 次：非有限值 0，最大 |x| 1.0701（ω = 5、α = 1、y = 1）。C2 的 4 模式 × ω 0.5–3（0.1）× r 0.3–1.3（0.1）共 1,144 次：非有限值 0，最大半徑 1.5405（只 y²、ω = 3、r = 1.3）；各模式最大半徑：只 y¹ 1.1859、只 y² 1.5405、AND 1.2485、NOT 0.9715。C1 的快速版本與原版相等，且 N = 2000 的前 500 個樣本與 N = 500 的結果完全相同（斷言通過）。

**C2 的 g₁**（r = 0.7；x = (0.5, 0.5)、(0, 0)、(1.2, 0)；容許 ±1e−5）：k = 100：0.499998、0.500000、0.500004；k = 50：0.550815、0.597234、0.492265；k = 10：0.678656、0.999764、0.189701；k = 1：0.761053、1.000000、0.000172。

**C2 資料**：圓盤第 0 點 (−0.70263, −0.13379)，最大半徑 0.9988；環（r = 0.7）第 0 點 (0.88782, 0.45707)，半徑範圍 0.7010–1.4985。

**C2 樣本**（500 個、seed 21、α = 0.5）

| 模式 | ω | r | ≤ R | ≥ r | 兩者 | < r | 平均半徑 | X0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 只 y¹ | 1.2 | 0.7 | 1.000 | 0.210 | 0.210 | 0.790 | 0.5194 | (0.1716, −0.3386) |
| 只 y² | 1.2 | 0.7 | 0.300 | 0.998 | 0.298 | 0.002 | 1.0993 | (−0.7313, 0.2760) |
| AND | 0.5 | 0.7 | 0.920 | 0.816 | 0.736 | 0.184 | 0.8191 | (0.5010, 0.2225) |
| AND | 1.0 | 0.7 | 0.948 | 0.952 | 0.900 | 0.048 | 0.8450 | (−0.4758, 0.4941) |
| AND | 1.2 | 0.7 | 0.954 | 0.972 | 0.926 | 0.028 | 0.8518 | (−0.5093, 0.4846) |
| AND | 2.0 | 0.7 | 0.956 | 0.994 | 0.950 | 0.006 | 0.8691 | (−0.5527, 0.4813) |
| AND | 3.0 | 0.7 | 0.946 | 1.000 | 0.946 | 0.000 | 0.8785 | (−0.6670, 0.3799) |
| AND | 1.2 | 0.5 | 0.946 | 0.986 | 0.932 | 0.014 | 0.7703 | (0.4648, 0.2465) |
| AND | 1.2 | 0.9 | 0.894 | 0.836 | 0.730 | 0.164 | 0.9419 | (−0.6977, 0.5440) |
| AND | 1.2 | 1.1 | 0.084 | 0.000 | 0.000 | 1.000 | 1.0364 | (−0.9686, 0.3877) |
| AND | 1.2 | 1.3 | 0.000 | 0.000 | 0.000 | 1.000 | 1.1308 | (−1.0418, 0.3867) |
| NOT y² | 1.2 | 0.7 | 1.000 | 0.000 | 0.000 | 1.000 | 0.0805 | (0.0874, −0.0652) |
| NOT y² | 1.2 | 0.5 | 1.000 | 0.102 | 0.102 | 0.898 | 0.1303 | (0.0003, 0.0251) |

**C3**（名次分母：運動控制 8、Kitchen 4；列順序同 Table 1：ME-HalfCheetah、ME-Hopper、ME-Walker2d、M-HalfCheetah、M-Hopper、M-Walker2d、MR-HalfCheetah、MR-Hopper、MR-Walker2d、Mixed-Kitchen、Partial-Kitchen）：運動控制平均重算 BC 51.856、CQL 77.611、IQL 76.956、DT 74.733、TT 78.878、MOReL 72.944、Diffuser 75.333、DD 81.822（與印出值最大差 0.044）；Kitchen BC 44.75、CQL 51.25、IQL 48.65、DD 61.00。DD 名次（列順序同 Table 1）：3、1、2、1、2、1、6、1、3、1、1；DD 最高 6／11 列（運動控制 4／9）。Table A3（DD − 較高對手）：$p=0$ +11.5（CQL，不重疊）、0.05 +14.1（Diffuser，不重疊）、0.10 +0.9（Diffuser，重疊）、0.15 −0.9（CQL，重疊）。Table 2 差值：CondDiffuser − Diffuser +3.7、+7.8、−20.3；DD − CondDiffuser +0.5、+13.0、+23.5。Table 3 平均 52.25、60.35、63.75（印出 52.3、60.4、63.8）。

---

## 6. 實作計畫

### 6.1 檔案與共用外殼

- 單一檔案 `tut/441/build/index.html`；CSS、JS、時間線資料內嵌，不 `fetch` 本機檔案；上線時複製為 `tutorials/decision-diffuser.html`。
- **直接沿用** `tut/447/build/index.html`（repo 內現有檔案；主題最接近，外殼來自 `tutorials/prediction-targets-k-diff.html`）。複製的部分與位置（行號為 447 建置檔目前的行號，建置時以選擇器／函式名為準）：CSS 的 `:root` tokens 與 `.plate`、`.tscroll`、`.algo`（第 276–283 行：`.algo .row` 單欄格線、`.row.hd` 表頭、`code` 程式列、`.ex` 說明列；本頁 S9 的 Alg. 1 兩欄對照直接用這組類別，≥ 768px 時 447 的媒體查詢把 `.row` 改為兩欄）；JS 函式 `el(tag,a,p)`、`tx(p,x,y,s,a)`、`lp(pts)`、`widthOf(box,min,max)`（第 856–860 行）、`announce(id,s)`（500ms debounce，第 862 行）、`frameSched(fn)`（rAF 合併，第 863 行）、`segInit(box,opts,cur,label,onPick)`（第 952 行）、`renderTimeline()`（第 1073 行）、`initToc()`、`initTheme()`（第 1089–1093 行）、`selfTest()` 外框（第 1097 行，保留 `ok(c,m)` 與輸出格式，刪除 447 的斷言）、`renderMath()`（第 1142 行）。其餘一併沿用：CSS tokens（含深色模式三段覆寫與 `pm-theme`）、字型載入、appbar、`#pm-bar`、章節目錄、`.plate`／`.plate-h`／`.plate-f`、callout、`<details>` 檢查題、「本頁推導」「推定」pill、`.tscroll` 表格橫捲容器、KaTeX 0.16.9 設定與 `renderMath()`、`?test` selfTest 框架（`ok()` 計數、`scrollWidth <= innerWidth` 檢查）、`renderTimeline()`、SVG 輔助（`el`、`lp`、`tx`、`widthOf`）、`mulberry32`、`gauss`。447 的 `basis`、`qmat`、`reverseStep`、`exactMean` 等 DCT 玩具函式全部刪除。
- 以下只列新增或變更。

### 6.2 新增 tokens、元件與連結

- 新增 pill「本頁玩具」（同「推定」樣式，文字不同）。圖表色沿用 447：`--accent`（樣本、DD）、`--c-ref`（light #3F6E8C／dark #8DB4D3：目標分布、資料集 1）、`--ink-3`（資料分布、資料集 2、其他方法）；建置時確認圖形 ≥ 3:1、文字 ≥ 4.5:1（兩種 `--surface`），不足則調整並回寫此處。
- 條件 chip 列（`.cond-chips`，三個 pill，`aria-current` 標示當前）。
- S4 一維高斯分數圖 `#s4-score`（靜態 SVG，`role="img"`）：橫軸 $x\in[-3,3]$，`--ink-2` 曲線為 $\mathcal N(0,1)$ 密度；在 $x=\pm0.5,\pm1.5,\pm2.5$ 的基線上畫水平箭頭指向 0，箭頭長度正比於 $|x|$（分數 $-x$），標「分數 = −x：指向密度較高處」；寬度 = 容器寬，高 160px，文字 12–13px。
- S6 狀態陣列：HTML 格線 4 × 8（JetBrains Mono 欄標），< 480px 改 4 × 5 並以「…」欄結尾。
- S9 流程圖 `#s9-loop`（HTML/CSS，`role="img"`＋完整 `aria-label`）：節點「觀察 s → 放入歷史 h（長度 C）→ 從 N(0, αI) 開始去噪 K 步（每步：寫入歷史、Eq. 8、縮放變異數）→ 取 (sₜ, sₜ₊₁) → 逆動力學 f(sₜ, sₜ₊₁) → 執行 aₜ → 回到觀察」；≥ 768px 橫向排列、回圈箭頭在下方；< 768px 直向堆疊、文字 13px。
- 站內連結：`../index.html#paper-441`；arXiv `https://arxiv.org/abs/2211.15657`；專案網站 `https://anuragajay.github.io/decision-diffuser/`；筆記 `<slug>.html`。

### 6.3 程式結構（單一 IIFE；函式與 `core.py` 一一對應，註解寫出對應名稱）

- 共用：`SCH = schedule(100)` → `{beta, abar, bt}`（`cosine_schedule`）；`erf(x)`（A&S 7.1.26）。
- C1：`C1_DATA={m:[.25,.55,.85],pi:[.5,.35,.15],s:.07,sy:.25,top:.7}`；`Z1`（100 × 2000 噪聲表，`draws_1d`）；`compsUncond()`、`compsTilt(y, omega)`、`epsMix(x, k, comps)`、`c1Sample(omega, alpha, y, n)` → `Float64Array(n)`（`c1_sample_fast`，n = 500 預覽或 2000）、`mixStats(comps)`、`sampleStats(x)`；`initC1()` 建立兩個圖版 `#c1a`（α 固定 1）與 `#c1b`，共用 `state={omega, y, alpha, busy}`（`#c1a` 計算時以 α = 1）。
- C2：`diskPts()`、`annPts(r)`（快取 $u$ 陣列）、`Z2`（100 × 500 × 2 噪聲表）、`logDens(X,C,V)`、`g1(...)`、`c2Run(mode, omega, r, token)`（`c2_sample_fast`；切片，`setTimeout` 每片 1 個擴散步）、`c2Rates(X, r)`；`initC2()`，`state={mode, omega, r, showData, token, stale}`，`cache=new Map()`。
- C3：`T1`、`T1_AVG`、`T1K`、`T1K_AVG`、`A3` 常數（與 `tables.py` 逐字相同）；`rowStats(i)` → `{max, winners, rank, denom, higher}`、`colAvg()`、`a3Stats(pi)`；`initC3()`，`state={tab, row, showAvg, pi}`。
- `selfTest()` 新增斷言（容許誤差同第 5 節）：
  - 排程 abar[50] = 0.493844、bt[100] = 0.998758；PRNG 前三個值；`c1Sample(1.4,0.5,1,500)` 等於 `c1Sample(1.4,0.5,1,2000)` 的前 500 個（誤差 1e−12）。
  - C1：`compsTilt(1,1)` 權重 0.0362／0.3666／0.5972；`c1Sample(1.4,0.5,1)` 平均 0.8306、標準差 0.0874、比例 0.8920、`x[0]` 0.84555；`c1Sample(0,0,1)` 標準差 0（所有值相同）；`mixStats(compsTilt(1,2))` 平均 0.8275。
  - C2（同步跑，selfTest 不在乎時間）：AND、ω = 1.2、r = 0.7：兩者 0.926、`X0` (−0.5093, 0.4846)；AND、r = 1.3：兩者 0.000、平均半徑 1.1308；NOT、r = 0.7：< r 為 1.000；`g1` 在 k = 50、x = (0.5, 0.5) 的值（第 5 節）。
  - C3：`colAvg()` DD 81.822；DD 為最高的列數 6；HalfCheetah-MR `rank/denom` = 6/8；Mixed Kitchen = 1/4；`a3Stats(3)` 差 −0.9 且重疊。
  - DOM：`#s9-loop` 存在且含「逆動力學」文字；每個 LO 的 `<details>` 存在 12 個；`#c1a`、`#c1b` 都存在；隱藏的公式版本帶 `aria-hidden="true"`。
  - 輸出 `selfTest: N passed, M failed`。

### 6.4 研究筆記與時間線

- 本篇筆記（`tut/441/notes/`，7 篇）：`decision-diffuser-lineage-hub`（Decision Diffuser 的研究脈絡總覽）；`inverse-dynamics-state-diffusion`（只擴散狀態序列，以逆動力學模型取出動作）；`classifier-free-guidance-return-conditioning`（無分類器引導、報酬條件與低溫取樣）；`implicit-dp-trajectory-stitching`（不用價值函數的軌跡拼接（隱式動態規劃））；`composing-constraints-and-skills`（組合限制與技能（組合式擴散））；`dd-experiments-and-ablations`（Decision Diffuser 的實驗結果與消融）；`dd-limitations-stochastic-dynamics`（Decision Diffuser 的限制與隨機動力學）。
- hub 引用的站內既有筆記（title 從 `scripts/notes_split/` 取）：`rl-mdp-q-learning-basics`（MDP、Q 函數與 Bellman backup：value-based RL 的基本記號）、`offline-rl-distribution-shift`（Offline RL 的分布偏移與 OOD 動作造成的 Q 值高估）、`sequence-modeling-offline-rl`（以序列模型處理離線 RL：Decision Transformer 與 Trajectory Transformer）、`return-conditioned-sequence-modeling`（以報酬為條件的監督式學習與序列建模）、`trajectory-diffusion-planning`（軌跡層級擴散規劃（Diffuser））、`ddpm-training-and-sampling-recap`（DDPM 的訓練與取樣流程）、`diffusion-classifier-guidance-background`（擴散模型的分類器引導與餘弦排程）；另外 S2、S8 用到 `eps-prediction-ddpm-score`（ε-prediction 與 DDPM、score-based 模型）、`offline-rl-benchmarks-and-baselines`（本論文使用的離線 RL 基準與比較方法）。
- **延伸閱讀依 hub 的 5 步閱讀順序**：步驟 1 → S1（`rl-mdp-q-learning-basics`）、S2（`offline-rl-distribution-shift`）；步驟 2 → S5；步驟 3 → S3（DDPM）、S4（分類器引導、ε 與分數）、S5（Diffuser）；步驟 4 的四篇 → S6（逆動力學）、S7（無分類器引導）、S2（拼接）、S10（組合）；步驟 5 → S11、S15。S15 列出 hub 加上全部筆記，順序同 hub。Hero 筆記連結指向 hub。
- **時間線**：`timeline.json`（15 筆，`year` 整數，含本文 2211.15657）原樣貼進 `<script type="application/json" id="timeline-data">`；依 `year`、同年依 `arxiv` 字串排序。
- **`TL_GROUP`**（15 個 ID 全部指定）：
  | 類別 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | 擴散模型 | 1503.03585、2006.11239、2011.13456 | `ddpm-training-and-sampling-recap` |
  | 引導 | 2102.09672、2207.12598 | `classifier-free-guidance-return-conditioning` |
  | 逆動力學 | 1606.07419、1804.08606 | `inverse-dynamics-state-diffusion` |
  | 離線 RL 基準與基線 | 2004.07219、2006.04779、2110.06169 | `dd-experiments-and-ablations` |
  | 報酬條件策略 | 1912.13465、2106.01345 | `return-conditioned-sequence-modeling` |
  | 擴散規劃 | 2205.09991 | `trajectory-diffusion-planning` |
  | 組合式擴散 | 2206.01714 | `composing-constraints-and-skills` |
  | 本文 | 2211.15657 | `decision-diffuser-lineage-hub` |
  建置時以程式檢查每個 ID 都在 `TL_GROUP`；遺漏者歸「其他」並在建置回報中列出。

### 6.5 文案規則（新增部分）

- 用詞固定：「擴散步」（$k$）與「環境時間步」（$t$）；「無分類器引導（classifier-free guidance）」「引導強度 ω」「低溫係數 α」「逆動力學模型（inverse dynamics model）」「滾動視界（receding horizon）」「軌跡拼接（trajectory stitching）」「正規化分數」；第一次出現附英文。
- 論文結果（附出處）、「作者認為／作者假設」、〔本頁推導〕、「本頁玩具」四種陳述分開；玩具數字不與論文數字同句。
- 不寫「DD 全面優於」「總是更好」；比較一律寫成「平均」或指名資料集。

### 6.6 建置後自我檢查

依 447 頁流程：390／768／1024／1440 截圖＋深色、鍵盤 focus、reduced-motion（headless Chrome 無法連 CDN，截圖時以本機 KaTeX `scratchpad/katex/0.16.9` 代換，截圖後刪除代換檔）；逐項評分並修正；`?test` 全數通過；逐條核對 12 個 LO 的參考答案出現在指定章節正文。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 玩具與論文的差距 | C1 是一維、C2 只建模最終位置；沒有時間序列、逆動力學、環境；核寬、標籤雜訊、資料比例、排程都是推定值；C2 的 92.6% 與論文 91.3% 接近是巧合。 | 「本頁玩具」pill＋圖版底部說明；觀察項逐條限定；不把任何玩具數字放進 LO 或與論文數字同句（LO6 只問定性結論）。 |
| R2 | 「隱式動態規劃」的地位 | 論文稱之為假設（hypothesize），只有 Maze2D 一張圖（App. A.1），無定量結果。 | S2、S11 寫成「作者假設」，不做元件。 |
| R3 | Alg. 1 的索引 | $x_0(\tau)=(s_t,\dots,s_{t+H-1})$ 與「前 length(h) 欄寫入歷史」的關係、$s_t$ 所在欄位都沒有寫明。 | 本頁慣例（推定，只用於流程圖文字）：計畫的前 length(h) 欄是歷史，最新觀察 $s$ 是第 length(h) 欄（1 起算），$(s_t,s_{t+1})$ 取該欄與下一欄。S9 照錄 Alg. 1，以註腳說明論文未寫明；流程圖不畫欄位索引。 |
| R4 | 原文數字的不一致 | Table 2 的 Diffuser Hopper-ME 107.6 對 Table 1 的 107.2；DD Hopper-ME 標準誤 ±1.6（Table 2、A1）對 ±1.8（Table 1）；α ∈ [0,1) 對 Table A1 的 α = 1.0；步態分類器 1e6 步（App. B）對 2e6 步（App. I.1）；Table 1 的平均列是四捨五入值（Kitchen CQL 51.25 印成 51.2）。 | 各表照錄，S6、S11、S13 以註腳指出。 |
| R5 | App. A.1 的文句 | 任務是 A 到 C，原文寫「form trajectories that goes from point A to point B」。 | 照錄並註明不一致，不改寫成 A 到 C 的定量主張。 |
| R6 | 步態比例 | Fig. 7 表格（Trot／Pace／Bound）Bound + Pace 為 1.4／38.5／60.1；正文寫「bounding 38.5%、pacing 60.1%」。 | 依表格欄位（Bound 60.1、Pace 38.5），註腳指出正文相反；筆記 `composing-constraints-and-skills` 也依表格。 |
| R7 | 排程與反向變異數 | 論文沒寫排程與 $\Sigma_k$；玩具用 cosine 與 $\tilde\beta_k$。C1 在 ω = 0、α = 1 時樣本標準差 0.2202 低於資料 0.2289（ω = 1 時 0.1605 對 0.1709），來自有限 $K$ 與 $\tilde\beta$ 的選擇。 | 標「推定」；S7 玩具段寫一句這個偏差，避免讀者以為 ω = 1 時樣本完全等於 $p(x\mid y)$。 |
| R8 | ε 與分數的符號 | Sec. 2.2 寫 ∝ 省略負號，App. D 有負號。 | 頁面用帶負號的式子並註明。 |
| R9 | 符號衝突 | α（排程／低溫）、R 與 r（報酬函數、報酬、獎勵／半徑）、β（Bernoulli／App. J 係數／玩具排程 βₖ）、p（丟棄機率／隨機動作機率／密度）、h（佇列／維度）、C（歷史長度／限制集合）、ω 與 $s$。 | 1.4 符號表列出；頁面在每個衝突第一次出現處就地註明。玩具排程的 βₖ 只出現在 `<details>` 內。 |
| R10 | 效能 | C2 每次約 4e7 次 exp；C1 約 1.2e6 次 exp。 | C2：每片 1 個擴散步、快取、只在 `change`（鍵盤 250ms debounce）後重算、首次捲入才算、拖動中舊樣本半透明。C1：拖動中算前 500 個樣本（與完整結果的前 500 個相同），放開後算 2000 個；rAF 合併且忙碌時跳過。不用 Web Worker（單檔頁面，切片已在 10ms 量級）。 |
| R11 | 結果的措辭 | 摘要寫「outperform existing offline RL approaches」；Table 1 中 DD 只在 4／9 運動控制列最高。 | Hero、S11、LO10 都寫出「平均最高、不是每列最高」並指名例子。 |
| R12 | 正規化分數的定義 | 論文只引用 Fu et al. 2020，未寫定義。 | S11 以一句說明並標「依 D4RL 定義」，連到 `offline-rl-benchmarks-and-baselines`。 |
| R14 | 條件獨立的兩處寫法 | Sec. 3.3 寫給定 $x_0(\tau)$，App. D 寫給定 $x_k(\tau)$；推導需要後者。 | 步驟 7、S10、LO9 並列兩處原文並說明推導需要哪一個。 |
| R15 | C1 的外推 | 大 ω 使玩具樣本越過資料中最好的行為；論文沒有外推的主張。 | y 滑桿上限 1；觀察 (4) 明寫是玩具現象。 |
| R13 | 相對連結 | 建置位置與上線位置不同。 | 接受；預覽只檢查外部連結與頁內錨點。 |

---

## 8. 修訂紀錄（round 1）

依 `reviews/round1.md`（30 項，2 項 P0）修訂，項次對應該檔：
- #1（P0）6.1 列出 447 建置檔中複製的 CSS 選擇器與函式簽名（含行號）；S9 用 447 的 `.algo`。#2（P0）C2 寫出 $\ell_i$ 與 $g_1$；`core.g1_at`、`sweep.py` 輸出 g₁ 驗收值（第 5 節）。
- #3–#17（fidelity）：條件獨立兩處寫法（步驟 7、LO9、R14）；y 上限 1、外推標為玩具現象（C1、R15，驗收列 y=1.2 換成 y=0.75）；LO3 加最佳預測器與邊際分布；Table 3 區間重疊與 OOD 註；ᾱ 索引差一（符號表、4.0）；App. K 改為未驗證的主張；排程理由；兩個理由；LO6 刪玩具句；ω=0／1 標本頁推導；NOT 集中為玩具假象；Fig. 6、7；暖啟動限定 Hopper-ME、作者報告；LO4 的 +0.5 在標準誤內與 Table 2 的混雜；Alg. 1 第 7 行記號。
- #18–#21：大綱改為 S0–S15（拆 S1/S2、S3/S4、原 S5 為 S7/S8）；S4 新增分數定義與一維高斯靜態圖；S5 先定義 $x_0(\tau)$；C1 拆成 C1a（S7）、C1b（S8）。
- #22–#30：`sweep.py` 全網格掃描；C1 拖動算前 500 個樣本、C2 每片 1 步；動態文字不用 ᾱ；C3 名次分母依列；C2 拖動中舊樣本半透明、鍵盤 250ms debounce；Hero 卡術語；用詞；aria-valuetext、min-height、reduced-motion 涵蓋 JS 動畫；隱藏公式 `aria-hidden`、LO／符號表窄螢幕堆疊。
- verify/：`core.py` 新增步驟優先噪聲表的快速版本（與原版斷言相等）與 `g1_at`；`sweep.py`（輸出 `sweep_out.txt`）；`check.py` 改用 y=0.75 列並重新產生 `check_out.txt`，其餘數字不變。

## 9. 修訂紀錄（readiness）

- 阻擋項：§6.4 的筆記清單統一在 S15；S1 就地定義 γ、ρ₀、𝒯。
- 非阻擋項：C3 陣列內嵌頁面；§5 C3 名次的列順序；§4.0 排程索引範圍；Hopper-ME Diffuser 顯示 107.2（Table 1）；Alg. 1 列索引的本頁慣例（R3）；Hero 卡 1 寫「報酬 y」。

## 10. 建置紀錄（build round）

- 產出 `build/index.html`（約 145 KB，單檔）。外殼沿用 447 建置檔（其外殼來自 k-diff 頁）：CSS tokens、`#pm-bar`、TOC、plate、KaTeX 設定與 `?test` 框架。`?test`：56 passed, 0 failed（1440 與 390 寬都通過，含無水平捲動檢查）。impeccable：0 個 anti-pattern（15 則 numbered-section-labels advisory 保留）。LO 自檢：`accept/page.txt` 中 12 個 LO 的關鍵詞與數字都出現在正文。
- 與設計的差異：S9 流程圖 id 為 `#s9-loop`（HTML/CSS）；Table 1 的 HTML 表由 `verify/tables.py` 在組裝時產生，並以 ▲ 標各列最高；`details.check` 共 13 個（S13、S14 無檢查題，S7 一個框含 LO5、LO6）；窄螢幕的 Sec. 3.4 損失改為三行（以 $\tilde y$ 表示丟棄後的條件，只在 KaTeX 內）；C2 圖例改用與圖上相同的點／叉號樣本，標籤加底色描邊。
- 截圖：`build/caps/`（1440 寬 C1a、C1b、C2、C3、hero；390 寬 hero、C1a、C2、C3、規劃迴圈；深色 1440 寬 C1b、C2）。截圖用的本機 KaTeX 預覽檔已刪除。headless Chrome 連 CDN 會卡住，測試一律以本機 KaTeX 代換。
