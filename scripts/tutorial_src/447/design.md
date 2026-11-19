# 設計文件：〈Planning with Diffusion for Flexible Behavior Synthesis〉互動教學

- 論文：Michael Janner*、Yilun Du*、Joshua B. Tenenbaum、Sergey Levine（* 同等貢獻），arXiv 2205.09991（v2），ICML 2022（PMLR 162）。資料庫編號 No. 447。方法名稱 **Diffuser**。
- 原文：`tut/447/src/paper.txt`（PDF 同目錄）。`tut/` = `scripts/tutorial_src/`。本文件的「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」「Alg. 1」皆指論文原文編號。論文只有 Eq. 1–3 有編號；未編號的式子以「Sec. 3.1 訓練目標」這種方式指稱。
- 產出：`tut/447/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/diffuser-planning-with-diffusion.html`（slug `diffuser-planning-with-diffusion`）。站內相對連結一律以上線位置為準。
- 驗算腳本（非頁面程式，與頁面同一演算法）：`tut/447/verify/core.py`（共用數學）、`check.py`（驗收表，輸出 `check_out.txt`）、`mc.py`（Monte Carlo 交叉檢查，`mc_out.txt`）、`c3_alpha.py`（C3 乾淨 J 模式的 α 網格，`c3_alpha_out.txt`）。C3 的兩種引導模式都在 `check.py`（`core.make_guide`）。
- 研究筆記：`tut/447/notes/`（總覽 `traj-diffusion-lineage-hub` ＋ 6 篇主題筆記 ＋ `timeline.json` 15 筆），見 6.4。
- 審查：`tut/447/reviews/`（fidelity、pedagogy、合併清單 `round1.md`）；修訂紀錄見第 8 節。
- 標記慣例：**〔本頁推導〕** = 本設計自行推導、論文沒有寫出；**推定** = 為貼近論文或為了玩具可視化而選定、論文沒有寫明的值；「作者認為」= 論文作者的解釋，不是實驗結果。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

基於模型的強化學習（model-based RL）常見做法是先學一個動態模型 $\hat f$，再交給經典的軌跡最佳化器規劃。作者指出這個組合實際上很少照描述運作：強力的最佳化器會利用學到模型的誤差，產生的計畫「更像對抗樣本而不是最佳軌跡」（Sec. 1）。本文改為把盡量多的軌跡最佳化流程併入生成模型，使「從模型取樣」與「用模型規劃」幾乎相同。具體做法是 **Diffuser**：一個對整條軌跡（狀態與動作排成的二維陣列，Eq. 2）建模的擴散模型 $p_\theta(\tau)$，以時間方向的一維卷積 U-Net 一次預測所有時間步（Sec. 3.1）。規劃寫成從擾動分布 $\tilde p_\theta(\tau)\propto p_\theta(\tau)h(\tau)$ 取樣（Eq. 1）：報酬最大化對應分類器引導取樣（Sec. 3.2，Eq. 3，Alg. 1），目標條件規劃對應影像補全（inpainting，Sec. 3.3）。在 Maze2D 長時程稀疏報酬任務中 Diffuser 單任務平均 119.5、多任務平均 129.4，最佳的 model-free 基準 IQL 為 47.0 與 16.9（Table 1）；積木堆疊平均 54.4，CQL 8.1、BCQ 0.0（Table 3）；D4RL 運動控制平均 77.5，與 CQL 77.6、IQL 77.0 相近（Table 2）。

### 1.2 研究動機（Sec. 1、Sec. 3 開頭）

1. **學到的模型不適合經典最佳化器**：經典軌跡最佳化器是為真實模型設計的；接上學到的模型時，最佳化器會找出模型預測錯誤、卻被誤判為高報酬的動作序列（Talvitie 2014；Ke et al. 2018）。因此當代 model-based 方法多半借用 model-free 的價值函數與策略梯度，或只用隨機射擊（random shooting）、交叉熵法（CEM）等不需梯度的簡單最佳化器（Sec. 1）。
2. **模型應依規劃需求設計**：動態模型通常依因果過程設計；作者主張改以「它將被用來規劃」為設計原則：動作分布和狀態動態一樣重要、長時程準確度比單步誤差重要、模型應與報酬函數無關以便重用於多個任務、計畫（而不只是預測）應隨經驗變好，並避免射擊式規劃的短視失敗（Sec. 1）。

### 1.3 貢獻（本頁整理自 Sec. 1；論文沒有編號清單）

1. 為軌跡資料設計的去噪擴散模型 Diffuser，以及一個行為合成的機率框架（Eq. 1）。
2. 兩種規劃方式：把分類器引導取樣重新解讀為報酬最大化（Sec. 3.2）、把影像補全重新解讀為目標條件規劃（Sec. 3.3）。
3. 四個性質（Sec. 4，Fig. 3）：長時程規劃、時間組合性（temporal compositionality）、可變長度計畫、任務組合性（task compositionality）。
4. 實驗：長時程（Maze2D/Multi2D）、測試時彈性（積木堆疊）、離線 RL（D4RL）、暖啟動加速（Sec. 5.4，Fig. 7）。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S2 末尾以可收合表格提供）

| 符號 | 意義 |
| --- | --- |
| $s_t,a_t$ | 第 $t$ 個規劃時間步的狀態、動作；動態 $s_{t+1}=f(s_t,a_t)$（Sec. 2.1） |
| $T$ | 規劃視界（planning horizon）；軌跡 $\tau=(s_0,a_0,\dots,s_T,a_T)$ |
| $r(s_t,a_t)$、$\mathcal J(\tau)$ | 單步報酬、軌跡目標 $\sum_{t=0}^T r(s_t,a_t)$ |
| $i$、$N$ | 擴散步（上標）、擴散總步數；$\tau^0$ 是無噪軌跡，$\tau^N\sim\mathcal N(0,I)$ |
| $\tau^i_{s_t}$、$\tau^i_{a_t}$ | 擴散步 $i$ 的軌跡中第 $t$ 個狀態、動作（Sec. 2.2 Notation）；無歧義時省略上標 $0$ |
| $q(\tau^i\mid\tau^{i-1})$、$p_\theta(\tau^{i-1}\mid\tau^i)$ | 前向加噪、反向去噪 |
| $\mu_\theta,\Sigma^i$ | 反向轉移的平均與（固定）共變異數 |
| $\epsilon_\theta(\tau^i,i)$ | 網路預測的噪聲 |
| $h(\tau)$、$\tilde p_\theta$ | 擾動函數、擾動分布（Eq. 1） |
| $\mathcal O_t$ | 第 $t$ 步「最佳」的二元變數，$p(\mathcal O_t=1)=\exp(r(s_t,a_t))$（Sec. 3.2） |
| $\mathcal J_\phi$、$\alpha$ | 學到的報酬預測器、引導尺度（Alg. 1） |
| $c_t$ | 第 $t$ 步的狀態限制值（Sec. 3.3） |
| 頁面專用 $H$ | 軌跡陣列的欄數（玩具中 $=T+1$） |
| $\beta_i,\bar\alpha_i,\tilde\beta_i$ | 擴散排程：第 $i$ 步加入的噪聲比例、累積保留比例 $\prod_{j\le i}(1-\beta_j)$、反向步的變異數；S2 以 cosine 排程的公式定義（讀數中 $\tilde\beta$ 寫成 βt） |

#### 步驟 1：問題設定——軌跡最佳化（Sec. 2.1）

$$a^*_{0:T}=\arg\max_{a_{0:T}}\mathcal J(s_0,a_{0:T})=\arg\max_{a_{0:T}}\sum_{t=0}^T r(s_t,a_t)$$

- 需要知道動態 $f$；學習式方法以近似模型 $\hat f$ 代替，這正是動機 1 的問題來源（Sec. 3 開頭）。
- 本文設定是**離線**：只用既有資料集訓練，不與環境互動收集新資料（Sec. 5.3「heterogeneous data of varying quality」）。

#### 步驟 2：擴散模型背景（Sec. 2.2）

$$p_\theta(\tau^0)=\int p(\tau^N)\prod_{i=1}^N p_\theta(\tau^{i-1}\mid\tau^i)\,d\tau^{1:N},\qquad p_\theta(\tau^{i-1}\mid\tau^i)=\mathcal N\big(\tau^{i-1}\mid\mu_\theta(\tau^i,i),\Sigma^i\big)$$

- $p(\tau^N)$ 是標準高斯；參數以最小化負對數似然的變分上界訓練；前向過程 $q$ 預先指定。
- **兩種時間**（Sec. 2.2 Notation）：上標 $i$ 是擴散步，下標 $t$ 是規劃時間步。例：$s^0_t$ 是無噪軌跡的第 $t$ 個狀態。頁面以此為第一個常見錯誤。
- 頁面在 S2 補上 DDPM 的標準式（Ho et al. 2020，論文引用但未寫出），只放後面用得到的：
  - 閉式加噪：$q(\tau^i\mid\tau^0)=\mathcal N(\sqrt{\bar\alpha_i}\,\tau^0,(1-\bar\alpha_i)I)$，也就是 $\tau^i=\sqrt{\bar\alpha_i}\,\tau^0+\sqrt{1-\bar\alpha_i}\,\epsilon$。可以從無噪軌跡一步跳到任何擴散步 $i$，不必逐步加噪。S7 的暖啟動用的就是這個式子：把上一個計畫當作 $\tau^0$，取 $i=k$。
  - 反向平均：$\mu=\frac{1}{\sqrt{1-\beta_i}}\big(\tau^i-\frac{\beta_i}{\sqrt{1-\bar\alpha_i}}\epsilon_\theta(\tau^i,i)\big)$，其中 $\beta_i=1-\bar\alpha_i/\bar\alpha_{i-1}$。
  - 排程定義（S2 內文，放在閉式加噪之前）：論文用 cosine 排程（Nichol & Dhariwal 2021）：$f(i)=\cos^2\!\big(\frac{i/N+0.008}{1.008}\cdot\frac\pi2\big)$，$\beta_i=\min(1-f(i)/f(i-1),\,0.999)$，$\bar\alpha_0=1$，$\bar\alpha_i=\prod_{j\le i}(1-\beta_j)$，$\tilde\beta_i=\beta_i\frac{1-\bar\alpha_{i-1}}{1-\bar\alpha_i}$。S3b 只回指這些式子，並說明本頁取 $\Sigma^i=\tilde\beta_iI$（推定）。

#### 步驟 3：把規劃寫成擾動分布取樣（Sec. 3，Eq. 1）

$$\tilde p_\theta(\tau)\propto p_\theta(\tau)\,h(\tau)\tag{Eq. 1}$$

- $h(\tau)$ 可放先前觀測（observation history）、想要的結果（目標）、或一般的待最佳化函數（報酬或成本）。
- 在 $\tilde p_\theta$ 中推論，相當於 Sec. 2.1 軌跡最佳化的機率版本：軌跡要在 $p_\theta$ 下物理上合理、在 $h$ 下高報酬或滿足限制。
- 動態資訊（$p_\theta$）與擾動（$h$）分開，所以**同一個擴散模型可重用於同一環境的多個任務**。這是任務組合性與 Multi2D、積木堆疊實驗的基礎。

#### 步驟 4：為規劃而設計的軌跡模型（Sec. 3.1）

1. **不依時間順序自迴歸**：考慮目標條件推論 $p(s_1\mid s_0,s_T)$，$s_1$ 同時依賴過去與未來。動態預測是因果的，決策與控制可以是反因果的（anti-causal）：現在的決策取決於未來（腳註 1：一般 RL 中來自「未來最佳」的假設，例如 $\log p(a_t\mid s_t,\mathcal O_{t:T})$，Levine 2018）。所以 Diffuser 同時預測計畫的所有時間步。
2. **時間局部性**：Fig. 2 是單一時間卷積的依賴圖：一個輸出只看到鄰近的過去與未來時間步。每個去噪步只能依軌跡的**局部一致性**做預測；組合許多去噪步後，局部一致性可以推動**全局連貫**。
3. **軌跡表示**（Eq. 2）：狀態與動作聯合預測，動作只是狀態之外的額外維度：

$$\tau=\begin{bmatrix}s_0&s_1&\cdots&s_T\\a_0&a_1&\cdots&a_T\end{bmatrix}\tag{Eq. 2}$$

   每一欄是一個規劃時間步。
4. **架構**：三個要求——(1) 非自迴歸地預測整條軌跡；(2) 每個去噪步時間局部；(3) 沿規劃時間維度等變（equivariant），沿狀態／動作特徵維度不等變。作者用重複的（時間）卷積殘差區塊組成類 U-Net 結構，把影像擴散的二維空間卷積換成一維時間卷積（Fig. A1）。全卷積，所以**視界由輸入維度決定而不是架構**，規劃時可以動態改變。App. C：6 個重複殘差區塊；每區塊兩個時間卷積，各接 group norm，最後 Mish；擴散步嵌入由單一全連接層產生，加到每區塊第一個時間卷積的激活上。
5. **訓練**（Sec. 3.1）：

$$\mathcal L(\theta)=\mathbb E_{i,\epsilon,\tau^0}\big[\|\epsilon-\epsilon_\theta(\tau^i,i)\|^2\big],\quad i\sim\mathcal U\{1,\dots,N\},\ \epsilon\sim\mathcal N(0,I)$$

   $\tau^i$ 是 $\tau^0$ 加上噪聲 $\epsilon$ 的結果；$\mu_\theta$ 可由 $\epsilon_\theta$ 閉式求得（Ho et al. 2020）；反向共變異數 $\Sigma^i$ 依 Nichol & Dhariwal (2021) 的 cosine 排程。App. C 第 9 點：預測噪聲 $\epsilon$ 或預測無噪資料 $\tau^0$，控制表現沒有顯著差別。

#### 步驟 5：報酬最大化＝引導取樣（Sec. 3.2，Eq. 3，Alg. 1）

- 控制即推論（control-as-inference，Levine 2018）：$\mathcal O_t$ 是二元變數，$p(\mathcal O_t=1)=\exp(r(s_t,a_t))$。取 $h(\tau)=p(\mathcal O_{1:T}\mid\tau)$：

$$\tilde p_\theta(\tau)=p(\tau\mid\mathcal O_{1:T}=1)\propto p(\tau)\,p(\mathcal O_{1:T}=1\mid\tau)$$

- 當 $p(\mathcal O_{1:T}\mid\tau^i)$ 足夠平滑，反向轉移可近似為高斯（Sohl-Dickstein et al. 2015）：

$$p_\theta(\tau^{i-1}\mid\tau^i,\mathcal O_{1:T})\approx\mathcal N(\tau^{i-1};\mu+\Sigma g,\Sigma)\tag{Eq. 3}$$
$$g=\nabla_\tau\log p(\mathcal O_{1:T}\mid\tau)\big|_{\tau=\mu}=\sum_{t=0}^T\nabla_{s_t,a_t}r(s_t,a_t)\big|_{(s_t,a_t)=\mu_t}=\nabla\mathcal J(\mu)$$

  $\mu,\Sigma$ 是原反向轉移 $p_\theta(\tau^{i-1}\mid\tau^i)$ 的參數。
- 〔本頁推導〕從 Eq. 3 到 $g=\nabla\mathcal J(\mu)$ 用到三個論文沒有寫出的條件：(1) 給定 $\tau$ 時 $\mathcal O_t$ 彼此獨立，所以 $\log p(\mathcal O_{1:T}=1\mid\tau)=\sum_t r(s_t,a_t)$；(2) $\exp(r)$ 要是機率，所以 $r\le0$。把報酬平移不改變 $g$，把報酬縮放會改變 $g$；(3) $p(\mathcal O\mid\tau^i)\approx p(\mathcal O\mid\mu)$：把帶噪的 $\tau^i$ 換成反向平均 $\mu$，並把雜訊層級的似然換成乾淨軌跡的報酬。第 (3) 點是另一個近似，不是恆等式。論文的 $\mathcal O$ 下標為 $1{:}T$、求和為 $t=0..T$，頁面照錄並加註腳。
- 實作：先在所有可用軌跡的狀態與動作上訓練 $p_\theta(\tau)$，再另外訓練 $\mathcal J_\phi$，以**帶噪軌跡樣本 $\tau^i$** 為輸入預測累積報酬（論文沒有寫明預測的是 $\tau^i$ 本身的報酬還是對應 $\tau^0$ 的報酬），用 $\mathcal J_\phi$ 的梯度依 Eq. 3 修改反向平均。以帶噪樣本為輸入，正是讓條件 (3) 的近似較不嚴重的做法。取樣出的軌跡只執行第一個動作，然後在標準的滾動視界控制（receding-horizon control）迴圈中重新規劃。
- **Alg. 1 Guided Diffusion Planning**（頁面以兩欄格線逐行對照，沿用 k-diff 頁 S8 的版式）：

  ```
  Require Diffuser μθ, guide J, scale α, covariances Σ^i
  while not done:
      observe state s; initialize plan τ^N ~ N(0, I)
      for i = N, ..., 1:
          μ ← μθ(τ^i)                              # 反向轉移參數
          τ^{i-1} ~ N(μ + α Σ ∇J(μ), Σ^i)          # 以報酬梯度引導
          τ^{i-1}_{s_0} ← s                        # 限制計畫的第一個狀態
      execute first action of plan τ^0_{a_0}
  ```

  Alg. 1 比 Eq. 3 多一個引導尺度 $\alpha$。論文的目標 $p(\tau)\exp(\mathcal J)$ 對應 $\alpha=1$；把 $\alpha$ 讀成「從 $p(\tau)\exp(\alpha\mathcal J)$ 取樣」是分類器引導的常見讀法（Dhariwal & Nichol 2021，論文有引用），本頁 C3 採用這個讀法作為精確參考，並標明這是本頁的詮釋。App. C：除 hopper-medium-expert 用 0.0001 外，所有任務 $\alpha=0.1$。$\alpha$ 的數值與 $\mathcal J_\phi$ 的尺度綁在一起，不能跨模型比較。
- 〔本頁推導〕Alg. 1 第 8 行的引導部分可寫成 $\mu+\alpha\tilde\beta_i\nabla\mathcal J(\mu)$（取 $\Sigma=\Sigma^i=\tilde\beta_iI$，見 R2）。若 $\mathcal J=-\tfrac12\sum_t\|s_t-c\|^2$，這一項讓 $\mu-c$ 乘上 $(1-\alpha\tilde\beta_i)$。$\alpha\tilde\beta_i>2$ 是這一項在單一步放大與 $c$ 距離的必要條件，但是否發散取決於所有步驟的累積增益（包含去噪本身的收縮）。本頁玩具（乾淨 J 模式，$N=20$）中，$\alpha=10$ 有 7 步 $\alpha\tilde\beta_i>2$，但沒有發散；$\alpha=10^{1.2}$ 有 10 步，並在第 13 步發散。

#### 步驟 6：目標條件 RL＝補全（Sec. 3.3）

- 有些問題是限制滿足而非報酬最大化，例如「終點在目標位置」。依 Eq. 2 的二維陣列表示，這是補全問題：狀態與動作限制就像影像中已觀測的像素，其餘位置由擴散模型以一致的方式填入。
- 擾動函數：觀測值為 Dirac delta，其他地方為常數。若 $c_t$ 是第 $t$ 步的狀態限制：

$$h(\tau)=\delta_{c_t}(s_0,a_0,\dots,s_T,a_T)=\begin{cases}+\infty&c_t=s_t\\0&\text{otherwise}\end{cases}$$

  動作限制的定義相同。$+\infty$ 是記號上的寫法，表示「只允許 $s_t=c_t$」，不是實際的密度值；頁面照錄並加註。
- 實作：從無擾動的反向過程 $\tau^{i-1}\sim p_\theta(\tau^{i-1}\mid\tau^i)$ 取樣，並在**每個**擴散步 $i\in\{0,1,\dots,N\}$ 之後把被限制位置的取樣值換成 $c_t$（論文原文的集合寫成含 0 到 $N$，頁面照錄）。
- 報酬最大化也需要補全：所有計畫必須從目前狀態開始，即 Alg. 1 第 10 行。
- 〔本頁推導〕替換法不是精確的條件取樣：被限制的值只能經由之後的去噪步傳到其他時間步。在本頁的高斯玩具（4.0）中，替換法產生的計畫平均在端點附近有跳躍，內部被拉向先驗平均；在 $N\le100$ 的範圍內，$N$ 越大差距越小，但沒有看到歸零（C2，第 5 節）。這是玩具中的現象，不代表論文模型的定量表現。

#### 步驟 7：四個性質（Sec. 4，Fig. 3）

| 性質 | 論文的機制說明 | Fig. 3 |
| --- | --- | --- |
| 學到的長時程規劃 | 規劃幾乎等於取樣（差別只有 $h$ 的引導），所以 Diffuser 作為長時程預測器的效果直接轉成長時程規劃的效果；在稀疏報酬下仍能產生可行軌跡，射擊式方法在這類設定中困難 | (a) |
| 時間組合性 | 以反覆改善局部一致性產生全局連貫軌跡，因此能把熟悉的子序列以新方式拼接；只用直線軌跡訓練，能產生在交點組合而成的 V 形軌跡 | (b) |
| 可變長度計畫 | 沿視界方向全卷積，視界由初始噪聲 $\tau^N$ 的大小決定 | (c) |
| 任務組合性 | 模型與報酬函數無關，是可能未來的先驗；可用輕量的 $h$（或多個 $h$ 的組合）引導，規劃訓練時沒見過的報酬 | (d) |

#### 步驟 8：暖啟動（Sec. 5.4，Fig. 7）

- 限制：計畫需要迭代產生，速度慢；開迴路執行時每一步都要重新產生計畫。
- 做法：從上一個計畫跑少數幾步**前向**擴散，再從這個部分加噪的軌跡跑相同步數的去噪，得到更新的計畫。〔本頁推導〕「前向 $k$ 步」可以用 S2 的閉式加噪一步完成：$\tau^k=\sqrt{\bar\alpha_k}\,\tau_{\text{prev}}+\sqrt{1-\bar\alpha_k}\,\epsilon$，再從 $i=k$ 去噪到 0。
- 執行方式：Alg. 1 是每個環境步都重新規劃（滾動視界），這是運動控制實驗的用法；Maze2D 則把取樣出的計畫當作**開迴路**計畫執行（Sec. 5.1）。Sec. 5.4 一方面說「execute plans open loop」，一方面又說每個執行步都要重新產生計畫，原文在這裡不一致，頁面照錄並註明。
- Fig. 7（Walker2d Medium-Expert）：每次重新產生計畫的去噪步數從 2 變到 100；只用十分之一的擴散步數時，只要以前一個時間步的計畫初始化，表現只小幅下降。
- 論文沒有說明上一個計畫是否要先平移一欄（因為已執行一步）。頁面照錄原文，不補充（R5）。

#### 步驟 9：實驗（Sec. 5，Table 1–3）

- **Maze2D（Sec. 5.1，Table 1）**：抵達目標得到報酬 1，其他地方沒有報酬塑形（reward shaping），可能要數百步才抵達。Diffuser 以補全條件化起點與目標，取樣的軌跡當作開迴路計畫。訓練資料是無方向的（在隨機選取的位置之間來回導航），評估是單任務（目標固定）；Multi2D 在每個 episode 開始時重新抽目標，Diffuser 不需重新訓練，只換條件目標。

  | 環境 | MPPI | CQL | IQL | Diffuser |
  | --- | --- | --- | --- | --- |
  | Maze2D U-Maze | 33.2 | 5.7 | 47.4 | 113.9 ± 3.1 |
  | Maze2D Medium | 10.2 | 5.0 | 34.9 | 121.5 ± 2.7 |
  | Maze2D Large | 5.1 | 12.5 | 58.6 | 123.0 ± 6.4 |
  | 單任務平均 | 16.2 | 7.7 | 47.0 | 119.5 |
  | Multi2D U-Maze | 41.2 | – | 24.8 | 128.9 ± 1.8 |
  | Multi2D Medium | 15.4 | – | 12.1 | 127.2 ± 3.4 |
  | Multi2D Large | 8.0 | – | 13.9 | 132.1 ± 5.8 |
  | 多任務平均 | 21.5 | – | 16.9 | 129.4 |

  分數超過 100 表示優於參考的專家策略。MPPI 使用**真實動態**，表現仍差，作者認為這說明即使沒有預測誤差，長時程規劃本身也困難。多任務 IQL 以 hindsight experience relabeling 改成目標條件（App. A.1：$\Delta\sim\mathrm{Geom}(1-\gamma)$，$g=s_{t+\Delta}$）。
- **積木堆疊（Sec. 5.2，Table 3）**：三種設定——無條件堆疊（越高越好）、條件堆疊（指定順序）、重新排列（匹配參考積木在新排列中的位置）；PDDLStream 產生的 10000 條示範軌跡；每次成功放置得到報酬 1。只用一個訓練好的 Diffuser，只換 $h$：無條件堆疊直接從 $p_\theta$ 取樣；條件堆疊與重新排列組合兩個 $h$（App. B：$h_{\text{match}}$ 是逐時間步的分類器，判斷狀態是否呈現 A 疊在 B 上；$h_{\text{contact}}(\tau)=\sum_{i=0}^{64}-1\cdot\|\tau^i_c-1\|^2$，$\tau^i_c$ 是狀態中表示手臂與積木 A 接觸的維度，作用在軌跡前 64 個時間步）。100 為完美執行，0 為隨機策略。

  | 環境 | BCQ | CQL | Diffuser |
  | --- | --- | --- | --- |
  | 無條件堆疊 | 0.0 | 24.4 | 58.7 ± 2.5 |
  | 條件堆疊 | 0.0 | 0.0 | 45.6 ± 3.1 |
  | 重新排列 | 0.0 | 0.0 | 58.9 ± 3.4 |
  | 平均 | 0.0 | 8.1 | 54.4 |

- **離線 RL（Sec. 5.3，Table 2）**：D4RL 運動控制；以 Sec. 3.2 引導、以 Sec. 3.3 條件化目前狀態；$\mathcal J_\phi$ 與擴散模型用同一批軌跡訓練。平均：BC 51.9、CQL 77.6、IQL 77.0、DT 74.7、TT 78.9、MOPO 42.1、MOReL 72.9、MBOP 47.8、Diffuser 77.5（150 個規劃種子的平均與標準誤）。作者的描述：優於 MOReL、MBOP、DT，遜於專為單任務設計的最佳離線方法。頁面摘錄平均列與 Medium-Expert 三列（HalfCheetah 88.9 ± 0.3、Hopper 103.3 ± 1.3、Walker2d 106.9 ± 0.2），並加一句說明較弱的列：Medium-Replay HalfCheetah 37.7 低於 CQL 45.5、TT 41.9、MOPO 53.1；Medium-Replay Walker2d 70.6 低於 TT 82.6。論文依 Kostrikov et al. (2022) 把每個任務中 ≥ 0.95 · 最大值的分數加粗。全表（9 列＋平均列，含所有基準）放在可收合的 `<details class="more">`，並依論文標出每列 ≥ 0.95·max 的粗體。
- **把 Diffuser 當動態模型放進 MPPI**：表現不比隨機好；作者認為 Diffuser 的效果來自建模與規劃的耦合，而不是開迴路預測準確度（Sec. 5.3）。

#### 步驟 10：實作細節（App. A、C）

| 項目 | 值（App. C 項次） |
| --- | --- |
| 架構 | U-Net，6 個殘差區塊；兩個時間卷積＋GN，Mish；擴散步嵌入以單一 FC 加到第一個卷積（1） |
| 最佳化 | Adam，學習率 4e−05，batch 32，500k 步（2） |
| 報酬預測器 $\mathcal J$ | U-Net 前半段＋最後一層線性輸出純量（3）；折扣 0.997，$\gamma>0.99$ 時規劃對折扣不敏感（8） |
| 視界 $T$ | 運動控制 32；積木堆疊 128；Maze2D／Multi2D U-Maze 128、Medium 265、Large 384（4） |
| 縮短視界 | 許多任務可縮短視界，但引導尺度要降低，例如 halfcheetah 視界 4 時用 0.001（5） |
| 擴散步數 $N$ | 運動控制 20，積木堆疊 100（6）；Maze2D 未寫明 |
| 引導尺度 $\alpha$ | 0.1；hopper-medium-expert 0.0001（7） |
| 預測目標 | 預測 $\epsilon$ 或 $\tau^0$ 對控制表現影響不大（9） |
| 基準調參 | IQL：temperature ∈ [3, 10]、expectile ∈ [0.65, 0.95]；BCQ：discount ∈ [0.9, 0.999]、tau ∈ [0.001, 0.01]；CQL 用預設值（App. A） |

常見錯誤（頁面 S10 逐條列出）：
1. 混淆兩種時間：上標 $i$ 是擴散步，下標 $t$ 是規劃時間步（Sec. 2.2）。App. B 的 $h_{\text{contact}}$ 中 $\tau^i$ 的 $i$ 指規劃時間步，與正文慣例不同，照錄並註明。
2. 只在最後一步補全目前狀態：論文在**每個**擴散步後替換（Alg. 1 第 10 行、Sec. 3.3）。〔本頁推論〕只在最後替換時，計畫仍然從 $s$ 開始，但其餘欄是在沒有限制的情況下去噪的，起點與其餘計畫不連貫。本頁玩具（C2 預設設定，程序平均）中，只在最後替換時第 0 欄到第 1 欄的跳躍是 1.4422，每步都替換時是 0.1676。論文沒有描述這種失敗情形。
3. 把 $\alpha$ 當成跨任務固定的常數：$\alpha$ 依任務調整（0.1 與 0.0001），縮短視界時也要降低（App. C 5、7），而且數值和 $\mathcal J_\phi$ 的尺度綁在一起。
4. 忽略 $\mathcal J_\phi$ 的輸入：論文以帶噪樣本 $\tau^i$ 為 $\mathcal J_\phi$ 的輸入（Sec. 3.2）。本頁 C3 顯示，若改用乾淨 $\mathcal J$ 在 $\mu$ 處求梯度，玩具中會出現越過目標與發散（標為玩具現象）。
5. 期待動作與狀態分開預測：Eq. 2 把動作當作額外維度一起去噪；執行的是計畫第一欄的動作 $\tau^0_{a_0}$。
6. 每步從頭規劃太慢：用暖啟動（Sec. 5.4）。

#### 步驟 11：限制（Sec. 5.4、Sec. 7）

計畫迭代產生，速度慢（Sec. 5.4）；在單任務 D4RL 上沒有超越專為單任務設計的離線方法（Sec. 5.3）。Sec. 7 結論：規劃與取樣幾乎相同，只差引導取樣的輔助擾動函數；作者認為結果指向一類新的、以擴散為基礎的深度 model-based RL 規劃程序。

---

## 2. 學習目標（驗收測試）

每章結尾以 `<details>`（摘要「檢查理解」）呈現該章檢查題；參考答案必須**在該章內文段落**寫出，不能只出現在元件、表格、圖或答案框。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明本文要解決的問題 | 「學動態模型＋經典軌跡最佳化器」為什麼常失敗？ | 最佳化器會利用學到模型的誤差，計畫更像對抗樣本而不是最佳軌跡（Sec. 1） | S1 |
| LO2 | 區分兩種時間記號 | $\tau^i_{s_t}$ 的 $i$ 與 $t$ 各指什麼？$\tau$ 不帶上標時指什麼？ | $i$ 是擴散步、$t$ 是規劃時間步；$\tau=\tau^0$，即無噪軌跡（Sec. 2.2） | S2 |
| LO3 | 描述 Diffuser 的軌跡表示與架構 | Diffuser 的輸入輸出是什麼形狀？為什麼用一維時間卷積？ | 二維陣列，每欄一個規劃時間步，列為狀態與動作維度（Eq. 2）；要沿時間等變、沿特徵不等變，且全卷積使視界由輸入大小決定（Sec. 3.1） | S3 |
| LO4 | 說明為何不自迴歸、為何時間局部 | 為什麼不能依時間順序逐步預測？單一去噪步只看局部，如何得到全局連貫？ | 決策可以是反因果的，例如 $p(s_1\mid s_0,s_T)$ 依賴未來狀態；每步只強化局部一致性，組合許多去噪步後局部一致性推動全局連貫（Sec. 3.1、Fig. 2） | S3 |
| LO5 | 寫出訓練目標與擴散步數的作用 | Diffuser 的訓練損失是什麼？運動控制用幾個擴散步？步數太少時取樣會怎樣？ | $\mathbb E\|\epsilon-\epsilon_\theta(\tau^i,i)\|^2$，$i\sim\mathcal U\{1..N\}$，cosine 排程；$N=20$（積木 100）；預測 $\epsilon$ 或 $\tau^0$ 差別不大（Sec. 3.1、App. C）。步數太少時，取樣分布與資料分布的差距變大（本頁玩具：樣本過度集中） | S3b |
| LO6 | 解釋擾動分布 | Eq. 1 是什麼？為什麼一個模型能用於多個任務？ | $\tilde p_\theta(\tau)\propto p_\theta(\tau)h(\tau)$；動態資訊在 $p_\theta$，任務資訊在 $h$，換任務只換 $h$（Sec. 3） | S4 |
| LO7 | 說明補全式規劃 | 目標條件規劃如何實作？報酬最大化為什麼也要補全？ | $h$ 對觀測值為 Dirac delta；每個擴散步後把被限制的位置換成 $c_t$；所有計畫都必須從目前狀態開始（Alg. 1 第 10 行）（Sec. 3.3） | S5 |
| LO8 | 說出替換法的近似性質 | 替換法為什麼不是精確的條件取樣？在本頁玩具中看到什麼？ | 被限制的值只能經由之後的去噪步傳到其他時間步；玩具中替換法的計畫平均在端點附近有跳躍、內部被拉向先驗平均，擴散步數增加時差距縮小〔本頁推導〕 | S5 |
| LO9 | 推導引導取樣 | 報酬如何進入取樣？Eq. 3 的平均是什麼？ | $p(\mathcal O_t=1)=\exp(r)$，$h=p(\mathcal O_{1:T}\mid\tau)$；反向轉移近似為 $\mathcal N(\mu+\Sigma g,\Sigma)$，$g=\nabla\mathcal J(\mu)$；Alg. 1 再乘引導尺度 $\alpha$，$\mathcal J_\phi$ 以帶噪軌跡訓練（Sec. 3.2） | S6 |
| LO10 | 說明 α 的作用與調整 | 增大 $\alpha$ 會怎樣？論文用什麼值？ | 計畫被更強地拉向高報酬區。論文 $\alpha=0.1$，hopper-medium-expert 用 0.0001，縮短視界時也要降低（App. C）。另外，Eq. 3 是近似，引導後的平均不一定等於 $p(\tau)\exp(\alpha\mathcal J)$ 的平均；本頁玩具中，用乾淨 J 引導時 $\alpha$ 過大會越過目標甚至發散，改用雜訊層級似然就不會〔本頁推導〕 | S6 |
| LO11 | 說明執行方式與暖啟動 | Diffuser 如何執行計畫？如何加速？ | 運動控制依 Alg. 1：只執行計畫的第一個動作，然後重新規劃；Maze2D 把計畫當作開迴路計畫執行（Sec. 5.1）。暖啟動：對上一個計畫跑少數幾步前向擴散，再去噪相同步數（可用 $q(\tau^k\mid\tau^0)$ 一步加噪）；只用十分之一的步數時，表現只小幅下降（Sec. 5.4、Fig. 7） | S7 |
| LO12 | 讀出實驗結論 | Maze2D、積木堆疊、D4RL 的平均分數與主要對照？ | Maze2D 單任務 119.5、多任務 129.4，IQL 47.0、16.9，MPPI（真實動態）16.2；積木 54.4 vs CQL 8.1；D4RL 77.5，與 CQL 77.6、IQL 77.0 相近；當作 MPPI 的動態模型則不比隨機好（Table 1–3、Sec. 5.3） | S9 |

---

## 3. 頁面大綱

單欄閱讀（文字欄 680–720px），元件放在加寬圖版（最大 1080px）。章節編號以 JetBrains Mono 顯示（`01 · …`），標題用陳述句。視覺記號：**軌跡陣列條**（2 列 × $H$ 欄的小方格條，Eq. 2 的示意），在 Hero、S3、C1 重複出現。

| # | id | 標題 | 內容（依序；每個符號在使用前定義） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈Diffuser：以擴散模型取樣完成軌跡規劃〉 | 副標「Diffuser 對整條軌跡建模，規劃等於從加上擾動的模型中取樣。」三張事實卡：①「Diffuser 一次產生一整條計畫：狀態與動作排成二維陣列，從純噪聲逐步去噪。」②「同一個模型換一個擾動函數 h 就能做不同任務：固定起點與終點是補全，最大化報酬是用報酬梯度引導。」③「在 Maze2D 長時程任務中平均分數 119.5，最佳 model-free 基準 IQL 為 47.0。」靜態陣列條（2×16，固定值，無動畫）。連結：arXiv、`../index.html#paper-447`、研究筆記總覽 `traj-diffusion-lineage-hub.html` | 1 螢幕 |
| S1 | `problem` | 01 · 學到的動態模型的預測誤差會被最佳化器利用 | MDP 的狀態、動作、報酬、動態（一段白話）；Sec. 2.1 目標式；model-based 流程與失敗原因（LO1 答案寫在內文）。「對抗樣本」第一次出現時給字面解釋：輸入被調整到讓模型輸出錯誤的值；「軌跡最佳化器」解釋為「搜尋一串動作使目標最大的演算法」；射擊式、CEM 只在括號中舉例。本文是離線設定。設計原則（1.2 第 2 點）移到 S4 末尾。延伸閱讀依 6.4 hub 第 1–2 步 | 350–450 字 |
| S2 | `diffusion` | 02 · 擴散模型以逐步去噪產生資料 | Sec. 2.2 兩式（`overflow-x` 容器）；cosine 排程與 $\beta_i,\bar\alpha_i,\tilde\beta_i$ 的定義；閉式 $q(\tau^i\mid\tau^0)$（標明 S7 會用到）；$\mu$ 與訓練損失需要的量；兩種時間記號（LO2 答案）；可收合符號表（含「βt＝β̃」的讀數寫法對照）。延伸閱讀依 hub 第 4 步 | 350–450 字 |
| S3 | `trajectory` | 03 · Diffuser 把整條軌跡當成一張二維陣列去噪 | Eq. 2 與陣列條；反因果例子 $p(s_1\mid s_0,s_T)$；時間局部性：每個去噪步只修正相鄰時間步之間的不一致，許多步組合後整條軌跡連貫（Fig. 2）；三個架構要求的白話版與「全卷積 → 視界由輸入大小決定」（LO3、LO4 答案）；「等變」、GN、Mish、App. C 架構細節放 `<details>` | 400–500 字 |
| S3b | `toy` | 04 · 訓練目標，以及本頁用來示範的玩具 | 訓練損失、$i$ 的抽樣、cosine 排程與 App. C 的 $N$（LO5 答案，含「步數太少時取樣分布偏離資料分布」一句）；玩具說明（頁面只寫：用一個答案已知的高斯軌跡分布代替訓練好的網路，因此去噪結果與精確答案都能算出；DCT 推導與噪聲預測閉式放 `<details>`）；回指 S2 的 $\tilde\beta_i$，並用一句說明 $\Sigma^i$ 的選擇（推定）；C1 | 350–450 字 + C1 |
| S4 | `perturb` | 05 · 規劃等於從擾動分布取樣 | Eq. 1；$h$ 的三種內容；動態與任務分離（LO6 答案）；作者的設計原則（1.2 第 2 點，由 S1 移來）；預告：S5 的 $h$ 是限制，S6 的 $h$ 是報酬。頁面說明為了先講較簡單的補全，S5、S6 的順序與論文 Sec. 3.2、3.3 相反 | 200–300 字 |
| S5 | `inpaint` | 06 · 固定起點與終點的規劃是補全問題 | Sec. 3.3 的 $h=\delta$；替換的實作；內嵌 Alg. 1 第 10 行的引文「τ^{i−1}_{s_0} ← s：限制計畫的第一個狀態」（Alg. 1 全文到 S6 才出現）（LO7 答案）；精確條件分布在玩具中可由線性方程求得（4.0）；C2；替換法的近似性質（LO8 答案寫在內文，屬定性描述；玩具數字另起一句並標「本頁玩具」） | 400–500 字 + C2 |
| S6 | `guide` | 07 · 最大化報酬是用報酬梯度引導取樣 | $\mathcal O_t$、$p(\mathcal O_t=1)=\exp(r)$；$h=p(\mathcal O\mid\tau)$；Eq. 3（`overflow-x` 容器，< 600px 拆兩行）與 $g=\nabla\mathcal J(\mu)$（加〔本頁推導〕的三個條件）；$\mathcal J_\phi$ 以帶噪軌跡為輸入；Alg. 1 兩欄對照；$\alpha$ 的值與「α=1 對應論文的目標、p(τ)exp(αJ) 是本頁採用的讀法」（LO9、LO10 答案寫在內文）；C3；C3 之後一段說明乾淨 J 模式的越過與發散是玩具簡化造成的現象。延伸閱讀依 hub 第 3–4 步，另附 `score-geometry-inverse-problems-guidance.html` | 500–600 字 + C3 |
| S7 | `loop` | 08 · 計畫的執行方式與暖啟動 | 運動控制的滾動視界迴圈（Alg. 1）與 Maze2D 的開迴路執行（Sec. 5.1），並註明 Sec. 5.4 原文的不一致；Sec. 5.4 暖啟動的兩步做法，以 S2 的 $q(\tau^k\mid\tau^0)$ 寫出前向 $k$ 步；Fig. 7 的結論（LO11 答案）；一張靜態 SVG：時間軸上「上一個計畫 →（前向 k 步）→ 部分加噪 →（去噪 k 步）→ 新計畫」，冷啟動對照「純噪聲 →（去噪 N 步）」，兩列以實線／虛線區分 | 300–400 字 + 靜態 SVG |
| S8 | `properties` | 09 · 擴散規劃器有四個性質 | 步驟 7 的表格；每個性質一段兩三句；可變長度回指 C1 的 $H$ 選項；任務組合性回指 S4 | 350–450 字 |
| S9 | `results` | 10 · 實驗：長時程、測試時彈性與離線 RL | Table 1 全表；Table 3 全表；Table 2 摘錄（平均列＋Medium-Expert 三列＋一句較弱的列與粗體規則，全表 `<details>`）；MPPI 變體；LO12 答案以一段文字寫出所有平均數字。延伸閱讀：本篇 `offline-rl-benchmarks-and-baselines.html`、`sequence-modeling-offline-rl.html` 優先；其次站內 `iql-algorithm-and-d4rl-evaluation.html`、`dt-transformers-in-rl-and-trajectory-transformer.html`、`batch-constrained-policy-constraint-methods.html` | 400–500 字 + 3 表 |
| S10 | `impl` | 11 · 實作設定與常見錯誤 | 步驟 10 的表；6 條常見錯誤 | 300–400 字 + 表 |
| S11 | `history` | 12 · 研究脈絡 | 時間線（6.4）＋一段話：model-based RL 與 MPPI → 控制即推論 → 擴散模型與引導 → 離線 RL 基準與序列模型式 RL（DT、TT）→ 本文 | 時間線 + 100–150 字 |
| S12 | `next` | 13 · 限制與延伸閱讀 | 步驟 11；筆記清單（6.4）；arXiv、程式碼網站 diffusion-planning.github.io | 200 字 + 清單 |

---

## 4. 互動元件規格

### 4.0 共用數學：高斯軌跡玩具（C1–C3 共用，`core.py` 逐式相同）

**為什麼用玩具**（頁面 S3 寫出）：瀏覽器中無法訓練 Diffuser。本頁用一個高斯軌跡分布當作「資料分布」，它的最佳噪聲預測 $\mathbb E[\epsilon\mid\tau^i]$ 有閉式解，所以可以把它當成一個訓練完美的 $\epsilon_\theta$，並且能算出精確的條件分布來比較。狀態是平面上的點 $s_t=(x_t,y_t)$；**玩具只規劃狀態，不含動作列**（點質量的動作可取 $a_t=s_{t+1}-s_t$；頁面寫明這是簡化）。

- **先驗**（推定，為可視化而選）：兩個座標各自獨立，$x_{0:H-1}\sim\mathcal N(0,Q^{-1})$，$Q=aI+bL^2$，$a=0.01$、$b=400$。$L$ 是 Neumann 邊界的二階差分矩陣（對角 −2、兩端對角 −1、上下次對角 1）。$Q$ 是五對角矩陣，但玩具使用的理想去噪器（下式）在時間上是稠密的，高噪聲時看到整條軌跡，**不**示範 Sec. 3.1 的「每個去噪步只看局部」。頁面在 S3b 寫明這一點，不把 $Q$ 的帶狀結構和時間局部性連在一起。
- **〔本頁推導〕對角化**：$L$ 的特徵向量是 DCT-II 基底 $\phi_k(t)=c_k\cos\!\big(\pi k(t+\tfrac12)/H\big)$，$c_0=\sqrt{1/H}$、$c_k=\sqrt{2/H}$；特徵值 $-\nu_k$，$\nu_k=4\sin^2(\pi k/2H)$。所以 $Q$ 的模式變異數為 $\lambda_k=1/(a+b\nu_k^2)$。`core.py` 已驗證 $\|\Phi\,\mathrm{diag}(1/\lambda)\Phi^\top-Q\|_\infty<10^{-10}$。實作以 $H\times H$ 矩陣 `PHI[t*H+k]` 做轉換（$O(H^2)$，不用 FFT）。
- **排程**（Nichol & Dhariwal cosine，$s=0.008$）：$f(i)=\cos^2\!\big(\frac{i/N+s}{1+s}\cdot\frac\pi2\big)$，$\beta_i=\min(1-f(i)/f(i-1),\,0.999)$，$\bar\alpha_0=1$，$\bar\alpha_i=\prod_{j\le i}(1-\beta_j)$，$\tilde\beta_i=\beta_i(1-\bar\alpha_{i-1})/(1-\bar\alpha_i)$（$\tilde\beta_1=0$）。反向共變異數取 $\Sigma^i=\tilde\beta_iI$（推定，R2）。
- **〔本頁推導〕最佳噪聲預測**：在 DCT 係數上 $\hat\epsilon_k=\sqrt{1-\bar\alpha_i}\,\hat\tau_k/(\bar\alpha_i\lambda_k+1-\bar\alpha_i)$（高斯條件期望）。
- **一個反向步**（`reverseStep(X,i,guide)`；`X` 為 2×H）：
  1. $\hat X=X\Phi$；$\hat E=$ 上式；$E=\hat E\Phi^\top$
  2. $\mu=(X-\frac{\beta_i}{\sqrt{1-\bar\alpha_i}}E)/\sqrt{1-\beta_i}$
  3. 若有引導：$\mu\leftarrow\mu+\tilde\beta_i\,g(\mu,i)$，$g$ 已含 $\alpha$（C3 的兩種模式，`core.make_guide`）；雜訊層級模式在 DCT 係數上算出 $\hat g$ 後以 $g=\hat g\Phi^\top$ 轉回時間域
  4. 若取樣且 $i>1$：$\mu\leftarrow\mu+\sqrt{\tilde\beta_i}\,z$，$z$ 依序抽 2×H 個標準常態（$i=1$ 時不抽）
  5. 補全：把每個限制 $(t,c)$ 寫入 $X[:,t]=c$
- **初始化**：$\tau^N$ 依序抽 2×H 個標準常態，再做一次補全。
- **PRNG**（固定，頁面與 `core.py` 相同）：`mulberry32(seed)`，標準寫法 `a=(a+0x6D2B79F5)|0; t=Math.imul(a^(a>>>15),1|a); t=(t+Math.imul(t^(t>>>7),61|t))^t; return ((t^(t>>>14))>>>0)/4294967296;`（uint32 除以 2³² 映射到 [0,1)）；`gauss()` 為 Box–Muller，取兩個均勻數 $u_1,u_2$，回傳 $\sqrt{-2\ln(1-u_1)}\cos(2\pi u_2)$（不快取 sin 分支）。抽樣順序：先座標 $x$ 的 $t=0..H-1$，再座標 $y$。C1 用 seed $s$；C2、C3 的第 $j$ 個樣本（$j=0..5$）用 seed $1000s+j$。C1 的鏈是**取樣**（有噪聲）；C2、C3 的程序平均用 $\tau^N=0$、$z=0$，樣本線則照常取樣。Hero 的靜態陣列條取 C1 演算法 seed 7、$H=16$、$N=20$ 的 $\tau^0$。預設 $s=1$，「重新抽樣」每按一次 $s$ 加 1。
- **〔本頁推導〕程序平均**：鏈的每一步都是 $X$ 的仿射函數加上獨立的零平均噪聲，所以 $\mathbb E[\tau^0]$ 等於「$\tau^N=0$、所有 $z=0$」跑一次的結果。`mc.py` 以 40000 個樣本比較：最大絕對差 C2 0.0076、C3 0.0070；把每個座標的差除以該座標的標準誤後，最大值為 1.28 與 1.96，符合約 128 個座標取最大值的預期範圍（約 3）。頁面把它畫成「程序平均」粗線。
- **精確參考**：精確條件平均是 $\min_x\ \tfrac12x^\top Px-\ell^\top x$ 在被固定位置取 $c$ 的解，每個座標解一次 $P_{FF}x_F=\ell_F-P_{FC}x_C$（$F$ 為自由位置，$C$ 為固定位置），以 Cholesky 分解求解（$H\le128$）。$a>0$ 保證 $Q$ 正定，所以 $P_{FF}$ 也正定；條件數最大約 $(a+b\cdot16)/a\approx6.4\times10^5$，雙精度可以處理。實作仍然檢查：任一主元 $\le10^{-12}$ 或結果非有限時，`exactMean` 回傳 `null`，頁面隱藏精確參考，並在讀數中顯示「精確參考無法計算」。C2：$P=Q$、$\ell=0$；C3：$P=Q+\alpha I$、$\ell=\alpha c\mathbf 1$（即 $p(\tau)\exp(\alpha\mathcal J(\tau))$，$\mathcal J=-\tfrac12\sum_t\|s_t-c\|^2$）。
- **數值保護**：分母 $\bar\alpha_i\lambda_k+1-\bar\alpha_i\ge\min(\lambda_k,1)>0$；$1-\bar\alpha_i\ge1-\bar\alpha_1\ge6.3\times10^{-4}$（$N\le100$）；$\sqrt{1-\beta_i}\ge\sqrt{0.001}$。每步檢查：任一值非有限或 $|x|>10^6$ 時停止並標示「發散」。繪圖以 `clipPath` 裁到座標框，座標框外的點不畫。

### C1 · 從噪聲到計畫：反向去噪鏈（S3b）

- **教的概念**：Eq. 2 的陣列一次整體去噪（非自迴歸）；擴散步數 $N$ 與欄數 $H$ 在取樣時選定；有限 $N$ 的取樣分布與資料分布有差距（LO5）。
- **輸入**：分段按鈕 $N\in\{5,10,20,50,100\}$（預設 20）；分段按鈕 $H\in\{16,32,64,128\}$（預設 64）；滑桿「擴散步 $i$」$\in[0,N]$ 整數（預設 0，即最終計畫；改 $N$ 時重設為 0）；滑桿「規劃時間步 $t$」$\in[0,H-1]$（預設 0）；按鈕「播放／暫停」（同一顆，切換文字與 `aria-pressed`；由目前的 $i$ 往 0，每 120ms 一步；拖動 $i$ 滑桿會中止播放；reduced-motion 時直接跳到 0）、「重新抽樣」。
- **計算**：改 $N$、$H$、seed 時重算整條鏈 $\tau^N,\dots,\tau^0$ 並快取（key `${N}|${H}|${seed}`）。重算以 `setTimeout` 切片，每片 10 個擴散步，完成前圖版顯示「計算中」，控制項維持可用；新的重算請求會取消舊的切片（token 計數）。播放中不觸發重算。拖動 $i$、$t$ 只重畫。全域讀數「取樣分布總變異數 ÷ 資料總變異數」$=\sum_kv_{0,k}/\sum_k\lambda_k$，其中 $v_{N,k}=1$，$v_{i-1,k}=A_{i,k}^2v_{i,k}+\tilde\beta_i$，$A_{i,k}=\big(1-\beta_i/(\bar\alpha_i\lambda_k+1-\bar\alpha_i)\big)/\sqrt{1-\beta_i}$〔本頁推導：無條件鏈在 DCT 模式上是獨立的一維線性遞迴〕。
- **顯示**：
  - 左：平面圖，座標 $[-4,4]^2$。$\tau^i$ 的狀態以折線＋小點畫出；$t=0$ 以圓形、$t=H-1$ 以方形標記；目前 $t$ 以放大的空心圓標記並與陣列條同步。$i>0$ 時折線用 `--ink-2`，$i=0$ 時用 `--accent`。
  - 下：軌跡陣列條（2 列 $x,y$ × $H$ 欄，每格以發散色階表示值，色階附圖例刻度 −3、0、3；色階：值 $v$ 截到 $[-3,3]$，$u=v/3$；$u<0$ 時 `color-mix(in oklab, var(--c-ref) ${-u*100}%, var(--surface-2))`，$u\ge0$ 時以 `--accent` 同法內插；每格以 0.5px `--line` 描邊，使接近 0、顏色接近底色的格子仍可分辨）；目前欄以 2px `--ink` 外框標示。標籤：「每一欄是一個規劃時間步（Eq. 2）」。
  - 讀數（JetBrains Mono）：$i$、$\tau^i_{s_t}$、變異數比；可展開的「排程數值」列出 ᾱᵢ、βᵢ、βtᵢ（＝β̃ᵢ，$i\ge1$），預設收合。
- **讀者應觀察到**：(1) $i=N$ 時是無結構的噪聲，$i$ 減小時整條計畫同時變平滑，不是從左到右逐欄產生；(2) 換 $H$ 後同一套公式產生不同長度的計畫。這示範的是「公式對任意 $H$ 都有定義」，不是同一組網路權重；論文中對應的機制是全卷積架構（Sec. 3.1、Fig. 3c），頁面寫明兩者的差別；(3) 本頁玩具中，$N=5$ 時變異數比約 0.15，$N=100$ 時約 0.89：步數太少時樣本過度集中（這是 βt 選擇與有限步數造成的玩具結果）。

### C2 · 補全：固定起點與終點（S5）

- **教的概念**：Sec. 3.3 的替換式補全；限制在每個擴散步後寫回；與精確條件分布的比較〔本頁推導〕。
- **輸入**：分段按鈕 $H\in\{32,64,128\}$（預設 64）；$N\in\{5,20,100\}$（預設 20）；起點 $A=(-1.2,-0.8)$ 固定；終點 $B$ 兩個滑桿 $B_x,B_y\in[-2,2]$ 步長 0.1（預設 $(1.2,0.8)$）；勾選框「顯示 6 個樣本」（預設開）、「顯示精確條件平均」（預設開）；「重新抽樣」。
- **計算**：程序平均（4.0，零噪聲）；6 個樣本（seed $1000s+j$）；精確條件平均（固定 $t=0$ 與 $t=H-1$；`exactMean` 回傳 `null` 時見 4.0）。
- **顯示**：平面圖 $[-4,4]^2$。樣本：`--ink-2` 細線（0.75 不透明度）；程序平均：`--accent` 粗實線，標「替換法（平均）」；精確：`--c-ref` 粗虛線，標「精確條件平均」；起點圓形、終點方形。右側小圖：$x$ 座標對 $t$（兩條平均線＋樣本），方便看端點跳躍。讀數：均方根差 $\mathrm{rms}=\sqrt{\frac1{2H}\sum(\text{程序}-\text{精確})^2}$；端點跳躍 $\|\bar\tau_{s_1}-\bar\tau_{s_0}\|$（程序與精確各一），並以「先驗單步長度的幾倍」呈現；先驗的典型單步長度 $\sqrt{\frac1{H-1}\sum_t\mathrm{Var}(x_{t+1}-x_t)}$（每座標）；限制誤差（恆為 0，顯示以證明寫回成功）。
- **讀者應觀察到**（皆為本頁玩具的結果）：(1) 每個樣本都精確通過 $A$ 與 $B$；(2) 程序平均在端點附近有跳躍（$H=64$、$N=20$：0.168，是先驗單步 0.113 的 1.5 倍；精確 0.007），內部比精確平均更靠近原點；(3) 在 $N\le100$ 的範圍內，$N$ 從 5 到 100，rms 從 0.439 降到 0.173，跳躍從 0.599 降到 0.073（0.65 倍單步），沒有看到歸零；(4) $H=128$ 時程序平均的內部幾乎停在原點（$t=H/4$ 為 −0.033，精確 −0.233）：限制資訊沿時間傳得不夠遠。頁面附一句：「這是本頁高斯玩具的結果；論文的 Diffuser 以真實資料訓練，Maze2D 分數見 S9。」

### C3 · 引導取樣：用報酬梯度推動計畫（S6）

- **教的概念**：Alg. 1 第 8 行；$\alpha$ 的作用；Eq. 3 與條件 (3) $p(\mathcal O\mid\tau^i)\approx p(\mathcal O\mid\mu)$ 是近似；梯度在哪個噪聲層級上計算會影響結果〔本頁推導〕。
- **簡化**（頁面寫出）：報酬 $r(s_t)=-\tfrac12\|s_t-c\|^2$（$\le0$，符合 $\exp(r)\le1$），$\mathcal J=\sum_{t=0}^{H-1}r(s_t)$；起點 $s=(-1.2,-0.8)$ 每步補全（Alg. 1 第 10 行）。不訓練 $\mathcal J_\phi$，改用兩種可以閉式計算的引導（分段按鈕「引導梯度」）：
  - **雜訊層級似然（預設）**：$g=\nabla_\tau\log\mathbb E[\exp(\alpha\mathcal J(\tau^0))\mid\tau^{i-1}=\tau]\big|_{\tau=\mu}$。在玩具中這個式子在 DCT 上是對角的：令 $\bar a=\bar\alpha_{i-1}$、$m_k=\sqrt{\bar a}\lambda_k/(\bar a\lambda_k+1-\bar a)$（後驗平均係數）、$S_k=\lambda_k(1-\bar a)/(\bar a\lambda_k+1-\bar a)$（後驗變異數），$\hat c$ 為常數向量 $c\mathbf 1$ 的 DCT 係數（只有 $\hat c_0=c\sqrt H$），則 $\hat g_k=m_k\cdot\big(-\alpha(m_k\hat\mu_k-\hat c_k)/(1+\alpha S_k)\big)$〔本頁推導：高斯後驗下二次報酬的指數期望〕。這最接近論文以帶噪樣本為 $\mathcal J_\phi$ 輸入的意圖，但它不是論文的 $\mathcal J_\phi$。$i=1$ 時 $\bar a=1$，與乾淨 J 相同。
  - **乾淨 J(μ)**：$g=-\alpha(\mu-c)$，也就是 Alg. 1 第 8 行直接套用真實報酬，跳過條件 (3) 的處理。
  - 兩種模式都以 $\mu+\tilde\beta_i g$ 更新。
- **輸入**：分段按鈕「引導梯度」（雜訊層級似然／乾淨 J(μ)，預設前者）；滑桿 $\log_{10}\alpha\in[-3,2]$，內部以整數 `la`∈[−30, 20] 儲存，$\alpha=10^{la/10}$（預設 −10，即 $\alpha=0.1$；標示「預設 0.1。α 的數值與 J 的尺度綁在一起，不能和論文的 0.1 直接比較」）；`aria-valuetext` 格式 `α = ${alpha.toPrecision(3)}`（例：「α = 15.8」）；目標 $c$ 兩個滑桿 $\in[-2,2]$ 步長 0.1（預設 $(1,1)$）；$H\in\{32,64\}$（預設 32）；$N\in\{20,100\}$（預設 20）；勾選框「顯示 6 個樣本」「顯示精確傾斜分布平均」（預設皆開）；「重新抽樣」。
- **計算**：$\mathcal J$ 讀數一律在平均路徑上計算：$\mathcal J=-\tfrac12\sum_t\|m_t-c\|^2$（$m$ 為程序平均或精確平均，對 $x,y$ 兩座標加總）。程序平均、6 個樣本、精確傾斜平均（4.0，$p(\tau)\exp(\alpha\mathcal J)$ 為本頁採用的讀法）；乾淨 J 模式另算 $\max_i\alpha\tilde\beta_i$ 與 $\#\{i:\alpha\tilde\beta_i>2\}$；發散偵測（記錄第一個輸出超出 $10^6$ 的步 $i$，即由 $\tau^i$ 算 $\tau^{i-1}$ 的那一步）。每次重算開始時清除 `divergedAt`；未發散的結果存入 `lastGood = {alpha, mode, H, N, c, ends, rms, J}`。
- **顯示**：平面圖 $[-4,4]^2$，目標 $c$ 以菱形標記；線型同 C2（「引導（平均）」實線、「精確 p(τ)exp(αJ)」虛線）。下方小長條：$\mathcal J$（程序平均）與 $\mathcal J$（精確平均）。讀數：終點 $\bar\tau_{s_{H-1}}$（兩者）、rms、兩個 $\mathcal J$；乾淨 J 模式加上 max αβtᵢ 與「αβtᵢ > 2 的步數」。發散時圖上不畫曲線，顯示「發散：第 i 步的數值超過 10⁶。αβtᵢ > 2 時，引導項在那一步放大與目標的距離；是否發散取決於所有步驟累積的增益。」下方以灰色顯示 `lastGood` 的讀數，並標明它的 α。之後把滑桿拉回不發散的值時立即恢復正常顯示。
- **讀者應觀察到**（皆為本頁玩具的結果）：(1) $\alpha=0.001$ 時幾乎沒有引導；(2) 雜訊層級似然模式下，$\alpha$ 增大時終點穩定地接近 $c$（$H=32,N=20$：$\alpha=0.1$ 0.886、$\alpha=1$ 0.988、$\alpha=100$ 0.9997；精確 1.079、1.007、0.9999），即使 $\alpha=100$ 也不發散，中間最大值維持 1.2；(3) 切換到乾淨 J 模式，$\alpha=1$ 時終點越過 $c$ 到 1.826，$\alpha=10$ 時中間數值可到 9.6×10³（雖然有 7 步 αβtᵢ > 2，終點仍為 1.299），$\alpha\ge10^{1.2}\approx15.8$ 時發散（第 13 步，10 步 αβtᵢ > 2）。頁面接著寫：越過與發散來自「在 μ 處用乾淨報酬」的簡化，是玩具現象，不代表 Diffuser 的性質；論文以帶噪樣本為 $\mathcal J_\phi$ 的輸入，並對各任務調整 $\alpha$（0.1、0.0001），縮短視界時也降低 $\alpha$（App. C）；(4) 兩種模式下，引導後的平均和精確平均之間都有差距（rms 約 0.1–0.5），反映 Eq. 3 是近似。

### 4.4 共同的無障礙、排版與互動規則

- **KaTeX**：靜態正文、`<label>`、表頭用 KaTeX，載入後對每個章節容器呼叫一次 `renderMathInElement`。所有 display 數學（Sec. 2.2 的積分式、Eq. 2 的矩陣、Eq. 3、$h=\delta$ 的分段式）放在 `.katex-display{overflow-x:auto;overflow-y:hidden}` 中；Eq. 3 在 < 600px 時拆成兩行；學習目標表與符號表包在 `overflow-x:auto` 容器中。
- **動態文字**：元件內所有動態文字（讀數、圖上標籤、`aria-label`、`aria-valuetext`、live region）用 Unicode，**不用組合字元**（U+0303 在 JetBrains Mono 的 fallback 中顯示不一致）：β̃ᵢ 寫成「βtᵢ」，符號表註明「βt ＝ β̃」；上下標只用預組字元（ⁱ、ᵢ、₀ 等）；等寬字型堆疊 `"JetBrains Mono","IBM Plex Mono",ui-monospace,monospace`。建置時以 390px 截圖確認讀數沒有缺字方框。
- **非顏色編碼**：起點圓形、終點方形、目標菱形；程序平均實線、精確平均虛線、樣本細線；每條線有直接文字標籤。陣列條的色階另附數值圖例，格子有描邊，目前欄的數值顯示在讀數中。
- **對比**（light 與 dark 都要驗證，在 `--surface` 上）：`--accent`、`--c-ref` 的線 ≥ 3:1；樣本線改用 `--ink-2`（dark #B5B0A5 在 #1F1E1B 上約 8:1）；陣列條描邊 `--line` 只是輔助，主要的區分靠色階兩端與描邊。不足 3:1 時調整明度並回寫 6.2。
- **觸控**：不使用 hover；所有選取以滑桿完成（C1 的 $t$ 滑桿取代 hover 查值）；滑桿與按鈕觸控區 ≥ 44px。
- **更新與降級**：`input` 事件以 `requestAnimationFrame` 合併為每影格一次重算／重畫。C2、C3 的降級規則以查表決定，不量測時間：$H\cdot N>4000$（即 $H=64,N=100$、$H=128,N=100$）時，拖動中只算程序平均與精確平均（2 條），放開（`change`）後再以 `setTimeout` 分 6 片算樣本（每片一條鏈）；其他組合每影格完整計算 7 條鏈（最多 $7\times20\times4\times128^2\approx9.2\times10^6$ 次乘加）。
- **live region**：`aria-valuetext` 隨滑桿更新（這是滑桿本身的屬性）；讀數區的 `aria-live="polite"` 與 SVG `aria-label` 只在 `change` 事件或停止輸入 500ms 後更新，兩者共用同一個 debounce，避免重複宣告。C1 播放中不更新，暫停或結束時宣告一次。
- 分段按鈕 `role="radiogroup"`＋`role="radio"`／`aria-checked`，roving tabindex，方向鍵切換。
- `prefers-reduced-motion: reduce`：取消 transition；C1「播放」直接跳到 $i=0$；Hero 陣列條本來就是靜態的。
- 窄螢幕（< 600px）：圖在上、控制在中、讀數在下（兩欄 `<dl>`）；SVG 寬度等於容器寬度（`ResizeObserver` 重畫），圖上文字 12–13px；C2 右側小圖移到平面圖下方；陣列條在 $H=128$ 時每格最窄 2px，不足時改為橫向捲動容器。390px 時的檢查清單：無水平捲動（selfTest 檢查 `scrollWidth <= innerWidth`）、讀數不換行截斷、分段按鈕可換行。

---

## 5. 驗收數值表

全部取自 `verify/check_out.txt`、`mc_out.txt`、`c3_alpha_out.txt`（同一演算法、雙精度）。**容許誤差**：表中所有值 ±1e-4（表值四捨五入到 4 位小數；JS 與 numpy 只差浮點加總順序）；$\mathcal J$ ±1e-3；發散步數必須完全相同。

**先驗與排程**

| 項目 | 值 |
| --- | --- |
| $\lambda_0,\lambda_1,\lambda_2$（$H=64$） | 100.0000、81.1591、21.2319 |
| 先驗單步長度（每座標）$H=16,32,64,128$ | 0.0837、0.1092、0.1128、0.1122 |
| $N=20$：$\bar\alpha_1$、$\bar\alpha_{10}$、$\bar\alpha_{20}$ | 0.992007、0.493844、6.060e−06 |
| $N=20$：$\beta_1$、$\beta_{20}$、$\tilde\beta_2$、$\tilde\beta_{20}$ | 0.007993、0.999000、0.005750、0.992952 |
| $\max_i\tilde\beta_i$：$N=5,10,20,50,100$ | 0.905134、0.974956、0.992952、0.998031、0.998758 |

**C1 變異數比** $\sum v_0/\sum\lambda$

| H | N=5 | 10 | 20 | 50 | 100 |
| --- | --- | --- | --- | --- | --- |
| 16 | 0.1157 | 0.3153 | 0.5621 | 0.7835 | 0.8836 |
| 32 | 0.1482 | 0.3563 | 0.5917 | 0.8004 | 0.8930 |
| 64 | 0.1488 | 0.3608 | 0.5958 | 0.8029 | 0.8946 |
| 128 | 0.1546 | 0.3687 | 0.6014 | 0.8061 | 0.8963 |

Monte Carlo（$H=64,N=20$，40000 樣本）0.5935，與閉式 0.5958 一致。

**C1 PRNG 與鏈**（seed 1，$H=64$，$N=20$）：前三個均勻數 0.627073941、0.002735721、0.527447040；$\tau^{20}$ 的 $x_{0..2}$ = 1.404339、1.215755、−0.510377，$y_0=-0.791545$。

| i | rms(τⁱ) | τⁱ 第 0 欄 (x, y) | τⁱ 第 63 欄 (x, y) |
| --- | --- | --- | --- |
| 20 | 0.9467 | (1.4043, −0.7915) | (−0.2904, 0.1819) |
| 10 | 1.1180 | (1.3922, 1.8816) | (1.0807, −2.2986) |
| 5 | 1.1252 | (1.4380, 2.4804) | (−0.0994, −1.5251) |
| 1 | 1.1052 | (1.0863, 2.1561) | (−0.2034, −1.6909) |
| 0 | 1.1062 | (1.1098, 2.2962) | (−0.2061, −1.5896) |

**C2**（$A=(-1.2,-0.8)$；跳躍為程序平均／精確平均）

| H | N | B | rms | 跳躍 程序 | 跳躍 精確 | $t=H/4$ 程序 (x, y) | $t=H/4$ 精確 (x, y) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 5 | (1.2, 0.8) | 0.4393 | 0.5992 | 0.0068 | (−0.1113, −0.0742) | (−0.7141, −0.4761) |
| 64 | 20 | (1.2, 0.8) | 0.2455 | 0.1676 | 0.0068 | (−0.3456, −0.2304) | (−0.7141, −0.4761) |
| 64 | 100 | (1.2, 0.8) | 0.1731 | 0.0732 | 0.0068 | (−0.4449, −0.2966) | (−0.7141, −0.4761) |
| 64 | 20 | (1.5, −1.0) | 0.3228 | 0.1737 | 0.0068 | (−0.3518, −0.1931) | (−0.7064, −0.5225) |
| 32 | 20 | (1.2, 0.8) | 0.1949 | 0.2167 | 0.0170 | (−0.4818, −0.3212) | (−0.7768, −0.5178) |
| 128 | 20 | (1.2, 0.8) | 0.2162 | 0.1790 | 0.0069 | (−0.0329, −0.0219) | (−0.2333, −0.1555) |
| 128 | 100 | (1.2, 0.8) | 0.1655 | 0.0836 | 0.0069 | (−0.0671, −0.0447) | (−0.2333, −0.1555) |

$x$ 座標剖面（$H=64,N=20,B=(1.2,0.8)$；$t$：程序／精確）：0：−1.2000／−1.2000；1：−1.0606／−1.1943；8：−0.6552／−1.0357；32：0.0088／0.0232；62：1.0606／1.1943。樣本（seed 1000）$t=1$：(−1.3286, −0.9428)，跳躍 0.1922；（seed 1001）$t=1$：(−1.8869, 0.0940)，跳躍 1.1274。

**C3 乾淨 J(μ) 模式**（$s=(-1.2,-0.8)$，$c=(1,1)$；終點只列 $x$，$y$ 見 `check_out.txt`）

| H | N | α | 終點 x 程序 | 終點 x 精確 | rms | J 程序 | J 精確 | max αβ̃ᵢ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 20 | 0.001 | −0.0592 | −0.4020 | 0.3213 | −59.164 | −87.876 | 0.0010 |
| 32 | 20 | 0.01 | 0.0717 | 0.1610 | 0.2094 | −50.465 | −63.589 | 0.0099 |
| 32 | 20 | 0.1 | 1.0997 | 1.0789 | 0.5193 | −12.341 | −34.138 | 0.0993 |
| 32 | 20 | 1 | 1.8263 | 1.0067 | 0.8933 | −20.202 | −19.377 | 0.9930 |
| 32 | 20 | 10 | 1.2986 | 0.9992 | 0.3769 | −7.621 | −11.137 | 9.9295 |
| 64 | 20 | 0.1 | 1.1637 | 0.9035 | 0.3996 | −13.362 | −34.799 | 0.0993 |
| 32 | 100 | 0.1 | 1.5652 | 1.0789 | 0.8173 | −14.668 | −34.138 | 0.0999 |
| 32 | 100 | 10 | 1.0546 | 0.9992 | 0.2086 | −6.856 | −11.137 | 9.9876 |

發散（第一個輸出超過 $10^6$ 的步 $i$）：$H=32,N=20$：$\alpha=10^{1.2}$ 第 13 步、$10^{1.3}$ 第 14 步、$10^{1.5}$ 第 16 步、$\alpha=100$ 第 17 步；$H=64,N=20$：$10^{1.2}$ 第 13 步、$10^{1.3}$ 第 15 步；$H=32,N=100$：$10^{1.2}$ 第 91 步、$10^{1.5}$ 第 95 步。三組設定在滑桿網格上第一個發散的值都是 $10^{1.2}$；$\alpha=10^{1.1}$（$H=32,N=20$）不發散，終點 2.5664，中間最大絕對值 1.40e+05。其他（乾淨 J）：$c=(-1,1.5)$、$\alpha=0.1$、$H=32,N=20$：終點程序 (−1.1934, 1.6886)、精確 (−0.8857, 1.5379)、rms 0.5426；樣本（seed 1000）終點 (2.5324, 1.2526)，$\mathcal J=-21.780$。$N=20$ 時 αβtᵢ > 2 的步數：$\alpha=10$ 為 7，$\alpha=10^{1.2}$ 為 10。

**C3 雜訊層級似然模式（預設）**（同上設定；所有列中間最大絕對值都是 1.2000，即沒有越過起點的量級）

| H | N | α | 終點 x 程序 | 終點 x 精確 | rms | J 程序 | J 精確 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 20 | 0.001 | −0.0159 | −0.4020 | 0.3585 | −56.231 | −87.876 |
| 32 | 20 | 0.01 | 0.3248 | 0.1610 | 0.3820 | −36.001 | −63.589 |
| 32 | 20 | 0.1 | 0.8862 | 1.0789 | 0.4808 | −13.095 | −34.138 |
| 32 | 20 | 1 | 0.9878 | 1.0067 | 0.3727 | −7.946 | −19.377 |
| 32 | 20 | 10 | 0.9981 | 0.9992 | 0.2271 | −6.049 | −11.137 |
| 32 | 20 | 15.85 | 0.9987 | 0.9996 | 0.2019 | −5.787 | −10.010 |
| 32 | 20 | 100 | 0.9997 | 0.9999 | 0.1152 | −5.032 | −6.717 |
| 64 | 20 | 0.1 | 0.8557 | 0.9035 | 0.3475 | −13.798 | −34.799 |
| 32 | 100 | 0.1 | 0.9419 | 1.0789 | 0.4960 | −13.252 | −34.138 |
| 32 | 100 | 100 | 0.9999 | 0.9999 | 0.0997 | −5.256 | −6.717 |

樣本（seed 1000，$\alpha=0.1$，$H=32,N=20$）終點 (1.6999, 1.1710)，$\mathcal J=-17.225$；seed 1001 終點 (2.3413, 1.6619)，$\mathcal J=-33.540$。

**常見錯誤 2 的數字**（C2 預設，程序平均）：只在最後替換時跳躍 1.4422，每步替換時 0.1676。

**Monte Carlo**（`mc_out.txt`，40000 樣本）：C2、C3 的程序平均與樣本平均的最大絕對差 0.0076、0.0070；每座標差／標準誤的最大值 1.28、1.96。

---

## 6. 實作計畫

### 6.1 檔案與共用外殼

- 單一檔案 `tut/447/build/index.html`；CSS、JS、時間線資料全部內嵌，不 `fetch` 本機檔案。上線時複製為 `tutorials/diffuser-planning-with-diffusion.html`。
- **直接沿用**最新建置頁（例如 `tut/406/build/index.html`，其外殼來自 `tutorials/prediction-targets-k-diff.html`）：CSS tokens（含深色模式三段覆寫、`pm-theme`）、字型載入、appbar、`#pm-bar` 導覽、章節目錄（sticky 左欄／窄螢幕橫列）、`.plate` 圖版、callout、`<details>` 檢查題、「本頁推導」與「推定」pill、表格橫捲容器、KaTeX 0.16.9 設定與 `renderMath()`、`?test` selfTest 框架（`ok()` 計數、`#selftest` 輸出、`scrollWidth <= innerWidth` 檢查）、時間線卡片列元件、SVG 輔助函式（`el`、`scaleLin`、`linePath`）、`schedule(fn)`、`debounce(fn,500)`、`radiogroup(el,onChange)`、Alg. 對照兩欄格線（k-diff 頁 S8）。
- 以下只列新增或變更的部分。

### 6.2 新增 tokens、元件與連結

- 圖表色：`--accent` = 程序平均／最終計畫；新增 `--c-ref`（light #3F6E8C／dark #8DB4D3，沿用 k-diff 頁 `--c-par` 色值）= 精確參考；樣本用 `--ink-2`（`--ink-3` 在深色模式的線條對比不足）。陣列條發散色階：負值 `--c-ref`、正值 `--accent`、0 為 `--surface-2`，以 `color-mix` 內插，每格加 0.5px `--line` 描邊。建置時確認圖形對比 ≥ 3:1、文字 ≥ 4.5:1（light／dark 兩種 `--surface`），不足則調整並回寫此處。
- 新元件：軌跡陣列條（`arrayStrip(svg, X, tSel)`，Hero 用靜態版：2×16 格、以固定 seed 產生的值）；平面圖（`planeView(svg, layers)`，layers 為 `{pts, style, label}` 陣列，含起點／終點／目標標記與 `clipPath`）。
- S7 靜態 SVG（`id="s7-warm"`，viewBox 0 0 640 200，寬度 100%，< 600px 時兩列改為上下堆疊、文字 12px）：兩列時間軸，上列「冷啟動：純噪聲 τᴺ →（去噪 N 步）→ 計畫」，下列「暖啟動：上一個計畫 →（前向 k 步）→ 部分加噪 →（去噪 k 步）→ 新計畫」；箭頭標步數，冷啟動虛線、暖啟動實線，每段有文字標籤。
- 站內連結：Hero「在關聯圖中查看」→ `../index.html#paper-447`；arXiv `https://arxiv.org/abs/2205.09991`；程式碼與視覺化 `https://diffusion-planning.github.io`；筆記 → `<slug>.html`。

### 6.3 程式結構（單一 IIFE）

- 共用數學：`PRIOR={a:0.01,b:400}`；`basis(H)`（快取 `Float64Array` H×H）、`lam(H)`、`qmat(H)`（五對角，存成稠密 H×H）、`schedule(N)` → `{beta, abar, bt}`（長度 N+1）；`reverseStep(X, i, sch, H, guide, z)`；`runChain({H, N, seed, cond, guide, noise})` → `{X0, hist, divergedAt}`（`hist` 只在 C1 保留）；`exactMean(H, fixed, extraPrec, lin)`（Cholesky）；`varRatio(H, N)`；`stepSd(H)`；`rms(A,B)`；`rng(seed)`、`gauss(r)`。函式與 `core.py` 一一對應，名稱註解寫出對應的 Python 函式。
- C1：`initC1()`，`state={N,H,seed,i,t,playing,token}`；`cache` 以 `${N}|${H}|${seed}` 為 key 存整條鏈；`computeChainSliced(key, token)` 每片 10 步。
- 共用：`makeGuide(mode, alpha, c, H, N)` 對應 `core.make_guide`；`heavy(H,N)` = `H*N>4000`（4.4 降級規則）。
- C2：`initC2()`，`state={H,N,bx,by,showS,showE,seed}`。
- C3：`initC3()`，`state={mode,la,cx,cy,H,N,showS,showE,seed,divergedAt,lastGood}`，`alpha=10**(la/10)`（`la` 為整數，避免浮點累加）。
- `selfTest()` 新增斷言（容許誤差同第 5 節）：
  - 對角化：`H=32` 時 $\Phi\,\mathrm{diag}(1/\lambda)\Phi^\top$ 與 `qmat(32)` 最大差 < 1e-8。
  - 排程：$N=20$ 的 $\bar\alpha_{10}=0.493844$、$\tilde\beta_{20}=0.992952$。
  - PRNG：seed 1 前三個均勻數（第 5 節）。
  - C1：`varRatio(64,20)=0.5958`；seed 1、$H=64$、$N=20$ 的 $\tau^0$ 第 0 欄 (1.1098, 2.2962)。
  - C2：$H=64,N=20,B=(1.2,0.8)$ 的 rms 0.2455、跳躍 0.1676。
  - C3 雜訊層級似然：$H=32,N=20$ 時，$\alpha=0.1$ 的終點 x 為 0.8862，$\alpha=100$ 為 0.9997 且不發散。乾淨 J：$\alpha=0.1$ 的終點 x 為 1.0997，精確 1.0789；$\alpha=10^{1.2}$（`la=12`）在第 13 步發散，再把 `la` 設回 −10 後 `divergedAt` 為 null。
  - exactMean：傳入非正定矩陣（人為把 $P_{FF}$ 的一個對角元設為 −1）時回傳 null。
  - S7：`#s7-warm` 存在，且含「前向 k 步」「去噪 k 步」「去噪 N 步」三個文字標籤。
  - 結果輸出 `selfTest: N passed, M failed`。

### 6.4 研究筆記與時間線

- 本篇筆記（`tut/447/notes/`，7 篇）：總覽 `traj-diffusion-lineage-hub`（Diffuser 研究脈絡總覽）；`model-based-planning-abstraction-barrier`（學習式動態模型與軌跡最佳化的分工）；`control-as-inference-guided-sampling`（控制即推論：把獎勵最大化改寫為條件取樣）；`diffusion-classifier-guidance-background`（擴散模型的分類器引導與餘弦排程）；`offline-rl-benchmarks-and-baselines`（本論文使用的離線 RL 基準與比較方法）；`sequence-modeling-offline-rl`（以序列模型處理離線 RL：Decision Transformer 與 Trajectory Transformer）；`trajectory-diffusion-planning`（軌跡層級擴散規劃（Diffuser））。
- hub 引用的共用筆記（不在本篇資料夾；title 從來源取）：`rl-mdp-q-learning-basics`（MDP、Q 函數與 Bellman backup：value-based RL 的基本記號）與 `offline-rl-distribution-shift`（Offline RL 的分布偏移與 OOD 動作造成的 Q 值高估），兩者都來自 `tut/568/notes/`；`eps-prediction-ddpm-score`（ε-prediction 與 DDPM、score-based 模型）與 `ddpm-training-and-sampling-recap`（DDPM 的訓練與取樣流程），兩者都在 `scripts/notes_split/`，已上線於 `tutorials/`。
- **延伸閱讀依 hub 的 6 步閱讀順序**：
  | hub 步驟 | 筆記 | 放在哪一章 |
  | --- | --- | --- |
  | 1 RL 基礎 | `rl-mdp-q-learning-basics`、`offline-rl-distribution-shift` | S1 |
  | 2 模型＋規劃器 | `model-based-planning-abstraction-barrier` | S1 |
  | 3 控制即推論 | `control-as-inference-guided-sampling` | S6 |
  | 4 擴散基礎 | `eps-prediction-ddpm-score`、`ddpm-training-and-sampling-recap` 放 S2；`diffusion-classifier-guidance-background` 放 S6 | S2、S6 |
  | 5 基準與比較 | `offline-rl-benchmarks-and-baselines`、`sequence-modeling-offline-rl`（本篇，優先）；其次是站內的 `iql-algorithm-and-d4rl-evaluation`（No. 569）、`batch-constrained-policy-constraint-methods`（No. 570）、`dt-transformers-in-rl-and-trajectory-transformer`（No. 571） | S9 |
  | 6 本文方法 | `trajectory-diffusion-planning` | S3、S12 |
  S12 列出 hub 加上表中全部筆記，順序同上；另附 `score-geometry-inverse-problems-guidance`（S6，作為引導方法的延伸）。每項附 frontmatter 的 `title`（同上括號）。Hero 的筆記連結指向 `traj-diffusion-lineage-hub.html`。
- **時間線**：`timeline.json`（15 筆，`year` 為整數，已包含 2205.09991）原樣貼進 `<script type="application/json" id="timeline-data">`。排序依 `year`，同年再依 `arxiv` 字串排序，確保穩定。刪除「補本文卡片」的邏輯。
- **`TL_GROUP`**（15 個 ID 全部指定，不留「其他」）：
  | 類別 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | model-based RL 與軌跡最佳化 | 1509.01149（MPPI）、1708.02596（Nagabandi et al.）、1805.12114（PETS） | `model-based-planning-abstraction-barrier` |
  | 控制即推論 | 1805.00909 | `control-as-inference-guided-sampling` |
  | 擴散模型與引導 | 1503.03585、2006.11239、2102.09672、2105.05233 | `diffusion-classifier-guidance-background` |
  | 離線 RL 基準與資料 | 1707.01495（HER，用於多任務 IQL 的目標重標記，App. A.1）、2004.07219（D4RL）、2006.04779（CQL）、2110.06169（IQL） | `offline-rl-benchmarks-and-baselines` |
  | 序列模型式 RL | 2106.01345（DT）、2106.02039（TT） | `sequence-modeling-offline-rl` |
  | 本文 | 2205.09991 | `trajectory-diffusion-planning` |
  建置時以程式檢查 `timeline.json` 的每個 ID 都在 `TL_GROUP` 中；若有遺漏，歸入「其他」並在建置回報中列出（R11）。

### 6.5 文案規則（新增部分）

- 用詞：「擴散步」（上標 $i$）與「規劃時間步」（下標 $t$）全文固定，不寫成「步驟」；「補全（inpainting）」「引導取樣（guided sampling）」「擾動函數 $h$」「滾動視界（receding horizon）」「暖啟動（warm-start）」；第一次出現附英文。
- 三種陳述分開：論文結果（附出處）、「作者認為」、〔本頁推導〕；玩具相關的數字一律寫成「本頁玩具中」，不與論文數字並列在同一句。
- 玩具與論文的界線：C1–C3 圖版底部各一行「讀數由本頁的高斯軌跡玩具依 4.0 的公式即時計算，不是論文的模型」。
- 公式旁標論文編號；Eq. 3 在 < 600px 時拆成兩行。

### 6.6 建置後自我檢查

依 k-diff／406 頁流程：390／768／1024／1440 截圖＋深色、鍵盤 focus、reduced-motion（headless Chrome 無法連 CDN，截圖時以本機 KaTeX `scratchpad/katex/0.16.9` 代換）；逐項評分（字體、留白、視覺層級、色彩、動態、微互動、響應式、原創性）並修正；`?test` 全數通過；逐條核對 12 個學習目標的參考答案出現在指定章節內文。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 玩具與論文模型的差距 | 高斯先驗只有一個模式，無法呈現時間組合性（拼接）、多模態或積木堆疊；替換法在玩具中的跳躍、乾淨 J 模式的越過與發散門檻都是玩具的數字；玩具去噪器在時間上是非局部的，不示範 Sec. 3.1 的局部性；C1 換 $H$ 是換公式，不是同一組權重。 | 頁面每個元件標明「本頁玩具」，觀察項逐條加上這個限定；四個性質中只有可變長度由 C1 直接示範，其餘以 Fig. 3 的文字描述；C2 觀察 4 加上不代表論文表現的說明。 |
| R2 | $\Sigma^i$ 的選擇 | 論文只寫「Σ^i follow the cosine schedule」（Sec. 3.1），沒有寫是 $\beta_i$ 還是 $\tilde\beta_i$；Alg. 1 第 8 行也同時用 $\Sigma$ 與 $\Sigma^i$。 | 本頁取 $\Sigma=\Sigma^i=\tilde\beta_iI$，標「推定」。變異數比與 C3 的數字依此計算。 |
| R3 | 先驗參數 $a,b$ | 為可視化而選（平面 $[-4,4]$、單步約 0.11），不是論文的值。 | 標「推定」；頁面只說明它的作用（$a$：整體位置的散布；$b$：平滑程度）。 |
| R4 | 論文記號的不一致 | $\mathcal O_{1:T}$ 對比 $\sum_{t=0}^T$；補全的擴散步集合寫成 $\{0,1,\dots,N\}$；App. B 的 $h_{\text{contact}}$ 求和 $i=0..64$（65 項）卻寫「前 64 個時間步」，且 $i$ 在此指規劃時間步；Sec. 5.2 正文說比較 BCQ 與 CQL，App. A.2 的「Multi-task」段寫「evaluate BCQ and CQL in the multi-task setting」。 | 頁面照錄並以註腳指出，不修改原式。 |
| R5 | 暖啟動細節 | 論文沒有寫是否平移上一個計畫；Fig. 7 的步數範圍 2–100 與 App. C 運動控制 $N=20$ 不一致，論文未解釋。Fig. 7 沒有數據表。 | S7 只照錄文字結論與步數範圍，不做元件、不重繪曲線、不補充平移規則。原先規劃的滾動視界／暖啟動元件已刪除：玩具中結果對種子敏感且不呈現 Fig. 7 的趨勢。 |
| R6 | 視界數字 | App. C 的 Medium 視界 265 不是 2 的冪次，可能是筆誤，論文未更正；$T$ 與欄數 $T+1$ 的關係未說明。 | 照錄 265；頁面玩具以 $H$ 表示欄數並說明與 $T$ 的關係。 |
| R7 | $\mathcal J_\phi$ 的訓練目標 | 論文寫「predict the cumulative rewards of trajectory samples τ^i」，沒有寫目標是 $\tau^i$ 還是對應 $\tau^0$ 的報酬；Alg. 1 在 $\mu$ 求梯度。 | 照錄原文。C3 提供兩種閉式引導：雜訊層級似然 $\log\mathbb E[\exp(\alpha\mathcal J(\tau^0))\mid\tau^{i-1}]$（預設；在 $\alpha\le100$ 內不發散、不越過目標，`check.py`）與乾淨 J(μ)（對照組）。第 1 版曾試 $\nabla\mathcal J(\mathbb E[\tau^0\mid\tau])$，那個版本少了 $(1+\alpha S_k)^{-1}$ 的阻尼，因此不穩定；這不是正確的雜訊層級似然，已改正。兩者都不是論文的 $\mathcal J_\phi$。 |
| R8 | Maze2D 的 $N$ | App. C 只給運動控制 20、積木 100。 | 步驟 10 表格寫「未寫明」。 |
| R9 | 元件效能 | C2/C3 在 $H=128$、$N=100$ 時每次重算約 4.6e7 次乘加；C1 重算若同步執行會凍結 UI。 | 以 $H\cdot N>4000$ 查表決定降級（4.4），樣本以 setTimeout 分片計算；C1 以 10 步為一片計算並快取。 |
| R10 | 初學者門檻 | 讀者不熟 RL 與擴散。 | S1 精簡為 MDP＋失敗原因＋離線設定，術語加字面解釋；S2 只放用得到的式子；S3 拆成 S3/S3b，架構細節收進 details；玩具說明以白話放在 C1 之前，DCT 收進 details；各章檢查題只問該章內文。 |
| R11 | 研究筆記更新 | notes 已完成（7 篇＋15 筆時間線，6.4）。 | 建置前重新讀取 `notes/`；新增的 ID 若不在 `TL_GROUP` 中，歸入「其他」並回報；JSON 解析失敗時時間線區顯示「時間線資料尚未提供」。 |
| R13 | 執行方式的原文不一致 | Maze2D 以開迴路執行（Sec. 5.1）；Sec. 5.4 同時寫「open loop」與「每步重新產生計畫」。 | S7 與 LO11 分開寫出運動控制（Alg. 1）與 Maze2D 的執行方式，並以註腳指出 Sec. 5.4 的不一致。 |
| R14 | 精確參考的詮釋 | $p(\tau)\exp(\alpha\mathcal J)$ 是分類器引導的常見讀法；論文的目標只有 $\alpha=1$。 | C3 的精確曲線標為「本頁採用的讀法」；不寫「α 不是溫度」。 |
| R12 | 相對連結 | 建置位置與上線位置不同。 | 接受；預覽只檢查外部連結與頁內錨點。 |

---

## 8. 修訂紀錄（round 1）

依 `reviews/round1.md`（27 項，3 項 P0）修訂。項次對應：
- #1（P0）常見錯誤 2 改為「起點與其餘計畫不連貫」〔本頁推論〕，並附玩具數字 1.4422 對 0.1676（`check.py`）。
- #2（P0）、#12：步驟 2 與 S2 加入閉式 $q(\tau^i\mid\tau^0)$，作為暖啟動前向 $k$ 步的公式（步驟 8）；S7 的 SVG 納入 selfTest；LO11 引用 Sec. 5.4，並分開寫運動控制與 Maze2D 的執行方式；新增 R13。
- #3（P0）：6.4 依 notes 改寫（15 個 ID 的 TL_GROUP、hub 閱讀順序、568 與 notes_split 的共用筆記和 title、時間線排序）。
- #4、#5、#6、#7：步驟 5 與 C3 改寫：累積增益的說法、α=1 對應論文目標、三個條件、C3 加入雜訊層級似然模式（預設）、越過與發散標為乾淨 J 的玩具現象、α 不與論文值比較；新增 R14，R7 改寫。
- #8、#9、#10：C2「N ≤ 100 內縮小、未見歸零」與單步倍數；C1「公式對任意 H 有定義」；4.0 不再把 $Q$ 的帶狀結構與時間局部性連在一起；R1 補充。
- #11：LO5、LO8、LO10 改寫；C1 觀察 3 綁到 LO5。
- #13–#16：1.3 標題、MC 敘述與 mc.py 比值、Table 2 較弱列與粗體規則、常見錯誤 4 與 $+\infty$ 記號註。
- #17–#21：大綱：S0、S1 標題改為陳述句；S1 精簡、設計原則移到 S4；S2 精簡；S3 拆成 S3/S3b；玩具說明白話化、DCT 收進 details；S5 內嵌 Alg. 1 第 10 行。章節編號改為 01–13。
- #22–#27：C3 的 `lastGood`/`divergedAt`/valuetext；4.4 以查表降級並以 setTimeout 分片；exactMean 的失敗路徑（4.0、selfTest）；描邊、`--ink-2` 樣本線、深色模式對比；不用組合字元（βt）、display 數學放進橫捲容器、390px 檢查清單；C1 播放／暫停、同年排序。
- verify/：`core.make_guide`（兩種引導模式）；`check.py` 新增雜訊層級似然表、只在最後替換、αβt>2 的步數；`mc.py` 輸出差／標準誤；所有輸出重新產生，原有數字不變。


## 9. 修訂紀錄（readiness）

- 阻擋項：cosine 排程與 $\beta_i,\bar\alpha_i,\tilde\beta_i$ 的定義移到 S2 內文，符號表一致；S3b 只回指。
- 非阻擋項：C3 的 $\mathcal J$ 在平均路徑上計算；$g=\hat g\Phi^\top$；$j=0..5$；Table 2 全表；mulberry32 常數與 uint32→[0,1) 映射；色階截斷與內插；Hero seed 7；C1 取樣、C2/C3 程序平均 z=0。

## 10. 建置紀錄（build round）

- 產出 `build/index.html`（約 108 KB，單檔；外殼沿用 k-diff 頁的 CSS tokens、`#pm-bar`、TOC、plate、KaTeX 設定與 `?test` 框架）。`?test`：87 passed, 0 failed（1440 與 390 寬皆通過，含無水平捲動檢查）。impeccable：0 anti-pattern（13 則 numbered-section-labels advisory 保留，屬 DESIGN.md 的目錄式編號）。LO 自檢：`accept/page.txt` 中 12 個 LO 的關鍵詞與數字皆出現在內文。
- 與設計的差異：S7 的暖啟動圖改以 HTML/CSS 流程圖實作（`#s7-warm`，`role="img"`），不用 SVG，使 390px 時文字維持實際字級；`.quote` 不用左側色條（impeccable side-tab）。章節 data-toc 編號為 00–13（S3b 佔 04）。
- 截圖：`build/caps/`（1440 寬 C1–C3、390 寬 hero／C1–C3／實驗、深色 390 C3 與 1440 C1）。截圖用的本機 KaTeX 預覽檔已刪除。
