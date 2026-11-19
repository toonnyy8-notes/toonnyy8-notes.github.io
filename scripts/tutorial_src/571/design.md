# 設計文件：〈Decision Transformer: Reinforcement Learning via Sequence Modeling〉互動教學

- 論文：Lili Chen、Kevin Lu、Aravind Rajeswaran、Kimin Lee、Aditya Grover、Michael Laskin、Pieter Abbeel、Aravind Srinivas、Igor Mordatch，arXiv 2106.01345（v2，2021 年 6 月）。資料庫編號 No. 571。
- 原文：`tut/571/src/paper.txt`（PDF 同目錄；上標在 txt 中遺失，例如「105」實為 $10^5$，已對照 PDF 第 18–19 頁）。「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」「Alg. 1」皆指論文原文編號。`tut/` = `scripts/tutorial_src/`。
- 產出：`tut/571/build/index.html`（單一 HTML 檔）。上線位置：`tutorials/decision-transformer.html`（slug `decision-transformer`）。站內相對連結一律以上線位置為準。
- 驗算腳本（非頁面程式，與頁面演算法逐步相同）：`tut/571/verify/c1.py`（C1）、`chain.py`（C2）、`graph.py`（C3）、`tables.py`（C4 與內文算術）。輸出存於同目錄 `*_out.txt`。
- 研究筆記：`tut/571/notes/`（總覽 `dt-lineage-overview` + 6 篇主題筆記 + `timeline.json`，12 筆）；共用背景筆記 `rl-mdp-q-learning-basics`、`offline-rl-distribution-shift` 位於 `tut/568/notes/`。
- 標記慣例：**〔本頁推導〕** = 本設計自行推導、論文沒有寫出；**〔本頁簡化〕** = 互動元件為了能在瀏覽器即時計算而替換論文的元件（例如以計數表取代 GPT）；**推定** = 論文沒有寫明、由本設計選定或校準的值；「作者認為」「作者推測」= 論文作者的解釋而非實驗結果。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

本文把強化學習（RL）改寫成**條件式序列建模**：把一條軌跡寫成 returns-to-go（從當下到結尾的報酬總和）、狀態、動作三種 token 交錯的序列（Eq. 2），用 GPT 架構（因果遮罩的 Transformer）在離線資料上以監督式損失預測動作。測試時給定期望的報酬（例如專家水準）與起始狀態作為提示，每執行一步就把目標報酬減去得到的獎勵，再自迴歸地產生下一個動作（Sec. 3、Alg. 1）。方法不擬合價值函數、不計算策略梯度、不使用 Bellman 備份（bootstrapping）與折扣。在 Atari（1% DQN-replay）、D4RL（OpenAI Gym）與 Key-to-Door 上，Decision Transformer（DT）與當時最好的 model-free 離線 RL 方法 CQL 相當或更好（Sec. 4、Table 1–2）；在需要長期信用分配與獎勵延遲的設定中優勢較明顯（Sec. 5.4、5.6，Table 6–7）。

### 1.2 研究動機（Sec. 1）

1. Transformer 在語言與影像上能大規模建模高維分佈；作者想知道「生成式軌跡建模」能否**取代**傳統 RL 演算法，而不只是當作 RL 演算法中的一個元件（引 [4, 5]）。
2. 以序列建模目標訓練，可繞過 TD 學習的 bootstrapping（Sutton & Barto 所稱「deadly triad」之一，會使 RL 不穩定），也不必折扣未來獎勵（折扣可能導致短視行為）。
3. Self-attention 可直接做信用分配；作者認為 Bellman 備份傳遞獎勵很慢，且容易受「干擾」訊號影響（引 [7]）。
4. 離線 RL 從次佳的固定資料學策略，傳統方法受誤差傳遞與價值高估之苦（引 [8]）；以序列建模訓練時，指定「要哪種技能」只需選擇報酬 token 作為生成提示。

### 1.3 貢獻

1. 軌跡表示：以 returns-to-go 取代即時獎勵，得到可自迴歸訓練與生成的序列（Eq. 2）。
2. 架構：每種模態一個線性嵌入 + LayerNorm、每個時間步一個學習到的時間步嵌入（三個 token 共用），送入 GPT，只在狀態 token 的輸出上預測動作（Sec. 3、Fig. 1、Alg. 1）。
3. 實驗：Atari 四款遊戲、D4RL 三種資料集 × 三個環境 + Reacher、Key-to-Door（Sec. 4、5.4）。
4. 分析：%BC 對照、報酬條件的準確度、脈絡長度消融、信用分配、作為 critic、稀疏獎勵、為何不需要價值悲觀（Sec. 5.1–5.8）。
5. 圖上最短路徑的示意例（Sec. 1、Fig. 2、App. A.3）。

### 1.4 從作者的角度逐步拆解

#### 符號表（依章分組；頁面在每章末以可收合「本章符號」列出該章新增的符號，S1 只列 S1 的符號）

| 章 | 符號 | 意義 |
| --- | --- | --- |
| S1 | $(S,A,P,R)$ | MDP：狀態集、動作集、轉移 $P(s'|s,a)$、獎勵函數 $r=R(s,a)$ |
| S1 | $s_t,a_t,r_t$ | 時間步 $t$ 的狀態、動作、**獎勵**（單步）$r_t=R(s_t,a_t)$ |
| S1 | $\tau$、$T$ | 軌跡 $(s_0,a_0,r_0,\dots,s_T,a_T,r_T)$ 與最後一個時間步（Sec. 2.1 從 0 起算；Eq. 2 從 1 起算，頁面依 Eq. 2 從 1 起算並註明） |
| S1 | $\pi$ | 策略：給定狀態（或歷史）選動作的規則（論文只寫 policy，未給符號；本頁記號） |
| S1 | $R_t$ | **報酬**（多步總和）$\sum_{t'=t}^{T}r_{t'}$（Sec. 2.1） |
| S2 | $\gamma$、$V(s)$、$Q(s,a)$ | 折扣因子（DT 不使用）、價值函數、動作價值函數 |
| S2（C2） | $H,\ i,\ \alpha,\ n$ | 鏈的狀態數、狀態編號、更新步長、備份次數（本頁記號） |
| S3 | $\hat R_t$ | returns-to-go（RTG），與 $R_t$ 同值；Sec. 3 的記號，頁面之後一律用它 |
| S4 | $K$、$m$ | 脈絡長度（最近 $K$ 個時間步，$3K$ 個 token）；視窗內實際的時間步數 $m\le K$（本頁記號） |
| S4 | $x_i,\ q_i,k_i,v_i,\ z_i$ | Eq. 1：第 $i$ 個 token 的輸入嵌入、query／key／value、self-attention 輸出 |
| S5 | $G$ | 測試時給定的目標報酬（論文程式碼的 `target_return`；本頁記號） |
| S6（C3） | $L,\ L_0,\ C[s][L][a]$ | 剩餘步數 $L=-\hat R$、固定目標模式的起始目標、計數表（本頁記號） |
| S9 | $X$ | %BC 的百分位：只複製依 episode 報酬排序前 $X\%$ 的時間步 |

#### 步驟 1：問題設定——離線 RL（Sec. 2.1）

以下依論文推導順序，每一步說明作者想回答什麼、得到什麼、下一步為什麼需要它。

- MDP $(S,A,P,R)$；軌跡由狀態、動作、獎勵組成；從時間步 $t$ 起的報酬 $R_t=\sum_{t'=t}^{T}r_{t'}$；RL 的目標是最大化期望報酬 $\mathbb E[\sum_{t=1}^{T}r_t]$。
- 離線 RL：不能與環境互動，只能使用由**任意策略**收集的固定、有限軌跡資料。作者指出這比線上更難，因為代理不能探索、不能收集額外回饋。
- 頁面說明（S1，2–3 句，標「本頁說明」）：資料中的策略可能很差（例如隨機）；作者的說法是要「從固定、有限的經驗產生最有效的行為」（producing maximally effective behavior from fixed, limited experience，Sec. 1），本頁以「希望學到比資料中的策略更好的行為」轉述。S1 同時用一句話區分：**獎勵** $r_t$ 是單步的值，**報酬** $R_t$ 是多步總和。

#### 步驟 2：TD 學習的三個問題（Sec. 1、Sec. 4 的 TD 描述、Sec. 5.7）

- 名詞（S2 內文各用一句字面定義；論文沒有定義這些詞，定義標「本頁說明」）：
  - **TD 學習／Bellman 備份**：用「這一步的獎勵＋下一個狀態的估計價值」更新目前狀態的價值估計。一般形式 $Q(s_t,a_t)\leftarrow r_t+\gamma\max_{a'}Q(s_{t+1},a')$（論文沒有寫出此式）。
  - **bootstrapping**：更新目標裡含有自己的估計值。
  - **deadly triad**：Sutton & Barto 指出的三個條件——函數近似、bootstrapping、off-policy 學習——同時出現時訓練可能發散（論文只寫 bootstrapping 是其中之一）。
  - **CQL（Conservative Q-Learning）**：一種離線 TD 方法，刻意壓低資料中沒出現過的動作的價值估計；論文稱為「TD learning with value pessimism」，這種壓低就是**價值悲觀**。
  - 論文的「Bellman 備份傳遞獎勵很慢」：每做一次備份，獎勵的資訊只往前移一個狀態。
- 作者列出的三個問題（Sec. 1）：bootstrapping 是 deadly triad 之一；折扣可能造成短視（short-sighted）行為；Bellman 備份逐步傳遞獎勵、慢且受干擾訊號影響。
- Sec. 5.7（作者推測，conjecture）：TD 方法學到近似價值函數後**對它最佳化**，會放大並利用近似誤差，所以離線 TD 方法需要策略正則化或保守性；DT 不把學到的函數當作最佳化目標，因此不需要。（頁面在 S2 引述，S11 回指。）
- C2 用的式子（S2 內文寫出）：單一動作、只評估不選動作的鏈（**策略評估**，沒有 max），同步更新 $V_i\leftarrow(1-\alpha)V_i+\alpha(r_i+\gamma V_{i+1})$，$H$ 個狀態，只有離開 $s_H$ 時 $r_H=1$，從 $V\equiv0$ 開始。
- 內文只講 $\alpha=1$：第 $n$ 次備份後恰好最右 $n$ 個狀態有值，所以要 $H$ 次 $V_1$ 才有值，極限 $\gamma^{H-1}$（$H=30,\gamma=0.99$：0.747）。$\alpha<1$ 時大約慢 $1/\alpha$ 倍：$H=30,\gamma=1$ 時 $V_1$ 達一半極限需 $n=30,59,119,297$（$\alpha=1,0.5,0.25,0.1$，`chain.py`）。
- 〔本頁推導〕（放在 S2 的 `<details>`「進階：閉式解」；先定義：$\mathrm{Bin}(n,\alpha)$ 是做 $n$ 次獨立試驗、每次成功機率 $\alpha$ 時的成功次數，$\Pr[\cdot]$ 表示事件的機率；每次備份可看成一次「前緣是否往前推一格」的試驗）：
  $$V_i^{(n)}=\gamma^{H-i}\,\Pr\big[\mathrm{Bin}(n,\alpha)\ge H-i+1\big].$$
  證明：對 $n$ 歸納；令 $m=H-i+1$，$\Pr[\mathrm{Bin}(n{+}1,\alpha)\ge m]=(1-\alpha)\Pr[\mathrm{Bin}(n,\alpha)\ge m]+\alpha\Pr[\mathrm{Bin}(n,\alpha)\ge m-1]$ 正好是更新式。任何 $\alpha>0$ 在 $n=H$ 時 $V_1>0$，但 $\alpha<1$ 時極小（$\alpha=0.5,\gamma=0.99,n=30$：$6.96\times10^{-10}$），所以不能說「一樣快」。
- RTG 的對照：RTG 是對**已結束**的離線軌跡做一次反向累加（後綴和），每個時間步立即得到 $\hat R_t=1$，不需要迭代、不折扣。
- 頁面必須說明的限制：(a) C2 是表格、單一動作、確定性的示意，不是論文的實驗；(b) RTG 是**收集資料的策略**實際得到的報酬，不是最佳價值，所以 DT 要「以報酬為條件」才能挑出好的行為（S3）；(c) 折扣使 $V_1$ 的極限變小（$\gamma=0.9,H=100$：$3.0\times10^{-5}$），這是**目標函數不同**（折扣報酬 vs 未折扣報酬），不是傳遞速度的問題；論文對折扣只說它「可能導致短視行為」。

#### 步驟 3：軌跡表示——returns-to-go（Sec. 3，Eq. 2）

- 作者的需求：表示法要讓 Transformer 學到有意義的模式，並能在測試時**條件式**地產生動作。直接輸入獎勵不合適，因為希望模型依據「未來想得到的報酬」而不是「過去的獎勵」產生動作。
- 定義 $\hat R_t=\sum_{t'=t}^{T}r_{t'}$（未折扣），軌跡表示為
  $$\tau=\big(\hat R_1,s_1,a_1,\hat R_2,s_2,a_2,\dots,\hat R_T,s_T,a_T\big)\tag{Eq. 2}$$
- 測試時（Sec. 3 第一段）：指定期望表現（例如成功為 1、失敗為 0）與起始狀態作為條件；執行產生的動作後，把目標報酬**減去得到的獎勵**，重複直到 episode 結束。Alg. 1：`R = R + [R[-1] - r]`。
- 〔本頁推導〕（S3 內文寫出前提：實際測試時獎勵取決於模型產生的動作，所以這只在獎勵序列相同的假設下成立）若測試時的獎勵序列與某條訓練軌跡相同，測試時的 $\hat R_t$ 與訓練時的 RTG 在每個 $t$ 都相差同一個常數，即「測試時給的初始目標 − 該軌跡的實際報酬」（由遞減規則逐項相減）；初始目標等於實際報酬時兩者完全相同。（符號 $G$ 到 S5 才引入，S3 只用文字。）C1 示範這一點。

#### 步驟 4：架構（Sec. 2.2、Sec. 3，Eq. 1，Fig. 1，Alg. 1）

- Self-attention（Eq. 1）：$z_i=\sum_{j=1}^{n}\mathrm{softmax}\big(\{\langle q_i,k_{j'}\rangle\}_{j'=1}^{n}\big)_j\cdot v_j$。GPT 加上因果遮罩：第 $i$ 個 token 只看 $j\in[1,i]$。作者認為 attention 經由 query 與 key 的相似度「隱式地形成狀態與報酬的關聯」，這就是信用分配（Sec. 2.2）。
- 輸入最近 $K$ 個時間步，共 $3K$ 個 token（RTG、狀態、動作各一）。每種模態一個線性層投影到嵌入維度，接 LayerNorm；影像狀態改用卷積編碼器。
- **時間步嵌入**：每個時間步學一個嵌入，加到該時間步的三個 token 上——與 Transformer 標準的逐 token 位置嵌入不同，因為一個時間步對應三個 token（Sec. 3、Alg. 1 `pos_embedding = embed_t(t)  # per-timestep (note: not per-token)`）。
- S4 先用兩句說明標準位置嵌入：Transformer 本身不知道 token 的順序，所以通常給第 1、2、3…個 token 各加一個位置向量；DT 改成同一時間步的三個 token 共用一個向量。Eq. 1 逐項說明（$q_i$ 與每個 $k_j$ 的內積經 softmax 成為權重，對 $v_j$ 加權平均）。
- 〔本頁推導〕token 位置（從 1 起算）：時間步 $j$ 的 $\hat R$、$s$、$a$ 分別在 $3j-2$、$3j-1$、$3j$。預測 $a_t$ 的是 $s_t$ 那個位置的輸出，因果遮罩下它看到 $3m-1$ 個 token（$m$ 為視窗內的時間步數），不包含 $a_t$ 本身。

#### 步驟 5：訓練（Sec. 3，Alg. 1）

- 從資料抽長度 $K$ 的片段組成 minibatch；$s_t$ 位置的預測頭以交叉熵（離散動作）或均方誤差（連續動作）預測 $a_t$；各時間步的損失取平均。
- 主實驗不預測狀態或 RTG：作者沒有發現能提升表現，但框架允許；Sec. 5.5（critic）與 App. A.3（最短路徑）的模型另外預測 RTG token。
- 論文內部的不一致（S5 註記一次）：Sec. 3 內文說由 $s_t$ 位置的預測頭預測 $a_t$；Alg. 1 的註解寫「select hidden states for action prediction tokens」（`unstack(hidden_states).actions`）。頁面依 Sec. 3 內文。

#### 步驟 6：示意例——圖上最短路徑（Sec. 1、Fig. 2、App. A.3，Fig. 6）

- 固定有向圖；在目標節點獎勵 0，否則 −1；觀測與動作都是節點編號；有邊則移到動作指定的節點，否則留在原地。RTG 是負的剩餘路徑長，最大化 RTG 等於走最短路徑。
- 資料：1,000 條隨機遊走軌跡，每條 $T=10$ 步；隨機圖 20 個節點，edge sparsity coefficient 0.1。沒有任何專家示範。
- 模型：此例的 GPT 同時產生動作與 RTG token（與主實驗只預測動作不同，S6 開頭先說明）。生成時以 prior $P_{\text{prior}}(\hat R=k)\propto T+1-k$ 偏好短路徑；論文沒有說明 $k$，照字面（$\hat R\le0$）會偏好長路徑，所以頁面寫成「$k$ = 路徑步數，即 $\hat R=-k$（推定）」。prior 式放在 S6 的 `<details>`，並註明 C3 不使用它。與 GPT 的 RTG 機率結合：$P(\hat R_t|\dots)=P_{\text{GPT}}(\hat R_t|s_{0:t},a_{0:t-1},\hat R_{0:t-1})\times P_{\text{prior}}(\hat R_t)^{10}$。作者指出 prior 與 RTG 預測都由模型自己計算，不需要最佳路徑長等外部資訊。
- 結果（Fig. 6，橫軸 1–9 步；圖說提到 ∞ 但圖上沒有 ∞ 欄）：生成路徑的步數分佈接近最短路徑。作者給的兩個原因：資料中可能剛好有對應最短路徑的片段；或生成的路徑把次佳片段**拼接**（stitching）成資料中沒有的新路徑，後者占生成路徑的 15.8%。
- 作者據此說明：結合序列建模與事後（hindsight）報酬資訊，不需要動態規劃就能改進策略（Sec. 1）。此 prior 只用在這個例子。
- C3 以〔本頁簡化〕重現此例：GPT 換成計數表 $C[s][L][a]$（$K=1$，$L=-\hat R$ 為剩餘步數），「每步取最大回報」對應 Fig. 2 圖說的 generating largest possible return at each node。細節見 C3。

#### 步驟 7：實驗設定與基準（Sec. 4）

- 主要比較：model-free、基於 TD 的離線 RL（CQL [14] 為主，另有 BEAR、BRAC、AWR，Atari 另有 REM、QR-DQN），因為 DT 本質上也是 model-free；以及行為複製（BC），因為同樣是基於似然的策略學習。
- Atari（Sec. 4.1）：DQN-replay 的 1%（5,000 萬筆轉移中的 50 萬筆）；3 個 seed 的平均與 ±（Sec. 4.1 內文寫 mean and standard deviation，Table 1、2、12、13 的說明寫 mean and variance；頁面標「± 的意義未定」）；以職業玩家分數正規化：100 = 職業玩家、0 = 隨機策略，即 $100\cdot\frac{\text{raw}-\text{random}}{\text{gamer}-\text{random}}$（〔本頁推導〕把協定寫成式子，並以 Table 11、12 驗證：Breakout DT 76.9 → 267.5）。$K=30$（Pong 為 50）。BC 使用相同架構與超參數但沒有 RTG 條件。CQL、REM、QR-DQN 的數字取自原論文。
- Table 1（gamer-normalized）：Breakout DT 267.5±97.5、CQL 211.1；Qbert 15.4±11.4、104.2；Pong 106.1±8.1、111.9；Seaquest 2.5±0.4、1.7。BC 138.9、17.3、85.2、2.1；QR-DQN 17.1、0.0、18.0、0.4；REM 8.9、0.0、0.5、0.7。結論（Sec. 4.1）：4 款中 3 款與 CQL 相當；對 REM、QR-DQN 四款都持平或勝出；對 BC，Qbert 上 BC 略高（17.3 vs 15.4，差距在標示的 ± 之內），表格說明寫「outperforms other baselines in most games」。BC（行為複製，behavior cloning）在 S7 第一次出現時定義：直接以監督式學習模仿資料中的動作，不使用報酬。
- D4RL（Sec. 4.2）：Medium（約達專家三分之一分數的策略產生的 100 萬步）、Medium-Replay（訓練到 medium 水準的代理的 replay buffer，約 25k–400k 步）、Medium-Expert（medium 100 萬步 + expert 100 萬步）；HalfCheetah、Hopper、Walker 與自建的 2D Reacher（目標條件、稀疏獎勵）。100 = 專家。CQL 數字取自原論文，BC 由作者自己跑，其他取自 D4RL 論文；Reacher 只跑 CQL。
- Table 2 平均：不含 Reacher DT 74.7、CQL 63.9、BEAR 48.2、BRAC-v 36.9、AWR 34.3、BC 46.4；全部設定 DT 69.2、CQL 54.2、BC 47.7（BC 的兩個平均見 R1）。

#### 步驟 8：討論實驗（Sec. 5）

- **5.1 %BC**：只對依 episode 報酬排序前 $X\%$ 的時間步做 BC，$X=100\%$ 即標準 BC，$X\to0\%$ 只複製最好的一條。作者強調選 $X$ 需要在環境中評估，所以 %BC 不是實際可用的方法，只用來理解 DT。D4RL（Table 3，只有 Medium 與 Medium-Replay 兩種資料集、各含 Reacher 共 8 列，與 Table 2 的列組合不同）：資料充足時 %BC 可與其他離線 RL 持平或更好，DT 在多數環境與最好的 %BC 相當（8 列平均 DT 56.1、10%BC 56.7、25%BC 52.7、40%BC 49.4、100%BC 39.5、CQL 43.5；DT 平均見 R1）。Atari 1% 資料（Table 4）：%BC 弱，例如 Breakout 10%BC 28.5、25%BC 73.5、40%BC 108.2、100%BC 138.9，DT 267.5。作者解讀：資料少時 DT 能利用所有軌跡（即使與目標報酬不同）來泛化。
- **5.2 報酬分佈**：改變目標報酬，實際得到的報酬與目標高度相關（Fig. 4）；Pong、HalfCheetah、Walker 幾乎貼合 oracle 線；Seaquest 可用高於資料中最大 episode 報酬的目標提示，作者稱「有時能外推」。
- **5.3 脈絡長度**（Table 5，Atari）：$K=1$ 明顯變差：Breakout 267.5→73.9、Qbert 15.1→13.6、Pong 106.1→2.5、Seaquest 2.5→0.6。作者的假設：表示「策略的分佈」時，脈絡讓 Transformer 辨認是哪個策略產生了動作。
- **5.4 長期信用分配**（Table 6，Key-to-Door）：三階段（有鑰匙的房間 → 空房間 → 有門的房間），第一階段撿到鑰匙、第三階段到門才得二元獎勵；資料是隨機動作軌跡；脈絡用整個 episode。成功率 1K 條：DT 71.8%、CQL 13.1%、BC 1.4%、%BC（只用成功 episode）69.9%、Random 3.1%；10K 條：94.6%、13.3%、1.6%、95.1%、3.1%。
- **5.5 作為 critic**：讓 DT 也輸出 RTG token，且第一個 RTG 由模型預測（學 $p(\hat R_1)$）；預測的成功機率隨 episode 事件更新（Fig. 5 左），attention 集中在撿鑰匙與到門（Fig. 5 右）。
- **5.6 稀疏獎勵**（Table 7，Hopper）：獎勵全部延到最後一步。延遲版 DT 107.3/60.7/78.5（ME/M/MR），原始 DT 107.6/67.6/82.7；CQL 延遲 9.0/5.2/2.0、原始 111.0/58.0/48.6；BC 59.9/63.9/27.6、%BC 102.6/65.9/70.6（模仿方法與獎勵無關）。
- **5.7** 見步驟 2。**5.8**：作者認為 DT 可作為線上 RL 的行為生成模型，例如與 Go-Explore 結合（未做實驗）。

#### 步驟 9：與相關工作的區分（Sec. 6）

- 離線 RL 既有作法：限制動作空間、價值悲觀、在學到的動力學模型中加悲觀；DT 不學動力學模型，所以只比 model-free 方法（Sec. 6.1）。
- Upside-down RL（UDRL）與 Reward-Conditioned Policies 同樣以目標報酬為條件做監督式學習；作者說差別在於動機轉向序列建模，實作上主要差在脈絡長度與架構；Kumar et al. [44] 最接近 DT 的 $K=1$，而 Sec. 5.3 顯示長脈絡較好（Sec. 6.2）。
- 同期的 Trajectory Transformer [49] 另外預測狀態與報酬並離散化，含 model-based 成分（Sec. 6.2）。
- 信用分配：先前方法需另外學獎勵分解或 critic；DT 讓這些性質在 Transformer 中自然出現（Sec. 6.3）。條件式語言生成多假設固定「類別」，RL 的報酬條件隨時間遞減（Sec. 6.4）。

#### 步驟 10：實作細節（App. A，Table 8–10）

| 項目 | Atari（Table 8） | OpenAI Gym（Table 9） | Gym BC（Table 10） |
| --- | --- | --- | --- |
| 程式基礎 | minGPT | Huggingface Transformers | MLP |
| layers / heads / emb. | 6 / 8 / 128 | 3 / 1 / 128 | 3 / — / 256 |
| batch | 512 Pong；128 其他 | 64 | 64 |
| 脈絡 $K$ | 50 Pong；30 其他 | 20 HalfCheetah、Hopper、Walker；5 Reacher | — |
| RTG 條件 | Breakout 90（≈1× 資料最大）、Qbert 2500（≈5×）、Pong 20（≈1×）、Seaquest 1450（≈5×） | HalfCheetah 6000、Hopper 3600、Walker 5000、Reacher 50 | — |
| nonlinearity | ReLU（編碼器）、GeLU（其他） | ReLU | ReLU |
| 編碼器 | channels 32, 64, 64；filters 8×8, 4×4, 3×3；strides 4, 2, 1 | — | — |
| dropout | 0.1 | 0.1 | 0.1 |
| learning rate | $6\times10^{-4}$ | $10^{-4}$ | $10^{-4}$ |
| 最佳化 | Adam betas (0.9, 0.95)；grad norm clip 1.0；weight decay 0.1 | AdamW（PyTorch 預設）；grad norm clip 0.25；weight decay $10^{-4}$ | weight decay $10^{-4}$ |
| 學習率排程 | 線性 warmup + cosine decay；warmup tokens 512∗20；final tokens 2∗500000∗K | 前 $10^5$ 步線性 warmup | 前 $10^5$ 步線性 warmup |
| 訓練長度 | max epochs 5 | $10^5$ gradient steps | $2.5\times10^4$ gradient steps |

常見錯誤（頁面 S13 逐條列出）：

1. 時間步嵌入做成逐 token：論文是每個時間步一個嵌入，加到該步三個 token（Sec. 3、Alg. 1）。
2. 把即時獎勵當輸入：輸入的是 RTG（未折扣的後綴和），測試時每步遞減 `R[-1] - r`（Alg. 1）。
3. 對所有 token 計算損失：主實驗只預測動作，損失只在動作上（Sec. 3）；Sec. 5.5 與 App. A.3 的版本另外預測 RTG，是刻意的延伸。
4. 目標報酬隨意選：Atari 用資料最大報酬的 1× 或 5×（App. A.1）；Gym 依專家表現，HalfCheetah 用 50%，因為資料的相對報酬較低（App. A.2.1）。作者寫「more possibilities exist for principled return-to-go conditioning」。
5. 忽略 Atari 的差異：嵌入後用 Tanh 而非 LayerNorm（作者說對表現影響不大）；狀態用 DQN 編碼器加一層線性投影（App. A.1）。
6. 脈絡長度一體適用：Atari 30（Pong 50）、Gym 20、Reacher 5（目標條件、episode 較短）、Key-to-Door 用整個 episode（App. A、Sec. 5.4）。
7. 測試時忘了截斷脈絡：每步只保留最後 $K$ 個時間步（Alg. 1 `R[-K:]`）。

#### 步驟 11：限制與作者的展望（Sec. 7）

- 作者展望：大規模資料可受益於自監督預訓練；更精細的 RTG／狀態／動作嵌入，例如以**報酬分佈**為條件來處理隨機環境（目前以確定的報酬為條件）；以 Transformer 建模狀態演化，作為 model-based RL 的替代。
- 風險：Transformer 在 MDP 中的錯誤類型與負面後果研究不足；訓練資料可能帶入破壞性偏差；有心人的獎勵設計可能透過報酬條件產生非預期行為。
- 頁面另寫（不是論文的話，標「本頁說明」）：Sec. 7 的「隨機環境」一句表示以確定報酬為條件在隨機轉移下可能不足；本頁不延伸討論。

---

## 2. 學習目標（驗收測試）

每個參考答案都必須寫在指定章節的**內文**（不能只出現在元件、表格、圖或答案框）。每章末以 `<details>`「檢查理解」照錄題目與參考答案。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明離線 RL 的設定 | 離線 RL 與一般（線上）RL 的差別是什麼？為什麼較難？ | 只能用任意策略收集的固定、有限軌跡資料，不能與環境互動；因為不能探索、不能收集額外回饋（Sec. 2.1） | S1 |
| LO2 | 說出 TD 學習的三個問題 | 作者列出 TD 學習的哪三個問題？Sec. 5.7 如何解釋 DT 不需要價值悲觀？ | bootstrapping（deadly triad 之一）、折扣造成短視、Bellman 備份傳遞獎勵慢且受干擾訊號影響（Sec. 1）；作者推測 TD 方法對學到的價值函數做最佳化會放大其誤差，DT 不把學到的函數當最佳化目標（Sec. 5.7） | S2 |
| LO3 | 理解備份的傳遞速度 | C2 的鏈（$H$ 個狀態、只有結尾有獎勵）中，$\alpha=1$ 的同步備份要幾次 $V_1$ 才不為 0？RTG 呢？ | $H$ 次，極限為 $\gamma^{H-1}$（$H=30,\gamma=0.99$ 時 0.747）；$\alpha<1$ 時大約慢 $1/\alpha$ 倍；RTG 對已結束的軌跡做一次後綴和，每步立即為 1（〔本頁推導〕） | S2 |
| LO4 | 定義 returns-to-go 並說明理由 | $\hat R_t$ 怎麼定義？為什麼不直接輸入獎勵？ | $\hat R_t=\sum_{t'=t}^{T}r_{t'}$（未折扣）；希望模型依未來想得到的報酬產生動作，而不是依過去的獎勵（Sec. 3） | S3 |
| LO5 | 描述輸入序列與嵌入 | 脈絡長度 $K$ 對應幾個 token？時間步嵌入與一般位置嵌入有何不同？ | $3K$ 個（RTG、狀態、動作交錯，Eq. 2）；每個時間步一個學習到的嵌入，加到該步三個 token 上，不是逐 token（Sec. 3） | S4 |
| LO6 | 說明訓練目標 | 主實驗中哪個位置的輸出預測 $a_t$？用什麼損失？ | $s_t$ 位置（Sec. 3 內文）；離散動作用交叉熵、連續動作用均方誤差，各步平均；主實驗不預測狀態與 RTG，Sec. 5.5 與 App. A.3 的版本另外預測 RTG（Sec. 3） | S5 |
| LO7 | 說明測試時的生成流程 | 測試時如何設定與更新 RTG？Atari 的目標報酬怎麼選？ | 先給目標報酬與起始狀態；每執行一步，RTG 減去得到的獎勵，並只保留最後 $K$ 步（Sec. 3、Alg. 1）；Atari 用資料最大報酬的 1× 或 5×（App. A.1） | S5 |
| LO8 | 解釋最短路徑例 | 訓練資料是什麼？生成時以什麼為條件？拼接占多少？ | 1,000 條 $T=10$ 的隨機遊走（20 節點圖），無專家示範；在每個節點產生最大可能的 RTG 並以它為條件產生動作（Fig. 2）；15.8% 的生成路徑由次佳片段拼接、不在資料中（Sec. 1、App. A.3） | S6 |
| LO9 | 讀出 Atari 結果 | Atari 用多少資料？分數如何正規化？DT 與 CQL 比較如何？ | DQN-replay 的 1%（5,000 萬中的 50 萬筆轉移）；$100\cdot(\text{raw}-\text{random})/(\text{gamer}-\text{random})$；論文的結論是 4 款中 3 款與 CQL 相當（competitive）；逐款看：Breakout DT 較高（267.5 vs 211.1），Pong、Seaquest 接近（106.1 vs 111.9、2.5 vs 1.7），Qbert 落後（15.4 vs 104.2）（Sec. 4.1、Table 1） | S7 |
| LO10 | 讀出 D4RL 結果 | D4RL 上 DT 與 CQL 的平均分數？ | 不含 Reacher：74.7 vs 63.9；全部設定：69.2 vs 54.2（Table 2） | S8 |
| LO11 | 解讀 %BC 與脈絡長度消融 | (a) %BC 在資料充足（D4RL）與資料少（Atari 1%）時的結論？(b) $K=1$ 時 Pong 分數？ | (a) D4RL 上 %BC 可與其他離線 RL 持平，DT 在多數環境與最好的 %BC 相當；Atari 上 %BC 弱、DT 較好（Sec. 5.1）；(b) 106.1→2.5，作者假設脈絡用來辨認產生動作的策略（Sec. 5.3、Table 5） | S9 |
| LO12 | 讀出信用分配與稀疏獎勵結果 | Key-to-Door 1K 條隨機軌跡的成功率？獎勵延遲到最後一步時 Hopper 上 DT 與 CQL 如何變化？ | DT 71.8%、CQL 13.1%（Table 6）；DT 107.6/67.6/82.7→107.3/60.7/78.5，CQL 111.0/58.0/48.6→9.0/5.2/2.0（Table 7） | S10、S11 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），圖版最大 1080px。章節編號以 JetBrains Mono 顯示（`01 · …`），標題用陳述句。每個符號在首次使用的章節定義（第 1.4 節符號表第三欄）；每個元件放在它用到的概念之後。視覺記號：一條「token 列」（RTG、狀態、動作三種形狀：RTG 菱形、狀態圓形、動作方形，以形狀＋文字區分），在 hero、S3、S4、C1 重複出現。

| # | id | 標題 | 內容 | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈Decision Transformer：把強化學習寫成序列建模〉 | 副標：「這篇論文把軌跡寫成報酬、狀態、動作交錯的序列，用 GPT 預測動作；測試時以想要的報酬作為提示。」靜態 token 列 SVG（只用文字標籤「報酬總和」「狀態」「動作」三種形狀重複兩組，不用符號；動作 token 上方標「預測」）。三張事實卡（見下）。連結：arXiv、`../index.html#paper-571`（在關聯圖中查看）、`dt-lineage-overview.html`、程式碼網站 | 1 螢幕 |
| S1 | `offline` | 01 · 離線 RL 只能使用固定的資料 | MDP、軌跡、策略 $\pi$；獎勵 $r_t$（單步）與報酬 $R_t$（多步總和）的區分；離線 RL 定義與難處（答 LO1）；資料可能來自隨機策略（本頁說明）。章末「本章符號」。延伸：`rl-mdp-q-learning-basics.html`、`offline-rl-distribution-shift.html` | 300–400 字 |
| S2 | `td` | 02 · TD 學習的三個問題 | TD／Bellman 備份、bootstrapping、deadly triad、CQL 與價值悲觀各一句字面定義；作者的三個問題與 Sec. 5.7 推測（答 LO2）；C2 的策略評估式；$\alpha=1$ 需 $H$ 次（答 LO3）；閉式解放在 `<details>`；C2；圖下三點限制。延伸：`dt-offline-rl-baselines-datasets.html`、`dt-credit-assignment-lineage.html` | 450–550 字 + C2 |
| S3 | `rtg` | 03 · 以 returns-to-go 取代即時獎勵 | $\hat R_t$ 定義、Eq. 2（< 600px 手動換成兩行）、理由（答 LO4）；RTG 是資料策略的實際報酬，所以要以它為條件；遞減規則與常數差（本頁推導，寫明獎勵序列相同的前提）。延伸：`return-conditioned-sequence-modeling.html` | 300–400 字 |
| S4 | `arch` | 04 · GPT 以因果遮罩讀取 3K 個 token | 標準位置嵌入兩句 → Eq. 1 逐項說明 → 因果遮罩 → $3K$ token、模態嵌入＋LayerNorm、時間步嵌入（答 LO5）；token 位置 $3j-2,3j-1,3j$（本頁推導）；兩張靜態 SVG：Fig. 1 重繪、$m=2$（$t=2$，$s_t$ 為第 5 個 token）的 6×6 遮罩矩陣（列＝查詢 token、欄＝可看的 token，可看的格填色並標「✓」，$s_t$ 列以粗框標出）。延伸：`dt-transformer-lm-background.html` | 400–500 字 + 2 張靜態 SVG |
| S5 | `loop` | 05 · 訓練只預測動作，測試時遞減目標報酬 | 訓練損失（答 LO6，含 Sec. 5.5／A.3 預測 RTG 的例外與 Alg. 1 註解的不一致）；Alg. 1 兩欄對照（≥ 768px 左程式碼右說明，逐行對齊；< 768px 每行下接說明，程式碼區 `overflow-x:auto`）；測試流程與目標報酬選法（答 LO7）；C1。延伸：`dt-method-and-results.html` | 400–500 字 + Alg. 1 + C1 |
| S6 | `graph` | 06 · 隨機遊走資料也能產生最短路徑 | 開頭說明此例模型另外預測 RTG；App. A.3 設定、Fig. 6 結論（橫軸 1–9）、15.8% 拼接（答 LO8）；prior 式（$k$＝路徑步數，推定）放 `<details>` 並註明 C3 不用；〔本頁簡化〕清單；C3 | 400–500 字 + C3 |
| S7 | `atari` | 07 · Atari：1% 資料下與 CQL 相當 | BC 定義；資料、正規化式（答 LO9）、基準、Table 1 摘錄（DT、CQL、BC 三欄）；Qbert 上 BC 略高的例外；± 的意義未定；C4a 正規化計算器 | 300–400 字 + 小表 + C4a |
| S8 | `gym` | 08 · D4RL：12 個設定中 7 個分數最高 | 三種資料集、Reacher、正規化、平均（答 LO10）；逐列最高者：DT 7 列、BRAC-v 3 列、CQL 2 列（`tables_out.txt`）；Fig. 3 只用文字描述（論文未給數字） | 250–350 字 |
| S9 | `analysis` | 09 · DT 不只是在資料子集上做行為複製 | 5.1 %BC（Table 3 的列組合）、5.2 報酬條件與外推、5.3 脈絡長度（答 LO11） | 400–500 字 |
| S10 | `credit` | 10 · 長期信用分配：明顯勝過 CQL，與 %BC 相當 | 5.4 Key-to-Door（答 LO12 前半）、5.5 critic（Fig. 5 文字描述；再提一次此版本預測 RTG） | 300–400 字 |
| S11 | `sparse` | 11 · 獎勵延遲時 DT 下降 0.3–6.9 分，CQL 下降 46.6–102.0 分 | 5.6 稀疏獎勵（答 LO12 後半；註記 Table 7 BC 59.9 與 Table 2 的 79.6 不同）、5.7 回指 S2、5.8；C4b 結果表 | 300–400 字 + C4b |
| S12 | `related` | 12 · 與 UDRL、RCP、Trajectory Transformer 的差別 | Sec. 6 摘要（步驟 9）。延伸：`dt-transformers-in-rl-and-trajectory-transformer.html` | 250–300 字 |
| S13 | `impl` | 13 · 實作設定與常見錯誤 | 步驟 10 全表（< 480px 改為每個項目一張堆疊卡片）＋ 7 條常見錯誤 | 300 字 + 表 |
| S14 | `history` | 14 · 研究脈絡 | 時間線（6.4）+ 一段文字：DQN → Transformer → 離線 TD（BCQ、BEAR、CQL、D4RL）→ 報酬條件（UDRL、RCP）→ 信用分配（RUDDER）→ 本文 | 時間線 + 100–150 字 |
| S15 | `next` | 15 · 限制與延伸閱讀 | Sec. 7（步驟 11）；筆記清單（6.4）；arXiv、程式碼 | 200 字 + 清單 |

Hero 事實卡：
1. 「輸入不是即時獎勵，而是從現在到結尾的報酬總和（returns-to-go），與狀態、動作交錯成序列。」
2. 「訓練只是監督式地預測資料中的動作；沒有價值函數、沒有 Bellman 備份、沒有折扣。」
3. 「測試時給定想要的報酬，每走一步就減去得到的獎勵。在機器人控制的離線資料集上，平均分數高於當時最好的離線 TD 方法（74.7 對 63.9，見第 8 節）。」

---

## 4. 互動元件規格

### 4.0 共用規則

- 所有元件的讀數由頁面即時計算；論文數字只出現在 C4 的資料與內文。
- PRNG（C3）：`mulberry32(seed)`，與 k-diff 頁相同的實作：`a=a+0x6D2B79F5|0; t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296`。`graph.py` 以 32 位元遮罩逐位元重現。
- 更新：`input` 只記錄狀態，以 `requestAnimationFrame` 合併重畫（`sched`）；live region 與 SVG `aria-label` 只在 `change` 或停止輸入 500ms 後更新（`announce`）。
- 分段按鈕用 `radiogroup()`（roving tabindex、方向鍵）；滑桿有 `<label>` 與 `aria-valuetext`；觸控區 ≥ 44px；不在圖上做 hover 選取或拖曳，所有選取都有滑桿／下拉選單／按鈕的對應操作。
- 動態文字（讀數、圖上標籤、aria）不呼叫 KaTeX，也**不用**組合字元 U+0302 與 Unicode 下標字母（ₜ 等）：
  - HTML 讀數：RTG 一律寫「RTG」加 `<sub>`（例 `RTG<sub>1</sub>`、`RTG<sub>T+1</sub>`），次方用 `<sup>`（`γ<sup>H−1</sup>`），變數用 `<i>`。
  - SVG：下標用 `<tspan baseline-shift="sub" font-size="75%">`；token 圖形上的 R̂ 由 `hatR(x,y,idx)` 畫出：文字「R」＋上方一條 6px 寬的折線帽號（path）＋下標 tspan。
  - aria／live region：純文字「RTG 1」「gamma 的 H−1 次方」。
  - 允許的非 ASCII 字元：中文、γ、α、−、×、≥、≤、∞、·、→、…、✓。建置時在 390px 截圖確認無缺字。

### C1 · Returns-to-go、token 視窗與測試時遞減（S5）

- **教的概念**：$\hat R_t$ 是後綴和（LO4）；最近 $K$ 步組成 $3K$ 個 token、時間步嵌入三個 token 共用、$s_t$ 位置在因果遮罩下看到哪些 token（LO5、LO6）；測試時 $\hat R\leftarrow\hat R-r$ 與只保留最後 $K$ 步（LO7）。
- **輸入**：
  - 分段按鈕「獎勵序列」：稀疏 `[0,0,0,0,0,0,0,1]`（Key-to-Door 風格，$T=8$）／密集 `[1,0,2,1,0,3,1,2]`（$T=8$，自訂示例）／最短路徑 `[-1,-1,-1,-1,0]`（$T=5$，App. A.3 的獎勵規則）。預設稀疏。
  - 分段按鈕「模式」：訓練（RTG 由資料算出）／測試（RTG 由目標 $G$ 遞減）。預設訓練。
  - 滑桿「目前時間步 $t$」：1…$T$，預設 $T$。
  - 滑桿「脈絡長度 $K$」：1…$T$，預設 3（頁面註明論文 Atari 用 30、Pong 用 50、Gym 用 20、Reacher 用 5）。換序列時 $t,K$ 夾到 $[1,T]$。
  - 滑桿「目標報酬 $G$」（只在測試模式顯示）：整數，範圍 [該序列報酬 − 10, 該序列報酬 + 10]，預設等於該序列的實際報酬。
- **計算**（`c1.py` 相同）：
  - 訓練：`rtg[t] = r[t] + rtg[t+1]`，`rtg[T+1]=0`（由後往前一次）。
  - 測試：`R[1]=G; R[t+1]=R[t]-r[t]`（假設環境回饋的獎勵就是該預設序列；圖下寫明「只為示範遞減規則，實際獎勵取決於模型產生的動作」）。另顯示 `R[T+1]`（episode 結束時剩下的目標）與「與訓練 RTG 的差」＝ $G-\sum r$（每步相同，本頁推導）。
  - 視窗：首個時間步 $a=\max(1,t-K+1)$，$m=t-a+1$。訓練模式：輸入 $3m$ 個 token，$s_t$ 位置看到 $3m-1$ 個（$a_t$ 在輸入中但被遮罩）。測試模式：$a_t$ 尚未產生，輸入只有 $3m-1$ 個 token，$s_t$ 是最後一個並看到全部（Alg. 1 在取樣後才把動作接到序列）。
- **顯示**：
  - 上方表格：列 $t$、$r_t$、$\hat R_t$（訓練或測試）；目前視窗內的欄以 `--accent-soft` 底色並加上方括號標記「視窗」。
  - 下方 token 列 SVG：視窗內 $3m$ 個 token，依序 $\hat R,s,a$（菱形／圓形／方形＋文字 `R̂3`、`s3`、`a3`），每組三個 token 下方一條橫括號標「時間步嵌入 t=3（三個 token 共用）」；位置編號（$3j-2,3j-1,3j$，以視窗內的相對位置 1…$3m$ 顯示）。$s_t$ 的輸出畫一個向上箭頭標「預測 aₜ」；它能看到的 $3m-1$ 個 token 實線外框；訓練模式 $a_t$ 虛線外框並標「遮罩：看不到」，測試模式 $a_t$ 畫成虛線空位並標「尚未產生」。
  - 窄螢幕：≥ 600px 單列橫排；< 600px 改為每個時間步一列（每列 3 個 token 橫排，列左標「t=6」並以括號標時間步嵌入），token 圖形最小 28px、標籤 12px 以上，不縮小文字。
  - 讀數：「輸入 token 數」（訓練 3m／測試 3m−1）、「s<sub>t</sub> 看到的 token 數」（3m−1）、「RTG<sub>1</sub>」；測試模式另顯示「結束時剩餘 RTG<sub>T+1</sub>」與「與訓練 RTG 的差」。
- **數值防護**：$t,K$ 夾在 $[1,T]$；$G$ 為整數；無除法。
- **讀者應觀察到**：稀疏序列的 RTG 每步都是 1（最後一步的獎勵立即反映到所有時間步）；測試模式 $G$ 每增加 1，整列 RTG 都加 1；$K=1$ 時只剩 3 個 token（DT with no context，S9 會用到）。
- **無障礙**：形狀＋文字區分模態；表格有 `<caption>`；token 列 SVG 的 `aria-label` 例「訓練模式，視窗含時間步 6 到 8，共 9 個 token；s 8 看到 8 個」。

### C2 · Bellman 備份的傳遞速度 vs returns-to-go（S2）

- **教的概念**：同步表格備份每次只把獎勵往前推一步（$\alpha=1$）或更慢（$\alpha<1$，約 $1/\alpha$ 倍），並乘上折扣；RTG 一次後綴和就給每步完整報酬（LO2、LO3）。〔本頁推導〕的示意（單一動作的策略評估），不是論文實驗。
- **輸入**：滑桿 $H\in\{5,\dots,100\}$（預設 30）；滑桿 $\gamma\in[0.90,1.00]$，步長 0.001（預設 0.99）；滑桿 $\alpha\in[0.05,1]$，步長 0.05（預設 1）；滑桿「備份次數 $n$」$\in\{0,\dots,400\}$（預設 10）；分段按鈕「縱軸」：除以極限（預設）／絕對值；按鈕「逐次播放」（`aria-pressed`，文字在「逐次播放」與「暫停」間切換）。
- **播放與 reduced-motion（P0 修正）**：先算目標 $n^\dagger$＝$n\le400$ 中使「已傳到的狀態數 $=H$」的最小 $n$；不存在時 $n^\dagger=400$，並在讀數區顯示「400 次內未全部傳到（α 越小需要越多次，約 H/α 次）」。播放：$n$ 從目前值每 80ms 加 1，到 $n^\dagger$ 或按「暫停」停止；若目前 $n\ge n^\dagger$，從 0 開始。reduced-motion：直接把 $n$ 設為 $n^\dagger$，不做動畫。$n^\dagger$ 以遞增 $n$ 逐次更新同一個陣列求得（最多 400 次，每次 $O(H)$），不是每個 $n$ 重算。
- **計算**（`chain.py` 相同）：`V = Float64Array(H+2)`（索引 1…H，`V[H+1]=0`）；重複 $n$ 次：`old = V.slice(); for i=1..H: V[i] = (1-α)·old[i] + α·((i===H?1:0) + γ·old[i+1])`。輸入變動時從 0 重算（最多 400×100 次運算）。
  - 「已傳到」：$V_i\ge\tfrac12\gamma^{H-i}(1-10^{-9})$；$10^{-9}$ 的容差避免 $\Pr=0.5$ 剛好相等時的浮點差異（例：$H=30,\alpha=0.5$ 時 $n^\dagger=59$，$\Pr[\mathrm{Bin}(59,0.5)\ge30]=0.5$）。
  - 讀數：$V_1$、極限 $\gamma^{H-1}$、已傳到的狀態數、「已有值的最左狀態」（$V_i\ge\tfrac12\gamma^{H-i}$ 的最小 $i$）、$n^\dagger$。
- **數值防護**：$\gamma=1$ 時極限為 1；極限在 $\gamma=0.9,H=100$ 時小到 $3\times10^{-5}$，所以預設縱軸為**除以極限** $V_i/\gamma^{H-i}\in[0,1]$（極限恆 > 0，不會除以 0）；絕對值模式縱軸固定 [0, 1.05]，長條高度 < 0.5px 時不畫，並在圖上方顯示「部分長條小於 1 像素，請切換為『除以極限』」。$V_1$ 讀數 < $10^{-3}$ 時以 `toPrecision(3)` 的科學記號顯示，不顯示為 0。
- **顯示**：橫軸狀態 $i=1\dots H$（左為起點、右為獎勵所在）；每個狀態一根長條（`--accent` 實心）；除以極限模式在 $y=1$ 畫一條虛線「極限」，絕對值模式每個狀態畫一段細虛線為 $\gamma^{H-i}$（`--ink-3`）；上方另一列小方塊為 RTG（全部 = 1，`--c-train` 空心方塊，標「RTG（一次後綴和）」）；「已有值的最左狀態」以向下小箭頭標出。窄螢幕：$H>50$ 且寬度 < 600px 時長條間距為 0，橫軸只標 1、H/2、H。
- **讀者應觀察到**：$\alpha=1$ 時第 $n$ 次後恰好最右 $n$ 個狀態有值；$H=30,\gamma=0.99$ 在 $n=29$ 時 $V_1=0$、$n=30$ 時 $V_1=0.747$；$\alpha=0.5$ 約要兩倍次數；$\alpha=0.05,H=100$ 在 400 次內只傳到 20 個狀態；切換到絕對值模式並設 $\gamma=0.9,H=100$，左半部長條看不見——折扣報酬的目標本身就小，這與傳遞速度是不同的問題（圖下註明）。
- **無障礙**：長條與極限以實心／虛線＋文字區分；播放中不更新 live region，停止時宣告「n = 30，已傳到 30 個狀態，共 30 個，V 1 = 0.747」。

### C3 · 圖上最短路徑：以報酬為條件的計數模型（S6）

- **教的概念**：只用隨機遊走資料，以「最大的 RTG」為條件就能走出短路徑；路徑可由不同軌跡的片段拼接而成，甚至比資料中從同一起點出發的最好軌跡更短；固定目標並遞減時，路徑長度跟著目標變。前兩點對應 Sec. 1、Fig. 2、App. A.3（LO8）；固定目標模式是〔本頁推導〕的延伸，把 Alg. 1 的遞減規則（LO7）放到這個例子，論文沒有這個實驗（Fig. 4 是 Atari／D4RL 的結果，只作類比）。
- **〔本頁簡化〕與推定**（頁面逐條寫在圖下）：
  1. GPT 換成計數表 $C[s][L][a]$（等於 $K=1$ 的表格模型，參見 S12 的 RCP 與 Sec. 5.3）；不使用 App. A.3 的 prior，改用「每步取資料中該節點出現過的最大 RTG」（Fig. 2 圖說）。
  2. 「edge sparsity coefficient 0.1」解讀為每條有向邊獨立以機率 0.1 存在（推定）。
  3. 隨機遊走在每步從目前節點的出邊中均勻選一條；沒有出邊則留在原地（推定；論文只寫 random walk）。抵達目標即結束。
  4. 在 $T=10$ 步內沒到目標的軌跡，RTG 記為 −∞（與 Fig. 2 中間圖的 −∞ 一致），不進入計數表。
  5. 固定目標模式（本頁推導）：$C[s][L]$ 為空時改用最近的可用 $L'$（同距取較小者）；$C[s]$ 全空視為失敗。路徑可以繞圈（例：起點 6、$L_0=5$ 走 6→1→15→6→12→0），所以「實際步數跟著目標」部分是靠繞圈補足步數。
  6. 拼接比例與論文的 15.8% 不可直接比較（模型、圖與資料都不同）；直方圖橫軸 1–10 與 ∞，Fig. 6 只畫 1–9。
- **輸入**：
  - 按鈕「換一張圖」（seed + 1）與 seed 顯示；預設 seed = 40（推定：選擇所有 19 個非目標節點都能抵達目標、且出現拼接的圖）。
  - 分段按鈕「資料量（軌跡數）」：50 / 100 / 200 / 500 / 1000，預設 1000（論文值）。較小的資料集是同一串流的前綴。
  - 分段按鈕「生成方式」：每步取最大 RTG（預設）／固定目標並遞減。
  - 下拉選單「起點」：節點 1…19，預設 5。
  - 滑桿「目標剩餘步數 $L_0$」（$=-\hat R_1$；只在固定目標模式）：1…10，預設 7。
- **演算法**（`graph.py` 相同，抽樣順序固定）：
  1. `rng = mulberry32(seed)`；節點 0 為目標；`for i in 0..19, for j in 0..19, i≠j: if rng() < 0.1 → 邊 i→j`（出邊依 j 遞增）。
  2. 產生 1000 條軌跡（全部先產生，再依資料量取前綴）：`s = 1 + floor(rng()·19)`；最多 10 步：若 `s==0` 停；出邊非空則 `s = out[s][floor(rng()·out[s].length)]`（沒有出邊不抽亂數、留在原地）；記錄。
  3. 對前 $N_{\text{traj}}$ 條中有到達目標者（第一次到達索引 $g$）：$t<g$ 時 `C[s_t][g−t][s_{t+1}] += 1`。
  4. BFS（反向邊，自目標）得最短距離 `dist`。
  5. 每步取最大 RTG：在 $s$，$L^*=\min\{L: C[s][L]\ne\emptyset\}$；無則失敗（∞）；動作＝$\arg\max_a C[s][L^*][a]$（同數取最小編號）；最多 10 步。
  6. 固定目標：$L=L_0$；在 $s$ 若 $C[s][L]$ 為空，改用可用 $L'$ 中 $|L'-L|$ 最小者（同距取較小的 $L'$）並令 $L=L'$；取 argmax 動作；走一步後 $L\leftarrow L-1$（即 $\hat R\leftarrow\hat R-r$，$r=-1$）；若 $C[s]$ 全空則失敗（∞）；最多 10 步。
  7. 拼接：成功的生成路徑（節點序列，可含重複節點）不是前 $N_{\text{traj}}$ 條訓練軌跡中任何一條的連續子序列（只比對目前資料量的前 $N_{\text{traj}}$ 條，與 `graph.py` 及第 5 節相同；其中沒到達目標的軌跡不含節點 0，比對與否不影響結果）。
- **顯示**：
  - 左圖（節點半徑 11px，欄距 ≥ 56px，列距 ≥ 30px；圖寬 < 560px 時 SVG 固定 560px 寬並放在 `overflow-x:auto` 容器中，路徑表是文字替代）：分層佈局，橫向依 `dist`（0 在最左；無法抵達的節點放最右一欄並標「無法抵達」），同欄依編號由上而下排列；有向邊為細灰線加箭頭；目前生成路徑以 `--accent` 粗線＋序號標出；起點雙圈、目標實心。
  - 右側路徑表：每步 $s$、使用的 $L$（$\hat R=-L$）、動作、該 $(s,L)$ 的計數。
  - 讀數：生成步數、BFS 最短步數、資料中從起點出發的最短剩餘步數 $\min L$、是否拼接。
  - 下方直方圖（Fig. 6 的對照）：橫軸步數 1…10 與 ∞，三組長條：隨機遊走（資料中各軌跡到達目標的步數，`--ink-3` 斜線）、最短路徑（19 個起點的 BFS 距離，`--c-train` 空心）、本頁模型（19 個起點的生成步數，`--accent` 實心）；縱軸為比例。下方一行摘要：「19 個起點中成功 a、最短 b、拼接 c（c/a）」。
  - 固定目標模式另顯示小圖：橫軸 $L_0=1\dots10$、縱軸實際步數（目前起點各跑一次），附 $y=x$ 虛線「oracle」（對照 Fig. 4）。
- **數值防護**：無可抵達節點的 seed（例如 20、28）時摘要顯示「此圖沒有節點能抵達目標，請換一張圖」，直方圖照畫；`C` 用 `Map`，鍵 `s*16+L`。
- **讀者應觀察到**（seed 40）：資料 1000 條時 19 個起點全部走出最短路徑，其中 2 條是拼接；起點 5 的資料最短需 8 步，生成路徑 7 步（等於 BFS 最短）；資料 200 條時拼接 5 條、最短 16 條；固定目標模式起點 6 的實際步數隨 $L_0$ 增加（2, 2, 3, 3, 5, 6, 7, 8, 9, 10）。
- **無障礙**：路徑以粗線＋序號、節點以形狀區分；直方圖三組以填充樣式＋圖例文字區分；所有操作都有按鈕或選單，不需點圖。路徑表即為圖的文字替代。

### C4 · 分數正規化（C4a，S7）與結果表（C4b，S11）

- **教的概念**：C4a——Atari 正規化公式（LO9）；C4b——只用來查閱 Table 1–7 的完整數字（LO9–LO12 的讀表），本身不教新概念；頁面在 C4b 圖版標題下寫明這一點。
- **C4a 輸入與計算**：下拉「遊戲」（預設 Breakout）＋數字輸入「原始分數」（預設 76.9）。輸出 $100(\text{raw}-\text{random})/(\text{gamer}-\text{random})$（一位小數），並寫出代入後的算式（例「100 × (76.9 − 2) ÷ (30 − 2) = 267.5」）。Table 11：Breakout 2／30、Qbert 164／13455、Pong −21／15、Seaquest 68／42055（random／gamer）。非數字輸入顯示「請輸入數字」。分母恆 > 0。
- **C4b 輸入**：分段按鈕「表格」：Atari（Table 1）／D4RL（Table 2）／%BC D4RL（Table 3）／%BC Atari（Table 4）／脈絡長度（Table 5）／Key-to-Door（Table 6）／延遲獎勵（Table 7）；預設 D4RL。勾選框「以長條圖顯示」（預設未勾：表格）。勾選框「顯示重算檢查」（預設**未勾**；勾選後顯示重算平均與由原始分數重新正規化的值）。
- **資料**：照錄第 1.4 節步驟 7–8 與原文表格（建置時逐格對照 `paper.txt` 第 353–620、1129–1161 行）。Table 12 原始分數：Breakout 76.9／61.1／6.8／4.5／40.9；Qbert 2215.8／14012.0／156.0／160.1／2464.1；Pong 17.1／19.3／−14.5／−20.8／9.7；Seaquest 1129.3／779.4／250.1／370.5／968.6（DT／CQL／QR-DQN／REM／BC）。Table 13（10／25／40%BC）：Breakout 10.0／22.6／32.3；Qbert 1045／2302.5／1674.1；Pong −20.3／−16.2／5.2；Seaquest 521.3／549.3／758。
- **計算**：每列最佳值加粗（論文的粗體規則：最佳平均）。「顯示重算檢查」開啟時：平均列旁加一欄重算值（一位小數）；Table 1／4 每格旁顯示由 Table 12／13 正規化的值；差 > 0.15 的格加文字標記「重算不同」（標記是按鈕，點擊或 Enter 展開一行說明，觸控可用）。說明一行：「論文數字照錄；重算差異的列表見頁尾註」。
- **長條圖的座標**：每一列（環境或遊戲）各自縮放，縱軸範圍 $[\min(0,v_{\min}),\ \max(0,v_{\max})]$ 再上下各留 10%；含負值時畫 $y=0$ 基線並標「0」；數值標在長條末端。Table 6 的縱軸固定 0–100%。原始分數（Table 12／13）只以表格呈現，不畫長條（尺度從 17 到 14012）。
- **數值防護**：Table 2 Reacher 列的 BEAR／BRAC-v／AWR 為「—」，平均只在完整欄位計算（與論文相同）；「—」在長條圖中不畫並標「無資料」。
- **顯示**：表格（含 `<caption>`，四欄以上包在橫捲容器）；長條圖模式 DT `--accent` 實心、其他方法 `--surface-2`＋`--ink` 外框並各自標方法名。
- **讀者應觀察到**：C4a 預設 267.5；C4b 開啟重算檢查後：D4RL 平均重算 DT 74.7、CQL 63.9，BC 不含 Reacher 47.7、全部 46.4（與論文兩格對調）；Table 3 DT 平均重算 54.9（論文 56.1）。
- **無障礙**：表格為主要呈現，長條圖為輔；標記用文字而非只用顏色。

---

## 5. 驗收數值表

全部由 `verify/` 腳本產生，與頁面算法相同。容許誤差：整數與表格照錄值必須完全相同；浮點讀數 $|x-x_{ref}|\le10^{-6}$（顯示到 3 位小數者以顯示值比對）。

**C1**（`c1_out.txt`）

| 序列 | 訓練 RTG $\hat R_1\dots\hat R_T$ | 測試 $G$ | 測試 $\hat R_1\dots\hat R_T$ | 剩餘 $\hat R_{T+1}$ | 差 |
| --- | --- | --- | --- | --- | --- |
| 稀疏 | 1,1,1,1,1,1,1,1 | 6 | 6,6,6,6,6,6,6,6 | 5 | 5 |
| 密集 | 10,9,9,7,6,6,3,2 | 8 | 8,7,7,5,4,4,1,0 | −2 | −2 |
| 最短路徑 | −4,−3,−2,−1,0 | 1 | 1,2,3,4,5 | 5 | 5 |

視窗（訓練模式的輸入／看到；測試模式輸入＝看到＝3m−1）：$(t,K)=(1,3)$：$m=1$、3 token、看到 2；$(5,3)$：起點 3、9／8；$(8,8)$：24／23；$(8,1)$：3／2；$(4,20)$：起點 1、12／11。

**C2**（`chain_out.txt`；迭代與閉式解最大差 $1.1\times10^{-14}$）

| H | γ | α | n | V₁ | V_{H/2} | 已傳到 | γ^(H−1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 0.99 | 1 | 10 | 0 | 0 | 10 | 0.747172 |
| 30 | 0.99 | 1 | 29 | 0 | 0.860058 | 29 | 0.747172 |
| 30 | 0.99 | 1 | 30 | 0.747172 | 0.860058 | 30 | 0.747172 |
| 30 | 0.99 | 0.5 | 30 | 6.96e-10 | 0.367905 | 15 | 0.747172 |
| 30 | 0.99 | 0.5 | 60 | 0.411908 | 0.860001 | 30 | 0.747172 |
| 30 | 0.99 | 0.1 | 100 | 1.83e-8 | — | 10 | 0.747172 |
| 30 | 0.99 | 0.1 | 297 | 0.377397 | — | 30 | 0.747172 |
| 100 | 0.99 | 1 | 100 | 0.369730 | 0.605006 | 100 | 0.369730 |
| 100 | 0.9 | 1 | 100 | 3.0e-5 | 0.005154 | 100 | 3.0e-5 |

（$V_{H/2}$ 指 $i=H/2$；「已傳到」＝ $V_i\ge\frac12\gamma^{H-i}$ 的個數。）$H=30,\gamma=1$：$V_1$ 達 0.5 所需 $n$：$\alpha=1,0.5,0.25,0.1$ → 30、59、119、297。

播放／reduced-motion 目標 $n^\dagger$（$\gamma=0.99$；門檻含 $1-10^{-9}$ 容差）：$H=30$：$\alpha=1$ → 30；0.5 → 59；0.05 → 400（未全部傳到，已傳到 20）。$H=100$：$\alpha=1$ → 100；0.5 → 199；0.25 → 399；0.05 → 400（未全部傳到，已傳到 20）。除以極限模式：$H=100,\gamma=0.9,\alpha=1,n=50$ 時 $i=51$ 為 1.0、$i=50$ 為 0。

**C3**（`graph_out.txt`，seed 40）

- 邊數 47；`dist`（節點 0…19）= 0, 4, 3, 1, 3, 7, 2, 5, 1, 4, 4, 6, 1, 5, 1, 3, 2, 6, 6, 5。
- 資料 1000：成功 19、最短 19、拼接 2（起點 5、17）；隨機遊走成功率 0.156；起點 5：資料 $\min L=8$，生成路徑 5→17→7→1→15→6→12→0（7 步）。直方圖（1…10、∞）：隨機遊走 0.059, 0.020, 0.008, 0.013, 0.004, 0.008, 0.012, 0.008, 0.014, 0.010, 0.844；最短與本頁模型皆為 4/19, 2/19, 3/19, 3/19, 3/19, 3/19, 1/19, 0, 0, 0, 0。
- 資料 200：成功 19、最短 16、拼接 5（起點 1、5、7、17、18）；起點 5 生成 8 步（資料 $\min L=9$）；起點 10 生成 6 步（最短 4）。
- 資料 50 / 100 / 500：成功 10 / 16 / 19，最短 10 / 14 / 18，拼接 1 / 5 / 2。
- 固定目標（資料 1000）起點 6，$L_0=1\dots10$ → 2, 2, 3, 3, 5, 6, 7, 8, 9, 10；起點 15 → 3, 3, 3, 4, 4, 6, 7, 8, 9, 10。
- 其他 seed 的摘要（資料 1000）：seed 1 → 可抵達 19、成功 19、拼接 0；seed 24 → 19、19、2；seed 20 → 可抵達 0。

**C4 與內文算術**（`tables_out.txt`）

- 正規化 Table 12 → Table 1：Breakout 267.5／211.07／17.14／8.93／138.93；Qbert 15.44／104.19／−0.06／−0.03／17.31；Pong 105.83／111.94／18.06／0.56／85.28；Seaquest 2.53／1.69／0.43／0.72／2.14。差 > 0.15：Pong DT（105.8 vs 106.1）。
- Table 13 → Table 4：差 > 0.15：Qbert 40%BC（11.36 vs 11.8）、Pong 10%BC（1.94 vs 2.5）。
- Table 2 重算：不含 Reacher DT 74.733、CQL 63.911、BEAR 48.244、BRAC-v 36.889、AWR 34.278、BC 47.689；全部 DT 69.242、CQL 54.233、BC 46.400。
- Table 3 重算：DT 54.913、10%BC 56.688、25%BC 52.688、40%BC 49.425、100%BC 39.612、CQL 43.513。
- Table 7：DT 延遲 − 原始 = −0.3、−6.9、−4.2；CQL −102.0、−52.8、−46.6。50 萬／5,000 萬 = 1%。

---

## 6. 實作計畫

### 6.1 檔案與共用外殼

- 單一檔案 `tut/571/build/index.html`；CSS、JS、時間線資料全部內嵌，不 `fetch` 本機檔案。上線時複製為 `tutorials/decision-transformer.html`。
- **直接沿用** `tut/406/build/index.html`（最新建置頁，源自 `tutorials/prediction-targets-k-diff.html`）的外殼：CSS tokens 與深色模式三段覆寫（`pm-theme`）、字型載入、appbar、`#pm-bar`、章節目錄（`initToc`）、`.plate`／`.plate-h`／`.plate-lead`／`.plate-f`、callout、`<details>` 檢查題、「本頁推導」與「推定」pill、「作者認為」引述樣式、表格橫捲容器、KaTeX 0.16.9 設定與 `renderMath()`、`?test` 的 `selfTest()` 框架（含 `scrollWidth <= innerWidth`）、時間線卡片列（`renderTimeline`）、SVG 輔助（`el`、`lp`、`arrow`、`frame`、`chartW`、`ro`）、`sched`、`announce`、`radiogroup`。
- 截圖時 KaTeX 改指向本機 `scratchpad/katex/0.16.9`（headless Chrome 無法連 CDN）；上線版維持 cdnjs。
- 以下只列新增或變更的部分。

### 6.2 新增 tokens、樣式與連結

- 圖表色：`--accent` = DT／TD 備份值／生成路徑；新增 `--c-train`（light #3F6E8C／dark #8DB4D3，沿用 406 頁）用於 RTG 與最短路徑系列。建置時確認圖形對比 ≥ 3:1、文字 ≥ 4.5:1。
- 新增「本頁簡化」pill（與「本頁推導」同樣式）。
- `hatR(x,y,idx)`（SVG 的 R̂ 帽號＋下標，見 4.0）；斜線等 pattern 的線色以 `stroke="var(--ink-3)"` 設定，深色模式自動切換。
- 顯示式：沿用 `.katex-display{overflow-x:auto}`；Eq. 2 在 < 600px 時以兩個 `$$…$$` 手動斷行（$\hat R_1,s_1,a_1,$ ／ $\hat R_2,\dots,a_T$）；C2 閉式解只在 `<details>` 內。
- 「本章符號」：每章末一個可收合 `<details>`，內容取自 1.4 符號表該章的列。
- token 形狀元件：`tokenGlyph(kind, label)` 回傳 SVG 群組（RTG 菱形、狀態圓形、動作方形），hero、S3、S4、C1 共用。
- 站內連結：`../index.html#paper-571`；arXiv `https://arxiv.org/abs/2106.01345`；程式碼 `https://sites.google.com/berkeley.edu/decision-transformer`；筆記 `<slug>.html`。

### 6.3 程式結構（單一 IIFE）

- C1：`initC1()`，`state={preset,mode,t,K,G}`；`rtg(r)`、`testRtg(r,G)`、`win(t,K)` → `{a,m,total,seen}`。
- C2：`initC2()`，`state={H,g,a,n,yMode,playing}`；`chainV(H,g,a,n)` → `Float64Array`；`reached(V,H,g)`（含 $1-10^{-9}$ 容差）；`nDagger(H,g,a)` → `{n, all}`（遞增更新、上限 400）。
- C3：`initC3()`，`state={seed,ntraj,mode,start,L0}`；`buildWorld(seed)` → `{out, trajs(1000), dist}`（快取 seed）；`countTable(trajs, ntraj)` → `Map`（快取 `seed|ntraj`）；`genGreedy(C,s)`、`genTarget(C,s,L0)`、`isStitched(path, trajs, ntraj)`；`hist(values)`。
- C4：`initC4a()`（S7，`normAtari(game, raw)`）與 `initC4b()`（S11），`TABLES` 常數（每表 `{id, caption, cols, rows, avg?:[{label, paper:[…], rows:filter}]}`）、`renderTable()`／`renderBars()`（每列獨立縮放，範圍含 0）。
- `selfTest()` 新增斷言（第 5 節容許誤差）：
  - C1：`rtg([1,0,2,1,0,3,1,2])` = `[10,9,9,7,6,6,3,2]`；`testRtg(path,1)` 末項 5；`win(5,3,'train')` = `{a:3,m:3,total:9,seen:8}`、`win(5,3,'test')` = `{a:3,m:3,total:8,seen:8}`。
  - C2：`chainV(30,.99,1,29)[1]===0`、`chainV(30,.99,1,30)[1]` = 0.747172；`chainV(30,.99,.5,60)[1]` = 0.411908；並對 $(H,\gamma,\alpha,n)=(50,0.95,0.3,200)$ 比對閉式 $\gamma^{H-i}\Pr[\mathrm{Bin}(n,\alpha)\ge H-i+1]$（pmf 遞推，誤差 < 1e-12）；`nDagger(30,.99,.5)` = `{n:59, all:true}`、`nDagger(100,.99,.05)` = `{n:400, all:false}` 且已傳到 20。
  - C3：seed 40 的 `dist` 陣列；資料 1000 拼接 2、資料 200 拼接 5 與最短 16；起點 5 的生成路徑；起點 6 固定目標序列。
  - C4：`normAtari('Breakout',76.9)` = 267.5；Table 2 DT 不含 Reacher 平均 74.7；Table 3 DT 重算 54.9。

### 6.4 研究筆記與時間線

- 來源 `tut/571/notes/timeline.json`（12 筆，欄位 `year`、`title`、`authors_short`、`arxiv`、`role`、`source`），原樣貼進 `<script type="application/json" id="timeline-data">`；排序用 `Number(year)`。
- `TL_GROUP`（以 arXiv ID 為 key；不在表中歸「其他」，並在建置回報列出）：

  | 類別 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | RL 基礎 | 1312.5602 | `rl-mdp-q-learning-basics.html` |
  | 序列模型 | 1706.03762、2005.14165 | `dt-transformer-lm-background.html` |
  | 離線 TD 方法與資料 | 1812.02900、1906.00949、2006.04779、2004.07219 | `dt-offline-rl-baselines-datasets.html` |
  | 報酬條件監督學習 | 1912.02877、1912.13465 | `return-conditioned-sequence-modeling.html` |
  | 信用分配 | 1806.07857 | `dt-credit-assignment-lineage.html` |
  | RL 中的 Transformer | 1910.06764 | `dt-transformers-in-rl-and-trajectory-transformer.html` |
  | 本文 | 2106.01345 | `dt-method-and-results.html` |

- S15 延伸閱讀（依總覽 `dt-lineage-overview` 的閱讀順序）：`dt-lineage-overview`、`rl-mdp-q-learning-basics`、`offline-rl-distribution-shift`、`dt-transformer-lm-background`、`dt-offline-rl-baselines-datasets`、`return-conditioned-sequence-modeling`、`dt-credit-assignment-lineage`、`dt-transformers-in-rl-and-trajectory-transformer`、`dt-method-and-results`。每項附 frontmatter 的 `title`（建置時從檔案讀取，不手抄）。

### 6.5 文案規則（新增部分）

- 用詞：「returns-to-go（RTG，從當下到結尾的報酬總和）」「報酬」（return）與「獎勵」（reward）嚴格區分；「Bellman 備份」「信用分配（credit assignment）」「行為複製（behavior cloning，BC）」「拼接（stitching）」「常見錯誤」。第一次出現的專有名詞附英文。
- 三種陳述分開標示：論文結果（附出處）、「作者認為／作者推測」、〔本頁推導〕／〔本頁簡化〕；推定值加「推定」pill。
- 第 1 節以外不得新增論文數字；頁面數字須與第 1、5 節一致。

### 6.6 建置後自我檢查

依 406 頁流程：390／768／1024／1440 截圖＋深色、鍵盤 focus、reduced-motion；逐項評分（字體、留白、視覺層級、色彩、動態、微互動、響應式、原創性）並修正；`?test` 全數通過；逐條核對 12 個學習目標的答案出現在指定章節內文。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 論文平均數字與各列不符 | Table 2 BC 的兩個平均對調（重算不含 Reacher 47.7、全部 46.4；論文 46.4、47.7）；Table 3 DT 平均重算 54.9，論文 56.1。 | 內文照錄論文值；C4 並列重算值並標「重算不同」；LO 答案只用論文值與未受影響的數字。 |
| R2 | 正規化與原始分數不符 | 由 Table 12／13 重算：Pong DT 105.8（論文 106.1）、Qbert 40%BC 11.4（11.8）、Pong 10%BC 1.9（2.5）。 | C4「由原始分數重新正規化」顯示差異；不推測原因。 |
| R3 | 同一數字在兩表不同 | Qbert DT：Table 1 為 15.4、Table 5 為 15.1。Hopper Medium-Expert 的 BC：Table 2 為 79.6、Table 7 為 59.9（與 Table 2 HalfCheetah 的 BC 值相同）。 | S9／S11 引用時各標表號；C4b 照錄；在 S11 加一句註記。 |
| R4 | ± 的意義 | Sec. 4.1 內文寫 mean and standard deviation of 3 seeds；Table 1、2、12、13 的說明寫 mean and variance。 | 頁面照錄兩處說法並標「± 的意義未定」。 |
| R5 | C3 的推定與簡化 | sparsity 0.1 的解讀、隨機遊走選邊方式、抵達即停、計數表取代 GPT、不使用 prior、seed 40 的挑選。 | 圖下逐條列出（C3「本頁簡化與推定」）；拼接比例不與 15.8% 比較。 |
| R6 | App. A.3 的文字 | 原文「we do assume knowledge of the optimal path length upfront」與同段「avoids the need for any external or oracle information like the optimal path length」矛盾，前句可能漏了 not。 | 頁面只引用後句，並在設計文件記錄此疑點；不在頁面上推測。 |
| R7 | C2 是示意 | 論文沒有鏈實驗；實際 TD 方法用函數近似、取樣更新與 max 運算。 | C2 標「本頁推導」；圖下寫明限制；內文以「作者認為」引述 Sec. 1 的論點。 |
| R8 | 未給數字的圖 | Fig. 3（各領域平均）、Fig. 4、Fig. 5、Fig. 6 只有圖。 | 只做文字描述，不讀圖、不重繪曲線。 |
| R9 | 時間索引 | Sec. 2.1 軌跡從 $t=0$ 起、期望報酬從 $t=1$ 起；Eq. 2 從 1 起。 | 頁面統一從 1 起並在符號表註明。 |
| R10 | 共用筆記 | `rl-mdp-q-learning-basics`、`offline-rl-distribution-shift` 位於 `tut/568/notes/`，上線前須確認已發布。 | 建置時檢查兩個 slug；若尚未上線，連結保留並在建置回報列出。 |
| R11 | 初學者門檻 | 讀者不熟 RL 與 TD；S2 在介紹 DT 前先講 TD。 | S1 不寫 Bellman 式；S2 先給名詞的字面定義，內文只講 α=1，閉式解放 `<details>`；符號依章在章末「本章符號」列出。 |
| R12 | Table 9 warmup | txt 為「105」，PDF 為 $10^5$；與訓練總步數 $10^5$ 相同（整段訓練都在 warmup）。 | 照錄 $10^5$，不評論。 |
| R13 | 學習目標數 | 審查建議把 LO11、LO12 拆成更多題。 | 不拆：任務規定 10–12 個；LO11 縮成 (a)(b) 兩小題，LO12 的答案分在 S10、S11 兩章內文。 |
| R14 | Alg. 1 與 Sec. 3 的預測位置 | Sec. 3 說 $s_t$ 位置預測 $a_t$；Alg. 1 註解寫 action prediction tokens。 | 頁面依 Sec. 3，S5 註記一次。 |

---

## 8. 修訂紀錄（round 1）

逐條對應 `reviews/round1.md`：
1. C2 播放／reduced-motion 目標 $n^\dagger$ 與 400 次後備規則、門檻容差；`chain.py` 增加 `jump_target`；第 5 節與 selfTest 加數值。
2. prior 的 $k$＝路徑步數（推定）；prior 式移入 `<details>`；LO8 改問條件與拼接。
3. C3 固定目標模式標本頁推導；後備、繞圈、$C[s]$ 全空＝失敗、拼接比對範圍寫入簡化清單與演算法。
4. Table 1 對 BC 的 Qbert 例外。 5. 預測 RTG 的例外（步驟 5、LO6、常見錯誤 3、S5、S6、S10）。 6. S10 標題。 7. C1 測試模式 3m−1 與「尚未產生」。
8. Alg. 1 註解不一致（步驟 5、R14）。 9. ± 的兩種說法（步驟 7、R4）。 10. Table 3 列組合。 11. $V_1>0$ 但極小、γ=1 的條件。 12. 折扣是不同目標（步驟 2、C2）。
13. S2 改寫為策略評估式，閉式解進 `<details>`，內文講 α=1 與約 1/α 倍。 14. S3 前提。 15. S1「本頁說明」與原文。 16. Fig. 6 軸 1–9。
17. C2 縱軸「除以極限」預設。 18. C1 < 600px 每時間步一列、token ≥ 28px。 19. 動態文字規則與 `hatR`。 20. 符號表依章分組並補缺漏、每章「本章符號」。
21. S2 名詞定義（bootstrapping、deadly triad、CQL、價值悲觀），BC 在 S7 定義。 22. C4b 每列獨立縮放、含負值、原始分數只用表格。 23. C4 拆為 C4a（S7）與 C4b（S11），重算檢查預設關閉。
24. S10 拆成 S10／S11，其後章節順延（related S12、impl S13、history S14、next S15）；LO 不增加（R13）。 25. C3 窄螢幕最小 560px 橫捲與節點尺寸。
26. S4 位置嵌入兩句、Eq. 1 逐項、6×6 遮罩矩陣圖；Eq. 2 窄螢幕斷行。 27. S2 標題、「已有值的最左狀態」、獎勵／報酬定義、播放鈕 `aria-pressed`、pattern 用 CSS 變數、Alg. 1 橫捲。

## 9. 修訂紀錄（readiness）

- 阻擋 1：S3 的常數差改用文字，不用 $G$。阻擋 2：Hero 的 token 列改用文字標籤、事實卡 3 不再出現 CQL／D4RL／Reacher 名稱。阻擋 3：拼接判定明寫「前 $N_{\text{traj}}$ 條」。
- 非阻擋：S8 標題改為「12 個設定中 7 個分數最高」（逐列最高者 DT 7、BRAC-v 3、CQL 2，見 `tables_out.txt`）；S11 標題改寫為實際降幅；LO9 逐款描述；C1 註記補 Pong 50、Reacher 5；S2 閉式解先定義 Bin 與 Pr；遮罩矩陣寫明 $t=2$。timeline.json 是資料檔，不是筆記，不加 frontmatter。符號重用（Eq. 1 的 $n$ 與 C2 的 $n$）：頁面在 S4 把 Eq. 1 的 token 數寫成「$N$ 個 token」，避免與 C2 的 $n$ 混淆。

## 10. 建置紀錄（build round）

- 外殼取自 `tut/406/build/index.html`（k-diff 頁同源）的 CSS、appbar、TOC、plate、details、KaTeX、selfTest、時間線；新增樣式只在檔尾 `No. 571 additions`。
- selfTest 54 項全過（390 與 1440 寬度，含無水平捲動）；impeccable 0 項，15 則 numbered-section-labels advisory 保留（DESIGN.md 的目錄式編號）。
- 與設計的差異：C2 圖頂部依序排 RTG 方塊列、「已有值的最左狀態」標記、縱軸標籤（避免 390px 重疊）；C4b 長條圖改為每列一張水平長條圖（每列各自縮放、含 0 基線），比垂直分組長條在 390px 可讀；Eq. 2 窄螢幕以兩個顯示式斷行（`.eq2-w`／`.eq2-n`）。
