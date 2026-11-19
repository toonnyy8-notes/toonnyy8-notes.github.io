# 設計文件：〈Titans: Learning to Memorize at Test Time〉互動教學

- 論文：Ali Behrouz、Peilin Zhong、Vahab Mirrokni（Google Research），arXiv 2501.00663**v1**（2024-12-31）。資料庫編號 No. 243。
- 原文：`tut/243/src/paper.txt`（PDF 同目錄；`tut/243` = `scripts/tutorial_src/243`）。本文件中的「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」一律指論文 v1 的編號。
- 產出：`tut/243/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/titans-test-time-memory.html`（page slug `titans-test-time-memory`），所有站內相對連結以上線位置為準。
- 驗算腳本（不是頁面程式）：`tut/243/verify/`
  - `prng.py`：mulberry32 + Box–Muller，已用 node 確認與 JS 位元一致（seed 7 前三個值 0.0117047532、0.0619582576、0.9769076328）。
  - `widgets.js`：C1–C4 的**參考實作**，頁面直接複製其中的函式；`cd verify && node widgets.js > accept_node.txt` 重新產生驗收輸出（含 C1 步長對照、多 seed 摘要與極端值區塊），第 5 節驗收表以它為準。
  - `c1_update.py`、`c2_deep.py`、`c3_chunk.py`：同一演算法的 Python 版，與 JS 輸出逐項比對，小數點後 4 位一致。執行方式：`cd verify && PYTHONPATH=. python3 c1_update.py`（三個檔案都要 `PYTHONPATH` 指向 `verify/`，否則 `import prng` 失敗）。
  - `tables.py`：論文表格的內部一致性檢查（1.4 步驟 11）。
- 研究筆記：`tut/243/notes/` 已產生（1 hub＋7 topic＋`timeline.json`），第 3 節與 6.6 已依實際 slug 填入。
- 標記慣例：本設計自行推導、論文沒有寫出的結論標「〔本頁推導〕」；為了讓玩具元件可以執行、由本頁選定（論文沒有給）的數值標「推定」。頁面使用同樣的標籤（6.5）。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

Transformer 的注意力精確，但成本是上下文長度的平方，只能處理固定長度的視窗；線性 Transformer 與現代線性 RNN 把歷史壓縮進固定大小的向量或矩陣，速度快，但很長的上下文放不進小狀態（Sec. 1）。作者從「記憶」的角度重新看這些模型：每個模型都有一個寫入（更新）步驟和一個讀取步驟（Eq. 6–7）。本文提出**神經長期記憶模組**（neural long-term memory module，論文稱 LMM）：記憶本身是一個多層感知器（MLP）$\mathcal M$，在**測試時**以梯度下降學習把 key 對應到 value，損失為 $\|\mathcal M_{t-1}(k_t)-v_t\|_2^2$（Eq. 12）。這個梯度被作者稱為**驚訝度**（surprise）；更新規則加上動量（過去的驚訝度，Eq. 9–10）與遺忘閘 $\alpha_t$（等同 weight decay，Eq. 13–14）。作者把序列切成 chunk，chunk 內的梯度都在 chunk 起點的記憶上計算，使訓練可以用矩陣乘法平行化（Eq. 16–18）。以這個記憶模組加上注意力（短期記憶）與可學習但不依賴輸入的**持久記憶**（persistent memory，Eq. 19），作者組出 **Titans** 的三種變體：記憶作為上下文（MAC，Eq. 21–25）、記憶作為閘控分支（MAG，Eq. 26–28）、記憶作為一層（MAL，Eq. 29–31）。在 170M／340M／400M／760M 參數的語言模型（Table 1 列出 340M／400M／760M）、S-NIAH、BABILong、時間序列與 DNA 任務上，作者報告 Titans 與單獨的 LMM 在多數指標上優於同規模的 Transformer++ 與現代線性遞迴模型（Table 1–5；表格中有不支持這句話的格子，例如 760M 的 Gated DeltaNet-H2 Wiki ppl 19.88 低於 Titans (MAC) 的 19.93，完整清單見步驟 11），作者並聲稱可擴展到超過 2M 的上下文視窗（Abstract、Sec. 6）。

### 1.2 研究動機（Sec. 1）

1. **效率與表達力的矛盾**：線性模型是為了長上下文才用的，但長上下文又無法妥善壓縮進小的向量或矩陣狀態（Sec. 1 第 2 段）。
2. **記憶觀點**：RNN 以向量為記憶、先壓縮再寫入；Transformer 把 key／value 直接附加到記憶、不壓縮；線性 Transformer 壓縮成矩陣（Sec. 1「Memory Perspective」）。作者據此提出五個問題：
   - Q1 記憶要用什麼結構？Q2 適當的更新機制是什麼？Q3 適當的讀取方式是什麼？
   - Q4 如何設計一個包含多個互相連結記憶模組的高效架構？Q5 是否需要**深層**記憶模組才能有效儲存久遠的過去？
3. 作者引用神經心理學文獻作為設計靈感：違反預期（令人驚訝）的事件比較容易被記住（Mandler 2014）；記憶由短期、工作、長期等多個系統組成（Cowan 2008）。頁面把這些寫成「作者的動機」，不當作頁面自己的比喻（6.7）。

### 1.3 貢獻（Sec. 1「Contributions and Roadmap」）

1. 神經長期記憶（Sec. 3.1）：以驚訝度（梯度）＋動量＋遺忘閘在測試時學習記憶；作者指出此機制等價於以 mini-batch 梯度下降、動量、weight decay 最佳化一個 meta 神經網路。
2. 平行化訓練演算法（Sec. 3.2）：延伸 TTT（Yu Sun et al. 2024）的 mini-batch 張量化，把 weight decay 與動量也寫成矩陣乘法與線性遞迴。
3. 持久記憶（Sec. 3.3）與 Titans 三種架構 MAC／MAG／MAL（Sec. 4）。
4. 實驗（Sec. 5）：語言模型、常識推理、needle-in-haystack、BABILong、時間序列、DNA；深度記憶的效果與吞吐量；消融。
5. 理論宣稱（Theorem 4.1）：Titans 能解 TC⁰ 以外的問題；v1 正文沒有附證明（見 R7）。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S1 末尾以可收合表格提供；與論文的差異在右欄註明）

| 符號 | 意義 |
| --- | --- |
| $x\in\mathbb R^{N\times d_{in}}$、$x_t$ | 輸入序列（$N$ 個 token）、第 $t$ 個 token（列向量）。本文 $t$ 是 token 索引，不是擴散時間 |
| $W_Q,W_K,W_V$ | 投影矩陣；$q_t=x_tW_Q$、$k_t=x_tW_K$、$v_t=x_tW_V$（Eq. 11、15） |
| $\mathcal M_t$ | 處理完第 $t$ 個 token 後的長期記憶（一個神經網路的參數狀態）。論文同時用 $\mathcal M$ 與 $M$（Eq. 10、14 寫 $M_{t-1}$），也用 M 代表注意力遮罩；頁面記憶一律寫 $\mathcal M$，遮罩寫「mask」 |
| $\mathcal M(\cdot)$、$\mathcal M^*(\cdot)$ | 前向傳遞「會」／「不會」調整權重（Sec. 2）；讀取用 $\mathcal M^*$ |
| $\ell(\mathcal M_{t-1};x_t)$ | 聯想記憶損失 $\|\mathcal M_{t-1}(k_t)-v_t\|_2^2$（Eq. 12） |
| $S_t$ | 驚訝度（動量項）；$\eta_t$ 驚訝度衰減、$\theta_t$ 當下驚訝度的權重（即學習率） |
| $\alpha_t\in[0,1]$ | 遺忘閘（weight decay）。與擴散模型的 α 無關 |
| $L_{\mathcal M}$ | 記憶 MLP 的層數 |
| $b$、$t'$、$\beta_i$ | chunk 大小；$t'$ = 第 $t$ 個 token 所在 chunk 的起點索引，本頁寫 $t'=t-1-\mathrm{mod}(t-1,b)$（論文印 $t-\mathrm{mod}(t,b)$，與其文字不符，見 R1）；$\beta_i=\prod_{j=1}^{i}(1-\alpha_j)$（Eq. 16） |
| $\|A\|_F$ | Frobenius 範數：矩陣所有元素平方和的平方根（C1 讀數用） |
| ppl | perplexity（困惑度）：模型對下一個 token 的平均不確定程度，越低越好 |
| $P=[p_1\ \cdots\ p_{N_p}]$ | 持久記憶（$N_p$ 個可學習向量，Eq. 19） |
| $S^{(i)}$、$C$、$h_t$、$N_l$ | 第 $i$ 個片段、片段長度、從記憶讀出的歷史資訊（Eq. 21）、Fig. 3a 說明中的長期記憶 token 數 |
| SW-Attn、SW-Attn$^*$ | 滑動視窗注意力、帶前綴（prefix）的滑動視窗注意力（Fig. 3b） |
| $\otimes$ | 任意非線性閘控（Sec. 4.2） |

以下依論文順序排列。每一步寫「作者想回答什麼」「做了什麼」「得到什麼」「下一步為什麼需要它」。

#### 步驟 1：用「寫入／讀取」統一序列模型（Sec. 2.1，Eq. 1–7）

- 注意力（Eq. 1–2）：$Q=xW_Q,\ K=xW_K,\ V=xW_V$；$y_i=\sum_{j=1}^{i}\frac{\exp(Q_i^\top K_j/\sqrt{d_{in}})V_j}{\sum_{\ell=1}^{i}\exp(Q_i^\top K_\ell/\sqrt{d_{in}})}$。
- 線性注意力（Eq. 3）把 softmax 換成核函數 $\phi$，$\phi(x,y)=\phi(x)\phi(y)$；核取單位矩陣時（RetNet）可寫成遞迴：
  $$\mathcal M_t=\mathcal M_{t-1}+K_t^\top V_t,\qquad y_t=Q_t\mathcal M_t\tag{Eq. 4, 5}$$
- 一般 RNN：$\mathcal M_t=f(\mathcal M_{t-1},x_t)$（寫入，Eq. 6）、$y_t=g(\mathcal M_t,x_t)$（讀取，Eq. 7）。**原文 Eq. 7 下方一句寫成「f is the read and g is the write」，與方程式旁的標籤相反**；頁面依方程式標籤（f 寫入、g 讀取），並在常見錯誤中註明（R1）。
- 作者的觀察：Eq. 4 是「加法式」寫入，長上下文會使記憶溢出（memory overflow）。既有改進分兩個方向：(1) 加入遺忘閘（GLA、LRU、Griffin、xLSTM、Mamba2）；(2) 改進寫入（Delta rule：寫入前先移除舊值；DeltaNet、Gated DeltaNet）。
- 為什麼需要這一步：後文每個設計（更新規則、記憶結構、讀取方式）都對應 Q1–Q3。頁面 S1 用一張比較表：RNN（向量記憶、壓縮）／線性 Transformer（矩陣記憶、壓縮）／Transformer（成長中的 KV、不壓縮）／LMM（深層 MLP 參數、以梯度寫入）。

#### 步驟 2：把記憶當成線上學習，驚訝度＝梯度（Sec. 3.1，Eq. 8）

- 作者的論點：直接訓練網路記住訓練資料會限制泛化、有隱私問題，且測試資料可能分布外；因此需要一個**在測試時學習如何記住／遺忘**的線上 meta 模型。
- 訓練長期記憶被視為線上學習：把 $x_1,\dots,x_{t-1}$ 壓縮進 $\mathcal M_t$ 的參數。
  $$\mathcal M_t=\mathcal M_{t-1}-\theta_t\underbrace{\nabla\ell(\mathcal M_{t-1};x_t)}_{\text{Surprise}}\tag{Eq. 8}$$
- **用詞釐清（R1）**：原文寫「its gradient with respect to the input」，但 Eq. 8 的 $\nabla\ell(\mathcal M_{t-1};x_t)$ 用來更新 $\mathcal M$ 的參數，必須是**對記憶參數**的梯度。頁面寫：「Eq. 8 的梯度是損失對記憶參數的梯度；論文文字稱之為對輸入的梯度，兩者不同，更新式只能用前者。」梯度越大，表示 $x_t$ 與記憶已存的內容差異越大。
- 作者指出的問題：一次很大的驚訝之後，梯度可能變得很小，使更新停在平坦區域，錯過後續的資訊。這引出步驟 3。

#### 步驟 3：過去的驚訝度——動量（Sec. 3.1，Eq. 9–10）

$$\mathcal M_t=\mathcal M_{t-1}+S_t,\qquad S_t=\eta_t\underbrace{S_{t-1}}_{\text{past}}-\theta_t\underbrace{\nabla\ell(\mathcal M_{t-1};x_t)}_{\text{momentary}}\tag{Eq. 9, 10}$$

- 形式與帶動量的梯度下降相同；$S_t$ 是「跨序列長度的驚訝度記憶」。
- $\eta_t$ 與 $\theta_t$ 都是**依資料而定**（$x_t$ 的函數）：$\eta_t\to0$ 忽略上一次驚訝（例如語境改變），$\eta_t\to1$ 完全納入上一次驚訝（token 與最近的過去高度相關）。
- 論文沒有寫出 $\eta_t,\theta_t,\alpha_t$ 如何由 $x_t$ 計算（R3）；頁面元件用常數，並寫明「論文中三者依 token 而定，在外層迴圈學習」。

#### 步驟 4：要記住什麼——聯想記憶損失與內外兩層迴圈（Sec. 3.1，Eq. 11–12）

$$k_t=x_tW_K,\quad v_t=x_tW_V;\qquad \ell(\mathcal M_{t-1};x_t)=\|\mathcal M_{t-1}(k_t)-v_t\|_2^2\tag{Eq. 11, 12}$$

- **內層迴圈**：最佳化 $\mathcal M$ 的權重（測試時也在進行）。**外層迴圈**：最佳化整個架構的其他參數；$W_K,W_V$ 在內層損失中是超參數（類比 meta-learning，Nichol 2018）。
- 頁面在 S2 用兩欄表列出「內層更新什麼／外層更新什麼」。

#### 步驟 5：遺忘機制（Sec. 3.1，Eq. 13–14）

$$\mathcal M_t=(1-\alpha_t)\mathcal M_{t-1}+S_t,\qquad S_t=\eta_tS_{t-1}-\theta_t\nabla\ell(\mathcal M_{t-1};x_t)\tag{Eq. 13, 14}$$

- $\alpha_t\to0$：更新時不影響過去的內容；$\alpha_t\to1$：清空整個記憶。作者稱此為 weight decay，並說它與現代 RNN 的閘控機制密切相關（Mamba2、LRU）。
- 〔本頁推導〕把 Eq. 13–14 展開到每一個 token，就是 C1 的演算法：$e_t=\mathcal M_{t-1}k_t-v_t$，$g_t=e_tk_t^\top$（線性記憶、損失取 $\frac12\|\cdot\|^2$，與 App. C 相同），$S_t=\eta S_{t-1}-\theta g_t$，$\mathcal M_t=(1-\alpha)\mathcal M_{t-1}+S_t$。當 $\|k_t\|=1$ 時驚訝度大小 $\|g_t\|_F=\|e_t\|$。
- **係數 ½（頁面在 C1 圖版下方寫一行）**：Eq. 12 的損失沒有 ½，梯度是 $2e_tk_t^\top$。C1、C3 取 $\frac12\|\cdot\|^2$，所以 C1 的 θ 等於照 Eq. 12 寫法時的 2θ；例如 C1 的 θ = 0.3 相當於 Eq. 12 的 θ = 0.15。

#### 步驟 6：記憶結構與讀取（Sec. 3.1「Memory Architecture」「Retrieving a Memory」，Eq. 15）

- 記憶 = $L_{\mathcal M}\ge1$ 層的簡單 MLP。作者的理由：以矩陣 $\mathcal M=W\in\mathbb R^{d_{in}\times d_{in}}$ 為記憶等價於最佳化 $\|W_{t-1}k_t-v_t\|_2^2$，這是**線上線性迴歸**，最佳解假設歷史資料的依賴關係是線性的；至少兩層的 MLP 比線性模型更有表達力（Hornik et al. 1989）。Sec. 5.5 以實驗比較 $L_{\mathcal M}=1,2,3,4$。
- 讀取：$q_t=x_tW_Q$，$y_t=\mathcal M^*(q_t)$（Eq. 15），即不更新權重的前向傳遞。
- 論文沒有寫 MLP 的寬度、是否含偏置、啟動函數（Sec. 4.4 的 SiLU 只用於 q／k／v 的計算）；C2 的設定全部是推定（R3）。

#### 步驟 7：平行化訓練（Sec. 3.2，Eq. 16–18，Fig. 1）

- 依序更新需要 $O(N)$ FLOPs，但為了發揮 TPU／GPU，需要張量化並多用矩陣乘法。作者沿用 TTT 的做法：把序列切成大小 $b\ge1$ 的 chunk，**chunk 內所有梯度都在 chunk 起點的記憶 $\mathcal M_{t'}$ 上計算**（mini-batch 梯度下降）。
  $$\mathcal M_t=(1-\alpha_t)\mathcal M_{t-1}-\theta_t\nabla\ell(\mathcal M_{t-1};x_t)=\beta_t\mathcal M_0-\sum_{i=1}^{t}\theta_i\frac{\beta_t}{\beta_i}\nabla\ell(\mathcal M_{t'};x_i)\tag{Eq. 16}$$
  （此式不含動量；$\beta_i=\prod_{j\le i}(1-\alpha_j)$。）
- **$t'$ 的讀法（R1，〔本頁推導〕）**：論文印 $t'=t-\mathrm{mod}(t,b)$，但下一句說「第一個 chunk，$t=b$，所以 $t'=0$」；照公式 $t=b$ 時 $t'=b$，而且 $b=1$ 時 $t'=t$，梯度會在 $\mathcal M_t$（更新後）而非 $\mathcal M_{t-1}$ 上計算。本頁依論文文字把 $t'$ 讀成 chunk 的起點，寫成 $t'=t-1-\mathrm{mod}(t-1,b)$（$t=1,\dots,b$ 時 $t'=0$）。頁面照錄原式，旁註這個差異。
- **第二個等號的範圍（〔本頁推導〕）**：把梯度點換成 $\mathcal M_{t'}$ 之後，第二個等號只在**單一 chunk 內**（$t\le b$，所有 $i\le t$ 共用 $\mathcal M_0$）成立；跨 chunk 時，第 $i$ 項的梯度點是 $i$ 所在 chunk 的起點 $\mathcal M_{i'}$，不是同一個 $\mathcal M_{t'}$。這正是論文只寫第一個 chunk 的原因。在上述 $t'$ 讀法下，$b=1$ 時 $\mathcal M_{i'}=\mathcal M_{i-1}$，與逐 token 更新完全相同。
- 第一個 chunk（$t=b$、$t'=0$）、線性記憶 $\mathcal M_t=W_t$ 時（Eq. 17）：$\nabla\ell(W_0;x_t)=(W_0x_t-x_t)x_t^\top\Rightarrow\sum_{i=1}^{b}\theta_i\frac{\beta_b}{\beta_i}\nabla\ell(W_0;x_i)=\Theta_bB_b(W_0X-X)X^\top$，$\Theta_b=\mathrm{diag}(\theta_1,\dots,\theta_b)$，$B_b=\mathrm{diag}(\beta_b/\beta_1,\dots,\beta_b/\beta_b)$（論文寫「類推定義」，本頁寫出；頁面正文使用一般形式，不出現 $B_b$ 符號）。每個 chunk 只需存自己的 $\Theta,B$。
- 〔本頁推導〕Eq. 17 把 key 與 value 都寫成 $x$，且省略 $\|\cdot\|^2$ 求導的係數 2。一般形式（損失取 $\frac12\|\cdot\|^2$、$K=[k_1\cdots k_b]$、$V=[v_1\cdots v_b]$ 為行向量排成的矩陣）：$\sum_i c_i\nabla\ell(W_0;x_i)=(W_0K-V)\,\mathrm{diag}(c)\,K^\top$。頁面寫一般形式並附這一行說明。
- 動量（Eq. 18）：$S_t=\eta_tS_{t-1}-\theta_tu_t$，$u_t=\nabla\ell(\mathcal M_{t'};x_t)$。所有 $u_t$ 可同時算出，Eq. 18 是以 $u_t$ 為輸入、$\eta_t$ 為轉移值的線性遞迴，可以用平行 associative scan（S5）計算。
- **頁面對 associative scan 的兩句說明（S5 內文）**：「$S_t=\eta_tS_{t-1}+x_t$ 這種線性遞迴的每一步都可寫成一對 $(\eta_t,x_t)$；兩個相鄰步驟可以先合併成一步 $(\eta_2\eta_1,\ \eta_2x_1+x_2)$，而且合併的順序不影響結果。因此可以像比賽淘汰表一樣兩兩合併，$N$ 步只需約 $\log_2N$ 輪平行運算。」〔本頁推導〕
- 〔本頁推導〕含動量與遺忘時（索引一律 1-based，$l,m=1,\dots,n$，$n$ = chunk 內 token 數，常數 $\alpha,\eta,\theta$，約定 $0^0=1$）：
  $$\mathcal M_{\text{end}}=(1-\alpha)^n\mathcal M_{t'}+c_S\,S_{t'}-\theta\,E\,\mathrm{diag}(w)\,K^\top,\qquad S_{\text{end}}=\eta^nS_{t'}+E\,\mathrm{diag}(q)\,K^\top$$
  $E=\mathcal M_{t'}K-V$，$w_l=\sum_{m=l}^{n}(1-\alpha)^{n-m}\eta^{m-l}$，$c_S=\sum_{m=1}^{n}(1-\alpha)^{n-m}\eta^{m}$，$q_l=-\theta\eta^{n-l}$。$\eta=0$ 時（$0^0=1$）$w_l=(1-\alpha)^{n-l}=\beta_n/\beta_l$、$q_l$ 只剩 $l=n$ 一項，回到 Eq. 16–17。`widgets.js` 的 `matmulEnds` 以 0-based 迴圈實作（$l'=l-1$，故程式中為 `eta**(n-1-l')`；JS 的 `0**0===1`）。`c3_chunk.py` 驗證此式與「梯度凍結在 chunk 起點的逐 token 迴圈」差 $<3\times10^{-14}$（η = 0.6 與 η = 0 皆驗證）。
- **數值注意（〔本頁推導〕）**：Eq. 16 的 $\beta_t/\beta_i$ 若照字面先算 $\beta$ 再相除，$\alpha_j=1$ 時 $\beta_i=0$ 會得到 $0/0$，$\alpha$ 接近 1 時連乘會下溢；應直接以 $\prod_{j=i+1}^{t}(1-\alpha_j)$ 計算比值。
- 參數依 chunk 而定（Sec. 3.2 末段）：若 $\alpha,\theta,\eta$ 在 chunk 內為常數，Eq. 17 的 $\Theta$ 只需一個純量，Eq. 18 變成線性非時變系統（LTI），可用全域卷積計算。**實驗中三者是 token 的函數**；chunk 版本留作未來工作。頁面只在 S5 正文寫「實驗中 $\alpha,\theta,\eta$ 依 token 而定」，LTI／全域卷積放進預設收合的「進階」`<details>`（讀者未學過 LTI 與卷積，不設檢查題）。
- 論文 v1 沒有寫實驗用的 chunk 大小 $b$（R3）。

#### 步驟 8：持久記憶（Sec. 3.3，Eq. 19–20）

$$x_{new}=[p_1\ p_2\ \cdots\ p_{N_p}]\ \|\ x\tag{Eq. 19}$$

- 長期記憶是「情境記憶」（輸出完全依賴上下文）；另需一組可學習、不依賴輸入的參數儲存任務知識。三個觀點：
  1. 記憶觀點：掌握一個任務需要記住「如何做」這個任務。
  2. 前饋網路觀點：Sukhbaatar et al. (2019) 指出把 FFN 的 ReLU 換成 Softmax 會得到類似注意力、但權重不依賴資料的形式 $FFN(x)=W_V\,\mathrm{Softmax}(W_Kx)$（Eq. 20）；持久記憶放在序列開頭，會產生不依賴輸入的注意力權重。
  3. 技術觀點：因果遮罩的注意力偏向序列最前面的 token；開頭的可學習參數可以重新分配注意力權重（Han et al. 2024；Xiao et al. 2024）。
- 測試時持久記憶參數固定（Sec. 4.1 末段 (3)(i)）。

#### 步驟 9：三種把記憶放進架構的方式（Sec. 4，Eq. 21–31，Fig. 2–5）

作者把注意力解讀為短期記憶（依賴關係精確、視窗有限），神經記憶解讀為長期記憶。

**MAC（Memory as a Context，Sec. 4.1）**：序列切成長度 $C$ 的片段 $S^{(i)}$，$i=1,\dots,N/C$。

$$h_t=\mathcal M^*_{t-1}(q_t),\ q_t=S^{(t)}W_Q;\quad \tilde S^{(t)}=[p_1\cdots p_{N_p}]\ \|\ h_t\ \|\ S^{(t)};\quad y_t=\mathrm{Attn}(\tilde S^{(t)})\tag{Eq. 21–23}$$
$$\mathcal M_t=\mathcal M_{t-1}(y_t),\qquad o_t=y_t\otimes\mathcal M^*_t(y_t)\tag{Eq. 24, 25}$$

- 片段內是完整因果注意力（Fig. 3a）。Eq. 24 表示以 $y_t$ 經前向傳遞更新記憶。
- 作者列的優點：(1) 注意力同時看到歷史與當前，可以決定是否需要長期記憶的資訊；(2) 注意力幫助記憶只存有用的資訊；(3) 測試時持久記憶固定、注意力權重做 in-context learning、長期記憶持續學習。

**MAG（Memory as Gating，Sec. 4.2）**：不分片段，一支直接以輸入更新記憶，另一支用滑動視窗注意力。

$$\tilde x=[p_1\cdots p_{N_p}]\ \|\ x;\quad y=\text{SW-Attn}^*(\tilde x);\quad o=y\otimes\mathcal M(\tilde x)\tag{Eq. 26–28}$$

- $\mathcal M(\tilde x)$ 表示對序列遞迴後的記憶最終輸出。實驗中以可學習的向量權重分別正規化 $y$ 與 $\mathcal M(\tilde x)$，再接非線性 $\sigma(\cdot)$。作者稱 SWA 為精確的短期記憶，神經記憶為逐漸淡去的記憶（fading memory）；也可視為不同結構的多頭架構（Hymba）。

**MAL（Memory as a Layer，Sec. 4.3）**：

$$\tilde x=[p_1\cdots p_{N_p}]\ \|\ x;\quad y=\mathcal M(\tilde x);\quad o=\text{SW-Attn}(y)\tag{Eq. 29–31}$$

- 文獻中常見的混合模型都是這種逐層堆疊；缺點是模型能力受限於每一層，無法利用注意力與記憶的互補。實驗中以類似 H3 的架構、把序列模型換成 LMM。
- 注意 Eq. 31 是 SW-Attn（沒有星號），Eq. 27 是 SW-Attn$^*$（帶前綴）。C4 依方程式：MAL 的持久記憶只經由記憶層進入注意力（R5）。
- **LMM（無注意力）**：MAL 的特例，把 LMM 單獨當作序列模型；作者期望長期記憶在沒有短期記憶時也能獨立運作。

**架構細節（Sec. 4.4）**：所有區塊用殘差連接；q／k／v 的計算使用 SiLU，q 與 k 做 ℓ2 正規化；q／k／v 投影後各接一層 1D depthwise-separable 卷積；最終輸出投影前用正規化與線性層閘控（Mehta et al. 2023）。

**Theorem 4.1**（原文照錄）：Transformer、對角線性遞迴模型與 DeltaNet 受限於 TC⁰，Titans 能解決 TC⁰ 以外的問題，因此在狀態追蹤任務上理論表達力更強。v1 正文沒有證明（R7）；頁面寫成「論文宣稱，本頁不展開」，並加一句說明：「TC⁰ 是計算複雜度理論中的一個電路類別；論文把 Transformer 等模型能解的問題歸在此類，並宣稱 Titans 能解此類以外的問題。」

#### 步驟 10：與現代遞迴模型的關係（App. C，Eq. 32–35）

- 線性記憶 $\mathcal M_t=W_t$、損失 $\ell=\frac12\|\mathcal M_tk_t-v_t\|_2^2$ 時（Eq. 32–33）：$\mathcal M_t=\mathrm{diag}(1-\alpha_t)\mathcal M_{t-1}+S_t$，$S_t=\mathrm{diag}(\eta_t)S_{t-1}-\mathrm{diag}(\theta_t)(\mathcal M_{t-1}k_t^\top k_t-v_t^\top k_t)$。**原文 Eq. 32 右邊寫成 $\mathcal M_t$，應為 $\mathcal M_{t-1}$**（R1）；Eq. 33 的轉置寫法與本頁「行向量 key、$\mathcal M k$」慣例不同，頁面統一寫 $(\mathcal M_{t-1}k_t-v_t)k_t^\top$。
- DeltaNet 視為最佳化 $\frac12\|S_tk_t-v_t\|_2^2$ 的線上學習（Eq. 34）：$S_{t+1}=S_t-\theta_t\nabla L=S_t(I-\theta_tk_tk_t^\top)+\theta_tv_tk_t^\top$。Gated DeltaNet 是 Eq. 34 再加 weight decay。論文寫（App. C）：「setting $\eta_t=0$ results in both formulations to be equivalent」。
- 〔本頁推導〕由 Eq. 13–14 取 $\eta=0$：$\mathcal M_t=\mathcal M_{t-1}\big((1-\alpha_t)I-\theta_tk_tk_t^\top\big)+\theta_tv_tk_t^\top$；再取 $\alpha_t=0$ 即 Eq. 34（DeltaNet）。這一步可以由本論文的方程式驗證。
- **與 Gated DeltaNet 的對應只到「形式相近」（R13）**：本論文沒有寫出 Gated DeltaNet 的遞迴式。依審查者對 Gated DeltaNet 原論文的記憶，其閘門 $\alpha$ 同時乘在舊記憶與 delta 修正項上（形如 $\alpha M(I-\beta kk^\top)+\beta vk^\top$），若是如此，上式的衰減只作用在 $\mathcal M_{t-1}$ 本身，兩者無法以重新參數化互換。本頁未核對原論文，因此頁面寫：「論文稱 η = 0 時兩者等價；本頁只能從 Eq. 13–14 推出上式，它與 Gated DeltaNet 的更新規則形式相近，細節（衰減是否也作用在修正項）需查原論文。」LO11 與 C1 chip 皆依此措辭。
- LMM 推廣 Gated DeltaNet 的三點：(1) 動量規則（同時考慮過去與當下的驚訝度）；(2) 深層記憶（Gated DeltaNet 需要閉式遞迴，只能用矩陣記憶）；(3) 非線性遞迴：chunk 之間非線性、chunk 內線性。作者說 RWKV-7 也用同樣的形式與損失。
- Longhorn（Eq. 35）：$S_{t+1}=S_t(I-\delta_tk_tk_t^\top)+\delta_tv_tk_t^\top$，$\delta_t=\frac{\theta_t}{1+\theta_tk_tk_t^\top}$，以隱式線上學習推得；缺少遺忘閘。
- TTT layer 與 LMM 的三個差異：(1) TTT 沒有遺忘機制；(2) TTT 只用當下驚訝度；(3) TTT 允許深層記憶，但沒有實驗評估其利弊。作者宣稱 LMM 是第一個使用動量更新規則的線性遞迴模型。
- 作者強調：既有混合模型（Griffin、DeltaNet、Gated DeltaNet、H3、Mamba2、Samba）都是逐層設計；Titans 展示其他組合方式。App. A 另稱 Titans (MAC) 是 RMT 的推廣（以神經記憶取代小型向量記憶）。

#### 步驟 11：實驗（Sec. 5，Table 1–5，Fig. 6–9）

- **設定（Sec. 5.1）**：四種規模 170M、340M、400M、760M；前三者用 FineWeb-Edu 15B token、760M 用 30B token。訓練流程沿用 Gated DeltaNet：Llama 2 tokenizer（詞彙 32K）、訓練長度 4K token、AdamW、學習率 4e-4、cosine annealing、batch 0.5M token、weight decay 0.1。400M 基準的數字取自 Gated DeltaNet 論文（App. B）。v1 註腳：更大模型的結果將在下一版報告。
- **語言模型與常識推理（Table 1）**，摘錄（Wiki ppl ↓／Avg acc ↑）：

  | 規模 | Transformer++ | Gated DeltaNet | TTT | Titans (LMM) | MAC | MAG | MAL |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 340M / 15B | 31.52 / 42.92 | 27.01 / 45.42 | 27.44 / 44.51 | 26.18 / 46.17 | 25.43 / 47.36 | 25.07 / 47.54 | 24.69 / 46.55 |
  | 760M / 30B | 25.21 / 48.69 | 21.18 / 49.69 | 24.17 / 47.32 | 20.04 / 51.56 | 19.93 / 52.51 | 18.61 / 52.50 | 19.07 / 50.97 |

  760M 混合基準：Samba 20.63／51.08、Gated DeltaNet-H2 19.88／51.49。作者的解讀（Sec. 5.2，頁面標「作者宣稱」）：非混合模型中 LMM 最好；三種 Titans 都優於 Samba 與 Gated DeltaNet-H2；MAC 與 MAG 接近，MAC 在較長依賴上較好；MAG、MAC 都優於 MAL。
  **表格中與作者宣稱不一致的格子**（頁面在 S9 表下逐條列出，標「本頁核對」）：
  1. Wiki ppl 上 MAL 優於 MAC 與 MAG：340M 24.69 對 25.43、25.07；400M 23.93 對 MAC 25.61。
  2. 760M Wiki ppl：MAC 19.93 高於 Gated DeltaNet-H2 19.88（ppl 越低越好）。400M：Gated DeltaNet-H2 24.19 低於 MAC 25.61。
  3. 760M Avg acc：MAL 50.97 低於 Samba 51.08 與 Gated DeltaNet-H2 51.49。
  4. 760M MAL 的 SIQA 30.98，比相鄰各列（38.98–41.87）低約 10；Avg 50.97 與 8 欄平均一致，所以不是 Avg 算錯，疑似該格筆誤。照錄並註明。
  頁面的說法：「Avg acc 上 MAC、MAG 在列有混合基準的 400M 與 760M 都高於 Samba 與 Gated DeltaNet-H2（340M 未列混合基準）；MAL 的 Avg 與各模型的 Wiki ppl 則不是每格都最好。」
- **S-NIAH（Table 2，RULER，2K–16K）**，16K 欄（PK／N／W）：TTT 88.4／4.4／0.0；Mamba2 5.4／0.0／0.0；DeltaNet 71.4／5.4／0.0；LMM 96.2／80.2／80.6；MAC 98.4／97.4／95.2；MAG 97.4／98.6／88.2；MAL 97.8／96.4／90.4。三種 Titans 中沒有一種在 16K 欄每格都最好（N 欄 MAG 98.6 高於 MAC 97.4）。作者把 LMM 的優勢歸因於：相對 TTT 有動量與遺忘、相對 Mamba2 有深層非線性記憶與移除記憶的能力、相對 DeltaNet 有遺忘機制。
- **BABILong（Fig. 6）**：few-shot 與 fine-tuning 兩種設定，Titans (MAC) 優於所有基準（含 GPT-4）；與 Llama3.1-8B + RAG 相比參數少約 70 倍。正文只有文字敘述、沒有數值（R6），頁面不寫圖中數值。
- **深度記憶（Sec. 5.5，Fig. 7–8）**：$L_{\mathcal M}=1,2,3,4$，Pile 子集，Mamba 為基準。記憶越深，各序列長度的 perplexity 都較好；參數較少時深層記憶對序列長度更穩健。吞吐量：所有模型對上下文長度呈線性（tokens/s 不隨長度變），記憶越深訓練越慢，呈線性趨勢——效果與效率的權衡。Fig. 7b 標題寫 360M，與 Sec. 5.1 的 340M 不同（R2）。
- **時間序列（Table 3）**：以 Simba 框架、把 Mamba 模組換成神經記憶；例 ETTm1 MSE／MAE 0.358／0.387（Simba 0.383／0.396），Traffic 0.415／0.289。作者寫「outperforming all baselines」（Sec. 5.6）；但 Traffic 的 MAE 0.289 高於 iTransformer 的 0.282（MSE 0.415 仍最低）。頁面加註腳指出這一格（〔本頁核對〕）。**DNA（Table 4）**：GenomicsBenchmarks top-1 準確率，LMM 在 Enhancer Cohn 75.2（最高）、Enhancer Ens 89.6（最高）、Human Reg. 89.3（低於 HyenaDNA 93.8）；作者用詞為「competitive」。
- **效率（Fig. 9）**：記憶模組比 Mamba2 與 Gated DeltaNet 稍慢（深層記憶、更有表達力的轉移；Mamba2 的核心高度最佳化）；Titans (MAL) 比基準與記憶模組本身快，原因是 FlashAttention 實作的 SWA。
- **消融（Table 5）**（ppl ↓／Reasoning acc ↑／Long Context acc ↑）：LMM 27.01／47.83／92.68；+Attn (MAC) 26.67／48.65／97.95；(MAG) 25.70／48.60／96.70；(MAL) 25.91／47.87／96.91；Linear Memory 28.49／46.97／85.34；w/o Convolution 28.73／45.82／90.28；w/o Momentum 28.98／45.49／87.12；w/o Weight Decay 29.04／45.11／85.60；w/o Persistent Memory 27.63／46.35／92.49。作者：所有元件都有正面貢獻，貢獻由大到小為 weight decay、動量、卷積、持久記憶。這個排序在 ppl（29.04 > 28.98 > 28.73 > 27.63）與 Reasoning（45.11 < 45.49 < 45.82 < 46.35）上成立；Long Context 上最低的是 Linear Memory 85.34，其次才是 w/o Weight Decay 85.60、w/o Momentum 87.12。頁面寫明「依 ppl 與 Reasoning」。
- **表格內部一致性（`tables.py`，〔本頁推導〕；頁面以註腳呈現，不改原數字）**：
  1. Table 5 的四個 LMM/Titans 列等於 Table 1 的 **400M** 列：ppl = (Wiki ppl + LMB ppl)/2（27.01、26.67、25.70、25.91 完全一致），Reasoning = Table 1 的 Avg；Long Context = Table 2 該模型 12 格的平均（92.68、97.95、96.70、96.91 完全一致）。正文稱 Long Context 為「long context NIAH (BABILong)」，但數值與 S-NIAH 一致（R2）。
  2. Table 1 的 Avg 欄與 8 個準確率欄的平均不符的列：760M TTT（表列 47.32，平均 49.37）、400M Gated DeltaNet-H2（47.69 vs 47.80）、400M Titans (LMM)（47.83 vs 47.865）。其餘列一致。
  3. 760M 的三個 Titans 列沒有標混合模型的星號（340M、400M 有）。

#### 步驟 12：實作細節與常見錯誤（頁面 S10 逐條列出）

1. 把驚訝度當成對輸入的梯度：更新式需要對記憶參數的梯度（步驟 2）。
2. 把 Eq. 7 下方的 read／write 文字當真：f 是寫入、g 是讀取（步驟 1）。
3. 忘記 chunk 內的梯度都在 $\mathcal M_{t'}$ 計算：$b>1$ 時結果與逐 token 更新不同；$b=1$ 才相同（步驟 7、C3）。
4. 以 $\beta_t/\beta_i$ 相除計算衰減權重：$\alpha=1$ 時 0/0，應直接連乘（步驟 7）。
5. Eq. 17 的 $(W_0x_t-x_t)x_t^\top$ 是 key = value = $x$ 的簡寫；一般情形是 $(W_0k_t-v_t)k_t^\top$，係數 2 取決於損失是否乘 ½（步驟 7）。
6. MAC 與 MAG／MAL 的注意力範圍不同：MAC 片段內完整因果注意力＋記憶 token＋持久記憶；MAG 帶前綴的 SWA；MAL 不帶前綴的 SWA（步驟 9、C4）。
7. Eq. 32 右邊的 $\mathcal M_t$ 應為 $\mathcal M_{t-1}$（步驟 10）。
8. 論文 v1 未說明：chunk 大小、記憶 MLP 寬度、$\eta_t/\theta_t/\alpha_t$ 的參數化、片段長度 $C$、視窗大小、$N_p$、$N_l$、閘控非線性 $\sigma$ 的形式。復現時這些都要自行決定。
9. 照字面使用 Eq. 16 的 $t'=t-\mathrm{mod}(t,b)$：$t=b$ 時得到 $b$ 而非 0，$b=1$ 時梯度會在更新後的 $\mathcal M_t$ 上計算；應取 chunk 起點 $t'=t-1-\mathrm{mod}(t-1,b)$（步驟 7、R1）。

#### 步驟 13：限制（作者未設獨立限制章節；以下由論文內容整理）

- v1 為初版：作者在註腳表示仍在完成更大模型的結果；程式碼「將會釋出」（Sec. 6）。
- 超過 2M 上下文的宣稱只出現在 Abstract、Sec. 1 與 Sec. 6，正文沒有對應的數值表格。
- Theorem 4.1 沒有證明；多項實作超參數未寫明（步驟 12 第 8 條）。
- 頁面在 S12 以「論文寫明的／論文未寫明的」兩欄呈現，不加入論文以外的評論。

---

## 2. 學習目標（驗收測試）

每個參考答案都必須寫在該章節的**內文**中（不能只出現在元件、表格、圖或答案框）。每章結尾以 `<details>` 呈現該章的檢查題（題目與參考答案照搬）。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 用寫入／讀取描述三類序列模型 | 線性注意力的記憶如何寫入與讀取？Transformer 的記憶與它有何不同？ | $\mathcal M_t=\mathcal M_{t-1}+K_t^\top V_t$、$y_t=Q_t\mathcal M_t$（Eq. 4–5），壓縮成固定大小矩陣；Transformer 把 key／value 附加到記憶、不壓縮（Sec. 1、2.1） | S1 |
| LO2 | 說明驚訝度的定義 | Eq. 8 的驚訝度是什麼？是對什麼求梯度？ | 聯想記憶損失 $\ell(\mathcal M_{t-1};x_t)$ 的梯度，對記憶參數求梯度（更新式需要它）；梯度越大表示輸入與過去的資料差異越大（Eq. 8） | S2 |
| LO3 | 區分內外兩層迴圈 | 內層迴圈與外層迴圈各更新哪些參數？ | 內層：記憶 $\mathcal M$ 的權重，以 $\|\mathcal M_{t-1}(k_t)-v_t\|_2^2$ 在測試時更新；外層：整個架構的其他參數（含 $W_K,W_V$，在內層損失中是超參數）（Eq. 11–12） | S2 |
| LO4 | 說明動量的作用 | 為什麼要加入過去的驚訝度 $S_{t-1}$？$\eta_t\to0$ 代表什麼？ | 大驚訝之後梯度可能變小，使更新停在平坦區域而錯過後續資訊；$S_t=\eta_tS_{t-1}-\theta_t\nabla\ell$ 讓驚訝度延續；$\eta_t\to0$ 忽略上一次驚訝（例如語境改變）（Eq. 9–10） | S3 |
| LO5 | 說明遺忘閘 | Eq. 13 中 $\alpha_t\to0$ 與 $\alpha_t\to1$ 各代表什麼？它等同什麼？ | $\alpha_t\to0$ 不影響過去的內容；$\alpha_t\to1$ 清空記憶；等同 weight decay，與現代 RNN 的閘控相關（Eq. 13–14） | S3 |
| LO6 | 讀懂 C1 的權衡 | 在 C1 預設情境（seed 11）中，動量與遺忘各帶來什麼可觀察的效果？為什麼要和「θ = 0.75、η = 0」比較？ | 動量使驚訝度下降後 $\|S_t\|$ 仍延續（$\|S_{40}\|$：η = 0.6 為 0.152，η = 0 為 0.092）。情境 B 結束時的回想誤差由 0.025（η = 0）降到 0.008（η = 0.6），但常數 η 讓有效步長變成 θ/(1−η) = 0.75，而 θ = 0.75、η = 0 也降到 0.011；seed 11–30 的中位數為 0.0087 對 0.0083，所以這個玩具中誤差下降大多來自步長放大，不能當作動量本身的好處。遺忘使回到情境 A 前的 A 回想誤差變大（θ = 0.3、η = 0.6；α = 0：0.387；α = 0.02：0.457；α = 0.1：0.582），即遺忘的代價是保留較少（本頁玩具，〔本頁推導〕） | S3 |
| LO7 | 說明為何需要深層記憶 | 矩陣記憶等價於什麼問題？Sec. 5.5 的深度實驗結論是什麼？ | 線上線性迴歸 $\|W_{t-1}k_t-v_t\|^2$，假設依賴關係是線性的；$L_{\mathcal M}$ 越大 perplexity 越好、對序列長度越穩健，但訓練吞吐量隨深度線性下降（Sec. 3.1、5.5，Fig. 7–8） | S4 |
| LO8 | 說明平行化訓練 | chunk 內的梯度在哪個記憶上計算？$b=1$ 時與逐 token 更新的關係？動量如何平行？ | 都在 chunk 起點 $\mathcal M_{t'}$（Eq. 16；本頁把 $t'$ 寫成 $t-1-\mathrm{mod}(t-1,b)$，因為論文印的 $t-\mathrm{mod}(t,b)$ 與其「第一個 chunk 的 $t'=0$」不符）；在這個讀法下 $b=1$ 時完全相同；動量 Eq. 18 是以 $u_t$ 為輸入的線性遞迴，用平行 associative scan 計算（Sec. 3.2） | S5 |
| LO9 | 說明持久記憶 | 持久記憶如何加入序列？列出作者的三個動機。測試時是否更新？ | 把 $N_p$ 個可學習向量接在序列前面（Eq. 19）；記憶觀點（儲存任務知識）、前饋網路觀點（不依賴資料的注意力權重，Eq. 20）、技術觀點（緩解因果注意力偏向開頭 token）；測試時固定（Sec. 3.3、4.1） | S6 |
| LO10 | 比較三種架構 | MAC、MAG、MAL 各如何結合記憶與注意力？哪一種在長上下文任務最好？ | MAC：把 $h_t=\mathcal M^*_{t-1}(q_t)$ 與持久記憶接在片段前，做片段內注意力，再以 $y_t$ 更新記憶；MAG：帶前綴 SWA 與記憶並行、以閘控結合；MAL：記憶層之後接 SWA；論文認為 MAC 在長上下文最好：Table 5 Long Context 97.95 最高，BABILong（Fig. 6）只報告 MAC；但 Table 2 並非每格最好（16K 的 N 欄 MAG 98.6 > MAC 97.4）（Eq. 21–31，Sec. 5） | S7 |
| LO11 | 連到現代遞迴模型 | 令 $\eta_t=0$ 時 LMM 的線性版本對應哪個模型？LMM 推廣了它的哪三點？與 TTT 的差異？ | 論文稱等價於 Gated DeltaNet（DeltaNet 加 weight decay，Eq. 32–34）；本頁能驗證的是 η = 0、α = 0 時即 DeltaNet（Eq. 34），與 Gated DeltaNet 只是更新規則形式相近；動量規則、深層記憶、chunk 間非線性遞迴；TTT 沒有遺忘、只用當下驚訝度、未評估深層記憶（App. C） | S8 |
| LO12 | 讀出主要結果 | 760M 時 Titans (MAG) 與 Transformer++ 的 Wiki ppl？依 ppl，消融中移除哪個元件傷害最大？ | MAG 18.61 vs Transformer++ 25.21（Table 1）；依 ppl（與 Reasoning）是移除 weight decay（ppl 29.04，LMM 為 27.01），其次動量 28.98、卷積 28.73；Long Context 上最差的則是 Linear Memory 85.34（Table 5） | S9 |

---

## 3. 頁面大綱

單欄閱讀（文字欄 680–720px），互動元件放在加寬圖版（最大 960px）。章節編號以 JetBrains Mono 顯示（`03 · 動量與遺忘`）。**視覺記號**：一條水平「token 軸」，上方三條帶狀區分持久記憶（序列開頭、固定）、注意力視窗（當前、精確）、長期記憶（過去、壓縮），在 hero、C4、S7 重複出現。章節標題一律用陳述句。

| # | id | 標題 | 內容 | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈Titans：在測試時學習記憶〉 | 副標：「這篇論文把長期記憶做成一個在測試時以梯度下降更新的 MLP，再與注意力、持久記憶組成三種架構。」副標下一行定義：「測試時＝模型訓練完成、部署後處理新序列的時候；此時記憶模組的權重仍在更新。」token 軸記號（靜態 SVG）＋一行圖例（三種帶狀各一個色塊與文字：「持久記憶（S6）」「注意力視窗（S7）」「長期記憶（S2–S5）」，文字為章節錨點連結）。三張事實卡（見下）。連結：arXiv、`../index.html#paper-243`、延伸閱讀總覽 | 1 螢幕 |
| S1 | `memory` | 01 · 序列模型都有寫入與讀取兩個步驟 | **前置說明兩段**（先於任何方程式）：(1) 注意力的輸入是 $N$ 個 token 向量，輸出是同樣 $N$ 個向量，第 $i$ 個輸出是前面各 token 內容的加權平均，權重由相似度決定；softmax 把一列分數變成總和為 1 的正權重。(2) q／k／v 的角色：每個 token 投影出 query（我要找什麼）、key（我可以被什麼找到）、value（被找到時交出的內容）；$q_i$ 與 $k_j$ 的內積決定權重。接著 Eq. 1–2、Eq. 4–5、Eq. 6–7；Eq. 3 與核函數 φ 放進預設收合的 `<details>`「線性注意力的一般形式」；比較表（步驟 1）；read／write 文字錯置的註記。末尾可收合符號表 | 550–650 字 + 表 |
| S2 | `surprise` | 02 · 記憶以梯度學習 key 到 value 的對應 | Eq. 8、11、12；驚訝度是對記憶參數的梯度（LO2）；內外兩層迴圈兩欄表（LO3）；章末 Q1–Q5（從 S1 移來，此時讀者已知道記憶以梯度寫入），每題附「本頁在 S○ 回答」的錨點連結（Q1、Q5 → S4；Q2 → S2–S3；Q3 → S4 的 Eq. 15；Q4 → S7） | 400–500 字 |
| S3 | `momentum` | 03 · 動量延續驚訝度，遺忘閘清除舊內容 | 分兩個小節：**3a 動量**（Eq. 9–10、η 與 θ 的極端值、有效步長 θ/(1−η) 一句）；**3b 遺忘**（Eq. 13–14、α 的極端值、〔本頁推導〕逐 token 演算法、係數 ½ 一行）。之後 **C1**；C1 下方內文寫出 LO6（含步長對照與 seed 說明） | 550–650 字 + C1 |
| S4 | `deep` | 04 · 深層記憶能存下非線性的對應 | 矩陣記憶 = 線上線性迴歸；MLP $L_{\mathcal M}$；Eq. 15 讀取；**C2**；Sec. 5.5 文字結論（Fig. 7–8 不重繪） | 350–450 字 + C2 |
| S5 | `parallel` | 05 · 把序列切成 chunk 就能用矩陣乘法訓練 | Eq. 16–18；$t'$ 的讀法與第二個等號只在單一 chunk 內成立；一般形式與含動量的 chunk 公式〔本頁推導〕；associative scan 兩句說明（步驟 7）；β 比值的數值注意；LTI／全域卷積放進收合的「進階」；**C3** | 500–600 字 + C3 |
| S6 | `persistent` | 06 · 持久記憶是放在序列開頭的可學習參數 | Eq. 19–20；三個動機；測試時固定 | 250–350 字 |
| S7 | `variants` | 07 · Titans 以三種方式結合記憶與注意力 | MAC Eq. 21–25、MAG Eq. 26–28、MAL Eq. 29–31、LMM；三欄比較卡；**C4**；Sec. 4.4 細節清單；Theorem 4.1（論文宣稱，附 TC⁰ 一句說明，步驟 9） | 550–650 字 + C4 |
| S8 | `relations` | 08 · LMM 推廣了 DeltaNet、Longhorn 與 TTT | Eq. 32–35；η = 0 的對應〔本頁推導〕與「與 Gated DeltaNet 只到形式相近」（步驟 10）；三點推廣；TTT 三差異；App. A.1 的三代線性遞迴分類一句（第一代資料無關的衰減、第二代閘控、第三代以線上學習或 delta rule 更新）；對照表（遺忘／動量／深層記憶／遞迴型態） | 400–500 字 + 表 |
| S9 | `results` | 09 · 實驗中 Titans 在多數指標上優於同規模的 Transformer 與線性遞迴模型 | Table 1 摘錄＋「作者宣稱」與表格例外清單（步驟 11，逐欄列出）、Table 2 16K 欄、Table 5 全表（排序註明依 ppl／Reasoning）、Table 3（含 Traffic MAE 註腳）／Table 4 各一句、BABILong 與 Fig. 7–9 文字結論；一致性註腳（步驟 11） | 500–600 字 + 3 表 |
| S10 | `impl` | 10 · 實作設定與常見錯誤 | Sec. 5.1 訓練設定小表、Sec. 4.4 細節；步驟 12 的 9 條 | 300–400 字 |
| S11 | `history` | 11 · 研究脈絡（由 notes/timeline.json 提供） | 時間線（6.6，13 條，2014–2024）＋一段文字：外部記憶與分段記憶（NTM、Transformer-XL、RMT）→ 線性注意力與快速權重 → 閘控線性遞迴（Mamba、Griffin）→ Delta rule（DeltaNet、Gated DeltaNet）→ 梯度式更新（TTT）→ 動量＋遺忘＋深層記憶（本文）。Schmidhuber 1992 不在 arXiv、不在時間線，文字中寫「依 Titans 的轉述」 | 時間線 + 100–150 字 |
| S12 | `next` | 12 · 限制與延伸閱讀 | 步驟 13；筆記清單（6.6，hub `titans-lineage-overview` 在前）；arXiv | 200 字 + 連結 |

Hero 三張事實卡（術語在卡內定義）：
1. 「長期記憶是一個小型神經網路。每讀到一個新 token，它就做一步梯度下降，讓這個 token 的索引向量（key）能取回對應的內容向量（value）；部署後處理新序列時也照做。」
2. 「梯度越大，表示新 token 與已記住的內容越不同。作者稱之為驚訝度，並加上動量（延續過去的驚訝度）與遺忘閘（衰減舊內容）。」
3. 「Titans 把這個記憶與注意力、持久記憶組合成三種架構。760M 參數時，三者在 Wikipedia 文字上的 perplexity（困惑度，越低越好）為 18.61–19.93，Transformer++ 為 25.21。」

章節中的延伸閱讀（上線後同在 `tutorials/`；**新**＝`tut/243/notes/` 本次產生的筆記）：S1 → **`memory-view-of-sequence-models.html`**、**`linear-attention-fast-weights.html`**；S2 → **`titans-neural-memory-surprise-update.html`**、`ttt-linear-mlp.html`；S3 → **`gated-linear-recurrence-forgetting.html`**；S5 → `ttt-linear-mlp.html`、`lact-large-chunk-ttt.html`；S7 → **`titans-architectures-mac-mag-mal.html`**、**`recurrent-memory-segment-transformers.html`**；S8 → **`delta-rule-deltanet-family.html`**、`ttt-sequence-modeling-overview.html`、`ttt-theory-and-applications.html`；S11、S12 → hub **`titans-lineage-overview.html`** 在最前，S12 另加 `test-time-training-naming.html`。順序與 hub 的閱讀順序一致（R11 已解決）。

---

## 4. 互動元件規格

### 4.0 共用

- PRNG：`mulberry32(seed)`；`gauss(r)`：Box–Muller，每次取兩個均勻數，回傳 $\sqrt{-2\ln(1-u_1)}\cos(2\pi u_2)$（只用 cos 分支）。
- 發散保護（所有元件同一規則）：每一步後若任何參數絕對值 $>10^3$，或損失不是有限數，立即停止，畫到前一步為止，在讀數區顯示「第 N 步數值發散：θ 或 η 太大」，並把讀數改為「—」。圖上縱軸以對數刻度顯示的量，超過上限時畫在上緣並加向上箭頭標記，不畫出界；低於下限或等於 0 時，`scaleLog` 一律先取 `max(v, floor)`（floor = 軸下限），畫在下緣並加向下箭頭標記，避免 `log(0)` 產生 -Infinity／NaN 使 path 中斷。線性縱軸同理，以 clamp 處理。
- 發散訊息放在獨立的 `role="alert"` 元素，發生時**立即**宣告一次（不經 500ms 去抖），同一次執行不重複宣告；重設時清空。
- live region：讀數以 `aria-live="polite"` 宣告，只在 `change`（放開滑桿）或停止輸入 500ms 後更新；動畫進行中不宣告，只在完成或暫停時宣告一次。
- 每個圖版底部一行：「讀數由本頁依 Eq. ○○ 即時計算；設定值為本頁推定，論文未給。」
- **資料表**（C1、C2、C3）：圖版右上一個切換鈕「顯示資料表」（`aria-expanded`、`aria-controls`），展開一個 `.tscroll` 內的 `<table>`（`<caption>` 寫目前設定，容器 `tabindex="0"`）。C1、C3：每 8 步一列（t = 8、16、…、96，12 列），欄為各曲線的值；C2：每 250 個 token 一列（12 列）＋目前 token。發散後的列顯示「—」。這是鍵盤與螢幕閱讀器取得逐點值的方式；圖本身不依賴 hover。
- **情境帶與深色模式**：C1、C3 的「情境 A／B」背景帶用 `--band-a`、`--band-b`。淺色：`--ink` 4%／9% 不透明度；深色：`--ink` 7%／14% 不透明度（在 `:root` 與兩段深色覆寫中都要定義）。帶狀只作背景，三段同時以文字「情境 A／B／A」標示，不靠顏色傳達。hatch 圖樣的 stroke 走 CSS 變數，深色模式隨之改變。

### C1 · 驚訝度、動量與遺忘：一個線性記憶的測試時學習（S3）

- **教的概念**：Eq. 13–14 的三個係數各自做什麼；驚訝度（梯度大小）在語境切換時升高；動量使更新在驚訝之後延續；遺忘閘使舊情境的內容衰減。
- **資料（推定，頁面寫出）**：$d=8$。兩個情境 A、B，各有 4 個單位長度 key（8 維高斯向量正規化）與一個對應矩陣 $A_c$（元素 $\mathcal N(0,1)/\sqrt8$），value $v=A_ck$。token 序列長 96：$t=1$–32 為 A、33–64 為 B、65–96 回到 A；每個 token 從該情境的 4 個 key 中均勻抽一個。生成順序（固定，確保與驗算一致）：A 的 4 個 key → B 的 4 個 key → $A_A$ → $A_B$（各列優先）→ 96 個 key 索引（`min(3, floor(r()*4))`）。seed 預設 11，「重新抽樣」每按一次 +1。
- **演算法**（逐 token，參考 `widgets.js` 的 `runC1`）：$e_t=\mathcal M_{t-1}k_t-v_t$；驚訝度 $\|g_t\|_F=\|e_t\|$；$S_t=\eta S_{t-1}-\theta e_tk_t^\top$；$\mathcal M_t=(1-\alpha)\mathcal M_{t-1}+S_t$；$\mathcal M_0=S_0=0$。更新後記錄 $\|S_t\|_F$ 與兩個情境的回想誤差 $E_c(t)=\frac14\sum_{k\in c}\frac12\|\mathcal M_tk-A_ck\|^2$。
- **讀者操作**：滑桿 θ ∈ [0, 1]（步長 0.01，預設 0.30）、η ∈ [0, 0.95]（0.01，預設 0.60）、α ∈ [0, 0.2]（0.005，預設 0.02）；四個預設 chip，以參數命名（S3 時讀者還沒學過 TTT 與 DeltaNet）：「η = 0、α = 0」「η = 0、α = 0.02」「η = 0.6、α = 0.02（預設）」（θ 都是 0.3）與「θ = 0.75、η = 0（等效步長對照）」。前兩個 chip 下方小字：「S8 會說明：前者即 DeltaNet 的更新規則（Eq. 34），後者論文稱與 Gated DeltaNet 等價（本頁只確認形式相近）」（依步驟 10 的推導：η = 0、α = 0 時 Eq. 13–14 化為 Eq. 34；TTT 與 LMM 的差異另見 S8，不掛在 chip 上）。「重新抽樣」：seed +1；圖版標題列以等寬字顯示「seed 11」，seed ≠ 11 時在讀數上方顯示提示「內文數字依 seed 11；目前 seed 為 N，數值會不同」，旁邊一個「回到 seed 11」按鈕。拖動滑桿即時重算（96 步 × 64 元素，<1ms）。
- **顯示**：兩張共用橫軸 $t\in[1,96]$ 的圖，背景以淺色帶與文字「情境 A／B／A」標出三段。
  - 上圖「驚訝度」：$\|g_t\|$ 畫成細長條（`--accent`），$\|S_t\|_F$ 畫成折線（`--c-par`，實線＋圓點標記）；線性縱軸 [0, 2]，超出上限依 4.0 規則。
  - 下圖「回想誤差」（對數縱軸 $[10^{-3},10^{3}]$）：$E_A$ 實線（`--c-par`，標「情境 A」）、$E_B$ 虛線（`--c-perp`，標「情境 B」），直接文字標籤在線尾。
  - 讀數表（JetBrains Mono，4 位小數）：預設只顯示 3 個——$E_B(64)$（動量／步長）、$E_A(64)$（遺忘）、$\|S_{40}\|_F$（動量延續）；其餘（切換後 8 個 token 的平均驚訝度 $t$ = 33–40、65–72，$E_A(32)$、$E_A(96)$）放在預設收合的「全部讀數」`<details>`。
  - 圖版下方一行：「本元件的損失取 ½‖·‖²，θ = 0.3 相當於照 Eq. 12（無 ½）寫法的 θ = 0.15。」
- **讀者應觀察到**：
  1. 每次切換（$t=33$、65）驚訝度升高，之後下降。
  2. η > 0 時，驚訝度下降後 $\|S_t\|$ 仍維持一段時間（預設 $\|S_{40}\|=0.152$，η = 0 時 0.092）；η 太大（0.9）會來回震盪、誤差變大。
  3. **動量與步長的混淆（頁面明寫）**：常數 η 時，持續同方向的梯度累積成 θ/(1−η) 倍的步長（η = 0.6 → 0.75）。$E_B(64)$：η = 0 為 0.0251，η = 0.6 為 0.0084，但「θ = 0.75、η = 0」也有 0.0106；seed 11–30 的中位數為 0.0352（η = 0）、0.0087（η = 0.6）、0.0083（θ = 0.75、η = 0），η = 0.6 只在 14／20 個 seed 低於對照。所以在這個玩具裡，誤差下降大多是步長效果；作者主張的「驚訝度延續」好處，本元件只能以 $\|S_t\|$ 的延續呈現，不能以誤差證明。
  4. α 越大，回到情境 A 之前 $E_A(64)$ 越大（θ = 0.3、η = 0.6 固定；α = 0／0.02／0.1：0.387／0.457／0.582）；α = 0 時情境 A 的內容部分保留。這個玩具只呈現遺忘的**代價**；作者論證的好處是長序列中的容量管理，證據是 Table 5（移除 weight decay 的 ppl 29.04 最差）。頁面明寫此限制。
- **無障礙**：圖 `role="img"`，`aria-label` 摘要目前三個讀數與 seed；逐點值由「顯示資料表」取得（4.0）；長條與折線以形狀（長條／圓點折線）及文字區分。

### C2 · 記憶深度：線性記憶與 MLP 記憶（S4）

- **教的概念**：矩陣（線性）記憶只能學到線性對應（線上線性迴歸）；$L_{\mathcal M}\ge2$ 的 MLP 能在測試時學到非線性對應；深度增加計算量（Sec. 3.1、5.5）。
- **設定（推定，頁面寫出）**：key 為純量 $k\sim U[-2,2]$，value $v=\sin(2k)$。$L_{\mathcal M}=1$：$\mathcal M(k)=wk$（無偏置，對應矩陣記憶）。$L_{\mathcal M}\ge2$：寬度 16 的 MLP，隱藏層 SiLU、含偏置，輸出層無偏置。初始化（seed 5）：$W\sim\mathcal N(0,1)\sqrt{1/\text{fan\_in}}$，隱藏偏置 $\sim0.5\,\mathcal N(0,1)$，依層序、各列優先生成。$L_{\mathcal M}=1$ 時 $w$ 也是隨機初始（fan_in = 1，$w\sim\mathcal N(0,1)$，seed 5 取第一個高斯值），不是 0。資料流 seed 9：第 $t$ 個 key $=-2+4r()$。
- **演算法**（`runDeep`）：每個 token 一次前向＋解析反向傳播（損失 $\frac12(\mathcal M(k)-v)^2$；$\mathrm{silu}'(a)=\sigma(a)(1+a(1-\sigma(a)))$；$\sigma$ 依正負號分支計算避免 exp 溢位），所有參數依 Eq. 13–14 更新（每個參數一個 $S$）。共 3000 個 token。評估：201 點等距格點上的 MSE $\frac1{201}\sum(\mathcal M(k)-\sin2k)^2$。
- **讀者操作**：分段按鈕 $L_{\mathcal M}$ = 1／2／3／4（預設 2）；θ ∈ [0, 0.05]（0.001，預設 0.010）、η ∈ [0, 0.95]（0.01，預設 0.90）、α ∈ [0, 0.01]（0.0005，預設 0）；「執行」／「暫停」、「重設」；「比較四種深度」按鈕（同步跑四次 3000 步，約 0.2 秒，填入比較表）。滑桿拖動中（`input` 事件）只更新滑桿旁的數值文字，不重設；放開（`change` 事件）或切換分段按鈕時才重設並停在第 0 步，不自動重新播放（避免拖動時連續重設造成閃爍）。動畫每影格 30 個 token；`document.visibilitychange` 變成 hidden 時自動暫停（按鈕回到「執行」），回到頁面不自動續播。reduced-motion 時一次算完再畫。
- **顯示**：
  - 左圖：橫軸 $k\in[-2,2]$，目標 $\sin2k$（`--ink-3` 虛線，標「目標 v」），目前記憶 $\mathcal M^*(k)$（`--accent` 實線，標「記憶讀出」），最近 30 個 key 以小點畫在目標曲線上（標「最近寫入」）。
  - 右圖：格點 MSE 對 token 數（對數縱軸 $[10^{-5},10]$），每 50 個 token 取一點；水平虛線標「最佳線性 0.402」（**201 點格點**上的無偏置線性最小平方解 $w=0.1660$ 的格點 MSE 0.4018）與「記憶為 0：0.439」（格點 MSE 0.4389）。兩條線都以評估用的 201 點等距格點（含端點、等權）計算，不是 $U[-2,2]$ 上的連續期望值；連續期望值為 $w=\mathbb E[k\sin2k]/\mathbb E[k^2]=0.1742$、MSE 0.3977、記憶為 0 時 $\mathbb E[\sin^22k]=0.4382$，差異來自格點端點的權重。數值由 `verify/widgets.js` 的「C2 ref」行輸出（兩種算法並列），`c2_deep.py` 前兩行輸出相同的格點值。頁面虛線用格點值，因為右圖的 MSE 也是格點 MSE。
  - 讀數：token 數、MSE、參數數（1／48／320／592）、每 token 乘加數（前向＋反向約為參數數的 3 倍，標「相對計算量」）。
  - 比較表：$L_{\mathcal M}$、參數數、第 100／300／1000／3000 個 token 的 MSE。
- **讀者應觀察到**：
  1. $L_{\mathcal M}=1$ 的 MSE 停在 0.40–0.48，不會低於最佳線性解 0.402（格點 MSE 的下界；線上學習追蹤的是連續分布的 $w\approx0.174$，其格點 MSE 0.4019，貼近但不低於虛線）。
  2. $L_{\mathcal M}\ge2$ 在 3000 個 token 內遠低於線性解：$L=2$ 降到約 $2\times10^{-3}$（0.00235），$L=3$、4 降到 $1.5\times10^{-4}$ 以下（0.00010、0.00014）。
  3. 深度越大參數與計算量越大，對應 Fig. 8 的吞吐量下降；這個玩具不測量時間，只顯示乘加數。
  4. θ = 0、α > 0 時記憶衰減到 0，MSE 趨近 0.439（θ = 0、η = 0、α = 0.01：四種深度在 3000 步皆為 0.4389）。
  5. 極端值：θ = 0.05、η = 0.95、α = 0 時 $L=2$ 於第 103 步、$L=4$ 於第 58 步發散；α = 0.01 時 $L=2$ 仍於第 2082 步發散，$L=1$、3、4 不發散。

### C3 · chunk 平行化：凍結的梯度點與矩陣乘法形式（S5）

- **教的概念**：chunk 內所有梯度都在 $\mathcal M_{t'}$ 計算（Eq. 16）；這樣一整個 chunk 的更新可以寫成一次矩陣乘法 $E\,\mathrm{diag}(w)K^\top$（Eq. 17 的一般形式，含動量〔本頁推導〕）；chunk 越大，順序步數越少，但與逐 token 更新的差距越大。
- **資料與參數**：沿用 C1 的資料流（seed 與 C1 同步）；θ = 0.3、α = 0.02 固定，η 以分段按鈕選 0 或 0.6（預設 0.6）。
- **讀者操作**：分段按鈕 chunk 大小 $b$ = 1／2／4／8／16／32（預設 4）；η = 0／0.6。
- **計算**（`loopEnds`、`matmulEnds`、`runC1({chunk:b})`）：
  1. 迴圈形式：逐 token 更新，但 $e_t$ 用 chunk 起點的 $\mathcal M_{t'}$。
  2. 矩陣乘法形式：每個 chunk 先算 $E=\mathcal M_{t'}K-V$，再用 1.4 步驟 7 的 $w_l$、$c_S$、$(1-\alpha)^n$ 與 $q_l=-\theta\eta^{n-l}$（$S$ 的係數；索引 1-based，$0^0=1$，與步驟 7 相同）得出 chunk 終點的 $\mathcal M,S$。$w_l$ 直接以雙重迴圈連乘求和（$b\le32$），不做 $\beta$ 相除。
  3. 兩者在所有 chunk 終點的最大絕對差；以及 $b$ 與 $b=1$ 的回想誤差曲線。
- **顯示**：
  - 上方一列 token 方格（96 格），以粗邊框分成 chunk；每個 chunk 標「梯度點 $\mathcal M_{t'}$」於左緣。
  - 主圖：目前情境的回想誤差（$t\le32$ 與 $t>64$ 畫 $E_A$，其餘畫 $E_B$）對 $t$，對數縱軸 $[10^{-3},10^{3}]$；$b=1$ 為 `--ink-3` 細虛線（標「逐 token」），所選 $b$ 為 `--accent` 實線（標「chunk = b」）。
  - 讀數：順序步數 $\lceil96/b\rceil$；兩種形式的最大差（顯示為「< 1e-12，兩者相同」或實際值）；$E_A(96)$；$\|\mathcal M^{(b)}_{96}-\mathcal M^{(1)}_{96}\|_F$。
  - 圖下說明：「$b$ 大時，同一 chunk 內同一個 key 會被重複套用同一個（過時的）誤差，效果接近把 θ 放大，因此在這組固定參數下 $b\ge16$ 誤差快速增大。論文中 $\theta_t$ 依 token 而定，也沒有給出實驗用的 $b$。」〔本頁推導〕
- **讀者應觀察到**：
  1. 兩種形式完全相同（差 $<3\times10^{-14}$）：平行化不改變數學，只改變計算方式。
  2. $b=1$ 與逐 token 更新相同；$b$ 越大順序步數越少，但偏離越大（η = 0.6：$\|\Delta\mathcal M\|$ = 0.130、0.714、2.05、202 對應 b = 2、4、8、16）。
  3. η = 0 時較不敏感（$b=8$ 的 $E_A(96)$ 為 0.050，η = 0.6 為 0.600）。

### C4 · 三種架構的注意力範圍與記憶路徑（S7）

- **教的概念**：Fig. 3 的 mask 與 Eq. 21–31：每個 query 位置直接注意哪些 token、哪些 token 只能經由長期記憶到達、哪些不可見；以及注意力配對數的差別。
- **讀者操作**：分段按鈕 MAC／MAG／MAL（預設 MAC）；滑桿 query 位置 $t\in[1,16]$（預設 10），也可點選矩陣的列；滑桿片段長度 $C\in\{2,4,8\}$（MAC，預設 4）、視窗 $w\in\{2,4,8\}$（MAG／MAL，預設 4）、$N_p\in\{0,\dots,4\}$（預設 2）、$N_l\in\{0,\dots,4\}$（MAC，預設 2；切到 MAG／MAL 時滑桿 `disabled` 並以小字註明「只用於 MAC」，矩陣中 $h$ 欄群組整組隱藏，不留空欄）。序列長度 $N=16$ 固定。
- **規則**（`maskRow`，逐格分類；優先順序：直接 > 經由記憶 > 不可見）：
  - MAC：持久記憶與 $N_l$ 個記憶 token「直接」；序列 token $j\le t$ 且與 $t$ 同片段「直接」；$j$ 在先前片段「經由記憶」（$\mathcal M_{t-1}$ 已以先前片段的 $y$ 更新，Eq. 24）；$j>t$「不可見」。
  - MAG：持久記憶「直接」（SW-Attn$^*$ 帶前綴）；$t-w<j\le t$「直接」；$j\le t-w$「經由記憶」（記憶分支 $\mathcal M(\tilde x)$ 處理過 $j\le t$，再以閘控結合）；$j>t$ 不可見。
  - MAL：$t-w<j\le t$「直接」（注意的是記憶層輸出 $y_j$）；持久記憶與 $j\le t-w$「經由記憶」（Eq. 31 不帶前綴，持久記憶只經由記憶層）；$j>t$ 不可見。
  - 注意力配對數（所有 query 加總，只計「直接」）：MAC $\sum_t(N_p+N_l+((t-1)\bmod C)+1)$；MAG $\sum_t(N_p+\min(t,w))$；MAL $\sum_t\min(t,w)$；對照「完整因果注意力＋前綴」$\sum_t(N_p+t)$。
- **顯示**：矩陣 16 列（query）×（$N_p$ + MAC 時 $N_l$ + 16）行。三種格子以填色＋圖樣區分：直接 = `--accent` 實心；經由記憶 = `--c-par` 斜線圖樣（SVG `<pattern>`）；不可見 = 空白細框。欄群組上方標「持久記憶」「記憶 token $h$（本片段）」「序列」。目前 query 列加粗外框；其餘列以 40% 不透明度顯示。右側（窄螢幕在下方）：三類格數、配對數與完整注意力的比值、一句該變體的資料流（例：MAC「讀記憶 → 片段注意力 → 以 $y_t$ 寫記憶」）。token 軸記號在矩陣上方重複。
  - **尺寸**：格寬 = `max(14px, floor(容器寬 / 欄數))`，上限 28px；最多 24 欄（$N_p$ 4 + $N_l$ 4 + 16）時 390px 寬下為 14px，矩陣總寬 336px＋列標籤。若仍超出，矩陣容器 `overflow-x:auto`、`tabindex="0"`、`aria-label="注意力矩陣（可橫向捲動）"`，頁面本身不得出現橫向捲動。點選目標是整列（列高 ≥ 24px），不是單格，所以觸控不受格寬限制。
- **讀者應觀察到**（預設值）：
  1. $t=10$：MAC 直接 6、經由記憶 8、不可見 6；MAG 6／6／6；MAL 4／8／6。
  2. 配對數：MAC 104、MAG 90、MAL 58，完整因果注意力＋前綴 168；$N$ 增大時 MAC 與 MAG 只隨 $N$ 線性成長〔本頁推導〕。
  3. 片段開頭的 token（例如 MAC 的 $t=9$）直接注意的範圍最小，較早的資訊全靠記憶 token $h$。
- **頁面註記**（R4）：Eq. 21 的 $h_t$ 長度等於片段長度，Fig. 3a 說明寫「$N_l$ 個長期記憶 token」；本頁依 Fig. 3a 把 $N_l$ 當作獨立參數，且讓片段內所有 query 都能看到 $h$（推定）。
- **無障礙**：矩陣為 `role="grid"`。鍵盤用 roving tabindex：只有目前 query 列 `tabindex="0"`，其餘列 `-1`；上下方向鍵與 Home／End 移動目前列並同步 $t$ 滑桿。只有目前列的格子帶 `aria-label`（「key j：直接／經由記憶／不可見」），列本身 `aria-label`「query t：直接 a、經由記憶 b、不可見 c」；非目前列的格子 `aria-hidden="true"`，避免螢幕閱讀器逐一讀出 384 格。點選與鍵盤皆可，無 hover 依賴。

---

## 5. 驗收數值表

來源：`verify/accept_node.txt`，由 `cd verify && node widgets.js > accept_node.txt` 產生（本輪已把 C1 步長對照、多 seed 摘要、C2 極端值與衰減、C2 參考線（格點值與連續期望值並列）、C3 發散檢查加進 `widgets.js`，重新產生後檔案與腳本輸出逐行相同）。Python 版（`PYTHONPATH=. python3 c*.py`）逐項一致到小數點後 4 位。頁面使用相同演算法與 seed，容許誤差寫在各表下方。

**C1**（seed 11；讀數為 4 位小數；「—」表示發散後不顯示）

| 設定 θ／η／α | 平均驚訝度 33–40 | 平均驚訝度 65–72 | $E_A(32)$ | $E_B(64)$ | $E_A(64)$ | $E_A(96)$ | $\|S_{40}\|$ | 發散 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3／0／0（η 0、α 0） | 0.7419 | 0.6049 | 0.0160 | 0.0182 | 0.2927 | 0.0118 | 0.1015 | 否 |
| 0.3／0／0.02（η 0、α 0.02） | 0.6848 | 0.6949 | 0.0306 | 0.0251 | 0.4167 | 0.0333 | 0.0917 | 否 |
| 0.3／0.6／0.02（預設） | 0.6388 | 0.6315 | 0.0142 | 0.0084 | 0.4571 | 0.0303 | 0.1518 | 否 |
| 0.3／0.6／0 | 0.6627 | 0.5924 | 0.0085 | 0.0051 | 0.3872 | 0.0109 | 0.1573 | 否 |
| 0.3／0.6／0.1 | 0.5875 | 0.6958 | 0.0498 | 0.0494 | 0.5816 | 0.0787 | 0.1482 | 否 |
| 0.3／0.9／0.02 | 0.7400 | 0.9681 | 0.2126 | 0.4188 | 0.9215 | 2.6196 | 0.4433 | 否 |
| 0.75／0／0.02（步長對照） | — | — | 0.0133 | 0.0106 | 0.4554 | 0.0235 | 0.0521 | 否 |
| 1／0.95／0 | 6.8568 | —（第 71 步停止） | 57.6689 | 97817.3693（上緣箭頭） | 74360.0851（上緣箭頭） | — | 16.9011 | 第 71 步 |

$\|g_1\|=1.0973$（所有設定）。步長對照列的平均驚訝度不列入驗收（腳本未輸出）。多 seed 摘要（seed 11–30，$E_B(64)$ 中位數）：η 0 為 0.0352、θ 0.75／η 0 為 0.0083、預設 η 0.6 為 0.0087；預設低於 η 0 者 20／20，低於步長對照者 14／20。容許：±0.0005。

**C2**（MSE，seed 5／9）

| 設定 | $L_{\mathcal M}$（參數數） | t = 100 | 300 | 1000 | 3000 |
| --- | --- | --- | --- | --- | --- |
| θ 0.01、η 0.9、α 0（預設） | 1（1） | 0.43364 | 0.47573 | 0.47698 | 0.45461 |
| 同上 | 2（48） | 0.39953 | 0.30957 | 0.05388 | 0.00235 |
| 同上 | 3（320） | 0.35577 | 0.08919 | 0.01137 | 0.00010 |
| 同上 | 4（592） | 0.31611 | 0.06345 | 0.00061 | 0.00014 |
| θ 0.05、η 0、α 0 | 1／2／3／4 | 0.41212／0.42364／0.39174／0.36170 | 0.41961／0.36587／0.27396／0.12016 | 0.40995／0.02073／0.01256／0.00123 | 0.43184／0.00097／0.00115／0.00027 |

參考線（`accept_node.txt` 的「C2 ref」行，201 點格點）：最佳無偏置線性 $w=0.1660$、MSE 0.4018；記憶為 0 時 0.4389。同一行並列 $U[-2,2]$ 連續期望值（閉式）$w=0.1742$、MSE 0.3977、記憶為 0 時 0.4382，僅供對照，不作驗收值；頁面與驗收一律用格點值。極端值：θ 0.05、η 0.95、α 0 時 $L=2$ 於第 103 步、$L=4$ 於第 58 步發散（顯示發散訊息）；同設定 α 0.01 時 $L=2$ 於第 2082 步發散，其餘深度不發散；θ 0、η 0、α 0.01 時四種深度在 3000 步皆為 0.4389。容許：相對 ±1%（MLP 經 3000 步，跨引擎 exp 可能有 1 ulp 差）。

**C3**（θ 0.3、α 0.02）

| η | b | 順序步數 | 兩形式最大差 | $E_A(32)$ | 情境 B 平均 $E_B$ | $E_A(96)$ | $\|\mathcal M^{(b)}_{96}-\mathcal M^{(1)}_{96}\|_F$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6 | 1 | 96 | 3e-16 | 0.0142 | 0.0672 | 0.0303 | 0 |
| 0.6 | 2 | 48 | 4e-16 | 0.0142 | 0.0762 | 0.0416 | 0.1302 |
| 0.6 | 4（預設） | 24 | 1e-15 | 0.0192 | 0.1173 | 0.1420 | 0.7140 |
| 0.6 | 8 | 12 | 8e-16 | 0.1935 | 0.1507 | 0.6000 | 2.0515 |
| 0.6 | 16 | 6 | 3e-14 | 28.3908 | 53.5064 | 8880.80（上緣箭頭） | 201.86 |
| 0.6 | 32 | 3 | 2e-14 | 15.0235 | 92.2950 | 3685.54（上緣箭頭） | 142.82 |
| 0 | 1 | 96 | 1e-16 | 0.0306 | 0.1021 | 0.0333 | 0 |
| 0 | 4 | 24 | 7e-16 | 0.0290 | 0.0926 | 0.0457 | 0.2116 |
| 0 | 8 | 12 | 6e-16 | 0.0315 | 0.0942 | 0.0500 | 0.2136 |
| 0 | 16 | 6 | 7e-16 | 0.5026 | 0.3163 | 0.3727 | 1.3287 |
| 0 | 32 | 3 | 3e-15 | 1.5290 | 2.1024 | 15.3085 | 9.5585 |

（b = 16、32 時參數未超過 $10^3$，不觸發停止；誤差超過縱軸上限時畫上緣箭頭。）容許：兩形式最大差 $<10^{-12}$；其餘 ±0.0005（大於 100 的值 ±0.1%）。

**C4**（N = 16、$N_p$ = 2、C = 4、$N_l$ = 2、w = 4）

| 變體 | t = 10 直接／記憶／不可見 | t = 4 直接／記憶／不可見 | 配對數 |
| --- | --- | --- | --- |
| MAC | 6／8／6 | 8／0／12 | 104 |
| MAG | 6／6／6 | 6／0／12 | 90 |
| MAL | 4／8／6 | 4／2／12 | 58 |
| 完整因果＋前綴 | — | — | 168 |

整數，必須完全相同。

---

## 6. 實作計畫

### 6.1 檔案與沿用

- 單一檔案 `tut/243/build/index.html`，CSS／JS／時間線資料全部內嵌，不 `fetch` 本機檔案；上線時複製為 `tutorials/titans-test-time-memory.html`。
- **完全沿用** `tutorials/prediction-targets-k-diff.html` 的頁面外殼，不重新設計：`<head>`（Google Fonts 同一串、KaTeX 0.16.9 cdnjs 三檔 `defer`）、`:root` tokens 與深色模式三段覆寫（含 `--c-par`、`--c-perp` 與調整後的深色 `--ink-3` #9C978C）、`.shell` 兩欄格線與 sticky 章節目錄（< 1024px 改頂部橫列）、appbar 與主題切換（`pm-theme`，try/catch）、`#pm-bar` 導覽、`.plate`／`.plate-h`／`.plate-lead`／`.plate-f` 圖版、`.callout`、`.facts`、`.eq`＋`.eq-n` 方程式編號、`.check` `<details>` 檢查題、`.tscroll` 橫捲表格與 `table.stack` 窄螢幕堆疊、`.seg` 分段按鈕、`.chip`、`.rng` 滑桿、`.readout`、`.algo` 程式碼兩欄對照、時間線卡片列與 `renderTimeline`、`el`／`scaleLin`／`scaleLog`／`linePath` SVG 輔助、`?test` 的 `selfTest()` 框架、`renderMathInElement` 設定、reduced-motion 處理、「本頁推導」pill 標籤。
- 以下只列**新增**的部分。

### 6.2 新增內容

- `<title>`：「Titans 測試時記憶」；`<meta name="description">`：「圖解 Behrouz、Zhong、Mirrokni 的 Titans：以梯度（驚訝度）、動量與遺忘在測試時更新的深層神經記憶，chunk 平行化訓練，以及 MAC、MAG、MAL 三種架構。」
- Hero token 軸 SVG（靜態，viewBox 寬 720）：16 個 token 方格；前 2 格標「持久記憶」（`--ink-3` 斜線圖樣）；最後 4 格上方帶狀「注意力視窗」（`--accent` 外框）；其餘上方帶狀「長期記憶（已壓縮）」（`--c-par` 斜線）。窄螢幕（< 480px）改為 8 格。SVG 下方一行 HTML 圖例（三個色塊＋文字，文字連到 `#persistent`、`#variants`、`#surprise`），圖例不放在 SVG 內，以便縮放與點選。C4 的配色與圖樣與此一致。
- S1 比較表、S2 內外迴圈兩欄表、S7 三欄變體卡（每張：方程式、資料流一句、注意力類型）、S8 對照表（列：DeltaNet、Gated DeltaNet、Longhorn、TTT、LMM；欄：遺忘閘、動量、深層記憶、遞迴型態；只填論文 App. C 寫明的內容，其餘填「論文未比較」）、S9 三張表與一致性註腳、S10 訓練設定小表。
- S5 的 `.algo` 兩欄：左欄 chunk 演算法虛擬碼（逐行），右欄中文說明：
  ```
  for each chunk [t'+1 .. t'+b]:
    E = M_{t'} K - V            # 所有誤差一次算出（矩陣乘法）
    U_l = E[:,l] K[:,l]^T       # 各 token 的梯度，都在 M_{t'} 上
    S_t = η S_{t-1} - θ U_t     # 線性遞迴，可用 associative scan
    M_t = (1-α) M_{t-1} + S_t   # 衰減權重直接連乘，不用 β_t/β_i
  ```
- C1–C4：從 `verify/widgets.js` 複製 `mulberry32`、`gauss`、`makeStream`、`mv`、`runC1`、`loopEnds`、`matmulEnds`、`initMLP`、`fwd`、`step`、`evalMSE`、`runDeep`、`maskRow`、`counts`、`pairs`，不改演算法；每個元件一個 `initC1()`…`initC4()`，只操作自己的容器。C2 的 `runDeep` 改為可分段（保存 `P,S,r,t` 狀態，每影格 30 步）；「比較四種深度」直接呼叫原版。C3 與 C1 共用 `makeStream` 的結果（同 seed 時快取）。
- SVG `<pattern id="hatch-mem">`（斜線 45°、`--c-par`）與 `<pattern id="hatch-persist">`（`--ink-3`），在 C4 與 hero 共用；深色模式下由 CSS 變數控制 stroke。
- `selfTest()` 新增斷言（`?test`）：
  - `runC1()` 預設：`out[63].EB` = 0.0084、`out[63].EA` = 0.4571（±0.0005）；`runC1({theta:1,eta:.95,alpha:0}).diverged === 71`。
  - `runC1({theta:.75,eta:0}).out[63].EB` = 0.0106（±0.0005，步長對照）。
  - `loopEnds` 與 `matmulEnds` 在 b = 4、η = 0.6 與 b = 4、η = 0（檢查 $0^0=1$）的最大差皆 < 1e-12；`runC1({chunk:8}).out[95].EA` = 0.6000（±0.0005）。
  - `runDeep({L:1}).res[3000]` = 0.45461、`runDeep({L:3}).res[3000]` = 0.00010（相對 ±1%，0.00010 用絕對 ±2e-5）；`runDeep({L:2,theta:.05,eta:.95}).div === 103`；`runDeep({L:2,theta:.05,eta:.95,alpha:.01}).div === 2082`。
  - C2 參考線（以 `evalMSE` 同一 201 點格點計算）：最佳線性 $w$ = 0.1660、MSE = 0.4018，記憶為 0 的 MSE = 0.4389（±0.0001）；三者與 `accept_node.txt`「C2 ref」行的格點值相同，不是連續期望值 0.1742／0.3977／0.4382。
  - `scaleLog` 對 0、負數、NaN 輸入回傳有限值（下緣），C1 θ = 0 時圖的 `d` 屬性不含 `NaN`。
  - `counts('MAC',{N:16,Np:2,C:4,Nl:2,w:4,t:10})` = {6,8,6}；`pairs` 三變體 = 104／90／58。
  - `document.documentElement.scrollWidth <= innerWidth`。
  - 輸出 `selfTest: N passed, M failed`。

### 6.3 連結

- 導覽列 `#pm-bar`：「← Paper Master」→ `../index.html`；「教學與筆記」→ `index.html`。
- Hero「在關聯圖中查看」→ `../index.html#paper-243`；arXiv `https://arxiv.org/abs/2501.00663`。
- 筆記 → `<slug>.html`（同在 `tutorials/`）。

### 6.4 數學排版

- 公式右側以等寬字標論文編號（「Eq. 13」）。所有 `.eq` 預設 `overflow-x:auto; tabindex="0"`（附 `aria-label`「方程式 Eq. n，可橫向捲動」），保證任何公式都不撐出頁寬。另外在 < 600px 時手動拆行最寬的四條：Eq. 16、Eq. 9–10（含 underbrace）、Eq. 21–25、步驟 7 的含動量 chunk 公式（每個等號或分號後換行，用 KaTeX `aligned`）。Eq. 8、13–14、17 靠橫捲處理。SVG 內文字、讀數一律用 Unicode（「‖S‖」「θ = 0.30」），不用 TeX。
- 第 1 節以外不得新增論文數字；頁面中所有論文數字附出處。

### 6.5 引用與標籤

- 三種陳述分開寫：論文結果；作者的解釋（「作者認為」「作者歸因於」）；本頁推導（「本頁推導」pill＋一行依據）。推定值用「推定」pill（同樣式，文字不同）。
- 頁面「本頁推導」項目：Eq. 13–14 的逐 token 展開、Eq. 16 第二個等號的條件與 $b=1$ 等價、Eq. 17 一般形式、含動量的 chunk 公式、β 比值的數值注意、η = 0 的 DeltaNet 形式、C3 的「b 大 ≈ θ 放大」說明、C4 配對數的線性成長、Table 5 與 Table 1／2 的對應、Table 1 Avg 不一致、$t'$ 的讀法、associative scan 的兩句說明、C1 的動量／步長對照。
- 「本頁核對」項目（同 pill 樣式，文字「本頁核對」）：Table 1 與作者宣稱不一致的格子、Table 2 16K 的 MAG／MAC、Table 3 Traffic MAE、Table 5 排序所依的指標。

### 6.6 研究脈絡與延伸閱讀（notes 已產生）

- 來源：`tut/243/notes/timeline.json`。建置時原樣貼進 `<script type="application/json" id="timeline-data">`，格式與 238 相同（`year`、`title`、`authors_short`、`arxiv`、`role`、`source`）；不手改條目。檔案不存在或解析失敗時整節顯示「時間線資料尚未提供」並隱藏控制鈕（沿用 238 的處理）。
- `timeline.json` 實際有 13 條（2014–2024）。分類常數 `TL_GROUP`（以 arXiv ID 為 key，不在表中的歸「其他」）：
  - 外部與分段記憶：1410.5401（NTM）、1901.02860（Transformer-XL）、2207.06881（RMT）、2412.09764（Memory Layers）
  - 注意力：1706.03762（Transformer）
  - 線性注意力與快速權重：2006.16236、2102.11174
  - 閘控線性遞迴：2312.00752（Mamba）、2402.19427（Griffin）
  - Delta rule：2406.06484（DeltaNet）、2412.06464（Gated DeltaNet）
  - 梯度式更新：2407.04620（TTT）
  - 本文：2501.00663
  時間線沒有 Longhorn、Mamba2、RetNet；S8、S11 文字提到它們時不加時間線錨點。本文條目的 year 為 2024（arXiv 提交日 2024-12-31），照錄。
- 延伸閱讀清單：hub `titans-lineage-overview` 在前，topic 依 hub 的閱讀順序：`memory-view-of-sequence-models`、`linear-attention-fast-weights`、`gated-linear-recurrence-forgetting`、`delta-rule-deltanet-family`、`recurrent-memory-segment-transformers`、`titans-neural-memory-surprise-update`、`titans-architectures-mac-mag-mal`（標題取各檔 frontmatter 的 `title`）；再列站內既有的 `ttt-sequence-modeling-overview`、`ttt-linear-mlp`、`lact-large-chunk-ttt`、`ttt-theory-and-applications`、`test-time-training-naming`（重複者只列一次）。

### 6.7 文案

- 繁體中文、台灣用語、字面學術語氣；不用比喻與宣傳形容詞。論文的人類記憶類比一律寫成「作者以……為動機」。
- 用詞：「驚訝度」（surprise）、「遺忘閘」（forgetting gate）、「持久記憶」（persistent memory）、「長期記憶模組」（LMM）、「片段」（segment）、「chunk」保留英文；專有名詞第一次出現附英文。
- 「測試時」= 推論時，第一次出現（hero 副標下一行）時說明：模型部署後處理新序列時，記憶模組的權重仍在更新。

### 6.8 建置後自我檢查

沿用 238 的 5.10：390／768／1024／1440 截圖、深色、鍵盤 focus、reduced-motion；另加：(a) 深色模式下 `--c-par`、`--c-perp`、`--accent` 對 `--bg` 與對情境帶 `--band-b` 的非文字對比 ≥ 3:1（以 WCAG 公式計算並記錄數值，不足時調整深色值）；(b) 390px 下 C4 在 $N_p=N_l=4$ 時頁面無橫向捲動；(c) 每個 `.eq` 在 390px 不撐出頁寬；(d) 螢幕閱讀器（VoiceOver 或 NVDA）走一遍 C4 grid 與 C1 資料表；八項逐項評分直到沒有可見改進；`?test` 全數通過；逐條核對 12 個學習目標的參考答案確實寫在對應章節內文。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 論文 v1 的筆誤 | Eq. 7 下方 read／write 文字對調；Eq. 32 右邊 $\mathcal M_t$ 應為 $\mathcal M_{t-1}$；Eq. 17 key = value = $x$ 且省略係數 2；「對輸入的梯度」與 Eq. 8 的參數梯度不符；Eq. 10、14 混用 $M$／$\mathcal M$；Eq. 16 的 $t'=t-\mathrm{mod}(t,b)$ 與下一句「$t=b$ 所以 $t'=0$」不符（照公式 $t'=b$，且 $b=1$ 時梯度點變成 $\mathcal M_t$） | 頁面照錄方程式，於旁註與常見錯誤說明；不靜默修正。$t'$ 採 chunk 起點讀法 $t-1-\mathrm{mod}(t-1,b)$，$b=1$ 等價只在此讀法下成立 |
| R2 | 表格不一致 | Table 1 三列 Avg 與平均不符；760M MAL SIQA 30.98 疑似筆誤；Table 5 Long Context 數值等於 S-NIAH 平均，正文稱 BABILong；Fig. 7b 寫 360M；760M Titans 列無星號 | 照錄原數字，以註腳列出 `tables.py` 的檢查結果，標「本頁推導」 |
| R3 | 未寫明的超參數 | chunk 大小、MLP 寬度與偏置、$\eta_t/\theta_t/\alpha_t$ 的參數化、$C$、視窗、$N_p$、$N_l$、閘控 $\sigma$、層數 | 元件用推定值並標「推定」；S10 第 8 條列出 |
| R4 | MAC 的 $h_t$ 長度 | Eq. 21 的 $h_t$ 長度等於片段長度（逐 token 讀取），Fig. 3a 說明寫 $N_l$ 個 token；若 $h_t$ 逐 token 讀取且對片段內所有 query 可見，前面的 query 可能看到後面 token 的資訊，論文未討論 | C4 依 Fig. 3a 把 $N_l$ 當獨立參數並標「推定」；頁面只陳述兩處寫法不同，不推論實作 |
| R5 | MAL 的前綴 | Eq. 31 是 SW-Attn（無星號），Eq. 27 是 SW-Attn$^*$ | C4 依方程式；頁面註明依據 |
| R6 | 只有圖的結果 | BABILong（Fig. 6）、深度（Fig. 7–8）、吞吐量（Fig. 9）沒有正文數值；2M 上下文只在 Abstract／Sec. 1／Sec. 6 | 只寫文字結論，不估讀圖值 |
| R7 | Theorem 4.1 | v1 沒有證明 | 寫「論文宣稱」，不設學習目標 |
| R8 | C1 只呈現遺忘的代價 | 線性 delta 規則下，遺忘在玩具中只降低保留；作者論證的好處（容量管理）需要長序列 | 頁面明寫，並引 Table 5（w/o weight decay 最差） |
| R9 | C3 大 chunk 發散 | 固定 θ 下 $b\ge16$ 誤差暴增是玩具設定造成，不代表論文 | 圖下說明（〔本頁推導〕），並註明論文 θ 依 token 而定、未給 $b$ |
| R10 | 跨引擎浮點 | C2 的 MLP 經 3000 步，V8 與 libm 的 exp 可能差 1 ulp | 已以 node 與 Python 比對（4 位一致）；selfTest 用相對 ±1% |
| R11 | 研究筆記（已解決） | 第一版撰寫時 `tut/243/notes/` 不存在；本輪已產生 8 篇筆記與 13 條時間線 | slug 已填進第 3 節各章與 6.6；保留缺檔時顯示「時間線資料尚未提供」的處理。筆記中的外部論文內容都標「依 Titans 的描述」，頁面引用時沿用此限定 |
| R12 | 論文版本 | 後續版本可能更新數字（v1 註腳預告） | 頁首與引用註明「依 arXiv v1」 |
| R13 | Gated DeltaNet 對應的細節 | 本頁只由 Eq. 13–14 與 Eq. 34 推出 η = 0 的形式 $\mathcal M((1-\alpha)I-\theta kk^\top)+\theta vk^\top$；Gated DeltaNet 原論文的遞迴不在本論文內，若其衰減也乘在 delta 修正項上（審查者的記憶，未核對），兩者無法重新參數化互換 | 頁面寫「論文稱等價；本頁只確認更新規則形式相近，且 η = α = 0 時即 DeltaNet」；LO11、步驟 10、C1 chip 小字一致。**未決**：建置前若能查 arXiv:2412.06464 的遞迴式，可把「形式相近」改成精確敘述；本設計不依賴外部論文 |
| R14 | 符號衝突 | α（遺忘）、$t$（token 索引）、M（記憶／遮罩）與站內擴散教學的用法不同 | 符號表明列；記憶一律 $\mathcal M$ |
| R15 | C1 的動量效果與步長混淆 | 常數 η 使有效步長變為 θ/(1−η)；θ = 0.75、η = 0 的 $E_B(64)$（0.0106）接近預設（0.0084），seed 11–30 中位數 0.0083 對 0.0087 | 加步長對照 chip 與驗收列；頁面明寫「誤差下降大多是步長效果，動量在本元件只以 $\|S_t\|$ 延續呈現」；所有 C1 數字標 seed 11。未另設計能分離動量好處的玩具（需要非平穩或稀疏梯度，超出本頁範圍） |
| R16 | 作者宣稱與表格不完全一致 | Sec. 5.2「三種 Titans 都優於…、MAG/MAC 優於 MAL」、Sec. 5.6「outperforming all baselines」、Sec. 5.9 的元件排序，各有反例（步驟 11） | 宣稱標「作者宣稱」，反例逐條列在表下並標「本頁核對」；不改原數字 |
| R17 | 頁面變長 | S1 加前置說明、S3 分兩小節、S9 加例外清單，總字數約增加 300 字 | 新增內容多放在預設收合的 `<details>`（φ、LTI、C1 全部讀數）；正文只加必要句子 |

---

## 8. 修訂紀錄

- **Round 1**（審查清單：`reviews/round1.md`，26 項，全部處理；需要外部資料才能完成的部分記在 R13、R15）
  - 正確性：η = 0 與 Gated DeltaNet 改為「論文稱等價、本頁只確認形式相近」（步驟 10、LO11、R13）；$t'$ 改採 chunk 起點讀法並列入 R1，第二個等號限定在單一 chunk 內（步驟 7、LO8、常見錯誤 9）；含動量 chunk 公式統一 1-based、$q_l=-\theta\eta^{n-l}$、$0^0=1$；C2 觀察 2 的數字矛盾已修正；C1 係數 ½ 對照 Eq. 12。
  - 結果敘述：Table 1 作者宣稱與例外清單（含 760M MAC/GDN-H2 Wiki、MAL Avg、SIQA 30.98）；Table 2 MAG N 欄；Table 3 Traffic MAE（核對後是 iTransformer 0.282，不是審查所寫的 RLinear）；Table 5 排序註明指標（LO10、LO12、R16）。
  - C1：加入「θ = 0.75、η = 0」步長對照 chip 與驗收列、seed 11–30 摘要；LO6 改寫為「誤差下降大多是步長效果」；chip 改以參數命名；顯示目前 seed（R15）。
  - 驗算：`widgets.js` 加入步長對照、多 seed、C2 極端值與衰減、C3 發散區塊，`accept_node.txt` 由腳本重新產生；文件寫明 `PYTHONPATH=.` 指令；修正 `widgets.js` 中 C2／C3 對調的區塊註解。
  - 教學與建置：S1 前置說明、φ 收合、Q1–Q5 移到 S2 末；S3 分 3a／3b、讀數收合；S5 加 associative scan 兩句、LTI 收合；hero 定義 perplexity 與「測試時」、token 軸圖例；TC⁰、Frobenius 範數、ppl 進符號表或內文；C4 格寬、橫捲、roving tabindex、$N_l$ 停用；資料表切換；`scaleLog` 下限規則；發散訊息 `role="alert"`；C2 只在 `change` 重設、`visibilitychange` 暫停；深色情境帶不透明度與 6.8 對比檢查；`.eq` 一律可橫捲並手動拆行四條。
  - 研究筆記：`notes/` 的 8 篇筆記與 13 條時間線已填入第 3 節延伸閱讀、S8（App. A.1 三代分類）、S11 與 6.6 的 `TL_GROUP`；R11 標為已解決。
- **Readiness 修正**（Goldfish NO-GO 的兩項阻擋）
  - C1 chip 小字：「前者對應 TTT 的更新規則」改為「前者即 DeltaNet 的更新規則（Eq. 34）」，與步驟 10、LO11 的推導一致；TTT 差異只放在 S8。
  - C2 參考線：0.4018／0.1660／0.4389 是 201 點格點（含端點、等權）上的值，與評估用 `evalMSE` 同一格點；連續 $U[-2,2]$ 期望值為 0.3977／0.1742／0.4382（審查者手算的 0.398／0.174／0.438 即此）。`widgets.js` 新增「C2 ref」行並列兩者，`accept_node.txt` 已重新產生；第 4 節 C2（右圖、觀察 1）、第 5 節 C2 參考線與 6.2 `selfTest()` 已寫明採格點值並新增參考線斷言。
- **Build 前修正**（Readiness GO 的非阻擋意見）
  - 1.1 規模補上 170M；C2 寫明 $L_{\mathcal M}=1$ 的 $w$ 為隨機初始；步驟 7 寫出 $B_b$；LO6 與 C1 觀察 4 標明 η = 0.6。
  - `notes/delta-rule-deltanet-family.md` 的「形式相同」改為「論文稱兩者等價；本頁只確認形式相近」，與 S8、R13 一致。
- **無背景讀者測試修正**（LO4 partial）
  - 讀者只答出「動量讓驚訝度延續」，沒有連到動機。S3 3a 在 Eq. 9–10 之後的段首補一句因果：只用 Eq. 8 時更新量等於當下梯度，大驚訝後梯度變小、記憶停在平坦區域而寫不進後續資訊；$\eta_tS_{t-1}$ 在當下梯度接近 0 時仍推動 $\mathcal M_t$ 更新，所以仍能吸收後續資訊。
