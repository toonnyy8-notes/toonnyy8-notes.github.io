# 設計文件：〈Elastic Token Compression for Pixel-Space Diffusion Transformers〉互動教學

- 論文：Eduard Zamfir、Christian Reisswig、Zongwei Wu、Yongqin Xian、Radu Timofte，arXiv 2608.29281（v1，2026-08-29）。資料庫編號 No. 217。方法名 RTI（Region Token Interface）。
- 原文：`tut/217/src/paper.txt`（PDF 同目錄）。本文件的「Eq. n」「Sec. n」「App. X」「Fig. n」「Tab. n」皆指論文原文編號（Eq. 1–4 在 Sec. 3，Eq. 5–6 在 App. C）。
- 產出：`tut/217/build/index.html`（單一 HTML 檔）。上線位置：`tutorials/rti-elastic-token-compression.html`（slug `rti-elastic-token-compression`）。所有站內相對連結以上線位置為準。
- 驗算腳本：`tut/217/verify/common.py`（與頁面相同的演算法）、`c1_orders.py`、`c1_table.py`、`c2_partition.py`、`c3_read.py`、`c4_cost.py`；輸出存於 `verify/out_*.txt`；腳本可用 `python3 -I <script>` 直接執行（自行把腳本目錄加入 `sys.path`）。第 5 節驗收表全部出自這些輸出。
- 研究筆記：`tut/217/notes/`：總覽 `rti-lineage-hub` + 6 篇主題筆記 + `timeline.json`（12 筆）。系列筆記在 `tut/series-token-compression/notes/`（總覽 `tokc-lineage-hub`）。連結見 6.2。
- 同批次：系列教學 `token-compression-lineage.html`（ToMe、diffusion 中的 token merging、hourglass/U 形階層 Transformer、像素空間 DiT、自適應 tokenization），姊妹教學 HiT `hit-hierarchical-diffusion-transformer.html`（No. 585，`../index.html#paper-585`）。共同背景一律連過去，本頁只講 RTI 新增的部分。
- 標記慣例：〔本頁推導〕＝本設計自行推導、論文沒有寫出；「推定」＝為了讓元件可算而選定、論文沒有寫明的值。頁面沿用同樣標籤。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

擴散 Transformer（DiT）把影像切成規則的 patch 格，每個 patch 一個 token，每個 patch 在每一層、每個取樣步的計算量都相同。像素空間模型沒有 VAE 先移除低階空間冗餘，浪費最大（Sec. 1）。作者探測凍結的像素文字到影像模型 MiniT2I-B/16，發現冗餘在空間上集中於平坦區域、在時間上隨影像成形而增加、在深度上集中於中段區塊（Fig. 2）。這個形狀推出四個要求：分組必須**連通**、大小**隨內容**、每組要有**位置**、冗餘內容要**匯總而非刪除**（Sec. 1）。作者用 Hilbert 曲線排列 patch：曲線上相鄰位置在影像中一定相鄰，所以任何連續段都是連通區域，二維分組變成一維切割。RTI 在相鄰特徵距離最大的 $R-1$ 處切開（Eq. 1–2），以 READ 把每段軟匯集成一個區域 token（Eq. 3），只讓中段核心區塊處理這 $R$ 個 token，再以 WRITE 散回每個 patch（Eq. 4）。主幹凍結、只訓練 LoRA 與介面，訓練時每步隨機抽 $R$，所以一個 checkpoint 適用訓練範圍（64–512）內的預算。B/16 在 $R=512$ 以 1.84× 速度得到 GenEval 87.8（dense 87.2），$R=256$ 以 2.15× 得到 84.7（Tab. 1）；摘要寫「在 2.0× 速度追平 dense，在 2.6× 仍接近」。

### 1.2 研究動機（Sec. 1）

1. 自然影像的細節集中在少數區域，擴散取樣後段才解析細節；DiT 對每個位置、每個取樣步分配相同計算量。
2. 潛空間模型由 VAE 先移除空間冗餘；像素空間模型（JiT 等）的每個區塊都要處理這些冗餘。
3. 既有縮減方法各失去一項性質：相似度合併（ToMe 系列）的組可能來自畫面兩端，失去連通與位置；潛變數瓶頸（Perceiver-IO、RIN、ELIT）同時放棄分組與位置；條件計算（Mixture-of-Depths 等）刪除了應該匯總的內容。

### 1.3 貢獻（Sec. 1 列出三點）

1. 量測凍結像素擴散 Transformer 的冗餘與其形狀（Fig. 2、App. C.1）。
2. RTI：彈性、Hilbert 排序的區域介面，適配凍結主幹（Sec. 3）。
3. RTI 以兩倍吞吐量追平 dense 基準，領先其他縮減方法（Sec. 4）。

結論（Sec. 5）另強調：「新的不是機制，而是觀察到：在 patch 上訓練好的模型，其 token 單位可以換成區域。」

### 1.4 從作者的角度逐步拆解

#### 符號表（設計文件用的總表；頁面不放單一總表，而在 S1、S2、S5、S6、S7、S8、S11 開頭各放「本節符號」小框，只列該節首次出現的符號）

名稱衝突的處理：格邊長一律寫 $q$（不用 $n$）；成本寫 $\rho$（不用 $c$）；長度比寫 $\nu=n/N$（不用 $r$）；注意力比重寫 $\eta$。$c$ 只指區域，$r$ 只作索引。

| 符號 | 意義 |
| --- | --- |
| $N$ | patch token 數；512²、patch 16 時 $32\times32=1024$（Sec. 3.1） |
| $R$ | 預算：區域 token 數，$R\ll N$ |
| $x_t$、$\hat x$ | 取樣狀態（在噪聲與乾淨影像間內插，rectified flow）；網路預測的乾淨影像 |
| $[s_0,s_1]$ | 核心區塊區間；B/16 為 $[3,13]$（17 塊中），L/16 為 $[3,19]$（23 塊中） |
| $h_j^{(s_0)}\in\mathbb R^d$ | 進入核心時 patch $j$ 的特徵 |
| $\pi$、$\pi(j)$ | Hilbert 曲線在 32×32 格上誘導的順序；第 $j$ 個位置的 patch |
| $d_j$ | 曲線上相鄰兩 patch 的特徵距離（Eq. 1） |
| $C$ | 切點集合（Eq. 2） |
| $c$、$\lvert c\rvert$、$c(j)$ | 一個區域、其 patch 數、patch $j$ 所屬區域 |
| $\alpha_j$、$w$ | READ 的區域內 softmax 權重、單一可學習評分向量 $w\in\mathbb R^d$ |
| $z_c$、$z'_c$ | 區域 token（進核心前）與經過核心後的值 |
| $e_{\text{size}}[\cdot]$ | 尺寸嵌入，索引 $\lfloor\log_2\lvert c\rvert\rfloor$ |
| $g$ | WRITE 的小型可學習映射；$[\cdot\Vert\cdot]$ 為串接 |
| EV | 保留變異（Eq. 5）；$\bar h_{c(j)}$ 組平均，$\bar h$ 全域平均 |
| spread | 組內 patch 到組中心的平均距離（Eq. 6），單位為格 |

#### 步驟 1：基礎模型與成本來源（Sec. 3.1）

- MiniT2I-B/16：像素空間（JiT 系列）文字到影像 DiT，雙串流主幹 17 塊，影像 token 與凍結的 FLAN-T5 文字 token 並行。512²、patch 16 → $N=1024$。以 rectified flow 訓練，預測乾淨影像 $\hat x$ 而非噪聲殘差（沿用 JiT）。
- RTI 只縮短影像 token 序列；文字串流永遠全長（Sec. 3.2）。
- 頁面 S1 給初學者的最少背景：DiT 由一串區塊（block，每塊含注意力與 MLP）串接，MiniT2I-B/16 有 17 塊；一個取樣步＝整個網路一次前向計算，生成要跑很多步（本文 100 步）；DiT 把 patch 當 token、注意力成本隨 token 數平方成長、線性層隨 token 數線性成長；像素空間與潛空間的差別連到 `pixel-space-diffusion.html`、`x-prediction-manifold-view.html` 與系列教學，不重講。

#### 步驟 2：量測冗餘（Fig. 2，App. C.1，Eq. 5）

- 探測方法：給定分成 $R$ 組的分組，把每個 token 換成組平均，量保留變異

$$\mathrm{EV}=1-\frac{\sum_j\lVert h_j-\bar h_{c(j)}\rVert^2}{\sum_j\lVert h_j-\bar h\rVert^2}\tag{Eq. 5}$$

  EV = 1：匯集後的 token 完整攜帶表示；EV = 0：不比一個常數好。EV 量的是單一區塊激活的線性餘裕，不是輸出品質（App. C.1）。
- 比較三種分組（皆 $R=256$ of 1024）：adaptive（Hilbert 順序在最大 $R-1$ 個特徵差距處切）、fixed（同順序等距切；$R=256$ 時恰為 2×2 格）、skip（不是分組：保留得分最高的 $R$ 個 token，其餘換成平均）。
- 三軸結果：
  - 空間（Fig. 2a）：依完成影像的梯度能量排序 patch，最細節的 15% patch 占一半細節，前一半占 88%。
  - 時間（Fig. 2b）：把 $N=1024$ 匯集成 $R=256$，中段區塊的 EV 隨影像成形上升；自適應分割全程高於均勻格。
  - 深度（Fig. 2c）：EV 由第一塊的 61% 升到中段約 75%，再往最後一塊下降（Sec. 3.2）；skip（刪除而非匯集）回收少得多。
- $R=256$ 時匯集無法回收的 26% 是 RTI 保留全解析度 coda 與可學習 WRITE 的理由（App. C.1）。
- 探測設定：凍結 MiniT2I-B/16、bf16、30 Euler 步、guidance 5.0；held-out 合成提示（十個領域，已對 GenEval 與 PartiPrompts 去近重複）（App. C）。

#### 步驟 3：形狀推出四個要求，並預測各基準的失敗（Sec. 1、4.2、Tab. 4）

| 要求 | 來源 | 失去它的方法 |
| --- | --- | --- |
| 連通（contiguity） | 冗餘 patch 在平坦區彼此相鄰 | Feat Sim（相似度合併）、Latent Array（Sec. 4.2：「contiguity and position at once」） |
| 大小隨內容（content-sized） | 細節集中在少數 patch | 三個基準都失去 |
| 位置（position） | 每組涵蓋一個確定位置，token 要代表它 | Feat Sim、Latent Array |
| 匯總（summarization） | Fig. 2c：刪除失去匯集保留的東西 | Token Skip |

Tab. 4（$R=256$，64 個提示，凍結特徵；spread 單位為 32×32 格）。retained 是 Eq. 5 套在各方法自己的分組上、對核心 $[3,13]$ 各塊平均，不含任何可學習 WRITE（RTI 也不含），所以是對每種方法同等悲觀的下界（App. C.2）。size–detail：各組 fineness（$-\log_2$ 組大小）與其涵蓋細節的相關，取樣中段讀取。

| 方法 | spread ↓ | size–detail ↑ | retained ↑ |
| --- | --- | --- | --- |
| RTI | 0.49 | 0.50 | 0.73 |
| fixed boundaries | 0.71 | 0.00 | 0.67 |
| Feat Sim | 5.16 | 0.03 | 0.74 |
| Latent Array | 11.35 | 0.20 | −0.08 |
| Token Skip | 無分組 | 無分組 | 0.26 |
| 整張影像 | 12.24 | — | — |

- 作者的解讀：相似度合併保留最多資訊卻輸在品質，它放棄的是位置而非資訊；Latent Array 的 −0.08 表示其匯集 token 重建區塊比全域平均還差（Tab. 4 註）。
- 頁面在 S3 用此表並逐列說明；Sec. 5 寫這四個要求「在訓練任何縮減前就固定，事先說明了哪種縮減會失敗、如何失敗」。

#### 步驟 4：只壓縮核心（Sec. 3.2，Fig. 3）

- 沿用 recurrent-depth Transformer（Geiping et al. 2025）的 prelude／core／coda 結構：中段連續區塊 $[s_0,s_1]$ 為核心。READ 在 $s_0$ 前把 $N$ 個 patch token 匯集成 $R$ 個區域 token，核心每塊都只跑這段短序列，WRITE 在 $s_1$ 後散回 $N$ 個 patch；核心外的區塊維持全解析度。
- 兩端不適合壓縮的理由相反：prelude 仍在把原始 patch 轉成特徵，此時匯集會在模型處理前丟掉像素；coda 要把特徵轉回逐像素輸出，共享的區域更新會造成模糊。
- 主幹以 LoRA（rank 32，注意力與 MLP 投影）適配短序列；READ/WRITE 初始化為恆等，訓練從凍結模型的行為出發（Sec. 4.1、App. A）。
- 頁面 S4 放一張靜態架構 SVG（見 3 節）。延伸閱讀：`recursive-transformer-followups.html`（recurrent-depth）、系列教學的 hourglass 段落。

#### 步驟 5：Hilbert 順序（Sec. 3.1、App. B.4、C.2）

- Hilbert 曲線恰好走過 $2^k\times2^k$ 格的每一格一次，曲線上相鄰位置在格中一定相鄰；所以任何連續段都占據一個連通區域（不一定是正方形）。
- 更強的性質：每個二元（quadtree）方格恰好是一段連續位置。因此切割序列可以表達任何 quadtree 分割，也能表達 quadtree 做不到的分割。
- 與 quadtree 比較（Sec. 3.3）：quadtree 也有自適應大小，但每個區域綁定二元正方形，且每張影像每步都要建一棵樹；切 Hilbert 序列保留自適應大小、去掉這兩個缺點。
- 其他順序（App. B.4、Fig. 14a）：raster 連續段在行尾斷開；Morton（Z）連續段在二元方格邊界以對角跳躍連接。Morton 的二元方格也都是連續段，所以 Morton 與 Hilbert 只差在對角跳躍。
- 凍結模型上的比較（App. C.2，$R=256$ 自適應切割）：retained Hilbert 0.735、Morton 0.720、raster 0.697；組緊緻度（每 patch spread）1.49、1.73、3.85 格。Morton 對 Hilbert 差 0.015；依內容切 vs 等距切差 0.059；順序間全差距（Hilbert 對 raster 0.038）約為後者的三分之二。
- 〔本頁推導〕對齊的 $2^m$ 長度段在三種順序下都連通（raster 對齊段是整行或行的一段）；斷裂只發生在不對齊的段，而自適應切點可落在任何位置，所以順序的連通性在自適應切割下才重要。C1 以「所有起點中連通段的比例」呈現。
- 點雲 Transformer（PTv3）也用此曲線排序注意力序列；本文用它定義分組本身（Sec. 2）。

#### 步驟 6：內容自適應分割（Sec. 3.3，Eq. 1–2）

$$d_j=\big\lVert h^{(s_0)}_{\pi(j+1)}-h^{(s_0)}_{\pi(j)}\big\rVert_2,\quad j=1,\dots,N-1\tag{Eq. 1}$$
$$C=\{\,j: d_j\text{ 屬於最大的 }R-1\text{ 個}\,\}\tag{Eq. 2}$$

- 取相鄰切點之間的 $R$ 段為區域。切點落在最大特徵跳躍，均勻段保持合併，細節區一路切到單一 patch。
- 無參數、$O(N)$：一次掃描算差距，線性選擇取切點。
- 因為主幹預測乾淨影像，特徵追蹤影像結構，大差距標記內容邊界（Sec. 3.3）。切割取自當前狀態 $x_t$，每個取樣步重切（App. A），細區域隨物體成形聚集（Fig. 9a）。
- 〔本頁推導〕平手處理：論文未寫；頁面以「差距由大到小、同值取較小 $j$」排序，使結果確定。特徵全相同時（EV 分母為 0）讀數顯示「—」。
- 預算 $R$：訓練時從 $\{64,128,256,512\}$ 每步隨機抽（Sec. 4.1）。

#### 步驟 7：READ 與 WRITE（Sec. 3.4，Eq. 3–4）

$$\alpha_j=\operatorname{softmax}_{j\in c}\big(w^\top h_j^{(s_0)}\big),\qquad z_c=\sum_{j\in c}\alpha_jh_j^{(s_0)}+e_{\text{size}}\big[\lfloor\log_2\lvert c\rvert\rfloor\big]\tag{Eq. 3}$$

- softmax 只在區域 $c$ 內做，所以 READ 是「限制在區域內的單查詢注意力」，不是盲目平均，可強調區域中資訊多的 patch。
- 區域大小是任意整數，因此用索引 $\lfloor\log_2|c|\rfloor$，告訴核心這個 token 代表 1 個還是 16 個 patch。
- 位置：區域 token 帶的位置是其 patch 旋轉位置嵌入（RoPE）的平均。這些是單位相量，平均值在波長長於區域範圍時仍接近單位圓，波長短於區域範圍時縮向 0。平均等於依區域範圍對位置做低通：單一 patch 時精確，大區域時粗略。
- 〔本頁推導〕一列長 $L$、相鄰間隔 1 的直線區域，頻率 $\omega$ 的平均相量長度為 $\left|\dfrac{\sin(\omega L/2)}{L\sin(\omega/2)}\right|$（等比級數求和）；$L=1$ 時為 1，$\omega L\ll1$ 時接近 1（只適用直線段；旁瓣與 $\omega>\pi$ 的混疊除外）。C3 的區域是二維 Hilbert 段，不在此公式範圍內，頁面直接數值求和；公式只用於 L=2 這種恰為直線的情形做 selfTest。

$$h_j^{(s_1+1)}=h_j^{(s_0)}+g\Big(\big[\,h_j^{(s_0)}\,\Vert\,z'_{c(j)}-z_{c(j)}\,\big]\Big)\tag{Eq. 4}$$

- 天真的反運算把區域變化 $z'_c-z_c$ 原樣加到每個 patch；WRITE 讓每個 patch 由該變化與自己的匯集前特徵產生自己的更新，同區域的 patch 可以各自特化。
- 這也是繞過核心的跳接：每個 patch 保留進核心前的特徵，把核心的工作當修正量接收；所以核心從未看過個別 patch，coda 仍能產生逐像素輸出。
- READ/WRITE 都只在區域內作用，成本 $O(N)$ 與 $R$ 無關，且包含介面全部可學習參數。對比全域查詢的 $O(NR)$（Sec. 2）。
- 消融（Fig. 8b，$R=256$）：把可學習 READ/WRITE 換成平均＋廣播，GenEval −7.2。

#### 步驟 8：訓練與彈性（Sec. 4.1、4.3，Tab. 2）

| 項目 | MiniT2I-B/16 | MiniT2I-L/16 | JiT-B/16（類別條件） |
| --- | --- | --- | --- |
| 解析度／$N$ | 512² / 1024 | 512² / 1024 | 256² / 256 |
| 主幹區塊／核心 | 17 / 3–13 | 23 / 3–19 | 12 / 4–9 |
| 訓練預算 | {64, 128, 256, 512} | 同左 | {32, 64, 128, 192} |
| 步數 | 40K | 40K | 100K |
| 可訓練參數（占主幹） | 16.3M（5.9%） | 36.0M（3.8%） | 6.0M（4.4%） |
| 取樣 | 100 Euler 步、guidance 5.0 | 同左 | 50 Heun 步、guidance 3.0 於 [0.1, 1.0] |

共用：AdamW、lr 1e-4、2K warmup、global batch 256、LoRA rank 32、EMA 0.9999。文字到影像模型用 MiniT2I 約 120K 張的對齊資料池（512²），類別條件模型用 ImageNet-1K。JiT-L/16 用核心 8–19（24 塊中）。

- 彈性（Sec. 4.3、Fig. 8a）：同一 checkpoint 在沒訓練過的 $R\in\{96,192,384\}$ 落在同一條曲線上，在此範圍（64–512）內預算是訓練後才設定的可調參數。類別條件版在 130、193 也成立：訓練過的 $R=128$ FID 6.34，未訓練的 130 為 6.23（App. B.2）。
- 所有基準都是作者在同一介面中重新實作：同一凍結主幹、LoRA、配方、資料與預算協定，只換縮減規則（App. A）。Feat Sim 依 ToMe 取 $R$ 個等距錨點、每個 patch 依 cosine 相似度指派給最近錨點，READ/WRITE 照樣作用；Latent Array 以 $R$ 個無旋轉位置的可學習潛變數做交叉注意力讀寫；Token Skip 依 Mixture-of-Depths，中段區塊稠密／稀疏交替，稀疏塊只收前 12.5% 影像 token（容量 $C=0.125$，平均每塊 576 token），交替範圍 $[2,13]$。

#### 步驟 9：速度與上限（Sec. 4.1–4.2、App. A、Fig. 10、App. B.5、Tab. 3）

- 計時：RTX 4090、fused attention、每批 4 張（CFG 加倍為 8），三次交錯量測取中位數 images/s；包含分割、READ/WRITE、adapter 與文字串流。速度比為同一主幹的吞吐量比。
- 上限：prelude、coda 與文字串流不會縮短；B/16 的速度在約 2.5× 飽和，低於 $R\approx192$ 時更少的預算只降品質不加速（Sec. 4.2，只對 B/16 陳述）。L/16 主幹較長、可壓縮部分較多，$R=256/128/64$ 時仍升到 2.59×/2.86×/3.01×（Tab. 1、Fig. 16）。
- Fig. 10：量測時間比解析 FLOPs 下降得快（這些長度下注意力部分受記憶體頻寬限制）；分割與 READ/WRITE 為 $O(Nd)$，占不到 2%；所以論文報告量測吞吐量而非 FLOPs。
- 〔本頁推導〕影像 token 成本模型：每塊相對成本 $(1-\eta)\,\nu+\eta\,\nu^2$，$\nu=n/N$，$\eta=0$ 為只算線性層、$\eta=1$ 為只算注意力；$B$ 為區塊數。$n_{\text{core}}=s_1-s_0+1$ 為核心區塊數（含兩端點：$[3,13]\to11$、$[1,15]\to15$、$[3,19]\to17$、$[4,9]\to6$）。總相對成本 $\rho=\big[(B-n_{\text{core}})+n_{\text{core}}((1-\eta)\nu+\eta\nu^2)\big]/B$。對照（只是比例上的一致性檢查，不是用量測驗證模型）：
  - Tab. 3：$[1,15]$ 對 $[3,13]$ 在 $R=256$：$\eta=0$ 得 $5888/8960=0.657$（論文 0.66×），$\eta=1$ 得 0.439（論文「0.44× on attention」）。
  - 同成本區間：$\eta=0$ 解得 $R=460.8$，$\eta=1$ 解得 $R=572.4$，即論文的 $R\approx461$–572。論文附註（App. B.5）：此區間只計影像 token，全長的文字串流使真正的同吞吐量預算低於 $R=461$；且 $[1,15]$ 沒有計時，只知其速度上限高於 2.5×。
  - JiT 類別條件（12 塊、核心 6 塊、$N=256$）：$R=64,130,193$ 的 $\eta=0$ 成本 0.625、0.754、0.877，速度 1.600、1.326、1.140；論文寫 62/75/88% token 計算量與 1.61/1.33/1.14×。這些是論文依 token 計算量推出的解析值，主幹沒有計時（Fig. 11 說明），所以這裡只是兩個解析值彼此一致。
  - $R\to0$、只看影像 token 的上限：B/16 $17/6=2.83\times$，L/16 $23/6=3.83\times$。B/16 量測飽和約 2.5×，差距來自文字串流與固定開銷。頁面加一個「不縮短的開銷比例 $\tau$」滑桿：$\rho_\tau=\tau+(1-\tau)\rho$；〔本頁推導〕若 B/16 的 2.5× 飽和全歸於 $\tau$，$\tau=(0.4-6/17)/(1-6/17)\approx0.073$。$\tau$ 不影響同成本區間（兩邊同加）。
  - $\tau=0$ 時 Tab. 1 的量測點：B/16 512→1.84（模型帶 1.478–1.943）、256→2.15（1.943–2.542）；L/16 512→2.11、256→2.59、128→2.86 都在帶內，L/16 64→3.01 低於帶（3.257–3.791）。
- 核心長度（App. B.5，Tab. 3）：$[1,15]$ 在 $R=256$ 成本 0.66、GenEval 77.4、Pick 21.70（$[3,13]$：84.7、22.10），較差但便宜三分之一；在恢復同成本的 $R=512$ 得 85.4、22.14，GenEval +0.7 在參考列 ±1.7 內，PickScore +0.037（CI [+0.013, +0.061]）只占其尺度 0.17%，ImageReward +0.002（CI [−0.015, +0.019]）。作者結論：核心長度買的是計算量，不是品質。
- 少步數 vs 少 token（Sec. 4.2、App. B.3、Fig. 7、Fig. 13，L/16）：單步 dense 288 ms、$R=256$ 112 ms，區域 token 的一步便宜 2.6×（288/112 = 2.57）。四個配對成本 1.8/2.8/5.7/11.2 s（每對誤差 4% 內）：dense 6/10/20/39 步，RTI 16/25/50/100 步。1.8 s 時 GenEval 82.7 vs 65.9（差 16.8）；低於約 3 s RTI 明顯領先，同品質下便宜 2.1×；高於約 5 s 增加步數較划算，11.2 s 時 dense 87.6 vs 86.3。CLIPScore 上 dense 始終沒追上。作者附註：六步 Euler 是弱取樣器；少步數蒸餾模型配區域介面是自然的組合。

#### 步驟 10：實驗結果（Sec. 4.2–4.4，App. B）

Tab. 1 摘錄（GenEval 553 個提示的整體準確率；PartiPrompts 1632 個提示；± 為對同區塊 backbone 列的配對差 95% 範圍）：

| 模型 | R | img/s | 速度 | GenEval | CLIP | Pick | IR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MiniT2I-B/16 | 1024 | 0.40 | 1.00× | 87.2 | 28.23 | 22.45 | 1.18 |
| Token Skip | 576* | 0.48 | 1.19× | 82.3 | 28.08 | 21.93 | 1.06 |
| Feat Sim | 512 | 0.74 | 1.84× | 84.7 | 28.04 | 22.04 | 1.07 |
| RTI | 512 | 0.74 | 1.84× | 87.8 | 28.09 | 22.30 | 1.11 |
| Feat Sim | 256 | 0.86 | 2.15× | 80.5 | 27.73 | 21.57 | 0.93 |
| Latent Array | 256 | 0.86 | 2.15× | 45.7 | 23.30 | 19.78 | −0.65 |
| RTI | 256 | 0.86 | 2.15× | 84.7 | 28.04 | 22.06 | 1.06 |
| MiniT2I-L/16 | 1024 | 0.14 | 1.00× | 88.1 | 28.50 | 22.75 | 1.23 |
| Feat Sim | 512 | 0.29 | 2.11× | 83.1 | 28.80 | 22.34 | 1.16 |
| RTI | 512 | 0.29 | 2.11× | 87.5 | 28.58 | 22.63 | 1.20 |
| Feat Sim | 256 | 0.36 | 2.59× | 76.5 | 28.31 | 21.76 | 1.01 |
| RTI | 256 | 0.36 | 2.59× | 86.3 | 28.52 | 22.37 | 1.17 |

\* 不是預算：Token Skip 稠密與稀疏核心塊的平均 token 數。頁面表格保留 ± 欄（照錄原文，例 RTI B/16 R=256 GenEval 84.7±1.7）。

- 主結果：B/16 $R=256$ 以 2.15× 得 84.7（dense 87.2）；掃同一 checkpoint 的預算，從 $R=384$（2.00×）起差異落在噪聲內，但 dense 在每個偏好分數上仍領先。B/16 $R=512$ 同時在品質與速度上超過自己的 dense（87.8 vs 87.2，1.84×），此時 Feat Sim 只與 RTI 的 $R=256$ 列持平。L/16 $R=512$ 87.5 在噪聲內、2.11×；Fig. 4：2.11× 時 L/16 距 dense 0.6，B/16 在 2.15× 已失 2.5。作者說 L/16 區塊展示的是轉移到更長主幹，不是建議選 L/16。
- 與基準比較（Sec. 4.2）：同預算下的代價為「刪除而非匯總」2.4 GenEval（Token Skip；且它只有 1.19×，低估了差距）、「打散分組」4.2（Feat Sim）、「拿掉位置」39.0（Latent Array）。論文 Sec. 4.2 用「removing the position」稱呼第三項，但 Sec. 4.3 指出 latent array 同時移除分組與位置，其代價只是「錨定」單獨貢獻的上界；頁面一律寫「移除分組與位置（上界）；位置單獨的貢獻論文未分離」。Token Skip 保留的表示只有 RTI 分組的三分之一，稠密那一半核心使速度上限低於約 1.3×。Feat Sim 的 patch 離組中心 5.2 格（RTI 0.5），$R=64$ 時差距達 13.8 GenEval。Latent Array 訓練 40K 步後讀取注意力仍覆蓋 11.4 格（最大 12.2），作者指出 Perceiver-IO/RIN 是為從頭訓練設計的。
- 消融（Sec. 4.3、Fig. 8b）：等距切點（同順序、同預算、同介面）在 $R=128$ 失 2.7，預算越大越少（$R=256$ 時一區 4 patch，等距切點已接近自適應）；平均＋廣播失 7.2；換成 latent array 掉到 Latent Array 基準，這同時移除分組與位置，所以是「錨定」貢獻的上界。
- 分割行為（Sec. 4.4、Fig. 9、App. C.3）：細節＝$\hat x$ 的梯度能量；細區域＝1 patch、粗區域＝≥16 patch。兩者在最吵的步驟攜帶相同細節，隨影像成形拉開到約 6 倍；fineness 與 detail 相關 +0.58（$R=256$、取樣中段）。切割只讀進入核心的特徵、從不讀細節量，所以這個對齊是模型自己做到的。區域大小熵：$R=512$ 低、$R=64$ 高，取樣中都往中間漂移；預算決定大小範圍，影像決定細區域放在哪。
- FID（App. B.1，MJHQ-30K，L/16，clean-fid）：RTI 比 Feat Sim 好 8.6（$R=512$）與 13.3（$R=256$）；RTI $R=256$ 比 Feat Sim $R=512$ 還低。FID 是 RTI 在半數 token 時唯一沒追上 dense 的指標：$R=512$ 為 +2.4。作者結論：壓縮先損失分布保真度，後損失組合準確率。guidance 5.0 對 FID 偏高，各方法同等膨脹。
- 類別條件（App. B.2，ImageNet-256，JiT）：$R=130$ B/16 FID 6.23 vs Feat Sim 15.99（2.6× 低）；$R=64$ 用 62% token 計算量得 16.69，Token Skip 用 71% 得 19.11；$R=130$ 以四分之三的計算量勝過加深版 Token Skip（6.23 vs 7.02；加深版回到 dense 計算量但沒回到 dense 品質，7.02 vs 3.62）。L/16 $R=193$（88%）FID 2.42 vs 公開 dense 2.36；IS 313.5（$R=193$）、303.8（$R=130$）vs 298.5，作者提醒 IS 偏好自信的原型預測，主張由 FID 承擔。冗餘探測（Fig. 12）：各主幹 25% token 保留約 70% 表示（0.61–0.78）；256² 時深度輪廓淺（第一塊到峰值約 0.03），512² 為 0.09–0.13。
- 順序消融（App. B.4、Fig. 14–15）：$R=256$ 時三種順序差異在噪聲內；預算越小越依局部性排序。$R=64$：raster −0.268 PickScore（CI [−0.296, −0.239]）、−0.154 ImageReward（CI [−0.182, −0.125]）；Morton −0.153 / −0.083；GenEval 上 raster 在 $R=128$、64 失 3.2、4.6（遠超 ±1.6），Morton 維持參考值。作者解讀：行尾斷裂傷害構圖，Morton 的對角跳躍保留構圖、只損失知覺品質。

#### 步驟 11：與 HiT 及系列方法的異同（頁面 S13，短節）

| 面向 | RTI（本文） | HiT（No. 585，姊妹教學） | ToMe／Feat Sim |
| --- | --- | --- | --- |
| 起點 | 凍結的預訓練 checkpoint＋LoRA | 從頭端到端訓練 | 免訓練或少量調整 |
| 分組規則 | 無參數：切 Hilbert 序列 | 學習的親和度＋平均連結（average linkage）階層分群 | 特徵相似度 |
| 分組形狀 | 必定連通 | 可跨越遠處位置（HiT Sec. 3.2 原文） | 可跨越遠處位置 |
| 預算 | 使用者指定的精確 $R$ | 以壓縮損失調節批次平均比例，每張影像組數可變 | 固定合併比例 |
| 目標 | 推論加速 | 生成品質與訓練效率 | 推論加速 |

- 相同點：都在像素空間 DiT 中段處理較少 token、以軟權重匯集、以跳接把資訊送回原 patch 位置（HiT 的 skip projection 也初始化為恆等）。
- 頁面寫法：只列差異，HiT 細節連到姊妹教學；ToMe、hourglass、DC-DiT、ELIT 的背景連到系列教學 `token-compression-lineage.html`。本頁對 HiT 的陳述只取自 HiT 摘要與 Sec. 3.1–3.3，建置時與姊妹教學核對（R9）。
- 論文自己的定位（Sec. 2）：hourglass 形狀與階層 Transformer（Nawrot et al. 2022、Geiping et al. 2025）相同，但它們以固定步幅匯集；DC-DiT 也可從預訓練升級，但切塊由路由器學習、目標是平均比例、且在 VAE 潛空間；ELIT 是最接近的交叉注意力瓶頸，RTI 的 READ/WRITE 屬同一族但把潛變數錨定在空間上。

#### 步驟 12：實作設定與注意事項（頁面 S14 分兩組列出）

有實驗支持的常見錯誤：
1. 用 raster 順序切割：$R\le128$ 時行尾斷裂明顯傷害 GenEval（−3.2、−4.6；App. B.4）；Morton 只損失知覺分數。
2. 以平均＋廣播取代 READ/WRITE：GenEval −7.2（Fig. 8b）。
3. 換成不帶位置的 latent array：同時移除分組與位置，品質掉到 Latent Array 等級（45.7）；位置單獨的貢獻論文未分離（Sec. 4.3）。
4. 以 FLOPs 估速度：Fig. 10 量測時間低於 FLOP 曲線；B/16 速度約在 2.5× 飽和、$R<192$ 不再加速（L/16 仍會加速）。

論文的設計選擇（論文沒有做對照實驗，頁面寫「論文的做法」，不寫成錯誤）：
5. 只壓縮中段：依 Fig. 2c 兩端對每個 patch 敏感的推理，B/16 核心 $[3,13]$ of 17；App. B.5 只比較了 $[1,15]$（同成本下無差異）。
6. 每個去噪步都從當前狀態重建分割（App. A 的描述）。
7. 每步從 {64, 128, 256, 512} 抽 $R$；未訓練的 96/192/384 落在同一曲線（Sec. 4.3）；固定單一預算訓練未測試。
8. READ/WRITE 初始化為恆等，訓練從凍結模型行為出發（Sec. 4.1）。論文未給 $g$ 的層數與寬度、$e_{\text{size}}$ 的維度，頁面明寫未說明。

#### 步驟 13：限制（Sec. 5）

速度有上限（prelude、coda、文字串流不縮短）；核心無法區分同區域內的 patch；不是所有分組都是 Hilbert 區間；預算由使用者選，不由影像內容決定。

---

## 2. 學習目標（驗收測試）

每章結尾以 `<details>`（summary「檢查理解」）列出該章檢查題與參考答案；參考答案的內容必須同時寫在該章正文裡（不能只在元件、表格或答案框中）。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明為何像素空間 DiT 的 token 浪費最大 | 為什麼論文說浪費在像素空間模型最大？MiniT2I-B/16 在 512² 有幾個 patch token？ | 沒有 VAE 先移除低階空間冗餘，每個區塊都要處理；patch 16 → 32×32 = 1024 個（Sec. 1、3.1） | S1 |
| LO2 | 說出冗餘的三個軸與數字 | Fig. 2 在空間、時間、深度上各量到什麼？ | 空間：最細節的 15% patch 占一半細節，前一半占 88%；時間：EV 隨影像成形上升；深度：EV 由第一塊 61% 升到中段約 75% 再往末塊下降（Fig. 2、Sec. 3.2） | S2 |
| LO3 | 定義 EV 並連到設計 | EV（Eq. 5）的 1 與 0 代表什麼？$R=256$ 時未回收的 26% 導致什麼設計？ | 1：匯集後完整攜帶表示；0：不比常數好；26% 是 RTI 保留全解析度 coda 與可學習 WRITE 的理由（App. C.1） | S2 |
| LO4 | 列出四個要求與各基準失去哪一個 | 四個要求是什麼？Feat Sim、Latent Array、Token Skip 各失去什麼？ | 連通、大小隨內容、位置、匯總；Feat Sim 失去連通與位置，Latent Array 失去分組（因而連通）與位置，Token Skip 失去匯總；三者都失去大小隨內容（Sec. 1、4.2） | S3 |
| LO5 | 解釋 Hilbert 順序的性質 | 為什麼 Hilbert 序列的任一連續段都是連通區域？raster 與 Morton 何處斷開？ | 曲線上相鄰位置在格中一定相鄰；每個二元方格恰為一段，可表達任何 quadtree 分割；raster 在行尾斷開，Morton 在二元方格邊界對角跳躍（Sec. 3.1、App. B.4） | S5 |
| LO6 | 寫出分割規則 | RTI 如何決定 $R$ 個區域？成本與參數量？何時重算？ | 算曲線上相鄰 patch 的特徵距離 $d_j$（Eq. 1），在最大的 $R-1$ 個處切（Eq. 2）；$O(N)$、無參數；每個去噪步從當前 $x_t$ 重算（Sec. 3.3、App. A） | S6 |
| LO7 | 說明 READ 的成分 | 區域 token 由哪些部分組成？ | 區域內 softmax 權重 $\alpha_j=\mathrm{softmax}_{j\in c}(w^\top h_j)$ 的加權平均，加上尺寸嵌入 $e_{\text{size}}[\lfloor\log_2|c|\rfloor]$（Eq. 3） | S7 |
| LO8 | 說明區域 token 的位置 | 區域 token 的位置怎麼給？大區域的位置為何較粗？ | 取其 patch 旋轉位置嵌入（單位相量）的平均；波長長於區域範圍的分量仍接近長度 1，波長較短的分量縮向 0，所以位置依區域大小變粗（Sec. 3.4） | S8 |
| LO9 | 說明 WRITE 與天真廣播的差別 | Eq. 4 與「把 $z'_c-z_c$ 加到每個 patch」有何不同？為何 coda 仍能逐像素輸出？ | 每個 patch 以 $g([h_j\Vert z'_c-z_c])$ 產生自己的更新，可特化；形式上是繞過核心的跳接，patch 保留核心前特徵；換成平均＋廣播 GenEval −7.2（Eq. 4、Fig. 8b） | S9 |
| LO10 | 說明一個 checkpoint 如何適用多種預算 | 訓練時哪些參數更新？$R$ 怎麼選？未訓練的預算表現如何？ | 主幹凍結，只訓練 rank-32 LoRA 與 READ/WRITE（恆等初始化）；每步從 {64, 128, 256, 512} 隨機抽 $R$；未見過的 96、192、384 落在同一曲線（Sec. 4.1、4.3） | S10 |
| LO11 | 解釋速度上限、比較少步數與少 token | B/16 速度為何在約 2.5× 飽和？1.8 s/張時 RTI 為何勝過少步數的 dense？ | prelude、coda、文字串流不縮短，B/16 低於 $R\approx192$ 不再加速（L/16 仍會加速）；L/16 上區域 token 的一步便宜 2.6×（288 vs 112 ms），同時間 L/16 dense 6 步、L/16 RTI 16 步，GenEval 65.9 vs 82.7；高於約 5 s 增加步數較划算（Sec. 4.2、App. B.3） | S11 |
| LO12 | 讀出主結果與基準代價 | B/16 在 $R=256$、$R=512$ 的速度與 GenEval？同預算下三個基準的代價？ | $R=256$：2.15×、84.7（dense 87.2）；$R=512$：1.84×、87.8；刪除而非匯總 2.4（Token Skip 只有 1.19×，不是同預算比較，差距被低估）、打散分組 4.2、移除分組與位置 39.0（上界）GenEval（Tab. 1、Sec. 4.2–4.3） | S12 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），互動元件放在加寬圖版（最大 1080px）。章節編號以 JetBrains Mono 顯示（`05 · …`）。視覺記號：一條 8×8 Hilbert 曲線；hero 以 `--ink-3` 細線「描繪」它（stroke-dashoffset 動畫 1.2 s，reduced-motion 時直接顯示完成狀態），三個切點以 `--accent` 短橫標出；C1、S4 架構圖重複同一個 8×8 示意。章節標題用陳述句。每節先定義符號（「本節符號」小框），再放用到它的元件。

| # | id | 標題 | 內容 | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈RTI：把影像 patch 排成一條線，在內容變化最大處切開〉 | 副標：「這篇論文把預訓練像素擴散模型中段的 patch token 換成數量可調的區域 token，一個模型可在 64–512 個 token 的範圍內任選。」三張事實卡（下列）；連結 arXiv、`../index.html#paper-217`、`rti-lineage-hub.html`、`token-compression-lineage.html` | 1 螢幕 |
| S1 | `basics` | 01 · 擴散 Transformer 對每個 patch 分配相同計算量 | 區塊、取樣步＝一次前向、patch token、注意力 $O(n^2)$ 與線性層 $O(n)$、rectified flow 預測 $\hat x$（2–3 句）、像素空間 vs 潛空間、MiniT2I-B/16 與 $N=1024$。本節符號：$N$、$x_t$、$\hat x$。延伸：`pixel-space-diffusion.html`、`x-prediction-manifold-view.html`、`tokc-lineage-hub.html` | 400–500 字 |
| S2 | `redundancy` | 02 · 冗餘集中在平坦區域與中段區塊 | 先用最簡例子教 EV（Eq. 5）：把 32×32 格切成等大的 2×2 方格、每格換成平均（$R=256$），算保留變異；再說論文另用「依內容分組」（S6 才講做法），比較兩者。Fig. 2 三軸數字（區塊＝S1 定義）；skip＝只留 $R$ 個、其餘換成平均；26% → coda 與 WRITE。本節符號：$R$、EV、$\bar h$。延伸：`rti-redundancy-probe-four-requirements.html` | 400–500 字 |
| S3 | `requirements` | 03 · 冗餘的形狀決定壓縮必須保留的四個性質 | 先各用一句定義三個基準：Feat Sim（依特徵相似度把 patch 指派給錨點，ToMe 式）、Latent Array（$R$ 個可學習向量以交叉注意力讀整張圖）、Token Skip（只讓部分 token 進入某些區塊，其餘略過）；再定義 retained（Tab. 4 中的 EV，無 WRITE 的下界）、size–detail、spread（Eq. 6：$w_{rj}$ 和為 1，$p_r=\sum_jw_{rj}p_j$，$\mathrm{spread}_r=\sum_jw_{rj}\lVert p_j-p_r\rVert_2$；單 patch 0、整張影像 12.2）。然後四要求表、Tab. 4。延伸：`rti-prior-reductions-compared.html`、`depth-redundancy-adjacent-methods.html`、`tokc-vit-pruning-and-merging.html`、`tokc-hourglass-and-latent-interfaces.html` | 500–600 字 |
| S4 | `architecture` | 04 · RTI 只壓縮中段核心 | prelude／core／coda、兩端不適合壓縮的理由、LoRA、文字串流全長。本節符號：$s_0$、$s_1$（核心第一與最後一塊，含兩端點，B/16 為 3 與 13）。靜態 SVG（viewBox 0 0 720 200）：17 個區塊方塊，0–2 與 14–16 全高（標「N 個 patch token」），3–13 約四分之一高（標「R 個區域 token」），READ／WRITE 楔形在 2→3、13→14 之間，WRITE 上方一條跳接弧線標「核心前特徵跳接」；文字直接標籤。延伸：`recursive-transformer-followups.html`、`tokc-hourglass-and-latent-interfaces.html` | 300–400 字 + SVG |
| S5 | `hilbert` | 05 · Hilbert 曲線把二維分組變成一維切割 | 曲線定義、相鄰性、二元方格＝一段、quadtree 比較、raster／Morton；C1；App. C.2 數字。本節符號：$\pi$、$q$（格邊長）。延伸：`rti-hilbert-order-region-partition.html`、`tokc-content-adaptive-tokenization.html` | 400–500 字 + C1 |
| S6 | `partition` | 06 · 在特徵變化最大的地方切割 | Eq. 1–2、$O(N)$、無參數、每步重切、預算隨機抽；C2；Fig. 9 的時間行為（文字）；〔本頁推導〕平手規則。本節符號：$h_j^{(s_0)}$、$d_j$、$C$ | 450–550 字 + C2 |
| S7 | `read` | 07 · READ 把一個區域匯集成一個 token | Eq. 3：區域內 softmax（「單查詢注意力」）、尺寸嵌入索引 $\lfloor\log_2|c|\rfloor$；$O(N)$ vs $O(NR)$。本節符號：$c$、$|c|$、$\alpha_j$、$w$、$z_c$、$e_{\text{size}}$。延伸：`rti-read-write-interface.html` | 300–400 字 |
| S8 | `position` | 08 · 區域 token 的位置是相量的平均 | 先講 RoPE：位置 $p$ 在頻率 $\omega$ 上對應旋轉角 $\omega p$（單位相量 $e^{i\omega p}$），注意力只看兩個 token 的角度差。數值例：$\omega=1$、兩個相鄰位置 0 與 1，角度 0 與 1 弧度，平均相量長度 $\lvert\cos(1/2)\rvert=0.878$；$\omega=0.01$ 時為 0.99999。再定義「低通」：短波長（角度轉得快）的分量在平均後變小、長波長的保留。〔本頁推導〕直線公式與適用範圍；C3。本節符號：$\omega$、$\lvert m(\omega)\rvert$。延伸：`tokc-position-for-merged-tokens.html` | 400–500 字 + C3 |
| S9 | `write` | 09 · WRITE 把區域的變化散回每個 patch | Eq. 4、天真廣播對照、跳接、恆等初始化、消融 −7.2。本節符號：$z'_c$、$g$ | 250–350 字 |
| S10 | `training` | 10 · 隨機預算讓一個 checkpoint 涵蓋訓練範圍內的預算 | 凍結＋LoRA、預算抽樣、Tab. 2 摘要、彈性（Fig. 8a、App. B.2 6.34 vs 6.23）、基準實作方式。延伸：`rti-elastic-budget-lora-training.html`、`tokc-dynamic-chunking-bytes-to-pixels.html` | 350–450 字 + 表 |
| S11 | `cost` | 11 · 加速的上限來自不縮短的部分 | 計時協定、B/16 2.5× 飽和與 L/16 對照、Fig. 10、〔本頁推導〕成本模型與對照、核心長度（Tab. 3）與其附註；C4；少步數 vs 少 token（B.3）。本節符號：$B$（區塊數）、$n_{\text{core}}=s_1-s_0+1$（核心區塊數，含兩端點；B/16 [3,13] 為 11）、$\nu=n/N$、$\eta$、$\rho$、$\tau$ | 550–650 字 + C4 |
| S12 | `results` | 12 · 同預算下 RTI 領先每種縮減方法 | Tab. 1（含 ±）、主結果、基準代價（39.0 寫成上界）、消融、Fig. 9 分割行為、FID、類別條件、順序消融。延伸：`rti-results-and-limits.html`、`tokc-training-free-diffusion-merging.html`、`tokc-learned-routing-dynamic-dit.html` | 650–800 字 + 表 |
| S13 | `compare` | 13 · 與 HiT 及其他縮減方法的異同 | 1.4 步驟 11 的表與兩段文字；連 `hit-hierarchical-diffusion-transformer.html`（`../index.html#paper-585`）與 `token-compression-lineage.html` | 200–300 字 + 表 |
| S14 | `impl` | 14 · 實作設定與注意事項 | Tab. 2 全表、1.4 步驟 12 兩組條目 | 300–400 字 + 表 |
| S15 | `history` | 15 · 研究脈絡 | 時間線（6.4）+ 一段文字：ToMe → ToMeSD／ToDo → 混合解析度 tokenization → MoD → 依層與時間步的壓縮比 → JiT → ELIT、DC-DiT → RTI | 時間線 + 100–150 字 |
| S16 | `next` | 16 · 限制與延伸閱讀 | Sec. 5 四項限制；筆記清單（6.2 順序）；arXiv、專案頁 | 200 字 + 清單 |

Hero 事實卡（白話，不用未定義術語）：
1. 「自然影像的細節集中在少數小方塊（patch）。在已訓練好的像素擴散模型裡，中間幾層對平坦區域的計算大多是重複的。」
2. 「RTI 把所有小方塊排成一條不斷線的路徑，在相鄰內容變化最大的地方切開，每一段合成一個 token。」
3. 「同一個模型可在 64–512 個 token 之間任選：B/16 用 512 個時，速度 1.84 倍、GenEval 87.8（原模型 1024 個、87.2）。」

桌機（≥ 1024px）左側 sticky 章節目錄；< 1024px 頂部橫向捲動章節列（沿用參考頁 `#toc`／`#tocbar`）。

---

## 4. 互動元件規格

### 4.0 共用

- 格：$G=32$，$N=1024$；patch id `pid = row*32 + col`；座標 $p=(\text{col},\text{row})$。C1 的格邊長寫 $q$。
- 順序（位置 $j$ → (col,row)）：Hilbert 用 Wikipedia `d2xy`（`rx = 1 & (t>>1)`、`ry = 1 & (t ^ rx)`、`ry==0` 時若 `rx==1` 先鏡射 `x=s-1-x, y=s-1-y` 再交換 x、y，然後 `x += s*rx; y += s*ry; t >>= 2; s <<= 1`），起點 (0,0)、終點 (q−1,0)；Morton：偶數位元 → x、奇數位元 → y；raster：`(j % q, floor(j/q))`。
- 連通分量：4 鄰接 BFS。spread（Eq. 6）：歐氏距離（不平方），群內平均。
- PRNG：`mulberry32(seed)`（與參考頁相同）；`gauss(r)`：Box–Muller 只取 cos 分支，`sqrt(-2 ln(1-u1)) cos(2π u2)`。
- 玩具場景（C2、C3 共用；`common.py: scene('landscape')`，頁面須逐行移植）：128×128 px RGB，patch 4×4 px。`r = mulberry32(1)`，**每個像素依 row-major 恰好抽兩次** `n1, n2`。$u=(x+.5)/128$、$v=(y+.5)/128$，依序覆寫：
  1. $v<0.62$：天空 lerp([.55,.75,.95] → [.85,.92,.98], $v/0.62$)；否則地面 lerp([.45,.58,.32] → [.36,.48,.26], $(v-.62)/.38$)，再每通道加 $0.06(n_1-0.5)$。
  2. 太陽：$\lVert(u,v)-(.78,.18)\rVert<0.09$ → [.98,.86,.45]。
  3. 樹幹：$|u-.28|<.025$ 且 $.45<v<.70$ → [.40,.28,.18]。
  4. 樹冠：$\lVert(u,v)-(.28,.38)\rVert<0.15$ → [.20+.18(n₁−.5), .45+.18(n₁−.5), .20]。
  5. 招牌：$.52<u<.70$ 且 $.66<v<.82$ → [.95,.95,.92]；其中 $.69<v<.79$ 且 `floor(x/2) % 3 == 0` → [.10,.10,.12]。
  6. 每通道加 $0.01(n_2-0.5)$，截到 [0,1]。
- 噪聲場：`mulberry32(2)`，row-major、每像素 3 次 `gauss`（R、G、B）。混合 $x_t=t\,x+(1-t)\,\varepsilon$（$t=1$ 乾淨）。
- 特徵（**頁面寫明的簡化**）：$h_j$ = patch 的 48 維原始像素（順序 py, px, ch），代替論文「進入核心的特徵」；理由：JiT 的 patch 嵌入是像素 patch 的線性映射。細節 = 乾淨影像每 patch 的平均梯度能量（前向差分，最後一列／行差分為 0，三通道相加）。
- 主題：所有 canvas／SVG 圖由單一 `redrawAll()` 重畫；它在主題切換鈕、`matchMedia('(prefers-color-scheme: dark)')` 變更、視窗寬度變更（去抖 150ms）時呼叫；顏色一律在重畫時以 `getComputedStyle(document.documentElement)` 讀取 token。
- canvas：依 `devicePixelRatio` 設定實際像素（CSS 尺寸 × dpr），邊界線寬 = 1 CSS px × dpr。

### C1 · 三種順序下的連續段（S5）

- **教的概念**：Hilbert 的連續段必連通；raster 在行尾、Morton 在二元方格邊界斷開；對齊的 $2^m$ 段在三者都連通，自適應切點落在任意位置時差異才出現（Sec. 3.1、App. B.4、Fig. 14a）。
- **輸入**：分段按鈕「格大小 q」8 / 16 / 32（預設 8，同 Fig. 14a 的 8×8）；滑桿「段長 L」1 … min(64, q²)（預設 8，Fig. 14a 用的長度：R=128 時的平均區域）；滑桿「起點 p」0 … q²−L（預設 4，一段不對齊的 8 格，同時呈現 Morton 跳躍與 raster 換行；改 q 或 L 時夾到範圍內）；勾選「顯示整條曲線」（預設開）。頁面不宣稱預設段就是 Fig. 14a 圖中那一段。
- **顯示**：三個並排小圖（< 768px 縱向堆疊），各畫 q×q 格與該順序的路徑折線（`--ink-3`，1px）；被選段的格填 `--accent-soft`、段內路徑以 `--accent` 2px 畫出、非相鄰步（距離 ≠ 1）以虛線畫出並加「跳躍」標籤；每個連通分量的格內以數字 1、2… 標記（非顏色編碼）。各圖下讀數：連通分量數、spread、「所有長度 L 的段中連通者比例」。
- **計算**：改 q 或 L 時重算三順序的比例（q=32、L=64 時 ≤ 3×961 次 BFS，每次 ≤ 64 格），改 p 只重算選段。
- **觀察**：Hilbert 比例恆為 1.000；預設下 Morton 與 raster 都分成 2 塊；p 改成 0（對齊）時三者都連通；q=32、L=16 時 raster 只有 0.539 的段連通。
- **無障礙**：滑桿有 `aria-valuetext`（「起點 4，段長 8」）；三圖 `role="img"`、`aria-label` 描述分量數；live region 只在 `change` 或停止輸入 500ms 後更新。

### C2 · 內容自適應分割（S6）

- **教的概念**：Eq. 1–2；自適應切割讓平坦區成為大區域、細節區接近單 patch；等距切割大小固定；Feat Sim 的組散落全畫面（spread 大）；skip 刪除得多（EV 低）。三個基準已在 S3 定義，所以在此直接列出。
- **輸入**：分段按鈕「分組方法」自適應（RTI）／等距切割／特徵相似度（Feat Sim）（預設自適應）；分段按鈕「預算 R」64、96、128、192、256、384、512（預設 128）；滑桿「在輸入像素中混入噪聲：t」0–1、步長 0.05（預設 1；去抖 100ms）；在影像上點擊／觸控一個 patch 以突顯其所屬組。
- **鍵盤**：canvas 上覆一個 `tabindex=0` 的透明 `div`（`role="grid"` 不用，改 `role="application"` 加 `aria-label`「影像分割，方向鍵移動、Enter 選取」）；方向鍵移動焦點格，焦點格以 2px `--accent` 方框畫在 canvas 上；Enter／空白鍵選取。
- **t 滑桿的說明**（圖版上一行）：「這裡把噪聲混進輸入像素。論文的切割讀的是模型特徵，模型特徵追蹤的是乾淨影像估計的結構（Sec. 3.3），所以論文的分割在早期不會像這裡一樣變成隨機；此滑桿只示範『特徵不含結構時，切點不再對齊細節』。」
- **計算**（頁面即時算；與 `common.py` 相同）：
  - $h$ ← $x_t$ 的 patch 特徵；$d_j=\lVert h_{\pi(j+1)}-h_{\pi(j)}\rVert_2$（0-based $j=0..N-2$）。
  - 自適應：依 $(-d_j, j)$ 排序取前 $R-1$ 個 $j$，在 $j$ 與 $j+1$ 之間切。
  - 等距：切在 $j=\mathrm{round}(rN/R)-1$，$r=1..R-1$（R = 96、192、384 時組大小為 10 或 11）。
  - Feat Sim（推定細節）：$h$ 減去全域平均後正規化（範數 < 1e-12 視為 0 向量）；錨點 = raster pid $\lfloor rN/R\rfloor$，$r=0..R-1$；每個 patch 取 cosine 最大的錨點（同值取最小 $r$）。App. A 只寫「R 個等距錨點、依 cosine 指派到最近錨點」；中心化與錨點公式是本頁選擇，中心化與否會改變指派結果。
  - Skip（只算 EV；推定）：以 $\lVert h_j-\bar h\rVert^2$ 排序（同值取小 pid）保留前 $R$ 個，其餘換成全域平均 $\bar h$。
  - 指標：EV（Eq. 5，分母 < 1e-18 → 「—」）；spread（Eq. 6，均勻權重，群平均）；size–detail：fineness $=-\log_2|c|$ 與組內平均乾淨細節的 Pearson 相關（任一標準差 < 1e-12 → 0；等距切割在 R 整除 N 時恰為 0，R = 96 時為 −0.034）；最大區域大小。
- **顯示**：
  - 左：canvas（CSS 256×256，`image-rendering: pixelated`）畫 $x_t$（截到 [0,1]）；組邊界畫在相鄰 patch 屬於不同組處，兩層線：外層 3 CSS px 半透明白（alpha .7）、內層 1 CSS px `--ink`（深色主題時內層改 `--bg`），在亮暗底都可見；突顯組以斜線圖樣覆蓋（非顏色）並顯示其大小與 spread。
  - 下：「沿 Hilbert 順序的特徵差距」長條圖，canvas 實作；寬度 $W$ px 時把 1023 根長條分到 $W$ 個像素欄，每欄畫該欄內的最大值（max binning），有切點的欄以 `--accent` 畫並在頂端加三角標記；圖上標「R − 1 = 127 個切點」。等距方法時改畫等距刻線。
  - 右：讀數表，四列（自適應、等距、Feat Sim、Skip）× 欄（EV、spread、size–detail、最大區域）；目前方法列加粗。表下固定說明（照錄）：「玩具場景遠比真實模型特徵容易壓縮：R = 256 時自適應保留 0.995，論文的凍結模型約 0.74（App. C.1：約 26% 回收不了）；論文 Tab. 4 中 Feat Sim 保留 0.74、略高於 RTI 的 0.73，這裡的排序不同。請不要從這張表的 EV 排序下結論；它只示範分組形狀（spread、區域大小）的差別。玩具場景最細節的 15% patch 占 94.6% 細節（論文 Fig. 2a 為 50%）。」
- **效能**：Feat Sim 在 $R=512$ 約 2,500 萬次乘加；特徵依 t 快取（Map，key = t），方法切換不重算特徵。
- **觀察**：(1) $t=1$、$R=128$：自適應的最大區域 192 patch（天空）、spread 0.619，等距每區 8 patch、spread 1.144；(2) Feat Sim spread 6.438，點選其組可見散落全畫面；(3) skip EV 只有 0.242；(4) $t=0$ 時自適應的 size–detail 相關降到 −0.017（$t=1$ 為 +0.523）。
- **無障礙**：覆蓋層 `aria-label` 摘要（方法、R、t、EV、選取組大小）；隱藏的讀數表；觸控點擊等同 hover；live region 去抖 500ms。

### C3 · 區域 token 的權重與位置（S8）

- **教的概念**：READ 的區域內 softmax 權重（S7 已教）；尺寸索引；RoPE 相量平均造成依區域範圍的低通（Sec. 3.4）。
- **簡化**（頁面寫出）：區域 = Hilbert 序列上的一段 $[p,p+L)$；評分 $w^\top h_j$ 以固定替代分數 $\beta s_j$ 表示，$s_j$ 為玩具場景 $\log(\text{detail}_j+10^{-4})$ 對 1024 patch 標準化（推定）；RoPE 每軸 16 個頻率 $\omega_i=10000^{-i/16}$，$i=0..15$（$\omega_0=1$，波長 6.28 格；推定：對應 head 維度 64 的二維軸向 RoPE，論文未給）；位置平均用均勻權重（Sec. 3.4「其 patch 旋轉嵌入的平均」）。
- **輸入**：滑桿「起點 p」0–1023（預設 100）、「段長 L」1–256（預設 37）；$p+L>1024$ 時夾住 p＝1024−L；滑桿「評分銳利度 β」0–4、步長 0.1（預設 1；β = 0 即平均）；chip「L = 1」「L = 4」「L = 16」「L = 64」只改 L、保留目前 p（必要時夾住）。
- **計算**：$\alpha_j=\exp(\beta s_j-m)/\sum$（$m$ 為段內最大值）；READ 權重中心 $p_\alpha=\sum\alpha_jp_j$（只表示 READ 權重集中在哪裡，**不是** token 的位置）；spread（Eq. 6，權重 $\alpha$）；段內最大權重 $\max_j\alpha_j$；索引 $\lfloor\log_2L\rfloor$；每軸每頻率 $|m(\omega)|=\big|\frac1L\sum_je^{i\omega c_j}\big|$（數值求和）；「接近完整的頻率數」＝ $|m|\ge0.9$ 的頻率數（0.9 是本頁選的門檻，推定，S8 正文定義）。
- **顯示**：左：32×32 格縮圖（玩具影像淡化為底），段內格以圓點表示 $\alpha_j$（半徑 ∝ $\sqrt{\alpha_j}$），READ 權重中心以十字標出（標籤「READ 權重中心」）。右（標題「區域 token 的位置：均勻相量平均」）：橫軸波長 $2\pi/\omega$（對數 6.28–35,333 格），縱軸 $|m|\in[0,1]$；x 軸（實線＋圓形標記）、y 軸（虛線＋方形標記）兩條，各有直接標籤；兩條垂直範圍線：「x 範圍」（實線）與「y 範圍」（虛線）＝段在該軸的格數跨度（max − min + 1），各有文字標籤。讀數分兩組：「READ 權重（隨 β 改變）」：最大權重、READ 權重中心、spread（權重 α）；「位置（不隨 β 改變）」：尺寸索引、接近完整的頻率數（x / y）、λ = 6.28 時的 $|m_x|$、$|m_y|$。
- **觀察**：L = 1 時 16/16 個頻率完整；L 增大時短波長分量先縮小；β 增大時 READ 權重集中（最大權重上升、spread 下降、權重中心移動），而右圖的 $|m(\omega)|$ 曲線與「位置」讀數完全不變，因為論文的位置用均勻平均（5.3 中 p=100、L=37 的三列）。
- **無障礙**：圖 `role="img"`、`aria-label` 摘要；滑桿 `aria-valuetext`；live region 去抖 500ms。
- WRITE 不設元件：$g$ 的結構論文未給，無法忠實模擬；S9 以文字與 Eq. 4 說明。

### C4 · 成本模型、速度上限與同時間的步數（S11）

- **教的概念**：只有核心縮短，速度有上限；Tab. 3 的同成本區間；同樣的秒數可換更多步（Sec. 4.2、App. B.3、B.5）。
- **輸入**：分段按鈕「主幹與核心」：B/16 [3,13]（$B=17$，$n_{\text{core}}=11$，N=1024，預設）／B/16 [1,15]（$n_{\text{core}}=15$）／L/16 [3,19]（$B=23$，$n_{\text{core}}=17$）／JiT-B/16 c2i [4,9]（$B=12$，$n_{\text{core}}=6$，N=256）。切換時 R 重設為該主幹預設（JiT 130，其餘 256），只顯示該主幹的論文點。滑桿「預算 R」走離散清單（滑桿值＝清單索引，`aria-valuetext` 為 R 值）：N=1024 的主幹 `[16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]`；JiT `[16, 32, 64, 96, 130, 193, 256]`。清單包含所有預設值、論文預算與 §5 用到的值；曲線本身仍以 R = 16…N 的 200 個對數點連續畫出。滑桿「不縮短的開銷 τ」0–0.30、步長 0.005（預設 0）。
- **計算**：1.4 步驟 9 的模型（$n_{\text{core}}=s_1-s_0+1$，含兩端點），同時算 $\eta=0$、$\eta=1$；速度 = $1/\rho_\tau$；上限 = $1/(\tau+(1-\tau)(B-n_{\text{core}})/B)$，**B/16 [1,15] 不畫上限線**，改顯示「論文未計時；上限高於 2.5×（App. B.5）」。同成本區間（只在 B/16 [1,15] 顯示）：$\rho^\*$ 為 B/16 [3,13] 在 R=256 的成本（同一 $\eta$），$\eta=0$：$R=(\rho^\*B-(B-n_{\text{core}}))N/n_{\text{core}}$；$\eta=1$：$R=N\sqrt{(\rho^\*B-(B-n_{\text{core}}))/n_{\text{core}}}$；根號內 < 0 → 「—」；旁註論文附註（只計影像 token，真正同吞吐量低於 461；未計時）。
- **顯示**：上圖：橫軸 R（對數），縱軸速度 1× 到 $\lceil 1.1\cdot\max(\text{上限},\text{該主幹論文點})/0.5\rceil\cdot0.5$（[1,15] 用模型在 R=16 的值代替上限）；$\eta=0$（實線，標「只算線性層」）、$\eta=1$（虛線，標「只算注意力」），兩線間淡色帶；水平點線標上限；論文點以空心標記與數值標籤：B/16 512→1.84、256→2.15，L/16 512→2.11、256→2.59、128→2.86、64→3.01，標題「論文量測（Tab. 1、Fig. 16）」；JiT 64→1.61、130→1.33、193→1.14 改用十字標記，標題「論文解析值（token 計算量，未計時；App. B.2、Fig. 11）」。圖下說明（照錄）：「τ = 0 時，B/16 的兩個量測點與 L/16 在 R ≥ 128 的量測點落在帶內；L/16 在 R = 64 的 3.01× 低於帶。量測時間比 FLOPs 下降得快（Fig. 10），這個模型只對照比例，不預測吞吐量。」讀數：目前 R 的成本與速度（兩個 η）、上限、同成本區間。
- **下列（獨立小圖版，標題「同樣秒數能跑幾步（固定為 L/16，R = 256）」）**：滑桿「每張影像秒數 S」1.0–12.0、步長 0.1（預設 1.8）；dense $\lfloor S/0.288+10^{-9}\rfloor$、RTI $\lfloor S/0.112+10^{-9}\rfloor$ 兩條長條；配對表分欄：秒數｜論文 dense 步數｜論文 RTI 步數｜本頁 floor dense｜本頁 floor RTI（1.8：6、16、6、16；2.8：10、25、9、25；5.7：20、50、19、50；11.2：39、100、38、100）；另列 GenEval 1.8 s 65.9 vs 82.7、11.2 s 87.6 vs 86.3；附註「論文以成本差 4% 內配對，floor 可能少 1 步」。
- **觀察**：B/16 [3,13] 在 τ = 0 的上限 2.83×；τ = 0.073 時上限 2.50×（B/16 飽和值）；B/16 [1,15] 同成本區間 461–572；L/16 上限 3.83×，量測到 R=64 仍上升。
- **無障礙**：曲線以線型＋直接標籤區分；論文點有文字標籤與兩種標記；live region 去抖 500ms。

---

## 5. 驗收數值表（全部出自 `verify/out_*.txt`）

容許誤差依數量級：計數與整數精確；三位小數讀數 ±0.001（同算法，只差浮點加總順序）；EV、相關 ±0.002；spread ±0.002；同成本 R ±0.1。

### 5.1 C1（`c1_table.py`、`c1_orders.py`）

| q | L | p | Hilbert 分量 / spread | Morton 分量 / spread / 比例 | raster 分量 / spread / 比例 |
| --- | --- | --- | --- | --- | --- |
| 8 | 8 | 4（預設） | 1 / 1.144 | 2 / 1.498 / 0.561 | 2 / 2.093 / 0.140 |
| 8 | 8 | 0 | 1 / 1.144 | 1 / 1.144 / 0.561 | 1 / 2.000 / 0.140 |
| 8 | 4 | 6 | 1 / 0.707 | 2 / 1.144 / 0.525 | 2 / 3.043 / 0.656 |
| 16 | 16 | 40 | 1 / 1.824 | 2 / 2.350 / 0.531 | 2 / 4.057 / 0.066 |
| 32 | 16 | 300 | 1 / 1.718 | 1 / 1.718 / 0.507 | 1 / 4.000 / 0.539 |
| 32 | 64 | 100 | 1 / 3.131 | 2 / 5.000 / 0.533 | 1 / 8.039 / 1.000 |

Hilbert 比例在所有列皆為 1.000。32×32 對齊段平均 spread：L = 4 時 Hilbert = Morton = 0.707（= Tab. 4 fixed boundaries 0.71），raster 1.000。二元方格皆為對齊段（Hilbert、Morton：True）。

### 5.2 C2（`c2_partition.py`；最大區域精確）

| t | R | 自適應 EV / spread / 相關 / 最大 | 等距 EV / spread / 相關 | Feat Sim EV / spread / 相關 | Skip EV |
| --- | --- | --- | --- | --- | --- |
| 1 | 64 | 0.902 / 0.956 / +0.620 / 192 | 0.777 / 1.498 / 0.000 | 0.922 / 7.797 / +0.375 | 0.140 |
| 1 | 96 | 0.960 / 0.740 / +0.575 / 192 | 0.780 / 1.338 / −0.034 | 0.938 / 7.033 / +0.442 | 0.198 |
| 1 | 128 | 0.984 / 0.619 / +0.523 / 192 | 0.809 / 1.144 / 0.000 | 0.959 / 6.438 / +0.343 | 0.242 |
| 1 | 256 | 0.995 / 0.399 / +0.302 / 84 | 0.869 / 0.707 / 0.000 | 0.974 / 5.357 / +0.201 | 0.391 |
| 1 | 512 | 0.997 / 0.121 / +0.040 / 84 | 0.912 / 0.500 / 0.000 | 0.990 / 3.192 / +0.040 | 0.653 |
| 0.5 | 128 | 0.176 / 0.922 / +0.193 / 52 | 0.162 / 1.144 / 0.000 | 0.305 / 10.180 / +0.091 | 0.171 |
| 0 | 128 | 0.139 / 0.914 / −0.017 / 42 | 0.124 / 1.144 / 0.000 | 0.291 / 10.888 / +0.082 | 0.171 |

細節集中度：前 15% patch 占 0.946，前 50% 占 0.998。

### 5.3 C3（`c3_read.py`；「最大權重」欄取代原「參與比」，數值同腳本 `amax`）

| p | L | β | 尺寸索引 | READ 權重中心 (x, y) | spread（α） | 最大權重 | 接近完整 x / y（均勻平均） | $\lvert m_x\rvert,\lvert m_y\rvert$ @ λ=6.28（均勻平均） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1 | 0 | (0.000, 0.000) | 0.000 | 1.000 | 16 / 16 | 1.000, 1.000 |
| 0 | 2 | 1 | 1 | (0.000, 0.485) | 0.500 | 0.515 | 16 / 15 | 1.000, 0.878 |
| 0 | 4 | 1 | 2 | (0.495, 0.503) | 0.707 | 0.260 | 15 / 15 | 0.878, 0.878 |
| 0 | 16 | 1 | 4 | (1.470, 1.477) | 1.489 | 0.069 | 14 / 14 | 0.474, 0.474 |
| 0 | 64 | 1 | 6 | (4.562, 4.739) | 3.228 | 0.194 | 13 / 13 | 0.197, 0.197 |
| 100 | 37 | 0 | 5 | (10.703, 6.595) | 2.880 | 0.027 | 13 / 13 | 0.429, 0.287 |
| 100 | 37 | 1（預設） | 5 | (9.463, 7.716) | 2.008 | 0.136 | 13 / 13 | 0.429, 0.287 |
| 100 | 37 | 4 | 5 | (9.172, 7.079) | 0.869 | 0.363 | 13 / 13 | 0.429, 0.287 |

封閉式檢查：p = 0、L = 2（直線段）時 $|m_y(\omega)|=|\cos(\omega/2)|$，最大差 1.1e−16。S8 數值例：$\lvert\cos(0.5)\rvert=0.878$、$\lvert\cos(0.005)\rvert=0.99999$。

### 5.4 C4（`c4_cost.py`）

| 項目 | 本頁值 | 論文 |
| --- | --- | --- |
| [1,15]/[3,13] @ R=256，η=0 | 0.657 | 0.66×（Tab. 3） |
| 同上，η=1 | 0.439 | 0.44× on attention（App. B.5） |
| 同成本 R（η=0 / η=1） | 460.8 / 572.4 | ≈461–572（只計影像 token） |
| 上限 τ=0：B/16、L/16 | 2.833 / 3.833 | B/16 約 2.5× 飽和 |
| τ 使 B/16 上限 = 2.5 | 0.0727 | — |
| 模型帶 τ=0（η=0–η=1）：B/16 R=512、256 | 1.478–1.943、1.943–2.542 | 1.84、2.15 |
| 模型帶 τ=0：L/16 R=512、256、128、64 | 1.586–2.244、2.244–3.257、2.831–3.671、3.257–3.791 | 2.11、2.59、2.86、3.01 |
| JiT c2i R=64/130/193（η=0） | 1.600 / 1.326 / 1.140 | 1.61 / 1.33 / 1.14（解析值） |
| 步數 S=1.8/2.8/5.7/11.2 | dense 6/9/19/38，RTI 16/25/50/100 | 6/10/20/39 vs 16/25/50/100 |

### 5.5 selfTest（`?test`）

- `hilbertXY(32, j)` 相鄰步距皆為 1；`mortonXY(8,3)=(1,1)`、`mortonXY(8,4)=(2,0)`。
- C1 第一列與 q=32、L=16 raster 比例 0.539。
- 場景：`mulberry32(1)` 前三個值 0.6270739405881613、0.002735721180215478、0.5274470399599522；細節集中度 0.946。
- C2：$t=1$ 的 R=128 與 R=96 兩列（含等距相關 −0.034）。
- C3：預設列；L=2 Dirichlet 檢查 < 1e−12。
- C4：0.657、0.439、460.8、572.4、1.600、L/16 R=64 η=0 為 3.257。
- 無水平捲動：`document.documentElement.scrollWidth <= innerWidth`。輸出 `selfTest: N passed, M failed`。

---

## 6. 實作計畫

### 6.1 沿用與新增

單一檔案 `tut/217/build/index.html`，複製 `tutorials/prediction-targets-k-diff.html` 的外殼：CSS tokens（含深色模式）、appbar 與主題切換（`pm-theme`）、`#toc`／`#tocbar`、`#pm-bar`、`.plate`／`.plate-h`／`.plate-lead`／`.plate-f`、`.fact`、`.tag`（「本頁推導」「推定」兩種 pill）、`<details>` 檢查題、KaTeX 0.16.9 auto-render、`renderTimeline`、`initToc`、`initTheme`、`el`、`selfTest` 框架、Google Fonts。刪除參考頁 C1–C4 的程式與文案。

新增：
- 共用：`G, N, PX`、`hilbertXY(q,j)`、`mortonXY(q,j)`、`rasterXY(q,j)`、`orderCache`（key `${name}|${q}`）、`components(cells)`、`spreadOf(cells, w)`、`redrawAll()`、`tok(name)`（getComputedStyle 讀 token）、`setupCanvas(cv, cssW, cssH)`（dpr）。
- 場景：`buildScene()` → `{img, eps, detail, score}`，載入時算一次。
- C2：`featuresAt(t)`、`gapsOf(h)`、`labelsAdaptive`、`labelsFixed`、`labelsFeatSim`、`evOf`、`evSkip`、`groupStats`、`renderC2()`、`drawGapBars(cv, gaps, cuts)`（max binning）、`c2Focus`（鍵盤焦點格）。
- C3：`ropeOmegas`、`readRegion(p,L,beta)`、`renderC3()`。
- C4：`CORES`（b16、b16long、l16、jit）、`relCost(R,cfg,eta,tau)`、`isoR(cfg,eta)`、`rFromIndex(k,N)`、`MEASURED`、`ANALYTIC_JIT`、`renderC4()`、`renderSteps()`。
- 靜態 SVG：S4 架構圖、hero 8×8 Hilbert 曲線（由 `hilbertXY(8,·)` 產生，切點固定在位置 15、31、47；描繪動畫見第 3 節）。
- 數學排版：每個 display 公式包在 `<div class="eqwrap" tabindex="0">`（`overflow-x:auto`）；Eq. 3 在 < 600px 時拆成兩行（$\alpha_j$ 與 $z_c$ 各一行，用兩個 `$$`）；動態插入或 `<details>` 展開的內容呼叫 `renderMath(node)`。
- 色彩：不新增 token；三種順序與 η=0/1 以線型區分；C2 不做群組著色（只畫邊界）。

### 6.2 連結

- 論文：`../index.html#paper-217`；arXiv `https://arxiv.org/abs/2608.29281`；專案頁 `https://eduardzamfir.github.io/rti`。
- 系列與姊妹：`token-compression-lineage.html`（總覽筆記 `tokc-lineage-hub.html`）；`hit-hierarchical-diffusion-transformer.html`（`../index.html#paper-585`）。
- 本頁筆記（依總覽閱讀順序，S16 清單照此順序並附 frontmatter `title`）：`rti-lineage-hub`（總覽）、`rti-redundancy-probe-four-requirements`、`rti-prior-reductions-compared`、`rti-hilbert-order-region-partition`、`rti-read-write-interface`、`rti-elastic-budget-lora-training`、`rti-results-and-limits`。
- 系列筆記（只連結，不複寫）：`tokc-vit-pruning-and-merging`、`tokc-training-free-diffusion-merging`、`tokc-hourglass-and-latent-interfaces`、`tokc-content-adaptive-tokenization`、`tokc-dynamic-chunking-bytes-to-pixels`、`tokc-learned-routing-dynamic-dit`、`tokc-position-for-merged-tokens`。
- 既有筆記：`pixel-space-diffusion`、`x-prediction-manifold-view`、`depth-redundancy-adjacent-methods`、`recursive-transformer-followups`。
- 各節延伸閱讀位置見第 3 節表格。

### 6.3 文案與排版規則

- 繁體中文、台灣用語、字面學術語氣；不用比喻、反問、宣傳詞。避免「付代價／付清」「旋鈕」「抗壓縮」「任何預算」等說法，改寫為「計算量」「可調參數」「不適合壓縮」「訓練範圍（64–512）內的預算」。專有名詞首次出現附英文（區域 token（region token）、連續段（run）、二元方格（dyadic cell））。
- 區分論文結果、作者解讀（「作者認為」）、〔本頁推導〕、推定。每個論文數字附出處；元件讀數以圖版底一行說明來源。
- 頁面不得出現本設計文件以外的數字；SVG 與讀數用 Unicode，不用 TeX。
- 四欄以上表格放 `overflow-x:auto` 容器；Tab. 1 在 < 480px 時隱藏 CLIP/Pick/IR 欄並提供「顯示全部欄位」切換。

### 6.4 時間線（S15）

- 來源 `tut/217/notes/timeline.json`（12 筆），建置時原樣嵌入 `<script type="application/json" id="timeline-data">`；格式 `{year, title, authors_short, arxiv, role, source}`，依 year 排序（同年保持檔案順序）。
- `TL_GROUP`（key＝arXiv ID）與卡片連到的筆記：

  | 類別 | arXiv ID | 筆記 |
  | --- | --- | --- |
  | 基礎（DiT） | 2212.09748 | `rti-lineage-hub.html` |
  | 相似度合併 | 2210.09461 | `tokc-vit-pruning-and-merging.html` |
  | 擴散中的合併 | 2303.17604、2402.13573 | `tokc-training-free-diffusion-merging.html` |
  | 自適應 tokenization | 2304.00287 | `tokc-content-adaptive-tokenization.html` |
  | latent 瓶頸／介面 | 2212.11972、2603.12245 | `tokc-hourglass-and-latent-interfaces.html` |
  | token 跳過 | 2404.02258 | `rti-prior-reductions-compared.html` |
  | 動態壓縮比 | 2412.16822 | `tokc-learned-routing-dynamic-dit.html` |
  | 動態切塊 | 2603.06351 | `tokc-dynamic-chunking-bytes-to-pixels.html` |
  | 像素空間 DiT | 2511.13720 | `pixel-space-diffusion.html` |
  | 本文 | 2608.29281 | `rti-lineage-hub.html` |

  不在表中的條目歸「其他」並在建置回報列出。
- 呈現沿用參考頁：橫捲卡片列、類別篩選 chip（`aria-pressed`，未選淡化）、左右按鈕；本文卡片以 `--accent` 外框；JSON 解析失敗時整節顯示「時間線資料尚未提供」。

### 6.5 建置後自我檢查

截圖 390 / 768 / 1024 / 1440、深色、鍵盤 focus、reduced-motion；依字體、留白、視覺層級、色彩、動態、微互動、響應式、原創性評分並修正。切換主題確認所有圖重畫。跑 `?test`；逐條核對 12 個學習目標的答案出現在指定章節正文。390px 時 C2 canvas 為 100% 寬、長條圖以 max binning 重畫。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 玩具特徵不是模型特徵 | C2/C3 用原始像素 patch；EV 與排序和論文不同（玩具 R=256 保留 0.995，論文約 0.74）。 | C2 表下固定說明「不要從 EV 排序下結論」；論文數字只在正文。 |
| R2 | Skip、Feat Sim 的細節 | App. C.1/A 未定義 skip 得分與平均、Feat Sim 是否中心化與錨點公式。 | 標推定並註明中心化會改變指派。 |
| R3 | RoPE 參數 | 論文未給底數、head 維度、二維形式。 | 標推定；結論只依賴定性行為。 |
| R4 | 成本模型與量測 | 只對照比例；JiT 值是論文解析值。 | C4 分兩種標記與標題；圖下說明帶內／帶外情形。 |
| R5 | 步數 floor | 論文以 4% 內配對。 | 配對表照錄論文並附註。 |
| R6 | 摘要「2.0× 追平、2.6× 接近」 | 2.0× 對應 B/16 R=384；2.6× 對應 L/16 R=256（86.3 vs 88.1）。〔本頁推導：對應關係〕 | 正文寫明出處。 |
| R7 | 論文未說明的細節 | $g$、$e_{\text{size}}$ 維度、READ 溫度、核心編號是否 0-based。 | 注意事項第 8 條；核心照錄「[3, 13] of 17，三塊 prelude 與 coda」。 |
| R8 | t 滑桿的意義 | 混入的是輸入像素噪聲，不是模型特徵的狀態。 | 圖版說明（C2）；保留滑桿，因為它示範「特徵無結構時切點不對齊細節」（App. C.2 最吵步驟相關接近 0）。 |
| R9 | HiT 陳述 | 只取 HiT 摘要與 Sec. 3.1–3.3。 | 建置前與 `hit-hierarchical-diffusion-transformer` 頁核對。 |
| R10 | Fig. 2/4/9 曲線無數據 | 論文只給文字數字。 | 不重繪曲線。 |
| R11 | Feat Sim 在 JS 的成本 | R=512 約 2,500 萬乘加。 | 特徵快取、去抖；單次 > 50ms 時改用 `requestIdleCallback` 並先顯示其他三列。 |
| R12 | Tab. 1 在手機 | 9 欄。 | 窄螢幕欄位隱藏與切換。 |

---

## 8. 修訂紀錄（round 1）

| 審查項 | 處理 |
| --- | --- |
| 1 JiT 標為量測 | C4 改十字標記與「論文解析值（未計時）」；步驟 9 註明只是解析值彼此一致。 |
| 2 上限過度推廣 | 步驟 9、注意事項 4、LO11 限定 B/16；加 L/16 對照。 |
| 3 同成本附註、8.5× | 步驟 9 與 C4 加 App. B.5 附註；[1,15] 不畫上限、顯示「高於 2.5×」；y 軸自動縮放。 |
| 4 39.0 歸於位置 | 步驟 10、注意事項 3、LO12 改「移除分組與位置（上界）」。 |
| 5 t 滑桿 | 標籤改「在輸入像素中混入噪聲」，圖版說明；R8 記錄保留理由。 |
| 6 玩具 EV 與 26% 矛盾 | C2 表下固定說明；R1。 |
| 7 S2／Tab. 4 用未教術語 | S1 定義區塊與取樣步；S2 以 2×2 平均教 EV；S3 先定義三基準、retained、size–detail、spread。C2 的 Feat Sim／Skip 不另收進 `<details>`：S3 已定義，且比較正是 C2 的目的。 |
| 8 等距相關恆為 0 | 改「R 整除 N 時為 0」；新增 R=96 列（−0.034）與 selfTest。 |
| 9 S7 過密 | 拆為 S7 READ、S8 位置（含兩位置數值例，C3 在此）、S9 WRITE；WRITE 不設元件的理由寫在 C3 末。 |
| 10 符號表 | 改為各節「本節符號」小框；總表只留在設計文件。 |
| 11 Hero 術語 | 標題與三張卡改白話。 |
| 12 比喻／誇大 | 1.1、1.2、步驟 4、步驟 8 改寫；6.3 列禁用詞。 |
| 13 C2 鍵盤／canvas | 覆蓋層 tabindex、焦點框、dpr、雙層邊界線、長條圖 max binning。 |
| 14 主題重畫 | 4.0 `redrawAll()`。 |
| 15 C4 控制 | 對數滑桿索引公式、換主幹重設 R、步數圖版標「固定為 L/16，R = 256」。 |
| 16 C3 讀數與符號衝突 | 參與比換成最大權重；0.9 門檻標推定並在 S8 定義；$q$、$\rho$、$\nu$、$\eta$ 改名。 |
| 17 KaTeX 窄螢幕 | 6.1 `.eqwrap`、Eq. 3 拆行、`renderMath`。 |
| 18 帶內陳述 | C4 圖下說明與 5.4 改為精確情形。 |
| 19 Feat Sim 推定 | C2 標推定並註明中心化影響。 |
| 20 Tab. 4 背景 | 步驟 3 加 retained 下界說明；連通列加 Latent Array。 |
| 21 設計選擇 vs 錯誤 | 步驟 12 分兩組。 |
| 22 Dirichlet 範圍 | 步驟 7 加限制；C3 用數值求和。 |
| 23 Fig. 14a 段 | C1 預設 p=4（不對齊）並寫明不宣稱是圖中那段。 |
| 24 C3 chips 與範圍線 | chip 只改 L；兩條範圍線以線型區分。 |
| 25 6.3 規則與動畫 | 改「不得出現設計文件以外的數字」；hero 曲線描繪動畫（尊重 reduced-motion）。 |
| 26 verify 執行方式 | 腳本加 `sys.path` 插入，可用 `python3 -I` 執行；已重跑並更新 `out_*.txt`（C2 加 R=96、C4 加模型帶）。 |
| 筆記 | 6.2 填入 7 篇 `rti-` 筆記、7 篇 `tokc-` 系列筆記、系列與 HiT slug；6.4 依 timeline.json 填 `TL_GROUP`。 |

## 9. 修訂紀錄（readiness 後）

- C3：中心改稱「READ 權重中心」（隨 β），位置讀數改為均勻相量平均（不隨 β），讀數分兩組；觀察與 5.3 表頭一致（數值不變，`c3_read.py` 已含兩者）。RoPE 索引 $i=0..15$、$\omega_0=1$。
- C4：R 滑桿改離散清單（含 256、512、130、193 等所有 §5 值）；$n_{\text{core}}=s_1-s_0+1$ 寫入步驟 9、S11 符號框與 C4 輸入；S4 加 $s_0,s_1$ 符號框；步數表分「論文」與「本頁 floor」欄。
- LO12：註明 Token Skip 不在同預算（1.19× vs 2.15×）。

## 10. 建置紀錄（build round）

- 產出 `build/index.html`（約 116 KB）；原始片段在 `build/src/`（body.html、extra.css、app.js），組裝方式：參考頁 head 與 CSS 第 14–340 行＋extra.css＋body＋timeline.json＋app.js。
- `?test`：61 passed, 0 failed（1440 與 390 iframe 皆同；390 時 scrollWidth 375 ≤ 390）。impeccable：0 anti-pattern，16 項 numbered-section-labels advisory（DESIGN.md 的目錄式編號，保留）。LO 自檢：12 項參考答案的關鍵詞與數字都在 `accept/page.txt` 正文中。
- 與設計的差異：本節符號以常駐的虛線框呈現（不用 `<details>`，避免與 `details.check` 混淆）；C4 的步數圖版獨立為 C4b；C3 左圖底圖使用 C2 目前的 $x_t$ 影像；C1 在 q = 32 時不標分量數字（格太小），分量數只在讀數中顯示；390px 時 C2 canvas 改以 100% 寬度＋devicePixelRatio 重算。

- 驗收後補丁（10/12）：S5 二元方格句改寫為「每個二元方格恰好是 Hilbert 序列上的一段，任何 quadtree 分割都能用切割表達」（同段）；S9 寫明消融替換整個 READ/WRITE，−7.2 在同段；S11 每個速度與步數旁標 L/16 或 B/16；LO11 參考答案標 L/16。
