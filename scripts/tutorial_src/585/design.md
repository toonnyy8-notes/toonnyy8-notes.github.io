# 設計文件：〈Hierarchical Diffusion Transformer for End-to-End Image Generation〉（HiT）互動教學

- 論文：Xiaoyu Wu、Yifei Wang、Weihang Guo、Chen Wei、Lydia E Kavraki、Niloofar Mireshghallah、Zhiwei Steven Wu，arXiv 2611.22962（v1，2026-11-17）。資料庫編號 No. 585。
- 原文：`tutorial_src/585/src/paper.txt`。本文件的「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」皆指論文原文編號。沒有 advisor 備忘錄（`advisor.md` 不存在）。
- 產出：`tutorial_src/585/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/hit-hierarchical-diffusion-transformer.html`（slug `hit-hierarchical-diffusion-transformer`），所有站內相對連結以上線位置為準。
- 驗算：`tutorial_src/585/verify/check.py`（`python3 -I check.py`），與頁面相同的演算法；第 5 節數值全部由它印出。
- 參考實作：`tutorials/prediction-targets-k-diff.html` 與 `tutorial_src/238/design.md`（頁面外殼、tokens、元件樣式、selfTest、KaTeX 設定照用）。
- 文件慣例：本設計自行推導、論文沒有寫出的結論標「〔本頁推導〕」；為了教學而自訂、論文沒有給出的數值標「推定」。頁面用同樣的標籤（6.6）。
- 同批教學：系列教學 `token-compression-lineage.html`（token merging／壓縮／階層的研究脈絡，筆記前綴 `tokc-`）與兄弟教學 `rti-elastic-token-compression.html`（No. 217，〈Elastic Token Compression for Pixel-Space Diffusion Transformers〉，筆記前綴 `rti-`）。共同背景交給系列教學與其筆記，本頁只講 HiT 新增的部分。本頁筆記總覽為 `hit-lineage-hub`。
- 修訂：round 1 審查（`reviews/round1.md`）的處理見第 8 節。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

潛空間擴散模型先用另外訓練的編碼器把影像整理成精簡表示，再在那個空間學生成；像素空間（pixel-space）擴散則一個階段端到端訓練，但品質與訓練效率仍落後。作者假設差距的一部分來自「骨幹從頭到尾每個 patch 保留一個 token」，沒有鼓勵模型形成由細到粗的視覺階層（Abstract、Sec. 1）。HiT 在 Transformer 中加入兩次**分組（grouping）**：用學到的親和度（affinity，Eq. 2）以平均連結法（average linkage）把相關 token 合併成群組，每個群組以軟權重（soft weights，Eq. 9）對**所有**輸入 token 加權平均成一個群組 token（Eq. 3、6）；中間的 Transformer 只處理較短的群組序列，再用同一組軟權重**解組（ungrouping）**回原位置並加上跳接（Eq. 3、7）。群組數不固定：**壓縮損失**（Eq. 4）只要求整個批次的平均保留比例等於目標 $\rho=1/r$，所以每張影像、每個擴散時間的群組數可以不同。ImageNet 256 上 HiT-XL/16 在 600 epoch 達到 FID 1.48，在 FID ≈ 2.16 時比 JiT-H/16 少用 2.4 倍訓練計算、2.3 倍更新次數（Fig. 1d、Table 1）；文字到影像上 HiT-B/16 五項基準平均 65.93，高於 MiniT2I-B/16 的 64.64，forward FLOPs 少 31%（Table 2）。模型在沒有分割標註的情況下學會把物件、物件部位與背景分組，並在文字到影像中把字詞與對應影像區域放進同一群組（Sec. 4.4、Fig. 1、App. D）。

### 1.2 研究動機（Sec. 1）

1. 潛空間模型的階層來自編碼器預訓練，視覺組織是在生成訓練之外決定的。
2. 像素空間模型去掉這個階段，就必須在生成訓練中自己形成階層。論文的核心問題：「像素空間模型能否在生成訓練中學到有用的視覺階層，同時保持端到端？」
3. 規則的 patch 網格給每個等大區域相同數量的 token，不管內容多寡（引 Dieleman [12]）。作者因此主張要**學**分配方式，而不是用固定的空間切割決定。

### 1.3 貢獻

1. HiT 架構：遞迴的編碼器／精簡網路／解碼器，token 數 $N\to K_1\to\dots\to K_J\to\dots\to K_1\to N$（Eq. 1、Fig. 2），靈感來自 H-Net [20]。
2. 以對稱親和度＋平均連結法得到硬分組、以軟權重做池化與解組（Sec. 3.2、App. A.1），讓空間上不相連的區域也能共用一個 token。
3. 批次層級的壓縮損失（Sec. 3.3、App. A.2），群組成員與群組數隨影像與擴散時間改變。
4. 群組引導（group guidance，App. A.6）：用不同分組容量的預測做對比，不增加 CFG 以外的網路評估次數。
5. 實驗：ImageNet 256 FID 1.48；文字到影像五項基準；Table 3–12 的消融；學到的群組可視化。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S1 末尾以可收合表格提供）

| 符號 | 意義 |
| --- | --- |
| $x$、$\epsilon$、$t$ | 乾淨影像、標準高斯噪聲、擴散時間；$x_t=tx+(1-t)\epsilon$，$t=1$ 為乾淨影像、$t=0$ 為純噪聲（App. A.3、A.6） |
| $N$ | 某分組階段輸入的 token 數；ImageNet 256、patch 16 時第一階段 $N=256$（文字到影像 512 px 時為 1024 個影像 token 加有效字幕 token，見步驟 11） |
| router | 每個分組階段的分組模組：兩個投影產生 $q,k$，再做親和度、合併樹與軟權重（Sec. 3.1–3.2） |
| $d$ | 特徵寬度（B：768，L：1024，XL：1280；Table 8），所有階段相同 |
| $J$、$\ell$ | 分組層數（預設 $J=2$）、第 $\ell$ 層 |
| $K_\ell$、$K$ | 第 $\ell$ 層的群組數；單一階段內簡寫 $K$ |
| $H\in\mathbb R^{N\times d}$、$h_i$ | 分組前的 token 特徵與其第 $i$ 列 |
| $q_i,k_i$ | 兩個學到的投影，各自正規化為單位長度 |
| $S_{ij}$ | 對稱親和度（Eq. 2），範圍 $[-1,1]$ |
| $s(U,V)$ | 兩群組成員間的平均親和度（Eq. 8） |
| $\tau$ | 合併門檻：只保留分數 $\ge\tau$ 的合併 |
| $G_g$、$g(i)$ | 第 $g$ 個硬群組；token $i$ 所屬的硬群組 |
| $e_{ig}$、$A$ | token $i$ 對群組 $g$ 的平均親和度；軟權重 $A\in\mathbb R^{N\times K}$，每列和為 1（Eq. 9） |
| $T_r$、$T_b$ | 軟權重溫度（0.05）、群組數估計的 sigmoid 溫度（0.1）（Sec. 4.1） |
| $P$ | 池化矩陣 $P_{ig}=A_{ig}/\sum_jA_{jg}$，每行（column）和為 1 |
| $Z$、$M$、$\tilde z_g$ | 群組 token $Z=P^\top H$；精簡網路 $M$；$M(Z)$ 的第 $g$ 列 |
| $W_\ell$ | 第 $\ell$ 層跳接投影，$d\times d$，初始化為單位矩陣 |
| $r$、$\rho$ | 目標壓縮比（$r>1$）與目標保留比例 $\rho=1/r$；預設每層 $r=2$ |
| $f_h$、$f_s$ | 實際（硬）保留比例與其可微估計（軟） |
| $N_b,K_b,\tilde K_b$ | 批次中第 $b$ 個樣本的輸入 token 數、硬群組數、軟群組數估計（Eq. 11） |
| $s_{b,m}$ | 第 $b$ 個樣本完整合併樹第 $m$ 次合併的平均親和度 |
| $\lambda$ | 壓縮損失權重（0.01） |
| $r_c,r_u$、$s$ | 群組引導中條件／無條件分支的群組數倍率；引導強度（App. A.6） |

#### 步驟 1：背景——像素空間 x-prediction 的骨幹（Sec. 3.1、App. A.3–A.4）

- HiT 沿用 JiT [24]：網路 $f_\theta$ 預測乾淨影像，以速度損失訓練，寫成
  $$\mathcal L_{\text{diff}}=\mathbb E_{x,c,t,\epsilon}\left[\frac{\mathrm{MSE}\big(f_\theta(x_t,t,c),x\big)}{\max(1-t,a_{\min})^2}\right],\quad a_{\min}=0.05\tag{Eq. 12}$$
- 線性 patch 投影把噪聲影像切成 $N$ 個 token；最後的線性 head 輸出乾淨影像。保留 RMSNorm、QK 正規化、SwiGLU；去掉 JiT 的輸入 bottleneck（App. A.4）。
- 頁面不重講 x-prediction 與 flow matching，改連 `x-prediction-manifold-view.html`、`pixel-space-diffusion.html`、`aniso-pixel-patchified-setting.html`。
- 下一步的理由：JiT 的每一層都處理 256 個 token；HiT 要改的正是這個固定的 token 數。

#### 步驟 2：遞迴的階層骨幹（Sec. 3.1、Fig. 2、Eq. 1）

$$N\to K_1\to\cdots\to K_J\to\cdots\to K_1\to N\tag{Eq. 1}$$

- 每一層：編碼器 Transformer 區塊（不改 token 數）→ 分組 → 更深一層 → 解組 → 加上投影後的編碼器特徵（跳接）→ 解碼器區塊。最深處 $L$ 個區塊不再分組。每層有自己的 router（分組模組，步驟 3）與跳接投影 $W_\ell$（步驟 4）。頁面 S2 的圖只標「分組」「解組」「跳接」三個詞，並在圖說中各用一句定義，不出現 router、$W_\ell$。
- 五段 layout「3/3/6/3/3」依序是全解析度編碼器、中間編碼器、精簡核心、中間解碼器、全解析度解碼器的區塊數（Sec. 4.5）。ImageNet 消融的 HiT-B/16 參考設定：18 區塊（D18）、寬 768、兩次 2× 分組，目標 token 數 $256\to128\to64$（Table 4）。這組數字只適用於 ImageNet 256；文字到影像見步驟 11。
- 〔本頁推導〕「區塊 × token」總量：3/3/6/3/3 為 $3\cdot256+3\cdot128+6\cdot64+3\cdot128+3\cdot256=2688$；同樣 18 個區塊不分組為 $18\cdot256=4608$。這只是線性部分的比例，C3 給完整的推定成本模型。
- 下一步的理由：骨幹需要一個決定「哪些 token 合併、合併成幾個」的運算。

#### 步驟 3：親和度與硬分組（Sec. 3.2、Eq. 2、Eq. 8、Fig. 3）

$$S_{ij}=\tfrac12\big(q_i^\top k_j+q_j^\top k_i\big)\tag{Eq. 2}$$
$$s(U,V)=\frac1{|U||V|}\sum_{i\in U}\sum_{j\in V}S_{ij}\tag{Eq. 8}$$

- $q_i,k_i$ 是單位向量，所以 $q_i^\top k_j$ 是兩者夾角的餘弦（cosine similarity），範圍 $[-1,1]$。因為 $q\ne k$，$q_i^\top k_j$ 一般不等於 $q_j^\top k_i$；Eq. 2 取兩者平均，使 $S$ 對稱，合併才不依賴順序。$S_{ii}=q_i^\top k_i$ 一般不等於 1。
- 平均連結法（average linkage，一種由下而上的階層式分群）：從每個 token 自成一群開始，反覆合併 $s(U,V)$ 最高的兩群，得到一棵完整的合併樹（$N-1$ 次合併；畫成樹狀圖時，縱軸是合併分數）。保留分數 $\ge\tau$ 的合併，得到 $K$ 個硬群組 $\{G_1,\dots,G_K\}$（Sec. 3.2 原文：「We retain merges with affinity at least τ」）。
- 三個 token 的計算例（頁面 S3 開頭，數字由 `check.py` 算出；角度表示的二維單位向量，$q=k$）：角度 10°、22°、150° 時 $S_{12}=0.9781$、$S_{13}=-0.7660$、$S_{23}=-0.6157$；第一次合併 {1,2}（分數 0.9781）；第二次 $s(\{1,2\},\{3\})=(-0.7660-0.6157)/2=-0.6909$。$\tau=0.8$ 時只保留第一次合併，得到 2 個群組 {1,2}、{3}。
- 〔本頁推導，假設精確的平均連結法〕合併分數沿合併順序不增（可約性，reducibility），所以「保留分數 $\ge\tau$ 的合併」等於「最高分數低於 $\tau$ 時停止」。`check.py` 對 200 組隨機輸入確認合併分數不增。頁面放在可收合區塊。
- 親和度只看特徵，不看位置：相距很遠的 token 也能同群（Fig. 4c 的車身標記、Fig. 1 兩根香蕉同屬黃色群組）。與 H-Net 沿因果序列切塊不同，HiT 用雙向影像特徵上的親和度分組（Sec. 2）。
- 門檻 $\tau$ 訓練前校準後固定（App. A.3，步驟 6）；即使 $\tau$ 固定，群組成員與群組數也會隨每一步的特徵改變。
- Fig. 3 的範例（6 個 token、$\tau=0.8$、得到 $G_1=\{1,2,4\}$、$G_2=\{3,5,6\}$）論文註明「values are illustrative」。

#### 步驟 4：軟權重、池化、解組與跳接（Sec. 3.2、Eq. 3、6、7、9、App. A.1）

硬分組是離散的，梯度無法經由群組指派傳回親和度投影。作者用軟權重做池化與解組：

$$e_{ig}=\frac1{|G_g|}\sum_{j\in G_g}S_{ij},\qquad A_{ig}=\frac{\exp(e_{ig}/T_r)}{\sum_{g'=1}^K\exp(e_{ig'}/T_r)}\tag{Eq. 9}$$
$$Z=P^\top H,\qquad \tilde H=A\,M(Z)+HW_\ell^\top,\qquad P_{ig}=\frac{A_{ig}}{\sum_jA_{jg}}\tag{Eq. 3}$$

逐列寫出：$z_g=\sum_iA_{ig}h_i/\sum_iA_{ig}$（Eq. 6），$r_i=\sum_gA_{ig}\tilde z_g$（Eq. 7）。

- $P$ 的每一行和為 1，所以每個群組 token 是**所有**輸入特徵的加權平均（不是只有硬群組成員）。
- 解組重用分組前算好的 $A$；它恢復 token 位置，但不反轉有損的池化（App. A.1）。
- $W_\ell$ 初始化為單位矩陣；反向傳播時硬分割視為常數，梯度經由 $A$（在 $P$ 與解組兩處）傳到 $q,k$（Sec. 3.2）。
- 跳接投影不一定有幫助：B.2 的 12 區塊 adaLN 模型中，直接相加 $R+H$ 的 FID 11.65 優於預設的投影跳接 $R+HW_s^\top$ 的 12.27（Table 10）；作者保留投影是因為它是「簡單的融合規則」，允許轉換編碼器特徵，同時保留重建特徵的直接路徑。頁面寫這一句，不暗示 $W_\ell$ 改善品質。
- $T_r$ 是 softmax 的溫度：$T_r$ 越小，最大的 $e_{ig}$ 分到的權重越接近 1。
- 條件 token（8 個 class token＋1 個 time token）繞過分組與解組，不計入壓縮預算與跳接，直接陪同精簡序列進入注意力（App. A.4）。
- Table 3 的證據（HiT-B/16，80 epoch，FID）：只在硬群組內平均（均勻權重）10.32；硬群組內但用學到權重 10.16；所有 token＋學到權重＋硬查表解組（$r_i=\tilde z_{g(i)}$）7.97；預設（所有 token＋軟解組）6.61。作者結論：在聚合與讀回兩個方向都跨越硬群組邊界交換資訊較好。
- 頁面 C1（S4）讓讀者在 6 個 token 上即時看到 $S$、合併樹、硬群組、$A$ 與 $P$。

#### 步驟 5：壓縮損失（Sec. 3.3、Eq. 4–5、Eq. 10–11、App. A.2）

$$\mathcal L^{(\ell)}_{\text{comp}}=1+\frac{(f_h-\rho)(f_s-\rho)}{\rho(1-\rho)}\tag{Eq. 4}$$
$$\mathcal L^{(\ell)}_{\text{comp}}=\frac r{r-1}\big[(r-1)f_hf_s+(1-f_h)(1-f_s)\big]\tag{Eq. 10，H-Net 原式}$$
$$\tilde K_b=N_b-\sum_{m=1}^{N_b-1}\sigma\!\Big(\frac{s_{b,m}-\tau}{T_b}\Big),\qquad f_h=\frac{\sum_bK_b}{\sum_bN_b},\quad f_s=\frac{\sum_b\tilde K_b}{\sum_bN_b}\tag{Eq. 11}$$
$$\mathcal L=\mathcal L_{\text{diff}}+\lambda\sum_\ell\mathcal L^{(\ell)}_{\text{comp}}\tag{Eq. 5、23}$$

- $f_h,f_s$ 的分子分母都在整個批次與所有裝置上加總（App. A.3）。第二階段的 $N_b$ 是第一階段實際產生的群組數。
- $\sigma(x)=1/(1+e^{-x})$ 是 sigmoid；Eq. 11 把每次合併「是否 $\ge\tau$」的 0/1 判斷換成 0 到 1 之間的平滑值，$T_b$ 越小越接近 0/1。
- $f_h$ 視為常數，梯度只經由 $f_s$ 傳到合併分數 $s_{b,m}$（樹的拓樸固定，但合併親和度可微）。
- 〔本頁推導〕兩式相等：兩者都化成 $\big[f_hf_s+\rho(1-f_h-f_s)\big]/\big(\rho(1-\rho)\big)$（代入 $r=1/\rho$）。`check.py` 對多組值確認差 < 1e-12。
- 〔本頁推導〕$\partial\mathcal L^{(\ell)}_{\text{comp}}/\partial f_s=(f_h-\rho)/(\rho(1-\rho))$：$f_h>\rho$（群組太多）時為正，梯度下降會降低 $f_s$，也就是提高合併分數、鼓勵合併；$f_h<\rho$ 時相反；$f_h=\rho$ 時損失恆為 1、梯度為 0。這正是論文 Sec. 3.3「When $f_h>\rho$, the loss encourages more merging」的來源。
- 〔本頁推導〕若 $f_h=f_s=f$，$\mathcal L=1+(f-\rho)^2/(\rho(1-\rho))\ge1$。
- 因為只約束批次平均，群組數可以「隨影像內容與擴散時間改變」（Sec. 3.3 原文）。論文沒有報告「影像複雜度與群組數」的定量關係；頁面 C2 用合成樣本示範「平均達標、個別不同」，其中的個別差異來自本頁自訂的分數分布（推定）。B.3 的權重掃描：$\lambda\in\{0.005,0.01,0.02,0.03\}$ 的 FID 為 12.30、12.27、12.07、12.33（12 區塊 adaLN 模型，Table 11），作者結論對 $\lambda$ 不敏感。
- 頁面 C2 用四個合成樣本示範批次層級的 $f_h,f_s$、損失、梯度符號與 $\tau$ 的校準。

#### 步驟 6：門檻校準與訓練設定（Sec. 4.1、App. A.3）

- 訓練前在 1,024 個固定的帶噪訓練樣本上，由外層到內層依序校準各階段 $\tau$，使總壓縮比符合目標；之後在訓練與一般取樣時固定。
- 超參數：$T_r=0.05$、$T_b=0.1$、$\lambda=0.01$；AdamW $\beta=(0.9,0.95)$、weight decay 0、global batch 1,024。ImageNet 消融：HiT-B/16、80 epoch、EMA 0.9999、CFG 2.9（區間 $t\in(0.1,1.0)$）、50 步 Heun＋最後一步 Euler、FID-50k。
- GFLOPs 以每階段的**目標**群組數計算的名目值；實際成本依每張影像與時間的群組數而變（Sec. 4.1）。

#### 步驟 7：條件化與位置編碼（App. A.4、B.1、Table 6、Table 9）

- 去掉 adaLN，改用 in-context 條件 token（8 個 class token、1 個 time token，從第一個區塊加入）。作者假設在壓縮前、中、後分別做特徵調變會干擾精簡表示與全解析度跳接之間的互動。
- Table 6：12 區塊 2/2/4/2/2 改 token 條件，參數 135.0M → 91.3M，FID 12.27 → 11.55；加深到 18 區塊 3/3/6/3/3（133.9M）FID 6.61。同樣的重新分配在 JiT-B 反而使 FID 10.37 → 12.85。
- 全解析度保留影像 RoPE；條件 token 用恆等旋轉；精簡階段不加 RoPE（預設）。B.1：hard centroid RoPE 使 12 區塊模型 12.27 → 11.35；learned RoPE＋per-group MLP 使 18 區塊主模型 6.61 → 6.25；各變體差距小，為了簡單預設不用。

#### 步驟 8：壓縮階層與區塊分配的消融（Sec. 4.5、Table 4、5、7、12）

| 設定（HiT-B/16，18 區塊） | token 數 | GFLOPs | FID |
| --- | --- | --- | --- |
| 單階段 4×（3/12/3） | 256→64 | 19.1 | 9.59 |
| 2×4 | 256→128→32 | 20.8 | 6.89 |
| 4×2 | 256→64→32 | 17.8 | 9.54 |
| 4×4 | 256→64→16 | 17.1 | 10.78 |
| 2×2（預設） | 256→128→64 | 22.2 | 6.61 |
| layout 2/2/10/2/2、2/3/8/3/2、3/2/8/2/3 | 2×2 | 18.3、19.3、21.3 | 8.48、7.60、6.81 |
| 固定空間分組（第 1 階段水平配對、第 2 階段垂直配對；保留學到的 $q,k$ 投影與依輸入的軟池化／解組，只去掉合併樹與壓縮損失） | 256→128→64 | 22.1 | 7.11 |
| 三階段 2×2×2、2×2×4（3/3/2/2/2/3/3）；2×2×2（2/2/2/6/2/2/2） | — | 21.9、21.6；17.1 | 6.66、6.95；8.82 |

- 作者結論：適度、漸進的壓縮較好；2×4 與 4×2 的核心目標都是 32 token，但較晚壓縮（2×4）遠好於較早壓縮（4×2），表示保留較多 token 做中間處理有益；把區塊從核心移到外層改善 FID、但增加 FLOPs；學到的分組比固定空間分組好（7.11 → 6.61，參數相同、GFLOPs 22.1 對 22.2）。
- C3 讓讀者對這些設定即時計算推定的 GFLOPs，並對照論文的 FID。

#### 步驟 9：規模比較與主結果（Sec. 4.2、Table 1、Table 8、Fig. 1）

- Table 8（80 epoch、CFG 2.9）：JiT-B 25.2 GFLOPs／10.37 對 HiT-B 22.2／6.61；JiT-L 87.9／4.85 對 HiT-L（3/3/24/3/3）55.6／4.16；JiT-H 181.9／4.39 對 HiT-XL（10/10/8/10/10）176.9／3.64。
- Table 1：HiT-XL/16（959.5M、176.9 GFLOPs）280 epoch FID 1.87、600 epoch 1.48；JiT-H/16 600 epoch 1.86；使用 DINOv2 外部監督的 PixNerd-XL/16 2.15、PixelGen-XL/16 1.83、PixelDiT-XL/16 1.61；潛空間 REPA SiT-XL/2 1.42。
- Fig. 1d：FID ≈ 2.16 時 HiT 少用 2.4 倍訓練計算、2.3 倍更新次數（HiT 引導強度 2.6、JiT 2.4）。摘要寫「約 40% 的累積訓練 FLOPs」。

#### 步驟 10：群組引導與取樣時間偏移（App. A.6、Eq. 14–16）

$$\hat v_t=v_\varnothing^{(r_u(t))}(x_t,t)+s\Big(v_c^{(r_c(t))}(x_t,t)-v_\varnothing^{(r_u(t))}(x_t,t)\Big),\qquad r_c(t)\ge1\ge r_u(t)\tag{Eq. 14}$$

- $r=(r_1,r_2)$ 是各階段相對於預設平均群組數的倍率，透過調整門檻達成；群組數對每個輸入仍是自適應的。$r_c=r_u=(1,1)$ 時就是標準 CFG。只改門檻，不改權重、池化與讀回，也不增加 CFG 以外的網路評估。
- 動機：增加群組數給出更細的表示，但骨幹是在壓縮表示上訓練的，直接換操作區間可能變差；作者受 Sparse Guidance [22] 啟發，把「容量差」當成引導訊號。
- ImageNet（FID 1.48）：$s=2.6$，引導區間 $[0.1,1]$（區間外用條件預測），$r_c=(1.2,1)$，
  $$r_u(t)=\begin{cases}(1,1),&t<0.85\\(0.5,1),&t\ge0.85\end{cases}\tag{Eq. 15}$$
  時間網格偏移
  $$\sigma=1-t,\quad\sigma'=\frac{k\sigma}{1+(k-1)\sigma},\quad t'=1-\sigma',\quad k=1.7\tag{Eq. 16}$$
  JiT 用 $[0,1]$ 上均勻的 50 步 Heun＋最後一步 Euler；HiT 保留解算器與步數，只把網格往高噪聲端偏移。Fig. 1 的收斂曲線只有 600 epoch 端點用 $k=1.7$。
- ImageNet 的引導門檻依十個等寬時間區間各自校準（64 條取樣軌跡）；文字到影像用 128 組固定圖文對、六個噪聲等級，校準跨時間共用的門檻；只需 forward。
- 文字到影像：$r_c=(1.2,1.2)$、$r_u=(0.8,1)$、$s=6.5$，CFG 與分組調整在整個 $t\in[0,1]$ 都開啟，100 步 Euler。（App. C 的 Fig. 6 樣本另寫 CFG 7.0，頁面照錄兩者，不合併。）
- 〔本頁推導〕Eq. 16 的映射單調、固定端點（$t=0\mapsto0$、$t=1\mapsto1$），$k>1$ 時 $t'<t$（內點），把更多步數放在噪聲多的一端；C4 計算 50 步網格中落在各區間的步數。

#### 步驟 11：文字到影像的共同分組（Sec. 4.3、App. A.5、Table 2）

- 設定：Table 2 的 HiT-B/16 為 512×512、patch 16，影像為 32×32 = 1024 個 patch token（App. D 寫明 32×32 網格），加上有效字幕 token；可視化與 Fig. 6 用 HiT-B/16 D27（250k 預訓練＋40k 微調）。論文沒有寫出文字到影像各階段的目標 token 數，頁面不推算。

- 沿用 MiniT2I [38]：凍結的 FLAN-T5-Large（序列長 256）＋學到的投影＋兩個純文字前置區塊；外層保留影像與文字各自的投影與 MLP、以聯合注意力互動。
- 外層編碼器之後，影像特徵與**有效**字幕特徵串接後一起分組：$N_b=N_{\text{img}}+\sum_jm_{bj}$（Eq. 13，$m_{bj}$ 為字幕有效遮罩，含有效特殊 token）。padding 不參與樹、池化與預算；time token 繞過分組；字幕 dropout 的樣本只有影像進入分組。群組可以只含影像、只含文字或兩者皆有，所以精簡階段用共享區塊。
- Table 2（512×512）：HiT-B/16 0.25B、139.4 GFLOPs：GenEval 87.78、DPG 85.33、PRISM 58.62、CoRe 45.22、Spatial 52.68、平均 65.93；MiniT2I-B/16 0.26B、201.6 GFLOPs、平均 64.64；UniLiP-3B 1.60B、408.0 GFLOPs、平均 65.76。
- App. D.5 的可視化：字幕 token 的影像池化圖用 $(P_1P_2)_{\text{img},g}$；影像 patch 的字幕連結用 $((A_1A_2)(P_1P_2)^\top)[i,:]$ 的字幕部分；以兩階段共同最大值正規化後取平方根色階。

#### 步驟 12：常見錯誤（頁面 S11 逐條列出）

1. 只在硬群組內池化：Table 3 顯示 FID 10.16–10.32，遠差於對所有 token 加權的 6.61。
2. 解組用硬查表：7.97 對 6.61；解組要重用分組前的 $A$ 做軟混合。
3. 每張影像固定群組數：壓縮損失只約束批次的平均保留比例，個別影像的群組數可以不同（Sec. 3.3）。
4. 只在單一裝置上算比例：$f_h,f_s$ 的分子分母要先跨裝置加總再算損失；梯度累積時對各次 forward 的損失取平均（App. A.3）。
5. 讓梯度經過 $f_h$ 或樹的拓樸：$f_h$ 與硬分割視為常數，只有 $f_s$ 與軟權重可微。
6. 第二階段的分母用 $N/2$：要用第一階段實際產生的群組數（App. A.2）。
7. 條件 token 或 padding 參與分組：class／time token 與 padding 繞過分組與預算（App. A.4–A.5）。
8. 把 GFLOPs 當成實際成本：論文報告的是以目標群組數算的名目值。
9. 門檻在訓練中更新：基準門檻訓練前校準後固定；群組引導另用個別校準的推論門檻。
10. $T_r=0.05$ 很小：$A$ 接近硬指派，只有平均親和度接近的群組才會分到可見的權重（〔本頁推導〕：權重比 $A_{ig}/A_{ig'}=\exp((e_{ig}-e_{ig'})/T_r)$，親和度差 0.1 即約 $e^2\approx7.4$ 倍）。

#### 步驟 13：限制

論文沒有獨立的限制章節。頁面只寫論文明確提到的事實：實際成本隨群組數變化（Sec. 4.1）；在推論時直接增加群組數不一定改善生成（App. A.6）；精簡階段的位置編碼各變體差距小、尚未定案（App. B.1）。不自行補充其他限制。

---

## 2. 學習目標（驗收測試）

每章結尾以 `<details>`（summary「檢查理解」）列出該章檢查題與參考答案；參考答案的內容必須出現在該章內文，不能只出現在元件、表格或答案框。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說出作者的假設 | 作者認為像素空間 Transformer 落後潛空間模型的部分原因是什麼？ | 骨幹從頭到尾每個 patch 保留一個 token，沒有鼓勵把相關內容在逐漸變粗的層級彙整的視覺階層（Abstract、Sec. 1） | S1 |
| LO2 | 描述階層骨幹 | ImageNet 256 的 HiT-B/16（18 區塊）token 數如何變化？layout 3/3/6/3/3 的五個數字代表什麼？ | 目標平均 256→128→64→128→256；依序是全解析度編碼器、中間編碼器、精簡核心、中間解碼器、全解析度解碼器的區塊數（Eq. 1、Sec. 4.5、Table 4） | S2 |
| LO3 | 計算親和度與硬分組 | $S_{ij}$ 怎麼算？平均連結法保留哪些合併？ | $S_{ij}=\tfrac12(q_i^\top k_j+q_j^\top k_i)$，$q,k$ 為單位向量；反覆合併平均親和度最高的兩群，保留分數 $\ge\tau$ 的合併（Eq. 2、8、Sec. 3.2）。〔本頁推導，假設精確的平均連結法〕合併分數不增，所以等於最高分低於 $\tau$ 時停止 | S3 |
| LO4 | 解釋為何能非局部分組 | 為什麼空間上不相連的 patch 能在同一群組？ | 親和度只由特徵投影 $q,k$ 計算，不含位置；HiT 在雙向影像特徵上分組，不像 H-Net 沿因果序列切塊（Sec. 2、3.2） | S3 |
| LO5 | 寫出軟權重與池化 | $A_{ig}$ 與 $P_{ig}$ 怎麼定義？群組 token 由哪些 token 組成？ | $A_{ig}=\mathrm{softmax}_g(e_{ig}/T_r)$，$e_{ig}$ 是 token $i$ 對 $G_g$ 成員的平均親和度；$P_{ig}=A_{ig}/\sum_jA_{jg}$；群組 token 是所有輸入 token 的加權平均（Eq. 6、9、App. A.1） | S4 |
| LO6 | 說明解組與梯度路徑 | 解組怎麼做？梯度如何到達親和度投影？ | $r_i=\sum_gA_{ig}\tilde z_g$，即 $\tilde H=AM(Z)+HW_\ell^\top$：重用 $A$ 軟混合並加上初始化為單位矩陣的跳接；硬分割視為常數，梯度經由軟池化與解組權重 $A$ 傳到 $q,k$（Eq. 3、7、Sec. 3.2） | S5 |
| LO7 | 讀出軟交換的消融 | Table 3 中限制在硬群組內池化、或用硬查表解組，FID 各是多少？ | 硬群組內池化 10.32（均勻）／10.16（學到權重）；硬查表解組 7.97；預設 6.61（Table 3） | S5 |
| LO8 | 寫出壓縮損失並判斷梯度方向 | Eq. 4 是什麼？$f_h>\rho$ 時梯度下降如何改變 $f_s$？ | $1+(f_h-\rho)(f_s-\rho)/(\rho(1-\rho))$；對 $f_s$ 的梯度為 $(f_h-\rho)/(\rho(1-\rho))>0$，梯度下降降低 $f_s$（使合併分數上升，更多合併）；$f_h$ 視為常數（Eq. 4、Sec. 3.3） | S6 |
| LO9 | 說明批次層級預算 | $f_h$ 與 $f_s$ 如何計算？為什麼每張影像的群組數可以不同？ | $f_h=\sum_bK_b/\sum_bN_b$，$f_s=\sum_b\tilde K_b/\sum_bN_b$，$\tilde K_b=N_b-\sum_m\sigma((s_{b,m}-\tau)/T_b)$，跨批次與裝置加總；損失只約束平均比例（Eq. 11、Sec. 3.3） | S6 |
| LO10 | 讀出壓縮階層消融 | 2×4 與 4×2 的核心目標都是 32 token，FID 各是多少？作者的結論？ | 2×4 為 6.89、4×2 為 9.54；預設 2×2 為 6.61；適度、漸進的壓縮較好，保留較多 token 做中間處理有益（Table 4） | S7 |
| LO11 | 描述群組引導 | 群組引導比 CFG 多改了什麼？ImageNet 設定為何？ | 條件分支用較細的分組 $r_c\ge1$、無條件分支用較粗的 $r_u\le1$，只改門檻、不增加網路評估；ImageNet $s=2.6$（區間 $[0.1,1]$）、$r_c=(1.2,1)$、$t\ge0.85$ 時 $r_u=(0.5,1)$ 否則 $(1,1)$；時間偏移 $k=1.7$（僅 600 epoch 端點）（Eq. 14–16） | S9 |
| LO12 | 讀出主結果 | HiT 在 ImageNet 256 與文字到影像的關鍵數字？ | HiT-XL/16 600 epoch FID 1.48（280 epoch 1.87，JiT-H/16 600 epoch 1.86）；FID ≈ 2.16 時少 2.4 倍訓練計算；HiT-B/16 文字到影像平均 65.93 對 MiniT2I 64.64，GFLOPs 少 31%（Table 1、2、Fig. 1） | S10 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），互動元件放在加寬圖版（最大 1080px）。章節編號以 JetBrains Mono 顯示（`01 · …`）。視覺記號：一張「token 數變化圖」（256→128→64→128→256 的梯形帶，寬度＝token 數），在 Hero、S2、C3 重複出現。章節標題一律用陳述句。

| # | id | 標題 | 內容 | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈HiT：在像素空間擴散 Transformer 裡學出視覺階層〉 | 副標：「HiT 把相關的 patch token 合併成群組 token，讓中間的 Transformer 處理較短的序列，群組數由一個批次層級的預算決定。」三張事實卡；token 數變化圖；連結 arXiv、`../index.html#paper-585`、`hit-lineage-hub.html`、`token-compression-lineage.html` | 1 螢幕 |
| S1 | `why` | 01 · 像素空間模型每個 patch 都保留一個 token | 潛空間 vs 像素空間（2–3 句）；作者假設（LO1）；規則網格的分配問題。可收合「背景：x-prediction 與訓練損失」：$x_t=tx+(1-t)\epsilon$、$t=1$ 乾淨／$t=0$ 純噪聲、網路輸出乾淨影像、Eq. 12（說明分母 $\max(1-t,0.05)^2$ 把 x 誤差換算成速度誤差並避免 $t\to1$ 時除以 0）。末尾符號表。延伸：`hit-latent-hierarchy-and-pixel-space.html`、`pixel-space-diffusion.html`、`x-prediction-manifold-view.html` | 300–400 字 + 收合區 |
| S2 | `backbone` | 02 · 骨幹由編碼器、精簡核心與解碼器遞迴組成 | Eq. 1；Fig. 2 的 SVG 重繪（區塊寬度＝token 數；只標「分組」「解組」「跳接」，圖說各一句定義）；layout 記法（ImageNet D18）；條件 token 繞過分組；〔本頁推導〕2688 對 4608。延伸：`tokc-hourglass-and-latent-interfaces.html` | 300–400 字 + 靜態 SVG |
| S3 | `affinity` | 03 · 親和度與平均連結法決定硬群組 | 先講單位向量與餘弦；Eq. 2 與對稱化理由；三個 token 的計算例（步驟 3）；Eq. 8、合併樹與樹狀圖、門檻 $\tau$；非局部分組（LO3、LO4）。可收合：合併分數不增〔本頁推導〕；與 ToMe 二分配對、H-Net 因果切塊的對照（各一句，連 `hit-merging-baselines-fixed-rates.html`、`hit-hnet-blt-to-image-groups.html`；共同背景連 `tokc-vit-pruning-and-merging.html`、`tokc-dynamic-chunking-bytes-to-pixels.html`）；延伸：`hit-affinity-grouping.html` | 350–450 字 + 2 收合區 |
| S4 | `soft` | 04 · 每個群組 token 由所有輸入 token 加權平均而成 | 為什麼需要軟權重（硬指派不可微）；softmax 溫度 $T_r$；Eq. 9、$P$、Eq. 6（LO5）；C1 在此 | 300–400 字 + C1 |
| S5 | `ungroup` | 05 · 解組重用同一組軟權重並加上跳接 | Eq. 7 與等價的矩陣式 Eq. 3（只寫這一次）；$W_\ell$ 與 Table 10 的但書；梯度路徑；條件 token 繞過；Table 3（LO6、LO7） | 350–450 字 + 表 |
| S6 | `budget` | 06 · 壓縮損失只約束批次平均的保留比例 | sigmoid 定義；Eq. 11、Eq. 4（Eq. 10 收合在旁）、Eq. 5；梯度符號〔本頁推導〕；跨裝置加總；門檻校準（App. A.3）；Table 11；C2 在此（LO8、LO9）。延伸：`hit-compression-budget.html` | 450–550 字 + C2 |
| S7 | `ablate` | 07 · 適度而漸進的壓縮表現最好 | 步驟 8 表；名目 GFLOPs；C3；Table 6–7 結論；B.1 位置編碼一段（連 `tokc-position-for-merged-tokens.html`）（LO10） | 400–500 字 + C3 |
| S8 | `groups` | 08 · 學到的群組對應物件、部位與字詞 | Fig. 1、4、7–10 的文字描述（不重繪，R3）；跨時間（Fig. 9）；跨階段（Fig. 8）；文字到影像共同分組（Eq. 13、步驟 11 設定）；App. D.5 兩個可視化公式。延伸：`hit-compact-position-and-joint-grouping.html` | 350–450 字 |
| S9 | `guidance` | 09 · 群組引導以分組容量的差作為引導訊號 | Eq. 14、15、16；動機；校準；T2I 設定；C4（LO11）。延伸：`hit-group-guidance.html`、`tokc-learned-routing-dynamic-dit.html`（含 Sparse Guidance）、`diffusion-classifier-guidance-background.html` | 400–500 字 + C4 |
| S10 | `results` | 10 · HiT-XL/16 在 ImageNet 256 達到 FID 1.48 | Table 1 摘錄、Table 8、Table 2 摘錄、Fig. 1d 的兩個倍數（LO12）。延伸：`generative-eval-metrics.html` | 300–400 字 + 3 表 |
| S11 | `impl` | 11 · 實作設定與常見錯誤 | 步驟 6 超參數表、步驟 12 的 10 條 | 300–400 字 + 表 |
| S12 | `compare` | 12 · 與 RTI／HiT 的異同 | 對照表（下）；連 `rti-elastic-token-compression.html`、`token-compression-lineage.html`、`tokc-content-adaptive-tokenization.html` | 200–300 字 + 表 |
| S13 | `history` | 13 · 研究脈絡 | 時間線（6.7）＋一段：潛空間階層 → 像素空間 → token 合併（固定比例）→ GroupViT → BLT／H-Net → Sparse Guidance → HiT（與 6.7 的 `TL_GROUP` 一致） | 時間線 + 100–150 字 |
| S14 | `next` | 14 · 限制與延伸閱讀 | 步驟 13；筆記清單（依 `hit-lineage-hub` 閱讀順序）；arXiv、GitHub、Hugging Face、專案頁 | 200 字 + 清單 |

Hero 事實卡（不使用未定義術語）：
1. 「HiT 把特徵相近的 patch token 合併成一個群組 token；相距很遠的 patch 也能同組。」
2. 「群組數不固定：訓練只要求每一個分組階段在整個批次上平均保留一半的 token（兩階段後為四分之一），每張圖、每個噪聲程度可以不同。」
3. 「在 ImageNet 256 的影像品質指標 FID（越低越好）上達到 1.48；達到同樣品質時，比沒有分組的像素空間基準模型 JiT 少用 2.4 倍訓練計算（Fig. 1，FID ≈ 2.16）。」Hero 卡片內第一次出現 FID 與 JiT 時即以這種方式定義。

S12 對照表（RTI 事實取自 `tutorial_src/217/src/paper.txt` 摘要與 Sec. 3；建置前與 `rti-elastic-token-compression` 設計文件核對）：

| 面向 | HiT（本文） | RTI（No. 217） |
| --- | --- | --- |
| 訓練方式 | 從頭端到端訓練，分組與去噪一起學 | 改造預訓練的像素空間文字到影像模型，微調介面 |
| 群組形狀 | 由特徵親和度決定，可以不相連 | 沿 Hilbert 曲線切成連續區段，區域必相連 |
| 群組數 | 批次層級壓縮損失決定平均，個別樣本自適應 | 區域數 $R$ 在微調時隨機抽取，一個 checkpoint 服務多種預算 |
| 位置 | 精簡階段預設不加 RoPE（B.1 各變體差距小） | 每個區域 token 帶位置（patch rotary embedding 的平均） |
| 解組 | 重用軟權重 $A$ 混合回所有位置 | 把區域 token 的變化寫回其 patch |
| 主要目的 | 生成品質與訓練效率 | 推論加速（摘要：與 dense 品質相當時 2.0 倍速度） |

頁面只寫這張表的事實並各附出處；細節連兄弟教學。桌機（≥ 1024px）左側 sticky 章節目錄；< 1024px 頂部橫向章節列（同參考頁）。

---

## 4. 互動元件規格

### 4.0 共用

- sigmoid 分支式：$x\ge0$ 時 $1/(1+e^{-x})$，否則 $e^x/(1+e^x)$。softmax 先減最大值。
- 頁面不使用 PRNG：C2 的資料直接內嵌（R9）。
- 數值讀數 JetBrains Mono，以 `textContent` 更新；圖版中的 TeX 標籤（例如 $A_{ig}$、$P$）在載入時由 KaTeX 渲染一次，拖曳與滑桿輸入**不**呼叫 `renderMathInElement`。
- live region（`aria-live="polite"`）每個元件一個，只放一行摘要，在 `change`／`pointerup` 或停止輸入 500ms 後更新。
- 發散色階（C1 熱圖）：$S\in[-1,1]$；$S>0$ 用 `--c-par`、$S<0$ 用 `--c-perp`，不透明度 $0.12+0.73|S|$；格內數字預設顯示（兩位小數，負號用 U+2212），字色在不透明度 > 0.5 時用 `--surface`，否則 `--ink`。兩個 token 已有深色值，深色模式自動成立；建置時確認數字對底色 ≥ 4.5:1。

### C1 · 六個 token 的分組、軟權重與池化（S4）

- **教的概念**：Eq. 2 → 合併樹 → 門檻 $\tau$ 得硬群組 → Eq. 9 軟權重 → $P$（Eq. 6）；群組 token 讀取所有 token；$T_r$ 控制軟硬；群組不必在空間上相鄰（LO4 的視覺化）。
- **簡化**（頁面寫出）：每個 token 的 $q_i=k_i$ 是二維單位向量，以角度 $\theta_i$ 表示，所以 $S_{ij}=\cos(\theta_i-\theta_j)$、$S_{ii}=1$。後者是玩具設定的結果：論文中 $q\ne k$，$S_{ii}=q_i^\top k_i$ 一般小於 1，而 Eq. 9 的 $e_{ig}$ 對單獨成組的 token 只含這個自我項（推定的教學設定）。
- **輸入**：
  - 角度：預設 $[10,22,150,38,165,178]$（度，推定）。單位圓上 6 個端點各為 `role="slider"`（`aria-valuemin=0`、`aria-valuemax=359`、`aria-valuetext`「token 4：38 度」）：方向鍵 ±1°、Shift＋方向鍵 ±10°，超出範圍取 mod 360（359 + 1 → 0）。指標拖曳：`pointerdown` 時 `setPointerCapture`，端點與圓盤設 `touch-action:none`，角度 $=\mathrm{round}(\mathrm{atan2})\bmod360$（整數）。另附 6 個數字輸入（0–359）作為替代。
  - $\tau\in[-1,1]$，步長 0.005，預設 0.95。
  - $T_r\in[0.02,1]$，對數滑桿，預設 0.2（推定，R1）；chip「論文值 0.05」。
- **計算**（每次輸入即時，$N=6$）：
  1. $S$（Eq. 2）。
  2. 完整平均連結樹（Eq. 8）：每次在所有群組對中取 $s$ 最大者；差 ≤ 1e-12 視為同分，取先找到的對（$a<b$ 的字典序）；記錄 5 個合併分數。
  3. 硬群組：依序套用分數 $\ge\tau$ 的合併。
  4. $e_{ig}$、$A$（Eq. 9）、$P_{ig}=A_{ig}/\max(\sum_jA_{jg},10^{-300})$。
  5. 進階（預設收合，標「S6 會用到」）：$\tilde K=6-\sum_m\sigma((s_m-\tau)/0.1)$（Eq. 11，$T_b=0.1$）。
- **顯示**（四個面板）：
  1. 圓盤：6 個箭頭，編號 1–6；硬群組以填色深淺＋形狀（圓、方、三角、菱形、星、十字）雙重編碼。
  2. patch 網格（2×3）：把 6 個 token 放在固定位置——第一列 1、3、2，第二列 5、6、4——每格畫該 token 的群組形狀。圖說：「親和度不看位置。token 1 與 2 之間隔著 token 3，仍可同組。」
  3. 親和度熱圖（6×6，4.0 的發散色階＋數字），硬群組內的格子加外框；合併樹（樹狀圖）放在熱圖下方，縱軸為合併分數，$\tau$ 為水平虛線，高於虛線的合併畫實線、低於畫淡虛線。
  4. $A$ 矩陣（列＝token、行＝群組），每格長條＋兩位小數；點選群組行標題，下方列出該群組的 $P$ 行，並寫「群組 g 的 token = Σᵢ P_ig h_i」。
  - 版面：≥ 900px 為 2×2；600–899px 兩欄（圓盤＋網格一欄，熱圖／樹＋$A$ 一欄）；< 600px 單欄，順序 圓盤（最小 280px 見方）→ 網格（最小 240×160）→ $A$ → 熱圖（最小 280px 見方，格內數字 12px）→ 樹。
  - 讀數：$K$、5 個合併分數；進階區打開時加 $\tilde K$。
  - live region 摘要例：「τ = 0.950，4 個群組：{1,2}、{3}、{4}、{5,6}」。
- **預期觀察**：
  1. 預設 $K=4$，群組 {1,2}、{3}、{4}、{5,6}；token 4 的軟權重 0.404／0.001／0.596／0.000：它單獨成組，仍有 40% 權重流向 {1,2}。網格中 {1,2} 不相鄰。
  2. $\tau$ 降到 0.9：$K=2$，{1,2,4}、{3,5,6}（與 Fig. 3 相同的分法；網格中 1、2、4 不連成一片）。
  3. $T_r$ 調小，$A$ 趨近 0/1；調大，權重攤平。
  4. token 4 轉到 90°：它對 {1,2} 的權重降到 0.024。
- **數值防護**：$\tau$ 低於所有分數時 $K=1$、$A$ 全為 1；上限 1 時（合併分數皆 < 1）$K=6$。
- **無障礙**：每個面板 `role="img"`＋動態 `aria-label`；群組以形狀＋文字編碼，不只靠顏色；熱圖數字預設顯示。

### C2 · 批次預算：$f_h$、$f_s$、損失與門檻校準（S6）

- **教的概念**：Eq. 4／11；批次層級加總；梯度符號；樣本間群組數可以不同；門檻校準的概念（App. A.3）。
- **資料**（合成，推定）：4 個樣本，各 $N=256$ 個 token、255 個合併分數，以 `verify/c2_scores.json`（4×255 個四位小數，約 7 KB）原樣內嵌為 `C2_SC`。生成方式寫在 `check.py`：$\mathrm{clip}(c_b+0.18\xi,-1,1)$，$c_b=[0.92,0.80,0.68,0.50]$，四捨五入到四位小數，並避開 $\tau$ 網格（若為 0.005 的整數倍就加減 0.0001），因此任何網格 $\tau$ 都不會與分數相等。頁面標題寫「合成的 4 個樣本，不是論文數據；樣本間群組數的差距來自本頁自訂的分數分布」。
- **輸入**：$\tau$ 滑桿以整數 $i\in[-40,200]$ 表示，$\tau=i/200$（步長 0.005，預設 $i=160$，$\tau=0.80$）；$\rho$ 分段按鈕 1/2（預設）、1/4；$T_b\in[0.02,0.5]$，預設 0.1；按鈕「校準 τ」。
- **校準規則**（本頁自訂，推定；論文是在 1,024 個帶噪樣本上由外層到內層依序校準各階段，App. A.3）：由小到大掃描 $i$，取第一個使 $f_h\ge\rho$ 的 $i$，滑桿跳到該值。$f_h$ 對 $\tau$ 不減，所以這是滿足預算的最小門檻。
- **計算**：$K_b=256-\#\{s\ge\tau\}$；$\tilde K_b$（Eq. 11）；$f_h,f_s$；$\mathcal L$（Eq. 4）；$\partial\mathcal L/\partial f_s=(f_h-\rho)/(\rho(1-\rho))$。Eq. 10 放在 Eq. 4 旁的收合區，展開時顯示以 Eq. 10 算出的同一個值。
- **顯示**：
  - 四列長條：$K_b$（實心）與 $\tilde K_b$（空心外框）占 256 的比例，附數字。
  - 總比例軸：$f_h$（實心三角）、$f_s$（空心三角）、$\rho$（垂直虛線）。
  - 損失讀數與方向文字：梯度 > 0「降低 f_s（使合併分數上升）」；< 0「提高 f_s（使合併分數下降）」；$|f_h-\rho|<10^{-3}$「已達預算」。箭頭方向＋文字，不只靠顏色。
  - 直方圖：四個小圖（small multiples）各一列，標「樣本 1–4」，共用 x 軸 $[-1,1]$、40 個 bin，每列畫 $\tau$ 垂直線。
- **預期觀察**：
  1. $\tau=0.80$、$\rho=1/2$：$K_b=[63,112,193,241]$，$f_h=0.5947>\rho$，梯度 +0.3789。
  2. 按「校準 τ」：$\tau=0.745$，$K_b=[36,85,165,230]$，$f_h=0.5039$（離散計數，無法恰好 0.5）；前一格 $\tau=0.740$ 的 $f_h=0.4961<\rho$。平均達標，但各樣本保留比例不同（本頁合成分布造成）。
  3. Eq. 4 與 Eq. 10 的值永遠相同（〔本頁推導〕）。
- **數值防護**：$\rho\in\{1/2,1/4\}$；sigmoid 分支式；比較 $s\ge\tau$ 無同值（見資料）。
- **無障礙**：長條附數字；滑桿 `aria-valuetext`「τ = 0.800」；live region 摘要「τ = 0.745，f_h = 0.504，已達預算」。

### C3 · layout 與壓縮比的推定成本（S7）

- **教的概念**：同樣 18 個區塊，區塊放在哪一層、每層壓縮多少，決定 FLOPs；旁邊放論文 FID，看到「便宜」與「好」不一致（Table 4、5、12）。範圍：ImageNet 256、HiT-B/16、寬 768、in-context 條件。
- **推定成本模型**：每個區塊在 $m=n+9$ 個 token 上（$n$ 影像 token，9 條件 token）成本 $\alpha m+\beta m^2$，總成本 $\approx\alpha\sum m+\beta\sum m^2+\gamma$。
  - $\beta=2d/10^9=1.536\times10^{-6}$（注意力的 $QK^\top$ 與 $AV$ 各 $m^2d$ 次乘加，〔本頁推導〕）。
  - $\alpha=7.2852\times10^{-3}$、$\gamma=0.6229$：最小平方法擬合 Table 4、5、12 的 11 個設定（`check.py fit()`）。最大**擬合殘差** 0.25 GFLOPs；這是三參數模型對自己擬合資料的殘差，不是獨立預測或驗證。理論對照：$4d^2+3dh$、$h=2048$（推定，論文未給 SwiGLU 寬度）時 $\alpha=7.08\times10^{-3}$。
  - router、樹與池化成本未建模。
- **資料結構**：`{J, ratios:[r_1..r_J], layout:[b_0..b_{2J}]}`；第 $i$ 段的層級 $\ell(i)=i$（$i\le J$）或 $2J-i$（$i>J$），token 數 $n_{\ell}=256/\prod_{j\le\ell}r_j$（實數，不取整）。
- **載入時狀態**：預設 chip「2×2（3/3/6/3/3）」選取，即 `{J:2, ratios:[2,2], layout:[3,3,6,3,3]}`，推定 22.26、論文 22.2／FID 6.61（Table 4）。
- **輸入**：
  - 預設 chip：Table 4 五項、Table 5 三項、Table 12 三項、「不分組 12 區塊」（標籤「不分組 12 區塊（對照，非論文設定）」，`{J:0, ratios:[], layout:[12]}`）。Table 12 的 2×2×4 chip 為 `{J:3, ratios:[2,2,4], layout:[3,3,2,2,2,3,3]}`。
  - 自訂：階段數 $J$ 為 1／2／3，切換時載入預設 $J=1$：`ratios [4]`、`layout [3,12,3]`；$J=2$：`[2,2]`、`[3,3,6,3,3]`；$J=3$：`[2,2,2]`、`[3,3,2,2,2,3,3]`。每層壓縮比 2 或 4；每段區塊數步進器 1–24，左右對稱段同步改變；總區塊數顯示但不強制 18。
- **顯示**：
  - token 數變化圖：每段一個矩形，寬度 ∝ token 數、高度 ∝ 區塊數，標「3 × 256」等；跳接以弧線表示。
  - 讀數：推定 GFLOPs（一位小數）、$\sum m$、$\sum m^2$；設定與論文某列相同時顯示論文 GFLOPs、FID 與表號，否則「論文未測試此設定」。
  - 散點小圖：11 個論文設定（x＝推定 GFLOPs，y＝論文 FID），標記形狀區分 Table 4／5／12，目前設定為實心。
- **預期觀察**：
  1. 2×2 推定 22.26（論文 22.2）；4×2 推定 17.90（論文 17.8），較便宜但 FID 9.54 對 6.61。
  2. 2/2/10/2/2 推定 18.28，FID 8.48。
  3. 「不分組 12 區塊」推定 25.08；JiT-B 的 GFLOPs 是 25.2（Table 8），但 JiT 的條件化與輸入處理不同（App. A.4、Table 6），這個接近是巧合，不作為驗證。
- **數值防護**：區塊數下限 1；成本只顯示一位小數。
- **無障礙**：步進器可鍵盤操作；散點標記形狀＋文字；變化圖 `aria-label` 列出各段。

### C4 · 時間網格偏移與群組引導排程（S9）

- **教的概念**：Eq. 16 把 50 步網格移向噪聲多的一端；Eq. 15 的 $r_u(t)$ 在接近乾淨影像的一端（$t\ge0.85$）把第 1 階段的群組數倍率降為 0.5；引導區間 $[0.1,1]$；Eq. 14 只需兩次網路評估。
- **輸入**：$k\in[1,3]$，步長 0.05，預設 1.7；chip「HiT 600 epoch 端點（k = 1.7）」「均勻網格（k = 1，JiT 與 HiT 收斂曲線其他點）」。步數固定 50。
- **計算**：$t'_i=1-k\sigma_i/(1+(k-1)\sigma_i)$，$\sigma_i=1-i/50$，$i=0..50$。比較規則：每步起點 $t'_i$（$i=0..49$）先四捨五入到 $10^{-9}$（`Math.round(t*1e9)/1e9`），再用閉區間比較；避免 $k=1$、$i=5$ 時浮點 $0.09999999999999998$ 落在 $0.1$ 之外。計數：$t'<0.5$；$0.1\le t'\le1$（使用引導，$s=2.6$）；$t'\ge0.85$（$r_u=(0.5,1)$）。
- **顯示**：
  - 時間軸：左端標「t = 0 純噪聲」，右端標「t = 1 乾淨影像」；51 個刻度（均勻網格淡色在下、目前網格在上）；$t=0.1$、$0.5$、$0.85$ 參考線。
  - 倍率階梯圖（與時間軸共用 x）：四條線——$r_{c,1}=1.2$（實線）、$r_{u,1}$（虛線，$t<0.85$ 為 1、之後 0.5）、$r_{c,2}=r_{u,2}=1$（點線，標「第 2 階段兩分支」）；$t<0.1$ 的區段灰底標「只用條件預測，不引導」。y 軸 0.4–1.3，標「相對預設群組數的倍率」。
  - 映射曲線 $t\mapsto t'$ 與對角線（小圖）。
  - 讀數：三個步數、$t'_1$、$t'_{25}$、$t'_{49}$（四位小數）；一行寫出 Eq. 14 的兩次網路評估（與 CFG 相同次數）。
- **預期觀察**：$k=1.7$ 時 $t'_{25}=0.3704$（均勻 0.5），32 步從 $t<0.5$ 出發（均勻 25）；$t\ge0.85$ 的步數由 7 降為 4。
- **數值防護**：$k\ge1$ 時分母 $\ge1$；端點固定。
- **無障礙**：線型＋文字標籤；live region 摘要「k = 1.70：32 步在 t < 0.5，4 步使用 r_u = (0.5, 1)」。

不做的元件：重繪論文群組圖（R3）；FID 曲線（Fig. 1d 無數據）。

---

## 5. 驗收數值表（`verify/check.py` 輸出）

容許誤差原則：整數（群組數、$K_b$、步數）完全一致；其他值依量級給絕對或相對誤差，如各表所註。

### C1（角度 [10,22,150,38,165,178]，$T_b=0.1$）

| 條件 | $K$ | 群組 | $\tilde K$（進階） | 其他 |
| --- | --- | --- | --- | --- |
| 預設 τ=0.95、Tr=0.2 | 4 | {1,2},{3},{4},{5,6} | 4.002 | token 4 的 A = 0.404 / 0.001 / 0.596 / 0.000；群組 {1,2} 的 P 行 = 0.402, 0.341, 0, 0.257, 0, 0 |
| τ=0.9 | 2 | {1,2,4},{3,5,6} | 3.520 | — |
| τ=0.8，Tr=0.2 或 0.05 | 2 | {1,2,4},{3,5,6} | 2.744 | A 為 0/1（至三位小數） |
| τ=0.975 / 0.98 | 5 / 6 | — | 4.247 / 4.295 | — |
| τ=−1 | 1 | 全部 | 1.074 | P 全為 0.167 |
| 預設，token 4 = 60° / 90° / 120° | 4 | 同預設 | 4.346 / 4.432 / 4.356 | token 4 的 A[{1,2}] = 0.193 / 0.024 / 0.001 |

合併分數（預設）：0.9781、0.9744、0.9244、0.9221、−0.7468。S3 的三 token 例：$S_{12}=0.9781$、$S_{13}=-0.7660$、$S_{23}=-0.6157$、$s(\{1,2\},\{3\})=-0.6909$。容許：$S$、合併分數 ±1e-4；$\tilde K$ ±0.001；$A,P$ 的值 ≥ 0.01 時相對 ±0.5%，< 0.01 時絕對 ±0.0005。相對容許以 `check.py` 末段印出的五位有效數字參考值為準（例：token 4 的 A = 0.40354 / 0.00061676 / 0.59571 / 0.00013135；token 4 = 90° 時 A[{1,2}] = 0.023634），不以表中三位小數為準。

### C2（內嵌 `c2_scores.json`）

| ρ | τ | $K_b$ | $\tilde K_b$ | $f_h$ | $f_s$ | 𝓛 | ∂𝓛/∂f_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1/2 | 0.800（預設） | 63,112,193,241 | 84.6,119.7,176.3,222.8 | 0.5947 | 0.5894 | 1.0339 | +0.3789 |
| 1/2 | 0.600 | 11,22,80,173 | 26.1,46.2,95.9,161.4 | 0.2793 | 0.3219 | 1.1572 | −0.8828 |
| 1/2 | 0.900 | 123,166,227,253 | 130.7,163.9,208.9,239.4 | 0.7510 | 0.7255 | 1.2264 | +1.0039 |
| 1/2 | 0.740 | 35,81,163,229 | 61.9,94.2,153.0,208.3 | 0.4961 | 0.5053 | 0.9999 | −0.0156 |
| 1/2 | **0.745（校準結果）** | 36,85,165,230 | 63.6,96.3,155.0,209.6 | 0.5039 | 0.5122 | 1.0002 | +0.0156 |
| 1/4 | **0.585（校準結果）** | 9,18,71,161 | 23.6,42.3,90.1,155.6 | 0.2529 | 0.3043 | 1.0008 | +0.0156 |
| 1/4 | 0.800 | 同第一列 | 同第一列 | 0.5947 | 0.5894 | 1.6239 | +1.8385 |

Eq. 4 與 Eq. 10 差 < 1e-12。容許：$\tilde K_b$ ±0.05（顯示一位小數）；$f_h,f_s,\mathcal L$ ±1e-4；梯度取 max(相對 1e-3, 絕對 1e-4)。校準 $i$ 必須完全一致（149、117）。

### C3（$\alpha=7.2852\times10^{-3}$、$\beta=1.536\times10^{-6}$、$\gamma=0.6229$）

| 設定 | 論文 GFLOPs | 推定 | 擬合殘差 |
| --- | --- | --- | --- |
| 4×（3/12/3） | 19.1 | 19.33 | +0.23 |
| 2×4 | 20.8 | 20.82 | +0.02 |
| 4×2 | 17.8 | 17.90 | +0.10 |
| 4×4 | 17.1 | 17.19 | +0.09 |
| 2×2（3/3/6/3/3） | 22.2 | 22.26 | +0.06 |
| 2/2/10/2/2 | 18.3 | 18.28 | −0.02 |
| 2/3/8/3/2 | 19.3 | 19.26 | −0.04 |
| 3/2/8/2/3 | 21.3 | 21.28 | −0.02 |
| 3 階段 2×2×2（3/3/2/2/2/3/3） | 21.9 | 21.78 | −0.12 |
| 3 階段 2×2×4 | 21.6 | 21.54 | −0.06 |
| 3 階段 2×2×2（2/2/2/6/2/2/2） | 17.1 | 16.85 | −0.25 |
| 不分組 12 區塊 × 256 | （JiT-B 25.2，僅參考） | 25.08 | — |

容許：推定值 ±0.01（約相對 5e-4；頁面內嵌係數，不重新擬合）。

### C4

| k | $t'_1$ | $t'_{25}$ | $t'_{49}$ | 起點 $t<0.5$ | 在 [0.1,1] | $t\ge0.85$ |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.0200 | 0.5000 | 0.9800 | 25 | 45 | 7 |
| 1.7 | 0.0119 | 0.3704 | 0.9665 | 32 | 42 | 4 |
| 3.0 | 0.0068 | 0.2500 | 0.9423 | 38 | 37 | 2 |

容許：步數完全一致；$t'\ge0.1$ 時 ±1e-4，$t'<0.1$ 時相對 ±0.5%（參考值 $t'_1$ = 0.0118624、0.00675676，見 `check.py` 末段）。

---

## 6. 實作計畫

### 6.1 沿用的外殼（不重寫）

從 `tutorials/prediction-targets-k-diff.html` 複製：`<head>`（Google Fonts、KaTeX 0.16.9 與 auto-render、分隔符）、CSS tokens（含 `--c-par`、`--c-perp`、深色模式三段覆寫、`pm-theme`）、appbar、`.toc`／`.tocbar`、`.plate` 系列、callout、`<details>` 檢查題、表格橫捲容器、「本頁推導」pill、`#pm-bar`（`../index.html`、`index.html`）、`renderTimeline` 與卡片列、`?test` 的 `selfTest()` 框架（含 `scrollWidth <= innerWidth`）、`el`／`scaleLin`／`linePath`。

### 6.2 新增的部分

- 檔案 `tutorial_src/585/build/index.html`；slug `hit-hierarchical-diffusion-transformer`；論文連結 `../index.html#paper-585`；arXiv `https://arxiv.org/abs/2611.22962`。
- 圖表色：`--c-par` 用於群組／精簡階段與正親和度，`--c-perp` 用於全解析度／跳接與負親和度；不新增 token。「推定」pill：虛線外框，與「本頁推導」區分。
- 靜態 SVG：Hero 與 S2 的 token 數變化圖、S2 骨幹圖。
- JS（IIFE 內）：
  - C1：`affinity(ang)`、`avgLinkage(S)`→`[{s,U,V}]`、`hardGroups(S,tau)`、`softWeights(S,G,Tr)`→`{A,P}`、`kSoft(ms,N,tau,Tb)`、`sigmoid(x)`、`GRID_POS=[[0,0],[0,2],[0,1],[1,2],[1,0],[1,1]]`、`initC1()`。
  - C2：`C2_SC`（內嵌 JSON）、`budget(i,rho,Tb)`（`tau=i/200`）、`calibrate(rho)`→`i`、`initC2()`。
  - C3：`C3_COEF={a:7.285229e-3,b:1.536e-6,g:0.6229}`、`C3_PRESETS`（11 列＋不分組，各附論文 GFLOPs、FID、表號）、`costModel({J,ratios,layout})`、`initC3()`。
  - C4：`shiftT(t,k)`、`scheduleCounts(k)`、`ruSchedule(t)`、`initC4()`。
- `selfTest()` 斷言（第 5 節數值與容許）：C1 預設 $K=4$、token 4 的 $A$、τ=0.9 的群組、三 token 例 −0.6909、合併分數不增；C2 τ=0.80 的 $K_b$、$f_h$、梯度、Eq. 4 = Eq. 10、`calibrate(0.5)===149`、`calibrate(0.25)===117`；C3 2×2 = 22.26、4×2 = 17.90；C4 $k=1.7$ 的 32／42／4 與 $t'_{25}=0.3704$。輸出 `selfTest: N passed, M failed`。

### 6.3 數學排版

公式右側等寬字標 Eq. 編號；Eq. 3、9、10、11、14 在 < 600px 拆兩行；SVG 內用 Unicode；靜態 TeX 載入時渲染一次，動態數字用 `textContent`（4.0）；只有 `<details>` 首次展開時對其內容呼叫一次 `renderMathInElement`。

### 6.4 文案

繁體中文、台灣用語、字面學術語氣；不用比喻、反問、宣傳形容詞。用詞：「分組／解組」、「群組 token」、「親和度」、「平均連結法」、「門檻」、「壓縮損失」、「保留比例」、「群組引導」、「token 數變化圖」、「常見錯誤」；梯度方向寫「降低 $f_s$（使合併分數上升）」，不寫「鼓勵合併」。專有名詞第一次出現附英文。區分「論文結果」「作者認為」「本頁推導」「推定」。

### 6.5 無障礙、響應式與動態

- 滑桿 `<label>`＋`aria-valuetext`；live region 規則見 4.0；所有系列以形狀／線型＋文字區分；拖曳皆有鍵盤與數字輸入替代；觸控區 ≥ 44px；390px 無水平捲動；表格橫捲容器；C1 版面見 C1 規格。
- 動態：本頁沒有自動播放的動畫；只有 hover／狀態切換的 transition（160／240ms）。`prefers-reduced-motion: reduce` 時取消所有 transition 與 smooth scroll，元件行為不變。

### 6.6 引用與標籤

內文每個論文數字附出處；元件圖版底部一行來源說明；「本頁推導」「推定」兩種 pill 各附一行依據。

### 6.7 研究筆記與時間線

- 本頁筆記（`tutorial_src/585/notes/`，依 `hit-lineage-hub` 的閱讀順序；S14 照此順序列出並附 frontmatter `title`）：`hit-lineage-hub`（總覽）→ `hit-latent-hierarchy-and-pixel-space`（S1）→ `hit-hnet-blt-to-image-groups`（S3）→ `hit-merging-baselines-fixed-rates`（S3）→ `hit-affinity-grouping`（S3、S4）→ `hit-compression-budget`（S6）→ `hit-group-guidance`（S9）→ `hit-compact-position-and-joint-grouping`（S7 位置編碼段、S8 文字到影像）。建置時若筆記有增減，以當時檔案為準。
- 系列筆記（已確認 slug，只連結不重寫）：`tokc-lineage-hub`、`tokc-vit-pruning-and-merging`（S3）、`tokc-training-free-diffusion-merging`（S13）、`tokc-learned-routing-dynamic-dit`（S9）、`tokc-hourglass-and-latent-interfaces`（S2）、`tokc-dynamic-chunking-bytes-to-pixels`（S3）、`tokc-content-adaptive-tokenization`（S12）、`tokc-position-for-merged-tokens`（S7）。建置時確認各篇存在且主題與連結處相符。
- 既有站內筆記：`pixel-space-diffusion`、`x-prediction-manifold-view`、`aniso-pixel-patchified-setting`、`flow-matching-velocity-target`、`generative-eval-metrics`、`diffusion-classifier-guidance-background`。
- 教學：系列 `token-compression-lineage.html`；兄弟 `rti-elastic-token-compression.html`。
- 時間線：`notes/timeline.json` 原樣嵌入 `<script type="application/json" id="timeline-data">`；不存在或解析失敗時整節顯示「時間線資料尚未提供」。`timeline.json` 目前 15 筆（修訂時補入 2410.19324，與 `hit-latent-hierarchy-and-pixel-space` 的 arxiv 欄一致），`TL_GROUP`（arXiv → 類別 → 卡片筆記）：潛空間階層 2112.10752、2212.09748、2410.06940、2510.11690 → `hit-latent-hierarchy-and-pixel-space`；像素空間 2301.11093、2410.19324、2504.07963、2511.13720 → `hit-latent-hierarchy-and-pixel-space`；token 合併 2210.09461、2303.17604 → `hit-merging-baselines-fixed-rates`；圖文分組 2202.11094 → `hit-affinity-grouping`；序列切塊 2412.09871、2507.07955 → `hit-hnet-blt-to-image-groups`；稀疏引導 2601.01608 → `hit-group-guidance`；本文 2611.22962 → `hit-lineage-hub`。不在表中的新條目歸「其他」並在建置回報列出。S13 的文字段依此改為：潛空間階層 → 像素空間 → token 合併（固定比例）→ GroupViT → BLT／H-Net → Sparse Guidance → HiT。

### 6.8 建置後自我檢查

截圖 390／768／1024／1440＋深色、鍵盤 focus、reduced-motion；依字體、留白、視覺層級、色彩、動態、微互動、響應式、原創性逐項評分修正；`?test` 全數通過；逐條核對 12 個學習目標的答案出現在指定章節內文；C1 在 390px 單欄下拖曳與鍵盤操作皆可用。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | C1 的 $T_r$ 預設不是論文值 | 二維餘弦親和度差距大，$T_r=0.05$ 時 $A$ 全為 0/1 | 預設 0.2 標「推定」，提供「論文值 0.05」chip；常見錯誤 10 說明 $T_r$ 小的後果 |
| R2 | C3 是擬合模型 | 論文沒有 FLOPs 公式與 SwiGLU 寬度 | 頁面寫「推定」與「擬合殘差」，不寫預測或驗證；不分組對照註明為巧合 |
| R3 | 不重繪論文的群組圖 | 群組資料不在原文文字中 | S8 只用文字描述並附圖號；連專案頁 |
| R4 | 論文各處數字的表述不同 | 摘要：約 40% 累積訓練 FLOPs；Sec. 1：約一半的更新次數與 40% FLOPs；Fig. 1d：2.4 倍計算、2.3 倍更新。T2I：Table 2 設定 $s=6.5$，Fig. 6 樣本 CFG 7.0。摘要「約 6 倍參數、3 倍 FLOPs」對應 1.60/0.25 = 6.4、408.0/139.4 = 2.9 | 照錄各處原文並標出處，不合併、不換算 |
| R5 | ImageNet 學習率 | 正文只寫 T2I 的學習率 | S11 寫「ImageNet 遵循 JiT 配方」 |
| R6 | 筆記可能再變動 | 修訂時 `hit-` 筆記 8 篇與 timeline 14 筆已存在，`tokc-` 筆記 8 篇已存在 | 6.7 已填入；建置時重新讀取，有增減以當時檔案為準 |
| R7 | 去重 | 系列教學會講 ToMe、H-Net、hourglass、位置編碼 | 每個背景概念一句＋`tokc-` 連結；S12 只放 RTI 對照表 |
| R8 | C2 合成資料可能被誤認為論文數據 | | 圖版標題與 footer 明寫合成、推定 |
| R9 | 跨語言數值一致 | 原設計以 PRNG 生成 C2 分數，JS 與 Python 的 cos/log 末位差異可能改變計數 | 改為內嵌 `c2_scores.json`；分數避開 τ 網格，比較無同值 |
| R10 | 平均連結的同分處理 | | 差 ≤ 1e-12 視為同分，取先找到的群組對，與 check.py 相同 |
| R11 | 本頁推導範圍 | Eq. 4 = Eq. 10、梯度符號、$f_h=f_s$ 時的下界、合併分數不增、2688 對 4608、注意力的 $\beta$、Eq. 16 單調、$T_r$ 權重比 | 一律加「本頁推導」標籤與一行依據 |
| R12 | C1 的 $q=k$ 玩具設定 | $S_{ii}=1$ 與論文不同 | 元件說明寫明（C1 簡化） |
| R13 | C2 校準規則 | 單階段網格掃描是本頁自訂，與論文的逐階段校準不同 | 標「推定」並寫出論文做法 |

---

## 8. 修訂紀錄（round 1）

| 審查項 | 處理 |
| --- | --- |
| P0-1 校準矛盾 | 校準改為「τ 網格 $i/200$ 上第一個 $f_h\ge\rho$ 的 $i$」，滑桿跳到該值（0.745、0.585）；§5 加 0.740 與 0.745 兩列；selfTest 比對整數 $i$；標推定並寫 App. A.3 做法（C2、R13） |
| P0-2 PRNG | 改內嵌 `verify/c2_scores.json`，分數避開 τ 網格；重跑 check.py，§5 C2 更新（0.90 列的 $K_1$ 由 124 變 123 等，四位小數化所致） |
| P1-3 簡單／複雜影像 | 步驟 5 改為只引「隨影像內容與擴散時間改變」；C2 明寫個別差異來自合成分布 |
| P1-4 $S_{ii}=1$ | 步驟 3 寫對稱化理由與 $S_{ii}=q_i^\top k_i$；C1 簡化段說明玩具差異 |
| P1-5 C1 提前用 S6 概念 | $\tilde K$ 移到 C1 的收合「進階（S6 會用到）」；讀數與觀察不再提 |
| P1-6 LO4 無元件 | C1 新增 2×3 patch 網格面板，{1,2}、{1,2,4} 在網格中不相鄰 |
| P1-7 先備知識與密度 | S1 背景收合區；S2 只標分組／解組／跳接；S3 先講餘弦、三 token 計算例，單調性與對照移入收合；原 S4 拆成 S4（軟權重、池化）與 S5（解組、跳接、梯度、Table 3）；sigmoid、softmax 溫度在首次出現處定義 |
| P1-8 C4 方向與 LO11 | 時間軸兩端標「純噪聲／乾淨影像」；新增倍率階梯圖 |
| P1-9 C1 互動 | 加 `setPointerCapture`、`touch-action:none`、`role="slider"`、方向鍵 ±1°／Shift ±10°、mod 360、整數取整 |
| P1-10 390px | C1 三段版面與最小尺寸；Eq. 3、9 加入拆行清單 |
| P1-11 動態 KaTeX | 4.0、6.3：TeX 只渲染一次，數字用 `textContent`，live region 一行摘要 |
| P1-12 色彩 | 熱圖改發散色階＋數字預設顯示，深色沿用 token 深色值；直方圖改 small multiples |
| P2-13 | 刪除「32 個 in-context class token」，改寫為條件化與輸入處理不同（App. A.4、Table 6） |
| P2-14 | 「擬合殘差」，明寫非獨立預測；不分組對照為巧合 |
| P2-15 | R4 改為摘要 40% FLOPs、Sec. 1 一半更新、Fig. 1d 2.3／2.4 倍 |
| P2-16 | 步驟 8 固定分組列寫明保留 $q,k$ 投影與軟池化／解組 |
| P2-17 | LO2、符號表、步驟 2、C3 限定 ImageNet D18；步驟 11 加 T2I 設定（512 px、1024 patch、D27） |
| P2-18 | LO3 答案內標〔本頁推導，假設精確的平均連結法〕；步驟 3 引原文 |
| P2-19 | 步驟 4、S5 加 Table 10 但書 |
| P2-20 | 常見錯誤 3 拆成 3（固定群組數）與 4（跨裝置加總、梯度累積） |
| P2-21 | Hero 卡 2 改「每一個分組階段…（兩階段後為四分之一）」 |
| P2-22 | LO11 加「（僅 600 epoch 端點）」；C4 chip 同步 |
| P2-23 | 「token 數變化圖」；梯度方向文字；S4 標題改為「每個群組 token 由所有輸入 token 加權平均而成」 |
| P2-24 | Eq. 10 收合在 Eq. 4 旁；C3 資料結構與 $J$ 切換預設；6.5 寫明無自動動畫與 reduced-motion 行為 |
| 協調者訊息 | slug 確認：`token-compression-lineage`、`rti-elastic-token-compression`、8 篇 `tokc-`；`hit-` 筆記 8 篇與 timeline 14 筆已出現，6.7、大綱延伸連結、`TL_GROUP` 依實際檔案填入 |

## 9. 修訂紀錄（readiness）

- C4 比較規則：$t'$ 四捨五入到 1e-9 後用閉區間；$k=1$ 的引導步數改為 45（`check.py` 同規則重算，§5 更新）。
- C3 載入時狀態：2×2 預設 chip；不分組 12 區塊標為對照；2×2×4 chip 寫明 `ratios:[2,2,4]`。
- 頁面符號（不改論文公式本身，只在頁面上避免混淆）：壓縮比寫 $r$，解組輸出 $r_i$ 在頁面改寫為 $\mathbf r_i$（粗體）並在 Eq. 7 旁註明；群組引導的倍率保留 $r_c,r_u$ 並在 S9 開頭說明「與壓縮比 $r$ 無關」；引導強度在頁面寫 $w$（註明論文寫 $s$），$s(U,V)$ 與 $s_{b,m}$ 保留；sigmoid 寫 $\operatorname{sig}(\cdot)$，$\sigma$ 只用於噪聲程度 $1-t$（Eq. 16）。
- Eq. 14：$v_c^{(r)}$ 為條件 $c$ 下、群組數倍率 $r$ 時網路預測的速度，$v_\varnothing^{(r)}$ 為無條件（類別丟棄）預測；Eq. 14–15 的 $t$ 是偏移後網格上的時間 $t'$。
- Hero 卡在第一次出現時以白話定義 FID 與 JiT。
- S13 大綱與 6.7 一致；`timeline.json` 補 2410.19324（15 筆）；`c2_scores.json` 只保留 `verify/` 下一份（check.py 改為寫到自身目錄）。

## 10. 建置紀錄（build round）

- 產出 `build/index.html`（約 141 KB），由 `build/parts/`（head、css、css2、body1、body2、script.js）以 `build.sh` 組裝；時間線與 `verify/c2_scores.json` 在組裝時內嵌。
- 與設計的差異：C1 合併樹縱軸在 0.8 以上放大（分段比例，圖標題註明），因為預設的四次合併分數都在 0.92–0.98；token 數變化圖與 S2 骨幹圖以 HTML 實作（文字保持實際字級）；C1 的 patch 網格只標 token 編號，群組以形狀表示；C1 拖曳的 move/up 監聽掛在 window，避免重繪時失去 pointer capture。
- selfTest 74/74；A、P、t′₁ 的相對容許改用 check.py 末段的五位有效數字參考值（§5 已註明）。
- impeccable：0 項（只有 numbered-section-labels advisory，依 DESIGN.md 保留）。390px（iframe）scrollWidth 未超出；深色模式截圖確認熱圖數字可讀。
