# 設計文件：〈Generation as Reverse Encoding（RevFlow）〉互動教學

- 論文：Cheng Shi、Sibei Yang、Hanzhong Guo、Yipeng Qin、Pietro Lio、Yizhou Yu，arXiv 2611.34782（v1，2026-11-25）。資料庫編號 No. 606。程式碼：github.com/ChengShiest/RevFlow。
- 原文：`606/src/paper.txt`。本文件中的「Eq. n」「Sec. n」「App. A」「Fig. n」「Table n」「Def. 1」「Prop. 1」皆指論文原文編號。
- 產出：`606/build/index.html`（單一 HTML）。上線位置 `tutorials/revflow-reverse-encoding.html`（slug `revflow-reverse-encoding`），站內相對連結以上線位置為準。
- 驗算腳本（只用來核對本文件數字，算法與頁面相同）：`606/verify/common.py`（mulberry32、Box–Muller）、`c1.py`、`c2.py`、`c3.py`、`c3_scan.py`、`paper_numbers.py`。
- 研究筆記：`606/notes/`（總覽 `revflow-lineage-overview` + 6 篇主題 + `timeline.json`，14 筆）。
- 標記慣例：〔本頁推導〕＝本設計自行推導、論文沒有寫出；〔推定〕＝本頁為了做玩具或對齊論文而自選、論文未給的值；〔玩具〕＝互動元件中只為示意結構的設定。頁面沿用同樣的標籤（6.3）。
- **時間方向警告**：本論文 $t=0$ 是乾淨目標、$t=1$ 是純噪聲（Eq. 2），與站內 `prediction-targets-k-diff` 教學（$t=1$ 為資料）相反。本文件與頁面一律採本論文慣例。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

凍結的預訓練視覺編碼器（本文主模型為 DINOv3-L）把影像 $x$ 逐層轉成表徵 $h_\ell=E_\ell(x)$：淺層保留外觀與空間細節，深層強調語意。作者把這條由像素到語意的過程稱為**語意編碼路徑**（semantic encoding path），並主張生成應該**反向走這條路徑**：先由類別 $y$ 生成深層表徵 $z^L$，再以 $z^L$ 為條件生成淺層表徵 $z^M$，最後用輕量解碼器把 $z^M$ 解成像素（Eq. 1）。理論依據是 Prop. 1：乾淨目標預測器 $a^*=\mu(u)+\mathbb E[r\mid a_t,u]$ 的輸入 Jacobian 等於 $\frac{1-t}{t^2}\mathrm{Cov}(r\mid a_t,u)$；直接由類別生成淺層特徵時條件平均沒有資訊、殘差變異大（$V(h_7\mid y)=0.958\,V$），乾淨目標是尖銳的 Dirac 點，最大 Jacobian 在 $t\to0$ 發散；以深層特徵為條件時 $V(h_7\mid h_{23})\le0.426\,V$，作者**假設**殘差是窄的局部擾動，映射平滑（App. A）。兩個階段共用一個標準 DiT 骨幹（**HyC-DiT**）：Stage 1 用全域類別條件，Stage 2 用逐 token 的空間 AdaLN 條件（Eq. 5），並用乾淨目標預測（x-prediction）讓權重共享不退化（Fig. 5）。另外，因為生成的是編碼器表徵，可以直接在表徵空間做分佈匹配（**原生匹配**，native matching，NM），不必解碼再編碼，得到每階段一步的生成器（Eq. 7–8）。ImageNet 256×256 上，訓練 5 個 epoch 即達 FID 1.96（XL，690M）與 1.74（H，1.55B）；由 DINOv3-L 第 7 層解碼的 89M 解碼器重建 PSNR 30.25 dB（RAE 以 DINOv2-B 最後一層 latent、415M 解碼器為 18.09 dB；同一個 89M 解碼器改解第 23 層為 19.87 dB，Table 4a）；NM 後 RevFlow-XL 以 2 NFE（每階段 1 次）達 FID 2.85（Abstract、Table 1、Table 3）。

### 1.2 研究動機（Sec. 1）

1. 預訓練語意表徵加速生成學習（REPA [63]、RAE [66]），但抽象化讓「回到像素」更難：RAE 類方法把這個負擔交給獨立訓練的解碼器，其大小通常約為傳統 latent diffusion 解碼器的 10 倍；Fig. 2 中 RAE 把灰色聖誕帽重建成紅色。作者的說法：抽象化**轉移**而非**消除**難度。
2. VA-VAE [61]、RAEv2 [51] 改進目標空間（對齊或多層融合），但仍然只生成單一目標，再學一條獨立的解碼路徑。
3. 作者的提問：既然編碼器已經提供一條由像素到語意的路徑，為什麼還要另外學一條由表徵回到像素的路徑？
4. 與擴散的類比（Sec. 1）：前向擴散逐步移除資訊，生成以簡單的反向步驟補回；高斯噪聲大致依空間頻率移除資訊 [47]，編碼器則依**學到的順序**移除：先移除外觀與空間細節，最後才移除語意。反向時先確定語意，把表徵學習交給預訓練編碼器。

### 1.3 貢獻（Sec. 1 條列）

1. 「生成即反向編碼」：把編碼的反問題當成生成過程，沿凍結編碼器由深到淺反向走語意編碼路徑。
2. HyC-DiT：一個共用 Transformer 骨幹學所有階段；類別作全域條件、深層表徵經逐 token AdaLN 作空間條件；乾淨目標預測避開速度預測在共享下的退化，以約一半參數達到與分開兩個生成器相當的品質。
3. 預訓練語意表徵同時作為生成目標與原生分佈匹配空間；NM 讓單階段 RAE 與兩階段 RevFlow 都能少步生成，不需像素解碼與再編碼。
4. 快速收斂的訓練配方：標準 DiT 骨幹 5 個 epoch 達 1.96／1.74 FID。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面在第一次使用處定義：$x,y,E_\ell,h_\ell,\ell$ 在 S1 正文；其餘在 S2 **開頭**的符號表，S3 起使用；S4 的 $a,u,\mu,r,V,R^2$ 在 S4a 開頭再定義一次）

| 符號 | 意義 |
| --- | --- |
| $x$、$y$ | 影像、類別標籤 |
| $E_\ell$、$h_\ell=E_\ell(x)$ | 凍結編碼器第 $\ell$ 層的輸出（層編號從 0 起算） |
| $L$、$M$ | 深層、淺層的層編號；主模型 $L=23$、$M=7$（DINOv3-L），$M<L$ |
| $z^L=h_L$、$z^M=P_{\text{target}}(h_M)$ | Stage 1、Stage 2 的生成目標；$P_{\text{target}}$ 把 1024 通道壓到 128 通道、保留 16×16 空間格 |
| $t\in[0,1]$、$\epsilon$ | 時間（**$t=0$ 乾淨、$t=1$ 純噪聲**）、標準高斯噪聲 |
| $z^\ell_t$、$\hat z^\ell$ | 加噪目標（Eq. 2）、網路預測的乾淨目標（Eq. 3） |
| $f_\theta$、$c^\ell$ | 共用的網路、第 $\ell$ 階段的條件（$c^L=y$、$c^M=\tilde z^L$） |
| $\tilde z^L$、$\sigma$ | 加了條件噪聲的深層條件（Eq. 4）、條件噪聲強度 |
| $a$、$u$ | App. A 的一般記號：某階段的乾淨目標 $a\in\mathbb R^d$、條件 $u$ |
| $\mu(u)$、$r$ | 條件平均 $\mathbb E[a\mid u]$、殘差 $r=a-\mu(u)$ |
| $V(a)$、$V(a\mid u)$ | $\mathbb E\|a-\mathbb Ea\|^2$、$\mathbb E\|a-\mathbb E[a\mid u]\|^2$ |
| $R^2$ | 線性預測解釋的變異比例 |
| NFE | 網路評估次數（不含解碼器） |
| $s_t$、Jacobian、Dirac 點 | score $\nabla\log p_t$；$\nabla_{a_t}a^*$ 為輸出對輸入的偏導數矩陣；Dirac 點＝機率全部集中在一點的分佈（有限個訓練樣本） |

#### 步驟 1：語意編碼路徑與 RAE 的問題（Sec. 1–2、Fig. 2）

- 編碼器把影像轉成越來越抽象的表徵；本文的「編碼器」指 DINO、CLIP 這類為視覺理解預訓練的凍結編碼器，不是 VAE 編碼器（Sec. 1 註 1）。
- 重建式表徵（VAE latent）保留細節；語意表徵壓抑與語意判別無關的變異。後者讓生成好學，但回到像素時要補的細節更多。
- 生成比重建難：抽象化讓不同影像共享相近的表徵，反問題變成欠定（underdetermined），一個表徵對應多張可能的影像，而不是固定目標（Sec. 1）。
- Fig. 2 左：SD-VAE 23.40 dB（49M 解碼器）、RAE 18.09 dB（415M）、RevFlow 30.25 dB（89M）。注意三者的輸入不同：RAE 的 18.09 dB 來自 DINOv2-B 最後一層 latent（App. A.1），RevFlow 的 30.25 dB 來自 DINOv3-L 第 7 層。同編碼器、同 89M 解碼器的對照是 Table 4a：解第 23 層 19.87 dB、解第 7 層 30.25 dB。頁面引用 18.09 對 30.25 時一律附上編碼器與層，並同時給 19.87。
- 下一步的理由：若生成本來就要反向走編碼，可以把編碼路徑本身當成生成的骨架。

#### 步驟 2：Flow matching 與乾淨目標預測（Sec. 3.2–3.3，Eq. 2、3、6）

$$z^\ell_t=(1-t)\,z^\ell+t\,\epsilon^\ell,\qquad \epsilon^\ell\sim\mathcal N(0,I)\tag{Eq. 2}$$
$$\hat z^\ell=f_\theta(z^\ell_t,t,c^\ell),\qquad c^L=y,\quad c^M=\tilde z^L\tag{Eq. 3}$$

- 網路直接輸出乾淨目標（clean-target prediction，[32]），損失 $\mathcal L^\ell_{FM}=\mathbb E\|t^{-1}(\hat z^\ell-z^\ell)\|_2^2$（Sec. 3.3）。
- 〔本頁推導〕路徑速度 $v=dz_t/dt=\epsilon-z$；由 $\hat z$ 換算 $\hat v=(z_t-\hat z)/t$，則 $\hat v-v=(z-\hat z)/t$，所以 $\mathcal L_{FM}=\mathbb E\|\hat v-v\|^2$：網路輸出乾淨目標，損失等於速度空間的 MSE。生成時從 $t=1$ 的噪聲出發，以 Euler 步 $z_{t-\Delta}=z_t-\Delta\,\hat v$ 走到 $t=0$。
- 兩階段聯合訓練：$\mathcal L=w_c\mathcal L^L_{FM}+\mathcal L^M_{FM}$（Eq. 6），$w_c$ 平衡兩階段（數值未給）。

#### 步驟 3：實例化反向路徑（Sec. 3.1，Eq. 1，Fig. 3）

$$y\to z^L\to z^M\to x\tag{Eq. 1}$$

- **選層**：Fig. 3 顯示相鄰層的 CLS 表徵 cosine 相似度形成高相似區段（regime），區段之間有明顯轉折。走過每一層會大幅增加取樣成本，並累積上游的平均導引漂移，所以每個區段只取一個代表。作者檢查的編碼器都以「較早」與「較晚」兩個區段為主，因此 $K=2$（$\ell_2=L$、$\ell_1=M$）。
- DINOv3-L：$h_M$ 取第 7 層（早期區段末端），$h_L$ 取第 23 層（晚期區段末端，最後一個 Transformer block 的輸出，在最終 normalization 之前）；層編號從 0 起算。
- $z^L=h_L$；$z^M=P_{\text{target}}(h_M)$，在 16×16 格上把通道由 1024 壓到 128。解碼端：decompressor 還原通道後，12 個 block 的 Transformer 解碼器把格點映射到像素。
- 依 RAE [66]，$P_{\text{target}}$ 與像素解碼器在編碼器凍結下**一起為重建訓練**，生成訓練時固定。
- 每張訓練影像由編碼器與 $P_{\text{target}}$ 提供配對目標 $(z^L,z^M)$。

#### 步驟 4：平均導引與殘差（Def. 1、Prop. 1，App. A，Eq. 9–10）

- Def. 1：目標 $a\in\mathbb R^d$、條件 $u$；$\mu(u)=\mathbb E[a\mid u]$、$r=a-\mu(u)$（零平均），$V(a\mid u)=\mathbb E\|r\|^2$。加噪 $a_t=(1-t)a+t\epsilon$，減去 $(1-t)\mu(u)$ 得殘差插值 $r_t=(1-t)r+t\epsilon$。
- 對任一固定預測器 $g$：$V(a\mid u)\le(1-R^2)V(a)$，$R^2=1-\mathbb E\|a-g(u)\|^2/V(a)$，所以量到的 $R^2$ 給出 $V$ 的上界（線性 $g$ 的 $1-R^2$ 是上界，非線性預測器只會更緊）。
- 假設（Def. 1）：$\mathbb E\|a\|^2<\infty$；$\epsilon\sim\mathcal N(0,I_d)$ 與 $(a,u)$ 獨立；$t\in(0,1)$；Jacobian 是 $a^*$ 對 $a_t$ 取偏導（$(\nabla_{a_t}a^*)_{ij}=\partial a^*_i/\partial a_{t,j}$）。頁面 S4a 開頭先用字面定義條件平均、條件共變異數 $\mathrm{Cov}(r\mid a_t,u)=\mathbb E[rr^\top\mid\cdot]-\mathbb E[r\mid\cdot]\mathbb E[r\mid\cdot]^\top$、score、Jacobian 與其最大值（operator norm：輸入擾動被放大的最大倍數）、Dirac 點。
- Prop. 1（對每個 $t\in(0,1)$）：

$$a^*(a_t,t,u)=\mu(u)+\mathbb E[r\mid a_t,u]=\mu(u)+\frac{r_t+t^2s_t(a_t\mid u)}{1-t}\tag{Eq. 9}$$
$$\nabla_{a_t}a^*=\frac{1-t}{t^2}\,\mathrm{Cov}(r\mid a_t,u),\qquad \mathbb E\,\mathrm{tr}\,\nabla_{a_t}a^*=\frac{1-t}{t^2}\,V(a\mid a_t,u)\tag{Eq. 10}$$

  其中 $s_t=\nabla\log p_t$ 為 score，$V(a\mid a_t,u)=\mathbb E\|a-a^*\|^2$。證明：線性取期望得第一式；score–noise 恆等式 $s_t=-\epsilon^*/t$ 得第二式；對 Bayes 式微分、商法則中兩個 $-\frac1{t^2}\mathbb E[r\mid\cdot]r_t^\top$ 項相消得 Eq. 10。
- 作者的兩個情形（App. A，Prop. 1 之後）：
  1. 條件平均沒有資訊、$V(a\mid u)$ 寬（直接 $y\to h_M$）：目標來自乾淨影像，$P(r\mid u)$ 是**尖銳的 Dirac 點**。高 $t$ 時各模式重疊，$\mathbb E[r\mid a_t,y]\approx0$（細節被平均掉）；$t\to0$ 時在 Dirac 邊界 $\mathrm{Cov}$ 維持 $\Theta(V(a\mid y))$，$\|\nabla a^*\|_{op}\propto\frac{1-t}{t^2}V(a\mid y)\to\infty$。論文稱為「stiff Dirac jumps」；頁面字面寫成「最大 Jacobian 隨 $t^{-2}$ 增加，預測在模式之間急遽跳動，而不是平滑地修正」。
  2. 條件平均有資訊、$V(a\mid u)$ 窄：論文**假設**殘差是局部擾動 $\sigma_r^2I_d$（不是由 Prop. 1 證出，也不是對 $h_{23}\to h_7$ 量到的分佈形狀），在此假設下 $\mathrm{Cov}(r\mid a_t,u)=O(t^2)$ 抵消 $t^{-2}$，論文寫 $\|\nabla a^*\|_{op}<1$，映射在 $\mu(u)$ 附近平滑（也有利於 Fig. 5 的骨幹共享與 Sec. 3.4 的一步匹配）。頁面在 S4b 與 C1 都標為「論文的建模假設」。
- 〔本頁推導〕在情形 2 的假設下，一維高斯殘差 $r\sim\mathcal N(0,v)$ 的閉式：$\mathrm{Cov}(r\mid r_t)=\frac{v\,t^2}{(1-t)^2v+t^2}$，$\nabla a^*=\frac{(1-t)v}{(1-t)^2v+t^2}$，與 $r_t$ 無關、$t\to0$ 時趨近 1。它在 $t\in[0.01,0.99]$ 上的最大值略大於 1（$v=0.958$：1.200 於 $t\approx0.285$；$v=0.426$：1.097），所以「<1」嚴格說是「有界、約為 1」。頁面正文如實寫出：「論文寫 <1；本頁計算一維高斯情形的最大值為 1.20（v = 0.958）與 1.10（v = 0.426），仍有界」（`c1.py`；R2）。
- 〔本頁推導〕Dirac 情形的最大 Jacobian 由**模式間距**決定：$K$ 個等距、間距 $\Delta$ 的點，相鄰兩點中點的 $\mathrm{Cov}=\Delta^2/4$，所以 $J_{\max}\approx\frac{1-t}{t^2}\frac{\Delta^2}{4}$。固定點數時 $\Delta^2\propto V(a\mid u)$，這就是論文的 $\Theta(V)$；但同樣的變異、點數不同時差很多（$v=0.958$、$t=0.05$：$K=4,8,16$ 為 72.8、17.3、4.28）。論文的 $\Theta(V(a\mid y))$ 是模式彼此分離（$t$ 小於模式間距）時的漸近敘述。
- 〔本頁推導〕Eq. 10 第二行的**期望**跡 $\frac{1-t}{t^2}V(a\mid a_t,u)$ 在 Dirac 目標、$t\to0$ 時反而趨近 0：後驗幾乎處處確定，只在邊界附近不確定。這比論文寫的更強；論文關於 stiff Dirac jumps 的論證依靠的是最大值（operator norm），不是 Eq. 10 第二行。頁面字面描述：「在模式之間的大部分區域，預測幾乎不隨輸入改變；在兩個模式的分界處，預測從一個點跳到另一個點。」
- 下一步的理由：要讓 Stage 2 的殘差窄，需要一個能提供強平均導引的條件；深層表徵就是候選。

#### 步驟 5：編碼器層統計的證據（App. A.1，Fig. 7）

- 量測：DINOv3-L 全部 24 層、ImageNet 驗證集 50K 張（256×256），每張取 64 個 patch token；對每對層在同一位置擬合線性映射並回報 $R^2$。類別的最佳預測器是各類平均。Fig. 7c 用 mean-pooled 特徵訓練 LDA（每類 40 張訓練、10 張測試）。
- 兩個條件：
  - **(E1) 強的跨層平均導引**：$V(a\mid c)$ 遠小於 $V(a\mid y)$。$a=h_7$、$c=h_{23}$ 時線性 $R^2=57.4\%$，類別只解釋 4.2%，所以 $V(a\mid c)\le0.426\,V(a)$，而 $V(a\mid y)=0.958\,V(a)$。Stage 2 要生成的變異不到直接由類別生成的一半。$R^2$ 從 $c=h_8$ 的 96.2% 單調降到 $h_{23}$ 的 57.4%；每個 block 損失的變異由第一個 block 的 0.6% 增加到最後幾個的 15–22%（Fig. 7a）。層距越大，(E1) 越弱。
  - **(E2) 低變異、依類別組織的深層目標**：類別解釋 $h_{23}$ token 變異的 18.6%，但只解釋 $h_7$ 的 4.2%；mean-pooled 特徵的類別解釋變異 44.4% 對 23.6%；線性探針準確率 77.7% 對 23.2%（Fig. 7c）。當 $y$ 大致可由 $c$ 還原時 $V(a\mid c,y)\approx V(a\mid c)$，所以 Stage 2 只以 $c$ 為條件。
- 像素端：$V(x\mid h_\ell)$ 隨深度增加，並是任何確定性解碼器的下界 $\mathbb E\|x-D(h_\ell)\|^2\ge V(x\mid h_\ell)$；從深層解碼必須靠學到的先驗補細節，結果可能被平均或偏向模型偏好的樣式。所以最後一層要淺。但越淺 (E1) 越弱：淺層的位置是**重建與生成的權衡**。每多一層、上游預測 $\hat c$ 不完美，就多一份平均導引漂移 $\mathbb E\|\mu(c)-\mu(\hat c)\|^2$（以 $\sigma>0$ 的條件噪聲緩和；在同一相似區段內加層沒有好處）。

#### 步驟 6：HyC-DiT 架構（Sec. 3.2，Eq. 4–5，Fig. 4b）

- 骨幹共用；條件路徑在 Stage 1 接受全域類別嵌入，在 Stage 2 接受與目標空間對齊的 token。把條件壓成單一向量會失去與淺層目標的對應。
- 條件噪聲（Stage 2 訓練用真實影像的深層表徵、取樣時用 Stage 1 生成的表徵）：

$$\tilde z^L=\sqrt{1-\sigma}\,z^L+\sqrt{\sigma}\,\epsilon_c,\quad\epsilon_c\sim\mathcal N(0,I),\quad\sigma\sim\mathrm{Uniform}[\sigma_{\min},\sigma_{\max}]\tag{Eq. 4}$$

  $\sigma$ 也輸入模型，仿照 cascaded diffusion 的 noise-conditioning augmentation [22]。$\sigma_{\min},\sigma_{\max}$ 未給。論文只說擾動**訓練時**的條件，沒有說取樣時是否也加噪、取樣時的 $\sigma$ 輸入多少（R13）。代價與好處：條件加噪會削弱深層 token 與輸出的對應（元件 C2 顯示），好處是讓 Stage 2 不過度依賴真實深層表徵的細節、能承受 Stage 1 生成的誤差；拿掉時 FID 1.96 → 2.10（Table 4c），作者在 App. A 稱其緩和平均導引漂移 $\mathbb E\|\mu(c)-\mu(\hat c)\|^2$。〔本頁推導〕若 $z^L$ 每個座標變異數為 1，$\tilde z^L$ 的變異數仍為 $(1-\sigma)+\sigma=1$。
- 全域與空間 AdaLN：

$$m^L=T(t)+E_y(y)+E_s(1),\qquad m^M_i=P_{\text{cond}}(c^M_i)+T(t+1)+E_\sigma(\sigma)+E_s(2)\tag{Eq. 5}$$

  $T$ 時間嵌入、$E_y$ 類別嵌入、$E_s$ 可學的階段嵌入、$E_\sigma$ 條件噪聲嵌入、$P_{\text{cond}}$ 把深層條件線性投影到 Transformer 寬度、$i$ 為空間 token 索引。$T(t+1)$ 讓兩階段落在時間嵌入的不相交範圍。每個 DiT block 中標準 adaLN-Zero 投影把 $m^L$ 或 $m^M_i$ 映射成 shift、scale、gate（$\beta,\gamma,\alpha$）：Stage 1 所有 token 共用同一組調變；Stage 2 每個 token 各自一組，讓每個深層 token 直接控制對應的淺層 token。
- 共用：attention、FFN、AdaLN 調變 MLP。階段專屬：目標的輸入／輸出投影（維度不同）、條件映射（類別與空間表徵各一）。

#### 步驟 7：乾淨目標預測讓共享可行（Sec. 3.3，Fig. 5）

- 兩種參數化描述同一個流，但共享時最佳化行為不同：乾淨目標預測的輸出落在各階段條件平均 $\mathbb E[z^\ell\mid c^\ell]$ 附近；速度預測把階段專屬的殘差與高維噪聲 $\epsilon^\ell$ 耦合（App. A），在共享骨幹中互相干擾。
- Fig. 5：x-pred 分開 1.92 FID（1.361B）對共享 1.96（690M）；v-pred 分開 2.07 對共享 4.25。參數由 1.361B 減到 690M（約一半；690/1361=0.507）。
- 輔助頭（依 RAEv2 [51]）：中間某個 block 接一個輔助頭，預測與最終頭相同的乾淨目標；訓練時多一份監督，取樣時提供 **REPA guidance**：由輔助預測往最終預測外插，兩者來自同一次骨幹評估。
- 取樣：每階段從獨立高斯噪聲開始，先生成 $z^L$（條件 $y$），再生成 $z^M$（條件為生成的深層表徵），最後解碼。ImageNet 多步生成用每階段 25 個 Euler 步；用 AutoGuidance 時每步另外評估一個弱模型。

#### 步驟 8：原生表徵匹配（Sec. 3.4、4.2，Eq. 7–8，Table 3）

- 表徵空間的分佈匹配（FD-loss [59]、[15]、[11]）對 VAE 生成器需要先解碼成影像再擷取特徵；RevFlow 直接生成編碼器表徵，可以**在目標表徵空間原生匹配**。
- 一步串接（從 $t=1$ 的獨立噪聲出發）：

$$\hat z^L=f_\theta(\epsilon^L,1,y),\qquad \hat z^M=f_\theta(\epsilon^M,1,\hat z^L)\tag{Eq. 7}$$

- 參考表徵由 EMA 生成器以加噪的真實表徵產生，視為 stop-gradient 目標。每個類別 $y$ 內，用 drifting loss [11] $\mathcal L^y_{\text{stat}}$ 匹配多尺度、逐通道的視窗統計量，用 IMQ-MMD [18] $\mathcal L^{\ell,y}_{\text{IMQ}}$ 對齊每一層 $\ell\in\{L,M\}$ 的邊際分佈；兩者都直接作用在生成的表徵上，不需樣本配對，也不需另外學特徵網路。保留乾淨目標 flow matching 作為正則：

$$\mathcal L_{NM}=\mathbb E_y\Big[\lambda_s\mathcal L^y_{\text{stat}}+\sum_\ell\lambda_\ell\mathcal L^{\ell,y}_{\text{IMQ}}\Big]+\lambda_{FM}\mathcal L\tag{Eq. 8}$$

  $\mathcal L$ 為 Eq. 6；λ 值、IMQ 核參數未給。適應後每階段一次評估加凍結解碼器。
- 〔本頁推導〕為什麼不做 NM 的一步輸出會失敗：MSE 最佳的乾淨目標預測是後驗平均 $\mathbb E[z\mid z_t,c]$。$t=1$ 時 $z_1=\epsilon$ 與 $(z,c)$ 獨立，所以 $\mathbb E[z\mid z_1,c]=\mathbb E[z\mid c]$：對所有 $\epsilon$ 都輸出同一個條件平均（只在 $t=1$ 精確成立；Eq. 9 的第二式在 $t=1$ 是 0/0，不用它推）。Stage 2 收到的也是這個平均，所以一步樣本都是同一個條件平均表徵。這與 Table 3 中適應前 FID 約 280 一致，但論文沒有給出原因。頁面用語：「輸出相同的條件平均」，不用「崩塌」。
- CFG（classifier-free guidance，無分類器引導）第一次出現時定義：取樣時把有條件與無條件的預測相減後放大外插，以加強條件的影響；頁面在 S8 第一次提到 CFG 處寫這句。
- Table 3（NFE 不含解碼器；適應後無 CFG；每組固定 NFE）：MeanFlow（VAE，1 NFE）3.43（外部參考）；LightningDiT 800ep（VAE，1）264.56 → +NM 291.68；RAE（編碼器，1）284.19 → +NM 3.46；RevFlow（編碼器，2）281.78 → +NM 2.85。NM 用的 RevFlow-XL 變體深層階段有 DDT 頭（852M 生成器參數），從 80 epoch checkpoint 初始化，適應 1,000 次更新。
- 作者的結論（照原文語氣）：同一配方直接匹配 800 epoch LightningDiT 的 VAE latent，1 NFE 的 FID 由 264.56 變成 291.68；這個對比「支持」語意表徵同時作為生成目標與匹配空間，「所測的 VAE 對照沒有受益」（一個 checkpoint、一種配方，不推廣成「VAE 不能做 NM」）。
- MMD 與 IMQ（頁面 S8 正文在 C3 之前寫出）：最大平均差異（maximum mean discrepancy）以核函數 $k$ 比較兩組樣本：$\mathrm{MMD}^2=\mathbb E[k(x,x')]+\mathbb E[k(r,r')]-2\mathbb E[k(x,r)]$，$x,x'$ 來自生成分佈、$r,r'$ 來自參考分佈；兩分佈相同時為 0，不需要把生成樣本與參考樣本一一配對。IMQ（inverse multiquadric）核的常見形式 $k(x,y)=(1+\|x-y\|^2/h^2)^{-1/2}$，$h$ 為尺度〔推定：論文只寫 IMQ-MMD [18]，未給形式與尺度〕。

#### 步驟 9：實驗（Sec. 4，Table 1、2、4、5，Fig. 6）

- 訓練：**5 epoch 的執行**用 Muon [26] 與 AdamW，global batch 256（80 epoch 的設定論文未寫）；FID 以 50,000 張（每類 50 張）、ImageNet 訓練集參考統計量計算。
- Table 1（ImageNet 256，無 CFG／有 CFG 或 AutoGuidance）：

| 方法 | Epochs | 解碼器 | 參數 | FID（無 CFG） | FID（CFG 或 AutoGuidance） |
| --- | --- | --- | --- | --- | --- |
| RAE | 800 | 415M | 839M | 1.51 | 1.13 |
| RAE | 80 | 415M | 839M | 2.16 | 1.74 |
| RAE | 20 | 415M | 839M | 3.71 | – |
| HyC-DiT-XL + RevFlow | 5 | 89M | 690M | 1.96 | – |
| HyC-DiT-H + RevFlow | 5 | 89M | 1.55B | 1.74 | – |
| HyC-DiT-XL + RevFlow | 80 | 89M | 690M | 1.36 | 1.33 |
| HyC-DiT-H + RevFlow | 80 | 89M | 1.55B | 1.17 | 1.09 |

  欄名照論文「Without CFG guidance」：RevFlow 列在這一欄**仍用了 REPA guidance**（Table 1 註），RAE 列沒有引導；右欄 RevFlow 另加 AutoGuidance [28]。所以「1.96 對 2.16」是有 REPA guidance 對無引導的比較，頁面表格加此註。每個訓練樣本的前向計算 RevFlow-XL 1,652.92 GFLOPs、RAE 1,134.59；5 對 80 epoch，RevFlow-XL 約為 RAE 累積前向 FLOPs 的 1/11（`paper_numbers.py`：10.98），FID 更低（1.96 對 2.16；前者有 REPA guidance）。80 epoch 的 RevFlow-H（1.17／1.09）優於 RAE 800 epoch（1.51／1.13），epoch 少 10 倍。像素空間 JiT-L 200 epoch 無 CFG 為 16.21。
- Table 2（六個表徵空間的正規化 Fréchet 距離平均 FDr6，依 FD-loss [59]）：RevFlow-XL 5ep 4.52 → 80ep 3.05；RevFlow-H 80ep 2.40，表中最低；REPA-E 3.04、RAE-XL 3.26（皆 800ep）。RevFlow-H 在六項中有四項優於 REPA-E。
- Table 4（全部 DINOv3-L、DiT-XL、5 epoch、50 NFE、windowed REPA guidance；PSNR 為驗證集重建）：

| 變體 | 解碼器 | PSNR | FID |
| --- | --- | --- | --- |
| RevFlow：$z^L_{23}\to z^M_7\to x$ | 89M | 30.25 | 1.96 |
| (a) 只有深層 $z^L_{23}\to x$ | 89M | 19.87 | 2.65 |
| (a) 只有淺層 $z^M_7\to x$ | 89M | 30.25 | 15.94 |
| (a) VAE 目標 $z^L_{23}\to$ SD-VAE | 49M | 23.40 | 2.94 |
| (a) 三階段 $+z_{16}$ | 89M | 30.25 | 2.51 |
| (b) + pooled context tokens | 89M | 30.25 | 2.02 |
| (b) 空間注入只在 block 1/7/14/21 | 89M | 30.25 | 1.96 |
| (b) 空間注入只在第一層 | 89M | 30.25 | 2.03 |
| (c) $\sigma=0$ | 89M | 30.25 | 2.10 |
| (d) v-prediction | 89M | 30.25 | 4.25 |

- 淺層位置掃描（Sec. 4.3、App. A.1）：DINOv3-S，深層固定 L12，解碼器 87.12M；淺層由 L1 → L3 → L5，FID 4.36 → 3.14 → 2.69，PSNR 30.19 → 28.32 → 25.48 dB。
- App. A.1 對消融的解讀：去掉深層 → 單一階段承擔全部 $V(a\mid y)$（$R^2=4.2\%$），1.96 → 15.94；解碼深層 → PSNR 30.25 → 19.87、FID 2.65；第三階段（第 16 層，在晚期區段內）→ 2.51，累積漂移大於縮短層距的好處；$\sigma=0$ → 2.10；SD-VAE latent 不在產生 $h_{23}$ 的編碼路徑上 → 2.94。
- 文字到影像（Sec. 4.4，512×512）：凍結 Flan-T5 嵌入取代 Stage 1 的類別條件；在相同設定下重現 RAE，5k–35k 次迭代比較。Table 5：RAE（重現）GenEval 0.74、DPG-Bench 77.6；RevFlow 0.78、81.6。35k 時 GenEval 77.54%、DPG 81.55，比 RAE 高 3.65 與 4.04（Sec. 4.4 正文）；頁面 S9 在這句之後緊接 R6 的說明句（正文差值換算出 RAE 73.89% 與 77.51，與 Table 5 的 0.74、77.6 分別差 0.11 與 0.09，DPG 的差異無法以四捨五入解釋）；30k → 35k 仍在上升（76.25%、80.09）。達到 RAE 最佳紀錄的速度快 1.6 倍（GenEval）與 3.1 倍（DPG）（Fig. 6、App. A.1）。作者也說最佳化配方對速度有貢獻。

#### 步驟 10：實作細節與常見錯誤（頁面 S10 逐條列出）

1. 時間方向：$t=0$ 乾淨、$t=1$ 噪聲；損失權重 $t^{-2}$，取樣由 $t=1$ 走到 0。
2. 層編號從 0 起算；第 23 層取在最終 normalization 之前。
3. $P_{\text{target}}$ 與解碼器先為重建一起訓練（編碼器凍結），生成訓練時固定。
4. Stage 2 條件必須逐 token 注入 AdaLN（空間對齊）；pooled context tokens 沒有幫助（2.02），只在第一層注入較差（2.03），在 block 1/7/14/21 注入與預設相同（1.96）。
5. Stage 2 的時間嵌入用 $T(t+1)$；兩階段各有階段嵌入 $E_s$。
6. 條件噪聲 $\sigma$ 要同時輸入模型；拿掉（$\sigma=0$）FID 1.96 → 2.10。
7. 共享骨幹時用乾淨目標預測；v-prediction 共享為 4.25。
8. NFE 不含解碼器；NM 後無 CFG；Table 4 用 50 NFE（每階段 25 Euler 步）。
9. 論文未給：$\sigma_{\min},\sigma_{\max}$、$w_c$、$\lambda_s,\lambda_\ell,\lambda_{FM}$、IMQ 核尺度、drifting 統計的視窗設定、REPA guidance 的尺度與視窗、輔助頭的 block 位置、Muon／AdamW 的超參數。頁面照實寫「未給」。

#### 步驟 11：結論與限制（Sec. 5）

論文的結論：把凍結編碼器層級的反向寫成一連串生成轉移，配合共享骨幹與輕量解碼器改善收斂與重建；乾淨目標預測讓參數共享有效；原生匹配支援每階段一步生成；編碼器層級是組織影像生成的有用結構。論文**沒有限制章節**。頁面 S12 以「本頁整理」列出論文自身資料顯示的權衡（淺層位置的重建／生成權衡、加層的漂移、未給的超參數、T2I 只與自行重現的 RAE 比較），並標明是本頁整理。

---

## 2. 學習目標（驗收測試）

每章結尾以 `<details>`（summary「檢查理解」）放該章檢查題；**參考答案的內容必須寫在該章正文**（不只在元件、表格、圖或答案框）。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 寫出反向編碼路徑與兩個目標 | RevFlow 的生成順序是什麼？主模型的 $z^L$、$z^M$ 各取自哪一層、是什麼形狀？ | $y\to z^L\to z^M\to x$（Eq. 1）；$z^L=h_{23}$（DINOv3-L 最後一個 block 輸出、最終 norm 前），$z^M=P_{\text{target}}(h_7)$，16×16 格、1024 → 128 通道；層號從 0 起算 | S3 |
| LO2 | 寫出插值、時間方向與損失 | Eq. 2 怎麼寫？$t=0$、$t=1$ 各是什麼？乾淨目標預測的損失是什麼？ | $z_t=(1-t)z+t\epsilon$；$t=0$ 乾淨、$t=1$ 純噪聲；$\mathcal L_{FM}=\mathbb E\|t^{-1}(\hat z-z)\|^2$，等於速度 MSE（本頁推導） | S2 |
| LO3 | 說明為什麼要分兩階段、為什麼選兩層 | 為什麼不走過編碼器的每一層？為什麼 $K=2$？加第三階段的結果？ | 每層都走會大幅增加取樣成本並累積平均導引漂移；Fig. 3 相似度以較早與較晚兩個區段為主，每區段取一個代表；加第 16 層的第三階段 FID 1.96 → 2.51 | S3 |
| LO4 | 寫出 Prop. 1 | 最佳乾淨目標預測器與其 Jacobian 是什麼？ | $a^*=\mu(u)+\mathbb E[r\mid a_t,u]=\mu(u)+\frac{r_t+t^2s_t}{1-t}$；$\nabla a^*=\frac{1-t}{t^2}\mathrm{Cov}(r\mid a_t,u)$，期望跡 $\frac{1-t}{t^2}V(a\mid a_t,u)$（Eq. 9–10） | S4a |
| LO5 | 區分 Dirac 目標與局部擾動的行為 | 直接由類別生成淺層特徵時，$t\to0$ 發生什麼？以深層特徵為條件時呢？ | 乾淨目標是 Dirac 點，相鄰模式分界處 $\mathrm{Cov}$ 不隨 $t$ 縮小，最大 Jacobian $\approx\frac{1-t}{t^2}\cdot\frac{\Delta^2}4\to\infty$，$\Delta$ 為模式間距（論文寫成 $\propto\frac{1-t}{t^2}V(a\mid y)$，為模式分離時的漸近敘述）；以深層特徵為條件時，論文假設殘差是局部擾動，$\mathrm{Cov}=O(t^2)$，Jacobian 有界（論文寫 <1，本頁算出最大約 1.2） | S4b |
| LO6 | 讀出 (E1) 的量測 | $h_7$ 由 $h_{23}$ 與由類別預測時的 $R^2$？由此得到的變異界？ | 57.4% 對 4.2%；$V(h_7\mid h_{23})\le0.426\,V$，$V(h_7\mid y)=0.958\,V$（Fig. 7b） | S5 |
| LO7 | 讀出 (E2) 的量測與淺層的權衡 | 類別解釋 $h_{23}$ 與 $h_7$ 的變異比例？淺層往深移動時 FID 與 PSNR 怎麼變？ | token 變異 18.6% 對 4.2%；mean-pooled 44.4% 對 23.6%；探針 77.7% 對 23.2%。DINOv3-S 淺層 L1→L3→L5：FID 4.36→3.14→2.69，PSNR 30.19→28.32→25.48 dB | S5 |
| LO8 | 描述 HyC-DiT 的條件注入 | 兩階段的 AdaLN 條件向量各怎麼組成？哪些部分共享？ | $m^L=T(t)+E_y(y)+E_s(1)$，所有 token 同一組；$m^M_i=P_{\text{cond}}(c^M_i)+T(t+1)+E_\sigma(\sigma)+E_s(2)$，逐 token；$T(t+1)$ 使兩階段落在時間嵌入的不相交範圍（論文原文；$T$ 的形式未給）；共享 attention、FFN、AdaLN 調變 MLP，輸入／輸出投影與條件映射為階段專屬（Eq. 5） | S6 |
| LO9 | 說明條件噪聲 | 為什麼要對 Stage 2 的條件加噪？怎麼加？代價與好處各是什麼？ | 訓練用真實影像的深層表徵、取樣用 Stage 1 生成的；訓練時 $\tilde z^L=\sqrt{1-\sigma}z^L+\sqrt\sigma\epsilon_c$，$\sigma\sim U[\sigma_{\min},\sigma_{\max}]$ 並輸入模型（Eq. 4；取樣時是否加噪論文未說）；代價是條件與輸出的對應變弱，好處是緩和 Stage 1 誤差造成的平均導引漂移：$\sigma=0$ 時 FID 1.96 → 2.10（Table 4c） | S6 |
| LO10 | 說明乾淨目標預測與共享 | Fig. 5 中兩種預測目標在分開與共享時的 FID？作者的解釋？ | x-pred 1.92（分開，1.361B）→ 1.96（共享，690M）；v-pred 2.07 → 4.25。乾淨目標輸出落在條件平均附近，速度把階段專屬殘差與高維噪聲耦合，在共享骨幹中干擾 | S7 |
| LO11 | 說明原生匹配 | 不做 NM 的一步輸出為何失敗？NM 匹配什麼、結果如何？ | $t=1$ 時輸入 $\epsilon$ 與目標獨立，MSE 最佳預測 $\mathbb E[z\mid c]$ 對所有 $\epsilon$ 相同，所有樣本輸出同一個條件平均（本頁推導，與 FID 約 280 一致，論文未給原因）；NM 在生成的表徵上直接以 drifting 統計與 IMQ-MMD 匹配真實表徵分佈，不需解碼再編碼（Eq. 8）；RAE 284.19 → 3.46（1 NFE），RevFlow 281.78 → 2.85（2 NFE），所測的 VAE latent 對照 264.56 → 291.68（沒有受益） | S8 |
| LO12 | 讀出主要結果 | 5 epoch 的 FID？重建 PSNR 與解碼器大小？只有深層、只有淺層的消融？ | XL（690M）1.96、H（1.55B）1.74（無 CFG、有 REPA guidance）；DINOv3-L 第 7 層、89M 解碼器 30.25 dB，RAE（DINOv2-B 最後一層）415M 解碼器 18.09 dB；同一 89M 解碼器只有深層 2.65／19.87 dB，只有淺層 15.94／30.25 dB | S9 |

---

## 3. 頁面大綱

單欄閱讀（文字欄 680–720px），元件放加寬圖版（≤1080px）。章節編號以 JetBrains Mono 顯示。視覺記號：一條由左到右的「層軸」（像素 → 第 7 層 → 第 23 層 → 類別），編碼方向灰色箭頭向右、生成方向 `--accent` 箭頭向左；在 hero、S3 靜態圖、S5 長條圖重複出現。章節標題用陳述句。

| # | id | 標題 | 內容與順序（符號先定義再使用；概念先於元件） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈生成即反向編碼：沿凍結編碼器由深到淺生成影像〉 | 副標：「RevFlow 先生成編碼器的深層表徵，再生成淺層表徵，最後用小解碼器回到像素。」層軸（見下）；三張事實卡；連結 arXiv、`../index.html#paper-606`、`revflow-lineage-overview.html` | 1 螢幕 |
| S1 | `background` | 01 · 預訓練編碼器把影像逐層變成抽象表徵 | 先定義 $x$、$y$、層編號 $\ell$、$E_\ell$、$h_\ell=E_\ell(x)$；淺層保留外觀、深層強調語意；生成是欠定的反問題；RAE 生成最後一層並由 415M 解碼器回到像素（DINOv2-B 最後一層 18.09 dB，聖誕帽變紅；同編碼器同解碼器對照見 S9 的 19.87 對 30.25）；本文的提問。延伸：`revflow-generation-target-space.html`、`revflow-rae-decoder-burden.html`、`revflow-representation-alignment-lineage.html` | 350–450 字 |
| S2 | `flow` | 02 · Flow matching 用乾淨目標預測訓練 **開頭**為可收合符號表（預設展開）；Eq. 2（**時間方向 callout**）、Eq. 3 的一般形（$f_\theta(z_t,t,c)$）、$\mathcal L_{FM}$ 與速度 MSE 等價〔本頁推導〕、Euler 取樣。延伸：`revflow-flow-matching-dit-backbone.html`、`flow-matching-velocity-target.html` | 350–450 字 |
| S3 | `path` | 03 · 反向編碼路徑是 y → z^L → z^M → x | Eq. 1；Fig. 3 的相似度區段與 $K=2$；不走每一層的兩個理由；第 7、23 層（層號從 0、最終 norm 前）；$P_{\text{target}}$ 1024→128、16×16；12-block 解碼器；重建先訓練後固定；第三階段 2.51。靜態 SVG（見下） | 400–500 字 + 圖 |
| S4a | `theory` | 04a · 最佳預測器等於條件平均加上殘差的後驗平均 | 先備段（字面定義：條件平均、條件共變異數、score、Jacobian 與其最大值、Dirac 點）→ Def. 1（$a,u,\mu,r,V,R^2$ 上界）與假設（$\mathbb E\|a\|^2<\infty$、$\epsilon$ 與 $(a,u)$ 獨立、$t\in(0,1)$、Jacobian 對 $a_t$ 取）→ Prop. 1（Eq. 9–10）的敘述；證明概要放一般 `<details class="proof">`（不是 `class="check"`；`extract_page_text.py` 只以 regex 移除 `<details class="check…">`，所以 `.proof` 會保留在驗收文字中。LO4 的答案仍寫在 `<details>` 之外的正文） | 350–450 字 |
| S4b | `stiff` | 04b · Dirac 目標讓最大 Jacobian 隨 t⁻² 增加 | 情形 1（Dirac，模式間距決定最大值）、情形 2（**論文的建模假設**：局部擾動）、〔本頁推導〕高斯閉式（最大 1.20，論文寫 <1）、期望跡趨近 0 而論證靠最大值 → C1 | 350–450 字 + C1 |
| S5 | `evidence` | 05 · DINOv3-L 的層統計支持先生成深層 | 量測方法；(E1)、(E2) 數字；靜態長條圖（見下）；只有淺層 15.94；像素下界 $V(x\mid h_\ell)$；淺層位置的權衡（L1/L3/L5）；漂移。延伸：`revflow-feature-inversion-multistage.html` | 450–550 字 + 圖 |
| S6 | `hyc` | 06 · HyC-DiT 以同一骨幹處理全域與空間條件 | 共享與專屬部件；Eq. 5 每個符號；adaLN-Zero 的 $\beta,\gamma,\alpha$；Eq. 4、理由、代價與好處、$\sigma=0$ 2.10、取樣時是否加噪未說；注入位置消融（2.02、1.96、2.03）；C2（教 LO8、LO9） | 450–550 字 + C2 |
| S7 | `train` | 07 · 乾淨目標預測讓兩階段共享權重 | Fig. 5 四個數字（小長條圖，數字直接標）與解釋；Eq. 6；輔助頭與 REPA guidance；取樣流程（每階段 25 Euler 步、AutoGuidance）。延伸：`revflow-flow-matching-dit-backbone.html` | 300–400 字 |
| S8 | `nm` | 08 · 原生匹配讓每個階段一步生成 依序在正文寫出（C3 之前）：Eq. 7；「一步輸出等於條件平均」〔本頁推導〕；表徵空間匹配與解碼–再編碼的差別；MMD 的定義與 IMQ 核公式（1.4 步驟 8）；Eq. 8 各項；然後 C3；Table 3 全表；VAE 對照（保留語氣）。延伸：`revflow-few-step-native-matching.html`、`rae-distillation-drifting.html` | 450–550 字 + C3 + 表 |
| S9 | `results` | 09 · 五個 epoch 達到 FID 1.96，重建 30.25 dB | 訓練設定；Table 1 摘錄（1.4 步驟 9 表）；11 倍計算量；Table 2；Table 4（全表）；淺層掃描；T2I（Table 5、35k 數字）。延伸：`latent-diffusion-rae-overview.html`、`representation-autoencoder-rae.html` | 400–500 字 + 2 表 |
| S10 | `impl` | 10 · 實作設定與常見錯誤 | 1.4 步驟 10 的 9 條 | 300–400 字 |
| S11 | `history` | 11 · 研究脈絡 | 時間線（6.2 的 `TL_GROUP`）+ 一段把時間線接到層軸：DDPM／FM／DiT（骨幹）→ LDM、VA-VAE、REPA（目標空間與對齊）→ RAE、RAEv2（直接生成表徵）→ Latent Forcing（在像素前先對特徵去噪的多階段生成，照 timeline.json 的 role）→ FD-loss、MeanFlow（少步）→ 本文 | 時間線 + 100–150 字 |
| S12 | `next` | 12 · 結論與延伸閱讀 | Sec. 5 結論；「本頁整理」的權衡清單（論文無限制章節）；筆記清單；arXiv、GitHub | 200 字 + 清單 |

Hero 事實卡（不用未定義術語）：
1. 「影像生成被寫成編碼的反向：先生成凍結編碼器第 23 層的表徵，再生成第 7 層，最後解碼成像素。」
2. 「兩個階段共用一個 Transformer；網路直接預測去除噪聲後的乾淨表徵（而不是預測速度），讓共用權重不降低品質。」
3. 「RevFlow-XL（690M 參數）在 ImageNet 256 上訓練 5 個 epoch 達到 FID 1.96（Fréchet Inception Distance，越低表示生成影像的分佈越接近真實影像）；由第 7 層解碼的 89M 解碼器重建 30.25 dB，同一解碼器由第 23 層解碼只有 19.87 dB。」

Hero 層軸：四個節點「像素」「第 7 層（淺層）」「第 23 層（深層）」「類別」（S0 不用 $z^L$、$z^M$、$x$、$y$ 等符號，它們在 S1–S2 才定義）等距排列；上方灰色細箭頭標「編碼」（向右），下方 `--accent` 粗箭頭標「生成」（向左，分三段標 Stage 1、Stage 2、解碼）。< 480px 改為直立排列（由上到下：類別 → 第 23 層 → 第 7 層 → 像素），箭頭方向同義。HTML + CSS 實作，不用 SVG 文字縮放。

S3 靜態 SVG（viewBox 0 0 720 260）：左側編碼器 24 層以細長條表示，第 7、23 層加粗並標名；淺層 16×16×1024 經 $P_{\text{target}}$ 變成 16×16×128；右側三個方框「Stage 1：$f_\theta(\cdot,t,y)$」「Stage 2：$f_\theta(\cdot,t,\tilde z^L)$」「解碼器 89M」，以 `--accent` 箭頭串接；兩個 Stage 方框以同一底色並標「共用骨幹」。圖下說明一行。窄螢幕改為直立版（同一 SVG 的第二個 viewBox 版本，以 CSS `display` 切換）。

S5 靜態長條圖：水平條，0–100%，四組，每組兩條並在條尾直接標數字：「$h_7$ 由 $h_{23}$ 解釋 57.4%／由類別 4.2%」「類別解釋 token 變異：$h_{23}$ 18.6%／$h_7$ 4.2%」「類別解釋 mean-pooled 變異：44.4%／23.6%」「線性探針準確率：77.7%／23.2%」。條以實心／斜線填充區分兩者（非只靠色）。

S7 小長條圖：四條（x-pred 分開 1.92、x-pred 共享 1.96、v-pred 分開 2.07、v-pred 共享 4.25），條尾標數字，共享以斜線填充。

---

## 4. 互動元件規格

共用：PRNG `mulberry32(seed)`（與參考頁相同的一行實作）；`gauss(r)` 為 Box–Muller 只取 cos 分支 $\sqrt{-2\ln(1-u_1)}\cos(2\pi u_2)$。Python 對照在 `verify/common.py`（已逐位元對照 JS 寫法）。

共用排程：所有超過約 1e5 次運算的工作用同一個時間切片迴圈 `slice(work, onDone)`：每次 `requestAnimationFrame` 執行工作單元直到用掉 12ms，再讓出主執行緒；reduced-motion 只改變「中途不重畫、完成後一次畫」，不改變切片。

共用 live region：每個元件一個 `<p class="sr" aria-live="polite" id="cN-live">`，只透過參考頁的 `announce(id, text)`（500ms debounce）更新；各元件的模板文字寫在下方。

SVG 文字一律用 Unicode（例：「t」「J_max」→「最大 Jacobian」「σ」「γ」「MMD²」），不放 KaTeX；需要 TeX 的軸標改用 SVG 外的 HTML 標題列。動態讀數若含 TeX，寫入後呼叫 `renderMath(node)`。

### C1 · Prop. 1：最佳預測器的最大 Jacobian（S4b）

- **教的概念**：Eq. 9–10；Dirac 目標在 $t\to0$ 的最大 Jacobian 隨 $t^{-2}$ 增加、由模式間距決定，期望跡反而趨近 0；在論文的局部擾動**假設**下 Jacobian 有界；殘差變異 $v$ 越小，最大 Jacobian 越小（為什麼要先有平均導引）。
- **簡化（頁面寫出）**：一維目標、條件固定；平均 $\mu(u)$ 只平移，圖中取 $\mu=0$，橫軸即 $r_t$。〔玩具〕Dirac 模型用 $K$ 個等距點：$q_k=k-\frac{K-1}2$，除以其標準差後乘 $\sqrt v$（零平均、變異數 $v$）。真實資料每類有上千個點，$K$ 只示意。
- **輸入**：
  - 預設 chip（同時設定模型與 $v$）：「直接 y → h₇：Dirac，v = 0.958」（預設）、「h₂₃ → h₇：論文假設的局部擾動，v = 0.426」、「h₂₃ → h₇ 若仍是 Dirac：v = 0.426」。chip 下方固定說明：「0.958 是類別平均預測 $h_7$ 剩下的變異比例（類別平均是由 y 預測的最佳值，所以是實際值）；0.426 是由 $h_{23}$ **線性**預測 $h_7$ 剩下的比例，只是上界。兩者都在 1024 維的 $h_7$ 上量，不是 128 維的 $z^M$（Fig. 7b）。把它們當成一維殘差變異是本頁的玩具對應；局部擾動是論文的建模假設，這個 chip 不是『$h_{23}\to h_7$ 的殘差真的不是 Dirac』的證據。」
  - 分段按鈕「殘差模型」：Dirac 點／局部高斯。
  - 滑桿 $v\in[0.01,1]$，步長 0.001，預設 0.958。
  - 分段按鈕 $K\in\{4,8,16\}$（只在 Dirac），預設 8。
  - 滑桿 $t\in[0.01,0.99]$，步長 0.01，預設 0.10。
- **計算（頁面即時）**：
  - 後驗（Dirac）：$\ell_k=-\frac{(r_t-(1-t)r_k)^2}{2t^2}$，log-sum-exp 正規化得 $w_k$；$\mathbb E[r\mid r_t]=\sum w_kr_k$；$\mathrm{Cov}=\max(0,\sum w_kr_k^2-(\sum w_kr_k)^2)$。
  - 高斯：$\mathbb E[r\mid r_t]=\frac{(1-t)v}{(1-t)^2v+t^2}r_t$，$\mathrm{Cov}=\frac{vt^2}{(1-t)^2v+t^2}$。
  - $J_{\max}(t)=\frac{1-t}{t^2}\max_{r_t}\mathrm{Cov}$：在 $[(1-t)r_{\min}-4t,(1-t)r_{\max}+4t]$ 取 801 個等距點，**另加 $K-1$ 個中點** $(1-t)(r_k+r_{k+1})/2$（小 $t$ 時峰寬約 $t^2/((1-t)\Delta)$，等距網格會漏掉；`c1.py` 以每個中點附近 2 萬點細搜確認與此法一致）。
  - $\overline J(t)=\mathbb E\,\mathrm{tr}\nabla a^*=\frac{1-t}{t^2}\cdot\frac1K\sum_k\frac{\sum_n\phi(\xi_n)\mathrm{Cov}((1-t)r_k+t\xi_n)}{\sum_n\phi(\xi_n)}$，$\xi_n$ 為 $[-8,8]$ 的 801 個等距點、$\phi(\xi)=e^{-\xi^2/2}$。高斯時 $J_{\max}=\overline J=\frac{(1-t)v}{(1-t)^2v+t^2}$。
  - 也顯示 $V(a\mid a_t,u)/V(a\mid u)=\overline J\cdot\frac{t^2}{(1-t)v}$。
  - 右圖的 $t$ 掃描：60 個對數等距點 $t\in[0.01,0.99]$；只在模型、$v$、$K$ 改變時重算（150ms debounce 後啟動）。**以時間切片執行**，工作單元為「一個 $t$ 點」（$K=16$ 時約 2e5 次 exp，約 2–4ms）；重算期間右圖保留舊曲線並設 `opacity:.35`、標題旁顯示「計算中」，完成後換成新曲線；新的設定變更會取消進行中的掃描（以遞增的 job id 判斷）。拖動 $t$ 只移動標記並同步重算該點讀數（一個點，不切片）。結果以 key `${model}|${v}|${K}` 快取。
- **數值保護**：$t$ 限在 $[0.01,0.99]$（$t=0$ 時 Eq. 10 除以 0、$t=1$ 時 $a_t$ 不含資訊）；log-sum-exp 防止 $t$ 小時 exp 下溢成 0/0；$\mathrm{Cov}$ 取 $\max(0,\cdot)$ 防止相減誤差變負；右圖縱軸對數 $[10^{-3},10^{4}]$，數值 $<10^{-3}$ 畫在下緣並在讀數顯示「< 0.001」。
- **顯示**：
  - 左圖「t 時的最佳預測 E[r | r_t]」：橫軸 $r_t\in[-3.2,3.2]$，縱軸 $[-1.8,1.8]$；曲線為 `--accent` 實線；Dirac 點以縱軸上的菱形標記；參考線 $r_t/(1-t)$（「完全相信輸入」）`--ink-3` 點線。
  - 右圖「Jacobian 對 t」：橫軸 $t$（對數），三條：$J_{\max}$（目前模型，`--accent` 實線，標「最大」）、$\overline J$（目前模型，`--c-par` 虛線，標「期望跡」）、同 $v$ 的高斯參考（`--c-perp` 點線，標「局部擾動（假設）」）；目前 $t$ 以垂直細線標示。
  - 讀數（JetBrains Mono，4 位有效數字）：$t$、$J_{\max}$、$\overline J$、$V(a\mid a_t,u)/V(a\mid u)$；Dirac 時另顯示閉式近似 $\frac{1-t}{t^2}\frac{\Delta^2}4$（$\Delta=\sqrt{v}\sqrt{12/(K^2-1)}$）以對照。
  - 圖版底部：「讀數由本頁依 Eq. 9–10 即時計算。局部擾動是論文的建模假設；高斯閉式、最大與期望之分為本頁推導。」
- **讀者應觀察到**：
  1. 「直接 y → h₇」：$t$ 由 0.5 降到 0.02，左圖由平滑 S 形變成「大部分區域平坦、分界處陡升」的形狀，$J_{\max}$ 由 1.307 升到 111.8，$\overline J$ 降到 0。
  2. 「局部擾動 v = 0.426」：任何 $t$ 的 Jacobian 都在 1.1 以內（論文寫 <1，本頁算出最大 1.097），左圖為直線。
  3. Dirac 且 $v=0.426$：仍會發散；$t\to0$ 時 $J_{\max}$ 約為 $v=0.958$ 的 0.445 倍（$=0.426/0.958$，因 $\Delta^2\propto v$），$t=0.10$ 時為 0.45（1.846／4.106）。這是兩個**界**的比值，不是實測的 Jacobian 比值。
  4. 同 $v=0.958$、$t=0.05$，$K=4,8,16$ 的 $J_{\max}$ 為 72.8、17.3、4.28：最大值由模式間距決定。
  5. 高 $t$（0.9）時三種設定的 $\mathbb E[r\mid r_t]$ 都接近 0：細節被平均掉。
- **無障礙**：曲線以線型＋直接標籤區分；滑桿 `aria-valuetext`（「t = 0.10」「v = 0.958」）；SVG `role="img"` 與描述目前設定的 `aria-label`；沒有 hover 專屬資訊。live 模板：「{模型}，v = {v}，t = {t}：最大 Jacobian {Jmax}，期望跡 {Jbar}。」掃描完成時另宣告「已更新 Jacobian 曲線」。

### C2 · HyC-DiT 的條件路徑：全域與逐 token 調變（S6，教 LO8、LO9）

- **教的概念**：Eq. 5（Stage 1 所有 token 同一組調變；Stage 2 每個 token 由對應的深層 token 決定）；Eq. 4 條件噪聲的代價（深層 token 與調變的對應變弱）；好處以論文數字呈現（$\sigma=0$ 時 FID 2.10 對 1.96，Table 4c），玩具不模擬好處。
- **簡化〔玩具〕（頁面寫出）**：8×8 token（真實為 16×16）、寬度 8；所有權重為固定 seed 的隨機值，只示意結構，**數值與相關係數都來自隨機權重，不代表訓練後的模型**；只畫 scale $\gamma$（shift、gate 的計算方式相同）；真實 adaLN-Zero 的輸出層初始化為 0，玩具略過；論文只在訓練時擾動條件，玩具的 $\sigma$ 滑桿代表訓練時的一個取樣值。
- **玩具定義**（與 `c2.py` 相同）：
  - token 索引 $i=8r+c$，列 $r$ 與行 $c$ 皆 0-based（0..7）。深層條件 $c_i=(c_{i1},c_{i2})$：$c_{i1}$ 為物體遮罩（$(c-4.5)^2+(r-3)^2\le2.2^2$ 為 1，否則 0；中心刻意偏離格中心 (3.5, 3.5)，讓圖樣不對稱；共 16 個 token：(1,4)(1,5)(2,3)–(2,6)(3,3)–(3,6)(4,3)–(4,6)(5,4)(5,5)）標準化為零平均單位變異；$c_{i2}$ 為行號 $c$ 標準化。
  - 權重：`mulberry32(11)` 依序抽 $P_{\text{cond}}$（8×2，×0.8）、$E_y$（3×8，×0.5）、$E_s$（2×8，×0.5）、$e_\sigma$（8，×0.5）、$w_\gamma$（8，$/\sqrt8$），皆逐列逐元素以 `gauss` 抽取。陣列 `ES[0]` 對應 Eq. 5 的 $E_s(1)$、`ES[1]` 對應 $E_s(2)$（程式 0-based，論文以階段編號 1、2）。
  - $T(t)=[\cos\omega_jt,\sin\omega_jt]_{j=0..3}$，$\omega_j=\frac\pi2 2^j$。〔推定〕論文未給 $T$ 的形式。注意：此選擇下 $j=2,3$ 的座標週期為 1 與 0.5，$T_j(t+1)=T_j(t)$，$j=2,3$ 無法分開兩階段；$j=0$（Stage 1 走第一象限、Stage 2 走第二象限）與 $j=1$（$\omega_1=\pi$：$\sin$ 在 Stage 1 為非負、Stage 2 為非正，上下半圓）可以分開。頁面只畫 $j=0$ 這一對座標並標〔推定〕，不寫「整個嵌入不相交」；論文對 $T(t+1)$ 的敘述照錄原文。
  - Stage 1：$m=T(t)+E_y[y]+E_s[0]$，$\gamma=w_\gamma\cdot\mathrm{SiLU}(m)$，所有 token 相同。
  - Stage 2：$\epsilon_{c,i}$ 由 `mulberry32(seed)`（初始 12）逐 token 抽兩個；$\tilde c_i=\sqrt{1-\sigma}c_i+\sqrt\sigma\epsilon_{c,i}$；$m_i=P_{\text{cond}}\tilde c_i+T(t+1)+\sigma e_\sigma+E_s[1]$；$\gamma_i=w_\gamma\cdot\mathrm{SiLU}(m_i)$。〔玩具〕$E_\sigma(\sigma)=\sigma e_\sigma$。
- **輸入**：分段按鈕「階段」1／2（預設 2）；分段按鈕「類別 y」0／1／2（只在 Stage 1，預設 0）；滑桿 $t\in[0,1]$ 步長 0.01 預設 0.5；滑桿 $\sigma\in[0,1]$ 步長 0.01 預設 0.2（只在 Stage 2；旁註「論文的 $\sigma_{\min},\sigma_{\max}$ 未給，本滑桿開放整個 [0, 1]」）；按鈕「重新抽 $\epsilon_c$」（seed +1）；勾選「在格中顯示數值」（預設關）。
- **顯示**：
  - 左：輸入條件格（Stage 2 顯示 $\tilde c_{i1}$；Stage 1 顯示單一類別標籤方塊並註「沒有空間條件」）。
  - 中：$\gamma_i$ 熱圖。色階為色盲友善的發散色階（藍–白–橙，端點以 `--c-par`／`--accent` 為基準，light 與 dark 各一組，中點分別為 `--surface`），附水平圖例（最小值、0、最大值）。數值關閉時每格顯示「+」或「−」記號（$|\gamma_i|<0.05$ 顯示「0」）；數值開啟時顯示 2 位小數。遮罩內的 token 加粗框。
  - 右：時間嵌入第一對座標的半圓圖，Stage 1 段（第一象限弧）實線、Stage 2 段（第二象限弧）虛線，目前 $(\cos\omega_0t',\sin\omega_0t')$ 以點標示（$t'=t$ 或 $t+1$），標題標〔推定〕。
  - 讀數：$\gamma$ 的 token 間標準差、$\gamma$ 與遮罩的相關係數（Stage 1 顯示「—（所有 token 相同）」）；下方固定一行：「條件噪聲的好處無法在玩具中顯示；論文：$\sigma=0$ 時 FID 2.10，預設 1.96（Table 4c）。」
  - 熱圖為 `role="grid"`（每列 `role="row"`、每格 `role="gridcell"`），roving tabindex：方向鍵移動、Home／End 到該列首尾、Ctrl+Home／Ctrl+End 到全格首尾、Tab 離開格子。點選或 focus 某格：讀數框顯示該 token 的 $(r,c)$、$c_i$、$\tilde c_i$、$\gamma_i$（取代 hover）。
- **數值保護**：$\sigma=1$ 時 $\sqrt{1-\sigma}=0$ 正常；標準差 $<10^{-12}$ 時不算相關係數；SiLU 用 `x/(1+Math.exp(-x))`。`c2.py` 掃描全部輸入範圍（$t$、$\sigma$ 步長 0.01／0.05、$y$=0..2、seed 12..61）得 $\max|m|=6.37$，無溢位風險；selfTest 斷言預設 seed 下 $\max|m|<10$。
- **讀者應觀察到**：Stage 1 的 $\gamma$ 全格相同（標準差 0），只隨 $t$、$y$ 改變；Stage 2 的 $\gamma$ 圖樣跟著物體遮罩（$\sigma=0$ 時相關係數絕對值在 $t=0.5$ 為 0.892、$t=1$ 為 0.965，見 §5.2）；$\sigma$ 增加，相關係數絕對值下降，$\sigma=1$ 時接近 0（條件只剩噪聲）；第一對時間嵌入座標上兩段弧分開。
- **無障礙**：熱圖格 `aria-label`（「第 r 列第 c 行，γ = …，遮罩內／外」）；live 模板：「階段 {s}，t = {t}，σ = {σ}：γ 標準差 {sd}，與遮罩的相關係數 {corr}。」

### C3 · 一步輸出的條件平均與 IMQ-MMD 原生匹配（S8）

- **教的概念**：不做 NM 的一步輸出等於條件平均〔本頁推導，正文已先寫〕；以 MMD 直接比較生成集合與參考集合，梯度把樣本推開、覆蓋參考分佈；不需要配對樣本（Eq. 8 的 IMQ 項）。
- **簡化〔玩具〕（頁面寫出，圖版內一段）**：二維「表徵」、單一類別；直接最佳化 120 個生成點本身（論文最佳化的是一步生成器的參數，玩具省略生成器）；只用 IMQ-MMD，省略 drifting 統計項與 flow matching 正則；參考集固定（論文用 EMA 生成器產生加噪真實表徵的參考）。另外寫明：「初始點是平均加上 0.05 倍的擾動，這個擾動讓點之間有互斥的梯度；若生成器對所有 ε 輸出**完全相同**的點，$\sum\nabla k=0$，點不會被推開，真實的生成器必須學會把不同的 ε 對到不同的輸出。參考集三團各 80 點是人為設定，所以終點三團各約 40 點。」
- **資料**（與 `c3.py` 相同）：參考集 $R$：240 點，第 $p$ 點屬於群 $p\bmod3$（群 A 中心 $(-1.2,-0.6)$、群 B $(1.2,-0.6)$、群 C $(0,1.1)$），標準差 0.25，`mulberry32(21)` 依序抽 x、y。初始生成點：$x_i=\bar R+0.05\,\epsilon_i$（$\bar R$ 為參考集平均，`mulberry32(22)`）。
- **損失**（V 統計量，恆 ≥ 0）：$k(x,y)=\sum_{h\in H}(1+\|x-y\|^2/h^2)^{-1/2}$；
  $\mathrm{MMD}^2=\frac1{N_g^2}\sum_{i,j}k(x_i,x_j)+\frac1{N_r^2}\sum_{p,q}k(r_p,r_q)-\frac2{N_gN_r}\sum_{i,p}k(x_i,r_p)$。〔推定〕IMQ 的形式與尺度；$H$ 預設多尺度 $\{0.1,0.3,1.0\}$。
- **解析梯度**：$\nabla_xk=-\sum_h\frac{x-y}{h^2}(1+\|x-y\|^2/h^2)^{-3/2}$；$\frac{\partial\mathrm{MMD}^2}{\partial x_i}=\frac2{N_g^2}\sum_{j\ne i}\nabla_xk(x_i,x_j)-\frac2{N_gN_r}\sum_p\nabla_xk(x_i,r_p)$（`c3.py` 以有限差分核對：0.033771 對 0.033771）。
- **更新**：$x_i\leftarrow x_i-\eta N_g\,\partial\mathrm{MMD}^2/\partial x_i$，每點步長上限 0.2（超過則等比縮短）；共 200 步。迴圈順序與 Python 相同（$i$ 外層；先 $X$ 後 $R$），冪次用 `Math.pow(q,-0.5)`、`Math.pow(q,-1.5)`。$R$–$R$ 項只在初始化算一次。每步約 $120\times360\times3\approx1.3\times10^5$ 次核計算；以共用時間切片迴圈執行（工作單元＝1 步，每個切片 12ms 內盡量多步），一般模式每個切片後重畫，reduced-motion 只在完成後畫一次。
- **輸入**：分段按鈕「核尺度 H」：0.1／0.3／1.0／多尺度（預設多尺度）；分段按鈕「步長 η」：0.1／0.25／0.5（預設 0.25）；按鈕「執行／暫停」：從目前步數繼續，到第 200 步**停止**並把按鈕改為「已完成（重設後可再執行）」；按鈕「重設為一步輸出」：把生成點還原為 `mulberry32(22)` 產生的同一組初始點、步數歸 0。改變 H 或 η 時自動重設。
- **顯示**：散佈圖 $[-2.2,2.2]\times[-1.8,2.2]$：參考點依群用不同空心形狀（群 A 圓、群 B 方、群 C 菱形，`--ink-3`），三個中心以小十字並直接標「A」「B」「C」；生成點為實心三角形（`--accent`）。讀數：步數、$\mathrm{MMD}^2$（4 位小數）、最近中心為 A／B／C 的生成點數、距最近中心 < 0.5（2 個標準差）的生成點數。圖下對照文字：「論文 Table 3：RevFlow 適應前 281.78，NM 後 2.85（2 NFE）。」
- **數值保護**：$h\ge0.1$；步長上限防止小核時暴衝；若任一座標為 NaN 或 $|x|>10$，停止並顯示「數值發散，已重設」後重設。
- **讀者應觀察到**：第 0 步所有生成點集中在平均附近（$\mathrm{MMD}^2=1.8612$、A/B/C 最近點數 2/1/117、接近任一中心的點 0）；多尺度 25 步內 $\mathrm{MMD}^2$ 降到 0.003，三群各約 40 點；只用 $h=1.0$ 時前 10 步只有整體擴散（接近中心的點 0 → 31），之後才分開成三群。
- **無障礙**：形狀與文字標籤區分群與兩組點；執行中不更新 live region；live 模板（完成、暫停或重設時一次）：「第 {n} 步，MMD² = {m}，群 A / B / C 各 {a} / {b} / {c} 點。」

---

## 5. 驗收數值表（由 `verify/` 腳本以頁面相同算法產生）

### 5.1 C1（`c1.py`）

容許誤差：$J_{\max}$、$\overline J$ 相對 ±0.5%（或絕對 ±0.001，取大者）；比例 ±0.001。Eq. 9 恆等式（解析 score）在 9 個點誤差 < 1e-12。

| 模型 | v | K | t | $J_{\max}$ | $\overline J$ | $V(a\mid a_t)/V(a\mid u)$ |
| --- | --- | --- | --- | --- | --- | --- |
| Dirac | 0.958 | 8 | 0.02 | 111.8 | 0.000 | 0.0000 |
| Dirac | 0.958 | 8 | 0.05 | 17.34 | 0.002328 | 0.0000 |
| Dirac | 0.958 | 8 | 0.10（預設） | 4.106 | 0.5897 | 0.0068 |
| Dirac | 0.958 | 8 | 0.20 | 1.289 | 1.061 | 0.0554 |
| Dirac | 0.958 | 8 | 0.50 | 1.307 | 0.9543 | 0.4981 |
| Dirac | 0.958 | 8 | 0.90 | 0.1177 | 0.1169 | 0.9883 |
| Dirac | 0.426 | 8 | 0.02 | 49.70 | 0.000 | 0.0000 |
| Dirac | 0.426 | 8 | 0.10 | 1.846 | 0.9414 | 0.0246 |
| Dirac | 0.958 | 4 | 0.05 | 72.81 | 0.000 | 0.0000 |
| Dirac | 0.958 | 16 | 0.05 | 4.283 | 0.5292 | 0.0015 |
| 高斯 | 0.958 | – | 0.10 | 1.097 | 1.097 | 0.0127 |
| 高斯 | 0.426 | – | 0.02 | 1.019 | 1.019 | 0.0010 |
| 高斯 | 0.426 | – | 0.10 | 1.080 | 1.080 | 0.0282 |
| 高斯 | 0.426 | – | 0.50 | 0.5975 | 0.5975 | 0.7013 |

另：Dirac 點（K=8，v=0.958）= ±0.2136、±0.6408、±1.0679、±1.4951；閉式 $\frac{0.98}{0.02^2}\cdot\frac{\Delta^2}4=111.77$（`paper_numbers.py`）；$t=0.05$ 時 $\mathbb E[r\mid r_t]$ 在 $r_t=-0.5,-0.2,0.05$ 為 −0.6408、−0.2136、0.2135（階梯）。高斯 $\max_t J$：$v=0.958$ 1.1996、$v=0.426$ 1.0971。比值 $J_{\max}(v{=}0.426)/J_{\max}(v{=}0.958)$（Dirac，K=8）：$t=0.02$ 為 0.4445、$t=0.10$ 為 0.4496。掃描以時間切片執行不改變任何數值。

### 5.2 C2（`c2.py`；容許 ±0.0005）

| 階段 | y | t | σ | mean γ | std γ | corr(γ, 遮罩) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0.5 | – | 0.3684 | 0.0000 | – |
| 1 | 1 | 0.5 | – | 0.3408 | 0.0000 | – |
| 1 | 2 | 1.0 | – | −0.2721 | 0.0000 | – |
| 2 | – | 0.5 | 0.0 | 0.8661 | 0.9450 | −0.8921 |
| 2 | – | 0.5 | 0.2（預設） | 0.7631 | 0.9029 | −0.7463 |
| 2 | – | 0.5 | 0.5 | 0.6245 | 0.8689 | −0.5521 |
| 2 | – | 0.5 | 1.0 | 0.4723 | 0.8364 | 0.0777 |
| 2 | – | 1.0 | 0.0 | 0.7102 | 0.8421 | −0.9649 |

預設（Stage 2、t=0.5、σ=0.2、seed 12）的 $\gamma_0,\gamma_{27},\gamma_{63}$ = 1.0747、0.1930、1.2654。相關係數的正負號取決於隨機權重，頁面文字只談絕對值。遮罩 token 數 16（0-based (r,c) 見 C2 定義）；全輸入範圍 $\max|m|=6.37$（`c2.py` 的 `max_abs_m()`）。

### 5.3 C3（`c3.py`／`c3_scan.py`；$\eta=0.25$）

容許誤差：$\mathrm{MMD}^2$ 相對 ±5%（或絕對 ±2e-4，取大者）；點數 ±2（浮點加總順序差異可能讓個別點換邊）。第 0 步必須完全一致。

| H | 第 0 步 | 10 | 25 | 50 | 100 | 200 | 200 步 A/B/C 點數 | 200 步接近中心 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 多尺度（預設） | 1.8612 | 0.0150 | 0.0030 | 0.0017 | 0.0009 | 0.0007 | 39/41/40 | 107 |
| 0.1 | 0.7486 | 0.0388 | 0.0096 | 0.0040 | 0.0018 | 0.0010 | 38/39/43 | 105 |
| 0.3 | 0.7791 | 0.0236 | 0.0041 | 0.0013 | 0.0005 | 0.0002 | 39/40/41 | 107 |
| 1.0 | 0.3334 | 0.0842 | 0.0074 | 0.0013 | 0.0002 | 0.0001 | 39/41/40 | 105 |

「執行」到第 200 步停止；「重設」回到與第 0 步完全相同的點。第 0 步 A/B/C 最近點數 2/1/117、接近中心 0；$h=1.0$ 第 10 步接近中心 31。$\eta=0.5$ 多尺度在 50 步後停在約 0.01（步長造成的抖動），所以預設 0.25。

---

## 6. 實作計畫

### 6.1 沿用（不重寫）

單檔 `606/build/index.html`，從 `tutorials/prediction-targets-k-diff.html` 複製外殼：`<head>`（Google Fonts、KaTeX 0.16.9 cdnjs 與 auto-render 設定、`renderMath()`）、CSS tokens（含 `--c-par`、`--c-perp` 與深色模式覆寫、`pm-theme`）、appbar 與主題切換、`nav.toc`（桌機 sticky 左欄／窄螢幕橫列）、`#pm-bar`（`../index.html`、`index.html`）、`.plate`／`.plate-h`／`.plate-f`／`.ctl`／`.seg`／`.chip`／`.rng`／`.readout`、`.fact`、`.qa`／`details.check`（另新增 `details.proof`，樣式同 `.check` 但不被驗收腳本移除）、`.tscroll` 表格容器、`.eq`／`.eq-n` 方程式編號、callout、時間線卡片列與 `renderTimeline()`、JS 工具（`$`、`el`、`tx`、`lin`、`lgs`、`lp`、`frame`、`arrow`、`announce`、`mulberry32`）、`selfTest()` 骨架與 `?test` 觸發、reduced-motion 處理、`scrollWidth <= innerWidth` 檢查。刪除 238 專屬的內容、元件與 `TL_GROUP`。

### 6.2 新增

- 文字與表：S0–S12（第 3 節）、Table 1 摘錄、Table 3、Table 4、符號表。所有數字只能來自第 1 節。
- 靜態圖：hero 層軸（HTML/CSS，含 < 480px 直立版）、S3 管線 SVG（寬／窄兩版）、S5 長條圖、S7 Fig. 5 長條圖（SVG，數字直接標、斜線 pattern 區分共享）。
- 元件函式：
  - C1：`diracPts(K,v)`、`post(rt,t,R)` → `[e1,cov]`、`covMax(t,R)`、`covExp(t,R)`（共用預先算好的 `XI`、`PHI`）、`gaussJ(t,v)`、`initC1()`、`sweepC1()`（快取 key `${model}|${v}|${K}`）、`renderC1()`。
  - C2：`C2W`（初始化時以 seed 11 抽好的權重）、`MASK`、`CH1`、`CH2`、`temb(t)`、`gammasC2(stage,t,y,sigma,seed)`、`statsC2(g)`、`initC2()`、`renderC2()`。
  - C3：`refC3()`、`genC3()`（每次重設都以 `mulberry32(22)` 重新產生）、`mmdGrad(X,R,H)` → `{m, G}`、`stepC3()`、`initC3()`、`runC3()`（透過 `slice()`，到第 200 步停止）。
  - 共用：`slice(unit, done, {redraw})`：`requestAnimationFrame` 內以 `performance.now()` 執行 `unit()` 直到 12ms，回傳 false 時呼叫 `done()`；以 job id 取消舊工作。C1 掃描與 C3 都用它。
- 數學排版與版面（4 節開頭的規則之外）：寬公式（Eq. 5、Eq. 8、Eq. 9–10、Eq. 4）包在 `<div class="eq-scroll" tabindex="0" role="region" aria-label="公式 Eq. n，可橫向捲動">`，`overflow-x:auto`；390px 時窄版 SVG 的文字 ≥ 12px（用窄版 viewBox 而不是縮小）；斜線 pattern 的線色用 `currentColor`／CSS 變數，`--c-par`、`--c-perp`、C2 色階端點在 dark 模式各有覆寫值（沿用參考頁的 dark 值，色階另定義 `--div-neg`、`--div-pos`，light 與 dark 各一組，對 `--surface` 對比 ≥ 3:1）。
- `selfTest()` 斷言（`?test`）：Eq. 9 恆等式（3 點，1e-12）；C1 預設 $J_{\max}=4.106$、$\overline J=0.5897$（±0.5%）、Dirac v=0.958 K=8 t=0.02 $J_{\max}=111.8$、高斯 v=0.426 t=0.5 為 0.5975；C2 Stage 1 std = 0、Stage 2 預設 corr = −0.7463（±0.0005）、遮罩 token 數 = 16、預設 seed 下 Stage 2 在 $t\in\{0,0.5,1\}$、$\sigma\in\{0,0.5,1\}$ 的 $\max|m|<10$；C3 第 0 步 $\mathrm{MMD}^2=1.8612$（±1e-4）、多尺度 25 步 0.0030（±5%）且 A/B/C 各 ≥ 35 點、重設後第 0 步 $\mathrm{MMD}^2$ 再次為 1.8612；`scrollWidth <= innerWidth`。輸出一行 `selfTest: N passed, M failed`。
- 時間線：`notes/timeline.json` 原樣貼進 `<script type="application/json" id="timeline-data">`（14 筆）。`TL_GROUP`：

  | 類別 | arXiv | 卡片連到的筆記 |
  | --- | --- | --- |
  | 骨幹與訓練 | 2006.11239、2210.02747、2212.09748 | `revflow-flow-matching-dit-backbone.html` |
  | 目標空間 | 2112.10752 | `revflow-generation-target-space.html` |
  | 表徵對齊 | 2410.06940、2501.01423 | `revflow-representation-alignment-lineage.html` |
  | 編碼器 | 2304.07193、2508.10104 | `revflow-lineage-overview.html` |
  | 生成表徵（RAE） | 2510.11690、2605.18324 | `revflow-rae-decoder-burden.html` |
  | 多階段 | 2602.11401 | `revflow-feature-inversion-multistage.html` |
  | 少步與匹配 | 2505.13447、2604.28190 | `revflow-few-step-native-matching.html` |
  | 本文 | 2611.34782 | `revflow-lineage-overview.html` |

- S12 延伸閱讀（依總覽閱讀順序）：`revflow-lineage-overview`、`revflow-flow-matching-dit-backbone`、`revflow-generation-target-space`、`revflow-representation-alignment-lineage`、`revflow-rae-decoder-burden`、`revflow-feature-inversion-multistage`、`revflow-few-step-native-matching`；站內既有 `latent-diffusion-rae-overview`、`representation-autoencoder-rae`、`rae-followups`、`rae-distillation-drifting`、`flow-matching-velocity-target`（皆 `<slug>.html`，附 frontmatter `title`）。
- 連結：arXiv `https://arxiv.org/abs/2611.34782`；關聯圖 `../index.html#paper-606`；GitHub `https://github.com/ChengShiest/RevFlow`。

### 6.3 文案與標記

- 繁體中文、台灣用語、字面學術語氣；不用比喻與反問。用詞：「表徵」（representation）、「條件平均」、「殘差」、「乾淨目標預測」、「原生匹配」、「常見錯誤」。專有名詞第一次出現附英文。
- 〔本頁推導〕、〔推定〕、〔玩具〕在頁面以 `--ink-3` 邊框 pill 標示並附一行依據；區分「論文結果」「作者的解釋」「本頁推導」。
- 每個 LO 的參考答案寫進對應章節正文；建構完成後跑 `extract_page_text.py` 自檢 12 題。

### 6.4 建置後自我檢查

`?test` 全數通過；impeccable 0 項；截圖只看新元件、靜態圖與 390px（沿用外殼，不重跑完整審查迴圈）；確認 hero 層軸窄版與 S3 SVG 窄版不重疊、無水平捲動。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 時間方向與站內其他教學相反 | 本文 $t=0$ 乾淨；`prediction-targets-k-diff` 為 $t=1$ 資料 | S2 加 callout；常見錯誤第 1 條 |
| R2 | 「Jacobian < 1」的嚴格性 | 〔本頁推導〕一維高斯殘差的 $\nabla a^*$ 在 $t$ 小時略大於 1（v=0.958 最大 1.200） | 頁面正文寫論文原文「<1」並緊接寫「本頁計算：一維高斯情形最大 1.20（v = 0.958）、1.10（v = 0.426），有界」；並標明局部擾動本身是論文的建模假設 |
| R3 | 一步輸出等於條件平均是本頁推導 | 論文只給 Table 3 數字，未寫原因；只在 $t=1$ 精確成立 | S8 標〔本頁推導〕，寫「一致，但論文未給原因」；LO11 答案註明 |
| R4 | 玩具與論文數字的對應 | C1 把 Fig. 7b 的 0.958、0.426 當一維殘差變異；$K$ 個 Dirac 點、C2 的權重與時間頻率、C3 的 IMQ 形式與尺度皆非論文設定 | 元件內逐項標〔玩具〕或〔推定〕，圖版底部說明 |
| R5 | 未給的超參數 | 見 1.4 步驟 10 第 9 條 | 照寫「未給」，不補值 |
| R6 | T2I：正文差值與 Table 5 不一致 | 正文 35k 的 77.54% − 3.65 = 73.89%（Table 5 RAE 0.74，四捨五入相容）；81.55 − 4.04 = 77.51（Table 5 RAE 77.6，差 0.09，捨入無法解釋） | **頁面以 Table 5 為主**（表格列 RAE 0.74／77.6、RevFlow 0.78／81.6）；引用正文的 35k 數字與差值時，緊接一句：「正文的差值換算出 RAE 為 73.89% 與 77.51，與 Table 5 的 0.74、77.6 分別相差 0.11 與 0.09 個百分點／分；DPG 的差異無法以四捨五入解釋，本頁照錄兩處原文。」不並列而不說明 |
| R7 | Eq. 8 的括號位置 | 抽出的文字中括號字形不完整；本文件採 $\mathbb E_y[\lambda_s\mathcal L_{\text{stat}}+\sum_\ell\lambda_\ell\mathcal L_{\text{IMQ}}]+\lambda_{FM}\mathcal L$（與「每個類別內匹配」及「FM 為正則」的敘述一致） | 已由協調者對照 PDF p.6 確認正確（round 1） |
| R8 | Fig. 3、Fig. 7 的逐層數據不可得 | 只有文字給出的端點（96.2%、57.4%、0.6%、15–22%） | 不重繪曲線，只畫文字給出的數字 |
| R9 | C1 計算量 | $K=16$ 掃描約 1.2e7 次 exp | 只在模型／v／K 改變時重算並快取；150ms debounce 後以 12ms 時間切片執行，重算中顯示淡化的舊曲線 |
| R10 | C3 浮點路徑依賴 | 粒子動態對加總順序敏感 | 固定迴圈順序與 `Math.pow`；容許誤差放寬到 ±5%、點數 ±2；第 0 步要求完全一致 |
| R11 | 筆記與時間線可能更新 | 平行撰寫中，建構前 notes 可能增減 | 建構時重讀 `notes/`；未列入 `TL_GROUP` 的 arXiv 歸「其他」並回報 |
| R12 | 相對連結 | 建置位置與上線位置不同 | 只在上線位置驗證站內連結 |
| R13 | 取樣時的條件噪聲 | 論文只說訓練時擾動 Stage 2 條件並把 σ 輸入模型，沒有說取樣時的 σ | 頁面寫「論文未說明取樣時是否加噪」；C2 的 σ 滑桿標為訓練時的取樣值 |
| R14 | C2 遮罩中心（round 1 第 13 項的部分保留） | 審查問中心是否應為 (3.5, 3.5)；本設計**保留** (4.5, 3)：刻意偏離格中心，讓 $\gamma$ 圖樣與遮罩的對應不因對稱而看不出方向。已寫明 0-based、token 清單，並在 selfTest 斷言 16 個 | 保留，理由如左 |
| R15 | C1 計算量的處理（round 1 第 12 項的選擇） | 審查提供「時間切片」或「降到 40 點／401 格」兩案；本設計選時間切片，保留 60 點／801 格，使 §5.1 的數值不變 | 採時間切片 |

---

## 8. 修訂紀錄（round 1，對應 `reviews/round1.md`）

| 項目 | 處理 |
| --- | --- |
| 1 | C3 η 改為分段按鈕；重設以 `mulberry32(22)` 還原；「執行」到 200 步停止 |
| 2 | 共用 `slice()` 12ms 時間切片；reduced-motion 只延後重畫 |
| 3 | 步驟 4 情形 2、LO5、C1 chip 說明與圖版底部標「論文的建模假設」；1.20 在正文如實寫出（R2） |
| 4 | 步驟 4 新增「最大 Jacobian 由模式間距決定」與 K=4/8/16 數字；LO5 改寫；C1 觀察 4 |
| 5 | Table 1 欄名改「無 CFG」，註明 RevFlow 列含 REPA guidance；1.96 對 2.16 加註 |
| 6 | 1.1、步驟 1、hero 卡 3、S1、LO12 寫明編碼器與層，並引用 19.87 dB |
| 7 | C2 時間嵌入只畫第一對座標並標〔推定〕，寫明 $j=2,3$ 不分開；C2 標 LO8、LO9；LO9 題目改問代價與好處 |
| 8 | 步驟 6 與 LO9 補好處（2.10 對 1.96）與「只在訓練時」；C2 註明相關係數來自隨機權重（R13） |
| 9 | S4 拆為 S4a／S4b；先備段字面定義；Def. 1 假設寫入步驟 4；證明概要放 `details.proof` |
| 10 | 步驟 8 新增 MMD 與 IMQ 的正文定義；S8 大綱規定在 C3 之前寫出 |
| 11 | 符號表移到 S2 開頭；S1 先定義 $x,y,\ell,E_\ell,h_\ell$；符號表補 score／Jacobian／Dirac |
| 12 | C1 掃描改時間切片＋淡化舊曲線（R15）；觀察 3 改寫 0.445／0.45 並說明是界的比值、量在 1024 維 $h_7$ |
| 13 | 刪除未驗證宣稱，改為 `max_abs_m()` 全範圍 6.37 並加 selfTest；mask 0-based 與 token 清單；`ES[0]`↔$E_s(1)$；中心保留（R14） |
| 14 | C2 熱圖 `role="grid"`、色盲友善色階＋圖例、+/− 記號；每元件一個 `announce()` 與模板；C3 三群以形狀與 A/B/C 標籤區分 |
| 15 | SVG 只用 Unicode、動態讀數呼叫 `renderMath`；寬公式 `eq-scroll` 容器；窄版 SVG ≥ 12px；hatch 與資料色的 dark 值 |
| 16 | 「剛性」改為字面描述（最大 Jacobian 隨 t⁻² 增加）；「崩塌」→「輸出相同的條件平均」；「擠」→「集中在」 |
| 17 | 一步輸出推導改用後驗平均形式，註明只在 t=1 精確、Eq. 9 在 t=1 為 0/0；「相符」→「一致，但論文未給原因」 |
| 18 | C3 簡化段補：擾動提供互斥梯度、完全相同時 Σ∇k=0、40/40/40 為人為設定、論文另有 drifting 與 FM 正則 |
| 19 | VAE 結論恢復「支持…；所測的 VAE 對照沒有受益」 |
| 20 | Muon／AdamW、batch 256 限定 5 epoch 執行 |
| 21 | 期望跡趨近 0 維持〔本頁推導〕，並寫明剛性論證靠最大值而非 Eq. 10 第二行 |

## 9. 修訂紀錄（readiness NO-GO 後）

- R6 改為以 Table 5 為主，正文差值與表格的差異（0.11、0.09）明寫；S0 層軸改用「第 23 層（深層）」等字面標籤；C2 類別預設 0。
- C2 時間嵌入改為「j = 0、1 可分開，j = 2、3 不行」；App. A.1 的兩個條件改名 (E1)/(E2)，避免與元件 C1/C2 混淆；（5.4）→（6.3）、（5.6）→ 6.2；CFG 於 S8 首次出現處定義；hero 卡 3 寫明 RevFlow-XL、690M；C2 相關係數改為 0.892／0.965；S11 串接加入 Latent Forcing。

## 10. 修訂紀錄（GO 後、建構前）

- Hero 卡 2 把「乾淨目標」改為字面說明；卡 3 定義 FID。S9 在「高 3.65 與 4.04」之後緊接 R6 說明句。S11 的 Latent Forcing 用 timeline.json 的 role 文字。

## 11. 建置紀錄（build round）

- 組裝：`build/src/assemble.py` 把參考頁的 head/CSS 與 JS 工具（第 1–340、911–941、969–970 行）接上 `src/extra.css`、`src/body.html`、`src/app.js`、`src/tail.js` 與 `notes/timeline.json`，輸出 `build/index.html`（約 110 KB）。`src/localcopy.sh` 產生不連外的測試副本（本機 headless Chrome 無法連 Google Fonts／cdnjs，需 `KATEX_DIR` 與 `--user-data-dir`）。
- 與設計的差異：S3 管線圖、S5／S7 長條圖、hero 層軸改用 HTML/CSS（不用 SVG），窄螢幕自然換行、文字維持實際字級；第二組長條改為「空心」而非斜線（impeccable：repeating-gradient stripes）；hero 生成箭頭線改用背景漸層（impeccable：side-tab）。C2 熱圖與條件格為 HTML grid。
- 驗證：`?test` 61/61（1440 與 390 寬）；`extract_page_text.py` 後 12 個 LO 的關鍵詞與數字都在 `accept/page.txt`；impeccable 0 anti-pattern（只剩 numbered-section-labels advisory）；390px 無水平捲動（selfTest 斷言）；KaTeX 已渲染（selfTest 斷言）；無 console error。截圖在 `build/caps/`。
