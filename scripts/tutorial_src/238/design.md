# 設計文件：〈Understanding and Learning Prediction Targets in Diffusion Models〉互動教學

- 論文：Qing Jin、Chaoyang Wang，arXiv 2601.21419（v3，NeurIPS 2026）。資料庫編號 No. 238。
- 原文：`tut/238/src/paper.txt`（PDF 同目錄）。本文件中的「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」皆指論文原文編號。
- 產出：`tut/238/build/index.html`（單一 HTML 檔）。上線位置：`tutorials/prediction-targets-k-diff.html`（page slug `prediction-targets-k-diff`）。所有站內相對連結都以這個上線位置為準。
- 數值驗證腳本（非頁面程式，僅供核對本文件數字）：`tut/238/verify/check.py`、`check2.py`、`table.py`、`check5.py`、`check6.py`。第 4 節的驗收表以 `check6.py`（與頁面相同的 $M=1024$、$k$ 步長 0.005、解析梯度）為準。
- 研究筆記：`tut/238/notes/`（總覽 `prediction-targets-overview` + 7 篇主題筆記 + `timeline.json`），上線後為 `tutorials/<slug>.html`。
- 文件慣例：凡是**本設計自行推導、論文沒有寫出**的結論，一律標記「〔本頁推導〕」。頁面上用同樣的標籤（見 5.6）。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

擴散模型與 flow matching 要讓神經網路從帶噪輸入 $z$ 預測某個目標：噪聲（ε-prediction）、速度（v-prediction）或乾淨資料（x-prediction）。實務上，低維資料多用 ε 或 v，高解析度像素空間則轉向 x，但沒有理論說明「最佳目標為什麼取決於資料」。本文把目標寫成連續的一族 $u = kx-(1-k)n$（Eq. 13），$k=0,0.5,1$ 分別對應 ε、v、x（差一個常數倍）。作者分析一個「所有噪聲強度共用同一個權重矩陣」的單層線性去噪器，發現它的學習動態拆成兩個互不干擾的模式：沿資料子空間的**平行模式**（資料重建）與垂直於子空間的**垂直模式**（消除噪聲）。最佳損失中平行項正比於內在維度 $d$、垂直項正比於餘維度 $D-d$（Eq. 12）。在均勻時間取樣、u-MSE 損失的條件下，最佳參數為 $k^*=D/(D+d)$（Eq. 14）：資料填滿空間時 $k^*=0.5$（v），資料只占極低維子空間時 $k^*\to1$（x）。作者再用受控實驗顯示時間條件、取樣分布與損失會移動這個最佳值（Sec. 3.4、Fig. 1），因此提出 **k-Diff**：把 $k=\mathrm{sigmoid}(w_k)$ 當成一個可訓練的純量，與網路一起用反向傳播訓練（Sec. 4、Eq. 15）。在 ImageNet 256 的潛空間（LightningDiT-XL/1）中 $k$ 由 0.5 收斂到 0.66（Fig. 3b 的長期收斂值），在像素空間（JiT-B/16）中 30 個 epoch 內收斂到接近 1.0，FID 與固定目標基準相近（Table 1）。

### 1.2 研究動機（Sec. 1）

作者列出三個未解問題：

1. 最佳目標是否只能在 ε、v、x 三者中選？低維時 v 是否普遍最佳？高維時 x 是否必要？
2. 先前文獻（JiT [8]）以流形假說定性說明 x-prediction 適合稀疏高維資料，但沒有定量關係：最佳目標能否由環境維度 $D$ 與內在維度 $d$ 解析推得？
3. 實務上沒有選擇目標的原則，只能試錯。

### 1.3 貢獻

1. 一個通用形式 $z=\alpha x+\sigma n,\ u=\varphi x+\psi n$（Eq. 1），把各種目標視為同一目標函數的特例。
2. 單層線性去噪器的學習動態分析：平行／垂直兩個模式（Eq. 9）、最佳權重與最佳損失（Theorem 3.1，Eq. 11–12）。
3. $k$ 參數化下的閉式解 $k^*=D/(D+d)$（Theorem 3.2，Eq. 14），以及推廣到資料二階動差矩陣 $\Sigma=\mathbb E[xx^\top]$ 為一般半正定矩陣時的 $k^*=D/(D+\mathrm{Tr}\,\Sigma)$（App. G，Eq. 72）。
4. 受控維度掃描（$D=128$）：時間條件、取樣分布、損失如何移動最佳 $k$（Sec. 3.4，Fig. 1；App. H）。
5. k-Diff：可訓練的 $k$，在潛空間與像素空間的 ImageNet 生成上驗證（Sec. 4–5）。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S1 末尾以可收合表格提供同一份內容）

| 符號 | 意義 |
| --- | --- |
| $x\in\mathbb R^D$ | 乾淨資料 |
| $n\in\mathbb R^D$ | 標準高斯噪聲；Alg. 1 程式碼裡寫成 `e`，兩者相同 |
| $D$、$d$ | 環境維度（資料向量的長度）、內在維度（資料實際所在子空間的維度） |
| $t\in[0,1]$ | 擴散時間；本文 $t=1$ 為乾淨資料、$t=0$ 為純噪聲 |
| $\alpha,\sigma$ | 前向過程係數，flow matching 中 $\alpha=t,\ \sigma=1-t$ |
| $\varphi,\psi$ | 目標係數，$u=\varphi x+\psi n$ |
| $u$ | 網路要預測的目標；$\hat u$ 為網路輸出 |
| $v=x-n$ | 速度（velocity）目標。「v-loss／v-MSE」指以 $\|\hat v-v\|^2$ 為損失，「v-prediction」指網路直接輸出 $v$，兩者是不同概念 |
| $I$ | 對 $\alpha$ 積分的區間，通常為 $[0,1]$ |
| $D\alpha$ | 訓練時 $\alpha$（即 $t$）的取樣機率測度 |
| $\kappa$、$\widetilde{D\alpha}$ | 損失換算因子與有效測度（S7 才在頁面上引入） |
| $P$ | $D\times d$ 矩陣，$P^\top P=I_{d\times d}$；$PP^\top$ 為投影到資料子空間的矩陣 |
| $W$、$W_\parallel$、$W_\perp$ | 線性去噪器權重及其平行、垂直分量 |
| $\tau$、$\tilde t$ | 學習率的倒數、訓練時間（與擴散時間 $t$ 無關） |
| $\Sigma$ | $\mathbb E[xx^\top]$，只在 App. G 推廣時使用 |

以下依論文推導順序排列。每一步說明「作者想回答什麼」「做了什麼假設」「得到什麼」「下一步為什麼需要它」。

#### 步驟 1：把所有目標寫成同一個形式（Sec. 3，Eq. 1）

$$z=\alpha x+\sigma n,\qquad u=\varphi x+\psi n \tag{Eq. 1}$$

- $\alpha,\sigma$ 定義前向過程（通常預先給定），$\varphi,\psi$ 定義網路要預測的目標（由設計者選）。
- 例：flow matching 的 v-prediction 取 $\alpha=t,\ \sigma=1-t,\ \varphi=1,\ \psi=-1$；x-prediction 取 $\varphi=1,\ \psi=0$。
- **時間方向慣例**：本文 $\alpha=t$，所以 $t=1$ 是乾淨資料、$t=0$ 是純噪聲，與 DDPM「$t$ 越大越吵」相反。
- 作者以 $\alpha$ 為自變數，其餘三者是 $\alpha$ 的函數；$x,n$ 與 $\alpha$ 無關。
- 為什麼網路要預測目標（頁面 S1 用 2–3 句說明）：生成時從 $t=0$ 的純噪聲出發，反覆用網路的預測換算出速度 $v$，沿 $dz/dt=v$ 往 $t=1$ 前進（Alg. 2 的 Euler 步 `z_next = z + (t_next - t) * v_pred`）。網路預測得越準，終點越接近資料分布。

#### 步驟 2：訓練時比較的變數不一定是 u——比例因子 κ（Eq. 3–5，App. A）

網路輸出 $\hat u$，但損失可能算在另一個變數 $w=\xi x+\eta n$ 上（$\xi,\eta$ 為係數，例如 $w=v$ 時 $\xi=1,\eta=-1$）。由 $z,\hat u$ 解出 $\hat x,\hat n$（Eq. 16–18）後可得

$$\hat w-w=\kappa(\hat u-u),\qquad \kappa=\frac{\xi\sigma-\eta\alpha}{\varphi\sigma-\psi\alpha}\tag{Eq. 3, 22}$$

- u-loss：$\kappa=1$（Eq. 23）；x-loss：$\kappa=\sigma/(\varphi\sigma-\psi\alpha)$（Eq. 24）；ε-loss：$\kappa=-\alpha/(\varphi\sigma-\psi\alpha)$（Eq. 25）；v-loss：$\kappa=(\alpha+\sigma)/(\varphi\sigma-\psi\alpha)$（Eq. 26）。
- 因此任何 w-MSE 都可寫成 u-MSE 乘上權重 $\kappa^2$：

$$L=\frac12\int_I \widetilde{D\alpha}\;\mathbb E\,\|Wz-u\|^2,\qquad \widetilde{D\alpha}:=D\alpha\cdot\kappa^2\tag{Eq. 4, 5}$$

- 為什麼需要這一步：後面所有結論都用有效測度 $\widetilde{D\alpha}$ 表示，換取樣分布或換損失，只是換 $\widetilde{D\alpha}$。
- 〔本頁推導〕在 $k$ 參數化與 flow matching 下，v-loss 的 $\kappa=1/\big(k(1-t)+(1-k)t\big)$（把 $\xi=1,\eta=-1,\alpha=t,\sigma=1-t,\varphi=k,\psi=-(1-k)$ 代入 Eq. 26，$\alpha+\sigma=1$）。
- 頁面安排：κ 與有效測度延到 S7（第一次用到時）才講；S2 不出現。

#### 步驟 3：最簡化的問題設定（Sec. 3.1）

- 資料：$x=P\tilde x$，$\tilde x\in\mathbb R^d$ 是內在座標，$P$ 是 $D\times D$ 正交矩陣的 $D\times d$ 子矩陣，$P^\top P=I_{d\times d}$。此設定沿用 JiT [8]。
- $PP^\top$ 是投影矩陣：對任意向量 $y$，$PP^\top y$ 是 $y$ 在資料子空間上的分量，$(I-PP^\top)y$ 是垂直分量；兩者相加等於 $y$，且 $(PP^\top)^2=PP^\top$、$PP^\top(I-PP^\top)=0$（Eq. 34）。頁面 S3 用 $D=2,d=1$ 的圖示說明（與 C1 同一個平面）。
- 去噪器：單層線性 $\hat u=Wz$（Eq. 2），$W\in\mathbb R^{D\times D}$，**所有噪聲強度共用同一個 $W$**（論文後文稱 time-shared）。
- 白化假設：$\mathbb E[\tilde x\tilde x^\top]=I_{d\times d}$（Eq. 6）；噪聲 $\mathbb E[nn^\top]=I$（Eq. 7）。
- 梯度下降寫成連續時間的微分方程：$\tau\,dW/d\tilde t=-\partial_W L$（Eq. 27），也就是學習率很小時，參數隨訓練時間 $\tilde t$ 沿負梯度方向連續移動。頁面 S3 以一句話與這個式子說明。
- 為什麼這樣簡化：線性模型讓期望值可以寫成閉式，投影矩陣讓問題能沿子空間與其正交補拆開。

#### 步驟 4：學習動態拆成兩個模式（Sec. 3.2，Eq. 8–10，App. B）

由 $x,n$ 獨立、$\mathbb E[n]=0$（Eq. 30）與 Eq. 6–7 得

$$\mathbb E[zz^\top]=\alpha^2PP^\top+\sigma^2 I,\qquad \mathbb E[uz^\top]=\varphi\alpha PP^\top+\psi\sigma I\tag{Eq. 32}$$

代入梯度：

$$\tau\frac{dW}{d\tilde t}=-\int\widetilde{D\alpha}\,\big(\alpha^2WPP^\top+\sigma^2W-\varphi\alpha PP^\top-\psi\sigma I\big)\tag{Eq. 8}$$

兩個期望值只由 $PP^\top$ 與 $I$ 組成；定義 $W_\parallel=WPP^\top$、$W_\perp=W(I-PP^\top)$（Eq. 10），把 Eq. 8 右乘 $PP^\top$ 與 $I-PP^\top$，用 Eq. 34 得到

$$\tau\frac{dW_\parallel}{d\tilde t}=-\int\widetilde{D\alpha}\big[(\alpha^2+\sigma^2)W_\parallel-(\varphi\alpha+\psi\sigma)PP^\top\big]\tag{Eq. 9a}$$
$$\tau\frac{dW_\perp}{d\tilde t}=-\int\widetilde{D\alpha}\big[\sigma^2W_\perp-\psi\sigma(I-PP^\top)\big]\tag{Eq. 9b}$$

- 平行模式含 $\alpha,\varphi$（與資料有關），作者稱為**資料重建**。
- 垂直模式不含 $\alpha,\varphi$，只看 $\sigma,\psi$：子空間外沒有資料，那裡的 $z$ 只有噪聲 $\sigma n_\perp$，目標只有 $\psi n_\perp$。作者稱為**消除噪聲**。
- 兩個模式對 $\varphi,\psi$ 的要求不同，最佳設計要權衡兩者。這是全文的核心觀察。
- 頁面處理：頁面在 S4 用 u-MSE、$\kappa=1$ 的寫法（$\widetilde{D\alpha}=D\alpha$）呈現 Eq. 8–9，並註明「S7 會把 $D\alpha$ 換成一般的 $\widetilde{D\alpha}$」。

#### 步驟 5：最佳權重與最佳損失（Theorem 3.1，Eq. 11–12，App. C）

令 Eq. 9 等於 0：

$$W^*=\frac{\int\widetilde{D\alpha}(\varphi\alpha+\psi\sigma)}{\int\widetilde{D\alpha}(\alpha^2+\sigma^2)}PP^\top+\frac{\int\widetilde{D\alpha}\,\psi\sigma}{\int\widetilde{D\alpha}\,\sigma^2}(I-PP^\top)\tag{Eq. 11}$$

代回 Eq. 4（用 $\mathrm{Tr}(PP^\top)=d$、$\mathrm{Tr}(I-PP^\top)=D-d$，Eq. 48–49；積分皆在區間 $I$ 上）：

$$\Delta^*=\underbrace{\frac d2\left[\int\widetilde{D\alpha}(\varphi^2+\psi^2)-\frac{\big(\int\widetilde{D\alpha}(\varphi\alpha+\psi\sigma)\big)^2}{\int\widetilde{D\alpha}(\alpha^2+\sigma^2)}\right]}_{\text{平行（流形內）}}+\underbrace{\frac{D-d}2\left[\int\widetilde{D\alpha}\,\psi^2-\frac{\big(\int\widetilde{D\alpha}\,\psi\sigma\big)^2}{\int\widetilde{D\alpha}\,\sigma^2}\right]}_{\text{垂直（環境殘差）}}\tag{Eq. 12}$$

作者的四點 Remark：
1. 平行項正比於 $d$，垂直項正比於餘維度 $D-d$。
2. 平行項依賴 $\alpha,\varphi$；垂直項與兩者無關。
3. $D\approx d$ 時平行項主導；$D\gg d$ 時垂直項主導。
4. 取 $\psi=0$（x-prediction）時垂直項為 0。

- 垂直項的來源（頁面 C1b 呈現）：若每個噪聲強度各有一個矩陣，垂直方向的最佳增益是 $\psi/\sigma$（App. H，Eq. 76b），此時預測 $\tfrac{\psi}{\sigma}\cdot\sigma n_\perp=\psi n_\perp$ 與目標完全相同。共享的 $W_\perp$ 只能取一個加權平均 $b^*=\int\widetilde{D\alpha}\psi\sigma/\int\widetilde{D\alpha}\sigma^2$，預測 $b^*\sigma n_\perp$ 與目標 $\psi n_\perp$ 的差 $(b^*\sigma-\psi)n_\perp$ 就是殘差。
- 垂直殘差為 0 的條件：Eq. 12 垂直方括號由 Cauchy–Schwarz 不等式（Eq. 78b）非負，等號成立若且唯若 $\psi/\sigma$ 在 $\widetilde{D\alpha}$ 的支撐上為常數。在 $k$ 參數化與 flow matching 下，$\psi/\sigma=-(1-k)/(1-t)$，只有 $k=1$ 時為常數（恆為 0）〔本頁推導：等號條件的套用〕。
- 下一步的理由：直接對 Eq. 12 做變分求最佳 $\varphi(\alpha),\psi(\alpha)$ 太複雜，結果還依賴損失（$\kappa$）、取樣分布與積分區間 $I$。所以作者再加限制。

#### 步驟 6：k 參數化與閉式解（Sec. 3.3，Theorem 3.2，Eq. 13–14，App. D）

令 $\varphi=k,\ \psi=-(1-k)$，$0\le k\le1$ 且與 $t$ 無關：

$$u=kx-(1-k)n\tag{Eq. 13}$$

- 搭配 flow matching（$\alpha=t,\sigma=1-t$）：$k=0$ 得 $u=-n$（ε，差一個負號）；$k=0.5$ 得 $u=\tfrac12(x-n)=\tfrac12 v$；$k=1$ 得 $u=x$。論文寫「up to a constant scaling factor」。
- Theorem 3.2 的四個條件：白化子空間資料（$\Sigma=PP^\top$）；共享線性去噪器；$t\sim U[0,1]$；$\kappa=1$（u-MSE）。另有 $\alpha=t,\sigma=1-t$ 且 $k$ 與 $t$ 無關。
- App. D：均勻取樣下 $\int\widetilde{D\alpha}=1$，$\int\alpha=\int(1-\alpha)=1/2$，$\int\alpha^2=\int(1-\alpha)^2=1/3$（Eq. 53）。代入得

$$\Delta^*=d\left(\frac{k^2}4-\frac k4+\frac5{16}\right)+\frac{(D-d)(1-k)^2}{8}=\frac1{16}\big[2(D+d)k^2-4Dk+(2D+3d)\big]\tag{Eq. 54c, 54d}$$

  （平行方括號 $=\tfrac12k^2-\tfrac12k+\tfrac58$，垂直方括號 $=\tfrac14(1-k)^2$，各乘 $\tfrac d2$、$\tfrac{D-d}2$。）

$$k^*=\frac{D}{D+d}\tag{Eq. 14, 55}$$

- 三個區間（Sec. 3.3）：$D\approx d$（稠密資料）$k^*\approx0.5$，v 最佳；$D\gg d$（稀疏資料）$k^*\to1$，x 最佳；一般情形 $0.5<k^*<1$，論文寫「neither conventional target is optimal」。
- 兩項的權衡：平行項在 $k=0.5$ 最小（值 $d/4$），垂直項在 $k=1$ 最小（值 0）。$k^*$ 是兩者依 $d$ 與 $D-d$ 加權後的折衷。
- 〔本頁推導〕超出最小值的損失為 $\Delta^*(k)-\Delta^*(k^*)=\frac{D+d}{8}(k-k^*)^2$。例：$D=128,d=32$：$k^*=0.8$；$\Delta^*$ 在 $k=0,0.5,1,0.8$ 分別為 $22,\ 11,\ 10,\ 9.2$。

#### 步驟 7：推廣到一般二階動差（App. G，Eq. 56–72）

- 若 $\Sigma=\sum_i\lambda_iq_iq_i^\top$，動態沿每個特徵向量獨立（Eq. 61），每個模式的最佳損失為 Eq. 68；特徵值大的模式偏好 $\varphi=0$（Eq. 69），特徵值小的模式偏好 $\psi=0$（Eq. 70）。
- Theorem 3.2 的條件下 $k^*=D/(D+\mathrm{Tr}\,\Sigma)$（Eq. 72）；$\Sigma=PP^\top$ 時 $\mathrm{Tr}\,\Sigma=d$，回到 Eq. 14。
- 頁面處理：只在 S6 末尾用一句話與 Eq. 72 說明（特徵值、SNR 等詞不出現），不設學習目標、不做元件。

#### 步驟 8：閉式解的適用範圍——受控實驗（Sec. 3.4，Fig. 1，App. H）

設定：各向同性高斯資料放在 $\mathbb R^{128}$ 的 $d$ 維子空間（$D=128$，11 個 $d$ 值），$k=\mathrm{sigmoid}(w_k)$ 與去噪器一起訓練，同一 $d$ 共用資料、$k$ 跨 $t$ 共用。Fig. 1 四格的結論（數字照原文）：

| Fig. 1 位置 | 設定 | 結果 |
| --- | --- | --- |
| 左上 | 共享線性、均勻 $t$、u-MSE（Theorem 3.2 原條件） | 學到的 $k$ 與 $D/(D+d)$ 相差在 0.027 內 |
| 右上 | 加入時間條件（八個等寬時間區間各一個線性映射；時間嵌入 MLP），均勻 $t$、u-MSE | $k$ 偏離 $D/(D+d)$；分區預測與學到的八區間 $k$ 相差在 0.025 內；MLP 與八區間模型相差在 0.012 內（平均差 0.004） |
| 左下 | 共享線性，改取樣與損失 | logit-normal 取樣使最佳值在 $d\ge48$ 時變成 $k=0$（Theorem 3.1 預測誤差 0.026 內）；再改用 v-MSE 使高 $d$ 的最佳值離開邊界（預測誤差 0.045 內） |
| 右下 | 時間條件 + logit-normal + v-MSE | 分區母體最佳值不再能定量預測訓練結果，大 $d$ 尤其明顯。$d=128$：八區間母體最佳值 0.212，訓練出的八區間模型 0.659，時間嵌入 MLP 0.612 |

- 「母體最佳值」指對資料與噪聲取期望後（無限多樣本）算出的最佳 $k$；「訓練值」指實際以梯度下降、有限樣本訓練得到的 $k$。
- logit-normal 取樣：$\mathrm{logit}(t)=\ln\frac{t}{1-t}$ 服從常態分布 $\mathcal N(\mu,s^2)$；μ < 0 時取樣集中在 $t<0.5$（噪聲較多的一側）。頁面在 C3 之前用一段文字與密度圖說明。
- App. H（雜訊強度相依的線性模型 $\hat u=W(\alpha)z$，每個 $\alpha$ 各有一個矩陣）：$W^*(\alpha)=\frac{\varphi\alpha+\psi\sigma}{\alpha^2+\sigma^2}PP^\top+\frac{\psi}{\sigma}(I-PP^\top)$（Eq. 76b），$\Delta^*_{dep}=\frac d2\int\widetilde{D\alpha}\frac{(\varphi\sigma-\psi\alpha)^2}{\alpha^2+\sigma^2}$（Eq. 77c），垂直殘差完全消失；且由 Eq. 78a，平行項也不大於共享模型的平行項，合計 $\Delta^*_{dep}\le\Delta^*$（Eq. 79）。頁面在 C3 之前用一段文字說明 $W(\alpha)$ 與「N 區間」是共享模型與雜訊強度相依模型之間的中間情形。
- 作者的結論：知道 $d$ 與 $D$ 並不足以決定實際學到的 $k$，所以要在實際訓練設定下學 $k$。這是 k-Diff 的動機。
- 〔本頁推導〕以下三點由 Eq. 12 / 77c 推得，並以 `check6.py` 數值確認，頁面 C3 會呈現：
  1. 均勻取樣 + u-MSE + 雜訊強度相依模型：剩下的損失只有平行項 $\frac d2\int\frac{(\varphi\sigma-\psi\alpha)^2}{\alpha^2+\sigma^2}dt$，被積函數在 $(k,t)\to(1-k,1-t)$ 下不變，所以 $\Delta^*_{dep}(k)=\Delta^*_{dep}(1-k)$；它是 $k$ 的凸二次函數，最小值因此在 $k=0.5$，與 $d$ 無關。區間數 $N$ 增加時，母體最佳 $k$ 從 $D/(D+d)$ 往 0.5 移動（$d=32$：$N=1,2,4,8,16,32,64$ 分別為 0.80、0.73、0.67、0.605、0.56、0.535、0.515）。這是線性模型的母體結果，不預測實際網路：Sec. 3.4 右下格顯示母體預測失準，而時間條件化的 JiT-B/16 在像素空間學到接近 1.0（Sec. 5.2）。
  2. v-MSE + 雜訊強度相依模型：$\kappa^2(\varphi\sigma-\psi\alpha)^2=1$，所以 $\Delta^*_{dep}$ 與 $k$ 無關，損失無法決定 $k$。
  3. 對稱取樣分布（$p(t)=p(1-t)$）+ u-MSE + 共享線性：最佳值仍是 $D/(D+d)$。因此 Fig. 1 中使 $k\to0$ 的 logit-normal 必然不對稱。本頁採 μ = −0.8、s = 0.8（與 JiT 像素空間設定相同，Table 3），算出的母體最佳值與論文報告的「$d\ge48$ 時 $k=0$」及「$d=128$ 八區間 0.212」一致。這是**用論文數字校準的推定值，不是論文寫明的設定**（論文 Sec. 3.4 未給參數），頁面必須這樣寫，見 R1。

#### 步驟 9：k-Diff（Sec. 4，Eq. 15，Alg. 1–2，App. E–F）

$$L(\theta,w_k)=\frac12\int\widetilde{D\alpha}\;\mathbb E\,\|u_\theta(z)-u\|^2,\qquad k=\mathrm{sigmoid}(w_k)\tag{Eq. 15}$$

- $u$ 透過 $k$ 依賴 $w_k$（Eq. 13）。$\theta$ 與 $w_k$ 一起反向傳播更新。
- **損失空間的規則**（頁面與 C3、C4 一律依此）：Eq. 15 寫在 u 空間，權重 $\widetilde{D\alpha}=D\alpha\cdot\kappa^2$。實際訓練（Alg. 1、Table 3）用 v-loss，即 $w=v$、$\kappa=1/\big(k(1-t)+(1-k)t\big)$；此時 $\frac12\int\widetilde{D\alpha}\,\|\hat u-u\|^2=\frac12\int D\alpha\,\|\hat v-v\|^2$，與 Alg. 1 的 `l2_loss(v - v_pred)`（$t$ 依 $D\alpha$ 抽樣）完全相同。因為 $\kappa$ 依賴 $k$，對 $w_k$ 的梯度包含 $\kappa^2$ 隨 $k$ 的變化。Theorem 3.2 與 Fig. 1 的「u-MSE」則是 $\kappa=1$。頁面中「u-MSE」「v-MSE」兩詞只指這兩種 $\kappa$。
- 〔本頁推導〕Alg. 1 的 `v` 在代數上恆等於 $x-e$，與 $k$ 無關（由 Eq. 17：$x=\frac{(1-k)z+(1-t)u}{k(1-t)+(1-k)t}$、$n=\frac{kz-tu}{k(1-t)+(1-k)t}$ 相減）。所以實際訓練永遠在同一個 v 空間比較，$k$ 只決定網路輸出如何換算成 v。
- 時間相依版本 $k(t)$：$[0,1]$ 切成 $N$ 個區間（例如 $N=128$），每個端點 $t_i$ 有一個可訓練的 $w_k(t_i)$，區間內對 sigmoid 後的端點值做線性內插（App. F）。
- 演算法（App. E；程式碼中的 `e` 即本文的 $n$）：

  訓練（Alg. 1）：
  ```
  t = sample_t(); e = randn_like(x); k = sigmoid(w_k)
  z = t*x + (1-t)*e
  u = k*x - (1-k)*e
  v = ((1-2k)*z + u) / (k*(1-t) + (1-k)*t)
  u_pred = net(z, t)
  v_pred = ((1-2k)*z + u_pred) / (k*(1-t) + (1-k)*t)
  loss = l2_loss(v - v_pred)
  ```
  取樣（Alg. 2，以 Euler 步寫出）：`v_pred` 同上，`z_next = z + (t_next - t) * v_pred`。**論文報告的所有結果都用 Heun 解算器 50 步（NFE = 99）**（Sec. 5.1、App. I、Table 3），不是 Euler。
- 數值穩定（Sec. 4，沿用 [8]）：分母 $k(1-t)+(1-k)t$ 在必要時截斷到最小 0.05。Table 3 的欄名是「clip of (1 − t) in division」：潛空間「x-prediction 用 0.05，其他（含 k-Diff）用 0」，像素空間 0.05。$k=1$ 時分母是 $1-t$，在 $t\to1$ 歸零；$k=0$ 時分母是 $t$，在 $t\to0$ 歸零；$0<k<1$ 時分母不小於 $\min(k,1-k)$。
- C3、C4 的截斷處理：不截斷 $\kappa^2$。理由：只有設定 C 用 v-MSE，其取樣密度 logit-normal 在端點趨近 0 的速度快於 $\kappa^2$ 增長，積分有限；並且已驗證加上 0.05 截斷後，設定 C 共享線性在所有驗收 $d$ 的最佳值（步長 0.005）完全不變（`check6.py`）。
- 作者強調 k-Diff 與特徵對齊、取樣法、自編碼器改良等方法正交，可直接加入既有流程（Sec. 4）。

#### 步驟 10：實驗（Sec. 5，Table 1–2，Fig. 2–5）

- 潛空間（ImageNet 256，LightningDiT-XL/1，VA-VAE tokenizer）：64 epoch，CFG 1.5，Heun 50 步（NFE = 99）。v-prediction 基準 FID 2.08（原論文報告 2.11），k-Diff 2.05。延長到 800 epoch：最終 FID 1.34，最低 1.22 出現在第 384 epoch（之後有過擬合趨勢，Fig. 3a）。Table 1 用最終 checkpoint：LightningDiT-XL/1 FID 1.35 / IS 295.3，k-Diff 1.34 / 301.7。
- 潛空間的 $k$：Fig. 3b（橫軸為正規化 epoch）顯示 $k$ 從 0.5 單調上升，長期收斂值 0.66。Table 2 記錄第 64 epoch 的值 0.522。800 epoch 這個數字屬於 Fig. 3a 的 FID 曲線。
- 像素空間（JiT-B/16，ImageNet 256）：200 epoch 時 k-Diff 與 x-prediction 的 FID 都是 4.70（Fig. 2b）；$k$ 在 30 個 epoch 內從 0.5 升到接近 1.0（Fig. 3b）。Table 1 像素空間列（論文未寫明這些列的訓練 epoch 數）：JiT-B/16 3.66 / 275.1，k-Diff 3.64 / 274.9；ImageNet 512（JiT-B/32）：JiT 4.02 / 271.0，k-Diff 4.03 / 277.1。
- 消融（Table 2，LightningDiT-XL/1，64 epoch）：單一 $k$（學到 0.522）FID 2.05；兩個獨立參數 $k_0x-k_1\varepsilon$（學到 0.556、0.509）FID 2.04；128 區間時間相依 $k(t)$ FID 2.17（較差）。
- Fig. 5a（128 區間 $k(t)$）：論文寫「排除 $t=0$ 與 $t=1$ 附近的邊界區域後」，$k(t)$ 隨 $t$ 單調下降，在 $t\approx0$（邊界區域內側）約 0.56，接近 $t=1$ 時降到約 0.3；而緊鄰邊界（$t\in\{0,1\}$）的參數停在初始值 0.5。作者歸因於 logit-normal（μ = 0、σ = 1）取樣密度在端點趨近 0、梯度不足（Fig. 5b）。

#### 步驟 11：實作細節（App. I，Table 3）

| 項目 | 潛空間 LightningDiT-XL | 像素空間 JiT-B |
| --- | --- | --- |
| tokenizer | VA-VAE | — |
| depth / hidden / heads | 28 / 1152 / 16 | 12 / 768 / 12 |
| image size / patch size | 256 / 1 | 256（另有 512）/ image_size ÷ 16 |
| bottleneck | — | 128 |
| in-context class tokens / start block | — | 32 / 4 |
| dropout | 0 | 0 |
| loss | v-loss | v-loss |
| epochs | 64（消融）、800 | 200（消融）、600 |
| warmup epochs | 0 | 5 |
| optimizer | Adam，β1, β2 = 0.9, 0.95 | 同左 |
| batch / lr / schedule / weight decay | 1024 / 2e-4 / constant / 0 | 同左 |
| EMA decay | {0.9996, 0.9998, 0.9999} | 同左 |
| time sampler | logit(t) ~ N(0.0, 1.0²) | logit(t) ~ N(−0.8, 0.8²) |
| noise scale | 1.0 | 1.0 × image_size ÷ 256 |
| clip of (1 − t) in division | x-prediction 0.05，其他 0 | 0.05 |
| class token drop（CFG 用） | 0.1 | 0.1 |
| 取樣 | Heun 50 步，時間點在 [0, 1] 線性分布，CFG 1.5，CFG 區間 [0.1, 1] | Heun 50 步，同左，CFG 2.9，CFG 區間 [0.1, 1] |

常見錯誤（頁面 S10 逐條列出）：
1. 時間方向弄反：本文 $t=1$ 是資料、$t=0$ 是噪聲。
2. 分母 $k(1-t)+(1-k)t$ 未截斷：Sec. 4 寫「必要時截斷到 0.05」；Table 3 寫潛空間 x-prediction 0.05、其他 0，像素空間 0.05。兩處原文都照錄。
3. 把損失算在 u 空間：Alg. 1 與 Table 3 用 v-loss；$k$ 改變的是輸出參數化，v 目標本身與 $k$ 無關。
4. $k$ 的參數化與初始化：$k=\mathrm{sigmoid}(w_k)$、初始 $w_k=0$（$k=0.5$，Fig. 3b 由 0.5 出發）。論文沒有說明 $w_k$ 是否使用與網路相同的學習率，也沒有說明 EMA（decay {0.9996, 0.9998, 0.9999}）是否套用在 $w_k$。
5. 用時間相依 $k(t)$ 期待更好結果：Table 2 中 128 區間 $k(t)$ 的 FID 2.17 差於常數 $k$ 的 2.05；取樣密度低的端點參數學不動（App. F）。
6. 取樣時忘了用學到的 $k$ 換算 v：Alg. 2 以 Euler 步寫出換算，論文報告的結果用 Heun 50 步（NFE = 99）。

#### 步驟 12：限制（Sec. 6）

閉式解假設白化子空間資料、共享線性去噪器、均勻時間取樣、u-MSE。受控實驗顯示學到的目標會隨去噪器、取樣、損失改變，且分區母體最佳值不一定等於訓練值。推廣到非線性模型與更大規模的文字到影像系統，以及釐清目標選擇如何影響樣本品質，仍是開放問題。

---

## 2. 學習目標（驗收測試）

最終頁面必須讓讀者能回答下列問題；Goldfish 審查時逐條檢查「指定章節是否確實教到、參考答案是否能從頁面內容得出」。章節編號見第 3 節。每個章節結尾以 `<details>` 呈現該章的檢查題（題目與參考答案原文照搬）。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 寫出本論文 flow matching 的前向過程與時間方向 | 本文的 $z$ 怎麼寫？$t=0$、$t=1$ 各代表什麼？ | $z=tx+(1-t)n$；$t=1$ 為乾淨資料，$t=0$ 為純噪聲（Sec. 3，$\alpha=t,\sigma=1-t$） | S1 |
| LO2 | 用 $k$ 表示三種傳統目標 | $u=kx-(1-k)n$ 中 $k=0,0.5,1$ 各得到什麼？ | $k=0$：$u=-n$（ε，差負號）；$k=0.5$：$u=\tfrac12(x-n)=\tfrac12v$；$k=1$：$u=x$（Eq. 13） | S2 |
| LO3 | 說明 κ 的角色 | 用 v-loss 訓練時，損失與 u-MSE 的關係是什麼？ | $\hat v-v=\kappa(\hat u-u)$，損失等於以 $\kappa^2$ 加權的 u-MSE，即有效測度 $\widetilde{D\alpha}=D\alpha\cdot\kappa^2$（Eq. 3–5、26） | S7 |
| LO4 | 解釋學習動態為何拆成兩個模式 | 為什麼 Eq. 8 可以拆成 $W_\parallel$ 與 $W_\perp$？ | $\mathbb E[zz^\top]$ 與 $\mathbb E[uz^\top]$ 都只由 $PP^\top$ 與 $I$ 組成，$PP^\top$ 與 $I-PP^\top$ 是互相正交的投影（Eq. 32、34） | S4 |
| LO5 | 區分兩個模式的物理意義 | 垂直模式的動態與哪些參數無關？代表什麼過程？ | 與 $\alpha$、$\varphi$ 無關（只含 $\sigma,\psi$）；代表消除噪聲，平行模式代表資料重建（Sec. 3.2） | S4 |
| LO6 | 讀懂 Eq. 12 的兩項 | 最佳損失的平行項、垂直項各正比於什麼？取 $\psi=0$ 時發生什麼？ | 平行項 ∝ $d$，垂直項 ∝ $D-d$；$\psi=0$（x-prediction）時垂直項為 0（Theorem 3.1 Remark） | S5 |
| LO7 | 說明共享模型為何留下垂直殘差 | 共享線性模型的垂直殘差從哪裡來？雜訊強度相依模型為何沒有？ | 每個 $t$ 的理想垂直增益 $\psi/\sigma$ 隨 $t$ 改變，共享 $W_\perp$ 只能取 $\int\widetilde{D\alpha}\psi\sigma/\int\widetilde{D\alpha}\sigma^2$；$W(\alpha)$ 可在每個 $\alpha$ 取 $\psi/\sigma$，Eq. 77c 只剩 $d$ 項。兩者的差距不只在垂直項：由 Eq. 78a，雜訊強度相依模型的平行項也較小或相等（App. H） | S5、S7 |
| LO8 | 計算閉式最佳 $k$ | Theorem 3.2 下 $k^*$ 是多少？$D=128,d=32$ 時 $k^*$ 與 $\Delta^*$ 在 ε、v、x、$k^*$ 的值？ | $k^*=D/(D+d)=0.8$；$\Delta^*=\tfrac1{16}[2(D+d)k^2-4Dk+(2D+3d)]$，值為 22、11、10、9.2（Eq. 14、54d） | S6 |
| LO9 | 說出三個維度區間 | $d=D$ 與 $d/D\to0$ 時 $k^*$ 分別趨近什麼？ | $d=D$：0.5（v）；$d/D\to0$：1（x）；其他情形 $0.5<k^*<1$，三個傳統目標都不是最佳（Sec. 3.3） | S6 |
| LO10 | 列出閉式解的假設與會移動 $k^*$ 的因素 | Theorem 3.2 依賴哪四個條件？Fig. 1 中改變哪些設定會讓學到的 $k$ 偏離 $D/(D+d)$？ | 白化子空間資料、共享線性去噪器、$t\sim U[0,1]$、u-MSE；時間條件、取樣分布、損失。例：Fig. 1 左下（共享線性 + logit-normal + u-MSE）$d\ge48$ 時最佳 $k=0$；Fig. 1 右下（時間條件 + logit-normal + v-MSE）$d=128$ 時八區間母體最佳 0.212，訓練出的八區間模型 0.659（Sec. 3.4、Sec. 6） | S7 |
| LO11 | 描述 k-Diff 的做法與數值注意事項 | k-Diff 如何參數化 $k$？實際損失算在哪個空間？分母何時趨近 0？論文如何截斷？ | $k=\mathrm{sigmoid}(w_k)$，純量，與網路一起反向傳播（Eq. 15），初始 0.5；損失為 v-loss，$v_{pred}=((1-2k)z+u_{pred})/(k(1-t)+(1-k)t)$；$k\to1,t\to1$ 或 $k\to0,t\to0$ 時分母趨近 0；Sec. 4 寫必要時截斷到 0.05，Table 3：像素空間 0.05，潛空間 x-prediction 0.05、其他 0（Sec. 4、Alg. 1、Table 3） | S8、S10 |
| LO12 | 讀出實驗結果 | 潛空間與像素空間學到的 $k$ 各是多少？時間相依 $k(t)$ 的消融結果？ | 潛空間（LightningDiT-XL/1）0.5→0.66（Fig. 3b 長期收斂值；Table 2 第 64 epoch 為 0.522），64 epoch FID 2.05 vs v-pred 2.08；像素空間（JiT-B/16）30 epoch 內→約 1.0，200 epoch FID 皆 4.70；128 區間 $k(t)$ FID 2.17，差於常數 $k$ 的 2.05（Sec. 5、Table 2） | S9 |

---

## 3. 頁面大綱

全頁單欄閱讀（文字欄 680–720px），互動元件放在加寬的「圖版」（最大 1080px）。章節編號以 JetBrains Mono 顯示（`01 · 前向過程`）。視覺記號：一條 0 到 1 的水平「k 軸」，標出 ε（0）、v（0.5）、x（1），在 hero、C2、C4、S9 重複出現。章節標題一律用陳述句。

| # | id | 標題 | 目的 | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈擴散模型的預測目標：從 ε、v、x 到連續的 k〉 | 副標一句：「這篇論文把三種預測目標放進同一個參數 k，推得最佳 k 由資料維度決定，並讓 k 在訓練中學出來。」k 軸標出 ε、v、x 與兩個學到的值（見下方 hero 規格）。三張白話事實卡（見下方）。連結：arXiv、`../index.html#paper-238`（在關聯圖中查看）、研究筆記總覽 `prediction-targets-overview.html` | 1 螢幕 |
| S1 | `basics` | 01 · 前向過程把資料與噪聲線性混合 | 給初學者：$z=tx+(1-t)n$、時間方向、為什麼網路要預測目標（生成時沿 $dz/dt=v$ 從噪聲走到資料，2–3 句）、三種目標 ε／v／x 的公式。C1a 在此。末尾可收合的符號表（1.4 符號表）。延伸閱讀：`eps-prediction-ddpm-score.html`、`flow-matching-velocity-target.html` | 400–500 字 + C1a |
| S2 | `k` | 02 · 一個參數 k 涵蓋三種目標 | Eq. 13；$k$ 與 ε/v/x 的對應（含常數倍）；只講 u，不講 κ。延伸閱讀：`v-prediction-and-preconditioning.html`、`generalized-target-k-parameterization.html` | 250–350 字 |
| S3 | `linear` | 03 · 分析對象是單層共享線性去噪器 | $x=P\tilde x$、$d$ 與 $D$；$PP^\top$ 作為投影（$D=2,d=1$ 靜態 SVG，viewBox 0 0 360 240：水平粗線為資料子空間，向量 $y=(1.6,1.1)$ 以 `--ink` 箭頭畫出，虛線投影到水平分量 $PP^\top y$（`--c-par`）與垂直分量 $(I-PP^\top)y$（`--c-perp`），兩分量各有直接標籤，直角記號標在投影點；與 C1 同一平面）；單層共享 $W$；白化假設；梯度下降寫成 $\tau\,dW/d\tilde t=-\partial_WL$（Eq. 27） | 350–450 字 + 1 張靜態 SVG |
| S4 | `modes` | 04 · 學習動態拆成平行與垂直兩個模式 | Eq. 32 → Eq. 8 → Eq. 34 → Eq. 9a/9b；物理意義（資料重建、消除噪聲）。以 $\kappa=1$ 的寫法呈現。延伸閱讀：`diffusion-dynamics-theory-background.html` | 400–500 字 |
| S5 | `residual` | 05 · 共享權重在垂直方向留下殘差 | Eq. 11、Eq. 12、四點 Remark、殘差為 0 的條件（Eq. 78b）；C1b 在此 | 400–500 字 + C1b |
| S6 | `kstar` | 06 · 最佳 k 等於 D/(D+d) | Theorem 3.2 的條件、App. D（Eq. 53–55）、三個區間、〔本頁推導〕超出損失；C2；末尾一句 Eq. 72。延伸閱讀：`x-prediction-manifold-view.html` | 400–500 字 + C2 |
| S7 | `scope` | 07 · 時間條件、取樣與損失會移動最佳 k | 依序：(a) κ 與有效測度 $\widetilde{D\alpha}$（Eq. 3–5，四種 loss 的 κ 小表）；(b) logit-normal 取樣的定義與密度小圖；(c) N 區間與雜訊強度相依模型 $W(\alpha)$ 的說明（App. H）；(d) C3；(e) 一段明確說明「母體最佳值與訓練值不同」：$d=128$ 八區間母體最佳 0.212，訓練值 0.659、MLP 0.612，所以作者改為直接訓練 $k$；(f) Fig. 1 四格表。延伸閱讀：`loss-weighting-and-elbo.html` | 650–800 字 + C3 |
| S8 | `kdiff` | 08 · k-Diff 把 k 當成可訓練參數 | Eq. 15 與損失空間規則、Alg. 1–2（≥ 768px 為兩欄格線：左欄程式碼逐行、右欄同一列的中文說明，以 CSS grid 逐行對齊；< 768px 改為每行程式碼下方接說明）、v 目標恆等於 $x-e$〔本頁推導〕、Heun 與 Euler、$k(t)$ 版本；C4。延伸閱讀：`generalized-target-k-parameterization.html` | 450–550 字 + C4 |
| S9 | `results` | 09 · 實驗中 k 在潛空間停在 0.66、在像素空間趨近 1 | Table 1 摘錄（k-Diff 與其基準，3 組）、Table 2 消融、k 軸上的學到值（標明出處：Fig. 3b 長期值、Table 2 第 64 epoch）；文字描述 Fig. 2、3、5 的結論，不重繪曲線（論文沒有給出曲線數據）。延伸閱讀：站內既有 `pixel-space-diffusion.html` | 350–450 字 + 2 張表 + k 軸 |
| S10 | `impl` | 10 · 實作設定與常見錯誤 | Table 3（1.4 步驟 11 全表）、6 條常見錯誤 | 300–400 字 + 表 |
| S11 | `history` | 11 · 研究脈絡（由 notes/timeline.json 提供） | 時間線（5.7）+ 一段文字把時間線接到 k 軸：ε（DDPM、score SDE）→ v（Progressive Distillation、EDM、flow matching）→ 加權觀點（Kingma & Gao）→ x（SiD2、JiT）→ 連續的 k（本文） | 時間線 + 100–150 字 |
| S12 | `next` | 12 · 限制與延伸閱讀 | Sec. 6 限制；研究筆記清單（依總覽的閱讀順序，5.7）；arXiv | 200 字 + 連結清單 |
| — | — | 頁尾 | 引用格式、arXiv 連結；固定導覽列 `#pm-bar` | — |

Hero 規格（S0）：
- 三張事實卡（不使用未定義術語）：
  1. 「同一個式子 $u=kx-(1-k)n$ 包含三種常見目標：k = 0 是預測噪聲，k = 0.5 是預測速度，k = 1 是預測乾淨資料。」
  2. 「在簡化的線性模型中，最佳 k 等於 D/(D+d)：D 是資料向量的長度，d 是資料實際所在子空間的維度。資料越集中在低維子空間，最佳 k 越接近 1。」
  3. 「k-Diff 只多訓練一個數字 k。在 ImageNet 256 上，潛空間模型學到 0.66，像素空間模型學到接近 1。」
- k 軸：刻度 ε（0）、v（0.5）、x（1）標在軸下方；學到的值「潛空間 0.66」「像素空間 ≈ 1.0」標在軸上方。≥ 480px：兩個標記並排在軸上方；< 480px（含 358px 內容寬）：「像素空間 ≈ 1.0」的標籤靠右對齊、垂直位置比「潛空間 0.66」高一行（用短引線連到標記），確保任何兩個標籤的外框不重疊。實作時以 `getBBox()` 檢查重疊，重疊則把較右的標籤再上移一行。

桌機（≥ 1024px）左側窄欄放章節目錄（sticky，Distill 風格），目前章節以 `--accent` 標示；< 1024px 改為頂部可橫向捲動的章節列。

---

## 4. 互動元件規格

### 4.0 共用數學（C1b、C3、C4 共用同一份函式，見 5.4）

- 時間積分一律用中點法：$t_j=(j+0.5)/M$，$M=1024$，權重 $1/M$。$M=1024$ 讓 $N\in\{1,2,4,\dots,64\}$ 個等寬區間都分到整數個點（第 $b$ 區間為索引 $j\in[bM/N,(b+1)M/N)$）。
- 取樣分布：`uniform`：$p(t)=1$；`logitn`：$p(t)=\dfrac{1}{s\sqrt{2\pi}\,t(1-t)}\exp\!\Big(-\dfrac{(\ln\frac t{1-t}-\mu)^2}{2s^2}\Big)$，$\mu=-0.8,\ s=0.8$（推定值，見 R1）。
- 損失權重（損失空間規則見 1.4 步驟 9）：`u`：$\kappa^2=1$；`v`：$\kappa^2=1/\big(k(1-t)+(1-k)t\big)^2$。不截斷（理由見步驟 9）。
- 三種「訓練設定」（只開放 Fig. 1 用過的組合）：**A** = 均勻 + u-MSE；**B** = logit-normal + u-MSE；**C** = logit-normal + v-MSE。均勻 + v-MSE 不開放：$k\to1$ 時 $\int dt/(1-t)^2$ 發散，論文也沒有用這個組合。

### C1 · 二維中的 x、n、z、u（分成 C1a 與 C1b 兩個圖版，共用同一批抽樣點與 seed）

PRNG（固定，確保每次載入相同）：`mulberry32(seed)` 產生 [0,1) 均勻數；`gauss()` 用 Box–Muller，每次呼叫取兩個均勻數 $u_1,u_2$，回傳 $\sqrt{-2\ln(1-u_1)}\cos(2\pi u_2)$（只用 cos 分支，不快取 sin 分支）。每個點依序抽 $\tilde x$、$n_\parallel$、$n_\perp$，共 150 點；突顯樣本為第一個滿足 $|\tilde x|>1$、$|n_\perp|>0.8$、$|n_\parallel|<1.5$ 的點（找不到則用第 0 點），讓箭頭長度足以辨識。初始 seed = 7，「重新抽樣」每按一次 seed 加 1。

共同簡化（頁面寫出）：$D=2,\ d=1$。資料子空間是水平軸，$x=(\tilde x,0)$，$\tilde x\sim\mathcal N(0,1)$（白化，Eq. 6；與 Sec. 3.4 的高斯資料一致）。$n=(n_\parallel,n_\perp)\sim\mathcal N(0,I)$。抽樣用 seed 固定的 PRNG（預設 seed = 7）。

#### C1a（S1）：前向過程

- **教的概念**：$z=tx+(1-t)n$；$t$ 從 0 到 1，點雲從純噪聲收縮到資料所在的直線。
- **讀者操作**：滑桿 $t\in[0,1]$，步長 0.01，預設 0.5；按鈕「重新抽樣」（換 seed）。
- **顯示**：正方形 SVG，座標 $[-3,3]^2$。水平軸畫成粗線並標「資料所在的直線」（S3 之後才稱為子空間）。150 個背景點畫 $z_i=t\,x_i+(1-t)\,n_i$（淺色），拖動 $t$ 時同一批點連續移動。一個突顯樣本：從原點畫 $t\,x$（`--c-par`，實線＋圓形端點），接著畫 $(1-t)\,n$（`--c-perp`，虛線＋方形端點），終點為 $z$（實心點）；三段各有直接文字標籤。讀數：$t$、$\tilde x$、$n$、$z$（兩位小數）。不出現 $k$、$u$。
- **讀者應觀察到**：$t=0$ 時點雲是圓形高斯（純噪聲），$t=1$ 時所有點落在水平線上（資料）；$z$ 永遠是 $x$ 與 $n$ 的線性混合。

#### C1b（S5）：目標向量與垂直殘差

- **教的概念**：$u$ 的垂直分量 $-(1-k)n_\perp$ 是純噪聲；共享線性模型在垂直方向只能用一個增益，$k<1$ 時留下殘差（Theorem 3.1 Remark 4、Eq. 78b）。
- **簡化**：殘差計算用均勻 $t$、u-MSE（Theorem 3.2 條件）。
- **讀者操作**：滑桿 $t$（同 C1a，預設 0.5）；滑桿 $k\in[0,1]$，步長 0.01，預設 0.5；三個 chip「ε（k = 0）」「v（k = 0.5）」「x（k = 1）」；「重新抽樣」。
- **顯示**：
  - 左圖（與 C1a 相同座標）：加上 $u=kx-(1-k)n$（`--accent` 粗箭頭），虛線畫出 $u_\parallel=k\tilde x-(1-k)n_\parallel$ 與 $u_\perp=-(1-k)n_\perp$ 的投影。數值表（JetBrains Mono，兩位小數）：$z_\parallel=t\tilde x+(1-t)n_\parallel,\ z_\perp=(1-t)n_\perp,\ u_\parallel,\ u_\perp$。
  - 右圖上半「垂直方向：目標與共享模型的預測（每單位 $n_\perp$）」：橫軸 $t\in[0,1]$，縱軸 $[-1.6,0.6]$。
    - 水平線 $\psi=-(1-k)$：目標 $u_\perp$ 對 $n_\perp$ 的係數（標「目標」）。若每個 $t$ 各有一個矩陣（App. H），預測 $\frac{\psi}{\sigma}\cdot\sigma=\psi$，與這條線重合（標「雜訊強度相依模型」）。
    - 直線 $b^*\sigma(t)=-1.5(1-k)(1-t)$：共享模型的預測係數，$b^*=\int\psi\sigma/\int\sigma^2=-1.5(1-k)$（Eq. 11 在均勻 $t$、u-MSE 下），標「共享模型」，`--c-perp` 虛線。
    - 兩線之間塗淺色；目前的 $t$ 用垂直細線標示並與左圖同步。
    - 這張圖沒有發散值（取代原先 $\psi/\sigma$ 在 $t\to1$ 發散的畫法）。
  - 右圖下半「每個 $t$ 的殘差 $\tfrac12 e(t)^2$」：$e(t)=b^*\sigma-\psi=(1-k)(1.5t-0.5)$，畫 $\tfrac12e(t)^2$ 曲線並填滿到 0；縱軸 $[0,0.6]$。填滿面積 $=\int_0^1\tfrac12e^2dt=(1-k)^2/8$，正好等於讀數。
  - 讀數：「每個垂直維度的殘差 $=(1-k)^2/8$」（Eq. 54c，$D-d=1$）；$k=0,0.5,1$ 時分別顯示 0.125、0.031、0.000。附註一行：「共享模型與雜訊強度相依模型的差距還包含平行項（Eq. 78a），這張圖只畫垂直部分。」
- **讀者應觀察到**：
  1. $z_\perp=(1-t)n_\perp$：垂直方向的輸入只有噪聲。
  2. $u_\perp=-(1-k)n_\perp$：只有 $k=1$ 時目標的垂直分量為 0。
  3. $k=1$ 時兩線都在 0，殘差為 0；$k$ 越小，兩線差距越大，殘差 ∝ $(1-k)^2$。
  4. 這就是 Eq. 12 垂直項的來源，也是 $D\gg d$ 時偏好大 $k$ 的原因。

### C2 · 平行項與垂直項的權衡：k* = D/(D+d)（S6）

- **教的概念**：Theorem 3.2；平行項偏好 $k=0.5$、垂直項偏好 $k=1$；最佳 $k$ 只取決於 $d/D$；三個區間。
- **簡化**：完全使用 Eq. 54c（Theorem 3.2 的四個條件），無額外近似。
- **讀者操作**：
  - 滑桿 $D$：值集合 $\{2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536\}$，預設 128。
  - 滑桿 $d$：對數刻度，整數 $1\ldots D$（以 $\log_2$ 均分 200 段後四捨五入、去重），預設 32；改變 $D$ 時若 $d>D$ 則設為 $D$。
- **計算與顯示**：
  - 主圖：橫軸 $k\in[0,1]$（k 軸，標 ε/v/x），縱軸 $\Delta^*/D$，固定範圍 $[0,0.35]$（最大值出現在 $d=D,k=0$：$5/16=0.3125$）。三條曲線，以線型與直接標籤區分，不只靠顏色：
    - 平行項 $\frac dD\big(\frac{k^2}4-\frac k4+\frac5{16}\big)$（`--c-par`，實線，曲線右端標「平行」）
    - 垂直項 $\frac{D-d}D\cdot\frac{(1-k)^2}8$（`--c-perp`，虛線，標「垂直」）
    - 總和（`--ink`，粗實線，標「總和」）
    - $k^*$ 處畫 `--accent` 實心點與垂直虛線，標「k* = 0.800」。
  - 讀數表（絕對值 $\Delta^*$，Eq. 54d；「比 k* 多」用〔本頁推導〕$\frac{D+d}8(k-k^*)^2$）：列 ε（k = 0）、v（0.5）、x（1）、k*；欄 k、Δ*、比 k* 多。數字用千分位（例如 $D=65{,}536$ 時 Δ* 很大）。
  - $k^*$ 顯示：一般三位小數；$k^*>0.999$ 時改為顯示到第一個非 9 的小數位為止，最多 6 位（例：$D=65536,d=1$，$k^*=65536/65537=0.9999847\ldots$，顯示 0.99998）。
  - 小圖：橫軸 $\rho=d/D$（對數，$10^{-5}$ 到 1），縱軸 $k^*=1/(1+\rho)\in[0.5,1]$，標出目前的點；在 $\rho=1$ 標「v」，在左端標「→ x」。
- **預設值下的正確輸出**（驗收）：$k^*=0.800$；$\Delta^*$：ε 22.000、v 11.000、x 10.000、k* 9.200；比 k* 多：12.800、1.800、0.800、0.000。
- **讀者應觀察到**：
  1. 同比例放大 $D$ 與 $d$，$k^*$ 不變（只看 $d/D$）。
  2. $d=D$ 時垂直項消失，$k^*=0.5$；$d\ll D$ 時垂直項主導，$k^*\to1$。
  3. $d<D$ 時 $0.5<k^*<1$，ε、v、x 都不是最小值。

### C3 · 訓練設定如何移動最佳 k（S7）

- **教的概念**：閉式解只在 Theorem 3.2 條件下成立；時間條件（N 區間、雜訊強度相依）、取樣分布、損失都會移動母體最佳值；母體最佳值與實際訓練值可能不同（Sec. 3.4、Fig. 1、App. H）。
- **簡化**（頁面寫出）：只計算母體最佳值，不訓練網路；不含時間嵌入 MLP；logit-normal 用推定參數 μ = −0.8、s = 0.8，頁面註明「論文 Sec. 3.4 未寫明參數；此值與 JiT 設定相同，算出的結果與論文報告的 $d\ge48$ 時 $k=0$、$d=128$ 八區間 0.212 一致，屬於用論文數字校準的推定值」。
- **計算**：
  - 對固定 $k$，$\Delta^*(k,d)=d\cdot P(k)+(D-d)\cdot Q(k)$，$P,Q$ 與 $d$ 無關。$\varphi=k,\psi=-(1-k)$ 在一個區間內是常數，因此先求區間內的加權和 $S_0=\sum w,\ S_\alpha=\sum w t,\ S_\sigma=\sum w(1-t),\ S_{\alpha\alpha}=\sum wt^2,\ S_{\sigma\sigma}=\sum w(1-t)^2$（$w_j=p(t_j)\kappa^2(t_j)/M$），再用因式分解後的形式（避免 $k\to1$ 時兩個接近 0 的大數相減）：
    - $P_b=\tfrac12\big[(\varphi^2+\psi^2)S_0-(\varphi S_\alpha+\psi S_\sigma)^2/(S_{\alpha\alpha}+S_{\sigma\sigma})\big]$
    - $Q_b=\tfrac12\,\psi^2\big[S_0-S_\sigma^2/S_{\sigma\sigma}\big]$
    - 分母 $<10^{-300}$ 時該商以 0 計；$P_b,Q_b$ 取 $\max(0,\cdot)$。
    - 共享線性：$N=1$；$N$ 區間：$P=\sum_bP_b,\ Q=\sum_bQ_b$（「summed population loss」，Sec. 3.4）。
    - 雜訊強度相依（Eq. 77c）：$P=\tfrac12\sum_jw_j\frac{(\varphi\sigma_j-\psi\alpha_j)^2}{\alpha_j^2+\sigma_j^2}$，$Q=0$。
  - $k$ 網格 0 到 1、步長 0.005（201 點）；每個 $d\in\{1,\dots,128\}$ 取使 $\Delta^*$ 最小的網格點（同值取最小的 $k$）。
  - 平坦偵測：若某 $d$ 的 $\max_k\Delta^*-\min_k\Delta^*<10^{-9}\cdot\max_k|\Delta^*|$，該曲線不畫，改在圖例顯示「損失與 k 無關，k 無法由損失決定」（設定 C + 雜訊強度相依必然觸發，〔本頁推導〕）。
  - 每次換設定或 $N$ 重算一次，結果放進快取。
- **讀者操作**：
  - 導覽步驟列（四個按鈕，預設停在第 1 步；按「下一步」前進，也可直接點任一步）：
    1. 「Theorem 3.2 的條件」：設定 A，只顯示共享線性與 $D/(D+d)$。
    2. 「加入時間條件」：設定 A，顯示 N 區間（預設 $N=8$）與雜訊強度相依曲線，$N$ 控制項出現。
    3. 「改用 logit-normal 取樣」：設定 B，有效權重副圖出現。
    4. 「再改用 v-MSE」：設定 C，論文報告的三個點出現。
  - 第 2 步之後可自由操作：分段按鈕「訓練設定」A / B / C；分段按鈕「時間區間數 N」1、2、4、8、16、32、64；勾選框「顯示雜訊強度相依模型（App. H）」。
  - 選取 $d$：滑桿 $d\in[1,128]$，預設 32（鍵盤與觸控皆可用），主圖上同步顯示垂直指標線與 tooltip；桌機滑鼠移動也可改變 $d$。tooltip 固定在圖版右上角的讀數框，不跟隨指標，觸控不會被手指遮住。
  - 副圖滑桿 $k$（預設 0.5），只在設定 B、C 顯示。
- **顯示**：
  - 主圖：橫軸 $d\in[1,128]$（$D=128$ 固定），縱軸 $k^*\in[0,1]$（左側標 ε/v/x 刻度）。曲線以線型＋直接標籤區分：$D/(D+d)$（`--ink-3` 細虛線，標「D/(D+d)」）、共享線性（`--accent` 實線，標「共享」）、N 區間（`--c-par` 長虛線，標「N 區間」）、雜訊強度相依（`--c-perp` 點線，標「雜訊相依」）。
  - 設定 C 時另外畫論文報告的三個點（空心標記，各自標文字）：八區間母體最佳 0.212、訓練出的八區間模型 0.659、時間嵌入 MLP 0.612，標題「論文報告（d = 128）」。
  - 主圖下方一行文字顯示該設定在論文中的對應敘述與誤差（照 1.4 步驟 8 表格）。
  - 副圖「有效權重 $\widetilde{D\alpha}(t)=p(t)\kappa^2(t)$」（只在 B、C）：橫軸 $t$，曲線除以最大值正規化到 [0,1]。B 顯示權重集中在 $t<0.5$；C 的 $\kappa^2$ 再依 $k$ 改變權重分布。
- **驗收數值**（`check6.py`：$M=1024$、步長 0.005，與頁面算法相同，應完全一致；容許 ±0.005 以涵蓋浮點差異）：

  | 設定 | 去噪器 | d=1 | 4 | 16 | 32 | 48 | 64 | 96 | 128 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | A | 共享 | 0.990 | 0.970 | 0.890 | 0.800 | 0.725 | 0.665 | 0.570 | 0.500 |
  | A | 8 區間 | 0.960 | 0.870 | 0.695 | 0.605 | 0.565 | 0.540 | 0.515 | 0.500 |
  | A | 雜訊相依 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
  | B | 共享 | 0.935 | 0.775 | 0.375 | 0.115 | 0.000 | 0.000 | 0.000 | 0.000 |
  | B | 8 區間 | 0.455 | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
  | B | 雜訊相依 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
  | C | 共享 | 0.995 | 0.985 | 0.940 | 0.845 | 0.710 | 0.545 | 0.240 | 0.080 |
  | C | 8 區間 | 1.000 | 0.995 | 0.980 | 0.955 | 0.915 | 0.840 | 0.480 | 0.210 |
  | C | 雜訊相依 | 平坦 | 平坦 | 平坦 | 平坦 | 平坦 | 平坦 | 平坦 | 平坦 |

  設定 A、$d=32$ 的區間數掃描：N = 1, 2, 4, 8, 16, 32, 64 → 0.800, 0.730, 0.670, 0.605, 0.560, 0.535, 0.515。（共享線性 A 與 $D/(D+d)$ 的差異只來自 0.005 網格。）
- **讀者應觀察到**：
  1. 第 1 步：共享線性與 $D/(D+d)$ 重合（Theorem 3.2）。
  2. 第 2 步：區間數越多，母體最佳 $k$ 越接近 0.5；雜訊強度相依時恆為 0.5〔本頁推導：去掉垂直殘差後剩下的平行項 $(\varphi\sigma-\psi\alpha)^2/(\alpha^2+\sigma^2)$ 對 $k\leftrightarrow1-k$ 對稱〕。頁面在此加一句：「這是線性模型的母體結果，不代表實際網路；時間條件化的 JiT-B/16 在像素空間學到接近 1.0（Sec. 5.2）。」
  3. 第 3 步：logit-normal 使高 $d$ 的母體最佳值落到 $k=0$（論文：$d\ge48$）。
  4. 第 4 步：v-MSE 使最佳值離開邊界；$d=128$ 時八區間母體最佳 0.212（本頁算出 0.210，網格差異），但論文訓練出的模型是 0.659／0.612。母體分析在這裡失準，這是論文改為直接訓練 $k$ 的理由。

### C4 · k-Diff 的玩具版：k 作為可訓練參數（S8）

- **教的概念**：k-Diff 把 $k=\mathrm{sigmoid}(w_k)$ 和去噪器一起用梯度下降訓練（Eq. 15）；不需要知道 $d$，$k$ 會走到該設定下的母體最佳值；學到的值取決於取樣分布與損失。
- **簡化**（頁面寫出）：
  - 用母體損失（對 $x,n$ 取期望、對 $t$ 用 $M=1024$ 中點積分），不抽樣本，因此軌跡是確定的。
  - 去噪器限制為 $W=a\,PP^\top+b\,(I-PP^\top)$：從 $W=0$ 出發時，Eq. 8 的梯度只落在這兩個矩陣的線性組合中，所以這個限制不改變梯度流的解〔本頁推導〕。$a$ 是平行模式係數、$b$ 是垂直模式係數。
  - 只做共享線性去噪器、$D=128$。
- **損失**（每個環境維度；由 $Wz-u=(a\alpha-\varphi)x+(a\sigma-\psi)n_\parallel+(b\sigma-\psi)n_\perp$ 取期望）：
  $$\mathcal L=\frac{1}{2D}\sum_j w_j\,\ell_j,\quad \ell_j=d\big(e_{1j}^2+e_{2j}^2\big)+(D-d)\,e_{3j}^2,$$
  $$e_1=at-k,\quad e_2=a(1-t)+(1-k),\quad e_3=b(1-t)+(1-k),\quad w_j=p(t_j)\kappa^2(t_j)/M.$$
- **梯度（必須用解析式）**：
  - $\partial\mathcal L/\partial a=\frac1{2D}\sum_jw_j\cdot2d\,(e_1t+e_2(1-t))$
  - $\partial\mathcal L/\partial b=\frac1{2D}\sum_jw_j\cdot2(D-d)\,e_3(1-t)$
  - $\partial\mathcal L/\partial k=\frac1{2D}\sum_j\big[w_j\big(-2d(e_1+e_2)-2(D-d)e_3\big)+w'_j\,\ell_j\big]$，其中設定 A、B 的 $w'_j=0$；設定 C 的 $w'_j=-2p(t_j)(1-2t_j)/\big(M\,(k(1-t_j)+(1-k)t_j)^3\big)$（$\kappa^2$ 隨 $k$ 的變化，對應 Alg. 1 中 `v_pred` 的分母依賴 $k$）。
  - $\partial\mathcal L/\partial w_k=\partial\mathcal L/\partial k\cdot k(1-k)$。
  - 實作只有一條路徑：每步對 1024 個點直接迴圈一次（30 步／影格 ≈ 3 萬次迴圈，不需要預先算動差）。
- **最佳化**：Adam，$\beta_1=0.9,\beta_2=0.95$（Table 3 的值），$\epsilon=10^{-8}$，含偏差修正；學習率固定 0.02（玩具用值；論文網路用 2e-4，頁面註明不同）；不使用學習率衰減：已驗證固定學習率下 $k$ 在第 1000 步與第 3000 步的差異 ≤ 0.001（見下表），沒有可見抖動。初始 $a=b=w_k=0$（$k=0.5$，與 Fig. 3b 相同）。總步數 3000，每個動畫影格 30 步（約 1.7 秒完成）。
- **讀者操作**：
  - 分段按鈕「訓練設定」：A / B / C（預設 A）。
  - 滑桿 $d\in\{1,\dots,128\}$，預設 32。
  - 按鈕：「執行」／「暫停」、「重設」。改變設定或 $d$ 時自動重設。
- **顯示**：
  - 左圖：$k$ 對步數（橫軸 0–3000）。水平虛線標「C3 網格值（步長 0.005）」：同一設定與 $d$ 下 C3 的共享線性母體最佳值（直接呼叫 C3 的計算與快取）；設定 A 另標 $D/(D+d)$。C4 以連續參數最佳化，終值與網格值可差到半個網格（例：B、$d=32$ 為 0.117 對 0.115；C、$d=32$ 為 0.847 對 0.845），圖下以一行文字說明此差異在網格解析度內。右側附一條直立 k 軸，標出目前的 $k$。
  - 右圖：$a$（`--c-par` 實線，標「a（平行）」）、$b$（`--c-perp` 虛線，標「b（垂直）」）對步數，以及依目前 $k$ 由 Eq. 11 算出的平衡值 $a^*(k)=\frac{\sum w(\varphi\alpha+\psi\sigma)}{\sum w(\alpha^2+\sigma^2)}$、$b^*(k)=\frac{\sum w\psi\sigma}{\sum w\sigma^2}$（細點線，隨 $k$ 移動）。$d=D$ 時 $b$ 沒有對應維度，標「d = D，無垂直維度」。
  - 讀數：步數、$k$、$\mathcal L$、$a$、$b$。
  - 圖下對照 Alg. 1：「這個玩具省略了 `sample_t` 與 `randn_like`（改成取期望），保留了 `k = sigmoid(w_k)`、`u = k*x - (1-k)*e` 與 v-MSE 下依賴 $k$ 的換算分母」。
- **驗收數值**（`check6.py`，解析梯度，與頁面算法相同；第 100 / 300 / 1000 / 3000 步的 $k$，與第 3000 步的 $a,b,\mathcal L$）：

  | 設定 | d | k@100 | k@300 | k@1000 | k@3000 | a | b | 𝓛 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | A | 4 | 0.783 | 0.966 | 0.970 | 0.970 | 0.705 | −0.045 | 0.0096 |
  | A | 32 | 0.749 | 0.800 | 0.800 | 0.800 | 0.450 | −0.300 | 0.0719 |
  | A | 64 | 0.667 | 0.666 | 0.667 | 0.667 | 0.250 | −0.499 | 0.1354 |
  | A | 128 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.2500 |
  | B | 4 | 0.656 | 0.774 | 0.774 | 0.774 | 0.174 | −0.319 | 0.0112 |
  | B | 32 | 0.407 | 0.117 | 0.117 | 0.117 | −0.909 | −1.249 | 0.0522 |
  | B | 64 | 0.248 | 0.017 | 0.000 | 0.000 | −1.100 | −1.415 | 0.0793 |
  | C | 4 | 0.680 | 0.956 | 0.987 | 0.987 | 0.748 | −0.020 | 0.0299 |
  | C | 32 | 0.653 | 0.847 | 0.847 | 0.847 | 0.433 | −0.234 | 0.2373 |
  | C | 64 | 0.600 | 0.543 | 0.543 | 0.543 | −0.194 | −0.652 | 0.4663 |
  | C | 128 | 0.390 | 0.082 | 0.082 | 0.081 | −1.011 | 0.000 | 0.8716 |

  容許誤差：$k$ ±0.002、$a,b$ ±0.002、$\mathcal L$ ±0.0002（同一算法，只差浮點加總順序）。交叉檢查：A、$d=32$ 的 $\mathcal L=0.0719=9.2/128$，與 Eq. 54d 一致。
- **讀者應觀察到**：
  1. 沒有輸入 $d$，$k$ 仍從 0.5 走到 C3 算出的母體最佳值（設定 A 即 $D/(D+d)$）。
  2. 換取樣分布或損失，同一份資料學到的 $k$ 不同（$d=32$：A 0.800、B 0.117、C 0.847）。
  3. $k$ 收斂後，$a,b$ 停在 Eq. 11 的平衡值。
- **為什麼不顯示兩模式的速度差**：鬆弛速率比較（均勻、u-loss 時 $2/3$ vs $1/3$，〔本頁推導〕）只對純梯度下降成立；這裡用 Adam（與論文訓練設定一致並在 3000 步內收斂），Adam 逐參數正規化步長，速度比較不再有意義。S4 文字中可提一句速率比較。

---

## 5. 實作計畫

### 5.1 檔案與函式庫

- 單一檔案 `tut/238/build/index.html`；CSS、JS、時間線資料全部內嵌。不得 `fetch` 本機檔案。上線時由 PM 複製為 `tutorials/prediction-targets-k-diff.html`。
- 外部資源只允許：
  - Google Fonts（`&display=swap`）：Newsreader 400/500/600、Noto Serif TC 600/700、IBM Plex Sans 400/500/600、Noto Sans TC 400/500/700、JetBrains Mono 400/500。每個 font-family 都附系統 fallback（見 5.2）。
  - KaTeX 0.16.9（cdnjs）：`katex.min.css`、`katex.min.js`、`contrib/auto-render.min.js`，加 `defer`。分隔符 `$…$`（行內）與 `$$…$$`（區塊）。KaTeX 載入失敗時保留原始 TeX 文字，頁面其餘功能不受影響。
- 不用 D3：所有圖都是手寫 SVG，線性與對數比例尺各寫一個小函式即可。
- 頁面重量：建置後用瀏覽器 DevTools 在模擬行動網路（Fast 4G）下確認首次載入的傳輸量；HTML 本身目標 < 200 KB（不含字型與 KaTeX）。中文字型由 Google Fonts 分片載入，不另外子集化。

### 5.2 設計方向與 tokens（內嵌自 DESIGN.md，建置者不需另外查閱）

- 依 AGENTS.md 步驟 6-3：Anthropic 官網的編輯式排版 + Material Design 3 的元件形狀與狀態層；以字面敘述為主，不用比喻與華麗辭藻；以 Awwwards／Webby／FWA 得獎作品為視覺標竿；完成後逐項自我檢查（5.9）。
- 字型：

  | 用途 | 字型 | fallback |
  | --- | --- | --- |
  | 標題（拉丁） | Newsreader 400/500/600 | Georgia, serif |
  | 標題（中文） | Noto Serif TC 600/700 | "Songti TC", "PMingLiU", serif |
  | 內文 | IBM Plex Sans + Noto Sans TC | "PingFang TC", "Microsoft JhengHei", system-ui, sans-serif |
  | 編號、數值、metadata | JetBrains Mono 400/500 | ui-monospace, Menlo, monospace |

  字級（px）：12 / 13 / 15 / 17 / 20 / 24 / 32 / 44 / 64。圖上標籤 13px，最小 12px。標題與內文字級比約 4:1（Hero 64 對內文 17）。
- 色彩：

  | token | light | dark | 用途 |
  | --- | --- | --- | --- |
  | `--bg` | #F6F4EE | #161513 | 頁面底色 |
  | `--surface` | #FFFFFF | #1F1E1B | 圖版、卡片 |
  | `--surface-2` | #EEEBE3 | #2A2925 | 輸入元件、hover |
  | `--ink` | #1D1C19 | #EEEBE4 | 主文字 |
  | `--ink-2` | #57534B | #B5B0A5 | 次要文字 |
  | `--ink-3` | #67635A | #8E897F | metadata（AA 對比） |
  | `--line` | #DCD7CB | #36342F | 分隔線 |
  | `--accent` | #B4532A | #E58C66 | 唯一介面強調色：選取、連結、focus、k* |
  | `--accent-soft` | #F3E2D8 | #3B2820 | 選取背景 |
  | `--c-par` | #3F6E8C | #8DB4D3 | 圖表專用：平行模式、資料方向 |
  | `--c-perp` | #6B7A3A | #A9BC7C | 圖表專用：垂直模式、噪聲方向 |

  `--c-par`、`--c-perp` 是本頁新增（DESIGN.md 只定義介面色），只用在圖中資料系列。實作時確認兩色在 `--surface` 上對比 ≥ 3:1（圖形）、作為文字標籤時 ≥ 4.5:1，不足就調整明度並回寫此表。
- 間距（4px 基準）：4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 / 96。圓角：6（tag）、10（輸入框、按鈕）、16（卡片、圖版）、999（pill）。陰影只用在浮動層（tooltip、固定導覽列）。
- 動態：160ms（hover）／240ms（狀態切換）／400ms（捲動引導）；easing `cubic-bezier(.2,0,0,1)`。每個動畫都有功能（回饋、狀態改變、引導視線）。
- 深色模式：`@media (prefers-color-scheme: dark)` 內以 `:root:not([data-theme="light"])` 覆寫，另以 `:root[data-theme="dark"]` 覆寫一次；頁首提供主題切換鈕，偏好存在 `localStorage` 的 `pm-theme`（讀寫都包 try/catch，失敗時跟隨系統）。`body` 設定 `background: var(--bg)`。
- `prefers-reduced-motion: reduce`：取消所有 transition 與 smooth scroll；C4 的處理見 5.4。
- 響應式：390–1440px。390px 時 16px 側邊留白、無水平捲動；元件控制列在圖下方堆疊；SVG 用 `viewBox` 等比縮放，圖上文字最小 12px（窄螢幕減少刻度數量而不是縮小字）。768px 以上控制列可放圖右側。滑桿與按鈕觸控區 ≥ 44px。四欄以上的表格（Table 1、Table 3、C2 讀數表）包在 `overflow-x:auto` 的容器中並加左右漸層提示；< 480px 時 Table 3 改為每個項目一張堆疊卡片（項目名、潛空間、像素空間三行）。
- 頁首 appbar：sticky，高 64px，背景 `color-mix(in oklab, var(--bg) 86%, transparent)`，捲動後加 1px `--line` 底線；內容為頁面短標題與主題切換鈕。
- 固定導覽列（照錄 benchmark 的結構，顏色改用 tokens）：

  ```html
  <nav id="pm-bar" aria-label="Paper Master">
    <a href="../index.html">← Paper Master</a>
    <a href="index.html">教學與筆記</a>
  </nav>
  ```
  樣式：`position:fixed; left:16px; bottom:16px; z-index:9999; display:flex; gap:8px; font:500 13px/1 'IBM Plex Sans','Noto Sans TC',system-ui,sans-serif`；兩個連結都是 pill（`padding:10px 14px; border-radius:999px; min-height:20px; display:inline-flex; align-items:center; text-decoration:none`），第一個 `background: var(--ink); color: var(--bg)`，第二個 `background: var(--bg); color: var(--ink); border:1px solid var(--line)`，都加 `box-shadow:0 4px 16px rgba(0,0,0,.15)`。頁面底部預留 72px，避免導覽列遮住頁尾。
- 其他站內連結：Hero「在關聯圖中查看」→ `../index.html#paper-238`（關聯圖以 `#paper-<id>` 定位論文）；arXiv `https://arxiv.org/abs/2601.21419`；研究筆記 → `<slug>.html`（同在 `tutorials/`）。

### 5.3 頁面結構

```
<header class="appbar">（短標題、主題切換）
<nav class="toc">（章節目錄，桌機 sticky 左欄／窄螢幕頂部橫列）
<main>
  <section id="top"> … <section id="basics"> … <section id="next">
<footer>（引用、arXiv）
<nav id="pm-bar">
<script type="application/json" id="timeline-data">…</script>
<script>（共用數學 → 元件 → 時間線 → KaTeX 呼叫 → selfTest）</script>
```

### 5.4 程式結構與函式名稱

全部寫在一個 `<script>`，以 IIFE 包住，不建立框架或類別階層。

共用數學：
- `const M = 1024; const TGRID = Float64Array.from({length:M}, (_, j) => (j + 0.5) / M);`
- `const SETTINGS = { A:{sampler:'uniform', loss:'u'}, B:{sampler:'logitn', loss:'u'}, C:{sampler:'logitn', loss:'v'} };`
- `const LOGITN = { mu: -0.8, s: 0.8 };`
- `DENS = { uniform: Float64Array, logitn: Float64Array }`：預先算好的 $p(t_j)$。
- `weights(setting, k)` → Float64Array $w_j=p(t_j)\kappa^2(t_j)/M$。
- `modeParts(setting, k, nBins)` → `{P, Q}`；`nBins = 0` 代表雜訊強度相依。實作依 4.0 與 C3 的因式分解式、分母保護與 $\max(0,\cdot)$。
- `kStarCurve(setting, nBins)` → `{ks: Float64Array(128), flat: boolean[]}`，索引 `d-1`。內部先對 201 個 $k$ 求 `{P,Q}`，再對每個 $d$ 取最小並做平坦偵測。
- `closedForm(D, d, k)` → `{par, perp, total}`（Eq. 54c，未除以 $D$）。
- `kStarClosed(D, d)` → $D/(D+d)$。
- `rng(seed)`：mulberry32；`gauss(r)`：Box–Muller。

元件（每個元件一個 init 函式，各自只碰自己的 DOM 容器）：
- C1：`initC1()` 負責 `#c1a` 與 `#c1b`；共用 `c1 = {seed, pts:[{xt, np, nq}]}`；`renderC1a(t)`、`renderC1b(t, k)`。
- C2：`initC2()`，`state = {D, d}`，`renderC2()`，`D_VALUES` 陣列，`fmtK(k)`（4 位以上小數規則）、`fmtNum(x)`（`toLocaleString('zh-TW', {maximumFractionDigits:3})`）。
- C3：`initC3()`，`state = {step, setting, nBins, showDep, d, kSide}`，`cache = new Map()`（key `${setting}|${nBins}`），`renderC3()`。
- C4：`initC4()`，`state = {setting, d, step, a, b, w, m:[0,0,0], v:[0,0,0], hist:{k:[], a:[], b:[]}, running}`；`gradC4(state)` → `{g:[ga,gb,gw], L}`（解析式）；`stepC4(state)`（一次 Adam 步）；`runC4()` 用 `requestAnimationFrame`，每影格 30 步。reduced-motion 時：按「執行」以 `setTimeout(…, 0)` 分段，每段 500 步，全部完成後一次重畫（3000 步總運算約 300 萬次乘加，不會長時間阻塞，分段只是保險）。
- SVG 輔助：`el(tag, attrs)`、`scaleLin(d0, d1, r0, r1)`、`scaleLog(d0, d1, r0, r1)`、`linePath(points)`、`axis(...)`。只寫實際用到的。
- 時間線：`renderTimeline(data)`（5.7）。

效能：C3 每次重算約 201 × 1024 × 2 次運算（雜訊相依另一次），同步完成；C4 若單影格超過 12ms，改為每影格 10 步。

自我檢查（非平凡數值邏輯留一個可執行的檢查）：
- `selfTest()`：網址含 `?test` 時執行，用 `console.assert` 比對：
  - `closedForm(128, 32, k).total` 在 $k=0,0.5,1,0.8$ 為 22、11、10、9.2（誤差 1e-9）。
  - `kStarCurve('A',1)` 在 $d=32$ 為 0.800；`kStarCurve('A',8)` 在 $d=4$ 為 0.870；`kStarCurve('B',1)` 在 $d=48$ 為 0；`kStarCurve('C',8)` 在 $d=128$ 為 0.210；`kStarCurve('C',0).flat` 全為 true（誤差 ±0.005）。
  - C4 設定 A、$d=32$ 跑 1000 步後 $k=0.800\pm0.002$；設定 C、$d=32$ 跑 3000 步後 $k=0.847\pm0.002$。
  - 結果以一行 `console.log('selfTest: N passed, M failed')` 輸出。

### 5.5 數學排版

- 公式旁以等寬字標註論文編號，例如右對齊的「Eq. 14」。`.katex-display{overflow-x:auto; overflow-y:hidden}`，避免 390px 時長公式撐破版面；Eq. 12 在 < 600px 時拆成兩行（平行項、垂直項各一行）。
- KaTeX 不處理 SVG 內文字：圖上標籤、tooltip、讀數用 Unicode（例：「k* = 0.800」「Δ*/D」「d/D」），不用 TeX。
- 動態插入的 DOM（`<details>` 展開的檢查題答案、C3 的論文敘述行）若含 TeX，插入後呼叫 `renderMathInElement(node, opts)`。
- 頁面中的所有方程式與數字必須與第 1 節一致；第 1 節以外不得新增論文數字。

### 5.6 引用與標籤規則

- 內文：每個論文數字後註明出處（Eq./Table/Fig./Sec.）。
- 互動元件的讀數不逐一加出處；每個元件圖版底部放一行說明：「讀數由本頁依 Eq. ○○ 即時計算」，若用到〔本頁推導〕的式子，再加「本頁推導」標籤與一行推導依據。
- 〔本頁推導〕在內文用小標籤「本頁推導」（`--ink-3`，邊框 pill）標示，並附一行依據（例如「由 Eq. 17 相減」）。
- 區分三種陳述：論文結果、論文作者的解釋（用「作者認為」「作者歸因於」）、本頁推導。

### 5.7 研究脈絡（由 notes/timeline.json 提供）

- 來源：`tut/238/notes/timeline.json`。建置時把檔案內容原樣貼進 `<script type="application/json" id="timeline-data">`，頁面用 `JSON.parse` 讀取，不手改條目。
- 格式（建置時以實際檔案為準）：陣列，每筆
  ```json
  {"year": 2020, "title": "Denoising Diffusion Probabilistic Models", "authors_short": "Ho, Jain, Abbeel",
   "arxiv": "2006.11239", "role": "以噪聲預測（ε-prediction）訓練，本論文視為早期標準目標（k = 0）。",
   "source": "2601.21419 related work"}
  ```
  目前 12 筆：1503.03585、2006.11239、2011.13456、2202.00512、2206.00364、2210.02747、2303.00848、2403.03206、2410.19324、2402.18491、2511.13720、2601.21419（本文，作者欄為「本論文」）。
- 分類（依各條目的 `role` 文字人工對應，寫成頁面中的常數 `TL_GROUP`，以 arXiv ID 為 key；不在表中的條目歸「其他」）：

  | 類別 | 標籤 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- | --- |
  | 基礎 | 擴散模型 | 1503.03585 | `prediction-targets-overview.html` |
  | ε（k = 0） | 預測噪聲 | 2006.11239、2011.13456 | `eps-prediction-ddpm-score.html` |
  | v（k = 0.5） | 預測速度 | 2202.00512、2206.00364 | `v-prediction-and-preconditioning.html` |
  | v（k = 0.5） | 預測速度 | 2210.02747、2403.03206 | `flow-matching-velocity-target.html` |
  | 加權 | 損失加權 | 2303.00848 | `loss-weighting-and-elbo.html` |
  | x（k = 1） | 預測資料 | 2410.19324、2511.13720 | `x-prediction-manifold-view.html` |
  | 理論 | 動態分析 | 2402.18491 | `diffusion-dynamics-theory-background.html` |
  | 本文 | 連續的 k | 2601.21419 | `generalized-target-k-parameterization.html` |

- 呈現：
  - 水平捲動的卡片列（`overflow-x:auto`，`scroll-snap-type:x mandatory`），依 `year` 排序（同年保持檔案順序）。卡片：年份（JetBrains Mono）、類別標籤（pill，ε／v／x 三類在標籤上附 k 值）、標題、作者、`role`（最多三行，超出以 `line-clamp` 截斷，完整文字放 `title` 屬性）、兩個連結「arXiv」→ `https://arxiv.org/abs/{arxiv}` 與「筆記」→ 上表的 slug。
  - 本文卡片以 `--accent` 外框與「本文」標籤突出。若 JSON 中沒有 `2601.21419`，在列尾自動加一張本文卡片（年份 2026，role 取 1.1 第一句）。
  - 類別篩選 chip（`aria-pressed`），點選後未選類別的卡片淡化（opacity .35），不移除。
  - 左右按鈕（`aria-label`「往前捲動」「往後捲動」），鍵盤左右鍵可捲動；reduced-motion 下捲動行為為 `auto`。
  - 缺欄位：缺 `arxiv` 不顯示 arXiv 連結；缺 `role` 只顯示標題與作者；JSON 解析失敗時整節顯示「時間線資料尚未提供」並隱藏控制鈕。
- S12「延伸閱讀」依總覽的閱讀順序列出：`prediction-targets-overview.html`（總覽）、`eps-prediction-ddpm-score.html`、`v-prediction-and-preconditioning.html`、`flow-matching-velocity-target.html`、`loss-weighting-and-elbo.html`、`x-prediction-manifold-view.html`、`generalized-target-k-parameterization.html`、`diffusion-dynamics-theory-background.html`；以及站內既有的 `pixel-space-diffusion.html`、`velocity-irreducible-variance.html`。每項附筆記 frontmatter 的 `title`。

### 5.8 文案規則

- 繁體中文、台灣用語、字面學術語氣；不用反問句、比喻、宣傳形容詞。章節標題用陳述句。
- 用詞：「權衡」（不用「拉鋸」）；「k 作為可訓練參數」（不用「讓 k 自己學」）；「k 軸」（不用「k 尺」）；「常見錯誤」（不用「陷阱」）。
- 專有名詞第一次出現附英文（例如「內在維度（intrinsic dimension）」）。

### 5.9 無障礙

- 每個滑桿有 `<label>` 與 `aria-valuetext`（例如「k = 0.80」）。
- SVG 圖 `role="img"` 並附 `aria-label` 描述目前狀態；`aria-label` 與讀數區的 `aria-live="polite"` 只在 `change` 事件（放開滑桿）或輸入停止 500ms 後更新，不在每次 `input` 更新。
- C4：動畫過程不更新 live region，只在完成或暫停時宣告一次「第 N 步，k = …」。
- 平行／垂直、各曲線一律以線型或標記形狀＋直接文字標籤區分，不只靠顏色。
- focus 樣式：2px `--accent` 外框、2px 位移。
- 表格有 `<caption>`；檢查題 `<details>` 的 `<summary>` 為「檢查理解」。

### 5.10 建置後自我檢查（AGENTS.md 步驟 6-3）

截圖寬度 390 / 768 / 1024 / 1440，加深色模式、鍵盤 focus、reduced-motion 各一張；依字體、留白、視覺層級、色彩、動態、微互動、響應式、原創性逐項評分，有差距就修正再檢查，直到沒有可見的改進空間。另跑 `?test` 確認 selfTest 全數通過，逐條核對第 2 節 12 個學習目標，並確認 Hero k 軸在 358px 內容寬時標籤不重疊。

---

## 6. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | Fig. 1 的 logit-normal 參數 | 論文 Sec. 3.4 只寫「logit-normal」，未給 μ、σ。〔本頁推導〕對稱分布 + u-MSE 時最佳值仍是 $D/(D+d)$，無法得到論文的 $k=0$；μ = −0.8、σ = 0.8（Table 3 的 JiT 像素空間設定）算出的值與 $d\ge48$ 時 $k=0$、$d=128$ 八區間 0.212 一致；μ = −0.8、σ = 1.0 在 $d=48$ 給出 0.137，不符。 | 頁面採 μ = −0.8、σ = 0.8，明寫「用論文數字校準的推定值，不是論文寫明的設定」。 |
| R2 | 論文數字的不完整性 | Fig. 1 只給文字描述的誤差（0.027、0.025、0.012、0.026、0.045）與 $d=128$ 的三個值；11 個 $d$ 值與 MLP 結果沒有數據。 | C3 只畫母體最佳值與論文明確給出的三個點，不重繪論文的實驗標記。 |
| R3 | $k$ 值的出處 | Table 2 為第 64 epoch 的 0.522；0.66 是 Fig. 3b（正規化 epoch 橫軸）的長期收斂值；800 epoch 屬於 Fig. 3a 的 FID 曲線。 | S9 與 LO12 寫兩個值並各標出處，不把 0.66 寫成「800 epoch 的值」。 |
| R4 | 論文未說明的實作細節 | $w_k$ 的學習率、EMA 是否套用在 $w_k$；截斷 0.05 在 Sec. 4 寫作「分母」，在 Table 3 寫作「clip of (1 − t) in division」。 | 常見錯誤第 2、4 條照錄兩處原文並寫明未說明之處。 |
| R5 | Table 1 的 512 結果與 epoch | JiT-B/32：FID 4.02 → 4.03（略高）、IS 271.0 → 277.1（較高）；論文文字寫「consistently matches or outperforms」。Table 1 像素空間列的訓練 epoch 未寫明。 | 頁面照錄數字，不寫「全面優於」；不註明像素空間列的 epoch。 |
| R6 | 圖表色 | DESIGN.md 只定義單一強調色；本頁需要平行／垂直兩個資料系列色。 | 新增 `--c-par`、`--c-perp`，僅用於圖中資料系列；實作時驗證對比並回寫 5.2。 |
| R7 | 相對連結 | 建置位置 `tut/238/build/index.html` 與上線位置 `tutorials/prediction-targets-k-diff.html` 不同；所有站內連結以上線位置撰寫，在建置位置預覽時會失效。 | 接受；預覽時只檢查外部連結與頁內錨點。 |
| R8 | 筆記與時間線可能更新 | 建置前 notes 若有增減，`TL_GROUP` 與延伸閱讀清單需同步。 | 建置時重新讀取 `notes/`；新增的 arXiv ID 未在 `TL_GROUP` 中則歸「其他」並在建置回報中列出。 |
| R9 | 元件效能 | 低階手機上 C3 重算與 C4 動畫可能掉影格。 | C3 快取；C4 單影格超過 12ms 改為 10 步／影格。 |
| R10 | 初學者門檻 | 讀者不熟擴散模型；S1 必須在不需要 $k$、$u$ 的情況下獨立可讀。 | C1 已拆成 C1a（S1，只有 $t$）與 C1b（S5）；Goldfish 審查時檢查 S1。 |
| R11 | 本頁推導的範圍 | 頁面含 7 項〔本頁推導〕（κ_v、超出損失、對稱性 k* = 0.5、v-MSE 平坦、對稱取樣、v = x − e、W 的兩參數限制）；讀者可能誤認為論文結論。 | 一律加「本頁推導」標籤與依據；C3 第 2 步明寫這是線性模型母體結果。 |
| R12 | C3 與論文 0.212 的網格差異 | 頁面 $k$ 步長 0.005 算出 0.210，論文為 0.212。 | 頁面同時顯示兩者並註明差異來自網格；不把步長改細（201 點已足以呈現趨勢）。 |

---

## 7. 修訂紀錄（round 1）

| 項目 | 變更 |
| --- | --- |
| 1 | 步驟 9 新增「損失空間的規則」：Eq. 15 的 u 空間＋$\widetilde{D\alpha}$ 與 Alg. 1 v-loss 等價；u-MSE／v-MSE 只指 κ=1 與 κ_v；C3、C4 依此（4.0、C4 梯度含 $w'_j$）。 |
| 2 | 步驟 9、常見錯誤 2、LO11 統一照錄 Sec. 4 與 Table 3（像素 0.05；潛空間 x-pred 0.05、其他 0）；C3/C4 不截斷，理由與 `check6.py` 驗證（截斷後結果不變）寫在步驟 9。 |
| 3 | 步驟 9 與常見錯誤 6：Alg. 2 以 Euler 寫出，報告結果用 Heun 50 步（NFE 99）。 |
| 4 | 1.4 新增符號表（$I$、$\Sigma$、`e`=n、v 目標 vs v-loss/v-prediction）；1.3 先定義 Σ；步驟 2 定義 ξ、η；步驟 5 標明積分區間 $I$。 |
| 5 | 步驟 10 依 Fig. 5a 原文重寫：排除邊界區域後 0.56→0.3，緊鄰邊界停在 0.5。 |
| 6 | 0.66 標為「Fig. 3b 長期收斂值」，800 epoch 歸 Fig. 3a；0.522 標為 Table 2 第 64 epoch（1.1、步驟 10、LO12、S9、R3）。 |
| 7 | LO10 答案為每個數字標明 Fig. 1 格位（左下、右下）與設定。 |
| 8 | 步驟 10 註明 Table 1 像素空間列未寫 epoch；R5 補充。 |
| 9 | 步驟 11 的 Table 3 補 warmup、EMA、bottleneck、in-context tokens／start block、class-token drop、dropout、取樣時間點；常見錯誤 4 提 EMA。 |
| 10 | 5.2 內嵌 DESIGN.md 字型、色彩、間距、動態 tokens、appbar 與 `#pm-bar` 標記；5.7 內嵌 timeline.json 格式；slug `prediction-targets-k-diff` 寫入檔頭與 5.1；`#paper-238` 定位方式寫明。 |
| 11 | 步驟 8 推導 1 與 C3 觀察 2：標〔本頁推導〕、寫出剩餘平行項 $(\varphi\sigma-\psi\alpha)^2/(\alpha^2+\sigma^2)$、對稱＋凸性、並註明不預測實際網路（JiT 學到 ≈1.0）。 |
| 12 | 步驟 5 新增殘差為 0 的條件（Eq. 78b：ψ/σ 為常數，k 族中只有 k=1）。 |
| 13 | C1b 附註與 LO7 答案：差距也包含平行項（Eq. 78a）；步驟 8 App. H 段補充。 |
| 14 | 步驟 8、C3、R1 改為「與…一致」並明寫「用論文數字校準的推定值」。 |
| 15 | 步驟 6、C2 觀察 3、LO9 加上 $0.5<k^*<1$ 的限定。 |
| 16 | C1 拆為 C1a（S1：只有 t、seed、z）與 C1b（S5：k、chips、殘差圖）；S1 加「為什麼網路要預測目標」2–3 句（步驟 1）。 |
| 17 | 大綱重排：κ/$\widetilde{D\alpha}$ 延到 S7；S3 加投影與梯度 ODE；原 S4 拆成 S4（Eq. 32→8→34→9，教 LO4）與 S5（Eq. 11–12）。 |
| 18 | S7 先講 κ、logit-normal、N 區間與 $W(\alpha)$ 再放 C3；C3 加四步導覽（A→N 區間→B→C）；副圖只在 B/C；S7(e) 明寫母體≠訓練（0.212 vs 0.659）。 |
| 19 | Hero 改為三張白話事實卡；k 軸標籤上下分層、< 480px 錯開並以 `getBBox()` 檢查重疊。 |
| 20 | S3 靜態圖改為 $D=2,d=1$，與 C1 同一平面。 |
| 21 | 原 S6（App. G）併入 S6 末尾一句 Eq. 72，不出現特徵值/SNR，不設 LO。 |
| 22 | 全文用詞替換（權衡、k 作為可訓練參數、k 軸、常見錯誤）；章節標題改為陳述句；5.8 列出規則。 |
| 23 | 5.6：元件讀數改為每個圖版一行來源說明＋「本頁推導」標籤，不逐數字引用。 |
| 24 | C4 改為解析梯度（含 v-MSE 的 $w'_j$），單一路徑直接迴圈；固定學習率並以驗證數據說明無抖動；驗收表改由 `check6.py`（同算法）產生、容許誤差收緊。 |
| 25 | C1b 改畫 ψ 與 $b^*\sigma(t)$ 兩條有界直線，下半畫 $\tfrac12e(t)^2$，填滿面積等於讀數 $(1-k)^2/8$；不再畫發散的 ψ/σ。 |
| 26 | C3 `modeParts`：M=1024、索引分區、區間動差＋因式分解（Q 提出 ψ²）、分母 < 1e-300 保護、max(0,·)；驗收表改為 M=1024、步長 0.005 的結果。 |
| 27 | 5.5：SVG/讀數用 Unicode、動態 DOM 呼叫 `renderMathInElement`、`.katex-display` 可橫捲、Eq. 12 窄螢幕拆行；5.2：四欄以上表格橫捲容器、Table 3 窄螢幕改堆疊卡片。 |
| 28 | 5.9：live region 於 change／500ms debounce 更新；C4 只宣告完成狀態；曲線以線型＋直接標籤區分；C3 改用 $d$ 滑桿與固定讀數框（觸控可用）。 |
| 29 | 5.4：C4 reduced-motion 以 `setTimeout` 分段（每段 500 步）後一次重畫。 |
| 30 | C2：縱軸固定 [0, 0.35]；ρ 軸到 1e-5；$k^*>0.999$ 顯示到第一個非 9 位數（最多 6 位）；千分位。 |
| 31 | 5.1：Google Fonts `display=swap`、5.2 字型表附系統 fallback、頁面重量檢查（Fast 4G、HTML < 200 KB）。 |
| 筆記 | S1、S2、S4、S6、S7、S8、S9 各加延伸閱讀連結；S11 依 `TL_GROUP` 把每張卡片連到對應筆記；S12 依總覽閱讀順序列出全部筆記（`<slug>.html`）。 |


## 8. 修訂紀錄（build 前）

- C3 $d$ 滑桿預設 32。
- C4 參考虛線標為「C3 網格值（步長 0.005）」，並說明 0.117 對 0.115、0.847 對 0.845 的差異在網格解析度內。
- C1 PRNG 固定：mulberry32、Box–Muller 只取 cos 分支、抽樣順序與 seed 規則。
- S3 靜態圖與 S8 兩欄對照版面寫入大綱。

## 9. 建置紀錄（build round）

- 實作與設計的差異（皆已在頁面中落實，記錄於此供審查）：
  - C1a 的 $t\cdot x$ 與 $(1-t)\cdot n$ 以「實線箭頭／虛線箭頭＋圖例」區分，未使用圓形／方形端點。
  - Hero 與 S9 的 k 軸以 HTML 實作（文字保持實際字級），標籤分層改用通用演算法：依左緣排序，與同列前一個標籤重疊（間距 < 8px）就上移一列；不再以 480px 斷點區分。
  - C4 左圖的 y 軸刻度標「ε 0／v 0.5／x 1」，取代另外繪製的直立 k 軸。
  - C4 在首次捲入視窗（可見 50%）時自動執行一次；reduced-motion 時不自動執行。
  - 所有圖表的 SVG 寬度等於容器實際寬度（視窗改變寬度時重畫），圖上文字維持 12–13px，不隨 viewBox 縮放。
  - 深色模式 `--ink-3` 由 #8E897F 調為 #9C978C（impeccable 檢出 #8E897F 在 `--surface-2` 上為 4.2:1；新值 5.0:1）。
  - 圓角卡片的 3px 上邊框改為標籤前的線型樣本（impeccable：border-accent-on-rounded）；表格、演算法格線、分段按鈕加 3–4px 內距（cramped-padding）；Hero 標題取消斜體（italic-serif-display）；圖版與卡片標題由 h4 改為 h3（skipped-heading）。
  - `selfTest()` 另外檢查 `scrollWidth <= innerWidth`（無水平捲動）。
- impeccable 的 advisory「numbered-section-labels」保留：章節編號是 DESIGN.md 的目錄式編號設計，非失敗項。
- 驗收後補丁（Haiku 盲測 9/12 全對、3 題部分正確後）：S1 先寫一般式 $z=\alpha x+\sigma n,\ u=\varphi x+\psi n$ 再特化；S4 在 Eq. 9b 後加可收合推導（右乘兩個投影、交叉項因 $PP^\top(I-PP^\top)=0$ 消失）；S5 內文寫出 Eq. 78a（雜訊強度相依模型的平行項也不大於共享模型）；S7 加一句「v-loss = 以 κ² 加權的 u-MSE，$\widetilde{D\alpha}=D\alpha\cdot\kappa^2$，κ = 1/(k(1−t)+(1−k)t)」；S7 的「母體最佳值 ≠ 訓練值」改為獨立 callout（0.212／0.659／0.612）；S8 內文寫出截斷規則的兩處來源（Sec. 4 與 Table 3）；S9 加 FID 摘要句（2.05 vs 2.08、4.70、2.17 vs 2.05）。
