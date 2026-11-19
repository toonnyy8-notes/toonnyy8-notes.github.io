# 設計文件：〈Understanding and Learning Prediction Targets in Diffusion Models〉互動教學

- 論文：Qing Jin、Chaoyang Wang，arXiv 2601.21419（v3，NeurIPS 2026）。資料庫編號 No. 238。
- 原文：`tut/238/src/paper.txt`（PDF 同目錄）。本文件中的「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」皆指論文原文編號。
- 產出：`tut/238/build/index.html`（單一 HTML 檔）。
- 數值驗證腳本（非頁面程式，僅供核對本文件數字）：`tut/238/verify/check.py`、`check2.py`、`table.py`、`check5.py`。本文件所有標示「已驗證」的數字都由這些腳本算出。
- 文件慣例：凡是**本設計自行推導、論文沒有寫出**的結論，一律標記「〔本設計推導〕」。頁面上也必須用同樣方式標示（見 5.7）。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

擴散模型與 flow matching 要讓神經網路從帶噪輸入 $z$ 預測某個目標：噪聲（ε-prediction）、速度（v-prediction）或乾淨資料（x-prediction）。實務上，低維資料多用 ε 或 v，高解析度像素空間則轉向 x，但沒有理論說明「最佳目標為什麼取決於資料」。本文把目標寫成連續的一族 $u = kx-(1-k)n$（Eq. 13），$k=0,0.5,1$ 分別對應 ε、v、x（差一個常數倍）。作者分析一個「所有噪聲強度共用同一個權重矩陣」的單層線性去噪器，發現它的學習動態拆成兩個互不干擾的模式：沿資料子空間的**平行模式**（資料重建）與垂直於子空間的**垂直模式**（消除噪聲）。最佳損失中平行項正比於內在維度 $d$、垂直項正比於餘維度 $D-d$（Eq. 12）。在均勻時間取樣、u-MSE 損失的條件下，最佳參數為 $k^*=D/(D+d)$（Eq. 14）：資料填滿空間時 $k^*=0.5$（v），資料只占極低維子空間時 $k^*\to1$（x）。作者再用受控實驗顯示時間條件、取樣分布與損失會移動這個最佳值（Sec. 3.4、Fig. 1），因此提出 **k-Diff**：把 $k=\mathrm{sigmoid}(w_k)$ 當成一個可學的純量，與網路一起用反向傳播訓練（Sec. 4、Eq. 15）。在 ImageNet 256 的潛空間（LightningDiT-XL/1）中 $k$ 由 0.5 收斂到 0.66，在像素空間（JiT-B/16）中 30 個 epoch 內收斂到接近 1.0，FID 與固定目標基準相當或略好（Table 1）。

### 1.2 研究動機（Sec. 1）

作者列出三個未解問題：

1. 最佳目標是否只能在 ε、v、x 三者中選？低維時 v 是否普遍最佳？高維時 x 是否必要？
2. 先前文獻（JiT [8]）以流形假說定性說明 x-prediction 適合稀疏高維資料，但沒有定量關係：最佳目標能否由環境維度 $D$ 與內在維度 $d$ 解析推得？
3. 實務上沒有選擇目標的原則，只能試錯。

### 1.3 貢獻

1. 一個通用形式 $z=\alpha x+\sigma n,\ u=\varphi x+\psi n$（Eq. 1），把各種目標視為同一目標函數的特例。
2. 單層線性去噪器的學習動態分析：平行／垂直兩個模式（Eq. 9）、最佳權重與最佳損失（Theorem 3.1，Eq. 11–12）。
3. $k$ 參數化下的閉式解 $k^*=D/(D+d)$（Theorem 3.2，Eq. 14），以及非白化資料的推廣 $k^*=D/(D+\mathrm{Tr}\,\Sigma)$（App. G，Eq. 72）。
4. 受控維度掃描（$D=128$）：時間條件、取樣分布、損失如何移動最佳 $k$（Sec. 3.4，Fig. 1；App. H）。
5. k-Diff：可學的 $k$，在潛空間與像素空間的 ImageNet 生成上驗證（Sec. 4–5）。

### 1.4 從作者的角度逐步拆解

以下依論文推導順序排列。每一步說明「作者想回答什麼」「做了什麼假設」「得到什麼」「下一步為什麼需要它」。

#### 步驟 1：把所有目標寫成同一個形式（Sec. 3，Eq. 1）

$$z=\alpha x+\sigma n,\qquad u=\varphi x+\psi n \tag{Eq. 1}$$

- $x\in\mathbb R^D$ 是乾淨資料，$n\in\mathbb R^D$ 是高斯噪聲。$\alpha,\sigma$ 定義前向過程（通常預先給定），$\varphi,\psi$ 定義網路要預測的目標（由設計者選）。
- 例：flow matching 的 v-prediction 取 $\alpha=t,\ \sigma=1-t,\ \varphi=1,\ \psi=-1$；x-prediction 取 $\varphi=1,\ \psi=0$。
- **時間方向慣例**：本文 $\alpha=t$，所以 $t=1$ 是乾淨資料、$t=0$ 是純噪聲。這與 DDPM「$t$ 越大越吵」相反。頁面必須明講（初學者最常在這裡混淆）。
- 作者以 $\alpha$ 為自變數，其餘三者是 $\alpha$ 的函數；$x,n$ 與 $\alpha$ 無關。

#### 步驟 2：訓練時比較的變數不一定是 u——比例因子 κ（Eq. 3–5，App. A）

網路輸出 $\hat u$，但損失可能算在另一個變數 $w=\xi x+\eta n$ 上（例如速度 $v$）。由 $z,\hat u$ 解出 $\hat x,\hat n$（Eq. 16–18）後可得

$$\hat w-w=\kappa(\hat u-u),\qquad \kappa=\frac{\xi\sigma-\eta\alpha}{\varphi\sigma-\psi\alpha}\tag{Eq. 3, 22}$$

- u-loss：$\kappa=1$（Eq. 23）；x-loss：$\kappa=\sigma/(\varphi\sigma-\psi\alpha)$（Eq. 24）；ε-loss：$\kappa=-\alpha/(\varphi\sigma-\psi\alpha)$（Eq. 25）；v-loss：$\kappa=(\alpha+\sigma)/(\varphi\sigma-\psi\alpha)$（Eq. 26）。
- 因此任何 w-MSE 都可寫成 u-MSE 乘上權重 $\kappa^2$：

$$L=\frac12\int \widetilde{D\alpha}\;\mathbb E\,\|Wz-u\|^2,\qquad \widetilde{D\alpha}:=D\alpha\cdot\kappa^2\tag{Eq. 4, 5}$$

  $D\alpha$ 是訓練時 $\alpha$ 的取樣機率測度（例如 $U[0,1]$ 或 logit-normal [18]），$\kappa^2$ 是隨 $\alpha$ 變化的損失權重。
- 為什麼需要這一步：後面所有結論都用「有效測度」$\widetilde{D\alpha}$ 表示，換取樣分布或換損失，只是換 $\widetilde{D\alpha}$。Sec. 3.4 會用到這一點。
- 〔本設計推導〕在 $k$ 參數化與 flow matching 下，v-loss 的 $\kappa=1/\big(k(1-t)+(1-k)t\big)$（把 $\xi=1,\eta=-1,\alpha=t,\sigma=1-t,\varphi=k,\psi=-(1-k)$ 代入 Eq. 26）。

#### 步驟 3：最簡化的問題設定（Sec. 3.1）

- 資料：$x=P\tilde x$，$\tilde x\in\mathbb R^d$ 是內在座標，$P\in\mathbb R^{D\times d}$ 是 $D\times D$ 正交矩陣的子矩陣，$P^\top P=I_{d\times d}$。$d$ 是內在維度、$D$ 是環境維度。此設定沿用 JiT [8]。
- 去噪器：單層線性 $\hat u=Wz$（Eq. 2），$W\in\mathbb R^{D\times D}$，**所有噪聲強度共用同一個 $W$**（論文後文稱 time-shared）。
- 白化假設：$\Sigma_{\tilde x\tilde x}=\mathbb E[\tilde x\tilde x^\top]=I_{d\times d}$（Eq. 6）；噪聲 $\Sigma_{nn}=I$（Eq. 7）。一般 $\Sigma$ 的情形在 App. G。
- 為什麼這樣簡化：線性模型讓期望值可以寫成閉式，而且 $PP^\top$ 是投影矩陣，後面能把問題沿子空間與其正交補拆開。

#### 步驟 4：學習動態拆成兩個模式（Sec. 3.2，Eq. 8–10，App. B）

梯度下降 $\tau\,dW/d\tilde t=-\partial_W L$（Eq. 27），用 $\mathbb E[zz^\top]=\alpha^2PP^\top+\sigma^2 I$、$\mathbb E[uz^\top]=\varphi\alpha PP^\top+\psi\sigma I$（Eq. 32）得到

$$\tau\frac{dW}{d\tilde t}=-\int\widetilde{D\alpha}\,\big(\alpha^2WPP^\top+\sigma^2W-\varphi\alpha PP^\top-\psi\sigma I\big)\tag{Eq. 8}$$

$\tau$ 是學習率的倒數，$\tilde t$ 是訓練時間（不是擴散時間）。定義 $W_\parallel=WPP^\top$、$W_\perp=W(I-PP^\top)$（Eq. 10），右乘兩個互相正交的投影（Eq. 34）得到

$$\tau\frac{dW_\parallel}{d\tilde t}=-\int\widetilde{D\alpha}\big[(\alpha^2+\sigma^2)W_\parallel-(\varphi\alpha+\psi\sigma)PP^\top\big]\tag{Eq. 9a}$$
$$\tau\frac{dW_\perp}{d\tilde t}=-\int\widetilde{D\alpha}\big[\sigma^2W_\perp-\psi\sigma(I-PP^\top)\big]\tag{Eq. 9b}$$

- 平行模式含 $\alpha,\varphi$（與資料有關），作者稱為**資料重建**。
- 垂直模式完全不含 $\alpha,\varphi$，只看 $\sigma,\psi$：子空間外沒有資料，那裡的 $z$ 只有噪聲 $\sigma n_\perp$，目標只有 $\psi n_\perp$。作者稱為**消除噪聲**。
- 兩個模式對 $\varphi,\psi$ 的要求不同，所以最佳設計要權衡兩者。這是全文的核心觀察。
- 〔本設計推導〕兩模式的鬆弛速率分別是 $\int\widetilde{D\alpha}(\alpha^2+\sigma^2)$ 與 $\int\widetilde{D\alpha}\,\sigma^2$；均勻取樣、u-loss 時為 $2/3$ 與 $1/3$。頁面只在文字中提及，不放進互動元件（見 4.4 說明）。

#### 步驟 5：最佳權重與最佳損失（Theorem 3.1，Eq. 11–12，App. C）

令 Eq. 9 等於 0：

$$W^*=\frac{\int\widetilde{D\alpha}(\varphi\alpha+\psi\sigma)}{\int\widetilde{D\alpha}(\alpha^2+\sigma^2)}PP^\top+\frac{\int\widetilde{D\alpha}\,\psi\sigma}{\int\widetilde{D\alpha}\,\sigma^2}(I-PP^\top)\tag{Eq. 11}$$

代回 Eq. 4（用 $\mathrm{Tr}(PP^\top)=d$、$\mathrm{Tr}(I-PP^\top)=D-d$，Eq. 48–49）：

$$\Delta^*=\underbrace{\frac d2\left[\int\widetilde{D\alpha}(\varphi^2+\psi^2)-\frac{\big(\int\widetilde{D\alpha}(\varphi\alpha+\psi\sigma)\big)^2}{\int\widetilde{D\alpha}(\alpha^2+\sigma^2)}\right]}_{\text{平行（流形內）}}+\underbrace{\frac{D-d}2\left[\int\widetilde{D\alpha}\,\psi^2-\frac{\big(\int\widetilde{D\alpha}\,\psi\sigma\big)^2}{\int\widetilde{D\alpha}\,\sigma^2}\right]}_{\text{垂直（環境殘差）}}\tag{Eq. 12}$$

作者的四點 Remark：
1. 平行項正比於 $d$，垂直項正比於餘維度 $D-d$。
2. 平行項依賴 $\alpha,\varphi$；垂直項與兩者無關。
3. $D\approx d$ 時平行項主導；$D\gg d$ 時垂直項主導。
4. 取 $\psi=0$（x-prediction）時垂直項為 0。

- 為什麼垂直項不為 0（直觀版，頁面要用元件 C1 講）：在單一噪聲強度下，垂直方向的理想增益是 $\psi/\sigma$（App. H，Eq. 76b）。這個值隨 $\sigma$（也就是 $t$）改變；共享的 $W_\perp$ 只能取一個加權平均 $\int\widetilde{D\alpha}\psi\sigma/\int\widetilde{D\alpha}\sigma^2$，其餘部分就是殘差。只有 $\psi=0$ 時理想增益處處為 0，共享矩陣才能完全吻合。
- 下一步的理由：直接對 Eq. 12 做變分求最佳 $\varphi(\alpha),\psi(\alpha)$ 太複雜，結果還依賴損失（$\kappa$）、取樣分布與積分區間 $I$。所以作者再加限制。

#### 步驟 6：k 參數化與閉式解（Sec. 3.3，Theorem 3.2，Eq. 13–14，App. D）

令 $\varphi=k,\ \psi=-(1-k)$，$0\le k\le1$ 且與 $t$ 無關：

$$u=kx-(1-k)n\tag{Eq. 13}$$

- 搭配 flow matching（$\alpha=t,\sigma=1-t$）：$k=0$ 得 $u=-n$（ε，差一個負號）；$k=0.5$ 得 $u=\tfrac12(x-n)=\tfrac12 v$；$k=1$ 得 $u=x$。論文寫「up to a constant scaling factor」。
- Theorem 3.2 的四個條件：$\alpha=t,\sigma=1-t$；$k$ 與 $t$ 無關；$t\sim U[0,1]$；$\kappa=1$（u-MSE）。在白化子空間資料、共享線性去噪器下。
- App. D 的計算：均勻取樣下 $\int\widetilde{D\alpha}=1$，$\int\alpha=\int(1-\alpha)=1/2$，$\int\alpha^2=\int(1-\alpha)^2=1/3$（Eq. 53）。代入得

$$\Delta^*=d\left(\frac{k^2}4-\frac k4+\frac5{16}\right)+\frac{(D-d)(1-k)^2}{8}=\frac1{16}\big[2(D+d)k^2-4Dk+(2D+3d)\big]\tag{Eq. 54c, 54d}$$

  （平行方括號 $=\tfrac12k^2-\tfrac12k+\tfrac58$，垂直方括號 $=\tfrac14(1-k)^2$，各乘 $\tfrac d2$、$\tfrac{D-d}2$；已用 `check.py` 數值核對。）
- 對 $k$ 微分為 0：

$$k^*=\frac{D}{D+d}\tag{Eq. 14, 55}$$

- 三個區間（Sec. 3.3）：$D\approx d$（稠密資料）$k^*\approx0.5$，v 最佳；$D\gg d$（稀疏資料）$k^*\to1$，x 最佳；一般情形 $k^*$ 介於 0.5 與 1 之間，三個傳統目標都不是最佳。
- 兩項的拉鋸：平行項在 $k=0.5$ 最小（值 $d/4$），垂直項在 $k=1$ 最小（值 0）。$k^*$ 是兩者依 $d$ 與 $D-d$ 加權後的折衷。
- 〔本設計推導〕超出最小值的損失為 $\Delta^*(k)-\Delta^*(k^*)=\frac{D+d}{8}(k-k^*)^2$。例：$D=128,d=32$：$k^*=0.8$；$\Delta^*$ 在 $k=0,0.5,1,0.8$ 分別為 $22,\ 11,\ 10,\ 9.2$。

#### 步驟 7：推廣到一般共變異數（App. G，Eq. 56–72）

- 若 $\mathbb E[xx^\top]=\Sigma=\sum_i\lambda_iq_iq_i^\top$，動態沿每個特徵向量獨立（Eq. 61），每個模式的最佳損失為 Eq. 68。
- 特徵值大（高 SNR，$\lambda_i\gg1$）的模式：Eq. 69 在 $\varphi=0$（ε）最小；特徵值小（$\lambda_i\ll1$）的模式：Eq. 70 在 $\psi=0$（x）最小。
- 在 Theorem 3.2 的條件下：$\Delta^*=\frac18\big[(D+\mathrm{Tr}\,\Sigma)k^2-2Dk+\mathrm{Tr}\big((I+4\Sigma)(I+\Sigma)^{-1}\big)\big]$（Eq. 71e），$k^*=D/(D+\mathrm{Tr}\,\Sigma)$（Eq. 72）。$\Sigma=PP^\top$ 時 $\mathrm{Tr}\,\Sigma=d$，回到 Eq. 14。
- 頁面處理：只用文字與一條公式說明，不做元件。

#### 步驟 8：閉式解的適用範圍——受控實驗（Sec. 3.4，Fig. 1，App. H）

設定：各向同性高斯資料放在 $\mathbb R^{128}$ 的 $d$ 維子空間（$D=128$，11 個 $d$ 值），$k=\mathrm{sigmoid}(w_k)$ 與去噪器一起學，同一 $d$ 共用資料、$k$ 跨 $t$ 共用。Fig. 1 四格的結論（數字照原文）：

| Fig. 1 位置 | 設定 | 結果 |
| --- | --- | --- |
| 左上 | 共享線性、均勻 $t$、u-MSE（Theorem 3.2 原條件） | 學到的 $k$ 與 $D/(D+d)$ 相差在 0.027 內 |
| 右上 | 加入時間條件（八個等寬時間區間各一個線性映射；時間嵌入 MLP），均勻 $t$、u-MSE | $k$ 偏離 $D/(D+d)$；分區預測與學到的八區間 $k$ 相差在 0.025 內；MLP 與八區間模型相差在 0.012 內（平均差 0.004） |
| 左下 | 共享線性，改取樣與損失 | logit-normal 取樣使最佳值在 $d\ge48$ 時變成 $k=0$（Theorem 3.1 預測誤差 0.026 內）；再改用 v-MSE 使高 $d$ 的最佳值離開邊界（預測誤差 0.045 內） |
| 右下 | 時間條件 + logit-normal + v-MSE | 分區母體最佳值不再能定量預測訓練結果，大 $d$ 尤其明顯。$d=128$：八區間母體最佳值 0.212，訓練出的八區間模型 0.659，時間嵌入 MLP 0.612 |

- App. H（雜訊強度相依的線性模型 $\hat u=W(\alpha)z$）：最佳 $W^*(\alpha)=\frac{\varphi\alpha+\psi\sigma}{\alpha^2+\sigma^2}PP^\top+\frac{\psi}{\sigma}(I-PP^\top)$（Eq. 76b），最佳損失 $\Delta^*_{dep}=\frac d2\int\widetilde{D\alpha}\frac{(\varphi\sigma-\psi\alpha)^2}{\alpha^2+\sigma^2}$（Eq. 77c），垂直殘差完全消失，且 $\Delta^*_{dep}\le\Delta^*$（Eq. 78–79）。
- 作者的結論：知道 $d$ 與 $D$ 並不足以決定實際學到的 $k$，所以要在實際訓練設定下學 $k$。這是 k-Diff 的動機。
- 〔本設計推導〕以下三點由 Eq. 12 / 77c 直接推得，並以 `table.py` 數值確認，頁面元件 C3 會呈現：
  1. 均勻取樣 + u-MSE + 雜訊強度相依模型：$\Delta^*_{dep}(k)=\Delta^*_{dep}(1-k)$，因此 $k^*=0.5$，與 $d$ 無關。區間數 $N$ 增加時，最佳 $k$ 從 $D/(D+d)$ 往 0.5 移動（$d=32$：$N=1,2,4,8,16,32,64$ 分別為 0.80、0.73、0.67、0.605、0.56、0.535、0.515）。
  2. v-MSE + 雜訊強度相依模型：$\kappa^2(\varphi\sigma-\psi\alpha)^2=1$，所以 $\Delta^*_{dep}$ 與 $k$ 無關，$k$ 不可辨識。
  3. 對稱取樣分布（$p(t)=p(1-t)$）+ u-MSE + 共享線性：最佳值仍是 $D/(D+d)$。因此 Fig. 1 中使 $k\to0$ 的 logit-normal 必然不對稱；用 JiT 的 $\mu=-0.8,\sigma=0.8$（Table 3 像素空間設定）可重現論文的「$d\ge48$ 時 $k=0$」與「$d=128$ 八區間 0.212」。論文 Sec. 3.4 沒有寫明參數，見第 6 節風險 R1。

#### 步驟 9：k-Diff（Sec. 4，Eq. 15，Alg. 1–2，App. E–F）

$$L(\theta,w_k)=\frac12\int\widetilde{D\alpha}\;\mathbb E\,\|u_\theta(z)-u\|^2,\qquad k=\mathrm{sigmoid}(w_k)\tag{Eq. 15}$$

- $u$ 透過 $k$ 依賴 $w_k$（Eq. 13）。$\theta$ 與 $w_k$ 一起反向傳播更新。
- 時間相依版本 $k(t)$：$[0,1]$ 切成 $N$ 個區間（例如 $N=128$），每個端點 $t_i$ 有一個可學的 $w_k(t_i)$，區間內對 sigmoid 後的端點值做線性內插（App. F）。
- 演算法（App. E）：

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
  取樣（Alg. 2，Euler）：`v_pred` 同上，`z_next = z + (t_next - t) * v_pred`。

- 〔本設計推導〕Alg. 1 的 `v` 在代數上恆等於 $x-e$，與 $k$ 無關（由 Eq. 17：$x=\frac{(1-k)z+(1-t)u}{k(1-t)+(1-k)t}$、$n=\frac{kz-tu}{k(1-t)+(1-k)t}$ 相減）。所以實際訓練的損失永遠在同一個 v 空間比較，$k$ 只決定「網路輸出如何換算成 v」。這對應 Eq. 15 中 $\widetilde{D\alpha}$ 含 v-loss 的 $\kappa^2$。Table 3 也寫明兩組實驗的 loss 都是 v-loss。
- 數值穩定：分母 $k(1-t)+(1-k)t$ 截斷到最小 0.05（Sec. 4，沿用 [8]）。$k=1$ 時分母是 $1-t$，在 $t\to1$ 歸零；$k=0$ 時分母是 $t$，在 $t\to0$ 歸零。Table 3：潛空間「x-prediction 用 0.05，其他用 0」，像素空間 0.05。
- 作者強調 k-Diff 與特徵對齊、取樣法、自編碼器改良等方法正交，可直接加入既有流程（Sec. 4）。

#### 步驟 10：實驗（Sec. 5，Table 1–3，Fig. 2–5）

- 潛空間（ImageNet 256，LightningDiT-XL/1，VA-VAE tokenizer）：64 epoch，CFG 1.5，Heun 50 步（NFE = 99）。v-prediction 基準 FID 2.08（原論文報告 2.11），k-Diff 2.05。延長到 800 epoch：最終 FID 1.34，最低 1.22 出現在第 384 epoch（之後有過擬合趨勢）。Table 1 用最終 checkpoint：LightningDiT-XL/1 FID 1.35 / IS 295.3，k-Diff 1.34 / 301.7。$k$ 從 0.5 單調上升到 0.66（Fig. 3b）。
- 像素空間（JiT-B/16，ImageNet 256）：200 epoch 時 k-Diff 與 x-prediction 的 FID 都是 4.70；$k$ 在 30 個 epoch 內從 0.5 升到接近 1.0。Table 1：JiT-B/16 3.66 / 275.1，k-Diff 3.64 / 274.9。ImageNet 512（JiT-B/32）：JiT 4.02 / 271.0，k-Diff 4.03 / 277.1。
- 消融（Table 2，LightningDiT-XL/1，64 epoch）：單一 $k$（學到 0.522）FID 2.05；兩個獨立參數 $k_0x-k_1\varepsilon$（學到 0.556、0.509）FID 2.04；128 區間時間相依 $k(t)$ FID 2.17（較差）。Fig. 5a：$k(t)$ 在 $t\approx0$ 約 0.56，隨 $t$ 增加而下降，接近 $t=1$ 時約 0.3；端點附近停在初始值 0.5，作者歸因於 logit-normal 取樣密度在端點趨近 0、梯度不足（Fig. 5b）。
- 注意：Table 2 的 0.522 是第 64 epoch 的值，Fig. 3b 的 0.66 是長期訓練（800 epoch）收斂值，兩者不矛盾（Sec. 5.1 寫「monotonically increases」）。頁面要同時標出 epoch。

#### 步驟 11：實作細節（App. I，Table 3）

| 項目 | 潛空間 LightningDiT-XL | 像素空間 JiT-B |
| --- | --- | --- |
| depth / hidden / heads | 28 / 1152 / 16 | 12 / 768 / 12 |
| patch size | 1 | image_size / 16 |
| loss | v-loss | v-loss |
| epochs | 64（消融）、800 | 200（消融）、600 |
| optimizer | Adam，β1, β2 = 0.9, 0.95 | 同左 |
| batch / lr / schedule | 1024 / 2e-4 / constant | 同左 |
| time sampler | logit(t) ~ N(0.0, 1.0²) | logit(t) ~ N(−0.8, 0.8²) |
| noise scale | 1.0 | 1.0 × image_size / 256 |
| (1 − t) 除法截斷 | x-prediction 0.05，其他 0 | 0.05 |
| 取樣 | Heun 50 步，CFG 1.5，CFG 區間 [0.1, 1] | Heun 50 步，CFG 2.9，CFG 區間 [0.1, 1] |

實作陷阱（頁面第 10 節逐條列出）：
1. 時間方向：$t=1$ 是資料、$t=0$ 是噪聲。
2. 分母 $k(1-t)+(1-k)t$ 必須截斷（≥ 0.05）。
3. 損失算在 v 空間（Alg. 1、Table 3），不是直接比較 u；$k$ 改變的是輸出參數化。
4. $k$ 用 sigmoid 參數化、初始 $w_k=0$（$k=0.5$，Fig. 3b 由 0.5 出發）。論文沒有說明 $w_k$ 是否使用與網路相同的學習率或權重衰減（權重衰減為 0），見風險 R4。
5. 常數 $k$ 優於 128 區間 $k(t)$（Table 2）；若用 $k(t)$，取樣密度低的端點參數學不動（App. F）。
6. 取樣時同樣要用學到的 $k$ 換算 v（Alg. 2）。

#### 步驟 12：限制（Sec. 6）

閉式解假設白化子空間資料、共享線性去噪器、均勻時間取樣、u-MSE。受控實驗顯示學到的目標會隨去噪器、取樣、損失改變，且分區母體最佳值不一定等於訓練值。推廣到非線性模型與更大規模的文字到影像系統，以及釐清目標選擇如何影響樣本品質，仍是開放問題。

---

## 2. 學習目標（驗收測試）

最終頁面必須讓讀者能回答下列問題；Goldfish 審查時逐條檢查「指定章節是否確實教到、參考答案是否能從頁面內容得出」。章節編號見第 3 節。每個章節結尾以 `<details>` 呈現該章的檢查題（題目與參考答案原文照搬）。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 寫出本論文 flow matching 的前向過程與時間方向 | 本文的 $z$ 怎麼寫？$t=0$、$t=1$ 各代表什麼？ | $z=tx+(1-t)n$；$t=1$ 為乾淨資料，$t=0$ 為純噪聲（Sec. 3，$\alpha=t,\sigma=1-t$） | S1 |
| LO2 | 用 $k$ 表示三種傳統目標 | $u=kx-(1-k)n$ 中 $k=0,0.5,1$ 各得到什麼？ | $k=0$：$u=-n$（ε，差負號）；$k=0.5$：$u=\tfrac12(x-n)=\tfrac12v$；$k=1$：$u=x$（Eq. 13） | S2 |
| LO3 | 說明 κ 的角色 | 用 v-loss 訓練時，損失與 u-MSE 的關係是什麼？ | $\hat v-v=\kappa(\hat u-u)$，損失等於以 $\kappa^2$ 加權的 u-MSE，即有效測度 $\widetilde{D\alpha}=D\alpha\cdot\kappa^2$（Eq. 3–5、26） | S2 |
| LO4 | 解釋學習動態為何拆成兩個模式 | 為什麼 Eq. 8 可以拆成 $W_\parallel$ 與 $W_\perp$？ | $\mathbb E[zz^\top]$ 與 $\mathbb E[uz^\top]$ 都只由 $PP^\top$ 與 $I$ 組成，$PP^\top$ 與 $I-PP^\top$ 是互相正交的投影（Eq. 32、34） | S4 |
| LO5 | 區分兩個模式的物理意義 | 垂直模式的動態與哪些參數無關？代表什麼過程？ | 與 $\alpha$、$\varphi$ 無關（只含 $\sigma,\psi$）；代表消除噪聲，平行模式代表資料重建（Sec. 3.2） | S4 |
| LO6 | 讀懂 Eq. 12 的兩項 | 最佳損失的平行項、垂直項各正比於什麼？取 $\psi=0$ 時發生什麼？ | 平行項 ∝ $d$，垂直項 ∝ $D-d$；$\psi=0$（x-prediction）時垂直項為 0（Theorem 3.1 Remark） | S4 |
| LO7 | 說明共享模型為何留下垂直殘差 | 共享線性模型的垂直殘差從哪裡來？雜訊強度相依模型為何沒有？ | 每個 $t$ 的理想垂直增益 $\psi/\sigma$ 隨 $t$ 改變，共享 $W_\perp$ 只能取 $\int\widetilde{D\alpha}\psi\sigma/\int\widetilde{D\alpha}\sigma^2$；$W(\alpha)$ 可在每個 $\alpha$ 取 $\psi/\sigma$，Eq. 77c 只剩 $d$ 項（App. H） | S4、S7 |
| LO8 | 計算閉式最佳 $k$ | Theorem 3.2 下 $k^*$ 是多少？$D=128,d=32$ 時 $k^*$ 與 $\Delta^*$ 在 ε、v、x、$k^*$ 的值？ | $k^*=D/(D+d)=0.8$；$\Delta^*=\tfrac1{16}[2(D+d)k^2-4Dk+(2D+3d)]$，值為 22、11、10、9.2（Eq. 14、54d） | S5 |
| LO9 | 說出三個維度區間 | $d=D$ 與 $d/D\to0$ 時 $k^*$ 分別趨近什麼？ | $d=D$：0.5（v）；$d/D\to0$：1（x）；其他介於 0.5 與 1 之間（Sec. 3.3） | S5 |
| LO10 | 列出閉式解的假設與會移動 $k^*$ 的因素 | Theorem 3.2 依賴哪四個條件？Fig. 1 中改變哪些設定會讓學到的 $k$ 偏離 $D/(D+d)$？ | 白化子空間資料、共享線性去噪器、$t\sim U[0,1]$、u-MSE；時間條件、取樣分布、損失（例：logit-normal + u-MSE 時 $d\ge48$ 的最佳 $k=0$；$d=128$ 八區間母體最佳 0.212，訓練值 0.659）（Sec. 3.4、Sec. 6） | S7 |
| LO11 | 描述 k-Diff 的做法與數值陷阱 | k-Diff 如何參數化 $k$？實際損失算在哪個空間？分母何時需要截斷？ | $k=\mathrm{sigmoid}(w_k)$，純量，與網路一起反向傳播（Eq. 15），初始 0.5；損失為 v-loss，$v_{pred}=((1-2k)z+u_{pred})/(k(1-t)+(1-k)t)$；分母截斷到 ≥ 0.05，$k\to1,t\to1$ 或 $k\to0,t\to0$ 時分母趨近 0（Sec. 4、Alg. 1、Table 3） | S8、S10 |
| LO12 | 讀出實驗結果 | 潛空間與像素空間學到的 $k$ 各是多少？時間相依 $k(t)$ 的消融結果？ | 潛空間（LightningDiT-XL/1）0.5→0.66（800 epoch；第 64 epoch 為 0.522），FID 2.05 vs v-pred 2.08（64 epoch）；像素空間（JiT-B/16）30 epoch 內→約 1.0，200 epoch FID 皆 4.70；128 區間 $k(t)$ FID 2.17，差於常數 $k$ 的 2.05（Sec. 5、Table 2） | S9 |

---

## 3. 頁面大綱

全頁單欄閱讀（文字欄 680–720px），互動元件放在加寬的「圖版」（最大 1080px）。章節編號以 JetBrains Mono 顯示（`01 · 目標`），延續 DESIGN.md 的目錄式編號。視覺記號：一條 0 到 1 的水平「k 尺」，標出 ε（0）、v（0.5）、x（1），在 hero、C2、C4、S9 重複出現，讓讀者把每個結果放回同一條尺上。

| # | id | 標題（暫定） | 目的 | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈擴散模型該預測什麼：從 ε、v 到 x 的連續目標 k〉 | 一句話結論；k 尺標出 ε、v、x、潛空間學到的 0.66、像素空間約 1.0；三張事實卡（$k^*=D/(D+d)$；兩個模式；k-Diff 只多一個純量）；論文連結與 `../index.html#paper-238` | 1 螢幕 |
| S1 | `basics` | 01 · 擴散模型在學什麼 | 給初學者：前向過程 $z=tx+(1-t)n$、時間方向、三種目標的公式與互換（Eq. 1、17）。C1 上半部（向量圖）在此首次出現 | 400–500 字 + C1 |
| S2 | `k` | 02 · 一個參數涵蓋三種目標 | Eq. 13；$k$ 與 ε/v/x 的對應；κ 與有效測度 $\widetilde{D\alpha}$（Eq. 3–5、22–26，用小表列四種 loss 的 κ） | 350–450 字 |
| S3 | `linear` | 03 · 最簡單的去噪器 | Sec. 3.1 設定：$x=P\tilde x$、$d$ 與 $D$、單層共享線性 $W$、白化假設（Eq. 2、6、7）；說明為什麼選線性模型 | 300–400 字 + 1 張靜態 SVG（$\mathbb R^3$ 中的 2 維平面示意，標 $d=2,D=3$） |
| S4 | `modes` | 04 · 兩個模式：資料重建與消除噪聲 | Eq. 8–12；物理意義；C1 下半部（垂直增益圖）解釋殘差來源 | 500–600 字 + C1 |
| S5 | `kstar` | 05 · 最佳 k = D/(D+d) | Theorem 3.2、App. D（Eq. 53–55）、三個區間、〔本設計推導〕超出損失；C2 | 400–500 字 + C2 |
| S6 | `general` | 06 · 資料不是白化的時候 | App. G：Eq. 68–72，高 SNR 模式偏好 ε、低 SNR 模式偏好 x、$k^*=D/(D+\mathrm{Tr}\Sigma)$ | 200–300 字 |
| S7 | `scope` | 07 · 時間條件、取樣與損失會移動最佳 k | Sec. 3.4 表格（第 1.4 節步驟 8 的表）、App. H、三點〔本設計推導〕；C3 | 500–600 字 + C3 |
| S8 | `kdiff` | 08 · k-Diff：讓 k 自己學 | Eq. 15、Alg. 1–2（程式碼區塊，左右對照「每行在做什麼」）、v 目標恆等於 $x-e$〔本設計推導〕、$k(t)$ 版本；C4 | 450–550 字 + C4 |
| S9 | `results` | 09 · 實驗結果 | Table 1 摘錄（k-Diff 與其基準，共 3 組）、Table 2 消融、k 尺上的學到值（標 epoch）；文字描述 Fig. 2、3、5 的結論，不重繪曲線（論文沒有給出曲線數據） | 350–450 字 + 2 張表 + k 尺 |
| S10 | `impl` | 10 · 實作細節與陷阱 | Table 3 精簡版、6 條陷阱（1.4 步驟 11） | 300–400 字 + 表 |
| S11 | `history` | 11 · 研究脈絡（由 notes/timeline.json 提供） | 時間線（規格見 5.6）+ 一段文字把時間線接到 k 尺：ε（DDPM 時期）→ v（Progressive Distillation、flow matching）→ x（SiD2、JiT）→ 連續的 k | 時間線 + 100–150 字 |
| S12 | `next` | 12 · 限制與延伸閱讀 | Sec. 6 限制；連到研究筆記（5.6 列出 slug）與論文 | 200 字 + 連結清單 |
| — | — | 頁尾 | 引用格式、arXiv 連結、固定導覽列 `#pm-bar` | — |

桌機（≥ 1024px）左側窄欄放章節目錄（sticky，Distill 風格），目前章節以 `--accent` 標示；< 1024px 改為頂部可橫向捲動的章節列。

---

## 4. 互動元件規格

共用數學（四個元件共用同一份函式，見 5.4）：
- 時間積分一律用中點法：$t_j=(j+0.5)/M$，$M=1000$，權重 $1/M$。
- 取樣分布：`uniform`：$p(t)=1$；`logitn`：$p(t)=\dfrac{1}{s\sqrt{2\pi}\,t(1-t)}\exp\!\Big(-\dfrac{(\mathrm{logit}\,t-\mu)^2}{2s^2}\Big)$，$\mu=-0.8,\ s=0.8$。
- 損失權重：`u`：$\kappa^2=1$；`v`：$\kappa^2=1/\big(k(1-t)+(1-k)t\big)^2$（不截斷；logit-normal 密度在端點衰減得比 $\kappa^2$ 增長快，積分有限）。
- 三種「訓練設定」（只開放 Fig. 1 用過的組合）：**A** = 均勻 + u-MSE；**B** = logit-normal + u-MSE；**C** = logit-normal + v-MSE。均勻 + v-MSE 不開放：$k\to1$ 時 $\int dt/(1-t)^2$ 發散，論文也沒有用這個組合。

### C1 · 二維中的 x、n、z、u（S1 首次出現，S4 再次引用）

- **教的概念**：前向過程與目標向量；$u$ 的垂直分量 $-(1-k)n_\perp$ 是純噪聲；共享線性模型在垂直方向只能用一個增益，因此 $k<1$ 時留下殘差（Theorem 3.1 Remark 4、App. H Eq. 76b）。
- **簡化**（頁面需寫出）：$D=2,\ d=1$。資料子空間是水平軸，$x=(\tilde x,0)$，$\tilde x\sim\mathcal N(0,1)$（白化，Eq. 6；與 Sec. 3.4 的高斯資料一致）。$n=(n_\parallel,n_\perp)\sim\mathcal N(0,I)$。殘差使用均勻 $t$、u-MSE（Theorem 3.2 條件）。
- **讀者操作**：
  - 滑桿 $t\in[0,1]$，步長 0.01，預設 0.5。
  - 滑桿 $k\in[0,1]$，步長 0.01，預設 0.5；旁邊三個 chip「ε (k=0)」「v (k=0.5)」「x (k=1)」直接設定 $k$。
  - 按鈕「重新抽樣」：換一組 seed（預設 seed = 7）。
- **計算與顯示**：
  - 上圖（正方形 SVG，座標範圍 $[-3,3]^2$）：
    - 水平軸畫成粗線並標「資料子空間（d = 1）」，垂直軸標「垂直方向（D − d = 1）」。
    - 150 個背景點：固定的 $(\tilde x_i,n_i)$，畫 $z_i=t\,x_i+(1-t)\,n_i$（淺色）。拖動 $t$ 時同一批點連續移動。
    - 一個突顯樣本：從原點畫 $t\,x$（`--c-par`）、接著畫 $(1-t)\,n$（`--c-perp`），終點為 $z$（實心點）；從原點畫 $u=kx-(1-k)n$（`--accent` 粗箭頭），並用虛線畫出 $u_\parallel=k\tilde x-(1-k)n_\parallel$ 與 $u_\perp=-(1-k)n_\perp$ 的投影。
    - 數值表（JetBrains Mono，兩位小數）：$\tilde x,\ n_\parallel,\ n_\perp,\ z_\parallel=t\tilde x+(1-t)n_\parallel,\ z_\perp=(1-t)n_\perp,\ u_\parallel,\ u_\perp$。
  - 下圖「垂直方向需要的增益」（寬幅 SVG）：
    - 橫軸 $t\in[0,1]$；縱軸顯示範圍 $[-4,0.5]$（超出範圍的曲線裁切，並在裁切處標「→ −∞」）。
    - 曲線 $g(t)=\psi/\sigma=-(1-k)/(1-t)$：每個 $t$ 單獨訓練時的最佳垂直增益（Eq. 76b）。
    - 水平線 $b^*=\int\psi\sigma/\int\sigma^2=-1.5(1-k)$：共享線性模型的最佳垂直增益（Eq. 11 在均勻 $t$、u-MSE 下）。
    - 兩者之間塗淺色，代表殘差來源；目前的 $t$ 用垂直細線標示並與上圖同步。
    - 讀數：「每個垂直維度的殘差 $=\tfrac12\int(b^*\sigma-\psi)^2dt=(1-k)^2/8$」（Eq. 54c，$D-d=1$）。$k=0,0.5,1$ 時分別顯示 0.125、0.031、0.000。
- **讀者應觀察到**：
  1. $z_\perp=(1-t)n_\perp$：垂直方向的輸入只有噪聲。
  2. $u_\perp=-(1-k)n_\perp$：只有 $k=1$ 時目標的垂直分量為 0。
  3. $k=1$ 時 $g(t)\equiv0=b^*$，殘差為 0；$k$ 越小，$g(t)$ 與 $b^*$ 的差距越大，殘差 ∝ $(1-k)^2$。
  4. 這就是 Eq. 12 垂直項的來源，也是 $D\gg d$ 時偏好大 $k$ 的原因。

### C2 · 兩項損失的拉鋸：k* = D/(D+d)（S5）

- **教的概念**：Theorem 3.2；平行項偏好 $k=0.5$、垂直項偏好 $k=1$；最佳 $k$ 只取決於 $d/D$；三個區間。
- **簡化**：完全使用 Eq. 54c（Theorem 3.2 的四個條件），無額外近似。
- **讀者操作**：
  - 滑桿 $D$：對數刻度，值集合 $\{2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536\}$，預設 128。
  - 滑桿 $d$：對數刻度，整數 $1\ldots D$（以 $\log_2$ 均分 200 段後四捨五入、去重），預設 32；改變 $D$ 時若 $d>D$ 則設為 $D$。
- **計算與顯示**：
  - 主圖：橫軸 $k\in[0,1]$（即 k 尺，標 ε/v/x），縱軸 $\Delta^*/D$（每個環境維度的損失，讓不同 $D$ 的曲線尺度可比）。三條曲線：
    - 平行項 $\frac dD\big(\frac{k^2}4-\frac k4+\frac5{16}\big)$（`--c-par`）
    - 垂直項 $\frac{D-d}D\cdot\frac{(1-k)^2}8$（`--c-perp`）
    - 總和（`--ink`，較粗）
    - $k^*=D/(D+d)$ 處畫 `--accent` 實心點與垂直虛線，標「k* = 0.800」。
  - 讀數表（絕對值 $\Delta^*$，Eq. 54d；超出量用〔本設計推導〕$\frac{D+d}8(k-k^*)^2$）：

    | 目標 | k | Δ* | 比 k* 多 |
    | --- | --- | --- | --- |
    | ε | 0 | … | … |
    | v | 0.5 | … | … |
    | x | 1 | … | … |
    | k* | D/(D+d) | … | 0 |

  - 小圖：橫軸 $\rho=d/D$（對數，$10^{-4}$ 到 1），縱軸 $k^*=1/(1+\rho)\in[0.5,1]$，標出目前的點；在 $\rho=1$ 標「v」，在左端標「→ x」。
- **預設值下的正確輸出**（驗收）：$k^*=0.800$；$\Delta^*$：ε 22.000、v 11.000、x 10.000、k* 9.200；超出量：12.800、1.800、0.800、0。
- **讀者應觀察到**：
  1. 同比例放大 $D$ 與 $d$，$k^*$ 不變（只看 $d/D$）。
  2. $d=D$ 時垂直項消失，$k^*=0.5$；$d\ll D$ 時垂直項主導，$k^*\to1$。
  3. 一般情形下 ε、v、x 都不是最小值。

### C3 · 訓練設定如何移動最佳 k（S7）

- **教的概念**：閉式解只在 Theorem 3.2 條件下成立；時間條件（分區線性、雜訊強度相依）、取樣分布、損失都會移動最佳值；母體最佳值與實際訓練值可能不同（Sec. 3.4、Fig. 1、App. H）。
- **簡化**（頁面需寫出）：只計算**母體最佳值**（對資料與噪聲取期望後的最佳 $k$），不訓練網路；不含時間嵌入 MLP；logit-normal 參數取 $\mu=-0.8,s=0.8$，並註明「論文 Sec. 3.4 未寫明參數，此值與 JiT 設定相同，且能重現論文報告的 $d\ge48$ 時 $k=0$、$d=128$ 八區間 0.212」。
- **計算**：
  - 對固定 $k$，Eq. 12 可寫成 $\Delta^*(k,d)=d\cdot P(k)+(D-d)\cdot Q(k)$，其中 $P,Q$ 與 $d$ 無關：
    - 共享線性：$P=\frac12\big[\int w(\varphi^2+\psi^2)-\frac{(\int w(\varphi\alpha+\psi\sigma))^2}{\int w(\alpha^2+\sigma^2)}\big]$，$Q=\frac12\big[\int w\psi^2-\frac{(\int w\psi\sigma)^2}{\int w\sigma^2}\big]$，$w=p(t)\kappa^2(t)$，$\varphi=k,\psi=-(1-k),\alpha=t,\sigma=1-t$。
    - $N$ 區間線性：把 $[0,1]$ 切成 $N$ 個等寬區間，各區間內用同樣公式（積分只在區間內），$P,Q$ 為各區間之和（「summed population loss」，Sec. 3.4）。
    - 雜訊強度相依（App. H，Eq. 77c）：$P=\frac12\int w\frac{(\varphi\sigma-\psi\alpha)^2}{\alpha^2+\sigma^2}$，$Q=0$。
  - $k$ 網格 0 到 1、步長 0.005（201 點）；每個 $d\in\{1,\dots,128\}$ 取使 $\Delta^*$ 最小的網格點。
  - 平坦偵測：若某條曲線在所有 $k$ 上 $\max\Delta^*-\min\Delta^*<10^{-9}\cdot\max|\Delta^*|$，不畫該曲線，改在圖例顯示「損失與 k 無關，k 無法由損失決定」（設定 C + 雜訊強度相依必然觸發，〔本設計推導〕）。
  - 每次換設定重算一次（約 201 × 1000 × 3 次運算，可同步完成）。
- **讀者操作**：
  - 分段按鈕「訓練設定」：A / B / C（預設 A）。
  - 分段按鈕或滑桿「時間區間數 N」：1、2、4、8、16、32、64（預設 8），控制「N 區間線性」曲線。
  - 勾選框「顯示雜訊強度相依模型（App. H）」（預設勾選）。
  - 主圖上的指標（滑鼠移動或鍵盤左右鍵）選取 $d$，tooltip 顯示各曲線在該 $d$ 的 $k^*$。
  - 副圖滑桿 $k$（預設 0.5）。
- **顯示**：
  - 主圖：橫軸 $d\in[1,128]$（$D=128$ 固定），縱軸 $k^*\in[0,1]$（左側標 ε/v/x 刻度）。曲線：$D/(D+d)$ 虛線（Theorem 3.2）、共享線性（`--accent` 實線）、N 區間線性（`--c-par`）、雜訊強度相依（`--c-perp` 點線）。
  - 設定 C 時另外畫論文報告的三個點（空心標記，標「論文報告，d = 128」）：八區間母體最佳 0.212、訓練出的八區間模型 0.659、時間嵌入 MLP 0.612。
  - 主圖下方一行文字顯示該設定在論文中的對應敘述與誤差（照 1.4 步驟 8 表格）。
  - 副圖「有效權重 $\widetilde{D\alpha}(t)=p(t)\kappa^2(t)$」：橫軸 $t$，曲線除以最大值正規化到 [0,1]，隨副圖 $k$ 與設定更新。用來說明：B 把權重集中在 $t<0.5$（噪聲多的一側），C 的 $\kappa^2$ 再把權重推向分母小的一端。
- **驗收數值**（$M=1000$、$k$ 步長 0.001 算出；頁面步長 0.005，允許誤差 ±0.006）：

  | 設定 | 去噪器 | d=1 | 4 | 16 | 32 | 48 | 64 | 96 | 128 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | A | 共享 | 0.992 | 0.970 | 0.889 | 0.800 | 0.727 | 0.667 | 0.571 | 0.500 |
  | A | 8 區間 | 0.960 | 0.869 | 0.695 | 0.607 | 0.566 | 0.542 | 0.515 | 0.500 |
  | A | 雜訊相依 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
  | B | 共享 | 0.936 | 0.774 | 0.376 | 0.117 | 0.000 | 0.000 | 0.000 | 0.000 |
  | B | 8 區間 | 0.455 | 0.028 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
  | B | 雜訊相依 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
  | C | 共享 | 0.997 | 0.987 | 0.940 | 0.847 | 0.712 | 0.543 | 0.240 | 0.082 |
  | C | 8 區間 | 0.999 | 0.996 | 0.982 | 0.955 | 0.913 | 0.838 | 0.481 | 0.212 |
  | C | 雜訊相依 | 平坦 | | | | | | | |

  設定 A、$d=32$ 的區間數掃描：N = 1, 2, 4, 8, 16, 32, 64 → 0.80, 0.73, 0.67, 0.605, 0.56, 0.535, 0.515。
- **讀者應觀察到**：
  1. A + 共享：曲線與 $D/(D+d)$ 重合（Theorem 3.2）。
  2. A：區間數越多，最佳 $k$ 越接近 0.5；雜訊強度相依時恆為 0.5，因為垂直殘差消失（App. H），只剩偏好 0.5 的平行項。
  3. B：logit-normal 使高 $d$ 的最佳值落到 $k=0$（論文：$d\ge48$）。
  4. C：v-MSE 使最佳值離開邊界；$d=128$ 時八區間母體最佳 0.212，但論文訓練出的模型是 0.659 / 0.612，母體分析在這裡失準。這就是論文改為「直接學 $k$」的理由。

### C4 · k-Diff 的玩具版：讓 k 自己學（S8）

- **教的概念**：k-Diff 把 $k=\mathrm{sigmoid}(w_k)$ 和去噪器一起用梯度下降學（Eq. 15）；不需要知道 $d$，$k$ 也會走到該設定下的最佳值；學到的值取決於取樣分布與損失。
- **簡化**（頁面需寫出）：
  - 用母體損失（對 $x,n$ 取期望、對 $t$ 用 $M=1000$ 中點積分），不抽樣本，因此軌跡是確定的。
  - 去噪器限制為 $W=a\,PP^\top+b\,(I-PP^\top)$：從 $W=0$ 出發時，Eq. 8 的梯度只落在這兩個矩陣的線性組合中，所以這個限制不改變梯度流的解（〔本設計推導〕）。$a$ 是平行模式係數、$b$ 是垂直模式係數。
  - 只做共享線性去噪器、$D=128$。
- **損失**（每個環境維度）：
  $$\mathcal L(a,b,w_k)=\frac{1}{2D}\sum_j \frac{p(t_j)\kappa^2(t_j)}{M}\Big[d\big((a t_j-k)^2+(a(1-t_j)+(1-k))^2\big)+(D-d)\big(b(1-t_j)+(1-k)\big)^2\Big]$$
  （由 $Wz-u=(a\alpha-\varphi)x+(a\sigma-\psi)n_\parallel+(b\sigma-\psi)n_\perp$ 取期望得到；設定 C 的 $\kappa^2$ 隨 $k$ 改變，梯度要包含這一項。）
- **最佳化**：Adam，$\beta_1=0.9,\beta_2=0.95$（Table 3 的值），$\epsilon=10^{-8}$，學習率 0.02（玩具用值；論文網路用 2e-4，頁面需註明不同），初始 $a=b=w_k=0$（$k=0.5$，與 Fig. 3b 相同）。梯度用中央差分 $h=10^{-5}$（3 個參數 × 2 次損失評估）或解析式，二擇一，結果需符合下表。總步數 3000，每個動畫影格執行 30 步。
- **讀者操作**：
  - 分段按鈕「訓練設定」：A / B / C（預設 A）。
  - 滑桿 $d\in\{1,\dots,128\}$，預設 32。
  - 按鈕：「執行」／「暫停」、「重設」。改變設定或 $d$ 時自動重設。
- **顯示**：
  - 左圖：$k$ 對步數（橫軸 0–3000）。水平虛線：同一設定與 $d$ 下 C3 的共享線性母體最佳值（直接呼叫 C3 的計算）；設定 A 另標 $D/(D+d)$。右側附一條直立 k 尺，標出目前的 $k$。
  - 右圖：$a$、$b$ 對步數，以及依目前 $k$ 由 Eq. 11 算出的平衡值 $a^*(k)=\frac{\int w(\varphi\alpha+\psi\sigma)}{\int w(\alpha^2+\sigma^2)}$、$b^*(k)=\frac{\int w\psi\sigma}{\int w\sigma^2}$（虛線，隨 $k$ 移動）。
  - 讀數：步數、$k$、$\mathcal L$、$a$、$b$。
  - 圖下對照 Alg. 1：「這個玩具省略了 `sample_t` 與 `randn_like`（改成取期望），保留了 `k = sigmoid(w_k)` 與 `u = k*x - (1-k)*e`」。
- **驗收數值**（`check5.py`，第 100 / 300 / 1000 / 3000 步的 $k$，與最終 $a,b$）：

  | 設定 | d | k@100 | k@300 | k@1000 | k@3000 | a | b |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | A | 4 | 0.783 | 0.966 | 0.970 | 0.970 | 0.704 | −0.045 |
  | A | 32 | 0.749 | 0.800 | 0.800 | 0.800 | 0.448 | −0.300 |
  | A | 64 | 0.667 | 0.666 | 0.667 | 0.667 | 0.250 | −0.499 |
  | A | 128 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 |
  | B | 4 | 0.656 | 0.774 | 0.774 | 0.774 | 0.174 | −0.319 |
  | B | 32 | 0.407 | 0.117 | 0.117 | 0.117 | −0.909 | −1.250 |
  | B | 64 | 0.248 | 0.017 | 0.000 | 0.000 | −1.100 | −1.415 |
  | C | 4 | 0.680 | 0.956 | 0.987 | 0.987 | 0.749 | −0.020 |
  | C | 32 | 0.653 | 0.847 | 0.847 | 0.848 | 0.433 | −0.236 |
  | C | 64 | 0.600 | 0.543 | 0.543 | 0.543 | −0.193 | −0.652 |
  | C | 128 | 0.390 | 0.082 | 0.082 | 0.081 | −1.011 | 0.000 |

  允許誤差：$k$ ±0.01、$a,b$ ±0.02（中央差分與解析梯度的細微差異）。$d=D$ 時 $b$ 沒有對應維度，保持 0。
- **讀者應觀察到**：
  1. 沒有輸入 $d$，$k$ 仍從 0.5 走到 C3 算出的最佳值（設定 A 即 $D/(D+d)$）。
  2. 換取樣分布或損失，同一份資料學到的 $k$ 不同（$d=32$：A 0.800、B 0.117、C 0.847）。
  3. $k$ 收斂後，$a,b$ 停在 Eq. 11 的平衡值。
- **為什麼不顯示兩模式的速度差**：論文的鬆弛速率比較（$2/3$ vs $1/3$）只對純梯度下降成立；這裡用 Adam（為了與論文訓練設定一致並在 3000 步內收斂），Adam 會逐參數正規化步長，速度比較不再有意義。

---

## 5. 實作計畫

### 5.1 檔案與函式庫

- 單一檔案 `tut/238/build/index.html`；CSS、JS、時間線資料全部內嵌。不得 `fetch` 本機檔案。
- 外部資源只允許：
  - Google Fonts：Newsreader 400/500/600、Noto Serif TC 600/700、IBM Plex Sans 400/500/600、Noto Sans TC 400/500/700、JetBrains Mono 400/500。
  - KaTeX 0.16.9（cdnjs）：`katex.min.css`、`katex.min.js`、`contrib/auto-render.min.js`，分隔符 `$…$`（行內）與 `$$…$$`（區塊）。KaTeX 載入失敗時保留原始 TeX 文字，頁面其餘功能不受影響。
- 不用 D3：所有圖都是手寫 SVG，線性與對數比例尺各寫一個小函式即可。

### 5.2 設計方向

- 依 AGENTS.md 步驟 6-3：Anthropic 官網的編輯式排版 + Material Design 3 的元件形狀與狀態層；以字面敘述為主，不用比喻與華麗辭藻。
- Tokens 全部取自 DESIGN.md（`--bg`、`--surface`、`--surface-2`、`--ink`、`--ink-2`、`--ink-3`（用 #67635A，DESIGN.md 審查紀錄的修正值）、`--line`、`--accent`、`--accent-soft`；字級 12/13/15/17/20/24/32/44/64；間距 4 的倍數；圓角 6/10/16/999；動態 160/240/400ms、`cubic-bezier(.2,0,0,1)`）。
- 圖表專用色（DESIGN.md 只規定介面強調色，以下兩色只用在圖中資料系列，與群集色的用法相同）：

  | token | light | dark | 用途 |
  | --- | --- | --- | --- |
  | `--c-par` | #3F6E8C | #8DB4D3 | 平行模式、資料方向 |
  | `--c-perp` | #6B7A3A | #A9BC7C | 垂直模式、噪聲方向 |

  實作時須確認兩色在 `--surface` 上對比 ≥ 3:1（圖形）且作為文字標籤時 ≥ 4.5:1，不足就調整明度並回寫本表。
- 深色模式：`@media (prefers-color-scheme: dark)` 內以 `:root:not([data-theme="light"])` 覆寫，另以 `:root[data-theme="dark"]` 覆寫一次；頁首提供主題切換鈕，偏好存在 `localStorage` 的 `pm-theme`（讀寫都包 try/catch，失敗時跟隨系統）。`body` 必須設定 `background: var(--bg)`。
- `prefers-reduced-motion: reduce`：取消所有 transition 與捲動動畫；C4 按「執行」時同步算完 3000 步並一次畫出完整軌跡。
- 響應式：390–1440px。390px 時 16px 側邊留白、無水平捲動；元件控制列在圖下方堆疊；SVG 用 `viewBox` 等比縮放，圖上文字最小 12px（必要時在窄螢幕減少刻度數量而不是縮小字）。768px 以上控制列可放圖右側。滑桿與按鈕觸控區 ≥ 44px。
- 無障礙：每個滑桿有 `<label>` 與 `aria-valuetext`（例如「k = 0.80」）；SVG 圖 `role="img"` 並附 `aria-label` 描述目前狀態（數值變化時更新）；讀數區 `aria-live="polite"`；focus 樣式為 2px `--accent` 外框。
- 固定導覽列：沿用 benchmark 的 `#pm-bar`（左下角兩顆 pill：「← Paper Master」→ `../index.html`、「教學與筆記」→ `index.html`），顏色改用 tokens。Hero 另放「在關聯圖中查看」→ `../index.html#paper-238` 與 arXiv 連結 `https://arxiv.org/abs/2601.21419`。

### 5.3 頁面結構

```
<header class="appbar">（標題縮寫、主題切換）
<nav class="toc">（章節目錄，桌機 sticky 左欄／窄螢幕頂部橫列）
<main>
  <section id="top">…  <section id="basics">… <section id="next">
<footer>（引用、arXiv）
<nav id="pm-bar">
<script type="application/json" id="timeline-data">…</script>
<script>（共用數學 → 元件 → 時間線 → KaTeX → selfTest）</script>
```

### 5.4 程式結構與函式名稱

全部寫在一個 `<script>`，以 IIFE 包住，不建立框架或類別階層。

共用數學：
- `const M = 1000; const TGRID = Float64Array.from({length:M}, (_, j) => (j + 0.5) / M);`
- `const SETTINGS = { A:{sampler:'uniform', loss:'u'}, B:{sampler:'logitn', loss:'u'}, C:{sampler:'logitn', loss:'v'} };`
- `const LOGITN = { mu: -0.8, s: 0.8 };`
- `density(sampler, t)` → $p(t)$。預先算好 `DENS.uniform`、`DENS.logitn` 兩個 Float64Array。
- `kappa2(loss, k, t)` → $\kappa^2$。
- `weights(setting, k)` → Float64Array $w_j=p(t_j)\kappa^2(t_j)/M$。
- `modeParts(setting, k, nBins)` → `{P, Q}`；`nBins = 0` 代表雜訊強度相依（App. H）。
- `kStarCurve(setting, nBins)` → `{ks: Float64Array(128) | null, flat: boolean}`，索引 `d-1`。內部先對 201 個 $k$ 求 `{P,Q}`，再對每個 $d$ 取最小。
- `closedForm(D, d, k)` → `{par, perp, total}`（Eq. 54c，未除以 $D$）。
- `kStarClosed(D, d)` → $D/(D+d)$。
- `rng(seed)`：mulberry32；`gauss(r)`：Box–Muller。

元件（每個元件一個 `initC1()`…`initC4()`，各自只碰自己的 DOM 容器 `#c1`…`#c4`）：
- C1：`state = {t, k, seed, pts: [{xt, np, nq}]}`；`renderC1()`。
- C2：`state = {D, d}`；`renderC2()`；`D_VALUES` 陣列。
- C3：`state = {setting, nBins, showDep, d, kSide}`；`cache = new Map()`（key `${setting}|${nBins}`）存 `kStarCurve` 結果；`renderC3()`。
- C4：`state = {setting, d, step, a, b, w, m:[0,0,0], v:[0,0,0], hist:{k:[], a:[], b:[]}, running}`；`lossC4(a, b, w, setting, d)`；`stepC4(state)`（一次 Adam 步）；`runC4()` 用 `requestAnimationFrame`，每影格 30 步。
- SVG 輔助：`svg(tag, attrs)`、`scaleLin(d0, d1, r0, r1)`、`scaleLog(d0, d1, r0, r1)`、`path(points)`、`axis(...)`。只寫實際用到的。
- 時間線：`renderTimeline(data)`（5.6）。

自我檢查（Ponytail 規則：非平凡數值邏輯留一個可執行的檢查）：
- `selfTest()`：網址含 `?test` 時執行，用 `console.assert` 比對：
  - `closedForm(128, 32, k).total` 在 $k=0,0.5,1,0.8$ 為 22、11、10、9.2（誤差 1e-9）。
  - `kStarCurve('A',1)` 在 $d=32$ 為 0.800；`kStarCurve('A',8)` 在 $d=4$ 為 0.869；`kStarCurve('B',1)` 在 $d=48$ 為 0；`kStarCurve('C',8)` 在 $d=128$ 為 0.212；`kStarCurve('C',0).flat === true`（誤差 ±0.006）。
  - C4 設定 A、$d=32$ 跑 1000 步後 $k$ 在 0.800 ± 0.01。
  - 結果以一行 `console.log('selfTest: N passed, M failed')` 輸出。

### 5.5 數學排版

- 公式旁以等寬字標註論文編號，例如右對齊的「Eq. 14」。
- 頁面中的所有方程式與數字必須與第 1 節一致；第 1 節以外不得新增論文數字。
- 〔本設計推導〕的內容在頁面上用一個小標籤「本頁推導」（`--ink-3`，邊框 pill）標示，並附一行推導依據（例如「由 Eq. 17 相減」）。

### 5.6 研究脈絡（由 notes/timeline.json 提供）

- 來源：`tut/238/notes/timeline.json`，由另一個 agent 撰寫。建置時把檔案內容原樣貼進 `<script type="application/json" id="timeline-data">`；頁面用 `JSON.parse` 讀取。不在頁面中手改條目內容。
- 目前觀察到的格式（建置時以實際檔案為準）：陣列，每筆 `{year, title, authors_short, arxiv, role, source}`。
- 呈現方式：
  - 水平捲動的卡片列（`overflow-x: auto`，`scroll-snap-type: x mandatory`），依 `year` 排序（同年保持檔案順序）。每張卡片：年份（JetBrains Mono）、標題、作者、`role`（一到兩行）、arXiv 連結 `https://arxiv.org/abs/{arxiv}`。
  - 不對條目做自動分類或上色；只把本論文（`arxiv === "2601.21419"`）的卡片以 `--accent` 外框與「本文」標籤突出。若 timeline.json 本身不含本論文條目，在列尾自動加一張本論文卡片（年份 2026，role 取自 1.1 的第一句）。
  - 左右按鈕（`aria-label`「往前捲動」「往後捲動」），鍵盤左右鍵可捲動；捲動行為在 reduced-motion 下用 `auto`。
  - 缺欄位處理：缺 `arxiv` 則不顯示連結；缺 `role` 則只顯示標題與作者；JSON 解析失敗則整節顯示「時間線資料尚未提供」並隱藏控制鈕。
- 時間線下方的延伸閱讀連到同目錄的研究筆記（`tut/238/notes/` 目前的 slug）：`prediction-targets-overview.html`（總覽）、`eps-prediction-ddpm-score.html`、`v-prediction-and-preconditioning.html`、`flow-matching-velocity-target.html`、`loss-weighting-and-elbo.html`、`x-prediction-manifold-view.html`、`generalized-target-k-parameterization.html`、`diffusion-dynamics-theory-background.html`；以及站內既有的 `pixel-space-diffusion.html`。建置時以 notes 目錄實際檔名為準。

### 5.7 文案規則

- 繁體中文、台灣用語、字面學術語氣；不用反問句、比喻、宣傳形容詞。
- 專有名詞第一次出現附英文（例如「內在維度（intrinsic dimension）」）。
- 每個數字後註明出處（Eq./Table/Fig./Sec.）。
- 區分三種陳述：論文結果、論文作者的解釋（用「作者認為」「作者歸因於」）、本頁推導（標籤）。

### 5.8 建置後自我檢查（AGENTS.md 步驟 6-3）

截圖寬度 390 / 768 / 1024 / 1440，加深色模式、鍵盤 focus、reduced-motion 各一張；依字體、留白、視覺層級、色彩、動態、微互動、響應式、原創性逐項評分，有差距就修正。另跑 `?test` 確認 selfTest 全數通過，並逐條核對第 2 節 12 個學習目標。

---

## 6. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | Fig. 1 的 logit-normal 參數 | 論文 Sec. 3.4 只寫「logit-normal」，未給 μ、σ。〔本設計推導〕對稱分布 + u-MSE 時最佳值仍是 $D/(D+d)$，無法得到論文的 $k=0$；μ = −0.8、σ = 0.8（Table 3 的 JiT 像素空間設定）可重現 $d\ge48$ 時 $k=0$ 與 $d=128$ 八區間 0.212。μ = −0.8、σ = 1.0 則在 $d=48$ 給出 0.137，不符。 | 頁面採 μ = −0.8、σ = 0.8，並寫明這是推定值與理由。 |
| R2 | 論文數字的不完整性 | Fig. 1 只給文字描述的誤差（0.027、0.025、0.012、0.026、0.045）與 $d=128$ 的三個值；11 個 $d$ 值與 MLP 結果沒有數據。 | C3 只畫母體最佳值與論文明確給出的三個點，不重繪論文的實驗標記。 |
| R3 | Table 2 與 Fig. 3b 的 $k$ 值 | 第 64 epoch 0.522、長期收斂 0.66；頁面若只寫一個值會誤導。 | S9 與 LO12 同時寫兩個值並標 epoch。 |
| R4 | 論文未說明的實作細節 | $w_k$ 的學習率、是否套用 EMA、截斷 0.05 套用在分母還是只在 $(1-t)$（Sec. 4 寫分母，Table 3 寫「clip of (1 − t) in division」）。 | 頁面陷阱清單照錄兩處原文，並寫明論文未說明 $w_k$ 的學習率。 |
| R5 | Table 1 的 512 結果 | JiT-B/32：FID 4.02 → 4.03（略差）、IS 271.0 → 277.1（較好）。論文文字寫「consistently matches or outperforms」。 | 頁面照錄數字，不改寫成「全面優於」。 |
| R6 | 圖表色 | DESIGN.md 只定義單一強調色；本頁需要平行／垂直兩個資料系列色。 | 新增 `--c-par`、`--c-perp`，僅用於圖中資料系列；實作時驗證對比並回寫 5.2 的表。 |
| R7 | 最終檔名與相對連結 | 建置位置是 `tut/238/build/index.html`，上線位置應在 `tutorials/` 下，檔名（slug）未定；`../index.html`、`index.html` 與筆記連結都假設位於 `tutorials/`。 | 由 PM 決定 slug；連結維持 benchmark 的相對路徑寫法。 |
| R8 | 時間線與筆記仍在撰寫 | `notes/timeline.json` 與筆記可能在建置前更新，條目與 slug 可能改變。 | 建置時重新讀取；渲染程式依 5.6 的缺欄位處理容錯。 |
| R9 | 元件效能 | C3 每次換設定約 60 萬次浮點運算、C4 每影格 30 步 × 6 次損失評估 × 1000 點；在低階手機上可能掉影格。 | C3 結果以 `cache` 快取；C4 若單影格超過 12ms，改為每影格 10 步。 |
| R10 | 初學者門檻 | 讀者不熟擴散模型；S1 必須在 C1 出現前把 $z$、$t$ 方向、三種目標講清楚，否則 S4 以後無法理解。 | Goldfish 審查時特別檢查 S1 是否獨立可讀。 |
