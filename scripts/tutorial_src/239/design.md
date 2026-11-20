# 設計文件：〈Learning Anisotropic Prediction Targets for Flow Models〉（AnisoFlow）互動教學

- 論文：Rayhan Zirvi、Hansheng Chen（共同一作）、Gordon Wetzstein、Leonidas Guibas，arXiv 2611.17767v1（2026-11-13）。資料庫編號 No. 239。
- 原文：`tut/239/src/paper.txt`。本文件的「Eq. n」「§n」「App. X」「Fig. n」「Table n」皆指論文原文編號。顧問筆記 `tut/239/advisor.md` 已核對過圖 3/4/5/7 與表 1–4 的 PDF。
- 產出：`tut/239/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/anisoflow.html`（slug `anisoflow`）；所有站內相對連結以上線位置為準。
- 驗算腳本：`tut/239/verify/check.py`（只用 Python 標準庫，固定 seed，可單獨執行；輸出存於 `verify/check_out.txt`）。第 5 節的驗收表全部由它產生，算法與頁面相同。
- 研究筆記：`tut/239/notes/`（總覽 `aniso-lineage-overview` + 6 篇主題筆記 + `timeline.json`），上線後為 `tutorials/<slug>.html`。
- 外殼與最接近的鄰居：No. 238〈預測目標與 k-Diff〉`tutorials/prediction-targets-k-diff.html`（設計文件 `tut/238/design.md`）。本頁重用它的外殼，並在 S1 連過去作為共同背景。**時間方向相反**：No. 238 是 $z=tx+(1-t)n$（$t=1$ 為資料），本論文是 $z_t=(1-t)x+t\epsilon$（$t=0$ 為資料）。
- 標記：〔本頁推導〕＝本設計自行推導、論文沒寫；「推定」＝為配合論文而校準、論文沒寫明的值（本頁沒有推定值，見 R1）。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

Flow model 的網路要從帶雜訊輸入 $z_t$ 預測一個目標：速度 $u=\epsilon-x$、雜訊 $\epsilon$ 或乾淨資料 $x$。這三者在給定 $(z_t,t)$ 時資訊等價，但回歸問題不同。AnisoFlow 把目標改成 $u_W=W\epsilon-x$（Eq. 3），$W=B\,\mathrm{diag}(w_1,\dots,w_D)B^\top$ 為半正定矩陣（Eq. 4）：在一組可學的正交基底 $B$ 的每個方向上，雜訊有自己的非負權重 $w_i$，資料係數固定為 $-1$。網路輸出 $\hat u_W$ 經過閉式還原 $\hat u=(\alpha_tW+\sigma_tI)^{-1}((I-W)z_t+\hat u_W)$（Eq. 5）換回速度，損失仍是速度空間的 flow matching 損失，所以架構、損失、取樣器都不變。因為還原是精確的，任何 $w_i$ 的最小可達損失都相同（Eq. 11–12），$w_i$ 只改變「網路要擬合的函數有多難」。作者在 PCA 基底與局部凍結誤差模型下推得單一時刻的偏好權重 $w_i^*(t)=\frac{\alpha_t C_i(t)}{\sigma_t A_i(t)}$（Eq. 7），在可分離假設下時間依賴抵銷成 $w_i^*=\bar C_i/\bar A_i$（Eq. 23），再加兩個假設得到 $w_i^*\propto\lambda_i$（Eq. 24）。像素空間的 JiT-H/16 實際學到 $w_i\approx0.15\lambda_i$（Fig. 3a），ImageNet 256 的 FID 由 1.90 降到 1.55（Table 1）。

### 1.2 研究動機（§1、§2）

1. 目標選擇影響訓練與樣本品質（作者引用 Karras 2022、Salimans & Ho 2022），而現有選擇只佔設計空間的一角：latent 模型多用速度，近期像素模型（JiT、PixelU）用乾淨資料；k-Diff 在標準目標間做純量內插，AsymFlow 遮掉部分雜訊。
2. 像素空間的 patchify 讓一個 token 含許多像素座標（JiT 為 $16\times16\times3=768$ 維），各方向的資料變異數差異極大。速度目標在每個方向放單位雜訊權重，使低變異方向難以預測；x 預測去掉雜訊，但在高變異方向較難，而且速度還原在低雜訊時病態（誤差放大 $\propto1/\sigma_t$，§2）。
3. 作者想用「每個方向各學一個雜訊權重」解決這個權衡。

### 1.3 貢獻（§1 末段）

(a) 可學習的目標族 AnisoFlow，標準目標皆為特例；(b) 理論分析，把偏好的雜訊權重連到網路的預測誤差，解釋近線性縮放與近似時間不變；(c) 論文原文「state-of-the-art pixel-space generation on ImageNet」（§1；§6.1.1 與 Table 2 的範圍是「best in-class FID … among plain DiT-like pixel-space models」〔本頁註：Table 2 的 U-ViT-like 類別中 SiD2 UViT/1 為 1.38，所以頁面一律寫 best in-class 的限定〕），以及文字生成影像的品質與低精度穩定性改善。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S1 末尾以可收合表格提供）

| 符號 | 意義 |
| --- | --- |
| $x$、$\epsilon$ | 乾淨資料、標準高斯雜訊 $\mathcal N(0,I)$ |
| $t\in[0,1]$ | 時間；**$t=0$ 為資料、$t=1$ 為雜訊**（與 No. 238 相反） |
| $\alpha_t=1-t$、$\sigma_t=t$ | 線性路徑係數（Eq. 1），$\alpha_t+\sigma_t=1$；頁面中常省略下標寫成 $\alpha,\sigma$ |
| $z_t$ | $\alpha_tx+\sigma_t\epsilon$ |
| $u$ | 速度 $\epsilon-x=\partial_tz_t$；$u^*=\mathbb E[u\mid z_t,t]$ |
| $D$ | 資料維度；patchify 模型中為 patch 維度（JiT：768） |
| $W$、$B$、$w_i$、$b_i$ | 半正定目標算子、其正交特徵基底、特徵值（方向 $i$ 的雜訊權重）、第 $i$ 個基底向量 |
| $u_W$、$\hat u_W$ | AnisoFlow 目標 $W\epsilon-x$、網路對它的預測 |
| $\lambda_i$ | PCA 方向 $i$ 的資料變異數 $\mathbb E[x_i^2]$ |
| $A_i,C_i,\Gamma_i$ | 雜訊估計誤差、資料估計誤差的二階矩與交叉矩（Eq. 14） |
| $\bar A_i,\bar C_i,q_i(t)$ | 可分離形式的方向尺度與共同時間函數（Eq. 20） |
| $G$、$a_t$、$b_t$ | spectral update 的動差矩陣與兩個純量（Eq. 40–41） |
| $\sigma_{\min}$ | 推論時還原分母的下限 $4\times10^{-2}$（App. B.1） |

#### 步驟 1：flow matching 與三種標準目標（§3）

$$z_t=\alpha_tx+\sigma_t\epsilon,\quad\alpha_t=1-t,\ \sigma_t=t\tag{Eq. 1}$$
$$\mathcal L_{FM}=\mathbb E_{t,x,\epsilon}\big[\|u_\theta(z_t,t)-u\|^2\big],\quad u=\epsilon-x\tag{Eq. 2}$$

- 平方誤差下最佳預測是條件均值 $\mathbb E[u\mid z_t,t]$，它定義 ODE 速度場；取樣時從 $z_1\sim\mathcal N(0,I)$ 由 $t=1$ 積分到 $t=0$。
- 換算（§3，用 $\alpha+\sigma=1$）：ε 預測 $\hat u=(\hat\epsilon-z_t)/\alpha_t$；x 預測 $\hat u=(z_t-\hat x)/\sigma_t$。x 預測的誤差被放大 $1/\sigma_t$，在低雜訊（$t\to0$）時病態。
- 與 No. 238 的對應：No. 238 的 $k$ 目標 $kx-(1-k)n$ 除以 $-k$ 得 $\frac{1-k}{k}n-x$，即 AnisoFlow 的純量情形 $w=(1-k)/k$：$k=0.5\Rightarrow w=1$，$k=1\Rightarrow w=0$，$k\to0\Rightarrow w\to\infty$〔本頁推導；目標不含 $t$，所以與時間方向無關〕。

#### 步驟 2：線性目標族每個方向只有一個自由度（App. A.1）

- 單一方向的線性目標 $y=\varphi x+\psi\epsilon$。乘上非零常數 $c$ 只是把網路輸出乘 $c$，可表示的函數集合不變，最佳預測也同樣縮放（$\mathbb E[cy\mid z_t,t]=c\,\mathbb E[y\mid z_t,t]$）。所以固定 $\varphi=-1$，只剩 $w=-\psi/\varphi$。$\varphi=0$（純 ε 預測）只能當 $w\to\infty$ 的極限。
- 矩陣版：$y=Kx+L\epsilon$，$K$ 可逆時乘 $T=-K^{-1}$ 得 $u_W=W\epsilon-x$，$W=-K^{-1}L$。$K$ 奇異（純 ε 預測）不在正規形內。
- 作者的但書：正規化不改變可表示的預測器集合，但可能改變損失幾何、梯度與最佳化。

#### 步驟 3：目標與特例（§4，Eq. 3–4，Fig. 2）

$$u_W:=W\epsilon-x,\qquad W=B\,\mathrm{diag}(w_1,\dots,w_D)B^\top,\ w_i\ge0\tag{Eq. 3, 4}$$

- 基底座標下 $u_{W,i}=w_i\epsilon_i-x_i$：資料分量保持全維度，雜訊分量每個方向各有權重。
- 特例（差符號與整體尺度）：$w_i=0$ 為 x 預測；$w_i=1$ 為速度；$w_i\equiv w$ 為純量內插（等同 k-Diff）；PCA 基底下 $w_i\in\{0,1\}$ 為 AsymFlow 的二元低秩遮罩；$w_i\to\infty$ 為 ε 預測。
- patchify 模型中 $W$ 作用在每個 patch、所有位置共用；影像層級算子為 $I_N\otimes W$（App. B.2）。

#### 步驟 4：速度還原（Eq. 5、8–10）

$$(1-w)z_t+u_W=(\alpha_tw+\sigma_t)(\epsilon-x)\tag{Eq. 9}$$

推導（用 $\alpha-1=-\sigma$）：$x$ 係數 $(1-w)\alpha-1=-(\alpha w+\sigma)$；$\epsilon$ 係數 $(1-w)\sigma+w=\sigma+\alpha w$。所以

$$\hat u_i=\frac{(1-w_i)z_{t,i}+\hat u_{W,i}}{\alpha_tw_i+\sigma_t},\qquad \hat u=(\alpha_tW+\sigma_tI)^{-1}\big((I-W)z_t+\hat u_W\big)\tag{Eq. 5, 10}$$

- Eq. 10 對任何使 $\alpha_tW+\sigma_tI$ 可逆的 $W$ 成立，不需要對稱。
- 代入檢查：$w=0\Rightarrow\hat u=(z_t-\hat x)/\sigma$（$\hat u_W=-\hat x$）；$w=1\Rightarrow\hat u=\hat u_W$；$w\to\infty\Rightarrow\hat u\to(\hat\epsilon-z_t)/\alpha$。
- 誤差傳遞：$\hat u_i-u_i=(\hat u_{W,i}-u_{W,i})/(\alpha_tw_i+\sigma_t)$（$z_t$ 項相消）。放大倍率 $1/(\alpha_tw_i+\sigma_t)$：$w_i=0$ 時為 $1/t$；$w_i=1$ 時恆為 1；$w_i>1$ 時小於 1（§4 Velocity recovery）。
- 〔本頁推導〕兩路線混合讀法：若 $\hat u_{W,i}=w_i\hat\epsilon_i-\hat x_i$，則 $\hat u_i=\pi_\epsilon\frac{\hat\epsilon_i-z_{t,i}}{\alpha}+\pi_x\frac{z_{t,i}-\hat x_i}{\sigma}$，$\pi_\epsilon=\frac{\alpha w_i}{\alpha w_i+\sigma}$、$\pi_x=\frac{\sigma}{\alpha w_i+\sigma}$，兩者和為 1。$w_i$ 調的是「ε 路線」與「x 路線」兩個速度估計的混合比例。
- 推論時分母下限 $\sigma_{\min}=4\times10^{-2}$，基線與 AnisoFlow 都用（App. B.1），最大放大倍率因此為 25 倍〔本頁推導：$1/0.04$〕。

#### 步驟 5：為什麼限制為半正定（App. A.1 末段）

- 還原在時刻 $t$ 失效 ⇔ $-t/(1-t)$ 是 $W$ 的特徵值；$t$ 掃過 $(0,1)$ 時這個量掃過整條負實軸。例：$W=-I$ 在 $t=1/2$ 奇異。
- 一般實矩陣可能有複數特徵值、非正規時條件數差，也沒有逐方向的正交還原；限制為對稱後特徵值全為實數。嚴格說，只有落在 $(-\infty,0)$ 的特徵值會在某個 $t$ 造成奇異；在對稱族中，保證所有內部 $t$ 都非奇異的條件是特徵值非負，即 $W\succeq0$（論文的實用限制）。
- 頁面小例（LO4 用）：$W=-I$ 時分母 $\alpha_t(-1)+\sigma_t=2t-1$，在 $t=1/2$ 為 0（App. A.1）。
- $w_i=0$ 只在 $t=0$ 奇異；取樣器不計算 $t=0$。這是設計選擇，不是唯一可行族（論文明說一般族「代數上可行但不實用」）。

#### 步驟 6：損失算在還原後的速度上（§4 Training objective、Eq. 6、11、12）

- $\mathcal L(\theta,w)=\mathbb E_{t,x,\epsilon}\|\hat u(z_t,t)-u\|^2$，**不是** $\|\hat u_W-u_W\|^2$。
- 方向 $i$、時刻 $t$ 的損失：
$$E_i(w_i;t)=\frac{\mathbb E[(\hat u_{W,i}-u_{W,i})^2\mid t]}{(\alpha_tw_i+\sigma_t)^2}\tag{Eq. 6}$$
- 最佳目標預測（對 $u_W=(\alpha w+\sigma)u-(1-w)z_t$ 取條件期望，$z_t$ 為已知量）：
$$u^*_{W,i}=(\alpha_tw_i+\sigma_t)u^*_i-(1-w_i)z_{t,i}\tag{Eq. 11}$$
還原後恆為 $u^*_i$，所以**最小可達損失與 $w_i$ 無關**。
- 分解（交叉項因條件均值正交性為 0）：
$$E_i(w_i;t)=\underbrace{\frac{\mathbb E[(\hat u_{W,i}-u^*_{W,i})^2\mid t]}{(\alpha_tw_i+\sigma_t)^2}}_{\text{超額預測誤差}}+\underbrace{\mathbb E[(u_i-u^*_i)^2\mid t]}_{\text{不可約誤差}}\tag{Eq. 12}$$
只有超額誤差與 $w_i$ 有關：$w$ 的梯度完全來自網路的不完美。
- 線性模型（App. A.5）：網路換成最佳時間條件線性預測器時，$\Phi^*_t(W)=(\sigma_tW-\alpha_t\Sigma)\Sigma_{z,t}^{-1}$（Eq. 43，$\Sigma_{z,t}=\alpha_t^2\Sigma+\sigma_t^2I$），還原後 $(\sigma_tI-\alpha_t\Sigma)\Sigma_{z,t}^{-1}$ 與 $W$ 無關（Eq. 44）。論文原文：線性模型「既不識別權重也不識別基底」。所以**在最佳（時間條件）線性預測器下，損失與 $W$ 無關（Eq. 44），這類玩具無法學出 $w\propto\lambda$**（容量受限、有限樣本或非最佳的模型不在 Eq. 44 涵蓋範圍內，本頁也不做）；頁面的玩具只示範這個不變性（C2）或「給定假設 ⇒ 結果」的計算（C3、C4），並在圖版第一行寫明。
- 〔本頁推導〕純量版（$x\sim\mathcal N(0,\lambda)$）：$\Phi^*=\frac{\sigma w-\alpha\lambda}{\alpha^2\lambda+\sigma^2}$，還原係數 $\frac{\sigma-\alpha\lambda}{\alpha^2\lambda+\sigma^2}$，最小速度誤差 $\frac{\lambda}{\alpha^2\lambda+\sigma^2}$（由 $\mathrm{Var}(u)=1+\lambda$、$\mathrm{Cov}(u,z)=\sigma-\alpha\lambda$、$\mathrm{Var}(z)=\alpha^2\lambda+\sigma^2$ 與 $\alpha+\sigma=1$ 化簡）；目標空間的最小誤差為 $(\alpha w+\sigma)^2$ 乘上它，隨 $w$ 改變。

#### 步驟 7：局部凍結誤差模型與單一時刻的偏好權重（§5、Eq. 13–19、Eq. 7）

- 假設：(a) $\hat u_{W,i}\approx w_i\hat\epsilon_i-\hat x_i$；(b) 誤差 $e_{n,i}=\hat\epsilon_i-\epsilon_i^*$、$e_{x,i}=\hat x_i-x_i^*$ 的條件二階統計量在收斂附近不隨 $w_i$ 變（frozen-error）。以條件均值為基準可把不可約部分分離出去：$\alpha_t(x_i-x_i^*)+\sigma_t(\epsilon_i-\epsilon_i^*)=0$。
$$\mathbb E[(\hat u_{W,i}-u^*_{W,i})^2\mid t]=A_iw_i^2-2\Gamma_iw_i+C_i\tag{Eq. 13}$$
$$A_i=\mathbb E[e_{n,i}^2\mid t],\ \Gamma_i=\mathbb E[e_{n,i}e_{x,i}\mid t],\ C_i=\mathbb E[e_{x,i}^2\mid t]\tag{Eq. 14}$$
- 令 $f(w)=(Aw^2-2\Gamma w+C)/(\alpha w+\sigma)^2$：
$$f'(w)=\frac{2\big[w(A\sigma+\alpha\Gamma)-(\Gamma\sigma+\alpha C)\big]}{(\alpha w+\sigma)^3}\tag{Eq. 17}$$
$$w_i^*(t)=\frac{\Gamma_i\sigma_t+\alpha_tC_i}{A_i\sigma_t+\alpha_t\Gamma_i}\tag{Eq. 18}$$
當 $A\sigma+\alpha\Gamma>0$ 且 $\Gamma\sigma+\alpha C>0$ 時為 $w\ge0$ 上的內部最小；交叉項可忽略時
$$w_i^*(t)\approx\frac{\alpha_tC_i(t)}{\sigma_tA_i(t)}\tag{Eq. 19 = Eq. 7}$$
$\Gamma=0$、$A,C>0$、$t\in(0,1)$ 時為唯一最小點。作者的解讀：資料較難估（$C$ 大）的方向偏好較大雜訊權重；雜訊較難估（$A$ 大）的方向偏好較小權重。
- 〔本頁推導〕邊界情形（C3 需要）：記 $P=A\sigma+\alpha\Gamma$、$Q=\Gamma\sigma+\alpha C$，$f'\propto wP-Q$。$P>0,Q\le0$：$f$ 在 $w\ge0$ 遞增，最小在 $w=0$；$P\le0,Q>0$：$f$ 遞減，下確界為 $w\to\infty$ 的 $A/\alpha^2$；$|\Gamma|\lt\sqrt{AC}$ 時 $P\le0$ 與 $Q\le0$ 不會同時成立（兩條件相乘需 $\rho^2\ge1$，$\rho=\Gamma/\sqrt{AC}$）。
- 〔本頁推導〕$\Gamma=0$ 時最小值 $f^*=\frac{AC}{\alpha^2C+\sigma^2A}$，即 $1/f^*=\alpha^2/A+\sigma^2/C$：ε 路線的速度誤差變異數為 $A/\alpha^2$、x 路線為 $C/\sigma^2$，最佳混合是逆變異數加權，其比例 $(\alpha^2/A):(\sigma^2/C)$ 等於步驟 4 的 $\alpha w:\sigma$，解出的 $w$ 即 Eq. 19。$f(0)=C/\sigma^2$、$f(\infty)=A/\alpha^2$ 都不小於 $f^*$。

#### 步驟 8：時間依賴抵銷 ⇒ 一組權重（Eq. 20–23）

- $\alpha_t/\sigma_t$ 從 $\infty$ 變到 0，看似各時刻偏好非常不同的權重；但小 $t$ 時 $z_t$ 以資料為主，$x$ 好估、$\epsilon$ 難估，大 $t$ 時相反（§5）。作者提出**充分條件**（假設，不是推導結果）：
$$A_i(t)=\alpha_tq_i(t)\bar A_i,\qquad C_i(t)=\sigma_tq_i(t)\bar C_i,\qquad\Gamma_i=0\tag{Eq. 20}$$
$$\frac{\partial}{\partial w}\frac{A_iw^2+C_i}{(\alpha w+\sigma)^2}=\frac{2\alpha_t\sigma_tq_i(t)\bar A_i}{(\alpha_tw+\sigma_t)^3}\Big(w-\frac{\bar C_i}{\bar A_i}\Big)\tag{Eq. 21–22}$$
$$w_i^*=\bar C_i/\bar A_i\tag{Eq. 23}$$
- 為什麼要對 $t$ 平均（頁面 S6 第一段，C4 的前置）：訓練時每個樣本的 $t$ 依時間分佈 $p(t)$ 抽樣，而 $w_i$ 跨時間共用，所以訓練實際最小化的是時間平均損失 $\bar E_i(w_i)=\mathbb E_{t\sim p}[E_i(w_i;t)]$（Eq. 16）。若各時刻偏好的 $w$ 不同，平均的最佳值會依 $p(t)$ 把哪些時刻看得較重而改變；若各時刻偏好相同，$p(t)$ 就不影響結果。
- 每個 $q_i(t)>0$ 的內部時刻共用這個最小點，所以它也是任何內部時間分佈下時間平均損失的最小點（需期望損失有限；$q_i>0$ 的集合有正機率時唯一）。
- 證據：四個時間區間（訓練時間分佈的四分位數切分）各學一組權重，曲線幾乎重合，FID 4.92（共用）對 4.96（隨時間），160 epochs（§6.1.2；Fig. 5 圖說標明 JiT-B/16）。App. A.3：$t\approx0.25$–$0.80$ 五個可靠 bin 內 $\alpha_t/\sigma_t$ 變化 11.7 倍，估計的偏好權重中位數只變 1.19 倍（Fig. 8c）。
- 對照：無資訊極限（論文稱 zero-skill limit；本頁譯為「無資訊極限」以避免擬人用語，App. A.2）。網路對方向 $i$ 沒有 $t$ 以外的資訊，$\hat u_{W,i}=0$：
$$\bar E_i(w_i)=\mathbb E_t\Big[\frac{w_i^2+\lambda_i}{(\alpha_tw_i+\sigma_t)^2}\Big]\tag{Eq. 25}$$
分子不隨 $t$ 變，沒有抵銷；逐時刻最小點 $w_i^*(t)=\frac{\alpha_t}{\sigma_t}\lambda_i$（Eq. 26），偏好的常數權重取決於時間分佈。
- 〔本頁推導〕均勻 $t$ 時 $\int_0^1\frac{dt}{(w+t(1-w))^2}=\frac1w$，所以 Eq. 25 $=w+\lambda/w$，常數最佳權重為 $\sqrt{\lambda}$（C4 驗收）。這不是論文結論；只說明無資訊極限下的最佳常數權重依時間分佈而定，且不是 $\propto\lambda$。
- 兩個「$\propto\lambda$」不要混：無資訊極限是 $w^*(t)=(\alpha/\sigma)\lambda$（隨時間、係數已知）；Eq. 24 是 $w^*=\bar C/\bar A\approx c\lambda$（不隨時間、$c$ 未知）。
- 與 EDM 的關係（Eq. 27–30）：$t\lt1$ 時令 $\tilde z_{t,i}=x_i+s_t\epsilon_i$，$s_t=\sigma_t/\alpha_t$；把 EDM 的純量資料變異數換成 $\lambda_i$，$c_{\mathrm{skip},i}=\lambda_i/(\lambda_i+s_t^2)$，$x_i-c_{\mathrm{skip},i}\tilde z_{t,i}=(1-c_{\mathrm{skip},i})(x_i-w_i\epsilon_i)$，$w_i=\frac{c_{\mathrm{skip},i}s_t}{1-c_{\mathrm{skip},i}}=\frac{\lambda_i}{s_t}=\frac{\alpha_t}{\sigma_t}\lambda_i$，恰為無資訊極限的逐點最佳。這是 EDM 的逐方向延伸，不是 EDM 原式。

#### 步驟 9：近線性頻譜律（Eq. 24、App. A.3）

- 兩個額外假設：雜訊各向同性 ⇒ $\bar A_i\approx\bar A$；網路對每個資料座標的相對誤差相近 ⇒ $\bar C_i\propto\lambda_i$。合起來 $w_i^*\propto\lambda_i$（Eq. 24）。常數 $c$ 理論不預測（取決於誤差結構、時間取樣器、最佳化；Limitations 亦承認）。線性律是由假設放入的，真正的支持是事後量測。
- 觀測：JiT-H/16 $w_i\approx0.15\lambda_i$（Fig. 3a，裁到變異數 ≥ 0.015）；Fig. 4b（JiT-B/16）圖中畫了參考線 $w=0.2\lambda$（參考線，非論文報告的擬合斜率）。未裁版（Fig. 9，App. C）在約第 256 個方向（patch embedding 維度 256）之後明顯偏離。
- App. A.3 一致性檢查：600-epoch AnisoFlow-H/16、凍結 PCA 基底的 checkpoint；單一輸出頭無法拆出 $\hat\epsilon,\hat x$，因此在凍結網路的中間特徵上擬合兩個輔助線性讀出頭（加權 ridge，30K 訓練圖擬合、另 30K 評估）。60 個 PCA 方向中保留 18 個（重建差異 < 5%、還原速度 MSE 增加 < 1%），$\lambda_i$ 範圍 0.205–171。以樣本殘差 $a_i,b_i,g_i$（Eq. 31–32）組成 $A^{\rm eff}_i=a_i+\frac{\alpha}{\sigma}g_i=A_i+\frac{\alpha}{\sigma}\Gamma_i$、$C^{\rm eff}_i=b_i+\frac{\sigma}{\alpha}g_i=C_i+\frac{\sigma}{\alpha}\Gamma_i$（Eq. 35，不可約項 $F_i$ 精確相消），使 Eq. 18 寫成 $w_i^*(t)=\frac{\alpha C^{\rm eff}_i}{\sigma A^{\rm eff}_i}$（Eq. 36，不需假設 $\Gamma=0$）。結果：$\bar A^{\rm eff}_i\propto\lambda_i^{-0.09}$、$\bar C^{\rm eff}_i\propto\lambda_i^{1.00}$，比值 $\propto\lambda_i^{1.09}$，學到的權重指數 1.07；$(\bar C^{\rm eff}_i/\bar A^{\rm eff}_i)/w_i$ 中位數約 0.97；門檻放寬到 10–20% 時方向數 45–51，比值指數仍約 1.09。
- 跨方向耦合（App. C，Fig. 10，JiT-B/16 固定 PCA 基底，32 個自由方向）：高變異方向不受背景影響，最低變異方向較敏感；逐方向分析是近似。

#### 步驟 10：學習基底（§4、App. A.4–A.5）

- Stiefel Adam（App. A.4.1）：把 $B$ 當成 Stiefel 流形 $\{B^\top B=I\}$ 上的參數；投影 $P_B(A)=A-B\,\mathrm{sym}(B^\top A)$，一階動量為矩陣、二階動量為每個基底一個純量，步後以 QR 回到流形並把動量投影到新切空間（Eq. 38）。歐氏度量下的投影–搬運法，不是平行搬運。
- Spectral update（App. A.4.2，預設）：令 $r:=\mathrm{sg}(\hat u_W)+x$，$\hat u_i-u_i=(b_i^\top r-w_ib_i^\top\epsilon)/(\alpha w_i+\sigma)$。平方取期望並丟掉與 $b_i$ 無關的 $w_i^2$ 項：
$$L_B(B)=\sum_ib_i^\top M_ib_i+\text{const},\quad M_i=\mathbb E_t\Big[\frac{S_t-2w_iH_t}{(\alpha_tw_i+\sigma_t)^2}\Big]\tag{Eq. 39}$$
$S_t=\mathbb E[rr^\top\mid t]$、$H_t=\frac12\mathbb E[r\epsilon^\top+\epsilon r^\top\mid t]$；給定網路輸出與權重時為精確式。每個 $b_i$ 由不同的 $M_i$ 計分，一般不是特徵值問題。
- 假設 1（$M_i$ 兩兩可交換）與命題 2：最佳 $B=B_\star\Pi\,\mathrm{diag}(s)$；證明把 $L_B$ 寫成雙隨機矩陣 $[\Omega_{ji}^2]$ 的線性函數，最小值落在排列矩陣（Birkhoff–von Neumann）。實作取均勻平均：
$$G:=\frac1D\sum_iM_i=\mathbb E\big[a_trr^\top-b_t(r\epsilon^\top+\epsilon r^\top)\big],\ a_t=\frac1D\sum_j\frac{1}{(\alpha_tw_j+\sigma_t)^2},\ b_t=\frac1D\sum_j\frac{w_j}{(\alpha_tw_j+\sigma_t)^2}\tag{Eq. 40–41}$$
$B\leftarrow\mathrm{eigvecs}(G)$。$G$ 有重根時特徵向量只確定到子空間內的旋轉。交叉項不可省略（省略會改變基底目標）。
- Remark 3：特徵向量依特徵值**升冪**排列，$w_i$ 綁在 rank $i$；啟發式，不保證解命題 2 的指派問題；欄符號在 $B\,\mathrm{diag}(w)B^\top$ 中相消。
- 線上估計（Eq. 42）：minibatch 內 $M$ 張圖、每張 $N$ 個 patch，以 patch 數正規化；EMA $G\leftarrow(1-\beta)G+\beta\hat G$，$\beta=10^{-5}$，每步特徵分解（App. B.1，跨裝置平均）。
- 與 PCA 的關係：網路輸出為零時（論文：「as is common at initialization」，常見但不保證）$r=x$，$\mathbb E[x\epsilon^\top]=0$，$G\propto\mathbb E[xx^\top]$（置中資料即共變異數）。$G$ 初始化為 $I$，加上單位矩陣倍數不改特徵向量，所以第一次更新得到**單一 minibatch** 估計的特徵向量，即帶雜訊的 PCA 基底。〔本頁推導〕$W=I$ 時 $a_t=b_t=1$，與 $t$ 無關（C5 用）。
- 線性模型（App. A.5）：$W$ 與 $\Sigma$ 可交換時所有 $M_i$ 與 $G$ 在共同基底下對角，假設 1 精確成立，PCA 基底為不動點；局部穩定條件 $|\rho_{jk}|\lt1$，$\rho_{jk}=L_{jk}(w_j-w_k)/(\bar\kappa_j-\bar\kappa_k)$（Eq. 48–49）。範圍：不證全域收斂，也不證網路＋權重＋EMA 聯合動態的穩定性。頁面只用一句話帶過 Eq. 48–49，不做元件。
- 初始化（§4）：預設 $B=I$、$w_i=1$（$W=I$，即速度預測）。可選 PCA 初始化：$B$ 為 PCA 基底、$w_i=0.15\lambda_i$（$c=0.15$ 取自收斂的 JiT-H/16）。

#### 步驟 11：實驗（§6、Table 1–4、App. B）

- Table 1（JiT-H/16，FID@160ep / @600ep，訓練 sec/iter，推論 sec/step）：JiT（Frozen zeros）2.68 / 1.90 / 0.803 / 0.179；AsymFlow（Frozen binary、Frozen PCA）2.34 / 1.76 / 0.803 / 0.179；AnisoFlow 凍結 PCA 基底 2.25 / 1.63 / 0.805 / 0.180；Stiefel+PCA init 2.32 / 1.68 / 0.818 / 0.180；Spectral+PCA init 2.28 / 1.63 / 0.819 / 0.180；Stiefel 無 init 2.28 / 1.60 / 0.818 / 0.180；Spectral 無 init 2.30 / 1.55 / 0.819 / 0.180。
- 作者的解讀：凍結 PCA 基底上學連續權重已大幅改善（1.63），收益主要來自權重本身；由 $W=I$ 學基底最終最佳（1.55）。注意：PCA init 對 160 epochs 的影響不一致（spectral 2.28 vs 2.30 支持加速；Stiefel 2.32 vs 2.28 相反）。每個模型與訓練長度各自 grid search CFG 與 guidance interval（Table 5，例如 Spectral 無 init @600ep 為 2.1 / [0, 0.86]），無多 seed、無誤差棒。
- AsymFlow 的兩個數字不可混用：Table 1 的 1.76 是作者在同一 JiT-H/16 骨幹、同設定下重跑的對照（凍結二值遮罩）；Table 2 的 1.57 是已發表數字，標 †（使用預訓練表徵編碼器等，見 Table 2 表註），AnisoFlow 沒有使用 †。
- Table 2（ADM 與 JiT 協定混用；* 為 JiT 協定，可比 ADM 協定低至多 0.08）：AnisoFlow-H/16 1.55（953M、365 GFLOPs）；JiT-H/16 1.86*；AsymFlow-H/16† 1.57；HyperDiT-H† 1.56*；SiD2 UViT/1 1.38（U-ViT-like 類別）。「best in-class」只限 plain DiT-like 類別。
- 開銷：推論 +0.6%、訓練至多 +2.0%（§6.1.1、Table 1）。
- T2I 512（Table 3，JiT-L/16-T2I 917M x-pred. → AnisoFlow-L/16-T2I 918M）：OneIG 0.512→0.516、LongText* 0.862→0.864、GenEval* 0.820→0.814（變差）、DPG* 85.7→85.8、HPSv3 7.71→8.03、COCO-10k FID 22.0→20.1、pFID 24.4→22.7。256 階段 PCA init + spectral update，512 階段凍結基底只學權重（App. B.3、Table 6）。
- 低精度推論（§6.2、Fig. 1b、6b）：W8A8（只量化 transformer 線性層的權重與 activation），同 prompt、同 seed；INT8、FP8、blockwise FP8 下 x 預測出現 patch 接縫或強雜訊，AnisoFlow 穩定。**只有定性圖，沒有量化指標**。Fig. 1a 的百萬像素樣本來自額外的 1MP + SFT 階段，未做定量評估（App. B.3）。
- Latent（Table 4，FID / JiT FID*）：SiT-XL/2-REPA 1.44 / 1.41 → Aniso 1.44 / 1.40；RAEv2 1.05 / 0.98 → Aniso 1.05 / 0.96。學到的權重接近各自預設（SiT 接近速度、RAEv2 接近 x 預測，Fig. 7），曲線平坦很多，近線性假設不直接適用（§6.3）。RAEv2 的基線與 AnisoFlow 都改成均勻時間取樣與 $\sigma_{\min}=0.25$（基線 JiT FID 1.06→0.98），另調 EMA 係數 $10^{-4}$、權重學習率 10 倍（App. B.4）。

#### 步驟 12：實作細節與常見錯誤（App. B）

1. 時間方向：本論文 $t=0$ 是資料、$t=1$ 是雜訊，取樣由 $t=1$ 積分到 0（與 No. 238 相反）。
2. 損失算在還原後的速度 $\|\hat u-u\|^2$，不是 $\|\hat u_W-u_W\|^2$（§4）。
3. 權重參數化 $w_i=\exp(\eta_i)$（嚴格正，App. B.1），$\eta$ 與模型同學習率、從訓練開始就學、跨時間共用。正文寫「非負」，但可學的 $w_i$ 取不到 0；Table 1 中凍結的 $w=0$ 是 JiT 基線的設定（不經 exp 參數化）。
4. 推論分母下限 $\sigma_{\min}=4\times10^{-2}$（基線與 AnisoFlow 皆然，沿用 AsymFlow）；RAEv2 為 0.25。
5. Spectral update：$G$ 由 stop-gradient 預測計算、初始化為 $I$、EMA $\beta=10^{-5}$、跨裝置平均、每步特徵分解、特徵向量升冪排列並以 rank 綁權重；交叉項要保留。Stiefel 變體：EMA 模型的基底每次 EMA 後以 QR 重新正交化。
6. PCA：用 100,000 張訓練圖、去均值的 patch、在網路看到的正規化像素空間算共變異數（App. B.2）。$W=I$ 變體不需要 PCA。
7. 不要把 $c=0.15$ 當普適常數（B/16 參考線為 0.2；理論不預測 $c$）。
8. FID 協定：1.90 為 ADM 協定（Table 1），1.86* 為 JiT 協定（Table 2）。

限制（§7 Limitations）：分析是局部預測誤差模型，不描述完整訓練動態，也不決定頻譜縮放的常數；收益在像素空間最大，latent 空間學到的目標接近既有預設。AI Use Statement：作者用 LLM 協助驗證與修飾 §5 與 App. A 的推導。

---

## 2. 學習目標（驗收測試）

規則：每題的參考答案必須寫在指定章節的**內文段落**中（一個段落、只在一個章節內完整出現），不能只出現在元件、表格、圖或答案框。每個章節結尾以 `<details>`「檢查理解」照錄該章題目與參考答案。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 寫出本論文的路徑與時間方向 | $z_t$ 怎麼寫？$t=0$、$t=1$ 各是什麼？速度是什麼？與 No. 238 有何不同？ | $z_t=(1-t)x+t\epsilon$；$t=0$ 為資料、$t=1$ 為雜訊；$u=\epsilon-x$（Eq. 1）；No. 238 的 $t=1$ 是資料，方向相反 | S1 |
| LO2 | 說明目標族為何只有雜訊權重 | 為什麼每個方向只需要一個參數？$w_i$ 取哪些值對應 x、速度、ε、k-Diff、AsymFlow？ | 整體尺度可被網路吸收，固定資料係數 $-1$ 後只剩 $w=-\psi/\varphi$（App. A.1）；$w_i=0$ x、$w_i=1$ 速度、$w_i\to\infty$ ε、$w_i\equiv w$ k-Diff、PCA 基底下 $w_i\in\{0,1\}$ AsymFlow（§4） | S2 |
| LO3 | 寫出並推導速度還原 | 由 $\hat u_{W,i}$ 怎麼得到 $\hat u_i$？關鍵恆等式是什麼？ | $\hat u_i=((1-w_i)z_{t,i}+\hat u_{W,i})/(\alpha_tw_i+\sigma_t)$；恆等式 $(1-w)z_t+u_W=(\alpha_tw+\sigma_t)(\epsilon-x)$，用到 $\alpha_t+\sigma_t=1$（Eq. 5、9） | S3 |
| LO4 | 解釋半正定限制 | 還原何時失效？為什麼最後限制成 $W\succeq0$？ | $-t/(1-t)$ 是 $W$ 的特徵值時失效，它掃過整條負實軸；限制為對稱使特徵值為實數，保證所有 $t$ 非奇異的條件是特徵值非負；例：$W=-I$ 的分母 $2t-1$ 在 $t=1/2$ 為 0（App. A.1） | S3 |
| LO5 | 計算誤差放大倍率 | 目標誤差換到速度時放大多少？$w_i=0$ 與 $w_i=1$ 各是多少？推論時如何截斷？ | $1/(\alpha_tw_i+\sigma_t)$；$w_i=0$ 為 $1/t$，$w_i=1$ 恆為 1；分母下限 $\sigma_{\min}=0.04$，誤差倍率最多 25 倍，但截斷改的是分母本身，對準確的預測也會造成偏差（Eq. 5、App. B.1） | S3 |
| LO6 | 理解 $w$ 不改變最小損失 | 為什麼選不同 $w_i$ 的最小可達損失相同？那 $w_i$ 的梯度從哪裡來？線性模型會學到什麼 $w$？ | 還原可逆，最佳目標預測 $u^*_{W,i}=(\alpha w_i+\sigma)u^*_i-(1-w_i)z_{t,i}$ 還原後恆為 $u^*_i$（Eq. 11）；只有超額預測誤差除以 $(\alpha w_i+\sigma)^2$ 與 $w_i$ 有關（Eq. 12），梯度來自網路的不完美；最佳線性預測器的還原速度與 $W$ 無關，線性模型不識別權重也不識別基底（Eq. 44） | S4 |
| LO7 | 寫出單一時刻的偏好權重 | 凍結誤差模型下 $w_i^*(t)$ 是什麼？$C$ 變大時權重往哪走？ | $w_i^*(t)=(\Gamma_i\sigma_t+\alpha_tC_i)/(A_i\sigma_t+\alpha_t\Gamma_i)$（Eq. 18），交叉項可忽略時 $\alpha_tC_i/(\sigma_tA_i)$（Eq. 7）；資料較難估（$C$ 大）偏好較大雜訊權重 | S5 |
| LO8 | 說明時間抵銷與無資訊極限對照 | 訓練最小化的是哪個損失？什麼條件讓所有時刻共用一個權重？共用值是多少？無資訊極限時為何不行？ | $t$ 依 $p(t)$ 抽樣且 $w_i$ 跨時間共用，所以最小化時間平均 $\mathbb E_t[E_i(w_i;t)]$（Eq. 16）；$A_i=\alpha_tq_i\bar A_i$、$C_i=\sigma_tq_i\bar C_i$、$\Gamma_i=0$（Eq. 20，充分條件）⇒ 每個時刻都在 $w_i^*=\bar C_i/\bar A_i$ 最小（Eq. 23），因此與 $p(t)$ 無關；無資訊極限時損失分子 $w_i^2+\lambda_i$ 不隨 $t$ 變，逐時刻最佳 $(\alpha_t/\sigma_t)\lambda_i$ 隨時間改變（Eq. 25–26） | S6 |
| LO9 | 區分近線性律的假設與證據 | $w_i^*\propto\lambda_i$ 用了哪兩個額外假設？理論給出常數 $c$ 嗎？實測證據是什麼？ | $\bar A_i\approx\bar A$、$\bar C_i\propto\lambda_i$（Eq. 24）；不給 $c$；JiT-H/16 學到 $w_i\approx0.15\lambda_i$；A.3 以凍結網路上的輔助線性讀出估計（一致性檢查、非直接量測，60 個方向中依重建準確度保留 18 個）得到 $\bar C^{\rm eff}\propto\lambda^{1.00}$、$\bar A^{\rm eff}\propto\lambda^{-0.09}$、比值 $\lambda^{1.09}$，學到的權重指數 1.07 | S7 |
| LO10 | 描述 spectral update | 基底怎麼更新？網路輸出為零時第一次更新得到什麼？ | $B\leftarrow$ $G=\mathbb E[a_trr^\top-b_t(r\epsilon^\top+\epsilon r^\top)]$ 的特徵向量，升冪排列並以 rank 綁權重，$G$ 以 EMA（$\beta=10^{-5}$）追蹤（Eq. 40–41、Remark 3、App. B.1）；網路輸出為零時（初始化時常見）$r=x$，$G\propto\mathbb E[xx^\top]$，因 $G$ 初始為 $I$，第一次更新是單一 minibatch（所有圖的所有 patch）估計的帶雜訊 PCA 基底（App. A.4.2） | S8 |
| LO11 | 讀出實驗結果 | ImageNet 256 主結果、凍結 PCA 基底與 AsymFlow 的 FID、時間區間消融、latent 與 T2I 的結論？ | JiT-H/16 1.90→1.55（Spectral 無 init，600 ep）；同骨幹對照中凍結 PCA 基底 1.63、AsymFlow 重跑 1.76（Table 2 的 AsymFlow 1.57 是使用預訓練表徵編碼器的 † 已發表數字，設定不同）；共用權重 4.92 對隨時間 4.96（JiT-B/16，160 ep）；latent 持平（SiT 1.44/1.44、RAEv2 1.05/1.05）；T2I 除 GenEval（0.820→0.814）外皆改善（Table 1、3、4，§6.1.2） | S9 |
| LO12 | 列出實作要點與常見錯誤 | 權重怎麼參數化與初始化？損失算在哪？量化穩定性的證據是什麼？ | $w_i=\exp(\eta_i)$，預設 $B=I$、$w_i=1$（可選 PCA init $w_i=0.15\lambda_i$）；損失算在還原後的速度；推論分母下限 0.04；量化穩定性只有 W8A8 的定性圖（Fig. 1b、6b），沒有量化指標（App. B.1、B.3） | S10 |

---

## 3. 頁面大綱

單欄閱讀（文字欄 680–720px），元件放在加寬圖版（最大 1080px）。章節編號以 JetBrains Mono 顯示（`01 · 路徑與目標`），標題一律為陳述句。視覺記號：一條對數刻度的水平「w 軸」，左端「0（x）」、中間「1（速度）」、右端「→ ∞（ε）」，在 hero、S2、C1、C3 重複出現（與 No. 238 的 k 軸呼應）。每個符號在第一次使用前定義；每個元件出現前，所需概念已在內文講過。

| # | id | 標題 | 內容（依序） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈AnisoFlow：每個方向各學一個雜訊權重〉 | 副標一句；w 軸（見 hero 規格）；三張事實卡；連結 arXiv、`../index.html#paper-239`（在關聯圖中查看）、筆記總覽 `aniso-lineage-overview.html` | 1 螢幕 |
| S1 | `basics` | 01 · Flow matching 讓網路回歸速度 | Eq. 1–2、$t$ 方向（醒目 callout：與 No. 238 相反）、取樣方向；三種目標與換算、$1/\sigma$ 放大；連結 No. 238 `prediction-targets-k-diff.html` 作為共同背景，並寫出 $w=(1-k)/k$〔本頁推導〕；符號表（可收合）。LO1 答案段落 | 450–550 字 |
| S2 | `family` | 02 · 目標族只需要雜訊權重 | 說明「非負」是論文的族定義、實作用 $w_i=\exp(\eta_i)$ 故嚴格為正（一句）；先用兩句定義「方向 $i$ 的資料變異數 $\lambda_i=\mathbb E[x_i^2]$」與「PCA 基底＝資料共變異數的正交特徵向量，依變異數排序」（S7、S8 再深入）；k-Diff 與 AsymFlow 各一句定義（k-Diff：所有方向共用一個可學純量，見 No. 238；AsymFlow：PCA 前幾個方向用速度、其餘用 x 預測，見筆記）；App. A.1 正規化（純量→矩陣）；Eq. 3–4；特例；patch $D=768$、$I_N\otimes W$；靜態圖「目標族」（Fig. 2b 的示意：五個小格，各畫 24 根長條，$w_i$ 分別為全 0、全 1、全 0.5、前 6 根為 1 其餘 0、以及 $0.15\lambda_i$ 的示意曲線；$\lambda_i$ 用 $\lambda_i=50\cdot0.75^{i}$（$i=0..23$）合成，圖說寫「示意，$\lambda$ 為合成值，論文未提供數值」；長條依 $\lambda$ 由大到小排列）。LO2 答案段落。延伸：`aniso-interpolated-targets-edm-kdiff-asymflow.html`、`aniso-pixel-patchified-setting.html` | 350–450 字 + 靜態 SVG |
| S3 | `recovery` | 03 · 閉式還原把預測換回速度 | Eq. 9 推導（係數逐項）、Eq. 5/10、三個代入檢查（LO3 段落）；誤差傳遞與放大倍率、$\sigma_{\min}$ 截斷會改變分母本身（LO5 段落）；〔本頁推導〕兩路線混合；C1；C1 之後的小節「03.2 為什麼限制為半正定」：PSD 理由與 $W=-I$ 在 $t=1/2$ 奇異的小例（LO4 段落；此項只由段落教，不設元件，見 8 未採納 B15）。延伸：`aniso-target-normal-form-and-recovery.html` | 550–650 字 + C1 |
| S4 | `loss` | 04 · 權重不改變最小損失，只改變擬合難度 | 損失在速度空間；Eq. 6、11、12；梯度來自不完美；線性模型 Eq. 43–44 與「玩具學不出 $w\propto\lambda$」的明確說明；C2。LO6 答案段落 | 450–550 字 + C2 |
| S5 | `frozen` | 05 · 凍結誤差模型給出單一時刻的偏好權重 | 先接 S4：「超額誤差」就是網路誤差的二階矩，把它拆成雜訊估計誤差 $A$ 與資料估計誤差 $C$（假設 (a)(b)，Eq. 13–14）；並定義相關係數 $\rho=\Gamma/\sqrt{AC}$；先只講 $\Gamma=0$：Eq. 19、作者解讀、〔本頁推導〕逆變異數讀法；C3（預設 ρ = 0）；再一段介紹交叉項 Γ、Eq. 17–18 與〔本頁推導〕邊界情形，對應 C3 的「進階」開關。LO7 答案段落 | 450–550 字 + C3 |
| S6 | `time` | 06 · 在可分離誤差假設下，最佳權重與 t 無關 | 第一段：訓練時 $t\sim p(t)$、權重跨時間共用 ⇒ 最小化時間平均（Eq. 16），並舉均勻與 logit-normal 兩種 $p(t)$；動機；Eq. 20–23（標明是充分條件）；Fig. 5 與 Fig. 8c 證據；無資訊極限 Eq. 25–26；〔本頁推導〕均勻 $t$ 時 $\sqrt\lambda$；EDM Eq. 27–30；C4。LO8 答案段落 | 550–650 字 + C4 |
| S7 | `spectrum` | 07 · 再加兩個假設得到 w 與變異數近似成正比 | Eq. 24 兩假設；$c$ 不被預測；0.15（H/16）與 0.2（B/16）；A.3 一致性檢查（設定、18 方向；先用一句定義有效誤差 $A^{\rm eff}_i=A_i+\frac{\alpha}{\sigma}\Gamma_i$、$C^{\rm eff}_i=C_i+\frac{\sigma}{\alpha}\Gamma_i$（Eq. 35，由樣本殘差算出、不可約項相消），以及時間平均 $\bar A^{\rm eff}_i=\mathbb E_t[A^{\rm eff}_i/\alpha_t]$、$\bar C^{\rm eff}_i=\mathbb E_t[C^{\rm eff}_i/\sigma_t]$（Eq. 37）；指數、0.97、門檻放寬）；Fig. 9 瓶頸；Fig. 10 耦合。一張 2 欄表「A.3 量測」（項目、值）。**不放元件**（玩具無法呈現湧現，S4 已說明）。LO9 答案段落。延伸：`aniso-spectral-weight-analysis.html` | 450–550 字 + 表 |
| S8 | `basis` | 08 · 基底由一個動差矩陣的特徵向量更新 | 本文只留：$r$ 的定義、Eq. 40–41 的 $G$、$B\leftarrow\mathrm{eigvecs}(G)$、升冪排序與 rank 綁權重（Remark 3）、EMA（$\beta=10^{-5}$）、第一次更新＝帶雜訊 PCA、初始化；C5。LO10 答案段落。`<details>`「進階」：Stiefel Adam 一句、Eq. 39 與「一般不是特徵值問題」一句、假設 1＋命題 2 一句、A.5 的不動點與 $|\rho_{jk}|\lt1$ 一句 | 400–500 字 + C5 + 進階 |
| S9 | `results` | 09 · 實驗：像素空間 FID 由 1.90 降到 1.55 | Table 1 全表；Table 2 摘錄（5 列）；時間區間 4.92/4.96；T2I Table 3；量化（定性）；latent Table 4；開銷。注意事項 callout（AsymFlow 1.76 為同骨幹重跑、1.57 為 † 已發表數字；協定 *；無誤差棒；Table 5 逐模型 guidance；GenEval 變差）。LO11 答案段落。延伸：`aniso-quantization-conditioning.html`、`aniso-latent-models-check.html` | 450–550 字 + 4 張表 |
| S10 | `impl` | 10 · 實作設定與常見錯誤 | 步驟 12 的 8 條；一條回指 S9 的量化證據（W8A8、Fig. 1b／6b，只有定性圖）；Table 7（新增狀態與運算）摘要。LO12 答案段落 | 300–400 字 + 清單 |
| S11 | `history` | 11 · 研究脈絡 | 時間線（6.6）+ 一段文字沿 w 軸串接：速度（flow matching）→ ε／v 的選擇（Progressive Distillation、EDM）→ x（JiT）→ 純量內插（k-Diff）與二元（AsymFlow）→ 逐方向（本文） | 時間線 + 100–150 字 |
| S12 | `next` | 12 · 限制與延伸閱讀 | §7 Limitations、App. A.5 範圍聲明；筆記清單（6.6） | 200 字 + 清單 |

Hero 規格（S0）：
- 副標：「這篇論文讓 flow model 的預測目標在每個方向各有一個可學習的雜訊權重，並分析學到的權重為什麼大約與資料變異數成正比。」
- 三張事實卡（每張只用卡內自行定義的詞）：
  1. 「網路不一定要直接預測速度。AnisoFlow 讓網路預測 Wε − x：x 是乾淨影像，ε 是加入的雜訊，W 在每個方向給雜訊一個可學習的非負權重。」
  2. 「網路輸出可以用一條精確公式換回速度。因此不論權重取多少，可能達到的最小損失都一樣；權重只改變網路要擬合的函數有多難。」
  3. 「在像素空間，學到的權重大約是該方向資料變異數的 0.15 倍。ImageNet 256×256 的 FID（越低代表生成影像的統計越接近真實影像）由 1.90 降到 1.55。」
- w 軸（HTML 實作，沿用 No. 238 hero k 軸的標籤分層演算法）：對數刻度 $10^{-3}$ 到 $10^{2}$，左端外側另放「0」刻度（x 預測），刻度 1 標「速度」，右端箭頭標「→ ∞ ε」；軸下方一行註記「像素空間學到的權重約為該方向資料變異數 λ 的 0.15 倍（Fig. 3a）」，不畫範圍區段（範圍只能目視，避免變成校準值）。

桌機（≥ 1024px）左側 sticky 章節目錄；< 1024px 頂部橫向章節列（同 No. 238）。

---

## 4. 互動元件規格

### 4.0 共用規則

- **圖版第一行**（`.plate-lead`）寫出該玩具的假設與「它不能證明什麼」。C2–C4 皆須含「此玩具不能學出 $w\propto\lambda$」或等義句（論文 A.5：線性/高斯模型的損失與 $w$ 無關）。
- **PRNG（照抄，與 No. 238 相同）**：
  ```js
  function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
  function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}
  ```
  seed 239 的前 4 個均勻數為 0.105665、0.373234、0.519065、0.387548（`check.py` 與 node 皆已核對）。
- **離散值滑桿**（保證驗收值為精確小數）：`<input type=range>` 的值是索引，顯示值查表。
  - `W_VALUES = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 20, 50, 100]`
  - `POS_VALUES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2, 4, 5, 10, 20, 50, 100]`（含 C2 預設 4）（λ、$A$、$C$、$\bar C/\bar A$ 共用）
- **觀察文字**：不得寫死針對特定輸入值的觀察句。每個元件的「目前狀態」句由計算結果以模板產生（例：「目前 w* = {w*}，比 w = 1 的損失低 {pct}%」）；第 4 節列的「讀者應觀察到」只放進內文作為一般敘述，或寫成「試試看：把 ρ 調到負值」這類操作提示。
- **表格**：讀數表每格「標籤在前、值在後」（例：`<th>w*</th><td>0.1000</td>`，或卡片內 `w* = 0.1000`）。
- **無障礙**（全部元件）：每個滑桿有 `<label>` 與 `aria-valuetext`；SVG `role="img"` + `aria-label`；**視覺讀數表與圖每次 `input` 即時更新，讀數表本身不設 aria-live**；每個圖版另有一個 `.sr-only` 的 `aria-live="polite"` 區，只放一句摘要（例：「t = 0.30，w = 1，速度誤差 0.1000」），在 `change` 或輸入停止 500ms 後更新（`announce()`）；SVG 的 `aria-label` 也在同一時機更新；曲線以線型＋標記形狀＋直接文字標籤區分；沒有任何資訊只靠 hover 取得（數值都在讀數表；觸控可用滑桿操作）；觸控區 ≥ 44px。
- 色彩：資料／x 路線用 `--c-par`（實線、圓形標記），雜訊／ε 路線用 `--c-perp`（虛線、方形標記），目前選取值用 `--accent`。

### C1 · 目標、還原與誤差放大（S3）

- **教的概念**：Eq. 5 的還原是精確的；目標誤差 $\delta$ 換到速度時變成 $\delta/(\alpha w+\sigma)$；$w=0$ 在低雜訊時放大 $1/t$；$\sigma_{\min}$ 截斷；〔本頁推導〕兩路線混合比例。
- **圖版第一行**：「純量示範：抽一組 x、ε ~ N(0, 1)，給網路的目標預測加上一個固定誤差 δ，看還原後的速度誤差。這裡不訓練網路。」
- **輸入**：$t$ 滑桿 0.01–1.00，步長 0.01，預設 0.30；$w$（`W_VALUES`，預設 1）；chip「x 預測 w = 0」「速度 w = 1」「w = 10」；$\delta$ 滑桿 −0.50–0.50，步長 0.01，預設 0.10；「重新抽樣」（seed 由 239 起，每按 +1）；勾選「推論截斷 σ_min = 0.04」（預設**不**勾選）。截斷依 App. B.1 作用在**分母本身**：勾選時 $den'=\max(den,0.04)$ 進入還原 $\hat u$，因此改變 û、速度誤差與圖中目前 $w$ 的曲線。
- **計算**（每次輸入即時）：`r = mulberry32(seed); x = gauss(r); e = gauss(r)`；$\alpha=1-t$、$\sigma=t$；$z=\alpha x+\sigma e$；$u_W=we-x$；$\hat u_W=u_W+\delta$；$den=\alpha w+\sigma$；$\hat u=((1-w)z+\hat u_W)/den$；$u=e-x$；誤差 $\hat u-u$；放大 $1/den$；$\pi_\epsilon=\alpha w/den$、$\pi_x=\sigma/den$。勾選截斷時改用 $den'=\max(den,0.04)$：$\hat u'=((1-w)z+\hat u_W)/den'=(den\cdot u+\delta)/den'$，誤差拆成兩項〔本頁推導〕：偏差 $u\,(den/den'-1)$（即使 $\delta=0$ 也存在）＋放大後誤差 $\delta/den'$（倍率上限 25）。
- **數值保護**：$t\ge0.01$ 所以 $den\ge0.01$；讀數格式見 6.1 的 `fmt`。
- **顯示**：
  - 左：讀數表（兩欄：項目、值）：α、σ、x、ε、z、u_W、û_W、還原 û、真實 u、使用的分母、速度誤差 û − u、誤差倍率、π_ε、π_x；勾選截斷時再加兩列「截斷偏差 = u(den/den′ − 1)」「放大後誤差 = δ/den′」。表下模板句與表格用**同一個分母**：未截斷「速度誤差 = δ / den = {δ} / {den} = {err}」；截斷「速度誤差 = 偏差 {bias} + δ / den′ = {bias} + {δ} / {den′} = {err}」。
  - 右：放大倍率曲線，橫軸 $t\in(0,1]$（取 $t_j=j/400$，$j=4..400$，即 $t\in[0.01,1]$，與滑桿下限一致；$w=0$ 的最大值 $1/0.01=100$），縱軸對數 $[10^{-2},10^{2}]$；所有曲線以 `clipPath` 裁到繪圖區，超出部分不畫（在此格點與 `W_VALUES` 下，$1/den\in[1/100.0,100]$ 必在範圍內，裁切只是保險）。三條：$w=0$（`--c-par` 虛線＋每 50 個格點一個圓形標記，標「x 預測：1/t」）、$w=1$（`--ink-3` 點線＋方形標記，標「速度：1」）、目前 $w$（`--accent` 實線、線寬比參考線多 1px，三角形標記，標「w = {w}」；若目前 $w$ 等於 0 或 1，標籤改為「目前 = x 預測」或「目前 = 速度」，不另畫重疊的標籤）；勾選截斷時目前曲線在 25 處截平，另畫水平細線 $y=25$ 標「σ_min = 0.04 → 25×」。目前 $t$ 以垂直細線標示。
- **讀者應觀察到**（一般敘述）：不截斷時，還原 û 減去 δ 的影響後恆等於真實 u；截斷生效時（$den\lt0.04$），即使 δ = 0 也有偏差；$w=1$ 時放大倍率恆為 1；$w=0$、$t\to0$ 時放大倍率接近 $1/t$；$w>1$ 時放大倍率小於 1。

### C2 · 線性模型：權重改變目標，不改變還原後的速度（S4）

- **教的概念**：Eq. 11 與 Eq. 43–44 的純量版：拖動 $w$，最佳目標預測（左圖迴歸線）改變，還原後的速度預測（右圖迴歸線）不變；最小速度誤差與 $w$ 無關。
- **圖版第一行**：「假設：x ~ N(0, λ)，網路是最佳線性預測器。這時還原後的損失與 w 無關（論文 Eq. 44），所以這個玩具學不出任何 w，也不能用來說明 w ∝ λ。」
- **輸入**：λ（`POS_VALUES`，預設 4）；$t$ 0.01–0.99，步長 0.01，預設 0.50；$w$（`W_VALUES`，預設 1）；「重新抽樣」（seed 由 239 起）。
- **計算**：$v_z=\alpha^2\lambda+\sigma^2$；$\Phi^*=(\sigma w-\alpha\lambda)/v_z$；還原係數兩種算法並列：`kRec = ((1-w)+Phi)/(alpha*w+sigma)`、`kCF = (sigma-alpha*lam)/v_z`；最小速度誤差 $\lambda/v_z$；目標空間最小誤差 $(\alpha w+\sigma)^2\lambda/v_z$。散佈點 200 個：每點依序 `g1 = gauss(r), g2 = gauss(r)`，$x=\sqrt\lambda g_1$、$\epsilon=g_2$、$z=\alpha x+\sigma\epsilon$；左圖畫 $(z,\ w\epsilon-x)$，右圖畫 $(z,\ \epsilon-x)$。
- **數值保護**：$v_z\ge\min(\alpha,\sigma)^2\cdot\min(\lambda,1)>0$（$t\in[0.01,0.99]$）。讀數「|kRec − kCF|」顯示科學記號，預期 < 1e−12。
- **顯示**：兩個正方形散佈圖（≥ 600px 左右並排；< 600px 上下堆疊，每張寬度等於容器寬）。左圖點用圓形、右圖點用方形，兩圖各有標題「目標 u_W 與 z」「速度 u 與 z」。橫軸皆為 $\pm3\sqrt{v_z}$；左圖縱軸 $\pm3\sqrt{w^2+\lambda}$（隨 $w$ 重設，刻度標數字），右圖縱軸固定 $\pm3\sqrt{1+\lambda}$。左圖畫直線 $y=\Phi^*z$（`--accent`），右圖畫 $y=\kappa z$（`--ink` 粗線）。讀數表：Φ*、還原係數（經 Eq. 5）、還原係數（閉式）、|差|、最小速度誤差、目標空間最小誤差。模板句：「目標空間的最小誤差為 {tgt}，換回速度後為 {vel}；後者在任何 w 下都相同。」
- **讀者應觀察到**：左圖的點雲與迴歸線隨 $w$ 改變；右圖完全不動；最小速度誤差不變而目標空間誤差隨 $w$ 改變。

### C3 · 凍結誤差模型：單一時刻的偏好權重（S5）

- **教的概念**：Eq. 17–19；交叉項 $\Gamma$ 的影響；邊界情形〔本頁推導〕；$\Gamma=0$ 時的逆變異數讀法〔本頁推導〕。
- **圖版第一行**：「假設：凍結誤差模型（Eq. 13），A、C、ρ 由你指定，不是從網路量到的。圖只顯示『給定這些誤差，哪個 w 最好』，不能說明網路實際的誤差為何如此。真實網路的 A、C 會隨 t 變化（論文 §5：A 隨 t 下降、C 隨 t 上升），兩者的抵銷見 S6；此處固定 A、C 只為看公式。」
- **輸入**：$t$ 0.05–0.95，步長 0.05，預設 0.50；$A$（`POS_VALUES`，預設 1）；$C$（`POS_VALUES`，預設 0.1）；勾選「進階：交叉項 Γ」（預設不勾，ρ = 0，圖與讀數只顯示 Eq. 19、f* 與內部最小）；勾選後才出現相關係數 ρ −0.9–0.9，步長 0.1，預設 0；$\Gamma=\rho\sqrt{AC}$（保證 $|\Gamma|\lt\sqrt{AC}$），並顯示情形判斷與邊界標記。
- **計算**：$f(w)=(Aw^2-2\Gamma w+C)/(\alpha w+\sigma)^2$；$P=A\sigma+\alpha\Gamma$、$Q=\Gamma\sigma+\alpha C$；$P>0,Q>0$：$w^*=Q/P$（內部）；$P>0,Q\le0$：最小在 $w=0$；否則：下確界在 $w\to\infty$，值 $A/\alpha^2$。另算 Eq. 19 近似 $(\alpha/\sigma)C/A$、$f(0)=C/\sigma^2$、$A/\alpha^2$、$f(1)$；ρ = 0 時另顯示 $f^*=1/(\alpha^2/A+\sigma^2/C)$（〔本頁推導〕標籤），並與 $f(w^*)$ 並列。
- **顯示**：主圖橫軸對數 $w\in[10^{-3},10^{3}]$（另於左端外側畫 $w=0$ 的點，標「w = 0：C/σ²」），縱軸對數，範圍取曲線在畫面內的 min/max 各外擴 20%。右端畫水平虛線 $A/\alpha^2$ 標「w → ∞」。$w^*$ 用 `--accent` 實心點＋垂直虛線；若 $w^*\notin[10^{-3},10^3]$，在對應邊緣畫三角形箭頭並標數值；邊界情形則在 $w=0$ 點或右端虛線上加環形標記並寫「最小值在邊界」。Eq. 19 近似（ρ ≠ 0 時）以空心菱形標出。圖上方嵌一條 w 軸小刻度（0／1／∞）。
- **讀數表**：情形（內部／w = 0／w → ∞）、w*（Eq. 18）、Eq. 19 近似、f(w*)、f(0) = C/σ²、f(1)、A/α²、（ρ = 0）f* 逆變異數式。模板句依情形產生，例如「內部最小：w* = {w*}，損失比速度目標（w = 1）低 {100(1−f(w*)/f(1))}%」。
- **讀者應觀察到**：增大 $C$ 或減小 $A$，$w^*$ 變大；（固定 A、C 時）$t$ 變小使 $w^*$ 變大，這正是 S6 要說明的「看似隨時間改變」，真實網路中 A、C 的時間變化會抵銷它；ρ ≠ 0 時 Eq. 18 偏離 Eq. 19；ρ 夠負時最小值落到邊界。

### C4 · 時間抵銷 vs 無資訊極限（S6）

- **教的概念**：Eq. 20–23 下所有時刻共用最小點 $\bar C/\bar A$，時間平均的最佳常數權重不受時間分佈影響；無資訊極限（Eq. 25–26）下逐時刻最小點為 $(\alpha/\sigma)\lambda$，最佳常數權重隨時間分佈改變；EDM 對應（Eq. 27–30）。
- **圖版第一行**（依模式切換）：可分離：「假設：Eq. 20 的可分離誤差，取 Ā = 1、q(t) ≡ 1、Γ = 0。這是論文的充分條件，不是從網路推得；玩具只算『若成立，最小點在哪』，不能學出 w ∝ λ。」無資訊極限：「假設：網路對這個方向沒有資訊，目標預測恆為 0（Eq. 25）。」
- **輸入**：模式分段按鈕「可分離誤差（Eq. 20）」「無資訊極限（Eq. 25）」，預設可分離；參數 $\bar C/\bar A$ 或 λ（`POS_VALUES`；可分離預設 0.1、無資訊極限預設 1；切換模式時各自記住）；時間分佈分段按鈕「均勻」「logit-normal」，預設均勻；logit-normal 的 μ −1.5–1.5，步長 0.5，預設 0，$s=1$ 固定（定義在本論文的 $t$：$\mathrm{logit}(t)=\ln\frac{t}{1-t}\sim\mathcal N(\mu,1)$，μ > 0 偏向雜訊端）。頁面註明「此分佈只是示範用的選項，未含 flow shift；論文 A.3 只說訓練用 logit-normal，Table 6 的 T2I 為 LogitNormal(0, 1) 再加 flow shift，RAEv2 為均勻取樣再 shift，所以這裡的 μ 不能與論文設定直接比較」。
- **計算**：
  - 時間格點 $t_j=(j+0.5)/1024$，$j=0..1023$；權重 $p_j$：均勻為 1，logit-normal 為 $\frac{1}{\sqrt{2\pi}t_j(1-t_j)}e^{-(\mathrm{logit}t_j-\mu)^2/2}$；再正規化使 $\sum p_j=1$。
  - 逐時刻曲線 $f_t(w)$：可分離 $(\alpha w^2+\sigma r)/(\alpha w+\sigma)^2$（$r=\bar C/\bar A$）；無資訊極限 $(w^2+\lambda)/(\alpha w+\sigma)^2$。逐時刻最小點用解析式：可分離 $r$，無資訊極限 $(\alpha/\sigma)\lambda$；最小值以解析最小點代回。
  - 時間平均 $\bar f(w)=\sum_jp_jf_{t_j}(w)$；搜尋範圍比圖寬：$\log_{10}w\in[-4,4]$、步長 0.002（4001 點）取最小索引 $i$；$i=0$ 或 $4000$ 時不修正，讀數顯示「≤ 10⁻⁴」或「≥ 10⁴（超出搜尋範圍）」；否則若 $y_{i-1}-2y_i+y_{i+1}>0$，以拋物線修正 $\log_{10}\hat w\mathrel{+}=0.002\cdot\frac12\frac{y_{i-1}-y_{i+1}}{y_{i-1}-2y_i+y_{i+1}}$。$\hat w$ 超出圖的 $[10^{-3},10^3]$ 時在邊緣畫箭頭並標數值。（驗收組合中最極端的 λ = 100、μ = −1.5 為 108.6，在範圍內。）
  - 排程：核心函式 `c4Search(mode, par, sampler, mu)` 是純同步函式。互動時以 `chunkRun()` 分塊執行（每塊 400 個 $w$ 格點，塊間 `setTimeout(0)`），計算期間圖版右上顯示「計算中…」並保留上一組結果；新輸入到來時取消舊工作（遞增 job id）。結果以 `Map` 快取（key `${mode}|${par}|${sampler}|${mu}`）。`?test` 直接呼叫同步的 `c4Search`，不經分塊或 Worker。
  - EDM（只在無資訊極限）：$t=0.5$ 時 $s=\sigma/\alpha$、$c_{\rm skip}=\lambda/(\lambda+s^2)$、$w=c_{\rm skip}s/(1-c_{\rm skip})$，並列 $(\alpha/\sigma)\lambda$。
- **數值保護**：$w\ge10^{-4}$ 使所有分母 $\ge\min(10^{-3},\cdot)>0$；$t_j\in(0,1)$ 不含端點；可分離在 $w\to0$ 時 $\bar f$ 有 $\int r/t$ 的對數發散，格點下界避開。
- **顯示**：主圖橫軸對數 $w\in[10^{-3},10^{3}]$，縱軸**對數** $f_t(w)/\min_wf_t(w)\in[1,100]$（正規化後各時刻曲線可比；預設可分離 r = 0.1 在大 w 約 90、無資訊極限 t = 0.9 在 w = 10 約 23）；曲線以 `clipPath` 裁到繪圖區，超過 100 的部分在上緣截斷並在截斷處畫一個向上小三角形。五條曲線 $t=0.1,0.3,0.5,0.7,0.9$，以五種線型（實、長虛、短虛、點、點虛）＋標記形狀（圓、方、三角、菱、叉）＋圖右側（< 600px 時圖下方）的圖例區分（不在曲線端點放文字，避免重疊）；每條曲線的最小點畫對應標記。時間平均最佳 $\hat w$ 以 `--accent` 垂直粗線標「時間平均最佳 ŵ」。副圖（只在 logit-normal）：時間權重 $p(t)$ 正規化到最大值 1。
- **讀數表**：t = 0.1 … t = 0.9 的逐時刻最小點（五列，標籤在前）、時間平均最佳 ŵ、（可分離）C̄/Ā、（無資訊極限）EDM：s、c_skip、w。模板句：可分離「五個時刻的最小點都是 {r}；時間平均最佳 ŵ = {ŵ}」；無資訊極限「逐時刻最小點由 {w(0.1)} 變到 {w(0.9)}；時間平均最佳 ŵ = {ŵ}，換時間分佈後會改變」。
- **讀者應觀察到**：可分離模式下五個標記重疊、換時間分佈 $\hat w$ 不動；無資訊極限模式下標記沿 w 軸散開、換 μ 時 $\hat w$ 移動；均勻分佈下無資訊極限的 $\hat w=\sqrt\lambda$〔本頁推導〕。

### C5 · Spectral update 的第一步是帶雜訊的 PCA（S8）

- **教的概念**：$W=I$、網路輸出為零時 $r=x$、$a_t=b_t=1$，$\hat G=\frac1n\sum(xx^\top-(x\epsilon^\top+\epsilon x^\top))$，其主特徵向量即第一次更新的基底方向；batch 越大越接近 PCA 主軸；兩個變異數接近時方向不穩（特徵值間隙小）。
- **圖版第一行**：「設定：D = 2 的高斯資料，W = I，網路輸出為 0（論文說這在初始化時常見，並非保證）。只示範第一次 spectral update，不示範之後的訓練。實際訓練的 n = 圖數 × 每張圖的 patch 數（Eq. 42，量級在數萬以上），這裡的小 n 只為看趨勢。」
- **輸入**：$\lambda_2$（分段：0.05、0.1、0.25、0.5、0.8、0.95；預設 0.25），$\lambda_1=1$ 固定；主軸角度 θ 0°–175°，步長 5°，預設 30°；batch 大小 $n$（分段：8、32、128、512、2048；預設 32）；「重新抽樣」（seed 由 239 起，每按 +1）。
- **計算**：$\Sigma=R(\theta)\mathrm{diag}(1,\lambda_2)R(\theta)^\top$。每個樣本依序 `ga, gb, e1, e2 = gauss(r)×4`；$p=\sqrt{\lambda_1}g_a$、$q=\sqrt{\lambda_2}g_b$、$x=(c p-s q,\ s p+c q)$（$c=\cos\theta$、$s=\sin\theta$）；累加 $g_{11}\mathrel{+}=x_1^2-2x_1e_1$、$g_{22}\mathrel{+}=x_2^2-2x_2e_2$、$g_{12}\mathrel{+}=x_1x_2-(x_1e_2+e_1x_2)$，最後除以 $n$。主特徵向量角度 $\phi=\frac12\mathrm{atan2}(2g_{12},g_{11}-g_{22})$；誤差 $\phi-\theta$ 換成度並包到 $[-90°,90°)$：`((d+90)%180+180)%180-90`。統計：以同一 seed 重新建立 PRNG，連續抽 200 個 batch（第一個即顯示的那個），取 |誤差| 的中位數（排序後第 100 與 101 個的平均）。核心 `c5Stats(lam2, theta, n, seed)` 為同步函式；互動時以 `chunkRun()` 每塊 10 個 batch 分塊（n = 2048 時共 1.6M 次 gauss），期間顯示「計算中…」、單次結果先畫、直方圖完成後再畫；`?test` 同步呼叫。
- **數值保護**：$g_{12}=0$ 且 $g_{11}=g_{22}$（重根）時不畫方向，讀數顯示「重根：方向不確定」。
- **顯示**：正方形散佈圖 $[-3,3]^2$，畫本 batch 的 $x$ 點（最多畫 512 點，超過時只畫前 512 點並註明）；PCA 主軸（`--c-par` 虛線，標「PCA 主軸」）與 $\hat G$ 主特徵向量（`--accent` 實線，標「第一次更新」），兩者皆畫成穿過原點的雙向線段；副特徵向量畫成細線。右側：200 次 |誤差| 的直方圖（0°–90°，每 5° 一格），中位數以垂直線標示。讀數表：本次角度誤差、200 次中位數 |誤差|、$\hat G$ 的兩個特徵值（升冪，對應 Remark 3 的 rank 1、2）。
- **讀者應觀察到**：$n$ 變大，中位數誤差變小；$\lambda_2$ 接近 1 時誤差變大（特徵值間隙小）；升冪排列下主軸是 rank 2。

---

## 5. 驗收數值表

全部由 `verify/check.py` 產生（`python3 check.py`，約 20 秒；輸出 `check_out.txt`）。頁面與腳本用相同算法與 PRNG，容許誤差只涵蓋浮點加總順序。

### 5.1 C1（seed 239：x = −0.3305、ε = −0.9203；δ = 0.10）

未截斷：

| t | w | z | u_W | 還原 û | 速度誤差 | 誤差倍率 | π_ε | π_x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.30 | 1 | −0.5074 | −0.5899 | −0.4899 | 0.1000 | 1.0000 | 0.7000 | 0.3000 |
| 0.30 | 0 | −0.5074 | 0.3305 | −0.2565 | 0.3333 | 3.3333 | 0.0000 | 1.0000 |
| 0.05 | 0 | −0.3599 | 0.3305 | 1.4101 | 2.0000 | 20.0000 | 0.0000 | 1.0000 |
| 0.05 | 10 | −0.3599 | −8.8727 | −0.5794 | 0.0105 | 0.1047 | 0.9948 | 0.0052 |
| 0.01 | 0 | −0.3363 | 0.3305 | 9.4101 | 10.0000 | 100.0000 | 0.0000 | 1.0000 |

勾選截斷：上表前四列的 den ≥ 0.04，數值不變；t = 0.01、w = 0：den′ = 0.0400、還原 û′ = 2.3525、速度誤差 2.9424 = 截斷偏差 0.4424 + 放大後誤差 2.5000（倍率 25）。真實 u = −0.5899。容許 ±0.0001。

### 5.2 C2（λ = 4，t = 0.5；與 seed 無關）

| w | Φ* | 還原係數 | 最小速度誤差 | 目標空間最小誤差 |
| --- | --- | --- | --- | --- |
| 0 | −1.6 | −1.2 | 3.2 | 0.8 |
| 0.1 | −1.56 | −1.2 | 3.2 | 0.968 |
| 1 | −1.2 | −1.2 | 3.2 | 3.2 |
| 10 | 2.4 | −1.2 | 3.2 | 96.8 |

另：λ = 0.04、t = 0.5：還原係數 1.846154、最小速度誤差 0.153846（w = 0 時 Φ* = −0.076923，w = 10 時 19.153846）；λ = 4、t = 0.1：還原係數 −1.076923、最小速度誤差 1.230769。容許 ±1e−6；|kRec − kCF| < 1e−12。

### 5.3 C3

| t | A | C | ρ | 情形 | w*（Eq. 18） | Eq. 19 | f(w*) | f(0) | f(1) | A/α² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 1 | 0.1 | 0 | 內部 | 0.1000 | 0.1000 | 0.3636 | 0.4000 | 1.1000 | 4.0000 |
| 0.5 | 1 | 0.1 | 0.5 | 內部 | 0.2229 | 0.1000 | 0.2118 | 0.4000 | 0.7838 | 4.0000 |
| 0.5 | 1 | 0.1 | −0.5 | w = 0 | 0 | 0.1000 | 0.4000 | 0.4000 | 1.4162 | 4.0000 |
| 0.2 | 1 | 0.1 | 0 | 內部 | 0.4000 | 0.4000 | 0.9615 | 2.5000 | 1.1000 | 1.5625 |
| 0.8 | 1 | 0.1 | 0 | 內部 | 0.0250 | 0.0250 | 0.1553 | 0.1563 | 1.1000 | 25.0000 |
| 0.5 | 1 | 10 | −0.9 | w → ∞ | ∞ | 10.0000 | 4.0000 | 40.0000 | 16.6921 | 4.0000 |
| 0.5 | 0.01 | 10 | 0.9 | 內部 | 34.9098 | 1000.0000 | 0.0072 | 40.0000 | 9.4408 | 0.0400 |

ρ = 0 列的 f*（逆變異數式）與 f(w*) 相同。容許 ±0.0001（f(0) 在 t = 0.8 精確值為 0.15625，頁面 `toFixed(4)` 顯示 0.1563）。

### 5.4 C4

逐時刻最小點：可分離 r = 1.5 → 五個時刻皆 1.5；無資訊極限 λ = 1 → t = 0.1, 0.3, 0.5, 0.7, 0.9 依序 9.0000、2.3333、1.0000、0.4286、0.1111。

時間平均最佳 ŵ（容許相對誤差 ±0.5%）：

| 模式 | 參數 | 均勻 | LN μ = −1.5 | μ = −1 | μ = −0.5 | μ = 0 | μ = 0.5 | μ = 1 | μ = 1.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 可分離 | r = 1.5 | 1.5 | — | 1.5 | — | 1.5 | — | 1.5 | — |
| 可分離 | r = 0.1 | 0.1 | — | 0.1 | — | 0.1 | — | 0.1 | 0.1 |
| 無資訊極限 | λ = 1 | 1 | 2.457 | 1.764 | 1.316 | 1 | 0.7598 | 0.5668 | 0.4069 |
| 無資訊極限 | λ = 0.1 | 0.3162 | — | 0.3702 | — | 0.2169 | — | 0.1102 | — |
| 無資訊極限 | λ = 10 | 3.162 | — | 9.077 | — | 4.611 | — | 2.701 | — |

極端組合：無資訊極限 λ = 100、μ = −1.5 → 108.6；λ = 0.01、μ = 1.5 → 0.009211；λ = 100 均勻 → 10；λ = 0.01 均勻 → 0.1；可分離 r = 100、μ = −1.5 → 100；r = 0.01、μ = 1.5 → 0.01（皆在搜尋範圍內）。均勻欄等於 $\sqrt\lambda$（〔本頁推導〕的交叉檢查）。EDM：λ = 1、t = 0.5：s = 1、c_skip = 0.5、w = 1；λ = 4、t = 0.2：s = 0.25、c_skip = 0.9846、w = 16（= (α/σ)λ）。

### 5.5 C5（λ₁ = 1、θ = 30°、seed 239；角度單位為度）

| λ₂ | n | 本次誤差 | 200 次中位數 |誤差| |
| --- | --- | --- | --- |
| 0.25 | 8 | −23.69 | 20.09 |
| 0.25 | 32 | 12.49 | 11.16 |
| 0.25 | 128 | 0.25 | 5.68 |
| 0.25 | 512 | −2.61 | 2.79 |
| 0.25 | 2048 | −2.23 | 1.36 |
| 0.8 | 8 | −21.53 | 34.44 |
| 0.8 | 128 | 11.31 | 23.40 |
| 0.8 | 2048 | −5.27 | 6.65 |

容許 ±0.01°。

### 5.6 `?test` selfTest 斷言（頁面內）

- `mulberry32(239)` 前兩個均勻數為 0.105665、0.373234（±1e−6）。
- C1：t = 0.01、w = 0、δ = 0.1 → 未截斷速度誤差 10.0000；截斷時 û′ = 2.3525、偏差 0.4424、放大後誤差 2.5000；t = 0.3、w = 1 → π_ε = 0.7。
- C2：λ = 4、t = 0.5 對 `W_VALUES` 全部值，|kRec − (−1.2)| < 1e−12。
- C3：表 5.3 第 1、2、6 列（含情形字串）。
- C4（同步 `c4Search`）：無資訊極限 λ = 1 均勻 → ŵ = 1（±0.5%）；λ = 100、μ = −1.5 → 108.6（±0.5%）；μ = 1 → 0.5668（±0.5%）；可分離 r = 1.5、μ = −1 → 1.5（±0.5%）。
- C5（同步 `c5Stats`）：λ₂ = 0.25、n = 32 → 本次誤差 12.49（±0.01）、中位數 11.16（±0.01）。
- 版面：`document.documentElement.scrollWidth <= innerWidth`。
- 輸出一行 `selfTest: N passed, M failed`。

---

## 6. 實作計畫

### 6.1 檔案與外殼（只列與 No. 238 不同之處）

- 單一檔案 `tut/239/build/index.html`；CSS、JS、時間線全部內嵌，不 `fetch`。上線時複製為 `tutorials/anisoflow.html`。
- **整段沿用** No. 238 已上線頁（`tutorials/prediction-targets-k-diff.html`）的：`<head>`（Google Fonts、KaTeX 0.16.9 cdnjs + auto-render、`renderMath()` 設定與 `load` 後備）、CSS tokens（含深色模式與 `--c-par`、`--c-perp`、深色 `--ink-3` #9C978C）、appbar 與主題切換（`pm-theme`）、`.toc` 章節目錄、`.plate`/`.plate-h`/`.plate-lead`/`.plate-f`/`.readout`/`.ctl`/`.seg`/`.chip`/`.fact`/`.check`/`.tscroll`/`.eq`+`.eq-n` 元件樣式、`#pm-bar` 標記與樣式、時間線 `renderTimeline()` 的版面與篩選 chip、`el()`/`scaleLin`/`scaleLog`/`mulberry32`/`gauss`、`selfTest()` 骨架（`?test` 觸發、`console.assert`、最後一行摘要）。
- 新增／修改：
  - appbar 編號「No. 239」、短標題「AnisoFlow：可學習的各向異性預測目標」。
  - Hero 的 w 軸（取代 k 軸；標籤分層演算法沿用）。
  - S2 靜態「目標族」SVG（第 3 節規格）。
  - `.callout.conv`：S1 的時間方向對照框（兩欄：本論文 $z_t=(1-t)x+t\epsilon$、$t=0$ 資料｜No. 238 $z=tx+(1-t)n$、$t=1$ 資料），附連結 `prediction-targets-k-diff.html`。
  - 離散值滑桿 helper：`bindList(input, values, onChange)`（設定 `min=0 max=len-1 step=1`、`aria-valuetext`、顯示值）。
  - 元件：`initC1()`…`initC5()`，每個只碰自己的容器；C4 用 `Map` 快取；C5 的 200 次統計在 `change` 時算（≤ 200×2048×4 次 gauss，約 50ms）。
  - 讀數格式：`fmt(v)` = `|v|>=1e4 ? v.toExponential(3) : v.toFixed(4)`；`Infinity` 顯示「∞」。驗收表數字依此格式（例：1000.0000）。
  - 分塊執行 helper `chunkRun(steps, perChunk, onDone)`（`setTimeout(0)` 分塊、job id 取消）；`?test` 不經過它。
  - 文字 debounce：`announce(el, text)`，500ms。
- 站內連結：Hero「在關聯圖中查看」→ `../index.html#paper-239`；arXiv `https://arxiv.org/abs/2611.17767`；筆記 `<slug>.html`。

### 6.2 數學排版

- TeX 中的小於號一律寫 `\lt`（避免被 HTML 解析成標籤）；大於號可寫 `>`，但為一致建議 `\gt`。
- SVG 文字、讀數、tooltip 用 Unicode，不用 TeX。只用預組字元：ŵ（U+0175）、û（U+00FB），不用組合附標；下標用 `<tspan baseline-shift="sub" font-size="0.8em">`（σ<sub>min</sub>、u<sub>W</sub>、λ<sub>2</sub>），HTML 讀數用 `<sub>`。建置時截圖確認以下字元在 IBM Plex Sans / Noto Sans TC / JetBrains Mono 皆有字形：ŵ û α σ λ ε Γ ρ Φ κ π ∞ ∝ ≈ ≤ ≥ − × Ā C̄（C̄ 為組合字元，若缺字改寫為「C_bar」以 tspan 上劃線 `text-decoration:overline` 呈現）。
- 動態插入的 TeX（`<details>` 答案）插入後呼叫 `renderMathInElement`。
- `.katex-display{overflow-x:auto}`，且每個可橫捲的公式容器加 `tabindex="0" role="region" aria-label="Eq. n（可左右捲動）"`，使鍵盤可捲動；Eq. 12 與 Eq. 40–41 在 < 600px 拆兩行。
- 第 1 節以外不得新增論文數字；頁面所有數字與第 1 節一致。

### 6.3 引用與標籤

- 內文每個論文數字後註明出處（Eq./Table/Fig./§/App.）。
- 〔本頁推導〕在頁面用 pill「本頁推導」＋一行依據。本頁共 8 項：$w=(1-k)/k$、兩路線混合、25× 上限、純量線性玩具的閉式、凍結誤差邊界情形、逆變異數讀法、無資訊極限均勻 $t$ 時 $\sqrt\lambda$、$W=I$ 時 $a_t=b_t=1$。
- 區分三種陳述：論文結果、作者的解讀（「作者認為」）、本頁推導。「充分條件」「一致性檢查」「定性」等限定詞照論文原意保留。

### 6.4 文案

繁體中文、台灣用語、字面學術語氣；不用比喻、反問、宣傳形容詞。專有名詞第一次出現附英文（例：「各向異性（anisotropic）」「凍結誤差模型（frozen-error model）」）。用詞：「雜訊權重」（不用「噪聲權重」，全頁統一）、「還原」、「常見錯誤」、「w 軸」。

### 6.5 響應式與無障礙

同 No. 238（390–1440px、16px 側邊留白、無水平捲動、四欄以上表格放 `.tscroll`、< 480px 時 Table 1 改為每列一張卡片、focus 2px `--accent`、`prefers-reduced-motion` 取消 transition）。本頁沒有自動播放動畫。

### 6.6 研究脈絡與延伸閱讀

- 時間線：把 `tut/239/notes/timeline.json` 原樣貼進 `<script type="application/json" id="timeline-data">`；目前 13 筆。分類常數 `TL_GROUP`（以 arXiv ID 為 key，未列者歸「其他」）：

  | 類別 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | 擴散基礎 | 2006.11239、2011.13456 | `prediction-targets-overview.html` |
  | 速度回歸 | 2210.02747、2209.03003 | `flow-matching-velocity-target.html` |
  | 目標選擇 | 2202.00512、2206.00364 | `aniso-interpolated-targets-edm-kdiff-asymflow.html` |
  | 像素骨幹 | 2212.09748、2511.13720 | `aniso-pixel-patchified-setting.html` |
  | 內插前驅 | 2601.21419、2605.12964 | `aniso-interpolated-targets-edm-kdiff-asymflow.html` |
  | 條件數 | 2404.00456 | `aniso-quantization-conditioning.html` |
  | 條件數 | 2603.02337 | `aniso-spectral-weight-analysis.html` |
  | 本文 | 2611.17767 | `aniso-target-normal-form-and-recovery.html` |

  2601.21419（k-Diff）卡片另加站內教學連結 `prediction-targets-k-diff.html`。
- S12 延伸閱讀（總覽的閱讀順序）：`aniso-lineage-overview.html`（總覽）、`prediction-targets-overview.html`、`flow-matching-velocity-target.html`、`prediction-targets-k-diff.html`（教學）、`aniso-pixel-patchified-setting.html`、`aniso-interpolated-targets-edm-kdiff-asymflow.html`、`aniso-target-normal-form-and-recovery.html`、`aniso-spectral-weight-analysis.html`、`aniso-quantization-conditioning.html`、`aniso-latent-models-check.html`；另列站內 `pixel-space-diffusion.html`、`x-prediction-manifold-view.html`。每項附 frontmatter `title`。

### 6.7 建置後自我檢查

截圖 390／768／1024／1440，加深色、鍵盤 focus、reduced-motion；依字體、留白、視覺層級、色彩、動態、微互動、響應式、原創性逐項評分並修正。跑 `?test` 全數通過；逐條核對 12 個學習目標的答案段落確實在指定章節內文中。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 玩具與論文結果的關係 | 最佳線性預測器下還原後的損失與 $W$ 無關（Eq. 44），本頁的玩具都無法重現 $w\propto\lambda$ 的出現；要重現需訓練容量受限的非線性網路。C3、C4 的輸入（A、C、ρ、$\bar C/\bar A$）是使用者指定的假設值。 | 每個圖版第一行寫明假設與限制；S7 不放元件；沒有推定值。 |
| R2 | 「理論預測 w ∝ λ」的誤讀 | Eq. 24 的線性律來自兩個額外假設，等於把結論放入假設；證據是 A.3 的事後量測（一致性檢查，非直接量測）。 | S7 內文明寫；LO9 同時問假設與證據。 |
| R3 | 時間方向與 No. 238 相反 | 讀者從 No. 238 過來容易混淆。 | S1 callout 並排兩式；常見錯誤第 1 條；LO1。 |
| R4 | Table 1 差距的可靠度 | 0.02–0.05 的差別無多 seed、無誤差棒，guidance 逐模型 grid search；PCA init 在 160 ep 的效果不一致。 | S9 callout 照列，不寫「全面優於」。 |
| R5 | FID 協定混用 | 1.90（ADM）與 1.86*（JiT）不是同一數；Table 2 的 1.56* 用 JiT 協定；「best in-class」限 plain DiT-like。 | S9 表格保留 * 與 † 並在表註說明。 |
| R6 | 量化與 Fig. 1 | 低精度穩定性只有定性圖；Fig. 1a 來自未評估的 1MP + SFT 階段。 | S9、LO12 明寫「定性」；不轉載圖片。 |
| R7 | 圖讀數 | Fig. 3a 權重範圍、Fig. 7 量級只能目視；論文沒有給 λ 譜的數值。 | Hero 不畫權重範圍，只引 0.15λ；S2 靜態圖標「示意，λ 為合成值」。 |
| R8 | $w_i\ge0$ 與 $\exp(\eta)$ | 正文寫非負，實作嚴格正；JiT 的「Frozen (zeros)」不在可學範圍內。 | 常見錯誤第 3 條。 |
| R9 | C4 的 logit-normal 參數 | 論文未給 A.3 訓練取樣的參數，頁面的 μ 選項是示範用。 | 頁面註明「示範用選項，非論文設定」。 |
| R10 | C4、C5 運算量 | C4 每組 4001×1024 ≈ 4M 次；C5 在 n = 2048 時 1.6M 次 gauss；低階手機可能數百 ms。 | 互動時 `chunkRun()` 分塊並顯示「計算中…」、保留上一組結果、快取；`?test` 在主執行緒同步執行（headless 的 virtual-time-budget 不等 Worker）。 |
| R11 | 筆記可能更新 | 建置前 notes 若增減，`TL_GROUP` 與延伸閱讀需同步。 | 建置時重讀 `notes/`；未分類 ID 歸「其他」並在建置回報列出。 |
| R12 | 相對連結 | 建置位置與上線位置不同，預覽時站內連結失效。 | 接受；預覽只檢查外部連結與頁內錨點。 |
| R13 | 驗收表的捨入 | Python `round`／格式化與 JS `toFixed` 在 .5 邊界可能差 1 個末位（5.3 的 0.15625）。 | 驗收容許末位 ±1。 |

---

## 8. 修訂紀錄（round 1）

| 項目 | 變更 |
| --- | --- |
| F1 | 步驟 11、LO11：AsymFlow 1.76 為同骨幹重跑對照，1.57 為 † 已發表數字，設定不同。 |
| F2／B1 | C1 截斷改為作用在分母（$den'=\max(den,0.04)$ 進入 û），誤差拆成偏差＋δ/den′；模板句與表格用同一分母；預設不勾；`check.py` 新增截斷欄，5.1 更新。 |
| F3 | LO9：改為「以輔助線性讀出估計、一致性檢查、非直接量測、60 取 18」。 |
| F4／B4 | C3 第一行加「真實網路 A、C 隨 t 變化，抵銷見 S6」；觀察句改寫；S6 第一段與步驟 8 加 $p(t)$／時間平均（Eq. 16）；LO8 加問「訓練最小化哪個損失」；「零技能」全文改為「無資訊極限」（註明論文用語 zero-skill）。 |
| F5 | 步驟 6、C2、R1 收窄為「最佳線性預測器下損失與 W 無關（Eq. 44）」。 |
| F6 | Hero 不畫權重範圍區段，只引 $w\approx0.15\lambda$。 |
| F7 | 常見錯誤 3：可學的 w 取不到 0，凍結 0 是 JiT 基線設定。 |
| F8 | C5 第一行加實際 n 的量級與「輸出為零常見但不保證」；LO10 同步。 |
| F9 | C4 註明未含 flow shift，並列 Table 6 的設定。 |
| F10 | 步驟 5、LO4：「保證所有 t 非奇異的條件是特徵值非負」，加 $W=-I$ 小例。 |
| F11 | 1.3(c) 照錄論文原文並加〔本頁註〕說明 best in-class 限定。 |
| F12 | 步驟 9：0.2λ 為 Fig. 4b 參考線，非擬合值。 |
| B2 | S8 本文只留 r、G、升冪排序、EMA、第一步 PCA；Stiefel、Eq. 39、命題 2、A.5 移入 `<details>` 進階。 |
| B3 | S2 先定義 λ_i、PCA 基底、k-Diff、AsymFlow。 |
| B5 | 視覺讀數即時；另設 `.sr-only` live 區放 debounce 摘要。 |
| B6 | C4 搜尋範圍擴為 $[10^{-4},10^4]$，邊界顯示「≥ 10⁴」；分塊執行＋「計算中…」；`?test` 同步。 |
| B7 | S5 先教 Γ = 0；C3 的 ρ 與邊界情形放在「進階」開關後。 |
| B8 | S3 的 PSD 論證移到 C1 之後的 03.2 小節。 |
| B9 | SVG 只用預組字元與 tspan 下標，列出待驗字元。 |
| B10 | C2 < 600px 上下堆疊；C4 改用側邊圖例；公式捲動區加 tabindex/role/aria-label。 |
| B11 | `fmt` 門檻改為 1e4，驗收表改為 1000.0000。 |
| B12 | C1 曲線加標記形狀，目前 w 等於 0 或 1 時改標籤並加粗。 |
| B13 | C5 統計分塊執行，`?test` 同步。 |
| B14 | S6 標題改為「在可分離誤差假設下，最佳權重與 t 無關」；「時間桶」改「時間區間」。 |

未採納：
- B15（為 LO4 加元件）：只採納其替代方案「內文標明此項只由段落教」，並加 $W=-I$ 在 $t=1/2$ 的算式小例。理由：畫 $\alpha w+\sigma$ 對 $t$ 的 $w\lt0$ 曲線需要開放負權重，與本頁所有滑桿 $w\ge0$ 的設定衝突，一行算式已足以呈現。LO9 依 S7 不放元件的既定理由（玩具無法呈現 w ∝ λ 的出現）維持只由段落與表格教。
- B6 的 Web Worker：改用主執行緒分塊（單檔、無需 Blob Worker），效果相同且 `?test` 可同步執行。

## 9. 修訂紀錄（readiness）

- 阻擋 1：C1 時間格點改為 $j=4..400$（$t\ge0.01$），曲線裁到繪圖區。
- 阻擋 2：C4 縱軸改為對數 $[1,100]$，超出處截斷並畫三角形。
- 阻擋 3：S7 加 Eq. 35、37 的有效誤差定義句。
- 非阻擋 1–5：5.3 改 0.1563；S10 回指 S9 量化證據；S5 定義 ρ；hero 註記改用文字定義 λ；S2 說明非負與 exp 嚴格正的差別。

## 10. 建置紀錄（build）

- 來源：`build/src/`（head.html、body1–3.html、app.js），`build/assemble.py` 組成 `build/index.html` 並內嵌 timeline.json；輸出不含 file:// 路徑。
- 與設計的差異：讀數表改為兩欄式格線（`div.rot`，標籤在前、值在後），避免 15 列長表；符號表與 S8「進階」用 `details.more`（不被驗收腳本剝除，LO 答案不放在其中）；hero w 軸在 &lt; 560px 隱藏 10⁻³、10² 刻度以免重疊；`el()` 的 stroke-width 以 inline style 設定（CSS class 會蓋過屬性）。
- 驗證：`?test` 24/24（1440 與 390px iframe 皆是，selfTest 在 KaTeX 渲染後執行）；impeccable 0 anti-pattern（僅 numbered-section-labels advisory）；LO1–LO12 答案皆在 accept/page.txt；390px 無水平捲動；無 console 錯誤。截圖在 `build/caps/`。
