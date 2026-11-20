## LO1

- 答：$z_t=(1-t)x+t\epsilon$（Eq. 1，$\alpha_t=1-t$、$\sigma_t=t$）。$t=0$ 是資料 $x$，$t=1$ 是雜訊 $\epsilon$。速度是 $u=\partial_t z_t=\epsilon-x$。與 No. 238 不同：No. 238 寫 $z=tx+(1-t)n$，$t=1$ 才是資料，時間方向相反。
- 引用：「t=0 為資料、t=1 為雜訊」
- 信心：high

## LO2

- 答：整體尺度可被網路吸收，所以固定資料係數 $\varphi=-1$，每個方向只剩 $w=-\psi/\varphi$（App. A.1）。取值對應：$w=0$ 是 x 預測；$w=1$ 是速度 $\epsilon_i-x_i$；$w\to\infty$ 是 ε 預測；所有方向共用一個 $w$ 是 k-Diff；PCA 基底下 $w\in\{0,1\}$ 是 AsymFlow。AnisoFlow 讓每個 $w_i$ 與基底 $B$ 都學出來。
- 引用：「只剩一個參數 w=−ψ/φ」
- 信心：high

## LO3

- 答：$u_i=\dfrac{(1-w_i)z_{t,i}+\hat u_{W,i}}{\alpha_tw_i+\sigma_t}$（Eq. 5）。關鍵恆等式是 $(1-w)z_t+u_W=(\alpha_tw+\sigma_t)(\epsilon-x)$（Eq. 9），它用到 $\alpha+\sigma=1$。矩陣形式是 $\hat u=(\alpha_tW+\sigma_tI)^{-1}((I-W)z_t+\hat u_W)$（Eq. 10），只要可逆即可，不需要對稱。
- 引用：「(1−w)zt+uW=(αtw+σt)(ϵ−x)」
- 信心：high

## LO4

- 答：分母 $\alpha_tw+\sigma_t$ 為零時還原失效。在時刻 $t$ 奇異，若且唯若 $-t/(1-t)$ 是 $W$ 的特徵值；$t\in(0,1)$ 時它掃過整條負實軸。例如 $W=-I$ 時分母是 $2t-1$，在 $t=1/2$ 為零。一般非對稱矩陣可能有複數特徵值，條件數也可能很差。限制為 $W\succeq0$ 是為了讓特徵值全為非負實數，從而保證所有 $t$ 都非奇異。論文說一般族「代數上可行但不實用」。$w=0$ 只在 $t=0$ 奇異，而取樣器不計算 $t=0$。
- 引用：「保證所有 t 都非奇異的條件是特徵值非負」
- 信心：high

## LO5

- 答：$\hat u_i-u_i=(\hat u_{W,i}-u_{W,i})/(\alpha_tw_i+\sigma_t)$，目標誤差被放大 $1/(\alpha_tw_i+\sigma_t)$ 倍。$w_i=0$ 時放大 $1/t$，低雜訊時發散；$w_i=1$ 時放大倍率恆為 1。推論時分母下限設為 $\sigma_{\min}=0.04$，所以最多放大 25 倍（本頁推導：1/0.04）。但截斷改的是分母本身，即使預測完全準確，還原速度也會有偏差。
- 引用：「推論時論文把分母下限設為」
- 信心：high

## LO6

- 答：還原可逆。把最佳目標 $u^*_{W,i}=(\alpha_tw_i+\sigma_t)u^*_i-(1-w_i)z_{t,i}$（Eq. 11）代入還原式，恰得 $u^*_i$。損失拆成超額預測誤差（除以 $(\alpha w_i+\sigma_t)^2$，與 $w$ 有關）加上不可約誤差（與 $w$ 無關），交叉項為零。所以最小可達損失與 $w$ 無關，$w$ 的梯度只來自網路的不完美。若網路是最佳時間條件線性預測器，還原後的速度預測 $(\sigma_tI-\alpha_t\Sigma)\Sigma_{z,t}^{-1}$ 與 $W$ 無關（Eq. 44），線性模型學不出 $w$，也不識別基底。
- 引用：「線性模型不識別權重，也不識別基底」
- 信心：high

## LO7

- 答：凍結誤差模型下，Γ=0 時 $w_i^*(t)=\alpha_tC_i/(\sigma_tA_i)$（Eq. 7 = Eq. 19）。$A$ 是雜訊估計誤差的二階矩，$C$ 是資料估計誤差的二階矩。完整式為 $w^*=(\Gamma\sigma_t+\alpha_tC)/(A\sigma_t+\alpha_t\Gamma)$（Eq. 18）。$C$ 變大時，$w^*$ 變大：資料較難估的方向偏好較大的雜訊權重。
- 引用：「資料較難估（C 大）的方向偏好較大的雜訊權重」
- 信心：high

## LO8

- 答：訓練最小化時間平均損失 $\bar E_i(w_i)=\mathbb E_{t\sim p}[E_i(w_i;t)]$（Eq. 16）。共用權重成立的充分條件是可分離誤差假設（Eq. 20）：$A_i(t)=\alpha_tq_i(t)\bar A_i$、$C_i(t)=\sigma_tq_i(t)\bar C_i$、$\Gamma_i=0$。此時導數只取決於 $w-\bar C_i/\bar A_i$，共用值為 $w_i^*=\bar C_i/\bar A_i$（Eq. 21–23），與 $p(t)$ 無關。無資訊極限下，逐時刻最佳是 $w_i^*(t)=(\alpha_t/\sigma_t)\lambda_i$（Eq. 26），隨 $t$ 變化，所以沒有單一常數同時對所有時刻最佳，最佳常數會依 $p(t)$ 而定。
- 引用：「每個 qi(t)>0 的時刻都在」
- 信心：medium（「不行」是從「最佳常數依 p(t) 而定」推得，頁面未直接寫出「不行」二字）

## LO9

- 答：兩個額外假設（Eq. 24）：(1) 雜訊各向同性，$\bar A_i\approx\bar A$；(2) 資料估計誤差與資料能量成正比，$\bar C_i\propto\lambda_i$。理論不給常數 $c$，它取決於網路誤差結構、取樣器與最佳化。實測證據：JiT-H/16 學到 $w_i\approx0.15\lambda_i$（Fig. 3a）。App. A.3 的一致性檢查（非直接量測，18 個方向）得到 $\bar C^{\rm eff}\propto\lambda^{1.00}$、$\bar A^{\rm eff}\propto\lambda^{-0.09}$，比值指數 1.09，學到權重指數 1.07。
- 引用：「理論給出常數 c 嗎」（答案見上）；實測引用：「JiT-H/16 學到 wi≈0.15λi」
- 信心：high

## LO10

- 答：spectral update：$G=\mathbb E[a_trr^\top-b_t(r\epsilon^\top+\epsilon r^\top)]$（Eq. 40–41），$r=\mathrm{sg}(\hat u_W)+x$，$B$ 取 $G$ 的特徵向量，依升冪排列，$w_i$ 綁在 rank $i$ 上。網路輸出為零時 $r=x$，$\mathbb E[x\epsilon^\top]=0$，所以 $G\propto\mathbb E[xx^\top]$，第一次更新得到的是單一 minibatch 估計的帶雜訊 PCA 基底。
- 引用：「第一次更新得到的是單一 minibatch」
- 信心：high

## LO11

- 答：
  - ImageNet 256 主結果：JiT-H/16 由 FID 1.90 降到 1.55（Spectral update、無 PCA 初始化、600 epochs）。
  - 凍結 PCA 基底、只學權重：1.63。AsymFlow 同骨幹重跑：1.76。
  - 時間區間消融（JiT-B/16，160 epochs）：共用權重 4.92，隨時間權重 4.96；四個區間的曲線幾乎重合。
  - Latent 模型：SiT-XL/2-REPA 1.44 對 1.44，RAEv2 1.05 對 1.05，與預設持平。
  - T2I：除 GenEval（0.820→0.814）外各指標改善，例如 HPSv3 7.71→8.03。
- 引用：「共用權重 4.92 對隨時間權重 4.96」
- 信心：high

## LO12

- 答：權重參數化為 $w_i=\exp(\eta_i)$，與模型同學習率、從訓練開始就學、跨時間共用；預設 $B=I$、$w_i=1$，可選 PCA 初始化 $w_i=0.15\lambda_i$。損失算在還原後的速度上：$\|\hat u-u\|^2$。量化穩定性只有定性證據：W8A8（INT8、FP8、blockwise FP8）下，x 預測基線出現 patch 接縫或強雜訊，AnisoFlow 保持穩定（Fig. 1b、6b），沒有量化指標。
- 引用：「wi=exp⁡(ηi) 參數化」
- 信心：high

## 讀者可能卡住的地方

1. 時間方向：$t=0$ 是資料、$t=1$ 是雜訊，與 No. 238 相反；ε 與 n 是同一個量，要先確認哪一端是資料。
2. Eq. 9 與 Eq. 10 的還原推導需要先接受 $\alpha+\sigma=1$；$W\succeq0$ 的理由（$-t/(1-t)$ 掃過負實軸）是頁面的推導，不容易直接看出。
3. Spectral update：頁面寫 $\beta=10^{-5}$ 的 EMA，又說第一次更新就得到 PCA 基底，兩者的關係頁面沒有解釋；rank 與權重綁定是「啟發式配對」。
4. $w_i^*(t)=\alpha_tC_i/(\sigma_tA_i)$ 中 $A$、$C$ 是估計誤差的二階矩，頁面的 C3 玩具只是固定 $A$、$C$ 的示範，不是實測；$w^*=(\alpha/\sigma)\lambda$（隨時間變）與 $w^*=c\lambda$（常數、$c$ 未知）容易混淆。
5. 數字口徑：Table 1 的 1.90 用 ADM 協定，Table 2 的 1.86* 用 JiT 協定；AsymFlow 的 1.76 與 1.57 也不是同一設定；頁面沒有多 seed 或誤差棒。
