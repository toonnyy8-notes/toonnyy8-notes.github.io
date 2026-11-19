# 設計文件：〈Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning〉互動教學

- 論文：Zhendong Wang、Jonathan J Hunt、Mingyuan Zhou，arXiv 2208.06193（v3），ICLR 2023。資料庫編號 No. 443。方法名稱 **Diffusion-QL**（Diffusion Q-learning）。
- 原文：`tut/443/src/paper.txt`（PDF 同目錄；`tut/` = `scripts/tutorial_src/`）。「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」「Alg. 1」皆指論文原文編號。論文只有 Eq. 1–4 有編號；未編號的式子以「Sec. 3.1 的 μθ 式」「Sec. 3.1 的排程式」指稱。
- 產出：`tut/443/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/diffusion-ql.html`（slug `diffusion-ql`）。站內相對連結以上線位置為準。
- 驗算腳本（與頁面同一演算法、雙精度；`python3 -I <script>` 執行；輸出存於同目錄 `*_out.txt`）：`verify/core.py`（PRNG、排程、理想去噪器與其解析 Jacobian、Eq. 1 取樣鏈、玩具 Q；執行時做有限差分檢查）、`c1.py`、`c2.py`、`c3.py` + `c3_run.py`（C3 訓練，參數 `256 0.03 600 2,5,10 0,0.1,0.3,1,3`）、`c4.py` + `c4_data.json`（表格算術）。
- 研究筆記：`tut/443/notes/`（總覽 `diffusion-ql-lineage-hub` + 6 篇主題筆記 + `timeline.json` 15 筆），見 6.4。
- 標記慣例：**〔本頁推導〕** = 本設計自行推導、論文沒有寫出；**推定** = 為玩具可視化而選定、論文沒有寫明的值；**玩具** = 不是論文的實驗，不重現論文數字；**背景（論文外）** = 論文沿用但未寫出的既有知識（DDPM 記號 $\bar\alpha_i$、D4RL 正規化分數）；「作者認為」= 作者的解釋。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

離線強化學習（offline RL）只能用一個事先收集好的固定資料集學習策略。若策略改進時去評估資料中沒出現的動作，Q 值估計常常過高，學到的策略會偏向這些分布外（out-of-distribution, OOD）動作（Sec. 1）。一類解法是**策略正則化**：限制新策略不要離產生資料的行為策略太遠。作者認為這類方法表現略差，主因是策略類別表達力不足：常用的對角高斯策略只有一個峰，而由多個策略收集的資料常是多峰的；以最大概似（等於最小化 KL(資料‖策略)）擬合時，高斯的密度覆蓋所有資料峰，也把高密度放在峰與峰之間沒有資料的區域（Sec. 1）。本文把策略寫成**以狀態為條件的擴散模型**：從高斯噪聲 $a^N$ 出發，用噪聲預測網路 $\epsilon_\theta$ 反覆去噪 $N$ 次得到動作 $a^0$（Sec. 3.1、Eq. 1）。訓練損失是擴散模型的去噪損失 $\mathcal L_d$（Eq. 2，作用如行為複製）加上 $-\alpha\,\mathbb E[Q_\phi(s,a^0)]$（Eq. 3），Q 值的梯度經由 Eq. 1 的整條反向鏈傳回 $\theta$。在 2D bandit 上，擴散模型能複製四峰的行為分布，加入 Q 值項後收斂到報酬最高的峰（Sec. 4、Fig. 1）；在 D4RL 的四個領域（Gym、AntMaze、Adroit、Kitchen），Diffusion-QL 的領域平均分數都高於 Table 1 中所有基準，但 20 個任務中有 3 個由其他方法得到最高分（Table 1，見步驟 10）。代價是每個動作要算 $N$ 次網路；作者估計 $N=5$ 時推論**可能**比一步前饋策略慢 4–5 倍（App. G「could be」，未給量測條件）。

### 1.2 研究動機（Sec. 1）

1. 既有離線 RL 分四類：(1) 正則化策略與行為策略的距離；(2) 讓價值函數對 OOD 動作給低值；(3) 模型式方法（學環境動態後悲觀規劃）；(4) 以報酬引導的序列預測。本文屬第 (1) 類。
2. 作者認為第 (1) 類表現略差有兩個原因：**策略類別不夠有表達力**（對角高斯無法表示多峰、偏斜、維度相關）；**正則化方式不合適**（KL 需要明確密度值，MMD 需要每個狀態多個動作樣本，兩者都要先另外學一個行為複製模型，這個模型本身又受表達力限制，誤差會傳給正則化）。
3. 作者用 Sec. 4 的 2D bandit 說明這兩個問題在簡單任務上就會出現。

### 1.3 貢獻（本頁整理自 Sec. 1、Sec. 6）

1. 以條件擴散模型（MLP 架構的 DDPM）表示策略 $\pi_\theta(a\mid s)$；作者認為去噪損失「可視為」一種以樣本為基礎、不需要另外學行為模型的策略正則化（Sec. 1、3.1；加入 $\mathcal L_q$ 後它只近似於行為複製項）。
2. 把 Q 值最大化項線性加到擴散損失，Q 梯度在訓練時穿過整條反向鏈（Sec. 3.2、Eq. 3）。
3. 2D bandit 的視覺化與 $N$ 的消融（Sec. 4、Fig. 1–2、App. A）。
4. D4RL 四個領域的結果、$N$ 的影響、兩個元件的消融、離線模型選擇（Sec. 5、Table 1–2、App. D–H）。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S1、S3 末尾以可收合表格提供；第三欄為首次定義的章節）

| 符號 | 意義 | 首次定義 |
| --- | --- | --- |
| $s,a,r,s'$ | 狀態、動作、獎勵、下一個狀態 | S1 |
| $\gamma$ | 折扣因子，$\gamma\in[0,1)$ | S1 |
| $\pi_\theta(a\mid s)$ | 參數為 $\theta$ 的策略 | S1 |
| $Q^\pi(s,a)$、$Q_\phi$ | 動作價值（Q 值）；$Q_\phi$ 為參數 $\phi$ 的 Q 網路 | S1 |
| $\mathcal D$、$\pi_b$ | 離線資料集 $\{(s,a,r,s')\}$、收集它的行為策略 | S1 |
| $t$（下標） | 軌跡時間步，$a_t$ 是第 $t$ 步的動作 | S1 |
| $i$（上標）、$N$ | 擴散步，$i\in\{1,\dots,N\}$；$a^i$ 是第 $i$ 個擴散步的帶噪動作，$a^0$ 是輸出動作 | S3 |
| $\beta_i$、$\alpha_i=1-\beta_i$、$\bar\alpha_i=\prod_{j\le i}\alpha_j$、$c_i=\beta_i/\sqrt{1-\bar\alpha_i}$ | 噪聲排程；$\bar\alpha_i$ 為 DDPM 記號，論文直接使用未寫出定義（背景）；$c_i$ 為本頁為排版引入的縮寫 | S3 |
| $\epsilon$、$\epsilon_\theta(a^i,s,i)$ | 標準高斯噪聲、噪聲預測網路 | S3 |
| $\mathcal L_d$、$\mathcal L_q$ | 去噪（行為複製）損失、Q 值損失 | S4b、S5 |
| $\eta$、$\alpha$ | Q 值項的超參數；$\alpha=\eta/\mathbb E_{\mathcal D}|Q_\phi|$（注意：與 $\alpha_i$ 不同，頁面以下標區分並在 S5 寫明） | S5 |
| $m_k$、$s$、$v_i$、$q_k$、$\bar m$、$\epsilon^*$ | 玩具專用：第 $k$ 峰的中心、峰的標準差 0.05、帶噪動作的每維變異數 $\bar\alpha_is^2+1-\bar\alpha_i$、後驗責任、責任加權的中心、理想去噪器 | S4b |
| $R_k$、$z_k$、$w_k$、$g_k$、$\sigma_Q$、$\tilde Q$ | 玩具專用：第 $k$ 峰的獎勵平均、權重 logit、混合權重 $w=\mathrm{softmax}(z)$、平滑核權重、平滑寬度 0.2、平滑 Q（不與 $r$、$\rho$、$\alpha_i$ 共用字母） | S7 |
| $\phi_1,\phi_2,\phi'_1,\phi'_2,\theta'$、$\rho$ | 兩個 Q 網路、對應的目標網路、目標策略、目標網路更新係數 | S6 |

#### 步驟 1：離線 RL 與 OOD 動作（Sec. 1–2）

- MDP $\mathcal M=\{\mathcal S,\mathcal A,P,R,\gamma,d_0\}$；目標是最大化 $\mathbb E[\sum_{t=0}^\infty\gamma^tr(s_t,a_t)]$；$Q^\pi(s_t,a_t)$ 為從 $(s_t,a_t)$ 出發、之後依 $\pi$ 行動的期望折扣報酬（Sec. 2）。
- 離線設定：只有 $\mathcal D\triangleq\{(s,a,r,s')\}$，由 $\pi_b$ 收集，不能與環境互動。
- 問題：策略改進會評估資料外的動作，這些 Q 值估計不準；天真的方法偏好被高估的 OOD 動作（Sec. 1）。頁面 S1 連到 `offline-rl-distribution-shift.html`，不重講。
- 〔背景〕S1 正文兩句教 TD 目標（S6 的 Eq. 4 需要）：$Q^\pi$ 滿足 $Q^\pi(s,a)=\mathbb E[r+\gamma Q^\pi(s',a')]$，$a'\sim\pi(\cdot\mid s')$；實作上以網路 $Q_\phi$ 擬合「目標值」$y=r+\gamma Q(s',a')$，最小化 $(y-Q_\phi(s,a))^2$。$a'$ 不在資料中，這正是 OOD 高估進入學習的地方。

#### 步驟 2：既有策略正則化與其限制（Sec. 1、2、4）

- 名詞（S2 正文各一句）：CVAE（條件變分自編碼器）＝以潛變數 $z$ 與解碼器 $p(a\mid s,z)$ 表示 $\pi(a\mid s)$ 的生成模型，以 ELBO 訓練（Sec. 4）；MMD（最大均值差異）＝只用兩組樣本估計兩個分布差距的量，因此每個狀態要多個動作樣本（Sec. 1）。
- 代表方法（Sec. 2、4）：BCQ（以 CVAE 複製行為策略，再學一個有界的擾動）；BEAR（以 MMD 限制策略接近 CVAE 複製的行為策略，策略為 Tanh-Gaussian）；TD3+BC（在策略改進加 BC 項）；BRAC（比較 KL、MMD、Wasserstein 對偶）；IQL（以 in-sample 學到的優勢加權行為複製）。
- 作者把 BCQ、BEAR 稱為**兩步正則化**：先學行為複製，再限制策略在複製結果附近；效果取決於複製品質（Sec. 4）。
- BC-MLE：$\pi_\theta=\mathcal N(\mu_\theta(s),\Sigma_\theta(s))$，$\Sigma_\theta$ 對角，最大化 $\mathbb E_{\mathcal D}[\log\pi_\theta(a\mid s)]$（Sec. 4），即最大概似估計（MLE）。〔背景〕這等於最小化 $\mathrm{KL}(\text{資料}\,\|\,\pi_\theta)$（KL 的第一個參數是資料分布，Sec. 1 的寫法「from the data distribution to the policy」）：資料有密度而策略密度很小的地方懲罰極大，所以單峰高斯必須讓密度覆蓋所有資料峰，包括峰之間沒有資料的區域（mode-covering）。
- 〔本頁推導〕Sec. 4 的四峰資料（中心 $(0,\pm0.8)$、$(\pm0.8,0)$，標準差 0.05）用對角高斯做閉式 MLE（論文的 BC-MLE 是訓練出的網路，這裡是本頁的閉式解）：高斯的 MLE 就是樣本平均與樣本變異數。由對稱性 $\mu=(0,0)$；第一維 $a_1$：一半資料來自上、下峰（中心 $a_1=0$），一半來自左、右峰（中心 $a_1=\pm0.8$），所以 $\mathrm{Var}(a_1)=\mathbb E[a_1^2]-0^2=\tfrac12(0^2+0.05^2)+\tfrac12(0.8^2+0.05^2)=\tfrac12\cdot0.64+0.0025=0.3225$（標準差 0.568），第二維相同。頁面以這兩行寫出。擬合高斯在原點的密度 $1/(2\pi\cdot0.3225)=0.4935$，是**同一個高斯**在任一資料中心的密度 $0.4935\,e^{-0.64/0.645}=0.1830$ 的 **2.70 倍**；而真實資料在原點的密度幾乎為 0。在 $|a|<0.3$ 的圓內，高斯放了 13.0% 的機率（$1-e^{-0.09/0.645}$），真實資料是 0。這對應論文 Fig. 1「BC-MLE 高密度區正是真實分布的低密度區」的敘述。

#### 步驟 3：擴散模型背景（Sec. 2）

- 前向鏈：$q(x_{1:T}\mid x_0)=\prod_tq(x_t\mid x_{t-1})$，$q(x_t\mid x_{t-1})=\mathcal N(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_tI)$。
- 反向鏈：$p_\theta(x_{0:T})=\mathcal N(x_T;0,I)\prod_tp_\theta(x_{t-1}\mid x_t)$，以最大化 ELBO 訓練；取樣從 $x_T$ 走到 $x_0$。條件模型只要改成 $p_\theta(x_{t-1}\mid x_t,c)$。
- 〔背景〕$x_i=\sqrt{\bar\alpha_i}\,x_0+\sqrt{1-\bar\alpha_i}\,\epsilon$（多步前向的閉式），Eq. 2 用的就是這個式子。頁面 S3 用一句話寫出並連到 `ddpm-training-and-sampling-recap.html`。
- 注意 Sec. 2 的擴散步用下標 $t$、$T$；Sec. 3 起改為上標 $i$、$N$（下標 $t$ 留給軌跡時間）。頁面一律用 Sec. 3 的寫法。

#### 步驟 4：擴散策略（Sec. 3.1、Eq. 1）

$$\pi_\theta(a\mid s)=p_\theta(a^{0:N}\mid s)=\mathcal N(a^N;0,I)\prod_{i=1}^Np_\theta(a^{i-1}\mid a^i,s)$$

- 輸出動作是鏈的終點 $a^0$。（論文把聯合分布寫成 $\pi_\theta(a\mid s)$；嚴格說策略是 $a^0$ 的邊際分布，頁面註明。）
- 沿用 DDPM：$p_\theta(a^{i-1}\mid a^i,s)=\mathcal N(\mu_\theta,\beta_iI)$，$\mu_\theta(a^i,s,i)=\frac1{\sqrt{\alpha_i}}\big(a^i-\frac{\beta_i}{\sqrt{1-\bar\alpha_i}}\epsilon_\theta(a^i,s,i)\big)$。
- 取樣（Eq. 1）：$a^N\sim\mathcal N(0,I)$，對 $i=N,\dots,1$：

$$\begin{aligned}c_i&=\frac{\beta_i}{\sqrt{1-\bar\alpha_i}},\\ a^{i-1}&=\frac{a^i-c_i\,\epsilon_\theta(a^i,s,i)}{\sqrt{\alpha_i}}+\sqrt{\beta_i}\,\epsilon,\quad\epsilon\sim\mathcal N(0,I)\end{aligned}\tag{Eq. 1}$$

  （與論文的 $\frac{a^i}{\sqrt{\alpha_i}}-\frac{\beta_i}{\sqrt{\alpha_i(1-\bar\alpha_i)}}\epsilon_\theta+\sqrt{\beta_i}\epsilon$ 代數相同；頁面以兩行版呈現，並在旁註原式。）

  $i=1$ 時 $\epsilon=0$（沿用 DDPM，改善樣本品質）。
- 作者列的三個性質（Sec. 1）：表達力強、能表示多峰；擴散損失是強的分布匹配，可視為不需額外行為複製的樣本式正則化；逐步去噪讓 Q 值引導能作用在每一個反向步。

#### 步驟 5：去噪損失就是行為複製（Sec. 3.1、Eq. 2）

$$\begin{aligned}\mathcal L_d(\theta)=\;&\mathbb E\Big[\big\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_i}\,a+\sqrt{1-\bar\alpha_i}\,\epsilon,\ s,\ i)\big\|^2\Big],\\ &i\sim\mathcal U,\ \epsilon\sim\mathcal N(0,I),\ (s,a)\sim\mathcal D\end{aligned}\tag{Eq. 2}$$

- $\mathcal U$ 是 $\{1,\dots,N\}$ 上的均勻分布。$\mathcal L_d$ 讓模型學 $\pi_b(a\mid s)$。
- 只需要從 $\mathcal D$ 與目前策略取樣，**不需要知道 $\pi_b$ 的密度**（資料來自人類示範時密度可能無法取得）；作者稱之為比兩步策略更乾淨的做法（Sec. 3.1）。
- 每個資料點只抽一個 $i$ 就能訓練 $\mathcal L_d$；但取樣（Eq. 1）要算 $N$ 次 $\epsilon_\theta$，所以 $N$ 要小。

#### 步驟 6：小 N 用的噪聲排程（Sec. 3.1）

$$\beta_i=1-\alpha_i=1-\exp\!\Big(-\beta_{\min}\tfrac1N-0.5(\beta_{\max}-\beta_{\min})\tfrac{2i-1}{N^2}\Big),\quad\beta_{\min}=0.1,\ \beta_{\max}=10.0$$

- 出自 Xiao et al. (2021)，是 Song et al. (2021) 變異數保持（VP）SDE 的離散化。
- 〔本頁推導〕因為 $\sum_{j=1}^i(2j-1)=i^2$，$\bar\alpha_i=\exp\!\big(-0.1\,\tfrac iN-4.95\,\tfrac{i^2}{N^2}\big)$ 只依賴 $i/N$。所以：(a) 不論 $N$，最後一步 $\bar\alpha_N=e^{-5.05}=0.006409$，帶噪動作中只剩 $\sqrt{\bar\alpha_N}=0.0801$ 倍的原動作；(b) 各個 $N$ 的 $\bar\alpha_i$ 都落在同一條曲線 $\bar\alpha(\tau)=\exp(-0.1\tau-4.95\tau^2)$ 上（$\tau=i/N$）；$\beta_i=1-\bar\alpha_i/\bar\alpha_{i-1}$ 則不是同一函數上的點，$N$ 小時每步的 $\beta_i$ 大（$N=5$：$\beta_1=0.1959$、$\beta_5=0.8350$；$N=50$：$\beta_1=0.0040$、$\beta_{50}=0.1796$）。數值見第 5 節 C1。

#### 步驟 7：加入 Q 值項（Sec. 3.2、Eq. 3）

$$\pi=\arg\min_{\pi_\theta}\ \mathcal L(\theta)=\mathcal L_d(\theta)+\mathcal L_q(\theta)=\mathcal L_d(\theta)-\alpha\cdot\mathbb E_{s\sim\mathcal D,\ a^0\sim\pi_\theta}\big[Q_\phi(s,a^0)\big]\tag{Eq. 3}$$

- $\mathcal L_d$ 單獨不能超越行為策略；$\mathcal L_q$ 讓模型偏好取樣高價值動作。
- $a^0$ 由 Eq. 1 **重參數化**：固定 $a^N$ 與各步 $\epsilon$，$a^0$ 是 $\theta$ 的可微函數，$\nabla_aQ$ 穿過整條反向鏈傳回 $\theta$。這是在**訓練時**注入 Q 引導，不是取樣時的引導（Sec. 2 與 Diffuser 的對照）。
- 正規化（沿用 TD3+BC）：$\alpha=\dfrac{\eta}{\mathbb E_{(s,a)\sim\mathcal D}[|Q_\phi(s,a)|]}$，分母只做正規化、不微分。
- 〔本頁推導〕分母不微分，所以把 $Q_\phi$ 乘上任意**正**常數 $c$，$\alpha Q_\phi$ 不變（限乘法；把 $Q_\phi$ 加上常數會改變 $\mathbb E|Q|$，因而改變 $\alpha$）。這不表示 $\eta$ 可跨資料集共用：論文的 $\eta$ 從 0.005 到 4.5 依領域調整（Table 3）。
- 作者的整理：$\mathcal L_d$ 與 $\mathcal L_q$ 一起作用才得到最佳表現（Sec. 3.2、5.2）。

#### 步驟 8：Q 函數的學習與完整演算法（Sec. 3.2、Eq. 4、Alg. 1）

$$\mathbb E_{(s_t,a_t,s_{t+1})\sim\mathcal D,\ a^0_{t+1}\sim\pi_{\theta'}}\Big[\big\|r(s_t,a_t)+\gamma\min_{i=1,2}Q_{\phi'_i}(s_{t+1},a^0_{t+1})-Q_{\phi_i}(s_t,a_t)\big\|^2\Big]\tag{Eq. 4}$$

- 兩個 Q 網路取較小的目標值（double Q-learning 技巧，Hasselt 2010）；目標網路 $Q_{\phi'_1},Q_{\phi'_2}$ 與目標策略 $\pi_{\theta'}$；下一個動作 $a^0_{t+1}$ 也由 Eq. 1 從 $\pi_{\theta'}$ 取樣。Eq. 4 中 $\min_{i=1,2}$ 的 $i$ 是 Q 網路編號，不是擴散步（頁面註明）。
- Alg. 1 每次迭代：抽 mini-batch → 由 $\pi_{\theta'}$ 取 $a^0_{t+1}$ → 用 Eq. 4 更新 $Q_{\phi_1},Q_{\phi_2}$（可加 CQL 的 max Q backup；頁面放在註腳，見步驟 12）→ 由 $\pi_\theta$ 取 $a^0_t$ → 最小化 Eq. 3 → 軟更新 $\theta'=\rho\theta'+(1-\rho)\theta$、$\phi'_i=\rho\phi'_i+(1-\rho)\phi_i$。
- 論文**沒有寫出** $\rho$、$\gamma$ 的數值，也沒有寫 $\mathcal L_q$ 用哪一個 Q 網路（$Q_{\phi_1}$、兩者最小或平均）。頁面照錄並在常見錯誤中標「論文未寫明」。

#### 步驟 9：2D bandit（Sec. 4、Fig. 1–2、App. A、App. H）

- 資料：動作 $a\in[-1,1]^2$，$M=10000$ 筆，四個等權高斯，中心 $(0,0.8)$、$(0.8,0)$、$(0,-0.8)$、$(-0.8,0)$，標準差 $(0.05,0.05)$。
- 行為複製（Fig. 1 第一列；BC-Diffusion 用 $N=50$）：擴散模型抓到四個峰；BC-MLE 只有一個峰、標準差大；CVAE 抓到四峰但在峰之間放了高密度（換 seed 時有時抓不到全部四峰）；MMD 的 Tanh-Gaussian 把密度集中在邊界附近。
- 加入獎勵：每個資料點的獎勵從高斯抽樣，平均由所屬中心決定、標準差 0.5；圖例（Fig. 1 第二列第一格）：上 $\mathcal N(3.0,0.5)$、右 $\mathcal N(1.5,0.5)$、下 $\mathcal N(5.0,0.5)$、左 $\mathcal N(0.0,0.5)$（位置依 PDF 圖中顏色核對）。獎勵函數未知、需要學。所有方法訓練 1000 epochs。
- 結果（Sec. 4 原文）：TD3+BC 無法收斂到最佳峰（其行為複製把大部分密度放在沒有資料的區域）；BCQ 把動作放在 CVAE 找出的四個對角角落；BEAR-MMD 的動作近乎隨機分布；**Diffusion-QL 收斂到最下方（報酬最高）的峰**。作者歸因於：擴散策略能還原涵蓋所有峰的行為策略，Q 學習再在這個區域內找最佳動作。
- $N$ 的影響（Fig. 2）：BC-Diffusion 隨 $N$ 增加越來越準，$N=50$ 準確還原；加入 Q 學習後中等的小 $N$ 已經表現好；$N=2$ 時仍有少數動作落在沒有資料的區域，$N=50$ 時策略被限制在正確的資料區域。$N$ 是表達力與計算成本的權衡；D4RL 用 $N=5$。
- App. A：四個中心改到角落 $(\pm0.8,\pm0.8)$，結論相同（CVAE 只抓到兩個對角峰、MMD 集中在邊界、Diffusion-QL 收斂到最佳角落）。
- App. H（TD3+BC-GM；頁面放在 S10 消融）：把 TD3+BC 的確定性 actor 換成高斯混合（mixture density network），BC 項改為混合模型的最大概似。適當的 $K$ 能複製四峰，但 Q 學習時無法收斂到最佳位置，總是在次佳峰上留一些密度；D4RL 上 $K=3$ 表現差（Table 5：halfcheetah/hopper/walker2d-medium-expert 為 49.0／40.0／66.5，TD3+BC 為 90.4／98.0／110.1，Diffusion-QL 為 96.8／111.1／110.1）。作者列的可能原因：高維難擬合、需事先選 $K$、每個成分是對角高斯。

#### 步驟 10：D4RL 實驗（Sec. 5、Table 1、Fig. 3）

- 四個領域：Gym-MuJoCo（較容易，含不少接近最佳的軌跡）、AntMaze（稀疏獎勵；資料中沒有從起點到終點的完整好軌跡，策略必須把不同軌跡的片段接起來走到終點，論文稱 stitch）、Adroit（多為人類示範、資料範圍窄，需要強正則化）、Kitchen（依序完成 4 個子任務，需要長期價值）。
- 訓練：Gym 2000 epochs、其他 1000 epochs，每 epoch 1000 次梯度步，batch 256（Sec. 5、App. C）。
- $N$ 的消融（Fig. 3，$N\in[2,5,10,20]$，三個 medium-expert 任務）：$N$ 越大收斂越快、越穩定；選 $N=5$，訓練時間與 CQL 相近（Sec. 5）。
- 評估（Table 1 標題）：Gym 為 5 個獨立訓練的模型 × 10 條軌跡（50 次 rollout），其他領域 5 × 100（500 次）；分數為正規化分數的平均 ± 標準誤。App. C 卻寫「6 個模型」，兩處不一致（R3）。
- 結果（Table 1；c4.py 核算）：

  | 領域 | Diffusion-QL 平均 | 平均最高的單一基準 | 該任務最高的基準高於 Diffusion-QL 的任務 |
  | --- | --- | --- | --- |
  | Gym（9 任務） | 88.0 | 77.6（CQL） | hopper-medium（MoRel 95.4 vs 90.5±4.6）、walker2d-medium-expert（Onestep RL 113.0 vs 110.1±0.3） |
  | AntMaze（6） | 69.6 | 63.0（IQL） | antmaze-umaze-diverse（CQL 84.0 vs 66.2±8.6） |
  | Adroit（2） | 65.1 | 56.5（BCQ） | 無 |
  | Kitchen（3） | 69.0 | 53.3（IQL） | 無 |

  論文 Table 1 標題寫「在所有領域明顯勝過所有先前方法」，Sec. 1 寫「多數任務」，Sec. 6 寫「across all tasks」。頁面只陳述上表：**四個領域的平均都最高；逐任務比較時 20 個任務中 17 個最高**。
- Table 1 基準一行說明（S9 正文，論文 Sec. 5 的分類）：BC（行為複製）；AWAC、AWR（優勢加權）；Onestep RL（只做一步策略改進）；TD3+BC、BCQ、BEAR、BRAC（策略正則化）；IQL（in-sample 學 Q 再加權行為複製）；CQL、REM（Q 值約束：對 OOD 動作壓低 Q）；MoRel（模型式：學環境模型後悲觀規劃）；DT、Diffuser（序列模型）；SAC（線上演算法直接用於離線）。分數為 D4RL 正規化分數（背景：0 對應隨機策略、100 對應專家）。
- 作者的解釋（Sec. 5.1）：Gym 的 medium 資料由訓練到約專家 1/3 表現的 SAC 收集，分布分散難學，擴散模型能複製；AntMaze 需要強而穩定的 Q 學習，適當的 $\eta$ 下 Diffusion-QL 勝過先前方法；Adroit 用較小的 $\eta$（強正則化）；Kitchen 需要長期價值。

#### 步驟 11：消融（Sec. 5.2、Table 2）

| 方法（3 個 Gym medium-expert 任務平均） | 平均 |
| --- | --- |
| BC-CVAE | 79.9 |
| BC-Diffusion | 85.9 |
| BCQ-CVAE | 105.1 |
| BCQ-Diffusion | 106.0 |
| CVAE-QL | 76.6 |
| Diffusion-QL | 106.9 |

- 作者的結論：BC-Diffusion > BC-CVAE（擴散模型較會複製）；同在 BCQ 框架下 BCQ-Diffusion > BCQ-CVAE；BCQ 對偏移的硬性限制會限制策略改進，Diffusion-QL 更好；CVAE 與自由的 Q 學習結合時失敗（CVAE-QL，walker2d 50.5±38.2）。
- 逐任務並非全部由 Diffusion-QL 最高：walker2d-medium-expert 的 BCQ-Diffusion 114.0±0.5 高於 Diffusion-QL 111.2±0.9。頁面照錄。
- Table 2 的 Diffusion-QL 數字（97.2、112.3、111.2）與 Table 4 的**線上**模型選擇欄相同，而非 Table 1 的離線選擇值（96.8、111.1、110.1）；論文未說明 Table 2 用哪種選擇（R4）。

#### 步驟 12：實作細節與模型選擇（App. B–F）

- 網路（App. B）：$\epsilon_\theta$ 為 3 層 MLP、Mish 激活、256 隱藏單元；輸入為 $a^i$、狀態 $s$、$i$ 的正弦位置嵌入三者串接。兩個 Q 網路同樣設定。皆用 Adam。
- 獎勵（App. C）：Gym 與 Kitchen 用原始獎勵；Adroit 獎勵標準化；AntMaze 依 CQL 的建議修改。
- 超參數（App. E、Table 3）：只調三個——策略學習率 $\{10^{-3},3\times10^{-4},3\times10^{-5}\}$（Q 網路固定 $3\times10^{-4}$）、$\eta$、是否用 max Q backup（AntMaze 中除 antmaze-umaze 外都用；其他領域不用，Table 3）。$\eta$：Gym 全為 1.0；AntMaze 0.5–4.5（umaze 0.5、umaze-diverse 2.0、medium-play 2.0、medium-diverse 3.0、large-play 4.5、large-diverse 3.5）；Adroit pen-human 0.15、pen-cloned 0.1（學習率 $3\times10^{-5}$）；Kitchen 0.005。規則：Adroit、Kitchen 需要更多正則化（小 $\eta$），AntMaze 需要更多 Q 學習（大 $\eta$）。
- 離線模型選擇（Sec. 5、App. D）：作者觀察到 $\mathcal L_d$ 的變化比線上表現晚出現（論文稱 lagging indicator）；訓練中監看 $\mathcal L_d$、存多個 checkpoint，$\mathcal L_d$ 上升時停止，選 $\mathcal L_d$ **第二或第三低**的 checkpoint（不選最低，因為目標是策略學習而非行為複製）。只用第二低時：Gym 平均 87.6、AntMaze 69.1。正文的結果都用離線選擇。
- 線上選擇（App. F、Table 4）：允許少量線上評估挑最佳 checkpoint 時，平均為 Gym 89.3、AntMaze 75.4、Adroit 68.3、Kitchen 71.6（離線：88.0、69.6、65.1、69.0）。
- 常見錯誤（S11 逐條，各附出處或標籤）：
  1. 忘了 $i=1$ 時不加噪聲（Sec. 3.1）。
  2. 讓梯度流過 $\alpha$ 的分母 $\mathbb E|Q|$；論文寫明不微分（Sec. 3.2）。
  3. 取樣 $a^0$ 時切斷梯度（例如在無梯度模式下跑反向鏈）；$\mathcal L_q$ 需要梯度穿過整條鏈（Sec. 3.2）。
  4. 把擴散步 $i$ 與軌跡時間 $t$ 混用，或把 Eq. 4 的 $\min_{i=1,2}$ 當成擴散步（Sec. 3.1 記號說明）。
  5. 以為 $\eta$ 可跨領域共用：Table 3 的 $\eta$ 從 0.005 到 4.5（App. E）。
  6. 以最低 $\mathcal L_d$ 選 checkpoint：論文選第二或第三低（App. D）。
  7. 期待論文給出 $\rho$、$\gamma$、$\mathcal L_q$ 使用哪個 Q：論文未寫明（Alg. 1），重現時以所用程式碼為準。

#### 步驟 13：限制（App. G）

推論時 Eq. 1 要算 $N$ 次 $\epsilon_\theta$；作者估計 $N=5$ 的擴散策略「可能」比一步前饋策略慢 4–5 倍（could be，未給量測條件），需要快速反應的場景可能無法部署。作者引用一組加速擴散取樣的文獻（Lu et al. 2022、Wang et al. 2022、Xiao et al. 2021、Salimans & Ho 2022、Song et al. 2020、Zheng et al. 2022），並提出「訓練後蒸餾成前饋策略」的可能性。

---

## 2. 學習目標（驗收測試）

每個參考答案都必須寫在指定章節的**正文**（不能只出現在元件、表格、圖或答案框）。每章末以 `<details>`「檢查題與參考答案」照錄題目與參考答案。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明離線 RL 的困難 | 離線 RL 只能用什麼學習？為什麼天真的策略改進會失敗？ | 只用行為策略 $\pi_b$ 收集的固定資料集 $\mathcal D$，不能與環境互動；策略改進會評估資料外動作，其 Q 值常被高估，策略偏向這些 OOD 動作（Sec. 1–2） | S1 |
| LO2 | 列出既有策略正則化的兩個問題 | 作者認為策略正則化表現較差的兩個原因？ | (1) 策略類別表達力不足：對角高斯無法表示多峰、偏斜、維度相關；(2) 正則化方式不合適：KL 需要密度值、MMD 需要每個狀態多個樣本，且需先另學行為複製模型（兩步），複製誤差傳給正則化（Sec. 1、4） | S2 |
| LO3 | 用數字說明 mode-covering | 對 Sec. 4 的四峰資料做對角高斯 MLE，得到什麼？擬合高斯在原點的密度 ÷ 同一高斯在資料中心的密度？ | 平均 $(0,0)$，每維變異數 $\tfrac12\cdot0.64+0.0025=0.3225$（標準差 0.568）；比值 2.70（真實資料在原點的密度約為 0），$|a|<0.3$ 內放 13.0% 機率，真實資料為 0（〔本頁推導〕） | S2 |
| LO4 | 寫出擴散策略的取樣 | 擴散策略如何從狀態 $s$ 產生動作？上標 $i$ 與下標 $t$ 的差別？最後一步有何不同？ | $a^N\sim\mathcal N(0,I)$，依 Eq. 1 從 $i=N$ 到 $1$ 去噪，輸出 $a^0$；$i$ 是擴散步、$t$ 是軌跡時間步；$i=1$ 時不加噪聲（Sec. 3.1） | S3 |
| LO5 | 說明 $\mathcal L_d$ 為何是行為複製與正則化 | Eq. 2 在學什麼？為什麼不需要知道 $\pi_b$？ | 以 $\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_i}a+\sqrt{1-\bar\alpha_i}\epsilon,s,i)\|^2$ 學 $\pi_b(a\mid s)$；只需從 $\mathcal D$ 取樣，不需 $\pi_b$ 的密度，也不需另學行為模型（Sec. 3.1） | S4b |
| LO6 | 讀懂小 $N$ 的噪聲排程 | Sec. 3.1 的 $\beta_{\min},\beta_{\max}$？$\bar\alpha_N$ 與 $N$ 的關係？ | $\beta_{\min}=0.1$、$\beta_{\max}=10.0$（VP SDE，Xiao et al.）；$\bar\alpha_i=\exp(-0.1\,i/N-4.95\,i^2/N^2)$ 只依賴 $i/N$，所以任何 $N$ 下 $\bar\alpha_N=e^{-5.05}\approx0.0064$（〔本頁推導〕） | S4a |
| LO7 | 寫出 Diffusion-QL 的目標 | Eq. 3 是什麼？$\alpha$ 怎麼算？Q 的梯度如何到達 $\theta$？ | $\mathcal L=\mathcal L_d-\alpha\,\mathbb E[Q_\phi(s,a^0)]$；$\alpha=\eta/\mathbb E_{\mathcal D}|Q_\phi(s,a)|$，分母不微分；$a^0$ 由 Eq. 1 重參數化，梯度穿過整條反向鏈（訓練時注入，不是取樣時引導）（Sec. 3.2） | S5 |
| LO8 | 寫出 Q 函數的學習 | Eq. 4 的目標值？用到哪些網路？ | $r+\gamma\min_{i=1,2}Q_{\phi'_i}(s_{t+1},a^0_{t+1})$，$a^0_{t+1}$ 由目標策略 $\pi_{\theta'}$ 以 Eq. 1 取樣；兩個 Q 網路、兩個目標 Q 網路、目標策略，以 $\rho$ 軟更新（Sec. 3.2、Alg. 1） | S6 |
| LO9 | 描述 2D bandit 的結果 | 四個峰的獎勵平均？Diffusion-QL、BCQ、TD3+BC、BEAR-MMD 各收斂到哪裡？C3 玩具中 $\eta$ 增加時發生什麼？ | 上 3.0、右 1.5、下 5.0、左 0.0（標準差 0.5）；Diffusion-QL 收斂到下方峰；BCQ 在四個對角角落；TD3+BC 無法集中到最佳峰；BEAR-MMD 近乎隨機（Sec. 4）。玩具（$N=5$、600 步）：$\eta=0$ 時四個權重都約 0.25，$\eta=1$ 時下方權重約 0.90（玩具，不重現論文） | S7 |
| LO10 | 說明 $N$ 的權衡 | $N$ 增加有何好處與代價？論文在 D4RL 用多少？玩具中的 N 效應來自哪裡？ | 論文：$N$ 越大越能還原資料分布（$N=50$ 準確）、加入 Q 時越能把動作限制在資料區（$N=2$ 仍有少數動作落在無資料區）；代價是取樣要算 $N$ 次網路；D4RL 用 $N=5$，訓練時間與 CQL 相近，作者估計推論「可能」比一步策略慢 4–5 倍（Sec. 4–5、App. G）；論文只說 N 是表達力與成本的權衡，未分解原因；玩具用理想去噪器，其 N 效應只來自 Eq. 1 的步數與排程（〔本頁推導〕） | S8 |
| LO11 | 正確讀出 D4RL 結果 | 四個領域的平均分數？是否每個任務都最高？ | Gym 88.0（平均最高的單一基準 CQL 77.6）、AntMaze 69.6（IQL 63.0）、Adroit 65.1（BCQ 56.5）、Kitchen 69.0（IQL 53.3）；20 個任務中 3 個由其他方法最高：hopper-medium（MoRel 95.4）、walker2d-medium-expert（Onestep RL 113.0）、antmaze-umaze-diverse（CQL 84.0）（Table 1）；基準：MoRel 為模型式、Onestep RL 只做一步策略改進、CQL 對 OOD 動作壓低 Q（Sec. 5） | S9 |
| LO12 | 讀出消融與實作設定 | Table 2 的主要數字？$\eta$ 在各領域的範圍？論文如何離線選 checkpoint？ | BC-CVAE 79.9、BC-Diffusion 85.9、BCQ-CVAE 105.1、BCQ-Diffusion 106.0、CVAE-QL 76.6、Diffusion-QL 106.9；$\eta$：Gym 1.0、AntMaze 0.5–4.5、Adroit 0.1–0.15、Kitchen 0.005；選 $\mathcal L_d$ 第二或第三低的 checkpoint（Table 2、Table 3、App. D） | S10、S11 |

LO12 的分配：Table 2 在 S10 正文；$\eta$ 與模型選擇在 S11 正文。

---

## 3. 頁面大綱

單欄閱讀（680–720px），圖版加寬到最大 1080px。內部代號 S0–S13（S4 拆為 S4a、S4b），頁面上的章節編號依序為 01–14。章節編號以 JetBrains Mono 顯示（`01 · …`），標題用陳述句。每個符號在首次使用的章節定義（符號表第三欄）；每個元件放在它用到的概念之後。

| # | id | 標題 | 內容 | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈Diffusion-QL：以擴散模型表示離線強化學習的策略〉 | 副標：「這篇論文把策略寫成一個從噪聲逐步去噪產生動作的擴散模型，並在它的訓練損失中加入一項鼓勵高價值動作的項。」三張事實卡（見下）；不放公式。連結：arXiv、`../index.html#paper-443`、`diffusion-ql-lineage-hub.html`、程式碼 `https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL` | 1 螢幕 |
| S1 | `offline` | 01 · 離線 RL 只能從固定資料集學習 | MDP、策略、Q 值（字面＋一式）；Bellman 關係與 TD 目標 $y=r+\gamma Q(s',a')$ 兩句（S6 需要）；$\mathcal D$、$\pi_b$；OOD 高估（$a'$ 不在資料中）；四類既有方法一句各。答 LO1。延伸：`rl-mdp-q-learning-basics.html`、`offline-rl-distribution-shift.html` | 400–500 字 |
| S2 | `gauss` | 02 · 對角高斯策略無法表示多峰的行為分布 | 策略正則化；CVAE、MMD 各一句定義；BCQ／BEAR／TD3+BC／IQL 一行各；兩步正則化；MLE = 最小化 KL(資料‖策略) 與 mode-covering 的字面說明；〔本頁推導〕0.3225 的兩行混合變異數計算與 2.70（正文寫出 LO3，註明是閉式 MLE、不是論文 Fig. 1 的網路）；C2a（只有「真實資料／對角高斯」兩個分頁）。答 LO2、LO3。延伸：`gaussian-policy-multimodality-limits.html`、`batch-constrained-policy-constraint-methods.html` | 500–600 字 + C2a |
| S3 | `diffusion` | 03 · 擴散策略以 N 次去噪產生動作 | 前向加噪（一句＋閉式，背景）、反向鏈、$\pi_\theta$ 式、$\mu_\theta$、Eq. 1（aligned 多行）、$i=1$ 無噪聲、記號（$i$ vs $t$）、三個性質。末尾符號表。答 LO4。延伸：`ddpm-training-and-sampling-recap.html`、`diffusion-policy-class.html` | 450–550 字 |
| S4a | `schedule` | 04 · 小 N 用的噪聲排程 | 排程式；〔本頁推導〕$\bar\alpha_i$ 只依賴 $i/N$、$\bar\alpha_N=e^{-5.05}$（正文寫出 LO6）；C1。答 LO6 | 250–300 字 + C1 |
| S4b | `bc` | 05 · 去噪損失可視為行為複製 | Eq. 2（aligned，期望下標獨立一行）；不需 $\pi_b$ 密度；只抽一個 $i$；**理想去噪器**（正文先定義 $m_k$、$s$、$v_i$、$q_k$、$\bar m$，再寫 4.0 的 $\epsilon^*$ 式；3–4 句字面說明：「若資料分布已知，最好的噪聲預測是給定帶噪動作時噪聲的條件平均，對高斯混合有閉式解；本頁用它代替訓練好的網路」），之後才出現 C2b（加上「擴散（理想去噪器）」分頁與 $N$ 選擇）；一句「玩具的 N 效應只來自 Eq. 1 的步數與排程，不是網路容量」。答 LO5。延伸：`eps-prediction-ddpm-score.html` | 400–500 字 + C2b |
| S5 | `coupling` | 06 · 在去噪損失上加一項 Q 值 | Eq. 3；「$\alpha$ 與排程的 $\alpha_i$ 不同」；$\alpha$ 正規化與〔本頁推導〕尺度不變；重參數化與鏈上的梯度（靜態 SVG：$a^N\to a^{N-1}\to\dots\to a^0\to Q$ 的方塊鏈，前向實線箭頭、梯度虛線反向箭頭，標「梯度穿過每一步」）；訓練時注入 vs 取樣時引導。答 LO7。延伸：`diffusion-ql-loss-coupling.html`、`td3bc-algorithm-and-d4rl.html` | 400–500 字 + 靜態 SVG |
| S6 | `critic` | 07 · Q 函數以兩個 Q 網路與目標網路學習 | 先回顧 S1 的 TD 目標，再 Eq. 4（aligned：先定義目標值 $y$，再寫平方誤差）；double Q；目標網路與軟更新 ρ；Alg. 1 兩欄（≥ 768px 左程式碼、右說明；< 768px 每行下方接說明）；max Q backup 放註腳（AntMaze 除 umaze 外使用）；「論文未寫明 ρ、γ」；「Eq. 4 的 $i$ 是 Q 網路編號」。答 LO8。延伸：`ddpg-td3-actor-critic.html` | 450–550 字 + 演算法 |
| S7 | `bandit` | 08 · 2D bandit 中擴散策略收斂到報酬最高的峰 | Sec. 4 設定與四個獎勵；Fig. 1 兩列的文字描述（照 1.4 步驟 9）；作者的歸因；App. A 一句；C3 前 3 句玩具參數化（4 個峰的權重 $w=\mathrm{softmax}(z)$、平滑 Q、Adam 是一種依梯度更新參數的方法）；C3（玩具）；正文寫出 LO9 的論文結果與玩具數字；一段「玩具的限制」：Q 精確、只有 4 個權重、沒有 OOD 高估，所以大 $\eta$ 在玩具中沒有代價、也看不出去噪損失防止 OOD 的作用；論文依領域調 $\eta$（Adroit 0.1–0.15、Kitchen 0.005、AntMaze 0.5–4.5，Table 3）。答 LO9 | 450–550 字 + C3 |
| S8 | `steps` | 09 · 擴散步數 N 權衡表達力與計算 | Fig. 2 兩列結論（論文：$N=2$ 有「少數」動作在無資料區）、Fig. 3、$N=5$、訓練時間與 CQL 相近、作者估計推論「可能」慢 4–5 倍（App. G）；指回 C2b、C3 並寫出玩具對照（C2b：$N=2$ 落在峰附近 0.1%、$N=50$ 為 100%）；明寫「論文說 N 是表達力與成本的權衡，未分解原因；玩具的 N 效應只來自取樣步數與排程（理想去噪器沒有容量限制），比例也遠大於論文所說的少數」。答 LO10。延伸：`diffusion-policy-inference-cost.html` | 300–400 字 |
| S9 | `results` | 10 · D4RL 四個領域的平均分數都最高，但不是每個任務 | 四個領域的特性（「拼接」寫成字面定義）；Table 1 各基準一行說明（1.4 步驟 10）；訓練與評估設定（5 模型的標題 vs App. C 的 6 模型，照錄兩者）；D4RL 正規化分數一句（背景）；C4；正文寫出 LO11 的四個平均（稱「平均最高的單一基準」）與 3 個例外；作者的解釋（標「作者認為」）。答 LO11。延伸：`diffusion-policy-experiments-d4rl.html`、`offline-rl-benchmarks-and-baselines.html` | 500–600 字 + C4 |
| S10 | `ablation` | 11 · 消融比較擴散模型與 CVAE、Q 值項與 BCQ 式改進 | Table 2（靜態表 + 正文數字）、作者的結論、walker2d 例外、Table 2 與 Table 4 線上欄相同（R4）；App. H（mixture density network 一句定義、$K$、Table 5 數字）。答 LO12 前半 | 350–450 字 + 2 表 |
| S11 | `impl` | 12 · 實作設定、模型選擇與常見錯誤 | App. B 網路、App. C 獎勵、Table 3（靜態表；< 480px 改為每任務一張卡片：任務名、lr、η、max Q）、$\eta$ 規則（正文寫出 LO12 後半）、App. D 模型選擇（「$\mathcal L_d$ 的變化比線上表現晚出現」）與 87.6／69.1、Table 4 摘要、常見錯誤 7 條。答 LO12 後半 | 400–500 字 + 表 |
| S12 | `history` | 13 · 研究脈絡 | 時間線（6.4）＋一段：擴散模型（2015、2019、2020、Xiao 2021）→ 策略正則化（BCQ、BEAR、BRAC、TD3+BC、IQL）→ CQL → 同期的擴散規劃 Diffuser 與多峰行為複製 BeT → 本文 | 時間線 + 100–150 字 |
| S13 | `next` | 14 · 限制與延伸閱讀 | App. G 限制與加速方向；研究筆記清單（6.4）；arXiv、程式碼 | 150 字 + 連結 |

Hero 事實卡（只用一般詞彙；卡內第一次出現的術語當場以括號定義）：
1. 「離線強化學習只能從一個固定的資料集學習策略（在每個狀態選動作的規則）。資料若由多種策略收集，動作的分布常有多個峰；常用的高斯策略只有一個峰。」
2. 「本文把策略寫成擴散模型：從隨機噪聲出發，重複去噪 N 次得到一個動作。訓練時在去噪損失上再加一項，鼓勵模型產生 Q 值（對動作長期報酬的估計）高的動作。」
3. 「在 D4RL 標準測試的四類任務上，本方法的類別平均分數都高於平均最高的單一基準（例如 Gym 88.0 對 CQL 的 77.6）；逐任務比較時，20 個任務中有 3 個由其他方法最高。」

---

## 4. 互動元件規格

### 4.0 共用：bandit 玩具與理想去噪器（C2、C3 共用，`core.py` 逐式相同）

**為什麼用玩具**（頁面 S4b 寫出）：瀏覽器中無法訓練論文的 MLP。本頁用 Sec. 4 的四峰資料，但把 $\epsilon_\theta$ 換成**理想去噪器**：對一個高斯混合分布，最佳噪聲預測 $\mathbb E[\epsilon\mid a^i]$ 有閉式解。它相當於「訓練到完美的網路」，所以玩具只示範 Eq. 1 的取樣與 Eq. 3 的權衡，**不重現論文 Fig. 1–2 的圖形或數字**（論文的網路有有限容量與訓練誤差，例如論文的 BC-Diffusion(N=5) 呈菱形分布，玩具的不同）。

- 資料（Sec. 4）：中心 `MU = [[0,0.8],[0.8,0],[0,-0.8],[-0.8,0]]`（上、右、下、左），標準差 `SD = 0.05`，等權。獎勵平均 `RBAR = [3.0, 1.5, 5.0, 0.0]`（Fig. 1 圖例；文中記為 $R_k$）。
- 排程：1.4 步驟 6 的式子；`schedule(N)` 回傳 $\beta,\alpha,\bar\alpha$（$\bar\alpha$ 以累乘計算；selfTest 對照閉式 $\exp(-0.1i/N-4.95i^2/N^2)$）。
- 〔本頁推導〕理想去噪器：設策略的混合權重 $w=\mathrm{softmax}(z)$（$z\in\mathbb R^4$），各成分 $\mathcal N(m_k,s^2I)$，$m_k$=`MU`、$s$=`SD`。帶噪動作 $x=\sqrt{\bar\alpha}\,a+\sqrt{1-\bar\alpha}\,\epsilon$ 的分布為 $\sum_kw_k\mathcal N(\sqrt{\bar\alpha}m_k,vI)$，$v=\bar\alpha s^2+1-\bar\alpha$。後驗責任 $q_k\propto w_k\exp(-\|x-\sqrt{\bar\alpha}m_k\|^2/2v)$，$\bar m=\sum_kq_km_k$，
  $$\epsilon^*(x,i)=\frac{\sqrt{1-\bar\alpha_i}}{v_i}\big(x-\sqrt{\bar\alpha_i}\,\bar m(x)\big).$$
  Jacobian（C3 用）：$\partial\epsilon^*/\partial x=\frac{\sqrt{1-\bar\alpha}}v\big(I-\frac{\bar\alpha}v\,C\big)$，$C=\sum_kq_km_km_k^\top-\bar m\bar m^\top$；$\partial\epsilon^*/\partial z_j=-\frac{\sqrt{1-\bar\alpha}\sqrt{\bar\alpha}}v\,q_j(m_j-\bar m)$。`core.py` 以有限差分驗證（誤差 < 1e-6）。
- 數值保護：後驗責任 $q_k$（其指數包含 $\log w_k=z_k-\mathrm{LSE}(z)$，即 softmax 也在同一個 log-sum-exp 內）與 $\tilde Q$ 的權重 $g_k$ 都用 log-sum-exp（減去最大指數後再取 exp），正規化分母再取 $\max(\cdot,10^{-300})$；遠離所有峰（平方距離 ÷ $2\sigma_Q^2$ > 745）時不會出現 0/0；$v\ge\bar\alpha s^2+1-\bar\alpha>0$ 恆成立（$N=50$ 時 $1-\bar\alpha_1=1-e^{-0.00398}=0.00397>0$）；Eq. 1 的分母 $\sqrt{1-\bar\alpha_i}\ge0.063$（$N=50,i=1$），不會除以 0。
- 取樣鏈 `sampleChain(aN, Z, N, logits, wantT)`：Eq. 1 逐步；$i=1$ 不加噪聲；`wantT` 時同步傳遞切向量 $T^i=\partial a^i/\partial z$（2×4）：$T^N=0$，$T^{i-1}=\big(T^i-c_i(\partial_x\epsilon^*\,T^i+\partial_z\epsilon^*)\big)/\sqrt{\alpha_i}$，$c_i=\beta_i/\sqrt{1-\bar\alpha_i}$。
- 本頁不截斷動作到 $[-1,1]$：論文正文沒有提到截斷；理想去噪器的輸出在玩具中都留在範圍內（c2.py：所有 $N$ 的 outside = 0.000）。
- PRNG：`mulberry32(seed)`；`gauss(r)`：Box–Muller，每次取兩個均勻數 $u_1,u_2$，回傳 $\sqrt{-2\ln(1-u_1)}\cos(2\pi u_2)$（只用 cos 分支）。抽樣順序以 `core.py`／`c2.py`／`c3.py` 為準（下文寫出）。

### C1 · 噪聲排程（S4a）

- **教的概念**：Sec. 3.1 的排程；$N$ 小時每步噪聲大；$\bar\alpha_N$ 與 $N$ 無關（〔本頁推導〕）。
- **輸入**：分段按鈕 $N\in\{2,5,10,20,50\}$（`role="radiogroup"`，預設 5）。
- **顯示**：
  - 上圖：橫軸 $i/N\in[0,1]$，縱軸 $[0,1]$。曲線 $\bar\alpha(\tau)=\exp(-0.1\tau-4.95\tau^2)$（`--ink-3` 細虛線，標「訊號保留比例 的共同曲線」）；目前 $N$ 的 $\bar\alpha_i$ 為實心圓點（`--accent`）；$\beta_i$ 為以 $i/N$ 為中心的細長條（`--c-q`，半透明，寬度 $0.6/N$），標「每步噪聲 β」。兩組以標記形狀（圓點 vs 長條）＋直接標籤區分。
  - 讀數：SVG 外的 HTML 表，**每個 $i$ 一列**，欄標題為靜態 KaTeX（$i$、$\beta_i$、$\bar\alpha_i$、$\sqrt{\bar\alpha_i}$，共 4 欄），儲存格為 4 位小數純數字；$N\le5$ 列出全部，$N\ge10$ 只列 $i=1,2,N-1,N$ 四列並在中間加一列「…」；表格包在 `.tscroll` 內（4 欄在 358px 內不需捲動）。表下一行：「任何 N 下最後一步都只剩約 0.080 倍的原動作（$\bar\alpha_N=e^{-5.05}$）」。
- **計算**：直接用公式，無迴圈風險。
- **讀者應觀察到**：(1) $\bar\alpha_i$ 的圓點永遠落在同一條曲線上（$\beta_i$ 長條不是同一函數上的點）；(2) $N=2$ 時 $\beta$ 為 0.724、0.977，$N=50$ 時每步都小於 0.18；(3) 最後一點固定在 0.0064。
- 圖下：「讀數由本頁依 Sec. 3.1 的排程式即時計算；〔本頁推導〕$\bar\alpha_i$ 只依賴 $i/N$。」

### C2 · 行為複製：對角高斯與擴散（S2 實例 C2a、S4b 實例 C2b）

- **教的概念**：對角高斯 MLE 的 mode-covering（S2）；理想去噪器搭配 Eq. 1 能還原四峰，$N$ 越大越準（S4b、S8）。C2b 只在 S4b 的理想去噪器說明之後出現。
- **兩個實例**：C2a（S2）分頁「真實資料」「對角高斯（MLE）」；C2b（S4b）另加「擴散（理想去噪器）」與 $N$ 分段按鈕 $\{1,2,3,5,10,20,50\}$（預設 50）。兩者共用 seed 與快取。按鈕「重新抽樣」：seed + 1（預設 seed = 7）。
- **抽樣（每種 1000 點 `S=1000`）**：
  - 真實資料：每點先 `k = min(3, floor(4*r()))`，再 `gauss` ×2，$a=m_k+0.05g$。
  - 對角高斯：每點 `gauss` ×2，乘 $\sqrt{0.3225}$（〔本頁推導〕MLE）。另畫 1σ、2σ 等密度圓（虛線）。
  - 擴散：每點先抽 $a^N$（2 個 `gauss`），再依 $i=N,\dots,2$ 每步抽 2 個 `gauss` 作為 $z$；然後以 `logits = 0`（等權，即 $\pi_b$）跑 Eq. 1。
- **顯示**：正方形散點圖，座標 $[-1.2,1.2]^2$（框線標出動作範圍 $[-1,1]^2$；對角高斯落在框外的點畫在邊緣外並計數）；四個資料中心以空心十字＋淡色圓圈（半徑 0.2，標「資料峰」）標示；原點附近半徑 0.3 的虛線圓標「峰之間沒有資料的區域」。點用 `--c-q`（真實）、`--ink-2`（高斯）、`--accent`（擴散）三色並以點形（圓／方／菱形）區分；點半徑 2.5px、不透明度 .55（深色模式同值，顏色由 token 切換）。
- **讀數表**（SVG 外 HTML；3 位小數）：「落在資料峰 0.2 以內的比例」「落在中央無資料區域（半徑 0.3）的比例」「落在動作範圍外的比例」，以及四個峰各自的比例（上／右／下／左）。對角高斯另一行：「擬合高斯在原點的密度 ÷ 同一高斯在峰中心的密度 = 2.70（本頁閉式 MLE）」。對角高斯分頁的讀數與範圍外比例加「本頁推導／玩具」標籤：論文的 BC-MLE 是訓練出的網路，論文未說明它如何處理 $[-1,1]^2$ 邊界。
- **讀者應觀察到**：(1) 高斯只有 10.6% 的點在峰附近，13.6% 在中央無資料區域（seed 7；閉式 13.0%）；(2) 擴散 $N=50$ 時 100% 在峰附近、四峰各約 25%；(3) $N=2$ 時 63.6% 落在中央無資料區域，幾乎沒有點在峰附近；$N=5$ 約一半；(4) 玩具中 $N=1$ 的樣本全部在原點附近（只有一步時理想去噪器只能輸出後驗平均附近的點）。
- 圖下：「玩具：擴散分頁用理想去噪器代替訓練好的網路。理想去噪器沒有容量限制，所以這裡的 N 效應只來自 Eq. 1 的步數與排程（另外 $a^N$ 的起點與真實的第 N 步分布也不完全相同：$\sqrt{\bar\alpha_N}=0.08$）。論文只說 N 是表達力與成本的權衡，未分解原因；本圖不重現論文 Fig. 1–2。」
- **計算量**：$N=50$ 時 1000 × 50 × 4 成分 ≈ 20 萬次指數運算，同步完成並依 `${seed}|${N}` 快取。

### C3 · Diffusion-QL 玩具：去噪損失與 Q 值項的權衡（S7）

- **教的概念**：Eq. 3 的兩項如何加總；$\eta$ 越大策略越集中到報酬最高的峰；梯度經 Eq. 1 的鏈傳回參數；$N$ 小時樣本會落在峰之間。**不能**教的（頁面明寫）：去噪損失為什麼重要——玩具的 Q 精確、只有 4 個權重可訓練、沒有 OOD 高估，所以大 $\eta$ 沒有代價；論文則依領域調 $\eta$（Table 3）。
- **簡化**（頁面寫出，圖版標題旁「玩具」pill）：
  - 策略的可訓練參數只有 4 個混合權重的 logit $z_k$（成分位置與寬度固定在資料的值）；$\epsilon_\theta$ 是這個混合的理想去噪器（4.0）。初始 $z=0$（等權，等於行為策略；論文從頭訓練網路，這裡省略行為複製的學習過程）。
  - 單一狀態（bandit），沒有 Eq. 4 的自舉；Q 不學習，用**推定**的平滑函數 $\tilde Q(a)=\sum_kR_kg_k(a)/\sum_kg_k(a)$，$g_k=\exp(-\|a-m_k\|^2/2\sigma_Q^2)$，$\sigma_Q=0.2$（Nadaraya–Watson 平滑四個獎勵平均）。這個 $\tilde Q$ 在資料外不會高估，所以玩具**不**示範 OOD 高估。
  - $\alpha=\eta/2.375$（Eq. 3 的 $\alpha$，不是排程的 $\alpha_i$）：分母 2.375 是 $|\tilde Q|$ 在四個中心的平均（=四個獎勵平均的平均絕對值），常數、不微分。
- **損失與解析梯度**（每步）：
  - $\mathcal L_d$：抽 $B=256$ 個樣本；每個樣本依序抽 `k=min(3,floor(4r()))`、2 個 `gauss`（資料噪聲，$a=m_k+0.05g$）、`i = 1 + min(N-1, floor(N*r()))`、2 個 `gauss`（$\epsilon$）。$\mathcal L_d=\frac1B\sum\|\epsilon-\epsilon^*(x,i;z)\|^2$，$\nabla_z\mathcal L_d=-\frac2B\sum(\epsilon-\epsilon^*)^\top\partial_z\epsilon^*$。
  - $\mathcal L_q$：接著抽 $B$ 個鏈：每個先 $a^N$（2 個 `gauss`），再 $i=N..2$ 各 2 個 `gauss`。以 `sampleChain(..., wantT=true)` 得 $a^0$ 與 $T^0$；$\mathcal L_q=-\alpha\frac1B\sum\tilde Q(a^0)$，$\nabla_z\mathcal L_q=-\frac\alpha B\sum\nabla_a\tilde Q(a^0)^\top T^0$，$\nabla_a\tilde Q=\sum_k(R_k-\tilde Q)\,\bar g_k\,(m_k-a)/\sigma_Q^2$（$\bar g$ 為正規化權重）。
  - 所有 $\mathcal L_d$ 樣本先於 $\mathcal L_q$ 樣本抽取；同一個 PRNG（seed 7）在步與步之間連續使用。
- **最佳化**：Adam，學習率 0.03（玩具值），$\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$，含偏差修正（論文用 Adam 但未寫 β；這組是 Adam 的常用預設，推定）。總步數 600。
- **輸入**：滑桿 $\eta\in[0,3]$，步長 0.05，預設 1；chip「η = 0（只有去噪損失）」「η = 0.3」「η = 1」「η = 3」；分段按鈕 $N\in\{2,5,10,20\}$（預設 5）；按鈕「執行」／「暫停」、「重設」。改變 $\eta$ 或 $N$ 時自動重設（不自動執行）。重設時以 `cancelAnimationFrame`／`clearTimeout` 取消待執行的 handle，並把 `state.runId` 加 1；每個排程的回呼開頭比對 `runId`，不符就直接返回（避免舊迴圈繼續寫入新狀態）。
- **執行方式（不阻塞主執行緒）**：`requestAnimationFrame` 迴圈，每影格執行 Adam 步直到該影格用掉 8ms（`performance.now()`），至少 1 步；$N=5$ 時 600 步約 0.3 秒、$N=20$ 約 1.5 秒（估計，建置時量測並記錄）。reduced-motion 時以 `setTimeout(…,0)` 分段，每段同樣以 8ms 時間預算決定步數（至少 1 步），不重畫中間狀態，完成後一次重畫。
- **顯示**（≥ 768px 左右兩格；< 768px 上下堆疊，散點圖寬 = 內容寬（390px 時 358px），軌跡圖高 200px，控制列在最下方，所有按鈕觸控區 ≥ 44px）：
  - 左：散點圖（座標同 C2），每 50 步與結束時用**評估樣本**重畫：1000 條鏈，seed 11，抽樣順序同 C2 擴散分頁，參數為目前的 $z$。四個峰旁直接標獎勵平均（「3.0」「1.5」「5.0」「0.0」）。
  - 右：四條權重軌跡 $w_k$ 對步數（0–600），以顏色＋線型區分（下峰 `--accent` 粗實線、上 `--c-q` 實線、右 `--ink-3` 虛線、左 `--ink-2` 點線）＋線尾直接標籤（「下 5.0」等）；下方一條小圖畫 Ld 與 −Lq 的 20 步移動平均（SVG 標籤寫「Ld」「−Lq」純文字）。
  - 讀數（HTML 表）：列標題為靜態 KaTeX（步數、$w_{\text{上}}$…、$\mathcal L_d$、$\mathcal L_q$），值儲存格只放純數字（動態 DOM 不插入 TeX）；結束後再列評估指標：「落在峰附近（距峰中心 < 0.2）」「落在下峰附近（< 0.2）」「平滑 Q 的平均」；讀數區標題旁保留「玩具」pill。
  - 圖下兩行：「玩具：只有 4 個混合權重可訓練，Q 為推定的平滑函數，不重現論文 Fig. 1 的數字。」「讀數由本頁依 Eq. 1–3 即時計算，梯度為解析式（本頁推導）。」
- **讀者應觀察到**（$N=5$）：(1) $\eta=0$：四個權重在 0.25 附近擺動（MC 噪聲，約 ±0.03）；(2) $\eta=0.3$：下峰約 0.65；$\eta=1$：約 0.90；$\eta=3$：約 0.98——$\eta$ 越大越集中，而且在玩具中沒有代價（Q 精確、無 OOD 高估）；這**不**代表實際任務可以用大 $\eta$，論文 Adroit 用 0.1–0.15、Kitchen 用 0.005；(3) 權重 0.90 時評估樣本只有 72% 落在下峰附近：$N=5$ 的鏈本身會把部分樣本放在峰之間；$N=10$ 同為 $\eta=1$ 時為 76%；(4) $N=2$、$\eta=0$ 時沒有樣本在峰附近；$\eta=1$ 時 63% 在下峰附近，其餘在峰之間——與論文 Fig. 2「$N=2$ 仍有少數動作落在無資料區」方向相同，但玩具的比例遠大於論文所說的「少數」，且原因只是取樣步數（理想去噪器）。

### C4 · 結果瀏覽器（S9）

- **教的概念**：Table 1 的領域平均與逐任務比較；Table 4 的離線 vs 線上選擇。
- **輸入**：分頁（`role="tablist"`）「Table 1：與基準比較」「Table 4：離線與線上模型選擇」。Table 1 分頁：領域分段按鈕 Gym／AntMaze／Adroit／Kitchen（預設 Gym）；基準顯示「該任務最高的基準」（預設）或「全部基準」切換。分頁為 `role="tablist"`，左右方向鍵切換分頁、Home/End 到首尾。
- **顯示**：
  - Table 1：每個任務一組水平長條：Diffusion-QL（`--accent` 實心＋誤差鬚 ±標準誤）、該任務最高的基準（斜線圖樣，線色 `currentColor`，深色模式自動跟隨；方法名直接標籤）；「全部基準」時其餘方法為細灰長條（標籤在 ≥ 768px 顯示於右側，< 768px 只在點選或鍵盤 focus 該任務時於下方列表顯示；每個任務列 `tabindex=0`、`aria-expanded`，Enter/Space 展開）。Diffusion-QL 不是最高的任務在任務名前加「▲ 基準較高」文字標記（非只靠顏色）。下方「領域平均」列：Diffusion-QL vs 平均最高的單一基準（例：Gym 為 CQL 77.6），照錄論文 Avg；註明「不是各任務最高值的平均」。
  - < 600px：任務名在長條上方一行（不截斷，可換行），長條寬 = 內容寬；橫軸固定 0–120（Gym）、0–100（其他）。
  - Table 4：每任務兩條（離線實心、線上空心），四個領域平均照錄。
  - 每個分頁附完整數字的 `<table>`（捲動容器、`<caption>`）。
- **資料**：`verify/c4_data.json`（建置時逐字貼為 JS 常數 `RES`）。
- **讀者應觀察到**：四個領域平均都是 Diffusion-QL 最高；3 個任務標「基準較高」；AntMaze 的標準誤大（二元成功／失敗）；線上選擇在每個任務都不低於離線選擇。
- 圖下：「Table 1 的基準分數取自各方法的論文、Fu et al. 2020 或 Kostrikov et al. 2021b（Sec. 5）；Diffusion-QL 為離線模型選擇。」

### 4.5 共同的無障礙與互動規則

沿用 570 頁 4.5：靜態 TeX 用 KaTeX（`.katex-display{overflow-x:auto;overflow-y:hidden}`，Eq. 1、3、4 以 `aligned` 多行寫成，390px 寬時不需橫捲；若仍超出則可橫捲並有右側漸層提示）；**動態文字與 SVG 標籤不使用任何 Unicode 組合字元**（U+0300–U+036F），也不放 $\bar\alpha$、$\tilde Q$ 等需要附加符號的符號：SVG 用中文詞（「每步噪聲 β」「訊號保留比例」「平滑 Q」），帶附加符號或下標的符號只出現在 SVG 外的靜態 KaTeX 表頭，動態值儲存格只放純數字，動態文字寫「Ld」「Lq」；Eq. 1、Eq. 2、Eq. 3、Eq. 4 都以 aligned 多行寫（Eq. 1 先定義 $c_i$），390px 截圖確認不需橫捲；`input` 以 rAF 合併重畫；live region 與 SVG `aria-label` 只在 `change` 或停止輸入 500ms 後更新（C3 只在完成／暫停時宣告「第 N 步，下峰權重 …」）；分段按鈕 roving tabindex；觸控區 ≥ 44px；不在圖上做 hover 才能看到的資訊（C4 用點選／focus）；reduced-motion 下無補間；焦點 2px `--accent`；`ResizeObserver` 重畫，圖上文字 12–13px。

---

## 5. 驗收數值表

全部由 `verify/` 腳本產生（同演算法、雙精度）。容許誤差寫在各表；論文表格數字逐字相同。

**C1**（`c1_out.txt`；容許 ±0.0001）

| N | β（依 i） | ᾱ（依 i） |
| --- | --- | --- |
| 2 | 0.7240, 0.9768 | 0.2760, 0.0064 |
| 5 | 0.1959, 0.4588, 0.6358, 0.7549, 0.8350 | 0.8041, 0.4352, 0.1585, 0.0389, 0.0064 |
| 10 | 0.0578 … 0.6135 | 0.9422, 0.8041, …, 0.0166, 0.0064 |
| 20 | 0.0172 … 0.3859 | 0.9828 … 0.0064 |
| 50 | 0.0040 … 0.1796 | 0.9960 … 0.0064 |

所有 $N$：$\bar\alpha_N=0.006409$，$\sqrt{\bar\alpha_N}=0.0801$。$N=10$ 的偶數項等於 $N=5$ 的各項（只依賴 $i/N$）。

**C2**（seed 7，S = 1000，`c2_out.txt`；比例容許 ±0.002（同算法；只差浮點加總順序時，最多一兩個點跨越門檻）；閉式值 ±0.0001）

閉式：每維變異數 0.3225、標準差 0.5679；原點密度 0.4935、峰中心密度 0.1830、比 2.697；$P(|a|<0.3)=0.1302$。

| 分布 | 峰附近（< 0.2） | 中央無資料區域（< 0.3） | 範圍外 | 上／右／下／左（峰附近） |
| --- | --- | --- | --- | --- |
| 真實資料 | 1.000 | 0.000 | 0.000 | 0.257／0.236／0.255／0.252 |
| 對角高斯 | 0.106 | 0.136 | 0.136 | 0.032／0.024／0.022／0.028 |
| 擴散 N = 1 | 0.000 | 1.000 | 0.000 | — |
| N = 2 | 0.001 | 0.636 | 0.000 | 0.000／0.001／0.000／0.000 |
| N = 3 | 0.119 | 0.177 | 0.000 | 0.034／0.029／0.031／0.025 |
| N = 5 | 0.484 | 0.036 | 0.000 | 0.122／0.123／0.113／0.126 |
| N = 10 | 0.825 | 0.003 | 0.000 | 0.200／0.199／0.229／0.197 |
| N = 20 | 0.974 | 0.001 | 0.000 | 0.248／0.244／0.248／0.234 |
| N = 50 | 1.000 | 0.000 | 0.000 | 0.255／0.243／0.248／0.254 |

第一個擴散樣本（selfTest 用，容許 ±1e-4）：N = 5 → (−0.7729, 0.0369)；N = 50 → (−0.7513, 0.0037)。seed 8、9 的 N = 2／5／50 峰附近比例 0.000–0.001／0.472–0.483／1.000（趨勢不隨 seed 改變）。

**C3**（seed 7、評估 seed 11，B = 256，學習率 0.03，600 步，`c3_out.txt`。容許誤差：最後 50 步（第 551–600 步）平均權重 ±0.02；單步權重只供對照（步間抖動可達 0.014，不作 selfTest）；評估比例 ±0.01、平滑 Q 平均 ±0.02。容許的理由：頁面逐樣本加總、腳本依擴散步分組加總，600 步 Adam 後可能累積微小差異）

權重順序為（上, 右, 下, 左），分別為第 150／300／600 步：

| N | η | 第 150 步 | 第 300 步 | 第 600 步 | 第 551–600 步平均 | 峰附近 | 下峰附近 | Q 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0 | .279 .237 .240 .244 | .255 .248 .239 .258 | .249 .272 .245 .234 | .261 .246 .253 .240 | 0.000 | 0.000 | 2.370 |
| 2 | 1 | .042 .070 .830 .058 | .037 .071 .833 .060 | .032 .086 .816 .066 | .033 .081 .817 .069 | 0.634 | 0.634 | 4.965 |
| 5 | 0 | .246 .263 .229 .262 | .233 .246 .251 .269 | .234 .251 .249 .267 | .241 .258 .244 .257 | 0.498 | 0.129 | 2.285 |
| 5 | 0.1 | .244 .221 .373 .162 | .240 .197 .384 .179 | .228 .198 .406 .168 | .242 .212 .375 .171 | 0.522 | 0.225 | 2.826 |
| 5 | 0.3 | .169 .117 .644 .070 | .175 .096 .663 .066 | .181 .095 .647 .078 | .185 .116 .624 .076 | 0.600 | 0.421 | 3.589 |
| 5 | 1 | .063 .017 .907 .013 | .065 .017 .910 .008 | .078 .019 .896 .007 | .090 .018 .883 .008 | 0.792 | 0.721 | 4.516 |
| 5 | 3 | .018 .005 .971 .006 | .019 .002 .976 .003 | .021 .001 .977 .001 | .020 .001 .978 .001 | 0.900 | 0.877 | 4.819 |
| 10 | 0 | .251 .223 .284 .241 | .281 .256 .265 .198 | .237 .275 .239 .248 | .233 .261 .253 .253 | 0.830 | 0.211 | 2.363 |
| 10 | 1 | .083 .020 .880 .017 | .054 .020 .917 .010 | .080 .029 .882 .009 | .061 .021 .910 .009 | 0.909 | 0.764 | 4.379 |
| 10 | 3 | .028 .008 .955 .008 | .019 .004 .974 .003 | .014 .003 .983 .001 | .013 .002 .984 .001 | 0.954 | 0.920 | 4.831 |

（其餘組合見 `c3_out.txt`，每列末附 avg(551–600)。正文引用的玩具數字：$N=5$：η = 0 約 0.25、η = 0.3 約 0.65、η = 1 約 0.90、η = 3 約 0.98；評估：$N=5,\eta=1$ 下峰 0.72，$N=10$ 0.76，$N=2$ 0.63。）

**C4 與內文算術**（`c4_out.txt`）：各領域平均最高的單一基準 CQL 77.6、IQL 63.0、BCQ 56.5、IQL 53.3；Diffusion-QL 較低的任務恰為 3 個（hopper-medium −4.9、walker2d-medium-expert −2.9、antmaze-umaze-diverse −17.8）。重算平均與論文 Avg 的小差異（只在 R5 提及，頁面照錄論文）：Gym Diffusion-QL 87.9（論文 88.0）、Adroit BRAC-v −0.9（−1.0）、Table 2 CVAE-QL 76.7（76.6）、Table 4 Gym 線上 89.4（89.3）。Table 2 的 Diffusion-QL 欄 = Table 4 線上欄（97.2、112.3、111.2）。Table 5 的 TD3+BC halfcheetah-medium-expert 為 90.4，Table 1 為 90.7。

---

## 6. 實作計畫

### 6.1 檔案與共用外殼

- 單一檔案 `tut/443/build/index.html`；CSS、JS、時間線資料全部內嵌，不 `fetch` 本機檔案。上線時複製為 `tutorials/diffusion-ql.html`。
- **直接沿用** `tut/570/build/index.html` 的外殼（以 `tut/570/build/parts/assemble.py` 的方式組裝：外殼 + `parts/body.html` + `parts/widgets.js` + `parts/extra.css`）：CSS tokens（含深色模式三段覆寫、`pm-theme`）、字型、appbar、`#pm-bar`、章節目錄 `initToc`、`.plate` 圖版、callout、`<details>` 檢查題（`.qa`）、「本頁推導」「推定」「背景（論文外）」pill、「作者認為」引述、表格橫捲容器 `.tscroll`、演算法兩欄 `code-grid`、KaTeX 0.16.9 與 `renderMath()`、`?test` selfTest 框架（`ok()`、`#selftest`、`scrollWidth <= innerWidth`）、`renderTimeline`、`radiogroup()`、`announce()`、`ro()`（ResizeObserver）、SVG 輔助 `el`/`lp`/`tx`/`vline`/`marker`/`hbar`、`rng`/`gauss`。以下只列新增或變更。

### 6.2 新增 tokens、標籤與連結

- 圖表色：沿用 `--accent`（擴散樣本、Diffusion-QL、下峰）與 570 頁的 `--c-q`（light #3F6E8C／dark #8DB4D3），本頁用於真實資料點、上峰與 β 長條；不新增其他色。
- 新增「玩具」pill（與「推定」同樣式）。
- 站內連結：`../index.html#paper-443`；arXiv `https://arxiv.org/abs/2208.06193`；程式碼 `https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL`；筆記 `<slug>.html`。
- S5 靜態 SVG（viewBox 0 0 640 120；< 600px 改為 viewBox 0 0 320 260 的兩列排版，第一列 $a^N\to a^{N-1}\to\cdots$、第二列 $\to a^0\to Q$）：方塊內上標以 `<tspan dy="-0.5em" font-size="0.75em">` 實作（不用 `baseline-shift`），不用組合字元；寬版 viewBox 只在 ≥ 768px 使用（640 寬縮放到 ≥ 704px 時文字仍 ≥ 12px），< 768px 用窄版。

### 6.3 程式結構（單一 IIFE，`parts/widgets.js`）

- 共用（與 `core.py` 同名同式）：`MU`、`RBAR`、`SD=0.05`、`BMIN=0.1`、`BMAX=10`、`schedule(N)`、`epsStar(x, y, i, sch, logits, wantJac)`（回傳 `{ex, ey, Jx:[4], Jl:[8]}`，避免配置陣列：呼叫端傳入可重用的 Float64Array）、`sampleChain(...)`、`qToy(x, y, wantGrad)`、`QNORM=2.375`、`ELLQ=0.2`。
- C1：`initC1()`，`renderC1(N)`。
- C2：`initC2(container, {withDiffusion})`；共用 `c2State={seed, cache:Map}`；`drawsTruth(seed)`、`drawsGauss(seed)`、`drawsDiff(seed,N)`（回傳 Float64Array 2×S）、`metricsC2(A)`。
- C3：`initC3()`；`state={eta,N,step,logits,m,v,rng,hist:{w:[],Ld:[],Lq:[]},running}`；`gradC3(state)` → `{g:[4], Ld, Lq}`；`stepC3(state)`（一次 Adam）；`runC3()`（rAF + 8ms 預算）；`evalC3(logits,N)`（seed 11，1000 條鏈）。
- C4：`initC4()`，常數 `RES`（`c4_data.json` 逐字），`renderT1(domain, mode)`、`renderT4()`；「該任務最高的基準」「平均最高的單一基準」與例外在頁面中由資料計算（不手寫）。
- `selfTest()` 新增斷言（容許誤差見第 5 節）：
  - `schedule(5).abar[4]`、`schedule(50).abar[49]` 皆 = 0.006409；`schedule(10).abar[1] == schedule(5).abar[0]`（1e-12）。
  - 有限差分檢查 `epsStar` 的 `Jx`、`Jl` 在一個固定點（誤差 1e-6）。
  - C2：高斯變異數 0.3225；seed 7 擴散 N = 50 峰附近比例 1.000、N = 2 中央無資料區域 0.636（±0.002）；N = 5 第一個樣本 (−0.7729, 0.0369)（±1e-4）。
  - C3（與頁面相同的損失與 PRNG 順序）：$N=5,\eta=1$ 跑 600 步，第 551–600 步平均下峰權重 0.883 ± 0.02，且最後一步下峰權重 > 0.8；$\eta=0$ 時最後 50 步平均四權重皆在 [0.22, 0.28]。（selfTest 中同步執行，約 0.3 秒，只在 `?test` 時。）
  - C3 重設競態：開始執行後立即重設，下一個影格後 `step` 必須為 0。
  - C4：重算的平均最高單一基準 77.6、63.0、56.5、53.3；例外任務數 = 3。
  - 掃描所有 SVG `<text>` 與 `aria-live` 節點，不得含 U+0300–U+036F。

### 6.4 研究筆記與時間線

- 來源 `tut/443/notes/timeline.json`（15 筆），原樣貼進 `<script type="application/json" id="timeline-data">`；排序用 `Number(year)`。
- `TL_GROUP`（以 arXiv ID 為 key；不在表中歸「其他」）：

  | 類別 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | 擴散模型 | 1503.03585、1907.05600、2006.11239、2112.07804 | `ddpm-training-and-sampling-recap.html` |
  | Actor-critic | 1509.02971 | `ddpg-td3-actor-critic.html` |
  | 策略正則化 | 1812.02900、1906.00949、1911.11361、2106.06860、2110.06169 | `batch-constrained-policy-constraint-methods.html` |
  | 基準與保守 Q | 2004.07219、2006.04779 | `offline-rl-benchmarks-and-baselines.html` |
  | 同期工作 | 2205.09991、2206.11251 | `diffusion-in-rl-concurrent-work.html` |
  | 本文 | 2208.06193 | `diffusion-policy-class.html` |

- S13 延伸閱讀（依總覽的閱讀順序）：`diffusion-ql-lineage-hub`、`rl-mdp-q-learning-basics`、`offline-rl-distribution-shift`、`batch-constrained-policy-constraint-methods`、`ddpg-td3-actor-critic`、`gaussian-policy-multimodality-limits`、`ddpm-training-and-sampling-recap`、`diffusion-policy-class`、`diffusion-ql-loss-coupling`、`diffusion-in-rl-concurrent-work`、`diffusion-policy-experiments-d4rl`、`diffusion-policy-inference-cost`；每項附 frontmatter 的 `title`（本篇 7 篇讀 `tut/443/notes/`，其餘讀 `scripts/notes_split/` 或對應 `tut/<id>/notes/`；建置時已確認全部在 `tutorials/` 有對應頁）。另列站內教學 `td3-bc-minimalist-offline-rl.html`、`conservative-q-learning.html`、`implicit-q-learning.html`、`diffuser-planning-with-diffusion.html`。

### 6.5 文案規則（新增部分）

- 用詞：「離線強化學習（offline RL）」「行為策略（behavior policy）」「行為複製（behavior cloning, BC）」「分布外（OOD）」「策略正則化」「擴散步」「軌跡時間步」「重參數化」「目標網路」「常見錯誤」。mode-covering 寫成字面句「高斯的密度覆蓋所有資料峰，包括峰之間沒有資料的區域」；不用「蓋住」「拉」「空白區」「落後指標」「拼接」（後者改為字面定義並附論文用詞 stitch）。第一次出現附英文。
- 字面敘述：寫「使模型偏好取樣高 Q 值的動作」「把策略限制在資料附近」「往…靠近」，不寫「學會」「拉回」等擬人說法（引述作者時用「作者認為」）。`<details>` 的 summary 為「檢查題與參考答案」。
- 不寫「Diffusion-QL 在每個任務都最好」「兩個元件缺一不可」等超出表格的說法；結果敘述一律對應 1.4 步驟 10–11 的表。
- 玩具相關段落一律附「玩具」pill，並寫「不重現論文數字」。
- 第 1 節以外不得新增論文數字；頁面數字須與第 1、5 節一致。

### 6.6 建置後自我檢查

依 570 頁流程：390／768／1024／1440 截圖＋深色、鍵盤 focus、reduced-motion（截圖用本機 KaTeX 副本 `scratchpad/katex/0.16.9`，headless Chrome 無法連 CDN）；逐項評分並修正；`?test` 全數通過；逐條核對 12 個學習目標的答案出現在指定章節正文；390px 時 C3 兩圖上下堆疊、C4 任務名不截斷、C1 表格不需橫捲、Eq. 1–4 不撐破版面；量測 C3 在 $N=20$ 時單影格最長耗時（目標 < 16ms）並記錄於建置紀錄。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 玩具的可信度 | C2、C3 用理想去噪器與推定的 $\tilde Q$；C3 只訓練 4 個 logit、從行為策略出發、沒有自舉與 OOD 高估。玩具的比例（例如 N = 2 的 63.6%）不是論文的數字。 | 一律標「玩具」；正文寫明不重現 Fig. 1–2；論文結果只用 Sec. 4 原文描述。 |
| R2 | 獎勵與位置的對應 | paper.txt 的圖例文字順序與位置無法直接對應。 | 已以 300–500 dpi 渲染 PDF 第 6 頁核對顏色：上（藍）3.0、左（橘）0.0、右（綠）1.5、下（紅）5.0。 |
| R3 | 評估模型數不一致 | Table 1 標題「5 個模型」，App. C「6 個模型」。 | 頁面兩處都照錄並註明不一致。 |
| R4 | Table 2 的模型選擇 | Table 2 的 Diffusion-QL 欄等於 Table 4 線上欄；論文未說明。 | S10 照錄並以〔本頁推導〕註記「數字與 Table 4 線上欄相同」。 |
| R5 | 平均值的捨入 | 重算 Gym Diffusion-QL 平均 87.9 vs 論文 88.0 等四處小差異。 | 頁面照錄論文；只在 C4 附註一行。 |
| R6 | 結論的過度概括 | Table 1 標題與 Sec. 6 的「all」說法與逐任務數字不符（3 個例外）；Table 2 walker2d 例外。 | 只陳述表格數字（6.5）；Hero 第 3 卡寫明 17／20。 |
| R7 | 論文未寫明的細節 | $\rho$、$\gamma$、$\mathcal L_q$ 用哪個 Q、Adam 的 β、是否截斷動作。 | 常見錯誤第 7 條；玩具不截斷並寫明理由。 |
| R8 | 符號衝突 | $\alpha$（Q 權重）與 $\alpha_i$（排程）同字母；Eq. 4 的 $i$ 是 Q 網路編號。 | S5、S6 各加一句區分；符號表列出。 |
| R9 | C3 的計算量 | 每步 $B(1+N)$ 次去噪器求值與 2×4 切向量；$N=20$ 時 600 步約 3 百萬次求值。 | rAF + 8ms 預算分段；N 選項不含 50；建置時量測。 |
| R10 | 浮點與 PRNG 一致性 | C3 的 600 步 Adam 可能放大加總順序差異。 | 容許 ±0.01；selfTest 只驗 $N=5$ 兩個設定。 |
| R11 | 筆記更新 | 建置前若 notes 增減，`TL_GROUP` 與延伸閱讀需同步。 | 建置時重新讀取 `notes/`；未分類的 arXiv ID 歸「其他」並在建置回報列出。 |
| R12 | 相對連結 | 建置位置與上線位置不同，預覽時站內連結失效。 | 接受；預覽只檢查外部連結與頁內錨點。 |

---

## 8. 修訂紀錄（round 1）

依 `reviews/round1.md`（fidelity 13 + pedagogy 15，去重 24 項）全部處理，無駁回。主要變更：玩具 N 效應與論文表達力論點分開（S4b、S8、C2 圖下、LO10）；C3 明寫看不出去噪損失的作用、大 η 在玩具中無代價，對照 Table 3；App. G 保留「could be」（1.1、步驟 13、LO10、S8）；η 尺度不變限乘正常數；max Q backup 不用於 antmaze-umaze；「該任務最高的基準」與「平均最高的單一基準」分開；玩具符號改為 $R_k,q_k,z_k,w_k,\sigma_Q$；S1 加 TD 目標、S2 加 KL 方向／MLE／MMD／CVAE 與 0.3225 的兩行計算、S9 加各基準一行與「拼接」字面定義；S4 拆為 S4a（排程、C1）與 S4b（Eq. 2、理想去噪器、C2b），App. H 移到 S10；Eq. 1 先定義 $c_i$、Eq. 2 aligned；C3 重設取消 handle 並遞增 `runId`、reduced-motion 以 8ms 預算分段、selfTest 改用最後 50 步平均（`c3.py`／`c3_run.py` 新增 avg(551–600) 並重跑，評估值不變）、softmax 與 Σg 皆 log-sum-exp；C1 表格轉置；SVG 上標改 `dy`、窄版斷點 768；動態讀數不插入 TeX；用詞與 a11y（C4 列 `tabindex`/`aria-expanded`、tablist 方向鍵、點樣式、深色斜線）。

## 9. 修訂紀錄（readiness）

NO-GO B1：符號表拆為 S4b（$m_k,s,v_i,q_k,\bar m,\epsilon^*$，S4b 正文先定義）與 S7（$R_k,z_k,w_k,g_k,\sigma_Q,\tilde Q$）兩列。非阻斷：Hero 第 3 卡改用「平均最高的單一基準」；C3 重述 0.2 門檻並保留讀數「玩具」pill；前端逐項移植 Python 的 PRNG 與抽樣順序，以 `c2_out.txt`、`c3_out.txt` 為基準；站內教學四頁已確認存在於 `tutorials/`。

## 10. 建置紀錄（build round）

- 產出 `build/index.html`（約 115 KB），由 `build/parts/assemble.py` 從 `tutorials/prediction-targets-k-diff.html` 的外殼（CSS tokens、共用 JS helpers、章節目錄與主題切換）加上 `parts/body.html`、`parts/widgets.js`、`parts/extra.css`、`notes/timeline.json`、`verify/c4_data.json` 組成；406/570 外殼的 `radiogroup`、`sched`、`details.sym` 等少數補充放在 `extra.css`／`widgets.js`。`parts/preview.sh`、`parts/shoot.sh` 產生本機 KaTeX 預覽與單一圖版截圖（預覽檔用完即刪）。
- `?test`：37 passed, 0 failed（390 與 1440 寬皆同）。含 C1 排程與閉式、ε* Jacobian 有限差分、C2 全部驗收比例與首樣本、C3（N = 5、η = 1：最後 50 步平均下峰 0.883 ± 0.02、評估 0.792／0.721／4.516；η = 0 權重在 [0.22, 0.28]）、重設取消、C4 摘要、無組合字元、無水平捲動。無 console 訊息；KaTeX 正常渲染。
- 視覺一輪：C1 最右長條溢出 → 右邊距隨長條寬調整、移除圖內「最後一步」標籤；C3 軸標籤與刻度重疊 → 加大下邊距；峰的獎勵標籤改放在峰外並加底色描邊（`halo`），390px 時不再被裁切。
- 與設計的差異：C4 改用 HTML/CSS 長條（文字不隨縮放變小），「該任務最高的基準」以外框長條、Diffusion-QL 以實心長條區分，不用斜線圖樣；「全部基準」模式在所有寬度直接列出每個基準（不需要展開，因此不用 `aria-expanded`）。
- `npx impeccable detect`：0 anti-patterns（14 則 numbered-section-label 提示，允許）。
- LO 自查：`accept/page.txt`（12 題）中 12 個學習目標的關鍵詞與數字皆在正文。
