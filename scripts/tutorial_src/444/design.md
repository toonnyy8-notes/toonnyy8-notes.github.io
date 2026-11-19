# 設計文件：〈IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies〉互動教學

- 論文：Philippe Hansen-Estruch、Ilya Kostrikov、Michael Janner、Jakub Grudzien Kuba、Sergey Levine（UC Berkeley），arXiv 2304.10573（v2，2023-05-19）。資料庫編號 No. 444。方法簡稱 IDQL。程式碼 `https://github.com/philippe-eecs/IDQL`（Abstract）。
- 原文：`tut/444/src/paper.txt`（`tut/` = `scripts/tutorial_src/`）。「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」「Alg. n」皆指論文原文編號。
- 產出：`tut/444/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/idql-diffusion-actor-critic.html`（slug `idql-diffusion-actor-critic`），站內相對連結一律以上線位置為準。
- 前置教學：No. 569 IQL（已建置，`tut/569/build/index.html`，上線 `implicit-q-learning.html`）。本頁**不重教** expectile、Eq. 5–7 的 IQL 推導、Theorem 3、AWR 細節；S1 只用一章回顧並連過去。
- 驗算腳本：`tut/444/verify/common.py`（erfc、常態 CDF、玩具混合分布、三種 V 的解法、mulberry32，含自我檢查）、`c1.py`、`c2.py`、`c3.py`、`c4.py`、`tables.py`。輸出存為同名 `*_out.txt`。第 5 節驗收表全部取自這些輸出；全部以 `python3 -I` 執行。
- 研究筆記：`tut/444/notes/`（總覽 `idql-lineage-hub` + 5 篇主題筆記 + `timeline.json` 14 筆）；另引用既有 `rl-mdp-q-learning-basics`、`offline-rl-distribution-shift`、`iql-lineage-overview`、`expectile-regression-value-learning`、`awr-advantage-weighted-policy-extraction`、`ddpm-training-and-sampling-recap`、`trajectory-diffusion-planning`（`scripts/notes_split/`），以及同批 No. 443 的 `diffusion-policy-class`、`gaussian-policy-multimodality-limits`（`tut/443/notes/`，見 R10）。
- 文件慣例：本設計自行推導、論文沒有寫出的結論標「〔本頁推導〕」；為玩具元件選定、論文沒有給出的數值標「推定」；取自官方程式碼而非論文文字的內容標「程式碼」。頁面用同樣三種 pill。**所有玩具元件只示範機制，不重現論文的任何圖表數字**；頁面在每個圖版標題下寫「玩具（推定）：不是論文的實驗」。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

Offline RL 只用固定資料集學策略。IQL（Kostrikov et al. [26]）只用資料中的動作訓練 critic：以 expectile 損失讓 $V_\psi(s)$ 擬合 $Q_{\hat\theta}(s,a)$（Eq. 1），再以 $r+\gamma V_\psi(s')$ 為目標訓練 $Q_\theta$（Eq. 2），最後用 AWR（Eq. 3）抽出一個單峰高斯策略。問題是：這個 Q 函數對應哪一個策略並不清楚（Sec. 1）。IDQL 把 Eq. 1 的損失推廣成任意凸損失 $f$（Eq. 6），並把 $V$ 定義成某個「隱含 actor」$\pi_{imp}$ 下 $Q$ 的期望（Eq. 7）。Theorem 4.1 證明：只要 $f$ 凸且 $f'(0)=0$，最佳的 $V^*$ 也是 $\pi_{imp}(a|s)\propto\mu(a|s)\,|f'(Q-V^*)|/|Q-V^*|$ 下的平方損失最小解，所以 IQL 是一個 actor-critic 方法，隱含 actor 是行為策略 $\mu$ 的重新加權（Eq. 8）。這個 actor 可能多峰，單峰高斯 AWR 近似不了（Fig. 1）。因此 IDQL 用擴散模型學行為策略 $\mu_\phi$（Eq. 4），評估時每個狀態抽 $N$ 個候選動作，以 critic 算出的權重重新抽樣（Alg. 2），或直接取 Q 最大的候選（Eq. 13）。擴散模型與 critic 的訓練完全分開。D4RL 上 IDQL-A 總分 1213.8（DQL-A 1209.7、IQL-A 1070.4）；每個領域只允許一組超參數時 IDQL-1 為 1150.9（IQL-1 997.9、DQL-1 957.4、CQL-1 854.3），訓練 60 分鐘（-A）／40 分鐘（-1）（Table 1）。

### 1.2 研究動機（Sec. 1、4）

1. IQL 不查詢資料外動作，但「學到的值函數對應哪個策略」不清楚，因此難以理解 IQL 類演算法的瓶頸（Sec. 1）。
2. Kostrikov et al. 證明 $\tau\to1$ 時 IQL 回到 Q-learning，但實際使用 $0.5<\tau<1$，這時 Q 對應的策略沒有描述（Sec. 4）。
3. IQL 的 critic 訓練與顯式 actor 完全分開，這避免了不穩定與超參數敏感；代價是策略抽取必須準確近似隱含 actor（Sec. 1、4.2）。

### 1.3 貢獻（Sec. 1）

1. 把 IQL 推廣為 actor-critic 方法：critic 損失 $f$ 的選擇決定一個隱含 actor，它可以寫成行為策略的重新加權（Theorem 4.1、Eq. 8）。
2. 以擴散模型為基礎的策略抽取：擴散行為模型的樣本加上重新加權，近似隱含 actor 或最大化 critic 值（Alg. 2、Eq. 13）。
3. D4RL 上優於先前方法（包括以 IQL 為基礎與使用擴散模型的方法），跨領域只需少量調參，可用於 offline 與 online 微調；作者寫「據我們所知，第一個把擴散用於 online 微調的方法」（Sec. 2）。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S1 末尾以可收合表格提供；每章開頭另有「本章符號」小表）

| 符號 | 意義 | 首次使用 |
| --- | --- | --- |
| $s,a,s',r(s,a),\gamma$ | 狀態、動作、下一狀態、獎勵、折扣 | S1 |
| $\mathcal D=\{s,a,r,s'\}$ | 固定資料集 | S1 |
| $\mu(a\mid s)$ | 收集資料的行為策略（behavior policy）（Sec. 3）。注意論文也用 $\mu_\phi$ 表示擴散網路，頁面在 S6 明寫兩者關係 | S1 |
| $Q_\theta,Q_{\hat\theta},V_\psi$ | Q 網路、target Q 網路（參數 $\hat\theta$ 為 $\theta$ 的指數移動平均）、V 網路 | S1 |
| $Q^\pi,V^\pi$ | 策略 $\pi$ 的動作價值與狀態價值，$V^\pi(s)=\mathbb E_{a\sim\pi}[Q^\pi(s,a)]$；兩者都依賴策略 | S1 |
| $\tau$ | expectile／quantile 的階數 | S1 |
| $L_2^\tau(u)=\lvert\tau-\mathbb 1(u<0)\rvert u^2$ | expectile 損失（Eq. 1） | S1 |
| $\alpha$ | AWR 的溫度（Eq. 3）與指數損失的溫度（Eq. 11）；論文兩處同名。擴散的 $\alpha_t$、$\bar\alpha_t$ 一律帶下標 | S1、S3 |
| $f$、$f'$ | 一般凸損失、對引數 $u=Q-V$ 的導數。論文寫 $f'=\partial f/\partial V(s)$（依連鎖律差一個負號），但證明實際使用對引數的導數；頁面以對引數的導數為準並以小字註明 | S2 |
| $V^*(s)$ | Eq. 6 的最小解 | S2 |
| $\pi_{imp}(a\mid s)$ | 隱含 actor（Eq. 7、Theorem 4.1） | S2 |
| $w(s,a)$ | 重要性權重 $\lvert f'(Q-V^*)\rvert/\lvert Q-V^*\rvert$（Eq. 8） | S2 |
| $V^2_\tau,V^1_\tau,V_{exp}$ | expectile、quantile、指數損失的解（Eq. 9–11） | S3 |
| $\pi_{awr}$ | $\propto\mu\exp(\alpha Q)$，$V_{exp}$ 是它的正規化常數（Sec. 4.1） | S3 |
| $\mu_\phi(a\mid s)$ | 擴散行為模型（Alg. 1–2、Eq. 4）；S5 只當成可抽樣的黑盒，S6 才講訓練與取樣 | S5 |
| $N$ | 每個狀態抽的候選動作數（Alg. 2） | S5 |
| $T,\beta_t,\alpha_t,\bar\alpha_t,\epsilon$ | 擴散步數、噪聲排程、$\alpha_t=1-\beta_t$、$\bar\alpha_t=\prod_{i\le t}\alpha_i$、標準高斯噪聲。論文把 $\bar\alpha_t$ 寫成 $\hat\alpha_t$，且未定義 $\alpha_t,\hat\alpha_t$（頁面依 DDPM [20] 補上，標「本頁補充」） | S6 |
| $\eta,\lambda$ | target EMA 係數、學習率（Alg. 1） | S5 |

#### 步驟 1：IQL 回顧（Sec. 3，Eq. 1–3）——頁面 S1，只回顧不重教

$$L_V(\psi)=\mathbb E_{(s,a)\sim\mathcal D}\big[L_2^\tau(Q_{\hat\theta}(s,a)-V_\psi(s))\big],\quad L_2^\tau(u)=\lvert\tau-\mathbb 1(u<0)\rvert u^2\tag{Eq. 1}$$
$$L_Q(\theta)=\mathbb E_{(s,a,s')\sim\mathcal D}\big[(r(s,a)+\gamma V_\psi(s')-Q_\theta(s,a))^2\big]\tag{Eq. 2}$$
$$L_\pi(\phi)=\mathbb E_{(s,a)\sim\mathcal D}\big[\exp(\alpha(Q_{\hat\theta}(s,a)-V_\psi(s)))\log\pi_\phi(a|s)\big]\tag{Eq. 3}$$

- 兩個損失都只用資料中的動作（Sec. 3）。Eq. 3：$\alpha\in[0,\infty]$ 平衡「利用 critic」與「行為複製」。原文寫「minimizing」，但這是要**最大化**的加權對數概似（IQL 教學的 R3 記錄了 IQL 原文 Eq. 7 的同類問題；本頁見 R3）。
- 頁面：三式加一段話與連結「推導與 expectile 的性質見 [IQL 教學](implicit-q-learning.html)」。S1 另加兩句前置：策略 $\pi$ 的價值 $Q^\pi(s,a)$ 是在 $s$ 採取 $a$、之後依 $\pi$ 行動的期望折扣回報，$V^\pi(s)=\mathbb E_{a\sim\pi}[Q^\pi(s,a)]$；所以每個 Q 函數都對應「某個策略」，換策略就換 Q。一句白話說明「資料外動作（OOD action）」並連 `offline-rl-distribution-shift.html`。S1 結尾提出本文的問題：Eq. 1–2 學到的 $Q$ 是哪一個策略的值？

#### 步驟 2：推廣 critic 損失並定義隱含 actor（Sec. 4.1，Eq. 6–7）

$$V^*(s)=\arg\min_{V(s)}\ \mathbb E_{a\sim\mu(a|s)}\big[f(Q(s,a)-V(s))\big]\tag{Eq. 6}$$
$$V(s)=\mathbb E_{a\sim\pi_{imp}(a|s)}[Q(s,a)]\tag{Eq. 7}$$

- $\pi_{imp}$ 的具體形式在 Theorem 4.1（Eq. 8）才給出，頁面在 Eq. 7 旁寫「π_imp 的形式見下方 Eq. 8」。記 $L_V^f(V(s))=\mathbb E_{a\sim\mu}[f(Q(s,a)-V(s))]$（Eq. 6 的目標，Alg. 1 使用）。
- $f=L_2^\tau$ 時回到 IQL。Eq. 7 是 actor-critic 中值函數的慣用定義：$V$ 是某個策略下 $Q$ 的平均。問題變成：哪個 $\pi_{imp}$ 使 Eq. 6 的解滿足 Eq. 7？

#### 步驟 3：Theorem 4.1 與證明（Sec. 4.1，App. B）

**Theorem 4.1**：對每個狀態 $s$ 與滿足 $f'(0)=0$ 的凸損失 $f$，Eq. 6 的解也是 $\arg\min_{V(s)}\mathbb E_{a\sim\pi_{imp}}[(Q(s,a)-V(s))^2]$ 的解，其中

$$\pi_{imp}(a|s)\propto\mu(a|s)\frac{|f'(Q(s,a)-V^*(s))|}{|Q(s,a)-V^*(s)|},\qquad w(s,a)=\frac{|f'(Q(s,a)-V^*(s))|}{|Q(s,a)-V^*(s)|}\tag{Eq. 8}$$

證明四行（App. B，頁面全寫、每行附一句說明；$f'$ 為對引數 $u=Q-V$ 的導數）：
1. 目標對 $V$ 凸，最小解滿足一階條件 $0=\frac{\partial}{\partial V}\mathbb E_\mu[f(Q-V)]=-\mathbb E_{a\sim\mu}[f'(Q-V^*)]$（負號來自 $\partial(Q-V)/\partial V=-1$）。
2. $f$ 凸且 $f'(0)=0$，所以 $f'(x)$ 與 $x$ 同號：$f'(x)=|f'(x)|\cdot x/|x|$。
3. 代入：$0=\mathbb E_\mu\big[\frac{|f'(Q-V^*)|}{|Q-V^*|}(Q-V^*)\big]=Z_{imp}\,\mathbb E_{\pi_{imp}}[Q-V^*]$，$Z_{imp}$ 為正規化常數。
4. $\mathbb E_{\pi_{imp}}[Q-V^*]=0$ 正是 $\tfrac12\mathbb E_{\pi_{imp}}[(Q-V)^2]$ 在 $V=V^*$ 的一階條件，所以 $V^*=\mathbb E_{\pi_{imp}}[Q]$（即 Eq. 7）。
- 隱含假設（頁面一行 callout，〔本頁推導〕：證明用到但定理未寫）：$f$ 可微；$V^*$ 存在；$Q=V^*$ 處權重有定義（$f''(0)$ 有限）或該處機率為 0；$0<Z_{imp}<\infty$。quantile 不滿足第一與第四條（見步驟 4）。
- 注意事項（頁面寫成三點 callout）：(a) $\pi_{imp}$ 由 $V^*$ 定義，是「先有 $V^*$ 再讀出 actor」的關係〔本頁推導：定點的說法〕；(b) critic 訓練不需要建出 $\pi_{imp}$，只需在訓練結束後抽取（Sec. 4.1）；(c) 權重的複雜形狀使表達力弱的策略類別難以近似（Sec. 4.1，Eq. 8 之後）。
- 〔本頁推導〕Eq. 9 去掉了 $f'=2|\tau-\mathbb 1(u<0)|u$ 的常數 2；正規化後不影響 $\pi_{imp}$。

#### 步驟 4：三個損失與三個隱含 actor（Sec. 4.1，Eq. 9–12，App. C）

| 損失 | $f(u)$ | 解 | 權重 |
| --- | --- | --- | --- |
| Expectile | $L_2^\tau(u)$ | $V^2_\tau$ | $w^\tau_2=\lvert\tau-\mathbb 1(Q<V^2_\tau)\rvert$（Eq. 9） |
| Quantile | $\lvert\tau-\mathbb 1(u<0)\rvert\,\lvert u\rvert$ | $V^1_\tau$ | $w^\tau_1=\lvert\tau-\mathbb 1(Q<V^1_\tau)\rvert/\lvert Q-V^1_\tau\rvert$（Eq. 10） |
| Exponential（linex） | 原文寫 $\alpha\exp(u)-\alpha u$（見下方註） | $V_{exp}=\frac1\alpha\log\sum_a\exp(\alpha Q+\log\mu)$（Eq. 11） | $w_{exp}=\alpha\lvert\exp(\alpha(Q-V_{exp}))-1\rvert/\lvert Q-V_{exp}\rvert$（Eq. 12） |

- Expectile：Q 高於 $V$ 的動作權重 $\tau$、低於的權重 $1-\tau$，比為 $\tau/(1-\tau)$（$\tau=0.7$ 為 2.33，$0.9$ 為 9）〔本頁推導：比值〕。作者：$\tau$ 越大越偏離行為策略。
- Quantile：作者寫「平衡中位數行為表現與 Q-learning」，$\tau$ 越大外插越多；「這個隱含策略的點應該聚集在分位數附近」（Sec. 4.1）。〔本頁推導〕兩點：(i) quantile 的 $f$ 在 0 不可微，Theorem 4.1 的 $f'(0)=0$ 條件不成立（只能用次梯度理解）；(ii) 若 $Q$ 在 $\mu$ 下有連續密度且在 $V^1_\tau$ 處為正，$\int\mu\,w_1\,da$ 在 $Q=V$ 附近以 $\int dq/|q-V|$ 發散，正規化後 $\pi_{imp}$ 是 $V^1_\tau$ 處的點質量，與作者「聚集在分位數附近」一致。
- Exponential 的式子註：Eq. 11、Eq. 12 與 App. C.1 的推導對應 $f(u)=\exp(\alpha u)-\alpha u$（$f'(u)=\alpha(\exp(\alpha u)-1)$，$f'(0)=0$）；Sec. 4.1 文字寫 $\alpha\exp(u)-\alpha u$。頁面寫 App. C.1 的形式並以小字註明原文寫法（R3）。
- $V_{exp}$ 是 AWR 策略 $\pi_{awr}\propto\exp(\alpha Q+\log\mu)$ 的正規化常數，作者稱「這個損失等於加上 KL 散度約束」，與 Garg et al. [14]（EQL）、Xu et al. [49] 的 critic 相同、與 SAC 的 soft policy 相似（Sec. 4.1）。App. C.2：$D_{KL}(\mu\Vert\pi_{exp})=\mathbb E_\mu[\alpha(V-Q)]$（原文最後一行寫成 $\mathbb E_{(s,a)\sim\mathcal D}$；C.2 中的 $\pi_{exp}$ 指 $\mu\exp(\alpha(Q-V))$，即 $\pi_{awr}$）。注意方向是 $\mu\Vert\pi$，與 AWR 約束常用的 $\pi\Vert\mu$ 相反；頁面讀數標「KL(μ‖π_awr)」。
- 〔本頁推導〕另一個方向：$D_{KL}(\pi_{awr}\Vert\mu)=\mathbb E_{\pi_{awr}}[\alpha(Q-V_{exp})]$，所以 $V_{exp}=\mathbb E_{\pi_{awr}}[Q]-\tfrac1\alpha D_{KL}(\pi_{awr}\Vert\mu)$：$V_{exp}$ 是「AWR 策略的平均 Q 減去 KL 懲罰」，這說明下一點的 $\mathbb E_{\pi_{awr}}[Q]>V_{exp}$（玩具 α = 1：$2.3980-0.2435=2.1545$，`c1_out.txt`）。
- 〔本頁推導，C1 呈現〕$\pi_{awr}$ 與 Theorem 4.1 的 $\pi_{imp}$（用 $w_{exp}$）不是同一個分布：$\mathbb E_{\pi_{imp}}[Q]=V_{exp}$（定理保證），但 $\mathbb E_{\pi_{awr}}[Q]>V_{exp}$（玩具中 $\alpha=1$：2.3980 對 2.1545）。作者寫「若 IQL 搭配 AWR 抽取，這個 $f$ 會給出對應的正確 critic」（Sec. 4.1），本頁不改寫這句，只並列兩個分布。
- Fig. 2 的比較（照原文）：bandit 上 expectile 隨 $\tau$ 平滑上升、quantile 對應累積分布函數、exponential 對齊最大值；隱含策略中 expectile 覆蓋範圍廣、quantile 緊密集中（原文此處寫「around the mean」，與 Sec. 4.1 的「around the quantile」不同，R3）、exponential 在最大值附近有覆蓋。作者：exponential 看似最佳，但實務上不穩定（Sec. 4.1、6.3）。

#### 步驟 5：為什麼單峰高斯 AWR 不夠（Sec. 4.2，Fig. 1）

- 隱含 actor 是行為策略的重新加權，行為策略多峰時它也多峰；標準 IQL 用單峰條件高斯近似，而 critic 不知道這個近似、不會調整（一般 actor-critic 中 critic 會隨 actor 的限制調整）（Sec. 4.2）。作者的假設：解耦的好處只有在最終顯式 actor 夠強時才能實現。
- Fig. 1：2D bandit、三個動作群、$r(a_1,a_2)=a_1+a_2$、0.9 expectile；各種反溫度的 AWR 高斯都近似錯誤，擴散抽取法的樣本正確。
- 為什麼不用「擴散模型 + AWR 權重損失」：表達力強的模型在重要性加權目標下會提高所有訓練點的概似、不論權重（[6, 48]）；作者實驗中 AWR 加權的 DDPM 沒有幫助（App. F.5：只抽 1 個樣本時通常比 IDQL 差，最好時與多樣本設定相當；結論「AWR 不必要，且多一個訓練超參數」）。

#### 步驟 6：IDQL 的演算法（Sec. 4.2、5，Alg. 1–2，Eq. 13）

```
Alg. 1（訓練）  超參數：學習率 λ、EMA η
  while 未收斂:
    ψ ← ψ − λ∇ψ L_V^f(ψ)          # Eq. 6（IDQL 用 expectile，即 Eq. 1）
    θ ← θ − λ∇θ L_Q(θ)            # Eq. 2
    θ_targ ← (1 − η)θ_targ + ηθ    # η = 0.005（App. E）
    φ ← φ − λ∇φ L_µ(φ)            # 擴散行為複製（第 06 章，Eq. 4），與 critic 無關
Alg. 2（評估）  超參數：每個狀態的樣本數 N
  觀察 s；抽 a_i ~ µφ(a|s), i = 1..N
  算 w(s, a_i)（Eq. 9、10、12，或 13 的 one-hot）
  p_i = w(s,a_i) / Σ_j w(s,a_j)；依 p_i 做類別抽樣得到動作
```
（頁面程式碼用 `θ_targ`，避免等寬字型中的組合字元。）

$$\pi(s)=\arg\max_{a_i\sim\mu(a|s),\ i=1..N}Q(s,a_i)\tag{Eq. 13}$$

- 作者寫 Alg. 2「給出正確隱含 actor 的樣本」（Sec. 4.2）。〔本頁推導〕這是自正規化重要性抽樣：$N=1$ 時就是 $\mu$，$N\to\infty$ 時趨近 $\pi_{imp}$；只有後者是作者那句話成立的情形，中間的 $N$ 沒有一般結論（C2 只顯示一個玩具）。
- 〔本頁推導〕用 quantile（Eq. 10）時權重在 $Q=V$ 發散，連續的 $\mu$ 下 Alg. 2 的正規化不穩定（同步驟 4 的點質量）；有限個候選時只要某個候選很接近 $V$，它就拿走幾乎全部權重。
- 實務上取 Q 最大的候選（Eq. 13，權重為 one-hot）評估表現較好，類比「隨機 actor 訓練 critic、確定性 actor 評估」（[5, 17]）（Sec. 4.2）。
- IDQL = 上述一般演算法 + 擴散 $\mu_\phi$ + DDPM 目標（Sec. 5）。理論上任何生成模型都可以，但作者發現擴散模型最好（Sec. 4.2）。
- 擴散模型只在評估時與 critic 互動，所以調參便宜（Sec. 2）；訓練時間 IDQL-A 60m，比 DQL 240m、SfBC 785m 快（Table 1–2、Sec. 6.1）。
- Online 微調的兩種做法（Sec. 4.2）：(1) 凍結 $\mu_\phi$，用 argmax 動作探索，只微調 $Q_\theta,V_\psi$（Table 3「IDQL-Max」）；(2) 用 $\pi_{imp}$ 探索，三者都微調（「IDQL-Imp」）。

#### 步驟 7：擴散行為模型（Sec. 3，Eq. 4–5，App. E、F.3、G）

$$L_\mu(\phi)=\mathbb E_{t\sim U(1,T),\,\epsilon\sim\mathcal N(0,I),\,(s,a)\sim\mathcal D}\big[\lVert\epsilon-\mu_\phi(\sqrt{\bar\alpha_t}\,a+\sqrt{1-\bar\alpha_t}\,\epsilon,\ s,\ t)\rVert\big]\tag{Eq. 4}$$
$$a_{t-1}\leftarrow\frac1{\sqrt{\alpha_t}}\Big(a_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\mu_\phi(a_t|s,t)\Big)+\sqrt{\beta_t}\,\epsilon,\quad t=T,\dots,1,\ a_T\sim\mathcal N(0,I)\tag{Eq. 5}$$

- 前向過程 $q(a_t|a_{t-1})=\mathcal N(\sqrt{1-\beta_t}a_{t-1},\beta_tI)$。網路 $\mu_\phi$ 預測加入的噪聲 $\epsilon$（論文稱 score network）。Eq. 4 原文範數未平方（DDPM [20] 為平方），頁面照錄並註明（R3）。
- 程式碼（`jaxrl5/networks/diffusion.py` 的 `ddpm_sampler`）：最後一步（時間索引 0，即 Eq. 5 的 $t=1$）不加噪聲；`clip_sampler` 開啟時每一步都截斷到 $[-1,1]$，最終輸出再截斷一次；Eq. 5 文字寫成每一步都加噪聲。頁面寫論文式子，並以「程式碼」pill 註明這兩點。
- 設定（App. E、F.3）：$T=5$；排程 variance preserving（VP，[43]）。F.3：2D 資料集 $T=50$ 最好、D4RL $T=5$ 最好（作者推測控制資料較確定，仍是開放問題）；增加 $T$ 對 D4RL 有小幅負面影響；$T>20$ 評估很慢；排程影響大，VP 最好、cosine 在小 $T$ 也很好、linear 需要太大的 $T$；作者認為訊噪比、尤其第一個加噪步很關鍵。
- 三個排程的公式（**程式碼**，論文只給名稱與引用；頁面放在 `details.more`）：VP $\beta_t=1-\exp\!\big(-\frac{b_{min}}T-\frac{(b_{max}-b_{min})(2t-1)}{2T^2}\big)$，$b_{min}=0.1,b_{max}=10$；linear：$\beta$ 由 $10^{-4}$ 線性到 $0.02$；cosine：$\bar\alpha(t)=\cos^2\!\big(\frac{t/T+0.008}{1.008}\cdot\frac\pi2\big)/\bar\alpha(0)$，$\beta_t=\mathrm{clip}(1-\bar\alpha_t/\bar\alpha_{t-1},0,0.999)$。〔本頁推導〕VP 的 $\bar\alpha_T=\exp(-5.05)=0.0064$，與 $T$ 無關；linear 在 $T=5$ 時 $\bar\alpha_T=0.9506$，終點遠非純噪聲。
- 網路（Sec. 5、App. G、F.6）：簡單 MLP 會產生很多分布外樣本（Fig. 3）；增加 batch 與容量改善，但仍有很多離群點；採用帶 layer norm 的殘差網路 LN_Resnet：Dense(hidden) → $n$ 個 MLPResNet block（Dropout → LayerNorm → Dense(4·hidden) → Activation → Dense(hidden) → Add）→ Activation → Dense(action_dim)，hidden 256、$n=3$。F.6：容量大且正則化好的網路很關鍵；dropout 大幅減少過擬合；layer norm 有小幅好處。F.4：batch size 對 antmaze 重要、對 locomotion 影響小，不如架構與容量重要。
- 離群點為什麼危險：它們可能落在已訓練 Q 函數的分布外，得到錯誤的高 Q 值（Sec. 5）。這把 S6 接到 S7 與 C3。

#### 步驟 8：$N$ 與架構的交互作用（Sec. 6.4，Fig. 4）

- LN_Resnet 對 $N$ 不敏感：locomotion 上 $N$ 越大越好，antmaze 上 $N$ 影響小；MLP（三層、啟動層數相近）則 $N$ 越大表現越差，antmaze 尤其明顯（Sec. 6.4）。
- 〔本頁推導，C3 呈現〕一個與 Sec. 5 說法相容的機制：若行為模型以比例 $\varepsilon$ 產生離群樣本，且 critic 高估它們，argmax 選到至少一個離群樣本的機率約為 $1-(1-\varepsilon)^N$，隨 $N$ 上升。這是機制示意，不是 Fig. 4 的重現。

#### 步驟 9：critic 損失消融（Sec. 6.3，Fig. 4）

- expectile 與 quantile 搭配 argmax 抽取都表現好；exponential 不穩定且較差，作者認為 EQL／IVR 的 exponential critic 需要更多調參與技巧（那些論文也回報不穩定）。
- 隱含策略抽取不如 argmax；作者認為原因是隱含策略有非零機率選到差的動作。整體以 expectile + 貪婪抽取最強（Sec. 6.3）。結論章：雖可推廣到多種損失，原本的 expectile 在現有任務上最好（Sec. 7）。

#### 步驟 10：實驗設定與結果（Sec. 6.1、6.2，Table 1–5，App. D–F）

- 兩種協定：「-A」為論文回報值（任意調參）；「-1」每個領域（locomotion、antmaze）只能選一個主要超參數值（IDQL：locomotion 全用 $\tau=0.7$、antmaze 全用 $\tau=0.9$，共兩個值）。-1 的基線（CQL、IQL、DQL）是作者用各自的程式重跑；作者寫這「可能無法公平代表每個演算法」，目的是展示開箱即用的表現並給出公平比較的協定（App. D.2）。
- IDQL 自己的 -A 與 -1 用相同的 τ（App. E），差別在步數（critic 1.5M／1M、actor 3M／2M）與 $N$（-A：antmaze 32、locomotion 128；-1：64）。所以 IDQL「-A → -1 只小幅下降」不是調參敏感度的證據，頁面明寫。-1 的掃描：CQL $\lambda\in\{1,2,5,10\}$、DQL $\eta\in\{1.0,2.0,2.5,3.0\}$、IQL 與 IDQL $\tau\in\{0.6,0.7,0.8,0.9\}$（App. D.2）。
- Table 1 摘錄（總分）：

| | CQL-A | IQL-A | DQL-A | IDQL-A | CQL-1 | IQL-1 | DQL-1 | IDQL-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| locomotion-v2 | 698.5 | 692.4 | 791.2 | 739.2 | 674.3 | 629.5 | 741.2 | 711.6 |
| antmaze-v0 | 303.6 | 378.0 | 418.5 | 474.6 | 180.0 | 368.4 | 216.4 | 439.3 |
| total | 1002.1 | 1070.4 | 1209.7 | 1213.8 | 854.3 | 997.9 | 957.4 | 1150.9 |
| training time | 80m | 20m | 240m | 60m | 80m | 20m | 240m | 40m |

- 作者敘述（照原文）：標準協定下 locomotion 與最佳先前方法相當、antmaze 優於先前方法；-1 協定下 IDQL 只小幅下降，先前方法下降得多、antmaze 尤其明顯；antmaze 上比最佳方法（IQL）高 +70 分（$439.3-368.4=70.9$）。
- 表格實際的 -A → -1 變化（`tables_out.txt`，頁面列成一行小表）：antmaze CQL −123.6、IQL −9.6、DQL −202.1、IDQL −35.3；locomotion CQL −24.2、IQL −62.9、DQL −50.0、IDQL −27.6；total CQL −147.8、IQL −72.5、DQL −252.3、IDQL −62.9。頁面寫：「antmaze 下降多」對 CQL、DQL 成立，對 IQL 不成立（IQL 只降 9.6，比 IDQL 少）；IDQL 對 IQL 的 antmaze 差距由 96.6（-A）縮為 70.9（-1）。-1 下 locomotion 由 DQL 最高（741.2 對 711.6），IDQL 是 total 最高。-A 的 locomotion 也由 DQL 最高（791.2 對 739.2）。頁面不寫「全面最佳」。
- Table 2（含 %BC、DT、TD3、EQL、SfBC）：IDQL total 1213.8 最高，antmaze 474.6 最高（SfBC 445.2、DQL 418.5、EQL 386）。Table 4 為逐任務全表（IDQL 10 個 seed 平均）。Table 5 Maze2D：IDQL 237.5、IQL 94、Diffuser 358.5，作者推測基於模型的規劃在 Maze2D 泛化較好。
- 訓練步數：-A critic 1.5M、擴散 3M 步；critic 超過 2M 步會略不穩（App. D.1）。-1 為 1M／2M（App. E）。獎勵處理沿用 IQL：locomotion 標準化、antmaze 減 1。
- Table 3（online 微調總分，offline → online）：RLPD 0.0 → 571.0；IQL 408.2 → 473.7；IDQL-Max 459.9 → 559.1；IDQL-Imp 462 → 548.2；Cal-QL 只有四個任務。作者：pre-training 與微調後都大幅優於 IQL，微調上「remains competitive」with RLPD 與 Cal-QL 且 pre-training 較強，主要增益來自 antmaze-large（Sec. 6.2）。頁面註腳：IDQL 兩種做法的總分都低於 RLPD 571.0；IDQL-Imp 的 antmaze-umaze-diverse 由 78.4 降到 73.0；RLPD 的 0.0 是沒有 offline 預訓練，不是分數比較。Fig. 9：凍結 actor 只需 100k 樣本即達峰值。微調設定：critic 預訓練 1M、擴散 2M 步；每個環境步 critic 1 次梯度，若也微調 actor 則 2 次（App. D.3）。
- 〔tables.py〕Table 3 的 IQL offline 總分印成 408.2，但六列相加為 370.1；online 473.7 與列和相符。Table 1 IDQL-A antmaze-med-play 為 84.5，Table 4 為 84.2，總分 474.6 與 84.2 相符。另外 IQL-A locomotion（印 692.4、列和 692.6）、CQL-1 antmaze（180.0、179.0）、IQL-1 antmaze（368.4、367.6）、DQL-1 total（印 957.4，741.2 + 216.4 = 957.6）不一致。頁面一律照原文印出，在表下小字列出這些差異（R4）。

#### 步驟 11：實作設定（App. E、D）

| 項目 | 值 |
| --- | --- |
| critic 與 V 網路 | 同 IQL：2 層 MLP、hidden 256、ReLU |
| 學習率（所有網路）／最佳化 | 3e-4／Adam；actor 用 cosine decay（長度 = actor 步數） |
| batch | critic 256、actor 1024 |
| τ（expectile） | 0.7（locomotion）、0.9（antmaze） |
| τ（quantile）／α（exponential） | 0.6／0.8；1.0／0.5（locomotion／antmaze） |
| 步數 | critic 1.5e6（-A）、1e6（-1）；actor 3e6（-A）、2e6（-1） |
| target EMA | 0.005 |
| 擴散 | $T=5$、VP 排程、LN_Resnet 3 個 block、hidden 256、dropout 0.1 |
| $N$ | 32（antmaze -A）、128（locomotion -A）、64（-1） |

（App. E 表中「Critic/Actor Pre-Training Steps 1e6/2e6」標為「(Figure 3)」，與 App. D.3 的微調預訓練步數相同，應指 Table 3；頁面寫「微調預訓練」並註明原文標示，R3。）

常見錯誤（頁面 S8 逐條列出）：
1. 在 critic 訓練中查詢擴散模型的樣本：IDQL 的 critic 只用資料中的 $(s,a)$；擴散模型只在評估時出現（Sec. 2）。
2. 用 AWR 權重訓練擴散模型：表達力強的模型會提高所有點的概似，作者實驗無幫助（Sec. 4.2、App. F.5）。
3. 評估時用重新抽樣而不是 argmax：作者發現 argmax 評估較好，隱含策略抽取較差（Sec. 4.2、6.3）。
4. 用一般 MLP 當擴散網路並加大 $N$：離群點被 critic 高估，$N$ 越大越差（Sec. 5、6.4）。
5. 照 DDPM 的 linear 排程配 $T=5$：linear 需要太大的 $T$（App. F.3）。
6. 把 Eq. 3 當成要最小化的式子（正負號，R3）；把 Eq. 4 的範數當成未平方照抄（DDPM 為平方）。

#### 步驟 12：限制（Sec. 7、App. D.3）

小維度動作空間（例如 locomotion）有過擬合問題，加 dropout 可緩解但損及其他環境；Adroit 上 online 微調效果不好，作者推測過度預訓練有害，且 Adroit 需要大量探索、行為正則化可能有害（App. D.3）。

---

## 2. 學習目標（驗收測試）

每章結尾以 `<details class="check">` 呈現該章檢查題（題目與參考答案照搬）。**每一條參考答案都必須寫在指定章節的正文段落中**（不只在元件、表格、圖或答案框裡）。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說出 IQL 留下的問題 | IQL 的 V 與 Q 怎麼訓練？本文要回答什麼問題？ | V 以 expectile 損失 $L_2^\tau(Q_{\hat\theta}-V_\psi)$ 擬合資料動作的 Q（Eq. 1），Q 以 $r+\gamma V_\psi(s')$ 為目標做平方損失（Eq. 2），都只用資料中的動作；問題是學到的 Q 對應哪一個策略不清楚（Sec. 1） | S1 |
| LO2 | 寫出推廣後的 critic 與隱含 actor 的定義 | Eq. 6 與 Eq. 7 各是什麼？ | Eq. 6：$V^*=\arg\min_V\mathbb E_{a\sim\mu}[f(Q-V)]$，$f$ 為任意凸損失，$f=L_2^\tau$ 即 IQL；Eq. 7：$V(s)=\mathbb E_{a\sim\pi_{imp}}[Q(s,a)]$ | S2 |
| LO3 | 敘述並證明 Theorem 4.1 | 定理的條件與結論？證明的關鍵一步？ | $f$ 凸且 $f'(0)=0$；$V^*$ 也最小化 $\mathbb E_{\pi_{imp}}[(Q-V)^2]$，$\pi_{imp}\propto\mu\,|f'(Q-V^*)|/|Q-V^*|$（Eq. 8 的 $w$）；關鍵：$f'(x)=|f'(x)|\,x/|x|$，把一階條件 $\mathbb E_\mu[f'(Q-V^*)]=0$ 改寫成 $\mathbb E_{\pi_{imp}}[Q-V^*]=0$（App. B） | S2 |
| LO4 | 讀出 expectile 的隱含 actor | Eq. 9 的權重？$\tau=0.9$ 時上下兩側權重比？C1 預設（τ = 0.9）的 $V$ 與 $\mathbb E_{\pi_{imp}}[Q]$？ | $w=\lvert\tau-\mathbb 1(Q<V)\rvert$：Q 高於 $V$ 的動作權重 τ、低於的 1−τ，比為 9；τ 越大越偏離行為策略；C1：$V=2.5770$，$\mathbb E_{\pi_{imp}}[Q]=2.5770$（兩者相等即 Theorem 4.1） | S3 |
| LO5 | 讀出 quantile 與 exponential 的隱含 actor | Eq. 10–12 的權重與 $V_{exp}$？$V_{exp}$ 與 AWR 的關係？App. C.2 算的是哪個方向的 KL？ | quantile $w=\lvert\tau-\mathbb 1(Q<V)\rvert/\lvert Q-V\rvert$，點聚集在分位數附近；$V_{exp}=\frac1\alpha\log\sum_a\exp(\alpha Q+\log\mu)$ 是 $\pi_{awr}\propto\exp(\alpha Q+\log\mu)$ 的正規化常數，作者稱等於加上 KL 散度約束；$w_{exp}=\alpha\lvert\exp(\alpha(Q-V))-1\rvert/\lvert Q-V\rvert$；App. C.2：$D_{KL}(\mu\Vert\pi_{exp})=\mathbb E_\mu[\alpha(V-Q)]$（μ 在前） | S3 |
| LO6 | 比較三種損失（理論與實驗） | Fig. 2 中三者的值與隱含策略有何不同？Sec. 6.3 的實驗結論？ | 值：expectile 隨 τ 平滑上升、quantile 對應累積分布函數、exponential 對齊最大值；策略：expectile 覆蓋廣、quantile 緊密集中、exponential 在最大值附近；實驗：expectile、quantile 搭配 argmax 都好，exponential 不穩定且較差，隱含策略抽取不如 argmax，expectile + 貪婪抽取最強 | S9 |
| LO7 | 說明為何不用單峰高斯 AWR，也不用 AWR 訓練擴散模型 | 兩個理由？ | 隱含 actor 是行為策略的重新加權，可能多峰，單峰高斯近似不了，而解耦的 critic 不會隨 actor 調整（Sec. 4.2、Fig. 1）；表達力強的模型在重要性加權目標下會提高所有訓練點的概似，AWR 加權的 DDPM 沒有幫助（Sec. 4.2、App. F.5） | S4 |
| LO8 | 描述 IDQL 的訓練與抽取 | Alg. 1 每步更新什麼？Alg. 2 怎麼選動作？評估時實際用什麼？ | ψ（Eq. 6/1）、θ（Eq. 2）、θ_targ 以 EMA η = 0.005、φ（Eq. 4，擴散行為複製，與 critic 無關）；抽 N 個 $a_i\sim\mu_\phi$，權重正規化成 $p_i$ 後做類別抽樣；評估用 Eq. 13 取 Q 最大的候選（表現較好） | S5 |
| LO9 | 說出擴散行為模型的訓練、取樣與設定 | Eq. 4 預測什麼？T 與排程？為什麼用 LN_Resnet？ | 預測加入的噪聲 ε（DDPM 目標），Eq. 5 從 $a_T\sim\mathcal N(0,I)$ 逐步去噪；$T=5$、VP 排程（2D 資料 $T=50$ 最好；linear 需要太大的 T）；簡單 MLP 產生很多離群樣本，離群點可能被 critic 給出錯誤的高 Q 值；LN_Resnet 3 個 block、hidden 256、dropout 0.1；T = 5 是 D4RL 上的最佳值，本頁玩具沒有網路誤差，T 越大越好，兩者不矛盾 | S6、S7 |
| LO10 | 說明 N 與架構的交互作用 | Sec. 6.4 的結果？C3 預設（ε = 0.01）下 N = 8 與 N = 256 的期望真實獎勵？ | LN_Resnet：locomotion 上 N 越大越好、antmaze 影響小；MLP：N 越大越差，antmaze 尤其明顯；C3（玩具）：2.6880 對 0.2552，選到離群樣本的機率 $1-(1-\varepsilon)^N$ 由 0.0773 升到 0.9236 | S7 |
| LO11 | 讀出 D4RL 結果 | Table 1 的 total（-A 與 -1）與訓練時間？-1 是什麼協定？antmaze 的 +70 指什麼？ | -A：IDQL 1213.8、DQL 1209.7、IQL 1070.4、CQL 1002.1；-1（每個領域只選一個主要超參數值，基線為作者重跑）：IDQL 1150.9、IQL 997.9、DQL 957.4、CQL 854.3；IDQL 60m（-A）／40m（-1），DQL 240m、IQL 20m；+70：-1 協定 antmaze 總分 IDQL 439.3 對 IQL 368.4（-A 時差距 96.6；IQL 的 antmaze 只降 9.6） | S9 |
| LO12 | 讀出 online 微調結果與限制 | 兩種微調做法與 Table 3 總分？論文寫的限制？ | IDQL-Max 凍結擴散模型、argmax 探索、只微調 Q 與 V：459.9 → 559.1；IDQL-Imp 用隱含策略探索、三者都微調：462 → 548.2；IQL 408.2 → 473.7、RLPD 0.0 → 571.0；限制：小維度動作空間過擬合（dropout 緩解但損及他處）、Adroit 微調效果不好 | S10 |

---

## 3. 頁面大綱

外殼沿用 IQL 頁（單欄 680–720px、圖版最大 1080px、桌機左側 sticky 目錄、章節編號 JetBrains Mono、陳述句標題）。視覺記號：一條「從行為策略到隱含 actor」的橫向長條（左 $\mu$、右 $\pi_{imp}$，中間標「× w」），在 hero、S2、S5 重複出現。比喻性動詞一律改字面敘述（例：「AWR 高斯有 17.87% 的機率落在低密度區」，不寫「把質量放在」；「argmax 的平均隨 N 上升」，不寫「推向」）。

| # | id | 標題 | 內容（依序；每個符號在使用前定義） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈IDQL：IQL 是 actor-critic 方法，策略用擴散模型抽取〉 | 副標見下；三張事實卡；「μ → × w → π_imp」長條（長條下一行小字：「μ 是收集資料的行為策略，w 是 critic 算出的權重，π_imp 是 critic 隱含的策略；第 02 章定義」）。連結：arXiv、`../index.html#paper-444`、總覽 `idql-lineage-hub.html`、程式碼、前置教學 `implicit-q-learning.html` | 1 螢幕 |
| S1 | `recap` | 01 · IQL 只用資料中的動作學 Q 與 V | 一句 offline RL 與 OOD 動作；$s,a,r,\gamma,\mathcal D,\mu,Q,V,\hat\theta,\tau$；$Q^\pi$、$V^\pi$ 兩句（步驟 1）；Eq. 1–3；「本章只回顧，推導見 IQL 教學」callout；本文的問題；LO1 答案；符號表。延伸閱讀：`rl-mdp-q-learning-basics.html`、`offline-rl-distribution-shift.html`、`iql-lineage-overview.html`、`expectile-regression-value-learning.html` | 400–500 字 |
| S2 | `theorem` | 02 · 凸且 f′(0) = 0 的損失都對應一個隱含 actor | Eq. 6、7；$f'$ 記號（含論文符號註）；Theorem 4.1；四行證明（每行一句說明）；Eq. 8；隱含假設 callout；三點 callout；LO2、LO3 答案。延伸閱讀：`idql-implicit-policy-and-critic-choice.html` | 450–550 字 |
| S3 | `losses` | 03 · 三種損失給出三種隱含 actor | 三種損失表（1.4 步驟 4；< 600px 改為每個損失一張堆疊卡片）；主文講 expectile 與 exponential（含原文寫法註、$\pi_{awr}$、兩個方向的 KL）；quantile 一段主文（權重與「聚集在分位數附近」）＋兩點〔本頁推導〕收在 `details.more`；玩具 bandit 的設定（先寫再放元件）→ C1；LO4、LO5 答案。延伸閱讀：`in-sample-implicit-value-regularization-eql-ivr.html` | 500–600 字 + C1 |
| S4 | `extract` | 04 · 單峰高斯無法表示多峰的隱含 actor | Fig. 1 文字描述；解耦與 critic 不調整；以 AWR 權重訓練擴散模型為何不行（App. F.5；一句「擴散模型是可以表示多峰分布的生成模型，第 06 章介紹」）；LO7 答案；一句預告「第 05 章末的 C2 用玩具比較兩種抽取」。延伸閱讀：`awr-advantage-weighted-policy-extraction.html`、`gaussian-policy-multimodality-limits.html` | 350–450 字 |
| S5 | `algo` | 05 · IDQL 把 critic 訓練與擴散行為模型分開 | 開頭一句：「$\mu_\phi$ 是用擴散模型學到的行為策略；本章只把它當成可以抽樣的黑盒，訓練與取樣在第 06 章」；Alg. 1–2（≥768px 兩欄：左程式碼、右逐行中文；<768px 每行下接說明）；Eq. 13；自正規化重要性抽樣〔本頁推導〕（N = 1 與 N → ∞ 兩端）；quantile 權重在 Alg. 2 中發散一句；訓練時間；兩種 online 微調做法；C2；LO8 答案。延伸閱讀：`diffusion-behavior-model-policy-extraction.html`、`diffusion-policy-class.html` | 450–550 字 + C2 |
| S6 | `diffusion` | 06 · D4RL 設定中，擴散行為模型以五步去噪產生動作 | 先寫 $\mu$ 與 $\mu_\phi$ 的關係；前向過程、$\beta_t,\alpha_t,\bar\alpha_t$（本頁補充）；Eq. 4、5；程式碼註記；App. F.3 的結論（三個排程公式放 `details.more`）；C4，元件後一段正文寫 R9 的說明（F.3 中 D4RL 以 T = 5 最好；玩具沒有網路誤差所以 T 越大越好；兩者不矛盾）；LO9 前半答案。延伸閱讀：`ddpm-training-and-sampling-recap.html`、`diffusion-bc-ddpm-and-network-design.html` | 500–600 字 + C4 |
| S7 | `outliers` | 07 · 離群樣本使 argmax 對 N 敏感 | Fig. 3 文字描述與 LN_Resnet 結構（靜態 SVG 方塊圖，App. G）；F.4、F.6；Sec. 6.4；C3；LO9 後半與 LO10 答案 | 400–500 字 + 1 張靜態 SVG + C3 |
| S8 | `impl` | 08 · 實作設定與常見錯誤 | App. E 表（1.4 步驟 11）、六條常見錯誤 | 250–350 字 + 表 |
| S9 | `results` | 09 · 每個領域只調一個超參數時，IDQL 的總分最高 | 兩種協定、掃描範圍、-1 基線為重跑、IDQL -A 與 -1 的差別在步數與 N；Table 1 總分表（< 600px 第一欄 sticky）；作者敘述＋實際的 -A → -1 變化小表（IQL antmaze 只降 9.6）；Table 2 一句；Maze2D 一句；critic 損失消融（Fig. 2 的作者敘述 + Sec. 6.3，連回 C1）；表下差異小字；LO6、LO11 答案。延伸閱讀：`idql-evaluation-and-online-finetuning.html`、`offline-rl-benchmarks-and-baselines.html` | 500–600 字 + 2 表 |
| S10 | `finetune` | 10 · Offline 預訓練後可以繼續 online 微調 | 兩種做法；Table 3 總分表（含「IQL 408.2 照原文印出，列和 370.1」與 RLPD／umaze-diverse 註腳）；Fig. 9 一句；App. D.3 設定；限制；LO12 答案 | 250–350 字 + 表 |
| S11 | `history` | 11 · 研究脈絡 | 時間線（6.4）+ 一段文字：AWR → IQL → EQL／IVR（critic 推廣）；DDPM → Diffuser → ARQ／DQL／SfBC（擴散與 RL，DQL 連 `diffusion-ql.html` 與 `diffusion-ql-lineage-hub.html`）→ IDQL | 時間線 + 100–150 字 |
| S12 | `next` | 12 · 延伸閱讀 | 筆記清單（6.4）、arXiv、程式碼 | 100 字 + 連結 |
| — | — | 頁尾 | 引用格式、`#pm-bar` | — |

Hero：
- 副標：「這篇論文證明 IQL 的 critic 隱含一個由行為策略重新加權而成的策略，再用擴散模型產生候選動作、以 critic 的權重從中挑選。」
- 三張事實卡（每張只用卡內已定義的詞）：
  1. 「Offline RL 只能用一份固定的資料集學策略。IQL 只用資料中出現過的動作訓練 critic（估計動作價值的網路），但沒有說明這個 critic 對應哪一個策略。」
  2. 「本文證明：critic 的損失一旦選定，就對應一個策略，它等於收集資料的策略乘上一個由 critic 算出的權重。這個策略可能有多個峰。」
  3. 「IDQL 用擴散模型（逐步去噪的生成模型）學資料中的動作分布，每個狀態抽 N 個候選，選 critic 值最高者。D4RL 上 locomotion 與 antmaze 各只選一個 τ 時，總分 1150.9，IQL 為 997.9。」

390px：所有圖版控制列在圖下方堆疊；圖寬等於容器寬（約 358px），圖上文字 12–13px；橫軸刻度減為 3–4 個；兩欄 Alg. 版面改單欄；表格包 `.tscroll`；Eq. 4、5、8 在 < 600px 用 `aligned` 兩行版本（HTML 中兩種都寫，以 media query 切換；非作用版本 `display:none` 且 `aria-hidden="true"`，螢幕閱讀器只讀一次；selfTest 只計可見公式）。C3 的 9 個 N 按鈕在 < 600px 排成兩列（5 + 4）。

---

## 4. 互動元件規格

### 4.0 共用：玩具 bandit 與規則

- **玩具 bandit（推定，C1–C4 共用）**：單一狀態，一維動作 $a$，$Q(a)=a$（線性獎勵，與 Fig. 1 的 $r=a_1+a_2$ 同為線性；頁面寫「為了把動作與 Q 值畫在同一軸上」）。行為策略 $\mu$ 是三群高斯混合：中心 $m=(1,2,3)$（低、中、高獎勵群，對應 Fig. 2 的三群）、共同標準差 $s=0.15$、權重預設 $(0.3,0.5,0.2)$。權重滑桿只調高獎勵群 $p_h$，其餘兩群維持 3:5：$(\tfrac38(1-p_h),\tfrac58(1-p_h),p_h)$。$\mathbb E_\mu[Q]=1.9$。
- **計算範圍與網格**：$[LO,HI]=[m_1-10s,\ m_3+10s]=[-0.5,4.5]$；密度與動差用 2001 點梯形法（`common.py` 的 `grid`、`moments`）。
- **常態 CDF**：`erfc` 用 Numerical Recipes 的 `erfcc`（Chebyshev，相對誤差 < 1.2e-7，`common.py` 逐行相同）；$\Phi(z)=\tfrac12\mathrm{erfc}(-z/\sqrt2)$。
- **三種 V 的解法**（`common.py`）：
  - expectile：解 $\tau\,\mathbb E[(Q-V)_+]-(1-\tau)(\mathbb E[(Q-V)_+]+V-\mathbb E[Q])=0$，每群 $\mathbb E[(Q-V)_+]=s\varphi(z)+(m-V)(1-\Phi(z))$，$z=(V-m)/s$；
  - quantile：解 $F_\mu(V)=\tau$；
  - 兩者都在 $[LO,HI]$ 上二分 200 次（左端函數值 > 0、右端 < 0，必有根）。
  - exponential：閉式 $V_{exp}=\frac1\alpha\log\sum_k\pi_k\exp(\alpha m_k+\tfrac12\alpha^2s^2)$〔本頁推導：Eq. 11 的和換成對高斯的積分〕，以 log-sum-exp 計算。
- **權重的數值保護**：`w_exp`：$|\alpha u|<10^{-12}$ 時回傳極限 $\alpha^2$，否則 $\alpha\,|\mathrm{expm1}(\alpha u)|/|u|$；密度 $\mu\cdot w$ 在 $\alpha=10$、$a=4.5$ 時約 $10^{-22}\cdot10^{11}$，不溢位。quantile 的權重在 $Q=V$ 發散：只畫形狀，不正規化（C1）。
- **PRNG**：`mulberry32(seed)` 與 Box–Muller（只取 cos 分支），與 IQL 頁相同函式；從 $\mu$ 抽樣的呼叫順序固定為：`u = rnd()` 決定群（依累積權重，`u < c` 的第一群）→ `gauss(rnd)`（兩次 `rnd()`）→ $a=m_k+s\cdot z$。
- **動態文字**（讀數、`aria-label`、`aria-valuetext`、live region、SVG 標籤）一律 Unicode，不呼叫 KaTeX；**不得含組合字元（U+0300–U+036F）**：寫「V*」「π_imp」「θ_targ」。ᾱ（U+1FB1）不在 IBM Plex Sans、Noto Sans TC、JetBrains Mono 中，不使用：讀數與 aria 文字寫「abar_T」，SVG 標籤寫 `<tspan text-decoration='overline'>α</tspan><tspan baseline-shift='sub' font-size='75%'>T</tspan>`。selfTest 的檢查在原始碼中寫成跳脫形式 `/[\u0300-\u036f]/`（不得寫入字面組合字元），只掃描讀數節點、SVG `<text>` 與所有 `aria-label`／`aria-valuetext`。下標用 `<tspan baseline-shift="sub" font-size="75%">`。
- 互動規則沿用 IQL 頁：`input` 只更新狀態並以 rAF 合併重畫；live region 與 SVG `aria-label` 只在 `change` 或停止輸入 500ms 後更新（`announce()`）；所有選取都有滑桿或按鈕，hover 資訊同時出現在固定讀數框（觸控可用）；曲線以線型／標記形狀＋直接文字標籤區分；分段按鈕用 `radiogroup()`。
- **重運算**：C2、C4 的蒙地卡羅以 `setTimeout(0)` 分段（C2 每段 200 次抽取、C4 每段 500 個樣本）。每次啟動把元件的 `runId` 加 1，分段函式開頭比對自己的 token，不是目前的 `runId` 就直接結束（丟棄舊結果）；計算中讀數顯示「計算中…」並停用該元件全部控制項；完成時才寫入快取（key 含全部輸入）並 `announce()` 一次。C1、C3 為確定性計算，同步；C1 的「V 對超參數」曲線以 `(loss, p_h)` 快取，建置時以 `performance.now()` 實測並寫回第 8 節（> 16ms 時改為分段）。

### C1 · 三種 critic 損失的隱含 actor（S3）

- **教的概念**：Theorem 4.1；$\pi_{imp}=\mu\cdot w$；三種 $w$ 的形狀；$\mathbb E_{\pi_{imp}}[Q]=V^*$；exponential 的 $\pi_{awr}$ 與 $\pi_{imp}$ 不同；quantile 的點質量。
- **輸入**：分段按鈕「損失」：expectile（預設）／quantile／exponential。滑桿 τ ∈ [0.5, 0.99]，步長 0.01，預設 0.9（expectile、quantile 顯示）；滑桿 α：$\log_{10}\alpha\in[-1,1]$，步長 0.01，預設 α = 1（exponential 顯示，App. E locomotion 值；`aria-valuetext` 寫「α = 1.00」，不寫對數值）；滑桿 $p_h\in[0.02,0.6]$，步長 0.01，預設 0.2；按鈕「回到預設」。
- **計算**：$V^*$（4.0）；網格上 $\mu(a)$、$w(a)$、$\pi_{imp}(a)=\mu w/Z$；$\mathbb E_{\pi_{imp}}[Q]$、$\mathrm{sd}_{\pi_{imp}}$；exponential 另算 $\pi_{awr}$（三群各自平移 $\alpha s^2$、權重 $\propto\pi_k\exp(\alpha m_k+\tfrac12\alpha^2s^2)$）的平均、$\mathrm{KL}(\mu\Vert\pi_{awr})=\alpha(V_{exp}-\mathbb E_\mu[Q])$（App. C.2）與 $\mathrm{KL}(\pi_{awr}\Vert\mu)=\alpha(\mathbb E_{\pi_{awr}}[Q]-V_{exp})$〔本頁推導〕。副圖「V 對超參數」：目前損失的 $V$ 曲線，τ 或 $\log_{10}\alpha$ 均分 300 點。
- **顯示**（兩圖，<600px 垂直堆疊）：
  1. 主圖：橫軸 $a=Q(a)\in[0,4]$。$\mu$（`--c-beh` 淺填色＋細實線，標「μ：資料中的動作」）；$\pi_{imp}$（`--accent` 粗實線，標「π_imp」）；exponential 時加 $\pi_{awr}$（`--ink-3` 虛線，標「π_awr」）；$V^*$ 垂直線（`--accent` 虛線，標「V* = 2.577」）。下方窄條畫 $w(a)$（`--ink-2` 階梯或曲線，標「w」）。縱軸上限 = 顯示曲線最大值 × 1.1。quantile：$\pi_{imp}$ 不正規化，改畫 $\mu\cdot|\tau-\mathbb 1(a<V)|/\max(|a-V|,h)$ 的形狀，$h=0.02$（動作單位，網格步長 0.0025 的 8 倍，只為避免除以 0 而設，不影響任何讀數）；曲線先除以它在 $|a-V|\ge0.1$ 範圍內的最大值，再乘以縱軸高度的 0.8，超過縱軸頂端的部分裁掉；在 $a=V$、縱軸頂端畫「→ ∞」文字標記（`--accent`，置中於 V 垂直線上方）；圖上文字「積分發散：π_imp 是 V 處的點質量（本頁推導）」。
  2. 副圖：橫軸 τ（或 α，對數），縱軸 $V$；目前值實心點；quantile 曲線即累積分布函數的反函數。
  - 讀數表：$V^*$、$\mathbb E_{\pi_{imp}}[Q]$（標「Theorem 4.1：應等於 V*」）、$\mathrm{sd}_{\pi_{imp}}$、$\mathbb E_\mu[Q]$；expectile 另列「上下權重比 τ/(1−τ)」；quantile 的 $\mathbb E_{\pi_{imp}}$、sd 欄顯示「V*／0（點質量）」；exponential 另列 $\mathbb E_{\pi_{awr}}[Q]$、「KL(μ‖π_awr)」與「KL(π_awr‖μ)」，並列一行「V* = E_awr[Q] − KL(π_awr‖μ)/α」的數值核對。
- **讀者應觀察到**：(1) expectile τ = 0.5 時 $\pi_{imp}=\mu$、$V=1.9$；τ 增大時高群被放大、$V$ 平滑上升；(2) expectile 與 exponential 的 $\mathbb E_{\pi_{imp}}[Q]$ 都與 $V^*$ 相等（到 4 位小數；quantile 為點質量，平均即 $V^*$）；(3) exponential：$\pi_{awr}$ 的平均高於 $V_{exp}$（α = 1：2.3980 對 2.1545），兩者不是同一個分布；差額 0.2435 正是 KL(π_awr‖μ)/α；(4) quantile 在 τ = 0.8 時 $V=2.5191$ 落在中群與高群之間的低密度區，點質量放在資料很少的動作上〔本頁推導，玩具現象〕；(5) 把 $p_h$ 降到 0.05，同樣 τ = 0.9 的 $V$ 降到 2.1248。

### C2 · 抽取方法：單峰高斯 AWR 對重新抽樣（S5 末尾；S4 的文字先說明理由）

- **教的概念**：隱含 actor 多峰時，任何單峰高斯都有相當比例的機率落在群與群之間的低密度區（以動差擬合 $\pi_{imp}$ 的最佳高斯也一樣）；AWR 高斯另外還有權重不一致的問題（它的目標是 $\mu e^{\alpha Q}$，不是 $\pi_{imp}$；Sec. 2 稱 AWR 的 KL 約束與 critic 捕捉的策略不一致）；Alg. 2 的重新抽樣只會選到行為模型的樣本；$N=1$ 時就是 $\mu$，$N$ 大時接近 $\pi_{imp}$；argmax（Eq. 13）的平均隨 $N$ 上升。**簡化**：行為模型直接從真實的 $\mu$ 抽樣（代表完美訓練的擴散模型）；critic 用精確的 $V^*$（expectile）。
- **輸入**：滑桿 AWR 溫度 α ∈ [0, 10]，步長 0.1，預設 3（範圍上限的理由見 R7）；滑桿 τ ∈ [0.5, 0.99]，步長 0.01，預設 0.9；分段按鈕 $N$：1、4、16、64（預設）、256；分段按鈕「抽取」：重新抽樣（Alg. 2，預設）／argmax（Eq. 13）；按鈕「重新抽樣（seed + 1）」「回到 seed 1」，顯示目前 seed。
- **計算**：
  - AWR 高斯（加權最大概似對 $\mu\exp(\alpha Q)$ 的閉式解，`common.py` 的 `awr_gauss`）：傾斜後的混合仍為三群高斯，平均 $m_k+\alpha s^2$、權重 $\propto\pi_k\exp(\alpha m_k+\tfrac12\alpha^2s^2)$；高斯的平均與變異數取其一、二階動差。〔本頁推導〕$V$ 在單一狀態內是常數，不影響 AWR 的結果（與 IQL 頁 C4 同一事實）。
  - 單峰參考線：以動差擬合 $\pi_{imp}$ 的高斯 $\mathcal N(\mathbb E_{\pi_{imp}}[Q],\mathrm{sd}_{\pi_{imp}}^2)$（單峰高斯中與 $\pi_{imp}$ 一、二階動差相同者），它的低密度質量與 TV 只反映「單峰」這個限制。
  - 低密度區：網格上 $\mu(a)<0.01\cdot\max\mu$ 的點（門檻是自選值，推定）（頁面圖上以斜線底紋標「資料密度 < 1% 峰值」）。AWR 高斯在低密度區的質量、與 $\pi_{imp}$ 的總變異距離 $\mathrm{TV}=\tfrac12\int|p-q|$（網格）。
  - 蒙地卡羅（K = 2000 次「抽 N 個候選、選一個」）：每次依 4.0 的順序抽 $N$ 個 $a_i$；算 $w_i=$ Eq. 9；**一定**呼叫一次 `u = rnd()`（argmax 模式也呼叫，使兩模式的亂數串一致）；重新抽樣選累積 $w_i/\sum w$ 首次超過 $u$ 的 $i$（捨入使累積和從未超過 $u$ 時選最後一個），argmax 選 $\max a_i$。統計：平均、標準差、三群比例（以最近中心歸群）、落在低密度區的比例。
- **顯示**：橫軸 $a\in[0,4]$。$\mu$（淺填色）；$\pi_{imp}$（`--accent` 虛線，標「π_imp（目標）」）；AWR 高斯（`--ink` 實線，標「AWR 高斯」）；動差擬合高斯（`--ink-3` 點線，標「最佳單峰高斯」）；抽取結果的直方圖（bin 0.05，`--c-beh` 空心長條，標「IDQL 抽到的動作」）；低密度區斜線底紋。讀數：AWR 平均／標準差／低密度質量／「AWR 目標與 π_imp 的差距（TV）」；最佳單峰高斯的低密度質量／TV；抽取結果的平均／三群比例（並列 $\pi_{imp}$ 的三群比例 0.1156／0.1925／0.6919）／低密度比例。
- **讀者應觀察到**：(1) τ = 0.9：最佳單峰高斯（平均 2.5770、標準差 0.7057）有 21.28% 的機率落在低密度區，TV = 0.4928，$\pi_{imp}$ 本身只有 0.40%，這是單峰的限制；α = 3 的 AWR 高斯（平均 2.9506、標準差 0.3638）低密度 17.87%、TV = 0.4313，還混入了權重不一致；(2) 重新抽樣 N = 1 時抽到的就是 $\mu$（三群 0.315／0.4995／0.1855），N = 256 時接近 $\pi_{imp}$（0.12／0.187／0.693）；(3) argmax 的平均隨 N 上升，N = 64 時平均 3.2472、全部在高群；N = 256 時 15.35% 在高群的分布尾端（μ 密度低於門檻）——這些是 μ 的合法樣本，不是離群樣本。

### C3 · 離群樣本與 N：argmax 的敏感度（S7）

- **教的概念**：Sec. 5 說離群樣本可能被 critic 給出錯誤的高 Q 值；Sec. 6.4 說 MLP 的 N 越大表現越差。本元件示範一個相容的機制〔本頁推導〕。圖版標題下寫：「玩具（推定）：不是 Fig. 4 的重現；離群比例 ε 與離群區間都是自選值。」
- **模型（推定）**：行為模型 = $(1-\varepsilon)\mu+\varepsilon\,\mathrm{U}[3.5,5]$（離群樣本落在資料最大值之上）；critic 對所有動作外插 $\hat Q(a)=a$（因此高估離群樣本）；真實獎勵：正常樣本 $a$，離群樣本 0。
- **計算**（`c3.py`）：$G=(1-\varepsilon)F_\mu+\varepsilon F_U$；最大值的密度 $NG^{N-1}g$；
  $\mathbb E[r]=\int NG^{N-1}\big[(1-\varepsilon)\mu(x)\,x+\varepsilon u(x)\cdot0\big]dx$，$P(\text{選到離群})=\int NG^{N-1}\varepsilon u(x)dx$；網格 $[-0.5,5]$ 6001 點梯形法（$F_\mu,\mu$ 預先算一次）。$N=1$ 時 $NG^{N-1}\equiv1$。
- **輸入**：滑桿 ε：索引滑桿 0–8 對應 $\{0,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1\}$，預設索引 5（ε = 0.01），`aria-valuetext`「ε = 0.01」，旁邊顯示目前 ε；分段按鈕 N：1、2、4、…、256（預設 64）。
- **顯示**：橫軸 $\log_2N$（刻度 1、4、16、64、256），兩條曲線：目前 ε 的 $\mathbb E[r]$（`--accent` 實線＋圓點，標「ε = 0.01」）與 ε = 0 的參考（`--c-beh` 虛線＋方點，標「沒有離群（ε = 0）」）；下方小圖畫 $P(\text{選到離群})$（`--ink-2` 實線）與 $1-(1-\varepsilon)^N$（只畫空心菱形點、不畫線，避免被實線蓋住；標「≈ 1 − (1 − ε)^N」）。目前 N 的垂直細線。讀數：$\mathbb E[r]$、$P$、$1-(1-\varepsilon)^N$、ε = 0 的 $\mathbb E[r]$。
- **讀者應觀察到**：(1) ε = 0：N 越大越好（1.9000 → 3.3382），與 LN_Resnet 在 locomotion 的方向相同；(2) ε = 0.01：N = 8 時最高 2.6880，N = 256 掉到 0.2552；(3) ε 越大，最佳 N 越小（ε = 0.05 時 N = 4 最高 2.1504）。頁面明寫：這只說明「離群樣本 + argmax」會讓大 N 變差，不代表論文的 MLP 有某個 ε。

### C4 · 擴散取樣：排程與步數 T（S6）

- **教的概念**：Eq. 5 從純噪聲逐步去噪；$\bar\alpha_T$ 必須接近 0 才符合 $a_T\sim\mathcal N(0,I)$；VP 排程在任何 $T$ 都使 $\bar\alpha_T=0.0064$，linear 在小 $T$ 時遠離 0；即使噪聲預測完美，$T$ 小時仍有離散化誤差，產生落在群與群之間的樣本。**簡化**：用貝氏最佳的噪聲預測 $\epsilon^*(a_t,t)=-\sqrt{1-\bar\alpha_t}\,\partial_a\log p_t(a_t)$，$p_t=\sum_k\pi_k\mathcal N(\sqrt{\bar\alpha_t}m_k,\ \bar\alpha_ts^2+1-\bar\alpha_t)$〔本頁推導：Eq. 4 的最小解，即完美訓練的網路〕，所以這裡沒有網路誤差，只有取樣誤差。圖版標題下寫：「F.3 中 D4RL 以 T = 5 最好、2D 資料以 T = 50 最好；本玩具沒有網路誤差，只呈現取樣的離散化，結論不能直接套到 D4RL。」同一段說明也寫在元件後的正文段落中（LO9）。
- **輸入**：分段按鈕「排程」VP（預設）／cosine／linear（公式 1.4 步驟 7，標「程式碼」）；分段按鈕 T：2、5（預設）、10、20、50；按鈕「重新抽樣（seed + 1）」「回到 seed 1」。
- **索引（一律 1-based，與 Eq. 5 和 VP 公式的 $2t-1$ 一致）**：`betas(name, T)` 回傳長度 T+1 的陣列 `beta[1..T]`（`beta[0]` 不用）；`alpha[k] = 1 − beta[k]`；`abar[0] = 1`、`abar[k] = abar[k−1]·alpha[k]`，$\bar\alpha_T$ = `abar[T]`。VP：`beta[k] = 1 − exp(−0.1/T − 0.5·9.9·(2k−1)/T²)`；linear：`beta[k] = 1e-4 + (0.02 − 1e-4)(k−1)/(T−1)`；cosine：`f(i) = cos²((i/T + 0.008)/1.008·π/2)`，`beta[k] = clip(1 − f(k)/f(k−1), 0, 0.999)`。
- **計算**（`c4.py` 的 `arrays`、`sample`）：K = 4000；每個樣本 `x = gauss(rnd)`（$a_T$）；`for k = T..1`：`e = epsStar(x, abar[k])`（log-sum-exp 算群權重）、$x\leftarrow(x-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}e)/\sqrt{\alpha_k}$、**每步都抽** `z = gauss(rnd)`，只在 $k>1$ 時加 $\sqrt{\beta_k}z$（程式碼的時間索引 $k-1>0$；每步都抽使亂數串固定）。改寫後的 c4.py 輸出與 round 0 逐位相同。不截斷。統計：平均、標準差、三群比例、低密度比例（同 C2 的門檻）、$\bar\alpha_T$。cosine 的 $\beta$ 截斷到 0.999 保證 $\sqrt{1-\bar\alpha_t}>0$、$\alpha_t>0$。
- **顯示**：上圖 $\beta_k$ 長條（每個 k 一條，T = 50 時只標首尾數值）與 $\bar\alpha_k$ 折線；下圖樣本直方圖（bin 0.05）疊 $\mu$ 密度、低密度區底紋。讀數：$\bar\alpha_T$、樣本平均／標準差（目標 1.9000／0.7159）、三群比例（目標 0.3／0.5／0.2）、低密度比例。
- **讀者應觀察到**：(1) VP、T = 5：三群 0.2948／0.505／0.2003，低密度 7.80%；T = 50 降到 0.85%；(2) linear、T = 5：$\bar\alpha_T=0.9506$，起點不是噪聲，92% 的樣本落在低群附近（平均 0.7533）；(3) T = 2 時任何排程都明顯偏離（VP 中群 0.8728）。

---

## 5. 驗收數值表

腳本與頁面同一算法、雙精度。除另註，容許誤差如下；selfTest 中所有蒙地卡羅斷言統一用 ±0.02（本表的 ±0.01 是腳本與頁面同 PRNG 時的預期差距）：閉式／二分得到的值 ±0.0005；網格動差（sd、TV、低密度質量）±0.002；蒙地卡羅統計（C2、C4，同 PRNG、同呼叫順序；JS 與 Python 的 `Math.cos/log` 末位差異可能翻轉少數抽樣）平均與標準差 ±0.01、比例 ±0.01；$\bar\alpha_T$ ±0.0001。

**C1**（`c1_out.txt`；預設 $p_h=0.2$，$\mathbb E_\mu[Q]=1.9000$，$\mathrm{sd}_\mu=0.7159$）

| 損失 | 超參數 | $V^*$ | $\mathbb E_{\pi_{imp}}[Q]$ | $\mathrm{sd}_{\pi_{imp}}$ | 其他 |
| --- | --- | --- | --- | --- | --- |
| expectile | τ = 0.5 / 0.7 / 0.9 / 0.99 | 1.9000 / 2.1391 / 2.5770 / 3.0069 | 同 $V^*$ | 0.7159 / 0.7479 / 0.7057 / 0.4306 | 比 1 / 2.3333 / 9 / 99 |
| quantile | τ = 0.5 / 0.7 / 0.8 / 0.9 / 0.99 | 1.9620 / 2.1262 / 2.5191 / 3.0000 / 3.2467 | 點質量 | 0 | $F(V)=\tau$（±1e-6） |
| exponential | α = 0.1 / 1 / 2 / 10 | 1.9257 / 2.1545 / 2.3737 / 2.9516 | 同 $V^*$ | 0.7174 / 0.7205 / 0.7032 / 0.5220 | $\mathbb E_{\pi_{awr}}$ 1.9515 / 2.3980 / 2.7571 / 3.2249；KL(μ‖π_awr) 0.0026 / 0.2545 / 0.9473 / 10.5157；KL(π_awr‖μ) 0.0026 / 0.2435 / 0.7668 / 2.7332 |

$p_h=0.05$：expectile τ = 0.9 → 2.1248；quantile τ = 0.9 → 2.2066；exponential α = 1 → 1.8613。

**C2**（`c2_out.txt`；τ = 0.9，$V^*=2.5770$，$\pi_{imp}$ 低密度質量 0.0040，三群 0.1156／0.1925／0.6919；$\mu$ 本身低密度質量 0.0035）

| AWR α | 0 | 1 | 3 | 10 |
| --- | --- | --- | --- | --- |
| 高斯平均 / 標準差 | 1.9000 / 0.7159 | 2.3980 / 0.6695 | 2.9506 / 0.3638 | 3.2249 / 0.1504 |
| 低密度質量 | 0.1501 | 0.1714 | 0.1787 | 0.1139 |
| TV（對 τ = 0.9 的 π_imp） | 0.5696 | 0.5064 | 0.4313 | 0.6267 |

最佳單峰高斯（動差擬合 π_imp）：τ = 0.9 → 平均 2.5770、標準差 0.7057、低密度質量 0.2128、TV 0.4928；τ = 0.7 → 2.1391、0.7479、0.1633、0.4257。

蒙地卡羅（K = 2000、seed 1；欄：平均／標準差／三群比例／低密度比例）：

| N | 重新抽樣 | argmax |
| --- | --- | --- |
| 1 | 1.8676 / 0.7120 / .315 .4995 .1855 / .0025 | 同左 |
| 4 | 2.3104 / 0.7714 / .182 .3275 .4905 / .0030 | 2.6541 / 0.4889 / .0075 .389 .6035 / .0050 |
| 16 | 2.5137 / 0.7318 / .1345 .2195 .646 / .0035 | 3.0920 / 0.1988 / 0 .032 .968 / .0110 |
| 64 | 2.5442 / 0.7353 / .133 .1935 .6735 / .0035 | 3.2472 / 0.0885 / 0 0 1 / .0435 |
| 256 | 2.5756 / 0.7156 / .12 .187 .693 / .0045 | 3.3353 / 0.0683 / 0 0 1 / .1535 |

**C3**（`c3_out.txt`；欄 $\mathbb E[r]$ / $P(\text{離群})$）

| ε \ N | 1 | 4 | 8 | 16 | 64 | 256 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.9000 / 0 | 2.6402 / 0 | 2.9130 / 0 | 3.0952 / 0 | 3.2462 / 0 | 3.3382 / 0 |
| 0.001 | 1.8981 / .0010 | 2.6296 / .0040 | 2.8898 / .0080 | 3.0461 / .0159 | 3.0448 / .0620 | 2.5843 / .2259 |
| 0.01 | 1.8810 / .0100 | 2.5361 / .0394 | 2.6880 / .0773 | 2.6355 / .1485 | 1.7063 / .4744 | 0.2552 / .9236 |
| 0.05 | 1.8050 / .0500 | 2.1504 / .1855 | 1.9326 / .3366 | 1.3623 / .5599 | 0.1219 / .9625 | 0.0000 / 1.0000 |

**C4**（`c4_out.txt`；K = 4000、seed 1；欄：$\bar\alpha_T$／平均／標準差／三群／低密度）

| 排程 | T | 結果 |
| --- | --- | --- |
| VP | 2 | 0.0064 / 1.8919 / 0.3050 / .1022 .8728 .025 / .0960 |
| VP | 5 | 0.0064 / 1.9063 / 0.6278 / .2948 .505 .2003 / .0780 |
| VP | 50 | 0.0064 / 1.8911 / 0.7164 / .3078 .4905 .2018 / .0085 |
| cosine | 5 | 0.0001 / 1.9160 / 0.6811 / .3018 .4778 .2205 / .0510 |
| cosine | 50 | 0.0000 / 1.9022 / 0.7182 / .2995 .4968 .2037 / .0100 |
| linear | 5 | 0.9506 / 0.7533 / 0.4770 / .9197 .0755 .0047 / .3583 |
| linear | 50 | 0.6030 / 1.2440 / 0.5430 / .738 .233 .029 / .0100 |

（其餘 T 見 `c4_out.txt`，頁面全部可選。）

**論文表格核對**（`tables_out.txt`，已改為 0.05 容許誤差的比較）：見 1.4 步驟 10；-A → -1 變化：antmaze CQL −123.6／IQL −9.6／DQL −202.1／IDQL −35.3，total −147.8／−72.5／−252.3／−62.9。

---

## 6. 實作計畫

### 6.1 檔案與共用外殼

- 單一檔案 `tut/444/build/index.html`，CSS／JS／時間線資料全部內嵌，不 `fetch` 本機檔案；上線時複製為 `tutorials/idql-diffusion-actor-critic.html`。
- **直接複製** `tut/569/build/index.html` 的外殼，不重寫：CSS tokens（含深色模式三段覆寫、`pm-theme`、`--wall`）、字型、appbar、`#pm-bar`、章節目錄、`.plate` 圖版、callout、`details.check`（檢查題）與 `details.more`（其他可收合）、「本頁推導」「推定」pill、「本章符號」小表、`.tscroll`、KaTeX 0.16.9 設定與 `renderMath()`、可橫捲公式（tabindex、aria-label）與窄螢幕 `aligned` 切換、`?test` selfTest（`ok()` 計數、`#selftest`、`scrollWidth <= innerWidth`）、`renderTimeline()`、SVG 輔助（`el`、`scaleLin`、`scaleLog`、`linePath`）、`announce()`、`radiogroup()`、`schedule()`、`ResizeObserver` 重畫、Hero 軸的 HTML 元件與標籤分層演算法（改成 μ → π_imp 長條）、`mulberry32`、`gauss`。
- 以下只列新增或變更。

### 6.2 新增 tokens、pill 與連結

- 沿用 `--c-beh`（行為策略 μ、資料）；新增 pill「程式碼」（樣式同「推定」，文字不同）。圖表色不新增。新元素的深色模式：低密度區斜線底紋以 SVG `<pattern>` 畫，線色 `var(--line)`、底色透明；C4 的 β 長條用 `--c-beh`、$\bar\alpha$ 折線用 `--ink-2`；全部經 tokens，兩種主題自動切換。本頁沒有動畫與 transition（外殼的 transition 在 reduced-motion 下已取消）。
- 站內連結：Hero「在關聯圖中查看」→ `../index.html#paper-444`；arXiv `https://arxiv.org/abs/2304.10573`；程式碼 `https://github.com/philippe-eecs/IDQL`；前置教學 `implicit-q-learning.html`；同批教學 `diffusion-ql.html`（slug `diffusion-ql`，協調者已確認；S4 一句「DQL 把擴散模型當 actor 並加 Q 最大化項，見 Diffusion-QL 教學」）；筆記 → `<slug>.html`。

### 6.3 程式結構（單一 IIFE）

- 共用：`erfc`、`Phi`、`phi`、`MIX = {m:[1,2,3], s:0.15}`、`pis(pH)`、`mixPdf`、`mixCdf`、`upperPM`、`bisect`、`vExpectile(tau, pi)`、`vQuantile(tau, pi)`、`vExp(alpha, pi)`、`wExpectile`、`wExp`、`GRID`（2001 點）、`moments(dens)`、`awrGauss(alpha, pi)`、`sampleMu(rnd, pi)`、`isLow(a, pi)`。逐式對應 `common.py`。
- C1：`initC1()`，`state={loss, tau, logAlpha, pH}`，`c1Values(state)` → `{V, eImp, sdImp, eAwr, kl, dens:{mu, imp, awr, w}}`。
- C2：`initC2()`，`state={alpha, tau, N, mode, seed}`；`c2Analytic(alpha, tau)`；`c2MC(tau, N, mode, seed, K, onDone)`（分段）；`cache=new Map()`（key `${tau}|${N}|${mode}|${seed}`）。
- C3：`initC3()`，`state={eps, N}`；`c3Curve(eps)` → 9 個 N 的 `{er, pOut}`；預先算 `F`、`f`。
- C4：`initC4()`，`state={sched, T, seed}`；`betas(name, T)`、`epsStar(x, abar, pi)`、`c4MC(...)`（分段）。
- `selfTest()` 新增斷言（容許誤差見第 5 節）：
  - `erfc` 對 7 個點與高精度值相差 < 2e-7（值寫死在測試中：取自 Python `math.erfc`）。
  - `vExpectile(0.9, pis(0.2))` = 2.5770；`vQuantile(0.8, ·)` = 2.5191；`vExp(1, ·)` = 2.1545；τ = 0.5 時 expectile = 1.9000。
  - Theorem 4.1：expectile τ ∈ {0.7, 0.9}、exponential α ∈ {0.1, 1, 10} 的 $|\mathbb E_{\pi_{imp}}[Q]-V^*|<10^{-3}$。
  - `awrGauss(3, ·)` 平均 2.9506、標準差 0.3638；TV(α = 3) = 0.4313。
  - `c2MC(0.9, 64, 'imp', 1, 2000)` 平均 2.5442；`'max'` 平均 3.2472（±0.02）；動差擬合高斯的 TV = 0.4928。
  - 分段取消：連續啟動兩次 `c2MC`，第一次的 onDone 不得被呼叫。
  - `c3Curve(0.01)`：N = 8 → 2.6880、N = 256 → 0.2552。
  - `betas('vp', 5)` 的 `abar[5]` = 0.0064、`betas('vp',5)[1]` = 0.1959（$1-e^{-0.218}$）；`betas('linear', 5)` → `abar[5]` = 0.9506；C4 VP、T = 5、seed 1 的三群比例 .2948／.505／.2003（±0.02）。
  - 頁面文字與 SVG `<text>` 沒有 U+0300–U+036F。
  - `mulberry32(1)` 前兩個輸出 = 0.6270739405881613、0.002735721180215478。

### 6.4 研究筆記與時間線

- 來源 `tut/444/notes/timeline.json`（14 筆；欄位 `year`、`title`、`authors_short`、`arxiv`、`role`、`source`），原樣貼進 `<script type="application/json" id="timeline-data">`；排序 `Number(year)`，同年保持檔案順序。
- `TL_GROUP`（以 arXiv ID 為 key；不在表中歸「其他」並在建置回報列出）：

  | 類別 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | 擴散模型 | 1503.03585、2006.11239 | `diffusion-bc-ddpm-and-network-design.html` |
  | 策略抽取 | 1910.00177 | `awr-advantage-weighted-policy-extraction.html` |
  | IQL 與其推廣 | 2110.06169 | `idql-implicit-policy-and-critic-choice.html` |
  | soft value 與指數型 critic | 1801.01290、2301.02328、2303.15810 | `in-sample-implicit-value-regularization-eql-ivr.html` |
  | 擴散 × RL | 2206.00695、2209.14548 | `diffusion-behavior-model-policy-extraction.html` |
  | 軌跡擴散 | 2205.09991 | `trajectory-diffusion-planning.html` |
  | 擴散 × RL（DQL） | 2208.06193 | `diffusion-ql-lineage-hub.html`（卡片另加「教學」連結 `diffusion-ql.html`） |
  | 基線 | 2106.06860、2302.02948 | `idql-evaluation-and-online-finetuning.html` |
  | 本文 | 2304.10573 | `idql-lineage-hub.html` |

- S12 延伸閱讀（依總覽 `idql-lineage-hub` 的閱讀順序）：`idql-lineage-hub`、`rl-mdp-q-learning-basics`、`offline-rl-distribution-shift`、`iql-lineage-overview`、`expectile-regression-value-learning`、`awr-advantage-weighted-policy-extraction`、`idql-implicit-policy-and-critic-choice`、`in-sample-implicit-value-regularization-eql-ivr`、`diffusion-policy-class`、`diffusion-behavior-model-policy-extraction`、`diffusion-bc-ddpm-and-network-design`、`idql-evaluation-and-online-finetuning`；另列 `diffusion-ql-lineage-hub`（Diffusion-QL 總覽）、`gaussian-policy-multimodality-limits`、`ddpm-training-and-sampling-recap`、`trajectory-diffusion-planning`，以及教學 `diffusion-ql.html`、`implicit-q-learning.html`。每項附 frontmatter 的 `title`（建置時從檔案讀取）。

### 6.5 文案與排版（新增部分）

- 用詞：「隱含 actor（implicit actor）」「行為策略（behavior policy）」「重新加權」「重新抽樣（resampling）」「候選動作」「離群樣本（outlier）」「排程（schedule）」「常見錯誤」。第一次出現附英文。
- 三種陳述分開：論文結果（附出處）、「作者認為」、〔本頁推導〕；玩具數值加「推定」，程式碼內容加「程式碼」。
- 第 1 節以外不得新增論文數字；頁面數字須與第 1、5 節一致。

### 6.6 建置後自我檢查

沿用 IQL 頁流程：390／768／1024／1440 截圖（新元件與 390px 各一輪）＋深色、鍵盤 focus、reduced-motion；`?test` 全數通過；impeccable 0 項；`extract_page_text.py` 確認 12 條 LO 參考答案的關鍵詞與數字出現在正文。截圖用本機 KaTeX（`scratchpad/katex/0.16.9`，headless Chrome 連不到 CDN），上線檔仍指向 cdnjs。C2、C4 的分段計算以 `performance.now()` 實測單段耗時寫回第 8 節（目標每段 < 16ms）。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 玩具不是論文的實驗 | 論文的 Fig. 1–3 bandit 與 2D 資料沒有給參數；C1–C4 全部自選。 | 每個圖版標「玩具（推定）：不是論文的實驗」；頁面只聲稱「示範論文敘述的機制」，不說「重現」。 |
| R2 | quantile 與 Theorem 4.1 | quantile 的 $f$ 在 0 不可微，不滿足 $f'(0)=0$；連續分布下 $\pi_{imp}$ 不可正規化（點質量）。兩點都是〔本頁推導〕，論文沒寫。 | 標「本頁推導」；與作者「聚集在分位數附近」並列，不說論文錯。 |
| R3 | 原文筆誤與不一致 | Eq. 3 寫 minimizing（應最大化）；exponential 的 $f$ 在文字（$\alpha e^u-\alpha u$）與 Eq. 11–12、App. C.1（$e^{\alpha u}-\alpha u$）不一致；App. C.2 最後一行期望下標寫成 $\mathcal D$；Fig. 2 段落寫 quantile 集中在「mean」，Sec. 4.1 寫「quantile」；$f'$ 定義為 $\partial f/\partial V$ 但證明用對引數的導數；Eq. 4 範數未平方；$\alpha_t,\hat\alpha_t$ 未定義；App. E「(Figure 3)」應指 Table 3。 | 頁面寫一致的版本並以小字註明原文寫法；拼字不提。 |
| R4 | 表格內部不一致 | Table 3 IQL offline 總分 408.2 但列和 370.1；Table 1 與 Table 4 的 IDQL antmaze-med-play（84.5／84.2）；IQL-A locomotion、CQL-1 antmaze、IQL-1 antmaze、DQL-1 total 差 0.2–1.0（`tables_out.txt`）。 | 照原文印出，表下小字列出；不推測原因。 |
| R5 | $\pi_{awr}$ 與 $\pi_{imp}$（exponential） | 論文把 $V_{exp}$ 連到 AWR，又用 Theorem 4.1 給出 $w_{exp}$；兩者是不同分布（C1 數值）。 | C1 並列兩者並標「本頁推導」；不改寫作者「AWR 抽取時這個 critic 是正確的」一句。 |
| R6 | 程式碼與論文文字的差異 | 取樣最後一步不加噪、截斷 [−1, 1]；排程公式。來源 `jaxrl5/networks/diffusion.py`（已讀取）。 | 頁面標「程式碼」；C4 照程式碼做、不截斷（玩具動作不在 [−1, 1]）。 |
| R7 | C2 的 α 上限 | 連續 $\mu$ 下 $\mu e^{\alpha a}$ 會把每群平移 $\alpha s^2$；α = 30 時平移 0.675，傾斜分布本身離開資料（低密度質量 0.96），混淆「單峰」這個重點。真實 AWR 只對資料點加權，不會平移。 | α 上限 10（平移 ≤ 0.225）；頁面一句說明「玩具用連續分布計算加權最大概似」。 |
| R8 | C3 的機制是推測 | 論文說離群點可能得到錯誤的高 Q（Sec. 5）、MLP 的 N 越大越差（Sec. 6.4），沒有說兩者的因果量化。 | C3 標「本頁推導、推定」，明寫不代表 MLP 有某個 ε。 |
| R9 | C4 與 F.3 的方向 | 玩具中 T 越大越好（沒有網路誤差），論文 D4RL 以 T = 5 最好、增加 T 小幅變差。 | 圖版標題下明寫差異與原因；不把玩具結論套到 D4RL。 |
| R10 | 同批教學與跨教學筆記 | 教學 slug `diffusion-ql` 已由協調者確認；`diffusion-ql-lineage-hub`、`diffusion-policy-class`、`gaussian-policy-multimodality-limits` 在 `tut/443/notes/`，與本頁同批上架。444 的 `diffusion-behavior-model-policy-extraction` 與 443 的 `diffusion-in-rl-concurrent-work` 範圍不同（前者 SfBC／ARQ／EMaQ／IDQL，後者 Diffuser／DT），不重複。 | 建置時確認 443 筆記仍以這些 slug 存在。 |
| R11 | 符號重名 | 論文的 $\mu$ 同時指行為策略與擴散網路 $\mu_\phi$（Eq. 4–5 中網路輸出的是噪聲）；$\alpha$ 同時是 AWR 溫度、指數損失溫度、擴散係數。 | 符號表與 S6 開頭明寫；擴散係數一律帶下標。 |
| R13 | 作者的 -1 敘述與表格 | 作者寫先前方法在 -1 下「下降得多，antmaze 尤其明顯」，但 IQL 的 antmaze 只降 9.6（IDQL 降 35.3）；IDQL 的 -A 與 -1 τ 相同，差在步數與 N。 | S9 照錄作者敘述並並列實際變化；標題只寫「總分最高」。 |
| R12 | 相對連結 | 建置位置與上線位置不同，預覽時站內連結失效。 | 接受；預覽只檢查外部連結與頁內錨點。 |

---

## 8. 修訂紀錄

### round 1（對應 `reviews/round1.md` 29 項）

| # | 變更 |
| --- | --- |
| 1–2 | C1 quantile 的 h = 0.02 與「→ ∞」位置；C4 全面改 1-based（`beta[1..T]`、`abar[0..T]`、k = T..1、k > 1 加噪），c4.py 改寫後輸出逐位相同 |
| 3、6 | S9 標題改「總分最高」；步驟 10 加 -A → -1 變化表、-1 基線為重跑（App. D.2）、IDQL -A／-1 差在步數與 N；Hero 卡 3 改「各只選一個 τ」；R13 |
| 4 | c2.py 加動差擬合 π_imp 的單峰高斯；C2 顯示並改寫觀察 (1) |
| 5 | tables.py 改容許誤差比較；DQL-1 的 total（957.4 vs 957.6）確實不一致，保留並改寫描述 |
| 7–9 | f′ 定義與論文符號註；Theorem 4.1 隱含假設 callout；KL 方向、〔本頁推導〕V_exp = E_awr[Q] − KL(π_awr‖μ)/α（c1.py 核對） |
| 10–14 | argmax 尾端樣本非離群；N 只寫兩端；quantile 在 Alg. 2 發散；每步截斷；Table 3「remains competitive」與註腳；Eq. 8 之後；R3 改寫 IQL 比較 |
| 17–20 | S5 μ_φ 黑盒句、Alg. 1 φ 列；C2 移到 S5 末尾、Fig. 2／Sec. 6.3 與 LO6 移到 S9；S3、S6 細節收進 `details.more`；S1 加 Q^π、V^π |
| 21–24 | 雙版公式 aria-hidden；損失表堆疊卡片、Table 1 sticky 首欄；ᾱ 改 abar_T／overline tspan；regex 跳脫、只掃動態節點；MC run token 取消與完成時 announce；C4 的 T 說明寫進正文，S6 標題限定 D4RL，LO9 加限定 |
| 25–29 | C3 ε 索引滑桿、N 兩列、公式只畫點；C1 α 的 aria-valuetext；S2 標題加條件；字面動詞；C1 曲線快取；MC 斷言 ±0.02；新元素深色模式；diffusion-ql 與其 hub 連結；筆記不重複的確認 |

### readiness（NO-GO → 修正）

- TL_GROUP：2205.09991 → `trajectory-diffusion-planning`；1801.01290 與 2301.02328、2303.15810 → `in-sample-implicit-value-regularization-eql-ivr`（該筆記 arxiv 欄含三者）。
- C2 τ 滑桿步長 0.01；Eq. 7 加往 Eq. 8 的指引；定義 $L_V^f$；selfTest MC 容許誤差統一 ±0.02。
- `#paper-444` 由關聯圖 JS 產生；`diffusion-ql.html` 同批上架（協調者確認）。

### build 紀錄

- 外殼取自 `tut/569/build/index.html`（CSS 第 1–376 行、JS 輔助函式），內容與元件為新寫；組裝腳本在 scratchpad `b444/assemble.py`（不屬於產出）。
- 實作差異：hero 與 S2 的「μ × w = π_imp」以三格小圖（同一玩具、τ = 0.9）取代 HTML 長條，S5 不重複；C1 的 w 另畫一條窄圖；C2 低密度區以 SVG pattern 斜線；圖例的低密度樣本改為虛線框（impeccable 標記 repeating-gradient）。
- 驗證：`?test` 35 passed、0 failed（1440 與 390 寬）；KaTeX 249 個節點、0 個 katex-error；無 console 錯誤；impeccable 0 項（12 項 numbered-section-labels advisory）；12 條 LO 參考答案的關鍵詞與數字都出現在 `accept/page.txt`。截圖在 `build/caps/`。
