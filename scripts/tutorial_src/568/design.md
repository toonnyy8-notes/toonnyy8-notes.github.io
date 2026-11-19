# 設計文件：〈Conservative Q-Learning for Offline Reinforcement Learning〉互動教學

- 論文：Aviral Kumar、Aurick Zhou、George Tucker、Sergey Levine，arXiv 2006.04779（v3，2020-08-19）。資料庫編號 No. 568。
- 原文：`tut/568/src/paper.txt`（`tut/` = `scripts/tutorial_src/`）。本文件中的「Eq. n」「Sec. n」「App. X」「Thm. n」「Table n」「Fig. n」「Alg. 1」皆指論文原文編號。
- 產出：`tut/568/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/conservative-q-learning.html`（slug `conservative-q-learning`），所有站內相對連結以上線位置為準。
- 顧問筆記：`tut/568/advisor.md`（本文件已逐條對照原文核對；顧問的推導在此改標〔本頁推導〕）。
- 數值驗證：`tut/568/verify/check.py`（純 Python 標準函式庫，與頁面 JS 同算法），輸出 `verify/check_out.txt`。第 5 節驗收表全部取自該輸出。
- 研究筆記：`tut/568/notes/`（總覽 `cql-lineage-hub` + 6 篇主題筆記 + `timeline.json`）。
- 標記慣例：〔本頁推導〕＝論文沒有寫出、由本設計推得；「推定」＝為了教學自訂或校準、論文沒有給的數值；「作者認為」＝論文作者的解釋而非證明。頁面沿用同樣標籤（6.2）。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

離線強化學習（offline RL）只能用一份事先收集好的資料集 $\mathcal D$ 學策略，不能再與環境互動。把一般的 off-policy 演算法（Q-learning、actor-critic）直接用在離線資料上，Bellman backup 會查詢資料中沒有出現的動作（out-of-distribution，OOD）的 Q 值；策略又被訓練成最大化 Q，於是偏向 Q 值被錯誤高估的 OOD 動作（Sec. 2）。本文提出 **Conservative Q-Learning（CQL）**：在標準 Bellman 誤差上加一個 Q 值正則項，壓低某個分佈 $\mu$ 下的 Q 值、抬高資料分佈 $\hat\pi_\beta$ 下的 Q 值（Eq. 2）。在表格式設定下（每個 $(s,a)$ 的 Q 值是獨立的數字，沒有函數近似），且 α 夠大（無取樣誤差時任何 α > 0）、$\hat\pi_\beta>0$，作者證明只加「壓低」項的 Eq. 1 得到逐點下界（Thm. 3.1），加上「抬高」項的 Eq. 2 只保證策略的期望值 $\hat V^\pi\le V^\pi$，但下界更緊（Thm. 3.2）。把 $\mu$ 換成「最大化目前 Q 的分佈」得到 CQL(R) 家族（Eq. 3），熵正則化的特例 CQL(H) 的正則項是 log-sum-exp（Eq. 4）。理論上 CQL 的更新會擴大資料內與 OOD 動作的 Q 值差距（Thm. 3.4），等價於最佳化一個帶懲罰的經驗回報（Thm. 3.5），並對行為策略給出 ζ-安全改進（Thm. 3.6）。實作只需在 SAC 或 QR-DQN 上加約 20 行程式碼（Sec. 1、Sec. 4）。摘要與 Sec. 7 稱 CQL 在許多任務上的最終回報是先前方法的 2–5 倍，「especially when learning from complex and multi-modal data distributions」；Sec. 6 的 Gym 段落則寫混合資料集上「sometimes as much as 2-3x」，單一策略資料集上只是「by a small margin」。

### 1.2 研究動機（Sec. 1–2）

1. 離線資料量遠大於線上能收集的量，但直接把 value-based off-policy 方法用於離線資料，會因為從 OOD 動作 bootstrapping 與過擬合而表現差，症狀是過度樂觀的價值估計（Sec. 1）。
2. 既有方法（BCQ、BEAR、BRAC 等）限制學到的策略靠近行為策略，多數需要另外估計 $\pi_\beta$，因此受限於估計品質；資料來自多個來源時 $\pi_\beta$ 特別複雜（Sec. 5）。
3. 作者的出發點：若能學到保守（下界）的價值估計，就能直接處理高估。又因為策略評估與改進只用到策略下的期望值，只需要期望值的下界，不需要逐點下界，後者會帶來多餘的低估（Sec. 1）。

### 1.3 貢獻（Sec. 1、3、4、6）

1. CQL 框架：Eq. 1、Eq. 2 的保守策略評估，及 min-max 的 CQL(R) 家族（Eq. 3）與兩個實例 CQL(H)（Eq. 4）、CQL(ρ)（Eq. 7）。
2. 理論：逐點下界（Thm. 3.1）、期望值下界且較緊（Thm. 3.2）、策略慢變時仍是下界（Thm. 3.3）、gap-expanding（Thm. 3.4）、懲罰化目標（Thm. 3.5）、安全策略改進（Thm. 3.6）；線性與 NTK 函數近似的延伸（Thm. D.1、D.2）；抬高項只能用 $\pi_\beta$ 的必要性（Thm. D.3）。
3. 實作：SAC（連續控制）與 QR-DQN（離散控制）上的修改、α 的 Lagrange 自動調整（Eq. 30）、連續動作下 log-sum-exp 的重要性取樣估計（App. F）。
4. 實驗：D4RL Gym（Table 1）、AntMaze／Adroit／Kitchen（Table 2）、離線 Atari（Fig. 1、Table 3）、下界的實證檢查（Table 4）、消融（Table 5–7）。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S1 末尾以可收合表格提供；各章開頭的「本章符號」只列該章新增者）

| 符號 | 意義 | 首次出現 |
| --- | --- | --- |
| $s,a,r(s,a),T(s'\mid s,a),\gamma$ | 狀態、動作、獎勵、轉移機率、折扣因子 $\gamma\in(0,1)$；$\lvert r\rvert\le R_{\max}$ | S1 |
| $\pi(a\mid s)$ | 要評估或要學的策略 | S1 |
| $Q^\pi,V^\pi$ | 真實的動作價值與狀態價值，$V^\pi(s)=\mathbb E_{a\sim\pi}[Q^\pi(s,a)]$ | S1 |
| $\mathcal B^*$ | Bellman 最佳化算子 $\mathcal B^*Q(s,a)=r(s,a)+\gamma\mathbb E_{s'\sim T}[\max_{a'}Q(s',a')]$（Sec. 2），Q-learning 用它 | S1 |
| $\mathcal B^\pi$、$P^\pi$ | Bellman 算子 $\mathcal B^\pi Q=r+\gamma P^\pi Q$；$P^\pi Q(s,a)=\mathbb E_{s'\sim T,a'\sim\pi}[Q(s',a')]$ | S1 |
| $\mathcal D$、$\pi_\beta$、$\hat\pi_\beta$ | 資料集、收集資料的行為策略、資料中的經驗行為策略（每個狀態的動作比例） | S2 |
| $\lvert\mathcal D(s,a)\rvert$、$\lvert\mathcal D(s)\rvert$ | 資料中 $(s,a)$ 與 $s$ 的出現次數 | S7 |
| $\hat{\mathcal B}^\pi$ | 只用資料樣本算的經驗 Bellman 算子 | S2 |
| 表格式（tabular）、不動點（fixed point） | 每個 $(s,a)$ 一個獨立數值、沒有函數近似；$x=f(x)$ 的解，反覆套用 $f$ 收斂到的值 | S1 |
| $\hat Q^k$、$\hat Q^\pi$、$\hat V^\pi$ | CQL 第 $k$ 次迭代的 Q、迭代的不動點、$\hat V^\pi(s)=\mathbb E_\pi[\hat Q^\pi(s,a)]$ | S3 |
| $\mu(a\mid s)$ | 被「壓低 Q 值」的動作分佈 | S3 |
| $\alpha$ | 權衡係數（tradeoff factor） | S3 |
| $D_{\mathrm{CQL}}(\pi,\pi_\beta)(s)$ | $\sum_a\pi(a\mid s)\big(\pi(a\mid s)/\pi_\beta(a\mid s)-1\big)$ | S4 |
| $C_{r,T,\delta}$、$\delta$ | 取樣誤差的集中不等式常數、失敗機率 | S7 |
| $\mathcal R(\mu)$、$\mathcal H(\mu)$、$D_{\mathrm{KL}}$、softmax、Lagrangian | CQL(R) 的正則項；熵 $-\sum\mu\log\mu$；KL 散度 $\sum\mu\log(\mu/\rho)$；$\mathrm{softmax}(Q)_a=e^{Q_a}/\sum_b e^{Q_b}$；把限制式乘上乘子加進目標的函數 | S8 |
| $\rho(a\mid s)$ | 只指 KL 正則的先驗分佈（本頁不再用 ρ 表示其他量；Eq. 5 的狀態分佈寫成 $\rho_0(s)$，論文原文寫 $\rho(s)$） | S8 |
| $D_{\mathrm{TV}}(p,q)$ | 總變異距離 $\tfrac12\sum_a\lvert p(a)-q(a)\rvert$ | S10 |
| $d^\pi_M(s)$ | 策略 π 在 MDP $M$ 中的正規化折扣狀態分佈 $(1-\gamma)\sum_t\gamma^t\Pr(s_t=s)$；$d^{\pi_\beta}$ 為資料的狀態分佈 | S2（$d^{\pi_\beta}$）、S12（一般 $d^\pi_{\hat M}$） |
| $\hat M$、$J(\pi,M)$ | 由資料轉移構成的經驗 MDP（轉移與獎勵都用資料中的頻率與平均）、策略在 MDP $M$ 中的折扣回報 | S12 |
| 重要性取樣（importance sampling） | 用分佈 $q$ 的樣本估計 $p$ 下的期望：$\mathbb E_p[f]=\mathbb E_q[f\,p/q]$ | S13 |
| $\tau$ | Lagrange 版本的 Q 值差距門檻（Eq. 30） | S9（C4 τ 模式）、S13 |

#### 步驟 1：問題設定與分佈偏移（Sec. 2）

- 資料 $\mathcal D$ 取樣自 $d^{\pi_\beta}(s)\pi_\beta(a\mid s)$，$\hat\pi_\beta(a\mid s):=\sum_{s,a\in\mathcal D}\mathbf 1[s=s,a=a]/\sum_{s\in\mathcal D}\mathbf 1[s=s]$。
- actor-critic 交替做（Sec. 2 兩式）：策略評估 $\hat Q^{k+1}\leftarrow\arg\min_Q\mathbb E_{s,a,s'\sim\mathcal D}\big[\big((r+\gamma\mathbb E_{a'\sim\hat\pi^k}[\hat Q^k(s',a')])-Q(s,a)\big)^2\big]$；策略改進 $\hat\pi^{k+1}\leftarrow\arg\max_\pi\mathbb E_{s\sim\mathcal D,a\sim\pi^k}[\hat Q^{k+1}(s,a)]$。
- **action distribution shift**：目標值中的 $a'$ 取自學到的策略，但 Q 只在 $\pi_\beta$ 的動作上訓練過；策略最大化 Q，會偏向 Q 值錯誤偏高的 OOD 動作。線上 RL 可以實際試做該動作來修正，離線不行。
- 論文明寫：Q 的訓練**沒有** state distribution shift（backup 不查詢資料外的狀態），但策略在測試時可能有。
- 下一步的理由：既有方法限制策略；作者改為直接讓 Q 值保守。

#### 步驟 2：只加「壓低」項，得到逐點下界（Sec. 3.1，Eq. 1、Eq. 11、Thm. 3.1）

$$\hat Q^{k+1}\leftarrow\arg\min_Q\ \alpha\,\mathbb E_{s\sim\mathcal D,a\sim\mu(a\mid s)}[Q(s,a)]+\tfrac12\mathbb E_{s,a\sim\mathcal D}\big[(Q(s,a)-\hat{\mathcal B}^\pi\hat Q^k(s,a))^2\big]\tag{Eq. 1}$$

- $\mu(s,a)=d^{\pi_\beta}(s)\mu(a\mid s)$：狀態邊際與資料相同，因為 Q 的訓練不查詢未見狀態、只查詢未見動作。
- 表格式、每輪精確最小化時，對單一格 $Q(s,a)$ 微分設 0〔本頁推導：論文只寫「令導數為 0」〕：目標中含 $Q(s,a)$ 的部分是 $\alpha d(s)\mu(a\mid s)Q(s,a)+\tfrac12 d(s)\hat\pi_\beta(a\mid s)(Q(s,a)-\hat{\mathcal B}^\pi\hat Q^k(s,a))^2$，導數 $\alpha\mu+\hat\pi_\beta(Q-\hat{\mathcal B}^\pi\hat Q^k)=0$，得

$$\hat Q^{k+1}(s,a)=\hat{\mathcal B}^\pi\hat Q^k(s,a)-\alpha\frac{\mu(a\mid s)}{\hat\pi_\beta(a\mid s)}\tag{Eq. 11}$$

- $\hat\pi_\beta$ 在分母，是因為 Bellman 誤差以資料密度加權、懲罰以 $\mu$ 加權：資料越少的動作，同樣的 α 壓得越低〔本頁推導〕。若 $\mu$ 在 $\hat\pi_\beta=0$ 的動作上有質量，該格只有 $\alpha\mu Q$ 一項、最小值為 $-\infty$，所以 Thm. 3.1 要求 $\mathrm{supp}\,\mu\subset\mathrm{supp}\,\hat\pi_\beta$〔本頁推導〕；推導另假設 $\hat\pi_\beta(a\mid s)>0$ 對所有 $a$（Sec. 3.1）。
- 不動點：$\mathcal B^\pi Q=r+\gamma P^\pi Q$，所以 $\hat Q^\pi=(I-\gamma P^\pi)^{-1}[r-\alpha\mu/\hat\pi_\beta]$（加上取樣誤差見步驟 5）。關鍵事實：$(I-\gamma P^\pi)^{-1}=\sum_t\gamma^t(P^\pi)^t$ 的每個元素非負，每一步少一點，不動點就少一點（App. C）。
- **Thm. 3.1**：對任何 $\mathrm{supp}\,\mu\subset\mathrm{supp}\,\hat\pi_\beta$，機率 $\ge1-\delta$，$\forall s\in\mathcal D,a$：
  $$\hat Q^\pi(s,a)\le Q^\pi(s,a)-\alpha\Big[(I-\gamma P^\pi)^{-1}\frac{\mu}{\hat\pi_\beta}\Big](s,a)+\Big[(I-\gamma P^\pi)^{-1}\frac{C_{r,T,\delta}R_{\max}}{(1-\gamma)\sqrt{\lvert\mathcal D\rvert}}\Big](s,a).$$
  α 夠大時逐點下界成立；$\hat{\mathcal B}^\pi=\mathcal B^\pi$ 時任何 $\alpha>0$ 都成立（主文；App. C 寫「$\alpha\ge0$」，見 R3）。注意：App. C 的 α 門檻除以 $\min_{s,a}\mu/\hat\pi_\beta$ 並寫「since μ(a|s) > 0」，所以有取樣誤差時需 μ 在每個動作上 > 0；μ 有零的格（例如確定性 μ）任何 α 都抵銷不了該格的取樣誤差，只在無取樣誤差時成立〔本頁推導：門檻式的分母為 0〕。頁面 S3 只陳述 $\hat{\mathcal B}^\pi=\mathcal B^\pi$ 的版本（不出現 $C_{r,T,\delta}$），完整式在 S7。

#### 步驟 3：加上「抬高」項，下界變緊但只對期望值成立（Sec. 3.1，Eq. 2、Eq. 13、Eq. 14、Thm. 3.2）

$$\hat Q^{k+1}\leftarrow\arg\min_Q\ \alpha\big(\mathbb E_{s\sim\mathcal D,a\sim\mu}[Q(s,a)]-\mathbb E_{s\sim\mathcal D,a\sim\hat\pi_\beta}[Q(s,a)]\big)+\tfrac12\mathbb E_{s,a,s'\sim\mathcal D}\big[(Q-\hat{\mathcal B}^\pi\hat Q^k)^2\big]\tag{Eq. 2}$$

- 同樣逐格微分（多一項 $-\alpha\pi_\beta$；無取樣誤差、$\hat\pi_\beta=\pi_\beta$）：
  $$\hat Q^{k+1}(s,a)=\mathcal B^\pi\hat Q^k(s,a)-\alpha\Big[\frac{\mu(a\mid s)}{\pi_\beta(a\mid s)}-1\Big]\tag{Eq. 13}$$
- $\mu<\pi_\beta$ 的動作，括號為負，等於**加上**正量 $\alpha(1-\mu/\pi_\beta)$，所以不再是逐點下界（App. C；論文說可構造三狀態二動作反例但未給出）。
- 對 $a\sim\pi$ 取期望（Eq. 14）：$\hat V^{k+1}(s)=\mathcal B^\pi\hat V^k(s)-\alpha\,\mathbb E_{a\sim\pi}[\mu/\pi_\beta-1]$。令 $\mu=\pi$，懲罰量是
  $$D_{\mathrm{CQL}}(s)=\sum_a\pi\Big(\frac{\pi}{\pi_\beta}-1\Big)=\sum_a\frac{(\pi-\pi_\beta)^2}{\pi_\beta}+\underbrace{\sum_a\pi_\beta\Big(\frac{\pi}{\pi_\beta}-1\Big)}_{=0}\ \ge0$$
  （App. C 的推導：把 $\pi$ 寫成 $(\pi-\pi_\beta)+\pi_\beta$；第二項為 $\sum\pi-\sum\pi_\beta=0$）。等號若且唯若 $\pi=\pi_\beta$。〔本頁推導〕這正是卡方散度 $\chi^2(\pi\Vert\pi_\beta)=\sum_a\pi^2/\pi_\beta-1$，論文沒有用這個名稱。數值例：$\pi_\beta=(0.5,0.5)$、$\pi=(0.9,0.1)$：兩動作 $\pi/\pi_\beta-1=\pm0.8$，第二個動作的 Q 被抬高 $0.8\alpha$；期望懲罰 $0.9\times0.8-0.1\times0.8=0.64=0.16/0.5+0.16/0.5$。
- **Thm. 3.2**（$\mu=\pi$，$\forall s\in\mathcal D$）：$\hat V^\pi(s)\le V^\pi(s)-\alpha\big[(I-\gamma P^\pi)^{-1}\mathbb E_\pi[\pi/\hat\pi_\beta-1]\big](s)+\big[(I-\gamma P^\pi)^{-1}\frac{C_{r,T,\delta}R_{\max}}{(1-\gamma)\sqrt{\lvert\mathcal D\rvert}}\big](s)$，此處 $1/\sqrt{\lvert\mathcal D\rvert}$ 改指狀態計數。α 門檻：$\alpha>\frac{C_{r,T,\delta}R_{\max}}{1-\gamma}\cdot\max_{s\in\mathcal D}\frac1{\sqrt{\lvert\mathcal D(s)\rvert}}\cdot\big[\sum_a\pi(a\mid s)(\pi(a\mid s)/\hat\pi_\beta(a\mid s)-1)\big]^{-1}$；$\hat{\mathcal B}^\pi=\mathcal B^\pi$ 時任何 $\alpha>0$。
- 「更緊」的精確量〔本頁推導〕：Eq. 1 取 $\mu=\pi$ 時每步期望懲罰 $\alpha\,\mathbb E_\pi[\pi/\pi_\beta]=\alpha(D_{\mathrm{CQL}}+1)$，Eq. 2 是 $\alpha D_{\mathrm{CQL}}$，每步差常數 α；因 $(I-\gamma P^\pi)^{-1}\mathbf 1=\mathbf 1/(1-\gamma)$，無取樣誤差時兩者的 $\hat V$ 恰好差 $\alpha/(1-\gamma)$。C2 直接顯示這個差。
- 邊界情形〔本頁推導〕：$\pi=\hat\pi_\beta$ 的狀態 $D_{\mathrm{CQL}}=0$，α 門檻中的 $[\cdot]^{-1}$ 無限大：有取樣誤差時，評估行為策略本身沒有任何 α 能保證下界。論文沒有討論。C3 顯示此情形。

#### 步驟 4：抬高項為什麼只能用 $\hat\pi_\beta$（App. D.2，Eq. 19–23，Thm. D.3）

- 把抬高項的分佈換成任意 $\nu$（ν-CQL，Eq. 19），表格式更新為 $\hat Q^{k+1}_\nu=\mathcal B^\pi\hat Q^k_\nu-\alpha_k(\mu-\nu)/\pi_\beta$（Eq. 20）。作者解 $\max_\nu\min_\pi\sum_a\pi(\pi-\nu)/\pi_\beta$：內層最佳 $\pi^*=\tfrac12\nu+\tfrac12\pi_\beta$（Eq. 22），代回後最佳 $\nu=\pi_\beta$、懲罰值 0（Eq. 23）。
- 〔本頁推導〕代入 Eq. 23 得懲罰 $=\tfrac14\big(1-\sum_a\nu^2/\pi_\beta\big)=-\tfrac14\chi^2(\nu\Vert\pi_\beta)\le0$，$\nu\ne\pi_\beta$ 時嚴格為負。
- 意義：若要對**所有** π 都保證下界，只有 $\nu=\pi_\beta$ 可以。前提：表格式、$\mu=\pi$、內層解用 KKT 時假設最壞的 π 是全支撐（App. D.2「If π(a|s) is full support」）。這是最壞情況 π 意義下的必要性，不是對每個固定 π 都必要。頁面只用文字與一行式子交代，不做元件。

#### 步驟 5：取樣誤差（Sec. 3.1、App. C、App. D.3）

- 集中性假設（Assumption D.1）：$\lvert\hat r-r(s,a)\rvert\le C_{r,\delta}/\sqrt{\lvert\mathcal D(s,a)\rvert}$、$\Vert\hat T-T\Vert_1\le C_{T,\delta}/\sqrt{\lvert\mathcal D(s,a)\rvert}$，推得 $\hat{\mathcal B}^\pi\hat Q^k-\mathcal B^\pi\hat Q^k\le(C_{r,\delta}+\gamma C_{T,\delta}2R_{\max}/(1-\gamma))/\sqrt{\lvert\mathcal D(s,a)\rvert}$（App. D.3）。主文與 Thm. 3.1／3.2 則寫成 $C_{r,T,\delta}R_{\max}/((1-\gamma)\sqrt{\lvert\mathcal D(s,a)\rvert})$（形式不一致，R3）。
- 經驗 Bellman 算子可能比真實算子高，例如隨機轉移中資料只含到樂觀的下一狀態（App. D.3）。
- Thm. 3.1 的 α 門檻（App. C）：$\alpha\ge\max_{s,a}\frac{C_{r,T,\delta}R_{\max}}{(1-\gamma)\sqrt{\lvert\mathcal D(s,a)\rvert}}\cdot\max_{s,a}\big(\frac{\mu(a\mid s)}{\hat\pi_\beta(a\mid s)}\big)^{-1}$。
- 作者的結論（Sec. 3.1 末）：$\lvert\mathcal D(s,a)\rvert$ 增加時，保證下界所需的理論 α 下降；無限資料時極小的 α 就夠。C3 以 bandit 的閉式解呈現 $1/\sqrt N$ 的關係（教學設定，不是論文的常數）。

#### 步驟 6：從策略評估到策略學習：CQL(R) 家族（Sec. 3.2，Eq. 3、Eq. 4、App. A、Eq. 7）

- 每個策略迭代都做完整評估太貴；策略通常由 Q 導出，所以讓 $\mu$ 去近似「最大化目前 Q 的策略」：
  $$\min_Q\max_\mu\ \alpha\big(\mathbb E_{s\sim\mathcal D,a\sim\mu}[Q]-\mathbb E_{s\sim\mathcal D,a\sim\hat\pi_\beta}[Q]\big)+\tfrac12\mathbb E_{s,a,s'\sim\mathcal D}\big[(Q-\hat{\mathcal B}^{\pi_k}\hat Q^k)^2\big]+\mathcal R(\mu)\tag{Eq. 3}$$
- $\mathcal R(\mu)=-D_{\mathrm{KL}}(\mu,\rho)$ 時 $\mu(a\mid s)\propto\rho(a\mid s)\exp(Q(s,a))$。取 $\rho=\mathrm{Unif}$（等價於 $\mathcal R=\mathcal H(\mu)$）得 **CQL(H)**：
  $$\min_Q\ \alpha\,\mathbb E_{s\sim\mathcal D}\Big[\log\sum_a\exp(Q(s,a))-\mathbb E_{a\sim\hat\pi_\beta}[Q(s,a)]\Big]+\tfrac12\mathbb E_{s,a,s'\sim\mathcal D}\big[(Q-\hat{\mathcal B}^{\pi_k}\hat Q^k)^2\big]\tag{Eq. 4}$$
- 內層解（App. A 只給答案；中間步驟〔本頁推導〕）：$\max_\mu\sum_x\mu f-\sum_x\mu\log\mu$ s.t. $\sum\mu=1$ 的 Lagrangian 對 $\mu(x)$ 微分：$f(x)-\log\mu(x)-1+\lambda=0\Rightarrow\mu^*=\exp(f)/Z$；代回值 $=\log Z=\log\sum_x\exp f(x)$。另有標準不等式 $\max_aQ\le\log\sum_a\exp Q\le\max_aQ+\log\lvert\mathcal A\rvert$〔本頁推導〕：log-sum-exp 是 max 的平滑版本，所以 Sec. 3.2 稱為 soft-maximum。
- KL 版本代回值為 $\log\sum_x\rho(x)\exp f(x)$〔本頁推導〕，$\rho=\mathrm{Unif}$ 時只差常數 $-\log\lvert\mathcal A\rvert$。取 $\rho=\hat\pi^{k-1}$（前一個策略）得 **CQL(ρ)**；論文的 Eq. 7 第一項寫成 $\mathbb E_{a\sim\rho}[Q\exp(Q)/Z]$（$=\mathbb E_{\mu^*}[Q]$）並註明「modulo some normalization terms」，不是完整代回，頁面不寫成「精確代入」。App. A 的 CQL(ρ) 式子寫 $+D_{\mathrm{KL}}$，與主文 $\mathcal R=-D_{\mathrm{KL}}$ 不一致（R3）。
- 作者觀察：CQL(ρ) 在高維動作空間較穩定，因為 log-sum-exp 的取樣估計變異大（Sec. 3.2、Table 2 說明）。CQL 可用 $\mathcal B^*$ 寫成 Q-learning，或寫成 actor-critic（Sec. 3.2）。App. A 另有 CQL(var)（Eq. 8–9，與分佈穩健最佳化的關係），頁面只提一句。
- 〔本頁推導〕bandit（$\gamma=0$、單一狀態）時 Eq. 4 的駐點條件為 $Q(a)=r(a)-\alpha[\mathrm{softmax}(Q)(a)/\pi_\beta(a)-1]$，是 Eq. 13 取 $\mu=\mathrm{softmax}(Q)$ 的隱式版本；目標對 $Q$ 嚴格凸（Hessian $=\alpha(\mathrm{diag}(p)-pp^\top)+\mathrm{diag}(\pi_\beta)$）。把駐點條件對 $a$ 加總得 $\sum_a\pi_\beta(a)(Q(a)-r(a))=0$：**資料分佈下的平均 Q 不變**，正則項只是把 Q 值從 $\mu$ 偏好、資料少的動作移到資料多的動作。C4 用這個例子。

#### 步驟 7：策略會變時下界還在嗎（Thm. 3.3，App. C）

- 設 $\pi_{\hat Q^k}(a\mid s)\propto\exp(\hat Q^k(s,a))$，$D_{\mathrm{TV}}(\hat\pi^{k+1},\pi_{\hat Q^k})\le\varepsilon$（策略相對 Q 慢變）。若 $\mathbb E_{\pi_{\hat Q^k}}\big[\frac{\pi_{\hat Q^k}}{\hat\pi_\beta}-1\big]\ge\max_{a:\hat\pi_\beta(a\mid s)>0}\frac{\pi_{\hat Q^k}(a\mid s)}{\hat\pi_\beta(a\mid s)}\cdot\varepsilon$，則 $\hat V^{k+1}(s)\le V^{k+1}(s)$。主文此節不含取樣誤差。
- 證明拆解：$\mathbb E_{\hat\pi^{k+1}}[\hat Q^{k+1}]=\mathbb E_{\hat\pi^{k+1}}[\mathcal B^\pi\hat Q^k]-\underbrace{\mathbb E_{\pi_{\hat Q^k}}[\pi_{\hat Q^k}/\hat\pi_\beta-1]}_{(a)\ \text{低估量}}+\underbrace{\sum_a(\pi_{\hat Q^k}-\hat\pi^{k+1})\pi_{\hat Q^k}/\hat\pi_\beta}_{(b)}$，(b) $\le D_{\mathrm{TV}}\cdot\max_a\pi_{\hat Q^k}/\hat\pi_\beta$。左式（LHS）＝若策略恰為 soft-optimal 時的保守量，右式＝策略與之不同時最多可能的高估。
- 〔本頁推導〕證明的式子省略了 α；補上後 (a)(b) 都乘同一個 α，所以 α 不**顯式**出現在條件中。但 α 經 $\hat Q^k$ 隱含進入：$\pi_{\hat Q^k}\propto\exp\hat Q^k$，α 越大，$\hat Q^k$ 把資料少的動作壓得越低，$\pi_{\hat Q^k}$ 越靠近 $\hat\pi_\beta$，左式 $\mathbb E[\pi_{\hat Q^k}/\hat\pi_\beta-1]$ 越接近 0，於是 ε 也必須更小。頁面不寫「α 不能幫忙」，只寫「條件主要靠 ε 小滿足，且 α 大時 ε 要更小」。
- 作者認為：這個條件是把策略學習率設為 3e-5（Q 為 3e-4 或 1e-4）的依據，「as dictated by Theorem 3.3」（Sec. 4）。實作的 SAC 策略是高斯分佈，不是 $\mathrm{softmax}(\hat Q)$，所以這是作者的設計理由，不是已驗證的保證。

#### 步驟 8：gap-expanding（Thm. 3.4，App. C、App. B）

- 陳述：對足夠大的 $\alpha_k$，$\forall s$，$\mathbb E_{\pi_\beta}[\hat Q^k]-\mathbb E_{\mu_k}[\hat Q^k]>\mathbb E_{\pi_\beta}[Q^k]-\mathbb E_{\mu_k}[Q^k]$：學到的 Q 把資料內動作與 $\mu_k$ 動作的期望差距拉得比真實 Q 大。作者認為這使 $\pi^k\propto\exp(\hat Q^k)$ 被拉近資料分佈，隱含地避免 OOD 動作（Sec. 3.2）。
- 一步的計算：懲罰 $\alpha_k(\mu_k-\pi_\beta)/\pi_\beta$ 在 $\mu_k$ 下的期望 $=\alpha_k\hat\Delta^k$，$\hat\Delta^k=\sum_a\mu_k(\mu_k-\pi_\beta)/\pi_\beta\ge0$；在 $\pi_\beta$ 下的期望 $=\alpha_k\sum_a(\mu_k-\pi_\beta)=0$。所以差距多出 $\alpha_k\hat\Delta^k$。
- 〔本頁推導〕照 Eq. 13 的符號重算，$\alpha_k\hat\Delta^k$ 前應為正號、條件分子為 $(\mu_k-\pi_\beta)^\top[\mathcal B^{\pi_k}\hat Q^k-Q^{k+1}](s,\cdot)$；論文 App. C 印的是「$-\alpha_k\hat\Delta^k$」、$(\pi_\beta-\mu_k)^\top[\cdot]$，且方括號內寫 $\mathcal B^{\pi_k}\hat Q^k-Q^k$（應為 $Q^{k+1}=\mathcal B^{\pi_k}Q^k$）；定理敘述用 $\hat Q^k$、證明用 $\hat Q^{k+1}$。結論（α 夠大即成立）不受影響，頁面不逐字抄那幾行的正負號（R3）。寬鬆界：分子以 $D_{\mathrm{TV}}(\pi_\beta,\mu_k)R_{\max}/(1-\gamma)$ 取代；有取樣誤差再加 $D_{\mathrm{TV}}(\hat\pi_\beta,\mu_k)\cdot2C_{r,T,\delta}R_{\max}/(1-\gamma)$。
- 〔本頁推導〕C4 是 Thm. 3.4 的 **γ = 0 駐點類比**，不是定理本身：定理是一步更新、$\mu_k$ 取自前一輪；C4 用自洽的 $\mu=\mathrm{softmax}(\hat Q)$，且 γ = 0 時 $\mathcal B\hat Q=r=\mathcal BQ$，條件中的 (a) 項為 0，所以任何 α > 0 都成立（「α 夠大」在此不起作用）。在這個類比中 $\hat Q-r=-\alpha(\mu/\pi_\beta-1)$，差距恰好多出 $\alpha\,\chi^2(\mu\Vert\pi_\beta)$（`check.py` 驗證到 1e-4）。
- 實證（App. B，Fig. 2）：hopper-expert、hopper-medium 上畫 Eq. 10 的 $\hat\Delta^k=\mathbb E_{s,a\sim\mathcal D}[\max_{a'_1..a'_N\sim\mathrm{Unif}}\hat Q^k(s,a')-\hat Q^k(s,a)]$。CQL 大致為負，BEAR 為正且隨訓練上升；hopper-expert 上 BEAR 出現「unlearning」。注意：Fig. 2 的 $\hat\Delta^k$ 與 Thm. 3.4 證明中的 $\hat\Delta^k$ 是同一符號、不同的量，「好」在 Fig. 2 是負值、在證明中是 $\ge0$（R3、常見錯誤 8）。App. B 也說明這種 OOD 高估來自函數近似，表格式不會出現。

#### 步驟 9：CQL 最佳化的是什麼（Thm. 3.5 = Lemma D.3.1，Eq. 5、Eq. 26）

- 設 $\hat Q^\pi$ 是 Eq. 2（$\mu=\pi$）在經驗 MDP $\hat M$ 中的不動點，則 $\pi^*:=\arg\max_\pi\mathbb E_{s\sim\rho_0(s)}[\hat V^\pi(s)]$（$\rho_0$ 為起始狀態分佈，論文寫 $\rho(s)$）等價於
  $$\pi^*\leftarrow\arg\max_\pi\ J(\pi,\hat M)-\alpha\frac1{1-\gamma}\mathbb E_{s\sim d^\pi_{\hat M}(s)}\big[D_{\mathrm{CQL}}(\pi,\hat\pi_\beta)(s)\big]\tag{Eq. 5}$$
- 證明只有一句：$\hat Q^\pi$ 是經驗 MDP 中、獎勵改為 $r(s,a)-\alpha[\pi(a\mid s)/\hat\pi_\beta(a\mid s)-1]$ 的 Bellman 不動點。〔本頁推導〕由 Eq. 13 直接讀出；對 π 取期望後每步扣 $\alpha D_{\mathrm{CQL}}(s)$，$d^\pi$ 是正規化的折扣狀態分佈，所以有 $1/(1-\gamma)$；懲罰依賴 π 本身，不是固定獎勵的普通 MDP。

#### 步驟 10：安全策略改進（Thm. 3.6 = Thm. D.4，Eq. 6、Eq. 27、Lemma D.4.1）

- 假設：Assumption D.1（獎勵與轉移的集中性），且 $\hat M$ 由依 $\hat\pi_\beta$ 取樣的動作產生（Lemma D.4.1）；$\pi^*$ 最佳化 Eq. 5。結論 $J(\pi^*,M)\ge J(\hat\pi_\beta,M)-\zeta$，機率 $1-\delta$：
  $$\zeta=2\Big(\frac{C_{r,\delta}}{1-\gamma}+\frac{\gamma R_{\max}C_{T,\delta}}{(1-\gamma)^2}\Big)\mathbb E_{s\sim d^{\pi^*}_{\hat M}}\Big[\frac{\sqrt{\lvert\mathcal A\rvert}}{\sqrt{\lvert\mathcal D(s)\rvert}}\sqrt{D_{\mathrm{CQL}}(\pi^*,\hat\pi_\beta)(s)+1}\Big]-\underbrace{\big(J(\pi^*,\hat M)-J(\hat\pi_\beta,\hat M)\big)}_{\ge\alpha\frac1{1-\gamma}\mathbb E[D_{\mathrm{CQL}}(\pi^*,\hat\pi_\beta)(s)]}\tag{Eq. 6}$$
- 第一項＝$\hat M$ 與 $M$ 不同（取樣誤差）造成的損失；第二項＝CQL 在經驗 MDP 中帶來的改進。$\lvert\mathcal D(s)\rvert$ 大時第一項小，較小的 α 就足以改進（Sec. 3.3）。
- 證明骨架（App. D.4）：(1) $\pi^*$ 最佳化 Eq. 5 且 $D_{\mathrm{CQL}}(\hat\pi_\beta,\hat\pi_\beta)=0$，所以 $J(\pi^*,\hat M)-\alpha\frac1{1-\gamma}\mathbb E[D_{\mathrm{CQL}}]\ge J(\hat\pi_\beta,\hat M)$（證明第一行漏寫 $1/(1-\gamma)$，Eq. 26–27 有）。(2) Lemma D.4.1 對任意 π 給出 $\lvert J(\pi,\hat M)-J(\pi,M)\rvert$ 的上界；$\sqrt{\lvert\mathcal A\rvert}\sqrt{D_{\mathrm{CQL}}+1}$ 來自 Cauchy–Schwarz：$\big(\sum_a\pi/\sqrt{\hat\pi_\beta}\big)^2\le\lvert\mathcal A\rvert\sum_a\pi^2/\hat\pi_\beta=\lvert\mathcal A\rvert(D_{\mathrm{CQL}}+1)$（論文把 $\pi/\sqrt{\hat\pi_\beta}$ 記為 $\alpha(s,a)$，與權衡係數撞名）。(3) 對 $\pi^*$ 與 $\hat\pi_\beta$ 各用一次，後者的誤差項被前者逐點蓋住，故係數 2。
- 數值例（推定：只為說明兩項的權衡，常數自訂）：若第一項 = 2.0，α = 1、γ = 0.9、$\mathbb E[D_{\mathrm{CQL}}]=0.3$，第二項 $\ge1\times10\times0.3=3.0$，所以 $\zeta\le2.0-3.0=-1.0$：$\pi^*$ 在真實 MDP 中至少比 $\hat\pi_\beta$ 好 1.0。資料量變為 4 倍時第一項減半為 1.0（∝ $1/\sqrt{\lvert\mathcal D(s)\rvert}$），ζ ≤ −2.0。
- 頁面要講對：ζ 可以是負的（第二項大於第一項），那才是真正的改進；ζ > 0 時只保證「不比行為策略差超過 ζ」。改進是相對 $\hat\pi_\beta$，不是相對最優策略。Sec. 5 說與 Laroche et al. 相比有類似的 horizon 二次方依賴與計數的平方根倒數依賴。

#### 步驟 11：函數近似的延伸（App. D.1，Thm. D.1、D.2）

- Thm. D.1（線性 $Q=w^\top F(s,a)$，$F^\top DF$ 可逆）：只保證 $d^{\pi_\beta}$ 下**期望**的下界（不是逐狀態），α 門檻的分子是「真實價值無法被線性類表示」造成的高估，分母是投影後的懲罰項（證明要求它為正，作者以對 π 取最小值說明）；能表示時任何 α > 0 都夠。
- Thm. D.2（NTK）：無限寬度線性化、每輪一步梯度；敘述在「if:」之後缺少條件式（R3）。頁面只用兩句交代，§7 作者自己承認深度網路的嚴格分析留給未來。

#### 步驟 12：實作（Sec. 4、Alg. 1、App. F）

- Alg. 1：$G_Q$、$G_\pi$ 是每輪對 Q 與策略做的梯度步數（論文未給值）。每步先用 $G_Q$ 個梯度步以 CQL(H)（或一般 CQL(R)）目標更新 $Q_\theta$（Q-learning 用 $\mathcal B^*$，actor-critic 用 $\mathcal B^{\pi_{\phi_t}}$）；actor-critic 再以 $G_\pi$ 步、SAC 式熵正則 $\mathbb E[Q_\theta(s,a)-\log\pi_\phi(a\mid s)]$ 更新策略。不需要估計行為策略：$\mathbb E_{a\sim\hat\pi_\beta}[Q]$ 直接用資料中的動作平均（Sec. 4）。
- 連續控制建在 SAC（RLkit）上，用 CQL(H)，α 以 Lagrange 對偶梯度自動調整；離散控制建在 QR-DQN 上，α 固定（Sec. 4、App. F）。
- **Eq. 30（CQL-Lagrange）**：$\min_Q\max_{\alpha\ge0}\ \alpha\big(\mathbb E_{s\sim d^{\pi_\beta}}[\log\sum_a\exp Q-\mathbb E_{a\sim\pi_\beta}[Q]]-\tau\big)+\tfrac12\mathbb E[(Q-\mathcal B^{\pi_k}\hat Q^k)^2]$。Q 值差距小於 τ 時 α 趨近 0；大於 τ 時 α 變大。τ 越小 → α 越大、越保守。〔本頁推導〕在 C4 的 bandit 上，$\log\sum_ae^{Q_a}-\sum_a\pi_\beta(a)Q_a\ge\mathcal H(\pi_\beta)$（凸共軛，等號在 $\mathrm{softmax}(Q)=\pi_\beta$），所以 $\tau\le\mathcal H(\pi_\beta)$ 時沒有有限的 α 能讓差距降到 τ，α → ∞；這是 τ 過小時 α 爆大的一個機制示意，不是論文的分析。
- τ 的選擇（App. F）：Gym MuJoCo 10.0；Kitchen、Adroit 5.0。初步實驗 τ = 2.0 時 α 漲到百萬量級、Q 被低估到約 −1e6；人類示範資料上 τ = 10.0 擋不住高估（Q > 1e+6）。AntMaze 的 τ 未在此段列出（Table 7 欄標 τ = 10.0）。
- 固定 α 版本 α = 5.0；Atari：Table 3 的 10% 資料 α = 1.0、1% 資料 α = 4.0、Fig. 1 α = 0.5（App. F）。
- 連續動作的 log-sum-exp（App. F）：從 $\mathrm{Unif}(a)$ 與目前策略 $\pi(a\mid s)$ 各取 $N=10$ 個動作做重要性取樣：$\log\sum_a e^{Q}\approx\log\Big(\frac1{2N}\sum_{a_i\sim\mathrm{Unif}}\frac{e^{Q(s,a_i)}}{\mathrm{Unif}(a)}+\frac1{2N}\sum_{a_i\sim\pi}\frac{e^{Q(s,a_i)}}{\pi(a_i\mid s)}\Big)$。離散動作直接用 `logsumexp`。
- 其他：Q 學習率預設 3e-4（試過 1e-4、3e-4 無顯著差）；策略學習率 3e-5（試過 3e-5、1e-4、3e-4；1e-4 在 hopper-medium、antmaze-medium-play 較好但在 Adroit 差）；D4RL 訓練 1M 梯度步；Atari 依 Agarwal et al. 報告 5X 梯度步後的回報；保留 SAC 的 twin-Q 與 soft target；超參數選擇只看資料上的預測 Q 值，不需線上評估（App. F）。另有 max-backup 變體（從策略取 10 個動作取最大），在 Kitchen、AntMaze 上比報告數字更好，故 Table 2 不是最佳設定（App. F；該式漏寫 γ）。
- 未寫明：SAC 熵溫度、$G_Q$／$G_\pi$、Lagrange α 的學習率與初值；所有表格都沒有標準差。

常見錯誤（頁面 S13 逐條列出；原第 7 條「混比表格」移到 S14 的表格之後）：
1. 說「CQL 學到 Q 的逐點下界」：只有 Eq. 1（Thm. 3.1）是逐點；實際用的 Eq. 2／Eq. 4 只保證 $\mathbb E_\pi[\hat Q]\le V^\pi$，資料中常見的動作可能被高估（Thm. 3.2 前文）。
2. 說「任何 α > 0 都給下界」：只在無取樣誤差且表格式時成立；否則 α 要超過門檻。
3. 把 τ 的方向弄反：τ 是 Q 值差距的預算，τ 小 → α 大；τ = 2.0 時 α 到百萬量級。
4. 策略學習率與 Q 相同：論文用 3e-5 對 3e-4，依據 Thm. 3.3 的慢變條件。
5. 高維連續動作直接用 CQL(H) 的取樣估計：重要性權重變異大，Adroit 上 CQL(ρ) 較穩定。
6. 以為 CQL 要估計 $\pi_\beta$：演算法不需要；但所有定理的式子都含 $\hat\pi_\beta$。
7. 把 Fig. 2 的 $\hat\Delta^k$ 當成 Thm. 3.4 的量（見步驟 8）。

#### 步驟 13：實驗（Sec. 6、App. B、App. G）

- **D4RL Gym（Table 1，正規化分數，4 seeds）**：單一策略資料（random／medium／expert）上 CQL 與最佳先前方法相近或略高；多策略混合資料（mixed／medium-expert／random-expert）多數列大幅領先，「sometimes as much as 2-3x」。〔本頁推導：逐列比對〕9 列混合資料中 CQL(H) 有 8 列最高，但 hopper-medium-expert 低於 BC（111.0 vs 111.9），halfcheetah-mixed 只高 0.3（46.2 vs 45.9），halfcheetah-medium-expert 為 1.2 倍（62.4 vs 51.7）；其餘 6 列為 1.9 倍以上。摘錄：halfcheetah-random-expert 92.5（SAC 53.0）、walker2d-random-expert 91.1（BRAC-v 2.7）、hopper-random-expert 110.5（BRAC-v 11.1）、hopper-mixed 48.6（BEAR 25.3）、walker2d-medium-expert 98.7（BC 11.3）。〔本頁推導：逐列比對〕18 列中 CQL(H) 最高的有 12 列；不是最高的 6 列：hopper-random 10.8（BRAC-v 12.0）、halfcheetah-medium 44.4（BRAC-v 45.5）、walker2d-medium 79.2（BRAC-v 81.3）、halfcheetah-expert 104.8（BEAR 108.2）、hopper-expert 109.9（BEAR 110.3）、hopper-medium-expert 111.0（BC 111.9）。
- Table 1 的 9 列混合資料（欄序照原文：SAC、BC、BEAR、BRAC-p、BRAC-v、CQL(H)；正規化分數）：

  | 任務 | SAC | BC | BEAR | BRAC-p | BRAC-v | CQL(H) |
  | --- | --- | --- | --- | --- | --- | --- |
  | halfcheetah-medium-expert | 1.8 | 35.8 | 51.7 | 43.8 | 45.3 | 62.4 |
  | walker2d-medium-expert | 1.9 | 11.3 | 10.8 | −0.3 | 0.9 | 98.7 |
  | hopper-medium-expert | 1.6 | 111.9 | 4.0 | 1.1 | 0.8 | 111.0 |
  | halfcheetah-random-expert | 53.0 | 1.3 | 24.6 | 30.2 | 2.2 | 92.5 |
  | walker2d-random-expert | 0.8 | 0.7 | 1.9 | 0.2 | 2.7 | 91.1 |
  | hopper-random-expert | 5.6 | 10.1 | 10.1 | 5.8 | 11.1 | 110.5 |
  | halfcheetah-mixed | −2.4 | 38.4 | 36.2 | 45.6 | 45.9 | 46.2 |
  | hopper-mixed | 3.5 | 11.8 | 25.3 | 0.7 | 0.8 | 48.6 |
  | walker2d-mixed | 1.9 | 11.3 | 10.8 | −0.3 | 0.9 | 26.7 |

- **AntMaze／Adroit／Kitchen（Table 2）**：CQL(H) 在 antmaze-medium-play 61.2、medium-diverse 53.7、large-play 15.8、large-diverse 14.9，其他方法除 BEAR 的 medium-diverse 8.0 外皆 0；CQL(ρ) 在中、大迷宮只有 4.6／5.1／3.2／2.3。Adroit human：pen 37.5／55.8（CQL(H)／CQL(ρ)，BC 34.4）、door 9.9／9.1（BC 0.5）。Kitchen CQL(H) 43.8／49.8／51.0（BC 33.8／33.8／47.5）。論文說 Adroit 上是先前最佳離線 RL 方法的 2–9 倍。例外（照錄，不寫「全面最好」）：pen-cloned BC 56.9 高於 CQL 39.2／40.3；relocate-cloned 同為 −0.1；kitchen-complete 的 CQL(ρ) 31.3 低於 BC 33.8，所以「全部超過 40%」只對 CQL(H) 成立；antmaze-umaze BEAR 73.0、BRAC-v 70.0 也高於 BC 65.0。
- **離線 Atari（Fig. 1、Table 3）**：設定 (1) 前 20% 樣本，Fig. 1 四個遊戲（Pong、Breakout、Qbert、Seaquest），CQL 在 3/4 遊戲穩定；設定 (2) 1%／10% 樣本（Table 3）。1%：Pong 19.3（REM −6.9）、Breakout 61.1（REM 11.0）、Q*bert 14012.0（QR-DQN 383.6）、Seaquest 779.4（QR-DQN 672.9）、Asterix 592.4（QR-DQN 166.3、REM 386.5；`paper.txt` 文字層把 592.4 擠到上一行，已以 `src/paper.pdf` 第 9 頁 200 dpi 渲染圖確認同列三格為 QR-DQN 166.3、REM 386.5、CQL(H) 592.4（粗體））。論文寫 Q*bert 36 倍、Breakout 6 倍〔本頁推導：14012.0/383.6 ≈ 36.5；61.1/11.0 ≈ 5.6〕。10% 資料 CQL 並非全勝：Seaquest 3674.1 < REM 3936.6；Asterix 156.3 < QR-DQN 189.2。
- **下界的實證（Table 4，預測值 − 真實折扣回報）**：CQL(H) −43.20／−10.93／−7.48（hopper-medium-expert／mixed／medium）；CQL（Eq. 1）−151.36／−22.87／−156.70；BEAR 65.93／1399.46／4.32；ensemble 的高估範圍 2.47e3–26.03e12；hopper-medium-expert 與 hopper-mixed 隨 ensemble 變大單調下降，只有 hopper-medium 非單調（26.03e12、437.57e6、1.12e12、885e3）。這是實驗觀察，不是對深度網路的證明。
- **消融（App. G，原始回報）**：Table 5 CQL(H) 優於 CQL(ρ)（halfcheetah-medium-expert 7234.5 vs 3995.6）；Table 6 去掉抬高項（Eq. 1）表現下降（hopper-medium 1866.1 vs 1028.4）；Table 7 Lagrange 版優於 α = 5.0（antmaze-medium-diverse 0.53 vs 0.21）。

#### 步驟 14：限制（Sec. 7）

嚴格理論只涵蓋表格式、線性與部分非線性（NTK）情形；深度網路的分析留給未來。離線 RL 會像監督式學習一樣過擬合，需要類似驗證誤差的早停方法。〔本頁推導〕另外，理論與實作之間有落差：Thm. 3.1 要求 $\mathrm{supp}\,\mu\subset\mathrm{supp}\,\hat\pi_\beta$，但 CQL(H) 的 $\mu\propto\exp Q$ 是全支撐；Thm. 3.2 要求 $\mu=\pi$，實作中 $\mu$ 是 Q 的 softmax、π 是另外訓練的 SAC 策略，由 Thm. 3.3 的慢變條件銜接。

---

## 2. 學習目標（驗收測試）

每個參考答案都必須寫在指定章節的**正文段落**中（不能只出現在元件、表格、圖或檢查題答案框）。答案中的〔本頁推導〕與「作者認為」標記在頁面正文與檢查題答案中都要保留。章節編號依第 3 節（S0–S16）。章節結尾以 `<details>` 呈現檢查題（題目與參考答案照搬）。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明離線 RL 的高估來源 | 為什麼離線 actor-critic 會高估 Q 值？Q 的訓練有沒有 state distribution shift？ | Bellman 目標值用學到的策略取 $a'$，但 Q 只在 $\pi_\beta$ 的動作上訓練；策略最大化 Q 而偏向 Q 被錯誤高估的 OOD 動作，離線無法試做修正。Q 的訓練沒有 state distribution shift（backup 不查詢資料外狀態），測試時策略可能有（Sec. 2） | S2 |
| LO2 | 推導 Eq. 1 的逐格解與逐點下界 | 表格式下 Eq. 1 每一輪的解是什麼？為什麼不動點是逐點下界？ | $\hat Q^{k+1}=\hat{\mathcal B}^\pi\hat Q^k-\alpha\mu/\hat\pi_\beta$（Eq. 11）；每步都減去非負量，$(I-\gamma P^\pi)^{-1}$ 元素非負，所以不動點 $\hat Q^\pi\le Q^\pi$ 逐點成立（無取樣誤差時任何 α > 0；有取樣誤差時 α 要超過門檻且 μ 在每個動作上 > 0；Thm. 3.1） | S3 |
| LO3 | 說明 Eq. 2 為何不是逐點下界 | Eq. 2 的逐格解是什麼？哪些動作的 Q 會被抬高？ | 表格式、無取樣誤差（$\hat{\mathcal B}^\pi=\mathcal B^\pi$、$\hat\pi_\beta=\pi_\beta$）時 $\hat Q^{k+1}=\mathcal B^\pi\hat Q^k-\alpha(\mu/\pi_\beta-1)$（Eq. 13）；$\mu(a\mid s)<\pi_\beta(a\mid s)$ 的動作被加上 $\alpha(1-\mu/\pi_\beta)$ | S4 |
| LO4 | 證明期望懲罰非負 | 為什麼 $D_{\mathrm{CQL}}(s)\ge0$？何時為 0？$\pi_\beta=(0.5,0.5)$、$\pi=(0.9,0.1)$ 時是多少？ | $D_{\mathrm{CQL}}=\sum_a(\pi-\pi_\beta)^2/\pi_\beta+\sum_a\pi_\beta(\pi/\pi_\beta-1)$，第二項為 0；所以 ≥ 0，等號若且唯若 $\pi=\pi_\beta$；數值 0.64（App. C） | S5 |
| LO5 | 說明抬高項為何用 $\hat\pi_\beta$ | 若把抬高項的分佈換成其他 ν 會怎樣？ | 表格式、$\mu=\pi$，且考慮全支撐的最壞 π 時，只有 $\nu=\pi_\beta$ 能保證懲罰非負；$\nu\ne\pi_\beta$ 時存在 π 使懲罰為負（$-\tfrac14\chi^2(\nu\Vert\pi_\beta)$〔本頁推導〕）、下界失效（Thm. D.3） | S5 |
| LO6 | 量化 Eq. 1 與 Eq. 2 的差距 | 無取樣誤差、$\mu=\pi$ 時，Eq. 1 與 Eq. 2 的 $\hat V$ 差多少？為什麼？ | 差 $\alpha/(1-\gamma)$：每步期望懲罰分別為 $\alpha(D_{\mathrm{CQL}}+1)$ 與 $\alpha D_{\mathrm{CQL}}$，差 α，經 $(I-\gamma P^\pi)^{-1}\mathbf 1=\mathbf 1/(1-\gamma)$ 累積〔本頁推導〕 | S6 |
| LO7 | 說明取樣誤差與 α 的關係 | 有取樣誤差時 α 的門檻如何隨資料量變化？評估 $\pi=\hat\pi_\beta$ 時會怎樣？ | 門檻正比於 $1/\sqrt{\lvert\mathcal D\rvert}$，資料越多所需 α 越小（Thm. 3.1、3.2，Sec. 3.1 末）；$\pi=\hat\pi_\beta$ 時 $D_{\mathrm{CQL}}=0$，沒有 α 能保證下界〔本頁推導〕 | S7 |
| LO8 | 推導 CQL(H) 的 log-sum-exp | $\max_\mu\mathbb E_\mu[Q]+\mathcal H(\mu)$ 的解與最大值是什麼？ | $\mu^*\propto\exp Q$，最大值 $\log\sum_a\exp Q(s,a)$，即 Eq. 4 的第一項（App. A）；它介於 $\max_aQ$ 與 $\max_aQ+\log\lvert\mathcal A\rvert$ 之間〔本頁推導〕 | S8 |
| LO9 | 說出策略學習的兩個保證 | Thm. 3.3 要求什麼條件？實作上如何落實？Thm. 3.4 說了什麼？ | Thm. 3.3（無取樣誤差）：$D_{\mathrm{TV}}(\hat\pi^{k+1},\pi_{\hat Q^k})\le\varepsilon$ 且 $\mathbb E_{\pi_{\hat Q^k}}[\pi_{\hat Q^k}/\hat\pi_\beta-1]\ge\max_a(\pi_{\hat Q^k}/\hat\pi_\beta)\cdot\varepsilon$，即策略相對 $\pi_{\hat Q^k}\propto\exp\hat Q^k$ 慢變；作者認為這是策略學習率 3e-5（Q 為 3e-4）的依據。Thm. 3.4：α 夠大時資料內動作與 $\mu_k$ 動作的期望 Q 差距大於真實 Q 的差距（gap-expanding） | S10、S11 |
| LO10 | 讀懂安全策略改進 | Thm. 3.6 的 ζ 由哪兩項組成？ζ 為負代表什麼？ | 第一項是 $\hat M$ 與 $M$ 不同的取樣誤差損失（∝ $1/\sqrt{\lvert\mathcal D(s)\rvert}$），第二項是 CQL 在經驗 MDP 中的改進（≥ $\alpha\frac1{1-\gamma}\mathbb E[D_{\mathrm{CQL}}]$）；ζ < 0 時 $\pi^*$ 確實優於 $\hat\pi_\beta$（Eq. 6） | S12 |
| LO11 | 描述實作與調參 | 連續控制的 CQL 怎麼實作？τ 小會怎樣？log-sum-exp 怎麼估？ | 在 SAC 上用 CQL(H)，α 以 Eq. 30 的 Lagrange 對偶梯度調整（Gym τ = 10.0、Kitchen／Adroit 5.0）；τ 小 → α 大，τ = 2.0 時 α 到百萬量級、Q 低估到約 −1e6；從 Unif 與目前策略各取 N = 10 個動作做重要性取樣（App. F） | S13 |
| LO12 | 正確讀出實驗結果 | CQL 在哪類資料上優勢最大？舉一個 CQL 不是最好的例子。Table 4 說明什麼？ | 多策略混合資料（mixed／medium-expert／random-expert），例如 hopper-random-expert 110.5 vs 11.1；例外如 pen-cloned BC 56.9 > CQL 39.2、hopper-expert BEAR 110.3 > 109.9；Table 4 中 CQL(H) 的預測值低於真實回報（−43.20／−10.93／−7.48），且比 Eq. 1 版本不保守 | S14 |

---

## 3. 頁面大綱

單欄閱讀（文字欄 680–720px），圖版加寬到最大 1080px。章節編號以 JetBrains Mono 顯示（`03 · 只壓低 Q 值得到逐點下界`）。章節標題一律陳述句、不擬人。全頁的視覺記號：一條「Q 值軸」（水平數線），每個動作一個標記，真實值為空心圓、CQL 估計為實心點，以箭頭相連；在 Hero、C1、C4 重複出現。「本節先定義」欄列出該節第一次出現、必須在使用前用一句話定義的詞（定義文字見 1.4 符號表）。

| # | id | 標題 | 內容（依序） | 本節先定義 | 長度 |
| --- | --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈保守 Q 學習：壓低離線資料外動作的 Q 值〉 | 副標：「CQL 在 Bellman 誤差上加一個正則項，壓低學到的 Q 值；在論文的假設下，策略的估計價值不超過真實價值。」三張事實卡（見下）；靜態 Q 值軸示意；連結 arXiv、`../index.html#paper-568`、`cql-lineage-hub.html` | — | 1 螢幕 |
| S1 | `basics` | 01 · Q 函數由 Bellman 算子的不動點定義 | MDP、π、$Q^\pi$、$V^\pi$、$\mathcal B^\pi$、$P^\pi$；2 狀態小例子說明 $Q^\pi=(I-\gamma P^\pi)^{-1}r$ 與 $(I-\gamma P^\pi)^{-1}=\sum_t\gamma^t(P^\pi)^t$ 元素非負；策略評估／改進交替（Sec. 2 兩式）；末尾符號表。延伸：`rl-mdp-q-learning-basics.html` | 表格式、不動點、off-policy（用別的策略收集的資料學習）、actor-critic（另有一個策略網路最大化 Q）、SAC 與 QR-DQN（兩個現成的線上演算法，本文的底座，各一句） | 500–600 字 |
| S2 | `shift` | 02 · 離線資料讓策略偏向被高估的動作 | 步驟 1；LO1 答案寫在此段。延伸：`offline-rl-distribution-shift.html` | 行為策略、經驗行為策略、OOD 動作、經驗 Bellman 算子 | 350–450 字 |
| S3 | `eq1` | 03 · 只壓低 Q 值得到逐點下界 | 步驟 2：Eq. 1 → 逐格微分（可收合推導）→ Eq. 11 → 不動點 → Thm. 3.1 的 $\hat{\mathcal B}^\pi=\mathcal B^\pi$ 版（不出現 $C_{r,T,\delta}$）；support 條件 | μ、α | 400–500 字 |
| S4 | `eq2` | 04 · 再抬高資料上的 Q 值，部分動作的 Q 會變大 | 步驟 3 前半：Eq. 2 → Eq. 13（含假設）；**C1**。延伸：`cql-conservative-policy-evaluation.html` | — | 300–400 字 + C1 |
| S5 | `dcql` | 05 · 策略下的期望懲罰非負，所以期望值仍是下界 | Eq. 14 → $D_{\mathrm{CQL}}$ 恆等式與 0.64 數值例 → Thm. 3.2（無取樣誤差版）→ 步驟 4（ν-CQL 一段）；回指 C1 的 D 讀數 | 卡方散度（一句，標〔本頁推導〕） | 400–500 字 |
| S6 | `mdp` | 06 · 每一步的懲罰經 Bellman 迭代累積 | 步驟 3 後半：懲罰經 $(I-\gamma P^\pi)^{-1}$ 傳播；$\alpha/(1-\gamma)$〔本頁推導〕；**C2** | — | 300–400 字 + C2 |
| S7 | `sampling` | 07 · 取樣誤差決定 α 至少要多大 | 步驟 5：集中性假設、完整的 Thm. 3.1／3.2（含取樣誤差項）、兩個 α 門檻、μ > 0 的條件、$1/\sqrt{\lvert\mathcal D\rvert}$、$\pi=\hat\pi_\beta$ 邊界情形；**C3** | $\lvert\mathcal D(s,a)\rvert$、$C_{r,T,\delta}$、δ、集中不等式（一句直覺） | 450–550 字 + C3 |
| S8 | `cqlr` | 08 · 以 Q 的 softmax 作為被壓低的分佈：CQL(R) 與 CQL(H) | 先用 3–4 句定義熵、KL、softmax、log-sum-exp、Lagrangian；步驟 6 前半：Eq. 3、KL 正則的解、Lagrangian 推導（可收合）、Eq. 4、soft-max 不等式 | 熵、KL、softmax、log-sum-exp（$\log\sum_a e^{Q_a}$）、Lagrangian、ρ（只此一義）、$\mathcal B^*$（回指 S1） | 400–500 字 |
| S9 | `cqlh` | 09 · CQL(H) 把資料少的動作的 Q 值移給資料多的動作 | CQL(ρ) 與 Eq. 7（含「modulo normalization」說明）、資料平均不變〔本頁推導〕；**C4**（含 τ 模式，接 Eq. 30 的預告）。延伸：`cql-algorithm-variants-and-guarantees.html` | — | 300–400 字 + C4 |
| S10 | `slow` | 10 · 策略變得夠慢時，估計值仍是下界 | 步驟 7：Thm. 3.3 的條件、(a)(b) 拆解、α 隱含進入、作者認為的 3e-5 | $D_{\mathrm{TV}}$、soft-optimal 策略 $\pi_{\hat Q^k}$ | 300–400 字 |
| S11 | `gap` | 11 · 資料內與資料外動作的 Q 值差距被擴大 | 步驟 8：Thm. 3.4、一步的兩個期望；以 C4 預設讀數為例（差距 −0.0424 → 0.0836，增加 0.1260 = αχ²），並註明 C4 是 γ = 0 駐點類比；Fig. 2 的文字描述與符號提醒 | — | 350–450 字 |
| S12 | `spi` | 12 · CQL 等於最佳化帶懲罰的經驗回報 | 步驟 9、10：Eq. 5、假設、Eq. 6、兩項意義、ζ 數值例（推定）、ζ 正負；步驟 11 兩句 | $d^\pi_M$、$\hat M$、$J(\pi,M)$、起始分佈 $\rho_0$ | 450–550 字 |
| S13 | `impl` | 13 · 實作只需在 SAC 或 QR-DQN 上改 Q 的目標 | 步驟 12：Alg. 1（兩欄格線；< 768px 改為逐行下接說明）、Eq. 30 與 τ（回指 C4 τ 模式）、IS 估計式、超參數表、常見錯誤 7 條。延伸：`cql-vs-policy-constraint-methods.html` | 重要性取樣、twin-Q（兩個 Q 網路取較小者）、Lagrange 對偶梯度（α 依限制是否被違反而增減）、$G_Q$／$G_\pi$（Alg. 1 每輪 Q 與策略的梯度步數，論文未給值） | 450–550 字 + 表 |
| S14 | `results` | 14 · 多數混合資料集上 CQL 明顯領先 | 步驟 13：Table 1 摘錄（9 列混合資料全列 + 單一策略的例外）、Table 2 摘錄、Table 3、Table 4、消融三句；表格之後放「混比表格」常見錯誤（欄序、正規化 vs 原始回報）。不重繪曲線。延伸：`cql-experiments-d4rl-atari.html` | D4RL 正規化分數（0 = 隨機策略、100 = 專家，一句）、BC | 450–550 字 + 4 表 |
| S15 | `history` | 15 · 研究脈絡（選讀） | 時間線（6.4）+ 一段文字：DQN → SAC／QR-DQN → TD3 雙 Q → BCQ／BEAR／BRAC → D4RL 與綜述 → CQL；不對應學習目標 | — | 時間線 + 100–150 字 |
| S16 | `next` | 16 · 限制與延伸閱讀 | 步驟 14；筆記清單（依總覽閱讀順序）；arXiv | — | 200 字 + 清單 |

Hero 三張事實卡（不使用未定義術語）：
1. 「離線強化學習只能用現成資料。策略會偏向資料中沒出現、價值被估得太高的動作。」
2. 「CQL 在訓練 Q 函數時，壓低策略偏好的動作的 Q 值、抬高資料中動作的 Q 值。在表格式設定下，只要壓低的強度 α 夠大（沒有取樣誤差時任何 α > 0 都可以），策略的估計價值不超過真實價值。」
3. 「實作只在 SAC 或 QR-DQN 上多約 20 行。論文摘要稱在許多任務上回報是先前方法的 2–5 倍，尤其是來源複雜的資料；Gym 混合資料集上為 2–3 倍（Sec. 6）。」

桌機（≥ 1024px）左側 sticky 章節目錄；< 1024px 改為頂部橫向捲動章節列（沿用外殼）。

---

## 4. 互動元件規格

### 4.0 共用規則

- **共用數學**：`norm(w)`（權重正規化）、`dcql(pi, pb)`＝$\sum\pi(\pi/\pi_b-1)$、`chi2(pi, pb)`＝$\sum(\pi-\pi_b)^2/\pi_b$、`solve(A, b)`（部分選主元高斯消去）、`lse(q)`（先減 max）、`softmax(q)`（先減 max）、`erfc(x)`。全部與 `check.py` 同名同算法。比值 $\pi(a\mid s)/\pi_\beta(a\mid s)$ 在程式中叫 `ratio`，頁面上一律寫成 π/π̂β 的分數形式，不用 ρ。
- **控制項分層**：每個圖版預設只露出 1–3 個主要控制（各元件列明）與預設組按鈕；其餘輸入收在 `<details class="advanced">`（摘要列「進階設定」）中。權重輸入一律是整數權重滑桿、即時正規化並顯示機率（兩位小數）；$\pi_\beta$ 權重最小 1（保證 $\hat\pi_\beta>0$，Sec. 3.1 的假設）；π 權重最小 0，全為 0 時以均勻計算並提示「π 至少要有一個正權重，暫以均勻分佈計算」。
- **更新頻率**：`input` 事件只寫入狀態，以 `requestAnimationFrame` 合併為每影格一次重畫；live region（`aria-live="polite"`）與 SVG `aria-label` 只在 `change` 或停止輸入 500ms 後更新。
- **文字與符號**：動態文字（讀數、圖上標籤、`aria-*`）不用 Unicode 組合字元（U+0302 等）與下標數字。HTML 讀數中的帽子符號寫成 `<span class="hat">V</span>`（CSS 以 `::before` 在字母上方放小號「^」，`aria-hidden` 處理後另給 `aria-label`）；SVG 標籤改用文字（「真實」「Eq. 1」「CQL 估計」「資料比例」）；本節規格中為了簡潔寫成 π̂β、Q̂、V̂ 的動態標籤，實作時一律依此規則轉寫（HTML 用 hat span，SVG 用文字詞）；`aria-label`／`aria-valuetext` 用中文詞（例「CQL 估計值減真實值等於負 0.80」「α 等於 0.50」）。
- **無障礙**：滑桿有 `<label>` 與 `aria-valuetext`；分段按鈕 `role="radiogroup"`、roving tabindex、方向鍵切換；資料系列以「線型或標記形狀＋直接文字標籤」區分（真實值空心圓、Eq. 1 方形、Eq. 2／CQL 實心圓；正負以 ▲／▼ 標示），不只靠顏色；沒有 hover 才看得到的資訊，所有讀數都在圖下讀數表（觸控等同）。觸控區 ≥ 44px。
- **超出範圍**：任何軸上超出範圍的點釘在邊界，改畫空心三角「‹」或「›」並在旁邊寫實際數值；箭頭在邊界處畫鋸齒截斷記號（兩條短斜線），表示長度被截斷。
- **窄螢幕（< 768px）**：每個圖版單欄堆疊（圖 → 主要控制 → 進階設定 → 讀數），多張圖一律上下排列；每張 SVG 寬度等於容器寬度（`ResizeObserver` 重畫），圖上文字固定 12–13px；刻度數量減半。讀數表與四欄以上的表格放在 `overflow-x:auto` 容器中，該容器 `tabindex="0"`、`role="region"`、`aria-label` 為表名，讓鍵盤能橫捲；`.katex-display` 同樣加 `tabindex="0"` 與 `role="group"`。Eq. 2、Eq. 3、Eq. 4、Eq. 6、Eq. 30、Thm. 3.1／3.2 在 < 600px 時以 `aligned` 拆行（正則項一行、Bellman 誤差一行）。
- **圖版底部**一行來源：「讀數由本頁依 Eq. ○○ 即時計算」；用到〔本頁推導〕或自訂設定時加「本頁推導」／「推定」pill 與一句依據。
- **色彩**：真實值 `--c-true`、Eq. 1 `--c-eq1`、CQL 估計 `--accent`。C2 的 Eq. 1 斜線填色用 SVG `<pattern>`，描線色 `var(--c-eq1)`、底色 `var(--surface)`，深色模式隨 token 切換（pattern 寫在每次重畫的 `<defs>` 內，不寫死色碼）。

### C1 · 單一狀態：Eq. 1 與 Eq. 2 的逐動作差異（S4，S5 回指）

- **教的概念**：Eq. 11 每個動作都被壓低（逐點下界）；Eq. 13 把 $\mu<\hat\pi_\beta$ 的動作抬高，但 π 下的期望仍被壓低 $\alpha D_{\mathrm{CQL}}$；$D_{\mathrm{CQL}}$ 兩種算法相等且 ≥ 0；Eq. 1 比 Eq. 2 多低估 α（γ = 0 時的 $\alpha/(1-\gamma)$）。
- **設定**（頁面寫出）：單一狀態、三個動作 a1 a2 a3、γ = 0（所以 $Q=r$）、無取樣誤差、$\mu=\pi$。獎勵固定 $r=(1.0,0.6,0.8)$（推定：教學用自訂值）。
- **主要控制**：α 滑桿 0–3，步長 0.05，預設 0.5；π 預設組分段按鈕：「π 偏好 a3」權重 (10, 20, 70)（預設）、「π 偏好 a1」(80, 10, 10)、「π = π̂β」（π 權重設成目前的 $\pi_\beta$ 權重）。
- **進階設定**：$\pi_\beta$ 權重 3 個滑桿（整數 1–100，預設 50, 30, 20）；π 權重 3 個滑桿（整數 0–100）；「重設」。
- **計算**：`ratio[a]`＝$\pi_a/\pi_{\beta,a}$；Eq. 1：$\hat Q_1(a)-Q(a)=-\alpha\,\mathrm{ratio}_a$；Eq. 2：$\hat Q_2(a)-Q(a)=-\alpha(\mathrm{ratio}_a-1)$；$V=\sum\pi r$；$\hat V_1-V$、$\hat V_2-V$ 以 $\sum_a\pi_a(\hat Q(a)-Q(a))$ 直接加總（不用化簡式，兩者分別等於 $-\alpha(D+1)$ 與 $-\alpha D$ 本身就是檢查）；$D$ 以 `dcql` 與 `chi2` 各算一次並排顯示。
- **顯示**：三條 Q 值軸（每個動作一條，範圍固定 [−3, 2]，超出依 4.0 處理）：空心圓「真實」、方形「Eq. 1」、實心圓「Eq. 2」，箭頭從真實指向估計值，上升 ▲、下降 ▼。每條軸左側標動作名、π̂β、π、π/π̂β。讀數表：列「真實」「Eq. 1」「Eq. 2」，欄 V 或估計 V、與 V 的差；另一列兩種算法的 $D_{\mathrm{CQL}}$。一行結論文字依狀態切換：「Eq. 2 抬高了 π < π̂β 的動作，但估計 V 仍比 V 低 α·D」／「π = π̂β：D = 0，Eq. 2 的估計 V 等於 V」。
- **數值保護**：$\pi_\beta$ 權重 ≥ 1，所以 ratio ≤ 201（π 權重 (100,0,0)、π_β 權重 (1,100,100) 時）；$D<10^{-12}$ 顯示 0.000。
- **讀者應觀察到**：(1) α = 0 時三個標記重合；(2) Eq. 1 的箭頭全部向下；(3) Eq. 2 中 π 小於 π̂β 的動作箭頭向上，但估計 V 仍 < V；(4) 「π = π̂β」後 Eq. 2 不改變任何 Q，Eq. 1 仍整體下移 α；(5) 兩種 D 永遠相等。

### C2 · 三狀態 MDP：懲罰的累積與不動點（S6）

- **教的概念**：多步時每步的懲罰經 $(I-\gamma P^\pi)^{-1}$ 累積；Eq. 1 與 Eq. 2 的估計 V 在每個狀態恰差 $\alpha/(1-\gamma)$〔本頁推導〕；Eq. 2 的估計 V 仍 ≤ V，但個別 $(s,a)$ 可能高估；迭代時每一輪都在真實迭代之下。
- **MDP**（推定：教學用自訂例子；頁面以 SVG 狀態圖畫出）：狀態 s0 s1 s2，動作「左」「右」。轉移：s0 左→s0、右→s1；s1 左→s0；s1 右：以機率 0.8 到 s2、以機率 0.2 到 s0；s2 左→s1、右→s2。獎勵只有 $r(s_2,\text{右})=1$。
- **主要控制**：α 滑桿 0–1，步長 0.01，預設 0.1；γ 滑桿 0–0.95，步長 0.05，預設 0.9；迭代步數 k 滑桿 0–100，步長 1，預設 100；預設組按鈕：「預設」、「γ = 0.5（看高估）」、「π = π̂β（均勻）」（π̂β 全設 0.5、π 設 0.5）。
- **進階設定**：$\pi_\beta(\text{右}\mid s_i)$ 三個滑桿（[0.02, 0.98]，步長 0.02，預設 0.8, 0.5, 0.2）；π(右｜s)（三狀態共用）0–1，步長 0.05，預設 0.9。
- **計算**：6×6 矩陣 $A=I-\gamma P^\pi$（索引 (s0左, s0右, s1左, s1右, s2左, s2右)）；`solve` 求 $Q=A^{-1}r$、$\hat Q_1=A^{-1}(r-\alpha\,\mathrm{ratio})$、$\hat Q_2=A^{-1}(r-\alpha(\mathrm{ratio}-1))$；$V(s)=\sum_a\pi Q$；$D_{\mathrm{CQL}}(s)$。迭代：從 0 開始、以精確 $\mathcal B^\pi$ 套用 Eq. 11／Eq. 13 與無懲罰的真實迭代，記錄 $k=0..100$ 的 $V^k(s_0)$。$\gamma\le0.95$ 時 $\Vert A\Vert_\infty\le1+\gamma$、$\Vert A^{-1}\Vert_\infty\le1/(1-\gamma)\le20$，條件數 ≤ 39，不需額外保護。迭代只在 π、π̂β、α、γ 改變時重算；拖動 k 只移動指標線。
- **顯示**（≥ 768px：上排狀態圖＋橫條圖，下排兩張折線圖；< 768px 四張依序堆疊）：
  - 狀態圖：三個圓、帶機率標籤的箭頭，箭頭粗細 ∝ π 的機率，旁邊小字 π̂β。
  - 橫條圖：6 組 $(s,a)$，每組三根：真實（空心框）、Eq. 1（斜線 pattern）、Eq. 2（實心）；縱軸自動：$[\min(0,\text{全部值}),\max(\text{全部值})]$ 外擴 10%，刻度取 1／2／5 × 10ⁿ；$\hat Q_2>Q$ 的組加 ▲ 與文字「高估」。
  - 折線圖 A「價值」：$V^k(s_0)$ 對 k，實線真實、虛線 Eq. 1、點線 Eq. 2；縱軸自動（同上規則，包含不動點值）；右端每條線標兩個數：「k = 100：…」與「不動點：…」，並以水平細虛線畫出不動點；目前 k 的垂直細線。
  - 折線圖 B「與真實迭代的差」：$\hat V_1^k(s_0)-V^k(s_0)$ 與 $\hat V_2^k(s_0)-V^k(s_0)$，零線加粗；縱軸自動（含 0）。讓 k = 1 的 −0.0063 等小差距可見（預設下縱軸約 [−3.4, 0.3]）。
  - 讀數表：每個狀態的 V、Eq. 1 估計 V、Eq. 2 估計 V、兩者之差、$-\alpha/(1-\gamma)$、$D_{\mathrm{CQL}}(s)$；另一行「目前 k 的 Eq. 2 差值」。
- **讀者應觀察到**：(1) Eq. 1 與 Eq. 2 估計 V 之差三格都等於 $-\alpha/(1-\gamma)$（預設 −1.000）；(2) γ 越大，同一 α 造成的低估越大；(3) 「γ = 0.5」預設組下 (s1, 左) 出現「高估」，但三個狀態的 Eq. 2 估計 V 都 < V；(4) 「π = π̂β」預設組下 D = 0，Eq. 2 與真實值重合；(5) 折線圖 B 中 Eq. 2 的差從 k = 1 起就是負的（−0.0063）；(6) γ = 0.95 時 k = 100 的值（Eq. 2：8.5316）與不動點（8.5902）尚差 0.06。

### C3 · 取樣誤差：資料越多，保證下界所需的 α 越小（S7）

- **教的概念**：有取樣誤差時估計 V 可能高於 V；α 要超過門檻；門檻隨資料量以 $1/\sqrt N$ 下降、與 $1/\hat D$ 成正比；$\pi=\hat\pi_\beta$ 時沒有 α 有效。
- **pill 文字**（推定＋本頁推導）：「高斯噪聲是本頁的教學設定。α_δ 只重現 Thm. 3.2 門檻中的 1/√N 與 1/D̂ 兩個因子，不是定理的門檻：定理用狀態計數與最壞情況常數 C，這裡用每個動作的計數與高斯分位數。」
- **設定**：單一狀態、三動作、γ = 0、$\mu=\pi$、真實獎勵 $r=(1.0,0.6,0.8)$（同 C1）；每次觀察到的獎勵 = $r(a)+\sigma\varepsilon$，$\varepsilon\sim\mathcal N(0,1)$。「目標筆數」N；計數 $n_a=\max(1,\mathrm{round}(N\pi_{\beta,a}))$，實際總筆數 $\sum n_a$ 可能 ≠ N（讀數另列），$\hat\pi_\beta=n/\sum n$。CQL（Eq. 2 的表格解）：$\hat Q(a)=\bar r(a)-\alpha(\pi_a/\hat\pi_{\beta,a}-1)$。於是估計 V − V $=\sum_a\pi_a(\bar r(a)-r(a))-\alpha\hat D$，第一項 $\sim\mathcal N(0,s_N^2)$，$s_N=\sigma\sqrt{\sum_a\pi_a^2/n_a}$：
  $$P(\hat V>V)=1-\Phi\big(\alpha\hat D/s_N\big),\qquad \alpha_\delta=z_{1-\delta}\,s_N/\hat D.$$
- **主要控制**：目標筆數 N 分段按鈕 {10, 30, 100, 300, 1000, 3000, 10000}，預設 100；α 滑桿 0–2，步長 0.01，預設 0.2；按鈕「π = π̂β」：把 π 權重設成目前的計數 $n_a$（所以 $\hat D=0$ 精確成立，與取整一致）。
- **進階設定**：σ 滑桿 0.1–2.0，步長 0.1，預設 1.0；δ 分段按鈕 {0.10, 0.05, 0.01}，預設 0.05（z = 1.2816／1.6449／2.3263）；$\pi_\beta$、π 權重滑桿（同 C1 預設）。
- **計算**：$\Phi(x)=1-\tfrac12\mathrm{erfc}(x/\sqrt2)$，`erfc` 用 Numerical Recipes `erfcc`（相對誤差 < 1.2e-7）：$z=\lvert x\rvert$，$t=1/(1+z/2)$，$y=t\exp(-z^2-1.26551223+t(1.00002368+t(0.37409196+t(0.09678418+t(-0.18628806+t(0.27886807+t(-1.13520398+t(1.48851587+t(-0.82215223+t\cdot0.17087277)))))))))$，$x\ge0$ 回傳 $y$，否則 $2-y$；閉式，不用 Monte Carlo。
- **數值保護**：$\hat D<10^{-12}$ → P = 0.5（與 α 無關）、$\alpha_\delta=\infty$，讀數顯示「D̂ = 0：任何 α 都不能讓高估機率低於 0.5」，左圖曲線畫成 0.5 水平線，右圖該點不畫並在圖上寫同一句；α = 0 → P = 0.5；$P<10^{-6}$ 顯示「< 0.000001」。
- **顯示**（≥ 768px 左右並排；< 768px 上下）：
  - 左圖：橫軸 α ∈ [0, 2]（與滑桿相同），縱軸 P ∈ [0, 0.5]；7 條 N 曲線（細線、各自直接標「N = 10」…；目前 N 加粗），水平虛線 δ，目前 α 的垂直線與交點。若目前 $\alpha_\delta>2$，在右緣畫「›」並標「α_δ = 數值（超出圖範圍）」。
  - 右圖：log–log，橫軸 N ∈ [10, 10000]；縱軸 $\alpha_\delta$ 範圍取 7 個有限值的 [min/2, max×2]，再夾在 [1e-3, 1e2] 內，超出的點依 4.0 釘邊；7 個圓點連線，加一條過 N = 100 點、斜率 −1/2 的參考虛線（標「∝ 1/√N」），目前 N 的點放大。
  - 讀數：$n_a$、實際總筆數、π̂β、$\hat D$、$s_N$、P、$\alpha_\delta$。
- **讀者應觀察到**：(1) N 增為 10 倍，$\alpha_\delta$ 縮為約 $1/\sqrt{10}$；(2) 固定 α，N 越大高估機率越小；(3) σ 加倍，$\alpha_\delta$ 加倍；(4) 「π = π̂β」使所有曲線停在 0.5，任何 N 都一樣。

### C4 · CQL(H)：log-sum-exp、駐點與差距擴大（S9，S11、S13 回指）

- **教的概念**：$\max_\mu\mathbb E_\mu[Q]+\mathcal H(\mu)=\log\sum\exp Q$，$\mu^*=\mathrm{softmax}(Q)$，介於 max 與 max + log|A|；CQL(H) 壓低資料中罕見而 Q 高的動作；資料平均 Q 不變〔本頁推導〕；資料內與 μ 下的差距被擴大 $\alpha\chi^2(\mu\Vert\pi_\beta)$；τ 越小 α 越大（Eq. 30）。
- **標題下的 pill**（本頁推導）：「γ = 0 的駐點類比：μ 取自 Q̂ 本身的 softmax、沒有 Bellman 遞迴，所以 Thm. 3.4 的『α 夠大』在此任何 α > 0 都成立。這不是 Thm. 3.4 的一步更新本身。」並註明：表格式下沒有函數近似造成的 OOD 高估，這裡只示範正則項如何改變 Q。
- **設定**：單一狀態、四動作、γ = 0、無取樣誤差；真實獎勵 $r=(1.0,0.6,0.2,1.4)$（推定；a4 最高）。目標（Eq. 4 的 bandit 形式，Bellman 誤差以 $\pi_\beta$ 加權）：$f(Q)=\alpha\big(\mathrm{lse}(Q)-\sum_a\pi_\beta(a)Q(a)\big)+\tfrac12\sum_a\pi_\beta(a)(Q(a)-r(a))^2$。
- **主要控制**：$\pi_\beta$ 預設組分段按鈕：「a4 罕見」(0.45, 0.35, 0.15, 0.05)（預設）、「均勻」(0.25 ×4)、「a4 常見」(0.10, 0.20, 0.20, 0.50)；模式分段按鈕「直接設定 α」（預設）／「以 τ 決定 α（Eq. 30）」；前者 α 滑桿 0–5，步長 0.05，預設 1.0；後者 τ 滑桿 1.10–1.60，步長 0.01，預設 1.30。無進階設定。
- **計算**：
  - 解 $\hat Q$：阻尼 Newton。$Q\leftarrow r$；每次 $p=\mathrm{softmax}(Q)$，$g_a=\alpha(p_a-\pi_{\beta,a})+\pi_{\beta,a}(Q_a-r_a)$，$H=\alpha(\mathrm{diag}(p)-pp^\top)+\mathrm{diag}(\pi_\beta)$（正定），解 $Hd=-g$；Armijo 回溯（$c=10^{-4}$，減半，下限 1e-10）；停止於 $\max\lvert g\rvert<10^{-9}(1+\alpha)$ 或 50 次（`check.py` 實測 ≤ 6 次）。
  - 讀數：$\mu=\mathrm{softmax}(\hat Q)$、$\mathcal H(\mu)$、lse、max、$\mathbb E_{\pi_\beta}[\hat Q]$、$\mathbb E_{\pi_\beta}[r]$、$\widehat{\mathrm{gap}}=\mathbb E_{\pi_\beta}[\hat Q]-\mathbb E_\mu[\hat Q]$、$\mathrm{gap}=\mathbb E_{\pi_\beta}[r]-\mathbb E_\mu[r]$、$\alpha\chi^2(\mu\Vert\pi_\beta)$。
  - τ 模式〔本頁推導〕：令 $G(\alpha)=\mathrm{lse}(\hat Q_\alpha)-\mathbb E_{\pi_\beta}[\hat Q_\alpha]$（Eq. 30 方括號內的量）。Eq. 30 對 α ≥ 0 取 max 的鞍點：$G(0)\le\tau$ 時 $\alpha^*=0$；$\tau\le\mathcal H(\pi_\beta)$ 或 $G(10^4)>\tau$ 時 $\alpha^*=\infty$（$G\ge\mathcal H(\pi_\beta)$，凸共軛下界）；否則在 $\log\alpha\in[\log10^{-6},\log10^4]$ 上二分 60 次求 $G(\alpha^*)=\tau$（$G$ 隨 α 單調下降，`check.py` 驗證）。圖上顯示 $G(\alpha)$ 對 α（對數橫軸 1e-3–1e4）的曲線、水平線 τ 與 $\mathcal H(\pi_\beta)$（標「τ 低於此線時 α → ∞」）。$\alpha^*=\infty$ 時讀數顯示「α → ∞：τ 不大於 H(π̂β) = 1.161」，Q 值軸改畫 α = 1e4 的解並加註；α* 有限時 Q 值軸、μ 橫條與讀數表一律畫 $\hat Q(\alpha^*)$（等同直接設定 α = α* 的畫面），並在讀數中顯示 α*（四位小數）。
  - 快取：鍵為 `${pb}|${i}`，i 為整數步數（α 模式 `Math.round(alpha/0.05)`，τ 模式 `'t'+Math.round(tau/0.01)`）。
- **顯示**：上：四條 Q 值軸（範圍 [−1.5, 2]）：空心圓「真實 r」、實心圓「CQL 估計」；每條軸右側兩根小橫條：π̂β（實心）與 μ（框線），各附數值。中：「soft-max」數線，標 max、lse、max + log 4 三點，及「E_μ[Q] + H(μ)」讀數（與 lse 相等）。下：讀數表兩欄「真實 r」「CQL 估計」，列 E_π̂β、E_μ、差距；再一列「差距增加量」與「α·χ²(μ‖π̂β)」並排。τ 模式另加 $G(\alpha)$ 小圖。
- **讀者應觀察到**：(1) 「a4 罕見」下 α 增加時 a4 的估計 Q 從 1.40 降到 1.0 以下並低於 a1，Q 最大的動作從 a4 換成 a1；(2) $\mathbb E_{\pi_\beta}$ 下的估計 Q 永遠等於 $\mathbb E_{\pi_\beta}[r]$；(3) 差距增加量永遠等於 αχ²；(4) $\mathbb E_\mu[\hat Q]+\mathcal H(\mu)=$ lse，且 max ≤ lse ≤ max + log 4；(5) 「a4 常見」時 a4 不被壓低（α = 1 時由 1.40 略升到 1.45）；(6) τ 模式中 τ 從 1.6 降到 1.17，α* 從 0 升到約 3；τ ≤ 1.161 時 α → ∞。

---

## 5. 驗收數值表

全部取自 `verify/check_out.txt`（`python3 -I check.py`）。頁面與腳本同算法，容許誤差只涵蓋浮點加總順序：機率、Q、V 與差距 ±5e-4，$P$ ±1e-5（絕對），$\alpha_\delta$ ±5e-4。

**C1**（$r=(1.0,0.6,0.8)$）

| $\pi_\beta$ 權重 | π 權重 | α | $\pi/\pi_\beta$ | $\hat Q_1-Q$ | $\hat Q_2-Q$ | V | $\hat V_1-V$ | $\hat V_2-V$ | D（兩法） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50,30,20 | 10,20,70 | 0.5 | 0.2000, 0.6667, 3.5000 | −0.1000, −0.3333, −1.7500 | 0.4000, 0.1667, −1.2500 | 0.7800 | −1.3017 | −0.8017 | 1.60333 |
| 50,30,20 | 50,30,20 | 0.5 | 1, 1, 1 | −0.5 ×3 | 0 ×3 | 0.8400 | −0.5000 | 0.0000 | 0 |
| 50,30,20 | 10,20,70 | 0.0 | 同第 1 列 | 0 ×3 | 0 ×3 | 0.7800 | 0 | 0 | 1.60333 |
| 1,1,98 | 10,20,70 | 1.0 | 10.0000, 20.0000, 0.7143 | −10, −20, −0.7143 | −9, −19, 0.2857 | 0.7800 | −5.5000 | −4.5000 | 4.50000 |
| 50,50,1 | 90,10,0 | 1.0 | 1.8180, 0.2020, 0 | −1.8180, −0.2020, 0 | −0.8180, 0.7980, 1.0000 | 0.9600 | −1.6564 | −0.6564 | 0.65640 |

另：二動作例 $\pi_\beta=(0.5,0.5)$、$\pi=(0.9,0.1)$ 的 $D=0.64$（S5 正文）。

**C2**（MDP 見 4.C2；索引順序 (s0左, s0右, s1左, s1右, s2左, s2右)）

| 設定（π_β(右), π(右), α, γ） | V（s0,s1,s2） | $\hat V_1$ | $\hat V_2$ | $\hat V_1-\hat V_2$ | $D_{\mathrm{CQL}}$ | 高估的 (s,a) | $\hat V_2^k(s_0)$，k = 0, 1, 5, 20, 100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (0.8,0.5,0.2), 0.9, 0.1, 0.9（預設） | 5.8287, 6.5483, 7.8387 | 2.7136, 3.1798, 4.1049 | 3.7136, 4.1798, 5.1049 | −1.0000 ×3 | 0.0625, 0.6400, 3.0625 | 無 | 0, −0.0063, 0.8197, 3.1124, 3.7135 |
| 同上，α = 0 | 同上 | = V | = V | 0 | 同上 | 無 | 真實：0, 0, 1.3697, 4.9031, 5.8285 |
| 同上，γ = 0.5 | 0.3097, 0.6539, 1.6958 | −0.0364, 0.1593, 0.9122 | 0.1636, 0.3593, 1.1122 | −0.2000 ×3 | 同上 | (s1,左)：0.1618 > 0.1549 | 0, −0.0063, 0.1338, 0.1636, 0.1636 |
| (0.5,0.5,0.5), 0.5, 0.1, 0.9 | 1.0157, 1.2414, 1.9248 | 0.0157, 0.2414, 0.9248 | = V | −1.0000 ×3 | 0 ×3 | 無 | 0, 0, 0.2359, 0.8536, 1.0156 |
| 預設，γ = 0.95 | 13.3450, 14.1254, 15.4615 | 6.5902, 7.0998, 8.0568 | 8.5902, 9.0998, 10.0568 | −2.0000 ×3 | 同預設 | 無 | 0, −0.0063, 0.9726, 5.0445, 8.5316 |
| 預設，π(右) = 0.3 | 0.2060, 0.2823, 0.6546 | −1.9280, −1.6931, −1.1957 | −0.9280, −0.6931, −0.1957 | −1.0000 ×3 | 1.5625, 0.1600, 0.0625 | 無 | — |

預設的 Q：真實 (5.2458, 5.8935, 5.2458, 6.6930, 5.8935, 8.0548)；Eq. 2 (3.3923, 3.7493, 3.4223, 4.2640, 3.8493, 5.2444)。

**C3**（$\pi_\beta$ (50,30,20)、π (10,20,70)、σ = 1.0、α = 0.2）

| N | $n_a$ | $\hat D$ | $s_N$ | $P(\hat V>V)$ | $\alpha_{0.10}$ | $\alpha_{0.05}$ | $\alpha_{0.01}$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 5,3,2 | 1.60333 | 0.51023 | 0.26485 | 0.40784 | 0.52346 | 0.74030 |
| 30 | 15,9,6 | 1.60333 | 0.29458 | 0.13818 | 0.23547 | 0.30222 | 0.42741 |
| 100（預設） | 50,30,20 | 1.60333 | 0.16135 | 0.02344 | 0.12897 | 0.16553 | 0.23410 |
| 300 | 150,90,60 | 1.60333 | 0.09315 | 0.00029 | 0.07446 | 0.09557 | 0.13516 |
| 1000 | 500,300,200 | 1.60333 | 0.05102 | < 1e-6 | 0.04078 | 0.05235 | 0.07403 |
| 3000 | 1500,900,600 | 1.60333 | 0.02946 | < 1e-6 | 0.02355 | 0.03022 | 0.04274 |
| 10000 | 5000,3000,2000 | 1.60333 | 0.01613 | < 1e-6 | 0.01290 | 0.01655 | 0.02341 |

邊界：α = 0 → P = 0.50000；π = π̂β → $\hat D=0$、P = 0.5、$\alpha_\delta=\infty$；σ = 2.0、N = 100 → P = 0.16018、$\alpha_{0.05}=0.33106$；$\pi_\beta$ 權重 (1,1,98)、N = 10 → $n=(1,1,10)$（實際總筆數 12）、$\hat D=0.188$、P = 0.45244；此時按「π = π̂β」把 π 權重設為 (1,1,10) → $\hat D=0$、P = 0.5。σ = 2.0、N = 10、δ = 0.01 → $\alpha_{0.01}=1.4806$（仍在 [0, 2] 軸內）。

**C4**（$r=(1.0,0.6,0.2,1.4)$；Newton 迭代 ≤ 6 次）

| $\pi_\beta$ | α | $\hat Q$ | μ | lse（= E_μ[Q̂]+H） | $\mathbb E_{\pi_\beta}[\hat Q]$ | $\widehat{\mathrm{gap}}$ | gap | 增加量 = αχ² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a4 罕見 | 0 | 1.0000, 0.6000, 0.2000, 1.4000 | 0.2769, 0.1856, 0.1244, 0.4131 | 2.2841 | 0.7600 | −0.2315 | −0.2315 | 0 |
| a4 罕見 | 0.5 | 1.0618, 0.7055, 0.1642, 0.2128 | 0.3944, 0.2762, 0.1607, 0.1687 | 1.9922 | 0.7600 | 0.0841 | −0.0684 | 0.1526 |
| a4 罕見 | 1.0（預設） | 1.0776, 0.7475, 0.1286, −0.1162 | 0.4151, 0.2984, 0.1607, 0.1258 | 1.9568 | 0.7600 | 0.0836 | −0.0424 | 0.1260 |
| a4 罕見 | 2.0 | 1.0913, 0.7880, 0.0920, −0.4137 | 0.4295, 0.3171, 0.1581, 0.0953 | 1.9365 | 0.7600 | 0.0664 | −0.0248 | 0.0912 |
| a4 罕見 | 5.0 | 1.1054, 0.8286, 0.0564, −0.7174 | 0.4405, 0.3340, 0.1543, 0.0712 | 1.9252 | 0.7600 | 0.0387 | −0.0114 | 0.0501 |
| 均勻 | 1.0 | 0.9107, 0.7094, 0.4870, 1.0930 | 0.2723, 0.2227, 0.1783, 0.3268 | 2.2115 | 0.8000 | −0.0499 | −0.0990 | 0.0491 |
| a4 常見 | 1.0 | 0.3793, 0.5948, 0.3850, 1.4522 | 0.1621, 0.2010, 0.1630, 0.4739 | 2.1990 | 0.9600 | 0.0280 | −0.0187 | 0.0467 |
| a4 常見 | 5.0 | 0.0534, 0.5777, 0.5094, 1.4745 | 0.1189, 0.2009, 0.1876, 0.4926 | 2.1827 | 0.9600 | 0.0158 | −0.0066 | 0.0223 |

（$\max_a\hat Q$ 與 lse 的其他組合見 `check_out.txt`；α = 0 時 max = 1.4000、lse = 2.2841 ≤ 1.4 + log 4 = 2.7863。）

**C4 τ 模式**（Eq. 30 鞍點；`c4tau`）：$\mathcal H(\pi_\beta)$ 與 $G(0)$：a4 罕見 1.1611／1.5241；均勻 1.3863／1.4841；a4 常見 1.2206／1.3241。a4 罕見的 $G(\alpha)$，α = 0.5, 1, 5, 50, 1e4：1.2322, 1.1968, 1.1652, 1.1612, 1.1611。a4 罕見的 α*：τ = 2.0 → 0；1.6 → 0；1.5 → 0.0093；1.4 → 0.0676；1.3（預設）→ 0.2074；1.2 → 0.9245；1.17 → 2.982；1.15 → ∞。容許：α* 相對誤差 1e-3 或絕對誤差 5e-5（表中數值四捨五入到 4 位）。

---

## 6. 實作計畫

### 6.1 檔案與外殼

- 單一檔案 `tut/568/build/index.html`；CSS、JS、時間線資料全部內嵌，不 `fetch` 本機檔案。上線時複製為 `tutorials/conservative-q-learning.html`。
- **直接沿用** `tutorials/prediction-targets-k-diff.html`（238 參考頁；406 頁也是同一外殼）的下列部分，複製後只換內容與元件：CSS tokens 與深色模式三段覆寫、`pm-theme`、字型載入、appbar、`#pm-bar`、章節目錄、`.plate` 圖版、callout、`<details>` 檢查題、「本頁推導」與「推定」pill、表格橫捲容器、KaTeX 0.16.9 設定與 `renderMath()`、`?test` selfTest 框架（`ok()` 計數、`#selftest` 輸出、`scrollWidth <= innerWidth`）、時間線卡片列（`renderTimeline`、`TL_GROUP`）、SVG 輔助（`el`、`scaleLin`、`scaleLog`、`linePath`）、`schedule`／`debounce`／`radiogroup`。以下只列新增或變更。
- 離線截圖：無頭 Chrome 無法連 CDN，截圖時把 KaTeX 路徑替換為本機 `scratchpad/katex/0.16.9/`（只用於截圖副本，不改上線檔）。

### 6.2 新增 tokens、文案與連結

- 圖表色：沿用 `--accent`（CQL／Eq. 2 估計值）；新增 `--c-true`（light #3F6E8C／dark #8DB4D3，沿用 k-diff 頁 `--c-par` 色值，用於真實值）與 `--c-eq1`（light #6B7A3A／dark #A9BC7C，沿用 `--c-perp` 色值，用於 Eq. 1）。建置時確認在 `--surface` 上圖形對比 ≥ 3:1、文字 ≥ 4.5:1。
- 「作者認為」引述樣式沿用 406；可收合推導 `<details class="advanced">`（摘要列「推導細節」）用於 S3 逐格微分、S7 Lagrangian、S9 Cauchy–Schwarz。
- 用詞：「保守（conservative）」「下界（lower bound）」「行為策略（behavior policy）」「分佈外動作（OOD action）」「權衡係數 α」「常見錯誤」；專有名詞第一次出現附英文。不用比喻、反問句、宣傳形容詞。
- 站內連結：Hero「在關聯圖中查看」→ `../index.html#paper-568`；arXiv `https://arxiv.org/abs/2006.04779`；筆記 → `<slug>.html`。

### 6.3 程式結構（單一 IIFE）

- 共用數學：4.0 列出的函式；另 `erfc(x)`（`erfcc` 係數）、`Z_DELTA={0.10:1.2816,0.05:1.6449,0.01:2.3263}`。
- C1：`initC1()`，`state={wb:[50,30,20], wp:[10,20,70], alpha:.5, preset:'a3'}`，`c1Calc(state)`（回傳 4.C1 的所有量），`renderC1()`。
- C2：`initC2()`，`state={pbR:[.8,.5,.2], pR:.9, alpha:.1, gamma:.9, k:100}`；常數 `MDP_T`、`MDP_R`；`c2Calc(state)` → `{Q,Q1,Q2,V,V1,V2,D,it0,it1,it2}`（迭代只在 π、π_β、α、γ 改變時重算；拖動 k 只移動指標線）；`niceAxis(min,max)`（外擴 10%、1／2／5 刻度，C2 的三張自動軸共用）。
- C3：`initC3()`，`state={N:100, sigma:1, alpha:.2, delta:.05, wb, wp}`；`c3Calc(state, N)` → `{n, total, pbh, D, s, P, adelta}`；7 條曲線各 201 個 α 點（[0, 2]），每次輸入重算（閉式，可忽略）；「π = π̂β」按鈕寫入 `wp = c3Calc(state, state.N).n`。
- C4：`initC4()`，`state={pb:'rare', mode:'alpha', alpha:1, tau:1.3}`，`PB4` 常數，`c4Solve(pb, alpha)`（阻尼 Newton，4×4 用共用 `solve`）、`gapReg(pb, alpha)`、`c4Tau(pb, tau)`（二分，同 `check.py` 的 `c4tau`）；快取鍵為整數步數（4.C4）。$G(\alpha)$ 小圖取 61 個對數等距 α 點，只在換預設組時重算。
- 時間線：沿用卡片列；`TL_GROUP` 見 6.4；`year` 在本頁 JSON 中是數字，排序仍用 `Number(year)`。
- `selfTest()` 新增斷言（容許誤差見第 5 節）：
  - C1 預設：$\hat V_1-V=-1.3017$、$\hat V_2-V=-0.8017$、`dcql` 與 `chi2` 相差 < 1e-12；二動作例 D = 0.64。
  - C2 預設：$\hat V_1-\hat V_2=-1.0000$（三格，1e-9）；$\hat V_2(s_0)=3.7136$；γ = 0.5 時 (s1,左) 為唯一高估格；迭代 $\hat V_2^{100}(s_0)=3.7135$。
  - C3：N = 100 時 P = 0.02344、$\alpha_{0.05}=0.16553$；N = 1000 時 $\alpha_{0.05}=0.05235$；π = π̂β 時 P = 0.5。
  - C3：權重 (1,1,98)、N = 10 時按鈕後 $\hat D=0$、P = 0.5。
  - C4：「a4 罕見」α = 1 時 $\hat Q_4=-0.1162$、$\mathbb E_{\pi_\beta}[\hat Q]=0.7600$、增加量 = αχ² = 0.1260（兩者相差 < 1e-6）、lse = $\mathbb E_\mu[\hat Q]+\mathcal H$（1e-9）；τ = 1.3 時 α* = 0.2074、τ = 1.15 時 α* = ∞。
  - 動態文字不含 U+0300–U+036F 組合字元（掃描所有讀數節點與 `aria-*` 屬性）。
  - 輸出一行 `selfTest: N passed, M failed`。

### 6.4 研究筆記與時間線

- 來源 `tut/568/notes/timeline.json`（11 筆，欄位 `year`（數字）、`title`、`authors_short`、`arxiv`、`role`、`source`），原樣貼進 `<script type="application/json" id="timeline-data">`。本文 2006.04779 已在 JSON 中，以 `--accent` 外框與「本文」標籤突出。
- `TL_GROUP`（以 arXiv ID 為 key；不在表中歸「其他」並在建置回報中列出）：

  | 類別 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | 線上 RL 基礎 | 1312.5602、1710.10044、1801.01290、1802.09477 | `rl-mdp-q-learning-basics.html` |
  | 策略限制 | 1812.02900、1906.00949、1910.00177、1911.11361 | `cql-vs-policy-constraint-methods.html` |
  | 分佈偏移與綜述 | 2005.01643 | `offline-rl-distribution-shift.html` |
  | 基準 | 2004.07219 | `cql-experiments-d4rl-atari.html` |
  | 本文 | 2006.04779 | `cql-lineage-hub.html` |

- S13 延伸閱讀（依總覽閱讀順序，附 frontmatter `title`）：`cql-lineage-hub`、`rl-mdp-q-learning-basics`、`offline-rl-distribution-shift`、`cql-conservative-policy-evaluation`、`cql-algorithm-variants-and-guarantees`、`cql-vs-policy-constraint-methods`、`cql-experiments-d4rl-atari`。

### 6.5 數學排版

- 公式右側以等寬字標論文編號（「Eq. 13」）。Thm. 3.1、3.2、Eq. 6 在 < 600px 時拆成多行（`aligned`）；`.katex-display` 可橫捲。
- 第 1 節以外不得新增論文數字；頁面數字須與第 1、5 節一致。

### 6.6 建置後自我檢查

沿用 406 的流程（390px 時特別檢查 C2 四張圖的堆疊、C3 兩圖的上下排列、可橫捲容器的鍵盤操作）：390／768／1024／1440 截圖＋深色、鍵盤 focus、reduced-motion；逐項評分（字體、留白、視覺層級、色彩、動態、微互動、響應式、原創性）並修正到沒有可見的改進空間；`?test` 全數通過；逐條核對 12 個學習目標的答案出現在指定章節正文；逐節核對第 3 節「本節先定義」的詞在第一次使用前已定義；本頁沒有自動播放的動畫（C2 的迭代以滑桿 k 檢視，不自動播放）。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 教學用自訂數值 | C1／C3 的 $r$、C2 的 MDP、C4 的 $r$ 與 $\pi_\beta$ 預設組都是自訂；論文沒有表格式範例。 | 每個圖版標「推定」pill 與「教學用自訂例子，論文沒有給」。 |
| R2 | C3 的高斯噪聲模型 | 論文的 $C_{r,T,\delta}$ 無數值；高斯噪聲與閉式 $P$ 是本頁設定，只重現 $1/\sqrt N$ 的結構。 | 明寫「不重現論文的門檻常數」；α 門檻以 $\alpha_\delta$ 命名，不稱為「論文的 α」。 |
| R3 | 論文內部不一致與筆誤 | 取樣誤差界三種寫法（主文 $C/\sqrt{\lvert\mathcal D\rvert}$、定理 $CR_{\max}/((1-\gamma)\sqrt{\cdot})$、App. D.3）；δ 同時表示失敗機率與零計數的大值；Thm. 3.2 門檻主文用狀態計數、App. C 用 $(s,a)$ 計數；Thm. 3.1「α > 0」vs App. C「α ≥ 0」；App. C 把 Thm. 3.2 的證明標成「prove Theorem 3.3」、把 Thm. 3.1 稱「Proposition 3.2」、Eq. 12 第一個 $\mathcal B$ 應為 $\hat{\mathcal B}$；Thm. 3.3 證明省略 α；Thm. 3.4 證明的正負號；App. A 的 $+D_{\mathrm{KL}}$；Thm. D.2 缺條件；Thm. D.3 寫 $\hat V^{k+1}$ 應為 $V^{k+1}$；Thm. D.4 第一行漏 $1/(1-\gamma)$。 | 頁面採正確版本並在 S3／S8／S9 的可收合推導中註一句「論文此處符號有誤」；不逐條列出全部筆誤（S13 一行帶過）。 |
| R4 | 表格的疑似錯誤 | Table 1 的 walker2d-medium-expert 與 walker2d-mixed 五個 baseline 完全相同（1.9, 11.3, 10.8, −0.3, 0.9）；Table 7 的 hopper-mixed 1866.1 與 hopper-medium 相同、與 Table 6 的 1563.2 不同；Table 3 圖說寫 setting (1)、正文說 (2)；Asterix (1%) 文字層錯位。 | 頁面照錄並在表下註「原文如此」；Table 3 用正文的 setting (2)；Asterix (1%) 依 PDF 渲染圖為 QR-DQN 166.3、REM 386.5、CQL(H) 592.4。 |
| R5 | 倍數宣稱 | 摘要「2–5x」（「especially … complex and multi-modal」，未限定混合資料）、Sec. 7「In many cases … 2-5x」、Gym「2–3x」、Adroit「2–9x」、Atari「36x、6x」都來自特定任務；單一策略資料只是「small margin」；9 列混合資料中 1 列輸（hopper-medium-expert）、1 列接近持平（halfcheetah-mixed）、1 列只是 1.2 倍的明確勝出（halfcheetah-medium-expert）。 | Hero 卡 3 照錄摘要與 Sec. 6 的兩種倍數；S14 標題用「多數混合資料集」並逐列寫出這 3 列。 |
| R6 | Fig. 1 與遊戲數 | 正文說 5 個遊戲，Fig. 1 只有 4 個；Seaquest 後的「∗」沒有註腳。 | 只寫「Fig. 1 的四個遊戲」，不提 ∗。 |
| R7 | 超參數敘述不一 | 策略學習率 §4 寫 3e-5、App. F 先寫「3e-5 or 1e-4」後定為 3e-5；Q 學習率「3e-4 or 1e-4」，預設 3e-4；AntMaze 的 τ 未列；Table 7 圖說「generally comparable」vs 正文「Lagrange 在所有任務較好、AntMaze 差可達 30%」。 | 依 App. F 的「預設」敘述；兩處原文都照錄。 |
| R8 | 理論與實作的落差 | supp 條件、μ = π、無取樣誤差版的 Thm. 3.3／3.4、NTK 假設（步驟 14）。 | S13 明列；S7、S8 各一句提醒。 |
| R9 | 引用編號 | 論文正文把 SAC 標為 [21]，但參考文獻 [21] 是 Haarnoja et al. 2017（deep energy-based policies）。 | 頁面以 arXiv 1801.01290（timeline 中的 SAC）連結，不引用 [21] 編號。 |
| R10 | 初學者門檻 | 讀者不熟 Bellman 算子與 $(I-\gamma P^\pi)^{-1}$，以及熵、KL、TV、重要性取樣等工具。 | S1 以 2 狀態的小例子說明元素非負；第 3 節「本節先定義」欄逐節列出必須先定義的詞；S10、S12 是全頁最抽象的兩節，各配一個例子（C4 讀數、ζ 數值例）。 |
| R11 | 元件數值極端值 | C1 的 π/π̂β 最大 201；C3 $P$ 下溢；C4 α = 5 時梯度精度。 | 軸上邊界釘點；$P<10^{-6}$ 文字化；Newton 停止條件 1e-9（4.C4）。 |
| R12 | 相對連結 | 建置位置與上線位置不同，預覽時站內連結失效。 | 接受；預覽只檢查外部連結與頁內錨點。 |
| R13 | C4 與 Thm. 3.4 的關係 | C4 是 γ = 0、自洽 μ 的駐點類比，Thm. 3.4 的「α 夠大」條件在其中自動成立。 | C4 標題下 pill 明寫「不是定理本身」；S11 以讀數為例時重複這句。 |
| R14 | τ 模式的機制 | 「τ ≤ H(π̂β) 時 α → ∞」是 bandit 上的凸共軛結果，論文只有 App. F 的經驗觀察（τ = 2.0 時 α 到百萬量級）。 | 標〔本頁推導〕，並寫明「機制示意，不是論文的分析」；不把 1.161 與論文的 τ = 2.0 對應。 |

---

## 8. 修訂紀錄（round 1，對應 `reviews/round1.md` 的編號）

| # | 變更 |
| --- | --- |
| 1 | 步驟 7 改寫：α 不顯式出現，但經 $\pi_{\hat Q^k}$ 隱含進入，α 大時 ε 須更小；學習率連結改標「作者認為」並註明 SAC 策略不是 softmax(Q)。 |
| 2 | 1.1 照錄摘要（2–5x，複雜多模態資料）與 Sec. 6（Gym 2–3x、單一策略 small margin）；Hero 卡 3、S14 標題改寫；步驟 13 加 9 列混合資料的逐列比對（8 列最高，3 個弱例）。 |
| 3 | 1.1 與 Hero 卡 2 加「α 夠大（無取樣誤差時任何 α > 0）、π̂β > 0」。 |
| 4 | 步驟 13 的 Table 4 改為「範圍 2.47e3–26.03e12；只有 hopper-medium 非單調」。 |
| 5 | 步驟 2、LO2 加「有取樣誤差時需 μ 在每個動作上 > 0」。 |
| 6 | 步驟 8、C4、S11 標明「γ = 0 駐點類比」；步驟 8 揭露 $[\mathcal B\hat Q^k-Q^k]$ 應為 $Q^{k+1}$ 與 $\hat Q^k$／$\hat Q^{k+1}$ 的不一致。 |
| 7 | LO9 答案含 Thm. 3.3 不等式、「無取樣誤差」與「作者認為」。 |
| 8 | LO3 答案加 Eq. 13 的三個假設。 |
| 9 | 步驟 4、LO5 加「表格式、μ = π、全支撐的最壞 π」。 |
| 10 | 步驟 9 加「μ = π、在 $\hat M$ 中的不動點」；步驟 10 加 Assumption D.1 與 $\hat M$ 的取樣模型。 |
| 11 | C3 的教的概念與 pill 改為「只重現 1/√N 與 1/D̂ 兩個因子，不是定理門檻」。 |
| 12 | 步驟 11 補 Thm. D.1 分母條件半句。 |
| 13 | 第 2 節開頭加規則：〔本頁推導〕與「作者認為」標記在頁面答案中保留。 |
| 14 | 符號表加表格式、不動點、熵、KL、softmax、Lagrangian、D_TV、d^π、M̂、重要性取樣；大綱加「本節先定義」欄，涵蓋 off-policy、actor-critic、SAC、QR-DQN、twin-Q、Lagrange 對偶梯度、D4RL 正規化分數、BC。 |
| 15 | 比值改寫為 π/π̂β（程式 `ratio`）；ρ 只指 KL 先驗；Eq. 5 的狀態分佈寫 $\rho_0(s)$ 並註論文寫法。 |
| 16 | S3 只陳述 $\hat{\mathcal B}=\mathcal B$ 版的 Thm. 3.1；完整 Thm. 3.1／3.2 移到 S7；符號表的首次出現欄更新。 |
| 17 | 原 S8 拆為 S10（Thm. 3.3）與 S11（Thm. 3.4，以 C4 讀數為例）；S12 加 ζ 數值例（推定）。 |
| 18 | 原 S4 拆為 S4（Eq. 13＋C1）與 S5（D_CQL、Thm. 3.2、ν）；原 S7 拆為 S8（CQL(R)／CQL(H)＋定義）與 S9（CQL(ρ)＋C4）。全頁 S0–S16，LO 章節對應同步更新。 |
| 19 | C3「π = π̂β」改設 π 權重 = $n_a$；N 改稱「目標筆數」並列實際總筆數；第 5 節與 selfTest 加 (1,1,98)、N = 10 的按鈕例（D̂ = 0）。 |
| 20 | C3 α 滑桿與左軸改 [0, 2]，α_δ > 2 時釘邊標示；右圖 log 軸自動取範圍並夾在 [1e-3, 1e2]；驗收加 σ = 2、N = 10、δ = 0.01 的 α_δ = 1.4806。 |
| 21 | C2 加折線圖 B「與真實迭代的差」；三張圖縱軸自動（`niceAxis`）；右端同標 k = 100 值與不動點；觀察 (5)(6) 改寫。 |
| 22 | 4.0 加窄螢幕規則（單欄堆疊、SVG 寬度隨容器、可橫捲容器 `tabindex="0"`、Eq. 2–4、6、30 拆行）；C2、C3 寫明兩種寬度的排列。 |
| 23 | C1 預設只露出 α 與 π 預設組；C2 只露出 α、γ、k 與預設組；其餘收進「進階設定」。 |
| 24 | 4.0 規定動態文字不用組合字元與下標數字（`<span class="hat">`、SVG 文字標籤、中文 aria）；selfTest 掃描 U+0300–U+036F。 |
| 25 | Hero 標題改「保守 Q 學習：壓低離線資料外動作的 Q 值」；S8 標題改「以 Q 的 softmax 作為被壓低的分佈」；常見錯誤「混比表格」移到 S14 表格之後。未採納「12／18 列」寫法（見 round1.md）。 |
| 26 | C4 快取鍵改整數步；4.0 加 pattern 用 token、截斷箭頭記號；C4 加「以 τ 決定 α」模式（Eq. 30 鞍點、$G\ge\mathcal H(\pi_\beta)$，〔本頁推導〕），`check.py` 新增 `c4tau` 與驗收數值；時間線標「選讀」。未採納為 LO5、LO12 加元件（見 round1.md）。 |

## 9. 修訂紀錄（readiness）

- B1：符號表加 $\mathcal B^*$（S1 定義，S8／S13 回指）。
- B2：Asterix (1%) 以 `src/paper.pdf` 第 9 頁渲染圖確認：QR-DQN 166.3、REM 386.5、CQL(H) 592.4；步驟 13 與 R4 一致。
- 非阻擋項：$d^{\pi_\beta}$ 首次出現改 S2；R5 改為「1 輸、1 持平、1 列 1.2 倍勝出」；C2 轉移機率逐項寫明；C3 內嵌 erfcc 係數；步驟 13 內嵌 Table 1 九列混合資料全欄；S8 定義 log-sum-exp；$G_Q$／$G_\pi$ 定義；C2 k 滑桿步長 1；C4 τ 模式 α* 有限時畫 $\hat Q(\alpha^*)$；6.1 改為明列沿用 k-diff 頁外殼的部分（建置說明指定 k-diff 頁）。


## 10. 建置紀錄（build round）

- 外殼取自 `tutorials/prediction-targets-k-diff.html`（CSS tokens、appbar、`#pm-bar`、章節目錄、plate／callout、`details.check`、KaTeX 設定、`?test` 框架、時間線元件）；移除 k 軸樣式，新增 `--c-true`／`--c-eq1`（以 var 指到 `--c-par`／`--c-perp`，深色模式自動切換）、`details.advanced`、`.concl`、Q 值軸與 C2 pattern 的 SVG 類別。
- 與設計的差異：分段按鈕沿用外殼的 `aria-pressed` 按鈕（未改 radiogroup）；C4 的資料比例／μ 橫條放在讀數表內（不在 Q 值軸右側）；讀數中的帽子符號改寫為「估計 V」等文字，不用 hat span（mono 字型下位置不穩）；C3 右圖的 ∝1/√N 參考線與資料點完全重合，改以寬半透明線畫在點下方並在圖例註明。
- 驗收：`?test` 49 passed, 0 failed（1440 與 390 寬）；`impeccable detect --no-advisory` 0 項（16 項 numbered-section-labels 為 advisory）；`extract_page_text.py` 後 12 個 LO 的關鍵詞與數字皆在 `accept/page.txt`；無主控台錯誤；截圖見 `build/caps/`。
