# 設計文件：〈Offline Reinforcement Learning with Implicit Q-Learning〉互動教學

- 論文：Ilya Kostrikov、Ashvin Nair、Sergey Levine，arXiv 2110.06169（v1，2021-10-12）。資料庫編號 No. 569。方法簡稱 IQL。程式碼：`https://github.com/ikostrikov/implicit_q_learning`（論文註腳 1）。
- 原文：`tut/569/src/paper.txt`（`tut/` = `scripts/tutorial_src/`）。本文件中的「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」「Alg. 1」皆指論文原文編號。
- 產出：`tut/569/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/implicit-q-learning.html`（slug `implicit-q-learning`），站內相對連結一律以上線位置為準。
- 驗算腳本：`tut/569/verify/common.py`（expectile 閉式解、mulberry32，含自我檢查）、`numbers.py`（C1、C2、C4、Fig. 1 中圖）、`maze.py`／`maze_noise.py`／`maze_pop.py`（C3）、`teleport.py`（R2）。輸出存為同名 `*_out.txt`。第 5 節驗收表全部取自這些輸出。
- 研究筆記：`tut/569/notes/`（總覽 `iql-lineage-overview` + 5 篇主題筆記 + `timeline.json` 14 筆）；另引用 No. 568 的 `rl-mdp-q-learning-basics`、`offline-rl-distribution-shift`（位於 `tut/568/notes/`）。
- 文件慣例：本設計自行推導、論文沒有寫出的結論標「〔本頁推導〕」；為了玩具元件而選定、論文沒有給出的數值標「推定」。頁面用同樣兩種 pill。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

Offline RL 只用固定資料集學策略，不能再與環境互動。標準的 Q-learning 目標（Eq. 1）含 $\max_{a'}Q_{\hat\theta}(s',a')$，必須查詢資料中沒有出現的動作；這些動作的 Q 值可能有誤，取最大值時常導致高估（Sec. 3）。先前方法用策略約束或值函數正則化處理這個問題。IQL 改為**完全不查詢資料外的動作**：把「狀態價值」看成隨資料集動作而變的隨機變數，用 **expectile regression**（非對稱 L2 損失）估計它在 $a\sim$ 資料分布下的上 expectile（Eq. 5），再用這個 $V_\psi$ 做 SARSA 形式的 MSE 備份訓練 $Q_\theta$（Eq. 6）。$\tau=0.5$ 時就是 SARSA（估計行為策略的值），$\tau\to1$ 時趨近「只在資料支撐內取最大值」的 Q-learning（Theorem 3）。最後用 advantage-weighted regression（Eq. 7）從資料動作中抽出策略，同樣不查詢資料外動作。IQL 在 D4RL 的 Ant Maze 總分 378.0（CQL 303.6），locomotion 總分 692.4（CQL 698.5），訓練 20 分鐘（作者以 JAX 重寫的 CQL 為 80 分鐘，原始 CQL 實作超過 4 小時；Table 1、Sec. 5.2），並可接續做 online 微調（Table 2）。

### 1.2 研究動機（Sec. 1–2）

1. 提升策略需要估計「非資料動作」的值，這與分布偏移（distributional shift）互相衝突；先前方法以約束策略（BCQ、BEAR、BRAC、AWAC、TD3+BC、CRR）或壓低 OOD 動作值（CQL、Fisher-BRC）處理，造成「策略改進幅度」與「誤估風險」的權衡（Sec. 1）。
2. 作者觀察：在分布內的約束不一定足以避免值函數外插（Sec. 1；App. D 以 BCQ 為例：生成模型仍可能產生資料外動作）。
3. 單步（single-step）方法只學行為策略的 $Q^{\pi_\beta}$ 再取一次貪婪策略（Onestep RL、AWR），或完全不用值函數（Decision Transformer）。它們也不查詢 OOD 動作，在 locomotion 上表現好，但在需要「拼接」（stitching）次佳軌跡的 Ant Maze 上很差（Sec. 2、5.1）。
4. 問題：能否只用資料內的 $(s,a)$，同時保有多步動態規劃（multi-step dynamic programming）？

### 1.3 貢獻（Sec. 1、6）

1. IQL：不查詢 OOD 動作、但做多步動態規劃的 offline RL 演算法；作者寫「據我們所知，第一個同時具備這兩項特性的方法」（Sec. 6）。
2. 實作只改 SARSA 式 TD 的損失函數，多一個用非對稱 L2 損失訓練的 critic $V_\psi$（Abstract）。
3. 理論：$\tau\to1$ 時 $V_\tau$ 收斂到資料支撐內的最佳值（Lemma 1、Lemma 2、Corollary 2.1、Theorem 3）。
4. D4RL 上的 state-of-the-art 結果，Ant Maze 大幅領先；計算快（1M 次更新在一張 GTX1080 上少於 20 分鐘，Sec. 6）；適合 online 微調（Sec. 5.3）。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S1 末尾以可收合表格提供；每章開頭另有「本章符號」小表）

| 符號 | 意義 | 首次使用 |
| --- | --- | --- |
| $s,a,s'$ | 狀態、動作、下一個狀態 | S1 |
| $r(s,a)$、$\gamma$ | 獎勵、折扣因子 | S1 |
| $p(s'\mid s,a)$ | 環境轉移（dynamics） | S1 |
| $\pi(a\mid s)$、$\pi_\beta$ | 策略；收集資料的行為策略（behavior policy） | S1 |
| $\mu(a\mid s)$ | 資料中狀態 $s$ 下動作 $a$ 的比例。Sec. 4.4 使用但未定義；本頁解讀為與 $\pi_\beta$ 同一個分布（頁面標「本頁解讀」） | S3 |
| $Q^{\pi_\beta}$ | 行為策略的 Q 值（Eq. 2–3 的解） | S2 |
| $\mathcal A_\beta(s)$ | 資料支撐內的動作集合 $\{a:\pi_\beta(a\mid s)>0\}$ | S2 |
| $Q^*$ | 限制在資料支撐內的最佳 Q（Corollary 2.1） | S5 |
| $V^*$ | $V^*(s)=\max_{a:\pi_\beta(a|s)>0}Q^*(s,a)$，資料支撐內的最佳狀態值（本頁記號；C3 中支撐為全部動作時即一般的最佳值） | S5 |
| $V_\tau,Q_\tau$ | Eq. 5、6 的母體最佳解（Sec. 4.4 遞迴） | S5 |
| $Q^1,Q^2$ | clipped double Q 的兩個 Q 網路 | S8 |
| $\mathcal D$ | 固定的資料集，元素為 $(s,a,r,s',a')$ | S1 |
| $Q(s,a)$、$V(s)$ | 狀態-動作價值、狀態價值 | S1 |
| $\theta,\hat\theta$ | Q 網路參數、target network 參數（Polyak 平均） | S1 |
| $\psi$、$\phi$ | V 網路參數、策略網路參數 | S4、S7 |
| $\tau$ | expectile 的階數，$\tau\in(0,1)$ | S3 |
| $L_2^\tau(u)$ | 非對稱平方損失 $\lvert\tau-\mathbb 1(u<0)\rvert u^2$ | S3 |
| $m_\tau$、$\mathbb E^\tau$ | 隨機變數的 $\tau$-expectile | S3、S5 |
| $\beta$ | AWR 的反溫度（inverse temperature） | S7 |
| $A(s,a)=Q(s,a)-V(s)$ | 優勢（advantage） | S7 |
| $\alpha$ | target network 的 Polyak 係數（0.005） | S8 |

#### 步驟 1：offline RL 與 Eq. 1 的問題（Sec. 3）

MDP $(S,A,p_0,p,r,\gamma)$，目標 $\pi^*=\arg\max_\pi\mathbb E_\pi[\sum_t\gamma^tr(s_t,a_t)]$。常見的 offline 方法沿用 TD 誤差：

$$L_{TD}(\theta)=\mathbb E_{(s,a,s')\sim\mathcal D}\big[(r(s,a)+\gamma\max_{a'}Q_{\hat\theta}(s',a')-Q_\theta(s,a))^2\big]\tag{Eq. 1}$$

策略 $\pi(s)=\arg\max_aQ_\theta(s,a)$。論文的說法：資料外的動作 $a'$ 可能給出錯誤的 $Q_{\hat\theta}(s',a')$，而策略被定義成取估計 Q 的最大值，常導致高估（Sec. 3）。本頁補充的解釋（標「本頁解讀」）：這些 $(s',a')$ 不在資料中，損失從不比較它們，錯誤因此不會被修正；取最大值時結果多半落在高估的值上。頁面在此說明「分布偏移」一詞，連到 `offline-rl-distribution-shift.html`。

S1 的教學補強（頁面）：TD 目標一行數值例（$r=1,\gamma=0.9,\max_{a'}Q=5\Rightarrow$ 目標 5.5）；target network 一行說明（目標中的 Q 用參數緩慢跟隨 $\theta$ 的複本 $\hat\theta$，避免目標隨每次更新跳動）；靜態小圖（SVG 360×200）：一個狀態的 6 個動作，真實 Q 皆為 1，資料中的 3 個動作估計值接近 1，資料外 3 個動作的估計誤差為 +1.5、−0.8、+0.4，max 取到 2.5（標「取最大值落在高估的值上」）。這是示意數值（推定）。

#### 步驟 2：SARSA 只用資料動作，但只學到行為策略（Sec. 4，Eq. 2–4）

$$L(\theta)=\mathbb E_{(s,a,s',a')\sim\mathcal D}\big[(r(s,a)+\gamma Q_{\hat\theta}(s',a')-Q_\theta(s,a))^2\big]\tag{Eq. 2}$$

$a'$ 取自資料，因此不查詢 OOD 動作。MSE 擬合目標的平均，在容量無限、沒有取樣誤差時

$$Q_{\theta^*}(s,a)\approx r(s,a)+\gamma\,\mathbb E_{s'\sim p(\cdot|s,a),\,a'\sim\pi_\beta(\cdot|s')}[Q_{\hat\theta}(s',a')]\tag{Eq. 3}$$

（原文 Eq. 3 下標寫 $a'\sim\pi_\beta(\cdot|s)$，應為 $s'$；頁面寫 $s'$，並在 R3 記錄。）這是 $Q^{\pi_\beta}$。先前單步方法（Brandfonbrener et al. 2021、Peng et al. 2019）學 $Q^{\pi_\beta}$ 後訓練策略最大化它；locomotion 上可行，在需要多步動態規劃的任務很差。IQL 想學的是資料支撐內的最大值：

$$L(\theta)=\mathbb E_{(s,a,s')\sim\mathcal D}\Big[\big(r(s,a)+\gamma\max_{a'\in A,\ \pi_\beta(a'|s')>0}Q_{\hat\theta}(s',a')-Q_\theta(s,a)\big)^2\Big]\tag{Eq. 4}$$

下一步的問題：不能直接對資料中「沒出現的那些 $a'$」取 max，也不想訓練生成模型去取樣候選動作（App. D 指出 BCQ 的生成模型仍可能產生 OOD 動作）。

#### 步驟 3：expectile regression（Sec. 4.1，Fig. 1）

隨機變數 $X$ 的 $\tau\in(0,1)$ expectile 定義為非對稱最小平方問題的解：

$$m_\tau=\arg\min_{m}\ \mathbb E_{x\sim X}[L_2^\tau(x-m)],\qquad L_2^\tau(u)=\lvert\tau-\mathbb 1(u<0)\rvert\,u^2$$

- $\tau=0.5$：$L_2^{0.5}(u)=\tfrac12u^2$，解為平均值（MSE）。$\tau>0.5$：降低 $x<m_\tau$ 的權重、提高 $x>m_\tau$ 的權重（Fig. 1 左）。例：$\tau=0.9$ 時 $u=+0.5$ 的損失 $0.225$，$u=-0.5$ 的損失 $0.025$，比為 9。
- 條件版本：$\arg\min_{m_\tau(x)}\mathbb E_{(x,y)\sim\mathcal D}[L_2^\tau(y-m_\tau(x))]$；可用 SGD 最佳化，梯度不偏（Sec. 4.1）。Fig. 1 右：$\tau\approx1$ 近似每個 $x$ 下 $y$ 的支撐內最大值。
- 與 quantile regression（非對稱 $\ell_1$）相關；作者選 expectile 是因為只需改 MSE，且實驗上比 quantile 略好（註腳 2）。
- 〔本頁推導〕一階條件：$\tau\,\mathbb E[(X-m)_+]=(1-\tau)\,\mathbb E[(m-X)_+]$，等價於 $m_\tau$ 是加權平均 $m_\tau=\mathbb E[wX]/\mathbb E[w]$，其中 $x\ge m_\tau$ 的權重 $w=\tau$、$x<m_\tau$ 的權重 $w=1-\tau$。所以 $m_\tau$ 一定落在支撐的最小值與最大值之間，且機率為 0 的值完全不影響 $m_\tau$。這是頁面 C1 的閉式算法與「OOD 動作不影響」的依據。
- 〔本頁推導〕標準常態的 expectile：$m_{0.9}=0.8616$、$m_{0.99}=1.7174$（對稱：$m_{0.1}=-0.8616$），對應 Fig. 1 中圖的五條線（`numbers_out.txt`）。

#### 步驟 4：為什麼另外學一個 V——轉移的隨機性（Sec. 4.2，Eq. 5–6）

直接把 Eq. 2 的平方改成 $L_2^\tau$：$L(\theta)=\mathbb E[L_2^\tau(r+\gamma Q_{\hat\theta}(s',a')-Q_\theta(s,a))]$。缺點：目標值的隨機性同時來自動作 $a'$ 與環境轉移 $s'\sim p(\cdot|s,a)$。大的目標值可能只是偶然轉移到高回報狀態的樣本（原文 “lucky” sample），而不代表存在一個能達到該值的動作（Sec. 4.2）。解法是拆成兩個損失：

$$L_V(\psi)=\mathbb E_{(s,a)\sim\mathcal D}\big[L_2^\tau(Q_{\hat\theta}(s,a)-V_\psi(s))\big]\tag{Eq. 5}$$
$$L_Q(\theta)=\mathbb E_{(s,a,s')\sim\mathcal D}\big[(r(s,a)+\gamma V_\psi(s')-Q_\theta(s,a))^2\big]\tag{Eq. 6}$$

- Eq. 5 只對動作取 expectile（$Q$ 已對轉移取過平均）；Eq. 6 用 MSE，把轉移的隨機性平均掉。
- 兩式都不需要顯式策略，也只用資料中的動作（Sec. 4.2）。Eq. 5 用 target network $Q_{\hat\theta}$；Eq. 6 的 $V_\psi(s')$ 不用 target。
- 〔本頁推導〕C2 的單狀態例子：$a_1$ 的目標恆為 5；$a_2$ 以機率 $p$ 得 10、否則 0；資料中 $a_2$ 佔比 $q$。單一損失對每個 $(s,a)$ 各擬合一個 $Q$，所以 $Q_{\text{單一}}(a_2)$ 是 $\{10:p,\ 0:1-p\}$ 的 $\tau$-expectile，$\tau\to1$ 時對任何 $p>0$ 都趨近 10；IQL 的 Eq. 6 用 MSE，$Q(a_2)=10p$。狀態值都用 Eq. 5 的形式對動作取 expectile：單一損失版 $V=\mathbb E^\tau\{5:1-q,\ Q_{\text{單一}}(a_2):q\}$，IQL 版 $V=\mathbb E^\tau\{5:1-q,\ 10p:q\}$，後者 $\tau\to1$ 時趨近 $\max(5,10p)$（資料中最好動作的期望值）。$p=0.3,q=0.5,\tau=0.9$：$Q_{\text{單一}}(a_2)=7.9412$（IQL 3），$V$ 單一損失版 7.6471、IQL 4.8000、參考值 5。$p\in\{0,1\}$（確定性轉移）時兩者相同。（round 0 曾把動作與轉移併成一個分布 $\{5,10,0\}$，該做法不是論文的逐 $(s,a)$ 擬合，已捨棄。）

#### 步驟 5：理論——從 SARSA 到 Q-learning 的一條連續譜（Sec. 4.4，App. A）

記號：$\mathbb E^\tau_{x\sim X}[x]$ 為 $X$ 的 $\tau$-expectile。Eq. 5、6 的最佳解遞迴定義為

$$V_\tau(s)=\mathbb E^\tau_{a\sim\mu(\cdot|s)}[Q_\tau(s,a)],\qquad Q_\tau(s,a)=r(s,a)+\gamma\,\mathbb E_{s'\sim p(\cdot|s,a)}[V_\tau(s')]$$

- **Lemma 1**：$X$ 有界、支撐上確界 $x^*$，則 $\lim_{\tau\to1}m_\tau=x^*$（證明概要：expectile 隨 $\tau$ 單調不減且以 $x^*$ 為上界）。
- **Lemma 2**：$\tau_1<\tau_2\Rightarrow V_{\tau_1}(s)\le V_{\tau_2}(s)$（App. A.1：把 $V_{\tau_1}$ 遞迴展開，逐層把 $\mathbb E^{\tau_1}$ 換成較大的 $\mathbb E^{\tau_2}$，仿照 policy improvement 的證明）。
- **Corollary 2.1**：$V_\tau(s)\le\max_{a:\pi_\beta(a|s)>0}Q^*(s,a)$，其中 $Q^*(s,a)=r+\gamma\mathbb E_{s'}[\max_{a':\pi_\beta(a'|s')>0}Q^*(s',a')]$ 是限制在資料支撐內的最佳 Q（證明：凸組合不大於最大值）。
- **Theorem 3**：$\lim_{\tau\to1}V_\tau(s)=\max_{a:\pi_\beta(a|s)>0}Q^*(s,a)$。頁面在此定義 $V^*(s)$ 為右式（本頁記號），S6 與 C3 沿用。
- 假設：這些結果針對 $V_\tau,Q_\tau$ 的母體最佳解，即論文說的「在某些假設下」（Sec. 4.2）——無限容量、無取樣誤差（Sec. 4 對 Eq. 3 的敘述），Lemma 1 另需有界支撐（獎勵有界）。論文的 Theorem 3 證明只有一行（結合 Lemma 1 與 Corollary 2.1）。有限資料時結論不保證成立：C3 的有限資料集在 τ = 1 時起點值高於 $V^*$。
- $\tau$ 越接近 1 近似越好，但最佳化越困難，所以 $\tau$ 是超參數；方法因 Theorem 3 命名為 implicit Q-learning；涵蓋 SARSA（$\tau=0.5$）到 Q-learning（$\tau\to1$）的整個譜（Sec. 4.4）。
- 〔本頁推導〕$\tau=0.5$ 時 $V(s)=\mathbb E_{a\sim\mu}[Q(s,a)]$，代入 Eq. 6 即 Eq. 3（SARSA）。
- 〔本頁推導〕表格版的遞迴可用定點迭代求解：$m_\tau(x+c)=m_\tau(x)+c$ 且 $m_\tau$ 對每個分量單調，所以 $V\mapsto\mathbb E^\tau[r+\gamma PV]$ 在 sup norm 下是 $\gamma$-壓縮，迭代收斂（適用條件：有限狀態、有界獎勵、固定的 $\hat\mu$ 與 $\hat P$；終點 G 的值固定為 0）。C3 依此實作。

#### 步驟 6：多步動態規劃的必要性——玩具迷宮（Sec. 5.1，Fig. 2）

論文設定：u 形迷宮，單一起點與單一終點；進入終點得獎勵 10，其餘 0；以機率 0.25 轉移到隨機狀態，否則到指令的狀態；資料集為 1 條最佳軌跡 + 99 條均勻隨機動作軌跡；$\gamma=0.9$；IQL 用 $\tau=0.95$。結果：單步方法（Onestep RL）的值離終點越遠衰減得越快、被雜訊主導（原文 “dominated by noise”），策略很差；IQL 的值接近真正的最佳值（Fig. 2 b–d）。

頁面的 C3 是這個實驗的表格版重建（不是論文的程式），所有自選數值標「推定」：迷宮形狀（7×8 格、36 個可走格）、「隨機狀態」解讀為「隨機方向」（R2）、軌跡長度上限 100、以表格經驗模型取代神經網路、以固定雜訊模擬策略抽取時的近似誤差。表格版結果（seed 1）：$V_{0.5}(\text{起點})=0.0007$，$V_{0.95}=0.8136$，$V^*=1.1186$；seed 1–5 的 $V_{0.5}(S)$ 都在 0.0004–0.0007、$V_{0.95}(S)$ 在 0.70–0.84（`maze_noise_out.txt`）。值的形狀與 Fig. 2 的定性結論一致。

策略的部分屬於**本頁解讀**：表格版沒有近似誤差，σ = 0 時 τ = 0.5 的貪婪策略在 seed 1 就是最佳策略（1.119），因為只需要動作之間的大小順序。論文說單步方法的策略很差、值被雜訊主導，但沒有說原因是函數近似誤差。本頁加入固定雜訊 σ 來呈現一種可能的機制：σ 大於動作之間的 Q 差距時，貪婪動作由 σ 決定。σ = 0.01 時，seed 1–5 的 τ = 0.5 起點回報為 0.000–0.051，τ = 0.9 為 0.616–1.119。σ = 0 的結果依資料而定（seed 4 所有 τ 都是 0.616）。

#### 步驟 7：策略抽取——advantage-weighted regression（Sec. 4.3，Eq. 7）

$$L_\pi(\phi)=\mathbb E_{(s,a)\sim\mathcal D}\big[\exp(\beta(Q_{\hat\theta}(s,a)-V_\psi(s)))\log\pi_\phi(a|s)\big]\tag{Eq. 7}$$

- $\beta\in[0,\infty)$ 是反溫度：小 $\beta$ 接近行為複製（behavioral cloning），大 $\beta$ 趨向 Q 的最大值（Sec. 4.3）。
- 只用資料動作，不查詢 OOD 動作；先前研究證明它等於在分布約束下最大化 Q（Peters & Schaal 2007、Peng et al. 2019、Nair et al. 2020），即「隱含的約束」（Sec. 1、2）。策略完全不影響值函數訓練，所以策略抽取可以與 TD 學習同時或之後做（Sec. 4.3）。
- 實作：指數化後的優勢截斷到 $(-\infty,100]$（App. B，沿用 Brandfonbrener et al.）。
- **符號注意**：Eq. 7 是要**最大化**的加權對數概似；Alg. 1 寫 $\phi\leftarrow\phi-\lambda_\pi\nabla_\phi L_\pi$（梯度下降）。實作時損失取 $-\mathbb E[\cdots]$（R3）。
- 〔本頁推導〕在同一個狀態內、**Q 固定時**，$\exp(-\beta V(s))$ 是所有動作共用的常數，對正規化後的策略沒有影響；若不截斷，該狀態的策略與 $V$ 無關。截斷後，$V$ 決定哪些動作的權重被壓到 100。真實 IQL 中 τ 主要透過 TD 學習（Eq. 5–6）改變學到的 Q 與 V 而影響策略；這一點 C4 不呈現（C4 的 Q 固定），頁面在 C4 旁明寫。

#### 步驟 8：演算法總結與實作細節（Alg. 1，Sec. 4.3，App. B–C）

```
初始化 ψ, θ, θ_targ, φ
TD 學習（IQL）：每個梯度步
    ψ ← ψ − λ_V ∇ψ L_V(ψ)            # Eq. 5，Q 用 min(Q¹_targ, Q²_targ)
    θ ← θ − λ_Q ∇θ L_Q(θ)            # Eq. 6，兩個 Q 網路各自擬合同一個目標
    θ_targ ← (1 − α) θ_targ + α θ    # Polyak, α = 0.005
策略抽取（AWR）：每個梯度步
    φ ← φ − λ_π ∇φ L_π(φ)            # Eq. 7（實作用負號），A 用 min(Q¹_targ, Q²_targ) − V
```
（三個學習率 $\lambda_V,\lambda_Q,\lambda_\pi$：Adam，學習率 $3\cdot10^{-4}$，actor 用 cosine schedule（App. B）；頁面在程式碼下方一行寫出。頁面程式碼區塊用 `θ_targ` 取代 θ̂，避免等寬字型的組合字元錯位；正文 KaTeX 仍寫 $\hat\theta$。）

- 兩個階段可以同時進行（online 微調時就同時進行，Sec. 4.3）。
- Clipped double Q-learning（Fujimoto et al. 2018）：維持兩個 Q 網路 $Q^1,Q^2$（參數 $\theta^1,\theta^2$，各有 target），V 更新與策略更新時取兩者的最小值（Sec. 4.3）。頁面 S8 用一段說明：兩個獨立的估計同時高估的機會較小，取 min 是抑制 S1 所說的高估的一般做法（取 min 的理由為本頁補充，論文只寫用法）。
- 超參數（App. B–C）：

| 項目 | 值 |
| --- | --- |
| Ant Maze | $\tau=0.9$，$\beta=10.0$；獎勵減 1 |
| MuJoCo locomotion | $\tau=0.7$，$\beta=3.0$；獎勵除以資料集中最佳與最差軌跡回報的差 |
| Kitchen、Adroit | $\tau=0.7$，$\beta=0.5$；策略網路 dropout 0.1 |
| 微調（僅 dexterous manipulation 設定） | $\tau=0.8$，$\beta=3.0$；offline 25000 步（App. C 的句子對 IQL 與 AWAC 是否共用此數字有歧義；頁面在此數字旁加「原文有歧義」pill，並以一句說明兩種讀法）；探索用高斯噪聲 $\sigma=0.03$。Ant Maze 微調的 τ、β 論文未另外給出 |
| 微調流程 | offline 1M 梯度步 → online 1M 環境步，每個環境步 1 次梯度更新，其他設定不變 |
| 網路 | 所有網路皆為 2 層 MLP、256 隱藏單元、ReLU |
| 最佳化 | Adam，學習率 $3\cdot10^{-4}$；actor 學習率用 cosine schedule |
| 策略 | 高斯分布，標準差與狀態無關 |
| target 更新 | soft update，$\alpha=0.005$ |
| 優勢截斷 | $\exp(\beta A)$ 截斷到 $(-\infty,100]$ |
| 評估 | locomotion：10 條評估軌跡 × 10 個 seed；Ant Maze：100 條評估軌跡 |
| 框架 | JAX + Flax |

常見錯誤（頁面 S8 逐條列出）：
1. 在 TD 目標中用策略產生的動作或對所有動作取 max：IQL 的 Eq. 5、6 只用資料中的 $(s,a)$ 與 $s'$。
2. 把 expectile 直接套在 TD 目標上（單一損失）：會把偶然轉移到高回報狀態的結果當成動作的價值（Sec. 4.2）。
3. Eq. 7 的正負號：要最大化加權對數概似，實作損失取負號。
4. 忘記截斷 $\exp(\beta A)$：App. B 截斷到 100；不截斷時大 $\beta$ 會讓少數樣本主導（C4 的有效樣本比例）。
5. 把 τ 設成接近 1：Theorem 3 說 $\tau\to1$ 近似越好，但作者指出最佳化較難（Sec. 4.4），實際用 0.7–0.9；τ 越大不一定越好。
6. Ant Maze 的獎勵要減 1、locomotion 的獎勵要正規化（App. B）。

#### 步驟 9：實驗結果（Sec. 5.2、Table 1、Fig. 3、App. B Table 3）

- Table 1（D4RL，正規化分數）：

| 總分 | BC | 10%BC | DT | AWAC | Onestep RL | TD3+BC | CQL | IQL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| locomotion-v2 | 466.7 | 666.2 | 672.6 | 450.7 | 684.6 | 677.4 | 698.5 | 692.4 |
| antmaze-v0 | 100.2 | 134.2 | 112.2 | 107.7 | 125.3 | 163.8 | 303.6 | 378.0 |
| total | 566.9 | 800.4 | 784.8 | 558.4 | 809.9 | 841.2 | 1002.1 | 1070.4 |
| runtime | 10m | 10m | 960m | 20m | ≈20m | 20m | 80m | 20m |

- Ant Maze 各列（IQL 對 CQL）：umaze 87.5 / 74.0、umaze-diverse 62.2 / 84.0、medium-play 71.2 / 61.2、medium-diverse 70.0 / 53.7、large-play 39.6 / 15.8、large-diverse 47.5 / 14.9。單步方法在 medium、large 接近 0（Onestep RL 0.3、0.0、0.0、0.0）。
- Kitchen 總分 159.8（BC 154.5、CQL 144.6）；Adroit 總分 118.1（BC 104.5、CQL 93.6）；total+kitchen+adroit 1348.3（CQL 1240.3、BC 825.9）。作者認為 Adroit 與 Kitchen 鑑別力較低，多數 RL 方法與模仿學習相近（Sec. 5.2）。
- 作者寫「所有方法都使用 “-v2” 環境」以求公平，因此 CQL 分數高於其他論文的報告值（Sec. 5.2）；但 Table 1 的 Ant Maze 列仍標為 “-v0”，頁面照錄並註明。
- Runtime：原始 CQL 實作 1M 次更新超過 4 小時，作者的 JAX 版 80 分鐘；IQL 約比它快 4 倍（Sec. 5.2）。
- Fig. 3（四格：antmaze-medium／large × play／diverse，τ = 0.9、0.7、0.5）：τ = 0.9 的回報遠高於較小的 τ，0.7 與 0.5 在這些資料集上都很低；作者結論：需要拼接的任務必須用較大的 expectile（Sec. 5.2）。論文沒有給出曲線數據，頁面只作文字描述，不寫 0.7 與 0.5 的排序。

#### 步驟 10：online 微調（Sec. 5.3、Table 2、App. C）

| 總分（offline → online 1M 步） | AWAC | CQL | IQL |
| --- | --- | --- | --- |
| antmaze-v0 | 107.7 → 108.3 | 151.5 → 231.1 | 370.1 → 473.7 |
| hand-v0（pen、door、relocate binary） | 46.7 → 103.1 | 31.5 → 9.9 | 38.1 → 124.0 |
| total | 154.4 → 211.4 | 182.8 → 241.0 | 408.2 → 597.7 |

（CQL offline total 182.8 照原文；其兩列相加為 151.5 + 31.5 = 183.0，頁面以小字註明「照原文印出」。）

- 作者給的兩個理由：offline 表現強、提供好的初始化；加權行為複製式的策略抽取先前已被證明有利於 online 改進（Nair et al. 2020）。
- 例外：pen-binary-v0 上 AWAC 44.6 → 70.3 優於 IQL 37.4 → 60.7。
- CQL 的微調數字是作者以原始實作重跑的結果，作者重現的 offline 表現較差，尤其在 medium、large Ant Maze（App. C）；因此 Table 2 的 CQL offline 起點（antmaze 總分 151.5）不能與 Table 1 的 303.6 比較。Table 2 的 IQL offline 起點（如 antmaze-umaze 86.7）也與 Table 1（87.5）不同，論文未說明（R4）。

#### 步驟 11：限制

論文沒有獨立的限制章節。頁面 S12 只列論文中明寫的取捨：$\tau$ 越接近 1 最佳化越難（Sec. 4.4）；連續動作仍需在最後抽取一個 actor（App. D）；單步與多步方法難以直接比較（Table 1 註）；Kitchen、Adroit 鑑別力低（Sec. 5.2）；CQL 的 offline 結果作者無法重現到原論文水準（App. C）。

---

## 2. 學習目標（驗收測試）

每章結尾以 `<details>` 呈現該章檢查題（題目與參考答案照搬）。**每一條參考答案都必須寫在指定章節的正文段落中**（不是只在元件、表格、圖或答案框裡）。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明 Eq. 1 為何在 offline 下出問題 | Eq. 1 的哪一項會查詢資料外的動作？為什麼會造成高估？ | $\max_{a'}Q_{\hat\theta}(s',a')$；資料外的 $a'$ 可能給出錯誤的 Q 值，而策略取估計 Q 的最大值，常導致高估（Sec. 3）。本頁解讀：損失從不比較這些 $(s',a')$，錯誤不會被修正 | S1 |
| LO2 | 區分 SARSA 與 Eq. 4 的目標 | Eq. 2 學到的是什麼？IQL 想學的 Eq. 4 與它差在哪裡？ | Eq. 2 的 $a'$ 取自資料，學到行為策略的 $Q^{\pi_\beta}$（Eq. 3）；Eq. 4 對 $s'$ 下資料支撐內（$\pi_\beta(a'|s')>0$）的動作取最大值 | S2 |
| LO3 | 寫出非對稱 L2 損失 | $L_2^\tau(u)$ 是什麼？$\tau=0.5$ 與 $\tau=0.9$ 有何不同？ | $L_2^\tau(u)=\lvert\tau-\mathbb 1(u<0)\rvert u^2$；$\tau=0.5$ 是 MSE（解為平均）；$\tau=0.9$ 時正殘差權重 0.9、負殘差 0.1，比為 9（Sec. 4.1） | S3 |
| LO4 | 計算 expectile 並說明 τ→1 的極限 | C1 預設分布（Q = 1,2,3,4,8；μ = .30,.30,.20,.15,.05）在 τ = 0.5、0.9、0.99 的 expectile？τ→1 呢？μ = 0 的動作（Q = 12）有影響嗎？ | 2.5、4.0714、7.0678；τ→1 趨近支撐內最大值 8（Lemma 1）；μ = 0 的動作完全不影響〔本頁推導：expectile 是 $x\ge m$ 權重 τ、$x<m$ 權重 1−τ 的加權平均〕 | S3 |
| LO5 | 說明為何另外學 V | 為何不直接把 TD 目標的平方換成 $L_2^\tau$？C2 預設（p = 0.3、q = 0.5、τ = 0.9）兩種做法的值？ | 目標值的隨機性也來自環境轉移，大目標值可能只是偶然轉移到高回報狀態的樣本（Sec. 4.2）；C2：單一損失的 Q(a₂) = 7.9412（IQL 為 10p = 3），狀態值單一損失版 7.6471、IQL 4.8000、資料中最好動作的期望值 5（τ→1 時分別趨近 10 與 5） | S4 |
| LO6 | 寫出 IQL 的兩個值函數損失 | Eq. 5、Eq. 6 各對什麼取 expectile／平均？哪一式用 target network？ | Eq. 5：$L_2^\tau(Q_{\hat\theta}(s,a)-V_\psi(s))$，只對資料動作取 expectile，用 target $Q_{\hat\theta}$；Eq. 6：$(r+\gamma V_\psi(s')-Q_\theta)^2$，MSE 把轉移的隨機性平均掉 | S4 |
| LO7 | 敘述理論結果 | τ = 0.5 與 τ→1 各對應什麼方法？Lemma 2 與 Theorem 3 說什麼？ | τ = 0.5 是 SARSA（估計 $Q^{\pi_\beta}$），τ→1 是資料支撐內的 Q-learning；Lemma 2：τ 越大 $V_\tau$ 越大（單調）；Theorem 3：$\lim_{\tau\to1}V_\tau(s)=\max_{a:\pi_\beta(a|s)>0}Q^*(s,a)$；前提是母體最佳解（無限容量、無取樣誤差、獎勵有界） | S5 |
| LO8 | 說明多步動態規劃的效果 | 玩具迷宮（Fig. 2）的設定？在 C3 預設下 τ = 0.5、0.95 的起點值與 V* 比較？ | 進入終點獎勵 10、機率 0.25 隨機轉移、1 條最佳 + 99 條隨機軌跡、γ = 0.9、IQL τ = 0.95；C3（seed 1）：$V_{0.5}$ = 0.0007、$V_{0.95}$ = 0.8136、$V^*$ = 1.1186，τ = 0.5 的值離終點越遠衰減越快 | S6 |
| LO9 | 說明 AWR 策略抽取 | Eq. 7 中 β 的作用？為何不查詢 OOD 動作？論文怎麼截斷？ | β 小時接近行為複製，β 大時趨向 Q 的最大值；只對資料中的 $(s,a)$ 加權對數概似，相當於分布約束下最大化 Q；$\exp(\beta A)$ 截斷到 $(-\infty,100]$（Sec. 4.3、App. B） | S7 |
| LO10 | 列出演算法與實作要點 | Alg. 1 每步更新哪些參數？clipped double Q 用在哪？Ant Maze 與 locomotion 的 τ、β？ | ψ（Eq. 5）、θ（Eq. 6）、θ̂ Polyak（α = 0.005），之後 φ（Eq. 7）；維持兩個 Q 網路，V 與策略更新時取兩者的最小值；Ant Maze τ = 0.9、β = 10，locomotion τ = 0.7、β = 3 | S8 |
| LO11 | 讀出 D4RL 結果 | IQL 與 CQL 在 Ant Maze、locomotion 的總分與 runtime？Fig. 3 的結論？ | Ant Maze 378.0 vs 303.6；locomotion 692.4 vs 698.5；20 分鐘 vs 80 分鐘（作者的 JAX 版 CQL）；Fig. 3：medium／large Ant Maze 上 τ = 0.9 的回報遠高於較小的 τ，需要拼接的任務要用較大的 expectile（Table 1、Fig. 3、Sec. 5.2） | S9 |
| LO12 | 讀出微調結果 | Table 2 三個方法的總分變化？作者給的兩個理由？ | IQL 408.2 → 597.7、CQL 182.8 → 241.0、AWAC 154.4 → 211.4；offline 初始化強，以及加權行為複製式的策略抽取有利於 online 改進（Sec. 5.3） | S10 |

---

## 3. 頁面大綱

外殼與 k-diff 頁相同：單欄閱讀（680–720px），圖版加寬（最大 1080px），桌機左側 sticky 章節目錄，章節編號 JetBrains Mono（`01 · …`），章節標題用陳述句。視覺記號：一條 τ 軸（0.5 = SARSA、→1 = Q-learning），在 hero、S3、S5、S6 重複出現。

| # | id | 標題 | 內容（依序；每個符號在使用前定義） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈IQL：只用資料中的動作做 Q-learning〉 | 副標：「IQL 用非對稱平方損失估計資料動作的上 expectile，不查詢資料外的動作，也能做多步動態規劃。」τ 軸（軸下一行小字：「τ 是 IQL 的 expectile 階數，第 03 章定義」；0.5 標「SARSA」、1 標「資料內 Q-learning」，0.7 與 0.9 標「locomotion」「Ant Maze」）。三張事實卡（見下）。連結：arXiv、`../index.html#paper-569`、總覽 `iql-lineage-overview.html`、程式碼 | 1 螢幕 |
| S1 | `offline` | 01 · Offline RL 只能使用固定資料集 | MDP、$\pi$、$Q$、$V$；TD 目標與一行數值例；target network 一行說明；Eq. 1；分布偏移；靜態小圖（1.4 步驟 1）；LO1 答案（論文說法與本頁解讀分開）。末尾符號表。延伸閱讀：`rl-mdp-q-learning-basics.html`、`offline-rl-distribution-shift.html`、`iql-policy-constraint-and-regularization-methods.html` | 650–750 字 + 1 張靜態 SVG |
| S2 | `sarsa` | 02 · SARSA 只用資料動作，但只學到行為策略 | Eq. 2、Eq. 3（附原文下標註記）、單步方法、stitching 的白話定義、Eq. 4；LO2 答案。延伸閱讀：`one-step-offline-rl-and-sequence-modeling.html` | 350–450 字 |
| S3 | `expectile` | 03 · Expectile 以非對稱平方損失逼近支撐內的最大值 | 先定義 $\mu(a|s)$（資料中動作的比例）；定義、$L_2^\tau$、9 倍權重例子、加權平均〔本頁推導〕、Lemma 1 先以白話預告；與 quantile 的比較（註腳 2）；C1；LO3、LO4 答案寫在 C1 前後的正文。延伸閱讀：`expectile-regression-value-learning.html` | 450–550 字 + C1 |
| S4 | `two-critics` | 04 · 另外學一個 V，避免把偶然轉移到高回報狀態的樣本當成動作的價值 | 單一損失式與其缺點；Eq. 5、6；C2（其「資料中最好動作的期望值」加前向註記「第 05 章 Theorem 3」）；LO5、LO6 答案 | 400–500 字 + C2 |
| S5 | `theory` | 05 · τ 從 0.5 到 1 連接 SARSA 與 Q-learning | $\mathbb E^\tau$ 記號、$V_\tau/Q_\tau$ 遞迴（$\mu=\pi_\beta$ 標本頁解讀）、Lemma 1、Lemma 2（App. A.1 的展開以可收合 `<details>`）、Corollary 2.1、Theorem 3 與其假設、τ 的取捨、τ = 0.5 即 SARSA〔本頁推導〕；LO7 答案 | 450–550 字 |
| S6 | `maze` | 06 · 玩具迷宮顯示多步動態規劃的差異 | 論文 Fig. 2 設定（數字照原文）→ 頁面重建的推定項目清單 → 元件前先各用一句定義「近似誤差 σ」與「母體模式」→ C3（旁附三步引導：先看 τ = 0.5、再 0.95、再切換母體）→ 結果段落（LO8 數字寫在正文）與「策略差異是本頁解讀」一段；壓縮性質〔本頁推導〕一句 | 450–550 字 + C3 |
| S7 | `awr` | 07 · 策略以優勢加權的行為複製抽取 | Eq. 7、β、隱含約束、截斷 100、正負號注意、V 在單一狀態內相消〔本頁推導〕；C4；LO9 答案。延伸閱讀：`awr-advantage-weighted-policy-extraction.html` | 400–500 字 + C4 |
| S8 | `impl` | 08 · 演算法總結與實作設定 | Alg. 1（≥768px 兩欄：左程式碼、右中文逐行說明；<768px 每行下接說明）、clipped double Q 一段（兩個 Q 網路、取 min、連回 S1 的高估）、超參數表（1.4 步驟 8）、6 條常見錯誤；LO10 答案。延伸閱讀：`iql-algorithm-and-d4rl-evaluation.html` | 350–450 字 + 表 |
| S9 | `results` | 09 · IQL 在需要拼接的 Ant Maze 上領先 | Table 1 摘錄（總分表 + Ant Maze 六列 IQL/CQL/Onestep）、-v2 說明、runtime、Fig. 3 文字描述、Kitchen/Adroit 一段；LO11 答案 | 350–450 字 + 2 表 |
| S10 | `finetune` | 10 · Offline 初始化之後可以繼續 online 微調 | App. C 流程、Table 2 總分表、兩個理由、pen 例外、CQL 重跑註記；LO12 答案 | 250–350 字 + 表 |
| S11 | `history` | 11 · 研究脈絡 | 時間線（6.4）+ 一段文字：約束（BCQ、BEAR、BRAC、TD3+BC）／正則化（CQL）→ 加權回歸（AWR、AWAC、CRR）→ 單步（Onestep RL、DT）→ expectile（分位數 RL 的相關工作）→ IQL | 時間線 + 100–150 字 |
| S12 | `next` | 12 · 論文中提到的取捨與延伸閱讀 | 1.4 步驟 11；筆記清單（6.4）；arXiv、程式碼 | 200 字 + 連結 |
| — | — | 頁尾 | 引用格式、`#pm-bar` | — |

Hero 三張事實卡：
1. 「標準 Q-learning 對下一個狀態的所有動作取最大值；offline 資料中沒有的動作，其 Q 值無法被資料修正，容易被高估。」
2. 「IQL 的 V 用非對稱平方損失擬合資料動作的 Q 值。τ = 0.5 時是平均（SARSA），τ 接近 1 時接近資料中最好動作的值。」
3. 「在 D4RL 的 Ant Maze 上，IQL 總分 378.0，CQL 303.6；在一張 GTX1080 上訓練 1M 步少於 20 分鐘。」

---

## 4. 互動元件規格

### 4.0 共用數學與規則

- `expectile(xs, ps, tau)`：離散分布的精確 expectile（`common.py` 逐行相同）。步驟：
  1. 取 $p>0$ 的項，依 $x$ 遞增排序為 $(x_0,p_0),\dots,(x_{n-1},p_{n-1})$（重複值保留為不同項）。$n=0$ 回傳 `NaN`（元件顯示「無資料」）。$\tau\ge1$ 回傳 $x_{n-1}$；$\tau\le0$ 回傳 $x_0$。
  2. 前綴和 $P_k=\sum_{i<k}p_i$、$S_k=\sum_{i<k}p_ix_i$（$k=0..n$）。
  3. 對 $k=0,1,\dots,n-1$（前 $k$ 項在 $m$ 之下、權重 $1-\tau$；其餘 $n-k$ 項 $\ge m$、權重 $\tau$；上半部永遠非空，因為 $m\le x_{n-1}$）：$S^{lo}_p=P_k,\ S^{lo}_{px}=S_k,\ S^{hi}_p=P_n-P_k,\ S^{hi}_{px}=S_n-S_k$，$m=\dfrac{\tau S^{hi}_{px}+(1-\tau)S^{lo}_{px}}{\tau S^{hi}_p+(1-\tau)S^{lo}_p}$。若（$k=0$ 或 $x_{k-1}\le m+10^{-12}$）且 $x_k\ge m-10^{-12}$，回傳 $m$。分母 $\ge\min(\tau,1-\tau)\,P_n>0$。
  4. 退路（理論上不會觸發，浮點保險）：在 $[x_0,x_{n-1}]$ 上對遞減的一階殘差 $g(m)=\sum p_i\,w_i\,(x_i-m)$（$x_i\ge m$ 時 $w_i=\tau$，否則 $1-\tau$）做 200 次二分，回傳中點。
- **τ = 1 的狀態**（C1、C3、C4 的 τ 滑桿上限為 0.999 或 0.99）：每個元件的狀態另有布林 `tauIsOne`。按「τ = 1（取最大值）」chip 時設為 true，滑桿移到右端但**不把 1 寫回** τ；計算時 `tauIsOne ? 1 : tau` 傳入 `expectile`。拖動或用鍵盤調整滑桿即清除旗標。旗標為 true 時讀數、`aria-valuetext` 與圖上標籤一律顯示「τ = 1（取最大值）」。C2 的 τ 軸不含 1，不設此 chip。
- `L2tau(u, tau)` $=\lvert\tau-\mathbb 1(u<0)\rvert u^2$。
- PRNG：`mulberry32(seed)`（與 k-diff 頁相同函式；Python 版在 `common.py`，已與 node 逐位比對）。高斯：Box–Muller，每次取兩個均勻數 $u_1,u_2$，回傳 $\sqrt{-2\ln(1-u_1)}\cos(2\pi u_2)$。
- 動態文字（讀數、`aria-label`、`aria-valuetext`、live region）一律 Unicode（「m₀.₉ = 4.071」「V(起點) = 0.814」），不呼叫 KaTeX。SVG 標籤中的下標用 `<tspan baseline-shift="sub" font-size="75%">`（例：m<tspan>τ</tspan>），因為 Unicode 沒有 τ 下標。
- 公式在 390px：外殼的 `.katex-display{overflow-x:auto;overflow-y:hidden}` 保留，另加 `tabindex="0"`、`role="region"`、`aria-label="公式 Eq. n，可水平捲動"`；Eq. 1、4、7 在 < 600px 改用 `aligned` 兩行版本（Eq. 1：第一行 $L_{TD}(\theta)=\mathbb E_{(s,a,s')\sim\mathcal D}\big[\big(r+\gamma\max_{a'}Q_{\hat\theta}(s',a')$，第二行 $-Q_\theta(s,a)\big)^2\big]$；Eq. 4 的 max 下標改寫成 $\mathcal A_\beta(s')$；Eq. 7：第一行 $L_\pi(\phi)=\mathbb E_{(s,a)\sim\mathcal D}\big[\exp(\beta(Q_{\hat\theta}-V_\psi))$，第二行 $\cdot\log\pi_\phi(a|s)\big]$）。兩種版本都寫在 HTML，以 CSS media query 切換顯示。
- 動畫：本頁元件沒有自動播放的動畫，狀態改變直接重畫；外殼的 transition 在 `prefers-reduced-motion: reduce` 下已取消。
- `input` 事件只更新狀態，以 `requestAnimationFrame` 合併重畫；live region（`aria-live="polite"`）與 SVG `aria-label` 只在 `change` 或停止輸入 500ms 後更新（沿用外殼的 `announce()`）。
- 不在圖上做拖曳或 hover 專屬的資訊：所有選取都有滑桿／按鈕；hover 資訊同時出現在固定讀數框。
- 曲線、長條以線型／標記形狀＋直接文字標籤區分，不只靠顏色。

### C1 · Expectile 探索器（S3）

- **教的概念**：$L_2^\tau$ 的形狀（Fig. 1 左）；expectile 是使非對稱損失最小的值；$\tau$ 增加時 $m_\tau$ 單調上升並趨近支撐內最大值（Lemma 1）；機率為 0 的動作不影響結果。
- **情境**（頁面寫出）：一個狀態下資料集中出現 5 個動作，表中為它們的 Q 值與在資料中的比例 $\mu$；另有一個資料中**沒有**的動作 $a_6$，Q 網路對它的外插值 12（可能錯誤）。
- **輸入**：
  - 滑桿 $\tau\in[0.01,0.999]$，步長 0.001，預設 0.9；chip「τ = 0.5」「τ = 0.9」「τ = 0.99」「τ = 1（取最大值）」，最後一個設 `tauIsOne`（4.0）。
  - 五個動作的 Q 值數字輸入（範圍 −5…15，步長 0.5；預設 1、2、3、4、8）與 $\mu$ 滑桿（0…1，步長 0.01；預設 .30、.30、.20、.15、.05）；$\mu$ 在計算時正規化為和 1，全部為 0 時顯示「無資料」並停用讀數。按鈕「回到預設」。
  - 勾選框「讓 a₆ 出現在資料中（μ = 0.01）」：勾選時 $a_6$ 以 0.01 加入、其餘乘 0.99。
- **計算**：$m_\tau=$`expectile`；平均 $m_{0.5}$；損失曲線 $\mathcal L(m)=\sum_i\mu_iL_2^\tau(Q_i-m)$，$m$ 取 400 點於 $[\min Q-1,\max Q+1]$（含 $a_6$ 的 Q）；$\tau\mapsto m_\tau$ 曲線取 $x=-\log_{10}(1-\tau)$ 在 $[\log_{10}2,\,3]$（即 τ ∈ [0.5, 0.999]）均分 300 點。
- **顯示**（三個小圖，<600px 垂直堆疊）：
  1. 「動作的 Q 值」：橫向長條（長度 = $\mu$，以 `--c-beh` 實心；$a_6$ 為虛線外框、標「資料中沒有」）；縱軸為 Q 值，水平線：平均（`--ink-3` 虛線，標「平均 τ=0.5」）、$m_\tau$（`--accent` 粗實線，標「m_τ」）、支撐內最大值（細點線，標「資料內最大」）、包含 $a_6$ 的全部動作最大值 $\max(Q_1..Q_5,Q_6)$（`--ink-3` 點虛線，標「對所有動作取 max」；以實際值計算，不寫死 12）。Q 值相同的長條合併畫成一條並列出動作名；水平線標籤彼此重疊（間距 < 14px）時依序往下錯開一行。
  2. 「非對稱損失 $L_2^\tau(u)$」：$u\in[-1,1]$，目前 $\tau$（實線）與 $\tau=0.5$（虛線）兩條，標出 $u=\pm0.5$ 的值。
  3. 「$m_\tau$ 隨 τ 的變化」：橫軸 $x=-\log_{10}(1-\tau)$，範圍 $[\log_{10}2,3]$，刻度標 τ 值 0.5（加「SARSA」）、0.9、0.99、0.999；曲線單調上升，右側水平點線標資料內最大值（τ→1 的極限）；目前 τ 以實心點標示（τ < 0.5 時點畫在左邊界並標「τ < 0.5 不在圖中」；`tauIsOne` 時點畫在右邊界的極限線上）。
  - 讀數表：τ、$m_\tau$（4 位小數）、平均、資料內最大、$\mathcal L(m_\tau)$。
- **數值保護**：Q 值輸入非數字時保留上一個合法值；`tauIsOne` 時 `expectile` 走 $\tau\ge1$ 分支；曲線 3 不計算 $\tau=1$（以極限線表示）。
- **讀者應觀察到**：τ = 0.5 時 $m$ = 2.5（平均）；τ = 0.9 時 4.0714；τ = 0.99 時 7.0678；τ = 1 時 8；未勾選時 $a_6$ 不改變任何讀數；勾選後 τ = 0.9 的 $m$ 變為 4.5549，τ→1 時趨近 12。

### C2 · 單一損失與 Eq. 5–6 的差別（S4）

- **教的概念**：把 expectile 直接套在 TD 目標上（單一損失），每個 $(s,a)$ 的 Q 會擬合「轉移結果」的上 expectile，偶然轉移到高回報狀態的結果被當成動作的價值；IQL 先用 MSE 對轉移取平均（Eq. 6），再只對動作取 expectile（Eq. 5）。
- **情境**（頁面以小示意圖畫出）：狀態 $s$ 有兩個資料動作。$a_1$：確定性，回報 5。$a_2$：以機率 $p$ 到高回報狀態（回報 10），否則到低回報狀態（回報 0）。資料中 $a_2$ 的比例 $q$。為了只看一步，回報直接當作 $r+\gamma V(s')$ 的值（頁面寫出這個簡化，標「推定」）。
- **輸入**：滑桿 $p\in[0,1]$（步長 0.01，預設 0.3）；$q\in[0.01,0.99]$（預設 0.5）；$\tau\in[0.5,0.999]$（步長 0.001，預設 0.9）。
- **計算**（〔本頁推導〕，見 1.4 步驟 4；`numbers.py` 的 `c2`）：
  - 單一損失：$Q_1(a_1)=5$；$Q_1(a_2)=$`expectile([10,0],[p,1-p],tau)`；$V_1=$`expectile([5,Q1a2],[1-q,q],tau)`。
  - IQL：$Q(a_1)=5$、$Q(a_2)=10p$；$V=$`expectile([5,10*p],[1-q,q],tau)`。
  - 參考值：$\max(5,10p)$，標「資料中最好動作的期望值（第 05 章 Theorem 3 的極限）」。
- **曲線取樣**：τ 在 [0.5, 0.999] 線性均分 300 點。
- **顯示**：左：兩個動作的結果分布（長條高度 = 機率，標回報值），旁邊兩列小讀數「Q(a₂)：單一損失 ／ IQL」。右：橫軸 τ ∈ [0.5, 0.999]，三條 $V$ 曲線（單一損失 `--c-naive` 虛線＋三角標記，標「單一損失」；IQL `--accent` 實線＋圓點，標「IQL（Eq. 5–6）」；參考值 `--ink-3` 點線），目前 τ 為垂直細線。讀數：$Q_1(a_2)$、$10p$、$V_1$、$V$、參考值（4 位小數）與「$V_1$ − 參考值」。
- **讀者應觀察到**：預設下 $Q_1(a_2)=7.9412$、IQL 的 $Q(a_2)=3$；$V_1=7.6471$、$V=4.8000$、參考值 5；τ→1 時 $Q_1(a_2)$ 對任何 $p>0$ 都趨近 10（$p=0.05$、τ = 0.99 時已是 8.3898），$V$ 趨近 5；把 $p$ 拉到 0 或 1（確定性轉移）兩條曲線重合。

### C3 · 玩具迷宮：τ 如何讓值函數做多步傳遞（S6）

- **教的概念**：IQL 的值遞迴（Sec. 4.4）；τ = 0.5（SARSA，行為策略的值）離終點越遠衰減越快；τ 增大時值接近最佳值 $V^*$（Lemma 2、Theorem 3）；（本頁解讀，非論文結論）在策略抽取有近似誤差 σ 時，τ = 0.5 的動作之間 Q 差距小於 σ，貪婪動作由 σ 決定。
- **環境（全部標「推定」，論文只給 1.4 步驟 6 的設定）**：
  - 格子（`#` 為牆，G 終點，S 起點），列 0 在上：
    ```
    ........
    ........
    ..####..
    ..####..
    ..####..
    ..####..
    G.####.S
    ```
    可走格依列優先編號（36 格），動作順序 上、右、下、左。撞牆或出界則留在原地。
  - 轉移：機率 0.75 照指令方向走；機率 0.25 改為均勻隨機方向（含原方向，因此照指令方向的總機率 0.8125）。進入 G 得獎勵 10 並結束（G 的值固定 0）；其餘獎勵 0；$\gamma=0.9$。
  - 真實最佳值 $V^*$：在真實 MDP 上做值迭代至最大變化 $<10^{-12}$。最佳策略：對每格取 $Q^*$ 最大的動作，平手取動作順序中第一個（容許 $10^{-12}$）。
- **資料集**（`make_dataset(seed)`）：`rnd = mulberry32(seed)`；第 0 條軌跡用最佳策略，第 1–99 條用均勻隨機動作；都從 S 出發、最多 100 步、進入 G 即停。每一步的亂數呼叫順序固定：〔隨機軌跡才有〕`a = floor(rnd()*4)` → `rnd() < 0.25` 判斷是否滑動 → 〔滑動時〕`a = floor(rnd()*4)`。紀錄計數 $N(s,a,s')$。
- **表格版 IQL**（`iql_tabular(N, tau)`，即 Sec. 4.4 的 $V_\tau,Q_\tau$ 遞迴，以經驗模型取代真實轉移）：
  - $\hat\mu(a|s)\propto N(s,a)$；$Q(s,a)=\sum_{s'}\frac{N(s,a,s')}{N(s,a)}\big[10\cdot\mathbb 1(s'=G)+\gamma V(s')\mathbb 1(s'\ne G)\big]$，只對 $N(s,a)>0$ 計算。
  - $V(s)=$`expectile`$(\{Q(s,a)\},\{N(s,a)\},\tau)$，只對資料中出現的狀態；從 $V\equiv0$ 開始迭代（先更新全部 Q，再更新全部 V），直到最大變化 $<10^{-10}$ 或 5000 次。
  - 資料中沒有外出轉移的狀態（從未當過 $s$，包括只出現在 $s'$ 的狀態，例如第 100 步截斷時的最後狀態、seed 2 的 (6,1)）：$V$ 固定為初值 0，不參與 expectile；熱圖畫成斜線底紋並標「無外出資料」，數值不印。這保證 $Q$ 的計算中 $V(s')$ 永遠有定義（`maze.py` 同樣以初值 0 處理）。
- **策略抽取與評估**：每格取 $\arg\max_a[Q(s,a)+\sigma\,\xi(s,a)]$（只在資料動作中，平手取最小動作編號）；無資料的格用均勻隨機策略。$\xi(s,a)$：`mulberry32(seed+1000)` 依 $s$ 遞增、$a$ 遞增的順序對全部 36×4 對各抽一次 Box–Muller 高斯（與資料集無關）。$\sigma$ 模擬函數近似的誤差（推定，論文沒有這個參數；頁面寫明「表格版沒有近似誤差，因此加入固定雜訊代表神經網路的誤差」）。在真實 MDP 上做策略評估（迭代至 $<10^{-12}$），讀數為起點的期望折扣回報 $V^{\pi}(S)$。
- **模式**（收在「進階」內）：分段按鈕「資料：有限資料集（預設）／母體」。「母體」使用真實轉移機率與均勻行為策略（四個動作都在支撐內），此時 τ = 1 精確等於 $V^*$（Theorem 3），且 $V_\tau(S)$ 隨 τ 單調上升（Lemma 2）。
- **輸入**：
  - 常駐：滑桿 τ ∈ [0.5, 0.99]，步長 0.01，預設 0.95；chip「0.5（SARSA）」「0.7」「0.9」「0.95」「1（取最大值）」（設 `tauIsOne`）；分段按鈕「近似誤差 σ」：0、0.001、0.01（預設）、0.03、0.1。
  - 「進階」`<details>`（預設收合）：資料模式（上述）；顯示切換「V_τ／V*／差 V* − V_τ」；勾選框「顯示貪婪策略箭頭」（預設開，箭頭本身常駐顯示）；「重新產生資料（seed + 1）」與「回到 seed 1」兩個按鈕，旁邊顯示「目前 seed = n」。LO8 的數字只在 seed 1 成立，頁面在讀數旁註明。
  - 元件上方三步引導按鈕：「1 看 τ = 0.5」「2 看 τ = 0.95」「3 切換到母體、τ = 1」，各自設定對應狀態（第 3 步自動展開進階）。
- **顯示**：
  - 熱圖（`role="img"`，`aria-label` 摘要「起點 V = …，終點旁 V = …」；另附 sr-only 的 7×8 `<table>` 列出每格數值與箭頭方向）：每格以單一色相明度表示值（0 → `--surface-2`，10 → `--c-beh`；差值模式 0 → `--surface-2`、最大值 → `--accent`），**每格同時印出數值（兩位小數）**，所以不依賴顏色；牆為 `--ink` 實心；S、G 以文字標示。箭頭以 `--ink` 畫在格中央。
  - 旁邊的小圖：橫軸 τ ∈ [0.5, 1]，縱軸 $V_\tau(S)$，以 21 個 τ（0.50, 0.525, …, 1）預先算出的折線＋$V^*$ 水平點線；目前 τ 為實心點。
  - 讀數：資料集統計（出現的狀態數、轉移數）、$V_\tau(S)$、$V^*(S)$、迭代次數、貪婪策略的 $V^\pi(S)$ 與 $V^*(S)$ 的比。
- **效能**：一次 `iql_tabular` 最多約 185 次迭代 × 140 對（每次含 35 次小排序）；21 點折線在元件首次進入視窗時（IntersectionObserver）才計算，換資料或模式時重算並快取（key `${mode}|${seed}`）。建置時以 `performance.now()` 實測單次與 21 點的耗時並寫回本文件第 9 節；若 21 點超過 50ms，改為分批在 `requestIdleCallback` 中計算。
- **讀者應觀察到**：(1) τ = 0.5 時值只在終點附近大，起點約 0.0007；(2) τ = 0.95 時起點 0.8136，形狀接近 $V^*$；(3)（本頁解讀）σ = 0 時 τ = 0.5 與 τ = 0.9 的貪婪策略都達到最佳回報 1.119（只需動作的大小順序）；σ = 0.01 時 τ = 0.5 降到 0.051，τ = 0.9 仍為 1.119——σ 大於動作之間的 Q 差距時，貪婪動作由 σ 決定；(4) 有限資料集 τ = 1 的起點值 1.2620 **高於** $V^*$ = 1.1186（推測：經驗轉移的抽樣誤差被最大值選中；Corollary 2.1 的上界是對真實 MDP 而言），「母體」模式下則不會超過；(5) seed 1–5 的 $V_{0.5}(S)$ 都在 0.0004–0.0007、$V_{0.95}(S)$ 在 0.70–0.84；σ = 0.01 時 τ = 0.5 的策略回報 0.000–0.051、τ = 0.9 為 0.616–1.119（`maze_noise_out.txt` 的 seed 表）。σ = 0 的策略結果依資料而定（seed 4 所有 τ 都是 0.616）；seed 2 可看到「無外出資料」的格子。

### C4 · AWR 策略抽取：β、截斷與隱含約束（S7）

- **教的概念**：Eq. 7 是加權的行為複製；β = 0 為行為複製，β 增大時策略往資料中優勢大的動作集中，但不會離開資料範圍；截斷 100 限制少數樣本的權重；Q 固定時，不截斷的策略與 V 無關〔本頁推導〕。**限制（頁面在圖版標題下明寫）**：這裡的 Q 是固定的函數，所以 τ 只透過 V 與截斷影響結果；真實 IQL 中 τ 主要透過 TD 學習（Eq. 5–6）改變學到的 Q 與 V，這個效果見 C3。
- **情境**（一維動作、單一狀態；全部數值推定，頁面寫明是示意）：資料動作 $a_i=-0.8+0.05i$，$i=0..22$（$[-0.8,0.3]$，23 個），資料量（相對）$n_i=\exp(-(a_i+0.25)^2/(2\cdot0.25^2))$；Q 函數 $Q(a)=1-(a-0.6)^2$，最大值在資料外的 $a=0.6$。
- **輸入**：滑桿 β ∈ [0, 30]（步長 0.1，預設 3）；chip 依 App. B 的配對同時設定 β 與 τ：「Kitchen/Adroit：β = 0.5、τ = 0.7」「locomotion：β = 3、τ = 0.7」「Ant Maze：β = 10、τ = 0.9」；滑桿 τ ∈ [0.5, 0.99]（步長 0.01，預設 0.7；另有 chip「τ = 1」設 `tauIsOne`）；勾選框「截斷 exp(βA) ≤ 100」（預設開）。
- **計算**：$V=$`expectile`$(Q(a_i),n_i,\tau)$；$A_i=Q(a_i)-V$；截斷開：$w_i=\min(\exp(\min(\beta A_i,700)),100)$；截斷關：$w_i=\exp(\beta A_i-\max_j\beta A_j)$（log-sum-exp 式平移，避免溢位；共同倍數在正規化後相消）。$c_i=n_iw_i$。高斯策略（加權最大概似）：$\bar a=\sum c_ia_i/\sum c_i$，$s=\sqrt{\sum c_i(a_i-\bar a)^2/\sum c_i}$。有效樣本比例 $\mathrm{ESS}=(\sum n_iw_i)^2/(\sum n_i\cdot\sum n_iw_i^2)$；被截斷的資料比例 $=\sum_{\beta A_i>\ln100}n_i/\sum n_i$（截斷關時為 0）。
- **顯示**：上圖：橫軸 $a\in[-1,1]$；Q 曲線（資料範圍內實線，資料外虛線並在 $[0.3,1]$ 加淺底紋與標籤「資料中沒有的動作：IQL 從不查詢」）；水平線 $V$（標「V = m_τ」）。下圖：資料量長條 $n_i$（`--c-beh`，空心）與加權後 $c_i$（`--accent`，實心，兩者各自正規化到最大值 1；被截斷的長條頂端加短橫線標記）；策略密度 $\mathcal N(\bar a,s^2)$（`--accent` 實線）與行為密度（β = 0 的結果，`--ink-3` 虛線）。兩條密度共用右側 y 軸（標「機率密度」），上限 = 兩者峰值較大者 × 1.1（β = 30 不截斷時策略峰值約為行為密度的 4.5 倍，軸隨之放大）；長條使用左側 0–1 的相對刻度，圖旁註明「長條各自正規化到最大值 1」。讀數：$V$、$\bar a$、$s$、ESS、截斷比例。
- **讀者應觀察到**：(1) β = 0：$\bar a=-0.2500$（資料平均）；(2) β = 3：$\bar a=-0.0426$；β = 10：0.1469（τ = 0.9）；平均值永遠不超過資料中最大的動作 0.3；(3) 關閉截斷後改變 τ，讀數完全不變（因為這裡 Q 固定）；開啟截斷、β = 10 時 τ = 0.5、0.7、0.9 的 $\bar a$ 分別為 0.0597、0.1175、0.1469；(4) 不截斷、β = 30 時 ESS 只剩 0.0231，截斷時 0.3237（τ = 0.7）。

---

## 5. 驗收數值表

腳本與頁面同一算法、雙精度。容許誤差：表中數字 ±0.00005（5 位小數者，如 C1 的損失）、±0.0005（4 位小數者）或 ±0.005（2–3 位者），另註明者除外。

**Fig. 1 中圖（`numbers_out.txt`，S3 正文用）**：N(0,1) 的 $m_{0.01},m_{0.1},m_{0.5},m_{0.9},m_{0.99}$ = −1.7174、−0.8616、0、0.8616、1.7174。

**C1**（預設分布；`numbers_out.txt`）

| τ | 0.01 | 0.1 | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $m_\tau$ | 1.0493 | 1.4412 | 2.5000 | 3.0263 | 4.0714 | 5.1053 | 7.0678 | 7.8919 | 8.0000 |
| $\mathcal L(m_\tau)$ | 0.04826 | 0.42382 | 1.32500 | 1.42974 | 1.12929 | 0.84895 | 0.27773 | 0.03231 | 0 |

勾選 $a_6$（μ = 0.01）：τ = 0.9 → 4.5549，τ = 0.99 → 7.7826。$L_2^{0.9}(\pm0.5)$ = 0.225／0.025；$L_2^{0.5}(0.5)=0.125$。

**C2**（`numbers_out.txt`；欄：$Q_1(a_2)$ 單一損失 ／ $V_1$ 單一損失 ／ $V$ IQL ／ 參考值；IQL 的 $Q(a_2)=10p$）

| p, q | τ = 0.5 | 0.7 | 0.9 | 0.99 | 1 |
| --- | --- | --- | --- | --- | --- |
| 0.3, 0.5（預設） | 3.0000 / 4.0000 / 4.0000 / 5 | 5.0000 / 5.0000 / 4.4000 / 5 | 7.9412 / 7.6471 / 4.8000 / 5 | 9.7697 / 9.7220 / 4.9800 / 5 | 10 / 10 / 5 / 5 |
| 0.7, 0.5 | 7.0000 / 6.0000 / 6.0000 / 7 | 8.4483 / 7.4138 / 6.4000 / 7 | 9.5455 / 9.0909 / 6.8000 / 7 | 9.9569 / 9.9073 / 6.9800 / 7 | 10 / 10 / 7 / 7 |
| 0.3, 0.1 | 3.0000 / 4.8000 / 4.8000 / 5 | 5.0000 / 5.0000 / 4.9091 / 5 | 7.9412 / 6.4706 / 4.9756 / 5 | 9.7697 / 9.3723 / 4.9978 / 5 | 10 / 10 / 5 / 5 |
| 0.05, 0.5 | 0.5000 / 2.7500 / 2.7500 / 5 | 1.0937 / 3.8281 / 3.6500 / 5 | 3.2143 / 4.8214 / 4.5500 / 5 | 8.3898 / 8.3559 / 4.9550 / 5 | 10 / 10 / 5 / 5 |
| 1.0, 0.5 | 10 / 7.5 / 7.5 / 10 | 10 / 8.5 / 8.5 / 10 | 10 / 9.5 / 9.5 / 10 | 10 / 9.95 / 9.95 / 10 | 10 / 10 / 10 / 10 |
| 0.0, 0.5 | 0 / 2.5 / 2.5 / 5 | 0 / 3.5 / 3.5 / 5 | 0 / 4.5 / 4.5 / 5 | 0 / 4.95 / 4.95 / 5 | 0 / 5 / 5 / 5 |

（τ = 1 欄只供腳本核對極限；C2 滑桿上限 0.999。）

**C3**（seed 1，`maze_out.txt`、`maze_noise_out.txt`、`maze_pop_out.txt`）

$V^*(S)=1.1186$。資料集：35/35 個非終點狀態出現、9841 次轉移、140 個 $(s,a)$ 對。探針格 $(r,c)$ = (6,1)、(0,0)、(0,4)、(0,7)、(6,7)=S：

| 模式 | τ | 迭代 | $V_\tau$ 於探針格 | $V^\pi(S)$，σ = 0 / 0.001 / 0.01 / 0.03 / 0.1 |
| --- | --- | --- | --- | --- |
| 有限 | 0.5 | 185 | 8.956, 0.295, 0.052, 0.010, 0.0007 | 1.119 / 0.401 / 0.051 / 0.000 / 0.000 |
| 有限 | 0.7 | 167 | 9.469, 1.321, 0.497, 0.193, 0.0434 | 1.119 / 1.119 / 1.119 / 0.154 / 0.099 |
| 有限 | 0.9 | 96 | 9.723, 3.481, 1.904, 1.138, 0.5136 | 1.119 / 1.119 / 1.119 / 1.114 / 0.417 |
| 有限 | 0.95 | 76 | 9.766, 4.312, 2.465, 1.588, 0.8136 | 1.061 / 1.061 / 1.068 / 1.067 / 0.398 |
| 有限 | 0.99 | 61 | 9.797, 5.137, 3.056, 2.068, 1.1583 | 1.056 / 1.056 / 1.053 / 1.056 / 0.397 |
| 有限 | 1 | 56 | 9.804, 5.366, 3.227, 2.208, 1.2620 | 1.056 / 1.056 / 1.053 / 1.056 / 0.396 |
| 母體 | 0.5 | 181 | 6.135, 0.198, 0.030, 0.005, 0.0003 | — |
| 母體 | 0.9 | 102 | 9.202, 3.314, 1.839, 1.074, 0.4539 | — |
| 母體 | 0.99 | 66 | 9.650, 4.603, 2.866, 1.920, 1.0332 | — |
| 母體 | 1 | 62 | 9.696, 4.756, 2.989, 2.027, 1.1186（= $V^*$，最大誤差 8e-11） | — |
| $V^*$ | — | — | 9.696, 4.756, 2.989, 2.027, 1.1186 | — |

迭代次數容許 ±2（收斂判準的浮點差異）。

seed 1–5（`maze_noise_out.txt` seed 表；$V^\pi(S)$ 欄為 σ = 0 / 0.01）：

| seed | τ = 0.5 的 $V^\pi(S)$ | τ = 0.9 的 $V^\pi(S)$ | $V_{0.5}(S)$ | $V_{0.95}(S)$ |
| --- | --- | --- | --- | --- |
| 1 | 1.119 / 0.051 | 1.119 / 1.119 | 0.0007 | 0.8136 |
| 2 | 1.100 / 0.038 | 0.714 / 1.096 | 0.0006 | 0.7020 |
| 3 | 1.119 / 0.015 | 1.119 / 1.114 | 0.0004 | 0.8157 |
| 4 | 0.616 / 0.000 | 0.616 / 0.616 | 0.0005 | 0.8373 |
| 5 | 0.720 / 0.043 | 1.034 / 1.034 | 0.0005 | 0.7199 |

seed 2：34/35 狀態有外出資料（(6,1) 沒有）。

**C4**（`numbers_out.txt`；欄：V／$\bar a$／s／ESS／截斷比例）

| τ | β | 截斷 | V | ā | s | ESS | 截斷比例 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 任意 | 0 | 任意 | — | −0.2500 | 0.2328 | 1.0000 | 0 |
| 0.7 | 0.5 | 開 | 0.3541 | −0.2059 | 0.2272 | 0.9652 | 0 |
| 0.7 | 3 | 開 | 0.3541 | −0.0426 | 0.1907 | 0.5162 | 0 |
| 0.5 | 10 | 開 | 0.2233 | 0.0597 | 0.1173 | 0.2625 | 0.1274 |
| 0.7 | 10 | 開 | 0.3541 | 0.1175 | 0.1153 | 0.1615 | 0.0344 |
| 0.9 | 10 | 開 | 0.5326 | 0.1469 | 0.1196 | 0.1136 | 0 |
| 任意 | 10 | 關 | — | 0.1469 | 0.1196 | 0.1136 | 0 |
| 0.7 | 30 | 開 | 0.3541 | 0.0167 | 0.1113 | 0.3237 | 0.3041 |
| 任意 | 30 | 關 | — | 0.2590 | 0.0520 | 0.0231 | 0 |

---

## 6. 實作計畫

### 6.1 檔案與共用外殼

- 單一檔案 `tut/569/build/index.html`，CSS／JS／時間線資料全部內嵌，不 `fetch` 本機檔案；上線時複製為 `tutorials/implicit-q-learning.html`。
- **直接複製** `tut/406/build/index.html`（最新建置頁）的外殼，不重寫：CSS tokens（含深色模式三段覆寫、`pm-theme`）、字型載入、appbar、`#pm-bar`、章節目錄（sticky 左欄／窄螢幕橫列）、`.plate` 圖版、callout、`<details>` 檢查題（`.qa`）、「本頁推導」與「推定」pill、「本章符號」小表、表格橫捲容器（`.tscroll`）、KaTeX 0.16.9 設定與 `renderMath()`、`?test` selfTest（`ok()` 計數、`#selftest` 輸出、`scrollWidth <= innerWidth` 檢查）、時間線卡片列 `renderTimeline()`、SVG 輔助（`el`、`scaleLin`、`scaleLog`、`linePath`）、`announce()`、`radiogroup()`（roving tabindex）、`schedule()`（rAF 合併）、`ResizeObserver` 重畫。
- 以下只列新增或變更。

### 6.2 新增 tokens 與連結

- 圖表色：`--c-beh`（light #3F6E8C／dark #8DB4D3，沿用 k-diff 頁 `--c-par` 色值）表示資料／行為策略；`--c-naive`（light #6B7A3A／dark #A9BC7C，沿用 `--c-perp` 色值）表示單一損失；IQL 與目前值用 `--accent`。建置時確認在兩種 `--surface` 上圖形對比 ≥ 3:1、文字 ≥ 4.5:1，不足則調整並回寫此處。
- C3 熱圖色階：先寫純色 fallback（`fill` 屬性由 JS 以兩個 token 的 RGB 線性內插算出），再以 `@supports (color: color-mix(in oklab, red, blue))` 時改用 `color-mix(in oklab, var(--c-beh) X%, var(--surface-2))`，X = 100·V/10；格內數字顏色依 X > 55 改用 `--surface`，否則 `--ink`（兩種主題各檢查一次對比）。
- 站內連結：Hero「在關聯圖中查看」→ `../index.html#paper-569`；arXiv `https://arxiv.org/abs/2110.06169`；程式碼 `https://github.com/ikostrikov/implicit_q_learning`；筆記 → `<slug>.html`。

### 6.3 程式結構（單一 IIFE）

- 共用：`expectile(xs, ps, tau)`、`L2tau(u, tau)`、`mulberry32(seed)`、`gauss(rnd)`。
- C1：`initC1()`，`state={tau, q:[5], mu:[5], incA6}`，`renderC1()`。
- C2：`initC2()`，`state={p, q, tau}`，`c2Values(p,q,tau)` → `{qNaive, qIql, vNaive, vIql, truth}`。
- `expectile` 的退路寫成獨立函式 `bisectExpectile(pts, tau)`，供 selfTest 直接呼叫。
- C3：`initC3()`；常數 `GRID`、`DIRS`、`GAMMA=0.9`、`SLIP=0.25`、`R_GOAL=10`、`HORIZON=100`、`N_RAND=99`；函式 `trueP(s,a)`、`valueIteration()`、`greedyTrue(V)`、`makeDataset(seed)`、`populationCounts()`、`iqlTabular(N, tau)` → `{V, Q, acts, iters}`、`noiseTable(seed)`、`extract(Q, acts, xi, sigma)`、`evalPolicy(pol)`；`state={mode, seed, tau, sigma, arrows, view}`；`cache=new Map()`。逐式對應 `maze.py`／`maze_noise.py`／`maze_pop.py`。
- C4：`initC4()`，常數 `A4`、`N4`、`Q4`；`awr(tau, beta, clip)` → `{V, mean, std, ess, clipFrac, c}`。
- `selfTest()` 新增斷言（容許誤差見第 5 節）：
  - `expectile([1,2,3,4,8],[.3,.3,.2,.15,.05], τ)`：τ = 0.5 → 2.5、0.9 → 4.0714、0.99 → 7.0678、1 → 8；加入 `(12, 0)` 後不變。
  - 重複值與極端 τ：`expectile([2,2,2],[1,1,1],0.999)` = 2；`expectile([1,3,3,8],[.25,.25,.25,.25],0.999)` = 7.9830；預設分布 τ = 0.999 → 7.8919；空支撐回傳 NaN；退路分支以人為呼叫 `bisectExpectile` 驗證與閉式解相差 < 1e-9。
  - `tauIsOne`：C1 設旗標後讀數為 8 且滑桿 `aria-valuetext` 含「τ = 1」；拖動滑桿後旗標清除。
  - 隨機 50 組（`mulberry32(42)`）檢查一階條件殘差 $\lvert\sum p\lvert\tau-\mathbb 1(x<m)\rvert(x-m)\rvert<10^{-9}$。
  - `c2Values(0.3,0.5,0.9)` → $Q_1(a_2)$ 7.9412、$V_1$ 7.6471、$V$ 4.8000、參考值 5。
  - `mulberry32(1)` 前兩個輸出 = 0.6270739405881613、0.002735721180215478（與 Python 一致）。
  - C3 seed 2：$V$ 陣列沒有 NaN（「無外出資料」的格為 0）。
  - C3 seed 1：`valueIteration()[S]` = 1.1186；`iqlTabular(N,0.5)` 的 $V(S)$ = 0.0007、τ = 0.95 → 0.8136、τ = 1 → 1.2620；σ = 0.01 時 τ = 0.5 的 $V^\pi(S)$ = 0.051、τ = 0.9 → 1.119；母體 τ = 1 與 $V^*$ 的最大差 < 1e-8。
  - C4：`awr(0.7,3,true).mean` = −0.0426；`awr(0.9,10,true).mean` = 0.1469；`awr(0.5,30,false)` 與 `awr(0.9,30,false)` 的 mean 相同（1e-12）。

### 6.4 研究筆記與時間線

- 來源 `tut/569/notes/timeline.json`（14 筆，`year` 為數字；欄位 `year`、`title`、`authors_short`、`arxiv`、`role`、`source`），原樣貼進 `<script type="application/json" id="timeline-data">`。排序用 `Number(year)`，同年保持檔案順序。
- `TL_GROUP`（以 arXiv ID 為 key；不在表中歸「其他」，並在建置回報列出）：

  | 類別 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | 策略約束與值正則化 | 1812.02900、1906.00949、1911.11361、2006.04779、2106.06860 | `iql-policy-constraint-and-regularization-methods.html` |
  | 加權回歸 | 1910.00177、2006.09359、2006.15134 | `awr-advantage-weighted-policy-extraction.html` |
  | 單步與序列建模 | 2106.08909、2106.01345 | `one-step-offline-rl-and-sequence-modeling.html` |
  | 分布式 RL（expectile 的相關工作） | 1806.06923 | `expectile-regression-value-learning.html` |
  | 基準與元件 | 2004.07219、1802.09477 | `iql-algorithm-and-d4rl-evaluation.html` |
  | 本文 | 2110.06169 | `iql-lineage-overview.html` |

- S12 延伸閱讀（依總覽閱讀順序）：`iql-lineage-overview`、`rl-mdp-q-learning-basics`、`offline-rl-distribution-shift`、`iql-policy-constraint-and-regularization-methods`、`awr-advantage-weighted-policy-extraction`、`one-step-offline-rl-and-sequence-modeling`、`expectile-regression-value-learning`、`iql-algorithm-and-d4rl-evaluation`。每項附 frontmatter 的 `title`（後兩個 No. 568 筆記的標題建置時從 `tut/568/notes/` 讀取）。

### 6.5 文案與排版（新增部分）

- 用詞：「資料外的動作（out-of-distribution／out-of-sample action）」「資料支撐內（in-support）」「行為策略（behavior policy）」「行為複製（behavioral cloning）」「拼接（stitching）」「多步動態規劃」「單步方法」「優勢（advantage）」「反溫度 β」「常見錯誤」。第一次出現附英文。
- 三種陳述分開標示：論文結果（附出處）、「作者認為」、〔本頁推導〕；玩具元件的自選數值加「推定」pill。
- 公式旁標論文編號；Eq. 4 的 max 下標在 < 600px 時改用 `\max_{a'\in\mathcal A_\beta(s')}` 並在下方一行定義 $\mathcal A_\beta(s')=\{a':\pi_\beta(a'|s')>0\}$。
- 第 1 節以外不得新增論文數字；頁面數字須與第 1、5 節一致。

### 6.6 建置後自我檢查

沿用流程：390／768／1024／1440 截圖（只看新元件與 390px，最多 1 輪）＋深色、鍵盤 focus、reduced-motion；`?test` 全數通過；impeccable 0 項；執行 `extract_page_text.py` 確認 12 條 LO 參考答案的關鍵詞與數字出現在正文。截圖用本機 KaTeX（`scratchpad/katex/0.16.9`），上線檔仍指向 cdnjs。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 玩具迷宮不是論文的程式 | 論文沒有給迷宮形狀、軌跡長度、網路設定；C3 是表格版重建，數字不能與 Fig. 2 逐格比較。 | 所有自選值標「推定」；頁面只聲稱「定性結論一致」。 |
| R2 | 「隨機狀態」的解讀 | 原文 “transitions to a random state”。解讀為全迷宮均勻跳躍（可能直接跳進 G）時，右臂各格的 $V^*$ 都約 2.72–2.76（`teleport.py`），距離效應被抹平；本頁解讀為隨機方向（相鄰格）。 | 頁面寫明這是本頁的解讀。 |
| R3 | 原文的筆誤 | Eq. 3 下標 $\pi_\beta(\cdot|s)$ 應為 $s'$；Eq. 7 寫成要最大化的式子，Alg. 1 卻寫梯度下降；Sec. 4.1 “thoroughly studies”。 | 頁面寫正確版本並以小字註明原文寫法（Eq. 3、Eq. 7）；拼字不提。 |
| R4 | Table 1 與 Table 2 的 offline 分數不同 | IQL：antmaze-umaze 87.5（Table 1）vs 86.7（Table 2），論文未說明。CQL：Table 2 起點 151.5 是作者重跑、重現較差（App. C），與 Table 1 的 303.6 不可比。App. C 寫「offline 實驗（Table 2）報告原論文結果」，所指對象論文不清楚。 | 兩表分開呈現、各標出處；只寫論文寫了什麼與「論文未說明／不清楚」。 |
| R5 | σ 雜訊是本頁的解讀 | 表格版沒有近似誤差，seed 1 中 τ = 0.5 的貪婪策略在 σ = 0 時是最佳的；論文沒有說單步方法的策略差是因為近似誤差。雜訊只加在策略抽取，沒有進入備份；σ = 0 的結果依資料而定（seed 4）。 | 頁面把策略差異標為「本頁解讀」，明寫 σ = 0 的結果；σ 標「推定」；不把 σ 結果放在 Fig. 2 結論旁當作論文結果。 |
| R6 | 有限資料下 τ = 1 高估 | seed 1 的 $V_1(S)=1.2620>V^*$；Corollary 2.1 是對真實 MDP 的敘述。原因（抽樣誤差被最大值選中）沒有單獨驗證。 | 寫「推測」並以「母體」模式對照；可作為 τ < 1 的一個額外理由，但頁面不說這是論文的理由。 |
| R7 | Fig. 3 沒有數據 | 只有曲線圖；不讀圖。 | 文字描述 τ = 0.9 > 0.7 > 0.5 的排序與作者結論。 |
| R8 | C4 是示意 | 單一狀態、一維動作；真實 IQL 的策略是跨狀態共享的網路，V 的尺度會影響不同狀態間的相對權重。「V 相消」只在單一狀態內成立。 | 〔本頁推導〕標籤中寫明「單一狀態內」。 |
| R9 | 跨教學筆記（已關閉） | `rl-mdp-q-learning-basics`、`offline-rl-distribution-shift` 屬 No. 568（CQL）的筆記；協調者確認 No. 568 先於 No. 569 上架並以這兩個 slug 發布。 | 名稱固定，照用。 |
| R10 | 初學者門檻 | 讀者可能不熟 TD、target network。 | S1 以 2–3 句白話解釋 TD 與 target network；每章「本章符號」小表；符號在使用前定義。 |
| R11 | 相對連結 | 建置位置與上線位置不同，預覽時站內連結失效。 | 接受；預覽只檢查外部連結與頁內錨點。 |

---

## 8. 修訂紀錄（round 1，對應 `reviews/round1.md`）

| # | 變更 |
| --- | --- |
| 1 | 4.0 新增 `tauIsOne` 規則；C1、C3、C4 的「τ = 1」chip 改設旗標 |
| 2 | 4.0 完整寫出 `expectile`（S^lo/S^hi、k 範圍、二分退路）；`common.py` 同步並加重複值、τ = 0.999 自測；selfTest 同步 |
| 3 | C2 改為逐動作的單一損失（Q₁(a₂) = 7.9412、V₁ = 7.6471）；步驟 4、LO5、第 5 節 C2 表更新；混合版記為已捨棄 |
| 4 | 步驟 9、LO11：Fig. 3 為四格，只寫 τ = 0.9 遠高於較小 τ |
| 5 | C4：Q 固定的限制寫入概念與圖版；chip 依 App. B 配對 β 與 τ；密度共用 y 軸 |
| 6 | 步驟 6、C3 概念與觀察 3、R5：策略差異標為本頁解讀，寫明 σ = 0 的結果 |
| 7 | 步驟 5、LO7：Theorem 3 假設；壓縮性質的適用條件 |
| 8 | C3：無外出資料的狀態 V = 0，標示方式；selfTest 檢查無 NaN |
| 9 | C3：常駐只有 τ、σ，其餘收進進階；三步引導；S6 元件前先定義 σ 與母體 |
| 10 | C3：回到 seed 1、顯示目前 seed；第 5 節加 seed 1–5 表（`maze_noise.py`） |
| 11 | 符號表補 μ、Q^{πβ}、𝒜_β、Q*、V_τ、Q¹/Q²；μ = πβ 標本頁解讀；S3 先定義 μ；C2 參考值前向註記 |
| 12 | S1 放寬為 650–750 字，加 TD 數值例、target 說明、靜態小圖；LO1 分開論文說法與本頁解讀 |
| 13 | 步驟 8、S8、LO10：clipped double Q 一段，Alg. 1 註明取 min |
| 14 | S4 標題、步驟 4、C2 名稱、常見錯誤 5、C3 觀察改為字面敘述 |
| 15 | 4.0：公式可橫捲（tabindex、aria-label），Eq. 1、4、7 窄螢幕兩行版本 |
| 16 | C1 圖 3：x = −log10(1−τ)，τ ∈ [0.5, 0.999]，刻度 0.5/0.9/0.99/0.999 |
| 17 | C1 圖 1：全部動作最大值以實際值計算；重疊長條合併、標籤錯開 |
| 18 | C3 熱圖 role=img + sr-only 表；SVG 下標用 tspan；Alg. 1 用 θ_targ；color-mix 純色 fallback |
| 19 | 4.0 寫明本頁無動畫；C3 21 點首次進入視窗才算，建置時實測寫回 |
| 20–24 | 步驟 10／R4（CQL 151.5 不可比、「論文不清楚」）；1.1 與 hero 的 runtime 限定詞；-v2 照原文並註明 -v0 標籤；R6 改「推測」；微調列標 dexterous 設定 |
| 25 | verify 腳本加 `sys.path.insert`，已以 `python3 -I` 重跑，`*_out.txt` 更新（數字與 round 0 相同，C2 與 seed 表為新增） |


## 9. 建置紀錄（readiness 修正 + build）

- readiness 阻擋項：R9 關閉（No. 568 先上架，slug 固定）；$V^*$ 定義加入 S5 與符號表；Hero τ 軸下加一行「τ 是 IQL 的 expectile 階數，第 03 章定義」。
- 非阻擋項：Table 2 CQL 182.8「照原文印出」（列和 183.0）；「offline 25000 步」加「原文有歧義」pill 並附兩種讀法；Alg. 1 下方寫 Adam 3·10⁻⁴；5 位小數容許 ±0.00005；C2 τ 軸 300 點。
- C3 21 點折線實測：node 中 seed 1 一次算完 63 ms（單次 `iqlTabular` 2–10 ms）；超過 50 ms，因此頁面以 12 ms 為一批、`setTimeout` 分段計算，元件首次進入視窗才開始。
- 實作差異：其他可收合區塊用 `details.more`（`details.check` 只留給檢查題，供驗收腳本移除）；C3 牆面新增 `--wall` token（深色模式下 `--ink` 太亮）；數學中的 `<` 在 HTML 內一律寫成 `&lt;`；Hero 的 τ 軸沿用 k 軸的 HTML 元件與標籤分層演算法。
- 驗證：`?test` 136 passed、0 failed（1440 與 390 寬）；impeccable 0 項（12 項 numbered-section-labels advisory，屬 DESIGN.md 的目錄式編號）；無 console 錯誤；KaTeX 237 個節點；12 條 LO 參考答案的關鍵數字與用語都出現在 `accept/page.txt` 正文。
