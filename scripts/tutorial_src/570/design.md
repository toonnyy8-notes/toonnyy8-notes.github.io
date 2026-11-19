# 設計文件：〈A Minimalist Approach to Offline Reinforcement Learning〉互動教學

- 論文：Scott Fujimoto、Shixiang Shane Gu，arXiv 2106.06860（v2，NeurIPS 2021）。資料庫編號 No. 570。
- 原文：`tut/570/src/paper.txt`（PDF 同目錄）。本文件中的「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」皆指論文原文編號。`tut/` = `scripts/tutorial_src/`。
- 產出：`tut/570/build/index.html`（單一 HTML 檔）。上線位置：`tutorials/td3-bc-minimalist-offline-rl.html`（slug `td3-bc-minimalist-offline-rl`）。站內相對連結一律以上線位置為準。
- 驗算腳本（非頁面程式，與頁面同一演算法、雙精度；皆可用 `python3 -I` 執行）：`tut/570/verify/toy.py`（C1 玩具、PRNG、內部最佳解閉式）、`c2.py`（C2 與連續性檢查）、`c4.py`（C4 表格加總、± 核算與內文算術）。輸出存於同目錄 `*_out.txt`。
- 研究筆記：`tut/570/notes/`（總覽 `td3bc-lineage-overview` + 6 篇主題筆記 + `timeline.json`，14 筆）；另引用 No. 568 的 `rl-mdp-q-learning-basics`、`offline-rl-distribution-shift`（位於 `tut/568/notes/`）。
- 標記慣例：**〔本頁推導〕** = 本設計自行推導、論文沒有寫出；**推定** = 為了貼近論文敘述而校準、論文沒有寫明的值；**背景（論文外）** = 論文引用但未寫出的既有知識（TD3 細節、D4RL 正規化分數、官方程式碼）；「作者認為」= 作者的解釋而非實驗結果。
- 修訂：round 1 審查的處理見 `reviews/round1.md`（26 項）與第 8 節。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

離線強化學習（offline RL）只能從一個固定的資料集學習，不能再與環境互動。直接把 off-policy 演算法用在離線設定，會遇到外插誤差（extrapolation error）：價值函數對資料集中沒有出現的動作估計不準，而策略改進會選擇這些被高估的動作（Sec. 1、3）。既有方法以各種方式讓策略靠近資料，但同時加入許多實作調整、額外超參數與生成模型，難以重現、調參，也拖慢執行時間（Sec. 4、Table 1）。本文只對 TD3 做兩項修改：在策略更新中加入行為複製（behavior cloning, BC）項 $-(\pi(s)-a)^2$，並以 $\lambda=\alpha/\frac1N\sum|Q(s_i,a_i)|$、$\alpha=2.5$ 平衡兩項（Eq. 3、5）；再把狀態特徵正規化為平均 0、標準差 1（Eq. 4）。這個演算法稱為 TD3+BC。在 D4RL MuJoCo 的 15 個資料集上總分 979.3，Fisher-BRC 974.6，CQL 764.3（Table 2；Total 的 ± 為各列 ± 之和，見步驟 8）；訓練時間 39 分鐘，Fisher-BRC 2 小時 8 分、CQL 4 小時 11 分（Table 3）。作者也指出一個未解問題：所有離線訓練出的策略在單次評估內與不同評估點之間的表現變異都很大（Fig. 2–3）。

### 1.2 研究動機（Sec. 1、Sec. 4）

1. **實作與調參的複雜度**：離線 RL 方法建立在 TD3 或 SAC 上，卻再加入架構、學習率、預訓練等「非演算法」的修改（Table 1）。離線設定本來就不能與環境互動來驗證這些修改，因此修改越多代價越高。拿掉這些修改後 CQL 與 Fisher-BRC 在許多任務上大幅退步（Fig. 1），而兩篇論文都沒有提到訓練／驗證切分，作者「只能假設這些修改是針對評估資料集（D4RL）調整的」（Sec. 4）。
2. **額外計算**：CQL 的 logsumexp 需要對多個取樣動作求值，Fisher-BRC 需要另外訓練生成模型（Sec. 4、Table 3）。
3. **訓練後策略的不穩定**：見 1.4 步驟 9。

### 1.3 貢獻

1. TD3+BC：對 TD3 只改策略更新的一行與狀態正規化（Eq. 3–5），唯一新增的超參數是 $\alpha$（Sec. 5、Sec. 7）。
2. 在 D4RL 上達到與當時最佳方法相當的表現，訓練時間少於一半（Table 2–3）。
3. 分析既有方法的實作調整（Table 1、Fig. 1）與離線策略的不穩定（Fig. 2–3），作為開放問題提出。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S1–S2 末尾以可收合表格提供；第三欄為首次定義的章節）

| 符號 | 意義 | 章節 |
| --- | --- | --- |
| $(\mathcal S,\mathcal A,R,p,\gamma)$ | MDP：狀態空間、動作空間、獎勵函數、轉移、折扣因子（Sec. 3） | S1 |
| $\pi$ | 策略；本文為確定性策略 $\pi:\mathcal S\to\mathcal A$ | S1 |
| $Q^\pi(s,a)$ | 在 $s$ 採取 $a$ 後依 $\pi$ 行動的期望折扣報酬 $\mathbb E_\pi[\sum_t\gamma^tr_{t+1}\mid s_0=s,a_0=a]$（Sec. 3） | S1 |
| $\mathcal D$ | 固定資料集，元素 $(s,a,r,s')$；由未知的行為策略（behavior policy）收集 | S1 |
| $Q$、$Q_1,Q_2$、$Q'$、$\pi'$ | critic（價值網路）；TD3 用兩個；加撇號者為目標網路 | S2 |
| $q$、$\hat Q$、$\bar a$ | 只在玩具（C1–C2）使用：真實期望報酬、用資料擬合的 Q、資料動作平均 | S3 |
| $\lambda$ | Q 項的權重（Eq. 3） | S5 |
| $\alpha$ | 唯一的超參數，$\lambda=\alpha/\overline{|Q|}$，預設 2.5（Eq. 5）；S5 先以一句定義（「λ 等於 α 除以 Q 的平均絕對值，α = 2.5，理由見第 06 章」），S6 給完整推導 | S5 |
| $\overline{|Q|}$ | 頁面記號：$\frac1N\sum_{(s_i,a_i)}|Q(s_i,a_i)|$，實作在 mini-batch 上估計 | S5 |
| $s_i,\mu_i,\sigma_i,\epsilon$ | 狀態第 $i$ 個特徵、其資料集平均與標準差、常數 $10^{-3}$（Eq. 4） | S7 |

#### 步驟 1：離線 RL 與外插誤差（Sec. 1、3）

- 離線 RL（Sec. 1：「historically known as batch RL」）從大小固定、由任意且可能未知的過程收集的資料集學習。好處：不需要與環境互動，可以使用既有紀錄或人類操作者的資料。
- 多數 off-policy 演算法可直接用於離線設定，但表現差，原因是外插誤差：策略評估時，對資料集中沒有的狀態–動作對估計不準；這又影響策略改進：最大化估計值的更新會選到被高估的分布外（out-of-distribution, OOD）動作（Sec. 1，引用 Fujimoto et al. 2019b）。
- 解法的共同想法：讓學到的策略靠近資料產生過程（行為策略）。名稱依「靠近」的實作方式而異：batch-constrained、KL-control、behavior-regularized、policy constraint（Sec. 1）。
- BC：以監督學習直接模仿資料集的動作；表現高度依賴收集資料的策略（Sec. 3）。
- 頁面 S3 以 C1 的玩具示意（只顯示「只最大化 $\hat Q$」）。玩具的高估來自迴歸外推，不是論文所說經由自舉累積的價值誤差；頁面稱「示意」，不稱「再現」。

#### 步驟 2：底層演算法 TD3（Eq. 1；細節為背景（論文外），出自 Fujimoto et al. 2018）

- 背景（論文外）：critic 以時序差分（TD）目標擬合 $Q$：對資料中的 $(s,a,r,s')$，讓 $Q(s,a)$ 接近 $y=r+\gamma Q'(s',\pi'(s'))$。$Q',\pi'$ 是目標網路：參數緩慢追隨 $Q,\pi$ 的副本，用來穩定目標值。用自己的估計當目標稱為自舉（bootstrapping）。
- actor 以確定性策略梯度（Silver et al. 2014）更新：調整策略，使 critic 對它選的動作給出較高的值：

$$\pi=\arg\max_\pi\ \mathbb E_{(s,a)\sim\mathcal D}\big[Q(s,\pi(s))\big]\tag{Eq. 1}$$

- 背景（論文外）：TD3 的三個設計。(a) clipped double-Q：兩個 critic，目標取較小者 $y=r+\gamma\min_{j=1,2}Q'_j(s',\tilde a)$；(b) 目標策略平滑 $\tilde a=\mathrm{clip}\big(\pi'(s')+\mathrm{clip}(\xi,-c,c),-a_{\max},a_{\max}\big)$，$\xi\sim\mathcal N(0,\sigma^2)$；(c) 延遲更新：每 2 次 critic 更新做 1 次 actor 與目標網路更新。對應的超參數在 Table 4：policy noise 0.2、noise clipping (−0.5, 0.5)、policy update frequency 2。
- Table 4 的其餘值（頁面放在 S8）：Adam；critic、actor 學習率皆 3e-4；mini-batch 256；折扣 0.99；目標更新率 5e-3；critic 與 actor 皆 2 層隱藏層、每層 256、ReLU；$\alpha=2.5$。圖說：「TD3 的超參數未從 TD3 GitHub 修改。」
- 〔本頁推導〕（頁面「進階」框）離線時 critic 目標中的 $\tilde a$ 來自 $\pi'$，不是資料中的下一個動作；$\pi$ 偏離資料時目標值就要評估 OOD 動作，誤差經由自舉傳遞。Eq. 3 只改 actor，但 actor 被拉向資料動作時，$\pi'(s')$ 也傾向靠近資料；這只在資料中的狀態 $s'$ 上、且依賴網路的泛化，沒有保證。

#### 步驟 3：既有方法的實作調整（Sec. 4、Table 1、Fig. 1）

Table 1（† 增加超參數，‡ 增加計算成本）：

| | CQL（Kumar et al. 2020） | Fisher-BRC（Kostrikov et al. 2021） | TD3+BC |
| --- | --- | --- | --- |
| 演算法調整 | 對 critic 加正則項†；以取樣近似 logsumexp‡ | 訓練生成模型†‡；critic 換成 offset 函數；offset 函數的梯度懲罰† | 加入 BC 項† |
| 實作調整 | 架構†‡；actor 學習率†；預訓練 actor；移除 SAC 熵項；評估時取樣動作取最大‡ | 架構†‡；獎勵加成†；移除 SAC 熵項 | 狀態正規化 |

- 頁面 S4 的版本（S4 在 S5 之前，術語各附一行白話）：「正則項」= 加在 critic 損失上的額外懲罰；「logsumexp」= 需要對多個取樣動作計算 Q 的一種平滑最大值；「生成模型」= 另外訓練一個模仿資料動作分布的網路；「offset 函數」與「梯度懲罰」= Fisher-BRC 改寫 critic 的方式；「SAC 熵項」= SAC 為鼓勵隨機動作而加的項；「架構」= 網路層數等。TD3+BC 欄寫「加一項（第 05 章）」與「狀態正規化（第 07 章）」。
- CQL 另有只用模仿學習訓練 actor 的預訓練階段，評估時從策略取樣一組動作取最大者；Fisher-BRC 對每筆轉移加常數獎勵加成；兩者都移除目標更新中的 SAC 熵項並修改網路架構（Sec. 4）。超參數：CQL 預訓練 40e3 步、取樣動作 10 個、actor 學習率 3e-5、3 層隱藏層（Table 5）；Fisher-BRC 5 個高斯成分、梯度懲罰 0.1、獎勵加成 5、3 層隱藏層（Table 6）。
- Fig. 1：拿掉這些實作調整後的百分比差，許多任務大幅下降。作者強調這不是要指出既有方法有根本缺陷，而是主張應偏好對底層演算法做最少的調整（Sec. 4）。Fig. 1 沒有數字資料，不重繪。

#### 步驟 4：加上 BC 項（Eq. 3；Sec. 1 的 Eq. 2 與 Eq. 3 相同）

$$\pi=\arg\max_\pi\ \mathbb E_{(s,a)\sim\mathcal D}\Big[\lambda Q(s,\pi(s))-\big(\pi(s)-a\big)^2\Big]\tag{Eq. 3}$$

- 只多一個超參數 $\lambda$ 控制正則強度；「只需修改一行程式碼」（Sec. 1）。作者的理由：最小化 KL 散度等是合理的做法，「但沒有根本的論證說明哪一種散度或距離比另一種更好」，所以選最簡單的（Sec. 5）。
- 〔本頁推導〕對固定 $s$，若資料中該狀態的動作平均為 $\bar a$、變異數為 $v$，則 $\mathbb E_a[(\pi-a)^2]=(\pi-\bar a)^2+v$；BC 項使策略輸出接近資料動作的平均，懲罰隨距離平方增加。$\lambda\to0$ 時 $\pi\to\bar a$（純 BC），$\lambda\to\infty$ 時回到 Eq. 1（純 TD3）。
- 〔本頁推導〕$v$ 是常數，不影響 $\arg\max$；資料集若是多個策略的混合（例如 medium-expert），$\bar a$ 可能落在兩群動作之間。頁面只在 S5 正文提一句。

#### 步驟 5：$\lambda$ 的正規化（Eq. 5）

- 作者觀察：RL（最大化 $Q$）與模仿（最小化 BC 項）的平衡對 $Q$ 的尺度很敏感。若動作範圍是 $[-1,1]$，BC 項最多為 4，而 $Q$ 的範圍取決於獎勵尺度（Sec. 5）。因此定義

$$\lambda=\frac{\alpha}{\frac1N\sum_{(s_i,a_i)}|Q(s_i,a_i)|}\tag{Eq. 5}$$

- 實務上在 mini-batch 上估計平均，不在整個資料集；這一項雖含 $Q$，但不對它微分，只用來縮放損失。附帶好處：$\nabla_aQ(s,a)$ 也與 $Q$ 的尺度有關，正規化也把各任務的學習率正規化（Sec. 5）。$\alpha=2.5$。
- 「最多為 4」〔本頁推導〕：每個動作維度差的平方最大 $2^2=4$；若 BC 項是 $\|\pi(s)-a\|^2$ 的總和，上限是 $4\cdot\dim\mathcal A$。官方程式碼（背景（論文外），`github.com/sfujim/TD3_BC`）用 `F.mse_loss(pi, action)`，對 batch 與動作維度取平均，上限因此是 4。
- 官方程式碼（背景（論文外））：`pi = actor(state); Q = critic.Q1(state, pi); lmbda = alpha/Q.abs().mean().detach(); actor_loss = -lmbda*Q.mean() + F.mse_loss(pi, action)`。$\overline{|Q|}$ 用 $Q(s,\pi(s))$，而 Eq. 5 寫 $Q(s_i,a_i)$。頁面以 Eq. 5 為主，常見錯誤第 3 條註明差異（R2）。
- 〔本頁推導〕理想化的尺度不變：若所有獎勵乘上 $c>0$ 且 critic 恰好學到 $cQ$，則 $\lambda$ 除以 $c$，$\lambda Q$ 不變，最佳策略不變。實際網路（初始化、Adam、自舉、終止遮罩）不會恰好學到 $cQ$，因此只是近似不變。論文本身沒有宣稱不變性，只說平衡「對 Q 的尺度很敏感」。
- 〔本頁推導〕平移不變性不成立：$Q$ 加上常數會改變 $\overline{|Q|}$，因此改變 $\lambda$ 與最佳策略。在多步 MDP 中獎勵加上 $b$ 大約使 $Q$ 平移 $b/(1-\gamma)$（$\gamma=0.99$ 時約 $100b$），還受終止狀態影響；C2 的一步玩具中 $b$ 直接是 $Q$ 的平移。
- App. B 的選擇建議：以領域知識直接取 $\lambda\approx\alpha/$預期平均價值；若價值函數因外插誤差發散，就減小 $\lambda$（加重 BC）；若表現接近行為策略，就考慮增大 $\lambda$。

#### 步驟 6：狀態正規化（Eq. 4）

$$s_i=\frac{s_i-\mu_i}{\sigma_i+\epsilon},\qquad \epsilon=10^{-3}\tag{Eq. 4}$$

- $\mu_i,\sigma_i$ 是第 $i$ 個特徵在資料集上的平均與標準差。作者說明這是許多深度 RL 演算法常用的實作細節（Raffin et al. 2019），列出它是為了完全透明，也因為它在離線 RL 中帶來不小的效益，而且資料集固定，特別適合（Sec. 5）。
- 論文沒有解釋效益的機制。〔本頁推導〕（頁面「進階」框，文字、無元件）一個標準解釋：特徵尺度差異大或平均遠離 0 時，損失在不同參數方向上的曲率差異大，梯度下降需要很小的學習率與很多步；正規化使各特徵尺度相同、平均為 0。限制：TD3 用 Adam（Table 4），Adam 對每個參數各自縮放步長，會抵消一部分純尺度差異（平均偏移與特徵相關則不會），所以這個解釋的量級不能外推到 TD3+BC。
- 效益的範圍：消融中拿掉正規化影響最小，但在多個任務上仍有幫助（Sec. 6、Fig. 5）；AntMaze 中正規化有害，因此不使用（App. C.1）；加到 CQL 與 Fisher-BRC 上影響很小（App. C.2、Fig. 7）。
- 〔本頁推導〕實作要點：$\mu,\sigma$ 只從資料集算一次，評估時環境給的狀態也要用同一組 $\mu,\sigma$ 正規化；$\epsilon$ 避免常數特徵（$\sigma_i=0$）除以 0。

#### 步驟 7：完整演算法（Sec. 5）

「Eq. 3、4、5 就是對 TD3 的全部修改，在多數程式庫中只需改幾行」（Sec. 5）。頁面 S8 以偽碼呈現（〔本頁推導〕依 Eq. 3–5 與背景 TD3 寫成，不是論文原文），逐行標示「TD3 原有」或「TD3+BC 新增」：

```
正規化：對資料集所有狀態求 μ, σ；s ← (s − μ)/(σ + 1e-3)        # 新增（Eq. 4）
for t = 1 … 1e6:
  取 mini-batch (s, a, r, s′)，256 筆
  ã ← clip(π′(s′) + clip(0.2·N(0,1), −0.5, 0.5), −1, 1)          # TD3
  y ← r + 0.99 · min(Q′₁(s′,ã), Q′₂(s′,ã))                        # TD3
  更新 Q₁, Q₂ 最小化 (Q_j(s,a) − y)²                              # TD3
  if t mod 2 == 0:
    λ ← 2.5 / mean|Q₁(s, a)|（不微分）                             # 新增（Eq. 5；官方程式碼用 Q₁(s, π(s))）
    更新 π 最大化 mean[λ·Q₁(s, π(s)) − (π(s) − a)²]                # 新增 BC 項（Eq. 3）
    目標網路 θ′ ← 0.005·θ + 0.995·θ′                              # TD3
```

（終止狀態的 `not_done` 遮罩省略，頁面附註一句。偽碼中的下標為預先組合的 Unicode 字元 ₁₂，不是組合附加符號。）

#### 步驟 8：實驗（Sec. 6、App. B–C）

- 資料集（頁面 S9 逐一定義；背景（論文外），D4RL）：random = 隨機初始化策略收集；medium = 訓練到中等表現的策略收集；medium-replay = 訓練到中等表現過程中的回放緩衝區；medium-expert = medium 與 expert 各半混合；expert = 訓練完成的策略收集。正規化分數 $=100\times\frac{\text{分數}-\text{隨機策略分數}}{\text{專家策略分數}-\text{隨機策略分數}}$。
- 比較方法（S9 各一行）：BC（模仿資料動作，作者自己的實作）；BRAC-p（以散度懲罰策略偏離行為策略）；AWAC（以優勢加權的 BC）；CQL（在 critic 上加保守懲罰）；Fisher-BRC（以 Fisher 散度正則化 critic）。CQL、Fisher-BRC 以作者程式碼重跑；BRAC 取自 D4RL 與 CQL 論文；AWAC 取自其論文（Sec. 6）。
- 協定：D4RL v0；每個演算法訓練 1 百萬步，每 5000 步評估一次，每次 10 個 episode；Table 2 為最後 10 次評估、5 個 seed 的平均正規化分數；論文寫「± 為 seed 間標準差」（Table 2 圖說）。
- 〔本頁推導〕Total 的 ± 等於 15 列 ± 的總和（BC 91.5、CQL 61.5、Fisher-BRC 108.3、TD3+BC 33.4 皆完全相等，`c4_out.txt`），不是總分在 seed 間的標準差；Table 9 的 ±（57.4、45.6、16.6）也是如此。頁面不把 Total 的 ± 解釋為穩定度。
- 調參：只調 $\alpha\in\{1,2,2.5,3,4\}$，在 Hopper-medium-v0 與 Hopper-expert-v0 上、用一個不出現在最終結果的 seed（App. B）。
- Table 2 總分：BC 595.3、BRAC-p 284.1、AWAC「-」（無 medium-replay）、CQL 764.3、Fisher-BRC 974.6、TD3+BC 979.3（TD3+BC 比 Fisher-BRC 高 4.7，以論文 Total 計）。弱點：random 資料，例如 HalfCheetah-random TD3+BC 10.2±1.3，Fisher-BRC 32.2±2.2、BRAC-p 23.5。逐列核對：TD3+BC 在 15 個任務上都不低於 BC（`c4_out.txt`），與 Sec. 6「從不低於 vanilla BC，即使在 expert 任務」一致。
- 學習曲線（Fig. 4）：TD3+BC 與 Fisher-BRC 的學習速度與最終表現相近，不需要預訓練生成模型。腳註：CQL 作者程式碼的結果低於其論文報告值。
- 執行時間（Table 3；GTX 1080 + i7-6700K，1 百萬步，不含評估）：

  | | 實作調整 | 演算法調整 | 總計 |
  | --- | --- | --- | --- |
  | CQL（GitHub） | 25m | 1h 29m | 4h 11m |
  | Fisher-BRC（GitHub，TensorFlow） | 39m | 33m | 2h 12m |
  | Fisher-BRC（作者以 PyTorch 重寫） | 15m | 58m | 2h 8m |
  | TD3+BC | < 1s | < 5s | 39m |

  〔本頁推導〕總計減兩欄得底層演算法時間約 2h 17m、1h 0m、55m、39m；CQL 總計約為 TD3+BC 的 6.4 倍，Fisher-BRC（PyTorch）約 3.3 倍。
- 消融（Fig. 5）：沒有 BC 項時 RL 演算法本身不足以達到高表現（部分 random 資料集除外）；拿掉正規化影響最小但仍有幫助。$\alpha$ 敏感度（Fig. 6）：$\alpha=2$ 與 $3$ 幾乎沒有差異；偏向模仿（$\alpha=1$）或偏向 RL（$\alpha=4$）時部分任務開始退步。
- App. C.1：random 與 expert 各取一半混合（Table 7）：TD3+BC HalfCheetah 101.9±7.6、Hopper 112.2±0.3、Walker2d 28.8±23.4；CQL 73.3、110.8、100.3；Fisher-BRC 105.8、111.9、32.0。作者的假設：Walker2d 落後與 BC 在 Walker2d-expert 上表現差有關。AntMaze（Table 8，不使用正規化）：Umaze 78.6±33.3、Umaze-Diverse 71.4±20.7、Medium-Diverse 10.6、Medium-Play 3.0、Large-Diverse 0.2、Large-Play 0.0。
- App. C.3：D4RL v2 上與 Decision Transformer（DT，以 Transformer 做序列建模的離線 RL）比較（Table 9）。在 DT 有結果的 9 個任務上 Total(DT) 為 DT 672.6、TD3+BC 677.4；TD3+BC 全部 15 個任務 1013.2，沒有額外調參。DT 數字取自其論文、3 個 seed；TD3+BC 5 個 seed（Table 9 圖說）。訓練時間（Fig. 8）：DT 16h 21m，DT 只訓練 100k 步時 1h 38m，TD3+BC 39m。

#### 步驟 9：訓練後策略的不穩定（Sec. 4、Fig. 2–3）

- 線上 RL 可以用之前的 checkpoint 或繼續訓練；離線 RL 依定義只評估一次，單一策略的重要性大增（Sec. 4）。
- Fig. 2：最後一次評估的 10 個 episode 中最差 episode 的百分比差（單一時間點內的變異）；Fig. 3：最後 10 次評估中最差一次的百分比差（一段時間內的變異）。線上 TD3 的變異小，所有離線演算法（BC、CQL、Fisher-BRC、TD3+BC）的變異都很大。推定：百分比差的比較基準是平均表現（圖說未寫明，R5）。
- 作者認為：這不太可能來自策略熵（TD3+BC 的策略是確定性的），假設來自分布偏移與對未見狀態的泛化不佳（Ross et al. 2011）；連極簡的 TD3+BC 也重現此現象，作者據此認為這是所有離線訓練共有的根本問題（Sec. 4）。這也質疑只報告最終策略平均值的評估慣例。

#### 步驟 10：限制（Sec. 4、Sec. 7（結論）、App. A（Broader Impact）、App. C.1）

- 不穩定問題未解（Sec. 4、Sec. 7）；random 資料表現弱（App. C.1 開頭：「TD3+BC 的一個疑慮是在 random 資料上表現差」）。
- App. A（Broader Impact）：相對既有最佳方法的表現增益很小，作者預期影響在於可及性與易用性，而不是擴大應用範圍；離線 RL 受資料集偏差影響。

---

## 2. 學習目標（驗收測試）

每個參考答案都必須寫在指定章節的正文（不能只出現在元件、表格、圖或答案框）。每章末以 `<details>`「檢查理解」照錄題目與參考答案。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 定義離線 RL | 離線 RL 與一般 RL 的差別？好處是什麼？ | 只從固定、由未知過程收集的資料集學習，不能與環境互動；不需要承擔與環境互動的成本與風險，可用既有紀錄（Sec. 1、3） | S1 |
| LO2 | 說明外插誤差 | 為什麼 off-policy 演算法直接用於離線設定表現差？ | 外插誤差：對資料集中沒有的狀態–動作對估計不準，最大化估計值的策略更新會選到被高估的 OOD 動作（Sec. 1、3） | S3 |
| LO3 | 寫出 TD3 的策略更新與三個設計 | TD3 的 actor 最大化什麼？TD3 有哪三個設計？ | $\mathbb E[Q(s,\pi(s))]$（Eq. 1）；clipped double-Q 目標（兩個 critic 取較小者）、目標策略平滑（噪聲 0.2、截斷 ±0.5）、延遲更新（每 2 次 critic 更新做 1 次 actor 更新）（背景，Table 4） | S2 |
| LO4 | 列舉既有方法的複雜度 | CQL 與 Fisher-BRC 各列出哪些實作調整？拿掉後如何？ | CQL：架構、actor 學習率、預訓練 actor、移除 SAC 熵項、評估時取樣取最大；Fisher-BRC：架構、獎勵加成、移除 SAC 熵項；拿掉後許多任務大幅退步（Table 1、Fig. 1） | S4 |
| LO5 | 寫出 TD3+BC 的目標 | Eq. 3 是什麼？$\lambda\to0$ 與 $\lambda\to\infty$ 各得到什麼？ | $\max_\pi\mathbb E[\lambda Q(s,\pi(s))-(\pi(s)-a)^2]$；$\lambda\to0$ 為純 BC（$\pi\to$ 資料動作平均），$\lambda\to\infty$ 為 TD3（Eq. 3、〔本頁推導〕） | S5 |
| LO6 | 解釋玩具中的權衡 | C1 預設玩具（seed 7，推定常數）中，只最大化 $\hat Q$、純 BC、TD3+BC（$\alpha=2.5$）各選哪個動作？真實價值？效益隨 α 單調嗎？ | 只最大化 $\hat Q$：$a=1$，$\hat Q=7.18$ 但真實 4.03；純 BC：$a=-0.268$，4.30；TD3+BC（α = 2.5）：$a=0.381$，4.83。不單調：α = 2 時 4.99，α = 3 時 4.25 已低於純 BC，α ≥ 3.4 時策略停在動作上界 1 | S5 |
| LO7 | 說明 $\lambda$ 正規化 | $\lambda$ 怎麼算？$\alpha$ 是多少？為什麼要除以 $\overline{|Q|}$？ | $\lambda=\alpha/\frac1N\sum|Q(s_i,a_i)|$，$\alpha=2.5$，在 mini-batch 上估計、不微分；BC 項在動作範圍 $[-1,1]$ 下最多 4，$Q$ 的範圍隨獎勵尺度改變，也使 $\nabla_aQ$ 的尺度一致（Eq. 5、Sec. 5） | S6 |
| LO8 | 知道正規化的不變性與其界限 | 獎勵乘上 10 或 Q 加上常數，TD3+BC 的最佳策略會改變嗎？ | 理想化下（critic 恰好學到 $cQ$）乘上正數不變，實際網路只近似不變；Q 加上常數會改變 $\overline{|Q|}$，因而改變 $\lambda$ 與策略；多步問題中獎勵加 $b$ 約使 Q 加 $b/(1-\gamma)$（〔本頁推導〕） | S6 |
| LO9 | 寫出狀態正規化與其適用範圍 | Eq. 4 是什麼？為什麼適合離線？哪裡不用？ | $s_i\leftarrow(s_i-\mu_i)/(\sigma_i+\epsilon)$，$\epsilon=10^{-3}$；資料集固定，統計量只需算一次；AntMaze 中有害而不使用，消融中影響最小（Eq. 4、Sec. 5–6、App. C.1） | S7 |
| LO10 | 讀出主要結果 | Table 2 中 TD3+BC、Fisher-BRC、CQL 的總分？TD3+BC 的弱點？Total 的 ± 是什麼？ | 979.3、974.6、764.3；random 資料較弱（HalfCheetah-random 10.2 vs Fisher-BRC 32.2）；從不低於 BC；Total 的 ± 等於各列 ± 之和（本頁核算），不是總分的標準差（Table 2、Sec. 6） | S10 |
| LO11 | 讀出計算成本與調參方式 | 三者的訓練時間？$\alpha$ 如何選出？$\alpha$ 的敏感度？ | TD3+BC 39m、Fisher-BRC 2h 8m（PyTorch）、CQL 4h 11m；$\alpha\in\{1,2,2.5,3,4\}$ 在 Hopper-medium／expert 單一未使用 seed 上調；2 與 3 無差異，1 與 4 部分任務退步（Table 3、App. B、Fig. 6） | S9、S10 |
| LO12 | 說明不穩定問題 | Fig. 2 與 Fig. 3 量的是什麼？作者的假設？ | Fig. 2：最後一次評估 10 個 episode 中最差者；Fig. 3：最後 10 次評估中最差者；離線策略變異都大，線上 TD3 小；作者假設來自分布偏移與對未見狀態泛化不佳，不是策略熵（Sec. 4） | S11 |

LO11 的分配：調參協定在 S9 正文，時間與 α 敏感度在 S10 正文。

---

## 3. 頁面大綱

單欄閱讀（680–720px），圖版加寬到最大 1080px。章節編號以 JetBrains Mono 顯示（`01 · …`），標題用陳述句。每個符號在首次使用的章節定義（符號表第三欄）；每個元件放在它用到的概念之後。

| # | id | 標題 | 內容 | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈TD3+BC：在 TD3 上加一項行為複製的離線強化學習〉 | 副標：「這篇論文在一個既有的強化學習演算法上只加一項模仿資料動作的損失，再把狀態正規化，就達到當時最佳方法的表現，訓練時間少於一半。」三張事實卡（見下）；不放公式。連結：arXiv、`../index.html#paper-570`（在關聯圖中查看）、`td3bc-lineage-overview.html`、程式碼 `https://github.com/sfujim/TD3_BC` | 1 螢幕 |
| S1 | `offline` | 01 · 離線 RL 只能使用固定的資料集 | MDP、策略、$Q^\pi$ 的定義（字面敘述＋一式）；線上 vs 離線（靜態 SVG：上列「agent ⇄ 環境」迴圈，下列「資料集 → agent」單向）；行為策略；BC 定義。答 LO1。延伸：`rl-mdp-q-learning-basics.html`、`offline-rl-distribution-shift.html` | 350–450 字 |
| S2 | `td3` | 02 · TD3 用 critic 的估計更新確定性策略 | 先一段 critic：TD 目標 $r+\gamma Q'(s',\pi'(s'))$、目標網路、自舉；再 Eq. 1 與一句白話；三個設計各一句（背景標籤）；「進階」框放〔本頁推導〕critic 目標使用 $\pi'$。末尾符號表。答 LO3。延伸：`ddpg-td3-actor-critic.html` | 400–500 字 |
| S3 | `extrap` | 03 · 外插誤差使策略選到資料中沒有的動作 | 外插誤差定義；四種名稱；玩具 3 句介紹（單一狀態、一步問題；$q$ 是真實期望報酬；$\hat Q$ 是只用資料擬合的二次曲線；$\bar a$ 是資料動作平均）；C1 的 S3 實例（只顯示「只最大化 $\hat Q$」，有「重新抽樣」，無步驟列）。答 LO2。延伸：`batch-constrained-policy-constraint-methods.html` | 350–450 字 + C1（S3 實例） |
| S4 | `complex` | 04 · 既有方法加入許多實作調整 | Table 1 的 S4 版本（1.4 步驟 3，含白話）、Fig. 1 文字描述、Table 5–6 摘錄、訓練／驗證切分的論點、計算成本預告。答 LO4。延伸：`rl-implementation-details-reproducibility.html` | 350–450 字 + 表 |
| S5 | `bc` | 05 · TD3+BC 在策略更新中加入行為複製項 | α 與 $\overline{|Q|}$ 的一句定義（符號表）；「差異」方塊：Eq. 1 → Eq. 3，新增部分以 `--accent-soft` 底色標示；作者的理由；〔本頁推導〕$(\pi-\bar a)^2+v$；兩個極限；混合資料的一句；C1 的 S5 實例；正文寫出 LO6 的數字與「不單調」。答 LO5、LO6。延伸：`rl-plus-behavior-cloning-prior-work.html`、`advantage-weighted-bc-methods.html` | 450–550 字 + C1 |
| S6 | `lambda` | 06 · λ 以 Q 的平均絕對值正規化 | 「最多為 4」（含維度說明）；Eq. 5；mini-batch、不微分、學習率；理想化尺度不變與平移不成立（〔本頁推導〕）；C2（標「理想化」）；App. B 的選擇建議。答 LO7、LO8 | 400–500 字 + C2 |
| S7 | `norm` | 07 · 狀態特徵正規化為平均 0、標準差 1 | Eq. 4、ε；作者的兩個理由；效益範圍（Fig. 5、Fig. 7、AntMaze）；評估時同一組 μ, σ；「進階」框：〔本頁推導〕曲率解釋與 Adam 的限制（文字）。答 LO9 | 300–400 字 |
| S8 | `algo` | 08 · 完整演算法只比 TD3 多幾行 | 1.4 步驟 7 的偽碼（≥ 768px 兩欄：左程式碼、右「TD3 原有／新增」標籤與說明；< 768px 每行下方接說明）；Table 4 全表；常見錯誤 6 條 | 300–400 字 + 偽碼 + 表 |
| S9 | `setup` | 09 · 實驗使用 D4RL 的 15 個資料集 | 五種資料集定義、正規化分數（背景）、比較方法各一行、協定、調參協定（LO11 前半） | 300–400 字 |
| S10 | `results` | 10 · TD3+BC 與 Fisher-BRC 相當，訓練時間少於一半 | C4；正文寫出 LO10、LO11 後半的數字；± 為各列 ± 之和的說明；Fig. 4、5、6 文字描述；App. C.1、C.3 摘要（含 DT 的 seed 數）。延伸：`td3bc-algorithm-and-d4rl.html` | 400–500 字 + C4 |
| S11 | `unstable` | 11 · 離線訓練的策略表現變異很大 | Fig. 2、3 的定義與結論、作者的假設、對評估慣例的質疑；限制（Sec. 7、App. A）。答 LO12 | 250–350 字 |
| S12 | `history` | 12 · 研究脈絡 | 時間線（6.4）+ 一段文字：DDPG → TD3 → BCQ／BEAR／BRAC（策略約束）→ AWR／CRR（加權 BC）→ D4RL → CQL、Fisher-BRC → TD3+BC | 時間線 + 100–150 字 |
| S13 | `next` | 13 · 延伸閱讀 | 研究筆記清單（6.4）、arXiv、程式碼 | 連結清單 |

Hero 事實卡（只用一般詞彙）：
1. 「離線強化學習只能從一個固定的資料集學習。對資料中沒有出現的動作，學到的價值估計常常過高。」
2. 「這篇論文在策略的訓練目標中加一項：策略輸出的動作與資料中動作之差的平方。這一項使策略輸出接近資料中的動作。」
3. 「在 15 個標準測試資料集上，這個方法的總分與當時最好的方法相當（979.3 對 974.6），訓練時間是 39 分鐘對 2 小時 8 分（後者為作者以 PyTorch 重寫的版本）。」

常見錯誤（S8，逐條，每條附出處或標籤）：
1. 評估時忘了用資料集的 μ, σ 正規化環境狀態（〔本頁推導〕，Eq. 4）。
2. 讓梯度流過 $\lambda$ 的分母；論文寫明不微分（Sec. 5）。
3. $\overline{|Q|}$ 用 $Q(s_i,a_i)$（Eq. 5）或 $Q(s,\pi(s))$（官方程式碼，背景（論文外））——兩者不同，重現時以所用程式碼為準。
4. BC 項用平方和而非平均，使「最多為 4」的比例改變，$\alpha=2.5$ 不再對應論文設定（〔本頁推導〕）。
5. 在 AntMaze 套用狀態正規化；論文在該領域不用（App. C.1）。
6. 只看最終策略的平均分數；論文顯示離線策略的變異很大（Fig. 2–3）。

---

## 4. 互動元件規格

### 4.0 共用：玩具問題（C1、C2 共用同一份函式與資料）

頁面寫出的簡化：**單一狀態、一步的問題**（沒有轉移，$Q$ 就是期望獎勵），動作 $a\in[-1,1]$ 一維；critic 換成二次多項式迴歸。玩具只示意「擬合的 Q 在資料外不可靠」與 BC 項的權衡；論文的外插誤差還經由自舉累積，玩具沒有這一層。所有常數為推定（為了讓 α = 2.5 落在好的區域而校準，R1）。

- PRNG：`mulberry32(seed)`；`gauss(r)`：Box–Muller，每次取兩個均勻數 $u_1,u_2$，回傳 $\sqrt{-2\ln(1-u_1)}\cos(2\pi u_2)$（只用 cos 分支）。預設 seed = 7，「重新抽樣」每次 seed + 1（C1 兩個實例與 C2 共用 seed）。
- 真實期望報酬：$q(a)=B_0+\exp\!\big(-\frac{(a-0.2)^2}{2\cdot0.3^2}\big)$，$B_0=4$（基準值；真實任務的 $Q$ 通常遠離 0，這個基準讓 $\overline{|Q|}$ 有意義）。
- 資料（$N=40$）：依序對每筆抽 $a_i=\mathrm{clip}(-0.3+0.3\,g,-1,1)$，再抽 $r_i=q(a_i)+0.05\,g$（$g$ 為 `gauss`，a、r 交錯抽）。
- 擬合：$\bar a=\frac1N\sum a_i$，$v=\frac1N\sum(a_i-\bar a)^2$，$s_a=\sqrt v$；若 $s_a<10^{-6}$，元件顯示「資料動作相同，無法擬合」並停止繪圖（預設與 seed 1–200 都不會觸發）。特徵 $\phi(a)=(1,u,u^2)$，$u=(a-\bar a)/s_a$；解 $(\frac1N\Phi^\top\Phi+\mathrm{diag}(0,10^{-3},10^{-3}))c=\frac1N\Phi^\top r$（3×3 高斯消去、部分選主元）；$\hat Q(a)=c\cdot\phi(a)$。
- 動作網格：$G=2001$ 點，$a_j=-1+2j/2000$。
- TD3+BC 目標（〔本頁推導〕的資料平均形式）：$J(\pi)=\lambda\hat Q(\pi)-(\pi-\bar a)^2-v$，$\lambda=\alpha/\max(\overline{|\hat Q|},10^{-12})$，$\overline{|\hat Q|}=\frac1N\sum_i|\hat Q(a_i)|$。$\pi^*=\arg\max_jJ(a_j)$，同值取第一個索引。「只最大化 $\hat Q$」= $\arg\max_j\hat Q(a_j)$。
- 〔本頁推導〕內部最佳解與截斷：$\hat Q$ 是 $u$ 的二次式，$J'(\pi)=0$ 得 $\pi_{\rm int}=\bar a+\dfrac{\lambda c_1s_a}{2s_a^2-2\lambda c_2}$，在 $2s_a^2-2\lambda c_2>0$（$J$ 為凹函數）時成立。$\lambda$ 增加時 $\pi_{\rm int}$ 連續右移；越過動作上界 1 後，$\pi^*$ 停在 1（被動作範圍截斷）。預設資料：$\pi_{\rm int}$ 在 α ≈ 3.32 到達 1；$J$ 要到 α ≈ 5.07 才變成凸函數，因此 α ∈ [0, 5.07) 內 $\pi^*$ 都是連續的。頁面正文用這一段說明「為什麼大 α 時策略停在 1」，不使用「跳躍」或「目標變凸」的說法。

### C1 · 外插誤差與 BC 項的權衡（S3 實例、S5 實例）

- **教的概念**：只最大化 $\hat Q$ 會選到資料外、被高估的動作（S3）；BC 項使策略輸出接近資料動作，$\alpha$ 決定位置；中間的 $\alpha$ 可以勝過純 BC 與純 TD3，且效益隨 α 不單調（S5）。
- **兩個實例**：`#c1a`（S3）只有標題「只最大化擬合的 Q」與按鈕「重新抽樣」，沒有步驟列，不顯示 $J$、副圖與 $\alpha$ 控制項。`#c1b`（S5）有：滑桿 $\alpha\in[0,10]$，步長 0.1，預設 2.5；chip（`role="radiogroup"`）「α = 0（純 BC）」「α = 2.5（論文）」「只有 Q（α → ∞）」；按鈕「重新抽樣」。
- **顯示**（SVG 內只用中文字詞標籤，不用含組合附加符號的字元）：
  - 主圖：橫軸 $a\in[-1,1]$；縱軸依目前 seed 取 $[\min,\max]$（$q$ 與 $\hat Q$ 在網格上的極值），上下各留 5%。資料動作以軸下方的短刻度（rug）呈現，並以淺色帶標出 $[\min a_i,\max a_i]$「資料範圍」。曲線：真實期望報酬（`--ink-3` 虛線，標「真實值」）、擬合值（`--c-q` 實線，標「擬合的 Q（只用資料）」）。S5 實例另畫 $J(\pi)$ 平移到同一縱軸（$J$ 的最大值對齊 $\hat Q(\pi^*)$，形狀不變，`--accent` 細實線，標「TD3+BC 目標（平移後）」）。
  - 標記（形狀＋文字）：方形「資料平均動作」、三角形「只最大化擬合值」、圓形「TD3+BC」（S5）。每個標記有一條到真實值曲線的垂直細線。
  - 副圖（S5）：橫軸 $\alpha\in[0,10]$（101 點，步長 0.1），縱軸 $q(\pi^*(\alpha))$；水平虛線標「純 BC」與「只最大化擬合值」；目前 $\alpha$ 為垂直細線；$\pi_{\rm int}$ 到達 1 的 α 以短刻度標「到達動作上界」。
  - 讀數：SVG 外的 HTML 表。列標題為靜態 KaTeX（$\bar a$、$\lambda$、$\pi$、$\hat Q(\pi)$、$q(\pi)$、$\hat Q-q$），欄為三種動作；儲存格只放純數字（3 位小數）。`aria-label` 用中文詞（「TD3+BC 動作 0.381，真實值 4.834」）。
  - 圖下：「讀數由本頁依 Eq. 3、Eq. 5 即時計算；玩具為單一狀態、一步、二次多項式 critic，常數為推定，只作示意。」
- **讀者應觀察到**：(1) 擬合值在資料範圍內貼近真實值，範圍外往上彎，在 $a=1$ 高估約 3.2；(2) 只最大化擬合值選 $a=1$，真實值比純 BC 還低；(3) $\alpha$ 從 0 增加，策略連續離開 $\bar a$，真實值先升後降，α ≈ 3.32 之後停在動作上界 1；(4) 換 seed 時細節改變，但「只最大化擬合值落在資料範圍外」的情形常見（正文不引用 seed 統計，R1）。

### C2 · 獎勵尺度、Q 的平移與 λ 正規化（S6，理想化）

- **教的概念**：在理想化擬合下，Eq. 5 讓結果與 Q 的尺度無關；固定 $\lambda$ 時尺度決定偏向 RL 或 BC；正規化對平移不成立（〔本頁推導〕）。圖版標題旁加「理想化」pill，圖下寫：「這裡假設 critic 恰好學到 c(Q̂ + b)；真實的 TD3 critic 只會近似如此。b 是 Q 的平移；在多步任務中，獎勵加 b 約使 Q 加 b/(1−γ)。」
- **輸入**：滑桿 $\log_{10}c\in[-2,2]$，步長 0.1，預設 0（顯示 $c$）；滑桿 $b\in[-8,8]$，步長 0.1，預設 0；分段按鈕「λ = α ÷ Q 的平均絕對值（論文）」／「固定 λ = α」；$\alpha$ 固定 2.5（顯示）。
- **計算**：$\hat Q_{c,b}=c(\hat Q+b)$（理想化假設；對迴歸而言，ridge 不懲罰截距且對目標線性，所以對獎勵 $c(r+b)$ 重擬合恰好得到這個式子〔本頁推導〕）。$J$、$\pi^*$ 同 4.0（不含常數 $v$）。另算在 $a=\bar a$ 處 Q 項的梯度 $g=\lambda c\,c_1/s_a$。
- **顯示**：主圖橫軸 $\log_{10}c\in[-2,2]$，取樣 401 點（步長 0.01），縱軸 $\pi^*\in[-1,1]$；兩條折線「正規化」（`--accent` 實線）與「固定 λ」（`--ink-3` 虛線）；目前 $c$ 為垂直細線；資料範圍以淺色帶標示；目前 $b$ 下重算。兩條曲線都連續（`c2.py` 檢查：固定 λ 在 $\log_{10}c\in[-0.7,-0.52]$ 陡升，−0.52 起停在 1；401 點取樣足以畫出陡升而不需斷線）。讀數為 SVG 外 HTML 表（Q 的平均絕對值、λ、π\*、真實值、g，4 位有效數字）。
- **數值保護與警示**：分母以 $\max(\cdot,10^{-12})$ 防止除以 0（滑桿範圍內不會觸發）。c 在 λQ 中相消，所以警示只看 $m(b)=\frac1N\sum_i|\hat Q(a_i)+b|$：$m(b)<0.5$（即 λ > 2α = 5）時在讀數下顯示「Q 的平均絕對值小於 0.5，λ 大於 5」。`c2.py`：seed 7 在滑桿範圍的最小值為 0.2810（b = −4.3），$m<0.5$ 的範圍為 b ∈ [−4.8, −3.9]。
- **讀者應觀察到**：(1) 正規化時 $\pi^*$ 對所有 $c$ 都是 0.381，$g$ 恆為 0.6576（理想化下）；(2) 固定 λ 時 $c=0.01$ 幾乎是 BC（−0.253），$c$ 增加時策略連續右移，$\log_{10}c\ge-0.52$ 停在 $a=1$，$g$ 與 $c$ 成正比；(3) 正規化時 $b=+4$ 使 π\* 變為 −0.036、$b=-2$ 使 π\* 停在 1：Q 的基準值改變 λ。

### C4 · 結果瀏覽器（S10）

- **教的概念**：Table 2、Table 3、Table 9 / Fig. 8 的數字；TD3+BC 與 Fisher-BRC 相當、random 較弱、計算少。
- **輸入**：分頁（`role="tablist"`）「D4RL v0 分數（Table 2）」「訓練時間（Table 3、Fig. 8）」「v2 與 DT（Table 9）」。Table 2 分頁：資料集分段按鈕（全部／random／medium／medium-replay／medium-expert／expert；≥ 600px 預設「全部」，< 600px 移除「全部」、預設 random，一次只顯示一個資料集）；演算法勾選框（6 個，預設 BC、CQL、Fisher-BRC、TD3+BC）。
- **顯示**：
  - Table 2：每個任務一組水平長條（填充圖樣＋直接標籤：TD3+BC `--accent` 實心、Fisher-BRC 斜線、CQL 點、BC 空心、BRAC-p 橫線、AWAC 交叉線；pattern 線色用 `currentColor`，深色模式自動跟隨），有 ± 的畫誤差鬚。「-」顯示「無資料」。下方總分列照錄論文 Total，附註：「Total 的 ± 等於各列 ± 之和（本頁核算），不是總分的標準差。」另一行〔本頁推導〕：「Fisher-BRC 各列加總為 975.2，論文 Total 為 974.6。」旁附完整數字表（`<table>`，捲動容器）。
  - 訓練時間：堆疊水平長條（底層演算法〔本頁推導〕／實作調整／演算法調整），四個方法 + DT、DT（100k）（DT 只有總計，單色長條並註「Fig. 8，只有總計」）；軸為分鐘。
  - Table 9：9 個有 DT 結果的任務並排長條 + 兩個 Total(DT)，註「DT 取自其論文、3 個 seed；TD3+BC 5 個 seed」。
- **資料**：6.3 的 JSON 常數。
- **讀者應觀察到**：TD3+BC 總分最高但只比 Fisher-BRC 高 4.7；Fisher-BRC 在 Hopper-medium-expert（±43.3）與 Walker2d-expert（±32.4）的 seed 間差異大；random 列 TD3+BC 低於 Fisher-BRC 與 BRAC-p；時間約為 Fisher-BRC 的 0.30。

### 4.5 共同的無障礙與互動規則

沿用 406 頁的 4.5 規則：靜態 TeX 用 KaTeX（`.katex-display{overflow-x:auto; overflow-y:hidden}`，Eq. 3、Eq. 5 在 390px 可橫捲並有右側漸層提示），動態文字一律用不含組合附加符號的 Unicode 或中文詞；SVG 內不放 $\hat Q$、$\bar a$ 等組合字元；< 600px 單欄、`ResizeObserver` 重畫、圖上文字 12–13px；`input` 以 rAF 合併重畫；live region 與 SVG `aria-label` 只在 `change` 或停止輸入 500ms 後更新；分段按鈕 roving tabindex；觸控區 ≥ 44px；不在圖上做點擊或拖曳；reduced-motion 下無補間；焦點 2px `--accent`。本頁沒有自動播放的動畫。計算量：C1 換 seed 時重算 101 × 2001 次（約 20 萬次求值），α 變動只查快取；C2 每次重畫 401 × 2001 次（約 80 萬次乘加，< 10ms），都在主執行緒同步完成。

---

## 5. 驗收數值表

全部由 `verify/` 腳本產生。容許誤差：動作 ±0.001（網格步長 0.001）；其餘讀數相對 $10^{-3}$；表格數字逐字相同。

**C1**（seed 7，`c1_out.txt`）：$\bar a=-0.2678$，$v=0.08509$，$s_a=0.2917$，資料範圍 $[-0.829,0.254]$，$c=(4.32324,0.33739,0.07384)$，$\overline{|\hat Q|}=4.39708$，$q(\bar a)=4.2964$。

| α | λ | π\* | q(π\*) | Q̂(π\*) | π_int（閉式） | J'' |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | −0.268 | 4.2962 | 4.3231 | −0.2678 | −2 |
| 1 | 0.2274 | −0.104 | 4.5984 | 4.5360 | | |
| 2 | 0.4548 | +0.167 | 4.9940 | 4.9903 | | |
| 2.5 | 0.5686 | +0.381 | 4.8336 | 5.4391 | | |
| 3 | 0.6823 | +0.699 | 4.2507 | 6.2528 | 0.6994 | −0.816 |
| 3.2 | 0.7278 | +0.874 | | | 0.8745 | −0.737 |
| 3.3 | 0.7505 | +0.977 | | | 0.9768 | −0.697 |
| 3.4 | 0.7732 | +1.000（截斷） | 4.0286 | 7.1846 | 1.0915 | −0.658 |
| 5.1 | 1.1599 | +1.000 | 4.0286 | 7.1846 | 無（J 變凸） | +0.013 |

$\pi_{\rm int}=1$ 於 α = 3.3212；$J''=0$ 於 α = 5.0668。（α = 0 時 π\* 為最接近 ā 的網格點，故 q 為 4.2962 而非 4.2964。）seed 1：α = 2.5 → π\* = +0.264、q = 4.9775。推定常數下的 200 seed 統計（只供建置者核對，不上頁面）：只最大化擬合值落在資料範圍外 192 次；α = 2.5 勝過純 BC 177 次。

**C2**（seed 7，α = 2.5，`c2_out.txt`）

| 模式 | c | λ | π\* | q(π\*) | g（a = ā） |
| --- | --- | --- | --- | --- | --- |
| 正規化 | 0.01 / 1 / 100 | 56.856 / 0.56856 / 0.0056856 | +0.381 | 4.8336 | 0.6576 |
| 固定 λ | 0.01 | 2.5 | −0.253 | 4.3198 | 0.02892 |
| 固定 λ | 0.1 | 2.5 | −0.083 | 4.6409 | 0.2892 |
| 固定 λ | 1 / 10 / 100 | 2.5 | +1.000 | 4.0286 | 2.892 / 28.92 / 289.2 |

固定 λ：π\* 首次達到 1 於 $\log_{10}c=-0.52$（0.05 步長掃描為 −0.5）；連續性檢查（401 點）三種情形（固定 λ、b = 0；正規化 b = 4；正規化 b = −2）皆無不連續，固定 λ 在 $[-0.71,-0.52]$ 每 0.01 上升 0.02–0.08。滑桿範圍內 $\min_b m(b)=0.2810$（b = −4.3），$m<0.5$ 於 b ∈ [−4.8, −3.9]；b = −4.4 時 λ = 8.693、π\* = +1.000。正規化、$c=1$ 的平移：$b=-4$ → λ = 6.162、π\* = +1.000；$b=-2$ → 1.043、+1.000；$b=+4$ → 0.2977、−0.036（q = 4.7339）；$b=+8$ → 0.2017、−0.126（q = 4.5541）。

**C4 與內文算術**（`c4_out.txt`）：Table 2 各欄加總 BC 595.3、BRAC-p 284.1、CQL 764.3、TD3+BC 979.3（皆等於論文 Total），Fisher-BRC 975.2（論文 974.6）；各列 ± 之和 BC 91.5、CQL 61.5、Fisher-BRC 108.3、TD3+BC 33.4（皆等於論文 Total ±）；Table 9 ± 之和 57.4、45.6、16.6（皆等於論文）。TD3+BC 低於 BC 的任務：無。論文 Total 差 4.7（列加總差 4.1，只在 R3 提及）。Table 3 底層時間 137、60、55、39 分；總計 / TD3+BC：6.44、3.38、3.28。Fig. 8：DT / TD3+BC = 25.15，DT(100k) / TD3+BC = 2.51。Table 9 Total(DT)：672.6、677.4；TD3+BC 15 任務 1013.2。

---

## 6. 實作計畫

### 6.1 檔案與共用外殼

- 單一檔案 `tut/570/build/index.html`；CSS、JS、時間線資料全部內嵌，不 `fetch` 本機檔案。上線時複製為 `tutorials/td3-bc-minimalist-offline-rl.html`。
- **直接沿用** `tut/406/build/index.html`（建置前確認存在；其外殼即 `tutorials/prediction-targets-k-diff.html` 的外殼）：CSS tokens（含深色模式三段覆寫、`pm-theme`）、字型載入、appbar、`#pm-bar` 導覽、章節目錄、`.plate` 圖版、callout、`<details>` 檢查題、「本頁推導」「推定」pill、「作者認為」引述樣式、「進階」框、表格橫捲容器、KaTeX 0.16.9 設定與 `renderMath()`、`?test` selfTest 框架（`ok()` 計數、`#selftest` 輸出、`scrollWidth <= innerWidth` 檢查）、時間線卡片列、`radiogroup()`、`schedule()`、`debounce()`、SVG 輔助（`el`、`scaleLin`、`linePath`）。
- 以下只列新增或變更的部分。

### 6.2 新增 tokens、標籤與連結

- 圖表色：`--accent` 用於 TD3+BC／正規化；新增 `--c-q`（light #3F6E8C／dark #8DB4D3，沿用 k-diff 頁 `--c-par` 色值）用於擬合的 Q。建置時確認在 `--surface` 上圖形 ≥ 3:1、文字 ≥ 4.5:1。
- 新增「背景（論文外）」pill、「理想化」pill（與「本頁推導」同樣式）。
- 偽碼兩欄元件（S8）：`<ol class="code-grid">`，每行 `<li><code>…</code><span class="tag new|old">…</span><p>…</p></li>`。
- 站內連結：`../index.html#paper-570`；arXiv `https://arxiv.org/abs/2106.06860`；程式碼 `https://github.com/sfujim/TD3_BC`；筆記 `<slug>.html`。

### 6.3 程式結構（單一 IIFE）

- 玩具：`rng(seed)`、`gauss(r)`、`TOY={MU_B:-0.3,SD_B:0.3,N:40,SD_R:0.05,RIDGE:1e-3,A_STAR:0.2,W_Q:0.3,B0:4,G:2001}`、`qTrue(a)`、`toyData(seed)`、`fitQ(A,R)` → `{c,abar,v,sa,Qh(a)}` 或 `null`（$s_a<10^{-6}$）、`solve3(M,b)`、`argmaxJ(fit,lam,scale,off)`、`piInterior(fit,lam)`、`policyAt(fit,A,alpha,{scale,off,norm})`。
- C1：`initC1(container, {full})`，兩個實例共用 `toyState={seed,data,fit,sweep}`；`sweep` 快取 101 個 α 的結果，換 seed 時重算。
- C2：`initC2()`，`state={logc,b,norm}`；主圖每次重算 401 個 $c$。
- C4：`initC4()`，常數如下（建置時逐字貼上；與 `c4.py` 相同），`renderT2()`、`renderTime()`、`renderT9()`。

```json
{"T2":{"algs":["BC","BRAC-p","AWAC","CQL","Fisher-BRC","TD3+BC"],
 "rows":[
  ["random","HalfCheetah",[2.0,0.1],[23.5],[2.2],[21.7,0.9],[32.2,2.2],[10.2,1.3]],
  ["random","Hopper",[9.5,0.1],[11.1],[9.6],[10.7,0.1],[11.4,0.2],[11.0,0.1]],
  ["random","Walker2d",[1.2,0.2],[0.8],[5.1],[2.7,1.2],[0.6,0.6],[1.4,1.6]],
  ["medium","HalfCheetah",[36.6,0.6],[44.0],[37.4],[37.2,0.3],[41.3,0.5],[42.8,0.3]],
  ["medium","Hopper",[30.0,0.5],[31.2],[72.0],[44.2,10.8],[99.4,0.4],[99.5,1.0]],
  ["medium","Walker2d",[11.4,6.3],[72.7],[30.1],[57.5,8.3],[79.5,1.0],[79.7,1.8]],
  ["medium-replay","HalfCheetah",[34.7,1.8],[45.6],null,[41.9,1.1],[43.3,0.9],[43.3,0.5]],
  ["medium-replay","Hopper",[19.7,5.9],[0.7],null,[28.6,0.9],[35.6,2.5],[31.4,3.0]],
  ["medium-replay","Walker2d",[8.3,1.5],[-0.3],null,[15.8,2.6],[42.6,7.0],[25.2,5.1]],
  ["medium-expert","HalfCheetah",[67.6,13.2],[43.8],[36.8],[27.1,3.9],[96.1,9.5],[97.9,4.4]],
  ["medium-expert","Hopper",[89.6,27.6],[1.1],[80.9],[111.4,1.2],[90.6,43.3],[112.2,0.2]],
  ["medium-expert","Walker2d",[12.0,5.8],[-0.3],[42.7],[68.1,13.1],[103.6,4.6],[101.1,9.3]],
  ["expert","HalfCheetah",[105.2,1.7],[3.8],[78.5],[82.4,7.4],[106.8,3.0],[105.7,1.9]],
  ["expert","Hopper",[111.5,1.3],[6.6],[85.2],[111.2,2.1],[112.3,0.2],[112.2,0.2]],
  ["expert","Walker2d",[56.0,24.9],[-0.2],[57.0],[103.8,7.6],[79.9,32.4],[105.7,2.7]]],
 "total":[[595.3,91.5],[284.1],null,[764.3,61.5],[974.6,108.3],[979.3,33.4]]},
 "T3":[["CQL（GitHub）",25,89,251],["Fisher-BRC（GitHub，TensorFlow）",39,33,132],["Fisher-BRC（PyTorch 重寫）",15,58,128],["TD3+BC",0,0,39]],
 "F8":[["DT",981],["DT（100k 步）",98],["TD3+BC",39]],
 "T9":{"rows":[["medium","HalfCheetah",[42.6,0.1],[48.3,0.3]],["medium","Hopper",[67.6,1],[59.3,4.2]],["medium","Walker2d",[74.0,1.4],[83.7,2.1]],
  ["medium-replay","HalfCheetah",[36.6,0.8],[44.6,0.5]],["medium-replay","Hopper",[82.7,7],[60.9,18.8]],["medium-replay","Walker2d",[66.6,3],[81.8,5.5]],
  ["medium-expert","HalfCheetah",[86.8,1.3],[90.7,4.3]],["medium-expert","Hopper",[107.6,1.8],[98.0,9.4]],["medium-expert","Walker2d",[108.1,0.2],[110.1,0.5]]],
  "totalDT":[[672.6,16.6],[677.4,45.6]],"totalAll":[1013.2,57.4]}}
```

（T3 欄位：名稱、實作調整分、演算法調整分、總計分；TD3+BC 的 < 1s、< 5s 以 0 分繪製、標籤照錄原文。）

- `selfTest()` 新增斷言（容許誤差見第 5 節）：
  - C1 seed 7：$\bar a=-0.2678$、$\overline{|\hat Q|}=4.39708$；α = 2.5 → π\* = 0.381、q = 4.8336；α = 3.3 → π\* = 0.977 且 `piInterior` = 0.9768；∞ → π = 1.000、Q̂ = 7.1846。
  - C1 連續性：α 從 0 到 3.3 每 0.1 的 π\* 差 ≤ 0.13（防止誤加跳躍邏輯）。
  - C2：正規化 $c=0.01$ 與 $c=100$ 的 π\* 相同（0.381）；固定 λ、$c=0.01$ → −0.253；正規化 $b=4$ → −0.036。
  - C4：`T2` 欄加總 979.3、595.3、764.3；TD3+BC ± 之和 33.4；Table 9 TD3+BC 加總 1013.2。

### 6.4 研究筆記與時間線

- 來源 `tut/570/notes/timeline.json`（14 筆，欄位 `year`、`title`、`authors_short`、`arxiv`、`role`、`source`），原樣貼進 `<script type="application/json" id="timeline-data">`。排序用 `Number(year)`。
- `TL_GROUP`（以 arXiv ID 為 key；不在表中歸「其他」）：

  | 類別 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | Actor-critic | 1509.02971、1802.09477 | `ddpg-td3-actor-critic.html` |
  | RL + BC | 1709.10089 | `rl-plus-behavior-cloning-prior-work.html` |
  | 實作細節 | 1709.06560、2005.12729 | `rl-implementation-details-reproducibility.html` |
  | 策略約束 | 1812.02900、1906.00949、1911.11361 | `batch-constrained-policy-constraint-methods.html` |
  | 加權 BC | 1910.00177、2006.15134 | `advantage-weighted-bc-methods.html` |
  | 基準與比較方法 | 2004.07219、2006.04779、2103.08050 | `td3bc-algorithm-and-d4rl.html` |
  | 本文 | 2106.06860 | `td3bc-algorithm-and-d4rl.html` |

- S13 延伸閱讀（依總覽閱讀順序）：`td3bc-lineage-overview`、`rl-mdp-q-learning-basics`、`offline-rl-distribution-shift`、`ddpg-td3-actor-critic`、`batch-constrained-policy-constraint-methods`、`advantage-weighted-bc-methods`、`rl-plus-behavior-cloning-prior-work`、`rl-implementation-details-reproducibility`、`td3bc-algorithm-and-d4rl`。每項附 frontmatter 的 `title`；第 2、3 篇來自 No. 568（`tut/568/notes/`）。

### 6.5 文案規則（新增部分）

- 用詞：「離線強化學習（offline RL）」「外插誤差（extrapolation error）」「行為複製（behavior cloning, BC）」「行為策略（behavior policy）」「分布外（OOD）」「正規化分數」「常見錯誤」。第一次出現附英文。
- 字面敘述：寫「最大化估計值的更新會選到被高估的動作」「使策略輸出接近資料中的動作」，不寫「策略偏好」「拉回」「學會」等擬人說法。
- 四種陳述分開標示：論文結果（附出處）、「作者認為」、〔本頁推導〕、背景（論文外）；玩具常數加「推定」，C2 加「理想化」。
- 第 1 節以外不得新增論文數字；頁面數字須與第 1、5 節一致。

### 6.6 建置後自我檢查

依 406 頁流程：390／768／1024／1440 截圖＋深色、鍵盤 focus、reduced-motion（截圖用本機 KaTeX 副本）；逐項評分並修正；`?test` 全數通過；逐條核對 12 個學習目標的答案出現在指定章節正文；確認 SVG 內沒有 U+0302／U+0304 組合字元（`grep -P '\x{0302}|\x{0304}'` 只能出現在 KaTeX 原始碼中）。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 玩具的校準 | C1–C2 的常數（$B_0=4$、行為分布、二次 critic）是為了讓 α = 2.5 落在好的區域而選的；換常數時最佳 α 會移動。玩具沒有自舉，高估只來自迴歸外推。 | 一律標「推定」；正文寫明「玩具展示權衡的形狀，不重現論文的 α 選擇」；不引用 seed 統計；論文的 α 證據只引用 App. B 與 Fig. 6。 |
| R2 | Eq. 5 與官方程式碼不一致 | Eq. 5 用 $Q(s_i,a_i)$；官方程式碼用 $Q(s,\pi(s))$ 的 Q1（已讀 `TD3_BC.py`）。 | 頁面與偽碼以 Eq. 5 為主；偽碼註解與常見錯誤第 3 條並列兩者並標「背景（論文外）」。 |
| R3 | Table 2 的 Fisher-BRC 總分 | 各列加總 975.2，論文 Total 974.6。已用 `pdftotext -layout` 核對 PDF 第 7 頁，15 個值與 paper.txt 相同，差異在論文本身。 | 頁面照錄論文 Total（差距一律以論文 Total 計，4.7），C4 附一行〔本頁推導〕註記。 |
| R4 | 「最多為 4」的維度 | 論文的 $(\pi(s)-a)^2$ 未說明是平方和或平均。 | 正文說明兩種情形並引用官方程式碼的 `mse_loss`（背景（論文外））。 |
| R5 | Fig. 1–3、5–7 沒有數字 | 只有長條圖；「百分比差」的比較基準未寫明。 | 只做文字描述，不重繪；比較基準標「推定」。 |
| R6 | 正規化機制不是論文主張 | 論文沒有說明 Eq. 4 的機制；曲率解釋是一般最佳化知識，且 Adam 會抵消部分尺度差異。 | C3 已刪除；S7「進階」框以文字說明並標〔本頁推導〕與 Adam 限制。 |
| R7 | TD3 細節的出處 | TD3 的三個設計不在本文，只在 Table 4 列出超參數。 | 標「背景（論文外）」並連到 `ddpg-td3-actor-critic.html`；偽碼標明為本頁依 Eq. 3–5 寫成。 |
| R8 | 跨教學的筆記（已關閉） | `rl-mdp-q-learning-basics`、`offline-rl-distribution-shift` 位於 `tut/568/notes/`。 | 協調者確認 No. 568 會先於 No. 570 上線並發布這兩篇；保留連結。 |
| R9 | 相對連結 | 建置位置與上線位置不同，預覽時站內連結失效。 | 接受；預覽只檢查外部連結與頁內錨點。 |
| R10 | 初學者門檻 | 讀者不熟 actor-critic 與 D4RL。 | S2 先講 critic 再講 actor；S9 單獨定義資料集與比較方法；S4 術語附白話。 |
| R11 | 大 α 時策略停在 1 | 讀者可能不理解為什麼 α ≥ 3.4 時讀數不再變。 | 4.0 的〔本頁推導〕：內部最佳解連續越過動作上界後被截斷；$J$ 在 α < 5.07 時始終是凹函數。副圖標出「到達動作上界」的 α。 |

---

## 8. 修訂紀錄（round 1）

依 `reviews/round1.md` 的 26 項處理。主要變更：刪除 C3（GD 與論文的 Adam 不符、無學習目標、主執行緒成本）與 `c3.py`；C1／C2 的「跳躍／目標變凸」改為「內部最佳解 + 截斷」，`toy.py` 加 `interior()`、`c2.py` 加連續性檢查並可用 `python3 -I` 執行；尺度不變改為理想化敘述；Total 的 ± 改為「各列 ± 之和」；S9 拆為實驗設定（S9）與結果（S10），其後章節順延到 S13；Hero 不再使用未定義術語，Eq. 1 → Eq. 3 差異方塊移到 S5；SVG 內不放組合字元；C4 資料以 JSON 附在 6.3。未採納：第 11 項（「historically known as batch RL」是 Sec. 1 原文，不是論文外資訊，改為附出處）。

## 9. 修訂紀錄（readiness）

NO-GO 三項：α 的首次定義改為 S5（S5 一句定義，S6 推導）；R8 關閉，保留 No. 568 的兩篇筆記連結；C2 警示改為 $m(b)<0.5$，以 `c2.py` 算出最小值 0.2810。非阻斷：步驟 4 改為只引 Eq. 3 並註明 Eq. 2 相同；C1 觀察改為 α ≈ 3.32；Hero 註明 2h 8m 為 PyTorch 版；已確認 `tut/406/build/index.html` 存在。

## 10. 建置紀錄（build round）

- 產出 `build/index.html`（約 101 KB），由 `build/parts/assemble.py` 從 `tut/406/build/index.html` 的外殼（CSS tokens、`#pm-bar`、章節目錄、plate、KaTeX、時間線、`selfTest` 框架）與 `parts/body.html`、`parts/widgets.js`、`parts/extra.css` 組成；`parts/preview.sh` 產生本機 KaTeX 預覽、`parts/shoot.sh` 截取單一圖版（預覽檔用完即刪）。
- `?test`：63 passed, 0 failed（1440 與 390 寬皆同；含 390 無水平捲動）；無 console 錯誤；KaTeX 正常渲染。
- 視覺一輪：C1 讀數表由側欄移到圖下（側欄 260px 放不下三欄）；C1 副圖最小高度 170px、兩條參考線標籤上下錯開、「到達動作上界」標籤移到上緣；C4 橫軸擴到 140（Fisher-BRC Hopper-medium-expert 的誤差鬚到 133.9）；C4 圖例改為行內 SVG（移除 repeating-gradient）。
- `npx impeccable detect`：0 anti-patterns（13 則 numbered-section-label 提示，允許）。
- LO 自查：`accept/page.txt` 中 12 個學習目標的關鍵詞與數字皆在正文；LO12 補上「策略熵」一詞。
