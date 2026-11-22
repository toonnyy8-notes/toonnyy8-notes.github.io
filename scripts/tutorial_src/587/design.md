# 設計文件：〈Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization〉互動教學

- 論文：Zhicheng Cai、Xinyuan Guo、Hanlin Wu、Wei-Ying Ma、Ya-Qin Zhang、Hao Zhou（清華 AIR、SIA-Lab of Tsinghua AIR and ByteDance Seed），arXiv 2607.10169v2，ICML 2026（PMLR 306）。資料庫編號 No. 587。
- 原文：`tut/587/src/paper.txt`（PDF 同目錄）。「Eq. n」「Sec. n」「Table n」「Fig. n」「Prop. n」皆指論文原文編號。論文**沒有附錄**，也沒有演算法框；第 2.2 節的 PPO 目標式沒有編號（本文記為「PPO 式」）。
- 產出：`tut/587/build/index.html`（單檔）。上線位置 `tutorials/ripo-riemannian-isometric-clip.html`（slug `ripo-riemannian-isometric-clip`），站內連結以上線位置為準。
- 驗算腳本（`tut/587/verify/`，皆已執行，輸出在同名 `.out`）：`check_w1.py`（W1）、`check_w2.py`（W2）、`sim.js`＋`check_w3.py`（W3；JS 為頁面要貼上的同一份程式，Python 為獨立移植，兩者結果逐位一致）、`check_paper.py`（論文數字的內部一致性）。第 5 節的數字全部由這些腳本產生。
- 研究筆記：`tut/587/notes/`（hub `ripo-lineage-hub` + 6 篇 topic + `timeline.json`，13 筆）。既有共用筆記一律連結、不重寫（見 6.6）。
- 標記慣例：〔本頁推導〕＝論文沒有寫、由本頁推得的結論；〔教學示意〕＝本頁自行設計的玩具模型（頁面圖版底部寫「此模型為本頁設計的示意，不是論文的實驗或結果」）。本頁**沒有**「推定」值：所有玩具參數都是自選，不對應論文數字。頁面上兩種標籤都用 pill 顯示。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

PPO-Clip 把重要性比值 $r=\pi_\theta/\pi_{\theta_{old}}$ 限制在固定區間 $(1-\epsilon,1+\epsilon)$，等於用 $(r-1)^2$（Eq. 3）衡量新舊策略的差異。作者指出這是歐氏度量；依 TRPO，策略差異應以 KL 散度衡量，而 KL 的二次近似是 $\tfrac12\sum_a\pi_{\theta_{old}}(a|s)(r_{s,a}-1)^2$（Eq. 7），權重是舊機率本身。作者只取被取樣 token 的那一項 $\tfrac12\pi_{old}(r-1)^2$（Eq. 9）來比較：$|r-1|=0.2$ 時，$\pi_{\theta_{old}}=0.8$ 的值是 0.016，$\pi_{\theta_{old}}=0.01$ 只有 0.0002（Sec. 3.2）。作者主張（Prop. 3.1，以一個數字例子與實驗支持，沒有證明）這使稀有但有價值的 token 更新不足，策略過早集中，造成探索崩潰（exploration collapse）。解法 Riemannian Isometric Clip（RIC）要求每個 token 的這一項有相同的上限，得到隨舊機率變化的 clip 範圍 $\epsilon_{s,a}=\sqrt{\delta/\pi_{\theta_{old}}(a|s)}$（Eq. 11）：低機率 token 的範圍變大；高機率 token 的範圍是否比 PPO 小取決於 δ（Sec. 4.1 的 δ = 0.02 例子中較小，實驗預設 δ = 0.05 時處處比 ε = 0.2 大，見步驟 6）。作者再從 importance sampling 的變異論證：clip 邊界上單一樣本的變異貢獻在低機率端約為 $O(\delta)$（Eq. 14，「同質變異」）。把 RIC 放進 GRPO 式目標並用 token 層級 loss，就是 RIPO（Eq. 15）。在 Qwen3-1.7B/4B/8B-Base 與 Llama3.2-3B-Instruct、七個數學競賽基準上，RIPO 的平均分數都最高，Qwen3-8B-Base 平均 38.5，GRPO 28.5（Table 1）。

### 1.2 研究動機（Sec. 1、3.1）

1. LLM RL 的主流演算法（GRPO 等）沿用 PPO-Clip。DAPO（Yu et al., 2025）觀察到 PPO-Clip 讓已偏好的「exploitation token」更容易再提高機率，低機率的「exploration token」很難提高，最後探索崩潰。
2. 長程推理的狀態—動作空間很大，需要大量探索才能找到可行解；探索崩潰限制 RL 的擴展。
3. 既有修正（DAPO Clip-Higher、DCPO 動態閾值、GSPO/GMPO 序列層級比值、CISPO/GPPO/SAPO 保留被 clip token 的梯度）作者稱為經驗性的「症狀修正」，沒有指出 PPO-Clip 失效的根本原因（Sec. 1、2.3）。

### 1.3 貢獻（Sec. 1 作者自述三點）

1. 指出 PPO-Clip 的歐氏 clipping 與策略的 Riemannian 流形幾何不一致，造成病態更新與探索崩潰（Sec. 3，Prop. 3.1）。
2. 提出 RIPO：保證流形上等距的策略更新，並說明它帶來較好的偏差—變異權衡（Sec. 4，Prop. 4.1）。
3. 實驗：4 個模型、7 個數學基準、5 個基準演算法（Sec. 5 與 5.1 稱 six，5.1 又稱「all these seven RL algorithms」，但只列出 GRPO、DAPO、GSPO、GMPO、DCPO 五個；頁面寫「5 個（論文稱 six，只列出五個）」）；AIME24 相對 GRPO 最多約 60%（Qwen3-1.7B-Base 18.3 vs 11.3，`check_paper.py` 算得 +61.9%）。另有 PPO 目標、程式與搜尋任務的推廣（Sec. 5）。

### 1.4 符號表（頁面 S1 末尾以可收合表格提供）

| 符號 | 意義 |
| --- | --- |
| $s$、$a$ | 狀態與動作。LLM 中 $s=(q,o_{<t})$ 是題目加已產生的前綴，$a=o_t$ 是下一個 token |
| $\pi_\theta$、$\pi_{\theta_{old}}$ | 目前要更新的策略、產生 rollout 的行為策略（behavior policy）；簡寫 $\pi$、$\pi_{old}$ |
| $r_{s,a}(\theta)=\pi_\theta(a|s)/\pi_{\theta_{old}}(a|s)$ | 重要性比值（importance ratio）；GRPO 中逐 token 記為 $r_{i,t}$ |
| $\hat A$、$\hat A_{i,t}$ | 優勢估計；PPO 用 GAE，GRPO 用群組標準化 $(R_i-\mathrm{mean})/\mathrm{std}$ |
| $\epsilon$；$\epsilon_{low},\epsilon_{high}$ | PPO 的固定 clip 範圍；DAPO 的上下分離範圍 |
| $\delta$ | TRPO 的 trust region 半徑（Eq. 1）；在 RIC 中是每個 token 允許的幾何距離（Eq. 9、11，見 1.5 步驟 6 關於係數 2 的說明） |
| $F(\theta)$ | Fisher 資訊矩陣 $\mathbb E_{a\sim\pi_{\theta_{old}}(\cdot|s)}[\nabla\log\pi\,\nabla\log\pi^\top]$（在 $\theta_{old}$ 取值；Eq. 1 的約束再對 $s\sim\rho_{\theta_{old}}$ 平均） |
| $d_{clip}$、$d_{geom}$ | PPO-Clip 隱含的歐氏距離（Eq. 3）、KL 誘導的幾何距離（Eq. 8–9） |
| $\epsilon_{s,a}(\pi_{old})$、$\epsilon_{i,t}(\pi_{old})$ | RIC 的動態 clip 範圍 $\sqrt{\delta/\pi_{old}}$（Eq. 11、15） |
| $x$、$A(x)$、$v(x)$ | Sec. 4.2 的樣本（狀態—動作對）、其優勢、變異貢獻 $\pi_{old}(x)r(x)^2$ |
| $q$、$o_i$、$G$、$R_i$ | 題目、第 $i$ 條回答、群組大小、獎勵（答對 1、答錯 0） |

### 1.5 從作者的角度逐步拆解

每一步：作者想回答什麼、假設、得到什麼、下一步為什麼需要它。

#### 步驟 1：TRPO 的 KL trust region（Sec. 2.1，Eq. 1）

$$\max_\theta\ \mathbb E_{s\sim\rho_{\theta_{old}},a\sim\pi_{\theta_{old}}}\Big[\tfrac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}\hat A(s,a)\Big]\quad\text{s.t.}\quad\mathbb E_{s}\big[D_{KL}(\pi_{\theta_{old}}(\cdot|s)\,\|\,\pi_\theta(\cdot|s))\big]\le\delta\tag{Eq. 1}$$

- TRPO 證明在 KL 約束下最佳化 surrogate 可保證單調改進，但約束最佳化計算昂貴，難以擴展。
- 作者取用的是「策略差異應以 KL 衡量」這個前提。背景（單調改進下界）連結既有筆記 `trpo-monotonic-improvement-tv-bound.html`，頁面不重講。

#### 步驟 2：PPO-Clip 與 GRPO（Sec. 2.2–2.3，PPO 式、Eq. 2）

$$J_{PPO}(\theta)=\mathbb E\big[\min\big(r(\theta)\hat A,\ \mathrm{clip}(r(\theta),1-\epsilon,1+\epsilon)\hat A\big)\big]\tag{PPO 式}$$

$$J_{GRPO}(\theta)=\mathbb E_{q\sim\mathcal D,\{o_i\}\sim\pi_{\theta_{old}}}\Big[\frac1G\sum_{i=1}^G\frac1{|o_i|}\sum_{t=1}^{|o_i|}\min\big(r_{i,t}\hat A_{i,t},\ \mathrm{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat A_{i,t}\big)\Big]\tag{Eq. 2}$$

- clip 的作用（頁面 S1 必須寫出）：$\hat A>0$ 時若 $r>1+\epsilon$，min 取到常數 $(1+\epsilon)\hat A$，該 token 的梯度為 0；$\hat A<0$ 時若 $r<1-\epsilon$ 亦同。clip 不是把 $\pi_\theta$ 硬性截斷：參數共享與多次 mini-batch 更新仍可能讓比值超出範圍。$\hat A$ 的正負決定該 token 的機率被推高或壓低。
- PPO 以一階 clipped surrogate 啟發式地近似 TRPO 的二階約束；GRPO 去掉價值模型，優勢改為群組標準化 $\hat A_{i,t}=(R_i-\mathrm{mean}(\{R_i\}))/\mathrm{std}(\{R_i\})$。
- DAPO 把範圍改成 $(1-\epsilon_{low},1+\epsilon_{high})$、$\epsilon_{high}>\epsilon_{low}$（Clip-Higher）。
- 頁面 S1 只講 PPO 式；Eq. 2（GRPO）與 TRPO Eq. 1 移到 S7 開頭（RIPO 以 GRPO 為基底時才需要），詳細背景連結 `ppo-grpo-dapo-llm-rl-objectives.html`、`policy-gradient-baselines-score-function.html`。

#### 步驟 3：探索崩潰的數字例子（Sec. 3.1）

- $\epsilon=0.2$：$\pi_{old}=0.8$ 的動作「maximum possible updated probability」為 $0.96$（增加 0.16）；$\pi_{old}=0.01$ 為 $0.012$（增加 0.002）。頁面照引這個說法，並註明：這是對 $\hat A>0$ 的 token 而言；比值超過 $1+\epsilon$ 後該 token 不再提供梯度（步驟 2），所以實際上是「繼續推高這個 token 的梯度在此處停止」，不是硬性上限。
- DAPO 把 $\epsilon_{high}$ 由 0.2 提到 0.28：0.01 的上限變成 0.0128（只多 0.0008）；0.8 的上限由 0.96 變成 1（$0.8\times1.28=1.024$，被機率上限 1 截斷）。作者認為這反而加劇行為多樣性的下降。
- 作者的結論：PPO 式 clip 不分動作是否已被大量利用，一律以相同比值截斷。下一步要說明這來自幾何不一致。

#### 步驟 4：KL 的二次近似把權重放在 $\pi_{old}$（Sec. 3.2，Eq. 4–8）

頁面 S3 的教學順序（LO3 只要求認得式子與名稱，屬「知道」層級；LO4 是重點）：
- 先給一維直覺〔本頁說明〕：KL 在新舊策略相同時為 0 且為最小值，所以在附近的值約為「½ × 曲率 × 偏移²」。多維時曲率換成一個矩陣，這個矩陣就是 Fisher 矩陣；「度量」一詞在本頁只指「用來把參數偏移換算成距離的這個矩陣」。
- 再寫 Eq. 4；Eq. 5–6 的代數放在可收合推導；Eq. 7 寫在正文。
- 正文必須寫出 Eq. 6 的假設：新機率約等於舊機率加上一階變化，$\Delta\pi\approx\nabla\pi^\top\Delta\theta$，只在更新很小時成立。由此 Eq. 7 的被加項 $(\Delta\pi)^2/\pi_{old}=\pi_{old}(r-1)^2$，權重是 $\pi_{old}$（LO4 的答案由這兩句可直接推得）。

1. KL 在 $\theta_{old}$ 二階展開（零階與一階項為 0）：$D_{KL}(\pi_{\theta_{old}}\|\pi_\theta)\approx\tfrac12\Delta\theta^\top F(\theta)|_{\theta_{old}}\Delta\theta$，$\Delta\theta=\theta-\theta_{old}$（Eq. 4）。$F$ 的期望對 $a\sim\pi_{\theta_{old}}(\cdot|s)$ 取。Fisher 矩陣定義一個 Riemannian 度量。原文 Eq. 4 第二行寫成 $\Phi(\theta)+\dots$，應為 $\Phi(\theta_{old})+\dots$（筆誤，頁面直接寫正確形式並在註腳說明）。
2. 把 $F$ 寫開：$\Delta\theta^\top F\Delta\theta=\sum_a\frac{1}{\pi_{\theta_{old}}(a|s)}\big(\nabla_\theta\pi_\theta(a|s)^\top|_{\theta_{old}}\Delta\theta\big)^2$（Eq. 5，用 $\nabla\log\pi=\nabla\pi/\pi$）。
3. 一階展開 $\pi_\theta\approx\pi_{\theta_{old}}+\nabla\pi^\top\Delta\theta$（Eq. 6），代入得

$$D_{KL}(\pi_{\theta_{old}}\|\pi_\theta)\approx\frac12\sum_a\frac{(\pi_\theta(a|s)-\pi_{\theta_{old}}(a|s))^2}{\pi_{\theta_{old}}(a|s)}=\frac12\sum_a\pi_{\theta_{old}}(a|s)\,(r_{s,a}(\theta)-1)^2\tag{Eq. 7}$$

$$d_{geom}(\pi_{\theta_{old}},\pi_\theta)\propto\pi_{\theta_{old}}\cdot(r(\theta)-1)^2\tag{Eq. 8}$$

- 意義：高機率區的距離被放大、低機率區被縮小，幾何不均勻。
- 〔本頁推導〕Eq. 7 中間那一式正是 $\tfrac12\chi^2(\pi_\theta\|\pi_{\theta_{old}})$（Pearson χ²）。後續的 GPPO 論文（arXiv 2610.13015）把 RIPO 歸為「χ² 散度 × Atomic 估計器」，見既有筆記 `gppo-design-space-unification.html`、`f-divergence-geometry-proximal-regions.html`（頁面放在可收合段落，以一句話連結，不展開）。
- 〔本頁推導〕Eq. 7 是二階近似，只在 $r$ 接近 1 時準確。$\pi_{old}=0.01$、$r=2.414$ 時（RIC 邊界，δ = 0.02）：單 token 項（Eq. 9）恰為 0.01，二元二階量測 0.0101，二元精確 KL 0.00543（`check_w1.out`；二元量測的定義見步驟 5）。頁面在 W1 進階讀數中列出，提醒讀者這是近似。

#### 步驟 5：PPO-Clip 的歐氏假設與 trust region 觀點（Sec. 3.2，Eq. 3、Prop. 3.1）

$$d_{clip}(\pi_{\theta_{old}},\pi_\theta)=\Big(\frac{\pi_\theta}{\pi_{\theta_{old}}}-1\Big)^2=(r(\theta)-1)^2\tag{Eq. 3}$$

- PPO-Clip 等價於約束 $|r-1|<\epsilon$，把相同比值偏差視為相同的策略變化，隱含假設策略變化在整個流形上均勻。
- Trust region 例子（Sec. 3.2 以單 token 項 $\tfrac12\pi_{old}(r-1)^2$ 計，不是完整 KL）：$|r-1|=0.2$ 時，$\pi_{old}=0.8$ 為 $0.5\times0.8\times0.2^2=0.016$，$\pi_{old}=0.01$ 為 $0.5\times0.01\times0.2^2=0.0002$，差 80 倍。作者的說法是後者遠小於 trust region 允許的距離 δ。
- 頁面 S4 正文（不是只在 callout）在第一次出現 0.016／0.0002 時加一句：「Eq. 7 對所有動作求和，包含其餘 token 因總和為 1 而跟著改變的部分；Eq. 9 與這個例子只保留被取樣 token 的一項。」
- Prop. 3.1：PPO-Clip 錯誤地使用歐氏度量，導致低機率區過度保守、高機率區過度激進，最終探索崩潰。**論文沒有證明**：「歐氏度量 ↔ 幾何距離不一致」由 Eq. 3 與 Eq. 8 的比較推得（有推導）；「因此探索崩潰」只由 Sec. 3.1 的數字例子與 Sec. 5 的熵曲線支持（觀察）。頁面以「作者主張」敘述，並把「推導出的部分」與「觀察支持的部分」分成兩句。
- 〔本頁推導〕二元量測（W1 進階讀數的定義，頁面在 S4 callout 教）：提高一個 token 的機率必然壓低其他 token（總和為 1）。把動作空間分成「該 token」與「其餘」兩類，Eq. 7 的和只剩兩項，得二階距離 $\tfrac12(\Delta\pi)^2/(\pi_{old}(1-\pi_{old}))$，比單 token 項多一個因子 $1/(1-\pi_{old})$：$\pi_{old}=0.01$ 時幾乎相同（0.010101 vs 0.010000），$\pi_{old}=0.8$ 時是 5 倍（0.05 vs 0.01）。以二元量測計，PPO 例子的兩個值是 0.080 與 0.000202（約 400 倍），二元精確 KL 為 0.176 與 0.000179（約 1000 倍）（`check_w1.out`）。也就是單 token 量測在高機率區低估分布層級的變化；DPPO（Qi et al., 2026，arXiv 2602.04879）的 Binary 估計器即加入互補質量，此描述出自 GPPO 論文，見 `divergence-estimator-topologies.html`。

#### 步驟 6：Riemannian Isometric Clip（Sec. 4.1，Eq. 9–11）

要求每個 token 的單 token 距離有相同的上限 δ（Eq. 9 是不等式：只有到達 clip 邊界的 token 等於上限，範圍內的 token 較小；且 min 只讓一側起作用，$\hat A>0$ 看上界、$\hat A<0$ 看下界）：

$$d_{geom}(\pi_{\theta_{old}}(a|s),\pi_\theta(a|s))\triangleq\tfrac12\pi_{\theta_{old}}(a|s)(r_{s,a}(\theta)-1)^2\le\delta\tag{Eq. 9}$$

$$|r_{s,a}(\theta)-1|\le\sqrt{\frac{2\delta}{\pi_{\theta_{old}}(a|s)}}\tag{Eq. 10}$$

$$|r_{s,a}(\theta)-1|\le\epsilon_{s,a}(\pi_{\theta_{old}}),\qquad\epsilon_{s,a}(\pi_{\theta_{old}})=\sqrt{\frac{\delta}{\pi_{\theta_{old}}(a|s)}}\tag{Eq. 11}$$

- Eq. 11 把係數 2 吸收進超參數 δ。因此**以 Eq. 11 的 δ 而言，每個 token 在 Eq. 9 單位下的距離上限是 δ/2**。頁面明寫這一點（論文的例子即是如此：δ = 0.02 時距離 0.01）。
- 例子（Sec. 4.1，δ = 0.02）：$\pi_{old}=0.8$ 的上限 0.92（PPO-Clip 為 0.96）；$\pi_{old}=0.01$ 的上限 0.024（PPO-Clip 為 0.012）；兩者在上邊界的單 token 距離都是 0.01（PPO-Clip 為 0.016 與 0.0002）。〔本頁推導〕精確值 $0.8(1+\sqrt{0.025})=0.9265$，論文寫 0.92（截位）；$0.01(1+\sqrt2)=0.02414$。頁面寫「0.92（精確值 0.9265）」。
- Prop. 4.1：RIC 保證每個狀態—動作更新的幾何距離相同（本頁改寫為「距離上限相同」，理由見上）；低機率動作可更新較多，高機率動作較少；同時緩解探索崩潰並平衡偏差—變異。**論文沒有證明**：「距離上限相同」由 Eq. 9–11 直接成立；「高機率動作較少」只在 $\pi_{old}>\delta/\epsilon^2$ 時相對 PPO 成立（下一點）；「緩解探索崩潰」由 Sec. 5 實驗支持；「平衡偏差—變異」是步驟 7 的定性論證。頁面依此分句。
- 〔本頁推導〕RIC 何時比 PPO 緊：$\sqrt{\delta/\pi_{old}}<\epsilon\iff\pi_{old}>\delta/\epsilon^2$。δ = 0.02、ε = 0.2 時分界在 0.5；**論文實驗的預設 δ = 0.05（Sec. 5.1）時分界為 1.25，也就是對所有 $\pi_{old}\le1$，$\epsilon_{s,a}\ge\sqrt{0.05}=0.224>0.2$**：上側永遠比 PPO 寬，只是寬的程度隨機率改變。另外機率不能超過 1，有效上限是 $\min(1+\epsilon_{s,a},1/\pi_{old})$；δ = 0.05 時 $\pi_{old}\ge0.8$ 的 token 上限直接是 1（$\pi_{old}=0.8$：RIC 與 DAPO 都到 1.0）。所以 Sec. 4.1 例子中「高機率 token 較不激進」的效果依賴 δ 的取值；頁面把這段寫在 S5b，並讓 W1 進階檢視可切換 δ = 0.02／0.05。
- 〔本頁推導〕下側 $1-\sqrt{\delta/\pi_{old}}$ 在 $\pi_{old}\le\delta$ 時 ≤ 0，比值本來就 ≥ 0，等於沒有下限；在 $\pi_{old}<4\delta$ 時低於 0.5（見步驟 9 的雙重 clipping）。

#### 步驟 7：幾何等距與同質變異（Sec. 4.2，Eq. 12–14）

$$\mathbb E_{x\sim\pi_\theta}[A(x)]=\mathbb E_{x\sim\pi_{\theta_{old}}}[r(x)A(x)],\qquad r(x)=\frac{\pi_\theta(x)}{\pi_{\theta_{old}}(x)}\tag{Eq. 12}$$

$$\mathbb V_{x\sim\pi_{\theta_{old}}}[r(x)A(x)]\approx\mathbb E_{x\sim\pi_{\theta_{old}}}[r(x)^2A(x)^2]=\sum_x\pi_{\theta_{old}}(x)\,r(x)^2A(x)^2\tag{Eq. 13}$$

- 頁面 S6 在 W2 之前必須用正文寫出的先備（2–4 句）：(a) 為何不偏：$\sum_x\pi_{old}(x)\frac{\pi_\theta(x)}{\pi_{old}(x)}A(x)=\sum_x\pi_\theta(x)A(x)$，比值剛好抵消取樣分布的差異；(b) 截斷把權重 $r$ 換成較小的 $w$，期望值少了 $\sum_x(\pi_\theta(x)-\pi_{old}(x)w(x))A(x)$，這就是偏差；取 $A\equiv1$ 時單一 token 的偏差為 $\pi_\theta-\pi_{old}w$；(c) $v$ 衡量該樣本對估計值二階動差（因而對變異）的貢獻；(d) 變異 = 二階動差 − 平均值平方，作者依 Tokdar & Kass (2010)、Murphy (2012) 認為後者通常可忽略，所以只看二階動差；(e)「同質」在此指「每個 token 的 $v$ 大小相同、不隨 $\pi_{old}$ 改變」，先用這句白話，再給術語 homoscedasticity。
- 作者引 Tokdar & Kass (2010)、Murphy (2012)：變異由二階動差主導，平方均值項通常可忽略。
- 定義樣本的變異貢獻 $v(x)=\pi_{\theta_{old}}(x)r(x)^2$。$\pi_{old}(x)$ 很小時 $v(x)=\pi_\theta(x)^2/\pi_{\theta_{old}}(x)$ 可任意大（頁面用「可任意大」，不用「爆炸」）：作者認為變異過大來自 $\pi_{old}$ 的長尾。
- PPO-Clip 截斷 $r\le1+\epsilon$：邊界附近 $v(x')=\pi_{old}(x')(1+\epsilon)^2\to0$（$\pi_{old}\to0$），變異小，但忽略被截斷樣本對目標的貢獻，偏差大。
- RIC：

$$v(x')=\pi_{\theta_{old}}(x')\Big(1+\sqrt{\tfrac{\delta}{\pi_{\theta_{old}}(x')}}\Big)^2\approx O(\delta)\tag{Eq. 14}$$

  作者稱之為與密度無關、常數階的變異；結論是變異嚴格小於標準 importance sampling、偏差遠小於 PPO-Clip。「幾何解讀」：等距更新使每個樣本對 trust region 的二階貢獻相同，進而在 importance sampling 中造成統計上的同質變異（homoscedasticity）；PPO-Clip 則是異質變異。
- 〔本頁推導〕Eq. 14 可精確寫成 $v=(\sqrt{\pi_{old}}+\sqrt\delta)^2=\pi_{old}+2\sqrt{\delta\pi_{old}}+\delta$。它在 $\pi_{old}\ll\delta$ 時趨近 δ；$\pi_{old}\gg\delta$ 時約為 $\pi_{old}$。所以「與密度無關」是低機率端的漸近性質。δ = 0.05：$\pi_{old}=10^{-5},10^{-3},0.1,0.5$ 時 $v$ = 0.0514、0.0651、0.2914、0.8662（`check_w2.out`）。
- 〔本頁推導〕$O(\delta)$ 是**單一樣本**的上界；總二階動差是所有被截斷 token 的和，約為「被截斷的低機率 token 數 × δ」，仍隨長尾 token 數增加。頁面加這一句。
- 〔本頁推導〕「變異嚴格小於 IS」只在比值確實超過 $1+\sqrt{\delta/\pi_{old}}$ 時才有差別；比值在範圍內時 RIC 與未截斷 IS 完全相同。同理，$\pi_{old}>\delta/\epsilon^2$ 的 token 上 RIC 比 PPO 緊，那裡的偏差反而比 PPO 大。
- 簡化說明：Sec. 4.2 把 clip 當成「把比值截斷到上限」（$w=\min(r,1+\epsilon)$）來分析；PPO 實際的 min-clip 是讓被截斷的項梯度為 0。W2 依論文的截斷寫法計算，頁面註明。

#### 步驟 8：RIPO 目標（Sec. 4.3，Eq. 15）

$$J_{RIPO}(\theta)=\mathbb E_{q,\{o_i\}\sim\pi_{\theta_{old}}}\Big[\frac{1}{\sum_{i=1}^G|o_i|}\sum_{i=1}^G\sum_{t=1}^{|o_i|}\min\big(r_{i,t}\hat A_{i,t},\ \mathrm{clip}(r_{i,t},1-\epsilon_{i,t}(\pi_{old}),1+\epsilon_{i,t}(\pi_{old}))\hat A_{i,t}\big)\Big]\tag{Eq. 15}$$

- 與 Eq. 2 的差別只有兩處：(a) clip 範圍改為每個 token 的 $\epsilon_{i,t}=\sqrt{\delta/\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}$；(b) 正規化由 $\frac1G\sum_i\frac1{|o_i|}$ 改為 $\frac1{\sum_i|o_i|}$（DAPO 的 token 層級 loss，平衡長短回答的梯度影響）。優勢沿用 GRPO 的群組相對估計。
- 名稱註腳：RIPO 唸作「ripple」。

#### 步驟 9：實驗設定（Sec. 5.1）

| 項目 | 值 |
| --- | --- |
| 模型 | Llama3.2-3B-Instruct、Qwen3-1.7B-Base、Qwen3-4B-Base、Qwen3-8B-Base |
| 基準演算法 | GRPO（PPO-Clip）、DAPO（Clip-Higher）、GSPO（序列層級）、GMPO（幾何平均比值）、DCPO（動態自適應），皆用各自預設超參數；Sec. 5.5 另比 GPPO（Su et al., 2025，arXiv 2508.07629）與 Clip-Cov |
| RIPO 的 δ | 0.05 |
| 雙重 clipping | GRPO、DAPO、DCPO、RIPO 都對比值使用 dual clipping，「lower and upper bounds set to 0.5 and 10」 |
| KL 懲罰 | 全部移除 |
| 優勢 | 七種方法都用群組相對優勢 |
| 訓練資料 | DAPO-Math-17k（17,917 題） |
| rollout | 每題 8 條，最大回應長度 16,384 token；每次迭代 1024 條 rollout（train batch 128 題 × 8），$\pi_\theta$ 以 mini-batch 16 更新 8 次 |
| 獎勵 | 答對 1、答錯 0 |
| 最佳化 | token-mean loss 聚合；300 步；AdamW，固定學習率 1×10⁻⁶ |
| 平台 | VeRL（HybridFlow），8×A100 |
| 評估 | AIME24、AIME25、AMC23、HMMT25、BRUMO25、CMIMC25、SMT25；重複 8 次取 Avg@8 |

#### 步驟 10：實驗結果（Sec. 5.2–5.8）

- **Table 1（Avg@8 平均）**：RIPO 在四個模型都最高。相對 GRPO：Qwen3-1.7B-Base 15.4 vs 11.2（+37.2%）、Llama3.2-3B-Instruct 8.6 vs 6.4（+34.4%）、Qwen3-4B-Base 30.1 vs 25.7（+17.1%）、Qwen3-8B-Base 38.5 vs 28.5（+35.1%）。百分比照錄論文；以表中四捨五入後的平均重算，Qwen3-1.7B-Base 為 +37.5%（W4 加註腳）。Qwen3-8B-Base 的 AIME24：RIPO 43.8、GRPO 31.7、DCPO 36.3、GMPO 41.7。注意 RIPO 不是每格都最好（例：Qwen3-8B AIME25 GMPO 30.0 > RIPO 29.2；Qwen3-1.7B AMC23 DCPO 46.9 > 46.1），頁面照錄，不寫「全面領先」。
- **Fig. 1（Qwen3-8B-Base 訓練動態，文字描述）**：(a) RIPO 在 40 步內超過 GRPO 訓練 200 步的 AIME24 表現，作者稱五倍 token 效率；(b) GRPO 熵迅速崩潰到接近 0，DAPO 熵不受控地上升，RIPO 先下降後穩定在中等範圍；(c) 其他方法梯度範數有頻繁尖峰，RIPO 幾乎無波動；(d) DCPO、GMPO 很少觸發 clip，GSPO、DAPO 被 clip 的 token 多出數個數量級，RIPO 介於中間。
- **Table 2（δ 消融，Qwen3-8B-Base，AIME24 Avg@8）**：

  | δ_low | 0.02 | 0.05 | 0.05 | 0.08 | 0.05 | 0.08 | 0.08 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | δ_high | 0.02 | 0.04 | 0.05 | 0.08 | 0.02 | 0.02 | 0.04 |
  | Avg@8 | 40.8 | 41.7 | 43.8 | 42.1 | 28.8 | 27.5 | 27.9 |

  〔本頁說明〕表現大降的三組都是 $\delta_{high}\le0.04$ 且 $\delta_{low}/\delta_{high}\ge2$；0.05/0.04 同樣不對稱但得 41.7。每個設定只有一次訓練的 AIME24 Avg@8，沒有誤差範圍。作者：0.02–0.08 對稱設定穩定；δ_high 明顯大於 δ_low 時表現大幅下降（原文如此寫）。**注意表中後三欄其實是 δ_low > δ_high**（0.05/0.02、0.08/0.02、0.08/0.04），與正文「δ_high 大於 δ_low」及「機率更容易增加」的解釋方向相反；見 R2。Fig. 2：對稱設定下獎勵穩定上升、熵逐漸下降並穩定；高度不對稱時獎勵突然下降、熵急遽上升（原文「entropy explosion」）。
- **Table 3（Qwen3-8B-Base，不同動機的 clip）**：AIME24 GRPO 31.7、GPPO 31.7、Clip-Cov 36.3、RIPO 43.8；BRUMO25 33.3、38.3、36.3、47.5（全表照錄在頁面）。
- **Table 4（PPO 目標 + GAE，GSM8K Avg@1，Qwen2.5-Instruct）**：RIPO-Clip 在 0.5B/1.5B/7B/14B 為 61.1/81.6/93.5/94.4，PPO-Clip 為 58.0/79.2/91.7/93.2。設定：GSM8K 7K 訓練／1K 測試，最大長度 8,192，每次迭代 256 條 rollout、train batch 512、更新 2 次、mini-batch 256，價值模型與策略同模型且無 warmup，15 epoch（435 步），學習率 1e−6。Fig. 3：PPO-Clip 熵迅速衰減到接近 0，RIPO-Clip 維持一定熵、梯度較平滑。
- **Table 5（Pass@k，Qwen3-8B-Base）**：AIME-25 Pass@128 Base 16.7、GRPO 53.3、DAPO 40.0、DCPO 58.9、RIPO 60.0；HMMT-25 Pass@128 Base 6.7、GRPO 30.0、DAPO 26.7、DCPO 40.0、RIPO 45.3。Base 在 k = 16 左右即飽和。作者據此稱 RIPO 突破基底模型的能力邊界。
- **Table 6（Qwen3-8B-Base）**：程式（Eurus-Code 訓練；Codeforces、CodeContest、TACO、APPS）平均 RIPO 44.9、GRPO 39.7（+13.2%）；搜尋（Search-R1 訓練；TriviaQA、PopQA、HotpotQA、WikiMultiHopQA）平均 43.4、37.7（+15.1%）。

### 1.6 實作細節與常見錯誤（頁面 S9 逐條列出）

論文沒有演算法框或程式碼。頁面給一段〔本頁示意〕的 PyTorch 風格虛擬碼，標明「依 Eq. 11、15 寫成，不是作者的程式」：

```
logp_old = logp_old.detach()                  # 行為策略在 rollout 時的 log-prob
ratio    = exp(logp - logp_old)
eps      = sqrt(delta / exp(logp_old))        # Eq. 11，逐 token
surr     = min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A)
loss     = -(surr * mask).sum() / mask.sum()  # Eq. 15：除以全部 token 數
```

1. **ε 用舊機率算**：$\epsilon_{i,t}$ 依 $\pi_{\theta_{old}}$（rollout 時的機率，不帶梯度），不是目前的 $\pi_\theta$；每個 token 不同。
2. **δ 的係數**：Eq. 11 已吸收 Eq. 10 的 2；照 Eq. 9 的寫法，每個 token 的距離上限是 δ/2。換算論文例子時別多乘或少乘 2。
3. **極小機率**：$\pi_{old}\to0$ 時 $\epsilon\to\infty$。論文沒有寫額外的下限保護；實驗用雙重 clipping（0.5 與 10）。〔本頁推導〕若「上界 10」指比值上限，則 $\pi_{old}<\delta/81$（δ = 0.05 時約 6.2×10⁻⁴）的 token 實際上限由 10 決定，而不是 RIC。雙重 clipping 的確切語意論文沒有說明（見 R3）。
4. **正規化方式**：Eq. 15 是 token 層級（除以 $\sum_i|o_i|$），與 Eq. 2 的 GRPO（先對每條回答平均）不同。
5. **上下對稱**：Table 2 中 $\delta_{low}/\delta_{high}\ge2$ 的三組表現大幅下降（各一次訓練）；預設對稱 δ = 0.05。
6. **機率上限**：有效比值上限是 $\min(1+\epsilon,1/\pi_{old})$；高機率 token 的 RIC 上限常常不起作用（步驟 6）。
7. **無 KL 懲罰**：實驗移除 KL 項（Sec. 5.1）；RIC 本身就是 trust region 的替代。

### 1.7 限制與本頁的觀察

- 作者在 Sec. 6 只總結貢獻，未列限制。以下為本頁觀察（頁面 S12 以「本頁推導」標籤列出）：(a) Eq. 7 是二階近似，RIC 允許的低機率比值（例如 5.47）已遠離 $r\approx1$；(b) 單 token 約束不控制分布層級的總變化（步驟 5）；(c) 預設 δ = 0.05 時 RIC 上側處處比 ε = 0.2 寬（步驟 6），實驗中的增益無法只歸因於「限制高機率 token」；(d) 同質變異是低機率端的漸近性質（步驟 7）；(e) 論文沒有消融「固定 ε 調大」或「不 clip」的對照，所以無法從論文數據區分「機率相依的形狀」與「整體放寬」各自的貢獻。W3 玩具模型中，完全不 clip（0.395）與 RIC δ = 0.05（0.370）幾乎相同，上界放寬到 1.0、下界保持 0.2 的 DAPO 式設定更高（0.505）；所以在這個玩具裡，RIC 的效果主要來自「對稀有 token 放寬上界」，不能用來支持「形狀」本身的價值；玩具也沒有大更新的代價，無法顯示為何需要 trust region（〔教學示意〕，R9）。

---

## 2. 學習目標（驗收測試）

共 13 條。每個參考答案都必須寫在該章節的正文段落中（不能只出現在元件、表格、圖或答案框）。每章結尾以 `<details>`（summary「檢查理解」）照錄題目與參考答案。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 寫出重要性比值與 PPO-Clip 的允許範圍 | $r$ 怎麼定義？PPO-Clip 把它限制在哪裡？超出範圍時發生什麼？ | $r=\pi_\theta(a|s)/\pi_{\theta_{old}}(a|s)$；限制在 $(1-\epsilon,1+\epsilon)$，等價於 $|r-1|<\epsilon$；$\hat A>0$ 且 $r>1+\epsilon$（或 $\hat A<0$ 且 $r<1-\epsilon$）時該 token 的梯度為 0（Sec. 2.2、3.2） | S1 |
| LO2 | 用數字說明 PPO-Clip 對低機率 token 的限制 | ε = 0.2 時，$\hat A>0$ 的 $\pi_{old}=0.8$ 與 0.01 的 token，論文所說的「最大可能更新後機率」是多少？DAPO（$\epsilon_{high}=0.28$）呢？ | 0.96（+0.16）與 0.012（+0.002）；DAPO 為 1（0.8 × 1.28 = 1.024 被截到 1）與 0.0128（+0.0008）（Sec. 3.1） | S2 |
| LO3 | （知道層級）認得 KL 的二次近似與 Fisher 矩陣 | KL 在 $\theta_{old}$ 附近的二階近似是什麼？ | $D_{KL}(\pi_{\theta_{old}}\|\pi_\theta)\approx\tfrac12\Delta\theta^\top F\Delta\theta$，$F=\mathbb E_{a\sim\pi_{old}}[\nabla\log\pi\nabla\log\pi^\top]$ 為 Fisher 資訊矩陣，即一維「½ × 曲率 × 偏移²」中曲率的多維版本（Eq. 4–5） | S3 |
| LO4 | 把 KL 近似改寫成比值的加權平方 | 在 $\Delta\pi\approx\nabla\pi^\top\Delta\theta$ 的假設下，Eq. 7 中 $(r-1)^2$ 的權重是什麼？代表什麼？ | 權重是 $\pi_{\theta_{old}}(a|s)$：$D_{KL}\approx\tfrac12\sum_a\pi_{old}(r-1)^2$；同一比值偏差在高機率區距離大、低機率區距離小（Eq. 6–8） | S3 |
| LO5 | 說明歐氏假設及其後果 | PPO-Clip 隱含什麼距離？$|r-1|=0.2$ 時，以論文的單 token 項計，兩個 token 的距離各是多少？這個量少算了什麼？ | $d_{clip}=(r-1)^2$，與 $\pi_{old}$ 無關；$\tfrac12\pi_{old}(r-1)^2$：$\pi_{old}=0.8$ 為 0.016、0.01 為 0.0002（Eq. 3、Sec. 3.2）；它只保留被取樣 token 一項，少算其餘 token 跟著改變的部分 | S4 |
| LO6 | 推導 RIC 的 clip 範圍 | 由 $\tfrac12\pi_{old}(r-1)^2\le\delta$ 得到什麼？實作式為何沒有 2？ | $|r-1|\le\sqrt{2\delta/\pi_{old}}$（Eq. 10）；係數 2 吸收進 δ，$\epsilon_{s,a}=\sqrt{\delta/\pi_{old}}$（Eq. 11），故每個 token 的距離上限為 δ/2，只有到達 clip 邊界時才等於上限 | S5a |
| LO7 | 計算 RIC 的例子 | δ = 0.02 時 $\pi_{old}=0.8$、0.01 的上限與邊界上的距離？ | 0.92（精確 0.9265）與 0.024（0.0241）；兩者邊界上的距離都是 0.01（PPO-Clip：0.016 與 0.0002）（Sec. 4.1） | S5a |
| LO8 | 判斷 RIC 何時比 PPO 緊 | RIC 的 ε 在什麼條件下小於 PPO 的 ε？論文預設 δ = 0.05 時呢？ | $\pi_{old}>\delta/\epsilon^2$；δ = 0.05、ε = 0.2 時分界 1.25，所以對所有 $\pi_{old}$ 都有 $\epsilon_{s,a}\ge0.224>0.2$；$\pi_{old}=0.8$ 時上限受 π ≤ 1 限制而到 1.0〔本頁推導〕 | S5b |
| LO9 | 比較三種權重的偏差與變異貢獻 | clip 邊界上 $v=\pi_{old}w^2$ 對未截斷 IS、PPO、RIC 各是多少？$\pi_{old}\to0$ 時如何？偏差怎麼定義？ | IS：$\pi_\theta^2/\pi_{old}$，可任意大，偏差 0；PPO：$\pi_{old}(1+\epsilon)^2\to0$ 但偏差大；RIC：$\pi_{old}(1+\sqrt{\delta/\pi_{old}})^2=(\sqrt{\pi_{old}}+\sqrt\delta)^2\to\delta$；$A\equiv1$ 時單一 token 偏差為 $\pi_\theta-\pi_{old}w$（Eq. 13–14；精確式與偏差式為本頁推導） | S6 |
| LO10 | 列出 RIPO 相對 GRPO 的改動與實驗設定 | RIPO 的目標改了哪兩處？預設 δ、雙重 clipping、KL 怎麼設？ | clip 範圍改為逐 token 的 $\epsilon_{i,t}=\sqrt{\delta/\pi_{old}}$；正規化改為 $1/\sum_i|o_i|$（token 層級）；δ = 0.05，雙重 clipping 0.5 與 10，移除 KL 懲罰（Eq. 15、Sec. 5.1） | S7 |
| LO11 | 讀出主要實驗結果 | Qwen3-8B-Base 上 RIPO 與 GRPO 的平均？δ 消融與 Pass@128 的結論？ | 38.5 vs 28.5（+35.1%）；對稱 δ 從 0.02 到 0.08 為 40.8–43.8，$\delta_{low}/\delta_{high}\ge2$ 的三組降到 27.5–28.8（各一次訓練）；AIME-25 Pass@128 為 60.0（GRPO 53.3、Base 16.7）（Table 1、2、5） | S8 |
| LO12 | 區分 RIPO 與 Score Centering | 兩者對 $\pi_{old}$ 的假設與修改位置有何不同？ | RIPO 假設 $\pi_{old}$ 就是實際取樣的策略，修改 clip 上下界的形狀；Score Centering（Marek & Ryabinin, arXiv 2609.20807）處理 sampler $q$ 與 trainer $p$ 不一致，從 score 減去 sampler 下的期望 score（加法修正）；兩篇論文都沒有組合實驗 | S10 |
| LO13 | 說出玩具 bandit 示範了什麼、沒示範什麼 | 在 W3 中，固定 ε = 0.2 為何很少找到稀有的好動作？RIC 的增益能歸因於「形狀」嗎？ | B 被取樣到時比值超過 1.2 後梯度為 0，每次迭代只能小幅上升，策略先集中到 A；RIC 讓低機率 token 的上界變大（$p_B=0.01$、δ = 0.05 時上界約 3.24）。不能：不 clip 的對照（0.395）與 RIC（0.370）相近，上界 1.0 的 DAPO 式更高（0.505）；玩具中大更新沒有代價，只能示範「上界太小會阻礙發現」〔教學示意〕 | S7 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），互動圖版加寬到最大 1080px。章節編號以 JetBrains Mono 顯示（`01 · …`）。視覺記號：一條橫向「機率軸」（對數刻度 $10^{-4}$ 到 1），在 hero、W1、W2 重複出現，標出 0.01 與 0.8 兩個例子 token。標題一律用陳述句。用詞：不用「消耗」「預算」，改用「距離（KL 近似值）」「更新幅度的上限」；不用「太緊／太鬆」，改用「上限較小／較大」；不用「爆炸」「發散」，改用「可任意大」。

| # | id | 標題 | 內容 | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈RIPO：讓 PPO 的比值範圍隨舊機率改變〉 | 副標「PPO-Clip 對每個 token 用相同的比值範圍。以 KL 散度的近似衡量，這使低機率 token 能變動的距離上限較小、高機率 token 較大。RIPO 讓範圍隨舊機率改變。」三張事實卡（見下）；連結 arXiv `https://arxiv.org/abs/2607.10169`、`../index.html#paper-587`（在關聯圖中查看）、`ripo-lineage-hub.html`、同批教學 `score-centering.html` | 1 螢幕 |
| S1 | `basics` | 01 · PPO-Clip 限制重要性比值 | 只教：LLM 中的 $s,a$；$\pi_\theta,\pi_{old}$；$r$；$\hat A$（正負決定機率被推高或壓低）；PPO 式與 clip 讓梯度為 0 的規則（LO1）。「trust region」在此給一句定義：「每次更新允許新舊策略相差的最大距離」。符號表（可收合）。GRPO、GAE、TRPO、DAPO、Fisher 都不在此出現。延伸閱讀：`policy-gradient-baselines-score-function.html`、`ppo-grpo-dapo-llm-rl-objectives.html` | 300–400 字 |
| S2 | `collapse` | 02 · 同一個比值範圍對低機率 token 的增量很小 | Sec. 3.1 的數字（LO2），照引「maximum possible updated probability」並說明它是「繼續推高的梯度停止處」；為何增量小很重要：比值超過 1.2 後這個 token 不再被推高，被取樣到一次也只能升一點；DAPO 的 $\epsilon_{low},\epsilon_{high}$ 定義與數字；探索崩潰的定義。一張靜態小表：$\pi_{old}$ 0.8／0.01 × PPO／DAPO 的上限與增量。延伸閱讀：`ripo-euclidean-clip-mismatch.html`、`llm-rl-clipping-variants-heuristics.html` | 300–350 字 + 小表 |
| S3 | `kl` | 03 · KL 散度的二次近似以舊機率加權 | 依 1.5 步驟 4 的教學順序：一維直覺 → Eq. 4（Fisher 定義、$a\sim\pi_{old}$）→ Eq. 6 假設寫在正文 → Eq. 7、8；Eq. 5–6 代數、「Eq. 7 為 ½χ²」、Eq. 4 筆誤註腳都放可收合段落。TRPO 的 Eq. 1 在此以一句話引入（「TRPO 用 KL 不超過 δ 定義 trust region」），連結 `trpo-monotonic-improvement-tv-bound.html`、`ripo-kl-fisher-ratio-geometry.html`、`f-divergence-geometry-proximal-regions.html` | 400–500 字 |
| S4 | `mismatch` | 04 · PPO-Clip 隱含歐氏距離 | Eq. 3；均勻性假設；例子 0.016 vs 0.0002（正文寫明「以單 token 項計」並加 1.5 步驟 5 的那句 Eq. 7 vs Eq. 9 說明，LO5）；Prop. 3.1（分「推導」「觀察」兩句）；callout「單 token 量測與二元量測」〔本頁推導〕：定義二元二階量測與二元精確 KL（W1 進階讀數用），400 倍／1000 倍；連結 `divergence-estimator-topologies.html` | 400–450 字 |
| S5a | `ric` | 05 · RIC 讓每個 token 的距離上限相同 | Eq. 9 → 10 → 11、係數 2、上限只在邊界達到、min 只有一側起作用（LO6）；δ = 0.02 例子（LO7）；Prop. 4.1（分句標出已證／觀察）；**W1（基本檢視）**。延伸閱讀：`ripo-isometric-clip-ric.html` | 350–450 字 + W1 |
| S5b | `delta` | 05b · 實驗預設 δ = 0.05 時 RIC 的上界處處比 PPO 大 | 〔本頁推導〕交界 $\delta/\epsilon^2$、δ = 0.05 的後果（LO8 答案寫成正文：「分界 1.25，因此對所有 π_old，ε_{s,a} ≥ 0.224 > 0.2；π_old = 0.8 時上限受 π ≤ 1 限制而到 1.0，與 DAPO 相同」）、機率上限 1、下側 ≤ 0；正文提示「在上方圖版按『δ = 0.05』並開啟『進階』，比較 π_old = 0.8 的上限」。W1 的進階檢視（DAPO、二元欄位）在這裡才介紹 | 250–300 字 |
| S6 | `variance` | 06 · RIC 使 clip 邊界上單一樣本的變異貢獻接近 δ | 1.5 步驟 7 的先備 (a)–(e)（正文）→ Eq. 12 → 13；$v(x)$；三種權重與偏差定義（LO9 寫成正文）；Eq. 14；〔本頁推導〕精確式、漸近性、單一樣本 vs 總和；截斷簡化說明；**W2**。延伸閱讀：`ripo-bias-variance-homoscedasticity.html`、`importance-sampling-corrections-llm-rl.html` | 500–600 字 + W2 |
| S7 | `ripo` | 07 · RIPO 是換上 RIC 的 GRPO | 先引入 GRPO：Eq. 2、群組標準化優勢（GAE 只在 Table 4 用到，一句帶過）；Eq. 15、兩處改動（LO10）、δ = 0.05、雙重 clipping、無 KL。接著一段白話介紹玩具（「bandit 是只有一步的決策問題：策略在 6 個回答中選一個，每個回答有固定的答對機率；策略是 softmax(logits)，logits 是 6 個可訓練數字；每次答對與否是擲一次機率為答對率的硬幣（伯努利）；seed 是亂數起點，固定 seed 讓結果可重現」）→ **W3** → 正文寫出 LO13 答案（含不 clip 對照的結論） | 450–550 字 + W3 |
| S8 | `results` | 08 · RIPO 在四個模型的平均分數都最高 | Table 1 摘要句（LO11 數字寫進正文）、Fig. 1 四點文字、Table 2（含 R2 方向註記與「各一次訓練」）、Table 3–6 摘要；**W4**。延伸閱讀：`ripo-experiments-results.html`、`sequence-level-ratio-gspo-gmpo.html` | 450–550 字 + W4 |
| S9 | `impl` | 09 · 實作設定與常見錯誤 | 1.5 步驟 9 表格、1.6 虛擬碼（`<pre>`，標〔本頁示意〕）、7 條常見錯誤 | 300–400 字 + 表 |
| S10 | `compare` | 10 · 與 Score Centering 的異同 | 一段話＋對照表（3.1）；另一個比較角度（依 GPPO 論文，Generalized PPO，arXiv 2610.13015），不用分類術語，直接寫各方法的比值範圍：PPO 為固定的 1 ± ε；DPPO（arXiv 2602.04879）約束機率差 |π − π_old| ≤ δ，等價於 1 ± δ/π_old；RIPO 為 1 ± √(δ/π_old)。三者的範圍隨舊機率變小而放寬的速度不同（不變、∝ 1/π_old、∝ 1/√π_old）。連結 `gppo-design-space-unification.html` 與教學 `generalized-ppo-llm-rl.html` | 250–350 字 + 表 |
| S11 | `history` | 11 · 研究脈絡 | 時間線（6.5）＋一段文字：TRPO（KL 約束）→ PPO（固定 ε）→ GRPO（群組優勢）→ DAPO、DCPO、GSPO、GMPO、Clip-Cov、GPPO（Su et al., 2508.07629）（經驗修正）→ RIPO（依 KL 近似決定範圍） | 時間線 + 100–150 字 |
| S12 | `next` | 12 · 限制與延伸閱讀 | 1.7 的 (a)–(e)（標本頁推導）；研究筆記清單（6.6）；arXiv | 200–250 字 + 清單 |
| — | — | 頁尾 | 引用格式、arXiv；`#pm-bar` | — |

目錄中 S5a、S5b 顯示為「05 · RIC」與「05b · δ = 0.05」。

Hero 三張事實卡（不使用未定義術語）：
1. 「PPO-Clip 對每個 token 用同一個比值範圍：機率被推高到原來的 1.2 倍後就不再被推高。原本機率 0.8 的 token 可到 0.96，原本 0.01 的 token 只到 0.012。」
2. 「若用第 3、4 節介紹的距離式衡量新舊策略的差距，同樣 1.2 倍的變化，原本機率 0.8 的 token 的距離是原本機率 0.01 的 80 倍（0.016 對 0.0002）。」卡片底部連結「見第 4 節」→ `#mismatch`。
3. 「RIPO 讓範圍隨舊機率變化，使每個 token 的距離上限相同（見第 5 節）。Qwen3-8B-Base 在七個數學基準的平均分數為 38.5，GRPO 為 28.5。」

Hero 機率軸：對數軸 $10^{-4}$–1，兩個標記 0.01、0.8；每個標記上方兩條短橫線表示允許的機率區間：PPO（ε = 0.2，`--ink-2` 實線＋圓形端點）與 RIC（δ = 0.02，`--accent` 粗線＋方形端點），旁邊直接文字標籤。文字替代：軸容器 `role="img"`，`aria-label`「機率軸：舊機率 0.01 時 PPO 允許到 0.012、RIC（δ = 0.02）到 0.024；舊機率 0.8 時 PPO 到 0.96、RIC 到 0.93」。標籤分層演算法（照錄 benchmark 建置紀錄）：所有標籤以 HTML 絕對定位，依左緣 x 排序；逐一放置，若與同列前一個已放置標籤的水平間距 < 8px，就上移一列（列高 = 字高 + 4px），直到不重疊；用短引線連到標記。

### 3.1 S10 對照表（照 `ripo-vs-score-centering.html`）

| 項目 | RIPO（2607.10169） | Score Centering（2609.20807） |
| --- | --- | --- |
| 關注的問題 | clip 的形狀（幾何）導致探索崩潰 | 訓練引擎與推論引擎機率不一致造成的 drift |
| 對 $\pi_{old}$ 的假設 | $\pi_{old}$ 是實際取樣的策略 | sampler $q$ 與 trainer $p$ 不同 |
| 做法 | 固定 ε 換成 $\sqrt{\delta/\pi_{old}}$ | score 減去 sampler 下的期望 score（加法修正） |
| 修改的位置 | clip 上下界 | policy gradient 的 score 項 |

正文寫出 LO12 的參考答案。Score Centering 欄位的描述出自 Score Centering 論文（arXiv 2609.20807 §4，經 `ripo-vs-score-centering.html` 整理）。再加一句並標〔本頁推導〕：「兩者作用位置不同，原則上可以疊加；但 RIPO 論文沒有討論訓練與推論不一致，兩篇也都沒有組合實驗。」連結 `score-centering.html`、`training-inference-mismatch-sources.html`。

---

## 4. 互動元件規格

### 4.0 共用

- 縮寫：「IS」第一次出現時寫「重要性取樣（importance sampling，IS）」（S6 正文開頭），之後才用 IS。
- 所有圖都是手寫 SVG（沿用 benchmark 的 `el`、`scaleLin`、`scaleLog`、`linePath`）。圖寬等於容器寬，視窗改變時重畫，圖上文字 12–13px。SVG 內文字只用 Unicode 與 `<tspan>`（下標用 `baseline-shift`），不放 KaTeX。
- 方法的編碼（不只靠顏色）：PPO/GRPO＝`--ink-2` 實線＋圓形標記，標「PPO」；DAPO＝`--c-dapo` 長虛線＋三角形，標「DAPO」；RIC/RIPO＝`--accent` 粗實線＋方形，標「RIC」；不 clip（只在 W3）＝`--ink-3` 點線＋菱形，標「不 clip」；未截斷 IS（只在 W2）＝`--ink-3` 點線＋叉形，直接標「IS」。寬度 ≥ 560px 時標籤放在曲線右端；< 560px 時圖內註記（曲線標籤、「受 π ≤ 1 限制」、交界線文字）改成圖下方的圖例列（線型樣本＋文字），圖內只保留線型。
- 滑桿：`<label>`、`aria-valuetext`；live region（`aria-live="polite"`）只在 `change` 或停止輸入 500ms 後更新。所有 hover 資訊都有不靠 hover 的替代：選取點由滑桿控制，讀數固定在圖版內的讀數框。
- 對數滑桿：內部 `<input type="range">` 值 $u\in[0,1000]$ 整數，$x=10^{\log_{10}a+u(\log_{10}b-\log_{10}a)/1000}$；`aria-valuetext` 為 3 位有效數字加名稱（例「舊機率 0.0100」）；鍵盤：方向鍵每次 10、PageUp/PageDown 每次 100（自行處理 `keydown`，`preventDefault` 原生步長）。
- 讀數表在 < 560px 時轉置（方法為欄、量測為列），第一欄 `position:sticky; left:0`，外層 `.tscroll`（`overflow-x:auto`）。W4 的表格檢視與 S8 的 Table 2 同規則。
- reduced-motion：重畫、`<details>` 展開、分頁切換都不加 transition。

### W1 · clip 範圍與單 token 距離（S5a 基本檢視，S5b 進階檢視）

- **教的概念**：PPO 與 RIC 的允許機率區間如何隨 $\pi_{old}$ 變化；以 Eq. 9 衡量，PPO 在上邊界的距離 ∝ $\pi_{old}$，RIC 恆為 δ/2；進階：δ 的取值決定 RIC 在高機率區是否比 PPO 小；DAPO；二元量測顯示單 token 項是近似。
- **檢視**：預設「基本」：只有 PPO、RIC 兩條曲線，輸入只有 $\pi_{old}$ 與 δ（固定 0.02，δ chips 隱藏），讀數表只有 $\pi_{min}$、$\pi_{max}$、$\Delta\pi$、單 token 距離四欄。切換鈕「進階（第 5b 節）」（`aria-pressed`）開啟：DAPO 曲線、ε／$\epsilon_{high}$／δ 滑桿與 δ chips、讀數欄位歐氏、二元二階、二元 KL。
- **輸入**：
  - $\pi_{old}$：對數滑桿 $[10^{-4},0.99]$，預設 0.01；chips「0.01」「0.8」（論文例子）。
  - 進階：ε 0.05–0.5，步長 0.01，預設 0.2；$\epsilon_{high}$ $[\epsilon,0.6]$，步長 0.01，預設 0.28（ε 調高超過它時自動設為 ε）；δ 0.005–0.10，步長 0.005，預設 0.02；chips「δ = 0.02（Sec. 4.1 例子）」「δ = 0.05（實驗預設）」。
- **計算**（`check_w1.py` 同算法）：對每個方法
  - 上側範圍 $h$：PPO ε；DAPO $\epsilon_{high}$；RIC $\sqrt{\delta/\pi_{old}}$。下側 $l$：PPO、DAPO ε（DAPO 的 $\epsilon_{low}$ 取 ε）；RIC 同 $h$。
  - $r_{hi}=\min(1+h,\,1/\pi_{old})$，$r_{lo}=\max(0,1-l)$；$\pi_{max}=\pi_{old}r_{hi}$，$\pi_{min}=\pi_{old}r_{lo}$；$\Delta\pi=\pi_{max}-\pi_{old}$。
  - 量測（都在上邊界 $r=r_{hi}$）：歐氏 $(r-1)^2$（Eq. 3）；單 token $\tfrac12\pi_{old}(r-1)^2$（Eq. 9）；二元二階 $\tfrac12\Delta\pi^2/(\pi_{old}(1-\pi_{old}))$〔本頁推導〕；二元精確 KL 以 log1p 形式計算〔本頁推導〕：$-\pi_{old}\,\mathrm{log1p}(\Delta\pi/\pi_{old})-(1-\pi_{old})\,\mathrm{log1p}(-\Delta\pi/(1-\pi_{old}))$。
- **數值保護**：
  - $\pi_{old}\le0.99$ 使 $1-\pi_{old}\ge0.01$；$\pi_{old}\ge10^{-4}$ 使 $\sqrt{\delta/\pi_{old}}\le31.7$。
  - 二元 KL：$\pi_{max}\ge1-10^{-12}$（例：DAPO 在 $\pi_{old}=0.8$ 剛好到 1）時第二項 $(1-\pi_{old})\ln\frac{1-\pi_{old}}{1-\pi_{max}}\to\infty$，不計算，顯示「∞（其餘 token 機率歸零）」；其他情況用 log1p 式，$\pi_{old}=10^{-4}$、$\Delta\pi$ 很小時不會相消。
  - RIC 下側 $1-h<0$ 時顯示「無下限（r ≥ 0）」。
  - 下圖縱軸 $[10^{-6},0.2]$：值小於 $10^{-6}$（例：$\pi_{old}=10^{-4}$、ε = 0.05 時 PPO 為 1.25×10⁻⁷）時曲線在下緣截斷，截斷處畫向下小三角，讀數表照常顯示實際值。上緣不會超出：滑桿範圍內最大值是 DAPO $\epsilon_{high}=0.6$ 在 $\pi_{old}=1/1.6$ 的 $\tfrac12(1-\pi_{old})^2/\pi_{old}=0.1125$（〔本頁推導〕，受 $r_{hi}\le1/\pi_{old}$ 限制）。
- **顯示**：
  - 上圖「允許的新機率與舊機率之比（上限）」：橫軸 $\pi_{old}$（對數 $10^{-4}$–1），縱軸 $r_{hi}$（對數 1–40）；機率上限造成的折彎處（$r_{hi}=1/\pi_{old}$）畫成細點線並標「受 π ≤ 1 限制」。進階檢視另畫交界 $\pi_{old}=\delta/\epsilon^2$ 的垂直虛線（若 ≤ 1），標「右側 RIC 上限較小」；> 1 時在圖下顯示「此 δ 下 RIC 的上限處處比 PPO 大」。
  - 下圖「上邊界的單 token 距離 ½π_old(r−1)²」：同橫軸；RIC 為水平線 δ/2（高機率端被 π ≤ 1 截斷時下彎），PPO、DAPO 為斜率 1 的直線。
  - 目前 $\pi_{old}$ 以垂直細線同步標在兩圖。
  - 讀數表（列：方法；欄見「檢視」），JetBrains Mono，4 位有效數字；< 560px 轉置（4.0）。
- **讀者應觀察到**：(1) $\pi_{old}=0.01$：PPO 只允許到 0.012，RIC（δ = 0.02）到 0.0241，單 token 距離 0.0002 vs 0.01；(2) $\pi_{old}=0.8$、δ = 0.02：RIC 上限 0.9265 低於 PPO 的 0.96；（進階）切到 δ = 0.05：RIC 上限 1.0，與 DAPO 相同，交界線消失；(3)（進階）高機率時二元量測是單 token 的 $1/(1-\pi_{old})$ 倍；(4)（進階）低機率大比值時二元精確 KL 明顯小於二階近似。

### W2 · 單一 token 的偏差與變異（S6）

- **教的概念**：Eq. 13–14；未截斷 IS、PPO 截斷、RIC 截斷三種權重的變異貢獻 $v=\pi_{old}w^2$ 與偏差；RIC 何時與 IS 相同。正文已先教 1.5 步驟 7 的先備 (a)–(e)。
- **簡化**（頁面寫出）：依 Sec. 4.2 把 clip 寫成截斷 $w=\min(r,\text{上限})$；取 $A\equiv1$，該 token 的期望貢獻應為 $\pi_\theta$，截斷後為 $\pi_{old}w$，差 $\pi_\theta-\pi_{old}w$ 即偏差。只看上側（論文只分析 $r\le$ 上限）。
- **輸入**：
  - $\pi_{old}$ 對數滑桿 $[10^{-5},0.9]$，預設 0.001。
  - 目標比值 $r$：對數滑桿，下界 1、上界 $\min(1000,1/\pi_{old})$，預設 10；$\pi_\theta=\pi_{old}r$。$\pi_{old}$ 改變時：保留 $r$ 的數值；若 $r>1/\pi_{old}$ 則設為 $1/\pi_{old}$，並在滑桿旁顯示「r 已調為 1/π_old（π_θ 不能超過 1）」3 秒（同時寫入 live region 一次）；滑桿的上界與 thumb 位置依新範圍重算。
  - ε（0.05–0.5，預設 0.2）、δ（0.005–0.1，預設 0.05）；勾選框「套用雙重 clipping 上界 10（若其語意為比值上限，見 R3）」，預設不勾。
- **計算**（`check_w2.py` 同算法）：上限 IS ∞、PPO $1+\epsilon$、RIC $1+\sqrt{\delta/\pi_{old}}$；勾選時三者再取 $\min(\cdot,10)$。$w=\min(r,\text{上限})$，$v=\pi_{old}w^2$，偏差 $=\pi_\theta-\pi_{old}w$。
- **數值保護**：$\pi_{old}\ge10^{-5}$；$r\le1/\pi_{old}$；IS 的 $v$ 最大 $=1/\pi_{old}\le10^5$，用科學記號顯示；縱軸外的值同 W1 以小三角標示。
- **顯示**：
  - 左圖「clip 邊界上的變異貢獻」：橫軸 $\pi_{old}$（對數 $10^{-5}$–1），縱軸對數 $10^{-6}$–10。曲線：PPO $\pi_{old}(1+\epsilon)^2$、RIC $(\sqrt{\pi_{old}}+\sqrt\delta)^2$、IS 曲線 $\pi_\theta^2/\pi_{old}$，其中 $\pi_\theta$ 固定為目前滑桿給的 $\pi_{old}r$（只畫 $\pi_{old}\le\pi_\theta$ 的部分，直接標「IS（π_θ = …）」）；水平參考線 δ（標「δ」）。目前點：IS 叉形、PPO 圓形、RIC 方形，畫在各自 $(\pi_{old},v)$。
  - 右側讀數表：列 IS／PPO／RIC，欄「權重 w」「變異貢獻 v」「偏差 π_θ − π_old·w」，另一行「此 token 是否被 RIC 截斷：是／否」。
- **讀者應觀察到**：(1) 預設（$\pi_{old}=0.001,r=10$）：IS $v=0.1$、偏差 0；PPO $v=0.00144$、偏差 0.0088；RIC $v=0.0651$、偏差 0.00193，介於兩者之間；(2) 把 $r$ 降到 8.07 以下，RIC 與 IS 完全相同；(3) $\pi_{old}\to10^{-5}$ 時 RIC 曲線趨近 δ，PPO 趨近 0，IS 可任意大；(4) $\pi_{old}$ 大於 $\delta/\epsilon^2$ 時 RIC 比 PPO 截得更多（δ = 0.05 時不會發生；δ 調到 0.01 時分界 0.25）。

### W3 · 玩具 bandit：稀有的好動作能否被找到（S7）〔教學示意〕

- **教的概念**：固定的小上界讓偶爾被取樣到的低機率好動作每次只能小幅上升，策略先集中到高機率的次佳動作；RIC 對低機率 token 放寬上界。並附「不 clip」對照，讓讀者看到在這個玩具中 RIC 的效果主要來自放寬，而不是形狀（LO13）。頁面寫「此模型為本頁設計的示意，不是論文的實驗或結果」。
- **模型**（全部自選；S7 正文先用白話說明 bandit、softmax logits、伯努利獎勵、seed）：6 個動作（「回答」），表格式 softmax 策略，logits 初始為初始機率的對數。初始機率：A $=0.81-p_B$、B $=p_B$、其餘 4 個各 0.0475。答對機率：A 0.5、B 1.0、其餘 0。每次迭代：從 $\pi_{old}$ 取樣 $G=8$ 個動作，獎勵為伯努利；GRPO 優勢 $(R-\text{mean})/\text{std}$（母體標準差，除以 G）；std < 1e−12（8 個獎勵全同）則略過此迭代並計數。對同一批樣本做 $m=4$ 次全批梯度上升，學習率 1.0，目標 $\frac1G\sum_i\min(r_i\hat A_i,\mathrm{clip}(r_i,1-l_i,1+h_i)\hat A_i)$。梯度解析式：該項未被 clip 時，$\partial r_i/\partial z=r_i(e_{a_i}-\pi)$；判定「未被 clip」：$\hat A_i>0$ 時 $r_i\le1+h_i$，$\hat A_i<0$ 時 $r_i\ge1-l_i$。$h_i,l_i$：GRPO 皆 ε；DAPO $l=\epsilon$、$h=\epsilon_{high}$；RIC $h_i=l_i=\sqrt{\delta/\pi_{old}(a_i)}$；不 clip $h=l=\infty$。$T=200$，$S=200$ 個 seed（seed = 1…200），PRNG `mulberry32`，取樣用累積機率比較 `u < c`。程式碼即 `verify/sim.js` 的 `runSeed`／`batch`（含 `'none'` 分支與 `skipped` 計數），原樣貼入。
- **輸入**：$p_B$ 分段按鈕 0.005／0.01／0.02（預設 0.01）；ε 0.1–1.0 步長 0.05（預設 0.2）；$\epsilon_{high}$ $[\epsilon,1.0]$ 步長 0.01（預設 0.28）；δ 0.005–0.2 步長 0.005（預設 0.05）；按鈕「執行」。四條線（GRPO、DAPO、RIC、不 clip）每次一起跑，共 800 次模擬。
- **執行與狀態**：
  - 狀態機：`idle`（按鈕「執行」，讀數區「尚未執行」）→ `running`（按鈕停用、文字「執行中 n/800」）→ `done`（按鈕「重新執行」）；`done` 或 `idle` 時改任何參數 → `stale`（按鈕「參數已變更，按執行更新」，舊結果淡化 opacity .45 仍顯示）。`running` 時改參數：記為 stale，當次結果完成後立刻進入 `stale`。
  - 不自動執行：首次捲入視窗後以 `requestIdleCallback`（不支援時 `setTimeout(…, 500)`）啟動一次預設參數的執行，不阻塞首次繪製；之後只由按鈕觸發。
  - 每個 `setTimeout` 片段跑 20 個 seed；每次執行遞增 `runToken`，片段與 `onDone` 先檢查 token，不符就丟棄（舊執行不能覆蓋新結果）。
  - live region 只在 25%、50%、75%、100% 宣告一次（「玩具模擬完成 50%」；100% 時宣告四個方法第 200 次的找到比例）。
  - 儲存：每個方法三個 `Float64Array(201)`（找到比例、平均熵、平均 $\pi_B$），加 `skipped` 總數與期望獎勵；四個方法共約 19 KB。單次全跑約 0.4 s（node 實測 24 組 batch 共約 2.5 s）。
- **顯示**：
  - 左圖「找到 B 的比例」：橫軸迭代 0–200，縱軸 0–1，值為 $\pi_B>0.5$ 的 seed 比例，四條線（線型與標記見 4.0）。
  - 右圖「平均熵」：同橫軸，縱軸 0–$\ln6$（1.79）。圖下一句：「所有方法的熵最後都接近 0，因為這個問題只有一個最佳動作；論文 Fig. 1(b) 的『熵穩定在中等範圍』無法在此重現。」
  - 讀數表（第 200 次迭代）：方法、參數、找到 B 的比例、平均熵、平均期望獎勵（$0.5\pi_A+\pi_B$）、「每個 seed 平均略過的迭代數」（skipped / 200）。表下一句：「略過的迭代是 8 個回答的獎勵全相同、優勢全為 0 的情形；找到 B 之後幾乎每次都全對，所以找到越多的方法略過越多。」
- **讀者應觀察到**：(1) 預設下 GRPO 0.015、DAPO 0.100、RIC 0.370、不 clip 0.395；(2) RIC 與不 clip 相近：在這個玩具中大更新沒有代價，RIC 的增益主要來自對 B 放寬上界；(3) 把 DAPO 的 $\epsilon_{high}$ 調到 1.0 得 0.505，高於 RIC：只放寬上界（保留下界 0.2）在此更有效，所以玩具不能支持「形狀」本身的價值；(4) $p_B=0.005$ 時都難找到 B（GRPO 0.000、RIC 0.155、不 clip 0.165），$p_B=0.02$ 時都較容易。

### W4 · 實驗結果瀏覽（S8）

- **教的概念**：讀 Table 1、2、5；RIPO 的平均最高但不是每格最高；δ 上下比例的影響；Pass@k 隨 k 的變化。
- **輸入**：分頁（`role="tablist"`，方向鍵切換，`aria-selected`）「Table 1 各基準」「Table 2 δ 消融」「Table 5 Pass@k」。Table 1 頁：模型分段按鈕（4 個，預設 Qwen3-8B-Base），基準下拉（7 個 + 平均，預設平均）。Table 5 頁：資料集 AIME-25／HMMT-25。
- **資料**：內嵌 `const T1, T2, T5`，數字逐字照錄原文表格（建置時從 `src/paper.txt` 複製，不手打；selfTest 抽查 6 格）。
- **顯示**：
  - Table 1：水平長條圖（Base、GRPO、DAPO、GSPO、GMPO、DCPO、RIPO），RIPO 用 `--accent` 並在方法名後加「（本文）」，其他 `--ink-3`；長條末端數字；該欄最高值加「最高」文字標籤（並列都標）。平均欄顯示論文括號中的相對 GRPO 百分比，註腳「百分比照錄論文；以表中四捨五入後的平均重算，Qwen3-1.7B-Base 為 +37.5%（論文 +37.2%）」。
  - Table 2：點圖，橫軸 Avg@8（25–45），7 個設定依原表順序列出「δ_low / δ_high」；對稱設定實心圓、不對稱空心圓，並列印數字。圖下照錄 R2 的方向註記與「每個設定一次訓練」。
  - Table 5：折線圖，橫軸 k ∈ {1,8,16,32,64,128}（對數 2），縱軸 Pass@k；5 條（Base、GRPO、DAPO、DCPO、RIPO），標記形狀各異，右端直接標籤。Avg@8 不畫。
  - 每頁有 `<details>`「以表格檢視」提供原始數字表（`<caption>` 註明出處）。分頁切換或 `<details>` 展開後若插入含 TeX 的節點，呼叫 `renderMathInElement(node, opts)`。
- **數值保護**：無計算。
- **讀者應觀察到**：(1) Table 1 平均欄：四個模型中 RIPO 都最高（Qwen3-8B-Base 38.5 vs GRPO 28.5）；切到個別基準時 RIPO 不是每格最高（Qwen3-8B AIME25 GMPO 30.0 > 29.2；Qwen3-1.7B AMC23 DCPO 46.9 > 46.1）；(2) Table 2：對稱的 0.02、0.05、0.08 為 40.8、43.8、42.1，0.05/0.04 為 41.7；$\delta_{low}/\delta_{high}\ge2$ 的三組（0.05/0.02、0.08/0.02、0.08/0.04）降到 28.8、27.5、27.9；(3) Table 5：Base 在 k = 16 後持平（AIME-25 16.7），RIPO 在 k = 128 最高（AIME-25 60.0、HMMT-25 45.3），DAPO 在 AIME-25 的 Pass@128（40.0）低於 GRPO（53.3）。

---

## 5. 驗收數值表

所有數字由第 1 行列出的腳本產生；頁面算法相同。

### 5.1 W1（`check_w1.out`；容許誤差：機率 ±5e−5，距離 ±1e−6 相對 0.1%）

| δ | $\pi_{old}$ | 方法 | $r_{lo}$ | $r_{hi}$ | $\pi_{max}$ | $\Delta\pi$ | 歐氏 | 單 token | 二元二階 | 二元 KL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.02 | 0.8 | PPO | 0.8000 | 1.2000 | 0.9600 | 0.16000 | 0.0400 | 0.016000 | 0.080000 | 0.176030 |
| 0.02 | 0.8 | DAPO | 0.8000 | 1.2500 | 1.0000 | 0.20000 | 0.0625 | 0.025000 | 0.125000 | ∞ |
| 0.02 | 0.8 | RIC | 0.8419 | 1.1581 | 0.9265 | 0.12649 | 0.0250 | 0.010000 | 0.050000 | 0.082748 |
| 0.02 | 0.5 | RIC | 0.8000 | 1.2000 | 0.6000 | 0.10000 | 0.0400 | 0.010000 | 0.020000 | 0.020411 |
| 0.02 | 0.1 | RIC | 0.5528 | 1.4472 | 0.1447 | 0.04472 | 0.2000 | 0.010000 | 0.011111 | 0.008907 |
| 0.02 | 0.01 | PPO | 0.8000 | 1.2000 | 0.0120 | 0.00200 | 0.0400 | 0.000200 | 0.000202 | 0.000179 |
| 0.02 | 0.01 | DAPO | 0.8000 | 1.2800 | 0.0128 | 0.00280 | 0.0784 | 0.000392 | 0.000396 | 0.000335 |
| 0.02 | 0.01 | RIC | 0（無下限） | 2.4142 | 0.0241 | 0.01414 | 2.0000 | 0.010000 | 0.010101 | 0.005430 |
| 0.02 | 0.001 | RIC | 0（無下限） | 5.4721 | 0.0055 | 0.00447 | 20.000 | 0.010000 | 0.010010 | 0.002783 |
| 0.05 | 0.8 | RIC | 0.7500 | 1.2500 | 1.0000 | 0.20000 | 0.0625 | 0.025000 | 0.125000 | ∞ |
| 0.05 | 0.5 | RIC | 0.6838 | 1.3162 | 0.6581 | 0.15811 | 0.1000 | 0.025000 | 0.050000 | 0.052680 |
| 0.05 | 0.01 | RIC | 0（無下限） | 3.2361 | 0.0324 | 0.02236 | 5.0000 | 0.025000 | 0.025253 | 0.010873 |
| 0.05 | 0.001 | RIC | 0（無下限） | 8.0711 | 0.0081 | 0.00707 | 50.000 | 0.025000 | 0.025025 | 0.005008 |

邊界情況：$\pi_{old}=10^{-4}$、RIC δ = 0.02：單 token 0.010000、二元二階 0.010001、二元 KL 0.001143（log1p 式）；$\pi_{old}=10^{-4}$、PPO ε = 0.05：單 token 1.25×10⁻⁷（低於下圖下緣，畫截斷標記）。

其他：交界 $\delta/\epsilon^2$：δ = 0.02 → 0.5，δ = 0.05 → 1.25（不顯示交界線）；RIC 上限被 π ≤ 1 截斷的起點：δ = 0.02 為 $\pi_{old}>0.8682$，δ = 0.05 為 $>0.8$。

### 5.2 W2（`check_w2.out`；容許 ±1e−6 或相對 0.1%）

| $\pi_{old}$ | $r$ | 上界 10 | IS w / v / 偏差 | PPO w / v / 偏差 | RIC w / v / 偏差 |
| --- | --- | --- | --- | --- | --- |
| 0.001 | 10 | 否 | 10 / 0.100000 / 0 | 1.2 / 0.001440 / 0.008800 | 8.0711 / 0.065142 / 0.001929 |
| 0.0001 | 100 | 否 | 100 / 1.000000 / 0 | 1.2 / 0.000144 / 0.009880 | 23.3607 / 0.054572 / 0.007664 |
| 0.0001 | 100 | 是 | 10 / 0.010000 / 0.009000 | 1.2 / 0.000144 / 0.009880 | 10 / 0.010000 / 0.009000 |
| 0.01 | 2 | 否 | 2 / 0.040000 / 0 | 1.2 / 0.014400 / 0.008000 | 2 / 0.040000 / 0（未截斷） |
| 0.3 | 1.3333 | 否 | 1.3333 / 0.533333 / 0 | 1.2 / 0.432000 / 0.040000 | 1.3333 / 0.533333 / 0 |

邊界包絡（ε = 0.2、δ = 0.05）：$\pi_{old}$ = 1e−5、1e−4、1e−3、0.01、0.1、0.5 → PPO 0.000014、0.000144、0.001440、0.014400、0.144000、0.720000；RIC 0.051424、0.054572、0.065142、0.104721、0.291421、0.866228。

### 5.3 W3（`verify/sim_js.out`，`check_w3.py` 獨立移植逐位一致；容許：比例 ±0.02、熵 ±0.005）

格式「找到 B 的比例 / 平均熵」，第 25、50、100、200 次迭代；獎勵為第 200 次的平均期望獎勵；略過＝每個 seed 平均略過的迭代數（skipped 總數 / 200）。「DAPO 0.2/1.0」即 DAPO 介面把 $\epsilon_{high}$ 調到 1.0。

| $p_B$ | 方法 | @25 | @50 | @100 | @200 | 獎勵 | 略過 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 | GRPO ε = 0.2 | 0.000 / 0.174 | 0.005 / 0.100 | 0.015 / 0.059 | 0.015 / 0.035 | 0.506 | 3.39 |
| 0.01 | DAPO 0.2/0.28 | 0.005 / 0.181 | 0.010 / 0.117 | 0.065 / 0.068 | 0.100 / 0.038 | 0.548 | 11.55 |
| 0.01 | RIC δ = 0.05 | 0.065 / 0.136 | 0.175 / 0.089 | 0.275 / 0.038 | 0.370 / 0.023 | 0.684 | 46.16 |
| 0.01 | 不 clip | 0.130 / 0.112 | 0.215 / 0.079 | 0.300 / 0.035 | 0.395 / 0.022 | 0.696 | 52.51 |
| 0.01 | RIC δ = 0.02 | 0.050 / 0.138 | 0.140 / 0.086 | 0.270 / 0.049 | 0.385 / 0.025 | 0.690 | 45.09 |
| 0.01 | GRPO ε = 1.0 | 0.050 / 0.109 | 0.065 / 0.060 | 0.110 / 0.042 | 0.195 / 0.023 | 0.596 | 21.61 |
| 0.01 | RIC δ = 0.005 | 0.030 / 0.133 | 0.070 / 0.099 | 0.205 / 0.056 | 0.345 / 0.038 | 0.673 | 34.87 |
| 0.01 | DAPO 0.2/1.0 | 0.100 / 0.226 | 0.235 / 0.135 | 0.325 / 0.082 | 0.505 / 0.041 | 0.750 | 59.87 |
| 0.005 | GRPO ε = 0.2 | 0.000 / 0.158 | 0.000 / 0.090 | 0.000 / 0.051 | 0.000 / 0.029 | 0.498 | 1.49 |
| 0.005 | DAPO 0.2/0.28 | 0.000 / 0.161 | 0.000 / 0.094 | 0.005 / 0.053 | 0.010 / 0.031 | 0.503 | 2.23 |
| 0.005 | RIC δ = 0.05 | 0.015 / 0.094 | 0.045 / 0.066 | 0.100 / 0.043 | 0.155 / 0.021 | 0.577 | 18.50 |
| 0.005 | 不 clip | 0.030 / 0.091 | 0.055 / 0.065 | 0.115 / 0.035 | 0.165 / 0.021 | 0.582 | 20.39 |
| 0.02 | GRPO ε = 0.2 | 0.045 / 0.248 | 0.135 / 0.152 | 0.250 / 0.097 | 0.355 / 0.052 | 0.676 | 41.04 |
| 0.02 | DAPO 0.2/0.28 | 0.085 / 0.281 | 0.265 / 0.170 | 0.450 / 0.086 | 0.580 / 0.062 | 0.788 | 70.64 |
| 0.02 | RIC δ = 0.05 | 0.320 / 0.164 | 0.520 / 0.089 | 0.645 / 0.038 | 0.785 / 0.022 | 0.891 | 111.93 |
| 0.02 | 不 clip | 0.370 / 0.126 | 0.565 / 0.071 | 0.660 / 0.030 | 0.795 / 0.016 | 0.896 | 117.96 |

第 0 次迭代的熵：$p_B$ = 0.01 為 0.804、0.005 為 0.780、0.02 為 0.843。

### 5.4 論文數字的一致性（`check_paper.out`）

容許：重算百分比與論文差 ≤ 0.5 個百分點視為一致（四捨五入造成）。Table 1 平均相對 GRPO 重算：+37.5%（論文 37.2%，差異來自四捨五入前的值）、+34.4%、+17.1%、+35.1%；AIME24 相對 GRPO 最大 +61.9%（Qwen3-1.7B-Base），對應摘要「up to 60%」。Sec. 5.1：128 × 8 = 1024 條，1024 / 128 = 8 次更新，一致。Table 6 程式平均重算 +13.1%（論文 13.2%）。

---

## 6. 實作計畫

單檔 `tut/587/build/index.html`，上線為 `tutorials/ripo-riemannian-isometric-clip.html`。**沿用** `tutorials/prediction-targets-k-diff.html` 的整個外殼：CSS tokens 與深色模式、appbar 與主題切換（`pm-theme`）、`#pm-bar`、左側章節目錄／窄螢幕頂部章節列、`.plate`／`.plate-h`／`.plate-f`／`.readout`／`.callout`（`.note`）／`.fact`／`.check`（`<details>`）／`.tscroll` 元件、`.eq` + `.eq-n` 方程式編號、KaTeX 0.16.9 設定與 `renderMath()`、`el`／`scaleLin`／`scaleLog`／`linePath`、時間線 `renderTimeline` 與卡片列樣式、`?test` 的 `selfTest()` 框架（含 `scrollWidth <= innerWidth` 檢查）、5.6／5.8／5.9 的引用、文案、無障礙規則。以下只列新增或不同處。

### 6.1 新增 tokens

- `--c-dapo`：light #3F6E8C、dark #8DB4D3（借用 benchmark 的 `--c-par` 色值，改名以符合語意）。只用於圖中 DAPO 系列。已算對比：light 對 `--surface` #FFFFFF 5.5:1、對 `--bg` #F6F4EE 5.0:1；dark 對 `--surface` #1F1E1B 7.6:1、對 `--bg` #161513 8.3:1，圖形與文字標籤都合格。`--ink-3`（IS／不 clip 點線）沿用 benchmark 的 #67635A（對白 6.0:1）與 dark #9C978C（對 #1F1E1B 5.7:1）。
- 刪除 benchmark 的 `--c-perp`（本頁不用）。

### 6.2 頁面連結

- Hero：arXiv `https://arxiv.org/abs/2607.10169`；`../index.html#paper-587`；`ripo-lineage-hub.html`；`score-centering.html`。
- S10：`score-centering.html`、`generalized-ppo-llm-rl.html`（GPPO 教學）、`gppo-design-space-unification.html`、`training-inference-mismatch-sources.html`、`ripo-vs-score-centering.html`。

### 6.3 程式結構（新函式）

- W1：`initW1()`；`ricEps(delta, po)`；`bounds(method, po, P)` → `{rlo, rhi}`；`measures(po, r)` → `{pmax, dp, eucl, atom, binchi, binkl}`（與 `check_w1.py` 同式）；`renderW1()`。
- W2：`initW2()`；`capsW2(po, eps, delta, dual)`；`rowW2(po, r, P)`；`renderW2()`。
- W3：`initW3()`；貼入 `verify/sim.js` 的 `mulberry32`、`softmax`、`entropy`、`runSeed`、`batch`（含 `'none'` 分支與 `trace.skipped`；去掉 node 專用的 `module.exports` 與 `require.main` 區塊），另寫 `batchChunked(cfgs, pB, token, onProgress, onDone)` 以 `setTimeout` 每片 20 seed 呼叫 `runSeed`（trace 累加邏輯與 `batch` 相同，每片開頭比對 `token === runToken`）；`state = {status:'idle'|'running'|'done'|'stale', runToken, results}`；`renderW3(results)`。
- 版面：`.eq` 設 `overflow-x:auto; overflow-y:hidden`，在 390px 檢查 Eq. 15（三層分式）與 Eq. 2 不撐破版面；Eq. 15 在 < 600px 時把 $\min(\cdot)$ 換到第二行（`\\` + `aligned`）。
- KaTeX：靜態 DOM 中的 TeX 在初始化時 `renderMath()` 一次；`<details>` 的 `toggle` 與 W4 分頁切換後對新插入節點呼叫 `renderMathInElement`。
- W4：`initW4()`；`const T1 = {model: {method: [7 個基準…, 平均]}}`、`const T2 = [[δlow, δhigh, avg], …]`、`const T5 = {dataset: {method: [Avg@8, P@1, P@8, P@16, P@32, P@64, P@128]}}`；`renderW4(tab)`；tablist 鍵盤左右鍵切換。
- Hero 機率軸：`renderHeroAxis()`（HTML 實作，沿用 benchmark k 軸的標籤分層演算法）。

### 6.4 selfTest（`?test`）

- W1：二元 KL 在 DAPO、$\pi_{old}=0.8$ 時回傳 `Infinity`；$\pi_{old}=10^{-4}$、RIC δ = 0.02 時 = 0.001143（±1e−6）。`bounds('PPO',0.8)` → $\pi_{max}$ 0.96；`bounds('PPO',0.01)` → 0.012；`bounds('DAPO',0.01)` → 0.0128；`bounds('DAPO',0.8)` → 1.0；RIC δ = 0.02：0.8 → 0.92649、0.01 → 0.024142；`measures` 單 token：PPO 0.8 → 0.016、0.01 → 0.0002；RIC 恆為 δ/2（0.01）；二元 KL（δ = 0.02，0.01）= 0.005430（±1e−6）。
- W2：$\pi_{old}=0.001$、$r=10$：RIC $w=8.0711$、$v=0.065142$、偏差 0.001929；包絡 RIC(1e−5) = 0.051424。
- W3：`batch({method:'ppo',eps:0.2},0.01)` 第 200 次 found = 0.015；`batch({method:'ric',delta:0.05},0.01)` = 0.370；`batch({method:'none'},0.01)` = 0.395（皆 ±0.02）；GRPO 的 skipped 總數 677（±20）。
- 頁面：`document.querySelectorAll('.katex-error').length === 0`；`scrollWidth <= innerWidth`。
- W4：抽查 T1 Qwen3-8B-Base RIPO 平均 38.5、GRPO 28.5、AIME24 RIPO 43.8；T2 第 3 欄 43.8；T5 AIME-25 RIPO Pass@128 60.0、HMMT-25 45.3。
- 輸出 `selfTest: N passed, M failed`。

### 6.5 研究脈絡（notes/timeline.json）

- 建置時把 `tut/587/notes/timeline.json` 原樣貼進 `<script type="application/json" id="timeline-data">`。目前 13 筆：1502.05477、1506.02438、1707.06347、2409.19256、2402.03300、2501.12948、2503.14476、2505.22617、2507.18071、2507.20673、2508.07629、2509.02333、2607.10169（本文）。
- `TL_GROUP`（以 arXiv ID 為 key，卡片連到的筆記）：

  | 類別 | arXiv ID | 筆記 |
  | --- | --- | --- |
  | trust region | 1502.05477、1506.02438 | `trpo-monotonic-improvement-tv-bound.html` |
  | 固定 clip | 1707.06347、2402.03300、2501.12948 | `ppo-grpo-dapo-llm-rl-objectives.html` |
  | clip 修正 | 2503.14476、2505.22617、2509.02333、2508.07629 | `llm-rl-clipping-variants-heuristics.html` |
  | 序列層級 | 2507.18071、2507.20673 | `sequence-level-ratio-gspo-gmpo.html` |
  | 平台 | 2409.19256 | `ripo-experiments-results.html` |
  | 本文 | 2607.10169 | `ripo-isometric-clip-ric.html` |

  注意 2508.07629（Klear-Reasoner 的 GPPO，gradient-preserving）與站內教學 `generalized-ppo-llm-rl.html` 的 GPPO（arXiv 2610.13015）是不同方法；卡片與 S8、S10 文字一律寫全名或 arXiv ID 區分（R6）。`llm-rl-clipping-variants-heuristics` 未涵蓋 2508.07629，建置時改連 `ripo-experiments-results.html`。

### 6.6 延伸閱讀（S12，依 hub 的閱讀順序）

`ripo-lineage-hub.html`（總覽）→ `policy-gradient-baselines-score-function.html` → `trpo-monotonic-improvement-tv-bound.html` → `ppo-grpo-dapo-llm-rl-objectives.html` → `ppo-clip-tv-single-sample-proxy.html` → `llm-rl-clipping-variants-heuristics.html` → `sequence-level-ratio-gspo-gmpo.html` → `ripo-kl-fisher-ratio-geometry.html` → `ripo-euclidean-clip-mismatch.html` → `ripo-isometric-clip-ric.html` → `ripo-bias-variance-homoscedasticity.html` → `ripo-experiments-results.html` → `ripo-vs-score-centering.html` → `importance-sampling-corrections-llm-rl.html`；另列 `f-divergence-geometry-proximal-regions.html`、`divergence-estimator-topologies.html`、`gppo-design-space-unification.html`，以及教學 `score-centering.html`、`generalized-ppo-llm-rl.html`。每項附 frontmatter 的 `title`（建置時讀取，不手打）。

### 6.7 文案規則（補充 benchmark 5.8）

- 用詞：「探索崩潰」（exploration collapse）、「重要性比值」、「行為策略」、「trust region」（不翻）、「同質變異」（homoscedasticity，第一次附英文）、「等距」（isometric）。不寫「黎曼」以外的譯名混用：統一「Riemannian 流形」。
- 區分三種陳述：論文結果、作者的解釋（「作者認為」）、本頁推導／教學示意。
- 本文件中每個標〔本頁推導〕〔本頁說明〕〔本頁示意〕〔教學示意〕的段落，頁面上都要在該句旁顯示對應的 pill 標籤（`.tag`），不得省略。
- R2 的 Table 2 註記在頁面上逐字使用：「表中後三欄為 δ_low 大於 δ_high，與正文敘述方向不同，原文如此。」

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 例子的截位 | Sec. 4.1 寫 0.92，精確值 0.9265。 | 頁面寫「0.92（精確值 0.9265）」。 |
| R2 | Table 2 方向與正文相反 | 正文說「δ_high 明顯大於 δ_low 時表現下降」並解釋為「機率更容易增加」；表中後三欄是 δ_low > δ_high（0.05/0.02、0.08/0.02、0.08/0.04）。可能是表頭列對調或正文筆誤，無法判斷。 | 頁面照錄表與正文，加一行註記「表中後三欄為 δ_low 大於 δ_high，與正文敘述方向不同，原文如此」。LO11 只寫「$\delta_{low}/\delta_{high}\ge2$ 的三組」，不寫正文的方向；並註明 0.05/0.04 也不對稱但得 41.7、每組只有一次訓練。 |
| R3 | 雙重 clipping 語意 | Sec. 5.1 只寫「lower and upper bounds set to 0.5 and 10」，引 Ye et al. (2020)。常見的 dual-clip 是在優勢為負時把目標下限設為 $c\hat A$；本文的 0.5 與 10 可能是比值的上下限。兩種解讀對 RIC 的作用不同（例如比值下限 0.5 時，$\pi_{old}<4\delta$ 的 token 的 RIC 下側被取代）。 | 頁面照錄原文，不選定解讀；W2 的「上界 10」勾選框標明「若其語意為比值上限」。 |
| R4 | Sec. 5.6 batch 數字矛盾 | 「每次迭代 256 條 rollout、train batch 512」互相矛盾；GSM8K 訓練集 7,473 題，7473 / 256 取整 29 × 15 = 435，與「435 步」吻合，推測 batch 為 256 題。 | 頁面照錄原文，不寫推測；只在 S9 註一句「原文 batch 數字互相矛盾」。 |
| R5 | 預設 δ 下 RIC 處處較寬 | δ = 0.05 時 RIC 上側處處比 ε = 0.2 寬（〔本頁推導〕），論文沒有「固定 ε 調大」的對照。 | S5b 正文寫明（LO8）；S12 列為觀察 (c)(e)；W1 進階檢視提供 δ 切換；W3 加入「不 clip」對照，結果顯示玩具中的增益主要來自放寬上界（LO13）。 |
| R6 | 兩個 GPPO | Sec. 5.5 的 GPPO 是 Su et al. 2025（arXiv 2508.07629，gradient-preserving）；站內 `generalized-ppo-llm-rl` 是 arXiv 2610.13015。 | 頁面每次出現都附 arXiv ID 或全名；S10 用「GPPO（Generalized PPO，arXiv 2610.13015）」。 |
| R7 | 二階近似的有效範圍 | RIC 允許低機率 token 的比值遠離 1（δ = 0.05、$\pi_{old}=0.001$ 時 8.07），Eq. 7 的近似不再準確；論文沒有討論。 | W1 列二元精確 KL；S12 觀察 (a)。 |
| R8 | Fig. 1–3 沒有數據 | 只能引用正文對曲線的描述。 | 不重繪曲線；W4 只畫表格數據。 |
| R9 | 玩具結論外推 | W3 的結果依賴自選的動作數、獎勵、學習率；熵最後都接近 0，與論文 Fig. 1(b) 不同。 | 圖版標〔教學示意〕＋固定說明句；正文不把 W3 數字當論文證據。round 1 加入不 clip 對照後，原先「差異來自形狀」的說法已撤回：不 clip 0.395 ≈ RIC 0.370，DAPO 上界 1.0 為 0.505；W3 改為只示範「固定的小上界阻礙發現稀有動作」，並明寫玩具中大更新沒有代價、不能說明為何需要 trust region。 |
| R10 | 浮點一致性 | W3 的取樣以 `u < c` 判定，JS 與 Python 的 `exp` 可能差 1 ulp 而改變個別 seed。目前逐位一致。 | 驗收容許比例 ±0.02。 |
| R11 | Eq. 4 筆誤 | 原文 $\Phi(\theta)\approx\Phi(\theta)+\dots$ 應為 $\Phi(\theta_{old})$。 | 頁面寫正確式並以註腳說明。 |
| R12 | 相對連結 | 建置位置與上線位置不同，預覽時站內連結失效。 | 接受；預覽只檢查頁內錨點與外部連結。 |
| R13 | 筆記可能更新 | 平行撰寫的 `ripo-*` 筆記若改 slug，6.5、6.6 需同步。 | 建置時重新讀取 `notes/` 與 `notes_split/`，不存在的 slug 在建置回報中列出。 |
| R14 | S5 拆節後的編號 | S5a／S5b 在目錄顯示「05」「05b」，其餘章節編號不變，避免改動所有交叉引用。 | 接受。 |
| R15 | 審查意見的處理 | round 1 的 13 項 P1、10 項 P2 全部採納，沒有駁回。部分採納的細節：二元量測欄位保留，但移到 W1 進階檢視並在 S4 callout 先教（審查也允許此選項）；「trust region」保留英文術語（它不是比喻，第一次出現時在 S1 給定義），只刪除「消耗」「預算」等比喻用語。 | — |

---

## 8. 修訂紀錄（round 1）

| 審查項 | 變更 |
| --- | --- |
| P1-1 | Hero 卡 2 改為「以 Eq. 9 的單 token 項衡量」；1.1、步驟 5 寫明 Sec. 3.2 是單 token 項；S4 正文加 Eq. 7 vs Eq. 9 一句；步驟 5 補二元量測下的 400 倍、1000 倍。 |
| P1-2 | 「消耗相同距離」全部改為「距離上限相同」；步驟 6 寫明 Eq. 9 為不等式、只有邊界達到、min 只有一側起作用；LO6、LO7 加「邊界上」。 |
| P1-3 | 步驟 5、6 把 Prop. 3.1、4.1 拆成「已推導」與「實驗觀察／定性論證」，寫明論文沒有證明；1.1 改為「作者主張」。 |
| P1-4 | Hero 副標與 1.1 加上 δ 的限定（δ = 0.02 例子 vs δ = 0.05 預設）；副標改用「距離上限較小／較大」。 |
| P1-5 | S1 只教 $s,a,\pi,r,\hat A$、clip 與 PPO 式，加 trust region 一句定義；GRPO Eq. 2 移到 S7，TRPO Eq. 1 移到 S3，Fisher 只在 S3。 |
| P1-6 | Hero 卡 3 不再出現 δ 與公式；S2 加「比值超過 1.2 後梯度為 0」並照引「maximum possible updated probability」，步驟 2、3 寫明 clip 讓梯度為 0、不是硬性上限；LO1、LO2 同步。 |
| P1-7 | S3 加一維直覺、「度量」定義；Eq. 5–6、½χ²、筆誤註腳收合；Eq. 6 假設寫進正文；LO3 標「知道層級」、LO4 加假設；$F$ 的期望標明 $a\sim\pi_{old}$（步驟 4、符號表）。 |
| P1-8 | S5 拆成 S5a（推導、LO6–7、W1 基本檢視）與 S5b（δ = 0.05、LO8 寫成正文、提示操作 W1 進階）；W1 新增基本／進階檢視，DAPO 與二元欄位只在進階。 |
| P1-9 | 步驟 7 加先備 (a)–(e)（不偏、截斷造成的偏差、$v$ 的意義、略去平方均值、「同質」白話）；偏差式寫進正文與 LO9；加「O(δ) 是單一樣本、總和隨尾部 token 數增加」。 |
| P1-10 | W2 的 $r$ 滑桿上界改為 $\min(1000,1/\pi_{old})$，$\pi_{old}$ 改變時保留 $r$ 並夾到上界，提示 3 秒並寫入 live region 一次。 |
| P1-11 | W1 二元 KL 改 log1p 式並定義 $\pi_{max}=1$ 的 ∞ 分支；下圖縱軸改 $[10^{-6},0.2]$，下緣截斷畫小三角（上緣最大 0.1125 不會超出）；W2 IS 曲線的 $\pi_	heta$ 定義為目前滑桿值；W3 讀數表加「平均略過的迭代數」。`check_w1.py` 改用 log1p（輸出不變）並加邊界情況。 |
| P1-12 | W3 改為 idle 後啟動一次、之後只由按鈕；`runToken` 防止舊結果覆蓋；四態狀態機；live region 只在 25/50/75/100%；寫明儲存量。 |
| P1-13 | S7 在 W3 前用白話定義 bandit、softmax logits、伯努利、seed；新增 LO13；`sim.js`、`check_w3.py` 加「不 clip」分支與 skipped 計數並重跑（逐位一致）；結論改為「玩具中增益主要來自放寬上界，不支持形狀本身」，1.7 (e)、R5、R9 同步。 |
| P2-14 | 步驟 4 三種量測分別命名（單 token 0.01、二元二階 0.0101、二元 KL 0.00543）。 |
| P2-15 | LO11、步驟 10、常見錯誤 5 改為「$\delta_{low}/\delta_{high}\ge2$ 的三組」，註明 0.05/0.04 與單次訓練。 |
| P2-16 | 1.3 改為 5 個基準演算法並註明原文「six」「seven」。 |
| P2-17 | 步驟 10 與 W4 加 +37.2% vs +37.5% 註腳。 |
| P2-18 | S10「可以疊加」標〔本頁推導〕；Score Centering、DPPO、GPPO 的描述標明出處論文。 |
| P2-19 | Fig. 1(d) 改為「被 clip 的 token 多出數個數量級」。 |
| P2-20 | 刪除「消耗」「預算」「爆炸」「發散」「太緊／太鬆」；3 節開頭列出替換規則。 |
| P2-21 | 4.0：< 560px 讀數表轉置＋sticky 首欄，圖內註記改圖下圖例；6.3：`.eq` 橫捲、Eq. 15 窄螢幕換行。 |
| P2-22 | 4.0：SVG 只用 Unicode／tspan；6.3：`renderMath` 初始化一次、details／分頁後局部重算；selfTest 檢查無 `.katex-error`。 |
| P2-23 | 4.0：各系列線型＋標記（IS 叉形、不 clip 菱形）、對數滑桿 aria-valuetext 與方向鍵 10／PageUp 100、reduced-motion 無 transition；W4 RIPO 加「（本文）」；Hero 機率軸加 `aria-label`；6.1 寫出 `--c-dapo` 與 `--ink-3` 的實際對比值。 |

## 9. 修訂紀錄（readiness 後）

- W4 補「讀者應觀察到」三點；Hero 卡 2 改白話並連到第 4 節；S10 刪除 TV／χ²／估計器分類術語，改寫各方法的比值範圍；IS 第一次出現展開；`verify/sim_js.out` 路徑統一；5.4 加容許誤差；Hero 標籤分層演算法內嵌；6.7 規定所有本頁推導標籤上頁、R2 註記逐字使用。


## 10. 建置紀錄（build round）

- 產出 `build/index.html`（約 114 KB，單檔）；原始片段在 `build/src/`（`body.html`、`extra.css`、`app.js`，由 `assemble.py` 與 benchmark 外殼、`notes/timeline.json` 組合）。W3 程式為 `verify/sim.js` 原樣貼入。
- 與設計的差異：非檢查題的可收合段落用 `details.more`（`details.check` 只用於檢查題，供擷取腳本剝除）；W1 基本檢視固定 ε = 0.2、δ = 0.02，關閉進階時滑桿重設；曲線右端標籤以 `placeLabels` 避免重疊；W1 曲線只畫到 π_old = 0.99（與滑桿上限一致）；W4 分頁列加 4px 內距（impeccable cramped-padding）。
- 驗收：`?test` 24/24 通過（390 與 1440 寬，含無水平捲動與無 `.katex-error`）；`accept/page.txt` 逐條核對 LO1–LO13 的關鍵數字與用語皆在正文；impeccable 0 項（只有 numbered-section-labels 建議）。
- 環境限制：headless Chrome 在 `--virtual-time-budget` 下載入 Google Fonts 或 cdnjs KaTeX 會停住，因此預覽與 selfTest 使用去掉 Google Fonts、KaTeX 指向本機 0.16.9 的暫存副本；截圖（`build/caps/`）因此使用系統 fallback 字型。
