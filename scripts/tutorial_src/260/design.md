# 設計文件：〈Reinforcement Learning via Self-Distillation〉互動教學

- 論文：Jonas Hübotter、Frederike Lübeck、Lejs Behric、Anton Baumann 等（ETH Zurich、MPI-IS、MIT、Stanford），arXiv 2601.20802（v2，2026-02-16）。程式碼 `https://github.com/lasgroup/SDPO`。資料庫編號 No. 260。
- 原文：`tut/260/src/paper.txt`（PDF 同目錄）。本文件的「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」皆指論文原文編號。
- 產出：`tut/260/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/rl-via-self-distillation.html`（page slug `rl-via-self-distillation`），所有站內相對連結以上線位置為準。
- 數值驗證：`tut/260/verify/check.py`（輸出存於 `check.out`），與頁面使用相同算法；第 5 節驗收表全部取自該輸出。
- 研究筆記：`tut/260/notes/`（7 篇 + `timeline.json`，已產生；沒有 `index.json`，閱讀順序取 hub 筆記 `sdpo-lineage-overview` 的編號清單）。slug 與分類見 6.5–6.6。
- 標記慣例：本設計自行推導、論文沒有寫出的結論標「〔本頁推導〕」；本頁為教學而**自行構造**的玩具數值標「〔示意〕」（不是論文數據）；用論文數字反推的設定標「推定」。頁面使用同樣的標籤。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

以可驗證獎勵做強化學習（RLVR，例如程式題的單元測試通過與否）時，模型每次嘗試只得到一個純量獎勵，GRPO 等方法把同一個優勢值套到整個回答的每個 token，且當一組 rollout 獎勵全相同時優勢全為 0、學習停止。許多環境其實會回傳文字回饋（執行錯誤、未通過的測試、評審意見），本文把這個設定稱為「以豐富回饋做強化學習」（RLRF）。作者提出 **SDPO**（Self-Distillation Policy Optimization）：把「同一個模型、在提示中加上回饋」當作**自我教師** $\pi_\theta(\cdot\mid x,f)$，以逐 token 的 KL 散度把自我教師的下一個 token 分布蒸餾回不看回饋的學生 $\pi_\theta(\cdot\mid x)$（Eq. 1）。其梯度等於優勢為 $\log\frac{\pi_\theta(\hat y_t\mid x,f,y_{<t})}{\pi_\theta(\hat y_t\mid x,y_{<t})}$ 的 logit 層級策略梯度（Prop. 2.1、Sec. 2.1），因此只要在既有 RLVR 管線中替換優勢即可實作。在沒有文字回饋的科學問答與工具使用上，SDPO 把同組成功的 rollout 當回饋，整體最終正確率 70.2% 對 GRPO 66.6%（Sec. 1），回答長度平均短 3 倍以上（Table 8）；在 LiveCodeBench v6 上 48.8% 對 41.2%，並以 4 倍少的生成次數達到 GRPO 的最終正確率（Sec. 4、Fig. 1）；在測試時對單一難題反覆自我蒸餾：在 9 題 very hard 題上達到 22% 發現機率所需的生成數比 best-of-k 與多輪對話少約 3 倍，在 19 題 hard 題上達到 67% 時少約 2.4 倍；以首次成功的平均生成數計，加速是 1.3 倍（hard）與 1.2 倍（very hard）（Sec. 5.2、Fig. 13、Table 10）。

### 1.2 研究動機（Sec. 1）

1. 作者認為 RLVR 的限制在於**資訊量**，不在 RL 本身：純量獎勵只有一個數字，環境狀態（例如錯誤訊息）沒有傳回模型（Fig. 2 說明文字稱為 information bottleneck）。
2. GRPO 的優勢 $r_i-\mathrm{mean}\{r_i\}$ 在整個回答上是常數；全組獎勵相同時優勢為 0，學習停止（Sec. 1）。
3. 從強教師蒸餾可提供逐 token 的稠密監督，但線上學習的目標是超越現有模型，通常沒有更強的教師（Sec. 1）。
4. LLM 已具備上下文學習（in-context learning）：看到回饋後常能指出錯誤（Fig. 3 的 LeetCode 式回饋）。作者把這個能力用來做信用分配（credit assignment），而不是重新取樣一個新答案（Sec. 1）。

### 1.3 貢獻

1. 形式化 RLRF：回饋可以是代理系統所到達任何狀態的 token 化表示（Sec. 1、Fig. 2）。
2. SDPO 演算法（Eq. 1、Alg. 1、Table 2 模板）與其梯度（Prop. 2.1，Eq. 2），以及與 GRPO 的對照：只換優勢（Sec. 2.1）。
3. 實作：top-K 加尾項的近似蒸餾（Eq. 11）、正規化教師（EMA 或信任域，Eq. 9–10）、Jensen–Shannon 散度（Sec. 2.3）、離策略的 logit 層級 PPO 損失（Eq. 13）。
4. 三組實驗：無豐富回饋（Sec. 3）、有豐富回饋（Sec. 4，含規模、信用分配粒度、教師演進、遺忘、與 GRPO 結合、回饋種類消融）、測試時自我蒸餾（Sec. 5，discovery@k）。
5. 附帶觀點：SDPO 可視為以自我教師定義隱含獎勵的最大熵 RL（App. C.1，Eq. 17–19）。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S1 末尾以可收合表格提供同一份內容）

| 符號 | 意義 |
| --- | --- |
| $x$ | 問題（prompt） |
| $y=(y_1,\dots,y_{\lvert y\rvert})$ | 模型的回答（一個 rollout），$y_t$ 為第 $t$ 個 token，$y_{<t}$ 為其前綴；App. A 以 $T$ 表示長度 |
| $V$、$\hat y_t$ | 詞彙表；位置 $t$ 上任一候選 token（不一定是實際取樣的 $y_t$） |
| $\pi_\theta(\cdot\mid x,y_{<t})$ | 學生：目前的策略（語言模型），參數 $\theta$ |
| $f$ | 豐富回饋：環境輸出（如執行錯誤）及／或同組中成功的解答 |
| $\pi_\theta(\cdot\mid x,f,y_{<t})$、$q_\theta$ | 自我教師：同一組參數，提示中多了 $f$；App. A 記為 $q_\theta(\cdot\mid x,f):=\pi_\theta(\cdot\mid\mathrm{reprompt}(x,f))$ |
| $r$、$G$ | 純量獎勵（常為 0/1）；每題的 rollout 數（群組大小） |
| $J(\theta)$ | 目標：期望獎勵 $J(\theta)=\mathbb E_{x,\,y\sim\pi_\theta}[r]$（S2 首次定義） |
| $A_{i,t}$ | 第 $i$ 個 rollout、位置 $t$ 的優勢（advantage） |
| $K$ | top-K 蒸餾保留的 token 數（Table 12：100 或 20） |
| $\alpha$ | 教師正規化的更新率（EMA 或信任域，Eq. 10）；與 Eq. 3 的 $\lambda$、Eq. 17 的 $\lambda$ 無關 |
| $\lambda$ | Eq. 3：GRPO 與 SDPO 優勢的混合權重（$\lambda=0.9$，Fig. 11）；Eq. 17 另有一個熵係數 $\lambda$，頁面寫成 $\lambda_{\mathrm{ent}}$ 以區分 |
| $k$ | Sec. 5 的嘗試次數（discovery@k、pass@k）。頁面不再用 $k$ 表示其他東西 |

#### 步驟 1：RLVR 與 RLRF（Sec. 1，Fig. 2–3）

- RLVR：給定 $x$，取樣 $y\sim\pi_\theta(\cdot\mid x)$，得到純量 $r\in\mathbb R$，常為二元（單元測試通過／失敗）。
- RLRF：環境回傳 token 化的回饋 $f$，可同時包含獎勵與狀態觀察（執行錯誤、LLM 評審回饋）。Fig. 2 例：火星氣候探測者號問題答「100 lbf-s」，RLVR 只得 −1.0，RLRF 得到「Please answer in Newton-seconds.」。
- App. C 補充：從經典 RL 看，這些領域是部分可觀測的（POMDP）；RLVR／RLHF 丟掉了觀察通道。
- 為什麼需要下一步：要把回饋變成信用分配，先要知道 RLVR 現在怎麼分配。

#### 步驟 2：GRPO 的優勢（Sec. 2.1，footnote 3）

$$A^{\mathrm{GRPO}}_{i,t}:=r_i-\mathrm{mean}\{r_i\}_{i=1}^G\quad(\text{與 }t\text{ 無關})$$

- 本文的 GRPO 優勢**不除以標準差**（footnote 3，沿用 Liu et al. 2025b）。
- 優勢只施加在實際取樣的 token $y_{i,t}$，且在同一 rollout 內為常數。
- 全組 $r_i$ 相同 → 所有優勢為 0 → 沒有梯度（Sec. 1）。
- 〔本頁推導〕若每個 rollout 獨立、成功機率皆為 $p$，一組 $G$ 個 rollout 優勢全為 0 的機率為 $(1-p)^G+p^G$。$G=8$：$p=0.1$ 時 0.4305，$p=0.5$ 時 0.0078。每批 32 題（Table 12 question batch size）中，$p=0.1$ 時期望約 $32\times0.4305=13.8$ 題沒有梯度。這是獨立同機率的簡化，頁面須註明。

#### 步驟 3：自我教師與模板（Sec. 2，Table 2，Fig. 4）

- 核心觀察（Sec. 2 原文框）：同一個策略扮演兩個角色——初次作答的學生，與事後判斷動作價值的教師。
- 教師看到更多資訊，因此「應該」比學生準確（Sec. 2）。
- Table 2 模板（User 段落依序）：`prompt`；「Correct solution: successful_previous_rollout」（同組有成功解才放，否則略過）；「The following is feedback from your unsuccessful earlier attempt: environment_output」（原嘗試失敗且沒有成功解時才放，否則略過）；「Correctly solve the original question.」。Assistant 段是 `original_response`：學生原本的回答，**只用來在教師條件下重新計算 log 機率**，教師不重新生成。若原嘗試本身成功，就把它當作 correct solution 傳入。
- 不需重新取樣：只是多一次前向傳播計算原回答在教師上下文下的 log 機率（Sec. 1）。
- Fig. 4 例（Qwen3-8B）：問題「Write a python function that returns all numbers from 1 to n. Answer briefly.」，回答 `return list(range(1, n + 1))`，回饋「Don't include n.」。圖中逐 token 顯示 $\log(P_{\text{teacher}}/P_{\text{student}})$：紅色（負）集中在少數位置，其他接近 0；作者強調教師在沒有給出解答的情況下，只靠回顧就定位錯誤，且激活是稀疏的。
- 頁面 C2 用一個**自行構造**的同題玩具表（〔示意〕，見 4.2）重現這個現象，必須在圖版上寫明「數值為本頁構造，不是論文 Fig. 4 的數據」。

#### 步驟 4：SDPO 損失與 stopgrad（Eq. 1，Alg. 1）

$$\mathcal L_{\mathrm{SDPO}}(\theta):=\sum_t \mathrm{KL}\big(\pi_\theta(\cdot\mid x,y_{<t})\,\big\|\,\mathrm{stopgrad}(\pi_\theta(\cdot\mid x,f,y_{<t}))\big)\tag{Eq. 1}$$

- $\mathrm{KL}(p\|q)=\sum_i p(i)\log\frac{p(i)}{q(i)}$。學生在第一個位置（reverse KL 方向：對學生分布取期望）。
- stopgrad 阻止梯度流入教師；否則教師分布也會被拉向學生分布，訓練訊號不再反映 $f$（Sec. 2）。
- Alg. 1：重複 {取題 $x$；取樣 $G$ 個回答；評估得回饋 $f_i$；計算教師 log 機率 $\log\pi_\theta(y_{i,t}\mid x,f_i,y_{i,<t})$；對 $\mathcal L_{\mathrm{SDPO}}$ 做梯度下降}。
- Fig. 14 虛擬碼：`logprobs_student=compute_log_prob(batch)`（T×V）；`teacher_batch=reprompt(batch, teacher_context)`；`logprobs_teacher=compute_log_prob(teacher_batch).detach()`；`per_token_loss=divergence(...)`；以 `token-mean` 聚合。`divergence` 可為 reverse-KL、forward-KL 或 Jensen–Shannon。

#### 步驟 5：梯度＝以自我教師估計優勢的策略梯度（Prop. 2.1，Eq. 2，App. B.1）

$$\nabla\mathcal L_{\mathrm{SDPO}}(\theta)=\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\Big[\sum_{t=1}^{\lvert y\rvert}\mathbb E_{\hat y_t\sim\pi_\theta(\cdot\mid x,y_{<t})}\Big[\log\frac{\pi_\theta(\hat y_t\mid x,y_{<t})}{\pi_\theta(\hat y_t\mid x,f,y_{<t})}\cdot\nabla_\theta\log\pi_\theta(\hat y_t\mid x,y_{<t})\Big]\Big]\tag{Eq. 2}$$

- 證明要點（App. B.1）：把 KL 展開為 $-\sum\pi A$，乘積法則得兩項；含 $\nabla A$ 的項用 score trick 化為 $\nabla\sum_{\hat y}\pi=\nabla1=0$。
- 因此負梯度方向是 $\mathbb E[A^{\mathrm{SDPO}}\nabla\log\pi]$，其中
$$A^{\mathrm{SDPO}}_{i,t}(\hat y_{i,t})=\log\frac{\pi_\theta(\hat y_{i,t}\mid x,f_i,y_{i,<t})}{\pi_\theta(\hat y_{i,t}\mid x,y_{i,<t})}$$
  教師較看好的 token 優勢為正、較不看好的為負，兩者完全一致時才為 0（Sec. 2.1）。
- **適用範圍**：Prop. 2.1 只對 Eq. 1 的 reverse KL 證明。Sec. 3 實際用 Jensen–Shannon（Table 12），論文沒有給 JS 的優勢形式；頁面寫到 JS 梯度時一律標〔本頁推導〕。
- 與 GRPO 的兩點差異（Sec. 2.1）：(1) 從 1 位元回饋擴展到任意 token 序列的回饋；(2) 優勢是稠密的 logit 層級（對每個位置的每個候選 token 各一個值）。
- 〔本頁推導〕對單一位置、把學生 logits $z$ 當自由參數時：$-\partial\mathcal L/\partial z_k=\pi_k\big(A_k-\mathbb E_{\pi}[A]\big)$，且 $\mathbb E_\pi[A]=-\mathrm{KL}(\pi\|q)$。也就是說，KL 本身扮演了 baseline。依據：$\nabla_z\log\pi(\hat y)=e_{\hat y}-\pi$ 代入 Eq. 2。C3 會顯示這個等式。
- App. A.1：Eq. 5 的逐 token 估計式（假設產生 $y$ 的取樣分布固定）就是 Prop. 2.1；序列層級 KL（Eq. 7）的梯度多一項 $\mathbb E[\sum_t\mathrm{KL}_t\nabla\log\Pi_\theta(s_t)]$（Eq. 8，Amini et al. 2025），作者試過但「沒有可量測的增益」。頁面以可收合段落呈現。
- App. C.1（可收合）：最大熵 RL $\arg\max\mathbb E[\sum_t r(y_t\mid x,y_{<t})]+\lambda_{\mathrm{ent}}H[\pi_\theta]$（Eq. 17）等價於最小化 $\sum_t\mathrm{KL}(\pi_\theta\|\pi^\star)$（Eq. 19）；取隱含獎勵 $r=\log q(y_t\mid x,f,y_{<t})$、$\lambda_{\mathrm{ent}}=1$ 即得 Eq. 1。作者據此稱 SDPO 為以回顧模型隱含構造稠密獎勵的最大熵 RL，並連到逆強化學習。

#### 步驟 6：讓它在 GPU 上可行（Sec. 2.2，App. A.3–A.4，Fig. 5）

- 唯一額外計算是教師 log 機率，可平行化，遠快於序列生成。Fig. 5（micro batch size 2）：每步時間增加 +17.1%（實心，無程式環境）與 +5.8%（淺色，含程式環境）。〔對照 PDF 圖例確認：實心＝無程式環境、淺色＝含程式環境。〕
- 完整 KL 需同時保留學生與教師的全詞彙 logits。top-K 近似（Eq. 11，top-K **依學生**選）：
$$\mathcal L\approx\sum_t\Big[\sum_{\hat y\in\mathrm{top}_K(\pi)}\pi(\hat y)\log\frac{\pi(\hat y)}{q(\hat y)}+\Big(1-\sum_{\mathrm{top}_K}\pi\Big)\log\frac{1-\sum_{\mathrm{top}_K}\pi}{1-\sum_{\mathrm{top}_K}q}\Big]\tag{Eq. 11}$$
  作者：$K=100$ 時幾乎沒有記憶體負擔且保留大部分資訊（Sec. 2.2）。Table 12：Sec. 3 用 $K=100$，Sec. 4–5 用 $K=20$。
- 〔本頁推導〕Eq. 11 等於把非 top-K 的 token 合併成一個「尾」類別後的精確 KL；由 log-sum 不等式，合併只會讓 KL 變小或不變，所以近似值 ≤ 精確值，且當尾部所有 token 的 $q/\pi$ 比值相同時兩者相等。
- 每個 rollout 有 $\lvert y\rvert\cdot(K+1)$ 個不同的優勢值（Sec. 4.2）。Sec. 4.2 描述 logit 層級時用「學生前 100 個 token」，但 Table 12 的 Sec. 4 主實驗設定是 $K=20$；頁面不寫「主實驗有 $\lvert y\rvert\cdot101$ 個優勢」，只寫 $\lvert y\rvert(K+1)$ 並註明兩個 $K$ 的出處。
- 離策略（App. A.4）：GRPO 基準用 Eq. 12（PPO 截斷、截斷重要性取樣 TIS、clip-higher、固定長度正規化）；SDPO 推廣為 Eq. 13 的 logit 層級損失，對每個候選 token 以 $\pi_{\theta_{old}}$ 機率加權。實驗中 TIS 施加在 token 層級而非 logit 層級。頁面只用一段文字與 Eq. 13 的結構說明，不展開。

#### 步驟 7：穩定性（Sec. 2.3，App. A.2，App. B.2，Table 4）

- 教師在訓練中會變，可自舉也可能不穩定。兩個修正：(1) 正規化教師；(2) 對稱的 Jensen–Shannon 散度（Agarwal et al. 2024 在外部教師的線上蒸餾中也觀察到更穩定）。
- 信任域約束 $\sum_t\mathrm{KL}(q\|q_{\theta_{ref}})\le\epsilon$（Eq. 9）；閉式解（Prop. B.1）
$$q^*(y_t\mid x,f,y_{<t})\propto\exp\big((1-\alpha)\log q_{\theta_{ref}}+\alpha\log q_\theta\big)\tag{Eq. 10, 16}$$
  $\alpha=1/\lambda_L$，$\lambda_L$ 是 Lagrange 乘子（頁面寫成 $\lambda_L$，與 Eq. 3 的 $\lambda$ 區分）。$q_{\theta_{ref}}$ 是**初始**教師，所以 $q^*$ 永遠以固定比例錨定在初始教師上。App. B.2 末：Chen et al. (2025c) 用 $\pi_{\theta_{ref}}$ 當參考，作者觀察到不如 $q_{\theta_{ref}}$。
- EMA：$\theta'\leftarrow(1-\alpha)\theta'+\alpha\theta$。EMA 需額外 GPU 記憶體但沒有執行時間開銷；信任域要多算一次 $q_{\theta_{ref}}$ 的 log 機率但不需額外記憶體（若 $\theta_{ref}$ 本來就用於 KL 正規化）。
- 〔本頁推導〕從 $\theta'_0=\theta_0$ 開始，經過 $n$ 次 **EMA 更新**後初始參數的權重為 $(1-\alpha)^n$：$\alpha=0.01$、$n=80$ 為 0.4475；$\alpha=0.05$、$n=80$ 為 0.0165。論文沒有說 EMA 是每個優化步還是每批生成更新一次。Sec. 4 的 question batch 32、mini batch 1（Table 12）表示一個訓練步最多有 32 次優化步；若每個優化步都更新，第 80 個訓練步時 $n$ 可達 2560，$\alpha=0.01$ 的權重約 $7\times10^{-12}$。頁面的 $n$ 軸一律標「EMA 更新次數 $n$」並附這段說明（R13）。
- Table 4（LCBv6，至第 90 步，3 seeds 標準誤）：$q_\theta$（不正規化）最佳 36.1 ± 1.6／平均 29.8 ± 1.3，最終發散；$q_{\theta_{ref}}$（凍結初始教師）48.8 ± 0.7／44.4 ± 0.2；信任域 50.6 ± 0.9／45.6 ± 0.2；EMA 49.3 ± 0.3／45.3 ± 0.2（後兩者 $\alpha=0.01$）。作者：信任域與 EMA 勝過凍結教師，表示教師透過與學生共享參數而進步；但凍結教師也表現不錯。

#### 步驟 8：沒有豐富回饋的環境（Sec. 3，Table 3、7、8，Fig. 6–7，App. E.2.1）

- 只有純量獎勵時，把同批次同題的成功 rollout 當作失敗 rollout 的「回饋」。
- 任務：SciKnowEval L3 推理子集（化學、物理、生物、材料）與 ToolAlpaca 工具呼叫；模型 Qwen3-8B、Olmo3-7B-Instruct；報告 avg@16 對牆鐘時間（不含初始化與驗證）；每次實驗 4 張 GH200，約 6 小時。
- 基準：改良版 GRPO（非對稱截斷、不做有偏正規化、修正推論框架造成的離策略偏差），每批生成做 4 個離策略 mini batch 步；另報 on-policy GRPO。Sec. 3 的 SDPO 嚴格 on-policy（每批生成一個梯度步；Table 12 Sec. 3 的 mini batch 32＝question batch 32）。Sec. 4–5 的 mini batch 是 1，這句話不適用於那兩節。
- Table 3（1h／5h 最高 avg@16）摘錄：Qwen3-8B 化學 GRPO 65.9／74.5、SDPO 73.2／80.9；Olmo3-7B-Instruct 化學 GRPO 39.7／56.7、SDPO 68.0／80.0。Sec. 3.2 寫 SDPO 勝過 GRPO 的是「almost all runs」：5h 欄有三格 GRPO 較高——Qwen3-8B 生物 59.9 > 56.8、Olmo3 生物 55.8 > 52.8、Olmo3 工具使用 65.0 > 62.1。
- Olmo3 化學：SDPO 在 50 分鐘達到 GRPO 5h 正確率（6× 加速）；5h 正確率高出 10 個百分點以上；回答長度縮短 11×（Fig. 6）。
- Table 8（on-policy，各任務平均長度）：Qwen3-8B GRPO 820.8 對 SDPO 255.8（3.2×）；Olmo3 1095.4 對 343.9（3.2×）。
- Fig. 7：同一題 GRPO 5,549 tokens（25 次 “Wait”、9 次 “No.”、5 次 “Hmm.”，「101.85 ≈ 69.3」重複四次，並寫出 “Wait I’m going in circles”）答錯 B；SDPO 764 tokens 答對 C。作者認為與稠密信用分配產生的稀疏優勢有關（Fig. 21）——這是作者的解釋，頁面用「作者認為」。
- 〔本頁計算〕「70.2% vs 66.6%」（Sec. 1）論文未說明如何彙總；Table 3 十個 5h 欄位的算術平均為 SDPO 70.0、GRPO 66.8（`check.py`）。頁面照錄 70.2／66.6 並在註腳寫明這個差異（R2）。

#### 步驟 9：有豐富回饋的程式題（Sec. 4，Fig. 1、8、10、11，Table 4–6、9）

- LCBv6：131 題（2025 年 2–5 月）。公開測試（私有測試的 50% 隨機子集）用於訓練時的回饋，私有測試用於驗證；報告 4 個 rollout 的平均正確率；預設 Qwen3-8B。
- 主結果：SDPO 48.8% 對 GRPO 41.2%；公開排行榜上最強的 instruct 模型 Claude Sonnet 4 為 40.5%、Claude Opus 4 為 39.7%；SDPO 以 4× 少的生成次數達到 GRPO 最終正確率（Fig. 1）。Table 9（第 80 步，3 seeds 標準差）：GRPO 41.2 ± 0.8、GSPO 40.1 ± 2.3、CISPO 41.2 ± 1.8、SDPO 48.8 ± 0.6。依難度（Fig. 15）SDPO 在中、難題的提升較大（易／中／難：GRPO 90.1%／37.8%／11.6%，SDPO 91.7%／46.6%／18.9%，圖中標示的相對提升為 +2%／+23%／+63%）。
- 規模（Fig. 8、17）：Qwen3 0.6B→8B，模型越大 SDPO 領先越多；Qwen2.5-Instruct 在 7B 勝過 GRPO、在「8B」相近、在 1.5B 落後 GRPO（照錄 Sec. 4.1 原文，見 R3）。作者：自我教師的回顧能力似乎隨規模湧現。
- 信用分配粒度（Sec. 4.2，Fig. 10 左）：logit 層級（Sec. 4.2 寫「學生前 100 個 token」；Sec. 4 主實驗的 Table 12 設定為 $K=20$）> token 層級（每個位置最可能的 token）> 序列層級（把 SDPO 優勢平均成一個純量，像 GRPO 一樣）> GRPO。即使序列層級也勝過 GRPO → 豐富回饋本身就有用，稠密信用分配再加分。
- 教師演進（Sec. 4.3，Fig. 10 右）：自我教師的生成正確率在訓練中顯著上升；學生後期超過**初始**教師 → 自舉（bootstrapping）。
- 遺忘（Sec. 4.4，Table 5）：LCBv6／holdout 平均：Base 27.9／43.5；SFT on self-teacher 42.7／41.4；GRPO 41.2／41.8；SDPO 48.8／42.4。SFT 基準需要 2× 生成次數。
- 與 GRPO 結合（Sec. 4.5，Eq. 3）：$A^{\mathrm{SDPO+GRPO}}=\lambda A^{\mathrm{GRPO}}+(1-\lambda)A^{\mathrm{SDPO}}$，$\lambda=0.9$。GRPO 優勢對 $J(\theta)$ 無偏（Monte Carlo），SDPO 優勢有偏但通常變異較低（類比 bootstrap）。結果：在 Qwen3-0.6B 上明顯勝過 SDPO，在較強模型上略遜（Fig. 11）。
- 回饋種類（Sec. 4.6，Table 6，訓練至第 60 步）：

| $f$ | 訓練前教師正確率 | 相同輸出比例 ↓ | SDPO 學生正確率 | 平均熵 |
| --- | --- | --- | --- | --- |
| output | 32.5 ± 0.5 | 13.7 ± 0.6 | 39.9 ± 1.1 | 0.40 |
| own solution | 42.4 ± 1.0 | 12.1 ± 0.7 | 42.6 ± 1.3 | 0.41 |
| output + own solution | 42.5 ± 1.2 | 10.1 ± 0.2 | 48.3 ± 1.4 | 0.38 |
| $y$ + output + own solution | 39.3 ± 0.8 | 30.0 ± 0.9 | 44.5 ± 1.3 | 0.23 |

  環境輸出與成功解互補；把學生原嘗試 $y$ 放進 $f$ 會讓教師偏向學生、降低熵與探索。對模板的語法變化不敏感。

#### 步驟 10：測試時自我蒸餾（Sec. 5，Eq. 4，Fig. 12–13，Table 10–11）

- 定義（Def. 5.1）：發現時間＝第一次得到獎勵 1 的嘗試序號；$\mathrm{discovery@}k:=P(\text{發現時間}\le k)$（Eq. 4）。對固定模型的獨立取樣（best-of-k），它等於 pass@k。
- 作者論證（Sec. 5、footnote 9，沒有實驗量測）：在單一題目上做 RLVR 不會勝過 best-of-k，因為二元獎勵在第一次成功前沒有訊號。SDPO 每次嘗試都得到回饋，可在找到解之前就修正；並把上下文 $c=(y_k,f_k)$ 壓縮進權重（把 $\pi_\theta(\cdot\mid x,c)$ 蒸餾成 $\pi_{\theta'}(\cdot\mid x)$，Fig. 12），不受上下文長度限制。
- 設定：Qwen3-8B 的 pass@64 < 0.5 為 hard、< 0.03 為 very hard；保留任一方法在 512 步、5 seeds 內至少解出一次的題目 → 19 題 hard（其中 9 題 very hard，Table 10 以 * 標示）。best-of-k 用 2944 個獨立 rollout 的無偏 pass@k 估計（Chen et al. 2021b）；多輪基準以 FIFO 滑動視窗保留過去回饋（最大 prompt 32k，模型上限 40k）；SDPO 每次更新用 question batch 1、16 個 rollout（Table 12 Sec. 5：學習率 1e-6、$K=20$、reverse-KL、優勢截斷 5.0、EMA 0.01）。
- 〔本頁推導〕pass@64 < 0.03 等價於單次成功率 $p<1-0.97^{1/64}=4.76\times10^{-4}$，在 2944 次取樣中期望成功約 1.4 次；pass@64 < 0.5 等價於 $p<1.08\times10^{-2}$（期望約 31.7 次）。
- 結果（Sec. 5.2）：very hard 題 discovery@2750：多輪 35.6%、best-of-k 41.5%、SDPO 53.2%；very hard 題達到 22% 發現機率所需生成數 SDPO 少約 3×（只有這個題組與分位是 3×）。hard 題 SDPO discovery@2750 為 78%，達到 67% 時少約 2.4×。多輪與 best-of-k 分別只解出 68.4% 與 72.3% 的題目。多輪的上下文在 hard 題第 837（±466）步、very hard 題第 1007（±349）步用盡。Q3 只有 SDPO 解出：Sec. 5.2 寫 SDPO 最早在第 321 次嘗試找到解（20 次自我蒸餾更新、每次 16 個 rollout；論文沒說是哪個 seed）。Table 10 中 Q3 的平均首次成功（5 seeds 平均）是 SDPO 1987、best-of-k ≥ 2750、加速 1.4×。頁面兩個數字並列並分別標「最早一次」「平均」。
- Table 10（首次成功的平均生成數，截斷在 2750）：Hard 894／1145／1141（SDPO／best-of-k／多輪，1.3×；這一列是全部 19 題，包含 9 題 very hard，894 正好是 SDPO 欄 19 個值的平均）；Very hard 1739／2180／2121（1.2×）；單題最高 13.6×（Q120）。
- Table 11：初始自我教師在 19 題中有 15 題正確率恰為 0%（15/19≈79%；論文正文寫「78%」，為截斷值，頁面寫「15 題（約 79%）」），其餘 < 7%（最高 Q43 6.25%）——單輪回饋不足以解題，但其信用分配足以逐步改進策略。
- 〔本頁推導〕best-of-k 的截斷平均首次成功時間為 $\sum_{j=0}^{2749}(1-p)^j=\frac{1-(1-p)^{2750}}{p}$；C5 用它對照 Table 10 的定義。

#### 步驟 11：實作細節（App. E，Table 12–13）

| 項目 | Sec. 3（無回饋） | Sec. 4（有回饋） | Sec. 5（TTT） |
| --- | --- | --- | --- |
| 模型 | Qwen3-8B、Olmo3-7B-Instruct | Qwen3-8B | Qwen3-8B |
| thinking | False | False | False |
| 最大 prompt／回答長度 | 2048／8192 | 同左 | 同左 |
| question batch／mini batch／rollouts | 32／32／8 | 32／1／8 | 1／1／16 |
| 取樣溫度；驗證 rollouts、溫度、top-p | 1.0；16、0.6、0.95 | 1.0；4、0.6、0.95 | 1.0；— |
| top-K 蒸餾 | 100 | 20 | 20 |
| 散度 | Jensen–Shannon | Reverse-KL | Reverse-KL |
| 優勢截斷 | — | — | 5.0 |
| 教師 EMA 更新率 | 0.05 | 0.01 | 0.01 |
| rollout IS clip | 2 | 2 | 2 |
| AdamW 學習率（constant）／warmup | 1e-5／10 | 1e-6／0 | 1e-6／0 |
| weight decay／梯度裁剪 | 0.01／1.0 | 同左 | 同左 |

GRPO（Table 13，Sec. 3）：mini batch 8（預設）／32（on-policy），ε-high 0.28，KL 係數 0，學習率 1e-6（預設）／1e-5（on-policy），warmup 10。超參數搜尋（App. E.2.1）：學習率 {1e-5, 1e-6}、mini batch {8, 32}；SDPO 另搜 forward KL 與 Jensen–Shannon。環境：verl、FSDP2、vLLM、PyTorch 2.7.0、CUDA 12.8。

常見錯誤（頁面 S10 逐條列出）：
1. 忘記對教師 stopgrad：教師分布被拉向學生分布，訊號不再反映回饋（Sec. 2）。
2. 把學生原嘗試 $y$ 放進回饋段落：Table 6 顯示熵降到 0.23、學生 44.5（低於 48.3）。注意：模板 Assistant 段的 `original_response` 是被評分的對象，必須保留；被消融的是「回饋中也放入 $y$」。
3. 不正規化教師：Table 4 中 $q_\theta$ 最終發散（36.1）。
4. 同時保留兩份全詞彙 logits：記憶體翻倍；用 Eq. 11 的 top-K（依學生選）加尾項，尾項不能省。
5. 在弱模型上期待增益：Qwen2.5-1.5B 落後 GRPO；可改用 SDPO+GRPO（$\lambda=0.9$）。
6. 散度方向與種類：Eq. 1 是 $\mathrm{KL}(\text{學生}\|\text{教師})$；Table 12 中 Sec. 3 實際用 JS、Sec. 4–5 用 reverse-KL。
7. 基準的 GRPO 優勢不除以標準差（footnote 3）；離策略修正的 TIS 在 SDPO 中施加於 token 層級（App. A.4）。

#### 步驟 12：限制與未來方向（Sec. 7）

依賴模型的上下文學習能力（弱模型可能不如 GRPO）；依賴回饋品質（無資訊或誤導的回饋無法學習）；小模型、短生成時額外的教師 log 機率計算佔比較大。未來：長時程與代理環境、大規模多任務訓練、沒有驗證器的文字回饋任務、系統性研究推理行為差異（如模板的影響）。

---

## 2. 學習目標（驗收測試）

每個章節結尾以 `<details>`（summary「檢查理解」）呈現該章檢查題與參考答案。**每條參考答案的內容必須出現在指定章節的正文段落中**，不能只出現在元件讀數、表格、圖或答案框裡；Goldfish 審查逐條檢查。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 區分 RLVR 與 RLRF | RLVR 與 RLRF 的環境各回傳什麼？作者認為 RLVR 的主要限制是什麼？ | RLVR 回傳純量獎勵 $r$（常為二元）；RLRF 回傳 token 化回饋 $f$（如執行錯誤、評審意見），可同時包含獎勵與狀態；作者認為限制在於純量獎勵只含一個數字、沒有傳回環境狀態，不在 RL 本身（Sec. 1、Fig. 2） | S1 |
| LO2 | 計算 GRPO 優勢並說明何時優勢全為 0 | 本文 GRPO 優勢怎麼算？一組 rollout 獎勵全相同時會怎樣？$G=8$、每次成功率 0.1 時這種情況的機率？ | $A_{i,t}=r_i-\mathrm{mean}\{r_i\}$，不除以標準差，同一 rollout 內與 $t$ 無關；全相同時優勢全為 0、無梯度；機率 $(1-p)^G+p^G=0.4305$〔本頁推導，假設獨立同機率〕（Sec. 1、2.1、footnote 3） | S2 |
| LO3 | 描述自我教師與模板 | 自我教師是什麼？Table 2 模板中的回饋段落何時放入？`original_response` 的作用？ | 同一組參數的 $\pi_\theta(\cdot\mid x,f)$；有同組成功解時放「Correct solution」，原嘗試失敗且無成功解時放環境輸出；`original_response` 是學生原回答，只用來在教師條件下重算 log 機率，不重新生成（Sec. 2、Table 2） | S3 |
| LO4 | 寫出 SDPO 損失並說明 stopgrad | Eq. 1 是什麼？為什麼要 stopgrad？ | $\sum_t\mathrm{KL}(\pi_\theta(\cdot\mid x,y_{<t})\|\mathrm{stopgrad}(\pi_\theta(\cdot\mid x,f,y_{<t})))$；阻止梯度流入教師，避免教師分布被拉向學生分布而不再反映 $f$（Sec. 2） | S4a |
| LO5 | 讀懂 SDPO 優勢 | SDPO 的優勢是什麼？何時為 0、為正、為負？和 GRPO 差在哪兩點？ | $A^{\mathrm{SDPO}}=\log\frac{\pi_\theta(\hat y\mid x,f,y_{<t})}{\pi_\theta(\hat y\mid x,y_{<t})}$；學生與教師完全一致時為 0，教師較看好為正、較不看好為負；回饋從 1 位元擴展到任意 token 序列，優勢變成稠密的 logit 層級；此形式只對 reverse KL 證明（Prop. 2.1、Sec. 2.1） | S4b |
| LO6 | 說明 top-K 近似 | Eq. 11 如何近似 KL？top-K 依誰選？論文用多少 K？近似值與精確值的大小關係？ | 只對學生的前 K 個 token 計算 KL 項，其餘合併成一個尾項；依學生選；Sec. 3 用 100、Sec. 4–5 用 20；近似值 ≤ 精確值，尾部 $q/\pi$ 比值皆相同時相等〔本頁推導〕（App. A.3、Table 12） | S5 |
| LO7 | 說明教師正規化 | 兩種正規化教師怎麼定義？Table 4 中不正規化的教師結果如何？ | 信任域 $q^*\propto\exp((1-\alpha)\log q_{\theta_{ref}}+\alpha\log q_\theta)$，$q_{\theta_{ref}}$ 為初始教師；EMA $\theta'\leftarrow(1-\alpha)\theta'+\alpha\theta$；不正規化最佳 36.1、最終發散，信任域 50.6、EMA 49.3、凍結 48.8（App. A.2、Table 4） | S6 |
| LO8 | 說明無回饋環境的做法與結果 | 只有純量獎勵時 SDPO 用什麼當回饋？整體結果與回答長度？ | 同批次同題的成功 rollout；最終正確率 70.2% 對 66.6%；平均回答長度短 3.2×（Table 8），Olmo3 化學 11× 短、6× 加速（Sec. 3） | S7 |
| LO9 | 讀出程式題主結果與規模效應 | LCBv6 上 SDPO 與 GRPO 的最終正確率？模型規模的影響？ | 48.8% 對 41.2%，4× 少的生成數達到 GRPO 最終正確率；模型越大領先越多，Qwen2.5-1.5B 上落後 GRPO（Sec. 4、4.1） | S8a |
| LO10 | 區分三種信用分配粒度 | logit／token／序列層級 SDPO 各是什麼？排序？序列層級勝過 GRPO 代表什麼？ | logit：學生前 K 個 token（Sec. 4.2 寫 100，Sec. 4 主實驗 Table 12 為 20）及尾項；token：每個位置最可能的 token；序列：把 SDPO 優勢平均成單一純量；logit > token > 序列 > GRPO；豐富回饋本身就帶來增益，不只靠稠密信用分配（Sec. 4.2、Fig. 10） | S8b |
| LO11 | 說明回饋種類消融 | 哪種回饋組合最好？把學生原嘗試放進回饋有什麼影響？ | 環境輸出＋成功解最好（學生 48.3）；加入 $y$ 讓教師偏向學生、相同輸出比例升到 30.0%、熵降到 0.23、學生降到 44.5（Table 6） | S8b |
| LO12 | 定義 discovery@k 並讀出測試時結果 | discovery@k 的定義？作者為什麼認為在單題上做 RLVR 不會勝過 best-of-k？pass@64 < 0.03 代表單次成功率小於多少？very hard 題的 discovery@2750？ | $P(\text{發現時間}\le k)$；作者論證（footnote 9，未量測）：二元獎勵在第一次成功前沒有訊號；$p<1-0.97^{1/64}\approx4.76\times10^{-4}$〔本頁推導〕；SDPO 53.2%、best-of-k 41.5%、多輪 35.6%（Def. 5.1、Eq. 4、Sec. 5.2） | S9 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），互動元件放在加寬圖版（最大 1080px）。章節編號 JetBrains Mono（`01 · …`），章節標題用陳述句。全頁視覺記號：**一列 token 方格**（每格一個 token，下方以帶符號長條表示優勢），在 hero、C2、C2b 重複出現；GRPO 一律畫成整列等高長條、SDPO 畫成逐格長條。

章節 id 沿用 S0–S12，S4 與 S8 各拆成 a／b 兩章；頁面顯示編號依序為 01–14（下表「顯示」欄）。術語密度規則：每章正文只引入該章標題需要的術語；證明、baseline 推導、序列層級估計式、最大熵 RL 一律放在可收合段落（`<details>`，summary 以「推導：」開頭），正文不依賴它們。

| # | 顯示 | id | 標題 | 目的 | 長度 |
| --- | --- | --- | --- | --- | --- |
| S0 | — | `top` | Hero：〈用自我蒸餾做強化學習：以加入回饋後的同一模型分布作為逐 token 訓練目標〉 | 副標：「SDPO 把同一個模型在提示中加入回饋後的下一個 token 分布當作目標，蒸餾回不看回饋的模型，得到逐 token 的訓練訊號。」一列 token 方格（Fig. 4 例 `return list(range(1, n + 1))`，GRPO 列整列 −0.375、SDPO 列只有 ` +`、` 1` 明顯為負，標〔示意〕）；方格下方一行：「GRPO：同組 8 次嘗試中 3 次成功，這次失敗，所以每個 token 的優勢都是 0 − 3/8 = −0.375。SDPO：每個 token 各有一個值。」三張事實卡（見下）。連結：arXiv、GitHub、`../index.html#paper-260`、研究脈絡總覽 `sdpo-lineage-overview.html` | 1 螢幕 |
| S1 | 01 | `rlrf` | 純量獎勵只含一個數字，文字回饋還帶有環境狀態 | 語言模型作為策略 $\pi_\theta(y\mid x)=\prod_t\pi_\theta(y_t\mid x,y_{<t})$；rollout、獎勵；RLVR vs RLRF（Fig. 2 兩欄靜態圖，規格見 6.4；Fig. 3 回饋範例以等寬字方塊照錄）；POMDP 一句；符號表（可收合） | 350–450 字 |
| S2 | 02 | `grpo` | GRPO 用群組平均估計優勢 | 段落「$\nabla\log\pi$ 是什麼」（≤120 字，見下）；目標 $J(\theta)=\mathbb E[r]$（期望獎勵，先於梯度式定義）；策略梯度 $\nabla J=\mathbb E[A\nabla\log\pi]$，優勢＝比平均好多少；$A^{\mathrm{GRPO}}$、不除標準差、只作用在取樣 token、整列常數；全組獎勵相同時優勢全為 0 與〔本頁推導〕機率；強教師蒸餾的稠密監督與「沒有更強教師」的矛盾；C1 | 450–550 字 + C1 |
| S3 | 03 | `teacher` | 同一個模型看到回饋後成為自我教師 | Sec. 2 核心觀察框；Table 2 模板（以 User／Assistant 兩欄方塊照錄，四個 placeholder 以 `--accent` 等寬字）；放入規則；不重新生成；Fig. 4 例；C2（只顯示逐 token 的 $\log(q/\pi)$，不顯示 KL） | 400–500 字 + C2 |
| S4a | 04 | `loss` | SDPO 的損失是學生到自我教師的逐 token KL | KL 定義（一句加一個兩 token 數值例）；Eq. 1；stopgrad；Alg. 1（虛擬碼方塊） | 300–350 字 |
| S4b | 05 | `grad` | KL 的梯度就是以自我教師打分的策略梯度 | Prop. 2.1、Eq. 2（正文只寫結論，B.1 證明要點可收合）；$A^{\mathrm{SDPO}}$ 與 GRPO 對照表（兩欄：作用對象、是否隨 $t$ 變、何時為 0、需要什麼）；適用範圍：只對 reverse KL 證明；可收合：〔本頁推導〕baseline 等式、Eq. 5–8 序列層級估計式、App. C.1 最大熵 RL | 300–400 字 |
| S5 | 06 | `practice` | top-K 近似讓 logit 蒸餾放得進記憶體 | Fig. 14 虛擬碼；三種散度的定義（reverse KL、forward KL $\mathrm{KL}(q\|\pi)$、JS $\tfrac12\mathrm{KL}(\pi\|m)+\tfrac12\mathrm{KL}(q\|m)$，$m=\tfrac{\pi+q}2$——JS 的混合比例 1/2 為本頁採用，見 R4；JS 沒有論文給的優勢形式）；Eq. 11，正文明寫一句「近似值 ≤ 精確值，尾部所有 token 的 $q/\pi$ 比值皆相同時兩者相等〔本頁推導〕」（LO6 依賴此句；只有 log-sum 不等式的推導過程可收合）；$\lvert y\rvert(K+1)$ 與兩個 $K$ 的出處；Fig. 5 成本（+17.1%／+5.8%）；離策略 Eq. 13 一段；C3 | 450–550 字 + C3 |
| S6 | 07 | `stable` | 正規化教師讓自舉保持穩定 | 教師會變的利弊；Eq. 9、Eq. 10／Prop. B.1；EMA；兩者的記憶體與時間取捨；〔本頁推導〕EMA 權重 $(1-\alpha)^n$ 與更新次數的說明；Table 4；JS 一句；C4 | 400–500 字 + C4 |
| S7 | 08 | `nofb` | 沒有文字回饋時以成功的 rollout 當回饋 | Sec. 3 設定、基準（「嚴格 on-policy」限定 Sec. 3）；Table 3 摘錄（化學、生物兩列 × 兩模型）並列出 GRPO 較高的三格；6× 加速、11× 短；Table 8；Fig. 7 兩欄摘錄（照錄引文片段與 token 數）；70.2／66.6 註腳 | 400–500 字 + 2 表 |
| S8a | 09 | `code` | 程式題的豐富回饋帶來更大的提升 | LCBv6 設定；主結果與 Claude 排行；Table 9；Fig. 15 難度；規模（Fig. 8、17）；教師演進與自舉；Table 5 遺忘 | 350–450 字 + 2 表 |
| S8b | 10 | `ablate` | 逐 logit 的信用分配與兩種回饋各有貢獻 | 粒度消融（定義三種粒度；$K$ 的出處說明）＋ C2b；Eq. 3 SDPO+GRPO；Table 6 回饋種類 | 350–450 字 + C2b + 1 表 |
| S9 | 11 | `ttt` | 測試時自我蒸餾加速找到第一個解 | Def. 5.1、Eq. 4；best-of-k＝pass@k；作者論證 RLVR 無訊號（footnote 9）；Fig. 12 三列示意（SVG，規格見 6.4）；設定與 hard／very hard 定義；〔本頁推導〕$p$ 門檻；C5；Sec. 5.2 結果（每個倍數附題組與分位：3×＝very hard、22%；2.4×＝hard、67%；平均首次成功 1.3×／1.2×）；Table 10 摘錄（兩列彙總，註明 Hard 列含 very hard；Q3 標「最早一次 321」與「平均 1987」；Q120）；Table 11 | 550–650 字 + C5 |
| S10 | 12 | `impl` | 實作設定與常見錯誤 | Table 12（1.4 步驟 11 全表）+ Table 13 摘要；7 條常見錯誤 | 300–400 字 + 表 |
| S11 | 13 | `history` | 研究脈絡 | 時間線（6.6）+ 一段文字：策略梯度／PPO → GRPO 與 RLVR → 線上蒸餾（強教師）→ 情境蒸餾／自我蒸餾 → 本文 | 時間線 + 120 字 |
| S12 | 14 | `next` | 限制與延伸閱讀 | Sec. 7 限制與未來方向；研究筆記清單（6.5 順序）；站內既有 `test-time-training-naming.html`、`ttt-theory-and-applications.html`（對應 App. C「與測試時訓練的關係」） | 200 字 + 連結 |
| — | — | — | 頁尾 | 引用格式、arXiv、GitHub；`#pm-bar` | — |

S2「$\nabla\log\pi$ 是什麼」段落內容（≤120 字）：模型在一個位置對每個候選 token 輸出分數（logit）$z$，經 softmax 得機率 $\pi$。$\nabla_z\log\pi(\hat y)=e_{\hat y}-\pi$：讓被選 token 的分數上升、其他依機率下降。數值例：三個候選 $\pi=(0.5,0.3,0.2)$，選第 1 個，方向是 $(+0.5,-0.3,-0.2)$。乘上優勢 $A$：$A>0$ 時沿此方向走（提高這個 token 的機率），$A<0$ 時反向走。

S4a KL 數值例：$p=(0.5,0.5)$、$q=(0.9,0.1)$，$\mathrm{KL}(p\|q)=0.5\log\frac{0.5}{0.9}+0.5\log\frac{0.5}{0.1}=0.511$；$p=q$ 時為 0。

Hero 事實卡（不使用未定義術語與縮寫）：
1. 「一般的強化學習訓練中，模型每次嘗試只得到一個分數；程式環境其實還會回傳錯誤訊息。本文把這些文字回饋直接變成訓練訊號。」
2. 「教師就是模型自己：把錯誤訊息放進提示，重新計算原回答每個 token 的機率。在論文的例子中，機率明顯下降的 token 正好是錯誤所在；這是一個例子，不是保證。」
3. 「在 LiveCodeBench v6 上，Qwen3-8B 以 SDPO 訓練達到 48.8%，GRPO 為 41.2%。」

Hero token 方格：11 格，每格下方兩列長條（GRPO、SDPO），長條方向表示正負並附數值文字（不只靠顏色）；< 480px 時方格可橫向捲動（規格見 4.0）。數值取自 4.2 表（logit 層級、有回饋、c = 3）。

桌機（≥ 1024px）左側 sticky 章節目錄；< 1024px 頂部橫向捲動章節列（沿用參考頁）。

---

## 4. 互動元件規格

共用：所有機率運算在 log 空間做；softmax 先減最大值；`kl(p,q)` 對 $p_i=0$ 的項以 0 計；所有讀數 JetBrains Mono；每個圖版底部一行「讀數由本頁依 Eq. ○○ 即時計算」，用到〔本頁推導〕或〔示意〕時加對應標籤。

### 4.0 版面與無障礙規格（全頁適用，390px 為最小驗收寬度）

- 顯示式：`.katex-display{overflow-x:auto;overflow-y:hidden}`。Eq. 2、10、11、16 在 390px 下允許式內橫捲，頁面本身不得橫捲。
- 表格：所有表格（Table 3／4／5／6／9／10／12、符號表、C5 小表）外包 `<div class="tbl-wrap">`（`overflow-x:auto`），左右以 `mask-image` 漸層提示可捲動；`tbl-wrap` 有 `tabindex="0"` 與 `aria-label`（表名），讓鍵盤可捲動。
- SVG：SVG 內只放純文字標籤，不放 KaTeX；π、q、A、θ、α 用 Unicode，下標用 `<tspan baseline-shift>` 或改寫為 `q_ref`。SVG 寬度以 `viewBox` 縮放，文字最小 11px（390px 寬時換算）。
- token 方格（hero、C2、C2b）：容器 `role="listbox"`、`aria-label`，每格 `role="option"`、`aria-selected`；roving tabindex（只有選取格 `tabindex="0"`），←／→ 移動、Home／End 到首尾，Enter／Space 選取並更新詳細面板。hero 方格不可選取，用 `role="img"`＋完整文字 `aria-label`。
- live region：沿用參考頁 `announce()`（debounced）。連續動畫（C3「執行到 200 步」）期間不呼叫 `announce()`，結束後只播報一次最終讀數。
- 教師的斜線填充 `pattern` 在深色模式下另設線色（`--c-tea` dark 值）並重測圖形對比 ≥ 3:1。

### C1 · GRPO 的群組優勢與「優勢全為 0」（S2）

- **教的概念**：$A_i=r_i-\mathrm{mean}$；整列常數；全組相同時為 0；「一組優勢全為 0」的機率隨 $p$、$G$ 的變化。
- **輸入**：群組大小 $G\in\{4,8,16\}$（分段按鈕，預設 8）；$G$ 個 rollout 切換鈕（`aria-pressed`，通過＝實心圓＋「✓」、失敗＝空心圓＋「✗」），預設 $r=[1,0,0,1,0,0,0,0]$（$G$ 改變時前 $\lceil G/4\rceil$ 個設為通過、其餘失敗）；按鈕「全部失敗」「全部通過」「依 p 隨機取樣」；滑桿 $p\in[0,1]$，步長 0.01，預設 0.10，滑桿旁註「右圖用 p；左圖用手動切換的結果，按『依 p 隨機取樣』可用 p 產生一組」。
- **計算**：$\bar r=\frac1G\sum r_i$；$A_i=r_i-\bar r$；優勢全為 0 的機率 $P_G(p)=(1-p)^G+p^G$（$p=0$ 或 1 時為 1，直接計算無數值問題）；每批 32 題的期望題數 $32P_G(p)$。隨機取樣：`mulberry32(seed)`，seed 從 1 開始、每按一次加 1；$r_i=[u_i<p]$。
- **顯示**：左：$G$ 列 rollout，每列是一條固定 12 格的 token 帶，帶下方長條高度＝$A_i$（正向上、負向下，附數值），強調「同一列每格相同」。全部相同時顯示提示列「所有優勢為 0：這一題本步沒有梯度」。右：$P_G(p)$ 對 $p$ 的三條曲線（$G=4$ 細虛線、$8$ 實線、$16$ 點線，各有直接標籤），目前 $p$ 與 $G$ 處畫 `--accent` 點。讀數：$\bar r$、各 $A_i$、$P_G(p)$、「每批 32 題中約 N 題沒有梯度」。圖下註「假設各 rollout 獨立、成功率相同〔本頁推導〕」。
- **讀者應觀察到**：(1) 優勢總和恆為 0；(2) 全對或全錯時全為 0；(3) $p$ 很小時（難題）優勢全為 0 的機率接近 1，$G$ 越大越低；$p=0.1$ 時多按幾次「依 p 隨機取樣」，常會出現全部失敗。

### C2 · 自我教師的逐 token 機率比（S3）

- **教的概念**：教師以「同一模型＋回饋」重算原回答的機率；$\log q(y_t)/\pi(y_t)$ 稀疏且帶正負；GRPO 整列常數。S3 只用到機率與 log；KL、Eq. 11、粒度留到 C2b（S8b）。圖版上這個量標為「log(q/π)：教師相對學生的機率比（05 章會說明它就是 SDPO 的優勢）」。
- **資料**（〔示意〕，本頁構造；以 Fig. 4 的題目為背景，分詞為示意分詞）。每列：取樣 token（＝學生 top-1）與另兩個候選；學生 $\pi$、教師 $q$ 的 top-3 機率；尾＝`Math.max(0, 1 − 三者之和)`，尾為 0 時略過該項（與 C3 一致）。

| t | token（候選 2、3） | $\pi$ top-3 | $q$ top-3（有回饋） |
| --- | --- | --- | --- |
| 1 | `def`（`class`、`import`） | .97 .01 .01 | .97 .01 .01 |
| 2 | ` numbers_up_to_n`（` get_numbers`、` nums`） | .90 .05 .03 | .90 .05 .03 |
| 3 | `(n):`（`(n: int):`、`():`） | .98 .01 .005 | .97 .02 .005 |
| 4 | ` return`（` result`、` for`） | .80 .10 .05 | .82 .09 .05 |
| 5 | ` list`（` [`、` range`） | .70 .25 .03 | .68 .27 .03 |
| 6 | `(range`（`(map`、`(filter`） | .95 .02 .01 | .95 .02 .01 |
| 7 | `(1,`（`(n`、`(0,`） | .92 .03 .03 | .90 .02 .06 |
| 8 | ` n`（` n-1`、` n+1`） | .96 .02 .01 | .93 .05 .005 |
| 9 | ` +`（`))`、` -`） | .85 .14 .005 | .06 .92 .01 |
| 10 | ` 1`（` 0`、` n`） | .97 .02 .005 | .40 .55 .01 |
| 11 | `))`（`)`、`)]`） | .97 .02 .005 | .95 .03 .01 |

- **輸入**：回饋切換「無回饋（教師上下文與學生相同）」／「環境回饋：Don't include n.」（預設後者）；GRPO 列用的「同組其他 7 個 rollout 中的成功數」$c\in\{0,\dots,7\}$（$G=8$，被畫的 rollout 本身失敗，預設 3）；點選任一 token 打開詳細面板（鍵盤規格見 4.0），預設選第 9 格。
- **計算**：
  - 無回饋：$q:=\pi$，所有 log 比值為 0。
  - 位置 $t$ 的 4 個值 $\log(q_j/\pi_j)$（3 個候選＋尾）；方格長條顯示取樣 token 的值。
  - GRPO 列：此 rollout 失敗（$r=0$），組平均 $c/8$，$A=0-c/8$（$c=7$ 時 −0.875）。
- **顯示**：token 方格列（等寬字），下方兩列長條：GRPO（`--ink-3`，整列等高）與 SDPO（正值 `--c-pos` 向上、負值 `--c-neg` 向下，附帶符號數值，兩位小數）。詳細面板：該位置的兩組長條（學生實心、教師斜線填充）與每個候選的 log(q/π)（含尾）。讀數：選取格的 log(q/π)、GRPO 值。不顯示 KL、Eq. 11、序列平均。
- **讀者應觀察到**：(1) 大多數位置 $|A|<0.04$，只有 ` +`（−2.65）與 ` 1`（−0.89）明顯為負；(2) 教師在 ` +` 位置把機率移到 `))`（該候選 log 比值 +1.88），也就是指出「應該在 n 後面直接結束括號」；(3) $c=0$ 時 GRPO 列全為 0，但 SDPO 仍有訊號（環境輸出不需要成功解）。

### C2b · 三種信用分配粒度（S8b）

- **教的概念**：logit／token／序列層級 SDPO 與 GRPO 各給出幾個不同的優勢值；序列層級把局部訊號平均掉。放在 S8b，讀者已在 S4a、S5 學過 KL 與 Eq. 11，不需捲回 S3。
- **資料**：共用 `C2_ROWS`（4.2 表，有回饋）；GRPO 固定 $c=3$（−0.375）。
- **輸入**：只有選取格（listbox，規格見 4.0，預設第 9 格），四列共用同一個選取。不另設粒度切換：四列同時顯示，方便比較（審查建議為切換鈕，改成並列的原因見修訂紀錄）。
- **計算**：
  - logit 層級：每格 4 個值 $\log(q_j/\pi_j)$（3 候選＋尾）；位置損失＝Eq. 11（$K=3$）。
  - token 層級：只保留學生 top-1 與尾（Eq. 11，$K=1$）——本頁對「most likely token」的解讀（R5）。
  - 序列層級：$\bar A=\frac1{11}\sum_t\log(q(y_t)/\pi(y_t))=-0.3296$，每格同一值。
  - GRPO：每格 −0.375。
  - 不同優勢值個數：logit $11\times4=44$、token $11\times2=22$、序列 1、GRPO 1（本例 $K=3$；論文 Sec. 4 主實驗 $K=20$，即每格 21 個）。
- **顯示**：四列 token 方格，列名在左、優勢值個數在右。logit 列每格畫 4 條細長條，token 列 2 條，序列與 GRPO 列每格 1 條等高長條。下方讀數：選取格的 Eq. 11 損失（$K=3$、$K=1$）、總損失、$\bar A$。
- **讀者應觀察到**：(1) 序列層級把第 9 格的 −2.65 稀釋成每格 −0.33；(2) token 層級在第 9 格只剩取樣 token 與尾，看不到教師偏好的 `))`（+1.88）；(3) 位置損失 $K=1$ 與 $K=3$ 在第 9 格幾乎相同（1.978 對 1.983），差別在其他候選的方向資訊。

### C3 · 單一位置的 logit 蒸餾（S5）

- **教的概念**：Eq. 1 在一個位置上的作用；reverse KL 時梯度＝$\pi_k(A_k-\mathbb E_\pi A)$；三種散度的差異；Eq. 11 的近似誤差。
- **資料**（〔示意〕，C2 第 9 格的完整 6-token 版本，前 3 個與 C2 一致、尾部拆開）：

| token | ` +` | `))` | ` -` | `)` | ` *` | `,` |
| --- | --- | --- | --- | --- | --- | --- |
| $\pi_0$ | .85 | .14 | .005 | .003 | .001 | .001 |
| $q$ | .06 | .92 | .01 | .008 | .0015 | .0005 |

- **簡化**（頁面寫出）：把這個位置的學生 logits $z\in\mathbb R^6$ 當成獨立參數（實際模型中所有位置共用 $\theta$）；教師固定（stopgrad）；純梯度下降。
- **輸入**：散度 reverse KL（預設）／forward KL／JS；學習率 $\eta\in\{0.5,1,2\}$（分段，預設 1）；按鈕「+1 步」「+10 步」「執行到 200 步」「重設」；滑桿 $K\in\{1,\dots,6\}$（預設 3）。
- **計算**（解析梯度，不用數值微分）：$z_0=\log\pi_0$；$\pi=\mathrm{softmax}(z)$。
  - reverse：$L=\sum\pi\log(\pi/q)$；$g_j=\log(\pi_j/q_j)$。
  - forward：$L=\sum q\log(q/\pi)$；$\partial L/\partial z_k=\pi_k-q_k$。
  - JS：$m=(\pi+q)/2$，$L=\tfrac12\mathrm{KL}(\pi\|m)+\tfrac12\mathrm{KL}(q\|m)$；$g_j=\tfrac12\log(\pi_j/m_j)$〔本頁推導：$\partial\mathrm{JS}/\partial\pi_j=\tfrac12\log(\pi_j/m_j)$〕。
  - reverse 與 JS：$\partial L/\partial z_k=\pi_k(g_k-\sum_j\pi_jg_j)$。
  - 更新 $z\leftarrow z-\eta\nabla_zL$。
  - top-K（只對 reverse KL 顯示；Eq. 11 原本就是 reverse KL）：依目前 $\pi$ 由大到小（同值依索引）取前 $K$；尾質量**以剩餘項直接加總**，不用 $1-\sum$（避免相消）；尾 $\pi$ 質量為 0 時略過尾項。
- **數值保護**：$\log\pi$ 用 $z-\mathrm{logsumexp}(z)$；$q$ 無零值；$\pi_j$ 下限不需要（softmax 永遠 > 0，但若 $<10^{-300}$ 該項以 0 計）；執行上限 200 步，避免無限迴圈。
- **顯示**：上：6 組長條（學生實心、教師斜線），值標在長條上。中：優勢 $A_j=\log(q_j/\pi_j)$ 的帶符號長條，並以水平虛線標 $\mathbb E_\pi[A]$（reverse 時＝$-L$）。下：$-\nabla_zL$ 長條（標「這一步 logits 的移動方向」）與一列公式說明，依散度切換：reverse 顯示「$-\nabla_z L=\pi(A-\mathbb E_\pi A)$（Prop. 2.1 的單一位置版本〔本頁推導〕）」並列出兩者的數值以示相等；forward 顯示「$-\nabla_z L=q-\pi$」；JS 顯示「$-\nabla_z L=-\pi(g-\mathbb E_\pi g)$，$g=\tfrac12\log(\pi/m)$〔本頁推導；論文沒有給 JS 的優勢形式〕」。右：損失對步數（y 軸對數 $[10^{-4},3]$，繪點用 `max(L,1e-4)`，圖下註「低於 1e-4 的值畫在下緣」）。讀數：步數、$L$、π(`))`)、reverse KL 精確值、top-K 近似值、差值。reduced-motion 時「執行到 200 步」直接算完一次重畫；否則每影格 4 步；執行中暫停 live region（4.0）。
- **讀者應觀察到**：(1) reverse KL 第 0 步時 ` +` 的移動方向為負、`))` 為正，且 $-\nabla$ 正好等於 $\pi_k(A_k-\mathbb E_\pi A)$；(2) 三種散度都讓 $\pi$ 收斂到 $q$；在這組〔示意〕數值上 JS 起步較慢（初始梯度較小），這是本頁玩具例的結果，不是論文的結論；(3) $K$ 增加時近似值單調不減並在 $K=5$ 等於精確值；$K=2\to3$ 時差值不變，因為 ` -` 與其後尾部的 $q/\pi$ 比值相同（都是 2）。

### C4 · 信任域教師與 EMA（S6）

- **教的概念**：Eq. 10 是 log 空間的幾何內插；$\alpha$ 越小越貼近初始教師；EMA 中初始**參數**的權重 $(1-\alpha)^n$ 隨更新次數衰減，而信任域在 log **分布**內插中給初始教師的權重固定為 $1-\alpha$。兩者量的是不同的東西，頁面並列只為說明「會衰減」與「固定」的差別〔本頁推導〕。
- **資料**（〔示意〕）：$q_{\theta_{ref}}$＝C3 的 $q$（初始教師）；$q_\theta$＝漂移後的教師（已偏向學生的答案）：` +` .70、`))` .25、` -` .02、`)` .02、` *` .005、`,` .005。
- **輸入**：滑桿 $\alpha\in[0,1]$，步長 0.01，預設 0.01；chip「0.01（Table 4；Table 12 Sec. 4–5）」「0.05（Table 12 Sec. 3）」「0.5」「1」；滑桿「EMA 更新次數 $n$」$\in[0,200]$，預設 80（只影響 EMA 小圖）。滑桿下註：「論文沒有說明 EMA 每個優化步還是每批生成更新一次；Sec. 4 一個訓練步最多有 32 次優化步，若每步都更新，訓練步數與 $n$ 不相同（R13）。」
- **計算**：$\ell_j=(1-\alpha)\log q_{ref,j}+\alpha\log q_{\theta,j}$；$q^*=\mathrm{softmax}(\ell)$；$\mathrm{KL}(q^*\|q_{ref})$、$\mathrm{KL}(q^*\|q_\theta)$；EMA 權重 $w(n)=(1-\alpha)^n$（$\alpha=0$ 時為 1，$\alpha=1$、$n\ge1$ 時為 0；以 `Math.pow` 直接算）。
- **顯示**：三組長條（$q_{ref}$、$q^*$、$q_\theta$，以填充樣式區分並有直接標籤）；兩個 KL 讀數；兩張分開的小圖並排（< 600px 上下堆疊），各有自己的 y 軸標題：左「EMA：初始參數在 θ′ 中的權重」，畫 $w(n)$ 對 $n$（實線，目前 $n$ 處標點）；右「信任域：log 空間內插中初始教師的權重」，畫 $1-\alpha$ 對 $\alpha$（實線，目前 $\alpha$ 處標點）。兩圖之間標〔本頁推導〕與一句「兩者單位不同，不能直接比大小」。
- **讀者應觀察到**：(1) $\alpha=0.01$ 時 $q^*$ 幾乎等於初始教師（KL 0.00004）；(2) $\mathrm{KL}(q^*\|q_{ref})$ 隨 $\alpha$ 單調上升；(3) $\alpha=0.05$、$n=80$ 次 EMA 更新後，初始參數權重只剩 0.0165；信任域在 log 內插中給初始教師的權重固定為 0.95，不隨時間變。

### C5 · discovery@k 與難題門檻（S9）

- **教的概念**：Eq. 4；固定模型的 best-of-k：$1-(1-p)^k$；hard／very hard 門檻對應的單次成功率；截斷平均首次成功時間；2944 樣本的無偏 pass@k 估計。
- **輸入**：模式切換「單次成功率 $p$」（預設）／「2944 次中成功 $c$ 次」；$p$：對數滑桿 $\log_{10}p\in[-5,-1]$，200 段，預設 $p=10^{-3}$；chip「very hard 上限（4.76e-4）」「hard 上限（1.08e-2）」；$c\in\{0,\dots,100\}$，預設 1；x 軸切換線性（預設，0–2750，與 Fig. 13 相同）／對數（從 $k=1$ 到 2750，$k=0$ 不畫）。
- **計算**：
  - $\mathrm{disc}(p,k)=-\mathrm{expm1}(k\cdot\mathrm{log1p}(-p))$。
  - 達到目標 $\tau$ 的最小 $k$：$\lceil\mathrm{log1p}(-\tau)/\mathrm{log1p}(-p)\rceil$，> 2750 時顯示「> 2750」；`kFor` 在 $p\le0$ 時回傳 `Infinity`，同樣顯示「> 2750」。
  - 截斷平均：$-\mathrm{expm1}(2750\cdot\mathrm{log1p}(-p))/p$。
  - $c$ 模式：$\widehat{\mathrm{pass@}k}=1-\prod_{i=0}^{k-1}\frac{n-c-i}{n-i}$，$n=2944$；$n-c<k$ 時為 1；$c=0$ 時為 0。以累乘逐 $k$ 產生整條曲線（一次 $O(n)$），不用階乘。
- **顯示**：曲線（`--accent` 實線）；水平虛線 22%、50%；垂直細線 $k=64$。讀數：pass@64、discovery@2750、$k$(22%)、$k$(50%)、截斷平均。下方小表照錄論文的彙總值（very hard：SDPO 53.2%、best-of-k 41.5%、多輪 35.6%；hard：SDPO 78%），說明文字固定放在小表**正下方**（不放在圖版底部或註腳）：「這些是 9／19 題、5 seeds 的平均，不是某個單一 $p$ 的曲線；各題 $p$ 不同時，平均曲線不等於平均 $p$ 的曲線。」
- **讀者應觀察到**：(1) very hard 上限 $p=4.76\times10^{-4}$ 時 pass@64 = 0.030、達到 22% 需要 523 次；(2) $p=10^{-4}$ 時 2750 次內只有 24%；(3) 2944 次中只成功 1 次時，估計的 pass@64 為 0.0217，已低於 0.03。

---

## 5. 驗收數值表（`verify/check.py` → `check.out`，與頁面算法相同）

C1（容許 ±1e-4）：

| 情況 | 期望值 |
| --- | --- |
| $r=[1,0,0,1,0,0,0,0]$ | $\bar r=0.25$；$A=+0.75$（通過）、$-0.25$（失敗） |
| $P_8(p)$，$p=$ 0.01／0.05／0.1／0.3／0.5／0.9 | 0.9227／0.6634／0.4305／0.0577／0.0078／0.4305 |
| $P_4(0.1)$、$P_{16}(0.1)$ | 0.6562、0.1853 |
| 每批 32 題，$G=8,p=0.1$ | 13.8 |

C2（容許 ±0.001）：

| 位置 | 3 | 4 | 5 | 7 | 8 | 9 | 10 | 11 | 1,2,6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $A(y_t)$ | −0.010 | +0.025 | −0.029 | −0.022 | −0.032 | −2.651 | −0.886 | −0.021 | 0.000 |
| KL（$K=3$） | 0.0031 | 0.0019 | 0.0011 | 0.0116 | 0.0150 | 1.9827 | 0.7791 | 0.0052 | 0.0000 |
| KL（$K=1$） | 0.0019 | 0.0013 | 0.0009 | 0.0024 | 0.0081 | 1.9780 | 0.7694 | 0.0049 | 0.0000 |

第 9 格逐 logit 優勢（` +`、`))`、` -`、尾）：−2.651、+1.883、+0.693、+0.693。序列層級 $\bar A=-0.3296$（總和 −3.6258）。GRPO 列：$c=0$ → 0，$c=3$ → −0.375，$c=7$ → −0.875（$c$ 上限 7，見 round1 #13）。

C3（容許 ±1e-4；`check.py` 已以中央差分驗證三種解析梯度）：

- 第 0 步：$A$＝[−2.651, +1.883, +0.693, +0.981, +0.405, −0.693]；$\mathbb E_\pi A=-1.9836$＝$-L$；$-\nabla$＝[−0.5672, +0.5413, +0.0134, +0.0089, +0.0024, +0.0013]。

| 散度 | η | L@0 | L@1 | L@10 | L@50 | L@200 | $\pi$(`))`)@10 | @200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reverse | 1 | 1.9836 | 1.2332 | 0.0074 | 0.0039 | 0.0006 | 0.9132 | 0.9214 |
| reverse | 0.5 | 1.9836 | 1.6374 | 0.0271 | 0.0050 | 0.0021 | 0.8720 | 0.9245 |
| reverse | 2 | 1.9836 | 0.4727 | 0.0051 | 0.0021 | 0.0002 | 0.9276 | 0.9201 |
| forward | 1 | 1.5881 | 0.5629 | 0.0124 | 0.0038 | 0.0003 | 0.8996 | 0.9205 |
| JS | 1 | 0.3664 | 0.3371 | 0.0668 | 0.0023 | 0.0010 | 0.6459 | 0.9257 |

top-K（第 0 步，精確 1.98356）：K = 1…6 近似值 1.97797、1.98274、1.98274、1.98327、1.98356、1.98356。

C4（容許 ±1e-4）：

| α | 0 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $q^*$(` +`) | 0.0600 | 0.0621 | 0.0714 | 0.0847 | 0.1386 | 0.2864 | 0.4956 | 0.7000 |
| $\mathrm{KL}(q^*\|q_{ref})$ | 0 | 0.00004 | 0.00121 | 0.00534 | 0.04468 | 0.26958 | 0.77575 | 1.44371 |
| $\mathrm{KL}(q^*\|q_\theta)$ | 1.03406 | 1.02526 | 0.98822 | 0.93771 | 0.75853 | 0.39960 | 0.09989 | 0 |

EMA $(1-\alpha)^n$：$\alpha=0.01$：n = 10/20/50/80/100/200 → 0.9044/0.8179/0.6050/0.4475/0.3660/0.1340；$\alpha=0.05$ → 0.5987/0.3585/0.0769/0.0165/0.0059/0.0000。

C5（機率 ±1e-4，$k$ 完全相同，平均 ±0.1）：

| $p$ | pass@64 | disc@2750 | $k$(22%) | $k$(50%) | 截斷平均 |
| --- | --- | --- | --- | --- | --- |
| 1e-5 | 0.0006 | 0.0271 | > 2750 | 69315 | 2712.5 |
| 1e-4 | 0.0064 | 0.2404 | 2485 | 6932 | 2404.4 |
| 4.758e-4 | 0.0300 | 0.7299 | 523 | 1457 | 1533.9 |
| 1e-3 | 0.0620 | 0.9362 | 249 | 693 | 936.2 |
| 1.077e-2 | 0.5000 | 1.0000 | 23 | 64 | 92.8 |

$c$ 模式（$n=2944$）：$c$ = 0/1/2/5/32 → pass@64 = 0/0.0217/0.0430/0.1041/0.5069；pass@2750 = 0/0.9341/0.9957/1.0000/1.0000。門檻：very hard $p<4.758\times10^{-4}$（期望 1.401 次）、hard $p<1.077\times10^{-2}$（期望 31.713 次）。

---

## 6. 實作計畫

### 6.1 檔案與沿用

單一檔案 `tut/260/build/index.html`，CSS／JS／時間線資料全部內嵌，不 `fetch` 本機檔。上線時複製為 `tutorials/rl-via-self-distillation.html`。**直接沿用參考頁 `tutorials/prediction-targets-k-diff.html` 的外殼**：`<head>`（Google Fonts、KaTeX 0.16.9 cdnjs 與 auto-render、分隔符）、CSS tokens 與深色模式、`appbar`、`tocbar`／`toc`（`data-toc`／`data-tt`）、`hero`、`ch`、`plate`／`plate-h`／`plate-lead`／`plate-f`、`callout`、`card`、本頁推導標籤、`<details>` 檢查題、時間線卡片列、`#pm-bar`、`initToc`／`initTheme`／`renderMath`／`announce`（debounced live region）、`el`／`frame`／`arrow`／`widthOf` SVG 輔助、`?test` 的 `selfTest()` 結構與 `scrollWidth <= innerWidth` 檢查。以下只列新增或變更。

### 6.2 新增 tokens（只用在圖表資料系列）

刪除參考頁的 `--c-par`、`--c-perp`，新增：

| token | light | dark | 用途 |
| --- | --- | --- | --- |
| `--c-stu` | #3F6E8C | #8DB4D3 | 學生分布（實心） |
| `--c-tea` | #6B7A3A | #A9BC7C | 教師分布（斜線填充 `pattern`） |
| `--c-pos` | #2F6FA8 | #86B6E3 | 正優勢（與 Fig. 4／21 的藍＝正一致） |
| `--c-neg` | #B23A3A | #E58A8A | 負優勢（紅＝負） |

正負一律同時以長條方向（上／下）與帶符號數值表示，不只靠顏色。實作時確認在 `--surface` 上圖形對比 ≥ 3:1、文字標籤 ≥ 4.5:1，不足就調明度並回寫此表。`--c-neg` 與 `--accent` 色相接近：`--accent` 只用於選取、focus、目前值標記，不用於資料。

### 6.3 新增 JS（單一 IIFE；函式名固定）

- 共用：`softmax(z)`、`lse(z)`、`kl(p,q)`、`fmt(x,d)`、`mulberry32(seed)`、`initListbox(el, n, onSelect)`（4.0 的 roving tabindex，hero 以外的方格共用）。
- C1：`grpoAdv(r)`、`collapseProb(p,G)`、`sampleC1()`、`initC1()`、`renderC1()`；state `{G, r:[], p, seed:1}`。
- C2：`const C2_ROWS=[{tok, cand:[…3], pi:[…3], q:[…3]}, …]`（照 4.2 表）；`c2Adv(row, fb)`（尾＝`Math.max(0,1-sum)`）、`initC2()`、`renderC2()`；state `{fb:true, c:3, sel:8}`（`c`∈0..7）。
- C2b：`c2KL(row, K)`、`initC2b()`、`renderC2b()`；state `{sel:8}`；與 C2 只共用 `C2_ROWS` 與 `c2Adv`，選取狀態各自獨立。
- C3（`divLossGrad` 的 JS 分支標〔本頁推導〕註解）：`const C3_PI0=[.85,.14,.005,.003,.001,.001], C3_Q=[.06,.92,.01,.008,.0015,.0005], C3_TOK=[' +','))',' -',')',' *',','];` `divLossGrad(z,q,div)` → `{L, g}`；`topKApprox(pi,q,K)`；`initC3()`、`stepC3(n)`、`renderC3()`；state `{div:'rkl', eta:1, z, step, hist:[]}`。
- C4：`const C4_QTH=[.70,.25,.02,.02,.005,.005];` `trTeacher(qref,qth,alpha)`；`initC4()`、`renderC4()`；state `{alpha:0.01, n:80}`。
- C5：`disc(p,k)`、`kFor(p,tau)`（$p\le0$ → `Infinity`）、`tmean(p,B)`、`passkCurve(n,c,kmax)`（累乘，回傳 Float64Array）；`initC5()`、`renderC5()`；state `{mode:'p', lp:-3, c:1, xlog:false}`。
- `selfTest()`（`?test`）：`collapseProb(0.1,8)=0.4305`；`c2Adv(row9,true)=-2.651`、序列平均 −0.3296；`divLossGrad` reverse 第 0 步 $L=1.9836$、η=1 跑 10 步 $L=0.0074$、JS 跑 50 步 0.0023；`topKApprox(K=1)=1.97797`、K=4 = 1.98327；`trTeacher` α=0.5 時 `KL(q*||qref)=0.26958`；`disc(4.758119e-4,64)=0.0300`、`kFor(4.758119e-4,0.22)=523`、`tmean(1e-3)=936.2`；`passkCurve(2944,1)[64]=0.0217`；`kFor(0,0.22)===Infinity`；C2 `c=7` 時 GRPO 值 −0.875；C2 每列尾 ≥ 0；全部 ±1e-4（k 精確），並保留參考頁的無水平捲動檢查。輸出一行 `selfTest: N passed, M failed`。

### 6.4 新增靜態圖

- S1：RLVR／RLRF 兩欄（Fig. 2 結構重繪），SVG，viewBox 0 0 720 240，每欄 340 寬；< 600px 改為兩個各自 viewBox 0 0 340 240 的 SVG 上下堆疊（先 RLVR、後 RLRF）。每欄節點與箭頭：
  - 欄標題（上方置中）：「RLVR：可驗證獎勵」／「RLRF：豐富回饋」。
  - 上方方框「模型（agent）」、下方方框「環境」。
  - 向下箭頭（模型 → 環境，右側）標「回答 y」。
  - 向上箭頭（環境 → 模型，左側）：RLVR 標「獎勵 r = −1.0」（細線，`--ink-3`）；RLRF 標「回饋 f」，旁邊等寬字小框照錄「Please answer in Newton-seconds.」（粗線，`--ink`）。
  - 兩欄共用左側的題目小框（< 600px 時放在兩圖之上）：「Q：How much impulse did the thrusters generate for the Mars Climate Orbiter?」「A：100 lbf-s」。
  - `role="img"`，`aria-label` 用一句話描述兩欄差異。
- S3：Table 2 模板方塊（HTML，不用 SVG）。
- S8b：原「四列 token 方格靜態小圖」改由互動元件 C2b 以 JS 產生（4.2），不另做靜態圖。
- S9：Fig. 12 三列示意（SVG，viewBox 0 0 720 210，每列高 60）。每列由左到右是方框與箭頭，列 $j$（$j=1,2,3$）：
  - 列 1：「問題 x」→「θ₀」→「回答 y₁」→「回饋 f₁」。
  - 列 2：「問題 x」→「θ₀」→「SDPO(y₁, f₁)」→「θ₁」→「回答 y₂」→「回饋 f₂」。
  - 列 3：「問題 x」→「θ₀」→「SDPO(y₁, f₁)」→「θ₁」→「SDPO(y₂, f₂)」→「θ₂」→「回答 y₃」→「回饋 f₃」。
  - 「SDPO(·)」方框用 `--c-tea` 外框，表示「把上一輪的 (y, f) 蒸餾進權重」；θ 方框用 `--c-stu`。每列右端的回饋以虛線箭頭連到下一列的 SDPO 方框（輸入），θ 方框是輸出。
  - < 600px：每列改為垂直堆疊的節點清單（HTML `<ol>`，不用 SVG），保持相同文字與順序。
  - 圖下一句：「上下文不放進提示，而是逐輪寫進參數；圖中每一輪畫成一次嘗試；實驗中每次 SDPO 更新用同一題的 16 個 rollout（Table 12 Sec. 5：question batch 1、rollouts 16）。」

### 6.5 連結

Hero「在關聯圖中查看」→ `../index.html#paper-260`；arXiv `https://arxiv.org/abs/2601.20802`；GitHub `https://github.com/lasgroup/SDPO`；筆記 `<slug>.html`（同在 `tutorials/`）。

筆記（`notes/` frontmatter 的 slug；閱讀順序依 hub 的編號清單）：

| 順序 | slug | 標題 | 頁面中的位置 |
| --- | --- | --- | --- |
| hub | `sdpo-lineage-overview` | SDPO 的研究脈絡總覽 | S0 連結、S11 開頭、S12 第一條 |
| 1 | `rlvr-grpo-credit-assignment` | RLVR、GRPO 與信用分配瓶頸 | S2 段末、S12 |
| 2 | `knowledge-distillation-foundations` | 知識蒸餾基礎 | S4a 段末、S12 |
| 3 | `on-policy-distillation-imitation` | On-policy 蒸餾與線上模仿學習 | S2 段末（「沒有更強教師」）、S12 |
| 4 | `context-distillation-self-distillation` | Context distillation 與自蒸餾 | S3 段末、S12 |
| 5 | `rich-feedback-and-retrospection` | 豐富回饋、事後回顧與 feedback-conditioned policy | S1 段末、S12 |
| 6 | `test-time-training-for-rl` | 測試時訓練與 test-time RL | S9 段末、S12 |
| 站內既有 | `test-time-training-naming`、`ttt-theory-and-applications` | — | S12 |

降級規則：建置時檢查 `tutorials/<slug>.html` 是否存在（筆記尚未轉成 HTML 時不存在），不存在就不輸出該連結，並在建置回報列出；不留空連結或〔待填〕字樣。頁面不引用筆記的內文數字（見 R14）。

### 6.6 研究脈絡（由 notes/timeline.json 提供）

- 建置時把 `notes/timeline.json`（13 筆，2010–2026）原樣貼進 `<script type="application/json" id="timeline-data">`，格式與呈現（卡片列、篩選 chip、本文卡片 `--accent` 外框、缺欄位與解析失敗處理）完全沿用參考頁。JSON 解析失敗或為空時整個時間線卡片列不輸出，S11 只留文字段落。
- 分類常數 `TL_GROUP`（以 arXiv ID 為 key，類別 → 筆記）：

| 類別 | arXiv ID | 連到筆記 |
| --- | --- | --- |
| 策略梯度與 RLVR | 1707.06347、2402.03300、2501.12948、2305.20050 | `rlvr-grpo-credit-assignment` |
| 蒸餾 | 1503.02531、1606.07947 | `knowledge-distillation-foundations` |
| 線上蒸餾與模仿學習 | 1011.0686、2306.13649 | `on-policy-distillation-imitation` |
| 情境蒸餾／自我蒸餾 | 2209.15189、1705.08439、2006.07733 | `context-distillation-self-distillation` |
| 測試時訓練 | 2504.16084 | `test-time-training-for-rl` |
| 本文 | 2601.20802 | — |

  timeline 沒有「從回饋學習」類的條目（Self-Refine、RLEF 等只在筆記中），因此不設這個 chip。不在表中的條目歸「其他」並在建置回報列出。
- S12 延伸閱讀依 6.5 表的順序列出筆記與其 frontmatter `title`。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 筆記已產生但沒有 `index.json`，也還沒有 HTML | 7 篇筆記與 `timeline.json` 已在 `notes/`；閱讀順序只存在於 hub 的編號清單。 | 6.5 寫死 slug 與順序；連結依 6.5 的降級規則，缺 HTML 就不輸出；時間線解析失敗時不輸出卡片列（6.6）。 |
| R2 | 70.2% vs 66.6% 無法由表格重現 | Table 3 的 5h 欄平均為 70.0／66.8，Table 7 為 71.1／68.4（`check.py`）；論文未定義彙總方式。 | 照錄 Sec. 1 數字並加註腳寫明差異，不自行改數字。 |
| R3 | 論文內部不一致 | (a) App. E.2.1 說 Table 3 是「各模型各資料集分別選超參數」，Table 7 標題卻說 Table 3 是「每個方法全域選一組」、Table 7 才是分別選；(b) Sec. 4.1 寫「Qwen2.5-8B」，但 Fig. 17 的 Qwen2.5 軸為 1.5／3／7B。 | 採 Table 3／7 標題的說法（較具體），並在 S7 註腳指出 E.2.1 的不同說法；(b) 照錄原文並註「Fig. 17 橫軸只有 1.5、3、7B」。 |
| R4 | JS 散度的混合比例 | 論文只寫「symmetric Jensen–Shannon」，未給 β。 | C3 用 β = 1/2（對稱定義），頁面註明為本頁採用。 |
| R5 | token 層級 SDPO 的確切定義 | Sec. 4.2 只寫「the most likely token at each position」，是否含尾項不明。 | C2 以 Eq. 11、$K=1$（top-1＋尾）實作並標「本頁解讀」。 |
| R6 | 〔示意〕數值被誤認為論文數據 | C2–C4 的分布全為本頁構造。 | 每個圖版標題旁放「示意數值」pill，圖版底部一行說明；Hero 方格同樣標示。 |
| R7 | C5 與論文曲線不可直接比較 | 論文 discovery@k 是 19／9 題、5 seeds 的平均，各題 $p$ 不同。 | C5 不在曲線上疊論文點，只在下方小表照錄彙總值並說明。 |
| R8 | 「優勢全為 0」機率的簡化 | 實際各 rollout 成功率隨題目與訓練改變，非獨立同分布。 | C1 圖下註明假設；LO2 答案帶「本頁推導，假設獨立同機率」。 |
| R9 | 「6× speedup」「11×」「3×」出處 | 6× 與 11× 只針對 Olmo3 化學（Fig. 6）；3× 短是 Table 8 平均（3.2×）；Fig. 7 的「7× 短」是單一例子；Sec. 5 的 3× 只指 very hard 題、22% 發現機率（hard 題 67% 是 2.4×，平均首次成功 1.3×／1.2×）。 | 頁面每個倍數都附任務、題組與分位，不寫成全面結論。 |
| R10 | 符號衝突 | 論文同時用 $\lambda$（Eq. 3 混合、Eq. 17 熵係數、B.2 Lagrange 乘子）、$k$（嘗試次數、B.1 的索引）、$\alpha$（EMA 與信任域）。 | 頁面：Eq. 17 寫 $\lambda_{\mathrm{ent}}$、B.2 的 $\lambda$ 只在可收合推導內出現；B.1 的 $A_{t,k}$ 改寫為 $A_t(\hat y)$；$k$ 只用於 Sec. 5。 |
| R11 | Fig. 5 的對應 | 文字擷取無法判斷 +5.8%／+17.1% 對應哪個設定。 | 已渲染 PDF 第 5 頁確認：實心（無程式環境）+17.1%、淺色（含程式環境）+5.8%。 |
| R12 | 相對連結 | 建置位置與上線位置不同，預覽時站內連結失效。 | 接受；預覽時只檢查外部連結與頁內錨點。 |
| R13 | EMA 的更新頻率 | 論文沒有說 EMA 每個優化步還是每批生成更新一次；Sec. 4 一個訓練步最多 32 次優化步。 | 本頁的 $(1-\alpha)^n$ 只以「EMA 更新次數」表示，不換算成訓練步；C4 滑桿下加註。無法由論文解決，保留。 |
| R14 | 筆記內文與論文不一致 | `test-time-training-for-rl` 寫「發現解的速度為 RLVR 基線的 3 倍」：論文比較的是 best-of-k 與多輪對話，且 3× 只在 very hard、22% 分位。 | 頁面只連結筆記、不引用筆記數字；建置回報提請筆記作者修正（本設計不改筆記）。 |
| R15 | JS 的優勢形式 | Prop. 2.1 只對 reverse KL；Sec. 3 主實驗用 JS（Table 12），論文未給 JS 的優勢或梯度形式。 | S4b、S5、C3 寫明範圍；JS 梯度標〔本頁推導〕；C3 的 JS 收斂行為標〔示意〕。 |

---

## 修訂紀錄

- **Round 1（Fidelity＋Pedagogy 審查，彙整見 `reviews/round1.md` 共 24 項）**
  - 數字範圍：Sec. 5 的 3×／2.4×／1.3×／1.2× 都附題組與分位；Q3 分開標「最早一次 321」與「平均 1987」；Table 10 的 Hard 列註明含 very hard；Table 3 GRPO 較高的三格全部列出；「嚴格 on-policy」限定 Sec. 3；$K=100$（Sec. 4.2 文字）與 $K=20$（Table 12）分開；「RLVR 不會勝過 best-of-k」改為作者論證；Prop. 2.1 的 reverse-KL 範圍與 JS 的〔本頁推導〕／〔示意〕標籤。
  - EMA：$n$ 改為「EMA 更新次數」並加更新頻率說明（R13）；C4 把 EMA 與信任域分成兩張各有量綱標題的小圖。
  - 結構：S4 拆成 S4a／S4b，S8 拆成 S8a／S8b，顯示編號 01–14；S2 新增 $\nabla\log\pi$ 段落與數值例，S4a 新增 KL 數值例；推導全部收進可收合段落。
  - C2：只顯示 log(q/π)，不顯示 KL 與 Eq. 11；$c$ 改為 0..7；尾項 `max(0,·)`。粒度移到新元件 C2b（S8b）。審查建議 C2b 用粒度切換鈕，本設計改為四列並列：同一張圖同時比較 44／22／1／1 個值比來回切換直接，也少一個狀態。
  - C1 加「依 p 隨機取樣」；C3 公式列依散度切換、y 軸 `max(L,1e-4)`；C5 對數軸從 $k=1$、`kFor(p≤0)=Infinity`、說明文字放在小表正下方。
  - 文字：hero 標題、S1、S2 標題與 stopgrad 說明改成字面描述；hero 卡 1 去掉縮寫，卡 2 加「在論文的例子中」；hero 方格旁說明 −0.375 的由來。
  - 建置規格：新增 4.0（390px 顯示式與表格捲動、SVG 純文字、listbox 與 roving tabindex、動畫期間暫停 live region、深色模式斜線對比）；6.4 補 Fig. 2 與 Fig. 12 的節點、箭頭與堆疊規格。
  - 研究筆記：填入 7 個 slug、閱讀順序、`TL_GROUP` 對照與缺檔降級規則（6.5–6.6）；記錄筆記中 3× 說法的不一致（R14）。
- **Readiness 修正（Goldfish NO-GO）**
  - S2 先定義 $J(\theta)=\mathbb E[r]$ 再寫 $\nabla J$；符號表加 $J(\theta)$。
  - S5：Eq. 11 的不等式結論（近似值 ≤ 精確值、尾部比值相同時相等）移入正文，只把推導收合；LO6 不再依賴收合內容。
  - Table 11：15/19≈79%，改寫為「15 題（約 79%）」並註明論文正文的「78%」為截斷值。
- **Readiness 非阻擋項（建置前套用）**
  - 步驟 7：Lagrange 乘子改寫為 $\lambda_L$，避免與 Eq. 3 的 $\lambda$ 混淆（R10）。
  - C3 讀數 π(`))`) 多餘括號已刪。
  - S7、S8a 正文定義 avg@16（每題 16 次取樣的平均正確率）、GSPO（序列層級重要性比的 GRPO 變體）、CISPO（截斷重要性權重本身的變體）；TIS 已在步驟 6 定義。
  - S9 的「batch size 16」改用 Table 12 用語：question batch 1、16 個 rollout。
  - Hero 方格的〔示意〕標籤放在方格標題列，始終可見；文字改成「以 Fig. 4 的題目為背景」。
  - 規模結果（Fig. 8、17）只以文字呈現，不畫圖（論文沒有給可讀出的數字）。
  - 筆記連結：`add_tutorial.py` 上架時會把 7 篇筆記轉成 `tutorials/<slug>.html`，因此全部輸出連結（取代 6.5 的降級規則）。
- **建置紀錄**
  - S9 的 Table 10 摘錄另加 Q69（0.5×），說明也有單題比 best-of-k 慢；S8a 的 Table 9 加列「只更新高熵 token」一列（照錄論文）。
  - 推導類 `<details>` 用 `class="deriv"`，只有檢查題用 `class="check"`（驗收腳本只移除後者）。
  - 建置環境中 headless Chrome 對外部 CDN 的請求後來一律卡住，`?test`、截圖與 `extract_page_text.py` 改用把 KaTeX 換成本機副本、移除 Google Fonts 的測試複本執行（頁面本身不變）；截圖因此使用系統後備字型。
- **Elephant 修補（無脈絡讀者只讀正文的 LO 測驗：LO5、LO8、LO9 部分達成）**
  - LO5：S4b「讀法」段落末直接寫出「此優勢形式只對 reverse KL 證明（Prop. 2.1），forward KL／JS 未證明」，不再只靠後面的「適用範圍」段。
  - LO8：S7 開頭新增「結果摘要」段：70.2／66.6、平均短 3.2×、Olmo3 化學短 11×、50 分鐘達 GRPO 5h 正確率（6× 加速）。
  - LO9：S8a 主結果把「4× 少的生成次數」改寫成獨立的效率句；規模段先給結論（越大領先越多、1.5B 落後），「8B」一句改為說明無法對應 Fig. 17、不據此下結論（R3）。
