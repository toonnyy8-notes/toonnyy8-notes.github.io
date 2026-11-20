# 設計文件：〈On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics〉互動教學

- 論文：Julianna Piskorz、Antonin Berthon（同等貢獻）、Mihaela van der Schaar，University of Cambridge，arXiv 2609.35259（v1，2026-09-28，preprint）。資料庫編號 No. 136。
- 原文：`tut/136/src/paper.txt`（PDF 同目錄）。「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」皆指論文原文編號。注意論文正文與附錄的表格各自編號（正文沒有表；App. D 有 Table 1–6），本文件的「Table n」都指附錄表。
- 產出：`tut/136/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/on-policy-vs-off-policy-distillation.html`（slug `on-policy-vs-off-policy-distillation`）。站內相對連結以上線位置為準。
- 驗算腳本：`tut/136/verify/check.py`（與頁面同演算法；`python3 check.py` 印出第 4 節所有驗收值並跑 assert，最後一行 `all asserts passed`）。
- 研究筆記：`tut/136/notes/`（總覽 `odd-lineage-hub` + 7 篇主題筆記 + `timeline.json` 14 筆），上線後為 `tutorials/<slug>.html`。建置時重新讀取，若有增減以實際檔案為準。
- 標籤慣例：〔本頁推導〕＝本頁自行推導、論文沒寫出；〔示範值〕＝元件為了教學自訂的數值，與論文無關；〔圖讀值〕＝從論文圖中的數字標籤讀出（論文正文沒寫）；「作者認為／作者推測」＝作者的解釋，與量測結果分開寫。百分比的差一律寫「百分點（pp）」。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

大型語言模型的後訓練常用「強到弱蒸餾」（strong-to-weak distillation）：讓小的學生模型模仿大的教師模型的下一個 token 機率分布。訓練文本（rollout）可以由教師產生（off-policy distillation，OffPD），也可以由學生自己產生（on-policy distillation，OnPD）。先前研究把 on-policy 學習與較少的災難性遺忘、較稀疏的參數更新、較好的泛化連在一起，但那些比較（多半是 SFT 對 RLVR）同時改變了目標函數、監督來源、密度與最佳化設定。本文在蒸餾設定中**獨立改變三個因素**：rollout 策略（OnPD／OffPD）、token 層級 KL 方向（forward／reverse）、學習率（1×10⁻⁵／5×10⁻⁵）。結果（Sec. 4）：rollout 策略對最終準確率、遺忘、更新稀疏度都沒有一致的優勢；KL 方向較清楚地區分任務表現；學習率主導遺忘與稀疏度。作者再用 logit 梯度分析（Sec. 5.1、App. B–C）說明 forward KL 的更新對 rollout 變化有線性上界、reverse KL 沒有與分布無關的上界，並以連續的學生–教師 rollout 光譜 $\pi_\lambda$（Sec. 5.2）驗證：forward KL 對 rollout 穩健，reverse KL 敏感並偏好學生產生的 rollout。on-policy 資料在較難的 Countdown-4E 上兩種 KL 方向都較好，但這個優勢在後續 RLVR 後不穩定保留（Sec. 5.3）。

### 1.2 研究動機（Sec. 1–2）

1. 有一派觀點把「訓練資料由哪個策略產生」視為區分後訓練方法的主因：on-policy 被認為能減少災難性遺忘（Shenfeld et al., 2025；Chen et al., 2025）、產生更稀疏的更新（Mukherjee et al., 2025）、泛化更好（Chu et al., 2025 等）。
2. 這些證據多來自 SFT 與 RLVR 的比較或公開 checkpoint 分析，同時混入了目標函數、獎勵、監督密度、最佳化規模、梯度截斷與學習率的差異（Sec. 2）。學習率本身已知會強烈影響學習–遺忘權衡（Catalan-Tatjer & Geiping；Rofin et al., 2026）。
3. 實務意義：on-policy 方法訓練全程要持續生成，off-policy 可重用既有資料。如果好處其實來自其他設計選擇，額外成本就可能不必要（Sec. 1）。
4. 作者把上述報告當成「待檢驗的假說」，在只改 rollout 來源、其餘流程固定的蒸餾設定中檢驗（Sec. 2 末段）。

### 1.3 貢獻（Sec. 1 條列）

1. 受控研究：拆開 rollout 策略、KL 方向、學習率，發現 on-policy rollout 在最終準確率、災難性遺忘、參數更新稀疏度上沒有一致的優勢。
2. 依目標函數而異的敏感度解釋：梯度分析（Eq. 4–5、Theorem C.1、C.2、Corollary C.3）與學生–教師光譜實驗顯示 forward KL 對 rollout 穩健、reverse KL 明顯較敏感。
3. on-policy 何時有幫助：訓練穩定性與輸出涵蓋度主要由 KL 方向決定，遺忘與稀疏度由學習率決定；on-policy rollout 一致地改善對較難任務變體的泛化，並可能減少教師風格的附帶轉移。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S1 末尾以可收合表格提供同一份內容）

| 符號 | 意義 |
| --- | --- |
| $x$、$y$ | 輸入（prompt）與輸出序列；$x\sim p_{\text{data}}$ |
| $\mathcal V$、$\lvert\mathcal V\rvert$ | 詞彙表與其大小（論文記為 $M$；本頁不用 $M$ 這個字母，避免與其他符號混淆） |
| $y_{\lt n}$、$h_n$ | 第 $n$ 個 token 之前的輸出前綴；$h_n:=(x,y_{\lt n})$（Eq. 16） |
| $L_y$ | 輸出 $y$ 的長度 |
| $\pi_S^\theta$、$\pi_T$ | 學生（參數 $\theta$）與教師的下一個 token 分布 |
| $\rho$ | rollout 策略：產生訓練用輸出序列的策略 |
| $D$ | 比較學生與教師下一個 token 分布的距離函數（本文用 KL） |
| $z_S^\theta$ | 學生在某個前綴的 logit 向量，$\pi_S^\theta=\mathrm{softmax}(z_S^\theta)$ |
| $J_\theta(h_n)$ | logit 對參數的 Jacobian $\partial z_S^\theta(h_n)/\partial\theta$（Eq. 21） |
| $B$ | $J_\theta^\top$ 的最大奇異值上界（Eq. 26） |
| $\mathrm{TV}(P,Q)$ | total variation 距離 $\tfrac12\lVert P-Q\rVert_1$ |
| $r_n(v)$ | 學生對教師的 log 機率比 $\log\frac{\pi_S^\theta(v\mid h_n)}{\pi_T(v\mid h_n)}$（Eq. 42） |
| $\lambda$ | rollout 光譜參數；$\lambda=-1$ 為 OffPD，$\lambda=1$ 為 OnPD |
| $\alpha$、$c$ | 光譜外插時的可信度門檻（0.2）與 log 比截斷值（$2\ln3$）（App. D.5） |
| $\tau$ | 稀疏度門檻：參數變化量小於 $\tau$ 視為未更新（主結果 $10^{-6}$） |

#### 步驟 1：把兩種蒸餾寫成同一個目標（Sec. 3，Eq. 1）

$$L(\theta)=\mathbb E_{x\sim p_{\text{data}}}\,\mathbb E_{y\sim\rho(\cdot\mid x)}\Big[\frac1{L_y}\sum_{n=1}^{L_y}D\big(\pi_S^\theta(\cdot\mid x,y_{\lt n})\,\Vert\,\pi_T(\cdot\mid x,y_{\lt n})\big)\Big]\tag{Eq. 1}$$

- 目標對 rollout 策略 $\rho$ 走過的前綴取 token 層級差異的平均。$\rho=\pi_T$ 得 OffPD，$\rho=\pi_S^\theta$ 得 OnPD。
- OnPD 中損失也透過抽樣 $y\sim\pi_S^\theta$ 依賴 $\theta$；為了可計算，梯度不穿過抽樣過程，把抽到的軌跡當成固定（stop-gradient，沿用 Agarwal et al., 2023；Tang & Munos, 2025）。App. C 稱這樣得到的更新為**半梯度**（semi-gradient，Eq. 19）：期望內的分布在微分時固定；$\rho$ 不依賴 $\theta$ 時它就是一般梯度。
- 為什麼需要這一步：把 rollout 來源抽成一個符號 $\rho$，才能在不改其他東西的情況下比較。

#### 步驟 2：KL 的兩個方向，以及為什麼 rollout 與 KL 方向是兩個獨立選擇（Sec. 3，Eq. 2–3，App. C.5）

$$D_{\text{F-KL}}=\sum_{v\in\mathcal V}\pi_T(v)\log\frac{\pi_T(v)}{\pi_S(v)},\qquad D_{\text{R-KL}}=\sum_{v\in\mathcal V}\pi_S(v)\log\frac{\pi_S(v)}{\pi_T(v)}\tag{Eq. 2, 3}$$

- 本文在整個詞彙表上計算 KL（因為能取得兩個模型的完整輸出分布），避免由抽樣 token 估計 KL 的額外變異（Sec. 3，引 DeepSeek-AI et al., 2026）。
- forward KL 以教師機率加權，強烈懲罰學生沒涵蓋教師給高機率的 token，常稱 **mode-covering**；reverse KL 以學生機率加權，懲罰學生把機率放在教師認為不太可能的 token，常稱 **mode-seeking**（Sec. 3）。
- 慣例配對（OffPD＋forward、OnPD＋reverse）的兩個來源（Sec. 3）：
  1. 序列層級 KL 的兩種鏈式分解（Proposition C.4，Eq. 69–70）：$D_{\mathrm{KL}}(\pi_T(\cdot|x)\Vert\pi_S^\theta(\cdot|x))=\sum_{n=1}^H\mathbb E_{y_{\lt n}\sim\pi_T}D_{\text{F-KL}}(\dots)$，$D_{\mathrm{KL}}(\pi_S^\theta\Vert\pi_T)=\sum_n\mathbb E_{y_{\lt n}\sim\pi_S^\theta}D_{\text{R-KL}}(\dots)$。但這只是**目標值**的恆等式：stop-gradient 使 OnPD reverse-KL 的更新一般不等於序列層級 reverse KL 的完整梯度（Sec. 3、App. C.5）。另外實驗以實際長度 $L_y$ 正規化，實作目標一般不是序列 KL 的常數倍（App. C.5 末段）。
  2. 單樣本估計：在固定前綴下，$v\sim\pi_T$ 時 $\log[\pi_T(v)/\pi_S^\theta(v)]$ 是 forward KL 的不偏單樣本估計，$v\sim\pi_S^\theta$ 時 $\log[\pi_S^\theta(v)/\pi_T(v)]$ 是 reverse KL 的不偏估計，所以只用 rollout 自己的 token 估計 KL 時配對很自然。
- 本文在全詞彙上算 KL，KL 方向不再限制 rollout 策略；而且鏈式恆等式與抽樣解釋都**不代表**慣例配對較好最佳化或表現較好（Sec. 3）。因此作者把兩者當成獨立設計選擇，做成 2×2。

#### 步驟 3：受控實驗設計（Sec. 4.1，App. D）

| 項目 | 設定 |
| --- | --- |
| 因素 | rollout（OnPD／OffPD）× KL（forward／reverse）× 學習率（1×10⁻⁵／5×10⁻⁵）；Countdown-3 另掃 {1,2,3,4,5,6}×10⁻⁵（Fig. 2） |
| 模型 | 主實驗 Llama-3.1-8B 教師 → Llama-3.2-1B 學生；App. A.5 為 Qwen2.5-7B → Qwen2.5-1.5B。同家族以共用 tokenizer 與詞彙表 |
| 教師 | 每個任務各訓練一個教師並凍結（SFT 500 步，Science 與 MedReason 再做 GRPO），以建立足夠的能力差距（App. D.2.2）。Llama 教師準確率：MedReason 85.5、Science 67.5、Countdown-3 99.0（%，Table 2） |
| 任務 | MedReason（醫學）、Science（SciKnowEval 科學選擇題）、Countdown-3（三個運算元的算術）；教師回答平均長度 94–135 tokens（Sec. 6、Table 2） |
| 訓練 | 全參數微調 150 步，每設定 3 個 seed（42、43、44） |
| 指標 | 留出集 ID 準確率；災難性遺忘＝7 個 OOD 基準平均分數在訓練前後的下降（MMLU-Pro、TruthfulQA、IFEval、HumanEval、EQ-Bench、BBQ、ToxiGen，App. D.1.2）；更新稀疏度 $S_\tau=\frac1P\sum_i\mathbb I[\lvert\theta_i^{\text{final}}-\theta_i^{\text{init}}\rvert\lt\tau]$，$\tau=10^{-6}$（App. D.4，沿用 Mukherjee et al., 2025） |
| 溫度 | 教師 $\pi_T=\mathrm{softmax}(z_T/0.5)$、學生 $\pi_S^\theta=\mathrm{softmax}(z_S^\theta/1.0)$，抽樣與算 KL 都用同一組溫度；唯一的介入是 rollout 分布是 $\pi_T$ 還是 $\pi_S^\theta$（App. D.2.2） |

#### 步驟 4：主結果——rollout 沒有一致優勢（Sec. 4.2，Fig. 1–2）

- 最終準確率（三資料集平均）：最佳平均準確率 OnPD 72%、OffPD 73%。forward KL 在所有 rollout 與學習率下為 71–73%，reverse KL 為 35%–72%，且對學習率敏感得多。
- Fig. 1 長條標籤〔圖讀值，已對照 PDF 第 5 頁〕（ID 準確率 %，LR 1×10⁻⁵ / 5×10⁻⁵）：OnPD·F-KL 72 / 72；OnPD·R-KL 72 / 35；OffPD·F-KL 71 / 73；OffPD·R-KL 35 / 49。更新稀疏度（%）：OnPD·F-KL 85.3 / 51.9；OnPD·R-KL 88.5 / 58.5；OffPD·F-KL 88.8 / 59.5；OffPD·R-KL 89.6 / 60.0。
- 遺忘：低學習率時 OOD 平均最多變動 1.3 個百分點；高學習率時下降 11.2–14.0 個百分點。rollout 之間的差異相對小。使用 forward KL 時，降低學習率能消除大部分遺忘，同時保持相近的 ID 準確率。
- 稀疏度：低學習率 85.3–89.6%，高學習率 51.9–60.0%。在每一組配對比較中 OffPD 的更新至少與 OnPD 一樣稀疏；KL 方向是次要因素，on-policy rollout 上 reverse KL 一般比 forward KL 稀疏。
- 學習率掃描（Countdown-3，Fig. 2）：稀疏度隨學習率約線性下降，OOD 表現也一致變差，兩種 rollout 趨勢相近；相同學習率下 OffPD 幾乎在每組比較中遺忘較少，但差異小於學習率的效果。forward KL 下小學習率可同時達到最佳任務表現而不損失既有能力。作者把「最終表現相近但遺忘程度差很多」比擬為 Catalan-Tatjer & Geiping 在 SFT 中描述的「cliff」現象。
- 〔本頁推導〕由上列圖讀值計算「只改一個因素、另兩個固定」的配對絕對差（C4 呈現）：稀疏度的學習率配對差 29.3–33.4 pp，rollout 配對差 1.1–7.6 pp，KL 配對差 0.5–6.6 pp；ID 準確率的 rollout 配對差在 forward KL 下為 1 pp，在 reverse KL 下為 37 與 14 pp。這是描述性的配對差，不是變異數分解或因果歸因。

#### 步驟 5：logit 梯度揭示不對稱（Sec. 5.1，Eq. 4–5，App. B）

固定前綴、教師分布與 $\theta$ 無關時（App. B，Eq. 9–15）：

$$\nabla_\theta D_{\text{F-KL}}=\sum_{v}\big(\pi_S^\theta(v)-\pi_T(v)\big)\nabla_\theta(z_S^\theta)_v\tag{Eq. 4, 13}$$
$$\nabla_\theta D_{\text{R-KL}}=\sum_{v}\pi_S^\theta(v)\Big(\log\frac{\pi_S^\theta(v)}{\pi_T(v)}-D_{\text{R-KL}}\Big)\nabla_\theta(z_S^\theta)_v\tag{Eq. 5}$$

- 推導關鍵：softmax 的導數 $\partial\pi_S(u)/\partial z_v=\pi_S(u)(\mathbb I[u=v]-\pi_S(v))$（Eq. 9）；reverse 用 $\partial[x\log x]/\partial x=\log x+1$ 與 $\sum_u\pi_S(u)r(u)=D_{\text{R-KL}}$ 化簡（Eq. 14–15）。reverse 式假設凡 $\pi_S^\theta(v)>0$ 處 $\pi_T(v)>0$，否則 reverse KL 無限大（App. B 末）。
- forward 對 logit 的導數 $\pi_S^\theta(v)-\pi_T(v)$：只要兩分布不同就非零，每個座標落在 $[-1,1]$。
- reverse 對 logit 的導數以 $\pi_S^\theta(v)$ 加權：學生機率趨近 0 時導數消失，即使教師給這個 token 很高機率 → 難以找回學生遺漏的教師 mode（mode-seeking）。反過來，學生給某 token 可觀機率而教師認為極不可能時，$\log\pi_S^\theta/\pi_T$ 可任意大，可能造成尖銳、高變異的更新（Sec. 5.1）。
- 作者的推論（Sec. 5.1 末）：forward KL 在任何兩分布不一致的前綴都提供訊號；reverse KL 強調「學生支持、教師不支持」的 token，因此更依賴走過學生自己的前綴分布 → 預期 reverse KL 從 on-policy rollout 獲益較多。
- 〔本頁推導〕兩個梯度向量的座標和都是 0（$\sum_v(\pi_S-\pi_T)=0$；$\sum_v\pi_S(r-D_{\text{R-KL}})=D_{\text{R-KL}}-D_{\text{R-KL}}=0$），所以把所有 logit 加同一常數不改變損失；C1 用這點做數值自檢。

#### 步驟 6：rollout 穩定性的理論（App. C.1–C.4）

- 軌跡層級梯度：$g^\theta_{\text{F-KL}}(x,y)=\frac1{L_y}\sum_nJ_\theta(h_n)^\top(\pi_S^\theta(\cdot|h_n)-\pi_T(\cdot|h_n))$（Eq. 24），半梯度 $\bar\nabla_\theta L=\mathbb E_x\mathbb E_{y\sim\rho}[g]$（Eq. 25）。
- **Theorem C.1**（forward KL 的 rollout 穩定性）：若 $\lVert J_\theta(h_n)^\top v\rVert_2\le B\lVert v\rVert_2$（Eq. 26），則
  $$\big\lVert\bar\nabla_\theta L_{D_{\text{F-KL}}}(\theta;\rho)-\bar\nabla_\theta L_{D_{\text{F-KL}}}(\theta;\rho')\big\rVert_2\le2\sqrt2\,B\;\mathbb E_x\big[\mathrm{TV}(\rho(\cdot|x),\rho'(\cdot|x))\big]\tag{Eq. 27}$$
  且單一軌跡梯度二階動差 $\le2B^2$（Eq. 28）。證明：任兩機率向量 $\lVert p-q\rVert_2\le\sqrt2$ → 每 token $\le\sqrt2B$（Eq. 30）→ 軌跡平均仍 $\le\sqrt2B$（Eq. 33）→ 有界函數在兩分布下期望差 $\le2\sup\lVert g\rVert\,\mathrm{TV}$（Eq. 34）。
- **Theorem C.2**（reverse KL 沒有與分布無關的上界）：不存在與學生、教師分布無關的有限常數 $C$ 使 $\lVert\Delta\bar\nabla\rVert_2\le C\,\mathbb E_x[\mathrm{TV}]$（Eq. 46），即使詞彙只有兩個 token、Jacobian 最大奇異值為 1。反例（Eq. 47–57）：長度 2 的兩條軌跡 $y^{(0)}$、$y^{(\delta)}$；$y^{(0)}$ 各前綴上學生＝教師（梯度 0）；$y^{(\delta)}$ 第二個前綴上 $\pi_S=(\tfrac12,\tfrac12)$、$\pi_T=(\delta,1-\delta)$，$0\lt\delta\lt\tfrac12$，直接 logit 參數化 $J^\top=I$。此處 logit 梯度 $=\tfrac14\log\frac{1-\delta}\delta(1,-1)$（Eq. 51），軌跡梯度範數 $\frac{\sqrt2}8\log\frac{1-\delta}\delta$（Eq. 53）。令 $\rho'=\Delta_{y^{(0)}}$、$\rho=(1-\varepsilon)\Delta_{y^{(0)}}+\varepsilon\Delta_{y^{(\delta)}}$，TV $=\varepsilon$（Eq. 55），更新差 $=\frac{\varepsilon\sqrt2}8\log\frac{1-\delta}\delta$（Eq. 57），$\delta\downarrow0$ 時發散。反例中教師所有機率都嚴格為正，所以適用於 softmax 語言模型。
- **Corollary C.3**（受控 log 比下的 reverse-KL 上界）：若每個前綴上 $\max_vr_n(v)-\min_vr_n(v)\le R$（Eq. 58），則差 $\le2BR\,\mathbb E_x[\mathrm{TV}]$（Eq. 59），二階動差 $\le B^2R^2$（Eq. 60）；中間步驟：每個前綴 $\lVert\nabla_zD_{\text{R-KL}}\rVert_2\le R$（Eq. 63–65）。
- 作者寫明的範圍（App. C.2、C.4 的 Interpretation）：定理控制的是**一步的期望半梯度**；不證明不同 rollout 必然得到相似模型或相似最終表現，也不直接解釋遺忘與稀疏度的差異；不主張 reverse KL 一定不穩定、也不主張學生 rollout 一定較好。實驗應解讀為「與穩定性結果一致」，不是它的直接推論。作者另指出：rollout 機率 $\varepsilon$ 的變動可被局部梯度 $\log(1/\delta)$ 放大成 $\varepsilon\log(1/\delta)$ 的更新差；全域梯度截斷能限制實際梯度範數而緩解此失效模式，但不等於施加 Eq. 58 的條件。
- 〔本頁推導〕同一反例改用 forward KL：第二前綴 logit 梯度 $\pi_S-\pi_T=(\tfrac12-\delta)(1,-1)$，軌跡梯度範數 $\frac{\sqrt2}2(\tfrac12-\delta)$，更新差 $\varepsilon\frac{\sqrt2}2(\tfrac12-\delta)\le\frac{\sqrt2}4\varepsilon$，遠低於 Eq. 27 的 $2\sqrt2\varepsilon$（$B=1$）。reverse 的更新差在 $\log\frac{1-\delta}\delta=16$，即 $\delta=1/(1+e^{16})\approx1.125\times10^{-7}$ 時等於 $2\sqrt2\varepsilon$，之後更大。

#### 步驟 7：連續的 rollout 光譜 $\pi_\lambda$（Sec. 5.2，App. D.5，Eq. 75–80）

$$(\hat z_\lambda)_v=\tfrac12(\log\pi_S(v)+\log\pi_T(v))+\tfrac\lambda2(\log\pi_S(v)-\log\pi_T(v))=\tfrac{1+\lambda}2\log\pi_S(v)+\tfrac{1-\lambda}2\log\pi_T(v)\tag{Eq. 75–76}$$

- $\pi_\lambda=\mathrm{softmax}(\hat z_\lambda)$。$\lambda=-1$ 還原 $\pi_T$（OffPD），$\lambda=1$ 還原 $\pi_S$（OnPD），$\lambda=0$ 為兩者正規化後的幾何中點；$\lambda\lt-1$ 偏向教師覺得合理而學生不覺得的 token，$\lambda>1$ 反之。
- $\lvert\lambda\rvert>1$ 時某一係數變負，會不成比例地放大「兩個模型都認為不太可能、但被偏好的一方相對沒那麼不可能」的 token。防護（類似 Li et al., 2023 的 contrastive decoding）：
  - $r_v=\log\pi_S(v)-\log\pi_T(v)$；可信度遮罩 $M_S(v)=\mathbb I[\log\pi_S(v)\ge\max_u\log\pi_S(u)+\log\alpha]$，$M_T$ 同理（Eq. 77–78）。
  - 截斷後的加成 $b_S(v)=M_S(v)\min\{\max\{r_v,0\},c\}$，$b_T(v)=M_T(v)\min\{\max\{-r_v,0\},c\}$（Eq. 79–80）。
  - $\tilde z_\lambda=\hat z_\lambda$（$\lvert\lambda\rvert\le1$）；$\log\pi_S+\frac{\lambda-1}2b_S$（$\lambda>1$）；$\log\pi_T+\frac{-\lambda-1}2b_T$（$\lambda\lt-1$）；$\pi_\lambda=\mathrm{softmax}(\tilde z_\lambda)$。
  - 遮罩只管外插加成，遮罩外的 token 保留錨點策略的機率；保持 $\pi_{-1}=\pi_T$、$\pi_1=\pi_S$ 與區間邊界的連續性。實驗用 $\alpha=0.2$、$c=2\log3\approx2.2$；在 $\lambda\in[-2,2]$ 內，相對錨點策略的兩兩勝算放大最多 3 倍。
- 〔本頁推導〕3 倍的來由：$\lambda>1$ 時 $\tilde z_\lambda-\log\pi_S=\frac{\lambda-1}2b_S$，任兩 token 的勝算比被乘上 $\exp(\frac{\lambda-1}2(b_S(v)-b_S(u)))\le\exp(\frac{\lambda-1}2c)$，$\lambda=2$ 時為 $e^{\ln3}=3$。不加防護時同一倍數為 $\exp(\frac{\lambda-1}2(\max r-\min r))$，沒有上界。
- Fig. 4 確認此抽樣方式能改變生成軌跡上學生與教師的似然。

#### 步驟 8：光譜實驗結果（Sec. 5.2，Fig. 3；App. A.5，Fig. 15）

- Countdown-3、150 步。forward KL：整個光譜上 held-out 準確率都高於 80%，在學習率 1×10⁻⁵ 時變動僅 5.2 個百分點。OnPD＋forward KL 即使經過一段大量不連貫、高熵 rollout 的階段仍能有效學習（App. A.10），作者據此指出最終表現相近可能掩蓋很不同的最佳化路徑。
- reverse KL：平均表現隨 $\lambda$ 大幅變化，數個設定 seed 間變異大。低學習率時強烈受益於學生偏好的 rollout（$\lambda>0$），教師偏好的 rollout 表現變差；高學習率時仍不穩定，偶爾崩潰。
- 遺忘與稀疏度隨 rollout 的變化相對小，主要受學習率影響。
- Qwen2.5（Fig. 15，單一 seed 42）：forward KL 穩健；reverse KL 對教師偏好 rollout（$\lambda\lt0$）高度敏感，$\lambda\ge0$ 時大幅恢復並接近 forward KL。5×10⁻⁵ 下學生偏好 rollout 反而使 forward KL 遺忘愈來愈嚴重——作者把它當作「on-policy 不必然減少遺忘」的額外證據。

#### 步驟 9：準確率以外的差異（Sec. 5.3，Fig. 5–6，App. A.4、A.11、D.6–D.7）

- 輸出涵蓋度：pass@1 改善不一定帶來 pass@k 改善（Yue et al., 2025）。LR 1×10⁻⁵ 的 Countdown-3 上，$\lambda>0$ 時兩種 KL 的 pass@1 相近，但 forward KL 的 pass@10 明顯較高；Qwen 結果相同（Fig. 16）。作者結論：KL 方向比 rollout 更影響輸出涵蓋度。
- 泛化到較難任務：同一批 Countdown-3 checkpoint 不再訓練，直接評估 Countdown-4E（四個運算元，取自 1–10，溫度 0.5）。兩種 KL 方向的表現都隨光譜逐漸上升，較 on-policy 的 rollout（$\lambda>0$）比 off-policy 一致高 10–15%（論文原文「10 − 15% higher pass@k accuracy」，未寫明是百分點還是相對百分比；頁面照錄並註明，見 R3）。這是本文第一個 on-policy 明顯較好的設定，與 SFT／RL 泛化研究一致（Yuan et al., 2026 等）。
- 後續 RLVR（Countdown-4，300 步，GRPO＋DAPO loss，無 KL 懲罰，LoRA rank 256，RLVR 學習率 2×10⁻⁶，獎勵 2／0；24 次訓練，App. D.6）：LR 1×10⁻⁵ 蒸餾的 reverse-KL checkpoint 起初進步快（含起始準確率低的 OffPD），後來獎勵崩潰；OffPD＋forward KL（1×10⁻⁵）與 OffPD＋reverse KL（5×10⁻⁵，Fig. 26）穩定進步，訓練結束時最高，儘管起點接近 0%。結論：on-policy checkpoint 起初的泛化優勢沒有可靠地轉成 RLVR 後的優勢，與 Zhang et al.（2026a）一致。曲線是訓練獎勵，不是留出準確率；正確率＝平均獎勵的一半（App. D.6）。
- 教師風格轉移（App. A.11，Fig. 24，每條件 1 次訓練）：Qwen2.5-3B 教師帶正確示範並被要求只用西班牙文推理，學生 Qwen2.5-1.5B 在 Science 上蒸餾，GPT-5 判定回答語言。西班牙文比例（LR 1×10⁻⁵ / 5×10⁻⁵）〔圖讀值〕：OnPD·F-KL 90.6% / 100.0%；OnPD·R-KL 0.0% / 0.1%；OffPD·F-KL 82.6% / 99.9%；OffPD·R-KL 61.7% / 100.0%。作者的解釋（假說）：在學生產生的英文前綴上教師仍支持英文續寫，mode-seeking 的 reverse KL 可以對齊這個英文 mode 而不必涵蓋西班牙文；off-policy rollout 則即使用 reverse KL 仍會轉移風格。

#### 步驟 10：穩健性檢查（Sec. 6，App. A.1–A.3、A.10、A.12）

- **抽樣 KL**（App. A.2）：forward 抽 $y_n\sim\pi_T$，損失 $\hat L_{\text{F-KL}}=-\log\pi_S^\theta(y_n|x,y_{\lt n})$；reverse 抽 $y_n\sim\pi_S^\theta$，score-function 替代損失 $\hat L_{\text{R-KL}}=\mathrm{sg}[\log\pi_S^\theta(y_n|\cdot)-\log\pi_T(y_n|\cdot)]\log\pi_S^\theta(y_n|\cdot)$（sg＝stop-gradient）。這個額外 token 在每個位置獨立抽樣，不必等於 rollout 中的 token。結論不變：forward 穩健，reverse 敏感且脆弱，學習率主導遺忘與稀疏度（Fig. 9）。
- **移除梯度截斷**（App. A.1，Fig. 7–8）：主實驗以全域範數 1.0 截斷，且截斷前範數在每個記錄步都超過 1.0；reverse KL 的範數更大、更不穩、早期有尖峰。移除截斷後 forward KL 在較大學習率下仍有效（OffPD 尤其），reverse KL 明顯變差；遺忘與稀疏度仍主要由學習率決定。作者提出的可能解釋（明寫為 plausible explanation）：Adam 以 $\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ 更新，正規化抵銷均勻的梯度尺度變化；孤立的梯度尖峰會讓 $m_t$ 偏向非典型批次並膨脹較慢衰減的 $v_t$，壓抑之後的修正更新——對 OnPD 特別不利，因為早期的破壞性更新會改變之後產生 rollout 的策略。這個機制可以損害任務學習而不增加遺忘。
- **較長的推理**（App. A.3）：Qwen2.5-Math-1.5B-Instruct → Qwen2.5-1.5B-Instruct，Numina–MATH 2 萬題（MATH 訓練集 7,498 題＋NuminaMath 合成題 12,502 題），教師回答平均 622 tokens，1,000 步，每 100 步評估，回報軌跡上 MATH-500 最高值（作者註明此值可能偏樂觀）。未訓練 53.1%，選出的 checkpoint 為 53.3%–59.0%；低學習率時 OnPD＋reverse KL 最高（Fig. 10 標籤 59.0%）。學習率 1×10⁻⁵ 時 OOD 大致保持、稀疏度 92–93%；5×10⁻⁵ 時 OOD 由約 41.4% 降到 33.7–35.9%、稀疏度 63–70%。每條件只有 1 次訓練，作者把 OnPD 的優勢視為提示性的。OnPD 用 vLLM 生成時以截斷的重要性權重 $w(y)=\frac1{\lvert y\rvert}\sum_t\min(2,\exp[\log p_\theta(y_t|\cdot)-\log q(y_t|\cdot)])$ 修正抽樣分布差異（沿用 Shenfeld et al., 2026）。
- **低 forward KL 可能掩蓋高熵的 on-policy rollout**（App. A.10，Eq. 6）：$D_{\mathrm{KL}}(T\Vert S)=\mathrm{CE}(T,S)-H(T)$。Science 上 OnPD＋forward KL 的第 10–20 步，forward KL 變小，但教師熵與交叉熵都接近詞彙表尺度（12 nats ≈ 16.3 萬個等效 token 選擇）——小 KL 來自兩個大數幾乎抵消，不是有把握的預測。低熵區沿軌跡往後推進；作者推測這是自我強化的機制：早期 token 改善後，教師在後段前綴上才能給出有用的監督。第 50 步時 Science 平均 37%、MedReason 56%，OOD 只降 1.3 與 0.8 個百分點。作者推測差異來自取樣溫度（評估 0.5、訓練 1.0）：重新以 {0.3, 0.5, 0.7, 1.0} 取樣，準確率到 0.7 都高，1.0 時驟降，此時教師與學生熵都超過 7 nats。
- **生成溫度消融**（App. A.12，MedReason）：預設 OnPD 生成溫度 1.0、OffPD 0.5，互換後（KL 用的溫度不變）。forward KL：低學習率 OnPD 67.0→66.7%、OffPD 68.2→68.0%；高學習率 OnPD 55.7→63.0%、OffPD 62.8→60.7%。reverse KL：OnPD 69.3→67.5%（1×10⁻⁵）、59.3→36.0%（5×10⁻⁵，seed 間變異大）；OffPD 5.7→27.0%（1×10⁻⁵）、46.0→60.3%（5×10⁻⁵）。作者結論：OnPD 與 OffPD 可達到的最佳表現不受生成溫度影響。

#### 步驟 11：實作細節（App. D.3 Table 3、D.5、D.6）

| 項目 | 值 |
| --- | --- |
| 最佳化步數 | 150（主比較用第 150 步的 checkpoint） |
| 最佳化器 | Fused AdamW，β₁ = 0.9，β₂ = 0.999，ε = 10⁻⁸，weight decay 0 |
| 學習率排程 | 常數，前 10 步線性 warm-up |
| 批次 | 每裝置 1、梯度累積 32 → 每步 32 個 prompt；每個 prompt 1 條軌跡 |
| 梯度截斷 | 全域範數 1.0 |
| 溫度 | 學生 1.0（抽樣與 KL）、教師 0.5（抽樣與 KL）；top-p 1.0 |
| 最大長度 | prompt 2,048；completion 512（Countdown-3）／1,024（Science、MedReason） |
| 損失 | 每個非 padding 位置在全詞彙上算 KL，對 completion token 取平均，再對樣本平均 |
| gradient checkpointing | on-policy 開啟、off-policy 關閉（僅為記憶體管理） |
| 光譜 | $\alpha=0.2$，$c=2\log3$，$\lambda\in[-2,2]$ |
| 評估 | 每資料集前 200 題，學生溫度 0.5、top-p 1.0，每題 1 個回答 |

常見錯誤（頁面 S11 逐條列出）：
1. 把 OnPD 與 reverse KL 綁在一起：rollout 決定走過哪些前綴，KL 方向決定在這些前綴上怎麼比較；全詞彙 KL 下兩者獨立（Sec. 3）。
2. 以為 OnPD reverse KL 的更新就是序列層級 reverse KL 的梯度：stop-gradient 使它只是半梯度（App. C.5）。
3. 在不同學習率下比較 OnPD 與 OffPD：學習率主導遺忘與稀疏度（Sec. 4.2），比較必須在相同學習率下配對。
4. $\lvert\lambda\rvert>1$ 外插不加防護：會放大兩個模型都認為不太可能的 token；用遮罩 $\alpha=0.2$ 與截斷 $c=2\log3$（App. D.5）。
5. 把低 forward KL 當成 rollout 品質好：$D_{\mathrm{KL}}=\mathrm{CE}-H$，兩個大數相消也會得到小 KL（App. A.10）。
6. 抽樣版 reverse KL 少了 stop-gradient：替代損失中的 log 比必須 sg，否則不是 reverse KL 梯度的不偏估計（App. A.2）。
7. 只看 pass@1：forward KL 的 pass@10 優勢與 on-policy 在 Countdown-4E 的泛化優勢都不會出現在同任務 pass@1（Sec. 5.3）。

#### 步驟 12：限制（Sec. 7）

學生模型最多 1.5B 參數，任務推理長度最多約 2,000 tokens；教師固定。作者建議未來的 on-policy 蒸餾研究把 OffPD 列為標準基準；並認為 SFT 與 RL 之間觀察到的多數差異難以只歸因於 rollout 策略，目標函數、獎勵、監督密度與最佳化程序值得進一步研究。

---

## 2. 學習目標（驗收測試）

每個參考答案必須以一段完整的段落寫在指定章節的正文（`<p>`）中，不能只出現在元件、表格、圖或 `<details>` 答案框。章節結尾的 `<details>`「檢查理解」照搬題目與答案。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 用 rollout 策略定義兩種蒸餾 | Eq. 1 中 OffPD 與 OnPD 各怎麼設定 $\rho$？OnPD 的梯度如何處理抽樣？ | OffPD 取 $\rho=\pi_T$（教師產生軌跡），OnPD 取 $\rho=\pi_S^\theta$（學生產生軌跡）；OnPD 的梯度不穿過抽樣，軌跡視為固定（stop-gradient），得到的是半梯度（Sec. 3、Eq. 19） | S1 |
| LO2 | 區分 forward 與 reverse KL | 兩個方向各以誰的機率加權？各常被稱為什麼？ | forward KL $\sum_v\pi_T\log(\pi_T/\pi_S)$ 以教師加權，懲罰學生沒涵蓋教師的高機率 token，稱 mode-covering；reverse KL $\sum_v\pi_S\log(\pi_S/\pi_T)$ 以學生加權，懲罰學生把機率放在教師認為不太可能的 token，稱 mode-seeking（Eq. 2–3） | S2 |
| LO3 | 說明為何 rollout 與 KL 方向可以獨立 | 慣例配對的兩個來源是什麼？為什麼本文可以拆開？ | 來源一：教師 rollout＋forward、學生 rollout＋reverse 是序列層級 KL 的兩種鏈式分解（Prop. C.4），但只在目標值層級成立；來源二：用單一抽樣 token 估計 KL 時配對自然。本文在全詞彙上算 KL，KL 方向不限制 rollout，且兩個來源都不代表慣例配對表現較好（Sec. 3） | S2 |
| LO4 | 描述受控實驗設計 | 獨立改變哪三個因素？遺忘與稀疏度怎麼量？ | rollout（OnPD／OffPD）、KL 方向（forward／reverse）、學習率（1×10⁻⁵／5×10⁻⁵）；Llama-3.1-8B → Llama-3.2-1B，150 步、3 seeds。遺忘＝7 個 OOD 基準平均分數的下降；稀疏度＝參數變化量絕對值小於 $\tau=10^{-6}$ 的參數比例（Sec. 4.1、App. D.4） | S3 |
| LO5 | 讀出主結果 | 最佳準確率、遺忘、稀疏度在 OnPD／OffPD 與兩個學習率下各是多少？ | 最佳平均準確率 OnPD 72%、OffPD 73%；forward KL 71–73%、reverse KL 35%–72%。遺忘：低學習率最多 1.3 個百分點，高學習率 11.2–14.0 個百分點。稀疏度：低學習率 85.3–89.6%，高學習率 51.9–60.0%，每組配對中 OffPD 至少與 OnPD 一樣稀疏（Sec. 4.2） | S4 |
| LO6 | 寫出 logit 梯度並說明不對稱 | forward 與 reverse KL 對學生 logit 的導數是什麼？各在何時變大或消失？ | forward：$\pi_S(v)-\pi_T(v)$，每座標在 $[-1,1]$，兩分布不同就非零；reverse：$\pi_S(v)(\log\frac{\pi_S(v)}{\pi_T(v)}-D_{\text{R-KL}})$，學生機率趨近 0 時消失（難以找回遺漏的教師 mode），學生給可觀機率而教師機率趨近 0 時可任意大（Eq. 4–5、Sec. 5.1） | S5 |
| LO7 | 陳述兩個 rollout 穩定性定理 | Theorem C.1 的上界是什麼？Theorem C.2 的反例給出多大的更新差？ | C.1：$\lVert\Delta\bar\nabla\rVert_2\le2\sqrt2B\,\mathbb E_x[\mathrm{TV}]$（Eq. 27）；C.2：兩 token 反例中 TV $=\varepsilon$，更新差 $=\frac{\varepsilon\sqrt2}8\log\frac{1-\delta}\delta$，$\delta\to0$ 時發散，所以不存在與分布無關的常數；log 比範圍受 $R$ 控制時恢復上界 $2BR\,\mathbb E_x[\mathrm{TV}]$（Eq. 57、59） | S6 |
| LO8 | 說出定理不證明什麼 | 這些定理能否推出「forward KL 下不同 rollout 得到相同最終表現」？ | 不能。定理只控制一步的期望半梯度，上界依賴軌跡分布距離與 Jacobian 上界 $B$；不證明不同 rollout 得到相似模型或表現，也不解釋遺忘與稀疏度。實驗結果只能解讀為與定理一致（App. C.2、C.4） | S6 |
| LO9 | 定義 rollout 光譜 | $\pi_\lambda$ 在 $\lambda=-1,0,1$ 是什麼？$\lvert\lambda\rvert>1$ 時如何防護？ | $\hat z_\lambda=\frac{1+\lambda}2\log\pi_S+\frac{1-\lambda}2\log\pi_T$；$\lambda=-1$ 為 $\pi_T$（OffPD），$\lambda=1$ 為 $\pi_S$（OnPD），$\lambda=0$ 為正規化幾何中點。$\lvert\lambda\rvert>1$ 時只對被偏好方機率至少為其最大值 $\alpha=0.2$ 倍的 token 加上截斷到 $c=2\log3$ 的 log 比加成，使 $\lambda\in[-2,2]$ 內相對錨點的勝算放大最多 3 倍（Eq. 75–80、App. D.5） | S7 |
| LO10 | 讀出光譜實驗 | 光譜上 forward 與 reverse KL 的表現如何？遺忘與稀疏度呢？ | forward KL 準確率全光譜高於 80%，LR 1×10⁻⁵ 時只變動 5.2 個百分點；reverse KL 隨 $\lambda$ 大幅變化，低學習率時受益於學生偏好的 rollout（$\lambda>0$），高學習率時不穩定、偶爾崩潰；遺忘與稀疏度受 rollout 影響小，主要由學習率決定（Sec. 5.2、Fig. 3） | S8 |
| LO11 | 說出 on-policy 有幫助的情形與其限度 | 本文在哪裡觀察到 on-policy 的優勢？RLVR 後呢？ | Countdown-4E（較難的 OOD 任務）上，兩種 KL 方向下較 on-policy 的 rollout（$\lambda>0$）的 pass@k 一致高 10–15%；OnPD＋reverse KL 幾乎不轉移教師的西班牙文風格（0.0%／0.1%）。但後續 300 步 RLVR 後，最強的持續表現來自 off-policy checkpoint，on-policy 的初始優勢沒有可靠保留（Sec. 5.3） | S9 |
| LO12 | 說出三項穩健性檢查的結論 | 改用抽樣 KL、移除梯度截斷、改用長推理任務後，結論如何？ | 抽樣 KL：forward 仍穩健、reverse 仍敏感，學習率仍主導遺忘與稀疏度；移除截斷：reverse KL 明顯變差、forward 在較大學習率仍有效，沒有一致的 OnPD–OffPD 差距；Numina–MATH（教師回答平均 622 tokens）：低學習率時 OnPD＋reverse KL 的 MATH-500 最高（59.0%），但每條件只有 1 次訓練，作者視為提示性（Sec. 6、App. A.1–A.3） | S10 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），圖版加寬到最大 1080px；章節編號 JetBrains Mono（`01 · …`），章節標題用陳述句。視覺記號：一個 2×2 的「設計格」（列＝rollout OnPD／OffPD，欄＝KL F／R），在 hero、S2、S4 重複出現；學習率以格內左右兩半（淺色＝1×10⁻⁵、深色＝5×10⁻⁵，另附文字「LR 1e-5／5e-5」）表示。

| # | id | 標題 | 內容（依序；每個符號先定義再使用） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈On-policy 或 off-policy：蒸餾動態的受控比較〉 | 副標：「作者在相同流程下分別改變 rollout 來源、KL 方向與學習率，發現 rollout 來源不是主要因素。」三張事實卡（下方）；連結 arXiv、`../index.html#paper-136`、筆記總覽 `odd-lineage-hub.html` | 1 螢幕 |
| S1 | `setup` | 01 · 蒸餾目標對 rollout 走過的前綴取平均 | 定義 token、詞彙表、前綴、下一個 token 分布、學生／教師、rollout 策略 $\rho$；Eq. 1；OffPD／OnPD；stop-gradient 與半梯度（含 LO1 答案段落）；強到弱蒸餾的動機兩句。末尾符號表。延伸閱讀：`knowledge-distillation-foundations.html`、`odd-onpd-vs-offpd-setup.html`、`odd-strong-to-weak-distillation-practice.html` | 400–500 字 |
| S2 | `kl` | 02 · KL 方向與 rollout 策略是兩個獨立選擇 | Eq. 2–3、mode-covering／seeking（LO2 段落）；慣例配對的兩個來源、Prop. C.4 只在目標值層級、全詞彙 KL 使兩者獨立（LO3 段落）；2×2 設計格。延伸閱讀：`odd-token-kl-direction-gradients.html` | 400–500 字 |
| S3 | `design` | 03 · 實驗只改三個因素 | 1.4 步驟 3 的表（模型、任務、教師、指標、溫度）；遺忘與稀疏度的定義（LO4 段落）；為什麼每個任務各訓練教師。延伸閱讀：`odd-on-policy-forgetting-sparsity-claims.html` | 300–400 字 + 表 |
| S4 | `main` | 04 · rollout 策略沒有一致優勢，學習率主導遺忘與稀疏度 | Sec. 4.2 四段結果（LO5 段落）、Fig. 2 學習率掃描、cliff；C4 在此；C4 下方一段〔本頁推導〕配對差摘要（步驟 4 末項），並寫明「配對差是描述統計，不是因果歸因」。延伸閱讀：`odd-learning-rate-adam-clipping-rlvr.html` | 450–550 字 + C4 |
| S5 | `grad` | 05 · forward KL 的 logit 梯度有界，reverse KL 沒有 | 定義 logit、softmax 導數（Eq. 9）；Eq. 4–5（LO6 段落）；mode-seeking 的梯度解釋；作者的推論「reverse 預期更受益於 on-policy」以「作者認為」標示；座標和為 0〔本頁推導〕；C1。延伸閱讀：`odd-token-kl-direction-gradients.html` | 400–500 字 + C1 |
| S6 | `bound` | 06 · rollout 相近時 forward 更新也相近，reverse 不一定 | 先定義 Jacobian、$B$、TV、半梯度；Theorem C.1（Eq. 27，證明三步）；Theorem C.2 反例（Eq. 48–57）；Corollary C.3（Eq. 58–59）（LO7 段落）；定理範圍（LO8 段落）；forward 版反例〔本頁推導〕；C2 | 550–650 字 + C2 |
| S7 | `spectrum` | 07 · 光譜 $\pi_\lambda$ 在教師與學生之間連續內插與外插 | Eq. 75–76、$\lambda=-1,0,1$；外插問題；遮罩、截斷、3 倍上界〔本頁推導說明〕（LO9 段落）；C3 | 400–500 字 + C3 |
| S8 | `spectrum-results` | 08 · 光譜上 forward KL 穩定，reverse KL 偏好學生 rollout | Fig. 3 結果（LO10 段落）、Qwen Fig. 15、高熵 OnPD 路徑一句（連到 S10） | 250–350 字 |
| S9 | `beyond` | 09 · on-policy 資料改善較難任務的泛化，但 RLVR 後不保留 | pass@1 vs pass@10；Countdown-4E；RLVR；風格轉移（4 組數字表）（LO11 段落）；作者假說另段標示。延伸閱讀：`odd-sft-vs-rl-generalization-claims.html`、`context-distillation-self-distillation.html` | 450–550 字 + 1 表 |
| S10 | `robust` | 10 · 三項穩健性檢查不改變主要結論 | 抽樣 KL（兩個估計式）、梯度截斷（Adam 解釋標為作者的可能解釋）、Numina–MATH（LO12 段落）；A.10 的 CE − H 相消；A.12 溫度消融摘要。延伸閱讀：`odd-learning-rate-adam-clipping-rlvr.html` | 500–600 字 |
| S11 | `impl` | 11 · 實作設定與常見錯誤 | 步驟 11 表、7 條常見錯誤 | 300–400 字 + 表 |
| S12 | `history` | 12 · 研究脈絡 | 時間線（6.4）＋一段文字：Hinton 2015 logit 蒸餾 → Kim & Rush 2016 序列蒸餾 → Agarwal 2023 on-policy 蒸餾 → Xu 2024 交錯取樣 → 2025–2026 對 on-policy 好處的主張（遺忘、稀疏、泛化）與學習率研究 → 本文的受控檢驗 | 時間線 + 100–150 字 |
| S13 | `next` | 13 · 限制與延伸閱讀 | Sec. 7 限制與建議；延伸閱讀清單（5.6） | 200 字 + 連結 |
| — | — | 頁尾 | 引用、arXiv；`#pm-bar` | — |

Hero 三張事實卡（卡內只用卡內定義的詞）：
1. 「蒸餾（distillation）讓小模型（學生）模仿大模型（教師）每一步的下一個字詞機率。訓練用的文本可以由教師寫（off-policy，OffPD），也可以由學生自己寫（on-policy，OnPD）。」
2. 「作者固定其他設定，只改變文本由誰寫、比較機率的方式（KL 散度的兩個方向：以教師加權的 forward、以學生加權的 reverse）與學習率。由誰寫文本對最終準確率沒有一致的影響：最佳平均準確率 OnPD 72%、OffPD 73%。」
3. 「學習率決定有多少參數幾乎沒動（變化小於 10⁻⁶ 的參數比例）：學習率 1×10⁻⁵ 時 85.3–89.6%，5×10⁻⁵ 時 51.9–60.0%。」

章節目錄：≥ 1024px 左側 sticky 窄欄；< 1024px 頂部橫向捲動列（沿用參考頁）。

---

## 4. 互動元件規格

共通規則：
- 每個圖版第一行 `<p class="plate-lead">` 寫出該玩具的內建假設（下列「假設行」照錄）。元件不顯示任何寫死的「預期觀察」文字；說明性的觀察寫在圖版下方正文，以「預設值下」或「預設按鈕下」開頭，只描述預設或預設按鈕狀態。
- 每個元件只有一個 `aria-live="polite"` 讀數區，500ms debounce（最後一次 `input` 後 500ms 或 `change` 時才更新）；SVG 的 `aria-label` 同時更新。
- 沒有 hover-only 資訊：所有數值都在讀數區或圖上直接標註。
- 所有計算都是 O(數十) 次運算，在主執行緒同步完成，沒有需要移到 worker 的重運算（C3 的曲線 81 點 × 6 token）。
- 390px：圖版內距 12px，SVG 寬度＝容器寬（重畫於 resize，debounce 150ms），控制列在圖下方堆疊，觸控區 ≥ 44px，圖上文字 12–13px，不隨 viewBox 縮放。

### C1 · 一個前綴上的 forward／reverse KL 與 logit 梯度（S5）

- **教的概念**：Eq. 4–5；forward 梯度座標在 $[-1,1]$；reverse 梯度在教師機率趨近 0 時變大、在學生機率趨近 0 時消失。
- **假設行**：「詞彙表 5 個 token（A–E）；只有 token A 的機率可調，其餘機率依固定比例 0.4 : 0.3 : 0.2 : 0.1 分給 B–E，學生與教師相同〔示範值〕；直接 logit 參數化（Jacobian＝單位矩陣）。」
- **輸入**：
  - 滑桿「學生給 A 的機率 $\pi_S(A)$」：滑桿值 $u\in[-6,\log_{10}0.9]$，步長 0.01，$\pi_S(A)=10^u$；顯示三位有效數字。
  - 滑桿「教師給 A 的機率 $\pi_T(A)$」：同上。
  - 預設按鈕（同時設定兩者的精確值，不經滑桿捨入）：「預設」（0.5, 0.01）、「兩者一致」（0.3, 0.3）、「教師幾乎不給 A」（0.5, 1e-6）、「學生漏掉 A」（1e-6, 0.6）。載入時套用「預設」。
- **計算**（函式 `klParts(aS, aT)`，頁面唯一定義）：
  - $\pi(v)$：$\pi(A)=a$，$\pi(B..E)=(1-a)\cdot[0.4,0.3,0.2,0.1]$。
  - $r_v=\ln\pi_S(v)-\ln\pi_T(v)$；$D_F=\sum\pi_T(\ln\pi_T-\ln\pi_S)$；$D_R=\sum\pi_Sr_v$。
  - $g_F=\pi_S-\pi_T$；$g_R=\pi_S\odot(r-D_R)$；$\lVert g\rVert_2$；$R=\max r-\min r$。
  - 保護：$a\in[10^{-6},0.9]$，所有機率 ≥ $10^{-7}$，無 log(0)；自檢 $\lvert\sum g_F\rvert\lt10^{-12}$、$\lvert\sum g_R\rvert\lt10^{-9}$。
- **顯示**：兩個長條圖面板（≥ 768px 左右並排；< 768px 上下堆疊），每面板 5 根長條（A–E，中央零線）。forward 面板縱軸固定 $[-1,1]$，標「每座標的界 ±1」虛線；reverse 面板縱軸自動取 $\pm\max(1,\max\lvert g_R\rvert)\times1.1$，並在軸頂用文字標出目前刻度上限（避免讀者以為兩面板同尺度）。長條：forward 實心、reverse 斜線填充；每根長條末端標數值（兩位小數）。下方小表：$D_F$、$D_R$（nats）、$\lVert g_F\rVert_2$ 與界 $\sqrt2$、$\lVert g_R\rVert_2$ 與界 $R$（Eq. 65）、token A 的兩個梯度座標。
- **讀數區（live）**：「π_S(A) = …，π_T(A) = …；forward KL = …，reverse KL = …；A 的 logit 梯度 forward …、reverse …」。
- **圖版底部來源行**：「讀數由本頁依 Eq. 2–5、Eq. 65 即時計算；座標和為 0 的檢查為本頁推導。」
- **正文觀察（圖版下方）**：「教師幾乎不給 A」時 reverse 的 A 座標 3.45、超出 forward 的界；「學生漏掉 A」時 forward 的 A 座標 −0.60（把 A 往上推），reverse 的 A 座標約 0——這就是 mode-seeking 的梯度面。

### C2 · rollout 相差 ε 時，更新最多差多少（S6）

- **教的概念**：Theorem C.1 的線性上界、Theorem C.2 反例的發散、Corollary C.3 的條件式上界。
- **假設行**：「論文 Theorem C.2 的兩 token 反例：學生 (1/2, 1/2)、教師 (δ, 1−δ)、直接 logit 參數化（B = 1）；rollout 以機率 ε 走到這個前綴。forward 版數值為本頁用同一反例推導。」
- **輸入**：滑桿 $\log_{10}\delta\in[-12,\log_{10}0.49]$，步長 0.01，預設 $\delta=0.01$（預設值直接設定，不經捨入）；滑桿 $\varepsilon\in[0.01,1]$，步長 0.01，預設 0.1。
- **計算**（`boundParts(delta, eps)`）：$R=\mathrm{log1p}(-\delta)-\ln\delta$；reverse 差 $=\varepsilon\frac{\sqrt2}8R$（Eq. 57）；forward 差 $=\varepsilon\frac{\sqrt2}2(\tfrac12-\delta)$〔本頁推導〕；forward 界 $2\sqrt2\varepsilon$（Eq. 27，$B=1$）；reverse 條件界 $2R\varepsilon$（Eq. 59，$B=1$）。保護：$\delta\le0.49$ 使 $R>0$；用 `log1p` 避免 $\delta$ 很小時 $1-\delta$ 的捨入。
- **顯示**：主圖橫軸 $\log_{10}\delta$（−12 到 −0.31，刻度每 2 個十進位一格；390px 時每 4 個），縱軸「更新差 ÷ ε」用對數刻度 $[10^{-3},10]$（刻度 0.001, 0.01, 0.1, 1, 10），低於 $10^{-3}$ 的部分截在底端。三條線：reverse 差（`--accent` 實線＋圓點標記，直接標籤「reverse（Eq. 57）」）、forward 差（`--c-par` 長虛線＋方形標記，「forward（本頁推導）」）、forward 通用界 $2\sqrt2\approx2.83$（`--ink-3` 點線水平線，「forward 的界（Eq. 27）」）。目前 δ 用垂直細線標示，reverse 曲線與水平界的交點（δ ≈ 1.1×10⁻⁷）以空心圓標出並附文字。右側（< 768px 時在圖下）讀數表：R、reverse 差、reverse 條件界、forward 差、forward 界。
- **讀數區（live）**：「δ = …，ε = …；reverse 更新差 …，forward 更新差 …（界 …）」。
- **來源行**：「Eq. 27、53、57、59 即時計算；forward 版反例與交點為本頁推導。」
- **正文觀察**：預設值下 reverse 差 0.0812、forward 差 0.0346；δ 每小 10 倍，reverse 差增加 $\varepsilon\frac{\sqrt2}8\ln10\approx0.407\varepsilon$〔本頁推導〕，forward 差趨近 $\frac{\sqrt2}4\varepsilon$ 後不再增加。

### C3 · rollout 光譜 π_λ（S7）

- **教的概念**：Eq. 75–80；$\lambda=\pm1$ 還原兩個錨點；外插時遮罩與截斷的作用；3 倍勝算上界。
- **假設行**：「單一前綴、6 個 token（A–F）、學生與教師分布固定〔示範值〕；不模擬訓練，只顯示抽樣分布 π_λ。token F 是兩個模型都認為不太可能、但學生相對較不排斥的 token。」
- **示範分布**：$\pi_S=[0.45,0.25,0.15,0.08,0.05,0.02]$，$\pi_T=[0.03,0.57,0.25,0.10,0.0499,0.0001]$（兩者和皆為 1）。
- **輸入**：滑桿 $\lambda\in[-2,2]$，步長 0.05，預設 0；切換「使用遮罩與截斷（App. D.5）」，預設開；按鈕「λ = −1（OffPD）」「λ = 0」「λ = 1（OnPD）」「λ = 2」。
- **計算**（`spectrum(lam, safe)`，與 `check.py` 同；$\alpha=0.2$、$c=2\ln3$）：依 1.4 步驟 7 算 $\tilde z$（`safe=false` 時一律用 $\hat z$）；softmax 先減最大值。輸出 $\pi_\lambda$；單步幾何平均機率〔本頁定義〕$G_S=\exp(\sum_v\pi_\lambda(v)\ln\pi_S(v))$、$G_T$ 同理（Fig. 4 是軌跡上的幾何平均 token 機率，這裡只有一步）；$\lvert\lambda\rvert>1$ 時「相對錨點的最大勝算放大倍數」$A=\exp(\max_vd_v-\min_vd_v)$，$d_v=\tilde z_v-\ln(\text{錨點})_v$，錨點 $\lambda>1$ 為 $\pi_S$、$\lambda\lt-1$ 為 $\pi_T$；$\lvert\lambda\rvert\le1$ 顯示「—（內插區）」。
- **顯示**：
  - 上圖：每個 token 三根並排細長條 $\pi_S$（實心）、$\pi_T$（空心外框）、$\pi_\lambda$（`--accent` 斜線），圖例用填充樣式＋文字；縱軸 [0, 0.8]，每根標兩位小數（< 0.01 標「<0.01」）。390px：token 改為 6 列水平長條（每列三條），避免擁擠。
  - 下圖：橫軸 λ ∈ [−2, 2]（標「← 教師偏好」「學生偏好 →」與 −1 OffPD、1 OnPD 刻度），兩條曲線 $G_S$（`--c-par` 實線＋「G_S」標籤）、$G_T$（`--c-perp` 虛線＋「G_T」標籤），曲線點以 81 個 λ（步長 0.05）計算，隨切換重算；目前 λ 用垂直線標示。
  - 讀數表：λ、$G_S$、$G_T$、勝算放大倍數、token F 的 $\pi_\lambda$。
- **讀數區（live）**：「λ = …，防護 開／關；π_λ(A…F) = …；最大勝算放大 …」。
- **來源行**：「依 Eq. 75–80（α = 0.2，c = 2 ln 3）即時計算；G_S、G_T 與 3 倍上界的推導為本頁內容；分布為示範值。」
- **正文觀察**：λ = 2 時開啟防護，A 的加成被截斷在 c，最大勝算放大正好 3.00，F 被遮罩擋下（π_λ(F) = 0.0105）；關閉防護時 F 升到 0.1164、放大 21.35 倍——這是 App. D.5 要避免的「放大兩方都不看好的 token」。

### C4 · 只改一個因素時的配對差（S4）

- **教的概念**：Sec. 4 的論證方式——在其他兩個因素固定下比較；學習率對稀疏度的配對差遠大於 rollout；rollout 對 ID 準確率的影響取決於 KL 方向（呼應 Sec. 5）。
- **假設行**：「數值是 Fig. 1 長條上的標籤（三個資料集平均、3 seeds；準確率標籤為整數）〔圖讀值〕。配對差＝只改一個因素、另兩個固定時的絕對差（百分點）；這是描述統計，不是變異數分解或因果歸因。」
- **資料（頁面常數 `FIG1`）**：ACC 與 SPA 兩張表，鍵 (rollout, KL, LR)，值見 1.4 步驟 4。
- **輸入**：分段按鈕「指標」：ID 準確率／更新稀疏度（預設 稀疏度）；點擊或以鍵盤（Tab＋Enter）選取任一點。
- **計算**（`pairDiffs(metric, factor)`）：對 factor ∈ {rollout, KL, LR}，列舉另兩個因素的 4 種組合，回傳 [(組合, 值1, 值2, |差|)]，差四捨五入到 0.1。
- **顯示**：三列（「改 rollout」「改 KL 方向」「改學習率」），共用橫軸 0–40 pp（刻度 0, 10, 20, 30, 40）。每列 4 個點，形狀區分被固定的組合（圓、方、三角、菱形，圖例附文字），點旁直接標短碼（例如「R·1e-5」「On·F」）；重疊的點垂直錯開 10px。下方表格列出選取列的 4 組：方法名在前、數值在後（例如「OnPD·R-KL·LR 1e-5：72 → OffPD：35，差 37」）。390px：三列各自一張小圖（寬 = 容器寬），表格改為堆疊列表。
- **讀數區（live）**：選取點時宣告「改 rollout，固定 R-KL 與 LR 1e-5：OnPD 72，OffPD 35，差 37 個百分點」。
- **來源行**：「數值讀自 Fig. 1；配對差為本頁計算。」
- **正文觀察**（圖下段落，對應步驟 4 末項）：稀疏度的學習率配對差 29.3–33.4 pp，rollout 1.1–7.6 pp，KL 0.5–6.6 pp；ID 準確率改 rollout 的配對差在 forward KL 下為 1 與 1 pp，在 reverse KL 下為 37 與 14 pp。

---

## 5. 驗收數值表

全部由 `verify/check.py` 產生（與頁面同式；`python3 check.py` 已執行至 `all asserts passed`）。

### C1（容許誤差 ±0.0005）

| 預設 | π_S(A) | π_T(A) | forward KL | reverse KL | forward g_A | reverse g_A | ‖g_F‖₂ | ‖g_R‖₂ | R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 預設 | 0.5 | 0.01 | 0.6371 | 1.6145 | 0.4900 | 1.1488 | 0.5587 | 1.3098 | 4.5951 |
| 兩者一致 | 0.3 | 0.3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 教師幾乎不給 A | 0.5 | 1e-6 | 0.6931 | 6.2146 | 0.5000 | 3.4539 | 0.5701 | 3.9380 | 13.8155 |
| 學生漏掉 A | 1e-6 | 0.6 | 7.6163 | 0.9163 | −0.6000 | −0.0000 | 0.6841 | 0.0000 | 14.2210 |
| （極端，selfTest 用） | 0.9 | 1e-6 | 2.3026 | 12.1089 | 0.9000 | 1.4411 | 1.0262 | 1.6432 | 16.0127 |

（「學生漏掉 A」的 reverse g_A 真值約 −1.3×10⁻⁵，顯示為 −0.0000；頁面以兩位小數顯示為 0.00。）

### C2（ε = 0.1；容許誤差 ±0.0001）

| δ | R | reverse 差 | forward 差 | forward 界 | reverse 條件界 |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 1.0986 | 0.0194 | 0.0177 | 0.2828 | 0.2197 |
| 0.01（預設） | 4.5951 | 0.0812 | 0.0346 | 0.2828 | 0.9190 |
| 1e-4 | 9.2102 | 0.1628 | 0.0353 | 0.2828 | 1.8420 |
| 1e-8 | 18.4207 | 0.3256 | 0.0354 | 0.2828 | 3.6841 |
| 1e-12 | 27.6310 | 0.4885 | 0.0354 | 0.2828 | 5.5262 |

交點：$\delta=1/(1+e^{16})=1.1254\times10^{-7}$，此時 reverse 差/ε = 2.828427 = $2\sqrt2$（容許 ±1e-6）。

### C3（容許誤差 ±0.0005；π_λ 依 A–F 順序）

| λ | 防護 | π_λ | G_S | G_T | 勝算放大 |
| --- | --- | --- | --- | --- | --- |
| −2 | 開 | 0.0220 0.6313 0.2367 0.0733 0.0366 0.0001 | 0.1946 | 0.3537 | 1.5100 |
| −2 | 關 | 0.0057 0.6362 0.2386 0.0826 0.0368 0.0000 | 0.1905 | 0.3645 | 21.3542 |
| −1.5 | 開 | 0.0258 0.6015 0.2439 0.0859 0.0429 0.0001 | 0.1896 | 0.3350 | 1.2288 |
| −1 | 任一 | 0.0300 0.5700 0.2500 0.1000 0.0499 0.0001 | 0.1844 | 0.3157 | — |
| 0 | 任一 | 0.1403 0.4558 0.2338 0.1080 0.0603 0.0017 | 0.1925 | 0.2192 | — |
| 1 | 任一 | 0.4500 0.2500 0.1500 0.0800 0.0500 0.0200 | 0.2416 | 0.0867 | — |
| 1.5 | 開 | 0.5863 0.1881 0.1128 0.0602 0.0376 0.0150 | 0.2818 | 0.0667 | 1.7321 |
| 1.5 | 關 | 0.6228 0.1431 0.0928 0.0532 0.0352 0.0529 | 0.2675 | 0.0447 | 4.6211 |
| 2 | 開 | 0.7105 0.1316 0.0789 0.0421 0.0263 0.0105 | 0.3244 | 0.0525 | 3.0000 |
| 2 | 關 | 0.7175 0.0682 0.0478 0.0295 0.0206 0.1164 | 0.2593 | 0.0219 | 21.3542 |

不變式：λ = ±1 時 π_λ 與錨點完全相同（誤差 < 1e-12）；防護開時任何 λ 的勝算放大 ≤ 3（+1e-12）。

### C4（精確值，四捨五入到 0.1）

| 指標 | 改的因素 | 4 組配對（固定組合：值1 → 值2，差） |
| --- | --- | --- |
| 準確率 | rollout（On→Off） | F·1e-5：72→71，1；F·5e-5：72→73，1；R·1e-5：72→35，37；R·5e-5：35→49，14 |
| 準確率 | KL（F→R） | On·1e-5：72→72，0；On·5e-5：72→35，37；Off·1e-5：71→35，36；Off·5e-5：73→49，24 |
| 準確率 | LR（1e-5→5e-5） | On·F：72→72，0；On·R：72→35，37；Off·F：71→73，2；Off·R：35→49，14 |
| 稀疏度 | rollout | F·1e-5：85.3→88.8，3.5；F·5e-5：51.9→59.5，7.6；R·1e-5：88.5→89.6，1.1；R·5e-5：58.5→60.0，1.5 |
| 稀疏度 | KL | On·1e-5：85.3→88.5，3.2；On·5e-5：51.9→58.5，6.6；Off·1e-5：88.8→89.6，0.8；Off·5e-5：59.5→60.0，0.5 |
| 稀疏度 | LR | On·F：85.3→51.9，33.4；On·R：88.5→58.5，30.0；Off·F：88.8→59.5，29.3；Off·R：89.6→60.0，29.6 |

---

## 6. 實作計畫

### 6.1 沿用與檔案

- 單一檔案 `tut/136/build/index.html`；上線為 `tutorials/on-policy-vs-off-policy-distillation.html`。
- **直接沿用** `tutorials/prediction-targets-k-diff.html` 的外殼：`<head>`（Google Fonts、KaTeX 0.16.9 cdnjs＋auto-render、分隔符）、`:root` tokens 與深色模式覆寫（含 `--c-par`、`--c-perp` 與建置紀錄中調整後的 `--ink-3` 深色值 #9C978C）、appbar 與主題切換（`pm-theme`）、`#pm-bar`、`.toc`、`.ch`／`.ch-num`、`.prose`、`.eq`／`.eq-n`、`.plate`／`.plate-h`／`.plate-lead`／`.plate-f`、`.readout`、`.seg`、`.chip`、`.rng`、`.tag`（「本頁推導」pill）、`.check` details、`.tscroll` 表格容器、SVG 輔助 `el`／`scaleLin`／`scaleLog`／`linePath`、時間線 `renderTimeline` 與其 CSS、`selfTest` 外框（`?test`，輸出 `selfTest: N passed, M failed` 到 console、`document.title` 與 `#selftest`，並檢查 `scrollWidth <= innerWidth`）。
- 連結：論文 `../index.html#paper-136`；arXiv `https://arxiv.org/abs/2609.35259`；筆記 `<slug>.html`。
- 頁尾聲明：「本頁的方程式與數字取自論文 v1；標示『本頁推導』『示範值』『圖讀值』的內容不是論文正文的結論。」

### 6.2 新增的部分

- 新 `.tag` 變體：`.tag.demo`（「示範值」）、`.tag.fig`（「圖讀值」），樣式同 `.tag`，文字不同；正文中「作者認為／作者推測」不加 tag，以文字區分。
- 2×2 設計格元件 `.grid22`（hero、S2、S4 共用一個 `renderGrid22(el, highlight)`）：四格標「OnPD·F」「OnPD·R」「OffPD·F」「OffPD·R」，格內兩半標「1e-5」「5e-5」；390px 時每格最小 150px 寬。
- 四個元件：`initC1()`（`klParts`）、`initC2()`（`boundParts`）、`initC3()`（`spectrum`）、`initC4()`（`FIG1`、`pairDiffs`）。每個函式在全頁只定義一次，元件與 selfTest 都呼叫同一份。
- 一個共用 `liveAnnounce(el, text)`：每元件一個 debounce timer（500ms）。
- 長條圖輔助 `bars(svg, data, opts)`（C1、C3 共用；支援實心、外框、斜線三種填充，斜線用 `<pattern>`，深色模式下 pattern 顏色取 CSS 變數）。
- TeX：HTML 內的小於號一律寫 `\lt`（例：$y_{\lt n}$、$0\lt\delta\lt\tfrac12$）；SVG 與讀數用 Unicode（「π_S(A)」「λ」「δ」「‖g‖₂」）。

### 6.3 selfTest 項目（誤差同第 5 節）

1. `klParts(0.5,0.01)`：DF 0.6371、DR 1.6145、gR_A 1.1488；`klParts(0.9,1e-6)`：nR ≤ R、maxF ≤ 1；五組預設的座標和 ≈ 0。
2. `boundParts(0.01,0.1)`：rev 0.0812、fwd 0.0346；`boundParts(1/(1+Math.exp(16)),1).rev` = 2.828427 ± 1e-6。
3. `spectrum(2,true)`：π_F 0.0105、amp 3.0000；`spectrum(2,false)`：π_F 0.1164；`spectrum(±1,·)` 等於錨點；λ 由 −2 到 2 每 0.05 的 amp ≤ 3。
4. `pairDiffs('SPA','LR')` = [33.4, 30.0, 29.3, 29.6]；`pairDiffs('ACC','rollout')` = [1, 1, 37, 14]。
5. 每個 LO 的參考答案關鍵數字（72、73、1.3、11.2、14.0、85.3、51.9、5.2、10–15、59.0）出現在對應章節的 `<p>` 內（以 `section#id p` 的 textContent 檢查）。
6. 無水平捲動（390px 模擬時由截圖檢查；selfTest 檢查當前寬度）。

### 6.4 研究脈絡（S12）

- 來源 `tut/136/notes/timeline.json`（陣列，14 筆，欄位 `year`、`title`、`authors_short`、`arxiv`、`role`、`source`），原樣嵌入 `<script type="application/json" id="timeline-data">`，沿用參考頁的 `renderTimeline`（橫向捲動卡片、類別篩選 chip、前後捲動鈕、缺欄位與解析失敗的降級）。
- `TL_GROUP`（以 arXiv ID 為 key；不在表中的歸「其他」）：

  | 類別 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | 蒸餾基礎 | 1503.02531、1606.07947 | `knowledge-distillation-foundations.html` |
  | on-policy 蒸餾 | 2306.13649、2410.11325 | `odd-onpd-vs-offpd-setup.html` |
  | 被檢驗的說法：遺忘與稀疏 | 2509.04259、2510.18874、2505.11711、2601.19897 | `odd-on-policy-forgetting-sparsity-claims.html` |
  | 被檢驗的說法：泛化 | 2501.17161 | `odd-sft-vs-rl-generalization-claims.html` |
  | 最佳化與 KL 估計 | 1412.6980、2506.09477、2604.13627 | `odd-learning-rate-adam-clipping-rlvr.html`（2506.09477 連 `odd-token-kl-direction-gradients.html`） |
  | 同期 on-policy 蒸餾研究 | 2604.13016 | `odd-recent-opd-studies-2026.html` |
  | 本文 | 2609.35259 | `odd-lineage-hub.html` |

- 延伸閱讀（S13）依總覽的閱讀順序：`odd-lineage-hub.html`（總覽）、`knowledge-distillation-foundations.html`、`on-policy-distillation-imitation.html`、`odd-onpd-vs-offpd-setup.html`、`odd-token-kl-direction-gradients.html`、`odd-on-policy-forgetting-sparsity-claims.html`、`odd-sft-vs-rl-generalization-claims.html`、`odd-recent-opd-studies-2026.html`、`odd-strong-to-weak-distillation-practice.html`、`odd-learning-rate-adam-clipping-rlvr.html`；另列既有 `context-distillation-self-distillation.html`、`rlvr-grpo-credit-assignment.html`、`ppo-grpo-dapo-llm-rl-objectives.html`。每項附筆記 frontmatter 的 `title`。

### 6.5 建置後自檢

`?test` 全數通過；`extract_page_text.py` 抽出正文後逐條確認 12 個 LO 答案在正文；截圖 390 / 1440 與深色模式各一張，檢查新元件（C1–C4、設計格）；impeccable 0 項。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | Fig. 1 長條對應 | 論文正文只給範圍；各長條的值來自圖中標籤，pdftotext 排版混亂。已以 PDF 第 5 頁影像核對：淺色＝1×10⁻⁵、深色＝5×10⁻⁵，對應如步驟 4。 | 頁面標「圖讀值」；C4 假設行寫明來源。 |
| R2 | 準確率標籤為整數 | 配對差 0、1、2 pp 的解析度受整數標籤限制。 | C4 正文註明「準確率標籤為整數，1 pp 以內的差異無法區分」。 |
| R3 | 「10–15% higher」 | Sec. 5.3 未說明是百分點還是相對百分比；Fig. 5 縱軸 0–40%，兩者都可能。 | 頁面照錄「10–15%」並加註「論文未說明是百分點或相對比例」。 |
| R4 | 論文內部不一致 | (a) App. D.4 寫「Appendix A.6 報告 τ = 10⁻⁸」，實際在 A.7；(b) Fig. 18（τ = 10⁻⁸）的標籤與 Fig. 1（τ = 10⁻⁶）完全相同，論文未說明；(c) App. A.11 第二段寫「OnPD with forward KL」對風格轉移穩健，與 Fig. 24 及 Sec. 5.3（OnPD＋reverse KL）矛盾；(d) App. A.2 寫「draw $x_n\sim\pi_S^\theta$」，前後文應為 $y_n$。 | 頁面依圖與正文主線（(c) 採 reverse KL；(d) 寫 $y_n$），不引用 Fig. 18 數字；在 S10／S9 腳註簡述 (b)(c)。 |
| R5 | 單次訓練的結果 | Fig. 15（Qwen 光譜）、A.3（MATH）、A.11（風格）每條件只有 1 次訓練。 | 正文提到這些結果時都註明「單一 seed／每條件 1 次」。 |
| R6 | 作者解釋與量測混淆 | Adam 尖峰解釋（A.1）、溫度解釋（A.10）、英文 mode 解釋（A.11）、自我強化機制（A.10）都是作者的假說。 | 一律寫「作者推測／作者提出的可能解釋」，與量測句分開成不同句子。 |
| R7 | 玩具與論文現象的關係 | C1–C3 是單一前綴或兩 token 的示範，不模擬訓練；不能顯示「forward KL 最終表現穩健」。 | 假設行寫明；正文不說玩具「證明」實驗結果，只說它呈現 Eq. 4–5、Eq. 57、Eq. 75–80 的計算。C3 的 G_S、G_T 只表示抽樣分布偏向誰，與 Fig. 3 的準確率無關。 |
| R8 | C2 的跨式比較 | forward 界 $2\sqrt2B$ 與 reverse 差屬不同目標；交點只是數字比較。 | 正文寫「reverse 的更新差在 δ ≈ 1.1×10⁻⁷ 時超過 forward 的通用上界」，不暗示兩者可直接互換。 |
| R9 | notes 可能更新 | 建置前 notes 若有增減，`TL_GROUP` 與延伸閱讀需同步。 | 建置時重新讀 `notes/`；未列入 `TL_GROUP` 的 arXiv ID 歸「其他」並在建置回報列出。 |
| R10 | 符號衝突 | 論文用 $M$ 表詞彙大小、$\alpha$ 同時是 D.5 的遮罩門檻；$\epsilon$（Adam）與 $\varepsilon$（TV 反例）。 | 頁面用 $\lvert\mathcal V\rvert$ 表詞彙大小；$\alpha$ 只用於遮罩；Adam 的 ε 以文字「Adam 的 ε」稱呼。 |
| R11 | 圖讀值精度 | Fig. 24 與 Fig. 10 的標籤為圖中文字。 | 標「圖讀值」，只用到小數一位。 |
