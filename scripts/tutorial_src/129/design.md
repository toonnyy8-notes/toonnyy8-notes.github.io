# 設計文件：〈Score Centering Stabilizes Off-policy Reinforcement Learning〉互動教學

- 論文：Martin Marek、Max Ryabinin（Together AI），arXiv 2609.20807v1（2026-09-17）。資料庫編號 No. 129。
- 原文：`tut/129/src/paper.txt`（PDF 同目錄）。「Eq. n」「§n」「Fig. n」「Table n」「註 n」皆指論文原文編號（註 = 腳註）。
- 顧問筆記：`tut/129/advisor.md`。其中標〔讀圖〕的名次判斷來自文字抽取檔的標籤位置，**本頁一律不用**；圖的結論只引用論文正文與圖說的文字（見 R1）。
- 產出：`tut/129/build/index.html`（單檔）。上線位置 `tutorials/score-centering-off-policy-rl.html`（slug `score-centering-off-policy-rl`）；站內連結以上線位置為準。
- 驗算：`tut/129/verify/check.py`（輸出 `check.out`，已跑完，`all asserts passed`）。第 5 節的數字全部由它產生，演算法與頁面相同。
- 研究筆記：`tut/129/notes/`（hub `score-centering-lineage-hub` + 6 篇 topic + `timeline.json`，15 筆）。
- 標記慣例：本頁自行推導、論文沒有寫的結論標〔本頁推導〕；為了做教學示意而**自行設計**的玩具模型標〔教學示意〕，頁面上同樣以 pill 標籤顯示，並在圖版底部寫「此模型為本頁設計的示意，不是論文的實驗或結果」。本文沒有「用論文數字校準的推定值」（所有玩具參數都是自選，不對應論文數字），因此不使用「推定」標籤。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

用 policy gradient 訓練 LLM 時，rollout 由推論引擎（sampler，機率 $q$）產生，梯度由訓練引擎（trainer，機率 $p$）計算。兩者理論上是同一個模型，實際上因精度、kernel、非同步更新而輸出不同，稱為訓練與推論不一致（training-inference mismatch，TIM）。本文把每個 prefix 上的期望更新拆成兩項（Eq. 3）：$\mathbb E_q[R\,s_{y_t}]=\mathbb E_q[R]\,\bar s+\mathrm{Cov}_q(R,s_{y_t})$。第二項（signal）才帶有「哪個 token 帶來哪個 reward」的資訊；第一項（drift）只依賴平均 reward，方向 $\bar s=\sum_v q_v\nabla\log p_v$ 是「以 sampler 為 teacher 的 cross-entropy」的負梯度，也就是把 trainer 蒸餾向 sampler。因為 sampler 是 trainer 的有偏副本、又定期從 trainer 同步，作者認為這個偏差在回饋迴路中累積，是 TIM 下不穩定的主因。作者提出 score centering（SC）：把每個 score 減去 $\bar s$（Eq. 4），期望更新變成 $\mathrm{Cov}_q(R,s_{y_t})$（Eq. 5），drift 在每個 prefix 精確歸零。實作只需 sampler 的 top-k logprob（$k=128$），以純量 loss 表示（Eq. 6），並可與 TIS／MIS 疊加（Eq. 10、12）。在 Qwen3-0.6B（Countdown）與 Qwen3-30B-A3B-Base（INTELLECT-2 math）上，嚴重量化時 SC 優於 IS 類方法（30B、INT8 W/A + INT4 KV：SC 30%、TIS 12%、其餘 < 5%，§5.4）；嚴重 staleness 時，SC 疊加 TIS／MIS 最好（§5.3）。

### 1.2 研究動機（§1、§2）

1. RL 佔 LLM 訓練算力的比例增加；PPO、GRPO、DAPO、IcePop 都建立在 policy gradient（Eq. 1）上。
2. Eq. 1 需要兩次 forward：sampler 產生 rollout、trainer 算梯度。多數 codebase 兩者有小差異，造成表現下降（Khatri et al., 2026）甚至 reward 崩潰（Liu et al., 2025b）。
3. 完全消除 TIM 代價很高：batch-invariant kernel 需要大量工程且 GPU 使用率較差；bf16 → fp16 只能大幅減少、不能消除；非同步與 continuous batching 本身就產生 staleness（§2.1）。作者的目標是用演算法讓 TIM 下的訓練穩定，以保留硬體使用率。
4. 現有修正幾乎都基於 importance sampling（IS），而實務上都要截斷或遮罩比值，必然引入偏差（§2.2）。
5. SFT 在更嚴重的不一致（另一個模型離線產生的資料）下仍穩定，為什麼 RL 對小小的數值差異這麼敏感（§3）？

### 1.3 貢獻（§6 作者自述兩點，加上實驗）

1. 解釋：TIM 下的 PG 更新含 drift 項，作用是朝 sampler 蒸餾；sampler 定期同步回 trainer，偏差在回饋迴路中累積（§4、§6）。
2. 方法：score centering，一個加法修正項；精確版無超參數，可寫成純量 loss（Eq. 6），可與 IS 疊加（App. A.2）；top-k 近似（App. A.1）。
3. 實驗：合成權重雜訊、量化、staleness 三類 TIM，0.6B 與 30B 兩個模型（§5）。

### 1.4 符號表（頁面 S4a 開頭以可收合表格提供）

| 符號 | 意義 |
| --- | --- |
| $y$、$y_t$、$y_{<t}$ | 一條 rollout（模型產生的完整回答）、第 $t$ 個 token、它之前的 prefix |
| $p_\theta$、$q_\theta$ | trainer、sampler 的機率（同一組參數 $\theta$ 的兩份實作）；省略下標時寫 $p,q$ |
| $v$；$p_v,q_v$ | 詞彙中的 token；固定 prefix 下 $v$ 的 next-token 機率 |
| $s_v=\nabla_\theta\log p_v$ | token $v$ 的 score（永遠用 trainer 計算）；$s_{y_t}$ 為被取樣 token 的 score |
| $\bar s=\sum_v q_vs_v$ | sampler 下的期望 score |
| $\tilde s_{y_t}=s_{y_t}-\bar s$ | 中心化後的 score（Eq. 4） |
| $R$ | 整條 rollout 的 reward 或 advantage |
| $\mathbb E_q,\mathbb E_p$ | 固定 prefix 後，對其餘 rollout 的期望（token 由 $q$ 或 $p$ 取樣） |
| $r=p/q$；$w=f(r)$ | importance ratio；IS 方法給的權重（TIS：$\min(r,2)$；MIS：$r\in[0.5,5]$ 時為 $r$，否則 0） |
| $\hat w_v=f(p_v/\hat q_v)$ | 以重建分布 $\hat q$ 計算的權重：head 上等於 $w_v$，tail 上恆為 $f(1/\rho)$（Eq. 10） |
| $H,T$；$\rho$；$\hat q$；$\alpha$ | top-k 的 head 與 tail；tail 質量比（Eq. 7）；重建的 sampler 分布；$\alpha=\rho f(1/\rho)$（Eq. 10） |
| $G$；$A_i$ | 每個 prompt 的 rollout 數；第 $i$ 條 rollout 的 advantage（Eq. 13） |
| $\mathrm{sg}[\cdot]$ | stop-gradient |

玩具模型專用（S4b 才引入）：$z\in\mathbb R^V$ 為單一 prefix 的 trainer logits，$p=\mathrm{softmax}(z)$；$e_v$ 為第 $v$ 個標準基底向量；$Q_v=\mathbb E[R\mid y_t=v]$；$\lambda$ 為 C1、C3 的不一致程度（不用 ε，避免與程式的 `eps` 混淆）；$\sigma$ 為 C4 的擾動尺度；$\mathrm{Ent}(p)$ 為熵（不用 $H$，$H$ 保留給 head）；$N$ 為 C3 總步數；$m_q^T,m_p^T$ 為 sampler／trainer 的尾部質量；C2 的路徑「甲／乙」、成功機率 $h_甲,h_乙$、drift 係數 $c$。Eq. 13 的 $L_i$ 是 rollout 長度，與 loss $L$ 無關。

### 1.5 從作者的角度逐步拆解

每一步：作者想回答什麼、假設、得到什麼、為什麼下一步需要它。

#### 步驟 1：policy gradient 需要兩個引擎（§1，Eq. 1）

$$\nabla_\theta\,\mathbb E_{p_\theta}[R]=\mathbb E_{p_\theta}\big[R\,\nabla_\theta\log p_\theta(y)\big]\tag{Eq. 1}$$

- $\nabla_\theta\log p_\theta(y)$ 稱為 score。等式只在「算梯度的 policy 就是產生 rollout 的 policy」時成立（§2.1）。
- 一條 rollout 的更新是各 token score 的和，每個都乘同一個 $R$（§4 Notation）。所以作者把分析縮到「固定 prefix 下，一個 token 的期望貢獻」。

#### 步驟 2：TIM 的來源（§2.1）

依嚴重程度：(a) 兩引擎精度不同或來自兩套 codebase，各有難以察覺的 bug；(b) 同一套 kernel 也會因浮點加法不可結合而不同——註 2：fp4_e2m1 下 $(0.5+0.5)+2=3$，但 $0.5+(0.5+2)=2$。取樣是自回歸、訓練沿序列維度平行，兩邊以不同輸入形狀呼叫 kernel，reduction 順序不同；(c) 即使 kernel 完全相同，非同步訓練與 continuous batching 讓一個 batch、甚至一條 rollout 的不同片段來自不同 checkpoint（staleness）；agentic coding 這類長時程環境最嚴重。
- 〔本頁推導〕fp4_e2m1 的非負可表示值為 0, 0.5, 1, 1.5, 2, 3, 4, 6；$0.5+2=2.5$ 落在 2 與 3 正中間，就近偶數捨入到 2，再加 0.5 得 2.5 又捨入到 2。頁面只用一句話說明，不做元件。

#### 步驟 3：IS 精確但高變異，實務必須截斷（§2.2，Eq. 2，Table 1）

$$\nabla_\theta\,\mathbb E_{p_\theta}[R]=\mathbb E_{p_\theta}[R\,\nabla_\theta\log p_\theta(y)]=\mathbb E_{q_\theta}\Big[\frac{p_\theta(y)}{q_\theta(y)}R\,\nabla_\theta\log p_\theta(y)\Big]\tag{Eq. 2}$$

- Eq. 2 寫在序列層級；實務上逐 token 套用（Zheng et al., 2025）。比值在稀有 token 上可任意大，梯度變異膨脹；所以實務方法都截斷或遮罩比值，必然引入偏差。
- 〔本頁推導〕只有序列層級的比值精確修正；逐 token 比值不修正 prefix 的造訪分布。頁面以一句話註明。
- Table 1（照錄，Level／Stat／Low／High／Outside／Sign；上標 P = 原論文預設、V = verl 預設，照原表保留）：Naive IS token ratio 0 ∞ – both；TIS^V token ratio 0 2.0 clip both；MIS（IcePop）^P token ratio 0.5 5.0 drop both；PPO^{P,V} token ratio 0.8 1.2 drop ppo；DAPO^{P,V} token ratio 0.8 1.28 drop ppo；GSPO^{P,V} seq ratio 0.9997 1.0004 drop ppo；TOPR^P seq ratio 0 1.0 clip neg；DPPO^P token tv 0 0.2 drop ppo。頁面上 S2 只放 TIS、MIS 兩列並解釋欄位，完整 8 列放在 `<details>`「Table 1 全表」中。Sign：both 對所有 token 加權；neg 只對負 advantage；ppo 為 PPO 的悲觀 clip（取加權與截斷目標的最小值）。dual clipping：ratio > 3 的負 advantage 貢獻丟棄（GSPO 丟整條回答）。
- 實作附註（Table 1 最後一欄，頁面照錄並強調「比的是修正規則，不是原論文完整配方」）：TIS 省略 CISPO 的 dynamic sampling 與長度懲罰；MIS 省略 IcePop 在 policy 版本之間的 PPO surrogate；PPO dual clipping 3；DAPO dual clipping 3、省略 dynamic sampling 與 overlong penalty；GSPO token 梯度加總（非回答平均）並加 dual clipping 3；DPPO 不做比值截斷。

#### 步驟 4：SFT 與 RL 只差兩件事（§3，Fig. 2，註 3、4）

- 先定義（頁面 S3 第一段，避免前向引用）：advantage $A$ 是實際用來乘 score 的權重，可以是 reward 本身，也可以是 reward 減去某個基準；group centering 指每個 prompt 取 $G$ 條 rollout，$A_i=R_i-\frac1G\sum_jR_j$（Eq. 13 的形式），同組 advantage 總和為 0。
- PG 在二元 reward、advantage 為 +1/0、且完全離線（sampler 從不更新）時就是 SFT。所以 reward 模式與 staleness 是 SFT 與 RL 的兩個關鍵差異。
- 註 4：on-policy 時 reward 加常數不改變期望梯度，group centering 只把它縮放 $1-1/G$；中心化的目的是降低變異（Greensmith et al., 2004；Liu et al., 2025c）。但離線時不成立：+1/0 的 SFT 收斂到最大概似估計，0/−1 的 loss 無界、可能崩潰（Ren & Sutherland, 2025）。
- 實驗設定：Qwen3-1.7B、Countdown，把 sampler 權重相對 trainer 擾動「a small amount」（論文未給數值）。group-centered 由哪種 reward 中心化：B.1 寫「除非另外說明」都用 Eq. 13（$R_i\in\{-1,+1\}$），Fig. 2 未另述，所以頁面寫「依論文預設應為 ±1，Fig. 2 未另外說明」。
- Fig. 2 圖說（照錄）：離線只有 +1/0 穩定；線上（有 TIM）以 +1/0 最不穩定、group-centered 最穩定。
- 作者**假設**（原文 "We hypothesize"）有兩個機制：離線負 reward 不穩定（Ren & Sutherland 的 logprob 無下界機制，本文不探討）；線上全正 reward 不穩定。線上 +1/0 本身就是蒸餾：trainer 擬合 sampler（自己的有偏副本）的成功 rollout，而 sampler 每步從 trainer 同步。與離線蒸餾唯一的差別是 teacher 跟著 student 移動。
- 下一步：需要把「跟著動的 teacher」寫成公式。

#### 步驟 5：on-policy 下常數 reward 不產生學習（§4 第一段）

$$\mathbb E_p[s_{y_t}]=\sum_vp_v\nabla_\theta\log p_v=\nabla_\theta\sum_vp_v=\nabla_\theta1=0$$

對任何分布成立。所以常數 reward 下 on-policy 期望梯度 $R\cdot0=0$。TIM 下 token 由 $q$ 取樣、score 由 $p$ 算：$\mathbb E_q[s_{y_t}]=\bar s\ne0$（$\sum_vq_v\nabla p_v/p_v$ 不能合併成 $\nabla\sum_vp_v$）。

#### 步驟 6：drift／signal 分解（§4，Eq. 3，註 5）

由 $\mathrm{Cov}(A,B)=\mathbb E[AB]-\mathbb E[A]\mathbb E[B]$（註 5），取 $A=R$、$B=s_{y_t}$：

$$\underbrace{\mathbb E_q[R\,s_{y_t}]}_{\text{PG 更新}}=\underbrace{\mathbb E_q[R]\,\bar s}_{\text{drift}}+\underbrace{\mathrm{Cov}_q(R,s_{y_t})}_{\text{signal}}\tag{Eq. 3}$$

- drift 只透過平均 $\mathbb E_q[R]$ 依賴 reward，不知道哪條 rollout 成功；只有共變異數項看得到哪個 token 帶來哪個 reward。
- $\bar s$ 的意義：cross-entropy $\mathrm{CE}(q,p)=-\sum_vq_v\log p_v$（把 $q$ 當常數）的負梯度正是 $\bar s$。所以 PG 在每個 prefix 把 trainer 朝 sampler 蒸餾，強度為該 prefix 的期望 reward。〔本頁推導〕等價於 $-\nabla_\theta\mathrm{KL}(q\|p)$，因為 $q$ 的熵與 $\theta$ 無關。
- on-policy 時 $\bar s=0$，drift 消失。§1 的說法是 drift「若且唯若」兩個 policy 相同時為 0。〔本頁說明〕這個「若且唯若」是寬鬆說法：drift $=\mathbb E_q[R]\bar s$，在 $\mathbb E_q[R]=0$ 時也為 0（C1 的 ±1 例：λ = 0.5 時 ‖drift‖ 只有 0.0098）；在 logit 座標下 $\bar s=q-p$，確實只有 $p=q$ 時為 0，但對真實網路 $\bar s=J^\top(q-p)$（$J$ 為 logits 對 $\theta$ 的 Jacobian），$p\ne q$ 時也可能為 0。頁面在 S4b 以一段「本頁說明」寫出，Hero 不再重複這句。

#### 步驟 7：固定 teacher 無害，跟著動的 teacher 累積誤差（§4 "Drift is an artifact…"）

- 朝固定 teacher 蒸餾收斂到該 teacher 的最大概似擬合。sampler 不是固定的：它是 trainer 的有偏副本（量化或過時），每步把 trainer 推向 sampler，trainer 權重又同步回 sampler，誤差在回饋迴路中累積而不收斂。作者以此解釋 Fig. 2 線上 +1/0 最不穩定，並引 Dong et al.（2026）同樣指出系統性偏差經由正回饋累積。
- 這一步是**文字論證**：論文沒有定理、沒有發散速率。頁面用 C3（〔教學示意〕）呈現其中一部分，並寫明 C3 的範圍：C3 的偏差是固定的 logit 位移 $\lambda\xi$（類似量化造成的固定偏差），所以它呈現的是「固定偏差在每次同步後持續累積」，logits 大致隨步數線性成長；它**不是**論文所說誤差互相放大的回饋迴路（compounding），也不模擬 staleness（隨時間改變的偏差）。
- §5.1 用同一個論點解釋為何溫和 TIM 下各方法難以區分：drift 需要更多步才能累積；Qiu et al.（2026）認為 fp8 rollout 穩定與 Microsoft AI Team（2026）bf16 仍發散，作者認為兩者都與「drift 隨訓練累積」相容。C3 中 $t_{1/2}$ 約與 $1/\lambda$ 成正比，這是 C3 固定線性偏差假設的結果，**不是論文的主張**，頁面在 C3 圖版下明寫。

#### 步驟 8：group centering 不消除 drift（§4）

- Group centering 讓同一 prompt 的 advantage 總和為 0，但 drift 發生在單一 prefix，而在特定 prefix 上期望 advantage 不為 0：容易導向正確答案的 prefix 期望 advantage 為正，已含錯誤的 prefix 為負。trainer 在前者之後被拉向 sampler，在後者之後被推離 sampler，與該 token 是否影響 reward 無關。Group centering 相對 +1/0 會縮小 drift（與 Fig. 2 中它最穩定一致），但不消除。
- 〔本頁推導〕定量版：$G$ 條 rollout 獨立，第 $i$ 條的 prefix 只影響 $R_i$，$A_i=R_i-\frac1G\sum_jR_j$，則
  $$\mathbb E[A_i\mid y_{i,<t}]=\Big(1-\frac1G\Big)\big(\mathbb E_q[R\mid y_{i,<t}]-\mathbb E_q[R]\big).$$
  這個 prefix 的 drift 為上式乘 $\bar s(y_{i,<t})$。對 prefix 取平均為 0（prompt 層級總和為 0），但各 prefix 一般不為 0。$G\to\infty$ 時係數趨近 $\mathbb E_q[R\mid\text{prefix}]-\mathbb E_q[R]$，也就是「以 prompt 平均為基準的 advantage」；它等於未中心化 ±1 reward 的條件期望（$2h-1$）只在 prompt 平均 reward 剛好為 0 時成立（C2 中即 $\pi_甲=0.5$ 且 $h_甲+h_乙=1$）。單步（bandit）情形中 prefix 就是 prompt，上式恆為 0，這就是 C3（bandit）不放 group centering、另做 C2（兩步）的理由。

#### 步驟 9：score centering（§4，Eq. 4–5）

$$\tilde s_{y_t}=s_{y_t}-\bar s\tag{Eq. 4}$$
$$\mathbb E_q[R\,\tilde s_{y_t}]=\mathbb E_q[R]\underbrace{\mathbb E_q[\tilde s_{y_t}]}_{=\bar s-\bar s=0}+\mathrm{Cov}_q(R,\tilde s_{y_t})=\mathrm{Cov}_q(R,\tilde s_{y_t})=\mathrm{Cov}_q(R,s_{y_t})\tag{Eq. 5}$$

- 最後一個等號：$\bar s$ 在給定 prefix 下是常數，與任何隨機量的共變異數為 0。
- 條件：中心化的期望與取樣使用**同一個** $q$；$\bar s$ 可精確計算（對詞彙求和）。
- 與 on-policy 的唯一差別（論文的說法，固定 prefix 下）：共變異數在 $q$ 而非 $p$ 下量（on-policy 與 IS 都在 $p$ 下量）。所以 SC 的期望更新**不等於**真實 policy gradient（§6 Limitations）。$p=q$ 時修正項為 0，SC 是 no-op。
- 〔本頁推導〕Eq. 5 是固定 prefix 的陳述；整條序列上 prefix 的造訪分布仍是 $q$，SC 不修正這一點。

#### 步驟 10：與 IS、與 baseline 的關係（§4 兩段）

- IS 也精確消除 drift，但它是隨機的**乘法**修正，值依賴被取樣 token，增加變異；實務截斷/遮罩後重新引入 drift。SC 是給定 prefix 下確定的**加法**修正，不依賴被取樣 token，可精確計算。兩者以獨立方式修正，可以疊加。
- 〔本頁推導〕「截斷重新引入 drift」只在截斷真正生效處成立：TIS 下若所有 $p_v\le2q_v$，則 $w_v=r_v$，$\sum_vq_vw_vs_v=\sum_vp_vs_v=0$。C1b 與 C4 顯示這點。
- baseline（Williams, 1992）與 score-function control variate（Ranganath et al., 2014）都依賴 $\mathbb E_p[s]=0$ 做變異縮減；SC 是**偏差修正**：減掉的向量在 prefix 下確定，目的是移動更新的平均。離線時，若 baseline 等於該 prefix 的期望 reward（精確 value function），期望上也消除 drift；SC 不需要 critic 就得到同樣的期望更新。〔本頁推導〕驗證：$\mathbb E_q[(R-V)s]=\mathbb E_q[R]\bar s+\mathrm{Cov}-V\bar s=\mathrm{Cov}_q(R,s)$，$V=\mathbb E_q[R\mid\text{prefix}]$。
- 古典離線 RL 中期望 score 需要對動作空間做難解的期望；LLM 的 TIM 情境特殊在期望 score 非零而且能以 next-token logprob 的和精確算出。

#### 步驟 11：純量 loss（App. A.3，Eq. 11）

$$L=-R\Big(\log p_{y_t}-\sum_v\mathrm{sg}[q_v]\log p_v\Big)\tag{Eq. 11}$$

$-\nabla_\theta L=R(s_{y_t}-\bar s)=R\,\tilde s_{y_t}$。（Eq. 6、11、12 的括號：fidelity 審查注意到文字抽取檔看似沒有括號；已對照 PDF 第 6 頁，Eq. 6 印的是 $-R\big(\log p_{y_t}-\sum\cdots\big)$，有括號，與程式一致，本頁照抄。）被減的是 $R\bar s$（用這條 rollout 自己的 $R$），其期望 $\mathbb E_q[R]\bar s$ 正是 drift。符號：Eq. 13 的 $J_{\text{base}}$ 為最大化目標，Eq. 6/11/12 的 $L$ 為最小化 loss。〔本頁推導〕展開為 $-R\log p_{y_t}-R\cdot\mathrm{CE}(q,p)$：$R>0$ 時第二項在提高 CE，即反向抵銷 PG 隱含的蒸餾。

#### 步驟 12：top-k 近似（§4.1、App. A.1，Eq. 6–8）

- 記憶體（App. A.1）：Qwen3 詞彙 152K，一個 token 的完整 fp32 logprob 約 608KB；1024 條 rollout × 序列長 32K 需約 20TB。所以只記錄 sampler 的 top-k（$k=128$）。
- 尾部以 trainer 分布按比例填補：
$$\hat q_v=\begin{cases}q_v&v\in H\\\rho\,p_v&v\in T\end{cases},\qquad\rho=\frac{1-\sum_{v\in H}q_v}{1-\sum_{v\in H}p_v}\tag{Eq. 7}$$
- 利用 $\mathbb E_p[s]=0$，期望只需 head：
$$\mathbb E_{\hat q}[s_{y_t}]=\sum_{H}q_vs_v+\rho\sum_Tp_vs_v=\sum_Hq_vs_v+\rho\Big(\underbrace{\mathbb E_p[s_{y_t}]}_{=0}-\sum_Hp_vs_v\Big)=\sum_{v\in H}(q_v-\rho p_v)s_v\tag{Eq. 8}$$
$$L=-R\Big(\log p_{y_t}-\sum_{v\in H}\mathrm{sg}[q_v-\rho p_v]\log p_v\Big)\tag{Eq. 6}$$
- 〔本頁推導〕殘餘 drift $=\mathbb E_q[R]\sum_{v\in T}(q_v-\rho p_v)s_v$；當 sampler 尾部與 trainer 尾部成比例（$q_v=\rho p_v,\ \forall v\in T$）時為 0。這是充分條件；在 logit 座標下殘餘為 $q-\hat q$，此時也是必要條件。
- 經驗（App. A.4）：$k=128$ 時 head 平均覆蓋 > 99.9% 的 sampler 質量；唯一例外是 30B 的 INT8 W/A + INT4 KV：平均 99.45%、最差 batch 95.8%，即使如此 $k=32$ 與 $k=128$ 仍與全詞彙 SC 相當（Fig. 5）。成本：同硬體下 $k=128$ 的 wall-clock 與基準相差 1% 以內（0.6B 與 30B；自製 JAX sampler；vLLM、SGLang 原生提供 top-k logprob，但額外開銷**未測**）。
- 程式（§4.1）：`tail_mass_ratio = (1 - exp(samp_logp).sum()) / (1 - exp(train_head_logp).sum())`，`head_prob_residual = exp(samp_logp) - tail_mass_ratio * exp(train_head_logp)`，`logp_correction = (stop_gradient(head_prob_residual) * train_head_logp).sum()`，回傳 `-advantage * (train_logp[sampled_token] - logp_correction)`。§4.1 短版沒有除零保護；App. A.3 完整版以 `eps=1e-6` 為兩個尾部質量設下限。

#### 步驟 13：與 IS 疊加（App. A.2，Eq. 9–10、12）

$$R\big(w_{y_t}s_{y_t}-\mathbb E_q[w_{y_t}s_{y_t}]\big),\qquad\mathbb E_q[w_{y_t}s_{y_t}]=\sum_vq_vw_vs_v\tag{Eq. 9}$$

常數 reward 下期望為 0，與 $f$ 無關。top-k 時 sampler 機率只知道 $\hat q$，所以權重也用 $\hat q$ 計算：$\hat w_v=f(p_v/\hat q_v)$（App. A.2；head 上等於 $w_v$）。尾部 $p_v/\hat q_v=1/\rho$ 為常數，所以
$$\mathbb E_{\hat q}[\hat w_{y_t}s_{y_t}]=\sum_{v\in H}(q_vw_v-\alpha p_v)s_v,\qquad\alpha=\rho f(1/\rho)\tag{Eq. 10}$$
$$L=-R\Big(\mathrm{sg}[w_{y_t}]\log p_{y_t}-\sum_{v\in H}\mathrm{sg}[q_vw_v-\alpha p_v]\log p_v\Big)\tag{Eq. 12}$$

- 檢查：$f=1\Rightarrow\alpha=\rho$（回到 Eq. 8）；$f(r)=r\Rightarrow\alpha=1$ 且 $q_vw_v=p_v$，中心化項為 0（精確 IS 本來就零期望）。
- 〔本頁推導〕App. A.3 程式中被取樣 token 的權重用真實的 `samp_token_logp`（即使它在 head 之外），而中心化項假設尾部權重恆為 $f(1/\rho)$；所以 top-k 疊加版的零期望只在 $\hat q=q$ 時精確。

#### 步驟 14：實驗（§5、App. B）

- 共同目標（Eq. 13）：$J_{\text{base}}(\theta)=\frac{\sum_iA_i\sum_{t=1}^{L_i}\log p_\theta(y_{i,t}\mid y_{i,<t})}{\sum_iL_i}$，$A_i=R_i-\bar R_{\text{group}}$，$R_i\in\{-1,+1\}$；除以 batch 的總 token 數。每個修正規則單獨加在這個目標上；同一 sampler、trainer、optimizer、advantage；每 batch 一步 SGD；所有比值用 sampler 記錄的機率（§5.1、B.2）。
- 超參數（B.1）：SGD、固定學習率 $10^{-2}$（依 Mukherjee et al., 2026；表現與 AdamW 相當，省下最多 240GB 記憶體）；每 prompt 8 個 completion。0.6B：64 prompt/batch（512 序列）、最大長度 512；30B：16 prompt/batch（128 序列）、最大長度 1024。各方法用原論文或 verl 預設值，未調參。
- 為何刻意加重 TIM（§5.1）：自然設定下 TIM 不夠嚴重，幾個 GPU 小時內無法在最強的方法之間看到統計顯著差異；作者把嚴重 TIM 的短期訓練當作「溫和 TIM 的長期訓練」的代理。
- 合成權重雜訊（§5.2，Fig. 3，eval accuracy）：$\theta_{\text{sampler}}=\theta_{\text{trainer}}+\Delta\theta$，$\Delta\theta\sim\mathcal N(0,0.01^2),\mathcal N(0,0.02^2),\mathcal N(0,0.05^2)$，開始時抽一次後固定。三個尺度下 SC、TIS、MIS 及 MIS+SC、TIS+SC 表現最好；最大雜訊下只有 SC 與 SC+MIS／TIS 穩定。DPPO 約在 step 160、80、20 崩潰；TIS 在最小雜訊下穩定，在較大兩個尺度約 step 180、40 崩潰。
- 量化與 staleness（§5.3，Fig. 4，training accuracy）：只量化 sampler（weights、activations、KV cache），trainer 為 bf16；staleness 為推論引擎**每 64 步才更新一次**（不是「最大 staleness 64」）。量化下 SC（單獨與疊加 TIS／MIS）最好；staleness 下 SC + TIS／MIS 最好，且優於單獨 SC。作者歸因於 SC 在 sampler 下量共變異數（Eq. 5）：TIS 部分修正取樣分布，SC 去掉剩下的 drift；建議 trainer 可能在兩次同步間遠離 sampler 時就疊加 IS。PPO、DAPO 撐過 staleness 但在量化與權重雜訊下崩潰，作者認為這與其 clip 是為 policy 移動造成的比值設計一致。作者也說實務上應盡量頻繁同步、兩邊用同一量化方案。
- 30B（§5.4，Fig. 1，training accuracy）：MoE，router indices 未 replay。FP8 W/A + FP8 KV：未修正 PG 也穩定，達 58%；FP8 W/A + FP4 KV：PG 200 步內崩潰、MIS 在訓練後段崩潰，SC 52%、TIS 51% 穩定；INT8 W/A + INT4 KV：SC 30%、TIS 12%，其餘 < 5%。
- top-k 消融（App. A.4，Fig. 5）：$k=32$、$k=128$ 與全詞彙在所有設定相當。
- 讀圖須知（B.3、Table 2）：陰影為 ±1 standard error；Fig. 2、3 為 held-out eval accuracy，Fig. 1、4、5 為 training accuracy（取樣 rollout 的平均 reward，經置中移動平均平滑）。0.6B、1.7B 每方法 3 個 seed；30B 先每方法 1 seed，再只為表現最好的方法加 seed：FP8 KV 面板全單 seed，FP4 KV／INT4 KV 面板 SC 5／4、TIS 4／3、MIS 3／3、naive IS 2／1，其餘單 seed。Table 2：Fig. 2 18 runs 190 H100-hours；Fig. 3 108／280；Fig. 4 72／1,120；Fig. 1 47／3,750；Fig. 5 51／840；合計 296 runs、6,180 H100-hours。

#### 步驟 15：限制（§6）

SC 消除 drift，但剩下的更新在 sampler 而非 trainer 下量共變異數（Eq. 5），這在嚴重 staleness 下有影響；主要結果來自刻意嚴重的不一致與短序列，作為溫和不一致長期訓練的代理。

頁面另以「本頁說明」列出（不是論文的限制段，但都由論文內容可查）：結論句「under severe quantization, it is the only method that trains stably」與 §5.4 中 FP4 KV 下 TIS 51% 也穩定、§5.3 中 SC+TIS/MIS 同樣表現最好並不完全一致，本頁以 §5 的具體數字為準；SC 的結果只有 0.6B 與 30B（1.7B 只用於 Fig. 2）；疊加版與 staleness 只在 0.6B 測過；30B 多數曲線單 seed；「drift 是原因」的證據是 Eq. 3 的代數 + 回饋迴路論證 + 「只消 drift 就穩定」的介入實驗，沒有發散定理；「無超參數」指精確版，top-k 版有 $k$ 與 eps。

---

## 2. 學習目標（驗收測試）

每個參考答案必須以完整段落寫在指定章節的**正文**中（不能只在元件、表格、圖或答案框）。每章末以 `<details>`「檢查理解」照搬題目與參考答案。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明 TIM 是什麼、從哪來 | 什麼是 TIM？列出三類來源。 | rollout 由 sampler $q$ 產生、梯度由 trainer $p$ 計算，$q\ne p$ 即 TIM（§2.1）。來源：精度不同或兩套 codebase 的 bug；同一 kernel 因浮點不可結合與不同輸入形狀改變 reduction 順序（fp4_e2m1：$(0.5+0.5)+2=3$、$0.5+(0.5+2)=2$）；非同步與 continuous batching 造成 staleness | S1 |
| LO2 | 說明 IS 修正與它的代價 | Eq. 2 如何修正 TIM？為何實務方法都截斷比值，代價是什麼？ | 以 $p/q$ 重新加權：$\mathbb E_p[R\nabla\log p]=\mathbb E_q[(p/q)R\nabla\log p]$，序列層級精確；比值在稀有 token 上可任意大，變異膨脹，所以實務上截斷或遮罩（Table 1），代價是引入偏差（§2.2） | S2 |
| LO3 | 讀懂 SFT 與 RL 的對照實驗 | PG 在什麼條件下等於 SFT？Fig. 2 線上與離線各以哪種 reward 最不穩定？作者據此提出什麼假設？ | 二元 reward、advantage 為 +1/0（未中心化）且完全離線時；離線時負 reward（+1/−1、group-centered）不穩定、只有 +1/0 穩定；線上 +1/0 最不穩定、group-centered 最穩定；作者假設有兩個不同機制，線上的不穩定來自 teacher（sampler）跟著 student 移動（§3） | S3 |
| LO4 | 推導 $\mathbb E_p[s]=0$ 並說明 TIM 下為何不成立 | 為什麼常數 reward 下 on-policy 期望梯度為 0？TIM 下是多少？ | $\mathbb E_p[s]=\sum_vp_v\nabla\log p_v=\nabla\sum_vp_v=\nabla1=0$；TIM 下 token 由 $q$ 取樣，$\mathbb E_q[s]=\sum_vq_v\nabla\log p_v=\bar s$，一般不為 0（§4） | S4a |
| LO5 | 寫出並解讀 Eq. 3 | 寫出 drift／signal 分解。哪一項帶有 reward 的資訊？ | $\mathbb E_q[Rs_{y_t}]=\mathbb E_q[R]\bar s+\mathrm{Cov}_q(R,s_{y_t})$；drift 只依賴平均 reward，signal（共變異數）才知道哪個 token 帶來哪個 reward（Eq. 3） | S4b |
| LO6 | 說明 drift 為何造成不穩定 | drift 的方向是什麼？為什麼朝固定 teacher 蒸餾無害、TIM 下卻累積？ | $\bar s$ 是以 sampler 為 teacher 的 CE 負梯度，PG 在每個 prefix 把 trainer 蒸餾向 sampler，強度為期望 reward；固定 teacher 收斂到其最大概似擬合；sampler 是 trainer 的有偏副本且定期同步回來，誤差在回饋迴路中累積（§4）。這是文字論證，沒有定理 | S5 |
| LO7 | 說明 group centering 為何不消除 drift | group centering 讓 advantage 總和為 0，為何 drift 仍在？ | drift 發生在單一 prefix，特定 prefix 的期望 advantage 不為 0（好的 prefix 為正、含錯的為負），trainer 在前者後被拉向、後者後被推離 sampler；group centering 只縮小 drift（§4）。〔本頁推導〕$\mathbb E[A_i\mid\text{prefix}]=(1-1/G)(\mathbb E_q[R\mid\text{prefix}]-\mathbb E_q[R])$ | S6 |
| LO8 | 推導 SC 並指出它與 on-policy 的差別 | SC 怎麼定義？期望更新是多少？是無偏的 policy gradient 嗎？ | $\tilde s=s-\bar s$（Eq. 4），$\mathbb E_q[\tilde s]=0$，期望更新 $=\mathrm{Cov}_q(R,s_{y_t})$（Eq. 5）；不是：共變異數在 $q$ 而非 $p$ 下量；$p=q$ 時 SC 為 no-op | S7 |
| LO9 | 比較 SC、IS 與 value baseline | SC 與 IS 的修正方式有何不同？與 value baseline 的關係？ | IS 是隨機的乘法修正、依賴被取樣 token、增加變異、截斷後重新引入 drift；SC 是在 prefix 下確定的加法修正、可精確計算，兩者可疊加；等於期望 reward 的精確 value baseline 期望上也消除 drift，但需要 critic，SC 不需要（§4） | S7 |
| LO10 | 把 SC 寫成純量 loss | 寫出 Eq. 11，說明 sg 的作用與為何 $-\nabla L=R\tilde s$。 | $L=-R(\log p_{y_t}-\sum_v\mathrm{sg}[q_v]\log p_v)$；sg 讓 $q_v$ 不被微分，所以 $-\nabla L=R(s_{y_t}-\sum_vq_vs_v)=R(s_{y_t}-\bar s)$（App. A.3） | S8a |
| LO11 | 說明 top-k 近似與疊加 | 為何只存 top-k？$\rho$ 怎麼定義？中心化項為何只需 head？疊加 IS 時 $\alpha$ 是多少？ | 完整 logprob 一個 token 約 608KB，1024×32K 約 20TB；$\rho=(1-\sum_Hq_v)/(1-\sum_Hp_v)$，尾部用 $\rho p_v$；因 $\mathbb E_p[s]=0$，$\mathbb E_{\hat q}[s]=\sum_H(q_v-\rho p_v)s_v$（Eq. 8）；疊加時 $\alpha=\rho f(1/\rho)$（Eq. 10）；$k=128$ head 平均覆蓋 > 99.9%（30B INT4 KV 為 99.45%、最差 95.8%），$k=32$ 也與全詞彙相當；〔本頁推導〕head-only 的 IS 疊加版只在 $\hat q=q$（尾部與 trainer 成比例）時精確零期望 | S8b |
| LO12 | 讀出實驗結果與適用範圍 | 30B 三個量化等級的結果？staleness 下哪個最好、作者如何解釋？主要限制？ | FP8 KV：PG 也穩定 58%；FP4 KV：PG 200 步內崩潰，SC 52%、TIS 51%；INT8 W/A + INT4 KV：SC 30%、TIS 12%、其餘 < 5%（§5.4）；staleness（每 64 步同步）下 SC+TIS/MIS 最好且優於單獨 SC，因 SC 在 sampler 下量共變異數（§5.3）；結果來自刻意嚴重的不一致與短序列（§6）。結論句「嚴重量化下 SC 是唯一穩定的方法」不精確：FP4 KV 下 TIS 51% 也穩定，0.6B 量化下 SC+TIS/MIS 同樣最好；以 §5 的數字為準 | S9 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），圖版最寬 1080px；章節編號用 JetBrains Mono（`04a · …`）；標題用陳述句，只寫論文的結果或明標為論文的論證。全頁的視覺記號為「trainer 與 sampler 對照圖」：上列 trainer $p$（`--c-p` 實線）、下列 sampler $q$（`--c-q` 虛線），兩列之間的箭頭標「同步」；出現在 hero、S1 圖、C3 圖版頂端。每個符號在第一次使用的段落內定義；第一次出現時寫「推論引擎（sampler，$q$）」「訓練引擎（trainer，$p$）」，之後只用 sampler／trainer。

先備知識的放置（每項一到兩句，放在第一次用到的段落）：log-derivative trick $\nabla_\theta\mathbb E_p[R]=\sum_yR\nabla p=\sum_yp\,R\nabla\log p$（S1，導出 Eq. 1）；advantage 與 group centering（S3 第一段）；共變異數 $\mathrm{Cov}(A,B)=\mathbb E[AB]-\mathbb E[A]\mathbb E[B]$（S4b 開頭）；熵 $\mathrm{Ent}(p)=-\sum_vp_v\log p_v$、cross-entropy、KL（S5 開頭）；value function（critic）＝「給定 prefix 時期望 reward 的估計器」（S7 baseline 段之前）。

| # | id | 標題 | 內容（依序） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈Score centering：減去期望 score，讓 off-policy RL 穩定〉 | 副標：「推論引擎與訓練引擎的輸出略有不同時，policy gradient 的期望更新多出一項 drift；score centering 在每個 prefix 減去一個確定的修正項把它消掉。」三張事實卡（見下）。連結：arXiv、`../index.html#paper-129`、hub `score-centering-lineage-hub.html` | 1 螢幕 |
| S1 | `tim` | 01 · 訓練引擎與推論引擎的輸出並不相同 | rollout、reward、score 定義；log-derivative trick → Eq. 1；兩次 forward；TIM 定義與三類來源；fp4 例子〔本頁推導：捨入過程〕；靜態對照圖（標「rollout ↓」「梯度 ↑」「同步 ←」）。LO1 段落。延伸：`training-inference-mismatch-sources.html`、`policy-gradient-baselines-score-function.html` | 450–550 字 + 1 SVG |
| S2 | `is` | 02 · Importance sampling 能精確修正，但實務上必須截斷 | Eq. 2；序列 vs 逐 token；變異；TIS、MIS 兩列與欄位說明；`<details>` Table 1 全表（含 P/V 上標、Sign 說明、實作附註）。LO2 段落。延伸：`importance-sampling-corrections-llm-rl.html` | 350–450 字 + 小表 |
| S3 | `sft` | 03 · SFT 在離線資料下穩定；線上 RL 在 TIM 下不穩定 | advantage、group centering 定義；PG = SFT 的條件（含二元 reward）；註 4；Fig. 2 設定與圖說結論（不重繪）；group-centered 的 reward 基準（依預設應為 ±1，Fig. 2 未另述）；作者的兩機制假設（標明「作者假設」）；「teacher 隨 student 更新」。LO3 段落。延伸：`sft-vs-rl-offline-negative-rewards.html`、`ppo-grpo-dapo-llm-rl-objectives.html` | 400–500 字 |
| S4a | `escore` | 04a · On-policy 時期望 score 為 0，TIM 時不為 0 | 符號表（可收合）；固定 prefix 的視角；$\mathbb E_p[s]=0$ 推導；$\bar s$ 定義與 $\mathbb E_q[s]=\bar s$。LO4 段落 | 300–350 字 |
| S4b | `drift` | 04b · Policy gradient 更新分成 drift 與 signal | 共變異數定義；註 5 → Eq. 3；drift 不含 reward 資訊；用白話說明 $\mathbb E_q[R\mid\text{prefix}]$＝「從這個 prefix 繼續由 sampler 產生，平均能拿到的 reward」；〔本頁說明〕「若且唯若」的寬鬆處；玩具引入：參數 = logits $z$，$s_v=e_v-p$〔本頁推導〕，$\bar s=q-p$；數值例：C1 預設（λ = 0.5、常數 reward）逐步算出 $p,q,\bar s$ 與「PG = drift、signal = 0」（取第 5 節數字），再放 C1a。LO5 段落。延伸：`score-centering-drift-decomposition.html` | 450–550 字 + C1a |
| S5 | `loop` | 05 · 論文論證：drift 是朝 sampler 的蒸餾，並在回饋迴路中累積 | 熵、CE、KL 定義；CE 與 $\bar s$；固定 teacher vs 隨 student 更新的 teacher；「論文以文字論證，沒有定理」；C3〔教學示意〕＋範圍聲明（固定偏差的線性累積，不是 compounding；$t_{1/2}\propto1/\lambda$ 不是論文主張）。LO6 段落 | 450–550 字 + C3 |
| S6 | `group` | 06 · Group centering 只縮小 drift | 註 4 的 $1-1/G$；§4 的 prefix 論證；〔本頁推導〕$\mathbb E[A_i\mid\text{prefix}]$ 與 $G\to\infty$ 的極限；為何單步 bandit 看不到；C2。LO7 段落 | 350–450 字 + C2 |
| S7 | `sc` | 07 · Score centering 減去期望 score | Eq. 4、Eq. 5（逐行）；$\mathrm{Cov}_q$ vs $\mathrm{Cov}_p$；no-op；回到 C3：開啟「score centering」曲線，常數 reward 下不動，+1/0 下熵緩慢下降，作為 $\mathrm{Cov}_q\ne\mathrm{Cov}_p$ 的示意；SC vs IS、截斷生效才有 drift〔本頁推導〕；value function 定義 → baseline 段〔本頁推導：驗證式〕；C1b。LO8、LO9 段落 | 550–650 字 + C1b |
| S8a | `loss` | 08a · Score centering 可寫成純量 loss | Eq. 11、sg、符號慣例（$J$ 最大化、$L$ 最小化）、〔本頁推導〕CE 展開。LO10 段落 | 250–300 字 |
| S8b | `topk` | 08b · 只用 sampler 的 top-k logprob 就能實作 | 608KB／20TB；Eq. 7 → Eq. 8 → Eq. 6（依此順序）；§4.1 程式逐行對照（`.pg two`，≤ 600px 改單欄、`pre` 橫捲）；eps；C4；`<details>`「與 IS 疊加」：Eq. 9、10、12、兩個檢查、$\hat q=q$ 的限定。LO11 段落（含疊加的 $\alpha$，放正文不放 details） | 550–650 字 + C4 |
| S9 | `results` | 09 · 實驗：在最嚴重的量化設定下，score centering 與 TIS 的差距最大 | 共同目標 Eq. 13（註明 $L_i$ 為長度，與 loss $L$ 不同）與超參數；為何加重 TIM；§5.2–5.4 文字數字（30B 三欄表）；A.4；讀圖須知。不重繪曲線、不引用由標籤位置推得的名次。LO12 段落（含結論句的修正） | 500–600 字 + 2 表 |
| S10 | `pitfalls` | 10 · 實作設定與常見錯誤 | 常見錯誤 8 條（見下） | 300–400 字 |
| S11 | `history` | 11 · 研究脈絡 | 時間線（6.4）+ 一段：PG/baseline → PPO/GRPO/DAPO → TIM 與 FP16 → IS 修正家族 → 蒸餾/負梯度 → 本文 | 時間線 + 120 字 |
| S12 | `next` | 12 · 限制與延伸閱讀 | §6 Limitations；「本頁說明」清單（1.5 步驟 15 第二段）；筆記清單 | 250 字 + 連結 |

Hero 事實卡（不用未定義術語；卡內不出現 SC、TIS、量化等縮寫）：
1. 「強化學習訓練 LLM 時，產生回答的推論引擎與計算梯度的訓練引擎，對同一段文字給出的機率略有不同。」
2. 「這個差異讓期望更新多出一項：即使每個回答的 reward 都相同，模型也會被拉向推論引擎的分布。論文稱之為 drift。」
3. 「score centering 在每個位置減去一個由推論引擎機率算出的固定修正項，期望上 drift 為 0。論文中 30B 參數模型在最嚴重的低精度推論設定下，訓練準確率為 30%，次佳方法為 12%。」

常見錯誤（S10）：
1. 用 $\mathbb E_q[R]\bar s$ 而不是 $R\bar s$ 當修正：loss 中減的是每條 rollout 自己的 $R$ 乘 $\bar s$（Eq. 11）。
2. 忘了 sg：$q_v$、$q_v-\rho p_v$、$q_vw_v-\alpha p_v$、$w_{y_t}$ 都要 stop-gradient（Eq. 6、11、12）。
3. 只在 top-k 上算期望：$\sum_Hq_vs_v$ 不是 Eq. 8；必須減 $\rho p_v$ 項，尾部才被建模。
4. 尾部質量除以 0：§4.1 短版沒有保護，用 App. A.3 的 `eps=1e-6` 下限。
5. head 取 trainer 的 top-k：head 是 **sampler** 記錄的 top-k；trainer 的 logprob 在這些 id 上取值（`train_logp[topk_ids]`）。
6. 以為 group centering 已經足夠（§4）。
7. 以為 SC 取代 IS：嚴重 staleness 下要疊加 TIS／MIS（§5.3）。
8. 把「staleness 64」讀成最大 staleness 64：是每 64 步才同步（§5.3）。

---

## 4. 互動元件規格

共同規則：
- 全部閉式或解析更新，無數值微分；所有計算在主執行緒完成且每次 < 2ms（C3 最重：4 × 3000 步 × 5 維）；滑桿 `input` 以 `requestAnimationFrame` 合併（每影格最多重算一次），不需 Worker。
- 玩具都在 logit 座標中計算：參數就是單一 prefix 的 logits $z$，$s_v=\nabla_z\log p_v=e_v-p$〔本頁推導〕。圖版底部固定一行：「本元件的參數化（θ = logits）為本頁的簡化；論文的 score 是對整個網路參數的梯度。」softmax 一律先減最大值。向量長度用 L2 norm。
- **JS 產生的文字（讀數、aria-label、SVG 標籤）不得含 U+0304 等組合字元**：$\bar s$ 在動態文字中寫「平均 score」，$\mathbb E_q[R]$ 寫「E_q[R]」，$t_{1/2}$ 寫「t½」。SVG 內一律純 Unicode 文字，KaTeX 只用於 HTML 正文（頁面載入後一次 `renderMathInElement`，不做 lazy render：頁面公式數量不需要）。
- 很小的誤差值（恆等檢查）顯示為 `x.toExponential(1)`，並在 < 1e-9 時加「✓」；其他讀數 4 位小數。
- 鍵盤與觸控：所有逐項資訊（C1、C2、C4 的長條）都可用鍵盤取得：長條群組 `tabindex=0`，左右鍵移動選取（roving tabindex），選取項的數值顯示在固定讀數框（不跟隨指標）；每個圖版另有「以表格顯示」切換鈕，切換成同內容的 `<table>`。
- 窄螢幕（< 600px）：C4 只標 head／tail 分界與 $k$，不標每根長條的數值（數值在讀數框與表格）；`.katex-display` 可橫捲；selfTest 對每個圖版檢查 `scrollWidth <= clientWidth`。
- SVG `<pattern>`（斜線、點狀）的 stroke 用 `currentColor`，容器設 `color: var(--ink-2)`，深色模式自動跟隨。
- `prefers-reduced-motion: reduce`：取消所有 transition；C3 直接畫出最終曲線（本來就不做動畫）。

### C1 · drift 與 signal（C1a 在 S4b，C1b 在 S7，共用狀態）

- **常數**：4 個 token（編號 1–4）。trainer logits $z=[1.2,0.6,0,-0.8]$；sampler 偏移方向 $\xi=[-0.5,0.8,0.2,-0.5]$；$p=\mathrm{softmax}(z)$，$q=\mathrm{softmax}(z+\lambda\xi)$。
- **輸入**：滑桿 $\lambda\in[0,2]$，步長 0.05，預設 0.5（標「不一致程度 λ」）；reward chip：「常數 +1」$Q=[1,1,1,1]$（預設）、「+1/0」$Q=[0.9,0.2,0.6,0.1]$、「±1」$Q=[0.8,-0.6,0.2,-0.8]$（＝2×(+1/0) − 1）；四個 $Q_v$ 滑桿 $[-1,1]$、步長 0.05（改動後 chip 變「自訂」）。
- **計算**（$\bar R_q=q\cdot Q$，$\bar R_p=p\cdot Q$）：$\bar s=q-p$；PG $=q\odot Q-\bar R_q\,p$；drift $=\bar R_q(q-p)$；signal $=q\odot(Q-\bar R_q)$；on-policy $=p\odot(Q-\bar R_p)$〔本頁推導：$s_v=e_v-p$ 代入 Eq. 3〕。C1b 另算：SC $=$ signal；逐 token 精確 IS（$f=r$）$=$ on-policy；TIS $=(q\odot w\odot Q)-\big(\sum_vq_vw_vQ_v\big)p$，$w=\min(p/q,2)$。分解誤差 $\|\text{PG}-\text{drift}-\text{signal}\|$。
- **顯示**：上排 $p$、$q$ 並排長條（$p$ 實心、$q$ 斜線）。C1a：每 token 一組，系列 PG（實心）／drift（斜線）／signal（點狀）；讀數 E_q[R]、‖平均 score‖、‖PG‖、‖drift‖、‖signal‖、分解誤差。C1b：系列 SC／on-policy／TIS；讀數 ‖SC − on-policy‖、‖TIS − on-policy‖、「max p/q = …（> 2 時 TIS 截斷生效）」。C1b 圖版註：「$Q_v$ 固定，未建模 continuation 在 $q$ 與 $p$ 下的差異；所以這裡逐 token IS 恰好等於 on-policy，真實序列中逐 token IS 不修正 prefix 的分布。」
- **數值保護**：softmax 使 $q_v>0$，$p/q$ 不會除以 0。
- **讀者應觀察到**：(1) 常數 reward：signal 為 0，PG 全部是 drift；λ = 0 時 drift 也為 0。(2) +1/0 與 ±1 的 signal 只差 2 倍（共變異數對平移不變），PG 卻差一個 $\bar s$；±1 時 drift 很小，因為這個 prefix 的 E_q[R] 接近 0（對照 S4b 的「本頁說明」）。(3) C1b：λ 增大，SC 與 on-policy 的差變大。(4) C1b：λ < 1.068 時 TIS 與 on-policy 完全相同（截斷未生效），之後出現差異。
- **無障礙**：`aria-valuetext`（「λ = 0.50」）；圖 `role="img"`，`aria-label` 與讀數 `aria-live="polite"` 在 `change` 或停止輸入 500ms 後更新。

### C2 · group centering 下各 prefix 的 drift 係數（S6）〔教學示意〕

- **模型**：每條 rollout 第一步選路徑甲或乙（sampler 選甲的機率 $\pi_甲$），之後成功的機率 $h_甲$、$h_乙$。第二步的 drift 為「係數 $c$ × 該 prefix 的 $\bar s$」，本元件只畫係數。
- **輸入**：$h_甲\in[0,1]$ 預設 0.8；$h_乙\in[0,1]$ 預設 0.2；$\pi_甲\in[0.05,0.95]$ 預設 0.5（步長 0.01）；$G\in\{2,4,8,16,64\}$，預設 8。
- **計算**（$\bar R_{01}=\pi_甲h_甲+(1-\pi_甲)h_乙$）：+1/0：$c_甲=h_甲$，$c_乙=h_乙$；±1：$c=2h-1$；group-centered（由 ±1，Eq. 13）：$c=(1-\tfrac1G)\,2(h-\bar R_{01})$〔本頁推導〕。摘要：prompt 平均 $\pi_甲c_甲+(1-\pi_甲)c_乙$、均方根 $\sqrt{\pi_甲c_甲^2+(1-\pi_甲)c_乙^2}$。不放 score centering 列（S7 才定義；S7 正文回指：SC 下兩個係數都是 0）。
- **顯示**：三列（三種 reward）× 兩根水平長條（甲實心、乙斜線），以 0 為中心；數值旁附「拉向 sampler」（> 0）／「推離 sampler」（< 0），照 §4 原文 "pulled toward / pushed away"。右側摘要欄。圖版註：「group-centered 由 ±1 計算（Eq. 13，論文預設）。」
- **讀者應觀察到**：group-centered 的 prompt 平均恆為 0，但甲、乙各自不為 0、正負相反；只有 $h_甲=h_乙$ 時歸零；$G$ 越大，係數越接近 $2(h-\bar R_{01})$（以 prompt 平均為基準），它與 ±1 的值只在 prompt 平均 ±1 reward 為 0 時相同（例：$\pi_甲=0.5$、$h_甲+h_乙=1$）；$\pi_甲=0.8$ 時 $G\to\infty$ 的 $c_甲=0.24$，±1 為 0.60。
- **驗證**：`check.py` 以 40 萬組蒙地卡羅核對閉式（誤差 < 0.01），並核對 $G=10^6$ 的極限值。

### C3 · 固定偏差下的 drift 累積（S5；S7 再開啟 SC 曲線）〔教學示意〕

圖版首行（`--ink-2`）：「此模型為本頁設計的示意，不是論文的實驗，也不重現 Fig. 1–5。sampler 與 trainer 之間是一個固定的 logit 偏差（類似量化），所以這裡看到的是固定偏差在每次同步後線性累積，不是論文所說誤差互相放大的回饋迴路，也沒有模擬 staleness。」

- **模型**：5 個 token，初始 logits $z_0=0$（$\mathrm{Ent}=\ln5=1.609$）。$q_n=\mathrm{softmax}(z_n+\lambda\xi)$，$\xi=[0.6,-0.2,0.1,-0.3,-0.2]$（每步由 trainer 同步，偏差固定）。期望更新的梯度上升 $z_{n+1}=z_n+\eta\,g_n$（無取樣雜訊）。曲線：
  - PG（sampler 隨 trainer 同步）：$g=q_n\odot Q-(q_n\cdot Q)p_n$。
  - PG（固定 teacher）：同上但 $q\equiv\mathrm{softmax}(z_0+\lambda\xi)$。
  - on-policy（＝逐 token 精確 IS）：$g=p_n\odot(Q-p_n\cdot Q)$。
  - score centering（勾選框「顯示 score centering（S7 定義）」，S5 預設不勾；S7 內有按鈕捲回 C3 並勾選）：$g=q_n\odot(Q-q_n\cdot Q)$。
- **輸入**：reward chip「常數 R = 1」（預設，$Q=\mathbf 1$）／「+1/0」（$Q=[1,1,0,0,0]$，兩個正確 token 同分）；滑桿 $\lambda\in[0.01,1]$ 對數刻度，預設 0.1；$\eta\in\{0.5,1,2\}$，預設 1；總步數 $N=3000$。
- **顯示**：主圖 trainer 熵 $\mathrm{Ent}(p_n)$（nats）對步數，**橫軸對數** $n+1\in[1,3001]$（早期崩潰在 10–300 步，線性軸看不到）；曲線以實線／長虛線／細實線／點線＋線尾標籤區分；副圖（只在 +1/0）trainer 準確率 $p_n\cdot Q$。讀數：各曲線在第 100、1000、3000 步的熵；各曲線熵首次 ≤ 初值一半的步數 t½（未達到顯示「未達到」）；PG（同步）的 $\|z_N-z_0\|$。圖下一行：「t½ 約與 1/λ 成正比，這是固定線性偏差的結果，不是論文的主張。」
- **數值保護**：熵用 $-\sum p\log\max(p,10^{-300})$；softmax 減最大值（不對 $\|z\|$ 大小做保證）。
- **讀者應觀察到**（`check.py`）：(1) 常數 reward：on-policy 不動；固定 teacher 收斂到 $p=q$，熵停在 $\mathrm{Ent}(\mathrm{softmax}(\lambda\xi))$；只有同步的 PG 持續把機率集中到 sampler 偏好的 token，熵趨近 0。(2) λ 越小 t½ 越長（0.05→280、0.1→140、0.5→29、1.0→15），仍會發生。(3) 固定 teacher 在 λ = 1 也停住，對應 §6「重點是 teacher 是否隨 student 更新」這句話的示意。(4) S7 開啟 SC：常數 reward 下 SC 完全不動；+1/0 下 SC 熵緩慢下降（λ = 0.1 時約 473 步降到 0.35，PG 在 100 步內），on-policy 停在約 ln 2。正文只把它當成 $\mathrm{Cov}_q\ne\mathrm{Cov}_p$ 的示意：SC 剩下的更新以 sampler 機率加權，在兩個同分 token 之間偏向 sampler 偏好者；不連結到 §5.3 的 staleness 結果，也不宣稱 LLM 中會發生。
- **無障礙**：曲線不靠顏色；改變輸入後 500ms 宣告「λ = 0.10，PG 第 3000 步熵 0.032，t½ = 140」。

### C4 · top-k 重建與 IS 疊加（S8b）

- **常數**：$V=20$；trainer logits $z_v=-0.4v$（$v=0..19$）；sampler 擾動 $\xi$ =
  `[0.31,-1.12,0.54,0.87,-0.43,1.25,-0.76,0.12,-1.38,0.66,0.95,-0.21,-0.88,1.47,0.03,-0.59,1.06,-1.21,0.38,-0.27]`；$q=\mathrm{softmax}(z+\sigma\xi)$。
- **輸入**：$\sigma\in[0,1.5]$ 步長 0.05 預設 0.5；$k\in\{1..20\}$ 預設 8；權重函數：SC（$f=1$，預設）／TIS + SC／MIS + SC／精確 IS（$f=r$）。
- **計算**：
  - head $H$＝$q$ 最大的 $k$ 個 token（stable sort），tail ＝其餘。
  - 尾部質量 $m_q^{T}$、$m_p^{T}$：直接對 tail 求和（避免 $1-\sum$ 相消），以 $\max(\cdot,10^{-6})$ 下限（App. A.3 的 `eps`）；$\rho=m_q^T/m_p^T$；$\alpha=\rho f(1/\rho)$。$k=20$ 時兩者被下限為 $10^{-6}$，$\rho=1$，tail 為空。
  - $w=f(p/q)$；精確中心化向量 $c=q\odot w-(q\cdot w)p$；head-only（Eq. 8／10）$\hat c=\sum_{v\in H}(q_vw_v-\alpha p_v)(e_v-p)$。
  - 恆等檢查：在 $\hat q$ 上全詞彙求和與 $\hat c$ 的差（< 1e-12）。圖版註：「恆等檢查只驗證 Eq. 8／10 的代數（相對於重建的 $\hat q$）；相對於真實 $q$ 的誤差是下方的殘餘。疊加 IS 時，head-only 版只在 $\hat q=q$ 時精確零期望。」
  - 殘餘 drift（每單位 E_q[R]）$\|c-\hat c\|$；相對殘餘 $\|c-\hat c\|/\|q-p\|$，**$\|q-p\|<10^{-12}$ 時顯示「—（無 drift）」**；IS 單獨留下的 drift $\|c\|$。
- **顯示**：20 根長條（對數縱軸 $[10^{-4},1]$）：$p$ 細線標記、$q$ 實心、$\hat q$ 的尾部斜線；head 區塊底色帶與「head（k = 8）」標籤。讀數表：$\rho$、$\alpha$、$m_q^T$、$m_p^T$、‖c‖、‖ĉ‖、殘餘、相對殘餘、恆等誤差、max／min $p/q$。
- **讀者應觀察到**：(1) $k$ 增大，相對殘餘快速下降（σ = 0.5、SC：$k=1,4,8,16$ → 0.9445、0.3111、0.0650、0.0010）。(2) σ = 0 時 $\rho=1$、相對殘餘顯示「—」。(3) 精確 IS：$\alpha=1$、$\hat c=c=0$。(4) MIS 在 σ = 0.5 時比值都在 [0.5, 5]（0.518–2.152），$\|c\|=0$；σ = 1.0 時有比值出界，$\|c\|=0.0513$。

---

## 5. 驗收數值表（`verify/check.py` → `check.out`）

容許誤差：所有讀數 ±0.0005（同一算法，只差浮點加總順序）；整數步數 ±1。

C1（$\lambda=0.5$；$p=[0.5037,0.2764,0.1517,0.0682]$，$q=[0.3826,0.4022,0.1635,0.0518]$，$\bar s=[-0.1211,0.1257,0.0118,-0.0164]$，‖平均 score‖ = 0.1758）：

| reward | E_q[R] | ‖PG‖ | ‖drift‖ | ‖signal‖ = ‖SC‖ | ‖on-policy‖ | ‖SC − on-policy‖ |
| --- | --- | --- | --- | --- | --- | --- |
| 常數 +1 | 1.0000 | 0.1758 | 0.1758 | 0.0000 | 0.0000 | 0.0000 |
| +1/0 | 0.5280 | 0.1082 | 0.0928 | 0.1957 | 0.1889 | 0.0270 |
| ±1 | 0.0560 | 0.3818 | 0.0098 | 0.3913 | 0.3778 | 0.0540 |

$\lambda=1.0$、+1/0：‖drift‖ 0.1582、‖SC − on-policy‖ 0.0480。C1b 的 TIS：截斷在 $\lambda\approx1.068$ 開始生效；常數 reward 下 ‖TIS‖ 在 $\lambda=1.5$、2.0 為 0.0909、0.1675（此前為 0）；+1/0 下 ‖TIS − on-policy‖ 在 1.5、2.0 為 0.0856、0.1576。

C2（預設 $h_甲=0.8,h_乙=0.2,\pi_甲=0.5$）：

| 設定 | 模式 | c_甲 | c_乙 | prompt 平均 | 均方根 |
| --- | --- | --- | --- | --- | --- |
| G = 8 | +1/0 | +0.8000 | +0.2000 | +0.5000 | 0.5831 |
| G = 8 | ±1 | +0.6000 | −0.6000 | 0.0000 | 0.6000 |
| G = 8 | group-centered | +0.5250 | −0.5250 | 0.0000 | 0.5250 |
| G = 2 | group-centered | +0.3000 | −0.3000 | 0.0000 | 0.3000 |
| G = 8、π_甲 = 0.8 | group-centered | +0.2100 | −0.8400 | 0.0000 | 0.4200 |
| G = 10⁶、π_甲 = 0.8 | group-centered | +0.2400 | −0.9600 | 0.0000 | — |
| G = 10⁶、π_甲 = 0.5 | group-centered | +0.6000 | −0.6000 | 0.0000 | — |
| G = 8、h_甲 = h_乙 = 0.5 | group-centered | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

C3（$\eta=1$、$N=3000$；熵 Ent 單位 nats，初值 1.6094；C2 的 G = 10⁶ 列為極限檢查，頁面 G 選項最大 64）：

| 模式 | λ | PG（同步）Ent@100／1000／3000 | t½ | 固定 teacher Ent@3000 | SC Ent@3000 | on-policy Ent@3000 |
| --- | --- | --- | --- | --- | --- | --- |
| 常數 | 0.05 | 1.514／0.182／0.061 | 280 | 1.609 | 1.609 | 1.609 |
| 常數 | 0.1 | 1.124／0.091／0.032 | 140 | 1.609 | 1.609 | 1.609 |
| 常數 | 0.5 | 0.201／0.023／0.008 | 29 | 1.595 | 1.609 | 1.609 |
| 常數 | 1.0 | 0.114／0.014／0.005 | 15 | 1.545 | 1.609 | 1.609 |
| +1/0 | 0.1 | 0.211／0.025／0.009 | 24 | 0.701 | 0.124 | 0.697 |
| +1/0 | 0.5 | 0.090／0.011／0.004 | 12 | 0.681 | 0.035 | 0.697 |

常數模式 λ = 0.1 的 ‖z₃₀₀₀ − z₀‖：PG（同步）6.19、固定 teacher 0.07、SC 與 on-policy 0.00。t½ 讀數（各曲線）：常數模式只有 PG（同步）有值，其餘顯示「未達到」；λ × t½ 在 λ = 0.02、0.05、0.1、0.2 都是 14.0（玩具中 t½ ∝ 1/λ，不是論文主張）。+1/0 模式中 t½ 對所有曲線都會達到（ln 5 → ln 2 本身就低於一半：λ = 0.1 時 PG 24、SC 46、on-policy 66、固定 teacher 143），所以該模式頁面改看 Ent@3000 與「熵 ≤ 0.35 的步數」：SC λ = 0.1 為 473、λ = 0.5 為 93（PG 皆在 100 步內）。固定 teacher 的理論終點：常數模式 $\mathrm{Ent}(\mathrm{softmax}(\lambda\xi))$（λ = 0.1：1.6089；λ = 1.0：1.5455）。

C4（$k$ 列出 1／4／8／16／20；σ = 0 時 ‖q − p‖ = 0，相對殘餘顯示「—」）：

| σ | f | ρ（k = 8） | α（k = 8） | 相對殘餘 k = 1／4／8／16／20 | ‖c‖ |
| --- | --- | --- | --- | --- | --- |
| 0.5 | SC | 0.9530 | 0.9530 | 0.9445／0.3111／0.0650／0.0010／0 | 0.1262 |
| 0.5 | TIS | 0.9530 | 1.0000 | 0.0081／0.0081／0.0081／0.0000／0 | 0.0010 |
| 0.5 | MIS、IS | 0.9530 | 1.0000 | 全為 0 | 0.0000 |
| 1.0 | SC | 0.6162 | 0.6162 | 0.9865／0.1948／0.0650／0.0009／0 | 0.2100 |
| 1.0 | TIS | 0.6162 | 1.0000 | 0.4566／0.3141／0.0657／0.0013／0 | 0.0959 |
| 1.0 | MIS | 0.6162 | 1.0000 | 0.2445／0.0719／0.0719／0.0000／0 | 0.0513 |
| 1.0 | IS | 0.6162 | 1.0000 | 全為 0 | 0.0000 |

其他：σ = 1.0、k = 4 時 ρ = 0.4160、TIS 的 α = 0.8320（$f(1/\rho)=\min(2.404,2)=2$）；σ = 0.5 時 max／min $p/q$ = 2.152／0.518，σ = 1.0 時 5.173／0.299；恆等誤差全部 < 1e-16 量級。

---

## 6. 實作計畫

沿用 `tutorials/prediction-targets-k-diff.html` 的頁殼：CSS tokens（含深色模式三段覆寫與 `pm-theme`）、appbar、`#pm-bar`、章節 TOC（桌機 sticky 左欄／窄螢幕橫列）、`.plate`／`.plate-h`／`.plate-f`／`.readout`／`.seg`／`.rng`／`.tag`（本頁推導 pill）／`.check` details、`.pg two` 程式兩欄格線、`.tscroll` 表格捲動、KaTeX 0.16.9 setup 與 `renderMathInElement`、`announce()` debounce、`el()`、時間線 rail、`?test` 的 `selfTest()`。單檔 `tut/129/build/index.html`，CSS/JS/時間線全內嵌，不 fetch 本機檔。以下只列新增或不同處。

### 6.1 新增 tokens 與標籤

- `--c-p`（trainer）light #3F6E8C／dark #8DB4D3；`--c-q`（sampler）light #6B7A3A／dark #A9BC7C（即參考頁的 `--c-par`／`--c-perp`，改名以符合語意；對比已在參考頁驗證）。trainer 實線、sampler 虛線，任何圖都不只靠顏色。
- 新 pill：「教學示意」（`.tag.toy`，與「本頁推導」同樣式、文字不同）。C2、C3 圖版標題旁必放；C1、C4 不放（它們只是 Eq. 3／8／10 在小詞彙上的精確計算，參數化簡化另有圖版註）。
- 長條填滿樣式：SVG `<pattern>` 斜線（`hatch`）與點狀（`dots`），供 C1、C2、C4 區分系列。

### 6.2 程式結構（單一 IIFE）

- 共用：`softmax(z)`、`dot(a,b)`、`norm(a)`、`entropy(p)`、`fmt(x,n)`（負號用 U+2212）、`fmtErr(x)`（`toExponential(1)`，< 1e-9 加「✓」）、`barKeys(group, n, onSel)`（roving tabindex＋左右鍵）、`tableToggle(plate)`（以表格顯示）。
- KaTeX 中 CJK 下標寫 `h_{\text{甲}}`、`\pi_{\text{甲}}`。
- C1：`C1 = {Z, XI, PRESET}`；`state1 = {lam, Q, preset, sel}`；`c1Compute(lam,Q)` → `{p,q,Rq,Rp,sbar,pg,drift,signal,onp,tis,maxR}`；`renderC1a()`、`renderC1b()`（兩者都在 `state1` 改變時重畫）。
- C2：`state2 = {hJia,hYi,piJia,G,sel}`；`c2Compute()` → 三模式 `{c1,c2,avg,rms}`；`renderC2()`。
- C3：`C3 = {XI, Q:{const,b01}}`；`state3 = {mode, lam, eta, showSC}`；`c3Run()` 一次跑完四條 3000 步曲線（SC 也算，只是預設不畫；Float64Array 存熵與準確率），`tHalf(arr)` 回傳步數或 `null`（顯示「未達到」），`renderC3()`。滑桿拖動時以 rAF 合併；實測若單次 > 8ms，改為拖動中只跑到 1000 步、`change` 後補完（預期 < 2ms，不會觸發）。
- C4：`C4 = {V, Z, XI}`；`F = {sc, tis, mis, is}`；`state4 = {sigma, k, f, sel}`；`c4Compute()` → 第 4 節所列全部量；`renderC4()`。
- `selfTest()`（`?test`）新增斷言（誤差 5e-4）：C1 +1/0 λ = 0.5 的 ‖drift‖ 0.0928、‖signal‖ 0.1957；常數 reward 時 ‖signal‖ < 1e-12；C2 預設 group-centered $c_甲=0.525$、prompt 平均 0，π_甲 = 0.8、G = 10⁶ 時 $c_甲=0.24$；C3 常數 λ = 0.1 PG（同步）Ent@3000 = 0.032、$t_{1/2}=140$、SC 與 on-policy 的 t½ 為「未達到」；C4 σ = 0 時相對殘餘字串為「—」；C4 σ = 0.5、k = 8、SC 相對殘餘 0.0650、恆等誤差 < 1e-12、IS 時 ‖ĉ‖ < 1e-12；以及沿用的 `scrollWidth <= innerWidth`。輸出 `selfTest: N passed, M failed`。

### 6.3 連結

- Hero：arXiv `https://arxiv.org/abs/2609.20807`；「在關聯圖中查看」`../index.html#paper-129`；程式碼 `https://github.com/martin-marek/score-centering`（註 1）。
- 筆記（同目錄 `<slug>.html`）：hub `score-centering-lineage-hub`；依 hub 閱讀順序 `policy-gradient-baselines-score-function`、`ppo-grpo-dapo-llm-rl-objectives`、`training-inference-mismatch-sources`、`importance-sampling-corrections-llm-rl`、`sft-vs-rl-offline-negative-rewards`、`score-centering-drift-decomposition`；hub 另連的站內既有筆記 `rlvr-grpo-credit-assignment`、`knowledge-distillation-foundations`、`on-policy-distillation-imitation`、`rl-mdp-q-learning-basics`（已確認存在於 `scripts/notes_split/`，直接連結）。

### 6.4 時間線（S11）

- `notes/timeline.json` 原樣貼入 `<script type="application/json" id="timeline-data">`（15 筆，含本文 2609.20807）。
- `TL_GROUP`（arXiv → [類別, 筆記 slug]）：
  - 蒸餾／SFT：1503.02531、2501.12948、2506.01347 → `sft-vs-rl-offline-negative-rewards`
  - RL 目標：1707.06347、2402.03300、2503.14476、2503.20783 → `ppo-grpo-dapo-llm-rl-objectives`
  - TIM 來源：2505.24298、2510.26788 → `training-inference-mismatch-sources`
  - IS 修正：2503.14286、2507.18071、2510.18855、2602.04879、2603.19470 → `importance-sampling-corrections-llm-rl`
  - 本文：2609.20807 → `score-centering-drift-decomposition`
- 呈現與缺欄處理沿用參考頁；本文卡片 `--accent` 外框。

### 6.5 自我檢查

截圖 390／768／1024／1440 + 深色、鍵盤 focus、reduced-motion；八項評分迴圈（AGENTS.md 步驟 6-3）；`?test` 全過；逐條核對 12 個 LO 的參考答案都在正文段落中；grep 建置檔確認沒有 U+0304 等組合字元出現在 JS 字串裡；確認頁面沒有任何「Fig. n 中 X 排在 Y 之上」類的讀圖名次。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 圖的名次 | 論文沒有曲線數值表；顧問筆記的名次判斷來自文字抽取檔的標籤位置，不可靠（例：Fig. 4 右圖單獨 SC 的位置）。 | 只引用正文與圖說的文字結論（§5.2–5.4、各圖 caption）；不重繪曲線；S9 註明「論文未提供曲線數據」。 |
| R2 | 玩具模型被誤認為論文結果 | C2、C3 是本頁設計；C3 的熵崩潰、$t_{1/2}\propto1/\lambda$ 都不是論文量測或主張；C3 是固定偏差的線性累積，不是論文的 compounding 回饋迴路。 | 「教學示意」pill + 圖版首行範圍聲明 + t½ 圖下聲明；正文以「在這個玩具中」開頭；LO 的參考答案不依賴 C2/C3 的數字（LO7 的公式標〔本頁推導〕）。 |
| R3 | C3 中 SC 也降熵 | +1/0 模式下 SC 熵緩慢下降，讀者可能以為與論文矛盾。 | SC 曲線在 S5 預設隱藏，S7 才開啟；正文只把它當成 $\mathrm{Cov}_q\ne\mathrm{Cov}_p$ 的示意，不連結到 §5.3 的 staleness 結果（C3 的偏差是固定的，不模擬 staleness），不宣稱 LLM 中會發生。 |
| R4 | logit 座標簡化 | $s_v=e_v-p$ 只在「參數 = logits」時成立；真實網路還乘上 Jacobian。 | 每個元件圖版註明；S4b 正文一句說明。 |
| R5 | 結論句與 §5 不完全相容 | 「only method that trains stably」vs FP4 KV 下 TIS 51% 穩定、SC+TIS/MIS 同樣最好。 | 頁面以 §5 數字為準，S12「本頁說明」列出。 |
| R6 | 站外筆記 | hub 連到 `rlvr-grpo-credit-assignment`、`knowledge-distillation-foundations`、`on-policy-distillation-imitation`、`rl-mdp-q-learning-basics` 四篇非本批筆記。本批 6 篇 topic（含 `ppo-grpo-dapo-llm-rl-objectives`、`importance-sampling-corrections-llm-rl`）與本頁一起上架，直接連結。 | 已確認四篇都在 `scripts/notes_split/`，直接連結。 |
| R7 | Fig. 2 細節未給 | 擾動大小、離線資料產生方式、group-centered 的基準 reward 皆未說明。 | S3 照寫「論文未給」擾動大小與離線資料；group-centered 寫「依 B.1 預設（Eq. 13）應為 ±1，Fig. 2 未另述」；C2 採 ±1。 |
| R8 | Table 2 的 Fig. 5「51 runs」 | 顧問依圖例 n 加總對不上（59 或 41）。 | 頁面照錄 51，不做推論。 |
| R9 | 模型名稱 | §1 寫 Qwen3-0.6B，§5.1/B.1 寫 Qwen3-0.6B-Instruct。 | 頁面用 Qwen3-0.6B-Instruct，首次出現加註 §1 的寫法。 |
| R10 | 相對連結 | 建置位置與上線位置不同。 | 接受；預覽只檢查外部連結與頁內錨點。 |

---

## 8. 修訂紀錄（round 1）

審查合併與處理見 `reviews/round1.md`（27 項：P0 0、P1 15、P2 12）。主要變更：

| 項目 | 變更 |
| --- | --- |
| 1 | C3 改名「固定偏差下的 drift 累積」；圖版首行明寫是固定（類量化）偏差的線性累積，不是論文的 compounding 回饋迴路，不模擬 staleness；t½ ∝ 1/λ 標為「不是論文主張」（步驟 7、C3、R2）。 |
| 2 | C3 的 SC 曲線 S5 預設隱藏、S7 開啟，只作為 Cov_q ≠ Cov_p 的示意；刪除與 §5.3 staleness 的連結（C3、R3）。 |
| 3 | C2「G 越大越接近 ±1」改為「趨近 2(h − prompt 平均 reward)，只有 prompt 平均 ±1 reward 為 0 時等於 ±1 值」；驗算加 G = 10⁶ 極限（步驟 8、C2、第 5 節）。 |
| 4 | 步驟 6 加〔本頁說明〕：§1 的「drift = 0 若且唯若 p = q」為寬鬆說法（E_q[R] = 0、Jacobian 零空間）；Hero 不再用這句。 |
| 5 | S4 拆為 S4a（E_p[s] = 0，LO4）／S4b（Eq. 3 + 玩具 + 數值例，LO5）；S8 拆為 S8a（Eq. 11，LO10）／S8b（top-k，Eq. 7→8→6，疊加放 details，LO11）。 |
| 6 | 前向引用：advantage 與 group centering 移到 S3 第一段定義；C2 移除 SC 列；C3 SC 曲線延到 S7。先備知識（log-derivative trick、共變異數、熵/CE/KL、value function）各放在第一次使用處。 |
| 7 | 符號去衝突：λ（取代 ε）、Ent（取代 H）、N（取代 T）、m_q^T／m_p^T、token 1–4、路徑甲／乙與 h_甲／h_乙、註明 L_i。 |
| 8 | LO3 加「二元 reward」；LO11 加 q̂ = q 的限定；LO12 加結論句的修正；步驟 12「只有」改為充分條件；Table 1 恢復 P/V 上標；Fig. 2 group-centered 改寫為「依預設應為 ±1」。 |
| 9 | C1b 加「Q_v 固定、未建模 continuation」註；C4 加恆等檢查範圍註、σ = 0 顯示「—」；小誤差以指數表示。 |
| 10 | 元件共同規則：鍵盤 roving tabindex、「以表格顯示」、窄螢幕標籤策略、SVG 純 Unicode、pattern 用 currentColor、reduced-motion；C3 橫軸改對數、t½「未達到」、拿掉 ‖z‖ < 10 的保證。 |
| 11 | 標題改為字面陳述：S3、S5（標明「論文論證」）、S9（限定量化與 0.6B/30B）；視覺記號改稱「trainer 與 sampler 對照圖」；Hero 卡去除未定義縮寫，卡 3 寫成「每個位置減去固定修正項」；S2 只放 TIS/MIS 兩列，全表進 details。 |
| 12 | 未採用：Eq. 6/11 括號（PDF 第 6 頁有括號，原設計正確）；KaTeX lazy render（公式量小，一次渲染即可）。 |


## 9. 修訂紀錄（readiness）

- Blocker：Eq. 10 的 $\hat w$ 未定義 → 步驟 13 在首次出現處定義 $\hat w_v=f(p_v/\hat q_v)$，並加入 §1.4 符號表；頁面 S8b 同樣在首次出現處定義。
- S9 標題縮小範圍為「在最嚴重的量化設定下，score centering 與 TIS 的差距最大」（FP8 KV 下 PG 已達 58%、FP4 KV 下 SC 52% 對 TIS 51%）。
- 四篇非本批筆記已確認存在於 `scripts/notes_split/`，直接連結（R6）。

## 10. 建置紀錄

- 單檔 `build/index.html`（頁殼沿用 `tutorials/prediction-targets-k-diff.html` 的 CSS、TOC、`#pm-bar`、KaTeX 與 selfTest）。`?test`：79 passed、0 failed（1440 與 390）。impeccable：0 anti-patterns（只剩 numbered-section-labels advisory）。
- 與設計的差異：C1、C2、C4 的長條以左右鍵選取、點選也可；C2 的數值標籤移到左側欄（避免與長條重疊）；C3 的 on-policy 曲線先畫、固定 teacher 的虛線畫在其上（兩者重合時可見）；legend 色塊改為 inline SVG（impeccable 標記 repeating-gradient）；`.ch-num` 字距 .06em → .04em（wide-tracking）。
