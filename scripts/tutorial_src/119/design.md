# 設計文件：〈Latent On-Policy Self-Distillation〉互動教學

- 論文：Guibin Zhang、Jiayang Lyu 等，arXiv 2608.13040v1（2026-08-13）。資料庫編號 No. 119。程式碼 github.com/bingreeky/LOPD。
- 原文：`scripts/tutorial_src/119/src/paper.txt`（PDF 同目錄）。本文件的「Eq. n」「§n」「App. X」「Fig. n」「Table n」「Alg. 1」皆指論文原文編號。
- 顧問筆記：`scripts/tutorial_src/119/advisor.md`（本文件已吸收；頁面不引用它）。
- 產出：`scripts/tutorial_src/119/build/index.html`（單檔）。上線位置 `tutorials/latent-on-policy-self-distillation.html`（slug `latent-on-policy-self-distillation`）；站內連結一律以上線位置撰寫。
- 驗算：`scripts/tutorial_src/119/verify/check.py` → `check.out`（已執行完畢，無背景工作）。第 5 節所有驗收值都由它以**與頁面相同的演算法**產生。
- 研究筆記：`scripts/tutorial_src/119/notes/`（hub `lopd-lineage-hub` + 6 篇主題筆記 + 13 筆 `timeline.json`）。
- 標籤：**〔本頁推導〕**＝論文沒有寫、本頁自行推導或算術（`check.py` 已數值驗證）；**〔示範值〕**＝本頁自訂的玩具數字；**作者認為**＝論文作者的解釋，不是量測結果。本論文沒有任何定理或證明，頁面不得出現「定理」「證明了」等字。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

On-policy 自蒸餾（OPSD）讓同一個語言模型扮演兩個角色：學生只看任務與互動歷史、自己取樣軌跡；老師多看一段「特權脈絡」（privileged context，例如答案、回饋、技能或成功軌跡），在學生走過的每個 token 位置給出下一個 token 的分布，學生以 token 層級的 divergence 向老師靠近。既有方法的特權脈絡都由設計者的固定規則產生（Eq. 2）。LOPD 把這個規則參數化：從經驗庫檢索 3 筆成功軌跡，用「凍結 backbone＋LoRA 的 encoder」與 QFormer 式壓縮器把每筆壓成 32 個連續向量，共 96 個 latent token 放進老師的輸入（Eq. 4–7）。老師是**凍結的 base checkpoint θ̄ 加上 latent 脈絡**，不是與學生同權重的模型（§3.3）。蒸餾損失是老師 top-M（M=20）加合併尾部項的 reverse KL（Eq. 9–10），而且老師一側沒有 stop-gradient，梯度穿過凍結網路回到 composer。作者指出 composer 只最小化蒸餾損失時，老師分布會移向學生，因此加入 privileged-margin 約束：以結果獎勵 A=2r−1 加權老師在學生取樣 token 上的 log 機率優勢 Δ，要求 Δ ≥ m=0.05，用對偶變數 β 執行（Eq. 11–13）。推論時只留學生。實驗在工具使用（EnvScaler、BFCL-v3、ACEBench）與程式生成（LiveCodeBench、EvalPlus）上，10 組 backbone × benchmark 的彙總欄都是 LOPD 最高（Table 1–2）；消融顯示 m=0 訓練出的學生（0.551）低於以凍結 composer 訓練出的學生（0.573），m=0.05 時最高（0.637）（Fig. 3）。多數差距很小，且論文沒有重複訓練的變異估計。

### 1.2 研究動機（§1）

1. OPD 在學生自己走到的狀態上給密集監督，比離線模仿更少訓練／推論不匹配，比 RLVR 的稀疏結果獎勵更細（§1）。
2. OPSD 不需要更強的外部老師，但把核心問題從「哪個老師夠強」換成「老師該看什麼脈絡」（§1）。
3. 既有脈絡（答案、反思文字、成功 rollout、檢索文件）都由設計者事先決定格式；作者主張這限制了「老師學會從經驗中保留什麼」（§1 Research Question）。這是主張，不是量測結果。

### 1.3 貢獻（§1 原文三點）

1. Context re-formulation：特權脈絡由手工文字改為端到端可學習的 latent substrate。
2. LOPD：檢索經驗 → latent 脈絡 → 與學生聯合 on-policy 蒸餾；privileged-margin 防止老師退化成學生。
3. 實驗：三個 backbone、七個 benchmark 上超過 RLVR 與 OPSD 類方法；以「少於 30% 的 rollout 預算」超過 GRPO 與 Skill-SD。注意：這是聯集，不是 3×7 全格：Qwen3-4B 跑了全部 7 個（工具類 3＋程式類 4），Qwen3-8B 只有 3 個工具類，Olmo3-7B 只有 4 個程式類（Table 1–2），論文未說明原因。

### 1.4 從作者的角度逐步拆解

#### 符號表（設計用彙總。頁面**不放總表**，只在每章末尾放一張可收合小表，列該章新定義的符號；「章」欄為小表所在章）

| 符號 | 意義 | 章 |
| --- | --- | --- |
| $\mathcal V$、logit、softmax | 詞表；模型對每個 token 的未正規化分數；$p(v)=e^{z_v}/\sum_u e^{z_u}$ | S1 |
| $x$、$o_{\le t}$、$a_{<t}$、$s_t=(x,o_{\le t},a_{<t})$ | 任務、到第 $t$ 輪為止的觀察、之前的動作、第 $t$ 輪的狀態 | S2 |
| $a_t=(a_{t,1},\dots,a_{t,L_t})$、$a_{t,<n}$ | 第 $t$ 輪動作的 token 序列與其長度；第 $n$ 個 token 之前的前綴（prefix） | S2 |
| $\tau$、$\lvert\tau\rvert$、$r(\tau)\in[0,1]$ | 軌跡、輪數、結果獎勵（EnvScaler 為完成子任務比例，TACO 為 0/1，App. B.2） | S2 |
| $\pi^S_\theta$、$\pi^T$ | 學生（參數 $\theta$）、老師 | S3 |
| $c_{fix}=\Phi_{fix}(x,E)$、$E$ | 固定規則產生的特權脈絡；經驗來源 | S3 |
| $D$、$D_{KL}$、$\mathrm{sg}[\cdot]$ | token 層級 divergence、KL divergence、stop-gradient（前向照算、反向不傳梯度） | S3 |
| $p^S_{t,n}$、$p^T_{t,n}$、$\mathcal V^M_{t,n}$、$\perp$、$\tilde p$、$\omega_{t,n}$ | 學生／老師在 prefix $(t,n)$ 的下一 token 分布；老師前 $M$ 名；尾部事件；合併後分布；動作 token 遮罩 $\in\{0,1\}$ | S4 |
| $\Phi_\phi$、$\phi=(\psi,\chi)$、$c_\phi$、$\langle e\rangle$、$J$（＝$n_{ret}$）、$K$、$d$ | composer、其 LoRA 與壓縮器參數、latent 脈絡、一個連續向量 token、檢索筆數、每筆向量數、hidden 維度 | S5 |
| $\bar\theta$、$\phi_0$ | 凍結的 base checkpoint（S5 在 Eq. 7 處先行定義）；冷啟動後的 composer（S6） | S5 |
| $u$、$q$ | 〔本頁推導用〕老師 logits 與 $q=\mathrm{softmax}(u)$（理想化模型） | S7 |
| $\delta_{t,n}$、$A(\tau)=2r-1$、$\Delta(\phi)$、$Q(a)$、$\bar Q$ | 每 token privilege、軌跡優勢、margin 統計量；〔本頁推導〕選了 token $a$ 之後整條軌跡 $A$ 的平均值，與其在學生分布下的平均 | S8 |
| $m$、$\beta$、$\eta_\beta$、$\lambda$ | margin、對偶變數、對偶步長、anchor 權重 | S9 |

#### 步驟 1：OPSD 的形式化（§3.1，Eq. 1–3）

$$a_t\sim\pi^S_\theta(\cdot\mid s_t),\qquad \tau=(s_1,a_1,\dots,s_{|\tau|},a_{|\tau|})\sim\pi^S_\theta(\cdot\mid x)\tag{Eq. 1}$$
$$c_{fix}=\Phi_{fix}(x,E),\qquad \pi^T_\theta(\cdot\mid s_t,c_{fix})=\pi_\theta(\cdot\mid[s_t;c_{fix}])\tag{Eq. 2}$$
$$\mathcal L_{OPSD}(\theta)=\mathbb E_{\tau\sim\pi^S_\theta}\Big[\sum_{t=1}^{|\tau|}\sum_{n=1}^{L_t}D\big(\mathrm{sg}[\pi^T_\theta(\cdot\mid s_t,c_{fix},a_{t,<n})]\,\big\|\,\pi^S_\theta(\cdot\mid s_t,a_{t,<n})\big)\Big]\tag{Eq. 3}$$

- 一般 OPSD 的老師與學生同權重 $\theta$，老師一側 stop-gradient。$E$ 可以是答案、推理軌跡、回饋、示範或檢索軌跡。
- 作者寫「supervision quality is bounded by a pre-defined context constructor」——這是主張，沒有證明。
- 基線的特權脈絡（§4.1、App. B.1）：OPSD＝oracle 軌跡放 system message、凍結老師、全詞表 forward KL；SDFT＝示範放 user message、EMA 老師、全詞表 forward KL；SDPO＝同組成功兄弟軌跡、EMA 老師、reverse KL＋top-K＋尾部；Skill-SD＝UCB1 選出的技能摘要、PPO-clip＋重要性加權取樣 token reverse KL；GRPO＝無老師，G=4、PPO-clip ε=0.2。OPSD/SDFT 的示範池在覆蓋不足時由 DeepSeek-V4-Pro、Qwen3.7-Max 補足。各基線保留原論文的 KL 方向與老師形式，所以 §4.1「只差在特權脈絡與蒸餾損失」並非只差一項。

#### 步驟 2：把脈絡建構器參數化（Eq. 4）

$c_\phi=\Phi_\phi(x,E)=\langle e_1\rangle\oplus\cdots\oplus\langle e_K\rangle$（Eq. 4）。保留 OPSD 的不對稱：老師看 $c_\phi$，學生只看 $s_t$。Eq. 4 只寫 $K$ 個 token 並稱 $K$ 為「每筆經驗的 token 數」，Eq. 7 則是 $J\cdot K$ 個（內部不一致，頁面在 S14 列出）。

#### 步驟 3：Composer（§3.2，Eq. 5–7；App. A.1、A.3）

- 經驗庫 $B=\{(x_i,\tau_i)\}$：只存**訓練集**上的成功 rollout（工具類超過 reward 門檻、程式類通過全部測試；門檻數值未給），格式為任務描述＋省略冗長觀察的 action–result trace（App. A.3）。
- 檢索（Eq. 15–16）：$z^{doc}_i=f_{ret}([x_i;\tau_i])/\|\cdot\|_2$，$z_q=f^{query}_{ret}(x)/\|\cdot\|_2$（查詢端只編碼任務，加 instruction prompt）；$\mathrm{Ret}(x)=\text{top-}n_{ret}\ z_q^\top z^{doc}_i$。$f_{ret}$＝Qwen3-Embedding-8B，$d_{ret}=4{,}096$，FAISS IndexFlatIP 精確內積。查詢向量預先快取。
- 編碼（Eq. 5）：$H_j=\mathrm{Enc}_\psi(x,m_j)\in\mathbb R^{T_j\times d}$。Encoder＝凍結 backbone＋LoRA（rank 8、α=16、dropout 0，每個 block 的 $W_q,W_k,W_v,W_o$、gate、up、down 七個投影）。輸入格式「Task to solve:\n x \n\nReference past trajectory:\n m_j」（App. A.5）。
- 壓縮（Eq. 6）：$E_j=\mathrm{Comp}_\chi(H_j;Q_\chi)\in\mathbb R^{K\times d}$，$Q_\chi\in\mathbb R^{K\times d}$ 為學習 query，初始化 $\mathcal N(0,1/\sqrt d)$（1/√d 是標準差或變異數未說明）；8 層 cross-attention **共用參數**，heads 與 hidden 維度沿用 backbone，FFN 倍率 4；每筆經驗獨立壓縮。
- 串接（Eq. 7）：$c_\phi(x,E)=\bigoplus_{j=1}^{J}(\langle e_{j,1}\rangle\oplus\cdots\oplus\langle e_{j,K}\rangle)$，$\pi^T_{\bar\theta,\phi}(\cdot\mid s,c_\phi)=\pi_{\bar\theta}(\cdot\mid[s;c_\phi])$。預設 $J=3$、$K=32$，共 96 個 latent token（§4.1）。同一個量在 §3.2/§4.1 叫 $J$，在附錄與 §4.3 叫 $n_{ret}$。

#### 步驟 4：注入方式（App. A.4，Eq. 17–18）與冷啟動（App. A.2，Eq. 14）

- 在第一個 user message 中插入 sentinel `<|LATENT PH|>`，前後是框架文字（工具類：「The following is a REFERENCE EXAMPLE of how a DIFFERENT, already-solved task was handled…」／「--- end of reference example --- Now complete YOUR task below…」；程式類：「The following is a reference from a similar programming problem…」／「--- end of reference ---」，App. A.5）。依 sentinel 切開分別 tokenize，中間插入 $J\cdot K$ 個 pad id：$\texttt{input\_ids}=[t_1..t_a,\ t_{pad}\times JK,\ t_{a+1}..t_L]$（Eq. 17）。
- embedding 後、第一層之前替換：$X[p_k]\leftarrow\langle e_k\rangle,\ k=1..JK$（Eq. 18），用 `torch.cat` 拼接切片而非 in-place 賦值，autograd 才追得到。SGLang 用 `positional_embed_overrides`，vLLM 用 `prompt_embeds`；backbone 不改。等價性檢查：以已知 token embedding 填入時，兩條路徑的 greedy 輸出相同（App. A.4）。
- Encoder 的 LoRA 只在編碼經驗時啟用，老師主 forward 與推論時關閉，所以老師＝純 base model＋latent（App. A.4）。
- 冷啟動：$\mathcal L_{init}(\phi)=-\mathbb E_{(x,y^\star)\sim D_{exp},E=\mathrm{Ret}(x)}[\log\pi_{\bar\theta}(y^\star\mid s,c_\phi(x,E))]$（Eq. 14），$y^\star$ 為 base model 自己的成功軌跡；backbone 凍結，梯度穿過凍結層回到 latent 位置（作者稱與 prefix-tuning 同機制）。AdamW、lr $10^{-5}$、batch 8、grad clip 3.0、$n_{ret}=3$；資料量與步數「依 domain 調整」，未給數字。得到 $\phi_0$。

#### 步驟 5：蒸餾損失（§3.3，Eq. 8–10）

$$p^S_{t,n}=\pi^S_\theta(\cdot\mid s_t,a_{t,<n}),\qquad p^T_{t,n}=\pi^T_{\bar\theta,\phi}(\cdot\mid s_t,c_\phi(x,E),a_{t,<n})\tag{Eq. 8}$$
$$\tilde p_{t,n}(v)=\begin{cases}p_{t,n}(v)&v\in\mathcal V^M_{t,n}\\ 1-\sum_{u\in\mathcal V^M_{t,n}}p_{t,n}(u)&v=\perp\end{cases}\quad(\text{老師、學生同式，}\mathcal V^M\text{ 為老師前 }M\text{ 名})\tag{Eq. 9}$$
$$\mathcal L_{distill}(\theta,\phi)=\mathbb E_{x\sim\mathcal D}\mathbb E_{\tau\sim\pi^S_\theta}\Big[\frac{\sum_{t,n}\omega_{t,n}D_{KL}(\tilde p^S_{t,n}\|\tilde p^T_{t,n})}{\sum_{t,n}\omega_{t,n}}\Big]\tag{Eq. 10}$$

- 老師 $\bar\theta$ 是「從同一 backbone 初始化的凍結參考副本」；學生一更新兩者權重就不同。老師的 forward activations 留在計算圖中，因此 $p^T$ 對 $c_\phi$、進而對 $\phi$ 可微；**Eq. 10 沒有對老師 stop-gradient**，$\mathcal L_{distill}$ 同時對 $\theta$ 與 $\phi$ 有梯度（與 Eq. 3 不同）。
- 學生在前（reverse KL）；作者的理由：「so that the student concentrates on teacher-supported behavior」。
- $M=20$ 是蒸餾截斷；取樣的 top-k 也是 20（Table 4），兩者不同。
- 〔本頁推導〕$D_{KL}(\tilde p^S\|\tilde p^T)=\sum_{v\in\mathcal V^M}p^S(v)\log\frac{p^S(v)}{p^T(v)}+\tilde p^S(\perp)\log\frac{\tilde p^S(\perp)}{\tilde p^T(\perp)}\le D_{KL}(p^S\|p^T)$（log-sum 不等式：合併只會丟失可區分的資訊；$M$ 越大分割越細，截斷 KL 不減）。學生在尾部內如何分配不受監督，只有尾部總量受監督。
- top-M 出處互相矛盾：§3.3「we follow (Zhao et al., 2026)」（即 OPSD）；§4.1「SDPO-style top-M logits plus a tail bucket」；App. B.1 寫 OPSD 用全詞表 forward KL、SDPO 才是 reverse KL＋top-K＋尾部。

#### 步驟 6：退化論述（§3.3）與〔本頁推導〕的梯度

- 原文：「an unconstrained composer can instead minimize Equation (10) by moving $p^T$ toward $p^S$, yielding uninformative latent context without improving the student」。對策 ❶ 冷啟動、❷ privileged margin。
- 〔本頁推導〕理想化：單一 prefix、完整詞表、學生 logits $z$、老師 logits $u$ 可自由移動、$p=\mathrm{softmax}(z)$、$q=\mathrm{softmax}(u)$、$\mathrm{KL}=\sum_v p_v\log(p_v/q_v)$。則
  $$\frac{\partial\,\mathrm{KL}}{\partial u_k}=q_k-p_k,\qquad \frac{\partial\,\mathrm{KL}}{\partial z_k}=p_k\big[(\log p_k-\log q_k)-\mathrm{KL}\big].$$
  在此理想化模型中，對 $u$ 做梯度下降使 $q$ 逐步移向 $p$，損失梯度本身不含脈絡是否有資訊的訊號；真實系統中梯度還要經過 $\partial u/\partial\phi$，學生也同時移向老師。實際上 $u$ 只能經由 96 個 latent 向量與凍結網路改變，損失也是 top-M 版本。`check.py` 有限差分最大誤差 6.6e-10、1.1e-9。

#### 步驟 7：Privileged margin（Eq. 11–12）

$$\delta_{t,n}(\phi)=\log\pi^T_{\bar\theta,\phi}(a_{t,n}\mid s_t,c_\phi,a_{t,<n})-\mathrm{sg}\big[\log\pi^S_\theta(a_{t,n}\mid s_t,a_{t,<n})\big]\tag{Eq. 11}$$
$$\Delta(\phi)=\mathbb E_\tau\Big[\frac{\sum_{t,n}\omega_{t,n}A(\tau)\,\delta_{t,n}(\phi)}{\sum_{t,n}\omega_{t,n}}\Big],\qquad A(\tau)=2r(\tau)-1\in[-1,1]\tag{Eq. 12}$$

- $a_{t,n}$ 是學生取樣到的 token；sg 擋掉往 $\theta$ 的梯度，δ 只訓練 $\phi$。兩個 log 機率在 forward 時已算好，δ 只需 gather。
- Eq. 12 是逐軌跡比值再取期望；Alg. 1 第 12 行寫「masked tokens 上的加權平均」，batch 內聚合方式不明。
- 〔本頁推導〕固定 prefix，**若動作恰從 δ 中的 $p^S$ 取樣**：$\mathbb E_{a\sim p^S}[\delta]=\sum_a p^S(a)\log\frac{p^T(a)}{p^S(a)}=-D_{KL}(p^S\|p^T)\le0$。這是期望值，單一批次的取樣平均可以為正。論文以溫度 0.7、top-p 0.95、top-k 20 取樣（Table 4），計算 $p^S$ 的溫度未說明，所以此等式在論文設定下不一定成立。$Q(a)$ 的白話：選了 token $a$ 之後，整條軌跡 $A$ 的平均值。令 $Q(a)=\mathbb E[A\mid\text{prefix},a]$、$\bar Q=\sum_a p^S(a)Q(a)$，該 prefix 對 Δ 分子的期望貢獻
  $$c=\sum_a p^S(a)Q(a)\log\frac{p^T(a)}{p^S(a)}=-\bar Q\,D_{KL}(p^S\|p^T)+\sum_a p^S(a)\big(Q(a)-\bar Q\big)\log\frac{p^T(a)}{p^S(a)}.$$
  第一項在 $\bar Q>0$（平均獎勵過半）時與 margin 作對，在 $\bar Q<0$ 時為正、反而獎勵 KL；因此當 $\bar Q\ge0$ 時，Δ 為正必須靠第二項（共變項），即「老師在導向較高獎勵的 token 上給更高機率」。參考量（測試集代理值，不是訓練 rollout 的 $\bar Q$；EnvScaler 的 r 是完成子任務比例，不是成功率）：Table 3 的平均獎勵換成 $2r-1$ 為 $2(0.486)-1=-0.028$、$2(0.637)-1=0.274$。
- 〔本頁推導〕連帶後果：學生蒸餾成功（$p^S\to p^T$）也會讓 δ→0、Δ→0<m，margin 要求老師持續領先學生。論文沒有討論，也沒有報告訓練中 Δ 或 β 的軌跡。

#### 步驟 8：完整目標與 Alg. 1（Eq. 13）

$$\min_{\theta,\phi}\max_{\beta\ge0}\ \mathcal L_{distill}(\theta,\phi)+\beta\big(m-\Delta(\phi)\big)+\lambda\big\|c_\phi-\mathrm{sg}[c_{\phi_0}]\big\|_2^2\tag{Eq. 13}$$

- β 初值 0；每次迭代 $\beta\leftarrow[\beta+\eta_\beta(m-\Delta)]_+$。$m=0.05$（§4.1），$\lambda=0.2$、$\eta_\beta=0.5$、$lr_\phi=10^{-5}$（Table 4）。anchor 不需額外 forward（$c_{\phi_0}$ 每任務快取，Alg. 1 第 1 行）。
- Alg. 1：第 5 行學生取樣 $\tau_i$、$r_i=V(\tau_i)$、$A_i=2r_i-1$；6–7 行 $E_i=\mathrm{Ret}(x_i)$、$c_{\phi,i}=\Phi_\phi(x_i,E_i)$；8–11 行逐 prefix 算 $p^S,p^T$、top-M＋尾部、累加 KL、$\delta$；12 行 Δ；13 行以 Eq. 13 更新 $(\theta,\phi)$；14 行更新 β。
- 作者的排除論述（原文一行）：「If the composer degenerates to uninformative context, $\pi^T\to\pi^S$, $\delta\to0$, $\Delta\to0<m$, and the dual penalty activates—structurally excluding the trivial solution.」沒有收斂或可行性分析。〔本頁推導〕m=0 時退化點的 Δ=0 滿足 Δ≥0，退化本身可行，所以這個排除論述需要 m>0；這也是 C3 在 m=0 時收斂到 q=p 的原因。
- 〔本頁推導〕Lagrangian 讀法：約束 $g(\phi)=m-\Delta\le0$；投影梯度上升使 Δ<m 時 β 增加、Δ>m 時 β 降到 0 為止（互補鬆弛）。β 項對 φ 的梯度 $-\beta\nabla_\phi\Delta=-\beta\,\mathbb E[A\,\nabla_\phi\log\pi^T(a_{t,n}\mid\cdot)]$（遮罩平均），是以學生樣本、沒有重要性權重的 REINFORCE 形式。數量級：Δ 停在 0 時 β 每步增加 $0.5\times0.05=0.025$；若每 32 rollouts（App. B.1）做一次對偶更新（本頁推論，論文未寫明），Fig. 4 的 1,600 rollouts 是 50 步，Δ 一直為 0 時 β=1.25。這是 Δ=0 的情形，不是上限：Δ<0 時每步增加 $0.5(m-\Delta)>0.025$。
- 〔本頁推導〕理想化不動點（學生固定、單一 prefix、老師 logits 自由、不計 anchor 與 top-M、β 固定）：$\partial(\mathrm{KL}-\beta\Delta)/\partial u_k=(q_k-p_k)-\beta(p_kQ_k-q_k\bar Q)$；令其為 0：
  $$q^*(a)=\frac{p(a)\,(1+\beta Q(a))}{1+\beta\bar Q},\qquad\text{需 }1+\beta Q(a)>0.$$
  β=0 時 $q^*=p$；Q 對所有 token 相同時 $q^*=p$；在 $q^*$ 處 $\Delta\ge0$（$Q\log(1+\beta Q)$ 為凸，Jensen），等號若且唯若 Q 在 p 下為常數。`check.py`：$q^*$ 處梯度最大 1.6e-16。

#### 步驟 9：推論

只部署學生 $\pi^S_\theta$；不需要經驗庫、檢索、composer 或 latent 脈絡（§3.3）。〔本頁觀察〕Table 3 的 Base＋Composer（＝推論時掛上 composer 的老師）已達 0.631，論文沒有討論「直接部署老師」這個選項。

#### 步驟 10：實驗數字（全部照錄；頁面只能用本節數字）

**Table 1（工具類，0–100）**：欄＝EnvScaler｜BFCL Base、Miss Func、Miss Param、Long Ctx、Avg｜ACE M-Step、M-Turn、Avg。

| Qwen3-4B | Env | Base | MissF | MissP | Long | BFCL Avg | M-Step | M-Turn | ACE Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 48.6 | 28.50 | 20.50 | 20.50 | 22.00 | 22.88 | 48.3 | 52.2 | 50.6 |
| GRPO | 61.8 | 33.00 | 20.50 | 24.50 | 23.00 | 25.25 | 58.3 | 54.4 | 56.0 |
| Skill-SD | 59.1 | 33.00 | 19.00 | 24.00 | 22.50 | 24.63 | 56.6 | 55.6 | 56.0 |
| OPSD | 51.2 | 31.50 | 19.50 | 20.50 | 29.00 | 25.13 | 41.6 | 53.3 | 48.6 |
| SDPO | 50.6 | 21.50 | 14.00 | 13.50 | 14.00 | 15.75 | 45.0 | 33.3 | 38.0 |
| SDFT | 50.3 | 31.50 | 17.00 | 17.00 | 21.50 | 21.75 | 53.3 | 47.8 | 50.0 |
| LOPD | 63.7 | 32.50 | 26.50 | 24.50 | 26.00 | 27.38 | 56.6 | 63.3 | 60.6 |

| Qwen3-8B | Env | Base | MissF | MissP | Long | BFCL Avg | M-Step | M-Turn | ACE Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 49.2 | 34.00 | 28.50 | 25.50 | 25.50 | 28.38 | 61.6 | 50.0 | 54.6 |
| GRPO | 57.3 | 36.00 | 28.00 | 27.00 | 25.00 | 29.00 | 65.0 | 53.3 | 58.0 |
| Skill-SD | 60.2 | 33.00 | 27.00 | 25.00 | 24.50 | 27.38 | 63.3 | 51.1 | 56.0 |
| OPSD | 52.0 | 30.50 | 25.00 | 23.50 | 24.00 | 25.75 | 60.0 | 47.8 | 52.7 |
| SDPO | 55.3 | 29.00 | 24.50 | 23.00 | 23.50 | 25.00 | 56.7 | 48.9 | 52.0 |
| SDFT | 56.2 | 31.50 | 26.00 | 24.50 | 25.50 | 26.88 | 65.0 | 47.8 | 54.7 |
| LOPD | 66.4 | 37.00 | 29.00 | 27.50 | 26.00 | 29.88 | 73.3 | 55.6 | 62.7 |

**Table 2（程式類 pass@1 %）**：欄＝LCB v5、v6、Avg｜HumanEval+、MBPP+、EvalPlus Avg。

| Qwen3-4B | v5 | v6 | LCB Avg | HE+ | MBPP+ | EP Avg | Olmo3-7B | v5 | v6 | LCB Avg | HE+ | MBPP+ | EP Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 47.67 | 41.22 | 45.61 | 85.37 | 75.93 | 78.79 | Vanilla | 47.31 | 44.27 | 46.34 | 86.59 | 70.37 | 75.28 |
| GRPO | 50.18 | 44.27 | 48.29 | 86.59 | 76.19 | 79.34 | GRPO | 49.46 | 45.80 | 48.29 | 89.02 | 72.75 | 77.67 |
| Skill-SD | 49.82 | 41.98 | 47.32 | 85.98 | 76.98 | 79.70 | Skill-SD | 49.10 | 45.03 | 47.80 | 87.20 | 72.22 | 76.75 |
| OPSD | 42.65 | 35.11 | 40.24 | 85.36 | 76.72 | 79.33 | OPSD | 44.09 | 45.04 | 44.39 | 87.80 | 71.96 | 76.75 |
| SDPO | 40.86 | 34.35 | 38.78 | 84.76 | 74.87 | 77.86 | SDPO | 49.82 | 43.51 | 47.80 | 88.41 | 72.75 | 77.49 |
| SDFT | 48.75 | 43.51 | 47.07 | 87.20 | 76.98 | 80.07 | SDFT | 47.67 | 44.27 | 46.58 | 89.02 | 73.02 | 77.86 |
| LOPD | 50.54 | 45.04 | 48.78 | 87.80 | 78.57 | 81.36 | LOPD | 53.41 | 45.80 | 50.98 | 90.24 | 73.28 | 78.41 |

- 〔本頁推導〕彙總欄算法（`check.py` 全部吻合，誤差 ≤0.006／ACE ≤0.06）：BFCL Avg＝四子集簡單平均；ACE Avg＝(20·M-Step＋30·M-Turn)/50；LCB Avg＝(279·v5＋131·v6)/410；EvalPlus Avg＝(164·HE＋378·MBPP)/542。論文未明說。
- 子欄反例（只有彙總欄全勝）：Qwen3-4B BFCL Base 32.50 < GRPO/Skill-SD 33.00；Long Ctx 26.00 < OPSD 29.00；Miss Param 24.50 與 GRPO 並列；ACE M-Step 56.6 < GRPO 58.3；Olmo3-7B LCB v6 45.80 與 GRPO 並列。
- 「Qwen3-8B 優勢更大」（§4.2）〔本頁推導差值〕：EnvScaler 1.9→6.2；ACE 4.6→4.7；BFCL 2.13→0.88（變小）。
- §4.2 其他數字：對 vanilla 的增益 1.50（BFCL，8B）到 17.2（EnvScaler，8B）；8B 上對 OPSD/SDFT/Skill-SD：EnvScaler 14.4/10.2/6.2、ACE 10.0/8.0/6.7；Olmo3-7B 對最強替代 +2.69（LCB）、+0.55（EvalPlus）。§4.2 也舉 SDPO、OPSD 低於 vanilla 的例子（4B SDPO：BFCL 22.88→15.75、ACE 50.6→38.0、LCB 45.61→38.78）。
- **Fig. 3**（EnvScaler，Qwen3-4B，學生不帶脈絡評測，App. B.4）：Frozen φ₀ 0.573；m=0 0.551；m=0.01 0.566；m=0.02 0.603；m=0.05 0.637；m=0.10 0.626；m=0.20 0.613。「Frozen φ₀」＝以凍結 composer 訓練（無聯合最佳化）。「Frozen φ₀」0.573 是以凍結 composer 訓練出的學生，不是 composer 或老師的分數。照 Eq. 13 字面，m=0 仍是約束 Δ≥0 且 β 會更新，但退化點（Δ=0）滿足這個約束；論文稱之為「without the margin constraint」，實作是關掉 β 還是 m=0 未說明。
- **Fig. 4**（Qwen3-4B EnvScaler，§4.3 文字）：LOPD 320 generations 後超過 0.61、576 時 0.637、之後 0.63–0.64 到 1,600；GRPO、Skill-SD 1,600 時 0.611、0.588。圖說「Mean reward over 1,600 rollouts」；App. B.4：「Mean reward is periodically evaluated on the EnvScaler test set」，即 Fig. 4 是訓練中定期在測試集上的評測曲線，與 Table 1 的差異可能來自 checkpoint 或評測流程，無法排除。**曲線無數值表，頁面不重畫。**
- **Fig. 5**：(a) K=8/16/32/64/128 → 56.5/56.3/63.7/60.6/62.9（%）；(b) $n_{ret}$=1 → 0.605、3 → 0.637，其餘點文字未給；(c–e) $n_{ret}=3$ 時 ACE M-Step 56.6、M-Turn 63.3、Avg 60.6。**(b–e) 其餘點不得補值。**
- **Table 3**（EnvScaler，Qwen3-4B）：Vanilla／Base＋Composer／LOPD：Reward 0.486／0.631／0.637；Interaction steps 11.12／16.31／17.04；Tool calls/step 3.50／1.21／1.11；First-step length 9,937／6,695／6,210；Reward/tool call 0.038／0.053／0.050；Repeated tool calls 8.89／4.49／5.25。Base＋Composer＝未訓練 backbone＋聯合訓練後的 composer（m=0.05）＋latent 脈絡（App. B.4）。§4.3：first-step 短 37.5%（〔本頁推導〕1−6210/9937=0.375）。〔本頁推導〕Reward/tool call 無法由同表重現：0.486/(11.12×3.50)=0.0125、0.637/(17.04×1.11)=0.0337，定義未給。
- **Fig. 6**：把凍結 LM head 套到 32 個 latent token，10 個工具例＋10 個程式例；投影是多語言與程式碎片，task conditioning 改變表面投影但讀不出程序，也不抄寫檢索解；作者自己寫「direct decodability alone does not establish which information the teacher functionally uses」。展示的檢索都是 rank 2（相似度 0.886、0.691）。〔本頁疑問〕兩例的 task id「env_174_rl-task_29」「taco_00713」看起來像訓練語料的編號；rank 1 是否就是任務自己的成功軌跡（自我檢索，見步驟 11）論文未說明。

#### 步驟 11：實作設定（Table 4、Table 5、App. B）

| 項目 | 工具類 | 程式類 |
| --- | --- | --- |
| 訓練資料 | EnvScaler 衍生 2,349 tasks | TACO（DeepCoder）約 7,000 |
| 生成預算（tokens） | 40,960 | 16,384 |
| 最大環境步數 | 30 | 單輪 |
| 解碼 | temperature 0.7、top-p 0.95、top-k 20（訓練與評測相同） | 同左 |
| LOPD | $lr_\phi=10^{-5}$、λ=0.2、η_β=0.5、m=0.05、J=3、K=32、M=20 | 同左 |
| 全部方法共用 | 每步 32 rollouts、AdamW lr $10^{-5}$、grad clip 1.0 | 同左 |
| 評測 | EnvScaler 200 題（與訓練不重疊）、BFCL-v3 4×200（FC mode）、ACEBench 50（M-Step 20、M-Turn 30，使用者模擬器 deepseek-v4-flash，max steps 20） | LCB v5 279、v6 131；HumanEval+ 164、MBPP+ 378；全部開 thinking mode |

未說明：學生是全參數或 LoRA、學生 lr（B.1 只說所有方法共用 $10^{-5}$）、總訓練步數（程式類完全沒給）、LOPD 每任務 rollout 數、seed 數、經驗庫大小、reward 門檻、冷啟動資料量；**訓練時 Ret(x) 是否排除任務 x 自己的成功軌跡**（庫由訓練集建成、查詢也來自訓練集；未排除則 latent 可能攜帶同題解答——此為本頁疑問）；ω 是否涵蓋 thinking token；算 $p^S,p^T$ 的溫度；anchor 範數是否平均。沒有對 λ、A 加權、冷啟動做消融。

#### 步驟 12：統計強度〔本頁推導，由表中百分比反推題數〕

- 程式類差距多為 1–3 題：Qwen3-4B LCB v5 141 vs 140／279、v6 59 vs 58／131、HumanEval+ 144 vs 143／164；Olmo3-7B EvalPlus 425 vs 422／542、MBPP+ 277 vs 276／378（完整表見 C4）。BFCL 每子集 200 題，0.5 分＝1 題。
- 論文沒有 seed 數、標準差或信賴區間；評測溫度 0.7 有取樣雜訊。
- ACEBench 只有 50 題，表中值落在 1/60 與 1/90 的格點上（例 48.3≈29/60、63.3≈57/90），暗示重複 3 次或部分給分，論文未說明。
- Fig. 5(a) K=64 比 K=32 低 3.1、比 K=128 低 2.3，非單調；可作單次實驗雜訊的粗略參考，許多主表差距小於它。
- K 與 $n_{ret}$ 依 EnvScaler **測試集**上的敏感度挑選（§4.3；App. A.1 寫 K 依 §4.3 決定）；m 的選法論文未說明，Fig. 3 的 m 掃描也在測試集上。沒有提到 validation split。
- 捨入不一致（0.01 級，不影響排名）：140/164 寫成 85.37 與 85.36；59/131 寫成 45.04 與 45.03；EvalPlus 78.79 由題數算為 78.78。

---

## 2. 學習目標（驗收測試）

每章結尾 `<details>`（summary「檢查理解」）重述題目與參考答案。**參考答案必須以一段自足的內文段落出現在指定章節的本文**（6.6 列出位置），不能只在 details、表格、圖或元件中。

| # | 學習目標 | 檢查題 | 參考答案 | 章 |
| --- | --- | --- | --- | --- |
| LO1 | 說明 OPSD 的結構 | OPSD 的老師與學生各看到什麼？監督在哪些狀態上計算？推論時用誰？ | 學生 $\pi^S_\theta$ 只看 $s_t=(x,o_{\le t},a_{<t})$ 並自己取樣軌跡；老師是同一個模型多看特權脈絡 $c_{fix}=\Phi_{fix}(x,E)$（答案、軌跡、回饋或技能，由設計者的固定規則產生），在學生走過的每個 prefix $(t,n)$ 給出下一 token 分布；損失是兩者的 token 層級 divergence，老師一側 stop-gradient（Eq. 2–3）。推論時只用學生，不給脈絡 | S3 |
| LO2 | 說出 LOPD 的老師是什麼、誰在訓練 | LOPD 的老師與學生是同一組權重嗎？哪些參數會更新？梯度怎麼到 composer？ | 不是。老師是凍結的 base checkpoint $\bar\theta$（與學生同架構、同初始化，學生開始更新後權重就不同）加上 latent 脈絡：$\pi_{\bar\theta}(\cdot\mid[s;c_\phi])$（Eq. 7、§3.3）。更新的是學生 $\theta$ 與 composer $\phi=(\psi,\chi)$（encoder LoRA 與 QFormer）；$\bar\theta$ 不更新，但老師的 activations 留在計算圖中，Eq. 10 沒有對老師 stop-gradient，梯度穿過凍結網路回到 $c_\phi$；encoder LoRA 在老師主 forward 時關閉（App. A.4） | S6 |
| LO3 | 描述 composer 的流程與數字 | 從任務 $x$ 到老師輸入中的 latent token，經過哪些步驟？數量多少？ | 經驗庫只存訓練集的成功 rollout；Qwen3-Embedding-8B（4,096 維）編碼查詢（只含任務）與文件（任務＋軌跡），FAISS 取 cosine 最高的 $J=3$ 筆（Eq. 15–16）；每筆由凍結 backbone＋LoRA（rank 8）編碼（Eq. 5），再由 8 層共用參數的 QFormer 式 cross-attention 以 $K$ 個學習 query 壓成 $K=32$ 個向量（Eq. 6）；串接成 96 個 latent token，在輸入中以 96 個 pad 佔位並於 embedding 後替換（Eq. 17–18，用 torch.cat）。composer 先以 Eq. 14 的 NLL 冷啟動得到 $\phi_0$ | S5 |
| LO4 | 寫出 top-M＋尾部的 reverse KL | 蒸餾損失比較哪些機率？支撐集由誰決定？合併尾部有什麼後果？ | 每個 prefix 取老師機率最高的 $M=20$ 個 token 為 $\mathcal V^M$，其餘合併成 ⊥，學生用同一支撐集（Eq. 9）；損失 $D_{KL}(\tilde p^S\|\tilde p^T)=\sum_{v\in\mathcal V^M}p^S\log(p^S/p^T)+\tilde p^S(\perp)\log(\tilde p^S(\perp)/\tilde p^T(\perp))$，學生在前（reverse KL），以 ω 對動作 token 平均（Eq. 10）。〔本頁推導〕截斷 KL 不大於完整 KL（log-sum 不等式），學生在尾部內的分配不受監督，只有尾部總量受監督 | S4 |
| LO5 | 解釋退化（collapse）的來源 | 為什麼只最小化 $\mathcal L_{distill}$ 會讓脈絡失去資訊？ | 老師一側沒有 stop-gradient，composer 可以改變老師分布來降低損失（§3.3）。〔本頁推導〕理想化下 $\partial D_{KL}(p\|q)/\partial u_k=q_k-p_k$，梯度下降使老師分布 $q$ 移向學生 $p$；在此理想化模型中，梯度本身不含脈絡是否有資訊的訊號；最小值 $q=p$ 對應沒有資訊的脈絡，學生得不到新訊號 | S7 |
| LO6 | 定義 δ 與 Δ 並說明 Δ 為正的條件 | δ、A、Δ 怎麼定義？為什麼不加權的 δ 期望上不為正？這需要什麼條件？ | $\delta=\log\pi^T(a_{t,n}\mid\cdot,c_\phi)-\mathrm{sg}[\log\pi^S(a_{t,n}\mid\cdot)]$，$a_{t,n}$ 是學生取樣的 token（Eq. 11）；$A=2r-1$；$\Delta=\mathbb E_\tau[\sum\omega A\delta/\sum\omega]$（Eq. 12）。〔本頁推導〕當動作恰從 δ 中的 $p^S$ 取樣時，$\mathbb E_{a\sim p^S}[\delta]=-D_{KL}(p^S\|p^T)\le0$：期望上不為正，單一批次的取樣平均可以為正；論文以溫度 0.7、top-p、top-k 取樣，條件未必成立。令 $Q(a)$＝選了 token $a$ 之後整條軌跡 A 的平均值、$\bar Q$ 為其學生平均，一個 prefix 的貢獻 $=-\bar Q\cdot\mathrm{KL}+\sum p^S(Q-\bar Q)\delta$；當 $\bar Q\ge0$ 時 Δ 為正必須靠共變項，也就是老師在導向較高獎勵的 token 上給出比學生更高的機率；$\bar Q<0$ 時第一項本身為正 | S8 |
| LO7 | 說明對偶更新 | Eq. 13 是什麼？β 怎麼更新？預設值？ | $\min_{\theta,\phi}\max_{\beta\ge0}\mathcal L_{distill}+\beta(m-\Delta)+\lambda\|c_\phi-\mathrm{sg}[c_{\phi_0}]\|^2$；β 初值 0，$\beta\leftarrow[\beta+\eta_\beta(m-\Delta)]_+$；$m=0.05$、$\eta_\beta=0.5$、$\lambda=0.2$（§4.1、Table 4）。Δ<m 時 β 增加，Δ>m 時 β 降到 0 為止。〔本頁推導〕β 項對 φ 的梯度是以學生樣本、無重要性權重的 REINFORCE 形式 $-\beta\mathbb E[A\nabla_\phi\log\pi^T(a_{t,n})]$；Δ 停在 0 時 β 每步只增加 0.025 | S9 |
| LO8 | 讀懂理想化的不動點 | 〔本頁推導〕的理想化模型中，β 固定時老師收斂到哪裡？β=0 與 Q 為常數時呢？ | $q^*(a)=p(a)(1+\beta Q(a))/(1+\beta\bar Q)$（需 $1+\beta Q>0$），由 $(q_k-p_k)-\beta(p_kQ_k-q_k\bar Q)=0$ 得出；β=0 時 $q^*=p$（退化）；Q 對所有 token 相同時不論 β 多大 $q^*=p$。假設：學生固定、單一 prefix、老師 logits 自由、不計 anchor 與 top-M；這是本頁的解釋模型，不是論文結果 | S10 |
| LO9 | 讀出消融並區分量測與解釋 | Fig. 3 的七個數字？哪些是量測、哪些是作者解釋、哪些沒量？ | Frozen φ₀ 0.573；m=0 0.551；0.01 0.566；0.02 0.603；0.05 0.637；0.10 0.626；0.20 0.613（EnvScaler、學生不帶脈絡）。量測：m≤0.01 的學生低於以凍結 composer 訓練的學生，m≥0.02 高於它。〔本頁推導〕m=0 時退化點 Δ=0 仍滿足 Δ≥0，所以作者的排除論述需要 m>0。「無約束的蒸餾梯度讓 latent 脈絡退化」是作者認為。沒量：老師本身（例如 base＋φ₀）的 reward、訓練中 Δ 與 β 的軌跡；每點是單次訓練 | S11 |
| LO10 | 評估主結果的統計強度 | 「10 組都最好」指什麼？程式類差距換成題數是多少？還缺什麼統計資訊？ | 指 10 組 backbone × benchmark 的彙總欄（Table 1–2）；子欄有反例（Qwen3-4B BFCL Base 32.50<33.00、Long Ctx 26.00<29.00、ACE M-Step 56.6<58.3）。〔本頁推導〕Qwen3-4B 程式類對最強基線：LCB v5 141 vs 140／279、v6 59 vs 58／131、HumanEval+ 144 vs 143／164。缺：seed 數、標準差或信賴區間；評測溫度 0.7；K、$n_{ret}$ 依 EnvScaler 測試集選定（m 的選法未說明，其掃描也在測試集上） | S12 |
| LO11 | 界定「<30% rollout 預算」的範圍 | Fig. 4 實際給了哪些數字？30% 的說法對誰成立？與 Table 1 一致嗎？ | 文字只給：LOPD 320 generations 後超過 0.61、576 時 0.637、之後 0.63–0.64；GRPO、Skill-SD 在 1,600 時 0.611、0.588（§4.3）。〔本頁推導〕320/1,600=20%、576/1,600=36%；對 Skill-SD 成立，對 GRPO 只知道 LOPD 超過 0.61、不知是否超過 0.611；〔本頁推論〕論文未說預算是否計入建經驗庫與冷啟動所需的額外 base-model rollouts；Fig. 4 是訓練中定期在測試集上的評測（App. B.4）。Table 1 的 GRPO 61.8、Skill-SD 59.1 與 0.611、0.588 不一致，論文未解釋；曲線無數值，本頁不重畫 | S13 |
| LO12 | 指出缺乏證據的主張與內部矛盾 | 「latent 優於文字」「retrieval alone is insufficient」有實驗支持嗎？top-M 設計引自誰？ | 沒有。論文沒有「把檢索到的軌跡以文字放進老師脈絡」的 baseline，Frozen φ₀（0.573）是凍結的 latent composer，不是純檢索，所以兩個說法都沒有直接證據；Table 3 推論時的老師（Base＋Composer）0.631 與學生 0.637 相近。top-M 出處矛盾：§3.3 寫 follow Zhao et al.（OPSD），§4.1 寫 SDPO-style，App. B.1 說 OPSD 用全詞表 forward KL、SDPO 才用 top-K＋尾部 | S14 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），圖版加寬至 1080px；章節編號 JetBrains Mono（`01 · …`）；標題用陳述句。全頁視覺記號：**老師＝實心、學生＝斜線圖樣**（`--c-t`、`--c-s`，不只靠顏色），在 hero、F1、C1、C2、C3 一致。

| # | id | 標題 | 內容（依序；每個符號先定義再使用） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈LOPD：讓自蒸餾老師的特權脈絡可以訓練〉 | 副標「這篇論文從過去的成功解題紀錄找出相似的幾筆，轉成一串數字放進老師的輸入，並用最後的成績限制老師不要變得和學生一樣。」三張事實卡＋hero 圖（見下）；連結 arXiv、`../index.html#paper-119`、`lopd-lineage-hub.html`。其後 **「閱讀須知」callout**（見下） | 1 螢幕＋callout |
| S1 | `basics` | 01 · 語言模型在每個位置輸出下一個 token 的分布 | 詞表 → logit → softmax → log 機率 → 取樣一句（「依這個分布抽出下一個 token；溫度、top-k、top-p 是改變抽取方式的設定，本論文的值見 15」）。末尾本章符號小表 | 200–250 字 |
| S2 | `agent` | 02 · 多輪代理人的一條軌跡只在結尾得到一個獎勵 | 任務 $x$、觀察 $o$、動作 $a$、狀態 $s_t$、token 層級 $a_{t,n}$、前綴 $a_{t,<n}$、軌跡 $\tau$、結果獎勵 $r$ → RLVR（只用 $r$ 訓練，一條軌跡一個數）→ GRPO 一句（同題取 G 條、以組內相對獎勵加權的 RLVR 基線）。延伸：`rlvr-grpo-credit-assignment.html` | 250–300 字 |
| S3 | `opsd` | 03 · OPSD 讓同一個模型帶著特權脈絡當老師 | 知識蒸餾 → on-policy 與 off-policy → KL 定義與兩個方向（字面敘述：reverse KL $D_{KL}(p^S\|p^T)$ 對「學生機率高、老師機率低」的 token 懲罰較重；forward KL $D_{KL}(p^T\|p^S)$ 對「老師機率高、學生機率低」的 token 懲罰較重）→ stop-gradient → Eq. 1–3 → 基線脈絡表（步驟 1）→ 作者主張（標明是主張）→ **F1** → LO1 段落。延伸：`on-policy-distillation-imitation.html`、`lopd-opsd-self-teacher-privileged-context.html`、`lopd-privileged-context-substrates.html` | 450–550 字＋F1 |
| S4 | `distill` | 04 · 蒸餾損失只比較老師的前 M 名與合併後的尾部 | 先寫抽象的 $p^T_{t,n}=\pi^T(\cdot\mid s_t,c,a_{t,<n})$（$c$ 的構造見 05–06）→ Eq. 9、Eq. 10、ω → M=20 與取樣 top-k=20 不同 → 〔本頁推導〕「把尾部多個 token 合併成一項，只會丟失可區分的資訊，KL 不會因此變大」（log-sum 不等式的字面說法），學生在尾部內如何分配不受監督 → **C1** → top-M 出處矛盾一句（→14）→ LO4 段落 | 400–500 字＋C1 |
| S5 | `composer` | 05 · Composer 從檢索到的經驗產生 96 個連續向量 | Eq. 4 → 經驗庫與檢索（Eq. 15–16；cosine 相似度一句定義）→ LoRA 一句（凍結原權重、只訓練一個低秩增量）→ cross-attention 一句（每個 query 向量對一組 key/value 向量依相似度取加權平均）→ QFormer 一句（固定數量的學習 query 對變長序列做 cross-attention，輸出長度固定）→ Eq. 5–7（Eq. 7 出現處先以一句定義 $\bar\theta$：「訓練開始前的 base 模型權重，訓練中不更新」）→ LO3 段落 | 400–450 字 |
| S6 | `inject` | 06 · 向量替換輸入中的佔位 token，老師的權重保持凍結 | **F2**（Eq. 17–18，torch.cat）→ encoder LoRA 在老師主 forward 關閉 → 冷啟動 Eq. 14 → 凍結 θ̄、Eq. 10 無 sg、梯度穿過凍結網路 → LO2 段落。延伸：`lopd-latent-memory-for-agents.html`、`lopd-latent-reasoning-continuous-tokens.html` | 300–400 字＋F2 |
| S7 | `collapse` | 07 · 只最小化蒸餾損失時，老師分布可以移向學生來降低損失 | 原文論述 → 理想化設定（定義 $u,q$）→ 兩個梯度式〔本頁推導〕→「在此理想化模型中，損失梯度本身不含脈絡是否有資訊的訊號；真實系統中梯度還要經過 $\partial u/\partial\phi$，學生也同時移向老師」→ **C3-lite**（C3 的精簡版，見 4.C3）→ LO5 段落。延伸：`lopd-opd-failure-modes-and-stabilization.html` | 300–350 字＋C3-lite |
| S8 | `margin` | 08 · Privileged margin 以結果獎勵加權老師在取樣 token 上的優勢 | Eq. 11 → A=2r−1 → Eq. 12（註 Alg. 1 第 12 行聚合不明）→ $Q(a)$ 白話（「選了 token a 之後，整條軌跡 A 的平均值」）→ 逐步推導 $\mathbb E_{a\sim p^S}[\delta]=\sum_a p^S(a)(\log p^T(a)-\log p^S(a))=-\sum_a p^S(a)\log\frac{p^S(a)}{p^T(a)}=-D_{KL}$ 與成立條件（取樣分布＝δ 中的 $p^S$；這是期望，取樣平均可為正）→ 分解式 → **C2** → LO6 段落 | 450–550 字＋C2 |
| S9 | `dual` | 09 · 對偶變數 β 在 Δ 低於 m 時加大懲罰 | 受限最佳化一段（約束 $g=m-\Delta\le0$、Lagrangian、β≥0、投影 $[\cdot]_+$、互補鬆弛，各一句）→ Eq. 13、anchor → Alg. 1（兩欄：左行號＋原文、右中文；< 768px 每行下接說明）→ 排除論述原文＋〔本頁推導〕「m=0 時退化點 Δ=0 本身可行，排除論述需要 m>0」→ REINFORCE 一句（「本頁只用到它的梯度形式：以樣本的 log 機率梯度乘上獎勵」）→ β 數量級（Δ=0 情形，不是上限）→ LO7 段落 | 450–550 字 |
| S10 | `fixedpoint` | 10 · 理想化模型中，老師收斂到依期望獎勵重新加權的學生分布 | 假設清單 → 梯度 → $q^*$ → 三個特例（β=0、Q 為常數、$1+\beta Q\le0$）→ **C3** → LO8 段落 → 末段〔本頁推導〕「學生追上老師也使 Δ→0，margin 要求老師持續領先；論文未報告 Δ、β 軌跡」 | 350–450 字＋C3 |
| S11 | `ablation` | 11 · 消融：margin 太小時，學生低於以凍結 composer 訓練的學生 | B.4 設定 → **F3** → 量測句（只敘述數字）→「作者認為」段 → 沒量的東西（老師本身、Δ/β 軌跡、單次）→ m=0 字義不明＋退化點可行 → LO9 段落 | 350–450 字＋F3 |
| S12 | `results` | 12 · 主要結果：10 組彙總欄都最高，多數差距只有幾題 | benchmark 一句定義 → Table 1、2 彙總欄（完整表放 `<details>`）→ 彙總算法 → 子欄反例 → 8B 說法核對 → 覆蓋（聯集，不是 3×7 全格）→ 統計段（步驟 12）→ **SE 與 ±1.96 的定義段**（「N 題、答對率 $\hat p$ 時，單一分數因題目取樣造成的標準誤 $SE=\sqrt{\hat p(1-\hat p)/N}$；常態近似下約 95% 的取樣結果落在 $\hat p\pm1.96\,SE$。兩個方法在同一批題目上評測時，比較差異要用逐題配對的檢定，論文沒有逐題結果，所以本頁只顯示單一分數的 SE，不據此判斷顯著與否」）→ **C4** → LO10 段落 | 550–650 字＋2 表＋C4 |
| S13 | `efficiency` | 13 · 樣本效率、行為變化與 latent 解碼 | Fig. 4 文字點表（表題下註「論文只以文字給出；本頁不重畫」；Fig. 4 是訓練中定期在測試集上的評測，App. B.4）→ 30% 範圍；〔本頁推論〕論文未說是否計入建經驗庫與冷啟動的額外 rollouts → Table 1 vs Fig. 4 → Fig. 5 數字表 → Table 3 與三點保留（prompt 本來要求避免平行呼叫、Base＋Composer 0.631≈學生 0.637、Reward/tool call 不可重現）→ Fig. 6（含 task id／自我檢索〔本頁疑問〕）→ LO11 段落 | 450–550 字＋3 表 |
| S14 | `claims` | 14 · 論文主張與證據的對照 | 對照表（主張｜論文位置｜證據｜判讀），至少列：latent 優於文字、retrieval alone insufficient、老師＝同一模型、聯合訓練讓老師變強、<30% 預算、每欄最好、8B 優勢更大、**「三個 backbone、七個 benchmark」＝聯集，不是 3×7 全格（Qwen3-4B 全部 7 個、Qwen3-8B 只有 3 個工具類、Olmo3-7B 只有 4 個程式類）**、latent 可讀、margin 保證有資訊、top-M 出處、Eq. 4 vs 7、J vs $n_{ret}$、m=0 字義、Eq. 12 vs Alg. 1、Table 3 Reward/tool call、行為內化 → LO12 段落 | 表＋200 字 |
| S15 | `impl` | 15 · 實作設定與常見錯誤 | 步驟 11 表 → 常見錯誤 7 條（下）→ 未說明清單（含自我檢索疑問） | 350–450 字＋表 |
| S16 | `history` | 16 · 研究脈絡 | 時間線（6.7）＋一段：DAgger／蒸餾（2010–2015）→ context distillation（2022）→ LLM on-policy 蒸餾（2023）→ GRPO（2024）→ latent 推理與記憶（2024–2025）→ OPSD 家族（2026）→ 本文 | 時間線＋120 字 |
| S17 | `next` | 17 · 限制與延伸閱讀 | 論文自述限制很少（§5 只談展望），本頁整理：無定理、無 seed、無文字檢索 baseline、K／$n_{ret}$ 依測試集選定、老師 backbone 固定不進步；筆記清單（6.7）；arXiv | 200 字＋清單 |
| — | — | 頁尾 | 引用格式、arXiv；`#pm-bar` | — |

「長度」欄不含 LO 參考答案段落（每段約 120–200 字，另計）。

**Hero 事實卡**（每張只用卡內自己定義的詞；不出現 KL、latent、凍結等未定義詞）：
1. 「自蒸餾（self-distillation）：同一個語言模型扮演兩個角色。學生只看任務，自己寫出回答；老師多看一段額外資訊，在學生寫出的每個位置給出『下一個詞是什麼』的機率，學生學著讓自己的機率接近老師。」（詳見 03）
2. 「LOPD 讓這段額外資訊可以訓練：從過去成功的解題紀錄中找出 3 筆最相似的，轉成 96 組數字（向量，不是文字）放進老師的輸入。老師的權重在訓練中不更新，只有產生這些數字的模組與學生在學。」（詳見 05、06）
3. 「論文報告：在工具使用與程式生成共 10 組比較中，每組評測題目的彙總分數都是 LOPD 最高；但程式題的差距多半只有 1–3 題，且論文沒有多次重複訓練的差異範圍。」（詳見 12）

**Hero 圖**（靜態 SVG，`figHero(w)`）：上下兩條橫列。上列「學生」：方塊「任務＋互動歷史」→ 斜線長條組「下一個詞的機率」。下列「老師（權重不更新）」：同一方塊＋右側 3 段小格（標「3 筆紀錄 × 32 組數字」）→ 實心長條組。兩組長條之間一條雙向線，標「兩組機率的差異（詳見 04）」。斷點版面見 4.F。

**「閱讀須知」callout**（S0 結尾，`.callout`，每條連到所屬章節）：
1. 本論文沒有定理或證明；頁面中的梯度式、E[δ]=−KL、$q^*\propto p(1+\beta Q)$ 都是〔本頁推導〕。（→07–10）
2. 老師是凍結的 base checkpoint 加 latent 脈絡，不是與學生同權重的模型。（→06）
3. 「latent 優於文字」與「retrieval alone is insufficient」缺少證據：論文沒有以文字放入檢索軌跡的 baseline。（→14）
4. Table 1 與 Fig. 4 的 GRPO／Skill-SD 數字不一致；top-M 設計的引用來源前後矛盾。（→13、04）
5. 「少於 30% rollout 預算」只對 Skill-SD 明確成立；論文未說明預算是否計入建經驗庫與冷啟動的額外 rollouts。（→13）
6. 統計較弱：沒有多個 seed、程式類差距多為 1–3 題、K 與 $n_{ret}$ 依測試集選定（m 的選法未說明）。（→12）
7. Fig. 4、Fig. 5 的曲線沒有數值，本頁只列論文文字給出的點，不重畫。（→13）

**常見錯誤（S15）**：(1) 把老師當成與學生同權重——LOPD 是凍結 θ̄＋latent；(2) 對老師加 stop-gradient——Eq. 10 不加，否則 composer 收不到蒸餾梯度；(3) 用學生的前 M 名當支撐集——Eq. 9 用老師的；(4) 把蒸餾的 M=20 與取樣的 top-k=20 混為一談；(5) 用 in-place 賦值替換 placeholder embedding——須用 torch.cat；(6) 老師主 forward 忘了關掉 encoder LoRA；(7) 以為 m=0 等於「沒有約束」——照 Eq. 13 字面仍有 Δ≥0，但退化點 Δ=0 滿足它，所以 m=0 擋不住退化；論文未說明 m=0 的實作是關掉 β 還是把 m 設 0。

---

## 4. 互動元件規格

共用：所有計算為小型閉式或 ≤600 步的 4 維迭代，單次 < 1 ms，同步執行；**沒有重運算，不需要 Web Worker**（程式註解寫明；若 `performance.now()` 量到單次 > 8 ms，C3 改為只在 `change` 時重算）。**沒有任何亂數**。**圖版更新一律無動畫**（重畫直接替換，不加 transition），所以 reduced-motion 下行為相同。每個圖版底部一行「讀數由本頁依 ○○ 即時計算」並加對應標籤。讀數一律「標籤在前、數值在後」（例「截斷 reverse KL　0.5210」），以 `<dl>` 呈現**完整**數值（不只摘要）；SVG 設 `role="img"`，以 `aria-describedby` 指向該 `<dl>`。數值格式：`fmt(v,d)` 先做 `Math.abs(v) < 0.5·10^{-d} ? 0 : v`，再 `toFixed(d)`，不會出現「−0.0000」。每個元件恰好一個 live region `#<id>-live`（`aria-live="polite"`），只在 `change` 或輸入停止 500 ms 後宣告一次；SVG 的 `aria-label` 用同一個 debounce 更新。滑桿、勾選框（含 label 的可點區）、`<select>`、分段按鈕、`<details>` 的 `<summary>` 觸控區皆 ≥ 44px；滑桿 `aria-valuetext` 帶單位與值。圖版第一行（`.plate-assume`，`--ink-2`）列出該元件的內建假設。

共用函式（**每個只定義一次**，名稱與 `check.py` 對應）：`softmax(u)`（先減最大值）、`logSoftmax(u)`（$u_k-\mathrm{logsumexp}(u)$）、`kl(a,b)`（$a_i<10^{-15}$ 的項記 0）、`fmt(v,d)`。

### C1 · Top-M＋合併尾部的 reverse KL（S4）

- **第一行假設**：「8 個 token 的玩具詞表；老師與學生的 logits 都是〔示範值〕，不來自任何模型。論文的 M=20 是從約十萬個 token 中取 20 個，比例與這裡不同。」
- **教的概念**：Eq. 9–10；支撐集由老師決定；截斷 KL ≤ 完整 KL；尾部內部分配不受監督；reverse 與 forward 方向不同。
- **資料**〔示範值〕：老師 logits $U=[2.0,1.2,0.6,0.1,-0.3,-0.8,-1.2,-1.5]$；學生端點 logits $Z=[0.2,1.5,-0.5,1.0,0.4,-1.0,0.8,-0.6]$；token 標為 v1–v8。
- **輸入**：滑桿「學生位置 s（0＝等於老師，1＝示範學生）」∈[0,1]，步長 0.05，預設 1（假設列寫明 $z=U+s(Z-U)$）；滑桿「M」∈{1,…,8}，預設 3；勾選框「反轉學生在尾部的機率分配」，預設關，尾部少於 2 個 token 時 disabled 並附「尾部只有 0–1 個 token，無法重新分配」。
- **計算**（`c1(s,M,perm)`）：$p^T=\mathrm{softmax}(U)$；$p^S=\mathrm{softmax}(U+s(Z-U))$；支撐集＝依 $U$ 由大到小（同值取較小索引）的前 M 個；perm 為真時，把尾部 token（依索引排序）的學生機率順序反轉後寫回；**尾部質量以尾部各項直接相加**（不用 1−Σ，避免抵消誤差）；截斷 KL＝$\sum_{v\in\mathcal V^M}p^S\log(p^S/p^T)$＋（$\tilde p^S(\perp)\ge10^{-15}$ 時）$\tilde p^S(\perp)\log(\tilde p^S(\perp)/\tilde p^T(\perp))$；完整 reverse＝`kl(pS,pT)`，完整 forward＝`kl(pT,pS)`。
- **顯示**：每個 token 一列：標籤 → 老師實心長條＋值 → 學生斜線長條＋值；前 M 列加「前 M」pill，其餘列淡化並以括號合併到下方「⊥ 合併尾部」列（兩條合併長條）。`<dl>`：完整 reverse KL、完整 forward KL、截斷 reverse KL、截斷 − 完整、尾部質量（學生）、尾部質量（老師），另以 `<table>` 列出 16 個機率。390px：長條列標籤 40px、值右對齊，`<dl>` 在圖下方兩欄格線。
- **頁面敘述**（只寫對所有輸入都成立的性質）：截斷 KL 永遠不大於完整 KL；M 增加不會使截斷 KL 變小；M=8 或 M=7 時兩者相等；反轉尾部分配時截斷 KL 不變；s=0 時所有 KL 為 0。

### C2 · δ 與 Δ 的帳本（S8）

- **第一行假設**：「單一 prefix、4 個候選 token；學生分布 p、冷啟動老師 q₀ 與每個 token 之後的期望獎勵都是〔示範值〕。老師分布 $q=\mathrm{softmax}((1-s)\log p+s\log q_0)$ 是 logit 的線性內插，不是論文的機制。用精確期望取代取樣，等於假設動作恰從 p 取樣；論文以溫度 0.7、top-p、top-k 取樣，E[δ]=−KL 在那裡不一定成立。」
- **教的概念**：Eq. 11–12；A=2r−1；〔本頁推導〕$\mathbb E[\delta]=-\mathrm{KL}$（取樣分布＝p 時）；分解 $c=-\bar Q\cdot\mathrm{KL}+\text{共變項}$。
- **資料**〔示範值〕：$p=[0.40,0.30,0.20,0.10]$；$q_0=[0.55,0.15,0.25,0.05]$。
- **輸入**：滑桿「老師位置 s（0＝等於學生 p，1＝q₀）」∈[0,1]，步長 0.05，預設 1；四個滑桿「期望獎勵 r(v1)」…「期望獎勵 r(v4)」（內文記為 $\bar r(a)$，$Q(a)=2\bar r(a)-1$）∈[0,1]，步長 0.05，預設 [0.8,0.3,0.6,0.1]；三個預設按鈕「預設」「全部 0.5」「全部 0.9」。
- **計算**（`c2(s,rb)`）：$q=\mathrm{softmax}((1-s)\log p+s\log q_0)$；$Q_a=2\bar r_a-1$；$\delta_a=\log q_a-\log p_a$；$\mathbb E\delta=\sum p\delta$；$\mathrm{KL}=$`kl(p,q)`；$\bar Q=\sum pQ$；$c=\sum pQ\delta$；共變項 $=\sum p(Q-\bar Q)\delta$；第一項 $=-\bar Q\cdot\mathrm{KL}$。顯示 $c$ 與「第一項＋共變項」兩者（相等，作為自我驗證）。
- **顯示**：表格四列（每列 token）：p、q、Q=2r−1、δ（以正負方向的發散長條＋「＋/−」文字）、p·δ、p·Q·δ；表尾：E[δ]、−KL（並排，恆相等）、c、−Q̄·KL、共變項。< 480px 改為每個 token 一張卡片（「p　0.400」等，標籤在前），總計列為獨立卡片。
- **頁面敘述**（恆成立的性質）：在此元件的精確期望下 E[δ] 與 −KL 恆相等；s=0 時所有 δ、c 為 0；期望獎勵全部相同時共變項為 0、c=−Q̄·KL（期望獎勵大於 0.5 時為負、小於 0.5 時為正）。

### C3 · 只蒸餾、固定 β 與對偶更新下老師往哪裡走（完整版在 S10；精簡版 C3-lite 在 S7）

- **第一行假設**：「〔本頁推導〕的理想化模型：學生分布 p 固定不動；只有一個 prefix、4 個 token；老師 logits 可以自由移動（論文中只能透過 96 個 latent 向量改變）；用精確期望；不含 anchor 與 top-M；p、q₀、期望獎勵為〔示範值〕；步長 η=1 是玩具值，與論文 $lr_\phi=10^{-5}$ 無關；β 上限 5 是本頁的數值保護。結果不是論文的實驗。」
- **教的概念**：β=0 時老師收斂到學生（退化）；β 固定時收斂到 $q^*$；對偶更新使 Δ 停在 m；期望獎勵與 token 無關且 Q=0 時 margin 無法滿足、β 持續增加。
- **C3-lite（S7）**：同一個 `c3('A','fixed',0)`，只顯示長條組 (a) 與 KL 折線，唯一控制項是「檢視第幾步」滑桿（預設 600）；圖下一行「加入結果獎勵約束後的版本見 10」（不在 S9 之前使用 β）。與 C3 共用快取與 live region 之外的 DOM 各自獨立（C3-lite 有自己的 `#c3l-live`，算作另一個元件）。
- **輸入（完整版）**：導覽按鈕「① 只蒸餾（β = 0）」「② 固定 β = 0.5」「③ 對偶更新（m = 0.05）」；分段按鈕「期望獎勵設定」A＝[0.8,0.3,0.6,0.1]、B＝[0.5,0.5,0.5,0.5]、C＝[0.7,0.7,0.7,0.0]；分段按鈕「β 的來源」固定／對偶更新；滑桿 β∈[0,1] 步長 0.05（固定模式）；滑桿 m∈[0,0.2] 步長 0.01（對偶模式，$\eta_\beta=0.5$ 固定為論文值）；滑桿「檢視第幾步」∈[0,600]，預設 600。預設＝①、A。
- **計算**（`c3(preset,mode,beta,m)`，任一輸入改變時重算全部 600 步並快取 `${preset}|${mode}|${beta}|${m}`）：
  ```
  p=[.4,.3,.2,.1]; Q=2*rbar-1; Qbar=Σ p·Q; u=log(q0); b=(mode=='fixed')?beta:0; BMAX=5
  for i=0..600:
     lq=logSoftmax(u); q=exp(lq); KL=Σ p(log p − lq); Δ=Σ p·Q·(lq − log p)
     record(q,KL,Δ,b); if i==600 break
     g_k=(q_k−p_k) − b·(p_k·Q_k − q_k·Qbar)
     u=u−1.0·g;  u=u−max(u);  u_k=max(u_k,−30)            // 平移不改變 softmax；下限是數值保護
     if mode=='dual': b=min(BMAX, max(0, b+0.5·(m−Δ)))   // 用本步更新前的 Δ，順序同 Alg. 1 第 12–14 行
  qstar_k ∝ max(0, p_k(1+b_final·Q_k))
  ```
- **數值保護與穩定性**：logits 平移＋下限 −30；β 上限 5，達到上限時 β 折線末端加「已達上限 5（本頁保護）」標記與讀數註記；$q^*$ 的權重取 max(0,·)，若有 token 的 $1+\beta Q\le0$，顯示「q* 在 v○ 為 0：老師 logit 只能漸近地下降，600 步內不會到 0」。固定模式 β∈[0,1]、Q∈[−1,1] 使 $1+\beta Q\ge0$；對偶模式 β 可暫時超過 1（例 C、m=0.2 第 25 步 1.2705），由上述保護處理。預設只有 A、B、C 三組，`check.py` 已掃過三組 × 固定 β（0–1，步長 0.05）× 對偶 m（0–0.2，步長 0.01）：第 599→600 步 q 的最大變化 1.0e-6（C 固定 β=1 除外：該處 $q^*_4=0$，q₄ 單調緩降），沒有振盪。
- **顯示**：(a) 長條組：每個 token 三個標記——p（空心框）、第 k 步的 q（實心長條）、$q^*$（短橫線）；(b) 折線「Δ 對步數」，對偶模式加 m 的水平虛線（標「m」）；(c) 折線「β 對步數」。縱軸：Δ 取 [min(−0.05, minΔ), max(0.35, maxΔ)]，β 取 [0, max(1, maxβ)]（maxβ ≤ 5）。目前步數以垂直細線標在 (b)(c)。`<dl>`：步數、KL、Δ、β、q（四個值）、q*。390px：(a)(b)(c) 上下堆疊，各高 160px；控制列在圖下方。
- **頁面敘述**（恆成立或論述性）：β=0 時 q 必收斂到 p（梯度 q−p）；期望獎勵全部相同（Q 為常數）時 $q^*=p$，Δ=−Q·KL(p‖q)，只有 Q=0（期望獎勵 0.5，預設 B）時 Δ 恆為 0；Q<0 時 Δ=+|Q|·KL，一個不含獎勵資訊的脈絡也能先滿足 margin；β 每步增加 0.5·m 只在 Δ=0 時成立；對偶模式若收斂，收斂點滿足 Δ=m（或 β=0 且 Δ≥m）。數值以第 5 節驗收表為準，不在內文寫死。

### C4 · 分數換算成題數（S12）

- **第一行假設**：「〔本頁推導〕題數＝round(分數 × N / 100)；SE 是單一分數的題目取樣標準誤，假設每題答對與否獨立、同機率，只反映單次評測的取樣雜訊（溫度 0.7），不含訓練 seed 的變異（論文未報告）。這不是配對檢定：兩方法在同一批題目上評測，配對差的不確定性需要逐題結果，論文沒有提供。EnvScaler（連續 reward）與 ACEBench（數值不在整數題數格點上）不列入。」SE 與 ±1.96 的意義在 S12 內文、C4 之前先定義。
- **教的概念**：程式類與 BFCL 的差距換成題數；單一分數本身的題目取樣雜訊有多大。
- **輸入**：原生 `<select>`「比較項目」，14 個選項（標籤「Qwen3-4B · LiveCodeBench v5（N = 279）」格式），預設 `q4-lcb5`。資料（id、backbone、benchmark、N、LOPD、最強基線、基線分數）：
  `q4-bfcl` 4B BFCL Avg 800 27.38 GRPO 25.25；`q8-bfcl` 8B BFCL Avg 800 29.88 GRPO 29.00；`q4-lcb5` 4B LCB v5 279 50.54 GRPO 50.18；`q4-lcb6` 4B LCB v6 131 45.04 GRPO 44.27；`q4-lcb` 4B LCB Avg 410 48.78 GRPO 48.29；`q4-he` 4B HumanEval+ 164 87.80 SDFT 87.20；`q4-mbpp` 4B MBPP+ 378 78.57 Skill-SD／SDFT 76.98；`q4-ep` 4B EvalPlus Avg 542 81.36 SDFT 80.07；`o7-lcb5` Olmo LCB v5 279 53.41 SDPO 49.82；`o7-lcb6` Olmo LCB v6 131 45.80 GRPO 45.80；`o7-lcb` Olmo LCB Avg 410 50.98 GRPO 48.29；`o7-he` Olmo HumanEval+ 164 90.24 GRPO／SDFT 89.02；`o7-mbpp` Olmo MBPP+ 378 73.28 SDFT 73.02；`o7-ep` Olmo EvalPlus Avg 542 78.41 SDFT 77.86。
- **計算**（`c4(row)`）：$n_{LOPD}=\mathrm{round}(aN/100)$、$n_{base}=\mathrm{round}(bN/100)$；差距題數 $=n_{LOPD}-n_{base}$；差距百分點 $=a-b$；$SE=100\sqrt{\hat p(1-\hat p)/N}$，$\hat p=a/100$。
- **顯示**：`<dl>`（標籤在前）：「LOPD　50.54%　＝　141 / 279 題」「最強基線（GRPO）　50.18%　＝　140 / 279 題」「差距　1 題（0.36 個百分點）」「LOPD 分數的單一分數 SE（題目取樣，非配對檢定）　2.99 個百分點」。小圖：橫軸局部放大到 [a−3SE, a+3SE]，LOPD 為實心圓並畫 ±1.96·SE 誤差線（標「LOPD 單一分數的題目取樣範圍」），基線為空心菱形。**不顯示「差距÷SE」，也不輸出「之內／之外」之類的判定文字。** 390px：`<dl>` 一行一項，小圖全寬、高 72px。

### 4.F 靜態圖（非互動）與重畫機制

- 每個 fig 函式簽名為 `figX(container)`：讀 `w = container.clientWidth`，以 `w` 為 SVG 寬度（viewBox 寬＝w，不靠縮放），圖上文字固定 12–13px。共用一個 `ResizeObserver`，對所有 fig 容器與元件圖容器觀察寬度變化，150 ms debounce 後只重畫寬度改變的那個；首次在 `DOMContentLoaded` 繪製。每個 fig 有兩個明確分支：`w ≥ 480`（寬版）與 `w < 480`（窄版），以下列出。selfTest 檢查 390px 時所有 SVG `<text>` 的 computed font-size ≥ 12px。
- **Hero**（`figHero`）：寬版兩列左右排列（方塊 → 長條組），長條組之間雙向線在右側；窄版兩列各自上下排列（方塊在上、長條組在下），雙向線改在兩列之間垂直方向，標籤置於線旁。
- **F1**（S3，`figOPSD`）：寬版左右兩欄「OPSD：老師＝同權重 θ＋固定脈絡 c_fix（sg）」「LOPD：老師＝凍結 θ̄＋c_φ（無 sg，梯度回到 φ）」，各畫學生取樣 → 同一 prefix → 兩個分布 → divergence；窄版兩欄上下堆疊，每欄高度自動。
- **F2**（S6，`figSlots`）：寬版一條 token 列 `[t₁ … t_a | pad × 96 | t_{a+1} … t_L]`，96 格分成 3 組 × 32（每組標「經驗 j」），下方箭頭「embedding 後以 ⟨e_{j,k}⟩ 替換（Eq. 18，torch.cat）」；窄版 96 格改為 3 列 × 32 細格（每格寬 =(w−48)/32），組標籤放在每列左側上方，前後文字 token 各成一列。
- **F3**（S11，`figAblation`）：Fig. 3 的七個數字畫成**點圖**（橫軸 0.54–0.65，非零起點故不用長條），寬版標籤欄 160px 在左（「m = 0.05　0.637」），窄版標籤放在每個點的上一行、點圖佔全寬；Frozen φ₀ 0.573 畫垂直虛線並標「以凍結 composer 訓練的學生」。圖說：「重畫論文 Fig. 3 的數字；每點為單次訓練，論文未給變異。」

---

## 5. 驗收數值表（`verify/check.py` → `check.out`，與頁面同演算法）

容許誤差：同演算法只差浮點加總順序，除另註外一律 ±1e-4（顯示到 4 位小數時應完全一致）。

### C1（老師 $p^T$＝[0.4813, 0.2163, 0.1187, 0.0720, 0.0483, 0.0293, 0.0196, 0.0145]；s=1 時 $p^S$＝[0.0894, 0.3280, 0.0444, 0.1990, 0.1092, 0.0269, 0.1629, 0.0402]）

| s | M | 完整 reverse | 完整 forward | 截斷 reverse | 尾部（學生） | 尾部（老師） |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.6173 | 0.6705 | 0.3620 | 0.9106 | 0.5187 |
| 1 | 2 | 0.6173 | 0.6705 | 0.3682 | 0.5826 | 0.3024 |
| 1 | 3 | 0.6173 | 0.6705 | 0.5210 | 0.5382 | 0.1837 |
| 1 | 5 | 0.6173 | 0.6705 | 0.5302 | 0.2300 | 0.0634 |
| 1 | 7 | 0.6173 | 0.6705 | 0.6173 | 0.0402 | 0.0145 |
| 1 | 8 | 0.6173 | 0.6705 | 0.6173 | 0.0000 | 0.0000 |
| 0.5 | 1 | 0.1733 | 0.1697 | 0.1183 | 0.7552 | 0.5187 |
| 0.5 | 3 | 0.1733 | 0.1697 | 0.1583 | 0.3552 | 0.1837 |
| 0.5 | 5 | 0.1733 | 0.1697 | 0.1588 | 0.1284 | 0.0634 |
| 0 | 任意 | 0.0000 | 0.0000 | 0.0000 | 同老師 | — |

反轉尾部（s=1）：M=1 截斷 0.3620→0.3620、完整 0.6173→1.3361；M=3 截斷 0.5210→0.5210、完整 0.6173→0.8230；M=5 截斷 0.5302→0.5302、完整 0.6173→0.6080。

### C2

| s | r̄ | q | E[δ] | KL | Q̄ | c | −Q̄·KL | 共變項 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 預設 | 0.55, 0.15, 0.25, 0.05 | −0.1052 | 0.1052 | 0.0800 | 0.2240 | −0.0084 | 0.2324 |
| 0.5 | 預設 | 0.4808, 0.2175, 0.2292, 0.0725 | −0.0278 | 0.0278 | 0.0800 | 0.1140 | −0.0022 | 0.1162 |
| 0 | 預設 | 0.4, 0.3, 0.2, 0.1 | 0.0000 | 0.0000 | 0.0800 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 全部 0.5 | 同第一列 | −0.1052 | 0.1052 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 全部 0.9 | 同第一列 | −0.1052 | 0.1052 | 0.8000 | −0.0842 | −0.0842 | 0.0000 |
| 1 | 全部 0.1 | 同第一列 | −0.1052 | 0.1052 | −0.8000 | 0.0842 | 0.0842 | 0.0000 |

s=1、預設時 δ＝[0.3185, −0.6931, 0.2231, −0.6931]。（「全部 0.1」不在預設按鈕中，以滑桿設定。）

### C3（第 0、25、100、600 步；q 四位小數；q* 用最後的 β）

| 設定 | 步 | q | KL | Δ | β |
| --- | --- | --- | --- | --- | --- |
| A 固定 β=0（q*=p） | 0 | 0.55, 0.15, 0.25, 0.05 | 0.1052 | 0.2240 | 0 |
|  | 25 | 0.4008, 0.3009, 0.2014, 0.0969 | 0.0001 | 0.0030 | 0 |
|  | 100／600 | 0.4, 0.3, 0.2, 0.1 | 0.0000 | 0.0000 | 0 |
| A 固定 β=0.5（q*=0.5, 0.2308, 0.2115, 0.0577） | 25 | 0.5002, 0.2309, 0.2119, 0.0571 | 0.0337 | 0.1322 | 0.5 |
|  | 600 | 0.5, 0.2308, 0.2115, 0.0577 | 0.0332 | 0.1313 | 0.5 |
| A 固定 β=1（q*=0.5926, 0.1667, 0.2222, 0.0185） | 25 | 0.5899, 0.1634, 0.2193, 0.0274 | 0.1380 | 0.2735 | 1 |
|  | 600 | 0.5926, 0.1667, 0.2222, 0.0185 | 0.1667 | 0.3040 | 1 |
| B 固定 β=1（q*=p） | 0 | 0.55, 0.15, 0.25, 0.05 | 0.1052 | 0.0000 | 1 |
|  | 600 | 0.4, 0.3, 0.2, 0.1 | 0.0000 | 0.0000 | 1 |
| C 固定 β=1（q*=0.4444, 0.3333, 0.2222, 0） | 0 | 0.55, 0.15, 0.25, 0.05 | 0.1052 | 0.0549 | 1 |
|  | 100 | 0.4427, 0.3315, 0.2204, 0.0054 | 0.2020 | 0.3280 | 1 |
|  | 600 | 0.4441, 0.3330, 0.2219, 0.0010 | 0.3690 | 0.5006 | 1 |
| A 對偶 m=0 | 600 | 0.4, 0.3, 0.2, 0.1 | 0.0000 | 0.0000 | 0.0000 |
| A 對偶 m=0.05（q*=0.4398, 0.2724, 0.2046, 0.0832） | 25 | 0.4346, 0.2759, 0.2032, 0.0863 | 0.0035 | 0.0424 | 0.1868 |
|  | 100／600 | 0.4398, 0.2724, 0.2046, 0.0832 | 0.0049 | 0.0500 | 0.1944 |
| A 對偶 m=0.2 | 25 | 0.5579, 0.1854, 0.2111, 0.0456 | 0.0790 | 0.2026 | 0.8428 |
|  | 600 | 0.5436, 0.2006, 0.2166, 0.0392 | 0.0757 | 0.2000 | 0.7309 |
| B 對偶 m=0.05 | 25／100／600 | 0.4008…／p／p | — | 0.0000 | 0.6250／2.5000／5.0000（第 200 步起達上限 5，顯示上限標記） |
| C 對偶 m=0.05 | 25 | 0.4092, 0.3062, 0.2037, 0.0809 | 0.0023 | 0.0288 | 0.2375 |
|  | 600 | 0.4139, 0.3105, 0.2070, 0.0686 | 0.0068 | 0.0500 | 0.2663 |
| C 對偶 m=0.2 | 25 | 0.4400, 0.3264, 0.2129, 0.0207 | 0.0818 | 0.1881 | 1.2705 |
|  | 600 | 0.4362, 0.3272, 0.2181, 0.0185 | 0.0907 | 0.1999 | 0.7777 |

### C4（題數完全一致；SE ±0.01；「差距÷SE」欄只作驗算參考，頁面不顯示）

| id | LOPD 題數 | 基線題數 | 差距（題／pp） | SE（pp） | 差距÷SE |
| --- | --- | --- | --- | --- | --- |
| q4-bfcl | 219/800 | 202/800 | 17／2.13 | 1.58 | 1.35 |
| q8-bfcl | 239/800 | 232/800 | 7／0.88 | 1.62 | 0.54 |
| q4-lcb5 | 141/279 | 140/279 | 1／0.36 | 2.99 | 0.12 |
| q4-lcb6 | 59/131 | 58/131 | 1／0.77 | 4.35 | 0.18 |
| q4-lcb | 200/410 | 198/410 | 2／0.49 | 2.47 | 0.20 |
| q4-he | 144/164 | 143/164 | 1／0.60 | 2.56 | 0.23 |
| q4-mbpp | 297/378 | 291/378 | 6／1.59 | 2.11 | 0.75 |
| q4-ep | 441/542 | 434/542 | 7／1.29 | 1.67 | 0.77 |
| o7-lcb5 | 149/279 | 139/279 | 10／3.59 | 2.99 | 1.20 |
| o7-lcb6 | 60/131 | 60/131 | 0／0.00 | 4.35 | 0.00 |
| o7-lcb | 209/410 | 198/410 | 11／2.69 | 2.47 | 1.09 |
| o7-he | 148/164 | 146/164 | 2／1.22 | 2.32 | 0.53 |
| o7-mbpp | 277/378 | 276/378 | 1／0.26 | 2.28 | 0.11 |
| o7-ep | 425/542 | 422/542 | 3／0.55 | 1.77 | 0.31 |

所有 aN/100 與最近整數的差 ≤0.04（BFCL Avg 是 0.125 的倍數，27.375 捨入為 27.38，差 0.04；其餘 ≤0.03；`check.out` 列出原值），題數換算有效。

### 論文算術（`check.out` 末段）

彙總算法 0 個不符；4B/8B 對最強基線 Env 1.9/6.2、BFCL 2.13/0.88、ACE 4.6/4.7；first-step 0.3751；Reward/tool call 樸素比值 0.0125／0.0320／0.0337；320/1600=0.20、576/1600=0.36、30% 對應 480；A 換算 −0.028／0.274；β 每步 0.025、50 步上限 1.25；Fig. 5(a) 落差 3.1／2.3。C3 穩定性掃描：三組預設 × 固定 β 0–1 × 對偶 m 0–0.2，第 599→600 步 q 最大變化 1.0e-6（C 固定 β=1 除外），對偶掃描的 β 最大值 5.0000（上限）；B、m=0.2 第 50 步達上限。推導檢查（200 個隨機案例）：∂KL/∂u 誤差 6.6e-10、∂Δ/∂u 4.5e-10、∂KL/∂z 1.1e-9、q* 處梯度 1.6e-16、截斷 KL − 完整 KL ≤ 2.2e-16。

---

## 6. 實作計畫

只列與 238 頁（`tutorials/prediction-targets-k-diff.html`）不同或新增之處；其餘完全沿用：CSS tokens（含深色模式三處覆寫與 `pm-theme`）、Google Fonts 與 KaTeX 0.16.9、appbar、`#pm-bar`（`../index.html`、`index.html`）、`initToc`、`initTheme`、`.plate`／`.plate-h`／`.plate-f`／`.readout`／`.ctl`／`.seg`／`.tscroll`／`.check`／`.callout`、`announce(id,s)`、`renderMath`、`selfTest()` 外框（含 `scrollWidth <= innerWidth`）、`renderTimeline`。

### 6.1 檔案與連結

- `scripts/tutorial_src/119/build/index.html`，單檔，CSS／JS／timeline 內嵌，不 `fetch`。`<title>`「潛在脈絡自蒸餾 LOPD」。
- 連結：arXiv `https://arxiv.org/abs/2608.13040`；關聯圖 `../index.html#paper-119`；筆記 `<slug>.html`。
- 圖表色：`--c-t`（老師）淺 #3F6E8C／深 #8DB4D3、`--c-s`（學生）淺 #6B7A3A／深 #A9BC7C，三處 tokens 區塊都定義；學生一律加 45° 斜線 `<pattern>`（stroke 引用 token），老師實心。

### 6.2 新增 JS（IIFE 內；每個函式只有一個定義）

- 數學與格式：`softmax`、`logSoftmax`、`kl`、`fmt`。
- C1：`U1`、`Z1`、`c1(s,M,perm)` → `{pS,pT,sup,tailS,tailT,rev,fwd,revTr}`；`renderC1()`、`initC1()`。
- C2：`P`、`Q0`、`c2(s,rb)` → `{q,d,Ed,K,Qb,c,cov,first}`；`renderC2()`、`initC2()`。
- C3：`PRESETS`、`BMAX=5`、`c3(preset,mode,beta,m)` → `{hist:[{q,KL,D,b}], qstar, capped}`（Map 快取）；`renderC3()`、`initC3()`；`renderC3lite()`、`initC3lite()`（呼叫同一個 `c3`）。
- C4：`ROWS`、`c4(row)` → `{na,nb,dn,dpp,se}`；`renderC4()`、`initC4()`。
- 靜態圖：`figHero(el)`、`figOPSD(el)`、`figSlots(el)`、`figAblation(el)`（4.F：讀容器寬、兩個斷點分支）；`observeWidths()` 建立唯一的 `ResizeObserver`（150 ms debounce），登記所有 fig 與元件圖容器及其重畫函式。SVG 內文字用 `<text>` 與 Unicode，不放 KaTeX。
- 動態文字（讀數、aria-label、live region、SVG `<text>`）只用預組 Unicode 字元（「Δ」「β」「≥」「q*」）；不用組合上橫線，r̄ 在這些位置一律寫成「期望獎勵 r(v1)」，內文仍用 KaTeX `$\bar r(a)$`。

### 6.3 `selfTest()` 斷言（`?test`）

- `c1(1,3)`：revTr 0.5210、rev 0.6173、fwd 0.6705、tailS 0.5382（±1e-4）；`c1(1,3,true).revTr` 與 `c1(1,3).revTr` 差 < 1e-12；`c1(0.5,8).revTr` 與 `.rev` 差 < 1e-12；對 s∈{0,0.25,…,1}、M∈1..8 皆 revTr ≤ rev + 1e-12；`fmt(-1e-17,4)` === "0.0000"。
- `c2(1,[.8,.3,.6,.1])`：c 0.2240、cov 0.2324、|Ed+K| < 1e-12、|c−(first+cov)| < 1e-12；`c2(1,[.9,.9,.9,.9]).c` = −0.0842。
- `c3('A','fixed',0.5)` 第 600 步 q[0]=0.5000、Δ=0.1313；`c3('A','dual',0,0.05)` 第 600 步 Δ=0.0500、β=0.1944；`c3('B','dual',0,0.05)` 第 100 步 β=2.5000、第 600 步 β=5.0000 且 `capped===true`；對三組預設與 m∈{0,0.05,0.1,0.2}，所有步的 β ≤ 5 且 |q600−q599| < 1e-5；`c3('A','fixed',0)` 第 600 步 KL < 1e-6。
- `c4(ROWS[2])`：na 141、nb 140、se 2.99（±0.01）；14 列 |aN/100 − round| ≤ 0.05；C4 的 DOM 中不存在「之內」「之外」字串。
- 390px：每個 `.eq` 不撐開 `main`；所有 SVG `<text>` 的 computed font-size ≥ 12px；`scrollWidth <= innerWidth`。輸出一行 `selfTest: N passed, M failed`。

### 6.4 版面、公式與無障礙（新增部分）

- 390px：16px 側邊留白、無水平捲動；圖寬＝容器寬（4.F），圖上文字 12–13px 不縮放；Table 1、2 摘要表包 `overflow-x:auto`＋左右漸層，< 480px 改為每個方法一張卡片（「EnvScaler　63.7」等，標籤在前）；S14 對照表 < 480px 改為每條主張一張卡片（主張／位置／證據／判讀四行）；Alg. 1 兩欄在 < 768px 改為每行下接說明。
- **公式**：Eq. 3、10、12、13 **一律**寫成兩行 `aligned`（不分斷點），`\tag{Eq. n}` 放在 `aligned` 外層、同一個 `$$…$$` 內。建置時的 TeX 原文：
  - Eq. 3：`\begin{aligned}\mathcal L_{OPSD}(\theta)=\mathbb E_{\tau\sim\pi^S_\theta}\Big[\textstyle\sum_{t=1}^{|\tau|}\sum_{n=1}^{L_t}D\big(&\mathrm{sg}[\pi^T_\theta(\cdot\mid s_t,c_{fix},a_{t,\lt n})]\\&\big\|\,\pi^S_\theta(\cdot\mid s_t,a_{t,\lt n})\big)\Big]\end{aligned}\tag{Eq. 3}`
  - Eq. 10：`\begin{aligned}\mathcal L_{distill}(\theta,\phi)=\;&\mathbb E_{x\sim\mathcal D}\,\mathbb E_{\tau\sim\pi^S_\theta(\cdot\mid x)}\\&\Big[\frac{\sum_{t,n}\omega_{t,n}D_{KL}(\tilde p^S_{t,n}\|\tilde p^T_{t,n})}{\sum_{t,n}\omega_{t,n}}\Big]\end{aligned}\tag{Eq. 10}`
  - Eq. 12：`\begin{aligned}\Delta(\phi)&=\mathbb E_\tau\Big[\frac{\sum_{t,n}\omega_{t,n}A(\tau)\,\delta_{t,n}(\phi)}{\sum_{t,n}\omega_{t,n}}\Big],\\A(\tau)&=2r(\tau)-1\in[-1,1]\end{aligned}\tag{Eq. 12}`
  - Eq. 13：`\begin{aligned}\min_{\theta,\phi}\max_{\beta\ge0}\ &\mathcal L_{distill}(\theta,\phi)+\beta\big(m-\Delta(\phi)\big)\\&+\lambda\big\|c_\phi-\mathrm{sg}[c_{\phi_0}]\big\|_2^2\end{aligned}\tag{Eq. 13}`
  - 其他公式單行。每個區塊公式包在 `<div class="eq" tabindex="0" role="group" aria-label="Eq. n">`，`.eq{overflow-x:auto;overflow-y:hidden;max-width:100%}`，focus 樣式同全站（2px `--accent`），鍵盤可用方向鍵捲動。**HTML 內的 TeX 一律把 `<` 寫成 `\lt`、`>` 寫成 `\gt`**。
- 每個元件一個 live region（C1、C2、C3-lite、C3、C4 共 5 個），宣告範例：C1「M = 3，截斷 reverse KL 0.521，完整 0.617」；C3「第 600 步，Δ = 0.050，β = 0.194」；C4「LiveCodeBench v5：差距 1 題，單一分數 SE 2.99 個百分點」。
- 不依賴 hover：所有數值在 `<dl>`；C3 以「檢視第幾步」滑桿取代 hover 游標。
- 老師／學生以實心／斜線＋文字標籤區分；C3 的 p／q／q* 以空心框／實心長條／短橫線區分；C4 以實心圓／空心菱形區分。

### 6.5 引用與標籤

- 論文數字附出處（Eq./Fig./Table/§/App.）；元件讀數以圖版底部一行說明代替。
- 四種小標籤（`--ink-3` 邊框 pill）：「本頁推導」「本頁推論」「本頁疑問」「示範值」，以及「作者認為」，各附一行依據。**量測結果與作者解釋分段寫**：S11（Fig. 3 的解釋）、S13（Fig. 4「denser learning signal」、Table 3「internalizes procedural guidance」）、S12（「No hand-crafted context is universally optimal」的成因解釋）一律以「作者認為」開頭。

### 6.6 學習目標參考答案在內文的位置

每個 LO 的答案以**一段自足的段落**寫在章節本文（先定義段落中用到的每個符號），`<details>` 只重述：LO1 → S3 F1 之後一段；LO4 → S4 C1 之後末段；LO3 → S5 末段；LO2 → S6 末段；LO5 → S7 C3-lite 之後一段；LO6 → S8 C2 之後一段；LO7 → S9 β 數量級段之後一段；LO8 → S10 C3 之後一段；LO9 → S11 末段；LO10 → S12 C4 之後一段；LO11 → S13 末段；LO12 → S14 表後一段。

### 6.7 研究脈絡（S16）與延伸閱讀（S17）

- `notes/timeline.json`（13 筆，欄位 `year,title,authors_short,arxiv,role,source`）原樣貼入 `<script type="application/json" id="timeline-data">`。
- `TL_GROUP`（arXiv → 類別、筆記）：1011.0686、2306.08543 → 「on-policy 模仿／蒸餾」`on-policy-distillation-imitation.html`；1503.02531 → 「蒸餾基礎」`knowledge-distillation-foundations.html`；2209.15189 → 「context distillation」`context-distillation-self-distillation.html`；2402.03300 → 「RLVR」`rlvr-grpo-credit-assignment.html`；2412.06769 → 「latent 推理」`lopd-latent-reasoning-continuous-tokens.html`；2509.24704 → 「latent 記憶」`lopd-latent-memory-for-agents.html`；2601.18734、2601.19897 → 「OPSD」`lopd-opsd-self-teacher-privileged-context.html`；2601.20802 → 「OPSD」`sdpo-lineage-overview.html`；2604.10674 → 「特權脈絡」`lopd-privileged-context-substrates.html`；2606.11709 → 「失敗模式」`lopd-opd-failure-modes-and-stabilization.html`；2608.13040 → 「本文」`lopd-method-learnable-context-and-margin.html`（`--accent` 外框）。不在表中歸「其他」。
- S17 依 hub 閱讀順序：`lopd-lineage-hub`、`rlvr-grpo-credit-assignment`、`ppo-grpo-dapo-llm-rl-objectives`、`knowledge-distillation-foundations`、`on-policy-distillation-imitation`、`context-distillation-self-distillation`、`sdpo-lineage-overview`、`lopd-opsd-self-teacher-privileged-context`、`lopd-privileged-context-substrates`、`lopd-opd-failure-modes-and-stabilization`、`lopd-latent-reasoning-continuous-tokens`、`lopd-latent-memory-for-agents`、`lopd-method-learnable-context-and-margin`；另列 `retrieval-augmented-external-memory`、`rich-feedback-and-retrospection`（筆記中引用、站上已存在）。連結文字取 frontmatter `title`（站上既有筆記的 title 由 `scripts/notes_split/<slug>.md` 讀取）。

### 6.8 文案規則

繁體中文、台灣用語、字面學術語氣；不用比喻、反問、宣傳形容詞。用詞：特權脈絡（privileged context）、老師／學生、前綴（prefix）、退化（collapse）、對偶變數、彙總欄。專有名詞首次出現附英文。不寫「證明」「保證」（margin 只是軟約束）。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 理想化模型被誤讀為論文結果 | 05–07 的梯度式、E[δ]=−KL、q*、C3 都是本頁推導 | 每處加「本頁推導」標籤＋一行依據；C3 第一行列假設並寫「不是論文的實驗」。曾考慮「學生＋老師＋β 聯合」玩具：在 Q 固定的假設下 m>0 一律收斂到最佳 token、m 越大越快，會呈現與 Fig. 3（m=0.20 低於 0.05）相反的單調趨勢，而 m=0 低於凍結老師幾乎由假設決定，因此**不做**，改以內文說明「學生追上老師也使 Δ→0」 |
| R2 | Fig. 4、Fig. 5(b–e) 無數值 | 只能列文字給出的點 | 不畫曲線、不補值；表題註明 |
| R3 | Table 1 vs Fig. 4 | GRPO 61.8 vs 0.611、Skill-SD 59.1 vs 0.588；可能是最佳 checkpoint 對終值（推測） | 兩組數字並列並寫「論文未解釋」；推測原因時標明是推測 |
| R4 | 題數換算的前提 | BFCL、LCB、EvalPlus 假設每題 0/1 計分；EnvScaler 為連續 reward、ACE 不在格點 | C4 排除 Env 與 ACE，第一行寫明 |
| R5 | 檢索自我匹配 | 訓練時 Ret(x) 可能回傳同題成功軌跡；Fig. 6 的 task id 像訓練語料編號且只展示 rank 2；論文未說明 | 列在 S15「未說明」與 S13 Fig. 6 段，標「本頁疑問」，不寫成事實 |
| R6 | 「r̄」字元顯示 | 組合上橫線在部分字型錯位 | 6.2：動態文字寫「期望獎勵 r(v1)」，只在 KaTeX 內文用 $\bar r$ |
| R7 | 相對連結 | 建置位置與上線位置不同 | 接受；預覽只檢查外部連結與錨點 |
| R8 | 筆記更新 | 建置前 notes 若增減 | 重讀 `notes/`；新 arXiv 未在 `TL_GROUP` 者歸「其他」並回報 |
| R9 | 篇幅 | 18 個章節（S0–S17）、4 元件＋C3-lite、4 張靜態圖 | 完整 Table 1、2 放 `<details>`；S14 表格窄螢幕改卡片 |

---

## 8. 修訂紀錄（round 1，依 `reviews/round1.md`；前版 `design.round0.md`）

| 項目 | 變更 |
| --- | --- |
| 1 | 1.3 與 S14：「三個 backbone、七個 benchmark」改為聯集、非 3×7 全格（4B 全 7、8B 3 工具、Olmo3 4 程式）（#1） |
| 2 | 新增 4.F：靜態圖讀容器寬、單一 ResizeObserver 重畫、寬／窄兩分支；selfTest 檢查 390px 文字 ≥ 12px（#2） |
| 3 | 6.4：Eq. 3、10、12、13 一律兩行 aligned（附 TeX 原文），`\tag` 在外層；`div.eq` 可鍵盤捲動（#3） |
| 4 | 大綱重排為 S0–S17：S1 拆成 S1／S2，S4 拆成 S5／S6，S7 拆成 S9／S10；S7 新增 C3-lite；LO 章欄、符號表章欄、6.6、6.7、callout 連結同步（#9、#12、#13、#15） |
| 5 | E[δ]=−KL：補取樣條件、期望 vs 取樣平均、Q̄ 正負兩種情形；Q(a) 白話與三等號推導（步驟 7、LO6、S8、C2）（#4、#5、#10） |
| 6 | C3：常數 Q 的敘述改為 Δ=−Q·KL；β 上限 5 與標記；穩定性掃描寫入規格與 `check.py`；B 對偶驗收值 15.0000→5.0000（#6、#16） |
| 7 | β 數量級改為 Δ=0 情形、非上限；m=0 退化點可行、排除論述需 m>0（步驟 8、步驟 10、LO9、常見錯誤 7）（#7、#8） |
| 8 | C2／C1 滑桿改為「老師位置」「學生位置」，假設列寫明內插式（#11） |
| 9 | Hero 卡片與圖標籤只用卡內定義的詞（#14） |
| 10 | C4：S12 先定義 SE 與 ±1.96；讀數標「單一分數 SE，非配對檢定」；刪除「之內／之外」與「差距÷SE」（#17） |
| 11 | 用詞與出處：平均獎勵過半、測試集代理值、以凍結 composer 訓練的學生、Fig. 4 為測試集評測、預算是否計入未說明（本頁推論）、m 選法未說明、Fig. 6 自我檢索疑問、梯度句限縮到理想化模型（#18–#24） |
| 12 | 第 4 節共用：無動畫、`fmt` 去除 −0、`<dl>` 完整讀數＋`aria-describedby`、所有控制項觸控 ≥ 44px；比喻用詞改字面（#25–#28） |
| 13 | `verify/check.py`：C3 加 β 上限 5 與穩定性掃描；重新執行至完成，`check.out` 更新 |

## 9. 修訂紀錄（readiness 後）

- **Blocker**：$\bar\theta$ 在 S5 的 Eq. 7 處先行定義（訓練開始前的 base 模型權重，訓練中不更新）；符號表該列改為 S5。
- 符號表版面定案：頁面只用每章末的可收合小表，1.4 的表是設計用彙總。
- S7 的 C3-lite 指向文字改為「加入結果獎勵約束後的版本見 10」，S9 之前不出現 β。
- Eq. 4 照原文顯示 $K$ 個 token，緊接一句說明與 Eq. 7 的 $J\cdot K$ 不一致（頁面 S14 對照表也列）。
- m=0：論文未說明實作；頁面寫「本頁 C3 以字面 m=0、β 照常更新計算」，不替論文決定。
- $q^*$：權重取 $\max(0,1+\beta Q)$；若權重總和為 0（本頁預設不會發生）退回 $q^*=p$。
- 公式標號沿用 238/118 外殼的 `.eq > .eq-m + .eq-n`（標號在公式框外），`.eq-m` 加 `tabindex=0 role=group aria-label`；不使用 `\tag`。

## 10. 建置紀錄（build round）

- 原始碼：`build/src/`（head、css_base（118 外殼原樣）、css_add、body1–3、app_head／app_shared（118 共用函式原樣）、app_widgets、app_tail），`assemble.py` 組成 `build/index.html`；`lo.py` 為 LO 自我檢查。
- 與設計的差異：公式標號用外殼的 `.eq-n`（不用 `\tag`）；C4 讀數框改為單欄（側欄 260px 放不下兩欄）；`.ch-num`、`.plate-h .id` 字距由 .06em 調為 .04em、學生圖例改用 mask 斜線（impeccable：wide-tracking、repeating-stripes-gradient）；Eq. 5–6、7、17–18 也改為兩行 aligned（390px 時單行超寬）；C4 題數近整數斷言放寬為 ≤0.05（BFCL Avg 27.38 由 27.375 捨入）。
- 驗證：`?test` 1440 與 390px iframe 皆 63 passed、0 failed；impeccable 0 anti-patterns（17 則 numbered-section-labels advisory，依外殼保留）；LO 自我檢查 12/12；console 0 錯誤；截圖在 `build/caps/`。
