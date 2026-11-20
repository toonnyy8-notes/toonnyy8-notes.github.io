# 119 顧問筆記:Latent On-Policy Self-Distillation(LOPD,arXiv 2608.13040v1)

給教學設計者,不是給讀者。凡標 〔顧問推導〕 者為我自己的推導或算術,論文沒有寫;其餘數字與式號皆出自論文。
論文沒有定理、沒有附錄證明。附錄 A 是實作細節,附錄 B 是實驗設定。唯一的「論證」是 §3.3 一段口語的退化排除論述,加上 Fig. 3 的消融。

---

## 1. 論證鏈(8 步)

**步 1 — OPSD 的形式化(§3.1,Eq. 1–3)**
- 假設:多輪 agent,學生狀態 s_t = (x, o_≤t, a_<t);老師是「同一個模型 + 額外特權 context」,c_fix = Φ_fix(x, E),Φ_fix 是設計者寫死的規則(Eq. 2)。
- 內容:Eq. 3 在學生自己走過的每個 prefix (t, n) 上,用 token-level divergence D 比對 sg[老師] 與學生。
- 證明了什麼:沒有證明;只是定義。「supervision quality is bounded by a pre-defined context constructor」是主張,不是結果。

**步 2 — 把 context 建構器參數化(Eq. 4)**
- c_ϕ = Φ_ϕ(x, E) = ⟨e_1⟩ ⊕ … ⊕ ⟨e_K⟩,連續向量 token。保留不對稱:老師看得到 c_ϕ,學生只看 s_t。

**步 3 — Composer 的實作(§3.2,Eq. 5–7;App. A.1, A.3)**
- 檢索:經驗庫只存「成功 rollout」的任務描述 + 精簡 action–result trace;Qwen3-Embedding-8B(d_ret = 4,096)、FAISS IndexFlatIP、cosine top-J(Eq. 15–16)。文件端編碼 [x_i; τ_i],查詢端只編碼 x。
- 編碼:H_j = Enc_ψ(x, m_j) ∈ R^{T_j×d}(Eq. 5),凍結 backbone + LoRA(rank 8, α=16, dropout 0,每個 block 七個線性投影)。
- 壓縮:E_j = Comp_χ(H_j; Q_χ) ∈ R^{K×d}(Eq. 6),QFormer 式 cross-attention,8 層「共用參數」,FFN 倍率 4。
- 串接:c_ϕ = ⊕_{j=1..J} (e_{j,1} ⊕ … ⊕ e_{j,K}),老師 π^T_{θ̄,ϕ}(·|s, c_ϕ) = π_θ̄(·|[s; c_ϕ])(Eq. 7)。預設 J=3、K=32,共 96 個 latent token。
- 可訓練參數 ϕ = (ψ, χ)。老師 backbone θ̄ 固定。

**步 4 — Cold start(App. A.2,Eq. 14)**
- L_init(ϕ) = −E[log π_θ̄(y⋆ | s, c_ϕ(x, E))],y⋆ 是成功軌跡,E = Ret(x)。backbone 凍結,梯度穿過凍結層回到 latent 位置(論文自稱與 prefix-tuning 同機制)。得到 ϕ_0。
- AdamW、lr 10⁻⁵、batch 8、grad clip 3.0、n_ret=3。資料量與步數「依 domain 調整」,未給數字。

**步 5 — 蒸餾損失(§3.3,Eq. 8–10)**
- p^S_{t,n} = π^S_θ(·|s_t, a_{t,<n}),p^T_{t,n} = π^T_{θ̄,ϕ}(·|s_t, c_ϕ, a_{t,<n})(Eq. 8)。
- 老師 top-M + tail bucket ⊥(Eq. 9),M=20。支撐集 V^M_{t,n} 由「老師」決定,學生用同一個支撐集。
- L_distill = E_x E_τ [ Σ ω D_KL(p̃^S ‖ p̃^T) / Σ ω ](Eq. 10),ω ∈ {0,1}。Reverse KL(學生在前)。
- 關鍵:這裡沒有對老師做 stop-gradient。θ̄ 的權重不更新,但 activation 留在計算圖裡,所以 L_distill 同時對 θ 與 ϕ 有梯度。

**步 6 — 退化論述與 privileged margin(§3.3,Eq. 11–12)**
- 主張:只最小化 Eq. 10,composer 可以把 p^T 推向 p^S 來降低損失,得到沒資訊的 context。
- 對策 ❶ cold start;❷ 把 outcome reward 接進來:
  - δ_{t,n}(ϕ) = log π^T_{θ̄,ϕ}(a_{t,n}|s_t, c_ϕ, a_{t,<n}) − sg[log π^S_θ(a_{t,n}|s_t, a_{t,<n})](Eq. 11)。a_{t,n} 是學生取樣到的 token。sg 擋掉往 θ 的梯度,所以 δ 只訓練 ϕ。
  - A(τ) = 2r(τ) − 1 ∈ [−1, 1]。
  - ∆(ϕ) = E_τ[ Σ ω A(τ) δ_{t,n} / Σ ω ](Eq. 12)。
- 論文的排除論述(原文):context 退化 ⇒ π^T → π^S ⇒ δ → 0 ⇒ ∆ → 0 < m ⇒ dual penalty 啟動。這是唯一的「證明」,一行口語,沒有收斂性或可行性分析。

**步 7 — 完整目標與演算法(Eq. 13,Alg. 1)**
- min_{θ,ϕ} max_{β≥0} L_distill(θ,ϕ) + β(m − ∆(ϕ)) + λ‖c_ϕ − sg[c_{ϕ0}]‖²₂。
- β ← [β + η_β(m − ∆)]₊,β 初值 0。m=0.05,λ=0.2,η_β=0.5,lr_ϕ=10⁻⁵(Table 4)。
- 推論時只留學生:不需要經驗庫、檢索、composer、latent。

**步 8 — 實證(§4)**
- Table 1/2:十個「backbone × benchmark 彙總欄」LOPD 皆最高(Qwen3-4B 與 8B 各 3 個工具類彙總;Qwen3-4B 與 Olmo3-7B 各 2 個程式類彙總)。
- Fig. 3(EnvScaler,學生不帶 context 評測):Frozen ϕ_0 0.573;m=0 0.551;m=0.01 0.566;m=0.02 0.603;m=0.05 0.637;m=0.10 0.626;m=0.20 0.613。這是「joint optimization 有用且需要 margin」的唯一直接證據。
- Fig. 4(Qwen3-4B,EnvScaler):LOPD 在 320 generations 後超過 0.61,576 時到 0.637,之後停在 0.63–0.64;GRPO、Skill-SD 在 1,600 時為 0.611、0.588。
- Fig. 5(a):K=8/16/32/64/128 → 56.5/56.3/63.7/60.6/62.9。Fig. 5(b):n_ret=1 → 0.605,n_ret=3 → 0.637;其餘點文字未給數字。
- Table 3:Vanilla / Base+Composer / LOPD 的 reward 0.486 / 0.631 / 0.637;steps 11.12 / 16.31 / 17.04;tool calls/step 3.50 / 1.21 / 1.11;first-step length 9,937 / 6,695 / 6,210;reward/tool call 0.038 / 0.053 / 0.050;repeated tool calls 8.89 / 4.49 / 5.25。
- Fig. 6:把 LM head 套到 32 個 latent token 上,解出來是多語言與程式碎片的雜訊,讀不出程序。論文自己說這不能確立老師實際用了什麼資訊。

---

## 2. 必須教對的推導

### D1. Top-M + tail 的 reverse KL(Eq. 9–10)
- p̃(v) = p(v) for v ∈ V^M;p̃(⊥) = 1 − Σ_{u∈V^M} p(u)。老師、學生都用老師的 V^M。
- D_KL(p̃^S‖p̃^T) = Σ_{v∈V^M} p^S(v) log[p^S(v)/p^T(v)] + p̃^S(⊥) log[p̃^S(⊥)/p̃^T(⊥)]。
- 方向:學生在前。學生把質量放在老師機率低的地方會被重罰(mode-seeking);論文的說法是「student concentrates on teacher-supported behavior」。
- 〔顧問推導〕合併尾巴只會讓 KL 變小或不變(log-sum 不等式):D_KL(p̃^S‖p̃^T) ≤ D_KL(p^S‖p^T)。所以學生在老師 top-M 之外「怎麼分配」完全不受監督,只有總量受監督。
- 條件:M=20 是蒸餾用的截斷;取樣用的 top-k 也剛好是 20(Table 4),兩者是不同東西,別混。

### D2. 蒸餾梯度往哪裡推(退化的來源)〔顧問推導〕
設某 prefix 上學生 logits z、老師 logits u,p = softmax(z),q = softmax(u),KL = Σ p_v (log p_v − log q_v)。
- 對學生:∂KL/∂z_k = p_k[(log p_k − log q_k) − KL]。
- 對老師:∂KL/∂u_k = q_k − p_k。
- 讀法:對老師 logits 做梯度下降就是把 q 直接拉向 p。這就是論文說的「moving p^T toward p^S」的具體形式,而且這個方向對 ϕ 來說永遠存在,與 context 有沒有資訊無關。
- 條件:這是全詞表、老師 logits 可自由移動的理想化;實際上 u 只能透過 96 個 latent 向量與凍結網路改變,且損失是 top-M+tail 版本。兩式我已用有限差分數值驗證。

### D3. δ 的期望值是負的 KL;margin 與蒸餾的關係〔顧問推導〕
- 固定 prefix,對學生取樣取期望:E_{a∼p^S}[log p^T(a) − log p^S(a)] = −D_KL(p^S‖p^T) ≤ 0。
- 所以「不加權」的平均 δ 永遠 ≤ 0:老師在學生自己取樣的 token 上,平均而言給的機率不會比學生高。∆ 要成為正數,只能靠 A(τ) 與 token 的相關性。
- 令 Q(a) = E[A(τ) | prefix, a]、Q̄ = Σ_a p^S(a)Q(a)。該 prefix 對 ∆ 的貢獻 =
  Σ_a p^S(a) Q(a) log[p^T(a)/p^S(a)] = −Q̄·D_KL(p^S‖p^T) + Σ_a p^S(a)(Q(a) − Q̄) log[p^T(a)/p^S(a)]。
- 讀法:第一項在成功率過半(Q̄ > 0)時與 margin 作對,在成功率低於一半時反而獎勵 KL;第二項才是「老師比學生更偏好導向成功的 token」。∆ ≥ m 實際要求的是第二項夠大。
- 對照論文數字:EnvScaler 平均 reward 從 0.486 到 0.637,對應 Q̄ 的整體平均約 −0.028 到 +0.274(用 A = 2r − 1 換算,這個換算也是我的算術)。
- 連帶後果:學生蒸餾成功(p^S → p^T)本身也會讓 δ → 0、∆ → 0 < m。也就是 margin 要求老師「一直領先」,不只是排除 composer 偷懶。論文沒有討論這點,也沒有報告訓練中 ∆ 或 β 的軌跡。

### D4. Lagrangian 與 dual ascent(Eq. 13,Alg. 1 第 14 行)
- 原問題:min_{θ,ϕ} L_distill + λ·anchor,受限於 ∆(ϕ) ≥ m,即 g(ϕ) = m − ∆(ϕ) ≤ 0。
- Lagrangian:L + β·g,β ≥ 0;內層 max_β:若 g > 0 則 β → ∞(違反被無限懲罰),若 g < 0 則 β = 0。
- 投影梯度上升:β ← [β + η_β(m − ∆)]₊。∆ < m 時 β 增加,∆ > m 時 β 減少到 0 為止(互補鬆弛)。
- β 項對 ϕ 的梯度:−β ∇_ϕ ∆ = −β·E[A(τ) ∇_ϕ log π^T(a_{t,n}|·)]。〔顧問推導〕這是 reward 加權的 log-likelihood 梯度(REINFORCE 形式),樣本來自學生而非老師,沒有 importance weight。
- 〔顧問推導〕數量級:η_β=0.5、m=0.05,若 ∆ 停在 0,β 每步只增加 0.025;Fig. 4 的 1,600 generations 以每步 32 rollouts(App. B.1)計是 50 步,β 上限約 1.25。
- 條件:一階 primal–dual,非凸,沒有收斂保證;∆ 是 batch 估計(Alg. 1 第 12 行)。

### D5. 理想化的老師不動點(可做成互動玩具)〔顧問推導〕
在 D2 的理想化下(單一 prefix、老師 logits 自由、不計 anchor、不計 top-M),總損失對老師 logits 的期望梯度:
- ∂/∂u_k [KL − β∆] = (q_k − p_k) − β(p_k Q(a_k) − q_k Q̄)。
- 令其為 0:q*(a) = p(a)(1 + βQ(a)) / (1 + βQ̄),需要所有 1 + βQ(a) > 0。
- 讀法:β=0 時 q* = p(完全退化);β>0 時老師 = 學生分布依「該 token 的期望成敗」重新加權。Q 對所有 token 相同時 q* = p,不論 β 多大。
- 在 q* 處 ∆ = Σ p Q log(1+βQ) − Q̄ log(1+βQ̄) ≥ 0(f(Q)=Q log(1+βQ) 為凸,Jensen),等號成立若且唯若 Q 在 p 下為常數。
- 不動點梯度為零已數值驗證。這是我對機制的解釋,論文沒有這個式子,教學時必須標明是理想化模型,不是論文結果。

### D6. 梯度穿過凍結網路(App. A.2, A.4,Eq. 17–18)
- 輸入 = [t_1…t_a, pad×(J·K), t_{a+1}…t_L](Eq. 17);embedding 後 X[p_k] ← ⟨e_k⟩,k = 1…J·K(Eq. 18)。
- ∂L/∂ϕ = (∂L/∂X[p_k])·(∂e_k/∂ϕ)。第一個因子需要對凍結 backbone 做完整反向傳播:凍結的是權重,不是計算圖。
- 用 torch.cat 而非 in-place 賦值,autograd 才追得到。
- Encoder 的 LoRA 只在「編碼經驗」時啟用,主 forward(老師評估)時關閉,所以老師是純 base model + latent。

---

## 3. 常見誤讀、內部不一致、未說明之處

### 3.1 看起來對、但論文沒有證明的說法
1. **「老師與學生是同一個模型」**:LOPD 的老師是凍結的 base checkpoint θ̄,學生一更新兩者權重就不同(§3.3)。Eq. 2 的一般 OPSD 才是同權重 π_θ。老師 backbone 從頭到尾不進步,只有 context 在變。
2. **「學生學到比老師更強」/「老師很強」**:Table 3 的 Base+Composer(就是推論時的老師)是 0.631,學生 0.637。學生約等於老師。另一面:不訓練學生、只在推論時掛上 composer,就拿到幾乎全部增益;論文沒有討論這個部署選項。
3. **「joint optimization 讓老師變強」**:沒有量過。論文沒有報告 Base + cold-start composer(ϕ_0)的 reward,所以只知道學生 0.573 → 0.637,不知道老師本身有沒有變好。
4. **「retrieval alone is insufficient」(結論)**:沒有對應實驗。沒有「把檢索到的軌跡以純文字放進老師 context」的 baseline。Frozen ϕ_0(0.573)是「cold-start composer 不做 joint」,不是 retrieval-only。因此「latent 優於文字」沒有被隔離驗證。
5. **「低於 30% rollout budget 就超過 GRPO 與 Skill-SD」(摘要)**:Fig. 4 文字只給「320 generations 後超過 0.61」(320/1600 = 20%)與「576 時 0.637」(36%);GRPO 終值是 0.611。「超過 0.61」不等於「超過 0.611」,所以 30% 的說法只對 Skill-SD(0.588)明確成立,對 GRPO 要看圖上沒寫出數字的點。〔顧問推導〕
6. **rollout budget 不含前置成本**:經驗庫要先用 base model 在整個訓練集上 rollout 並篩成功者,composer 還要 cold-start SFT。這些 rollout 數、庫大小 |B|、cold-start 步數都沒給,也不在「1,600 generations」裡。每個 generation 的計算量也較高(3 次 encoder、多 96 token 的老師 forward + 反向)。〔顧問推導〕
7. **「LOPD 每一欄都最好」**:只有彙總欄。子欄反例(Table 1/2):Qwen3-4B BFCL Base 32.50 < GRPO/Skill-SD 33.00;Long Ctx 26.00 < OPSD 29.00;Miss Param 24.50 與 GRPO 並列;ACE M-Step 56.6 < GRPO 58.3;Olmo3-7B LCB v6 45.80 與 GRPO 並列。
8. **「Qwen3-8B 上優勢更大」(§4.2 原文)**:EnvScaler +1.9 → +6.2 成立;ACEBench +4.6 → +4.7 幾乎不變;BFCL-v3 +2.13 → +0.88 反而變小。〔顧問推導,差值由表算出〕
9. **「三個 backbone、七個 benchmark」**:沒有任何 backbone 跑滿七個。Qwen3-8B 只有工具類,Olmo3-7B 只有程式類,只有 Qwen3-4B 兩類都有。未說明原因。
10. **「latent 是可讀的 insights/tips」**:Fig. 1 畫了 insights/tips,但圖說寫明只用 trajectories;Fig. 6 顯示 latent 解碼是雜訊。
11. **「margin 保證老師有資訊」**:它只保證 reward 加權的 log-prob 差 ≥ m(見 D3),而且是軟約束(dual),不是硬保證。「structurally excluding the trivial solution」是口語論述。
12. **行為內化(Table 3)**:agentic system prompt 本來就寫「Avoid making parallel tool calls in one step」(App. A.5),而 vanilla 每步 3.50 次呼叫。行為改變也可以讀成「變得遵守 prompt」。〔顧問推導,詮釋〕

### 3.2 統計上的弱點(論文沒有 seed、沒有信賴區間)〔顧問推導,由表中百分比反推題數〕
- 程式類差距是 1–3 題:LCB v5 50.54 vs 50.18 = 141 vs 140 / 279;v6 45.04 vs 44.27 = 59 vs 58 / 131;HumanEval+ 87.80 vs 87.20 = 144 vs 143 / 164;Olmo EvalPlus 的 +0.55 = 425 vs 422 / 542。評測溫度 0.7(Table 5),有取樣雜訊。
- BFCL-v3 每子集 200 題,0.5 分 = 1 題;Qwen3-8B 彙總 29.88 vs 29.00 = 239 vs 232 / 800。
- ACEBench 只有 50 題(M-Step 20、M-Turn 30)。表中數值落在 1/60 與 1/90 的格點上(如 48.3 = 29/60、63.3 = 57/90),暗示重複 3 次或有部分給分,論文沒說。
- Fig. 5(a) 的 K=64 是 60.6,比 K=32 低 3.1、比 K=128 低 2.3,非單調;這個幅度可當作單次實驗雜訊的粗估,而許多主表差距小於它。
- K、n_ret、m 都是看 EnvScaler「測試集」的敏感度挑的(App. A.1 明寫 K 依 §4.3 決定);沒提 validation split。

### 3.3 內部不一致
1. **Table 1 vs Fig. 4**:Qwen3-4B EnvScaler,GRPO 在表中 61.8,Fig. 4 文字說終值 0.611;Skill-SD 表中 59.1,文字 0.588。LOPD 兩處一致(63.7 / 0.637)。未解釋(可能是最佳 checkpoint 對終值)。
2. **top-M 的出處**:§3.3 說「follow (Zhao et al., 2026)」(即 OPSD);§4.1 說「SDPO-style top-M logits plus a tail bucket」;App. B.1 說 OPSD 用全詞表 forward KL、SDPO 才是 reverse KL + top-K + tail。§3.3 的引用與後兩處衝突。
3. **Eq. 4 vs Eq. 7**:Eq. 4 的 c_ϕ 只有 K 個 token 且 K 定義為「每筆經驗的 token 數」;Eq. 7 是 J·K 個。
4. **J 與 n_ret**:同一個量兩個符號(§3.2/§4.1 用 J,附錄與 §4.3 用 n_ret)。
5. **「m=0 = 沒有 margin 約束」**:照 Eq. 13 字面,m=0 仍是約束 ∆ ≥ 0 且 β 會更新;論文把它稱為 without the margin constraint,且正文要求 m > 0。實際是關掉 β 還是 m 設 0,未說明。
6. **捨入不一致**:同一題數出現兩種寫法:140/164 寫成 85.37(Vanilla)與 85.36(OPSD);59/131 寫成 45.04 與 45.03;ACE 的 34/60 寫成 56.6(4B)與 56.7(8B)。EvalPlus 彙總 78.79 是由百分比加權算的,由題數算是 78.78。都是 0.01 級,不影響排名。
7. **Table 3 的 reward / tool call**:無法由同表其他列重現。0.486 /(11.12 × 3.50)= 0.0125,表列 0.038;LOPD 0.637 /(17.04 × 1.11)= 0.034,表列 0.050。定義未給(可能是逐軌跡比值的平均)。
8. **Eq. 12 vs Alg. 1 第 12 行**:前者是逐軌跡比值再取期望,後者寫「masked tokens 上的加權平均」,batch 內如何聚合不明。
9. **彙總欄的算法不統一(未明說)**:BFCL Avg 是四子集簡單平均;ACE Avg 是題數加權(20:30);LCB Avg 是題數加權(279:131);EvalPlus Avg 是題數加權(164:378)。我已全部驗算吻合。

### 3.4 未說明的超參數與設定
- 學生是全參數微調還是 LoRA;學生學習率(B.1 只說所有方法共用 AdamW lr 10⁻⁵、clip 1.0)。
- 總訓練步數(只有 Fig. 4 的 1,600 generations 可推;程式類完全沒給)、LOPD 每任務幾個 rollout、seed 數。
- 經驗庫大小、agentic 的「task-reward threshold」數值、cold-start 資料量與步數。
- **訓練時 Ret(x) 是否排除任務 x 自己的成功軌跡**:庫由訓練集成功 rollout 建成,查詢也來自訓練集,top-1 很可能就是同一題。Fig. 6 展示的都是「rank 2」。若未排除,latent 可攜帶同題解答,性質接近 oracle context。論文未說明;這是我的疑問,不是論文陳述。
- ω 遮罩是否涵蓋 thinking token(評測開 thinking mode)。
- 計算 p^S、p^T 時的溫度;anchor 範數是否對維度取平均;Q_χ 初始化 N(0, 1/√d) 的 1/√d 是標準差還是變異數。
- 沒有對 anchor(λ)、A(τ) 加權、cold start 做消融;Fig. 3 只掃 m。
- Baseline 的不對等:OPSD、SDFT 的示範池在覆蓋不足時由更強模型(DeepSeek-V4-Pro、Qwen3.7-Max)補;各 baseline 用各自原論文的 KL 方向與老師形式(SDFT、SDPO 是 EMA 老師,OPSD 凍結)。所以「只差 privileged context」(§4.1)不精確。

---

## 4. 哪些能在瀏覽器裡現場演示

### 可以(小型閉式或玩具計算)
1. **Top-M + tail 的 reverse KL**(D1):5–8 個 token 的兩個分布,拉桿調 M,顯示 KL 隨合併而下降;對照 forward / reverse 兩個方向的數值。
2. **退化動力學**(D2):單一 prefix,對老師 logits 做梯度下降 u ← u − η(q − p),看 q 收斂到 p。
3. **E[δ] = −KL**(D3):列舉所有 token 直接算,並展示 −Q̄·KL + 共變項的分解。
4. **Dual ascent**(D4):β ← [β + 0.5(0.05 − ∆)]₊ 的軌跡;讓使用者拖動 ∆。
5. **不動點 q* ∝ p(1 + βQ)**(D5):拉 β 與每個 token 的 Q,看老師分布如何偏離學生、∆ 是否過 m。必須標示為顧問的理想化模型。
6. **A = 2r − 1**:r ∈ [0,1] 對 A 的映射;r = 0.5 時該軌跡對 ∆ 的分子貢獻為 0。
7. **Cosine top-J 檢索**(Eq. 15–16):低維玩具向量。
8. **形狀記帳**:J × K = 3 × 32 = 96 個 placeholder 插在 prompt 中間(Eq. 17–18)。
9. **表格算術**:各彙總欄的加權方式、百分比反推題數(3.2 節)、Fig. 3 的七個數字長條圖、Fig. 5(a) 的五個數字。這些是重畫論文數字,不是重現實驗。

### 不可以
- Table 1/2、Fig. 3/4/5 的任何「結果」:需要 4B–8B 模型的 rollout、環境、訓練。
- Fig. 4 的曲線:論文文字只給 320、576、1,600 幾個點的數值,不能補畫中間。Fig. 5(b–e) 除 n_ret=1 與 3 外沒有文字數值。
- Fig. 6 的 LM-head 投影:需要真實 composer 與詞表。
- Table 3 的行為統計:需要真實 agent 軌跡。
- 「玩具 composer 學到有用 context」:任何小模型演示都是新實驗,結論不能歸給論文。

---

## 5. 先備知識(教學順序)

1. 語言模型的 next-token 分布:logits、softmax、log-prob、取樣(temperature、top-k、top-p)。
2. 多輪 agent:任務 x、觀察 o、動作 a(工具呼叫)、軌跡 τ、turn 與 token 兩層索引 (t, n)。
3. 以可驗證結果為獎勵的 RL(RLVR):獎勵稀疏、一條軌跡一個數;GRPO 只需當作「只用結果獎勵的 baseline」。
4. KL divergence;forward 與 reverse 的差別(mode-covering 對 mode-seeking)。
5. 知識蒸餾:學生比對老師的 token 分布,是每個 token 都有的密集訊號。
6. On-policy 對 off-policy:在學生自己走到的 prefix 上監督,減少訓練與推論的不匹配。
7. Self-distillation 與 privileged context:同一模型,老師多看一段資訊;推論時學生看不到。
8. Stop-gradient,以及「權重凍結」不等於「不在計算圖裡」。
9. Soft prompt / prefix tuning:連續向量當作輸入 token,梯度穿過凍結網路回到這些向量。
10. LoRA:凍結 backbone 加低秩可訓練增量。
11. Cross-attention 與 QFormer/Perceiver:固定數量的 learned queries 把變長序列壓成 K 個向量。
12. 稠密檢索:embedding、cosine similarity、top-J。
13. 受限最佳化:Lagrangian、dual variable、投影 dual ascent、互補鬆弛。
14. (讀 baseline 用)EMA 老師、PPO-clip、UCB1:知道名字與一句話用途即可。
