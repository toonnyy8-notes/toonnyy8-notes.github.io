# 119 Goldfish 驗收答案

**LO1**
答：學生只看狀態 $s_t=(x,o_{\le t},a_{\lt t})$；老師是同一模型多看特權脈絡 $c_{fix}$（答案、軌跡、回饋或技能）。監督在學生取樣軌跡的每個前綴 $(t,n)$ 上計算（Eq. 3）。推論時用學生本身，不給脈絡。
引文：「推論時只用學生，不給脈絡」
信心：high

**LO2**
答：不是同一組權重。老師是凍結的 $\bar\theta$ 加 latent 脈絡 $c_\phi$（Eq. 7）。更新的是學生 $\theta$ 與 composer $\phi=(\psi,\chi)$（encoder LoRA 與 QFormer）。Eq. 10 不對老師加 stop-gradient，梯度穿過凍結網路回到 $c_\phi$，再到 $\phi$。encoder LoRA 在老師主 forward 時關閉。
引文：「Eq. 10 沒有對老師加 stop-gradient」
信心：high

**LO3**
答：(1) 檢索：Qwen3-Embedding-8B 以任務為查詢、「任務＋軌跡」為文件，FAISS 取 cosine 最高的 $J=3$ 筆。(2) 凍結 backbone＋LoRA（rank 8）把 $(x,m_j)$ 編成 $T_j$ 個隱藏向量（Eq. 5）。(3) 8 層 QFormer 式 cross-attention，以 $K=32$ 個學習 query 壓成 $K$ 個向量（Eq. 6）。(4) 串接成 $J\cdot K=96$ 個 latent token（Eq. 7）。(5) 在標記處插入 96 個 pad，embedding 後以 torch.cat 替換（Eq. 17–18）。數量為 96。Eq. 4 只寫 $K$ 個，與 Eq. 7 不一致，實作以 Eq. 7 為準。
引文：「串接成 96 個 latent token」
信心：high

**LO4**
答：比較學生與老師在每個前綴的下一 token 分布（Eq. 8、10），損失為 $D_{KL}(\tilde p^S\|\tilde p^T)$，即 reverse KL。支撐集由**老師**機率最高的 $M=20$ 個 token 決定（Eq. 9），學生用同一支撐集，其餘合併為尾部事件 ⊥。合併尾部的後果：截斷 KL 不大於完整 KL（log-sum 不等式，本頁推導）；增加 M 不會使截斷 KL 變小；尾部內學生怎麼分配機率不受監督，只有尾部總量受監督。
引文：「論文取老師機率最高的 M 個 token 為支撐集」
信心：high

**LO5**
答：梯度會穿過老師回到 composer，composer 可以把老師移向學生來降低損失，而不是讓學生變好。理想化下 $\partial D_{KL}(p\|q)/\partial u_k=q_k-p_k$，最小值在 $q=p$，此時脈絡沒有資訊，學生也沒有新訊號。這是作者說法加上本頁的理想化推導。
引文：「最小值 q=p 對應沒有資訊的脈絡」
信心：high

**LO6**
答：$\delta_{t,n}(\phi)=\log\pi^T(a_{t,n}\mid s_t,c_\phi,a_{t,\lt n})-\mathrm{sg}[\log\pi^S(a_{t,n}\mid s_t,a_{t,\lt n})]$（Eq. 11），$a_{t,n}$ 是學生取樣的 token。$A=2r-1\in[-1,1]$。$\Delta(\phi)=\mathbb E_\tau[\sum\omega A\delta/\sum\omega]$（Eq. 12）。若動作恰從 $p^S$ 取樣，$\mathbb E_{a\sim p^S}[\delta]=-D_{KL}(p^S\|p^T)\le0$，因為 KL 非負。條件：必須精確從 $p^S$ 取樣，且計算 $p^S$ 的溫度與取樣一致。論文取樣用溫度 0.7、top-p 0.95、top-k 20，計算 $p^S$ 時的溫度未說明，所以條件未必成立。單一批次可以為正，Δ 為正要靠共變項（老師給較高獎勵的 token 較高機率）。
引文：「期望上不為正」
信心：high（公式）；medium（成立條件）

**LO7**
答：Eq. 13 是拉格朗日形式：$\min_{\theta,\phi}\max_{\beta\ge0}\mathcal L_{distill}+\beta(m-\Delta)+\lambda\|c_\phi-\mathrm{sg}[c_{\phi_0}]\|_2^2$。最後一項是 anchor，把脈絡拉向冷啟動的 $c_{\phi_0}$。β 更新為 $\beta\leftarrow[\beta+\eta_\beta(m-\Delta)]_+$（Alg. 1 第 14 行）：Δ 低於 m 時增加，高於 m 時減少到 0 為止。預設：β 初值 0，m=0.05，$\eta_\beta$=0.5，λ=0.2，composer 學習率 $10^{-5}$。
引文：「預設 m=0.05 （§4.1）」
信心：high

**LO8**
答（本頁推導的理想化模型，非論文結果）：β 固定時老師收斂到 $q^*(a)=p(a)(1+\beta Q(a))/(1+\beta\bar Q)$，即學生分布依期望獎勵重新加權，需 $1+\beta Q>0$。β=0 時 $q^*=p$，老師就是學生，對應退化。Q 對所有 token 相同時，不論 β 多大都是 $q^*=p$。
引文：「三個特例：β = 0 時 q∗=p」
信心：high

**LO9**
答：Fig. 3 的七個數字（EnvScaler，學生不帶脈絡評測）：Frozen φ₀ 0.573；m=0 為 0.551；0.01 為 0.566；0.02 為 0.603；0.05 為 0.637；0.10 為 0.626；0.20 為 0.613。
量測：這七個數字與 m≤0.01 低於凍結 composer、m≥0.02 高於它的排序（每點單次訓練，論文未給變異）。
作者解釋（非量測）：無約束的蒸餾梯度讓 latent 脈絡退化；margin 讓 composer 接觸到初始化時沒有的訊號。
沒量：老師本身（base＋φ₀）的 reward、訓練中 Δ 與 β 的軌跡、多 seed 的變異。
引文：「沒量：老師本身」
信心：high

**LO10**
答：「10 組都最好」指 10 組 backbone×benchmark 的彙總欄（Table 1–2）中 LOPD 最高；子欄有反例，例如 Qwen3-4B BFCL Base 32.50 低於 33.00。程式類差距換成題數：LiveCodeBench v5 為 141 比 140（/279，差 1 題）；v6 為 59 比 58（/131，差 1 題）；HumanEval+ 為 144 比 143（/164，差 1 題）。LCB v5 單一分數的 SE 約 2.99 個百分點。還缺：seed 數、標準差或信賴區間，以及配對檢定所需的逐題結果；K、$n_{ret}$ 依測試集選定，m 的選法也未說明。
引文：「論文沒有 seed 數、標準差或信賴區間」
信心：high

**LO11**
答：Fig. 4 只有文字給的點（曲線無數值，未重畫）：LOPD 在 320 generations 超過 0.61；576 時為 0.637；576 到 1,600 為 0.63–0.64；GRPO 在 1,600 為 0.611；Skill-SD 在 1,600 為 0.588。
30% 的說法：1,600 的 30% 是 480。對 Skill-SD 明確成立（320 時已超過 0.61，高於 0.588）。對 GRPO 只知道 LOPD 超過 0.61，無法判斷是否在 480 前超過 0.611。頁面未說明預算是否計入建經驗庫與冷啟動的 rollouts。
與 Table 1 不一致：Table 1 的 GRPO 為 61.8、Skill-SD 為 59.1，Fig. 4 終值為 0.611、0.588；LOPD 兩處一致（63.7 與 0.637）。論文未解釋。
引文：「少於 30% rollout 預算」
信心：high

**LO12**
答：沒有實驗支持。論文沒有把檢索到的軌跡以文字放入老師的 baseline；Frozen φ₀ 是凍結的 latent composer，不是純檢索。
top-M 的出處前後矛盾：§3.3 寫「we follow (Zhao et al., 2026)」（OPSD），§4.1 寫 SDPO-style top-M，App. B.1 則說 OPSD 用全詞表 forward KL、SDPO 才是 top-K 加尾部。頁面無法確定出處是哪一篇。
引文：「we follow (Zhao et al., 2026)」
信心：high

**Beginner 可能卡住的地方**
1. 前幾節寫「老師與學生是同一個模型」，要到 06 才說老師是凍結的 $\bar\theta$ 加 latent，兩者權重不同。
2. Eq. 4 的 $K$ 與 Eq. 7 的 $J\cdot K$ 數量不一致；同一個量 $J$ 與 $n_{ret}$ 有兩種符號。
3. stop-gradient 位置：Eq. 3 的老師有 sg，Eq. 10 的老師沒有，Eq. 11 的學生項有 sg，很容易混淆。
4. 蒸餾的 $M=20$（老師的支撐集）與取樣的 top-k=20 是兩回事。
5. m=0 是關掉 β，還是 m 設為 0 仍有約束，論文沒有說明；Δ 的聚合方式在 Eq. 12 與 Alg. 1 之間也不一致。
