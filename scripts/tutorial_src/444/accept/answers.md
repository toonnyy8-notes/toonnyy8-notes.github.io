# Goldfish 驗收答案（No. 444，僅依 page.txt）

**LO1 | IQL 的 V 與 Q 怎麼訓練？本文要回答什麼問題？**
- 答：V 用 expectile 損失 $L_V(\psi)=\mathbb E[L_2^\tau(Q_{\hat\theta}(s,a)-V_\psi(s))]$（Eq. 1）；Q 用平方損失 $L_Q(\theta)=\mathbb E[(r+\gamma V_\psi(s')-Q_\theta(s,a))^2]$（Eq. 2）。兩者都只用資料中的動作。本文要回答：Eq. 1–2 學到的 Q 對應哪一個策略。
- 段落：「學到的 Q 對應哪一個策略並不清楚」
- 信心：high

**LO2 | Eq. 6 與 Eq. 7 各是什麼？**
- 答：Eq. 6 是推廣後的 critic 更新 $V^*(s)=\arg\min_{V(s)}\mathbb E_{a\sim\mu}[f(Q(s,a)-V(s))]$，f 為任意凸損失；f 取 $L_2^\tau$ 時即 IQL。Eq. 7 是 $V(s)=\mathbb E_{a\sim\pi_{imp}}[Q(s,a)]$，定義隱含 actor $\pi_{imp}$：V 是某個策略下 Q 的平均。
- 段落：「V 是某個策略下 Q 的平均」
- 信心：high

**LO3 | 定理的條件與結論？證明的關鍵一步？**
- 條件：f 凸且 $f'(0)=0$（Theorem 4.1）。頁面另列隱含假設：f 可微、$V^*$ 存在、$Q=V^*$ 處權重有定義（或該處機率為 0）、$0<Z_{imp}<\infty$；quantile 損失不滿足第一與第四條。
- 結論：$V^*$ 也是 $\arg\min_V\mathbb E_{\pi_{imp}}[(Q(s,a)-V(s))^2]$ 的解，其中 $\pi_{imp}(a|s)\propto\mu(a|s)\,w(s,a)$，$w=\frac{|f'(Q-V^*)|}{|Q-V^*|}$（Eq. 8）。
- 關鍵一步：把 $f'(x)=|f'(x)|\cdot x/|x|$ 代入一階條件，得 $\mathbb E_{\pi_{imp}}[Q-V^*]=0$，故 $V^*=\mathbb E_{\pi_{imp}}[Q]$。
- 段落：「證明的關鍵一步是用」
- 信心：high

**LO4 | Eq. 9 的權重？τ=0.9 時上下兩側權重比？C1 預設的 V 與 $\mathbb E_{\pi_{imp}}[Q]$？**
- 答：Eq. 9 的權重 $w=|\tau-\mathbb 1(Q<V)|$：Q 高於 V 的動作權重為 τ，低於 V 的為 1−τ。τ=0.9 時比值為 9（τ=0.7 時為 2.33）。
- C1 預設（τ=0.9，頁面互動元件）：$V^*=2.5770$，$\mathbb E_{\pi_{imp}}[Q]=2.5770$，兩者相等即 Theorem 4.1。此為頁面的玩具數值（本頁推導），非論文結果。
- 段落：「τ = 0.9 時兩側權重比為 9」「expectile 的 V∗=2.5770」
- 信心：high

**LO5 | Eq. 10–12 的權重與 $V_{exp}$？$V_{exp}$ 與 AWR 的關係？App. C.2 算的是哪個方向的 KL？**
- Eq. 10（quantile）：$w=|\tau-\mathbb 1(Q<V)|/|Q-V|$。
- Eq. 11：$V_{exp}(s)=\frac1\alpha\log\sum_a\exp(\alpha Q(s,a)+\log\mu(a|s))$。Eq. 12：$w_{exp}=\frac{\alpha|\exp(\alpha(Q-V_{exp}))-1|}{|Q-V_{exp}|}$。
- 關係：$V_{exp}$ 是 AWR 策略 $\pi_{awr}\propto\exp(\alpha Q+\log\mu)$ 的正規化常數；$V_{exp}=\mathbb E_{\pi_{awr}}[Q]-\frac1\alpha D_{KL}(\pi_{awr}\Vert\mu)$。數字（α=1）：$V_{exp}=2.1545$，$\mathbb E_{\pi_{awr}}[Q]=2.3980$，差 0.2435。
- App. C.2 算的是 $D_{KL}(\mu\Vert\pi_{exp})=\mathbb E_\mu[\alpha(V-Q)]$，μ 在前；頁面另推導 $D_{KL}(\pi_{awr}\Vert\mu)$ 方向。
- 段落：「注意方向是 μ 在前」
- 信心：high

**LO6 | Fig. 2 中三者的值與隱含策略有何不同？Sec. 6.3 的實驗結論？**
- Fig. 2 值：expectile 隨 τ 平滑上升；quantile 對應累積分布函數；exponential 對齊最大值。隱含策略：expectile 覆蓋範圍廣；quantile 緊密集中；exponential 在最大值附近有覆蓋。頁面未給 Fig. 2 的具體數值。
- Sec. 6.3：expectile 與 quantile 搭配 argmax 抽取都表現好；exponential 不穩定且較差。隱含策略抽取不如 argmax（隱含策略有非零機率選到差的動作）。整體以 expectile 加貪婪抽取最強。
- 段落：「隱含策略抽取不如 argmax」
- 信心：medium（頁面對 quantile 的隱含策略形狀前後說法不一，Fig. 2 段落寫 "mean" 附近，與 Sec. 4.1 不同）

**LO7 | 兩個理由？**
- 理由一：隱含 actor 是行為策略的重新加權，可能多峰，單峰高斯近似不了，且解耦的 critic 不會隨 actor 調整。
- 理由二：表達力強的模型在重要性加權目標下，會提高所有訓練點的概似，所以 AWR 加權的擴散模型沒有幫助。
- 段落：「會提高所有訓練點的概似」
- 信心：high

**LO8 | Alg. 1 每步更新什麼？Alg. 2 怎麼選動作？評估時實際用什麼？**
- Alg. 1 每步：V 參數 ψ（Eq. 1/6）、Q 參數 θ（Eq. 2）、target 參數以 EMA η=0.005 更新、擴散參數 φ（Eq. 4，行為複製，與 critic 無關）。
- Alg. 2：抽 $a_i\sim\mu_\phi(a|s)$，i=1..N；算權重 w；$p_i=w(s,a_i)/\sum_j w(s,a_j)$ 後做類別抽樣。
- 評估實際用：Eq. 13，$\pi(s)=\arg\max_{a_i} Q(s,a_i)$（取 Q 最大的候選），因為表現較好。
- 段落：「每步更新 V 參數 ψ、Q 參數 θ」「直接取 Q 最大的候選在評估時表現較好」
- 信心：high

**LO9 | Eq. 4 預測什麼？T 與排程？為什麼用 LN_Resnet？**
- Eq. 4 預測加入的噪聲 ε：$L_\mu(\phi)=\mathbb E[\lVert\epsilon-\mu_\phi(\sqrt{\bar\alpha_t}a+\sqrt{1-\bar\alpha_t}\epsilon,s,t)\rVert]$（頁面寫未平方；DDPM 為平方）。
- T=5，VP 排程（D4RL）。VP 公式（官方程式碼，非論文文字）：$\beta_t=1-\exp(-\frac{b_{min}}T-\frac{(b_{max}-b_{min})(2t-1)}{2T^2})$，$b_{min}=0.1$，$b_{max}=10$。取樣 Eq. 5 從 $a_T\sim\mathcal N(0,I)$ 逐步去噪。
- LN_Resnet 理由：MLP 會產生很多分布外樣本，critic 可能給它們錯誤的高 Q 值，所以用容量大、正則化好的網路（hidden 256、3 個 block、dropout 0.1）。頁面未明說為何選 layer norm，僅寫 layer norm 有小幅好處。
- 段落：「擴散模型預測加入的噪聲 ε」
- 信心：high（公式）；medium（LN 理由）

**LO10 | Sec. 6.4 的結果？C3 預設（ε = 0.01）下 N = 8 與 N = 256 的期望真實獎勵？**
- Sec. 6.4：LN_Resnet 在 locomotion 上 N 越大越好，在 antmaze 上影響小；MLP 則 N 越大越差，antmaze 尤其明顯。
- C3（頁面玩具，非 Fig. 4 重現）：ε=0.01 時 N=8 的期望真實獎勵最高為 2.6880，N=256 掉到 0.2552。選到離群樣本的機率 $1-(1-\varepsilon)^N$ 由 0.0773 升到 0.9236。ε=0 時 E[r] 由 1.9000 升到 3.3382。
- 段落：「N = 8 的期望真實獎勵最高為 2.6880」
- 信心：high

**LO11 | Table 1 的 total（-A 與 -1）與訓練時間？-1 是什麼協定？antmaze 的 +70 指什麼？**
- total：-A 協定下 IDQL 1213.8、DQL 1209.7、IQL 1070.4、CQL 1002.1；-1 協定下 IDQL 1150.9、IQL 997.9、DQL 957.4、CQL 854.3。
- 訓練時間：IDQL 60 分鐘（-A）／40 分鐘（-1）；DQL 240 分鐘；IQL 20 分鐘。
- -A：論文回報的數字，允許任意調參。-1：locomotion、antmaze 各只能選一個主要超參數（IDQL 分別為 τ=0.7、0.9）。
- +70：-1 協定下 antmaze IDQL 439.3 對 IQL 368.4，差約 70。
- 段落：「每個領域（locomotion、antmaze）只能選一個主要超參數值」「antmaze 上比最佳方法（IQL）高 +70 分」
- 信心：high

**LO12 | 兩種微調做法與 Table 3 總分？論文寫的限制？**
- IDQL-Max：凍結擴散模型，用 argmax 動作探索，只微調 Q 與 V。IDQL-Imp：用隱含策略探索，三個網路都微調。
- Table 3（offline → online）：IDQL-Max 459.9 → 559.1；IDQL-Imp 462 → 548.2；IQL 408.2 → 473.7；RLPD 0.0 → 571.0；Cal-QL 只有四個任務。
- 限制：小維度動作空間（如 locomotion）有過擬合，dropout 可緩解但損及其他環境；Adroit 上 online 微調效果不好，作者推測過度預訓練有害，且 Adroit 需要大量探索。
- 段落：「IDQL-Max 459.9 → 559.1」「Adroit 上 online 微調效果不好」
- 信心：high

---

**驗收時容易卡住的地方（最多 5 點）**
1. 符號很多：$\pi_{imp}$、$\pi_{awr}$、$\pi_{exp}$、μ、$V^*$、$V_{exp}$ 容易混。Eq. 8 在頁面出現兩次。
2. 數字來源混淆：2.5770、2.6880、0.2552 等來自頁面互動玩具（標示「推定」或「本頁推導」），不是論文結果。
3. 原文寫法與程式碼不一致：Eq. 3 寫 minimizing 但實為最大化；Eq. 4 未平方；Eq. 5 在 t=1 是否加噪聲，程式碼與原式不同。
4. -A 與 -1 的差別與「+70」：頁面說 -1 的 IDQL 只小幅下降，但其步數與 N 也改變，不能當成調參敏感度的證據。
5. Eq. 9 的常數 2 被省略，以及 $|\tau-\mathbb 1(Q<V)|$ 的權重方向（Q 高於 V 取 τ）需要對照讀。
