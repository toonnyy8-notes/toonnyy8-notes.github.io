# Goldfish 驗收結果（僅依頁面文字）

**LO1**
- 答：$p(y\mid c)=\prod_{r}p(y_{rB+1:(r+1)B}\mid c,y_{\le rB})$。連鎖律，沒有近似。block 之間照 AR 順序，平行只在 block 內（B 個 token 一起生成）。因為 block 之間維持 causal，前面的 token 與表示不會因後面的 block 改變，所以 prefix 的 KV cache 可沿用。
- 段落：「平行只發生在 block 內」
- 信心：high

**LO2**
- 答：$I_t^i=(1-\tau^i(t))x_0^i+\tau^i(t)x_1^i$，$x_0$ 是高斯噪聲，$x_1$ 是乾淨 block 嵌入。$\tau$ 要可微、遞增，$\tau(0)=0$、$\tau(1)=1$；附錄另要求 $\tau^i\in C^1$，且 $t<1$ 時 $\dot\tau^i>0$。$\tau^i(t)>\tau^j(t)$（$i<j$）表示較前的位置較接近資料，會先被還原。頁面強調這個順序是訓練學出的，論文沒有把它設成約束。
- 段落：「較前的位置比較接近資料，會先被還原」
- 信心：high

**LO3**
- 答：Eq. 5 為 $b_t^i=\frac{\dot\tau^i}{1-\tau^i}(D_t^i\E-x^i)$，$t\in[0,1)$。速度由 denoiser 的機率向量乘嵌入矩陣 $\E$ 得到期望嵌入，再減去目前狀態。兩個關鍵步驟：(1) 由 Eq. 2 解出 $x_0$，得 $x_1-x_0=(x_1-I_t)/(1-\tau)$，所以 $\dot I_t=\dot\tau(x_1-x_0)$ 只依賴乾淨嵌入與目前狀態；(2) $x_1=\E_{y}$，故 $\Ex[x_1\mid c,I_t=x]=D_t(x;c)\E$。$t\to1$ 時分母趨近 0，所以限定 $t<1$。
- 段落：「推導的關鍵有兩步」
- 信心：high

**LO4**
- 答：$K=1$ 時 Eq. 8 退化成一次 denoiser 計算（Eq. 10）：$p=\hat D_0(\hat x_0;c)$、$\hat x_1=p\E$、$\hat y=\arg\max p$。理想的 $D_0(x_0;c)^i=p(y_i\mid c)$，即各位置獨立的邊際，噪聲不帶資訊。C1 玩具：prefix 固定時 hot dog 0.4、ice cream 0.3、sour cream 0.3；位置 1 邊際 argmax 是 hot（0.4），位置 2 是 cream（0.6），所以一步得到機率為 0 的「hot cream」。頁面模擬中 K=1 的有支撐比例為 0.000、TV 為 1.000（示意值）。
- 段落：「Eq. 8 退化成一次 denoiser 計算」「一步得到機率為 0 的「hot cream」」
- 信心：high

**LO5**
- 答：prefix 的時間固定為 1（無噪聲，經 adaLN）；prefix 維持 causal 且看不到 noisy block，block 位置可看整個 prefix。原因：prefix 的 K/V 因此與 $(x,t)$ 無關，可跨去噪步與後續 block 重用，prefix 輸出仍是 AR 預測 $f(c)$。嵌入由預訓練模型初始化，與 prefix、flow 共用，與 denoiser 一起訓練，但每個嵌入的 L2 norm 固定為預訓練值，只學方向，每次更新後投影回去。原因：norm 不受約束時會相對固定的 $\sigma$ 膨脹，有效 corruption 降低，去噪太容易。Table 3：固定 norm 38.0、不受約束 36.9、凍結 34.4。
- 段落：「prefix 恆為時間 1（無噪聲）」「理由：不受約束時嵌入 norm 會相對固定的」
- 信心：high

**LO6**
- 答：Eq. 6 三項權重都是 1，共用 transformer 參數：(1) AR CE 監督 prefix 的下一 token 預測；(2) denoising CE 監督 Eq. 4 中 block 各位置的機率；(3) 嵌入平方誤差 $\|\hat D^i\E-\E_{y_i}\|^2$ 監督決定速度的期望嵌入（Eq. 5）。消融（0.5B、B=4、32 步、IFEval／GSM8K／HumanEval 平均）：只用 CE 33.6、只用 MSE 29.7、CE & MSE 35.8，加上 AR 為 38.0。作者認為這是經驗設計。
- 段落：「這是經驗設計」
- 信心：high

**LO7**
- 答：Eq. 22：$\tau^i(t):=F_i^{-1}(t)$，$F_i(\tau)=\frac{\int_0^\tau\ell^i}{\int_0^1\ell^i}$，即 $\ell^i$ 的正規化累積分布。$\ell^i(\tau)$ 是位置 $i$ 在 tokenwise time $\tau$ 的期望 denoising CE。由 $F_i(\tau^i(t))=t$ 微分（頁面本頁推導）得 $\dot\tau^i(t)=(\int_0^1\ell^i)/\ell^i(\tau^i(t))$，即 $\dot\tau\propto1/\ell$：loss 高處 $\tau$ 增加慢，取樣集中在難的噪聲程度。估計（Alg. 2）：矩陣 $L\in\mathbb R^{B\times200}$，均勻初始化；每 $S=100$ 步做四次無梯度 forward，每個 noisy block 各自抽均勻時間，用 EMA（$\beta=0.99$）更新 $L_{i,m}$；$F_i$ 取該列的分段線性 CDF。
- 段落：「Tokenwise time 是 loss 的反累積分布」
- 信心：high（$\dot\tau$ 關係為頁面推導，論文只給 $\tau\propto$ loss 的說法）

**LO8**
- 答：Self-conditioning：訓練時以 $p_{sc}=0.5$ 先做一次無梯度預測；推論時沿用前一步的預測，不增加 forward。改善：無 32.3、有 38.1（Table 3）。Prefix guidance：每步多一次不含 prefix 的 forward，只處理 B 個 block token，$g=8.0$（Eq. 27）。改善：無 34.3、有 38.1（Table 3）。
- 段落：「Table 3：無 self-conditioning 32.3、有 38.1」「Table 3：無 prefix guidance 34.3、有 38.1」
- 信心：high

**LO9**
- 答：Eq. 7：$X_{s,t}(x;c)^i=\frac{1-\tau^i(t)}{1-\tau^i(s)}x^i+\frac{\tau^i(t)-\tau^i(s)}{1-\tau^i(s)}\delta_{s,t}(x;c)^i\E$，$0\le s<t\le1$。學到的 $\delta$ 要等於真正 flow map，需滿足兩個條件：diagonal（$\delta_{s,s}=\hat D_s$）與 semigroup（Eq. 16：$\delta_{s,t}^i=\gamma^i\delta_{s,u}^i+(1-\gamma^i)\delta_{u,t}(X_{s,u})^i$，$\gamma^i$ 為 Eq. 16 所列）。一步能保留位置間關聯（頁面推導）：$t\to1$ 時 $\gamma\to0$，權重集中在軌跡終點，$\delta_{0,1}$ 等於 ODE 終點 token 的 one-hot，$X_{0,1}$ 一步得到與多步相同的聯合結果；而 $\hat D_0$ 只有邊際。注意：學到的一步 flow map 仍有差距（7B：一步 49.0，多步 flow 75.9）。
- 段落：「為什麼一步 flow map 能保留位置間關聯」
- 信心：medium（關鍵理由是頁面自行推導，論文未明說）

**LO10**
- 答：草稿：一步 Euler（Eq. 10）產生 B 個 token，不用 prefix guidance。驗證：$a=\max\{j:\hat y_{i+1}=\arg\max f(c\oplus\hat y)_{N+i}\ \forall 1\le i<j\}$，commit 與 greedy AR 一致的最長前綴；第一個 token 可免費驗證，不一致時改 commit $\arg\max f(c)_N$，故每次至少 commit 一個 token。輸出等於 greedy AR 的邏輯：每個被接受的 token 等於以已接受前綴為條件的 greedy AR 預測（歸納，頁面補上歸納步）。數值上仍可能因 bfloat16 等實作而有小差異。TPF $=\sum_j T_j/\sum_j F_j$，$T_j$ 為 commit 的 token 數，$F_j$ 為 fused forward 次數加 1 次初始草稿；AR 的 TPF 定義為 1。
- 段落：「與 greedy AR 預測一致的最長草稿前綴」
- 信心：high

**LO11**
- 答：2.54× 來自 7B、$B=16$ 的自我推測解碼，三項平均（IFEval、GSM8K、HumanEval）：準確率 74.5 vs AR 73.9，TPS 100.2 vs 39.5（單張 H100、batch size 1、bfloat16，Table 8）。2.54× 不是純 flow 生成的速度。純 flow decoding 沒有單一步數數字：$B=4$、32–64 步時，0.5B 接近 AR SFT（38.1、38.5 vs 38.6），7B 超過（75.2、75.9 vs 73.9）（Table 5）。
- 段落：「2.54× 來自 7B、B=16」
- 信心：high

**LO12**
- 答：Table 1（single-token，六項平均）：0.5B 28.7 → 31.6、7B 58.4 → 60.9、8B 56.8 → 63.1。注意：這是平均，部分項目下降（7B HumanEval 78.7 → 76.8；8B BBH 71.6 → 70.7、HumanEval 82.3 → 79.9）；8B 的提升主要來自 IFEval 與 GSM8K（Table 6），且 8B 由 Base 起訓；作者把它解釋為 multi-token prediction 效果（作者的解釋）；沒有 seed 或信賴區間。訓練用的 tokens：總體少於 300 億；Sec. 3 寫 5–15 epochs（9–27B tokens），Table 4 寫 5 epochs 約 9B，論文未說明兩者關係。
- 段落：「訓練資料少於 300 億 tokens」「這是平均值，部分項目下降」
- 信心：high（數字）/ medium（token 數，頁面本身不一致）

---

## 初學者可能卡住的地方

1. **上標 $i$ 與次方。** 頁面說明 $\tau^i$ 的上標是 block 內位置，不是次方，但 $\tau^i(t)$、$\tau^i$、$\dot\tau^i$ 很容易被讀錯。
2. **Eq. 5 的代數。** 從 Eq. 2 消去 $x_0$ 的那一步沒有寫在正文，只在附錄推導或頁面的「本頁推導」區塊。
3. **玩具數字 vs 論文數字。** C1、C3、C4、C5 的數值都標「示意值」，但版面與論文數字相近，容易混淆（例如 hot cream、TV、λ）。
4. **Eq. 8 與 Alg. 3。** Eq. 8 是 τ 上的精確步，不是 base time 的 Euler 步；Alg. 3 的 guidance 取未 guided 的 argmax，正文 Eq. 8 沒寫。
5. **「2.54×」與「流生成加速」。** 很容易誤以為 flow 本身快 2.54 倍；它來自自我推測解碼，且 B=16 的六項平均準確率 58.0 略低於 AR 的 58.4。此外，「輸出等於 greedy AR」是邏輯等價，數值上取決於實作。
