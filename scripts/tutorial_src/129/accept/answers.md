# 129 Goldfish 驗收答案（僅依 page.txt）

**LO1**
答：TIM 指 trainer（p）與 sampler（q）對同一段文字給出的機率不同，使 Eq. 1 不再精確成立。來源分三類：(1) 精度不同，或兩套獨立 codebase 各自的隱性 bug；(2) 浮點加法不可結合，兩邊以不同輸入形狀呼叫同一 kernel，reduction 順序不同（例：fp4_e2m1 下 (0.5+0.5)+2=3，但 0.5+(0.5+2)=2）；(3) staleness，非同步執行與 continuous batching 使 batch 或 rollout 片段來自不同 checkpoint。
段落：「TIM 的來源依嚴重程度可分三類」
信心：高

**LO2**
答：Eq. 2 以比值 r=p/q 重新加權：E_p[R∇log p] = E_q[(p/q)R∇log p]，序列層級時修正精確。代價是變異數：稀有 token 的比值可以任意大。實務方法截斷（clip）或遮罩（mask）比值，因此必然引入偏差。Table 1 例：TIS 在區間 [0, 2.0] 外 clip；MIS 在 [0.5, 5.0] 外 drop。
段落：「而這必然引入偏差」
信心：高

**LO3**
答：PG 等於 SFT 的條件：二元 reward 且 advantage 為 +1/0（未中心化或正規化），並且訓練完全離線（sampler 從不更新）。Fig. 2：離線只有 +1/0 穩定；線上在 TIM 下 +1/0 最不穩定，group-centered 最穩定。離線下哪一種 reward 最不穩定：頁面未提供（只說 0/−1 的 loss 無界、可能崩潰，但未在 Fig. 2 中排序）。group-centered 用哪種 reward：Fig. 2 未說明，依 §B.1 預設應為 ±1。作者的假設有兩個機制：離線在負 reward 下不穩定（本頁不探討）；線上在全正 reward 下不穩定，因為線上 +1/0 訓練是 trainer 擬合 sampler 的成功 rollout，teacher 隨 student 更新。
段落：「線上訓練在 TIM 下則以 +1/0 最不穩定」、「作者注意到，在二元 reward」
信心：中（離線最不穩定的 reward 頁面未提供）

**LO4**
答：常數 reward R 下，on-policy 時 E_p[s] = Σ_v p_v∇log p_v = Σ_v∇p_v = ∇1 = 0，所以期望梯度為 R·0 = 0。TIM 下 token 由 q 取樣、score 由 p 計算，E_q[s] = Σ_v q_v∇log p_v = s̄，一般不為 0。頁面 C1a 例（λ=0.5、reward 恆為 1）：‖PG‖ = ‖drift‖ = 0.1758。
段落：「常數 reward 下 on-policy 期望梯度是」
信心：高

**LO5**
答：Eq. 3：E_q[R s_yt]（PG 更新）= E_q[R]·s̄（drift）+ Cov_q(R, s_yt)（signal）。只有 signal 帶有 reward 的逐 token 資訊；drift 只依賴 prefix 的平均 reward E_q[R|prefix]。
段落：「只有 signal（共變異數項）看得到」
信心：高

**LO6**
答：drift 的方向是 s̄ = −∇_θ CE(q,p)（q 視為常數），也就是朝 sampler 蒸餾，強度為該 prefix 的期望 reward。朝固定 teacher 蒸餾無害，會收斂到該 teacher 的最大概似擬合。但 sampler 是 trainer 的有偏副本（量化或過時），而且每一步把權重同步回 sampler，誤差在回饋迴路中累積而不收斂。頁面的證據是文字論證，沒有定理或發散速率。
段落：「誤差在回饋迴路中累積而不收斂」
信心：高（內容明確；頁面自承無定理）

**LO7**
答：group centering 讓同一 prompt 的 advantage 總和為 0，但 drift 發生在單一 prefix。特定 prefix 的期望 advantage 為 E[A_i|prefix] = (1−1/G)(E_q[R|prefix] − E_q[R])，一般不為 0：通往正確答案的 prefix 為正，含錯誤的 prefix 為負。trainer 在前者之後被拉向 sampler、在後者之後被推離，與該 token 是否影響 reward 無關。頁面 C2 例（h甲=0.8、h乙=0.2、π甲=0.5、G=8）：係數為 +0.525 與 −0.525。Group centering 只縮小 drift，不消除。
段落：「group centering 只縮小 drift」
信心：高

**LO8**
答：s̃ = s_yt − s̄（Eq. 4），即減去 sampler 下的期望 score，因此 E_q[s̃] = 0。期望更新為 E_q[R s̃] = Cov_q(R, s_yt)（Eq. 5）。它不是無偏的 policy gradient：與 on-policy（精確 IS）的唯一差別是共變異數在 q 下量，而非 p 下量。沒有 TIM（p=q）時修正項為 0，SC 是 no-op。
段落：「它不是無偏的 policy gradient」
信心：高

**LO9**
答：IS 是乘法、隨機的修正（依賴被取樣的 token），會增加變異，截斷後又重新引入 drift。SC 是加法、在給定 prefix 下確定的修正，可精確計算，兩者可以疊加。與 value baseline：若 baseline 是精確的 V = E_q[R|prefix]，期望上也能消除 drift，但需要 critic；SC 不需要 critic 就得到同樣的期望更新。On-policy 時 reward baseline 只改變變異、不改變平均，SC 則是偏差修正。
段落：「SC 不需要 critic 就得到同樣的期望更新」
信心：高

**LO10**
答：Eq. 11：L = −R(log p_yt − Σ_v sg[q_v] log p_v)。sg（stop-gradient）讓 q_v 被當成常數，不對它微分，因此 −∇_θ L = R(s_yt − Σ_v q_v s_v) = R(s_yt − s̄) = R s̃_yt。注意被減的是這條 rollout 自己的 R 乘 s̄；它的期望 E_q[R]s̄ 才是 drift。
段落：「被當成常數，不對它微分」
信心：高

**LO11**
答：
- 只存 top-k：完整 logprob 每個 token 約 608KB（152K 詞彙，fp32）；1024 條、每條 32K 約 20TB。實驗用 k=128。
- ρ = (1−Σ_{v∈H} q_v)/(1−Σ_{v∈H} p_v)，即 sampler tail 質量除以 trainer tail 質量（Eq. 7）。重建 q̂：head 用 q_v，tail 用 ρp_v。
- 中心化項只需 head：因 E_p[s] = 0，tail 之和等於 0 減 head 之和，得 Σ_{v∈H}(q_v − ρp_v)s_v（Eq. 8）。
- 疊加 IS：α = ρ f(1/ρ)（Eq. 10）。f=1 時 α = ρ；精確 IS（f(r)=r）時 α = 1；TIS（f(r)=min(r,2)）時 α = ρ·min(1/ρ, 2)。
段落：「只記錄 sampler 的 top-k logprob」
信心：高

**LO12**
答：
- 30B training accuracy：FP8 W/A + FP8 KV 下 PG 58% 也穩定，與修正法相當。FP8 W/A + FP4 KV 下 PG 200 步內崩潰，MIS 後段崩潰，SC 52%、TIS 51%。INT8 W/A + INT4 KV 下 SC 30%、TIS 12%，其餘 <5%。
- Staleness（推論每 64 步同步一次，0.6B 測得）下最好的是 SC 疊加 TIS 或 MIS，且優於單獨 SC。頁面未說明 TIS+SC 與 MIS+SC 之間誰較好。作者解釋：SC 在 sampler 下量共變異數（Eq. 5），TIS 部分修正取樣分布，SC 再去掉剩下的 drift。
- 主要限制（§6）：SC 消除 drift，但剩下的更新在 sampler 而非 trainer 下量共變異數，嚴重 staleness 下有影響；主要結果來自刻意嚴重的不一致與短序列，作為溫和不一致下長期訓練的代理。另外只有 0.6B 與 30B 兩個模型，30B 多數曲線單 seed，疊加與 staleness 只在 0.6B 測過。
- 頁面指出論文結論句「嚴重量化下 SC 是唯一穩定的方法」與數字不符（FP4 KV 下 TIS 也穩定）。
段落：「SC 52%、TIS 51% 保持穩定」
信心：高（30B 數字）；中（staleness 的 TIS 與 MIS 排序頁面未提供）

**常見卡關點（最多 5 點）**
1. 符號多：p/q、s、s_yt、s̄、s̃、sg、H/T、ρ、α 散在各節，符號表在 §4a 才集中出現。
2. log-derivative trick 與「score = ∇log p」直接假設讀者已知，頁面只給延伸閱讀連結。
3. Eq. 3 的 E[AB] = E[A]E[B] + Cov(A,B) 與 Σ_v q_v∇p_v/p_v ≠ ∇Σ_v p_v 的推論很快帶過。
4. 逐 token 與序列層級的差別（prefix 的出現機率仍由 q 決定）不易掌握，而 IS 只有序列層級精確。
5. 正負號與目標慣例：Eq. 11 的 L 是要最小化，Eq. 13 的 J_base 是要最大化；互動元件的 λ、σ、k 與各種數值（drift、signal、Ent@100 等）沒有說明如何讀。
