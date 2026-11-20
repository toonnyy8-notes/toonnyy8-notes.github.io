## LO1

**答**：OffPD 設 $\rho=\pi_T$（教師產生軌跡）；OnPD 設 $\rho=\pi_S^\theta$（學生產生軌跡）。OnPD 的抽樣讓損失依賴 $\theta$，論文讓梯度不穿過抽樣、把抽到的軌跡視為固定（stop-gradient），得到的是半梯度（semi-gradient，Eq. 19）。當 $\rho$ 不依賴 $\theta$（如 OffPD）時，半梯度就是一般梯度。
**passage**：「讓梯度不穿過抽樣，把抽到的軌跡視為固定（stop-gradient）」
**confidence**：high

## LO2

**答**：forward KL 以教師機率 $\pi_T$ 加權，常稱 mode-covering（懲罰學生沒涵蓋教師給高機率的 token）。reverse KL 以學生機率 $\pi_S$ 加權，常稱 mode-seeking（懲罰學生把機率放在教師認為不太可能的 token）。
**passage**：「以教師機率加權，懲罰學生沒涵蓋教師給高機率的 token」
**confidence**：high

## LO3

**答**：兩個來源。(1) 教師 rollout 配 forward KL、學生 rollout 配 reverse KL，對應序列層級 KL 的兩種鏈式分解（Prop. C.4，條件是固定長度 $H$、token 損失加總），但只在目標值層級成立；因 stop-gradient，OnPD reverse-KL 的更新一般不是序列層級 reverse KL 的完整梯度。(2) 單一抽樣 token 估計 KL 時，$v\sim\pi_T$ 估計的是 forward KL，$v\sim\pi_S$ 估計的是 reverse KL，所以 rollout 自然配上對應方向。
可以拆開，是因為本文在全詞彙上算 KL，KL 方向不限制 rollout；而且這兩個來源都不代表慣例配對較容易最佳化或表現較好。因此 rollout 策略與 KL 方向是兩個獨立的設計選擇。
**passage**：「本文在全詞彙上算 KL，KL 方向不限制 rollout」
**confidence**：high

## LO4

**答**：獨立改變三個因素：rollout（OnPD／OffPD）、KL 方向（forward／reverse）、學習率（$1\times10^{-5}$／$5\times10^{-5}$）。
遺忘：7 個 OOD 基準平均分數在訓練前後的下降（百分點 pp）。
稀疏度：參數變化量絕對值小於 $\tau=10^{-6}$ 的參數比例，$S_\tau=\frac1P\sum_i\mathbb I[\lvert\theta_i^{\text{final}}-\theta_i^{\text{init}}\rvert<\tau]$。
**passage**：「7 個 OOD 基準平均分數在訓練前後的下降」
**confidence**：high

## LO5

**答**：
- 最佳平均準確率：OnPD 72%，OffPD 73%（頁面未逐格分列 rollout × 學習率的最佳值）。
- 圖讀值的 ID 準確率（LR 1e-5 / 5e-5）：OnPD·F 72/72，OnPD·Rev 72/35，OffPD·F 71/73，OffPD·Rev 35/49。
- 遺忘：只依學習率給範圍，低學習率時 OOD 平均最多變動 1.3 pp，高學習率時下降 11.2–14.0 pp。逐格（OnPD／OffPD）的遺忘數字：頁面未提供。
- 稀疏度（圖讀值，LR 1e-5 / 5e-5）：OnPD·F 85.3/51.9，OnPD·Rev 88.5/58.5，OffPD·F 88.8/59.5，OffPD·Rev 89.6/60.0。整體範圍為 LR 1e-5 時 85.3–89.6%，5e-5 時 51.9–60.0%。
**passage**：「低學習率時 OOD 平均最多變動 1.3 個百分點」
**confidence**：medium（準確率與稀疏度的逐格值是頁面的圖讀值；遺忘沒有逐格數字）

## LO6

**答**：
- forward KL 對學生 logit 的導數是 $\pi_S(v)-\pi_T(v)$，每個座標落在 $[-1,1]$，只要兩分布不同就非零。因此它在任何兩分布不一致的前綴都提供訊號。
- reverse KL 對學生 logit 的導數是 $\pi_S(v)\big(\log\frac{\pi_S(v)}{\pi_T(v)}-D_{\text{R-KL}}\big)$。
- 變小或消失：學生給某 token 的機率 $\pi_S(v)\to0$ 時，該座標消失，所以 reverse KL 難以找回學生遺漏的教師 mode。
- 變大：學生給某 token 可觀機率而教師機率趨近 0 時，log 比可任意大，導數可任意大。
**passage**：「學生機率趨近 0 時這個座標消失」
**confidence**：high

## LO7

**答**：
- Theorem C.1 上界（Eq. 27）：$\lVert\bar\nabla_\theta L_{\text{F-KL}}(\theta;\rho)-\bar\nabla_\theta L_{\text{F-KL}}(\theta;\rho')\rVert_2\le2\sqrt2\,B\,\mathbb E_x[\mathrm{TV}(\rho,\rho')]$，其中 $B$ 是 $J_\theta^\top$ 最大奇異值的上界。
- Theorem C.2 反例：更新差 $=\varepsilon\frac{\sqrt2}{8}\log\frac{1-\delta}{\delta}$（Eq. 57），$\delta\to0$ 時發散，所以不存在與分布無關的常數。
- 具體數字（頁面 C3 示範，$B=1$，$\delta=0.01$，$\varepsilon=0.1$，為頁面即時計算）：reverse 更新差 0.0812；forward 更新差 0.0346（同一反例，非 forward 最壞情形，界為 0.2828）。
**passage**：「reverse 更新差（Eq. 57）：0.0812」
**confidence**：high

## LO8

**答**：不能。頁面明確寫道：定理只控制一步的期望半梯度，上界依賴完整軌跡分布的距離與 Jacobian 上界 $B$；它不證明不同 rollout 會得到相似的模型或表現，也不解釋遺忘與稀疏度的差異。實驗結果只能解讀為與定理一致，不是它的直接推論。
**passage**：「這些定理不能推出「forward KL 下不同 rollout」」
**confidence**：high

## LO9

**答**：
- $\pi_{-1}=\pi_T$（OffPD）；$\pi_0$ 是兩者正規化後的幾何中點；$\pi_1=\pi_S$（OnPD）。
- $\lvert\lambda\rvert>1$ 時某一係數變負，會放大兩模型都給低機率、只是被偏好方給得相對較高的 token。防護：只對被偏好方機率至少為其最大機率 $\alpha=0.2$ 倍的 token（可信度遮罩）加 log 比加成，並將加成截斷在 $c=2\log3$。如此在 $\lambda\in[-2,2]$ 內，相對錨點的兩兩勝算最多放大 3 倍（頁面 Eq. 75–80）。
**passage**：「只對被偏好方機率至少為其最大機率」
**confidence**：high

## LO10

**答**：
- forward KL：在整個光譜上準確率都高於 80%；學習率 $1\times10^{-5}$ 時只變動 5.2 pp，穩定。
- reverse KL：平均表現隨 $\lambda$ 大幅變化，seed 間變異大。低學習率時受益於學生偏好的 rollout（$\lambda>0$），教師偏好的 rollout（$\lambda<0$）表現變差；高學習率仍不穩定，偶爾準確率驟降（collapse）。
- 遺忘與稀疏度：受 rollout 影響相對小，主要由學習率決定。
**passage**：「forward KL 的準確率在整個光譜上都高於 80%」
**confidence**：high

## LO11

**答**：
- on-policy 的優勢出現在：Countdown-4E（較難的 OOD 任務，直接評估 Countdown-3 checkpoint）上，兩種 KL 方向下 $\lambda>0$ 的 rollout 的 pass@k 一致高 10–15%（頁面未說明是 pp 或相對比例）。另外，OnPD 加 reverse KL 幾乎不轉移教師的西班牙文風格（0.0%／0.1%）。
- RLVR 之後：on-policy 的初始優勢沒有可靠保留；後續 300 步 RLVR 後，最強的持續表現來自 off-policy checkpoint。
**passage**：「一致高 10–15%」
**confidence**：high

## LO12

**答**：
- 抽樣 KL（App. A.2）：forward 仍穩健、reverse 仍敏感，學習率仍主導遺忘與稀疏度。
- 移除梯度截斷（App. A.1）：reverse KL 明顯變差，forward KL 在較大學習率仍有效（在 OffPD 尤其明顯）；整體沒有一致的 OnPD–OffPD 差距。
- 長推理（App. A.3，Numina–MATH，1,000 步）：學習率 $1\times10^{-5}$ 時 OOD 大致保持、稀疏度 92–93%；$5\times10^{-5}$ 時 OOD 由約 41.4% 降到 33.7–35.9%、稀疏度 63–70%。OnPD 加 reverse KL 在低學習率的 MATH-500 最高為 59.0%，但這是各 checkpoint 的最高值（作者稱可能偏樂觀，且每條件只訓練 1 次），作者視為提示性。結論仍是學習率主導。
**passage**：「學習率 1×10⁻⁵ 時 OOD 大致保持」
**confidence**：medium（頁面未用一句話直接總結 A.3 的結論，以上依數字與作者註記推得）

---

## 初學者可能卡住的地方

1. **Eq. 1 的 $\rho$ 與半梯度**：要先懂「rollout 決定在哪些前綴上比較」，又要懂 stop-gradient 為何讓 OnPD 的更新只是半梯度。頁面的說法跨了 Sec. 3、Eq. 19 與 App. C.5。
2. **KL 方向與 rollout 是兩個獨立選擇**：很容易把「OffPD 配 forward」當成必然配對。頁面花一整節解釋慣例配對的兩個來源，但其中的鏈式分解只在目標值層級成立，初學者難以分辨。
3. **圖讀值與示範值混在一起**：C1 的配對差、Fig. 1 的數字、C2–C4 的數字各有不同來源（論文圖標籤、頁面推導、示範假設）。C3 的 $\delta=0.01$、$\varepsilon=0.1$ 與 $B=1$ 是示範假設，不是論文實驗結果。
4. **Theorem C.1 / C.2 的符號**：$B$、TV、$\varepsilon$、$\delta$、$\Delta r$ 等符號多且定義分散在不同段落，還要理解「不存在與分布無關的常數」的含義。
5. **光譜 $\pi_\lambda$ 的公式與防護**：$\hat z_\lambda$、$\tilde z_\lambda$ 的分段定義、遮罩 $\alpha$ 與截斷 $c$ 的作用，以及「勝算被放大 3 倍」的來由，頁面需要多個 Eq. 才能拼起來。另外，頁面沒有逐格（OnPD／OffPD × 學習率）給出遺忘數字，只有學習率範圍，初學者容易誤以為有逐格資料。
