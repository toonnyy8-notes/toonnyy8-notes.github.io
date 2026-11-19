# 139 Value Forcing：Goldfish 驗收作答（僅依 page.txt）

**LO1**
答：$\overset d=$ 表示兩邊隨機變數「分佈相同」，不是逐點相等。$\mathcal T^\pi$ 在 max-Wasserstein 距離下是 γ-contraction（頁面引 Bellemare et al. 2017，不重證），所以有唯一不動點 $Z^\pi$。
依據：「表示分佈相等：左右兩邊的隨機變數分佈相同」
信心：高

**LO2**
答：Eq. 3 回歸輸入 $x^\tau=(1-\tau)x^0+\tau x^1$，目標 $x^1-x^0$，損失 $\mathbb E\|v(x^\tau\mid\tau)-(x^1-x^0)\|_2^2$，$x^0\sim\mathcal N(0,I)$，$\tau\sim\mathrm{Unif}[0,1]$。Lemma 1（非正式）：定義 $u(x\mid\tau)=\mathbb E_{\mathcal X}[\tfrac{d}{d\tau}X(\tau)\mid X(\tau)=x]$ 會生成 $X(\tau)$ 的邊際路徑；對 $\tfrac{d}{d\tau}X$ 做平方誤差回歸即學到 $u$。取 $X=(X^0,X^1)$ 即回到 Eq. 3。
依據：「回歸目標本身可以很吵，條件平均才是速度」
信心：高

**LO3**
答：
- FlowCritic：只在終點 τ=1 施加 Bellman（TD 目標 $y=r+\gamma z^1$，以 Eq. 3 擬合）；中間層不施加。
- Value Flows：在所有 τ 天真地套乾淨 Bellman 方程；論文稱 τ=0 監督不正確，初始層不恢復高斯。
- PCBF：兩條路徑共用起點雜訊 $z^0$ 沿耦合路徑施加；論文指出邊際路徑因此缺乏 Bellman 一致性。
依據：「但不會在初始層恢復高斯」
信心：高

**LO4**
答：Eq. 5：$Z^\pi(s,a,\tau)\overset d=\tau r(s,a)+\gamma Z^\pi(S',A',\tau)+(1-\tau)\sqrt{1-\gamma^2}\,N$，$N\sim\mathcal N(0,1)$ 與初始雜訊獨立。τ=0 時兩側都應是標準高斯，但 $\gamma Z'$ 變異數只有 $\gamma^2$；加上 $\sqrt{1-\gamma^2}N$ 後變異數為 $\gamma^2+(1-\gamma^2)=1$，所以修正項不可省。單步係數：γ=0.999 時 $\sqrt{1-\gamma^2}=0.0447$（頁面表：γ=0.9 為 0.4359，0.99 為 0.1411，0.9999 為 0.0141）。
依據：「修正項不可少，省略它」
信心：高

**LO5**
答：Remark 1（Eq. 6）：$W_p^{\sup}(\mathcal T^\pi_\tau Z_1,\mathcal T^\pi_\tau Z_2)\le\gamma\,W_p^{\sup}(Z_1,Z_2)$，係數為 γ。與 τ 無關：證明用同一個均勻變數的分位數耦合與同一個 N，$\tau r$ 與高斯項相消，只剩 $\gamma(C_1-C_2)$。前提（頁面）：固定策略 π（policy evaluation）、固定 τ、γ<1、p∈[1,∞)、回報屬一致有界動差集合、回報為一維；Banach 還需 $(\mathcal Z,W_p^{\sup})$ 完備，頁面寫「論文沒有驗證」。這是理想算子的敘述，不是神經網路訓練的收斂保證。
依據：「所以係數與 τ 無關」
信心：高

**LO6**
答：修正版的不動點為 $\tau Z^\pi(s,a)+(1-\tau)G$，$G\sim\mathcal N(0,1)$，平均 $\tau Q^\pi$，變異數 $\tau^2\mathrm{Var}Z^\pi+(1-\tau)^2$。若每層都用乾淨方程（頁面模型），不動點在每個 τ 都是 $Z^\pi$ 本身，與 τ=0 必須是標準高斯矛盾；兩者在 τ=1 重合。
C2 預設（γ=0.9、τ=0、第 60 次迭代、對 1）：修正版平均 −0.0237、標準差 0.9998；乾淨算子平均 −3.6924、標準差 1.3262（精確值 −3.6986、1.3255）；每步距離比值最大 0.899927 ≤ 0.9。C2 是頁面玩具（粒子近似，誤差約 0.01–0.05），不是論文實驗。
依據：「修正版收斂到平均約 0、標準差約 1」
信心：中（定性結論高；數字為玩具讀數）

**LO7**
答：回歸輸入點（在目前 (s,a)）：$\tau r+\gamma z^\tau+(1-\tau)\sqrt{1-\gamma^2}\,\epsilon$；目標：$r+\gamma u_V^\pi(z^\tau\mid s',\tau)-\sqrt{1-\gamma^2}\,\epsilon$，其中 $z^\tau=\phi_V^\pi(z^0\mid s',\tau)$，$z^0,\epsilon$ 為獨立 $\mathcal N(0,1)$。
易錯兩點（頁面標為本頁假設）：(1) 輸入與目標用同一個 ε，但符號相反（輸入 $+(1-\tau)\sqrt{1-\gamma^2}\epsilon$，目標 $-\sqrt{1-\gamma^2}\epsilon$）；(2) 輸入中的 r 乘 τ，目標中的 r 不乘。另有第三點：$u_V$ 在未變換的 $z^\tau$、下一狀態 s′ 求值。
依據：「輸入點與目標用同一個 ε，但符號相反」
信心：高

**LO8**
答：Eq. 12 讓 $v_V(z^\tau\mid s,\tau)$ 回歸 $v_Q(z^\tau\mid s,a,\tau)$，$z^\tau\sim p_Q$、$a\sim\pi$，無獎勵、無 ε。Eq. 13 是 Eq. 11 換成 $v_V$ 並縮放獎勵（輸入 $\tau(1-\gamma)r+\dots$，目標 $(1-\gamma)r+\gamma v_V-\sqrt{1-\gamma^2}\epsilon$）。真實速度未知時，讓學到的 $v_V$、$v_Q$ 互為目標，交替最小化，更新一方時固定另一方（類似 Fitted Q-Evaluation）。實作多了獎勵乘 $(1-\gamma)$。
依據：「互為目標，交替最小化，更新一方時固定另一方」
信心：高

**LO9**
答：VF 以抽樣挑選（Eq. 14）從 $\pi_\beta$ 抽 K 個候選動作，取 Q 最大者（K=16）。IVF 的權重 $w_\mu=\mu$（若 Q≥V）或 $1-\mu$（若 Q<V），只由純量 Q、V 決定；動作取自資料。
A.2 結論（前提：類似 Fitted Q 迭代、每輪精確最小化、k→∞ 收斂到自洽不動點，分析時獎勵不乘 (1−γ)）：V 是 Q 的 μ-expectile，μ→1 時趨近資料支撐內的最佳 Q。頁面說 Eq. 22、23 是假設而非結論，神經網路下的收斂仍未證。
不對速度殘差做 expectile：App. D.5 指出 μ→1 時得到逐點 $\max_a v_Q$，積分後不等於 $\max_a Q$；頁面只寫「≠」，其餘為本頁推導。
依據：「權重由純量 Q、V 決定」
信心：中（A.2 完整證明頁面未展開）

**LO10**
答：γ=0.999（OGBench）、0.9999（OCBench）；action chunk H=5；候選動作 K=16；BC flow 積分 10 步（VF 訓練時 4 步）；return flow 積分 10 步。M：VF 訓練 M=1，評估 M=16 且 10 步積分；IVF 固定 M=16。一步估計式：$Q(s,a)\approx z^0+v_Q(z^0\mid s,a,0)$（前提是速度學得準）。其他：Adam、學習率 3×10⁻⁴、500K 梯度步、batch 1024、目標網路 EMA 更新率 0.005。
依據：「訓練時 M = 1，用一步估計」
信心：高

**LO11**
答：Fig. 2（4 個狀態）：平均 1-Wasserstein 距離，越低越好。VF 在 4 個狀態都低於 FlowCritic、Value Flows、PCBF。相對 PCBF：puzzle-4x4 降 81%（22 對 118），humanoidmaze-giant 降 89%（13 對 122）。VF 對 Oracle：cube-triple 115 對 92、puzzle-4x4 22 對 20（接近）；cube-quadruple 179 對 57（與 PCBF 184 接近）；humanoidmaze-giant 13 低於 Oracle 25，故 Oracle 不是下界。
能說明：在這 4 個狀態上，VF 的回報分佈較接近評估回報。不能說明：只有 4 個狀態；「沒有任何消融單獨拿掉高斯修正項」，故不能證明修正項是改善原因。
依據：「沒有任何消融單獨拿掉高斯修正項」
信心：高

**LO12**
答：OGBench 20 任務總平均（Table 2）：
- 10%：VF 48 對最佳 explicit 基線 43（QC）；IVF 43 對最佳 implicit 基線 27（Implicit IQN）。
- 100%：VF 57 對 52（C51、QC）；IVF 65 對 66（Implicit C51），IVF 略低。
- 拿掉 reward scaling（4 任務、最後三次評估平均，Fig. 5）：VF 約 86%→11%，IVF 約 76%→11%；與 Table 5 協定不同，不能直接比較。
- 一步 actor：頁面原句「reduces success rates by 44%」，未說明是相對下降或百分點；頁面未提供 Fig. 4 的成功率數字，故不宜換算。
依據：「VF 總平均 48 對最佳 explicit 基線 43」
信心：高（表內數字）；低（44% 的定義）

## 驗收者會卡住的地方（最多 5 點）
1. Eq. 1 的 $\overset d=$ 與 Eq. 5 的 τ=0 修正：頁面沒有逐步寫出為何 $\gamma^2+(1-\gamma^2)=1$ 的推導鏈，需要自己補。
2. τ 的方向：τ=0 是雜訊、τ=1 是資料，與 RL 時間步 t 不同；Lemma 1 的 X、expectile 的 μ、noise level 的 τ 容易混淆。
3. Eq. 11 的符號與 τ 乘在哪裡（輸入 r 乘 τ、目標不乘），以及 $u_V$ 與 $v_Q$ 分別在哪個點、哪個 (s,a) 或 s′ 求值，抄寫很容易出錯。
4. VF 的 V 損失用策略動作（Eq. 12），IVF 用資料動作（Eq. 16）；Q 速度條件在 (s,a)、V 速度在 Q 損失中條件在 s′，多個符號需要對照。
5. 實作數字不全：Table 6 沒有 M 那一列；取樣 $z^\tau$ 要積分幾步頁面沒說；「44%」定義不明；Fig. 2 回報軸未縮放且 (1−γ) 換算未說明。
