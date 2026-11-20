# 584 Goldfish 驗收答案

僅依據 page.txt 的頁面文字作答。

**LO1 | attention sink 指什麼？為什麼會出現？**
- 答：語言模型常把大量注意力權重放在第一個 token，與該 token 的語意無關（StreamingLLM 發現）。成因：softmax 強迫權重總和為 1。head 不需要完整分配時，剩下的權重分到第一個 token；它的 value 向量接近 0，所以不改變輸出。
- 引文：「softmax 強迫注意力權重總和為 1」
- 信心：high

**LO2 | Sink Attention 與 Gated Attention 各改了什麼？**
- 答：Sink Attention 在 softmax 分母加上一個可學習參數 sink，即 $\exp(\mathrm{sink})$（Eq. 1）。這個 sink 沒有 value，取代第一個 token 接收多出來的權重。Gated Attention 把每個 head 的輸出乘上 $G=\sigma(xW_\theta)$（Eq. 2），即每個 head 專屬的門控加上 sigmoid 激活。
- 引文：「在 softmax 的分母加上一個可學習的參數 sink」
- 信心：high

**LO3 | 什麼是 value drain？App. A 歸零實驗的結果？**
- 答：value drain 指第一個 token 的 value 向量 $\ell_2$ 範數接近 0。Fig. 2 中六個模型都符合，遠小於位置 1 到 16 的 token。App. A 把第一個 token 的 value 歸零：只對 $\alpha_{\text{sink}}>0.75$ 的 head 歸零時，Qwen3-8B 為 70.38（不介入為 70.36），Qwen3-14B 為 67.09（不介入為 66.91），幾乎不變。全部歸零時，Qwen3-8B 為 64.10，比不介入低 6.26；Qwen3-14B 為 65.90，比不介入低約 1.0。頁面另指出論文文字說「全部歸零只造成可忽略的下降」與表中數字不符。
- 引文：「Qwen3-8B 全部歸零的平均是 64.10」
- 信心：high

**LO4 | Vanilla 與 Sink Attention 的輸出如何寫成門控形式？$G$ 的 sigmoid 形式？**
- 答：兩者都寫成 $O=G\cdot\sum\tilde A v$，其中 $\tilde A=A/(1-A_{\text{sink}})$，$G=1-A_{\text{sink}}$（Eq. 3–4）。Sink Attention 是恆等式，因為 sink 的 value 恰為 0。Vanilla 只是近似，誤差為 $A_{0}v_0$，要求 $v_0\approx0$。
  - Sink 的 sigmoid 形式：$G=\sigma(\mathrm{LSE}_{\text{tok}}-\mathrm{sink})$（Eq. 23）。
  - Vanilla 的 sigmoid 形式：$G=\sigma(\mathrm{LSE}_{\neg0}-q^\top k_0/\sqrt{d_h})$（Eq. 16）。
- 引文：「Sink Attention 的 sigmoid 形式為」
- 信心：high

**LO5 | 注意力層中誰是專家、誰是路由器？隱式門控與 Gated Attention 共有哪兩個設計？**
- 答：多個 head 是獨立的專家；由 sink 形成的門控 $G=1-A_{\text{sink}}$ 是路由器。兩個共同設計：(1) head 專屬，每個 head 有自己的 sink 權重，門控彼此獨立；(2) sigmoid 形式。差別在於 Gated Attention 的門控由獨立投影 $W_\theta$ 算出（顯式），Vanilla 與 Sink 的門控藏在 softmax 裡（隱式）。
- 引文：「注意力層正好具備這兩者」
- 信心：high

**LO6 | Vanilla 模型的 query／key 分布有什麼限制？為何影響長脈絡？**
- 答：為了維持第一個 token 的高權重，後續 token 的 query 聚在 $k_0$ 附近，key 被推離 $k_0$。query 只能在很窄的角度範圍內變化，key 的有效維度也受限，點積的相對差異被壓縮，softmax 區分候選 token 的精度下降。在長脈絡中這會產生注意力雜訊，降低模型取回相關資訊的能力。Sink 與 Gated 的模型中 query、key 分散，沒有兩極化。這是作者以 Fig. 3 給出的定性解釋。
- 引文：「後續 token 的 query 聚在」
- 信心：medium（因果鏈是定性敘述，頁面未給數字證明）

**LO7 | $\mathrm{Imp}$ 與 Head Imbalance 如何定義？$\mathrm{Imp}=(0.9,0.1,0.1,0.1)$ 的 CV？**
- 答：$\mathrm{Imp}^{l,h}$ 是 batch 中所有 token 的平均門控值 $G$（Eq. 5）。Head Imbalance 是各層 CV 的平均，$\mathrm{CV}=\mathrm{std}/\mathrm{mean}$（Eq. 6）。例：平均 0.3，母體 std $=\sqrt{0.12}=0.3464$，所以 CV $=0.3464/0.3=1.1547$。頁面註明論文沒說 std 用母體或樣本版本，此處用母體版本。
- 引文：「各層的不平衡用變異係數」
- 信心：high（數值依母體 std；樣本 std 會不同）

**LO8 | Eq. 7 是什麼？$\lambda$ 取多少？上例的 $N\cdot\mathrm{CV}^2$？**
- 答：Eq. 7 是輔助損失 $\mathcal L_{\text{aux}}=\lambda\sum_l N_h[\mathrm{CV}]^2$，總損失為 $\mathcal L_{\text{base}}+\mathcal L_{\text{aux}}$。從頭訓練時 $\lambda=10^{-4}$（Sec. 5.1）。上例 $N\cdot\mathrm{CV}^2=2^2+3\times(2/3)^2=5.333$。
- 引文：「從頭訓練時 λ=10−4」
- 信心：high

**LO9 | 直接用 Eq. 7 微調會怎樣？Eq. 8 怎麼改？**
- 答：會出現 head pinning：主導 head 的門控幾乎不變，其餘 head 的門控被升高到與之相近，結果幾乎所有 head 都變高（Fig. 12 左）。Eq. 8 的做法：每層取重要度最高的 $m$ 個 head 為 shared heads，不放進損失；只對其餘 routed heads 計算 $(N_h-m)[\mathrm{CV}]^2$。微調時 $\lambda=10^{-2}$。頁面說論文沒有給 $m$ 的值。
- 引文：「直接用 Eq. 7 微調已訓練好的模型」
- 信心：high

**LO10 | 為何不能直接取注意力權重？作者如何得到 $G$？成本？Vanilla 由 kernel 的 LSE 算 $G$ 有什麼數值問題？**
- 答：Flash Attention 等融合 kernel 不輸出注意力權重；改用 eager 模式取出權重會犧牲效率。作者改用 kernel 前向計算已算好的 LSE。Sink：先跑標準 Flash Attention，再乘上 $G=\sigma(\mathrm{LSE}-\mathrm{sink})$，同一個 $G$ 也供輔助損失使用。Vanilla：$G=\sigma(\mathrm{LSE}_{\neg0}-q^\top k_0/\sqrt{d_h})$，但 kernel 的 LSE 含第一個 token，頁面未說明如何取得 $\mathrm{LSE}_{\neg0}$。成本：額外訓練延遲小於 2%。
  - 數值問題（頁面推導）：若用 $G=1-e^{z_{t,0}-\mathrm{LSE}}$ 相減，當 $A_0\to1$ 時會相消，LSE 本身四捨五入後相對精度流失。改用 expm1，或另外算 $\mathrm{LSE}_{\neg0}$ 即可避免。
  - Vanilla 的 $G$ 只用於重要度與損失，不可再乘到輸出上。
- 引文：「不輸出注意力權重」
- 信心：high

**LO11 | 2B 從頭訓練三種機制加上輔助損失後的平均分數？微調 LLaMA3.1-8B 的結果？**
- 答：2B 平均（base → base+aux）：Vanilla 45.46 → 46.83；Sink 46.35 → 46.50；Gated 46.30 → 47.70（Table 1）。LLaMA3.1-8B 微調：推理平均 56.50 → 58.97；LongBench 平均 40.51 → 43.98（Table 2）。頁面註明「Gated 在各規模最好」只在 base+aux 欄成立。
- 引文：「Gated 46.30→47.70」
- 信心：high

**LO12 | Sec. 6.1 的假說鏈為何？為何 Gated Attention 最早崩塌？**
- 答：假說鏈為 Value Drain → Attention Sink → Implicit Gating → Head Collapse（頁面明確標為假說）。Gated 的門控從訓練一開始就是顯式的，不需要先建立 sink，所以崩塌可以直接發生，像標準 MoE 的專家崩塌；這對應 Fig. 5 中最早、最嚴重的不平衡。Vanilla 要走完整條鏈，崩塌最晚才明顯。
- 引文：「Value Drain → Attention Sink」
- 信心：high（解釋為作者假說，頁面未另外證明）

## 讀者可能卡住的地方

1. **Vanilla 與 Sink 的差別**：Sink 的等式是精確成立，Vanilla 只是在 $v_0\approx0$ 時近似。初學者容易把 Eq. 3 當成兩者都成立。
2. **LSE 與 $\mathrm{LSE}_{\neg0}$ 的符號**：$\mathrm{LSE}_{\text{tok}}$、$\mathrm{LSE}_{\neg0}$、kernel 回傳的 LSE 三者不同，頁面沒有直接說明如何從 kernel 取得 $\mathrm{LSE}_{\neg0}$。
3. **std 的版本**：母體與樣本標準差會改變 CV 與損失值；頁面只說明它採母體版本，論文本身未說明。
4. **微調的 $m$ 與 $\mathcal T_m^l$**：論文沒給 $m$ 的值，也沒說明 top-m 是在微調前還是每個 batch 重選。
5. **λ 的兩個值**：從頭訓練 $10^{-4}$、微調 $10^{-2}$，容易混用。另外頁面指出 Fig. 5 的 CV 縱軸刻度不同，不能跨機制比較絕對值。
