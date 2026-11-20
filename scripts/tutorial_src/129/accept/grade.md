# 129 Goldfish 驗收評分

**Score: 12/12**

## Verdicts
- LO1 correct: TIM 定義、三類來源與 fp4 例子齊全。
- LO2 correct: 以 p/q 重加權、變異膨脹、clip/mask 引入偏差、Table 1 區間都有。
- LO3 correct (see note): 二元 reward +1/0 且完全離線、離線只有 +1/0 穩定（即負 reward 不穩定）、線上 +1/0 最不穩定、group-centered 最穩定、兩機制假設皆答對。
- LO4 correct: 推導與 TIM 下 E_q[s]=s̄ 正確。
- LO5 correct: Eq. 3 與 signal 帶 reward 資訊正確。
- LO6 correct: 方向、固定 teacher 無害、回饋迴路累積、無定理皆答對。
- LO7 correct: prefix 期望 advantage 與「只縮小」正確。
- LO8 correct: Eq. 4/5、非無偏、p=q 時 no-op 正確。
- LO9 correct: 乘法/加法、疊加、critic 差異正確。
- LO10 correct: Eq. 11、sg、-∇L 正確。
- LO11 correct: 608KB/20TB、ρ、Eq. 8、α=ρf(1/ρ) 正確。
- LO12 correct: 三級數字、SC+TIS/MIS 最好、解釋、限制與結論句修正都有。

## 「頁面未提供」檢查
- LO3「離線最不穩定的 reward」: 參考答案只要求「離線負 reward 不穩定、只有 +1/0 穩定」，不要求在負 reward 中排名。page.txt 第 135 行「離線訓練只有 +1/0 reward 穩定」與第 137 行「離線訓練在負 reward 下不穩定」已寫明。讀者其實答出，只是被題目的「最不穩定」措辭誤導。不扣分。
- LO12「TIS+SC vs MIS+SC」: 參考答案只要求「SC+TIS/MIS 最好且優於單獨 SC」，頁面第 490、505 行已述。參考答案不需此排序，不扣分。

## Fixes（皆為小幅，非必要）
1. LO3 句子（page.txt 第 135 行）：Fig. 2 圖說後加一句「離線時，+1/−1 與 group-centered 等含負值的 reward 都不穩定，只有 +1/0 穩定；圖說未在負 reward 之間排名。」避免讀者誤以為漏掉離線最不穩定者。
2. 卡關點 1（符號多）：在 S1 結尾加一行「符號速查：p=trainer、q=sampler、s=∇log p、s̄=E_q[s]」，並連到 §4a 符號表。
3. 卡關點 2（log-derivative trick）：Eq. 1 前加一行 ∇p = p∇log p，再接 E_p[R∇log p]。
4. 卡關點 3（Eq. 3 的 E[AB] 分解與 Σq∇p/p ≠ ∇Σp）：Eq. 3 下加一行 E[AB]=E[A]E[B]+Cov(A,B)，並明說 q_v∇log p_v 不能合併成 ∇Σ，所以 E_q[s]≠0。
5. 卡關點 4（逐 token vs 序列層級）：S2 加一句「prefix 出現機率仍由 q 決定，逐 token 比值不修正它，所以只有序列層級精確」。
6. 卡關點 5（慣例與互動元件）：在 Eq. 11 與 Eq. 13 旁標「L 最小化、J 最大化」，每個互動元件下加一行「λ、σ、k 代表什麼、該看哪個數字」。
