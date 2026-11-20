# 118 Fidelity review (Goldfish)

Scripts: `verify/check.py` re-run with `python3 -I`; output is identical to `verify/check.out` and matches the section 5 tables (C1 to C4 and the unit checks). I did not re-derive them. I also spot-checked these against paper.txt and found them correct: Eq. 1-7, 44.6/45.3/93.4/53.5, 6.4x, 2.5x, the Table 1 rows, Fig. 2/4/15/E.3 scores, 91.7/87.8, 93.4 -> 93.5, K=6/4, the >50% rule, the lambda defaults, and the Appendix B hyperparameters except the LR sweep (item 1).

1. P1. Quote: "GenRM LR | 掃描 {2e-6, 1e-6, 5e-7}" and LO12 "LR 2e-6（掃 {2e-6,1e-6,5e-7}）".
   Paper (App. B): "After doing a sweep of learning rates (LR), we find that an LR of [2e-6, 1e-6, 5e-7] works well ... (with LR=2e-6 generally being the best)". The set is the values that work well, not the sweep grid. The sweep grid is not given.
   Fix: write "掃描後 {2e-6,1e-6,5e-7} 表現良好，2e-6 通常最好" in both places.

2. P1. Quote: C1 observation (3) "μ = 1.5 時 RM accuracy 只有 77.3%，Best-of-16 卻有 84.7%：兩個指標不是同一件事".
   Paper (Sec. 4): both metrics are "correlated, RM accuracy only evaluates point-wise accuracy, Best-of-N evaluates ranking".
   The two numbers are not comparable. Best-of-N also depends on p and N: at N=1 it equals p=30%, which is below RM accuracy. The comparison implies BoN > RM acc, and that is not general.
   Fix: contrast the two quantities by definition (threshold vs ranking). Or show that at the same μ, BoN is 30% at N=1 and 84.7% at N=16.

3. P1. Quote: LO1 and step 1 "完美驗證器時 Best-of-N＝Pass@N＝1-(1-p)^N（p 為單份正確率…）".
   The paper uses Pass@N / oracle verifier as a dataset average (Fig. 10, 13).
   1-(1-p)^N holds per problem. With p as the dataset-average accuracy, real Pass@N is larger than the formula whenever per-problem p varies (e.g. some problems with p=0 and some with p=1). The same single-p assumption sits inside C1 and C3.
   Fix: say the formula is per problem and in the toy p is the same for every problem. Add a note that real Pass@N is a mean over problems.

4. P1. Quote: 1.1 "在固定生成器的 Best-of-N 評估中…5% → 45.3%、GSM8K 73% → 93.4%…、MMLU abstract algebra 37.9% → 53.5%".
   Paper: the abstract gives the same arrows with no start point. Table 1 labels 37.9 as "Base Model (Pass@1)". Fig. 1 says only "Δ represents the improvement ... using GenRM-CoT". Table 1 does not state N.
   Hero card 3 carries the "推定" label, but 1.1 does not. 37.9 -> 53.5 is not a verifier comparison; the like-for-like comparison against Disc-RM is 50% -> 53.5%. N for Table 1 is not specified (Fig. 7 uses Best-of-32).
   Fix: in 1.1 write "由 Pass@1（Table 1 明寫；其餘推定）". For Table 1 write "N 論文未於表中註明".

5. P2. Quote: "〔本頁推導〕…直接 GenRM 與判別式 RM 在數學上很接近".
   Paper (Sec. 3.1 / D.1): it only claims "comparable or slightly better".
   The equivalence holds only as an approximation for the correct-solution term and the incorrect-solution term. It also needs the other-token mass to be small, and the paper does not claim that. The parameters differ: a new cls head versus the shared LM head and the instruction I. "很接近" overstates the derivation.
   Fix: write "損失形式相同；是否近似取決於其他 token 質量，論文未量測".

6. P2. Quote: step 3 and C2 use p(Yes) = softmax with γ, plus the observation (3) "降低溫度把分數推向 0 或 1".
   Paper: γ is defined in Sec. 2 for generation. Eq. 4 never says the score uses γ ≠ 1.
   Fix: label the γ slider 〔示範值〕. Say that the paper does not tune γ for scoring (推定 γ=1).

7. P2. Quote: LO3 "只輸出一個數字…所以不能用 CoT 或與 SFT 統一".
   Paper (Sec. 5): Yang et al. (2024) train a reward model with an auxiliary generative SFT loss on a separate head. The paper's point is that this head is discarded and "no text is produced".
   "不能與 SFT 統一" is too strong.
   Fix: write "無法在同一個輸出頭上統一、也不能產生文字".

8. P2. Quote: LO7 "判別式 RM 對同一輸入只有一個確定分數，沒有可抽樣的推理路徑".
   Paper (Sec. 3.3): "which discriminative verifiers cannot do". This is the designer's rationale; the paper gives no reason.
   Fix: tag it 〔本頁說明〕 or attribute it to the designer.

9. P2. Quote: C3 first line "每條理由的 Yes 機率取 0 或 1" and "錯誤解的 Yes 票數為 K-Y".
   Eq. 7 averages continuous probabilities; Fig. 2 shows 0.0015, not a vote share.
   The model also assumes (a) the verifier is symmetric on correct and incorrect solutions and (b) the same h applies to both classes. Neither is stated in the assumptions line. The reasoning samples come from one model, so their errors are correlated, which the model ignores.
   Fix: add (a) and (b) to the assumptions line.

10. P2. Quote: Fig. 13 "有參考引導 91.7% vs 無引導 87.8%".
    Paper: the caption gives no metric. The figure shows Best-of-N with N up to 16, so I read it as Best-of-16.
    Fix: add "（Best-of-16，圖上讀出；Fig. 13 圖說未寫指標）".

11. P2. Quote: step 7 "Fig. 5…圖上另標演算法 1.5×、GSM8K 1.2×".
    Paper: only 6.4x is in the text, and its baseline is Disc RM at Best-of-32. The 1.5x and 1.2x appear as figure annotations; their baseline is not stated and could be self-consistency.
    Fix: drop them or write "基準論文未明寫".

12. P2. Quote: step 1 "論文寫兩者相關".
    Paper: "While these two metrics are correlated". OK, but the 〔本頁推導〕 "RM accuracy＝Φ(μ/2)" assumes a balanced test set. State this in C1.

13. P2. Quote: step 10 "GenRM-CoT 的增益依賴驗證理由的品質…K = 32 時推論成本為 32 次 CoT 生成".
    It is tagged as the designer's synthesis. The paper's own evidence is only Fig. 13/14. Mark "32 次" as 〔本頁推導〕, since K=32 is the default and the paper reports no cost.

No P0 found.
