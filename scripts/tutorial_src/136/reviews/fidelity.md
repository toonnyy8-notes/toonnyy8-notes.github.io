# Fidelity review (Goldfish), tutorial 136

Verification: `python3 -I verify/check.py` ran, `all asserts passed`; every C1-C4 value matches design.md section 5. I also re-derived the Eq. 51/53/57 constants by hand and checked the Fig. 1 labels against page 5 of the PDF (mapping correct). No P0 found. Eq./Sec./App. numbers I spot-checked (Eq. 1-6, 9, 13-16, 19, 21, 24-28, 30-34, 42-46, 57-60, 69-70, 75-80, Prop. C.4, Table 2/3) are correct.

1. P1. Location: S11 mistake 7, "forward KL 的 pass@10 優勢與 on-policy 在 Countdown-4E 的泛化優勢都不會出現在同任務 pass@1". Paper: Sec. 5.3 says only forward-KL pass@10 vs pass@1 differs on Countdown-3; the 4E claim is "10 − 15% higher pass@k", and Fig. 5 (right) plots both pass@1 and pass@10. Sec. 5.2 says reverse KL at LR 1e-5 gains strongly from lambda>0 even in ID accuracy. Fix: delete the 4E half; restate as "pass@1 hides forward KL's pass@10 advantage on the ID task".

2. P1. Location: step 10, "截斷前範數在每個記錄步都超過 1.0". Paper: App. A.1 "exceeds our threshold of 1.0 during most training steps"; Sec. 6 "throughout training". Fix: "多數訓練步驟".

3. P1. Location: step 10 A.12, "作者結論：OnPD 與 OffPD 可達到的最佳表現不受生成溫度影響". Paper: this is the Fig. 25 caption. The text adds "generation temperature has a moderate effect on OnPD at the higher learning rate" (55.7 to 63.0) and says reverse KL "is more sensitive to the generation distribution" (OffPD 5.7 to 27.0, OnPD 59.3 to 36.0). Fix: add these qualifiers; the unqualified sentence overstates.

4. P2. Location: C2 forward 〔本頁推導〕 "forward 差 0.0346". The (1/2,1/2) vs (delta,1-delta) construction is the paper's worst case for reverse KL. It is one instance, not forward's worst case, and Theorem C.1 gives only an upper bound. R8 covers the cross-comparison but not this reading. Fix: in the C2 assumption line and body, say the forward curve is "同一反例下的值，不是 forward 的最壞情形".

5. P2. Location: LO7 answer, "C.1：‖Δ∇̄‖₂≤2√2B E_x[TV]". Paper: Eq. 26 must hold "for every prompt x, every completion y having positive probability under rho or rho', every token position n". TV is between completion-level distributions (Eq. 55). The LO answer never defines B or the assumption, and Sec. 5.1 loosely says "prefix distributions". Fix: put the Eq. 26 condition and "TV over complete outputs" in the LO7 paragraph.

6. P2. Location: step 2 source 1 and LO3, Prop. C.4 identity. Paper: App. C.5 requires fixed horizon H (EOS absorbing) and a summed loss; the token-averaged form is exactly 1/H times it. The design states the identity without the condition. Fix: add "固定長度 H".

7. P2. Location: S11 mistake 6, "否則不是 reverse KL 梯度的不偏估計 (App. A.2)". Paper: A.2 only defines the sg surrogate and never says the no-sg version is biased. The claim is true by the product rule but is the page's own derivation. Fix: tag 〔本頁推導〕 and drop the App. A.2 attribution.

8. P2. Location: step 10 A.10, "重新以 {0.3, 0.5, 0.7, 1.0} 取樣". Paper: the Science checkpoint-50 (seed 42) re-sampled "on the same 20 prompts". Fix: state 20 prompts and a single checkpoint (R5 should cover this).

9. P2. Location: step 8 Qwen, "使 forward KL 遺忘愈來愈嚴重". Paper: A.5 also says "generally greater forgetting under reverse-KL, although the latter is less monotonic". Fix: add the reverse half; note single seed 42.

10. P2. Location: step 4 last bullet and C4 text, "forward KL 下為 1 與 1 pp" and the 0/1/2 pp diffs. The labels are integers, so each diff is uncertain by about ±1, and R2's caveat is not in the step-4 bullet. Also the "學習率配對差最大" result depends on picking a 5x LR step against binary factors. Fix: put the rounding caveat in the bullet and say the comparison is level-dependent.

11. P2. Location: step 11, "評估｜學生溫度 0.5 … 每題 1 個回答". Paper: A.4/Fig. 5 caption: pass@k uses 10 responses per question, Countdown-3 at T=1.0, Countdown-4E at T=0.5 with a 4,096-token limit. Fix: add the exception row; step 9 mentions only the 4E temperature.

12. P2. Location: LO11, "OnPD＋reverse KL 幾乎不轉移教師的西班牙文風格（0.0%／0.1%）". Paper: one run per condition (the design says so in step 9, not in the LO answer); Sec. 5.3 says "may help suppress". D.7 labels the task "ScienceQA", while step 9 says Science. Fix: add the single-run and hedged wording to the LO paragraph.

13. P2. Location: LO12 Numina, "MATH-500 最高（59.0%）". Paper: A.3 reports the maximum over checkpoints along the trajectory, which "may be optimistic". Step 10 says this, but the LO12 reference answer, which must stand alone, omits it. Also temperature 0.7/top-p 0.8 differs from the main setup.

14. P2. Location: C1 default shape 0.4:0.3:0.2:0.1. For B-E the log ratio is identical, so in "學生漏掉 A" the whole reverse gradient is zero (‖g_R‖=0.0000), not just coordinate A. This is a limit stationary point of the toy and is stronger than the coordinate-wise statement in Sec. 5.1. Fix: the body should say it is produced by the shared-shape assumption.

15. P2. Location: C1 table "‖g_R‖₂ 與界 R (Eq. 65)". Paper: Eq. 63-65 bound the l1 norm by R, so the l2 norm is also bounded. This is fine, but the bound is loose (1.31 vs 4.60). Add "此界不緊" to avoid implying tightness.
