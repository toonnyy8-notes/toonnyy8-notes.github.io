# Fidelity review: design.md vs paper (No. 571)

Scripts: c1, chain, graph, tables all run and reproduce *_out.txt (graph.py prints less by default; seed-40 detail needs argv; saved output matches the acceptance table). No P0 found. Sorted by severity.

1. P1. 步驟 6 prior / LO8: "P_prior(R̂=k) ∝ T+1−k ... 偏好最大的 RTG". Paper App. A.3: "P_prior(R̂ = k) ∝ T + 1 − k ... favors shorter paths". Read literally with R̂ in [−T,0], T+1−R̂ is larger for more negative R̂, i.e. favours LONGER paths. k must be the path length (R̂ = −k), which the paper never states. Fix: write "k = 路徑步數（R̂ = −k，推定）" in the formula, step 6 and LO8; do not quote it bare.

2. P1. C3 "教的概念 ... 固定目標並遞減時，路徑長度跟著目標變（Sec. 1、5.2、App. A.3）". The graph example only uses the largest-return prior (Fig. 2 caption, A.3); no fixed-target/decrement experiment exists there, and Sec. 5.2/Fig. 4 is Atari/D4RL. The nearest-available-L' fallback (step 6) and the "steps 2,2,3,3,5,…" curve (loops pad the path, e.g. [6,1,15,6,12,0]) are the page's own. Fix: drop A.3/Sec. 1 from the citation, add the fallback rule and "loops pad length" to the 〔本頁簡化〕/推定 list, and label the fixed-target mode 〔本頁推導〕.

3. P1. 步驟 7 Table 1 "對 REM、QR-DQN、BC 全部持平或勝出". Table 1: Qbert BC 17.3 > DT 15.4 (±11.4, within noise); the paper's own caption says "outperforms other baselines in most games". Fix: say "Qbert 上 BC 略高於 DT（17.3 vs 15.4，差在標示的變異內）".

4. P1. 常見錯誤 3 "只預測動作，損失只在動作上" and LO6 "不預測狀態與 RTG". Sec. 3 says this for the main experiments, but Sec. 5.5 predicts return tokens and App. A.3 has GPT "generate both actions and return-to-go tokens" (the design's own C3 uses this). Fix: "主實驗只預測動作；Sec. 5.5、App. A.3 另預測 RTG".

5. P1. S10 title "獎勵延遲與長期信用分配時優勢最明顯". Paper: DT "outperforms the RL baselines" (CQL). Against %BC (Table 6: 69.9 vs 71.8 at 1K, 95.1 > 94.6 at 10K; Table 7: %BC 102.6/65.9/70.6 vs DT 107.3/60.7/78.5) DT is level or worse. Fix: "...明顯勝過 TD 方法（CQL）；與 %BC 相當".

6. P1. C1 顯示 "token 總數 3m" in 測試 mode. At test time a_t does not exist yet (Alg. 1 appends a after sampling), so the input has 3m−1 tokens. Teaching 3m with a "masked" a_t is training-time only. Fix: in 測試 mode show 3m−1 and label a_t "尚未產生".

7. P2. Alg. 1 vs 步驟 4/LO6 "s_t 位置預測 a_t". Sec. 3 text says the head on token s_t predicts a_t, but Alg. 1 comments "select hidden states for action prediction tokens / unstack(...).actions". Paper is internally ambiguous; design cites only the text. Fix: note the discrepancy once (S5) and say the design follows Sec. 3 text.

8. P2. 步驟 7 "3 個 seed 的平均與 ±（論文寫 mean and variance）" / R4. Sec. 4.1 text says "mean and standard deviation of 3 seeds"; Table 1/2/12/13 captions say "variance". Fix: record both statements, label ± as "未定（text: std; caption: variance）".

9. P2. 步驟 8 5.1 Table 3 averages (DT 56.1, CQL 43.5). Table 3 contains only Medium and Medium-Replay incl. Reacher (8 rows, no Medium-Expert), so it is not comparable with Table 2's 74.7/69.2. Fix: state the row set next to the averages; likewise LO11.

10. P2. 步驟 2 〔本頁推導〕 "V₁>0 所需次數 = H（任何 α>0 皆同）". True, but for α<1 V₁ is astronomically small (6.96e-10 at α=0.5, n=30), so the claim reads as "TD is just as fast". Also "α=0.5 達一半極限需 59 次" is for γ=1 (limit 1), while the paragraph's example is γ=0.99. Fix: state "V₁>0 但極小", and say γ=1 for the 59/119/297 numbers.

11. P2. C2 "γ=0.9,H=100 時 V₁ 極限只有 3.0×10⁻⁵ ... RTG 仍為 1". This compares a discounted value with an undiscounted behaviour-policy return; the paper only says discounting "can induce short-sighted behaviors". The page note already says RTG ≠ optimal value; add that the discount gap is a different objective, not slower propagation.

12. P2. 步驟 2 shows Q(s,a)←r+γ max Q while C2 uses a single-action V backup with α. Not paper content; the design should say C2 is policy evaluation on one action (the "單一動作" caveat is there but the S2 TD formula has max and no α).

13. P2. 步驟 3 〔本頁推導〕 "相差同一個常數 G−R̂₁^train". Correct and verified (c1), but only under "same reward sequence"; in a real test the rewards depend on the actions. The caveat sits only in the C1 figure caption; add it to the S3 body text.

14. P2. 步驟 1 S1 "目標是學到比資料更好的策略". Paper (Sec. 1, 2.1): "producing maximally effective behavior from fixed, limited experience"; "better than data" is the page's gloss (the paper supports it only via Fig. 4/Seaquest extrapolation, "sometimes"). Mark as 本頁說明.

15. P2. 步驟 6 結果 / R6 "資料中可能剛好有對應最短路徑的片段". Fine, but design never says Fig. 6's x-axis is 1–9 (no 10/∞ bin plotted although caption mentions ∞) while C3's comparison histogram uses 1–10 + ∞; add "與 Fig. 6 軸不同".

Checked and correct: Eq. 1, Eq. 2, Alg. 1 quotes, Tables 1–7 and 8–13 numbers, normalization Table 11, 15.8%, 1% = 500k/50M, Key-to-Door numbers, Sec. 5.7 conjecture wording, chain recurrence proof (verified by hand, base case and i=H case), token positions 3j−2/3j−1/3j.
