# Fidelity review, No. 444 (IDQL)

Verification: c1-c4 and tables.py re-run with `python3 -I`; all five outputs are byte-identical to `*_out.txt`. Acceptance tables in design.md section 5 match the outputs. The `jaxrl5/networks/diffusion.py` claims (VP, linear and cosine schedules, no noise at t=0, clipping) match the repo file I fetched. Findings below are about what the scripts do not cover.

1. P1. Location: S9 title "只允許一組超參數時，IDQL 的差距最大" and 步驟 10 "其他方法下降較多（照原文）".
   Paper: Sec. 6.1 says prior methods "suffer considerably more, particularly on the more challenging antmaze domain".
   Table 1 contradicts this for IQL. Antmaze -A to -1 change: IDQL 474.6 to 439.3 (-35), IQL 378.0 to 368.4 (-10), DQL -202, CQL -124. The antmaze gap to IQL shrinks from 96.6 (-A) to 70.9 (-1).
   Fix: retitle (for example "-1 協定下 DQL 與 CQL 掉得最多"). Print the per-method drops, and state that the claim holds for CQL and DQL but not IQL.

2. P1. Location: C2 readout "TV（對 π_imp）" and 讀者應觀察 (1).
   The AWR Gaussian targets µ·exp(αQ), and π_imp (expectile) has different weights, so TV 0.43 mixes two effects: unimodality and weight mismatch. Sec. 2 says the extraction's KL constraint "is not consistent with the policy the critic captures". Fig. 1 attributes the failure to unimodality.
   Fix: add a reference row, the best unimodal Gaussian fitted directly to π_imp (moment match). Or word the TV as "AWR 目標與 π_imp 的差距", not as proof of unimodal failure.

3. P2. Location: 步驟 10 "IQL-A locomotion、CQL-1、IQL-1、DQL-1 的印出總分與列和差 0.2–1.0".
   DQL-1 antmaze is 216.4 printed and 216.4 summed. `tables_out.txt` flags it as MISMATCH only because of float compare. Real mismatches: IQL-A loco (692.4 vs 692.6), CQL-1 ant (180.0 vs 179.0), IQL-1 ant (368.4 vs 367.6), IDQL-A ant (84.5/84.2 issue).
   Fix: drop DQL-1 from the list and fix tables.py to use a tolerance.

4. P2. Location: 步驟 10 and 步驟 11 table, "-A = 任意調參 / -1 = 只調一個".
   App. E gives τ = 0.7 (locomotion) and 0.9 (antmaze) for IDQL, the same as -1. So IDQL-A vs IDQL-1 differs in steps (1.5M/3M vs 1M/2M) and N (128/32 vs 64), not in τ. The paper's "degrades only slightly" is not evidence about tuning sensitivity.
   Fix: say so in S9. Also add App. D.2's caveat that -1 baselines are authors' reruns and "may not fairly represent each algorithm".

5. P2. Location: Hero card 3 and S9 title, "只允許一組超參數".
   Paper (Sec. 6.1, D.2): one hyperparameter per domain (two values total for IDQL). Page wording can read as a single global value.
   Fix: write "每個領域一組 τ".

6. P2. Location: 步驟 2 symbol table "f′：對 V(s) 的導數（Sec. 4.1）" together with proof line 1 "0=−E[f′(Q−V*)]".
   Paper writes f′ = ∂f/∂V(s), which equals −f′(Q−V) under the chain rule. The paper's own proof then uses f′ as the derivative with respect to the argument.
   Fix: define f′ as the derivative of f with respect to its argument u = Q−V, and add one line noting the paper's sign slip.

7. P2. Location: Theorem 4.1 statement, assumptions.
   Paper: "convex f with f′(0)=0". The proof also needs these, none stated: f differentiable (quantile fails, which you note); the weight at Q=V* defined (needs f″(0) finite; Eq. 9's 0/0 is measure-zero only for continuous Q); Z_imp finite and positive; V* exists. Quantile on a continuous density violates Z_imp finiteness.
   Fix: add a one-line "隱含假設" callout.

8. P2. Location: 步驟 4 / C1 "KL = α(V_exp − E_µ[Q])" and LO5 "相當於 KL 約束".
   App. C.2 computes D_KL(µ‖π_exp), the reverse of the KL(π‖µ) that an AWR constraint normally bounds. Sec. 4.1 says only "effectively applies a KL-divergence constraint".
   Fix: label the widget row "KL(µ‖π_awr)". Add 〔本頁推導〕: V_exp = E_awr[Q] − (1/α)·KL(π_awr‖µ). This reconciles the paper's "AWR gets the correct critic" with E_awr[Q] > V_exp.

9. P2. Location: C2 觀察 (3) "N = 256 時 15.35% 落在高群尾端的低密度區（連到 S7）".
   These are valid µ samples (the toy behavior model is exact), not outliers. The "low density" threshold of 1% of peak is arbitrary. Linking them to S7's outlier hazard teaches something the paper does not support.
   Fix: drop the S7 link, or call it "分布尾端（不是離群）".

10. P2. Location: 步驟 6 "有限 N 時分布介於 µ（N=1）與 π_imp（N→∞）之間".
    Only the endpoints are exact. "Between" is unproven. The monotone means in C2 are one toy.
    Fix: write "N=1 為 µ、N→∞ 趨近 π_imp"; also say Alg. 2's "correct implicit actor" holds only as N→∞ (the paper's Sec. 4.2 sentence is unqualified).

11. P2. Location: 步驟 4 Eq. 10 and Alg. 2 "算 w（Eq. 9、10、12）".
    Eq. 10 weights diverge at Q=V, so Alg. 2 with quantile is ill-posed on a continuous µ. This is the same point-mass issue, but the page does not link it to Alg. 2.
    Fix: one sentence in S5.

12. P2. Location: C4 "程式碼：每步都抽 z、只在 t>0 加噪、不截斷" and R6.
    Confirmed against the repo, but the repo also clips at every step when `clip_sampler=True`, not only at the end. The design says "把動作截斷到 [−1,1]" without saying per-step.
    Fix: state "每步（clip_sampler 時）與最終輸出".

13. P2. Location: 步驟 10 Table 3 summary "與 RLPD、Cal-QL 相當".
    Paper wording is "remains competitive". IDQL totals (559.1, 548.2) are below RLPD (571.0), and IDQL-Imp antmaze-umaze-diverse drops 78.4 to 73.0. RLPD's offline 0.0 is "no pretraining", not a score.
    Fix: add a footnote to S10.

14. P2. Location: 步驟 3 注意事項 (c) "（Sec. 4.1 末段）".
    The sentence about weight complexity is mid-section after Eq. 8, before the three examples, not the last paragraph.
    Fix: cite "Eq. 8 之後".

15. P2. Location: R3 "與 IQL 原文同一處正負號問題".
    I could not verify that the IQL paper has the same sign problem.
    Fix: confirm against No. 569, or drop the clause.

Checked, no issue: Eq. 1-13 transcription; App. B proof; App. C.1 derivation and V_exp; App. E values (τ, α, N, T, EMA, batch, dropout); hidden 256 and n=3 (App. G); Table 1/2/3 numbers and totals; the 〔本頁推導〕 claims E_awr[Q] > V_exp, VP ᾱ_T = exp(−5.05), linear ᾱ_T = 0.9506. All LO reference answers match the paper and `*_out.txt`.
