# Fidelity review, No. 569 (IQL) design.md

Verification scripts: common/numbers/maze/maze_noise/maze_pop/teleport re-run (without `-I`, since they import siblings). seed-1 outputs match the stored *_out.txt (extra seeds in the stored files were not re-run). No numeric discrepancies. Paper tables 1 and 2 and App. B/C numbers match design.md. The issues below are conceptual and are not covered by the scripts.

1. P1. C2 "單一損失" is not the paper's single loss. Design: "單一損失：expectile([5,10,0],[1-q,q*p,q*(1-p)],tau)" and LO5 "單一損失 7.2727". Paper Sec. 4.2: L=E[L2τ(r+γQ(s',a') − Qθ(s,a))], where Qθ(s,a) is a per-(s,a) fit. Its expectile is therefore taken over transitions for each action, not pooled over actions and transitions into one distribution. A paper-faithful naive Q(a2)=expectile_τ{10:p,0:1-p}, which is 7.94 at τ=.9, p=.3. It tends to 10 for any p>0, and that is the paper's "lucky sample" point. Fix: state that pooling is a toy simplification where Q depends on s only, or switch to per-action naive Q and take the state value as max or expectile over actions. Do not call 7.2727 "the" single-loss value.

2. P1. Fig. 3 claim: "Ant Maze 六個資料集上，τ=0.9 優於 0.7、0.7 優於 0.5" (步驟 9, LO11). Paper Sec. 5.2 says only "crucial to compute a larger expectile ... better performance". The rendered PDF p.8 Fig. 3 shows 4 panels (medium/large, play/diverse), not 6. In the medium panels τ=0.7 and 0.5 are both near 0, so a strict 0.7>0.5 ordering is not supported. Fix: "larger τ (0.9) is much better than smaller τ on the medium/large datasets (Fig. 3)"; drop "六個" and the strict chain.

3. P1. C4 (and 步驟 7 〔本頁推導〕) shows τ affecting the policy only through clipping, with Q(a) fixed. In the paper τ shapes the policy mainly by determining Q/V in TD learning (Eq. 5–6). With the fixed Q in the toy, "改變 τ 讀數完全不變 (不截斷)" can teach that τ is nearly irrelevant to extraction. Fix: say explicitly that Q is held fixed here, and that in real IQL τ acts through the learned Q. The chip "β=10 (Ant Maze)" with default τ=0.7 mismatches the paper's Ant Maze τ=0.9.

4. P1. C3 noise σ manufactures the Fig. 2 result. At σ=0 the τ=0.5 greedy policy is optimal (1.119; R5 admits this). Paper Fig. 2 caption/Sec. 5.1: one-step "values decay much faster... resulting policy is highly suboptimal" and IQL's values are "no longer dominated by noise". The paper does not say function-approximation error causes it. Fix: label the "被雜訊主導" reading as the page's interpretation. Do not put "Fig. 2 c–d 的定性結論" next to the σ result as if the paper showed it. Seed 4 (σ=0 gives 0.616 for all τ) shows the outcome is data-dependent.

5. P1. Theorem 3 / LO7 omit assumptions. Paper Sec. 4.2/4.4: results hold "under some assumptions" (Sec. 4.2), with "unlimited capacity and no sampling error" (Sec. 4, Eq. 3 discussion). Lemma 1 needs bounded support. The proof of Thm 3 is one line ("Follows from combining Lemma 1 and Corollary 2.1"), which glosses over exchanging τ→1 with the recursion. V_τ is defined for the population optimum of Eq. 5–6. Fix: add "在無限容量、無取樣誤差、有界獎勵下" to LO7 and S5 (the C3 finite-data τ=1>V* illustrates why).

6. P2. μ vs π_β. Design: "頁面說明兩者指同一個分布". Paper never defines μ (it appears in V_τ=E^τ_{a∼μ(·|s)}). Fix: label it as the page's reading ("本頁解讀").

7. P2. Table 2 CQL offline start. Paper App. C: the authors' CQL reproduction is worse offline ("particularly on medium and large antmaze"), so Table 2 CQL antmaze offline 151.5 differs from Table 1's 303.6. Design step 10/LO12 only says "CQL 的微調數字是作者自己重跑". Fix: say the Table 2 CQL offline start (151.5) is the authors' rerun and not comparable with Table 1 (303.6). R4's "應指 Table 1 的 CQL" is a guess; write "paper unclear".

8. P2. 1.1 "訓練 20 分鐘（CQL 80 分鐘）" omits that 80m is the authors' JAX reimplementation (the original is >4h, Sec. 5.2). The hero card 3 also lacks the "one GTX1080" qualifier (Sec. 6).

9. P2. LO1 reference answer: "這些值的錯誤沒有資料修正" is attributed to Sec. 3. Sec. 3 says only that OOD actions "can produce erroneous values ... often leading to overestimation as the policy is defined to maximize". "No data to correct" is the page's gloss. Fix: separate the two.

10. P2. -v2 note: "作者把所有方法都放在 -v2 locomotion 環境". Paper Sec. 5.2 says -v2 for all methods (not locomotion only), while Table 1 still labels Ant Maze rows "-v0". Quote the paper without the locomotion restriction, or note the inconsistency.

11. P2. R6 〔本頁推導〕 ("經驗轉移的抽樣誤差被最大值挑中"). The population mode (V=V* exactly) supports this winner's-curse explanation, but there is no test isolating it, so word it as "likely" (推測).

12. P2. Step 5 contraction 〔本頁推導〕 is correct (monotone and shift-equivariant gives Blackwell). It holds for finite S, bounded r, and fixed μ̂, so say so. Also state that in C3, G has a fixed value of 0.

13. P2. Step 8 table: the "微調 τ=0.8, β=3.0, offline 25000 步" row. App. C is ambiguous about whether 25000 offline steps applies to IQL or only AWAC ("We found AWAC to overfit... used 25000"). The next sentence treats it as shared. Keep it in the table but mark it as "dexterous manipulation 設定", without extending it to Ant Maze finetuning, whose τ/β the paper does not give.
