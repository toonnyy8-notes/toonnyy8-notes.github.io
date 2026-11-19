# Fidelity review (Goldfish) — 443 design.md

Verification scripts: c1.py, c2.py, c4.py and c3_run.py (args 256 0.03 600 2,5,10 0,0.1,0.3,1,3) were run; outputs are byte-identical to c1_out/c2_out/c3_out/c4_out and match the section-5 acceptance tables. c4's MISMATCH lines (Gym 87.9 vs 88.0, Adroit BRAC-v) are the documented R5 rounding issues. Table 1/2/3/4/5 numbers, Eq. 1-4, App. B-H facts and the Fig. 1 reward legend (checked on the rendered figure: top 3.0, left 0.0, right 1.5, bottom 5.0) all agree with the paper. No P0 found. Remaining items:

1. P1 - Cause of the N effect. Location: "C2b ... N=2 時 63.6% 落在中央空白區" / LO10 "N 越大越能還原資料分布". Paper (Sec. 4): "as N increases, the diffusion model becomes more expressive"; "N serves as a trade-off between policy expressiveness and computational cost". The toy uses an exact denoiser, so its N effect is purely Eq. 1 discretisation under the VP schedule (a^N is not exactly N(0,I) either: sqrt(abar_N)=0.08), not network capacity. A reader may take the toy as the paper's mechanism. Fix: one sentence in S4/S8: "玩具的 N 效應來自取樣步數與排程；論文只說 N 是表達力與成本的權衡，未分解原因。"

2. P1 - C3 cannot show why L_d matters. Location: "Q 不學習，用推定的平滑函數 ... 在資料外不會高估" and observation "(2) η 越大越集中". Only 4 logits are trainable and Q-tilde is exact, so large eta is free (0.98 at eta=3), which contradicts the paper's lesson that eta is domain-tuned (Adroit 0.1-0.15, Kitchen 0.005; App. E "more policy regularization"). Fix: state in S7 that the toy has no cost for large eta; point to Table 3.

3. P1 - Hedge dropped. Location: 1.1 "N=5 時推論比一步前饋策略慢 4–5 倍"; LO10. Paper App. G: "could be four to five times slower". Fix: "可能慢 4–5 倍（作者估計，未給量測條件）".

4. P2 - Scale invariance overstated. Location: "〔本頁推導〕...η 的意義與獎勵尺度無關". True only for positive multiplicative rescaling of Q; shifts of Q change E|Q|, and the paper still uses eta from 0.005 to 4.5. Fix: "對 Q 乘正常數不變；不保證跨資料集可共用 eta".

5. P2 - LO3 ambiguity. "原點密度與資料中心密度的比" Fix: "擬合高斯在原點的密度 ÷ 該高斯在資料中心的密度 = 2.70" (it is not the true-data density; the true density at the origin is ~0).

6. P2 - Toy property presented near paper claim. Location: C2 "對角高斯...範圍外 13.6%". Paper never says how BC-MLE handles the [-1,1]^2 bound, and its BC-MLE is a trained network (Sigma_theta(s)), not the closed-form MLE; Fig. 1 shows a blob roughly within +-2. Label "範圍外" and the closed-form MLE as 〔本頁推導／玩具〕, not Fig. 1.

7. P2 - max Q backup. Location: step 12 "是否用 max Q backup（只用於 AntMaze）". Table 3: antmaze-umaze-v0 = False (5 of 6 AntMaze tasks True). Fix: "AntMaze 中除 umaze 外都用".

8. P2 - Two meanings of "最佳基準". C4/LO11 mix "每任務最佳基準" bars with "平均最高的基準（CQL 77.6）". The mean of per-task bests is higher than 77.6. Fix: label the average row "平均最高的單一基準（CQL）".

9. P2 - Symbol clash. C3 "Q-tilde = sum_k r_k g_k / sum_k g_k" uses r_k for reward means, while 4.0 uses r_k for posterior responsibilities in the same section/code. Rename (e.g. RBAR_k).

10. P2 - N=2 phrasing. C3 observation (4) "呼應論文 Fig. 2「N=2 仍有少數動作落在無資料區」" while the toy puts 37% of samples off-peak. Paper says "a few". Say "方向相同、比例遠大於論文所述".

11. P2 - Step 6 (b). The "同一條連續噪聲路徑" statement is correct for abar only; beta_i = 1 - abar_i/abar_{i-1} are not points of one function. Say "abar_i 落在同一條曲線上".

12. P2 - 1.3 item 1: "去噪損失本身就是一種...正則化". Paper: "can be seen as" / "acts as a form of" (Intro, Sec. 3). Once Lq is added, Ld is only an approximate BC term. Soften to "可視為".

13. P2 - Step 13. "Diffusion-GAN" is attributed to the list of speed-up works; the list is paper-cited (Lu, Wang, Xiao, Salimans & Ho, Song, Zheng) without names in the paper text. Cite as "App. G 引用的加速取樣文獻" instead of naming methods the paper does not name.
