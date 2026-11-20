Verify scripts (core, topk, accept w1/w2/kap) run and reproduce the design.md tables; the cited Table 4/9/13/14, FID and Table 2 numbers match paper.txt. Findings below are what the scripts do not cover.

1. P0 | 步驟 7: "rescale 在 α_t≥1-α_s 時 κ=1-η/(1-α_s)、否則 κ=1-η/α_t" | Suppl. A.5 (28) σ=η·σmax, σmax=min{1,(1-α_s)/α_t}; A.4 κ=1-σ/(1-α_s) | The cases are swapped. If α_t≥1-α_s then σmax=(1-α_s)/α_t and κ=1-η/α_t; otherwise σmax=1 and κ=1-η/(1-α_s). The 5.3 table (κ(0.5)=0.9000, κ(t=1)=0.9496) already follows the correct form, so the prose contradicts the table. Swap the two cases.

2. P1 | 步驟 11 item 1 "近似只在下一步的 normalizer"; LO11 "只有 Z̃ 是近似" | Sec. 4 eq (13) uses "≈", keeping only k of K embeddings | Truncating the weighted sum to the top-k rows is itself an approximation. The weights e^{K_i/τ}/Z̃ do not sum to 1, and the dropped mass is discarded. Values and indices are exact in distribution; (13) and Z̃ are both approximate. Fix LO11, 步驟 11 and S8/S9.

3. P1 | LO11 / 步驟 11: "U^(1)=V^{1/K}… K-i 個 U[0,U^(i)]" | Algo. 1: N=K-1; the clean coordinate is drawn separately as w̃ | The order statistics are over the K-1 zero-mean coordinates, so U^(1)=V^{1/(K-1)} and the next draw is the max of K-1-i uniforms. The reference answer taught as written is off by one. State N=K-1 and that the clean entry is drawn separately.

4. P1 | Hero card 2 "保持前向過程每個時刻的分佈"; 1.1 | Sec. 3 / A.2 prove marginals only for (11), which uses the true x | 步驟 5 of design.md itself says the guarantee is lost after x→x_θ in (12). Hero card 2 contradicts this. Say "真 posterior (11) 保持邊際；可取樣的 (12) 無此保證".

5. P1 | Hero card 2, 1.1: "在 [MASK] 先驗下它等於 ReMDM" | A.4 equivalence needs κ=1-σ/(1-α_s)∈[0,1] | ReMDM cap/rescale can have σ>1-α_s (κ<0), outside Sec. 3's κ∈[0,1] (design's own 步驟 7 and R4). A.4's Table 11 only claims "similar performance". Add "當 σ_t≤1-α_s" and the MDM carry-over condition (P2).

6. P1 | 步驟 9 "Ψ…持續下降" / Hero card 3 | Table 13: nucleus p=1.0 rows. MDLM cap η=0.005 goes 110.75 (512) → 131.85 (4096); MDLM rescale p=1.0 stays at about 105; Duo rescale 0.02 p=1.0 only reaches 63.70 | The continued decrease depends on p=0.9 and on the schedule. Hero card 3 and 步驟 9 (a) should state "p=0.9、rescale η=0.05"; cite the p=1.0 counterexample, which is in the paper.

7. P1 | LO10 answer "34.75 (5.35)→19.21 (5.17)" | Table 13: those are p=0.9, log-linear, η=0.05 (p=1.0 ancestral is 75.39) | The question and answer omit p, schedule and η. Likewise "25.63→15.05" is Train log-lin / Sample cos, Duo, 4096 steps, best over κ/t_on. Put the conditions in the answer.

8. P2 | 步驟 10: "改用 cosine 取樣後… MDLM 由 24.73 到 17.86" | Suppl. D.1.1 repeats this, but Table 7: MDLM cos-sampled ancestral = 23.93; 24.73 is log-lin sampled | Mixed baselines (paper's wording). Cite 23.93 as the like-for-like cosine baseline. Also note that Duo ancestral with p=0.9 is 22.41 (Table 7), so the "25.63→15.05" gain is against a nucleus-free baseline.

9. P2 | 6.6 I3 "(45) 用 K_k, Algo. 3 用 K_{k-1}" | Algo. 3 case 2: μ=E[·|X<K_{k-1}] "insert w̃, drop smallest K_k" | (45)'s K_k is the post-insertion index, which equals the old K_{k-1}, so this is a naming difference, not a discrepancy. The threshold is also looser than the known dropped K_k. Reword I3 and call it a design choice by the authors.

10. P2 | W2 / hero κ-axis "論文推薦值 0.95、0.99" | Sec. 5.1.2: only CIFAR, piecewise-constant on [t_off,t_on], cosine | Table 9: κ=0.95 over a wider window or in the wrong range is worse. For OWT the recommendation is rescale η=0.05, which is κ≈0.90–0.95 and not constant. Label the axis marks "CIFAR-10 僅限 [t_off,t_on] 內".

11. P2 | W3 觀察 (3) "對應 Table 13 中 32 步無增益" | Table 13: L=1024; MDLM ancestral p=0.9 keeps improving 70.34→37.91 up to 1024 steps | The toy has L=32, so T≤32 is NFE≤L, but its MDM ancestral line is already flat from T=8, whereas real ancestral improves until NFE≈L. The toy's flatness comes from the 推定 denoiser (ε_max chosen for clarity). Say so, and do not map toy T onto the paper's NFE.

12. P2 | 步驟 13: only "CFG 強度 1" | Table 6 caption: "greedy predictions on the last step" | CIFAR FIDs use a greedy final step, which the toy and W2 do not. Add it. Also W2's three denoiser vectors (and 偏錯, from the advisor) are not tagged 推定 though 6.5 requires it.
