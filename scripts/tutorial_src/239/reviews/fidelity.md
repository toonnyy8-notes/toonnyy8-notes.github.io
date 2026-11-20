Verify script: check.py re-run, output identical to check_out.txt, matches 5.1-5.5 tables. Equation/number spot-checks (Eq. 5-6, 9-14, 17-23, 25-30, 36, 39-44; Tables 1-6; A.3 numbers; 11.7x/1.19x; 4.92/4.96) all match the paper. No P0 found. Issues:

1. P1 | LO11 and step 11: "凍結 PCA 基底 1.63、AsymFlow 1.76" and Table 2 "AsymFlow-H/16† 1.57" | Table 1 AsymFlow 1.76 is a controlled re-run on the JiT-H/16 backbone; Table 2 reports AsymFlow 1.57 with † (uses a pretrained representation encoder; caption). | State in LO11 and S9 that 1.76 is the same-backbone ablation, 1.57 is the published †-class number, and that † ≠ AnisoFlow's setting.

2. P1 | C1 "放大倍率（截斷後）… 表下模板句 速度誤差 = δ × 放大倍率"; table row t=0.01,w=0: error 10.0, amp 100, clamped 25 | Paper B.1: the recovery denominator itself is lower-bounded by σ_min. Clamping changes û (even for a perfect prediction û = u·den/0.04, a bias), not only the error multiplier. The widget shows the unclamped û and error beside "25x", so the page implies error = δ×25 = 2.5 while the readout says 10. | Either recompute û with den' = max(den, 0.04) when the box is ticked (and show the bias), or label "截斷後放大倍率" as "上限 25 的誤差倍率，不含截斷造成的偏差". Make the template sentence use the same denominator.

3. P1 | LO9 reference answer "A.3 測得 C̄eff∝λ^1.00 … 比值 λ^1.09" | A.3 calls this "a consistency check … rather than a direct measurement": ε̂ and x̂ come from auxiliary linear readouts on frozen features, and only 18 of 60 directions pass the reconstruction filter (selected for reconstruction fidelity, not for the scaling). | Replace 測得 with 以輔助線性讀出估計; add "18/60 方向、非直接量測" to the reference answer.

4. P1 | C3 "讀者應觀察到… t 變小，同樣的 A, C 給出較大的 w*" | Paper §5/Eq. 20: A(t) falls and C(t) rises with t, and this is exactly what cancels α/σ. A fixed-(A,C) slider teaches the opposite of the paper's time-invariance point. | Add a plate line: "A、C 在真實網路隨 t 變化，抵銷見 S6；此處固定只為看公式".

5. P2 | Step 6 "任何線性或高斯玩具都不可能「學出」w∝λ" | Eq. 43-44 show only that the optimal (time-conditioned) linear predictor gives a recovered velocity independent of W. Capacity-limited, finite-sample or non-optimal Gaussian models are not covered. | Narrow it to "最佳線性預測器下損失與 W 無關（Eq. 44）".

6. P2 | Hero w 軸 "像素空間學到的範圍 [10^-3, 10^1]" | w≈0.15λ, and A.3 reports λ up to 171 (w≈26); Fig. 9 axes reach 10^2. The upper end 10^1 undershoots. It is a visual estimate, so it is effectively a calibrated value. | Extend to about 10^2 or drop the band and cite "0.15λ，λ 由 0.015 到 >100".

7. P2 | Step 12 #3 "w_i=0（JiT 的 Frozen zeros）實際上取不到" | B.1: w=exp(η) is strictly positive, so a learned weight cannot reach 0. JiT's frozen w=0 baseline is reachable because it is not parameterized this way. | Rewrite: "可學的 w 取不到 0；凍結的 0 是 JiT 基線設定".

8. P2 | C5 "n ∈ {8…2048}", LO10 "第一次更新是單一 minibatch 的帶雜訊 PCA 基底" | Eq. 42 averages over M·N patches (images × patches per image), not a handful of samples. n=8 errors of 20° misrepresent real training scale. The zero-output premise is "as is common at initialization", not guaranteed. | Add a plate note: "實際 n = 圖數×patch 數（至少數萬），這裡的小 n 只為看趨勢"; soften the premise.

9. P2 | C4 logit-normal "μ −1.5–1.5, s=1" | Table 6: training uses LogitNormal(0,1) before a flow shift (2.0 for T2I) and uniform sampling for RAEv2. The widget ignores the shift, so its μ is not comparable to the paper's. | Note it on the plate (the page already says the parameters are not from the paper; add "未含 flow shift").

10. P2 | Step 5 "非奇異條件變成特徵值非負" | The paper says W ⪰ 0 is the practical restriction. Strictly, only negative real eigenvalues in (-∞,0) cause a singularity; nonnegativity guarantees no singularity at every t. | Say "保證所有 t 非奇異的條件是特徵值非負".

11. P2 | Step 1.3(c) "最佳 FID（限 plain DiT-like 類別）" | The paper's contribution (c) says "state-of-the-art pixel-space generation" while Table 2 lists SiD2 UViT/1 at 1.38. The qualifier is the designer's, not the paper's. | Mark it 〔本頁註〕 or quote the paper's "best in-class".

12. P2 | Step 9 "B/16 的參考線為 w=0.2λ（Fig. 4b）" | Fig. 4 labels w=0.2λ in a reference-lines legend (with w=√λ, 0.3λ, 0). It is not stated as a fitted slope for B/16. | Say 參考線，非擬合值。
