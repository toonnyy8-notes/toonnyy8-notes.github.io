# Fidelity review: design.md vs paper (Value Forcing, arXiv 2611.02160) and advisor.md

Verification scripts: c1-c4, numbers.py re-run (python3 from verify/ cwd; `-I` breaks `import common`); stdout identical to *_out.txt. Table 2-5 / Table 5-ablation / Fig. 2, Fig. 6 numbers re-checked against paper.txt and the rendered PDF crops: all match. No P0 (factual error) found. Equations 1-17, 21, 24, Table 1, Table 6/7 references match.

1. P1. Location: S10 title "資料越少，VF 與 IVF 的優勢越明顯（但不是處處領先）". Paper (Sec 6.2): "especially beneficial when offline data is limited"; Tables 2-5: VF minus best explicit = +5 (10%), +6 (25%), +2 (50%), +5 (100%), which is not monotone (only IVF is: +16, +15, +8, -1). Fix: title "IVF 的優勢隨資料變少而擴大；VF 領先 2-6 點、不隨資料量單調變化".

2. P1. Location: 1.1 "VF 在 10%／25% 資料的總平均最高（48、54）". Paper says highest among explicit methods; at 25% IVF = 61 > VF = 54 (Table 3). Fix: write "explicit 方法中最高".

3. P1. Location: C2 / LO6 "乾淨方程在每個 τ 都收斂到 Z^π 本身" and C1 "乾淨版". Paper (Sec 2, D.2) only says Value Flows "naïvely apply the clean Bellman equation at all noise levels"; its DCFM target is v̄_Q without a γ factor (advisor 3.2-6), and a flow's τ=0 marginal is the input Gaussian by construction. The fixed point is that of a hypothetical clean marginal operator, not what Value Flows learns. Fix: label C1/C2 "clean 算子（本頁模型）", add "不是 Value Flows 實際學到的分佈"; paper's own wording is only "inconsistent supervision at τ=0, offering an explanation".

4. P1. Location: LO12 "哪個設計拿掉後影響最大？… reward scaling 86%→11%". Fig. 4 gives no numbers except "reduces by 44%" (relative vs points unknown, advisor 3.2-8); IVF alternatives only "much higher". Ranking across Fig. 4 and Fig. 5 is not supported. Fix: ask "reward scaling 拿掉後 VF/IVF 各是多少" and state the 44% verbatim; drop the "最大" ranking.

5. P1. Location: C3 modes "錯誤：目標用 +ε / 目標的 r 乘 τ", "常見錯誤框". These are advisor-identified transcription pitfalls, not errors observed in the paper or practice. Fix: call them "抄寫時易犯的符號錯誤（本頁假設）"; the "γ 接近 1 時錯誤變小" line should not be read as a result on training.

6. P2. Location: S4 "下一章顯示兩個算子的不動點完全不同". By 本頁 A(iv) they differ at every τ<1 and coincide at τ=1 (difference (1-τ)r, noise variance (1-τ)²(1-γ²) per step). Fix: "在 τ<1 不同，越靠近 τ=1 越小".

7. P2. Location: C1/C2 defaults γ=0.9 (c=0.436), C2 γ≤0.95. Paper uses 0.999/0.9999 (c=0.0447/0.0141); advisor says do not suggest the correction is large. Fix: caption in C1/C2 saying the defaults exaggerate the one-step effect; keep the existing c table adjacent.

8. P2. Location: C1 reward-scaling checkbox ("r_eff=(1-γ)r") leaves m′, s′ unscaled; scaled return Z′ would also shrink by (1-γ), so the printed W₂ (0.0014 at γ=0.999) mixes units. Fix: scale m′, s′ too or state that only the one-step τ=0 gap is shown.

9. P2. Location: symbol table/C1/C3 inputs: "s′ ∈ [0.05, 2]" (std of Z′) vs Eq. 11 "s′" (next state); C2 "x₀, x₁" (state-action pairs) vs S2 x⁰, x¹ (noise/target). Fix: rename std to σ′ and pairs to i∈{1,2} (or "pair A/B").

10. P2. Location: LO5 "前提是固定策略 π、固定 τ、一維回報". Prop. 1 also needs Z ∈ 𝒵 (uniformly bounded moments), p∈[1,∞), γ<1; Banach needs completeness (advisor B, unverified in paper). Fix: add to LO5 answer and S5 prerequisites.

11. P2. Location: 1.4 step 3 "FlowCritic（也代表 DFC、FlowIQN 的終點類）". Table 1: FlowIQN has ✗ at "Gaussians at initial noise levels" and quantile-coupled noise, unlike FlowCritic ✓. Fix: "DFC、Dual-Flow RL 同為終點類；FlowIQN 另有 quantile 耦合（Table 1: ✗/✗/✓）".

12. P2. Location: C2 "與精確值的差約 0.01–0.02". Script shows x₁, τ=0 corrected sd 0.9570 vs exact 1 (0.043) and mean -0.0159. Fix: write "約 0.02–0.05" (acceptance ±0.06 already covers it).

13. P2. Location: step 9 "return flow 積分 10 步（Table 6）" next to VF 訓練 M=1 一步. Paper: Table 6 "Return-flow integration steps 10" is a default for all methods; D.3 VF trains with one-step, evaluates with 10 (advisor 3.2-3). Fix: add one line "Table 6 的 10 步對應評估／其他方法；VF 訓練用一步" in 論文未說明 list.

14. P2. Location: step 8 / LO9 "V 是 Q 的 expectile". A.2 derivation also uses E[∫v dτ]=E[φ(z⁰|1)] (implicit E[z⁰]=0) and uniqueness only given π^µ_∞ (which depends on the solution; advisor 3.1-8). Design lists assumptions (exact optimization, self-consistent fixed point) but not these two. Fix: add as 本頁解讀 note in S7.

15. P2. Location: C4(b) "max 場平均高於 max_a Q" is 本頁 comparison-theorem derivation (correct for 1-D, shared z⁰, Lipschitz fields); paper only asserts "≠" (D.5). Already tagged 本頁推導; ensure the widget text says "≥ 成立於一維共用 z⁰，論文只寫 ≠" and keeps the existing "不是 Fig. 4 解釋" caveat.

Checked and fine: Eq. 5 two-end check, closed form τZ+(1-τ)G (derivation re-done), contraction proof steps, Eq. 11/13 signs, PCBF identity, 81%/89%, Table 2-5 totals, Table 5 ablation-task means 93.5/77, Fig. 2/6 values, 44%/86%/11% quotes, footnote 1 not expanded, no etymology invented for "Value Forcing", τ vs µ notation warned.
