Verification: I ran verify/check.py. It printed ALL ASSERTS PASSED and its output is identical to verify/check.out. The W1–W5 tables in design.md §5 and the LO6/LO8/LO10 numbers match it. I also hand-checked Table 1–3 against the paper, plus the gaps 11.0/7.0/3.1/4.1 and 1.6/1.8/2.6, the 400-step figures, App. B, and the §/eq/Step numbering. All of those are correct. I found no P0. Remaining issues, most severe first:

1. P1, W3 caption: 「它說明 (14) 的更新不會停在 PMD 的解 π⁺」 (and step-9 row 1 「更新不會停在 π⁺」).
   Paper: §3.2 Steps 2–4 add group estimation, binary KL, ω, mask M and cap C. The paper gives no fixed-point analysis and says nothing about stopping at π⁺.
   W3 only simulates Step 1 (population, tabular, no mask, no cap, μ fixed).
   Fix: say 「只做 Step 1 的線性化在此玩具中不停在 π⁺；(14) 是否停在 π⁺ 論文未分析」.

2. P1, hero card 3 / 1.1 / S8 end: 「與 GRPO 只差一個權重」, 「只差一處」, 「只差 sg(r) → min(sg(ω), C)」.
   Paper §3 (after eq 14): 「The mask M follows the GRPO clipping rule in (6), with the weight ω replacing the importance-sampling ratio」.
   So two things change: the weight, and the quantity the mask tests (ω instead of r). BPO's ε_low/ε_high are also unspecified.
   Fix: list both differences. The `.vs-code` side-by-side should also mark M's argument.

3. P1, theorem scope (步驟 8 / S6): it never says Theorem 1 optimizes over all of Π_μ.
   Paper §3.1: 「min_{π∈Π_μ}」, with Π = Δ(V)^S, i.e. every policy, tabular and unconstrained.
   It also describes a single PMD step relative to a fixed μ.
   Nothing guarantees the minimizer is reachable by a parametric LLM. Add this as an assumption beside 「唯一性的範圍」.

4. P2, W3 line 289 and note at line 145: 「線性化 (26)（用精確 V^μ、ϕ=1，即只做 Step 1)」.
   Paper: the linearization is the second line of (25). (26) is after Step 2 (group mean and 1/std).
   Fix: label it (25), consistent with LO8's 「式 25」.

5. P2, line 145 / W3: 「永遠不為零」.
   The table gives case I's b-component as 0, and the gradient is also 0 when R is constant.
   Fix: 「不隨 π 改變；一般不為零」.

6. P2, LO11 answer: 「消融 … 25.3–25.8 對 GRPO-ClipHigher 20.5」.
   Paper Table 2: ε=0.3 gives 24.1. 25.3–25.8 is the C sweep only; the paper says 25.4–25.8 for ε ∈ {0.05, 0.1, 0.2}.
   Fix: give both sweeps, or write 24.1–25.8.

7. P2, 〔示範值〕 tagging (line 9 convention, W1–W4).
   Only W5's mask thresholds carry 〔示範值〕. The following are not in the paper and are untagged:
   - W1: Q and η.
   - W2: μ, R, η.
   - W3: α=η.
   - W4: Δ.
   - The 4-of-16 group example. G=16 is the paper's, but 4 correct is invented.
   Fix: tag them.

8. P2, W3 line 290: 「Hessian 最大特徵值 ≤ 1.805/η」.
   The number is unexplained. At six corner/grid μ values I tried, λ_max(Σ P_μ g gᵀ) was ≤ 0.25, so about ≤ 0.25/η.
   The statement is true but the constant is unsupported. Drop it or derive it.

9. P2, common mistake 2: 「替 BPO 填入 ϵ_low=0.2、ϵ_high=0.28」 is labelled an error.
   Paper App. B says only 「identical settings except policy loss」 and 「GRPO-ClipHigher uses [0.8,1.28]」. BPO's thresholds are not given.
   Fix: 「論文未給，不應當作論文值」.

10. P2, LO10 / W5 「預期觀察」: 「ω 對高機率 token 敏感」 and 「高機率區相反」 are stated without conditions.
    The 2× only holds for the unsmoothed ratio. At ϵ=0.1, (0.98, 0.99) gives ω=1.0909, and ω is capped to [1/11, 11].
    Fix: state the ϵ dependence in the LO answer.

11. P2, line 128: 「softmax 下 μ 全支撐，此限制在實驗中不起作用」.
    The paper does not say this. It is an inference, and bf16 underflow in the rollout engine could break it.
    Fix: mark it 〔本頁推導〕 and hedge.

12. P2, line 144 / LO9: 「第一次更新等於 GRPO」.
    Paper §2.3 motivates μ≠π partly by rollout/training engine mismatch, so even the first update has r≠1 in practice.
    Fix: add 「（忽略引擎差異）」 to the LO9 answer. Line 144 already hedges, but the LO answer does not.

Checked and found correct: the (5)≡(3) mask derivation; the A-vs-Q and Jensen remarks; the logit-form residual; the 二元 KL gradient e_y−π; the ω range [0.0909, 11]; the ω>C condition p>(2.2+q)/3; the Lagrangian; the martingale proof steps; the Π_μ counterexample; the std example numbers; no values are presented as the paper's that are actually calibrated.