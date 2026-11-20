Fidelity review of design.md (tutorial 129) against paper.txt.

**Verification scripts:** I ran `verify/check.py` with `python3 -I`. It finishes with "done". Its numbers match the section 5 acceptance tables, including C1 ‖drift‖ 0.0928, the TIS clip onset at ε≈1.068, and the C4 residuals. I found no P0 (outright wrong) items. Every equation number, section reference, and quoted number I spot-checked matches the paper: Eq. 1–13, Table 1, Table 2, footnotes 2–5, the §5.4 numbers (58/52/51/30/12/<5%), the A.1 and A.4 numbers, and the B.1 hyperparameters. The items below are the remaining problems, most severe first.

1. **P1, C2 readout.** Design.md §4 C2 says "G 越大越接近 ±1 的值".
   - The paper (§4 and footnote 4) only says group centering rescales the on-policy gradient by 1−1/G. It says nothing like this about ±1.
   - Your own formula c = (1−1/G)·2(a−R̄₀₁) tends to 2(a−R̄₀₁) as G→∞, which is not 2a−1 in general.
   - The statement holds only at π_A=0.5. At π_A=0.8, G→∞ gives c_A=0.24, while ±1 gives 0.6. Your table row (G=8, π_A=0.8, c_A=0.21) shows this.
   - Fix: say "G 越大，係數趨近 2(a−R̄)；π_A=0.5 時才等於 ±1 的值".

2. **P1, C3 time scaling.** The design says t½ scales with ε and links it to §5.1's "mild TIM needs more steps".
   - In the toy the bias is a constant additive logit offset εξ. Drift is therefore a constant-direction push, so the logits grow roughly linearly (about ηεt).
   - The paper describes the error as "compounds in a feedback loop". The toy shows linear accumulation under a state-independent bias. The 1/ε scaling of t½ comes from that assumption.
   - Fix: add a sentence on the page that the toy shows persistent bias accumulation, not exponential compounding, and that the 1/ε law is not a paper claim. Likewise note that this fixed offset resembles quantization, not staleness.

3. **P1, C3 and the staleness link.** The design says "頁面用這點（SC 熵下降）連到 …§5.3「staleness 下應疊加 IS」".
   - The paper's §5.3 explanation is only "SC measures covariance under the sampler".
   - The toy's fixed offset is quantization-like. Staleness is a time-varying bias.
   - Fix: state that the toy illustrates Cov_q vs Cov_p but does not model staleness.

4. **P1, scope of "drift iff policies match".** Step 6 and the Hero card repeat §1's "equal to 0 if and only if training and sampling policies match".
   - The drift is E_q[R]·s̄, so it is also zero when E_q[R]=0. C1 shows ‖drift‖ 0.0098 for ±1 at ε=0.5.
   - In logit coordinates s̄=q−p, so s̄=0 iff p=q. For a real network s̄=Jᵀ(q−p) can vanish with p≠q.
   - Fix: add a 〔本頁說明〕 note that the paper's "iff" is loose.

5. **P2, "SC − on-policy" wording.** Step 9 says the "唯一差別" between SC and on-policy is Cov_q vs Cov_p, and C1 treats exact IS as equal to on-policy.
   - The toy fixes Q_v=E[R|y_t=v] and ignores that the continuation distribution (and thus R) also differs between q and p.
   - Step 3 notes per-token IS does not correct the prefix distribution, but C1b labels per-token IS "= on-policy" with no caveat.
   - Fix: on the C1b plate, add "Q_v 固定，未建模 continuation 的差異".

6. **P2, Eq. 6 and 11 formatting.** The paper text literally reads "L = −R log p_yt − Σ sg[…] log p_v".
   - The design writes −R(log p_yt − Σ …). That matches the code and the sentence "−∇L = R(s−s̄)" in A.3, but the paper's printed form has no parentheses.
   - Confirm against `paper.pdf`. If the paper does have a typo, S8 should say the page restores the sign and R factor from the code.

7. **P2, residual-drift condition.** Step 12 says the residual is 0 "只有" when the sampler and trainer tails are proportional.
   - That is sufficient, not necessary in general (the vector sum could cancel). It is necessary only in logit coordinates, where the residual is q−q̂.
   - Fix: change "只有" to "當…時".

8. **P2, LO3 answer.** "PG 在什麼條件下等於 SFT" answers "advantage 為 +1/0 且完全離線". The paper (§3) says "binary rewards … advantages +1/0 … completely offline".
   - Fix: add the binary-reward condition to the reference answer.

9. **P2, Table 1 superscripts.** The design drops the P/V (paper default / verl default) superscripts. TIS is "V", MIS is "P". Your statement that "各方法用原論文或 verl 預設值" is right, but the table loses which is which.
   - Fix: restore the superscripts.

10. **P2, Fig. 2 group-centered reward.** R7 and C2 say Fig. 2's group-centered baseline is "未說明". B.1 says "Unless otherwise noted, we use REINFORCE with group-centered rewards … Ri∈{−1,+1}" (Eq. 13), so ±1 is the likely default, though not stated for Fig. 2.
    - Fix: say "論文預設 Eq. 13 用 ±1，Fig. 2 未另述".

11. **P2, top-k overlay caveat.** Step 13 correctly notes the head-only SC+IS is exact only when q̂=q. Add the same caveat to the LO11 answer and the C4 plate. C4 shows the identity check against q̂, not against the true q (the "residual" column covers that).

12. **P2, conclusion vs §5.** The design handles "only method that trains stably" versus TIS 51% well (R5). The LO12 answer does not repeat it. Add it to the LO12 answer so a reader who memorises the Conclusion is corrected.

13. **P2, Hero card 3.** Card 3 says SC "從每個梯度項減去…平均值". More accurately, it subtracts a per-prefix expected score, which is deterministic given the prefix. This distinguishes SC from IS, which the paper stresses.

I found no values presented as the paper's that are actually calibrated (推定). The 〔本頁推導〕 items I checked are correct: E[A_i|prefix]=(1−1/G)(…), the fp4 rounding, the CE/KL equivalence, the value-baseline check, and the Eq. 10 sanity checks.