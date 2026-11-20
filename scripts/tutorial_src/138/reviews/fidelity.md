Ran `verify/check.py` with `python3 -I`. All asserts pass, and the output matches the design.md section 5 tables. I did not re-derive anything the script reproduces. Items below are what the script does not cover.

1. **P0, Binary and representation.** design.md: "MC 與 Atomic 的區間會隨表示法改變，Binary 不會" (step 3), and step 10 item 4.
   - Binary is invariant as a function, but design.md calibrates δ = f(0.8). δ differs by representation (−log r gives 0.2231; r−1−log r gives 0.0231), so the Binary region changes with it.
   - check.out shows this: KL-Binary at µ=0.1 is [0.2257, 2.2966] under |log r| and [0.4774, 1.7652] under r−1−log r.
   - The paper's literal §4.1 setup (f=−log r, δ≈0.22) gives the Bernoulli-KL region. I computed it: [0.043, 3.97] at µ=0.1 and [0.40, 1.60] at µ=0.5. This matches neither of the page's views.
   - Fix: delete the invariance claim. Say it holds only at equal δ. Add the literal −log r Binary view to C4.

2. **P1, invented KL-Binary default.** C4's default "KL |log r|" Binary is μ|log r|+(1−μ)|log q|. This is not in the paper and is not an f-divergence. The table 5.4 KL-Binary rows are therefore page constructions, yet the C4 note presents the construction as a paper-derived setting (「由論文 Eq. 18–20 … 推出」). Fix: label it, or default to the legitimate Bernoulli KL.

3. **P1, "no 推定 values" is overstated.** Header: "本頁沒有任何推定值". The TV δ is ambiguous. §4.1 gives δ_TV=0.2, but Eq. 14 has f=½|r−1|, so f(0.8)=0.1. The page picks TV-Atomic = 1±0.2/µ and TV-Binary = 1±0.1/µ. Taking the paper's literal δ=0.2 with Eq. 20, whose region is 1±δ/µ, gives 1±0.2/µ for Binary. That is a 2× difference. Hellinger uses the exact 0.011146, not the paper's "≈0.01" (which gives a lower bound of 0.81). Fix: tag these as 推定 and revise the header.

4. **P1, dual clip.** design.md: "這些 token 實際生效的是 dual clip"; C4 shows [max(lo,0.5), min(hi,10)]. Paper §4.1 only says "apply dual clipping to the importance ratio with lower and upper bounds of 0.5 and 10". It does not say how this interacts with the regions. Ye et al.'s dual-clip only caps r for Â<0. A lower bound at 0.5 is the page's reading. Fix: present it as a hypothetical overlay, and drop "實際生效".

5. **P1, step 6 item 1 versus the 11-algorithm claim.** "格子只有 11 個不同演算法" holds only under the page's own δ choices. TV-Atomic and TV-Binary coincide only up to a 2× δ, and the paper's δ is ambiguous (item 3). Fix: write "按本頁校準".

6. **P1, curvature note (M6).** design.md: 「阻力遞減」要用 r−1−log r 才說得通. Paper §3.2: "expansion toward large r faces diminishing marginal resistance", which is a statement about f''=1/r². That f'' is identical under both representations and decreases in r. The claim is overstated. Fix: say the paper's wording is about f'' and is valid; only the sign of f' differs.

7. **P1, Hero card 1.** "|r − 1| 正是兩個策略總變差距離的單樣本估計". Eq. 4 and the text say |r−1| estimates 2·D_TV, with the constant ignored (the page's own step 2 says 2D_TV). Fix: "TV 的單樣本估計（忽略常數 2）".

8. **P1, Hero card 2.** "可得到 12 種剪裁規則" and "依舊機率 µ 加權的估計". The page itself says only 11 are distinct. Binary is not simply µ-weighted; it adds a complement-mass term (Eq. 20). Fix: "12 格（11 種不同規則）".

9. **P2, App. A numbering.** Step 4 item 3: "相加得 Eq. 31；這等於 … data-processing (Eq. 30)". In the paper the sum is Eq. 30, which is the data-processing statement. Eq. 31 is the same inequality rewritten with β and V. Fix: "相加得 Eq. 30，代入 β、V 得 Eq. 31".

10. **P2, missing Thm 3 assumption.** Step 5 and LO7 drop the paper's condition "for any f-divergence with C_f^Vajda>0". Without it Eq. 8 is vacuous. Fix: state it.

11. **P2, LO1 support assumption.** "E_μ[r]=Σπ=1" and "E_μ[rA]=E_π[A]" need supp(π)⊆supp(µ). Fix: add one clause.

12. **P2, unbiasedness of MC.** Step 7: "MC 是無偏的". This holds for the paper's f. It does not hold for the default KL-|log r| MC, where E|log r| is not a divergence. It also holds only for |r−1| = 2D_TV. Fix: qualify.

13. **P2, "Atomic ranks first in every column" check (step 9 M9-4).** Paper claim: "Atomic ranks first in every column of the code table". The page's counter-evidence is a Codeforces tie at 68.2 between χ²-Atomic and H-Atomic. That tie is for second place, because KL-Atomic scores 70.7. Read as the Atomic estimator class, the paper's claim is true for KL-Atomic in all four code columns. Fix: say the claim holds and drop the tie as a counterexample.

14. **P2, LO11 δ.** "µ < 0.0969" assumes δ=0.2231 and |log r|. The paper says δ_KL≈0.22, which gives 0.0955. It also depends on item 4. Fix: state the dependence on δ and on the |log r| form.

15. **P2, §3.4 DAPO.** Step 6 item 5 correctly omits ε. But the page's KL-MC [0.8, 1.25] and the paper's [0.8, 1.26] come from different δ (0.2231 and 0.23), while the paper says both "match DAPO". Fix: do not call the page's value the DAPO-matching one.

Checked and correct against the paper: Eq. 1–20 numbering and content, Thm 1–3 statements, f'/f'' values, Eq. 48/50, Eq. 59, all Table 1 numbers and percentages, the training setup, LO12 numbers, and the C_f values (they match the local 2f''(1) formula).