Score: 11/12 correct, 1 partial (LO9)

LO1 correct - both difficulties (deadly triad; d^pi vs d^mu shift) match.
LO2 correct - Eq. 4, x_0 = H states, y = reward/constraint/skill.
LO3 correct - loss, optimal predictor, marginal p_k, sign note.
LO4 correct - states only, Eq. 7, both reasons, Table 2 numbers, +0.5 within SE, confound.
LO5 correct - p=0.25 drop to Ø, omega=0 uncond, omega=1 cond.
LO6 correct - score form and why sampling != p(x)p(y|x)^omega. (Reader also cited toy numbers; harmless.)
LO7 correct - two changes, alpha=0.5, Table A1 104.3/111.8/107.1.
LO8 correct - loop, C=20, 1.26 s, warm-start 0.48 s. (Reader says "<20% loss" for 5 steps, which matches page line 352.)
LO9 partial - reader found both phrasings, AND/NOT, no OR, feasibility, one-hot-sum failure, but did not say the derivation needs the x_k version (reader called it "未裁定"). Page DOES say it (page.txt line 367: "下面的推導對每個擴散步都需要給定 x_k 的版本；給定 x_0 的條件獨立一般不蘊含給定 x_k 的…"). Reader miss, but the sentence is buried mid-paragraph; a fix is cheap.
LO10 correct - 81.8/61 averages, 6 of 11 rows highest (4/9 locomotion), examples.
LO11 correct - Kuka 60.4 vs 52.3, multi 63.8, Bound 60.1 / Pace 38.5 / Trot 1.4, swap note.
LO12 correct - stochastic dynamics, Table A3, explanation, extra limits.

Stuck points
- #4 (x_0 vs x_k): page already states it (line 367), so it is a reader miss. Make it more visible (Fix 1).
- #5 (Alg. 1 history columns / 替換): page already says 替換 = inpainting (line 6 of Alg. 1) and says the paper does not specify which columns are history or where s_t sits (line 350). The "missing line 10" is not explained, so that remains valid (Fix 2).
- #1 (symbol clashes, alpha_k vs alpha, R, s vs omega) and #2 (unconditional noise notation): cosmetic. The symbol table exists. #2 is answered by the p=0.25 drop text. Optional Fix 3.
- #3 (sign/coefficient of noise-vs-score): LO3 text already says Sec. 2.2 omits them and App. D has the sign. No fix needed.

Fix list
1. S10, just before the derivation (page.txt line 367): add a bold, standalone sentence: "結論：推導需要的是『給定 x_k(τ)』的條件獨立（App. D 的寫法）；Sec. 3.3 寫的『給定 x_0(τ)』較弱，不足以支撐推導。" Put it right after the two quotations so it is not buried.
2. S9 Alg. 1 table: add a one-line note under the algorithm, e.g. "原演算法的行號跳過第 10 行，內容無缺漏（若確為排版缺號請註明；若不確定則寫『論文此處行號不連續，本頁照錄』）。" Verify against the paper first. Also add to the 替換 gloss: "inpainting = 把已知的欄固定為觀測值，只讓其餘欄由去噪決定"。
3. (Optional) In the Eq. 8 / Alg. 1 line 7 gloss, add: "ε_θ(x_k, k) 即 ε_θ(x_k, Ø, k)：無條件版本是同一網路輸入 Ø".
