# Acceptance grade, tutorial 127 (SOAR)

Score: 12/12 (all correct; no partial or wrong)

## Per-LO verdict
1. LO1 correct: fail@128 definition, zero advantage under binary reward, and the 120/20/10/0 format-graded reward (teaches format only) all match.
2. LO2 correct: 0.774 and 0.062 match.
3. LO3 correct: same pretrained init, teacher writes Q and A, never sees hard problems, reward = accuracy gain on Q_R.
4. LO4 correct: outer/inner stated; black-box reward, no differentiation through the inner loop. (Eq. 1 expectation not written out, but not required.)
5. LO5 correct: R formula, Q_R = 64, baseline subtraction for comparability, r = 4 averaged to cut variance. Greedy decoding was mentioned.
6. LO6 correct: 3-step moving average > 0.01, best dataset then median student, D_best = union. For "no promotion" the reader said all students are discarded, which matches the reference (back to baseline).
7. LO7 correct: premise pi0(S_acc) > 0, extra term proportional to sum A, RLOO sum = 0, REINFORCE keeps the term.
8. LO8 correct: PS vs PQ and the purpose of PQ; Hard-Only G=128 rules out "just more compute".
9. LO9 correct: 18.9 vs 9.6, percentage points, 9.3/12.4 = 0.75 with the right denominator.
10. LO10 correct: 8.3 vs 7.5 and 6.8 vs 6.4; no improvement; abilities roughly independent.
11. LO11 correct: u_i definition, 10.82 vs 34.91, one of three seeds collapsed.
12. LO12 correct: 32.8% vs 64.6%, C.5 Correct-only vs Well-posed, limitations (62 more questions, HARP only, 3 seeds).

## LO7 reachability without the optional block
Only partly. Outside ch.07 the page gives:
- ch.02 (line 71): "一個群組的優勢總和永遠是 0".
- ch.06 (line 167): "格式不符就重抽".
- Common mistakes (line 538): "重抽格式錯誤的輸出，卻用 REINFORCE 而不是 RLOO：Prop. 1 不成立（第 07 章）".

A reader who skips ch.07 sees that REINFORCE is a mistake. That reader is not told why, and not told the premise pi0(S_acc) > 0. The reader's own stuck point 5 says LO7 needs ch.07. This is acceptable for an optional chapter, but the one-line reason belongs in the main text.

## Fix list
1. Ch.06, right after "格式不符就重抽" (line 167): add a sentence such as "重抽讓老師的輸出分布從 pi0 變成 pi，多出的梯度項正比於優勢總和；RLOO 的優勢總和為 0，所以更新不偏（細節見選讀的第 07 章）。" This makes LO7's core answer reachable without the optional block.
2. Line 538 (common mistakes): extend with "因為 REINFORCE 的優勢總和不為 0，重抽造成的額外項會留下來".
3. Stuck point 1 (symbols p, s, s̄, G, g, r, R): add a short disambiguation line at the start of ch.02 or in the symbol table, for example "G = 每題取樣數；g = 資料集數；r = 每資料集學生數". Skip adding any more symbols.
4. Stuck point 2 (table writes "G = 128；原文 g = 128"): add a note that the paper uses g for the student group size here and this page writes G. This conflicts with g = number of datasets, so say so directly.
5. Stuck point 4 (inconsistent numbers: Vendi 31.75 vs 28.33, 63% vs 64.6%, 4x vs 3.40x): place a single "本頁以表格為準" line at the first occurrence of each. Ch.11 already does this for 63% vs 64.6%.
6. Stuck point 3 (Alg. 1 unspecified details): no change needed. The page already marks them 〔本頁約定〕; optionally group the three conventions into one list in ch.06.
