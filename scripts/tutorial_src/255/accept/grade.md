# 255 Goldfish acceptance grade

**Score: 12/12**

## Per-LO verdict
- LO1 correct: Eq.1 product, tau=1/alpha, power-then-normalise all present.
- LO2 correct: inequality, "sharpening", alpha>1 needed, alpha=1 equality, E1 noted.
- LO3 correct: 0.160/0.125 vs 0.160/0.250, sum 0.90 flaw, ordering unaffected (alpha threshold not asked).
- LO4 correct: sum-of-powers vs power-of-sum, F factor, all three equality conditions.
- LO5 correct: Eq.14, Eq.15, N>1 and alpha>1, 0.354 check, "does not prove correctness".
- LO6 correct: normaliser intractable, ratio cancels, Def.1 irreducible+aperiodic (Neal), extra positivity conditions.
- LO7 correct: Z_t product ratio, conditions, last-token-only A=1.
- LO8 correct: block structure, line 7 pi_k->pi_{k+1} (E4), line 10 no freeze, m from 1.
- LO9 correct: 8.84, 9600/14.14, 13.5-14.5, (T/B)^2/2.
- LO10 correct: the four lower-than-GRPO cells with exact numbers, 62-77% with Phi GPQA 22% exception, not causal.
- LO11 correct: k>1 text, "strictly better than both" overclaim, E5 reasoning.
- LO12 correct: covers every reference item plus extras (EOS, seeds, etc.).

No wrong or partial grades, so no LO-driven body fixes.

## Stuck points: assessment and fixes
1. Many symbols (x_<t, x_>t, eps, eps', N, F, Z_t, pi_k) - page already has a symbol table; fix: add a one-line pointer to the table at the first use of F and Z_t (S3b, S5b).
2. Hard to tell paper claims from page derivations - page already tags "(本頁推導)" inline; fix: add one sentence in the intro saying "(本頁推導) marks derivations not in the paper".
3. MH / stationary / detailed balance only one line - fix: add a one-line plain-language gloss in S5a ("chain visited states converge to frequency proportional to p^alpha").
4. Alg.1 index and "no frozen prefix" easy to misread - E4 callout exists; fix: move the pi_k vs pi_{k+1} note directly under the line-7 row, not only in the callout.
5. 8.84x vs 14.14x gap and T meaning - page already states T=679 is the MATH500 mean length and the E[T^2]>=E[T]^2 reason; fix: state in S7 that 8.84 is the paper figure and 13.5-14.5 is the page's recount, in one summary sentence before the arithmetic.

Items 1-5 are polish only; the reader still answered every LO correctly.

## Numbered fix list
1. S3b/S5b: link F and Z_t first use to the symbol table.
2. Intro: define the "(本頁推導)" marker once.
3. S5a: one-sentence intuition for stationary distribution.
4. S6: place the pi_{k+1} correction beside Alg.1 line 7.
5. S7: lead with a one-sentence "paper 8.84x vs page recount 13.5-14.5x" summary.
