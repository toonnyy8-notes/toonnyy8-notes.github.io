SCORE 10/12

| LO | Verdict | Note |
|---|---|---|
| 1 | partial | Omits "成本約與 token 數成正比（256–1024）" framing and "二次項到數千 token 才成為主項". Page states both (line 80: "二次項要到數千個 token 以上才成為主項"). Tester only said "減半約減半". Minor tester omission; no page fix. |
| 2 | correct | A–H all present. |
| 3 | correct | |
| 4 | correct | |
| 5 | partial | Missing ToMeSD r=50% FID 33.02 (base 33.12), 1.65 vs 3.09 s/im. Page states it (S6: "FID 33.02（基線 33.12），每張 1.65 對 3.09"). Tester omission; no page fix. |
| 6 | correct | 17/6=2.83x, 2.15x, ~2.5x saturation, 18%, 1.16x all present. |
| 7 | correct | |
| 8 | correct | |
| 9 | correct | |
| 10 | correct | |
| 11 | correct | |
| 12 | correct | |

Both partials are tester omissions of numbers the page does state (key figures sit in mid-paragraph, not in the 本節重點 summary). Optional page fix: add "33.02 / 1.65 vs 3.09" and "數千 token 才成主項" to the S2 and S6 本節重點 boxes so they match the reference.

Page/reference contradictions: none found (18.2%, 17/6, 2.15x, FID 6.61 vs 7.97 and 5.16 vs 0.49 all consistent).

Real beginner gaps worth a one-sentence fix:
1. 先備 does not define softmax, cosine similarity, or key/query/value; ToMe, log s and the H-Net p_t formula depend on them. Add one sentence each to S1.
2. Amdahl and 17/6: the page names the law but never shows the arithmetic. Add "3+11+3=17 blocks; if the middle 11 cost 0, the remaining 6 give 17/6".
3. "軟混合", "投影 skip" and "ratio loss" are used before they are defined. Give each a one-line gloss at first use.
4. Hilbert connectivity: add one sentence, "consecutive indices are grid neighbours, so any interval is a connected path".
5. Axes A–H are scattered across sections; add a one-line A–H legend in the S3 summary and repeat it where it is used.
