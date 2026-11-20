Goldfish review of design.md (No. 550). The three verify scripts, re-run with `python3 -I`, reproduce c1_c2_out, c3_out and c4_out byte for byte. The Table 3 values, the Table 8/13/15/16/17 numbers and the C2 boundaries (0.1823, 2.3979) also match the paper. No P0 found. Items are ordered by severity.

1. P1. "k=0 或 k=G … 這題對梯度沒有貢獻"; common mistake 3: "這些題目不提供梯度".
   - Paper: Eq. 1 puts −β·D_KL inside the sum for every o_i, whatever A_i is.
   - Fix: say the advantage term is 0 and there is no reward signal, but the KL term still gives a (small) gradient when β>0. C3's own gradient formula keeps the KL term for these groups.

2. P1. C1 plot "P0(p)=p^G+(1−p)^G" and observation (2) are shown for every f.
   - The formula holds only for pure binary accuracy rewards. With f<1, Eq. 4 makes r=acc+fmt, so format variance gives a signal even when all answers are correct or all wrong.
   - Fix: draw P0 only at f=1, or compute it as P(all r_i equal) = (p·f+(1−p)(1−f)... joint) and label the f=1 assumption.

3. P1. Step 3: "同一題其他輸出的平均獎勵取代價值模型當基準線".
   - Eq. 3 uses mean over all G rewards, including r_i itself. Fig. 3 says "group scores".
   - Fix: "同一題 G 個輸出（含自己）的平均".

4. P1. Step 6 and LO8: "反思詞…出現頻率…增加 5 到 7 倍".
   - App. C.2: "the count of the reflective words rises 5- to 7-fold".
   - A count rises with response length, so it is not a per-token frequency.
   - Fix: write "出現次數"; do not call it a rate or frequency.

5. P1. Step 7 and LO10: Dev1 reasoning drop "因冷啟動資料量小" stated as fact.
   - Sec. 4: "due to the limited size of the cold-start dataset". This is the authors' attribution, not a measured result.
   - Design 6.3 requires labelling this as 「作者認為」. Fix: label it.

6. P2. LO5 and step 4: "是 KL 的不偏估計量".
   - A.3 only says "an unbiased estimator of the KL divergence (Schulman 2020)".
   - Your one-line proof needs o∼π_θ and supp π_θ ⊆ supp π_ref.
   - Eq. 1 samples from π_old, so after the first mini-batch (ρ≠1) it is off-policy. The unbiasedness is for the KL value, not for the gradient of KL.
   - Fix: add both assumptions to the 〔本頁推導〕.

7. P2. LO10 and R4: "論文所寫的 25%、17% 為百分點".
   - Sec. 4 writes "improving by 25% and ArenaHard by 17%" and never says "points". Table 3 gives +25.5/+16.7 points (41.1%/22.1% relative).
   - Fix: write "與 Table 3 的百分點差吻合（論文未說明）" and tag it 〔本頁推導〕 in the LO answer too.

8. P2. Step 9, Table 16: "學習率 2e-6，其餘同 R1-Zero".
   - B.4.1: max length is fixed at 32,768, with no raise to 65,536 and no 8.2k jump. Other items (16 outputs, 32 questions/step, 400-step reference reset, 16 mini-batches) match.
   - Fix: "其餘同第一階段 RL 設定（最大長度固定 32,768）".

9. P2. C3 table row "β=0.1" (acc 0.781 vs 0.760 at β=0.001).
   - Larger KL speeds learning in this toy. This is an artifact of the 400-step reference resets, which here coincide with S=400. The paper supports no such claim.
   - Fix: add one sentence in the 〔本頁構造〕 caption, or do not show β=0.1 as a "learning faster" case.

10. P2. C2 chips and common mistake 5: ε=0.2 as "PPO 慣用".
    - The paper never uses or cites 0.2. R1-Zero ε is unstated. The paper only says ε=10 for stage 1 and that "the clip ratio plays a crucial role".
    - Fix: label 0.2 as a PPO convention (Schulman 2017), not from this paper, and say that at ε=10 clipping is almost inactive on the A<0 side.

11. P2. Hero card 3 and 1.1: "1.5B … 28.9% 高於 GPT-4o 的 9.3%".
    - Table 15: the paper says "surpass non-reasoning baselines on mathematical benchmarks". The 1.5B model is below GPT-4o on GPQA (33.8 vs 49.9) and LiveCodeBench (16.9 vs 32.9).
    - Fix: add "數學評測".

12. P2. Step 8: "AIME 2025 pass@1 75%".
    - E.2 does say "75% solve rate (Pass@1)", but Table 13 gives 11.3/15 as "Average Score". The USAMO index of o1 (261.0) exceeds R1's (256.7).
    - Fix: cite the E.2 sentence, and mention o1 261.0 next to the threshold comparison.

13. P2. R6: "兩個 86.7%".
    - There are three: R1-Zero cons@16 (Sec. 2.3), R1 cons@64 (E.4), and Distill-Llama-70B cons@64 (Table 15). Also 85.9 appears three times: Table 9 pure average, Table 11 jailbreak unsafe ratio, and 50-language score with risk control.
    - Fix: label each by model and metric.

14. P2. C4 "下降前 3" for R1 lists MMLU −0.2 and MMLU-Redux −0.2.
    - Table 3 bolds only differences with p<0.01. A −0.2 difference is noise.
    - Fix: note that the bold significance markers are not carried into the ranking, or add a ≥1 point threshold.

15. P2. Hero card 1 and LO1: "只依答案對錯給獎勵".
    - Eq. 4 and Sec. 2.2 add a format reward with equal weight. LO1 does say "(與格式)", but hero card 1 and the title do not.
    - Fix: say "答案對錯與格式".