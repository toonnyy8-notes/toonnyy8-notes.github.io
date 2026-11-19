# 443 design.md review: pedagogy and build readiness

1. P1 | S4 "450–550 字 + C1 + C2b" | S4 holds Eq. 2, the schedule formula, the derived closed form for ᾱ, the ideal denoiser (a mixture posterior with softmax responsibilities), two widgets, LO5 and LO6. The ideal denoiser appears only as "一段說明". It is also not stated to come before C2b's 擴散 tab, so the widget could be placed before its concept. | Split S4 into 4a (Eq. 2 and BC, with C2b's real-data and Gaussian tabs only) and 4b (schedule and C1). Teach the denoiser in 3–4 literal sentences with the mixture formula before the 擴散 tab appears, or move the tab to S8.

2. P1 | S6 "350–450 字 + 演算法" | Eq. 4 appears with no Bellman or TD target taught (S1 gives only the Q definition). Double Q, target networks, soft update, CQL max-Q backup and ρ/γ all arrive in one section. | Add 2 sentences in S1 or S6 on the TD target. Move the max-Q backup to a footnote. Raise the length to about 550.

3. P1 | S7 "500–600 字 + C3" | One section covers the 4 methods' results, Fig. 1, App. A, App. H (mixture density network, K, Table 5) and C3. C3 also needs softmax logits, Nadaraya–Watson smoothing and Adam, none of which is taught. | Move App. H to S10 or S11. Add 3 sentences on the toy's parameterization (softmax weights over the 4 peaks) before C3.

4. P1 | S2 "KL…mode-covering" and LO2 | KL direction, MLE, MMD and CVAE are used but never defined. The 0.3225 figure needs the variance of a mixture, which beginners do not know. | In the S2 body: one sentence on "MLE = 最小化 KL(資料‖策略)", one on what MMD and CVAE are, and the variance computation as 2 lines.

5. P1 | S9 / LO11 | MoRel, Onestep RL, CQL, IQL, AntMaze "拼接" and D4RL normalization are required answers but are never introduced. Only 4 baselines appear in S2. | Add a one-line description of each baseline named in LO11, and define "拼接" literally.

6. P1 | 4.5 "Eq. 1、3、4 以 aligned 多行" | Eq. 2 is not in the list, and its 3-distribution subscript on E will overflow at 390px. Eq. 1 has a nested fraction, β_i/√(α_i(1−ᾱ_i)) ε_θ(a^i,s,i). | Define c_i = β_i/√(1−ᾱ_i) first, then write Eq. 1 on 2 lines. Put Eq. 2 in aligned form with the expectation subscript on its own line. Test at 390px.

7. P1 | C3 "改變 η 或 N 時自動重設" and "setTimeout 每段 50 步" | The spec does not say to cancel the pending rAF or timeout on reset, so a race is possible. With N=20, a 50-step chunk is about 250 ms of main-thread blocking, which exceeds the 8 ms budget. | State that reset cancels the handle and bumps a run token. Chunk by time (about 8 ms) under reduced motion as well. Test the N=20 single-frame time.

8. P1 | 5 C3 / 6.3 selfTest "0.896 ± 0.01" | The weight at η=1, N=5 reads .907 / .910 / .896 at steps 150 / 300 / 600, so the step-to-step jitter (0.014) exceeds the tolerance. Different engines' Math.exp ordering will fail the test. | Set the tolerance to ±0.03, or average the last 50 steps. For the η=0 test, say it uses the same loss. Also include a floor check such as w_down > 0.8.

9. P1 | 4.0 "數值保護" | Only r_k uses log-sum-exp. softmax(ℓ) and Q̃'s Σg_k (ℓ_Q=0.2) can underflow to 0/0 for a far-away a (squared distance/0.08 > 745). | Specify LSE for the softmax and for Q̃. Clamp g's denominator with max(·,1e-300).

10. P1 | C3 and 4.0 symbols | r_k means responsibility in 4.0 and reward mean in Q̃, "Σ r_k g_k". ℓ (logit) collides with ℓ_Q. α in α=η/2.375 collides with α_i. | Rename: reward mean → R_k, ℓ_Q → σ_Q, responsibility → ρ_k. ρ is already used as the soft-update coefficient, so use w_k or q_k for responsibility instead.

11. P1 | C1 "N≥20 時只列四欄" | Rows are headers and columns are i, so N=10 gives 10 columns and N=5 gives 5. At 390px, N=10 overflows and the spec does not say the table sits in `.tscroll`. | Wrap in `.tscroll`, or transpose (i as rows, at most 5 shown, "…" for the rest) for N≥10.

12. P2 | 6.2 `<tspan baseline-shift="super">` | Browser support for baseline-shift is uneven (Firefox). The viewBox 640 SVG scaled into the 600–768px range drops text below 12px. | Use dy plus a smaller font-size. Set the narrow viewBox breakpoint to 768, or set a min width for the SVG.

13. P2 | C3 readouts "𝓛_d、𝓛_q" and C2 readouts | The dynamic DOM needs symbols with subscripts, but KaTeX in dynamic nodes is neither specified nor allowed. | Use static KaTeX labels with `<span>` value cells, or plain "Ld／Lq" text. Write the convention into the spec.

14. P2 | Literal style | "蓋住所有峰", "拉", "拼接", "落後指標", "空白區" and the details-summary phrase "檢查理解" read as metaphors or loose wording. | Write mode-covering literally ("高斯的密度覆蓋所有資料峰，包括峰間無資料的區域"). Use "往…靠近" or "限制在…附近" instead of "拉". Rename the `<details>` summary to "檢查題與參考答案".

15. P2 | A11y/dark | C3 right and left lines both use `--ink-3` and differ only by dash. C4 "點選／focus" rows have no stated tabindex or role. The tablist arrow-key behavior is stated only for the segmented buttons. Hatch patterns and the 1000-point dot radius and opacity have no dark-mode values. | Use a distinct token for the left line. Give each C4 row `tabindex=0` with an `aria-expanded` list. Specify tablist arrow keys. Specify dot radius (2.5px), opacity (.55), and a dark-mode hatch color.
