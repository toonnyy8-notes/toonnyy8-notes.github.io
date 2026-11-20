NO-GO

**Explain-back.** The page teaches that Gaussian diffusion on one-hot vectors, followed by argmax, gives at each time step the same marginal as uniform-state discrete diffusion (USDM), with retention coefficient T(alpha~) from (11) and the ODE (12). The paths differ, so the claim holds only per marginal; (39) and (40) give the argument, with the wording "not necessarily". Theorem 3.2 motivates the discrete likelihood, but its KL is infinite, so it is only a motivating argument. Duo's loss (17) equals UDLM's (49) algebraically. Curriculum learning replaces the argmax input with softmax(w/tau). Its time window is inconsistent with (11), so the page marks the reading as a presumption. P_DDT takes argmax along the straight-line PF-ODE with x_theta=x, so each token flips at most once. DCD distils with KL along P_DDT, with step size doubling per round. Components: W1 checks Monte Carlo frequencies against (28); W2a plots T(alpha~) for a chosen K; W2b adds three bands on the same alpha~ axis; W3 compares Gaussian P given w_s with USDM Q given z_s, with the TV distance; W4 steps USDM and P_DDT through T_steps and counts changes; W5 shows softmax(w/tau) with its p_x, gap and entropy. Each has defined inputs, defaults and expected readings.

**Blocking issues**

1. Undefined symbol. Step 1 and S2 use (1): "$q_t(\cdot\mid x;\alpha_t)=\mathrm{Cat}(\cdot;\alpha_tx+(1-\alpha_t)\pi)$". The S1 symbol list in 1.4 (K, V, x, m, Cat, 1, t, alpha_t, z_t, w_t, alpha~_t, sigma~_t, phi, Phi) and the S2 plan do not define pi. Step 1 gives its value (m for MDM, 1/K for USDM), but the page plan does not put that definition at first use. Fix: define pi at (1) in S2.

2. LO11 reference answer contradicts the page's own treatment. It says "教師是上一輪結束時的學生權重（非 EMA）". Step 11 and S11 say B.2.2 text describes a pre-training EMA teacher, while Alg. 2 uses theta_ema, and the page must list both readings (F8). A reader who follows B.2.2 would mark this answer wrong. Fix: scope the answer to "依 Alg. 1" and mention the Alg. 2 / B.2.2 reading.

**Non-blocking notes**

1. S12 Table 7 excerpt ("8 步 111.88 對 193.05" etc.) has no column headers. Duo is probably the first value, inferred from the 16-step row, but state it in the table header.
2. S0 static SVG ("K=3 的三根 w 長條 → one-hot") gives no values. Specify the w vector.
3. S14 ("100–150 字") and S15 ("200 字") have lengths but no content outline.
4. W4 DDT hist[0] is argmax over pure noise (a=0). Add one line saying this matches a uniform start, since the USDM start is uniform. Otherwise the first rows in 5.4 look inconsistent.
5. (3) has a plus sign and (12) a minus sign for Q_t. S2 details and S8 note resolve it. Make sure no code uses Q_t numerically.
6. Link `../index.html#paper-393`: I could not find an anchor `paper-393` in index.template.html by grep. Confirm the generated index provides it.
7. LO9 presents the 5.1 schedule (tau=0.001 for 500K, then 0) as fact. Add "依 §5.1" for consistency with F9.
8. Checked: all 7 notes have slug, title, cluster, kind, tags, arxiv, sources, related. timeline.json parses with 15 entries, and its categories match 6.6. All linked note slugs exist in notes/ or notes_split/. `diffusion-duality` is the page itself. `prediction-targets-k-diff` exists in tutorials_raw. I ran check_T.py, check_w3.py and check_loss.py as a spot check. Their output matches the 5.1 band table, the 5.3 TV rows and the 5.6 values.

Scope note: beyond the three files listed in the brief, I opened the notes frontmatter and ran the three verify scripts, which I did not read in full.
