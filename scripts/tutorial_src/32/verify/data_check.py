"""Arithmetic checks of numbers quoted from the paper (inconsistencies -> design R-table)."""
chk = lambda name, a, b: print(f"{name}: computed {a:+.2f} vs stated {b:+.2f} {'OK' if abs(a-b)<0.015 else 'MISMATCH'}")
chk("Abstract 4B math +6.49 (Fig3/Table3 49.07-42.58)", 49.07-42.58, 6.49)
chk("Abstract 4B math +6.49 (Table1 49.93-42.57)", 49.93-42.57, 6.49)
chk("Abstract 4B general +7.54 (Table2 31.15-26.34)", 31.15-26.34, 7.54)
chk("Sec3.2 8B math +5.51 (Table1 53.72-48.64)", 53.72-48.64, 5.51)
chk("Sec3.2 OctoThinker-3B +2.68 (29.32-26.64)", 29.32-26.64, 2.68)
chk("Sec3.2 8B base-challenger->R-Zero +1.52 (53.72-52.10)", 53.72-52.10, 1.52)
chk("Sec3.2 4B base-challenger->R-Zero +3.7 (49.93-45.01)", 49.93-45.01, 3.7)
chk("Sec3.3 8B general +5.13 (34.50-31.98)", 34.50-31.98, 5.13)
chk("Sec3.3 Octo-3B general +3.65 (11.12-7.47)", 11.12-7.47, 3.65)
chk("Sec4.1 w/o filtering general drop 'over 6' (31.15-26.69)", 31.15-26.69, 6.0)
chk("Sec4.3 4B +2.35 (51.34-48.99)", 51.34-48.99, 2.35)
chk("Sec4.3 8B +3.69 (56.84-53.15)", 56.84-53.15, 3.69)
chk("Sec4.4 static solver 59.0->47.0", 47.0-59.0, -12.0)
# Table 1 averages of the 7 math benchmarks
rows = {"4B base":[45.70,38.24,68.20,87.79,41.04,10.30,6.7,42.57],"4B R-Zero":[57.27,52.94,79.60,92.12,44.59,9.60,13.40,49.93],
        "8B base":[51.95,50.00,78.00,89.08,44.74,12.10,14.60,48.64],"8B R-Zero":[61.67,60.66,82.00,94.09,48.89,13.30,15.40,53.72]}
for k,v in rows.items(): print(f"Table1 {k}: mean of 7 = {sum(v[:7])/7:.2f}, stated Avg {v[7]}")
g = {"4B base":[20.88,50.58,7.57,26.34],"4B R-Zero":[27.55,55.47,10.42,31.15],"8B base":[28.33,58.97,8.63,31.98],"8B R-Zero":[31.38,61.53,10.60,34.50]}
for k,v in g.items(): print(f"Table2 {k}: mean of 3 = {sum(v[:3])/3:.2f}, stated {v[3]}")
# m=10 filter band
print("filter counts for delta=0.25, m=10:", [c for c in range(11) if abs(c/10-0.5) <= 0.25])
# Fig 3 peaks / Table 7 accuracy at peak
fig3 = {"0.6B":[16.77,27.39,27.15,25.05,25.15],"1.7B":[32.71,39.19,39.20,37.03,38.27],"4B":[42.58,48.06,48.44,49.07,46.52]}
t7 = {"0.6B":[70.6,53.4,50.8,44.0],"1.7B":[69.4,55.2,52.2,45.2],"4B":[71.0,56.2,48.8,42.2]}
steps = [0,15,30,45,60]
for k,v in fig3.items():
    i = max(range(5), key=lambda j: v[j]); print(f"Fig3 {k}: peak step {steps[i]} ({v[i]}), Table7 acc at peak {t7[k][i-1]}%")
# S3 static GRPO example and S6 derived numbers
import statistics as st
r = [1, 0, 1, 1, 0]; mu = st.mean(r); sd = st.stdev(r)
print("S3 example: mean", mu, "sample std", round(sd, 4), "adv", [round((x - mu) / sd, 3) for x in r])
for rho in (0.5, 0.9): print("rho", rho, "P(all 5 equal) =", round(rho**5 + (1 - rho)**5, 4))
print("m=5 band:", [c for c in range(6) if abs(c/5 - 0.5) <= 0.25 + 1e-12])
# round 1: Table 4 math averages vs Fig. 4 baseline, Table 5 row
t4 = {"Human":[57.97,55.15,80.8,92.04,48,9.58,10.31], "R-Zero":[57.27,52.94,79.6,92.12,44.59,9.6,13.4], "R-Zero+Human":[57.9,53.2,81.2,92.2,47.7,10.32,14.67]}
for k, v in t4.items(): print(f"Table4 {k}: math mean of 7 = {sum(v)/7:.2f}")
print("Table4 Human 50.55 vs Fig.4 dashed baseline 48.99 -> mismatch", round(50.55-48.99, 2))
wins = [max(t4, key=lambda k: t4[k][i]) for i in range(7)]; print("Table4 math per-column winner:", wins)
print("Table5 Solver(step15) column: 59.0 -> 53.0 -> 47.0")
