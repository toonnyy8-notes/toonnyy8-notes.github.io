"""C4 data (Table 2, 3, 7, 9, Fig. 8) and derived arithmetic."""
T2 = {  # (dataset, env): BC, BRAC-p, AWAC, CQL, FisherBRC, TD3+BC ; None = '-'
 ('random','HalfCheetah'):(2.0,23.5,2.2,21.7,32.2,10.2), ('random','Hopper'):(9.5,11.1,9.6,10.7,11.4,11.0), ('random','Walker2d'):(1.2,0.8,5.1,2.7,0.6,1.4),
 ('medium','HalfCheetah'):(36.6,44.0,37.4,37.2,41.3,42.8), ('medium','Hopper'):(30.0,31.2,72.0,44.2,99.4,99.5), ('medium','Walker2d'):(11.4,72.7,30.1,57.5,79.5,79.7),
 ('medium-replay','HalfCheetah'):(34.7,45.6,None,41.9,43.3,43.3), ('medium-replay','Hopper'):(19.7,0.7,None,28.6,35.6,31.4), ('medium-replay','Walker2d'):(8.3,-0.3,None,15.8,42.6,25.2),
 ('medium-expert','HalfCheetah'):(67.6,43.8,36.8,27.1,96.1,97.9), ('medium-expert','Hopper'):(89.6,1.1,80.9,111.4,90.6,112.2), ('medium-expert','Walker2d'):(12.0,-0.3,42.7,68.1,103.6,101.1),
 ('expert','HalfCheetah'):(105.2,3.8,78.5,82.4,106.8,105.7), ('expert','Hopper'):(111.5,6.6,85.2,111.2,112.3,112.2), ('expert','Walker2d'):(56.0,-0.2,57.0,103.8,79.9,105.7)}
ALG = ['BC','BRAC-p','AWAC','CQL','Fisher-BRC','TD3+BC']
REPORTED = [595.3, 284.1, None, 764.3, 974.6, 979.3]
for i, a in enumerate(ALG):
    vals = [v[i] for v in T2.values()]
    tot = None if None in vals else round(sum(vals), 1)
    print(f"{a:11s} sum={tot} reported={REPORTED[i]}")
# TD3+BC >= BC on every task?
print("TD3+BC < BC on:", [k for k, v in T2.items() if v[5] < v[0]])
print("TD3+BC best (incl ties) on:", sum(v[5] >= max(x for x in v if x is not None) for v in T2.values()), "/15")
# Table 3: base = total - impl - alg (minutes)
for name, tot, imp, alg in [('CQL(GitHub)', 251, 25, 89), ('Fisher-BRC(GitHub,TF)', 132, 39, 33), ('Fisher-BRC(ours,PyTorch)', 128, 15, 58), ('TD3+BC', 39, 0, 0)]:
    print(f"{name}: base≈{tot-imp-alg} min; total/TD3+BC={tot/39:.2f}")
print("DT 16h21m / TD3+BC:", round(981/39, 2), " DT(100k) 1h38m / TD3+BC:", round(98/39, 2))
T9 = {'HC-m':(42.6,48.3),'Hop-m':(67.6,59.3),'W-m':(74.0,83.7),'HC-mr':(36.6,44.6),'Hop-mr':(82.7,60.9),'W-mr':(66.6,81.8),'HC-me':(86.8,90.7),'Hop-me':(107.6,98.0),'W-me':(108.1,110.1)}
print("Table 9 Total(DT): DT", round(sum(v[0] for v in T9.values()),1), "TD3+BC", round(sum(v[1] for v in T9.values()),1))
T9_all_td3bc = [11.0,8.5,1.6] + [v[1] for v in T9.values()] + [96.7,107.8,110.2]
print("Table 9 TD3+BC total:", round(sum(T9_all_td3bc),1), "(reported 1013.2)")
# Total "±" in Table 2 / Table 9 equals the SUM of per-row ± (not a std of the summed score)
PM2 = {'BC':[0.1,0.1,0.2,0.6,0.5,6.3,1.8,5.9,1.5,13.2,27.6,5.8,1.7,1.3,24.9],
       'CQL':[0.9,0.1,1.2,0.3,10.8,8.3,1.1,0.9,2.6,3.9,1.2,13.1,7.4,2.1,7.6],
       'Fisher-BRC':[2.2,0.2,0.6,0.5,0.4,1.0,0.9,2.5,7.0,9.5,43.3,4.6,3.0,0.2,32.4],
       'TD3+BC':[1.3,0.1,1.6,0.3,1.0,1.8,0.5,3.0,5.1,4.4,0.2,9.3,1.9,0.2,2.7]}
for k, v in PM2.items(): print(f"Table 2 sum of per-row ± {k}: {sum(v):.1f}  (reported Total ±: BC 91.5, CQL 61.5, Fisher-BRC 108.3, TD3+BC 33.4)")
PM9 = [1.1,0.6,1.7,0.3,4.2,2.1,0.5,18.8,5.5,4.3,9.4,0.5,1.1,7,0.3]
print(f"Table 9 TD3+BC sum ± all {sum(PM9):.1f} (57.4), DT-subset {sum(PM9[3:12]):.1f} (45.6); DT {0.1+1+1.4+0.8+7+3+1.3+1.8+0.2:.1f} (16.6)")
print("Gap (paper Totals) TD3+BC - Fisher-BRC = %.1f; row sums 979.3 - 975.2 = %.1f" % (979.3-974.6, 979.3-975.2))
