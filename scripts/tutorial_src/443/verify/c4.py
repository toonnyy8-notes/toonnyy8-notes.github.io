"""C4: table arithmetic (averages, best baseline per task, where Diffusion-QL is not the top entry)."""
import json, os
D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c4_data.json')))
r1 = lambda x: round(x + 1e-9, 1)
for dom in D['T1']:
    n = len(dom['rows']); cols = list(zip(*[r[1] for r in dom['rows']]))
    avgs = [r1(sum(c) / n) for c in cols] + [r1(sum(r[2][0] for r in dom['rows']) / n)]
    print(dom['domain'], 'recomputed avg', avgs, 'paper', dom['avg'], 'MISMATCH' if avgs != dom['avg'] else 'ok')
    for name, base, (dq, se) in dom['rows']:
        b = max(base); who = [dom['algs'][j] for j, x in enumerate(base) if x == b]
        flag = 'DQL<best' if dq < b else ('tie' if dq == b else '')
        print('  %-30s DQL %.1f±%.1f best baseline %.1f %s %s  margin %.1f' % (name, dq, se, b, who, flag, dq - b))
    bavg = max(dom['avg'][:-1]); print('  best baseline avg %.1f vs DQL %.1f' % (bavg, dom['avg'][-1]))
t2 = D['T2']; cols = list(zip(*[r[1] for r in t2['rows']]))
print('T2 avg', [r1(sum(c[0] for c in col) / 3) for col in cols], 'paper', t2['avg'])
for name, vals in t2['rows']:
    best = max(v[0] for v in vals); print('  T2', name, 'best', [t2['algs'][j] for j, v in enumerate(vals) if v[0] == best])
dom_rows = {'Gym': D['T4'][:9], 'AntMaze': D['T4'][9:15], 'Adroit': D['T4'][15:17], 'Kitchen': D['T4'][17:]}
for k, rows in dom_rows.items():
    off = r1(sum(r[1][0] for r in rows) / len(rows)); on = r1(sum(r[2][0] for r in rows) / len(rows))
    print('T4', k, off, on, 'paper', D['T4avg'][k])
print('T2 DQL column vs T4 online (Gym ME):', [r[1][-1][0] for r in t2['rows']], [r[2][0] for r in D['T4'][6:9]])
print('T5 TD3+BC vs T1 TD3+BC (ME):', [r[1] for r in D['T5']], [r[1][5] for r in D['T1'][0]['rows'][6:9]])
