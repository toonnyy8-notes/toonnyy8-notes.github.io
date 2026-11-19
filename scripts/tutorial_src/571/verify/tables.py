# Checks paper table arithmetic used by C4 and body text (same formulas the page uses).
T11 = {'Breakout': (2, 30), 'Qbert': (164, 13455), 'Pong': (-21, 15), 'Seaquest': (68, 42055)}
norm = lambda g, raw: 100 * (raw - T11[g][0]) / (T11[g][1] - T11[g][0])
# Table 12 raw -> Table 1 normalized
T12 = {'Breakout': [76.9, 61.1, 6.8, 4.5, 40.9], 'Qbert': [2215.8, 14012.0, 156.0, 160.1, 2464.1],
       'Pong': [17.1, 19.3, -14.5, -20.8, 9.7], 'Seaquest': [1129.3, 779.4, 250.1, 370.5, 968.6]}
T1 = {'Breakout': [267.5, 211.1, 17.1, 8.9, 138.9], 'Qbert': [15.4, 104.2, 0.0, 0.0, 17.3],
      'Pong': [106.1, 111.9, 18.0, 0.5, 85.2], 'Seaquest': [2.5, 1.7, 0.4, 0.7, 2.1]}
print('Table12 -> normalized (DT, CQL, QR-DQN, REM, BC) vs Table 1')
for g in T12:
    print(g, [round(norm(g, r), 2) for r in T12[g]], T1[g])
T13 = {'Breakout': [10.0, 22.6, 32.3], 'Qbert': [1045, 2302.5, 1674.1], 'Pong': [-20.3, -16.2, 5.2], 'Seaquest': [521.3, 549.3, 758]}
T4 = {'Breakout': [28.5, 73.5, 108.2], 'Qbert': [6.6, 16.0, 11.8], 'Pong': [2.5, 13.3, 72.7], 'Seaquest': [1.1, 1.1, 1.6]}
print('Table13 -> normalized (10,25,40%BC) vs Table 4')
for g in T13:
    print(g, [round(norm(g, r), 2) for r in T13[g]], T4[g])
# Table 2
cols = ['DT', 'CQL', 'BEAR', 'BRAC-v', 'AWR', 'BC']
T2 = [('ME', 'HalfCheetah', 86.8, 62.4, 53.4, 41.9, 52.7, 59.9), ('ME', 'Hopper', 107.6, 111.0, 96.3, 0.8, 27.1, 79.6),
      ('ME', 'Walker', 108.1, 98.7, 40.1, 81.6, 53.8, 36.6), ('ME', 'Reacher', 89.1, 30.6, None, None, None, 73.3),
      ('M', 'HalfCheetah', 42.6, 44.4, 41.7, 46.3, 37.4, 43.1), ('M', 'Hopper', 67.6, 58.0, 52.1, 31.1, 35.9, 63.9),
      ('M', 'Walker', 74.0, 79.2, 59.1, 81.1, 17.4, 77.3), ('M', 'Reacher', 51.2, 26.0, None, None, None, 48.9),
      ('MR', 'HalfCheetah', 36.6, 46.2, 38.6, 47.7, 40.3, 4.3), ('MR', 'Hopper', 82.7, 48.6, 33.7, 0.6, 28.4, 27.6),
      ('MR', 'Walker', 66.6, 26.7, 19.2, 0.9, 15.5, 36.9), ('MR', 'Reacher', 18.0, 19.0, None, None, None, 5.4)]
print('Table 2 averages (paper: w/o Reacher 74.7 63.9 48.2 36.9 34.3 46.4; all 69.2 54.2 - - - 47.7)')
for i, c in enumerate(cols):
    wo = [r[2 + i] for r in T2 if r[1] != 'Reacher']
    al = [r[2 + i] for r in T2]
    print(c, 'w/o', round(sum(wo) / len(wo), 3), 'all', None if None in al else round(sum(al) / len(al), 3))
best = {}
for r in T2:
    vals = [v for v in r[2:] if v is not None]
    print(r[0], r[1], 'best=', cols[r[2:].index(max(vals))], max(vals))
T3 = [(42.6, 42.9, 43.0, 43.1, 43.1, 44.4), (67.6, 65.9, 65.2, 65.3, 63.9, 58.0), (74.0, 78.8, 80.9, 78.8, 77.3, 79.2),
      (51.2, 51.0, 48.9, 58.2, 58.4, 26.0), (36.6, 40.8, 40.9, 41.1, 4.3, 46.2), (82.7, 70.6, 58.6, 31.0, 27.6, 48.6),
      (66.6, 70.4, 67.8, 67.2, 36.9, 26.7), (18.0, 33.1, 16.2, 10.7, 5.4, 19.0)]
print('Table 3 averages (paper 56.1 56.7 52.7 49.4 39.5 43.5):', [round(sum(c) / 8, 3) for c in zip(*T3)])
print('Key-to-Door ratio etc: 500k/50M =', 500e3 / 50e6)
print('Table5 drop ratios Pong', round(2.5 / 106.1, 4), 'Breakout', round(73.9 / 267.5, 4))
# Table 7: delayed minus dense (Hopper ME, M, MR)
dtD, dtO, cqD, cqO = [107.3, 60.7, 78.5], [107.6, 67.6, 82.7], [9.0, 5.2, 2.0], [111.0, 58.0, 48.6]
print('Table7 DT delayed-dense', [round(a - b, 1) for a, b in zip(dtD, dtO)], 'CQL', [round(a - b, 1) for a, b in zip(cqD, cqO)])
print('Table6 1K: DT-CQL', round(71.8 - 13.1, 1), '10K:', round(94.6 - 13.3, 1))
