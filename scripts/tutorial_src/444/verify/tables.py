"""Sum checks of the paper's tables (numbers typed from src/paper.txt)."""
T1 = {  # Table 1 columns: CQL-A IQL-A DQL-A IDQL-A CQL-1 IQL-1 DQL-1 IDQL-1
'loco': [[44.0,47.4,51.1,51.0,46.4,47.6,50.6,49.7],[58.5,66.3,90.5,65.4,64.4,63.7,75.2,63.1],[72.5,78.3,87.0,82.5,81.6,81.9,83.4,80.2],
         [45.5,44.2,47.8,45.9,45.4,43.1,45.8,45.1],[95.0,94.7,101.3,92.1,88.5,42.5,94.5,82.4],[77.2,73.9,95.5,85.1,74.5,78.4,86.7,79.8],
         [91.6,86.7,96.8,95.9,64.6,88.1,93.3,94.4],[105.4,91.5,111.1,108.6,99.3,73.7,102.1,105.3],[108.8,109.6,110.1,112.7,109.6,110.5,109.6,111.6]],
'ant':  [[74.0,87.5,93.4,94.0,65.0,86.4,47.6,93.8],[84.0,62.2,66.2,80.2,41.3,62.4,35.8,62.0],[61.2,71.2,76.6,84.5,31.4,76.0,42.5,86.6],
         [53.7,70.0,78.6,84.8,25.8,74.8,46.3,83.5],[15.8,39.6,46.4,63.5,8.5,31.6,19.0,57.0],[14.9,47.5,57.3,67.9,7.0,36.4,25.2,56.4]],
'printed_loco': [698.5,692.4,791.2,739.2,674.3,629.5,741.2,711.6], 'printed_ant': [303.6,378.0,418.5,474.6,180.0,368.4,216.4,439.3],
'printed_total': [1002.1,1070.4,1209.7,1213.8,854.3,997.9,957.4,1150.9]}
cols = 'CQL-A IQL-A DQL-A IDQL-A CQL-1 IQL-1 DQL-1 IDQL-1'.split()
for j, c in enumerate(cols):
    l = sum(r[j] for r in T1['loco']); a = sum(r[j] for r in T1['ant'])
    bad = [n for n, d in (('loco', l - T1['printed_loco'][j]), ('ant', a - T1['printed_ant'][j]), ('total', T1['printed_loco'][j] + T1['printed_ant'][j] - T1['printed_total'][j])) if abs(d) > 0.05 + 1e-9]
    flag = ('  <-- MISMATCH ' + ','.join(bad)) if bad else ''
    print(f'Table1 {c}: loco {l:.1f} (printed {T1["printed_loco"][j]}), ant {a:.1f} (printed {T1["printed_ant"][j]}){flag}')
print('IDQL-A antmaze with Table 4 value 84.2 for medium-play:', round(474.6 - 84.5 + 84.2 + (sum(r[3] for r in T1['ant']) - 474.6) , 1), '-> sum', round(sum(r[3] for r in T1['ant']) - 84.5 + 84.2, 1))
print('IDQL-1 - IQL-1 antmaze =', round(439.3 - 368.4, 1))
# Table 3 (offline -> online)
T3 = {'IQL': [(86.7,96.0),(75.0,84.0),(72.0,95.0),(68.3,92.0),(25.5,46.0),(42.6,60.7)], 'printed_IQL': (408.2, 473.7),
      'Max': [(92.0,99.0),(78.7,85.6),(83.3,97.8),(84.7,98.0),(60.1,88.0),(61.1,90.7)], 'printed_Max': (459.9, 559.1),
      'Imp': [(93.5,99.5),(78.4,73.0),(84.4,94.0),(84.0,98.7),(60.3,90.0),(61.4,93.0)], 'printed_Imp': (462.0, 548.2),
      'RLPD': [(0,99.0),(0,99.0),(0,99.5),(0,98.0),(0,88.0),(0,87.5)], 'printed_RLPD': (0.0, 571.0)}
for k in ('IQL', 'Max', 'Imp', 'RLPD'):
    o = sum(a for a, _ in T3[k]); n = sum(b for _, b in T3[k])
    print(f'Table3 {k}: offline {o:.1f} (printed {T3["printed_"+k][0]}), online {n:.1f} (printed {T3["printed_"+k][1]})')
# Table 4 IDQL column
t4 = [51.0,65.4,82.5,45.9,92.1,85.1,95.9,108.6,112.7]; a4 = [94.0,80.2,84.2,84.8,63.5,67.9]
print('Table4 IDQL loco', round(sum(t4),1), 'ant', round(sum(a4),1))
print('antmaze -A -> -1 change:', {c: round(T1['printed_ant'][j+4] - T1['printed_ant'][j], 1) for j, c in enumerate(['CQL', 'IQL', 'DQL', 'IDQL'])})
print('locomotion -A -> -1 change:', {c: round(T1['printed_loco'][j+4] - T1['printed_loco'][j], 1) for j, c in enumerate(['CQL', 'IQL', 'DQL', 'IDQL'])})
print('total -A -> -1 change:', {c: round(T1['printed_total'][j+4] - T1['printed_total'][j], 1) for j, c in enumerate(['CQL', 'IQL', 'DQL', 'IDQL'])})
print('antmaze gap IDQL-IQL: -A', round(474.6 - 378.0, 1), ' -1', round(439.3 - 368.4, 1))
