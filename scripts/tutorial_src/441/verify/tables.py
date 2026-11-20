"""Paper tables (transcribed from paper.txt) and the live computations C3 performs."""
METHODS = ['BC','CQL','IQL','DT','TT','MOReL','Diffuser','DD']
T1 = [  # dataset, env, values (None = '-'), DD stderr
 ('Med-Expert','HalfCheetah',[55.2,91.6,86.7,86.8,95,53.3,79.8,90.6],1.3),
 ('Med-Expert','Hopper',[52.5,105.4,91.5,107.6,110.0,108.7,107.2,111.8],1.8),
 ('Med-Expert','Walker2d',[107.5,108.8,109.6,108.1,101.9,95.6,108.4,108.8],1.7),
 ('Medium','HalfCheetah',[42.6,44.0,47.4,42.6,46.9,42.1,44.2,49.1],1.0),
 ('Medium','Hopper',[52.9,58.5,66.3,67.6,61.1,95.4,58.5,79.3],3.6),
 ('Medium','Walker2d',[75.3,72.5,78.3,74.0,79,77.8,79.7,82.5],1.4),
 ('Med-Replay','HalfCheetah',[36.6,45.5,44.2,36.6,41.9,40.2,42.2,39.3],4.1),
 ('Med-Replay','Hopper',[18.1,95,94.7,82.7,91.5,93.6,96.8,100],0.7),
 ('Med-Replay','Walker2d',[26.0,77.2,73.9,66.6,82.6,49.8,61.2,75],4.3),
]
T1_AVG = [51.9,77.6,77,74.7,78.9,72.9,75.3,81.8]
T1K = [('Mixed','Kitchen',[51.5,52.4,51,None,None,None,None,65],2.8),
       ('Partial','Kitchen',[38,50.1,46.3,None,None,None,None,57],2.5)]
T1K_AVG = [44.8,51.2,48.7,None,None,None,None,61]
T2 = {'Diffuser':[107.6,58.5,96.8],'CondDiffuser':[111.3,66.3,76.5],'CondMLPDiffuser':[105.6,54.1,66.5],'DD':[111.8,79.3,100]}
A3_P = [0.00,0.05,0.10,0.15]
A3 = {'BC':[(55.2,1.5),(49.3,3.6),(25.8,3.8),(15.1,4.3)],
      'DD':[(84.7,2.2),(77.3,3.1),(53.2,4.1),(41.3,4.9)],
      'Diffuser':[(72.4,1.4),(63.2,2.9),(52.3,4.6),(41.6,5.1)],
      'CQL':[(73.2,2.3),(61.8,3.7),(51.2,4.3),(42.2,5.5)]}
T3 = [('Single Constraint - Stacking',45.6,58.0),('Single Constraint - Rearrangement',58.9,62.7)]
T3M = [60.3,67.2]

def report():
    out = []
    out.append('== Table 1 locomotion: recomputed column averages vs printed ==')
    for j,m in enumerate(METHODS):
        a = sum(r[2][j] for r in T1)/9
        out.append(f'{m:9s} recomputed {a:.3f} printed {T1_AVG[j]}  diff {a-T1_AVG[j]:+.3f}')
    out.append('== Table 1 kitchen averages ==')
    for j,m in enumerate(METHODS):
        if T1K[0][2][j] is None: continue
        a = (T1K[0][2][j]+T1K[1][2][j])/2
        out.append(f'{m:9s} recomputed {a:.3f} printed {T1K_AVG[j]}')
    out.append('== Table 1 per-row winner, DD rank (1 = highest; ties share rank) ==')
    nbest = 0
    for ds,env,v,se in T1+T1K:
        vals = [(x,m) for x,m in zip(v,METHODS) if x is not None]
        mx = max(x for x,_ in vals)
        win = [m for x,m in vals if x==mx]
        dd = v[-1]; rank = 1+sum(1 for x,_ in vals if x>dd)
        better = [f'{m} {x}' for x,m in vals if x>dd]
        if rank==1: nbest += 1
        out.append(f'{ds:10s} {env:11s} max {mx} by {win}; DD {dd}±{se} rank {rank}; higher than DD: {better}')
    out.append(f'DD rank 1 (incl. ties) in {nbest} of 11 rows')
    out.append('== Table 2: CondDiffuser vs Diffuser, DD vs CondDiffuser ==')
    for i,ds in enumerate(['Med-Expert','Medium','Med-Replay']):
        out.append(f"{ds}: Cond-Diff {T2['CondDiffuser'][i]-T2['Diffuser'][i]:+.1f}; DD-Cond {T2['DD'][i]-T2['CondDiffuser'][i]:+.1f}")
    out.append('== Table A3: DD minus best other; interval (mean±se) overlap with best other ==')
    for i,p in enumerate(A3_P):
        dd = A3['DD'][i]
        others = [(A3[m][i][0],A3[m][i][1],m) for m in ['Diffuser','CQL']]
        bo = max(others)
        ov = (dd[0]-dd[1]) <= (bo[0]+bo[1])
        out.append(f'p={p:.2f} DD {dd[0]}±{dd[1]} best of Diffuser/CQL {bo[2]} {bo[0]}±{bo[1]} diff {dd[0]-bo[0]:+.1f} overlap {ov}')
    out.append('== Table 3 averages ==')
    out.append(f'single Diffuser {(45.6+58.9)/2:.2f} DD {(58.0+62.7)/2:.2f}; multiple DD {(60.3+67.2)/2:.2f}')
    return '\n'.join(out)

if __name__ == '__main__':
    print(report())
