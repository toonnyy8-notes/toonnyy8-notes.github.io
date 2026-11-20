"""Acceptance values for the No. 441 page (same algorithm as the page JS). Output: check_out.txt"""
import sys, math; sys.path.insert(0, '.')
from core import *
import tables
L = []
p = L.append
beta, abar, bt = SCH
p('== schedule (cosine s=0.008, K=100) ==')
p(f'beta1 {beta[1]:.6f} beta50 {beta[50]:.6f} beta100 {beta[100]:.6f}')
p(f'abar1 {abar[1]:.6f} abar50 {abar[50]:.6f} abar100 {abar[100]:.4e}')
p(f'bt2 {bt[2]:.6f} bt50 {bt[50]:.6f} bt100 {bt[100]:.6f}')
r = mulberry32(1); p('mulberry32(1) first 3: ' + ', '.join(f'{r():.9f}' for _ in range(3)))
r = mulberry32(1); p('gauss seed1 first 2: ' + ', '.join(f'{gauss(r):.6f}' for _ in range(2)))
p('== C1 components ==')
p('p(x): ' + str([(round(w,4), m, round(v,5)) for w,m,v in comps_uncond()]))
for y in [1.0, 0.5]:
    c = comps_tilt(y, 1.0)
    p(f'p(x|y={y}): weights ' + ', '.join(f'{w:.4f}' for w,_,_ in c) + ' means ' + ', '.join(f'{m:.4f}' for _,m,_ in c) + f' var {c[0][2]:.6f}')
p('== C1 samples (N=2000, seed 1, K=100): mean / std / frac(x>0.7) ; exact tilted p(x)N(y;x,sy^2)^w : mean / std / frac ==')
for y in [1.0]:
    for om in [0, 1, 1.4, 2, 4]:
        ex = mix_stats(comps_tilt(y, om))
        for al in [1.0, 0.5, 0.0]:
            x = c1_sample(om, al, y)
            s = stats(x)
            p(f'y={y} w={om} a={al}: {s[0]:.4f} {s[1]:.4f} {s[2]:.4f} | exact {ex[0]:.4f} {ex[1]:.4f} {ex[2]:.4f} | x[0]={x[0]:.5f}')
for (y, om, al) in [(0.5, 1.4, 0.5), (0.5, 1.0, 1.0), (0.75, 1.4, 0.5), (0.0, 2.0, 0.5)]:
    ex = mix_stats(comps_tilt(y, om)); x = c1_sample(om, al, y); s = stats(x)
    p(f'y={y} w={om} a={al}: {s[0]:.4f} {s[1]:.4f} {s[2]:.4f} | exact {ex[0]:.4f} {ex[1]:.4f} {ex[2]:.4f} | x[0]={x[0]:.5f}')
p('== C1 range / histogram density (grid w in {0,1,2,4,5}, a in {0.05,0.5,1}, y in {0,0.5,1}) ==')
mx = 0; dm = {}
for om in [0, 1, 2, 4, 5]:
    for al in [1.0, 0.5, 0.05]:
        for y in [0.0, 0.5, 1.0]:
            x = c1_sample(om, al, y); mx = max(mx, float(abs(x).max()))
            h, _ = np.histogram(x, bins=64, range=(-0.2, 1.4), density=True); dm[al] = max(dm.get(al, 0), float(h.max()))
p(f'max |x| {mx:.4f}; max bar density by alpha: ' + ', '.join(f'a={a}: {v:.2f}' for a, v in dm.items()))
p('== C2 data ==')
C1d = disk_pts(); C2d = ann_pts(0.7)
p(f'disk pt0 ({C1d[0,0]:.5f},{C1d[0,1]:.5f}) max rho {np.sqrt((C1d**2).sum(1)).max():.4f}')
p(f'annulus r=0.7 pt0 ({C2d[0,0]:.5f},{C2d[0,1]:.5f}) min rho {np.sqrt((C2d**2).sum(1)).min():.4f} max rho {np.sqrt((C2d**2).sum(1)).max():.4f}')
p('== C2 samples (N=500, seed 21, alpha=0.5, h=0.05, K=100) ==')
runs = [('y1',1.2,0.7),('y2',1.2,0.7),('and',1.0,0.7),('and',1.2,0.7),('and',2.0,0.7),('not2',1.2,0.7),
        ('and',1.2,0.5),('and',1.2,0.9),('and',1.2,1.1),('and',1.2,1.3),('not2',1.2,0.5),('and',0.5,0.7),('and',3.0,0.7)]
for mode, om, rin in runs:
    X = c2_sample(mode, om, rin); q = c2_rates(X, rin)
    p(f'{mode:4s} w={om} r={rin}: le_R {q["le_R"]:.3f} ge_r {q["ge_r"]:.3f} both {q["both"]:.3f} lt_r {q["lt_r"]:.3f} rho_mean {q["rho_mean"]:.4f} rho_max {q["rho_max"]:.4f} X0=({X[0,0]:.4f},{X[0,1]:.4f})')
p(tables.report())
open('check_out.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
