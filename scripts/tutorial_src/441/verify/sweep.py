"""Full slider-grid stability sweep for C1 and C2 (same algorithm as page). Output: sweep_out.txt"""
import sys, math, time; sys.path.insert(0, '.')
from core import *
out = []
Z1 = draws_1d(1, NP1)
assert np.allclose(c1_sample_fast(1.4, 0.5, 1.0, Z1), c1_sample(1.4, 0.5, 1.0)), 'fast C1 != reference'
assert np.allclose(c1_sample_fast(1.4, 0.5, 1.0, Z1)[:500], c1_sample_fast(1.4, 0.5, 1.0, Z1[:, :500])), 'subset property'
out.append('C1 fast == reference; first 500 of N=2000 identical to an N=500 run (step-major draws, independent particles)')
mx = 0; arg = None; bad = 0; n = 0
for io in range(51):
    om = io / 10
    for ia in range(21):
        al = ia / 20
        for iy in range(21):
            y = iy / 20
            x = c1_sample_fast(om, al, y, Z1); n += 1
            if not np.all(np.isfinite(x)): bad += 1; continue
            m = float(abs(x).max())
            if m > mx: mx, arg = m, (om, al, y)
out.append(f'C1 grid omega 0..5/0.1 x alpha 0..1/0.05 x y 0..1/0.05 ({n} runs): non-finite runs {bad}; max|x| {mx:.4f} at (omega, alpha, y) = {arg}')
Z2 = draws_2d(21, NP2)
assert np.allclose(c2_sample_fast('and', 1.2, 0.7, Z2), c2_sample('and', 1.2, 0.7)), 'fast C2 != reference'
X0 = np.array([[0.5, 0.5], [0.0, 0.0], [1.2, 0.0]])
for k in [100, 50, 10, 1]:
    out.append(f'C2 g1 (r=0.7) at k={k}, x=(0.5,0.5),(0,0),(1.2,0): ' + ', '.join(f'{v:.6f}' for v in g1_at(X0, 0.7, k)))
t = time.time(); mx = 0; arg = None; bad = 0; n = 0; rmax = {}
for mode in ['y1', 'y2', 'and', 'not2']:
    for io in range(26):
        om = round(0.5 + io / 10, 1)
        for ir in range(11):
            rin = round(0.3 + ir / 10, 1)
            X = c2_sample_fast(mode, om, rin, Z2); n += 1
            if not np.all(np.isfinite(X)): bad += 1; continue
            m = float(np.sqrt((X ** 2).sum(1)).max())
            rmax[mode] = max(rmax.get(mode, 0), m)
            if m > mx: mx, arg = m, (mode, om, rin)
out.append(f'C2 grid 4 modes x omega 0.5..3/0.1 x r 0.3..1.3/0.1 ({n} runs, {time.time()-t:.0f}s): non-finite {bad}; max radius {mx:.4f} at {arg}; per mode ' + ', '.join(f'{k} {v:.4f}' for k, v in rmax.items()))
open('sweep_out.txt', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
