"""C3 (grouping comparator, clean 2-image batch) and C4 (restoration on the landscape) expected outputs.
Run: python3 -I c3_group.py"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

METHODS = ['grid', 'tome', 'rti', 'thresh', 'hit']
def group(method, R):
    if method == 'grid': return m_grid(R), m_grid(R), None
    if method == 'tome': return m_tome(BATCH[0], R), m_tome(BATCH[1], R), None
    if method == 'rti': return m_rti(BATCH[0], R), m_rti(BATCH[1], R), None
    if method == 'thresh': return thresh_batch(R)
    return linkage_batch(R)

t0 = time.time()
linkage_batch(32)
import common
for b, den in enumerate(common._DEN):
    sc = [m[2] for m in den]
    assert all(sc[i] >= sc[i + 1] - 1e-12 for i in range(len(sc) - 1)), 'inversion'
print('dendrograms monotone (no inversions): ok')
print()
print('C3  method  R   img        K    EV     spread conn   max   (last kept gap u* / merge score tau*)')
LAB = {}
for R in (16, 32, 64, 128):
    for m in METHODS:
        ll, lc, thr = group(m, R)
        for name, h, lab in (('landscape', BATCH[0], ll), ('checker', BATCH[1], lc)):
            LAB[(m, R, name)] = lab
            s = stats(h, lab)
            extra = f'  {thr:.6f}' if (thr is not None and name == 'landscape') else ''
            print(f'    {m:6s} {R:4d} {name:9s} {s["K"]:4d} {s["EV"]:.3f} {s["spread"]:6.3f} {s["conn"]:.3f} {s["maxsize"]:4d}{extra}')
    print()
print('C4 (landscape grouping from C3; y = clean, h = x_t features)')
print('    method  R    t     E0     E1     beta*  Ebeta*')
for R in (32, 64):
    for m in METHODS:
        lab = LAB[(m, R, 'landscape')]
        for t in (1.0, 0.98, 0.95, 0.9, 0.8, 0.5, 0.0):
            e0, e1, bs, eb = restore(H1, feats_t(t), lab)
            print(f'    {m:6s} {R:4d} {t:4.2f} {e0:.6f} {e1:.6f} {bs:7.4f} {eb:.6f}')
    print()
print('elapsed', round(time.time() - t0, 1), 's')
