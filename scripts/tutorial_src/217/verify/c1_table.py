"""C1 acceptance rows (n, L, p) -> components / spread per order, and connected fraction over all starts."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
rows = [(8, 8, 4), (8, 8, 0), (8, 4, 6), (16, 16, 40), (32, 16, 300), (32, 64, 100)]
for n, L, p in rows:
    out = []
    for nm in ('hilbert', 'morton', 'raster'):
        o = order(nm, n)
        c = o[p:p + L]
        frac = sum(components(o[s:s + L]) == 1 for s in range(n * n - L + 1)) / (n * n - L + 1)
        out.append(f"{nm}: comp={components(c)} spread={spread(c):.3f} frac={frac:.3f}")
    print(f"n={n} L={L} p={p} | " + " | ".join(out))
