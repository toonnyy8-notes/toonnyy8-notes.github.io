"""C1 acceptance: run connectivity / spread for the three orders."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import math
for n in (8, 32):
    print(f"## grid {n}x{n}")
    for name in ('hilbert', 'morton', 'raster'):
        o = order(name, n); NN = n*n
        steps = [math.hypot(o[j+1][0]-o[j][0], o[j+1][1]-o[j][1]) for j in range(NN-1)]
        nonadj = sum(1 for s in steps if s != 1)
        row = [name, f"nonadj_steps={nonadj}", f"maxstep={max(steps):.3f}"]
        for L in (2, 4, 8, 16, 64):
            if L > NN: continue
            allc = sum(1 for s in range(NN-L+1) if components(o[s:s+L]) == 1) / (NN-L+1)
            al = [o[s:s+L] for s in range(0, NN, L)]
            alc = sum(1 for c in al if components(c) == 1) / len(al)
            sp = sum(spread(c) for c in al) / len(al)
            row.append(f"L={L}: all={allc:.4f} aligned={alc:.3f} spread={sp:.3f}")
        print(" | ".join(row))
    # dyadic cells are runs
    for name in ('hilbert', 'morton'):
        o = order(name, n); pos = {c: j for j, c in enumerate(o)}
        ok = True
        s = 2
        while s <= n:
            for by in range(0, n, s):
                for bx in range(0, n, s):
                    js = sorted(pos[(bx+i, by+k)] for i in range(s) for k in range(s))
                    if js[-1]-js[0] != s*s-1 or js[0] % (s*s): ok = False
            s *= 2
        print(name, "dyadic cells are aligned runs:", ok)
# Fig 14a example: positions 8..15 on 8x8 (run of 8 at R=128-equivalent mean size)
for name in ('hilbert','morton','raster'):
    o = order(name, 8); c = o[8:16]
    print(name, "run 8..15", c, "components", components(c), f"spread {spread(c):.3f}")
