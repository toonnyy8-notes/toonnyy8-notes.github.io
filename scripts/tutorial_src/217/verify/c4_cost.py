"""C4 acceptance: image-token cost model, Tab. 3 iso-cost window, fitted overhead."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import rel_cost

def brentq(f, lo, hi):
    for _ in range(200):
        m = (lo + hi) / 2
        if (f(lo) < 0) == (f(m) < 0): lo = m
        else: hi = m
    return (lo + hi) / 2
def fit(cfg, pts):
    best = None
    for ia in range(0, 1001):
        a = ia / 1000
        for it in range(0, 301):
            tau = it / 1000
            e = sum((1 / rel_cost(R, *cfg, a=a, tau=tau) - s) ** 2 for R, s in pts.items())
            if best is None or e < best[0]: best = (e, a, tau)
    return best
B16 = (17, (3, 13)); L16 = (23, (3, 19)); B16L = (17, (1, 15))
# Tab. 3 / B.5 checks (image tokens only, tau=0)
ref = rel_cost(256, *B16, a=0)
print("linear-only  [1,15]/[3,13] @256:", rel_cost(256, *B16L, a=0) / ref)
ref2 = rel_cost(256, *B16, a=1)
print("attention-only ratio:", rel_cost(256, *B16L, a=1) / ref2)
for a in (0, 1):
    r = rel_cost(256, *B16, a=a)
    print(f"iso R (a={a}):", brentq(lambda R: rel_cost(R, *B16L, a=a) - r, 1, 1024))
# ceilings
for nm, cfg in (("B16", B16), ("L16", L16), ("B16[1,15]", B16L)):
    print(nm, "ceiling (R->0, tau=0, any a):", 1 / rel_cost(0, *cfg, a=0))
# fit (a, tau) to measured speedups
meas = {"B16": (B16, {512: 1.84, 256: 2.15}), "L16": (L16, {512: 2.11, 256: 2.59, 128: 2.86, 64: 3.01})}
for nm, (cfg, pts) in meas.items():
    e, a, tau = fit(cfg, pts)
    print(nm, f"fit a={a:.3f} tau={tau:.3f} rmse={(e/len(pts))**.5:.4f}", {R: round(1 / rel_cost(R, *cfg, a=a, tau=tau), 3) for R in (1024, 512, 384, 256, 192, 128, 64, 16, 0)})
# matched-time steps (B.3, L/16)
for sec in (1.8, 2.8, 5.7, 11.2):
    print(sec, "dense steps", int(sec / 0.288 + 1e-9), "RTI steps", int(sec / 0.112 + 1e-9))
print("step cost ratio 288/112 =", 288 / 112)
for nd, nr in ((6, 16), (10, 25), (20, 50), (39, 100)):
    print(nd, nr, round(nd * .288, 3), round(nr * .112, 3), f"diff {abs(nd*.288-nr*.112)/(nr*.112)*100:.1f}%")
# L/16 band at tau=0 vs measured
for R in (512, 256, 128, 64):
    print("L16 R", R, "a=0", round(1/rel_cost(R, *L16, a=0), 3), "a=1", round(1/rel_cost(R, *L16, a=1), 3))
for R in (512, 256):
    print("B16 R", R, "a=0", round(1/rel_cost(R, *B16, a=0), 3), "a=1", round(1/rel_cost(R, *B16, a=1), 3))
