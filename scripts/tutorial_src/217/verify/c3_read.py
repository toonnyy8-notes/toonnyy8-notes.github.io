"""C3 acceptance: READ soft-pool weights and averaged rotary phasors on Hilbert runs."""
import math
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
img = scene('landscape'); det = detail(img)
logd = [math.log(e + 1e-4) for e in det]
mu = sum(logd) / N; sd = (sum((v - mu) ** 2 for v in logd) / N) ** 0.5
score = [(v - mu) / sd for v in logd]
W = rope_freqs()
print("wavelengths:", [round(2 * math.pi / w, 2) for w in W])
def read(p, L, beta):
    pids = HIL_PID[p:p + L]
    cells = [(q % G, q // G) for q in pids]
    s = [beta * score[q] for q in pids]; m = max(s)
    ex = [math.exp(v - m) for v in s]; Z = sum(ex); al = [e / Z for e in ex]
    cx = sum(a * c[0] for a, c in zip(al, cells)); cy = sum(a * c[1] for a, c in zip(al, cells))
    uni = [1 / L] * L
    mx = [phasor_mag([c[0] for c in cells], uni, w) for w in W]
    my = [phasor_mag([c[1] for c in cells], uni, w) for w in W]
    return dict(size_idx=int(math.floor(math.log2(L))), cx=cx, cy=cy, spread=spread(cells, al),
                pr=1 / sum(a * a for a in al), amax=max(al),
                kx=sum(v >= 0.9 for v in mx), ky=sum(v >= 0.9 for v in my),
                mx0=mx[0], my0=my[0], mx3=mx[3], my3=my[3])
for p, L, beta in ((0, 1, 1), (0, 2, 1), (0, 4, 1), (0, 16, 1), (0, 64, 1), (0, 256, 1), (100, 37, 0), (100, 37, 1), (100, 37, 4), (512, 512, 1)):
    r = read(p, L, beta)
    print(f"p={p} L={L} beta={beta}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
# closed-form check: L=2 at p=0 -> cells (0,0),(0,1): |m_y| = |cos(w/2)| = dirichlet(w,2)
print("dirichlet check", max(abs(phasor_mag([0, 1], [.5, .5], w) - dirichlet(w, 2)) for w in W))
print("row run L=8 check", max(abs(phasor_mag(list(range(8)), [1/8]*8, w) - dirichlet(w, 8)) for w in W))
