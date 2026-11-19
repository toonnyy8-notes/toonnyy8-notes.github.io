"""C3: Diffusion-QL loss (Eq. 3) on the bandit toy. Policy = Eq. (1) chain driven by the ideal denoiser of a
4-component mixture with trainable logits (component means/sd fixed at the data's). Analytic gradients, Adam."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from core import *

B = 128; LR = 0.05; B1, B2, AEPS = 0.9, 0.999, 1e-8

def batch_draws(r, N):
    k = np.zeros(B, int); xn = np.zeros((B, 2)); ii = np.zeros(B, int); ep = np.zeros((B, 2))
    for b in range(B):
        k[b] = min(3, int(4 * r())); xn[b] = [gauss(r), gauss(r)]
        ii[b] = 1 + min(N - 1, int(N * r())); ep[b] = [gauss(r), gauss(r)]
    aN = np.zeros((B, 2)); Z = {i: np.zeros((B, 2)) for i in range(2, N + 1)}
    for b in range(B):
        aN[b] = [gauss(r), gauss(r)]
        for i in range(N, 1, -1):
            Z[i][b] = [gauss(r), gauss(r)]
    return k, xn, ii, ep, aN, Z

def loss_grad(lg, sch, N, eta, draws):
    k, xn, ii, ep, aN, Z = draws
    beta, alpha, abar = sch
    a = MU[k] + SD * xn
    Ld = 0.0; gLd = np.zeros(4)
    for i in range(1, N + 1):              # group by diffusion step (same maths as per-sample loop)
        m = ii == i
        if not m.any(): continue
        x = math.sqrt(abar[i - 1]) * a[m] + math.sqrt(1 - abar[i - 1]) * ep[m]
        e, _, Jl = eps_star(x, i, sch, lg, True)
        res = ep[m] - e
        Ld += (res ** 2).sum(); gLd += -2 * np.einsum('bi,bik->k', res, Jl)
    Ld /= B; gLd /= B
    a0, T = sample_chain(aN, Z, sch, lg, True)
    q, gq = q_toy(a0, True)
    al = eta / QNORM
    Lq = -al * q.mean(); gLq = -al * np.einsum('bi,bik->k', gq, T) / B
    return Ld, Lq, gLd + gLq, q.mean()

def train(N, eta, steps, seed=7, log=(100, 300, 1000)):
    r = mulberry32(seed); sch = schedule(N)
    lg = np.zeros(4); m = np.zeros(4); v = np.zeros(4); rec = {}
    for t in range(1, steps + 1):
        Ld, Lq, g, qm = loss_grad(lg, sch, N, eta, batch_draws(r, N))
        m = B1 * m + (1 - B1) * g; v = B2 * v + (1 - B2) * g * g
        lg = lg - LR * (m / (1 - B1 ** t)) / (np.sqrt(v / (1 - B2 ** t)) + AEPS)
        w = np.exp(lg - lg.max()); w /= w.sum()
        if t > steps - 50:
            wavg = w / 50 if t == steps - 49 else wavg + w / 50
        if t in log:
            rec[t] = (w.copy(), Ld, Lq, qm)
    rec['avg50'] = wavg
    return lg, rec

if __name__ == '__main__':
    import itertools
    Ns = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [5]
    etas = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0, 0.1, 0.3, 1, 3]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    for N, eta in itertools.product(Ns, etas):
        lg, rec = train(N, eta, steps, log=tuple(x for x in (100, 300, 600, 1000, 2000) if x <= steps))
        for t, (w, Ld, Lq, qm) in rec.items():
            print('N=%d eta=%.2f step=%4d w(top,right,bottom,left)=%s Ld=%.4f Lq=%.4f EQ=%.3f' % (
                N, eta, t, np.array2string(w, precision=3), Ld, Lq, qm))

def evaluate(lg, N, seed=11):
    from c2 import draws_diffusion, metrics
    aN, Z = draws_diffusion(seed, N)
    A = sample_chain(aN, Z, schedule(N), lg)
    near, centre, out, share = metrics(A)
    return near, share[2], q_toy(A).mean(), A
