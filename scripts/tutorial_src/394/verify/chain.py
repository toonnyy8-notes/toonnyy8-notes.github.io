"""W3 toy chain sampler: same algorithm as the page (C3).  python3 chain.py"""
import sys
from core import *

def run_chain(kind, sched, T, p, L=32, Ktok=8, emax=0.6, emin=0.05, seed=1):
    """kind 'mask'|'unif'. Returns final error fraction (positions != x*)."""
    K = Ktok + 1 if kind == 'mask' else Ktok
    pi = prior(kind, K); m = K - 1
    r = Mulberry32(seed)
    xs = [l % Ktok for l in range(L)]
    z = [m] * L if kind == 'mask' else [cat_sample(r, pi) for _ in range(L)]
    for i in range(T, 0, -1):
        t = i / T; s = (i - 1) / T; at = 1 - t; as_ = 1 - s
        c = sum(1 for l in range(L) if z[l] == xs[l]) / L
        eps = emax - (emax - emin) * c
        kap = kappa_sched(sched, t, at, as_, p)
        nz = []
        for l in range(L):
            if kind == 'mask' and z[l] != m:
                xth = onehot(z[l], K)                      # carry-over (SUBS)
            else:
                xth = [eps / (Ktok - 1)] * Ktok + ([0.0] if kind == 'mask' else [])
                xth[xs[l]] = 1 - eps
            nz.append(cat_sample(r, psi_theta(z[l], xth, at, as_, kap, pi)))
        z = nz
    return sum(1 for l in range(L) if z[l] != xs[l]) / L

def err_rate(kind, sched, T, p, R=64, **kw):
    return sum(run_chain(kind, sched, T, p, seed=1000 + rr, **kw) for rr in range(R)) / R

if __name__ == '__main__':
    p = dict(kap=float(sys.argv[1]) if len(sys.argv) > 1 else 0.95, ton=0.6, toff=0.1, eta=0.05)
    Ts = [8, 16, 32, 64, 128, 256]
    for kind in ('mask', 'unif'):
        for sched in ('ancestral', 'window', 'rescale'):
            print(kind, sched, ['%.3f' % err_rate(kind, sched, T, p) for T in Ts])
