"""C3 extension: policy extraction with fixed Gaussian 'approximation error' added to Q (page simplification)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math, sys
from common import mulberry32
from maze import *

def noise_table(seed):
    rnd = mulberry32(seed + 1000)
    xi = {}
    for s in range(NS):
        for a in range(4):
            u1, u2 = rnd(), rnd()
            xi[(s, a)] = math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)
    return xi

def extract_noisy(Q, acts, xi, sigma):
    pol = [None] * NS
    for s, lst in acts.items():
        qn = {a: Q[(s, a)] + sigma * xi[(s, a)] for a, _ in lst}
        best = max(qn.values())
        pol[s] = min(a for a in qn if qn[a] >= best - 1e-12)
    return pol

def seed_table():
    print('seed table: V^pi(S) at sigma 0 / 0.01 for tau 0.5 and 0.9; V_tau(S) at tau .5/.95')
    for seed in range(1, 6):
        N = make_dataset(seed); xi = noise_table(seed); out = []
        for tau in [0.5, 0.9]:
            V, Q, acts, it = iql_tabular(N, tau)
            out += ['%.3f' % eval_policy(extract_noisy(Q, acts, xi, s))[START] for s in (0, 0.01)]
        v5 = iql_tabular(N, 0.5)[0][START]; v95 = iql_tabular(N, 0.95)[0][START]
        print('  seed %d  tau.5: %s / %s   tau.9: %s / %s   V.5(S)=%.4f V.95(S)=%.4f' % ((seed,) + tuple(out) + (v5, v95)))

if __name__ == '__main__':
    seed_table()
    Vs = value_iteration()
    for seed in [int(x) for x in sys.argv[1:]] or [1]:
        N = make_dataset(seed); xi = noise_table(seed)
        print('seed', seed)
        for tau in [0.5, 0.7, 0.9, 0.95, 0.99, 1.0]:
            V, Q, acts, it = iql_tabular(N, tau)
            row = []
            for sigma in [0, 0.001, 0.01, 0.03, 0.1]:
                Vp = eval_policy(extract_noisy(Q, acts, xi, sigma))
                row.append('%.3f' % Vp[START])
            print('  tau=%.2f  V^pi(S) for sigma 0/.001/.01/.03/.1: %s' % (tau, ' '.join(row)))
