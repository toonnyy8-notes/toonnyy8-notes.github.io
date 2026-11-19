"""C3 'population' mode: true transitions, uniform behavior policy (support = all 4 actions)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maze import *
N = {}
for s in range(NS):
    if s == GOAL: continue
    for a in range(4):
        N[(s, a)] = {s2: p for s2, p in true_P(s, a).items()}  # weights used as 'counts'
Vs = value_iteration()
prev = -1
for tau in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 1.0]:
    V, Q, acts, it = iql_tabular(N, tau)
    assert V[START] >= prev - 1e-12; prev = V[START]
    probe = [(6, 1), (0, 0), (0, 4), (0, 7), (6, 7)]
    print('tau=%.3f it=%4d V(S)=%.4f probe[%s]' % (tau, it, V[START], ' '.join('%.3f' % V[IDX[p]] for p in probe)))
print('V*(S)=%.4f' % Vs[START], 'max|V_1 - V*| =', max(abs(a - b) for a, b in zip(V, Vs)))
