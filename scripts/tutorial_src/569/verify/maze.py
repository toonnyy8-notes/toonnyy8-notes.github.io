"""C3: tabular IQL on a toy u-maze (Sec. 5.1 / Fig. 2 analogue). Same algorithm as the page."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sys
from common import expectile, mulberry32

GRID = ["........",
        "........",
        "..####..",
        "..####..",
        "..####..",
        "..####..",
        "G.####.S"]
H_ROWS, W_COLS = len(GRID), len(GRID[0])
CELLS = [(r, c) for r in range(H_ROWS) for c in range(W_COLS) if GRID[r][c] != '#']
IDX = {rc: i for i, rc in enumerate(CELLS)}
GOAL = IDX[next(rc for rc in CELLS if GRID[rc[0]][rc[1]] == 'G')]
START = IDX[next(rc for rc in CELLS if GRID[rc[0]][rc[1]] == 'S')]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
NS = len(CELLS)
GAMMA, SLIP, R_GOAL = 0.9, 0.25, 10.0

def move(s, a):
    r, c = CELLS[s]; dr, dc = DIRS[a]
    rc = (r + dr, c + dc)
    return IDX.get(rc, s)

def true_P(s, a):
    """dict s'->prob: 0.75 commanded, 0.25 uniformly random direction (slip)."""
    out = {}
    for b in range(4):
        p = SLIP / 4 + ((1 - SLIP) if b == a else 0)
        s2 = move(s, a if False else b)
        out[s2] = out.get(s2, 0) + p
    return out

def value_iteration():
    V = [0.0] * NS
    for _ in range(5000):
        newV = [0.0] * NS; d = 0
        for s in range(NS):
            if s == GOAL: continue
            best = max(sum(p * ((R_GOAL if s2 == GOAL else 0) + GAMMA * (0 if s2 == GOAL else V[s2]))
                           for s2, p in true_P(s, a).items()) for a in range(4))
            newV[s] = best; d = max(d, abs(best - V[s]))
        V = newV
        if d < 1e-12: break
    return V

def q_true(V, s, a):
    return sum(p * ((R_GOAL if s2 == GOAL else 0) + GAMMA * (0 if s2 == GOAL else V[s2])) for s2, p in true_P(s, a).items())

def greedy_true(V):
    pol = []
    for s in range(NS):
        qs = [q_true(V, s, a) for a in range(4)]
        m = max(qs); pol.append(next(a for a in range(4) if qs[a] >= m - 1e-12))
    return pol

def step_env(s, a, rnd):
    if rnd() < SLIP:
        a = int(rnd() * 4)
    return move(s, a)

def make_dataset(seed, n_rand=99, horizon=100):
    """traj 0: optimal policy; traj 1..n_rand: uniform random actions. All start at START.
    PRNG call order per step: [random traj: action] -> slip test -> [if slip: slip direction]."""
    rnd = mulberry32(seed)
    pol = greedy_true(value_iteration())
    N = {}  # (s,a) -> {s2: count}
    for k in range(1 + n_rand):
        s = START
        for t in range(horizon):
            a = pol[s] if k == 0 else int(rnd() * 4)
            s2 = step_env(s, a, rnd)
            d = N.setdefault((s, a), {}); d[s2] = d.get(s2, 0) + 1
            if s2 == GOAL: break
            s = s2
    return N

def iql_tabular(N, tau, tol=1e-10, maxit=5000):
    """V(s)=E^tau_{a~mu_hat(.|s)}[Q(s,a)], Q(s,a)=sum_s' Nhat(s'|s,a)[r+gamma V(s')]. Only dataset (s,a)."""
    acts = {}
    for (s, a), d in N.items():
        acts.setdefault(s, []).append((a, sum(d.values())))
    V = [0.0] * NS
    Q = {}
    for it in range(maxit):
        for (s, a), d in N.items():
            n = sum(d.values())
            Q[(s, a)] = sum(c / n * ((R_GOAL if s2 == GOAL else 0) + GAMMA * (0 if s2 == GOAL else V[s2])) for s2, c in d.items())
        newV = V[:]; dmax = 0
        for s, lst in acts.items():
            m = expectile([Q[(s, a)] for a, _ in lst], [n for _, n in lst], tau)
            dmax = max(dmax, abs(m - V[s])); newV[s] = m
        V = newV
        if dmax < tol: break
    return V, Q, acts, it + 1

def eval_policy(pol):
    V = [0.0] * NS
    for _ in range(5000):
        d = 0; newV = [0.0] * NS
        for s in range(NS):
            if s == GOAL: continue
            a = pol[s]
            if a is None:
                v = sum(q_true(V, s, b) for b in range(4)) / 4
            else:
                v = q_true(V, s, a)
            newV[s] = v; d = max(d, abs(v - V[s]))
        V = newV
        if d < 1e-12: break
    return V

def extract(Q, acts):
    pol = [None] * NS
    for s, lst in acts.items():
        best = max(Q[(s, a)] for a, _ in lst)
        pol[s] = min(a for a, _ in lst if Q[(s, a)] >= best - 1e-12)
    return pol

if __name__ == '__main__':
    seeds = [int(x) for x in sys.argv[1:]] or [1]
    Vs = value_iteration()
    print('NS', NS, 'V*(S)=%.4f' % Vs[START])
    for seed in seeds:
        N = make_dataset(seed)
        visited = {s for (s, a) in N}
        nsa = sum(sum(d.values()) for d in N.values())
        print('seed', seed, 'visited states', len(visited), 'of', NS - 1, 'transitions', nsa, 'pairs', len(N))
        for tau in [0.5, 0.7, 0.9, 0.95, 0.99, 1.0]:
            V, Q, acts, it = iql_tabular(N, tau)
            pol = extract(Q, acts)
            Vp = eval_policy(pol)
            # value along a fixed probe set of cells
            probe = [(6, 1), (0, 0), (0, 4), (0, 7), (6, 7)]
            vals = ' '.join('%.3f' % V[IDX[p]] for p in probe)
            print(' tau=%.2f it=%4d V(S)=%.4f probe[%s]  greedy-policy true return from S=%.4f (V*=%.4f)' % (tau, it, V[START], vals, Vp[START], Vs[START]))
        print(' V* probe', ' '.join('%.3f' % Vs[IDX[p]] for p in [(6, 1), (0, 0), (0, 4), (0, 7), (6, 7)]))
