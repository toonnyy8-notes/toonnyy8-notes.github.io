"""C2: particle iteration of the flow Bellman operator T_tau vs the clean operator on a 2-pair toy chain.
Toy (all values 推定): pairs x0, x1 (fixed policy, so each pair is an (s,a)); rewards R; transition P.
Representation: N_AT equally weighted sorted atoms per pair.
One step for pair x: mixture over (x', j) with weight P[x][x']/N_AT of
   tau*R[x] + gamma*Z[x'][j] + (1-tau)*c*g   (corrected; g ~ N(0,1) from mulberry32, shared across runs)
   R[x] + gamma*Z[x'][j]                     (clean)
then exact W1 of the un-projected mixtures (for the contraction ratio), then quantile projection back to N_AT atoms
(atom k = smallest atom whose cumulative weight >= (k+0.5)/N_AT)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
from common import mulberry32, gauss, w1_sorted, w1_weighted, inv_norm

R = [0.0, -1.0]
P = [[0.7, 0.3], [0.4, 0.6]]
N_AT = 400

def init(kind):
    if kind == 'zero':
        return [[0.0] * N_AT for _ in range(2)]
    # 'spread': x0 uniform grid on [-10, 0], x1 uniform grid on [-2, 6]
    lo, width = [-10.0, -2.0], [10.0, 8.0]
    return [[lo[x] + width[x] * (k + 0.5) / N_AT for k in range(N_AT)] for x in range(2)]

def noise_table(seed, it):
    """noise for iteration it: array [x][x'][j]; one mulberry32 stream per iteration, seed*1000+it."""
    rnd = mulberry32(seed * 1000 + it)
    return [[[gauss(rnd) for j in range(N_AT)] for xp in range(2)] for x in range(2)]

def step(Z, gamma, tau, corrected, G):
    c = math.sqrt(1 - gamma * gamma)
    mix = []
    for x in range(2):
        xs = []; ws = []
        for xp in range(2):
            w = P[x][xp] / N_AT
            for j in range(N_AT):
                if corrected:
                    v = tau * R[x] + gamma * Z[xp][j] + (1 - tau) * c * G[x][xp][j]
                else:
                    v = R[x] + gamma * Z[xp][j]
                xs.append(v); ws.append(w)
        mix.append((xs, ws))
    return mix

def project(xs, ws):
    pts = sorted(zip(xs, ws)); out = []; cum = 0.0; i = 0
    for k in range(N_AT):
        q = (k + 0.5) / N_AT
        while i < len(pts) - 1 and cum + pts[i][1] < q - 1e-15:
            cum += pts[i][1]; i += 1
        out.append(pts[i][0])
    return out

def moments(gamma):
    """exact mean Q and variance of clean Z via linear systems (2x2)."""
    def solve(A, b):
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        return [(b[0] * A[1][1] - A[0][1] * b[1]) / det, (A[0][0] * b[1] - b[0] * A[1][0]) / det]
    A = [[1 - gamma * P[0][0], -gamma * P[0][1]], [-gamma * P[1][0], 1 - gamma * P[1][1]]]
    Q = solve(A, R)
    PQ = [P[x][0] * Q[0] + P[x][1] * Q[1] for x in range(2)]
    A2 = [[1 - gamma ** 2 * P[0][0], -gamma ** 2 * P[0][1]], [-gamma ** 2 * P[1][0], 1 - gamma ** 2 * P[1][1]]]
    M2 = solve(A2, [R[x] ** 2 + 2 * gamma * R[x] * PQ[x] for x in range(2)])
    return Q, [M2[x] - Q[x] ** 2 for x in range(2)]

def run(gamma, tau, K, seed=1):
    """Returns per-iteration log + final particles. Runs: A ('zero' init) and B ('spread' init) for both operators."""
    Zc = {'A': init('zero'), 'B': init('spread')}; Zk = {'A': init('zero'), 'B': init('spread')}
    log = []
    for it in range(1, K + 1):
        G = noise_table(seed, it)
        row = {}
        for name, Z, corr in [('corr', Zc, True), ('clean', Zk, False)]:
            before = max(w1_sorted(Z['A'][x], Z['B'][x]) for x in range(2))
            mA = step(Z['A'], gamma, tau, corr, G); mB = step(Z['B'], gamma, tau, corr, G)
            after = max(w1_weighted(mA[x][0], mA[x][1], mB[x][0], mB[x][1]) for x in range(2))
            Z['A'] = [project(*mA[x]) for x in range(2)]; Z['B'] = [project(*mB[x]) for x in range(2)]
            row[name] = (before, after, after / before if before > 0 else float('nan'))
        log.append(row)
    return log, Zc, Zk

def stats(a):
    m = sum(a) / len(a); return m, math.sqrt(sum((v - m) ** 2 for v in a) / len(a))

def reference(gamma, tau, Zclean_tau1, seed=7):
    """sample of tau*Z + (1-tau)*Nbar using converged clean (tau=1) particles and independent N(0,1) from mulberry32(seed)."""
    rnd = mulberry32(seed)
    return [sorted(tau * z + (1 - tau) * gauss(rnd) for z in Zclean_tau1[x]) for x in range(2)]

if __name__ == '__main__':
    for gamma in [0.9, 0.7]:
        Q, Var = moments(gamma)
        print(f'gamma={gamma}: exact Q={Q[0]:.4f},{Q[1]:.4f}  Var(Z)={Var[0]:.4f},{Var[1]:.4f}  sd={math.sqrt(Var[0]):.4f},{math.sqrt(Var[1]):.4f}')
        _, _, Zref = run(gamma, 1.0, 120)   # clean op at tau=1 == corrected at tau=1
        for tau in [0.0, 0.5, 1.0]:
            K = 60
            log, Zc, Zk = run(gamma, tau, K)
            maxratio = max(max(r['corr'][2], r['clean'][2]) for r in log)
            ref = reference(gamma, tau, Zref['A'])
            out = []
            for x in range(2):
                mc, sc = stats(Zc['A'][x]); mk, sk = stats(Zk['A'][x])
                ex_m = tau * Q[x]; ex_s = math.sqrt(tau ** 2 * Var[x] + (1 - tau) ** 2)
                out.append(f'x{x}: corr mean={mc:.4f} sd={sc:.4f} (exact {ex_m:.4f},{ex_s:.4f}) W1toRef={w1_sorted(Zc["A"][x], ref[x]):.4f} | clean mean={mk:.4f} sd={sk:.4f} (exact {Q[x]:.4f},{math.sqrt(Var[x]):.4f})')
            print(f' tau={tau} K={K}: max ratio={maxratio:.6f} (<= gamma {gamma}); ratio k=1,2,3,10 corr:',
                  ', '.join(f'{log[k-1]["corr"][2]:.4f}' for k in [1, 2, 3, 10]),
                  '; W1sup before step k=1,10,30,60 corr:', ', '.join(f'{log[k-1]["corr"][0]:.4g}' for k in [1, 10, 30, 60]),
                  '; clean:', ', '.join(f'{log[k-1]["clean"][0]:.4g}' for k in [1, 10, 30, 60]))
            for o in out: print('   ', o)
            assert maxratio <= gamma + 1e-9
            if tau == 0.0:
                # corrected tau=0 fixed point is N(0,1): W1 to N(0,1) quantile atoms
                qn = [inv_norm((k + 0.5) / N_AT) for k in range(N_AT)]
                print('    tau=0 corr W1 to N(0,1) quantile atoms:', [round(w1_sorted(Zc['A'][x], qn), 4) for x in range(2)])
    G = noise_table(1, 1)
    print('noise table k=1: total', 2 * 2 * N_AT, 'first g (pair1,next pair1,j=0)=', repr(G[0][0][0]), ' g[pair1][next2][0]=', repr(G[0][1][0]), ' g[pair2][next1][0]=', repr(G[1][0][0]))
    print('gamma index check: gamma=i/20 for i=10..19 ->', [i / 20 for i in range(10, 20)])
    print('c2 ok')
