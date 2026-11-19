"""Acceptance numbers for the CQL tutorial widgets (C1-C4).
Same algorithms the page uses; pure stdlib so it mirrors the JS line by line.
Run: python3 -I check.py > check_out.txt"""
import math

def norm(w):
    s = sum(w); return [x / s for x in w]

def dcql(pi, pb):                      # sum_a pi (pi/pb - 1)
    return sum(p * (p / b - 1) for p, b in zip(pi, pb))

def chi2(pi, pb):                      # sum_a (pi-pb)^2/pb
    return sum((p - b) ** 2 / b for p, b in zip(pi, pb))

# ---------------- C1: single-state bandit, gamma = 0 ----------------
R1 = [1.0, 0.6, 0.8]
def c1(wb, wp, alpha, r=R1):
    pb, pi = norm(wb), norm(wp)
    ratio = [p / b for p, b in zip(pi, pb)]
    q1 = [ri - alpha * x for ri, x in zip(r, ratio)]          # Eq.11, mu = pi
    q2 = [ri - alpha * (x - 1) for ri, x in zip(r, ratio)]    # Eq.13, mu = pi
    V = sum(p * ri for p, ri in zip(pi, r))
    V1 = sum(p * q for p, q in zip(pi, q1)); V2 = sum(p * q for p, q in zip(pi, q2))
    return dict(pb=pb, pi=pi, ratio=ratio, d1=[q - ri for q, ri in zip(q1, r)],
                d2=[q - ri for q, ri in zip(q2, r)], V=V, V1=V1, V2=V2,
                D=dcql(pi, pb), X=chi2(pi, pb))

# ---------------- C2: 3-state 2-action MDP ----------------
# T[s][a] = list of (s', prob); reward only at (s2, right) = 1
T = [[[(0, 1.0)], [(1, 1.0)]],
     [[(0, 1.0)], [(2, 0.8), (0, 0.2)]],
     [[(1, 1.0)], [(2, 1.0)]]]
REW = [[0, 0], [0, 0], [0, 1]]

def solve(A, b):                       # Gaussian elimination, partial pivoting
    n = len(b); M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(M[i][c])); M[c], M[p] = M[p], M[c]
        for i in range(n):
            if i != c:
                f = M[i][c] / M[c][c]
                for j in range(c, n + 1): M[i][j] -= f * M[c][j]
    return [M[i][n] / M[i][i] for i in range(n)]

def c2(pb_right, p_right, alpha, gamma, K=None):
    pol = [[1 - p_right, p_right]] * 3
    beh = [[1 - b, b] for b in pb_right]
    idx = lambda s, a: 2 * s + a
    # A = I - gamma P^pi on (s,a)
    A = [[0.0] * 6 for _ in range(6)]
    for s in range(3):
        for a in range(2):
            i = idx(s, a); A[i][i] += 1
            for s2, pr in T[s][a]:
                for a2 in range(2): A[i][idx(s2, a2)] -= gamma * pr * pol[s2][a2]
    r = [REW[s][a] for s in range(3) for a in range(2)]
    ratio = [pol[s][a] / beh[s][a] for s in range(3) for a in range(2)]
    Q = solve(A, r)
    Q1 = solve(A, [r[i] - alpha * ratio[i] for i in range(6)])
    Q2 = solve(A, [r[i] - alpha * (ratio[i] - 1) for i in range(6)])
    V = lambda q: [sum(pol[s][a] * q[idx(s, a)] for a in range(2)) for s in range(3)]
    D = [dcql(pol[s], beh[s]) for s in range(3)]
    out = dict(Q=Q, Q1=Q1, Q2=Q2, V=V(Q), V1=V(Q1), V2=V(Q2), D=D)
    if K is not None:                  # iterate Eq.11/13 from zero with exact B^pi
        def it(pen):
            q = [0.0] * 6; hist = []
            for k in range(K + 1):
                hist.append(V(q)[0])
                bq = [r[i] + gamma * sum(pr * sum(pol[s2][a2] * q[idx(s2, a2)] for a2 in range(2))
                                         for s2, pr in T[i // 2][i % 2]) for i in range(6)]
                q = [bq[i] - alpha * pen[i] for i in range(6)]
            return hist
        out['it0'] = it([0] * 6); out['it1'] = it(ratio); out['it2'] = it([x - 1 for x in ratio])
    return out

# ---------------- C3: sampling error (closed form, Gaussian reward noise) ----------------
def erfc(x):                           # Numerical Recipes erfcc, |rel err| < 1.2e-7
    z = abs(x); t = 1 / (1 + 0.5 * z)
    y = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
        t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
        t * (-0.82215223 + t * 0.17087277)))))))))
    return y if x >= 0 else 2 - y
ZD = {0.10: 1.2816, 0.05: 1.6449, 0.01: 2.3263}
def c3(wb, wp, N, sigma, alpha, delta=0.05):
    pb0, pi = norm(wb), norm(wp)
    n = [max(1, round(N * b)) for b in pb0]          # JS: Math.max(1, Math.round(...))
    pbh = norm(n)
    D = dcql(pi, pbh)
    s = sigma * math.sqrt(sum(p * p / k for p, k in zip(pi, n)))
    if D < 1e-12: return dict(n=n, D=D, s=s, P=0.5, adelta=math.inf)
    P = 0.5 * erfc(alpha * D / (s * math.sqrt(2)))
    return dict(n=n, D=D, s=s, P=P, adelta=ZD[delta] * s / D)

# ---------------- C4: CQL(H) bandit, damped Newton ----------------
R4 = [1.0, 0.6, 0.2, 1.4]
PB4 = {'rare': [0.45, 0.35, 0.15, 0.05], 'unif': [0.25] * 4, 'common': [0.10, 0.20, 0.20, 0.50]}
def lse(q):
    m = max(q); return m + math.log(sum(math.exp(x - m) for x in q))
def softmax(q):
    m = max(q); e = [math.exp(x - m) for x in q]; s = sum(e); return [x / s for x in e]
def f4(q, r, pb, al):
    return al * (lse(q) - sum(b * x for b, x in zip(pb, q))) + 0.5 * sum(b * (x - y) ** 2 for b, x, y in zip(pb, q, r))
def c4(pbkey, alpha, r=R4):
    pb = PB4[pbkey]; q = r[:]; it = 0
    for it in range(1, 51):
        p = softmax(q)
        g = [alpha * (pi - b) + b * (x - y) for pi, b, x, y in zip(p, pb, q, r)]
        if max(abs(x) for x in g) < 1e-9 * (1 + alpha): break
        H = [[alpha * ((p[i] if i == j else 0) - p[i] * p[j]) + (pb[i] if i == j else 0) for j in range(4)] for i in range(4)]
        d = solve(H, [-x for x in g]); t = 1.0; f0 = f4(q, r, pb, alpha); gd = sum(a * b for a, b in zip(g, d))
        while f4([x + t * y for x, y in zip(q, d)], r, pb, alpha) > f0 + 1e-4 * t * gd and t > 1e-10: t *= 0.5
        q = [x + t * y for x, y in zip(q, d)]
    mu = softmax(q)
    H = -sum(m * math.log(m) for m in mu if m > 0)
    E = lambda w, v: sum(a * b for a, b in zip(w, v))
    return dict(q=q, mu=mu, lse=lse(q), max=max(q), EmuH=E(mu, q) + H, iters=it,
                Epb_q=E(pb, q), Epb_r=E(pb, r), gap_hat=E(pb, q) - E(mu, q), gap_true=E(pb, r) - E(mu, r),
                chi=dcql(mu, pb))

def gap_reg(pbkey, alpha):            # E_s[lse(Q^) - E_pb Q^] at the CQL(H) solution (Eq. 30 bracket)
    o = c4(pbkey, alpha); return o['lse'] - o['Epb_q']
def c4tau(pbkey, tau, amax=1e4):       # Eq.30 saddle: alpha=0 if gap(0)<=tau; gap(alpha)=tau otherwise
    pb = PB4[pbkey]; Hb = -sum(b * math.log(b) for b in pb)
    if gap_reg(pbkey, 0.0) <= tau: return 0.0, Hb
    if tau <= Hb + 1e-9 or gap_reg(pbkey, amax) > tau: return math.inf, Hb
    lo, hi = math.log(1e-6), math.log(amax)          # bisection on log(alpha), 60 halvings
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if gap_reg(pbkey, math.exp(mid)) > tau: lo = mid
        else: hi = mid
    return math.exp(0.5 * (lo + hi)), Hb

if __name__ == '__main__':
    f = lambda v: '[' + ', '.join(f'{x:.4f}' for x in v) + ']'
    print('== C1 (r =', R1, ')')
    for wb, wp, al in [((50, 30, 20), (10, 20, 70), 0.5), ((50, 30, 20), (50, 30, 20), 0.5),
                       ((50, 30, 20), (10, 20, 70), 0.0), ((1, 1, 98), (10, 20, 70), 1.0), ((50, 50, 1), (90, 10, 0), 1.0)]:
        o = c1(wb, wp, al)
        print(wb, wp, 'a=', al, 'ratio', f(o['ratio']), 'dQ1', f(o['d1']), 'dQ2', f(o['d2']),
              f"V={o['V']:.4f} V1-V={o['V1']-o['V']:.4f} V2-V={o['V2']-o['V']:.4f} D={o['D']:.5f} chi2={o['X']:.5f}")
    print('advisor 2-action example D =', round(dcql([0.9, 0.1], [0.5, 0.5]), 6))
    print('\n== C2')
    for pbr, pr, al, g in [((0.8, 0.5, 0.2), 0.9, 0.1, 0.9), ((0.8, 0.5, 0.2), 0.9, 0.0, 0.9),
                           ((0.8, 0.5, 0.2), 0.9, 0.1, 0.5), ((0.5, 0.5, 0.5), 0.5, 0.1, 0.9), ((0.8, 0.5, 0.2), 0.9, 0.1, 0.95), ((0.8, 0.5, 0.2), 0.3, 0.1, 0.9)]:
        o = c2(pbr, pr, al, g, K=100)
        gap = [a - b for a, b in zip(o['V1'], o['V2'])]
        over = [i for i in range(6) if o['Q2'][i] > o['Q'][i] + 1e-12]
        print(pbr, pr, 'a=', al, 'g=', g, '\n  Q ', f(o['Q']), '\n  Q1', f(o['Q1']), '\n  Q2', f(o['Q2']),
              '\n  V ', f(o['V']), 'V1', f(o['V1']), 'V2', f(o['V2']), '\n  D', f(o['D']), 'V1-V2', f(gap),
              '-a/(1-g)=', round(-al / (1 - g), 4), 'overestimated (s,a) idx:', over,
              '\n  iter V(s0) k=0,1,5,20,100 :', [round(o['it2'][k], 4) for k in (0, 1, 5, 20, 100)],
              ' eq1:', [round(o['it1'][k], 4) for k in (0, 1, 5, 20, 100)], ' true:', [round(o['it0'][k], 4) for k in (0, 1, 5, 20, 100)])
    print('\n== C3 (pb=(50,30,20), pi=(10,20,70))')
    for N in (10, 30, 100, 300, 1000, 3000, 10000):
        for sg, al in [(1.0, 0.2)]:
            o = c3((50, 30, 20), (10, 20, 70), N, sg, al)
            print(f"N={N:5d} n={o['n']} D^={o['D']:.5f} s={o['s']:.5f} P(over)@a=0.2: {o['P']:.5f}  a_0.05={o['adelta']:.5f}  a_0.10={ZD[0.10]*o['s']/o['D']:.5f} a_0.01={ZD[0.01]*o['s']/o['D']:.5f}")
    o = c3((50, 30, 20), (10, 20, 70), 100, 1.0, 0.0); print('alpha=0 P =', o['P'])
    o = c3((50, 30, 20), (50, 30, 20), 100, 1.0, 0.5); print('pi = pb: D =', o['D'], 'P =', o['P'], 'adelta =', o['adelta'])
    o = c3((50, 30, 20), (10, 20, 70), 100, 2.0, 0.2); print('sigma=2 N=100 a=0.2 P =', round(o['P'], 5), 'a05', round(o['adelta'], 5))
    o = c3((1, 1, 98), (10, 20, 70), 10, 1.0, 0.2); print('pb=(1,1,98) N=10 n =', o['n'], 'D =', round(o['D'], 5), 'P =', round(o['P'], 5))
    o = c3((1, 1, 98), (1, 1, 10), 10, 1.0, 0.2); print('button pi := counts (1,1,10), N=10: D =', o['D'], 'P =', o['P'])
    o = c3((50, 30, 20), (10, 20, 70), 10, 2.0, 0.2, 0.01); print('sigma=2 N=10 delta=.01 a_delta =', round(o['adelta'], 5))
    print('\n== C4 (r =', R4, ')')
    for key in PB4:
        for al in (0.0, 0.5, 1.0, 2.0, 5.0):
            o = c4(key, al)
            print(f"{key:6s} a={al}: Q^={f(o['q'])} mu={f(o['mu'])} it={o['iters']} LSE={o['lse']:.4f} max={o['max']:.4f} "
                  f"Emu+H={o['EmuH']:.4f} Epb(Q^)={o['Epb_q']:.4f} Epb(r)={o['Epb_r']:.4f} gapHat={o['gap_hat']:.4f} gapTrue={o['gap_true']:.4f} "
                  f"diff={o['gap_hat']-o['gap_true']:.4f} a*chi={al*o['chi']:.4f}")
    print('\n== C4 tau mode (Eq. 30 saddle)')
    for key in ('rare', 'unif', 'common'):
        print(key, 'H(pb) =', round(c4tau(key, 99)[1], 4), 'gap(0) =', round(gap_reg(key, 0), 4),
              'gap(a) a=0.5,1,5,50,1e4:', [round(gap_reg(key, a), 4) for a in (0.5, 1, 5, 50, 1e4)])
    for tau in (2.0, 1.6, 1.5, 1.4, 1.3, 1.2, 1.17, 1.15):
        a, _ = c4tau('rare', tau); print(f'rare tau={tau}: alpha* =', 'inf' if a == math.inf else round(a, 4))
