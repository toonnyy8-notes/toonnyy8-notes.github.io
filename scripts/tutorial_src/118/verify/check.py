# Verification for tutorial 118 (GenRM). Same algorithms the page will use (pure Python, no numpy).
# Run: python3 check.py   -> prints acceptance tables + asserts + Monte Carlo cross-checks.
import math, random

# ---------- shared ----------
def erfc(x):  # Numerical Recipes erfcc, fractional error < 1.2e-7 (page uses identical code)
    z = abs(x); t = 1 / (1 + 0.5 * z)
    a = t * math.exp(-z*z - 1.26551223 + t*(1.00002368 + t*(0.37409196 + t*(0.09678418 + t*(-0.18628806 +
        t*(0.27886807 + t*(-1.13520398 + t*(1.48851587 + t*(-0.82215223 + t*0.17087277)))))))))
    return a if x >= 0 else 2 - a
def Phi(s): return 0.5 * erfc(-s / math.sqrt(2))
def phi(s): return math.exp(-0.5*s*s) / math.sqrt(2*math.pi)
def binom_pmf(N, p):
    assert 0 < p < 1  # sliders give p in [0.05,0.95]; no clamp (matches page)
    out = [(1-p)**N]
    for c in range(1, N+1): out.append(out[-1] * (N-c+1) / c * p / (1-p))
    return out

# ---------- C1: Best-of-N with Gaussian verifier scores ----------
H = 0.01
def win(c, m, mu):
    """P(max of c correct scores ~N(mu,1) > max of m incorrect ~N(0,1))."""
    if c == 0: return 0.0
    if m == 0: return 1.0
    lo, hi = min(0, mu) - 8, max(0, mu) + 8
    n = int(round((hi - lo) / H)); acc = 0.0
    for i in range(n + 1):
        s = lo + i * H; w = 0.5 if i in (0, n) else 1.0
        lc = math.log(max(Phi(s - mu), 1e-300)); li = math.log(max(Phi(s), 1e-300))
        acc += w * c * phi(s - mu) * math.exp((c-1)*lc + m*li)
    return acc * H
def best_of_n(p, N, mu):
    pmf = binom_pmf(N, p)
    return sum(pmf[c] * win(c, N-c, mu) for c in range(N+1))
def oracle(p, N): return 1 - (1-p)**N
def rm_acc(mu): return Phi(mu/2)

# ---------- C2: Yes-token score ----------
def c2(zy, zn, zo, g):
    a = [zy/g, zn/g, zo/g]; m = max(a); e = [math.exp(v-m) for v in a]; S = sum(e)
    pY, pN, pO = (v/S for v in e)
    two = 1/(1+math.exp(-(zy-zn)/g))
    loss = -(a[0] - m - math.log(S))          # -log pY, stable
    grad = [(pY-1)/g, pN/g, pO/g]              # d(-log pY)/d z_j
    return pY, pN, pO, two, loss, grad

# ---------- C3: majority voting (MajV@K) + Best-of-N with discrete scores ----------
def votes_pmf(K, qe, qh, h):
    """pmf over #correct-direction verdicts Y in 0..K, mixture of Binomials."""
    def bp(q):
        q = min(max(q, 0.0), 1.0); return [math.comb(K, y) * q**y * (1-q)**(K-y) for y in range(K+1)]
    e, d = bp(qe), bp(qh)
    return [(1-h)*e[y] + h*d[y] for y in range(K+1)]
def c3(K, qe, qh, h, p, N):
    v = votes_pmf(K, qe, qh, h)
    a = v[:]                       # correct solution: score level = Y (score Y/K)
    b = v[::-1]                    # incorrect solution: #Yes = K - Y
    # RM accuracy at threshold 0.5 (ties count half); symmetric classes
    acc = sum(a[l] for l in range(K+1) if 2*l > K) + 0.5*sum(a[l] for l in range(K+1) if 2*l == K)
    A = [0.0]; B = [0.0]           # A[l] = P(level < l)
    for l in range(K+1): A.append(A[-1]+a[l]); B.append(B[-1]+b[l])
    pmf = binom_pmf(N, p); bon = 0.0
    for C in range(1, N+1):
        M = N - C; s = 0.0
        for l in range(K+1):
            if a[l] == 0: continue
            for i in range(1, C+1):
                ti = math.comb(C, i) * a[l]**i * A[l]**(C-i)
                if ti == 0: continue
                for j in range(0, M+1):
                    tj = math.comb(M, j) * b[l]**j * B[l]**(M-j)
                    s += ti * tj * i / (i+j)
        bon += pmf[C] * s
    mean_c = sum(l/K*a[l] for l in range(K+1)); mean_i = sum(l/K*b[l] for l in range(K+1))
    return acc, bon, mean_c, mean_i

# ---------- C4: weighted self-consistency ----------
PRESETS = {
  'P1': {'truth': 'B', 'c': [('A',.40),('A',.35),('A',.30),('A',.45),('B',.90),('B',.85),('C',.20),('D',.95)]},
  'P2': {'truth': 'A', 'c': [('A',.90),('A',.92),('A',.88),('A',.95),('A',.90),('B',.30),('C',.20),('A',.85)]},
  'P3': {'truth': 'B', 'c': [('A',.30),('A',.30),('A',.30),('A',.30),('A',.30),('A',.30),('B',.85),('B',.80)]},
}
def c4(cands, K):
    letters = sorted({a for a, _ in cands})
    votes = {L: sum(1 for a, _ in cands if a == L) for L in letters}
    sc = max(letters, key=lambda L: (votes[L], -ord(L)))
    bon = cands[max(range(len(cands)), key=lambda i: (cands[i][1], -i))][0]
    wsum = {L: sum(s for a, s in cands if a == L) for L in letters}
    wsc = max(letters, key=lambda L: (round(wsum[L], 9), -ord(L)))
    topk = {L: sum(sorted([s for a, s in cands if a == L], reverse=True)[:K]) for L in letters}
    wtk = max(letters, key=lambda L: (round(topk[L], 9), -ord(L)))
    return sc, bon, wsc, wtk, wsum, topk

if __name__ == '__main__':
    print('== C1 Best-of-N (p, mu) rows; N = 1 2 4 8 16 32 64 ==')
    Ns = [1, 2, 4, 8, 16, 32, 64]
    for p, mu in [(0.3, 0.0), (0.3, 1.0), (0.3, 1.5), (0.3, 3.0), (0.1, 1.5), (0.6, 1.5)]:
        print(f'p={p} mu={mu}: ' + ' '.join(f'{best_of_n(p,N,mu):.4f}' for N in Ns) +
              f' | oracle@16={oracle(p,16):.4f} rmacc={rm_acc(mu):.4f}')
    for N in Ns: assert abs(best_of_n(0.3, N, 0.0) - 0.3) < 1e-4  # random pick = pass@1
    assert abs(best_of_n(0.3, 1, 1.5) - 0.3) < 1e-4 and abs(best_of_n(0.3, 16, 1.5) - 0.8470) < 5e-4
    # C3: default (h hidden -> 0) and non-monotone RM acc with h=0.2 (approaches 1-h from above)
    assert abs(c3(32, .8, .3, 0, .3, 16)[0] - 0.9999) < 5e-4
    assert c3(16, .8, .3, .2, .3, 16)[0] > c3(32, .8, .3, .2, .3, 16)[0] > 0.8
    # Pass@N is a per-problem formula: half p=0, half p=1 -> dataset Pass@N = 0.5, not 1-(0.5)^N
    assert 0.5*oracle(1e-12, 16) + 0.5*oracle(1-1e-12, 16) < 0.51 < oracle(0.5, 16)
    assert c4(PRESETS['P1']['c'],1)[:4]==('A','D','B','D') and c4(PRESETS['P3']['c'],4)[:4]==('A','B','A','B')
    # Monte Carlo cross-check
    random.seed(1); T = 200000; p, N, mu = 0.3, 16, 1.5; hit = 0
    for _ in range(T):
        best, ok = -1e9, False
        for _ in range(N):
            c = random.random() < p; s = random.gauss(mu if c else 0, 1)
            if s > best: best, ok = s, c
        hit += ok
    print('MC p=.3 N=16 mu=1.5:', hit/T, 'vs', round(best_of_n(p, N, mu), 4))

    print('== C2 ==')
    for args in [(2,0,-4,1), (2,0,0,1), (2,0,2,1), (2,0,-4,0.5), (0,0,-30,1)]:
        pY,pN,pO,two,loss,g = c2(*args)
        print(args, f'pY={pY:.4f} pN={pN:.4f} pO={pO:.4f} sig2way={two:.4f} loss={loss:.4f} grad=' + ','.join(f'{x:.4f}' for x in g))

    print('== C3 (qe,qh,h,p=.3,N=16) K = 1 2 4 8 16 32 -> rmacc / bon ==')
    Ks = [1, 2, 4, 8, 16, 32]
    for qe, qh, h in [(0.8, 0.3, 0.0), (0.8, 0.3, 0.2), (0.7, 0.3, 0.2), (0.6, 0.3, 0.0), (0.8, 0.3, 0.5)]:
        r = [c3(K, qe, qh, h, 0.3, 16) for K in Ks]
        print(f'qe={qe} qh={qh} h={h}: acc ' + ' '.join(f'{x[0]:.4f}' for x in r) + ' | bon ' + ' '.join(f'{x[1]:.4f}' for x in r))
    print('oracle@16 p=.3 =', round(oracle(0.3, 16), 4))
    # MC cross-check of discrete BoN with random tie-break
    random.seed(2); T = 100000; K, qe, qh, h, p, N = 4, 0.8, 0.3, 0.2, 0.3, 16; hit = 0
    for _ in range(T):
        sc = []
        for _ in range(N):
            c = random.random() < p; q = qh if random.random() < h else qe
            y = sum(random.random() < q for _ in range(K)); lvl = y if c else K - y
            sc.append((lvl, c))
        m = max(l for l, _ in sc); top = [c for l, c in sc if l == m]
        hit += random.choice(top)
    print('MC K=4 bon:', hit/T, 'vs', round(c3(K, qe, qh, h, p, N)[1], 4))

    print('== C4 ==')
    for name, P in PRESETS.items():
        for K in [1, 2, 4, 8]:
            sc, bon, wsc, wtk, ws, tk = c4(P['c'], K)
            print(name, 'truth', P['truth'], 'K', K, '| SC', sc, 'BoN', bon, 'WSC', wsc, 'WSC-top', wtk,
                  '| sums', {k: round(v, 2) for k, v in ws.items()}, 'topk', {k: round(v, 2) for k, v in tk.items()})
    print('paper check: MATH 44.6-28.0 =', round(44.6-28.0, 1), 'pp; GSM 93.4-73.0 =', round(93.4-73.0, 1),
          'pp; algo 45.3-5.0 =', round(45.3-5.0, 1), 'pp; MMLU AA 53.5-37.9 =', round(53.5-37.9, 1), 'pp')
    print('ALL ASSERTS PASSED')
