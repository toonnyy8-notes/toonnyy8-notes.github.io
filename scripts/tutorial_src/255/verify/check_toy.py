"""C3 toy: exact output distribution of Algorithm 1 (pi_{k+1} in line 7) on a small random LM.
Same algorithm the page worker uses. Run: python3 check_toy.py"""
import math, itertools

def mulberry32(a):
    a &= 0xFFFFFFFF
    def r():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return r

BETA = 3.0  # logit scale (demo value)
def prefix_code(pre, V):
    return sum(V**i for i in range(len(pre))) + sum(v * V**(len(pre)-1-i) for i, v in enumerate(pre))

def make_lm(V, T, seed):
    """logp[pre] = list of log p(v | pre), pre = tuple of generated tokens (x_1..x_{t-1})."""
    logp = {}
    for L in range(T):
        for pre in itertools.product(range(V), repeat=L):
            r = mulberry32(seed * 100003 + prefix_code(pre, V))
            lg = [BETA * r() for _ in range(V)]
            m = max(lg); z = m + math.log(sum(math.exp(x - m) for x in lg))
            logp[pre] = [x - z for x in lg]
    return logp

def lse(xs):
    m = max(xs); return m + math.log(sum(math.exp(x - m) for x in xs))

def build(V, T, seed, alpha, tau):
    logp = make_lm(V, T, seed)
    lprop = {pre: [x/tau - lse([y/tau for y in lp]) for x in lp] for pre, lp in logp.items()}
    logZ = {pre: lse([alpha*x for x in lp]) for pre, lp in logp.items()}  # log Z_t = log sum_v p^alpha
    return logp, lprop, logZ

def seq_logp(logp, x):
    return sum(logp[x[:t]][x[t]] for t in range(len(x)))

def dists(V, T, seed, alpha, tau):
    logp, lprop, logZ = build(V, T, seed, alpha, tau)
    seqs = list(itertools.product(range(V), repeat=T))
    lb = [seq_logp(logp, s) for s in seqs]
    base = [math.exp(x) for x in lb]
    la = [alpha*x for x in lb]; z = lse(la); powd = [math.exp(x - z) for x in la]
    ltemp = [sum(alpha*logp[s[:t]][s[t]] - logZ[s[:t]] for t in range(T)) for s in seqs]
    temp = [math.exp(x) for x in ltemp]
    return seqs, base, temp, powd, (logp, lprop, logZ)

def stage_kernels(V, T, B, alpha, model):
    """Sparse MH kernel per stage over all V^L states (independent of N)."""
    logp, lprop, logZ = model
    lw = {pre: [alpha*a - b for a, b in zip(logp[pre], lprop[pre])] for pre in logp}
    ks = []
    for k in range(math.ceil(T / B)):
        L = min((k+1)*B, T)
        states = list(itertools.product(range(V), repeat=L))
        # suffix sums: Sw[y][j] = sum_{t>=j} lw, Sq[y][j] = sum_{t>=j} lprop  (0-based t)
        Sw, Sq = {}, {}
        for y in states:
            aw = [0.0]*(L+1); aq = [0.0]*(L+1)
            for t in range(L-1, -1, -1):
                aw[t] = aw[t+1] + lw[y[:t]][y[t]]; aq[t] = aq[t+1] + lprop[y[:t]][y[t]]
            Sw[y], Sq[y] = aw, aq
        rows = {}
        for x in states:
            row = {}
            for m in range(1, L+1):
                pre = x[:m-1]; old = Sw[x][m-1]
                for suf in itertools.product(range(V), repeat=L-m+1):
                    y = pre + suf
                    new = Sw[y][m-1]
                    A = 1.0 if new >= old else math.exp(new - old)
                    pr = math.exp(Sq[y][m-1]) / L
                    row[y] = row.get(y, 0) + pr*A
                    row[x] = row.get(x, 0) + pr*(1-A)
            rows[x] = row
        ks.append((L, rows))
    return ks

def alg1_dist(V, T, B, alpha, model, N, ks=None):
    logp, lprop, logZ = model
    ks = ks or stage_kernels(V, T, B, alpha, model)
    mu = {(): 1.0}
    for L, rows in ks:
        while len(next(iter(mu))) < L:
            nmu = {}
            for x, w in mu.items():
                for v in range(V):
                    nmu[x+(v,)] = nmu.get(x+(v,), 0) + w*math.exp(lprop[x][v])
            mu = nmu
        for _ in range(N):
            nmu = {}
            for x, w in mu.items():
                for y, pr in rows[x].items():
                    nmu[y] = nmu.get(y, 0) + w*pr
            mu = nmu
    return mu

def tv(p, q): return 0.5*sum(abs(a-b) for a, b in zip(p, q))

def zt_identity_check(V, T, seed, alpha):
    """With tau=1/alpha the MH ratio equals prod_t Z_t(x'_<t)/Z_t(x_<t) over t>=m."""
    logp, lprop, logZ = build(V, T, seed, alpha, 1/alpha)
    worst = 0
    for x in itertools.product(range(V), repeat=T):
        for y in itertools.product(range(V), repeat=T):
            m = next((i for i in range(T) if x[i] != y[i]), T)
            def lw(s): return sum(alpha*logp[s[:t]][s[t]] - lprop[s[:t]][s[t]] for t in range(m, T))
            ratio = lw(y) - lw(x)
            zr = sum(logZ[y[:t]] for t in range(m, T)) - sum(logZ[x[:t]] for t in range(m, T))
            worst = max(worst, abs(ratio - zr))
    return worst

def single_steps(V, T, B, seed, alpha, tau, chain_seed, nsteps):
    """PRNG-driven trace of Algorithm 1's first stage (k=0, L=min(B,T)) for the single-step panel."""
    logp, lprop, logZ = build(V, T, seed, alpha, tau)
    r = mulberry32(chain_seed)
    def draw(pre):
        u = r(); c = 0
        for v in range(V):
            c += math.exp(lprop[pre][v])
            if u < c: return v
        return V-1
    L = min(B, T)
    x = ()
    for _ in range(L): x = x + (draw(x),)
    out = [("init", x)]
    for n in range(nsteps):
        m = 1 + math.floor(r()*L)
        y = x[:m-1]
        for _ in range(m-1, L): y = y + (draw(y),)
        lw = lambda s: sum(alpha*logp[s[:t]][s[t]] - lprop[s[:t]][s[t]] for t in range(m-1, L))
        lr = lw(y) - lw(x); A = min(1.0, math.exp(lr)); u = r()
        acc = u <= A
        out.append((n+1, m, y, round(A, 4), round(u, 4), acc))
        if acc: x = y
    return out

if __name__ == "__main__":
    r = mulberry32(1); print("mulberry32(1) first 3:", [round(r(), 6) for _ in range(3)])
    lm = make_lm(3, 6, 1)
    print("p(.|()) seed1:", [round(math.exp(v), 4) for v in lm[()]], " p(.|(0,)):", [round(math.exp(v), 4) for v in lm[(0,)]])
    print("Z_t identity max |diff| (V=2,T=4,a=4):", zt_identity_check(2, 4, 1, 4.0))
    names = "abc"
    for (V, T, seed, alpha, tau) in [(3, 6, 1, 4.0, 0.25), (3, 6, 1, 4.0, 1.0), (3, 6, 1, 2.0, 0.5), (2, 6, 1, 4.0, 0.25), (3, 6, 2, 4.0, 0.25), (3, 6, 1, 1.0, 1.0)]:
        seqs, base, temp, powd, model = dists(V, T, seed, alpha, tau)
        print(f"\nV={V} T={T} seed={seed} alpha={alpha} tau={tau}: TV(base,pow)={tv(base,powd):.4f} TV(temp,pow)={tv(temp,powd):.4f}")
        top = sorted(range(len(seqs)), key=lambda i: -powd[i])[:5]
        print("  top5 by p^a:", [("".join(names[v] for v in seqs[i]), round(base[i],4), round(temp[i],4), round(powd[i],4)) for i in top])
        for B in ([2, 1, 3, 6] if (V, seed, alpha, tau) == (3, 1, 4.0, 0.25) else [2]):
            line = []; ks = stage_kernels(V, T, B, alpha, model)
            for N in [0, 1, 2, 3, 5, 10, 20, 50]:
                mu = alg1_dist(V, T, B, alpha, model, N, ks)
                line.append(round(tv([mu.get(s, 0) for s in seqs], powd), 4))
            print(f"  B={B} TV(Alg1,pow) for N=0,1,2,3,5,10,20,50:", line)
    print("\nsingle-step trace V=3 T=6 L=6 (B=6) seed=1 a=4 tau=.25 chain_seed=3 (C3a default):")
    for row in single_steps(3, 6, 6, 1, 4.0, 0.25, 3, 8):
        print("  ", row[0], *[("".join(names[v] for v in z) if isinstance(z, tuple) else z) for z in row[1:]])
