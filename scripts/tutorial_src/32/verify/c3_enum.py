"""C3: exact pseudo-label reliability by enumerating answer-count vectors.
Model (page plate line 1): m i.i.d. Solver answers; correct w.p. q; wrong mass (1-q)
split as s to distractor W1 and (1-s)/(K-1) to each of W2..WK (K=1 -> s=1).
Ties in the majority vote are broken uniformly at random (expected value).
Same bucketing algorithm as the page: compositions aggregated by (n0, n1)."""
import math
from itertools import product

def compositions(m, parts):
    if parts == 1: yield (m,); return
    for i in range(m + 1):
        for rest in compositions(m - i, parts - 1): yield (i,) + rest

def buckets(m, K, delta=0.25):
    lf = [math.lgamma(i + 1) for i in range(m + 1)]
    B = {}
    for n in compositions(m, K + 1):
        coef = math.exp(lf[m] - sum(lf[x] for x in n))
        c = max(n); T = [i for i, x in enumerate(n) if x == c]
        corr = (1.0 / len(T)) if 0 in T else 0.0
        ok = 1.0 if abs(c / m - 0.5) <= delta + 1e-12 else 0.0
        r = 1 - 2 * abs(c / m - 0.5)
        key = (n[0], n[1] if K >= 1 else 0)
        b = B.setdefault(key, {"corr": 0.0, "pass": 0.0, "cp": 0.0, "r": 0.0,
                               "hc": [0.0] * (m + 1), "hw": [0.0] * (m + 1)})
        b["corr"] += coef * corr; b["pass"] += coef * ok; b["cp"] += coef * corr * ok
        b["r"] += coef * r; b["hc"][c] += coef * corr; b["hw"][c] += coef * (1 - corr)
    return B

def evaluate(B, m, K, q, s):
    if K == 1: s = 1.0
    p1 = (1 - q) * s; pr = (1 - q) * (1 - s) / (K - 1) if K > 1 else 0.0
    out = {"corr": 0.0, "pass": 0.0, "cp": 0.0, "r": 0.0, "hc": [0.0] * (m + 1), "hw": [0.0] * (m + 1)}
    for (n0, n1), b in B.items():
        nr = m - n0 - n1
        w = q ** n0 * p1 ** n1 * pr ** nr     # python 0**0 == 1, same as JS Math.pow
        for k in ("corr", "pass", "cp", "r"): out[k] += w * b[k]
        for c in range(m + 1): out["hc"][c] += w * b["hc"][c]; out["hw"][c] += w * b["hw"][c]
    out["c_given_pass"] = out["cp"] / out["pass"] if out["pass"] > 1e-12 else float("nan")
    return out

if __name__ == "__main__":
    cache = {}
    def run(m, K, q, s):
        if (m, K) not in cache: cache[(m, K)] = buckets(m, K)
        return evaluate(cache[(m, K)], m, K, q, s)
    # sanity: total probability = 1
    for m, K in [(10, 1), (10, 3), (20, 5), (5, 2)]:
        o = run(m, K, 0.37, 0.6); tot = sum(o["hc"]) + sum(o["hw"]); assert abs(tot - 1) < 1e-9, tot
    print("m  K  q    s    | P(label ok)  P(pass)  P(ok|pass)  E[r_unc]")
    cases = [(10,3,0.5,0.5),(10,3,0.7,0.5),(10,3,0.3,0.5),(10,1,0.3,1),(10,1,0.6,1),(10,1,0.5,1),
             (10,5,0.3,0.2),(10,5,0.3,0.6),(10,3,0.1,0.5),(10,3,0.9,0.5),(5,3,0.5,0.5),(20,3,0.5,0.5),(10,3,0.0,0.5),(10,3,1.0,0.5)]
    for m, K, q, s in cases:
        o = run(m, K, q, s)
        print(f"{m:2d} {K} {q:.2f} {s:.2f} | {o['corr']:.4f}      {o['pass']:.4f}   {o['c_given_pass']:.4f}      {o['r']:.4f}")
    o = run(10, 3, 0.5, 0.5)
    print("hist default (c: P&ok, P&wrong):", [(c, round(o['hc'][c], 4), round(o['hw'][c], 4)) for c in range(1, 11)])
    # sweep extremes for the curves panel
    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        o = run(10, 3, q, 0.5); print("sweep q", q, round(o["corr"], 4), round(o["pass"], 4), round(o["r"], 4))
    # q that maximizes E[r_unc] on 0.01 grid, default K,s
    best = max((run(10, 3, i / 100, 0.5)["r"], i / 100) for i in range(101)); print("argmax E[r] q =", best[1], round(best[0], 4))
    # Monte-Carlo cross-check of one case (independent of enumeration)
    import random; random.seed(1); N = 200000; ok = ps = 0
    m, K, q, s = 10, 3, 0.3, 0.5; probs = [q, (1-q)*s] + [(1-q)*(1-s)/(K-1)]*(K-1)
    for _ in range(N):
        cnt = [0]*(K+1)
        for a in random.choices(range(K+1), probs, k=m): cnt[a] += 1
        c = max(cnt); T = [i for i, x in enumerate(cnt) if x == c]; lab = random.choice(T)
        ok += lab == 0; ps += 3 <= c <= 7
    o = run(m, K, q, s); print("MC", ok/N, ps/N, "vs exact", round(o["corr"], 4), round(o["pass"], 4))
    assert abs(ok/N - o["corr"]) < 0.005 and abs(ps/N - o["pass"]) < 0.005
    print("compositions m=20,K=5:", sum(1 for _ in compositions(20, 6)), "buckets:", len(cache[(20, 5)]) if (20,5) in cache else None)
    # P(label ok | c) for a spread-out case (supports Sec. 2.3 'low consistency -> unreliable')
    o = run(10, 5, 0.3, 0.2)
    print("P(ok|c) K=5 s=0.2 q=0.3:", [(c, round(o['hc'][c] / (o['hc'][c] + o['hw'][c]), 3) if o['hc'][c] + o['hw'][c] > 1e-12 else None) for c in range(1, 11)])
    print("P(c<=2) K=5 s=0.2 q=0.3:", round(sum(o['hc'][c] + o['hw'][c] for c in (1, 2)), 4))
    # round 1: scenario buttons (q slider + wrong-answer distribution) and P(ok|c) direction
    SCEN = {"spread": (5, 0.20), "medium": (3, 0.50), "concentrated": (1, 1.0)}
    for name, (K, s) in SCEN.items():
        for q in (0.3, 0.5, 0.7):
            o = run(10, K, q, s)
            pc = [round(o['hc'][c] / (o['hc'][c] + o['hw'][c]), 3) if o['hc'][c] + o['hw'][c] > 1e-12 else None for c in range(1, 11)]
            top = "yes" if q > (1 - q) * s + 1e-12 else ("tie" if abs(q - (1 - q) * s) < 1e-12 else "no")
            print(f"scen {name} q={q} plurality={top} corr={o['corr']:.4f} pass={o['pass']:.4f} cp={o['c_given_pass']:.4f} P(ok|c)={pc}")
    print("s slider min per K:", [math.ceil(100 / K) / 100 for K in range(1, 6)])
