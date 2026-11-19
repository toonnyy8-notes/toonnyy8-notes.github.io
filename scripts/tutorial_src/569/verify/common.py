"""Shared math for the IQL tutorial (mirrors the page's JS exactly)."""
import math

def expectile(xs, ps, tau):
    """Exact tau-expectile of a discrete distribution (values xs, weights ps>=0).
    Solves tau*sum_{x>=m} p(x-m) = (1-tau)*sum_{x<m} p(m-x) piecewise in closed form.
    Entries with p<=0 are ignored (not in support)."""
    pts = sorted((x, p) for x, p in zip(xs, ps) if p > 0)
    if not pts:
        return float('nan')
    if tau >= 1: return pts[-1][0]
    if tau <= 0: return pts[0][0]
    n = len(pts)
    # prefix sums of p and p*x
    P = [0.0]; S = [0.0]
    for x, p in pts:
        P.append(P[-1] + p); S.append(S[-1] + p * x)
    # k = 0..n-1: lo = pts[:k] (below m, weight 1-tau), hi = pts[k:] (>= m, weight tau).
    # S_lo = sum of first k entries, S_hi = sum of the rest. hi is never empty (m <= max).
    for k in range(n):
        lp, ls = P[k], S[k]; hp, hs = P[n] - P[k], S[n] - S[k]
        m = (tau * hs + (1 - tau) * ls) / (tau * hp + (1 - tau) * lp)
        lo_ok = k == 0 or pts[k - 1][0] <= m + 1e-12
        hi_ok = pts[k][0] >= m - 1e-12
        if lo_ok and hi_ok:
            return m
    # fallback (should not trigger): bisection on the decreasing first-order residual g(m)
    lo, hi = pts[0][0], pts[-1][0]
    for _ in range(200):
        m = (lo + hi) / 2
        g = sum(p * (tau if x >= m else 1 - tau) * (x - m) for x, p in pts)
        lo, hi = (m, hi) if g > 0 else (lo, m)
    return (lo + hi) / 2

def L2tau(u, tau):
    return abs(tau - (1.0 if u < 0 else 0.0)) * u * u

def mulberry32(seed):
    st = [seed & 0xffffffff]
    def imul(a, b): return (a * b) & 0xffffffff
    def r():
        st[0] = (st[0] + 0x6D2B79F5) & 0xffffffff
        t = st[0]
        t = imul(t ^ (t >> 15), 1 | t)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & 0xffffffff) ^ t
        return ((t ^ (t >> 14)) & 0xffffffff) / 4294967296
    return r

if __name__ == '__main__':
    # self-checks
    assert abs(expectile([1, 2, 3], [1, 1, 1], 0.5) - 2) < 1e-12
    assert expectile([1, 2, 3], [1, 1, 0], 0.999999) < 2 + 1e-4
    # brute-force check of expectile vs minimizing the loss on a grid
    import random
    for _ in range(200):
        xs = [random.uniform(-5, 5) for _ in range(5)]; ps = [random.random() for _ in range(5)]
        tau = random.uniform(0.01, 0.99)
        m = expectile(xs, ps, tau)
        g = sum(p * abs(tau - (x < m)) * (x - m) for x, p in zip(xs, ps))
        assert abs(g) < 1e-9, g
    # duplicates and extreme tau
    assert abs(expectile([2, 2, 2], [1, 1, 1], 0.999) - 2) < 1e-12
    assert abs(expectile([1, 3, 3, 8], [.25, .25, .25, .25], 0.999) - 7.9830) < 1e-4, expectile([1, 3, 3, 8], [.25] * 4, 0.999)
    assert abs(expectile([1, 2, 3, 4, 8], [.3, .3, .2, .15, .05], 0.999) - 7.8919) < 1e-4
    print('common ok')
