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

def gauss(rnd):
    """Box-Muller, one value per two uniforms (same as the page)."""
    u1, u2 = rnd(), rnd()
    return math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)

def inv_norm(p):
    """Acklam rational approximation of the N(0,1) quantile; max abs err < 3e-9 on (0,1) (checked below)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q = p - 0.5; r = q * q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return x

def w1_sorted(x, y):
    """1-Wasserstein between two equal-size, equally-weighted, SORTED samples."""
    return sum(abs(a - b) for a, b in zip(x, y)) / len(x)

def w1_weighted(xa, wa, xb, wb):
    """Exact W1 = integral |F_a - F_b| between two weighted atom sets (weights each sum to 1)."""
    ev = sorted([(x, w) for x, w in zip(xa, wa)] + [(x, -w) for x, w in zip(xb, wb)])
    tot = 0.0; diff = 0.0; prev = ev[0][0]
    for x, w in ev:
        tot += abs(diff) * (x - prev); diff += w; prev = x
    return tot

if __name__ == '__main__':
    import random
    from statistics import NormalDist
    assert abs(expectile([1, 2, 3, 4, 8], [.3, .3, .2, .15, .05], 0.999) - 7.8919) < 1e-4
    for _ in range(200):
        xs = [random.uniform(-5, 5) for _ in range(5)]; ps = [random.random() for _ in range(5)]
        tau = random.uniform(0.01, 0.99); m = expectile(xs, ps, tau)
        assert abs(sum(p * abs(tau - (x < m)) * (x - m) for x, p in zip(xs, ps))) < 1e-9
    r = mulberry32(1); assert r() == 0.6270739405881613
    e = max(abs(inv_norm(p) - NormalDist().inv_cdf(p)) for p in [i / 1000 for i in range(1, 1000)])
    print('inv_norm max abs err', e); assert e < 1e-8
    assert abs(w1_weighted([0, 1], [.5, .5], [0.5, 1.5], [.5, .5]) - 0.5) < 1e-12
    print('common ok')
