"""C2: reward scale c and offset b; normalized vs fixed lambda (same algorithm as page)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy, math
A, R = toy.data(7); Qh, ab, va, c, sd = toy.fit(A, R)
def pol(alpha, scale, off, norm):
    f = lambda a: scale * (Qh(a) + off)   # Qhat of rescaled/shifted rewards (ridge fit is linear in targets; intercept unpenalized)
    QG = [f(a) for a in toy.GRID]
    m = max(sum(abs(f(a)) for a in A) / toy.N, 1e-12)
    lam = alpha / m if norm else alpha
    J = [lam * QG[j] - (toy.GRID[j] - ab) ** 2 for j in range(toy.G)]
    j = max(range(toy.G), key=lambda j: J[j]); p = toy.GRID[j]
    g = lam * scale * c[1] / sd   # d/da of lam*Qhat at a = abar
    return p, lam, m, g
print("abar=%.4f  q(abar)=%.4f" % (ab, toy.q(ab)))
for norm in (True, False):
    for s in [0.01, 0.1, 1, 10, 100]:
        p, lam, m, g = pol(2.5, s, 0, norm)
        print(f"norm={norm} c={s:<5} lam={lam:.5g} mean|Q|={m:.5g} pi={p:+.3f} q={toy.q(p):.4f} grad@abar={g:.4g}")
for off in [-4.3, -4, -2, 0, 4, 8]:
    p, lam, m, g = pol(2.5, 1, off, True)
    print(f"offset b={off:+}: mean|Q|={m:.4f} lam={lam:.4g} pi={p:+.3f} q={toy.q(p):.4f}")

if __name__ == '__main__':
    # continuity check of the C2 curves on the page grid (41 points of log10 c) and a fine grid
    for norm, off in [(False, 0), (True, 4), (True, -2)]:
        prev = None; jumps = []
        for i in range(401):
            lc = -2 + i * 0.01; p = pol(2.5, 10 ** lc, off, norm)[0]
            if prev is not None and abs(p - prev[1]) > 0.02: jumps.append((round(prev[0], 2), round(lc, 2), round(prev[1], 3), round(p, 3)))
            prev = (lc, p)
        print(f"norm={norm} b={off}: jumps(|dpi|>0.02 per 0.01 in log10 c) = {jumps}")
    # where fixed-lambda pi* reaches the bound
    for lc in [x / 20 for x in range(-40, 1)]:
        p = pol(2.5, 10 ** lc, 0, False)[0]
        if p >= 1: print("fixed lambda: pi*=1 first at log10 c =", lc); break
    # minimum of mean|Qhat+b| over the b slider (-8..8, step 0.1); c cancels in lambda*Q, so the warning uses m(b)=mean|Qhat+b|
    ms = [(round(-8 + 0.1 * k, 1), sum(abs(Qh(a) - 8 + 0.1 * k) for a in A) / toy.N) for k in range(161)]
    bmin, mmin = min(ms, key=lambda t: t[1])
    print(f"min over slider of mean|Qhat+b| = {mmin:.4f} at b = {bmin}; values b=-4.6..-4.0:", [(b, round(m, 4)) for b, m in ms if -4.65 < b < -3.95])
    print("b with mean|Qhat+b| < 0.5:", [b for b, m in ms if m < 0.5])
    print("pi* at b=-4.4 (norm, c=1):", pol(2.5, 1, -4.4, True)[:2])
