"""C3: invert Eq. 10. Given Delta N(t)-1 = F(t) on t in [5, T'] (F(T')=0), lambda_t - 1 = F(t-1) - 2F(t) + F(t+1), F(T'+1):=0.
Also the forward direction: F(t) = sum_{s=t+1}^{T'} (lambda_s - 1)(s - t). Same algorithm as the page."""
def F_shape(Tp, h, w):
    return {t: h * ((1 - w) * ((Tp - t) / (Tp - 5)) ** 2 + w * ((Tp - t) / (Tp - 5)) ** 3) for t in range(5, Tp + 1)}
def invert(Tp, h, w):
    F = F_shape(Tp, h, w); F[Tp + 1] = 0.0
    return {t: 1 + F[t - 1] - 2 * F[t] + F[t + 1] for t in range(6, Tp + 1)}
def forward(Tp, lam):  # lam: dict t->lambda for t in 6..Tp
    return {t: sum((lam[s] - 1) * (s - t) for s in range(max(t + 1, 6), Tp + 1)) for t in range(5, Tp + 1)}
if __name__ == '__main__':
    for Tp, h, w in [(20, 0.02, 0), (20, 0.02, 1), (20, 0.02, 0.5), (20, 0.1, 0)]:
        lam = invert(Tp, h, w); Fb = forward(Tp, lam); F = F_shape(Tp, h, w)
        err = max(abs(Fb[t] - F[t]) for t in range(5, Tp + 1))
        m = (6 + Tp) // 2
        print(f"T'={Tp} h={h} w={w}: lam6={lam[6]:.6f} lam{m}={lam[m]:.6f} lam{Tp-1}={lam[Tp-1]:.6f} lam{Tp}={lam[Tp]:.6f} 2h/(T'-5)^2={2*h/(Tp-5)**2:.6f} roundtrip_err={err:.1e}")
