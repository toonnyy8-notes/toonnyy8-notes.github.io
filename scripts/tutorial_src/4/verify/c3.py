"""C3: decoding error rate P_e(t) (Eq. 24) and tau(t) (Eq. 25), same algorithm as page.
P_e(t) = 1 - int phi(z) Phi(z + t/(1-t))^(V-1) dz   [本頁推導]
Page uses NR erfcc (frac err < 1.2e-7); we compare with math.erfc."""
import math

def log_erfc_nr(z):           # z >= 0, Numerical Recipes erfcc in log form
    t = 1 / (1 + 0.5 * z)
    poly = (-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
            t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return math.log(t) + poly

def log_erfc_ref(z):
    return math.log(math.erfc(z)) if z < 26 else (-z*z - math.log(z*math.sqrt(math.pi)))

def logPhi(x, le):
    z = -x / math.sqrt(2)
    if z >= 0: return math.log(0.5) + le(z)
    return math.log1p(-0.5 * math.exp(le(-z)))

Z0, Z1, HZ = -8.0, 8.0, 0.01
def Pe(t, V, le=log_erfc_nr, hz=HZ):
    if t >= 1: return 0.0
    c = t / (1 - t)
    n = int(round((Z1 - Z0) / hz)); acc = 0.0
    for i in range(n + 1):
        z = Z0 + i * hz
        f = math.exp(-z * z / 2) / math.sqrt(2 * math.pi) * math.exp((V - 1) * logPhi(z + c, le))
        acc += f * (0.5 if i in (0, n) else 1)
    return 1 - acc * hz

def lut(V, M=1000, le=log_erfc_nr):
    ts = [i / M for i in range(M + 1)]
    pe = [Pe(t, V, le) for t in ts]
    p0 = pe[0]
    tau = [min(1, max(0, (p0 - p) / p0)) for p in pe]
    return ts, pe, tau

def t_of_tau(ts, tau, y):
    if y <= 0: return 0.0
    if y >= 1: return 1.0
    for i in range(1, len(ts)):
        if tau[i] >= y:
            a = tau[i - 1]; b = tau[i]
            return ts[i - 1] + (ts[i] - ts[i - 1]) * ((y - a) / (b - a) if b > a else 0)
    return 1.0

if __name__ == '__main__':
    print('V      Pe(0)      1-1/V     Pe(.5)  Pe(.8)  t@tau=.05 .25 .5 .75 .95   uniform N=16 in window  ref-diff')
    for V in [2, 10, 100, 1000, 30522, 50257]:
        ts, pe, tau = lut(V)
        tq = [t_of_tau(ts, tau, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
        lo, hi = tq[0], tq[-1]
        N = 16
        inwin = sum(1 for n in range(N + 1) if lo <= n / N <= hi)
        d = max(abs(Pe(t, V) - Pe(t, V, log_erfc_ref, 0.002)) for t in (0, 0.3, 0.5, 0.7, 0.8, 0.9))
        print(f'{V:6d} {pe[0]:.6f} {1-1/V:.6f}  {Pe(0.5,V):.4f} {Pe(0.8,V):.4f}  ' + ' '.join(f'{v:.3f}' for v in tq) +
              f'   {inwin}/{N+1}   {d:.1e}')
    # monotone check
    ts, pe, tau = lut(30522)
    print('tau monotone (30522):', all(tau[i] <= tau[i+1] + 1e-12 for i in range(len(tau)-1)))
    for N in (4, 8, 16, 32, 64):
        print('V=30522 tau-grid N=%d:' % N, ' '.join(f'{t_of_tau(ts,tau,n/N):.3f}' for n in range(N+1)) if N <= 8 else '')
