"""C3: argmax over N behavior samples (Eq. 13) when the behavior model emits outliers (toy, 推定).
Behavior model = (1-eps) * mu + eps * Uniform[U_LO, U_HI]; critic Qhat(a) = a everywhere (extrapolates);
true reward r(a) = a for in-support samples, R_OUT for outlier samples.
E[r] = int N G^{N-1} [(1-eps) f(x) x + eps u(x) R_OUT] dx,  G = (1-eps) F + eps U."""
import sys, os, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

U_LO, U_HI, R_OUT = 3.5, 5.0, 0.0
pi = PI_DEFAULT
X0, X1, NG = LO, U_HI, 6001
hh = (X1 - X0) / (NG - 1)
XS = [X0 + i * hh for i in range(NG)]
F = [mix_cdf(x, pi) for x in XS]; f = [mix_pdf(x, pi) for x in XS]
U = [min(1.0, max(0.0, (x - U_LO) / (U_HI - U_LO))) for x in XS]
u = [1.0 / (U_HI - U_LO) if U_LO <= x <= U_HI else 0.0 for x in XS]

def er(N, eps):
    tot_r = tot_out = 0.0
    for i, x in enumerate(XS):
        G = (1 - eps) * F[i] + eps * U[i]
        gN = N * G ** (N - 1) if N > 1 else 1.0
        wt = hh if 0 < i < NG - 1 else hh / 2
        tot_r += wt * gN * ((1 - eps) * f[i] * x + eps * u[i] * R_OUT)
        tot_out += wt * gN * eps * u[i]
    return tot_r, tot_out

for eps in (0.0, 0.001, 0.01, 0.05):
    row = []
    for N in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        r, p = er(N, eps)
        row.append(f'N={N}: E[r]={r:.4f} P_out={p:.4f} (1-(1-eps)^N={1-(1-eps)**N:.4f})')
    print(f'## eps={eps}'); print('\n'.join(row))
