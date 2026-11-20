"""W4: top-k order statistics (Algo 1/3) + normaliser (Eq. 41-45, case-corrected). Same algorithm as page."""
import math, sys
from core import Mulberry32, gauss, norm_ppf, log_phi_cdf

def topk_zero_mean(r, N, k, sig):
    """Algo 1, log space. Draw order: l = N, N-1, ..., N-k+1. Returns descending list."""
    P = 0.0; out = []
    for l in range(N, N - k, -1):
        u = r()
        u = u if u > 0 else 1e-300            # guard log(0)
        P += math.log(u) / l
        one_minus_v = -math.expm1(P)          # 1-V without cancellation
        out.append(-sig * norm_ppf(one_minus_v) if one_minus_v < 0.5 else sig * norm_ppf(math.exp(P)))
    return out

def log_mu(c, sig, tau):
    """log E[exp(X/tau) | X < c], X~N(0,sig^2)  (Eq. 43/44, read as log of Eq. 40)."""
    return sig * sig / (2 * tau * tau) - log_phi_cdf(c / sig) + log_phi_cdf((c - sig * sig / tau) / sig)

def lse(xs):
    m = max(xs); return m + math.log(sum(math.exp(x - m) for x in xs))

def efficient(r, K, k, at, tau):
    sig = math.sqrt(1 - at * at)
    Kv = topk_zero_mean(r, K - 1, k, sig)
    wt = at + sig * gauss(r)
    if wt > Kv[-1]:                                   # case 2: clean token in top-k
        thr = Kv[-2] if k >= 2 else Kv[-1]            # Algo 3 uses K_{k-1} (pre-insertion index); k=1 -> K_k
        rnk = sum(1 for v in Kv if v > wt)
        Kv = Kv[:rnk] + [wt] + Kv[rnk:k - 1]
        logs = [v / tau for v in Kv] + [math.log(K - k) + log_mu(thr, sig, tau)]
        case = 2; pos = rnk
    else:                                             # case 1
        logs = [v / tau for v in Kv] + [wt / tau, math.log(K - k - 1) + log_mu(Kv[-1], sig, tau)]
        case = 1; pos = -1
    logZ = lse(logs)
    lam = [math.exp(v / tau - logZ) for v in Kv]
    return dict(vals=Kv, wt=wt, case=case, pos=pos, lam=lam, logZ=logZ)

def naive(r, K, k, at, tau, o=0):
    sig = math.sqrt(1 - at * at)
    w = [sig * gauss(r) for _ in range(K)]; w[o] += at
    logZ = lse([v / tau for v in w])
    idx = sorted(range(K), key=lambda i: -w[i])[:k]
    return dict(w=w, idx=idx, vals=[w[i] for i in idx], lam=[math.exp(w[i] / tau - logZ) for i in idx], logZ=logZ, clean_in=(o in idx))

def ztilde_from(w, k, at, tau, o=0):
    """Deterministic: apply Eq. 41/42 (case-corrected) to the SAME realised w. Returns log Z~, threshold choice as Algo 3."""
    sig = math.sqrt(1 - at * at); K = len(w)
    zm = sorted((w[i] for i in range(K) if i != o), reverse=True)
    if w[o] > zm[k - 1]:
        thr = zm[k - 2] if k >= 2 else zm[k - 1]
        top = sorted(zm[:k - 1] + [w[o]], reverse=True)
        logs = [v / tau for v in top] + [math.log(K - k) + log_mu(thr, sig, tau)]
    else:
        top = zm[:k]
        logs = [v / tau for v in top] + [w[o] / tau, math.log(K - k - 1) + log_mu(zm[k - 1], sig, tau)]
    return lse(logs), lse([v / tau for v in top])

if __name__ == '__main__':
    # (43) numeric check vs direct integration (advisor value 1.21165 at sig=.5,tau=.2,c=.7)
    sig, tau, c = 0.5, 0.2, 0.7
    n = 200000; lo = -8 * sig; h = (c - lo) / n; num = den = 0.0
    for i in range(n):
        x = lo + (i + 0.5) * h; f = math.exp(-x * x / (2 * sig * sig))
        num += math.exp(x / tau) * f; den += f
    print('eq43 check: integral', math.log(num / den), 'closed', log_mu(c, sig, tau))
    # Algo 1 vs brute force means (N=1000,k=3)
    N, k, R = 1000, 3, 4000
    r = Mulberry32(5); e = [0.0] * k
    for _ in range(R):
        v = topk_zero_mean(r, N, k, 1.0); e = [a + b / R for a, b in zip(e, v)]
    r = Mulberry32(6); b = [0.0] * k
    for _ in range(R):
        v = sorted((gauss(r) for _ in range(N)), reverse=True)[:k]; b = [a + x / R for a, x in zip(b, v)]
    print('algo1 means', ['%.3f' % x for x in e], 'brute', ['%.3f' % x for x in b])
