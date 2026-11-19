"""C4 toy MDP: constant per-token unmask prob s, exact DP over (step, #masked).
Same algorithm the page uses. Run: python3 -I c4_toy_mdp.py"""
from math import comb

def dp(L, T, s, gamma, mode):
    """Return Pfin[N], Cfin[N] (N=1..T) and P_unfinished. mode: 'train' (no fallback) / 'test' (fallback)."""
    mass = [0.0]*(L+1); corr = [0.0]*(L+1); mass[L] = 1.0; corr[L] = 1.0
    Pfin = [0.0]*(T+1); Cfin = [0.0]*(T+1)
    for n in range(T):
        nm = [0.0]*(L+1); nc = [0.0]*(L+1)
        for m in range(1, L+1):
            if mass[m] == 0.0: continue
            for j in range(m+1):
                p = comb(m, j) * s**j * (1-s)**(m-j)
                if p == 0.0: continue
                jj = j
                if j == 0:
                    if mode == 'train':
                        nm[m] += mass[m]*p; nc[m] += corr[m]*p; continue
                    jj = 1
                g = gamma**(jj-1)
                r = m - jj
                if r == 0:
                    Pfin[n+1] += mass[m]*p; Cfin[n+1] += corr[m]*p*g
                else:
                    nm[r] += mass[m]*p; nc[r] += corr[m]*p*g
        mass, corr = nm, nc
    return Pfin, Cfin, sum(mass)

def stats(L, T, s, gamma, p0, alpha, mode):
    Pfin, Cfin, Pun = dp(L, T, s, gamma, mode)
    def f(N): return 1.0 if alpha == 0 else (1 - N/T)**alpha
    Rm = p0*sum(Cfin[N]*f(N) for N in range(1, T+1))
    Ra = sum(p0*Cfin[N] - alpha*(N/T)*Pfin[N] for N in range(1, T+1)) - alpha*Pun
    acc = p0*sum(Cfin)
    nfe = sum(N*Pfin[N] for N in range(1, T+1)) + T*Pun
    return dict(Rm=Rm, Ra=Ra, acc=acc, nfe=nfe, unf=Pun)

S = [round(0.005*i, 3) for i in range(1, 201)]

def best(L, T, gamma, p0, alpha, mode, key):
    vals = [(stats(L, T, s, gamma, p0, alpha, mode)[key], s) for s in S]
    m = max(v for v, _ in vals)
    s = min(s for v, s in vals if abs(v - m) < 1e-12)  # tie (|Δ|<1e-12) -> smallest s, same as page
    return s, stats(L, T, s, gamma, p0, alpha, mode)

if __name__ == '__main__':
    L = T = 16; gamma = 0.90; p0 = 0.8
    # sanity: s=1 -> one step, all L tokens at once
    st = stats(L, T, 1.0, gamma, p0, 1, 'train')
    assert abs(st['nfe'] - 1) < 1e-12 and abs(st['acc'] - p0*gamma**(L-1)) < 1e-12
    # mass conservation
    Pf, Cf, Pu = dp(L, T, 0.3, gamma, 'train'); assert abs(sum(Pf)+Pu-1) < 1e-12
    Pf, Cf, Pu = dp(L, T, 0.3, gamma, 'test'); assert abs(sum(Pf)+Pu-1) < 1e-12 and Pu < 1e-15
    for gamma in (0.90, 0.95, 0.80):
      for mode in ('train', 'test'):
        print(f'== L=T={L} gamma={gamma} p0={p0} mode={mode}')
        for key in ('Rm', 'Ra'):
            for a in (0, 0.3, 1, 3, 10):
                s, st = best(L, T, gamma, p0, a, mode, key)
                print(f'  {key} alpha={a:<4} s*={s:.3f} E[R]={st[key]:.4f} acc={st["acc"]:.4f} nfe={st["nfe"]:.3f} unf={st["unf"]:.4f}')
    print('== fixed-s table (L=T=16, gamma=0.9, p0=0.8, alpha=1)')
    for mode in ('train', 'test'):
        for s in (0.1, 0.3, 0.5, 0.7, 1.0):
            st = stats(16, 16, s, 0.9, 0.8, 1, mode)
            print(f'  {mode} s={s}: ' + ' '.join(f'{k}={v:.4f}' for k, v in st.items()))

    print('== acceptance: train-mode optimum s*, then evaluated in test mode (fallback on)')
    for (L, g) in ((16, 0.90), (8, 0.90), (32, 0.90), (16, 0.95)):
        for key in ('Rm', 'Ra'):
            row = []
            for a in (0, 0.3, 1, 3, 10):
                s, _ = best(L, L, g, 0.8, a, 'train', key)
                te = stats(L, L, s, g, 0.8, a, 'test')
                row.append(f'a={a}: s*={s:.3f} testNFE={te["nfe"]:.2f} testAcc={te["acc"]:.3f}')
            print(f' L=T={L} g={g} {key}: ' + ' | '.join(row))
    print('== frontier points (test mode, L=T=16, g=0.9, p0=0.8)')
    for s in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        te = stats(16, 16, s, 0.9, 0.8, 0, 'test'); print(f'  s={s}: NFE={te["nfe"]:.3f} acc={te["acc"]:.4f}')
