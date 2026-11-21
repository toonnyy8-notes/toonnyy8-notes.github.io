"""C2: where to compress. Per-block forward FLOPs C(L) = 24 L d^2 + 4 L^2 d (m = 4, multiply-add = 2)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math

def C(L, d): return 24 * L * d * d + 4 * L * L * d
def C_kv(Lq, Lkv, d): return 24 * Lq * d * d + 4 * Lq * Lkv * d   # ToDo: projections & MLP full (conservative)
def C_sa(L, Lr, d): return 8 * Lr * d * d + 4 * Lr * Lr * d + 16 * L * d * d  # ToMeSD: merge for self-attn only, MLP full

def speedups(N, d, T, kappa):
    L, Lr = N + T, kappa * N + T
    dense17 = 17 * C(L, d)
    s_all = dense17 / (17 * C_sa(L, Lr, d))
    s_kv = dense17 / (17 * C_kv(L, Lr, d))
    s_core = dense17 / (6 * C(L, d) + 11 * C(Lr, d))
    r = math.sqrt(kappa)
    dense18 = 18 * C(L, d)
    s_nest = dense18 / (6 * C(L, d) + 6 * C(r * N + T, d) + 6 * C(kappa * N + T, d))
    return s_all, s_kv, s_core, s_nest

def qshare(L, d): return L / (6 * d + L)

if __name__ == '__main__':
    print('quadratic share L/(6d+L), d=768:', ', '.join(f'L={L}: {qshare(L,768):.4f}' for L in (256, 1024, 4096, 16384)))
    print('core-only bound (kappa->0, T=0) = 17/6 =', round(17/6, 4))
    print('nested bound (kappa->0, T=0) = 18/6 =', 3.0)
    print()
    print('   N     d    T  kappa | tomesd kv     core   nest')
    for N in (256, 1024, 4096, 16384):
        for kappa in (0.5, 0.25, 0.125, 1/16):
            a, k, c, n = speedups(N, 768, 0, kappa)
            print(f'{N:6d} {768:4d} {0:4d} {kappa:6.4f} | {a:6.3f} {k:6.3f} {c:6.3f} {n:6.3f}')
    print()
    for T in (0, 128, 256):
        for kappa in (0.5, 0.25):
            a, k, c, n = speedups(1024, 768, T, kappa)
            print(f'{1024:6d} {768:4d} {T:4d} {kappa:6.4f} | {a:6.3f} {k:6.3f} {c:6.3f} {n:6.3f}')
    print()
    for d in (384, 1152, 1536):
        a, k, c, n = speedups(1024, d, 0, 0.25)
        print(f'{1024:6d} {d:4d} {0:4d} {0.25:6.4f} | {a:6.3f} {k:6.3f} {c:6.3f} {n:6.3f}')
    # core-only bound with text tokens
    for T in (128, 256):
        L = 1024 + T
        print(f'core-only bound N=1024 T={T}: {17*C(L,768)/(6*C(L,768)+11*C(T,768)):.4f}')
