"""C1: one ToMe layer on 16 toy tokens (4x4), ToMeSD partitions, unmerge, proportional attention."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import numpy as np
from common import mulberry32

X = np.array([
    [0.58, 0.76, 0.94], [0.60, 0.77, 0.95], [0.62, 0.79, 0.95], [0.98, 0.86, 0.45],
    [0.64, 0.80, 0.96], [0.66, 0.82, 0.97], [0.22, 0.45, 0.20], [0.68, 0.83, 0.97],
    [0.45, 0.58, 0.32], [0.44, 0.57, 0.31], [0.40, 0.28, 0.18], [0.43, 0.56, 0.30],
    [0.42, 0.55, 0.30], [0.41, 0.54, 0.29], [0.40, 0.53, 0.28], [0.39, 0.52, 0.27]])
n = 16
MU = X.mean(axis=0)
KEY = X - MU                       # toy key = centered feature
SCALE = 4.0                        # q, k = SCALE * centered feature (推定)


def partition(mode, seed=1):
    if mode == 'alt':
        return list(range(0, n, 2)), list(range(1, n, 2))
    r = mulberry32(seed)
    dst = []
    for by in range(2):
        for bx in range(2):
            k = min(3, int(r() * 4))
            dst.append((2 * by + k // 2) * 4 + (2 * bx + k % 2))
    dst.sort()
    return [i for i in range(n) if i not in dst], dst


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def tome_layer(mode, r, seed=1):
    A, B = partition(mode, seed)
    best, bidx = [], []
    for a in A:
        sc = [cos(KEY[a], KEY[b]) for b in B]
        k = max(range(len(B)), key=lambda kk: (sc[kk], -kk))
        best.append(sc[k]); bidx.append(k)
    order = sorted(range(len(A)), key=lambda i: (-best[i], i))
    src = order[:r]
    groups = {b: [b] for b in B}
    for i in src:
        groups[B[bidx[i]]].append(A[i])
    unm = [A[i] for i in range(len(A)) if i not in src]
    toks = [[a] for a in unm] + [sorted(groups[b]) for b in B]     # ToMe App. D order: [unmerged A, B]
    feat = np.array([X[g].mean(axis=0) for g in toks])             # size-weighted average (sizes start at 1)
    size = np.array([len(g) for g in toks], float)
    return dict(A=A, B=B, best=best, bidx=bidx, src=src, toks=toks, feat=feat, size=size)


def unmerge_mse(res):
    xh = np.zeros_like(X)
    for g, f in zip(res['toks'], res['feat']):
        xh[g] = f
    return float(((X - xh) ** 2).sum(axis=1).mean())


def attn(q, K, V, bias=None):
    lg = (SCALE * q) @ (SCALE * K).T / math.sqrt(3)
    if bias is not None:
        lg = lg + bias
    w = np.exp(lg - lg.max()); w /= w.sum()
    return w @ V


def attention_check(res, qi):
    q = KEY[qi]
    dense = attn(q, KEY, X)
    km = res['feat'] - MU
    plain = attn(q, km, res['feat'])
    prop = attn(q, km, res['feat'], np.log(res['size']))
    rep = np.repeat(np.arange(len(res['toks'])), res['size'].astype(int))
    dup = attn(q, km[rep], res['feat'][rep])
    return (float(np.linalg.norm(plain - dense)), float(np.linalg.norm(prop - dense)),
            float(np.linalg.norm(prop - dup)))


if __name__ == '__main__':
    for mode, seed in (('alt', 1), ('rand2x2', 1), ('rand2x2', 2)):
        A, B = partition(mode, seed)
        print(f'partition {mode} seed={seed}: A={A} B(dst)={B}')
    print()
    for mode in ('alt', 'rand2x2'):
        res = tome_layer(mode, 0)
        print(f'[{mode}] best match per A (A -> B, cos):',
              ', '.join(f'{a}->{res["B"][k]} {s:.3f}' for a, k, s in zip(res['A'], res['bidx'], res['best'])))
        rmax = len(res['A'])
        for r in sorted(set([0, 2, 4, 6, 8, rmax])):
            if r > rmax: continue
            res = tome_layer(mode, r)
            pl, pr, du = attention_check(res, 0)
            pl6, pr6, _ = attention_check(res, 6)
            print(f'  r={r:2d} tokens={len(res["toks"]):2d} groups={[g for g in res["toks"] if len(g) > 1]} '
                  f'unmergeMSE={unmerge_mse(res):.4e} | q=0: plain={pl:.4e} prop={pr:.4e} |prop-dup|={du:.1e}'
                  f' | q=6: plain={pl6:.4e} prop={pr6:.4e}')
        print()
