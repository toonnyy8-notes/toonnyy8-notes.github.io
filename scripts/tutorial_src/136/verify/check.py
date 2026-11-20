"""Acceptance values for tutorial 136 widgets (same algorithms as the page).
Widget numbering follows page order (round 1): C1 = Fig.1 paired diffs (S4),
C2 = token KL & logit gradients (S5), C3 = rollout-stability gap (S6b), C4 = spectrum (S7)."""
from math import log, exp, sqrt, log1p

# ---------- C1: matched comparisons on Fig. 1 bar labels ----------
ACC = {('On','F',1):72,('On','F',5):72,('On','Rev',1):72,('On','Rev',5):35,
       ('Off','F',1):71,('Off','F',5):73,('Off','Rev',1):35,('Off','Rev',5):49}
SPA = {('On','F',1):85.3,('On','F',5):51.9,('On','Rev',1):88.5,('On','Rev',5):58.5,
       ('Off','F',1):88.8,('Off','F',5):59.5,('Off','Rev',1):89.6,('Off','Rev',5):60.0}
LEV = [('On','Off'),('F','Rev'),(1,5)]
def pair_diffs(M, f):
    out = []; others = [i for i in range(3) if i != f]
    for a in LEV[others[0]]:
        for b in LEV[others[1]]:
            k1 = [None]*3; k1[others[0]] = a; k1[others[1]] = b
            k2 = list(k1); k1[f] = LEV[f][0]; k2[f] = LEV[f][1]
            out.append(((a, b), M[tuple(k1)], M[tuple(k2)], round(abs(M[tuple(k1)] - M[tuple(k2)]), 1)))
    return out

# ---------- S5 recap: 2-token example (logits -> softmax -> KL -> gradient) ----------
def two_token():
    zS = [1.0, 0.0]; m = max(zS); e = [exp(z - m) for z in zS]; S = [x / sum(e) for x in e]
    T = [0.5, 0.5]
    DF = sum(t * (log(t) - log(s)) for s, t in zip(S, T))
    DR = sum(s * (log(s) - log(t)) for s, t in zip(S, T))
    gF = [s - t for s, t in zip(S, T)]
    gR = [s * ((log(s) - log(t)) - DR) for s, t in zip(S, T)]
    return S, DF, DR, gF, gR

# ---------- C2: token-level KL and logit gradients (Eq. 2-5, 13, 43, 65) ----------
SHAPE = [0.4, 0.3, 0.2, 0.1]          # tokens B..E share of (1 - p(A)), same for S and T
def dist(a): return [a] + [(1 - a) * s for s in SHAPE]
def kl_parts(aS, aT):
    S, T = dist(aS), dist(aT)
    r = [log(s) - log(t) for s, t in zip(S, T)]
    DF = sum(t * (log(t) - log(s)) for s, t in zip(S, T))
    DR = sum(s * ri for s, ri in zip(S, r))
    gF = [s - t for s, t in zip(S, T)]
    gR = [s * (ri - DR) for s, ri in zip(S, r)]
    n2 = lambda g: sqrt(sum(x * x for x in g))
    return dict(DF=DF, DR=DR, gF_A=gF[0], gR_A=gR[0], nF=n2(gF), nR=n2(gR), dr=max(r) - min(r),
                maxF=max(abs(x) for x in gF), sumF=sum(gF), sumR=sum(gR))

# ---------- C3: Theorem C.1 vs C.2 counterexample, raw gaps (Eq. 27, 48-57, 59) ----------
def bound_parts(delta, eps, B=1.0):
    dr = log1p(-delta) - log(delta)                # Δr = log((1-δ)/δ)
    return dict(dr=dr, rev=eps * sqrt(2) / 8 * dr,  # Eq. 57
                fwd=eps * sqrt(2) / 2 * (0.5 - delta),  # 本頁推導: same counterexample, forward KL
                fbound=2 * sqrt(2) * B * eps, rbound=2 * B * dr * eps)

# ---------- C4: rollout-policy spectrum (Eq. 75-80, α=0.2, c=2 ln 3) ----------
PS = [0.45, 0.25, 0.15, 0.08, 0.05, 0.02]
PT = [0.03, 0.57, 0.25, 0.10, 0.0499, 0.0001]
ALPHA, C = 0.2, 2 * log(3)
def softmax(z):
    m = max(z); e = [exp(x - m) for x in z]; s = sum(e); return [x / s for x in e]
def spectrum(lam, safe=True):
    lam = round(lam * 100) / 100                   # same rounding as the page
    lS = [log(p) for p in PS]; lT = [log(p) for p in PT]
    r = [a - b for a, b in zip(lS, lT)]
    zhat = [(1 + lam) / 2 * a + (1 - lam) / 2 * b for a, b in zip(lS, lT)]
    if not safe or abs(lam) <= 1: z = zhat
    elif lam > 1:
        bS = [(1 if a >= max(lS) + log(ALPHA) else 0) * min(max(x, 0), C) for a, x in zip(lS, r)]
        z = [a + (lam - 1) / 2 * b for a, b in zip(lS, bS)]
    else:
        bT = [(1 if a >= max(lT) + log(ALPHA) else 0) * min(max(-x, 0), C) for a, x in zip(lT, r)]
        z = [a + (-lam - 1) / 2 * b for a, b in zip(lT, bT)]
    p = softmax(z)
    anc = lS if lam > 1 else lT if lam < -1 else None
    amp = None
    if anc is not None:
        d = [a - b for a, b in zip(z, anc)]; amp = exp(max(d) - min(d))
    return p, amp

if __name__ == '__main__':
    print('== C1 (Fig. 1 paired diffs) ==')
    names = ['rollout', 'KL', 'LR']
    for M, nm in [(ACC, 'acc'), (SPA, 'sparsity')]:
        for f in range(3):
            print(nm, names[f], pair_diffs(M, f))
    assert [x[3] for x in pair_diffs(SPA, 2)] == [33.4, 30.0, 29.3, 29.6]
    assert [x[3] for x in pair_diffs(ACC, 0)] == [1, 1, 37, 14]
    print('== S5 two-token recap ==')
    S, DF, DR, gF, gR = two_token()
    print(f'pi_S={S[0]:.4f},{S[1]:.4f} DF={DF:.4f} DR={DR:.4f} gF={gF[0]:.4f},{gF[1]:.4f} gR={gR[0]:.4f},{gR[1]:.4f}')
    print('== C2 ==')
    for aS, aT in [(0.5, 0.01), (0.3, 0.3), (0.5, 1e-6), (1e-6, 0.6), (0.9, 1e-6)]:
        v = kl_parts(aS, aT)
        print(f'aS={aS:g} aT={aT:g} ' + ' '.join(f'{k}={x:.4f}' for k, x in v.items()))
        assert abs(v['sumF']) < 1e-12 and abs(v['sumR']) < 1e-9
        assert v['maxF'] <= 1 and v['nF'] <= sqrt(2) and v['nR'] <= v['dr'] + 1e-12
    print('gR_A exact for (1e-6,0.6):', kl_parts(1e-6, 0.6)['gR_A'])
    print('== C3 ==')
    for eps in (0.1, 1.0):
        for d in [0.25, 0.01, 1e-4, 1e-8, 1e-12]:
            v = bound_parts(d, eps)
            print(f'eps={eps} delta={d:g} ' + ' '.join(f'{k}={x:.4f}' for k, x in v.items()))
            assert v['fwd'] <= v['fbound'] and v['rev'] <= v['rbound']
    dx = 1 / (1 + exp(16)); print(f'crossing delta={dx:.4e} rev(eps=1)={bound_parts(dx,1)["rev"]:.6f} fbound={2*sqrt(2):.6f}')
    print('== C4 ==')
    for lam in [-2, -1.5, -1, 0, 1, 1.5, 2]:
        for safe in (True, False):
            p, amp = spectrum(lam, safe)
            print(f'lam={lam:+.1f} safe={int(safe)} p=' + ' '.join(f'{x:.4f}' for x in p)
                  + ' amp=' + ('—' if amp is None else f'{amp:.4f}'))
    for i in range(81):
        lam = -2 + 0.05 * i                         # float grid as the slider produces it
        p, amp = spectrum(lam, True)
        if abs(round(lam * 100) / 100) <= 1: assert amp is None
        else: assert amp <= 3 + 1e-12
    p, _ = spectrum(1); assert max(abs(a - b) for a, b in zip(p, PS)) < 1e-12
    p, _ = spectrum(-1); assert max(abs(a - b) for a, b in zip(p, PT)) < 1e-12
    print('all asserts passed')
