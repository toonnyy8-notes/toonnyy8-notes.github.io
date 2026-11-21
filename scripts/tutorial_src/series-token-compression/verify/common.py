"""Shared algorithms for the token-compression lineage widgets (mirror of the page JS; design.md 4.0).
Scene, PRNG and Hilbert order are identical to tutorial_src/217/verify/common.py; only the patch size differs
(PX = 8 -> 16 x 16 = 256 tokens, so average linkage stays cheap in the browser)."""
import math
import numpy as np

S = 128          # image side (px)
PX = 8           # patch side (px)
G = S // PX      # 16
N = G * G        # 256 tokens


def mulberry32(a):
    a &= 0xFFFFFFFF
    def r():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return r


def gauss(r):
    u1, u2 = r(), r()
    return math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)


def hilbert_d2xy(n, d):
    x = y = 0
    s, t = 1, d
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


HIL_PID = [y * G + x for (x, y) in (hilbert_d2xy(G, d) for d in range(N))]


def lerp(a, b, s):
    return [a[i] + (b[i] - a[i]) * s for i in range(3)]


def scene(name='landscape'):
    """Scenes of 217/common.py (unchanged): img[y][x] = [r,g,b]."""
    r = mulberry32(1)
    img = []
    for y in range(S):
        row = []
        for x in range(S):
            n1, n2 = r(), r()
            u, v = (x + 0.5) / S, (y + 0.5) / S
            if name == 'checker':
                c = [0.85, 0.80, 0.70] if ((x // 32) + (y // 32)) % 2 == 0 else [0.30, 0.40, 0.55]
                c = [min(1.0, max(0.0, ci + 0.01 * (n2 - 0.5))) for ci in c]
                row.append(c)
                continue
            if v < 0.62:
                c = lerp([0.55, 0.75, 0.95], [0.85, 0.92, 0.98], v / 0.62)
            else:
                c = lerp([0.45, 0.58, 0.32], [0.36, 0.48, 0.26], (v - 0.62) / 0.38)
                c = [ci + 0.06 * (n1 - 0.5) for ci in c]
            if math.hypot(u - 0.78, v - 0.18) < 0.09:
                c = [0.98, 0.86, 0.45]
            if abs(u - 0.28) < 0.025 and 0.45 < v < 0.70:
                c = [0.40, 0.28, 0.18]
            if math.hypot(u - 0.28, v - 0.38) < 0.15:
                c = [0.20 + 0.18 * (n1 - 0.5), 0.45 + 0.18 * (n1 - 0.5), 0.20]
            if 0.52 < u < 0.70 and 0.66 < v < 0.82:
                c = [0.95, 0.95, 0.92]
                if 0.69 < v < 0.79 and (x // 2) % 3 == 0:
                    c = [0.10, 0.10, 0.12]
            c = [min(1.0, max(0.0, ci + 0.01 * (n2 - 0.5))) for ci in c]
            row.append(c)
        img.append(row)
    return np.array(img)


def noise_field():
    r = mulberry32(2)
    return np.array([[[gauss(r) for _ in range(3)] for _ in range(S)] for _ in range(S)])


def features(img):
    """h[pid] (pid = row*G + col): 192-d flattened patch, order (py, px, ch)."""
    return img.reshape(G, PX, G, PX, 3).transpose(0, 2, 1, 3, 4).reshape(N, PX * PX * 3)


IMG = scene('landscape')
EPS = noise_field()
H1 = features(IMG)                      # clean landscape features
H_CHK = features(scene('checker'))      # clean checker features
BATCH = [H1, H_CHK]                     # C3 batch: index 0 landscape, 1 checker


def feats_t(t):
    """C4 pre-core features: landscape at t (t = 1 clean, RTI/JiT convention)."""
    return features(t * IMG + (1 - t) * EPS)


# ---------------- groupings: return label[pid] ----------------
def gaps(h):
    return [float(np.linalg.norm(h[HIL_PID[j + 1]] - h[HIL_PID[j]])) for j in range(N - 1)]


def runs_to_labels(cuts):
    cs = set(cuts)
    lab = [0] * N
    g = 0
    for j in range(N):
        lab[HIL_PID[j]] = g
        if j in cs:
            g += 1
    return lab


def m_grid(R):
    """equal-length Hilbert runs; for R = 4^k these are aligned square blocks (R=64 -> 2x2)."""
    L = N // R
    return runs_to_labels([r * L - 1 for r in range(1, R)])


def m_rti(h, R):
    d = gaps(h)
    idx = sorted(range(N - 1), key=lambda j: (-d[j], j))[:R - 1]
    return runs_to_labels(idx)


def thresh_batch(R):
    """Batch-budget idealisation (C3 method 4): pool the gaps of both images, sort by (-gap, image, j)
    and keep exactly the first 2R-2 cuts (ties broken by order, never by float threshold).
    Returns (labels_landscape, labels_checker, u_star = gap of the last kept cut)."""
    pool = []
    for b, h in enumerate(BATCH):
        d = gaps(h)
        pool += [(-d[j], b, j) for j in range(N - 1)]
    pool.sort()
    keep = pool[:2 * R - 2]
    cuts = [[j for (_, bb, j) in keep if bb == b] for b in range(2)]
    return runs_to_labels(cuts[0]), runs_to_labels(cuts[1]), -keep[-1][0]


def unit_centered(f, mu):
    c = f - mu
    n = np.linalg.norm(c)
    return c / n if n > 1e-12 else c * 0.0


def m_tome(h, R):
    """ToMe bipartite soft matching applied round after round (ToMe App. D order) until R tokens remain."""
    mu = h.mean(axis=0)
    toks = [dict(f=h[p].copy(), s=1, mem=[p]) for p in range(N)]
    while len(toks) > R:
        A, B = toks[0::2], toks[1::2]
        r = min(len(A), len(toks) - R)
        ua = [unit_centered(a['f'], mu) for a in A]
        ub = [unit_centered(b['f'], mu) for b in B]
        best, bidx = [], []
        for i in range(len(A)):
            sc = [float(ua[i] @ ub[k]) for k in range(len(B))]
            k = max(range(len(B)), key=lambda kk: (sc[kk], -kk))   # first max
            best.append(sc[k]); bidx.append(k)
        order = sorted(range(len(A)), key=lambda i: (-best[i], i))
        src = set(order[:r])
        for i in sorted(src):
            b = B[bidx[i]]; a = A[i]
            b['f'] = (b['s'] * b['f'] + a['s'] * a['f']) / (b['s'] + a['s'])
            b['s'] += a['s']; b['mem'] += a['mem']
        toks = [A[i] for i in range(len(A)) if i not in src] + B
    lab = [0] * N
    for g, tk in enumerate(toks):
        for p in tk['mem']:
            lab[p] = g
    return lab


def dendrogram(h):
    """Full average-linkage merge list on S_ij = cos(centered h_i, h_j): [(i, j, score), ...], N-1 merges.
    Ties: smallest (i, j) in row-major scan (page: i asc, j asc, strict >). Cluster kept at the smaller
    index. Average linkage has no inversions, so cutting by merge count == stopping at a threshold."""
    mu = h.mean(axis=0)
    U = np.array([unit_centered(h[p], mu) for p in range(N)])
    Ssum = U @ U.T
    size = np.ones(N)
    alive = np.ones(N, bool)
    iu = np.triu(np.ones((N, N), bool), 1)
    merges = []
    while alive.sum() > 1:
        avg = np.where(iu & alive[:, None] & alive[None, :], Ssum / np.outer(size, size), -np.inf)
        k = int(np.argmax(avg)); bi, bj = divmod(k, N)
        merges.append((bi, bj, float(avg[bi, bj])))
        Ssum[bi, :] += Ssum[bj, :]
        Ssum[:, bi] += Ssum[:, bj]
        size[bi] += size[bj]
        alive[bj] = False
    return merges


def cut(merges, m):
    """labels after replaying the first m merges (groups numbered by smallest member)."""
    root = list(range(N))
    for (i, j, _) in merges[:m]:
        for p in range(N):
            if root[p] == j:
                root[p] = i
    ids = {r: g for g, r in enumerate(sorted(set(root)))}
    return [ids[r] for r in root]


_DEN = None


def linkage_batch(R):
    """Batch-budget idealisation (C3 method 5): pool merge scores of both dendrograms, sort by
    (-score, image, merge index), keep exactly the first 2N-2R merges. Returns (lab_l, lab_c, tau)."""
    global _DEN
    if _DEN is None:
        _DEN = [dendrogram(h) for h in BATCH]
    pool = sorted((-sc, b, m) for b in range(2) for m, (_, _, sc) in enumerate(_DEN[b]))
    keep = pool[:2 * N - 2 * R]
    cnt = [sum(1 for (_, bb, _) in keep if bb == b) for b in range(2)]
    return cut(_DEN[0], cnt[0]), cut(_DEN[1], cnt[1]), -keep[-1][0]


# ---------------- metrics ----------------
def groups_of(lab):
    gs = {}
    for p, g in enumerate(lab):
        gs.setdefault(g, []).append(p)
    return list(gs.values())


def ev(h, lab):
    hbar = h.mean(axis=0)
    den = float(((h - hbar) ** 2).sum())
    if den < 1e-18:
        return None
    num = 0.0
    for ps in groups_of(lab):
        m = h[ps]
        num += float(((m - m.mean(axis=0)) ** 2).sum())
    return 1 - num / den


def ncomp(ps):
    cs = set(ps); seen = set(); k = 0
    for c in ps:
        if c in seen:
            continue
        k += 1; st = [c]; seen.add(c)
        while st:
            p = st.pop(); x, y = p % G, p // G
            for q in ((p + 1) if x + 1 < G else None, (p - 1) if x > 0 else None,
                      (p + G) if y + 1 < G else None, (p - G) if y > 0 else None):
                if q is not None and q in cs and q not in seen:
                    seen.add(q); st.append(q)
    return k


def spread(ps):
    cx = sum(p % G for p in ps) / len(ps); cy = sum(p // G for p in ps) / len(ps)
    return sum(math.hypot(p % G - cx, p // G - cy) for p in ps) / len(ps)


def stats(h, lab):
    gs = groups_of(lab)
    return dict(K=len(gs), EV=ev(h, lab), spread=sum(len(ps) * spread(ps) for ps in gs) / N,
                conn=sum(1 for ps in gs if ncomp(ps) == 1) / len(gs), maxsize=max(len(ps) for ps in gs))


def restore(y, h, lab):
    """C4: relative error E(beta) = sum||y_j - ybar_c - beta (h_j - hbar_c)||^2 / sum||y_j - ybar||^2.
    Returns (E0, E1, beta*, Ebeta*)."""
    den = float(((y - y.mean(axis=0)) ** 2).sum())
    a = b = c = 0.0
    for ps in groups_of(lab):
        dy = y[ps] - y[ps].mean(axis=0)
        dh = h[ps] - h[ps].mean(axis=0)
        a += float((dy * dy).sum()); b += float((dy * dh).sum()); c += float((dh * dh).sum())
    E = lambda beta: (a - 2 * beta * b + beta * beta * c) / den
    bs = b / c if c > 1e-12 else 0.0
    return E(0.0), E(1.0), bs, E(bs)
