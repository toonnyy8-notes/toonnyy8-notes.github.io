"""Shared algorithms for the RTI tutorial widgets (mirror of the page JS; see design.md 4.0)."""
import math
import numpy as np

G = 32          # patch grid side
N = G * G       # 1024 patch tokens
PX = 4          # pixels per patch side (toy scene)
S = G * PX      # 128 px image


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


# ---------- orders: position j -> (col, row) ----------
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


def morton_d2xy(n, d):
    x = y = 0
    b = 0
    while (1 << (2 * b)) < n * n:
        x |= ((d >> (2 * b)) & 1) << b
        y |= ((d >> (2 * b + 1)) & 1) << b
        b += 1
    return x, y


def raster_d2xy(n, d):
    return d % n, d // n


ORDERS = {'hilbert': hilbert_d2xy, 'morton': morton_d2xy, 'raster': raster_d2xy}


def order(name, n):
    return [ORDERS[name](n, d) for d in range(n * n)]


def components(cells):
    cs = set(cells)
    seen = set()
    k = 0
    for c in cells:
        if c in seen:
            continue
        k += 1
        st = [c]
        seen.add(c)
        while st:
            x, y = st.pop()
            for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nb in cs and nb not in seen:
                    seen.add(nb)
                    st.append(nb)
    return k


def spread(cells, w=None):
    """Eq. 6 with weights w (uniform if None): sum_j w_j ||p_j - p_r||_2."""
    m = len(cells)
    if w is None:
        w = [1.0 / m] * m
    cx = sum(wi * c[0] for wi, c in zip(w, cells))
    cy = sum(wi * c[1] for wi, c in zip(w, cells))
    return sum(wi * math.hypot(c[0] - cx, c[1] - cy) for wi, c in zip(w, cells))


# ---------- toy scenes ----------
def lerp(a, b, s):
    return [a[i] + (b[i] - a[i]) * s for i in range(3)]


def scene(name):
    """Return img[y][x] = [r,g,b] in [0,1], S x S."""
    r = mulberry32(1)
    img = []
    for y in range(S):
        row = []
        for x in range(S):
            n1, n2 = r(), r()          # always two draws per pixel, row-major
            u, v = (x + 0.5) / S, (y + 0.5) / S
            if name == 'landscape':
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
            else:  # checker: 4x4 board, each square 8x8 patches = 32 px
                if ((x // 32) + (y // 32)) % 2 == 0:
                    c = [0.85, 0.80, 0.70]
                else:
                    c = [0.30, 0.40, 0.55]
            c = [min(1.0, max(0.0, ci + 0.01 * (n2 - 0.5))) for ci in c]
            row.append(c)
        img.append(row)
    return img


def noise_field():
    r = mulberry32(2)
    return [[[gauss(r) for _ in range(3)] for _ in range(S)] for _ in range(S)]


def mix(img, eps, t):
    return [[[t * img[y][x][c] + (1 - t) * eps[y][x][c] for c in range(3)] for x in range(S)] for y in range(S)]


def patch_features(img):
    """h[pid] (pid = row*G + col): 48-d flattened patch, order (py, px, ch)."""
    h = np.zeros((N, PX * PX * 3))
    for pr in range(G):
        for pc in range(G):
            k = 0
            for yy in range(PX):
                for xx in range(PX):
                    for ch in range(3):
                        h[pr * G + pc, k] = img[pr * PX + yy][pc * PX + xx][ch]
                        k += 1
    return h


def detail(img):
    """Per-patch mean gradient energy (forward differences, 0 at the last row/col)."""
    e = np.zeros(N)
    for y in range(S):
        for x in range(S):
            g = 0.0
            for ch in range(3):
                if x + 1 < S:
                    g += (img[y][x + 1][ch] - img[y][x][ch]) ** 2
                if y + 1 < S:
                    g += (img[y + 1][x][ch] - img[y][x][ch]) ** 2
            e[(y // PX) * G + (x // PX)] += g
    return e / (PX * PX)


def detail_share(det, frac):
    s = np.sort(det)[::-1]
    k = int(round(frac * N))
    return s[:k].sum() / s.sum()


# ---------- groupings: return label[pid] ----------
HIL = order('hilbert', G)
HIL_PID = [y * G + x for (x, y) in HIL]


def gaps(h):
    return [float(np.linalg.norm(h[HIL_PID[j + 1]] - h[HIL_PID[j]])) for j in range(N - 1)]


def runs_to_labels(cut_after):
    """cut_after: sorted list of j (cut between position j and j+1, 0-based)."""
    lab = [0] * N
    g = 0
    cs = set(cut_after)
    for j in range(N):
        lab[HIL_PID[j]] = g
        if j in cs:
            g += 1
    return lab


def adaptive(h, R):
    d = gaps(h)
    idx = sorted(range(N - 1), key=lambda j: (-d[j], j))[:R - 1]
    return runs_to_labels(sorted(idx))


def fixed(R):
    return runs_to_labels([round(r * N / R) - 1 for r in range(1, R)])


def featsim(h, R):
    hc = h - h.mean(axis=0)
    nrm = np.linalg.norm(hc, axis=1)
    hn = np.where(nrm[:, None] > 1e-12, hc / np.maximum(nrm, 1e-12)[:, None], 0.0)
    anchors = [(r * N) // R for r in range(R)]          # raster pid
    sim = hn @ hn[anchors].T                             # N x R
    return [int(np.argmax(sim[p])) for p in range(N)]   # argmax = first max


def ev(h, lab):
    lab = np.array(lab)
    hbar = h.mean(axis=0)
    den = ((h - hbar) ** 2).sum()
    if den < 1e-18:
        return None
    num = 0.0
    for g in np.unique(lab):
        m = h[lab == g]
        num += ((m - m.mean(axis=0)) ** 2).sum()
    return 1 - num / den


def ev_skip(h, R):
    hbar = h.mean(axis=0)
    dev = ((h - hbar) ** 2).sum(axis=1)
    den = dev.sum()
    if den < 1e-18:
        return None
    keep = sorted(range(N), key=lambda p: (-dev[p], p))[:R]
    return 1 - (den - dev[keep].sum()) / den


def group_stats(lab, det):
    groups = {}
    for p, g in enumerate(lab):
        groups.setdefault(g, []).append(p)
    sp, fin, dd = [], [], []
    for g, ps in groups.items():
        cells = [(p % G, p // G) for p in ps]
        sp.append(spread(cells))
        fin.append(-math.log2(len(ps)))
        dd.append(float(np.mean(det[ps])))
    fin, dd = np.array(fin), np.array(dd)
    if fin.std() < 1e-12 or dd.std() < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(fin, dd)[0, 1])
    sizes = [len(ps) for ps in groups.values()]
    fine = [p for ps in groups.values() if len(ps) == 1 for p in ps]
    coarse = [p for ps in groups.values() if len(ps) >= 16 for p in ps]
    ratio = (float(np.mean(det[fine])) / float(np.mean(det[coarse]))
             if fine and coarse and np.mean(det[coarse]) > 1e-12 else None)
    return dict(groups=len(groups), spread=float(np.mean(sp)), corr=corr,
                maxsize=max(sizes), ratio=ratio, nfine=len(fine))


# ---------- READ: averaged rotary phasor ----------
def rope_freqs(F=16, base=10000.0):
    return [base ** (-i / F) for i in range(F)]


def phasor_mag(coords, w, omega):
    re = sum(wi * math.cos(omega * c) for wi, c in zip(w, coords))
    im = sum(wi * math.sin(omega * c) for wi, c in zip(w, coords))
    return math.hypot(re, im)


def dirichlet(omega, L):
    s = math.sin(omega / 2)
    if abs(s) < 1e-12:
        return 1.0
    return abs(math.sin(omega * L / 2) / (L * s))


# ---------- cost model ----------
def rel_cost(R, nblocks, core, a, tau=0.0, Nt=1024):
    s0, s1 = core
    r = R / Nt
    ncore = s1 - s0 + 1
    img = ((nblocks - ncore) * 1.0 + ncore * ((1 - a) * r + a * r * r)) / nblocks
    return tau + (1 - tau) * img
