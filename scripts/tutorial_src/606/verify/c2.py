"""C2 驗算：HyC-DiT 條件路徑的玩具（Eq. 4、Eq. 5 + adaLN 的 γ）。數值只示意結構，權重為固定 seed 的隨機值。"""
import math
from common import mulberry32, gauss

G, DM = 8, 8
OMEGA = [math.pi / 2 * 2 ** j for j in range(4)]

def stdz(v):
    m = sum(v) / len(v); s = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
    return [(x - m) / s for x in v]

MASK = [1.0 if (c - 4.5) ** 2 + (r - 3.0) ** 2 <= 2.2 ** 2 else 0.0 for r in range(G) for c in range(G)]
CH1 = stdz(MASK)
CH2 = stdz([float(c) for r in range(G) for c in range(G)])

r = mulberry32(11)
P = [[0.8 * gauss(r) for _ in range(2)] for _ in range(DM)]
EY = [[0.5 * gauss(r) for _ in range(DM)] for _ in range(3)]
ES = [[0.5 * gauss(r) for _ in range(DM)] for _ in range(2)]
ESIG = [0.5 * gauss(r) for _ in range(DM)]
WG = [gauss(r) / math.sqrt(DM) for _ in range(DM)]

def T(t):
    out = []
    for w in OMEGA: out += [math.cos(w * t), math.sin(w * t)]
    return out

silu = lambda x: x / (1 + math.exp(-x))

def eps(seed):
    rr = mulberry32(seed); return [(gauss(rr), gauss(rr)) for _ in range(G * G)]

def gammas(stage, t, y=0, sigma=0.0, seed=12):
    if stage == 1:
        m = [a + b + c for a, b, c in zip(T(t), EY[y], ES[0])]
        g = sum(w * silu(x) for w, x in zip(WG, m)); return [g] * (G * G)
    E = eps(seed); out = []
    a, b = math.sqrt(1 - sigma), math.sqrt(sigma)
    tt = T(t + 1)
    for i in range(G * G):
        z1 = a * CH1[i] + b * E[i][0]; z2 = a * CH2[i] + b * E[i][1]
        m = [P[k][0] * z1 + P[k][1] * z2 + tt[k] + sigma * ESIG[k] + ES[1][k] for k in range(DM)]
        out.append(sum(w * silu(x) for w, x in zip(WG, m)))
    return out

def stats(g):
    n = len(g); m = sum(g) / n; sd = math.sqrt(sum((x - m) ** 2 for x in g) / n)
    if sd < 1e-12: return m, 0.0, None
    mm = sum(MASK) / n; smk = math.sqrt(sum((x - mm) ** 2 for x in MASK) / n)
    cov = sum((x - m) * (k - mm) for x, k in zip(g, MASK)) / n
    return m, sd, cov / (sd * smk)

if __name__ == '__main__':
    print('mask tokens', int(sum(MASK)))
    print('| stage | y | t | σ | mean γ | std γ | corr(γ,mask) |')
    for y in range(3):
        for t in (0.0, 0.5, 1.0):
            m, sd, c = stats(gammas(1, t, y))
            print(f'| 1 | {y} | {t} | - | {m:.4f} | {sd:.4f} | - |')
    for t in (0.0, 0.5, 1.0):
        for s in (0.0, 0.2, 0.5, 0.8, 1.0):
            m, sd, c = stats(gammas(2, t, sigma=s))
            print(f'| 2 | - | {t} | {s} | {m:.4f} | {sd:.4f} | {c:.4f} |')
    # T(t) 與 T(t+1) 第一對座標的範圍
    print('T0 stage1 range cos', math.cos(0), math.cos(OMEGA[0]), ' stage2', math.cos(OMEGA[0]), math.cos(2 * OMEGA[0]))
    g = gammas(2, 0.5, sigma=0.2)
    print('γ token 0, 27, 63:', round(g[0], 4), round(g[27], 4), round(g[63], 4))

def max_abs_m():
    """全輸入範圍的 |m| 上界檢查（t、σ 步長 0.01，y 0..2，seed 12..61）。"""
    best = 0.0
    for ti in range(101):
        t = ti / 100
        for y in range(3):
            best = max(best, max(abs(a + b + c) for a, b, c in zip(T(t), EY[y], ES[0])))
        tt = T(t + 1)
        for si in range(0, 101, 5):
            s = si / 100; a, b = math.sqrt(1 - s), math.sqrt(s)
            for seed in range(12, 62, 7):
                E = eps(seed)
                for i in range(G * G):
                    z1 = a * CH1[i] + b * E[i][0]; z2 = a * CH2[i] + b * E[i][1]
                    for k in range(DM):
                        best = max(best, abs(P[k][0] * z1 + P[k][1] * z2 + tt[k] + s * ESIG[k] + ES[1][k]))
    return best

if __name__ == '__main__':
    print('mask (r,c) 0-based:', [(i // G, i % G) for i in range(G * G) if MASK[i]])
    print('max |m| over input range:', round(max_abs_m(), 3))
