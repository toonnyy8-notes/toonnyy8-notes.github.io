"""C3 驗算：一步輸出崩塌到條件平均，再以 IMQ-MMD（V 統計量）的解析梯度移動粒子。
與頁面同一算法：參考集 R（三團高斯，seed 21）、生成粒子 X（seed 22）、
k_h(x,y) = sum_h (1 + |x-y|^2/h^2)^(-1/2)，更新 x_i <- x_i - eta * Ng * dMMD2/dx_i，每粒子步長上限 0.2。
"""
import math, sys
from common import mulberry32, gauss

CENTERS = [(-1.2, -0.6), (1.2, -0.6), (0.0, 1.1)]
NR, NG, SD = 240, 120, 0.25
SCALES = {'0.1': [0.1], '0.3': [0.3], '1.0': [1.0], 'multi': [0.1, 0.3, 1.0]}

def make_ref():
    r = mulberry32(21); R = []
    for p in range(NR):
        cx, cy = CENTERS[p % 3]
        R.append((cx + SD * gauss(r), cy + SD * gauss(r)))
    return R

def make_gen(R, spread=0.05):
    mx = sum(p[0] for p in R) / NR; my = sum(p[1] for p in R) / NR
    r = mulberry32(22)
    return [[mx + spread * gauss(r), my + spread * gauss(r)] for _ in range(NG)]

def kern(dx, dy, hs):
    """回傳 (k, dk/d(dx), dk/d(dy))，dx = x - y。"""
    d2 = dx * dx + dy * dy; k = gx = gy = 0.0
    for h in hs:
        q = 1 + d2 / (h * h); k += q ** -0.5
        c = -(q ** -1.5) / (h * h); gx += c * dx; gy += c * dy
    return k, gx, gy

def mmd2_and_grad(X, R, hs):
    ng, nr = len(X), len(R)
    sxx = sxr = 0.0; G = [[0.0, 0.0] for _ in X]
    for i, (xi, yi) in enumerate(X):
        for j, (xj, yj) in enumerate(X):
            k, gx, gy = kern(xi - xj, yi - yj, hs); sxx += k
            if i != j: G[i][0] += 2 * gx / ng ** 2; G[i][1] += 2 * gy / ng ** 2
        for (xr, yr) in R:
            k, gx, gy = kern(xi - xr, yi - yr, hs); sxr += k
            G[i][0] -= 2 * gx / (ng * nr); G[i][1] -= 2 * gy / (ng * nr)
    return sxx / ng ** 2 - 2 * sxr / (ng * nr), G   # 省略與 X 無關的 R-R 項

def rr_term(R, hs):
    return sum(kern(a[0] - b[0], a[1] - b[1], hs)[0] for a in R for b in R) / len(R) ** 2

def coverage(X):
    cnt = [0, 0, 0]; near = 0
    for x, y in X:
        d = [math.hypot(x - cx, y - cy) for cx, cy in CENTERS]
        k = d.index(min(d)); cnt[k] += 1
        if d[k] < 2 * SD: near += 1
    return cnt, near

def run(scale, eta, steps, report=(0, 50, 100, 200, 400)):
    R = make_ref(); X = make_gen(R); hs = SCALES[scale]; rr = rr_term(R, hs)
    out = []
    for s in range(steps + 1):
        m, G = mmd2_and_grad(X, R, hs)
        if s in report:
            cnt, near = coverage(X)
            out.append((s, m + rr, cnt, near))
        if s == steps: break
        for i in range(NG):
            sx, sy = eta * NG * G[i][0], eta * NG * G[i][1]
            L = math.hypot(sx, sy)
            if L > 0.2: sx, sy = sx * 0.2 / L, sy * 0.2 / L
            X[i][0] -= sx; X[i][1] -= sy
    return out, X

if __name__ == '__main__':
    R = make_ref(); X = make_gen(R)
    # 梯度有限差分檢查
    hs = SCALES['multi']; m0, G = mmd2_and_grad(X, R, hs); e = 1e-6
    X[3][0] += e; m1, _ = mmd2_and_grad(X, R, hs); X[3][0] -= e
    print('fd check', (m1 - m0) / e, G[3][0])
    print('ref mean', sum(p[0] for p in R) / NR, sum(p[1] for p in R) / NR)
    print('gen[0..2]', X[:3])
    for scale in sys.argv[1:] or ['multi', '0.1', '0.3', '1.0']:
        for eta in (0.5, 1.0, 2.0):
            out, _ = run(scale, eta, 400)
            print(scale, eta, [(s, round(m, 5), c, n) for s, m, c, n in out])
