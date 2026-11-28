"""C1 驗算：Prop. 1 的最佳乾淨目標預測器與 Jacobian（與頁面相同算法）。
殘差模型：Dirac（K 個等距點，零平均、單位變異數後乘 s=sqrt(v)）或高斯 N(0, v)。
r_t = (1-t) r + t*xi。全部用純 Python 寫，方便逐行對照 JS。
"""
import math

def dirac_points(K, v):
    q = [k - (K - 1) / 2 for k in range(K)]
    sd = math.sqrt(sum(x * x for x in q) / K)
    s = math.sqrt(v)
    return [s * x / sd for x in q]

def post(rt, t, R):
    """回傳 (E[r|r_t], Cov(r|r_t))；log-sum-exp。"""
    lw = [-(rt - (1 - t) * r) ** 2 / (2 * t * t) for r in R]
    m = max(lw)
    w = [math.exp(x - m) for x in lw]
    Z = sum(w)
    e1 = sum(wi * r for wi, r in zip(w, R)) / Z
    e2 = sum(wi * r * r for wi, r in zip(w, R)) / Z
    return e1, max(0.0, e2 - e1 * e1)

XI = [-8 + 16 * i / 800 for i in range(801)]
PHI = [math.exp(-x * x / 2) for x in XI]
ZPHI = sum(PHI)

def exp_cov(t, R):
    """E_{r_t}[Cov(r|r_t)] = V(a|a_t,u)：每個成分 r_i，對 xi 在 [-8,8] 用 801 點、以 phi 正規化的加權和。"""
    tot = 0.0
    for ri in R:
        acc = 0.0
        for x, p in zip(XI, PHI):
            acc += p * post((1 - t) * ri + t * x, t, R)[1]
        tot += acc / ZPHI
    return tot / len(R)

def max_cov(t, R):
    lo = (1 - t) * min(R) - 4 * t
    hi = (1 - t) * max(R) + 4 * t
    pts = [lo + (hi - lo) * i / 800 for i in range(801)]
    pts += [(1 - t) * (R[k] + R[k + 1]) / 2 for k in range(len(R) - 1)]
    return max(post(x, t, R)[1] for x in pts)

def gauss_J(t, v):
    return (1 - t) * v / ((1 - t) ** 2 * v + t * t)

def fine_max_cov(t, R):
    # 獨立檢查：在每個中點附近做 2 萬點細搜尋
    best = 0.0
    for k in range(len(R) - 1):
        c = (1 - t) * (R[k] + R[k + 1]) / 2
        w = 0.5 * (1 - t) * (R[k + 1] - R[k])
        for i in range(20001):
            best = max(best, post(c - w + 2 * w * i / 20000, t, R)[1])
    return best

if __name__ == '__main__':
    # Eq. 9 恆等式：E[r|r_t] = (r_t + t^2 s_t)/(1-t)，s_t 用解析 score
    R = dirac_points(8, 0.958)
    for t in (0.1, 0.5, 0.9):
        for rt in (-1.3, 0.2, 0.77):
            lw = [-(rt - (1 - t) * r) ** 2 / (2 * t * t) for r in R]
            m = max(lw); w = [math.exp(x - m) for x in lw]; Z = sum(w)
            score = sum(wi * ((1 - t) * r - rt) / (t * t) for wi, r in zip(w, R)) / Z
            assert abs((rt + t * t * score) / (1 - t) - post(rt, t, R)[0]) < 1e-12
    print('Eq.9 identity ok')
    print('Dirac points K=8 v=0.958:', [round(x, 4) for x in R])
    print('\n| 模型 | v | K | t | J_max | E tr J | V(a|a_t)/V(a) | fine J_max |')
    rows = [('Dirac', 0.958, 8), ('Dirac', 0.426, 8), ('Dirac', 0.958, 16), ('Dirac', 0.958, 4)]
    for name, v, K in rows:
        R = dirac_points(K, v)
        for t in (0.02, 0.05, 0.1, 0.2, 0.5, 0.9):
            ec = exp_cov(t, R); mc = max_cov(t, R)
            f = (1 - t) / t ** 2
            fm = fine_max_cov(t, R) if t <= 0.1 else mc
            print(f'| {name} | {v} | {K} | {t} | {f*mc:.4g} | {f*ec:.4g} | {ec/v:.4f} | {f*fm:.4g} |')
    for v in (0.958, 0.426, 0.05):
        for t in (0.02, 0.05, 0.1, 0.2, 0.5, 0.9):
            J = gauss_J(t, v)
            print(f'| Gauss | {v} | - | {t} | {J:.4g} | {J:.4g} | {J*t*t/(1-t)/v:.4f} | - |')
    # 高斯情形 sup_t J
    for v in (0.958, 0.426, 0.05):
        best = max((gauss_J(i / 10000, v), i / 10000) for i in range(100, 9901))
        print(f'Gauss v={v}: max_t∈[0.01,0.99] J = {best[0]:.4f} at t={best[1]}')
    # 左圖：E[r|r_t] 在 t=0.05 的形狀（Dirac v=0.958 K=8）抽幾點
    R = dirac_points(8, 0.958)
    for rt in (-0.5, -0.2, -0.05, 0.0, 0.05, 0.2):
        print('t=0.05 rt', rt, 'E[r|rt]=%.4f' % post(rt, 0.05, R)[0])
