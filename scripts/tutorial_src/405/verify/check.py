"""Verification for No.405 design: same algorithms the page will use (float64, plain loops)."""
import math

T = 1000
def cosine_schedule(T=T, s0=0.008):
    f = lambda u: math.cos((u + s0) / (1 + s0) * math.pi / 2) ** 2
    beta = [min(1 - f((i + 1) / T) / f(i / T), 0.999) for i in range(T)]   # beta[t-1] = beta_t
    abar, p = [], 1.0
    for b in beta:
        p *= 1 - b; abar.append(p)
    return beta, abar
BETA, ABAR = cosine_schedule()
b = lambda t: BETA[t - 1]; ab = lambda t: ABAR[t - 1]

# ---------- C1: exposure-bias toy. 1-D data x0~N(0,s^2); Bayes x0-predictor scaled by (1+delta) ----------
# step t->t-1 (posterior-mean form, identical to Alg.2 line 4 when delta=0):
#   x_{t-1} = A_t x_t (+ sqrt(btil_t) z),  A_t = c1_t (1+delta) g_t + c2_t
#   g_t = sqrt(abar_t) s^2/(abar_t s^2 + 1 - abar_t);  c1 = sqrt(abar_{t-1}) beta_t/(1-abar_t)
#   c2 = sqrt(alpha_t)(1-abar_{t-1})/(1-abar_t);  btil_t = (1-abar_{t-1})/(1-abar_t) beta_t  (abar_0 = 1)
def c1_steps(s, delta):
    s2 = s * s; A = [None]; BT = [None]
    for t in range(1, T + 1):
        be, a_ = b(t), ab(t); ap = ab(t - 1) if t > 1 else 1.0
        c1 = math.sqrt(ap) * be / (1 - a_); c2 = math.sqrt(1 - be) * (1 - ap) / (1 - a_)
        g = math.sqrt(a_) * s2 / (a_ * s2 + 1 - a_)
        A.append(c1 * (1 + delta) * g + c2); BT.append((1 - ap) / (1 - a_) * be)
    return A, BT
def c1_curve(s, delta):
    """O(T): for every start t0, 1-D Frechet distance (stochastic, Tab.1 protocol) and
    mean |x0 - x0hat| (deterministic, Alg.4/Tab.6 protocol)."""
    s2 = s * s; A, BT = c1_steps(s, delta)
    G = [1.0]; N = [0.0]                      # G[t]=prod_{j<=t} A_j ; N[t]=sum_{2<=tau<=t} btil_tau G[tau-1]^2
    for t in range(1, T + 1):
        N.append(N[-1] + (BT[t] * G[t - 1] ** 2 if t > 1 else 0.0)); G.append(G[-1] * A[t])
    fd, l1 = [], []
    for t0 in range(1, T + 1):
        a_ = ab(t0); vin = a_ * s2 + 1 - a_; m = math.sqrt(a_) * s2
        v = G[t0] ** 2 * vin + N[t0]
        fd.append((math.sqrt(v) - s) ** 2)
        mse = max(0.0, G[t0] ** 2 * vin - 2 * G[t0] * m + s2)
        l1.append(math.sqrt(2 / math.pi) * math.sqrt(mse))
    return fd, l1
def c1_ratio_path(s, delta, t0):
    """std(xhat_tau)/std(x_tau) along one stochastic run started at t0 (tau = t0..0)."""
    s2 = s * s; A, BT = c1_steps(s, delta); a_ = ab(t0); v = a_ * s2 + 1 - a_; out = {t0: 1.0}
    for t in range(t0, 0, -1):
        v = A[t] ** 2 * v + (BT[t] if t > 1 else 0.0)
        at = ab(t - 1) if t > 1 else 1.0
        out[t - 1] = math.sqrt(v) / math.sqrt(at * s2 + 1 - at)
    return out
def c1_mc(s, delta, t0, n=20000, seed=1):
    import random
    r = random.Random(seed); A, BT = c1_steps(s, delta); xs = []
    for _ in range(n):
        a_ = ab(t0); x = math.sqrt(a_) * r.gauss(0, s) + math.sqrt(1 - a_) * r.gauss(0, 1)
        for t in range(t0, 0, -1):
            x = A[t] * x + (math.sqrt(BT[t]) * r.gauss(0, 1) if t > 1 else 0.0)
        xs.append(x)
    v = sum(x * x for x in xs) / n
    return (math.sqrt(v) - s) ** 2

# ---------- C2: geometry of DDPM / DDPM-IP / DDPM-y ----------
def c2(t, gamma, x0, eps, xi):
    a_ = ab(t); sg = math.sqrt(1 + gamma * gamma)
    xt = [math.sqrt(a_) * x0[i] + math.sqrt(1 - a_) * eps[i] for i in range(2)]
    yt = [xt[i] + math.sqrt(1 - a_) * gamma * xi[i] for i in range(2)]
    epsp = [(eps[i] + gamma * xi[i]) / sg for i in range(2)]
    yt_y = [math.sqrt(a_) * x0[i] + math.sqrt(1 - a_) * sg * epsp[i] for i in range(2)]
    d = math.dist(eps, epsp)
    return dict(xt=xt, yt=yt, yt_y=yt_y, epsp=epsp, dist=d,
                std_x=math.sqrt(1 - a_), std_y=math.sqrt((1 - a_) * (1 + gamma ** 2)),
                exp_dist2_perdim=2 - 2 / sg)

# ---------- C3: blur std and slope ratio ----------
def c3(t, gamma, s):
    a_ = ab(t); s2 = s * s
    return dict(abar=a_, pert=gamma * math.sqrt(1 - a_), blur=gamma * math.sqrt((1 - a_) / a_),
                c=math.sqrt(1 - a_) / (a_ * s2 + 1 - a_),
                cip=math.sqrt(1 - a_) / (a_ * s2 + (1 - a_) * (1 + gamma ** 2)),
                ratio=(a_ * s2 + 1 - a_) / (a_ * s2 + (1 - a_) * (1 + gamma ** 2)))

# ---------- C4: tables ----------
TAB3 = {  # dataset: {T': (ADM FID, ADM sFID, IP FID, IP sFID)}
 'CIFAR10': {1000:(2.99,4.76,2.76,4.05),300:(2.95,4.95,2.67,4.14),100:(3.37,5.66,2.70,4.51),80:(3.63,5.97,2.93,4.69)},
 'ImageNet32': {1000:(3.60,3.30,2.87,2.39),300:(3.58,3.48,2.74,2.58),100:(4.26,4.48,3.24,3.13),80:(4.61,4.76,3.57,3.33)},
 'LSUN64': {1000:(3.39,7.96,2.68,6.04),300:(3.31,8.39,2.60,5.98),100:(3.50,11.10,2.79,6.56),80:(4.17,12.60,2.95,6.93)},
 'CelebA64': {1000:(1.60,3.80,1.31,3.38),300:(1.82,4.25,1.43,3.36),100:(3.02,5.76,2.21,4.33),80:(3.75,6.80,2.67,4.69)},
 'FFHQ128': {1000:(9.65,12.53,2.98,5.59),300:(9.55,12.60,3.74,5.97),100:(14.52,16.02,5.94,7.90),80:(17.00,18.02,6.89,8.79)},
}
TAB4 = {'CIFAR10':(500,1000,2.99,460,80,2.93),'ImageNet32':(4500,1000,3.53,4000,80,3.50),
        'LSUN64':(300,1000,3.39,220,60,3.31),'CelebA64':(480,1000,1.60,300,200,1.53),'FFHQ128':(420,1000,9.65,180,60,8.72)}
TAB5 = {(0,'DDIM'):[14.21,7.50,5.17,4.66,4.29],(0,'DDIM-IP'):[10.54,5.70,4.66,4.52,4.27],
        (0.5,'DDIM'):[17.24,8.87,5.59,4.88,4.45],(0.5,'DDIM-IP'):[10.06,5.53,3.95,3.66,3.56]}
TAB1 = {'ADM':[0.983,1.808,2.587,3.105,3.544],'ADM-IP':[0.972,1.594,2.198,2.539,2.742]}

if __name__ == '__main__':
    print('schedule: beta_1=%.3e abar_1=%.6f abar_500=%.6f abar_1000=%.3e beta_1000=%.3f' % (b(1), ab(1), ab(500), ab(1000), b(1000)))
    for s_, d_ in [(0.5, 0.0), (0.5, 0.02), (0.5, -0.02), (0.5, 0.05), (1.0, 0.02), (0.2, 0.02)]:
        fd, l1 = c1_curve(s_, d_)
        print('C1 s=%.2f delta=%+.2f FD' % (s_, d_), ' '.join('%d:%.3e' % (t, fd[t-1]) for t in (10, 100, 300, 500, 700, 1000)),
              '| L1', ' '.join('%d:%.4f' % (t, l1[t-1]) for t in (100, 300, 600, 1000)))
    for s_, d_ in [(0.5, 0.02), (0.5, 0.05)]:
        rp = c1_ratio_path(s_, d_, 1000)
        print('C1 ratio path s=%.2f delta=%+.2f' % (s_, d_), ' '.join('%d:%.4f' % (t, rp[t]) for t in (1000, 700, 300, 100, 50, 10, 0)))
    for s_, d_, t0 in [(0.5, 0.02, 100), (0.5, 0.05, 300)]:
        fd, _ = c1_curve(s_, d_)
        print('C1 MC check s=%.2f delta=%+.2f t0=%d closed=%.4e mc=%.4e' % (s_, d_, t0, fd[t0-1], c1_mc(s_, d_, t0)))
    # C2 with fixed demo vectors
    for g in (0.1, 0.5, 1.0):
        r = c2(500, g, [0.6, -0.3], [0.8, 1.1], [-1.2, 0.5])
        print('C2 gamma=%.1f' % g, {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in r.items()})
    for g in (0.1, 0.5, 1.0):
        for t in (100, 500, 900, 1000):
            r = c3(t, g, 0.5)
            print('C3 g=%.1f t=%d' % (g, t), {k: round(v, 5) for k, v in r.items()})
    for ds, rows in TAB3.items():
        bestA = min(rows, key=lambda k: rows[k][0]); bestI = min(rows, key=lambda k: rows[k][2])
        ref = rows[1000][0]
        minT = min([k for k in rows if rows[k][2] <= ref], default=None)
        print('C4', ds, 'bestADM', bestA, rows[bestA][0], 'bestIP', bestI, rows[bestI][2],
              'minT IP<=ADM@1000', minT, 'gain@1000 FID %.2f sFID %.2f' % (rows[1000][0]-rows[1000][2], rows[1000][1]-rows[1000][3]),
              'gain@80 FID %.2f sFID %.2f' % (rows[80][0]-rows[80][2], rows[80][1]-rows[80][3]))
    for ds, (ia, sa, fa, ii, si, fi) in TAB4.items():
        print('Tab4', ds, 'sample speedup %.1f' % (sa / si), 'train ratio %.3f saving %.1f%%' % (ia / ii, 100 * (1 - ii / ia)))
    for eta_ in (0, 0.5):
        print('Tab5 eta', eta_, 'gains', [round(a - c, 2) for a, c in zip(TAB5[(eta_, 'DDIM')], TAB5[(eta_, 'DDIM-IP')])])
    print('Tab1 gap', [round(a - c, 3) for a, c in zip(TAB1['ADM'], TAB1['ADM-IP'])],
          'rel', [round(1 - c / a, 3) for a, c in zip(TAB1['ADM'], TAB1['ADM-IP'])])
