"""Monte Carlo check: zero-noise run == E[tau^0] of the stochastic chain (affine chain), and C1 variance ratio."""
import math, numpy as np, core
rng = np.random.default_rng(0)
def batch_chain(H, N, M, cond=None, guide=None):
    P = core.basis(H); l = core.lam(H); beta, abar, bt = core.cosine_schedule(N)
    X = rng.standard_normal((M, 2, H))
    def ap(X):
        for t, c in (cond or []): X[:, :, t] = c
        return X
    X = ap(X)
    for i in range(N, 0, -1):
        Xh = X @ P; eps = (math.sqrt(1 - abar[i]) * Xh / (abar[i] * l + 1 - abar[i])) @ P.T
        mu = (X - beta[i] / math.sqrt(1 - abar[i]) * eps) / math.sqrt(1 - beta[i])
        if guide: mu = mu + bt[i] * guide(mu, i)
        if i > 1: mu = mu + math.sqrt(bt[i]) * rng.standard_normal(mu.shape)
        X = ap(mu)
    return X
M = 40000
A = (-1.2, -0.8); B = (1.2, 0.8)
S = batch_chain(64, 20, M, cond=[(0, A), (63, B)])
m, _ = core.run_chain(64, 20, cond=[(0, A), (63, B)], noise=False)
print('C2 MC mean vs zero-noise run: max abs diff', np.abs(S.mean(0) - m).max().round(4), ' max |diff|/s.e. over coords', (np.abs(S.mean(0) - m) / (S.std(0) / math.sqrt(M) + 1e-12))[:, 1:].max().round(2), '(expected max of ~2H |N(0,1)| draws: about 3)')
c = np.array([1.0, 1.0]); g = lambda mu, i: -0.1 * (mu - c[None, :, None])
S = batch_chain(32, 20, M, cond=[(0, A)], guide=g)
m, _ = core.run_chain(32, 20, cond=[(0, A)], guide=core.make_guide('clean', 0.1, c, 32, 20), noise=False)
print('C3 MC mean vs zero-noise run: max abs diff', np.abs(S.mean(0) - m).max().round(4), ' max |diff|/s.e. over coords', (np.abs(S.mean(0) - m) / (S.std(0) / math.sqrt(M) + 1e-12))[:, 1:].max().round(2), '(expected max of ~2H |N(0,1)| draws: about 3)')
S = batch_chain(64, 20, M); P = core.basis(64)
v = ((S @ P) ** 2).mean(0).mean(0)
print('C1 MC var ratio', (v.sum() / core.lam(64).sum()).round(4), 'closed form', round(core.var_ratio(64, 20), 4))
