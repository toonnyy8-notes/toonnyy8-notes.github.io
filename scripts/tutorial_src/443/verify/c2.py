"""C2: behaviour cloning on the Sec. 4 bandit data. Ground truth vs diagonal-Gaussian MLE vs ideal-denoiser diffusion (Eq. 1)."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from core import *

S = 1000
def draws_diffusion(seed, N):
    r = mulberry32(seed)
    aN = np.zeros((S, 2)); Z = {i: np.zeros((S, 2)) for i in range(2, N + 1)}
    for s in range(S):
        aN[s] = [gauss(r), gauss(r)]
        for i in range(N, 1, -1):
            Z[i][s] = [gauss(r), gauss(r)]
    return aN, Z
def draws_truth(seed):
    r = mulberry32(seed); out = np.zeros((S, 2))
    for s in range(S):
        k = min(3, int(4 * r())); out[s] = MU[k] + SD * np.array([gauss(r), gauss(r)])
    return out
def draws_mle(seed):
    r = mulberry32(seed); sd = math.sqrt(VAR_MLE)
    return np.array([[sd * gauss(r), sd * gauss(r)] for _ in range(S)])

VAR_MLE = 0.5 * 0.8 ** 2 + SD ** 2      # per-dim MLE variance (mean is 0 by symmetry) = 0.3225
def metrics(A):
    dmin = np.sqrt(((A[:, None, :] - MU[None]) ** 2).sum(-1)).min(1)
    near = (dmin < 0.2).mean(); centre = (np.sqrt((A ** 2).sum(1)) < 0.3).mean()
    out = (np.abs(A) > 1).any(1).mean()
    k = ((A[:, None, :] - MU[None]) ** 2).sum(-1).argmin(1)
    share = [((k == j) & (dmin < 0.2)).mean() for j in range(4)]
    return near, centre, out, share

if __name__ == '__main__':
    print('VAR_MLE=%.4f sd=%.4f' % (VAR_MLE, math.sqrt(VAR_MLE)))
    p0 = 1 / (2 * math.pi * VAR_MLE); pc = p0 * math.exp(-0.64 / (2 * VAR_MLE))
    print('MLE density origin=%.4f centre=%.4f ratio=%.3f' % (p0, pc, p0 / pc))
    # analytic MLE probabilities
    print('MLE P(|a|<0.3)=%.4f' % (1 - math.exp(-0.09 / (2 * VAR_MLE))))
    print('truth P(within 0.2 of own centre)=%.5f' % (1 - math.exp(-0.04 / (2 * SD ** 2))))
    for name, A in [('truth', draws_truth(7)), ('mle', draws_mle(7))]:
        n, c, o, sh = metrics(A); print(name, 'near=%.3f centre=%.3f outside=%.3f' % (n, c, o), ['%.3f' % x for x in sh])
    lg0 = np.zeros(4)
    for N in (1, 2, 3, 5, 10, 20, 50):
        aN, Z = draws_diffusion(7, N)
        A = sample_chain(aN, Z, schedule(N), lg0)
        n, c, o, sh = metrics(A)
        print('N=%2d near=%.3f centre=%.3f outside=%.3f' % (N, n, c, o), ['%.3f' % x for x in sh], 'first sample', np.round(A[0], 4))
    # seed sensitivity
    for seed in (8, 9):
        for N in (2, 5, 50):
            aN, Z = draws_diffusion(seed, N); A = sample_chain(aN, Z, schedule(N), lg0)
            n, c, o, sh = metrics(A); print('seed', seed, 'N', N, 'near=%.3f centre=%.3f' % (n, c))
