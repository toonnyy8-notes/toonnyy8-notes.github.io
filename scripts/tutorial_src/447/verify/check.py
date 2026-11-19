"""Acceptance numbers for the No. 447 widgets (same algorithm as the page). Output: check_out.txt"""
import math, numpy as np, core
A = (-1.2, -0.8)
out = []
p = lambda *a: out.append(' '.join(str(x) for x in a))
f = lambda x, n=6: f'{x:.{n}f}'

p('== prior sanity ==')
for H in (16, 32, 64, 128):
    P = core.basis(H); l = core.lam(H)
    p(f'H={H} lam0={f(l[0],4)} lam1={f(l[1],4)} lam2={f(l[2],4)} eigerr={np.abs(P@np.diag(1/l)@P.T-core.Qmat(H)).max():.1e} step_sd={f(core.step_sd(H),4)}')

p('== schedule (cosine) ==')
for N in (5, 10, 20, 50, 100):
    beta, abar, bt = core.cosine_schedule(N)
    p(f'N={N} abar1={f(abar[1])} abar_mid={f(abar[N//2])} abarN={abar[N]:.3e} beta1={f(beta[1])} betaN={f(beta[N])} bt2={f(bt[2])} btN={f(bt[N])} max_bt={f(bt[1:].max())}')

p('== C1 variance ratio sum v0 / sum lam ==')
for H in (16, 32, 64, 128):
    p(f'H={H} ' + ' '.join(f'N{N}={f(core.var_ratio(H,N),4)}' for N in (5, 10, 20, 50, 100)))

p('== C1 seeded chain (seed=1, H=64, N=20) ==')
X, hist = core.run_chain(64, 20, seed=1)
r = core.mulberry32(1); p('first 3 uniforms seed1', [f(r(), 9) for _ in range(3)])
p('tauN[0,0:3]', [f(v) for v in hist[0][0, :3]], 'tauN[1,0]', f(hist[0][1, 0]))
for i in (20, 10, 5, 1, 0):
    Xi = hist[20 - i]
    p(f'i={i} rms={f(math.sqrt(np.mean(Xi**2)),4)} tau_i[:,0]={[f(v,4) for v in Xi[:,0]]} tau_i[:,63]={[f(v,4) for v in Xi[:,63]]}')

p('== C2 inpainting: procedure mean vs exact conditional ==')
for H in (32, 64, 128):
    for N in (5, 20, 100):
        for B in ((1.2, 0.8), (1.5, -1.0)):
            cond = [(0, A), (H - 1, B)]
            m, _ = core.run_chain(H, N, cond=cond, noise=False)
            ex = core.exact_mean(H, cond)
            jm = np.linalg.norm(m[:, 1] - m[:, 0]); je = np.linalg.norm(ex[:, 1] - ex[:, 0])
            p(f'H={H} N={N} B={B} rms={f(core.rms(m,ex),4)} jump_proc={f(jm,4)} jump_exact={f(je,4)} mid_proc={[f(v,4) for v in m[:,H//2]]} mid_exact={[f(v,4) for v in ex[:,H//2]]} q1_proc={[f(v,4) for v in m[:,H//4]]} q1_exact={[f(v,4) for v in ex[:,H//4]]}')
H, N = 64, 20; cond = [(0, A), (H - 1, (1.2, 0.8))]
for j in range(2):
    s, _ = core.run_chain(H, N, seed=1000 * 1 + j, cond=cond)
    p(f'C2 sample seed={1000+j} t=1 {[f(v,4) for v in s[:,1]]} t=32 {[f(v,4) for v in s[:,32]]} jump={f(np.linalg.norm(s[:,1]-s[:,0]),4)}')
p('profile H=64 N=20 B=(1.2,0.8) x-coord: t proc exact')
m, _ = core.run_chain(H, N, cond=cond, noise=False); ex = core.exact_mean(H, cond)
for t in (0, 1, 2, 4, 8, 16, 32, 48, 56, 60, 62, 63):
    p(t, f(m[0, t], 4), f(ex[0, t], 4))

p('== C3 guided sampling (Alg 1, true J at mu, start inpainted) ==')
c = np.array([1.0, 1.0])
def guided(H, N, al, cc, seed=None, mode='clean'):
    return core.run_chain(H, N, seed=seed, cond=[(0, A)], guide=core.make_guide(mode, al, cc, H, N), noise=seed is not None)
J = lambda X, cc: float(-0.5 * np.sum((X - cc[:, None]) ** 2))
for H in (32, 64):
    for N in (20, 100):
        beta, abar, bt = core.cosine_schedule(N)
        for al in (0.001, 0.01, 0.1, 1.0, 10.0, 31.6, 100.0):
            with np.errstate(all='ignore'):
                m, hist = guided(H, N, al, c)
            ex = core.exact_mean(H, [(0, A)], extra_prec=al, lin=(al * c[0], al * c[1]))
            div = next((N - k + 1 for k, Xh in enumerate(hist) if not np.all(np.abs(Xh) <= 1e6)), None)
            if div is not None:
                p(f'H={H} N={N} alpha={al} DIVERGED at i={div} (first level whose output exceeds 1e6) max_alpha_bt={f(al*bt[1:].max(),4)}')
                continue
            p(f'H={H} N={N} alpha={al} end_proc={[f(v,4) for v in m[:,-1]]} end_exact={[f(v,4) for v in ex[:,-1]]} rms={f(core.rms(m,ex),4)} J_proc={f(J(m,c),3)} J_exact={f(J(ex,c),3)} max_alpha_bt={f(al*bt[1:].max(),4)}')
H, N = 32, 20
for j in range(2):
    s, _ = guided(H, N, 0.1, c, seed=1000 + j)
    p(f'C3 sample seed={1000+j} alpha=0.1 end={[f(v,4) for v in s[:,-1]]} J={f(J(s,c),3)}')
cc = np.array([-1.0, 1.5])
m, _ = guided(32, 20, 0.1, cc); ex = core.exact_mean(32, [(0, A)], 0.1, (0.1 * cc[0], 0.1 * cc[1]))
p(f'C3 c=(-1,1.5) alpha=0.1 H32 N20 end_proc={[f(v,4) for v in m[:,-1]]} end_exact={[f(v,4) for v in ex[:,-1]]} rms={f(core.rms(m,ex),4)}')
p('== C3 noisy-level likelihood guidance (default mode on the page) ==')
for H, N in ((32, 20), (64, 20), (32, 100)):
    for al in (0.001, 0.01, 0.1, 1.0, 10.0, 10 ** 1.2, 100.0):
        m, hist = guided(H, N, al, c, mode='noisy')
        ex = core.exact_mean(H, [(0, A)], extra_prec=al, lin=(al * c[0], al * c[1]))
        p(f'H={H} N={N} alpha={al:.4g} end_proc={[f(v,4) for v in m[:,-1]]} end_exact={[f(v,4) for v in ex[:,-1]]} rms={f(core.rms(m,ex),4)} J_proc={f(J(m,c),3)} J_exact={f(J(ex,c),3)} peak={f(np.abs(np.array(hist)).max(),4)}')
for j in range(2):
    s, _ = guided(32, 20, 0.1, c, seed=1000 + j, mode='noisy')
    p(f'C3 noisy sample seed={1000+j} alpha=0.1 end={[f(v,4) for v in s[:,-1]]} J={f(J(s,c),3)}')
open('check_out.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))

# fidelity item 1: replacement only after the last step (i=0) vs after every step (C2 default, procedure mean)
out2 = []
H, N = 64, 20; cond = [(0, A), (H - 1, (1.2, 0.8))]
free, _ = core.run_chain(H, N, noise=False)              # unconstrained procedure mean is 0
X = free.copy(); X[:, 0] = A; X[:, H - 1] = (1.2, 0.8)
out2.append(f'final-only replacement H=64 N=20: jump={np.linalg.norm(X[:,1]-X[:,0]):.4f} (every-step 0.1676) t=1 x={X[0,1]:.4f}')
beta, abar, bt = core.cosine_schedule(20)
out2.append('alpha=10 N=20 steps with alpha*bt>2: ' + str(int(np.sum(10 * bt[1:] > 2))) + '; alpha=10^1.2: ' + str(int(np.sum(10 ** 1.2 * bt[1:] > 2))))
open('check_out.txt', 'a').write('\n'.join(out2) + '\n'); print('\n'.join(out2))
