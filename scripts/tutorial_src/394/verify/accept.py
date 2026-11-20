"""Acceptance values for design.md section 5. python3 accept.py [w1|w2|kap|w3|w4]"""
import sys, math
from core import *
f = lambda v: '[' + ', '.join('%.4f' % x for x in v) + ']'
which = sys.argv[1:] or ['w1', 'w2', 'kap', 'w4', 'w3']

if 'w1' in which:
    for kind in ('unif', 'mask'):
        K = 4; pi = prior(kind, K); x = onehot(0, K); at, as_ = 0.3, 0.6
        print('== W1', kind, 'K=4 at=.3 as=.6 x=e0')
        print(' q_t', f(q_marg(x, at, pi)), ' q_s', f(q_marg(x, as_, pi)))
        for z in range(K):
            if q_marg(x, at, pi)[z] == 0: continue
            print('  z_t=%d post' % z, f(post(z, x, at, as_, pi)), ' psi(k=.5)', f(psi_true(z, x, at, as_, 0.5, pi)), ' psi(k=0)', f(psi_true(z, x, at, as_, 0.0, pi)))
        for kap in (1, .5, 0):
            print('  marginal kappa=%g' % kap, f(marginal_after(x, at, as_, kap, pi)))
        # P(z_s = z_t) "stay probability" averaged over q_t, i.e. how much the chain keeps
        for kap in (1, .5, 0):
            st = sum(q_marg(x, at, pi)[z] * psi_true(z, x, at, as_, kap, pi)[z] for z in range(K))
            print('  P(z_s=z_t) kappa=%g: %.4f' % (kap, st))

if 'w2' in which:
    at, as_ = 0.3, 0.6
    pres = {'right': [0.85, 0.05, 0.05, 0.05], 'weak': [0.5, 0.3, 0.15, 0.05], 'wrong': [0.0, 0.6, 0.3, 0.1]}
    print('== W2 unif K=4 at=.3 as=.6 z_t=1, correct token 0')
    for name, xth in pres.items():
        for kap in (1.0, 0.9, 0.5):
            pt = psi_theta(1, xth, at, as_, kap, prior('unif', 4))
            print('  %-5s kappa=%.1f  q0t=%s  Psi=%s  P(correct)=%.4f' % (name, kap, f(q0t(1, xth, at, prior('unif', 4))), f(pt), pt[0]))
    print('  floor (1-k)(1-as)/K at k=.9:', (1 - .9) * (1 - as_) / 4)
    presM = {'right': [0.9, 0.05, 0.05, 0.0], 'wrong': [0.0, 0.7, 0.3, 0.0]}
    piM = prior('mask', 4)
    print('== W2 mask K=4 (A,B,C,[M]) at=.3 as=.6')
    for name, xth in presM.items():
        for z in (3, 1):
            xx = xth if z == 3 else onehot(z, 4)   # carry-over
            for kap in (1.0, 0.9):
                pt = psi_theta(z, xx, at, as_, kap, piM)
                print('  %-5s z_t=%s kappa=%.1f Psi=%s P(mask)=%.4f sigma_equiv=%.4f sigma_max=%.4f' % (
                    name, 'M' if z == 3 else 'B', kap, f(pt), pt[3], (1 - kap) * (1 - as_), sigma_max(at, as_)))

if 'kap' in which:
    print('== kappa curves alpha_t = 1 - t, T=128 (t_i = i/T)')
    T = 128
    for kind, eta in (('rescale', 0.05), ('rescale', 0.01), ('cap', 0.01), ('cap', 0.05)):
        ks = []
        for i in range(T, 0, -1):
            t = i / T; s = (i - 1) / T
            ks.append((t, kappa_sched(kind, t, 1 - t, 1 - s, dict(eta=eta))))
        kmin = min(ks, key=lambda a: a[1]); neg = [t for t, k in ks if k < 0]
        pick = {t: k for t, k in ks}
        print('  %s eta=%.2f  kappa(t=1)=%.4f kappa(.5)=%.4f kappa(.25)=%.4f kappa(1/T)=%.4f  min=%.4f@t=%.4f  #neg=%d (t<=%.4f)' % (
            kind, eta, pick[1.0], pick[0.5], pick[0.25], pick[1 / T], kmin[1], kmin[0], len(neg), max(neg) if neg else 0))
    for eta in (0.01, 0.06):
        print('  loop alpha_on=.9 eta=%.2f kappa=%.4f' % (eta, 1 - eta / (1 - 0.9)))

if 'w4' in which:
    from topk import efficient, naive, ztilde_from
    print('== W4 deterministic: same realised w (naive draw, seed 7), o=0')
    for K in (1000, 50257):
        for at in (0.34, 0.9):
            r = Mulberry32(7); sig = math.sqrt(1 - at * at)
            w = [sig * gauss(r) for _ in range(K)]; w[0] += at
            for k in (1, 3):
                for tau in (1e-3, 0.05, 0.3, 1.0):
                    logZ = math.log(sum(math.exp((v - max(w)) / tau) for v in w)) + max(w) / tau
                    lzt, ltop = ztilde_from(w, k, at, tau)
                    print('  K=%d at=%.2f k=%d tau=%g  logZ=%.4f logZ~=%.4f logZtop=%.4f  Z~/Z=%.4f Ztop/Z=%.4f' % (
                        K, at, k, tau, logZ, lzt, ltop, math.exp(lzt - logZ), math.exp(ltop - logZ)))
    print('== W4 stochastic: R=400 efficient draws, seed 11, K=50257 k=3')
    for at in (0.34, 0.9):
        for tau in (1e-3, 0.05):
            r = Mulberry32(11); R = 400; c2 = 0; top1 = 0; big = 0; s1 = 0
            for _ in range(R):
                e = efficient(r, 50257, 3, at, tau)
                c2 += e['case'] == 2; s1 += e['vals'][0] / R; top1 += e['lam'][0] / R; big += e['lam'][0] > 0.99
            print('  at=%.2f tau=%g  P(case2)=%.4f  mean K1=%.4f mean lam1=%.4f  frac(lam1>.99)=%.4f' % (at, tau, c2 / R, s1, top1, big / R))
    print('== W4 single draw seed 3 K=50257 k=3 at=.34 tau=1e-3 and tau=.05')
    for tau in (1e-3, 0.05):
        e = efficient(Mulberry32(3), 50257, 3, 0.34, tau)
        print('  ', {k: (f(v) if isinstance(v, list) else v) for k, v in e.items()})

if 'w3' in which:
    from chain import err_rate, run_chain
    p = dict(kap=0.9, ton=0.6, toff=0.1, eta=0.05)
    Ts = [8, 16, 32, 64, 128, 256, 512]
    print('== W3 defaults L=32 Ktok=8 emax=.5 emin=.02 R=32 seeds 1000..1031; window kap=.9 [.1,.6]; rescale eta=.05')
    for kind in ('mask', 'unif'):
        for sched in ('ancestral', 'window', 'rescale'):
            print('  %s %-9s' % (kind, sched), ' '.join('%.4f' % err_rate(kind, sched, T, p, R=32, emax=0.5, emin=0.02) for T in Ts))

if 'w4c' in which:
    from topk import efficient
    for at in (0.97, 0.99):
        r = Mulberry32(11); R = 400; c2 = 0; l1 = 0
        for _ in range(R):
            e = efficient(r, 50257, 3, at, 1e-3); c2 += e['case'] == 2; l1 += e['lam'][0] / R
        print('  w4c at=%.2f tau=1e-3 P(case2)=%.4f mean lam1=%.4f' % (at, c2 / R, l1))

if 'w1kz' in which:
    # LO5 (iv): kappa depending on z_t breaks the marginal. kappa(z_t = x) = kap, kappa(other) = 1.
    for kind in ('unif', 'mask'):
        K = 4; pi = prior(kind, K); x = onehot(0, K); at, as_ = 0.3, 0.6
        qt = q_marg(x, at, pi); out = [0.0] * K
        for z in range(K):
            if qt[z] == 0: continue
            kz = 0.5 if z == 0 else 1.0
            row = psi_true(z, x, at, as_, kz, pi)
            out = [o + qt[z] * rv for o, rv in zip(out, row)]
        qs = q_marg(x, as_, pi)
        print('  w1kz %s kappa(z=A)=.5 others 1: marginal %s  q_s %s  maxdiff %.4f' % (kind, f(out), f(qs), max(abs(a - b) for a, b in zip(out, qs))))

if 'w2b' in which:
    for at, as_ in ((0.3, 0.6), (0.7, 0.75)):
        sm = sigma_max(at, as_)
        print('  w2b at=%.2f as=%.2f sigma_max=%.4f 1-as=%.4f' % (at, as_, sm, 1 - as_),
              ' '.join('k(s=%.2f)=%.4f' % (s, kappa_from_sigma(s, as_)) for s in (0.04, 1 - as_, sm)))
