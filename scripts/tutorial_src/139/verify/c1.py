"""C1: Gaussian correction at noise level tau (closed form, toy Gaussian next-state return)."""
import math

def c1(gamma, tau, r, m1, s1):
    """Next-state clean return Z' ~ N(m1, s1^2). Returns (mean, std) of
    target  = law of tau*(r+gamma Z') + (1-tau) N        (correct noisy return)
    corr    = tau r + gamma Z'(tau) + (1-tau) sqrt(1-g^2) N, with Z'(tau)=tau Z' + (1-tau) N'
    clean   = r + gamma Z'(tau)                           (clean Bellman eq. applied at tau)
    and W2(clean, target)."""
    c = math.sqrt(max(0.0, 1 - gamma * gamma))
    var_next = tau * tau * s1 * s1 + (1 - tau) ** 2          # Var Z'(tau)
    tgt = (tau * (r + gamma * m1), math.sqrt(tau * tau * gamma * gamma * s1 * s1 + (1 - tau) ** 2))
    cor = (tau * r + gamma * tau * m1, math.sqrt(gamma * gamma * var_next + (1 - tau) ** 2 * c * c))
    cln = (r + gamma * tau * m1, math.sqrt(gamma * gamma * var_next))
    w2 = math.hypot(cln[0] - tgt[0], cln[1] - tgt[1])
    return dict(c=c, tgt=tgt, cor=cor, cln=cln, w2=w2)

if __name__ == '__main__':
    D = dict(r=-1.0, m1=-2.0, s1=0.5)
    for g in [0.0, 0.5, 0.9, 0.99, 0.999, 0.9999]:
        print(f'gamma={g}: sqrt(1-g^2)={math.sqrt(1-g*g):.4f}  1/(1-g)={1/(1-g) if g<1 else 0:.0f}')
    print('default r=-1, m1=-2, s1=0.5')
    for g in [0.9, 0.999]:
        for t in [0, 0.25, 0.5, 0.75, 1]:
            o = c1(g, t, **D)
            assert abs(o['tgt'][0] - o['cor'][0]) < 1e-12 and abs(o['tgt'][1] - o['cor'][1]) < 1e-12
            print(f"g={g} tau={t}: target/corr mean={o['tgt'][0]:.4f} sd={o['tgt'][1]:.4f} | clean mean={o['cln'][0]:.4f} sd={o['cln'][1]:.4f} | W2={o['w2']:.4f}")
    # tau sweep max location
    o = [c1(0.9, t / 100, **D)['w2'] for t in range(101)]
    print('argmax W2 over tau (g=.9):', o.index(max(o)) / 100, 'max', round(max(o), 4), 'at tau=1:', o[-1])
    # extreme: gamma=0 -> corr = tau r + N(0,(1-tau)^2) = target
    o = c1(0.0, 0.3, **D); print('gamma=0 tau=.3', o['tgt'], o['cln'])
    print('c1 ok')

if __name__ == '__main__':
    print('reward scaled by (1-gamma): r_eff = (1-gamma)*r')
    for g in [0.9, 0.999]:
        for t in [0, 0.5]:
            o = c1(g, t, r=(1 - g) * -1.0, m1=-2.0, s1=0.5)
            print(f"g={g} tau={t}: target mean={o['tgt'][0]:.4f} sd={o['tgt'][1]:.4f} | clean mean={o['cln'][0]:.4f} sd={o['cln'][1]:.4f} | W2={o['w2']:.4f}")
