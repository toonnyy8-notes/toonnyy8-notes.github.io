"""C4 (toy, values 推定):
(a) IVF expectile weight (Eq. 15) & implicit policy (Eq. 24) at one state with 4 dataset actions.
    V = expectile(Q, pi_beta, mu)  <=>  sum pi_beta w (Q - V) = 0 ;  pi_mu = pi_beta w / sum(pi_beta w);  check V == E_{pi_mu}[Q].
(b) App. D.5: pointwise max of velocities. Two actions with Gaussian Q-returns N(m_a, s_a^2), closed-form velocities
    u_a(z|tau) = m_a + k_a(tau)(z - tau m_a). 'Naive' V-flow: dz/dtau = max_a u_a(z|tau), z(0)=z0.
    E[z(1)] computed with J midpoint-quantile nodes z0_j = inv_norm((j+.5)/J) and Euler with S steps
    (velocity evaluated at tau_i = i/S, i=0..S-1)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import expectile, inv_norm

def implicit(Q, pb, mu):
    V = expectile(Q, pb, mu)
    w = [mu if q >= V else 1 - mu for q in Q]
    Z = sum(p * x for p, x in zip(pb, w))
    pi = [p * x / Z for p, x in zip(pb, w)]
    return V, w, pi, sum(p * q for p, q in zip(pi, Q))

def u(m, s, z, tau):
    sig2 = tau * tau * s * s + (1 - tau) ** 2
    return m + (tau * s * s - (1 - tau)) / sig2 * (z - tau * m)

def envelope(ms, ss, J=101, S=400):
    tot = 0.0; tot_a = [0.0] * len(ms)
    for j in range(J):
        z0 = inv_norm((j + 0.5) / J); z = z0; za = [z0] * len(ms)
        for i in range(S):
            t = i / S
            z += max(u(m, s, z, t) for m, s in zip(ms, ss)) / S
            za = [x + u(m, s, x, t) / S for x, m, s in zip(za, ms, ss)]
        tot += z; tot_a = [a + b for a, b in zip(tot_a, za)]
    return tot / J, [a / J for a in tot_a]

if __name__ == '__main__':
    Q = [-3.0, -2.0, -1.2, -0.5]; pb = [0.4, 0.3, 0.2, 0.1]
    print('(a) Q', Q, 'pi_beta', pb, 'mean', sum(p * q for p, q in zip(pb, Q)))
    for mu in [0.5, 0.7, 0.8, 0.9, 0.99, 0.999]:
        V, w, pi, EV = implicit(Q, pb, mu)
        assert abs(V - EV) < 1e-12
        print(f' mu={mu}: V={V:.4f}  w={w}  pi_mu={[round(p, 4) for p in pi]}  E_pi_mu[Q]={EV:.4f}')
    print(' mu->1 limit = max in support', max(Q))
    print('(b) envelope')
    cases = [([-2.0, -2.2], [0.3, 1.2]), ([-2.0, -2.0], [0.5, 0.5]), ([-2.0, -3.0], [0.3, 0.3]), ([-2.0, -2.5], [0.2, 1.5])]
    for ms, ss in cases:
        e, ea = envelope(ms, ss)
        e2, _ = envelope(ms, ss, J=201, S=1600)
        print(f' m={ms} s={ss}: E[naive z1]={e:.4f} (J=201,S=1600: {e2:.4f})  per-action Euler means={[round(x, 4) for x in ea]}  max_a Q={max(ms)}  gap={e - max(ms):.4f}')
        assert e >= max(ms) - 2e-3
    print('c4 ok')
