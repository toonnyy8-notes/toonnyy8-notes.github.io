"""C4: Prop. 3.5 pushforward on |S|=4: enumerate all 4^4 = 256 deterministic maps."""
from itertools import product
PRESETS = {'A': (0.4, 0.3, 0.2, 0.1), 'B': (0.25, 0.25, 0.25, 0.25), 'C': (1.0, 0.0, 0.0, 0.0)}
for k, mu in PRESETS.items():
    vecs = set(); first = set()
    for f in product(range(4), repeat=4):
        nu = [0.0] * 4
        for x, y in enumerate(f): nu[y] += mu[x]
        nu = tuple(round(v, 10) for v in nu)
        vecs.add(nu); first.add(nu[0])
    mumin = min(m for m in mu if m > 0)
    print(k, mu, 'mu_min/2 =', mumin / 2, '| reachable nu(1):', sorted(first), '| distinct nu vectors:', len(vecs))
