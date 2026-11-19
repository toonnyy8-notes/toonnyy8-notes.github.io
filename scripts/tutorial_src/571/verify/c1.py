# C1: returns-to-go (suffix sums), test-time decrement, token window counts.
PRESETS = {'sparse': [0, 0, 0, 0, 0, 0, 0, 1], 'dense': [1, 0, 2, 1, 0, 3, 1, 2], 'path': [-1, -1, -1, -1, 0]}
def rtg(r): return [sum(r[t:]) for t in range(len(r))]
def test(r, G):
    R = [G]
    for x in r: R.append(R[-1] - x)
    return R          # length T+1; R[T] is what remains after the episode
def window(t, K):     # 1-based t; returns (first timestep, m, tokens total, tokens attended by head at s_t)
    a = max(1, t - K + 1); m = t - a + 1
    return a, m, 3 * m, 3 * m - 1
for name, r in PRESETS.items():
    print(name, 'r=', r, 'RTG=', rtg(r), 'return=', sum(r))
    for G in (sum(r), sum(r) + 5, sum(r) - 2):
        R = test(r, G)
        print(f'  G={G}: test R_hat_1..T={R[:-1]} leftover={R[-1]} offset vs train={[a - b for a, b in zip(R[:-1], rtg(r))]}')
for t, K in [(1, 3), (2, 3), (5, 3), (8, 3), (8, 8), (8, 1), (4, 20)]:
    print(f't={t} K={K}: window starts {window(t, K)[0]}, m={window(t, K)[1]}, tokens={window(t, K)[2]}, attended={window(t, K)[3]}')
