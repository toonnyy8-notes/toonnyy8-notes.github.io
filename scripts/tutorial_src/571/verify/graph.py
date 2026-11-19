# C3: shortest-path toy (App. A.3) with a tabular return-conditioned model (K=1) in place of GPT.
# Mirrors the page algorithm exactly (mulberry32, draw order, tie-breaking).
import sys
from collections import deque
M32 = 0xFFFFFFFF
def imul(a, b): return ((a & M32) * (b & M32)) & M32
def mulberry32(seed):
    st = [seed & M32]
    def r():
        st[0] = (st[0] + 0x6D2B79F5) & M32
        a = st[0]
        t = imul(a ^ (a >> 15), 1 | a)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return r
N, GOAL, P, NTRAJ, T = 20, 0, 0.1, 1000, 10
INF = 99
def build(seed):
    rng = mulberry32(seed)
    out = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j and rng() < P: out[i].append(j)
    trajs = []
    for _ in range(NTRAJ):
        s = 1 + int(rng() * (N - 1)); tr = [s]
        for _ in range(T):
            if s == GOAL: break
            nb = out[s]
            if nb: s = nb[int(rng() * len(nb))]
            tr.append(s)
        trajs.append(tr)
    C = {}  # (s, L) -> {a: count}, finite L only
    for tr in trajs:
        if GOAL not in tr: continue
        g = tr.index(GOAL)
        for t in range(g):
            d = C.setdefault((tr[t], g - t), {}); d[tr[t + 1]] = d.get(tr[t + 1], 0) + 1
    # BFS distances to goal
    rev = [[] for _ in range(N)]
    for i in range(N):
        for j in out[i]: rev[j].append(i)
    dist = [INF] * N; dist[GOAL] = 0; q = deque([GOAL])
    while q:
        v = q.popleft()
        for u in rev[v]:
            if dist[u] == INF: dist[u] = dist[v] + 1; q.append(u)
    return out, trajs, C, dist
def best_action(C, s, L):
    d = C[(s, L)]; m = max(d.values()); return min(a for a in d if d[a] == m)
def avail(C, s): return sorted(L for (u, L) in C if u == s)
def greedy(C, s0):
    p = [s0]; s = s0
    for _ in range(T):
        if s == GOAL: break
        Ls = avail(C, s)
        if not Ls: return p, INF
        s = best_action(C, s, Ls[0]); p.append(s)
    return p, (len(p) - 1 if s == GOAL else INF)
def target(C, s0, L0):
    p = [s0]; s = s0; L = L0; used = []
    for _ in range(T):
        if s == GOAL: break
        Ls = avail(C, s)
        if not Ls: return p, INF, used
        if L not in Ls: L = min(Ls, key=lambda x: (abs(x - L), x))
        used.append(L)
        s = best_action(C, s, L); p.append(s); L = L - 1
    return p, (len(p) - 1 if s == GOAL else INF), used
def contains(tr, p):
    n = len(p); return any(tr[i:i + n] == p for i in range(len(tr) - n + 1))
def hist(vals, n):
    h = [0] * 11
    for v in vals: h[v - 1 if v <= 10 else 10] += 1
    return [round(c / n, 4) for c in h]
def stats(seed, verbose=False):
    out, trajs, C, dist = build(seed)
    reach = [s for s in range(1, N) if dist[s] < INF]
    rw = [tr.index(GOAL) if GOAL in tr else INF for tr in trajs]
    gen = {s: greedy(C, s) for s in range(1, N)}
    succ = [s for s in gen if gen[s][1] < INF]
    stitched = [s for s in succ if not any(contains(tr, gen[s][0]) for tr in trajs)]
    opt = [s for s in succ if gen[s][1] == dist[s]]
    r = dict(seed=seed, edges=sum(map(len, out)), reach=len(reach), succ=len(succ), opt=len(opt),
             stitched=len(stitched), rw_success=round(sum(v < INF for v in rw) / NTRAJ, 4))
    if verbose:
        print(r)
        print('out-edges:', {i: out[i] for i in range(N)})
        print('dist:', dist)
        print('random-walk hist (1..10, inf):', hist(rw, NTRAJ))
        print('shortest hist over 19 starts:', hist([dist[s] for s in range(1, N)], N - 1))
        print('DT-tabular hist over 19 starts:', hist([gen[s][1] for s in range(1, N)], N - 1))
        print('mean steps (reachable starts): shortest', round(sum(dist[s] for s in reach) / len(reach), 4),
              'DT', round(sum(gen[s][1] for s in succ) / max(1, len(succ)), 4))
        for s in range(1, N):
            print(f' start {s}: dist={dist[s]} gen={gen[s][0]} len={gen[s][1]} avail={avail(C, s)}'
                  f"{' STITCHED' if s in stitched else ''}")
        return out, trajs, C, dist
    return r
if __name__ == '__main__':
    if len(sys.argv) > 1:
        out, trajs, C, dist = stats(int(sys.argv[1]), True)
        if len(sys.argv) > 2:
            s0 = int(sys.argv[2])
            for L0 in range(1, 11):
                p, n, used = target(C, s0, L0)
                print(f'  target L0={L0}: path={p} steps={n} L used={used}')
    else:
        for sd in range(1, 41): print(stats(sd))
