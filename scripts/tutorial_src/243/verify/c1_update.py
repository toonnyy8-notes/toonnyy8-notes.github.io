"""C1: linear associative memory updated at test time by Eq. 13-14 (loss = 1/2||Mk-v||^2, App. C).
Same algorithm the page uses. Stream: d=8, contexts A(1-32) B(33-64) A(65-96), 4 unit keys per context."""
import math, sys
from prng import mulberry32, gauss
D, T, POOL = 8, 96, 4
def ctx_of(t): return 0 if t <= 32 or t > 64 else 1          # t is 1-indexed
def make_stream(seed):
    r = mulberry32(seed)
    keys = []
    for c in range(2):
        pool = []
        for _ in range(POOL):
            k = [gauss(r) for _ in range(D)]; n = math.sqrt(sum(x*x for x in k)); pool.append([x/n for x in k])
        keys.append(pool)
    mats = [[[gauss(r)/math.sqrt(D) for _ in range(D)] for _ in range(D)] for _ in range(2)]
    seq = []
    for t in range(1, T+1):
        c = ctx_of(t); j = min(POOL-1, int(r()*POOL)); seq.append((c, j))
    return keys, mats, seq
def mv(M, k): return [sum(M[i][j]*k[j] for j in range(D)) for i in range(D)]
def run(seed=11, theta=0.3, eta=0.6, alpha=0.02, chunk=1):
    keys, mats, seq = make_stream(seed)
    M = [[0.0]*D for _ in range(D)]; S = [[0.0]*D for _ in range(D)]
    out = []; Mref = M  # gradient point (chunk start)
    for t, (c, j) in enumerate(seq, 1):
        if (t-1) % chunk == 0: Mref = [row[:] for row in M]
        k = keys[c][j]; v = mv(mats[c], k)
        e = [a-b for a, b in zip(mv(Mref, k), v)]
        surprise = math.sqrt(sum(x*x for x in e))      # ||grad||_F = ||e||*||k||, ||k||=1
        for a in range(D):
            for b in range(D):
                S[a][b] = eta*S[a][b] - theta*e[a]*k[b]
                M[a][b] = (1-alpha)*M[a][b] + S[a][b]
        Snorm = math.sqrt(sum(x*x for row in S for x in row))
        rec = []
        for cc in range(2):
            s = 0.0
            for kk in keys[cc]:
                d_ = [a-b for a, b in zip(mv(M, kk), mv(mats[cc], kk))]; s += 0.5*sum(x*x for x in d_)
            rec.append(s/POOL)
        out.append(dict(t=t, c=c, surprise=surprise, S=Snorm, EA=rec[0], EB=rec[1]))
    return out, M
def summary(o):
    g = lambda t, k: o[t-1][k]
    return (f"surp@1 {g(1,'surprise'):.4f} surp@33 {g(33,'surprise'):.4f} surp@65 {g(65,'surprise'):.4f} | "
            f"EA@32 {g(32,'EA'):.4f} EB@64 {g(64,'EB'):.4f} EA@64 {g(64,'EA'):.4f} EA@96 {g(96,'EA'):.4f} | "
            f"S@33 {g(33,'S'):.4f} S@40 {g(40,'S'):.4f} max|S| {max(x['S'] for x in o):.3f}")
if __name__ == '__main__':
    presets = {'TTT-like (eta=0, a=0)':(0.3,0.0,0.0), 'Gated DeltaNet-like (eta=0, a=0.02)':(0.3,0.0,0.02),
               'Titans default (eta=0.6, a=0.02)':(0.3,0.6,0.02), 'eta=0.9':(0.3,0.9,0.02), 'a=0.1':(0.3,0.6,0.1),
               'a=1 (clear)':(0.3,0.6,1.0), 'theta=0':(0.0,0.6,0.02), 'theta=1,eta=.95,a=0':(1.0,0.95,0.0)}
    for n, (th, et, al) in presets.items():
        o, _ = run(theta=th, eta=et, alpha=al); print(f'{n:38s} {summary(o)}')
    o,_ = run(); print('first 6 surprises default:', [round(x['surprise'],4) for x in o[:6]])
    print('t=33..40 surprise default:', [round(x['surprise'],4) for x in o[32:40]])
    o,_ = run(theta=0.3,eta=0.0,alpha=0.02); print('t=33..40 surprise eta=0:', [round(x['surprise'],4) for x in o[32:40]])
