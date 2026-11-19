"""C2: memory depth L_M. 1-D key k ~ U[-2,2], value v = sin(2k). Memory M: L=1 -> w*k (matrix memory, no bias);
L>=2 -> MLP width H=16, SiLU, hidden biases, no output bias. Online test-time update Eq. 13-14 on every parameter,
loss 1/2 (M(k)-v)^2, analytic backprop. Same algorithm as the page."""
import math
from prng import mulberry32, gauss
H = 16
def sig(a): return 1/(1+math.exp(-a)) if a >= 0 else math.exp(a)/(1+math.exp(a))
def target(k): return math.sin(2*k)
def init(L, seed):
    r = mulberry32(seed); P = []
    if L == 1: return [{'W': [[gauss(r)]], 'b': None}]
    dims = [1] + [H]*(L-1) + [1]
    for l in range(L):
        fi, fo = dims[l], dims[l+1]
        W = [[gauss(r)*math.sqrt(1/fi) for _ in range(fi)] for _ in range(fo)]
        b = [gauss(r)*0.5 for _ in range(fo)] if l < L-1 else None
        P.append({'W': W, 'b': b})
    return P
def forward(P, k):
    hs = [[k]]; As = []
    for l, p in enumerate(P):
        a = [sum(wr[j]*hs[-1][j] for j in range(len(wr))) + (p['b'][i] if p['b'] else 0) for i, wr in enumerate(p['W'])]
        if l < len(P)-1:
            As.append(a); hs.append([x*sig(x) for x in a])
        else:
            return a[0], hs, As
def grads(P, k, v):
    out, hs, As = forward(P, k); d = [out - v]; G = [None]*len(P)
    for l in range(len(P)-1, -1, -1):
        p = P[l]; h = hs[l]
        gW = [[d[i]*h[j] for j in range(len(h))] for i in range(len(d))]
        gb = d[:] if p['b'] else None
        G[l] = {'W': gW, 'b': gb}
        if l > 0:
            dh = [sum(p['W'][i][j]*d[i] for i in range(len(d))) for j in range(len(h))]
            a = As[l-1]; d = [dh[j]*sig(a[j])*(1 + a[j]*(1-sig(a[j]))) for j in range(len(h))]
    return G, 0.5*(out-v)**2
def evalmse(P, n=201):
    return sum((forward(P, -2+4*i/(n-1))[0]-target(-2+4*i/(n-1)))**2 for i in range(n))/n
def nparams(P): return sum(len(p['W'])*len(p['W'][0]) + (len(p['b']) if p['b'] else 0) for p in P)
def run(L, theta, eta, alpha, T=3000, seed=5, data_seed=9, checkpoints=(100, 300, 1000, 3000)):
    P = init(L, seed); S = [{'W': [[0.0]*len(p['W'][0]) for _ in p['W']], 'b': [0.0]*len(p['b']) if p['b'] else None} for p in P]
    r = mulberry32(data_seed); res = {}; div = None
    for t in range(1, T+1):
        k = -2 + 4*r(); G, loss = grads(P, k, target(k))
        for p, s, g in zip(P, S, G):
            for i in range(len(p['W'])):
                for j in range(len(p['W'][0])):
                    s['W'][i][j] = eta*s['W'][i][j] - theta*g['W'][i][j]; p['W'][i][j] = (1-alpha)*p['W'][i][j] + s['W'][i][j]
            if p['b']:
                for i in range(len(p['b'])):
                    s['b'][i] = eta*s['b'][i] - theta*g['b'][i]; p['b'][i] = (1-alpha)*p['b'][i] + s['b'][i]
        if not math.isfinite(loss) or max(abs(x) for p in P for row in p['W'] for x in row) > 1e6:
            div = t; break
        if t in checkpoints: res[t] = evalmse(P)
    return res, div, P
if __name__ == '__main__':
    import sys
    lin_best = (sum(k*target(k) for k in [-2+4*i/200 for i in range(201)]) / sum(k*k for k in [-2+4*i/200 for i in range(201)]))
    print('best linear w (grid LS):', round(lin_best, 4), 'mse', round(sum((lin_best*k-target(k))**2 for k in [-2+4*i/200 for i in range(201)])/201, 4))
    print('mse of M=0:', round(sum(target(-2+4*i/200)**2 for i in range(201))/201, 4))
    cfgs = [(0.02,0.9,0.0),(0.01,0.9,0.0),(0.05,0.9,0.0),(0.02,0.0,0.0),(0.02,0.9,0.001),(0.05,0.0,0.0),(0.1,0.0,0.0)]
    if len(sys.argv) > 1: cfgs = [tuple(map(float, sys.argv[1:4]))]
    for th, et, al in cfgs:
        line = f'theta={th} eta={et} alpha={al}:'
        for L in [1,2,3,4]:
            res, div, P = run(L, th, et, al)
            line += f'  L{L}(n={nparams(P)}) ' + ('DIV@%d' % div if div else ' '.join(f'{res[c]:.4f}' for c in sorted(res)))
        print(line)
