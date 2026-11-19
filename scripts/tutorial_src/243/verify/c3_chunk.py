"""C3: chunked (mini-batch) memory update, Eq. 16-18. (a) per-token loop whose gradients use M at chunk start;
(b) matmul form: M_end = c_M*M_t' + c_S*S_t' - theta * E diag(w) K^T, w_l = sum_{m=l}^{end} (1-a)^{end-m} eta^{m-l}
(no beta_t/beta_i division). Compares (a) vs (b), and chunk b vs b=1."""
import math
from c1_update import make_stream, mv, D, T, POOL, run
def matmul_form(seed=11, theta=0.3, eta=0.6, alpha=0.02, b=8):
    keys, mats, seq = make_stream(seed)
    M = [[0.0]*D for _ in range(D)]; S = [[0.0]*D for _ in range(D)]
    ends = []
    for s0 in range(0, T, b):
        idx = list(range(s0, min(T, s0+b))); n = len(idx)
        K = [keys[seq[t][0]][seq[t][1]] for t in idx]
        E = [[a-c for a, c in zip(mv(M, K[i]), mv(mats[seq[idx[i]][0]], K[i]))] for i in range(n)]
        g = 1 - alpha
        w = [sum(g**(n-1-m) * eta**(m-l) for m in range(l, n)) for l in range(n)]   # coefficient of u_l in M_end
        cS_M = sum(g**(n-1-m) * eta**(m+1) for m in range(n))                       # coefficient of S_t' in M_end
        cM = g**n
        q = [-theta * eta**(n-1-l) for l in range(n)]                                 # coefficient of u_l in S_end
        newM = [[cM*M[a][c] + cS_M*S[a][c] - theta*sum(w[l]*E[l][a]*K[l][c] for l in range(n)) for c in range(D)] for a in range(D)]
        newS = [[eta**n*S[a][c] + sum(q[l]*E[l][a]*K[l][c] for l in range(n)) for c in range(D)] for a in range(D)]
        M, S = newM, newS; ends.append([row[:] for row in M])
    return ends
def loop_ends(b, **kw):
    # re-run loop and capture M at chunk ends
    import c1_update as c1
    keys, mats, seq = make_stream(kw.get('seed', 11))
    th, et, al = kw.get('theta',0.3), kw.get('eta',0.6), kw.get('alpha',0.02)
    M = [[0.0]*D for _ in range(D)]; S = [[0.0]*D for _ in range(D)]; ends=[]
    for t,(c,j) in enumerate(seq):
        if t % b == 0: Mref=[r[:] for r in M]
        k=keys[c][j]; e=[x-y for x,y in zip(mv(Mref,k),mv(mats[c],k))]
        for a in range(D):
            for cc in range(D):
                S[a][cc]=et*S[a][cc]-th*e[a]*k[cc]; M[a][cc]=(1-al)*M[a][cc]+S[a][cc]
        if (t+1)%b==0 or t==T-1: ends.append([r[:] for r in M])
    return ends
if __name__ == '__main__':
    for (th,et,al) in [(0.3,0.6,0.02),(0.3,0.0,0.02),(0.3,0.0,0.0)]:
        print(f'theta={th} eta={et} alpha={al}')
        _, M1 = run(theta=th, eta=et, alpha=al, chunk=1)
        for b in [1,2,4,8,16,32]:
            A = loop_ends(b, theta=th, eta=et, alpha=al); B = matmul_form(theta=th, eta=et, alpha=al, b=b)
            diff = max(abs(x-y) for MA,MB in zip(A,B) for ra,rb in zip(MA,MB) for x,y in zip(ra,rb))
            o, Mb = run(theta=th, eta=et, alpha=al, chunk=b)
            dev = math.sqrt(sum((x-y)**2 for ra,rb in zip(Mb,M1) for x,y in zip(ra,rb)))
            meanEB = sum(x['EB'] for x in o[32:64])/32
            print(f'  b={b:2d} steps={T//b:3d} max|loop-matmul|={diff:.1e}  EA@32={o[31]["EA"]:.4f} meanEB(33-64)={meanEB:.4f} EB@64={o[63]["EB"]:.4f} EA@96={o[95]["EA"]:.4f} ||M_b-M_1||={dev:.4f}')
