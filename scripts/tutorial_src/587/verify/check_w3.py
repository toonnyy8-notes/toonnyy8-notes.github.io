# W3 toy bandit: independent Python port of sim.js; compares against sim_js.out
import math,re
M=0xffffffff
def mulberry32(seed):
    a=[seed&M]
    def r():
        a[0]=(a[0]+0x6D2B79F5)&M; t=a[0]
        t=((t^(t>>15))*(t|1))&M
        t=(t^((t+(((t^(t>>7))*(t|61))&M))&M))&M
        return ((t^(t>>14))&M)/4294967296
    return r
def softmax(z):
    m=max(z); e=[math.exp(v-m) for v in z]; s=sum(e); return [v/s for v in e]
def ent(p): return -sum(q*math.log(q) for q in p if q>0)
SUCC=[0.5,1.0,0,0,0,0]
def run(cfg,seed,pB,T=200,G=8,m=4,lr=1.0,tr=None):
    rnd=mulberry32(seed);K=6;z=[math.log(v) for v in [0.81-pB,pB,0.0475,0.0475,0.0475,0.0475]]
    for it in range(T):
        po=softmax(z)
        if tr: tr['found'][it]+=po[1]>0.5; tr['H'][it]+=ent(po)
        acts=[];R=[]
        for g in range(G):
            u=rnd();c=0;a=K-1
            for k in range(K):
                c+=po[k]
                if u<c: a=k;break
            acts.append(a);R.append(1 if rnd()<SUCC[a] else 0)
        mu=sum(R)/G;sd=math.sqrt(sum((x-mu)**2 for x in R)/G)
        if sd<1e-12:
            if tr: tr['skipped']+=1
            continue
        A=[(x-mu)/sd for x in R]
        if cfg[0]=='none': lo=[math.inf]*G; hi=lo
        elif cfg[0]=='ric': lo=[math.sqrt(cfg[1]/po[a]) for a in acts]; hi=lo
        else: lo=[cfg[1]]*G; hi=[cfg[2]]*G
        for s in range(m):
            p=softmax(z);g=[0.0]*K
            for i,a in enumerate(acts):
                r=p[a]/po[a]
                if (r<=1+hi[i]) if A[i]>0 else (r>=1-lo[i]):
                    c=A[i]*r/G
                    for k in range(K): g[k]-=c*p[k]
                    g[a]+=c
            z=[z[k]+lr*g[k] for k in range(K)]
    return softmax(z)
def batch(cfg,pB,S=200,T=200):
    tr={'found':[0]*(T+1),'H':[0]*(T+1),'skipped':0}
    for s in range(S):
        p=run(cfg,s+1,pB,tr=tr); tr['found'][T]+=p[1]>0.5; tr['H'][T]+=ent(p)
    return [x/S for x in tr['found']],[x/S for x in tr['H']],tr['skipped']
js={}
for line in open('sim_js.out'):
    m=re.match(r'pB=(\S+) (.{14}) .*@200 (\S+)/(\S+)/.*skipped=(\d+)',line); js[(m.group(1),m.group(2).strip())]=(float(m.group(3)),float(m.group(4)),int(m.group(5)))
cases=[('0.01','GRPO eps0.2',('ppo',0.2,0.2)),('0.01','DAPO 0.2/0.28',('dapo',0.2,0.28)),('0.01','RIC d0.05',('ric',0.05)),('0.02','RIC d0.02',('ric',0.02)),('0.005','RIC d0.05',('ric',0.05)),('0.01','No clip',('none',)),('0.01','GRPO up1.0',('dapo',0.2,1.0))]
for pB,name,cfg in cases:
    f,H,sk=batch(cfg,float(pB))
    jf,jH,jsk=js[(pB,name)]
    print(f"pB={pB} {name:14s} py found@200={f[200]:.3f} H@200={H[200]:.3f} skipped={sk} | js {jf:.3f} {jH:.3f} {jsk}")
    assert abs(f[200]-jf)<=0.02 and abs(H[200]-jH)<=0.005
print('W3 OK (py vs js within tolerance)')
