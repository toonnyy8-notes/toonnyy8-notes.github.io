import random
def block(L,d,m=4): return 8*L*d*d + 4*m*L*d*d + 4*L*L*d
d=768
for N in (256,1024,4096):
    lin=8*N*d*d+16*N*d*d; quad=4*N*N*d
    print(N, 'lin GF',lin/1e9,'quad GF',quad/1e9,'quad share',round(quad/(lin+quad),3),'N/6d',round(N/(6*d),3))
# RTI B/16-like: 17 blocks, prelude 3, core 11, coda 3, N=1024, text T=256 (assumption)
def rti(R,N=1024,T=256,pre=3,core=11,coda=3):
    dense=(pre+core+coda)*block(N+T,d)
    c=(pre+coda)*block(N+T,d)+core*block(R+T,d)
    return dense/c
for R in (1024,512,384,256,128,64,1): print('R',R,'speedup(FLOPs,T=256)',round(rti(R),2),'T=0',round(rti(R,T=0),2))
# HiT ratio loss equivalence
for _ in range(5):
    r=random.uniform(1.1,5); rho=1/r; fh=random.random(); fs=random.random()
    a=r/(r-1)*((r-1)*fh*fs+(1-fh)*(1-fs)); b=1+(fh-rho)*(fs-rho)/(rho*(1-rho))
    assert abs(a-b)<1e-9
print('ratio-loss equivalence ok')
