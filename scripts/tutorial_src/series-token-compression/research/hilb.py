import cmath, math
def d2xy(n,d):
    x=y=0; t=d; s=1
    while s<n:
        rx=1&(t//2); ry=1&(t^rx)
        if ry==0:
            if rx==1: x=s-1-x; y=s-1-y
            x,y=y,x
        x+=s*rx; y+=s*ry; t//=4; s*=2
    return x,y
n=32; pts=[d2xy(n,i) for i in range(n*n)]
assert len(set(pts))==n*n
assert all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(pts,pts[1:]))
# every aligned dyadic 4x4 block is a contiguous run of 16
for k in range(0,n*n,16):
    xs={p[0]//4 for p in pts[k:k+16]}; ys={p[1]//4 for p in pts[k:k+16]}
    assert len(xs)==1 and len(ys)==1
# averaged phasor magnitude vs Dirichlet
w=0.7
for L in (1,2,4,8):
    m=abs(sum(cmath.exp(1j*w*p) for p in range(L))/L)
    f=abs(math.sin(L*w/2)/(L*math.sin(w/2)))
    assert abs(m-f)<1e-12
print('hilbert + dirichlet ok')
