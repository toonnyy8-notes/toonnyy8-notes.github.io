"""Acceptance values for the SOAR tutorial widgets (round 2). Same formulas the page uses. Exits non-zero on any failed assert."""
import math, json, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).parent
def close(a, b, tol, msg): assert abs(a-b) <= tol, f'{msg}: {a} vs {b}'
# ---------- C1 ----------
def pow1m(p,k): return math.exp(k*math.log1p(-p))
def passk(p,k): return -math.expm1(k*math.log1p(-p))
def nonzero(p,G): return passk(p,G) - p**G
print('C1 (binary-reward idealisation)')
for p in [1e-4,2e-3,1e-2]:
    print(f' p={p:g}: filter={pow1m(p,128):.4f} pass@1={passk(p,1):.4e} pass@8={passk(p,8):.4f} pass@32={passk(p,32):.4f} nzG32={nonzero(p,32):.4f} nzG128={nonzero(p,128):.4f}')
close(pow1m(.002,128),.7739,1e-4,'filter');close(passk(.002,8),.0159,1e-4,'pass@8');close(nonzero(.002,32),.0621,1e-4,'nz32');close(nonzero(.002,128),.2261,1e-4,'nz128')
close(passk(1e-4,1)/1e-4,1,1e-9,'expm1');close(nonzero(.5,32),1-2*.5**32,1e-12,'p=.5')
# ---------- C2 ----------
print('C2')
def softmax(th):
    m=max(th);e=[math.exp(t-m) for t in th];Z=sum(e);return [x/Z for x in e]
def c2(theta,S,z,R,est):
    pi0=softmax(theta);pS=sum(pi0[i] for i in S);pic=[pi0[i]/pS if i in S else 0 for i in range(6)]
    g=len(z);A=[R[i]-(sum(R)-R[i])/(g-1) for i in range(g)] if est=='rloo' else list(R)
    L=[sum(a*((j==zi)-pic[j]) for a,zi in zip(A,z)) for j in range(6)]
    Q=[sum(a*((j==zi)-pi0[j]) for a,zi in zip(A,z)) for j in range(6)]
    for j in range(6): close(L[j]-Q[j], sum(A)*(pi0[j]-pic[j]), 1e-15, 'L-Q identity')
    return pi0,pS,pic,A,L,Q
TH=[1.0,0.5,0.0,-0.5,0.8,0.2];S={0,1,2,3};Z=[0,1,1,3];R=[0.02,-0.01,0.00,0.03]
# finite-difference check that Q_j = sum_i A_i d/dtheta_j ln pi0(z_i) and L_j = ... ln pi(z_i)
def lnpi0(th,z): return math.log(softmax(th)[z])
def lnpi(th,z): p=softmax(th);return math.log(p[z]/sum(p[i] for i in S))
for est in ['rloo','reinforce']:
    pi0,pS,pic,A,L,Q=c2(TH,S,Z,R,est)
    for j in range(6):
        h=1e-6;tp=TH[:];tm=TH[:];tp[j]+=h;tm[j]-=h
        close(sum(a*(lnpi0(tp,z)-lnpi0(tm,z))/(2*h) for a,z in zip(A,Z)),Q[j],1e-8,'FD Q')
        close(sum(a*(lnpi(tp,z)-lnpi(tm,z))/(2*h) for a,z in zip(A,Z)),L[j],1e-8,'FD L')
    print(f' {est}: pi0={[round(x,4) for x in pi0]} pi0(S)={pS:.4f} 1/pi0(S)={1/pS:.3f} A={[round(x,5) for x in A]} sumA={sum(A):+.5f}')
    print(f'   L={[round(x,5) for x in L]}\n   Q={[round(x,5) for x in Q]} max|L-Q|={max(abs(l-q) for l,q in zip(L,Q)):.3e}')
pi0,_,_,A,L,Q=c2(TH,S,Z,R,'reinforce');assert Q[4]<L[4]==0 and Q[5]<0  # sumA>0 -> rejected outputs pushed DOWN
_,_,_,A2,L2,Q2=c2(TH,S,Z,[-0.02,0.01,0.0,-0.03],'reinforce');assert sum(A2)<0 and Q2[4]>0 and Q2[5]>0
print(f' sumA<0 case R=(-.02,.01,0,-.03): Q5={Q2[4]:+.5f} Q6={Q2[5]:+.5f} (raised)')
# ---------- C3a: integer net-change counts c_{d,j}, R = c/64 ----------
print('C3a')
def c3a(C,prev,tau=0.01):
    Rdj=[[c/64 for c in row] for row in C];Rd=[sum(r)/len(r) for r in Rdj];Rhat=sum(Rd)/len(Rd)
    MA=(sum(prev)+Rhat)/(len(prev)+1);promote=MA>tau
    ds=max(range(len(Rd)),key=lambda d:(Rd[d],-d));js=sorted(range(4),key=lambda j:(Rdj[ds][j],j))[1]
    A=[Rd[d]-(sum(Rd)-Rd[d])/(len(Rd)-1) for d in range(len(Rd))]
    return Rd,Rhat,MA,promote,ds,js,A
C=[[1,2,0,1],[-1,0,1,-1],[0,1,-1,0],[2,1,3,0]]
Rd,Rhat,MA,pr,ds,js,A=c3a(C,[0.010,0.014])
print(f' Rd={[round(x,6) for x in Rd]} Rhat={Rhat:.6f} MA={MA:.6f} promote={pr} d*={ds+1} j*={js+1} (c={C[ds][js]}) A={[round(x,6) for x in A]} sumA={sum(A):.1e}')
assert pr and ds==3 and js==1;close(MA,0.010930,1e-6,'MA')
_,_,MA2,pr2,*_=c3a(C,[0.002,0.006]);print(f' prev .002,.006: MA={MA2:.6f} promote={pr2}');assert not pr2
# ---------- C3b: shared + individual noise ----------
print('C3b')
def erfcc(x):
    z=abs(x);t=1/(1+0.5*z)
    r=t*math.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))))
    return r if x>=0 else 2-r
worst=max(abs(erfcc(x)/math.erfc(x)-1) for x in [i/100 for i in range(-500,1001)]);assert worst<1.2e-7
print(f' erfcc max rel err on [-5,10]: {worst:.2e}')
def c3b(mu,sd,sc,r,m,g=4,tau=0.01):
    sMA=math.sqrt((sc*sc+sd*sd/(g*r))/m);P=0.5*erfcc((tau-mu)/(sMA*math.sqrt(2)));return sMA,P,198*P
for args in [(0,.02,0,4,3),(0,.02,0,1,1),(0,.02,0,4,1),(0,.02,0,1,3),(0,.02,.005,4,3),(0,.02,.005,1,1),(0,.02,.01,4,3),(.01,.02,.005,4,3)]:
    s,P,E=c3b(*args);print(f' mu={args[0]} sd={args[1]} sc={args[2]} r={args[3]} m={args[4]}: sMA={s:.5f} P={P:.4e} E198={E:.3f}')
close(c3b(0,.02,0,4,3)[1],2.660e-4,1e-7,'P lower bound');close(c3b(0,.02,.005,4,3)[1],7.153e-3,2e-6,'P shared')
# ---------- C4: RESULTS from paper.txt ----------
print('C4')
subprocess.run([sys.executable,'-I',str(HERE/'extract_results.py')],check=True,capture_output=True)
RES=json.loads((HERE/'results.json').read_text())
assert [len(RES[k]) for k in ('MATH','HARP','OLYMP')]==[13,12,12]
row=lambda ds,name:next(r for r in RES[ds] if r['name']==name)['vals']
# spot-check transcriptions against the hand-copied design numbers
assert row('MATH','Hard-Only')==[[0.5,0.1],[1.7,0.4],[3.2,0.8],[5.7,1.5],[9.6,2.6]]
assert row('MATH','SOAR-PQ (Ours)')[0]==[1.7,1.0] and row('MATH','SOAR-PQ (Ours)')[4]==[18.9,5.3]
assert row('MATH','HARP train (128)')[4]==[23.0,3.9] and row('MATH','MATH train (Full)')[4]==[22.0,2.4]
assert row('HARP','SOAR-PQ (Ours)')[4]==[12.3,2.0] and row('HARP','MATH train (Full)')[4]==[16.2,0.4]
assert row('OLYMP','MATH train (128)')[4]==[14.6,1.4] and row('OLYMP','MATH train (Full)')[4]==[13.1,0.9]
assert row('MATH','Inference with Grounded-T')[4]==[8.3,1.1] and row('HARP','Inference with Grounded-T')[4]==[6.8,0.7]
DEFAULT={'MATH':'SOAR-PQ (Ours)','HARP':'SOAR-PQ (Ours)','OLYMP':'SOAR-PQ (HARP) (Ours)'}
def derive(ds,ki,name):
    m,sd=row(ds,name)[ki];ho,hsd=row(ds,'Hard-Only')[ki];full=row(ds,'MATH train (Full)')[ki][0]
    rec=(m-ho)/(full-ho) if full-ho>0 else None
    return round(m-ho,1),m/ho,rec,(m-sd)<=(ho+hsd)
for ds in RES:
    for name in [r['name'] for r in RES[ds] if 'Ours' in r['name'] or r['name'].startswith(('Intrinsic','Hard-Only (g'))]:
        for ki,k in [(0,1),(4,32)]:
            d,ra,rec,ov=derive(ds,ki,name);print(f' {ds} {name} pass@{k}: {d:+.1f}pp ratio={ra:.2f} rec={"—" if rec is None else f"{rec:.2f}"} overlap={ov}')
assert derive('MATH',4,'SOAR-PQ (Ours)')[:1]==(9.3,) and round(derive('MATH',4,'SOAR-PQ (Ours)')[2],2)==0.75
assert round(derive('HARP',4,'SOAR-PQ (Ours)')[2],2)==0.51 and round(derive('OLYMP',4,'SOAR-PQ (HARP) (Ours)')[2],2)==0.82
# ratio SDs for pass@1 (first-order propagation, page-derived)
for ds,name in [('MATH','SOAR-PQ (Ours)'),('HARP','SOAR-PQ (Ours)')]:
    (m,s),(h,hs)=row(ds,name)[0],row(ds,'Hard-Only')[0];ra=m/h;print(f' pass@1 ratio {ds}: {ra:.2f} ± {ra*math.hypot(s/m,hs/h):.2f} (delta method)')
# ---------- F1 ----------
print('F1')
def Er(s,m=32): return (1-s)-pow1m(s,m) if s<1 else 0.0
ps=1-32**(-1/31);print(f' s*={ps:.4f} E[u](s*)={Er(ps):.4f}');close(ps,.1058,1e-4,'s*');close(Er(ps),.8663,1e-4,'Er*')
print(' '+' '.join(f'{s}:{Er(s):.3f}' for s in [0.01,0.05,0.1,0.3,0.5,0.9]))
print('ALL CHECKS PASSED')
