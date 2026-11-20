// C3 math (mirrors verify/check.py section 3). Runs in a Blob Worker; the main thread also builds it with new Function for direct use and fallback.
const FV={TV:r=>.5*Math.abs(r-1),chi2:r=>(r-1)**2,KL:r=>r>0?-Math.log(r):Infinity,H:r=>(Math.sqrt(r)-1)**2};
function gb(f,V,b){return b*f(1+V/b)+(1-b)*f(1-V/(1-b))}
function Lf(f,V,J){let best=Infinity,bb=0;for(let j=0;j<J;j++){const b=(1-V)*(j+.5)/J,v=gb(f,V,b);if(v<best){best=v;bb=b}}return[best,bb]}
function table(){const out={};for(const k in FV){const L=new Float64Array(499);let C=Infinity,Vs=0;for(let i=1;i<500;i++){const V=i/500;L[i-1]=Lf(FV[k],V,1000)[0];const q=L[i-1]/(V*V);if(q<C){C=q;Vs=V}}out[k]={L,C,Vs}}return out}
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function dirichlet1(r,K){const g=[];let s=0;for(let i=0;i<K;i++){const e=-Math.log(1-r());g.push(e);s+=e}return g.map(e=>e/s)}
function pairs(seed,K,N=400){const r=mulberry32(seed),out=[];for(let n=0;n<N;n++){const P=dirichlet1(r,K),Q=dirichlet1(r,K);out.push([P,Q])}return out}
function scatter(fk,K,seed){const f=FV[fk],P=pairs(seed,K),pts=new Float64Array(2*P.length);let viol=0,minr=Infinity;
  P.forEach(([p,q],n)=>{let V=0,D=0;for(let i=0;i<K;i++){V+=Math.abs(p[i]-q[i]);if(q[i]>0)D+=q[i]*f(p[i]/q[i])}V/=2;pts[2*n]=V;pts[2*n+1]=D;
    if(V<1e-9)return;const L=Lf(f,V,400)[0],ra=D/L;if(ra<minr)minr=ra;if(D<L*(1-1e-3))viol++});
  return{pts,viol,minr,P0:P[0]}}
if(typeof window==='undefined'&&typeof self!=='undefined'){self.onmessage=e=>{const m=e.data;const res=m.type==='table'?table():scatter(m.f,m.K,m.seed);self.postMessage({reqId:m.reqId,type:m.type,res})}}
