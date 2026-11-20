// JS port of topk.py (W4). Draw order per efficient(): k uniforms (l = N..N-k+1, one r() each), then gauss(r) for w~ (two r()).
// naiveW: K gauss draws in index order, then w[0] += at.  node topk.js
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2);}
function erfcc(x){const z=Math.abs(x),t=1/(1+0.5*z);
  const r=t*Math.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))));
  return x>=0?r:2-r;}
function logPhi(x){if(x>-5)return Math.log(0.5*erfcc(-x/Math.SQRT2));const x2=x*x;
  return -x2/2-Math.log(-x)-0.5*Math.log(2*Math.PI)+Math.log1p(-1/x2+3/(x2*x2)-15/(x2*x2*x2));}
const PA=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00],
 PB=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01],
 PC=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00],
 PD=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00];
function ppfTail(p){let x;if(p<0.02425){const q=Math.sqrt(-2*Math.log(p));x=(((((PC[0]*q+PC[1])*q+PC[2])*q+PC[3])*q+PC[4])*q+PC[5])/((((PD[0]*q+PD[1])*q+PD[2])*q+PD[3])*q+1);}
  else{const q=p-0.5,r=q*q;x=(((((PA[0]*r+PA[1])*r+PA[2])*r+PA[3])*r+PA[4])*r+PA[5])*q/(((((PB[0]*r+PB[1])*r+PB[2])*r+PB[3])*r+PB[4])*r+1);}
  const e=0.5*erfcc(-x/Math.SQRT2)-p,u=e*Math.sqrt(2*Math.PI)*Math.exp(x*x/2);return x-u/(1+x*u/2);}
function normPpf(p){if(p<=0)return -Infinity;if(p>=1)return Infinity;return p>0.5?-ppfTail(1-p):ppfTail(p);}
const lse=a=>{const m=Math.max(...a);let s=0;for(const v of a)s+=Math.exp(v-m);return m+Math.log(s);};
function topkZeroMean(r,N,k,sig){let P=0;const out=[];for(let l=N;l>N-k;l--){let u=r();if(!(u>0))u=1e-300;P+=Math.log(u)/l;
  const om=-Math.expm1(P);out.push(om<0.5?-sig*normPpf(om):sig*normPpf(Math.exp(P)));}return out;}
function logMu(c,sig,tau){return sig*sig/(2*tau*tau)-logPhi(c/sig)+logPhi((c-sig*sig/tau)/sig);}
function efficient(r,K,k,at,tau){const sig=Math.sqrt(1-at*at);let V=topkZeroMean(r,K-1,k,sig);const wt=at+sig*gauss(r);let logs,cs,pos=-1;
  if(wt>V[k-1]){const thr=k>=2?V[k-2]:V[k-1];let rk=0;for(const v of V)if(v>wt)rk++;V=V.slice(0,rk).concat([wt],V.slice(rk,k-1));
    logs=V.map(v=>v/tau);if(K-k>0)logs.push(Math.log(K-k)+logMu(thr,sig,tau));cs=2;pos=rk;}
  else{logs=V.map(v=>v/tau);logs.push(wt/tau);if(K-k-1>0)logs.push(Math.log(K-k-1)+logMu(V[k-1],sig,tau));cs=1;}
  const logZ=lse(logs);return{vals:V,wt,case:cs,pos,lam:V.map(v=>Math.exp(v/tau-logZ)),logZ};}
function naiveW(r,K,at){const sig=Math.sqrt(1-at*at),w=new Float64Array(K);for(let i=0;i<K;i++)w[i]=sig*gauss(r);w[0]+=at;return w;}
function ztildeFrom(w,k,at,tau){const sig=Math.sqrt(1-at*at),K=w.length,top=[];// k largest zero-mean coords, descending
  for(let i=1;i<K;i++){const v=w[i];if(top.length<k||v>top[top.length-1]){let j=top.length<k?top.length:k-1;top[j]=v;while(j>0&&top[j]>top[j-1]){[top[j],top[j-1]]=[top[j-1],top[j]];j--;}}}
  let logs,tv;if(w[0]>top[k-1]){const thr=k>=2?top[k-2]:top[k-1];tv=top.slice(0,k-1).concat([w[0]]);logs=tv.map(v=>v/tau);if(K-k>0)logs.push(Math.log(K-k)+logMu(thr,sig,tau));}
  else{tv=top.slice();logs=tv.map(v=>v/tau);logs.push(w[0]/tau);if(K-k-1>0)logs.push(Math.log(K-k-1)+logMu(top[k-1],sig,tau));}
  const lz=lse(logs),lt=lse(tv.map(v=>v/tau));return{logZt:lz,logZtop:lt};}
function logZexact(w,tau){let m=-Infinity;for(const v of w)if(v>m)m=v;let s=0;for(const v of w)s+=Math.exp((v-m)/tau);return m/tau+Math.log(s);}
module.exports={mulberry32,gauss,erfcc,logPhi,normPpf,lse,topkZeroMean,logMu,efficient,naiveW,ztildeFrom,logZexact};
if(require.main===module){
  const f=a=>'['+a.map(x=>x.toFixed(4)).join(', ')+']';
  console.log('gauss seed1000:',[0,0,0].map((_,i,a)=>0).length, (()=>{const r=mulberry32(1000);return [gauss(r),gauss(r)].map(x=>x.toFixed(10)).join(' ')})());
  console.log('normPpf(.975)',normPpf(0.975).toFixed(8),' logMu(.7,.5,.2)',logMu(0.7,0.5,0.2).toFixed(9));
  for(const tau of[1e-3,0.05]){const e=efficient(mulberry32(3),50257,3,0.34,tau);console.log('single tau',tau,f(e.vals),e.wt.toFixed(4),'case',e.case,f(e.lam));}
  for(const [at,tau] of[[0.34,1e-3],[0.34,0.05],[0.9,1e-3],[0.97,1e-3],[0.99,1e-3]]){const r=mulberry32(11);let c2=0,s1=0,l1=0,big=0;const R=400;
    for(let i=0;i<R;i++){const e=efficient(r,50257,3,at,tau);c2+=e.case===2;s1+=e.vals[0]/R;l1+=e.lam[0]/R;big+=e.lam[0]>0.99;}
    console.log('stat at',at,'tau',tau,'Pcase2',(c2/R).toFixed(4),'K1',s1.toFixed(4),'lam1',l1.toFixed(4),'frac',(big/R).toFixed(4));}
  for(const [K,at,k,tau] of[[50257,.34,3,1e-3],[50257,.34,3,.05],[50257,.34,3,.3],[50257,.34,3,1],[50257,.34,1,.05],[50257,.9,3,.05],[50257,.9,3,.3],[1000,.34,3,.3],[1000,.9,1,.05]]){
    const w=naiveW(mulberry32(7),K,at),lz=logZexact(w,tau),z=ztildeFrom(w,k,at,tau);
    console.log('W4b',K,at,k,tau,'logZ',lz.toFixed(4),'Ztop/Z',Math.exp(z.logZtop-lz).toFixed(4),'Zt/Z',Math.exp(z.logZt-lz).toFixed(4),'sumlam',Math.exp(z.logZtop-z.logZt).toFixed(4));}
}
