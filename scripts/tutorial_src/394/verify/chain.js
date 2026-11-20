// JS reference port of chain.py (W3). node chain.js -> must print the same numbers as accept.py w3.
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const onehot=(i,K)=>{const v=new Array(K).fill(0);v[i]=1;return v};
const prior=(kind,K)=>kind==='mask'?onehot(K-1,K):new Array(K).fill(1/K);
function post(zt,x,at,as,pi){const ats=at/as,K=x.length,num=new Array(K);let S=0;
  for(let j=0;j<K;j++){const lik=(1-ats)*pi[zt]+(j===zt?ats:0);num[j]=lik*(as*x[j]+(1-as)*pi[j]);S+=num[j];}
  if(S<1e-300)return onehot(zt,K);for(let j=0;j<K;j++)num[j]/=S;return num;}
function psiTheta(zt,x,at,as,kap,pi){const a=post(zt,x,at,as,pi),c=post(zt,x,at,1,pi);return a.map((ai,j)=>kap*ai+(1-kap)*(as*c[j]+(1-as)*pi[j]));}
const sigmaMax=(at,as)=>at<=0?1:Math.min(1,(1-as)/at);
const kFromS=(sig,as)=>1-as<=0?1:1-sig/(1-as);
function kappaSched(kind,t,at,as,p){if(kind==='ancestral')return 1;if(kind==='window')return(p.toff<=t&&t<=p.ton)?p.kap:1;
  if(kind==='rescale')return kFromS(p.eta*sigmaMax(at,as),as);if(kind==='cap')return kFromS(Math.min(p.eta,sigmaMax(at,as)),as);}
function catSample(r,p){const u=r();let c=0;for(let i=0;i<p.length;i++){c+=p[i];if(u<c)return i;}for(let i=p.length-1;i>=0;i--)if(p[i]>0)return i;}
function runChain(kind,sched,T,p,seed,L=32,Ktok=8,emax=0.5,emin=0.02){const K=kind==='mask'?Ktok+1:Ktok,pi=prior(kind,K),m=K-1,r=mulberry32(seed);
  const xs=Array.from({length:L},(_,l)=>l%Ktok);let z=kind==='mask'?new Array(L).fill(m):xs.map(()=>catSample(r,pi));
  for(let i=T;i>=1;i--){const t=i/T,s=(i-1)/T,at=1-t,as=1-s;let c=0;for(let l=0;l<L;l++)if(z[l]===xs[l])c++;c/=L;
    const eps=emax-(emax-emin)*c,kap=kappaSched(sched,t,at,as,p),nz=new Array(L);
    for(let l=0;l<L;l++){let xth;if(kind==='mask'&&z[l]!==m)xth=onehot(z[l],K);else{xth=new Array(K).fill(0);for(let j=0;j<Ktok;j++)xth[j]=eps/(Ktok-1);xth[xs[l]]=1-eps;}
      nz[l]=catSample(r,psiTheta(z[l],xth,at,as,kap,pi));}z=nz;}
  let e=0;for(let l=0;l<L;l++)if(z[l]!==xs[l])e++;return e/L;}
const errRate=(kind,sched,T,p,R=32,emax=0.5)=>{let s=0;for(let k=0;k<R;k++)s+=runChain(kind,sched,T,p,1000+k,32,8,emax);return s/R;};
module.exports={runChain,errRate};
if(require.main===module){const p={kap:0.9,ton:0.6,toff:0.1,eta:0.05},Ts=[8,16,32,64,128,256,512];
  const em=+(process.argv[2]||0.5);
  for(const kind of['mask','unif'])for(const sch of['ancestral','window','rescale'])console.log(kind,sch.padEnd(9),Ts.map(T=>errRate(kind,sch,T,p,32,em).toFixed(4)).join(' '));}
