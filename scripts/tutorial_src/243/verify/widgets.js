// Reference JS for C1-C4 (the page copies these functions). Run: node widgets.js
'use strict';
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2);}
// ---------- C1/C2 stream ----------
const D=8,T=96,POOL=4;
const ctxOf=t=>(t<=32||t>64)?0:1; // t 1-indexed
function makeStream(seed){const r=mulberry32(seed);const keys=[];for(let c=0;c<2;c++){const pool=[];for(let i=0;i<POOL;i++){const k=Array.from({length:D},()=>gauss(r));const n=Math.hypot(...k);pool.push(k.map(x=>x/n));}keys.push(pool);}
  const mats=[0,1].map(()=>Array.from({length:D},()=>Array.from({length:D},()=>gauss(r)/Math.sqrt(D))));
  const seq=[];for(let t=1;t<=T;t++){seq.push([ctxOf(t),Math.min(POOL-1,Math.floor(r()*POOL))]);}return{keys,mats,seq};}
const mv=(M,k)=>M.map(row=>row.reduce((s,x,j)=>s+x*k[j],0));
const BIG=1e3; // divergence guard: any |parameter| > 1e3 stops the run
function runC1({seed=11,theta=0.3,eta=0.6,alpha=0.02,chunk=1}={}){
  const {keys,mats,seq}=makeStream(seed);const V=[0,1].map(c=>keys[c].map(k=>mv(mats[c],k)));
  let M=Array.from({length:D},()=>new Array(D).fill(0)),S=Array.from({length:D},()=>new Array(D).fill(0)),Mref=M;const out=[];let diverged=null;
  for(let t=1;t<=T;t++){const [c,j]=seq[t-1];if((t-1)%chunk===0)Mref=M.map(r=>r.slice());
    const k=keys[c][j],v=V[c][j],e=mv(Mref,k).map((x,i)=>x-v[i]);const surprise=Math.hypot(...e);
    let mx=0;for(let a=0;a<D;a++)for(let b=0;b<D;b++){S[a][b]=eta*S[a][b]-theta*e[a]*k[b];M[a][b]=(1-alpha)*M[a][b]+S[a][b];mx=Math.max(mx,Math.abs(M[a][b]));}
    const Sn=Math.hypot(...S.flat());const rec=[0,1].map(cc=>keys[cc].reduce((s,kk,i)=>{const p=mv(M,kk);return s+0.5*p.reduce((q,x,m)=>q+(x-V[cc][i][m])**2,0);},0)/POOL);
    out.push({t,c,surprise,S:Sn,EA:rec[0],EB:rec[1]});
    if(!(mx<BIG)){diverged=t;break;}}
  return{out,M,diverged};}
// ---------- C3 matmul form (0-based l: w_l uses eta^(m-l), q_l = -theta*eta^(n-1-l); JS 0**0 === 1) ----------
function matmulEnds({seed=11,theta=0.3,eta=0.6,alpha=0.02,b=8}={}){
  const {keys,mats,seq}=makeStream(seed);let M=Array.from({length:D},()=>new Array(D).fill(0)),S=M.map(r=>r.slice());const ends=[];const g=1-alpha;
  for(let s0=0;s0<T;s0+=b){const idx=[];for(let t=s0;t<Math.min(T,s0+b);t++)idx.push(t);const n=idx.length;
    const K=idx.map(t=>keys[seq[t][0]][seq[t][1]]);const E=idx.map((t,i)=>{const v=mv(mats[seq[t][0]],K[i]);return mv(M,K[i]).map((x,m)=>x-v[m]);});
    const w=[],q=[];for(let l=0;l<n;l++){let s=0;for(let m=l;m<n;m++)s+=g**(n-1-m)*eta**(m-l);w.push(s);q.push(-theta*eta**(n-1-l));}
    let cS=0;for(let m=0;m<n;m++)cS+=g**(n-1-m)*eta**(m+1);const cM=g**n,en=eta**n;
    const nM=M.map((row,a)=>row.map((x,c)=>{let s=0;for(let l=0;l<n;l++)s+=w[l]*E[l][a]*K[l][c];return cM*x+cS*S[a][c]-theta*s;}));
    const nS=S.map((row,a)=>row.map((x,c)=>{let s=0;for(let l=0;l<n;l++)s+=q[l]*E[l][a]*K[l][c];return en*x+s;}));
    M=nM;S=nS;ends.push(M.map(r=>r.slice()));}
  return ends;}
function loopEnds(o){const {keys,mats,seq}=makeStream(o.seed??11);const th=o.theta??0.3,et=o.eta??0.6,al=o.alpha??0.02,b=o.b;
  let M=Array.from({length:D},()=>new Array(D).fill(0)),S=M.map(r=>r.slice()),Mref=M;const ends=[];
  for(let t=0;t<T;t++){const [c,j]=seq[t];if(t%b===0)Mref=M.map(r=>r.slice());const k=keys[c][j],v=mv(mats[c],k),e=mv(Mref,k).map((x,i)=>x-v[i]);
    for(let a=0;a<D;a++)for(let cc=0;cc<D;cc++){S[a][cc]=et*S[a][cc]-th*e[a]*k[cc];M[a][cc]=(1-al)*M[a][cc]+S[a][cc];}
    if((t+1)%b===0||t===T-1)ends.push(M.map(r=>r.slice()));}return ends;}
// ---------- C2 deep memory ----------
const H=16;const sig=a=>a>=0?1/(1+Math.exp(-a)):Math.exp(a)/(1+Math.exp(a));const target=k=>Math.sin(2*k);
function initMLP(L,seed){const r=mulberry32(seed);if(L===1)return[{W:[[gauss(r)]],b:null}];const dims=[1,...Array(L-1).fill(H),1];const P=[];
  for(let l=0;l<L;l++){const fi=dims[l],fo=dims[l+1];const W=Array.from({length:fo},()=>Array.from({length:fi},()=>gauss(r)*Math.sqrt(1/fi)));const b=l<L-1?Array.from({length:fo},()=>gauss(r)*0.5):null;P.push({W,b});}return P;}
function fwd(P,k){const hs=[[k]],As=[];for(let l=0;l<P.length;l++){const p=P[l],h=hs[hs.length-1];const a=p.W.map((wr,i)=>wr.reduce((s,x,j)=>s+x*h[j],0)+(p.b?p.b[i]:0));
  if(l<P.length-1){As.push(a);hs.push(a.map(x=>x*sig(x)));}else return{out:a[0],hs,As};}}
function step(P,S,k,v,theta,eta,alpha){const {out,hs,As}=fwd(P,k);let d=[out-v];
  for(let l=P.length-1;l>=0;l--){const p=P[l],s=S[l],h=hs[l];let dh=null;
    if(l>0){dh=h.map((_,j)=>p.W.reduce((q,wr,i)=>q+wr[j]*d[i],0));}
    for(let i=0;i<p.W.length;i++){for(let j=0;j<h.length;j++){s.W[i][j]=eta*s.W[i][j]-theta*d[i]*h[j];p.W[i][j]=(1-alpha)*p.W[i][j]+s.W[i][j];}
      if(p.b){s.b[i]=eta*s.b[i]-theta*d[i];p.b[i]=(1-alpha)*p.b[i]+s.b[i];}}
    if(l>0){const a=As[l-1];d=dh.map((x,j)=>{const sg=sig(a[j]);return x*sg*(1+a[j]*(1-sg));});}}
  return 0.5*(out-v)**2;}
function evalMSE(P,n=201){let s=0;for(let i=0;i<n;i++){const k=-2+4*i/(n-1);s+=(fwd(P,k).out-target(k))**2;}return s/n;}
function runDeep({L=2,theta=0.01,eta=0.9,alpha=0,Tn=3000,seed=5,dataSeed=9,cps=[100,300,1000,3000]}={}){
  const P=initMLP(L,seed),S=P.map(p=>({W:p.W.map(r=>r.map(()=>0)),b:p.b?p.b.map(()=>0):null}));const r=mulberry32(dataSeed);const res={};let div=null;
  for(let t=1;t<=Tn;t++){const k=-2+4*r();const loss=step(P,S,k,target(k),theta,eta,alpha);
    let mx=0;for(const p of P)for(const row of p.W)for(const x of row)mx=Math.max(mx,Math.abs(x));
    if(!Number.isFinite(loss)||!(mx<=BIG)){div=t;break;}if(cps.includes(t))res[t]=evalMSE(P);}
  return{res,div,P};}
// ---------- C4 masks ----------
function maskRow(v,{N=16,Np=2,C=4,Nl=2,w=4,t}){ // returns {persist:[], mem:[], seq:[]} with 'direct'|'memory'|'none'
  const seq=[];for(let j=1;j<=N;j++){let cat='none';
    if(v==='MAC'){const seg=Math.ceil(t/C),sj=Math.ceil(j/C);if(j<=t)cat=(sj===seg)?'direct':'memory';}
    else{if(j<=t)cat=(j>t-w)?'direct':'memory';}seq.push(cat);}
  const persist=new Array(Np).fill(v==='MAL'?'memory':'direct');const mem=v==='MAC'?new Array(Nl).fill('direct'):[];return{persist,mem,seq};}
function counts(v,o){const r=maskRow(v,o);const all=[...r.persist,...r.mem,...r.seq];return{direct:all.filter(x=>x==='direct').length,memory:all.filter(x=>x==='memory').length,none:all.filter(x=>x==='none').length};}
function pairs(v,o){let s=0;for(let t=1;t<=o.N;t++)s+=counts(v,{...o,t}).direct;return s;}
if(require.main===module){
  const f=x=>x.toFixed(4);
  console.log('== C1');
  for(const [n,p] of Object.entries({'TTT':{theta:.3,eta:0,alpha:0},'GDN':{theta:.3,eta:0,alpha:.02},'Titans':{theta:.3,eta:.6,alpha:.02},'eta.9':{theta:.3,eta:.9,alpha:.02},'a.1':{theta:.3,eta:.6,alpha:.1},'a.2':{theta:.3,eta:.6,alpha:.2},'eta.6a0':{theta:.3,eta:.6,alpha:0},'div':{theta:1,eta:.95,alpha:0}})){
    const {out:o,diverged}=runC1(p);const g=(t,k)=>o[t-1]?f(o[t-1][k]):'-';const ms=(a,b)=>o.length>=b?f(o.slice(a-1,b).reduce((s,x)=>s+x.surprise,0)/(b-a+1)):'-';
    console.log(n.padEnd(7),`surp1 ${g(1,'surprise')} mSurp33-40 ${ms(33,40)} mSurp65-72 ${ms(65,72)} EA32 ${g(32,'EA')} EB64 ${g(64,'EB')} EA64 ${g(64,'EA')} EA96 ${g(96,'EA')} S40 ${g(40,'S')} div ${diverged}`);}
  console.log("== C3 chunk");
  for(const p of [{theta:.3,eta:.6,alpha:.02},{theta:.3,eta:0,alpha:.02}]){const ref=runC1({...p,chunk:1}).M;
    for(const b of [1,2,4,8,16,32]){const A=loopEnds({...p,b}),B=matmulEnds({...p,b});let d=0;A.forEach((M,i)=>M.forEach((r,a)=>r.forEach((x,c)=>{d=Math.max(d,Math.abs(x-B[i][a][c]));})));
      const R=runC1({...p,chunk:b});const o=R.out;const dev=Math.hypot(...R.M.flat().map((x,i)=>x-ref.flat()[i]));const mEB=o.length>=64?o.slice(32,64).reduce((s,x)=>s+x.EB,0)/32:NaN;
      console.log(JSON.stringify(p),`b=${b} steps=${Math.ceil(T/b)} maxdiff=${d.toExponential(1)} EA32=${f(o[31].EA)} meanEB=${f(mEB)} EA96=${o[95]?f(o[95].EA):'-'} dev=${f(dev)} div=${R.diverged}`);}}
  console.log("== C2 deep");
  for(const L of [1,2,3,4]){const {res,div,P}=runDeep({L});const np=P.reduce((s,p)=>s+p.W.length*p.W[0].length+(p.b?p.b.length:0),0);console.log(`L=${L} n=${np}`,div?('DIV@'+div):Object.entries(res).map(([k,v])=>k+':'+v.toFixed(5)).join(' '));}
  for(const L of [1,2,3,4]){const {res}=runDeep({L,eta:0,theta:0.05});console.log(`eta0 th.05 L=${L}`,Object.entries(res).map(([k,v])=>k+':'+v.toFixed(5)).join(' '));}
  console.log('== C1 step-size control (momentum with constant eta scales effective step by 1/(1-eta): 0.3/(1-0.6)=0.75)');
  {const p={theta:.75,eta:0,alpha:.02};const {out:o,diverged}=runC1(p);console.log('th.75eta0',`EA32 ${f(o[31].EA)} EB64 ${f(o[63].EB)} EA64 ${f(o[63].EA)} EA96 ${f(o[95].EA)} S40 ${f(o[39].S)} div ${diverged}`);
   const E=(q)=>{const v=[];for(let s=11;s<31;s++)v.push(runC1({...q,seed:s}).out[63].EB);return v;};const med=v=>{const z=[...v].sort((a,b)=>a-b);return (z[9]+z[10])/2;};
   const a=E({theta:.3,eta:0}),c=E({theta:.75,eta:0}),m=E({theta:.3,eta:.6});
   console.log('seeds11-30 EB64 median',`eta0 ${f(med(a))} th.75eta0 ${f(med(c))} eta.6 ${f(med(m))} mom<eta0 ${m.filter((x,i)=>x<a[i]).length}/20 mom<ctrl ${m.filter((x,i)=>x<c[i]).length}/20`);}
  console.log('== extremes');
  for(const L of [1,2,3,4])for(const al of [0,.01]){const {div}=runDeep({L,theta:.05,eta:.95,alpha:al});console.log('C2 extreme',L,JSON.stringify({theta:.05,eta:.95,alpha:al}),div?'DIV@'+div:'ok');}
  for(const L of [1,2,3,4]){const {res}=runDeep({L,theta:0,eta:0,alpha:.01});console.log('C2 decay',L,'th0 eta0 a.01 3000:',res[3000].toFixed(4));}
  {const ek=(Math.sin(4)/4-Math.cos(4))/2,es=0.5-Math.sin(8)/16;const ks=Array.from({length:201},(_,i)=>-2+4*i/200);const w=ks.reduce((s,k)=>s+k*target(k),0)/ks.reduce((s,k)=>s+k*k,0);console.log("C2 ref grid201 best-linear w:",w.toFixed(4),"mse:",(ks.reduce((s,k)=>s+(w*k-target(k))**2,0)/201).toFixed(4),"zero-mse:",(ks.reduce((s,k)=>s+target(k)**2,0)/201).toFixed(4),"| continuous U[-2,2] (closed form) w:",(ek/(4/3)).toFixed(4),"mse:",(es-ek*ek/(4/3)).toFixed(4),"zero-mse:",es.toFixed(4));}
  for(const b of [16,32])console.log('C3 b',b,'div@',runC1({chunk:b}).diverged);
  console.log('== C4');
  const o={N:16,Np:2,C:4,Nl:2,w:4};for(const v of ['MAC','MAG','MAL']){console.log(v,'t=10',JSON.stringify(counts(v,{...o,t:10})),'t=4',JSON.stringify(counts(v,{...o,t:4})),'pairs',pairs(v,o));}
  let full=0;for(let t=1;t<=16;t++)full+=2+t;console.log('full causal + prefix pairs',full);
}
module.exports={mulberry32,gauss,runC1,matmulEnds,loopEnds,runDeep,counts,pairs};
