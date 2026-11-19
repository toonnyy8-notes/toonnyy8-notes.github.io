// C3: localization-metrics model (same algorithm as page). Run: node c3.js
// clean logit u ~ (1-pi) N(M0,S0^2) + pi N(MH,SH^2); corrupted logit ~ N(me, se^2); positions independent; one corrupted position.
const U0=-30, U1=30, NU=6001, DU=(U1-U0)/(NU-1);
const UG=Float64Array.from({length:NU},(_,i)=>U0+i*DU);
function erf(x){const s=x<0?-1:1;x=Math.abs(x);const t=1/(1+0.3275911*x);
  return s*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x));}
const Phi=z=>0.5*(1+erf(z/Math.SQRT2)), sig=u=>1/(1+Math.exp(-u)), logit=c=>Math.log(c/(1-c));
const npdf=(u,m,s)=>Math.exp(-0.5*((u-m)/s)**2)/(s*Math.sqrt(2*Math.PI));
const wq=i=>(i===0||i===NU-1)?0.5*DU:DU;               // trapezoid weights
function meanSig(m,s){let a=0;for(let i=0;i<NU;i++)a+=wq(i)*npdf(UG[i],m,s)*sig(UG[i]);return a;}
function solveM(target,s){let lo=-20,hi=20;for(let k=0;k<200;k++){const mid=(lo+hi)/2;if(meanSig(mid,s)<target)lo=mid;else hi=mid;}return (lo+hi)/2;}
const MH=0, SH=1;
// easy clean tokens: solve (M0,S0) so that mean confidence = 0.970 (Table 28, clean) and P(c<=0.9) = 0.03 (Table 39: ~3% remasked)
const LT9=Math.log(9);
function m0For(S){return solveM(0.970,S);}
let sLo=0.3,sHi=6;for(let k=0;k<80;k++){const sm=(sLo+sHi)/2;const F=0.5*(1+erf((LT9-m0For(sm))/sm/Math.SQRT2));if(F<0.03)sLo=sm;else sHi=sm;}
const S0=+((sLo+sHi)/2).toFixed(3), M0=+m0For(S0).toFixed(3);
// cumulative binomial P(Bin(N,p)<=k) for k=0..KMAX-1, log-space; guards p<=0, p>=1
function binCdfs(N,p,KMAX,out){
  if(p<=0){out.fill(1);return;} if(p>=1){for(let k=0;k<KMAX;k++)out[k]=k>=N?1:0;return;}
  let lt=N*Math.log1p(-p),acc=0;const r=Math.log(p)-Math.log1p(-p);
  for(let k=0;k<KMAX;k++){ if(k>0){ if(k>N){out[k]=1;continue;} lt+=Math.log(N-k+1)-Math.log(k)+r; } acc+=Math.exp(lt); out[k]=Math.min(1,acc); }
}
function model(P){ // P={med,se,pi,n,tau,K}
  const me=logit(P.med), KMAX=10, hit=new Float64Array(KMAX), tmp=new Float64Array(KMAX);
  let cl=0, co=0, mass=0;
  for(let i=0;i<NU;i++){
    const u=UG[i], w=wq(i), fe=npdf(u,me,P.se);
    const fc=(1-P.pi)*npdf(u,M0,S0)+P.pi*npdf(u,MH,SH);
    cl+=w*fc*sig(u); co+=w*fe*sig(u); mass+=w*fe;
    const Fc=(1-P.pi)*Phi((u-M0)/S0)+P.pi*Phi((u-MH)/SH);
    binCdfs(P.n-1,Fc,KMAX,tmp);
    for(let k=0;k<KMAX;k++) hit[k]+=w*fe*tmp[k];
  }
  const lt=logit(P.tau);
  const Fct=(1-P.pi)*Phi((lt-M0)/S0)+P.pi*Phi((lt-MH)/SH), Fet=Phi((lt-me)/P.se);
  return {clean:cl,corr:co,gap:cl-co,hit:Array.from(hit),hitK:hit[P.K-1],fp5:1-hit[4]/5,
    remaskPct:100*((P.n-1)*Fct+Fet)/P.n, errRemasked:Fet, cleanRemasked:(P.n-1)*Fct,
    noneBelow:Math.pow(1-Fct,P.n-1)*(1-Fet), mass};
}
module.exports={model,solveM,M0,S0,MH,SH,logit,sig};
if(require.main===module){
  console.log('M0 =',M0,' S0 =',S0,' clean mean',model({med:.5,se:1,pi:0,n:2,tau:.9,K:1}).clean.toFixed(4),' P(clean<=0.9)',(0.5*(1+erf((LT9-M0)/S0/Math.SQRT2))).toFixed(4));
  const SE=1.5;
  const pre={};
  for(const [name,target] of [['base',0.705],['mdlm',0.567],['cdlm',0.192]]){
    const me=solveM(target,SE); pre[name]=+sig(me).toFixed(3);
    console.log(`preset ${name}: corrupted mean ${target} -> me ${me.toFixed(4)} median ${sig(me).toFixed(4)} -> rounded ${pre[name]}`);
  }
  const row=(lab,P)=>{const r=model(P);console.log(lab.padEnd(34),
    ['clean',r.clean,'corr',r.corr,'gap',r.gap,'hit1',r.hit[0],'hitK',r.hitK,'fp5',r.fp5,'remask%',r.remaskPct,'errRm',r.errRemasked,'clRm',r.cleanRemasked,'none',r.noneBelow,'mass',r.mass]
    .map(v=>typeof v==='number'?v.toFixed(4):v).join(' '));};
  console.log('\n--- acceptance rows (defaults n=56, tau=0.9, K=5, se=1.5, pi=0.05) ---');
  const D={se:SE,pi:0.05,n:56,tau:0.9,K:5};
  const R=(lab,o)=>row(lab,Object.assign({},D,o));
  R('Base-like',{med:pre.base}); R('MDLM-like',{med:pre.mdlm}); R('CDLM-like',{med:pre.cdlm});
  R('CDLM-like pi=0',{med:pre.cdlm,pi:0}); R('CDLM-like pi=0.1',{med:pre.cdlm,pi:0.1});
  R('Base-like pi=0',{med:pre.base,pi:0});
  R('CDLM-like tau=0.8',{med:pre.cdlm,tau:0.8}); R('CDLM-like tau=0.95',{med:pre.cdlm,tau:0.95});
  R('Base-like tau=0.8',{med:pre.base,tau:0.8}); R('Base-like tau=0.95',{med:pre.base,tau:0.95});
  R('CDLM-like n=20',{med:pre.cdlm,n:20}); R('CDLM-like n=200',{med:pre.cdlm,n:200});
  R('MDLM-like K=1',{med:pre.mdlm,K:1}); R('MDLM-like K=10',{med:pre.mdlm,K:10});
  R('extreme med=.01 se=4 pi=.2 n=200 tau=.99 K=10',{med:0.01,se:4,pi:0.2,n:200,tau:0.99,K:10});
  R('extreme med=.99 se=.3 pi=0 n=20 tau=.5 K=1',{med:0.99,se:0.3,pi:0,n:20,tau:0.5,K:1});
  R('Base-like n=5 (page minimum)',{med:pre.base,n:5}); R('CDLM-like n=5 (page minimum)',{med:pre.cdlm,n:5});
  R('sanity n=2 med=.5 pi=0 (script only; page n>=5, fp5 formula invalid for n<5)',{med:0.5,pi:0,n:2,K:1});
  // sanity: n=2 hit1 = P(ce < cc)
  // Monte Carlo cross-check for CDLM-like
  let a=12345;const rnd=()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};
  const gauss=()=>Math.sqrt(-2*Math.log(1-rnd()))*Math.cos(2*Math.PI*rnd());
  for(const [lab,med,pi] of [['MC MDLM-like pi=0',pre.mdlm,0],['MC Base-like pi=.05',pre.base,0.05]]){
    let h1=0,h5=0,R=200000;const me=logit(med);
    for(let r=0;r<R;r++){const ue=me+SE*gauss();let below=0;for(let j=0;j<55;j++){const uc=rnd()<pi?MH+SH*gauss():M0+S0*gauss();if(uc<ue)below++;}if(below<1)h1++;if(below<5)h5++;}
    console.log(lab,'hit1',(h1/R).toFixed(4),'hit5',(h5/R).toFixed(4));
  }
}
