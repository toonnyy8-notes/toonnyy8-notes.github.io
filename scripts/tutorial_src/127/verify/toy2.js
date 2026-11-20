// C3 toy v2: noisy inner loop with r repeats (algorithm the page will use).
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}
const sig=x=>1/(1+Math.exp(-x));
const HARD=Float64Array.from({length:64},(_,j)=>3+3*(j+0.5)/64);
function acc(s){let a=0;for(let j=0;j<64;j++)a+=sig(s-HARD[j]);return a/64}
const DEF={seed:1,T:200,g:4,n:64,r:4,L:10,eta:0.3,nu:0.3,sigT:0.5,mu0:-3,lrT:0.05,tau:0.01,win:3,learnTeacher:true};
function runToy(o={}){
  const P=Object.assign({},DEF,o);
  const rng=mulberry32(P.seed);let s=0,mu=P.mu0,m=0,v=0;const promos=[],hist=[],Rh=[];
  for(let t=0;t<P.T;t++){
    const X=[];for(let k=0;k<P.g;k++){const a=new Float64Array(P.n);for(let i=0;i<P.n;i++)a[i]=mu+P.sigT*gauss(rng);X.push(a)}
    const a0=acc(s),R=[],S=[];
    for(let k=0;k<P.g;k++){
      let s2=s;for(let l=0;l<P.L;l++){let gs=0;for(let i=0;i<P.n;i++){const p=sig(s2-X[k][i]);gs+=p*(1-p)}s2+=P.eta*gs/P.n}
      const runs=[];for(let j=0;j<P.r;j++){const sj=s2+P.nu*gauss(rng);runs.push({s:sj,R:acc(sj)-a0})}
      S.push(runs);R.push(runs.reduce((x,y)=>x+y.R,0)/P.r);
    }
    const Rhat=R.reduce((x,y)=>x+y,0)/P.g;Rh.push(Rhat);
    const w=Rh.slice(-P.win);const MA=w.reduce((x,y)=>x+y,0)/w.length;
    let promoted=false;
    if(MA>P.tau){let kb=0;for(let k=1;k<P.g;k++)if(R[k]>R[kb])kb=k;
      const srt=S[kb].slice().sort((x,y)=>x.R-y.R);s=srt[Math.floor((P.r-1)/2)].s;promos.push(t);promoted=true}
    if(P.learnTeacher){
      const sumR=R.reduce((x,y)=>x+y,0);let gm=0;
      for(let k=0;k<P.g;k++){const A=R[k]-(sumR-R[k])/(P.g-1);let sk=0;for(let i=0;i<P.n;i++)sk+=(X[k][i]-mu)/(P.sigT*P.sigT);gm+=A*sk}
      gm/=P.g*P.n;
      m=0.9*m+0.1*gm;v=0.999*v+0.001*gm*gm;
      mu+=P.lrT*(m/(1-Math.pow(0.9,t+1)))/(Math.sqrt(v/(1-Math.pow(0.999,t+1)))+1e-8);
    }
    hist.push({t,Rhat,MA,mu,s,acc:acc(s),promoted});
  }
  return {promos,hist,final:hist[hist.length-1]};
}
module.exports={runToy,acc,DEF,mulberry32,gauss};
