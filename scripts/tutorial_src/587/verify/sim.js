// W3 toy bandit — page algorithm (to be pasted into page). Run: node sim.js
function mulberry32(a){return function(){a=(a+0x6D2B79F5)|0;let t=Math.imul(a^(a>>>15),a|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296}}
function softmax(z){let m=-Infinity;for(const v of z)if(v>m)m=v;const e=z.map(v=>Math.exp(v-m));let s=0;for(const v of e)s+=v;return e.map(v=>v/s)}
function entropy(p){let h=0;for(const q of p)if(q>0)h-=q*Math.log(q);return h}
const SUCC=[0.5,1.0,0,0,0,0];
function initP(pB){return [0.81-pB,pB,0.0475,0.0475,0.0475,0.0475]}
// cfg: {method:'ppo'|'dapo'|'ric', eps, epsH, delta}
function runSeed(cfg,seed,pB,T=200,G=8,m=4,lr=1.0,trace){
  const rnd=mulberry32(seed),K=6;let z=initP(pB).map(Math.log);
  for(let it=0;it<T;it++){
    const po=softmax(z);
    if(trace){trace.pB[it]+=po[1];trace.H[it]+=entropy(po);trace.found[it]+=po[1]>0.5?1:0}
    const acts=new Array(G),R=new Array(G);
    for(let g=0;g<G;g++){const u=rnd();let c=0,a=K-1;for(let k=0;k<K;k++){c+=po[k];if(u<c){a=k;break}}acts[g]=a;R[g]=rnd()<SUCC[a]?1:0}
    let mu=0;for(const x of R)mu+=x;mu/=G;let v=0;for(const x of R)v+=(x-mu)**2;const sd=Math.sqrt(v/G);
    if(sd<1e-12){if(trace)trace.skipped[0]++;continue}
    const A=R.map(x=>(x-mu)/sd);
    const lo=new Array(G),hi=new Array(G);
    for(let i=0;i<G;i++){if(cfg.method==='none'){lo[i]=hi[i]=Infinity}else if(cfg.method==='ric'){lo[i]=hi[i]=Math.sqrt(cfg.delta/po[acts[i]])}else{lo[i]=cfg.eps;hi[i]=cfg.method==='dapo'?cfg.epsH:cfg.eps}}
    for(let s=0;s<m;s++){
      const p=softmax(z),g=new Array(K).fill(0);
      for(let i=0;i<G;i++){const a=acts[i],r=p[a]/po[a];
        const act=A[i]>0?(r<=1+hi[i]):(r>=1-lo[i]);
        if(act){const c=A[i]*r/G;for(let k=0;k<K;k++)g[k]-=c*p[k];g[a]+=c}}
      for(let k=0;k<K;k++)z[k]+=lr*g[k];
    }
  }
  return softmax(z);
}
function batch(cfg,pB,S=200,T=200){
  const tr={pB:new Float64Array(T+1),H:new Float64Array(T+1),found:new Float64Array(T+1),skipped:new Float64Array(1)};
  let rew=0;
  for(let s=0;s<S;s++){const p=runSeed(cfg,s+1,pB,T,8,4,1.0,tr);tr.pB[T]+=p[1];tr.H[T]+=entropy(p);tr.found[T]+=p[1]>0.5?1:0;rew+=0.5*p[0]+p[1]}
  for(const k of ['pB','H','found'])for(let i=0;i<=T;i++)tr[k][i]/=S;
  return {tr,rew:rew/S};
}
if(typeof module!=='undefined'&&require.main===module){
  const out=[];
  const cfgs=[['GRPO eps0.2',{method:'ppo',eps:0.2}],['DAPO 0.2/0.28',{method:'dapo',eps:0.2,epsH:0.28}],['RIC d0.05',{method:'ric',delta:0.05}],['RIC d0.02',{method:'ric',delta:0.02}],['GRPO eps1.0',{method:'ppo',eps:1.0}],['RIC d0.005',{method:'ric',delta:0.005}],['No clip',{method:'none'}],['GRPO up1.0',{method:'dapo',eps:0.2,epsH:1.0}]];
  for(const pB of [0.01,0.005,0.02])for(const [n,c] of cfgs){const {tr,rew}=batch(c,pB);
    const at=i=>`${tr.found[i].toFixed(3)}/${tr.H[i].toFixed(3)}/${tr.pB[i].toFixed(3)}`;
    out.push(`pB=${pB} ${n.padEnd(14)} found/H/pB @0 ${at(0)} @25 ${at(25)} @50 ${at(50)} @100 ${at(100)} @200 ${at(200)} rew=${rew.toFixed(3)} skipped=${tr.skipped[0]}`)}
  console.log(out.join('\n'));
}
module.exports={runSeed,batch,mulberry32};
