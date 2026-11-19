// C2: Algorithm 1 on a lookup-table toy model (same algorithm as page). Run: node c2.js
const CANON=['for','i','in','range','(','n',')',':','sum','+=','arr','[','i',']'];
const START=CANON.slice(); START[12]='j';
// conf(arm,pos,token): confidence of `token` at `pos`; argmax(arm,pos): token predicted when pos is masked (illustrative values)
const BASEC=[0.99,0.99,0.99,0.88,0.99,0.97,0.99,0.99,0.95,0.98,0.99,0.99,0.97,0.99];
const ARMS={
  mdlm:{conf:(p,t)=>t===CANON[p]?BASEC[p]:(p===12&&t==='j')?0.98:(p===8&&t==='total')?0.60:0.05, argmax:p=>p===8?'total':CANON[p]},
  cdlm:{conf:(p,t)=>t===CANON[p]?BASEC[p]:(p===12&&t==='j')?0.19:0.05, argmax:p=>CANON[p]}
};
function refine(arm,tau,T){
  const A=ARMS[arm]; let z=START.slice(); const log=[]; let passes=0;
  for(let t=0;t<=T;t++){
    passes++;
    const zt=z.map((tok,p)=>tok==='[M]'?A.argmax(p):tok);       // fill masks, keep visible
    if(t===T){log.push({t,filled:zt.join(' ')});return {out:zt,passes,log,ok:zt.every((x,p)=>x===CANON[p])};}
    const c=zt.map((tok,p)=>A.conf(p,tok));
    const tp=Math.round(tau*100); const r=[];c.forEach((v,p)=>{if(Math.round(v*100)<=tp)r.push(p)}); // integer percent: avoids float edge at tau=c=0.99
    log.push({t,filled:zt.join(' '),remask:r.map(p=>p+':'+zt[p]+'('+c[p]+')').join(', ')});
    if(r.length===0) return {out:zt,passes,log,ok:zt.every((x,p)=>x===CANON[p])};
    z=zt.map((tok,p)=>r.includes(p)?'[M]':tok);
  }
}
module.exports={refine,CANON,START,ARMS};
if(require.main===module){
  for(const arm of ['mdlm','cdlm']) for(const tau of [0.80,0.90,0.96,0.99]) for(const T of [1,4]){
    const R=refine(arm,tau,T);
    console.log(`${arm} tau=${tau} T=${T}: passes=${R.passes} correct=${R.ok}  final="${R.out.join(' ')}"`);
    R.log.forEach(l=>console.log(`   t=${l.t} remask=[${l.remask??'(return)'}]`));
  }
}
