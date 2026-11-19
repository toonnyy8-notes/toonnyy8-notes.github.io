// C1: corruption process + supervised positions (same algorithm as page). Run: node c1.js
const TOK=['for','k','in','range','(','2',',','n','-','1',')',':','if','n','%','k','==','0',':','return','False'];
const VOC=['while','x','+','7','def','*','len','3','<','None','total','j','/','or','5','pass','[',']','>=','is'];
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
// role per position: 'M' masked, 'N' replaced, 'C' clean visible. Draw order per position: u1,u2,u3 (always all three).
function corrupt(seed,lam,alpha,mode){
  const r=mulberry32(seed), out=[];
  for(let i=0;i<TOK.length;i++){
    const u1=r(),u2=r(),u3=r();
    if(u1<lam){out.push({role:'M',tok:'[M]'});continue;}
    if(mode!=='mdlm'&&u2<alpha){let j=Math.floor(u3*VOC.length);if(VOC[j]===TOK[i])j=(j+1)%VOC.length;out.push({role:'N',tok:VOC[j]});continue;}
    out.push({role:'C',tok:TOK[i]});
  }
  return out;
}
// per-position loss weight: masked 1/|M|, replaced lamNoise/|N|, clean ctw/|C\N| (empty set -> term omitted)
function weights(seq,mode,lamNoise=0.1,ctw=0.1){
  const cnt={M:0,N:0,C:0};seq.forEach(s=>cnt[s.role]++);
  const w=seq.map(s=>s.role==='M'?1/cnt.M:s.role==='N'?(mode==='mdlm'?0:lamNoise/cnt.N):(mode==='ctw'?ctw/cnt.C:0));
  return {w,cnt,total:{M:cnt.M?1:0,N:(mode!=='mdlm'&&cnt.N)?lamNoise:0,C:(mode==='ctw'&&cnt.C)?ctw:0}};
}
// analytic (本頁推導)
const fracAt=(lam,a)=>({M:lam,N:(1-lam)*a,C:(1-lam)*(1-a)});
const fracAvg=a=>({M:0.5,N:a/2,C:(1-a)/2});
const pEmptyAt=(lam,a,n)=>Math.pow(1-(1-lam)*a,n);
const pEmptyAvg=(a,n)=>a===0?1:(1-Math.pow(1-a,n+1))/(a*(n+1));
module.exports={TOK,VOC,mulberry32,corrupt,weights,fracAt,fracAvg,pEmptyAt,pEmptyAvg};
if(require.main===module){
  // default seed: first seed >=1 such that at lam=0.3, alpha=0.1 (cdlm) there is >=1 N and |M| in [4,8]
  let seed=1;for(;seed<1000;seed++){const s=corrupt(seed,0.3,0.1,'cdlm');const c={M:0,N:0,C:0};s.forEach(x=>c[x.role]++);if(c.N>=1&&c.M>=4&&c.M<=8)break;}
  console.log('default seed',seed);
  for(const [lam,a,mode] of [[0.3,0.1,'cdlm'],[0.3,0.1,'mdlm'],[0.3,0.1,'ctw'],[0.0,0.1,'cdlm'],[1.0,0.1,'cdlm'],[0.3,0.5,'cdlm'],[0.3,0,'cdlm']]){
    const s=corrupt(seed,lam,a,mode), W=weights(s,mode);
    console.log(`lam=${lam} a=${a} ${mode}: `+s.map(x=>x.role+':'+x.tok).join(' '));
    console.log('   counts',JSON.stringify(W.cnt),'weights M',W.cnt.M?(1/W.cnt.M).toFixed(4):'-','N',W.cnt.N&&mode!=='mdlm'?(0.1/W.cnt.N).toFixed(4):'-','C',mode==='ctw'&&W.cnt.C?(0.1/W.cnt.C).toFixed(4):'-','group totals',JSON.stringify(W.total));
  }
  console.log('\nanalytic fractions / P(N empty)');
  for(const [lam,a] of [[0.3,0.1],[0.5,0.1],[0.3,0.5],[0,0.1],[1,0.1]]){const f=fracAt(lam,a);console.log(`lam=${lam} a=${a}: M ${f.M.toFixed(4)} N ${f.N.toFixed(4)} C ${f.C.toFixed(4)} | P(N empty) n=21 ${pEmptyAt(lam,a,21).toFixed(4)} n=4096 ${pEmptyAt(lam,a,4096).toExponential(3)}`);}
  for(const a of [0.1,0.04,0.5,0.9]){const f=fracAvg(a);console.log(`avg over lam~U[0,1], a=${a}: M ${f.M} N ${f.N.toFixed(4)} C ${f.C.toFixed(4)} | P(N empty) n=21 ${pEmptyAvg(a,21).toFixed(4)} n=4096 ${pEmptyAvg(a,4096).toFixed(5)}`);}
  // Monte Carlo check of averaged P(N empty), n=21, a=0.1
  const r=mulberry32(99);let e=0,R=200000;for(let k=0;k<R;k++){const lam=r();let any=false;for(let i=0;i<21;i++){const m=r()<lam;const rep=r()<0.1;if(!m&&rep)any=true;}if(!any)e++;}
  console.log('MC P(N empty) n=21 a=0.1:',(e/R).toFixed(4));
}
