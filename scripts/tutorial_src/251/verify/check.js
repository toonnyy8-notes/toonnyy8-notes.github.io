const C=require('./core.js'),D=require('./table1.json');const f=(x,n=3)=>x.toFixed(n);
console.log('## C1 cell counts');const cc=C.cellCounts(D.rows);for(const k of Object.keys(cc).sort())console.log(k,cc[k]);
for(const R of ['I','E'])console.log(R,D.rows.filter(r=>r[3]===R).length);
const rule={};for(const r of D.rows)for(const [x,s] of r[8]){rule[x]=rule[x]||{T2:0,txt:0};rule[x][s==='T2'?'T2':'txt']++}console.log('rules',JSON.stringify(rule));
const yr={};for(const r of D.rows){yr[r[2]]=yr[r[2]]||{I:0,E:0};yr[r[2]][r[3]]++}console.log('years',JSON.stringify(yr));
console.log('<=2023',D.rows.filter(r=>r[2]<=2023).length,' 2025-2026',D.rows.filter(r=>r[2]>=2025).length);
console.log('\n## C2 (d,T,seed,mode,a,b,g) -> c_i list, e_i list, fro');
const P={lin:[1,0,1],dec:[.9,0,1],del:[1,1,1],gdel:[.9,.9,1]};
const cases=[[8,16,7,'rand'],[8,8,7,'ortho'],[8,12,7,'ortho'],[8,4,7,'rand'],[16,16,7,'rand'],[32,16,7,'rand']];
for(const [d,T,s,m] of cases)for(const p in P){const r=C.c2run(d,T,s,m,...P[p]);const mc=r.c.reduce((a,b)=>a+b)/T,me=r.e.reduce((a,b)=>a+b)/T;
 console.log(`d=${d} T=${T} ${m} ${p}: meanC=${f(mc)} meanE=${f(me)} c_first=${f(r.c[0])} c_last=${f(r.c[T-1])} e_last=${f(r.e[T-1])} fro=${f(r.fro)} over=${r.over.filter(x=>x).length}`)}
// analytic checks
{const r=C.c2run(8,8,7,'ortho',.9,0,1);console.log('ortho decay c_i vs 0.9^(T-1-i):',r.c.map((x,i)=>f(x)+'/'+f(.9**(7-i))).join(' '))}
{const r=C.c2run(8,9,7,'ortho',1,0,1),r2=C.c2run(8,9,7,'ortho',1,1,1);console.log('ortho T=9 item0 additive c0',f(r.c[0]),'e0',f(r.e[0]),' delta c0',f(r2.c[0]),'e0',f(r2.e[0]),' last delta c8',f(r2.c[8]),'e8',f(r2.e[8]))}
console.log('\n## C3 footprint (floats/layer), W=4096 s=4 rho=0.1');
for(const L of [256,4096,32768,1048576]){const o=C.footprint(L,4096,4,0.1);console.log(L,JSON.stringify(o))}
console.log('\n## C4 hash stats');
for(const [n,M] of [[1e5,1e6],[1e6,1e6],[1e4,1e6],[1e7,1e6],[1e5,1e8],[100,1e3]]){const h=C.hashStats(n,M);const nv=C.nVis(n,M),sm=C.hashSim(nv,C.MV,7);
 console.log(`n=${n} M=${M}: load=${f(h.load,4)} occ=${Math.round(h.occ)} occFrac=${f(h.occFrac,4)} coll=${f(h.coll,4)} | toy nVis=${nv} simColl=${f(sm.coll,4)} simOcc=${sm.occ} closedColl(MV)=${f(C.hashStats(nv,C.MV).coll,4)}`)}
console.log('\n## C2 rms crosstalk vs sqrt((T-1)/d), additive random keys');
for(const [d,T] of [[8,16],[16,16],[32,16],[32,8]]){const r=C.c2run(d,T,7,'rand',1,0,1);const rms=Math.sqrt(r.e.reduce((a,b)=>a+b*b,0)/T);console.log(d,T,'rms_e',f(rms),'pred',f(Math.sqrt((T-1)/d)))}
console.log('\n## C2 "平均 20 次" (seeds 7-26, additive (1,0,1), random keys)');
for(const [d,T] of [[8,16],[32,16]]){let sm=0,sq=0;for(let s=7;s<=26;s++){const r=C.c2run(d,T,s,'rand',1,0,1);sm+=r.e.reduce((a,b)=>a+b,0)/T;sq+=r.e.reduce((a,b)=>a+b*b,0)/T}
 console.log(d,T,'mean_e',f(sm/20),'rms_e',f(Math.sqrt(sq/20)),'est',f(Math.sqrt((T-1)*(d-1))/d))}
