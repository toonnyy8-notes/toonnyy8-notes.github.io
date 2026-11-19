
/* ---------- self test (?test) ---------- */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t)=>Math.abs(a-b)<=t+1e-12;
  /* C1 */
  const cc=cellCounts(ROWS);ok(ROWS.length===47,'C1 47 rows');
  [['I|on|S',10],['I|on|L',19],['I|off|L',1],['E|off|L',10],['E|on|L',7]].forEach(([k,v])=>ok(cc[k]===v,`C1 cell ${k}=${cc[k]}`));
  ['I|off|S','E|on|S','E|off|S'].forEach(k=>ok(!cc[k],'C1 zero '+k));
  ok(ROWS.filter(r=>r[3]==='I').length===30&&ROWS.filter(r=>r[3]==='E').length===17,'C1 I30 E17');
  const rc=x=>ROWS.filter(r=>r[8].some(y=>y[0]===x)).length;
  [['opt',4],['st',8],['sig',4],['adm',3],['obj',4]].forEach(([x,v])=>ok(rc(x)===v,`C1 rule ${x}=${rc(x)}`));
  ok(ROWS.filter(r=>r[8].some(y=>y[0]==='st'&&y[1]==='T2')).length===5,'C1 st T2=5');
  const yc=y=>ROWS.filter(c1filter({rep:'all',upd:'all',per:'all',rules:new Set(),yr:y})).length;ok(yc('le2023')===10,'C1 le2023');ok(yc('2025p')===28,'C1 2025p');
  ok($$('#c1-bnd article').length===8,'C1 8 boundary cards');
  const pills=()=>$$('#c1-map .pill').length;
  c1step(2);ok(pills()===10&&ROWS.filter(c1filter(c1)).every(r=>r[3]==='I'&&r[4]==='on'),'C1 step2 short=10 I/on');
  c1step(3);ok(pills()===4,'C1 step3 opt=4');c1step(4);ok(pills()===10,'C1 step4 le2023=10');
  c1step(1);ok(pills()===47,'C1 step1 all');c1reset();renderC1(false);$('#c1-adv').open=false;
  /* C2 */
  const c2t=[[8,16,'rand',[1,0,1],[.806,1.054,.539,1.437,.916,3.591]],[8,16,'rand',[1,1,1],[.377,.605,.321,1,0,2.471]],[8,16,'rand',[.9,0,1],[.425,.606,-.005,1.207,.410,2.089]],[8,16,'rand',[.9,.9,1],[.263,.439,.074,1,0,1.822]],
    [32,16,'rand',[1,0,1],[1.027,.674,1.037,1.084,.707,4.053]],[32,16,'rand',[1,1,1],[.809,.424,.651,1,0,3.577]],[8,12,'ortho',[1,0,1],[1.143,.648,1.096,1.272,.962,3.704]],[8,12,'ortho',[1,1,1],[.738,.324,.096,1,0,2.828]]];
  c2t.forEach(([d,T,m,p,v])=>{const r=c2run(d,T,7,m,...p),got=[r.c.reduce((a,b)=>a+b)/T,r.e.reduce((a,b)=>a+b)/T,r.c[0],r.c[T-1],r.e[T-1],r.fro];
    ok(got.every((g,i)=>near(g,v[i],.0005)),`C2 d=${d} T=${T} ${m} ${p}: ${got.map(x=>x.toFixed(3)).join(',')}`)});
  {const r=c2run(8,8,7,'ortho',.9,0,1);ok(r.c.every((c,i)=>near(c,.9**(7-i),1e-9))&&near(r.c[0],.478,.0005),'C2 ortho decay c_i=0.9^(T-i)')}
  ok(c2run(8,12,7,'ortho',1,1,1).over.filter(Boolean).length===4,'C2 ortho T=12: 4 overwritten');
  {const S=c2state(8,16,7,'rand',1,0,1);let f=0;for(const v of S)f+=v*v;ok(near(Math.sqrt(f),c2run(8,16,7,'rand',1,0,1).fro,1e-12),'C2 heat map state = c2run state')}
  {const a=c2avg(8,16),b=c2avg(32,16);ok(near(a.me,1.207,.0005)&&near(a.rms,1.264,.0005),`C2 avg20 d=8: ${a.me.toFixed(3)} ${a.rms.toFixed(3)}`);ok(near(b.me,.660,.0005)&&near(b.rms,.672,.0005),`C2 avg20 d=32: ${b.me.toFixed(3)} ${b.rms.toFixed(3)}`)}
  /* C3 */
  const exp3={256:[524288,524288,131072,131072,16384,1179648,184320],4096:[8388608,8388608,2097152,131072,16384,1703936,970752],32768:[67108864,8396800,16777216,131072,16384,2097152,6842368],1048576:[2147483648,8396800,536870912,131072,16384,2752512,214880256]};
  for(const L in exp3){const o=footprint(+L,4096,4,.1),ks=['kv','win','mla','lin','ssm','loglin','ham'];ok(ks.every((k,i)=>o[k]===exp3[L][i]),'C3 L='+L)}
  /* C4 */
  const c4t=[[1e5,1e6,.1,95163,.0952,102,.1078,96],[1e6,1e6,1,632121,.6321,1024,.6240,651],[1e4,1e6,.01,9950,.0099,10,0,10],[1e7,1e6,10,999955,1,4096,.9805,1002],[1e5,1e8,.001,99950,.001,1,0,1]];
  c4t.forEach(([n,M,load,occ,coll,nv,sc,so])=>{const h=hashStats(n,M),v=nVis(n,M),s=hashSim(v,MV,7);
    ok(near(h.load,load,1e-4)&&Math.round(h.occ)===occ&&near(h.coll,coll,1e-4),`C4 closed n=${n} M=${M}`);
    ok(v===nv&&s.coll.toFixed(4)===sc.toFixed(4)&&s.occ===so,`C4 toy n=${n} M=${M}: nv=${v} coll=${s.coll.toFixed(4)} occ=${s.occ}`)});
  /* page */
  for(let i=1;i<=12;i++)ok($$(`[data-lo="LO${i}"]`).length>0,'LO'+i+' has a [data-lo] paragraph');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ---------- boot ---------- */
initTheme();initToc();initC1();renderS4();initC2();initC3();initC4();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderC1(false);renderS4();renderC2();renderC3();renderC4()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))selfTest();
})();
</script>
</body>
</html>
