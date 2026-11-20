
/* ================= timeline groups (design.md 6.7) ================= */
const TL_GROUP={'1011.0686':['opd','on-policy-distillation-imitation'],'2306.08543':['opd','on-policy-distillation-imitation'],'1503.02531':['kd','knowledge-distillation-foundations'],
 '2209.15189':['ctx','context-distillation-self-distillation'],'2402.03300':['rl','rlvr-grpo-credit-assignment'],'2412.06769':['lr','lopd-latent-reasoning-continuous-tokens'],
 '2509.24704':['lm','lopd-latent-memory-for-agents'],'2601.18734':['opsd','lopd-opsd-self-teacher-privileged-context'],'2601.19897':['opsd','lopd-opsd-self-teacher-privileged-context'],
 '2601.20802':['opsd','sdpo-lineage-overview'],'2604.10674':['sub','lopd-privileged-context-substrates'],'2606.11709':['fail','lopd-opd-failure-modes-and-stabilization'],
 '2608.13040':['me','lopd-method-learnable-context-and-margin']};
const TL_NAME={opd:'on-policy 模仿／蒸餾',kd:'蒸餾基礎',ctx:'context distillation',rl:'RLVR',lr:'latent 推理',lm:'latent 記憶',opsd:'OPSD',sub:'特權脈絡',fail:'失敗模式',me:'本文',other:'其他'};

/* ================= self test (design.md 6.3 and section 5) ================= */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t=1e-4)=>Math.abs(a-b)<=t+1e-12,nearV=(a,b,t=1e-4)=>a.every((v,i)=>near(v,b[i],t));
  // C1 (section 5 table)
  ok(nearV(c1(1,3).pT,[.4813,.2163,.1187,.0720,.0483,.0293,.0196,.0145]),'C1 pT');ok(nearV(c1(1,3).pS,[.0894,.3280,.0444,.1990,.1092,.0269,.1629,.0402]),'C1 pS s=1');
  [[1,1,.6173,.6705,.3620,.9106,.5187],[1,2,.6173,.6705,.3682,.5826,.3024],[1,3,.6173,.6705,.5210,.5382,.1837],[1,5,.6173,.6705,.5302,.2300,.0634],[1,7,.6173,.6705,.6173,.0402,.0145],[1,8,.6173,.6705,.6173,0,0],
   [.5,1,.1733,.1697,.1183,.7552,.5187],[.5,3,.1733,.1697,.1583,.3552,.1837],[.5,5,.1733,.1697,.1588,.1284,.0634]].forEach(([s,M,rv,fw,tr,ts,tt])=>{const o=c1(s,M);ok(near(o.rev,rv)&&near(o.fwd,fw)&&near(o.revTr,tr)&&near(o.tailS,ts)&&near(o.tailT,tt),`C1 s=${s} M=${M}`)});
  ok([1,2,3,5,8].every(M=>{const o=c1(0,M);return near(o.rev,0)&&near(o.revTr,0)}),'C1 s=0 all zero');
  [[1,.3620,1.3361],[3,.5210,.8230],[5,.5302,.6080]].forEach(([M,tr,rv])=>{const o=c1(1,M,true);ok(near(o.revTr,tr)&&near(o.rev,rv),'C1 perm M='+M)});
  ok(Math.abs(c1(1,3,true).revTr-c1(1,3).revTr)<1e-12,'C1 perm invariance');ok(Math.abs(c1(.5,8).revTr-c1(.5,8).rev)<1e-12,'C1 M=8 equals full');
  {let good=true;for(let s=0;s<=1;s+=.25)for(let M=1;M<=8;M++){const o=c1(s,M);if(o.revTr>o.rev+1e-12)good=false}ok(good,'C1 truncated <= full')}
  ok(fmt(-1e-17,4)==='0.0000','fmt no negative zero');
  // C2
  {const D=[.8,.3,.6,.1];const o=c2(1,D);ok(nearV(o.q,[.55,.15,.25,.05])&&nearV(o.d,[.3185,-.6931,.2231,-.6931])&&near(o.Ed,-.1052)&&near(o.K,.1052)&&near(o.Qb,.08)&&near(o.c,.224)&&near(o.first,-.0084)&&near(o.cov,.2324),'C2 s=1 default');
   ok(Math.abs(o.Ed+o.K)<1e-12&&Math.abs(o.c-(o.first+o.cov))<1e-12,'C2 identities');
   const h=c2(.5,D);ok(nearV(h.q,[.4808,.2175,.2292,.0725])&&near(h.Ed,-.0278)&&near(h.c,.114)&&near(h.first,-.0022)&&near(h.cov,.1162),'C2 s=0.5');
   const z=c2(0,D);ok(near(z.Ed,0)&&near(z.c,0)&&near(z.cov,0),'C2 s=0');
   [[.5,0,0],[.9,.8,-.0842],[.1,-.8,.0842]].forEach(([r,qb,c])=>{const o2=c2(1,[r,r,r,r]);ok(near(o2.Qb,qb)&&near(o2.c,c)&&near(o2.cov,0),'C2 constant r='+r)})}
  // C3
  const C3T=[['A','fixed',0,0,[[25,[.4008,.3009,.2014,.0969],.0001,.0030,0],[600,[.4,.3,.2,.1],0,0,0]],[.4,.3,.2,.1]],
   ['A','fixed',.5,0,[[25,[.5002,.2309,.2119,.0571],.0337,.1322,.5],[600,[.5,.2308,.2115,.0577],.0332,.1313,.5]],[.5,.2308,.2115,.0577]],
   ['A','fixed',1,0,[[25,[.5899,.1634,.2193,.0274],.1380,.2735,1],[600,[.5926,.1667,.2222,.0185],.1667,.3040,1]],[.5926,.1667,.2222,.0185]],
   ['B','fixed',1,0,[[0,[.55,.15,.25,.05],.1052,0,1],[600,[.4,.3,.2,.1],0,0,1]],[.4,.3,.2,.1]],
   ['C','fixed',1,0,[[0,[.55,.15,.25,.05],.1052,.0549,1],[100,[.4427,.3315,.2204,.0054],.2020,.3280,1],[600,[.4441,.3330,.2219,.0010],.3690,.5006,1]],[.4444,.3333,.2222,0]],
   ['A','dual',0,0,[[600,[.4,.3,.2,.1],0,0,0]],null],
   ['A','dual',0,.05,[[25,[.4346,.2759,.2032,.0863],.0035,.0424,.1868],[600,[.4398,.2724,.2046,.0832],.0049,.0500,.1944]],[.4398,.2724,.2046,.0832]],
   ['A','dual',0,.2,[[25,[.5579,.1854,.2111,.0456],.0790,.2026,.8428],[600,[.5436,.2006,.2166,.0392],.0757,.2000,.7309]],null],
   ['C','dual',0,.05,[[25,[.4092,.3062,.2037,.0809],.0023,.0288,.2375],[600,[.4139,.3105,.2070,.0686],.0068,.0500,.2663]],null],
   ['C','dual',0,.2,[[25,[.4400,.3264,.2129,.0207],.0818,.1881,1.2705],[600,[.4362,.3272,.2181,.0185],.0907,.1999,.7777]],null]];
  C3T.forEach(([pr,mo,be,m,rows,qs])=>{const r=c3(pr,mo,be,m);ok(rows.every(([k,q,K,D,b])=>{const h=r.hist[k];return nearV(h.q,q)&&near(h.KL,K)&&near(h.D,D)&&near(h.b,b)})&&(!qs||nearV(r.qstar,qs)),`C3 ${pr} ${mo} β=${be} m=${m}`)});
  {const r=c3('B','dual',0,.05);ok(near(r.hist[25].b,.625)&&near(r.hist[100].b,2.5)&&near(r.hist[600].b,5)&&r.capped,'C3 B dual capped at 5')}
  {let good=true;['A','B','C'].forEach(pr=>[0,.05,.1,.2].forEach(m=>{const r=c3(pr,'dual',0,m);if(r.hist.some(h=>h.b>BMAX+1e-12))good=false;if(r.hist[600].q.some((v,i)=>Math.abs(v-r.hist[599].q[i])>=1e-5))good=false}));ok(good,'C3 beta<=5 and |q600-q599|<1e-5')}
  ok(c3('A','fixed',0,0).hist[600].KL<1e-6,'C3 beta=0 collapses');ok(c3('C','fixed',1,0).zero.length===1,'C3 q* zero flag');
  // C4
  const C4T=[[141,140,1,2.99],[59,58,1,4.35],[200,198,2,2.47],[144,143,1,2.56],[297,291,6,2.11],[441,434,7,1.67],[149,139,10,2.99],[60,60,0,4.35],[209,198,11,2.47],[148,146,2,2.32],[277,276,1,2.28],[425,422,3,1.77]];
  {const a=c4(ROWS[0]),b=c4(ROWS[1]);ok(a.na===219&&a.nb===202&&near(a.se,1.58,.01)&&b.na===239&&b.nb===232&&near(b.se,1.62,.01),'C4 BFCL rows')}
  C4T.forEach(([na,nb,dn,se],i)=>{const o=c4(ROWS[i+2]);ok(o.na===na&&o.nb===nb&&o.dn===dn&&near(o.se,se,.01),'C4 '+ROWS[i+2][0])});
  ok(ROWS.every(r=>Math.abs(r[4]*r[3]/100-Math.round(r[4]*r[3]/100))<=.05&&Math.abs(r[6]*r[3]/100-Math.round(r[6]*r[3]/100))<=.05),'C4 near-integer counts');
  ok(!/之內|之外/.test($('#c4').textContent),'C4 has no inside/outside verdict');
  // DOM and layout
  ok($$('details.check').length===12,'check boxes: '+$$('details.check').length);
  ok(['c1','c2','c3l','c3','c4'].every(id=>!!$('#'+id)&&$$('#'+id+' [aria-live]').length===1),'plates with one live region each');
  {const tl=JSON.parse($('#timeline-data').textContent);ok(tl.length===13&&tl.every(e=>TL_GROUP[e.arxiv]),'timeline groups')}
  ok(!!document.querySelector('.katex')||typeof katex==='undefined','KaTeX rendered');
  ok(!document.querySelector('.katex-error'),'no KaTeX errors');
  ok($$('.eq-m').every(n=>n.getBoundingClientRect().right<=document.documentElement.clientWidth+1),'equations stay inside viewport');
  {const small=$$('svg text').filter(t=>t.getBoundingClientRect().width>0&&parseFloat(getComputedStyle(t).fontSize)<12);ok(!small.length,'svg text >= 12px ('+small.length+' smaller)')}
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ================= boot ================= */
document.body.insertAdjacentHTML('afterbegin','<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs><pattern id="hatch-s" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" style="fill:var(--surface)"/><line x1="0" y1="0" x2="0" y2="5" style="stroke:var(--c-perp)" stroke-width="2.2"/></pattern></defs></svg>');
initTheme();initToc();fillTables();drawFigs();initC1();initC3lite();initC2();initC3();initC4();renderTimeline();observeWidths();
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg','option'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))addEventListener('load',()=>setTimeout(selfTest,300));
})();
