
/* ================= self test (design.md section 5) ================= */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t=1e-4)=>Math.abs(a-b)<=t+1e-15,nearv=(a,b,t=1e-4)=>a.length===b.length&&a.every((v,i)=>near(v,b[i],t));
  // C1
  {const v=c1Vals({lp:Math.log10(.002),G:32,k:8});ok(near(v.F,.7739)&&near(v.pk,.0159)&&near(v.nz,.0621)&&near(v.allw,93.8,.05),'C1 default')}
  {const v=c1Vals({lp:Math.log10(.002),G:128,k:32});ok(near(v.pk,.0621)&&near(v.nz,.2261),'C1 G=128')}
  {const v=c1Vals({lp:-2,G:32,k:32});ok(near(v.F,.2763)&&near(v.pk,.2750),'C1 p=0.01')}
  ok(Math.abs(passK(1e-4,1)/1e-4-1)<1e-9,'C1 expm1');ok(Math.abs(nonzeroG(.5,32)-(1-2*.5**32))<1e-12,'C1 p=0.5');ok(passK(1,8)===1&&nonzeroG(1,32)===0&&log1mPow(1,128)===0,'C1 p=1 guard');
  // C2
  {const o=computeC2(C2DEF);ok(near(o.pS,.6341)&&Math.abs(o.sumA)<1e-12&&o.diff<1e-12&&nearv(o.L,[.01333,-.04,0,.02667,0,0],1e-5),'C2 RLOO default')}
  {const s=Object.assign({},C2DEF,{est:'reinforce'}),o=computeC2(s);ok(near(o.sumA,.04,1e-12)&&near(o.Q[4],-.00945,1e-5)&&near(o.Q[5],-.00519,1e-5)&&near(o.diff,.00945,1e-5)&&/壓低/.test(c2Verdict(s,o))&&/5、6/.test(c2Verdict(s,o)),'C2 REINFORCE default')}
  {const s=Object.assign({},C2DEF,{est:'reinforce',R:C2DEF.R.map(r=>-r)}),o=computeC2(s);ok(near(o.Q[4],.00945,1e-5)&&near(o.Q[5],.00519,1e-5)&&/抬高/.test(c2Verdict(s,o)),'C2 flipped')}
  {const s={S:[1,0,1,1,1,0],z:[0,2,3,4],R:[.03,-.02,.01,.015],est:'reinforce'},o=computeC2(s),pi0=o.pi0;ok(o.L.every((l,j)=>Math.abs(l-o.Q[j]-o.sumA*(pi0[j]-o.pic[j]))<1e-15)&&/2、6/.test(c2Verdict(s,o)),'C2 identity, custom S')}
  {const s={S:[1,1,1,1,1,1],z:[0,1,1,3],R:C2DEF.R,est:'reinforce'};ok(/沒有被拒絕/.test(c2Verdict(s,computeC2(s))),'C2 no rejected')}
  // C3
  {const o=computeC3a(C3DEF,[.010,.014],.01);ok(nearv(o.Rd,[.015625,-.003906,0,.023438],1e-6)&&near(o.Rhat,.008789,1e-6)&&near(o.MA,.010930,1e-6)&&o.promote&&o.ds===3&&o.js===1&&Math.abs(o.A.reduce((a,b)=>a+b,0))<1e-15,'C3a default')}
  {const o=computeC3a(C3DEF,[.002,.006],.01);ok(near(o.MA,.005596,1e-6)&&!o.promote,'C3a no promotion')}
  {const a=computeC3b(0,.02,.005,4,3),b=computeC3b(0,.02,0,4,3),c=computeC3b(0,.02,0,1,1),d=computeC3b(.01,.02,.005,4,3);
   ok(Math.abs(a.P/7.153e-3-1)<1e-3&&near(a.E,1.416,2e-3),'C3b shared');ok(Math.abs(b.P/2.660e-4-1)<1e-3&&near(b.E,.053,1e-3),'C3b lower bound');ok(Math.abs(c.P/.15866-1)<1e-4&&near(c.E,31.41,.01),'C3b r=1 m=1');ok(near(d.P,.5,1e-7),'C3b mu=tau')}
  // C4
  ok(RES.MATH.length===13&&RES.HARP.length===12&&RES.OLYMP.length===12,'C4 row counts');
  {const t=(ds,ki,n,dp,ra,rc,ov)=>{const o=deriveC4(ds,ki,n);ok(o.s.dpp===dp&&o.s.ratio===ra&&o.s.rec===rc&&(ov===undefined||o.overlap===ov),`C4 ${ds} k${KS[ki]} ${n}: ${o.s.dpp} ${o.s.ratio} ${o.s.rec}`)};
   t('MATH',4,'SOAR-PQ (Ours)','+9.3','1.97×','0.75',false);t('MATH',0,'SOAR-PQ (Ours)','+1.2','3.40×','0.55');t('HARP',4,'SOAR-PQ (Ours)','+4.1','1.50×','0.51');
   t('OLYMP',4,'SOAR-PQ (HARP) (Ours)','+5.1','1.74×','0.82',true);t('MATH',4,'Hard-Only (g = 128)','+2.8','1.29×','0.23',true)}
  ok(!!rowOf('OLYMP',DEFROW.OLYMP)&&!!rowOf('MATH',DEFROW.MATH)&&!!rowOf('HARP',DEFROW.HARP),'C4 default rows exist');
  // F1 + ratio SD
  {const ss=1-Math.pow(32,-1/31);ok(near(ss,.1058)&&near((1-ss)-log1mPow(ss,32),.8663),'F1 peak')}
  {const r=(m,s,h,hs)=>[m/h,m/h*Math.hypot(s/m,hs/h)];const a=r(1.7,1,.5,.1),b=r(.7,.3,.4,.1);ok(near(a[0],3.40,.005)&&near(a[1],2.11,.005)&&near(b[0],1.75,.005)&&near(b[1],.87,.005),'pass@1 ratio SDs')}
  ok(Math.abs(erfcc(1)-0.157299207)<5e-8&&Math.abs(erfcc(-1)-1.842700793)<5e-8,'erfcc');
  // DOM
  ok($$('details.check').length===11,'check boxes: '+$$('details.check').length);
  ok(['c1','c2','c3','c4','f0','f1','f2'].every(id=>!!$('#'+id)),'plates/figures present');
  ok($$('.algo .row').length===30,'Alg. 1 rows: '+$$('.algo .row').length);
  {const tl=JSON.parse($('#timeline-data').textContent);ok(tl.length===15&&tl.every(e=>TL_GROUP[tlKey(e)]),'timeline groups')}
  ok(!!document.querySelector('.katex')||typeof katex==='undefined','KaTeX rendered');
  ok(!document.querySelector('.katex-error'),'no KaTeX errors');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ================= boot ================= */
initTheme();initToc();initC1();initC3();initC2();initC4();drawF0();drawF1();drawF2();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{c1Draw();c2Draw();c3aDraw();if(!$('#c3b').hidden)c3bDraw();c4Draw();drawF0();drawF1();drawF2()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg','option'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))addEventListener('load',()=>setTimeout(selfTest,50));
})();
