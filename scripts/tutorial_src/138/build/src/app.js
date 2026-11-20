(()=>{
'use strict';
/*@COMMON@*/
const VSRC=$('#w-vajda').textContent;
const VJ=new Function(VSRC+';return{FV,gb,Lf,table,scatter,mulberry32,pairs}')();
const fx=v=>!Number.isFinite(v)?'∞':(Math.abs(v)>=1e4?v.toExponential(2):f4(v));
const yLog=(lo,hi,r0,r1)=>{const s=lin(lo,hi,r0,r1);return v=>s(Math.log10(Math.min(Math.max(v,10**lo),10**hi)))};
function hatchDefs(svg,id){const d=el('defs',null,svg),p=el('pattern',{id,patternUnits:'userSpaceOnUse',width:6,height:6,patternTransform:'rotate(45)'},d);el('line',{x1:0,y1:0,x2:0,y2:6,class:'hatch-ln'},p)}
function sizeSvg(svg,W,H){svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H)}

/* ================= shared math (mirrors verify/check.py) ================= */
const FN={TV:r=>.5*Math.abs(r-1),TV2:r=>Math.abs(r-1),chi2:r=>(r-1)**2,H:r=>(Math.sqrt(r)-1)**2,
  KLabs:r=>r>0?Math.abs(Math.log(r)):Infinity,KLnn:r=>r>0?r-1-Math.log(r):Infinity,KLneg:r=>r>0?-Math.log(r):Infinity};
function bisect(g,lo,hi,it=100){let glo=g(lo);for(let i=0;i<it;i++){const m=.5*(lo+hi),gm=g(m);if((gm<=0)===(glo<=0)){lo=m;glo=gm}else hi=m}return .5*(lo+hi)}
function region(D,delta,rmax){const g=r=>D(r)-delta;let lo,hi,lc,hc;if(g(rmax)<=0){hi=rmax;hc=true}else{hi=bisect(g,1,rmax);hc=false}
  if(g(1e-12)<=0){lo=0;lc=true}else{lo=bisect(g,1e-12,1);lc=false}return{lo,hi,lc,hc}}
function dhat(fn,est,mu){const f=FN[fn];if(est==='MC')return f;if(est==='Atom')return r=>mu*f(r);
  return r=>{const q=(1-mu*r)/(1-mu);if(q<0)return Infinity;return mu*f(r)+(1-mu)*f(q===0?1e-300:q)}}
const CAL={page:{TV:['TV2','TV2','TV2',.2],chi2:['chi2','chi2','chi2',.04],H:['H','H','H',FN.H(.8)],KL:['KLabs','KLabs','KLneg',-Math.log(.8)]},
  literal:{TV:['TV','TV','TV',.2],chi2:['chi2','chi2','chi2',.04],H:['H','H','H',.01],KL:['KLneg','KLneg','KLneg',.22]}};
const ESTS=['MC','Atom','Bin'],DIVS=['TV','chi2','H','KL'],DIVN={TV:'TV',chi2:'χ²',H:'Hellinger',KL:'KL'},ESTN={MC:'MC',Atom:'Atomic',Bin:'Binary'};
function cell(cal,div,est,mu){const c=CAL[cal][div],fn=c[ESTS.indexOf(est)];return region(dhat(fn,est,mu),c[3],1/mu)}
function assertMonotone(cal,div,mu){const c=CAL[cal][div];let ok=true;ESTS.forEach((est,i)=>{const D=dhat(c[i],est,mu);let p=-Infinity;
  for(let k=0;k<=200;k++){const y=D(1+(1/mu-1)*k/200);if(Number.isFinite(y)&&y<p-1e-12&&c[i]!=='KLneg')ok=false;if(Number.isFinite(y))p=y}
  p=Infinity;for(let k=0;k<=200;k++){const y=D(1e-6+(1-1e-6)*k/200);if(Number.isFinite(p)&&y>p+1e-12)ok=false;p=y}});
  if(!ok)console.error('C4: non-monotone Dhat',cal,div,mu);return ok}

/* ================= C1 PPO-Clip ================= */
const c1={s:1,e:20,r:100};
function c1Val(s,R,E){const cl=s>0?R>100+E:R<100-E,c=cl?(s>0?100+E:100-E):R;return{L:s*c/100,g:cl?0:s,cl}}
function c1Draw(){const box=$('#c1-fig'),svg=$('svg',box),W=widthOf(box,280,640),H=250,L=44,R=12,T=12,B=32;sizeSvg(svg,W,H);hatchDefs(svg,'hc1');
  const{s,e,r}=c1,x=lin(0,2,L,W-R),y=s>0?lin(-.1,2.1,H-B,T):lin(-2.1,.1,H-B,T);
  for(let v=0;v<=2;v+=.5){el('line',{x1:x(v),x2:x(v),y1:T,y2:H-B,class:'grid'},svg);tx(svg,x(v),H-B+18,v.toFixed(1),{class:'tk','text-anchor':'middle'})}
  const ys=s>0?[0,1,2]:[-2,-1,0];ys.forEach(v=>{el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-6,y(v)+4,String(v),{class:'tk','text-anchor':'end'})});
  const a=s>0?x((100+e)/100):x(0),b=s>0?x(2):x((100-e)/100);el('rect',{x:a,y:T,width:b-a,height:H-B-T,fill:'url(#hc1)',opacity:.6},svg);
  tx(svg,(a+b)/2,T+16,'梯度為 0',{class:'al halo','text-anchor':'middle'});
  [100-e,100+e].forEach(v=>el('line',{x1:x(v/100),x2:x(v/100),y1:T,y2:H-B,class:'cur'},svg));
  el('path',{d:lp([[x(0),y(0)],[x(2),y(2*s)]]),class:'ref'},svg);
  const pts=[];for(let R2=0;R2<=200;R2++)pts.push([x(R2/100),y(c1Val(s,R2,e).L)]);el('path',{d:lp(pts),class:'tot'},svg);
  const v=c1Val(s,r,e);el('circle',{cx:x(r/100),cy:y(v.L),r:5,class:'kdot'},svg);
  tx(svg,W-R,H-6,'r',{class:'al','text-anchor':'end'});tx(svg,L,T-2,'L(r)',{class:'al'});
  svg.setAttribute('aria-label',`PPO-Clip：Â ${s>0?'為正':'為負'}，ε = ${(e/100).toFixed(2)}，r = ${(r/100).toFixed(2)}，梯度 ${v.g}`);
  $('#c1-epso').textContent=(e/100).toFixed(2);$('#c1-ro').textContent=(r/100).toFixed(2);
  const bound=v.cl?(s>0?'上界 1 + ε':'下界 1 − ε'):'無';
  $('#c1-read').innerHTML=`<dt>L(r)</dt><dd>${v.L.toFixed(2)}</dd><dt>∂L/∂r</dt><dd>${v.g}</dd><dt>作用中的界</dt><dd>${bound}</dd>`;
  announce('c1',`r = ${(r/100).toFixed(2)}，${v.cl?'被剪裁，梯度為 0':'未剪裁，梯度 '+v.g}`)}
function initC1(){segInit($('#c1-sign'),[1,-1],1,v=>v>0?'Â > 0':'Â < 0',v=>{c1.s=v;c1Draw()});
  $('#c1-eps').addEventListener('input',e=>{c1.e=+e.target.value;requestAnimationFrame(c1Draw)});
  $('#c1-r').addEventListener('input',e=>{c1.r=+e.target.value;requestAnimationFrame(c1Draw)});c1Draw()}

/* ================= C2 f curves and MC interval ================= */
const c2={rmin:80,kl:'KLabs'};
const KLN={KLabs:'KL |log r|',KLnn:'KL r−1−log r',KLneg:'KL −log r'};
function mcUpper(fn,rminPct){const rm=rminPct/100,d=FN[fn](rm);
  if(fn==='TV2'||fn==='TV')return 2-rm;if(fn==='chi2')return 1+Math.sqrt(d);if(fn==='H')return(1+Math.sqrt(d))**2;
  if(fn==='KLabs')return 1/rm;if(fn==='KLnn')return bisect(r=>FN.KLnn(r)-d,1,50);return Infinity}
function c2Rows(){return[['TV2','TV2 = |r−1|','c-tv'],['chi2','χ²','c-chi'],['H','Hellinger','c-h'],[c2.kl,KLN[c2.kl],'c-kl']]}
function c2Draw(){const box=$('#c2-fig'),svg=$('svg',box),W=widthOf(box,280,640),H=250,L=36,R=12,T=12,B=32;sizeSvg(svg,W,H);
  const x=lin(.3,2,L,W-R),y=lin(0,3,H-B,T),rm=c2.rmin/100;const cp=el('clipPath',{id:'c2clip'},el('defs',null,svg));el('rect',{x:L,y:T,width:W-R-L,height:H-B-T},cp);
  [.5,1,1.5,2].forEach(v=>{el('line',{x1:x(v),x2:x(v),y1:T,y2:H-B,class:'grid'},svg);tx(svg,x(v),H-B+18,v.toFixed(1),{class:'tk','text-anchor':'middle'})});
  [0,1,2,3].forEach(v=>tx(svg,L-6,y(v)+4,String(v),{class:'tk','text-anchor':'end'}));
  el('line',{x1:L,x2:W-R,y1:y(1),y2:y(1),class:'ink'},svg);tx(svg,W-R,y(1)-6,'門檻 f/δ = 1',{class:'al halo','text-anchor':'end'});
  el('line',{x1:x(rm),x2:x(rm),y1:T,y2:H-B,class:'cur'},svg);
  const g=el('g',{'clip-path':'url(#c2clip)'},svg);
  const labs=[];c2Rows().forEach(([fn,nm,cls])=>{const d=FN[fn](rm),pts=[];for(let k=0;k<=340;k++){const r=.3+k*1.7/340;let v=FN[fn](r)/d;v=Math.max(-.2,Math.min(3.3,v));pts.push([x(r),y(v)])}
    el('path',{d:lp(pts),class:cls},g);const yr=Math.min(3,Math.max(.1,FN[fn](2)/d));labs.push([y(yr)+4,nm])});
  labs.sort((p,q)=>p[0]-q[0]);let prev=-1e9;labs.forEach(l=>{l[0]=Math.max(l[0],prev+17,T+14);prev=l[0]});   // stack right-end labels without overlap
  labs.forEach(([yy,nm])=>tx(svg,W-R-2,yy,nm,{class:'al halo','text-anchor':'end'}));
  if(c2.kl==='KLneg')tx(svg,x(1.6),y(0)-8,'−log r：r > 1 時為負',{class:'tk halo','text-anchor':'middle'});
  tx(svg,W-R,H-6,'r',{class:'al','text-anchor':'end'});
  const sc=v=>Math.max(0,Math.min(100,(v-.3)/1.7*100));let ivl='',tb='';const ups={};
  c2Rows().forEach(([fn,nm])=>{const hi=mcUpper(fn,c2.rmin),d=FN[fn](rm);ups[fn]=hi;const open=!Number.isFinite(hi)||hi>2;
    ivl+=`<span class="nm">${nm}</span><span class="tr"><span class="bd${open?' open':''}" style="left:${sc(rm)}%;right:${100-sc(Number.isFinite(hi)?hi:2)}%"></span></span><span class="tx">[${f4(rm)}, ${Number.isFinite(hi)?f4(hi):'∞（無上界）'}]</span>`;
    tb+=`<tr><td>${nm}</td><td class="num">${d.toFixed(6)}</td><td class="num">${f4(rm)}</td><td class="num">${Number.isFinite(hi)?f4(hi):'∞'}</td></tr>`});
  $('#c2-ivl').innerHTML=ivl;$('#c2-tbl tbody').innerHTML=tb;$('#c2-rmino').textContent=rm.toFixed(2);
  const same=Math.round(ups.TV2*1e4)===Math.round(ups.chi2*1e4);$('#c2-same').textContent=same?'TV 與 χ² 的區間相同。':'';
  svg.setAttribute('aria-label',`f/δ 曲線，r_min = ${rm.toFixed(2)}`);
  announce('c2',`r_min = ${rm.toFixed(2)}；上界 TV ${f4(ups.TV2)}、χ² ${f4(ups.chi2)}、Hellinger ${f4(ups.H)}、${KLN[c2.kl]} ${Number.isFinite(ups[c2.kl])?f4(ups[c2.kl]):'無上界'}`)}
function initC2(){segInit($('#c2-kl'),['KLabs','KLnn','KLneg'],c2.kl,k=>({KLabs:'|log r|',KLnn:'r − 1 − log r',KLneg:'−log r'})[k],k=>{c2.kl=k;c2Draw()});
  $('#c2-rmin').addEventListener('input',e=>{c2.rmin=+e.target.value;requestAnimationFrame(c2Draw)});c2Draw()}

/* ================= C3 Vajda (worker) ================= */
const c3={f:'chi2',K:3,seed:1,V:30,B:50,tab:null,sc:null,latest:0,applied:[]};
const TEST=/[?&]test\b/.test(location.search);
let WK=null;try{WK=new Worker(URL.createObjectURL(new Blob([VSRC],{type:'text/javascript'})))}catch(e){WK=null}
const WK_OK=!!WK;
// ?test: headless --virtual-time-budget does not wait for worker threads, so the self test runs the same source on the main thread
if(TEST&&WK){WK.terminate();WK=null}
let reqSeq=0;const pend=new Map();
function runLocal(m){setTimeout(()=>{const res=m.type==='table'?VJ.table():VJ.scatter(m.f,m.K,m.seed);onW({reqId:m.reqId,type:m.type,res})},0)}
function onW(d){const p=pend.get(d.reqId);pend.delete(d.reqId);if(p)p.res(d)}
if(WK){WK.onmessage=e=>onW(e.data);WK.onerror=()=>{WK=null;pend.forEach(p=>runLocal(p.msg))}}
function wreq(msg){msg.reqId=++reqSeq;return new Promise(res=>{pend.set(msg.reqId,{res,msg});if(WK)WK.postMessage(msg);else runLocal(msg)})}
let tableP=null;function c3Table(){if(!tableP)tableP=wreq({type:'table'}).then(d=>{c3.tab=d.res;c3Draw();return d.res});return tableP}
function c3Scatter(){$('#c3-prog').textContent='計算中…';const p=wreq({type:'scatter',f:c3.f,K:c3.K,seed:c3.seed});c3.latest=reqSeq;const id=reqSeq;
  return p.then(d=>{if(d.reqId!==c3.latest)return null;c3.sc=d.res;c3.applied.push(id);$('#c3-prog').textContent='';c3Draw();c3Announce();return d})}
function c3Draw(){const box=$('#c3-fig'),svg=$('svg',box),W=widthOf(box,280,640),H=290,L=44,R=12,T=24,B=32;sizeSvg(svg,W,H);
  const x=lin(0,1,L,W-R),y=yLog(-4,4,H-B,T),f=VJ.FV[c3.f],V=c3.V/100,b=c3.B/100;
  for(let e=-4;e<=4;e++){el('line',{x1:L,x2:W-R,y1:y(10**e),y2:y(10**e),class:'grid'},svg);if(e%2===0)tx(svg,L-6,y(10**e)+4,'1e'+e,{class:'tk','text-anchor':'end'})}
  [0,.25,.5,.75,1].forEach(v=>tx(svg,x(v),H-B+18,String(v),{class:'tk','text-anchor':'middle'}));
  if(c3.sc){const P=c3.sc.pts;for(let n=0;n<P.length;n+=2){const D=P[n+1];if(!Number.isFinite(D))continue;
    if(D<1e-4||D>1e4){const yy=D<1e-4?H-B-2:T+2,xx=x(P[n]),s=D<1e-4?-1:1;el('path',{d:`M${xx-3} ${yy+3*s}L${xx+3} ${yy+3*s}L${xx} ${yy-3*s}Z`,class:'sc-tri'},svg)}
    else el('circle',{cx:x(P[n]),cy:y(D),r:2.2,class:'sc'},svg)}}
  if(c3.tab){const t=c3.tab[c3.f],pts=[],pq=[];for(let i=1;i<500;i++){const v=i/500;pts.push([x(v),y(t.L[i-1])]);pq.push([x(v),y(t.C*v*v)])}
    el('path',{d:lp(pq),class:'ref'},svg);el('path',{d:lp(pts),class:'tot'},svg)}
  const [lv]=VJ.Lf(f,V,1000),g=VJ.gb(f,V,b);el('line',{x1:x(V),x2:x(V),y1:y(lv),y2:y(g),class:'cur'},svg);
  el('path',{d:`M${x(V)} ${y(g)-6}L${x(V)+6} ${y(g)}L${x(V)} ${y(g)+6}L${x(V)-6} ${y(g)}Z`,class:'dia'},svg);
  tx(svg,W-R,H-6,'V = D_TV',{class:'al','text-anchor':'end'});tx(svg,L,T-10,'D_f（對數）',{class:'al'});
  $('#c3-vo').textContent=V.toFixed(2);$('#c3-bo').textContent=b.toFixed(2);
  const t=c3.tab&&c3.tab[c3.f];
  $('#c3-read').innerHTML=`<dt>C_f（格點）</dt><dd>${t?t.C.toFixed(3):'…'}</dd><dt>取到的 V</dt><dd>${t?t.Vs.toFixed(3):'…'}</dd><dt>L_f(V)</dt><dd>${lv.toPrecision(5)}</dd><dt>g_β(V)</dt><dd>${g.toPrecision(5)}</dd><dt>違反數</dt><dd>${c3.sc?c3.sc.viol+' / 400':'…'}</dd><dt>懲罰倍率 1/C_f</dt><dd>${t?(1/t.C).toFixed(3):'…'}</dd>`;
  svg.setAttribute('aria-label',`Vajda 下界，f = ${c3.f}，K = ${c3.K}，V = ${V.toFixed(2)}，β = ${b.toFixed(2)}`)}
function c3Announce(){const t=c3.tab&&c3.tab[c3.f];announce('c3',`f = ${c3.f}，C_f ≈ ${t?t.C.toFixed(3):'計算中'}，違反數 ${c3.sc?c3.sc.viol:'計算中'}`)}
function c3SyncB(){const bs=$('#c3-b'),mx=Math.floor((1-c3.V/100)*100)-1;bs.max=String(mx);if(c3.B>mx){c3.B=mx;bs.value=String(mx)}}
function initC3(){segInit($('#c3-f'),['TV','chi2','KL','H'],c3.f,k=>({TV:'TV',chi2:'χ²',KL:'KL',H:'Hellinger'})[k],k=>{c3.f=k;c3Draw();c3Scatter()});
  segInit($('#c3-k'),[2,3,6],c3.K,null,k=>{c3.K=k;c3Scatter()});
  $('#c3-seed').addEventListener('click',()=>{c3.seed++;c3Scatter()});
  $('#c3-v').addEventListener('input',e=>{c3.V=+e.target.value;c3SyncB();requestAnimationFrame(c3Draw)});
  $('#c3-b').addEventListener('input',e=>{c3.B=+e.target.value;requestAnimationFrame(c3Draw)});
  $('#c3-best').addEventListener('click',()=>{const bb=VJ.Lf(VJ.FV[c3.f],c3.V/100,1000)[1];c3SyncB();c3.B=Math.max(1,Math.min(+$('#c3-b').max,Math.round(bb*100)));$('#c3-b').value=String(c3.B);c3Draw()});
  c3SyncB();c3Draw();c3Table();c3Scatter()}

/* ================= C4 regions vs mu ================= */
const c4={lm:-100,div:'KL',cal:'page',dual:false,curves:new Map()};
const muOf=lm=>Math.min(10**(lm/100),.999);
const MUG=Array.from({length:300},(_,i)=>10**(-4+i*(4+Math.log10(.999))/299));
function c4Curves(){const key=`${c4.div}|${c4.cal}`;if(!c4.curves.has(key))c4.curves.set(key,ESTS.map(est=>MUG.map(mu=>cell(c4.cal,c4.div,est,mu))));return c4.curves.get(key)}
function c4Fmt(c){return`[${c.lc?'0*':(c.lo<1e-4?c.lo.toExponential(1):f4(c.lo))}, ${c.hc?'1/µ*':fx(c.hi)}]`}
function c4Draw(){const box=$('#c4-fig'),svg=$('svg',box),W=widthOf(box,280,640),H=290,L=44,R=12,T=24,B=32;sizeSvg(svg,W,H);hatchDefs(svg,'hc4');
  const x=lin(-4,0,L,W-R),y=yLog(-2,4,H-B,T),mu=muOf(c4.lm);
  for(let e=-2;e<=4;e++){el('line',{x1:L,x2:W-R,y1:y(10**e),y2:y(10**e),class:'grid'},svg);tx(svg,L-6,y(10**e)+4,e===0?'1':'1e'+e,{class:'tk','text-anchor':'end'})}
  for(let e=-4;e<=0;e++)tx(svg,x(e),H-B+18,e===0?'1':'1e'+e,{class:'tk','text-anchor':'middle'});
  el('path',{d:`M${x(-4)} ${y(1e4)}L${x(0)} ${y(1)}L${x(0)} ${y(1e4)}Z`,fill:'url(#hc4)',opacity:.6},svg);
  tx(svg,x(-.6),y(3e3),'π > 1，不可行',{class:'tk halo','text-anchor':'middle'});
  if(c4.dual)[.5,10].forEach(v=>{el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),class:'ref'},svg);tx(svg,W-R,y(v)-4,'dual clip（假設）'+v,{class:'tk halo','text-anchor':'end'})});
  const cv=c4Curves(),cls=['tot','par dash','perp dot'];
  cv.forEach((arr,k)=>{['lo','hi'].forEach(side=>{const pts=arr.map((c,i)=>[x(Math.log10(MUG[i])),y(side==='lo'?(c.lc?1e-2:c.lo):c.hi)]);el('path',{d:lp(pts),class:cls[k]},svg)});
    tx(svg,L+4,y(arr[0].hi)+(k===0?-6:12+k*2),ESTN[ESTS[k]],{class:'al halo'})});
  el('line',{x1:x(Math.log10(mu)),x2:x(Math.log10(mu)),y1:T,y2:H-B,class:'cur'},svg);
  tx(svg,W-R,H-6,'µ（對數）',{class:'al','text-anchor':'end'});tx(svg,L,T-10,'r（對數）',{class:'al'});
  $('#c4-muo').textContent=mu<.01?mu.toExponential(2):mu.toFixed(3);
  const NAMES={'TV|MC':'GRPO','KL|MC':'DAPO','TV|Bin':'DPPO','chi2|Atom':'RIPO'};let tb='';const mcs={};
  DIVS.forEach(div=>{tb+=`<tr class="${div===c4.div?'sel':''}"><td>${DIVN[div]}</td>`;ESTS.forEach(est=>{const c=cell(c4.cal,div,est,mu);if(est==='MC')mcs[div]=c;
    const nm=NAMES[`${div}|${est}`];let s=(nm?`<small><b>${nm}</b></small>`:'')+c4Fmt(c);if(c4.dual)s+=`<small>疊加後 [${f4(Math.max(c.lo,.5))}, ${f4(Math.min(c.hi,10))}]</small>`;if(est==='MC'&&(div==='TV'||div==='chi2'))s+='<small data-same>同一區間</small>';tb+=`<td class="num">${s}</td>`});tb+='</tr>'});
  $('#c4-tbl tbody').innerHTML=tb;const same=Math.abs(mcs.TV.lo-mcs.chi2.lo)<1e-9&&Math.abs(mcs.TV.hi-mcs.chi2.hi)<1e-9;$$('#c4-tbl [data-same]').forEach(n=>n.classList.toggle('hide',!same));
  const at=cell(c4.cal,c4.div,'Atom',mu);svg.setAttribute('aria-label',`剪裁區間隨 µ 變化，${DIVN[c4.div]}，µ = ${mu.toPrecision(3)}`);
  announce('c4',`µ = ${mu.toPrecision(3)}，${DIVN[c4.div]}–Atomic 區間 ${c4Fmt(at)}`)}
function initC4(){segInit($('#c4-div'),DIVS,c4.div,d=>DIVN[d],d=>{c4.div=d;assertMonotone(c4.cal,c4.div,muOf(c4.lm));c4Draw()});
  segInit($('#c4-cal'),['page','literal'],c4.cal,c=>c==='page'?'本頁校準（推定）':'論文字面值',c=>{c4.cal=c;assertMonotone(c4.cal,c4.div,muOf(c4.lm));c4Draw()});
  $('#c4-dual').addEventListener('change',e=>{c4.dual=e.target.checked;c4Draw()});
  $('#c4-mu').addEventListener('input',e=>{c4.lm=+e.target.value;requestAnimationFrame(c4Draw)});assertMonotone(c4.cal,c4.div,muOf(c4.lm));c4Draw()}

/* ================= C5 estimator expectations ================= */
const MU5=[.5,.25,.12,.08,.04,.01],F5={TV:FN.TV,chi2:FN.chi2,KL:FN.KLnn,H:FN.H};
const c5={f:'chi2',l:30};
function c5Calc(lam,fk){const f=F5[fk],w=MU5.map(m=>m**(1-lam)),Z=w.reduce((a,b)=>a+b,0),PI=w.map(v=>v/Z);
  const rows=MU5.map((m,i)=>{const p=PI[i],r=p/m,h=f(r),bi=m*f(r)+(1-m)*f((1-p)/(1-m));return{m,p,r,h,a:m*h,bi,vm:m*h*h,va:m**3*h*h}});
  const D=rows.reduce((s,o)=>s+o.m*o.h,0),EA=rows.reduce((s,o)=>s+o.m*o.a,0),EB=rows.reduce((s,o)=>s+o.m*o.bi,0),S2M=rows.reduce((s,o)=>s+o.vm,0),S2A=rows.reduce((s,o)=>s+o.va,0);
  return{rows,D,EA,EB,S2M,S2A}}
function c5Draw(){const lam=c5.l/100,q=c5Calc(lam,c5.f),box=$('#c5-fig'),svg=$('svg',box),W=widthOf(box,280,640),rowH=34,L=116,R=72,T=8,H=T+4*rowH+8;sizeSvg(svg,W,H);
  const items=[['真實 D_f',q.D,'bar-d'],['E[MC]',q.D,'bar-mc'],['E[Atomic]',q.EA,'bar-at'],['E[Binary]',q.EB,'bar-bi']],mx=Math.max(q.D,1e-12),x=lin(0,mx,L,W-R);
  items.forEach(([nm,v,cls],i)=>{const yy=T+i*rowH;tx(svg,L-8,yy+19,nm,{class:'al','text-anchor':'end'});el('rect',{x:L,y:yy+6,width:Math.max(0,x(v)-L),height:18,rx:3,class:cls},svg);tx(svg,x(v)+6,yy+19,v.toFixed(5),{class:'labm'})});
  $('#c5-lo').textContent=lam.toFixed(2);
  const rat=v=>q.D>0?(v/q.D).toFixed(3):'—';
  $('#c5-read').innerHTML=`<dt>E[Atomic] / D_f</dt><dd>${rat(q.EA)}</dd><dt>E[Binary] / D_f</dt><dd>${rat(q.EB)}</dd><dt>E[MC²]</dt><dd>${q.S2M.toPrecision(4)}</dd><dt>E[Atomic²]</dt><dd>${q.S2A.toPrecision(4)}</dd><dt>Var MC</dt><dd>${(q.S2M-q.D*q.D).toPrecision(4)}</dd><dt>Var Atomic</dt><dd>${(q.S2A-q.EA*q.EA).toPrecision(4)}</dd>`;
  const e3=v=>v===0?'0':v.toPrecision(3);
  $('#c5-tbl tbody').innerHTML=q.rows.map((o,i)=>`<tr><td data-h="token">${i+1}</td><td class="num" data-h="µ">${o.m}</td><td class="num" data-h="π">${f4(o.p)}</td><td class="num" data-h="r">${f4(o.r)}</td><td class="num" data-h="f(r)">${e3(o.h)}</td><td class="num" data-h="µ·f(r)">${e3(o.a)}</td><td class="num" data-h="Binary">${e3(o.bi)}</td><td class="num" data-h="v_MC">${e3(o.vm)}</td><td class="num" data-h="v_Atom">${e3(o.va)}</td></tr>`).join('');
  svg.setAttribute('aria-label',`估計器期望：D_f ${q.D.toFixed(5)}，E[Atomic] ${q.EA.toFixed(5)}，E[Binary] ${q.EB.toFixed(5)}`);
  announce('c5',`λ = ${lam.toFixed(2)}，D_f ${q.D.toFixed(5)}，E[Atomic] ${q.EA.toFixed(5)}，E[Binary] ${q.EB.toFixed(5)}`)}
function initC5(){segInit($('#c5-f'),['TV','chi2','KL','H'],c5.f,k=>({TV:'TV',chi2:'χ²',KL:'KL',H:'Hellinger'})[k],k=>{c5.f=k;c5Draw()});
  $('#c5-l').addEventListener('input',e=>{c5.l=+e.target.value;requestAnimationFrame(c5Draw)});c5Draw()}

function initEqVersions(){const mq=matchMedia('(max-width:599px)');const apply=()=>{$$('.wide-eq').forEach(e=>e.setAttribute('aria-hidden',String(mq.matches)));$$('.narrow-eq').forEach(e=>e.setAttribute('aria-hidden',String(!mq.matches)))};apply();mq.addEventListener('change',apply)}

/* ================= timeline ================= */
const TL_GROUP={'1502.05477':['th','trpo-monotonic-improvement-tv-bound'],'1506.02438':['th','trpo-monotonic-improvement-tv-bound'],
 '1707.06347':['pp','ppo-clip-tv-single-sample-proxy'],'2402.03300':['pp','ppo-clip-tv-single-sample-proxy'],'2501.12948':['pp','ppo-clip-tv-single-sample-proxy'],
 '2503.14476':['cv','llm-rl-clipping-variants-heuristics'],'2505.22617':['cv','llm-rl-clipping-variants-heuristics'],'2511.20347':['cv','llm-rl-clipping-variants-heuristics'],
 '2507.18071':['sq','sequence-level-ratio-gspo-gmpo'],'2507.20673':['sq','sequence-level-ratio-gspo-gmpo'],
 '2602.04879':['es','divergence-estimator-topologies'],'2607.10169':['es','divergence-estimator-topologies'],'2610.13015':['me','gppo-design-space-unification']};
const TL_NAME={th:'理論',pp:'PPO / GRPO',cv:'剪裁變體',sq:'序列單位',es:'估計器',me:'本文',other:'其他'};
/*@TIMELINE@*/
/*@TOCTHEME@*/

/* ================= self test (design.md section 5) ================= */
async function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t=1e-4)=>Math.abs(a-b)<=t+1e-12,rg=(c,lo,hi,t=1e-4)=>near(c.lo,lo,t)&&near(c.hi,hi,t);
  // 5.1 C1
  [[1,[[50,.5,1],[80,.8,1],[100,1,1],[120,1.2,1],[150,1.2,0]]],[-1,[[50,-.8,0],[80,-.8,-1],[100,-1,-1],[120,-1.2,-1],[150,-1.5,-1]]]].forEach(([s,rows])=>rows.forEach(([R,L,g])=>{const v=c1Val(s,R,20);ok(near(v.L,L,1e-12)&&v.g===g,`C1 s=${s} r=${R}`)}));
  // 5.2 C2
  ok(near(mcUpper('TV2',80),1.2)&&near(mcUpper('chi2',80),1.2)&&near(mcUpper('H',80),1.2223)&&near(mcUpper('KLabs',80),1.25)&&near(mcUpper('KLnn',80),1.2308),'C2 r_max at 0.8');
  ok(near(FN.H(.8),.011146,1e-6)&&near(FN.KLnn(.8),.023144,1e-6)&&near(FN.TV(.8),.1,1e-12)&&near(FN.TV2(.8),.2,1e-12),'C2 deltas');
  ok(near(mcUpper('H',70),1.3534)&&near(mcUpper('KLnn',70),1.3755)&&near(mcUpper('KLnn',90),1.1071)&&mcUpper('KLneg',80)===Infinity,'C2 r_min 0.7/0.9, -log r');
  // 5.3 C3
  {const r=VJ.mulberry32(1);ok([.627073941,.002735721,.52744704].every(v=>near(r(),v,1e-9)),'mulberry32 seed 1')}
  {const [P,Q]=VJ.pairs(1,3)[0];ok([.5673,.0016,.4311].every((v,i)=>near(P[i],v))&&[.5117,.4457,.0426].every((v,i)=>near(Q[i],v)),'pairs(1,3)[0]')}
  {const L=(k,V)=>VJ.Lf(VJ.FV[k],V,1000)[0];ok(near(L('chi2',.75),3.001,1e-4)&&near(L('KL',.5),.5323,1e-4)&&near(L('H',.25),.06351,1e-5)&&near(L('TV',.5),.5,1e-5),'L_f values')}
  {const g=k=>VJ.gb(VJ.FV[k],.3,.5);ok(near(g('chi2'),.36,1e-5)&&near(g('KL'),.22314,1e-5)&&near(g('H'),.10263,1e-5),'binary pair g_beta')}
  {const t=await c3Table();ok(near(t.TV.C,1.002,.002)&&near(t.chi2.C,4,.002)&&near(t.KL.C,2,.002)&&near(t.H.C,1,.002),`C_f ${t.TV.C.toFixed(4)} ${t.chi2.C.toFixed(4)} ${t.KL.C.toFixed(4)} ${t.H.C.toFixed(4)}`)}
  {const save=[c3.f,c3.K,c3.seed];c3.f='chi2';c3.K=6;c3.seed=1;const p1=c3Scatter(),id1=reqSeq;c3.K=3;const p2=c3Scatter(),id2=reqSeq;const[r1,r2]=await Promise.all([p1,p2]);
   ok(r1===null&&r2&&c3.applied.includes(id2)&&!c3.applied.includes(id1),'C3 reqId drops stale reply');ok(r2&&r2.res.viol===0,'C3 violations 0 (chi2, K=3)');
   [c3.f,c3.K,c3.seed]=save;await c3Scatter()}
  // 5.4 C4
  ok(rg(cell('page','KL','Atom',.1),.1074,9.3132)&&rg(cell('page','KL','Bin',.1),.0433,3.9748)&&rg(cell('page','KL','MC',.1),.8,1.25),'C4 page KL mu=0.1');
  ok(rg(cell('page','H','MC',.1),.8,1.2223)&&rg(cell('page','H','Atom',.1),.4438,1.7792)&&rg(cell('page','H','Bin',.1),.4599,1.7179),'C4 page H mu=0.1');
  ok(rg(cell('page','chi2','Atom',.1),.3675,1.6325)&&rg(cell('page','chi2','Bin',.1),.4,1.6)&&near(cell('page','TV','Atom',.1).hi,3)&&near(cell('page','TV','Bin',.1).hi,2),'C4 page chi2/TV mu=0.1');
  ok(near(cell('page','KL','Bin',.01).hi,23.4571)&&near(cell('page','H','Atom',.01).hi,4.226)&&near(cell('page','chi2','Bin',.01).hi,2.99)&&cell('page','KL','Atom',.01).hc,'C4 page mu=0.01');
  ok(rg(cell('page','H','Bin',.5),.7903,1.2097)&&rg(cell('page','KL','Bin',.5),.4,1.6),'C4 page mu=0.5');
  ok(rg(cell('literal','TV','MC',.1),.6,1.4)&&near(cell('literal','TV','Atom',.1).hi,5)&&near(cell('literal','TV','Bin',.1).hi,3)&&rg(cell('literal','H','Bin',.1),.4835,1.6761)&&rg(cell('literal','KL','Bin',.1),.0447,3.9494),'C4 literal mu=0.1');
  {const c=cell('literal','KL','MC',.1);ok(near(c.lo,.8025)&&c.hc,'C4 literal KL MC no upper bound')}
  ok(near(cell('literal','H','Atom',.01).hi,4)&&near(cell('literal','KL','Bin',.01).hi,23.2053),'C4 literal mu=0.01');
  ok(DIVS.every(d=>assertMonotone('page',d,.1)&&assertMonotone('literal',d,.01)),'C4 monotone');
  {const a=region(dhat('TV2','Atom',.05),.2,20),b=region(dhat('TV2','Bin',.05),.4,20);ok(near(a.lo,b.lo,1e-9)&&near(a.hi,b.hi,1e-9),'TV-Atom(d) = TV-Bin(2d)')}
  ok(MUG.length===300&&near(MUG[0],1e-4,1e-12)&&near(MUG[299],.999,1e-9),'C4 mu grid');
  // 5.5 C5
  {const q=c5Calc(.3,'chi2');ok(near(q.D,.0836,1e-5)&&near(q.EA,.01268,1e-5)&&near(q.EB,.02279,1e-5)&&near(q.S2M,.07706,1e-5),'C5 chi2 lam=0.3')}
  {const q=c5Calc(.6,'KL');ok(near(q.D,.14494,1e-5)&&near(q.EA,.03301,1e-5)&&near(q.EB,.05,1e-5),'C5 KL lam=0.6')}
  {const q=c5Calc(.3,'TV');ok(near(q.EB,2*q.EA,1e-12),'C5 TV: E[Bin] = 2 E[Atom]')}
  ok(WK_OK,'Blob Worker created');
  // DOM
  ok($$('details.check').length===11,'check boxes: '+$$('details.check').length);
  ok($$('#t1-math tbody tr').length===18&&$$('#t1-code tbody tr').length===18,'Table 1 rows');
  {const tl=JSON.parse($('#timeline-data').textContent);ok(tl.length===13&&tl.every(e=>TL_GROUP[e.arxiv]),'timeline groups')}
  ok($$('.plate').every(p=>p.scrollWidth<=p.clientWidth+1),'plates fit: '+$$('.plate').filter(p=>p.scrollWidth>p.clientWidth+1).map(p=>p.id).join(','));
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ================= boot ================= */
initTheme();initToc();initEqVersions();initC1();initC2();initC3();initC4();initC5();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{c1Draw();c2Draw();c3Draw();c4Draw();c5Draw()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))setTimeout(selfTest,0);
})();
