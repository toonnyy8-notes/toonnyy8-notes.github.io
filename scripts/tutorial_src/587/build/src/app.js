(()=>{
'use strict';
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
function el(tag,a,p){const e=document.createElementNS(NS,tag);if(a)for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function tx(p,x,y,s,a){const t=el('text',Object.assign({x,y},a||{}),p);t.textContent=s;return t}
const lin=(d0,d1,r0,r1)=>v=>r0+(v-d0)/(d1-d0)*(r1-r0);
const lgs=(d0,d1,r0,r1)=>{const a=Math.log(d0),b=Math.log(d1);return v=>r0+(Math.log(v)-a)/(b-a)*(r1-r0)};
function lp(pts){let s='';for(let i=0;i<pts.length;i++)s+=(i?'L':'M')+pts[i][0].toFixed(1)+' '+pts[i][1].toFixed(1);return s}
function widthOf(box,min,max){return Math.max(min,Math.min(max,Math.floor(box.clientWidth||max)))}
function frame(svg,o){
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${o.W} ${o.H}`);svg.setAttribute('width',o.W);svg.setAttribute('height',o.H);
  const m=o.m,x0=m.l,x1=o.W-m.r,y0=o.H-m.b,y1=m.t;
  const X=(o.xlog?lgs:lin)(o.x[0],o.x[1],x0,x1),Y=(o.ylog?lgs:lin)(o.y[0],o.y[1],y0,y1);
  const g=el('g',null,svg);
  (o.yt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:String(v),yy=Y(val);el('line',{x1:x0,x2:x1,y1:yy,y2:yy,class:'grid'},g);tx(g,x0-8,yy+4,lab,{class:'tk','text-anchor':'end'})});
  (o.xt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:String(v),xx=X(val);el('line',{x1:xx,x2:xx,y1:y0,y2:y0+4,class:'axl'},g);tx(g,xx,y0+18,lab,{class:'tk','text-anchor':'middle'})});
  el('line',{x1:x0,x2:x1,y1:y0,y2:y0,class:'axl'},g);
  if(o.xl)tx(g,(x0+x1)/2,o.H-4,o.xl,{class:'al','text-anchor':'middle'});
  if(o.yl)tx(g,4,12,o.yl,{class:'al'});
  return {X,Y,g,x0,x1,y0,y1};
}
// marker shapes: circle (PPO), triangle (DAPO), square (RIC), diamond (no clip), cross (IS)
function mark(g,shape,x,y,cls){const s=5;
  if(shape==='c')return el('circle',{cx:x,cy:y,r:s,class:cls+'-f'},g);
  if(shape==='s')return el('rect',{x:x-s,y:y-s,width:2*s,height:2*s,class:cls+'-f'},g);
  if(shape==='t')return el('path',{d:`M${x} ${y-s-1}L${x+s+1} ${y+s}L${x-s-1} ${y+s}Z`,class:cls+'-f'},g);
  if(shape==='d')return el('path',{d:`M${x} ${y-s-1}L${x+s+1} ${y}L${x} ${y+s+1}L${x-s-1} ${y}Z`,class:cls+'-f'},g);
  if(shape==='x'){el('path',{d:`M${x-s} ${y-s}L${x+s} ${y+s}M${x-s} ${y+s}L${x+s} ${y-s}`,style:'stroke:var(--ink-3);stroke-width:2.5'},g)}}
function clampTri(g,x,y,up){el('path',{d:up?`M${x} ${y-4}L${x+5} ${y+4}L${x-5} ${y+4}Z`:`M${x} ${y+4}L${x+5} ${y-4}L${x-5} ${y-4}Z`,class:'ink-f'},g)}
function placeLabels(g,x,items,ymin,ymax){items=items.slice().sort((a,b)=>a.y-b.y);let last=-1e9;items.forEach(o=>{o.yy=Math.max(o.y+4,last+14);last=o.yy});const over=last-ymax;if(over>0)items.forEach(o=>{o.yy-=over});items.forEach(o=>{tx(g,x,Math.max(ymin,o.yy),o.t,{class:'lab'})})}
const live={};function announce(id,s,now){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);if(now){n.textContent=s;return}live[id]=setTimeout(()=>{n.textContent=s},500)}
const p4=v=>v===Infinity?'∞（其餘 token 機率歸零）':v===0?'0':Math.abs(v)<1e-4?v.toExponential(3):(+v.toPrecision(4)).toString();
const narrow=()=>innerWidth<560;
function table(tb,cols,rows){ // rows: [{name, vals:[...]}]; transposed on narrow screens
  const th=tb.tHead||tb.createTHead(),bd=tb.tBodies[0]||tb.createTBody();
  if(!narrow()){th.innerHTML='<tr><th>方法</th>'+cols.map(c=>`<th class="num">${c}</th>`).join('')+'</tr>';bd.innerHTML=rows.map(r=>`<tr${r.hl?' class="hl"':''}><td>${r.name}</td>`+r.vals.map(v=>`<td class="num">${v}</td>`).join('')+'</tr>').join('')}
  else{th.innerHTML='<tr><th>量測</th>'+rows.map(r=>`<th class="num">${r.name}</th>`).join('')+'</tr>';bd.innerHTML=cols.map((c,i)=>`<tr><td>${c}</td>`+rows.map(r=>`<td class="num">${r.vals[i]}</td>`).join('')+'</tr>').join('')}}
// log slider: integer 0..1000 mapped to [a,b]; arrows 10, PageUp/Down 100
function logSlider(inp,a,b){const la=Math.log10(a);let lb=Math.log10(b);
  const o={get:()=>10**(la+(+inp.value)*(lb-la)/1000),set:v=>{inp.value=String(Math.round((Math.log10(v)-la)/(lb-la)*1000))},setMax:v=>{lb=Math.log10(v)}};
  inp.addEventListener('keydown',e=>{const st={ArrowRight:10,ArrowUp:10,ArrowLeft:-10,ArrowDown:-10,PageUp:100,PageDown:-100}[e.key];if(st===undefined)return;e.preventDefault();inp.value=String(Math.max(0,Math.min(1000,+inp.value+st)));inp.dispatchEvent(new Event('input',{bubbles:true}));inp.dispatchEvent(new Event('change',{bubbles:true}))});
  return o}

/* ---------- W1 math (same as verify/check_w1.py) ---------- */
const ricEps=(delta,po)=>Math.sqrt(delta/po);
function bounds(method,po,P){let lo,hi;if(method==='PPO'){lo=hi=P.eps}else if(method==='DAPO'){lo=P.eps;hi=P.epsH}else{lo=hi=ricEps(P.delta,po)}
  return{rlo:Math.max(0,1-lo),rhi:Math.min(1+hi,1/po),lo}}
function measures(po,r){const p=po*r,dp=p-po,atom=.5*po*(r-1)**2,eucl=(r-1)**2,binchi=.5*dp*dp/(po*(1-po));
  const binkl=p>=1-1e-12?Infinity:-po*Math.log1p(dp/po)-(1-po)*Math.log1p(-dp/(1-po));return{pmax:p,dp,atom,eucl,binchi,binkl}}
const W1D={eps:.2,epsH:.28,delta:.02};
const w1={po:.01,adv:false,P:Object.assign({},W1D)};
let w1s;
function renderW1(){const P=w1.adv?w1.P:W1D,po=w1.po,box=$('#w1-fig'),svgs=$$('svg',box),W=widthOf(box,280,620),nar=narrow();
  $('#w1-po').textContent=po.toPrecision(3);$('#w1-p').setAttribute('aria-valuetext','舊機率 '+po.toPrecision(3));
  $('#w1-eo').textContent=w1.P.eps.toFixed(2);$('#w1-eho').textContent=w1.P.epsH.toFixed(2);$('#w1-do').textContent=w1.P.delta.toFixed(3);
  const meths=w1.adv?['PPO','DAPO','RIC']:['PPO','RIC'],cls={PPO:'ppo',DAPO:'dapo',RIC:'acc'},shp={PPO:'c',DAPO:'t',RIC:'s'};
  const xt=[[1e-4,'10⁻⁴'],[1e-3,'10⁻³'],[1e-2,'0.01'],[.1,'0.1'],[1,'1']],N=240,xs=Array.from({length:N+1},(_,i)=>10**(-4+4*i/N)).map(v=>Math.min(v,.99));
  // top: r_hi
  let F=frame(svgs[0],{W,H:230,m:{l:48,r:nar?12:64,t:26,b:40},x:[1e-4,1],xlog:true,y:[1,40],ylog:true,xt,yt:[[1.2,'1.2'],[2,'2'],[5,'5'],[10,'10'],[40,'40']],xl:'舊機率 π_old（對數）',yl:'比值上限 r'});
  const cap=xs.filter(v=>1/v<=40).map(v=>[F.X(v),F.Y(1/v)]);el('path',{d:lp(cap),class:'capl'},F.g);
  if(!nar)tx(F.g,F.X(.04),F.Y(1/.04)-6,'受 π ≤ 1 限制',{class:'tk'});
  meths.forEach(m=>{const pts=xs.map(v=>[F.X(v),F.Y(Math.min(40,bounds(m,v,P).rhi))]);el('path',{d:lp(pts),class:cls[m]},F.g);
    const b=bounds(m,po,P);mark(F.g,shp[m],F.X(po),F.Y(Math.min(40,b.rhi)),cls[m]);
  });
  if(!nar)placeLabels(F.g,F.x1+6,meths.map(m=>({t:m,y:F.Y(Math.min(40,bounds(m,.99,P).rhi))})),F.y1,F.y0);
  el('line',{x1:F.X(po),x2:F.X(po),y1:F.y0,y2:F.y1,class:'cur'},F.g);
  const cross=P.delta/(P.eps*P.eps);
  if(w1.adv){if(cross<=1){el('line',{x1:F.X(Math.max(1e-4,cross)),x2:F.X(Math.max(1e-4,cross)),y1:F.y0,y2:F.y1,class:'ref'},F.g);if(!nar)tx(F.g,F.X(Math.max(1e-4,cross))+4,F.y1+10,'右側 RIC 上限較小',{class:'tk'})}}
  // bottom: atom distance at upper bound
  const lo=1e-6,hi=.2;
  F=frame(svgs[1],{W,H:230,m:{l:48,r:nar?12:64,t:26,b:40},x:[1e-4,1],xlog:true,y:[lo,hi],ylog:true,xt,yt:[[1e-6,'10⁻⁶'],[1e-4,'10⁻⁴'],[1e-2,'0.01'],[.2,'0.2']],xl:'舊機率 π_old（對數）',yl:'上邊界的單 token 距離 ½π_old(r−1)²'});
  meths.forEach(m=>{let tri=null;const pts=xs.map(v=>{const a=measures(v,bounds(m,v,P).rhi).atom;if(a<lo&&tri===null)tri=v;return[F.X(v),F.Y(Math.max(lo,Math.min(hi,a)))]});
    el('path',{d:lp(pts),class:cls[m]},F.g);
    let lastUnder=null;xs.forEach(v=>{if(measures(v,bounds(m,v,P).rhi).atom<lo)lastUnder=v});if(lastUnder!==null)clampTri(F.g,F.X(lastUnder),F.y0-6,false);
    const a=measures(po,bounds(m,po,P).rhi).atom;if(a<lo)clampTri(F.g,F.X(po),F.y0-6,false);else mark(F.g,shp[m],F.X(po),F.Y(Math.min(hi,a)),cls[m]);
  });
  if(!nar)placeLabels(F.g,F.x1+6,meths.map(m=>({t:m,y:F.Y(Math.max(lo,Math.min(hi,measures(.99,bounds(m,.99,P).rhi).atom)))})),F.y1,F.y0);
  el('line',{x1:F.X(po),x2:F.X(po),y1:F.y0,y2:F.y1,class:'cur'},F.g);
  // table
  const cols=w1.adv?['π_min','π_max','Δπ','單 token（Eq. 9）','歐氏（Eq. 3）','二元二階','二元 KL']:['π_min','π_max','Δπ','單 token（Eq. 9）'];
  const rows=meths.map(m=>{const b=bounds(m,po,P),q=measures(po,b.rhi);const pmin=po*b.rlo;
    const v=[m==='RIC'&&1-b.lo<0?'無下限（r ≥ 0）':p4(pmin),p4(q.pmax),p4(q.dp),p4(q.atom)];if(w1.adv)v.push(p4(q.eucl),p4(q.binchi),p4(q.binkl));return{name:m,vals:v,hl:m==='RIC'}});
  table($('#w1-tab'),cols,rows);
  $('#w1-tab caption').textContent=w1.adv?`上邊界的量測（ε = ${P.eps.toFixed(2)}、ε_high = ${P.epsH.toFixed(2)}、δ = ${P.delta.toFixed(3)}）`:'上邊界的量測（基本檢視 δ = 0.02、ε = 0.2）';
  const ric=bounds('RIC',po,P);
  $('#w1-ro').innerHTML=`RIC ε = √(δ/π_old) = <b class="acc">${p4(ricEps(P.delta,po))}</b><br>分界 δ/ε² = <b>${p4(cross)}</b>${cross>1?'（RIC 上限處處較大）':''}`;
  announce('w1',`舊機率 ${po.toPrecision(3)}：PPO 上限 ${p4(measures(po,bounds('PPO',po,P).rhi).pmax)}，RIC 上限 ${p4(measures(po,ric.rhi).pmax)}`);
}
function initW1(){w1s=logSlider($('#w1-p'),1e-4,.99);w1s.set(w1.po);
  $('#w1-p').addEventListener('input',()=>{w1.po=w1s.get();renderW1()});
  $$('#w1 [data-p]').forEach(b=>b.addEventListener('click',()=>{w1.po=+b.dataset.p;w1s.set(w1.po);renderW1()}));
  $$('#w1 [data-d]').forEach(b=>b.addEventListener('click',()=>{w1.P.delta=+b.dataset.d;$('#w1-d').value=b.dataset.d;renderW1()}));
  $('#w1-e').addEventListener('input',e=>{w1.P.eps=+e.target.value;if(w1.P.epsH<w1.P.eps){w1.P.epsH=w1.P.eps;$('#w1-eh').value=w1.P.eps}renderW1()});
  $('#w1-eh').addEventListener('input',e=>{w1.P.epsH=Math.max(w1.P.eps,+e.target.value);e.target.value=w1.P.epsH;renderW1()});
  $('#w1-d').addEventListener('input',e=>{w1.P.delta=+e.target.value;renderW1()});
  $('#w1-adv').addEventListener('click',e=>{w1.adv=!w1.adv;e.target.setAttribute('aria-pressed',String(w1.adv));$('#w1').classList.toggle('adv',w1.adv);
    if(!w1.adv){w1.P=Object.assign({},W1D);$('#w1-e').value=.2;$('#w1-eh').value=.28;$('#w1-d').value=.02}renderW1()});
  renderW1()}

/* ---------- W2 (same as verify/check_w2.py) ---------- */
function capsW2(po,eps,delta,dual){const c={IS:Infinity,PPO:1+eps,RIC:1+Math.sqrt(delta/po)};if(dual)for(const k in c)c[k]=Math.min(c[k],10);return c}
function rowW2(po,r,P){const pt=po*r,c=capsW2(po,P.eps,P.delta,P.dual),o={};for(const k in c){const w=Math.min(r,c[k]);o[k]={w,v:po*w*w,bias:pt-po*w}}return o}
const w2={po:.001,r:10,eps:.2,delta:.05,dual:false};let w2p,w2r,w2t=0;
function renderW2(){const {po,r}=w2,P=w2,pt=po*r,box=$('#w2-fig'),W=widthOf(box,280,620),nar=narrow();
  $('#w2-po').textContent=po.toPrecision(3);$('#w2-ro2').textContent=r.toPrecision(3);$('#w2-eo').textContent=w2.eps.toFixed(2);$('#w2-do').textContent=w2.delta.toFixed(3);
  $('#w2-p').setAttribute('aria-valuetext','舊機率 '+po.toPrecision(3));$('#w2-r').setAttribute('aria-valuetext','目標比值 '+r.toPrecision(3));
  const lo=1e-6,hi=10,F=frame($('svg',box),{W,H:300,m:{l:48,r:nar?12:56,t:26,b:40},x:[1e-5,1],xlog:true,y:[lo,hi],ylog:true,
    xt:[[1e-5,'10⁻⁵'],[1e-3,'10⁻³'],[.1,'0.1'],[1,'1']],yt:[[1e-6,'10⁻⁶'],[1e-4,'10⁻⁴'],[1e-2,'0.01'],[1,'1']],xl:'舊機率 π_old（對數）',yl:'clip 邊界上的變異貢獻 v'});
  const N=200,xs=Array.from({length:N+1},(_,i)=>10**(-5+5*i/N)).map(v=>Math.min(v,.9999)),cl=v=>F.Y(Math.max(lo,Math.min(hi,v)));
  el('line',{x1:F.x0,x2:F.x1,y1:F.Y(P.delta),y2:F.Y(P.delta),class:'ref'},F.g);
  const cP=v=>v*Math.min(1+P.eps,P.dual?10:Infinity)**2,cR=v=>v*Math.min(1+Math.sqrt(P.delta/v),P.dual?10:Infinity)**2,cI=v=>v*Math.min(pt/v,P.dual?10:Infinity)**2;
  el('path',{d:lp(xs.map(v=>[F.X(v),cl(cP(v))])),class:'ppo'},F.g);el('path',{d:lp(xs.map(v=>[F.X(v),cl(cR(v))])),class:'acc'},F.g);
  const xi=xs.filter(v=>v<=pt);if(xi.length>1)el('path',{d:lp(xi.map(v=>[F.X(v),cl(cI(v))])),class:'nocl'},F.g);
  if(xi.length&&cI(1e-5)>hi)clampTri(F.g,F.X(1e-5)+6,F.y1+6,true);
  if(!nar){placeLabels(F.g,F.x1+4,[{t:'PPO',y:cl(cP(.9999))},{t:'RIC',y:cl(cR(.9999))},{t:'δ',y:F.Y(P.delta)}],F.y1,F.y0);if(xi.length>1){const xl=Math.max(xi[0],xi[Math.floor(xi.length*.15)]);tx(F.g,F.X(xl)+10,cl(cI(xl))-6,`IS（π_θ = ${pt.toPrecision(3)}）`,{class:'tk'})}}
  const rw=rowW2(po,r,P);[['IS','x','nocl'],['PPO','c','ppo'],['RIC','s','acc']].forEach(([k,s,c])=>{const v=rw[k].v;if(v>hi)clampTri(F.g,F.X(po),F.y1+6,true);else mark(F.g,s,F.X(po),cl(v),c)});
  el('line',{x1:F.X(po),x2:F.X(po),y1:F.y0,y2:F.y1,class:'cur'},F.g);
  const f=v=>v===0?'0':Math.abs(v)>=1e4||Math.abs(v)<1e-4?v.toExponential(3):(+v.toPrecision(5)).toString();
  table($('#w2-tab'),['權重 w','變異貢獻 v','偏差 π_θ − π_old·w'],['IS','PPO','RIC'].map(k=>({name:k==='IS'?'未截斷 IS':k,vals:[f(rw[k].w),f(rw[k].v),f(rw[k].bias)],hl:k==='RIC'})));
  const cut=r>capsW2(po,P.eps,P.delta,false).RIC;
  $('#w2-ro').innerHTML=`π_θ = π_old·r = <b>${pt.toPrecision(3)}</b>　RIC 上限 1 + √(δ/π_old) = <b class="acc">${(1+Math.sqrt(P.delta/po)).toPrecision(4)}</b>　此 token 是否被 RIC 截斷：<b>${cut?'是':'否（與未截斷 IS 相同）'}</b>`;
  announce('w2',`舊機率 ${po.toPrecision(3)}，比值 ${r.toPrecision(3)}：IS v = ${f(rw.IS.v)}，PPO v = ${f(rw.PPO.v)}，RIC v = ${f(rw.RIC.v)}`);
}
function syncR(){const rmax=Math.min(1000,1/w2.po);w2r.setMax(rmax);if(w2.r>rmax+1e-12){w2.r=rmax;const h=$('#w2-hint');h.textContent='r 已調為 1/π_old（π_θ 不能超過 1）';announce('w2','r 已調為 1/π_old',true);clearTimeout(w2t);w2t=setTimeout(()=>{h.textContent=''},3000)}w2r.set(Math.max(1,w2.r))}
function initW2(){w2p=logSlider($('#w2-p'),1e-5,.9);w2r=logSlider($('#w2-r'),1,1000);w2p.set(w2.po);syncR();
  $('#w2-p').addEventListener('input',()=>{w2.po=w2p.get();syncR();renderW2()});
  $('#w2-r').addEventListener('input',()=>{w2.r=Math.min(w2r.get(),1/w2.po);renderW2()});
  $('#w2-e').addEventListener('input',e=>{w2.eps=+e.target.value;renderW2()});$('#w2-d').addEventListener('input',e=>{w2.delta=+e.target.value;renderW2()});
  $('#w2-dual').addEventListener('change',e=>{w2.dual=e.target.checked;renderW2()});renderW2()}

/* ---------- W3 toy bandit (verbatim from verify/sim.js) ---------- */
function mulberry32(a){return function(){a=(a+0x6D2B79F5)|0;let t=Math.imul(a^(a>>>15),a|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296}}
function softmax(z){let m=-Infinity;for(const v of z)if(v>m)m=v;const e=z.map(v=>Math.exp(v-m));let s=0;for(const v of e)s+=v;return e.map(v=>v/s)}
function entropy(p){let h=0;for(const q of p)if(q>0)h-=q*Math.log(q);return h}
const SUCC=[0.5,1.0,0,0,0,0];
function initP(pB){return [0.81-pB,pB,0.0475,0.0475,0.0475,0.0475]}
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
function newTrace(T){return{pB:new Float64Array(T+1),H:new Float64Array(T+1),found:new Float64Array(T+1),skipped:new Float64Array(1),rew:0}}
function addFinal(tr,p,T){tr.pB[T]+=p[1];tr.H[T]+=entropy(p);tr.found[T]+=p[1]>0.5?1:0;tr.rew+=0.5*p[0]+p[1]}
function normTrace(tr,S,T){for(const k of ['pB','H','found'])for(let i=0;i<=T;i++)tr[k][i]/=S;tr.rew/=S}
function batch(cfg,pB,S=200,T=200){const tr=newTrace(T);for(let s=0;s<S;s++){const p=runSeed(cfg,s+1,pB,T,8,4,1.0,tr);addFinal(tr,p,T)}normTrace(tr,S,T);return{tr,rew:tr.rew}}
const W3S=200,W3T=200,W3M=[['GRPO','ppo','c'],['DAPO','dapo','t'],['RIC','acc','s'],['不 clip','nocl','d']];
const w3={status:'idle',runToken:0,res:null,pB:.01,eps:.2,epsH:.28,delta:.05,staleAfter:false};
function w3cfgs(){return[{method:'ppo',eps:w3.eps},{method:'dapo',eps:w3.eps,epsH:w3.epsH},{method:'ric',delta:w3.delta},{method:'none'}]}
function w3params(i){return[`ε = ${w3.eps.toFixed(2)}`,`ε = ${w3.eps.toFixed(2)}, ε_high = ${w3.epsH.toFixed(2)}`,`δ = ${w3.delta.toFixed(3)}`,'—'][i]}
function batchChunked(cfgs,pB,token,onProgress,onDone){const trs=cfgs.map(()=>newTrace(W3T)),total=cfgs.length*W3S;let ci=0,s=0,done=0;
  const step=()=>{if(token!==w3.runToken)return;for(let n=0;n<20&&ci<cfgs.length;n++){const p=runSeed(cfgs[ci],s+1,pB,W3T,8,4,1.0,trs[ci]);addFinal(trs[ci],p,W3T);done++;if(++s>=W3S){s=0;ci++}}
    onProgress(done,total);if(ci<cfgs.length)setTimeout(step,0);else{trs.forEach(t=>normTrace(t,W3S,W3T));if(token===w3.runToken)onDone(trs)}};setTimeout(step,0)}
function w3btn(){const b=$('#w3-run'),st=$('#w3-st');b.disabled=w3.status==='running';
  b.textContent={idle:'執行',running:'執行中',done:'重新執行',stale:'參數已變更，按執行更新'}[w3.status];
  if(w3.status==='idle')st.textContent='尚未執行';if(w3.status==='stale')st.textContent='目前顯示的是舊參數的結果';
  $('#w3-tw').classList.toggle('stale',w3.status==='stale');$$('#w3 .fig').forEach(f=>f.classList.toggle('stale',w3.status==='stale'))}
function runW3(){const token=++w3.runToken;w3.status='running';w3.staleAfter=false;w3btn();let q=0;const pB=w3.pB,snap=[0,1,2,3].map(w3params);
  batchChunked(w3cfgs(),pB,token,(d,t)=>{$('#w3-st').textContent=`執行中 ${d}/${t}`;const nq=Math.floor(4*d/t);if(nq>q&&nq<4){q=nq;announce('w3',`玩具模擬完成 ${nq*25}%`,true)}},
    trs=>{w3.res={trs,pB,snap};w3.status=w3.staleAfter?'stale':'done';w3btn();if(w3.status==='done')$('#w3-st').textContent=`B 初始機率 ${pB}，200 個 seed × 200 次迭代`;renderW3();
      announce('w3','玩具模擬完成 100%：'+W3M.map((m,i)=>`${m[0]} ${trs[i].found[W3T].toFixed(3)}`).join('，'),true)})}
function renderW3(){const nar=narrow();
  const draw=(box,key,ymax,yl,yt)=>{const W=widthOf(box,260,520),F=frame($('svg',box),{W,H:240,m:{l:44,r:nar?12:58,t:26,b:40},x:[0,W3T],y:[0,ymax],xt:[0,50,100,150,200],yt,xl:'迭代次數',yl});
    if(!w3.res){tx(F.g,(F.x0+F.x1)/2,(F.y0+F.y1)/2,'尚未執行',{class:'al','text-anchor':'middle'});return}
    w3.res.trs.forEach((tr,i)=>{const pts=[];for(let t=0;t<=W3T;t+=2)pts.push([F.X(t),F.Y(tr[key][t])]);el('path',{d:lp(pts),class:W3M[i][1]},F.g);
      [50,100,150,200].forEach(t=>mark(F.g,W3M[i][2],F.X(t),F.Y(tr[key][t]),W3M[i][1]))});
    if(!nar)placeLabels(F.g,F.x1+8,w3.res.trs.map((tr,i)=>({t:W3M[i][0],y:F.Y(tr[key][W3T])})),F.y1,F.y0)};
  draw($('#w3-fa'),'found',1,'找到 B 的比例',[[0,'0'],[.25,'0.25'],[.5,'0.5'],[.75,'0.75'],[1,'1']]);
  draw($('#w3-fb'),'H',Math.log(6),'平均熵',[[0,'0'],[.5,'0.5'],[1,'1'],[1.5,'1.5'],[Math.log(6),'ln 6']]);
  if(!w3.res){table($('#w3-tab'),['參數','找到 B 的比例','平均熵','期望獎勵','平均略過的迭代'],W3M.map(m=>({name:m[0],vals:['—','—','—','—','—']})));return}
  table($('#w3-tab'),['參數','找到 B 的比例','平均熵','期望獎勵','平均略過的迭代'],w3.res.trs.map((tr,i)=>({name:W3M[i][0],vals:[w3.res.snap[i],tr.found[W3T].toFixed(3),tr.H[W3T].toFixed(3),tr.rew.toFixed(3),(tr.skipped[0]/W3S).toFixed(2)],hl:i===2})))}
function initW3(){const mark_stale=()=>{if(w3.status==='running')w3.staleAfter=true;else if(w3.status!=='idle'){w3.status='stale'}w3btn()};
  $$('#w3-pb button').forEach(b=>b.addEventListener('click',()=>{w3.pB=+b.dataset.v;$$('#w3-pb button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));mark_stale()}));
  $('#w3-e').addEventListener('input',e=>{w3.eps=+e.target.value;$('#w3-eo').textContent=w3.eps.toFixed(2);if(w3.epsH<w3.eps){w3.epsH=w3.eps;$('#w3-eh').value=w3.eps;$('#w3-eho').textContent=w3.eps.toFixed(2)}mark_stale()});
  $('#w3-eh').addEventListener('input',e=>{w3.epsH=Math.max(w3.eps,+e.target.value);e.target.value=w3.epsH;$('#w3-eho').textContent=w3.epsH.toFixed(2);mark_stale()});
  $('#w3-d').addEventListener('input',e=>{w3.delta=+e.target.value;$('#w3-do').textContent=w3.delta.toFixed(3);mark_stale()});
  $('#w3-run').addEventListener('click',runW3);w3btn();renderW3();
  const start=()=>{if(w3.status==='idle')runW3()};
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){io.disconnect();('requestIdleCallback' in window)?requestIdleCallback(start,{timeout:2000}):setTimeout(start,500)}},{threshold:.1});io.observe($('#w3'))}}

/* ---------- W4 results ---------- */
const BENCH=['AIME24','AIME25','AMC23','HMMT25','BRUMO25','CMIMC25','SMT25','平均'];
const METH=['Base','GRPO','DAPO','GSPO','GMPO','DCPO','RIPO'];
const T1={'Qwen3-1.7B-Base':{Base:[2.1,0.0,28.4,0.0,11.3,0.0,6.6,6.9,'−38.4%'],GRPO:[11.3,10.8,33.8,0.0,15.0,0.3,7.3,11.2,'+0.0%'],DAPO:[15.0,12.1,39.7,0.8,9.2,1.9,7.8,12.4,'+10.7%'],GSPO:[16.3,11.2,40.0,1.6,12.9,2.8,8.5,13.3,'+19.0%'],GMPO:[17.5,12.5,40.9,0.8,15.0,3.8,6.6,13.9,'+24.1%'],DCPO:[17.1,11.3,46.9,0.8,15.4,1.3,10.8,14.8,'+32.1%'],RIPO:[18.3,12.9,46.1,1.3,15.4,3.2,10.4,15.4,'+37.2%']},
 'Llama3.2-3B-Instruct':{Base:[5.4,0.0,17.8,0.0,1.3,0.0,0.0,3.5,'−45.3%'],GRPO:[16.3,1.3,24.5,0.4,1.7,0.0,0.7,6.4,'+0.0%'],DAPO:[20.1,1.7,25.3,0.8,2.5,0.3,0.9,7.4,'+15.6%'],GSPO:[17.5,1.7,25.7,0.4,3.8,0.6,0.9,7.2,'+12.5%'],GMPO:[16.7,1.3,27.7,0.4,2.8,0.9,1.7,7.4,'+15.6%'],DCPO:[16.7,2.1,24.7,0.4,2.5,0.6,1.4,6.9,'+7.8%'],RIPO:[22.1,2.1,28.1,0.8,4.3,0.9,1.7,8.6,'+34.4%']},
 'Qwen3-4B-Base':{Base:[10.0,8.3,50.0,0.0,18.3,1.3,8.9,13.8,'−46.3%'],GRPO:[30.4,20.8,63.4,7.5,25.8,12.4,19.8,25.7,'+0.0%'],DAPO:[31.7,21.3,69.4,9.2,26.3,10.0,21.7,27.1,'+5.4%'],GSPO:[32.1,21.3,70.3,9.2,26.7,10.0,18.2,26.8,'+4.3%'],GMPO:[32.5,26.7,68.8,11.3,26.3,10.6,22.5,28.4,'+10.5%'],DCPO:[31.7,24.2,71.3,8.8,25.0,15.0,19.8,27.8,'+8.2%'],RIPO:[32.9,27.5,73.1,10.0,31.3,14.7,21.5,30.1,'+17.1%']},
 'Qwen3-8B-Base':{Base:[3.8,3.3,43.4,1.3,15.0,3.1,10.8,11.5,'−59.6%'],GRPO:[31.7,20.8,66.6,12.9,33.3,12.5,21.9,28.5,'+0.0%'],DAPO:[33.7,22.1,70.9,7.9,33.3,18.1,28.3,30.6,'+7.4%'],GSPO:[37.5,19.6,70.1,15.0,36.3,13.8,32.5,32.1,'+12.6%'],GMPO:[41.7,30.0,76.9,11.7,35.8,17.8,33.0,35.3,'+23.9%'],DCPO:[36.3,27.5,72.2,15.4,42.8,16.3,31.6,34.5,'+21.1%'],RIPO:[43.8,29.2,79.7,16.7,47.5,18.1,34.2,38.5,'+35.1%']}};
const T2=[[0.02,0.02,40.8],[0.05,0.04,41.7],[0.05,0.05,43.8],[0.08,0.08,42.1],[0.05,0.02,28.8],[0.08,0.02,27.5],[0.08,0.04,27.9]];
const KS=[1,8,16,32,64,128],T5={'AIME-25':{Base:[3.3,5.0,12.3,16.7,16.7,16.7,16.7],GRPO:[20.8,20.4,36.5,40.8,45.5,50.1,53.3],DAPO:[22.1,20.6,33.4,35.7,37.4,39.1,40.0],DCPO:[27.5,26.4,39.5,44.7,48.5,54.7,58.9],RIPO:[29.2,30.4,43.2,47.0,50.8,55.6,60.0]},
 'HMMT-25':{Base:[1.3,0.6,3.5,5.2,6.3,6.7,6.7],GRPO:[12.9,11.7,19.3,21.6,24.1,26.6,30.0],DAPO:[7.9,7.7,16.7,20.3,23.4,25.7,26.7],DCPO:[15.4,15.5,26.8,30.6,34.3,37.5,40.0],RIPO:[16.7,16.7,28.3,32.6,37.3,41.4,45.3]}};
const w4={tab:'t1',model:'Qwen3-8B-Base',bench:7,ds:'AIME-25'};
function renderW4(){const box=$('#w4-fig'),svg=$('svg',box),W=widthOf(box,280,900),nar=narrow();
  $('#w4-c1').classList.toggle('hide',w4.tab!=='t1');$('#w4-c5').classList.toggle('hide',w4.tab!=='t5');
  let cap='',tbl='';
  if(w4.tab==='t1'){const D=T1[w4.model],b=w4.bench,vals=METH.map(m=>D[m][b]),mx=Math.max(...vals.slice(1));
    const lw=nar?92:110,H=7*34+48,F=frame(svg,{W,H,m:{l:lw,r:nar?70:120,t:16,b:32},x:[0,b===2?100:Math.max(10,Math.ceil(Math.max(...vals)/10)*10)],y:[0,7],xt:[],yt:[],xl:'Avg@8'});
    METH.forEach((m,i)=>{const y=F.Y(7-i)+6,h=22,v=D[m][b];el('rect',{x:F.x0,y,width:Math.max(1,F.X(v)-F.x0),height:h,rx:3,style:`fill:${m==='RIPO'?'var(--accent)':'var(--ink-3)'};opacity:${m==='RIPO'?1:.55}`},F.g);
      tx(F.g,F.x0-8,y+15,m==='RIPO'?'RIPO（本文）':m,{class:'lab','text-anchor':'end'});
      tx(F.g,F.X(v)+6,y+15,v.toFixed(1)+(b===7&&!nar?`（${D[m][8]}）`:'')+(m!=='Base'&&v===mx?' 最高':''),{class:'labm'})});
    cap=`${w4.model}，${BENCH[b]}。`+(b===7?'平均欄括號內為論文報告的相對 GRPO 百分比。':'');
    tbl=`<table><caption>Table 1：${w4.model}（Avg@8）</caption><thead><tr><th>方法</th>${BENCH.map(x=>`<th class="num">${x}</th>`).join('')}</tr></thead><tbody>${METH.map(m=>`<tr${m==='RIPO'?' class="hl"':''}><td>${m}</td>${D[m].slice(0,8).map(v=>`<td class="num">${v.toFixed(1)}</td>`).join('')}</tr>`).join('')}</tbody></table>`}
  else if(w4.tab==='t2'){const F=frame(svg,{W,H:7*34+60,m:{l:nar?96:120,r:24,t:16,b:40},x:[25,45],y:[0,7],xt:[25,30,35,40,45],yt:[],xl:'AIME24 Avg@8'});
    T2.forEach(([a,b,v],i)=>{const y=F.Y(7-i)+17,sym=a===b;el('line',{x1:F.x0,x2:F.x1,y1:y,y2:y,class:'grid'},F.g);el('circle',{cx:F.X(v),cy:y,r:6,style:sym?'fill:var(--accent)':'fill:var(--surface);stroke:var(--ink);stroke-width:2'},F.g);
      tx(F.g,F.x0-8,y+4,`${a} / ${b}`,{class:'labm','text-anchor':'end'});tx(F.g,F.X(v)+(v>40?-12:12),y-10,v.toFixed(1),{class:'labm','text-anchor':v>40?'end':'start'})});
    tx(F.g,F.x0-8,F.y1+6,'δ_low / δ_high',{class:'tk','text-anchor':'end'});
    cap='實心：對稱設定；空心：不對稱設定。表中後三欄為 δ_low 大於 δ_high，與正文敘述方向不同，原文如此。每個設定只有一次訓練。';
    tbl=`<table><caption>Table 2：δ 消融（Qwen3-8B-Base，AIME24 Avg@8）</caption><thead><tr><th class="num">δ_low</th><th class="num">δ_high</th><th class="num">Avg@8</th></tr></thead><tbody>${T2.map(r=>`<tr><td class="num">${r[0]}</td><td class="num">${r[1]}</td><td class="num">${r[2]}</td></tr>`).join('')}</tbody></table>`}
  else{const D=T5[w4.ds],ms=['Base','GRPO','DAPO','DCPO','RIPO'],sh={Base:'x',GRPO:'c',DAPO:'t',DCPO:'d',RIPO:'s'},cl={Base:'nocl',GRPO:'ppo',DAPO:'dapo',DCPO:'ink',RIPO:'acc'};
    const F=frame(svg,{W,H:300,m:{l:44,r:nar?16:70,t:20,b:40},x:[1,128],xlog:true,y:[0,70],xt:KS,yt:[0,20,40,60],xl:'k（對數）',yl:'Pass@k'});
    const ends=[];ms.forEach(m=>{const pts=KS.map((k,i)=>[F.X(k),F.Y(D[m][i+1])]);el('path',{d:lp(pts),class:cl[m]},F.g);pts.forEach(p=>m==='DCPO'?el('path',{d:`M${p[0]} ${p[1]-6}L${p[0]+6} ${p[1]}L${p[0]} ${p[1]+6}L${p[0]-6} ${p[1]}Z`,class:'ink-f'},F.g):mark(F.g,sh[m],p[0],p[1],cl[m]));ends.push({m,y:pts[5][1]})});
    if(!nar)placeLabels(F.g,F.x1+8,ends.map(o=>({t:o.m,y:o.y})),F.y1,F.y0)
    cap=`${w4.ds}：Pass@128 為 RIPO ${D.RIPO[6]}、GRPO ${D.GRPO[6]}、Base ${D.Base[6]}。`;
    tbl=`<table><caption>Table 5：${w4.ds}（Qwen3-8B-Base）</caption><thead><tr><th>方法</th><th class="num">Avg@8</th>${KS.map(k=>`<th class="num">Pass@${k}</th>`).join('')}</tr></thead><tbody>${ms.map(m=>`<tr${m==='RIPO'?' class="hl"':''}><td>${m}</td>${D[m].map(v=>`<td class="num">${v.toFixed(1)}</td>`).join('')}</tr>`).join('')}</tbody></table>`}
  $('#w4-cap').textContent=cap;$('#w4-tbl').innerHTML=tbl;svg.setAttribute('aria-label',cap)}
function initW4(){$('#w4-m').innerHTML=Object.keys(T1).map(m=>`<button type="button" data-v="${m}" aria-pressed="${m===w4.model}">${m}</button>`).join('');
  $$('#w4-m button').forEach(b=>b.addEventListener('click',()=>{w4.model=b.dataset.v;$$('#w4-m button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));renderW4()}));
  $('#w4-b').innerHTML=BENCH.map((x,i)=>`<option value="${i}"${i===7?' selected':''}>${x}</option>`).join('');$('#w4-b').addEventListener('change',e=>{w4.bench=+e.target.value;renderW4()});
  $$('#w4-ds button').forEach(b=>b.addEventListener('click',()=>{w4.ds=b.dataset.v;$$('#w4-ds button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));renderW4()}));
  const tabs=$$('#w4-tabs [role=tab]'),sel=t=>{tabs.forEach(x=>{const on=x===t;x.setAttribute('aria-selected',String(on));x.tabIndex=on?0:-1});w4.tab=t.dataset.t;renderW4()};
  tabs.forEach((t,i)=>{t.addEventListener('click',()=>sel(t));t.addEventListener('keydown',e=>{const d=e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0;if(!d)return;e.preventDefault();const n=tabs[(i+d+tabs.length)%tabs.length];n.focus();sel(n)})});
  renderW4()}

/* ---------- hero probability axis ---------- */
function heroAxis(){const node=$('#pax-hero');node.innerHTML='<div class="px-track"><div class="px-line"></div></div>';const tr=node.firstChild,X=v=>(Math.log10(v)+4)/4*100;
  [[1e-4,'10⁻⁴','l'],[1e-3,'10⁻³',''],[1e-2,'0.01',''],[.1,'0.1',''],[1,'1','r']].forEach(([v,s,c])=>{const t=document.createElement('span');t.className='px-tick '+c;t.style.left=X(v)+'%';t.textContent=s;tr.appendChild(t)});
  const items=[];
  [[.01,.012,.0241],[.8,.96,.9265]].forEach(([po,pp,pr])=>{const d=document.createElement('span');d.className='px-dot';d.style.left=X(po)+'%';tr.appendChild(d);
    [['ppo',pp,'PPO',30],['ric',pr,'RIC',42]].forEach(([c,pm,name,b])=>{const bar=document.createElement('span');bar.className='px-bar '+c;bar.style.left=X(po)+'%';bar.style.width=`calc(${X(pm)-X(po)}% )`;bar.style.bottom=b+'px';tr.appendChild(bar);
      const lab=document.createElement('span');lab.className='px-lab '+c;lab.innerHTML=`${name} <span class="v">${pm<.05?pm.toFixed(3):pm.toFixed(2)}</span>`;lab.style.left=X(pm)+'%';tr.appendChild(lab);const st=document.createElement('span');st.className='px-stem';st.style.left=X(pm)+'%';tr.appendChild(st);items.push({lab,st,b})})});
  const lb=document.createElement('span');lb.className='px-lab';lb.style.cssText='left:0;bottom:-30px;font-size:12px;color:var(--ink-3);background:none';lb.textContent='舊機率（對數）· 圓點為舊機率，線段為允許的新機率（PPO ε = 0.2，RIC δ = 0.02）';tr.appendChild(lb);
  layoutHero(node,items)}
function layoutHero(node,items){if(!items)items=node._items;node._items=items;const tr=node.firstChild;const base=56,rowH=22;
  items.forEach(it=>{it.lab.style.bottom=base+'px';it.lab.style.transform='translateX(-50%)'});
  const tw=tr.getBoundingClientRect();items.forEach(it=>{const r=it.lab.getBoundingClientRect();if(r.right>tw.right)it.lab.style.transform='translateX(-100%)';if(r.left<tw.left)it.lab.style.transform='translateX(0)'});
  const s=items.map(it=>({it,r:it.lab.getBoundingClientRect()})).sort((a,b)=>a.r.left-b.r.left);const rows=[];let mx=0;
  s.forEach(o=>{let r=0;while(rows[r]!==undefined&&o.r.left<rows[r]+8)r++;rows[r]=o.r.right;mx=Math.max(mx,r);const bt=base+r*rowH;o.it.lab.style.bottom=bt+'px';o.it.st.style.bottom=(o.it.b+2)+'px';o.it.st.style.height=(bt-o.it.b-2)+'px'});
  tr.style.marginTop=(mx*rowH+base+8)+'px';tr.style.marginBottom='36px'}

/* ---------- timeline ---------- */
const TL_GROUP={'1502.05477':['tr','trpo-monotonic-improvement-tv-bound'],'1506.02438':['tr','trpo-monotonic-improvement-tv-bound'],
 '1707.06347':['fix','ppo-grpo-dapo-llm-rl-objectives'],'2402.03300':['fix','ppo-grpo-dapo-llm-rl-objectives'],'2501.12948':['fix','ppo-grpo-dapo-llm-rl-objectives'],
 '2503.14476':['mod','llm-rl-clipping-variants-heuristics'],'2505.22617':['mod','llm-rl-clipping-variants-heuristics'],'2509.02333':['mod','llm-rl-clipping-variants-heuristics'],'2508.07629':['mod','ripo-experiments-results'],
 '2507.18071':['seq','sequence-level-ratio-gspo-gmpo'],'2507.20673':['seq','sequence-level-ratio-gspo-gmpo'],'2409.19256':['plat','ripo-experiments-results'],'2607.10169':['me','ripo-isometric-clip-ric']};
const TL_NAME={tr:'trust region',fix:'固定 clip',mod:'clip 修正',seq:'序列層級',plat:'平台',me:'依機率的 clip · 本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  if(!data.some(e=>e.arxiv==='2607.10169'))data.push({year:2026,title:'Beyond Euclidean Clipping',authors_short:'本論文',arxiv:'2607.10169',role:'等距 clip ε = √(δ/π_old)'});
  data=data.map((e,i)=>({e,i})).sort((a,b)=>(+a.e.year||0)-(+b.e.year||0)||a.i-b.i).map(o=>o.e);
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];used.add(g[0]);const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${g[1]?`<a href="${g[1]}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
  const on=new Set(used);
  Object.keys(TL_NAME).filter(k=>used.has(k)).forEach(k=>{const b=document.createElement('button');b.type='button';b.className='chip';b.setAttribute('aria-pressed','true');b.textContent=TL_NAME[k].split(' · ')[0];b.dataset.g=k;
    b.addEventListener('click',()=>{if(on.has(k)&&on.size===used.size){on.clear();on.add(k)}else if(on.has(k)){on.delete(k);if(!on.size)used.forEach(x=>on.add(x))}else on.add(k);
      $$('.chip',chips).forEach(c=>c.setAttribute('aria-pressed',String(on.has(c.dataset.g))));$$('.ev',track).forEach(ev=>ev.classList.toggle('dim',!on.has(ev.dataset.g)))});chips.appendChild(b)});
  const rail=$('#rail'),by=dir=>rail.scrollBy({left:dir*Math.min(rail.clientWidth*.8,560),behavior:RM?'auto':'smooth'});
  $('#tl-prev').addEventListener('click',()=>by(-1));$('#tl-next').addEventListener('click',()=>by(1));
  rail.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){by(1);e.preventDefault()}if(e.key==='ArrowLeft'){by(-1);e.preventDefault()}})}

/* ---------- toc, theme, appbar ---------- */
function initToc(){const secs=$$('[data-toc]'),ol=$('#toc'),bar=$('#tocbar');
  secs.forEach(s=>{ol.insertAdjacentHTML('beforeend',`<li><a href="#${s.id}"><span class="n">${s.dataset.toc}</span><span>${s.dataset.tt}</span></a></li>`);bar.insertAdjacentHTML('beforeend',`<a href="#${s.id}">${s.dataset.toc} ${s.dataset.tt}</a>`)});
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;$$('.toc a,.tocbar a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+id));const ab=$(`.tocbar a[href="#${id}"]`);if(ab&&bar.offsetParent)bar.scrollTo({left:ab.offsetLeft-16,behavior:RM?'auto':'smooth'})}})},{rootMargin:'-40% 0px -55% 0px'});
  secs.forEach(s=>io.observe(s));
  addEventListener('scroll',()=>$('#appbar').classList.toggle('raised',scrollY>8),{passive:true})}
function initTheme(){let saved=null;try{saved=localStorage.getItem('pm-theme')}catch(e){}if(saved)document.documentElement.dataset.theme=saved;
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){}})}

/* ---------- self test ---------- */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t)=>Math.abs(a-b)<=t+1e-12,P2={eps:.2,epsH:.28,delta:.02},P5={eps:.2,epsH:.28,delta:.05};
  const pm=(m,po,P)=>measures(po,bounds(m,po,P).rhi);
  ok(near(pm('PPO',.8,P2).pmax,.96,1e-9),'PPO 0.8');ok(near(pm('PPO',.01,P2).pmax,.012,1e-9),'PPO 0.01');ok(near(pm('DAPO',.01,P2).pmax,.0128,1e-9),'DAPO 0.01');ok(near(pm('DAPO',.8,P2).pmax,1,1e-9),'DAPO 0.8');
  ok(near(pm('RIC',.8,P2).pmax,.92649,5e-5),'RIC 0.8');ok(near(pm('RIC',.01,P2).pmax,.024142,5e-6),'RIC 0.01');
  ok(near(pm('PPO',.8,P2).atom,.016,1e-9)&&near(pm('PPO',.01,P2).atom,.0002,1e-12),'atom PPO');
  [.001,.01,.1,.5].forEach(po=>ok(near(pm('RIC',po,P2).atom,.01,1e-9),'atom RIC '+po));
  ok(near(pm('RIC',.01,P2).binkl,.005430,1e-6),'binKL RIC 0.01');ok(pm('DAPO',.8,P2).binkl===Infinity,'binKL DAPO 0.8 inf');ok(near(pm('RIC',1e-4,P2).binkl,.001143,1e-6),'binKL 1e-4');
  ok(near(pm('RIC',.8,P5).pmax,1,1e-9)&&near(pm('RIC',.5,P5).pmax,.6581,5e-5)&&near(pm('RIC',.01,P5).binchi,.025253,1e-6),'delta 0.05 rows');
  const w=rowW2(.001,10,{eps:.2,delta:.05,dual:false});ok(near(w.RIC.w,8.0711,5e-5)&&near(w.RIC.v,.065142,1e-6)&&near(w.RIC.bias,.001929,1e-6)&&near(w.PPO.v,.00144,1e-9)&&near(w.IS.v,.1,1e-9),'W2 default');
  const w2d=rowW2(1e-4,100,{eps:.2,delta:.05,dual:true});ok(near(w2d.RIC.v,.01,1e-9)&&near(w2d.IS.bias,.009,1e-9),'W2 dual');
  ok(near(1e-5*(1+Math.sqrt(.05/1e-5))**2,.051424,1e-6),'W2 envelope');
  const b1=batch({method:'ppo',eps:.2},.01),b2=batch({method:'ric',delta:.05},.01),b3=batch({method:'none'},.01);
  ok(near(b1.tr.found[200],.015,.02)&&near(b1.tr.skipped[0],677,20),'W3 GRPO '+b1.tr.found[200]+' skipped '+b1.tr.skipped[0]);ok(near(b2.tr.found[200],.37,.02),'W3 RIC '+b2.tr.found[200]);ok(near(b3.tr.found[200],.395,.02),'W3 none '+b3.tr.found[200]);
  ok(T1['Qwen3-8B-Base'].RIPO[7]===38.5&&T1['Qwen3-8B-Base'].GRPO[7]===28.5&&T1['Qwen3-8B-Base'].RIPO[0]===43.8&&T2[2][2]===43.8&&T5['AIME-25'].RIPO[6]===60&&T5['HMMT-25'].RIPO[6]===45.3,'W4 data');
  ok(document.querySelectorAll('.katex-error').length===0,'no katex-error');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ---------- boot ---------- */
initTheme();initToc();initW1();initW2();initW3();initW4();renderTimeline();heroAxis();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderW1();renderW2();renderW3();renderW4();layoutHero($('#pax-hero'))})});
document.fonts&&document.fonts.ready.then(()=>layoutHero($('#pax-hero')));
const MOPT={delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false};
function renderMath(node){if(typeof renderMathInElement!=='function')return;renderMathInElement(node||document.body,MOPT)}
$$('details').forEach(d=>d.addEventListener('toggle',()=>{if(d.open&&!d.querySelector('.katex'))renderMath(d)}));
let tested=false;const runTest=()=>{if(tested||!/[?&]test\b/.test(location.search))return;tested=true;selfTest()};
if(document.readyState==='loading')addEventListener('DOMContentLoaded',()=>renderMath());else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath();runTest()});
})();
