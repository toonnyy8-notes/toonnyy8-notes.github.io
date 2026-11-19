(()=>{
'use strict';
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
function el(tag,a,p){const e=document.createElementNS(NS,tag);if(a)for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function tx(p,x,y,s,a){const t=el('text',Object.assign({x,y},a||{}),p);t.textContent=s;return t}
const lin=(d0,d1,r0,r1)=>v=>r0+(v-d0)/(d1-d0)*(r1-r0);
function lp(pts){let s='';for(let i=0;i<pts.length;i++)s+=(i?'L':'M')+pts[i][0].toFixed(1)+' '+pts[i][1].toFixed(1);return s}
function widthOf(box,min,max){return Math.max(min,Math.min(max,Math.floor(box.clientWidth||max)))}
function sizeSvg(svg,W,H){svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H)}
function frame(svg,o){
  sizeSvg(svg,o.W,o.H);
  const m=o.m,x0=m.l,x1=o.W-m.r,y0=o.H-m.b,y1=m.t;
  const X=lin(o.x[0],o.x[1],x0,x1),Y=lin(o.y[0],o.y[1],y0,y1);
  const g=el('g',null,svg);
  (o.yt||[]).forEach(v=>{const yy=Y(v);el('line',{x1:x0,x2:x1,y1:yy,y2:yy,class:'grid'},g);tx(g,x0-8,yy+4,(o.yf||String)(v),{class:'tk','text-anchor':'end'})});
  (o.xt||[]).forEach(v=>{const xx=X(v);el('line',{x1:xx,x2:xx,y1:y0,y2:y0+4,class:'axl'},g);tx(g,xx,y0+18,(o.xf||String)(v),{class:'tk','text-anchor':'middle'})});
  el('line',{x1:x0,x2:x1,y1:y0,y2:y0,class:'axl'},g);
  if(o.xl)tx(g,(x0+x1)/2,o.H-4,o.xl,{class:'al','text-anchor':'middle'});
  if(o.yl)tx(g,4,12,o.yl,{class:'al'});
  return {X,Y,g,x0,x1,y0,y1};
}
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}
const pct=v=>(v*100).toFixed(1)+'%';
function segInit(box,vals,cur,label,on){box.innerHTML='';vals.forEach(v=>{const b=document.createElement('button');b.type='button';b.dataset.v=v;b.textContent=label?label(v):String(v);b.setAttribute('aria-pressed',String(v===cur));b.addEventListener('click',()=>{$$('button',box).forEach(x=>x.setAttribute('aria-pressed',String(x===b)));on(v)});box.appendChild(b)})}
function segSet(box,cur){$$('button',box).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===String(cur))))}
/* markers: hollow/filled circle, square, diamond, triangle */
function marker(g,shape,x,y,r,cls){
  if(shape==='circle')return el('circle',{cx:x,cy:y,r,class:cls},g);
  if(shape==='square')return el('rect',{x:x-r,y:y-r,width:2*r,height:2*r,class:cls},g);
  if(shape==='diamond')return el('path',{d:`M${x} ${y-r*1.3}L${x+r*1.3} ${y}L${x} ${y+r*1.3}L${x-r*1.3} ${y}Z`,class:cls},g);
  return el('path',{d:`M${x} ${y-r*1.2}L${x+r*1.1} ${y+r*.8}L${x-r*1.1} ${y+r*.8}Z`,class:cls},g)}

/* ================= toy (design 4.0; same as verify/toy.py) ================= */
const sigmoid=z=>z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));
const kfun=t=>t/(2*(1-t)**2);
function flowU(u,s,t,maxSteps){maxSteps=maxSteps||20000;
  if(u===0||t<=s)return{u,failed:false};
  const sg=u>0?1:-1,rt=t>=1?Infinity:-Math.log(1-t);let r=-Math.log(1-s),steps=0;
  const f=(rr,v)=>2*Math.tanh(kfun(1-Math.exp(-rr))*v)-v;
  while(r<rt){const k=kfun(1-Math.exp(-r));
    if(Math.abs(k*u)>20){const dr=rt-r;return{u:2*sg+(u-2*sg)*(dr===Infinity?0:Math.exp(-dr)),failed:false}}
    if(steps>=maxSteps)return{u,failed:true};
    const h=Math.min(.01,.5/(2*k+1),rt-r),k1=f(r,u),k2=f(r+h/2,u+h/2*k1),k3=f(r+h/2,u+h/2*k2),k4=f(r+h,u+h*k3);
    u=u+h/6*(k1+2*k2+2*k3+k4);if(u*sg<=0)return{u:0,failed:false};r+=h;steps++}
  return{u,failed:false}}
const toC=x=>{const a1=x[0]-x[1],a2=x[2]-x[3];return[a1+a2,a1-a2,x[0]+x[1],x[2]+x[3]]};
const fromC=(u,w,s1,s2)=>{const a1=(u+w)/2,a2=(u-w)/2;return[(s1+a1)/2,(s1-a1)/2,(s2+a2)/2,(s2-a2)/2]};
function Xm(x,s,t){const[u,w,s1,s2]=toC(x),r=flowU(u,s,t),c=(1-t)/(1-s),y=fromC(r.u,w*c,1-(1-s1)*c,1-(1-s2)*c);y.failed=r.failed;return y}
function pNY(x,t){const u=toC(x)[0];if(t>=1)return u>0?1:u<0?0:.5;return sigmoid(t*u/(1-t)**2)}
function Dn(x,t){const p=pNY(x,t);return[p,1-p,p,1-p]}
function delta(x,s,t){if(t-s<1e-12)return Dn(x,s);const Y=Xm(x,s,t);return Y.map((v,i)=>((1-s)*v-(1-t)*x[i])/(t-s))}
const gamma=(s,u,t)=>t>=1?0:(1-t)*(u-s)/((1-u)*(t-s));
function euler(x,N){for(let n=0;n<N;n++){const t0=n/N,t1=(n+1)/N,d=Dn(x,t0);x=x.map((v,i)=>v+(t1-t0)*(d[i]-v)/(1-t0))}return x}
function flowmap(x,N){let failed=false;for(let n=0;n<N;n++){x=Xm(x,n/N,(n+1)/N);failed=failed||x.failed}x.failed=failed;return x}
function classify(x){const d1=x[0]-x[1],d2=x[2]-x[3];if(Math.abs(d1)<1e-12||Math.abs(d2)<1e-12)return'tie';return d1>0?(d2>0?'NY':'ND'):(d2>0?'SY':'SD')}
function distOnehot(x){const o1=x[0]>=x[1],o2=x[2]>=x[3],o=[+o1,+!o1,+o2,+!o2];let s=0;for(let i=0;i<4;i++)s+=(x[i]-o[i])**2;return Math.sqrt(s)}
function samples(seed,n){const r=mulberry32(seed),xs=[];for(let i=0;i<n;i++)xs.push([gauss(r),gauss(r),gauss(r),gauss(r)]);return xs}
function highlightIndex(xs){for(let i=0;i<xs.length;i++){const a1=xs[i][0]-xs[i][1],a2=xs[i][2]-xs[i][3];if(Math.abs(a1+a2)>.5&&a1*a2<0)return i}return 0}
/* discrete masked-diffusion toy (C1a) */
function discreteClosed(N){const inv=1/(2*N);return{NY:(1-inv)/2,SD:(1-inv)/2,ND:inv/2,SY:inv/2,invalid:inv}}
function discreteSample(N,r){let y=[null,null];for(let n=0;n<N;n++){const s=n/N,t=(n+1)/N,p=(t-s)/(1-s),un=[y[0]===null&&r()<p,y[1]===null&&r()<p],ys=y.slice(),nw=y.slice();
  for(let l=0;l<2;l++)if(un[l]){const o=ys[1-l];nw[l]=o===null?r()<.5:o}y=nw}return y}
const W1=['san','new'],W2=['diego','york'];

/* ================= C1a ================= */
const c1a={N:1,seed:101};
function renderC1a(){const N=c1a.N,c=discreteClosed(N),box=$('#c1a-fig'),svg=$('svg',box),W=widthOf(box,300,860),rowH=38,H=4*rowH+8;sizeSvg(svg,W,H);
  const rows=[['new york',c.NY,0],['san diego',c.SD,0],['new diego',c.ND,1],['san york',c.SY,1]],lw=Math.min(150,W*.32),bx=lw,bw=W-lw-64;
  rows.forEach(([name,v,bad],i)=>{const y=4+i*rowH;tx(svg,0,y+22,name+(bad?'（無效）':''),{class:bad?'al':'lab'});
    el('rect',{x:bx,y:y+6,width:bw,height:22,rx:4,class:'track'},svg);
    const w=Math.max(0,bw*v);el('rect',{x:bx,y:y+6,width:w,height:22,rx:4,class:bad?'bar-x':'bar-v'},svg);
    if(bad)el('rect',{x:bx,y:y+6,width:w,height:22,rx:4,class:'hatch'},svg);
    tx(svg,bx+w+8,y+22,pct(v),{class:'labm'})});
  svg.setAttribute('aria-label',`N = ${N}：new york ${pct(c.NY)}、san diego ${pct(c.SD)}、new diego ${pct(c.ND)}、san york ${pct(c.SY)}（後兩者無效）`);
  $('#c1a-ro').textContent=`N = ${N}　無效率 = 1/(2N) = ${pct(c.invalid)}`;
  const r=mulberry32(c1a.seed),sbox=$('#c1a-s');sbox.innerHTML='';
  for(let i=0;i<16;i++){const y=discreteSample(N,r),bad=y[0]!==y[1],sp=document.createElement('span');sp.className='samp'+(bad?' bad':'');
    const words=W1[+y[0]]+' '+W2[+y[1]];sp.innerHTML=bad?`<s>${words}</s><em>無效</em>`:words;sbox.appendChild(sp)}}
function initC1a(){segInit($('#c1a-n'),[1,2,4,8,16,32,64],1,v=>'N = '+v,v=>{c1a.N=v;renderC1a()});
  $('#c1a-re').addEventListener('click',()=>{c1a.seed++;renderC1a()});renderC1a()}

/* ================= C1b ================= */
const c1b={sampler:'euler',N:1,seed:7,xs:[],hi:0,cache:new Map()};
function c1bSample(){c1b.xs=samples(c1b.seed,200);c1b.hi=highlightIndex(c1b.xs);c1b.cache.clear()}
function c1bEnds(sampler,N){const key=sampler+'|'+N;if(c1b.cache.has(key))return c1b.cache.get(key);
  const ends=c1b.xs.map(x=>sampler==='euler'?euler(x,N):flowmap(x,N));const cnt={NY:0,SD:0,ND:0,SY:0,tie:0,fail:0};let md=0;
  ends.forEach(e=>{if(e.failed)cnt.fail++;else cnt[classify(e)]++;md+=distOnehot(e)});
  const r={ends,cnt,md:md/ends.length};c1b.cache.set(key,r);return r}
function renderC1b(){const box=$('#c1b-fig'),svg=$('svg',box),W=widthOf(box,280,520),m={l:14,r:14,t:28,b:40},P=W-m.l-m.r,H=P+m.t+m.b;sizeSvg(svg,W,H);
  const S=lin(-4,4,m.l,m.l+P),SY=lin(-4,4,m.t+P,m.t),g=el('g',null,svg),cl=v=>Math.max(-4,Math.min(4,v));
  el('rect',{x:S(0),y:SY(0),width:S(4)-S(0),height:SY(-4)-SY(0),class:'hatch'},g);
  el('rect',{x:S(-4),y:SY(4),width:S(0)-S(-4),height:SY(0)-SY(4),class:'hatch'},g);
  for(let v=-3;v<=3;v++){if(!v)continue;el('line',{x1:S(v),x2:S(v),y1:SY(-4),y2:SY(4),class:'faint'},g);el('line',{x1:S(-4),x2:S(4),y1:SY(v),y2:SY(v),class:'faint'},g)}
  el('rect',{x:S(-4),y:SY(4),width:P,height:P,fill:'none',class:'axl'},g);
  el('line',{x1:S(0),x2:S(0),y1:SY(-4),y2:SY(4),class:'axl'},g);el('line',{x1:S(-4),x2:S(4),y1:SY(0),y2:SY(0),class:'axl'},g);
  tx(g,S(3.85),SY(3.85)+12,'new york',{class:'lab','text-anchor':'end'});tx(g,S(-3.85),SY(-3.85)-6,'san diego',{class:'lab'});
  tx(g,S(3.85),SY(-3.85)-6,'new diego（無效）',{class:'al','text-anchor':'end'});tx(g,S(-3.85),SY(3.85)+12,'san york（無效）',{class:'al'});
  [-4,-2,0,2,4].forEach(v=>tx(g,S(v),m.t+P+14,String(v),{class:'tk','text-anchor':'middle'}));
  tx(g,m.l+P/2,H-4,'a₁ = x¹ new − x¹ san',{class:'al','text-anchor':'middle'});tx(g,m.l,16,'a₂ = x² york − x² diego',{class:'al'});
  [[1,1],[-1,-1]].forEach(([a,b])=>{const X=S(a),Y=SY(b);el('path',{d:`M${X-6} ${Y-6}L${X+6} ${Y+6}M${X+6} ${Y-6}L${X-6} ${Y+6}`,stroke:'var(--ink)','stroke-width':2.5},g)});
  const A=x=>[x[0]-x[1],x[2]-x[3]];
  const pt=(a,cls,r)=>{const out=Math.abs(a[0])>4||Math.abs(a[1])>4,X=S(cl(a[0])),Y=SY(cl(a[1]));return out?marker(g,'tri',X,Y,r+1,cls):el('circle',{cx:X,cy:Y,r,class:cls},g)};
  c1b.xs.forEach(x=>pt(A(x),'startp',2.4));
  const R=c1bEnds(c1b.sampler,c1b.N);R.ends.forEach(e=>pt(A(e),'endp',3.2));if(R.cnt.tie)tx(g,S(0)+8,SY(0)+18,`平手 × ${R.cnt.tie}（終點在原點）`,{class:'labm'});
  // highlighted sample
  const x0=c1b.xs[c1b.hi],traj=[A(x0)];let cur=x0;for(let i=1;i<=200;i++){cur=Xm(cur,(i-1)/200,i/200);traj.push(A(cur))}
  el('path',{d:lp(traj.map(a=>[S(cl(a[0])),SY(cl(a[1]))])),class:'traj'},g);
  const its=[A(x0)];let xi=x0;for(let n=0;n<c1b.N;n++){const t0=n/c1b.N,t1=(n+1)/c1b.N;
    if(c1b.sampler==='euler'){const d=Dn(xi,t0),p=xi;xi=p.map((v,i)=>v+(t1-t0)*(d[i]-v)/(1-t0))}else xi=Xm(xi,t0,t1);its.push(A(xi))}
  if(c1b.sampler==='euler')el('path',{d:lp(its.map(a=>[S(cl(a[0])),SY(cl(a[1]))])),class:'acc',style:'stroke-width:1.6'},g);
  its.slice(1).forEach(a=>{const X=S(cl(a[0])),Y=SY(cl(a[1]));if(c1b.sampler==='euler')el('circle',{cx:X,cy:Y,r:4.5,class:'hollow',style:'stroke:var(--accent)'},g);else marker(g,'square',X,Y,4,'acc-f')});
  el('circle',{cx:S(cl(its[0][0])),cy:SY(cl(its[0][1])),r:5.5,fill:'none',stroke:'var(--accent)','stroke-width':2},g);
  const c=R.cnt,dc=discreteClosed(c1b.N),nm=c1b.sampler==='euler'?'Euler':'flow map';
  $('#c1b-ro').innerHTML=`${nm}，N = ${c1b.N}（200 個樣本）<br>new york <b>${c.NY}</b>　san diego <b>${c.SD}</b><br>new diego <b>${c.ND}</b>　san york <b>${c.SY}</b>　平手 <b>${c.tie}</b>${c.fail?`<br>數值不收斂 <b>${c.fail}</b> 個`:''}<br>平均 ‖x̂₁ − one-hot‖ = <b>${R.md.toFixed(3)}</b><br>離散（C1a，同 N）無效率 = <b>${pct(dc.invalid)}</b><br>突顯樣本：索引 ${c1b.hi}，終點 ${({NY:'new york',SD:'san diego',ND:'new diego',SY:'san york',tie:'平手'})[classify(R.ends[c1b.hi])]}`;
  svg.setAttribute('aria-label',`${nm}，N = ${c1b.N}：new york ${c.NY}、san diego ${c.SD}、無效 ${c.ND+c.SY}、平手 ${c.tie}`+(c.tie===200?'：200 個樣本全部平手':''));
  svg.onclick=e=>{const rc=svg.getBoundingClientRect(),sc=W/rc.width,px=(e.clientX-rc.left)*sc,py=(e.clientY-rc.top)*sc;let bi=0,bd=Infinity;
    c1b.xs.forEach((x,i)=>{const a=A(x),d=(S(cl(a[0]))-px)**2+(SY(cl(a[1]))-py)**2;if(d<bd){bd=d;bi=i}});c1b.hi=bi;renderC1b()}}
function initC1b(){c1bSample();const sb=$('#c1b-s');$$('button',sb).forEach(b=>b.addEventListener('click',()=>{c1b.sampler=b.dataset.v;segSet(sb,c1b.sampler);renderC1b()}));segSet(sb,c1b.sampler);
  segInit($('#c1b-n'),[1,2,4,8,16,32],1,v=>'N = '+v,v=>{c1b.N=v;renderC1b()});
  $('#c1b-hi').addEventListener('click',()=>{c1b.hi=(c1b.hi+1)%c1b.xs.length;renderC1b()});
  $('#c1b-re').addEventListener('click',()=>{c1b.seed++;c1bSample();renderC1b()});renderC1b()}

/* ================= C2 ================= */
const C2DEF={s:.1,t:.9,u:.5,mid:true,x0:[.9,-.2,-.4,.3]};
const c2=Object.assign({seed:11},JSON.parse(JSON.stringify(C2DEF)));
function eq82(x,s,t,n){const h=(t-s)/n,pts=[];let acc=0,xt=x;for(let i=0;i<=n;i++){const tau=s+i*h;if(i)xt=Xm(xt,s+(i-1)*h,tau);const p=pNY(xt,tau);pts.push([tau,p]);acc+=p/(1-tau)**2*(i===0||i===n?1:i%2?4:2)}return{q:(1-s)*(1-t)/(t-s)*acc*h/3,pts}}
function c2compute(s,u,t,x0,withQ){const x=Xm(x0,0,s),Xsu=Xm(x,s,u),Xst=Xm(x,s,t),dsu=delta(x,s,u),dst=delta(x,s,t),dut=delta(Xsu,u,t),g=gamma(s,u,t),teach=dsu.map((v,i)=>g*v+(1-g)*dut[i]);
  const tdiff=Math.max(...dst.map((v,i)=>Math.abs(v-teach[i])));const traj=[];for(let i=0;i<=60;i++)traj.push(Xm(x,s,s+(t-s)*i/60));
  return{x,Xsu,Xst,dsu,dst,dut,g,teach,tdiff,traj,Ds:Dn(x,s),e82:withQ&&t<1?eq82(x,s,t,400):null}}
const MK=[{k:'x',name:'x_s（時間 s 的狀態）',short:'x s',shape:'circle',cls:'hollow',adv:0},{k:'Xsu',name:'X s,u',short:'X s,u',shape:'square',cls:'hollow',adv:1},{k:'Xst',name:'X s,t',short:'X s,t',shape:'diamond',cls:'hollow',adv:0},
  {k:'dsu',name:'δ s,u',short:'δ s,u',shape:'square',cls:'ink-f',adv:1},{k:'dut',name:'δ u,t(X s,u)',short:'δ u,t',shape:'tri',cls:'ink-f',adv:1},{k:'dst',name:'δ s,t',short:'δ s,t',shape:'diamond',cls:'acc-f',adv:0}];
function c2panel(box,R,pos,adv){const svg=$('svg',box),W=widthOf(box,240,420),m={l:34,r:12,t:12,b:34},P=W-m.l-m.r,H=P+m.t+m.b;sizeSvg(svg,W,H);
  const S=lin(-1.5,2,m.l,m.l+P),SY=lin(-1.5,2,m.t+P,m.t),g=el('g',null,svg),i0=pos*2,cl=v=>Math.max(-1.5,Math.min(2,v)),narrow=W<300;
  [-1,0,1,2].forEach(v=>{el('line',{x1:S(v),x2:S(v),y1:SY(-1.5),y2:SY(2),class:v?'faint':'axl'},g);el('line',{x1:S(-1.5),x2:S(2),y1:SY(v),y2:SY(v),class:v?'faint':'axl'},g);
    tx(g,S(v),m.t+P+14,String(v),{class:'tk','text-anchor':'middle'});tx(g,m.l-6,SY(v)+4,String(v),{class:'tk','text-anchor':'end'})});
  const names=pos?['x² york','x² diego']:['x¹ new','x¹ san'];
  tx(g,m.l+P/2,H-4,names[0],{class:'al','text-anchor':'middle'});tx(g,m.l+4,m.t+12,names[1],{class:'al'});
  el('line',{x1:S(1),y1:SY(0),x2:S(0),y2:SY(1),class:'simplex'},g);tx(g,S(.1),SY(1)-8,'simplex',{class:'al'});
  el('path',{d:lp(R.traj.map(v=>[S(cl(v[i0])),SY(cl(v[i0+1]))])),class:'trajd'},g);
  const labs=[];
  MK.forEach((mk,j)=>{if(mk.adv&&!adv)return;const v=R[mk.k],a=v[i0],b=v[i0+1],out=a<-1.5||a>2||b<-1.5||b>2,X=S(cl(a)),Y=SY(cl(b));
    marker(g,out?'tri':mk.shape,X,Y,mk.k==='dst'?5.5:4.5,mk.cls);
    if(mk.k==='dst'&&adv)el('circle',{cx:S(cl(R.teach[i0])),cy:SY(cl(R.teach[i0+1])),r:10,fill:'none',stroke:'var(--accent)','stroke-width':1.5},g);
    let lab=narrow?String(j+1)+' '+mk.short:mk.short;if(adv&&!narrow&&mk.k[0]==='d')lab+=' '+v[i0].toFixed(3);labs.push({X,Y,lab,acc:mk.k==='dst'})});
  // ponytail: greedy 1-D label stacking (min gap 15px); fine for ≤ 6 labels
  const LX=S(1.3);labs.sort((p,q)=>p.Y-q.Y);let last=-1e9;labs.forEach(l=>{l.ly=Math.max(l.Y,last+15);last=l.ly});
  const over=last-(m.t+P-4);if(over>0)labs.forEach(l=>l.ly-=over);
  labs.forEach(l=>{el('path',{d:`M${l.X+6} ${l.Y}L${LX-4} ${l.ly-4}`,class:'cur'},g);tx(g,LX,l.ly,l.lab,{class:'lab-s',style:l.acc?'fill:var(--accent)':''})});
  svg.setAttribute('aria-label',`位置 ${pos+1}：X s,t 座標和 ${(R.Xst[i0]+R.Xst[i0+1]).toFixed(2)}，δ s,t = (${R.dst[i0].toFixed(3)}, ${R.dst[i0+1].toFixed(3)})，座標和 1`)}
function c2eqPanel(R,s,t){const box=$('#c2-p3'),svg=$('svg',box),W=widthOf(box,280,760);
  if(!R.e82){sizeSvg(svg,W,40);tx(svg,0,24,'t = 1 時 δ s,1 = X s,1，權重集中在 τ → 1',{class:'al'});return}
  const F=frame(svg,{W,H:220,m:{l:40,r:14,t:16,b:34},x:[s,t],y:[0,1],yt:[0,.5,1],xt:[s,(s+t)/2,t],xf:v=>v.toFixed(2),yf:v=>String(v),xl:'τ'});
  const wmax=1/(1-t)**2,pts=[];for(let i=0;i<=100;i++){const tau=s+(t-s)*i/100;pts.push([F.X(tau),F.Y((1/(1-tau)**2)/wmax)])}
  const area=lp(pts)+`L${F.X(t)} ${F.Y(0)}L${F.X(s)} ${F.Y(0)}Z`;el('path',{d:area,class:'wfill'},F.g);el('path',{d:area,class:'hatch'},F.g);
  el('path',{d:lp(R.e82.pts.map(p=>[F.X(p[0]),F.Y(p[1])])),class:'par'},F.g);
  const q=R.e82.q;el('line',{x1:F.x0,x2:F.x1,y1:F.Y(q),y2:F.Y(q),class:'acc',style:'stroke-dasharray:6 5;stroke-width:2'},F.g);
  tx(F.g,F.x0+6,F.Y(q)+(q>.85?16:-6),`δ s,t 的 new 機率 ${q.toFixed(4)}（反解值 ${R.dst[0].toFixed(4)}）`,{class:'labm',style:'fill:var(--accent)'})}
function renderC2(){const{s,t}=c2,adv=$('#c2-adv').open,u=c2.mid?(s+t)/2:c2.u;c2.uEff=u;
  $('#c2-so').textContent=s.toFixed(2);$('#c2-to').textContent=t.toFixed(2);$('#c2-uo').textContent=u.toFixed(2);
  $('#c2-s').value=s;$('#c2-t').value=t;$('#c2-u').value=u;$('#c2-u').disabled=c2.mid;
  $('#c2-s').setAttribute('aria-valuetext','s = '+s.toFixed(2));$('#c2-t').setAttribute('aria-valuetext','t = '+t.toFixed(2));$('#c2-u').setAttribute('aria-valuetext','u = '+u.toFixed(2));
  const R=c2compute(s,u,t,c2.x0,adv);c2panel($('#c2-p1'),R,0,adv);c2panel($('#c2-p2'),R,1,adv);
  const narrow=widthOf($('#c2-p1'),240,420)<300;
  $('#c2-lg').innerHTML=MK.filter(mk=>adv||!mk.adv).map(mk=>{const j=MK.indexOf(mk)+1,f=mk.cls==='hollow'?'fill="var(--surface)" stroke="var(--ink)" stroke-width="1.5"':mk.cls==='acc-f'?'fill="var(--accent)"':'fill="var(--ink)"';
    const sh=mk.shape==='circle'?`<circle cx="8" cy="8" r="4.5" ${f}/>`:mk.shape==='square'?`<rect x="3.5" y="3.5" width="9" height="9" ${f}/>`:mk.shape==='diamond'?`<path d="M8 2L14 8L8 14L2 8Z" ${f}/>`:`<path d="M8 2.5L13.5 13L2.5 13Z" ${f}/>`;
    return `<li><svg viewBox="0 0 16 16">${sh}</svg>${narrow?`<span class="num">${j}</span>`:''}${mk.name}</li>`}).join('')+(adv?'<li><svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6" fill="none" stroke="var(--accent)" stroke-width="1.5"/></svg>半群教師（圓環）</li>':'')+'<li><svg viewBox="0 0 16 16"><line x1="1" y1="8" x2="15" y2="8" stroke="var(--ink-3)" stroke-width="1.5" stroke-dasharray="3 3"/></svg>軌跡 X s,τ(x_s)</li>';
  const sum=(v,i)=>(v[i]+v[i+1]).toFixed(4);
  $('#c2-ro').innerHTML=`X s,t 座標和：位置 1 <b>${sum(R.Xst,0)}</b>　位置 2 <b>${sum(R.Xst,2)}</b><br>δ s,t 座標和（= 1）：位置 1 <b>${sum(R.dst,0)}</b>　位置 2 <b>${sum(R.dst,2)}</b>`;
  $('#c2-ro2').innerHTML=`γ = <b>${R.g.toFixed(4)}</b><br>X s,u 座標和：位置 1 <b>${sum(R.Xsu,0)}</b>　位置 2 <b>${sum(R.Xsu,2)}</b><br>|δ s,t − 教師| 最大值 = <b>${R.tdiff.toExponential(1)}</b>`;
  if(adv)c2eqPanel(R,s,t);
  announce('c2',`s = ${s.toFixed(2)}，t = ${t.toFixed(2)}：X s,t 座標和 ${sum(R.Xst,0)} 與 ${sum(R.Xst,2)}，δ s,t 座標和 1；γ = ${R.g.toFixed(4)}`)}
function initC2(){const r2=v=>Math.round(v*100)/100;
  $('#c2-s').addEventListener('input',e=>{c2.s=Math.min(.97,r2(+e.target.value));c2.t=Math.max(c2.t,r2(c2.s+.02));if(!c2.mid)c2.u=Math.min(Math.max(c2.u,c2.s),c2.t,.99);renderC2()});
  $('#c2-t').addEventListener('input',e=>{c2.t=Math.max(.02,r2(+e.target.value));c2.s=Math.min(c2.s,r2(c2.t-.02));if(!c2.mid)c2.u=Math.min(Math.max(c2.u,c2.s),c2.t,.99);renderC2()});
  $('#c2-u').addEventListener('input',e=>{c2.u=Math.min(Math.max(r2(+e.target.value),c2.s),c2.t,.99);renderC2()});
  $('#c2-mid').addEventListener('change',e=>{c2.mid=e.target.checked;if(!c2.mid)c2.u=Math.min(Math.max(r2((c2.s+c2.t)/2),c2.s),c2.t,.99);renderC2()});
  $('#c2-adv').addEventListener('toggle',renderC2);
  $('#c2-re').addEventListener('click',()=>{const r=mulberry32(c2.seed++);c2.x0=[gauss(r),gauss(r),gauss(r),gauss(r)];renderC2()});
  $('#c2-reset').addEventListener('click',()=>{Object.assign(c2,JSON.parse(JSON.stringify(C2DEF)));c2.seed=11;$('#c2-mid').checked=true;renderC2()});renderC2()}

/* ================= C3 ================= */
function logErfc(z){const t=1/(1+.5*z);return Math.log(t)+(-z*z-1.26551223+t*(1.00002368+t*(.37409196+t*(.09678418+t*(-.18628806+t*(.27886807+t*(-1.13520398+t*(1.48851587+t*(-.82215223+t*.17087277)))))))))}
function logPhi(x){const z=-x/Math.SQRT2;return z>=0?Math.log(.5)+logErfc(z):Math.log1p(-.5*Math.exp(logErfc(-z)))}
const ZN=1600,ZH=.01,PHI=new Float64Array(ZN+1);for(let i=0;i<=ZN;i++){const z=-8+i*ZH;PHI[i]=Math.exp(-z*z/2)/Math.sqrt(2*Math.PI)}
function Pe(t,V){if(t>=1)return 0;const c=t/(1-t);let acc=0;for(let i=0;i<=ZN;i++){const z=-8+i*ZH;acc+=PHI[i]*Math.exp((V-1)*logPhi(z+c))*(i===0||i===ZN?.5:1)}return 1-acc*ZH}
function tauOf(pe){const p0=pe[0],tau=new Float64Array(pe.length);for(let i=0;i<pe.length;i++)tau[i]=Math.min(1,Math.max(0,(p0-pe[i])/p0));return tau}
function tOfTau(tau,y){if(y<=0)return 0;if(y>=1)return 1;const n=tau.length-1;for(let i=1;i<=n;i++)if(tau[i]>=y){const a=tau[i-1],b=tau[i];return(i-1)/n+(1/n)*(b>a?(y-a)/(b-a):0)}return 1}
const C3V=[2,10,100,1000,30522,50257],C3L={30522:'30,522（LM1B）',50257:'50,257（OWT）'};
const c3={V:30522,N:16,cache:new Map(),job:0,started:false};  // cache: V -> {lut:Float64Array(1001)|null, bg:Float64Array(201)|null, part:{arr,k}}
function c3entry(V){if(!c3.cache.has(V))c3.cache.set(V,{lut:null,bg:null});return c3.cache.get(V)}
function lutSync(V){const e=c3entry(V);if(!e.lut){const a=new Float64Array(1001);for(let i=0;i<=1000;i++)a[i]=Pe(i/1000,V);e.lut=a}return e.lut}
const idle=window.requestIdleCallback?f=>requestIdleCallback(f,{timeout:200}):f=>setTimeout(f,0);
function c3schedule(){const id=++c3.job,jobs=[];
  if(!c3entry(c3.V).lut)jobs.push({V:c3.V,n:1001,key:'lut'});
  C3V.forEach(V=>{const e=c3entry(V);if(!e.bg&&!e.lut)jobs.push({V,n:201,key:'bg'})});
  const run=()=>{if(id!==c3.job||!jobs.length)return;const j=jobs[0],e=c3entry(j.V);if(!e.part||e.part.key!==j.key)e.part={key:j.key,arr:new Float64Array(j.n),k:0};
    const p=e.part;for(let b=0;b<50&&p.k<j.n;b++,p.k++)p.arr[p.k]=Pe(p.k/(j.n-1),j.V);
    if(p.k>=j.n){e[j.key]=p.arr;e.part=null;jobs.shift()}renderC3();idle(run)};idle(run)}
function bgOf(V){const e=c3entry(V);if(e.lut){const a=new Float64Array(201);for(let i=0;i<=200;i++)a[i]=e.lut[i*5];return a}return e.bg}
function renderC3(){const V=c3.V,N=c3.N,e=c3entry(V),boxM=$('#c3-main'),W=widthOf(boxM,280,560);
  const F=frame($('svg',boxM),{W,H:330,m:{l:40,r:14,t:28,b:96},x:[0,1],y:[0,1],xt:[0,.2,.4,.6,.8,1],yt:[0,.25,.5,.75,1],xf:v=>String(v),yf:v=>String(v)});
  const lut=e.lut,tau=lut?tauOf(lut):null,win=tau?[tOfTau(tau,.05),tOfTau(tau,.95)]:null;
  if(win){el('rect',{x:F.X(win[0]),y:F.Y(1),width:F.X(win[1])-F.X(win[0]),height:F.Y(0)-F.Y(1),class:'band'},F.g);tx(F.g,(F.X(win[0])+F.X(win[1]))/2,F.Y(1)-8,'90% 解碼進度',{class:'al','text-anchor':'middle'})}
  const levels=[.25,.45,.6,.72,.84,.95];
  C3V.forEach((v,j)=>{const a=bgOf(v);if(!a||v===V)return;el('path',{d:lp(Array.from(a,(p,i)=>[F.X(i/200),F.Y(p)])),class:'ref'},F.g);
    let i=a.findIndex(p=>p<levels[j]);if(i>0)tx(F.g,F.X(i/200)-4,F.Y(levels[j])+4,(j?'':'|V| = ')+v.toLocaleString('en-US'),{class:'tk halo','text-anchor':'end'})});
  const cur=lut||(e.part&&e.part.key==='lut'?e.part.arr.subarray(0,e.part.k):null);
  if(cur&&cur.length>1)el('path',{d:lp(Array.from(cur,(p,i)=>[F.X(i/1000),F.Y(p)])),class:'tot'},F.g);
  if(!lut)tx(F.g,F.X(.5),F.Y(.5),'計算中…',{class:'lab','text-anchor':'middle'});
  else{const i=lut.findIndex(p=>p<levels[C3V.indexOf(V)]);if(i>0)tx(F.g,F.X(i/1000)-6,F.Y(levels[C3V.indexOf(V)])+4,'|V| = '+V.toLocaleString('en-US'),{class:'labm halo','text-anchor':'end'})}
  // two tick rows
  const yA=F.y0+36,yB=F.y0+70;tx(F.g,F.x0,yA+18,'等距 t 格點 n/N（短線）',{class:'tk'});
  el('line',{x1:F.x0,x2:F.x1,y1:yA,y2:yA,class:'faint'},F.g);el('line',{x1:F.x0,x2:F.x1,y1:yB,y2:yB,class:'faint'},F.g);
  for(let n=0;n<=N;n++)el('line',{x1:F.X(n/N),x2:F.X(n/N),y1:yA-4,y2:yA+4,stroke:'var(--ink-3)','stroke-width':1.5},F.g);
  let kU=0,kT=0;for(let n=0;n<=N;n++){if(win&&n/N>=win[0]&&n/N<=win[1])kU++}
  if(tau){for(let n=0;n<=N;n++){const tn=tOfTau(tau,n/N);if(tn>=win[0]&&tn<=win[1])kT++;el('line',{x1:F.X(tn),x2:F.X(tn),y1:yB-7,y2:yB+7,stroke:'var(--accent)','stroke-width':2},F.g)}}
  tx(F.g,F.x0,yB+22,'τ 格點 t(n/N)（長線）',{class:'tk'});
  $('svg',boxM).setAttribute('aria-label',lut?`|V| = ${V}：90% 解碼進度在 t = ${win[0].toFixed(3)} 到 ${win[1].toFixed(3)}；N = ${N} 時窗內等距格點 ${kU} 個、τ 格點 ${kT} 個`:'計算中');
  // sub plot
  const boxS=$('#c3-sub'),W2=widthOf(boxS,240,360),G=frame($('svg',boxS),{W:W2,H:206,m:{l:36,r:10,t:10,b:38},x:[0,1],y:[0,1],xt:[0,.5,1],yt:[0,.5,1],xf:String,yf:String,xl:'t'});
  el('line',{x1:G.X(0),y1:G.Y(0),x2:G.X(1),y2:G.Y(1),class:'ref'},G.g);tx(G.g,G.X(.62),G.Y(.62)-6,'τ = t',{class:'tk','text-anchor':'end'});
  if(tau)el('path',{d:lp(Array.from(tau,(p,i)=>[G.X(i/1000),G.Y(p)])),class:'acc',style:'stroke-width:2.5'},G.g);
  const f4=v=>v.toFixed(4),f3=v=>v.toFixed(3);
  $('#c3-ro').innerHTML=`|V| = ${V.toLocaleString('en-US')}，N = ${N}<br>Pₑ(0) = <b>${f4(Pe(0,V))}</b>　Pₑ(0.5) = <b>${f4(Pe(.5,V))}</b><br>Pₑ(0.8) = <b>${f4(Pe(.8,V))}</b><br>`+(tau?`t(0.05) = <b>${f3(win[0])}</b>　t(0.5) = <b>${f3(tOfTau(tau,.5))}</b>　t(0.95) = <b>${f3(win[1])}</b><br>落在窗內的格點（共 ${N+1} 個）：等距 <b>${kU}</b> 個；τ 格點 <b class="acc">${kT}</b> 個`:'t(τ) 計算中…')}
function initC3(){segInit($('#c3-v'),C3V,c3.V,v=>'|V| = '+(C3L[v]||v.toLocaleString('en-US')),v=>{c3.V=v;if(c3.started)c3schedule();renderC3()});
  segInit($('#c3-n'),[4,8,16,32,64],c3.N,v=>'N = '+v,v=>{c3.N=v;renderC3()});renderC3();
  const start=()=>{if(c3.started)return;c3.started=true;c3schedule()};
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(x=>x.isIntersecting)){io.disconnect();start()}});io.observe($('#c3'))}else start()}

/* ================= C4 ================= */
const C4MU={A:[.4,.3,.2,.1],B:[.25,.25,.25,.25],C:[1,0,0,0]};
function pushforwards(mu){const vecs=new Set(),first=new Set();for(let c=0;c<256;c++){const f=[c&3,c>>2&3,c>>4&3,c>>6&3],nu=[0,0,0,0];for(let x=0;x<4;x++)nu[f[x]]+=mu[x];
  const r=nu.map(v=>Math.round(v*1e10)/1e10);vecs.add(r.join(','));first.add(r[0])}return{nu1Set:[...first].sort((a,b)=>a-b),nVectors:vecs.size}}
const c4={mu:'A',target:.05};
function renderC4(){const mu=C4MU[c4.mu],P=pushforwards(mu),mumin=Math.min(...mu.filter(v=>v>0)),T=c4.target;
  $('#c4-to').textContent=T.toFixed(3);$('#c4-t').setAttribute('aria-valuetext','ν(y₀) = '+T.toFixed(3));
  const bb=$('#c4-mubar'),Wb=widthOf(bb,220,360),F=frame($('svg',bb),{W:Wb,H:170,m:{l:36,r:8,t:12,b:30},x:[0,4],y:[0,1],yt:[0,.5,1],yf:String});
  mu.forEach((v,i)=>{const x0=F.X(i+.18),x1=F.X(i+.82);el('rect',{x:x0,y:F.Y(v),width:x1-x0,height:F.Y(0)-F.Y(v),class:'bar-v'},F.g);tx(F.g,(x0+x1)/2,F.Y(0)+16,'y'+'₀₁₂₃'[i],{class:'tk','text-anchor':'middle'});tx(F.g,(x0+x1)/2,F.Y(v)-4,String(v),{class:'tk','text-anchor':'middle'})});
  const lb=$('#c4-line'),W=widthOf(lb,260,520),svg=$('svg',lb),H=90;sizeSvg(svg,W,H);const X=lin(0,1,16,W-16),y=48;
  el('line',{x1:X(0),x2:X(1),y1:y,y2:y,class:'axl',style:'stroke-width:2;stroke:var(--ink)'},svg);
  [0,.25,.5,.75,1].forEach(v=>{el('line',{x1:X(v),x2:X(v),y1:y,y2:y+6,class:'axl'},svg);tx(svg,X(v),y+22,String(v),{class:'tk','text-anchor':'middle'})});
  P.nu1Set.forEach(v=>el('circle',{cx:X(v),cy:y,r:5,class:'endp'},svg));
  marker(svg,'diamond',X(mumin/2),y,5,'hollow');
  el('path',{d:`M${X(T)} ${y-9}L${X(T)-7} ${y-23}L${X(T)+7} ${y-23}Z`,class:'acc-f'},svg);
  let best=P.nu1Set[0];P.nu1Set.forEach(v=>{if(Math.abs(v-T)<Math.abs(best-T))best=v});const ok=Math.abs(best-T)<1e-9;
  svg.setAttribute('aria-label',`可達的 ν(y₀)：${P.nu1Set.join('、')}；目標 ${T.toFixed(3)} ${ok?'可達':'不可達'}`);
  $('#c4-ro').innerHTML=`目標 ν(y₀) = ${T.toFixed(3)}：<b class="${ok?'':'acc'}">${ok?'可達':'不可達'}</b><br>最近可達值 <b>${best}</b>　差距 <b>${Math.abs(best-T).toFixed(3)}</b><br>μ_min / 2 = <b>${mumin/2}</b><br>256 個映射推出 <b>${P.nVectors}</b> 種不同的 ν；可達的 ν(y₀) 有 <b>${P.nu1Set.length}</b> 個`}
function initC4(){const sb=$('#c4-mu');$$('button',sb).forEach(b=>b.addEventListener('click',()=>{c4.mu=b.dataset.v;segSet(sb,c4.mu);renderC4()}));segSet(sb,c4.mu);
  $('#c4-t').addEventListener('input',e=>{c4.target=Math.round(+e.target.value*1000)/1000;renderC4()});renderC4()}

/* ================= timeline ================= */
const TL_GROUP={'1711.02281':['par','fmlm-parallel-decoding-motivation'],'2410.21035':['par','fmlm-parallel-decoding-motivation'],
 '2107.03006':['disc','fmlm-discrete-diffusion-factorization'],'2310.16834':['disc','fmlm-discrete-diffusion-factorization'],'2406.07524':['disc','fmlm-discrete-diffusion-factorization'],'2506.10892':['disc','fmlm-discrete-diffusion-factorization'],
 '2205.14217':['cont','fmlm-continuous-text-diffusion'],'2211.15089':['cont','fmlm-continuous-text-diffusion'],
 '2210.02747':['fm','fmlm-flow-matching-interpolant'],
 '2303.01469':['map','fmlm-flow-map-distillation-lineage'],'2406.07507':['map','fmlm-flow-map-distillation-lineage'],'2505.18825':['map','fmlm-flow-map-distillation-lineage'],'2505.13447':['map','fmlm-flow-map-distillation-lineage'],
 '2602.16813':['me','fmlm-onehot-denoiser-two-time']};
const TL_NAME={par:'平行解碼',disc:'離散擴散',cont:'連續文字擴散',fm:'flow matching',map:'flow map',me:'本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.map((e,i)=>({e,i})).sort((a,b)=>(+a.e.year||0)-(+b.e.year||0)||a.i-b.i).map(o=>o.e);
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];used.add(g[0]);const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${g[1]?`<a href="${g[1]}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
  const on=new Set(used);
  Object.keys(TL_NAME).filter(k=>used.has(k)).forEach(k=>{const b=document.createElement('button');b.type='button';b.className='chip';b.setAttribute('aria-pressed','true');b.textContent=TL_NAME[k];b.dataset.g=k;
    b.addEventListener('click',()=>{if(on.has(k)&&on.size===used.size){on.clear();on.add(k)}else if(on.has(k)){on.delete(k);if(!on.size)used.forEach(x=>on.add(x))}else on.add(k);
      $$('.chip',chips).forEach(c=>c.setAttribute('aria-pressed',String(on.has(c.dataset.g))));$$('.ev',track).forEach(ev=>ev.classList.toggle('dim',!on.has(ev.dataset.g)))});chips.appendChild(b)});
  const rail=$('#rail'),by=dir=>rail.scrollBy({left:dir*Math.min(rail.clientWidth*.8,560),behavior:RM?'auto':'smooth'});
  $('#tl-prev').addEventListener('click',()=>by(-1));$('#tl-next').addEventListener('click',()=>by(1));
  rail.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){by(1);e.preventDefault()}if(e.key==='ArrowLeft'){by(-1);e.preventDefault()}})}

/* ================= toc, theme, appbar ================= */
function initToc(){const secs=$$('[data-toc]'),ol=$('#toc'),bar=$('#tocbar');
  secs.forEach(s=>{ol.insertAdjacentHTML('beforeend',`<li><a href="#${s.id}"><span class="n">${s.dataset.toc}</span><span>${s.dataset.tt}</span></a></li>`);bar.insertAdjacentHTML('beforeend',`<a href="#${s.id}">${s.dataset.toc} ${s.dataset.tt}</a>`)});
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;$$('.toc a,.tocbar a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+id));const ab=$(`.tocbar a[href="#${id}"]`);if(ab&&bar.offsetParent)bar.scrollTo({left:ab.offsetLeft-16,behavior:RM?'auto':'smooth'})}})},{rootMargin:'-40% 0px -55% 0px'});
  secs.forEach(s=>io.observe(s));
  addEventListener('scroll',()=>$('#appbar').classList.toggle('raised',scrollY>8),{passive:true})}
function initTheme(){let saved=null;try{saved=localStorage.getItem('pm-theme')}catch(e){}if(saved)document.documentElement.dataset.theme=saved;
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){}})}

/* ================= self test (design 5.1–5.5, 6.3) ================= */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t)=>Math.abs(a-b)<=t+1e-12;
  // 5.1 C1a
  const t51={1:[.5,.25],2:[.25,.375],4:[.125,.438],8:[.0625,.469],16:[.0313,.484],32:[.0156,.492],64:[.0078,.496]};
  for(const N in t51){const c=discreteClosed(+N);ok(near(c.invalid,1/(2*N),1e-9)&&near(c.invalid,t51[N][0],5e-5)&&near(c.NY,t51[N][1],5e-4)&&near(c.SD,c.NY,1e-12),'C1a N='+N)}
  // 5.2 C1b
  const xs=samples(7,200);ok(near(xs[0][0],.1420,1e-4)&&near(xs[0][1],-.8643,1e-4)&&near(xs[0][2],-1.0064,1e-4)&&near(xs[0][3],.0709,1e-4),'PRNG seed 7 first sample');
  const cnt=(ends)=>{const c={NY:0,SD:0,ND:0,SY:0,tie:0};ends.forEach(e=>c[classify(e)]++);return c},md=ends=>ends.reduce((a,e)=>a+distOnehot(e),0)/ends.length;
  const ref=xs.map(x=>flowmap(x,1));
  [[1,0,0,200,1.000],[2,98,102,0,.441],[4,98,102,0,.032],[8,98,102,0,0],[16,98,102,0,0],[32,98,102,0,0]].forEach(([N,a,b,tie,d])=>{const E=xs.map(x=>euler(x,N)),c=cnt(E);ok(c.NY===a&&c.SD===b&&c.ND+c.SY===0&&c.tie===tie&&near(md(E),d,.001),`C1b Euler N=${N}: ${JSON.stringify(c)} d=${md(E).toFixed(4)}`)});
  [1,2,4,8,16,32].forEach(N=>{const E=N===1?ref:xs.map(x=>flowmap(x,N)),c=cnt(E),dev=Math.max(...E.map((e,i)=>Math.max(...e.map((v,j)=>Math.abs(v-ref[i][j])))));ok(c.NY===98&&c.SD===102&&c.ND+c.SY+c.tie===0&&near(md(E),0,.001)&&(N===1||dev<1e-9),`C1b flowmap N=${N} dev=${dev.toExponential(1)}`)});
  const hi=highlightIndex(xs),hx=xs[hi],vn=(a,b,t)=>a.every((v,i)=>near(v,b[i],t));
  ok(hi===2&&vn(hx,[-.8134,.8304,.5425,.4031],1e-4)&&near(toC(hx)[0],-1.5044,1e-4),'C1b highlight index 2');
  ok(vn(Xm(hx,0,.5),[-.2586,.7671,.4194,.5535],1e-4)&&vn(Xm(hx,0,.9),[-.0498,.9515,.0858,.9087],1e-4)&&vn(Xm(hx,0,1),[0,1,0,1],1e-4)&&vn(euler(hx,2),[.1818,.8182,.1818,.8182],1e-4),'C1b highlight trajectory');
  ok(flowU(.4,0,.5).failed===false&&flowU(.4,0,.5,1).failed===true,'flowU failed flag');
  // 5.3 C2
  const x0=[.9,-.2,-.4,.3],rows=[[.1,.5,.9,.1,.5711,.9795,.9386,[.85,.45],[.97,.89]],[.3,.6,.95,.0577,.6661,.9978,.9787,[.88,.56],[.985,.945]],[0,.25,.5,.3333,.5171,.5872,.5638,[.775,.175],[.85,.45]],[0,.5,1,0,.5638,1,1,[.85,.45],[1,1]]];
  rows.forEach(([s,u,t,g,a,b,c,su,st])=>{const R=c2compute(s,u,t,x0,true),sum=(v,i)=>v[i]+v[i+1];
    ok(near(R.g,g,1e-4)&&near(R.dsu[0],a,1e-4)&&near(R.dut[0],b,1e-4)&&near(R.dst[0],c,1e-4)&&near(sum(R.Xsu,0),su[0],1e-4)&&near(sum(R.Xsu,2),su[1],1e-4)&&near(sum(R.Xst,0),st[0],1e-4)&&near(sum(R.Xst,2),st[1],1e-4)&&R.tdiff<1e-8&&near(sum(R.dst,0),1,1e-9)&&near(sum(R.dst,2),1,1e-9)&&(t>=1||Math.abs(R.e82.q-R.dst[0])<1e-6),
      `C2 (${s},${u},${t}) g=${R.g.toFixed(4)} dst=${R.dst[0].toFixed(4)} td=${R.tdiff.toExponential(1)}${R.e82?' q82diff='+Math.abs(R.e82.q-R.dst[0]).toExponential(1):''}`)});
  const R0=c2compute(.1,.5,.9,x0,false);ok(vn(R0.x,[.8606,-.1306,-.3094,.3194],1e-4)&&vn(R0.Xsu,[.7319,.1181,.0819,.3681],1e-4)&&vn(R0.Xst,[.93,.04,.8,.09],1e-4)&&near(R0.Ds[0],.5112,1e-4)&&near(R0.dst[2],.9386,1e-4)&&near(R0.Xsu[3],.3681,1e-4),'C2 default vectors');
  // 5.4 C3
  const t54={2:[.5,.2398,.0023,[.081,.311,.488,.619,.735],10],10:[.9,.6591,.0168,[.203,.487,.618,.702,.777],9],100:[.99,.9175,.0854,[.45,.642,.715,.764,.812],5],1000:[.999,.9834,.236,[.603,.716,.764,.798,.833],4],30522:[.99997,.9987,.5439,[.707,.773,.804,.828,.854],2],50257:[.99998,.9991,.5873,[.717,.779,.809,.831,.856],2]};
  for(const V in t54){const [p0,p5,p8,tq,k]=t54[V],lut=lutSync(+V),tau=tauOf(lut),q=[.05,.25,.5,.75,.95].map(y=>tOfTau(tau,y));let kU=0,kT=0;for(let n=0;n<=16;n++){if(n/16>=q[0]&&n/16<=q[4])kU++;const tn=tOfTau(tau,n/16);if(tn>=q[0]&&tn<=q[4])kT++}
    ok(near(Pe(0,+V),p0,1e-4)&&near(Pe(0,+V),1-1/V,1e-6)&&near(Pe(.5,+V),p5,1e-4)&&near(Pe(.8,+V),p8,1e-4)&&q.every((v,i)=>near(v,tq[i],.002))&&kU===k&&kT===15,`C3 |V|=${V}: t=${q.map(v=>v.toFixed(3)).join(',')} kU=${kU} kT=${kT}`)}
  {const tau=tauOf(lutSync(30522)),g8=[0,.745,.773,.791,.804,.816,.828,.842,1];ok(g8.every((v,n)=>near(tOfTau(tau,n/8),v,.002)),'C3 τ grid N=8')}
  // 5.5 C4
  [['A',11,202,.05],['B',5,35,.125],['C',2,4,.5]].forEach(([k,n1,nv,h])=>{const P=pushforwards(C4MU[k]);ok(P.nu1Set.length===n1&&P.nVectors===nv&&Math.min(...C4MU[k].filter(v=>v>0))/2===h,`C4 ${k}: ${P.nu1Set.length} values, ${P.nVectors} vectors`)});
  ok(pushforwards(C4MU.A).nu1Set.every((v,i)=>near(v,i/10,1e-9))&&!pushforwards(C4MU.A).nu1Set.some(v=>near(v,.05,1e-9)),'C4 A reachable set');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  ok(!!document.querySelector('.katex'),'KaTeX rendered');
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ================= boot ================= */
initTheme();initToc();initC1a();initC1b();initC2();initC3();initC4();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderC1a();renderC1b();renderC2();renderC3();renderC4()})});
let mathDone=false;
function renderMath(){if(mathDone||typeof renderMathInElement!=='function')return;mathDone=true;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],ignoredClasses:['readout','samples','mk-legend'],throwOnError:false});
  $$('.katex-display').forEach(d=>{const n=d.closest('.eq');const lab=n&&$('.eq-n',n)?$('.eq-n',n).textContent:'式子';d.setAttribute('tabindex','0');d.setAttribute('aria-label',lab+'，可水平捲動')})}
const test=/[?&]test\b/.test(location.search);
function afterMath(){renderMath();if(test)requestAnimationFrame(()=>setTimeout(selfTest,0))}
if(document.readyState==='complete')afterMath();else addEventListener('load',afterMath);
})();
