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
function svgOf(box,W,H){const svg=$('svg',box);svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);return svg}
const f4=v=>(Math.abs(v)<5e-5?0:v).toFixed(4),f3=v=>(Math.abs(v)<5e-4?0:v).toFixed(3),f2=v=>(Math.abs(v)<.005?0:v).toFixed(2);
const g6=v=>{if(v===0)return'0';const s=Number(v.toPrecision(6));return Math.abs(s)<1e-3||Math.abs(s)>=1e5?s.toExponential(3).replace('e','×10^'):String(s)};
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function frameSched(fn){let q=0;return(...a)=>{if(q)return;q=requestAnimationFrame(()=>{q=0;fn(...a)})}}
function segInit(box,opts,cur,label,onPick){box.textContent='';opts.forEach(o=>{const b=document.createElement('button');b.type='button';b.setAttribute('role','radio');
    b.textContent=label?label(o):String(o);b.dataset.v=String(o);box.appendChild(b)});
  const sync=v=>$$('button',box).forEach(b=>{const on=b.dataset.v===String(v);b.setAttribute('aria-checked',String(on));b.setAttribute('aria-pressed',String(on));b.tabIndex=on?0:-1});sync(cur);
  box.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;const v=opts.find(o=>String(o)===b.dataset.v);sync(v);onPick(v)});
  box.addEventListener('keydown',e=>{const bs=$$('button',box),i=bs.findIndex(b=>b.tabIndex===0);let j=i;if(e.key==='ArrowRight'||e.key==='ArrowDown')j=(i+1)%bs.length;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')j=(i-1+bs.length)%bs.length;else return;e.preventDefault();bs[j].focus();bs[j].click()});
  return sync}
function rangeInit(id,fmt,onInput,onChange){const s=$('#'+id),o=$('#'+id+'o');const upd=()=>{const t=fmt(+s.value);o.textContent=t[0];s.setAttribute('aria-valuetext',t[1]||t[0])};upd();
  s.addEventListener('input',()=>{upd();onInput(+s.value)});s.addEventListener('change',()=>{upd();onChange(+s.value)});return{s,upd}}

/* ================= shared math (mirrors verify/core.py) ================= */
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}  // core.mulberry32
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}                                                     // core.gauss
const TOY={w:[.4,.6],mu:[-1.2,1.2],s:[.3,.3]},LW=TOY.w.map(Math.log);
function comp(t){return TOY.mu.map((m,k)=>[(1-t)*m,(1-t)**2*TOY.s[k]*TOY.s[k]+t*t])}                                                                 // core.comp
function density(x,t){const c=comp(t);let s=0;for(let k=0;k<2;k++){const [m,v]=c[k];s+=TOY.w[k]*Math.exp(-((x-m)**2)/(2*v))/Math.sqrt(2*Math.PI*v)}return s}  // core.density
function velC(x,t,c,lv){                                                                                                                               // core.velocity (c, log v precomputed)
  const l0=LW[0]-(x-c[0][0])**2/(2*c[0][1])-.5*lv[0],l1=LW[1]-(x-c[1][0])**2/(2*c[1][1])-.5*lv[1],mx=Math.max(l0,l1),p0=Math.exp(l0-mx),p1=Math.exp(l1-mx),z=p0+p1;
  let e0=0;e0+=p0/z*(TOY.mu[0]+(1-t)*TOY.s[0]*TOY.s[0]/c[0][1]*(x-c[0][0]));e0+=p1/z*(TOY.mu[1]+(1-t)*TOY.s[1]*TOY.s[1]/c[1][1]*(x-c[1][0]));return(x-e0)/t}
function velocity(x,t){const c=comp(t);return velC(x,t,c,[Math.log(c[0][1]),Math.log(c[1][1])])}
function sigmaT(t,a,T){const tt=t<1-1e-12?t:1-1/T;return a*Math.sqrt(tt/(1-tt))}                                                                          // core.sigma
// core.simulate: snapshots[j] at t_j = 1 - j/T. RNG: M initial draws, then per step, per particle (SDE only).
function simulate(a,T,seed,sde,M=4000){const r=mulberry32(seed),x=new Float64Array(M);for(let m=0;m<M;m++)x[m]=gauss(r);
  const snaps=[Float64Array.from(x)],dt=-1/T,sq=Math.sqrt(-dt);let divergedAt=null;
  for(let j=0;j<T;j++){const t=1-j/T,sg=sde?sigmaT(t,a,T):0,k2=sg*sg/(2*t),c=comp(t),lv=[Math.log(c[0][1]),Math.log(c[1][1])];let bad=false;
    for(let m=0;m<M;m++){const xm=x[m],v=velC(xm,t,c,lv),drift=v+k2*(xm+(1-t)*v),nz=sde?gauss(r):0;const nx=xm+drift*dt+sg*sq*nz;x[m]=nx;if(!(Math.abs(nx)<=1e6))bad=true}
    snaps.push(Float64Array.from(x));if(bad){divergedAt=j+1;break}}
  return{snaps,divergedAt}}
function erf(x){const s=x>=0?1:-1;x=Math.abs(x);const t=1/(1+.3275911*x);                                                                             // core.erf (A&S 7.1.26)
  const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-x*x);return s*y}
function cdf(x,t){const c=comp(t);let s=0;for(let k=0;k<2;k++){const [m,v]=c[k];s+=TOY.w[k]*.5*(1+erf((x-m)/Math.sqrt(2*v)))}return s}                    // core.cdf
function w1(xs,t,lo=-4,hi=4,n=801){const s=Float64Array.from(xs).sort(),M=s.length,h=(hi-lo)/(n-1);let k=0,acc=0,prev=null;                         // core.w1
  for(let i=0;i<n;i++){const g=lo+i*h;while(k<M&&s[k]<=g)k++;const d=Math.abs(k/M-cdf(g,t));if(prev!==null)acc+=.5*(d+prev)*h;prev=d}return acc}
function stats(xs){const M=xs.length;let m=0;for(const x of xs)m+=x;m/=M;let v=0,p=0;for(const x of xs){v+=(x-m)**2;if(x>0)p++}return[m,Math.sqrt(v/M),p/M]}  // core.stats
const EXACT0=[.24,1.2134,.6];
// GRPO advantage (core.advantages): population std; std < 1e-12 -> all zero (推定)
function advantages(R){const G=R.length;let m=0;for(const x of R)m+=x;m/=G;let v=0;for(const x of R)v+=(x-m)**2;const s=Math.sqrt(v/G);
  return{A:s<1e-12?R.map(()=>0):R.map(x=>(x-m)/s),m,s}}
const drawReward=(r,kind,p)=>kind==='binary'?(r()<p?1:0):.5+.15*gauss(r);                                                                            // core.draw_reward
const trueZ=(R,kind,p)=>kind==='binary'?(R-p)/Math.sqrt(p*(1-p)):(R-.5)/.15;                                                                       // core.true_z
function advMC(G,kind,p,seed,K=2000){const r=mulberry32(seed);let se=0,deg=0;                                                                       // core.adv_mc
  for(let q=0;q<K;q++){const R=[];for(let i=0;i<G;i++)R.push(drawReward(r,kind,p));const {A,s}=advantages(R);if(s<1e-12)deg++;let e=0;for(let i=0;i<G;i++)e+=(A[i]-trueZ(R[i],kind,p))**2;se+=e}
  return[Math.sqrt(se/(K*G)),deg/K]}
const pDegenerate=(G,kind,p)=>kind==='binary'?p**G+(1-p)**G:0;
function clipObj(rho,A,eps){const c=Math.min(Math.max(rho,1-eps),1+eps),u=rho*A,k=c*A;return[Math.min(u,k),u<=k+1e-9?A:0]}                              // core.clip_obj
function klCoeff(j,T,a){const t=1-j/T,sg=sigmaT(t,a,T);return(sg*(1-t)/(2*t)+1/sg)**2/(2*T)}                                                        // core.kl_coeff
function stepPolicy(j,T,a,dv,z){const t=1-j/T,adt=1/T,sg=sigmaT(t,a,T),sd=sg*Math.sqrt(adt),shift=(1+sg*sg*(1-t)/(2*t))*dv*(-adt);                   // core.step_policy
  const kl1=shift**2/(2*sg*sg*adt),kl2=adt/2*(sg*(1-t)/(2*t)+1/sg)**2*dv*dv,logr=((sd*z)**2-(sd*z-shift)**2)/(2*sd*sd);return{t,sg,sd,shift,kl1,kl2,logr}}
const shared={ei:20};

/* ================= time strips (Hero, S8) ================= */
function strip(svg,x0,y0,w,h,T,label,acc){const cw=w/T;for(let j=0;j<T;j++)el('rect',{x:x0+j*cw,y:y0,width:Math.max(cw-(cw>6?2:1),1),height:h,class:acc?'cell-acc':'cell',rx:cw>6?3:1},svg);if(label)tx(svg,x0,y0-6,label,{class:'al'})}
function renderHero(){const box=$('#hero-strip'),W=widthOf(box,260,560),svg=svgOf(box,W,64);strip(svg,0,8,W,24,10,null,true);
  tx(svg,0,52,'t = 1 噪聲',{class:'tk'});tx(svg,W,52,'t = 0 資料',{class:'tk','text-anchor':'end'});tx(svg,W/2,52,'10 步',{class:'tk','text-anchor':'middle'})}
function renderCost(){const box=$('#s8-cost'),W=widthOf(box,280,760),narrow=W<520,svg=svgOf(box,W,narrow?150:126);
  const lw=narrow?0:150,x0=lw,w=W-lw,r1=narrow?24:20,r2=narrow?86:72;
  tx(svg,0,narrow?r1-6:r1+16,'訓練取樣：T = 10（SDE）',{class:'al'});strip(svg,x0,narrow?r1:r1,w,22,10,null,true);
  tx(svg,0,narrow?r2-6:r2+16,'推論：T = 40',{class:'al'});strip(svg,x0,r2,w,22,40,null,false);
  tx(svg,x0,narrow?r2+40:r2+44,'每個提示詞 24 × 10 = 240 次 vs 24 × 40 = 960 次模型前向',{class:'tk'})}

/* ================= C1 · group-relative advantage ================= */
const GS=[2,3,4,6,8,12,16,24,32,48];
const c1={kind:'continuous',gi:7,pp:30,seed:1,token:0,curve:null,curveKey:'',stale:false};
const c1p=()=>c1.pp/100;
function c1curve(){const key=`${c1.kind}|${c1.pp}`;if(c1.curveKey===key&&c1.curve&&c1.curve.length===GS.length)return;const tok=++c1.token,kind=c1.kind,p=c1p(),out=[];c1.stale=true;c1.curveKey=key;c1.curve=out;
  let i=0;const step=()=>{if(tok!==c1.token)return;const [rm,dg]=advMC(GS[i],kind,p,7);out.push({G:GS[i],rm,dg});i++;if(i<GS.length)setTimeout(step,0);else{c1.stale=false;c1draw();announce('c1',c1sum())}c1draw()};setTimeout(step,0)}
function c1group(){const r=mulberry32(c1.seed),G=GS[c1.gi],R=[];for(let i=0;i<G;i++)R.push(drawReward(r,c1.kind,c1p()));return Object.assign({R},advantages(R))}
function c1sum(){const G=GS[c1.gi],pt=c1.curve&&c1.curve.find(d=>d.G===G);return `G = ${G}，RMSE ${pt?f3(pt.rm):'計算中'}，全相同機率 ${f3(pDegenerate(G,c1.kind,c1p()))}`}
function c1draw(){const G=GS[c1.gi],g=c1group(),boxA=$('#c1-a'),boxB=$('#c1-b'),avail=widthOf(boxA,240,520),l=36,rp=48,W=Math.max(avail,l+rp+5*G),bw=(W-l-rp)/G;
  {const H=104,svg=svgOf(boxA,W,H),y=lin(0,1,H-8,10);[0,.5,1].forEach(v=>{el('line',{x1:l,x2:W-rp,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,l-6,y(v)+4,String(v),{class:'tk','text-anchor':'end'})});
    g.R.forEach((v,i)=>{const vv=Math.max(0,Math.min(1,v)),x=l+i*bw+1;el('rect',{x,y:y(vv),width:Math.max(bw-2,1),height:y(0)-y(vv),class:'bar'},svg);
      if(v>1||v<0){const yy=v>1?y(1)-2:y(0)+2,d=v>1?-6:6;el('path',{d:`M${x+bw/2-4} ${yy}L${x+bw/2+4} ${yy}L${x+bw/2} ${yy+d}Z`,class:'acc-f'},svg)}})}
  {const H=150,svg=svgOf(boxB,W,H),y=lin(-4,4,H-10,10),defs=el('defs',null,svg),pat=el('pattern',{id:'hatch',width:5,height:5,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},defs);
    const gg=el('g',{style:'color:var(--c-ref)'},svg);el('rect',{width:5,height:5,fill:'none'},pat);el('line',{x1:0,y1:0,x2:0,y2:5,stroke:'currentColor','stroke-width':2},pat);
    [-4,-2,0,2,4].forEach(v=>{el('line',{x1:l,x2:W-rp,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,l-6,y(v)+4,v<0?'−'+(-v):String(v),{class:'tk','text-anchor':'end'})});
    el('line',{x1:l,x2:W-rp,y1:y(0),y2:y(0),class:'axl'},svg);tx(svg,W-rp+6,y(0)+4,'組平均',{class:'tk'});
    g.A.forEach((a,i)=>{const av=Math.max(-4,Math.min(4,a)),x=l+i*bw+1,w=Math.max(bw-2,1);
      if(av>=0)el('rect',{x,y:y(av),width:w,height:y(0)-y(av),class:'bar'},svg);else el('rect',{x,y:y(0),width:w,height:y(av)-y(0),fill:'url(#hatch)',stroke:'currentColor','stroke-width':1},gg);
      if(Math.abs(a)>4){const yy=a>0?y(4):y(-4),d=a>0?-6:6;el('path',{d:`M${x+bw/2-4} ${yy}L${x+bw/2+4} ${yy}L${x+bw/2} ${yy+d}Z`,class:'ink-f'},svg)}});
    if(g.s<1e-12){tx(svg,l+8,24,'整組獎勵相同：std = 0，所有優勢為 0；',{class:'warn'});tx(svg,l+8,42,'替代目標項沒有梯度，KL 項仍有。',{class:'warn'})}
  }
  {const box=$('#c1-c'),W2=widthOf(box,240,520),H=160,r2=c1.kind==='binary'?34:10,svg=svgOf(box,W2,H),x=v=>lin(Math.log(2),Math.log(48),l,W2-r2)(Math.log(v)),y=lin(0,1,H-28,10);
    [0,.5,1].forEach(v=>{el('line',{x1:l,x2:W2-r2,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,l-6,y(v)+4,String(v),{class:'tk','text-anchor':'end'});if(c1.kind==='binary')tx(svg,W2-r2+6,y(v)+4,String(v),{class:'tk'})});
    [2,6,12,24,48].forEach(v=>tx(svg,x(v),H-12,String(v),{class:'tk','text-anchor':'middle'}));
    const d=c1.curve||[],gp=el('g',c1.stale?{class:'stale'}:null,svg);
    if(d.length){el('path',{d:lp(d.map(q=>[x(q.G),y(Math.min(1,q.rm))])),class:'acc'},gp);d.forEach(q=>el('circle',{cx:x(q.G),cy:y(Math.min(1,q.rm)),r:3.5,class:'acc-f'},gp));
      if(c1.kind==='binary'){el('path',{d:lp(d.map(q=>[x(q.G),y(q.dg)])),class:'ink2-thin'},gp);d.forEach(q=>el('rect',{x:x(q.G)-3.5,y:y(q.dg)-3.5,width:7,height:7,fill:'var(--surface)',stroke:'var(--ink-2)','stroke-width':1.5},gp))}
      const cur=d.find(q=>q.G===G);if(cur)el('circle',{cx:x(G),cy:y(Math.min(1,cur.rm)),r:8,class:'hollow'},svg)}
    if(c1.stale)tx(svg,l+8,y(.5),d.length?'p 已變更或計算中，曲線更新中':'計算中',{class:'warn'})}
  const cur=c1.curve&&!c1.stale?c1.curve.find(q=>q.G===G):null;let sA=0,sA2=0;g.A.forEach(a=>{sA+=a;sA2+=a*a});
  $('#c1-read').textContent=`組平均 ${f4(g.m)} · 組標準差 ${f4(g.s)}\nΣA ${Math.abs(sA)<1e-12?'0':sA.toExponential(2)} · ΣA² ${f4(sA2)}\nRMSE（G = ${G}）${cur?f4(cur.rm):'更新中'}\n全相同機率：閉式 ${f4(pDegenerate(G,c1.kind,c1p()))} · 模擬 ${cur?f4(cur.dg):'更新中'}`}
function initC1(){const sched=frameSched(c1draw);
  segInit($('#c1-kind'),['continuous','binary'],c1.kind,v=>v==='binary'?'二元':'連續',v=>{c1.kind=v;syncP();c1curve();c1draw();announce('c1',c1sum())});
  const syncP=()=>{const on=c1.kind==='binary';$('#c1-p').disabled=!on;$('#c1-pnote').classList.toggle('hide',on)};syncP();
  rangeInit('c1-g',v=>[String(GS[v]),`G = ${GS[v]}`],v=>{c1.gi=v;sched()},v=>{c1.gi=v;c1draw();announce('c1',c1sum())});
  rangeInit('c1-p',v=>[(v/100).toFixed(2),`p = ${(v/100).toFixed(2)}`],v=>{c1.pp=v;c1.stale=true;sched()},v=>{c1.pp=v;c1curve();c1draw()});
  $('#c1-seed').addEventListener('click',()=>{c1.seed++;c1draw()});c1curve();c1draw()}

/* ================= C2 · clipped surrogate ================= */
const c2={ai:10,ri:100};
function c2draw(){const A=c2.ai/10,eps=shared.ei/100,rho=c2.ri/100,box=$('#c2-fig'),W=widthOf(box,260,560),H=236,l=36,r=14,svg=svgOf(box,W,H),x=lin(.5,1.5,l,W-r),y=lin(-3,3,H-32,10);
  el('rect',{x:x(1-eps),y:10,width:x(1+eps)-x(1-eps),height:H-42,class:'band'},svg);tx(svg,x(1),24,'裁剪範圍',{class:'tk','text-anchor':'middle'});
  [-3,-2,-1,0,1,2,3].forEach(v=>{el('line',{x1:l,x2:W-r,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,l-6,y(v)+4,v<0?'−'+(-v):String(v),{class:'tk','text-anchor':'end'})});
  const tk=[.5,1,1.5,1-eps,1+eps].sort((a,b)=>a-b);tk.forEach(v=>tx(svg,x(v),H-12,(Math.round(v*100)/100).toString(),{class:'tk','text-anchor':'middle'}));
  const xs=[];for(let i=0;i<=200;i++)xs.push(.5+i/200);xs.push(1-eps,1+eps);xs.sort((a,b)=>a-b);
  el('path',{d:lp(xs.map(v=>[x(v),y(Math.max(-3,Math.min(3,v*A)))])),class:'ink2-thin'},svg);
  el('path',{d:lp(xs.map(v=>[x(v),y(Math.max(-3,Math.min(3,clipObj(v,A,eps)[0])))])),class:'acc'},svg);
  if(A!==0){const zl=A>0?[1+eps,1.5]:[.5,1-eps],ym=y(Math.max(-3,Math.min(3,clipObj((zl[0]+zl[1])/2,A,eps)[0])));tx(svg,(x(zl[0])+x(zl[1]))/2,ym+(A>0?20:-10),'替代目標項梯度 0',{class:'tk','text-anchor':'middle'})}
  const [L,gr]=clipObj(rho,A,eps);el('circle',{cx:x(rho),cy:y(Math.max(-3,Math.min(3,L))),r:6,class:'hollow'},svg);
  const clipped=gr===0&&A!==0;$('#c2-read').textContent=`L ${f3(L)} · 斜率 ∂L/∂ρ ${f2(gr)}\n${clipped?'已裁剪：替代目標項梯度 0，KL 項不受影響':A===0?'A = 0：替代目標項為 0':'未裁剪'}`;
  return{L,gr,clipped}}
function initC2(){const sched=frameSched(c2draw),sum=()=>{const r=c2draw();announce('c2',`ρ = ${(c2.ri/100).toFixed(2)}，L = ${f3(r.L)}，${r.clipped?'已裁剪':'未裁剪'}`)};
  rangeInit('c2-a',v=>[(v/10).toFixed(1),`A = ${(v/10).toFixed(1)}`],v=>{c2.ai=v;sched()},v=>{c2.ai=v;sum()});
  rangeInit('c2-e',v=>[(v/100).toFixed(2),`ε = ${(v/100).toFixed(2)}`],v=>{shared.ei=v;sched();c4draw()},v=>{shared.ei=v;sum();c4draw()});
  rangeInit('c2-r',v=>[(v/100).toFixed(2),`ρ = ${(v/100).toFixed(2)}`],v=>{c2.ri=v;sched()},v=>{c2.ri=v;sum()});c2draw()}

/* ================= C3 · ODE vs SDE toy ================= */
const c3={mode:'sde',ak:14,T:10,j:10,showOde:true,seed:1,sim:null,stale:false,w:null};const odeCache=new Map();
function odeSim(T,seed){const k=`${T}|${seed}`;if(!odeCache.has(k))odeCache.set(k,simulate(0,T,seed,false));return odeCache.get(k)}
function c3compute(){const a=c3.ak/20;c3.sim=c3.mode==='ode'?odeSim(c3.T,c3.seed):simulate(a,c3.T,c3.seed,true)}
function c3metrics(){const j=c3.j,t=1-j/c3.T,s=c3.sim,o=odeSim(c3.T,c3.seed);const ok=s.divergedAt==null||j<s.snaps.length;
  return{t,sw:ok&&s.divergedAt==null?w1(s.snaps[j],t):null,ow:w1(o.snaps[j],t)}}
function c3draw(){const j=c3.j,T=c3.T,t=1-j/T,s=c3.sim,o=odeSim(T,c3.seed),div=s.divergedAt!=null,snap=div?null:s.snaps[j];
  {const box=$('#c3-hist'),W=widthOf(box,260,600),H=210,l=34,r=8,svg=svgOf(box,W,H),x=lin(-4,4,l,W-r),y=lin(0,1,H-24,10);
    [0,.5,1].forEach(v=>{el('line',{x1:l,x2:W-r,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,l-6,y(v)+4,String(v),{class:'tk','text-anchor':'end'})});
    [-4,-2,0,2,4].forEach(v=>tx(svg,x(v),H-8,v<0?'−'+(-v):String(v),{class:'tk','text-anchor':'middle'}));tx(svg,l+4,22,'密度',{class:'tk'});
    const hist=xs=>{const c=new Array(64).fill(0);let out=0;for(const v of xs){const b=Math.floor((v+4)/.125);if(b>=0&&b<64)c[b]++;else out++}return{d:c.map(n=>n/(xs.length*.125)),out}};
    let outN=0;if(snap){const h=hist(snap);outN=h.out;h.d.forEach((d,b)=>{const dd=Math.min(1,d);if(dd>0)el('rect',{x:x(-4+b*.125),y:y(dd),width:x(-4+.125)-x(-4),height:y(0)-y(dd),class:'bar'},svg);
      if(d>1){const cx=x(-4+(b+.5)*.125);el('path',{d:`M${cx-3} ${y(1)}L${cx+3} ${y(1)}L${cx} ${y(1)-5}Z`,class:'acc-f'},svg)}})}
    if(c3.showOde&&c3.mode==='sde'){const h=hist(o.snaps[j]),pts=[];h.d.forEach((d,b)=>{const yy=y(Math.min(1,d));pts.push([x(-4+b*.125),yy],[x(-4+(b+1)*.125),yy])});el('path',{d:lp(pts),class:'ink2-step'},svg)}
    const ep=[];for(let i=0;i<=240;i++){const xv=-4+8*i/240;ep.push([x(xv),y(Math.min(1,density(xv,t)))])}el('path',{d:lp(ep),class:'cref'},svg);
    const pk=0;tx(svg,W-r,22,'精確邊際 p_t',{class:'tk','text-anchor':'end'});
    if(div)tx(svg,l+8,y(.6),`數值發散：第 ${s.divergedAt} 步有粒子超過 10⁶`,{class:'warn'});
    else if(outN)tx(svg,l+8,y(.85),`範圍外 ${outN} 個`,{class:'tk'})}
  {const box=$('#c3-traj'),W=widthOf(box,240,420),H=210,l=30,r=8,svg=svgOf(box,W,H),x=lin(1,0,l,W-r),y=lin(-4,4,H-24,10);
    [-4,-2,0,2,4].forEach(v=>{el('line',{x1:l,x2:W-r,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,l-6,y(v)+4,v<0?'−'+(-v):String(v),{class:'tk','text-anchor':'end'})});
    tx(svg,l,H-8,'t = 1',{class:'tk'});tx(svg,W-r,H-8,'t = 0',{class:'tk','text-anchor':'end'});
    const cp='c3cp';const defs=el('defs',null,svg),c=el('clipPath',{id:cp},defs);el('rect',{x:l,y:10,width:W-l-r,height:H-34},c);const gg=el('g',{'clip-path':`url(#${cp})`},svg);
    if(c3.showOde&&c3.mode==='sde')for(let m=0;m<12;m++)el('path',{d:lp(o.snaps.map((sn,k)=>[x(1-k/T),y(Math.max(-5,Math.min(5,sn[m])))])),class:'ink2-thin'},gg);
    for(let m=0;m<12;m++)el('path',{d:lp(s.snaps.map((sn,k)=>[x(1-k/T),y(Math.max(-5,Math.min(5,sn[m])))])),class:c3.mode==='sde'?'acc-thin':'ink2-step'},gg);
    el('line',{x1:x(t),x2:x(t),y1:10,y2:H-24,class:'cur'},svg)}
  const os=stats(o.snaps[j]),ss=snap?stats(snap):null,ex=j===T?EXACT0:null,w=c3.w;
  const lab=c3.mode==='sde'?'SDE':'ODE';
  $('#c3-read').textContent=`t = ${f2(t)}（第 ${j} / ${T} 步）\n${lab}：平均 ${ss?f4(ss[0]):'—'}，標準差 ${ss?f4(ss[1]):'—'}，P(x>0) ${ss?f4(ss[2]):'—'}，W₁ ${c3.stale?'更新中':w&&w.sw!=null?f4(w.sw):'—'}\n`+
    (c3.mode==='sde'?`ODE：平均 ${f4(os[0])}，標準差 ${f4(os[1])}，P(x>0) ${f4(os[2])}，W₁ ${c3.stale?'更新中':w?f4(w.ow):'—'}\n`:'')+
    (ex?`精確：平均 ${f4(ex[0])}，標準差 ${f4(ex[1])}，P(x>0) ${f4(ex[2])}`:'精確邊際見直方圖虛線')}
function c3full(){c3compute();c3.w=c3metrics();c3.stale=false;c3draw();
  announce('c3',`t = ${f2(1-c3.j/c3.T)}，${c3.mode==='sde'?'SDE':'ODE'} W₁ ${c3.w.sw!=null?f3(c3.w.sw):'發散'}`+(c3.mode==='sde'?`，ODE W₁ ${f3(c3.w.ow)}`:''))}
function initC3(){const sched=frameSched(()=>{c3compute();c3draw()}),jr=$('#c3-j');
  const syncA=()=>{$('#c3-a').disabled=c3.mode==='ode'};
  segInit($('#c3-mode'),['sde','ode'],c3.mode,v=>v.toUpperCase(),v=>{c3.mode=v;syncA();c3full()});syncA();
  segInit($('#c3-T'),[5,10,20,40],c3.T,null,v=>{c3.T=v;c3.j=v;jr.max=v;jr.value=v;jupd.upd();c3full()});
  rangeInit('c3-a',v=>[(v/20).toFixed(2),`a = ${(v/20).toFixed(2)}`],v=>{c3.ak=v;c3.stale=true;sched()},v=>{c3.ak=v;c3full()});
  const jupd=rangeInit('c3-j',v=>{const t=1-v/c3.T;return[`t = ${f2(t)}`,`顯示時刻 t = ${f2(t)}，第 ${v} 步`]},v=>{c3.j=v;c3.w=c3metrics();c3draw()},v=>{c3.j=v;c3full()});
  $('#c3-ode').addEventListener('change',e=>{c3.showOde=e.target.checked;c3draw()});
  $('#c3-seed').addEventListener('click',()=>{c3.seed++;c3full()});c3full()}

/* ================= C4 · Gaussian step policy, ratio, KL ================= */
const c4={ak:14,T:10,j:5,dk:20,zk:10};
function c4draw(){const a=c4.ak/20,dv=c4.dk/100,z=c4.zk/10,eps=shared.ei/100,p=stepPolicy(c4.j,c4.T,a,dv,z),cj=klCoeff(c4.j,c4.T,a),kl2=cj*dv*dv;
  {const box=$('#c4-den'),W=widthOf(box,260,560),H=170,l=10,r=10,svg=svgOf(box,W,H),half=4*p.sd+Math.abs(p.shift),x=lin(-half,half,l,W-r),pk=1/(p.sd*Math.sqrt(2*Math.PI)),y=lin(0,1.1*pk,H-26,10);
    el('line',{x1:l,x2:W-r,y1:y(0),y2:y(0),class:'axl'},svg);[-half,0,half].forEach((v,i)=>tx(svg,x(v),H-8,(v>=0?'':'−')+g6(Math.abs(v)),{class:'tk','text-anchor':i===0?'start':i===2?'end':'middle'}));
    const curve=m=>{const pts=[];for(let i=0;i<=200;i++){const xv=-half+2*half*i/200;pts.push([x(xv),y(Math.exp(-((xv-m)**2)/(2*p.sd*p.sd))*pk)])}return lp(pts)};
    el('path',{d:curve(0),class:'cref'},svg);el('path',{d:curve(p.shift),class:'acc'},svg);
    {const s=p.shift<0?-1:1;tx(svg,x(p.shift+s*1.6*p.sd)+s*4,y(.45*pk),'πθ',{class:'lab','text-anchor':s<0?'end':'start'});tx(svg,x(-s*1.6*p.sd)-s*4,y(.45*pk),'π′',{class:'lab','text-anchor':s<0?'start':'end'})}
    const xs=x(Math.max(-half,Math.min(half,p.sd*z)));el('line',{x1:xs,x2:xs,y1:y(0),y2:16,class:'cur'},svg);el('path',{d:`M${xs} ${y(0)-14}L${xs+7} ${y(0)-7}L${xs} ${y(0)}L${xs-7} ${y(0)-7}Z`,class:'hollow'},svg);
    tx(svg,l,14,`峰值 ${g6(pk)}`,{class:'tk'})}
  {const box=$('#c4-bar'),T=c4.T,avail=widthOf(box,240,560),l=44,r=8,W=Math.max(avail,l+r+4*T),H=160,svg=svgOf(box,W,H),bw=(W-l-r)/T,y=v=>lin(-5,2,H-24,10)(Math.log10(Math.max(1e-5,Math.min(100,v))));
    [-5,-4,-3,-2,-1,0,1,2].forEach(e=>{el('line',{x1:l,x2:W-r,y1:y(10**e),y2:y(10**e),class:'grid'},svg);tx(svg,l-6,y(10**e)+4,'10'+String(e).replace('-','⁻').replace(/\d/g,d=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[d]),{class:'tk','text-anchor':'end'})});
    for(let j=0;j<T;j++){const c=klCoeff(j,T,a),x0=l+j*bw+.5;el('rect',{x:x0,y:y(c),width:Math.max(bw-1,1),height:y(1e-5)-y(c),class:'bar'},svg);if(j===c4.j)el('rect',{x:x0-1,y:y(c)-1,width:Math.max(bw-1,1)+2,height:y(1e-5)-y(c)+2,fill:'none',stroke:'var(--ink)','stroke-width':2},svg)}
    tx(svg,l,H-6,'t = 1',{class:'tk'});tx(svg,W-r,H-6,'t = 0',{class:'tk','text-anchor':'end'});
    const xc=l+(c4.j+.5)*bw;tx(svg,Math.min(W-r-40,Math.max(l+40,xc)),Math.max(22,y(cj)-6),'c_j = '+g6(cj),{class:'labm','text-anchor':'middle'})}
  const lr=Math.max(-50,Math.min(50,p.logr)),rho=Math.exp(lr),inside=rho>=1-eps&&rho<=1+eps;
  $('#c4-read').textContent=`t ${f2(p.t)} · σ ${g6(p.sg)} · 標準差 ${g6(p.sd)}\n平均差 δ ${g6(p.shift)}\nKL₁ ${g6(p.kl1)} · KL₂ ${g6(kl2)} · c_j ${g6(cj)}\nlog ρ ${g6(p.logr)} · ρ ${p.logr>50?'> e⁵⁰':p.logr<-50?'< e⁻⁵⁰':g6(rho)}\nρ ${inside?'在':'不在'}裁剪範圍 [${(1-eps).toFixed(2)}, ${(1+eps).toFixed(2)}] 內（ε = ${eps.toFixed(2)}）`;
  return{p,cj,rho}}
function c4sum(){const r=c4draw();announce('c4',`第 ${c4.j} 步，KL ${g6(r.cj*(c4.dk/100)**2)}，ρ ${g6(r.rho)}`)}
function initC4(){const sched=frameSched(c4draw),jr=$('#c4-j');
  rangeInit('c4-a',v=>[(v/20).toFixed(2),`a = ${(v/20).toFixed(2)}`],v=>{c4.ak=v;sched()},v=>{c4.ak=v;c4sum()});
  segInit($('#c4-T'),[10,40],c4.T,null,v=>{c4.T=v;c4.j=Math.floor(v/2);jr.max=v-1;jr.value=c4.j;ju.upd();c4sum()});
  const ju=rangeInit('c4-j',v=>[`${v}（t = ${f2(1-v/c4.T)}）`,`步 ${v}，t = ${f2(1-v/c4.T)}`],v=>{c4.j=v;sched()},v=>{c4.j=v;c4sum()});
  rangeInit('c4-d',v=>[(v/100).toFixed(2),`Δv = ${(v/100).toFixed(2)}`],v=>{c4.dk=v;sched()},v=>{c4.dk=v;c4sum()});
  rangeInit('c4-z',v=>[(v/10).toFixed(1),`z = ${(v/10).toFixed(1)}`],v=>{c4.zk=v;sched()},v=>{c4.zk=v;c4sum()});c4draw()}

/* ================= timeline ================= */
const TL_GROUP={'2006.11239':['df','flowgrpo-diffusion-sde-background'],'2010.02502':['df','flowgrpo-diffusion-sde-background'],'2011.13456':['df','flowgrpo-diffusion-sde-background'],
 '2209.03003':['fm','flowgrpo-flow-matching-backbone'],'2210.02747':['fm','flowgrpo-flow-matching-backbone'],'2303.08797':['fm','flowgrpo-flow-matching-backbone'],'2403.03206':['fm','flowgrpo-flow-matching-backbone'],
 '1910.00177':['rl','rl-finetuning-text-to-image-diffusion'],'2305.13301':['rl','rl-finetuning-text-to-image-diffusion'],'2305.18290':['rl','rl-finetuning-text-to-image-diffusion'],'2409.08861':['rl','rl-finetuning-text-to-image-diffusion'],
 '2402.03300':['gr','flowgrpo-denoising-mdp-and-grpo'],'2505.05470':['me','flowgrpo-design-choices-and-results']};
const TL_NAME={df:'擴散模型與分數 SDE',fm:'流匹配與統一框架',rl:'生成模型的 RL 與偏好對齊',gr:'LLM 的 GRPO',me:'本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.slice().sort((a,b)=>(Number(a.year)||0)-(Number(b.year)||0)||String(a.arxiv).localeCompare(String(b.arxiv)));
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];used.add(g[0]);const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div><div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${g[1]?`<a href="${g[1]}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
  const on=new Set(used);
  Object.keys(TL_NAME).filter(k=>used.has(k)).forEach(k=>{const b=document.createElement('button');b.type='button';b.className='chip';b.setAttribute('aria-pressed','true');b.textContent=TL_NAME[k];b.dataset.g=k;
    b.addEventListener('click',()=>{if(on.has(k)&&on.size===used.size){on.clear();on.add(k)}else if(on.has(k)){on.delete(k);if(!on.size)used.forEach(x=>on.add(x))}else on.add(k);
      $$('.chip',chips).forEach(c=>c.setAttribute('aria-pressed',String(on.has(c.dataset.g))));$$('.ev',track).forEach(ev=>ev.classList.toggle('dim',!on.has(ev.dataset.g)))});chips.appendChild(b)});
  const rail=$('#rail'),by=dir=>rail.scrollBy({left:dir*Math.min(rail.clientWidth*.8,560),behavior:RM?'auto':'smooth'});
  $('#tl-prev').addEventListener('click',()=>by(-1));$('#tl-next').addEventListener('click',()=>by(1));
  rail.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){by(1);e.preventDefault()}if(e.key==='ArrowLeft'){by(-1);e.preventDefault()}});
  return data}

/* ================= toc, theme ================= */
function initToc(){const secs=$$('[data-toc]'),ol=$('#toc'),bar=$('#tocbar');
  secs.forEach(s=>{ol.insertAdjacentHTML('beforeend',`<li><a href="#${s.id}"><span class="n">${s.dataset.toc}</span><span>${s.dataset.tt}</span></a></li>`);bar.insertAdjacentHTML('beforeend',`<a href="#${s.id}">${s.dataset.toc} ${s.dataset.tt}</a>`)});
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;$$('.toc a,.tocbar a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+id));const ab=$(`.tocbar a[href="#${id}"]`);if(ab&&bar.offsetParent)bar.scrollTo({left:ab.offsetLeft-16,behavior:RM?'auto':'smooth'})}})},{rootMargin:'-40% 0px -55% 0px'});
  secs.forEach(s=>io.observe(s));addEventListener('scroll',()=>$('#appbar').classList.toggle('raised',scrollY>8),{passive:true})}
function initTheme(){let saved=null;try{saved=localStorage.getItem('pm-theme')}catch(e){}if(saved)document.documentElement.dataset.theme=saved;
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){}})}

/* ================= self test (acceptance tables: design.md section 5) ================= */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t=1e-4)=>Math.abs(a-b)<=t+1e-12,rel=(a,b,t=1e-5)=>Math.abs(a-b)<=t*Math.abs(b)+1e-15;
  {const r=mulberry32(1);ok([.627074,.002736,.527447].every(v=>near(r(),v,1e-6)),'mulberry32 seed 1');const g=mulberry32(1);ok(near(gauss(g),1.4043)&&near(gauss(g),1.2158),'gauss seed 1')}
  ok(near(velocity(.5,.5),-1.0646)&&near(velocity(-1,1),-1.24)&&near(velocity(1,.1),-1.2183),'velocity');
  ok(near(density(0,0),.0004)&&near(density(1.2,0),.7979)&&near(density(0,1),.3989),'density');
  ok(near(sigmaT(.5,.7,10),.7)&&near(sigmaT(1,.7,10),2.1)&&near(sigmaT(1,.7,40),4.3715),'sigma');
  {const r=mulberry32(1),R=[];for(let i=0;i<24;i++)R.push(drawReward(r,'continuous',.3));const {A,m,s}=advantages(R);let s2=0;A.forEach(a=>s2+=a*a);
    ok(near(R[0],.7107)&&near(R[1],.6824)&&near(R[2],.4234)&&near(m,.5272)&&near(s,.1235)&&near(A[0],1.4858)&&near(A[1],1.2568)&&near(A[2],-.8398)&&near(s2,24),'C1 continuous group')}
  {const r=mulberry32(1),R=[];for(let i=0;i<24;i++)R.push(drawReward(r,'binary',.3));const {A,m,s}=advantages(R);
    ok(R.join('')==='010001000000101101001110'&&near(m,.375)&&near(s,.4841)&&near(Math.max(...A),1.291)&&near(Math.min(...A),-.7746),'C1 binary group')}
  ok(advantages([1,1]).A.every(a=>a===0),'C1 degenerate group');
  [['continuous',.3,[[2,.9228],[6,.5156],[12,.3576],[24,.2536],[48,.1788]]],['binary',.3,[[2,.7022,.572],[6,.4167,.116],[12,.3204,.0165],[24,.2276,.0015],[48,.1583,0]]],['binary',.05,[[2,.845,.9005],[6,.5751,.7385],[12,.4238,.5425],[24,.315,.282],[48,.2795,.076]]]]
    .forEach(([kind,p,rows])=>rows.forEach(([G,rm,dg])=>{const [a,b]=advMC(G,kind,p,7);ok(near(a,rm)&&(dg===undefined||near(b,dg)),`advMC ${kind} p=${p} G=${G}`)}));
  ok(near(pDegenerate(24,'binary',.05),.292)&&near(pDegenerate(2,'binary',.3),.58)&&near(pDegenerate(24,'binary',.3),.0002),'pDegenerate');
  [[1,.7,.7,1],[1,.9,.9,1],[1,1,1,1],[1,1.1,1.1,1],[1,1.3,1.2,0],[-1,.7,-.8,0],[-1,.9,-.9,-1],[-1,1,-1,-1],[-1,1.1,-1.1,-1],[-1,1.3,-1.3,-1]].forEach(([A,r,L,g])=>{const [l,gg]=clipObj(r,A,.2);ok(near(l,L)&&gg===g,`clip A=${A} rho=${r}`)});
  {const sv=[c2.ai,c2.ri,shared.ei];c2.ai=10;c2.ri=120;shared.ei=20;ok(!c2draw().clipped,'C2 boundary rho=1+eps is unclipped');[c2.ai,c2.ri,shared.ei]=sv;c2draw()}
  {const a=simulate(0,10,1,true).snaps[10],b=simulate(0,10,1,false).snaps[10];let mx=0;for(let i=0;i<a.length;i++)mx=Math.max(mx,Math.abs(a[i]-b[i]));ok(mx===0&&near(a[0],1.4485),'C3 a=0 equals ODE')}
  const C3T=[['ode',0,5,.272,1.0705,.6178,.1426],['ode',0,10,.2554,1.1402,.6075,.0712],['ode',0,40,.2438,1.1953,.5998,.0175],['sde',.1,10,.2602,1.1411,.61,.0725],['sde',.7,10,.2315,1.1756,.5975,.0429],['sde',1.5,10,.1986,1.1995,.583,.0481],
    ['sde',2.5,10,.1093,1.2343,.5487,.1375],['sde',.7,5,.2401,1.1368,.5988,.0841],['sde',2.5,5,.0085,15.4839,.503,2.0233],['sde',.1,40,.2438,1.1963,.602,.017],['sde',.7,40,.2086,1.2075,.5877,.0356],['sde',1.5,40,.1992,1.214,.583,.041],['sde',2.5,40,.2167,1.2174,.5903,.0237]];
  C3T.forEach(([md,a,T,m,s,p,w])=>{const sim=simulate(a,T,1,md==='sde'),x=sim.snaps[T],st=stats(x);ok(sim.divergedAt==null&&near(st[0],m)&&near(st[1],s)&&near(st[2],p)&&near(w1(x,0),w,5e-4),`C3 ${md} a=${a} T=${T}`)});
  {const sn=simulate(.7,10,1,true).snaps;ok(near(sn[0][0],1.4043)&&near(sn[1][0],1.143)&&near(sn[2][0],.5522)&&near(sn[3][0],-.4426)&&near(w1(sn[5],.5),.0297,5e-4),'C3 snapshots');
   ok(near(w1(simulate(0,40,1,false).snaps[20],.5),.0147,5e-4),'C3 ODE T=40 t=0.5')}
  const C4T=[[5,10,.7,.2,1,.7,.221359,-.0249,.00632663,-.118813],[0,10,.7,.2,1,2.1,.664078,-.02,.000453515,-.0305704],[9,10,.7,.2,1,.233333,.0737865,-.0249,.0569397,-.3944],[9,10,.1,.2,1,.0333333,.0105409,-.0201,1.81805,-3.7249],
    [9,10,1.4,.2,1,.466667,.147573,-.0396,.0360037,-.304346],[20,40,.7,.2,1,.7,.11068,-.006225,.00158166,-.057825],[39,40,.7,.2,1,.11209,.0177229,-.006225,.0616847,-.412924]];
  C4T.forEach(([j,T,a,dv,z,sg,sd,sh,kl,lr])=>{const p=stepPolicy(j,T,a,dv,z);ok(rel(p.sg,sg)&&rel(p.sd,sd)&&rel(p.shift,sh)&&rel(p.kl1,kl)&&rel(p.kl2,kl)&&rel(klCoeff(j,T,a)*dv*dv,kl)&&rel(p.logr,lr),`C4 (${j},${T},${a})`)});
  {const sum=(T,a)=>{let s=0;for(let j=0;j<T;j++)s+=klCoeff(j,T,a);return s};ok(rel(sum(10,.7),3.06231)&&rel(sum(10,.1),97.9709)&&rel(sum(10,1.4),1.932)&&rel(sum(10,2),2.17148)&&rel(sum(40,.7),5.18619),'C4 coefficient sums');
   ok(rel(klCoeff(0,10,.7),.0113379)&&rel(klCoeff(9,10,.7),1.42349)&&rel(klCoeff(39,40,.7),1.54212),'C4 coefficients')}
  {const p=stepPolicy(5,10,.7,0,1);ok(p.kl1===0&&p.kl2===0&&p.logr===0&&rel(klCoeff(5,10,.7),.158166)&&Number.isFinite(klCoeff(5,10,.7)),'C4 dv=0 has no NaN')}
  {const nodes=[...$$('.readout'),...$$('svg text'),...$$('[aria-label]'),...$$('[aria-valuetext]')];const bad=nodes.filter(n=>/[̀-ͯ]/.test(n.textContent+(n.getAttribute('aria-label')||'')+(n.getAttribute('aria-valuetext')||'')));ok(bad.length===0,'no combining marks in dynamic text: '+bad.length)}
  {const w=$('#s8-cost');ok(!!w&&w.textContent.includes('T = 10')&&w.textContent.includes('T = 40'),'S8 diagram labels')}
  {const fb=$('#math-fallback'),was=fb.classList.contains('hide');mathFallback(true);const shown=!fb.classList.contains('hide');mathFallback(!was);ok(shown,'KaTeX fallback notice')}
  {const tl=JSON.parse($('#timeline-data').textContent);ok(tl.length===13&&tl.every(e=>TL_GROUP[e.arxiv]&&!('role' in e)&&!('source' in e)),'timeline groups')}
  ok(!!document.querySelector('.katex')||!window.katex,'KaTeX rendered');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ================= boot ================= */
function mathFallback(on){$('#math-fallback').classList.toggle('hide',!on)}
initTheme();initToc();renderHero();renderCost();initC1();initC2();initC3();initC4();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderHero();renderCost();c1draw();c2draw();c3draw();c4draw()})});
function renderMath(){if(typeof renderMathInElement!=='function')return false;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false});return true}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex')&&!renderMath())mathFallback(true);if(/[?&]test\b/.test(location.search))selfTest()});
})();
