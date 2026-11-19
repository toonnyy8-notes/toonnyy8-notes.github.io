(()=>{
'use strict';
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
const LN2=Math.LN2;
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
  (o.yt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:(o.yf?o.yf(v):String(v)),yy=Y(val);el('line',{x1:x0,x2:x1,y1:yy,y2:yy,class:'grid'},g);tx(g,x0-8,yy+4,lab,{class:'tk','text-anchor':'end'})});
  (o.xt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:(o.xf?o.xf(v):String(v)),xx=X(val);el('line',{x1:xx,x2:xx,y1:y0,y2:y0+4,class:'axl'},g);tx(g,xx,y0+18,lab,{class:'tk','text-anchor':'middle'})});
  el('line',{x1:x0,x2:x1,y1:y0,y2:y0,class:'axl'},g);
  if(o.xl)tx(g,(x0+x1)/2,o.H-4,o.xl,{class:'al','text-anchor':'middle'});
  if(o.yl)tx(g,4,12,o.yl,{class:'al'});
  return {X,Y,g,x0,x1,y0,y1};
}
function arrow(g,x0,y0,x1,y1,cls,dash){const dx=x1-x0,dy=y1-y0,L=Math.hypot(dx,dy);if(L<2)return;const ux=dx/L,uy=dy/L,h=Math.min(9,L*.45);
  el('line',{x1:x0,y1:y0,x2:x1-ux*h*.7,y2:y1-uy*h*.7,class:cls+(dash?' dash':'')},g);
  el('path',{d:`M${x1} ${y1}L${x1-ux*h-uy*h*.45} ${y1-uy*h+ux*h*.45}L${x1-ux*h+uy*h*.45} ${y1-uy*h-ux*h*.45}Z`,class:cls+'-f'},g)}
function darrow(g,x0,y0,x1,y1,cls){const mx=(x0+x1)/2,my=(y0+y1)/2;arrow(g,mx,my,x0,y0,cls);arrow(g,mx,my,x1,y1,cls)}
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}
function ticks(lo,hi,n){const raw=(hi-lo)/n,mag=10**Math.floor(Math.log10(raw)),st=[1,2,5,10].find(k=>k*mag>=raw-1e-15)*mag,out=[];for(let v=Math.ceil(lo/st-1e-9)*st;v<=hi+1e-12;v+=st)out.push(+v.toPrecision(10));return out}
const fz=(v,n)=>(Math.abs(v)<.5*10**-n?0:v).toFixed(n);
function fe(v,n=3){if(v===0)return'0';const a=Math.abs(v);return a<1e-3||a>=1e5?v.toExponential(n-1).replace('e','×10^').replace('^+','^').replace(/\^(-?\d+)/,(m,e)=>supd(e)):v.toPrecision(n)}
function supd(e){const S={'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'};return[...e].map(c=>S[c]).join('')}
function niceList(lo,hi){const s=new Set();for(let e=-3;e<=1;e++)for(let m=10;m<100;m++){const v=+(m*10**(e-1)).toPrecision(2);if(v>=lo-1e-12&&v<=hi+1e-12)s.add(v)}return[...s].sort((a,b)=>a-b)}
const fmtV=v=>v<.1?v.toFixed(3):v.toFixed(2);

/* time-sliced jobs: a newer job with the same key cancels the older one */
const JOB={};
function runJob(key,gen,onDone,onProg){const id=(JOB[key]=(JOB[key]||0)+1);
  const step=()=>{if(JOB[key]!==id)return;const t0=performance.now();let r;
    do{r=gen.next();if(!r.done&&r.value!==undefined&&onProg)onProg(r.value)}while(!r.done&&performance.now()-t0<30);
    if(JOB[key]!==id)return;if(r.done)onDone(r.value);else setTimeout(step,0)};
  setTimeout(step,0)}
function runSync(gen){let r;while(!(r=gen.next()).done);return r.value}

/* ---------- enumeration on {0,1}^6 (C2, C3; same algorithm as verify/w23_exact.py) ---------- */
const ED=6,EN=64,POP=new Uint8Array(EN),HAM=new Uint8Array(EN*EN);
for(let x=0;x<EN;x++){let c=0,y=x;while(y){c+=y&1;y>>=1}POP[x]=c}
for(let x=0;x<EN;x++)for(let y=0;y<EN;y++)HAM[x*EN+y]=POP[x^y];
const PNAME={codebook:'碼本',copy:'複製',parity:'同位',indep:'獨立'};
function preset(name){const q=new Float64Array(EN);
  if(name==='copy'){q[0]=q[EN-1]=.5}
  else if(name==='parity'){for(let x=0;x<EN;x++)if(POP[x]%2===0)q[x]=1}
  else if(name==='codebook'){[5,10,19,28,38,45,51,60].forEach(c=>q[c]=1)}
  else for(let x=0;x<EN;x++)q[x]=.8**POP[x]*.2**(ED-POP[x]);
  let s=0;for(const v of q)s+=v;for(let x=0;x<EN;x++)q[x]/=s;return q}
const PRE={};for(const k in PNAME)PRE[k]=preset(k);
function kernelW(t){const a=Math.exp(-t),b=(1-a)/2,w=new Float64Array(ED+1);for(let k=0;k<=ED;k++)w[k]=(a+b)**(ED-k)*b**k;return w}
function push(q,t){const w=kernelW(t),o=new Float64Array(EN);for(let x=0;x<EN;x++){const qx=q[x];if(!qx)continue;for(let y=0;y<EN;y++)o[y]+=qx*w[HAM[x*EN+y]]}return o}
function H(p){let s=0;for(const v of p)if(v>0)s-=v*Math.log(v);return s}
function dtc(p){const hx=H(p),m=new Float64Array(EN/2);let s=0;for(let i=0;i<ED;i++){const b=1<<i;let k=0;for(let x=0;x<EN;x++)if(!(x&b))m[k++]=p[x]+p[x|b];s+=hx-H(m)}return hx-s}
function tcOf(p){let s=0;for(let i=0;i<ED;i++){let p1=0;for(let x=0;x<EN;x++)if(x>>i&1)p1+=p[x];s+=H([p1,1-p1])}return s-H(p)}
function tcCond(q,s,h){const qs=push(q,s),wh=kernelW(h),J=new Float64Array(EN*EN),qt=new Float64Array(EN);
  for(let x=0;x<EN;x++){const a=qs[x];for(let y=0;y<EN;y++){const v=a*wh[HAM[x*EN+y]];J[x*EN+y]=v;qt[y]+=v}}
  const hq=H(qt),hc=H(J)-hq,j1=new Float64Array(EN),both=new Float64Array(2*EN),m=new Float64Array(EN*ED);let tot=0;
  for(let i=0;i<ED;i++){j1.fill(0);for(let x=0;x<EN;x++)if(x>>i&1)for(let y=0;y<EN;y++)j1[y]+=J[x*EN+y];
    for(let y=0;y<EN;y++){both[y]=qt[y]-j1[y];both[EN+y]=j1[y];m[y*ED+i]=qt[y]>1e-300?Math.min(1,Math.max(0,j1[y]/qt[y])):.5}
    tot+=H(both)-hq}
  return{tc:tot-hc,qs,qt,m}}
const cS=(s,h)=>Math.expm1(2*h)/(-Math.expm1(-s));
function schedule(a,delta=.001,T=6){const u0=Math.expm1(delta),U=Math.expm1(T),us=[u0];while(us[us.length-1]<U)us.push(Math.min((1+a)*us[us.length-1],U));return us.map(Math.log1p)}
function* scheduleGen(q,a){const ts=schedule(a),n=ts.length-1;let sumTC=0,sumB=0,maxC=0,p=new Float64Array(EN).fill(1/EN);const steps=[];
  for(let j=n-1;j>=0;j--){const s=ts[j],t=ts[j+1],h=t-s,r=tcCond(q,s,h),c=cS(s,h),dd=dtc(r.qs)-dtc(r.qt);
    maxC=Math.max(maxC,c);sumTC+=r.tc;sumB+=c*dd;steps.push({s,t,tc:r.tc,b:c*dd});
    const np=new Float64Array(EN);
    for(let y=0;y<EN;y++){const py=p[y];if(!py)continue;const mo=y*ED;for(let x=0;x<EN;x++){let v=py;for(let i=0;i<ED;i++){const mm=r.m[mo+i];v*=(x>>i&1)?mm:1-mm}np[x]+=v}}
    p=np;if((n-j)%25===0)yield}
  const qhat=push(p,ts[0]),qT=push(q,ts[n]);let init=0,kl=0;
  for(let x=0;x<EN;x++){if(qT[x]>0)init+=qT[x]*Math.log(qT[x]*EN);if(q[x]>0)kl+=q[x]*Math.log(q[x]/qhat[x])}
  const end=H(push(q,ts[0]))-H(q);
  return{ts,N:n,a,maxC,sumTC,sumB,threeA:3*a*dtc(q),init,initB:Math.exp(-ts[n])*ED*LN2,end,endB:ED*ts[0]*Math.log(2*Math.E/ts[0]),path:init+sumTC+end,kl,steps}}
const dtcCache=new Map();
function dtcCurve(name,t0,t1,n){const key=name+t0+t1+n;if(dtcCache.has(key))return dtcCache.get(key);const out=[];
  for(let i=0;i<n;i++){const t=t0*(t1/t0)**(i/(n-1));out.push([t,dtc(push(PRE[name],t))])}dtcCache.set(key,out);return out}

/* ---------- information curves, critical points, recovery (C1, C4; same as verify/w4_window.py) ---------- */
const LF=new Float64Array(4098);for(let n=1;n<LF.length;n++)LF[n]=LF[n-1]+Math.log(n);
const h2=p=>-p*Math.log(p)-(1-p)*Math.log1p(-p);
const Iunif=t=>LN2-h2((1-Math.exp(-t))/2);
const WG=new Float64Array(401),WW=new Float64Array(401);
for(let k=0;k<401;k++){const w=-10+k*.05;WG[k]=w;WW[k]=(k===0||k===400?1:k%2?4:2)*(.05/3)*Math.exp(-w*w/2)/Math.sqrt(2*Math.PI)}
const logcosh=x=>{x=Math.abs(x);return x+Math.log1p(Math.exp(-2*x))-LN2};
function Igauss(t){const u=Math.exp(-t)/Math.sqrt(-Math.expm1(-2*t)),u2=u*u;let s=0;for(let k=0;k<401;k++)s+=WW[k]*logcosh(u2+u*WG[k]);return u2-s}
function bisect(f,lo,hi,it=80){let flo=f(lo);for(let i=0;i<it;i++){const mid=(lo+hi)/2,fm=f(mid);if((fm>0)===(flo>0)){lo=mid;flo=fm}else hi=mid}return(lo+hi)/2}
function* bisectGen(f,lo,hi,it){let flo=f(lo);yield;for(let i=0;i<it;i++){const mid=(lo+hi)/2,fm=f(mid);yield;if((fm>0)===(flo>0)){lo=mid;flo=fm}else hi=mid}return(lo+hi)/2}
const tOfIunif=I=>{const b=bisect(b=>LN2-h2(b)-I,1e-15,.5-1e-15);return-Math.log1p(-2*b)};
const tStarUnif=k=>tOfIunif(k),tStarGauss=k=>bisect(t=>Igauss(t)-k,1e-4,10);
const maskedRec=lam=>lam<1e-12?1:-Math.expm1(-lam)/lam;
const maskedRecI=(d,k,I)=>{const e=d*(k-I);return e>700?0:maskedRec(Math.exp(e))};
const L80=bisect(x=>maskedRec(Math.exp(x))-.8,-10,10),L20=bisect(x=>maskedRec(Math.exp(x))-.2,-10,10),MASK_LNRATIO=L20-L80;
const XHI=Math.log(40),XLO=Math.log(1e-8);
const NV=161,VV=new Float64Array(NV),VW=new Float64Array(NV),VB=new Float64Array(NV);let VZ0=0;
for(let i=0;i<NV;i++){const v=-36+.25*i;VV[i]=v;VW[i]=(i===0||i===NV-1?1:i%2?4:2)*.25/3;VB[i]=v-Math.exp(v);VZ0+=VW[i]*Math.exp(VB[i])}
const lae=(a,b)=>{if(a===-Infinity)return b;if(b===-Infinity)return a;const m=Math.max(a,b);return m+Math.log1p(Math.exp(-Math.abs(a-b)))};
function makeUnifRec(d,kappa){
  const ll=new Float64Array(d+1);for(let j=0;j<=d;j++)ll[j]=kappa*d+LF[d]-LF[j]-LF[d-j]-d*LN2;
  let S=null;const memo=new Map();
  function setup(t){if(S&&S.t===t)return S;const beta=(1-Math.exp(-t))/2,L=Math.log(beta/(1-beta)),P0=new Float64Array(d+2),S1=new Float64Array(d+2);
    let acc=-Infinity;for(let j=0;j<=d;j++){P0[j]=acc;acc=lae(acc,ll[j])}P0[d+1]=acc;
    S1[d+1]=-Infinity;acc=-Infinity;for(let j=d;j>=0;j--){acc=lae(acc,ll[j]+j*L);S1[j]=acc}
    return S={t,beta,L,P0,S1}}
  function G(v,D){const L=S.L,c=v-D*L;const ja=Math.min(d+1,Math.max(0,Math.ceil((c-XHI)/(-L))));const jb=Math.min(d+1,Math.max(ja,Math.floor((c-XLO)/(-L))+1));
    const parts=[S.P0[ja],c+S.S1[jb]];for(let j=ja;j<jb;j++){const x=c+j*L;parts.push(ll[j]+Math.log(-Math.expm1(-Math.exp(x))))}
    let mx=-Infinity;for(const p of parts)if(p>mx)mx=p;if(mx>700)return Infinity;let s=0;for(const p of parts)if(p>-745)s+=Math.exp(p);return s}
  function pD(D){let tot=0;for(let i=0;i<NV;i++){const g=G(VV[i],D);if(g<700)tot+=VW[i]*Math.exp(VB[i]-g)}return Math.min(1,tot/VZ0)}
  function rec(t){setup(t);const beta=S.beta,sd=Math.sqrt(d*beta*(1-beta)),D0=Math.max(0,Math.floor(d*beta-7*sd-2)),D1=Math.min(d,Math.ceil(d*beta+7*sd+2)),n=D1-D0+1;
    const pmf=new Float64Array(n);for(let k=0;k<n;k++){const D=D0+k;pmf[k]=Math.exp(LF[d]-LF[D]-LF[d-D]+D*Math.log(beta)+(d-D)*Math.log1p(-beta))}
    const cache=new Map(),P=D=>{if(!cache.has(D))cache.set(D,pD(D));return cache.get(D)};
    const below=th=>{let lo=0,hi=n;while(lo<hi){const mid=(lo+hi)>>1;if(P(D0+mid)<th)hi=mid;else lo=mid+1}return lo};
    const b1=below(1-1e-10),b2=below(1e-10);let s=0;for(let k=0;k<b1;k++)s+=pmf[k];for(let k=b1;k<b2;k++)s+=pmf[k]*P(D0+k);return s}
  function recI(I){if(!memo.has(I))memo.set(I,rec(tOfIunif(I)));return memo.get(I)}
  function* widthGen(lo,hi){lo=lo??Math.max(1e-4,kappa-.2);hi=hi??Math.min(LN2-1e-4,kappa+.2);
    const rl=recI(lo);yield;const rh=recI(hi);yield;if(rl>.2||rh<.8)return{i20:NaN,i80:NaN,w:NaN};
    const i80=yield* bisectGen(I=>recI(I)-.8,lo,hi,26),i20=yield* bisectGen(I=>recI(I)-.2,lo,hi,26);return{i20,i80,w:i80-i20}}
  return{rec,recI,widthGen}}

/* ---------- lower-bound counting (C5; same as verify/w5_count.py) ---------- */
function counts(d,eta,rho=.2){const K=Math.floor(.5*d/LN2)-Math.ceil(.1*d/LN2)+1,ell=Math.ceil(Math.log2(d/(eta*rho**2))),nM=Math.ceil(K/(2*ell+1)),Lam=Math.log(d/(eta*rho**2)),nU=Math.ceil(Math.sqrt(d/Lam));
  return{K,ell,nM,Lam,nU,qmax:rho**2*K/(4*(2*ell+1))}}
const eq100=(c,eta,Q)=>[(2*c.ell+1)*Q/c.K,2**(1-c.ell)/eta,Q*2**(-c.ell)];

/* ---------- chips helper ---------- */
function chips(box,cur,onPick){box.innerHTML='';for(const k in PNAME){const b=document.createElement('button');b.type='button';b.className='chip';b.textContent=PNAME[k];b.dataset.v=k;b.setAttribute('aria-pressed',String(k===cur));
  b.addEventListener('click',()=>{$$('.chip',box).forEach(c=>c.setAttribute('aria-pressed',String(c===b)));onPick(k)});box.appendChild(b)}}
function logSlider(inp,list,v0){inp.max=list.length-1;inp.value=list.indexOf(v0)}

/* ---------- C1 ---------- */
const c1={seed:11,cells:[]};
const c1expect=t=>({touch:-Math.expm1(-t),change:-Math.expm1(-t)/2});
function c1sample(){const r=mulberry32(c1.seed);c1.cells=[];for(let i=0;i<32;i++){const x=r()<.5?-1:1,u=r(),rr=r()<.5?-1:1,g=gauss(r);c1.cells.push({x,u,r:rr,g})}}
function renderC1(){const t=+$('#c1-t').value,e=c1expect(t),a=Math.exp(-t),sg=t===0?0:Math.sqrt(-Math.expm1(-2*t));
  $('#c1-to').textContent=t.toFixed(2);$('#c1-t').setAttribute('aria-valuetext','t = '+t.toFixed(2));
  let nT=0,nC=0;const rows={x:[],u:[],m:[],g:[]},tab=[];
  c1.cells.forEach((c,i)=>{const touched=c.u<e.touch,uv=touched?c.r:c.x,chg=touched&&c.r!==c.x,gv=a*c.x+sg*c.g;if(touched)nT++;if(chg)nC++;
    const pm=v=>v>0?'+':'−';
    rows.x.push(`<span class="cell${c.x>0?' pos':''}">${pm(c.x)}</span>`);
    rows.u.push(`<span class="cell${uv>0?' pos':''}${chg?' chg':touched?' rs':''}">${pm(uv)}</span>`);
    rows.m.push(touched?'<span class="cell star">⋆</span>':`<span class="cell${c.x>0?' pos':''}">${pm(c.x)}</span>`);
    const cl=Math.max(-3,Math.min(3,gv)),hpx=Math.abs(cl)/3*20,top=cl>=0?22-hpx:22;
    rows.g.push(`<span class="cell g"><span class="mid"></span><span class="bar" style="top:${top.toFixed(1)}px;height:${Math.max(1,hpx).toFixed(1)}px"></span>${gv>3?'<span class="cl" style="top:1px">▲</span>':gv<-3?'<span class="cl" style="top:36px">▼</span>':''}<span class="gv">${gv<0?'−':''}${Math.abs(gv).toFixed(1)}</span></span>`);
    tab.push(`<tr><td class="num">${i+1}</td><td>${pm(c.x)}</td><td>${pm(uv)}${chg?'（改變）':touched?'（重抽，未變）':''}</td><td>${touched?'⋆':pm(c.x)}</td><td class="num">${gv<0?'−':''}${Math.abs(gv).toFixed(2)}</td></tr>`)});
  const row=(lab,sub,cells,aria)=>`<div class="c1row" role="img" aria-label="${aria}"><span class="c1lab">${lab}<b>${sub}</b></span><div class="cells" aria-hidden="true">${cells.join('')}</div></div>`;
  $('#c1-rows').innerHTML=row('X₀','乾淨序列',rows.x,`X₀ 列：32 格中 ${c1.cells.filter(c=>c.x>0).length} 格為 +`)+
    row('uniform','重抽',rows.u,`uniform 列：32 格中 ${nT} 格被重抽，${nC} 格改變`)+
    row('masked','遮罩 ⋆',rows.m,`masked 列：32 格中 ${nT} 格變成遮罩`)+
    row('Gaussian','e⁻ᵗX₀ + σₜG',rows.g,`Gaussian 列：每格為 e^(−t)·X₀ + σ_t·G，t = ${t.toFixed(2)}`);
  $('#c1-tab tbody').innerHTML=tab.join('');
  $('#c1-ro').innerHTML=`觸及比例（uniform 與 masked 相同）：<b>${nT}/32 = ${(nT/32).toFixed(3)}</b>，期望 1 − e⁻ᵗ = <b class="acc">${e.touch.toFixed(4)}</b><br>uniform 實際改變比例：<b>${nC}/32 = ${(nC/32).toFixed(3)}</b>，期望 β_t = (1 − e⁻ᵗ)/2 = <b class="acc">${e.change.toFixed(4)}</b>`;
  announce('c1',`t = ${t.toFixed(2)}：${nT} 格被觸及，uniform 有 ${nC} 格改變`)}
function initC1(){c1sample();$('#c1-t').addEventListener('input',renderC1);$('#c1-re').addEventListener('click',()=>{c1.seed++;c1sample();renderC1()});renderC1()}

/* ---------- C2 ---------- */
const S_LIST=niceList(.01,3),H_LIST=niceList(.01,1.5),A_LIST=niceList(.02,1);
const c2={p:'codebook'};
const HMIN=1e-9*ED*LN2,zr=v=>Math.abs(v)<HMIN?0:v;
const LOGT=[[.01,'0.01'],[.1,'0.1'],[1,'1'],[6,'6']];
function c2calc(p,s,h){const q=PRE[p],r=tcCond(q,s,h),ds=zr(dtc(r.qs)),dt=zr(dtc(r.qt)),dd=zr(ds-dt),c=cS(s,h),tc=zr(r.tc),bd=c*dd;return{c,ds,dt,dd,tc,bd,ratio:bd>1e-9?tc/bd:NaN}}
function renderC2(){const s=S_LIST[+$('#c2-s').value],h=H_LIST[+$('#c2-h').value],t=s+h,R=c2calc(c2.p,s,h);
  $('#c2-so').textContent=fmtV(s);$('#c2-ho').textContent=fmtV(h);$('#c2-s').setAttribute('aria-valuetext','s = '+fmtV(s));$('#c2-h').setAttribute('aria-valuetext','h = '+fmtV(h));
  const box=$('#c2-fig'),W=widthOf(box,280,560),D0=dtc(PRE[c2.p]),ymax=c2.p==='indep'?.1:D0*1.05;
  const F=frame($('svg',box),{W,H:270,m:{l:46,r:18,t:28,b:40},x:[.01,6],xlog:true,y:[0,ymax],xt:LOGT,yt:ticks(0,ymax,4),yf:v=>String(v),xl:'noise level t（對數刻度）',yl:'DTC(q_t)（nats）'});
  el('path',{d:lp(dtcCurve(c2.p,.01,6,80).map(([tt,v])=>[F.X(tt),F.Y(v)])),class:'unif'},F.g);
  const xs=F.X(s),xt=F.X(Math.min(6,t)),ys=F.Y(R.ds),yt=F.Y(R.dt);
  el('line',{x1:xs,x2:xs,y1:F.y0,y2:F.y1,class:'ref'},F.g);el('line',{x1:xt,x2:xt,y1:F.y0,y2:F.y1,class:'ref'},F.g);
  tx(F.g,xs-4,F.y1+12,'s',{class:'labm','text-anchor':'end'});tx(F.g,xt+4,F.y1+12,'t = s + h',{class:'labm'});
  const right=xt+90<F.x1,ax=right?xt+12:xs-12;
  el('line',{x1:xs,x2:ax,y1:ys,y2:ys,class:'cur'},F.g);el('line',{x1:xt,x2:ax,y1:yt,y2:yt,class:'cur'},F.g);
  if(Math.abs(ys-yt)>8)darrow(F.g,ax,ys,ax,yt,'acc');
  tx(F.g,right?ax+6:ax-6,(ys+yt)/2+4,'DTC 落差',{class:'lab','text-anchor':right?'start':'end',style:'fill:var(--accent)'});
  el('circle',{cx:xs,cy:ys,r:4.5,class:'acc-f'},F.g);el('circle',{cx:xt,cy:yt,r:4.5,class:'hollow'},F.g);
  $('svg',box).setAttribute('aria-label',`DTC(q_t) 對 t 的曲線；s = ${fmtV(s)}，t = ${fmtV(t)}，DTC 落差 = ${R.dd.toFixed(6)}`);
  // bars
  const bb=$('#c2-bars'),Wb=widthOf(bb,260,460),sv=$('svg',bb);sv.textContent='';sv.setAttribute('viewBox',`0 0 ${Wb} 200`);sv.setAttribute('width',Wb);sv.setAttribute('height',200);
  const mx=Math.max(R.bd,R.tc),bw=Wb-24,sc=v=>mx>1e-9?Math.max(1,v/mx*bw):0;
  tx(sv,0,30,'TC(Xₛ | Xₜ)（精確）',{class:'lab'});el('rect',{x:0,y:40,width:sc(R.tc),height:28,rx:3,class:'tcf'},sv);
  tx(sv,0,96,'上界 cₛ(h)·DTC 落差',{class:'lab'});el('rect',{x:.75,y:106,width:Math.max(0,sc(R.bd)-1.5),height:28,rx:3,class:'bnd',style:'stroke-width:1.5'},sv);
  const vt=(v,y,w)=>{const inside=w>bw*.55;tx(sv,inside?w-8:w+8,y,v,{class:'labm','text-anchor':inside?'end':'start',style:inside?'fill:var(--surface)':''})};
  if(mx>1e-9){vt(R.tc.toFixed(6),59,sc(R.tc));const wb=sc(R.bd),inb=wb>bw*.55;tx(sv,inb?wb-8:wb+8,125,R.bd.toFixed(6),{class:'labm','text-anchor':inb?'end':'start'})}
  tx(sv,0,170,mx>1e-9?`TC／上界 = ${R.ratio.toFixed(4)}`:'—（兩者皆接近 0：獨立時平行更新沒有誤差）',{class:'lab'});
  const rows=[['cₛ(h)',R.c.toFixed(4),'reverse DPI 係數 (e²ʰ − 1)/(1 − e⁻ˢ)'],['DTC(q_s)',R.ds.toFixed(6),'終點的 DTC'],['DTC(q_t)',R.dt.toFixed(6),'起點的 DTC'],['DTC 落差',R.dd.toFixed(6),'DTC(q_s) − DTC(q_t)，非負'],
    ['TC(Xₛ | Xₜ)',R.tc.toFixed(6),'這一步平行更新的精確 KL 誤差'],['上界',R.bd.toFixed(6),'Lemma 3.2：cₛ(h) × DTC 落差'],['TC／上界',isNaN(R.ratio)?'—':R.ratio.toFixed(4),isNaN(R.ratio)?'兩者皆接近 0':'越小代表上界越寬鬆']];
  $('#c2-tab').innerHTML='<thead><tr><th>量</th><th class="num">數值</th><th>意義</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${r[0]}</td><td class="num">${r[1]}</td><td>${r[2]}</td></tr>`).join('')+'</tbody>';
  announce('c2',`${PNAME[c2.p]}，s = ${fmtV(s)}，h = ${fmtV(h)}：TC = ${R.tc.toFixed(6)}，上界 = ${R.bd.toFixed(6)}`)}
function initC2(){chips($('#c2-chips'),c2.p,k=>{c2.p=k;renderC2()});logSlider($('#c2-s'),S_LIST,.5);logSlider($('#c2-h'),H_LIST,.2);
  ['#c2-s','#c2-h'].forEach(id=>$(id).addEventListener('input',renderC2));renderC2()}

/* ---------- C3 ---------- */
const c3={p:'codebook',res:null};
const LOGT3=[[.001,'0.001'],[.01,'0.01'],[.1,'0.1'],[1,'1'],[6,'6']];
function startC3(){const a=A_LIST[+$('#c3-a').value];$('#c3-ao').textContent=fmtV(a);$('#c3-a').setAttribute('aria-valuetext','a_U = '+fmtV(a));
  $('#c3-tab tbody').innerHTML='<tr><td colspan="3" class="busy">計算中…</td></tr>';
  runJob('c3',scheduleGen(PRE[c3.p],a),r=>{c3.res=r;drawC3();announce('c3',`N = ${r.N} 步，實際 KL = ${r.kl.toFixed(4)}`)})}
function drawC3(){const r=c3.res;if(!r)return;
  const tb=$('#c3-top'),W=widthOf(tb,280,900),D0=dtc(PRE[c3.p]),ymax=c3.p==='indep'?.1:D0*1.05;
  const F=frame($('svg',tb),{W,H:210,m:{l:46,r:18,t:24,b:52},x:[.001,6],xlog:true,y:[0,ymax],xt:LOGT3,yt:ticks(0,ymax,3),yf:v=>String(v),yl:'DTC(q_t)（nats）'});
  el('path',{d:lp(dtcCurve(c3.p,.001,6,90).map(([t,v])=>[F.X(t),F.Y(v)])),class:'unif'},F.g);
  const stride=Math.ceil((r.N+1)/200);for(let j=0;j<=r.N;j+=stride){const x=F.X(r.ts[j]);el('line',{x1:x,x2:x,y1:F.y0+22,y2:F.y0+30,class:'axl',style:'stroke:var(--accent)'},F.g)}
  tx(F.g,F.x0,F.y0+46,`排程格點 t_j（共 ${r.N+1} 個${stride>1?`，每 ${stride} 個畫 1 個`:''}）`,{class:'al'});
  const bt=$('#c3-bot'),G=frame($('svg',bt),{W,H:230,m:{l:46,r:18,t:24,b:40},x:[.001,6],xlog:true,y:[1e-8,1],ylog:true,xt:LOGT3,yt:[[1e-8,'10⁻⁸'],[1e-6,'10⁻⁶'],[1e-4,'10⁻⁴'],[1e-2,'10⁻²'],[1,'1']],xl:'noise level t（對數刻度）',yl:'每步的量（nats，對數）'});
  const clip=v=>Math.max(1e-8,Math.min(1,v));let ob='';
  r.steps.forEach(st=>{const x0=G.X(st.s),x1=G.X(st.t),w=Math.max(.6,x1-x0);
    if(st.tc>1e-8)el('rect',{x:x0,y:G.Y(clip(st.tc)),width:w,height:G.y0-G.Y(clip(st.tc)),class:'tcf',opacity:.85},G.g);
    if(st.b>1e-8)ob+=`M${x0.toFixed(1)} ${G.y0}V${G.Y(clip(st.b)).toFixed(1)}H${(x0+w).toFixed(1)}V${G.y0}`});
  if(ob)el('path',{d:ob,class:'bnd'},G.g);
  tx(G.g,G.x1-4,G.y1+12,'實心＝TC（實際）；外框＝c·DTC 落差（上界）',{class:'al','text-anchor':'end'});
  const f6=v=>v.toFixed(6);
  const rows=[['N（步數）',r.N,'排程格點數減 1'],['max c_j',`${r.maxC.toFixed(4)}（≤ 3a_U = ${(3*r.a).toFixed(3)}）`,'每步 reverse DPI 係數的最大值'],
    ['ΣTC',f6(r.sumTC),'各步實際的離散化誤差和'],['Σ c·ΔDTC',f6(r.sumB),'Lemma 3.8 的上界（各步 TC 換成 c·DTC 落差）'],['3a_U·DTC(q)',f6(r.threeA),'telescoping 後的上界'],
    ['init',`${fe(r.init)}（界 ${fe(r.initB)}）`,'初始化 KL(q_T‖Unif)，界 e⁻ᵀ·d log 2'],['end',`${f6(r.end)}（界 ${r.endB.toFixed(4)}）`,'末端 H(q_δU) − H(q)，界 d·δ_U·log(2e/δ_U)'],
    ['路徑 KL',f6(r.path),'init + ΣTC + end'],['實際 KL(q‖q̂)',f6(r.kl),'取樣器輸出分佈與 q 的 KL']];
  $('#c3-tab tbody').innerHTML=rows.map(x=>`<tr${x[0].startsWith('實際')?' class="hl"':''}><td>${x[0]}</td><td class="num">${x[1]}</td><td>${x[2]}</td></tr>`).join('');
  $('#c3-line').textContent=`KL(q‖q̂) = ${r.kl.toFixed(4)} ≤ 路徑 KL = init + ΣTC + end = ${r.path.toFixed(4)} ≤ init + Σc·ΔDTC + end = ${(r.init+r.sumB+r.end).toFixed(4)}`}
function initC3(){chips($('#c3-chips'),c3.p,k=>{c3.p=k;startC3()});logSlider($('#c3-a'),A_LIST,.1);
  $('#c3-a').addEventListener('input',()=>{const a=A_LIST[+$('#c3-a').value];$('#c3-ao').textContent=fmtV(a)});$('#c3-a').addEventListener('change',startC3);startC3()}

/* ---------- C4 ---------- */
const C4_D=[64,128,256,512,1024,2048,4096];
const c4={k:.2,di:2,curve:null};
const widthCache=new Map(),recCache=new Map();
const kKey=k=>k.toFixed(2);
function getRec(d,k){const key=d+'|'+kKey(k);if(!recCache.has(key))recCache.set(key,makeUnifRec(d,k));return recCache.get(key)}
const IG_CURVE=[];for(let i=0;i<=100;i++){const t=.01+i*(1.99/100);IG_CURVE.push([t,Iunif(t),Igauss(t)])}
function c4grid(k){const lo=Math.max(.005,k-.1),hi=Math.min(LN2-1e-6,k+.1),g=[];for(let i=0;i<81;i++)g.push(lo+(hi-lo)*i/80);return g}
function* curveGen(u,grid){const pts=[];for(const I of grid){pts.push([I,u.recI(I)]);yield}return pts}
function* widthsGen(k,first){const order=[first,...C4_D.filter(d=>d!==first)];for(const d of order){const key=kKey(k)+'|'+d;if(!widthCache.has(key)){widthCache.set(key,yield* getRec(d,k).widthGen());yield d}}}
function c4change(kChanged){c4.k=+$('#c4-k').value;c4.di=+$('#c4-d').value;const k=c4.k,d=C4_D[c4.di];
  if(kChanged){recCache.clear()}
  drawC4();
  runJob('c4c',curveGen(getRec(d,k),c4grid(k)),pts=>{c4.curve={key:kKey(k)+'|'+d,pts};drawC4Main()});
  runJob('c4w',widthsGen(k,d),()=>drawC4(),()=>drawC4())}
function drawC4(){drawC4Main();drawC4Width();drawC4Info()}
function drawC4Main(){const k=c4.k,d=C4_D[c4.di],grid=c4grid(k),lo=grid[0],hi=grid[80];
  const box=$('#c4-main'),W=widthOf(box,280,900);
  const F=frame($('svg',box),{W,H:300,m:{l:46,r:18,t:30,b:40},x:[lo,hi],y:[0,1],xt:ticks(lo,hi,W<500?4:8),xf:v=>v.toFixed(2),yt:[0,.2,.4,.6,.8,1],yf:v=>v.toFixed(1),xl:'每座標資訊量 I（nats）',yl:'recovery probability'});
  [.2,.8].forEach(v=>el('line',{x1:F.x0,x2:F.x1,y1:F.Y(v),y2:F.Y(v),class:'ref',style:'stroke-dasharray:2 4'},F.g));
  el('line',{x1:F.X(k),x2:F.X(k),y1:F.y0,y2:F.y1,class:'ref'},F.g);tx(F.g,F.X(k)+5,F.y1-8,'I = κ',{class:'labm'});
  const mp=[];for(let i=0;i<=240;i++){const I=lo+(hi-lo)*i/240;mp.push([F.X(I),F.Y(maskedRecI(d,k,I))])}el('path',{d:lp(mp),class:'mask'},F.g);
  if(d<=256){for(let m=Math.ceil(lo*d/LN2);m<=Math.floor(hi*d/LN2);m++){const I=m*LN2/d;el('rect',{x:F.X(I)-2.5,y:F.Y(maskedRecI(d,k,I))-2.5,width:5,height:5,class:'mask-f'},F.g)}}
  const key=kKey(k)+'|'+d,cv=c4.curve&&c4.curve.key===key?c4.curve.pts:null;
  if(cv)el('path',{d:lp(cv.map(([I,v])=>[F.X(I),F.Y(v)])),class:'unif'},F.g);else tx(F.g,F.x1-6,F.y0-10,'uniform 曲線計算中…',{class:'al','text-anchor':'end'});
  const m20=k-L20/d,m80=k-L80/d,mw=MASK_LNRATIO/d,yM=F.Y(.93);
  darrow(F.g,F.X(m20),yM,F.X(m80),yM,'mask-a');
  const wu=widthCache.get(key);
  if(W>=560)tx(F.g,F.X(m20)-8,yM+4,`masked 寬 ${mw.toPrecision(4)}`,{class:'labm','text-anchor':'end',style:'fill:var(--c-mask)'});
  if(wu&&!isNaN(wu.w)){const yU=F.Y(.07);darrow(F.g,F.X(wu.i20),yU,F.X(wu.i80),yU,'unif-a');const lab=`uniform 寬 ${wu.w.toPrecision(4)}`,xl=F.X(wu.i80)+8;if(W>=560)tx(F.g,xl+lab.length*7.2>F.x1?F.X(wu.i20)-8:xl,yU+4,lab,{class:'labm','text-anchor':xl+lab.length*7.2>F.x1?'end':'start',style:'fill:var(--c-unif)'})}
  const us=wu?(isNaN(wu.w)?'窗超出範圍':wu.w.toFixed(6)):'計算中…';
  $('#c4-do').textContent=d;$('#c4-ko').textContent=k.toFixed(2);$('#c4-d').setAttribute('aria-valuetext','d = '+d);$('#c4-k').setAttribute('aria-valuetext','κ = '+k.toFixed(2));
  $('#c4-ro').innerHTML=`d = <b>${d}</b>，κ = <b>${k.toFixed(2)}</b>　masked 窗寬 = 2.3698/d = <b>${mw.toFixed(6)}</b>（以 m 計 ${(MASK_LNRATIO/LN2).toFixed(3)} 個座標）<br>uniform 窗寬 = <b>${us}</b>${wu&&!isNaN(wu.w)?`　uniform／masked = <b class="acc">${(wu.w/mw).toFixed(2)}</b>`:''}`;
  $('svg',box).setAttribute('aria-label',`recovery probability 對 I，d = ${d}，κ = ${k.toFixed(2)}；masked 窗寬 ${mw.toFixed(5)}，uniform 窗寬 ${us}`);
  if(wu)announce('c4',`d = ${d}：masked 窗寬 ${mw.toFixed(5)}，uniform 窗寬 ${us}`)}
function slope(pts){const n=pts.length;if(n<3)return NaN;let sx=0,sy=0,sxx=0,sxy=0;pts.forEach(([x,y])=>{x=Math.log(x);y=Math.log(y);sx+=x;sy+=y;sxx+=x*x;sxy+=x*y});return(n*sxy-sx*sy)/(n*sxx-sx*sx)}
function drawC4Width(){const k=c4.k,box=$('#c4-w'),W=widthOf(box,260,460);
  const F=frame($('svg',box),{W,H:250,m:{l:50,r:14,t:26,b:40},x:[52,5000],xlog:true,y:[1e-4,.3],ylog:true,xt:[[64,'64'],[256,'256'],[1024,'1024'],[4096,'4096']],yt:[[1e-4,'10⁻⁴'],[1e-3,'10⁻³'],[1e-2,'10⁻²'],[.1,'0.1']],xl:'d（對數）',yl:'窗寬（nats，對數）'});
  const up=C4_D.map(d=>[d,widthCache.get(kKey(k)+'|'+d)]).filter(([,r])=>r&&!isNaN(r.w)).map(([d,r])=>[d,r.w]);
  const ref=(d0,w0,p,lab)=>{const y=d=>w0*(d0/d)**p,a=52,b=5000;el('line',{x1:F.X(a),y1:F.Y(Math.min(.3,y(a))),x2:F.X(b),y2:F.Y(Math.max(1e-4,y(b))),class:'ref'},F.g);tx(F.g,F.X(3000),F.Y(y(3000))-8,lab,{class:'labm','text-anchor':'middle'})};
  ref(64,MASK_LNRATIO/64,1,'∝ 1/d');if(up.length)ref(up[0][0],up[0][1],.5,'∝ 1/√d');
  C4_D.forEach(d=>{const w=MASK_LNRATIO/d;el('rect',{x:F.X(d)-4,y:F.Y(w)-4,width:8,height:8,class:'mask-f'},F.g)});
  up.forEach(([d,w])=>el('circle',{cx:F.X(d),cy:F.Y(w),r:4.5,class:'unif-f'},F.g));
  const sl=slope(up);
  $('#c4-tab tbody').innerHTML=C4_D.map(d=>{const r=widthCache.get(kKey(k)+'|'+d),u=r?(isNaN(r.w)?'窗超出範圍':r.w.toFixed(6)):'計算中…';return`<tr><td class="num">${d}</td><td class="num">${u}</td><td class="num">${(MASK_LNRATIO/d).toFixed(6)}</td><td class="num">${r&&!isNaN(r.w)?(r.w*Math.sqrt(d)).toFixed(3):'—'}</td></tr>`}).join('');
  c4.slope=sl;drawC4Info()}
function drawC4Info(){const k=c4.k,d=C4_D[c4.di],box=$('#c4-i'),W=widthOf(box,260,460);
  const F=frame($('svg',box),{W,H:250,m:{l:46,r:14,t:26,b:40},x:[0,2],y:[0,.7],xt:[0,.5,1,1.5,2],yt:[0,.2,.4,.6],yf:v=>v.toFixed(1),xl:'noise level t',yl:'I（nats）'});
  el('path',{d:lp(IG_CURVE.map(p=>[F.X(p[0]),F.Y(Math.min(.7,p[1]))])),class:'unif'},F.g);el('path',{d:lp(IG_CURVE.map(p=>[F.X(p[0]),F.Y(Math.min(.7,p[2]))])),class:'gaus'},F.g);
  el('line',{x1:F.x0,x2:F.x1,y1:F.Y(k),y2:F.Y(k),class:'ref'},F.g);tx(F.g,F.x1-4,F.Y(k)-6,'I = κ',{class:'labm','text-anchor':'end'});
  const tu=tStarUnif(k),tg=tStarGauss(k);
  [[tu,'unif-f','t⋆ unif'],[tg,'ink-f','t⋆ gauss']].forEach(([t,c,l],i)=>{if(t>2)return;el('line',{x1:F.X(t),x2:F.X(t),y1:F.Y(k),y2:F.y0,class:'cur'},F.g);el('circle',{cx:F.X(t),cy:F.Y(k),r:4.5,class:c},F.g);tx(F.g,F.X(t)+(i?6:-6),F.Y(k)+(i?-22:18)+(i?0:0),l,{class:'labm','text-anchor':i?'start':'end'})});
  const bs=(1-Math.exp(-tu))/2;
  $('#c4-ro2').innerHTML=`β⋆ = <b>${bs.toFixed(4)}</b>　t⋆_unif = <b>${tu.toFixed(4)}</b>　t⋆_gauss = <b>${tg.toFixed(4)}</b>　masked m⋆ = κd/log 2 = <b>${(k*d/LN2).toFixed(1)}</b>（m⋆/d = ${(k/LN2).toFixed(4)}）<br>參考：t = 0.5 時 I_unif = <b>${Iunif(.5).toFixed(4)}</b>、I_gauss = <b>${Igauss(.5).toFixed(4)}</b>　uniform 窗寬的 log–log 擬合斜率 = <b class="acc">${isNaN(c4.slope)?'（有效點不足 3 個）':c4.slope.toFixed(3)}</b>（masked 為 −1）`}
function initC4(){const k=$('#c4-k'),d=$('#c4-d');
  k.addEventListener('input',()=>{$('#c4-ko').textContent=(+k.value).toFixed(2)});d.addEventListener('input',()=>{$('#c4-do').textContent=C4_D[+d.value]});
  k.addEventListener('change',()=>c4change(true));d.addEventListener('change',()=>c4change(false));c4change(true)}

/* ---------- C5 ---------- */
const c5={p:16,eta:1e-3,seed:5,B:null,timer:0};
function c5reset(){clearInterval(c5.timer);c5.timer=0;$('#c5-scan').textContent='等距掃描';
  const d=2**c5.p,C=counts(d,c5.eta),r=mulberry32(c5.seed);c5.C=C;
  c5.B={M:{n:C.nM,win:Math.floor(r()*C.nM),q:new Uint8Array(C.nM),used:0,hit:false,last:''},U:{n:C.nU,win:Math.floor(r()*C.nU),q:new Uint8Array(C.nU),used:0,hit:false,last:''}};
  ['M','U'].forEach(k=>{const inp=$('#c5-q'+k);inp.max=c5.B[k].n;inp.value=1;c5qlabel(k)});drawC5()}
function c5qlabel(k){const inp=$('#c5-q'+k),n=c5.B[k].n;$('#c5-q'+k+'o').textContent=inp.value;inp.setAttribute('aria-valuetext',`第 ${inp.value}/${n} 段`)}
function c5query(k,i,quiet){const b=c5.B[k];if(b.hit&&quiet)return;if(!b.q[i]){b.q[i]=1;b.used++}const hit=i===b.win;if(hit)b.hit=true;
  b.last=hit?`第 ${i+1} 段：回答 = 真實 marginal（命中窗）`:`第 ${i+1} 段：回答 = 均勻（與 O₀ 相同）`;
  if(!quiet){drawC5();announce('c5',(k==='M'?'masked ':'uniform ')+b.last)}}
function c5band(k){const b=c5.B[k],box=$('#c5-b'+k),sv=$('svg',box),W=widthOf(box,260,900),H=40;sv.textContent='';sv.setAttribute('viewBox',`0 0 ${W} ${H}`);sv.setAttribute('width',W);sv.setAttribute('height',H);
  const sw=(W-2)/b.n,x=i=>1+i*sw;let run=-1;
  for(let i=0;i<=b.n;i++){const on=i<b.n&&b.q[i];if(on&&run<0)run=i;if(!on&&run>=0){el('rect',{x:x(run),y:1,width:Math.max(1,(i-run)*sw),height:H-2,class:'seg-q'},sv);run=-1}}
  if(b.n<=400)for(let i=1;i<b.n;i++)el('line',{x1:x(i),x2:x(i),y1:1,y2:H-1,class:'sep'},sv);
  if(b.hit){const w=Math.max(4,sw);el('rect',{x:Math.min(W-1-w,x(b.win)+sw/2-w/2),y:1,width:w,height:H-2,class:'seg-hit'},sv)}
  el('rect',{x:1,y:1,width:W-2,height:H-2,rx:4,class:'seg-bd'},sv);
  $('.st',box).textContent=b.last||'尚未查詢';$('.st',box).classList.toggle('hit',b.hit);
  sv.setAttribute('aria-label',`${k==='M'?'masked':'uniform'} 帶：${b.n} 段，已查詢 ${b.used} 段，${b.hit?'已命中窗':'尚未命中'}`)}
function drawC5(){const d=2**c5.p,C=c5.C,B=c5.B;
  $('#c5-po').textContent=d.toLocaleString('en-US');$('#c5-p').setAttribute('aria-valuetext',`d = 2^${c5.p} = ${d}`);
  $('.band-h span',$('#c5-bM')).textContent=`masked：已揭露數 m 的候選窗位置，n_M = ${C.nM} 段`;
  $('.band-h span',$('#c5-bU')).textContent=`uniform：時間 t 的候選窗位置，n_U = ${C.nU} 段`;
  c5band('M');c5band('U');
  const Q=B.M.used,e=eq100(C,c5.eta,Q),f=v=>v===0?'0':v<1e-3?fe(v,3):v.toPrecision(4);
  $('#c5-ro').innerHTML=`masked：已用查詢 Q = <b>${Q}</b>，n_M = <b>${C.nM}</b>，|𝒦| = <b>${C.K}</b>，ℓ = <b>${C.ell}</b><br>Eq. 100 三項（Q = ${Q}）：(2ℓ+1)Q/|𝒦| = <b>${f(e[0])}</b>，2^(1−ℓ)/η = <b>${f(e[1])}</b>，Q·2^(−ℓ) = <b>${f(e[2])}</b><br>第一項 ≤ ρ²/4 所允許的 Q_max = ρ²|𝒦|/(4(2ℓ+1)) = <b class="acc">${C.qmax.toFixed(3)}</b>${C.qmax<1?'<span class="flag">保證無意義（自選常數下 Q_max &lt; 1）</span>':''}<br>uniform：已用查詢 <b>${B.U.used}</b>，n_U = <b>${C.nU}</b>，Λ = <b>${C.Lam.toFixed(3)}</b>　段數比 n_M／n_U = <b class="acc">${(C.nM/C.nU).toFixed(1)}</b>`}
function c5scan(){if(c5.timer){clearInterval(c5.timer);c5.timer=0;$('#c5-scan').textContent='等距掃描';return}
  const ptr={M:0,U:0};let ticks=0;
  const tick=()=>{['M','U'].forEach(k=>{const b=c5.B[k];if(b.hit)return;const per=Math.ceil(b.n/32);for(let j=0;j<per&&ptr[k]<b.n&&!b.hit;j++){while(ptr[k]<b.n&&b.q[ptr[k]])ptr[k]++;if(ptr[k]<b.n)c5query(k,ptr[k]++,true)}});ticks++;drawC5();
    const done=(c5.B.M.hit&&c5.B.U.hit)||ticks>=32;if(done){clearInterval(c5.timer);c5.timer=0;$('#c5-scan').textContent='等距掃描';announce('c5',`掃描結束：masked 用 ${c5.B.M.used} 次、uniform 用 ${c5.B.U.used} 次查詢`)}return done};
  if(RM){while(!tick());return}
  $('#c5-scan').textContent='停止掃描';c5.timer=setInterval(tick,250)}
function initC5(){['M','U'].forEach(k=>{$('#c5-b'+k).innerHTML=`<div class="band-h"><span></span><span class="st">尚未查詢</span></div><svg role="img"></svg><div class="q"><div class="rng"><label class="lbl" for="c5-q${k}">查詢段號<output id="c5-q${k}o">1</output></label><input id="c5-q${k}" type="range" min="1" max="2" step="1" value="1"></div><button class="small-btn" type="button" id="c5-go${k}">查詢這一段</button></div>`;
    $('#c5-q'+k).addEventListener('input',()=>c5qlabel(k));$('#c5-go'+k).addEventListener('click',()=>c5query(k,+$('#c5-q'+k).value-1))});
  $('#c5-p').addEventListener('input',()=>{c5.p=+$('#c5-p').value;c5reset()});
  $$('#c5-eta button').forEach(b=>b.addEventListener('click',()=>{$$('#c5-eta button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));c5.eta=+b.dataset.v;c5reset()}));
  $('#c5-scan').addEventListener('click',c5scan);$('#c5-re').addEventListener('click',()=>{c5.seed++;c5reset()});c5reset()}

/* ---------- S3 static figure (Fig. 5 style) ---------- */
function drawS3(){const s=$('#s3-fig');s.textContent='';const cw=30,gap=4,x0=56,dN=8,i0=2;
  const cx=j=>x0+j*(cw+gap);
  const cond='fill:color-mix(in oklab,var(--ink-3) 45%,var(--surface));stroke:none';
  const cell=(j,y,kind)=>el('rect',{x:cx(j),y,width:cw,height:cw,rx:4,style:kind==='c'?cond:kind==='a'?'fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5':'fill:var(--surface);stroke:var(--ink-3);stroke-width:1'},s);
  const panel=(oy,title,top,bot,arc)=>{tx(s,0,oy+14,title,{class:'lab'});const yA=oy+40,yB=oy+92;
    tx(s,8,yA+20,'Xₛ',{class:'labm'});tx(s,8,yB+20,'Xₜ',{class:'labm'});
    for(let j=0;j<dN;j++){cell(j,yA,top(j));cell(j,yB,bot(j))}
    tx(s,cx(i0)+cw/2,yA-6,'i',{class:'labm','text-anchor':'middle'});
    const g0=cx(i0+1),g1=cx(dN-1)+cw;el('path',{d:`M${g0} ${yA-4}V${yA-10}H${g1}V${yA-4}`,class:'acc-l'},s);tx(s,(g0+g1)/2,yA-14,'i+1 … d',{class:'labm','text-anchor':'middle'});
    el('path',{d:arc(yA,yB),class:'acc-l',style:'stroke-dasharray:4 3'},s)};
  panel(0,'(a) 左式：I((Xₛ)ᵢ₊₁:d ; (Xₛ)ᵢ | Xₜ)',j=>j<i0?'w':'a',()=> 'c',(yA)=>`M${cx(i0)+cw/2} ${yA+cw+2}C${cx(i0)+cw/2} ${yA+cw+16} ${cx(5)+cw/2} ${yA+cw+16} ${cx(5)+cw/2} ${yA+cw+2}`);
  panel(150,'(b) 右式：I((Xₛ)ᵢ₊₁:d ; (Xₜ)ᵢ | (Xₜ)₋ᵢ)',j=>j<=i0?'w':'a',j=>j===i0?'a':'c',(yA,yB)=>`M${cx(i0+1)+4} ${yA+cw}L${cx(i0)+cw/2} ${yB}`);
}

/* ---------- timeline ---------- */
const TL_GROUP={'2006.11239':['base','eps-prediction-ddpm-score'],'2107.03006':['corr','dllm-three-corruption-paradigms'],'2211.15089':['corr','dllm-three-corruption-paradigms'],'2310.16834':['corr','dllm-three-corruption-paradigms'],
 '2406.07524':['corr','dllm-three-corruption-paradigms'],'2502.09992':['corr','dllm-three-corruption-paradigms'],'2205.14987':['samp','dllm-discrete-sampling-theory'],'2402.08095':['samp','dllm-discrete-sampling-theory'],
 '2511.04647':['mdtc','dllm-masked-schedule-dtc-theory'],'2510.25544':['mdtc','dllm-masked-schedule-dtc-theory'],'2403.01633':['cw','dllm-critical-windows-speciation'],'2604.10857':['lb','dllm-score-oracle-lower-bounds'],'2609.20539':['me','dllm-parallelism-separation-result']};
const TL_NAME={base:'基礎',corr:'三種腐蝕',samp:'取樣理論',mdtc:'masked DTC',cw:'critical window',lb:'query 下界',me:'本文',other:'其他'};
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

/* ---------- toc, theme, appbar ---------- */
function initToc(){const secs=$$('[data-toc]'),ol=$('#toc'),bar=$('#tocbar');
  secs.forEach(s=>{ol.insertAdjacentHTML('beforeend',`<li><a href="#${s.id}"><span class="n">${s.dataset.toc}</span><span>${s.dataset.tt}</span></a></li>`);bar.insertAdjacentHTML('beforeend',`<a href="#${s.id}">${s.dataset.toc} ${s.dataset.tt}</a>`)});
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;$$('.toc a,.tocbar a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+id));const ab=$(`.tocbar a[href="#${id}"]`);if(ab&&bar.offsetParent)bar.scrollTo({left:ab.offsetLeft-16,behavior:RM?'auto':'smooth'})}})},{rootMargin:'-40% 0px -55% 0px'});
  secs.forEach(s=>io.observe(s));
  addEventListener('scroll',()=>$('#appbar').classList.toggle('raised',scrollY>8),{passive:true})}
function initTheme(){let saved=null;try{saved=localStorage.getItem('pm-theme')}catch(e){}if(saved)document.documentElement.dataset.theme=saved;
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){}})}

/* ---------- self test ---------- */
function lum(hex){const h=hex.trim().replace('#','');const c=[0,2,4].map(i=>parseInt(h.slice(i,i+2),16)/255).map(v=>v<=.03928?v/12.92:((v+.055)/1.055)**2.4);return .2126*c[0]+.7152*c[1]+.0722*c[2]}
const contrast=(a,b)=>{const x=lum(a),y=lum(b);return(Math.max(x,y)+.05)/(Math.min(x,y)+.05)};
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t,m)=>ok(Math.abs(a-b)<=t+1e-12,`${m}: ${a} vs ${b}`);
  near(dtc(PRE.copy),LN2,1e-9,'DTC copy');near(dtc(PRE.parity),5*LN2,1e-9,'DTC parity');near(dtc(PRE.codebook),1.7329,1e-4,'DTC codebook');
  near(tcOf(PRE.copy),3.4657,1e-4,'TC copy');near(tcOf(PRE.codebook),2.0479,1e-4,'TC codebook');near(dtc(PRE.indep),0,1e-9,'DTC indep');
  [['codebook',.5,.2,.024176,.147974],['codebook',.05,.5,.798092,44.163802],['codebook',1,.5,.007402,.046549],['copy',.5,.2,.033732,.178660],['parity',.1,.05,.018617,.409784],['parity',.5,.2,.000796,.007046]]
    .forEach(([p,s,h,tc,b])=>{const R=c2calc(p,s,h);near(R.tc,tc,1e-6,`C2 ${p} ${s},${h} TC`);near(R.bd,b,1e-6,`C2 ${p} ${s},${h} bound`)});
  {const R=c2calc('indep',.5,.2);ok(R.tc===0&&R.bd===0&&isNaN(R.ratio),'C2 indep zero')}
  [['codebook',.1,136,.038090,.348041,.061368,.005906],['codebook',1,19,.277032,3.707104,.300310,.012022],['parity',.1,136,.025194,.674868,.045565,.007476],['indep',.1,136,0,0,.002499,.000007]]
    .forEach(([p,a,N,st,sb,path,kl])=>{const r=runSync(scheduleGen(PRE[p],a));ok(r.N===N,`C3 ${p} a=${a} N=${r.N}`);near(r.sumTC,st,1e-6,`C3 ${p} a=${a} ΣTC`);near(r.sumB,sb,1e-6,`C3 ${p} a=${a} Σbound`);near(r.path,path,1e-6,`C3 ${p} a=${a} path`);near(r.kl,kl,1e-6,`C3 ${p} a=${a} KL`)});
  near(c1expect(.5).touch,.3935,1e-4,'C1 touch');near(c1expect(.5).change,.1967,1e-4,'C1 change');
  near(tStarUnif(.2),.4937,1e-4,'t*unif .2');near(tStarGauss(.2),.5521,1e-4,'t*gauss .2');near(tStarUnif(.5),.1013,1e-4,'t*unif .5');near(tStarGauss(.1),.8534,1e-4,'t*gauss .1');
  [[.25,.34538,.43845],[.5,.19730,.22716],[1,.06928,.07268],[2,.00919,.00924]].forEach(([t,u,g])=>{near(Iunif(t),u,1e-5,'Iunif '+t);near(Igauss(t),g,1e-5,'Igauss '+t)});
  near(MASK_LNRATIO,2.3698,1e-4,'masked ln ratio');near(Math.exp(L80),.4642,1e-4,'lam80');near(Math.exp(L20),4.9651,1e-4,'lam20');
  {const u=makeUnifRec(256,.2);near(u.recI(.2),.5357,5e-4,'rec d=256 I=.2');near(u.recI(.15),.0591,5e-4,'rec d=256 I=.15');near(u.recI(.25),.9374,5e-4,'rec d=256 I=.25');
   const nw=runSync(u.widthGen(.199,.201));ok(isNaN(nw.w),'width NaN when bracket fails')}
  {const u=makeUnifRec(64,.2);near(u.recI(.22),.6797,5e-4,'rec d=64 I=.22');const w=runSync(u.widthGen()).w;ok(Math.abs(w/.105902-1)<.01,'width d=64 '+w)}
  [[4096,1e-3,2364,27,43,15],[1048576,1e-3,605110,35,8523,210],[65536,1e-3,37820,31,601,56],[1024,1e-3,591,25,12,8],[65536,1e-6,null,41,456,49]].forEach(([d,eta,K,ell,nM,nU])=>{const c=counts(d,eta);ok((K===null||c.K===K)&&c.ell===ell&&c.nM===nM&&c.nU===nU,`C5 d=${d} eta=${eta}: ${c.K},${c.ell},${c.nM},${c.nU}`)});
  near(counts(65536,1e-3).qmax,6.003,1e-3,'C5 Qmax 65536');{const e=eq100(counts(1024,1e-3),1e-3,12);near(e[0],1.036,1e-3,'Eq100 t1');near(e[1],5.96e-5,1e-7,'Eq100 t2');near(e[2],3.576e-7,1e-9,'Eq100 t3')}
  const root=document.documentElement,prev=root.dataset.theme;
  ['light','dark'].forEach(th=>{root.dataset.theme=th;const cs=getComputedStyle(root),bg=cs.getPropertyValue('--surface');['--c-unif','--c-mask','--ink-1','--accent'].forEach(v=>{const c=contrast(cs.getPropertyValue(v),bg);ok(c>=3,`contrast ${th} ${v} = ${c.toFixed(2)}`)})});
  if(prev)root.dataset.theme=prev;else delete root.dataset.theme;
  ok(root.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+root.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ---------- boot ---------- */
initTheme();initToc();initC1();initC2();initC3();initC4();initC5();drawS3();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderC2();drawC3();drawC4();drawC5()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath();if(/[?&]test\b/.test(location.search))setTimeout(selfTest,0)});
})();
