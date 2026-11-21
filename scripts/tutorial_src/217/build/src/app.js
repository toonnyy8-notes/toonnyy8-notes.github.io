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
const f3=v=>(Math.abs(v)<0.0005?0:v).toFixed(3);
function widthOf(box,min,max){return Math.max(min,Math.min(max,Math.floor(box.clientWidth||max)))}
function frame(svg,o){
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${o.W} ${o.H}`);svg.setAttribute('width',o.W);svg.setAttribute('height',o.H);
  const m=o.m,x0=m.l,x1=o.W-m.r,y0=o.H-m.b,y1=m.t;
  const X=(o.xlog?lgs:lin)(o.x[0],o.x[1],x0,x1),Y=lin(o.y[0],o.y[1],y0,y1);
  const g=el('g',null,svg);
  (o.yt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:(o.yf?o.yf(v):String(v)),yy=Y(val);el('line',{x1:x0,x2:x1,y1:yy,y2:yy,class:'grid'},g);tx(g,x0-8,yy+4,lab,{class:'tk','text-anchor':'end'})});
  (o.xt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:(o.xf?o.xf(v):String(v)),xx=X(val);el('line',{x1:xx,x2:xx,y1:y0,y2:y0+4,class:'axl'},g);tx(g,xx,y0+18,lab,{class:'tk','text-anchor':'middle'})});
  el('line',{x1:x0,x2:x1,y1:y0,y2:y0,class:'axl'},g);
  if(o.xl)tx(g,(x0+x1)/2,o.H-4,o.xl,{class:'al','text-anchor':'middle'});
  if(o.yl)tx(g,4,12,o.yl,{class:'al'});
  return {X,Y,g,x0,x1,y0,y1};
}
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function tok(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}

/* ---------- orders ---------- */
const G=32,N=1024,PX=4,S=128,FD=PX*PX*3;
function hilbertXY(n,d){let x=0,y=0,t=d;for(let s=1;s<n;s*=2){const rx=1&(t>>1),ry=1&(t^rx);if(ry===0){if(rx===1){x=s-1-x;y=s-1-y}const q=x;x=y;y=q}x+=s*rx;y+=s*ry;t>>=2}return[x,y]}
function mortonXY(n,d){let x=0,y=0;for(let b=0;(1<<(2*b))<n*n;b++){x|=((d>>(2*b))&1)<<b;y|=((d>>(2*b+1))&1)<<b}return[x,y]}
function rasterXY(n,d){return[d%n,Math.floor(d/n)]}
const ORD={hilbert:hilbertXY,morton:mortonXY,raster:rasterXY},orderCache=new Map();
function order(name,q){const k=name+'|'+q;if(!orderCache.has(k)){const a=[];for(let d=0;d<q*q;d++)a.push(ORD[name](q,d));orderCache.set(k,a)}return orderCache.get(k)}
function components(cells){const set=new Set(cells.map(c=>c[0]*4096+c[1])),seen=new Set();let k=0;const lab=new Map();
  for(const c of cells){const key=c[0]*4096+c[1];if(seen.has(key))continue;k++;const st=[c];seen.add(key);lab.set(key,k);
    while(st.length){const [x,y]=st.pop();for(const [a,b] of [[x+1,y],[x-1,y],[x,y+1],[x,y-1]]){const kk=a*4096+b;if(set.has(kk)&&!seen.has(kk)){seen.add(kk);lab.set(kk,k);st.push([a,b])}}}}
  return{count:k,lab}}
function spreadOf(cells,w){const m=cells.length;if(!w)w=new Array(m).fill(1/m);let cx=0,cy=0;for(let i=0;i<m;i++){cx+=w[i]*cells[i][0];cy+=w[i]*cells[i][1]}let s=0;for(let i=0;i<m;i++)s+=w[i]*Math.hypot(cells[i][0]-cx,cells[i][1]-cy);return{s,cx,cy}}
const HIL=order('hilbert',G),HIL_PID=HIL.map(([x,y])=>y*G+x);

/* ---------- toy scene ---------- */
const SC=(()=>{const img=new Float64Array(S*S*3),r=mulberry32(1);const lerp=(a,b,s)=>[a[0]+(b[0]-a[0])*s,a[1]+(b[1]-a[1])*s,a[2]+(b[2]-a[2])*s];
  for(let y=0;y<S;y++)for(let x=0;x<S;x++){const n1=r(),n2=r(),u=(x+.5)/S,v=(y+.5)/S;let c;
    if(v<0.62)c=lerp([0.55,0.75,0.95],[0.85,0.92,0.98],v/0.62);else{c=lerp([0.45,0.58,0.32],[0.36,0.48,0.26],(v-0.62)/0.38);c=c.map(ci=>ci+0.06*(n1-0.5))}
    if(Math.hypot(u-0.78,v-0.18)<0.09)c=[0.98,0.86,0.45];
    if(Math.abs(u-0.28)<0.025&&v>0.45&&v<0.70)c=[0.40,0.28,0.18];
    if(Math.hypot(u-0.28,v-0.38)<0.15)c=[0.20+0.18*(n1-0.5),0.45+0.18*(n1-0.5),0.20];
    if(u>0.52&&u<0.70&&v>0.66&&v<0.82){c=[0.95,0.95,0.92];if(v>0.69&&v<0.79&&Math.floor(x/2)%3===0)c=[0.10,0.10,0.12]}
    for(let ch=0;ch<3;ch++)img[(y*S+x)*3+ch]=Math.min(1,Math.max(0,c[ch]+0.01*(n2-0.5)))}
  const eps=new Float64Array(S*S*3),r2=mulberry32(2);for(let i=0;i<S*S*3;i++)eps[i]=gauss(r2);
  const det=new Float64Array(N);
  for(let y=0;y<S;y++)for(let x=0;x<S;x++){let g=0;for(let ch=0;ch<3;ch++){const v=img[(y*S+x)*3+ch];if(x+1<S)g+=(img[(y*S+x+1)*3+ch]-v)**2;if(y+1<S)g+=(img[((y+1)*S+x)*3+ch]-v)**2}det[((y/PX|0))*G+(x/PX|0)]+=g}
  for(let p=0;p<N;p++)det[p]/=PX*PX;
  const logd=Array.from(det,e=>Math.log(e+1e-4));const mu=logd.reduce((a,b)=>a+b,0)/N;const sd=Math.sqrt(logd.reduce((a,b)=>a+(b-mu)**2,0)/N);
  const score=Float64Array.from(logd,v=>(v-mu)/sd);
  return{img,eps,det,score}})();
function detailShare(frac){const s=Array.from(SC.det).sort((a,b)=>b-a),k=Math.round(frac*N);let a=0,t=0;for(let i=0;i<N;i++){t+=s[i];if(i<k)a+=s[i]}return a/t}
const featCache=new Map();
function featuresAt(t){const key=t.toFixed(2);if(featCache.has(key))return featCache.get(key);const h=new Float64Array(N*FD),{img,eps}=SC;
  for(let pr=0;pr<G;pr++)for(let pc=0;pc<G;pc++){let k=0;const base=(pr*G+pc)*FD;for(let yy=0;yy<PX;yy++)for(let xx=0;xx<PX;xx++)for(let ch=0;ch<3;ch++){const i=((pr*PX+yy)*S+pc*PX+xx)*3+ch;h[base+k++]=t===1?img[i]:t*img[i]+(1-t)*eps[i]}}
  featCache.set(key,h);return h}
function dist(h,a,b){let s=0;const A=a*FD,B=b*FD;for(let k=0;k<FD;k++){const d=h[A+k]-h[B+k];s+=d*d}return Math.sqrt(s)}
function gapsOf(h){const g=new Float64Array(N-1);for(let j=0;j<N-1;j++)g[j]=dist(h,HIL_PID[j+1],HIL_PID[j]);return g}
function runsToLabels(cuts){const cs=new Set(cuts),lab=new Int32Array(N);let g=0;for(let j=0;j<N;j++){lab[HIL_PID[j]]=g;if(cs.has(j))g++}return lab}
function cutsAdaptive(gaps,R){const idx=Array.from({length:N-1},(_,j)=>j);idx.sort((a,b)=>gaps[b]-gaps[a]||a-b);return idx.slice(0,R-1).sort((a,b)=>a-b)}
function cutsFixed(R){const c=[];for(let r=1;r<R;r++)c.push(Math.round(r*N/R)-1);return c}
function labelsFeatSim(h,R){const mean=new Float64Array(FD);for(let p=0;p<N;p++)for(let k=0;k<FD;k++)mean[k]+=h[p*FD+k]/N;
  const hn=new Float64Array(N*FD);for(let p=0;p<N;p++){let s=0;for(let k=0;k<FD;k++){const v=h[p*FD+k]-mean[k];hn[p*FD+k]=v;s+=v*v}s=Math.sqrt(s);for(let k=0;k<FD;k++)hn[p*FD+k]=s>1e-12?hn[p*FD+k]/s:0}
  const anc=[];for(let r=0;r<R;r++)anc.push(Math.floor(r*N/R));const lab=new Int32Array(N);
  for(let p=0;p<N;p++){let best=-Infinity,bi=0;for(let r=0;r<R;r++){const a=anc[r]*FD;let s=0;for(let k=0;k<FD;k++)s+=hn[p*FD+k]*hn[a+k];if(s>best){best=s;bi=r}}lab[p]=bi}return lab}
function evOf(h,lab){const mean=new Float64Array(FD);for(let p=0;p<N;p++)for(let k=0;k<FD;k++)mean[k]+=h[p*FD+k]/N;let den=0;for(let p=0;p<N;p++)for(let k=0;k<FD;k++)den+=(h[p*FD+k]-mean[k])**2;if(den<1e-18)return null;
  const groups=new Map();for(let p=0;p<N;p++){if(!groups.has(lab[p]))groups.set(lab[p],[]);groups.get(lab[p]).push(p)}let num=0;
  for(const ps of groups.values()){const m=new Float64Array(FD);for(const p of ps)for(let k=0;k<FD;k++)m[k]+=h[p*FD+k]/ps.length;for(const p of ps)for(let k=0;k<FD;k++)num+=(h[p*FD+k]-m[k])**2}
  return 1-num/den}
function evSkip(h,R){const mean=new Float64Array(FD);for(let p=0;p<N;p++)for(let k=0;k<FD;k++)mean[k]+=h[p*FD+k]/N;const dev=new Float64Array(N);let den=0;
  for(let p=0;p<N;p++){let s=0;for(let k=0;k<FD;k++)s+=(h[p*FD+k]-mean[k])**2;dev[p]=s;den+=s}if(den<1e-18)return null;
  const idx=Array.from({length:N},(_,p)=>p).sort((a,b)=>dev[b]-dev[a]||a-b);let kept=0;for(let i=0;i<R;i++)kept+=dev[idx[i]];return 1-(den-kept)/den}
function groupStats(lab){const groups=new Map();for(let p=0;p<N;p++){if(!groups.has(lab[p]))groups.set(lab[p],[]);groups.get(lab[p]).push(p)}
  const sp=[],fin=[],dd=[];let mx=0;for(const ps of groups.values()){sp.push(spreadOf(ps.map(p=>[p%G,(p/G)|0])).s);fin.push(-Math.log2(ps.length));let s=0;for(const p of ps)s+=SC.det[p];dd.push(s/ps.length);mx=Math.max(mx,ps.length)}
  const n=fin.length,mf=fin.reduce((a,b)=>a+b,0)/n,md=dd.reduce((a,b)=>a+b,0)/n;let sf=0,sd=0,c=0;for(let i=0;i<n;i++){sf+=(fin[i]-mf)**2;sd+=(dd[i]-md)**2;c+=(fin[i]-mf)*(dd[i]-md)}
  sf=Math.sqrt(sf/n);sd=Math.sqrt(sd/n);const corr=(sf<1e-12||sd<1e-12)?0:(c/n)/(sf*sd);
  return{groups:n,spread:sp.reduce((a,b)=>a+b,0)/n,corr,maxsize:mx,map:groups}}

/* ---------- READ / RoPE ---------- */
const OMEGA=Array.from({length:16},(_,i)=>10000**(-i/16));
function phasorMag(coords,w,om){let re=0,im=0;for(let i=0;i<coords.length;i++){re+=w[i]*Math.cos(om*coords[i]);im+=w[i]*Math.sin(om*coords[i])}return Math.hypot(re,im)}
function dirichlet(om,L){const s=Math.sin(om/2);return Math.abs(s)<1e-12?1:Math.abs(Math.sin(om*L/2)/(L*s))}
function readRegion(p,L,beta){const pids=HIL_PID.slice(p,p+L),cells=pids.map(q=>[q%G,(q/G)|0]);const s=pids.map(q=>beta*SC.score[q]),m=Math.max(...s);
  const ex=s.map(v=>Math.exp(v-m)),Z=ex.reduce((a,b)=>a+b,0),al=ex.map(e=>e/Z);const sp=spreadOf(cells,al);const uni=new Array(L).fill(1/L);
  const mx=OMEGA.map(w=>phasorMag(cells.map(c=>c[0]),uni,w)),my=OMEGA.map(w=>phasorMag(cells.map(c=>c[1]),uni,w));
  return{cells,al,sizeIdx:Math.floor(Math.log2(L)),cx:sp.cx,cy:sp.cy,spread:sp.s,amax:Math.max(...al),mx,my,kx:mx.filter(v=>v>=0.9).length,ky:my.filter(v=>v>=0.9).length}}

/* ---------- cost model ---------- */
const CORES={b16:{B:17,core:[3,13],N:1024,def:256,list:[16,32,64,96,128,192,256,384,512,768,1024],meas:[[512,1.84],[256,2.15]]},
  b16long:{B:17,core:[1,15],N:1024,def:256,list:[16,32,64,96,128,192,256,384,512,768,1024],meas:[]},
  l16:{B:23,core:[3,19],N:1024,def:256,list:[16,32,64,96,128,192,256,384,512,768,1024],meas:[[512,2.11],[256,2.59],[128,2.86],[64,3.01]]},
  jit:{B:12,core:[4,9],N:256,def:130,list:[16,32,64,96,130,193,256],meas:[],anly:[[64,1.61],[130,1.33],[193,1.14]]}};
const ncore=c=>c.core[1]-c.core[0]+1;
function relCost(R,c,eta,tau){const nu=R/c.N,nc=ncore(c),img=((c.B-nc)+nc*((1-eta)*nu+eta*nu*nu))/c.B;return tau+(1-tau)*img}
function isoR(eta){const ref=CORES.b16,c=CORES.b16long,rho=relCost(256,ref,eta,0),nc=ncore(c),v=(rho*c.B-(c.B-nc))/nc;if(v<0)return null;return eta===0?v*c.N:c.N*Math.sqrt(v)}
function ceilOf(c,tau){return 1/(tau+(1-tau)*(c.B-ncore(c))/c.B)}

/* ---------- hero curve & architecture ---------- */
function drawHero(){const s=$('#hero-curve');s.textContent='';const q=8,c=26,o=16;
  for(let i=0;i<q;i++)for(let j=0;j<q;j++)el('rect',{x:o+i*c,y:o+j*c,width:c,height:c,class:'cell'},s);
  const pts=order('hilbert',q).map(([x,y])=>[o+x*c+c/2,o+y*c+c/2]);el('path',{d:lp(pts),class:'curve',pathLength:1},s);
  [15,31,47].forEach(j=>{const a=pts[j],b=pts[j+1],mx=(a[0]+b[0])/2,my=(a[1]+b[1])/2,dx=b[0]-a[0],dy=b[1]-a[1],L=Math.hypot(dx,dy),nx=-dy/L*9,ny=dx/L*9;el('line',{x1:mx-nx,y1:my-ny,x2:mx+nx,y2:my+ny,class:'cut'},s)})}
function drawArch(){const box=$('#arch-fig'),svg=$('svg',box),W=widthOf(box,300,860),H=210;svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const B=17,gap=W<500?3:6,bw=(W-20-gap*(B-1))/B,base=170,full=100,small=26;const x=i=>10+i*(bw+gap);
  for(let i=0;i<B;i++){const core=i>=3&&i<=13,h=core?small:full;el('rect',{x:x(i),y:base-h,width:bw,height:h,rx:3,class:core?'blk-c':'blk'},svg);if(W>=560||i%2===0)tx(svg,x(i)+bw/2,base+16,String(i),{class:'tk','text-anchor':'middle'})}
  const wx1=x(3)-gap/2,wx2=x(14)-gap/2;
  [wx1,wx2].forEach((wx,k)=>{el('path',{d:k?`M${wx-3} ${base-small}L${wx+3} ${base-full}L${wx+3} ${base}Z`:`M${wx-3} ${base-full}L${wx+3} ${base-small}L${wx-3} ${base}Z`,class:'wedge',opacity:.5},svg);tx(svg,wx,base+34,k?'WRITE':'READ',{class:'labm','text-anchor':'middle'})});
  tx(svg,(x(0)+x(3)-gap)/2,base-full-8,W<500?'N 個':'N 個 patch token',{class:'al','text-anchor':'middle'});
  tx(svg,(x(14)+x(17)-gap)/2,base-full-8,W<500?'N 個':'N 個 patch token',{class:'al','text-anchor':'middle'});
  tx(svg,(x(3)+x(14)-gap)/2,base-small-10,'R 個區域 token（核心 3–13）',{class:'lab','text-anchor':'middle'});
  const ay=base-full-30;el('path',{d:`M${wx1} ${base-full-2}C${wx1} ${ay-14} ${wx2} ${ay-14} ${wx2} ${base-full-2}`,class:'ref'},svg);
  tx(svg,(wx1+wx2)/2,ay-16,'核心前特徵跳接（WRITE 的 h^(s0)）',{class:'al','text-anchor':'middle'})}

/* ---------- C1 ---------- */
const c1={q:8,L:8,p:4,all:true};
function c1frac(name,q,L){const o=order(name,q);let c=0;for(let s=0;s<=q*q-L;s++)if(components(o.slice(s,s+L)).count===1)c++;return c/(q*q-L+1)}
const c1fc=new Map();function c1fracC(name,q,L){const k=name+q+'|'+L;if(!c1fc.has(k))c1fc.set(k,c1frac(name,q,L));return c1fc.get(k)}
function c1compute(name,q,L,p){const o=order(name,q),cells=o.slice(p,p+L),cc=components(cells);return{o,cells,cc,spread:spreadOf(cells).s,frac:c1fracC(name,q,L)}}
function renderC1(){const {q,L,p}=c1;$('#c1-Lo').textContent=L;$('#c1-po').textContent=p;$('#c1-L').setAttribute('aria-valuetext',`段長 ${L}`);$('#c1-p').setAttribute('aria-valuetext',`起點 ${p}，段長 ${L}`);
  $$('#c1-q button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.v===q)));
  const msg=[];
  for(const name of ['hilbert','morton','raster']){const r=c1compute(name,q,L,p),svg=$('#c1-'+name),Wv=260,cs=(Wv-8)/q,o=4;svg.textContent='';svg.setAttribute('viewBox',`0 0 ${Wv} ${Wv}`);
    const C=([x,y])=>[o+x*cs+cs/2,o+y*cs+cs/2];
    for(const c of r.cells)el('rect',{x:o+c[0]*cs,y:o+c[1]*cs,width:cs,height:cs,class:'selc'},svg);
    if(q<=16)for(let i=0;i<=q;i++){el('line',{x1:o+i*cs,x2:o+i*cs,y1:o,y2:o+q*cs,class:'cellg'},svg);el('line',{y1:o+i*cs,y2:o+i*cs,x1:o,x2:o+q*cs,class:'cellg'},svg)}
    else el('rect',{x:o,y:o,width:q*cs,height:q*cs,class:'cellg'},svg);
    if(c1.all)el('path',{d:lp(r.o.map(C)),class:'path0'},svg);
    for(let i=0;i+1<r.cells.length;i++){const a=r.cells[i],b=r.cells[i+1],adj=Math.abs(a[0]-b[0])+Math.abs(a[1]-b[1])===1,A=C(a),Bp=C(b);el('line',{x1:A[0],y1:A[1],x2:Bp[0],y2:Bp[1],class:adj?'path1':'jump'},svg)}
    if(q<=16&&r.cc.count>1)for(const c of r.cells){const P=C(c);tx(svg,P[0],P[1]+4,String(r.cc.lab.get(c[0]*4096+c[1])),{class:'cnum','text-anchor':'middle'})}
    if(r.cells.length===1){const P=C(r.cells[0]);el('circle',{cx:P[0],cy:P[1],r:3,class:'acc-f'},svg)}
    svg.setAttribute('aria-label',`${name}：${r.cc.count} 個連通分量，spread ${r.spread.toFixed(3)}`);
    $('#c1-ro-'+name).innerHTML=`連通分量 <b>${r.cc.count}</b><br>spread <b>${r.spread.toFixed(3)}</b><br>連通比例 <b>${r.frac.toFixed(3)}</b>`;msg.push(`${name} ${r.cc.count} 塊`)}
  announce('c1',`q ${q}，起點 ${p}，段長 ${L}：${msg.join('，')}`)}
function c1clamp(){const Lmax=Math.min(64,c1.q*c1.q);c1.L=Math.min(c1.L,Lmax);const r=$('#c1-L');r.max=Lmax;r.value=c1.L;c1.p=Math.min(c1.p,c1.q*c1.q-c1.L);const rp=$('#c1-p');rp.max=c1.q*c1.q-c1.L;rp.value=c1.p}
function initC1(){$$('#c1-q button').forEach(b=>b.addEventListener('click',()=>{c1.q=+b.dataset.v;c1clamp();renderC1()}));
  $('#c1-L').addEventListener('input',e=>{c1.L=+e.target.value;c1clamp();renderC1()});$('#c1-p').addEventListener('input',e=>{c1.p=+e.target.value;renderC1()});
  $('#c1-all').addEventListener('change',e=>{c1.all=e.target.checked;renderC1()});c1clamp();renderC1()}

/* ---------- C2 ---------- */
const C2R=[64,96,128,192,256,384,512];
const c2={m:'adaptive',R:128,t:1,sel:null,focus:528,res:null};
const c2cache=new Map();
function c2results(t,R){const key=t.toFixed(2)+'|'+R;if(c2cache.has(key))return c2cache.get(key);const h=featuresAt(t),gaps=gapsOf(h),ca=cutsAdaptive(gaps,R),cf=cutsFixed(R);
  const labs={adaptive:runsToLabels(ca),fixed:runsToLabels(cf),featsim:labelsFeatSim(h,R)};const out={gaps,cuts:{adaptive:ca,fixed:cf},labs,stats:{},ev:{}};
  for(const m in labs){out.stats[m]=groupStats(labs[m]);out.ev[m]=evOf(h,labs[m])}out.ev.skip=evSkip(h,R);c2cache.set(key,out);return out}
let c2off=null;
function renderC2(){const r=c2results(c2.t,c2.R);c2.res=r;$('#c2-to').textContent=c2.t.toFixed(2);$('#c2-t').setAttribute('aria-valuetext','t = '+c2.t.toFixed(2));
  $$('#c2-m button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===c2.m)));$$('#c2-R button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.v===c2.R)));
  // image
  if(!c2off){c2off=document.createElement('canvas');c2off.width=S;c2off.height=S}
  const octx=c2off.getContext('2d'),id=octx.createImageData(S,S),h=featuresAt(c2.t);
  for(let y=0;y<S;y++)for(let x=0;x<S;x++){const p=((y/PX)|0)*G+((x/PX)|0),k=((y%PX)*PX+(x%PX))*3;for(let ch=0;ch<3;ch++)id.data[(y*S+x)*4+ch]=Math.round(255*Math.min(1,Math.max(0,h[p*FD+k+ch])));id.data[(y*S+x)*4+3]=255}
  octx.putImageData(id,0,0);
  const wrap=$('#c2-wrap'),cv=$('#c2-img');cv.style.width='100%';cv.style.height='auto';const css=Math.max(200,Math.min(384,Math.floor(wrap.getBoundingClientRect().width)||384)),dpr=window.devicePixelRatio||1;cv.width=Math.round(css*dpr);cv.height=Math.round(css*dpr);
  const ctx=cv.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(c2off,0,0,cv.width,cv.height);
  const cell=cv.width/G,lab=r.labs[c2.m],dark=document.documentElement.dataset.theme==='dark'||(!document.documentElement.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches);
  const segs=[];for(let y=0;y<G;y++)for(let x=0;x<G;x++){const p=y*G+x;if(x+1<G&&lab[p]!==lab[p+1])segs.push([(x+1)*cell,y*cell,(x+1)*cell,(y+1)*cell]);if(y+1<G&&lab[p]!==lab[p+G])segs.push([x*cell,(y+1)*cell,(x+1)*cell,(y+1)*cell])}
  const stroke=(col,w)=>{ctx.strokeStyle=col;ctx.lineWidth=w;ctx.beginPath();for(const s of segs){ctx.moveTo(s[0],s[1]);ctx.lineTo(s[2],s[3])}ctx.stroke()};
  stroke('rgba(255,255,255,.7)',3*dpr);stroke(dark?'#161513':'#1D1C19',1*dpr);
  let selTxt='點選或以方向鍵、Enter 選取一個 patch，突顯它所屬的組。';
  if(c2.sel!==null){const g=lab[c2.sel],ps=r.stats[c2.m].map.get(g);ctx.save();ctx.strokeStyle=tok('--accent');ctx.lineWidth=1.5*dpr;
    for(const p of ps){const x=(p%G)*cell,y=((p/G)|0)*cell;ctx.beginPath();ctx.moveTo(x,y+cell);ctx.lineTo(x+cell,y);ctx.moveTo(x,y+cell/2);ctx.lineTo(x+cell/2,y);ctx.moveTo(x+cell/2,y+cell);ctx.lineTo(x+cell,y+cell/2);ctx.stroke()}ctx.restore();
    const sp=spreadOf(ps.map(p=>[p%G,(p/G)|0])).s;selTxt=`選取的組：${ps.length} 個 patch，spread ${sp.toFixed(3)} 格（斜線標示）。`}
  $('#c2-sel').textContent=selTxt;
  if(document.activeElement===$('#c2-ov')){const fx=(c2.focus%G)*cell,fy=((c2.focus/G)|0)*cell;ctx.strokeStyle=tok('--accent');ctx.lineWidth=2*dpr;ctx.strokeRect(fx+dpr,fy+dpr,cell-2*dpr,cell-2*dpr)}
  // table
  const name={adaptive:'自適應（RTI）',fixed:'等距切割',featsim:'Feat Sim',skip:'Skip（不分組）'};const fmt=v=>v===null?'—':v.toFixed(3),fc=v=>(v>=0?'+':'−')+Math.abs(v).toFixed(3);
  $('#c2-tab tbody').innerHTML=['adaptive','fixed','featsim','skip'].map(m=>{const st=r.stats[m];return`<tr${m===c2.m?' class="on"':''}><td>${name[m]}</td><td class="num">${fmt(r.ev[m])}</td><td class="num">${st?st.spread.toFixed(3):'—'}</td><td class="num">${st?fc(st.corr):'—'}</td><td class="num">${st?st.maxsize:'—'}</td></tr>`}).join('');
  drawBars();
  $('#c2-ov').setAttribute('aria-label',`影像分割：${name[c2.m]}，R ${c2.R}，t ${c2.t.toFixed(2)}，EV ${fmt(r.ev[c2.m])}。方向鍵移動、Enter 選取`);
  announce('c2',`${name[c2.m]}，R = ${c2.R}，EV ${fmt(r.ev[c2.m])}，spread ${r.stats[c2.m].spread.toFixed(3)}`)}
function drawBars(){const r=c2.res,cv=$('#c2-bars'),box=$('#c2-barbox'),css=Math.max(240,box.clientWidth||600),H=96,dpr=window.devicePixelRatio||1;cv.width=Math.round(css*dpr);cv.height=Math.round(H*dpr);cv.style.height=H+'px';
  const ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);const Wp=cv.width,bins=new Float64Array(Wp),cutBin=new Uint8Array(Wp);let mx=0;
  for(let j=0;j<N-1;j++){const b=Math.min(Wp-1,Math.floor(j/(N-1)*Wp));bins[b]=Math.max(bins[b],r.gaps[j]);mx=Math.max(mx,r.gaps[j])}
  const showCuts=c2.m==='adaptive'||c2.m==='fixed';if(showCuts)for(const j of r.cuts[c2.m])cutBin[Math.min(Wp-1,Math.floor(j/(N-1)*Wp))]=1;
  const top=10*dpr,hh=cv.height-top-2*dpr,ink=tok('--ink-3'),acc=tok('--accent');
  for(let b=0;b<Wp;b++){const v=mx>0?bins[b]/mx:0;if(v<=0&&!cutBin[b])continue;ctx.fillStyle=cutBin[b]&&c2.m==='adaptive'?acc:ink;const hb=Math.max(1,v*hh);ctx.fillRect(b,cv.height-hb,1,hb)}
  if(c2.m==='fixed'){ctx.fillStyle=acc;for(let b=0;b<Wp;b++)if(cutBin[b])ctx.fillRect(b,0,1,top)}
  if(c2.m==='adaptive'){ctx.fillStyle=acc;for(let b=0;b<Wp;b++)if(cutBin[b]){ctx.beginPath();ctx.moveTo(b-3*dpr,0);ctx.lineTo(b+3*dpr,0);ctx.lineTo(b,6*dpr);ctx.fill()}}
  $('#c2-barcap').textContent=c2.m==='adaptive'?`R − 1 = ${c2.R-1} 個切點落在最大的差距上（頂端三角標記、長條以強調色畫出）。`:c2.m==='fixed'?`等距切割：R − 1 = ${c2.R-1} 個切點等距放置（頂端刻線），與差距大小無關。`:'Feat Sim 不沿序列切割，這裡只顯示差距本身。'}
function initC2(){$('#c2-R').innerHTML=C2R.map(R=>`<button type="button" data-v="${R}" aria-pressed="false">${R}</button>`).join('');
  $$('#c2-m button').forEach(b=>b.addEventListener('click',()=>{c2.m=b.dataset.v;renderC2()}));$$('#c2-R button').forEach(b=>b.addEventListener('click',()=>{c2.R=+b.dataset.v;renderC2()}));
  let tt=0;$('#c2-t').addEventListener('input',e=>{$('#c2-to').textContent=(+e.target.value).toFixed(2);clearTimeout(tt);tt=setTimeout(()=>{c2.t=+e.target.value;renderC2()},100)});
  const ov=$('#c2-ov');ov.addEventListener('click',e=>{const b=ov.getBoundingClientRect(),x=Math.floor((e.clientX-b.left)/b.width*G),y=Math.floor((e.clientY-b.top)/b.height*G);if(x<0||y<0||x>=G||y>=G)return;c2.sel=c2.focus=y*G+x;renderC2()});
  ov.addEventListener('keydown',e=>{const d={ArrowLeft:-1,ArrowRight:1,ArrowUp:-G,ArrowDown:G}[e.key];if(d!==undefined){const x=c2.focus%G,y=(c2.focus/G)|0;if((d===-1&&x===0)||(d===1&&x===G-1)||(d===-G&&y===0)||(d===G&&y===G-1)){e.preventDefault();return}c2.focus+=d;e.preventDefault();renderC2()}else if(e.key==='Enter'||e.key===' '){c2.sel=c2.focus;e.preventDefault();renderC2()}});
  ov.addEventListener('focus',renderC2);ov.addEventListener('blur',renderC2);renderC2()}

/* ---------- C3 ---------- */
const c3={p:100,L:37,beta:1};
function renderC3(){const {L,beta}=c3;c3.p=Math.min(c3.p,N-L);const p=c3.p;$('#c3-p').value=p;$('#c3-po').textContent=p;$('#c3-Lo').textContent=L;$('#c3-bo').textContent=beta.toFixed(1);
  $('#c3-p').setAttribute('aria-valuetext',`起點 ${p}`);$('#c3-L').setAttribute('aria-valuetext',`段長 ${L}`);$('#c3-b').setAttribute('aria-valuetext',`β = ${beta.toFixed(1)}`);
  $$('#c3-chips .chip').forEach(c=>c.setAttribute('aria-pressed',String(+c.dataset.L===L)));
  const r=readRegion(p,L,beta);
  const g=$('#c3-grid'),Wv=280,cs=(Wv-8)/G,o=4;g.textContent='';g.setAttribute('viewBox',`0 0 ${Wv} ${Wv}`);
  const im=el('image',{x:o,y:o,width:G*cs,height:G*cs,opacity:.35,preserveAspectRatio:'none'},g);if(c2off)im.setAttribute('href',c2off.toDataURL());
  el('rect',{x:o,y:o,width:G*cs,height:G*cs,class:'cellg'},g);
  r.cells.forEach(c=>el('rect',{x:o+c[0]*cs,y:o+c[1]*cs,width:cs,height:cs,class:'regc',opacity:.75},g));
  const amax=r.amax;r.cells.forEach((c,i)=>el('circle',{cx:o+c[0]*cs+cs/2,cy:o+c[1]*cs+cs/2,r:Math.max(.6,cs*.48*Math.sqrt(r.al[i]/amax)),class:'wdot'},g));
  const X=o+r.cx*cs+cs/2,Y=o+r.cy*cs+cs/2;el('line',{x1:X-7,x2:X+7,y1:Y,y2:Y,class:'xhair'},g);el('line',{x1:X,x2:X,y1:Y-7,y2:Y+7,class:'xhair'},g);
  const box=$('#c3-chart').parentNode,W=widthOf(box,260,520);
  const lam=OMEGA.map(w=>2*Math.PI/w);
  const F=frame($('#c3-chart'),{W,H:240,m:{l:44,r:14,t:24,b:40},x:[6,40000],xlog:true,y:[0,1.05],xt:[[6.28,'6.28'],[100,'100'],[1000,'1k'],[10000,'10k']],yt:[0,.5,1],yf:v=>String(v),xl:'波長 2π/ω（格，對數）',yl:'|m(ω)|'});
  const ext=a=>{const v=r.cells.map(c=>c[a]);return Math.max(...v)-Math.min(...v)+1},ex=ext(0),ey=ext(1);
  [[ex,'x 範圍','par'],[ey,'y 範圍','perp dash']].forEach(([e,l,cls],k)=>{const xx=F.X(Math.max(6,e));el('line',{x1:xx,x2:xx,y1:F.y1,y2:F.y0,class:cls,style:'stroke-width:1.2'},F.g);tx(F.g,xx+4,F.y0-10-k*16,l+' '+e,{class:'al'})});
  el('path',{d:lp(lam.map((l,i)=>[F.X(l),F.Y(r.mx[i])])),class:'par'},F.g);el('path',{d:lp(lam.map((l,i)=>[F.X(l),F.Y(r.my[i])])),class:'perp dash'},F.g);
  lam.forEach((l,i)=>{el('circle',{cx:F.X(l),cy:F.Y(r.mx[i]),r:3,class:'par-f'},F.g);el('rect',{x:F.X(l)-3,y:F.Y(r.my[i])-3,width:6,height:6,class:'perp-f'},F.g)});
  $('#c3-ro1').innerHTML=`READ 權重（隨 β 改變）<br>最大權重 <b>${r.amax.toFixed(3)}</b><br>權重中心 (x, y) <b>(${r.cx.toFixed(3)}, ${r.cy.toFixed(3)})</b><br>spread（權重 α） <b>${r.spread.toFixed(3)}</b>`;
  $('#c3-ro2').innerHTML=`位置（不隨 β 改變）<br>尺寸索引 ⌊log₂ L⌋ <b>${r.sizeIdx}</b><br>接近完整的頻率數 x / y <b>${r.kx} / ${r.ky}</b><br>λ = 6.28 時 |m_x|, |m_y| <b>${r.mx[0].toFixed(3)}, ${r.my[0].toFixed(3)}</b>`;
  $('#c3-chart').setAttribute('aria-label',`段長 ${L}：x 軸 ${r.kx}、y 軸 ${r.ky} 個頻率接近完整`);
  announce('c3',`起點 ${p}，段長 ${L}，β ${beta.toFixed(1)}：最大權重 ${r.amax.toFixed(3)}，接近完整頻率 ${r.kx} / ${r.ky}`)}
function initC3(){$('#c3-p').addEventListener('input',e=>{c3.p=+e.target.value;renderC3()});$('#c3-L').addEventListener('input',e=>{c3.L=+e.target.value;renderC3()});
  $('#c3-b').addEventListener('input',e=>{c3.beta=+e.target.value;renderC3()});$$('#c3-chips .chip').forEach(c=>c.addEventListener('click',()=>{c3.L=+c.dataset.L;$('#c3-L').value=c3.L;renderC3()}));renderC3()}

/* ---------- C4 ---------- */
const c4={cfg:'b16',R:256,tau:0};
function renderC4(){const c=CORES[c4.cfg],{R,tau}=c4;$$('#c4-cfg button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===c4.cfg)));
  const sl=$('#c4-R');sl.max=c.list.length-1;sl.value=c.list.indexOf(R);sl.setAttribute('aria-valuetext','R = '+R);$('#c4-Ro').textContent=R;$('#c4-tauo').textContent=tau.toFixed(3);$('#c4-tau').setAttribute('aria-valuetext','τ = '+tau.toFixed(3));
  const long=c4.cfg==='b16long',ce=ceilOf(c,tau),pts=[...c.meas,...(c.anly||[])];
  const top=long?1/relCost(16,c,1,tau):ce,ymax=Math.ceil(1.1*Math.max(top,...pts.map(q=>q[1]),1.5)/0.5)*0.5;
  const box=$('#c4-fig'),W=widthOf(box,280,640),yt=[];for(let v=1;v<=ymax+1e-9;v+=0.5)yt.push([v,v.toFixed(1)+'×']);
  const xt=c.N===1024?[16,64,256,1024]:[16,64,256];
  const F=frame($('svg',box),{W,H:300,m:{l:52,r:16,t:28,b:40},x:[16,c.N],xlog:true,y:[1,ymax],xt,yt,xl:'預算 R（對數）',yl:'速度（相對 dense）'});
  const a0=[],a1=[];for(let i=0;i<=200;i++){const r=16*(c.N/16)**(i/200);a0.push([F.X(r),F.Y(1/relCost(r,c,0,tau))]);a1.push([F.X(r),F.Y(1/relCost(r,c,1,tau))])}
  el('path',{d:lp(a0)+'L'+lp(a1.slice().reverse()).slice(1)+'Z',class:'band'},F.g);el('path',{d:lp(a0),class:'tot'},F.g);el('path',{d:lp(a1),class:'ref',style:'stroke-width:2'},F.g);
  if(!long){el('line',{x1:F.x0,x2:F.x1,y1:F.Y(ce),y2:F.Y(ce),class:'ceil'},F.g);tx(F.g,F.x1,F.Y(ce)-6,'上限 '+ce.toFixed(2)+'×',{class:'al','text-anchor':'end'})}
  else tx(F.g,F.x0+6,F.y1+14,'論文未計時；上限高於 2.5×（App. B.5）',{class:'al'});
  c.meas.forEach(([r,s])=>{el('circle',{cx:F.X(r),cy:F.Y(s),r:5,class:'meas'},F.g);tx(F.g,F.X(r)+8,F.Y(s)+4,s.toFixed(2),{class:'labm'})});
  (c.anly||[]).forEach(([r,s])=>{const x=F.X(r),y=F.Y(s);el('line',{x1:x-5,x2:x+5,y1:y-5,y2:y+5,class:'anly'},F.g);el('line',{x1:x-5,x2:x+5,y1:y+5,y2:y-5,class:'anly'},F.g);tx(F.g,x+8,y-6,s.toFixed(2),{class:'labm'})});
  if(c.meas.length)tx(F.g,F.x0+6,F.y1+14,'空心圓：論文量測（Tab. 1、Fig. 16）',{class:'al'});
  if(c.anly)tx(F.g,F.x0+6,F.y1+14,'十字：論文解析值（token 計算量，未計時）',{class:'al'});
  el('line',{x1:F.X(R),x2:F.X(R),y1:F.y1,y2:F.y0,class:'cur'},F.g);
  const c0=relCost(R,c,0,tau),c1v=relCost(R,c,1,tau);let ro=`n_core = <b>${ncore(c)}</b>，B = <b>${c.B}</b><br>R = <b>${R}</b>：成本 <b>${c0.toFixed(3)}</b>（η=0）／<b>${c1v.toFixed(3)}</b>（η=1）<br>速度 <b class="acc">${(1/c0).toFixed(3)}×</b>／<b class="acc">${(1/c1v).toFixed(3)}×</b><br>上限 <b>${long?'高於 2.5×（未計時）':ce.toFixed(3)+'×'}</b>`;
  if(long){const i0=isoR(0),i1=isoR(1);ro+=`<br>與 B/16 [3,13] 在 R = 256 同成本：R ≈ <b>${i0===null?'—':i0.toFixed(1)}</b>（η=0）至 <b>${i1===null?'—':i1.toFixed(1)}</b>（η=1）<br>只計影像 token；真正同吞吐量低於 461`}
  $('#c4-ro').innerHTML=ro;announce('c4',`R = ${R}，速度 ${(1/c0).toFixed(2)} 至 ${(1/c1v).toFixed(2)} 倍`)}
function renderC5(){const s=+$('#c5-s').value;$('#c5-so').textContent=s.toFixed(1);$('#c5-s').setAttribute('aria-valuetext',`${s.toFixed(1)} 秒`);
  const nd=Math.floor(s/0.288+1e-9),nr=Math.floor(s/0.112+1e-9),box=$('#c5-fig'),W=widthOf(box,240,420),svg=$('svg',box),H=120;svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const X=lin(0,110,90,W-50);[['dense',nd,'bar-d',20],['RTI R=256',nr,'bar-r',66]].forEach(([l,n,cls,y])=>{tx(svg,0,y+20,l,{class:'al'});el('rect',{x:90,y,width:Math.max(1,X(n)-90),height:30,rx:4,class:cls},svg);tx(svg,X(n)+6,y+20,n+' 步',{class:'labm'})})}
function initC4(){$$('#c4-cfg button').forEach(b=>b.addEventListener('click',()=>{c4.cfg=b.dataset.v;c4.R=CORES[c4.cfg].def;renderC4()}));
  $('#c4-R').addEventListener('input',e=>{c4.R=CORES[c4.cfg].list[+e.target.value];renderC4()});$('#c4-tau').addEventListener('input',e=>{c4.tau=+e.target.value;renderC4()});
  $('#c5-s').addEventListener('input',renderC5);renderC4();renderC5()}

/* ---------- timeline ---------- */
const TL_GROUP={'2212.09748':['base','rti-lineage-hub'],'2210.09461':['sim','tokc-vit-pruning-and-merging'],'2303.17604':['dm','tokc-training-free-diffusion-merging'],'2402.13573':['dm','tokc-training-free-diffusion-merging'],
 '2304.00287':['ada','tokc-content-adaptive-tokenization'],'2212.11972':['lat','tokc-hourglass-and-latent-interfaces'],'2603.12245':['lat','tokc-hourglass-and-latent-interfaces'],'2404.02258':['skip','rti-prior-reductions-compared'],
 '2412.16822':['dyn','tokc-learned-routing-dynamic-dit'],'2603.06351':['chunk','tokc-dynamic-chunking-bytes-to-pixels'],'2511.13720':['pix','pixel-space-diffusion'],'2608.29281':['me','rti-lineage-hub']};
const TL_NAME={base:'基礎（DiT）',sim:'相似度合併',dm:'擴散中的合併',ada:'自適應 tokenization',lat:'latent 瓶頸／介面',skip:'token 跳過',dyn:'動態壓縮比',chunk:'動態切塊',pix:'像素空間 DiT',me:'本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.map((e,i)=>({e,i})).sort((a,b)=>(+a.e.year||0)-(+b.e.year||0)||a.i-b.i).map(o=>o.e);
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];used.add(g[0]);const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${g[1]?`<a href="${g[1]}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
  const on=new Set(used);
  Object.keys(TL_NAME).filter(k=>used.has(k)).forEach(k=>{const b=document.createElement('button');b.type='button';b.className='chip';b.setAttribute('aria-pressed','true');b.textContent=TL_NAME[k];b.dataset.g=k;
    b.addEventListener('click',()=>{if(on.has(k)&&on.size===used.size){on.clear();on.add(k)}else if(on.has(k)){on.delete(k);if(!on.size)used.forEach(x=>on.add(x))}else on.add(k);
      $$('.chip',chips).forEach(c=>c.setAttribute('aria-pressed',String(on.has(c.dataset.g))));$$('.ev',track).forEach(ev=>ev.classList.toggle('dim',!on.has(ev.dataset.g)))});chips.appendChild(b)});
  const rail=$('#rail'),by=dir=>rail.scrollBy({left:dir*Math.min(rail.clientWidth*.8,560),behavior:RM?'auto':'smooth'});
  $('#tl-prev').addEventListener('click',()=>by(-1));$('#tl-next').addEventListener('click',()=>by(1));
  rail.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){by(1);e.preventDefault()}if(e.key==='ArrowLeft'){by(-1);e.preventDefault()}})}

/* ---------- toc, theme ---------- */
function initToc(){const secs=$$('[data-toc]'),ol=$('#toc'),bar=$('#tocbar');
  secs.forEach(s=>{ol.insertAdjacentHTML('beforeend',`<li><a href="#${s.id}"><span class="n">${s.dataset.toc}</span><span>${s.dataset.tt}</span></a></li>`);bar.insertAdjacentHTML('beforeend',`<a href="#${s.id}">${s.dataset.toc} ${s.dataset.tt}</a>`)});
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;$$('.toc a,.tocbar a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+id));const ab=$(`.tocbar a[href="#${id}"]`);if(ab&&bar.offsetParent)bar.scrollTo({left:ab.offsetLeft-16,behavior:RM?'auto':'smooth'})}})},{rootMargin:'-40% 0px -55% 0px'});
  secs.forEach(s=>io.observe(s));addEventListener('scroll',()=>$('#appbar').classList.toggle('raised',scrollY>8),{passive:true})}
function initTheme(){let saved=null;try{saved=localStorage.getItem('pm-theme')}catch(e){}if(saved)document.documentElement.dataset.theme=saved;
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){}});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',redrawAll)}
function redrawAll(){drawArch();renderC1();renderC2();renderC3();renderC4();renderC5()}

/* ---------- self test ---------- */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};const near=(a,b,t)=>a!==null&&Math.abs(a-b)<=t+1e-9;
  let adj=true;for(let j=0;j<N-1;j++){const a=hilbertXY(32,j),b=hilbertXY(32,j+1);if(Math.abs(a[0]-b[0])+Math.abs(a[1]-b[1])!==1)adj=false}ok(adj,'hilbert adjacency');
  ok(String(mortonXY(8,3))==='1,1'&&String(mortonXY(8,4))==='2,0','morton 3,4');
  [[8,8,4,[[1,1.144,1],[2,1.498,.561],[2,2.093,.140]]],[8,8,0,[[1,1.144,1],[1,1.144,.561],[1,2.000,.140]]],[16,16,40,[[1,1.824,1],[2,2.350,.531],[2,4.057,.066]]],[32,16,300,[[1,1.718,1],[1,1.718,.507],[1,4.000,.539]]]].forEach(([q,L,p,ex])=>
    ['hilbert','morton','raster'].forEach((nm,i)=>{const r=c1compute(nm,q,L,p);ok(r.cc.count===ex[i][0]&&near(r.spread,ex[i][1],.001)&&near(r.frac,ex[i][2],.001),`C1 ${nm} q${q} L${L} p${p}: ${r.cc.count} ${r.spread.toFixed(3)} ${r.frac.toFixed(3)}`)}));
  const r0=mulberry32(1);ok(r0()===0.6270739405881613&&r0()===0.002735721180215478&&r0()===0.5274470399599522,'mulberry32');
  ok(near(detailShare(.15),.946,.001)&&near(detailShare(.5),.998,.001),'detail share '+detailShare(.15).toFixed(3));
  const T2=[[1,64,[.902,.956,.620,192],[.777,1.498,0],[.922,7.797,.375],.140],[1,96,[.960,.740,.575,192],[.780,1.338,-.034],[.938,7.033,.442],.198],[1,128,[.984,.619,.523,192],[.809,1.144,0],[.959,6.438,.343],.242],
    [1,256,[.995,.399,.302,84],[.869,.707,0],[.974,5.357,.201],.391],[1,512,[.997,.121,.040,84],[.912,.500,0],[.990,3.192,.040],.653],[.5,128,[.176,.922,.193,52],[.162,1.144,0],[.305,10.180,.091],.171],[0,128,[.139,.914,-.017,42],[.124,1.144,0],[.291,10.888,.082],.171]];
  T2.forEach(([t,R,a,f,s,sk])=>{const r=c2results(t,R),A=r.stats.adaptive,Fx=r.stats.fixed,Fs=r.stats.featsim;
    ok(near(r.ev.adaptive,a[0],.002)&&near(A.spread,a[1],.002)&&near(A.corr,a[2],.002)&&A.maxsize===a[3],`C2 adaptive t${t} R${R}: ${r.ev.adaptive?.toFixed(3)} ${A.spread.toFixed(3)} ${A.corr.toFixed(3)} ${A.maxsize}`);
    ok(near(r.ev.fixed,f[0],.002)&&near(Fx.spread,f[1],.002)&&near(Fx.corr,f[2],.002),`C2 fixed t${t} R${R}: ${r.ev.fixed?.toFixed(3)} ${Fx.corr.toFixed(3)}`);
    ok(near(r.ev.featsim,s[0],.002)&&near(Fs.spread,s[1],.002)&&near(Fs.corr,s[2],.002),`C2 featsim t${t} R${R}: ${r.ev.featsim?.toFixed(3)} ${Fs.spread.toFixed(3)} ${Fs.corr.toFixed(3)}`);
    ok(near(r.ev.skip,sk,.002),`C2 skip t${t} R${R}: ${r.ev.skip?.toFixed(3)}`)});
  [[0,1,1,0,0,0,0,1,16,16,1,1],[0,2,1,1,0,.485,.5,.515,16,15,1,.878],[0,4,1,2,.495,.503,.707,.260,15,15,.878,.878],[0,16,1,4,1.470,1.477,1.489,.069,14,14,.474,.474],[0,64,1,6,4.562,4.739,3.228,.194,13,13,.197,.197],
   [100,37,0,5,10.703,6.595,2.880,.027,13,13,.429,.287],[100,37,1,5,9.463,7.716,2.008,.136,13,13,.429,.287],[100,37,4,5,9.172,7.079,.869,.363,13,13,.429,.287]].forEach(([p,L,b,si,cx,cy,sp,am,kx,ky,mx,my])=>{const r=readRegion(p,L,b);
    ok(r.sizeIdx===si&&near(r.cx,cx,.001)&&near(r.cy,cy,.001)&&near(r.spread,sp,.001)&&near(r.amax,am,.001)&&r.kx===kx&&r.ky===ky&&near(r.mx[0],mx,.001)&&near(r.my[0],my,.001),`C3 p${p} L${L} b${b}: ${r.cx.toFixed(3)} ${r.cy.toFixed(3)} ${r.spread.toFixed(3)} ${r.amax.toFixed(3)} ${r.kx}/${r.ky}`)});
  ok(Math.max(...OMEGA.map(w=>Math.abs(phasorMag([0,1],[.5,.5],w)-dirichlet(w,2))))<1e-12,'dirichlet L=2');
  ok(near(relCost(256,CORES.b16long,0,0)/relCost(256,CORES.b16,0,0),.657,.001),'Tab3 linear');ok(near(relCost(256,CORES.b16long,1,0)/relCost(256,CORES.b16,1,0),.439,.001),'Tab3 attn');
  ok(near(isoR(0),460.8,.1)&&near(isoR(1),572.4,.1),'iso R');ok(near(1/relCost(64,CORES.jit,0,0),1.600,.001)&&near(1/relCost(130,CORES.jit,0,0),1.326,.001)&&near(1/relCost(193,CORES.jit,0,0),1.140,.001),'JiT analytic');
  ok(near(1/relCost(64,CORES.l16,0,0),3.257,.001),'L16 R64 band');ok(near(ceilOf(CORES.b16,0),2.833,.001)&&near(ceilOf(CORES.l16,0),3.833,.001)&&near(ceilOf(CORES.b16,.0727),2.5,.001),'ceilings');
  ok(CORES.b16.list.includes(256)&&CORES.b16.list.includes(512)&&CORES.jit.list.includes(130)&&CORES.jit.list.includes(193),'C4 list');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'')}

/* ---------- boot ---------- */
initTheme();initToc();drawHero();drawArch();initC1();initC2();initC3();initC4();renderTimeline();
$('#tab1-more').addEventListener('click',e=>{const t=$('#tab1'),on=!t.classList.contains('all');t.classList.toggle('all',on);e.target.setAttribute('aria-pressed',String(on));e.target.textContent=on?'隱藏部分欄位':'顯示全部欄位'});
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;clearTimeout(rz);rz=setTimeout(redrawAll,150)});
new MutationObserver(redrawAll).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))selfTest();
})();
