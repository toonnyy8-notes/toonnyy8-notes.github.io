(()=>{
'use strict';
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
function el(tag,a,p){const e=document.createElementNS(NS,tag);if(a)for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function tx(p,x,y,s,a){const t=el('text',Object.assign({x,y},a||{}),p);t.textContent=s;return t}
// linear / log scales clamp to the axis range; NaN, 0 and negatives go to the lower edge (no -Infinity in paths)
const scaleLin=(d0,d1,r0,r1)=>v=>{if(!(v>=d0))v=d0;if(v>d1)v=d1;return r0+(v-d0)/(d1-d0)*(r1-r0)};
const scaleLog=(d0,d1,r0,r1)=>{const a=Math.log(d0),b=Math.log(d1);return v=>{if(!(v>d0))v=d0;if(v>d1)v=d1;return r0+(Math.log(v)-a)/(b-a)*(r1-r0)}};
function linePath(pts){let s='';for(let i=0;i<pts.length;i++)s+=(i?'L':'M')+pts[i][0].toFixed(1)+' '+pts[i][1].toFixed(1);return s}
const f4=v=>Number.isFinite(v)?v.toFixed(4):'—';
function widthOf(box,min,max){return Math.max(min,Math.min(max,Math.floor(box.clientWidth||max)))}
function frame(svg,o){
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${o.W} ${o.H}`);svg.setAttribute('width',o.W);svg.setAttribute('height',o.H);
  const m=o.m,x0=m.l,x1=o.W-m.r,y0=o.H-m.b,y1=m.t;
  const X=scaleLin(o.x[0],o.x[1],x0,x1),Y=(o.ylog?scaleLog:scaleLin)(o.y[0],o.y[1],y0,y1);
  const g=el('g',null,svg);
  if(o.bands)o.bands.forEach(([a,b,cls,lab])=>{el('rect',{x:X(a),y:y1,width:X(b)-X(a),height:y0-y1,class:cls},g);if(lab)tx(g,(X(a)+X(b))/2,y1+14,lab,{class:'al','text-anchor':'middle'})});
  (o.yt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:String(v),yy=Y(val);el('line',{x1:x0,x2:x1,y1:yy,y2:yy,class:'grid'},g);tx(g,x0-8,yy+4,lab,{class:'tk','text-anchor':'end'})});
  (o.xt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:String(v),xx=X(val);el('line',{x1:xx,x2:xx,y1:y0,y2:y0+4,class:'axl'},g);tx(g,xx,y0+18,lab,{class:'tk','text-anchor':'middle'})});
  el('line',{x1:x0,x2:x1,y1:y0,y2:y0,class:'axl'},g);
  if(o.xl)tx(g,(x0+x1)/2,o.H-4,o.xl,{class:'al','text-anchor':'middle'});
  return {X,Y,g,x0,x1,y0,y1};
}
// out-of-range marker: triangle at the top (up) or bottom (down) edge
function edgeMark(F,x,up){const y=up?F.y1:F.y0,d=up?-1:1;el('path',{d:`M${x-5} ${y-d*2}L${x+5} ${y-d*2}L${x} ${y+d*6}Z`,class:'warn'},F.g)}
function markOut(F,xs,vals,lo,hi){let st=0;for(let i=0;i<vals.length;i++){const v=vals[i],s=v>hi?1:(!(v>lo)?-1:0);if(s&&s!==st)edgeMark(F,xs[i],s>0);st=s}}
const LOGT=[[1e-3,'10⁻³'],[1e-2,'10⁻²'],[1e-1,'10⁻¹'],[1,'1'],[10,'10'],[100,'10²'],[1e3,'10³']];
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function toggleTable(btn,box){btn.addEventListener('click',()=>{const open=box.classList.toggle('hide')===false;btn.setAttribute('aria-expanded',String(open));btn.textContent=open?'隱藏資料表':'顯示資料表'})}

/* ---------- reference math (copied from verify/widgets.js, algorithms unchanged) ---------- */
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2);}
const D=8,T=96,POOL=4;
const ctxOf=t=>(t<=32||t>64)?0:1;
const streamCache=new Map();
function makeStream(seed){if(streamCache.has(seed))return streamCache.get(seed);const r=mulberry32(seed);const keys=[];for(let c=0;c<2;c++){const pool=[];for(let i=0;i<POOL;i++){const k=Array.from({length:D},()=>gauss(r));const n=Math.hypot(...k);pool.push(k.map(x=>x/n));}keys.push(pool);}
  const mats=[0,1].map(()=>Array.from({length:D},()=>Array.from({length:D},()=>gauss(r)/Math.sqrt(D))));
  const seq=[];for(let t=1;t<=T;t++){seq.push([ctxOf(t),Math.min(POOL-1,Math.floor(r()*POOL))]);}const s={keys,mats,seq};streamCache.set(seed,s);return s;}
const mv=(M,k)=>M.map(row=>row.reduce((s,x,j)=>s+x*k[j],0));
const BIG=1e3;
function runC1({seed=11,theta=0.3,eta=0.6,alpha=0.02,chunk=1}={}){
  const {keys,mats,seq}=makeStream(seed);const V=[0,1].map(c=>keys[c].map(k=>mv(mats[c],k)));
  let M=Array.from({length:D},()=>new Array(D).fill(0)),S=Array.from({length:D},()=>new Array(D).fill(0)),Mref=M;const out=[];let diverged=null;
  for(let t=1;t<=T;t++){const [c,j]=seq[t-1];if((t-1)%chunk===0)Mref=M.map(r=>r.slice());
    const k=keys[c][j],v=V[c][j],e=mv(Mref,k).map((x,i)=>x-v[i]);const surprise=Math.hypot(...e);
    let mx=0;for(let a=0;a<D;a++)for(let b=0;b<D;b++){S[a][b]=eta*S[a][b]-theta*e[a]*k[b];M[a][b]=(1-alpha)*M[a][b]+S[a][b];mx=Math.max(mx,Math.abs(M[a][b]));}
    const Sn=Math.hypot(...S.flat());const rec=[0,1].map(cc=>keys[cc].reduce((s,kk,i)=>{const p=mv(M,kk);return s+0.5*p.reduce((q,x,m)=>q+(x-V[cc][i][m])**2,0);},0)/POOL);
    out.push({t,c,surprise,S:Sn,EA:rec[0],EB:rec[1]});
    if(!(mx<BIG)){diverged=t;break;}}
  return{out,M,diverged};}
function matmulEnds({seed=11,theta=0.3,eta=0.6,alpha=0.02,b=8}={}){
  const {keys,mats,seq}=makeStream(seed);let M=Array.from({length:D},()=>new Array(D).fill(0)),S=M.map(r=>r.slice());const ends=[];const g=1-alpha;
  for(let s0=0;s0<T;s0+=b){const idx=[];for(let t=s0;t<Math.min(T,s0+b);t++)idx.push(t);const n=idx.length;
    const K=idx.map(t=>keys[seq[t][0]][seq[t][1]]);const E=idx.map((t,i)=>{const v=mv(mats[seq[t][0]],K[i]);return mv(M,K[i]).map((x,m)=>x-v[m]);});
    const w=[],q=[];for(let l=0;l<n;l++){let s=0;for(let m=l;m<n;m++)s+=g**(n-1-m)*eta**(m-l);w.push(s);q.push(-theta*eta**(n-1-l));}
    let cS=0;for(let m=0;m<n;m++)cS+=g**(n-1-m)*eta**(m+1);const cM=g**n,en=eta**n;
    const nM=M.map((row,a)=>row.map((x,c)=>{let s=0;for(let l=0;l<n;l++)s+=w[l]*E[l][a]*K[l][c];return cM*x+cS*S[a][c]-theta*s;}));
    const nS=S.map((row,a)=>row.map((x,c)=>{let s=0;for(let l=0;l<n;l++)s+=q[l]*E[l][a]*K[l][c];return en*x+s;}));
    M=nM;S=nS;ends.push(M.map(r=>r.slice()));}
  return ends;}
function loopEnds(o){const {keys,mats,seq}=makeStream(o.seed??11);const th=o.theta??0.3,et=o.eta??0.6,al=o.alpha??0.02,b=o.b;
  let M=Array.from({length:D},()=>new Array(D).fill(0)),S=M.map(r=>r.slice()),Mref=M;const ends=[];
  for(let t=0;t<T;t++){const [c,j]=seq[t];if(t%b===0)Mref=M.map(r=>r.slice());const k=keys[c][j],v=mv(mats[c],k),e=mv(Mref,k).map((x,i)=>x-v[i]);
    for(let a=0;a<D;a++)for(let cc=0;cc<D;cc++){S[a][cc]=et*S[a][cc]-th*e[a]*k[cc];M[a][cc]=(1-al)*M[a][cc]+S[a][cc];}
    if((t+1)%b===0||t===T-1)ends.push(M.map(r=>r.slice()));}return ends;}
const H=16;const sig=a=>a>=0?1/(1+Math.exp(-a)):Math.exp(a)/(1+Math.exp(a));const target=k=>Math.sin(2*k);
function initMLP(L,seed){const r=mulberry32(seed);if(L===1)return[{W:[[gauss(r)]],b:null}];const dims=[1,...Array(L-1).fill(H),1];const P=[];
  for(let l=0;l<L;l++){const fi=dims[l],fo=dims[l+1];const W=Array.from({length:fo},()=>Array.from({length:fi},()=>gauss(r)*Math.sqrt(1/fi)));const b=l<L-1?Array.from({length:fo},()=>gauss(r)*0.5):null;P.push({W,b});}return P;}
function fwd(P,k){const hs=[[k]],As=[];for(let l=0;l<P.length;l++){const p=P[l],h=hs[hs.length-1];const a=p.W.map((wr,i)=>wr.reduce((s,x,j)=>s+x*h[j],0)+(p.b?p.b[i]:0));
  if(l<P.length-1){As.push(a);hs.push(a.map(x=>x*sig(x)));}else return{out:a[0],hs,As};}}
function step(P,S,k,v,theta,eta,alpha){const {out,hs,As}=fwd(P,k);let d=[out-v];
  for(let l=P.length-1;l>=0;l--){const p=P[l],s=S[l],h=hs[l];let dh=null;
    if(l>0){dh=h.map((_,j)=>p.W.reduce((q,wr,i)=>q+wr[j]*d[i],0));}
    for(let i=0;i<p.W.length;i++){for(let j=0;j<h.length;j++){s.W[i][j]=eta*s.W[i][j]-theta*d[i]*h[j];p.W[i][j]=(1-alpha)*p.W[i][j]+s.W[i][j];}
      if(p.b){s.b[i]=eta*s.b[i]-theta*d[i];p.b[i]=(1-alpha)*p.b[i]+s.b[i];}}
    if(l>0){const a=As[l-1];d=dh.map((x,j)=>{const sg=sig(a[j]);return x*sg*(1+a[j]*(1-sg));});}}
  return 0.5*(out-v)**2;}
function evalMSE(P,n=201){let s=0;for(let i=0;i<n;i++){const k=-2+4*i/(n-1);s+=(fwd(P,k).out-target(k))**2;}return s/n;}
function runDeep({L=2,theta=0.01,eta=0.9,alpha=0,Tn=3000,seed=5,dataSeed=9,cps=[100,300,1000,3000]}={}){
  const P=initMLP(L,seed),S=P.map(p=>({W:p.W.map(r=>r.map(()=>0)),b:p.b?p.b.map(()=>0):null}));const r=mulberry32(dataSeed);const res={};let div=null;
  for(let t=1;t<=Tn;t++){const k=-2+4*r();const loss=step(P,S,k,target(k),theta,eta,alpha);
    let mx=0;for(const p of P)for(const row of p.W)for(const x of row)mx=Math.max(mx,Math.abs(x));
    if(!Number.isFinite(loss)||!(mx<=BIG)){div=t;break;}if(cps.includes(t))res[t]=evalMSE(P);}
  return{res,div,P};}
function maskRow(v,{N=16,Np=2,C=4,Nl=2,w=4,t}){
  const seq=[];for(let j=1;j<=N;j++){let cat='none';
    if(v==='MAC'){const seg=Math.ceil(t/C),sj=Math.ceil(j/C);if(j<=t)cat=(sj===seg)?'direct':'memory';}
    else{if(j<=t)cat=(j>t-w)?'direct':'memory';}seq.push(cat);}
  const persist=new Array(Np).fill(v==='MAL'?'memory':'direct');const mem=v==='MAC'?new Array(Nl).fill('direct'):[];return{persist,mem,seq};}
function counts(v,o){const r=maskRow(v,o);const all=[...r.persist,...r.mem,...r.seq];return{direct:all.filter(x=>x==='direct').length,memory:all.filter(x=>x==='memory').length,none:all.filter(x=>x==='none').length};}
function pairs(v,o){let s=0;for(let t=1;t<=o.N;t++)s+=counts(v,{...o,t}).direct;return s;}
const nParams=P=>P.reduce((s,p)=>s+p.W.length*p.W[0].length+(p.b?p.b.length:0),0);

/* ---------- hero token axis ---------- */
function heroAxis(){const box=$('#hero-axis'),narrow=box.clientWidth<480,n=narrow?8:16,W=narrow?360:720,cw=W/n,H=120,np=2,nw=narrow?3:4;
  box.innerHTML='';const s=el('svg',{viewBox:`0 0 ${W} ${H}`,'aria-hidden':'true'},box);
  const defs=el('defs',null,s);
  [['hx-mem','hatch-l'],['hx-per','hatch-p']].forEach(([id,cls])=>{const p=el('pattern',{id,width:6,height:6,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},defs);el('line',{x1:0,y1:0,x2:0,y2:6,class:cls},p)});
  const by=40,bh=14,cy=62,ch=cw-8>34?34:cw-8;
  // bands above the cells
  el('rect',{x:np*cw+2,y:by,width:(n-np-nw)*cw-4,height:bh,fill:'url(#hx-mem)',class:'',rx:3},s);
  tx(s,np*cw+4,by-8,'長期記憶（已壓縮）',{class:'al'});
  el('rect',{x:(n-nw)*cw+2,y:by,width:nw*cw-4,height:bh,rx:3,class:'cellacc'},s);
  tx(s,W-4,by-8,'注意力視窗',{class:'al','text-anchor':'end'});
  for(let i=0;i<n;i++){const x=i*cw+4;el('rect',{x,y:cy,width:cw-8,height:ch,rx:4,class:'cellbox'},s);if(i<np)el('rect',{x,y:cy,width:cw-8,height:ch,rx:4,fill:'url(#hx-per)'},s)}
  tx(s,4,by-8,'持久記憶',{class:'al'});
  el('line',{x1:np*cw,x2:W,y1:cy+ch+6,y2:cy+ch+6,class:'axl'},s);
  tx(s,W-4,cy+ch+20,'token →',{class:'tk','text-anchor':'end'});
}

/* ---------- C1 ---------- */
const c1={seed:11,theta:.3,eta:.6,alpha:.02,alerted:''};
function c1params(){return{seed:c1.seed,theta:c1.theta,eta:c1.eta,alpha:c1.alpha}}
function renderC1(){
  const R=runC1(c1params()),div=R.diverged,out=div?R.out.slice(0,-1):R.out;
  $('#c1-tho').textContent=c1.theta.toFixed(2);$('#c1-eto').textContent=c1.eta.toFixed(2);$('#c1-alo').textContent=c1.alpha.toFixed(3);
  $('#c1-th').setAttribute('aria-valuetext','θ = '+c1.theta.toFixed(2));$('#c1-et').setAttribute('aria-valuetext','η = '+c1.eta.toFixed(2));$('#c1-al').setAttribute('aria-valuetext','α = '+c1.alpha.toFixed(3));
  $$('#c1 .chip').forEach(c=>{const [a,b,d]=c.dataset.p.split(',').map(Number);c.setAttribute('aria-pressed',String(Math.abs(a-c1.theta)<1e-9&&Math.abs(b-c1.eta)<1e-9&&Math.abs(d-c1.alpha)<1e-9))});
  $('#c1-seed').textContent='seed '+c1.seed;
  $('#c1-hint').classList.toggle('hide',c1.seed===11);$('#c1-hint-t').textContent=`內文數字依 seed 11；目前 seed 為 ${c1.seed}，數值會不同。`;
  const bands=[[.5,32.5,'band-a','情境 A'],[32.5,64.5,'band-b','情境 B'],[64.5,96.5,'band-a','情境 A']];
  const xt=[1,16,32,48,64,80,96];
  // top
  const bt=$('#c1-top'),W=widthOf(bt,280,640);
  const F=frame($('svg',bt),{W,H:190,m:{l:40,r:56,t:22,b:34},x:[.5,96.5],y:[0,2],xt,yt:[[0,'0'],[.5,'0.5'],[1,'1'],[1.5,'1.5'],[2,'2']],xl:'token t',bands});
  const bw=Math.max(1.5,(F.X(2)-F.X(1))*.6);
  out.forEach(o=>{const x=F.X(o.t),y=F.Y(o.surprise);el('rect',{x:x-bw/2,y,width:bw,height:Math.max(0,F.y0-y),class:'bar'},F.g)});
  markOut(F,out.map(o=>F.X(o.t)),out.map(o=>o.surprise),-1,2);
  const sp=out.map(o=>[F.X(o.t),F.Y(o.S)]);el('path',{d:linePath(sp),class:'par',style:'stroke-width:2'},F.g);
  sp.forEach((p,i)=>{if(i%2===0)el('circle',{cx:p[0],cy:p[1],r:2.2,class:'par-f'},F.g)});
  markOut(F,sp.map(p=>p[0]),out.map(o=>o.S),-1,2);
  if(sp.length)tx(F.g,F.x1+6,sp[sp.length-1][1]+4,'‖S‖',{class:'labm',style:'fill:var(--c-par)'});
  // bottom
  const bb=$('#c1-bot'),W2=widthOf(bb,280,640);
  const G=frame($('svg',bb),{W:W2,H:210,m:{l:40,r:56,t:22,b:34},x:[.5,96.5],y:[1e-3,1e3],ylog:true,xt,yt:LOGT.filter((_,i)=>i%2===0),xl:'token t',bands});
  const pa=out.map(o=>[G.X(o.t),G.Y(o.EA)]),pb=out.map(o=>[G.X(o.t),G.Y(o.EB)]);
  el('path',{d:linePath(pa),class:'par'},G.g);el('path',{d:linePath(pb),class:'perp dash'},G.g);
  markOut(G,pa.map(p=>p[0]),out.map(o=>o.EA),1e-3,1e3);markOut(G,pb.map(p=>p[0]),out.map(o=>o.EB),1e-3,1e3);
  if(pa.length){let ya=pa[pa.length-1][1],yb=pb[pb.length-1][1];if(Math.abs(ya-yb)<14){if(ya<yb){ya-=7;yb+=7}else{ya+=7;yb-=7}}
    tx(G.g,G.x1+6,ya+4,'情境 A',{class:'lab',style:'fill:var(--c-par)'});tx(G.g,G.x1+6,yb+4,'情境 B',{class:'lab',style:'fill:var(--c-perp)'})}
  // readouts
  const at=t=>R.out[t-1]&&(!div||t<div)?R.out[t-1]:null;
  const v=(t,k)=>{const o=at(t);return o?f4(o[k]):'—'};
  const ms=(a,b)=>{if(!at(b))return'—';let s=0;for(let t=a;t<=b;t++)s+=R.out[t-1].surprise;return f4(s/(b-a+1))};
  $('#c1-ro').innerHTML=`E_B(64) = <b class="acc">${v(64,'EB')}</b>　動量／步長<br>E_A(64) = <b>${v(64,'EA')}</b>　遺忘<br>‖S₄₀‖_F = <b>${v(40,'S')}</b>　動量延續`;
  $('#c1-ro2').innerHTML=`‖g₁‖ = <b>${v(1,'surprise')}</b><br>平均驚訝度 t = 33–40：<b>${ms(33,40)}</b><br>平均驚訝度 t = 65–72：<b>${ms(65,72)}</b><br>E_A(32) = <b>${v(32,'EA')}</b><br>E_A(96) = <b>${v(96,'EA')}</b>`;
  const msg=div?`第 ${div} 步數值發散：θ 或 η 太大`:'';const key=div?JSON.stringify(c1params()):'';
  if(key!==c1.alerted){$('#c1-alert').textContent=msg;c1.alerted=key}
  const summ=`seed ${c1.seed}，θ ${c1.theta.toFixed(2)}、η ${c1.eta.toFixed(2)}、α ${c1.alpha.toFixed(3)}：E_B(64) ${v(64,'EB')}，E_A(64) ${v(64,'EA')}，‖S₄₀‖ ${v(40,'S')}`;
  $('#c1-top svg').setAttribute('aria-label','驚訝度與動量圖。'+summ+'。逐點值見資料表。');$('#c1-bot svg').setAttribute('aria-label','回想誤差圖。'+summ+'。逐點值見資料表。');
  $('#c1-cap').textContent=`每 8 步一列；${summ}`;
  $('#c1-tab tbody').innerHTML=Array.from({length:12},(_,i)=>{const t=8*(i+1),o=at(t);return`<tr><td class="num">${t}</td><td class="num">${o?f4(o.surprise):'—'}</td><td class="num">${o?f4(o.S):'—'}</td><td class="num">${o?f4(o.EA):'—'}</td><td class="num">${o?f4(o.EB):'—'}</td></tr>`}).join('');
  c1.summary=summ;
  return R;
}
function initC1(){
  const bind=(id,k)=>{const n=$(id);n.addEventListener('input',()=>{c1[k]=+n.value;renderC1();renderC3()});n.addEventListener('change',()=>announce('c1',c1.summary))};
  bind('#c1-th','theta');bind('#c1-et','eta');bind('#c1-al','alpha');
  $$('#c1 .chip').forEach(c=>c.addEventListener('click',()=>{const [a,b,d]=c.dataset.p.split(',').map(Number);c1.theta=a;c1.eta=b;c1.alpha=d;$('#c1-th').value=a;$('#c1-et').value=b;$('#c1-al').value=d;renderC1();announce('c1',c1.summary)}));
  $('#c1-re').addEventListener('click',()=>{c1.seed++;renderC1();renderC3();announce('c1',c1.summary)});
  $('#c1-back').addEventListener('click',()=>{c1.seed=11;renderC1();renderC3();announce('c1',c1.summary);$('#c1-re').focus()});
  toggleTable($('#c1-tab-b'),$('#c1-tab'));
  renderC1();}

/* ---------- C2 ---------- */
const C2T=3000,C2PER=30,REF_LIN=0.4018,REF_ZERO=0.4389;
const c2={L:2,theta:.01,eta:.9,alpha:0,st:null,raf:0};
function c2new(){const P=initMLP(c2.L,5),S=P.map(p=>({W:p.W.map(r=>r.map(()=>0)),b:p.b?p.b.map(()=>0):null}));
  return{P,S,r:mulberry32(9),t:0,div:null,hist:[[0,evalMSE(P)]],recent:[],running:false,alerted:false}}
function c2advance(st,n){for(let i=0;i<n&&st.t<C2T&&!st.div;i++){const k=-2+4*st.r();st.t++;const loss=step(st.P,st.S,k,target(k),c2.theta,c2.eta,c2.alpha);
  let mx=0;for(const p of st.P)for(const row of p.W)for(const x of row)mx=Math.max(mx,Math.abs(x));
  if(!Number.isFinite(loss)||!(mx<=BIG)){st.div=st.t;break}
  st.recent.push(k);if(st.recent.length>30)st.recent.shift();if(st.t%50===0)st.hist.push([st.t,evalMSE(st.P)])}}
function renderC2(){const st=c2.st;
  $$('#c2-L button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.v===c2.L)));
  $('#c2-tho').textContent=c2.theta.toFixed(3);$('#c2-eto').textContent=c2.eta.toFixed(2);$('#c2-alo').textContent=c2.alpha.toFixed(4);
  // function plot
  const fb=$('#c2-fn'),W=widthOf(fb,260,460);
  const F=frame($('svg',fb),{W,H:250,m:{l:40,r:12,t:16,b:36},x:[-2,2],y:[-1.6,1.6],xt:[-2,-1,0,1,2],yt:[[-1.5,'−1.5'],[-1,'−1'],[0,'0'],[1,'1'],[1.5,'1.5']],xl:'key k'});
  const tg=[];for(let i=0;i<=120;i++){const k=-2+4*i/120;tg.push([F.X(k),F.Y(target(k))])}el('path',{d:linePath(tg),class:'ref'},F.g);
  st.recent.forEach(k=>el('circle',{cx:F.X(k),cy:F.Y(target(k)),r:3,class:'hollow',style:'stroke-width:1.5'},F.g));
  if(!st.div){const mp=[],ys=[];for(let i=0;i<=120;i++){const k=-2+4*i/120,y=fwd(st.P,k).out;mp.push([F.X(k),F.Y(y)]);ys.push(y)}el('path',{d:linePath(mp),class:'acc',style:'stroke-width:2.5'},F.g);markOut(F,mp.map(p=>p[0]),ys,-1.6,1.6)}
  // mse plot
  const mb=$('#c2-mse'),W2=widthOf(mb,260,460);
  const G=frame($('svg',mb),{W:W2,H:250,m:{l:44,r:12,t:16,b:36},x:[0,C2T],y:[1e-5,10],ylog:true,xt:[[0,'0'],[1000,'1000'],[2000,'2000'],[3000,'3000']],yt:[[1e-5,'10⁻⁵'],[1e-3,'10⁻³'],[1e-1,'10⁻¹'],[10,'10']],xl:'token 數'});
  [[REF_LIN,'最佳線性 0.402',-6],[REF_ZERO,'記憶為 0：0.439',14]].forEach(([v,l,dy])=>{el('line',{x1:G.x0,x2:G.x1,y1:G.Y(v),y2:G.Y(v),class:'ref'},G.g);tx(G.g,G.x1,G.Y(v)+dy,l,{class:'al','text-anchor':'end'})});
  const hp=st.hist.map(([t,v])=>[G.X(t),G.Y(v)]);el('path',{d:linePath(hp),class:'acc',style:'stroke-width:2.5'},G.g);
  markOut(G,hp.map(p=>p[0]),st.hist.map(h=>h[1]),1e-5,10);
  if(hp.length)el('circle',{cx:hp[hp.length-1][0],cy:hp[hp.length-1][1],r:4,class:'kdot'},G.g);
  const np=nParams(st.P),cur=st.div?NaN:(st.t%50===0?st.hist[st.hist.length-1][1]:evalMSE(st.P));
  $('#c2-ro').innerHTML=`token 數 <b>${st.t}</b>　格點 MSE <b class="acc">${st.div?'—':cur.toFixed(5)}</b><br>參數數 <b>${np}</b>　相對計算量（每 token 乘加約 3 × 參數數）<b>${3*np}</b>`;
  if(st.div&&!st.alerted){$('#c2-alert').textContent=`第 ${st.div} 步數值發散：θ 或 η 太大`;st.alerted=true}
  const run=$('#c2-run');run.textContent=st.running?'暫停':(st.t>=C2T||st.div?'已完成':'執行');run.disabled=!st.running&&(st.t>=C2T||!!st.div);
  const summ=`L = ${c2.L}，θ ${c2.theta.toFixed(3)}、η ${c2.eta.toFixed(2)}、α ${c2.alpha.toFixed(4)}，token ${st.t}，MSE ${st.div?'發散':cur.toFixed(5)}`;
  $('#c2-mse svg').setAttribute('aria-label','格點 MSE 對 token 數。'+summ+'。逐點值見資料表。');c2.summary=summ;
  $('#c2-cap').textContent=`每 250 個 token 一列；${summ}`;
  const hm=new Map(st.hist);
  $('#c2-tab tbody').innerHTML=Array.from({length:12},(_,i)=>{const t=250*(i+1),v=hm.get(t);return`<tr><td class="num">${t}</td><td class="num">${v===undefined?'—':v.toFixed(5)}</td></tr>`}).join('')+`<tr class="hl"><td class="num">目前 ${st.t}</td><td class="num">${st.div?'—':cur.toFixed(5)}</td></tr>`;
}
function c2stop(){const st=c2.st;st.running=false;cancelAnimationFrame(c2.raf);renderC2();announce('c2',c2.summary)}
function c2run(){const st=c2.st;if(st.running){c2stop();return}if(st.t>=C2T||st.div)return;st.running=true;
  if(RM){c2advance(st,C2T);c2stop();return}
  const tick=()=>{if(!st.running)return;c2advance(st,C2PER);renderC2();if(st.t<C2T&&!st.div)c2.raf=requestAnimationFrame(tick);else c2stop()};c2.raf=requestAnimationFrame(tick)}
function c2reset(){if(c2.st){c2.st.running=false;cancelAnimationFrame(c2.raf)}c2.st=c2new();$('#c2-alert').textContent='';renderC2()}
function initC2(){
  $$('#c2-L button').forEach(b=>b.addEventListener('click',()=>{c2.L=+b.dataset.v;c2reset();announce('c2',c2.summary)}));
  [['#c2-th','theta',3],['#c2-et','eta',2],['#c2-al','alpha',4]].forEach(([id,k,d])=>{const n=$(id);
    n.addEventListener('input',()=>{$(id+'o').textContent=(+n.value).toFixed(d)});
    n.addEventListener('change',()=>{c2[k]=+n.value;c2reset();announce('c2',c2.summary)})});
  $('#c2-run').addEventListener('click',c2run);$('#c2-reset').addEventListener('click',()=>{c2reset();announce('c2',c2.summary)});
  $('#c2-cmp').addEventListener('click',()=>{const rows=[1,2,3,4].map(L=>{const r=runDeep({L,theta:c2.theta,eta:c2.eta,alpha:c2.alpha});const cells=[100,300,1000,3000].map(t=>r.res[t]!==undefined?r.res[t].toFixed(5):(r.div?`第 ${r.div} 步發散`:'—'));return`<tr${L===c2.L?' class="hl"':''}><td class="num">${L}</td><td class="num">${nParams(r.P)}</td>${cells.map(c=>`<td class="num">${c}</td>`).join('')}</tr>`});
    $('#c2-cmpbox tbody').innerHTML=rows.join('');$('#c2-cmpcap').textContent=`四種深度的格點 MSE（θ ${c2.theta.toFixed(3)}、η ${c2.eta.toFixed(2)}、α ${c2.alpha.toFixed(4)}）`;$('#c2-cmpbox').classList.remove('hide');announce('c2','已填入四種深度的比較表')});
  toggleTable($('#c2-tab-b'),$('#c2-tab'));
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&c2.st&&c2.st.running)c2stop()});
  c2reset();}

/* ---------- C3 ---------- */
const c3={b:4,eta:.6};
function c3compute(){const seed=c1.seed,p={seed,theta:.3,eta:c3.eta,alpha:.02};
  const R1=runC1({...p,chunk:1}),Rb=runC1({...p,chunk:c3.b});
  const A=loopEnds({...p,b:c3.b}),B=matmulEnds({...p,b:c3.b});let md=0;A.forEach((M,i)=>M.forEach((r,a)=>r.forEach((x,c)=>{md=Math.max(md,Math.abs(x-B[i][a][c]))})));
  const f1=R1.M.flat(),dev=Math.hypot(...Rb.M.flat().map((x,i)=>x-f1[i]));
  return{R1,Rb,md,dev,seed}}
function renderC3(){
  $$('#c3-b button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.v===c3.b)));$$('#c3-eta button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.v===c3.eta)));
  const r=c3compute(),b=c3.b;
  // token strip
  const sb=$('#c3-strip'),W=widthOf(sb,280,900),cw=(W-8)/T,s=$('svg',sb),Hs=46;s.textContent='';s.setAttribute('viewBox',`0 0 ${W} ${Hs}`);s.setAttribute('width',W);s.setAttribute('height',Hs);
  for(let t=0;t<T;t++){el('rect',{x:4+t*cw,y:18,width:cw,height:16,class:ctxOf(t+1)?'band-b':'band-a'},s)}
  for(let s0=0;s0<=T;s0+=b){const x=4+s0*cw;el('line',{x1:x,x2:x,y1:16,y2:36,class:'chunkline'},s);if(s0<T)el('path',{d:`M${x} 14L${x-3.5} 7L${x+3.5} 7Z`,class:'warn'},s)}
  el('line',{x1:4,x2:W-4,y1:18,y2:18,class:'axl'},s);el('line',{x1:4,x2:W-4,y1:34,y2:34,class:'axl'},s);
  tx(s,4,Hs-1,'t = 1',{class:'tk'});tx(s,W-4,Hs-1,'96',{class:'tk','text-anchor':'end'});
  $('#c3-strip svg').setAttribute('aria-label',`96 個 token 分成 ${Math.ceil(T/b)} 個 chunk，每個 chunk ${b} 個 token；三角形標出每個 chunk 的梯度點`);
  // main chart
  const fb=$('#c3-fig'),W2=widthOf(fb,280,640);
  const F=frame($('svg',fb),{W:W2,H:230,m:{l:40,r:16,t:22,b:34},x:[.5,96.5],y:[1e-3,1e3],ylog:true,xt:[1,16,32,48,64,80,96],yt:LOGT.filter((_,i)=>i%2===0),xl:'token t',bands:[[.5,32.5,'band-a','情境 A'],[32.5,64.5,'band-b','情境 B'],[64.5,96.5,'band-a','情境 A']]});
  const cur=o=>o.c===0?o.EA:o.EB;
  const p1=r.R1.out.map(o=>[F.X(o.t),F.Y(cur(o))]),pb=r.Rb.out.map(o=>[F.X(o.t),F.Y(cur(o))]);
  el('path',{d:linePath(p1),class:'ref'},F.g);el('path',{d:linePath(pb),class:'acc',style:'stroke-width:2.5'},F.g);
  markOut(F,pb.map(p=>p[0]),r.Rb.out.map(cur),1e-3,1e3);
  $('#c3-lg').textContent='chunk = '+b;
  const ea96=r.Rb.out[95]?r.Rb.out[95].EA:NaN,fmtBig=v=>!Number.isFinite(v)?'—':(v>=100?v.toFixed(2):v.toFixed(4));
  $('#c3-ro').innerHTML=`順序步數 ⌈96/b⌉ = <b>${Math.ceil(T/b)}</b><br>兩形式最大差 <b>${r.md<1e-12?'&lt; 1e-12，兩者相同':r.md.toExponential(1)}</b><br>E_A(96) = <b class="acc">${fmtBig(ea96)}</b>${ea96>1e3?'（超出上限）':''}<br>‖𝓜⁽ᵇ⁾₉₆ − 𝓜⁽¹⁾₉₆‖_F = <b>${fmtBig(r.dev)}</b>`;
  $('#c3-seedn').textContent=r.seed===11?'資料流與 C1 相同（seed 11）。':`資料流與 C1 相同（目前 seed ${r.seed}；內文數字依 seed 11）。`;
  const summ=`b = ${b}，η = ${c3.eta}：E_A(96) ${fmtBig(ea96)}，偏離 ${fmtBig(r.dev)}`;c3.summary=summ;
  $('#c3-fig svg').setAttribute('aria-label','目前情境的回想誤差。'+summ+'。逐點值見資料表。');
  $('#c3-th').textContent='chunk = '+b;$('#c3-cap').textContent=`每 8 步一列，目前情境的回想誤差；${summ}`;
  $('#c3-tab tbody').innerHTML=Array.from({length:12},(_,i)=>{const t=8*(i+1),a=r.R1.out[t-1],c=r.Rb.out[t-1];return`<tr><td class="num">${t}</td><td class="num">${a?fmtBig(cur(a)):'—'}</td><td class="num">${c?fmtBig(cur(c)):'—'}</td></tr>`}).join('');
}
function initC3(){$$('#c3-b button').forEach(x=>x.addEventListener('click',()=>{c3.b=+x.dataset.v;renderC3();announce('c3',c3.summary)}));
  $$('#c3-eta button').forEach(x=>x.addEventListener('click',()=>{c3.eta=+x.dataset.v;renderC3();announce('c3',c3.summary)}));
  toggleTable($('#c3-tab-b'),$('#c3-tab'));renderC3()}

/* ---------- C4 ---------- */
const OPT=[2,4,8];
const c4={v:'MAC',t:10,C:4,w:4,Np:2,Nl:2,N:16};
const FLOW={MAC:'MAC：讀記憶 → 片段注意力 → 以 yₜ 寫記憶。',MAG:'MAG：記憶分支與帶前綴的滑動視窗注意力並行 → 閘控結合。',MAL:'MAL：記憶層 → 不帶前綴的滑動視窗注意力。'};
const CATN={direct:'直接',memory:'經由記憶',none:'不可見'};
function c4opts(t){return{N:c4.N,Np:c4.Np,C:c4.C,Nl:c4.Nl,w:c4.w,t}}
function renderC4(focus){
  $$('#c4-v button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===c4.v)));
  $('#c4-to').textContent=c4.t;$('#c4-Co').textContent=c4.C;$('#c4-wo').textContent=c4.w;$('#c4-Npo').textContent=c4.Np;$('#c4-Nlo').textContent=c4.Nl;
  const mac=c4.v==='MAC';$('#c4-C').disabled=!mac;$('#c4-Nl').disabled=!mac;$('#c4-w').disabled=mac;
  $('#c4-note').textContent=mac?'視窗 w 只用於 MAG／MAL。':'片段長度 C 與記憶 token N_l 只用於 MAC。';
  const nl=mac?c4.Nl:0,cols=c4.Np+nl+c4.N,wrap=$('#c4-wrap'),gaps=(c4.Np?6:0)+(nl?6:0),avail=(wrap.clientWidth||600)-40-gaps-4;
  const cw=Math.max(14,Math.min(28,Math.floor(avail/cols)));const mx=$('#c4-mx');mx.style.setProperty('--cw',cw+'px');
  const grp=(n,l)=>n?`<span style="width:${n*cw}px" title="${l}">${l}</span>`:'';
  let h=`<div class="mx-groups" aria-hidden="true"><span class="mx-rl" style="border:0"></span>${grp(c4.Np,'持久記憶')}${c4.Np?'<i class="mx-gap"></i>':''}${grp(nl,'記憶 token h（本片段）')}${nl?'<i class="mx-gap"></i>':''}${grp(c4.N,'序列')}</div>`;
  for(let t=1;t<=c4.N;t++){const r=maskRow(c4.v,c4opts(t)),cur=t===c4.t,cn=counts(c4.v,c4opts(t));
    const cell=(cat,lab)=>`<span class="mx-c ${cat}" role="gridcell"${cur?` aria-label="${lab}：${CATN[cat]}"`:' aria-hidden="true"'}></span>`;
    h+=`<div class="mx-row${cur?' cur':''}" role="row" data-t="${t}" tabindex="${cur?0:-1}" aria-label="query ${t}：直接 ${cn.direct}、經由記憶 ${cn.memory}、不可見 ${cn.none}"><span class="mx-rl" aria-hidden="true">${t}</span>`;
    h+=r.persist.map((c,i)=>cell(c,'持久記憶 p'+(i+1))).join('')+(c4.Np?'<i class="mx-gap"></i>':'');
    h+=r.mem.map((c,i)=>cell(c,'記憶 token h'+(i+1))).join('')+(nl?'<i class="mx-gap"></i>':'');
    h+=r.seq.map((c,j)=>cell(c,'key '+(j+1))).join('')+'</div>'}
  mx.innerHTML=h;
  const cn=counts(c4.v,c4opts(c4.t)),o={N:c4.N,Np:c4.Np,C:c4.C,Nl:c4.Nl,w:c4.w},pr=pairs(c4.v,o);let full=0;for(let t=1;t<=c4.N;t++)full+=c4.Np+t;
  $('#c4-ro').innerHTML=`query t = <b>${c4.t}</b><br>直接 <b class="acc">${cn.direct}</b>　經由記憶 <b>${cn.memory}</b>　不可見 <b>${cn.none}</b><br>配對數 <b>${pr}</b>／完整因果＋前綴 <b>${full}</b>（${(pr/full).toFixed(2)}）`;
  $('#c4-flow').textContent=FLOW[c4.v];
  c4.summary=`${c4.v}，query ${c4.t}：直接 ${cn.direct}、經由記憶 ${cn.memory}、不可見 ${cn.none}；配對數 ${pr}，完整注意力 ${full}`;
  if(focus){const row=$(`.mx-row[data-t="${c4.t}"]`,mx);row&&row.focus()}
}
function c4setT(t,focus){c4.t=Math.max(1,Math.min(c4.N,t));$('#c4-t').value=c4.t;renderC4(focus);announce('c4',c4.summary)}
function initC4(){
  $$('#c4-v button').forEach(b=>b.addEventListener('click',()=>{c4.v=b.dataset.v;renderC4();announce('c4',c4.summary)}));
  $('#c4-t').addEventListener('input',e=>{c4.t=+e.target.value;renderC4()});$('#c4-t').addEventListener('change',()=>announce('c4',c4.summary));
  [['#c4-C','C',1],['#c4-w','w',1],['#c4-Np','Np',0],['#c4-Nl','Nl',0]].forEach(([id,k,opt])=>{$(id).addEventListener('input',e=>{c4[k]=opt?OPT[+e.target.value]:+e.target.value;renderC4()});$(id).addEventListener('change',()=>announce('c4',c4.summary))});
  $('#c4-C').setAttribute('aria-valuetext','C = 4');
  const mx=$('#c4-mx');
  mx.addEventListener('click',e=>{const row=e.target.closest('.mx-row');if(row)c4setT(+row.dataset.t,true)});
  mx.addEventListener('keydown',e=>{const m={ArrowDown:c4.t+1,ArrowUp:c4.t-1,Home:1,End:c4.N}[e.key];if(m!==undefined){e.preventDefault();c4setT(m,true)}});
  ['#c4-C','#c4-w'].forEach(id=>$(id).addEventListener('input',e=>e.target.setAttribute('aria-valuetext',(id==='#c4-C'?'C = ':'w = ')+OPT[+e.target.value])));
  $('#c4-w').setAttribute('aria-valuetext','w = 4');
  renderC4()}

/* ---------- timeline ---------- */
const TL_GROUP={'1410.5401':['ext','recurrent-memory-segment-transformers'],'1901.02860':['ext','recurrent-memory-segment-transformers'],'2207.06881':['ext','recurrent-memory-segment-transformers'],'2412.09764':['ext','recurrent-memory-segment-transformers'],
 '1706.03762':['attn','memory-view-of-sequence-models'],'2006.16236':['lin','linear-attention-fast-weights'],'2102.11174':['lin','linear-attention-fast-weights'],
 '2312.00752':['gate','gated-linear-recurrence-forgetting'],'2402.19427':['gate','gated-linear-recurrence-forgetting'],
 '2406.06484':['delta','delta-rule-deltanet-family'],'2412.06464':['delta','delta-rule-deltanet-family'],'2407.04620':['ttt','ttt-linear-mlp'],'2501.00663':['me','titans-neural-memory-surprise-update']};
const TL_NAME={ext:'外部與分段記憶',attn:'注意力',lin:'線性注意力與快速權重',gate:'閘控線性遞迴',delta:'Delta rule',ttt:'梯度式更新',me:'本文 · 動量＋遺忘＋深層記憶',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.map((e,i)=>({e,i})).sort((a,b)=>(+a.e.year||0)-(+b.e.year||0)||a.i-b.i).map(o=>o.e);
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];used.add(g[0]);const slug=g[1];const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${slug?`<a href="${slug}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
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

/* ---------- self test (acceptance tables, design.md §5 / §6.2) ---------- */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t)=>Math.abs(a-b)<=t+1e-12,rel=(a,b,r)=>Math.abs(a-b)<=r*Math.abs(b);
  // C1
  const d=runC1();ok(near(d.out[63].EB,.0084,5e-4),'C1 EB64 '+d.out[63].EB);ok(near(d.out[63].EA,.4571,5e-4),'C1 EA64 '+d.out[63].EA);ok(near(d.out[39].S,.1518,5e-4),'C1 S40');ok(near(d.out[0].surprise,1.0973,5e-4),'C1 g1');
  ok(runC1({theta:1,eta:.95,alpha:0}).diverged===71,'C1 div 71');
  ok(near(runC1({theta:.75,eta:0}).out[63].EB,.0106,5e-4),'C1 step-size control EB64');
  const c1t=[[{theta:.3,eta:0,alpha:0},[.7419,.6049,.0160,.0182,.2927,.0118,.1015]],[{theta:.3,eta:0,alpha:.02},[.6848,.6949,.0306,.0251,.4167,.0333,.0917]],[{theta:.3,eta:.6,alpha:0},[.6627,.5924,.0085,.0051,.3872,.0109,.1573]],[{theta:.3,eta:.6,alpha:.1},[.5875,.6958,.0498,.0494,.5816,.0787,.1482]],[{theta:.3,eta:.9,alpha:.02},[.7400,.9681,.2126,.4188,.9215,2.6196,.4433]]];
  c1t.forEach(([p,v])=>{const o=runC1(p).out,ms=(a,b)=>o.slice(a-1,b).reduce((s,x)=>s+x.surprise,0)/(b-a+1);const got=[ms(33,40),ms(65,72),o[31].EA,o[63].EB,o[63].EA,o[95].EA,o[39].S];got.forEach((g,i)=>ok(near(g,v[i],5e-4),`C1 ${JSON.stringify(p)} col${i} ${g}`))});
  // C3
  [[.6],[0]].forEach(([eta])=>{const A=loopEnds({eta,b:4}),B=matmulEnds({eta,b:4});let m=0;A.forEach((M,i)=>M.forEach((r,a)=>r.forEach((x,c)=>{m=Math.max(m,Math.abs(x-B[i][a][c]))})));ok(m<1e-12,'C3 maxdiff eta='+eta+' '+m)});
  ok(near(runC1({chunk:8}).out[95].EA,.6,5e-4),'C3 b8 EA96');
  [[.6,4,.1420,.7140],[0,8,.0500,.2136],[.6,16,8880.80,201.86]].forEach(([eta,b,ea,dv])=>{const p={theta:.3,eta,alpha:.02},R=runC1({...p,chunk:b}),f1=runC1({...p,chunk:1}).M.flat(),dev=Math.hypot(...R.M.flat().map((x,i)=>x-f1[i]));
    ok(ea>100?rel(R.out[95].EA,ea,1e-3):near(R.out[95].EA,ea,5e-4),`C3 eta${eta} b${b} EA96 ${R.out[95].EA}`);ok(dv>100?rel(dev,dv,1e-3):near(dev,dv,5e-4),`C3 eta${eta} b${b} dev ${dev}`)});
  // C2
  ok(rel(runDeep({L:1}).res[3000],.45461,.01),'C2 L1 3000');ok(near(runDeep({L:3}).res[3000],.0001,2e-5),'C2 L3 3000');ok(rel(runDeep({L:2}).res[1000],.05388,.01),'C2 L2 1000');
  ok(runDeep({L:2,theta:.05,eta:.95}).div===103,'C2 div 103');ok(runDeep({L:2,theta:.05,eta:.95,alpha:.01}).div===2082,'C2 div 2082');ok(runDeep({L:4,theta:.05,eta:.95}).div===58,'C2 div L4 58');
  {let sk=0,kk=0;for(let i=0;i<201;i++){const k=-2+4*i/200;sk+=k*target(k);kk+=k*k}const w=sk/kk;ok(near(w,.1660,1e-4),'C2 ref w '+w);
    ok(near(evalMSE([{W:[[w]],b:null}]),.4018,1e-4),'C2 ref best-linear mse');ok(near(evalMSE([{W:[[0]],b:null}]),.4389,1e-4),'C2 ref zero mse');
    ok(near(REF_LIN,.4018,1e-4)&&near(REF_ZERO,.4389,1e-4),'C2 plotted ref lines')}
  // scales and NaN-free paths
  const sl=scaleLog(1e-3,1e3,200,0);ok([0,-1,NaN].every(v=>Number.isFinite(sl(v))&&sl(v)===200),'scaleLog floor');
  {const sv=[c1.theta,c1.eta,c1.alpha];c1.theta=0;renderC1();const bad=$$('#c1 path').some(p=>/NaN|Infinity/.test(p.getAttribute('d')||''));ok(!bad,'C1 theta=0 paths finite');[c1.theta,c1.eta,c1.alpha]=sv;renderC1()}
  // C4
  const o={N:16,Np:2,C:4,Nl:2,w:4};
  [['MAC',10,[6,8,6]],['MAG',10,[6,6,6]],['MAL',10,[4,8,6]],['MAC',4,[8,0,12]],['MAG',4,[6,0,12]],['MAL',4,[4,2,12]]].forEach(([v,t,e])=>{const c=counts(v,{...o,t});ok(c.direct===e[0]&&c.memory===e[1]&&c.none===e[2],`C4 ${v} t=${t}`)});
  ok(pairs('MAC',o)===104&&pairs('MAG',o)===90&&pairs('MAL',o)===58,'C4 pairs');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ---------- boot ---------- */
initTheme();initToc();heroAxis();initC1();initC2();initC3();initC4();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{heroAxis();renderC1();renderC2();renderC3();renderC4()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))selfTest();
})();
