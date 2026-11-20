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
const f4=v=>(Math.abs(v)<5e-5?0:v).toFixed(4),f3=v=>(Math.abs(v)<5e-4?0:v).toFixed(3),f1=v=>v.toFixed(1);
const pct=v=>(100*v).toFixed(1)+'%';
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function segInit(box,opts,cur,label,onPick){box.textContent='';opts.forEach(o=>{const b=document.createElement('button');b.type='button';b.setAttribute('role','radio');
    b.textContent=label?label(o):String(o);b.dataset.v=String(o);box.appendChild(b)});
  const sync=v=>$$('button',box).forEach(b=>{const on=b.dataset.v===String(v);b.setAttribute('aria-checked',String(on));b.setAttribute('aria-pressed',String(on));b.tabIndex=on?0:-1});sync(cur);
  box.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;const v=opts.find(o=>String(o)===b.dataset.v);sync(v);onPick(v)});
  box.addEventListener('keydown',e=>{const bs=$$('button',box),i=bs.findIndex(b=>b.tabIndex===0);let j=i;if(e.key==='ArrowRight'||e.key==='ArrowDown')j=(i+1)%bs.length;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')j=(i-1+bs.length)%bs.length;else return;e.preventDefault();bs[j].focus();bs[j].click()});
  return sync}

/* ================= shared math (mirrors verify/core.py) ================= */
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}
const K=100;
function schedule(Kn){const s=.008,f=i=>Math.cos((i/Kn+s)/(1+s)*Math.PI/2)**2;const beta=new Float64Array(Kn+1),abar=new Float64Array(Kn+1),bt=new Float64Array(Kn+1);abar[0]=1;
  for(let i=1;i<=Kn;i++){beta[i]=Math.min(1-f(i)/f(i-1),.999);abar[i]=abar[i-1]*(1-beta[i])}
  for(let i=1;i<=Kn;i++)bt[i]=beta[i]*(1-abar[i-1])/(1-abar[i]);return{beta,abar,bt}}   // core.cosine_schedule
const SCH=schedule(K);
function erf(x){const s=x<0?-1:1;x=Math.abs(x);const t=1/(1+.3275911*x);const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-x*x);return s*y}
const Phi=z=>.5*(1+erf(z/Math.SQRT2));

/* ---------------- C1 toy (core.c1_*) ---------------- */
const C1D={m:[.25,.55,.85],pi:[.5,.35,.15],s:.07,sy:.25,top:.7};
const NP1=2000;
let Z1=null;function draws1(){if(Z1)return Z1;const r=mulberry32(1);Z1=new Float64Array(K*NP1);for(let i=0;i<K*NP1;i++)Z1[i]=gauss(r);return Z1}   // core.draws_1d (step-major)
function compsUncond(){return C1D.m.map((m,j)=>[C1D.pi[j],m,C1D.s*C1D.s])}
function compsTilt(y,omega){if(omega<=0)return compsUncond();const s2=C1D.s*C1D.s,e2=C1D.sy*C1D.sy/omega,v=s2*e2/(s2+e2);
  const lw=C1D.m.map((m,j)=>Math.log(C1D.pi[j])-.5*Math.log(s2+e2)-((y-m)**2)/(2*(s2+e2)));const mx=Math.max(...lw);const w=lw.map(l=>Math.exp(l-mx));const Z=w.reduce((a,b)=>a+b,0);
  return C1D.m.map((m,j)=>[w[j]/Z,(m*e2+y*s2)/(s2+e2),v])}
// prepared per-step constants for eps of a mixture: arrays of {lw, mean, V}
function prepComps(comps,k){const ab=SCH.abar[k],sa=Math.sqrt(ab);return comps.filter(c=>c[0]>=1e-300).map(([w,m,v])=>{const V=ab*v+1-ab;return[Math.log(w)-.5*Math.log(V),sa*m,V]})}
function epsMix(x,k,P){let mx=-Infinity;const L=new Array(P.length);for(let j=0;j<P.length;j++){const d=x-P[j][1];L[j]=P[j][0]-d*d/(2*P[j][2]);if(L[j]>mx)mx=L[j]}
  let Z=0,S=0;for(let j=0;j<P.length;j++){const r=Math.exp(L[j]-mx);Z+=r;S+=r*(P[j][1]-x)/P[j][2]}return -Math.sqrt(1-SCH.abar[k])*S/Z}
function c1Sample(omega,alpha,y,n){const Z=draws1(),{beta,abar,bt}=SCH,x=new Float64Array(n),sq=Math.sqrt(alpha);for(let i=0;i<n;i++)x[i]=Z[i]*sq;
  const cu=compsUncond(),cc=compsTilt(y,1);let j=1;
  for(let k=K;k>=1;k--){const Pu=prepComps(cu,k),Pc=prepComps(cc,k),c1=beta[k]/Math.sqrt(1-abar[k]),c2=1/Math.sqrt(1-beta[k]),ns=k>1?Math.sqrt(alpha*bt[k]):0,off=j*NP1;
    for(let i=0;i<n;i++){const eu=epsMix(x[i],k,Pu),ec=epsMix(x[i],k,Pc),e=eu+omega*(ec-eu);let mu=(x[i]-c1*e)*c2;if(k>1)mu+=ns*Z[off+i];x[i]=mu}
    if(k>1)j++}
  return x}
function mixStats(comps,thr=C1D.top){let m=0,s2=0,fr=0;comps.forEach(([w,mu,v])=>{m+=w*mu;s2+=w*(v+mu*mu);fr+=w*(1-Phi((thr-mu)/Math.sqrt(v)))});return{mean:m,sd:Math.sqrt(Math.max(s2-m*m,0)),frac:fr}}
function sampleStats(x){const n=x.length;let m=0;for(const v of x)m+=v;m/=n;let s=0,c=0;for(const v of x){s+=(v-m)**2;if(v>C1D.top)c++}return{mean:m,sd:Math.sqrt(s/n),frac:c/n,finite:x.every(Number.isFinite)}}
const mixPdf=(comps,x)=>comps.reduce((a,[w,mu,v])=>a+w*Math.exp(-((x-mu)**2)/(2*v))/Math.sqrt(2*Math.PI*v),0);

/* ---------------- C2 toy (core.c2_*) ---------------- */
const NP2=500,ND=200,H2=.05,ALPHA2=.5;
function diskPts(){const r=mulberry32(11),P=new Float64Array(2*ND);for(let n=0;n<ND;n++){const u1=r(),u2=r(),rho=Math.sqrt(u1),th=2*Math.PI*u2;P[2*n]=rho*Math.cos(th);P[2*n+1]=rho*Math.sin(th)}return P}
const ANN_U=(()=>{const r=mulberry32(12),U=new Float64Array(2*ND);for(let n=0;n<ND;n++){U[2*n]=r();U[2*n+1]=r()}return U})();
function annPts(rin){const P=new Float64Array(2*ND);for(let n=0;n<ND;n++){const rho=Math.sqrt(rin*rin+ANN_U[2*n]*(1.5*1.5-rin*rin)),th=2*Math.PI*ANN_U[2*n+1];P[2*n]=rho*Math.cos(th);P[2*n+1]=rho*Math.sin(th)}return P}
const DISK=diskPts();
let Z2=null;function draws2(){if(Z2)return Z2;const r=mulberry32(21);Z2=new Float64Array(K*NP2*2);for(let i=0;i<Z2.length;i++)Z2[i]=gauss(r);return Z2}
// set score + log density at (x,y): returns [sx, sy, l] with l = logsumexp - log n - log(2 pi V)   (core.set_score / set_logdens)
function setScore(x,y,C,sa,V,out){let mx=-Infinity;for(let n=0;n<ND;n++){const dx=x-sa*C[2*n],dy=y-sa*C[2*n+1],L=-(dx*dx+dy*dy)/(2*V);if(L>mx)mx=L}
  let Z=0,mxs=0,mys=0;for(let n=0;n<ND;n++){const cx=sa*C[2*n],cy=sa*C[2*n+1],dx=x-cx,dy=y-cy,w=Math.exp(-(dx*dx+dy*dy)/(2*V)-mx);Z+=w;mxs+=w*cx;mys+=w*cy}
  out[0]=(mxs/Z-x)/V;out[1]=(mys/Z-y)/V;out[2]=mx+Math.log(Z/ND)-Math.log(2*Math.PI*V)}
function g1At(x,y,rin,k){const ab=SCH.abar[k],sa=Math.sqrt(ab),V=ab*H2*H2+1-ab,a=[0,0,0],b=[0,0,0];setScore(x,y,DISK,sa,V,a);setScore(x,y,annPts(rin),sa,V,b);const m=Math.max(a[2],b[2]);const e1=Math.exp(a[2]-m),e2=Math.exp(b[2]-m);return e1/(e1+e2)}
function c2Init(alpha=ALPHA2){const Z=draws2(),X=new Float64Array(2*NP2),sq=Math.sqrt(alpha);for(let i=0;i<2*NP2;i++)X[i]=Z[i]*sq;return X}
function c2Step(X,k,mode,omega,C2,alpha=ALPHA2){const{beta,abar,bt}=SCH,ab=abar[k],sa=Math.sqrt(ab),V=ab*H2*H2+1-ab,f=-Math.sqrt(1-ab),c1=beta[k]/Math.sqrt(1-ab),c2=1/Math.sqrt(1-beta[k]),ns=k>1?Math.sqrt(alpha*bt[k]):0,Z=draws2(),off=(K-k+1)*NP2*2;
  const a=[0,0,0],b=[0,0,0];
  for(let i=0;i<NP2;i++){const x=X[2*i],y=X[2*i+1];setScore(x,y,DISK,sa,V,a);setScore(x,y,C2,sa,V,b);
    const m=Math.max(a[2],b[2]),e1=Math.exp(a[2]-m),e2=Math.exp(b[2]-m),g=e1/(e1+e2);
    for(let d=0;d<2;d++){const su=g*a[d]+(1-g)*b[d],eu=f*su,ea=f*a[d],eb=f*b[d];let e;
      if(mode==='y1')e=eu+omega*(ea-eu);else if(mode==='y2')e=eu+omega*(eb-eu);else if(mode==='and')e=eu+omega*((ea-eu)+(eb-eu));else e=eu+omega*((ea-eu)-(eb-eu));
      let mu=((d?y:x)-c1*e)*c2;if(k>1)mu+=ns*Z[off+2*i+d];X[2*i+d]=mu}}}
function c2Rates(X,rin){let le=0,ge=0,both=0,lt=0,sum=0,fin=true;for(let i=0;i<NP2;i++){const r=Math.hypot(X[2*i],X[2*i+1]);if(!Number.isFinite(r))fin=false;sum+=r;if(r<=1)le++;if(r>=rin)ge++;if(r>=rin&&r<=1)both++;if(r<rin)lt++}
  return{le_R:le/NP2,ge_r:ge/NP2,both:both/NP2,lt_r:lt/NP2,rho_mean:sum/NP2,finite:fin}}
function c2RunSync(mode,omega,rin){const X=c2Init(),C2=annPts(rin);for(let k=K;k>=1;k--)c2Step(X,k,mode,omega,C2);return X}

/* ---------------- C3 data (verify/tables.py, verbatim) ---------------- */
const METHODS=['BC','CQL','IQL','DT','TT','MOReL','Diffuser','DD'];
const T1=[['Med-Expert','HalfCheetah',[55.2,91.6,86.7,86.8,95,53.3,79.8,90.6],1.3],['Med-Expert','Hopper',[52.5,105.4,91.5,107.6,110.0,108.7,107.2,111.8],1.8],['Med-Expert','Walker2d',[107.5,108.8,109.6,108.1,101.9,95.6,108.4,108.8],1.7],
 ['Medium','HalfCheetah',[42.6,44.0,47.4,42.6,46.9,42.1,44.2,49.1],1.0],['Medium','Hopper',[52.9,58.5,66.3,67.6,61.1,95.4,58.5,79.3],3.6],['Medium','Walker2d',[75.3,72.5,78.3,74.0,79,77.8,79.7,82.5],1.4],
 ['Med-Replay','HalfCheetah',[36.6,45.5,44.2,36.6,41.9,40.2,42.2,39.3],4.1],['Med-Replay','Hopper',[18.1,95,94.7,82.7,91.5,93.6,96.8,100],0.7],['Med-Replay','Walker2d',[26.0,77.2,73.9,66.6,82.6,49.8,61.2,75],4.3],
 ['Mixed','Kitchen',[51.5,52.4,51,null,null,null,null,65],2.8],['Partial','Kitchen',[38,50.1,46.3,null,null,null,null,57],2.5]];
const T1_AVG=[51.9,77.6,77,74.7,78.9,72.9,75.3,81.8],T1K_AVG=[44.8,51.2,48.7,null,null,null,null,61];
const A3_P=[0,.05,.10,.15],A3={BC:[[55.2,1.5],[49.3,3.6],[25.8,3.8],[15.1,4.3]],DD:[[84.7,2.2],[77.3,3.1],[53.2,4.1],[41.3,4.9]],Diffuser:[[72.4,1.4],[63.2,2.9],[52.3,4.6],[41.6,5.1]],CQL:[[73.2,2.3],[61.8,3.7],[51.2,4.3],[42.2,5.5]]};
function rowStats(i){const v=T1[i][2],vals=v.map((x,j)=>[x,METHODS[j]]).filter(p=>p[0]!=null),mx=Math.max(...vals.map(p=>p[0])),dd=v[7];
  return{max:mx,winners:vals.filter(p=>p[0]===mx).map(p=>p[1]),rank:1+vals.filter(p=>p[0]>dd).length,denom:vals.length,higher:vals.filter(p=>p[0]>dd)}}
function colAvg(){const loco=METHODS.map((m,j)=>T1.slice(0,9).reduce((a,r)=>a+r[2][j],0)/9),kit=METHODS.map((m,j)=>T1[9][2][j]==null?null:(T1[9][2][j]+T1[10][2][j])/2);return{loco,kit}}
const ddBest=()=>T1.reduce((a,r,i)=>a+(rowStats(i).rank===1?1:0),0),ddBestLoco=()=>T1.slice(0,9).reduce((a,r,i)=>a+(rowStats(i).rank===1?1:0),0);
function a3Stats(pi){const dd=A3.DD[pi],o=[['Diffuser',...A3.Diffuser[pi]],['CQL',...A3.CQL[pi]]].sort((a,b)=>b[1]-a[1])[0];return{dd:dd[0],name:o[0],other:o[1],diff:dd[0]-o[1],overlap:(dd[0]-dd[1])<=(o[1]+o[2])}}

/* ================= S4 static score figure ================= */
function renderS4(){const box=$('#s4-score'),svg=$('svg',box),W=widthOf(box,300,640),H=170;svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const X=lin(-3,3,24,W-24),Y=lin(0,.42,H-34,10),pts=[];for(let i=0;i<=120;i++){const x=-3+6*i/120;pts.push([X(x),Y(Math.exp(-x*x/2)/Math.sqrt(2*Math.PI))])}
  el('line',{x1:X(-3),x2:X(3),y1:Y(0),y2:Y(0),class:'axl'},svg);el('path',{d:lp(pts),class:'ink'},svg);
  [-2.5,-1.5,-.5,.5,1.5,2.5].forEach(x=>{const y0=Y(0)-14,len=x*(X(1)-X(0))*.42;const x1=X(x),x2=x1-len;el('line',{x1,x2,y1:y0,y2:y0,class:'acc'},svg);const dir=Math.sign(-x);el('path',{d:`M${x2} ${y0}l${-dir*7} -4v8z`,class:'acc-f'},svg);el('circle',{cx:x1,cy:y0,r:2.5,class:'ink-f'},svg)});
  [-3,-2,-1,0,1,2,3].forEach(x=>tx(svg,X(x),H-14,String(x).replace('-','−'),{class:'tk','text-anchor':'middle'}));
  tx(svg,X(.3),Y(.4)+4,'N(0, 1) 的密度',{class:'al'});tx(svg,X(-3),Y(0)-26,'分數 = −x：指向密度較高處',{class:'al'})}

/* ================= C1 widget ================= */
const c1={omega:1.4,y:1,alpha:.5,busy:false,cache:{a:null,b:null}};
function c1Draw(which,preview){const box=$(`#c1${which}-fig`),svg=$('svg',box),alpha=which==='a'?1:c1.alpha,n=preview?500:NP1;
  const x=c1Sample(c1.omega,alpha,c1.y,n),st=sampleStats(x),tgt=compsTilt(c1.y,c1.omega),ts=mixStats(tgt),dat=compsUncond();
  const W=widthOf(box,280,620),H=220,L=34,R=10,T=12,B=30;svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const NB=64,lo=-.2,hi=1.4,bw=(hi-lo)/NB,cnt=new Float64Array(NB);if(st.finite)for(const v of x){const b=Math.floor((v-lo)/bw);if(b>=0&&b<NB)cnt[b]++}
  const dens=Array.from(cnt,c=>c/(n*bw));let curveMax=0;for(let i=0;i<=200;i++){const xx=lo+(hi-lo)*i/200;curveMax=Math.max(curveMax,mixPdf(tgt,xx),mixPdf(dat,xx))}
  const need=Math.max(curveMax,...dens),top=[2,4,8,16,32,48].find(v=>v>=need)||48;
  const X=lin(lo,hi,L,W-R),Y=lin(0,top,H-B,T);
  el('line',{x1:L,x2:W-R,y1:Y(0),y2:Y(0),class:'axl'},svg);[0,top/2,top].forEach(v=>{tx(svg,L-6,Y(v)+4,String(v),{class:'tk','text-anchor':'end'})});
  [0,.5,1].forEach(v=>tx(svg,X(v),H-10,String(v),{class:'tk','text-anchor':'middle'}));
  for(let b=0;b<NB;b++)if(dens[b]>0)el('rect',{x:X(lo+b*bw)+.5,y:Y(dens[b]),width:Math.max(X(lo+bw)-X(lo)-1,1),height:Y(0)-Y(dens[b]),class:'hbar'},svg);
  const curve=c=>{const p=[];for(let i=0;i<=200;i++){const xx=lo+(hi-lo)*i/200;p.push([X(xx),Y(Math.min(mixPdf(c,xx),top))])}return lp(p)};
  el('path',{d:curve(dat),class:'ink3'},svg);el('path',{d:curve(tgt),class:'ref-ln'},svg);
  const vl=(v,lab,cls,dy,end)=>{el('line',{x1:X(v),x2:X(v),y1:Y(0),y2:T,class:cls},svg);if(lab)tx(svg,X(v)+(end?-4:4),T+dy,lab,{class:'tk halo','text-anchor':end?'end':'start'})};
  vl(.7,'高報酬區 →','cur',10,true);vl(.85,'資料最好的行為','cur',10);vl(c1.y,'y','axl',26);
  const rd=$(`#c1${which}-read`);const rows=[['樣本平均（期望報酬）',f4(st.mean)],['樣本標準差',f4(st.sd)],['高報酬區比例',f4(st.frac)],['目標 平均',f4(ts.mean)],['目標 標準差',f4(ts.sd)],['目標 高報酬區比例',f4(ts.frac)]];
  rd.innerHTML=rows.map(([a,b])=>`<dt>${a}</dt><dd>${b}</dd>`).join('')+(preview?'<dt>預覽</dt><dd>500 個樣本</dd>':'')+(alpha===0?'<dt>α = 0</dt><dd>2000 個樣本完全相同</dd>':'')+(!st.finite?'<dt>狀態</dt><dd>數值溢位</dd>':'');
  svg.setAttribute('aria-label',`樣本直方圖：ω = ${c1.omega.toFixed(1)}，y = ${c1.y.toFixed(2)}，α = ${alpha.toFixed(2)}，樣本平均 ${f3(st.mean)}，高報酬區比例 ${f3(st.frac)}`);
  if(!preview)announce(`c1${which}`,`ω = ${c1.omega.toFixed(1)}，α = ${alpha.toFixed(2)}，樣本平均 ${f3(st.mean)}，標準差 ${f3(st.sd)}`);
  return st}
function initC1(){const sl={aw:$('#c1a-w'),ay:$('#c1a-y'),bw:$('#c1b-w'),by:$('#c1b-y'),ba:$('#c1b-a')};
  const sync=()=>{for(const p of ['a','b']){sl[p+'w'].value=Math.round(c1.omega*10);sl[p+'y'].value=Math.round(c1.y*20);$(`#c1${p}-wo`).textContent=c1.omega.toFixed(1);$(`#c1${p}-yo`).textContent=c1.y.toFixed(2);
      sl[p+'w'].setAttribute('aria-valuetext',`ω = ${c1.omega.toFixed(1)}`);sl[p+'y'].setAttribute('aria-valuetext',`y = ${c1.y.toFixed(2)}`)}
    sl.ba.value=Math.round(c1.alpha*20);$('#c1b-ao').textContent=c1.alpha.toFixed(2);sl.ba.setAttribute('aria-valuetext',`α = ${c1.alpha.toFixed(2)}`)};
  let q=0;const preview=()=>{if(q||c1.busy)return;q=requestAnimationFrame(()=>{q=0;c1.busy=true;c1Draw('a',true);c1Draw('b',true);c1.busy=false})};
  let t=0;const full=(d=0)=>{clearTimeout(t);t=setTimeout(()=>{c1Draw('a',false);c1Draw('b',false)},d)};
  const read=()=>{};const bind=(s,fn)=>{s.addEventListener('input',()=>{fn();sync();preview()});s.addEventListener('change',()=>{fn();sync();full(250)})};
  bind(sl.aw,()=>c1.omega=+sl.aw.value/10);bind(sl.bw,()=>c1.omega=+sl.bw.value/10);bind(sl.ay,()=>c1.y=+sl.ay.value/20);bind(sl.by,()=>c1.y=+sl.by.value/20);bind(sl.ba,()=>c1.alpha=+sl.ba.value/20);
  $('#c1a-reset').addEventListener('click',()=>{c1.omega=1.4;c1.y=1;sync();full()});$('#c1b-reset').addEventListener('click',()=>{c1.omega=1.4;c1.y=1;c1.alpha=.5;sync();full()});
  sync();full();void read}

/* ================= C2 widget ================= */
const C2MODES=['y1','y2','and','not2'],C2NAME={y1:'只用 y¹',y2:'只用 y²',and:'AND（Eq. 9）',not2:'y¹ AND NOT y²'};
const c2={mode:'and',omega:1.2,r:.7,showData:true,token:0,X:null,stale:false,cache:new Map(),started:false,rates:null};
function c2DrawPlane(){const box=$('#c2-plane'),svg=$('svg',box),W=widthOf(box,260,520),H=W;svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const S=lin(-1.8,1.8,8,W-8),Sy=lin(-1.8,1.8,H-8,8),R=v=>S(v)-S(0);
  el('rect',{x:4,y:4,width:W-8,height:H-8,class:'frame'},svg);
  if(c2.r<=1){const p=`M${S(1)} ${Sy(0)}A${R(1)} ${R(1)} 0 1 0 ${S(-1)} ${Sy(0)}A${R(1)} ${R(1)} 0 1 0 ${S(1)} ${Sy(0)}ZM${S(c2.r)} ${Sy(0)}A${R(c2.r)} ${R(c2.r)} 0 1 1 ${S(-c2.r)} ${Sy(0)}A${R(c2.r)} ${R(c2.r)} 0 1 1 ${S(c2.r)} ${Sy(0)}Z`;el('path',{d:p,class:'ring','fill-rule':'evenodd'},svg)}
  el('circle',{cx:S(0),cy:Sy(0),r:R(1),class:'circ'},svg);el('circle',{cx:S(0),cy:Sy(0),r:R(c2.r),class:'circ dash'},svg);
  
  if(c2.showData){const A=annPts(c2.r);for(let n=0;n<ND;n++){el('circle',{cx:S(DISK[2*n]),cy:Sy(DISK[2*n+1]),r:1.8,class:'d1'},svg);const x=S(A[2*n]),y=Sy(A[2*n+1]);el('path',{d:`M${x-2.5} ${y-2.5}l5 5M${x-2.5} ${y+2.5}l5 -5`,class:'d2'},svg)}}
  const g=el('g',{class:c2.stale?'stale':''},svg);
  /*labels drawn after points*/if(c2.X)for(let i=0;i<NP2;i++){const x=c2.X[2*i],y=c2.X[2*i+1];if(Math.abs(x)<1.8&&Math.abs(y)<1.8)el('circle',{cx:S(x),cy:Sy(y),r:2.4,class:'smp'},g)}
  tx(svg,S(.72),Sy(.78),'R = 1',{class:'lab halo'});tx(svg,S(0)+R(c2.r)*.72+4,Sy(0)+R(c2.r)*.72+14,'r',{class:'lab halo'});
  tx(svg,12,22,c2.r>1?'沒有同時滿足兩個限制的位置':'淺色環：兩個限制都滿足',{class:'al'});
  svg.setAttribute('aria-label',`平面圖：${C2NAME[c2.mode]}，ω = ${c2.omega.toFixed(1)}，r = ${c2.r.toFixed(1)}`+(c2.rates?`，兩者都滿足 ${pct(c2.rates.both)}`:''))}
function c2DrawHist(){const box=$('#c2-hist'),svg=$('svg',box),W=widthOf(box,240,420),H=120;svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const X=lin(0,1.8,24,W-8);el('line',{x1:24,x2:W-8,y1:H-22,y2:H-22,class:'axl'},svg);[0,.5,1,1.5].forEach(v=>tx(svg,X(v),H-6,String(v),{class:'tk','text-anchor':'middle'}));
  if(c2.X){const nb=36,c=new Array(nb).fill(0);for(let i=0;i<NP2;i++){const r=Math.hypot(c2.X[2*i],c2.X[2*i+1]),b=Math.floor(r/1.8*nb);if(b>=0&&b<nb)c[b]++}const mx=Math.max(1,...c),Y=lin(0,mx,H-22,8),g=el('g',{class:c2.stale?'stale':''},svg);
    c.forEach((v,b)=>{if(v)el('rect',{x:X(b*1.8/nb)+.5,y:Y(v),width:X(1.8/nb)-X(0)-1,height:H-22-Y(v),class:'hbar'},g)})}
  [[c2.r,'r'],[1,'1']].forEach(([v,l])=>{el('line',{x1:X(v),x2:X(v),y1:8,y2:H-22,class:'cur'},svg);tx(svg,X(v)+3,16,l,{class:'tk'})});tx(svg,W-8,16,'樣本半徑 ρ',{class:'tk','text-anchor':'end'})}
function c2Read(){const q=c2.rates,rd=$('#c2-read');if(!q){rd.innerHTML='<dt>狀態</dt><dd>尚未計算</dd>';return}
  const tg={y1:'le',y2:'ge',and:'both',not2:'lt'}[c2.mode];const row=(k,a,b)=>`<dt class="${k===tg?'tgt':''}">${a}</dt><dd class="${k===tg?'tgt':''}">${b}</dd>`;
  rd.innerHTML=row('le','‖s‖ ≤ 1',pct(q.le_R))+row('ge','‖s‖ ≥ r',pct(q.ge_r))+row('both','兩者都滿足',pct(q.both))+row('lt','‖s‖ < r',pct(q.lt_r))+row('m','平均半徑',f3(q.rho_mean))+(c2.stale?'<dt>狀態</dt><dd>放開後重算</dd>':'')+(!q.finite?'<dt>狀態</dt><dd>數值溢位</dd>':'');rd.classList.toggle('stale',c2.stale)}
function c2Render(){c2DrawPlane();c2DrawHist();c2Read()}
function c2Compute(){const key=`${c2.mode}|${c2.omega.toFixed(1)}|${c2.r.toFixed(1)}`,tok=++c2.token;c2.stale=false;
  if(c2.cache.has(key)){c2.X=c2.cache.get(key);c2.rates=c2Rates(c2.X,c2.r);$('#c2-prog').textContent='';c2Render();c2Announce();return}
  const X=c2Init(),C2=annPts(c2.r),mode=c2.mode,om=c2.omega,r=c2.r;let k=K;c2.stale=true;c2Render();
  const slice=()=>{if(tok!==c2.token)return;c2Step(X,k,mode,om,C2);k--;
    if(!RM)$('#c2-prog').textContent=`計算中 k = ${k}`;
    if(k>=1){setTimeout(slice,0);return}
    c2.cache.set(key,X);c2.X=X;c2.rates=c2Rates(X,r);c2.stale=false;$('#c2-prog').textContent='';c2Render();c2Announce()};
  if(RM)$('#c2-prog').textContent='計算中';setTimeout(slice,0)}
function c2Announce(){const q=c2.rates;if(!q)return;announce('c2',c2.r>1?`${C2NAME[c2.mode]}：沒有同時滿足兩個限制的位置；兩者都滿足 ${pct(q.both)}`:`${C2NAME[c2.mode]}，兩個限制都滿足 ${pct(q.both)}，‖s‖ < r 為 ${pct(q.lt_r)}`)}
function initC2(){segInit($('#c2-mode'),C2MODES,c2.mode,m=>C2NAME[m],m=>{c2.mode=m;c2Compute()});
  const w=$('#c2-w'),r=$('#c2-r');const upd=()=>{c2.omega=+w.value/10;c2.r=+r.value/10;$('#c2-wo').textContent=c2.omega.toFixed(1);$('#c2-ro').textContent=c2.r.toFixed(1);w.setAttribute('aria-valuetext',`ω = ${c2.omega.toFixed(1)}`);r.setAttribute('aria-valuetext',`r = ${c2.r.toFixed(1)}`)};upd();
  let t=0;[w,r].forEach(s=>{s.addEventListener('input',()=>{upd();c2.stale=true;c2Render()});s.addEventListener('change',()=>{upd();clearTimeout(t);t=setTimeout(c2Compute,250)})});
  $('#c2-data').addEventListener('change',e=>{c2.showData=e.target.checked;c2DrawPlane()});
  c2Render();const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)&&!c2.started){c2.started=true;io.disconnect();c2Compute()}},{threshold:.25});io.observe($('#c2'))}

/* ================= C3 widget ================= */
const c3={tab:'t1',row:4,showAvg:false,pi:1};
function c3Bars(){const box=$('#c3-bars'),svg=$('svg',box),W=widthOf(box,300,760),rowH=28,top=8,avg=c3.showAvg,H=top+METHODS.length*rowH+24;svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const L=72,X=lin(0,120,L,W-70),r=T1[c3.row],s=rowStats(c3.row),ca=colAvg(),isK=c3.row>=9;
  [0,40,80,120].forEach(v=>{el('line',{x1:X(v),x2:X(v),y1:top,y2:H-20,class:'grid'},svg);tx(svg,X(v),H-6,String(v),{class:'tk','text-anchor':'middle'})});
  METHODS.forEach((m,j)=>{const y=top+j*rowH,v=r[2][j];tx(svg,L-8,y+18,m,{class:'nm','text-anchor':'end'});
    if(v==null){tx(svg,L+4,y+18,'—（未報告）',{class:'tk'});return}
    el('rect',{x:L,y:y+6,width:X(v)-L,height:16,class:j===7?'bar-acc':'bar'},svg);
    let lab=f1(v)+(j===7?` ±${r[3]}`:'')+(v===s.max?'  最高':'');tx(svg,X(v)+(j===7?r[3]*(X(1)-X(0))+6:6),y+18,lab,{class:'labm'});
    if(j===7){el('line',{x1:X(v-r[3]),x2:X(v+r[3]),y1:y+14,y2:y+14,class:'err'},svg);el('line',{x1:X(v),x2:X(v),y1:top,y2:H-20,class:'cur'},svg)}
    if(avg){const a=isK?ca.kit[j]:ca.loco[j];if(a!=null)el('circle',{cx:X(a),cy:y+14,r:4,class:'mk'},svg)}});
  svg.setAttribute('aria-label',`${r[0]} · ${r[1]}：最高 ${s.winners.join('、')} ${s.max}；DD ${r[2][7]}，名次 ${s.rank}／${s.denom}`);
  const rd=$('#c3-read1');let h=`<dt>最高</dt><dd>${s.winners.join('、')} ${s.max}</dd><dt>DD 名次</dt><dd>${s.rank}／${s.denom}</dd><dt>高於 DD</dt><dd>${s.higher.length?s.higher.map(p=>p[1]+' '+p[0]).join('、'):'無'}</dd><dt>DD 為最高的列</dt><dd>${ddBest()}／11（運動控制 ${ddBestLoco()}／9；Kitchen 列只比 3 個方法）</dd>`;
  if(avg){const a=isK?ca.kit:ca.loco,p=isK?T1K_AVG:T1_AVG;h+=`<dt>${isK?'Kitchen':'運動控制'}平均（重算 → 印出）</dt><dd></dd>`+METHODS.map((m,j)=>a[j]==null?'':`<dt>${m}</dt><dd>${a[j].toFixed(3)} → ${p[j]}${Math.abs(a[j]-p[j])<=.05?' 一致':''}</dd>`).join('');h+='<dt>圖中圓圈</dt><dd>重算的平均</dd>'}
  rd.innerHTML=h;announce('c3',`${r[0]} ${r[1]}：DD 名次 ${s.rank}／${s.denom}`)}
function c3Line(){const box=$('#c3-line'),svg=$('svg',box),W=widthOf(box,300,760),narrow=W<560,H=narrow?220:260,L=40,R=narrow?12:84,T=12,B=28;svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const X=lin(0,.15,L+10,W-R-10),Y=lin(0,100,H-B,T);
  el('rect',{x:X(A3_P[c3.pi])-14,y:T,width:28,height:H-B-T,class:'band'},svg);
  [0,25,50,75,100].forEach(v=>{el('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v),class:'grid'},svg);tx(svg,L-6,Y(v)+4,String(v),{class:'tk','text-anchor':'end'})});
  A3_P.forEach(p=>tx(svg,X(p),H-8,p.toFixed(2),{class:'tk','text-anchor':'middle'}));
  const shape={DD:'c',Diffuser:'s',CQL:'t',BC:'d'};
  ['BC','CQL','Diffuser','DD'].forEach(m=>{const pts=A3[m].map((v,i)=>[X(A3_P[i]),Y(v[0])]);el('path',{d:lp(pts),class:m==='DD'?'acc':'ink',style:m==='DD'?'':'stroke-width:1.4'},svg);
    A3[m].forEach((v,i)=>{const x=X(A3_P[i]);el('line',{x1:x,x2:x,y1:Y(v[0]-v[1]),y2:Y(v[0]+v[1]),class:'err'},svg);const y=Y(v[0]),cl=m==='DD'?'acc-f':'mk';
      if(shape[m]==='c')el('circle',{cx:x,cy:y,r:4.5,class:cl},svg);else if(shape[m]==='s')el('rect',{x:x-4,y:y-4,width:8,height:8,class:cl},svg);else if(shape[m]==='t')el('path',{d:`M${x} ${y-5}l5 9h-10z`,class:cl},svg);else el('path',{d:`M${x} ${y-5}l5 5l-5 5l-5 -5z`,class:cl},svg)});
    if(!narrow){const last=pts[3];tx(svg,last[0]+10,last[1]+(m==='CQL'?-6:m==='Diffuser'?12:4),m,{class:'lab'})}});
  $('#c3-leg').innerHTML=narrow?'<span class="lg pt">DD（圓形、粗線）</span><span class="lg sq">Diffuser（方形）</span><span class="lg">CQL（三角形）</span><span class="lg dia">BC（菱形）</span>':'';
  const a=a3Stats(c3.pi),p=A3_P[c3.pi].toFixed(2);$('#c3-po').textContent=p;$('#c3-p').setAttribute('aria-valuetext',`p = ${p}`);
  $('#c3-read2').innerHTML=`<dt>p</dt><dd>${p}</dd><dt>DD</dt><dd>${a.dd}</dd><dt>較高的對手</dt><dd>${a.name} ${a.other}</dd><dt>差</dt><dd>${a.diff>=0?'+':'−'}${Math.abs(a.diff).toFixed(1)}</dd><dt>區間</dt><dd>${a.overlap?'重疊':'不重疊'}</dd>`;
  svg.setAttribute('aria-label',`Table A3 折線圖；p = ${p} 時 DD ${a.dd}，${a.name} ${a.other}`);announce('c3',`p = ${p}：DD ${a.dd}，較高的對手 ${a.name} ${a.other}，差 ${a.diff.toFixed(1)}`)}
function c3Render(){$('#c3-t1').classList.toggle('hide',c3.tab!=='t1');$('#c3-a3').classList.toggle('hide',c3.tab!=='a3');if(c3.tab==='t1')c3Bars();else c3Line()}
function initC3(){segInit($('#c3-tab'),['t1','a3'],c3.tab,t=>t==='t1'?'Table 1':'Table A3',t=>{c3.tab=t;c3Render()});
  const sel=$('#c3-row');T1.forEach((r,i)=>sel.insertAdjacentHTML('beforeend',`<option value="${i}">${r[0]} · ${r[1]}</option>`));sel.value=c3.row;
  sel.addEventListener('change',()=>{c3.row=+sel.value;c3Bars()});
  $('#c3-prev').addEventListener('click',()=>{c3.row=(c3.row+10)%11;sel.value=c3.row;c3Bars()});$('#c3-next').addEventListener('click',()=>{c3.row=(c3.row+1)%11;sel.value=c3.row;c3Bars()});
  $('#c3-avg').addEventListener('change',e=>{c3.showAvg=e.target.checked;c3Bars()});
  $('#c3-p').addEventListener('input',e=>{c3.pi=+e.target.value;c3Line()});c3Render()}

/* ================= dual-version equations: aria-hidden on the hidden copy ================= */
function initEqVersions(){const mq=matchMedia('(max-width:599px)');const apply=()=>{$$('.wide-eq').forEach(e=>e.setAttribute('aria-hidden',String(mq.matches)));$$('.narrow-eq').forEach(e=>e.setAttribute('aria-hidden',String(!mq.matches)))};apply();mq.addEventListener('change',apply)}

/* ================= timeline ================= */
const TL_GROUP={'1503.03585':['df','ddpm-training-and-sampling-recap'],'2006.11239':['df','ddpm-training-and-sampling-recap'],'2011.13456':['df','ddpm-training-and-sampling-recap'],
 '2102.09672':['gd','classifier-free-guidance-return-conditioning'],'2207.12598':['gd','classifier-free-guidance-return-conditioning'],
 '1606.07419':['id','inverse-dynamics-state-diffusion'],'1804.08606':['id','inverse-dynamics-state-diffusion'],
 '2004.07219':['of','dd-experiments-and-ablations'],'2006.04779':['of','dd-experiments-and-ablations'],'2110.06169':['of','dd-experiments-and-ablations'],
 '1912.13465':['rc','return-conditioned-sequence-modeling'],'2106.01345':['rc','return-conditioned-sequence-modeling'],
 '2205.09991':['dp','trajectory-diffusion-planning'],'2206.01714':['cp','composing-constraints-and-skills'],'2211.15657':['me','decision-diffuser-lineage-hub']};
const TL_NAME={df:'擴散模型',gd:'引導',id:'逆動力學',of:'離線 RL 基準與基線',rc:'報酬條件策略',dp:'擴散規劃',cp:'組合式擴散',me:'本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.slice().sort((a,b)=>(Number(a.year)||0)-(Number(b.year)||0)||String(a.arxiv).localeCompare(String(b.arxiv)));
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];if(!TL_GROUP[e.arxiv])console.warn('timeline: ungrouped',e.arxiv);used.add(g[0]);const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${g[1]?`<a href="${g[1]}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
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
  const near=(a,b,t=1e-4)=>Math.abs(a-b)<=t+1e-12;
  // schedule, PRNG
  ok(near(SCH.beta[1],.000631,1e-6)&&near(SCH.beta[50],.030593,1e-6)&&near(SCH.beta[100],.999,1e-6),'beta');
  ok(near(SCH.abar[1],.999369,1e-6)&&near(SCH.abar[50],.493844,1e-6)&&near(SCH.abar[100],2.4286e-7,1e-10),'abar');
  ok(near(SCH.bt[2],.000403,1e-6)&&near(SCH.bt[50],.029651,1e-6)&&near(SCH.bt[100],.998758,1e-6),'bt');
  {const r=mulberry32(1);ok([.627073941,.002735721,.52744704].every(v=>near(r(),v,1e-9)),'mulberry32 seed 1');const g=mulberry32(1);ok(near(gauss(g),1.404339,1e-6)&&near(gauss(g),1.215755,1e-6),'gauss seed 1')}
  // C1 components
  {const c=compsTilt(1,1);ok(near(c[0][0],.0362)&&near(c[1][0],.3666)&&near(c[2][0],.5972)&&near(c[0][1],.3045)&&near(c[1][1],.5827)&&near(c[2][1],.8609)&&near(c[0][2],.004544,1e-6),'p(x|y=1)');
   const d=compsTilt(.5,1);ok(near(d[0][0],.4377)&&near(d[1][0],.4782)&&near(d[2][0],.0841),'p(x|y=0.5)')}
  // C1 table
  const C1T=[[1,0,1,.4445,.2202,.1475,.30792],[1,0,.5,.4194,.177,.0665,.30316],[1,0,0,.3384,0,0,.33839],[1,1,1,.7407,.1605,.6085,.69304],[1,1,.5,.7584,.1267,.6635,.76235],[1,1,0,.821,0,1,.82098],
    [1,1.4,1,.8102,.1229,.8065,.83431],[1,1.4,.5,.8306,.0874,.892,.84555],[1,1.4,0,.8579,0,1,.85793],[1,2,1,.8683,.0775,.9515,.87414],[1,2,.5,.8776,.0443,.9915,.87669],[1,4,1,.9173,.0449,1,.91686],[1,4,.5,.9152,.0319,1,.91527],
    [.5,1.4,.5,.4481,.1271,.01,null],[.5,1,1,.4476,.1777,.076,null],[.75,1.4,.5,.6489,.1244,.3105,.59365],[0,2,.5,.2073,.0363,0,null]];
  C1T.forEach(([y,w,a,m,s,fr,x0])=>{const x=c1Sample(w,a,y,NP1),st=sampleStats(x);ok(near(st.mean,m,5e-4)&&near(st.sd,s,5e-4)&&near(st.frac,fr,.002)&&(x0==null||near(x[0],x0,1e-4)),`C1 y=${y} w=${w} a=${a}: ${st.mean.toFixed(4)} ${st.sd.toFixed(4)} ${st.frac.toFixed(4)} ${x[0].toFixed(5)}`)});
  const EX=[[1,0,.445,.2289,.1532],[1,1,.7388,.1709,.6071],[1,1.4,.7857,.1451,.7281],[1,2,.8275,.1171,.8467],[1,4,.8794,.0719,.9775],[.5,1.4,.4526,.175,.0693],[.75,1.4,.6328,.1783,.3397],[0,2,.2225,.0766,0]];
  EX.forEach(([y,w,m,s,fr])=>{const t=mixStats(compsTilt(y,w));ok(near(t.mean,m)&&near(t.sd,s)&&near(t.frac,fr),`target y=${y} w=${w}`)});
  {const a=c1Sample(1.4,.5,1,500),b=c1Sample(1.4,.5,1,2000);let mx=0;for(let i=0;i<500;i++)mx=Math.max(mx,Math.abs(a[i]-b[i]));ok(mx<1e-12,'C1 preview = first 500 of full')}
  {const x=c1Sample(0,0,1,NP1);ok(x.every(v=>v===x[0]),'alpha=0: all samples identical')}
  // C2 data, g1, samples
  ok(near(DISK[0],-.70263,1e-5)&&near(DISK[1],-.13379,1e-5),'disk pt0');{const A=annPts(.7);ok(near(A[0],.88782,1e-5)&&near(A[1],.45707,1e-5),'annulus pt0')}
  const G1=[[100,.499998,.5,.500004],[50,.550815,.597234,.492265],[10,.678656,.999764,.189701],[1,.761053,1,.000172]];
  G1.forEach(([k,a,b,c])=>ok(near(g1At(.5,.5,.7,k),a,1e-5)&&near(g1At(0,0,.7,k),b,1e-5)&&near(g1At(1.2,0,.7,k),c,1e-5),'g1 k='+k));
  const C2T=[['and',1.2,.7,.954,.972,.926,.028,.8518,-.5093,.4846],['and',1.2,1.3,0,0,0,1,1.1308,-1.0418,.3867],['not2',1.2,.7,1,0,0,1,.0805,.0874,-.0652],['y1',1.2,.7,1,.21,.21,.79,.5194,.1716,-.3386]];
  C2T.forEach(([mode,w,r,le,ge,both,lt,rm,x0,y0])=>{const X=c2RunSync(mode,w,r),q=c2Rates(X,r);ok(near(q.le_R,le,.004)&&near(q.ge_r,ge,.004)&&near(q.both,both,.004)&&near(q.lt_r,lt,.004)&&near(q.rho_mean,rm,.002)&&near(X[0],x0,1e-3)&&near(X[1],y0,1e-3),`C2 ${mode} w=${w} r=${r}: ${q.both} ${q.rho_mean.toFixed(4)} (${X[0].toFixed(4)},${X[1].toFixed(4)})`)});
  // C3
  {const ca=colAvg(),L=[51.856,77.611,76.956,74.733,78.878,72.944,75.333,81.822];ok(L.every((v,j)=>near(ca.loco[j],v,1e-3)),'loco averages');ok(near(ca.kit[0],44.75,1e-3)&&near(ca.kit[1],51.25,1e-3)&&near(ca.kit[2],48.65,1e-3)&&near(ca.kit[7],61,1e-3),'kitchen averages')}
  ok([3,1,2,1,2,1,6,1,3,1,1].every((v,i)=>rowStats(i).rank===v),'DD ranks');ok(ddBest()===6&&ddBestLoco()===4,'DD best rows 6 / 4');
  ok(rowStats(6).rank===6&&rowStats(6).denom===8&&rowStats(9).rank===1&&rowStats(9).denom===4,'rank denominators');
  {const R=[[11.5,false],[14.1,false],[.9,true],[-.9,true]];ok(R.every(([d,o],i)=>near(a3Stats(i).diff,d,1e-3)&&a3Stats(i).overlap===o),'A3 diffs')}
  // DOM
  {const s=$('#s9-loop');ok(!!s&&s.textContent.includes('逆動力學'),'S9 loop diagram')}
  ok($$('details.check').length===13,'check boxes: '+$$('details.check').length);
  ok(!!$('#c1a')&&!!$('#c1b'),'C1a and C1b plates');
  ok($$('.wide-eq,.narrow-eq').every(e=>e.hasAttribute('aria-hidden'))&&$$('.wide-eq,.narrow-eq').some(e=>e.getAttribute('aria-hidden')==='true'),'hidden equation copies are aria-hidden');
  {const tl=JSON.parse($('#timeline-data').textContent);ok(tl.length===15&&tl.every(e=>TL_GROUP[e.arxiv]),'timeline groups')}
  try{console.log('fonts.check JetBrains Mono ωαερØ:',document.fonts.check('13px "JetBrains Mono"','ωαερØ'))}catch(e){}
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ================= boot ================= */
initTheme();initToc();initEqVersions();renderS4();initC1();initC2();initC3();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderS4();c1Draw('a',false);c1Draw('b',false);c2Render();c3Render()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))setTimeout(selfTest,0);
})();
