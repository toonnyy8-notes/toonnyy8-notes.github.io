/* ---------- UI helpers (406/570 shell) ---------- */
function radiogroup(box,val,onChange){const bs=$$('button',box);
  const set=(v,focus)=>{bs.forEach(b=>{const on=b.dataset.v===String(v);b.setAttribute('aria-checked',String(on));b.tabIndex=on?0:-1;if(on&&focus)b.focus()})};
  bs.forEach((b,i)=>{b.type='button';b.setAttribute('role','radio');b.addEventListener('click',()=>{set(b.dataset.v);onChange(b.dataset.v)});
    b.addEventListener('keydown',e=>{const d={ArrowRight:1,ArrowDown:1,ArrowLeft:-1,ArrowUp:-1}[e.key];if(!d)return;e.preventDefault();const nb=bs[(i+d+bs.length)%bs.length];set(nb.dataset.v,true);onChange(nb.dataset.v)})});
  set(val);return set}
function sched(fn){let raf=0;return()=>{if(raf)return;raf=requestAnimationFrame(()=>{raf=0;fn()})}}
const f4=v=>v.toFixed(4);
function chartW(box){return widthOf(box,260,620)}

/* ---------- toy math (design §4.0; same algorithm as verify/core.py) ---------- */
const MU=[[0,.8],[.8,0],[0,-.8],[-.8,0]],RBAR=[3,1.5,5,0],SD=.05,BMIN=.1,BMAX=10,QNORM=2.375,SIGQ=.2,NSAMP=1000;
const PEAK=['上','右','下','左'];
function rng(seed){let a=seed>>>0;return()=>{a=(a+0x6D2B79F5)>>>0;let t=Math.imul(a^(a>>>15),a|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296}}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}
const schCache=new Map();
function schedule(N){if(schCache.has(N))return schCache.get(N);const beta=new Float64Array(N),alpha=new Float64Array(N),abar=new Float64Array(N);let p=1;
  for(let i=1;i<=N;i++){const b=1-Math.exp(-BMIN/N-0.5*(BMAX-BMIN)*(2*i-1)/(N*N));beta[i-1]=b;alpha[i-1]=1-b;p*=1-b;abar[i-1]=p}
  const S={N,beta,alpha,abar};schCache.set(N,S);return S}
// ideal denoiser of sum_k softmax(z)_k N(MU_k, SD^2 I); o.Jx = [xx,xy,yx,yy], o.Jl = 2x4 row-major
const ET={lg:new Float64Array(4),r:new Float64Array(4)};
function epsStar(x,y,i,S,z,o,jac){const ab=S.abar[i-1],sa=Math.sqrt(ab),s1=Math.sqrt(1-ab),v=ab*SD*SD+1-ab,lg=ET.lg,r=ET.r;let mx=-Infinity;
  for(let k=0;k<4;k++){const dx=x-sa*MU[k][0],dy=y-sa*MU[k][1];lg[k]=z[k]-(dx*dx+dy*dy)/(2*v);if(lg[k]>mx)mx=lg[k]}
  let sum=0;for(let k=0;k<4;k++){r[k]=Math.exp(lg[k]-mx);sum+=r[k]}sum=Math.max(sum,1e-300);
  let mbx=0,mby=0;for(let k=0;k<4;k++){r[k]/=sum;mbx+=r[k]*MU[k][0];mby+=r[k]*MU[k][1]}
  const f=s1/v;o.ex=f*(x-sa*mbx);o.ey=f*(y-sa*mby);
  if(!jac)return o;
  let cxx=0,cxy=0,cyy=0;for(let k=0;k<4;k++){cxx+=r[k]*MU[k][0]*MU[k][0];cxy+=r[k]*MU[k][0]*MU[k][1];cyy+=r[k]*MU[k][1]*MU[k][1]}
  cxx-=mbx*mbx;cxy-=mbx*mby;cyy-=mby*mby;const h=ab/v;
  o.Jx[0]=f*(1-h*cxx);o.Jx[1]=-f*h*cxy;o.Jx[2]=-f*h*cxy;o.Jx[3]=f*(1-h*cyy);
  const g=-s1*sa/v;for(let j=0;j<4;j++){o.Jl[j]=g*r[j]*(MU[j][0]-mbx);o.Jl[4+j]=g*r[j]*(MU[j][1]-mby)}
  return o}
const EO={ex:0,ey:0,Jx:new Float64Array(4),Jl:new Float64Array(8)};
// Eq. (1) chain for one sample; buf holds [aN_x,aN_y, z_N(x,y), z_{N-1}(x,y), ..., z_2(x,y)] at offset off; T (8) = d a / d z
function chainOne(buf,off,S,z,T,out){let ax=buf[off],ay=buf[off+1];const N=S.N;
  for(let i=N;i>=1;i--){const c=S.beta[i-1]/Math.sqrt(1-S.abar[i-1]),sq=Math.sqrt(S.alpha[i-1]);
    epsStar(ax,ay,i,S,z,EO,!!T);
    if(T){const J=EO.Jx,L=EO.Jl;for(let j=0;j<4;j++){const tx=T[j],ty=T[4+j];T[j]=(tx-c*(J[0]*tx+J[1]*ty+L[j]))/sq;T[4+j]=(ty-c*(J[2]*tx+J[3]*ty+L[4+j]))/sq}}
    ax=(ax-c*EO.ex)/sq;ay=(ay-c*EO.ey)/sq;
    if(i>1){const sb=Math.sqrt(S.beta[i-1]),p=off+2+(N-i)*2;ax+=sb*buf[p];ay+=sb*buf[p+1]}}
  out[0]=ax;out[1]=ay;return out}
// toy Q (推定): Nadaraya-Watson smoothing of RBAR, log-sum-exp guarded
const QT=new Float64Array(4);
function qToy(x,y,o){let mx=-Infinity;for(let k=0;k<4;k++){const dx=x-MU[k][0],dy=y-MU[k][1];QT[k]=-(dx*dx+dy*dy)/(2*SIGQ*SIGQ);if(QT[k]>mx)mx=QT[k]}
  let s=0;for(let k=0;k<4;k++){QT[k]=Math.exp(QT[k]-mx);s+=QT[k]}s=Math.max(s,1e-300);let q=0;for(let k=0;k<4;k++){QT[k]/=s;q+=QT[k]*RBAR[k]}
  if(o){let gx=0,gy=0;for(let k=0;k<4;k++){const w=QT[k]*(RBAR[k]-q);gx+=w*(MU[k][0]-x);gy+=w*(MU[k][1]-y)}o.gx=gx/(SIGQ*SIGQ);o.gy=gy/(SIGQ*SIGQ)}
  return q}
// sampling (order identical to verify/c2.py)
const VAR_MLE=0.5*0.8*0.8+SD*SD;
function drawsTruth(seed){const r=rng(seed),A=new Float64Array(2*NSAMP);for(let s=0;s<NSAMP;s++){const k=Math.min(3,Math.floor(4*r()));const g1=gauss(r),g2=gauss(r);A[2*s]=MU[k][0]+SD*g1;A[2*s+1]=MU[k][1]+SD*g2}return A}
function drawsGauss(seed){const r=rng(seed),sd=Math.sqrt(VAR_MLE),A=new Float64Array(2*NSAMP);for(let s=0;s<NSAMP;s++){A[2*s]=sd*gauss(r);A[2*s+1]=sd*gauss(r)}return A}
function drawsNoise(seed,N){const r=rng(seed),L=2*N,buf=new Float64Array(NSAMP*L);for(let s=0;s<NSAMP;s++)for(let j=0;j<L;j++)buf[s*L+j]=gauss(r);return buf}
function drawsDiff(seed,N,z){const buf=drawsNoise(seed,N),S=schedule(N),L=2*N,A=new Float64Array(2*NSAMP),o=[0,0];z=z||new Float64Array(4);
  for(let s=0;s<NSAMP;s++){chainOne(buf,s*L,S,z,null,o);A[2*s]=o[0];A[2*s+1]=o[1]}return A}
function metricsC2(A){let near=0,cen=0,out=0;const sh=[0,0,0,0];const n=A.length/2;
  for(let s=0;s<n;s++){const x=A[2*s],y=A[2*s+1];let dm=Infinity,km=0;for(let k=0;k<4;k++){const d=(x-MU[k][0])**2+(y-MU[k][1])**2;if(d<dm){dm=d;km=k}}
    const nr=Math.sqrt(dm)<0.2;if(nr){near++;sh[km]++}if(Math.sqrt(x*x+y*y)<0.3)cen++;if(Math.abs(x)>1||Math.abs(y)>1)out++}
  return{near:near/n,cen:cen/n,out:out/n,sh:sh.map(v=>v/n)}}

/* ---------- shared scatter ---------- */
function scatter(svg,box,A,cls,extra){const W=Math.min(widthOf(box,240,440),440),m=28,H=W;
  const G=frame(svg,{W,H,m:{l:m+8,r:8,t:8,b:m},x:[-1.2,1.2],y:[-1.2,1.2],xt:[-1,0,1],yt:[-1,0,1]});
  el('rect',{x:G.X(-1),y:G.Y(1),width:G.X(1)-G.X(-1),height:G.Y(-1)-G.Y(1),class:'box'},G.g);
  const px=(G.X(1)-G.X(0));
  MU.forEach((c,k)=>{el('circle',{cx:G.X(c[0]),cy:G.Y(c[1]),r:0.2*px,class:'peak'},G.g)});
  el('circle',{cx:G.X(0),cy:G.Y(0),r:0.3*px,class:'peak'},G.g);
  if(extra)extra(G,px);
  const n=A.length/2,g=el('g',null,svg),sym=cls==='p-gauss'?'sq':cls==='p-diff'?'dia':'c';
  for(let s=0;s<n;s++){const x=Math.max(-1.2,Math.min(1.2,A[2*s])),y=Math.max(-1.2,Math.min(1.2,A[2*s+1])),X=G.X(x),Y=G.Y(y);
    if(sym==='c')el('circle',{cx:X.toFixed(1),cy:Y.toFixed(1),r:2.5,class:cls},g);
    else if(sym==='sq')el('rect',{x:(X-2.2).toFixed(1),y:(Y-2.2).toFixed(1),width:4.4,height:4.4,class:cls},g);
    else el('path',{d:`M${X.toFixed(1)} ${(Y-3).toFixed(1)}l3 3l-3 3l-3-3z`,class:cls},g)}
  tx(svg,G.X(0),G.Y(0)+0.3*px+14,'峰之間沒有資料的區域',{class:'tk halo','text-anchor':'middle'});
  return G}

/* ---------- C1 schedule ---------- */
const c1s={N:5};
function renderC1(){const box=$('#c1-fig'),svg=$('svg',box),S=schedule(c1s.N),N=c1s.N,W=chartW(box),H=Math.round(Math.max(200,Math.min(260,W*.6)));
  const bw=Math.max(2,(W-50)*0.6/N),G=frame(svg,{W,H,m:{l:40,r:Math.max(12,bw/2+4),t:12,b:44},x:[0,1],y:[0,1],xt:[0,.5,1],yt:[0,.5,1],xl:'i / N',xf:v=>String(v)});
  for(let i=1;i<=N;i++){const x=G.X(i/N);el('rect',{x:x-bw/2,y:G.Y(S.beta[i-1]),width:bw,height:G.Y(0)-G.Y(S.beta[i-1]),class:'bbar'},G.g)}
  const pts=[];for(let j=0;j<=100;j++){const t=j/100;pts.push([G.X(t),G.Y(Math.exp(-0.1*t-4.95*t*t))])}el('path',{d:lp(pts),class:'ref'},G.g);
  for(let i=1;i<=N;i++){const x=G.X(i/N),y=G.Y(S.abar[i-1]);el('path',{d:`M${x} ${y-5}l5 5l-5 5l-5-5z`,class:'acc-f'},G.g)}
  svg.setAttribute('aria-label',`N = ${N} 的噪聲排程：最後一步訊號保留比例 0.0064`);
  const idx=N<=5?Array.from({length:N},(_,j)=>j+1):[1,2,null,N-1,N];
  $('#c1-body').innerHTML=idx.map(i=>i===null?'<tr><td>…</td><td>…</td><td>…</td><td>…</td></tr>':`<tr><td class="num">${i}</td><td class="num">${f4(S.beta[i-1])}</td><td class="num">${f4(S.abar[i-1])}</td><td class="num">${f4(Math.sqrt(S.abar[i-1]))}</td></tr>`).join('');
  $('#c1-nv').textContent=N}
function initC1(){radiogroup($('#c1-n'),c1s.N,v=>{c1s.N=+v;renderC1();announce('c1',`N = ${v}，最後一步訊號保留比例 0.0064`)})}

/* ---------- C2 behaviour cloning (two instances) ---------- */
const c2s={seed:7,cache:new Map()};
function c2Get(kind,N){const key=`${c2s.seed}|${kind}|${N}`;if(!c2s.cache.has(key)){const A=kind==='truth'?drawsTruth(c2s.seed):kind==='gauss'?drawsGauss(c2s.seed):drawsDiff(c2s.seed,N);c2s.cache.set(key,{A,m:metricsC2(A)})}return c2s.cache.get(key)}
const c2inst=[];
function renderC2(){c2inst.forEach(st=>{const box=st.box,fig=$('.c2-fig',box),svg=$('svg',fig),d=c2Get(st.tab,st.N);
  const cls=st.tab==='truth'?'p-true':st.tab==='gauss'?'p-gauss':'p-diff';
  scatter(svg,fig,d.A,cls,st.tab==='gauss'?(G,px)=>{const sd=Math.sqrt(VAR_MLE);[1,2].forEach(k=>el('circle',{cx:G.X(0),cy:G.Y(0),r:k*sd*px,class:'ref'},G.g))}:null);
  const name=st.tab==='truth'?'真實資料':st.tab==='gauss'?'對角高斯（MLE）':`擴散（理想去噪器，N = ${st.N}）`,m=d.m;
  svg.setAttribute('aria-label',`${name}：峰附近 ${f3(m.near)}，中央無資料區域 ${f3(m.cen)}`);
  $('.c2-ro',box).innerHTML=`<div class="tscroll rt"><table><caption>${name}${st.tab==='diff'?' <span class="tag toy">玩具</span>':st.tab==='gauss'?' <span class="tag">本頁推導</span>':''}</caption><tbody>
   <tr><th>落在峰附近（&lt; 0.2）</th><td class="num acc">${f3(m.near)}</td></tr>
   <tr><th>落在中央無資料區域（&lt; 0.3）</th><td class="num">${f3(m.cen)}</td></tr>
   <tr><th>落在動作範圍外</th><td class="num">${f3(m.out)}</td></tr>
   <tr><th>峰附近：上／右／下／左</th><td class="num">${m.sh.map(f3).join(' ／ ')}</td></tr>
   ${st.tab==='gauss'?'<tr><th>擬合高斯：原點密度 ÷ 峰中心密度</th><td class="num">2.70</td></tr>':''}</tbody></table></div>
   <div class="legend"><span class="lg ${st.tab==='truth'?'circ':st.tab==='gauss'?'sq':'dia'}">樣本</span><span class="lg ref">資料峰（半徑 0.2）與中央區域（半徑 0.3）${st.tab==='gauss'?'；1σ、2σ 等密度圓':''}</span></div>`;
  $$('.c2-seedv',box).forEach(n=>n.textContent=c2s.seed);
  const nr=$('.c2-nrow',box);if(nr)nr.style.visibility=st.tab==='diff'?'visible':'hidden'})}
function initC2(){[['#c2a','gauss',50],['#c2b','diff',50]].forEach(([id,tab,N])=>{const box=$(id),st={box,tab,N,id:id.slice(1)};c2inst.push(st);
  radiogroup($('.c2-tab',box),tab,v=>{st.tab=v;renderC2();announce(st.id,'已切換分布')});
  const nb=$('.c2-n',box);if(nb)radiogroup(nb,N,v=>{st.N=+v;renderC2();const m=c2Get('diff',st.N).m;announce(st.id,`N = ${v}，峰附近 ${f3(m.near)}`)});
  $('.c2-seed',box).addEventListener('click',()=>{c2s.seed++;c2s.cache.clear();renderC2();announce(st.id,`seed ${c2s.seed}`)})})}

/* ---------- C3 Diffusion-QL toy (design C3; same algorithm as verify/c3.py) ---------- */
const C3={B:256,LR:0.03,B1:0.9,B2:0.999,AE:1e-8,STEPS:600};
function c3New(N,eta){return{N,eta,S:schedule(N),r:rng(7),z:new Float64Array(4),m:new Float64Array(4),v:new Float64Array(4),step:0,
  hist:{w:[],ld:[],lq:[]},wavg:new Float64Array(4),Ld:NaN,Lq:NaN,ev:null,runId:0,h:0,ht:0,running:false,
  k:new Int32Array(C3.B),xn:new Float64Array(2*C3.B),ii:new Int32Array(C3.B),ep:new Float64Array(2*C3.B),buf:new Float64Array(C3.B*2*N)}}
const softmax=z=>{const m=Math.max(...z);const e=Array.from(z,v=>Math.exp(v-m));const s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s)};
const T8=new Float64Array(8),O2=[0,0],QG={gx:0,gy:0};
function c3Step(st){const B=C3.B,N=st.N,S=st.S,r=st.r,L=2*N;
  for(let b=0;b<B;b++){st.k[b]=Math.min(3,Math.floor(4*r()));st.xn[2*b]=gauss(r);st.xn[2*b+1]=gauss(r);st.ii[b]=1+Math.min(N-1,Math.floor(N*r()));st.ep[2*b]=gauss(r);st.ep[2*b+1]=gauss(r)}
  for(let b=0;b<B;b++)for(let j=0;j<L;j++)st.buf[b*L+j]=gauss(r);
  const g=[0,0,0,0];let Ld=0;
  for(let b=0;b<B;b++){const k=st.k[b],ax=MU[k][0]+SD*st.xn[2*b],ay=MU[k][1]+SD*st.xn[2*b+1],i=st.ii[b],ab=S.abar[i-1],sa=Math.sqrt(ab),s1=Math.sqrt(1-ab);
    const ex=st.ep[2*b],ey=st.ep[2*b+1];epsStar(sa*ax+s1*ex,sa*ay+s1*ey,i,S,st.z,EO,true);const rx=ex-EO.ex,ry=ey-EO.ey;Ld+=rx*rx+ry*ry;
    for(let j=0;j<4;j++)g[j]+=-2*(rx*EO.Jl[j]+ry*EO.Jl[4+j])}
  Ld/=B;for(let j=0;j<4;j++)g[j]/=B;
  const al=st.eta/QNORM,gq=[0,0,0,0];let qs=0;
  for(let b=0;b<B;b++){T8.fill(0);chainOne(st.buf,b*L,S,st.z,T8,O2);qs+=qToy(O2[0],O2[1],QG);for(let j=0;j<4;j++)gq[j]+=QG.gx*T8[j]+QG.gy*T8[4+j]}
  const Lq=-al*qs/B;for(let j=0;j<4;j++)g[j]+=-al*gq[j]/B;
  st.step++;const t=st.step;
  for(let j=0;j<4;j++){st.m[j]=C3.B1*st.m[j]+(1-C3.B1)*g[j];st.v[j]=C3.B2*st.v[j]+(1-C3.B2)*g[j]*g[j];
    st.z[j]-=C3.LR*(st.m[j]/(1-Math.pow(C3.B1,t)))/(Math.sqrt(st.v[j]/(1-Math.pow(C3.B2,t)))+C3.AE)}
  const w=softmax(st.z);if(t>C3.STEPS-50)for(let j=0;j<4;j++)st.wavg[j]+=w[j]/50;
  st.Ld=Ld;st.Lq=Lq;st.hist.w.push(w);st.hist.ld.push(Ld);st.hist.lq.push(Lq);return st}
function c3Eval(st){const A=drawsDiff(11,st.N,st.z),m=metricsC2(A);let q=0;for(let s=0;s<NSAMP;s++)q+=qToy(A[2*s],A[2*s+1]);return{A,near:m.near,bot:m.sh[2],q:q/NSAMP}}
let c3st=null;const c3ui={eta:1,N:5};
function c3Cancel(){if(!c3st)return;cancelAnimationFrame(c3st.h);clearTimeout(c3st.ht);c3st.h=0;c3st.ht=0;c3st.runId++;c3st.running=false}
function c3Reset(){c3Cancel();const id=c3st?c3st.runId+1:0;c3st=c3New(c3ui.N,c3ui.eta);c3st.runId=id;c3st.ev=c3Eval(c3st);$('#c3-run').textContent='執行';renderC3()}
function c3Run(){const st=c3st;if(st.step>=C3.STEPS){c3Reset();return c3Run()}
  st.running=true;$('#c3-run').textContent='暫停';const id=st.runId;
  const work=()=>{if(st!==c3st||id!==st.runId)return;const t0=performance.now();let lastEval=Math.floor(st.step/50);
    do{c3Step(st)}while(st.step<C3.STEPS&&performance.now()-t0<8);
    const done=st.step>=C3.STEPS;
    if(!RM&&(Math.floor(st.step/50)>lastEval||done))st.ev=c3Eval(st);
    if(done){st.running=false;st.ev=c3Eval(st);$('#c3-run').textContent='執行';renderC3();const w=softmax(st.z);announce('c3',`第 ${st.step} 步，下峰權重 ${f3(w[2])}`);return}
    if(!RM)renderC3();
    if(RM)st.ht=setTimeout(work,0);else st.h=requestAnimationFrame(work)};
  if(RM)st.ht=setTimeout(work,0);else st.h=requestAnimationFrame(work)}
function c3Pause(){c3Cancel();$('#c3-run').textContent='執行';renderC3();const w=softmax(c3st.z);announce('c3',`暫停於第 ${c3st.step} 步，下峰權重 ${f3(w[2])}`)}
function renderC3(){const st=c3st;if(!st)return;
  const sc=$('#c3-sc'),ssv=$('svg',sc);const G=scatter(ssv,sc,st.ev.A,'p-diff');
  [[.3,.95],[.8,.3],[.3,-.95],[-.8,.3]].forEach((p,k)=>tx(G.g,G.X(p[0]),G.Y(p[1])+4,'r '+RBAR[k].toFixed(1),{class:'labm halo','text-anchor':'middle'}));
  ssv.setAttribute('aria-label',`評估樣本：落在下峰附近 ${f3(st.ev.bot)}`);
  const wb=$('#c3-w'),wsv=$('svg',wb),W=chartW(wb),H=190;
  const Gw=frame(wsv,{W,H,m:{l:40,r:52,t:24,b:42},x:[0,C3.STEPS],y:[0,1],xt:[0,300,600],yt:[0,.25,.5,.75,1],xl:'步數',yl:'權重'});
  const cls=['w-top','w-right','w-bot','w-left'],hw=st.hist.w;
  if(hw.length){for(let k=0;k<4;k++){const pts=[[Gw.X(0),Gw.Y(.25)]];const stp=Math.max(1,Math.floor(hw.length/300));for(let s=0;s<hw.length;s+=stp)pts.push([Gw.X(s+1),Gw.Y(hw[s][k])]);pts.push([Gw.X(hw.length),Gw.Y(hw[hw.length-1][k])]);el('path',{d:lp(pts),class:cls[k]},Gw.g)}
    const last=hw[hw.length-1],ys=[0,1,2,3].map(k=>({k,y:Gw.Y(last[k])})).sort((a,b)=>a.y-b.y);for(let j=1;j<4;j++)if(ys[j].y-ys[j-1].y<13)ys[j].y=ys[j-1].y+13;
    ys.forEach(o=>tx(Gw.g,Gw.X(hw.length)+4,o.y+4,PEAK[o.k]+' '+RBAR[o.k].toFixed(1),{class:'tk'}))}
  else for(let k=0;k<4;k++)el('line',{x1:Gw.X(0),x2:Gw.X(6),y1:Gw.Y(.25),y2:Gw.Y(.25),class:cls[k]},Gw.g);
  const lb=$('#c3-l'),lsv=$('svg',lb),ld=st.hist.ld,lq=st.hist.lq;
  const ma=(a,s)=>{let t=0;const n=Math.min(20,s+1);for(let j=s-n+1;j<=s;j++)t+=a[j];return t/n};
  let ymax=1;for(let s=0;s<ld.length;s++)ymax=Math.max(ymax,ma(ld,s),-ma(lq,s));
  const Gl=frame(lsv,{W,H:130,m:{l:40,r:52,t:12,b:42},x:[0,C3.STEPS],y:[0,Math.ceil(ymax)],xt:[0,300,600],yt:[0,Math.ceil(ymax)],xl:'步數'});
  if(ld.length){const stp=Math.max(1,Math.floor(ld.length/300)),p1=[],p2=[];for(let s=0;s<ld.length;s+=stp){p1.push([Gl.X(s+1),Gl.Y(ma(ld,s))]);p2.push([Gl.X(s+1),Gl.Y(-ma(lq,s))])}
    el('path',{d:lp(p1),class:'ld'},Gl.g);el('path',{d:lp(p2),class:'lq'},Gl.g);
    tx(Gl.g,Gl.X(ld.length)+4,p1[p1.length-1][1]+4,'Ld',{class:'tk'});tx(Gl.g,Gl.X(ld.length)+4,p2[p2.length-1][1]+(Math.abs(p2[p2.length-1][1]-p1[p1.length-1][1])<12?16:4),'−Lq',{class:'tk'})}
  const w=softmax(st.z),cells=$$('#c3-ro td');cells[0].textContent=st.step;w.forEach((v,k)=>cells[1+k].textContent=f3(v));cells[5].textContent=isNaN(st.Ld)?'—':f3(st.Ld);cells[6].textContent=isNaN(st.Lq)?'—':f3(st.Lq);
  const ev=$$('#c3-ev td');ev[0].textContent=f3(st.ev.near);ev[1].textContent=f3(st.ev.bot);ev[2].textContent=f3(st.ev.q)}
function initC3(){const eta=$('#c3-eta'),eo=$('#c3-etao');
  const setChip=radiogroup($('#c3-chip'),'1',v=>{c3ui.eta=+v;eta.value=v;eo.textContent=(+v).toFixed(2);c3Reset()});
  eta.addEventListener('input',()=>{c3ui.eta=+eta.value;eo.textContent=c3ui.eta.toFixed(2);setChip(['0','0.3','1','3'].find(s=>+s===c3ui.eta)||'none');c3Reset()});
  radiogroup($('#c3-n'),'5',v=>{c3ui.N=+v;c3Reset()});
  $('#c3-run').addEventListener('click',()=>{if(c3st.running)c3Pause();else c3Run()});
  $('#c3-reset').addEventListener('click',()=>{c3Reset();announce('c3','已重設')});
  c3Reset()}

/* ---------- C4 results (data = verify/c4_data.json) ---------- */
const RES=/*C4DATA*/null;
const c4s={tab:'t1',dom:0,mode:'best'};
function bestOf(base,algs){let b=-Infinity,who=[];base.forEach((v,j)=>{if(v>b){b=v;who=[algs[j]]}else if(v===b)who.push(algs[j])});return{b,who}}
function c4Summary(){return RES.T1.map(D=>{const n=D.rows.length;const avgBest=bestOf(D.avg.slice(0,-1),D.algs);const exc=D.rows.filter(r=>r[2][0]<Math.max(...r[1])).map(r=>r[0]);return{dom:D.domain,dql:D.avg[D.avg.length-1],avgBest,exc,n}})}
function c4Bar(cls,label,v,max,err){const p=Math.max(0,v)/max*100;return `<div class="c4-b"><span class="who" title="${label}">${label}</span><span class="c4-track"><span class="c4-fill ${cls}" style="width:${Math.min(100,p).toFixed(2)}%"></span>${err?`<span class="c4-err" style="left:${((v-err)/max*100).toFixed(2)}%;width:${(2*err/max*100).toFixed(2)}%"></span>`:''}</span><span class="val">${v.toFixed(1)}${err?' ± '+err.toFixed(1):''}</span></div>`}
function renderC4(){const list=$('#c4-list'),lg=$('#c4-lg'),tb=$('#c4-tab');
  if(c4s.tab==='t1'){$('#c4-ctl').style.display='';const D=RES.T1[c4s.dom],max=D.domain==='Gym'?120:100;
    lg.innerHTML='<span class="lg dql">Diffusion-QL（± 標準誤）</span><span class="lg best">該任務最高的基準</span>'+(c4s.mode==='all'?'<span class="lg oth">其他基準</span>':'');
    let h='';D.rows.forEach(([name,base,dq])=>{const bo=bestOf(base,D.algs),lower=dq[0]<bo.b;
      h+=`<div class="c4-row"><div class="c4-name">${lower?'<span class="flag">▲ 基準較高</span>':''}${name}</div>`+c4Bar('dql','Diffusion-QL',dq[0],max,dq[1]);
      if(c4s.mode==='best')h+=c4Bar('best',bo.who.join('、'),bo.b,max);
      else D.algs.forEach((a,j)=>{h+=c4Bar(base[j]===bo.b?'best':'oth',a,base[j],max)});h+='</div>'});
    const ab=bestOf(D.avg.slice(0,-1),D.algs);
    h+=`<div class="c4-row avg"><div class="c4-name">領域平均（論文 Avg）</div>`+c4Bar('dql','Diffusion-QL',D.avg[D.avg.length-1],max)+c4Bar('best','平均最高：'+ab.who.join('、'),ab.b,max)+'</div>';
    list.innerHTML=h;
    tb.innerHTML=`<table><caption>Table 1 · ${D.domain}（完整數字）</caption><thead><tr><th>任務</th>${D.algs.map(a=>`<th class="num">${a}</th>`).join('')}<th class="num">Diffusion-QL</th></tr></thead><tbody>${D.rows.map(([n,b,d])=>`<tr><td>${n}</td>${b.map(v=>`<td class="num">${v.toFixed(1)}</td>`).join('')}<td class="num acc">${d[0].toFixed(1)} ± ${d[1].toFixed(1)}</td></tr>`).join('')}<tr><td>Average</td>${D.avg.map((v,j)=>`<td class="num${j===D.avg.length-1?' acc':''}">${v.toFixed(1)}</td>`).join('')}</tr></tbody></table>`}
  else{$('#c4-ctl').style.display='none';lg.innerHTML='<span class="lg dql">離線模型選擇（± 標準誤）</span><span class="lg on">線上模型選擇</span>';
    let h='';RES.T4.forEach(([n,off,on])=>{const max=n.startsWith('antmaze')||n.startsWith('pen')||n.startsWith('kitchen')?100:120;h+=`<div class="c4-row"><div class="c4-name">${n}</div>`+c4Bar('dql','離線',off[0],max,off[1])+c4Bar('on','線上',on[0],max,on[1])+'</div>'});
    list.innerHTML=h;
    tb.innerHTML=`<table><caption>Table 4 · 領域平均</caption><thead><tr><th>領域</th><th class="num">離線</th><th class="num">線上</th></tr></thead><tbody>${Object.entries(RES.T4avg).map(([k,v])=>`<tr><td>${k}</td><td class="num">${v[0].toFixed(1)}</td><td class="num">${v[1].toFixed(1)}</td></tr>`).join('')}</tbody></table>`}}
function initC4(){radiogroup($('#c4-dom'),'0',v=>{c4s.dom=+v;renderC4()});radiogroup($('#c4-mode'),'best',v=>{c4s.mode=v;renderC4()});
  const tabs=$$('#c4 [role=tab]');const sel=(t,focus)=>{tabs.forEach(b=>{const on=b===t;b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1;if(on&&focus)b.focus()});c4s.tab=t.id==='c4-t1'?'t1':'t4';renderC4()};
  tabs.forEach((t,i)=>{t.type='button';t.addEventListener('click',()=>sel(t));t.addEventListener('keydown',e=>{let j=null;if(e.key==='ArrowRight')j=(i+1)%tabs.length;if(e.key==='ArrowLeft')j=(i-1+tabs.length)%tabs.length;if(e.key==='Home')j=0;if(e.key==='End')j=tabs.length-1;if(j!==null){e.preventDefault();sel(tabs[j],true)}})});
  $('#t3-body').innerHTML=RES.T3.map(([n,lr,eta,mq])=>`<tr><td>${n}</td><td class="num" data-l="學習率">${lr.replace('e-','×10⁻').replace(/^(\d)×/,'$1 × ')}</td><td class="num" data-l="η">${eta}</td><td data-l="max Q backup">${mq?'是':'否'}</td></tr>`).join('');
  renderC4()}

/* ---------- timeline ---------- */
const TL_GROUP={'1503.03585':['dm','ddpm-training-and-sampling-recap'],'1907.05600':['dm','ddpm-training-and-sampling-recap'],'2006.11239':['dm','ddpm-training-and-sampling-recap'],'2112.07804':['dm','ddpm-training-and-sampling-recap'],
 '1509.02971':['ac','ddpg-td3-actor-critic'],
 '1812.02900':['pr','batch-constrained-policy-constraint-methods'],'1906.00949':['pr','batch-constrained-policy-constraint-methods'],'1911.11361':['pr','batch-constrained-policy-constraint-methods'],'2106.06860':['pr','batch-constrained-policy-constraint-methods'],'2110.06169':['pr','batch-constrained-policy-constraint-methods'],
 '2004.07219':['bench','offline-rl-benchmarks-and-baselines'],'2006.04779':['bench','offline-rl-benchmarks-and-baselines'],
 '2205.09991':['co','diffusion-in-rl-concurrent-work'],'2206.11251':['co','diffusion-in-rl-concurrent-work'],
 '2208.06193':['me','diffusion-policy-class']};
const TL_NAME={dm:'擴散模型',ac:'Actor-critic',pr:'策略正則化',bench:'基準與保守 Q',co:'同期工作',me:'本文',other:'其他'};
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

/* ---------- self test ---------- */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t)=>Math.abs(a-b)<=t+1e-9;
  // C1
  ok(near(schedule(5).abar[4],0.006409,5e-7)&&near(schedule(50).abar[49],0.006409,5e-7),'abar_N = e^-5.05');
  ok(near(schedule(10).abar[1],schedule(5).abar[0],1e-12),'abar depends on i/N');
  [[2,[.7240,.9768]],[5,[.1959,.4588,.6358,.7549,.8350]]].forEach(([N,b])=>b.forEach((v,i)=>ok(near(schedule(N).beta[i],v,1e-4),`beta N=${N} i=${i+1}`)));
  [.8041,.4352,.1585,.0389,.0064].forEach((v,i)=>ok(near(schedule(5).abar[i],v,1e-4),'abar N=5 i='+(i+1)));
  // jacobian finite differences
  {const S=schedule(5),z=new Float64Array([.3,-.2,.5,.1]),o={ex:0,ey:0,Jx:new Float64Array(4),Jl:new Float64Array(8)},p={ex:0,ey:0},m={ex:0,ey:0},h=1e-6;let err=0;
   for(const i of [1,3,5])for(const [x,y] of [[.31,-.42],[-.9,.2]]){epsStar(x,y,i,S,z,o,true);const J=Array.from(o.Jx),Jl=Array.from(o.Jl);
     epsStar(x+h,y,i,S,z,p);epsStar(x-h,y,i,S,z,m);err=Math.max(err,Math.abs((p.ex-m.ex)/(2*h)-J[0]),Math.abs((p.ey-m.ey)/(2*h)-J[2]));
     epsStar(x,y+h,i,S,z,p);epsStar(x,y-h,i,S,z,m);err=Math.max(err,Math.abs((p.ex-m.ex)/(2*h)-J[1]),Math.abs((p.ey-m.ey)/(2*h)-J[3]));
     for(let j=0;j<4;j++){const zp=new Float64Array(z),zm=new Float64Array(z);zp[j]+=h;zm[j]-=h;epsStar(x,y,i,S,zp,p);epsStar(x,y,i,S,zm,m);err=Math.max(err,Math.abs((p.ex-m.ex)/(2*h)-Jl[j]),Math.abs((p.ey-m.ey)/(2*h)-Jl[4+j]))}}
   ok(err<1e-6,'epsStar jacobian FD err='+err.toExponential(2))}
  // C2 (seed 7, c2_out.txt)
  ok(near(VAR_MLE,.3225,1e-12),'MLE variance 0.3225');
  const mt=metricsC2(drawsTruth(7)),mg=metricsC2(drawsGauss(7));
  ok(near(mt.near,1,.002)&&near(mt.sh[0],.257,.002)&&near(mt.sh[1],.236,.002),'C2 truth');
  ok(near(mg.near,.106,.002)&&near(mg.cen,.136,.002)&&near(mg.out,.136,.002),'C2 gauss');
  const tabC2={1:[0,1],2:[.001,.636],3:[.119,.177],5:[.484,.036],10:[.825,.003],20:[.974,.001],50:[1,0]};
  for(const N in tabC2){const m=metricsC2(drawsDiff(7,+N));ok(near(m.near,tabC2[N][0],.002)&&near(m.cen,tabC2[N][1],.002),`C2 diffusion N=${N}: near=${m.near} centre=${m.cen}`)}
  {const A5=drawsDiff(7,5),A50=drawsDiff(7,50);ok(near(A5[0],-.7729,1e-4)&&near(A5[1],.0369,1e-4),'C2 first sample N=5');ok(near(A50[0],-.7513,1e-4)&&near(A50[1],.0037,1e-4),'C2 first sample N=50')}
  // C3 (c3_out.txt)
  {const st=c3New(5,1);for(let i=0;i<C3.STEPS;i++)c3Step(st);const w=softmax(st.z),ev=c3Eval(st);
   ok(near(st.wavg[2],.883,.02),'C3 N=5 eta=1 avg bottom '+st.wavg[2].toFixed(3));ok(w[2]>.8,'C3 N=5 eta=1 last bottom > 0.8');
   ok(near(ev.near,.792,.01)&&near(ev.bot,.721,.01)&&near(ev.q,4.516,.02),`C3 N=5 eta=1 eval ${ev.near} ${ev.bot} ${ev.q.toFixed(3)}`)}
  {const st=c3New(5,0);for(let i=0;i<C3.STEPS;i++)c3Step(st);ok(Array.from(st.wavg).every(v=>v>=.22&&v<=.28),'C3 N=5 eta=0 avg weights in [0.22,0.28]: '+Array.from(st.wavg).map(f3).join(','))}
  {c3Run();c3Reset();ok(c3st.step===0&&!c3st.running&&c3st.h===0,'C3 reset cancels pending run')}
  // C4
  const sm=c4Summary();ok(sm.map(s=>s.avgBest.b).join()==='77.6,63,56.5,53.3','C4 best single-baseline averages');ok(sm.reduce((a,s)=>a+s.exc.length,0)===3,'C4 exceptions = 3');
  ok(sm.map(s=>s.dql).join()==='88,69.6,65.1,69','C4 DQL averages');
  // no combining characters in svg text / live regions
  const bad=$$('svg text, [aria-live]').filter(n=>/[\u0300-\u036f]/.test(n.textContent));ok(bad.length===0,'no combining characters ('+bad.length+')');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'')}
