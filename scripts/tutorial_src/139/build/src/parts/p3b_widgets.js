
/* ---------- shared math (mirrors verify/common.py, c1–c4.py) ---------- */
function bisectExpectile(pts,tau){let lo=pts[0][0],hi=pts[pts.length-1][0];
  for(let it=0;it<200;it++){const m=(lo+hi)/2;let g=0;for(const [x,p] of pts)g+=p*(x>=m?tau:1-tau)*(x-m);if(g>0)lo=m;else hi=m}return(lo+hi)/2}
function expectile(xs,ps,tau){const pts=[];for(let i=0;i<xs.length;i++)if(ps[i]>0)pts.push([xs[i],ps[i]]);
  if(!pts.length)return NaN;pts.sort((a,b)=>a[0]-b[0]);const n=pts.length;
  if(tau>=1)return pts[n-1][0];if(tau<=0)return pts[0][0];
  const P=[0],S=[0];for(const [x,p] of pts){P.push(P[P.length-1]+p);S.push(S[S.length-1]+p*x)}
  for(let k=0;k<n;k++){const lp=P[k],ls=S[k],hp=P[n]-P[k],hs=S[n]-S[k];
    const m=(tau*hs+(1-tau)*ls)/(tau*hp+(1-tau)*lp);
    if((k===0||pts[k-1][0]<=m+1e-12)&&pts[k][0]>=m-1e-12)return m}
  return bisectExpectile(pts,tau)}
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}
/* Acklam inverse normal CDF (same coefficients as verify/common.py) */
const AK=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00],
  BK=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01],
  CK=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00],
  DK=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00];
function invNorm(p){const pl=0.02425;let q,r;
  if(p<pl){q=Math.sqrt(-2*Math.log(p));return(((((CK[0]*q+CK[1])*q+CK[2])*q+CK[3])*q+CK[4])*q+CK[5])/((((DK[0]*q+DK[1])*q+DK[2])*q+DK[3])*q+1)}
  if(p>1-pl){q=Math.sqrt(-2*Math.log(1-p));return-(((((CK[0]*q+CK[1])*q+CK[2])*q+CK[3])*q+CK[4])*q+CK[5])/((((DK[0]*q+DK[1])*q+DK[2])*q+DK[3])*q+1)}
  q=p-0.5;r=q*q;return(((((AK[0]*r+AK[1])*r+AK[2])*r+AK[3])*r+AK[4])*r+AK[5])*q/(((((BK[0]*r+BK[1])*r+BK[2])*r+BK[3])*r+BK[4])*r+1)}
function w1Sorted(x,y){let s=0;for(let i=0;i<x.length;i++)s+=Math.abs(x[i]-y[i]);return s/x.length}
function w1Weighted(xa,wa,xb,wb){const n=xa.length,m=xb.length,ev=new Array(n+m);
  for(let i=0;i<n;i++)ev[i]=[xa[i],wa[i]];for(let i=0;i<m;i++)ev[n+i]=[xb[i],-wb[i]];
  ev.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);let tot=0,diff=0,prev=ev[0][0];
  for(const [x,w] of ev){tot+=Math.abs(diff)*(x-prev);diff+=w;prev=x}return tot}
function gaussVel(m,s,tau){const sig2=tau*tau*s*s+(1-tau)**2,k=(tau*s*s-(1-tau))/sig2;return{k,b:m-k*tau*m,sig:Math.sqrt(sig2)}}
const fmt=(v,d)=>{if(!isFinite(v))return'—';const t=(Math.abs(v)<0.5*10**-d?0:v).toFixed(d);return t.replace('-','−')};
const sched={};function schedule(id,fn){if(sched[id])return;sched[id]=requestAnimationFrame(()=>{sched[id]=0;fn()})}
function niceTicks(lo,hi,n){const span=hi-lo,step0=span/n,mag=10**Math.floor(Math.log10(step0)),step=[1,2,2.5,5,10].map(f=>f*mag).find(s=>s>=step0)||10*mag;const out=[];for(let v=Math.ceil(lo/step)*step;v<=hi+1e-9;v+=step)out.push(+v.toFixed(10));return out}
const tk=v=>fmt(v,Math.abs(v)<10&&v%1!==0?(Math.abs(v*10%1)>1e-9?2:1):0);
function setLabel(svg,s){svg.setAttribute('aria-label',s)}
function bindRange(id,outId,fmtOut,onInput,onChange){const r=$('#'+id),o=$('#'+outId);
  const sync=()=>{const t=fmtOut(+r.value);o.textContent=t;r.setAttribute('aria-valuetext',t)};
  r.addEventListener('input',()=>{sync();onInput(+r.value)});r.addEventListener('change',()=>{if(onChange)onChange(+r.value)});sync();return{r,sync}}

/* ---------- C1 ---------- */
const c1={g:.9,g4:false,t:.5,r:-1,m:-2,s:.5,scale:false};
function c1Values(g,tau,r,m1,s1){const c=Math.sqrt(Math.max(0,(1-g)*(1+g))),vn=tau*tau*s1*s1+(1-tau)**2;
  const tgt=[tau*(r+g*m1),Math.sqrt(tau*tau*g*g*s1*s1+(1-tau)**2)],cor=[tau*r+g*tau*m1,Math.sqrt(g*g*vn+(1-tau)**2*c*c)],cln=[r+g*tau*m1,Math.sqrt(g*g*vn)];
  return{c,tgt,cor,cln,w2:Math.hypot(cln[0]-tgt[0],cln[1]-tgt[1])}}
const c1G=()=>c1.g4?0.9999:c1.g;
const c1R=()=>c1.scale?(1-c1G())*c1.r:c1.r;
const npdf=(x,m,s)=>Math.exp(-.5*((x-m)/s)**2)/(s*Math.sqrt(2*Math.PI));
function renderC1(final){const g=c1G(),r=c1R(),v=c1Values(g,c1.t,r,c1.m,c1.s);
  /* fig 1 */
  {const box=$('#c1-f1'),svg=$('svg',box),W=widthOf(box,260,520),H=220;
   const ds=[v.tgt,v.cor,v.cln];let lo=Infinity,hi=-Infinity;ds.forEach(([m,s])=>{const ss=Math.max(s,1e-3);lo=Math.min(lo,m-4*ss);hi=Math.max(hi,m+4*ss)});
   const ymax=npdf(0,0,Math.max(v.tgt[1],1e-3))*1.3;
   const F=frame(svg,{W,H,m:{l:40,r:10,t:16,b:36},x:[lo,hi],y:[0,ymax],xt:niceTicks(lo,hi,5),xf:tk,yt:[0],xl:'回報值'});
   const curve=(m,s,cls,mk)=>{if(s<1e-6){const x=F.X(m);el('line',{x1:x,x2:x,y1:F.y0,y2:F.y1,class:cls},F.g);tx(F.g,x+4,F.y1+12,'點質量',{class:'tk'});return}
     const pts=[];let clipped=false;for(let i=0;i<=200;i++){const x=lo+(hi-lo)*i/200;let y=npdf(x,m,s);if(y>ymax){y=ymax;clipped=true}pts.push([F.X(x),F.Y(y)])}
     el('path',{d:lp(pts),class:cls},F.g);if(mk)for(let i=0;i<=200;i+=25){const [x,y]=pts[i];if(mk==='c')el('circle',{cx:x,cy:y,r:3.2,class:'ok-f'},F.g);else el('path',{d:`M${x} ${y-4}L${x+4} ${y+3}L${x-4} ${y+3}Z`,class:'cl-f'},F.g)}
     if(clipped)tx(F.g,F.X(m),F.y1+10,'已截斷',{class:'tk','text-anchor':'middle'})};
   curve(v.tgt[0],v.tgt[1],'tgt');curve(v.cor[0],v.cor[1],'ok','c');curve(v.cln[0],v.cln[1],'cl','t');
   tx(F.g,F.x0,12,'τ = '+fmt(c1.t,2)+' 的分佈',{class:'al'});
   if(final)setLabel(svg,`τ = ${fmt(c1.t,2)}：正確分佈平均 ${fmt(v.tgt[0],2)}、標準差 ${fmt(v.tgt[1],3)}；修正版相同；乾淨算子（本頁模型）平均 ${fmt(v.cln[0],2)}、標準差 ${fmt(v.cln[1],3)}；距離 W₂ = ${fmt(v.w2,3)}`)}
  /* fig 2 */
  {const box=$('#c1-f2'),svg=$('svg',box),W=widthOf(box,260,520),H=200;const pts=[];let ym=0;
   for(let i=0;i<=100;i++){const t=i/100,w=c1Values(g,t,r,c1.m,c1.s).w2;pts.push([t,w]);ym=Math.max(ym,w)}ym=Math.max(ym*1.15,1e-3);
   const F=frame(svg,{W,H,m:{l:48,r:10,t:16,b:36},x:[0,1],y:[0,ym],xt:[0,.25,.5,.75,1],yt:niceTicks(0,ym,3),yf:v=>fmt(v,ym<.01?4:ym<.1?3:2),xl:'noise level τ',yl:'W₂（乾淨算子 對 正確）'});
   el('path',{d:lp(pts.map(([t,w])=>[F.X(t),F.Y(w)])),class:'cl'},F.g);const xc=F.X(c1.t);el('line',{x1:xc,x2:xc,y1:F.y0,y2:F.y1,class:'cur'},F.g);
   el('circle',{cx:xc,cy:F.Y(v.w2),r:4.5,class:'acc-f'},F.g);
   if(final)setLabel(svg,`乾淨算子與正確分佈的距離隨 τ 下降：τ = 0 為 ${fmt(pts[0][1],4)}，τ = 1 為 0；目前 τ = ${fmt(c1.t,2)} 為 ${fmt(v.w2,4)}`)}
  $('#c1-ro').innerHTML=`<b>正確</b> 平均 ${fmt(v.tgt[0],4)}，標準差 ${fmt(v.tgt[1],4)}<br><b>修正版</b> 平均 ${fmt(v.cor[0],4)}，標準差 ${fmt(v.cor[1],4)}<br><b>乾淨算子（本頁模型）</b> 平均 ${fmt(v.cln[0],4)}，標準差 ${fmt(v.cln[1],4)}<br>W₂ = <span class="acc">${fmt(v.w2,4)}</span>　√(1−γ²) = ${fmt(v.c,4)}　1/(1−γ) = ${g===0?'1':fmt(1/(1-g),0)}${c1.scale?'<br>獎勵已乘 (1−γ)：r = '+fmt(r,4):''}`;
  if(final)announce('c1',`γ = ${c1.g4?'0.9999':fmt(g,3)}，τ = ${fmt(c1.t,2)}，W₂ = ${fmt(v.w2,4)}`)}
function initC1(){const rr=c1R;const gR=bindRange('c1-g','c1-go',x=>c1.g4?'0.9999':x.toFixed(3),x=>{c1.g=x;c1.g4=false;syncChips();schedule('c1',()=>renderC1(false))},()=>renderC1(true));
  const syncChips=()=>{$$('[data-g]',$('#c1')).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.g==='0.9999'?c1.g4:(!c1.g4&&Math.abs(+b.dataset.g-c1.g)<1e-9))));gR.sync()};
  $$('[data-g]',$('#c1')).forEach(b=>b.addEventListener('click',()=>{if(b.dataset.g==='0.9999'){c1.g4=true;gR.r.value=gR.r.max}else{c1.g4=false;c1.g=+b.dataset.g;gR.r.value=c1.g}syncChips();renderC1(true)}));
  bindRange('c1-t','c1-to',x=>x.toFixed(2),x=>{c1.t=x;schedule('c1',()=>renderC1(false))},()=>renderC1(true));
  bindRange('c1-r','c1-ro2',x=>fmt(x,1),x=>{c1.r=x;schedule('c1',()=>renderC1(false))},()=>renderC1(true));
  bindRange('c1-m','c1-mo',x=>fmt(x,1),x=>{c1.m=x;schedule('c1',()=>renderC1(false))},()=>renderC1(true));
  bindRange('c1-s','c1-so',x=>x.toFixed(2),x=>{c1.s=x;schedule('c1',()=>renderC1(false))},()=>renderC1(true));
  $('#c1-scale').addEventListener('change',e=>{c1.scale=e.target.checked;renderC1(true)});c1.syncChips=syncChips;renderC1(true)}

/* ---------- C2 ---------- */
const R2=[0,-1],P2=[[.7,.3],[.4,.6]],NAT=400,KMAX=80,KREF=120;
function c2Init(kind){if(kind==='zero')return[new Float64Array(NAT),new Float64Array(NAT)];const lo=[-10,-2],w=[10,8];
  return[0,1].map(x=>{const a=new Float64Array(NAT);for(let k=0;k<NAT;k++)a[k]=lo[x]+w[x]*(k+.5)/NAT;return a})}
function c2Noise(k){const r=mulberry32(1*1000+k),G=new Float64Array(4*NAT);for(let i=0;i<G.length;i++)G[i]=gauss(r);return G}
function c2Step(Z,g,tau,corr,G){const c=Math.sqrt(1-g*g),out=[];
  for(let x=0;x<2;x++){const xs=new Float64Array(2*NAT),ws=new Float64Array(2*NAT);
    for(let xp=0;xp<2;xp++){const w=P2[x][xp]/NAT,Zp=Z[xp],off=(x*2+xp)*NAT;
      for(let j=0;j<NAT;j++){const v=corr?tau*R2[x]+g*Zp[j]+(1-tau)*c*G[off+j]:R2[x]+g*Zp[j];xs[xp*NAT+j]=v;ws[xp*NAT+j]=w}}
    out.push({xs,ws})}return out}
function c2Project(m){const n=m.xs.length,idx=Array.from({length:n},(_,i)=>i);idx.sort((a,b)=>m.xs[a]-m.xs[b]||m.ws[a]-m.ws[b]);
  const out=new Float64Array(NAT);let cum=0,i=0;for(let k=0;k<NAT;k++){const q=(k+.5)/NAT;while(i<n-1&&cum+m.ws[idx[i]]<q-1e-15){cum+=m.ws[idx[i]];i++}out[k]=m.xs[idx[i]]}return out}
function c2Moments(g){const solve=(A,b)=>{const d=A[0][0]*A[1][1]-A[0][1]*A[1][0];return[(b[0]*A[1][1]-A[0][1]*b[1])/d,(A[0][0]*b[1]-b[0]*A[1][0])/d]};
  const Q=solve([[1-g*P2[0][0],-g*P2[0][1]],[-g*P2[1][0],1-g*P2[1][1]]],R2),PQ=[0,1].map(x=>P2[x][0]*Q[0]+P2[x][1]*Q[1]);
  const M2=solve([[1-g*g*P2[0][0],-g*g*P2[0][1]],[-g*g*P2[1][0],1-g*g*P2[1][1]]],[0,1].map(x=>R2[x]**2+2*g*R2[x]*PQ[x]));
  return{Q,Var:[0,1].map(x=>M2[x]-Q[x]**2)}}
function c2RefFrom(Zr,tau){const r=mulberry32(7);return[0,1].map(x=>{const a=new Float64Array(NAT);for(let j=0;j<NAT;j++)a[j]=tau*Zr[x][j]+(1-tau)*gauss(r);return a.sort()})}
const c2RefCache=new Map();
function* c2Gen(gi,ti){const g=gi/20,tau=ti/20;
  let Zr=c2RefCache.get(gi);
  if(!Zr){Zr=c2Init('zero');for(let k=1;k<=KREF;k++){Zr=c2Step(Zr,g,1,false,null).map(c2Project);if(k%10===0)yield['ref',k]}c2RefCache.set(gi,Zr)}
  const Z={cA:c2Init('zero'),cB:c2Init('spread'),kA:c2Init('zero'),kB:c2Init('spread')};
  const snap=[{c:Z.cA.map(a=>a.slice()),k:Z.kA.map(a=>a.slice())}],dist={c:[5],k:[5]},ratio={c:[NaN],k:[NaN]};
  for(let k=1;k<=KMAX;k++){const G=c2Noise(k);
    for(const [nm,A,B,corr] of [['c','cA','cB',true],['k','kA','kB',false]]){
      const before=Math.max(w1Sorted(Z[A][0],Z[B][0]),w1Sorted(Z[A][1],Z[B][1]));
      const mA=c2Step(Z[A],g,tau,corr,G),mB=c2Step(Z[B],g,tau,corr,G);
      const after=Math.max(w1Weighted(mA[0].xs,mA[0].ws,mB[0].xs,mB[0].ws),w1Weighted(mA[1].xs,mA[1].ws,mB[1].xs,mB[1].ws));
      Z[A]=mA.map(c2Project);Z[B]=mB.map(c2Project);ratio[nm].push(before>0?after/before:NaN);
      dist[nm].push(Math.max(w1Sorted(Z[A][0],Z[B][0]),w1Sorted(Z[A][1],Z[B][1])))}
    snap.push({c:Z.cA.map(a=>a.slice()),k:Z.kA.map(a=>a.slice())});if(k%4===0)yield['main',k]}
  return{g,tau,gi,ti,snap,dist,ratio,ref:c2RefFrom(Zr,tau),mom:c2Moments(g)}}
function c2Run(gi,ti){const it=c2Gen(gi,ti);let r;do{r=it.next()}while(!r.done);return r.value}
const c2={gi:18,ti:0,k:60,pair:0,res:null,gen:0,playing:0};
function c2Start(){const my=++c2.gen,it=c2Gen(c2.gi,c2.ti);$('#c2-figs').classList.add('busy');$('#c2-ro').classList.add('busy');$('#c2-k').disabled=true;
  const step=()=>{if(my!==c2.gen)return;const t0=performance.now();let r=it.next();while(!r.done&&performance.now()-t0<12)r=it.next();if(r.done){c2.res=r.value;$('#c2-st').textContent='完成';$('#c2-figs').classList.remove('busy');$('#c2-ro').classList.remove('busy');$('#c2-k').disabled=false;renderC2(true);return}
    $('#c2-st').textContent=r.value[0]==='ref'?`計算中：參考 ${r.value[1]}/${KREF}`:`計算中 ${r.value[1]}/${KMAX}`;requestAnimationFrame(step)};requestAnimationFrame(step)}
const stats=a=>{let m=0;for(const v of a)m+=v;m/=a.length;let s=0;for(const v of a)s+=(v-m)**2;return[m,Math.sqrt(s/a.length)]};
function renderC2(final){const R=c2.res;if(!R)return;const k=c2.k,x=c2.pair,S=R.snap[k],cA=S.c[x],kA=S.k[x],{Q,Var}=R.mom;
  if(!$('#c2-f1 svg')){$('#c2-f1').innerHTML='<svg role="img" aria-label="第 k 次迭代的分佈"></svg>';$('#c2-f2').innerHTML='<svg role="img" aria-label="兩個起點之間的距離"></svg>'}
  const exM=R.tau*Q[x],exS=Math.sqrt(R.tau**2*Var[x]+(1-R.tau)**2);
  /* fig 1: histograms */
  {const box=$('#c2-f1'),svg=$('svg',box),W=widthOf(box,260,520),H=220;let lo=Infinity,hi=-Infinity;
   for(const a of [cA,kA,R.ref[x]])for(const v of a){if(v<lo)lo=v;if(v>hi)hi=v}if(hi-lo<1e-9){lo-=.5;hi+=.5}
   const nb=40,bw=(hi-lo)/nb,hist=a=>{const h=new Array(nb).fill(0);for(const v of a)h[Math.min(nb-1,Math.floor((v-lo)/bw))]++;return h.map(c=>c/(a.length*bw))};
   const hc=hist(cA),hk=hist(kA);let ym=Math.max(...hc,...hk);if(R.tau===0)ym=Math.max(ym,npdf(0,0,1));ym*=1.12;
   const F=frame(svg,{W,H,m:{l:36,r:10,t:18,b:36},x:[lo,hi],y:[0,ym],xt:niceTicks(lo,hi,5),xf:tk,yt:[0],xl:'回報值（對 '+(x+1)+'）'});
   hc.forEach((h,i)=>{if(h>0)el('rect',{x:F.X(lo+i*bw),y:F.Y(h),width:Math.max(.5,F.X(lo+(i+1)*bw)-F.X(lo+i*bw)-.5),height:F.y0-F.Y(h),class:'okb'},F.g)});
   hk.forEach((h,i)=>{if(h>0)el('rect',{x:F.X(lo+i*bw),y:F.Y(h),width:Math.max(.5,F.X(lo+(i+1)*bw)-F.X(lo+i*bw)-.5),height:F.y0-F.Y(h),class:'clb'},F.g)});
   if(R.tau===0){const pts=[];for(let i=0;i<=120;i++){const z=lo+(hi-lo)*i/120;pts.push([F.X(z),F.Y(npdf(z,0,1))])}el('path',{d:lp(pts),class:'ref'},F.g)}
   const vl=(val,cls,lab,dy)=>{const xx=F.X(val);if(xx<F.x0||xx>F.x1)return;el('line',{x1:xx,x2:xx,y1:F.y0,y2:F.y1,class:cls},F.g);tx(F.g,Math.min(F.x1-4,Math.max(F.x0+4,xx)),F.y1-4+dy,lab,{class:'tk','text-anchor':xx>(F.x0+F.x1)/2?'end':'start'})};
   vl(exM,'ok','修正版的目標平均',0);if(Math.abs(Q[x]-exM)>1e-9)vl(Q[x],'cl','Q',12);
   tx(F.g,F.x0,12,'第 '+k+' 次迭代',{class:'al'});
   if(final)setLabel(svg,`第 ${k} 次迭代、對 ${x+1}：修正版平均 ${fmt(stats(cA)[0],2)}、標準差 ${fmt(stats(cA)[1],2)}；乾淨算子平均 ${fmt(stats(kA)[0],2)}、標準差 ${fmt(stats(kA)[1],2)}`)}
  /* fig 2: distances */
  {const box=$('#c2-f2'),svg=$('svg',box),W=widthOf(box,260,520),H=220,cl=v=>Math.max(1e-12,v);
   const ymin=cl(Math.min(...R.dist.c,...R.dist.k,5*R.g**KMAX));
   const F=frame(svg,{W,H,m:{l:52,r:10,t:18,b:36},x:[0,KMAX],y:[Math.log10(ymin),Math.log10(6)],xt:[0,20,40,60,80],yt:niceTicks(Math.log10(ymin),Math.log10(6),4).filter(v=>Number.isInteger(v)),yf:v=>'1e'+String(v).replace('-','−'),xl:'迭代次數 k',yl:'W₁ 上確界（對數）'});
   const P=a=>a.map((v,i)=>[F.X(i),F.Y(Math.log10(cl(v)))]);
   el('path',{d:lp(Array.from({length:KMAX+1},(_,i)=>[F.X(i),F.Y(Math.log10(cl(5*R.g**i)))])),class:'ref dot'},F.g);
   el('path',{d:lp(P(R.dist.c)),class:'ok'},F.g);el('path',{d:lp(P(R.dist.k)),class:'cl'},F.g);
   P(R.dist.c).forEach(([a,b],i)=>{if(i%10===0)el('circle',{cx:a,cy:b,r:3,class:'ok-f'},F.g)});
   P(R.dist.k).forEach(([a,b],i)=>{if(i%10===5)el('path',{d:`M${a} ${b-4}L${a+4} ${b+3}L${a-4} ${b+3}Z`,class:'cl-f'},F.g)});
   const xc=F.X(k);el('line',{x1:xc,x2:xc,y1:F.y0,y2:F.y1,class:'cur'},F.g);tx(F.g,F.x1,F.y1+10,'點線：5γ^k',{class:'tk','text-anchor':'end'});
   if(final)setLabel(svg,`兩個起點之間的 W₁ 上確界隨迭代下降，速率不慢於 γ = ${R.g.toFixed(2)} 的幾何級數`)}
  const [mc,sc]=stats(cA),[mk,sk]=stats(kA),last=k>0?R.ratio.c[k]:NaN;let mx=-Infinity;for(let i=1;i<=k;i++)mx=Math.max(mx,R.ratio.c[i],R.ratio.k[i]);
  $('#c2-ro').innerHTML=`<b>對 ${x+1}，第 ${k} 次迭代</b><br>修正版　平均 ${fmt(mc,4)}，標準差 ${fmt(sc,4)}（精確 ${fmt(exM,4)}，${fmt(exS,4)}）<br>乾淨算子（本頁模型）　平均 ${fmt(mk,4)}，標準差 ${fmt(sk,4)}（精確 ${fmt(Q[x],4)}，${fmt(Math.sqrt(Var[x]),4)}）<br>修正版與 τZ+(1−τ)G 參考樣本的 W₁ = ${fmt(w1Sorted(cA,R.ref[x]),4)}<br>最後一步比值 ${k>0?fmt(last,6):'—'}；1…k 的最大比值 ${k>0?fmt(mx,6):'—'} <span class="acc">≤ γ = ${R.g.toFixed(2)}</span>`;
  if(final)announce('c2',`第 ${k} 次迭代：修正版平均 ${fmt(mc,2)}、標準差 ${fmt(sc,2)}；乾淨算子平均 ${fmt(mk,2)}`)}
function initC2(){bindRange('c2-g','c2-go',i=>(i/20).toFixed(2),i=>{c2.gi=i},i=>{c2.gi=i;c2Start()});
  bindRange('c2-t','c2-to',i=>(i/20).toFixed(2),i=>{c2.ti=i},i=>{c2.ti=i;c2Start()});
  bindRange('c2-k','c2-ko',i=>String(i),i=>{c2.k=i;schedule('c2',()=>renderC2(false))},()=>renderC2(true));
  $$('[data-pair]',$('#c2')).forEach(b=>b.addEventListener('click',()=>{c2.pair=+b.dataset.pair;$$('[data-pair]',$('#c2')).forEach(o=>o.setAttribute('aria-pressed',String(o===b)));renderC2(true)}));
  $('#c2-play').addEventListener('click',()=>{if(!c2.res)return;const kr=$('#c2-k');if(RM){c2.k=KMAX}else{c2.k=0;const my=++c2.playing;const tick=()=>{if(my!==c2.playing)return;c2.k++;kr.value=c2.k;$('#c2-ko').textContent=c2.k;renderC2(c2.k===KMAX);if(c2.k<KMAX)requestAnimationFrame(tick)};requestAnimationFrame(tick)}kr.value=c2.k;$('#c2-ko').textContent=c2.k;renderC2(true)});
  c2Start()}

/* ---------- C3 ---------- */
function c3Values(g,tau,r,m1,s1,mode){const c=Math.sqrt(1-g*g),V=gaussVel(m1,s1,tau);
  const ax=[tau*r+g*tau*m1,g*V.sig,(1-tau)*c],uV0=V.k*(tau*m1)+V.b;
  const ay=[(mode==='taur'?tau*r:r)+g*uV0,g*V.k*V.sig,mode==='sign'?c:-c];
  const varx=ax[1]**2+ax[2]**2,cov=ax[1]*ay[1]+ax[2]*ay[2],slope=cov/varx,Qv=gaussVel(r+g*m1,g*s1,tau);
  return{slope,icpt:ay[0]-slope*ax[0],kQ:Qv.k,bQ:Qv.b,sQ:Qv.sig,varx,ax,ay,kV:V.k,bV:V.b,sV:V.sig}}
const C3S=(()=>{const r=mulberry32(3),o=[];for(let i=0;i<300;i++){const z=gauss(r),e=gauss(r);o.push([z,e])}return o})();
function ols(pts){const n=pts.length;let mx=0,my=0;for(const p of pts){mx+=p[0];my+=p[1]}mx/=n;my/=n;let sxx=0,sxy=0;for(const p of pts){sxx+=(p[0]-mx)**2;sxy+=(p[0]-mx)*(p[1]-my)}const b=sxy/sxx;return[b,my-b*mx]}
const c3={g:.9,t:.5,r:-1,m:-2,s:.6,z:1,e:-1,mode:'ok'};
function c3Data(){const o=c3Values(c3.g,c3.t,c3.r,c3.m,c3.s,c3.mode);const pts=C3S.map(([z,e])=>[o.ax[0]+o.ax[1]*z+o.ax[2]*e,o.ay[0]+o.ay[1]*z+o.ay[2]*e]);return{o,pts}}
function renderC3(final){const {o,pts}=c3Data(),[bo,ao]=ols(pts);
  /* fig 1 */
  {const box=$('#c3-f1'),svg=$('svg',box),W=widthOf(box,260,520),H=220;const path=z0=>Array.from({length:51},(_,i)=>{const t=i/50,s=Math.sqrt(t*t*c3.s*c3.s+(1-t)**2);return[t,t*c3.m+s*z0]});
   let lo=Infinity,hi=-Infinity;for(const z0 of [-2.5,2.5])for(const [,y] of path(z0)){lo=Math.min(lo,y);hi=Math.max(hi,y)}
   const F=frame(svg,{W,H,m:{l:40,r:10,t:18,b:36},x:[0,1],y:[lo,hi],xt:[0,.25,.5,.75,1],yt:niceTicks(lo,hi,4),yf:tk,xl:'noise level τ'});
   for(let i=0;i<=10;i++)el('path',{d:lp(path(-2+.4*i).map(([t,y])=>[F.X(t),F.Y(y)])),class:'pathf'},F.g);
   el('path',{d:lp(path(c3.z).map(([t,y])=>[F.X(t),F.Y(y)])),class:'pathb'},F.g);
   const xc=F.X(c3.t),zt=c3.t*c3.m+o.sV*c3.z;el('line',{x1:xc,x2:xc,y1:F.y0,y2:F.y1,class:'cur'},F.g);el('circle',{cx:xc,cy:F.Y(zt),r:5,class:'sel'},F.g);
   tx(F.g,F.x0,12,'下一狀態的 V flow 路徑',{class:'al'});
   if(final)setLabel(svg,`下一狀態的 V flow：11 條路徑從 τ = 0 的雜訊走到 τ = 1 的回報；選取樣本在 τ = ${fmt(c3.t,3)} 的位置 ${fmt(zt,3)}`)}
  /* fig 2 */
  {const box=$('#c3-f2'),svg=$('svg',box),W=widthOf(box,260,520),H=240;let xl=Infinity,xh=-Infinity,yl=Infinity,yh=-Infinity;
   for(const [x,y] of pts){xl=Math.min(xl,x);xh=Math.max(xh,x);yl=Math.min(yl,y);yh=Math.max(yh,y)}
   const pad=(xh-xl)*.05||.5;xl-=pad;xh+=pad;for(const f of [x=>o.kQ*x+o.bQ,x=>o.slope*x+o.icpt])for(const x of [xl,xh]){yl=Math.min(yl,f(x));yh=Math.max(yh,f(x))}
   const F=frame(svg,{W,H,m:{l:44,r:10,t:30,b:36},x:[xl,xh],y:[yl,yh],xt:niceTicks(xl,xh,4),xf:tk,yt:niceTicks(yl,yh,4),yf:tk,xl:'回歸輸入點 x'});
   for(const [x,y] of pts)el('circle',{cx:F.X(x),cy:F.Y(y),r:2,class:'scat'},F.g);
   const line=(a,b,cls)=>el('line',{x1:F.X(xl),y1:F.Y(a*xl+b),x2:F.X(xh),y2:F.Y(a*xh+b),class:cls},F.g);
   line(bo,ao,'ref dot');line(o.kQ,o.bQ,'ok');line(o.slope,o.icpt,'ref');
   const xs=o.ax[0]+o.ax[1]*c3.z+o.ax[2]*c3.e,ys=o.ay[0]+o.ay[1]*c3.z+o.ay[2]*c3.e;
   el('line',{x1:F.X(xs),x2:F.X(xs),y1:F.Y(ys),y2:F.Y(o.kQ*xs+o.bQ),class:'cur'},F.g);el('rect',{x:F.X(xs)-5,y:F.Y(ys)-5,width:10,height:10,class:'sel'},F.g);
   const same=Math.abs(o.slope-o.kQ)<1e-9&&Math.abs(o.icpt-o.bQ)<1e-9;tx(F.g,F.x0,12,same?'目標（縱軸）：條件平均與 u_Q 重合':'目標（縱軸）：條件平均偏離 u_Q',{class:'al'});
   if(final)setLabel(svg,`回歸資料：條件平均斜率 ${fmt(o.slope,4)}、截距 ${fmt(o.icpt,4)}；u_Q 斜率 ${fmt(o.kQ,4)}、截距 ${fmt(o.bQ,4)}；${same?'重合':'不重合'}`)}
  const zt=c3.t*c3.m+o.sV*c3.z,uV=o.kV*zt+o.bV,xs=o.ax[0]+o.ax[1]*c3.z+o.ax[2]*c3.e,ys=o.ay[0]+o.ay[1]*c3.z+o.ay[2]*c3.e,Ex=o.ax[0];
  $('#c3-ro').innerHTML=`<b>選取樣本</b>　z（τ 層）= ${fmt(zt,4)}，u_V = ${fmt(uV,4)}，輸入點 x = ${fmt(xs,4)}，目標 y = ${fmt(ys,4)}，u_Q(x) = ${fmt(o.kQ*xs+o.bQ,4)}<br><b>條件平均</b>　斜率 ${fmt(o.slope,4)}，截距 ${fmt(o.icpt,4)}<br><b>u_Q</b>　斜率 ${fmt(o.kQ,4)}，截距 ${fmt(o.bQ,4)}<br>300 點最小平方線　斜率 ${fmt(bo,4)}，截距 ${fmt(ao,4)}<br>在 x = E[x] 處：條件平均 − u_Q = <span class="acc">${fmt((o.slope-o.kQ)*Ex+o.icpt-o.bQ,4)}</span>`;
  if(final)announce('c3',`條件平均斜率 ${fmt(o.slope,4)}，u_Q 斜率 ${fmt(o.kQ,4)}`)}
function initC3(){const on=(id,o,f,key)=>bindRange(id,o,f,x=>{c3[key]=x;schedule('c3',()=>renderC3(false))},()=>renderC3(true));
  on('c3-g','c3-go',x=>x.toFixed(3),'g');on('c3-t','c3-to',x=>x.toFixed(3),'t');on('c3-r','c3-ro2',x=>fmt(x,1),'r');on('c3-m','c3-mo',x=>fmt(x,1),'m');on('c3-s','c3-so',x=>x.toFixed(2),'s');on('c3-z','c3-zo',x=>fmt(x,2),'z');on('c3-e','c3-eo',x=>fmt(x,2),'e');
  $$('[data-mode]',$('#c3')).forEach(b=>b.addEventListener('click',()=>{c3.mode=b.dataset.mode;$$('[data-mode]',$('#c3')).forEach(o=>o.setAttribute('aria-pressed',String(o===b)));renderC3(true)}));renderC3(true)}

/* ---------- C4 ---------- */
const Q4=[-3,-2,-1.2,-.5],PB4=[.4,.3,.2,.1];
function implicitPolicy(Q,pb,mu){const V=expectile(Q,pb,mu),w=Q.map(q=>q>=V?mu:1-mu),Z=pb.reduce((s,p,i)=>s+p*w[i],0),pi=pb.map((p,i)=>p*w[i]/Z);return{V,w,pi,EV:pi.reduce((s,p,i)=>s+p*Q[i],0)}}
function u4(m,s,z,t){const sig2=t*t*s*s+(1-t)**2;return m+(t*s*s-(1-t))/sig2*(z-t*m)}
function envelope(ms,ss,J=101,S=400,rec){let tot=0;const ta=ms.map(()=>0),paths=[];
  for(let j=0;j<J;j++){const z0=invNorm((j+.5)/J);let z=z0;const za=ms.map(()=>z0),keep=rec&&rec.includes(j),pth=keep?{max:[[0,z0]],a:ms.map(()=>[[0,z0]])}:null;
    for(let i=0;i<S;i++){const t=i/S;let mx=-Infinity;for(let a=0;a<ms.length;a++)mx=Math.max(mx,u4(ms[a],ss[a],z,t));z+=mx/S;for(let a=0;a<ms.length;a++)za[a]+=u4(ms[a],ss[a],za[a],t)/S;
      if(keep&&(i+1)%8===0){pth.max.push([(i+1)/S,z]);za.forEach((v,a)=>pth.a[a].push([(i+1)/S,v]))}}
    tot+=z;za.forEach((v,a)=>ta[a]+=v);if(keep)paths.push(pth)}
  return{mean:tot/J,per:ta.map(v=>v/J),paths}}
const c4={mu:.9,m:[-2,-2.2],s:[.3,1.2]};
function renderC4a(final){const r=implicitPolicy(Q4,PB4,c4.mu),mean=PB4.reduce((s,p,i)=>s+p*Q4[i],0);
  const box=$('#c4-f1'),svg=$('svg',box),W=widthOf(box,260,520),H=230;
  const F=frame(svg,{W,H,m:{l:40,r:14,t:30,b:40},x:[-3.4,-0.1],y:[0,1],xt:[-3,-2,-1.2,-.5],xf:v=>fmt(v,1),yt:[0,.25,.5,.75,1],yf:v=>fmt(v,2),xl:'動作的 Q 值',yl:'機率'});
  Q4.forEach((q,i)=>{const x=F.X(q);el('rect',{x:x-13,y:F.Y(PB4[i]),width:12,height:F.y0-F.Y(PB4[i]),class:'beh-o',fill:'none',stroke:'currentColor'},F.g).style.color='var(--ink-2)';
    el('rect',{x:x+1,y:F.Y(r.pi[i]),width:12,height:F.y0-F.Y(r.pi[i]),class:'acc-f'},F.g);tx(F.g,x,Math.min(F.Y(Math.max(PB4[i],r.pi[i]))-6,F.y0-6),'w='+fmt(r.w[i],3),{class:'tk','text-anchor':'middle'})});
  const vl=(v,cls,lab,row)=>{const x=F.X(v);el('line',{x1:x,x2:x,y1:F.y0,y2:F.y1-6,class:cls},F.g);tx(F.g,x,F.y1-10-row*12,lab,{class:'tk','text-anchor':'middle'})};
  vl(mean,'ref','平均',0);vl(r.V,'acc','V',1);vl(-.5,'ref dot','最大',0);
  if(final)setLabel(svg,`μ = ${fmt(c4.mu,3)}：V = ${fmt(r.V,4)}；隱式策略機率 ${r.pi.map(p=>fmt(p,3)).join('、')}`);
  $('#c4a-ro').innerHTML=`<b>μ = ${fmt(c4.mu,3)}</b>（圖中外框：π_β，實心：π^μ）<br>V（μ-expectile）= <span class="acc">${fmt(r.V,4)}</span><br>E_{π^μ}[Q] = ${fmt(r.EV,4)}<br>平均（μ = 0.5）= ${fmt(mean,4)}；資料內最大 = −0.5<br>π^μ = ${r.pi.map(p=>fmt(p,4)).join('、')}`;
  if(final)announce('c4',`μ = ${fmt(c4.mu,3)}，V = ${fmt(r.V,4)}`)}
function renderC4b(final){const E=envelope(c4.m,c4.s,101,400,[10,30,50,70,90]),mq=Math.max(...c4.m);
  const box=$('#c4-f2'),svg=$('svg',box),W=widthOf(box,260,520),H=230;let lo=Infinity,hi=-Infinity;E.paths.forEach(p=>[p.max,...p.a].forEach(a=>a.forEach(([,y])=>{lo=Math.min(lo,y);hi=Math.max(hi,y)})));
  const F=frame(svg,{W,H,m:{l:40,r:10,t:18,b:36},x:[0,1],y:[lo,hi],xt:[0,.25,.5,.75,1],yt:niceTicks(lo,hi,4),yf:tk,xl:'noise level τ'});
  E.paths.forEach(p=>{el('path',{d:lp(p.a[0].map(([t,y])=>[F.X(t),F.Y(y)])),class:'ok',style:'stroke-width:1.5'},F.g);el('path',{d:lp(p.a[1].map(([t,y])=>[F.X(t),F.Y(y)])),class:'cl',style:'stroke-width:1.5'},F.g);
    el('path',{d:lp(p.max.map(([t,y])=>[F.X(t),F.Y(y)])),class:'acc'},F.g);p.max.forEach(([t,y],i)=>{if(i%10===0)el('circle',{cx:F.X(t),cy:F.Y(y),r:3,class:'acc-f'},F.g)})});
  tx(F.g,F.x0,12,'五個起點分位數的路徑',{class:'al'});
  if(final)setLabel(svg,`逐點取 max 的路徑終點平均 ${fmt(E.mean,4)}，max Q = ${fmt(mq,1)}，差距 ${fmt(E.mean-mq,4)}`);
  $('#c4b-ro').innerHTML=`max_a Q = ${fmt(mq,4)}<br>逐點取 max 的終點平均 = ${fmt(E.mean,4)}<br>差距 = <span class="acc">${fmt(E.mean-mq,4)}</span><br>各動作自己的終點平均 = ${E.per.map(v=>fmt(v,4)).join('、')}`}
function initC4(){const mu=bindRange('c4-mu','c4-muo',x=>x.toFixed(3),x=>{c4.mu=x;syncMu();schedule('c4a',()=>renderC4a(false))},()=>renderC4a(true));
  const syncMu=()=>$$('[data-mu]',$('#c4')).forEach(b=>b.setAttribute('aria-pressed',String(Math.abs(+b.dataset.mu-c4.mu)<1e-9)));
  $$('[data-mu]',$('#c4')).forEach(b=>b.addEventListener('click',()=>{c4.mu=+b.dataset.mu;mu.r.value=c4.mu;mu.sync();syncMu();renderC4a(true)}));
  [['c4-m1','c4-m1o','m',0,1],['c4-m2','c4-m2o','m',1,1],['c4-s1','c4-s1o','s',0,2],['c4-s2','c4-s2o','s',1,2]].forEach(([id,o,k,i,d])=>bindRange(id,o,x=>fmt(x,d),x=>{c4[k][i]=x;schedule('c4b',()=>renderC4b(false))},()=>renderC4b(true)));
  renderC4a(true);renderC4b(true)}

/* ---------- static figures ---------- */
function renderF1(){const box=$('#f1-fig'),svg=$('svg',box),W=widthOf(box,300,720),narrow=W<600,rowH=narrow?84:64,H=rowH*3+30;
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const rows=[['FlowCritic 等：只在終點',[1],[]],['Value Flows：每層乾淨方程',[.25,.5,.75,1],[0]],['Value Forcing：每層修正方程',[0,.25,.5,.75,1],[]]];
  const x0=narrow?20:220,x1=W-24,X=t=>x0+(x1-x0)*t;
  rows.forEach(([lab,dots,crosses],r)=>{const y=r*rowH+(narrow?50:36);
    tx(svg,narrow?x0:12,narrow?y-26:y+5,lab,{class:'lab'});el('line',{x1:x0,x2:x1,y1:y,y2:y,class:'axl'},svg);
    [0,.5,1].forEach(t=>{el('line',{x1:X(t),x2:X(t),y1:y,y2:y+5,class:'axl'},svg);tx(svg,X(t),y+20,'τ = '+t,{class:'tk','text-anchor':'middle'})});
    dots.forEach(t=>el('circle',{cx:X(t),cy:y,r:6,class:'ink-f'},svg));
    crosses.forEach(t=>{el('line',{x1:X(t)-6,x2:X(t)+6,y1:y-6,y2:y+6,class:'cross'},svg);el('line',{x1:X(t)-6,x2:X(t)+6,y1:y+6,y2:y-6,class:'cross'},svg);tx(svg,X(t)+10,y-10,'不正確（τ = 0）',{class:'tk'})})})}
const FIG2={'cube-triple':[[92,13],[115,16],[212,14],[295,16],[204,15]],'cube-quadruple':[[57,6],[179,6],[203,9],[289,11],[184,9]],'puzzle-4x4':[[20,5],[22,5],[125,6],[134,7],[118,6]],'humanoidmaze-giant':[[25,3],[13,2],[127,3],[462,3],[122,3]]};
const F2N=['Oracle','VF','FlowCritic','Value Flows','PCBF'];
function renderF2(){const host=$('#f2-fig');host.innerHTML='';const W0=widthOf(host,280,960),W=W0>=736?Math.floor((W0-16)/2):W0,H=190,narrow=W<380;
  Object.entries(FIG2).forEach(([dom,vals],di)=>{const d=document.createElement('div');d.className='fig';host.appendChild(d);
    const svg=el('svg',{role:'img','aria-label':dom+'：'+vals.map((v,i)=>F2N[i]+' '+v[0]+'±'+v[1]).join('、')},d);
    const mx=Math.max(...vals.map(v=>v[0]+v[1]));const F=frame(svg,{W,H,m:{l:40,r:8,t:26,b:narrow?52:34},x:[0,5],y:[0,mx*1.18],yt:niceTicks(0,mx*1.18,3),yf:v=>String(v)});
    const defs=el('defs',null,svg);[['h1','M0 6L6 0'],['h2','M3 0L3 6'],['h3','M0 3L6 3']].forEach(([id,p])=>{const pt=el('pattern',{id:id+di,width:6,height:6,patternUnits:'userSpaceOnUse'},defs);el('path',{d:p,stroke:'currentColor','stroke-width':1.3},pt)});
    tx(F.g,F.x0,14,dom,{class:'lab'});
    vals.forEach(([v,e],i)=>{const bw=(F.x1-F.x0)/5*.6,x=F.X(i+.5)-bw/2,y=F.Y(v);
      const rc=el('rect',{x,y,width:bw,height:F.y0-y},F.g);
      if(i===0)rc.setAttribute('class','ora');else if(i===1)rc.setAttribute('class','vfb');else{rc.setAttribute('fill',`url(#h${i-1}${di})`);rc.setAttribute('class','hatch');rc.setAttribute('stroke','currentColor');rc.setAttribute('stroke-width','1')}
      tx(F.g,x+bw/2,y-5,v+'±'+e,{class:'tk','text-anchor':'middle'});
      const t=tx(F.g,x+bw/2,F.y0+14,F2N[i],{class:'tk','text-anchor':narrow?'end':'middle'});if(narrow)t.setAttribute('transform',`rotate(-35 ${x+bw/2} ${F.y0+14})`)})})}

/* ---------- tau axis (hero) ---------- */
function taxis(node){node.innerHTML='<div class="kx-track"><div class="kx-line"></div></div>';const tr=node.firstChild;const pos=t=>t*100;
  [[0,'0','l'],[.5,'0.5',''],[1,'1','r']].forEach(([t,v,c])=>{const s=document.createElement('span');s.className='kx-tick '+c;s.style.left=pos(t)+'%';s.innerHTML=`<b>τ</b> ${v}`;tr.appendChild(s)});
  [[0,'N(0,1) 雜訊',0],[.5,'每一層都施加 Bellman 一致性',1],[1,'回報分佈',0]].forEach(([t,l,a])=>{const d=document.createElement('div');d.className='kx-mark'+(a?' acc':'');d.style.left=pos(t)+'%';d.innerHTML=`<span class="kx-dot"></span><span class="kx-stem"></span><span class="kx-lab ${t>.85?'r':t<.15?'l':'c'}">${l}</span>`;tr.appendChild(d)});
  layoutKAxis(node)}
function layoutKAxis(node){const tr=node.firstChild;if(!tr)return;const labs=$$('.kx-lab',tr);labs.forEach(l=>{l.style.bottom='14px';l.previousSibling.style.height='8px'});
  const items=labs.map(l=>({l,r:l.getBoundingClientRect()})).sort((a,b)=>a.r.left-b.r.left);const rows=[];let mx=0;
  items.forEach(it=>{let r=0;while(rows[r]!==undefined&&it.r.left<rows[r]+8)r++;rows[r]=it.r.right;mx=Math.max(mx,r);it.l.style.bottom=(14+r*28)+'px';it.l.previousSibling.style.height=(8+r*28)+'px'});
  tr.style.marginTop=(mx*28+40)+'px'}

/* ---------- timeline ---------- */
const TL_GROUP={'1707.06887':['dist','distributional-rl-return-distributions'],'1710.10044':['dist','distributional-rl-return-distributions'],'1806.06923':['dist','distributional-rl-return-distributions'],
 '2210.02747':['fm','flow-matching-velocity-target'],'2209.03003':['fm','flow-matching-velocity-target'],
 '2503.09817':['end','flow-based-return-critics'],'2509.06863':['end','flow-based-return-critics'],'2509.23087':['end','flow-based-return-critics'],'2510.22686':['end','flow-based-return-critics'],'2605.08515':['end','flow-based-return-critics'],
 '2510.07650':['path','flow-bellman-operator-all-noise-levels'],'2605.08253':['path','flow-bellman-operator-all-noise-levels'],
 '2110.06169':['imp','expectile-regression-value-learning'],'2304.10573':['imp','value-forcing-vf-ivf-algorithms'],'2611.02160':['me','value-forcing-lineage-hub']};
const TL_NAME={dist:'分布式 RL',fm:'Flow matching',end:'以 flow 表示價值（終點目標）',path:'路徑上的 Bellman 一致性',imp:'隱式價值學習與抽樣挑選',me:'本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.map((e,i)=>({e,i})).sort((a,b)=>(+a.e.year||0)-(+b.e.year||0)||a.i-b.i).map(o=>o.e);
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];if(g[0]==='other')console.warn('timeline: ungrouped',e.arxiv);used.add(g[0]);const slug=g[1];const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${slug?`<a href="${slug}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
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
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t)=>Math.abs(a-b)<=t+1e-12;
  {const r=mulberry32(1);ok(r()===0.6270739405881613,'mulberry32')}
  ok(near(invNorm(.975),1.959963985,1e-8)&&near(invNorm(.01),-2.326347874,1e-8),'invNorm');
  ok(near(w1Weighted([0,1],[.5,.5],[.5,1.5],[.5,.5]),.5,1e-12),'w1Weighted');
  /* C1 */
  {const T=[[.9,0,0,1,-1,.9,1.005],[.9,.25,-.7,.7584,-1.45,.6843,.7536],[.9,.5,-1.4,.5483,-1.9,.5031,.502],[.9,.75,-2.1,.42,-2.35,.4056,.2504],[.9,1,-2.8,.45,-2.8,.45,0],
    [.999,0,0,1,-1,.999,1],[.999,.25,-.7495,.7603,-1.4995,.7596,.75],[.999,.5,-1.499,.5589,-1.999,.5585,.5],[.999,.75,-2.2485,.4504,-2.4985,.4502,.25],[.999,1,-2.998,.4995,-2.998,.4995,0]];
   T.forEach(([g,t,tm,ts,cm,cs,w])=>{const v=c1Values(g,t,-1,-2,.5);ok(near(v.tgt[0],tm,5e-4)&&near(v.tgt[1],ts,5e-4)&&near(v.cln[0],cm,5e-4)&&near(v.cln[1],cs,5e-4)&&near(v.w2,w,5e-4),`C1 g=${g} tau=${t}`)});
   [[.9,0,.1414],[.999,0,.0014],[.999,.5,.0007]].forEach(([g,t,w])=>ok(near(c1Values(g,t,(1-g)*-1,-2,.5).w2,w,5e-4),`C1 scaled g=${g} tau=${t}`));
   ok(near(c1Values(.9999,.3,-1,-2,.5).c,.0141,5e-4),'C1 c at .9999');
   const r=mulberry32(42);let worst=0;for(let i=0;i<20;i++){const v=c1Values(r(),r(),r()*4-2,r()*6-4,.05+r()*2);worst=Math.max(worst,Math.abs(v.tgt[0]-v.cor[0]),Math.abs(v.tgt[1]-v.cor[1]))}ok(worst<1e-12,'C1 corrected == target '+worst);
   $('[data-g="0.9999"]',$('#c1')).click();ok(c1.g4&&$('#c1-ro').textContent.includes('0.0141'),'C1 gamma .9999 chip');
   const gr=$('#c1-g');gr.value=.9;gr.dispatchEvent(new Event('input'));ok(!c1.g4,'C1 slider clears chip flag');renderC1(true)}
  /* C2 */
  {const G=c2Noise(1);ok(G.length===1600&&near(G[0],-0.060209008271024565,1e-12)&&near(G[400],-1.29139894185053,1e-12)&&near(G[800],-0.855905775769241,1e-12),'C2 noise table k=1');
   ok(18/20===0.9&&(18/20).toFixed(2)==='0.90','C2 gamma index');
   const m=c2Moments(.9);ok(near(m.Q[0],-3.6986,5e-4)&&near(m.Q[1],-5.0685,5e-4)&&near(Math.sqrt(m.Var[0]),1.3255,5e-4)&&near(Math.sqrt(m.Var[1]),1.3481,5e-4),'C2 exact moments');
   const cases=[[18,0,[-.0237,.9998,.0528,-3.6924,1.3262],[-.0159,.957,.0628,-5.0665,1.3537],.899927],[18,10,[-1.8592,.8266,.0445,-3.6924,1.3262],[-2.5407,.8217,.0307,-5.0665,1.3537],.899947],[14,0,[-.0299,1.0025,.0508,-.8866,.5758],[-.0179,.9504,.0583,-2.1523,.5991],.699993]];
   cases.forEach(([gi,ti,a,b,mr])=>{const R=c2Run(gi,ti),S=R.snap[60];[a,b].forEach((e,x)=>{const [mc,sc]=stats(S.c[x]),[mk,sk]=stats(S.k[x]);
     ok(near(mc,e[0],2e-3)&&near(sc,e[1],2e-3)&&near(w1Sorted(S.c[x],R.ref[x]),e[2],2e-3)&&near(mk,e[3],2e-3)&&near(sk,e[4],2e-3),`C2 gi=${gi} ti=${ti} pair${x+1}`);
     const ex=R.tau*R.mom.Q[x],es=Math.sqrt(R.tau**2*R.mom.Var[x]+(1-R.tau)**2);ok(near(mc,ex,.05)&&near(sc,es,.06),`C2 vs exact gi=${gi} ti=${ti} pair${x+1}`)});
     let mx=0;for(let i=1;i<=60;i++)mx=Math.max(mx,R.ratio.c[i],R.ratio.k[i]);ok(mx<=R.g+1e-9&&near(mx,mr,1e-6),`C2 max ratio gi=${gi} ti=${ti}: ${mx}`);
     if(gi===18&&ti===0){ok(near(R.ratio.c[1],.718,5e-4)&&near(R.ratio.c[2],.8051,5e-4)&&near(R.ratio.c[3],.8482,5e-4)&&near(R.ratio.c[10],.8594,5e-4),'C2 ratios k=1,2,3,10')}});
   const R1=c2Run(18,20);let same=true;for(let x=0;x<2;x++)for(let j=0;j<NAT;j++)if(R1.snap[60].c[x][j]!==R1.snap[60].k[x][j])same=false;ok(same,'C2 tau=1 operators identical')}
  /* C3 */
  {const T=[[.9,0,[-1,-2.8],[-.62,-2.8],[-1,-1.8]],[.9,.25,[-1.166,-3.6162],[-.6752,-3.2726],[-1.166,-2.8662]],[.9,.5,[-1.0969,-4.3357],[-.5085,-3.5119],[-1.0969,-3.8357]],[.9,.75,[-.1382,-3.0902],[.2812,-2.2095],[-.1382,-2.8402]],[.99,.5,[-.9568,-4.4056],[-.8979,-4.3179],[-.9568,-3.9056]]];
   T.forEach(([g,t,a,b,c])=>{[['ok',a],['sign',b],['taur',c]].forEach(([md,e])=>{const v=c3Values(g,t,-1,-2,.6,md);ok(near(v.slope,e[0],5e-4)&&near(v.icpt,e[1],5e-4),`C3 g=${g} tau=${t} ${md}`)})});
   let worst=0;[0,.25,.5,.75,.999].forEach(t=>{const v=c3Values(.9,t,-1,-2,.6,'ok');worst=Math.max(worst,Math.abs(v.slope-v.kQ),Math.abs(v.icpt-v.bQ),Math.abs(Math.sqrt(v.varx)-v.sQ))});ok(worst<1e-12,'C3 cond mean == u_Q '+worst);
   const o=c3Values(.9,.5,-1,-2,.6,'ok'),[b,a]=ols(C3S.map(([z,e])=>[o.ax[0]+o.ax[1]*z+o.ax[2]*e,o.ay[0]+o.ay[1]*z+o.ay[2]*e]));ok(near(b,-1.0736,2e-3)&&near(a,-4.3024,2e-3),'C3 OLS default')}
  /* C4 */
  {[[.5,-2.09],[.7,-1.769],[.8,-1.5579],[.9,-1.2971],[.99,-.6472],[.999,-.5158]].forEach(([mu,V])=>{const r=implicitPolicy(Q4,PB4,mu);ok(near(r.V,V,5e-4)&&Math.abs(r.V-r.EV)<1e-12,'C4a mu='+mu)});
   const r9=implicitPolicy(Q4,PB4,.9);ok([.1176,.0882,.5294,.2647].every((p,i)=>near(r9.pi[i],p,5e-4)),'C4a pi mu=.9');
   [[[-2,-2.2],[.3,1.2],-1.7213],[[-2,-2],[.5,.5],-2],[[-2,-3],[.3,.3],-2],[[-2,-2.5],[.2,1.5],-1.6536]].forEach(([m,s,e])=>{const E=envelope(m,s);ok(near(E.mean,e,3e-3)&&E.per.every((v,i)=>near(v,m[i],1e-9)),'C4b '+m+' '+s)})}
  /* no combining characters in dynamic text */
  {let bad=0;$$('svg text, .readout, .status').forEach(n=>{if(/[̀-ͯ]/.test(n.textContent))bad++});$$('[aria-label],[aria-valuetext]').forEach(n=>{if(/[̀-ͯ]/.test((n.getAttribute('aria-label')||'')+(n.getAttribute('aria-valuetext')||'')))bad++});ok(bad===0,'no combining chars: '+bad)}
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ---------- boot ---------- */
initTheme();initToc();initC1();initC2();initC3();initC4();renderF1();renderF2();renderTimeline();
$$('.kaxis').forEach(taxis);
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderC1(false);renderC2(false);renderC3(false);renderC4a(false);renderC4b(false);renderF1();renderF2();$$('.kaxis').forEach(layoutKAxis)})});
document.fonts&&document.fonts.ready.then(()=>$$('.kaxis').forEach(layoutKAxis));
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))selfTest();
})();
</script>
</body>
</html>
