(()=>{
'use strict';
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
const TEST=/[?&]test\b/.test(location.search);
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
function el(tag,a,p){const e=document.createElementNS(NS,tag);if(a)for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function tx(p,x,y,s,a){const t=el('text',Object.assign({x,y},a||{}),p);t.textContent=s;return t}
const lin=(d0,d1,r0,r1)=>v=>r0+(v-d0)/(d1-d0)*(r1-r0);
function lp(pts){let s='';for(let i=0;i<pts.length;i++)s+=(i?'L':'M')+pts[i][0].toFixed(1)+' '+pts[i][1].toFixed(1);return s}
function widthOf(box,min,max){return Math.max(min,Math.min(max,Math.floor(box.clientWidth||max)))}
function svgIn(box,W,H){const svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);return svg}
const f4=v=>Number.isFinite(v)?v.toFixed(4):'—',f3=v=>Number.isFinite(v)?v.toFixed(3):'—',f2=v=>v.toFixed(2);
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function segInit(box,opts,cur,label,onPick){box.textContent='';opts.forEach(o=>{const b=document.createElement('button');b.type='button';b.setAttribute('role','radio');
    b.textContent=label?label(o):String(o);b.dataset.v=String(o);box.appendChild(b)});
  const sync=v=>$$('button',box).forEach(b=>{const on=b.dataset.v===String(v);b.setAttribute('aria-checked',String(on));b.setAttribute('aria-pressed',String(on));b.tabIndex=on?0:-1});sync(cur);
  box.addEventListener('click',e=>{const b=e.target.closest('button');if(!b||b.disabled)return;const v=opts.find(o=>String(o)===b.dataset.v);sync(v);onPick(v)});
  box.addEventListener('keydown',e=>{const bs=$$('button',box),i=bs.findIndex(b=>b.tabIndex===0);let j=i;if(e.key==='ArrowRight'||e.key==='ArrowDown')j=(i+1)%bs.length;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')j=(i-1+bs.length)%bs.length;else return;e.preventDefault();bs[j].focus();bs[j].click()});
  return sync}
function dl(box,rows){box.innerHTML=rows.map(([k,v,c])=>`<dt>${k}</dt><dd${c?' class="acc"':''}>${v}</dd>`).join('')}
const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function hatchDef(svg){const d=el('defs',null,svg),p=el('pattern',{id:'hatch',width:4,height:4,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},d);el('rect',{width:4,height:4,class:'hatch-bg'},p);el('line',{x1:0,y1:0,x2:0,y2:4,class:'hatch-l'},p);
  const q=el('pattern',{id:'hatchA',width:6,height:6,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},d);el('line',{x1:0,y1:0,x2:0,y2:6,stroke:'currentColor','stroke-width':1.5,class:'hz'},q);}

/* ================= shared math (mirror verify/core.py, chain.js, topk.js) ================= */
const onehot=(i,K)=>{const v=new Array(K).fill(0);v[i]=1;return v};
const prior=(kind,K)=>kind==='mask'?onehot(K-1,K):new Array(K).fill(1/K);
const qMarg=(x,a,pi)=>x.map((xi,j)=>a*xi+(1-a)*pi[j]);
function post(zt,x,at,as,pi){const ats=at/as,K=x.length,num=new Array(K);let S=0;
  for(let j=0;j<K;j++){const lik=(1-ats)*pi[zt]+(j===zt?ats:0);num[j]=lik*(as*x[j]+(1-as)*pi[j]);S+=num[j];}
  if(S<1e-300)return onehot(zt,K);for(let j=0;j<K;j++)num[j]/=S;return num;}
const q0t=(zt,x,at,pi)=>post(zt,x,at,1,pi);
function psiTrue(zt,x,at,as,kap,pi){const a=post(zt,x,at,as,pi),b=qMarg(x,as,pi);return a.map((ai,j)=>kap*ai+(1-kap)*b[j]);}
function psiTheta(zt,x,at,as,kap,pi){const a=post(zt,x,at,as,pi),c=post(zt,x,at,1,pi);return a.map((ai,j)=>kap*ai+(1-kap)*(as*c[j]+(1-as)*pi[j]));}
function marginalAfter(x,at,as,kapOf,pi){const qt=qMarg(x,at,pi),K=x.length,out=new Array(K).fill(0);
  for(let z=0;z<K;z++){if(qt[z]===0)continue;const row=psiTrue(z,x,at,as,kapOf(z),pi);for(let j=0;j<K;j++)out[j]+=qt[z]*row[j];}return out;}
const sigmaMax=(at,as)=>at<=0?1:Math.min(1,(1-as)/at);
const kappaFromSigma=(sig,as)=>1-as<=0?1:1-sig/(1-as);
function kappaSched(kind,t,at,as,p){if(kind==='ancestral')return 1;if(kind==='window')return(p.toff<=t&&t<=p.ton)?p.kap:1;
  if(kind==='rescale')return kappaFromSigma(p.eta*sigmaMax(at,as),as);if(kind==='cap')return kappaFromSigma(Math.min(p.eta,sigmaMax(at,as)),as);return 1;}
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function catSample(r,p){const u=r();let c=0;for(let i=0;i<p.length;i++){c+=p[i];if(u<c)return i;}for(let i=p.length-1;i>=0;i--)if(p[i]>0)return i;}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2);}
function erfcc(x){const z=Math.abs(x),t=1/(1+0.5*z);
  const r=t*Math.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))));
  return x>=0?r:2-r;}
function logPhi(x){if(x>-5)return Math.log(0.5*erfcc(-x/Math.SQRT2));const x2=x*x;return -x2/2-Math.log(-x)-0.5*Math.log(2*Math.PI)+Math.log1p(-1/x2+3/(x2*x2)-15/(x2*x2*x2));}
const PA=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00],
 PB=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01],
 PC=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00],
 PD=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00];
function ppfTail(p){let x;if(p<0.02425){const q=Math.sqrt(-2*Math.log(p));x=(((((PC[0]*q+PC[1])*q+PC[2])*q+PC[3])*q+PC[4])*q+PC[5])/((((PD[0]*q+PD[1])*q+PD[2])*q+PD[3])*q+1);}
  else{const q=p-0.5,r=q*q;x=(((((PA[0]*r+PA[1])*r+PA[2])*r+PA[3])*r+PA[4])*r+PA[5])*q/(((((PB[0]*r+PB[1])*r+PB[2])*r+PB[3])*r+PB[4])*r+1);}
  const e=0.5*erfcc(-x/Math.SQRT2)-p,u=e*Math.sqrt(2*Math.PI)*Math.exp(x*x/2);return x-u/(1+x*u/2);}
function normPpf(p){if(p<=0)return -Infinity;if(p>=1)return Infinity;return p>0.5?-ppfTail(1-p):ppfTail(p);}
const lse=a=>{const m=Math.max(...a);let s=0;for(const v of a)s+=Math.exp(v-m);return m+Math.log(s);};
function topkZeroMean(r,N,k,sig){let P=0;const out=[];for(let l=N;l>N-k;l--){let u=r();if(!(u>0))u=1e-300;P+=Math.log(u)/l;
  const om=-Math.expm1(P);out.push(om<0.5?-sig*normPpf(om):sig*normPpf(Math.exp(P)));}return out;}
function logMu(c,sig,tau){return sig*sig/(2*tau*tau)-logPhi(c/sig)+logPhi((c-sig*sig/tau)/sig);}
function efficient(r,K,k,at,tau){const sig=Math.sqrt(1-at*at);let V=topkZeroMean(r,K-1,k,sig);const wt=at+sig*gauss(r);let logs,cs,pos=-1;
  if(wt>V[k-1]){const thr=k>=2?V[k-2]:V[k-1];let rk=0;for(const v of V)if(v>wt)rk++;V=V.slice(0,rk).concat([wt],V.slice(rk,k-1));
    logs=V.map(v=>v/tau);if(K-k>0)logs.push(Math.log(K-k)+logMu(thr,sig,tau));cs=2;pos=rk;}
  else{logs=V.map(v=>v/tau);logs.push(wt/tau);if(K-k-1>0)logs.push(Math.log(K-k-1)+logMu(V[k-1],sig,tau));cs=1;}
  const logZ=lse(logs);return{vals:V,wt,case:cs,pos,lam:V.map(v=>Math.exp(v/tau-logZ)),logZ};}
function naiveW(r,K,at){const sig=Math.sqrt(1-at*at),w=new Float64Array(K);for(let i=0;i<K;i++)w[i]=sig*gauss(r);w[0]+=at;return w;}
function ztildeFrom(w,k,at,tau){const sig=Math.sqrt(1-at*at),K=w.length,top=[];
  for(let i=1;i<K;i++){const v=w[i];if(top.length<k||v>top[top.length-1]){let j=top.length<k?top.length:k-1;top[j]=v;while(j>0&&top[j]>top[j-1]){const q=top[j];top[j]=top[j-1];top[j-1]=q;j--;}}}
  let logs,tv;if(w[0]>top[k-1]){const thr=k>=2?top[k-2]:top[k-1];tv=top.slice(0,k-1).concat([w[0]]);logs=tv.map(v=>v/tau);if(K-k>0)logs.push(Math.log(K-k)+logMu(thr,sig,tau));}
  else{tv=top.slice();logs=tv.map(v=>v/tau);logs.push(w[0]/tau);if(K-k-1>0)logs.push(Math.log(K-k-1)+logMu(top[k-1],sig,tau));}
  return{logZt:lse(logs),logZtop:lse(tv.map(v=>v/tau))};}
function logZexact(w,tau){let m=-Infinity;for(const v of w)if(v>m)m=v;let s=0;for(const v of w)s+=Math.exp((v-m)/tau);return m/tau+Math.log(s);}

/* ================= token-cell figures (hero, S1) ================= */
function cellRow(svg,x0,y,cells,sz){cells.forEach((c,i)=>{const x=x0+i*(sz+4);
  if(c===0)el('rect',{x,y,width:sz,height:sz,rx:2,class:'t-ok'},svg);else if(c===1)el('rect',{x:x+.6,y:y+.6,width:sz-1.2,height:sz-1.2,rx:2,class:'t-bad'},svg);else el('rect',{x:x+.6,y:y+.6,width:sz-1.2,height:sz-1.2,rx:2,class:'t-mask'},svg)})}
function drawHero(){const box=$('#hero-fig'),W=widthOf(box,300,520),gap=W<420?12:22,lab=W<420?74:92,sz=Math.min(13,Math.floor((W-lab-2*gap)/24-4)),gw=8*(sz+4),H=2*(sz+30)+18,svg=svgIn(box,W,H);hatchDef(svg);
  const rows=[['不重新抽樣',[[2,2,0,2,1,2,2,2],[0,0,0,2,1,0,2,0],[0,0,0,0,1,0,0,0]]],['重新抽樣',[[2,2,0,2,1,2,2,2],[0,0,0,2,2,0,2,0],[0,0,0,0,0,0,0,0]]]];
  rows.forEach(([name,snaps],r)=>{const y=12+r*(sz+30);tx(svg,0,y+sz-1,name,{class:'lab'});snaps.forEach((s,j)=>{cellRow(svg,lab+j*(gw+gap),y,s,sz);if(j<2)tx(svg,lab+j*(gw+gap)+gw+gap/2-2,y+sz-1,'→',{class:'lab','text-anchor':'middle'})})});
  ['早期','中期','結束'].forEach((s,j)=>tx(svg,lab+j*(gw+gap)+gw/2,H-2,s,{class:'tk','text-anchor':'middle'}))}
function drawS1(){const box=$('#s1-fig'),W=widthOf(box,300,720),two=W>=600,sz=13,gw=8*(sz+4),lab=56,H=two?120:236,svg=svgIn(box,W,H);hatchDef(svg);
  const sets=[['MDM（[MASK]）',[[0,0,0,0,0,0,0,0],[0,2,0,0,2,0,2,0],[2,2,2,2,2,2,2,2]]],['USDM（均勻）',[[0,0,0,0,0,0,0,0],[0,1,0,0,1,0,1,0],[1,1,0,1,1,1,1,1]]]];
  sets.forEach(([name,rows],g)=>{const ox=two?g*(lab+gw+40):0,oy=two?0:g*118;tx(svg,ox,oy+14,name,{class:'labm'});
    rows.forEach((r,i)=>{const y=oy+26+i*(sz+14);tx(svg,ox,y+sz-1,['t = 0','t = 0.5','t = 1'][i],{class:'tk'});cellRow(svg,ox+lab,y,r,sz)})})}

/* ================= W1 ================= */
const w1={prior:'unif',K:4,at:.3,as:.6,k:.5,dep:false};const LET='ABCDEFGH';
const tokName=(j,K,kind)=>kind==='mask'&&j===K-1?'[M]':LET[j];
function w1Calc(){const{K,at,as,k,dep}=w1,pi=prior(w1.prior,K),x=onehot(0,K),kapOf=z=>dep&&z!==0?1:k,qt=qMarg(x,at,pi),qs=qMarg(x,as,pi);
  const rows=[];for(let z=0;z<K;z++)rows.push(qt[z]>0?psiTrue(z,x,at,as,kapOf(z),pi):null);
  const mg=marginalAfter(x,at,as,kapOf,pi);let md=0;mg.forEach((v,j)=>md=Math.max(md,Math.abs(v-qs[j])));
  let ret=0;for(let z=0;z<K;z++)if(rows[z])ret+=qt[z]*rows[z][z];return{pi,qt,qs,rows,mg,md,ret}}
function w1Draw(){const o=w1Calc(),{K}=w1,box=$('#w1-heat'),narrow=box.clientWidth<360;box.style.gridTemplateColumns=`${narrow?"5.4em":"6em"} repeat(${K},minmax(0,1fr))`;box.style.setProperty('--n',K);
  let h='<div class="hh"></div>'+Array.from({length:K},(_,j)=>`<div class="hh">${tokName(j,K,w1.prior)}</div>`).join('');
  for(let z=0;z<K;z++){h+=`<div class="hr">${tokName(z,K,w1.prior)} · ${o.qt[z].toFixed(narrow?2:3)}</div>`;
    if(!o.rows[z]){h+=`<div class="hc off">此列不會出現</div>`;continue}
    o.rows[z].forEach((p,j)=>{h+=`<div class="hc${z===j?' hi':''}" style="background:color-mix(in oklab,var(--c-par) ${(p*100).toFixed(0)}%,var(--surface));color:${p>=.5?'var(--surface)':'var(--ink)'}">${p.toFixed(narrow?2:3)}</div>`})}
  box.innerHTML=h;
  const bb=$('#w1-bars'),W=widthOf(bb,260,560),H=150,svg=svgIn(bb,W,H),L=34,R=W-8,T=10,B=H-24,y=lin(0,1,B,T),gw=(R-L)/K,bw=Math.min(36,gw*.5);
  [0,.5,1].forEach(v=>{el('line',{x1:L,x2:R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-6,y(v)+4,v.toFixed(1),{class:'tk','text-anchor':'end'})});
  for(let j=0;j<K;j++){const cx=L+gw*(j+.5);el('rect',{x:cx-bw/2,y:y(o.mg[j]),width:bw,height:B-y(o.mg[j]),class:'par-f'},svg);
    el('rect',{x:cx-bw/2-4,y:y(o.qs[j]),width:bw+8,height:B-y(o.qs[j]),fill:'none',stroke:cssv('--ink'),'stroke-width':2},svg);tx(svg,cx,H-6,tokName(j,K,w1.prior),{class:'tk','text-anchor':'middle'})}
  const mdS=o.md<1e-12?'0（浮點誤差內）':f4(o.md);
  dl($('#w1-ro'),[['最大差',mdS,o.md>=1e-12],['保留率',f4(o.ret)]]);
  $('#w1-say').textContent=`κ = ${f2(w1.k)}${w1.dep?'（只用在 z_t = A 的列）':''}：保留率 ${f4(o.ret)}，邊際與 q_s 最大差 ${mdS}。`;
  $('#w1-bars svg').setAttribute('aria-label',`邊際與 q_s 長條，最大差 ${mdS}`);
  ['K','at','as','k'].forEach(n=>$('#w1-'+n+'o').textContent=n==='K'?w1.K:f2(w1[n]));
  announce('w1',$('#w1-say').textContent)}
function linkAlpha(pre,st,draw){const sa=$('#'+pre+'-at'),ss=$('#'+pre+'-as');
  sa.addEventListener('input',()=>{st.at=+sa.value;const mn=Math.min(1,+(st.at+.01).toFixed(2));if(st.as<mn){st.as=mn;ss.value=mn;announce(pre,`α_s 已調為 ${f2(mn)}`)}ss.min=mn;draw()});
  ss.addEventListener('input',()=>{st.as=Math.max(+ss.value,Math.min(1,+(st.at+.01).toFixed(2)));ss.value=st.as;draw()})}
function initW1(){segInit($('#w1-prior'),['unif','mask'],w1.prior,k=>k==='unif'?'均勻':'[MASK]',k=>{w1.prior=k;w1Draw()});
  $('#w1-K').addEventListener('input',e=>{w1.K=+e.target.value;w1Draw()});linkAlpha('w1',w1,w1Draw);
  const ks=$('#w1-k');ks.addEventListener('input',()=>{w1.k=+ks.value;w1Draw()});
  $('#w1-k1').addEventListener('click',()=>{w1.k=1;ks.value=1;w1Draw()});$('#w1-k0').addEventListener('click',()=>{w1.k=0;ks.value=0;w1Draw()});
  $('#w1-dep').addEventListener('change',e=>{w1.dep=e.target.checked;w1Draw()});w1Draw()}

/* ================= W2 / W2b ================= */
const DEN={unif:{right:[.85,.05,.05,.05],weak:[.5,.3,.15,.05],wrong:[0,.6,.3,.1]},mask:{right:[.9,.05,.05,0],wrong:[0,.7,.3,0]}};
const DEN_NAME={right:'猜對',weak:'偏錯',wrong:'全錯'};
const w2={prior:'unif',z:1,den:'wrong',at:.3,as:.6,k:.5,sig:.04};let w2zSync=null,w2dSync=null;
function w2Parts(kap){const K=4,pi=prior(w2.prior,K),carry=w2.prior==='mask'&&w2.z!==3,x=carry?onehot(w2.z,K):DEN[w2.prior][w2.den];
  const pred=post(w2.z,x,w2.at,w2.as,pi),c=q0t(w2.z,x,w2.at,pi),corr=c.map((v,j)=>w2.as*v+(1-w2.as)*pi[j]),mix=pred.map((v,j)=>kap*v+(1-kap)*corr[j]);return{pred,corr,mix,c,carry}}
function w2Draw(){const o=w2Parts(w2.k),o1=w2Parts(1),K=4,box=$('#w2-fig'),W=widthOf(box,280,560),H=180,svg=svgIn(box,W,H),L=34,R=W-8,T=10,B=H-24,y=lin(0,1,B,T),gw=(R-L)/K,bw=Math.min(18,gw/4.2);
  [0,.5,1].forEach(v=>{el('line',{x1:L,x2:R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-6,y(v)+4,v.toFixed(1),{class:'tk','text-anchor':'end'})});
  for(let j=0;j<K;j++){const cx=L+gw*(j+.5);[[o.pred,'par-f',-1.15],[o.corr,'perp-f',0],[o.mix,'acc-f',1.15]].forEach(([v,c,d])=>el('rect',{x:cx+d*bw-bw/2,y:y(v[j]),width:bw,height:B-y(v[j]),class:c},svg));
    tx(svg,cx,H-6,tokName(j,K,w2.prior)+(j===0?'（正確）':''),{class:'tk','text-anchor':'middle'})}
  const floor=(1-w2.k)*(1-w2.as)/K;if(w2.prior==='unif'&&floor>0){el('line',{x1:L,x2:R,y1:y(floor),y2:y(floor),class:'ref'},svg);tx(svg,L+4,y(floor)-5,'下限 '+f4(floor),{class:'tk halo'})}
  const rows=[['P(z_s = A)',o1.mix[0],o.mix[0]]];if(w2.prior==='mask')rows.push(['P(z_s = [M])',o1.mix[3],o.mix[3]]);else rows.push(['下限 (1−κ)(1−α_s)/K',0,floor]);
  $('#w2-tab').innerHTML=`<caption>讀數</caption><thead><tr><th>量</th><th class="num">κ = 1</th><th class="num">κ = ${f2(w2.k)}</th></tr></thead><tbody>`+rows.map(r=>`<tr><td>${r[0]}</td><td class="num">${f4(r[1])}</td><td class="num">${f4(r[2])}</td></tr>`).join('')+'</tbody>';
  $('#w2-carry').textContent=o.carry?'z_t 已解開：依 MDLM 的 carry-over 參數化，x_θ = e_{z_t}，去噪器選單不適用。':'';
  $$('#w2-den button').forEach(b=>{b.disabled=o.carry||!(b.dataset.v in DEN[w2.prior])});
  $('#w2-say').textContent=`目前設定：P(z_s = A) 由 κ = 1 的 ${f4(o1.mix[0])} 變為 κ = ${f2(w2.k)} 的 ${f4(o.mix[0])}`+(w2.prior==='mask'?`；P(z_s = [M]) = ${f4(o.mix[3])}。`:`；下限 ${f4(floor)}。`);
  ['at','as','k'].forEach(n=>$('#w2-'+n+'o').textContent=f2(w2[n]));announce('w2',$('#w2-say').textContent);w2bDraw()}
function w2Prior(k){w2.prior=k;if(k==='mask'){w2.z=3;if(!(w2.den in DEN.mask))w2.den='wrong'}else{w2.z=1}
  w2zSync=segInit($('#w2-z'),k==='mask'?[3,1]:[0,1,2,3],w2.z,v=>k==='mask'?(v===3?'[M]':'B（已解開）'):LET[v],v=>{w2.z=v;w2Draw()});
  w2dSync=segInit($('#w2-den'),k==='mask'?['right','wrong']:['right','weak','wrong'],w2.den,v=>DEN_NAME[v],v=>{w2.den=v;w2Draw()});w2Draw()}
function w2bDraw(){const sm=sigmaMax(w2.at,w2.as),s=$('#w2b-s');s.max=sm.toFixed(4);if(w2.sig>sm){w2.sig=+sm.toFixed(4);announce('w2b',`σ 已調為 ${f4(w2.sig)}`)}s.value=w2.sig;
  const box=$('#w2b-fig'),W=widthOf(box,280,560),H=190,svg=svgIn(box,W,H);hatchDef(svg);const L=44,R=W-10,T=12,B=H-30,kmin=Math.min(0,kappaFromSigma(sm,w2.as)),x=lin(0,sm||1,L,R),y=lin(kmin,1,B,T),bd=1-w2.as;
  if(bd<sm){const r=el('rect',{x:x(bd),y:T,width:R-x(bd),height:B-T,fill:'url(#hatchA)',class:'neg'},svg);r.style.color=cssv('--accent');r.style.opacity=.35;tx(svg,R-4,T+14,'κ < 0：超出 §3 的範圍',{class:'tk','text-anchor':'end'})}
  const yt=[1,.5,0];if(kmin<0)yt.push(+kmin.toFixed(2));yt.forEach(v=>{el('line',{x1:L,x2:R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-6,y(v)+4,v.toFixed(v%1?2:0),{class:'tk','text-anchor':'end'})});
  [0,sm].forEach(v=>tx(svg,x(v),H-12,v.toFixed(2),{class:'tk','text-anchor':'middle'}));tx(svg,(L+R)/2,H-1,'σ',{class:'tk','text-anchor':'middle'});tx(svg,4,T+4,'κ',{class:'tk'});
  el('path',{d:lp([[x(0),y(1)],[x(sm),y(kappaFromSigma(sm,w2.as))]]),class:'acc'},svg);const kc=kappaFromSigma(w2.sig,w2.as);el('circle',{cx:x(w2.sig),cy:y(kc),r:5,class:'dot-c'},svg);
  dl($('#w2b-ro'),[['σ_max',f4(sm)],['1 − α_s',f4(bd)],['κ',f4(kc),true],['Ψ([M] | z_t = B)',f4(Math.max(0,(1-kc)*(1-w2.as)))]]);$('#w2b-so').textContent=f2(w2.sig);
  $('#w2b-fig svg').setAttribute('aria-label',`σ = ${f2(w2.sig)} 時 κ = ${f4(kc)}`);announce('w2b',`σ = ${f2(w2.sig)}，κ = ${f4(kc)}`)}
function initW2(){segInit($('#w2-prior'),['unif','mask'],w2.prior,k=>k==='unif'?'均勻':'[MASK]',w2Prior);linkAlpha('w2',w2,w2Draw);
  $('#w2-k').addEventListener('input',e=>{w2.k=+e.target.value;w2Draw()});$('#w2b-s').addEventListener('input',e=>{w2.sig=+e.target.value;w2bDraw()});w2Prior('unif')}

/* ================= κ mini axes (S7, S8) ================= */
function drawKx(id,marks){const box=$('#'+id),W=widthOf(box,300,720),H=74,svg=svgIn(box,W,H),L=16,R=W-16,x=lin(.85,1,L,R),Y=40;
  el('line',{x1:L,x2:R,y1:Y,y2:Y,class:'ink'},svg);[.85,.9,.95,1].forEach(v=>{el('line',{x1:x(v),x2:x(v),y1:Y-4,y2:Y+4,class:'ink'},svg);tx(svg,x(v),Y+18,v.toFixed(2),{class:'tk','text-anchor':'middle'})});
  tx(svg,L,12,'κ 軸（局部：0.85–1）',{class:'tk'});tx(svg,x(1),12,'1 = ancestral',{class:'tk','text-anchor':'end'});
  marks.forEach(m=>{if(m.band){el('rect',{x:x(m.band[0]),y:Y-7,width:x(m.band[1])-x(m.band[0]),height:14,class:'band'},svg);tx(svg,(x(m.band[0])+x(m.band[1]))/2,H-2,m.t,{class:'tk','text-anchor':'middle'})}
    else{el('circle',{cx:x(m.v),cy:Y,r:5,class:'dot-c'},svg);tx(svg,x(m.v),H-2-(m.dy||0),m.t,{class:'tk','text-anchor':m.a||'middle'})}})}
function drawKxAll(){const n=widthOf($('#kx-lang'),300,720)<520;drawKx('kx-lang',[{band:[.9,.95],t:n?'OWT rescale η=0.05：κ≈0.90–0.95':'OWT：rescale η = 0.05，κ 約 0.90–0.95，隨 t 變化'}]);
  drawKx('kx-img',[{v:.95,t:n?'Duo 0.95':'CIFAR-10 Duo：0.95（只在 [t_off, t_on] 內）',a:'end'},{v:.99,t:n?'MDLM 0.99':'MDLM：0.99',a:'end',dy:0}])}

/* ================= W3 toy chain (mirror verify/chain.js) ================= */
const W3_TS=[8,16,32,64,128,256,512],W3_R=32,W3_L=32,W3_KT=8;
const w3={prior:'mask',sch:'window',kap:.9,ton:.6,toff:.1,eta:.05,em:.5,token:0,res:null,trace:null,row:-1,busy:false};
const bufA=new Float64Array(9),bufC=new Float64Array(9),bufP=new Float64Array(9),bufX=new Float64Array(9);
function postInto(out,zt,x,K,at,as,pi){const ats=at/as;let S=0;for(let j=0;j<K;j++){const lik=(1-ats)*pi[zt]+(j===zt?ats:0);out[j]=lik*(as*x[j]+(1-as)*pi[j]);S+=out[j];}
  if(S<1e-300){for(let j=0;j<K;j++)out[j]=j===zt?1:0;return}for(let j=0;j<K;j++)out[j]/=S}
function catSampleBuf(r,p,K){const u=r();let c=0;for(let i=0;i<K;i++){c+=p[i];if(u<c)return i;}for(let i=K-1;i>=0;i--)if(p[i]>0)return i;}
function runChain(kind,sched,T,p,seed,emax,trace){const K=kind==='mask'?W3_KT+1:W3_KT,pi=prior(kind,K),m=K-1,r=mulberry32(seed),L=W3_L;
  let z=new Array(L);if(kind==='mask')z.fill(m);else for(let l=0;l<L;l++)z[l]=catSample(r,pi);const nz=new Array(L);
  const tr=trace?{z:new Uint8Array(T*L),info:[]}:null;
  for(let i=T;i>=1;i--){const t=i/T,s=(i-1)/T,at=1-t,as=1-s;let c=0;for(let l=0;l<L;l++)if(z[l]===l%W3_KT)c++;c/=L;
    const eps=emax-(emax-0.02)*c,kap=kappaSched(sched,t,at,as,p);
    for(let l=0;l<L;l++){if(kind==='mask'&&z[l]!==m){for(let j=0;j<K;j++)bufX[j]=j===z[l]?1:0}else{for(let j=0;j<K;j++)bufX[j]=0;for(let j=0;j<W3_KT;j++)bufX[j]=eps/(W3_KT-1);bufX[l%W3_KT]=1-eps}
      postInto(bufA,z[l],bufX,K,at,as,pi);postInto(bufC,z[l],bufX,K,at,1,pi);for(let j=0;j<K;j++)bufP[j]=kap*bufA[j]+(1-kap)*(as*bufC[j]+(1-as)*pi[j]);nz[l]=catSampleBuf(r,bufP,K)}
    for(let l=0;l<L;l++)z[l]=nz[l];
    if(tr){const row=T-i;let nm=0;for(let l=0;l<L;l++){tr.z[row*L+l]=z[l];if(kind==='mask'&&z[l]===m)nm++}tr.info.push({t,c,eps,kap,nm})}}
  let e=0;for(let l=0;l<L;l++)if(z[l]!==l%W3_KT)e++;return trace?{err:e/L,tr}:e/L}
const w3params=()=>({kap:w3.kap,ton:w3.ton,toff:w3.toff,eta:w3.eta});
const w3cache=new Map();
function errCurve(kind,sched,p,emax,onProg,onDone,sync){const key=[kind,sched,p.kap,p.ton,p.toff,p.eta,emax].join('|');if(w3cache.has(key)){onDone(w3cache.get(key));return}
  const jobs=[];W3_TS.forEach((T,ti)=>{for(let k=0;k<W3_R;k++)jobs.push([ti,T,1000+k])});const acc=new Array(W3_TS.length).fill(0);let n=0;const tok=w3.token;
  const step=()=>{const t0=performance.now();while(n<jobs.length&&(sync||performance.now()-t0<12)){const[ti,T,sd]=jobs[n++];acc[ti]+=runChain(kind,sched,T,p,sd,emax)/W3_R}
    if(n<jobs.length){if(tok!==w3.token)return;onProg&&onProg(jobs[n][1]);setTimeout(step,0)}else{const res=acc;w3cache.set(key,res);if(sync||tok===w3.token)onDone(res)}};
  step()}
function w3Run(sync){w3.token++;w3.busy=true;$('#w3').classList.add('busy');$('#w3-status').textContent='計算中…';
  const p=w3params(),out={};let left=2;const fin=()=>{if(--left)return;w3.busy=false;$('#w3').classList.remove('busy');w3.res=out;
    w3.trace=runChain(w3.prior,w3.sch,innerWidth<600?32:64,p,1000,w3.em,true);w3.row=-1;w3Draw();$('#w3-status').textContent='完成：結果為目前設定';announce('w3','計算完成。'+$('#w3-say').textContent)};
  errCurve(w3.prior,'ancestral',p,w3.em,null,r=>{out.anc=r;fin()},sync);
  errCurve(w3.prior,w3.sch,p,w3.em,T=>{$('#w3-status').textContent=`計算中…（T = ${T}）`},r=>{out.sel=r;fin()},sync)}
function w3Stale(){w3.token++;if(w3.res){$('#w3-status').innerHTML='<span class="stale">結果已過期，請按「計算」</span>'}w3.busy=false;$('#w3').classList.remove('busy');w3Kap()}
function w3Draw(){w3Kap();const r=w3.res;if(!r)return;const box=$('#w3-err'),W=widthOf(box,280,560),H=214,svg=svgIn(box,W,H),L=40,R=W-78,T=10,B=H-40,x=lin(3,9,L,R),y=lin(0,.5,B,T);
  [0,.1,.2,.3,.4,.5].forEach(v=>{el('line',{x1:L,x2:R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-6,y(v)+4,v.toFixed(1),{class:'tk','text-anchor':'end'})});
  W3_TS.forEach((T,i)=>tx(svg,x(i+3),B+16,String(T),{class:'tk','text-anchor':'middle'}));tx(svg,(L+R)/2,H-2,'步數 T（log₂ 刻度）',{class:'tk','text-anchor':'middle'});
  const pa=r.anc.map((v,i)=>[x(i+3),y(Math.min(.5,v))]),ps=r.sel.map((v,i)=>[x(i+3),y(Math.min(.5,v))]);
  el('path',{d:lp(pa),class:'ref'},svg);pa.forEach(([a,b])=>el('circle',{cx:a,cy:b,r:3.5,class:'mk-o'},svg));
  if(w3.sch!=='ancestral'){el('path',{d:lp(ps),class:'acc-s'},svg);ps.forEach(([a,b])=>el('rect',{x:a-4,y:b-4,width:8,height:8,class:'dot-c'},svg))}
  const sep=Math.abs(pa[6][1]-ps[6][1])<14&&w3.sch!=='ancestral',ya=pa[6][1]+(sep&&pa[6][1]<=ps[6][1]?-6:4),yb=ps[6][1]+(sep&&pa[6][1]<=ps[6][1]?12:4);tx(svg,pa[6][0]+8,ya,'ancestral',{class:'tk'});if(w3.sch!=='ancestral')tx(svg,ps[6][0]+8,yb,w3.sch,{class:'tk'});
  const sname=w3.sch==='window'?`window（κ = ${f2(w3.kap)}，t ∈ [${f2(w3.toff)}, ${f2(w3.ton)}]）`:w3.sch==='rescale'?`rescale（η = ${w3.eta.toFixed(3)}）`:'ancestral';
  $('#w3-say').textContent=`目前設定（${w3.prior==='mask'?'[MASK]':'均勻'}，${sname}，ε_max = ${f2(w3.em)}）：T = 8 → 512，錯誤率 ${f3(r.sel[0])} → ${f3(r.sel[6])}；ancestral ${f3(r.anc[0])} → ${f3(r.anc[6])}。`;
  $('#w3-err svg').setAttribute('aria-label',$('#w3-say').textContent);w3Trace()}
function w3Kap(){const box=$('#w3-kap'),W=widthOf(box,280,560),H=160,svg=svgIn(box,W,H);hatchDef(svg);const T=128,p=w3params(),pts=[],cap=[];
  for(let i=T;i>=1;i--){const t=i/T,at=1-t,as=1-(i-1)/T;pts.push([t,kappaSched(w3.sch,t,at,as,p)]);cap.push([t,kappaSched('cap',t,at,as,p)])}
  const showCap=w3.sch==='rescale',kmin=Math.min(...pts.map(q=>q[1]),...(showCap?cap.map(q=>q[1]):[1])),lo=Math.min(.8,Math.floor((kmin-.05)*10)/10),L=40,R=W-14,Tp=26,B=H-40,x=lin(1,0,L,R),y=lin(lo,1.02,B,Tp);
  [1,lo].concat(kmin<0&&lo<0?[0]:[]).forEach(v=>{el('line',{x1:L,x2:R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-6,y(v)+4,v.toFixed(1),{class:'tk','text-anchor':'end'})});
  if(kmin<0){const r=el('rect',{x:L,y:y(0),width:R-L,height:B-y(0),fill:'url(#hatchA)'},svg);r.style.color=cssv('--accent');r.style.opacity=.3;tx(svg,R-4,B-4,'κ < 0',{class:'tk','text-anchor':'end'})}
  [1,.5,0].forEach(v=>tx(svg,x(v),B+16,'t = '+v,{class:'tk','text-anchor':v===1?'start':v===0?'end':'middle'}));tx(svg,R,H-2,'取樣方向 →',{class:'tk','text-anchor':'end'});
  if(showCap)el('path',{d:lp(cap.map(([t,k])=>[x(t),y(k)])),class:'ref dashp'},svg);el('path',{d:lp(pts.map(([t,k])=>[x(t),y(k)])),class:'acc-s'},svg);
  const mn=Math.min(...pts.map(q=>q[1]));tx(svg,L,12,`最小 κ = ${f4(mn)}`+(showCap?`；cap（虛線）最小 ${f4(Math.min(...cap.map(q=>q[1])))}`:''),{class:'tk'})}
function w3Trace(){const tr=w3.trace;if(!tr)return;const cv=$('#w3-trace'),L=W3_L,T=tr.tr.info.length,box=$('#w3-trace-box'),avail=Math.max(200,box.clientWidth||300),cs=Math.max(4,Math.min(12,Math.floor((avail-8)/L))),ch=Math.max(3,Math.min(cs,Math.floor(420/T)));
  const dpr=window.devicePixelRatio||1,Wp=L*cs+2,Hp=T*ch+2;cv.width=Wp*dpr;cv.height=Hp*dpr;cv.style.width=Wp+'px';cv.style.height=Hp+'px';const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);
  const ink=cssv('--ink'),ink2=cssv('--ink-2'),sur=cssv('--surface'),acc=cssv('--accent');g.fillStyle=sur;g.fillRect(0,0,Wp,Hp);
  const m=w3.prior==='mask'?W3_KT:-1;
  for(let row=0;row<T;row++)for(let l=0;l<L;l++){const v=tr.tr.z[row*L+l],X=1+l*cs,Y=1+row*ch,w=cs-1,h=ch-1;
    if(v===m){g.strokeStyle=ink2;g.lineWidth=.8;g.setLineDash([1.5,1.5]);g.strokeRect(X+.5,Y+.5,w-1,h-1);g.setLineDash([])}
    else if(v===l%W3_KT){g.fillStyle=ink;g.fillRect(X,Y,w,h)}
    else{g.fillStyle=sur;g.fillRect(X,Y,w,h);g.strokeStyle=ink;g.lineWidth=1;g.beginPath();g.moveTo(X,Y+h);g.lineTo(X+w,Y);g.stroke();g.strokeRect(X+.5,Y+.5,w-1,h-1)}}
  if(w3.row>=0){g.strokeStyle=acc;g.lineWidth=2;g.strokeRect(1,1+w3.row*ch,L*cs-1,ch)}
  cv._ch=ch;const i=w3.row,info=i>=0?tr.tr.info[i]:null;
  $('#w3-row').textContent=info?`第 ${i+1} 步：t = ${info.t.toFixed(3)}，c = ${info.c.toFixed(3)}，ε = ${info.eps.toFixed(3)}，κ_t = ${info.kap.toFixed(4)}`+(w3.prior==='mask'?`，[M] 數 = ${info.nm}`:''):`點選軌跡或按 ↑／↓ 選取一步（共 ${T} 步；最終錯誤率 ${f3(tr.err)}）。`}
function initW3(){segInit($('#w3-prior'),['mask','unif'],w3.prior,k=>k==='mask'?'[MASK]':'均勻',k=>{w3.prior=k;w3Stale()});
  const vis=()=>{$('#w3-kbox').hidden=$('#w3-onbox').hidden=$('#w3-offbox').hidden=w3.sch!=='window';$('#w3-ebox').hidden=w3.sch!=='rescale'};
  segInit($('#w3-sch'),['ancestral','window','rescale'],w3.sch,null,k=>{w3.sch=k;vis();w3Stale()});vis();
  const on=$('#w3-on'),off=$('#w3-off');
  $('#w3-k').addEventListener('input',e=>{w3.kap=+e.target.value;$('#w3-ko').textContent=f2(w3.kap);w3Stale()});
  on.addEventListener('input',()=>{w3.ton=+on.value;if(w3.toff>=w3.ton){w3.toff=Math.max(0,+(w3.ton-.05).toFixed(2));off.value=w3.toff;announce('w3',`t_off 已調為 ${f2(w3.toff)}`)}if(w3.ton<=w3.toff){w3.ton=+(w3.toff+.05).toFixed(2);on.value=w3.ton}w3Lab();w3Stale()});
  off.addEventListener('input',()=>{w3.toff=+off.value;if(w3.toff>=w3.ton){w3.ton=Math.min(1,+(w3.toff+.05).toFixed(2));on.value=w3.ton;announce('w3',`t_on 已調為 ${f2(w3.ton)}`)}if(w3.toff>=w3.ton){w3.toff=+(w3.ton-.05).toFixed(2);off.value=w3.toff}w3Lab();w3Stale()});
  $('#w3-eta').addEventListener('input',e=>{w3.eta=+e.target.value;w3Lab();w3Stale()});$('#w3-em').addEventListener('input',e=>{w3.em=+e.target.value;w3Lab();w3Stale()});
  $('#w3-run').addEventListener('click',()=>w3Run(false));
  const box=$('#w3-trace-box'),cv=$('#w3-trace');
  cv.addEventListener('click',e=>{if(!w3.trace)return;const r=cv.getBoundingClientRect(),row=Math.floor((e.clientY-r.top-1)/cv._ch);w3.row=Math.max(0,Math.min(w3.trace.tr.info.length-1,row));w3Trace()});
  box.addEventListener('keydown',e=>{if(!w3.trace)return;const n=w3.trace.tr.info.length;if(e.key==='ArrowDown'){w3.row=Math.min(n-1,w3.row+1)}else if(e.key==='ArrowUp'){w3.row=Math.max(0,w3.row-1)}else return;e.preventDefault();w3Trace();announce('w3',$('#w3-row').textContent)});
  w3Lab();w3Run(TEST)}
function w3Lab(){$('#w3-ko').textContent=f2(w3.kap);$('#w3-ono').textContent=f2(w3.ton);$('#w3-offo').textContent=f2(w3.toff);$('#w3-etao').textContent=w3.eta.toFixed(3);$('#w3-emo').textContent=f2(w3.em)}

/* ================= W4 ================= */
const w4={K:50257,k:3,at:.34,tau:.001,seed:3,stat:null,wCache:null};
function w4Lab(){$('#w4-Ko').textContent=w4.K;$('#w4-ko').textContent=w4.k;$('#w4-ato').textContent=f2(w4.at);$('#w4-atn').textContent=`σ̃_t = ${Math.sqrt(1-w4.at*w4.at).toFixed(3)}，log SNR = ${w4.at>0?Math.log(w4.at*w4.at/(1-w4.at*w4.at)).toFixed(2):'−∞'}`;$('#w4-tauo').textContent=w4.tau<.01?w4.tau.toExponential(1):w4.tau.toFixed(3)}
function w4aDraw(){const e=efficient(mulberry32(w4.seed),w4.K,w4.k,w4.at,w4.tau),box=$('#w4a-fig'),W=widthOf(box,260,560),H=150,svg=svgIn(box,W,H),L=34,R=W-8,T=12,B=H-26,y=lin(0,1,B,T),gw=(R-L)/w4.k,bw=Math.min(40,gw*.6);
  [0,.5,1].forEach(v=>{el('line',{x1:L,x2:R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-6,y(v)+4,v.toFixed(1),{class:'tk','text-anchor':'end'})});
  e.lam.forEach((l,i)=>{const cx=L+gw*(i+.5),cl=e.case===2&&i===e.pos;el('rect',{x:cx-bw/2,y:y(l),width:bw,height:Math.max(.5,B-y(l)),class:cl?'acc-f':'par-f'},svg);tx(svg,cx,H-8,(cl?'★ ':'')+(w4.k>4&&W<500?'':'第 '+(i+1)+' 大 ')+e.vals[i].toFixed(3),{class:'tk','text-anchor':'middle'})});
  const sl=e.lam.reduce((a,b)=>a+b,0);
  dl($('#w4a-ro'),[['case',e.case===1?'1（乾淨 token 不在前 k 名）':'2（乾淨 token 在第 '+(e.pos+1)+' 名，★）'],['w̃',f4(e.wt)],['λ',e.lam.map(f4).join(', ')],['Σλ',f4(sl),true],['seed',w4.seed]]);
  $('#w4a-fig svg').setAttribute('aria-label',`case ${e.case}，λ₁ = ${f4(e.lam[0])}，Σλ = ${f4(sl)}`);announce('w4',`case ${e.case}，λ₁ = ${f4(e.lam[0])}`)}
function w4Stats(){const R=400,r=mulberry32(11);let c2=0,s1=0,l1=0,big=0;const v1=[];for(let i=0;i<R;i++){const e=efficient(r,w4.K,w4.k,w4.at,w4.tau);c2+=e.case===2;s1+=e.vals[0]/R;l1+=e.lam[0]/R;big+=e.lam[0]>.99;v1.push(e.vals[0])}
  let nv=null;if(w4.K<=1e4){const rn=mulberry32(12);nv=[];for(let i=0;i<R;i++){const w=naiveW(rn,w4.K,w4.at);let m=-Infinity;for(let j=0;j<w.length;j++)if(w[j]>m)m=w[j];nv.push(m)}}
  w4.stat={c2:c2/R,s1,l1,big:big/R,v1,nv};w4StatDraw()}
function w4StatDraw(){const s=w4.stat,hb=$('#w4a-hist');hb.classList.toggle('off',!s);if(!s){$('#w4a-stat').innerHTML='';$('#w4a-hnote').textContent='';return}
  dl($('#w4a-stat'),[['P(case 2)',f4(s.c2)],['平均最大值',f4(s.s1)],['平均 λ₁',f4(s.l1)],['λ₁ > 0.99 的比例',f4(s.big),true]]);
  const W=widthOf(hb,260,560),H=130,svg=svgIn(hb,W,H),all=s.v1.concat(s.nv||[]),lo=Math.min(...all),hi=Math.max(...all)+1e-9,nb=20,L=10,R=W-10,B=H-22,bw=(R-L)/nb;
  const hist=a=>{const h=new Array(nb).fill(0);a.forEach(v=>h[Math.min(nb-1,Math.floor((v-lo)/(hi-lo)*nb))]++);return h},he=hist(s.v1),hn=s.nv?hist(s.nv):null,mx=Math.max(...he,...(hn||[1])),y=lin(0,mx,B,8);
  he.forEach((c,i)=>el('rect',{x:L+i*bw+1,y:y(c),width:bw-2,height:B-y(c),class:'par-f'},svg));if(hn)hn.forEach((c,i)=>el('rect',{x:L+i*bw+1,y:y(c),width:bw-2,height:B-y(c),fill:'none',stroke:cssv('--ink'),'stroke-width':1.5},svg));
  tx(svg,L,H-6,lo.toFixed(2),{class:'tk'});tx(svg,R,H-6,hi.toFixed(2),{class:'tk','text-anchor':'end'});tx(svg,(L+R)/2,H-6,'最大值',{class:'tk','text-anchor':'middle'});
  $('#w4a-hnote').textContent=s.nv?'實心＝高效法（400 次），外框＝暴力法（具現 K 個值再取最大，400 次）；兩者用不同亂數，只應在統計誤差內一致。':'K > 10⁴ 時不跑暴力法（避免一次具現上千萬個高斯值）；只顯示高效法的直方圖。'}
function w4W(){const key=w4.K+'|'+w4.at;if(!w4.wCache||w4.wCache.key!==key)w4.wCache={key,w:naiveW(mulberry32(7),w4.K,w4.at)};return w4.wCache.w}
function w4bDraw(){const w=w4W(),lz=logZexact(w,w4.tau),z=ztildeFrom(w,w4.k,w4.at,w4.tau),rt=Math.exp(z.logZtop-lz),rz=Math.exp(z.logZt-lz),sl=Math.exp(z.logZtop-z.logZt);
  const box=$('#w4b-bars'),W=widthOf(box,280,720),H=96,svg=svgIn(box,W,H),L=92,R=W-60,x=lin(0,Math.max(1.3,rz*1.05),L,R);
  el('line',{x1:x(1),x2:x(1),y1:4,y2:H-4,class:'ref'},svg);tx(svg,x(1)+4,12,'1',{class:'tk'});
  [['Z_top / Z',rt,'par-f'],['Z̃ / Z',rz,'acc-f']].forEach(([n,v,c],i)=>{const Y=18+i*36;tx(svg,0,Y+15,n,{class:'tk'});el('rect',{x:L,y:Y,width:Math.max(1,Math.min(x(v),R)-L),height:22,class:c},svg);tx(svg,Math.min(x(v),R)+6,Y+15,f4(v),{class:'mono-t'})});
  dl($('#w4b-ro'),[['log Z（真值）',f4(lz)],['Z_top / Z',f4(rt)],['Z̃ / Z',f4(rz),true],['Σλ = Z_top / Z̃',f4(sl)]]);
  const sb=$('#w4b-sweep'),W2=widthOf(sb,280,720),H2=170,s2=svgIn(sb,W2,H2),L2=40,R2=W2-12,T2=10,B2=H2-26,us=Array.from({length:31},(_,i)=>-3+i*.1),pts=us.map(u=>{const tau=Math.pow(10,u),l=logZexact(w,tau),zz=ztildeFrom(w,w4.k,w4.at,tau);return[u,Math.exp(zz.logZtop-l),Math.exp(zz.logZt-l)]});
  const ymax=Math.max(1.3,...pts.map(p=>p[2])),xs=lin(-3,0,L2,R2),ys=lin(0,ymax,B2,T2);[0,.5,1].forEach(v=>{el('line',{x1:L2,x2:R2,y1:ys(v),y2:ys(v),class:'grid'},s2);tx(s2,L2-6,ys(v)+4,v.toFixed(1),{class:'tk','text-anchor':'end'})});
  ['10⁻³','10⁻²','10⁻¹','1'].forEach((s,i)=>tx(s2,xs(-3+i),H2-10,s,{class:'tk','text-anchor':'middle'}));tx(s2,(L2+R2)/2,H2,'τ',{class:'tk','text-anchor':'middle'});
  el('path',{d:lp(pts.map(p=>[xs(p[0]),ys(p[1])])),class:'pow'},s2);el('path',{d:lp(pts.map(p=>[xs(p[0]),ys(p[2])])),class:'acc-s'},s2);
  const ux=Math.log10(w4.tau);el('line',{x1:xs(ux),x2:xs(ux),y1:T2,y2:B2,class:'refl'},s2);
  $('#w4b-say').textContent=`目前設定（K = ${w4.K}，k = ${w4.k}，α̃_t = ${f2(w4.at)}，τ = ${$('#w4-tauo').textContent}）：Z_top/Z = ${f4(rt)}，Z̃/Z = ${f4(rz)}，Σλ = ${f4(sl)}。`;
  $('#w4b-bars svg').setAttribute('aria-label',$('#w4b-say').textContent)}
let w4t=0;function w4All(){w4Lab();w4aDraw();w4.stat=null;w4StatDraw();clearTimeout(w4t);w4t=setTimeout(w4bDraw,TEST?0:120)}
function initW4(){$('#w4-K').addEventListener('input',e=>{w4.K=Math.round(Math.pow(10,+e.target.value));w4All()});
  $('#w4-gpt').addEventListener('click',()=>{w4.K=50257;$('#w4-K').value=4.7;w4All()});
  $('#w4-k').addEventListener('input',e=>{w4.k=+e.target.value;w4All()});$('#w4-at').addEventListener('input',e=>{w4.at=+e.target.value;w4All()});
  $('#w4-tau').addEventListener('input',e=>{w4.tau=+Math.pow(10,+e.target.value).toPrecision(3);w4All()});
  $('#w4a-rep').addEventListener('click',w4Stats);$('#w4a-new').addEventListener('click',()=>{w4.seed++;w4aDraw()});w4Lab();w4aDraw();w4StatDraw();w4bDraw()}

/* ================= timeline ================= */
const TL_GROUP={'1503.03585':['base','psi-discrete-diffusion-interpolating-process'],'2107.03006':['base','psi-discrete-diffusion-interpolating-process'],
 '2011.13456':['gpc','psi-predictor-corrector-gaussian-to-discrete'],'2205.14987':['ctmc','psi-ctmc-discrete-diffusion-samplers'],'2310.16834':['ctmc','psi-ctmc-discrete-diffusion-samplers'],'2407.15595':['ctmc','psi-ctmc-discrete-diffusion-samplers'],
 '2406.07524':['mdm','mdlm-masked-diffusion-objective'],'2207.12598':['usdm','psi-uniform-state-guidance-and-self-correction'],'2406.01572':['usdm','psi-uniform-state-guidance-and-self-correction'],'2412.10193':['usdm','psi-uniform-state-guidance-and-self-correction'],
 '2503.00307':['rem','psi-predictor-corrector-gaussian-to-discrete'],'2506.10892':['duo','diffusion-duality'],'2602.21185':['me','psi-superposition-posteriors-method']};
const TL_NAME={base:'基礎',gpc:'Gaussian PC',ctmc:'CTMC',mdm:'MDM',usdm:'Guidance／USDM',rem:'Remasking',duo:'Duo',me:'本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.slice().sort((a,b)=>(Number(a.year)||0)-(Number(b.year)||0));
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

/* ================= toc, theme, stack tables ================= */
function initToc(){const secs=$$('[data-toc]'),ol=$('#toc'),bar=$('#tocbar');
  secs.forEach(s=>{ol.insertAdjacentHTML('beforeend',`<li><a href="#${s.id}"><span class="n">${s.dataset.toc}</span><span>${s.dataset.tt}</span></a></li>`);bar.insertAdjacentHTML('beforeend',`<a href="#${s.id}">${s.dataset.toc} ${s.dataset.tt}</a>`)});
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;$$('.toc a,.tocbar a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+id));const ab=$(`.tocbar a[href="#${id}"]`);if(ab&&bar.offsetParent)bar.scrollTo({left:ab.offsetLeft-16,behavior:RM?'auto':'smooth'})}})},{rootMargin:'-40% 0px -55% 0px'});
  secs.forEach(s=>io.observe(s));addEventListener('scroll',()=>$('#appbar').classList.toggle('raised',scrollY>8),{passive:true})}
function redrawAll(){drawHero();drawS1();w1Draw();w2Draw();drawKxAll();w3Draw();if(!w3.res)w3Kap();w4aDraw();w4StatDraw();w4bDraw()}
function initTheme(){let saved=null;try{saved=localStorage.getItem('pm-theme')}catch(e){}if(saved)document.documentElement.dataset.theme=saved;
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){}redrawAll()});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',redrawAll)}
function stackTables(){$$('table.stack').forEach(t=>{const hs=$$('thead th',t).map(th=>th.textContent);$$('tbody tr',t).forEach(tr=>$$('td',tr).forEach((td,i)=>td.dataset.h=hs[i]||''))})}

/* ================= self test (design.md 5 and 6.3) ================= */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t=1e-4)=>Math.abs(a-b)<=t+1e-12,nearv=(a,b,t=1e-4)=>a.length===b.length&&a.every((v,i)=>near(v,b[i],t));
  // post vs Eq. (3) and Eq. (2); marginal; ReMDM
  const eq3=(zt,x,at,as)=>{const K=x.length,ats=at/as,z=onehot(zt,K),den=K*at*x[zt]+1-at;return x.map((xi,j)=>(K*at*z[j]*xi+(ats-at)*z[j]+(as-at)*xi+(1-ats)*(1-as)/K)/den)};
  const eq2=(zt,x,at,as)=>{const K=x.length,m=K-1;if(zt!==m)return x.slice();return x.map((xi,j)=>(as-at)/(1-at)*xi+(1-as)/(1-at)*(j===m?1:0))};
  let e3=0,e2=0;for(let xi=0;xi<4;xi++)for(let z=0;z<4;z++){const x=onehot(xi,4),a=post(z,x,.3,.6,prior('unif',4)),b=eq3(z,x,.3,.6);a.forEach((v,j)=>e3=Math.max(e3,Math.abs(v-b[j])))}
  {const xs=[.5,.3,.2,0];[3,0,1].forEach(z=>{const x=z===3?xs:onehot(z,4),a=post(z,x,.3,.6,prior('mask',4)),b=eq2(z,x,.3,.6);a.forEach((v,j)=>e2=Math.max(e2,Math.abs(v-b[j])))})}
  ok(e3<1e-12&&e2<1e-12,'post = Eq.(3), Eq.(2)');
  [0,.5,1,-.3].forEach(k=>['unif','mask'].forEach(pr=>{const pi=prior(pr,4),x=onehot(0,4),mg=marginalAfter(x,.3,.6,()=>k,pi),qs=qMarg(x,.6,pi);ok(mg.every((v,j)=>Math.abs(v-qs[j])<1e-12),`marginal κ=${k} ${pr}`)}));
  {const pi=prior('mask',4),sig=.1,k=kappaFromSigma(sig,.6),x=[.5,.3,.2,0];const r21m=x.map((xi,j)=>(.6-(1-sig)*.3)/.7*xi+(1-.6-sig*.3)/.7*(j===3?1:0)),r21u=onehot(1,4).map((v,j)=>(1-sig)*v+sig*(j===3?1:0));
   ok(nearv(psiTrue(3,x,.3,.6,k,pi),r21m,1e-12)&&nearv(psiTrue(1,onehot(1,4),.3,.6,k,pi),r21u,1e-12),'ReMDM Eq.(21) both rows')}
  // W1 (5.1)
  {const pi=prior('unif',4),x=onehot(0,4);ok(nearv(post(0,x,.3,.6,pi),[.9211,.0263,.0263,.0263])&&nearv(post(1,x,.3,.6,pi),[.5,.3571,.0714,.0714]),'W1 q_{s|t} rows');
   ok(nearv(psiTrue(0,x,.3,.6,.5,pi),[.8105,.0632,.0632,.0632])&&nearv(psiTrue(1,x,.3,.6,.5,pi),[.6,.2286,.0857,.0857]),'W1 Ψ rows');
   const sv={...w1};Object.assign(w1,{prior:'unif',K:4,at:.3,as:.6,dep:false});[[1,.625],[.5,.505],[0,.385]].forEach(([k,r])=>{w1.k=k;ok(near(w1Calc().ret,r),`W1 retention κ=${k}`)});
   w1.k=.5;w1.dep=true;const o=w1Calc();ok(near(o.md,.0525)&&nearv(o.mg,[.6475,.1175,.1175,.1175]),'W1 κ(z_t) switch unif');w1.prior='mask';ok(near(w1Calc().md,.06),'W1 κ(z_t) switch mask');
   w1.dep=false;[[1,.7],[.5,.58],[0,.46]].forEach(([k,r])=>{w1.k=k;ok(near(w1Calc().ret,r),`W1 mask retention κ=${k}`)});Object.assign(w1,sv)}
  // W2 / W2b (5.2)
  {const sv={...w2};Object.assign(w2,{prior:'unif',z:1,den:'wrong',at:.3,as:.6});
   ok(nearv(w2Parts(1).c,[0,.8028,.1479,.0493]),'W2 q_{0|t} wrong');ok(near(w2Parts(1).mix[0],.0352)&&near(w2Parts(.9).mix[0],.0417)&&near(w2Parts(.5).mix[0],.0676),'W2 P(A) κ=1,.9,.5');
   w2.den='weak';ok(nearv(w2Parts(1).c,[.3302,.5377,.0991,.033])&&near(w2Parts(.5).mix[0],.2434),'W2 weak');w2.den='right';ok(near(w2Parts(.9).mix[0],.4182),'W2 right');
   w2.prior='mask';w2.z=3;w2.den='wrong';ok(nearv(w2Parts(.9).mix,[0,.312,.1337,.5543]),'W2 mask z=[M]');w2.z=1;ok(nearv(w2Parts(.9).mix,[0,.96,0,.04]),'W2 mask z=B remask .04');Object.assign(w2,sv)}
  ok(near(sigmaMax(.3,.6),1)&&near(kappaFromSigma(.04,.6),.9)&&near(kappaFromSigma(1,.6),-1.5)&&near(sigmaMax(.7,.75),.3571)&&near(kappaFromSigma(sigmaMax(.7,.75),.75),-.4286),'W2b');
  // κ schedules (5.3)
  {const T=128,mins=k=>{let m=1;for(let i=T;i>=1;i--){const t=i/T;m=Math.min(m,kappaSched(k.s,t,1-t,1-(i-1)/T,{eta:k.e}))}return m};
   ok(near(mins({s:'cap',e:.05}),-.0579)&&near(mins({s:'cap',e:.01}),-.0159)&&near(mins({s:'rescale',e:.05}),.9),'κ schedule minima');
   ok(near(kappaSched('rescale',1,0,1/T,{eta:.05}),.9496)&&near(kappaSched('rescale',.5,.5,.5+1/T,{eta:.05}),.9),'rescale κ(t=1), κ(.5)')}
  // PRNG + W3 (sync)
  {const r=mulberry32(1000);ok(nearv([r(),r(),r()],[.795194906881,.827687913552,.691516105784],1e-12),'mulberry32(1000)');const g=mulberry32(1000);ok(near(gauss(g),.8351660354,1e-9),'gauss(seed 1000)')}
  {const p={kap:.9,ton:.6,toff:.1,eta:.05},er=(kind,s,T,em=.5)=>{let a=0;for(let k=0;k<32;k++)a+=runChain(kind,s,T,p,1000+k,em)/32;return a};
   ok(near(er('mask','window',512),.1953,.02)&&near(er('mask','ancestral',512),.3408,.02),'W3 mask T=512');ok(near(er('unif','rescale',64),.209,.02),'W3 unif rescale T=64');ok(near(er('mask','window',512,.2),.0596,.02),'W3 ε_max = 0.2');
   ok(!!w3.res&&near(w3.res.sel[6],.1953,.02)&&near(w3.res.anc[0],.3691,.02),'W3 default curve (sync path)')}
  // W4 (5.4)
  ok(near(logMu(.7,.5,.2),1.21165,1e-5)&&near(normPpf(.975),1.959964,1e-6),'logMu, normPpf');
  {const e=efficient(mulberry32(3),50257,3,.34,.001);ok(nearv(e.vals,[4.0994,3.5772,3.5308])&&near(e.wt,.3703)&&e.case===1&&nearv(e.lam,[1,0,0]),'W4a single draw')}
  {const r=mulberry32(11);let c2=0,l1=0,big=0;for(let i=0;i<400;i++){const e=efficient(r,50257,3,.9,.001);c2+=e.case===2;l1+=e.lam[0]/400;big+=e.lam[0]>.99}ok(near(c2/400,.03,.02)&&near(l1,.9907,.02)&&near(big/400,.945,.02),'W4a stats α̃=.9')}
  {const w=naiveW(mulberry32(7),50257,.34);[[.05,.9916,1.0297,.9631],[.3,.1129,1.0662,.1059]].forEach(([tau,a,b,c])=>{const l=logZexact(w,tau),z=ztildeFrom(w,3,.34,tau);ok(near(Math.exp(z.logZtop-l),a,1e-3)&&near(Math.exp(z.logZt-l),b,1e-3)&&near(Math.exp(z.logZtop-z.logZt),c,1e-3),`W4b τ=${tau}`)});
   ok(near(logZexact(w,.001),3748.5352,1e-3),'W4b log Z')}
  // DOM
  ok($$('details.check').length===10,'check boxes: '+$$('details.check').length);
  ok(['w1','w2','w2b','w3','w4a','w4b'].every(id=>!!$('#'+id)),'plates present');
  ok($('#w3 .plate-lead').textContent.trim().startsWith('假設：資料只有一個序列'),'W3 first line = assumptions');
  ok(!/κ|ReMDM|Gen\. PPL|q_|posterior/.test($$('#top .fact').map(f=>f.textContent).join('')),'hero cards avoid undefined terms');
  {const tl=JSON.parse($('#timeline-data').textContent);ok(tl.length===13&&tl.every(e=>TL_GROUP[e.arxiv]),'timeline groups')}
  ok(!!document.querySelector('.katex')||typeof katex==='undefined','KaTeX rendered');ok(!document.querySelector('.katex-error'),'no KaTeX errors');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: '+document.documentElement.scrollWidth+'>'+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ================= boot ================= */
initTheme();initToc();stackTables();drawHero();drawS1();initW1();initW2();drawKxAll();initW3();initW4();renderTimeline();
let lastW=innerWidth,rz=0;addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(redrawAll)});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(TEST)addEventListener('load',()=>setTimeout(selfTest,50));
{const m=location.search.match(/[?&]shot=([\w-]+)/);if(m){const e=document.getElementById(m[1]);if(e)$$('main>section').forEach(sec=>{if(!sec.contains(e))sec.hidden=true});redrawAll()}}  // screenshot helper: show only the section holding #id
})();
