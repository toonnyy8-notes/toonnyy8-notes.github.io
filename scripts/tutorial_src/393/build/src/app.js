(()=>{
'use strict';
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
function el(tag,a,p){const e=document.createElementNS(NS,tag);if(a)for(const k in a)if(a[k]!=null)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function tx(p,x,y,s,a){const t=el('text',Object.assign({x,y},a||{}),p);t.textContent=s;return t}
const lin=(d0,d1,r0,r1)=>v=>r0+(v-d0)/(d1-d0)*(r1-r0);
function lp(pts){let s='';for(let i=0;i<pts.length;i++)s+=(i?'L':'M')+pts[i][0].toFixed(1)+' '+pts[i][1].toFixed(1);return s}
function widthOf(box,min,max){return Math.max(min,Math.min(max,Math.floor(box.clientWidth||max)))}
function frame(svg,o){
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${o.W} ${o.H}`);svg.setAttribute('width',o.W);svg.setAttribute('height',o.H);
  const m=o.m,x0=m.l,x1=o.W-m.r,y0=o.H-m.b,y1=m.t;
  const X=lin(o.x[0],o.x[1],x0,x1),Y=lin(o.y[0],o.y[1],y0,y1);
  const g=el('g',null,svg);
  (o.yt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:(o.yf?o.yf(v):String(v)),yy=Y(val);el('line',{x1:x0,x2:x1,y1:yy,y2:yy,class:'grid'},g);tx(g,x0-8,yy+4,lab,{class:'tk','text-anchor':'end'})});
  (o.xt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:(o.xf?o.xf(v):String(v)),xx=X(val);el('line',{x1:xx,x2:xx,y1:y0,y2:y0+4,class:'axl'},g);tx(g,xx,y0+18,lab,{class:'tk','text-anchor':'middle'})});
  el('line',{x1:x0,x2:x1,y1:y0,y2:y0,class:'axl'},g);
  if(o.xl)tx(g,(x0+x1)/2,o.H-4,o.xl,{class:'al','text-anchor':'middle'});
  if(o.yl)tx(g,4,12,o.yl,{class:'al'});
  return {X,Y,g,x0,x1,y0,y1};
}
function hatch(svg,id,cls){const d=el('defs',null,svg);const p=el('pattern',{id,patternUnits:'userSpaceOnUse',width:6,height:6,patternTransform:'rotate(45)'},d);
  el('rect',{width:6,height:6,class:'pat-bg'},p);el('line',{x1:1.5,y1:0,x2:1.5,y2:6,class:cls},p)}
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
const setTxt=(id,s)=>{const n=document.getElementById(id);if(n)n.textContent=s};
const f4=v=>(Math.abs(v)<5e-5?0:v).toFixed(4);

/* ---------- shared math (design 4.0) ---------- */
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}
function argmaxFirst(v){let b=0;for(let i=1;i<v.length;i++)if(v[i]>v[b])b=i;return b}
function erfcNR(x){const z=Math.abs(x),t=1/(1+0.5*z);const r=t*Math.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))));return x>=0?r:2-r}
const Phi=z=>0.5*erfcNR(-z/Math.SQRT2);
const phi=z=>Math.exp(-z*z/2)/Math.sqrt(2*Math.PI);
const logPhi=z=>z>0?Math.log1p(-0.5*erfcNR(z/Math.SQRT2)):Math.log(0.5*erfcNR(-z/Math.SQRT2));
const ZH=0.005,NZ=4401,ZG=Float64Array.from({length:NZ},(_,i)=>-10+ZH*i);
const gCache=new Map();
function gTab(K){if(!gCache.has(K)){const g=new Float64Array(NZ);for(let i=0;i<NZ;i++)g[i]=Math.exp((K-1)*logPhi(ZG[i]));gCache.set(K,g)}return gCache.get(K)}
function Top(at,K){if(at<=0)return 0;if(at>=0.999999)return 1;
  const mu=at/Math.sqrt(1-at*at),g=gTab(K);let s=0;
  for(let i=0;i<NZ;i++){const w=(i===0||i===NZ-1)?0.5:1;s+=w*(phi(ZG[i]-mu)-phi(ZG[i]))*g[i]}
  s=s*ZH+(Phi(12)-Phi(12-mu));return Math.min(1,Math.max(0,K/(K-1)*s))}
function alphaForT(target,K){let lo=0,hi=0.999999;for(let i=0;i<60;i++){const mid=(lo+hi)/2;if(Top(mid,K)<target)lo=mid;else hi=mid}return(lo+hi)/2}
const bCache=new Map();function bandB(K){if(!bCache.has(K))bCache.set(K,[alphaForT(0.05,K),alphaForT(0.95,K)]);return bCache.get(K)}

/* ---------- W1 ---------- */
const w1={K:5,a:0.6,seed:7};
function w1Sample(K,at,seed){const r=mulberry32(seed),s=Math.sqrt(1-at*at),cnt=new Array(K).fill(0),w=new Float64Array(K);let first=null;
  for(let n=0;n<20000;n++){for(let k=0;k<K;k++)w[k]=(k===0?at:0)+s*gauss(r);cnt[argmaxFirst(w)]++;if(!first)first=Array.from(w)}
  return{freq:cnt.map(c=>c/20000),first}}
function renderW1(){const{K,a,seed}=w1,s=Math.sqrt(1-a*a),T=Top(a,K),px=T+(1-T)/K,po=(1-T)/K,S=w1Sample(K,a,seed);
  setTxt('w1-Ko',K);setTxt('w1-ao',a.toFixed(2));setTxt('w1-seed','seed '+seed);
  const th=k=>k===0?px:po,se=k=>Math.sqrt(th(k)*(1-th(k))/20000);
  const box=$('#w1-fig'),svg=$('svg',box),W=widthOf(box,280,560),H=230;
  const ymax=Math.max(0.3,Math.ceil((Math.max(...S.freq,px+2*se(0))+0.04)*10)/10);
  const f=frame(svg,{W,H,m:{l:44,r:8,t:28,b:30},x:[0,K],y:[0,ymax],yt:[0,ymax/2,ymax].map(v=>[v,v.toFixed(2)]),xt:[...Array(K)].map((_,k)=>[k+0.5,k===0?'x':String(k)])});
  hatch(svg,'w1-hatch','pat-sig');
  const bw=(f.X(1)-f.X(0))*0.56;
  S.freq.forEach((v,k)=>{const cx=f.X(k+0.5);el('rect',{x:cx-bw/2,y:f.Y(v),width:bw,height:Math.max(0,f.Y(0)-f.Y(v)),style:k===0?'fill:url(#w1-hatch)':null,class:k===0?'sig':'bar-sig','stroke-width':k===0?1.5:0},f.g);
    const t=th(k),e=2*se(k);el('line',{x1:cx,x2:cx,y1:f.Y(t-e),y2:f.Y(t+e),class:'wh'},f.g);el('line',{x1:cx-bw/2-4,x2:cx+bw/2+4,y1:f.Y(t),y2:f.Y(t),class:'th'},f.g);
    el('rect',{x:cx-4,y:f.Y(t)-4,width:8,height:8,transform:`rotate(45 ${cx} ${f.Y(t)})`,class:'ink-f'},f.g)});
  tx(f.g,4,12,'頻率',{class:'al'});
  // first sample
  const b2=$('#w1-one'),s2=$('svg',b2),W2=widthOf(b2,280,560),H2=120,wmin=Math.min(-1.5,...S.first),wmax=Math.max(2,...S.first),am=argmaxFirst(S.first);
  const g2=frame(s2,{W:W2,H:H2,m:{l:44,r:8,t:22,b:22},x:[0,K],y:[wmin,wmax],yt:[[0,'0']],xt:[...Array(K)].map((_,k)=>[k+0.5,k===0?'x':String(k)])});
  tx(g2.g,4,12,'第一個樣本的 w',{class:'al'});const bw2=(g2.X(1)-g2.X(0))*0.5;
  S.first.forEach((v,k)=>{const cx=g2.X(k+0.5),y0=g2.Y(0),y=g2.Y(v);el('rect',{x:cx-bw2/2,y:Math.min(y,y0),width:bw2,height:Math.abs(y-y0),class:k===0?'bar-sig':'bar-ink'},g2.g);
    if(k===am){el('rect',{x:cx-bw2/2-3,y:Math.min(y,y0)-3,width:bw2+6,height:Math.abs(y-y0)+6,class:'frame'},g2.g);tx(g2.g,cx,Math.min(y,y0)-7,'arg max',{class:'lab','text-anchor':'middle'})}});
  const mo=S.freq.slice(1).reduce((p,c)=>p+c,0)/(K-1);let dev=0;S.freq.forEach((v,k)=>{dev=Math.max(dev,Math.abs(v-th(k))/se(k))});
  setTxt('w1-r-s',s.toFixed(4));setTxt('w1-r-snr',(a/s).toFixed(4));setTxt('w1-r-a',T.toFixed(4));setTxt('w1-r-px',px.toFixed(4));setTxt('w1-r-po',po.toFixed(4));
  setTxt('w1-r-mx',S.freq[0].toFixed(4));setTxt('w1-r-mo',mo.toFixed(4));setTxt('w1-r-dev',dev.toFixed(2));
  svg.setAttribute('aria-label',`K = ${K}，alpha~ = ${a.toFixed(2)}：理論 p_x = ${px.toFixed(4)}，模擬 p_x = ${S.freq[0].toFixed(4)}`);
  return{px,mx:S.freq[0]}}
function initW1(){let raf=0;const go=()=>{cancelAnimationFrame(raf);raf=requestAnimationFrame(renderW1)};
  const say=()=>{const T=Top(w1.a,w1.K),px=T+(1-T)/w1.K;announce('w1',`理論 p_x = ${px.toFixed(4)}，模擬 p_x = ${$('#w1-r-mx').textContent}`)};
  $('#w1-K').addEventListener('input',e=>{w1.K=+e.target.value;go()});$('#w1-a').addEventListener('input',e=>{w1.a=+e.target.value;go()});
  $('#w1-K').addEventListener('change',say);$('#w1-a').addEventListener('change',say);
  $('#w1-re').addEventListener('click',()=>{w1.seed++;renderW1();say()});renderW1()}

/* ---------- W2 (two instances) ---------- */
const KLIST=[[2,'2'],[3,'3'],[10,'10'],[100,'100'],[1000,'1000'],[30522,'BERT 30522'],[50257,'GPT-2 50257']];
const cCache=new Map();
function curveT(K,zoom){const key=K+'|'+zoom;if(!cCache.has(key)){const a0=zoom?0.8:0,pts=[];for(let i=0;i<=400;i++){const a=a0+(1-a0)*i/400;pts.push([a,Top(a,K)])}cCache.set(key,pts)}return cCache.get(key)}
const BAND_A=[0.85,0.97],BAND_A2=[Math.sqrt(0.85),Math.sqrt(0.97)];
function initW2(id,o){const st={K:o.K,a:o.a,zoom:o.zoom,bands:o.bands};const seg=$('#'+id+'-K'),inp=$('#'+id+'-a'),zc=$('#'+id+'-z');
  KLIST.forEach(([k,l])=>{const b=document.createElement('button');b.type='button';b.textContent=l;b.dataset.v=k;b.setAttribute('aria-pressed',String(k===st.K));
    b.addEventListener('click',()=>{st.K=k;$$('button',seg).forEach(x=>x.setAttribute('aria-pressed',String(+x.dataset.v===k)));render();say()});seg.appendChild(b)});
  const setA=v=>{st.a=Math.min(1,Math.max(0,Math.round(v*1000)/1000));inp.value=st.a;render()};
  inp.addEventListener('input',()=>{st.a=+inp.value;render()});inp.addEventListener('change',say);
  inp.addEventListener('keydown',e=>{if(e.key==='PageUp'||e.key==='PageDown'){e.preventDefault();setA(st.a+(e.key==='PageUp'?0.01:-0.01));say()}});
  $$('.stepper button',inp.parentNode).forEach(b=>b.addEventListener('click',()=>{setA(st.a+ +b.dataset.d);say()}));
  zc.addEventListener('change',()=>{st.zoom=zc.checked;render()});
  function say(){const T=Top(st.a,st.K);announce(id,`T = ${T.toFixed(4)}，P(z=x) = ${(T+(1-T)/st.K).toFixed(4)}`)}
  function render(){const{K,a,zoom}=st,box=$('#'+id+'-fig'),svg=$('svg',box),W=widthOf(box,280,620),bandH=st.bands?54:0,H=250+bandH;
    const xa=zoom?0.8:0,xt=zoom?[0.8,0.85,0.9,0.95,1]:[0,0.25,0.5,0.75,1];
    const f=frame(svg,{W,H,m:{l:44,r:12,t:28,b:30+bandH},x:[xa,1],y:[0,1],yt:[0,0.25,0.5,0.75,1].map(v=>[v,String(v)]),xt:xt.map(v=>[v,String(v)])});
    tx(f.g,4,12,'T(alpha~)',{class:'al'});
    if(K!==2)el('path',{d:lp(curveT(2,zoom).map(([u,v])=>[f.X(u),f.Y(v)])),class:'ref'},f.g);
    el('path',{d:lp(curveT(K,zoom).map(([u,v])=>[f.X(u),f.Y(v)])),class:'acc'},f.g);
    const T=Top(a,K);
    if(a>=xa){el('line',{x1:f.X(a),x2:f.X(a),y1:f.y0,y2:f.y1,class:'cur'},f.g);el('circle',{cx:f.X(a),cy:f.Y(T),r:5,class:'kdot'},f.g)}
    if(st.bands){const pats=[['bandA','pat-ink'],['bandA2','pat-noi'],['bandB','pat-sig']];const d=el('defs',null,svg);
      pats.forEach(([n,c],i)=>{const p=el('pattern',{id:`${id}-${n}`,patternUnits:'userSpaceOnUse',width:6,height:6,patternTransform:`rotate(${[45,0,-45][i]})`},d);el('rect',{width:6,height:6,class:'pat-bg'},p);el('line',{x1:1.5,y1:0,x2:1.5,y2:6,class:c},p)});
      const B=bandB(K),rows=[['A',BAND_A,'bandA'],['A′',BAND_A2,'bandA2'],['B',B,'bandB']];
      rows.forEach(([lab,[u0,u1],n],i)=>{const y=f.y0+26+i*16,l=Math.max(xa,u0),r=Math.max(xa,u1);tx(f.g,f.x0-8,y+10,lab,{class:'labm','text-anchor':'end'});
        el('rect',{x:f.x0,y,width:f.x1-f.x0,height:12,class:'pat-bg',stroke:'var(--line)'},f.g);
        if(r>l)el('rect',{x:f.X(l),y,width:f.X(r)-f.X(l),height:12,style:`fill:url(#${id}-${n})`,stroke:'var(--ink-3)','stroke-width':1},f.g)});
      setTxt(id+'-r-A',`${f4(Top(BAND_A[0],K))}、${f4(Top(BAND_A[1],K))}`);setTxt(id+'-r-A2',`${f4(Top(BAND_A2[0],K))}、${f4(Top(BAND_A2[1],K))}`);setTxt(id+'-r-B',`${B[0].toFixed(4)}、${B[1].toFixed(4)}`)}
    const s=Math.sqrt(Math.max(0,1-a*a));
    setTxt(id+'-ao',a.toFixed(3));setTxt(id+'-r-a',a.toFixed(3));setTxt(id+'-r-s',s.toFixed(4));setTxt(id+'-r-snr',s>0?(a/s).toFixed(4):'∞');setTxt(id+'-r-T',f4(T));setTxt(id+'-r-p',f4(T+(1-T)/K));
    svg.setAttribute('aria-label',`K = ${K}：alpha~ = ${a.toFixed(3)} 時 T = ${T.toFixed(4)}`)}
  st.render=render;render();return st}

/* ---------- W3 ---------- */
const w3={ws:[1.2,0.6,0],as:0.9,at:0.6};
function pushK(ws,ats,att){const a=att/ats,s=Math.sqrt(1-a*a),c=ws.map(v=>a*v),P=[];
  for(let i=0;i<3;i++){let acc=0;for(let q=0;q<=1800;q++){const u=-9+0.01*q;let pr=phi(u);for(let j=0;j<3;j++)if(j!==i)pr*=Phi(u+(c[i]-c[j])/s);acc+=(q===0||q===1800?0.5:1)*pr}P.push(acc*0.01)}return P}
function usdmKernel(ws,ats,att){const zs=argmaxFirst(ws),tie=ws.filter(v=>v===ws[zs]).length>1,a=Top(att,3)/Top(ats,3);return{Q:[0,1,2].map(i=>a*(i===zs?1:0)+(1-a)/3),a,zs,tie}}
function renderW3(){const{ws,as,at}=w3,P=pushK(ws,as,at),K=usdmKernel(ws,as,at),tv=0.5*P.reduce((p,v,i)=>p+Math.abs(v-K.Q[i]),0);
  ['w0','w1','w2'].forEach((k,i)=>setTxt('w3-'+k+'o',ws[i].toFixed(2)));setTxt('w3-aso',as.toFixed(2));setTxt('w3-ato',at.toFixed(2));
  const box=$('#w3-fig'),svg=$('svg',box),W=widthOf(box,280,560),H=230;
  const f=frame(svg,{W,H,m:{l:44,r:8,t:30,b:34},x:[0,3],y:[0,1],yt:[0,0.25,0.5,0.75,1].map(v=>[v,String(v)]),xt:[[0.5,'第 1 類'],[1.5,'第 2 類'],[2.5,'第 3 類']]});
  hatch(svg,'w3-hatch','pat-noi');const bw=(f.X(1)-f.X(0))*0.32;
  for(let i=0;i<3;i++){const cx=f.X(i+0.5);
    [[P[i],cx-bw-2,'bar-sig',null],[K.Q[i],cx+2,'noi','url(#w3-hatch)']].forEach(([v,x,c,fill])=>{el('rect',{x,y:f.Y(v),width:bw,height:f.Y(0)-f.Y(v),class:c,style:fill?'fill:'+fill:null,'stroke-width':fill?1.5:0},f.g);tx(f.g,x+bw/2,f.Y(v)-5,v.toFixed(4),{class:'tk','text-anchor':'middle'})});
    if(i===K.zs)tx(f.g,cx,H-2,'z_s',{class:'lab','text-anchor':'middle'})}
  tx(f.g,4,12,'下一步機率',{class:'al'});
  const o=[0,1,2].filter(i=>i!==K.zs);
  setTxt('w3-cap',`非 z_s 的兩類：高斯 ${P[o[0]].toFixed(4)} 與 ${P[o[1]].toFixed(4)}，USDM 都是 ${K.Q[o[0]].toFixed(4)}。`+(K.tie?' arg max 並列，取索引較小者。':''));
  setTxt('w3-r-ag',(at/as).toFixed(4));setTxt('w3-r-ad',K.a.toFixed(4));setTxt('w3-r-zs','第 '+(K.zs+1)+' 類');setTxt('w3-r-sum',P.reduce((a,b)=>a+b,0).toFixed(6));setTxt('w3-r-tv',tv.toFixed(4));
  svg.setAttribute('aria-label',`高斯 ${P.map(v=>v.toFixed(3)).join('、')}；USDM ${K.Q.map(v=>v.toFixed(3)).join('、')}`);
  return{P,K,tv}}
function initW3(){let raf=0;const go=()=>{cancelAnimationFrame(raf);raf=requestAnimationFrame(renderW3)};const say=()=>announce('w3','總變異距離 = '+$('#w3-r-tv').textContent);
  ['w0','w1','w2'].forEach((k,i)=>{const n=$('#w3-'+k);n.addEventListener('input',()=>{w3.ws[i]=+n.value;go()});n.addEventListener('change',say)});
  const aS=$('#w3-as'),aT=$('#w3-at');
  aS.addEventListener('input',()=>{w3.as=+aS.value;aT.max=(w3.as-0.05).toFixed(2);if(w3.at>w3.as-0.05){w3.at=+(w3.as-0.05).toFixed(2);aT.value=w3.at}go()});
  aT.addEventListener('input',()=>{w3.at=Math.min(+aT.value,+(w3.as-0.05).toFixed(2));go()});aS.addEventListener('change',say);aT.addEventListener('change',say);
  $$('#w3 [data-ws]').forEach(b=>b.addEventListener('click',()=>{w3.ws=b.dataset.ws.split(',').map(Number);w3.ws.forEach((v,i)=>$('#w3-w'+i).value=v);renderW3();say()}));
  renderW3()}

/* ---------- W4 ---------- */
const VOCAB=['Really','Through','Little','Good','Always','Number','Building','Question','Other','Think','Date','Want','Dual','Diffusion','For','Discrete','Data'];
const TARGET=[12,13,14,15,16],VK=VOCAB.length;
function posteriorUS(zt,x,at,as,K){const ats=at/as,den=K*at*(zt===x?1:0)+1-at,p=new Float64Array(K);
  for(let i=0;i<K;i++)p[i]=(K*at*(i===zt&&i===x?1:0)+(ats-at)*(i===zt?1:0)+(as-at)*(i===x?1:0)+(1-ats)*(1-as)/K)/den;return p}
function runUSDM(Tst,seed){const r=mulberry32(seed);let z=TARGET.map(()=>Math.min(VK-1,Math.floor(r()*VK)));const hist=[z.slice()];
  for(let n=1;n<=Tst;n++){const at=(n-1)/Tst,as=n/Tst;z=z.map((zl,l)=>{const p=posteriorUS(zl,TARGET[l],at,as,VK),u=r();let c=0;for(let k=0;k<VK;k++){c+=p[k];if(u<c)return k}return VK-1});hist.push(z.slice())}return hist}
function runDDT(Tst,seed){const r=mulberry32(seed+1000),eps=TARGET.map(()=>Array.from({length:VK},()=>gauss(r))),hist=[];
  for(let n=0;n<=Tst;n++){const a=n/Tst,s=Math.sqrt(Math.max(0,1-a*a));hist.push(TARGET.map((x,l)=>argmaxFirst(eps[l].map((e,k)=>(k===x?a:0)+s*e))))}
  const flips=TARGET.map((x,l)=>{let j=-1;for(let k=0;k<VK;k++)if(k!==x&&(j<0||eps[l][k]>eps[l][j]))j=k;const D=eps[l][j]-eps[l][x];return{from:VOCAB[j],D,a:D>0?D/Math.sqrt(1+D*D):null}});
  return{hist,flips}}
const changes=h=>TARGET.map((_,l)=>{let c=0;for(let n=1;n<h.length;n++)if(h[n][l]!==h[n-1][l])c++;return c});
const w4={T:16,seed:7,n:0,timer:null};
function computeW4(){w4.u=runUSDM(w4.T,w4.seed);w4.d=runDDT(w4.T,w4.seed)}
function renderW4(){const{T,n}=w4,box=$('#w4-box');setTxt('w4-no',n);setTxt('w4-seed','seed '+w4.seed);$('#w4-n').max=T;$('#w4-n').value=n;
  const panel=(title,hist,tri,flips)=>{const ch=changes(hist);let h=`<div class="w4-p"><h4>${title}　<span class="w4-meta">總改變次數 ${ch.reduce((a,b)=>a+b,0)}</span></h4><div class="w4-row">`;
    TARGET.forEach((x,l)=>{const v=hist[n][l],done=[...Array(n).keys()].filter(k=>hist[k+1][l]!==hist[k][l]).length;
      let strip='';for(let k=1;k<=T;k++)strip+=`<span class="${hist[k][l]!==hist[k-1][l]?'ch':''}${k===n?' cur':''}"></span>`;
      h+=`<div class="w4-col"><div class="tok${v===x?' ok':''}">${v===x?'✓ ':''}${VOCAB[v]}</div><div class="tl-strip${tri?' tri':''}" aria-hidden="true">${strip}</div><div class="w4-meta">全程改變 ${ch[l]} 次${flips?`<br>翻轉點 ${flips[l].a===null?'不翻轉':'α~* = '+flips[l].a.toFixed(4)}`:''}</div></div>`});
    return h+'</div></div>'};
  box.innerHTML=panel('USDM ancestral（已知真值的 denoiser）',w4.u,false,null)+panel('P<sub>DDT</sub>',w4.d.hist,true,w4.d.flips);
  $('#w4-play').textContent=w4.timer?'暫停':'播放'}
function sayW4(){const{n}=w4,cu=(h)=>{let c=0;for(let k=1;k<=n;k++)TARGET.forEach((_,l)=>{if(h[k][l]!==h[k-1][l])c++});return c};
  announce('w4',`第 ${n} 步：${w4.u[n].map(v=>VOCAB[v]).join(' ')}；USDM 已改變 ${cu(w4.u)} 次，P_DDT 已改變 ${cu(w4.d.hist)} 次`)}
function stopW4(){if(w4.timer){clearInterval(w4.timer);w4.timer=null}}
function initW4(){const seg=$('#w4-T');$$('button',seg).forEach(b=>{b.setAttribute('aria-pressed',String(+b.dataset.v===w4.T));b.addEventListener('click',()=>{stopW4();w4.T=+b.dataset.v;w4.n=0;$$('button',seg).forEach(x=>x.setAttribute('aria-pressed',String(x===b)));computeW4();renderW4();sayW4()})});
  $('#w4-n').addEventListener('input',e=>{stopW4();w4.n=+e.target.value;renderW4()});$('#w4-n').addEventListener('change',sayW4);
  $('#w4-end').addEventListener('click',()=>{stopW4();w4.n=w4.T;renderW4();sayW4()});
  $('#w4-re').addEventListener('click',()=>{stopW4();w4.seed++;w4.n=0;computeW4();renderW4();sayW4()});
  $('#w4-play').addEventListener('click',()=>{if(w4.timer){stopW4();renderW4();sayW4();return}
    if(RM){w4.n=w4.T;renderW4();sayW4();return}if(w4.n>=w4.T)w4.n=0;
    w4.timer=setInterval(()=>{w4.n++;if(w4.n>=w4.T){stopW4();renderW4();sayW4()}else renderW4()},250);renderW4()});
  computeW4();renderW4()}

/* ---------- W5 ---------- */
const w5={K:8,a:0.9,lt:-1,seed:3};
function w5Compute(K,at,tau,seed){const r=mulberry32(seed),s=Math.sqrt(1-at*at),w=new Float64Array(K);for(let k=0;k<K;k++)w[k]=(k===0?at:0)+s*gauss(r);
  let mx=-Infinity;for(let k=0;k<K;k++)if(w[k]>mx)mx=w[k];let Z=0;const e=new Float64Array(K);for(let k=0;k<K;k++){e[k]=Math.exp((w[k]-mx)/tau);Z+=e[k]}
  let H=0,pm=0;for(let k=0;k<K;k++){const p=e[k]/Z;if(p>0)H-=p*Math.log(p);if(p>pm)pm=p}
  let j=1;for(let k=2;k<K;k++)if(w[k]>w[j])j=k;
  return{w,p:k=>e[k]/Z,px:e[0]/Z,pm,H,g:w[0]-w[j],am:argmaxFirst(w)}}
function renderW5(){const{K,a,lt,seed}=w5,tau=Math.pow(10,lt),R=w5Compute(K,a,tau,seed);
  setTxt('w5-ao',a.toFixed(2));setTxt('w5-to','τ = '+(tau<0.01?tau.toPrecision(2):tau.toFixed(3)));setTxt('w5-seed','seed '+seed);
  let idx;if(K<=8)idx=[...Array(K).keys()];else{const o=[];for(let k=1;k<K;k++){if(o.length<7||R.w[k]>R.w[o[6]]){o.push(k);o.sort((x,y)=>R.w[y]-R.w[x]);if(o.length>7)o.pop()}}idx=[0,...o]}
  const box=$('#w5-fig'),svg=$('svg',box),W=widthOf(box,280,560),H=330;svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const n=idx.length,x0=44,x1=W-8,X=i=>x0+(i+0.5)*(x1-x0)/n,bw=(x1-x0)/n*0.55,g=el('g',null,svg);
  const vals=idx.map(k=>R.w[k]),lo=Math.min(-1,...vals),hi=Math.max(1.5,...vals),Y1=lin(lo,hi,140,22);
  tx(g,4,12,'w（arg max 加外框）',{class:'al'});el('line',{x1:x0,x2:x1,y1:Y1(0),y2:Y1(0),class:'axl'},g);tx(g,x0-8,Y1(0)+4,'0',{class:'tk','text-anchor':'end'});
  idx.forEach((k,i)=>{const y=Y1(R.w[k]),y0=Y1(0);el('rect',{x:X(i)-bw/2,y:Math.min(y,y0),width:bw,height:Math.abs(y-y0),class:k===0?'bar-sig':'bar-ink'},g);
    if(k===R.am)el('rect',{x:X(i)-bw/2-3,y:Math.min(y,y0)-3,width:bw+6,height:Math.abs(y-y0)+6,class:'frame'},g)});
  const Y2=lin(0,1,300,186);tx(g,4,176,'softmax(w/τ)；虛線為 one-hot',{class:'al'});
  [0,0.5,1].forEach(v=>{el('line',{x1:x0,x2:x1,y1:Y2(v),y2:Y2(v),class:'grid'},g);tx(g,x0-8,Y2(v)+4,String(v),{class:'tk','text-anchor':'end'})});
  idx.forEach((k,i)=>{const p=R.p(k);el('rect',{x:X(i)-bw/2,y:Y2(p),width:bw,height:Y2(0)-Y2(p),class:k===0?'bar-sig':'bar-ink'},g);
    if(k===R.am)el('rect',{x:X(i)-bw/2,y:Y2(1),width:bw,height:Y2(0)-Y2(1),class:'ref'},g);
    tx(g,X(i),318,k===0?'x':String(k),{class:'tk','text-anchor':'middle'})});
  const fp=v=>1-v<1e-12&&v<1?'1 − <1e-12':v.toFixed(4);
  setTxt('w5-r-ok',R.am===0?'是':'否（第 '+R.am+' 類）');setTxt('w5-r-g',R.g.toFixed(4));setTxt('w5-r-gt',(R.g/tau).toFixed(1));setTxt('w5-r-px',fp(R.px));setTxt('w5-r-pm',fp(R.pm));setTxt('w5-r-H',(Math.abs(R.H)<1e-12?0:R.H).toFixed(4));
  svg.setAttribute('aria-label',`K = ${K}，alpha~ = ${a.toFixed(2)}，tau = ${tau.toPrecision(2)}：p_x = ${R.px.toFixed(4)}`);return R}
function initW5(){let raf=0;const go=()=>{cancelAnimationFrame(raf);raf=requestAnimationFrame(renderW5)};const say=()=>announce('w5',`p_x = ${$('#w5-r-px').textContent}，g/τ = ${$('#w5-r-gt').textContent}`);
  const seg=$('#w5-K');$$('button',seg).forEach(b=>{b.setAttribute('aria-pressed',String(+b.dataset.v===w5.K));b.addEventListener('click',()=>{w5.K=+b.dataset.v;$$('button',seg).forEach(x=>x.setAttribute('aria-pressed',String(x===b)));renderW5();say()})});
  $('#w5-a').addEventListener('input',e=>{w5.a=+e.target.value;go()});$('#w5-t').addEventListener('input',e=>{w5.lt=+e.target.value;go()});
  $('#w5-a').addEventListener('change',say);$('#w5-t').addEventListener('change',say);
  $('#w5-paper').addEventListener('click',()=>{w5.lt=-3;$('#w5-t').value=-3;renderW5();say()});
  $('#w5-re').addEventListener('click',()=>{w5.seed++;renderW5();say()});renderW5()}

/* ---------- S8 loss table ---------- */
function bars(m,xth,a,K){return[[...Array(K)].map((_,j)=>K*a*(j===m?1:0)+(1-a)),xth.map(q=>K*a*q+(1-a))]}
function f49(i,m,xth,a,ad,K){const[xb,xbt]=bars(m,xth,a,K);let s=0;for(let j=0;j<K;j++)s+=xb[j]/xb[i]*Math.log(xbt[i]*xb[j]/(xbt[j]*xb[i]));return -ad/(K*a)*(K/xb[i]-K/xbt[i]-s)}
function f17(i,m,xth,a,ad,K){const[xb,xbt]=bars(m,xth,a,K),kap=(1-a)/(K*a+1-a),eq=i===m;let S=0;for(let j=0;j<K;j++)S+=Math.log(xbt[i]/xbt[j]);
  let v=K/xb[i]-K/xbt[i]-(eq?kap:1)*S;if(!eq)v-=K*a/(1-a)*Math.log(xbt[i]/xbt[m]);v-=(eq?(K-1)*kap:-1/kap)*Math.log(kap);return -ad/(K*a)*v}
function lossXth(){const r=mulberry32(5),raw=[...Array(7)].map(()=>r()),Z=raw.reduce((a,b)=>a+b,0);return raw.map(v=>v/Z)}
function fillLossTable(){const xth=lossXth(),rows=$$('#loss-tab tbody tr');[[2,2],[4,2],[0,2]].forEach(([i,m],r)=>{const A=f49(i,m,xth,0.37,-1,7),B=f17(i,m,xth,0.37,-1,7),c=$$('td',rows[r]);
  c[1].textContent=A.toFixed(10).replace('-','−');c[2].textContent=B.toFixed(10).replace('-','−');const d=Math.abs(A-B);c[3].textContent=d===0?'0':d.toExponential(1)})}

/* ---------- timeline ---------- */
const TL_GROUP={'1503.03585':['gauss','duo-gaussian-diffusion-foundations'],'2006.11239':['gauss','duo-gaussian-diffusion-foundations'],'2011.13456':['gauss','duo-gaussian-diffusion-foundations'],'2107.00630':['gauss','duo-gaussian-diffusion-foundations'],
 '2107.03006':['disc','duo-discrete-diffusion-masked-and-uniform'],'2205.14987':['disc','duo-discrete-diffusion-masked-and-uniform'],'2310.16834':['disc','duo-discrete-diffusion-masked-and-uniform'],'2406.07524':['disc','duo-discrete-diffusion-masked-and-uniform'],'2412.10193':['disc','duo-discrete-diffusion-masked-and-uniform'],
 '1611.01144':['relax','duo-curriculum-softmax-relaxation'],'1611.00712':['relax','duo-curriculum-softmax-relaxation'],
 '2202.00512':['distill','duo-consistency-distillation-to-discrete'],'2303.01469':['distill','duo-consistency-distillation-to-discrete'],'2410.21035':['distill','duo-consistency-distillation-to-discrete'],
 '2506.10892':['me','duo-argmax-duality-theory']};
const TL_NAME={gauss:'高斯擴散',disc:'離散擴散',relax:'鬆弛',distill:'蒸餾',me:'本文',other:'其他'};
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

/* ---------- toc, theme ---------- */
function initToc(){const secs=$$('[data-toc]'),ol=$('#toc'),bar=$('#tocbar');
  secs.forEach(s=>{ol.insertAdjacentHTML('beforeend',`<li><a href="#${s.id}"><span class="n">${s.dataset.toc}</span><span>${s.dataset.tt}</span></a></li>`);bar.insertAdjacentHTML('beforeend',`<a href="#${s.id}">${s.dataset.toc} ${s.dataset.tt}</a>`)});
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;$$('.toc a,.tocbar a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+id));const ab=$(`.tocbar a[href="#${id}"]`);if(ab&&bar.offsetParent)bar.scrollTo({left:ab.offsetLeft-16,behavior:RM?'auto':'smooth'})}})},{rootMargin:'-40% 0px -55% 0px'});
  secs.forEach(s=>io.observe(s));
  addEventListener('scroll',()=>$('#appbar').classList.toggle('raised',scrollY>8),{passive:true})}
function initTheme(){let saved=null;try{saved=localStorage.getItem('pm-theme')}catch(e){}if(saved)document.documentElement.dataset.theme=saved;
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){}})}

/* ---------- self test (design §5, ★ items) ---------- */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t)=>Math.abs(a-b)<=t+1e-12;
  [[0.6,2,.404117],[0.6,5,.265547],[0.6,10,.183770],[0.85,30522,.007730],[0.97,30522,.452299]].forEach(([a,K,v])=>{const T=Top(a,K);ok(near(T,v,Math.max(1e-6,1e-5*v)),`T(${a},${K})=${T.toFixed(7)}`)});
  [0.1,0.3,0.6,0.9].forEach(a=>{const s=Math.sqrt(1-a*a);ok(near(Top(a,2),2*Phi(a/(Math.SQRT2*s))-1,1e-6),'K=2 closed form a='+a)});
  ok(Top(0,5)===0&&Top(0,30522)===0,'T(0)=0');
  const B=bandB(30522);ok(near(B[0],.9238,1e-4)&&near(B[1],.9856,1e-4),`band B ${B[0].toFixed(5)} ${B[1].toFixed(5)}`);
  const s1=w1Sample(5,0.6,7);ok(near(s1.freq[0],.4142,5e-4),'W1 MC p_x='+s1.freq[0]);
  const P=pushK([1.2,0.6,0],0.9,0.6),Kk=usdmKernel([1.2,0.6,0],0.9,0.6);
  [.5638,.2963,.1399].forEach((v,i)=>ok(near(P[i],v,1e-4),'W3 P'+i+'='+P[i].toFixed(5)));
  [.6109,.1946,.1946].forEach((v,i)=>ok(near(Kk.Q[i],v,1e-4),'W3 Q'+i));ok(near(Kk.a,.4163,1e-4),'W3 alpha_t|s');
  ok(near(P.reduce((a,b)=>a+b,0),1,1e-6),'W3 sum P');ok(near(0.5*P.reduce((p,v,i)=>p+Math.abs(v-Kk.Q[i]),0),.1018,1e-4),'W3 TV');
  const P3=pushK([1.2,0.3,0.3],0.9,0.6);[.5760,.2120,.2120].forEach((v,i)=>ok(near(P3[i],v,1e-4),'W3 equal P'+i));
  const u=runUSDM(16,7),d=runDDT(16,7);ok(changes(u).join()==='2,3,4,3,3','W4 USDM changes '+changes(u));ok(changes(d.hist).join()==='0,1,1,1,1','W4 DDT changes '+changes(d.hist));
  ok(near(d.flips[3].a,.9451,1e-4),'W4 flip Discrete');
  [4,8,16,32,64].forEach(T=>[7,8].forEach(sd=>{ok(runUSDM(T,sd)[T].join()===TARGET.join(),`W4 USDM final T=${T} seed=${sd}`);ok(changes(runDDT(T,sd).hist).every(c=>c<=1),`W4 DDT <=1 T=${T} seed=${sd}`)}));
  {let mx=0;const rr=mulberry32(99);for(let q=0;q<1000;q++){const K=2+Math.floor(rr()*16),at=rr()*0.99,as=at+rr()*(1-at),p=posteriorUS(Math.floor(rr()*K),Math.floor(rr()*K),at,as,K);mx=Math.max(mx,Math.abs(p.reduce((a,b)=>a+b,0)-1))}ok(mx<1e-12,'posterior sums')}
  ok(near(w5Compute(8,0.9,1,3).px,.3891,1e-4),'W5 K8 tau1');ok(near(w5Compute(8,0.9,0.1,3).px,.9999,1e-4),'W5 K8 tau.1');
  ok(near(w5Compute(30522,0.9,1,3).g,-.3832,1e-4),'W5 K30522 .9 g');ok(near(w5Compute(30522,0.97,1,3).g,.2543,1e-4),'W5 K30522 .97 g');
  const xth=lossXth();[[2,2],[4,2],[0,2]].forEach(([i,m])=>ok(near(f49(i,m,xth,.37,-1,7),f17(i,m,xth,.37,-1,7),1e-12),'f49=f17 i='+i));
  ok(near(f49(2,2,xth,.37,-1,7),-1.7499022162,1e-9),'f49 value');const xo=[0,0,1,0,0,0,0];ok(near(f49(2,2,xo,.37,-1,7),0,1e-12)&&near(f49(4,2,xo,.37,-1,7),0,1e-12),'oracle 0');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ---------- boot ---------- */
initTheme();initToc();initW1();const W2A=initW2('w2a',{K:10,a:0.6,zoom:false,bands:false}),W2B=initW2('w2b',{K:30522,a:0.97,zoom:true,bands:true});
initW3();initW4();initW5();fillLossTable();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderW1();W2A.render();W2B.render();renderW3();renderW5()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath();setTimeout(selfTest,50)});
})();
