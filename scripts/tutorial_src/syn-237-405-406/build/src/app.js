/* ---------- helpers (shell of the 405/406 pages) ---------- */
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
function el(tag,a,p){const e=document.createElementNS(NS,tag);if(a)for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function tx(p,x,y,s,a){const t=el('text',Object.assign({x,y},a||{}),p);t.textContent=s;return t}
const lin=(d0,d1,r0,r1)=>v=>r0+(v-d0)/(d1-d0)*(r1-r0);
const lgs=(d0,d1,r0,r1)=>{const a=Math.log(d0),b=Math.log(d1);return v=>r0+(Math.log(v)-a)/(b-a)*(r1-r0)};
function lp(pts){let s='';for(let i=0;i<pts.length;i++)s+=(i?'L':'M')+pts[i][0].toFixed(1)+' '+pts[i][1].toFixed(1);return s}
function widthOf(box,min,max){return Math.max(min,Math.min(max,Math.floor(box.clientWidth||max)))}
let clipN=0;
function frame(svg,o){
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${o.W} ${o.H}`);svg.setAttribute('width',o.W);svg.setAttribute('height',o.H);
  const m=o.m,x0=m.l,x1=o.W-m.r,y0=o.H-m.b,y1=m.t;
  const X=(o.xlog?lgs:lin)(o.x[0],o.x[1],x0,x1),Y=(o.ylog?lgs:lin)(o.y[0],o.y[1],y0,y1);
  const g=el('g',null,svg);
  (o.yt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:String(v),yy=Y(val);el('line',{x1:x0,x2:x1,y1:yy,y2:yy,class:'grid'},g);tx(g,x0-8,yy+4,lab,{class:'tk','text-anchor':'end'})});
  (o.xt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:String(v),xx=X(val);el('line',{x1:xx,x2:xx,y1:y0,y2:y0+4,class:'axl'},g);tx(g,xx,y0+18,lab,{class:'tk','text-anchor':'middle'})});
  el('line',{x1:x0,x2:x1,y1:y0,y2:y0,class:'axl'},g);
  if(o.xl)tx(g,(x0+x1)/2,o.H-4,o.xl,{class:'al','text-anchor':'middle'});
  if(o.yl)tx(g,4,12,o.yl,{class:'al'});
  const id='clip'+(++clipN),d=el('defs',null,svg),c=el('clipPath',{id},d);el('rect',{x:x0,y:y1-2,width:x1-x0,height:y0-y1+4},c);
  const pg=el('g',{'clip-path':`url(#${id})`},svg);
  return {X,Y,g,pg,x0,x1,y0,y1};
}
function radiogroup(box,val,onChange){const bs=$$('button',box);
  const set=(v,focus)=>{bs.forEach(b=>{const on=b.dataset.v===String(v);b.setAttribute('aria-checked',String(on));b.tabIndex=on?0:-1;if(on&&focus)b.focus()})};
  bs.forEach((b,i)=>{b.type='button';b.setAttribute('role','radio');b.addEventListener('click',()=>{if(b.disabled)return;set(b.dataset.v);onChange(b.dataset.v)});
    b.addEventListener('keydown',e=>{const d={ArrowRight:1,ArrowDown:1,ArrowLeft:-1,ArrowUp:-1}[e.key];if(!d)return;e.preventDefault();let j=i;do{j=(j+d+bs.length)%bs.length}while(bs[j].disabled&&j!==i);set(bs[j].dataset.v,true);onChange(bs[j].dataset.v)})});
  set(val);return set}
function sched(fn){let raf=0;return()=>{if(raf)return;raf=requestAnimationFrame(()=>{raf=0;fn()})}}
const fe=v=>{if(!isFinite(v))return '—';if(v===0)return '0';const a=Math.abs(v);return(a>=1e-3&&a<1e4)?(+v.toPrecision(3)).toString():v.toExponential(2)};
function ro(node,rows){node.innerHTML=rows.map(([k,v,acc])=>`<dt>${k}</dt><dd${acc?' class="acc"':''}>${v}</dd>`).join('')}
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},400)}
function chartW(box){return widthOf(box,260,640)}
const narrow=()=>innerWidth<600;
function slider(id,fmt,label,onv){const i=$('#'+id),o=$('#'+id+'o');const up=()=>{const v=+i.value;o.textContent=fmt(v);i.setAttribute('aria-valuetext',label+' = '+fmt(v));onv(v)};i.addEventListener('input',up);up();return i}

/* ---------- W0 ---------- */
const w0s={ls:-1,e:0.2,u:'x0'};
function renderW0(){const s=10**w0s.ls,e=w0s.e,v=w0s.u==='x0'?Math.sqrt(1-s*s)*e/(s*s):e/s;
  ro($('#w0-ro'),[['σ',s.toFixed(3)],['ᾱ = 1 − σ²',(1-s*s).toFixed(4)],[w0s.u==='x0'?'x₀ 誤差 e':'ε 誤差 η',e.toFixed(2)],['公式',w0s.u==='x0'?'√ᾱ·e / σ²':'η / σ'],['對應的分數誤差',(+v.toPrecision(4)).toString(),1]]);
  announce('w0',`σ = ${s.toFixed(3)}，分數誤差 ${(+v.toPrecision(4))}`)}
function initW0(){const R=sched(renderW0);slider('w0-s',v=>(10**v).toFixed(3),'σ',v=>{w0s.ls=v;R()});
  $('#w0-e').addEventListener('input',e=>{const v=+e.target.value;w0s.e=isFinite(v)?Math.max(0,Math.min(1,v)):0;R()});
  radiogroup($('#w0-u'),w0s.u,v=>{w0s.u=v;R()});renderW0()}

/* ---------- W1 ---------- */
const w1s={a:1,l:1.017,ls:-1,g:0.1,ip:false,ipShown:false,z:'full'};
const A1000=respace('cosine',1000);
function renderW1(){const s=w1s,svg=$('#w1-fig svg'),W=chartW($('#w1-fig')),H=330+12,zoom=s.z==='zoom',yr=zoom?[0.95,1.01]:[0,1.1];
  const G=frame(svg,{W,H,m:{l:46,r:12,t:48,b:40},x:[0.005,1],xlog:true,y:yr,xl:'σ（對數）',
    xt:[[0.01,'0.01'],[0.1,'0.1'],[1,'1']].concat(narrow()?[]:[[0.03,'0.03'],[0.3,'0.3']]),
    yt:zoom?[0.95,0.96,0.97,0.98,0.99,1.0].map(v=>[v,v.toFixed(2)]):[0,0.2,0.4,0.6,0.8,1.0].map(v=>[v,v.toFixed(1)])});
  // top t axis (cosine, T = 1000)
  const tl=narrow()?[1,10,100,500]:[1,10,50,100,200,500];
  el('line',{x1:G.x0,x2:G.x1,y1:G.y1-14,y2:G.y1-14,class:'faint'},G.g);
  tl.forEach(t=>{const xx=G.X(sigmaOfT(A1000,t));el('line',{x1:xx,x2:xx,y1:G.y1-18,y2:G.y1-14,class:'axl'},G.g);tx(G.g,xx,G.y1-22,String(t),{class:'tk','text-anchor':'middle'})});
  tx(G.g,G.x1,G.y1-36,'上軸：t（cosine 排程，推定）',{class:'tk','text-anchor':'end'});
  const N=200,xs=[];for(let i=0;i<=N;i++)xs.push(0.005*Math.pow(1/0.005,i/N));
  const yc=v=>Math.max(yr[0]-0.05,Math.min(yr[1]+0.05,v));
  el('path',{d:lp(xs.map(x=>[G.X(x),G.Y(yc(mTS(x,s.a)))])),class:'par'},G.pg);
  el('path',{d:lp([[G.x0,G.Y(yc(mES(s.l)))],[G.x1,G.Y(yc(mES(s.l)))]]),class:'perp dash'},G.pg);
  if(s.ipShown&&s.ip){el('path',{d:lp([[G.x0,G.Y(yc(mIPman(s.g)))],[G.x1,G.Y(yc(mIPman(s.g)))]]),class:'acc dot'},G.pg);
    el('path',{d:lp(xs.filter(x=>x<0.999).map(x=>[G.X(x),G.Y(yc(mIPgauss(x,s.g,0.25)))])),class:'acc-thin dash'},G.pg)}
  if(!narrow()){const xl=G.X(0.012);tx(G.g,xl,Math.max(G.y1+12,Math.min(G.y0-6,G.Y(yc(mTS(0.012,s.a)))-8)),'TS',{class:'lab'});tx(G.g,G.X(0.012),G.Y(yc(mES(s.l)))+16,'ES',{class:'lab'})}
  const sig=10**s.ls;el('line',{x1:G.X(sig),x2:G.X(sig),y1:G.y1,y2:G.y0,class:'cur'},G.g);
  // crossing
  let crossTxt='—';
  if(s.l===1)crossTxt='—（λ = 1 時 ES 不縮放）';else if(s.l<1)crossTxt='無交點（ES 放大 ε）';
  else{const sc=sigmaCross(s.l,s.a);crossTxt=sc.toFixed(4);
    if(sc<=0.991&&sc>=0.005){const cx=G.X(sc),cy=G.Y(mES(s.l));el('circle',{cx,cy,r:5,class:'kdot'},G.pg);
      let tc=1;while(tc<1000&&sigmaOfT(A1000,tc)<sc)tc++;tx(G.g,Math.min(cx-6,G.x1-4),cy+(zoom?22:-12),`σ* = ${sc.toFixed(4)}（t ≈ ${tc}）`,{class:'lab','text-anchor':'end'})}
    else tx(G.g,G.x1-4,G.Y(mES(s.l))-10,`σ* = ${sc.toFixed(4)} →`,{class:'lab','text-anchor':'end'})}
  if(zoom&&mTS(0.5,s.a)<0.95)tx(G.g,G.X(0.05),G.y0-8,'↓ TS 低於 0.95',{class:'tk','text-anchor':'middle'});
  const rows=[['σ',sig.toFixed(3)],['TS 乘數 σ^α',mTS(sig,s.a).toFixed(4)],['ES 乘數 1/λ',mES(s.l).toFixed(4)],['交點 σ*',crossTxt],['ES 等效指數 α_eq',isFinite(alphaEq(s.l,sig))?alphaEq(s.l,sig).toFixed(5):'—']];
  if(s.ipShown&&s.ip)rows.push(['IP 法向倍數',mIPman(s.g).toFixed(4),1],['IP 高斯玩具倍數',isFinite(mIPgauss(sig,s.g,0.25))?mIPgauss(sig,s.g,0.25).toFixed(4):'—',1]);
  ro($('#w1-ro'),rows);
  announce('w1',`σ = ${sig.toFixed(3)}：TS ${mTS(sig,s.a).toFixed(4)}，ES ${mES(s.l).toFixed(4)}`)}
function initW1(){const R=sched(renderW1);
  slider('w1-a',v=>v.toFixed(2),'α',v=>{w1s.a=v;R()});slider('w1-l',v=>v.toFixed(3),'λ',v=>{w1s.l=v;R()});
  slider('w1-s',v=>(10**v).toFixed(3),'σ',v=>{w1s.ls=v;R()});slider('w1-g',v=>v.toFixed(2),'γ',v=>{w1s.g=v;R()});
  $('#w1-ipc').addEventListener('change',e=>{w1s.ip=e.target.checked;R()});
  radiogroup($('#w1-z'),w1s.z,v=>{w1s.z=v;R()});renderW1()}

/* ---------- W2 ---------- */
const W2_SIG=[0.2,0.1,0.05,0.02,0.01];
const w2s={i:2,l:1.017,a:1,g:0.1,o:'es',ipShown:false};
const w2cache=new Map();
function w2dens(sigma,alpha){const k=sigma+'|'+alpha;if(w2cache.has(k))return w2cache.get(k);if(w2cache.size>40)w2cache.clear();
  const d=c2Marginal({a:1,b:1,sigma,alpha}).dens;w2cache.set(k,d);return d}
const PD=refCurve('pdata',1,1,0,180),UN=refCurve('unif',1,1,0,180);
function w2all(){const s=w2s,sg=W2_SIG[s.i];
  return {sg,ex:w2dens(sg,0),es:w2dens(sg,alphaEq(s.l,sg)),ts:w2dens(sg,s.a),ip:w2dens(sg*Math.sqrt(1+s.g*s.g),0),lim:vmPow(1/s.l,180)}}
const shift=d=>{const n=d.length,o=new Float64Array(n);for(let j=0;j<n;j++)o[j]=d[(j+n/2)%n];return o};  // index j -> phi = t - theta0
function drawW2(svg,W,H,D,ph0,ph1,inset){const s=w2s,nt=180,dt=2*Math.PI/nt,phi=j=>-Math.PI+(j+.5)*dt;
  const cur={es:D.es,ts:D.ts,ip:D.ip}[s.o],shown=[D.ex,cur,PD].concat(s.o==='es'?[D.lim]:[]).map(shift);let mx=0,mn=Infinity;
  for(const a of shown)for(let j=0;j<nt;j++){const p=phi(j);if(p>=ph0&&p<=ph1){mx=Math.max(mx,a[j]);mn=Math.min(mn,a[j])}}
  const yr=inset?[mn-(mx-mn)*0.05,mx+(mx-mn)*0.08]:[0,mx*1.05],dg=inset?3:2;
  const G=frame(svg,{W,H,m:{l:inset?48:46,r:10,t:10,b:inset?26:40},x:[ph0,ph1],y:yr,xl:inset?'':'t − θ₀（弧度）',
    xt:inset?[[-0.1,'−0.1'],[0,'0'],[0.1,'0.1']]:[[-Math.PI,'−π'],[-Math.PI/2,'−π/2'],[0,'0'],[Math.PI/2,'π/2'],[Math.PI,'π']],
    yt:[yr[0],(yr[0]+yr[1])/2,yr[1]].map(v=>[v,v.toFixed(dg)])});
  const path=(a,cls,mk)=>{const S=shift(a),pts=[];for(let j=0;j<nt;j++){const p=phi(j);if(p>=ph0-dt&&p<=ph1+dt)pts.push([G.X(p),G.Y(S[j])])}el('path',{d:lp(pts),class:cls},G.pg);
    if(mk)pts.forEach((q,i)=>{if(i%(inset?3:20)===(inset?1:10)){if(mk==='o')el('circle',{cx:q[0],cy:q[1],r:3.5,class:'mk-h par-s'},G.pg);else el('rect',{x:q[0]-3.5,y:q[1]-3.5,width:7,height:7,class:'mk-h acc-s'},G.pg)}})};
  path(PD,'ref');path(UN,'ref dot');path(D.ex,'ink thin');
  if(s.o==='es'){path(D.lim,'perp dot thin');path(D.es,'perp dash')}else if(s.o==='ts')path(D.ts,'par','o');else path(D.ip,'acc dot','s');
  if(inset)tx(G.g,G.x0+4,G.y1+12,'峰頂放大',{class:'tk'});
  return G}
function renderW2(){const s=w2s,D=w2all(),W=chartW($('#w2-fig'));
  drawW2($('#w2-fig svg'),W,300,D,-Math.PI,Math.PI,false);
  drawW2($('#w2-ins svg'),Math.min(W,360),160,D,-0.2,0.2,true);
  const rows=[['精確分數',D.ex],['ES 型 c = 1/λ',D.es],['TS c = σ^α',D.ts]].concat(s.ipShown?[['IP σ√(1+γ²)',D.ip]]:[]);
  $('#w2-tb tbody').innerHTML=rows.map(([n,d])=>`<tr><td>${n}</td><td class="num" data-h="TV(p_data)">${tv(d,PD).toFixed(4)}</td><td class="num" data-h="TV(均勻)">${tv(d,UN).toFixed(4)}</td><td class="num" data-h="最大／最小">${maxOverMin(d).toFixed(1)}</td></tr>`).join('');
  const cur={es:D.es,ts:D.ts,ip:D.ip}[s.o],tvx=tv(cur,D.ex);
  let f='讀數由本頁以 No. 237 的單位圓封閉式數值積分即時計算 <span class="tag">本頁推導</span>。';
  if(s.o==='es')f+=` σ → 0 極限 p^(1/λ) 與 p_data 的 TV = ${tv(D.lim,PD).toFixed(4)}。`;
  if(tvx<0.001)f+=` 所選修正與精確分數幾乎重合（TV = ${tvx.toFixed(4)}）。`;
  $('#w2-f').innerHTML=f;
  $('#w2-lg-es').classList.toggle('hide',s.o!=='es');$('#w2-lg-lim').classList.toggle('hide',s.o!=='es');$('#w2-lg-ts').classList.toggle('hide',s.o!=='ts');$('#w2-lg-ip').classList.toggle('hide',s.o!=='ip');
  announce('w2',`σ = ${D.sg}：ES 型 TV(p_data) ${tv(D.es,PD).toFixed(4)}，TS TV(均勻) ${tv(D.ts,UN).toFixed(4)}`)}
function initW2(){const R=sched(renderW2);
  radiogroup($('#w2-s'),w2s.i,v=>{w2s.i=+v;R()});w2s.setO=radiogroup($('#w2-o'),w2s.o,v=>{w2s.o=v;R()});
  $('#w2-ipb').disabled=true;
  slider('w2-l',v=>v.toFixed(3),'λ',v=>{w2s.l=v;R()});slider('w2-a',v=>v.toFixed(2),'α',v=>{w2s.a=v;R()});slider('w2-g',v=>v.toFixed(2),'γ',v=>{w2s.g=v;R()});
  renderW2()}

/* ---------- W3 ---------- */
const w3s={k:'cosine',Tp:100,e:0.2,g:0.1,t:20};
function w3band(R,g){let a=-1,b=-1;const runs=[];R.forEach(r=>{const inb=r.gM>=g/2&&r.gM<=2*g;if(inb){if(a<0)a=r.t;b=r.t}else if(a>=0){runs.push([a,b]);a=-1}});if(a>=0)runs.push([a,b]);return runs}
function renderW3(){const s=w3s,R=gammaMatch(s.k,s.Tp,s.e,'x0'),W=chartW($('#w3-fig')),H=320,lg=s.Tp===1000;
  const G=frame($('#w3-fig svg'),{W,H,m:{l:52,r:12,t:12,b:40},x:[1,s.Tp-1],xlog:lg,y:[1e-4,1e2],ylog:true,xl:'時間步 t'+(lg?'（對數）':''),
    xt:lg?[1,10,100,999].map(v=>[v,String(v)]):(s.Tp===20?[1,5,10,15,19]:[1,25,50,75,99]).map(v=>[v,String(v)]),
    yt:[1e-4,1e-2,1,1e2].map(v=>[v,v>=1?String(v):v.toExponential(0).replace('e-','e−')])});
  const runs=w3band(R,s.g),half=lg?0.5:0.5;
  runs.forEach(([a,b])=>{const xa=G.X(Math.max(1,a-half)),xb=G.X(Math.min(s.Tp-1,b+half));el('rect',{x:Math.min(xa,xb),y:G.y1,width:Math.abs(xb-xa)||2,height:G.y0-G.y1,class:'filla'},G.pg)});
  const yc=v=>G.Y(Math.max(5e-5,Math.min(2e2,v>0?v:5e-5)));
  if(s.e>0){el('path',{d:lp(R.map(r=>[G.X(r.t),yc(r.g1)])),class:'acc-thin dash'},G.pg);el('path',{d:lp(R.map(r=>[G.X(r.t),yc(r.gM)])),class:'acc'},G.pg)}
  el('line',{x1:G.x0,x2:G.x1,y1:G.Y(s.g),y2:G.Y(s.g),class:'ref'},G.pg);
  const r=R[s.t-1];el('line',{x1:G.X(s.t),x2:G.X(s.t),y1:G.y1,y2:G.y0,class:'cur'},G.g);
  if(s.e===0)tx(G.g,(G.x0+G.x1)/2,G.y0-10,'e = 0：沒有多出的變異數',{class:'lab','text-anchor':'middle'});
  ro($('#w3-ro'),[['t',String(s.t)],['σ_t',r.sigma.toFixed(3)],['c₁,t+1',fe(r.c1)],['單步 γ₁',fe(r.g1)],['多步 γ_M',fe(r.gM),1],['δ_t',fe(r.delta)],['一致區段',runs.length?runs.map(([a,b])=>a===b?`t = ${a}`:`t = ${a}..${b}`).join('、'):'無']]);
  $('#w3-pre').innerHTML=`<span class="tag">推定</span> 依推定：e 為常數、${s.k} 排程`;
  announce('w3',`t = ${s.t}：多步 γ ${fe(r.gM)}；一致區段 ${runs.length?runs.map(x=>x.join('..')).join('、'):'無'}`)}
function w3setTp(v){w3s.Tp=+v;const ti=$('#w3-t');ti.max=w3s.Tp-1;if(w3s.t>w3s.Tp-1){w3s.t=w3s.Tp-1}ti.value=w3s.t;$('#w3-to').textContent=w3s.t;ti.setAttribute('aria-valuetext','t = '+w3s.t)}
function initW3(){const R=sched(renderW3);
  radiogroup($('#w3-k'),w3s.k,v=>{w3s.k=v;R()});radiogroup($('#w3-T'),w3s.Tp,v=>{w3setTp(v);R()});
  slider('w3-e',v=>v.toFixed(2),'e',v=>{w3s.e=v;R()});slider('w3-g',v=>v.toFixed(2),'參考 γ',v=>{w3s.g=v;R()});
  slider('w3-t',v=>String(v),'t',v=>{w3s.t=v;R()});renderW3()}

/* ---------- reveal IP (after S6) ---------- */
function revealIP(scroll){if(w1s.ipShown&&!scroll)return;w1s.ipShown=true;w1s.ip=$('#w1-ipc').checked;w2s.ipShown=true;
  $$('.ipx').forEach(n=>n.classList.remove('hide'));$('#w2-ipb').disabled=false;$('#w2-ipb').textContent='IP';renderW1();renderW2();
  if(scroll)$('#w1').scrollIntoView({block:'start'})}
function initReveal(){$('#reveal-ip').addEventListener('click',()=>revealIP(true));
  const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){revealIP(false);io.disconnect()}},{rootMargin:'0px 0px -60% 0px'});io.observe($('#ip'))}

/* ---------- toc, theme ---------- */
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
function initToc(){const secs=$$('[data-toc]'),ol=$('#toc'),bar=$('#tocbar');
  secs.forEach(s=>{ol.insertAdjacentHTML('beforeend',`<li><a href="#${s.id}"><span class="n">${s.dataset.toc}</span><span>${s.dataset.tt}</span></a></li>`);bar.insertAdjacentHTML('beforeend',`<a href="#${s.id}">${s.dataset.toc} ${s.dataset.tt}</a>`)});
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;$$('.toc a,.tocbar a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+id));const ab=$(`.tocbar a[href="#${id}"]`);if(ab&&bar.offsetParent)bar.scrollTo({left:ab.offsetLeft-16,behavior:'auto'})}})},{rootMargin:'-40% 0px -55% 0px'});
  secs.forEach(s=>io.observe(s));
  addEventListener('scroll',()=>$('#appbar').classList.toggle('raised',scrollY>8),{passive:true})}
function initTheme(){let saved=null;try{saved=localStorage.getItem('pm-theme')}catch(e){}if(saved)document.documentElement.dataset.theme=saved;
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){}})}

/* ---------- self test (design §5/§6.3; tolerance: rel 1e-3 or half of the last tabulated digit) ---------- */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const chk=(x,ref,half,m)=>ok(Math.abs(x-ref)<=Math.max(1e-3*Math.abs(ref),half),`${m}: ${x} vs ${ref}`);
  // 5.1 W1
  chk(mES(1.017),0.98328,5e-6,'mES');chk(mIPman(0.1),0.99010,5e-6,'mIPman');
  [[0.5,[0.7071,0.5,0.3536],0.99432,0.02432],[0.1,[0.3162,0.1,0.0316],0.99961,0.00732],[0.01,[0.1,0.01,0.001],1.0,0.00366]].forEach(([s,ts,ip,ae])=>{
    [0.5,1,1.5].forEach((a,i)=>chk(mTS(s,a),ts[i],5e-5,`mTS ${s} ${a}`));chk(mIPgauss(s,0.1,0.25),ip,5e-6,'mIPgauss '+s);chk(alphaEq(1.017,s),ae,5e-6,'alphaEq '+s)});
  [[0.1,0.8449],[0.5,0.9668],[1,0.9833],[1.5,0.9888]].forEach(([a,v])=>chk(sigmaCross(1.017,a),v,5e-5,'sigmaCross '+a));
  [[1,0.00643],[10,0.02513],[50,0.08940],[100,0.16705],[200,0.31827],[500,0.71145]].forEach(([t,v])=>chk(sigmaOfT(A1000,t),v,5e-6,'sigma_t '+t));
  {let tc=1;while(sigmaOfT(A1000,tc)<1/1.017)tc++;ok(tc===883,'first t = '+tc)}
  ok(isNaN(alphaEq(1.017,1))&&isNaN(mIPgauss(1,0.1,0.25)),'W1 guards NaN');chk(alphaEq(1.017,0.99),1.6773,5e-5,'alphaEq 0.99');
  // 5.2 circle radial ratio
  {const s=0.01,r0=circleScorePolar(1.1,Math.PI/2,s),r1=circleScorePolar(1.1,Math.PI/2,s*Math.sqrt(1.01));chk(r1.rad/r0.rad,0.99010,5e-6,'radial ratio');chk(r1.tan/r0.tan,1.0,5e-6,'tan ratio')}
  // 5.3 W2
  [[0.2,[0.0348,0.0391,0.4287,0.0351],[0.5950,0.2297],0.0346,[2209.1,2518.1]],[0.05,[0.0023,0.0067,0.5673,0.0023],[0.6140,0.0633],0.0022,[2579.2,2947.7]],[0.01,[0.0001,0.0046,0.6082,0.0001],[0.6152,0.0127],0.0001,[2603.7,2976.1]]].forEach(([s,tp,tu,tl,mm])=>{
    const ex=w2dens(s,0),es=w2dens(s,alphaEq(1.017,s)),ts=w2dens(s,1),ip=w2dens(s*Math.sqrt(1.01),0);
    [ex,es,ts,ip].forEach((d,i)=>chk(tv(d,PD),tp[i],5e-5,`W2 TVp ${s} ${i}`));chk(tv(es,UN),tu[0],5e-5,'W2 TVu es '+s);chk(tv(ts,UN),tu[1],5e-5,'W2 TVu ts '+s);
    chk(tv(es,vmPow(1/1.017,180)),tl,5e-5,'W2 TVlim '+s);chk(maxOverMin(es),mm[0],0.05,'W2 mm es '+s);chk(maxOverMin(ex),mm[1],0.05,'W2 mm ex '+s)});
  chk(tv(vmPow(1/1.017,180),PD),0.0045,5e-5,'lim TV');chk(tv(vmPow(1/0.965,180),PD),0.0094,5e-5,'lim .965');chk(tv(vmPow(1/1.06,180),PD),0.0155,5e-5,'lim 1.06');
  {const d=c2Marginal({a:1,b:1,sigma:0.2,alpha:alphaEq(0.95,0.2)}).dens;let Z=0,fin=true;for(const v of d){Z+=v;if(!isFinite(v))fin=false}ok(fin&&Math.abs(Z*2*Math.PI/d.length-1)<1e-9,'W2 lambda 0.95 guard');chk(tv(d,PD),0.0217,5e-5,'W2 0.95 TV')}
  // 5.4 W3
  const g=(k,T,e)=>gammaMatch(k,T,e,'x0');
  {const R=g('cosine',20,0.2);chk(R[0].gM,30.99,0.005,'W3 20 t1');[[5,0.334,2.145e-1,2.630e-1,1.294e-4],[10,0.682,6.035e-2,8.865e-2,7.161e-6]].forEach(([t,s,a,b,d])=>{const r=R[t-1];chk(r.sigma,s,5e-4,'W3 20 s'+t);chk(r.g1,a,0,'W3 20 g1 '+t);chk(r.gM,b,0,'W3 20 gM '+t);chk(r.delta,d,0,'W3 20 d '+t)});
    ok(R.filter(r=>r.gM>=0.1).pop().t===9,'W3 20 >=0.1');ok(w3band(R,0.1).map(x=>x.join('..')).join()==='6..14','W3 20 band')}
  {const R=g('cosine',100,0.2);chk(R[0].gM,29.37,0.005,'W3 100 t1');[[10,0.155,2.265e-1,3.674e-1,1.023e-4],[50,0.706,1.330e-2,3.687e-2,2.301e-7]].forEach(([t,s,a,b,d])=>{const r=R[t-1];chk(r.sigma,s,5e-4,'W3 100 s'+t);chk(r.g1,a,0,'W3 100 g1 '+t);chk(r.gM,b,0,'W3 100 gM '+t);chk(r.delta,d,0,'W3 100 d '+t)});
    ok(R.filter(r=>r.gM>=0.1).pop().t===24,'W3 100 >=0.1');ok(w3band(R,0.1).map(x=>x.join('..')).join()==='16..39','W3 100 band')}
  {const R=g('cosine',1000,0.2);chk(R[0].gM,17.64,0.005,'W3 1000 t1');[[100,0.167,2.209e-2,1.037e-1,8.039e-7],[500,0.711,1.228e-3,1.141e-2,2.144e-9]].forEach(([t,s,a,b,d])=>{const r=R[t-1];chk(r.sigma,s,5e-4,'W3 1000 s'+t);chk(r.g1,a,0,'W3 1000 g1 '+t);chk(r.gM,b,0,'W3 1000 gM '+t);chk(r.delta,d,0,'W3 1000 d '+t)});
    ok(R.filter(r=>r.gM>=0.1).pop().t===102,'W3 1000 >=0.1');ok(w3band(R,0.1).map(x=>x.join('..')).join()==='62..169','W3 1000 band')}
  {const R=g('cosine',100,0.1);chk(R[0].gM,14.68,0.005,'W3 e.1 t1');chk(R[9].gM,1.837e-1,0,'W3 e.1 t10')}
  {const R=g('cosine',20,0.1);chk(R[0].delta,8.711e-3,0,'406 d1');chk(R[1].delta,6.054e-4,0,'406 d2');chk(R[4].delta,8.292e-6,0,'406 d5')}
  // 5.5 toy
  [[20,[0.18467,0.64902,0.96751]],[100,[0.23261,0.24172,0.21686]],[1000,[0.24800,0.25663,0.23113]]].forEach(([T,v])=>[1,1.017,0.965].forEach((l,i)=>chk(esGaussVar('cosine',T,0.25,l),v[i],5e-6,`toy ${T} ${l}`)));
  // 5.6 W0
  chk(Math.sqrt(1-0.01)*0.2/0.01,19.90,0.005,'W0 0.1');chk(Math.sqrt(1-1e-4)*0.2/1e-4,2000,0.5,'W0 0.01');
  // W3 clamp
  {const t0=w3s.t,T0=w3s.Tp;w3s.t=60;w3setTp(20);ok(w3s.t<=19,'W3 clamp t='+w3s.t);w3setTp(T0);w3s.t=t0;$('#w3-t').value=t0;$('#w3-to').textContent=t0;renderW3()}
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  ok(!!document.querySelector('.katex'),'KaTeX rendered');
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ---------- boot ---------- */
initTheme();initToc();initW0();initW1();initW2();initW3();initReveal();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderW1();renderW2();renderW3()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
let tested=false;const runTest=()=>{if(tested||!/[?&]test\b/.test(location.search))return;tested=true;selfTest()};
if(document.readyState==='loading')addEventListener('DOMContentLoaded',()=>{renderMath()});else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath();runTest()});
