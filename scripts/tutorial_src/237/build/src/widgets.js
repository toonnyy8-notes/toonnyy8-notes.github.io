/* ---------- page helpers ---------- */
const PI=Math.PI;
function segPath(pts,X,Y,y0,y1){let d='',pen=false,up=false,dn=false;for(const [a,v] of pts){if(!isFinite(v)||v>y1||v<y0){pen=false;if(v>y1)up=true;else dn=true;continue}d+=(pen?'L':'M')+X(a).toFixed(1)+' '+Y(v).toFixed(1);pen=true}return{d,up,dn}}
const tGrid=nt=>Float64Array.from({length:nt},(_,j)=>-PI+(j+.5)*2*PI/nt);
const T_TICKS=[[-PI,'−π'],[-PI/2,'−π/2'],[0,'0'],[PI/2,'π/2'],[PI,'π']];
const fx=(v,n)=>(Math.abs(v)<.5*Math.pow(10,-n)?0:v).toFixed(n).replace('-','−');
const fe=v=>v.toExponential(2).replace('e+','e+').replace('-','−');
const sup=n=>String(n).replace(/./g,c=>({'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'}[c]||c));

/* ---------- S2 static figure ---------- */
(function(){const s=$('#s2-fig'),a=1,b=2,k=58,X=v=>180+v*k,Y=v=>150-v*k;
  const P=t=>[a*Math.cos(t),b*Math.sin(t)],N=t=>{const nx=b*Math.cos(t),ny=a*Math.sin(t),L=Math.hypot(nx,ny);return[nx/L,ny/L]};
  const curve=off=>{let d='';for(let i=0;i<=240;i++){const t=2*PI*i/240,p=P(t),n=N(t);d+=(i?'L':'M')+X(p[0]+off*n[0]).toFixed(1)+' '+Y(p[1]+off*n[1]).toFixed(1)}return d+'Z'};
  el('path',{d:curve(.3)+curve(-.3),style:'fill:var(--c-par);opacity:.12;fill-rule:evenodd'},s);
  el('path',{d:curve(.3),class:'ref'},s);el('path',{d:curve(-.3),class:'ref'},s);
  el('path',{d:curve(0),class:'manifold',style:'fill:none'},s);
  const t0=.6,p=P(t0),n=N(t0),x=[p[0]+.25*n[0],p[1]+.25*n[1]];
  el('line',{x1:X(p[0]),y1:Y(p[1]),x2:X(x[0]),y2:Y(x[1]),class:'acc',style:'stroke-width:2.5'},s);
  const tg=[-n[1],n[0]],q=9/k,c1=[p[0]+q*n[0],p[1]+q*n[1]];
  el('path',{d:`M${X(p[0]+q*tg[0])} ${Y(p[1]+q*tg[1])}L${X(c1[0]+q*tg[0])} ${Y(c1[1]+q*tg[1])}L${X(c1[0])} ${Y(c1[1])}`,class:'proj',style:'stroke-dasharray:none'},s);
  el('circle',{cx:X(p[0]),cy:Y(p[1]),r:4,class:'ink-f'},s);el('circle',{cx:X(x[0]),cy:Y(x[1]),r:4.5,class:'acc-f'},s);
  tx(s,X(x[0])+8,Y(x[1])-4,'x',{class:'labm'});
  tx(s,X(p[0])-8,Y(p[1])+16,'P_M(x)',{class:'labm','text-anchor':'end'});
  tx(s,X(x[0])+8,Y(x[1])+14,'d_M = ½‖x − P_M(x)‖²',{class:'al'});
  tx(s,X(-1)-6,Y(0),'M',{class:'labm','text-anchor':'end'});
  tx(s,X(0),Y(2.3)-6,'T_M(ε)，ε = 0.3',{class:'al','text-anchor':'middle'});})();

/* ---------- hero error-order axis ---------- */
function eaxis(node){node.innerHTML='<div class="kx-track"><div class="kx-line"></div></div><p class="cap ecap">σ = 0.01 時的分數誤差量級（示意）；o(f) 表示比 f 小的量，見 02 節。TS 均勻：誤差 o(σ^β)、β &gt; −2，且需 α &gt; −β。</p>';const tr=node.firstChild;
  const pos=L=>(L+1)/6*100;
  for(let L=-1;L<=5;L++){const s=document.createElement('span');s.className='kx-tick'+(L===-1?' l':L===5?' r':'');s.style.left=pos(L)+'%';s.textContent=L===0?'1':L===1?'10':'10'+sup(L);tr.appendChild(s)}
  [[0,'<span class="v">o(1)</span>：還原 p_data'],[4,'<span class="v">o(σ⁻²) = o(10⁴)</span>：集中到流形']].forEach(([L,lab])=>{const d=document.createElement('div');d.className='kx-mark acc';d.style.left=pos(L)+'%';d.innerHTML=`<span class="kx-dot"></span><span class="kx-stem"></span><span class="kx-lab ${L>3?'r':'c'}">${lab}</span>`;tr.appendChild(d)});
  layoutKAxis(node)}
function layoutKAxis(node){const tr=node.firstChild;if(!tr)return;const labs=$$('.kx-lab',tr);labs.forEach(l=>{l.style.bottom='14px';l.previousSibling.style.height='8px'});
  const items=labs.map(l=>({l,r:l.getBoundingClientRect()})).sort((a,b)=>a.r.left-b.r.left);const rows=[];let mx=0;
  items.forEach(it=>{let r=0;while(rows[r]!==undefined&&it.r.left<rows[r]+8)r++;rows[r]=it.r.right;mx=Math.max(mx,r);it.l.style.bottom=(14+r*28)+'px';it.l.previousSibling.style.height=(8+r*28)+'px'});
  tr.style.marginTop=(mx*28+40)+'px'}

/* ---------- C1 ---------- */
const C1_SIG=[0.01,0.015,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.5];
const c1={i:6,phi:90,rho:1.1};
function c1Values(){const s=C1_SIG[c1.i],phi=c1.phi*PI/180,rho=c1.rho,T=eq6Terms(rho,phi,s),S=circleScorePolar(rho,phi,s),tlim=-KAPPA*Math.sin(phi-THETA0)/rho;
  return{s,phi,rho,T,S,tlim,lead:-(rho-1)/s/s}}
function renderC1(){const V=c1Values(),{s,phi,rho}=V;
  let box=$('#c1-f1'),svg=$('svg',box),W=widthOf(box,260,480),H=Math.round(W*.72);
  const F=frame(svg,{W,H,m:{l:50,r:12,t:24,b:38},x:[.6,1.4],y:[-.1,.05],xt:[.6,.8,1,1.2,1.4],xf:v=>v.toFixed(1),yt:[-.1,-.05,0,.05],yf:v=>fx(v,2),xl:'半徑 ρ（沿方向 φ 的射線）'});
  const pts=[],lim=[];for(let i=0;i<=120;i++){const r=.6+.8*i/120;pts.push([r,s*s*circleLogP(r*Math.cos(phi),r*Math.sin(phi),s)]);lim.push([r,-.5*(r-1)**2])}
  const A=segPath(pts,F.X,F.Y,-.1,.05),B=segPath(lim,F.X,F.Y,-.1,.05);
  el('path',{d:B.d,class:'ref'},F.g);el('path',{d:A.d,class:'par'},F.g);
  el('line',{x1:F.X(rho),x2:F.X(rho),y1:F.y0,y2:F.y1,class:'cur'},F.g);
  if(A.up||A.dn)tx(F.g,F.x1,F.y1-8,'σ²·log p_σ 部分'+(A.up?'超出範圍 ↑':'')+(A.up&&A.dn?'、':'')+(A.dn?'超出範圍 ↓':''),{class:'tk','text-anchor':'end'});
  svg.setAttribute('aria-label',`沿方向 ${c1.phi}° 的 σ² log p_σ，σ = ${s}`);
  box=$('#c1-f2');svg=$('svg',box);const W2=widthOf(box,260,480),H2=Math.round(W2*.72);
  const F2=frame(svg,{W:W2,H:H2,m:{l:50,r:12,t:24,b:38},x:[.01,.5],xlog:true,y:[-4,3],xt:[.01,.02,.05,.1,.2,.5],xf:v=>String(v),yt:[[-4,'10⁻⁴'],[-3,'10⁻³'],[-2,'10⁻²'],[-1,'10⁻¹'],[0,'1'],[1,'10'],[2,'10²'],[3,'10³']],xl:'σ（對數軸）'});
  const L10=v=>Math.log10(Math.max(Math.abs(v),1e-4)),ser={A:[],B:[],C:[],D:[],resid:[]};
  for(let j=0;j<60;j++){const sj=.01*Math.pow(50,j/59),T=eq6Terms(rho,phi,sj);for(const k in ser)ser[k].push([sj,L10(T[k])])}
  [['A','tot'],['B','perp dash'],['C','ink dot'],['D','par pdd'],['resid','acc thin']].forEach(([k,c])=>el('path',{d:lp(ser[k].map(([a,v])=>[F2.X(a),F2.Y(v)])),class:c},F2.g));
  el('line',{x1:F2.X(s),x2:F2.X(s),y1:F2.y0,y2:F2.y1,class:'cur'},F2.g);
  svg.setAttribute('aria-label',`Eq. 6 各項絕對值對 σ 的雙對數圖；ρ = ${rho.toFixed(2)}，φ = ${c1.phi}°`);
  const out={exact:fx(V.T.exact,4),A:fx(V.T.A,4),B:fx(V.T.B,4),C:fx(V.T.C,4),D:fx(V.T.D,4),resid:fe(V.T.resid),rad:fx(V.S.rad,3),lead:fx(V.lead,3),diff:fx(V.S.rad-V.lead,4),tan:fx(V.S.tan,4),tlim:fx(V.tlim,4),ratio:Math.abs(V.S.tan)<1e-9?'—（切向為 0）':(Math.abs(V.S.rad/V.S.tan)).toFixed(1)};
  $$('#c1-tab [data-v]').forEach(n=>{n.textContent=out[n.dataset.v]});
  return V}
function initC1(){const so=$('#c1-so'),po=$('#c1-po'),ro=$('#c1-ro'),sI=$('#c1-s'),pI=$('#c1-p'),rI=$('#c1-r');
  const sync=()=>{c1.i=+sI.value;c1.phi=+pI.value;c1.rho=+rI.value;const s=C1_SIG[c1.i];so.textContent=s.toFixed(3);po.textContent=String(c1.phi).replace('-','−')+'°';ro.textContent=c1.rho.toFixed(2);
    sI.setAttribute('aria-valuetext','σ = '+s.toFixed(3));pI.setAttribute('aria-valuetext','φ = '+c1.phi+' 度');rI.setAttribute('aria-valuetext','ρ = '+c1.rho.toFixed(2))};
  const say=()=>{const V=c1Values();announce('c1',`σ = ${V.s}，法向分量 ${fx(V.S.rad,3)}，切向分量 ${fx(V.S.tan,4)}，殘差 ${fe(V.T.resid)}`)};
  [sI,pI,rI].forEach(i=>{i.addEventListener('input',()=>{sync();renderC1()});i.addEventListener('change',say)});sync();renderC1()}

/* ---------- C2 ---------- */
const C2_SIG=[0.01,0.015,0.02,0.03,0.05,0.07,0.1,0.15,0.2];
const gridCache=new Map();
const C2I={};
function c2Markup(id,withV,alpha){return `<div class="pg side"><div class="fig"><svg class="c2m" role="img" aria-label="平穩分佈的角度密度"></svg><div class="legend c2lg"></div></div>
<div class="ctl">
 <div class="seg" data-k="shape" role="group" aria-label="流形形狀"><button type="button" data-v="circle" aria-pressed="true">圓 (1, 1)</button><button type="button" data-v="ellipse" aria-pressed="false">橢圓 (1, 2)</button></div>
 <div class="rng"><label class="lbl" for="${id}-s">噪音 σ<output></output></label><input id="${id}-s" data-k="i" type="range" min="0" max="8" step="1" value="4"></div>
 <div class="rng"><label class="lbl" for="${id}-a">縮放指數 α<output></output></label><input id="${id}-a" data-k="alpha" type="range" min="0" max="1.9" step="0.05" value="${alpha}"></div>
 <div class="ctl-row"><button class="chip" type="button" data-a="0">L（α = 0）</button><button class="chip" type="button" data-a="1">TS-1（α = 1）</button><button class="chip" type="button" data-a="1.5">TS-1.5</button></div>
 <div class="seg" data-k="err" role="group" aria-label="誤差場"><button type="button" data-v="none" aria-pressed="true">無誤差</button><button type="button" data-v="c3" aria-pressed="false">Θ(1) 誤差（C.3）</button><button type="button" data-v="tan" aria-pressed="false">切向誤差 σ^β∇cos t</button></div>
 <div class="rng c2beta hide"><label class="lbl" for="${id}-b">誤差階數 β<output></output></label><input id="${id}-b" data-k="beta" type="range" min="-1.9" max="1" step="0.1" value="-1"></div>
 ${withV?`<div class="rng"><label class="lbl" for="${id}-l">似然強度 λ<output></output></label><input id="${id}-l" data-k="lambda" type="range" min="0" max="3" step="0.1" value="1.5"></div>`:''}
 <svg class="c2i" role="img" aria-label="流形與法向寬度 w"></svg>
 <p class="readout"></p><p class="c2st" role="status"></p>
</div></div><p class="cap c2note"></p>`}
function initC2(id,withV){const R=$('#'+id),I={id,R,withV,shape:'circle',i:4,alpha:withV?1:0,err:'none',beta:-1,lambda:withV?1.5:0,token:0,nt:180,G:null,res:null,say:false};C2I[id]=I;
  $('.c2-body',R).innerHTML=c2Markup(id,withV,I.alpha);
  const outs=()=>{const s=C2_SIG[I.i];$$('input[type=range]',R).forEach(inp=>{const k=inp.dataset.k,o=inp.parentNode.querySelector('output');let t;
      if(k==='i')t=s.toFixed(3);else if(k==='alpha')t=I.alpha.toFixed(2);else if(k==='beta')t=fx(I.beta,1);else t=I.lambda.toFixed(1);o.textContent=t;inp.setAttribute('aria-valuetext',inp.labels[0].firstChild.textContent.trim()+' = '+t)});
    $('.c2beta',R).classList.toggle('hide',I.err!=='tan')};
  $$('input[type=range]',R).forEach(inp=>{inp.addEventListener('input',()=>{const k=inp.dataset.k;I[k]=+inp.value;outs();c2Request(I,I.shape==='ellipse'&&(k==='i'||k==='alpha')?90:I.nt)});
    inp.addEventListener('change',()=>{I.say=true;c2Request(I,180)})});
  $$('.seg',R).forEach(g=>$$('button',g).forEach(b=>b.addEventListener('click',()=>{$$('button',g).forEach(x=>x.setAttribute('aria-pressed',String(x===b)));I[g.dataset.k]=b.dataset.v;outs();I.say=true;c2Request(I,180)})));
  $$('.chip',R).forEach(b=>b.addEventListener('click',()=>{I.alpha=+b.dataset.a;$('[data-k=alpha]',R).value=I.alpha;outs();I.say=true;c2Request(I,180)}));
  outs();c2Request(I,180)}
function c2Shape(I){return I.shape==='circle'?[1,1]:[1,2]}
function c2Request(I,nt){const s=C2_SIG[I.i],[a,b]=c2Shape(I);I.nt=nt;
  const key=`${I.shape}|${s}|${I.alpha}|${nt}`,tok=++I.token,st=$('.c2st',I.R);let G=gridCache.get(key);
  if(G&&G.rowsDone===G.nt){st.textContent='';c2Finish(I,G);return}
  if(!G){if(gridCache.size>=40)gridCache.clear();G=c2Grid({a,b,sigma:s,alpha:I.alpha,nt});gridCache.set(key,G)}
  if(G.circle){c2FillRows(G,G.nt);st.textContent='';c2Finish(I,G);return}
  st.textContent='計算中…';
  requestAnimationFrame(()=>{const step=()=>{if(tok!==I.token)return;const t0=performance.now();while(G.rowsDone<G.nt&&performance.now()-t0<8)c2FillRows(G,Math.min(G.nt,G.rowsDone+3));
    if(G.rowsDone<G.nt)setTimeout(step,0);else{st.textContent='';c2Finish(I,G)}};step()})}
function c2Finish(I,G){I.G=G;I.res=c2Combine(G,{err:I.err,beta:I.beta,lambda:I.withV?I.lambda:0});renderC2(I);
  if(I.say){I.say=false;const r=I.res;announce(I.id,`σ = ${G.s}，α = ${G.alpha}：與 p_data 的總變差 ${r.tvP.toFixed(4)}，與內蘊均勻的總變差 ${r.tvU.toFixed(4)}`)}}
function renderC2(I){const G=I.G,r=I.res,[a,b]=c2Shape(I),nt=G.nt,T=tGrid(nt),lam=I.withV?I.lambda:0;
  const ref={pdata:refCurve('pdata',a,b,0,nt),unif:refCurve('unif',a,b,0,nt)};
  if(I.err==='c3'&&!I.withV)ref.c3lim=refCurve('c3lim',a,b,0,nt);
  if(I.withV){ref.post=refCurve('post',a,b,lam,nt);ref.postdata=refCurve('postdata',a,b,lam,nt)}
  r.tvP=tv(r.dens,ref.pdata);r.tvU=tv(r.dens,ref.unif);
  const box=$('.fig',I.R),svg=$('.c2m',I.R),W=widthOf(box,260,620),H=Math.round(Math.min(W*.62,360));
  let peak=0;for(const v of r.dens)peak=Math.max(peak,v);const ymax=Math.min(Math.max(.85,1.05*peak),4);
  const yt=[];const stp=ymax>2?1:ymax>1.2?.5:.2;for(let v=0;v<=ymax+1e-9;v+=stp)yt.push(+v.toFixed(2));
  const F=frame(svg,{W,H,m:{l:44,r:12,t:24,b:38},x:[-PI,PI],y:[0,ymax],xt:T_TICKS,yt,yf:v=>v.toFixed(1),xl:'角度參數 t',yl:'密度（對 t）'});
  const path=(arr,cls)=>{const sp=segPath(Array.from(arr,(v,j)=>[T[j],v]),F.X,F.Y,0,ymax);el('path',{d:sp.d,class:cls},F.g);return sp};
  const lg=[];
  path(ref.pdata,'perp dash');lg.push('<span class="lg perp">p_data（von Mises）</span>');
  path(ref.unif,'par dot2');lg.push('<span class="lg pardot">內蘊均勻 ∝ √det g</span>');
  if(ref.c3lim){path(ref.c3lim,'ink pdd');lg.push('<span class="lg pdd2">Thm 4.1 框架的極限（Θ(1) 誤差）</span>')}
  if(ref.post){path(ref.post,'ink pdd');lg.push('<span class="lg pdd2">均勻先驗後驗 ∝ e^{−v}√det g</span>');path(ref.postdata,'ref');lg.push('<span class="lg ref">資料先驗後驗 ∝ e^{−v}p_data</span>')}
  const sp=path(r.dens,'acc');lg.unshift(`<span class="lg acc">${G.alpha===0?'Langevin':'TS'} 平穩分佈（計算結果）</span>`);
  if(sp.up){let jm=0;r.dens.forEach((v,j)=>{if(v>r.dens[jm])jm=j});tx(F.g,Math.min(Math.max(F.X(T[jm]),F.x0+60),F.x1-60),F.y1-8,'峰值 = '+peak.toPrecision(3),{class:'tk','text-anchor':'middle'})}
  $('.c2lg',I.R).innerHTML=lg.join('');
  svg.setAttribute('aria-label',`平穩分佈的角度密度，${I.shape==='circle'?'圓':'橢圓'}、σ = ${G.s}、α = ${G.alpha}`);
  // inset
  const si=$('.c2i',I.R),S=170,k=(S/2-14)/(b+.5);si.setAttribute('viewBox',`0 0 ${S} ${S}`);si.setAttribute('width',S);si.setAttribute('height',S);si.textContent='';
  const wc=Math.min(G.w,.45),P=t=>[a*Math.cos(t),b*Math.sin(t)],Nn=t=>{const nx=b*Math.cos(t),ny=a*Math.sin(t),L=Math.hypot(nx,ny);return[nx/L,ny/L]};
  const cv=off=>{let d='';for(let i=0;i<=160;i++){const t=2*PI*i/160,p=P(t),n=Nn(t);d+=(i?'L':'M')+(S/2+k*(p[0]+off*n[0])).toFixed(1)+' '+(S/2-k*(p[1]+off*n[1])).toFixed(1)}return d+'Z'};
  el('path',{d:cv(wc)+cv(-wc),style:'fill:var(--accent);opacity:.18;fill-rule:evenodd'},si);el('path',{d:cv(0),class:'manifold',style:'fill:none;stroke-width:2'},si);
  tx(si,S/2,S-6,'法向寬度 w = '+G.w.toFixed(3)+(G.w>.45?'（帶寬截在 0.45）':''),{class:'tk','text-anchor':'middle'});
  // readout
  const lmm=r.log10MaxOverMin,mm=lmm>6?'≈ 10^'+Math.round(lmm):Math.pow(10,lmm).toPrecision(4);
  let inRange;if(G.alpha===0)inRange='α = 0：標準 Langevin，適用 Thm 4.1';else if(I.err==='tan')inRange=(G.alpha>Math.max(-I.beta,0)+1e-9)?'(β, α) 在 Thm 5.1 的範圍內':'定理不涵蓋（需 α > max(−β, 0)）';else inRange='α > 0：在 Thm 5.1 的範圍內';
  let h=`TV(結果, p_data) <b>${r.tvP.toFixed(4)}</b><br>TV(結果, 內蘊均勻) <b>${r.tvU.toFixed(4)}</b><br>最大／最小 <b>${mm}</b><br>w = σ^(1−α/2) <b>${G.w.toFixed(3)}</b>　θ = σ^(2−α) <b>${r.theta.toPrecision(3)}</b>`;
  if(I.withV){r.tvPost=tv(r.dens,ref.post);r.tvPD=tv(r.dens,ref.postdata);h+=`<br>TV(結果, 均勻先驗後驗) <b>${r.tvPost.toFixed(4)}</b><br>TV(結果, 資料先驗後驗) <b>${r.tvPD.toFixed(4)}</b>`}
  h+=`<br><span class="acc">${inRange}</span>`;$('.readout',I.R).innerHTML=h;
  const notes=[];if(G.w>=.3)notes.push('此時法向寬度大，圖中的 t 邊際是廣義極座標角度的邊際，還不是 Thm 5.1 所說的流形上的分佈。');
  if(I.err==='tan')notes.push('切向誤差恰好是 O(σ^β)，不是定理要求的 o(σ^β)，刻意落在邊界階數上；cos t(x) 在原點不可微，對應的 K 取不含原點的環形區域。');
  $('.c2note',I.R).textContent=notes.join(' ')}

/* ---------- C3 ---------- */
const C3I={};
function c3Verdict(beta,alpha,withAlpha){const L=[],e=1e-9;
  if(alpha<e){L.push([beta>=-2-e,'集中到 M（Thm 4.1(1)）']);L.push([beta>=-e,'還原 p_data（Thm 4.1(3)）']);
    if(beta>=-2-e&&beta<-e)L.push([null,'不保證還原；論文構造了收斂到 M 上任意分佈的例子（Thm 4.1(2)）'])}
  else{const ts=beta>-2+e&&alpha>Math.max(-beta,0)+e&&alpha<2-e;L.push([ts,'收斂到內蘊均勻（Thm 5.1，梯度場）']);
    if(Math.abs(alpha+beta)<e)L.push([null,'α = −β 是邊界；定理要求嚴格的 α > −β']);if(beta<=-2+e)L.push([null,'β ≤ −2：Eq. 9 要求 β > −2'])}
  return L}
function initC3(id,withAlpha){const R=$('#'+id),I={id,R,withAlpha,beta:-1,alpha:withAlpha?1:0};C3I[id]=I;
  const bI=$('#'+id+'-b'),aI=$('#'+id+'-a');
  const sync=()=>{I.beta=+bI.value;if(aI)I.alpha=+aI.value;bI.parentNode.querySelector('output').textContent=fx(I.beta,2);bI.setAttribute('aria-valuetext','β = '+fx(I.beta,2));
    if(aI){aI.parentNode.querySelector('output').textContent=I.alpha.toFixed(2);aI.setAttribute('aria-valuetext','α = '+I.alpha.toFixed(2))}};
  const say=()=>announce(id,c3Verdict(I.beta,I.alpha,withAlpha).map(([ok,t])=>ok===null?t:t+(ok?'：定理保證':'：定理不保證')).join('；'));
  [bI,aI].filter(Boolean).forEach(i=>{i.addEventListener('input',()=>{sync();renderC3(I)});i.addEventListener('change',say)});
  $$('.chip',R).forEach(c=>c.addEventListener('click',()=>{const [b,a]=c.dataset.p.split(',').map(Number);bI.value=b;if(aI)aI.value=a;sync();renderC3(I);say()}));
  $('svg',R).addEventListener('pointerdown',ev=>{const s=$('svg',R),bb=s.getBoundingClientRect(),m=I.m;if(!m)return;
    const px=(ev.clientX-bb.left)*(m.W/bb.width),py=(ev.clientY-bb.top)*(m.H/bb.height);
    const b=Math.round((-3+(px-m.x0)/(m.x1-m.x0)*4)/.05)*.05;bI.value=Math.max(-3,Math.min(1,b));
    if(aI){const a=Math.round(((m.y0-py)/(m.y0-m.y1)*2)/.05)*.05;aI.value=Math.max(0,Math.min(2,a))}sync();renderC3(I);say()});
  sync();renderC3(I)}
function renderC3(I){const R=I.R,box=$('.fig',R),svg=$('svg',R),W=widthOf(box,260,560),H=I.withAlpha?Math.round(Math.min(W*.78,420)):132;
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const x0=44,x1=W-12,y0=H-36,y1=I.withAlpha?46:56,X=lin(-3,1,x0,x1),g=el('g',null,svg);I.m={W,H,x0,x1,y0,y1};
  const defs=el('defs',null,svg),pat=el('pattern',{id:I.id+'-h',width:8,height:8,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},defs);el('line',{x1:0,y1:0,x2:0,y2:8,style:'stroke:var(--ink-3);stroke-width:1.5;opacity:.5'},pat);
  el('rect',{x:X(-3),y:y1,width:X(-2)-X(-3),height:y0-y1,fill:`url(#${I.id}-h)`},g);
  if(I.withAlpha){const Y=lin(0,2,y0,y1);
    el('path',{d:`M${X(-2)} ${Y(2)}L${X(1)} ${Y(2)}L${X(1)} ${Y(0)}L${X(0)} ${Y(0)}Z`,style:'fill:var(--accent-soft);stroke:none'},g);
    el('path',{d:`M${X(-2)} ${Y(2)}L${X(0)} ${Y(0)}L${X(1)} ${Y(0)}`,class:'ref'},g);el('line',{x1:X(-2),x2:X(1),y1:Y(2),y2:Y(2),class:'ref'},g);
    [0,.5,1,1.5,2].forEach(v=>{el('line',{x1:x0,x2:x1,y1:Y(v),y2:Y(v),class:'grid',style:'opacity:.5'},g);tx(g,x0-8,Y(v)+4,v.toFixed(1),{class:'tk','text-anchor':'end'})});
    tx(g,4,14,'縮放指數 α',{class:'al'});
    tx(g,X(.5),Y(1.15),'Thm 5.1',{class:'lab','text-anchor':'middle'});tx(g,X(.5),Y(1.15)+16,'保證均勻',{class:'al','text-anchor':'middle'});
    tx(g,X(-1.15),Y(.35),'α = −β',{class:'tk','text-anchor':'middle'});
    el('circle',{cx:X(I.beta),cy:Y(I.alpha),r:7,class:'acc-f'},g);el('circle',{cx:X(I.beta),cy:Y(I.alpha),r:10,style:'fill:none;stroke:var(--accent);stroke-width:1.5'},g)}
  else{const ym=(y0+y1)/2;el('rect',{x:X(-2),y:y1,width:X(0)-X(-2),height:y0-y1,style:'fill:var(--c-par);opacity:.12'},g);el('rect',{x:X(0),y:y1,width:X(1)-X(0),height:y0-y1,style:'fill:var(--accent-soft)'},g);
    tx(g,(X(-3)+X(-2))/2,ym+4,'不保證集中',{class:'al','text-anchor':'middle'});tx(g,(X(-2)+X(0))/2,ym+4,'保證集中',{class:'al','text-anchor':'middle'});tx(g,(X(0)+X(1))/2,ym-6,'保證集中',{class:'al','text-anchor':'middle'});tx(g,(X(0)+X(1))/2,ym+12,'與還原',{class:'al','text-anchor':'middle'});
    el('path',{d:`M${X(I.beta)} ${y1-4}l-7 -12h14z`,class:'acc-f'},g);el('line',{x1:X(I.beta),x2:X(I.beta),y1:y1-4,y2:y0,style:'stroke:var(--accent);stroke-width:2'},g)}
  [[-2,'o(σ⁻²)：Thm 4.1(1)'],[0,'o(1)：Thm 4.1(3)']].forEach(([b,l])=>{el('line',{x1:X(b),x2:X(b),y1:y1,y2:y0,class:'ink',style:'stroke-width:1.5'},g);tx(g,X(b)+(b===0?6:-6),y1-(I.withAlpha?10:26),l,{class:'tk','text-anchor':b===0?'start':'end'})});
  el('line',{x1:x0,x2:x1,y1:y0,y2:y0,class:'axl'},g);[-3,-2,-1,0,1].forEach(v=>{el('line',{x1:X(v),x2:X(v),y1:y0,y2:y0+4,class:'axl'},g);tx(g,X(v),y0+18,fx(v,0),{class:'tk','text-anchor':'middle'})});
  tx(g,(x0+x1)/2,H-4,'誤差階數 β（E_σ = o(σ^β)）',{class:'al','text-anchor':'middle'});
  $('ul.verdict',R).innerHTML=c3Verdict(I.beta,I.alpha,I.withAlpha).map(([ok,t])=>ok===null?`<li class="nt">${t}</li>`:`<li class="${ok?'ok':'no'}"><span aria-hidden="true">${ok?'●':'○'}</span>${t}：<b>${ok?'定理保證':'定理不保證'}</b></li>`).join('');
  const s=.01;let h=`σ = 0.01 時誤差量級 σ^β <b>${Math.pow(s,I.beta).toPrecision(3)}</b>`;
  if(I.withAlpha)h+=`<br>θ = σ^(2−α) <b>${Math.pow(s,2-I.alpha).toPrecision(3)}</b><br>w = σ^(1−α/2) <b>${Math.pow(s,1-I.alpha/2).toPrecision(3)}</b><br>σ^(α+β) <b>${Math.pow(s,I.alpha+I.beta).toPrecision(3)}</b><br>e^(2κσ^α)（漸近式） <b>${Math.exp(2*KAPPA*Math.pow(s,I.alpha)).toPrecision(4)}</b>`;
  $('.readout',R).innerHTML=h;
  svg.setAttribute('aria-label',I.withAlpha?`β = ${fx(I.beta,2)}，α = ${I.alpha.toFixed(2)}`:`β = ${fx(I.beta,2)}`)}

/* ---------- C4 ---------- */
const c4={sigma:.1,aTS:1,err:'none',sim:null,running:false,spf:25,colors:null,exact:null,frames:0};
function readColors(){const cs=getComputedStyle(document.documentElement),g=k=>cs.getPropertyValue(k).trim();c4.colors={ink:g('--ink'),ink2:g('--ink-2'),ink3:g('--ink-3'),acc:g('--accent'),par:g('--c-par'),perp:g('--c-perp'),line:g('--line')}}
function c4Opts(){return{sigma:c4.sigma,alphas:[0,c4.aTS],err:c4.err,dt:.1}}
function c4Reset(){c4.running=false;$('#c4-run').textContent='執行';c4.sim=c4Init(1000,237);
  c4.exact=c4.err==='rot'?null:[0,c4.aTS].map(al=>c2Marginal({a:1,b:1,sigma:c4.sigma,alpha:al,err:c4.err}).dens);
  $('#c4-exl').classList.toggle('hide',!c4.exact);$('.c4al').textContent=c4.aTS;drawC4()}
function fitCanvas(cv,w,h){const dpr=window.devicePixelRatio||1;cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);cv.style.width=w+'px';cv.style.height=h+'px';const ctx=cv.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return ctx}
function drawC4(){if(!c4.colors)readColors();const C=c4.colors,pd=refCurve('pdata',1,1,0,180),TT=tGrid(180);
  $$('#c4 .c4panel').forEach(pn=>{const p=+pn.dataset.p,X=c4.sim.P[p],N=X.length/2,w=Math.max(150,Math.min(360,pn.clientWidth));
    const sc=fitCanvas($('.sc',pn),w,w),cx=w/2,k=w/4.2;sc.clearRect(0,0,w,w);
    sc.strokeStyle=C.line;sc.lineWidth=1;sc.beginPath();sc.moveTo(0,cx);sc.lineTo(w,cx);sc.moveTo(cx,0);sc.lineTo(cx,w);sc.stroke();
    sc.strokeStyle=C.ink3;sc.lineWidth=1.5;sc.beginPath();sc.arc(cx,cx,k,0,2*PI);sc.stroke();
    sc.fillStyle=p?C.acc:C.par;sc.globalAlpha=.55;for(let i=0;i<N;i++){const x=cx+k*X[2*i],y=cx-k*X[2*i+1];if(x>-2&&x<w+2&&y>-2&&y<w+2)sc.fillRect(x-1.25,y-1.25,2.5,2.5)}sc.globalAlpha=1;
    const st=c4Stats(X,36),hh=150,hc=fitCanvas($('.hi',pn),w,hh),ml=4,mr=4,mb=18,mt=6,bw=(w-ml-mr)/36;hc.clearRect(0,0,w,hh);
    let mx=.85;st.hist.forEach(c=>{mx=Math.max(mx,1.1*c/N/(2*PI/36))});const Yv=v=>hh-mb-(v/mx)*(hh-mb-mt),Xt=t=>ml+(t+PI)/(2*PI)*(w-ml-mr);
    hc.fillStyle=p?C.acc:C.par;hc.globalAlpha=.3;st.hist.forEach((c,j)=>{const d=c/N/(2*PI/36);hc.fillRect(ml+j*bw+.5,Yv(d),bw-1,Yv(0)-Yv(d))});hc.globalAlpha=1;
    const line=(arr,col,dash,lw)=>{hc.strokeStyle=col;hc.lineWidth=lw;hc.setLineDash(dash);hc.beginPath();arr.forEach((v,j)=>{const x=Xt(TT[j]),y=Yv(Math.min(v,mx));j?hc.lineTo(x,y):hc.moveTo(x,y)});hc.stroke();hc.setLineDash([])};
    line(pd,C.perp,[6,5],2);line(new Float64Array(180).fill(1/(2*PI)),C.ink3,[1.5,4],2);if(c4.exact)line(c4.exact[p],C.acc,[],2.5);
    hc.strokeStyle=C.ink3;hc.lineWidth=1;hc.beginPath();hc.moveTo(ml,Yv(0));hc.lineTo(w-mr,Yv(0));hc.stroke();
    hc.fillStyle=C.ink3;hc.font='12px "JetBrains Mono", monospace';hc.textAlign='left';hc.fillText('−π',ml,hh-3);hc.textAlign='center';hc.fillText('0',w/2,hh-3);hc.textAlign='right';hc.fillText('π',w-mr,hh-3);
    $('.readout',pn).innerHTML=`步數 <b>${c4.sim.step}</b>　重設 <b>${c4.sim.resets[p]}</b><br>mean cos <b>${fx(st.meanCos,3)}</b>　mean sin <b>${fx(st.meanSin,3)}</b><br>TV(直方圖, 均勻) <b>${st.tvUnif.toFixed(3)}</b>`;pn.lastStats=st})}
function c4Done(){c4.running=false;$('#c4-run').textContent='執行';drawC4();const s=$$('#c4 .c4panel').map(p=>p.lastStats);announce('c4',`第 ${c4.sim.step} 步，L 的 mean cos = ${fx(s[0].meanCos,3)}，TS 的 mean cos = ${fx(s[1].meanCos,3)}`)}
function runC4(){if(c4.sim.step>=10000)c4Reset();c4.running=true;$('#c4-run').textContent='暫停';const o=c4Opts();
  if(RM){const chunk=()=>{if(!c4.running)return;for(let r=0;r<5&&c4.sim.step<10000;r++)c4Steps(c4.sim,Math.min(100,10000-c4.sim.step),o);drawC4();if(c4.sim.step<10000)setTimeout(chunk,0);else c4Done()};chunk();return}
  const fr=()=>{if(!c4.running)return;const t0=performance.now(),n=Math.min(c4.spf,10000-c4.sim.step);c4Steps(c4.sim,n,o);const dt=performance.now()-t0;
    c4.spf=Math.max(5,Math.min(200,Math.round(c4.spf*12/Math.max(dt,1))));if(++c4.frames%3===0)drawC4();if(c4.sim.step>=10000)c4Done();else requestAnimationFrame(fr)};requestAnimationFrame(fr)}
function initC4(){$$('#c4-s button').forEach(b=>b.addEventListener('click',()=>{$$('#c4-s button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));c4.sigma=+b.dataset.v;c4Reset()}));
  $$('#c4-e button').forEach(b=>b.addEventListener('click',()=>{$$('#c4-e button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));c4.err=b.dataset.v;c4Reset()}));
  const aI=$('#c4-a');aI.addEventListener('input',()=>{c4.aTS=+aI.value;$('#c4-ao').textContent=c4.aTS.toFixed(2);aI.setAttribute('aria-valuetext','α = '+c4.aTS.toFixed(2));c4Reset()});
  $('#c4-run').addEventListener('click',()=>{if(c4.running){c4.running=false;$('#c4-run').textContent='執行';c4Done()}else runC4()});
  $('#c4-reset').addEventListener('click',c4Reset);
  const recolor=()=>{readColors();drawC4()};new MutationObserver(recolor).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});matchMedia('(prefers-color-scheme: dark)').addEventListener('change',recolor);
  c4Reset();
  if(!RM&&!/[?&]test\b/.test(location.search)&&'IntersectionObserver' in window){const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){io.disconnect();if(!c4.running&&c4.sim.step===0)runC4()}},{threshold:.5});io.observe($('#c4'))}}

/* ---------- timeline ---------- */
const TL_GROUP={'1907.05600':['sm','score-geometry-score-matching-sde'],'2006.11239':['sm','score-geometry-score-matching-sde'],'2011.13456':['sm','score-geometry-score-matching-sde'],'2206.00364':['sm','score-geometry-score-matching-sde'],
 '2208.05314':['cv','score-geometry-diffusion-guarantees'],'2209.11215':['cv','score-geometry-diffusion-guarantees'],
 '2206.01018':['md','score-geometry-manifold-detection'],'2212.12611':['md','score-geometry-manifold-detection'],'2301.07882':['md','score-geometry-manifold-detection'],
 '2103.04715':['ip','score-geometry-inverse-problems-guidance'],'2207.12598':['ip','score-geometry-inverse-problems-guidance'],'2507.15397':['ip','score-geometry-inverse-problems-guidance'],'2112.10752':['ip','score-geometry-inverse-problems-guidance'],
 '2506.15385':['un','score-geometry-tempered-langevin'],'2509.24912':['me','score-geometry-lineage-hub']};
const TL_NAME={sm:'分數學習基礎',cv:'收斂保證',md:'流形偵測',ip:'逆問題與引導',un:'均勻採樣',me:'本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  if(!data.some(e=>e.arxiv==='2509.24912'))data.push({year:2025,title:'When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis',authors_short:'本論文',arxiv:'2509.24912',role:'小噪音時分數的主導項只含幾何；TS Langevin 在寬鬆誤差下取樣流形上的均勻分佈。'});
  data=data.map((e,i)=>({e,i})).sort((a,b)=>(+a.e.year||0)-(+b.e.year||0)||a.i-b.i).map(o=>o.e);
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];used.add(g[0]);const slug=g[1];const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${slug?`<a href="${slug}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
  const on=new Set(used);
  Object.keys(TL_NAME).filter(k=>used.has(k)).forEach(k=>{const b=document.createElement('button');b.type='button';b.className='chip';b.setAttribute('aria-pressed','true');b.textContent=TL_NAME[k];b.dataset.g=k;
    b.addEventListener('click',()=>{if(on.has(k)&&on.size===used.size){on.clear();on.add(k)}else if(on.has(k)){on.delete(k);if(!on.size)used.forEach(x=>on.add(x))}else on.add(k);
      $$('.chip',chips).forEach(c=>c.setAttribute('aria-pressed',String(on.has(c.dataset.g))));$$('.ev',track).forEach(ev=>ev.classList.toggle('dim',!on.has(ev.dataset.g)))});chips.appendChild(b)});
  const rail=$('#rail'),by=dir=>rail.scrollBy({left:dir*Math.min(rail.clientWidth*.8,560),behavior:RM?'auto':'smooth'});
  $('#tl-prev').addEventListener('click',()=>by(-1));$('#tl-next').addEventListener('click',()=>by(1));
  rail.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){by(1);e.preventDefault()}if(e.key==='ArrowLeft'){by(-1);e.preventDefault()}})}

/* ---------- self test ---------- */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t)=>Math.abs(a-b)<=t+1e-12,rel=(a,b,t)=>Math.abs(a-b)<=t*Math.abs(b)+1e-12;
  [[.5,.061549719185481304,.24249961258080195],[10,7.9429720831186956,.94859982595484596],[51,48.117624166490079,.99014704755894778],[1e4,9994.4759037814323,.99994999874987498]].forEach(([x,l,r])=>{ok(near(logI0(x),l,1e-6),'logI0 '+x);ok(near(ratioOverX(x)*x,r,1e-6),'ratio '+x)});
  // C1 table 5.1 (rho=1.1, phi=90)
  [[.5,-3.1372,-.0200,-.2258,1.419,-1.702,-1.3021,2.4526,.7,-.78431],[.2,-3.4563,-.1250,.6905,.2887,-3.210,-.7096,3.5331,.9,-.13825],[.1,-3.3533,-.5,1.3836,.07351,-10.521,-.5210,3.6174,2.9,-.03353],
   [.05,-4.2153,-2,2.0768,.01845,-40.471,-.4713,3.6321,11.1,-.01054],[.02,-13.8145,-12.5,2.9931,.002954,-250.457,-.4572,3.6357,68.9,-.00553],[.01,-50.6235,-50,3.6862,.0007386,-1000.455,-.4552,3.6362,275.1,-.00506]]
   .forEach(([s,ex,A,C,res,rad,dif,tan,ratio,s2])=>{const T=eq6Terms(1.1,PI/2,s),S=circleScorePolar(1.1,PI/2,s);
     ok(near(T.exact,ex,1e-4)&&near(T.A,A,1e-4)&&near(T.C,C,1e-4)&&rel(T.resid,res,.02)&&near(S.rad,rad,1e-3)&&near(S.rad+(.1)/s/s,dif,1e-4)&&near(S.tan,tan,1e-4)&&near(Math.abs(S.rad/S.tan),ratio,.05)&&near(s*s*T.exact,s2,1e-5),'C1 σ='+s)});
  {const T=eq6Terms(1.1,PI/2,.01);ok(near(T.B,-4.2628,1e-4)&&near(T.D,-.0477,1e-4),'C1 B,D')}
  {const T=eq6Terms(1,PI,.01),S=circleScorePolar(1,PI,.01);ok(near(T.exact,3.4232,1e-4)&&rel(T.resid,-1.875e-4,.02)&&near(S.rad,-.5,1e-3)&&Math.abs(S.tan)<1e-9,'C1 ρ=1 φ=180')}
  {const T=eq6Terms(.9,PI/6,.01),S=circleScorePolar(.9,PI/6,.01);ok(near(T.exact,-53.9876,1e-4)&&rel(T.resid,4.287e-4,.02)&&near(S.rad,999.444,1e-3)&&near(S.tan,2.2230,1e-4),'C1 ρ=0.9 φ=30')}
  // C2 table 5.2
  const C2T=[['c',.2,0,'none',0,.0348,.5980,2518],['c',.2,1,'none',0,.4287,.2297,4.783],['c',.05,0,'none',0,.0023,.6170,2948],['c',.05,.5,'none',0,.3957,.2624,5.968],['c',.05,1,'none',0,.5673,.0633,1.491],['c',.05,1.5,'none',0,.6070,.0142,1.093],
   ['c',.01,0,'none',0,.0001,.6182,2976],['c',.01,1,'none',0,.6082,.0127,1.083],['c',.01,1.5,'none',0,.6173,.0013,1.008],['c',.05,0,'c3',0,.7436,.2763,7.536,.0223],['c',.01,0,'c3',0,.7619,.2883,7.386,.0009],['c',.01,1,'c3',0,.6202,.0033,1.022],
   ['c',.01,0,'tan',-1,.9997,.9057,-83.4],['c',.01,1,'tan',-1,.8233,.2791,6.819],['c',.05,1,'tan',-1,.7961,.2387,4.954],['c',.01,1.5,'tan',-1,.6424,.0305,1.212],['c',.01,1,'tan',.5,.6085,.0124,1.081],
   ['e',.2,1,'none',0,.3679,.2419,4.907],['e',.05,0,'none',0,.0011,.5677,2970],['e',.05,1,'none',0,.5122,.0698,2.360],['e',.05,1.5,'none',0,.5597,.0190,1.999],['e',.01,1,'none',0,.5572,.0142,2.067],['e',.01,1.5,'none',0,.5702,.0060,1.973],
   ['e',.01,0,'c3',0,.7304,.4837,10293,.0024],['e',.01,1,'c3',0,.5676,.0130,2.168]];
  C2T.forEach(([sh,s,al,err,be,tP,tU,mm,c3l])=>{const a=1,b=sh==='c'?1:2,r=c2Marginal({a,b,sigma:s,alpha:al,err,beta:be}),P=refCurve('pdata',a,b),U=refCurve('unif',a,b);
    let good=near(tv(r.dens,P),tP,5e-4)&&near(tv(r.dens,U),tU,5e-4)&&(mm<0?near(r.log10MaxOverMin,-mm,.1):rel(Math.pow(10,r.log10MaxOverMin),mm,.005));
    if(c3l!==undefined)good=good&&near(tv(r.dens,refCurve('c3lim',a,b)),c3l,5e-4);ok(good,`C2 ${sh} σ=${s} α=${al} ${err}${err==='tan'?be:''}`)});
  [[1,.01,0,'postdata',.0001],[1,.01,0,'post',.8354],[1,.01,1,'post',.0077],[1,.01,1,'postdata',.8305],[2,.01,1,'post',.0082],[2,.01,0,'postdata',.0001]].forEach(([b,s,al,k,v])=>{const r=c2Marginal({a:1,b,sigma:s,alpha:al,err:'none',lambda:1.5});ok(near(tv(r.dens,refCurve(k,1,b,1.5)),v,5e-4),`C2b b=${b} α=${al} ${k}`)});
  {const A=c2Marginal({a:1,b:1,sigma:.05,alpha:1,err:'none'}),B=c2Marginal({a:1,b:1,sigma:.05,alpha:1,err:'none',forceQuad:true});let m=0;A.dens.forEach((v,j)=>{m=Math.max(m,Math.abs(v-B.dens[j]))});ok(m<1e-6,'C2 closed vs quad')}
  {const r=c2Marginal({a:1,b:1,sigma:.01,alpha:0,err:'tan',beta:-1.9});let z=0;r.dens.forEach(v=>{z+=v});ok(r.dens.every(isFinite)&&near(z*2*PI/r.dens.length,1,1e-9),'C2 overflow guard')}
  {const W=makeEllipseLogP(1,2,.01),F=makeEllipseLogP(1,2,.01,true);ok([[.3,2.05],[1.2,.1],[0,0]].every(p=>Math.abs(W(...p)-F(...p))<1e-12),'ellipse window = full')}
  // C3 table 5.3
  [[-1,1,true,false],[-1,1.5,true,true],[-1,0,true,false],[.5,1,true,true],[-2.5,1,false,false],[-1.9,1.95,true,true],[0,0,true,true]].forEach(([b,a,conc,third])=>{const v=c3Verdict(b,a,true),g=v.filter(x=>x[0]!==null);
    const okv=a===0?(g[0][0]===conc&&g[1][0]===third):(g[0][0]===third);ok(okv&&(b>=-2)===conc,`C3 (${b}, ${a})`)});
  [[-1,1,.01,.1,1,1.083],[-1,1.5,.1,.316,.1,1.008],[-1,0,1e-4,.01,100,2981],[.5,1,.01,.1,.001,1.083],[-1.9,1.95,.794,.891,.794,1.001]].forEach(([b,a,th,w,f,e])=>{const s=.01;ok(rel(Math.pow(s,2-a),th,.005)&&rel(Math.pow(s,1-a/2),w,.005)&&rel(Math.pow(s,a+b),f,.005)&&rel(Math.exp(8*Math.pow(s,a)),e,.005),`C3 numbers (${b}, ${a})`)});
  // C4 table 5.4
  [[.1,'none',1000,-.819,-.007,.574,-.204,-.020,.156],[.05,'none',1000,-.437,0,.296,-.083,-.026,.083],[.1,'rot',10000,.063,-.802,.572,-.098,-.210,.154]].forEach(([s,err,n,lc,ls,lt,tc,ts,tt])=>{const st=c4Init(1000,237);c4Steps(st,n,{sigma:s,alphas:[0,1],err,dt:.1});const A=c4Stats(st.P[0],36),B=c4Stats(st.P[1],36);
    ok([[A.meanCos,lc],[A.meanSin,ls],[A.tvUnif,lt],[B.meanCos,tc],[B.meanSin,ts],[B.tvUnif,tt]].every(([x,y])=>near(x,y,.01))&&st.resets[0]+st.resets[1]===0,`C4 σ=${s} ${err} @${n}: L ${A.meanCos.toFixed(3)} TS ${B.meanCos.toFixed(3)}`)});
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ---------- boot ---------- */
initTheme();initToc();initC1();initC2('c2a',false);initC2('c2b',true);initC3('c3a',false);initC3('c3b',true);initC4();renderTimeline();
$$('.eaxis').forEach(eaxis);
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderC1();Object.values(C2I).forEach(I=>{if(I.G)renderC2(I)});Object.values(C3I).forEach(renderC3);drawC4();$$('.eaxis').forEach(layoutKAxis)})});
document.fonts&&document.fonts.ready.then(()=>$$('.eaxis').forEach(layoutKAxis));
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))selfTest();
