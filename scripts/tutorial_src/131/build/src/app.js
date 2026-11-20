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
const UNDEF='undef';
function fmt(v,n=4){if(v===UNDEF)return'未定義';if(v===Infinity)return'∞';if(v===-Infinity)return'−∞';if(!Number.isFinite(v))return'—';if(Math.abs(v)<5*10**-(n+1))v=0;return v.toFixed(n).replace('-','−')}
const norm=(v,n)=>Number((+v).toFixed(n));
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function segInit(box,opts,cur,label,onPick){box.textContent='';opts.forEach(o=>{const b=document.createElement('button');b.type='button';b.setAttribute('role','radio');
    b.textContent=label?label(o):String(o);b.dataset.v=String(o);box.appendChild(b)});
  const sync=v=>$$('button',box).forEach(b=>{const on=b.dataset.v===String(v);b.setAttribute('aria-checked',String(on));b.setAttribute('aria-pressed',String(on));b.tabIndex=on?0:-1});sync(cur);
  box.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;const v=opts.find(o=>String(o)===b.dataset.v);sync(v);onPick(v)});
  box.addEventListener('keydown',e=>{const bs=$$('button',box),i=bs.findIndex(b=>b.tabIndex===0);let j=i;if(e.key==='ArrowRight'||e.key==='ArrowDown')j=(i+1)%bs.length;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')j=(i-1+bs.length)%bs.length;else return;e.preventDefault();bs[j].focus();bs[j].click()});
  return sync}
function dl(box,rows){box.innerHTML=rows.map(([k,v,c])=>`<dt>${k}</dt><dd${c?' class="acc"':''}>${v}</dd>`).join('')}
function slider(id,lab,min,max,step,val,ndec){return `<div class="rng"><label class="lbl" for="${id}">${lab} <output id="${id}o"></output></label><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></div>`}

/* ================= shared math (mirrors verify/check.py) ================= */
function lse(a){const m=Math.max(...a);if(!Number.isFinite(m))return m;let s=0;for(const v of a)s+=Math.exp(v-m);return m+Math.log(s)}
function softmax(z){const l=lse(z);return z.map(v=>Math.exp(v-l))}
const logit=p=>Math.log(p/(1-p)),sig=t=>1/(1+Math.exp(-t));
function ratio(num,den){if(Math.abs(den)<1e-12)return Math.abs(num)<1e-12?UNDEF:Infinity;return num/den}

/* ---------------- W1: single-state PMD (check.py w1) ---------------- */
const MU_PRE={skew:[.4,.3,.2,.1],unif:[.25,.25,.25,.25],conc:[.85,.05,.05,.05]};
const MU_NAME={skew:'偏斜',unif:'均勻',conc:'集中'};
function pmd(mu,Q,eta){const V=mu.reduce((a,m,i)=>a+m*Q[i],0),A=Q.map(q=>q-V);const l=mu.map((m,i)=>Math.log(m)+eta*A[i]);const logZ=lse(l);
  const lpi=l.map(v=>v-logZ),pi=lpi.map(Math.exp);let klMP=0,klPM=0,EA=0;
  for(let i=0;i<mu.length;i++){klMP+=mu[i]*(Math.log(mu[i])-lpi[i]);klPM+=pi[i]*(lpi[i]-Math.log(mu[i]));EA+=pi[i]*A[i]}
  return{V,A,pi,logZ,klMP,klPM,EA,obj:eta>0?EA-klPM/eta:NaN,logZQ:logZ+eta*V}}
const w1={mu:'skew',Q:[.5,0,-.5,1],eta:2};
function w1Draw(){const box=$('#w1-fig'),W=widthOf(box,300,560),H=250,svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const mu=MU_PRE[w1.mu],o=pmd(mu,w1.Q,w1.eta);const L=34,R=W-8,T=18,B=H-56,y=lin(0,1,B,T);
  const defs=el('defs',null,svg),pt=el('pattern',{id:'w1h',width:6,height:6,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},defs);el('line',{x1:0,y1:0,x2:0,y2:6,class:'hatch-ln'},pt);
  [0,.5,1].forEach(v=>{el('line',{x1:L,x2:R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-6,y(v)+4,v.toFixed(1),{class:'tk','text-anchor':'end'})});
  const gw=(R-L)/4,bw=Math.min(34,gw*.34);
  for(let i=0;i<4;i++){const cx=L+gw*(i+.5);
    el('rect',{x:cx-bw-2,y:y(mu[i]),width:bw,height:B-y(mu[i]),fill:'url(#w1h)',stroke:'var(--c-1)','stroke-width':1},svg);
    el('rect',{x:cx+2,y:y(o.pi[i]),width:bw,height:B-y(o.pi[i]),class:'c2-f'},svg);
    tx(svg,cx-bw/2-2,y(mu[i])-4,mu[i].toFixed(2),{class:'tk','text-anchor':'middle'});tx(svg,cx+2+bw/2,y(o.pi[i])-4,o.pi[i].toFixed(3),{class:'tk','text-anchor':'middle'});
    tx(svg,cx,B+18,'t'+(i+1),{class:'lab','text-anchor':'middle'});tx(svg,cx,B+36,'A '+(o.A[i]>=0?'+':'')+fmt(o.A[i],2),{class:'tk','text-anchor':'middle'})}
  el('line',{x1:L,x2:R,y1:B,y2:B,class:'axis0'},svg);
  svg.setAttribute('aria-label',`μ = (${mu.join(', ')})，π⁺ = (${o.pi.map(v=>v.toFixed(3)).join(', ')})`);
  dl($('#w1-read'),[['V',fmt(o.V)],['log Z',fmt(o.logZ),1],['KL(μ‖π⁺)',fmt(o.klMP),1],['兩者差',fmt(o.logZ-o.klMP)],['KL(π⁺‖μ)',fmt(o.klPM)],['最佳值',w1.eta>0?fmt(o.obj):'—'],['(1/η) log Z',w1.eta>0?fmt(o.logZ/w1.eta):'—'],['log Z（若用 Q）',fmt(o.logZQ)]]);
  $('#w1-etao').textContent=w1.eta.toFixed(1);w1.Q.forEach((q,i)=>{$('#w1-q'+i+'o').textContent=fmt(q,2)});
  announce('w1',`π⁺ = ${o.pi.map(v=>v.toFixed(3)).join('、')}；log Z = KL(μ‖π⁺) = ${fmt(o.logZ)}`)}
function initW1(){segInit($('#w1-mu'),['skew','unif','conc'],w1.mu,k=>MU_NAME[k],k=>{w1.mu=k;w1Draw()});
  $('#w1-q').innerHTML=w1.Q.map((q,i)=>slider('w1-q'+i,`t${i+1} 的 Q`,-1,1,.05,q)).join('');
  w1.Q.forEach((_,i)=>{const s=$('#w1-q'+i);s.setAttribute('aria-valuetext',`t${i+1} 的 Q = ${w1.Q[i].toFixed(2)}`);s.addEventListener('input',()=>{w1.Q[i]=norm(s.value,2);s.setAttribute('aria-valuetext',`t${i+1} 的 Q = ${w1.Q[i].toFixed(2)}`);w1Draw()})});
  const e=$('#w1-eta');e.addEventListener('input',()=>{w1.eta=norm(e.value,1);e.setAttribute('aria-valuetext','η = '+w1.eta.toFixed(1));w1Draw()});w1Draw()}

/* ---------------- W2 / W3: binary tree (check.py tree, residuals, grads, paths) ---------------- */
const LEAVES=['aa','ab','ba','bb'],ST=['x','a','b'];
const tree={muA:{x:.5,a:.5,b:.5},R:{aa:1,ab:0,ba:0,bb:0},eta:1,th:{x:0,a:0,b:0}};let w2path='aa';
function treeModel(T){const m=T.muA,R=T.R,V={a:m.a*R.aa+(1-m.a)*R.ab,b:m.b*R.ba+(1-m.b)*R.bb};V.x=m.x*V.a+(1-m.x)*V.b;
  const A={xa:V.a-V.x,xb:V.b-V.x,aa:R.aa-V.a,ab:R.ab-V.a,ba:R.ba-V.b,bb:R.bb-V.b};const thMu={},thPlus={},plus={},logZ={};
  ST.forEach(s=>{thMu[s]=logit(m[s]);thPlus[s]=thMu[s]+T.eta*(A[s+'a']-A[s+'b']);plus[s]=sig(thPlus[s]);logZ[s]=lse([Math.log(m[s])+T.eta*A[s+'a'],Math.log(1-m[s])+T.eta*A[s+'b']])});
  const P={};LEAVES.forEach(y=>{P[y]=(y[0]==='a'?m.x:1-m.x)*(y[1]==='a'?m[y[0]]:1-m[y[0]])});
  return{V,A,thMu,thPlus,plus,logZ,P,eta:T.eta,muA:m,R}}
const dTok=(M,th,s,y)=>M.eta*M.A[s+y]-((y==='a'?1:0)-M.muA[s])*(th[s]-M.thMu[s]);
function residuals(M,th){const d={};ST.forEach(s=>{d[s+'a']=dTok(M,th,s,'a');d[s+'b']=dTok(M,th,s,'b')});const delta={};let L=0,J=0,Ed2=0,ED1=0,ED2=0;
  const pi={x:sig(th.x),a:sig(th.a),b:sig(th.b)};
  LEAVES.forEach(y=>{const d1=d['x'+y[0]],d2=d[y];delta[y]=d1+d2;Ed2+=M.P[y]*delta[y]**2;ED1+=M.P[y]*d1*d1;ED2+=M.P[y]*d2*d2;
    J+=(y[0]==='a'?pi.x:1-pi.x)*(y[1]==='a'?pi[y[0]]:1-pi[y[0]])*M.R[y]});
  L=Ed2/(2*M.eta);const cond={};ST.forEach(s=>{cond[s]=M.muA[s]*d[s+'a']+(1-M.muA[s])*d[s+'b']});return{d,delta,L,J,Ed2,ED1,ED2,cond,pi}}
function gvec(M,y){const g={x:0,a:0,b:0};g.x+=(y[0]==='a'?1:0)-M.muA.x;g[y[0]]+=(y[1]==='a'?1:0)-M.muA[y[0]];return g}
function grads(M,th){const r=residuals(M,th),ge={x:0,a:0,b:0},gl={x:0,a:0,b:0};
  LEAVES.forEach(y=>{const g=gvec(M,y);ST.forEach(s=>{ge[s]+=-M.P[y]*(r.delta[y]/M.eta)*g[s];gl[s]+=-M.P[y]*(M.R[y]-M.V.x)*g[s]})});return{ge,gl,r}}
let pathCache={key:'',val:null};
function gradPaths(M,steps=200){const key=JSON.stringify([M.muA,M.R,M.eta]);if(pathCache.key===key)return pathCache.val;const lr=M.eta,out={ex:[],li:[]};
  ['ex','li'].forEach(kind=>{const th=Object.assign({},M.thMu);for(let k=0;k<=steps;k++){const{ge,gl,r}=grads(M,th);const g=kind==='ex'?ge:gl;out[kind].push({pi:[r.pi.x,r.pi.a,r.pi.b],L:r.L,J:r.J,g:[g.x,g.a,g.b]});ST.forEach(s=>{th[s]-=lr*g[s]})}});
  pathCache={key,val:out};return out}
let M2=treeModel(tree);
function w2Draw(){M2=treeModel(tree);const r=residuals(M2,tree.th);w2Tree(r);w2Ladder();w2Delta(r);w2bDraw(r);
  ['x','a','b'].forEach(s=>{$('#w2-mu'+s+'o').textContent=tree.muA[s].toFixed(2)});$('#w2-etao').textContent=tree.eta.toFixed(1);
  $$('.th-group').forEach(g=>ST.forEach(s=>{const inp=$(`input[data-s="${s}"]`,g);if(document.activeElement!==inp)inp.value=tree.th[s];$(`output[data-s="${s}"]`,g).textContent=`θ ${fmt(tree.th[s],2)} · π ${fmt(sig(tree.th[s]),4)}`;inp.setAttribute('aria-valuetext',`θ = ${fmt(tree.th[s],2)}，π(a|${s}) = ${fmt(sig(tree.th[s]),4)}`)}));
  dl($('#w2-read'),[['V(x)',fmt(M2.V.x)],['π⁺(a|x, a, b)',ST.map(s=>fmt(M2.plus[s],3)).join(', ')],['L',fmt(r.L,5),1],['J(π)',fmt(r.J)]]);
  w3Draw()}
function w2Tree(r){const box=$('#w2-tree'),W=widthOf(box,300,560),H=230,svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const pos={x:[34,H/2],a:[W*.42,58],b:[W*.42,H-58],aa:[W-44,24],ab:[W-44,88],ba:[W-44,H-88],bb:[W-44,H-24]};const pi=r.pi;
  const edges=[['x','a','a'],['x','b','b'],['a','aa','a'],['a','ab','b'],['b','ba','a'],['b','bb','b']];
  edges.forEach(([u,v,act])=>{const s=u,on=(u==='x'&&w2path[0]===v)||(w2path===v);el('line',{x1:pos[u][0],y1:pos[u][1],x2:pos[v][0],y2:pos[v][1],class:'edge'+(on?' on-path':'')},svg);
    const mx=(pos[u][0]+pos[v][0])/2,my=(pos[u][1]+pos[v][1])/2,m=act==='a'?tree.muA[s]:1-tree.muA[s],p=act==='a'?pi[s]:1-pi[s];
    tx(svg,mx,my-(act==='a'?8:-14),W<480?`${m.toFixed(2)}/${p.toFixed(2)}`:`${act} μ ${m.toFixed(2)} π ${p.toFixed(2)}`,{class:'tk halo','text-anchor':'middle'})});
  ['x','a','b'].forEach(s=>{el('circle',{cx:pos[s][0],cy:pos[s][1],r:15,class:'node'},svg);tx(svg,pos[s][0],pos[s][1]+4,s,{class:'lab','text-anchor':'middle'});tx(svg,pos[s][0]+(s==='x'?0:4),pos[s][1]+(s==='x'?-22:32),'V '+fmt(M2.V[s],2),{class:'tk halo','text-anchor':'middle'})});
  if(W<480)tx(svg,4,H-4,'邊上標示 μ/π（上方分支為 a，下方為 b）',{class:'tk'});LEAVES.forEach(y=>{el('rect',{x:pos[y][0]-20,y:pos[y][1]-12,width:40,height:24,rx:6,class:'leaf'},svg);tx(svg,pos[y][0],pos[y][1]+4,y,{class:'lab','text-anchor':'middle'});tx(svg,pos[y][0]+24,pos[y][1]+4,'R '+tree.R[y],{class:'tk'})});
  svg.setAttribute('aria-label',`二元樹：V(x) = ${fmt(M2.V.x,3)}，V(a) = ${fmt(M2.V.a,3)}，V(b) = ${fmt(M2.V.b,3)}；π(a|x, a, b) = ${ST.map(s=>pi[s].toFixed(3)).join(', ')}`)}
function w2Ladder(){const box=$('#w2-ladder'),W=widthOf(box,300,560),H=W<480?200:184,svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const y0=w2path[0],vals=[M2.V.x,M2.V[y0],tree.R[w2path]],A1=M2.A['x'+y0],A2=M2.A[w2path];const L=40,R=W-10,T=14,B=H-(W<480?66:50),y=lin(-1,1,B,T),xs=[L+20,L+(R-L)*.45,R-60];
  [-1,0,1].forEach(v=>{el('line',{x1:L,x2:R,y1:y(v),y2:y(v),class:v?'grid':'axis0'},svg);tx(svg,L-6,y(v)+4,String(v),{class:'tk','text-anchor':'end'})});
  const pts=[];vals.forEach((v,i)=>{if(i)pts.push([xs[i],y(vals[i-1])]);pts.push([xs[i],y(v)])});el('path',{d:lp(pts),class:'acc-s'},svg);
  ['V(x)',`V(${y0})`,'R'].forEach((s,i)=>{el('circle',{cx:xs[i],cy:y(vals[i]),r:4,class:'dot-c'},svg);tx(svg,xs[i],B+16,s,{class:'lab','text-anchor':'middle'})});
  tx(svg,xs[1]+6,(y(vals[0])+y(vals[1]))/2+4,'A₁ '+fmt(A1,2),{class:'tk halo'});tx(svg,xs[2]+6,(y(vals[1])+y(vals[2]))/2+4,'A₂ '+fmt(A2,2),{class:'tk halo'});
  const f1=`A₁ + A₂ = ${fmt(A1,2)} + ${fmt(A2,2)} = ${fmt(A1+A2,2)}`,f2=`= R − V(x) = ${fmt(tree.R[w2path]-M2.V.x,2)}`;if(W<480){tx(svg,8,H-22,f1,{class:'tk'});tx(svg,8,H-6,f2,{class:'tk'})}else tx(svg,L,H-6,f1+' '+f2,{class:'tk'});
  svg.setAttribute('aria-label',`路徑 ${w2path}：A₁ + A₂ = ${fmt(A1+A2,2)}，等於 R − V(x)`)}
function w2Delta(r){const box=$('#w2-delta'),W=widthOf(box,300,560),H=150,svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const m=Math.max(1,...LEAVES.map(y=>Math.abs(r.delta[y]))),L=44,R=W-60,x=lin(-m,m,L,R),rh=26;tx(svg,L,12,'各軌跡的殘差 δ',{class:'tk'});
  el('line',{x1:x(0),x2:x(0),y1:18,y2:18+rh*4,class:'axis0'},svg);
  LEAVES.forEach((y,i)=>{const v=r.delta[y],yy=22+i*rh;tx(svg,L-8,yy+13,y,{class:'lab','text-anchor':'end'});
    el('rect',{x:Math.min(x(0),x(v)),y:yy,width:Math.max(1,Math.abs(x(v)-x(0))),height:rh-8,class:v>=0?'pos':'neg'},svg);
    tx(svg,v>=0?Math.max(x(v),x(0))+6:Math.min(x(v),x(0))-6,yy+13,fmt(v,5),{class:'tk','text-anchor':v>=0?'start':'end'})});
  const zero=LEAVES.every(y=>Math.abs(r.delta[y])<1e-9);tx(svg,L,H-8,zero?'L = 0：π 等於 PMD 解 π⁺':`L = ${fmt(r.L,5)}`,{class:'lab'});
  svg.setAttribute('aria-label','δ(aa, ab, ba, bb) = '+LEAVES.map(y=>fmt(r.delta[y],4)).join(', '));announce('w2',`L = ${fmt(r.L,5)}，J = ${fmt(r.J)}`)}
function w2bDraw(r){const rows=ST.map(s=>`<tr><td>${s}</td><td class="num">${fmt(r.d[s+'a'])}</td><td class="num">${fmt(r.d[s+'b'])}</td><td class="num">${fmt(r.cond[s])} ✓</td></tr>`).join('');
  $('#w2b-out').innerHTML=`<div class="tscroll" tabindex="0" role="region" aria-label="token 殘差表"><table class="mini"><caption>token 殘差 d_π(s, a) 與 Σₐ μ(a|s) d_π(s, a)</caption><thead><tr><th>狀態 s</th><th class="num">d(s, a)</th><th class="num">d(s, b)</th><th class="num">Σ μ·d</th></tr></thead><tbody>${rows}</tbody></table></div>
  <p class="okline">E[δ²] = <b>${fmt(r.Ed2,5)}</b> = E[D₁²] + E[D₂²] = ${fmt(r.ED1,5)} + ${fmt(r.ED2,5)}</p>
  <p class="okline">E[δ²] = 0 ⇔ 每個 d_π = 0（μ 可達處）⇔ π = π⁺</p>`}
function setTh(th){ST.forEach(s=>{tree.th[s]=Math.max(-12,Math.min(12,th[s]))});w2Draw()}
function initW2(){
  const muCtl=s=>{const inp=$('#w2-mu'+s);inp.addEventListener('input',()=>{tree.muA[s]=norm(inp.value,2);inp.setAttribute('aria-valuetext',`μ(a|${s}) = ${tree.muA[s].toFixed(2)}`);resetPi()})};ST.forEach(muCtl);
  const e=$('#w2-eta');e.addEventListener('input',()=>{tree.eta=norm(e.value,1);resetPi()});
  $('#w2-R').innerHTML=LEAVES.map(y=>`<div class="ctl-row"><span class="mono-sm" style="width:2.6em">R(${y})</span><div class="seg" id="w2-R-${y}" role="radiogroup" aria-label="R(${y})"></div></div>`).join('');
  LEAVES.forEach(y=>segInit($('#w2-R-'+y),[-1,0,1],tree.R[y],v=>String(v).replace('-','−'),v=>{tree.R[y]=v;resetPi()}));
  $$('.th-group').forEach(g=>{g.innerHTML=ST.map(s=>`<div class="rng"><label class="lbl" for="th-${g.dataset.g}-${s}">θ(a|${s}) <output data-s="${s}"></output></label><input type="range" id="th-${g.dataset.g}-${s}" data-s="${s}" min="-12" max="12" step="0.01" value="0"></div>`).join('');
    $$('input',g).forEach(inp=>inp.addEventListener('input',()=>{tree.th[inp.dataset.s]=norm(inp.value,2);w2Draw()}))});
  $$('[data-pi]').forEach(b=>b.addEventListener('click',()=>{const k=b.dataset.pi;if(k==='mu')setTh(M2.thMu);else if(k==='plus')setTh(M2.thPlus);else setTh({x:logit(.7),a:logit(.6),b:logit(.5)})}));
  segInit($('#w2-path'),LEAVES,w2path,null,y=>{w2path=y;w2Draw()});
  resetPi(true)}
function resetPi(silent){M2=treeModel(tree);ST.forEach(s=>{tree.th[s]=M2.thMu[s]});w2Draw();if(!silent)announce('w2','μ、R 或 η 已改變，π 已重設為 μ')}

/* W3 */
const w3={k:0,ex:true,li:true,timer:0};
function w3Draw(){const P=gradPaths(M2),k=w3.k,e=P.ex[k],l=P.li[k];
  {const box=$('#w3-sq'),S=Math.min(widthOf(box,280,560),360),svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${S} ${S}`);svg.setAttribute('width',S);svg.setAttribute('height',S);
   const L=44,R=S-12,T=12,B=S-40,x=lin(0,1,L,R),y=lin(0,1,B,T);
   [0,.5,1].forEach(v=>{el('line',{x1:x(v),x2:x(v),y1:T,y2:B,class:'grid'},svg);el('line',{x1:L,x2:R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,x(v),B+16,v.toFixed(1),{class:'tk','text-anchor':'middle'});tx(svg,L-6,y(v)+4,v.toFixed(1),{class:'tk','text-anchor':'end'})});
   tx(svg,(L+R)/2,S-6,'π(a|x)',{class:'lab','text-anchor':'middle'});tx(svg,12,(T+B)/2,'π(a|a)',{class:'lab','text-anchor':'middle',transform:`rotate(-90 12 ${(T+B)/2})`});
   const draw=(arr,cls,mk)=>{el('path',{d:lp(arr.slice(0,k+1).map(o=>[x(o.pi[0]),y(o.pi[1])])),class:cls},svg);const o=arr[k];
     if(mk==='c')el('circle',{cx:x(o.pi[0]),cy:y(o.pi[1]),r:6,class:'c1-f'},svg);else el('path',{d:`M${x(o.pi[0])} ${y(o.pi[1])-7}l6 11h-12z`,class:'c2-f'},svg)};
   if(w3.li)draw(P.li,'c2','t');if(w3.ex)draw(P.ex,'c1','c');
   const px=x(M2.plus.x),py=y(M2.plus.a);el('path',{d:`M${px-7} ${py}h14M${px} ${py-7}v14`,class:'acc-s'},svg);tx(svg,px+9,py-8,'π⁺',{class:'lab halo'});
   el('circle',{cx:x(tree.muA.x),cy:y(tree.muA.a),r:5,fill:'none',stroke:'var(--ink-2)','stroke-width':1.5},svg);tx(svg,x(tree.muA.x)+8,y(tree.muA.a)+14,'μ',{class:'lab halo'});
   svg.setAttribute('aria-label',`第 ${k} 步：精確路徑 π(a|x) = ${e.pi[0].toFixed(3)}、π(a|a) = ${e.pi[1].toFixed(3)}；線性化路徑 ${l.pi[0].toFixed(3)}、${l.pi[1].toFixed(3)}；π⁺ = (${M2.plus.x.toFixed(3)}, ${M2.plus.a.toFixed(3)})`)}
  {const box=$('#w3-lj'),W=widthOf(box,280,560),H=280,svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
   const L=48,R=W-10,x=lin(0,200,L,R);
   // L panel (log)
   const T1=14,B1=118,yl=lin(-6,3,B1,T1),lg=v=>yl(Math.max(-6,Math.min(3,Math.log10(Math.max(v,1e-12)))));
   [-6,-3,0,3].forEach(v=>{el('line',{x1:L,x2:R,y1:yl(v),y2:yl(v),class:'grid'},svg);tx(svg,L-6,yl(v)+4,v===-6?'<10⁻⁶':'10'+({'-3':'⁻³','0':'⁰','3':'³'})[v],{class:'tk','text-anchor':'end'})});
   tx(svg,L,T1-2,'L（對數）',{class:'tk'});
   if(w3.li)el('path',{d:lp(P.li.map((o,i)=>[x(i),lg(o.L)])),class:'c2'},svg);if(w3.ex)el('path',{d:lp(P.ex.map((o,i)=>[x(i),lg(o.L)])),class:'c1'},svg);
   // J panel
   const T2=150,B2=H-30,yj=lin(-1,1,B2,T2);[-1,0,1].forEach(v=>{el('line',{x1:L,x2:R,y1:yj(v),y2:yj(v),class:'grid'},svg);tx(svg,L-6,yj(v)+4,String(v).replace('-','−'),{class:'tk','text-anchor':'end'})});
   tx(svg,L,T2-4,'J(π)',{class:'tk'});const jp=P.ex[200].J;el('line',{x1:L,x2:R,y1:yj(jp),y2:yj(jp),stroke:'var(--accent)','stroke-dasharray':'3 3','stroke-width':1.2},svg);tx(svg,R,yj(jp)-4,'J(π⁺)',{class:'tk','text-anchor':'end'});
   if(w3.li)el('path',{d:lp(P.li.map((o,i)=>[x(i),yj(o.J)])),class:'c2'},svg);if(w3.ex)el('path',{d:lp(P.ex.map((o,i)=>[x(i),yj(o.J)])),class:'c1'},svg);
   el('line',{x1:x(k),x2:x(k),y1:T1,y2:B2,stroke:'var(--ink-3)','stroke-width':1},svg);[0,100,200].forEach(v=>tx(svg,x(v),H-12,String(v),{class:'tk','text-anchor':'middle'}));tx(svg,R,H,'步數 k',{class:'tk','text-anchor':'end'})}
  const rd=(o,name)=>[[name,''],['π(a|x)',fmt(o.pi[0])],['π(a|a)',fmt(o.pi[1])],['π(a|b)',fmt(o.pi[2])],['L',o.L<1e-6?'< 10⁻⁶':fmt(o.L,5),1],['J',fmt(o.J)],['∇θ',o.g.map(v=>fmt(v,3)).join(', ')]];
  dl($('#w3-re'),rd(e,'精確'));dl($('#w3-rl'),rd(l,'線性化'));$('#w3-ko').textContent=k}
function initW3(){const s=$('#w3-k');s.addEventListener('input',()=>{w3.k=+s.value;w3Draw();announce('w3',w3Msg())});
  $('#w3-ex').addEventListener('change',e=>{w3.ex=e.target.checked;w3Draw()});$('#w3-li').addEventListener('change',e=>{w3.li=e.target.checked;w3Draw()});
  const btn=$('#w3-play');btn.addEventListener('click',()=>{if(w3.timer){cancelAnimationFrame(w3.timer);w3.timer=0;btn.textContent='播放';announce('w3',w3Msg());return}
    if(RM){w3.k=200;s.value=200;w3Draw();announce('w3',w3Msg());return}if(w3.k>=200)w3.k=0;btn.textContent='暫停';
    const step=()=>{w3.k=Math.min(200,w3.k+2);s.value=w3.k;w3Draw();if(w3.k<200)w3.timer=requestAnimationFrame(step);else{w3.timer=0;btn.textContent='播放';announce('w3',w3Msg())}};w3.timer=requestAnimationFrame(step)})}
function w3Msg(){const P=gradPaths(M2);return`第 ${w3.k} 步：精確 L = ${fmt(P.ex[w3.k].L,5)}，線性化 L = ${fmt(P.li[w3.k].L,5)}`}

/* ---------------- W4: logit-space gradients (check.py w5) ---------------- */
const MU4=[.4,.3,.2,.1];const w4={y:0,d:[0,1,0,0],show:{full:true,bin:true,bpo:true,grpo:false}};
const V4=[['full','完整 KL：e_y − μ','solid1'],['bin','二元 KL','hat2'],['bpo','BPO：ω(e_y − π)','dots'],['grpo','GRPO：r(e_y − π)','hollow']];
function logitGrads(mu,delta,y,eps){const zmu=mu.map(Math.log),pi=softmax(zmu.map((z,i)=>z+delta[i]));const q=mu[y],p=pi[y],omp=pi.reduce((a,v,i)=>i===y?a:a+v,0);
  const e=mu.map((_,i)=>i===y?1:0),pg=e.map((v,i)=>v-pi[i]),full=e.map((v,i)=>v-mu[i]),bin=pg.map(v=>(1-q)/omp*v),om=(1+eps-q)/(eps+omp),bpo=pg.map(v=>om*v),r=p/q,grpo=pg.map(v=>r*v);
  let kl=0;for(let i=0;i<mu.length;i++)kl+=mu[i]*(Math.log(mu[i])-Math.log(pi[i]));const klb=q*Math.log(q/p)+(1-q)*Math.log((1-q)/omp);
  const dot=bin.reduce((a,v,i)=>a+v*full[i],0),nb=Math.hypot(...bin),nf=Math.hypot(...full),cos=nb*nf<1e-12?NaN:dot/(nb*nf);
  return{pi,q,p,pg,full,bin,bpo,grpo,om,r,kl,klb,cos}}
function w4Draw(){const o=logitGrads(MU4,w4.d,w4.y,w5.eps),shown=V4.filter(v=>w4.show[v[0]]);const box=$('#w4-fig'),W=widthOf(box,300,560);
  const bh=12,gap=4,rowH=Math.max(1,shown.length)*(bh+gap)+22,H=rowH*4+30,svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const defs=el('defs',null,svg);const p1=el('pattern',{id:'w4h',width:5,height:5,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},defs);el('line',{x1:0,y1:0,x2:0,y2:5,class:'hatch2-ln'},p1);
  const p2=el('pattern',{id:'w4d',width:4,height:4,patternUnits:'userSpaceOnUse'},defs);el('circle',{cx:2,cy:2,r:1.2,fill:'var(--accent)'},p2);
  const m=Math.max(1,...shown.flatMap(v=>o[v[0]].map(Math.abs))),L=54,R=W-56,x=lin(-m,m,L,R);
  el('line',{x1:x(0),x2:x(0),y1:6,y2:H-22,class:'axis0'},svg);[-m,0,m].forEach(v=>tx(svg,x(v),H-6,fmt(v,1),{class:'tk','text-anchor':'middle'}));
  for(let i=0;i<4;i++){const y0=8+i*rowH;tx(svg,4,y0+14,'t'+(i+1)+(i===w4.y?' 取樣':''),{class:'lab'+(i===w4.y?' halo':''),'font-weight':i===w4.y?600:400});
    tx(svg,4,y0+30,'π '+o.pi[i].toFixed(3),{class:'tk'});
    shown.forEach(([k],j)=>{const v=o[k][i],yy=y0+4+j*(bh+gap),xa=Math.min(x(0),x(v)),w=Math.max(1,Math.abs(x(v)-x(0)));
      const a={x:xa,y:yy,width:w,height:bh};if(k==='full')a.class='c1-f';else if(k==='bin'){a.fill='url(#w4h)';a.stroke='var(--c-2)'}else if(k==='bpo'){a.fill='url(#w4d)';a.stroke='var(--accent)'}else{a.fill='none';a.stroke='var(--ink-2)';a['stroke-width']=1.5}
      el('rect',a,svg);tx(svg,v>=0?xa+w+4:xa-4,yy+10,fmt(v,3),{class:'tk','text-anchor':v>=0?'start':'end'})});
    if(i===w4.y)el('line',{x1:4,x2:44,y1:y0+18,y2:y0+18,stroke:'var(--accent)','stroke-width':2},svg)}
  $('#w4-lg').innerHTML=shown.map(v=>`<span class="lg box ${v[2]}">${v[1]}</span>`).join('');
  w4.d.forEach((v,i)=>{$('#w4-d'+i+'o').textContent=fmt(v,2);const s=$('#w4-d'+i);if(document.activeElement!==s)s.value=v});
  dl($('#w4-read'),[['q = μ(y)',fmt(o.q)],['p = π(y)',fmt(o.p)],['ω（ϵ = '+w5.eps.toFixed(2)+'）',fmt(o.om)],['r = p/q',fmt(o.r)],['KL(μ‖π)',fmt(o.kl,5)],['二元 KL（≤ KL）',fmt(o.klb,5)],['餘弦（二元, 完整）',Number.isFinite(o.cos)?fmt(o.cos):'—',1]]);
  svg.setAttribute('aria-label',`取樣 t${w4.y+1}：完整 KL 與二元 KL 的取樣分量都是 ${fmt(1-o.q,3)}；餘弦 ${Number.isFinite(o.cos)?fmt(o.cos):'—'}`);announce('w4',`ω = ${fmt(o.om)}，餘弦 = ${Number.isFinite(o.cos)?fmt(o.cos):'—'}`)}
function initW4(){segInit($('#w4-y'),[0,1,2,3],w4.y,i=>'t'+(i+1),i=>{w4.y=i;w4Draw()});
  $('#w4-d').innerHTML=w4.d.map((v,i)=>slider('w4-d'+i,`Δ${'₁₂₃₄'[i]}（t${i+1} 的 logit 偏移）`,-3,3,.05,v)).join('');
  w4.d.forEach((_,i)=>{const s=$('#w4-d'+i);s.addEventListener('input',()=>{w4.d[i]=norm(s.value,2);s.setAttribute('aria-valuetext',`Δ${i+1} = ${w4.d[i].toFixed(2)}`);w4Draw()})});
  $('#w4-mu').addEventListener('click',()=>{w4.d=[0,0,0,0];w4Draw()});$('#w4-only').addEventListener('click',()=>{w4.d=[0,0,0,0];w4.d[w4.y]=1;w4Draw()});
  $('#w4-show').innerHTML=V4.map(v=>`<label class="chk"><input type="checkbox" data-k="${v[0]}"${w4.show[v[0]]?' checked':''}>${v[1]}</label>`).join('');
  $$('#w4-show input').forEach(c=>c.addEventListener('change',()=>{w4.show[c.dataset.k]=c.checked;w4Draw()}));w4Draw()}

/* ---------------- W5: scalar r vs omega (check.py w4) ---------------- */
const w5={q:.5,p:.6,eps:.1,C:3,sg:1,mask:false,lo:.2,hi:.28};
function ratioCard(q,p,eps,C,sg,mask){const r=ratio(p,q),raw=ratio(1-q,1-p),om=ratio(1+eps-q,1+eps-p),capped=om===UNDEF?UNDEF:Math.min(om,C);
  const M=w=>{if(!mask||w===UNDEF)return 1;if(sg>0&&w>1+mask[1])return 0;if(sg<0&&w<1-mask[0])return 0;return 1};const Mr=M(r),Mw=M(om);
  return{r,raw,om,capped,Mr,Mw,wG:Mr===0?0:r,wB:Mw===0?0:capped}}
function w5Draw(){const o=ratioCard(w5.q,w5.p,w5.eps,w5.C,w5.sg,w5.mask?[w5.lo,w5.hi]:null);
  dl($('#w5-rg'),[['GRPO',''],['r = p/q',fmt(o.r)],['遮罩 M',w5.mask?(o.Mr?'1':'0（被截斷）'):'—'],['有效權重',fmt(o.wG),1]]);
  dl($('#w5-rb'),[['BPO',''],['(1−q)/(1−p)',fmt(o.raw)],['ω',fmt(o.om)],['min(ω, C)',fmt(o.capped)],['遮罩 M',w5.mask?(o.Mw?'1':'0（被截斷）'):'—'],['有效權重',fmt(o.wB),1]]);
  [['q',3],['p',3],['eps',2],['C',1],['lo',2],['hi',2]].forEach(([k,n])=>{const out=$('#w5-'+k+'o');if(out)out.textContent=w5[k].toFixed(n)});
  heatOverlay('w5-h1','r');heatOverlay('w5-h2','w');announce('w5',`r = ${fmt(o.r)}，ω = ${fmt(o.om)}，遮罩 = ${w5.mask?o.Mw:'未套用'}`);if($('#w4-d0'))w4Draw()}
function hex(c){c=c.trim();if(c[0]==='#'&&c.length>=7)return[1,3,5].map(i=>parseInt(c.slice(i,i+2),16));const m=c.match(/\d+/g);return m?m.slice(0,3).map(Number):[128,128,128]}
let heatKey='';
function heatGeom(wrap){const W=wrap.clientWidth||260,H=wrap.clientHeight||260;return{W,H,L:34,R:W-6,T:28,B:H-28}}
function drawHeat(force){const key=[w5.eps,document.documentElement.dataset.theme||'',matchMedia('(prefers-color-scheme: dark)').matches,$('#w5-h1').clientWidth].join('|');if(!force&&key===heatKey)return;heatKey=key;
  const cs=getComputedStyle(document.documentElement),cn=hex(cs.getPropertyValue('--c-par')),cp=hex(cs.getPropertyValue('--accent')),cm=hex(cs.getPropertyValue('--surface')),N=120,dpr=devicePixelRatio||1;
  [['w5-h1','r'],['w5-h2','w']].forEach(([id,kind])=>{const wrap=$('#'+id),cv=$('canvas',wrap),g=heatGeom(wrap),cw=g.R-g.L,ch=g.B-g.T;cv.width=Math.max(1,Math.round(cw*dpr));cv.height=Math.max(1,Math.round(ch*dpr));
    const off=document.createElement('canvas');off.width=N;off.height=N;const ox=off.getContext('2d'),img=ox.createImageData(N,N);
    for(let j=0;j<N;j++)for(let i=0;i<N;i++){const q=(i+.5)/N,p=1-(j+.5)/N;const v=kind==='r'?Math.log2(p/q):Math.log2((1+w5.eps-q)/(1+w5.eps-p));const t=Math.max(-1,Math.min(1,(Number.isFinite(v)?v:Math.sign(v)*4)/4));
      const c=t<0?cn:cp,a=Math.abs(t),o=4*(j*N+i);for(let k=0;k<3;k++)img.data[o+k]=Math.round(cm[k]+(c[k]-cm[k])*a);img.data[o+3]=255}
    ox.putImageData(img,0,0);const cx=cv.getContext('2d');cx.imageSmoothingEnabled=false;cx.drawImage(off,0,0,cv.width,cv.height)})}
function heatOverlay(id,kind){drawHeat(false);const wrap=$('#'+id),svg=$('svg',wrap),g=heatGeom(wrap);svg.textContent='';svg.setAttribute('viewBox',`0 0 ${g.W} ${g.H}`);
  const x=lin(0,1,g.L,g.R),y=lin(0,1,g.B,g.T);[0,.5,1].forEach(v=>{tx(svg,x(v),g.B+14,v.toFixed(1),{class:'tk','text-anchor':'middle'});tx(svg,g.L-4,y(v)+4,v.toFixed(1),{class:'tk','text-anchor':'end'})});
  tx(svg,g.R,g.B+26,'q',{class:'lab','text-anchor':'end'});tx(svg,g.L-4,g.T-8,'p',{class:'lab','text-anchor':'end'});
  const e=w5.eps,levels=[[.5,'0.5'],[1,'1'],[2,'2'],[w5.C,'C']];
  levels.forEach(([c,lab])=>{let a,b;if(kind==='r'){a=0;b=c}else{a=1+e-(1+e)/c;b=1/c}   // p = a + b q
    let q0=0,q1=1;if(Math.abs(b)>1e-12){const qa=(0-a)/b,qb=(1-a)/b;q0=Math.max(0,Math.min(qa,qb));q1=Math.min(1,Math.max(qa,qb))}else if(a<0||a>1)return;if(q1<=q0)return;
    el('line',{x1:x(q0),y1:y(a+b*q0),x2:x(q1),y2:y(a+b*q1),stroke:lab==='C'?'var(--accent)':'var(--ink)','stroke-width':lab==='1'?1.6:1.1,'stroke-dasharray':lab==='C'?'5 3':''},svg);
    tx(svg,x(q1)+(q1>.95?-2:3),y(a+b*q1)+(q1>.95?12:-3),lab,{class:'tk halo','text-anchor':q1>.95?'end':'start'})});
  const px=x(w5.q),py=y(w5.p);el('path',{d:`M${px-8} ${py}h16M${px} ${py-8}v16`,stroke:'var(--ink)','stroke-width':2.2},svg)}
function initW5(){const P=[[.5,.6],[.01,.02],[.98,.99],[.5,.9],[0,1],[1,0]];
  $('#w5-pre').innerHTML=P.map(([q,p],i)=>`<button class="small-btn" type="button" data-i="${i}">(${q}, ${p})</button>`).join('');
  $$('#w5-pre button').forEach(b=>b.addEventListener('click',()=>{const[q,p]=P[+b.dataset.i];w5.q=q;w5.p=p;$('#w5-q').value=q;$('#w5-p').value=p;w5Draw()}));
  [['q',3],['p',3],['eps',2],['C',1],['lo',2],['hi',2]].forEach(([k,n])=>{const s=$('#w5-'+k);s.addEventListener('input',()=>{w5[k]=norm(s.value,n);s.setAttribute('aria-valuetext',k+' = '+w5[k].toFixed(n));if(k==='eps')heatKey='';w5Draw()})});
  segInit($('#w5-sg'),[1,-1],w5.sg,v=>v>0?'Â > 0':'Â < 0',v=>{w5.sg=v;w5Draw()});
  $('#w5-m').addEventListener('change',e=>{w5.mask=e.target.checked;$('#w5-mbox').classList.toggle('hide',!w5.mask);w5Draw()});
  new MutationObserver(()=>{heatKey='';drawHeat(true)}).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{heatKey='';drawHeat(true)});w5Draw()}

/* ---------------- static figures ---------------- */
function drawChain(){const box=$('#s1-chain'),W=widthOf(box,300,640),H=110,svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const labs=W<420?['s₁ = x','s₂','…','(x, y)']:['s₁ = x','s₂ = (x, y₁)','…','s_{|y|+1} = (x, y)'];const n=labs.length,xs=labs.map((_,i)=>20+(W-40)*(i+.5)/n),bw=Math.min(130,(W-40)/n-18);
  labs.forEach((s,i)=>{if(s!=='…'){el('rect',{x:xs[i]-bw/2,y:30,width:bw,height:32,rx:8,class:i===n-1?'leaf':'node'},svg)}tx(svg,xs[i],51,s.replace('_{|y|+1}','|y|+1'),{class:'lab','text-anchor':'middle'});
    if(i<n-1){el('line',{x1:xs[i]+bw/2+2,x2:xs[i+1]-bw/2-6,y1:46,y2:46,class:'edge'},svg);tx(svg,(xs[i]+xs[i+1])/2,24,i===0?'接上 y₁':i===1?'接上 y₂…':'',{class:'tk','text-anchor':'middle'})}});
  tx(svg,xs[n-1],86,'R(x, y)',{class:'lab','text-anchor':'middle',fill:'var(--accent)'});tx(svg,xs[0],86,'中間獎勵 0',{class:'tk','text-anchor':'middle'})}
const T1AVG=[['GRPO-ClipHigher',39.5],['GSPO',43.5],['CISPO',47.4],['DPPO',46.4],['BPO',50.5]];
function drawBars(){const box=$('#s9-bars'),W=widthOf(box,300,640),H=180,svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const L=W<420?104:128,R=W-44,x=lin(0,60,L,R);tx(svg,L,12,'Table 1 平均欄（%）',{class:'tk'});
  T1AVG.forEach(([n,v],i)=>{const y=22+i*30;tx(svg,L-8,y+15,n,{class:'lab','text-anchor':'end'});el('rect',{x:L,y,width:x(v)-L,height:20,class:n==='BPO'?'bar-acc':'bar'},svg);tx(svg,x(v)+6,y+15,v.toFixed(1),{class:'tk'})})}
function drawAbl(){const box=$('#s9-abl'),W=widthOf(box,300,640),H=200,svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const rows=[['ϵ = 0.05',25.8],['ϵ = 0.1',25.4],['ϵ = 0.2',25.5],['ϵ = 0.3',24.1],['C = 2',25.3],['C = 3',25.4],['C = 4',25.8]],L=80,R=W-50,x=lin(18,28,L,R);
  [18,20,22,24,26,28].forEach(v=>{el('line',{x1:x(v),x2:x(v),y1:16,y2:H-24,class:'grid'},svg);tx(svg,x(v),H-8,String(v),{class:'tk','text-anchor':'middle'})});
  el('line',{x1:x(20.5),x2:x(20.5),y1:16,y2:H-24,stroke:'var(--ink-2)','stroke-dasharray':'5 4','stroke-width':1.5},svg);tx(svg,x(20.5)+4,14,'GRPO-ClipHigher 20.5',{class:'tk'});
  rows.forEach(([n,v],i)=>{const y=30+i*22;tx(svg,L-8,y+4,n,{class:'lab','text-anchor':'end'});el('circle',{cx:x(v),cy:y,r:5,class:i<4?'c1-f':'c2-f'},svg);tx(svg,x(v)+8,y+4,v.toFixed(1),{class:'tk'})})}

/* ================= timeline ================= */
const TL_GROUP={'1707.06347':['pg','bpo-ppo-grpo-ratio-clipping'],'2402.03300':['pg','bpo-ppo-grpo-ratio-clipping'],'2501.12948':['pg','bpo-ppo-grpo-ratio-clipping'],'2503.14476':['pg','bpo-ppo-grpo-ratio-clipping'],
 '2506.13585':['bl','bpo-trust-region-variants-gspo-cispo-dppo'],'2507.18071':['bl','bpo-trust-region-variants-gspo-cispo-dppo'],'2602.04879':['bl','bpo-trust-region-variants-gspo-cispo-dppo'],
 '2005.09814':['pmd','bpo-policy-mirror-descent'],'2102.00135':['pmd','bpo-policy-mirror-descent'],'2201.07443':['pmd','bpo-policy-mirror-descent'],
 '2410.01679':['val','bpo-terminal-reward-bellman-telescoping'],'2305.18290':['lr','bpo-likelihood-ratio-reparameterization'],'2404.12358':['lr','bpo-likelihood-ratio-reparameterization'],'2606.20008':['lr','bpo-likelihood-ratio-reparameterization'],
 '2609.15987':['me','bpo-derivation-and-results']};
const TL_NAME={pg:'PPO／GRPO',bl:'基線變體',pmd:'PMD',val:'價值估計',lr:'likelihood ratio',me:'本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.slice().sort((a,b)=>(Number(a.year)||0)-(Number(b.year)||0));
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];if(!TL_GROUP[e.arxiv])console.warn('timeline: ungrouped',e.arxiv);used.add(g[0]);const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${g[1]?`<a href="${g[1]}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
  const on=new Set(used);
  Object.keys(TL_NAME).filter(k=>used.has(k)).forEach(k=>{const b=document.createElement('button');b.type='button';b.className='chip';b.setAttribute('aria-pressed','true');b.textContent=TL_NAME[k];b.dataset.g=k;
    b.addEventListener('click',()=>{if(on.has(k)&&on.size===used.size){on.clear();on.add(k)}else if(on.has(k)){on.delete(k);if(!on.size)used.forEach(x=>on.add(x))}else on.add(k);
      $$('.chip',chips).forEach(c=>c.setAttribute('aria-pressed',String(on.has(c.dataset.g))));$$('.ev',track).forEach(ev=>ev.classList.toggle('dim',!on.has(ev.dataset.g)))});chips.appendChild(b)});
  const rail=$('#rail'),by=dir=>rail.scrollBy({left:dir*Math.min(rail.clientWidth*.8,560),behavior:RM?'auto':'smooth'});
  $('#tl-prev').addEventListener('click',()=>by(-1));$('#tl-next').addEventListener('click',()=>by(1));
  rail.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){by(1);e.preventDefault()}if(e.key==='ArrowLeft'){by(-1);e.preventDefault()}});return data}

/* ================= toc, theme ================= */
function initToc(){const secs=$$('[data-toc]'),ol=$('#toc'),bar=$('#tocbar');
  secs.forEach(s=>{ol.insertAdjacentHTML('beforeend',`<li><a href="#${s.id}"><span class="n">${s.dataset.toc}</span><span>${s.dataset.tt}</span></a></li>`);bar.insertAdjacentHTML('beforeend',`<a href="#${s.id}">${s.dataset.toc} ${s.dataset.tt}</a>`)});
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;$$('.toc a,.tocbar a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+id));const ab=$(`.tocbar a[href="#${id}"]`);if(ab&&bar.offsetParent)bar.scrollTo({left:ab.offsetLeft-16,behavior:RM?'auto':'smooth'})}})},{rootMargin:'-40% 0px -55% 0px'});
  secs.forEach(s=>io.observe(s));addEventListener('scroll',()=>$('#appbar').classList.toggle('raised',scrollY>8),{passive:true})}
function initTheme(){let saved=null;try{saved=localStorage.getItem('pm-theme')}catch(e){}if(saved)document.documentElement.dataset.theme=saved;
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){}})}

/* ================= self test (design.md section 5) ================= */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t=5e-5)=>Math.abs(a-b)<=t+1e-12,nearv=(a,b,t=5e-5)=>a.length===b.length&&a.every((v,i)=>near(v,b[i],t));
  // W1
  {const o=pmd(MU_PRE.skew,[.5,0,-.5,1],2);ok(nearv(o.pi,[.4943,.1364,.0334,.3359])&&near(o.logZ,.3884)&&near(o.klMP,.3884)&&Math.abs(o.logZ-o.klMP)<1e-12&&near(o.klPM,.3443)&&near(o.obj,.1942)&&near(o.logZQ,.7884)&&Math.abs(o.obj-o.logZ/2)<1e-12,'W1 default')}
  {const o=pmd(MU_PRE.skew,[.5,0,-.5,1],0);ok(nearv(o.pi,[.4,.3,.2,.1],1e-12)&&Math.abs(o.logZ)<1e-12,'W1 eta=0')}
  {const o=pmd(MU_PRE.skew,[.5,0,-.5,1],10);ok(nearv(o.pi,[.0262,.0001,0,.9736])&&near(o.logZ,5.7241)&&near(o.klPM,2.1433),'W1 eta=10')}
  {const o=pmd(MU_PRE.unif,[.5,0,-.5,1],2);ok(nearv(o.pi,[.2369,.0871,.0321,.6439])&&near(o.logZ,.5539),'W1 uniform')}
  {const o=pmd(MU_PRE.conc,[.5,0,-.5,1],2);ok(nearv(o.pi,[.8407,.0182,.0067,.1344])&&near(o.logZ,.1110)&&near(o.logZQ,1.0110),'W1 concentrated')}
  // W2
  const caseI={muA:{x:.5,a:.5,b:.5},R:{aa:1,ab:0,ba:0,bb:0},eta:1},caseII={muA:{x:.7,a:.3,b:.6},R:{aa:1,ab:0,ba:0,bb:1},eta:2};
  const ex={x:logit(.7),a:logit(.6),b:logit(.5)};const dv=r=>LEAVES.map(y=>r.delta[y]);
  {const M=treeModel(caseI);ok(near(M.V.x,.25,1e-12)&&near(M.V.a,.5,1e-12)&&near(M.V.b,0,1e-12)&&nearv(ST.map(s=>M.plus[s]),[.62246,.73106,.5],5e-6)&&nearv(ST.map(s=>M.logZ[s]),[.03093,.12011,0],5e-6),'W2 case I model');
   ST.forEach(s=>{const q=M.muA[s],p=M.plus[s];ok(Math.abs(M.logZ[s]-(q*Math.log(q/p)+(1-q)*Math.log((1-q)/(1-p))))<1e-12,'logZ = KL at '+s)});
   const a=residuals(M,M.thMu),b=residuals(M,M.thPlus),c=residuals(M,ex);
   ok(nearv(dv(a),[.75,-.25,-.25,-.25],1e-12)&&near(a.L,.09375,1e-12)&&near(a.J,.25,1e-12)&&near(a.ED1,.0625,1e-12)&&near(a.ED2,.125,1e-12),'W2 case I π=μ');
   ok(b.L<1e-12&&near(b.J,.4551),'W2 case I π=π⁺: L='+b.L);ok(nearv(dv(c),[.12362,-.47092,.17365,.17365],5e-6)&&near(c.L,.03717,5e-6)&&near(c.J,.42)&&near(c.ED1,.03015,5e-6)&&near(c.ED2,.04418,5e-6),'W2 case I example');
   [a,b,c].forEach((r,i)=>ok(ST.every(s=>Math.abs(r.cond[s])<1e-12)&&Math.abs(r.Ed2-r.ED1-r.ED2)<1e-12,'W2 cond/decomp '+i));
   LEAVES.forEach(y=>ok(Math.abs(M.A['x'+y[0]]+M.A[y]-(M.R[y]-M.V.x))<1e-12,'telescoping '+y))}
  {const M=treeModel(caseII);ok(near(M.V.x,.33,1e-12)&&near(M.V.a,.3,1e-12)&&near(M.V.b,.4,1e-12)&&nearv(ST.map(s=>M.plus[s]),[.6564,.76,.16875],5e-6)&&nearv(ST.map(s=>M.logZ[s]),[.00431,.47046,.46853],5e-6),'W2 case II model');
   const a=residuals(M,M.thMu),b=residuals(M,M.thPlus),c=residuals(M,ex);ok(nearv(dv(a),[1.34,-.66,-.66,1.34],1e-12)&&near(a.L,.2211,1e-12)&&near(a.ED1,.0084,1e-12)&&near(a.ED2,.876,1e-12),'W2 case II π=μ');
   ok(b.L<1e-12&&near(b.J,.7845),'W2 case II π⁺');ok(nearv(dv(c),[.46307,-.28417,-.49781,1.09672],5e-6)&&near(c.L,.06839,5e-6)&&near(c.J,.57),'W2 case II example')}
  // W3
  {const M=treeModel(caseI),P=gradPaths(M);const chk=(o,pi,L,tol=1e-4)=>nearv(o.pi,pi,tol)&&(L===0?o.L<1e-6:Math.abs(o.L-L)<=1e-4*Math.max(1,L));
   ok(chk(P.ex[10],[.6158,.6763,.5],.00442)&&chk(P.ex[50],[.6225,.7308,.5],0)&&chk(P.ex[200],[.6225,.7311,.5],0),'W3 case I exact');
   ok(chk(P.li[10],[.7773,.7773,.5],.07422)&&chk(P.li[50],[.9981,.9981,.5],5.85547)&&chk(P.li[200],[1,1,.5],111.03125),'W3 case I linearized');
   const g=grads(M,M.thPlus);ok(nearv([g.ge.x,g.ge.a,g.ge.b],[0,0,0],1e-12)&&nearv([g.gl.x,g.gl.a,g.gl.b],[-.125,-.125,0],1e-12),'W3 grads at π⁺');
   const h={x:.25,a:.125,b:.125};ok([10,50,200].every(k=>ST.every((s,i)=>Math.abs(sig(M.thPlus[s]+(1-h[s])**k*(M.thMu[s]-M.thPlus[s]))-P.ex[k].pi[i])<1e-12)),'W3 closed form')}
  {const M=treeModel(caseII),P=gradPaths(M);ok(nearv(P.ex[10].pi,[.6607,.6781,.3436],1e-4)&&nearv(P.li[50].pi,[.2222,1,.0011],1e-4)&&Math.abs(P.li[200].L-135.02274)<.02,'W3 case II')}
  // W4
  {const o=logitGrads(MU4,[0,1,0,0],0,.1);ok(nearv(o.pi,[.2639,.5381,.132,.066])&&nearv(o.bin,[.6,-.4386,-.1076,-.0538])&&nearv(o.bpo,[.6163,-.4505,-.1105,-.0552])&&nearv(o.grpo,[.4857,-.3551,-.0871,-.0435])&&near(o.om,.8373)&&near(o.cos,.9739)&&near(o.kl,.11574,5e-6)&&near(o.klb,.04367,5e-6),'W4 default')}
  {const o=logitGrads(MU4,[1,0,0,0],0,.1);ok(nearv(o.bin,o.full,1e-12)&&Math.abs(o.kl-o.klb)<1e-12&&near(o.om,1.5365),'W4 only y logit: binary = full')}
  {const o=logitGrads(MU4,[0,0,0,0],0,.1);ok(nearv(o.grpo,o.full,1e-12)&&nearv(o.bpo,o.full,1e-12)&&near(o.om,1,1e-12),'W4 π=μ all equal')}
  {const o=logitGrads(MU4,[0,0,0,2],3,.1);ok(nearv(o.grpo,[-1.1004,-.8253,-.5502,2.4758])&&near(o.om,1.5405),'W4 t4')}
  {const o=logitGrads([.5,.3,.2],[1,0,-1].map((z,i)=>z-Math.log([.5,.3,.2][i])),0,.1);ok(nearv(o.pi,[.6652,.2447,.09])&&nearv(o.pg,[.3348,-.2447,-.09]),'S5 3-token example')}
  // W5
  const W5T=[[.5,.6,1.2,1.25,1.2,1.2],[.01,.02,2,1.0102,1.0093,1.0093],[.98,.99,1.0102,2,1.0909,1.0909],[.5,.9,1.8,5,3,3],[0,1,Infinity,Infinity,11,3],[1,0,0,0,.0909,.0909],[.5,.5,1,1,1,1]];
  W5T.forEach(([q,p,r,raw,om,cp])=>{const o=ratioCard(q,p,.1,3,1,null);const eq=(a,b)=>b===Infinity?a===Infinity:near(a,b);ok(eq(o.r,r)&&eq(o.raw,raw)&&eq(o.om,om)&&eq(o.capped,cp),`W5 (${q}, ${p})`)});
  ok(near(ratioCard(.5,.6,0,3,1,null).om,1.25),'W5 eps=0');
  {const m=[.2,.28];const a=ratioCard(.5,.9,.1,3,1,m),b=ratioCard(.5,.9,.1,3,-1,m),c=ratioCard(.5,.2,.1,3,-1,m),d=ratioCard(.01,.02,.1,3,1,m);
   ok(a.Mr===0&&a.Mw===0&&b.Mr===1&&b.Mw===1&&near(b.wB,3)&&near(b.wG,1.8)&&c.Mr===0&&c.Mw===0&&near(c.om,.6667)&&d.Mr===0&&d.Mw===1,'W5 mask demo')}
  {const a=ratioCard(1,1,.1,3,1,null),b=ratioCard(1,1,0,3,1,null),c=ratioCard(0,0,.1,3,1,null),d=ratioCard(.5,1,0,3,1,[.2,.28]),e=ratioCard(0,.5,.1,3,1,[.2,.28]);
   ok(a.raw===UNDEF&&near(a.om,1)&&b.om===UNDEF&&b.capped===UNDEF&&c.r===UNDEF&&near(c.om,1)&&d.om===Infinity&&d.capped===3&&d.wB===0&&d.wG===0&&e.r===Infinity&&e.wG===0,'W5 undefined / ∞ / no NaN')}
  // paper arithmetic
  {const T=[[45.6,34.8,38.0,39.5],[50.3,35.5,44.6,43.5],[52.7,39.0,50.4,47.4],[55.8,39.2,44.2,46.4],[57.4,41.0,53.0,50.5]];ok(T.every(r=>Math.abs((r[0]+r[1]+r[2])/3-r[3])<.05),'Table 1 averages');
   ok([11,7,3.1,4.1].every((g,i)=>near(50.5-T[i][3],g,1e-9)),'Table 1 gaps')}
  // DOM
  ok($$('details.check').length===10,'check boxes: '+$$('details.check').length);
  ok(['w1','w2a','w2b','w3','w4','w5'].every(id=>!!$('#'+id)),'plates present');
  ok($$('.scope-band').length===5&&$$('.scope-band [aria-current=step]').length===4,'scope bands');
  {const tl=JSON.parse($('#timeline-data').textContent);ok(tl.length===15&&tl.every(e=>TL_GROUP[e.arxiv]),'timeline groups')}
  ok(!!document.querySelector('.katex')||typeof katex==='undefined','KaTeX rendered');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ================= boot ================= */
initTheme();initToc();drawChain();initW1();initW2();initW3();initW5();initW4();drawBars();drawAbl();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{drawChain();w1Draw();w2Draw();heatKey='';w5Draw();drawBars();drawAbl()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))addEventListener('load',()=>setTimeout(selfTest,50));
})();
