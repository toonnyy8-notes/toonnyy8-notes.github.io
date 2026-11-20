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
function frame(box,W,H){const svg=$('svg',box);svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);return svg}
const f4=v=>(Math.abs(v)<5e-5?0:v).toFixed(4),f3=v=>(Math.abs(v)<5e-4?0:v).toFixed(3),f2=v=>(Math.abs(v)<5e-3?0:v).toFixed(2);
const fmtInt=v=>Math.round(v).toLocaleString('en-US');
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function segInit(box,opts,cur,label,onPick){box.textContent='';opts.forEach(o=>{const b=document.createElement('button');b.type='button';b.setAttribute('role','radio');
    b.textContent=label?label(o):String(o);b.dataset.v=String(o);box.appendChild(b)});
  const sync=v=>$$('button',box).forEach(b=>{const on=b.dataset.v===String(v);b.setAttribute('aria-checked',String(on));b.setAttribute('aria-pressed',String(on));b.tabIndex=on?0:-1});sync(cur);
  box.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;const v=opts.find(o=>String(o)===b.dataset.v);sync(v);onPick(v)});
  box.addEventListener('keydown',e=>{const bs=$$('button',box),i=bs.findIndex(b=>b.tabIndex===0);let j=i;if(e.key==='ArrowRight'||e.key==='ArrowDown')j=(i+1)%bs.length;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')j=(i-1+bs.length)%bs.length;else return;e.preventDefault();bs[j].focus();bs[j].click()});
  return sync}
function pageKeys(input,frac){input.addEventListener('keydown',e=>{if(e.key!=='PageUp'&&e.key!=='PageDown')return;e.preventDefault();const mn=+input.min,mx=+input.max,st=Math.max(1,Math.round((mx-mn)*frac));input.value=Math.min(mx,Math.max(mn,+input.value+(e.key==='PageUp'?st:-st)));input.dispatchEvent(new Event('input'));input.dispatchEvent(new Event('change'))})}
function hatchDefs(svg){const d=el('defs',null,svg);
  const p=el('pattern',{id:'hatch',width:6,height:6,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},d);el('line',{x1:0,y1:0,x2:0,y2:6,class:'hatch-ln'},p);
  const q=el('pattern',{id:'hneg',width:5,height:5,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},d);el('line',{x1:0,y1:0,x2:0,y2:5,class:'hneg-ln'},q)}

/* ================= shared math (mirrors verify/*.py) ================= */
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
// Eq. 3 with the page's guard: std <= 1e-12 -> all advantages 0
function groupAdv(r,ddof=0){const G=r.length;let m=0;for(const v of r)m+=v;m/=G;let s=0;for(const v of r)s+=(v-m)**2;const sd=Math.sqrt(s/(G-ddof));
  const zero=!(sd>1e-12);return{A:r.map(v=>zero?0:(v-m)/sd),mean:m,sd,zero}}
function pEqual(p,f,G){const q0=(1-p)*(1-f),q1=p*(1-f)+(1-p)*f,q2=p*f;return q0**G+q1**G+q2**G}
// C2: per-output objective and its analytic derivative wrt x = log rho (pi_ref = pi_old)
function c2J(x,A,e,b){const rho=Math.exp(x),c=Math.min(Math.max(rho,1-e),1+e);return Math.min(rho*A,c*A)-b*(1/rho+x-1)}
function c2Active(rho,A,e){return A>0?rho<=1+e:(A<0?rho>=1-e:false)}
function c2dJ(x,A,e,b){const rho=Math.exp(x);return(c2Active(rho,A,e)?rho*A:0)-b*(1-1/rho)}
// C3 toy (verify/c3_toy_grpo.py)
const TOY={CMAX:.9,L0:1e4,S:.5,MU0:Math.log(2000),CAP0:32768,CAP1:65536,J:96,steps:2000,lr:.02,beta:.001,ref:400};
const ZQ=new Float64Array(TOY.J),WQ=new Float64Array(TOY.J);for(let j=0;j<TOY.J;j++){const z=-6+12*(j+.5)/TOY.J;ZQ[j]=z;WQ[j]=Math.exp(-z*z/2)/Math.sqrt(2*Math.PI)*12/TOY.J}
const accOf=L=>TOY.CMAX*(1-Math.exp(-L/TOY.L0));
function toyExpected(mu,cap){let a=0,l=0,tr=0;for(let j=0;j<TOY.J;j++){const L=Math.exp(mu+TOY.S*ZQ[j]),w=WQ[j];if(L<=cap){a+=w*accOf(L);l+=w*L}else{l+=w*cap;tr+=w}}return[a,l,tr]}
const capAt=(step,S)=>(S===null||step<S)?TOY.CAP0:TOY.CAP1;
function toyRun(seed,o){const G=o.G,S=o.S,N=TOY.steps,s2=TOY.S*TOY.S,r=mulberry32(seed);let mu=TOY.MU0,muref=mu,zero=0;
  const out={mu:new Float64Array(N),acc:new Float64Array(N),len:new Float64Array(N),tr:new Float64Array(N),zero:0};const ys=new Float64Array(G),rs=new Float64Array(G);
  for(let step=0;step<N;step++){if(step%TOY.ref===0)muref=mu;const muold=mu,cap=capAt(step,S);
    for(let i=0;i<G;i++){const u1=r(),u2=r(),z=Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2),y=muold+TOY.S*z,L=Math.exp(y),u3=r();ys[i]=y;rs[i]=(L<=cap&&u3<accOf(L))?1:0}
    let m=0;for(let i=0;i<G;i++)m+=rs[i];m/=G;let v=0;for(let i=0;i<G;i++)v+=(rs[i]-m)**2;const sd=Math.sqrt(v/G);const ok=sd>1e-12;if(!ok)zero++;
    let g=0;for(let i=0;i<G;i++){const a=ok?(rs[i]-m)/sd:0,y=ys[i],rref=Math.exp(((y-mu)**2-(y-muref)**2)/(2*s2));g+=(a-TOY.beta*(1-rref))*(y-mu)/s2/G}
    mu+=TOY.lr*g;const e=toyExpected(mu,capAt(step+1,S));out.mu[step]=mu;out.acc[step]=e[0];out.len[step]=e[1];out.tr[step]=e[2]}
  out.zero=zero;return out}
function toyMean(runs,s){const i=s-1,R=runs.length;let mu=0,a=0,l=0,t=0;runs.forEach(x=>{mu+=x.mu[i];a+=x.acc[i];l+=x.len[i];t+=x.tr[i]});return{median:Math.exp(mu/R),acc:a/R,len:l/R,tr:t/R}}

/* ================= C1 ================= */
const c1={G:16,p:.25,f:1,ddof:0,seed:11,manual:false,acc:null};
function c1Sample(){const r=mulberry32(c1.seed),acc=[],fmt=[];for(let i=0;i<c1.G;i++){const ua=r(),uf=r();acc.push(ua<c1.p?1:0);fmt.push(uf<c1.f?1:0)}
  if(c1.manual&&c1.acc)for(let i=0;i<c1.G;i++)acc[i]=c1.acc[i];return{acc,fmt}}
function c1ExitManual(){if(!c1.manual)return;c1.manual=false;c1.acc=null;$('#c1-manual').setAttribute('aria-pressed','false');$('#c1-back').classList.add('hide');$('#c1-outs').classList.remove('manual')}
function c1Render(){const {acc,fmt}=c1Sample(),G=c1.G,r=acc.map((a,i)=>a+fmt[i]),q=groupAdv(r,c1.ddof),k=acc.reduce((a,b)=>a+b,0);
  const outs=$('#c1-outs');outs.textContent='';outs.className='outs'+(G>16?' dots':'')+(c1.manual?' manual':'');
  for(let i=0;i<G;i++){if(G<=16){const b=document.createElement('button');b.type='button';b.className='oc'+(acc[i]?' ok':'');b.dataset.i=i;b.disabled=!c1.manual;
      b.setAttribute('aria-label',`輸出 ${i+1}：${acc[i]?'正確':'錯誤'}${fmt[i]?'':'，格式失敗'}，優勢 ${f2(q.A[i])}`);
      b.innerHTML=`<span class="m">${acc[i]?'✓':'✗'}</span><span class="fx">${fmt[i]?'':'格式 ✗'}</span>`;outs.appendChild(b)}
    else{const d=document.createElement('span');d.className='dt'+(acc[i]?' ok':'')+(fmt[i]?'':' nf');outs.appendChild(d)}}
  // bars
  const box=$('#c1-bars'),W=widthOf(box,240,520),H=150,svg=frame(box,W,H);hatchDefs(svg);
  let amax=2;q.A.forEach(a=>amax=Math.max(amax,Math.abs(a)));amax=Math.ceil(amax);
  const y=lin(-amax,amax,H-16,10),bw=W/G;el('line',{x1:0,x2:W,y1:y(0),y2:y(0),class:'axl'},svg);
  tx(svg,2,y(amax)+10,'+'+amax,{class:'tk'});tx(svg,2,y(-amax)-2,'−'+amax,{class:'tk'});
  q.A.forEach((a,i)=>{const x0=i*bw+bw*.15,w=Math.max(1,bw*.7);if(a>=0)el('rect',{x:x0,y:y(a),width:w,height:Math.max(0,y(0)-y(a)),class:'bpos'},svg);else el('rect',{x:x0,y:y(0),width:w,height:y(a)-y(0),class:'bneg'},svg)});
  $('#c1-bars svg').setAttribute('aria-label',`${G} 個輸出中 ${k} 個正確；優勢最大 ${f2(Math.max(...q.A))}，最小 ${f2(Math.min(...q.A))}`);
  const fl=[];if(q.zero)fl.push('<b>本組沒有學習訊號</b>：所有輸出獎勵相同，優勢全部為 0。');if(c1.f===1)fl.push('所有輸出都拿到格式獎勵 1，等於加上常數，優勢不變。');$('#c1-flags').innerHTML=fl.join(' ');
  let sA=0;q.A.forEach(a=>sA+=a);
  $('#c1-read').innerHTML=`<dt>正確數 k</dt><dd>${k} / ${G}</dd><dt>mean</dt><dd>${f4(q.mean)}</dd><dt>std</dt><dd>${f4(q.sd)}</dd><dt>A 最大</dt><dd class="tgt">${f4(Math.max(...q.A))}</dd><dt>A 最小</dt><dd>${f4(Math.min(...q.A))}</dd><dt>ΣA</dt><dd>${Math.abs(sA)<1e-12?'0':sA.toExponential(1)}</dd><dt>P(全組相同)</dt><dd>${pEqual(c1.p,c1.f,G).toExponential(2)}</dd>`;
  c1Peq();
  announce('c1',q.zero?`${G} 個輸出獎勵全部相同，沒有學習訊號`:`${G} 個輸出中 ${k} 個正確；優勢最大 ${f2(Math.max(...q.A))}，最小 ${f2(Math.min(...q.A))}`)}
function c1Peq(){const box=$('#c1-peq'),W=widthOf(box,220,420),H=130,L=40,R=8,T=18,B=24,svg=frame(box,W,H),G=c1.G,f=c1.f;
  const x=lin(0,1,L,W-R),yl=lin(-20,0,H-B,T),y=v=>yl(Math.max(-20,Math.log10(Math.max(v,1e-300))));
  [-20,-10,0].forEach(e=>{el('line',{x1:L,x2:W-R,y1:yl(e),y2:yl(e),class:'grid'},svg);tx(svg,L-4,yl(e)+4,e===0?'1':'1e'+e,{class:'tk','text-anchor':'end'})});
  [0,.5,1].forEach(v=>tx(svg,x(v),H-6,String(v),{class:'tk','text-anchor':'middle'}));
  const pts=[];for(let i=0;i<=200;i++){const p=i/200;pts.push([x(p),y(pEqual(p,f,G))])}el('path',{d:lp(pts),class:'acc'},svg);
  const cv=pEqual(c1.p,f,G);el('circle',{cx:x(c1.p),cy:y(cv),r:4.5,class:'hollow'},svg);if(cv<1e-20)tx(svg,x(c1.p),H-B-4,'↓',{class:'labm','text-anchor':'middle'});
  tx(svg,L,12,f===1?'f = 1：p^G + (1−p)^G':'含格式獎勵的一般式（f < 1）',{class:'al'})}
function initC1(){const G=$('#c1-G'),p=$('#c1-p'),f=$('#c1-f');
  const syncOut=()=>{$('#c1-Go').textContent=c1.G;$('#c1-po').textContent=c1.p.toFixed(2);$('#c1-fo').textContent=c1.f.toFixed(2);
    G.setAttribute('aria-valuetext','G = '+c1.G);p.setAttribute('aria-valuetext','p = '+c1.p.toFixed(2));f.setAttribute('aria-valuetext','f = '+c1.f.toFixed(2));$('#c1-manual').disabled=c1.G>16};
  const onIn=()=>{c1ExitManual();c1.G=+G.value;c1.p=+p.value/100;c1.f=+f.value/20;syncOut();c1Render()};
  [G,p,f].forEach(s=>{s.addEventListener('input',onIn);pageKeys(s,.05)});
  segInit($('#c1-std'),[0,1],0,d=>d?'樣本（÷(G−1)）':'母體（÷G）',d=>{c1ExitManual();c1.ddof=d;c1Render()});
  $('#c1-resample').addEventListener('click',()=>{c1ExitManual();c1.seed++;c1Render()});
  $('#c1-manual').addEventListener('click',()=>{if(c1.G>16)return;if(c1.manual){c1ExitManual();c1Render();return}c1.acc=c1Sample().acc;c1.manual=true;$('#c1-manual').setAttribute('aria-pressed','true');$('#c1-back').classList.remove('hide');c1Render()});
  $('#c1-back').addEventListener('click',()=>{c1ExitManual();c1Render()});
  $('#c1-outs').addEventListener('click',e=>{const b=e.target.closest('button.oc');if(!b||!c1.manual)return;const i=+b.dataset.i;c1.acc[i]=1-c1.acc[i];c1Render();const nb=$(`#c1-outs button[data-i="${i}"]`);if(nb)nb.focus()});
  syncOut();c1Render()}

/* ================= C2 ================= */
const c2={A:1,eps:.2,beta:.001,x:.3};
function c2Plot(box,fn,label,cls,H){const W=widthOf(box,260,560),L=34,R=10,T=10,B=24,svg=frame(box,W,H);hatchDefs(svg);
  const x=lin(-3,3,L,W-R),y=lin(-4,4,H-B,T),yc=v=>y(Math.max(-4,Math.min(4,v)));
  // zero-gradient (clipped) region
  const {A,eps}=c2;if(A>0&&Math.log(1+eps)<3){const x0=Math.max(-3,Math.log(1+eps));el('rect',{x:x(x0),y:T,width:x(3)-x(x0),height:H-B-T,class:'clipz'},svg);tx(svg,(x(x0)+x(3))/2,T+14,'梯度 = 0',{class:'al halo','text-anchor':'middle'})}
  if(A<0&&1-eps>0&&Math.log(1-eps)>-3){const x1=Math.log(1-eps);el('rect',{x:x(-3),y:T,width:x(x1)-x(-3),height:H-B-T,class:'clipz'},svg);tx(svg,(x(-3)+x(x1))/2,T+14,'梯度 = 0',{class:'al halo','text-anchor':'middle'})}
  [-4,-2,0,2,4].forEach(v=>{el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),class:v===0?'axl':'grid'},svg);tx(svg,L-4,y(v)+4,String(v),{class:'tk','text-anchor':'end'})});
  [-3,-2,-1,0,1,2,3].forEach(v=>tx(svg,x(v),H-6,String(v),{class:'tk','text-anchor':'middle'}));
  [1+eps,1-eps].forEach(b=>{if(b>0){const lb=Math.log(b);if(lb>-3&&lb<3)el('line',{x1:x(lb),x2:x(lb),y1:T,y2:H-B,class:'ref'},svg)}});
  fn.forEach(([f,c])=>{const pts=[];let arrow=null;for(let i=0;i<=200;i++){const xv=-3+6*i/200,v=f(xv);pts.push([x(xv),yc(v)]);if(!arrow&&Math.abs(v)>4)arrow=[x(xv),v>0?'↑':'↓',v>0?T+10:H-B-2]}
    el('path',{d:lp(pts),class:c},svg);if(arrow)tx(svg,arrow[0],arrow[2],arrow[1],{class:'labm halo','text-anchor':'middle'})});
  el('line',{x1:x(c2.x),x2:x(c2.x),y1:T,y2:H-B,class:'cur'},svg);tx(svg,W-R,H-B-4,label,{class:'al halo','text-anchor':'end'});
  svg.addEventListener('click',e=>{const rc=svg.getBoundingClientRect(),xv=(e.clientX-rc.left)/rc.width*W;const v=Math.max(-3,Math.min(3,(xv-L)/(W-R-L)*6-3));$('#c2-x').value=Math.round(v*100);$('#c2-x').dispatchEvent(new Event('input'))});
  return svg}
function c2Render(){const {A,eps,beta}=c2;
  c2Plot($('#c2-top'),[[v=>Math.exp(v)*A,'ref'],[v=>{const r=Math.exp(v),c=Math.min(Math.max(r,1-eps),1+eps);return Math.min(r*A,c*A)},'tot'],[v=>-beta*(Math.exp(-v)+v-1),'perp dot'],[v=>c2J(v,A,eps,beta),'acc']],'J(x)','',210);
  c2Plot($('#c2-bot'),[[v=>c2dJ(v,A,eps,beta),'acc']],'dJ/dx','',150);
  const rho=Math.exp(c2.x),cl=!c2Active(rho,A,eps)&&A!==0,c=Math.min(Math.max(rho,1-eps),1+eps),S=Math.min(rho*A,c*A),D=1/rho+c2.x-1;
  $('#c2-read').innerHTML=`<dt>ρ</dt><dd>${f4(rho)}</dd><dt>裁剪後代理項</dt><dd>${f4(S)}</dd><dt>β·D</dt><dd>${(beta*D).toExponential(2)}</dd><dt>J</dt><dd class="tgt">${f4(c2J(c2.x,A,eps,beta))}</dd><dt>dJ/dx</dt><dd>${f4(c2dJ(c2.x,A,eps,beta))}</dd><dt>裁剪</dt><dd>${A===0?'A = 0':(cl?'是（梯度只剩 KL）':'否')}</dd>${1-eps<=0?'<dt>下界</dt><dd>1 − ε ≤ 0</dd>':''}`;
  announce('c2',`A = ${A.toFixed(2)}，ε = ${eps<1?eps.toFixed(3):eps.toFixed(2)}，x = ${c2.x.toFixed(2)}：${cl?'已裁剪，代理項梯度為 0':'未裁剪'}${1-eps<=0?'；下界 1 − ε ≤ 0，負優勢一側不會被裁剪':''}`)}
function initC2(){const A=$('#c2-A'),le=$('#c2-le'),xs=$('#c2-x');
  const chips=$$('#c2-echips .chip');const syncChips=()=>chips.forEach(c=>c.setAttribute('aria-pressed',String(+c.dataset.e===c2.eps)));
  const outs=()=>{$('#c2-Ao').textContent=c2.A.toFixed(2);$('#c2-eo').textContent=c2.eps<1?c2.eps.toFixed(3):c2.eps.toFixed(2);$('#c2-xo').textContent=c2.x.toFixed(2);
    A.setAttribute('aria-valuetext','A = '+c2.A.toFixed(2));le.setAttribute('aria-valuetext','ε = '+$('#c2-eo').textContent);xs.setAttribute('aria-valuetext','x = '+c2.x.toFixed(2));syncChips()};
  A.addEventListener('input',()=>{c2.A=+A.value/20;outs();c2Render()});
  le.addEventListener('input',()=>{let e=10**(+le.value/100);[.2,10].forEach(s=>{if(Math.abs(e-s)/s<=.02)e=s});c2.eps=e;outs();c2Render()});
  xs.addEventListener('input',()=>{c2.x=+xs.value/100;outs();c2Render()});
  chips.forEach(c=>c.addEventListener('click',()=>{c2.eps=+c.dataset.e;le.value=Math.round(Math.log10(c2.eps)*100);outs();c2Render()}));
  segInit($('#c2-beta'),[0,.001,.1],.001,b=>b===0?'0':b===.001?'0.001（論文）':'0.1（放大以便觀察）',b=>{c2.beta=b;c2Render()});
  [A,le].forEach(s=>pageKeys(s,.05));pageKeys(xs,1/60);
  outs();c2Render()}

/* ================= C3 ================= */
const C3SEEDS=[1,2,3,4,5,6,7,8],C3ROWS=[1,400,800,1200,1600,1700,2000];
const c3={G:16,S:1600,token:0,runs:null};
function c3Chart(box,key,ylo,yhi,refs,fmtY,label){const W=widthOf(box,260,520),H=210,L=50,R=10,T=12,B=26,svg=frame(box,W,H);
  const x=lin(0,TOY.steps,L,W-R),y=lin(ylo,yhi,H-B,T);
  for(let i=0;i<=4;i++){const v=ylo+(yhi-ylo)*i/4;el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-4,y(v)+4,fmtY(v),{class:'tk','text-anchor':'end'})}
  [0,1000,2000].forEach(v=>tx(svg,x(v),H-8,String(v),{class:'tk','text-anchor':'middle'}));
  refs.forEach(([v,t])=>{el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),class:'ref'},svg);tx(svg,W-R-2,y(v)-4,t,{class:'tk halo','text-anchor':'end'})});
  if(c3.S!==null&&c3.S>0){el('line',{x1:x(c3.S),x2:x(c3.S),y1:T,y2:H-B,class:'cur'},svg);tx(svg,x(c3.S)+4,H-B-6,'上限 32,768 → 65,536',{class:'tk halo','text-anchor':x(c3.S)>W*.6?'end':'start',dx:x(c3.S)>W*.6?-8:0})}
  tx(svg,L,T+2,label,{class:'al halo',dy:10,dx:6});
  return {svg,x,y}}
function c3Draw(upto){const runs=c3.runs;if(!runs)return;const n=Math.max(1,Math.min(TOY.steps,upto));
  const a=c3Chart($('#c3-acc'),'acc',.1,.9,[[.6396,'上限 32,768 最佳'],[.7808,'上限 65,536 最佳']],v=>v.toFixed(2),'期望正確率');
  const b=c3Chart($('#c3-len'),'len',0,36000,[[18357,'上限 32,768 最佳'],[31026,'上限 65,536 最佳']],v=>(v/1000).toFixed(0)+'k','期望長度（tokens）');
  [[a,'acc'],[b,'len']].forEach(([c,k])=>{runs.forEach(rn=>{const pts=[];for(let i=0;i<n;i+=10)pts.push([c.x(i+1),c.y(rn[k][i])]);el('path',{d:lp(pts),class:'ink thin'},c.svg)});
    const pts=[];for(let i=0;i<n;i+=5){let s=0;runs.forEach(rn=>s+=rn[k][i]);pts.push([c.x(i+1),c.y(s/runs.length)])}el('path',{d:lp(pts),class:'acc'},c.svg)})}
function c3Finish(){const runs=c3.runs,m=toyMean(runs,TOY.steps),zf=runs.reduce((s,r)=>s+r.zero,0)/runs.length/TOY.steps;
  $('#c3-read').innerHTML=`<dt>第 2000 步平均正確率</dt><dd class="tgt">${m.acc.toFixed(3)}</dd><dt>平均期望長度</dt><dd>${fmtInt(m.len)}</dd><dt>中位數長度</dt><dd>${fmtInt(m.median)}</dd><dt>截斷比例</dt><dd>${m.tr.toFixed(3)}</dd><dt>無訊號群組比例</dt><dd>${zf.toFixed(3)}</dd>`;
  const rows=C3ROWS.slice();if(c3.S!==null&&c3.S>0&&!rows.includes(c3.S+100))rows.push(c3.S+100);rows.sort((p,q)=>p-q);
  $('#c3-tab tbody').innerHTML=rows.map(s=>{const t=toyMean(runs,s);return`<tr><td class="num">${s}</td><td class="num">${fmtInt(t.median)}</td><td class="num">${t.acc.toFixed(3)}</td><td class="num">${fmtInt(t.len)}</td><td class="num">${t.tr.toFixed(3)}</td></tr>`}).join('');
  $('#c3-prog').textContent='';announce('c3',`第 2000 步，平均正確率 ${m.acc.toFixed(3)}，平均長度 ${fmtInt(m.len)}`)}
function c3Run(){const tok=++c3.token,runs=[],idle=window.requestIdleCallback?(f=>requestIdleCallback(f,{timeout:200})):(f=>setTimeout(f,0));
  $('#c3-prog').textContent='計算中（0 / 8）';$('#c3-read').innerHTML='<dt>狀態</dt><dd>計算中</dd>';
  const next=()=>{if(tok!==c3.token)return;runs.push(toyRun(C3SEEDS[runs.length],{G:c3.G,S:c3.S}));$('#c3-prog').textContent=`計算中（${runs.length} / 8）`;
    if(runs.length<C3SEEDS.length){idle(next);return}
    c3.runs=runs;c3Draw(TOY.steps);c3Finish()};
  idle(next)}
function initC3(){segInit($('#c3-G'),[4,16,32],16,null,g=>{c3.G=g;c3Run()});
  segInit($('#c3-S'),['0','400','1000','1600','none'],'1600',s=>s==='none'?'不調高':s,s=>{c3.S=s==='none'?null:+s;c3Run()});
  c3Run()}

/* ================= C4 ================= */
const STAGE_NAMES=['R1-Zero','Dev1','Dev2','Dev3','R1'];
const STAGES=[{start:'DeepSeek-V3-Base',data:'推理題；規則式準確度 + 格式獎勵（直接 RL，不經 SFT）',base:null,cross:false,note:'R1-Zero 的推理強，但可讀性差、語言混雜（Sec. 3）。'},
  {start:'DeepSeek-V3-Base',data:'數千筆冷啟動長 CoT（以 R1-Zero 產生、過濾並改寫）做 SFT',base:0,cross:true,note:'指令遵循與偏好評測上升；推理退步，作者認為是冷啟動資料量小所致（Sec. 4）。'},
  {start:'Dev1',data:'推理題 RL；規則獎勵 + 語言一致性獎勵（Eq. 7），ε = 10',base:1,cross:false,note:'推理類評測明顯提升，AlpacaEval 只小幅提升（Sec. 4）。'},
  {start:'DeepSeek-V3-Base',data:'Dev2 拒絕取樣的約 60 萬筆推理 + 約 20 萬筆非推理資料做 SFT',base:2,cross:true,note:'加入非推理資料，AlpacaEval 2.0 與 Aider-Polyglot 提升（Sec. 4）。'},
  {start:'Dev3',data:'推理題用規則獎勵；通用題用獎勵模型 + 格式；加語言獎勵（Eq. 8–10）；溫度 0.7，1,700 步',base:3,cross:false,note:'主要提升通用指令與偏好評測；程式與數學只小幅提升（Sec. 4）。'}];
const T3=[['MMLU','English',[88.8,89.1,91.2,91.0,90.8]],['MMLU-Redux','English',[85.6,90.0,93.0,93.1,92.9]],['MMLU-Pro','English',[68.9,74.1,83.8,83.1,84.0]],['DROP','English',[89.1,89.8,91.1,88.7,92.2]],
 ['IF-Eval','English',[46.6,71.7,72.0,78.1,83.3]],['GPQA Diamond','English',[75.8,66.1,70.7,71.2,71.5]],['SimpleQA','English',[30.3,17.8,28.2,24.9,30.1]],['FRAMES','English',[82.3,78.5,81.8,81.9,82.5]],
 ['AlpacaEval2.0','English',[24.7,50.1,55.8,62.1,87.6]],['ArenaHard','English',[53.6,77.0,73.2,75.6,92.3]],['LiveCodeBench','Code',[50.0,57.5,63.5,64.6,65.9]],['Codeforces 百分位','Code',[80.4,84.5,90.5,92.1,96.3]],
 ['Codeforces Rating','Code',[1444,1534,1687,1746,2029]],['SWE Verified','Code',[43.2,39.6,44.6,45.6,49.2]],['Aider-Polyglot','Code',[12.2,6.7,25.6,44.8,53.3]],['AIME 2024','Math',[77.9,59.0,74.0,78.1,79.8]],
 ['MATH-500','Math',[95.9,94.2,95.9,95.4,97.3]],['CNMO 2024','Math',[88.1,58.0,73.9,77.3,78.8]],['CLUEWSC','Chinese',[93.1,92.8,92.6,91.6,92.8]],['C-Eval','Chinese',[92.8,85.7,91.9,86.4,91.8]],['C-SimpleQA','Chinese',[66.4,58.8,64.2,66.9,63.7]]];
const c4={node:4,sel:['AIME 2024','CNMO 2024','IF-Eval','AlpacaEval2.0','Aider-Polyglot']};
function stageDelta(n){const b=STAGES[n].base;if(b===null)return null;const d=T3.filter(r=>r[0]!=='Codeforces Rating').map(r=>[r[0],Math.round(r[2][n]*10)-Math.round(r[2][b]*10)]);
  const up=d.filter(x=>x[1]>=10).sort((p,q)=>q[1]-p[1]).slice(0,3),down=d.filter(x=>x[1]<=-10).sort((p,q)=>p[1]-q[1]).slice(0,3);return{up,down}}
const sgn=t=>(t>0?'+':'−')+(Math.abs(t)/10).toFixed(1);
function c4Card(){const n=c4.node,s=STAGES[n],d=stageDelta(n),cf=T3.find(r=>r[0]==='Codeforces Rating')[2];
  let h=`<h4>${STAGE_NAMES[n]}</h4><div>起點：${s.start}</div><div>資料與獎勵：${s.data}</div>`;
  if(!d)h+=`<div>比較對象：無（起點 DeepSeek-V3-Base 直接 RL；Table 3 沒有 V3-Base 欄）</div>`;
  else{h+=`<div>比較對象：${STAGE_NAMES[s.base]}${s.cross?` <span class="x">跨線比較：${n===1?'Dev1 不是由 R1-Zero 訓練而來':'Dev2 只提供資料'}</span>`:'（同一條訓練線）'}</div>`;
    const lst=(t,xs)=>xs.length===0?`<div>${t}：無（沒有 |Δ| ≥ 1.0 的評測）</div>`:xs.length===1?`<div>${t}：僅 ${xs[0][0]} ${sgn(xs[0][1])}</div>`:`<div>${t}（前 ${xs.length} 名，|Δ| ≥ 1.0）：</div><ul>${xs.map(x=>`<li>${x[0]} ${sgn(x[1])}</li>`).join('')}</ul>`;
    h+=lst('上升',d.up)+lst('下降',d.down);
    h+=`<div class="cf"><span class="x">另列（Elo 分數，不列入上方排名）</span>Codeforces Rating：${cf[s.base]} → ${cf[n]}（${cf[n]>cf[s.base]?'上升':'下降'}）</div>`}
  h+=`<div>論文敘述：${s.note}</div>`;$('#c4-card').innerHTML=h;
  announce('c4',`${STAGE_NAMES[n]}${d?`，與 ${STAGE_NAMES[s.base]} 比較${s.cross?'（跨線）':''}：上升最多 ${d.up[0]?d.up[0][0]+' '+sgn(d.up[0][1]):'無'}`:'：起點模型'}`)}
const SHAPES=['circle','square','tri','dia','cross','plus','ring','tri2'],SCOL=['--accent','--c-par','--c-perp','--c-3','--c-4','--ink','--ink-2','--ink-3'];
function marker(svg,s,x,y,col){const st=`stroke:var(${col});fill:var(${col})`;if(s==='circle')el('circle',{cx:x,cy:y,r:4,style:st},svg);else if(s==='ring')el('circle',{cx:x,cy:y,r:4,style:`stroke:var(${col});fill:var(--surface);stroke-width:2`},svg);
  else if(s==='square')el('rect',{x:x-4,y:y-4,width:8,height:8,style:st},svg);else if(s==='tri')el('path',{d:`M${x} ${y-5}L${x+5} ${y+4}L${x-5} ${y+4}Z`,style:st},svg);else if(s==='tri2')el('path',{d:`M${x} ${y+5}L${x+5} ${y-4}L${x-5} ${y-4}Z`,style:st},svg);
  else if(s==='dia')el('path',{d:`M${x} ${y-5}L${x+5} ${y}L${x} ${y+5}L${x-5} ${y}Z`,style:st},svg);else if(s==='cross')el('path',{d:`M${x-4} ${y-4}L${x+4} ${y+4}M${x+4} ${y-4}L${x-4} ${y+4}`,style:`stroke:var(${col});stroke-width:2.2`},svg);
  else el('path',{d:`M${x-5} ${y}L${x+5} ${y}M${x} ${y-5}L${x} ${y+5}`,style:`stroke:var(${col});stroke-width:2.2`},svg)}
function c4Slope(){const box=$('#c4-slope'),W=widthOf(box,280,560),narrow=W<440,H=280,L=34,R=narrow?92:120,T=14,B=30,svg=frame(box,W,H);
  const x=i=>L+(W-L-R)*i/4,y=lin(0,100,H-B,T);
  [0,25,50,75,100].forEach(v=>{el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-4,y(v)+4,String(v),{class:'tk','text-anchor':'end'})});
  STAGE_NAMES.forEach((s,i)=>{tx(svg,x(i),H-12,narrow?s.replace('R1-Zero','Zero'):s,{class:'tk','text-anchor':'middle','font-weight':i===c4.node?600:400});if(i===c4.node)el('line',{x1:x(i),x2:x(i),y1:T,y2:H-B,class:'cur'},svg)});
  [[0,1],[2,3]].forEach(([a,b])=>tx(svg,(x(a)+x(b))/2,H-1,'跨線',{class:'tk','text-anchor':'middle'}));
  const labs=[];
  c4.sel.forEach((name,k)=>{const r=T3.find(t=>t[0]===name);if(!r)return;const col=SCOL[k%8],v=r[2];
    for(let i=0;i<4;i++){const seg=el('line',{x1:x(i),y1:y(v[i]),x2:x(i+1),y2:y(v[i+1]),style:`stroke:var(${col});stroke-width:2`},svg);if(i===0||i===2)seg.setAttribute('stroke-dasharray','5 4')}
    v.forEach((u,i)=>marker(svg,SHAPES[k%8],x(i),y(u),col));labs.push({y:y(v[4]),name,k})});
  labs.sort((a,b)=>a.y-b.y);for(let i=1;i<labs.length;i++)if(labs[i].y-labs[i-1].y<14)labs[i].y=labs[i-1].y+14;
  const over=labs.length?labs[labs.length-1].y-(H-B):0;if(over>0)labs.forEach(l=>l.y-=over);
  labs.forEach(l=>{const r=T3.find(t=>t[0]===l.name),y0=y(r[2][4]);if(Math.abs(l.y-y0)>1)el('line',{x1:x(4)+6,y1:y0,x2:x(4)+14,y2:l.y,class:'proj'},svg);tx(svg,x(4)+16,l.y+4,l.name,{class:'tk'})});
  svg.setAttribute('aria-label','選取評測在五個模型的分數：'+c4.sel.map(n=>{const r=T3.find(t=>t[0]===n);return`${n} ${r[2].join('、')}`}).join('；'))}
function c4Chips(){const box=$('#c4-chips');box.textContent='';['English','Code','Math','Chinese'].forEach(cat=>{const g=document.createElement('span');g.className='grp';g.textContent=cat;box.appendChild(g);
  T3.filter(r=>r[1]===cat&&r[0]!=='Codeforces Rating').forEach(r=>{const b=document.createElement('button');b.type='button';b.className='chip';b.textContent=r[0];b.dataset.n=r[0];box.appendChild(b)})});
  const sync=()=>$$('.chip',box).forEach(b=>{const on=c4.sel.includes(b.dataset.n);b.setAttribute('aria-pressed',String(on));b.setAttribute('aria-disabled',String(!on&&c4.sel.length>=8))});sync();
  box.addEventListener('click',e=>{const b=e.target.closest('.chip');if(!b)return;const n=b.dataset.n,i=c4.sel.indexOf(n);if(i>=0)c4.sel.splice(i,1);else if(c4.sel.length<8)c4.sel.push(n);else return;sync();c4Slope()})}
function initC4(){const nodes=$$('#c4-lin button.mnode');const sync=()=>nodes.forEach(b=>{const on=+b.dataset.n===c4.node;b.setAttribute('aria-checked',String(on));b.tabIndex=on?0:-1});
  const pick=n=>{c4.node=n;sync();c4Card();c4Slope()};
  nodes.forEach(b=>b.addEventListener('click',()=>pick(+b.dataset.n)));
  $('#c4-lin').addEventListener('keydown',e=>{let d=0;if(e.key==='ArrowRight'||e.key==='ArrowDown')d=1;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')d=-1;else return;e.preventDefault();const n=(c4.node+d+5)%5;pick(n);$(`#c4-lin button[data-n="${n}"]`).focus()});
  $('#c4-full').innerHTML='<table><caption>Table 3（各階段結果）</caption><thead><tr><th>評測</th>'+STAGE_NAMES.map(s=>`<th class="num">${s}</th>`).join('')+'</tr></thead><tbody>'+T3.map(r=>`<tr><td>${r[0]}</td>${r[2].map(v=>`<td class="num">${v}</td>`).join('')}</tr>`).join('')+'</tbody></table>';
  c4Chips();sync();c4Card();c4Slope()}

/* ================= equations, timeline, toc, theme ================= */
function initEqVersions(){const mq=matchMedia('(max-width:599px)');const apply=()=>{$$('.wide-eq').forEach(e=>e.setAttribute('aria-hidden',String(mq.matches)));$$('.narrow-eq').forEach(e=>e.setAttribute('aria-hidden',String(!mq.matches)))};apply();mq.addEventListener('change',apply)}
const TL_GROUP={'2201.11903':['cot','r1-cot-and-test-time-compute'],'2110.14168':['cot','r1-cot-and-test-time-compute'],'2408.03314':['cot','r1-cot-and-test-time-compute'],'2001.08361':['cot','r1-cot-and-test-time-compute'],
 '2203.02155':['rlhf','r1-rlhf-ppo-background'],'1707.06347':['rlhf','r1-rlhf-ppo-background'],'1506.02438':['rlhf','r1-rlhf-ppo-background'],
 '2203.14465':['star','r1-star-self-generated-reasoning'],'2211.14275':['prm','r1-process-supervision-prm'],'2305.20050':['prm','r1-process-supervision-prm'],'2312.08935':['prm','r1-process-supervision-prm'],
 '2402.03300':['grpo','r1-grpo-outcome-rewards'],'1503.02531':['kd','r1-distillation-to-small-models'],'2501.12948':['me','r1-lineage-hub']};
const TL_NAME={cot:'思維鏈與推論時運算',rlhf:'RLHF 與 PPO',star:'自產推理',prm:'過程監督',grpo:'GRPO',kd:'蒸餾',me:'本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.map((e,i)=>[e,i]).sort((a,b)=>(Number(a[0].year)||0)-(Number(b[0].year)||0)||a[1]-b[1]).map(p=>p[0]);
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];if(!TL_GROUP[e.arxiv])console.warn('timeline: ungrouped',e.arxiv);used.add(g[0]);const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${g[1]?`<a href="${g[1]}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
  const on=new Set(used);
  Object.keys(TL_NAME).filter(k=>used.has(k)).forEach(k=>{const b=document.createElement('button');b.type='button';b.className='chip';b.setAttribute('aria-pressed','true');b.textContent=TL_NAME[k];b.dataset.g=k;
    b.addEventListener('click',()=>{if(on.has(k)&&on.size===used.size){on.clear();on.add(k)}else if(on.has(k)){on.delete(k);if(!on.size)used.forEach(x=>on.add(x))}else on.add(k);
      $$('.chip',chips).forEach(c=>c.setAttribute('aria-pressed',String(on.has(c.dataset.g))));$$('.ev',track).forEach(ev=>ev.classList.toggle('dim',!on.has(ev.dataset.g)))});chips.appendChild(b)});
  const rail=$('#rail'),by=dir=>rail.scrollBy({left:dir*Math.min(rail.clientWidth*.8,560),behavior:RM?'auto':'smooth'});
  $('#tl-prev').addEventListener('click',()=>by(-1));$('#tl-next').addEventListener('click',()=>by(1));
  rail.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){by(1);e.preventDefault()}if(e.key==='ArrowLeft'){by(-1);e.preventDefault()}})}
function initToc(){const secs=$$('[data-toc]'),ol=$('#toc'),bar=$('#tocbar');
  secs.forEach(s=>{ol.insertAdjacentHTML('beforeend',`<li><a href="#${s.id}"><span class="n">${s.dataset.toc}</span><span>${s.dataset.tt}</span></a></li>`);bar.insertAdjacentHTML('beforeend',`<a href="#${s.id}">${s.dataset.toc} ${s.dataset.tt}</a>`)});
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;$$('.toc a,.tocbar a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+id));const ab=$(`.tocbar a[href="#${id}"]`);if(ab&&bar.offsetParent)bar.scrollTo({left:ab.offsetLeft-16,behavior:RM?'auto':'smooth'})}})},{rootMargin:'-40% 0px -55% 0px'});
  secs.forEach(s=>io.observe(s));addEventListener('scroll',()=>$('#appbar').classList.toggle('raised',scrollY>8),{passive:true})}
function initTheme(){let saved=null;try{saved=localStorage.getItem('pm-theme')}catch(e){}if(saved)document.documentElement.dataset.theme=saved;
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){}})}

/* ================= self test (design.md section 5 / 6.2) ================= */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t=1e-4)=>Math.abs(a-b)<=t+1e-12;
  {const r=mulberry32(7);ok([.0117047532,.0619582576,.9769076328].every(v=>near(r(),v,1e-10)),'mulberry32(7)')}
  // C1
  {const r=[1,1,1,1].concat(Array(12).fill(0)),q=groupAdv(r,0);ok(near(q.A[0],1.7321)&&near(q.A[15],-.5774)&&near(q.mean,.25)&&near(q.sd,.4330),'groupAdv G16 k4');
   const q1=groupAdv(r,1);ok(near(q1.A[0],1.6771)&&near(q1.A[15],-.5590),'groupAdv ddof1');
   ok(groupAdv(Array(16).fill(1),0).A.every(a=>a===0)&&groupAdv(Array(16).fill(0),0).zero,'all equal -> zero');
   const qc=groupAdv(r.map(v=>v+1),0);ok(qc.A.every((a,i)=>near(a,q.A[i],1e-9)),'constant shift invariant');
   const qm=groupAdv([2,1,1,1],0);ok(near(qm.A[0],1.7321)&&near(qm.A[1],-.5774)&&near(qm.mean,1.25),'mixed reward');
   ok(near(groupAdv([1].concat(Array(15).fill(0))).A[0],3.8730)&&near(groupAdv([1].concat(Array(15).fill(0))).A[1],-.2582),'G16 k1');
   {const a=[1,1,1,1,1,1,1,1].concat(Array(56).fill(0)),g=groupAdv(a);ok(near(g.A[0],2.6458)&&near(g.A[63],-.3780),'G64 k8')}
   ok(Math.abs(pEqual(.25,1,16)/1.0023e-2-1)<1e-4&&Math.abs(pEqual(.25,.9,16)/3.3233e-3-1)<1e-4,'pEqual');
   ok(near(pEqual(.05,1,16),.4401)&&near(pEqual(.9,1,16),.1853)&&near(pEqual(.99,1,64),.5256),'P0 at f=1')}
  // C2
  {const T=[[0,1,.2,.001,1,1],[.5,1,.2,.001,1.19989,-.00039],[.5,1,10,.001,1.64861,1.64833],[-.5,-1,.2,.001,-.80015,.00065],[-.5,-1,10,.001,-.60668,-.60588],[.5,-1,.2,.001,-1.64883,-1.64911],[1,1,.2,.1,1.16321,-.06321],[-1,1,.2,.1,.29605,.53971],[2.5,1,10,.001,10.99842,-.00092]];
   T.forEach(([x,A,e,b,J,d])=>ok(near(c2J(x,A,e,b),J,1e-5)&&near(c2dJ(x,A,e,b),d,1e-5),`C2 x=${x} A=${A} e=${e} b=${b}`))}
  // C3
  {const e=toyExpected(Math.log(2000),32768);ok(near(e[0],.1776),'toyExpected acc '+e[0]);
   const runs=C3SEEDS.map(s=>toyRun(s,{G:16,S:1600}));const m1=toyMean(runs,1200),m2=toyMean(runs,2000),m1600=toyMean(runs,1600),m1700=toyMean(runs,1700);
   ok(near(m1.acc,.629,.03)&&near(m2.acc,.776,.03)&&Math.abs(m2.median/28331-1)<.1,`C3 default ${m1.acc.toFixed(3)} ${m2.acc.toFixed(3)} ${Math.round(m2.median)}`);
   ok(near(m1600.acc,.703,.03)&&near(m1700.acc,.773,.03)&&Math.abs(m1700.median/26268-1)<.1,`C3 jump ${m1600.acc.toFixed(3)} ${m1700.acc.toFixed(3)} ${Math.round(m1700.median)}`);
   const nv=C3SEEDS.map(s=>toyRun(s,{G:16,S:null}));ok(near(toyMean(nv,2000).acc,.630,.03),'C3 never raised '+toyMean(nv,2000).acc.toFixed(3));
   const s1=C3SEEDS.map(s=>toyRun(s,{G:16,S:1000}));ok(Math.abs(toyMean(s1,1100).median/26423-1)<.1&&near(toyMean(s1,1100).acc,.774,.03),'C3 S=1000 +100');
   const g4=C3SEEDS.map(s=>toyRun(s,{G:4,S:1600}));const zf=g4.reduce((a,r)=>a+r.zero,0)/8/TOY.steps;ok(near(zf,.225,.03),'C3 G=4 zero frac '+zf.toFixed(3))}
  // C4
  {const d1=stageDelta(1);ok(d1.up.map(x=>x[0]+x[1]).join()==='AlpacaEval2.0254,IF-Eval251,ArenaHard234'&&d1.down.map(x=>x[0]+x[1]).join()==='CNMO 2024-301,AIME 2024-189,SimpleQA-125','C4 Dev1');
   const d4=stageDelta(4);ok(d4.down.length===1&&d4.down[0][0]==='C-SimpleQA'&&d4.down[0][1]===-32&&d4.up[0][1]===255&&d4.up[1][1]===167,'C4 R1');
   const d2=stageDelta(2);ok(d2.down.length===1&&d2.down[0][1]===-38,'C4 Dev2');const d3=stageDelta(3);ok(d3.down.map(x=>x[1]).join()==='-55,-33,-24','C4 Dev3')}
  // DOM
  ok($$('details.check').length===14,'check boxes: '+$$('details.check').length);
  ok($$('.wide-eq,.narrow-eq').every(e=>e.hasAttribute('aria-hidden')),'equation copies aria-hidden');
  {const tl=JSON.parse($('#timeline-data').textContent);ok(tl.length===14&&tl.every(e=>TL_GROUP[e.arxiv]),'timeline groups')}
  ok(!!document.querySelector('.katex')||document.documentElement.classList.contains('nokatex'),'KaTeX rendered');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'')}

/* ================= boot ================= */
initTheme();initToc();initEqVersions();initC1();initC2();initC3();initC4();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{c1Render();c2Render();c3Draw(TOY.steps);c4Slope()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath();if(/[?&]test\b/.test(location.search))setTimeout(selfTest,50)});
})();
