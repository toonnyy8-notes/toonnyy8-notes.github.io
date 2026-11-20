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
function frame(box,min,max,H){const W=widthOf(box,min,max),svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);return{svg,W,H}}
const pct=v=>(100*v).toFixed(1)+'%';
const f4=v=>(Math.abs(v)<5e-5?0:v).toFixed(4).replace('-','−');
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
const lbl={};function setLabel(svg,id,s){clearTimeout(lbl[id]);lbl[id]=setTimeout(()=>svg.setAttribute('aria-label',s),500)}
function segInit(box,opts,cur,label,onPick){box.textContent='';opts.forEach(o=>{const b=document.createElement('button');b.type='button';b.setAttribute('role','radio');
    b.textContent=label?label(o):String(o);b.dataset.v=String(o);box.appendChild(b)});
  const sync=v=>$$('button',box).forEach(b=>{const on=b.dataset.v===String(v);b.setAttribute('aria-checked',String(on));b.setAttribute('aria-pressed',String(on));b.tabIndex=on?0:-1});sync(cur);
  box.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;const v=opts.find(o=>String(o)===b.dataset.v);sync(v);onPick(v)});
  box.addEventListener('keydown',e=>{const bs=$$('button',box),i=bs.findIndex(b=>b.tabIndex===0);let j=i;if(e.key==='ArrowRight'||e.key==='ArrowDown')j=(i+1)%bs.length;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')j=(i-1+bs.length)%bs.length;else return;e.preventDefault();bs[j].focus();bs[j].click()});
  return sync}
function dl(box,rows){box.innerHTML=rows.map(([k,v,c])=>`<dt>${k}<span class="sr">：</span></dt><dd${c?' class="acc"':''}>${v}${c?' <span class="nr">非論文結果</span>':''}<span class="sr">；</span></dd>`).join('')}
function axisLeft(svg,y,x0,ticks,fmt){ticks.forEach(t=>{el('line',{x1:x0,x2:x0-4,y1:y(t),y2:y(t),class:'axis'},svg);tx(svg,x0-7,y(t)+4,fmt(t),{class:'lab','text-anchor':'end'})})}

/* ================= shared math (mirrors verify/check.py) ================= */
function erfc(x){const z=Math.abs(x),t=1/(1+.5*z);
  const a=t*Math.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+t*(.09678418+t*(-.18628806+t*(.27886807+t*(-1.13520398+t*(1.48851587+t*(-.82215223+t*.17087277)))))))));
  return x>=0?a:2-a}
const Phi=s=>.5*erfc(-s/Math.SQRT2),phi=s=>Math.exp(-.5*s*s)/Math.sqrt(2*Math.PI);
function binomPmf(N,p){const out=[(1-p)**N];for(let c=1;c<=N;c++)out.push(out[c-1]*(N-c+1)/c*p/(1-p));return out}
function comb(n,k){let r=1;for(let i=1;i<=k;i++)r=r*(n-k+i)/i;return r}

/* ---------------- C1: Best-of-N with Gaussian scores (check.py C1) ---------------- */
const NS1=[1,2,4,8,16,32,64],H1=.01,arrCache=new Map();
function c1Arrays(mu10){if(arrCache.has(mu10))return arrCache.get(mu10);const mu=mu10/10,lo=Math.min(0,mu)-8,hi=Math.max(0,mu)+8,n=Math.round((hi-lo)/H1);
  const f=new Float64Array(n+1),lc=new Float64Array(n+1),li=new Float64Array(n+1),w=new Float64Array(n+1);
  for(let i=0;i<=n;i++){const s=lo+i*H1;w[i]=(i===0||i===n)?.5:1;f[i]=phi(s-mu);lc[i]=Math.log(Math.max(Phi(s-mu),1e-300));li[i]=Math.log(Math.max(Phi(s),1e-300))}
  const A={f,lc,li,w,n};arrCache.set(mu10,A);return A}
function win(c,m,A){if(c===0)return 0;if(m===0)return 1;let acc=0;for(let i=0;i<=A.n;i++)acc+=A.w[i]*c*A.f[i]*Math.exp((c-1)*A.lc[i]+m*A.li[i]);return acc*H1}
function bestOfN(p,N,mu10){const A=c1Arrays(mu10),pmf=binomPmf(N,p);let s=0;for(let c=0;c<=N;c++)s+=pmf[c]*win(c,N-c,A);return s}
const oracle=(p,N)=>1-(1-p)**N,rmAcc=mu=>Phi(mu/2);
const c1={p20:6,mu10:15,N:16},c1Cache=new Map();
function c1Curve(){const key=c1.p20+'|'+c1.mu10;if(!c1Cache.has(key)){const p=c1.p20/20;c1Cache.set(key,NS1.map(N=>bestOfN(p,N,c1.mu10)))}return c1Cache.get(key)}
function renderC1(){const p=c1.p20/20,mu=c1.mu10/10,bon=c1Curve(),ix=NS1.indexOf(c1.N);
  $('#c1-po').textContent=p.toFixed(2);$('#c1-mu'+'o').textContent=mu.toFixed(1);
  $('#c1-p').setAttribute('aria-valuetext','p = '+p.toFixed(2));$('#c1-mu').setAttribute('aria-valuetext','μ = '+mu.toFixed(1));
  {const {svg,W,H}=frame($('#c1-main'),300,620,230),L=44,R=W-14,T=14,B=H-30,x=lin(0,6,L,R),y=lin(0,1,B,T),narrow=W<480;
   [0,.25,.5,.75,1].forEach(t=>el('line',{x1:L,x2:R,y1:y(t),y2:y(t),class:'grid'},svg));axisLeft(svg,y,L,[0,.5,1],t=>(100*t)+'%');
   NS1.forEach((N,i)=>{if(narrow&&![1,4,16,64].includes(N))return;tx(svg,x(i),B+18,'N='+N,{class:'lab','text-anchor':'middle'})});
   el('path',{d:lp([[x(0),y(p)],[x(6),y(p)]]),class:'ref2-s'},svg);
   el('path',{d:lp(NS1.map((N,i)=>[x(i),y(oracle(p,N))])),class:'ref-s'},svg);
   el('path',{d:lp(bon.map((v,i)=>[x(i),y(v)])),class:'acc-s'},svg);bon.forEach((v,i)=>el('circle',{cx:x(i),cy:y(v),r:3.5,class:'acc-f'},svg));
   el('line',{x1:x(ix),x2:x(ix),y1:T,y2:B,class:'axis','stroke-dasharray':'2 3'},svg);
   const lx=x(ix)+(ix>3?-6:6),an=ix>3?'end':'start';tx(svg,lx,y(bon[ix])-8,'Best-of-'+c1.N+' '+pct(bon[ix]),{class:'lab13 halo','text-anchor':an});
   setLabel(svg,'c1m',`Best-of-N 曲線：N = ${c1.N} 時 ${pct(bon[ix])}，Pass@N ${pct(oracle(p,c1.N))}，Pass@1 ${pct(p)}`)}
  {const {svg,W,H}=frame($('#c1-dens'),300,620,130),L=14,R=W-14,T=12,B=H-24,lo=-4,hi=Math.max(4,mu+4),x=lin(lo,hi,L,R),y=lin(0,.42,B,T);
   const pts=g=>{const a=[];for(let i=0;i<=120;i++){const s=lo+(hi-lo)*i/120;a.push([x(s),y(phi(s-g))])}return a};
   el('line',{x1:L,x2:R,y1:B,y2:B,class:'axis'},svg);el('path',{d:lp(pts(0)),class:'neg-s'},svg);el('path',{d:lp(pts(mu)),class:'pos-s'},svg);
   el('line',{x1:x(mu/2),x2:x(mu/2),y1:T,y2:B,class:'axis','stroke-dasharray':'2 3'},svg);tx(svg,x(mu/2),B+16,'閾值 μ/2',{class:'lab','text-anchor':'middle'});
   tx(svg,x(0),y(.4)-2,'錯誤解',{class:'lab','text-anchor':'middle'});if(mu>=.8)tx(svg,x(mu),y(.4)-2,'正確解',{class:'lab','text-anchor':'middle'});
   setLabel(svg,'c1d',`分數分布：正確解平均比錯誤解高 ${mu.toFixed(1)}`)}
  dl($('#c1-read'),[['Pass@1',pct(p)],['Best-of-1',pct(bon[0])],['Best-of-'+c1.N,pct(bon[ix]),1],['Pass@'+c1.N,pct(oracle(p,c1.N))],['RM accuracy',pct(rmAcc(mu))]]);
  announce('c1',`p = ${p.toFixed(2)}，μ = ${mu.toFixed(1)}，Best-of-${c1.N} ${pct(bon[ix])}，RM accuracy ${pct(rmAcc(mu))}`)}
function initC1(){let raf=0;const upd=()=>{cancelAnimationFrame(raf);raf=requestAnimationFrame(renderC1)};
  $('#c1-p').addEventListener('input',e=>{c1.p20=+e.target.value;upd()});$('#c1-mu').addEventListener('input',e=>{c1.mu10=+e.target.value;upd()});
  segInit($('#c1-n'),NS1,c1.N,null,v=>{c1.N=v;renderC1()});renderC1()}

/* ---------------- C2: Yes-token score (check.py C2) ---------------- */
function c2(zy,zn,zo,g,target){const a=[zy/g,zn/g,zo/g],m=Math.max(...a),e=a.map(v=>Math.exp(v-m)),S=e[0]+e[1]+e[2],p=e.map(v=>v/S);
  const x=(zy-zn)/g,two=x<-700?0:1/(1+Math.exp(-x)),ti=target==='N'?1:0,loss=-(a[ti]-m-Math.log(S)),grad=p.map((v,j)=>(v-(j===ti?1:0))/g);return{p,two,loss,grad}}
const c2s={zy:20,zn:0,zo:-40,g:20,t:'Y'};
function renderC2(){const zy=c2s.zy/10,zn=c2s.zn/10,zo=c2s.zo/10,g=c2s.g/20,o=c2(zy,zn,zo,g,c2s.t);
  $('#c2-zyo').textContent=zy.toFixed(1);$('#c2-zno').textContent=zn.toFixed(1);$('#c2-zoo').textContent=zo.toFixed(1);$('#c2-go').textContent=g.toFixed(2);
  [['zy',zy],['zn',zn],['zo',zo]].forEach(([k,v])=>$('#c2-'+k).setAttribute('aria-valuetext',v.toFixed(1)));$('#c2-g').setAttribute('aria-valuetext','γ = '+g.toFixed(2));
  const names=['Yes','No','其他'],ti=c2s.t==='N'?1:0;
  $('#c2-bars').innerHTML=names.map((n,j)=>`<div class="c2bar${j===ti?' tgt':''}"><span>${n}${j===ti?'（目標）':''}</span><div class="tr"><div class="fl" style="width:${(100*o.p[j]).toFixed(2)}%"></div></div><span class="v">${f4(o.p[j])}</span></div>`).join('');
  dl($('#c2-read'),[['GenRM 分數 r = p(Yes)',f4(o.p[0]),1],['只看 Yes 與 No 的 sigmoid',f4(o.two)],['兩者差',f4(o.two-o.p[0])],['損失 −log p(目標)',f4(o.loss)]]);
  const dir=v=>v<0?'上升':v>0?'下降':'不變';
  $('#c2-grad').innerHTML='<thead><tr><th>logit</th><th>∂損失/∂z</th><th>梯度下降後</th></tr></thead><tbody>'+names.map((n,j)=>`<tr><td>z_${['Y','N','O'][j]}（${n}）</td><td>${f4(o.grad[j])}</td><td>${dir(Math.abs(o.grad[j])<5e-5?0:o.grad[j])}</td></tr>`).join('')+'</tbody>';
  announce('c2',`p(Yes) = ${f4(o.p[0])}，sigmoid = ${f4(o.two)}，損失 ${f4(o.loss)}`)}
function initC2(){['zy','zn','zo','g'].forEach(k=>$('#c2-'+k).addEventListener('input',e=>{c2s[k]=+e.target.value;renderC2()}));
  segInit($('#c2-t'),['Y','N'],c2s.t,v=>v==='Y'?'正確（目標 Yes）':'錯誤（目標 No）',v=>{c2s.t=v;renderC2()});renderC2()}

/* ---------------- C3: MajV@K (check.py C3) ---------------- */
const KS=[1,2,4,8,16,32];
function votesPmf(K,qe,qh,h){const bp=q=>{const o=[];for(let y=0;y<=K;y++)o.push(comb(K,y)*q**y*(1-q)**(K-y));return o};const e=bp(qe),d=bp(qh);return e.map((v,y)=>(1-h)*v+h*d[y])}
function c3(K,qe,qh,h,p,N){const a=votesPmf(K,qe,qh,h),b=a.slice().reverse();let acc=0;for(let l=0;l<=K;l++){if(2*l>K)acc+=a[l];else if(2*l===K)acc+=.5*a[l]}
  const A=[0],B=[0];for(let l=0;l<=K;l++){A.push(A[l]+a[l]);B.push(B[l]+b[l])}
  const pmf=binomPmf(N,p);let bon=0;
  for(let C=1;C<=N;C++){const M=N-C;let s=0;for(let l=0;l<=K;l++){if(a[l]===0)continue;
      for(let i=1;i<=C;i++){const ti=comb(C,i)*a[l]**i*A[l]**(C-i);if(ti===0)continue;for(let j=0;j<=M;j++){const tj=comb(M,j)*b[l]**j*B[l]**(M-j);s+=ti*tj*i/(i+j)}}}
    bon+=pmf[C]*s}
  return{acc,bon,a,b}}
const c3s={K:1,qe20:16,sys:false,h20:4,qh20:6},c3Cache=new Map();
function c3Params(){return{qe:c3s.qe20/20,h:c3s.sys?c3s.h20/20:0,qh:c3s.qh20/20}}
function c3All(){const P=c3Params(),key=`${P.qe}|${P.qh}|${P.h}`;if(!c3Cache.has(key))c3Cache.set(key,KS.map(K=>c3(K,P.qe,P.qh,P.h,.3,16)));return c3Cache.get(key)}
let c3Sync=null;
function renderC3(){const P=c3Params(),all=c3All(),ix=KS.indexOf(c3s.K),cur=all[ix],K=c3s.K;
  $('#c3-qeo').textContent=P.qe.toFixed(2);$('#c3-ho').textContent=(c3s.h20/20).toFixed(2);$('#c3-qho').textContent=(c3s.qh20/20).toFixed(2);
  $('#c3-qe').setAttribute('aria-valuetext','q_e = '+P.qe.toFixed(2));$('#c3-h').setAttribute('aria-valuetext','h = '+(c3s.h20/20).toFixed(2));$('#c3-qh').setAttribute('aria-valuetext','q_h = '+(c3s.qh20/20).toFixed(2));
  {const {svg,W,H}=frame($('#c3-hist'),280,480,220),L=56,R=W-12,T=12,B=H-26,mid=(T+B)/2,x=lin(0,1,L+10,R-10),mx=Math.max(...cur.a,...cur.b),y=lin(0,mx,0,(B-T)/2-4),bw=Math.max(2,Math.min(28,(R-L-20)/(K+1)-2));
   el('line',{x1:L,x2:R,y1:mid,y2:mid,class:'axis'},svg);
   for(let l=0;l<=K;l++){const cx=x(l/K);el('rect',{x:cx-bw/2,y:mid-y(cur.a[l]),width:bw,height:y(cur.a[l]),class:'pos-f'},svg);el('rect',{x:cx-bw/2,y:mid,width:bw,height:y(cur.b[l]),class:'neg-f',stroke:'var(--c-neg)','stroke-width':1},svg)}
   [0,.5,1].forEach(t=>tx(svg,x(t),B+18,t.toFixed(1),{class:'lab','text-anchor':'middle'}));tx(svg,4,T+12,'正確',{class:'lab'});tx(svg,4,B-4,'錯誤',{class:'lab'});
   el('line',{x1:x(.5),x2:x(.5),y1:T,y2:B,class:'axis','stroke-dasharray':'2 3'},svg);
   setLabel(svg,'c3h',`K = ${K} 時的分數分布直方圖`)}
  {const {svg,W,H}=frame($('#c3-curve'),280,480,220),L=44,R=W-14,T=12,B=H-26,x=lin(0,5,L,R),y=lin(0,1,B,T);
   [0,.5,1].forEach(t=>el('line',{x1:L,x2:R,y1:y(t),y2:y(t),class:'grid'},svg));axisLeft(svg,y,L,[0,.5,1],t=>(100*t)+'%');
   KS.forEach((k,i)=>tx(svg,x(i),B+18,'K='+k,{class:'lab','text-anchor':'middle'}));
   el('path',{d:lp([[x(0),y(oracle(.3,16))],[x(5),y(oracle(.3,16))]]),class:'ref-s'},svg);
   el('path',{d:lp(all.map((r,i)=>[x(i),y(r.acc)])),class:'pos-s'},svg);all.forEach((r,i)=>el('circle',{cx:x(i),cy:y(r.acc),r:3.5,class:'pos-f'},svg));
   el('path',{d:lp(all.map((r,i)=>[x(i),y(r.bon)])),class:'acc-s','stroke-dasharray':'6 3'},svg);all.forEach((r,i)=>el('rect',{x:x(i)-3.5,y:y(r.bon)-3.5,width:7,height:7,class:'acc-f'},svg));
   el('line',{x1:x(ix),x2:x(ix),y1:T,y2:B,class:'axis','stroke-dasharray':'2 3'},svg);
   setLabel(svg,'c3c',`K = ${K}：RM accuracy ${pct(cur.acc)}，Best-of-16 ${pct(cur.bon)}`)}
  dl($('#c3-read'),[['K',String(K)],['RM accuracy',(100*cur.acc).toFixed(2)+'%',1],['Best-of-16',(100*cur.bon).toFixed(2)+'%'],['Pass@16',pct(oracle(.3,16))]]);
  announce('c3',`K = ${K}，RM accuracy ${(100*cur.acc).toFixed(2)}%，Best-of-16 ${(100*cur.bon).toFixed(2)}%`)}
function initC3(){c3Sync=segInit($('#c3-k'),KS,c3s.K,null,v=>{c3s.K=v;renderC3()});
  $('#c3-next').addEventListener('click',()=>{c3s.K=KS[(KS.indexOf(c3s.K)+1)%KS.length];c3Sync(c3s.K);renderC3()});
  [['qe','qe20'],['h','h20'],['qh','qh20']].forEach(([id,k])=>$('#c3-'+id).addEventListener('input',e=>{c3s[k]=+e.target.value;renderC3()}));
  $('#c3-sys').addEventListener('change',e=>{c3s.sys=e.target.checked;$('#c3-sysbox').classList.toggle('hide',!c3s.sys);renderC3()});renderC3()}

/* ---------------- C4: weighted self-consistency (check.py C4) ---------------- */
const PRESETS={P1:{name:'P1 最高分是錯的',truth:'B',c:[['A',.40],['A',.35],['A',.30],['A',.45],['B',.90],['B',.85],['C',.20],['D',.95]]},
  P2:{name:'P2 與多數一致',truth:'A',c:[['A',.90],['A',.92],['A',.88],['A',.95],['A',.90],['B',.30],['C',.20],['A',.85]]},
  P3:{name:'P3 很多低分錯誤',truth:'B',c:[['A',.30],['A',.30],['A',.30],['A',.30],['A',.30],['A',.30],['B',.85],['B',.80]]}};
const LET=['A','B','C','D'],r9=v=>Math.round(v*1e9)/1e9;
function c4(cands,K){const letters=[...new Set(cands.map(c=>c[0]))].sort();const votes={},sum={},topk={};
  letters.forEach(L=>{const s=cands.filter(c=>c[0]===L).map(c=>c[1]);votes[L]=s.length;sum[L]=s.reduce((a,b)=>a+b,0);topk[L]=s.slice().sort((a,b)=>b-a).slice(0,K).reduce((a,b)=>a+b,0)});
  const pick=f=>letters.reduce((best,L)=>f(L)>f(best)?L:best,letters[0]);
  let bi=0;cands.forEach((c,i)=>{if(c[1]>cands[bi][1])bi=i});
  return{sc:pick(L=>votes[L]),bon:cands[bi][0],wsc:pick(L=>r9(sum[L])),wtk:pick(L=>r9(topk[L])),votes,sum,topk}}
const c4s={pre:'P1',truth:'B',c:PRESETS.P1.c.map(x=>x.slice()),K:4};let c4PreSync=null,c4TruthSync=null;
function c4Load(k){c4s.pre=k;const P=PRESETS[k];c4s.truth=P.truth;c4s.c=P.c.map(x=>x.slice());c4TruthSync(c4s.truth);buildC4List();renderC4()}
function c4Custom(){if(c4s.pre!=='custom'){c4s.pre='custom';c4PreSync('custom')}}
function buildC4List(){const box=$('#c4-list');box.innerHTML='';c4s.c.forEach((c,i)=>{const row=document.createElement('div');row.className='c4row';
  row.innerHTML=`<span>${i+1}</span><span class="ans" role="radiogroup" aria-label="第 ${i+1} 份的答案">${LET.map(L=>`<button type="button" role="radio" data-v="${L}">${L}</button>`).join('')}</span><input type="range" min="0" max="20" step="1" value="${Math.round(c[1]*20)}" aria-label="第 ${i+1} 份的驗證分數"><span class="v">${c[1].toFixed(2)}</span>`;
  const g=$('.ans',row),sync=()=>$$('button',g).forEach(b=>{const on=b.dataset.v===c4s.c[i][0];b.setAttribute('aria-checked',String(on));b.tabIndex=on?0:-1});sync();
  g.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;c4s.c[i][0]=b.dataset.v;sync();c4Custom();renderC4()});
  g.addEventListener('keydown',e=>{const bs=$$('button',g),k=bs.findIndex(b=>b.tabIndex===0);let j;if(e.key==='ArrowRight'||e.key==='ArrowDown')j=(k+1)%4;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')j=(k+3)%4;else return;e.preventDefault();bs[j].click();bs[j].focus()});
  const r=$('input',row);r.addEventListener('input',()=>{c4s.c[i][1]=+r.value/20;$('.v',row).textContent=c4s.c[i][1].toFixed(2);c4Custom();renderC4()});box.appendChild(row)})}
function renderC4(){$('#c4-ko').textContent=String(c4s.K);const o=c4(c4s.c,c4s.K);
  const card=(n,a)=>`<div>${n}<b>${a}</b><span class="${a===c4s.truth?'ok':'ng'}">${a===c4s.truth?'✓ 正確':'✗ 錯誤'}</span></div>`;
  $('#c4-res').innerHTML=card('SC（票數）',o.sc)+card('Best-of-N',o.bon)+card('加權（全部）',o.wsc)+card('加權 top-'+c4s.K,o.wtk);
  $('#c4-tab').innerHTML='<thead><tr><th>答案</th><th><abbr title="票數">票</abbr></th><th><abbr title="分數總和">總分</abbr></th><th><abbr title="最高 K_top 份的分數和">top-K</abbr></th></tr></thead><tbody>'+LET.map(L=>`<tr><td>${L}</td><td>${o.votes[L]||0}</td><td>${(o.sum[L]||0).toFixed(2)}</td><td>${(o.topk[L]||0).toFixed(2)}</td></tr>`).join('')+'</tbody>';
  announce('c4',`SC ${o.sc}，Best-of-N ${o.bon}，加權 ${o.wsc}，top-${c4s.K} ${o.wtk}；正確答案 ${c4s.truth}`)}
function initC4(){c4PreSync=segInit($('#c4-pre'),['P1','P2','P3','custom'],c4s.pre,k=>k==='custom'?'自訂':PRESETS[k].name,k=>{if(k!=='custom')c4Load(k);else c4s.pre='custom'});
  c4TruthSync=segInit($('#c4-truth'),LET,c4s.truth,null,v=>{c4s.truth=v;c4Custom();renderC4()});
  $('#c4-k').addEventListener('input',e=>{c4s.K=+e.target.value;renderC4()});buildC4List();renderC4()}

/* ---------------- static figures ---------------- */
function flowFig(id,nodes,hi){const box=$('#'+id),W=widthOf(box,280,680),row=W>=600,n=nodes.length,gap=26;
  const lh=17,maxL=Math.max(...nodes.map(t=>t.length)),hOf=t=>lh*(row?maxL:t.length)+20,tops=[];let acc=8;nodes.forEach(t=>{tops.push(acc);acc+=hOf(t)+gap});
  const H=row?hOf(nodes[0])+16:acc-gap+8;const {svg}=frame(box,280,680,H);
  const bw=row?(W-(n-1)*gap-8)/n:W-8;
  nodes.forEach((t,i)=>{const bh=hOf(t),x=row?4+i*(bw+gap):4,yy=row?8:tops[i];el('rect',{x,y:yy,width:bw,height:bh,rx:10,class:i===hi?'box-acc':'box'},svg);
    const y0=yy+(bh-lh*t.length)/2+lh-4;t.forEach((s,k)=>tx(svg,x+bw/2,y0+k*lh,s,{class:'lab13','text-anchor':'middle'}));
    if(i<n-1){if(row){const ax=x+bw+3,ay=yy+bh/2;el('path',{d:`M${ax} ${ay}H${ax+gap-6}M${ax+gap-11} ${ay-4}L${ax+gap-6} ${ay}L${ax+gap-11} ${ay+4}`,class:'arrow'},svg)}
      else{const ax=x+bw/2,ay=yy+bh+3;el('path',{d:`M${ax} ${ay}V${ay+gap-6}M${ax-4} ${ay+gap-11}L${ax} ${ay+gap-6}L${ax+4} ${ay+gap-11}`,class:'arrow'},svg)}}})}
function drawFigs(){flowFig('fig-direct',[['題目 x ＋ 解答 y','＋ 提示 I：','Is the answer','correct (Yes/No)?'],['微調過的','驗證器（LLM）'],['下一個 token 的機率','Yes／No／其他 token'],['分數 r = p(Yes)','（Eq. 4）']],3);
  flowFig('fig-cot',[['題目 x ＋ 解答 y','＋ I_CoT：','Let\'s verify','step by step.'],['驗證器生成','驗證理由 v_CoT'],['接上提示 I','Is the answer','correct (Yes/No)?'],['分數 r_CoT','= p(Yes)（Eq. 6）']],3);
  flowFig('fig-vote',[['同一份解答','抽 K 條理由'],['理由 1：No 0.2','理由 2：Yes 0.9','理由 3：Yes 0.8','…（Fig. 3 示意）'],['平均 K 個','p(Yes)'],['分數 r_MajV@K','（Eq. 7）']],3)}
function heroAxis(){const {svg,W,H}=frame($('#hero-axis'),280,640,96),L=12,R=W-12,y0=56,x=lin(0,1,L,R);
  el('line',{x1:L,x2:R,y1:y0,y2:y0,class:'axis'},svg);[0,.5,1].forEach(t=>{el('line',{x1:x(t),x2:x(t),y1:y0-4,y2:y0+4,class:'axis'},svg);tx(svg,x(t),y0+20,t.toFixed(1),{class:'lab','text-anchor':t===0?'start':t===1?'end':'middle'})});
  const mk=(v,lab,an,ly)=>{const cx=x(v);el('path',{d:`M${cx-5} ${y0-5}L${cx+5} ${y0+5}M${cx-5} ${y0+5}L${cx+5} ${y0-5}`,class:'xmark'},svg);if(ly<y0-20)el('line',{x1:cx,x2:cx,y1:y0-8,y2:ly+4,class:'axis'},svg);tx(svg,cx,ly,lab,{class:'lab13','text-anchor':an})};
  // ponytail: two fixed rows (labels sit at opposite ends; a second row avoids overlap at 358px)
  mk(.0015,'GenRM-CoT 0.0015（詳見 05、06）','start',y0-14);mk(.999,'判別式 RM 0.999（詳見 02）','end',W<560?y0-36:y0-14)}

/* ================= timeline ================= */
const TL_GROUP={'2009.01325':['disc','genrm-best-of-n-discriminative-verifiers'],'2110.14168':['disc','genrm-best-of-n-discriminative-verifiers'],
 '2211.14275':['proc','genrm-step-level-supervision-verifiers'],'2305.20050':['proc','genrm-step-level-supervision-verifiers'],
 '2201.11903':['cot','genrm-inference-time-compute-majority-voting'],'2203.11171':['cot','genrm-inference-time-compute-majority-voting'],'2407.21787':['cot','genrm-inference-time-compute-majority-voting'],
 '2206.05802':['crit','genrm-critiques-and-cot-reward-models'],'2306.05685':['judge','genrm-llm-as-a-judge'],'2401.10020':['judge','genrm-llm-as-a-judge'],
 '2203.14465':['uni','genrm-dpo-implicit-verifiers-unification'],'2305.18290':['uni','genrm-dpo-implicit-verifiers-unification'],'2402.06457':['uni','genrm-dpo-implicit-verifiers-unification'],
 '2408.15240':['me','genrm-yes-token-method']};
const TL_NAME={disc:'判別式 RM',proc:'過程／結果監督',cot:'CoT 與推論運算',crit:'批評模型',judge:'LLM 評審',uni:'生成與驗證統一',me:'本文',other:'其他'};
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

/* ================= self test (design.md 6.3 and section 5) ================= */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t=5e-4)=>Math.abs(a-b)<=t+1e-12;
  // C1
  ok(near(bestOfN(.3,16,15),.8470),'C1 BoN(.3,16,1.5)');ok(NS1.every(N=>near(bestOfN(.3,N,0),.3,1e-4)),'C1 mu=0 equals p');ok(near(Phi(.75),.7734,1e-4),'Phi(.75)');ok(near(bestOfN(.3,1,15),.3,1e-4),'C1 N=1 equals p');
  const T1=[[.3,10,[.3,.4093,.5187,.6176,.7002,.7657,.8163]],[.3,15,[.3,.4493,.6088,.7477,.847,.9084,.9438]],[.3,30,[.3,.5029,.7405,.9204,.9882,.9986,.9997]],[.1,15,[.1,.164,.252,.3614,.483,.6034,.7095]],[.6,15,[.6,.7707,.8844,.9428,.9697,.9826,.9893]]];
  T1.forEach(([p,m,row])=>ok(NS1.every((N,i)=>near(bestOfN(p,N,m),row[i])),`C1 table p=${p} mu=${m/10}`));
  ok(near(oracle(.1,16),.8147)&&near(rmAcc(1),.6915)&&near(rmAcc(3),.9332),'C1 Pass@16 / RM acc');
  // C2
  {const o=c2(2,0,0,1,'Y');ok(near(o.p[0],.787)&&near(o.two,.8808)&&near(o.loss,.2395),'C2 default-other')}
  {const o=c2(2,0,-4,1,'Y');ok(near(o.p[0],.8789)&&near(o.p[2],.0022)&&near(o.loss,.1291)&&near(o.grad[0],-.1211)&&near(o.grad[1],.1189),'C2 default')}
  {const o=c2(2,0,2,1,'Y');ok(near(o.p[0],.4683)&&near(o.loss,.7586)&&near(o.grad[2],.4683),'C2 z_O=2')}
  {const o=c2(2,0,-4,.5,'Y');ok(near(o.p[0],.982)&&near(o.grad[0],-.036)&&near(o.grad[1],.036),'C2 gamma=.5')}
  {const o=c2(0,0,-30,1,'Y');ok(Math.abs(o.p[0]-o.two)<1e-9&&near(o.loss,.6931),'C2 sigmoid limit')}
  // C3
  {const o=c3(4,.8,.3,.2,.3,16);ok(near(o.acc,.76)&&near(o.bon,.7248),'C3 K=4 h=.2')}
  ok(near(c3(32,.8,.3,0,.3,16).acc,.9999),'C3 K=32 h=0');
  {const T3=[[.8,.3,0,[.8,.8,.896,.9667,.9958,.9999],[.6313,.862,.9725,.9949,.9966,.9967]],[.8,.3,.2,[.7,.7,.76,.7985,.8066,.8018],[.4999,.6326,.7248,.784,.8412,.9005]],[.7,.3,.2,[.62,.62,.6704,.7244,.77,.7943],[.4115,.5058,.5841,.621,.6305,.6313]],[.6,.3,0,[.6,.6,.648,.7102,.7869,.8716],[.3913,.4873,.6073,.7372,.8672,.9579]],[.8,.3,.5,[.55,.55,.556,.5463,.5229,.5047],[.3437,.3709,.4174,.485,.5734,.6833]]];
   T3.forEach(([qe,qh,h,acc,bon])=>{const r=KS.map(K=>c3(K,qe,qh,h,.3,16));ok(r.every((x,i)=>near(x.acc,acc[i])&&near(x.bon,bon[i])),`C3 table qe=${qe} h=${h}`)})}
  {const v=votesPmf(32,.8,.3,.2);ok(Math.abs(v.reduce((a,b)=>a+b,0)-1)<1e-12,'C3 pmf sums to 1')}
  ok(c3(16,.8,.3,.2,.3,16).acc>c3(32,.8,.3,.2,.3,16).acc,'C3 RM acc approaches 1-h from above');
  // C4
  {const s=(k,K)=>{const o=c4(PRESETS[k].c,K);return[o.sc,o.bon,o.wsc,o.wtk].join('')};
   ok(s('P1',1)==='ADBD'&&s('P1',2)==='ADBB'&&s('P3',4)==='ABAB'&&s('P3',5)==='ABAB'&&s('P3',6)==='ABAA'&&s('P2',1)==='AAAA','C4 presets')}
  // DOM
  ok($$('details.check').length===10,'check boxes: '+$$('details.check').length);
  ok(['c1','c2','c3','c4'].every(id=>!!$('#'+id)),'plates present');
  {const tl=JSON.parse($('#timeline-data').textContent);ok(tl.length===14&&tl.every(e=>TL_GROUP[e.arxiv]),'timeline groups')}
  ok(!!document.querySelector('.katex')||typeof katex==='undefined','KaTeX rendered');
  ok(!document.querySelector('.katex-error'),'no KaTeX errors');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ================= boot ================= */
document.body.insertAdjacentHTML('afterbegin','<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs><pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" fill="var(--surface)"/><line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-2)" stroke-width="2"/></pattern></defs></svg>');
initTheme();initToc();heroAxis();drawFigs();initC1();initC2();initC3();initC4();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{heroAxis();drawFigs();renderC1();renderC3()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))addEventListener('load',()=>setTimeout(selfTest,50));
})();
