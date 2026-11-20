/*@TOY@*/
(()=>{
'use strict';
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
const TEST=/[?&]test\b/.test(location.search);
function el(tag,a,p){const e=document.createElementNS(NS,tag);if(a)for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function tx(p,x,y,s,a){const t=el('text',Object.assign({x,y},a||{}),p);t.textContent=s;return t}
const lin=(d0,d1,r0,r1)=>v=>r0+(v-d0)/(d1-d0)*(r1-r0);
function lp(pts){let s='';for(let i=0;i<pts.length;i++)s+=(i?'L':'M')+pts[i][0].toFixed(1)+' '+pts[i][1].toFixed(1);return s}
function widthOf(box,min,max){return Math.max(min,Math.min(max,Math.floor(box.clientWidth||max)))}
function svgIn(box,W,H){const svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);return svg}
function f4(v){if(!Number.isFinite(v))return'—';const a=Math.abs(v);if(a!==0&&a<1e-3)return v.toExponential(2).replace('e-','×10⁻').replace(/⁻(\d+)/,(m,d)=>'⁻'+d.split('').map(c=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[c]).join(''));return v.toFixed(4)}
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function announceNow(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);n.textContent=s}
function segInit(box,opts,cur,label,onPick){box.textContent='';opts.forEach(o=>{const b=document.createElement('button');b.type='button';b.setAttribute('role','radio');
    b.textContent=label?label(o):String(o);b.dataset.v=String(o);box.appendChild(b)});
  const sync=v=>$$('button',box).forEach(b=>{const on=b.dataset.v===String(v);b.setAttribute('aria-checked',String(on));b.setAttribute('aria-pressed',String(on));b.tabIndex=on?0:-1});sync(cur);
  box.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;const v=opts.find(o=>String(o)===b.dataset.v);sync(v);onPick(v)});
  box.addEventListener('keydown',e=>{const bs=$$('button',box),i=bs.findIndex(b=>b.tabIndex===0);let j=i;if(e.key==='ArrowRight'||e.key==='ArrowDown')j=(i+1)%bs.length;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')j=(i-1+bs.length)%bs.length;else return;e.preventDefault();bs[j].focus();bs[j].click()});
  return sync}
function dl(box,rows){box.innerHTML=rows.map(([k,v,c])=>`<dt>${k}</dt><dd${c?' class="acc"':''}>${v}</dd>`).join('')}
function bindRange(id,onInput,onChange){const s=$('#'+id);s.addEventListener('input',()=>onInput(+s.value));s.addEventListener('change',()=>{onChange&&onChange(+s.value)});return s}

/* ================= shared closed forms (mirror verify/check_closed.py) ================= */
// C1: Example 1. p = [p_aa, p_ab, p_ba, p_bb]
function ex1(p,a){const[aa,ab,ba,bb]=p,P=Math.pow,ra=aa+ab,rb=ba+bb;
  const powA=P(aa,a)+P(ab,a),powB=P(ba,a)+P(bb,a),tmpA=P(ra,a),tmpB=P(rb,a);
  const F=(x,y,r)=>r>0?P(x/r,a)+P(y/r,a):1;const Fa=F(aa,ab,ra),Fb=F(ba,bb,rb);
  const S=p.reduce((s,v)=>s+P(v,a),0),seqPow=p.map(v=>P(v,a)/S),PpowA=powA/(powA+powB),PtmpA=tmpA/(tmpA+tmpB);
  const c2=(x,y)=>{const zx=P(x,a),zy=P(y,a),z=zx+zy;return z>0?[zx/z,zy/z]:[.5,.5]},ca=c2(aa,ab),cb=c2(ba,bb);
  const seqTmp=[PtmpA*ca[0],PtmpA*ca[1],(1-PtmpA)*cb[0],(1-PtmpA)*cb[1]];
  return{powA,powB,tmpA,tmpB,Fa,Fb,PpowA,PtmpA,seqPow,seqTmp}}
const alphaStar=p=>Math.log(2)/Math.log(p[1]/p[2]);   // valid only for p_aa = 0 and p_ba = p_bb
// C2: Prop. 3
const sigm=z=>z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));
function prop3(e,ep,N,a){const lo=ep/Math.pow(N,1-1/a),lPos=a*Math.log(e),lNegPow=a*Math.log(ep)-(a-1)*Math.log(N),lNegTmp=a*Math.log(ep);
  return{lo,pPow:sigm(lPos-lNegPow),pTmp:sigm(lPos-lNegTmp),futHigher:e>ep/N,empty:N<=1||a<=1}}
// C4: Eq. 12
function cost(N,T,B){const K=Math.ceil(T/B);let tr=0;for(let k=1;k<=K;k++)tr+=Math.min(k*B,T)/2;
  return{K,approx:N*T/(4*B),exact:N*B*K*(K+1)/4/T,trunc:N*tr/T,init:N*tr/T+1}}

/* ================= S2 static figure ================= */
function drawSharpen(){const box=$('#s2-fig'),W=widthOf(box,300,640),H=210,svg=svgIn(box,W,H);
  const g=(y,m,s)=>Math.exp(-.5*((y-m)/s)**2)/(s*Math.sqrt(2*Math.PI)),xs=[],p=[],p4=[];
  for(let i=0;i<=600;i++){const y=-3+6*i/600;xs.push(y);p.push(.6*g(y,-1,.6)+.4*g(y,1.5,.4))}
  const tz=a=>{let s=0;for(let i=0;i<600;i++)s+=(a[i]+a[i+1])/2*(6/600);return s};const q=p.map(v=>v**4),z=tz(q);q.forEach((v,i)=>p4.push(v/z));
  const L=30,R=W-10,T=14,B=H-30,x=lin(-3,3,L,R),ym=Math.max(...p4),y=lin(0,ym*1.08,B,T);
  el('line',{x1:L,x2:R,y1:B,y2:B,class:'axis0'},svg);[-3,-1.5,0,1.5,3].forEach(v=>tx(svg,x(v),B+18,String(v).replace('-','−'),{class:'tk','text-anchor':'middle'}));
  el('path',{d:lp(xs.map((v,i)=>[x(v),y(p[i])])),class:'temp'},svg);el('path',{d:lp(xs.map((v,i)=>[x(v),y(p4[i])])),class:'pow dashp'},svg);
  const i1=xs.findIndex(v=>v>=-1.9);tx(svg,x(xs[i1])-4,y(p[i1])-6,'p',{class:'lab','text-anchor':'end'});const i2=p4.indexOf(ym);tx(svg,x(xs[i2])+8,y(ym)+4,'p⁴（正規化後）',{class:'lab halo'})}

/* ================= C1 ================= */
const C1P={orig:[0,.40,.25,.25],one:[0,.45,.275,.275]};const c1={tab:'orig',a:2};
function c1Draw(){const p=C1P[c1.tab],o=ex1(p,c1.a),as=alphaStar(p);
  // bars
  const box=$('#c1-bars'),W=widthOf(box,290,520),H=200,svg=svgIn(box,W,H);const defs=el('defs',null,svg),pt=el('pattern',{id:'c1h',width:6,height:6,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},defs);el('line',{x1:0,y1:0,x2:0,y2:6,class:'hatch-ln'},pt);
  const groups=[['次方的和（冪分佈）',o.PpowA],['和的次方（低溫）',o.PtmpA]],gw=W/2,B=H-46,T=26,y=lin(0,1,B,T),bw=Math.min(44,gw*.26);
  groups.forEach(([name,pa],gi)=>{const cx=gw*(gi+.5);tx(svg,cx,14,name,{class:'lab','text-anchor':'middle'});
    [[pa,'a','solid'],[1-pa,'b','hat']].forEach(([v,lab,kind],j)=>{const xx=cx+(j?4:-bw-4);el('rect',{x:xx,y:y(v),width:bw,height:B-y(v),class:kind==='solid'?'pow-f':'',fill:kind==='hat'?'url(#c1h)':null,stroke:'var(--c-pow)','stroke-width':1.2},svg);
      tx(svg,xx+bw/2,y(v)-5,v.toFixed(3),{class:'tk','text-anchor':'middle'});tx(svg,xx+bw/2,B+16,'x₀ = '+lab,{class:'tk','text-anchor':'middle'})});
    const big=pa>=.5?'a':'b';tx(svg,cx,B+36,'◀ 權重較大：'+big,{class:'tk','text-anchor':'middle'})});
  el('line',{x1:0,x2:W,y1:B,y2:B,class:'axis0'},svg);
  svg.querySelectorAll('rect').forEach(r=>{if(r.getAttribute('fill')==null)r.removeAttribute('fill')});
  svg.setAttribute('aria-label',`α = ${c1.a.toFixed(2)}：冪分佈 P(x₀ = a) = ${o.PpowA.toFixed(3)}，低溫 P(x₀ = a) = ${o.PtmpA.toFixed(3)}`);
  // curve over alpha
  const cb=$('#c1-curve'),W2=widthOf(cb,290,520),H2=190,s2=svgIn(cb,W2,H2),L=34,R=W2-10,T2=12,B2=H2-34,x=lin(1,8,L,R),yy=lin(0,1,B2,T2);
  [0,.5,1].forEach(v=>{el('line',{x1:L,x2:R,y1:yy(v),y2:yy(v),class:v===.5?'refl':'grid'},s2);tx(s2,L-6,yy(v)+4,v.toFixed(1),{class:'tk','text-anchor':'end'})});
  [1,2,4,6,8].forEach(v=>tx(s2,x(v),B2+18,(W2<480?'':'α = ')+v,{class:'tk','text-anchor':'middle'}));if(W2<480)tx(s2,R,B2+30,'α',{class:'tk','text-anchor':'end'});
  const A=[],Bp=[];for(let i=0;i<=140;i++){const a=1+7*i/140,r=ex1(p,a);A.push([x(a),yy(r.PpowA)]);Bp.push([x(a),yy(r.PtmpA)])}
  el('path',{d:lp(A),class:'pow'},s2);el('path',{d:lp(Bp),class:'temp dash'},s2);
  el('line',{x1:x(as),x2:x(as),y1:T2,y2:B2,class:'refl'},s2);tx(s2,x(as)+4,T2+10,'α* = '+as.toFixed(3),{class:'tk halo'});
  tx(s2,R,yy(ex1(p,8).PpowA)+14,'冪分佈',{class:'lab halo','text-anchor':'end'});tx(s2,R,yy(ex1(p,8).PtmpA)-6,'低溫',{class:'lab halo','text-anchor':'end'});
  el('circle',{cx:x(c1.a),cy:yy(o.PpowA),r:5,class:'dot-c'},s2);el('rect',{x:x(c1.a)-4,y:yy(o.PtmpA)-4,width:8,height:8,class:'dot-c'},s2);
  s2.setAttribute('aria-label',`P(x₀ = a) 對 α 的曲線；翻轉點 α* = ${as.toFixed(3)}`);
  const s=o.seqPow,t=o.seqTmp;
  dl($('#c1-read'),[['α',c1.a.toFixed(2)],['冪分佈權重 a',f4(o.powA)],['冪分佈權重 b',f4(o.powB)],['低溫權重 a',f4(o.tmpA),1],['低溫權重 b',f4(o.tmpB),1],
    ['冪分佈 ab',f4(s[1])],['冪分佈 ba = bb',f4(s[2])],['低溫 ab',f4(t[1])],['低溫 ba = bb',f4(t[2])]]);
  $('#c1-ao').textContent=c1.a.toFixed(2);$('#c1-a').setAttribute('aria-valuetext','α = '+c1.a.toFixed(2));
  announce('c1',`α = ${c1.a.toFixed(2)}；冪分佈 a ${f4(o.powA)} 對 b ${f4(o.powB)}；低溫 a ${f4(o.tmpA)} 對 b ${f4(o.tmpB)}`)}
function initC1(){segInit($('#c1-tab'),['orig','one'],c1.tab,k=>k==='orig'?'論文原例（總和 0.90）':'總和為 1 的版本',k=>{c1.tab=k;c1Draw()});
  bindRange('c1-a',v=>{c1.a=v;c1Draw()});$$('#c1-chips button').forEach(b=>b.addEventListener('click',()=>{c1.a=+b.dataset.a;$('#c1-a').value=c1.a;c1Draw()}));c1Draw()}

/* ================= C2 ================= */
const c2={e:.4,ep:.5,N:2,a:2,msg:''};
function c2Draw(){const o=prop3(c2.e,c2.ep,c2.N,c2.a),box=$('#c2-fig'),W=widthOf(box,290,640),H=132,svg=svgIn(box,W,H),L=14,R=W-14,x=lin(0,1,L,R),Y=58;
  if(!o.empty){el('rect',{x:x(o.lo),y:Y-16,width:Math.max(1,x(c2.ep)-x(o.lo)),height:32,class:'band'},svg);tx(svg,(x(o.lo)+x(c2.ep))/2,Y-40,'Eq. 14 區間',{class:'tk','text-anchor':'middle'});
    el('path',{d:`M${x(o.lo)+6} ${Y-16}H${x(o.lo)}V${Y+16}H${x(o.lo)+6}M${x(c2.ep)-6} ${Y-16}H${x(c2.ep)}V${Y+16}H${x(c2.ep)-6}`,class:'brk'},svg)}
  el('line',{x1:L,x2:R,y1:Y,y2:Y,class:'axis0'},svg);[0,.25,.5,.75,1].forEach(v=>{el('line',{x1:x(v),x2:x(v),y1:Y,y2:Y+5,class:'axis0'},svg);tx(svg,x(v),Y+60,v.toFixed(2),{class:'tk','text-anchor':'middle'})});
  const tri=x(c2.ep/c2.N);el('path',{d:`M${tri} ${Y+6}l-6 10h12z`,class:'mk-o'},svg);tx(svg,tri,Y+32,'ε′/N',{class:'tk halo','text-anchor':'middle'});
  const ex=x(c2.e);el('path',{d:`M${ex} ${Y-9}l9 9-9 9-9-9z`,class:'dot-c'},svg);tx(svg,ex,Y-21,'ε',{class:'lab halo','text-anchor':'middle'});
  svg.setAttribute('aria-label',o.empty?'區間為空':`Eq. 14 區間 (${o.lo.toFixed(3)}, ${c2.ep.toFixed(2)})，ε = ${c2.e.toFixed(2)}`);
  const bar=v=>`<span class="mbar"><span style="width:${(v*100).toFixed(1)}%"></span></span>`;
  $('#c2-rows').innerHTML=`<div class="crow"><span class="k">冪分佈：只在 x_t 與 x′_t 兩者之間選時，選 x_t 的機率</span><span class="v">${o.pPow.toFixed(4)}</span>${bar(o.pPow)}<span class="w">${o.pPow>.5?'x_t 權重較大':o.pPow<.5?'x′_t 權重較大':'相同'}</span></div>
   <div class="crow"><span class="k">低溫：只在兩者之間選時，選 x_t 的機率</span><span class="v">${o.pTmp.toFixed(4)}</span>${bar(o.pTmp)}<span class="w">${o.pTmp>.5?'x_t 權重較大':o.pTmp<.5?'x′_t 權重較大':'相同'}</span></div>
   <div class="crow"><span class="k">未來 likelihood：x_t 的唯一未來 對 x′_t 每條未來</span><span class="v">${c2.e.toFixed(3)} ${o.futHigher?'＞':'≤'} ${(c2.ep/c2.N).toFixed(3)}</span></div>
   <p class="note small">${o.empty?'區間為空（需 N > 1 且 α > 1）。':`Eq. 14 區間：(${o.lo.toFixed(4)}, ${c2.ep.toFixed(2)})；ε ${c2.e>o.lo&&c2.e<c2.ep?'在':'不在'}區間內。`}${c2.msg}</p>`;
  [['c2-e',c2.e,2,'ε'],['c2-ep',c2.ep,2,'ε′'],['c2-N',c2.N,0,'N'],['c2-a',c2.a,2,'α']].forEach(([id,v,d,n])=>{$('#'+id+'o').textContent=v.toFixed(d);$('#'+id).setAttribute('aria-valuetext',`${n} = ${v.toFixed(d)}`)});
  announce('c2',`ε = ${c2.e.toFixed(2)}，N = ${c2.N}：冪分佈選 x_t 的機率 ${o.pPow.toFixed(3)}，低溫 ${o.pTmp.toFixed(3)}`)}
function initC2(){const clamp=(who)=>{c2.msg='';if(c2.e+c2.ep>1+1e-9){if(who==='e')c2.ep=+(1-c2.e).toFixed(2),$('#c2-ep').value=c2.ep;else c2.e=+(1-c2.ep).toFixed(2),$('#c2-e').value=c2.e;c2.msg=' 已調整使 ε + ε′ ≤ 1（前綴為空的情形）。'}};
  bindRange('c2-e',v=>{c2.e=v;clamp('e');c2Draw()});bindRange('c2-ep',v=>{c2.ep=v;clamp('ep');c2Draw()});bindRange('c2-N',v=>{c2.N=v;c2Draw()});bindRange('c2-a',v=>{c2.a=v;c2Draw()});
  $('#c2-ex').addEventListener('click',()=>{Object.assign(c2,{e:.4,ep:.5,N:2,a:2,msg:''});['e','ep','N','a'].forEach(k=>$('#c2-'+k).value=c2[k]);c2Draw()});c2Draw()}

/* ================= toy (C3a, C3) ================= */
const TOY=toyCore(null);
// C3a
const c3a={tau:'inv',seed:3,st:null,hist:[],last:null};
function c3aModel(){return TOY.makeModel(3,1,4,c3a.tau==='inv'?.25:1)}
function c3aReset(){c3a.st=TOY.chainStart(c3aModel(),c3a.seed);c3a.hist=[];c3a.last=null;c3aDraw()}
const S=a=>a.map(v=>'abc'[v]).join('');
function cells(arr,m,cls){return `<div class="cells ${cls||''}">`+arr.map((v,i)=>`<span class="cell${m&&i>=m-1?' rs':''}">${'abc'[v]}<small>${i+1}</small></span>`).join('')+'</div>'}
function c3aDraw(){const o=c3a.last,st=c3a.st;let h=`<div class="c3a-row"><span class="k">目前序列</span>${cells(o?o.old:st.x,o?o.m:0)}</div>`;
  if(o){h+=`<div class="c3a-row"><span class="k">候選（m = ${o.m}）</span>${cells(o.y,o.m,'cand')}</div>`;
    const inv=c3a.tau==='inv';h+=`<div class="tscroll" tabindex="0" role="region" aria-label="接受率的逐項分解"><table class="mini"><caption>${inv?'τ = 1/α：每個位置的 log Z_t(x′) − log Z_t(x)（本頁推導）':'τ = 1：每個位置的 log w(x′_t) − log w(x_t)（Z_t 化簡不適用）'}</caption><thead><tr><th>位置 t</th><th class="num">差值</th></tr></thead><tbody>`+
      o.rows.map(r=>`<tr><td>t = ${r.t}</td><td class="num">${(inv?r.dz:r.dw).toFixed(4).replace('-','−')}</td></tr>`).join('')+`<tr class="hl"><td>Σ</td><td class="num">${o.sigma.toFixed(4).replace('-','−')}</td></tr></tbody></table></div>
      <p class="okline">A = min{1, exp(Σ)} = <b>${o.A.toFixed(4)}</b>；u = ${o.u.toFixed(4)}；${o.acc?'u ≤ A，接受 ✓':'u > A，拒絕 ✕'}</p>`}
  else h+='<p class="note small">按「下一步」執行一步 MH。</p>';
  $('#c3a-out').innerHTML=h;
  $('#c3a-hist').innerHTML=c3a.hist.length?`<div class="tscroll" tabindex="0" role="region" aria-label="最近 8 步"><table class="mini"><caption>最近 8 步</caption><thead><tr><th>步</th><th>m</th><th>候選</th><th class="num">A</th><th class="num">u</th><th>結果</th></tr></thead><tbody>`+
    c3a.hist.slice(-8).map(r=>`<tr><td>步 ${r.n}</td><td>m = ${r.m}</td><td class="mono">${S(r.y)}</td><td class="num">A = ${r.A.toFixed(4)}</td><td class="num">u = ${r.u.toFixed(4)}</td><td>${r.acc?'接受 ✓':'拒絕 ✕'}</td></tr>`).join('')+'</tbody></table></div>':'';
  $('#c3a-seed').textContent='chainSeed = '+c3a.seed}
function initC3a(){segInit($('#c3a-tau'),['inv','one'],c3a.tau,k=>k==='inv'?'1/α（論文推理任務）':'1',k=>{c3a.tau=k;c3aReset()});
  $('#c3a-step').addEventListener('click',()=>{const o=TOY.chainStep(c3a.st);c3a.last=o;c3a.hist.push(o);c3aDraw();announceNow('c3a',`第 ${o.n} 步：m = ${o.m}，A = ${o.A.toFixed(2)}，${o.acc?'接受':'拒絕'}`)});
  $('#c3a-reset').addEventListener('click',()=>{c3a.seed=c3a.seed%999+1;c3aReset();announceNow('c3a','重新開始，chainSeed = '+c3a.seed)});c3aReset()}
// C3 worker
let WK=null,WK_OK=false;
if(!TEST){try{WK=new Worker(URL.createObjectURL(new Blob(['('+toyCore.toString()+')(self)'],{type:'text/javascript'})));WK_OK=true}catch(e){WK=null}}
const c3={V:3,alpha:4,tau:'inv',B:2,N:10,seed:1,reqId:0,res:null};
const tauVal=()=>c3.tau==='inv'?1/c3.alpha:c3.tau==='one'?1:.5;
const c3key=()=>`${c3.V}|${c3.alpha}|${tauVal()}|${c3.B}|${c3.seed}`;
function c3Send(full){const id=++c3.reqId,msg=full?{reqId:id,key:c3key(),V:c3.V,alpha:c3.alpha,tau:tauVal(),B:c3.B,seed:c3.seed,nList:[c3.N]}:{reqId:id,key:c3key(),nList:[c3.N]};
  $('#c3').classList.add('busy');$('#c3-status').textContent='計算中…';
  if(WK){WK.postMessage(msg)}else setTimeout(()=>c3Recv(TOY.handle(Object.assign({},msg,full?{}:{V:c3.V,alpha:c3.alpha,tau:tauVal(),B:c3.B,seed:c3.seed}))),0)}
function c3Recv(r){if(r.reqId!==c3.reqId)return;if(r.progress){$('#c3-status').textContent=`計算中…（區塊 ${r.progress[0]}/${r.progress[1]}）`;return}
  if(r.miss){c3Send(true);return}c3.res=r;$('#c3').classList.remove('busy');$('#c3-status').textContent=`矩陣建構與 TV 曲線耗時 ${r.ms} ms`;c3Draw()}
if(WK){WK.onmessage=e=>c3Recv(e.data);WK.onerror=()=>{WK=null;c3Send(true)}}
let c3t=0;function c3Debounced(){clearTimeout(c3t);c3t=setTimeout(()=>c3Send(false),150)}
function c3Draw(){const r=c3.res;if(!r)return;
  // ranking
  const box=$('#c3-rank'),W=widthOf(box,290,640),narrow=W<480,rowH=narrow?40:24,lab=narrow?0:64,H=rowH*10+30,svg=svgIn(box,W,H),L=lab+6,R=W-8;
  const mx=Math.max(...r.top.map(t=>Math.max(t.base,t.temp,t.pow,t.alg))),x=lin(0,mx*1.04,L,R);
  const rows=r.top;
  rows.forEach((t,i)=>{const y0=10+i*rowH,yc=narrow?y0+26:y0+12;tx(svg,narrow?L:lab,narrow?y0+10:yc+4,t.s,{class:'mono-t','text-anchor':narrow?'start':'end'});
    el('line',{x1:L,x2:R,y1:yc,y2:yc,class:'grid'},svg);el('rect',{x:L,y:yc-5,width:Math.max(0,x(t.pow)-L),height:10,class:'pow-f op'},svg);
    el('circle',{cx:x(t.base),cy:yc,r:5,class:'mk-o'},svg);el('path',{d:`M${x(t.temp)} ${yc-6}l6 10h-12z`,class:'mk-t'},svg);el('path',{d:`M${x(t.alg)} ${yc-6}l6 6-6 6-6-6z`,class:'dot-c'},svg)});
  $('#c3-other').textContent=`其餘序列合計：base ${r.other.base.toFixed(3)}、低溫 ${r.other.temp.toFixed(3)}、p^α ${r.other.pow.toFixed(3)}、Alg. 1 ${r.other.alg.toFixed(3)}`;
  [0,mx/2,mx].forEach(v=>tx(svg,x(v),H-6,v.toFixed(2),{class:'tk','text-anchor':'middle'}));
  svg.setAttribute('aria-label',`前 10 條序列；最高 ${r.top[0].s}：base ${r.top[0].base.toFixed(3)}，低溫 ${r.top[0].temp.toFixed(3)}，p^α ${r.top[0].pow.toFixed(3)}，Alg. 1 ${r.top[0].alg.toFixed(3)}`);
  // TV curve
  const cb=$('#c3-tv'),W2=widthOf(cb,290,640),H2=200,s2=svgIn(cb,W2,H2),L2=40,R2=W2-12,T2=14,B2=H2-32,xs=lin(0,Math.sqrt(50),L2,R2),xn=n=>xs(Math.sqrt(n)),ym=Math.max(r.tvBase,r.tvTemp,...r.curve.map(c=>c.tv),.05),y=lin(0,ym*1.08,B2,T2);
  [0,ym/2,ym].forEach(v=>{el('line',{x1:L2,x2:R2,y1:y(v),y2:y(v),class:'grid'},s2);tx(s2,L2-6,y(v)+4,v.toFixed(2),{class:'tk','text-anchor':'end'})});
  [0,1,2,5,10,20,50].forEach(n=>tx(s2,xn(n),B2+18,String(n),{class:'tk','text-anchor':'middle'}));tx(s2,R2,B2+30,'N_MCMC（平方根刻度）',{class:'tk','text-anchor':'end'});
  el('line',{x1:L2,x2:R2,y1:y(r.tvTemp),y2:y(r.tvTemp),class:'refl'},s2);tx(s2,R2,y(r.tvTemp)-5,'TV(低溫, p^α)',{class:'tk halo','text-anchor':'end'});
  if(r.tvBase<=ym*1.08){el('line',{x1:L2,x2:R2,y1:y(r.tvBase),y2:y(r.tvBase),class:'refl'},s2);tx(s2,R2,y(r.tvBase)-5,'TV(base, p^α)',{class:'tk halo','text-anchor':'end'})}
  el('path',{d:lp(r.curve.map(c=>[xn(c.N),y(c.tv)])),class:'acc-s'},s2);r.curve.forEach(c=>el('circle',{cx:xn(c.N),cy:y(c.tv),r:3.5,class:'dot-c'},s2));
  el('path',{d:`M${xn(c3.N)} ${y(r.tvAlg)-8}l8 8-8 8-8-8z`,class:'mk-cur'},s2);
  s2.setAttribute('aria-label',`TV 對 N_MCMC：N = 0 時 ${r.curve[0].tv.toFixed(3)}，N = 50 時 ${r.curve[r.curve.length-1].tv.toFixed(3)}`);
  dl($('#c3-read'),[['TV(base, p^α)',r.tvBase.toFixed(4)],['TV(低溫, p^α)',r.tvTemp.toFixed(4)],[`TV(Alg. 1, p^α)，N = ${c3.N}`,r.tvAlg.toFixed(4),1]]);
  announce('c3',`N = ${c3.N}：TV(Alg. 1, p^α) = ${r.tvAlg.toFixed(4)}；TV(低溫, p^α) = ${r.tvTemp.toFixed(4)}`)}
function initC3(){segInit($('#c3-V'),[2,3],c3.V,null,v=>{c3.V=v;c3Send(true)});
  segInit($('#c3-tau'),['inv','one','half'],c3.tau,k=>({inv:'1/α（論文推理任務）',one:'1',half:'0.5'})[k],k=>{c3.tau=k;c3Send(true)});
  segInit($('#c3-B'),[1,2,3,6],c3.B,null,v=>{c3.B=v;c3Send(true)});
  bindRange('c3-a',v=>{c3.alpha=v;$('#c3-ao').textContent=v.toFixed(1);$('#c3-a').setAttribute('aria-valuetext','α = '+v.toFixed(1))},v=>c3Send(true));
  bindRange('c3-N',v=>{c3.N=v;$('#c3-No').textContent=String(v);$('#c3-N').setAttribute('aria-valuetext','N_MCMC = '+v);c3Debounced()});
  $('#c3-new').addEventListener('click',()=>{c3.seed=c3.seed%999+1;$('#c3-seed').textContent='seed = '+c3.seed;c3Send(true)});
  $('#c3-ao').textContent='4.0';$('#c3-No').textContent='10';
  if(TEST){c3.res=TOY.handle({reqId:0,key:c3key(),V:3,alpha:4,tau:.25,B:2,seed:1,nList:[10]});$('#c3-status').textContent='（測試模式：主執行緒計算）';c3Draw()}else c3Send(true)}

/* ================= C4 ================= */
const c4={N:10,T:679,B:192};
function c4Draw(){const o=cost(c4.N,c4.T,c4.B),mx=Math.max(o.approx,o.exact,o.trunc,o.init,1e-9);
  const rows=[['近似 N·T/(4B)（論文）',o.approx,'paper'],['Eq. 12 精確和',o.exact,'d'],['最後區塊截到 T',o.trunc,'d'],['截斷 + 初始化',o.init,'d']];
  $('#c4-bars').innerHTML=rows.map(([k,v,t])=>`<div class="cbar"><span class="k">${k}${t==='d'?' <span class="tag">本頁推導</span>':''}</span><span class="track"><span class="${t==='paper'?'pb':'db'}" style="width:${(v/mx*100).toFixed(1)}%"></span></span><span class="v">${v.toFixed(2)}×</span></div>`).join('');
  $('#c4-k').textContent=`K = ⌈T/B⌉ = ${o.K}；精確和用 K(K+1)/2 = ${o.K*(o.K+1)/2}；近似相當於 (T/B)²/2 = ${((c4.T/c4.B)**2/2).toFixed(2)}。`;
  const cb=$('#c4-fig'),W=widthOf(cb,290,640),H=190,svg=svgIn(cb,W,H),L=44,R=W-12,T=12,B=H-30,x=lin(0,3072,L,R);const pa=[],pi=[];let ym=1;
  for(let i=1;i<=128;i++){const t=Math.round(3072*i/128),c=cost(c4.N,t,c4.B);pa.push([t,c.approx]);pi.push([t,c.init]);ym=Math.max(ym,c.approx,c.init)}const y=lin(0,ym*1.05,B,T);
  [0,ym/2,ym].forEach(v=>{el('line',{x1:L,x2:R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-6,y(v)+4,v.toFixed(0)+'×',{class:'tk','text-anchor':'end'})});
  [0,1024,2048,3072].forEach(v=>tx(svg,x(v),B+18,'T = '+v,{class:'tk','text-anchor':v?'middle':'start'}));
  el('path',{d:lp(pa.map(([t,v])=>[x(t),y(v)])),class:'temp dash'},svg);el('path',{d:lp(pi.map(([t,v])=>[x(t),y(v)])),class:'acc-s'},svg);
  tx(svg,x(2300),y(cost(c4.N,2300,c4.B).init)-8,'截斷 + 初始化',{class:'tk halo','text-anchor':'middle'});tx(svg,x(2300),y(cost(c4.N,2300,c4.B).approx)+16,'近似',{class:'tk halo','text-anchor':'middle'});
  el('circle',{cx:x(c4.T),cy:y(o.init),r:5,class:'dot-c'},svg);
  svg.setAttribute('aria-label',`成本倍數對生成長度 T；目前 T = ${c4.T}：近似 ${o.approx.toFixed(2)} 倍，截斷 + 初始化 ${o.init.toFixed(2)} 倍`);
  [['c4-N',c4.N],['c4-T',c4.T],['c4-B',c4.B]].forEach(([id,v])=>{$('#'+id+'o').textContent=String(v)});$('#c4-Tn').value=c4.T;
  announce('c4',`N = ${c4.N}，T = ${c4.T}，B = ${c4.B}：近似 ${o.approx.toFixed(2)} 倍，Eq. 12 精確和 ${o.exact.toFixed(2)} 倍`)}
function initC4(){const cl=(v,a,b)=>Math.max(a,Math.min(b,Math.round(+v||a)));
  bindRange('c4-N',v=>{c4.N=cl(v,0,50);c4Draw()});bindRange('c4-T',v=>{c4.T=cl(v,1,3072);c4Draw()});bindRange('c4-B',v=>{c4.B=cl(v,1,3072);c4Draw()});
  $('#c4-Tn').addEventListener('change',e=>{c4.T=cl(e.target.value,1,3072);$('#c4-T').value=c4.T;c4Draw()});
  $$('#c4-chips button').forEach(b=>b.addEventListener('click',()=>{const[k,v]=b.dataset.set.split('=');c4[k]=+v;$('#c4-'+k).value=+v;c4Draw()}));c4Draw()}

/* ================= timeline ================= */
const TL_GROUP={'2203.02155':['rl','rswp-rlvr-grpo-background'],'2402.03300':['rl','rswp-rlvr-grpo-background'],'2501.12948':['rl','rswp-rlvr-grpo-background'],'2503.24290':['rl','rswp-rlvr-grpo-background'],
 '2504.13837':['sh','rswp-distribution-sharpening-debate'],'2506.02355':['sh','rswp-distribution-sharpening-debate'],'2509.06941':['sh','rswp-distribution-sharpening-debate'],
 '2505.22660':['ir','rswp-label-free-likelihood-rewards'],'2505.19590':['ir','rswp-label-free-likelihood-rewards'],'2506.10947':['ir','rswp-label-free-likelihood-rewards'],
 '2404.17546':['mc','rswp-metropolis-hastings-autoregressive'],'2503.02819':['an','rswp-annealed-tempered-sampling'],'2502.00921':['an','rswp-power-vs-low-temperature'],
 '2510.14901':['me','rswp-power-sampling-algorithm-results']};
const TL_NAME={rl:'RL 後訓練',sh:'銳化之爭',ir:'內部訊號獎勵',mc:'MCMC／SMC',an:'退火／pivotal token',me:'本文',other:'其他'};
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

/* ================= self test (design.md 5.3 and section 6) ================= */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t=1e-4)=>Math.abs(a-b)<=t+1e-12,nearv=(a,b,t=1e-4)=>a.length===b.length&&a.every((v,i)=>near(v,b[i],t));
  // C1 table (tol 1e-4)
  const C1T=[['orig',1,.4,.5,.4,.5,.4444,.4444,[.4444,.2778],[.4444,.2778]],['orig',2,.16,.125,.16,.25,.5614,.3902,[.5614,.2193],[.3902,.3049]],['orig',4,.0256,.0078,.0256,.0625,.7662,.2906,[.7662,.1169],[.2906,.3547]],
    ['orig',8,6.5536e-4,3.0518e-5,6.5536e-4,3.9063e-3,.9555,.1437,[.9555,.0222],[.1437,.4282]],['one',2,.2025,.1513,.2025,.3025,.5724,.4010,[.5724,.2138],[.4010,.2995]],['one',4,.0410,.0114,.0410,.0915,.7819,.3095,[.7819,.1091],[.3095,.3453]]];
  C1T.forEach(([k,a,pa,pb,ta,tb,PP,PT,sp,st])=>{const o=ex1(C1P[k],a);ok(near(o.powA,pa)&&near(o.powB,pb)&&near(o.tmpA,ta)&&near(o.tmpB,tb)&&near(o.PpowA,PP)&&near(o.PtmpA,PT)&&nearv([o.seqPow[1],o.seqPow[2]],sp)&&nearv([o.seqTmp[1],o.seqTmp[2]],st),`C1 ${k} α=${a}`)});
  {const o=ex1(C1P.orig,2);ok(near(o.Fa,1,1e-12)&&near(o.Fb,.5,1e-12)&&near(.4**2*o.Fa,o.powA,1e-12)&&near(.5**2*o.Fb,o.powB,1e-12),'C1 F decomposition')}
  ok(near(alphaStar(C1P.orig),1.4748)&&near(alphaStar(C1P.one),1.4075),'C1 α*');
  {const o=ex1([0,0,.5,.5],2);ok(o.Fa===1&&o.powA===0&&Math.pow(0,8)===0,'C1 empty row / 0^α')}
  // C2 table
  [[.4,.5,2,2,.3536,.5614,.3902,true],[.4,.5,2,1,.5,.4444,.4444,true],[.3,.5,2,2,.3536,.4186,.2647,true],[.3,.5,4,2,.25,.5902,.2647,true],[.3,.6,8,4,.1261,.9697,.0588,true],[.4,.5,1,4,.5,.2906,.2906,false],[.2,.7,64,8,.0184,1,0,true]]
   .forEach(([e,ep,N,a,lo,pp,pt,fh])=>{const o=prop3(e,ep,N,a);ok(near(o.lo,lo)&&near(o.pPow,pp)&&near(o.pTmp,pt)&&o.futHigher===fh,`C2 ${e},${ep},${N},${a}`)});
  ok(prop3(.4,.5,1,4).empty&&prop3(.4,.5,2,1).empty&&!prop3(.4,.5,2,2).empty,'C2 empty interval');
  // C4 table (tol 1e-3)
  [[10,679,192,4,8.841,14.138,13.483,14.483],[10,3072,192,16,40,42.5,42.5,43.5],[2,679,192,4,1.768,2.828,2.697,3.697],[10,679,679,1,2.5,5,5,6],[0,679,192,4,0,0,0,1],[10,679,1,679,1697.5,1700,1700,1701]]
   .forEach(([N,T,B,K,a,e,t,i])=>{const o=cost(N,T,B);ok(o.K===K&&near(o.approx,a,1e-3)&&near(o.exact,e,1e-3)&&near(o.trunc,t,1e-3)&&near(o.init,i,1e-3),`C4 ${N},${T},${B}`)});
  // toy anchors
  {const r=TOY.mulberry32(1);ok(nearv([r(),r(),r()],[.627074,.002736,.527447],1e-6),'mulberry32(1)')}
  {const M=TOY.makeModel(3,1,4,.25);ok(nearv(M.lp[0].map(Math.exp),[.2731,.4845,.2424])&&nearv(M.lp[1].map(Math.exp),[.7624,.1445,.0931]),'toy p(·|∅), p(·|a)')}
  ok(TOY.zIdentity(2,4,4)<1e-12,'Z_t identity');
  // C3 table (tol 1e-3)
  const C3T=[[3,1,4,'inv',2,.6886,.3150,[.3150,.2454,.1995,.1657,.1191,.0624,.0287,.0041]],[3,1,4,'inv',1,.6886,.3150,[.3150,.2411,.1879,.1523,.1080,.0566,.0246,.0029]],
    [3,1,4,'inv',3,.6886,.3150,[.3150,.2544,.2120,.1780,.1276,.0601,.0299,.0056]],[3,1,4,'inv',6,.6886,.3150,[.3150,.2645,.2306,.2033,.1622,.1056,.0550,.0081]],
    [3,1,4,1,2,.6886,.3150,[.6886,.4603,.3304,.2409,.1350,.0379,.0150,.0066]],[3,1,2,'inv',2,.4189,.1367,[.1367,.0704,.0400,.0248,.0107,.0015,.0004,0]],
    [2,1,4,'inv',2,.6596,.3247,[.3247,.3018,.2789,.2568,.2169,.1439,.0693,.0144]],[3,2,4,'inv',2,.7992,.4389,[.4389,.2824,.1888,.1300,.0663,.0218,.0139,.0051]],[3,1,1,1,2,0,0,[0,0,0,0,0,0,0,0]]];
  C3T.forEach(([V,seed,a,t,B,tb,tt,cv])=>{const tau=t==='inv'?1/a:t,key=`T|${V}|${a}|${tau}|${B}|${seed}`,r=TOY.handle({reqId:1,key,V,alpha:a,tau,B,seed,nList:[10]});
    ok(near(r.tvBase,tb,1e-3)&&near(r.tvTemp,tt,1e-3)&&nearv(r.curve.map(c=>c.tv),cv,1e-3),`C3 V=${V} seed=${seed} α=${a} τ=${t} B=${B}`)});
  {const full=TOY.handle({reqId:1,key:'T|3|4|0.25|2|1',V:3,alpha:4,tau:.25,B:2,seed:1,nList:[10]}),cached=TOY.handle({reqId:2,key:'T|3|4|0.25|2|1',nList:[10]}),miss=TOY.handle({reqId:3,key:'none',nList:[10]});
   ok(near(full.tvAlg,.0624,1e-3)&&Math.abs(full.tvAlg-cached.tvAlg)<1e-12&&miss.miss===true,'C3 key cache path = full path; miss reported');
   const r0=TOY.handle({reqId:4,key:'T|3|4|0.25|2|1',nList:[0]});ok(Math.abs(r0.tvAlg-r0.tvTemp)<1e-12,'C3 N=0 equals low-temperature');
   ok(full.top[0].s==='bacabb'&&nearv([full.top[0].base,full.top[0].temp,full.top[0].pow],[.0344,.6569,.4074]),'C3 top sequence')}
  // C3a trace (tol 1e-4)
  {const st=TOY.chainStart(TOY.makeModel(3,1,4,.25),3),exp=[[3,'bacabb',1,.0836,1],[5,'bacabb',1,.8798,1],[2,'bacabb',1,.5399,1],[1,'bacabb',1,.1543,1],[3,'bacaab',1,.8631,1],[4,'bacabb',.4335,.5818,0],[1,'bacabb',.4335,.5709,0],[5,'bacabb',.4335,.3723,1]];
   ok(S(st.x)==='bacabb','C3a init');exp.forEach(([m,y,A,u,acc],i)=>{const o=TOY.chainStep(st);ok(o.m===m&&S(o.y)===y&&near(o.A,A)&&near(o.u,u)&&o.acc===!!acc&&Math.abs(o.sigma-o.sigmaZ)<1e-12,`C3a step ${i+1}`)})}
  // worker creatable (the worker itself is not awaited under --virtual-time-budget; its code path is the same TOY.handle tested above)
  {let w=null;try{w=new Worker(URL.createObjectURL(new Blob(['('+toyCore.toString()+')(self)'],{type:'text/javascript'})))}catch(e){}ok(!!w,'Blob Worker can be created');if(w)w.terminate()}
  // DOM
  ok($$('details.check').length===11,'check boxes: '+$$('details.check').length);
  ok(['c1','c2','c3a','c3','c4'].every(id=>!!$('#'+id)),'plates present');
  ok($$('.erratum').length===5,'errata callouts: '+$$('.erratum').length);
  ok($('#c3 .plate-lead').textContent.trim().startsWith('本圖的 base model 是由亂數種子產生的小型語言模型'),'C3 first line');
  {const h=$('#top').textContent;ok(!/Metropolis|GRPO|8\.84/.test($$('#top .fact').map(f=>f.textContent).join('')),'hero cards avoid undefined terms')}
  {const tl=JSON.parse($('#timeline-data').textContent);ok(tl.length===14&&tl.every(e=>TL_GROUP[e.arxiv]),'timeline groups')}
  ok(!!document.querySelector('.katex')||typeof katex==='undefined','KaTeX rendered');
  ok(!document.querySelector('.katex-error'),'no KaTeX errors');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ================= boot ================= */
initTheme();initToc();drawSharpen();initC1();initC2();initC3a();initC3();initC4();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{drawSharpen();c1Draw();c2Draw();c3Draw();c4Draw()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(TEST)addEventListener('load',()=>setTimeout(selfTest,50));
})();
