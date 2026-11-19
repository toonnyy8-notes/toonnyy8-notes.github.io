<script>
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
function frame(svg,o){
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${o.W} ${o.H}`);svg.setAttribute('width',o.W);svg.setAttribute('height',o.H);
  const m=o.m,x0=m.l,x1=o.W-m.r,y0=o.H-m.b,y1=m.t;
  const X=lin(o.x[0],o.x[1],x0,x1),Y=lin(o.y[0],o.y[1],y0,y1);
  const g=el('g',null,svg);
  (o.yt||[]).forEach(v=>{const yy=Y(v);el('line',{x1:x0,x2:x1,y1:yy,y2:yy,class:'grid'},g);tx(g,x0-8,yy+4,o.yf?o.yf(v):String(v),{class:'tk','text-anchor':'end'})});
  (o.xt||[]).forEach(v=>{const xx=X(v);el('line',{x1:xx,x2:xx,y1:y0,y2:y0+4,class:'axl'},g);tx(g,xx,y0+18,o.xf?o.xf(v):String(v),{class:'tk','text-anchor':'middle'})});
  el('line',{x1:x0,x2:x1,y1:y0,y2:y0,class:'axl'},g);
  if(o.xl)tx(g,(x0+x1)/2,o.H-4,o.xl,{class:'al','text-anchor':'middle'});
  if(o.yl)tx(g,4,12,o.yl,{class:'al'});
  return {X,Y,g,x0,x1,y0,y1};
}
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const SUP={'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'};
function sci(x,d=2){if(x===0)return '0';const [m,e]=x.toExponential(d).split('e');return m+'×10'+String(+e).replace(/[-0-9]/g,c=>SUP[c])}
const minus=s=>String(s).replace(/-/g,'−');
function segInit(id,cb){$$('#'+id+' button').forEach(b=>b.addEventListener('click',()=>{$$('#'+id+' button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));cb(b.dataset.v)}))}
function segSet(id,v){$$('#'+id+' button').forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.v===String(v))))}
function roving(container,getT,setT){container.addEventListener('keydown',e=>{const btns=$$('button.tok',container);if(!btns.length)return;let t=getT();const n=btns.length;
  if(e.key==='ArrowRight')t=Math.min(n-1,t+1);else if(e.key==='ArrowLeft')t=Math.max(0,t-1);else if(e.key==='Home')t=0;else if(e.key==='End')t=n-1;else return;
  e.preventDefault();setT(t);const b=$$('button.tok',container)[t];if(b)b.focus()})}

/* ---------- shared math (design.md Sec. 4) ---------- */
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function isPrime(n){if(n<2)return false;if(n%2===0)return n===2;for(let i=3;i*i<=n;i+=2)if(n%i===0)return false;return true}
function nextPrimes(M,count,seen){seen=seen||new Set();const out=[];for(let c=M;out.length<count;c++)if(!seen.has(c)&&isPrime(c)){seen.add(c);out.push(c)}return out}
const MULTS=(()=>{const r=mulberry32(0);return [0,1,2].map(()=>BigInt(2*Math.floor(r()*2**30)+1))})();
const TOKRE=/\s*[\p{L}\p{N}]+|\s*[^\s\p{L}\p{N}]/gu;
function tokenize(s){return s.match(TOKRE)||[]}
function normKey(tok){let s=tok.normalize('NFKC').normalize('NFD').replace(/\p{M}/gu,'').toLowerCase().replace(/[ \t\r\n]+/g,' ');if(s===' ')return ' ';s=s.trim();return s||tok}
function compress(toks){const raw=new Map(),can=new Map(),rawIds=[],canonIds=[],keys=[];
  for(const t of toks){if(!raw.has(t))raw.set(t,raw.size+1);const k=normKey(t);if(!can.has(k))can.set(k,can.size+1);rawIds.push(raw.get(t));canonIds.push(can.get(k));keys.push(k)}
  return {rawIds,canonIds,keys,V:raw.size,Vp:can.size}}
function hashHeads(ids,t,n,primes){let mix=0n;for(let i=0;i<n;i++){const v=BigInt(t-i>=0?ids[t-i]:0)*MULTS[i];mix=i?(mix^v):v}return {mix,idx:primes.map(p=>Number(mix%BigInt(p)))}}
const LOG10R=48*Math.log10(2);
function collide(U,M,K){if(U<=1)return {p1:0,pAll:0,log10Ind:-Infinity,log10Floor:-Infinity};
  const p1=-Math.expm1((U-1)*Math.log1p(-1/M)),lk=K*Math.log10(M),lu=Math.log10(U-1);
  return {p1,pAll:Math.pow(p1,K),log10Ind:lu-lk,log10Floor:lu-Math.min(lk,LOG10R)}}
async function empirical(U,M,K,onProg,alive){alive=alive||(()=>true);
  const P=nextPrimes(M,K).map(BigInt),occ=P.map(()=>new Map()),tup=new Map(),z=new Int32Array(U*K),key=new Array(U);
  for(let i=0;i<U;i++){const a=Math.floor(i/1000),b=i%1000,mix=(BigInt(b+1)*MULTS[0])^(BigInt(a+1)*MULTS[1]);
    for(let k=0;k<K;k++){const v=Number(mix%P[k]);z[i*K+k]=v;occ[k].set(v,(occ[k].get(v)||0)+1)}
    const s=Array.from(z.subarray(i*K,i*K+K)).join(',');key[i]=s;tup.set(s,(tup.get(s)||0)+1);
    if((i+1)%2000===0){if(onProg)onProg(i+1);await new Promise(r=>setTimeout(r,0));if(!alive())return null}}
  let c1=0,cAll=0,full=0;
  for(let i=0;i<U;i++){if(occ[0].get(z[i*K])>=2)c1++;let all=true;for(let k=0;k<K;k++)if(occ[k].get(z[i*K+k])<2){all=false;break}if(all)cAll++;if(tup.get(key[i])>=2)full++}
  return {U,c1,cAll,full,P0:Number(P[0])}}
const frac=(c,U)=>(c/U).toFixed(4);
const sig=s=>s>=0?1/(1+Math.exp(-s)):Math.exp(s)/(1+Math.exp(s));
function gate(d,c,g,sq){let s=g*Math.sqrt(d)*c;if(sq)s=s===0?0:Math.sign(s)*Math.sqrt(Math.abs(s));return {s,a:sig(s),oma:sig(-s)}}
function field(t,N,w,delta,zero){const S=new Set(),taps=zero?1:w;for(let j=0;j<taps;j++){const c=t-j*delta;for(let i=0;i<N;i++)S.add(c-i)}return [...S].sort((a,b)=>a-b)}
const EMB=2*129280*2560/1e9;
const PRESETS={'2e20':{Pt:5.7,Pa:0.568,E0:106},'6e20':{Pt:9.9,Pa:0.993,E0:99},'27B':{Pt:26.7-EMB,Pa:3.8,E0:72}};
function alloc(p,rho){const P=PRESETS[p],Ps=P.Pt-P.Pa;return {Pt:P.Pt,Pa:P.Pa,Ps,Pm:rho*Ps,Pe:(1-rho)*Ps,E:6+rho*(P.E0-6)}}
function prefetch(o){const vol=o.tokens*1280*o.bytes,tx=vol/(o.bw*1e9)*1e3,win=o.mode==='engram'?(o.ell-1)*o.tl:0;return {vol,tx,win,stall:Math.max(0,tx-win)}}

/* ---------- static token bands (hero, Fig. 7) ---------- */
const FIG1='Only Alexander the Great could tame the horse Bucephalus.';
const vis=s=>s.replace(/ /g,'␣');
function staticBand(id,toks,ph,ends){const b=$('#'+id);if(!b)return;b.innerHTML=toks.map((t,i)=>`<span class="tok${ph.includes(i)?' ph':''}${ends.includes(i)?' end':''}">${esc(t)}</span>`).join('')}

/* ---------- C1 ---------- */
const C1TXT=[FIG1,'The cat saw the CAT. Thé café, the Café.'];
const c1={text:C1TXT[0],t:3,K:8,U:1e6,M:2262400,KB:8,run:0,emp:null};
function c1Model(){let toks=tokenize(c1.text);const trunc=toks.length>40;if(trunc)toks=toks.slice(0,40);return {toks,trunc,comp:compress(toks)}}
function renderC1A(){
  const {toks,trunc,comp}=c1Model(),T=toks.length;$('#c1-trunc').classList.toggle('hide',!trunc);
  const band=$('#c1-band'),sl=$('#c1-t');
  if(!T){band.innerHTML='<span class="note">沒有 token。</span>';$('#c1-ro').textContent='請輸入文字。';$('#c1-idx').innerHTML='';$('#c1-cat').innerHTML='';return}
  c1.t=Math.min(c1.t,T-1);sl.max=T-1;sl.value=c1.t;$('#c1-to').textContent=c1.t;
  const forms=new Map();comp.canonIds.forEach((c,i)=>{if(!forms.has(c))forms.set(c,new Set());forms.get(c).add(comp.rawIds[i])});
  const t=c1.t,in2=p=>p<=t&&p>t-2,in3=p=>p<=t&&p>t-3;
  let h='';
  for(const p of [-2,-1])h+=`<div class="tcol"><span class="tok pad">pad<span class="id">0</span></span><i class="${in2(p)?'on2':''}"></i><i class="${in3(p)?'on3':''}"></i></div>`;
  toks.forEach((tok,i)=>{const c=comp.canonIds[i],mg=forms.get(c).size>1;
    h+=`<div class="tcol"><button type="button" class="tok${mg?' merged':''}${i===t?' sel':''}" tabindex="${i===t?0:-1}" data-i="${i}" aria-label="位置 ${i}，原始 ${esc(tok.trim()||'空白')}，原始 ID ${comp.rawIds[i]}，標準 ID ${c}${mg?'，與其他原始 token 合併':''}"><span>${esc(vis(tok))}</span><span class="id">ID ${comp.rawIds[i]}</span><span class="nk">${esc(vis(comp.keys[i]))}</span><span class="id">${mg?'= ':''}標準 ${c}</span></button><i class="${in2(i)?'on2':''}"></i><i class="${in3(i)?'on3':''}"></i></div>`});
  band.innerHTML=h;
  $$('button.tok',band).forEach(b=>b.addEventListener('click',()=>{c1.t=+b.dataset.i;renderC1A()}));
  const seen=new Set(),P2=nextPrimes(10000,c1.K,seen),P3=nextPrimes(10000,c1.K,seen);
  const h2=hashHeads(comp.canonIds,t,2,P2),h3=hashHeads(comp.canonIds,t,3,P3);
  const red=comp.V?(1-comp.Vp/comp.V)*100:0;
  $('#c1-ro').innerHTML=`|V| = <b>${comp.V}</b> → |V′| = <b>${comp.Vp}</b>，縮小 <b class="acc">${red.toFixed(1)}%</b>；t = ${t}（${esc(vis(toks[t]))}，標準 ID ${comp.canonIds[t]}）`;
  const grid=(n,hh,P,lab)=>`<p class="idxlab">${n}-gram（${lab}）· mix = <span class="mono">${hh.mix}</span></p><div class="idxgrid" aria-label="${n}-gram 的 ${P.length} 個索引">${hh.idx.map((z,k)=>`<div><small>M${n},${k+1} = ${P[k]}</small>${z}</div>`).join('')}</div>`;
  $('#c1-idx').innerHTML=grid(2,h2,P2,'實線視窗')+grid(3,h3,P3,'虛線視窗');
  $('#c1-cat').innerHTML='<i class="n2"></i>'.repeat(c1.K)+'<i class="n3"></i>'.repeat(c1.K);
  announce('c1',`|V| = ${comp.V}，|V′| = ${comp.Vp}；t = ${t}；2-gram 索引 ${h2.idx.join('、')}；3-gram 索引 ${h3.idx.join('、')}`);
}
function fmtU(U){return U<1e7?U.toLocaleString('en-US'):sci(U,2)}
function p10(l){return isFinite(l)?`10<sup>${minus(l.toFixed(2))}</sup>`:'0'}
function renderC1B(){
  const {U,M,KB:K}=c1,r=collide(U,M,K);
  $('#c1-Uo').textContent=fmtU(U);$('#c1-Mo').textContent=M.toLocaleString('en-US');$('#c1-KBo').textContent=K;
  $$('#c1 [data-u]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.u===U)));$$('#c1 [data-m]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.m===M)));
  const pa=r.pAll<1e-3?sci(r.pAll,2):r.pAll.toFixed(4),differ=Math.abs(r.log10Ind-r.log10Floor)>1e-9,ok=U<Math.sqrt(2**48);
  $('#c1-ro2').innerHTML=`p₁ = <b>${r.p1.toFixed(6)}</b><br>K 頭都同格 p₁ᴷ = <b class="acc">${pa}</b><br><span class="${differ?'ro-dim':''}">E<sub>full</sub>（獨立模型）= ${p10(r.log10Ind)}</span>${differ?'<br><span class="note">此時 M<sup>K</sup> 超過 mix 範圍，獨立模型不適用</span>':''}<br>E<sub>full</sub>（示範雜湊，mix &lt; 2⁴⁸，推定）= <b>${p10(r.log10Floor)}</b><br>整張表沒有任何全部同格組合的條件：U² ≪ 2⁴⁸（U ≪ 1.7×10⁷）：<b>${ok?'滿足':'不滿足'}</b>`;
  const box=$('#c1-fig'),svg=$('svg',box),W=widthOf(box,260,460),H=220;
  const f=frame(svg,{W,H,m:{l:40,r:10,t:28,b:38},x:[0.5,16.5],y:[0,1],yt:[0,.5,1],xt:[1,4,8,12,16],xl:'頭數 K',yl:'p₁ᴷ'});
  const bw=Math.max(3,(f.x1-f.x0)/16*0.66);
  for(let k=1;k<=16;k++){const v=Math.pow(r.p1,k),x=f.X(k),y=f.Y(v);el('rect',{x:x-bw/2,y:Math.min(y,f.y0-1),width:bw,height:Math.max(1,f.y0-y),class:k===K?'acc-f':'par-f',opacity:k===K?1:.45},f.g)}
  tx(f.g,Math.min(f.X(K),f.x1-40),Math.max(f.Y(r.pAll)-8,28),pa,{class:'labm','text-anchor':'middle'});
  svg.setAttribute('aria-label',`K 個頭都同格的機率對 K 的長條圖；目前 K = ${K}，p₁ᴷ = ${pa.replace(/<[^>]+>/g,'')}`);
  const can=U<=20000&&M<=1e5;$('#c1-run').disabled=!can;
  $('#c1-runmsg').textContent=can?'':'實測僅在 U ≤ 20,000 且 M ≤ 100,000 時啟用。';
  announce('c1',`p₁ = ${r.p1.toFixed(6)}，K 頭都同格 = ${pa}，示範雜湊 E_full = 10^${minus(r.log10Floor.toFixed(2))}`);
}
function empText(e,K){const r=collide(e.U,e.P0,K),pa=r.pAll<1e-3?sci(r.pAll,2):r.pAll.toFixed(4);
  return `第 1 頭同格比例 = <b>${frac(e.c1,e.U)}</b>（${e.c1} / ${e.U}）<br>K 頭都同格比例 = <b class="acc">${frac(e.cAll,e.U)}</b>（${e.cAll} / ${e.U}）<br>索引組完全相同比例 = <b>${frac(e.full,e.U)}</b>（${e.full} / ${e.U}）<br>解析值（P₀ = ${e.P0}）：p₁ = ${r.p1.toFixed(4)}，p₁ᴷ = ${pa}`}
async function runC1(){const my=++c1.run,{U,M,KB:K}=c1,btn=$('#c1-run');btn.disabled=true;
  const e=await empirical(U,M,K,i=>{$('#c1-runmsg').textContent=`已處理 ${i.toLocaleString('en-US')} / ${U.toLocaleString('en-US')}`},()=>my===c1.run);
  if(!e)return;btn.disabled=false;$('#c1-runmsg').textContent='完成。';$('#c1-emp').innerHTML=empText(e,K);
  $('#c1-live').textContent=`實測完成：第 1 頭同格 ${frac(e.c1,U)}，K 頭都同格 ${frac(e.cAll,U)}，完全相同 ${frac(e.full,U)}`}
function c1Changed(){c1.run++;$('#c1-emp').textContent='參數已改變，按下按鈕重新實測。';renderC1B()}
function initC1(){
  const ta=$('#c1-text');ta.value=c1.text;
  $$('#c1 [data-s]').forEach(b=>b.addEventListener('click',()=>{c1.text=C1TXT[+b.dataset.s];ta.value=c1.text;c1.t=3;$$('#c1 [data-s]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));renderC1A()}));
  ta.addEventListener('input',()=>{c1.text=ta.value;$$('#c1 [data-s]').forEach(x=>x.setAttribute('aria-pressed',String(C1TXT[+x.dataset.s]===ta.value)));renderC1A()});
  $('#c1-t').addEventListener('input',e=>{c1.t=+e.target.value;renderC1A()});
  segInit('c1-K',v=>{c1.K=+v;renderC1A()});
  roving($('#c1-band'),()=>c1.t,t=>{c1.t=t;renderC1A()});
  $('#c1-U').addEventListener('input',e=>{c1.U=Math.round(10**+e.target.value);c1Changed()});
  $('#c1-M').addEventListener('input',e=>{c1.M=Math.round(10**+e.target.value);c1Changed()});
  $('#c1-KB').addEventListener('input',e=>{c1.KB=+e.target.value;c1Changed()});
  $$('#c1 [data-u]').forEach(b=>b.addEventListener('click',()=>{c1.U=+b.dataset.u;$('#c1-U').value=Math.log10(c1.U);c1Changed()}));
  $$('#c1 [data-m]').forEach(b=>b.addEventListener('click',()=>{c1.M=+b.dataset.m;$('#c1-M').value=Math.log10(c1.M);c1Changed()}));
  $('#c1-run').addEventListener('click',runC1);
  $('#c1-U').setAttribute('aria-valuetext','U = 1,000,000');
  renderC1A();renderC1B()}

/* ---------- C2 ---------- */
const c2={d:2560,c:0.3,g:1,sq:false,view:'c'};
function renderC2(){
  const {d,c,g,sq,view}=c2,r=gate(d,c,g,sq),k=Math.sqrt(d)*g;
  $('#c2-co').textContent=c.toFixed(3);$('#c2-go').textContent=g.toFixed(2);$('#c2-c').setAttribute('aria-valuetext',`cos θ = ${c.toFixed(3)}`);$('#c2-g').setAttribute('aria-valuetext',`γ = ${g.toFixed(2)}`);
  const box=$('#c2-fig'),svg=$('svg',box),W=widthOf(box,280,560),H=260;
  const xr=view==='c'?[-1,1]:[-6,6];
  const f=frame(svg,{W,H,m:{l:40,r:14,t:18,b:40},x:xr,y:[0,1],yt:[0,.5,1],xt:view==='c'?[-1,-.5,0,.5,1]:[-6,-3,0,3,6],xl:view==='c'?'cos θ':'s',yl:'α'});
  const sOf=x=>view==='c'?k*x:x,sq1=s=>s===0?0:Math.sign(s)*Math.sqrt(Math.abs(s));
  let xs=[];for(let i=0;i<=800;i++)xs.push(xr[0]+(xr[1]-xr[0])*i/800);
  if(view==='c'){for(const lim of [6/k,36/k]){const L=Math.min(1,lim);for(let i=0;i<400;i++)xs.push(-L+2*L*i/399)}xs.sort((a,b)=>a-b)}
  el('path',{d:lp(xs.map(x=>[f.X(x),f.Y(sig(sOf(x)))])),class:'acc'},f.g);
  el('path',{d:lp(xs.map(x=>[f.X(x),f.Y(sig(sq1(sOf(x))))])),class:'ref'},f.g);
  const sRaw=g*Math.sqrt(d)*c,px=view==='c'?c:Math.max(-6,Math.min(6,sRaw)),py=r.a;
  el('circle',{cx:f.X(px),cy:f.Y(py),r:5.5,class:'ink-f'},f.g);
  if(view==='s'&&Math.abs(sRaw)>6)tx(f.g,f.X(px)+(sRaw>0?-8:8),f.Y(py)+(sRaw>0?20:-12),`s = ${minus(sRaw.toFixed(2))} ${sRaw>0?'→':'←'}`,{class:'labm','text-anchor':sRaw>0?'end':'start'});
  tx(f.g,f.x1-4,f.Y(.92),'Eq. 4',{class:'lab','text-anchor':'end'});
  svg.setAttribute('aria-label',`α 曲線；目前 s = ${r.s.toFixed(4)}，α = ${r.a.toFixed(8)}`);
  const cb=$('#c2-circ'),cs=$('svg',cb),S=200,R=74,cx=S/2,cy=S/2;cs.textContent='';cs.setAttribute('viewBox',`0 0 ${S} ${S}`);cs.setAttribute('width',S);cs.setAttribute('height',S);
  el('circle',{cx,cy,r:R,class:'faint',fill:'none'},cs);
  const th=Math.acos(Math.max(-1,Math.min(1,c))),kx=cx+R*Math.cos(th),ky=cy-R*Math.sin(th);
  const ar=(x0,y0,x1,y1,cls,dash)=>{const dx=x1-x0,dy=y1-y0,L=Math.hypot(dx,dy),ux=dx/L,uy=dy/L,h=9;el('line',{x1:x0,y1:y0,x2:x1-ux*h*.7,y2:y1-uy*h*.7,class:cls+(dash?' dash':'')},cs);el('path',{d:`M${x1} ${y1}L${x1-ux*h-uy*h*.45} ${y1-uy*h+ux*h*.45}L${x1-ux*h+uy*h*.45} ${y1-uy*h-ux*h*.45}Z`,class:cls+'-f'},cs)};
  const rr=26;el('path',{d:`M${cx+rr} ${cy}A${rr} ${rr} 0 0 0 ${cx+rr*Math.cos(th)} ${cy-rr*Math.sin(th)}`,class:'ink',style:'stroke-width:1.2'},cs);
  ar(cx,cy,cx+R,cy,'perp');ar(cx,cy,kx,ky,'par',true);
  tx(cs,cx+R,cy+18,'h（Query）',{class:'al','text-anchor':'end'});tx(cs,kx+(Math.cos(th)>=0?6:-6),ky+(ky<cy?-6:16),'k（Key）',{class:'al','text-anchor':Math.cos(th)>=0?'start':'end'});
  tx(cs,cx,S-6,`θ = ${(th*180/Math.PI).toFixed(1)}°`,{class:'labm','text-anchor':'middle'});
  cs.setAttribute('aria-label',`h 與 k 的夾角 θ = ${(th*180/Math.PI).toFixed(1)} 度`);
  const oma=r.oma<1e-3?sci(r.oma,3):r.oma.toFixed(4);
  $('#c2-ro').innerHTML=`s = <b>${minus(r.s.toFixed(4))}</b><br>α = <b class="acc">${r.a.toFixed(8)}</b><br>1 − α = <b>${oma}</b>`;
  $('#c2-bar').style.width=(r.a*100).toFixed(2)+'%';
  announce('c2',`s = ${minus(r.s.toFixed(2))}，α = ${r.a.toFixed(8)}`);
}
function initC2(){
  segInit('c2-d',v=>{c2.d=+v;renderC2()});segInit('c2-view',v=>{c2.view=v;renderC2()});
  const setC=v=>{if(!isFinite(v))return;c2.c=Math.max(-1,Math.min(1,v));$('#c2-c').value=c2.c;$('#c2-cn').value=c2.c;renderC2()};
  $('#c2-c').addEventListener('input',e=>setC(+e.target.value));$('#c2-cn').addEventListener('change',e=>setC(+e.target.value));
  $('#c2-g').addEventListener('input',e=>{c2.g=10**+e.target.value;renderC2()});
  $('#c2-sq').addEventListener('change',e=>{c2.sq=e.target.checked;renderC2()});
  renderC2()}

/* ---------- C3 ---------- */
const c3={t:9,N:3,w:4,d:3,lock:true,zero:false};
const C3TOK=tokenize(FIG1);
function renderC3(){
  if(c3.lock)c3.d=c3.N;
  const {t,N,w,zero}=c3,d=c3.d,pos=field(t,N,w,d,zero),P=new Set(pos);
  $('#c3-wo').textContent=w;$('#c3-do').textContent=d;$('#c3-d').value=d;$('#c3-d').disabled=c3.lock;
  const minP=Math.min(pos[0],0),nPad=-minP,coll=nPad>3;
  const cells=[];if(coll)cells.push({k:'c',a:minP,b:-3});
  for(let p=coll?-2:minP;p<0;p++)cells.push({k:'p',p});
  C3TOK.forEach((tok,i)=>cells.push({k:'t',p:i,tok}));
  const colOf=p=>{if(coll&&p<=-3)return 1;return cells.findIndex(c=>c.k!=='c'&&c.p===p)+1};
  let h='';
  cells.forEach((c,i)=>{const col=i+1;
    if(c.k==='c')h+=`<span class="tok pad cov" style="grid-column:${col}">pad ×${c.b-c.a+1}<span class="id">位置 ${minus(c.a)}…${minus(c.b)}</span></span>`;
    else if(c.k==='p')h+=`<span class="tok pad${P.has(c.p)?' cov':''}" style="grid-column:${col}">pad<span class="id">${minus(c.p)}</span></span>`;
    else h+=`<button type="button" class="tok${P.has(c.p)?' cov':''}${c.p===t?' sel':''}" style="grid-column:${col}" tabindex="${c.p===t?0:-1}" data-i="${c.p}" aria-label="位置 ${c.p}：${esc(c.tok.trim())}${P.has(c.p)?'，在讀取範圍內':''}"><span>${esc(vis(c.tok))}</span><span class="id">${c.p}</span></button>`});
  const taps=zero?1:w;
  for(let j=0;j<taps;j++){const cc=t-j*d,a=colOf(cc-N+1),b=colOf(cc);h+=`<div class="brk s${j%4}" style="grid-row:${j+2};grid-column:${a}/${b+1}"><b>${j?`t − ${j*d}`:'t'}</b></div>`}
  const grid=$('#c3-grid');grid.style.gridTemplateColumns=`repeat(${cells.length},max-content)`;grid.style.paddingBottom='18px';grid.innerHTML=h;
  $$('button.tok',grid).forEach(b=>b.addEventListener('click',()=>{c3.t=+b.dataset.i;renderC3()}));
  const ins=pos.filter(p=>p>=0).length,span=pos[pos.length-1]-pos[0]+1,cont=span===pos.length;
  $('#c3-ro').innerHTML=`涵蓋位置數（含 pad）= <b class="acc">${pos.length}</b><br>句內 token 數 = <b>${ins}</b><br>跨度 = <b>${span}</b><br>pad 格數 = <b>${pos.length-ins}</b><br>連續：<b>${cont?'是':'否'}</b><br>位置：${pos.map(minus).join(', ')}`;
  announce('c3',`t = ${t}，涵蓋 ${pos.length} 個位置，句內 ${ins} 個，pad ${pos.length-ins} 個，${cont?'連續':'不連續'}；位置 ${pos.map(minus).join('、')}`);
}
function initC3(){
  segInit('c3-N',v=>{c3.N=+v;renderC3()});
  $('#c3-w').addEventListener('input',e=>{c3.w=+e.target.value;renderC3()});
  $('#c3-d').addEventListener('input',e=>{c3.d=+e.target.value;renderC3()});
  $('#c3-lock').addEventListener('change',e=>{c3.lock=e.target.checked;renderC3()});
  $('#c3-zero').addEventListener('change',e=>{c3.zero=e.target.checked;renderC3()});
  roving($('#c3-grid'),()=>c3.t,t=>{c3.t=t;renderC3()});
  renderC3()}

/* ---------- C4 ---------- */
const c4={p:'6e20',rho:0.8};
function renderC4(){
  const {p,rho}=c4,a=alloc(p,rho),pct=v=>(v/a.Pt*100);
  $('#c4-ro2').textContent=(rho*100).toFixed(1)+'%';$('#c4-r').setAttribute('aria-valuetext',`ρ = ${(rho*100).toFixed(1)}%`);
  const segs=$$('#c4-bar>div');[a.Pa,a.Pm,a.Pe].forEach((v,i)=>{segs[i].style.flex=`0 0 ${pct(v).toFixed(3)}%`});
  $('#c4-bar').setAttribute('aria-label',`P_tot ${a.Pt.toFixed(3)}B 的組成：啟用 ${a.Pa.toFixed(3)}B，MoE 未啟用 ${a.Pm.toFixed(3)}B，Engram ${a.Pe.toFixed(3)}B`);
  $('#c4-leg').innerHTML=[['sw-act','P_act（每個 token 啟用）',a.Pa],['sw-moe','MoE 未啟用（ρ·P_sparse）',a.Pm],['sw-eng','Engram（(1 − ρ)·P_sparse）',a.Pe]].map(([c,n,v])=>`<li><i class="${c}"></i><span>${n}</span><span class="v">${v.toFixed(3)}B</span><span class="p">${pct(v).toFixed(1)}%</span></li>`).join('');
  $('#c4-ro').innerHTML=`P<sub>sparse</sub> = ${a.Pt.toFixed(3)} − ${a.Pa.toFixed(3)} = <b>${a.Ps.toFixed(3)}B</b><br>P<sub>Engram</sub> = <b class="acc">${a.Pe.toFixed(3)}B</b><br>路由專家數 E = <b>${a.E.toFixed(1)}</b>（約 ${Math.round(a.E)} 個），每個 token 選 top-6`;
  $('#c4-fixed').classList.toggle('hide',p!=='27B');
  $$('#c4 [data-p]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.p===p)));
  const box=$('#c4-fig'),svg=$('svg',box),W=widthOf(box,300,880),top=p==='6e20'?78:0,H=top+144,nar=W<520;
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const x0=18,x1=W-18,X=lin(.3,1,x0,x1),ay=top+64;
  if(p==='6e20'){const Y=lin(1.7109,1.7248,top-14,top-56);tx(svg,x0,12,'驗證損失（6e20，論文值）',{class:'al'});
    el('line',{x1:X(.8),x2:X(.8),y1:Y(1.7109),y2:ay,class:'cur'},svg);el('line',{x1:X(1),x2:X(1),y1:Y(1.7248),y2:ay,class:'cur'},svg);
    el('circle',{cx:X(1),cy:Y(1.7248),r:5,class:'ink-f'},svg);el('circle',{cx:X(.8),cy:Y(1.7109),r:5,class:'ink-f'},svg);
    tx(svg,X(1)-10,Y(1.7248)+4,'1.7248（ρ = 100%）',{class:'labm','text-anchor':'end'});tx(svg,X(.8)-10,Y(1.7109)+4,'1.7109（ρ ≈ 80%）',{class:'labm','text-anchor':'end'})}
  tx(svg,x0,ay-40,'↑ 較高',{class:'al'});tx(svg,x1,ay-40,'↑ 較高',{class:'al','text-anchor':'end'});
  el('rect',{x:X(.75),y:ay-18,width:X(.8)-X(.75),height:18,class:'filla'},svg);
  el('line',{x1:x0,x2:x1,y1:ay,y2:ay,class:'ink'},svg);
  for(let v=3;v<=10;v++){const x=X(v/10);el('line',{x1:x,x2:x,y1:ay,y2:ay+6,class:'axl'},svg);if(!nar||[3,5,7,10].includes(v))tx(svg,x,ay+20,v*10+'%',{class:'tk','text-anchor':v===3?'start':v===10?'end':'middle'})}
  el('circle',{cx:X(.4),cy:ay,r:5,class:'hollow'},svg);el('circle',{cx:X(1),cy:ay,r:5,class:'hollow'},svg);
  tx(svg,X(.4)-6,ay+42,'40% 與純 MoE 相當',{class:'al'});tx(svg,X(.775),ay+(nar?62:42),'最佳 75–80%',{class:'lab','text-anchor':'middle'});
  tx(svg,x1,ay+62,'純 MoE',{class:'al','text-anchor':'end'});
  if(p==='27B'){const x=X(.743);el('path',{d:`M${x} ${ay-6}L${x+6} ${ay}L${x} ${ay+6}L${x-6} ${ay}Z`,class:'perp-f'},svg);tx(svg,x,ay+82,'74.3%',{class:'labm','text-anchor':'middle'})}
  const cx=X(rho);el('path',{d:`M${cx-7} ${ay-26}L${cx+7} ${ay-26}L${cx} ${ay-14}Z`,class:'acc-f'},svg);
  tx(svg,Math.max(x0+40,Math.min(x1-90,cx)),ay-30,`ρ = ${(rho*100).toFixed(1)}%`,{class:'labm','text-anchor':'middle'});
  svg.setAttribute('aria-label',`ρ 軸 30% 到 100%：40% 與純 MoE 相當，75% 到 80% 最佳，100% 為純 MoE；目前 ρ = ${(rho*100).toFixed(1)}%`);
  const marks=['空心點 40%：與純 MoE 相當（兩個預算的文字結果）','底色區間 75%–80%：兩個預算的最佳位置','空心點 100%：純 MoE（ρ = 1）'];
  if(p==='6e20')marks.push('實心點：6e20 預算的驗證損失 1.7248（ρ = 100%）與 1.7109（ρ ≈ 80%），只有這組有損失數字');
  if(p==='27B')marks.push('菱形 74.3%（Engram-27B，論文值，略低於最佳區間）；本頁由專家數推得 74.24%，見上方固定讀數');
  $('#c4-marks').innerHTML=marks.map(s=>`<li>${s}</li>`).join('');
  announce('c4',`ρ = ${(rho*100).toFixed(1)}%，P_Engram = ${a.Pe.toFixed(3)}B，專家數 ${a.E.toFixed(1)}`);
}
function initC4(){
  $$('#c4 [data-p]').forEach(b=>b.addEventListener('click',()=>{c4.p=b.dataset.p;renderC4()}));
  $('#c4-r').addEventListener('input',e=>{c4.rho=Math.round(+e.target.value*200)/200;renderC4()});
  renderC4()}

/* ---------- C5 ---------- */
const c5={mode:'engram',ell:2,tokens:512,bytes:2,bw:25,tl:0.5,size:100};
function niceStep(raw){const p=10**Math.floor(Math.log10(raw));for(const m of [1,2,5,10])if(m*p>=raw)return m*p;return 10*p}
function renderC5(){
  const o=c5,r=prefetch(o),L=(o.ell-1)*o.tl;
  $('#c5-lo').textContent=o.ell;$('#c5-no').textContent=o.tokens.toLocaleString('en-US');$('#c5-bwo').textContent=o.bw;$('#c5-to').textContent=o.tl.toFixed(2);$('#c5-szo').textContent=o.size+'B';
  const box=$('#c5-fig'),svg=$('svg',box),W=widthOf(box,300,620),H=170;
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const defs=el('defs',null,svg),pat=el('pattern',{id:'c5-hatch',width:6,height:6,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},defs);el('line',{x1:0,y1:0,x2:0,y2:6,class:'axl',style:'stroke-width:2'},pat);
  const x0=48,x1=W-12,Tmax=L+r.stall+o.tl,X=lin(0,Tmax,x0,x1),gy=40,py=100;
  const step=niceStep(Tmax/5),dec=Math.max(0,-Math.floor(Math.log10(step)));
  for(let v=0;v<=Tmax+1e-12;v+=step){const x=X(v);el('line',{x1:x,x2:x,y1:gy-8,y2:py+26,class:'grid'},svg);tx(svg,x,H-10,v.toFixed(dec),{class:'tk','text-anchor':'middle'})}
  tx(svg,4,H-10,'ms',{class:'tk'});
  tx(svg,4,gy+19,'GPU',{class:'al'});tx(svg,4,py+15,'PCIe',{class:'al'});
  for(let i=1;i<o.ell;i++){const a=X((i-1)*o.tl),b=X(i*o.tl);el('rect',{x:a+.5,y:gy,width:Math.max(1,b-a-1),height:28,rx:3,class:'gpu'},svg);
    if((i===1||i===o.ell-1)&&b-a>34)tx(svg,(a+b)/2,gy+19,`層 ${i}`,{class:'al','text-anchor':'middle'})}
  if(r.stall>0){const a=X(L),w=Math.max(4,X(L+r.stall)-a);el('rect',{x:a,y:gy,width:w,height:28,class:'hatch'},svg);el('rect',{x:a,y:gy,width:w,height:28,fill:'none',class:'axl'},svg);
    tx(svg,Math.min(a+2,x1-90),gy-12,`停頓 ${r.stall.toFixed(3)} ms`,{class:'labm'})}
  const ea=X(L+r.stall),eb=X(L+r.stall+o.tl);el('rect',{x:ea+.5,y:gy,width:Math.max(2,eb-ea-1),height:28,rx:3,class:'eng'},svg);
  tx(svg,eb-4,gy+44,`層 ${o.ell}（Engram）`,{class:'al','text-anchor':'end'});
  const ta=X(o.mode==='engram'?0:L),tw=Math.max(4,X(r.tx)-X(0));el('rect',{x:ta,y:py,width:tw,height:20,rx:3,class:'xfer'},svg);
  const lab=`傳輸 ${r.tx.toFixed(3)} ms`;if(tw>110)tx(svg,ta+6,py+15,lab,{class:'lab'});else tx(svg,Math.min(ta+tw+6,x1-100),py+15,lab,{class:'labm'});
  svg.setAttribute('aria-label',`時間軸：${o.mode==='engram'?'傳輸從 0 開始':'傳輸從第 '+o.ell+' 層開始'}，傳輸 ${r.tx.toFixed(5)} ms，可重疊 ${r.win.toFixed(3)} ms，停頓 ${r.stall.toFixed(5)} ms`);
  $('#c5-ro').innerHTML=`傳輸量 = <b>${(r.vol/1e6).toFixed(4)} MB</b><br>傳輸時間 Tₓ = <b>${r.tx.toFixed(5)} ms</b><br>可重疊時間 W = <b>${r.win.toFixed(3)} ms</b><br>停頓 = <b class="acc">${r.stall===0?'0':r.stall.toFixed(5)} ms</b><br>表大小 ${o.size}B（不影響傳輸量）`;
  announce('c5',`傳輸量 ${(r.vol/1e6).toFixed(4)} MB，傳輸 ${r.tx.toFixed(5)} ms，可重疊 ${r.win.toFixed(3)} ms，停頓 ${r.stall===0?'0':r.stall.toFixed(5)} ms`);
}
function initC5(){
  segInit('c5-mode',v=>{c5.mode=v;renderC5()});segInit('c5-b',v=>{c5.bytes=+v;renderC5()});
  $('#c5-l').addEventListener('input',e=>{c5.ell=+e.target.value;renderC5()});
  $('#c5-n').addEventListener('input',e=>{c5.tokens=2**+e.target.value;e.target.setAttribute('aria-valuetext',`${c5.tokens} 個 token`);renderC5()});
  $('#c5-bw').addEventListener('input',e=>{c5.bw=+e.target.value;renderC5()});
  $('#c5-t').addEventListener('input',e=>{c5.tl=+e.target.value;renderC5()});
  $('#c5-sz').addEventListener('input',e=>{c5.size=+e.target.value;renderC5()});
  $('#c5-n').setAttribute('aria-valuetext','512 個 token');
  renderC5()}

/* ---------- timeline ---------- */
const TL_GROUP={'1607.04606':['ng','ngram-hash-embedding-lineage'],'1709.03933':['ng','ngram-hash-embedding-lineage'],'2207.06366':['ng','ngram-hash-embedding-lineage'],'2501.16975':['ng','ngram-hash-embedding-lineage'],'2502.01637':['ng','ngram-hash-embedding-lineage'],
  '1701.06538':['moe','moe-conditional-computation-lineage'],'2101.03961':['moe','moe-conditional-computation-lineage'],'2401.06066':['moe','moe-conditional-computation-lineage'],
  '1907.05242':['pm','sparse-memory-layers-pkm-peer'],'2407.04153':['pm','sparse-memory-layers-pkm-peer'],
  '2012.14913':['ffn','ffn-as-key-value-memory'],'2002.08909':['rag','retrieval-augmented-external-memory'],'2601.07372':['me','overview']};
const TL_NAME={ng:'N-gram 與雜湊嵌入',moe:'MoE',pm:'參數式記憶',ffn:'FFN 作為鍵值記憶',rag:'檢索增強',me:'條件記憶 · 本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  if(!data.some(e=>e.arxiv==='2601.07372'))data.push({year:2026,title:'Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models',authors_short:'本論文',arxiv:'2601.07372',role:'以 N-gram 雜湊查表實作條件記憶，並研究記憶與 MoE 專家的分配。'});
  data=data.map((e,i)=>({e,i})).sort((a,b)=>(+a.e.year||0)-(+b.e.year||0)||a.i-b.i).map(o=>o.e);
  const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];used.add(g[0]);const slug=g[1]==='overview'?'engram-lineage-overview':g[1];const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${slug?`<a href="${slug}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
  const on=new Set(used);
  Object.keys(TL_NAME).filter(k=>used.has(k)).forEach(k=>{const b=document.createElement('button');b.type='button';b.className='chip';b.setAttribute('aria-pressed','true');b.textContent=TL_NAME[k].split(' · ')[0];b.dataset.g=k;
    b.addEventListener('click',()=>{if(on.has(k)&&on.size===used.size){on.clear();on.add(k)}else if(on.has(k)){on.delete(k);if(!on.size)used.forEach(x=>on.add(x))}else on.add(k);
      $$('.chip',chips).forEach(c=>c.setAttribute('aria-pressed',String(on.has(c.dataset.g))));$$('.ev',track).forEach(ev=>ev.classList.toggle('dim',!on.has(ev.dataset.g)))});chips.appendChild(b)});
  const rail=$('#rail'),by=dir=>rail.scrollBy({left:dir*Math.min(rail.clientWidth*.8,560),behavior:RM?'auto':'smooth'});
  $('#tl-prev').addEventListener('click',()=>by(-1));$('#tl-next').addEventListener('click',()=>by(1));
  rail.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){by(1);e.preventDefault()}if(e.key==='ArrowLeft'){by(-1);e.preventDefault()}})}

/* ---------- toc, theme, appbar ---------- */
function initToc(){const secs=$$('[data-toc]'),ol=$('#toc'),bar=$('#tocbar');
  secs.forEach(s=>{ol.insertAdjacentHTML('beforeend',`<li><a href="#${s.id}"><span class="n">${s.dataset.toc}</span><span>${s.dataset.tt}</span></a></li>`);bar.insertAdjacentHTML('beforeend',`<a href="#${s.id}">${s.dataset.toc} ${s.dataset.tt}</a>`)});
  $$('.toc a,.tocbar a').forEach(a=>a.addEventListener('click',e=>{const t=$(a.getAttribute('href'));if(!t)return;e.preventDefault();t.scrollIntoView({behavior:RM?'auto':'smooth'});history.replaceState(null,'',a.getAttribute('href'))}));
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;$$('.toc a,.tocbar a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+id));const ab=$(`.tocbar a[href="#${id}"]`);if(ab&&bar.offsetParent)bar.scrollTo({left:ab.offsetLeft-16,behavior:RM?'auto':'smooth'})}})},{rootMargin:'-40% 0px -55% 0px'});
  secs.forEach(s=>io.observe(s));
  addEventListener('scroll',()=>$('#appbar').classList.toggle('raised',scrollY>8),{passive:true})}
function initTheme(){let saved=null;try{saved=localStorage.getItem('pm-theme')}catch(e){}if(saved)document.documentElement.dataset.theme=saved;
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){}})}
function markOverflow(){$$('.eq').forEach(e=>{const k=$('.katex-display',e)||$('.eq-m',e);const o=k.scrollWidth>k.clientWidth+1;e.classList.toggle('is-overflow',o);
  if(o){k.tabIndex=0;k.setAttribute('aria-label','公式，可橫向捲動')}else{k.removeAttribute('tabindex');k.removeAttribute('aria-label')}})}

/* ---------- self test (design.md Sec. 5) ---------- */
async function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t)=>Math.abs(a-b)<=t+1e-12,rel=(a,b,t)=>Math.abs(a-b)<=t*Math.abs(b)+1e-300,eqA=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
  ok(eqA(MULTS.map(String),['572152369','708123','479473029']),'MULTS');
  const sn=new Set(),P2=nextPrimes(10000,8,sn),P3=nextPrimes(10000,8,sn);
  ok(eqA(P2,[10007,10009,10037,10039,10061,10067,10069,10079])&&eqA(P3,[10091,10093,10099,10103,10111,10133,10139,10141]),'primes M=10000 K=8');
  let tk=tokenize(FIG1),cp=compress(tk);
  ok(tk.length===10&&eqA(cp.rawIds,[1,2,3,4,5,6,3,7,8,9])&&eqA(cp.canonIds,[1,2,3,4,5,6,3,7,8,9])&&cp.V===9&&cp.Vp===9,'Fig.1 compress');
  let h=hashHeads(cp.canonIds,3,2,P2);ok(h.mix===2286486165n&&eqA(h.idx,[6749,178,7380,3525,3183,8723,7576,4541]),'Fig.1 t=3 2-gram');
  h=hashHeads(cp.canonIds,3,3,P3);ok(h.mix===2975947167n&&eqA(h.idx,[266,5931,4144,7487,6870,6663,8721,9871]),'Fig.1 t=3 3-gram');
  tk=tokenize(C1TXT[1]);cp=compress(tk);
  ok(eqA(cp.rawIds,[1,2,3,4,5,6,7,8,9,4,10,6])&&eqA(cp.canonIds,[1,2,3,1,2,4,1,5,6,1,5,4])&&cp.V===10&&cp.Vp===6,'compress example |V|=10 |V\'|=6');
  ok(eqA(hashHeads(cp.canonIds,3,2,P2).idx,[7135,2468,1972,8016,7241,7069,3746,7949]),'example t=3 2-gram');
  ok(eqA(hashHeads(cp.canonIds,3,3,P3).idx,[6159,6978,9651,1510,5858,519,5334,7011]),'example t=3 3-gram');
  [[2e4,10007,1,.864475,.864475,.301,.301],[2e4,10007,2,.864475,.747317,-3.700,-3.700],[2e4,10007,8,.864475,.311904,-27.701,-10.148],[1e6,2262400,1,.357256,.357256,-.355,-.355],
   [1e6,2262400,8,.357256,2.65358e-4,-44.837,-8.449],[1e9,2262400,8,1,1,-41.837,-5.449],[1e10,2262400,8,1,1,-40.837,-4.449]].forEach(([U,M,K,p1,pa,li,lf])=>{const r=collide(U,M,K);
    ok(near(r.p1,p1,5e-7)&&(pa<1e-3?rel(r.pAll,pa,2e-5):near(r.pAll,pa,5e-7))&&near(r.log10Ind,li,.001)&&near(r.log10Floor,lf,.001),`collide ${U} ${M} ${K}: ${r.p1} ${r.pAll} ${r.log10Ind} ${r.log10Floor}`)});
  for(const [U,M,K,c1v,cav,fu,s1,sa,sf] of [[20000,10000,2,17303,14905,2,'0.8651','0.7452','0.0001'],[20000,10000,8,17303,6155,0,'0.8651','0.3078','0.0000'],[5000,1000,4,4969,4844,0,'0.9938','0.9688','0.0000'],[20000,50000,8,6595,4,0,'0.3297','0.0002','0.0000']]){
    const e=await empirical(U,M,K);ok(e.c1===c1v&&e.cAll===cav&&e.full===fu,`empirical counts ${U},${M},${K}: ${e.c1} ${e.cAll} ${e.full}`);
    ok(frac(e.c1,U)===s1&&frac(e.cAll,U)===sa&&frac(e.full,U)===sf,`empirical strings ${U},${M},${K}`)}
  [[16,.0625,1,false,.25,.56217650,.4378],[16,.0625,1,true,.5,.62245933,.3775],[2560,.3,1,false,15.1789,.99999974,2.558e-7],[2560,.3,1,true,3.8960,.98008205,1.992e-2],[2560,.05,1,false,2.5298,.92620620,7.379e-2],
   [2560,0,1,false,0,.5,.5],[16,.3,1,false,1.2,.76852478,.2315],[2560,.3,.1,false,1.5179,.82022805,.1798],[4096,1,2,false,128,1,2.572e-56]].forEach(([d,c,g,sq,s,a,o])=>{const r=gate(d,c,g,sq);
    ok(near(r.s,s,1e-4)&&near(r.a,a,1e-8)&&rel(r.oma,o,2e-3),`gate ${d} ${c} ${g} ${sq}: s=${r.s} a=${r.a} 1-a=${r.oma}`)});
  const rng=(a,b)=>Array.from({length:b-a+1},(_,i)=>a+i);
  [[9,3,4,3,false,rng(-2,9),10,12,2],[11,3,4,3,false,rng(0,11),12,12,0],[11,3,4,1,false,rng(6,11),6,6,0],[11,3,4,3,true,rng(9,11),3,3,0],[4,3,4,3,false,rng(-7,4),5,12,7],[13,3,6,5,false,null,9,28,9]].forEach(([t,N,w,d,z,list,ins,span,pad])=>{
    const f=field(t,N,w,d,z),fi=f.filter(p=>p>=0).length;ok((!list||eqA(f,list))&&fi===ins&&f[f.length-1]-f[0]+1===span&&f.length-fi===pad&&(list||f.length===18),`field ${t},${N},${w},${d},${z}: ${f}`)});
  const E={'2e20':[106,86,81,46],'6e20':[99,80.4,75.75,43.2],'27B':[72,58.8,55.5,32.4]};
  for(const p in E)[1,.8,.75,.4].forEach((r,i)=>ok(near(alloc(p,r).E,E[p][i],.01),`alloc E ${p} rho=${r}`));
  ok(near(alloc('2e20',1).Ps,5.132,.01)&&near(alloc('6e20',1).Ps,8.907,.01)&&near(alloc('27B',1).Ps,22.238,.01),'alloc Psparse');
  ok(near(alloc('2e20',.8).Pe,1.026,.01)&&near(alloc('6e20',.8).Pe,1.781,.01)&&near(alloc('2e20',.4).Pe,3.079,.01)&&near(alloc('6e20',.4).Pe,5.344,.01),'alloc PEngram');
  const a27=alloc('27B',49/66);ok(near(a27.E,55,.01)&&near(a27.Pe,5.728,.01),'alloc 27B fixed row');
  const a745=alloc('27B',.745);ok(near(a745.E,55.17,.01)&&near(a745.Pe,5.671,.01),'alloc 27B rho=0.745');
  const base={tokens:512,bytes:2,bw:25,tl:.5,ell:2,mode:'engram'};
  [[{},1.3107,.05243,.5,0],[{mode:'moe'},1.3107,.05243,0,.05243],[{tl:.01},1.3107,.05243,.01,.04243],[{tokens:65536},167.77,6.71089,.5,6.21089],[{tokens:65536,ell:15},167.77,6.71089,7,0],[{ell:1},1.3107,.05243,0,.05243]].forEach(([o,v,t,w,s],i)=>{
    const r=prefetch(Object.assign({},base,o));ok(near(r.vol/1e6,v,v>100?.01:1e-4)&&near(r.tx,t,1e-4)&&near(r.win,w,1e-4)&&near(r.stall,s,1e-4),`prefetch row ${i+1}: ${r.tx} ${r.stall}`)});
  ok(/0\.8651（17303 \/ 20000）/.test(empText({U:20000,c1:17303,cAll:6155,full:0,P0:10007},8).replace(/<[^>]+>/g,'')),'C1 empirical display string');
  ok(!!document.querySelector('.katex'),'KaTeX rendered');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ---------- boot ---------- */
initTheme();initToc();
const f1=tokenize(FIG1);staticBand('hero-band',f1,[1,2,3],[]);
staticBand('f7a',['<bos>','Only','Alexander','the','Great','could','tame','the','horse','B','uce','phal','us','.'],[2,3,4],[4]);
staticBand('f7b',['<bos>','By','the','way',',','I','am','a','fan','of','the','Milky','Way','.'],[1,2,3,10,11,12],[3,12]);
initC1();initC2();initC3();initC4();initC5();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderC1B();renderC2();renderC4();renderC5();markOverflow()})});
let mathDone=false;
function renderMath(){if(mathDone||typeof renderMathInElement!=='function')return;mathDone=true;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false});markOverflow()}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{renderMath();markOverflow();if(/[?&]test\b/.test(location.search))selfTest()});
})();
</script>
</body>
</html>
