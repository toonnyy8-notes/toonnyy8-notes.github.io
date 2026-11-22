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
  (o.yt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:(o.yf?o.yf(v):String(v)),yy=Y(val);el('line',{x1:x0,x2:x1,y1:yy,y2:yy,class:'grid'},g);tx(g,x0-8,yy+4,lab,{class:'tk','text-anchor':'end'})});
  (o.xt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:(o.xf?o.xf(v):String(v)),xx=X(val);el('line',{x1:xx,x2:xx,y1:y0,y2:y0+4,class:'axl'},g);tx(g,xx,y0+18,lab,{class:'tk','text-anchor':'middle'})});
  el('line',{x1:x0,x2:x1,y1:y0,y2:y0,class:'axl'},g);
  if(o.xl)tx(g,(x0+x1)/2,o.H-4,o.xl,{class:'al','text-anchor':'middle'});
  if(o.yl)tx(g,4,12,o.yl,{class:'al'});
  return {X,Y,g,x0,x1,y0,y1};
}
function niceTicks(max,n){const raw=max/n,p=Math.pow(10,Math.floor(Math.log10(raw))),st=[1,2,2.5,5,10].map(v=>v*p).find(v=>v>=raw);const out=[];for(let v=0;v<=max+1e-9;v+=st)out.push(+v.toFixed(6));return out}
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function pressGroup(btns,on){btns.forEach(b=>b.setAttribute('aria-pressed',String(b===on)))}
const sub=(s,a,b)=>`${s}<sub>${a}${b??''}</sub>`;

/* ---------- shared math (mirrors verify/common.py) ---------- */
function erf(x){const z=Math.abs(x),t=1/(1+.5*z);
  const r=t*Math.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+t*(.09678418+t*(-.18628806+t*(.27886807+t*(-1.13520398+t*(1.48851587+t*(-.82215223+t*.17087277)))))))));
  return x>=0?1-r:r-1}
const Phi=x=>.5*(1+erf(x*Math.SQRT1_2)),phi=x=>Math.exp(-.5*x*x)/Math.sqrt(2*Math.PI);
const gelu=x=>x*Phi(x),dgelu=x=>Phi(x)+x*phi(x);
const mv=(A,x)=>A.map(r=>r.reduce((s,w,j)=>s+w*x[j],0));
const nrm=v=>Math.sqrt(v.reduce((s,a)=>s+a*a,0));
function fwd(W,b,V,h){const z=mv(W,h).map((v,i)=>v+b[i]);return{z,f:mv(V,z.map(gelu))}}
function jnorms(V,z){return z.map((zi,i)=>nrm(V.map(r=>r[i]*dgelu(zi))))}
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
const pct=v=>(v>=0?'+':'−')+Math.abs(v*100).toFixed(1)+'%';
const fx=(v,d)=>(v<0?'−':'')+Math.abs(v).toFixed(d);

/* ---------- network drawing (S2a, C1) ---------- */
function drawNet(svg,sel){
  const W=280,H=178;svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const inp=[[34,52],[34,112]],hid=[[140,52],[140,112]],out=[[246,30],[246,82],[246,134]];
  const g=el('g',null,svg);
  hid.forEach(a=>out.forEach(b=>el('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],class:'faint'},g)));
  for(let i=0;i<2;i++)for(let j=0;j<2;j++){const on=sel&&sel[0]===i&&sel[1]===j;el('line',{x1:inp[j][0],y1:inp[j][1],x2:hid[i][0],y2:hid[i][1],class:on?'acc':'axl'},g)}
  if(sel){const [i,j]=sel,mx=(inp[j][0]+hid[i][0])/2,my=(inp[j][1]+hid[i][1])/2-8;const t=tx(g,mx,my,'W',{class:'labm','text-anchor':'middle'});const s=el('tspan',{'baseline-shift':'sub','font-size':'75%'},t);s.textContent=`${i+1}${j+1}`}
  const node=(p,lab,idx)=>{el('circle',{cx:p[0],cy:p[1],r:13,class:'hollow'},g);const t=tx(g,p[0],p[1]+4,lab,{class:'labm','text-anchor':'middle'});const s=el('tspan',{'baseline-shift':'sub','font-size':'70%'},t);s.textContent=idx};
  inp.forEach((p,k)=>node(p,'h',k+1));hid.forEach((p,k)=>node(p,'z',k+1));out.forEach((p,k)=>node(p,'f',k+1));
  tx(g,34,H-8,'輸入 h',{class:'al','text-anchor':'middle'});tx(g,140,H-8,'z → GELU',{class:'al','text-anchor':'middle'});tx(g,246,H-8,'輸出 f',{class:'al','text-anchor':'middle'});
}
function renderGelu(){const box=$('#gelu-fig');const W=widthOf(box,280,640),H=230;
  const F=frame($('svg',box),{W,H,m:{l:40,r:12,t:14,b:36},x:[-4,3],y:[-.5,3],xt:[-4,-3,-2,-1,0,1,2,3],yt:[0,1,2,3],xl:'z',yf:v=>String(v)});
  const p1=[],p2=[];for(let k=0;k<=280;k++){const z=-4+7*k/280;p1.push([F.X(z),F.Y(gelu(z))]);p2.push([F.X(z),F.Y(dgelu(z))])}
  el('line',{x1:F.x0,x2:F.x1,y1:F.Y(0),y2:F.Y(0),class:'axl'},F.g);
  el('path',{d:lp(p1),class:'c1'},F.g);el('path',{d:lp(p2),class:'c2 dash'},F.g);
  el('circle',{cx:F.X(-.7518),cy:F.Y(0),r:4,class:'ink-f'},F.g);tx(F.g,F.X(-.7518)+6,F.Y(0)-30,'GELU′ = 0（z ≈ −0.75）',{class:'al'});
  el('line',{x1:F.X(-.7518)+4,y1:F.Y(0)-26,x2:F.X(-.7518),y2:F.Y(0)-5,class:'cur'},F.g);
  el('circle',{cx:F.X(-1.4142),cy:F.Y(dgelu(-1.4142)),r:4,class:'hollow'},F.g);tx(F.g,F.X(-1.4142),F.Y(dgelu(-1.4142))+20,'GELU′ < 0',{class:'al','text-anchor':'middle'});
  tx(F.g,F.X(2.2),F.Y(gelu(2.2))-8,'GELU',{class:'lab','text-anchor':'end'});tx(F.g,F.X(2.6),F.Y(1)-8,'GELU′',{class:'lab','text-anchor':'middle'})}

/* ---------- C1 ---------- */
const C1={W:[[1,-2],[.5,3]],x:[1,.5],V:[[1,2],[0,-1],[2,0]],b:[0,0]};
const c1base=fwd(C1.W,C1.b,C1.V,C1.x),c1J=jnorms(C1.V,c1base.z);
function c1true(i,j,dw){const W=C1.W.map(r=>r.slice());W[i][j]+=dw;const f1=fwd(W,C1.b,C1.V,C1.x).f;return nrm(f1.map((v,k)=>v-c1base.f[k]))}
const c1pred=(i,j,dw)=>c1J[i]*Math.abs(dw)*Math.abs(C1.x[j]);
function c1ymax(i,j){let m=0;for(let k=-300;k<=300;k++){const d=k/100;m=Math.max(m,c1true(i,j,d),c1pred(i,j,d))}return m*1.08}
const c1={i:0,j:0,dw:.1,ymax:0,curve:null};
function c1chip(){c1.ymax=c1ymax(c1.i,c1.j);c1.curve=[];for(let k=-300;k<=300;k++){const d=k/100;c1.curve.push([d,c1true(c1.i,c1.j,d)])}}
function renderC1(){const box=$('#c1-fig'),W=widthOf(box,280,600),H=300,{i,j,dw}=c1;
  const yt=niceTicks(c1.ymax,4).filter(v=>v<=c1.ymax);
  const F=frame($('svg',box),{W,H,m:{l:48,r:14,t:28,b:40},x:[-3,3],y:[0,c1.ymax],xt:[-3,-2,-1,0,1,2,3],xf:v=>v<0?'−'+(-v):String(v),yt,yf:v=>v.toFixed(v<10&&v%1?1:0),xl:'ΔW',yl:'‖Δf‖'});
  const z0=-C1.W[i][j];el('line',{x1:F.X(z0),x2:F.X(z0),y1:F.y0,y2:F.y0-8,class:'acc'},F.g);tx(F.g,F.X(z0),F.y0-12,'歸零',{class:'al','text-anchor':'middle'});
  el('path',{d:lp(c1.curve.map(p=>[F.X(p[0]),F.Y(p[1])])),class:'c1'},F.g);
  el('path',{d:lp([[-3,c1pred(i,j,-3)],[0,0],[3,c1pred(i,j,3)]].map(p=>[F.X(p[0]),F.Y(p[1])])),class:'c2 dash'},F.g);
  const tv=c1true(i,j,dw),pv=c1pred(i,j,dw);
  el('line',{x1:F.X(dw),x2:F.X(dw),y1:F.y0,y2:F.y1,class:'cur'},F.g);
  el('circle',{cx:F.X(dw),cy:F.Y(tv),r:5,class:'c1-f'},F.g);el('rect',{x:F.X(dw)-5,y:F.Y(pv)-5,width:10,height:10,class:'hollow2'},F.g);
  const lx=F.X(2.4);tx(F.g,lx,F.Y(c1true(i,j,2.4))-8,'真實',{class:'lab','text-anchor':'middle'});tx(F.g,F.X(-2.4),F.Y(c1pred(i,j,-2.4))-8,'一階預測',{class:'lab','text-anchor':'middle'});
  $('svg',box).setAttribute('aria-label',`W${i+1}${j+1}，ΔW = ${fx(dw,2)}：一階預測 ${pv.toFixed(5)}，真實 ${tv.toFixed(5)}`);
  drawNet($('#c1-net svg'),[i,j]);
  const S=c1J[i]*Math.abs(C1.W[i][j])*Math.abs(C1.x[j]),rel=tv>1e-12?(pv-tv)/tv:null;
  $('#c1-ro').innerHTML=`z${'<sub>'+(i+1)+'</sub>'} = <b>${fx(c1base.z[i],5)}</b>　GELU′(z${'<sub>'+(i+1)+'</sub>'}) = <b>${dgelu(c1base.z[i]).toFixed(5)}</b><br>‖J${'<sub>'+(i+1)+'</sub>'}‖ = <b>${c1J[i].toFixed(5)}</b>　|h${'<sub>'+(j+1)+'</sub>'}| = <b>${Math.abs(C1.x[j]).toFixed(5)}</b>　S = <b>${S.toFixed(5)}</b><br>一階預測 = <b>${pv.toFixed(5)}</b>　真實 = <b class="acc">${tv.toFixed(5)}</b><br>相對誤差 = <b>${rel===null?'—':pct(rel)}</b>`;
  $('#c1-dwo').textContent=fx(dw,2);$('#c1-dw').setAttribute('aria-valuetext',`ΔW = ${fx(dw,2)}`);
  announce('c1',`W${i+1}${j+1}，ΔW ${fx(dw,2)}：預測 ${pv.toFixed(5)}，真實 ${tv.toFixed(5)}，誤差 ${rel===null?'無':pct(rel)}`)}
function initC1(){const g=$('#c1-w');
  for(let i=0;i<2;i++)for(let j=0;j<2;j++){const b=document.createElement('button');b.type='button';b.className='chip';b.innerHTML=sub('W',i+1,j+1);b.dataset.i=i;b.dataset.j=j;b.setAttribute('aria-pressed',String(!i&&!j));
    b.addEventListener('click',()=>{c1.i=i;c1.j=j;pressGroup($$('button',g),b);c1chip();renderC1()});g.appendChild(b)}
  $('#c1-dw').addEventListener('input',e=>{c1.dw=+e.target.value;renderC1()});
  $('#c1-zero').addEventListener('click',()=>{c1.dw=-C1.W[c1.i][c1.j];$('#c1-dw').value=c1.dw;renderC1()});
  const tb=$('#c1-s tbody');for(let i=0;i<2;i++){const r=document.createElement('tr');r.innerHTML=`<th>i = ${i+1}</th>`+[0,1].map(j=>`<td class="num">${(c1J[i]*Math.abs(C1.W[i][j])*Math.abs(C1.x[j])).toFixed(5)}</td>`).join('');tb.appendChild(r)}
  c1chip();renderC1()}

/* ---------- C2 ---------- */
const C2={H:[1,-.2,1.5,.6],W:[[.8,-1.5,.3,.6],[-.4,2,-.9,.2],[1.2,.5,-.3,-1],[.1,-.7,.6,1.4]],V:[[1,-.5,.8,.3],[.4,1.2,-.6,.9],[-.7,.3,1.1,-.4]]};
const EPS=1e-9,skey=s=>Math.round(s/EPS);
const c2b=b3=>[0,0,b3,0];
function c2scores(b3){const {z,f}=fwd(C2.W,c2b(b3),C2.V,C2.H),J=jnorms(C2.V,z),mag=[],wanda=[],full=[];
  for(let i=0;i<4;i++)for(let j=0;j<4;j++){const w=Math.abs(C2.W[i][j]),h=Math.abs(C2.H[j]);mag.push(w);wanda.push(w*h);full.push(J[i]*w*h)}
  return{z,f,J,mag,wanda,full}}
function lowest(s,m){return new Set(s.map((v,k)=>[skey(v),k]).sort((a,b)=>a[0]-b[0]||a[1]-b[1]).slice(0,m).map(p=>p[1]))}
function zeroShift(b3,upd,f0){const W=C2.W.map(r=>r.slice());upd.forEach(k=>{W[k>>2][k&3]=0});const f1=fwd(W,c2b(b3),C2.V,C2.H).f;return nrm(f1.map((v,k)=>v-f0[k]))}
function ranks(s){const ks=s.map(skey),o=ks.map((v,k)=>k).sort((a,b)=>ks[a]-ks[b]),r=new Array(s.length);let i=0;
  while(i<o.length){let j=i;while(j+1<o.length&&ks[o[j+1]]===ks[o[i]])j++;for(let t=i;t<=j;t++)r[o[t]]=(i+j)/2+1;i=j+1}return r}
function spearman(a,b){const ra=ranks(a),rb=ranks(b),n=a.length,ma=ra.reduce((s,v)=>s+v,0)/n,mb=rb.reduce((s,v)=>s+v,0)/n;let num=0,da=0,db=0;
  for(let k=0;k<n;k++){num+=(ra[k]-ma)*(rb[k]-mb);da+=(ra[k]-ma)**2;db+=(rb[k]-mb)**2}return da>0&&db>0?num/Math.sqrt(da*db):null}
const c2cache=new Map();let c2tok=0;
function grayRun(b3,chunkSize,done){const key=b3.toFixed(2);if(c2cache.has(key)){done(c2cache.get(key));return}
  const tok=++c2tok,b=c2b(b3),base=fwd(C2.W,b,C2.V,C2.H),z=base.z.slice(),f0=base.f,on=new Uint8Array(16),acc=new Float64Array(17),cnt=new Float64Array(17);cnt[0]=1;let size=0,s=1;
  const a=new Float64Array(4);
  const chunk=()=>{if(tok!==c2tok)return;const end=Math.min(65536,s+chunkSize);
    for(;s<end;s++){const k=31-Math.clz32(s&-s),i=k>>2,j=k&3;on[k]^=1;z[i]+=(on[k]?-1:1)*C2.W[i][j]*C2.H[j];size+=on[k]?1:-1;
      for(let q=0;q<4;q++)a[q]=gelu(z[q]);let d=0;for(let r=0;r<3;r++){const V=C2.V[r];const fr=V[0]*a[0]+V[1]*a[1]+V[2]*a[2]+V[3]*a[3]-f0[r];d+=fr*fr}
      acc[size]+=Math.sqrt(d);cnt[size]++}
    if(s<65536)setTimeout(chunk,0);else{const means=Array.from(acc,(v,k)=>v/cnt[k]);c2cache.set(key,means);done(means)}};
  chunk()}
const c2={b3:0,m:4,foc:[0,0,0],rnd:null,rt:0};
const C2NAMES=[['mag','Magnitude',[0,1,0]],['wanda','Wanda',[0,1,1]],['full','Dowser',[1,1,1]]];
const FB=['‖J<sub>i</sub>‖','|W<sub>ij</sub>|','|h<sub>j</sub>|'];
function buildC2(){const box=$('#c2-maps');box.innerHTML='';
  C2NAMES.forEach(([key,name,fac],mi)=>{const d=document.createElement('div');d.className='hm-wrap';
    d.innerHTML=`<h4>${name}</h4><div class="factor-bar mini">${fac.map((o,q)=>`<span data-on="${o}"><b>${FB[q]}</b>${o?'用':'未用'}</span>`).join('<i>×</i>')}</div><p class="hm-upd" id="c2-u-${key}"></p><div class="hm" role="grid" aria-label="${name} 分數熱圖，方向鍵移動" id="c2-g-${key}"></div>`;
    box.appendChild(d);const g=$('.hm',d);
    for(let i=0;i<4;i++){const row=document.createElement('div');row.setAttribute('role','row');row.className='hm-row';
      for(let j=0;j<4;j++){const c=document.createElement('div');c.setAttribute('role','gridcell');c.className='hm-c';c.tabIndex=(i===0&&j===0)?0:-1;c.dataset.k=4*i+j;c.innerHTML='<span class="v"></span><span class="bar"></span><span class="kn"></span>';row.appendChild(c)}g.appendChild(row)}
    g.addEventListener('keydown',e=>{const mv={ArrowRight:1,ArrowLeft:-1,ArrowDown:4,ArrowUp:-4}[e.key];if(mv===undefined)return;e.preventDefault();
      let k=c2.foc[mi]+mv;if(k<0||k>15)return;if(Math.abs(mv)===1&&(k>>2)!==(c2.foc[mi]>>2))return;
      const cs=$$('.hm-c',g);cs[c2.foc[mi]].tabIndex=-1;cs[k].tabIndex=0;cs[k].focus();c2.foc[mi]=k})})}
function renderC2(){const {b3,m}=c2,S=c2scores(b3),masks={};
  C2NAMES.forEach(([key,name])=>{const s=S[key],mx=Math.max(...s),U=lowest(s,m);masks[key]=U;
    $$('#c2-g-'+key+' .hm-c').forEach((c,k)=>{const i=k>>2,j=k&3,v=Math.round(s[k]/EPS)*EPS,up=U.has(k);
      c.classList.toggle('upd',up);$('.v',c).textContent=v.toFixed(3);$('.bar',c).style.width=(mx>0?v/mx*100:0)+'%';$('.kn',c).textContent=k+1;
      c.setAttribute('aria-label',`第 ${i+1} 列第 ${j+1} 欄（格 ${k+1}），分數 ${v.toFixed(3)}，${up?'可更新':'凍結'}`)});
    $('#c2-u-'+key).textContent='可更新格子：'+[...U].sort((a,b)=>a-b).map(k=>k+1).join(', ')});
  const tb=$('#c2-tab tbody');tb.innerHTML='';
  C2NAMES.forEach(([key,name])=>{const U=masks[key],real=zeroShift(b3,U,S.f),ss=[...U].reduce((t,k)=>t+S.full[k],0);
    tb.insertAdjacentHTML('beforeend',`<tr${key==='full'?' class="hl"':''}><td>${name}</td><td class="mono">${[...U].sort((a,b)=>a-b).map(k=>k+1).join(', ')}</td><td class="num">${real.toFixed(4)}</td><td class="num">${ss.toFixed(4)}</td></tr>`)});
  const key=b3.toFixed(2),rnd=c2cache.get(key);
  tb.insertAdjacentHTML('beforeend',`<tr><td>隨機（平均）</td><td class="mono">所有 C(16, ${m}) 個子集</td><td class="num">${rnd?rnd[m].toFixed(4):'計算中'}</td><td class="num">—</td></tr>`);
  const sm=spearman(S.full,S.mag),sw=spearman(S.full,S.wanda),f=v=>v===null?'—':v.toFixed(3);
  $('#c2-ro').innerHTML=`z = (${S.z.map(v=>fx(v,2)).join(', ')})　GELU′(z) = (${S.z.map(v=>fx(dgelu(v),4)).join(', ')})<br>‖J<sub>i</sub>‖ = (${S.J.map(v=>v.toFixed(4)).join(', ')})<br>Spearman(Dowser, Magnitude) = <b>${f(sm)}</b>　Spearman(Dowser, Wanda) = <b>${f(sw)}</b>`;
  $('#c2-bo').textContent=fx(b3,2);$('#c2-b').setAttribute('aria-valuetext',`b₃ = ${fx(b3,2)}`);
  $('#c2-mo').textContent=`${m}（ρ = ${(m/16).toFixed(3)}）`;$('#c2-m').setAttribute('aria-valuetext',`m = ${m}，ρ = ${(m/16).toFixed(3)}`);
  const rf=zeroShift(b3,masks.full,S.f);
  announce('c2',`b₃ ${fx(b3,2)}，m ${m}：Dowser 遮罩歸零位移 ${rf.toFixed(4)}，一階總和 ${[...masks.full].reduce((t,k)=>t+S.full[k],0).toFixed(4)}`);
  $$('#c2-pre .chip').forEach(c=>c.setAttribute('aria-pressed',String(Math.abs(+c.dataset.b-b3)<1e-9)))}
function scheduleRandom(){clearTimeout(c2.rt);const b3=c2.b3;c2.rt=setTimeout(()=>grayRun(b3,4096,()=>{if(c2.b3===b3)renderC2()}),150)}
function initC2(){buildC2();
  $('#c2-b').addEventListener('input',e=>{c2.b3=+(+e.target.value).toFixed(2);renderC2();scheduleRandom()});
  $('#c2-m').addEventListener('input',e=>{c2.m=+e.target.value;renderC2()});
  $$('#c2-pre .chip').forEach(c=>c.addEventListener('click',()=>{c2.b3=+c.dataset.b;$('#c2-b').value=c2.b3;renderC2();scheduleRandom()}));
  renderC2();scheduleRandom()}

/* ---------- C3 ---------- */
const C3PRE={'34':[3,4],'11':[1,1],'3':[3],'1x8':[1,1,1,1,1,1,1,1],'big':[3,.5,.5,.5,.5,.5,.5,.5]};
const C3R=[1,2,4,8,16,32,64];
const c3={a:[3,4],R:8,seed:1};
function exactX(a){const d=a.length,n=1<<d,dist=new Map();let s1=0,s2=0;
  for(let b=0;b<n;b++){let x=0;for(let k=0;k<d;k++)x+=((b>>k)&1?-1:1)*a[k];s2+=x*x;s1+=Math.abs(x);const kk=Math.round(x/EPS);dist.set(kk,(dist.get(kk)||0)+1/n)}
  const na=nrm(a),var2=2*(na**4-a.reduce((s,v)=>s+v**4,0));
  return{e2:s2/n,e1:s1/n,na,var2,dist:[...dist].map(([k,p])=>[k*EPS,p]).sort((u,v)=>u[0]-v[0])}}
function mcX(a,R,seed){const r=mulberry32(seed),xs=[];for(let q=0;q<R;q++){let x=0;for(const v of a)x+=(r()<.5?-1:1)*v;xs.push(x)}
  return{xs,l2:Math.sqrt(xs.reduce((s,x)=>s+x*x,0)/R),l1:xs.reduce((s,x)=>s+Math.abs(x),0)/R}}
function repeatX(a,R){const L2=[],L1=[];for(let s=1;s<=200;s++){const m=mcX(a,R,s);L2.push(m.l2);L1.push(m.l1)}
  const mean=v=>v.reduce((s,x)=>s+x,0)/v.length,sd=v=>{const m=mean(v);return Math.sqrt(v.reduce((s,x)=>s+(x-m)**2,0)/(v.length-1))};
  return{m2:mean(L2),s2:sd(L2),m1:mean(L1),s1:sd(L1)}}
function buildComps(){const box=$('#c3-comps');box.innerHTML='';
  c3.a.forEach((v,k)=>{const d=document.createElement('div');d.className='rng';d.innerHTML=`<label class="lbl" for="c3-a${k}"><span>分量 a<sub>${k+1}</sub></span><output id="c3-ao${k}">${fx(v,2)}</output></label><input id="c3-a${k}" type="range" min="-3" max="3" step="0.05" value="${v}">`;
    box.appendChild(d);$('input',d).addEventListener('input',e=>{c3.a[k]=+e.target.value;$('#c3-ao'+k).textContent=fx(c3.a[k],2);e.target.setAttribute('aria-valuetext',`a${k+1} = ${fx(c3.a[k],2)}`);renderC3()})});
  $$('#c3-d button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.v===c3.a.length)))}
function renderC3(){const {a,R,seed}=c3,E=exactX(a),M=mcX(a,R,seed),Rp=repeatX(a,R);
  // histogram
  const hb=$('#c3-hist'),W=widthOf(hb,280,560),H=190,xs=E.dist.map(p=>p[0]),lo=Math.min(-1,...xs),hi=Math.max(1,...xs),pad=(hi-lo)*.08||1;
  const pm=Math.max(...E.dist.map(p=>p[1]));
  const F=frame($('svg',hb),{W,H,m:{l:40,r:12,t:22,b:46},x:[lo-pad,hi+pad],y:[0,pm*1.15],yt:niceTicks(pm*1.1,3),yf:v=>v.toFixed(2),xl:'X = ξᵀa（精確分布）'});
  const bw=Math.max(2,Math.min(28,(F.x1-F.x0)/(E.dist.length*1.6)));
  let gapMin=Infinity;for(let q=1;q<E.dist.length;q++)gapMin=Math.min(gapMin,F.X(E.dist[q][0])-F.X(E.dist[q-1][0]));const lab=E.dist.length<=9&&gapMin>=44;
  E.dist.forEach(([x,p])=>{el('rect',{x:F.X(x)-bw/2,y:F.Y(p),width:bw,height:F.y0-F.Y(p),class:'barc'},F.g);
    if(lab){tx(F.g,F.X(x),F.Y(p)-5,p.toFixed(3),{class:'tk','text-anchor':'middle'});tx(F.g,F.X(x),F.y0+16,fx(x,2),{class:'tk','text-anchor':'middle'})}});
  if(!lab)[lo,0,hi].forEach(v=>tx(F.g,F.X(v),F.y0+16,fx(v,2),{class:'tk','text-anchor':'middle'}));
  M.xs.forEach(x=>el('path',{d:`M${F.X(x)} ${F.y0+22}l-4 7h8z`,class:'acc-f'},F.g));
  tx(F.g,F.x1,F.y0+40,`▲ 本次 ${R} 個樣本`,{class:'al','text-anchor':'end'});
  $('svg',hb).setAttribute('aria-label',`X 的精確分布，共 ${E.dist.length} 個值；E[X²] = ${E.e2.toFixed(4)}`);
  // number line
  const lb=$('#c3-line'),Wl=widthOf(lb,280,560),top=Math.max(E.na,M.l2,M.l1,.5)*1.12;
  const svg=$('svg',lb);svg.textContent='';
  const marks=[[Math.SQRT1_2*E.na,'√½‖a‖','brk'],[E.na,'‖a‖','brk'],[E.e1,'E|X|','dia'],[M.l2,'L2 估計','cir'],[M.l1,'L1 估計','sq']];
  const X=lin(0,top,20,Wl-20),rows=[],lay=marks.map(m=>({m,x:X(m[0]),w:m[1].length*7+30})).sort((p,q)=>p.x-q.x);
  lay.forEach(o=>{let r=0;while(rows[r]!==undefined&&o.x-o.w/2<rows[r]+6)r++;rows[r]=o.x+o.w/2;o.r=r});
  const nr=Math.max(1,rows.length),Hl=70+nr*18;svg.setAttribute('viewBox',`0 0 ${Wl} ${Hl}`);svg.setAttribute('width',Wl);svg.setAttribute('height',Hl);
  const g=el('g',null,svg),yA=Hl-30;
  el('line',{x1:20,x2:Wl-20,y1:yA,y2:yA,class:'axl'},g);[0,top/2,top].forEach(v=>tx(g,X(v),yA+18,v.toFixed(2),{class:'tk','text-anchor':'middle'}));
  el('rect',{x:X(Math.SQRT1_2*E.na),y:yA-6,width:Math.max(0,X(E.na)-X(Math.SQRT1_2*E.na)),height:12,class:'band'},g);
  lay.forEach(o=>{const [v,lab,kind]=o.m,x=o.x,ly=yA-22-o.r*18;
    if(kind==='brk')el('line',{x1:x,x2:x,y1:yA-9,y2:yA+9,class:'ink'},g);
    else if(kind==='dia')el('path',{d:`M${x} ${yA-7}l7 7l-7 7l-7-7z`,class:'ink-f'},g);
    else if(kind==='cir')el('circle',{cx:x,cy:yA,r:5,class:'hollow2'},g);
    else el('rect',{x:x-5,y:yA-5,width:10,height:10,class:'hollow3'},g);
    el('line',{x1:x,x2:x,y1:ly+3,y2:yA-8,class:'cur'},g);tx(g,x,ly,lab,{class:'lab','text-anchor':'middle'})});
  svg.setAttribute('aria-label',`‖a‖ = ${E.na.toFixed(4)}，E|X| = ${E.e1.toFixed(4)}，L2 估計 ${M.l2.toFixed(4)}，L1 估計 ${M.l1.toFixed(4)}`);
  const ratio=E.na>0?(E.e1/E.na).toFixed(4):'—';
  $('#c3-ro').innerHTML=`精確：‖a‖² = <b>${(E.na*E.na).toFixed(4)}</b>　E[X²] = <b>${E.e2.toFixed(4)}</b><br>E|X| = <b>${E.e1.toFixed(4)}</b>　‖a‖ = <b>${E.na.toFixed(4)}</b>　E|X|/‖a‖ = <b class="acc">${ratio}</b><br>Var[X²] = <b>${E.var2.toFixed(4)}</b><br>本次（R = ${R}）：L2 = <b>${M.l2.toFixed(4)}</b>　L1 = <b>${M.l1.toFixed(4)}</b><br>200 次重複：L2 <b>${Rp.m2.toFixed(4)} ± ${Rp.s2.toFixed(4)}</b>　L1 <b>${Rp.m1.toFixed(4)} ± ${Rp.s1.toFixed(4)}</b>`;
  $('#c3-seed').textContent='seed = '+seed;
  announce('c3',`E|X|/‖a‖ = ${ratio}；L2 估計 ${M.l2.toFixed(4)}；L1 估計 ${M.l1.toFixed(4)}`)}
function initC3(){const dg=$('#c3-d');for(let d=1;d<=8;d++){const b=document.createElement('button');b.type='button';b.textContent=d;b.dataset.v=d;b.setAttribute('aria-label','d = '+d);
    b.addEventListener('click',()=>{const a=c3.a.slice(0,d);while(a.length<d)a.push(1);c3.a=a;buildComps();renderC3()});dg.appendChild(b)}
  const rg=$('#c3-R');C3R.forEach(R=>{const b=document.createElement('button');b.type='button';b.textContent=R;b.setAttribute('aria-label','R = '+R);b.setAttribute('aria-pressed',String(R===c3.R));
    b.addEventListener('click',()=>{c3.R=R;pressGroup($$('button',rg),b);renderC3()});rg.appendChild(b)});
  $$('#c3-pre .chip').forEach(c=>{c.setAttribute('aria-pressed',String(c.dataset.p==='34'));c.addEventListener('click',()=>{c3.a=C3PRE[c.dataset.p].slice();pressGroup($$('#c3-pre .chip'),c);buildComps();renderC3()})});
  $('#c3-re').addEventListener('click',()=>{c3.seed++;renderC3()});
  buildComps();renderC3()}

/* ---------- C4 (data: Tables 8/9, verbatim from verify/c4_tables.py) ---------- */
const C4_RHO=[.1,.25,.5,.75,.9];
const C4_TABLE={
 'NVILA|ImageNet-R':{full:[13.8,92.3,53.1,24.0],rand:[[54.5,90.7,72.6,68.0,.8],[49.0,91.5,70.2,63.8,.5],[39.2,92.2,65.7,55.0,.8],[25.6,92.6,59.1,40.1,5.5],[20.4,92.6,56.5,33.4,1.9]],ours:[[56.6,90.5,73.6,69.7],[53.0,91.7,72.4,67.2],[47.5,92.2,69.8,62.7],[35.5,92.5,64.0,51.3],[24.9,92.6,58.7,39.2]]},
 'LLaVA|ImageNet-R':{full:[9.4,89.8,49.6,17.1],rand:[[54.6,88.5,71.6,67.6,.1],[43.3,89.3,66.3,58.3,.2],[20.4,90.0,55.2,33.2,.5],[12.0,89.8,50.9,21.2,.3],[9.8,89.7,49.8,17.6,.2]],ours:[[55.8,88.6,72.2,68.5],[48.2,89.5,68.9,62.7],[27.9,89.7,58.8,42.6],[13.6,89.7,51.6,23.6],[10.2,89.9,50.1,18.3]]},
 'NVILA|COCO':{full:[12.9,98.2,55.6,22.9],rand:[[62.3,134.8,98.6,85.2,.1],[59.9,124.3,92.1,80.8,.3],[44.8,105.2,75.0,62.9,1.4],[27.1,98.6,62.8,42.5,3.4],[19.7,97.8,58.7,32.7,.9]],ours:[[62.7,135.1,98.9,85.7],[61.4,123.8,92.6,82.1],[55.2,106.4,80.8,72.7],[36.7,98.5,67.6,53.5],[22.9,97.7,60.3,37.1]]},
 'LLaVA|COCO':{full:[4.5,101.5,53.0,8.5],rand:[[58.8,120.8,89.8,79.1,.2],[55.9,105.2,80.6,73.0,.3],[33.1,100.5,66.8,49.8,3.5],[11.0,100.5,55.8,19.9,4.2],[6.3,100.5,53.4,11.9,2.7]],ours:[[59.0,120.8,89.9,79.3],[57.3,104.5,80.9,74.0],[39.2,100.8,70.0,56.5],[13.4,100.9,57.2,23.7],[6.7,101.3,54.0,12.5]]}};
const C4_AUP_SD={'NVILA|ImageNet-R':[1.0,.7,.9,4.4,1.4],'LLaVA|ImageNet-R':[.1,.2,.3,.2,.2],'NVILA|COCO':[.1,.4,1.4,2.7,.7],'LLaVA|COCO':[.3,.2,3.0,2.5,1.5]};
const C4_KEYS=[['NVILA|ImageNet-R','NVILA · ImageNet-R'],['LLaVA|ImageNet-R','LLaVA · ImageNet-R'],['NVILA|COCO','NVILA · COCO'],['LLaVA|COCO','LLaVA · COCO']];
const hs=(u,d)=>u+d>0?2*u*d/(u+d):0,avg=(u,d)=>(u+d)/2;
const c4={key:'NVILA|ImageNet-R',met:'H',col:0};
function c4vals(){const t=C4_TABLE[c4.key],H=c4.met==='H';
  return{ours:t.ours.map(o=>H?o[3]:o[0]),rand:t.rand.map((r,q)=>H?[r[3],r[4]]:[r[0],C4_AUP_SD[c4.key][q]]),full:H?t.full[3]:t.full[0]}}
function renderC4(){const box=$('#c4-fig'),W=widthOf(box,280,620),H=300,v=c4vals(),mn=c4.met==='H'?'H-score':'A_up';
  const F=frame($('svg',box),{W,H,m:{l:40,r:16,t:28,b:40},x:[-.5,5.5],y:[0,100],xt:[0,1,2,3,4,5].map(q=>[q,q<5?String(C4_RHO[q]):'1.0']),yt:[0,25,50,75,100],xl:'可更新比例 ρ',yl:mn});
  const cw=F.X(1)-F.X(0);
  el('rect',{x:F.X(c4.col)-cw/2,y:F.y1,width:cw,height:F.y0-F.y1,class:'colsel'},F.g);
  el('path',{d:lp(v.rand.map((r,q)=>[F.X(q),F.Y(r[0])])),class:'c2 dash'},F.g);
  v.rand.forEach((r,q)=>{const x=F.X(q);el('line',{x1:x,x2:x,y1:F.Y(r[0]-r[1]),y2:F.Y(r[0]+r[1]),class:'c2',style:'stroke-width:1.5'},F.g);el('rect',{x:x-5,y:F.Y(r[0])-5,width:10,height:10,class:'hollow3'},F.g)});
  el('path',{d:lp(v.ours.map((o,q)=>[F.X(q),F.Y(o)])),class:'c1'},F.g);
  v.ours.forEach((o,q)=>el('circle',{cx:F.X(q),cy:F.Y(o),r:5,class:'c1-f'},F.g));
  const sx=F.X(5),sy=F.Y(v.full);el('path',{d:`M${sx} ${sy-8}L${sx+2.4} ${sy-2.5}L${sx+7.6} ${sy-2.5}L${sx+3.4} ${sy+1}L${sx+5} ${sy+7}L${sx} ${sy+3.5}L${sx-5} ${sy+7}L${sx-3.4} ${sy+1}L${sx-7.6} ${sy-2.5}L${sx-2.4} ${sy-2.5}Z`,class:'ink-f'},F.g);
  tx(F.g,F.X(0)+8,F.Y(v.ours[0])-10,'重要性遮罩',{class:'lab'});
  [0,1,2,3,4,5].forEach(q=>{const r=el('rect',{x:F.X(q)-cw/2,y:F.y1,width:cw,height:F.y0-F.y1,class:'hit'},F.g);r.addEventListener('click',()=>{c4.col=q;renderC4()})});
  const c=c4.col;let txt;
  if(c<5){const o=v.ours[c],[rm,rs]=v.rand[c],gap=o-rm;txt=`ρ = <b>${C4_RHO[c]}</b>　${mn}<br>重要性遮罩 <b class="acc">${o.toFixed(1)}</b>　隨機 <b>${rm.toFixed(1)} ± ${rs.toFixed(1)}</b><br>差距 <b>${gap>=0?'+':'−'}${Math.abs(gap).toFixed(1)}</b>　差距 / sd = <b>${rs>0?(gap/rs).toFixed(1):'—'}</b>`}
  else txt=`ρ = <b>1.0</b>（全微調）　${mn} = <b>${v.full.toFixed(1)}</b>`;
  $('#c4-ro').innerHTML=txt;
  $('svg',box).setAttribute('aria-label',`${C4_KEYS.find(k=>k[0]===c4.key)[1]}，${mn}；目前欄 ρ = ${c<5?C4_RHO[c]:'1.0'}。左右方向鍵切換欄位`);
  announce('c4',$('#c4-ro').textContent)}
function renderCalc(){let u=+$('#c4-u').value,d=+$('#c4-d').value,msg='';
  if(!(u>=0&&u<=100)){u=Math.min(100,Math.max(0,u||0));msg+='A_up 的範圍是 0–100。'}
  if(!(d>=0&&d<=200)){d=Math.min(200,Math.max(0,d||0));msg+='A_down 的範圍是 0–200。'}
  if(Math.abs(u-33.38)<1e-9&&Math.abs(d-80.3)<1e-9)msg+='表中 Avg 印 66.9，由同列數字算得 56.8。';
  $('#c4-calc').innerHTML=`Avg = <b>${avg(u,d).toFixed(2)}</b>　H-score = <b class="acc">${hs(u,d).toFixed(2)}</b>`;$('#c4-msg').textContent=msg;
  $$('#c4-pre .chip').forEach(c=>c.setAttribute('aria-pressed',String(Math.abs(+c.dataset.u-u)<1e-9&&Math.abs(+c.dataset.d-d)<1e-9)))}
function initC4(){const kg=$('#c4-key');C4_KEYS.forEach(([k,n])=>{const b=document.createElement('button');b.type='button';b.textContent=n;b.setAttribute('aria-pressed',String(k===c4.key));b.addEventListener('click',()=>{c4.key=k;pressGroup($$('button',kg),b);renderC4()});kg.appendChild(b)});
  const mg=$('#c4-met');[['H','H-score'],['U','A_up']].forEach(([k,n])=>{const b=document.createElement('button');b.type='button';b.innerHTML=k==='U'?'A<sub>up</sub>':n;b.setAttribute('aria-pressed',String(k===c4.met));b.addEventListener('click',()=>{c4.met=k;pressGroup($$('button',mg),b);renderC4()});mg.appendChild(b)});
  $('#c4-fig svg').addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){c4.col=Math.max(0,Math.min(5,c4.col+(e.key==='ArrowRight'?1:-1)));renderC4();e.preventDefault()}else if(e.key==='Enter'||e.key===' '){renderC4();e.preventDefault()}});
  ['#c4-u','#c4-d'].forEach(s=>$(s).addEventListener('input',renderCalc));
  $$('#c4-pre .chip').forEach(c=>c.addEventListener('click',()=>{$('#c4-u').value=c.dataset.u;$('#c4-d').value=c.dataset.d;renderCalc()}));
  renderC4();renderCalc()}

/* ---------- 3x3 mask example ---------- */
function initMask3(){const v=[.90,.12,.55,.30,.05,.71,.48,.66,.20],low=new Set([1,4]);
  $('#mask3').innerHTML=v.map((s,k)=>`<div class="m3${low.has(k)?' upd':''}"><b>${s.toFixed(2)}</b><span>${low.has(k)?'M = 1 · 更新':'M = 0 · 凍結'}</span></div>`).join('')}

/* ---------- timeline ---------- */
const TL_GROUP={'1312.6211':'cl','1612.00796':'cl','1711.09601':'cl','1711.05769':'imp','2006.05467':'imp','2306.11695':'imp','2402.17762':'imp','2302.06600':'pm','2311.03099':'pm','2402.12048':'pm','2404.18466':'sp','2411.10928':'sp','2412.04468':'mdl','2602.04509':'me'};
const TL_NAME={cl:'遺忘與持續學習',imp:'重要性與剪枝',pm:'Post-merging',sp:'Sparse FT',mdl:'模型',me:'本文',other:'其他'};
const TL_NOTE={cl:'dowser-catastrophic-forgetting-mllm-fine-tuning',imp:'dowser-parameter-importance-pruning-scores',pm:'dowser-post-merging-methods',sp:'dowser-sparse-finetuning-hft-spider',mdl:'dowser-llava-nvila-setting',me:'dowser-functional-importance-theory-and-probing',other:'dowser-lineage-hub'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  if(!data.some(e=>e.arxiv==='2602.04509'))data.push({year:2026,title:'Model-Dowser',authors_short:'本論文',arxiv:'2602.04509',role:'本論文'});
  data=data.map((e,i)=>({e,i})).sort((a,b)=>(+a.e.year||0)-(+b.e.year||0)||a.i-b.i).map(o=>o.e);
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||'other';used.add(g);const a=document.createElement('article');a.className='ev'+(g==='me'?' me':'');a.dataset.g=g;
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}<a href="${TL_NOTE[g]}.html">筆記</a></div>`;track.appendChild(a)});
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

/* ---------- self test (acceptance tables of design.md §5) ---------- */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};const near=(a,b,t)=>Math.abs(a-b)<=t+1e-12;
  ok(near(erf(.5),.5204998778,2e-7),'erf(0.5)');
  // 5.1
  [[0,0,.01,.01118,.01127,-.8],[0,0,.1,.11180,.12071,-7.4],[0,0,.5,.55902,.77308,-27.7],[0,0,-1,1.11803,.35476,215.1],[0,1,2,1.11803,1.88130,-40.6],[1,0,-.5,1.21333,1.24037,-2.2],[1,1,.1,.12133,.12103,.2],[1,1,-3,3.63998,3.59732,1.2]]
    .forEach(([i,j,d,p,t,r])=>{const pp=c1pred(i,j,d),tt=c1true(i,j,d);ok(near(pp,p,1e-5)&&near(tt,t,1e-5)&&near((pp-tt)/tt*100,r,.1),`C1 (${i+1},${j+1}) ΔW=${d}: ${pp.toFixed(5)} ${tt.toFixed(5)}`)});
  [[0,0,7.2351],[0,1,3.3804],[1,0,7.8624],[1,1,3.9312]].forEach(([i,j,v])=>ok(near(c1ymax(i,j),v,1e-3),`C1 ymax (${i+1},${j+1})`));
  ok(near(gelu(2),1.9545,1e-4)&&near(2*gelu(1),1.6827,1e-4),'GELU homogeneity example');
  // 5.2
  const T2=[[0,2,{mag:[8,13,.1058,.1251],wanda:[10,13,.1716,.1922],full:[5,8,.0362,.0569]},.7588],
    [0,4,{mag:[3,8,11,13,.9339,1.1191],wanda:[8,10,13,14,.3152,.3621],full:[5,6,8,10,.1576,.1809]},1.3588],
    [0,8,{mag:[3,4,5,8,10,11,13,15,2.4122,2.7570],wanda:[2,4,5,6,8,10,13,14,1.1884,1.3778],full:[5,6,7,8,10,11,13,14,.7493,.9584]},2.4369],
    [-.75,4,{mag:[3,8,11,13,.7530,.7736],wanda:[8,10,13,14,.2683,.2853],full:[8,10,11,12,.6977,.0531]},1.2573],
    [-2,4,{mag:[3,8,11,13,.7206,.8186],wanda:[8,10,13,14,.2583,.2953],full:[5,6,8,10,.0973,.1141]},1.2060],
    [-4,4,{mag:[3,8,11,13,.7290,.7584],wanda:[8,10,13,14,.2648,.2819],full:[9,10,11,12,0,.0021]},1.2075]];
  T2.forEach(([b3,m,exp,rm])=>{const S=c2scores(b3);for(const k in exp){const e=exp[k],U=lowest(S[k],m),cells=[...U].map(q=>q+1).sort((a,b)=>a-b);
      const real=zeroShift(b3,U,S.f),ss=[...U].reduce((t,q)=>t+S.full[q],0);ok(cells.join()===e.slice(0,m).join()&&near(real,e[m],5e-4)&&near(ss,e[m+1],5e-4),`C2 b3=${b3} m=${m} ${k}: ${cells} ${real.toFixed(4)} ${ss.toFixed(4)}`)}
    let got=null;c2cache.delete(b3.toFixed(2));grayRun(b3,65536,r=>{got=r});ok(got&&near(got[m],rm,5e-4),`C2 random b3=${b3} m=${m}: ${got&&got[m].toFixed(4)}`)});
  {const S=c2scores(0);[['mag',4.0653,6.9935],['wanda',4.0595,6.8896],['full',3.2714,5.9122]].forEach(([k,r,s])=>{const U=lowest(S[k],15);ok(near(zeroShift(0,U,S.f),r,5e-4)&&near([...U].reduce((t,q)=>t+S.full[q],0),s,5e-4),'C2 m=15 '+k)});
    ok(near(c2cache.get('0.00')[15],3.8780,5e-4),'C2 random m=15');
    ok(near(spearman(S.full,S.mag),.296,5e-4)&&near(spearman(S.full,S.wanda),.631,5e-4),'C2 Spearman b3=0');}
  // 5.3
  [[[3],9,3,1,0],[[1,1],2,1,.7071,4],[[3,4],25,4,.8,576],[[1,1,1,1,1,1,1,1],8,2.1875,.7734,112],[[3,.5,.5,.5,.5,.5,.5,.5],10.75,3.0078,.9174,68.25]]
    .forEach(([a,e2,e1,r,v])=>{const E=exactX(a);ok(near(E.e2,e2,1e-4)&&near(E.e1,e1,1e-4)&&near(E.e1/E.na,r,1e-4)&&near(E.var2,v,1e-4),'C3 exact '+a)});
  [[1,1,1],[2,5,4],[8,5.5678,4.75],[64,5.3619,4.4688]].forEach(([R,l2,l1])=>{const M=mcX([3,4],R,1);ok(near(M.l2,l2,5e-5)&&near(M.l1,l1,5e-5),'C3 MC R='+R)});
  {const r=repeatX([3,4],8);ok(near(r.m2,4.9809,5e-5)&&near(r.s2,.9247,5e-5)&&near(r.m1,4.0825,5e-5)&&near(r.s1,1.0681,5e-5),'C3 repeat (3,4) R=8');
   const r1=repeatX([3,4],1);ok(near(r1.m2,4,5e-5)&&near(r1.s2,3.0075,5e-5),'C3 repeat R=1');const r8=repeatX([1,1,1,1,1,1,1,1],8);ok(near(r8.m2,2.6940,5e-5)&&near(r8.m1,2.1338,5e-5),'C3 repeat ones R=8');}
  // 5.4
  ok(near(hs(62.65,135.5),85.68,.005)&&near(avg(62.65,135.5),99.075,.006),'C4 calc Dowser');ok(near(hs(62.33,.1),.20,.005)&&near(avg(62.33,.1),31.215,.006),'C4 calc IconQA');ok(near(avg(33.38,80.3),56.84,.005)&&near(hs(33.38,80.3),47.16,.005),'C4 calc Tailor');
  const gaps={'NVILA|ImageNet-R':[1.7,3.4,7.7,11.2,5.8],'LLaVA|ImageNet-R':[.9,4.4,9.4,2.4,.7],'NVILA|COCO':[.5,1.3,9.8,11.0,4.4],'LLaVA|COCO':[.2,1.0,6.7,3.8,.6]};
  for(const k in gaps)gaps[k].forEach((g,q)=>ok(near(C4_TABLE[k].ours[q][3]-C4_TABLE[k].rand[q][3],g,.05),`C4 gap ${k} ρ=${C4_RHO[q]}`));
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  renderC2();
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ---------- boot ---------- */
initTheme();initToc();drawNet($('#net-fig svg'),[0,0]);renderGelu();initC1();initC2();initC3();initC4();initMask3();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderGelu();renderC1();renderC3();renderC4()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))selfTest();
})();
