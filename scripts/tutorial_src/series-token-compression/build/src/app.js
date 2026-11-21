(()=>{
'use strict';
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
const TEST=/[?&]test\b/.test(location.search);
function el(tag,a,p){const e=document.createElementNS(NS,tag);if(a)for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function tx(p,x,y,s,a){const t=el('text',Object.assign({x,y},a||{}),p);t.textContent=s;return t}
const lin=(d0,d1,r0,r1)=>v=>r0+(v-d0)/(d1-d0)*(r1-r0);
const lgs=(d0,d1,r0,r1)=>{const a=Math.log(d0),b=Math.log(d1);return v=>r0+(Math.log(v)-a)/(b-a)*(r1-r0)};
function lp(pts){let s='';for(let i=0;i<pts.length;i++)s+=(i?'L':'M')+pts[i][0].toFixed(1)+' '+pts[i][1].toFixed(1);return s}
function widthOf(box,min,max){return Math.max(min,Math.min(max,Math.floor(box.clientWidth||max)))}
function frame(svg,o){
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${o.W} ${o.H}`);svg.setAttribute('width',o.W);svg.setAttribute('height',o.H);
  const m=o.m,x0=m.l,x1=o.W-m.r,y0=o.H-m.b,y1=m.t;
  const X=(o.xlog?lgs:lin)(o.x[0],o.x[1],x0,x1),Y=lin(o.y[0],o.y[1],y0,y1);
  const g=el('g',null,svg);
  (o.yt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:(o.yf?o.yf(v):String(v)),yy=Y(val);el('line',{x1:x0,x2:x1,y1:yy,y2:yy,class:'grid'},g);tx(g,x0-8,yy+4,lab,{class:'tk','text-anchor':'end'})});
  (o.xt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:(o.xf?o.xf(v):String(v)),xx=X(val);el('line',{x1:xx,x2:xx,y1:y0,y2:y0+4,class:'axl'},g);tx(g,xx,y0+18,lab,{class:'tk','text-anchor':'middle'})});
  el('line',{x1:x0,x2:x1,y1:y0,y2:y0,class:'axl'},g);
  if(o.xl)tx(g,(x0+x1)/2,o.H-4,o.xl,{class:'al','text-anchor':'middle'});
  if(o.yl)tx(g,4,12,o.yl,{class:'al'});
  return {X,Y,g,x0,x1,y0,y1};
}
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}
function segInit(id,cb){$$('#'+id+' button').forEach(b=>b.addEventListener('click',()=>{$$('#'+id+' button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));cb(b.dataset.v)}))}
function segSet(id,v){$$('#'+id+' button').forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.v===String(v))))}
const sci=v=>{if(v===0)return'0';const e=Math.floor(Math.log10(Math.abs(v)));if(e>=-2)return v.toFixed(4);const m=v/10**e;return m.toFixed(2)+'×10'+String(e).replace('-','⁻').replace(/\d/g,c=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[c])};

/* ---------- stage bar ---------- */
const STAGES=[['第 0 期','2020–21'],['第 1 期','2021–22'],['第 2 期','2023–24'],['第 3 期','2022–25'],['第 4 期','2023–25'],['第 5 期','2023–25'],['第 6 期','2026']];
function renderStageBars(){$$('.stagebar').forEach(n=>{const cur=+n.dataset.stage,c2=n.dataset.stage2!==undefined?+n.dataset.stage2:-9;
  n.innerHTML=STAGES.map((s,i)=>`<span class="st${i===cur||i===c2?' on':''}"><b>${s[0]}</b><i>${s[1]}</i></span>`).join('')})}

/* =========================================================== C1 · ToMe toy */
const C1X=[[.58,.76,.94],[.60,.77,.95],[.62,.79,.95],[.98,.86,.45],[.64,.80,.96],[.66,.82,.97],[.22,.45,.20],[.68,.83,.97],
 [.45,.58,.32],[.44,.57,.31],[.40,.28,.18],[.43,.56,.30],[.42,.55,.30],[.41,.54,.29],[.40,.53,.28],[.39,.52,.27]];
const C1NAME=['天空','天空','天空','太陽','天空','天空','樹','天空','草','草','樹幹','草','草','草','草','草'];
const C1MU=[0,1,2].map(c=>C1X.reduce((a,x)=>a+x[c],0)/16);
const C1KEY=C1X.map(x=>x.map((v,c)=>v-C1MU[c]));
const dot3=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],nrm3=a=>Math.sqrt(dot3(a,a));
function cos3(a,b){const na=nrm3(a),nb=nrm3(b);return na>1e-12&&nb>1e-12?dot3(a,b)/(na*nb):0}
function c1partition(mode,seed){if(mode==='alt'){const A=[],B=[];for(let i=0;i<16;i++)(i%2?B:A).push(i);return{A,B}}
  const r=mulberry32(seed),dst=[];for(let by=0;by<2;by++)for(let bx=0;bx<2;bx++){const k=Math.min(3,Math.floor(r()*4));dst.push((2*by+(k>>1))*4+(2*bx+(k&1)))}
  dst.sort((a,b)=>a-b);return{A:[...Array(16).keys()].filter(i=>!dst.includes(i)),B:dst}}
function tomeLayer(mode,r,seed){const {A,B}=c1partition(mode,seed);const best=[],bidx=[];
  A.forEach(a=>{let bs=-Infinity,bk=0;B.forEach((b,k)=>{const s=cos3(C1KEY[a],C1KEY[b]);if(s>bs){bs=s;bk=k}});best.push(bs);bidx.push(bk)});
  const order=A.map((_,i)=>i).sort((i,j)=>(best[j]-best[i])||(i-j));const src=new Set(order.slice(0,r));
  const groups=new Map(B.map(b=>[b,[b]]));[...src].sort((a,b)=>a-b).forEach(i=>groups.get(B[bidx[i]]).push(A[i]));
  const toks=A.filter((_,i)=>!src.has(i)).map(a=>[a]).concat(B.map(b=>groups.get(b).slice().sort((x,y)=>x-y)));
  const feat=toks.map(g=>[0,1,2].map(c=>g.reduce((s,i)=>s+C1X[i][c],0)/g.length)),size=toks.map(g=>g.length);
  return{A,B,best,bidx,src,toks,feat,size}}
function unmergeMSE(res){let s=0;res.toks.forEach((g,t)=>g.forEach(i=>{for(let c=0;c<3;c++)s+=(C1X[i][c]-res.feat[t][c])**2}));return s/16}
function attn(q,K,V,bias){const lg=K.map((k,i)=>(16*dot3(q,k))/Math.sqrt(3)+(bias?bias[i]:0));const m=Math.max(...lg);const w=lg.map(v=>Math.exp(v-m));const z=w.reduce((a,b)=>a+b,0);
  return[0,1,2].map(c=>V.reduce((s,v,i)=>s+w[i]*v[c],0)/z)}
function attnCheck(res,qi){const q=C1KEY[qi],dense=attn(q,C1KEY,C1X);const km=res.feat.map(f=>f.map((v,c)=>v-C1MU[c]));
  const plain=attn(q,km,res.feat),prop=attn(q,km,res.feat,res.size.map(Math.log));
  const rk=[],rv=[];res.toks.forEach((g,t)=>{for(let k=0;k<g.length;k++){rk.push(km[t]);rv.push(res.feat[t])}});const dup=attn(q,rk,rv);
  const d=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);return{plain:d(plain,dense),prop:d(prop,dense),dup:d(prop,dup)}}
const c1={mode:'alt',seed:1,r:4,q:0};
const rgb=c=>`rgb(${c.map(v=>Math.round(Math.max(0,Math.min(1,v))*255)).join(',')})`;
function renderC1(){const rmax=c1.mode==='alt'?8:12;const R=$('#c1-r');R.max=rmax;if(c1.r>rmax)c1.r=rmax;R.value=c1.r;$('#c1-ro').textContent=c1.r;
  R.setAttribute('aria-valuetext','r = '+c1.r);$('#c1-re').disabled=c1.mode==='alt';
  const res=tomeLayer(c1.mode,c1.r,c1.seed),inA=new Set(res.A),srcTok=new Set([...res.src].map(i=>res.A[i]));
  const owner=new Array(16);res.toks.forEach((g,t)=>g.forEach(i=>owner[i]=t));
  const box=$('#c1-a');box.innerHTML='';const W=232,cs=52,o=12;
  const svg=el('svg',{viewBox:`0 0 ${W} ${W}`,width:'100%',role:'img','aria-label':`合併前：${c1.mode==='alt'?'交替':'隨機 2×2'}分割，B 集合為 ${res.B.join('、')}`},box);
  const cx=i=>o+(i%4)*cs+cs/2,cy=i=>o+Math.floor(i/4)*cs+cs/2;
  res.A.forEach((a,i)=>{const b=res.B[res.bidx[i]],kept=srcTok.has(a);el('line',{x1:cx(a),y1:cy(a),x2:cx(b),y2:cy(b),class:kept?'ink':'ref','stroke-width':kept?2.5:1.2},svg)});
  for(let i=0;i<16;i++){const g=el('g',{},svg);
    if(inA.has(i))el('rect',{x:cx(i)-17,y:cy(i)-17,width:34,height:34,rx:10,fill:rgb(C1X[i]),stroke:'var(--ink)','stroke-width':i===c1.q?3.5:1.5,'stroke-dasharray':'4 3'},g);
    else el('rect',{x:cx(i)-19,y:cy(i)-19,width:38,height:38,rx:2,fill:rgb(C1X[i]),stroke:'var(--ink)','stroke-width':i===c1.q?3.5:2},g);
    tx(g,cx(i),cy(i)+5,String(i),{class:'labm','text-anchor':'middle',fill:C1X[i][0]+C1X[i][1]+C1X[i][2]<1.6?'#fff':'#1D1C19',style:'fill:'+(C1X[i][0]+C1X[i][1]+C1X[i][2]<1.6?'#fff':'#1D1C19')})}
  // focusable buttons overlay for keyboard / touch
  const grid=document.createElement('div');grid.className='c1btns';grid.setAttribute('role','group');grid.setAttribute('aria-label','選擇 query token');
  for(let i=0;i<16;i++){const b=document.createElement('button');b.type='button';b.style.left=((cx(i)-cs/2)/W*100)+'%';b.style.top=((cy(i)-cs/2)/W*100)+'%';b.style.width=b.style.height=(cs/W*100)+'%';
    const tgt=inA.has(i)?(srcTok.has(i)?`，合併到 ${res.B[res.bidx[res.A.indexOf(i)]]}`:'，未合併'):'';
    b.setAttribute('aria-label',`token ${i}，${C1NAME[i]}，${inA.has(i)?'A 集合':'B 集合'}${tgt}${i===c1.q?'，目前的 query':''}`);b.setAttribute('aria-pressed',String(i===c1.q));
    b.addEventListener('click',()=>{c1.q=i;renderC1();const nb=$$('.c1btns button')[i];nb&&nb.focus()});grid.appendChild(b)}
  box.appendChild(grid);
  // merged row
  const sb=$('#c1-b');sb.textContent='';const n=res.toks.length,cols=Math.min(n,4),rows=Math.ceil(n/cols),bw=52;
  sb.setAttribute('viewBox',`0 0 232 ${Math.max(232,rows*62+20)}`);sb.setAttribute('width','100%');
  res.toks.forEach((g,t)=>{const x=12+(t%cols)*bw+(4-cols)*bw/2,y=14+Math.floor(t/cols)*62;el('rect',{x:x+6,y,width:40,height:40,rx:g.length>1?4:10,fill:rgb(res.feat[t]),stroke:'var(--ink)','stroke-width':1.5},sb);
    tx(sb,x+26,y+56,'s='+g.length,{class:'tk','text-anchor':'middle'})});
  sb.setAttribute('aria-label',`合併後 ${n} 個 token，大小 ${res.size.join('、')}`);
  // unmerge grid
  const sc=$('#c1-c');sc.textContent='';sc.setAttribute('viewBox','0 0 232 232');sc.setAttribute('width','100%');
  const letters='abcdefghijklmnop';const multi=res.toks.map((g,t)=>g.length>1?t:-1).filter(t=>t>=0);
  for(let i=0;i<16;i++){const t=owner[i];el('rect',{x:cx(i)-cs/2+2,y:cy(i)-cs/2+2,width:cs-4,height:cs-4,fill:rgb(res.feat[t]),stroke:'var(--line)'},sc);
    const L=multi.indexOf(t);if(L>=0){const f=res.feat[t];const lt=f[0]+f[1]+f[2]<1.6;tx(sc,cx(i),cy(i)+5,letters[L],{class:'labm','text-anchor':'middle',style:'fill:'+(lt?'#fff':'#1D1C19')})}}
  sc.setAttribute('aria-label','unmerge 後：'+(multi.length?multi.map((t,k)=>letters[k]+' 組 '+res.toks[t].join('、')).join('；'):'沒有合併'));
  const ac=attnCheck(res,c1.q),mse=unmergeMSE(res);
  $('#c1-out').innerHTML=`<b>分割</b> B（dst）＝${res.B.join('、')}<br><b>合併組</b> ${multi.length?multi.map(t=>'{'+res.toks[t].join(',')+'}').join(' '):'（無）'}<br><b>unmerge MSE</b> ${sci(mse)}<br><b>query ${c1.q}（${C1NAME[c1.q]}）</b> 與原始注意力的差：不加 log s ${sci(ac.plain)}；加 log s ${sci(ac.prop)}<br><b>加 log s 對複製 key s 份的差</b> ${ac.dup<1e-12?'&lt; 1e-12':sci(ac.dup)}`;
  announce('c1',`r = ${c1.r}，合併 ${multi.length} 組，unmerge MSE ${sci(mse)}`)}
function initC1(){segInit('c1-mode',v=>{c1.mode=v;renderC1()});$('#c1-re').addEventListener('click',()=>{c1.seed++;renderC1()});
  $('#c1-r').addEventListener('input',e=>{c1.r=+e.target.value;renderC1()});renderC1()}

/* =========================================================== C2 · cost model */
const Cb=(L,d)=>24*L*d*d+4*L*L*d;
function speedups(N,d,T,k){const L=N+T,Lr=k*N+T;const d17=17*Cb(L,d);
  const sa=d17/(17*(8*Lr*d*d+4*Lr*Lr*d+16*L*d*d)),kv=d17/(17*(24*L*d*d+4*L*Lr*d)),core=d17/(6*Cb(L,d)+11*Cb(Lr,d));
  const nest=18*Cb(L,d)/(6*Cb(L,d)+6*Cb(Math.sqrt(k)*N+T,d)+6*Cb(k*N+T,d));return{sa,kv,core,nest}}
const coreBound=(N,d,T)=>17*Cb(N+T,d)/(6*Cb(N+T,d)+11*Cb(T,d));
const qshare=(L,d)=>L/(6*d+L);
const c2={N:1024,d:768,T:0,v:.5,meas:true};
const c2k=()=>Math.pow(16,c2.v-1);
function renderC2(){const k=c2k();$('#c2-ko').textContent=k.toFixed(3);$('#c2-to').textContent=c2.T;$('#c2-k').setAttribute('aria-valuetext','κ = '+k.toFixed(3));
  const box=$('#c2-fig'),svg=$('svg',box),W=widthOf(box,300,640),H=Math.round(W*.62),top=c2.N===16384?7:3.5;
  const f=frame(svg,{W,H,m:{l:44,r:W<480?12:110,t:30,b:40},x:[1/16,1],xlog:true,y:[1,top],xt:[[1/16,'1/16'],[1/8,'1/8'],[.25,'1/4'],[.5,'1/2'],[1,'1']],yt:(top>4?[1,2,3,4,5,6,7]:[1,1.5,2,2.5,3,3.5]).map(v=>[v,v+'×']),xl:'保留比例 κ（對數）',yl:'加速'});
  const ks=[];for(let i=0;i<=120;i++)ks.push(Math.pow(16,i/120-1));
  const S=ks.map(x=>speedups(c2.N,c2.d,c2.T,x));const yc=v=>f.Y(Math.min(v,top));
  const series=[['core','tot','只縮中段'],['nest','npar dash','巢狀'],['sa','perp dot','ToMeSD 式'],['kv','ink dd','ToDo 式']];
  const ub=coreBound(c2.N,c2.d,c2.T);if(ub<=top){el('line',{x1:f.x0,x2:f.x1,y1:f.Y(ub),y2:f.Y(ub),class:'ref'},f.g);if(W>=480)tx(f.g,f.x1+6,f.Y(ub)+4,'上限 '+ub.toFixed(2)+'×',{class:'al'})}
  const ends=[];series.forEach(([key,cls,name])=>{el('path',{d:lp(ks.map((x,i)=>[f.X(x),yc(S[i][key])])),class:cls.includes('npar')?'par dash':cls==='tot'?'tot':cls.includes('perp')?'perp dot':'ink dd'},f.g);
    if(S.some(s=>s[key]>top))tx(f.g,f.x0+4,f.y1+12,'超出',{class:'al'});ends.push([S[0][key],name])});
  if(W>=480){ends.sort((a,b)=>b[0]-a[0]);let last=-1e9;ends.forEach(([v,name])=>{let y=yc(v)+4;if(y<last+15)y=last+15;last=y;tx(f.g,f.x1+6,y,name,{class:'lab'})})}
  const ck=speedups(c2.N,c2.d,c2.T,k);el('line',{x1:f.X(k),x2:f.X(k),y1:f.y0,y2:f.y1,class:'cur'},f.g);
  if(c2.meas&&c2.N===1024)[[.5,1.84],[.25,2.15]].forEach(([x,v])=>{const X=f.X(x),Y=f.Y(v);el('path',{d:`M${X} ${Y-7}L${X+7} ${Y}L${X} ${Y+7}L${X-7} ${Y}Z`,class:'hollow'},f.g);tx(f.g,X+10,Y-8,v.toFixed(2)+'×',{class:'labm'})});
  $('.c2leg').classList.toggle('narrow',W<480);
  const L=c2.N+c2.T,q=qshare(L,c2.d);$('#c2-q').innerHTML=`<span style="width:${(1-q)*100}%"></span><span style="width:${q*100}%"></span>`;$('#c2-qt').textContent=`單一 block：線性項 ${((1-q)*100).toFixed(1)}%，注意力二次項 ${(q*100).toFixed(1)}%`;
  $('#c2-out').innerHTML=`<b>κ = ${k.toFixed(3)}</b><br>只縮中段 <b>${ck.core.toFixed(3)}×</b><br>兩層巢狀 <b>${ck.nest.toFixed(3)}×</b><br>ToMeSD 式 <b>${ck.sa.toFixed(3)}×</b><br>ToDo 式 <b>${ck.kv.toFixed(3)}×</b><br>只縮中段上限 ${ub.toFixed(3)}×<br>二次項比例（L = ${L}）${(q*100).toFixed(1)}%`;
  svg.setAttribute('aria-label',`N = ${c2.N}、κ = ${k.toFixed(3)}：只縮中段 ${ck.core.toFixed(2)} 倍，巢狀 ${ck.nest.toFixed(2)} 倍，ToMeSD 式 ${ck.sa.toFixed(2)} 倍，ToDo 式 ${ck.kv.toFixed(2)} 倍`);
  announce('c2',`κ = ${k.toFixed(3)}，只縮中段 ${ck.core.toFixed(2)} 倍，ToDo 式 ${ck.kv.toFixed(2)} 倍`)}
function initC2(){segInit('c2-N',v=>{c2.N=+v;renderC2()});segInit('c2-d',v=>{c2.d=+v;renderC2()});
  $('#c2-t').addEventListener('input',e=>{c2.T=+e.target.value;renderC2()});$('#c2-k').addEventListener('input',e=>{c2.v=+e.target.value;renderC2()});
  $('#c2-m').addEventListener('change',e=>{c2.meas=e.target.checked;renderC2()});renderC2()}

/* =========================================================== shared scene / groupings (port of verify/common.py) */
const S=128,PX=8,G=16,N=256,FD=192;
function d2xy(n,d){let x=0,y=0,t=d;for(let s=1;s<n;s*=2){const rx=1&Math.floor(t/2),ry=1&(t^rx);if(ry===0){if(rx===1){x=s-1-x;y=s-1-y}const q=x;x=y;y=q}x+=s*rx;y+=s*ry;t=Math.floor(t/4)}return[x,y]}
const HIL_PID=[...Array(N).keys()].map(d=>{const[x,y]=d2xy(G,d);return y*G+x});
function lerp3(a,b,s){return[a[0]+(b[0]-a[0])*s,a[1]+(b[1]-a[1])*s,a[2]+(b[2]-a[2])*s]}
function scene(name){const r=mulberry32(1),img=new Float64Array(S*S*3);
  for(let y=0;y<S;y++)for(let x=0;x<S;x++){const n1=r(),n2=r(),u=(x+.5)/S,v=(y+.5)/S;let c;
    if(name==='checker'){c=((Math.floor(x/32)+Math.floor(y/32))%2===0)?[.85,.80,.70]:[.30,.40,.55]}
    else{if(v<.62)c=lerp3([.55,.75,.95],[.85,.92,.98],v/.62);else{c=lerp3([.45,.58,.32],[.36,.48,.26],(v-.62)/.38);c=c.map(ci=>ci+.06*(n1-.5))}
      if(Math.hypot(u-.78,v-.18)<.09)c=[.98,.86,.45];
      if(Math.abs(u-.28)<.025&&.45<v&&v<.70)c=[.40,.28,.18];
      if(Math.hypot(u-.28,v-.38)<.15)c=[.20+.18*(n1-.5),.45+.18*(n1-.5),.20];
      if(.52<u&&u<.70&&.66<v&&v<.82){c=[.95,.95,.92];if(.69<v&&v<.79&&Math.floor(x/2)%3===0)c=[.10,.10,.12]}}
    for(let ch=0;ch<3;ch++)img[(y*S+x)*3+ch]=Math.min(1,Math.max(0,c[ch]+.01*(n2-.5)))}
  return img}
function noiseField(){const r=mulberry32(2),e=new Float64Array(S*S*3);for(let i=0;i<S*S*3;i++)e[i]=gauss(r);return e}
function features(img){const h=new Float64Array(N*FD);for(let pr=0;pr<G;pr++)for(let pc=0;pc<G;pc++){let k=0;const o=(pr*G+pc)*FD;
  for(let yy=0;yy<PX;yy++)for(let xx=0;xx<PX;xx++)for(let ch=0;ch<3;ch++)h[o+k++]=img[((pr*PX+yy)*S+pc*PX+xx)*3+ch]}return h}
function gaps(h){const d=new Float64Array(N-1);for(let j=0;j<N-1;j++){const a=HIL_PID[j]*FD,b=HIL_PID[j+1]*FD;let s=0;for(let k=0;k<FD;k++){const t=h[b+k]-h[a+k];s+=t*t}d[j]=Math.sqrt(s)}return d}
function runsToLabels(cuts){const cs=new Set(cuts),lab=new Int32Array(N);let g=0;for(let j=0;j<N;j++){lab[HIL_PID[j]]=g;if(cs.has(j))g++}return lab}
function mGrid(R){const L=N/R,c=[];for(let r=1;r<R;r++)c.push(r*L-1);return runsToLabels(c)}
function mRti(h,R){const d=gaps(h);const idx=[...Array(N-1).keys()].sort((i,j)=>(d[j]-d[i])||(i-j)).slice(0,R-1);return runsToLabels(idx)}
function meanFeat(h){const mu=new Float64Array(FD);for(let p=0;p<N;p++)for(let k=0;k<FD;k++)mu[k]+=h[p*FD+k];for(let k=0;k<FD;k++)mu[k]/=N;return mu}
function unitCentered(f,mu){const c=new Float64Array(FD);let s=0;for(let k=0;k<FD;k++){c[k]=f[k]-mu[k];s+=c[k]*c[k]}s=Math.sqrt(s);if(s>1e-12)for(let k=0;k<FD;k++)c[k]/=s;else c.fill(0);return c}
function mTome(h,R){const mu=meanFeat(h);let toks=[];for(let p=0;p<N;p++)toks.push({f:h.slice(p*FD,(p+1)*FD),s:1,mem:[p]});
  while(toks.length>R){const A=toks.filter((_,i)=>i%2===0),B=toks.filter((_,i)=>i%2===1),r=Math.min(A.length,toks.length-R);
    const ua=A.map(a=>unitCentered(a.f,mu)),ub=B.map(b=>unitCentered(b.f,mu));const best=[],bidx=[];
    for(let i=0;i<A.length;i++){let bs=-Infinity,bk=0;for(let k=0;k<B.length;k++){let s=0;const x=ua[i],y=ub[k];for(let q=0;q<FD;q++)s+=x[q]*y[q];if(s>bs){bs=s;bk=k}}best.push(bs);bidx.push(bk)}
    const order=A.map((_,i)=>i).sort((i,j)=>(best[j]-best[i])||(i-j));const src=new Set(order.slice(0,r));
    [...src].sort((a,b)=>a-b).forEach(i=>{const b=B[bidx[i]],a=A[i],tot=b.s+a.s;for(let q=0;q<FD;q++)b.f[q]=(b.s*b.f[q]+a.s*a.f[q])/tot;b.s=tot;b.mem=b.mem.concat(a.mem)});
    toks=A.filter((_,i)=>!src.has(i)).concat(B)}
  const lab=new Int32Array(N);toks.forEach((t,g)=>t.mem.forEach(p=>lab[p]=g));return lab}
function threshBatch(R,HB){const pool=[];HB.forEach((h,b)=>{const d=gaps(h);for(let j=0;j<N-1;j++)pool.push([d[j],b,j])});
  pool.sort((x,y)=>(y[0]-x[0])||(x[1]-y[1])||(x[2]-y[2]));const keep=pool.slice(0,2*R-2);
  return{labs:[0,1].map(b=>runsToLabels(keep.filter(e=>e[1]===b).map(e=>e[2]))),thr:keep[keep.length-1][0]}}
/* dendrogram is self-contained so it can run inside a Blob Worker */
function dendrogram(h,N,FD){const mu=new Float64Array(FD);for(let p=0;p<N;p++)for(let k=0;k<FD;k++)mu[k]+=h[p*FD+k];for(let k=0;k<FD;k++)mu[k]/=N;
  const U=new Float64Array(N*FD);for(let p=0;p<N;p++){let s=0;for(let k=0;k<FD;k++){const v=h[p*FD+k]-mu[k];U[p*FD+k]=v;s+=v*v}s=Math.sqrt(s);for(let k=0;k<FD;k++)U[p*FD+k]=s>1e-12?U[p*FD+k]/s:0}
  const Ss=new Float64Array(N*N);for(let i=0;i<N;i++)for(let j=0;j<N;j++){let s=0;for(let k=0;k<FD;k++)s+=U[i*FD+k]*U[j*FD+k];Ss[i*N+j]=s}
  const size=new Float64Array(N).fill(1),alive=new Uint8Array(N).fill(1),merges=[];let nal=N;
  while(nal>1){let bs=-Infinity,bi=-1,bj=-1;for(let i=0;i<N;i++){if(!alive[i])continue;for(let j=i+1;j<N;j++){if(!alive[j])continue;const s=Ss[i*N+j]/(size[i]*size[j]);if(s>bs){bs=s;bi=i;bj=j}}}
    merges.push([bi,bj,bs]);for(let k=0;k<N;k++)Ss[bi*N+k]+=Ss[bj*N+k];for(let k=0;k<N;k++)Ss[k*N+bi]+=Ss[k*N+bj];size[bi]+=size[bj];alive[bj]=0;nal--}
  return merges}
function cut(merges,m){const root=new Int32Array(N);for(let p=0;p<N;p++)root[p]=p;for(let t=0;t<m;t++){const[i,j]=merges[t];for(let p=0;p<N;p++)if(root[p]===j)root[p]=i}
  const ids=new Map([...new Set(root)].sort((a,b)=>a-b).map((r,g)=>[r,g]));const lab=new Int32Array(N);for(let p=0;p<N;p++)lab[p]=ids.get(root[p]);return lab}
function linkageBatch(R,DEN){const pool=[];DEN.forEach((den,b)=>den.forEach((m,t)=>pool.push([m[2],b,t])));pool.sort((x,y)=>(y[0]-x[0])||(x[1]-y[1])||(x[2]-y[2]));
  const keep=pool.slice(0,2*N-2*R),cnt=[0,1].map(b=>keep.filter(e=>e[1]===b).length);return{labs:[cut(DEN[0],cnt[0]),cut(DEN[1],cnt[1])],thr:keep[keep.length-1][0]}}
function groupsOf(lab){const m=new Map();for(let p=0;p<N;p++){const g=lab[p];if(!m.has(g))m.set(g,[]);m.get(g).push(p)}return[...m.values()]}
function evOf(h,lab){const hb=meanFeat(h);let den=0;for(let p=0;p<N;p++)for(let k=0;k<FD;k++){const t=h[p*FD+k]-hb[k];den+=t*t}if(den<1e-18)return null;let num=0;
  groupsOf(lab).forEach(ps=>{const m=new Float64Array(FD);ps.forEach(p=>{for(let k=0;k<FD;k++)m[k]+=h[p*FD+k]});for(let k=0;k<FD;k++)m[k]/=ps.length;ps.forEach(p=>{for(let k=0;k<FD;k++){const t=h[p*FD+k]-m[k];num+=t*t}})});return 1-num/den}
function components(ps){const cs=new Set(ps),seen=new Set(),comps=[];for(const c of ps){if(seen.has(c))continue;const comp=[c],st=[c];seen.add(c);
  while(st.length){const p=st.pop(),x=p%G,y=Math.floor(p/G);for(const q of [x+1<G?p+1:-1,x>0?p-1:-1,y+1<G?p+G:-1,y>0?p-G:-1])if(q>=0&&cs.has(q)&&!seen.has(q)){seen.add(q);st.push(q);comp.push(q)}}comps.push(comp)}return comps}
function spreadOf(ps){let cx=0,cy=0;ps.forEach(p=>{cx+=p%G;cy+=Math.floor(p/G)});cx/=ps.length;cy/=ps.length;let s=0;ps.forEach(p=>{s+=Math.hypot(p%G-cx,Math.floor(p/G)-cy)});return s/ps.length}
function stats(h,lab){const gs=groupsOf(lab);let sp=0,conn=0,mx=0;gs.forEach(ps=>{sp+=ps.length*spreadOf(ps);if(components(ps).length===1)conn++;mx=Math.max(mx,ps.length)});
  return{K:gs.length,EV:evOf(h,lab),spread:sp/N,conn:conn/gs.length,maxsize:mx}}
function restore(y,h,lab){const yb=meanFeat(y);let D=0;for(let p=0;p<N;p++)for(let k=0;k<FD;k++){const t=y[p*FD+k]-yb[k];D+=t*t}let a=0,b=0,c=0;
  groupsOf(lab).forEach(ps=>{const my=new Float64Array(FD),mh=new Float64Array(FD);ps.forEach(p=>{for(let k=0;k<FD;k++){my[k]+=y[p*FD+k];mh[k]+=h[p*FD+k]}});for(let k=0;k<FD;k++){my[k]/=ps.length;mh[k]/=ps.length}
    ps.forEach(p=>{for(let k=0;k<FD;k++){const dy=y[p*FD+k]-my[k],dh=h[p*FD+k]-mh[k];a+=dy*dy;b+=dy*dh;c+=dh*dh}})});
  const E=be=>(a-2*be*b+be*be*c)/D,bs=c>1e-12?b/c:0;return{E,E0:E(0),E1:E(1),bs,Eb:E(bs)}}

/* lazily built shared data */
const DATA={};
function ensureScene(){if(DATA.img)return;DATA.img=[scene('landscape'),scene('checker')];DATA.eps=noiseField();DATA.H=DATA.img.map(features)}
let denPromise=null;
function ensureDendrograms(sync){ensureScene();if(DATA.DEN)return Promise.resolve(DATA.DEN);
  if(sync){DATA.DEN=DATA.H.map(h=>dendrogram(h,N,FD));return Promise.resolve(DATA.DEN)}
  if(denPromise)return denPromise;
  denPromise=new Promise(res=>{let w=null;try{const src='self.onmessage=e=>{const d=e.data;postMessage((' +dendrogram.toString()+ ')(new Float64Array(d.h),d.N,d.FD))}';
      const url=URL.createObjectURL(new Blob([src],{type:'text/javascript'}));const out=[null,null];let n=0;
      DATA.H.forEach((h,b)=>{const wk=new Worker(url);wk.onmessage=e=>{out[b]=e.data;wk.terminate();if(++n===2){URL.revokeObjectURL(url);DATA.DEN=out;res(out)}};wk.onerror=()=>{wk.terminate();if(!DATA.DEN){DATA.DEN=DATA.H.map(h=>dendrogram(h,N,FD));res(DATA.DEN)}};wk.postMessage({h:h.buffer.slice(0),N,FD})});w=1}
    catch(e){w=null}
    if(!w)setTimeout(()=>{DATA.DEN=DATA.H.map(h=>dendrogram(h,N,FD));res(DATA.DEN)},0)});
  return denPromise}

/* =========================================================== C3 */
const C3SHORT={grid:'1 網格',tome:'2 ToMe 式',rti:'3 RTI',thresh:'4 Hilbert 配額',hit:'5 平均連結'};
const C3NAMES={grid:'1 固定網格',tome:'2 ToMe 式合併',rti:'3 Hilbert 最大落差（RTI）',thresh:'4 Hilbert＋批次配額',hit:'5 平均連結＋批次配額'};
const c3={m:'rti',R:32,img:0,adv:false,sel:-1,cur:[7,7],cache:new Map(),busy:false};
function grouping(m,R){const key=m+'|'+R;if(c3.cache.has(key))return c3.cache.get(key);ensureScene();let out;
  if(m==='grid')out={labs:[mGrid(R),mGrid(R)],thr:null};else if(m==='tome')out={labs:DATA.H.map(h=>mTome(h,R)),thr:null};
  else if(m==='rti')out={labs:DATA.H.map(h=>mRti(h,R)),thr:null};else if(m==='thresh')out=threshBatch(R,DATA.H);
  else{if(!DATA.DEN)return null;out=linkageBatch(R,DATA.DEN)}
  out.st=out.labs.map((l,b)=>stats(DATA.H[b],l));c3.cache.set(key,out);return out}
function drawImg(cv,img,cssW){const dpr=Math.max(1,Math.min(3,devicePixelRatio||1));cv.width=Math.round(cssW*dpr);cv.height=Math.round(cssW*dpr);cv.style.width=cv.style.height=cssW+'px';
  const ctx=cv.getContext('2d');const id=new ImageData(S,S);for(let i=0;i<S*S;i++){for(let c=0;c<3;c++)id.data[i*4+c]=Math.round(Math.max(0,Math.min(1,img[i*3+c]))*255);id.data[i*4+3]=255}
  const off=document.createElement('canvas');off.width=off.height=S;off.getContext('2d').putImageData(id,0,0);ctx.imageSmoothingEnabled=false;ctx.setTransform(1,0,0,1,0,0);ctx.drawImage(off,0,0,cv.width,cv.height);ctx.setTransform(dpr,0,0,dpr,0,0);return ctx}
const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function renderC3(){ensureScene();$$('#c3-m .adv').forEach(b=>b.classList.toggle('hide',!c3.adv));
  const cv=$('#c3-cv'),W=Math.min(256,Math.max(200,cv.parentElement.clientWidth||256)),cs=W/G;const ctx=drawImg(cv,DATA.img[c3.img],W);
  const gp=grouping(c3.m,c3.R);
  if(!gp){$('#c3-note').textContent='計算平均連結的合併樹中……';ctx.fillStyle=cssv('--bg');ctx.globalAlpha=.6;ctx.fillRect(0,0,W,W);ctx.globalAlpha=1;
    ensureDendrograms(TEST).then(()=>{c3.busy=false;setBusy(false);renderC3();renderC4()});c3.busy=true;setBusy(true);return}
  const lab=gp.labs[c3.img];
  // boundaries with halo
  const segs=[];for(let y=0;y<G;y++)for(let x=0;x<G;x++){const p=y*G+x;if(x+1<G&&lab[p]!==lab[p+1])segs.push([(x+1)*cs,y*cs,(x+1)*cs,(y+1)*cs]);if(y+1<G&&lab[p]!==lab[p+G])segs.push([x*cs,(y+1)*cs,(x+1)*cs,(y+1)*cs])}
  [[cssv('--bg'),3],[cssv('--ink'),1.2]].forEach(([col,w])=>{ctx.strokeStyle=col;ctx.lineWidth=w;ctx.beginPath();segs.forEach(s=>{ctx.moveTo(s[0],s[1]);ctx.lineTo(s[2],s[3])});ctx.stroke()});
  let selTxt='點選或用方向鍵＋Enter 選取一個 patch。';
  if(c3.sel>=0){const g=lab[c3.sel],ps=[];for(let p=0;p<N;p++)if(lab[p]===g)ps.push(p);const comps=components(ps);
    ctx.save();ctx.beginPath();ps.forEach(p=>ctx.rect((p%G)*cs,Math.floor(p/G)*cs,cs,cs));ctx.clip();ctx.strokeStyle=cssv('--ink');ctx.globalAlpha=.75;ctx.lineWidth=1.5;ctx.beginPath();for(let d=-W;d<W;d+=6){ctx.moveTo(d,0);ctx.lineTo(d+W,W)}ctx.stroke();ctx.restore();
    if(comps.length>1){ctx.font='600 11px "JetBrains Mono",monospace';ctx.textAlign='center';comps.forEach((c,k)=>{const p=Math.min(...c),x=(p%G+.5)*cs,y=(Math.floor(p/G)+.5)*cs;ctx.fillStyle=cssv('--bg');ctx.fillRect(x-7,y-7,14,14);ctx.fillStyle=cssv('--ink');ctx.fillText(String(k+1),x,y+4)})}
    const sid=groupsOf(lab).findIndex(q=>q.includes(c3.sel))+1;selTxt=`patch（列 ${Math.floor(c3.sel/G)+1}，行 ${c3.sel%G+1}）屬於第 ${sid} 組，共 ${ps.length} 個 patch${comps.length>1?`，分成 ${comps.length} 塊不相連的區域（數字標示）`:'，相連成一塊'}。`;announce('c3',selTxt)}
  if(document.activeElement===cv){ctx.strokeStyle=cssv('--accent');ctx.lineWidth=2;ctx.strokeRect(c3.cur[0]*cs+1,c3.cur[1]*cs+1,cs-2,cs-2)}
  $('#c3-sel').textContent=selTxt;
  const ms=['grid','tome','rti'].concat(c3.adv?['thresh','hit']:[]);const tb=$('#c3-tab tbody');tb.innerHTML='';
  ms.forEach(m=>{const g2=grouping(m,c3.R);const tr=document.createElement('tr');if(m===c3.m)tr.className='hl';
    if(!g2){tr.innerHTML=`<td>${C3SHORT[m]}</td><td class="num" colspan="6">計算中</td>`;tb.appendChild(tr);return}
    const s=g2.st[c3.img];tr.innerHTML=`<td>${m===c3.m?'<b>'+C3SHORT[m]+'</b>':C3SHORT[m]}</td><td class="num${c3.img===0?' cur':''}">${g2.st[0].K}</td><td class="num${c3.img===1?' cur':''}">${g2.st[1].K}</td><td class="num">${s.EV==null?'—':s.EV.toFixed(3)}</td><td class="num">${s.spread.toFixed(3)}</td><td class="num">${s.conn.toFixed(3)}</td><td class="num">${s.maxsize}</td>`;tb.appendChild(tr)});
  const s=gp.st[c3.img];
  $('#c3-note').textContent=`目前：${C3NAMES[c3.m]}，R = ${c3.R}，${c3.img?'棋盤':'風景'}。批次平均 K = ${((gp.st[0].K+gp.st[1].K)/2).toFixed(1)}。`+(gp.thr!=null?`等效門檻 ${gp.thr.toFixed(c3.m==='hit'?6:4)}（只作讀數）。`:'')+'方法 4、5 兩張圖共用配額；1–3 每張圖各自 R 個。';
  cv.setAttribute('aria-label',`${C3NAMES[c3.m]}，R = ${c3.R}，${c3.img?'棋盤':'風景'}：K = ${s.K}，EV ${s.EV==null?'—':s.EV.toFixed(3)}，連通比例 ${s.conn.toFixed(3)}`);
  if(window.__c4ready)renderC4()}
function setBusy(b){$$('#c3-m button[data-v=hit]').forEach(x=>x.disabled=b)}
function initC3(){segInit('c3-m',v=>{c3.m=v;renderC3()});segInit('c3-R',v=>{c3.R=+v;renderC3()});segInit('c3-img',v=>{c3.img=+v;renderC3()});
  $('#c3-adv').addEventListener('change',e=>{c3.adv=e.target.checked;if(!c3.adv&&(c3.m==='thresh'||c3.m==='hit')){c3.m='rti';segSet('c3-m','rti')}if(c3.adv)ensureDendrograms(TEST).then(()=>renderC3());renderC3()});
  const cv=$('#c3-cv');
  cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect(),x=Math.floor((e.clientX-r.left)/r.width*G),y=Math.floor((e.clientY-r.top)/r.height*G);if(x<0||y<0||x>=G||y>=G)return;c3.cur=[x,y];c3.sel=y*G+x;renderC3()});
  cv.addEventListener('keydown',e=>{const k=e.key;let[x,y]=c3.cur;if(k==='ArrowRight')x=Math.min(G-1,x+1);else if(k==='ArrowLeft')x=Math.max(0,x-1);else if(k==='ArrowDown')y=Math.min(G-1,y+1);else if(k==='ArrowUp')y=Math.max(0,y-1);
    else if(k==='Enter'||k===' '){c3.sel=y*G+x;renderC3();e.preventDefault();return}else return;e.preventDefault();c3.cur=[x,y];renderC3()});
  cv.addEventListener('focus',()=>renderC3());cv.addEventListener('blur',()=>renderC3());
  const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){io.disconnect();renderC3();ensureDendrograms(TEST)}});io.observe($('#c3'));}

/* =========================================================== C4 */
const c4={t:.95,b:0};let c4feat=new Map();
function featsT(t){const key=t.toFixed(2);if(c4feat.has(key))return c4feat.get(key);ensureScene();const im=DATA.img[0],e=DATA.eps,m=new Float64Array(S*S*3);for(let i=0;i<m.length;i++)m[i]=t*im[i]+(1-t)*e[i];const v={img:m,h:features(m)};c4feat.set(key,v);return v}
function renderC4(){window.__c4ready=true;ensureScene();let m=c3.m,gp=grouping(m,c3.R);if(!gp){m='rti';gp=grouping('rti',c3.R)}const lab=gp.labs[0];
  $('#c4-src').innerHTML=`分組來源：${C3NAMES[m]}，R = ${c3.R}（風景圖，K = ${gp.st[0].K}）。<a href="#c3">在 C3 更改</a>`;
  const ft=featsT(c4.t),y=DATA.H[0],h=ft.h,res=restore(y,h,lab);
  $('#c4-to').textContent=c4.t.toFixed(2);$('#c4-bo').textContent=c4.b.toFixed(2);$('#c4-t').setAttribute('aria-valuetext','t = '+c4.t.toFixed(2));$('#c4-b-r').setAttribute('aria-valuetext','β = '+c4.b.toFixed(2));
  const box=$('.c4cv'),cw=Math.max(110,Math.min(200,Math.floor((box.clientWidth-16)/2)));
  drawImg($('#c4-a'),ft.img,cw);
  // reconstruction image: per patch yhat = ybar_c + beta (h - hbar_c)
  const gs=groupsOf(lab),out=new Float64Array(S*S*3);
  gs.forEach(ps=>{const my=new Float64Array(FD),mh=new Float64Array(FD);ps.forEach(p=>{for(let k=0;k<FD;k++){my[k]+=y[p*FD+k];mh[k]+=h[p*FD+k]}});for(let k=0;k<FD;k++){my[k]/=ps.length;mh[k]/=ps.length}
    ps.forEach(p=>{const pr=Math.floor(p/G),pc=p%G;let k=0;for(let yy=0;yy<PX;yy++)for(let xx=0;xx<PX;xx++)for(let ch=0;ch<3;ch++){out[((pr*PX+yy)*S+pc*PX+xx)*3+ch]=my[k]+c4.b*(h[p*FD+k]-mh[k]);k++}})});
  drawImg($('#c4-b'),out,cw);
  const svg=$('#c4-fig'),W=widthOf(svg.parentElement,260,520),H=Math.round(W*.5),top=Math.max(.05,Math.min(Math.max(res.E(-1),res.E(2)),4*Math.max(res.E0,res.E1,.01))*1.05);
  const f=frame(svg,{W,H,m:{l:52,r:14,t:30,b:38},x:[-1,2],y:[0,top],xt:[-1,-.5,0,.5,1,1.5,2],yt:[0,top/2,top].map(v=>[v,v.toFixed(v<.1?3:2)]),xl:'還原係數 β',yl:'相對誤差 E(β)'});
  const pts=[];for(let i=0;i<=150;i++){const b=-1+3*i/150,v=res.E(b);pts.push([f.X(b),f.Y(Math.min(v,top))])}el('path',{d:lp(pts),class:'tot'},f.g);
  el('line',{x1:f.X(c4.b),x2:f.X(c4.b),y1:f.y0,y2:f.y1,class:'cur'},f.g);
  const mk=(b,shape,lab)=>{if(b<-1||b>2)return;const X=f.X(b),Y=f.Y(Math.min(res.E(b),top));if(shape==='sq')el('rect',{x:X-5,y:Y-5,width:10,height:10,class:'hollow'},f.g);else if(shape==='tri')el('path',{d:`M${X} ${Y-7}L${X+6} ${Y+5}L${X-6} ${Y+5}Z`,class:'hollow'},f.g);else el('circle',{cx:X,cy:Y,r:5,class:'kdot'},f.g);tx(f.g,X+(shape==='dot'?-8:8),Y-10,lab,{class:'lab','text-anchor':shape==='dot'?'end':'start'})};
  mk(0,'sq','廣播');mk(1,'tri','殘差');mk(res.bs,'dot','β*');
  const out2=res.bs<-1||res.bs>2?'（超出滑桿範圍）':'';
  $('#c4-out').innerHTML=`<b>K</b> ${gp.st[0].K}<br><b>E(0)</b> 廣播 ${res.E0.toFixed(4)}<br><b>E(1)</b> 殘差 ${res.E1.toFixed(4)}<br><b>β*</b> ${res.bs.toFixed(3)}${out2}，E(β*) ${res.Eb.toFixed(4)}<br><b>目前 β = ${c4.b.toFixed(2)}</b>：E ${res.E(c4.b).toFixed(4)}`;
  svg.setAttribute('aria-label',`相對誤差拋物線：廣播 ${res.E0.toFixed(4)}，殘差 ${res.E1.toFixed(4)}，最佳 β ${res.bs.toFixed(3)}`);
  announce('c4',`t = ${c4.t.toFixed(2)}，廣播誤差 ${res.E0.toFixed(4)}，殘差誤差 ${res.E1.toFixed(4)}，最佳 β ${res.bs.toFixed(3)}`);return res}
function initC4(){$('#c4-t').addEventListener('input',e=>{c4.t=+e.target.value;renderC4()});$('#c4-b-r').addEventListener('input',e=>{c4.b=+e.target.value;renderC4()});
  $$('#c4 .chip').forEach(b=>b.addEventListener('click',()=>{if(b.dataset.b==='opt'){const gp=grouping(c3.m,c3.R)||grouping('rti',c3.R);c4.b=restore(DATA.H[0],featsT(c4.t).h,gp.labs[0]).bs}else c4.b=+b.dataset.b;
    $('#c4-b-r').value=Math.max(-1,Math.min(2,c4.b));renderC4()}));
  const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){io.disconnect();renderC4();ensureDendrograms(TEST)}});io.observe($('#c4'))}

/* =========================================================== Hilbert figure */
function renderHil(){const svg=$('#hil-fig svg'),n=8,cs=36,o=10,W=n*cs+2*o;svg.setAttribute('viewBox',`0 0 ${W} ${W}`);svg.setAttribute('width','100%');svg.style.maxWidth='340px';svg.textContent='';
  const P=[...Array(n*n).keys()].map(d=>d2xy(n,d)),c=([x,y])=>[o+x*cs+cs/2,o+y*cs+cs/2];
  for(let i=0;i<=n;i++){el('line',{x1:o,x2:o+n*cs,y1:o+i*cs,y2:o+i*cs,class:'grid'},svg);el('line',{y1:o,y2:o+n*cs,x1:o+i*cs,x2:o+i*cs,class:'grid'},svg)}
  const pat=el('pattern',{id:'hatch',width:6,height:6,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},el('defs',{},svg));el('line',{x1:0,y1:0,x2:0,y2:6,style:'stroke:var(--accent);stroke-width:2.5'},pat);
  for(let d=21;d<=33;d++){const[x,y]=P[d];el('rect',{x:o+x*cs,y:o+y*cs,width:cs,height:cs,fill:'url(#hatch)',opacity:.55},svg)}
  el('path',{d:lp(P.map(c)),class:'ink'},svg);
  for(let d=0;d<16;d++){const[x,y]=c(P[d]);el('circle',{cx:x,cy:y,r:9,class:'hollow'},svg);tx(svg,x,y+4,String(d),{class:'tk','text-anchor':'middle',style:'fill:var(--ink)'})}}

/* =========================================================== axis table */
const AX_COLS=[['stage','期別'],['A','A 縮減'],['B','B 分組'],['C','C 位置層'],['D','D 還原'],['E','E 訓練'],['F','F 預算'],['G','G 位置'],['cont','空間連續']];
const AX_ROWS=[
 ['hourglass','Hourglass','0','線性池化','固定步幅','中段','注意力上採樣＋殘差','從頭','固定','—','是'],
 ['perceiver','Perceiver IO／RIN','0／3','匯聚到 latent','學得（全域 cross-attention）','全程','輸出查詢／cross-attn 寫回','從頭','固定','latent（RTI 的 Latent Array 基線無位置）','否'],
 ['dynamicvit','DynamicViT','1','剪枝','學得重要性','逐層累進','不需（分類）','訓練','固定比例','原位置','否'],
 ['tome','ToMe','1','合併','key 相似度二部配對','每層累進','不需（分類）','免訓練','每層 r','log s','否'],
 ['tomesd','ToMeSD','2','合併','相似度＋隨機 2×2 dst','每 block 合併後還原','unmerge 複製','免訓練','固定比例','—','否'],
 ['todo','ToDo','2','只縮 K/V','固定網格','注意力','不需（query 全長）','免訓練','固定比例','原 query','是'],
 ['atedm','AT-EDM','2','剪枝','注意力排序','每層','相似度恢復','免訓練','隨時間步','—','否'],
 ['mod','MoD／DiffCR','3','跳過','學得 router','交替層／每層','殘差直通','訓練／微調','固定容量／學得逐層逐步','原位置','否'],
 ['dydit','DyDiT','3','跳過＋縮寬','學得','每層','直通','微調','隨時間步','原位置','否'],
 ['hdit','HDiT','4','合併 2×2','固定','多層巢狀','Pixel-Shuffle＋lerp skip','從頭','固定','軸向 RoPE','是'],
 ['jit','JiT','4','大 patch','固定','輸入','線性頭','從頭','固定','RoPE','是'],
 ['hnet','H-Net','5','切塊選代表','學得相鄰相似度','多階段','平滑＋重複＋殘差','從頭','ratio loss 平均','序列','是（一維）'],
 ['elit','ELIT','6','匯聚到 latent','學得 cross-attn，固定網格群組','中段','Write cross-attn','從頭／插入','彈性（尾端丟棄）','每群學得位置嵌入（未逐行核對）','群內否、群間是'],
 ['dcdit','DC-DiT','6','選代表 token','學得局部可預測性','骨幹前','de-chunk＋門控殘差','從頭／upcycle','彈性（比例）','原始 2D 座標','否（選點）'],
 ['rti','RTI','6','區域內加權池化','無參數 Hilbert 最大落差','單層中段','殘差 WRITE','凍結＋LoRA','彈性、每張精確 R','平均 RoPE 相量＋大小嵌入','是（保證）'],
 ['hit','HiT','6','軟加權池化','學得親和度＋平均連結','兩層巢狀','軟混合＋投影 skip','從頭','批次平均、單張可變','預設不加','否（允許非局部）']];
let axSel=null;
function renderAx(){const th=$('#ax-tab thead'),tb=$('#ax-tab tbody');
  th.innerHTML='<tr><th>方法</th>'+AX_COLS.map(([k,n])=>`<th class="${axSel===k?'axon':''}">${n}</th>`).join('')+'</tr>';
  tb.innerHTML=AX_ROWS.map(r=>`<tr id="ax-${r[0]}" class="${r[0]==='rti'||r[0]==='hit'?'key':''}"><td>${r[1]}</td>`+r.slice(2).map((v,i)=>`<td class="${axSel===AX_COLS[i][0]?'axon':''}">${v}</td>`).join('')+'</tr>').join('');
  const show=axSel?[axSel]:['B','D','F'];
  $('#ax-cards').innerHTML=AX_ROWS.map(r=>`<div class="axc${r[0]==='rti'||r[0]==='hit'?' key':''}"><h3>${r[1]} <span>第 ${r[2]} 期</span></h3>`+show.map(k=>{const i=AX_COLS.findIndex(c=>c[0]===k);return`<p><b>${AX_COLS[i][1]}</b>${r[2+i]}</p>`}).join('')+'</div>').join('')}
function initAx(){const ch=$('#ax-chips');AX_COLS.slice(1).forEach(([k,n])=>{const b=document.createElement('button');b.type='button';b.className='chip';b.textContent=n;b.setAttribute('aria-pressed','false');
  b.addEventListener('click',()=>{axSel=axSel===k?null:k;$$('#ax-chips .chip').forEach(c=>c.setAttribute('aria-pressed',String(c===b&&axSel===k)));renderAx()});ch.appendChild(b)});renderAx()}

/* =========================================================== timeline */
const TL_STAGE={'2006.03236':0,'2110.13711':0,'2107.14795':0,'2106.02034':1,'2210.09461':1,'2303.17604':2,'2402.13573':2,
 '2404.02258':3,'2410.03456':3,'2412.16822':3,'2601.01608':3,'2212.11972':3,'2209.12152':3,'2301.11093':4,'2401.11605':4,'2511.13720':4,
 '2304.00287':5,'2412.09871':5,'2507.07955':5,'2603.12245':6,'2603.06351':6,'2608.29281':6,'2611.22962':6};
const HG='tokc-hourglass-and-latent-interfaces',VP='tokc-vit-pruning-and-merging',TF='tokc-training-free-diffusion-merging',LR='tokc-learned-routing-dynamic-dit',CA='tokc-content-adaptive-tokenization',DC='tokc-dynamic-chunking-bytes-to-pixels';
const TL_NOTE={'2006.03236':HG,'2107.14795':HG,'2110.13711':HG,'2212.11972':HG,'2209.12152':HG,'2301.11093':HG,'2401.11605':HG,'2603.12245':HG,'2106.02034':VP,'2210.09461':VP,
 '2303.17604':TF,'2402.13573':TF,'2404.02258':LR,'2410.03456':LR,'2412.16822':LR,'2601.01608':LR,'2304.00287':CA,'2412.09871':DC,'2507.07955':DC,'2603.06351':DC,
 '2511.13720':'pixel-space-diffusion','2608.29281':'rti-lineage-hub','2611.22962':'hit-lineage-hub'};
const TL_AX={'2110.13711':'hourglass','2107.14795':'perceiver','2212.11972':'perceiver','2106.02034':'dynamicvit','2210.09461':'tome','2303.17604':'tomesd','2402.13573':'todo','2404.02258':'mod','2412.16822':'mod',
 '2410.03456':'dydit','2401.11605':'hdit','2511.13720':'jit','2507.07955':'hnet','2603.12245':'elit','2603.06351':'dcdit','2608.29281':'rti','2611.22962':'hit'};
const TL_PM={'2608.29281':217,'2611.22962':585,'2511.13720':349};
let TL_COUNT=0;
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.map((e,i)=>({e,i})).sort((a,b)=>(+a.e.year||0)-(+b.e.year||0)||a.i-b.i).map(o=>o.e);TL_COUNT=data.length;
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  data.forEach(e=>{const st=TL_STAGE[e.arxiv],key=st===undefined?'o':String(st),me=e.arxiv==='2608.29281'||e.arxiv==='2611.22962';const a=document.createElement('article');a.className='ev'+(me?' me':'');a.dataset.g=key;
    const links=[e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:'',TL_NOTE[e.arxiv]?`<a href="${TL_NOTE[e.arxiv]}.html">筆記</a>`:'',TL_AX[e.arxiv]?`<a href="#ax-${TL_AX[e.arxiv]}">對照表</a>`:'',TL_PM[e.arxiv]?`<a href="../index.html#paper-${TL_PM[e.arxiv]}">關聯圖</a>`:''].join('');
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${st===undefined?'其他':STAGES[st][0]}${me?' · 起點':''}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${links}</div>`;track.appendChild(a)});
  const used=[...new Set(data.map(e=>TL_STAGE[e.arxiv]===undefined?'o':String(TL_STAGE[e.arxiv])))].sort();const on=new Set();
  const apply=()=>{$$('.chip',chips).forEach(c=>c.setAttribute('aria-pressed',String(on.has(c.dataset.g))));$$('.ev',track).forEach(ev=>ev.classList.toggle('dim',on.size>0&&!on.has(ev.dataset.g)))};
  used.forEach(k=>{const b=document.createElement('button');b.type='button';b.className='chip';b.setAttribute('aria-pressed','false');b.textContent=k==='o'?'其他':STAGES[+k][0];b.dataset.g=k;
    b.addEventListener('click',()=>{on.has(k)?on.delete(k):on.add(k);apply()});chips.appendChild(b)});
  const rail=$('#rail'),by=dir=>rail.scrollBy({left:dir*Math.min(rail.clientWidth*.8,560),behavior:RM?'auto':'smooth'});
  $('#tl-prev').addEventListener('click',()=>by(-1));$('#tl-next').addEventListener('click',()=>by(1));
  rail.addEventListener('keydown',e=>{if(e.target!==rail)return;if(e.key==='ArrowRight'){by(1);e.preventDefault()}if(e.key==='ArrowLeft'){by(-1);e.preventDefault()}})}

/* ---------- toc, theme, appbar ---------- */
function initToc(){const secs=$$('[data-toc]'),ol=$('#toc'),bar=$('#tocbar');
  secs.forEach(s=>{ol.insertAdjacentHTML('beforeend',`<li><a href="#${s.id}"><span class="n">${s.dataset.toc}</span><span>${s.dataset.tt}</span></a></li>`);bar.insertAdjacentHTML('beforeend',`<a href="#${s.id}">${s.dataset.toc} ${s.dataset.tt}</a>`)});
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;$$('.toc a,.tocbar a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+id));const ab=$(`.tocbar a[href="#${id}"]`);if(ab&&bar.offsetParent)bar.scrollTo({left:ab.offsetLeft-16,behavior:RM?'auto':'smooth'})}})},{rootMargin:'-40% 0px -55% 0px'});
  secs.forEach(s=>io.observe(s));
  addEventListener('scroll',()=>$('#appbar').classList.toggle('raised',scrollY>8),{passive:true})}
function initTheme(){let saved=null;try{saved=localStorage.getItem('pm-theme')}catch(e){}if(saved)document.documentElement.dataset.theme=saved;
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){};renderC3();renderC4()})}

/* ---------- self test ---------- */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t)=>Math.abs(a-b)<=t+1e-12,rel=(a,b)=>Math.abs(a-b)<=Math.max(1e-7,Math.abs(b)*.01);
  // C1
  const c1t=[['alt',1,2,'12,13|14,15',1.8750e-05,1.1601e-02,3.6674e-06,1.2935e-02,1.0223e-04],['alt',1,4,'4,5|8,9|12,13|14,15',5.6250e-05,6.1202e-03,1.8171e-04,2.7758e-02,1.3127e-04],
   ['alt',1,6,'0,1,2|4,5|8,9|12,13|14,15',1.3958e-04,2.5462e-02,5.0009e-04,3.5379e-02,1.2480e-04],['alt',1,8,'0,1,2|4,5|8,9|12,13|6,10,14,15',4.5708e-03,2.6662e-02,6.8920e-04,6.6895e-02,2.5423e-02],
   ['rand',1,4,'4,5|11,12,13|14,15',6.6667e-05,5.7128e-03,1.8332e-04,2.6544e-02,1.7470e-04],['rand',1,8,'0,1,2|4,5,7|9,11,12,13|14,15',2.5104e-04,4.7824e-02,7.4574e-04,5.2807e-02,3.5345e-04],
   ['rand',1,12,'0,1,2|3,4,5,7|8,9,11,12,13|6,10,14,15',2.2166e-02,3.6665e-02,5.5233e-02,2.2589e-02,2.5621e-02]];
  c1t.forEach(([m,sd,r,grp,mse,p0,q0,p6,q6])=>{const res=tomeLayer(m,r,sd);const gs=res.toks.filter(g=>g.length>1).map(g=>g.join(',')).join('|');const a0=attnCheck(res,0),a6=attnCheck(res,6);
    ok(gs===grp,`C1 ${m} r=${r} groups ${gs}`);ok(rel(unmergeMSE(res),mse),`C1 ${m} r=${r} mse`);ok(rel(a0.plain,p0)&&rel(a0.prop,q0)&&rel(a6.plain,p6)&&rel(a6.prop,q6),`C1 ${m} r=${r} attn`);ok(a0.dup<1e-12&&a6.dup<1e-12,`C1 ${m} r=${r} dup`)});
  ok(c1partition('rand',1).B.join()==='2,4,12,15'&&c1partition('rand',2).B.join()==='3,4,9,14','C1 ToMeSD dst');
  // C2
  ok(near(qshare(1024,768),.1818,.0001)&&near(qshare(256,768),.0526,.0001)&&near(qshare(16384,768),.7805,.0001),'C2 qshare');
  ok(near(coreBound(1024,768,0),2.8333,1e-4)&&near(coreBound(1024,768,128),2.4269,1e-4)&&near(coreBound(1024,768,256),2.1746,1e-4),'C2 bounds');
  [[256,768,0,.25,1.401,1.041,1.967,1.737],[1024,768,0,.5,1.375,1.100,1.545,1.412],[1024,768,0,.25,1.600,1.158,2.030,1.796],[1024,768,0,.0625,1.776,1.205,2.587,2.366],[4096,768,0,.25,2.345,1.545,2.185,1.943],
   [16384,768,0,.25,4.686,2.412,2.381,2.130],[1024,768,256,.25,1.513,1.150,1.730,1.583],[1024,384,0,.25,1.857,1.300,2.095,1.857],[1024,1536,0,.25,1.468,1.081,1.990,1.758]]
   .forEach(([n,d,t,k,a,b,c,e])=>{const s=speedups(n,d,t,k);ok(near(s.sa,a,.001)&&near(s.kv,b,.001)&&near(s.core,c,.001)&&near(s.nest,e,.001),`C2 N=${n} d=${d} T=${t} k=${k}: ${s.sa.toFixed(3)} ${s.kv.toFixed(3)} ${s.core.toFixed(3)} ${s.nest.toFixed(3)}`)});
  // C3
  ensureDendrograms(true);
  const sum=DATA.DEN.every(den=>den.every((m,i)=>i===0||den[i-1][2]>=m[2]-1e-12));ok(sum,'C3 dendrograms monotone');
  const c3t=[['grid',32,0,32,.723,1.144,1,8],['tome',32,0,32,.837,3.947,.219,21],['tome',32,1,32,1,5.460,.438,106],['rti',32,0,32,.901,1.913,1,48],['rti',32,1,32,1,1.244,1,16],
   ['thresh',32,0,48,.962,1.598,1,48],['thresh',32,1,16,1,1.498,1,16],['hit',32,0,62,.997,3.227,.887,44],['hit',32,1,2,1,6.113,0,128],['hit',16,0,30,.946,4.282,.933,108],['hit',16,1,2,1,6.113,0,128]];
  c3t.forEach(([m,R,b,K,ev,sp,cn,mx])=>{const s=grouping(m,R).st[b];ok(s.K===K&&s.maxsize===mx&&near(s.EV,ev,.001)&&near(s.spread,sp,.002)&&near(s.conn,cn,.001),`C3 ${m} R=${R} img${b}: K=${s.K} EV=${s.EV.toFixed(3)} sp=${s.spread.toFixed(3)} conn=${s.conn.toFixed(3)} max=${s.maxsize}`)});
  ok(near(grouping('hit',16).thr,.751434,1e-6)&&near(grouping('thresh',32).thr,2.491506,1e-6),'C3 equivalent thresholds');
  // C4
  [['rti',32,1,.098539,0,1,0],['rti',32,.95,.098539,.034741,.7598,.027658],['rti',32,.9,.098539,.138963,.4066,.062739],['rti',32,.8,.098539,.555852,.1266,.088722],
   ['grid',32,.95,.276807,.035253,.9270,.033746],['tome',32,.95,.163212,.034930,.8545,.031096],['hit',32,.95,.002524,.029874,.0730,.002354],['rti',64,.95,.017027,.029646,.3582,.011320]]
   .forEach(([m,R,t,e0,e1,bs,eb])=>{const r=restore(DATA.H[0],featsT(t).h,grouping(m,R).labs[0]);const cl=(a,b)=>b>=.01?near(a,b,.001):rel(a,b)||near(a,b,1e-6);
     ok(cl(r.E0,e0)&&cl(r.E1,e1)&&near(r.bs,bs,.001)&&cl(r.Eb,eb)&&r.Eb<=Math.min(r.E0,r.E1)+1e-12,`C4 ${m} R=${R} t=${t}: ${r.E0.toFixed(6)} ${r.E1.toFixed(6)} ${r.bs.toFixed(4)} ${r.Eb.toFixed(6)}`)});
  ok(TL_COUNT===23,'timeline cards = '+TL_COUNT);
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ---------- boot ---------- */
initTheme();initToc();renderStageBars();renderHil();initC1();initC2();initC3();initC4();initAx();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderC2();if(DATA.img){renderC3();renderC4()}})});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{if(DATA.img){renderC3();renderC4()}});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(TEST){renderC3();renderC4();selfTest()}
})();
