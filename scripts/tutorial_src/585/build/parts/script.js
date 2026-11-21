(()=>{
'use strict';
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
function el(tag,a,p){const e=document.createElementNS(NS,tag);if(a)for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function tx(p,x,y,s,a){const t=el('text',Object.assign({x,y},a||{}),p);t.textContent=s;return t}
const lin=(d0,d1,r0,r1)=>v=>r0+(v-d0)/(d1-d0)*(r1-r0);
function lp(pts){let s='';for(let i=0;i<pts.length;i++)s+=(i?'L':'M')+pts[i][0].toFixed(1)+' '+pts[i][1].toFixed(1);return s}
const neg=s=>s.replace('-','−');
const f2=v=>neg((Math.abs(v)<0.005?0:v).toFixed(2)),f3=v=>neg((Math.abs(v)<0.0005?0:v).toFixed(3)),f4=v=>neg((Math.abs(v)<0.00005?0:v).toFixed(4));
function widthOf(box,min,max){return Math.max(min,Math.min(max,Math.floor(box.clientWidth||max)))}
function sizeSvg(svg,W,H){svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H)}
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
const sig=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));

/* ================= C1: grouping math (same as verify/check.py) ================= */
function affinity(ang){const v=ang.map(a=>[Math.cos(a*Math.PI/180),Math.sin(a*Math.PI/180)]);
  return v.map((qi,i)=>v.map((qj,j)=>0.5*((qi[0]*qj[0]+qi[1]*qj[1])+(qj[0]*qi[0]+qj[1]*qi[1]))))}
function avgLinkage(S){const n=S.length;let cl=S.map((_,i)=>({m:[i],node:i}));const nodes=S.map((_,i)=>({leaf:i})),merges=[];
  while(cl.length>1){let best=null;
    for(let a=0;a<cl.length;a++)for(let b=a+1;b<cl.length;b++){let s=0;for(const i of cl[a].m)for(const j of cl[b].m)s+=S[i][j];s/=cl[a].m.length*cl[b].m.length;if(!best||s>best.s+1e-12)best={s,a,b}}
    const A=cl[best.a],B=cl[best.b];merges.push({s:best.s,U:A.m.slice(),V:B.m.slice()});nodes.push({l:A.node,r:B.node,s:best.s});
    cl[best.a]={m:A.m.concat(B.m).sort((x,y)=>x-y),node:nodes.length-1};cl.splice(best.b,1)}
  return {merges,nodes,n}}
function hardGroups(tree,tau){let cl=Array.from({length:tree.n},(_,i)=>[i]);
  for(const m of tree.merges){if(m.s<tau)break;cl=cl.filter(c=>!m.U.includes(c[0])&&!m.V.includes(c[0])&&!m.U.includes(c[c.length-1])&&!m.V.includes(c[c.length-1]));cl.push(m.U.concat(m.V).sort((x,y)=>x-y))}
  return cl.sort((a,b)=>a[0]-b[0])}
function softWeights(S,G,Tr){const A=S.map((row,i)=>{const e=G.map(g=>g.reduce((s,j)=>s+row[j],0)/g.length),mx=Math.max(...e),w=e.map(x=>Math.exp((x-mx)/Tr)),z=w.reduce((a,b)=>a+b,0);return w.map(x=>x/z)});
  const col=G.map((_,g)=>Math.max(A.reduce((s,r)=>s+r[g],0),1e-300));const P=A.map(r=>r.map((x,g)=>x/col[g]));return {A,P}}
function kSoft(ms,N,tau,Tb){return N-ms.reduce((s,x)=>s+sig((x-tau)/Tb),0)}

const C1_DEF=[10,22,150,38,165,178],GRID_POS=[[0,0],[0,2],[0,1],[1,2],[1,0],[1,1]];
const SHAPES=['circle','square','triangle','diamond','star','cross'],SHAPE_CH=['●','■','▲','◆','★','✚'];
const c1={ang:C1_DEF.slice(),tau:0.95,Tr:0.2,sel:0};
const trToV=t=>Math.log(t/0.02)/Math.log(50),vToTr=v=>0.02*Math.pow(50,v);
function shape(g,k,x,y,r,fillOp){const a={class:'gshape','fill':'var(--c-par)','fill-opacity':fillOp};let e;
  if(k==='circle')e=el('circle',Object.assign({cx:x,cy:y,r},a),g);
  else if(k==='square')e=el('rect',Object.assign({x:x-r*.85,y:y-r*.85,width:r*1.7,height:r*1.7},a),g);
  else{let pts;if(k==='triangle')pts=[[0,-1.1],[1,.75],[-1,.75]];else if(k==='diamond')pts=[[0,-1.15],[1.05,0],[0,1.15],[-1.05,0]];
    else if(k==='star'){pts=[];for(let i=0;i<10;i++){const rr=i%2?.48:1.15,t=-Math.PI/2+i*Math.PI/5;pts.push([rr*Math.cos(t),rr*Math.sin(t)])}}
    else pts=[[-.35,-1],[.35,-1],[.35,-.35],[1,-.35],[1,.35],[.35,.35],[.35,1],[-.35,1],[-.35,.35],[-1,.35],[-1,-.35],[-.35,-.35]];
    e=el('polygon',Object.assign({points:pts.map(p=>(x+p[0]*r).toFixed(1)+','+(y+p[1]*r).toFixed(1)).join(' ')},a),g)}
  return e}
const grpOp=g=>[.85,.55,.3,.12,.7,.42][g%6];
function c1compute(){const S=affinity(c1.ang),tree=avgLinkage(S),G=hardGroups(tree,c1.tau),W=softWeights(S,G,c1.Tr);
  const gOf=new Array(6);G.forEach((g,gi)=>g.forEach(i=>gOf[i]=gi));return {S,tree,G,...W,gOf,ms:tree.merges.map(m=>m.s)}}
function renderC1(){const r=c1compute();if(c1.sel>=r.G.length)c1.sel=0;
  $('#c1-tauo').textContent=f3(c1.tau);$('#c1-tro').textContent=c1.Tr.toFixed(3);
  $('#c1-trp').setAttribute('aria-pressed',String(Math.abs(c1.Tr-0.05)<1e-9));
  drawDisc(r);drawGrid(r);drawHeat(r);drawTree(r);drawA(r);
  const gs=r.G.map(g=>'{'+g.map(i=>i+1).join(',')+'}').join('、');
  $('#c1-ro').innerHTML=`<b>K = ${r.G.length}</b> · 群組 ${gs}<br>合併分數：${r.ms.map(f4).join('、')}`;
  $('#c1-ks').textContent=`K̃ = 6 − Σ sig((s_m − τ)/T_b) = ${f3(kSoft(r.ms,6,c1.tau,0.1))}（T_b = 0.1，Eq. 11）；硬群組數 K = ${r.G.length}`;
  announce('c1',`τ = ${f3(c1.tau)}，${r.G.length} 個群組：${gs}`);return r}
function drawDisc(r){const box=$('#c1-disc'),svg=$('svg',box),W=widthOf(box,240,320),H=W,cx=W/2,cy=H/2,R=W/2-34;sizeSvg(svg,W,H);
  el('circle',{cx,cy,r:R,class:'faint',fill:'none'},svg);
  el('line',{x1:cx-R-6,x2:cx+R+6,y1:cy,y2:cy,class:'faint'},svg);el('line',{y1:cy-R-6,y2:cy+R+6,x1:cx,x2:cx,class:'faint'},svg);
  tx(svg,cx+R+8,cy+4,'0°',{class:'tk'});tx(svg,cx-6,cy-R-10,'90°',{class:'tk','text-anchor':'end'});
  c1.ang.forEach((a,i)=>{const t=a*Math.PI/180,x=cx+R*Math.cos(t),y=cy-R*Math.sin(t);el('line',{x1:cx,y1:cy,x2:x,y2:y,class:'ink',style:'stroke-width:1.5'},svg)});
  c1.ang.forEach((a,i)=>{const t=a*Math.PI/180,x=cx+R*Math.cos(t),y=cy-R*Math.sin(t),gi=r.gOf[i];
    const h=el('g',{class:'hdl',tabindex:0,role:'slider','aria-label':`token ${i+1} 的角度`,'aria-valuemin':0,'aria-valuemax':359,'aria-valuenow':a,'aria-valuetext':`token ${i+1}：${a} 度，群組 ${gi+1}`,'data-i':i},svg);
    el('circle',{cx:x,cy:y,r:22,fill:'transparent'},h);el('circle',{cx:x,cy:y,r:15,class:'ring'},h);shape(h,SHAPES[gi],x,y,8,grpOp(gi));
    const lx=cx+(R+26)*Math.cos(t),ly=cy-(R+26)*Math.sin(t)+4;tx(h,lx,ly,String(i+1),{class:'labm','text-anchor':'middle'});
    h.addEventListener('pointerdown',e=>{e.preventDefault();h.setPointerCapture(e.pointerId);h.focus();
      const move=ev=>{const m=svg.getScreenCTM();if(!m)return;const p=new DOMPoint(ev.clientX,ev.clientY).matrixTransform(m.inverse());setAng(i,Math.round(Math.atan2(cy-p.y,p.x-cx)*180/Math.PI),true)};
      const up=()=>{removeEventListener('pointermove',move);removeEventListener('pointerup',up);removeEventListener('pointercancel',up);focusHandle(i)};
      addEventListener('pointermove',move);addEventListener('pointerup',up);addEventListener('pointercancel',up)});
    h.addEventListener('keydown',e=>{const st=e.shiftKey?10:1;let d=0;if(e.key==='ArrowRight'||e.key==='ArrowUp')d=st;else if(e.key==='ArrowLeft'||e.key==='ArrowDown')d=-st;else return;e.preventDefault();setAng(i,c1.ang[i]+d);focusHandle(i)})});
  svg.setAttribute('aria-label','六個 token 的方向：'+c1.ang.map((a,i)=>`token ${i+1} ${a} 度`).join('，'))}
function focusHandle(i){const h=$(`#c1-disc .hdl[data-i="${i}"]`);if(h)h.focus()}
function setAng(i,a,dragging){a=((Math.round(a)%360)+360)%360;if(a===c1.ang[i]&&dragging)return;c1.ang[i]=a;const inp=$('#c1-ang'+i);if(inp)inp.value=a;renderC1()}
function drawGrid(r){const box=$('#c1-grid'),svg=$('svg',box),W=widthOf(box,240,360),cw=(W-8)/3,ch=Math.min(80,cw*.75),H=ch*2+8;sizeSvg(svg,W,H);
  GRID_POS.forEach(([row,col],i)=>{const x=4+col*cw,y=4+row*ch,gi=r.gOf[i];el('rect',{x,y,width:cw-4,height:ch-4,rx:8,fill:'var(--surface-2)',stroke:'var(--line)'},svg);
    shape(svg,SHAPES[gi],x+cw/2-2,y+ch/2-8,11,grpOp(gi));tx(svg,x+cw/2-2,y+ch-12,`token ${i+1}`,{class:'tk','text-anchor':'middle',style:'fill:var(--ink-2)'})});
  svg.setAttribute('aria-label','2×3 patch 網格：第一列 token 1、3、2，第二列 token 5、6、4；'+r.G.map((g,gi)=>`群組 ${gi+1} 含 token ${g.map(i=>i+1).join('、')}`).join('；'))}
function drawHeat(r){const box=$('#c1-hm'),svg=$('svg',box),W=widthOf(box,260,340),pad=22,c=(W-pad)/6,H=pad+c*6;sizeSvg(svg,W,H);
  for(let i=0;i<6;i++){tx(svg,pad+c*i+c/2,14,String(i+1),{class:'tk','text-anchor':'middle'});tx(svg,10,pad+c*i+c/2+4,String(i+1),{class:'tk','text-anchor':'middle'})}
  for(let i=0;i<6;i++)for(let j=0;j<6;j++){const s=r.S[i][j],op=0.12+0.73*Math.abs(s),x=pad+c*j,y=pad+c*i;
    el('rect',{x:x+1,y:y+1,width:c-2,height:c-2,rx:4,fill:s>=0?'var(--c-par)':'var(--c-perp)','fill-opacity':op.toFixed(3)},svg);
    if(r.gOf[i]===r.gOf[j])el('rect',{x:x+1,y:y+1,width:c-2,height:c-2,rx:4,class:'hm-box'},svg);
    tx(svg,x+c/2,y+c/2+4,f2(s),{class:'hm-t','text-anchor':'middle',fill:op>0.5?'var(--surface)':'var(--ink)'})}
  svg.setAttribute('aria-label','親和度熱圖，正值藍、負值綠，數字為 S_ij；粗框為同一硬群組')}
function drawTree(r){const box=$('#c1-tree'),svg=$('svg',box),W=widthOf(box,260,340),H=230,m={l:44,r:10,t:12,b:26};sizeSvg(svg,W,H);
  const lo=Math.min(-1,...r.ms),yb=H-m.b,ys=m.t+(yb-m.t)*0.62,Y=v=>v>=0.8?lin(0.8,1,ys,m.t)(v):lin(lo,0.8,yb,ys)(v);const nodes=r.tree.nodes,root=nodes.length-1,order=[];
  (function dfs(k){const nd=nodes[k];if(nd.leaf!==undefined){order.push(nd.leaf);return}dfs(nd.l);dfs(nd.r)})(root);
  const step=(W-m.l-m.r)/6,X={};order.forEach((lf,o)=>X[lf]=m.l+step*(o+.5));const pos=[];
  [1,0.95,0.9,0.85,0.8,0,-1].forEach(v=>{const y=Y(v);el('line',{x1:m.l,x2:W-m.r,y1:y,y2:y,class:'grid'},svg);tx(svg,m.l-6,y+4,neg(String(v)),{class:'tk','text-anchor':'end'})});
  el('line',{x1:m.l,x2:W-m.r,y1:ys,y2:ys,class:'axl',style:'stroke-dasharray:2 3'},svg);
  nodes.forEach((nd,k)=>{if(nd.leaf!==undefined){pos[k]=[X[nd.leaf],Y(1)];tx(svg,X[nd.leaf],H-8,String(nd.leaf+1),{class:'labm','text-anchor':'middle'})}else{
    const a=pos[nd.l],b=pos[nd.r],y=Y(nd.s),kept=nd.s>=c1.tau,cls=kept?'ink':'ref';el('path',{d:`M${a[0]} ${a[1]}V${y}H${b[0]}V${b[1]}`,class:cls,style:kept?'stroke-width:2':''},svg);pos[k]=[(a[0]+b[0])/2,y]}});
  const ty=Y(Math.max(lo,Math.min(1,c1.tau)));el('line',{x1:m.l,x2:W-m.r,y1:ty,y2:ty,class:'acc dash',style:'stroke-width:1.5'},svg);tx(svg,W-m.r,ty-5,'τ = '+f3(c1.tau),{class:'tk','text-anchor':'end',style:'fill:var(--accent)'});
  svg.setAttribute('aria-label','合併樹：合併分數 '+r.ms.map(f2).join('、')+'；實線為保留的合併')}
function drawA(r){const t=$('#c1-a');let h='<thead><tr><th scope="col">token</th>'+r.G.map((g,gi)=>`<th scope="col"><button type="button" data-g="${gi}" aria-pressed="${gi===c1.sel}">${SHAPE_CH[gi]} 群組 ${gi+1} {${g.map(i=>i+1).join(',')}}</button></th>`).join('')+'</tr></thead><tbody>';
  r.A.forEach((row,i)=>{h+=`<tr><th scope="row">${SHAPE_CH[r.gOf[i]]} ${i+1}</th>`+row.map(v=>`<td><span class="cell"><i style="width:${(v*48).toFixed(1)}px"></i>${v.toFixed(3)}</span></td>`).join('')+'</tr>'});
  t.innerHTML=h+'</tbody>';$$('button',t).forEach(b=>b.addEventListener('click',()=>{c1.sel=+b.dataset.g;renderC1()}));
  const g=c1.sel;$('#c1-p').innerHTML=`群組 ${g+1} 的 token = Σᵢ P_i${g+1} h_i，P 行 = `+r.P.map((row,i)=>`<b>${row[g].toFixed(3)}</b>·h${i+1}`).join(' + ')}
function initC1(){const box=$('#c1-angs');c1.ang.forEach((a,i)=>{box.insertAdjacentHTML('beforeend',`<label>${i+1}<input id="c1-ang${i}" type="number" min="0" max="359" step="1" value="${a}" aria-label="token ${i+1} 的角度（度）"></label>`)});
  c1.ang.forEach((_,i)=>$('#c1-ang'+i).addEventListener('change',e=>{const v=parseInt(e.target.value,10);if(Number.isFinite(v))setAng(i,v);else e.target.value=c1.ang[i]}));
  $('#c1-tau').addEventListener('input',e=>{c1.tau=+e.target.value;renderC1()});
  const trs=$('#c1-tr');trs.value=trToV(c1.Tr);trs.addEventListener('input',e=>{c1.Tr=Math.round(vToTr(+e.target.value)*1000)/1000;renderC1()});
  $('#c1-trp').addEventListener('click',()=>{c1.Tr=Math.abs(c1.Tr-0.05)<1e-9?0.2:0.05;trs.value=trToV(c1.Tr);renderC1()});
  $('#c1-reset').addEventListener('click',()=>{c1.ang=C1_DEF.slice();c1.tau=0.95;c1.Tr=0.2;c1.sel=0;$('#c1-tau').value=0.95;trs.value=trToV(0.2);c1.ang.forEach((a,i)=>$('#c1-ang'+i).value=a);renderC1()});
  renderC1()}

/* ================= C2: batch budget ================= */
const C2_SC=JSON.parse($('#c2-data').textContent),C2_N=256;
function budget(i,rho,Tb){const tau=i/200;const Kh=C2_SC.map(sc=>C2_N-sc.filter(s=>s>=tau).length),Ks=C2_SC.map(sc=>kSoft(sc,C2_N,tau,Tb));
  const fh=Kh.reduce((a,b)=>a+b,0)/(4*C2_N),fs=Ks.reduce((a,b)=>a+b,0)/(4*C2_N);const L4=1+(fh-rho)*(fs-rho)/(rho*(1-rho)),r=1/rho,L10=r/(r-1)*((r-1)*fh*fs+(1-fh)*(1-fs)),g=(fh-rho)/(rho*(1-rho));
  return {tau,Kh,Ks,fh,fs,L4,L10,g}}
function calibrate(rho,Tb){for(let i=-40;i<=200;i++)if(budget(i,rho,Tb||0.1).fh>=rho)return i;return 200}
const c2={i:160,rho:0.5,Tb:0.1};
function renderC2(){const b=budget(c2.i,c2.rho,c2.Tb);$('#c2-tauo').textContent=b.tau.toFixed(3);$('#c2-tbo').textContent=c2.Tb.toFixed(2);
  $('#c2-tau').setAttribute('aria-valuetext','τ = '+b.tau.toFixed(3));
  $('#c2-bars').innerHTML=b.Kh.map((k,j)=>`<div class="brow"><span>樣本 ${j+1}</span><span class="btrack"><span class="bh" style="width:${(k/256*100).toFixed(2)}%"></span><span class="bs" style="width:${(b.Ks[j]/256*100).toFixed(2)}%"></span></span><span class="v">K = ${k}，K̃ = ${b.Ks[j].toFixed(1)}</span></div>`).join('');
  const box=$('#c2-axis'),svg=$('svg',box),W=widthOf(box,280,560),H=64,X=lin(0,1,16,W-16);sizeSvg(svg,W,H);
  el('line',{x1:X(0),x2:X(1),y1:30,y2:30,class:'axl'},svg);[0,.25,.5,.75,1].forEach(v=>{el('line',{x1:X(v),x2:X(v),y1:30,y2:34,class:'axl'},svg);tx(svg,X(v),50,String(v),{class:'tk','text-anchor':'middle'})});
  el('line',{x1:X(c2.rho),x2:X(c2.rho),y1:6,y2:34,class:'acc dash',style:'stroke-width:1.5'},svg);tx(svg,X(c2.rho)-4,12,'ρ',{class:'lab','text-anchor':'end',style:'fill:var(--accent)'});
  const tri=(v,fill,dy)=>el('path',{d:`M${X(v)} ${30+dy}l-6 ${dy>0?9:-9}h12z`,fill:fill?'var(--ink)':'var(--surface)',stroke:'var(--ink)','stroke-width':1.5},svg);
  tri(b.fh,true,-2);tri(b.fs,false,2);tx(svg,X(b.fh)+8,20,'f_h',{class:'tk'});tx(svg,X(b.fs),62,'f_s',{class:'tk','text-anchor':'middle'});
  svg.setAttribute('aria-label',`保留比例軸：f_h = ${b.fh.toFixed(4)}，f_s = ${b.fs.toFixed(4)}，ρ = ${c2.rho}`);
  $('#c2-ro').innerHTML=`f_h = <b>${b.fh.toFixed(4)}</b> · f_s = <b>${b.fs.toFixed(4)}</b><br>𝓛 = <b>${b.L4.toFixed(4)}</b>（Eq. 4）<br>∂𝓛/∂f_s = <b class="acc">${(b.g>=0?'+':'−')+Math.abs(b.g).toFixed(4)}</b>`;
  const done=Math.abs(b.fh-c2.rho)<1e-3;$('#c2-dir').textContent=done?'● 已達預算':b.g>0?'↓ 降低 f_s（使合併分數上升）':'↑ 提高 f_s（使合併分數下降）';
  $('#c2-l10').textContent=`Eq. 10：r/(r−1)·[(r−1)f_h f_s + (1−f_h)(1−f_s)] = ${b.L10.toFixed(4)}（r = ${1/c2.rho}）`;
  drawHist(b.tau);announce('c2',`τ = ${b.tau.toFixed(3)}，f_h = ${b.fh.toFixed(3)}，${done?'已達預算':b.g>0?'群組太多':'群組太少'}`)}
function drawHist(tau){const box=$('#c2-hist'),svg=$('svg',box),W=widthOf(box,280,900),rowH=46,H=rowH*4+24,m={l:58,r:10},X=lin(-1,1,m.l,W-m.r);sizeSvg(svg,W,H);
  C2_SC.forEach((sc,j)=>{const cnt=new Array(40).fill(0);sc.forEach(s=>{cnt[Math.min(39,Math.max(0,Math.floor((s+1)/2*40)))]++});const mx=Math.max(...cnt),y0=rowH*(j+1)-4;
    tx(svg,4,y0-12,'樣本 '+(j+1),{class:'al'});el('line',{x1:m.l,x2:W-m.r,y1:y0,y2:y0,class:'axl'},svg);
    cnt.forEach((c,k)=>{if(!c)return;const x0=X(-1+k*.05),x1=X(-1+(k+1)*.05),h=c/mx*(rowH-14);el('rect',{x:x0+.5,y:y0-h,width:Math.max(1,x1-x0-1),height:h,fill:'var(--c-par)','fill-opacity':.55},svg)});
    const xt=X(tau);el('line',{x1:xt,x2:xt,y1:y0-rowH+8,y2:y0,class:'acc',style:'stroke-width:1.5'},svg)});
  [-1,-.5,0,.5,1].forEach(v=>tx(svg,X(v),H-6,neg(String(v)),{class:'tk','text-anchor':'middle'}));tx(svg,X(tau)+4,12,'τ',{class:'lab',style:'fill:var(--accent)'})}
function initC2(){const s=$('#c2-tau');s.addEventListener('input',e=>{c2.i=+e.target.value;renderC2()});
  $$('#c2-rho button').forEach(b=>b.addEventListener('click',()=>{c2.rho=+b.dataset.v;$$('#c2-rho button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));renderC2()}));
  $('#c2-tb').addEventListener('input',e=>{c2.Tb=+e.target.value;renderC2()});
  $('#c2-cal').addEventListener('click',()=>{c2.i=calibrate(c2.rho,c2.Tb);s.value=c2.i;renderC2()});renderC2()}

/* ================= C3: cost model ================= */
const C3_COEF={a:7.285229e-3,b:1.536e-6,g:0.6229},COND=9;
const C3_PRESETS=[
 {t:'Table 4',n:'4×（3/12/3）',J:1,ratios:[4],layout:[3,12,3],gf:19.1,fid:9.59},
 {t:'Table 4',n:'2×4',J:2,ratios:[2,4],layout:[3,3,6,3,3],gf:20.8,fid:6.89},
 {t:'Table 4',n:'4×2',J:2,ratios:[4,2],layout:[3,3,6,3,3],gf:17.8,fid:9.54},
 {t:'Table 4',n:'4×4',J:2,ratios:[4,4],layout:[3,3,6,3,3],gf:17.1,fid:10.78},
 {t:'Table 4',n:'2×2（3/3/6/3/3）',J:2,ratios:[2,2],layout:[3,3,6,3,3],gf:22.2,fid:6.61},
 {t:'Table 5',n:'2/2/10/2/2',J:2,ratios:[2,2],layout:[2,2,10,2,2],gf:18.3,fid:8.48},
 {t:'Table 5',n:'2/3/8/3/2',J:2,ratios:[2,2],layout:[2,3,8,3,2],gf:19.3,fid:7.60},
 {t:'Table 5',n:'3/2/8/2/3',J:2,ratios:[2,2],layout:[3,2,8,2,3],gf:21.3,fid:6.81},
 {t:'Table 12',n:'2×2×2（3/3/2/2/2/3/3）',J:3,ratios:[2,2,2],layout:[3,3,2,2,2,3,3],gf:21.9,fid:6.66},
 {t:'Table 12',n:'2×2×4（3/3/2/2/2/3/3）',J:3,ratios:[2,2,4],layout:[3,3,2,2,2,3,3],gf:21.6,fid:6.95},
 {t:'Table 12',n:'2×2×2（2/2/2/6/2/2/2）',J:3,ratios:[2,2,2],layout:[2,2,2,6,2,2,2],gf:17.1,fid:8.82},
 {t:'對照',n:'不分組 12 區塊（對照，非論文設定）',J:0,ratios:[],layout:[12],gf:null,fid:null}];
const J_DEF={1:{ratios:[4],layout:[3,12,3]},2:{ratios:[2,2],layout:[3,3,6,3,3]},3:{ratios:[2,2,2],layout:[3,3,2,2,2,3,3]}};
function counts(cfg){const n=[256];for(let j=0;j<cfg.J;j++)n.push(n[j]/cfg.ratios[j]);return n}
function segTokens(cfg){const n=counts(cfg),J=cfg.J;return cfg.layout.map((b,i)=>({b,n:n[i<=J?i:2*J-i]}))}
function costModel(cfg){let L=0,Q=0;segTokens(cfg).forEach(({b,n})=>{const m=n+COND;L+=b*m;Q+=b*m*m});return {gf:C3_COEF.a*L+C3_COEF.b*Q+C3_COEF.g,L,Q}}
const same=(a,b)=>a.J===b.J&&a.ratios.join()===b.ratios.join()&&a.layout.join()===b.layout.join();
const c3={cfg:{J:2,ratios:[2,2],layout:[3,3,6,3,3]}};
function renderC3(){const cfg=c3.cfg,cm=costModel(cfg),hit=C3_PRESETS.find(p=>same(p,cfg));
  $$('#c3-presets button').forEach(b=>b.setAttribute('aria-pressed',String(same(C3_PRESETS[+b.dataset.i],cfg))));
  $$('#c3-J button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.v===cfg.J)));
  const st=$('#c3-stack'),segs=segTokens(cfg);st.innerHTML=segs.map((s,i)=>{const op=i===0?'':(i<=cfg.J?'<div class="sr-op">↓ 分組</div>':'<div class="sr-op">↓ 解組 <b>＋ 跳接</b></div>');
    return op+`<div class="sr-row${s.n===256?' full':''}" style="width:${Math.max(28,s.n/256*100).toFixed(1)}%">${s.b} 區塊 × ${+s.n.toFixed(2)} token</div>`}).join('');
  st.setAttribute('aria-label','token 數變化圖：'+segs.map(s=>`${s.b} 區塊 × ${+s.n.toFixed(2)} token`).join('，'));
  const rb=$('#c3-ratios');rb.innerHTML=cfg.ratios.map((r,j)=>`<div class="ctl-row"><span class="lbl">第 ${j+1} 階段壓縮比</span><div class="seg" role="group" aria-label="第 ${j+1} 階段壓縮比">${[2,4].map(v=>`<button type="button" data-j="${j}" data-v="${v}" aria-pressed="${v===r}">${v}×</button>`).join('')}</div></div>`).join('');
  $$('button',rb).forEach(b=>b.addEventListener('click',()=>{const c=JSON.parse(JSON.stringify(c3.cfg));c.ratios[+b.dataset.j]=+b.dataset.v;c3.cfg=c;renderC3()}));
  const bb=$('#c3-blocks'),J=cfg.J,names=i=>i===J?(J===0?'全部區塊':'精簡核心'):i===0?'全解析度（編碼＋解碼各）':`第 ${i} 層（編碼＋解碼各）`;
  let h='';for(let i=0;i<=J;i++)h+=`<div class="stepper"><span>${names(i)}</span><button type="button" data-i="${i}" data-d="-1" aria-label="${names(i)}減少一個區塊">−</button><output>${cfg.layout[i]}</output><button type="button" data-i="${i}" data-d="1" aria-label="${names(i)}增加一個區塊">+</button></div>`;
  bb.innerHTML=h;$$('button',bb).forEach(b=>b.addEventListener('click',()=>{const c=JSON.parse(JSON.stringify(c3.cfg)),i=+b.dataset.i,v=Math.max(1,Math.min(24,c.layout[i]+ +b.dataset.d));c.layout[i]=v;c.layout[2*c.J-i]=v;c3.cfg=c;renderC3()}));
  const tot=cfg.layout.reduce((a,b)=>a+b,0);
  $('#c3-ro').innerHTML=`推定 GFLOPs = <b class="acc">${cm.gf.toFixed(1)}</b>（${cm.gf.toFixed(2)}）<br>Σm = ${cm.L.toFixed(0)}，Σm² = ${cm.Q.toFixed(0)}，總區塊數 ${tot}<br>`+(hit&&hit.gf!=null?`論文（${hit.t}）：GFLOPs ${hit.gf}，FID ${hit.fid.toFixed(2)}`:hit?'本頁對照，論文沒有這個設定':'論文未測試此設定');
  drawScatter(cm.gf,hit);announce('c3',`推定 ${cm.gf.toFixed(1)} GFLOPs`+(hit&&hit.fid?`，論文 FID ${hit.fid}`:''))}
function drawScatter(cur,hit){const box=$('#c3-sc'),svg=$('svg',box),W=widthOf(box,280,900),H=220,m={l:46,r:16,t:16,b:40};sizeSvg(svg,W,H);
  const X=lin(16,26,m.l,W-m.r),Y=lin(5.6,11.2,H-m.b,m.t);
  [6,7,8,9,10,11].forEach(v=>{el('line',{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),class:'grid'},svg);tx(svg,m.l-6,Y(v)+4,String(v),{class:'tk','text-anchor':'end'})});
  [16,18,20,22,24,26].forEach(v=>tx(svg,X(v),H-m.b+16,String(v),{class:'tk','text-anchor':'middle'}));
  tx(svg,(m.l+W-m.r)/2,H-4,'推定 GFLOPs',{class:'al','text-anchor':'middle'});tx(svg,4,12,'論文 FID',{class:'al'});
  C3_PRESETS.forEach(p=>{if(p.fid==null)return;const x=X(costModel(p).gf),y=Y(p.fid),k=p.t==='Table 4'?'circle':p.t==='Table 5'?'square':'triangle',on=hit===p;
    const s=shape(svg,k,x,y,on?8:6,on?1:.25);if(on){s.setAttribute('fill','var(--accent)');s.setAttribute('stroke','var(--accent)')}});
  const xc=X(Math.max(16,Math.min(26,cur)));el('line',{x1:xc,x2:xc,y1:m.t,y2:H-m.b,class:'acc dash',style:'stroke-width:1.5'},svg);tx(svg,xc+4,m.t+10,'目前 '+cur.toFixed(1),{class:'tk',style:'fill:var(--accent)'})}
function initC3(){const box=$('#c3-presets');let last='';C3_PRESETS.forEach((p,i)=>{if(p.t!==last){box.insertAdjacentHTML('beforeend',`<span class="grp-l">${p.t}</span>`);last=p.t}
    box.insertAdjacentHTML('beforeend',`<button class="chip" type="button" data-i="${i}" aria-pressed="false">${p.n}</button>`)});
  $$('button',box).forEach(b=>b.addEventListener('click',()=>{const p=C3_PRESETS[+b.dataset.i];c3.cfg={J:p.J,ratios:p.ratios.slice(),layout:p.layout.slice()};renderC3()}));
  $$('#c3-J button').forEach(b=>b.addEventListener('click',()=>{const J=+b.dataset.v;c3.cfg={J,ratios:J_DEF[J].ratios.slice(),layout:J_DEF[J].layout.slice()};renderC3()}));renderC3()}

/* ================= C4: time shift and schedule ================= */
function shiftT(t,k){const s=1-t;return 1-k*s/(1+(k-1)*s)}
const rnd9=t=>Math.round(t*1e9)/1e9;
function scheduleCounts(k){const g=[];for(let i=0;i<=50;i++)g.push(shiftT(i/50,k));const st=g.slice(0,50).map(rnd9);
  return {g,noisy:st.filter(t=>t<0.5).length,cfg:st.filter(t=>t>=0.1&&t<=1).length,ru:st.filter(t=>t>=0.85).length}}
const ruSchedule=t=>t>=0.85?[0.5,1]:[1,1];
const c4={k:1.7};
function renderC4(){const k=c4.k,sc=scheduleCounts(k);$('#c4-ko').textContent=k.toFixed(2);$('#c4-k').setAttribute('aria-valuetext','k = '+k.toFixed(2));
  $('#c4-hit').setAttribute('aria-pressed',String(Math.abs(k-1.7)<1e-9));$('#c4-uni').setAttribute('aria-pressed',String(Math.abs(k-1)<1e-9));
  const box=$('#c4-fig'),svg=$('svg',box),W=widthOf(box,300,900),H=268,m={l:44,r:16},X=lin(0,1,m.l,W-m.r);sizeSvg(svg,W,H);
  el('rect',{x:X(0),y:6,width:X(0.1)-X(0),height:H-52,fill:'var(--surface-2)'},svg);
  tx(svg,X(0)+4,20,'不引導',{class:'tk'});
  const uni=scheduleCounts(1).g;uni.forEach(t=>el('line',{x1:X(t),x2:X(t),y1:72,y2:82,class:'axl'},svg));
  sc.g.forEach(t=>el('line',{x1:X(t),x2:X(t),y1:44,y2:60,stroke:'var(--ink)','stroke-width':1.5},svg));
  el('line',{x1:X(0),x2:X(1),y1:62,y2:62,class:'axl'},svg);
  tx(svg,X(1),40,'目前網格',{class:'tk','text-anchor':'end'});tx(svg,X(1),96,'均勻網格（k = 1）',{class:'tk','text-anchor':'end'});
  [[0.1,'0.1'],[0.5,'0.5'],[0.85,'0.85']].forEach(([v,l])=>{el('line',{x1:X(v),x2:X(v),y1:30,y2:H-46,class:'cur'},svg);tx(svg,X(v),H-30,l,{class:'tk','text-anchor':'middle'})});
  const Y=lin(0.4,1.3,H-50,110);[0.5,1,1.2].forEach(v=>{el('line',{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),class:'grid'},svg);tx(svg,m.l-6,Y(v)+4,String(v),{class:'tk','text-anchor':'end'})});
  tx(svg,4,108,'倍率',{class:'al'});
  el('path',{d:lp([[X(.1),Y(1.2)],[X(1),Y(1.2)]]),class:'acc'},svg);
  el('path',{d:lp([[X(.1),Y(1)],[X(.85),Y(1)],[X(.85),Y(.5)],[X(1),Y(.5)]]),class:'perp dash'},svg);
  el('path',{d:lp([[X(.1),Y(1)+5],[X(1),Y(1)+5]]),class:'perp dot',style:'stroke:var(--ink-3)'},svg);
  tx(svg,X(0),H-2,'t = 0 純噪聲',{class:'tk'});tx(svg,X(1),H-2,'t = 1 乾淨影像',{class:'tk','text-anchor':'end'});
  svg.setAttribute('aria-label',`時間網格：k = ${k.toFixed(2)}，${sc.noisy} 步在 t < 0.5；倍率 r_c,1 = 1.2，r_u,1 在 t ≥ 0.85 為 0.5、其餘為 1`);
  const mb=$('#c4-map'),ms=$('svg',mb),w2=widthOf(mb,200,280),h2=w2;sizeSvg(ms,w2,h2);const X2=lin(0,1,36,w2-10),Y2=lin(0,1,h2-30,10);
  [0,.5,1].forEach(v=>{tx(ms,X2(v),h2-12,String(v),{class:'tk','text-anchor':'middle'});tx(ms,30,Y2(v)+4,String(v),{class:'tk','text-anchor':'end'})});
  el('path',{d:lp([[X2(0),Y2(0)],[X2(1),Y2(1)]]),class:'ref'},ms);const pts=[];for(let i=0;i<=100;i++){const t=i/100;pts.push([X2(t),Y2(shiftT(t,k))])}el('path',{d:lp(pts),class:'tot'},ms);
  tx(ms,X2(.5),h2-0,'t',{class:'al','text-anchor':'middle'});tx(ms,X2(0)+4,Y2(1)+4,"t′",{class:'al'});
  ms.setAttribute('aria-label','t 到 t′ 的映射曲線與對角線');
  $('#c4-ro').innerHTML=`起點 t′ &lt; 0.5：<b>${sc.noisy}</b> 步<br>起點在 [0.1, 1]（引導）：<b>${sc.cfg}</b> 步<br>起點 t′ ≥ 0.85（r_u = (0.5, 1)）：<b>${sc.ru}</b> 步<br>t′₁ = ${sc.g[1].toFixed(4)} · t′₂₅ = ${sc.g[25].toFixed(4)} · t′₄₉ = ${sc.g[49].toFixed(4)}<br>每步兩次網路評估：v_c^(r_c) 與 v_∅^(r_u)，與 CFG 相同`;
  announce('c4',`k = ${k.toFixed(2)}：${sc.noisy} 步在 t < 0.5，${sc.ru} 步使用 r_u = (0.5, 1)`)}
function initC4(){const s=$('#c4-k');s.addEventListener('input',e=>{c4.k=+e.target.value;renderC4()});
  $('#c4-hit').addEventListener('click',()=>{c4.k=1.7;s.value=1.7;renderC4()});$('#c4-uni').addEventListener('click',()=>{c4.k=1;s.value=1;renderC4()});renderC4()}

/* ================= timeline ================= */
const TL_GROUP={'2112.10752':['lat','hit-latent-hierarchy-and-pixel-space'],'2212.09748':['lat','hit-latent-hierarchy-and-pixel-space'],'2410.06940':['lat','hit-latent-hierarchy-and-pixel-space'],'2510.11690':['lat','hit-latent-hierarchy-and-pixel-space'],
 '2301.11093':['pix','hit-latent-hierarchy-and-pixel-space'],'2410.19324':['pix','hit-latent-hierarchy-and-pixel-space'],'2504.07963':['pix','hit-latent-hierarchy-and-pixel-space'],'2511.13720':['pix','hit-latent-hierarchy-and-pixel-space'],
 '2210.09461':['merge','hit-merging-baselines-fixed-rates'],'2303.17604':['merge','hit-merging-baselines-fixed-rates'],'2202.11094':['grp','hit-affinity-grouping'],
 '2412.09871':['chunk','hit-hnet-blt-to-image-groups'],'2507.07955':['chunk','hit-hnet-blt-to-image-groups'],'2601.01608':['guide','hit-group-guidance'],'2611.22962':['me','hit-lineage-hub']};
const TL_NAME={lat:'潛空間階層',pix:'像素空間',merge:'token 合併',grp:'圖文分組',chunk:'序列切塊',guide:'稀疏引導',me:'本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.map((e,i)=>({e,i})).sort((a,b)=>(+a.e.year||0)-(+b.e.year||0)||a.i-b.i).map(o=>o.e);
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];used.add(g[0]);const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${g[1]?`<a href="${g[1]}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
  const on=new Set(used);
  Object.keys(TL_NAME).filter(k=>used.has(k)).forEach(k=>{const b=document.createElement('button');b.type='button';b.className='chip';b.setAttribute('aria-pressed','true');b.textContent=TL_NAME[k];b.dataset.g=k;
    b.addEventListener('click',()=>{if(on.has(k)&&on.size===used.size){on.clear();on.add(k)}else if(on.has(k)){on.delete(k);if(!on.size)used.forEach(x=>on.add(x))}else on.add(k);
      $$('.chip',chips).forEach(c=>c.setAttribute('aria-pressed',String(on.has(c.dataset.g))));$$('.ev',track).forEach(ev=>ev.classList.toggle('dim',!on.has(ev.dataset.g)))});chips.appendChild(b)});
  const rail=$('#rail'),by=dir=>rail.scrollBy({left:dir*Math.min(rail.clientWidth*.8,560),behavior:RM?'auto':'smooth'});
  $('#tl-prev').addEventListener('click',()=>by(-1));$('#tl-next').addEventListener('click',()=>by(1));
  rail.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){by(1);e.preventDefault()}if(e.key==='ArrowLeft'){by(-1);e.preventDefault()}})}

/* ================= toc, theme ================= */
function initToc(){const secs=$$('[data-toc]'),ol=$('#toc'),bar=$('#tocbar');
  secs.forEach(s=>{ol.insertAdjacentHTML('beforeend',`<li><a href="#${s.id}"><span class="n">${s.dataset.toc}</span><span>${s.dataset.tt}</span></a></li>`);bar.insertAdjacentHTML('beforeend',`<a href="#${s.id}">${s.dataset.toc} ${s.dataset.tt}</a>`)});
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;$$('.toc a,.tocbar a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+id));const ab=$(`.tocbar a[href="#${id}"]`);if(ab&&bar.offsetParent)bar.scrollTo({left:ab.offsetLeft-16,behavior:RM?'auto':'smooth'})}})},{rootMargin:'-40% 0px -55% 0px'});
  secs.forEach(s=>io.observe(s));addEventListener('scroll',()=>$('#appbar').classList.toggle('raised',scrollY>8),{passive:true})}
function initTheme(){let saved=null;try{saved=localStorage.getItem('pm-theme')}catch(e){}if(saved)document.documentElement.dataset.theme=saved;
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){}})}

/* ================= self test (design.md section 5) ================= */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t)=>Math.abs(a-b)<=t+1e-12,rel=(a,b,r,ab)=>Math.abs(a-b)<=Math.max(r*Math.abs(b),ab);
  const run=(ang,tau,Tr)=>{const S=affinity(ang),tree=avgLinkage(S),G=hardGroups(tree,tau),W=softWeights(S,G,Tr);return {S,tree,G,A:W.A,P:W.P,ms:tree.merges.map(m=>m.s)}};
  const gstr=G=>G.map(g=>g.map(i=>i+1).join(',')).join('|');
  let r=run(C1_DEF,0.95,0.2);ok(gstr(r.G)==='1,2|3|4|5,6','C1 default groups '+gstr(r.G));ok(near(kSoft(r.ms,6,0.95,0.1),4.002,0.001),'C1 Ksoft default');
  [0.40354,0.00061676,0.59571,0.00013135].forEach((v,g)=>ok(v>=0.01?rel(r.A[3][g],v,0.005,0):near(r.A[3][g],v,0.0005),'C1 A[4]['+g+']='+r.A[3][g]));
  [0.40152,0.34096,8.0543e-5,0.25738,3.3982e-5,2.539e-5].forEach((v,i)=>ok(v>=0.01?rel(r.P[i][0],v,0.005,0):near(r.P[i][0],v,0.0005),'C1 P col0 i='+i));
  [0.9781,0.9744,0.9244,0.9221,-0.7468].forEach((v,i)=>ok(near(r.ms[i],v,1e-4),'C1 merge '+i));
  ok(gstr(run(C1_DEF,0.9,0.2).G)==='1,2,4|3,5,6','C1 tau=0.9 groups');ok(near(kSoft(run(C1_DEF,0.9,0.2).ms,6,0.9,0.1),3.520,0.001),'C1 Ksoft 0.9');
  ok(run(C1_DEF,0.975,0.2).G.length===5&&run(C1_DEF,0.98,0.2).G.length===6&&run(C1_DEF,-1,0.2).G.length===1,'C1 K at 0.975/0.98/-1');
  [[60,4.345645,0.19297],[90,4.431869,0.023634],[120,4.356455,0.0012173]].forEach(([a4,ks,a])=>{const ang=C1_DEF.slice();ang[3]=a4;const q=run(ang,0.95,0.2);ok(near(kSoft(q.ms,6,0.95,0.1),ks,0.001)&&(a>=0.01?rel(q.A[3][0],a,0.005,0):near(q.A[3][0],a,0.0005)),'C1 a4='+a4)});
  const S3=affinity([10,22,150]);ok(near((S3[0][2]+S3[1][2])/2,-0.6909,1e-4)&&near(S3[1][2],-0.6157,1e-4),'3-token example');
  let mono=true;for(let s=1;s<=200;s++){let x=s*2654435761>>>0;const ang=[];for(let i=0;i<8;i++){x=(x*1103515245+12345)>>>0;ang.push(x%360)}const ms=avgLinkage(affinity(ang)).merges.map(m=>m.s);for(let i=1;i<ms.length;i++)if(ms[i]>ms[i-1]+1e-12)mono=false}ok(mono,'merge scores non-increasing');
  const B=[[160,.5,[63,112,193,241],[84.6,119.7,176.3,222.8],.5947,.5894,1.0339,.3789],[120,.5,[11,22,80,173],[26.1,46.2,95.9,161.4],.2793,.3219,1.1572,-.8828],[180,.5,[123,166,227,253],[130.7,163.9,208.9,239.4],.7510,.7255,1.2264,1.0039],
    [148,.5,[35,81,163,229],[61.9,94.2,153.0,208.3],.4961,.5053,.9999,-.0156],[149,.5,[36,85,165,230],[63.6,96.3,155.0,209.6],.5039,.5122,1.0002,.0156],[117,.25,[9,18,71,161],[23.6,42.3,90.1,155.6],.2529,.3043,1.0008,.0156],[160,.25,[63,112,193,241],[84.6,119.7,176.3,222.8],.5947,.5894,1.6239,1.8385]];
  B.forEach(([i,rho,kh,ks,fh,fs,L,g])=>{const b=budget(i,rho,0.1);ok(b.Kh.join()===kh.join(),`C2 Kh i=${i} rho=${rho}: ${b.Kh}`);ok(ks.every((v,j)=>near(b.Ks[j],v,0.05)),`C2 Ks i=${i}`);
    ok(near(b.fh,fh,1e-4)&&near(b.fs,fs,1e-4)&&near(b.L4,L,1e-4)&&rel(b.g,g,1e-3,1e-4),`C2 fh/fs/L/g i=${i} rho=${rho}`);ok(Math.abs(b.L4-b.L10)<1e-12,'C2 Eq4=Eq10 i='+i)});
  ok(calibrate(0.5)===149&&calibrate(0.25)===117,'C2 calibrate');
  const exp=[19.33,20.82,17.90,17.19,22.26,18.28,19.26,21.28,21.78,21.54,16.85,25.08];C3_PRESETS.forEach((p,i)=>ok(near(costModel(p).gf,exp[i],0.01),`C3 ${p.n}: ${costModel(p).gf.toFixed(3)}`));
  [[1,25,45,7,.02,.5,.98],[1.7,32,42,4,.0118624,.37037,.966469],[3,38,37,2,.00675676,.25,.942308]].forEach(([k,a,b,c,t1,t25,t49])=>{const s=scheduleCounts(k);ok(s.noisy===a&&s.cfg===b&&s.ru===c,`C4 counts k=${k}: ${s.noisy}/${s.cfg}/${s.ru}`);
    ok(rel(s.g[1],t1,0.005,0)&&near(s.g[25],t25,1e-4)&&near(s.g[49],t49,1e-4),'C4 grid k='+k)});
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  ok(!!document.querySelector('.katex')||typeof renderMathInElement!=='function','KaTeX rendered');
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'')}

/* ================= boot ================= */
initTheme();initToc();initC1();initC2();initC3();initC4();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderC1();renderC2();renderC3();renderC4()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg','option'],throwOnError:false})}
const boot=()=>{renderMath();if(/[?&]test\b/.test(location.search))selfTest()};
if(document.readyState==='complete')boot();else addEventListener('load',boot);
})();
