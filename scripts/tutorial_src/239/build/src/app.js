(()=>{
'use strict';
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
const TEST=/[?&]test\b/.test(location.search);
function el(tag,a,p){const e=document.createElementNS(NS,tag);if(a)for(const k in a){if(k==='stroke-width')e.style.strokeWidth=a[k];else e.setAttribute(k,a[k])}if(p)p.appendChild(e);return e}
function tx(p,x,y,s,a){const t=el('text',Object.assign({x,y},a||{}),p);t.textContent=s;return t}
const lin=(d0,d1,r0,r1)=>v=>r0+(v-d0)/(d1-d0)*(r1-r0);
const lgs=(d0,d1,r0,r1)=>{const a=Math.log(d0),b=Math.log(d1);return v=>r0+(Math.log(v)-a)/(b-a)*(r1-r0)};
function lp(pts){let s='';for(let i=0;i<pts.length;i++)s+=(i?'L':'M')+pts[i][0].toFixed(1)+' '+pts[i][1].toFixed(1);return s}
function widthOf(box,min,max){return Math.max(min,Math.min(max,Math.floor(box.clientWidth||max)))}
let clipN=0;
/* frame: axes, grid, ticks; optional log axes; returns scales and a clipped group for data */
function frame(svg,o){
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${o.W} ${o.H}`);svg.setAttribute('width',o.W);svg.setAttribute('height',o.H);
  const m=o.m,x0=m.l,x1=o.W-m.r,y0=o.H-m.b,y1=m.t;
  const X=(o.xlog?lgs:lin)(o.x[0],o.x[1],x0,x1),Y=(o.ylog?lgs:lin)(o.y[0],o.y[1],y0,y1);
  const g=el('g',null,svg);
  (o.yt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:String(v),yy=Y(val);el('line',{x1:x0,x2:x1,y1:yy,y2:yy,class:'grid'},g);tx(g,x0-6,yy+4,lab,{class:'tk','text-anchor':'end'})});
  (o.xt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:String(v),xx=X(val);el('line',{x1:xx,x2:xx,y1:y0,y2:y0+4,class:'axl'},g);tx(g,xx,y0+18,lab,{class:'tk','text-anchor':'middle'})});
  el('line',{x1:x0,x2:x1,y1:y0,y2:y0,class:'axl'},g);el('line',{x1:x0,x2:x0,y1:y0,y2:y1,class:'axl'},g);
  if(o.xl)tx(g,(x0+x1)/2,o.H-4,o.xl,{class:'al','text-anchor':'middle'});
  if(o.yl)tx(g,4,12,o.yl,{class:'al'});
  const id='clip'+(++clipN);const cp=el('clipPath',{id},el('defs',null,svg));el('rect',{x:x0,y:y1-2,width:x1-x0,height:y0-y1+4},cp);
  const d=el('g',{'clip-path':`url(#${id})`},svg);
  return {X,Y,g,d,x0,x1,y0,y1};
}
function marker(p,shape,x,y,cls,r=4.5){
  if(shape==='c')return el('circle',{cx:x,cy:y,r,class:cls},p);
  if(shape==='s')return el('rect',{x:x-r,y:y-r,width:2*r,height:2*r,class:cls},p);
  if(shape==='t')return el('path',{d:`M${x} ${y-r-1}L${x+r+1} ${y+r}L${x-r-1} ${y+r}Z`,class:cls},p);
  if(shape==='d')return el('path',{d:`M${x} ${y-r-1}L${x+r+1} ${y}L${x} ${y+r+1}L${x-r-1} ${y}Z`,class:cls},p);
  return el('path',{d:`M${x-r} ${y-r}L${x+r} ${y+r}M${x+r} ${y-r}L${x-r} ${y+r}`,class:cls,'stroke-width':2},p);
}
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function setAria(svg,s){clearTimeout(svg._t);svg._t=setTimeout(()=>svg.setAttribute('aria-label',s),500)}
function fmt(v){if(v===Infinity)return'∞';if(!isFinite(v))return'—';return Math.abs(v)>=1e4?v.toExponential(3):(Math.abs(v)<5e-5?0:v).toFixed(4)}
const sig=v=>v>=1e4||v<1e-3?v.toExponential(2):+v.toPrecision(4)+'';
function rot(box,rows){box.innerHTML=rows.map(r=>`<div${r[2]?' class="hl"':''}><span>${r[0]}</span><b>${r[1]}</b></div>`).join('')}
const W_VALUES=[0,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.5,0.7,1,1.5,2,3,5,7,10,20,50,100];
const POS_VALUES=[0.01,0.02,0.05,0.1,0.2,0.5,1,1.5,2,4,5,10,20,50,100];
function bindList(inp,vals,def,out,cb,fv=String){inp.min=0;inp.max=vals.length-1;inp.step=1;inp.value=vals.indexOf(def);
  const upd=()=>{const v=vals[+inp.value];out.textContent=fv(v);inp.setAttribute('aria-valuetext',fv(v))};upd();
  inp.addEventListener('input',()=>{upd();cb()});return()=>vals[+inp.value]}
function bindRange(inp,out,cb,fv){const upd=()=>{out.textContent=fv(+inp.value);inp.setAttribute('aria-valuetext',fv(+inp.value))};upd();inp.addEventListener('input',()=>{upd();cb()});return()=>+inp.value}
function seg(node,cb){$$('button',node).forEach(b=>b.addEventListener('click',()=>{$$('button',node).forEach(x=>x.setAttribute('aria-pressed',String(x===b)));cb(b.dataset.v)}))}
/* chunked runner: interactive heavy loops; ?test calls the synchronous cores directly */
const jobs={};
function chunkRun(key,total,per,step,done){const id=(jobs[key]||0)+1;jobs[key]=id;let i=0;
  const go=()=>{if(jobs[key]!==id)return;const e=Math.min(total,i+per);step(i,e);i=e;if(i<total)setTimeout(go,0);else done()};go()}
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}

/* ---------- hero w axis ---------- */
function waxis(node){node.innerHTML='<div class="kx-track"><div class="kx-line"></div></div>';const tr=node.firstChild;
  const pos=w=>w===0?0:w===Infinity?100:12+(Math.log10(w)+3)/5*76;
  [[0,'0','x 預測','l'],[0.001,'10⁻³','','mn'],[1,'1','速度',''],[100,'10²','','mn'],[Infinity,'∞','ε 預測','r']].forEach(([w,n,v,c])=>{const s=document.createElement('span');s.className='kx-tick '+c;s.style.left=pos(w)+'%';s.innerHTML=`<b>${n}</b>${v?' '+v:''}`;tr.appendChild(s)});
  tr.style.marginTop='16px'}

/* ---------- S2 target family figure ---------- */
function renderFam(){const box=$('#fam-fig'),svg=$('svg',box),W=widthOf(box,300,920),cols=W>=640?5:W>=420?3:2;
  const LAM=Array.from({length:24},(_,i)=>50*0.75**i);
  const P=[['x 預測','w = 0',()=>0],['速度','w = 1',()=>1],['純量內插','w ≡ 0.5',()=>0.5],['AsymFlow','w ∈ {0, 1}',i=>i<6?1:0],['AnisoFlow','w = 0.15λ（示意）',i=>0.15*LAM[i]]];
  const pw=W/cols,ph=130,rows=Math.ceil(P.length/cols),H=rows*ph;svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  P.forEach(([name,sub,f],k)=>{const cx=(k%cols)*pw,cy=Math.floor(k/cols)*ph,x0=cx+10,x1=cx+pw-10,yb=cy+72,yt=cy+14,bw=(x1-x0)/24;
    el('line',{x1:x0,x2:x1,y1:yb,y2:yb,class:'axl'},svg);el('line',{x1:x0,x2:x1,y1:yt,y2:yt,class:'grid'},svg);
    for(let i=0;i<24;i++){const w=f(i),h=Math.min(w,1)*(yb-yt);if(h>0)el('rect',{x:x0+i*bw+bw*.15,y:yb-h,width:bw*.7,height:h,class:k===4?'acc-f':'par-f'},svg);if(w>1)el('path',{d:`M${x0+i*bw+bw/2} ${yt-8}l3 5h-6z`,class:'acc-f'},svg)}
    tx(svg,x0,yb+20,name,{class:'lab'});tx(svg,x0,yb+38,sub,{class:'tk'})});
  if(!svg._lab){svg._lab=1}}

/* ---------- C1 recovery & amplification ---------- */
const c1={seed:239};
function c1calc(t,w,d,seed,clamp){const r=mulberry32(seed),x=gauss(r),e=gauss(r),a=1-t,s=t,z=a*x+s*e,uW=w*e-x,uWh=uW+d,den=a*w+s,u=e-x;
  const dc=clamp?Math.max(den,0.04):den,uh=((1-w)*z+uWh)/dc;
  return{x,e,a,s,z,uW,uWh,den,dc,uh,u,err:uh-u,amp:1/dc,bias:u*(den/dc-1),noise:d/dc,pe:a*w/den,px:s/den}}
function initC1(){const ti=$('#c1-t'),wi=$('#c1-w'),di=$('#c1-d');
  c1.gt=bindRange(ti,$('#c1-to'),renderC1,v=>v.toFixed(2));c1.gw=bindList(wi,W_VALUES,1,$('#c1-wo'),renderC1);c1.gd=bindRange(di,$('#c1-do'),renderC1,v=>v.toFixed(2));
  $$('#c1-chips .chip').forEach(b=>b.addEventListener('click',()=>{wi.value=W_VALUES.indexOf(+b.dataset.w);wi.dispatchEvent(new Event('input'))}));
  $('#c1-cl').addEventListener('change',renderC1);$('#c1-re').addEventListener('click',()=>{c1.seed++;renderC1()});renderC1()}
function renderC1(){const t=c1.gt(),w=c1.gw(),d=c1.gd(),cl=$('#c1-cl').checked,o=c1calc(t,w,d,c1.seed,cl);
  $$('#c1-chips .chip').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.w===w)));
  const box=$('#c1-fig'),svg=$('svg',box),W=widthOf(box,280,560),H=280;
  const F=frame(svg,{W,H,m:{l:46,r:12,t:30,b:40},x:[0,1],y:[0.01,100],ylog:true,xt:[0,.25,.5,.75,1].map(v=>[v,String(v)]),yt:[[0.01,'0.01'],[0.1,'0.1'],[1,'1'],[10,'10'],[100,'100']],xl:'時間 t（0 = 資料，1 = 雜訊）',yl:'誤差倍率 1/(αw+σ)'});
  const curve=(ww,c)=>{const pts=[];for(let j=4;j<=400;j++){const tt=j/400;let den=(1-tt)*ww+tt;if(c)den=Math.max(den,0.04);pts.push([F.X(tt),F.Y(1/den)])}return pts};
  const draw=(ww,cls,shape,c,wd)=>{const p=curve(ww,c);el('path',{d:lp(p),class:cls,'stroke-width':wd||2.5},F.d);for(let j=0;j<p.length;j+=50)marker(F.d,shape,p[j][0],p[j][1],cls.split(' ')[0]+'-f',3.5)};
  draw(0,'par dash','c',false);draw(1,'ink dot','s',false,2);
  if(cl){const yy=F.Y(25);el('line',{x1:F.x0,x2:F.x1,y1:yy,y2:yy,class:'ref'},F.g);tx(F.g,F.x1-4,yy-6,'σmin = 0.04 → 25×',{class:'tk','text-anchor':'end'})}
  draw(w,'acc','t',cl,3.5);
  const xt=F.X(t);el('line',{x1:xt,x2:xt,y1:F.y0,y2:F.y1,class:'cur'},F.g);
  const lab=w===0?'目前 = x 預測':w===1?'目前 = 速度':'目前 w = '+w;const yl=F.Y(Math.min(100,Math.max(0.01,o.amp)));tx(F.g,Math.min(xt+6,F.x1-90),Math.max(F.y1+12,yl-8),lab,{class:'lab'});
  const rows=[['α',fmt(o.a)],['σ',fmt(o.s)],['x',fmt(o.x)],['ε',fmt(o.e)],['z',fmt(o.z)],['u<sub>W</sub> = wε − x',fmt(o.uW)],['û<sub>W</sub> = u<sub>W</sub> + δ',fmt(o.uWh)],['還原 û',fmt(o.uh),1],['真實 u',fmt(o.u)],['使用的分母',fmt(o.dc)],['速度誤差 û − u',fmt(o.err),1],['誤差倍率',fmt(o.amp)],['π<sub>ε</sub>',fmt(o.pe)],['π<sub>x</sub>',fmt(o.px)]];
  if(cl)rows.push(['截斷偏差 = u(den/den′ − 1)',fmt(o.bias)],['放大後誤差 = δ/den′',fmt(o.noise)]);
  rot($('#c1-ro'),rows);
  const clamped=cl&&o.den<0.04;
  $('#c1-tpl').textContent=clamped?`速度誤差 = 偏差 ${fmt(o.bias)} + δ / den′ = ${fmt(o.bias)} + ${fmt(d)} / ${fmt(o.dc)} = ${fmt(o.err)}`:`速度誤差 = δ / den = ${fmt(d)} / ${fmt(o.dc)} = ${fmt(o.err)}`;
  announce('c1',`t = ${t.toFixed(2)}，w = ${w}，速度誤差 ${fmt(o.err)}`);setAria(svg,`誤差倍率曲線；目前 w = ${w}，t = ${t.toFixed(2)} 時倍率 ${fmt(o.amp)}`)}

/* ---------- C2 linear model invariance ---------- */
const c2={seed:239};
function c2calc(lam,t,w){const a=1-t,s=t,vz=a*a*lam+s*s,phi=(s*w-a*lam)/vz,kRec=((1-w)+phi)/(a*w+s),kCF=(s-a*lam)/vz,vel=lam/vz;return{a,s,vz,phi,kRec,kCF,vel,tgt:(a*w+s)**2*vel}}
function initC2(){c2.gl=bindList($('#c2-l'),POS_VALUES,4,$('#c2-lo'),renderC2);c2.gt=bindRange($('#c2-t'),$('#c2-to'),renderC2,v=>v.toFixed(2));c2.gw=bindList($('#c2-w'),W_VALUES,1,$('#c2-wo'),renderC2);
  $('#c2-re').addEventListener('click',()=>{c2.seed++;renderC2()});renderC2()}
function renderC2(){const lam=c2.gl(),t=c2.gt(),w=c2.gw(),o=c2calc(lam,t,w);const r=mulberry32(c2.seed),P=[];
  for(let i=0;i<200;i++){const g1=gauss(r),g2=gauss(r),x=Math.sqrt(lam)*g1,e=g2;P.push([o.a*x+o.s*e,w*e-x,e-x])}
  const xr=3*Math.sqrt(o.vz);
  const plot=(box,yr,col,k,shape,cls,title)=>{const svg=$('svg',box),S=widthOf(box,220,420);const nt=v=>+v.toPrecision(2)+'';
    const F=frame(svg,{W:S,H:S,m:{l:44,r:10,t:14,b:36},x:[-xr,xr],y:[-yr,yr],xt:[-xr*2/3,0,xr*2/3].map(v=>[v,nt(v)]),yt:[-yr*2/3,0,yr*2/3].map(v=>[v,nt(v)]),xl:'z'});
    P.forEach(p=>marker(F.d,shape,F.X(p[0]),F.Y(p[col]),'pt',2.2));
    el('path',{d:lp([[F.X(-xr),F.Y(-xr*k)],[F.X(xr),F.Y(xr*k)]]),class:cls},F.d);setAria(svg,title+`，迴歸係數 ${fmt(k)}`)};
  plot($('#c2-fa'),3*Math.sqrt(w*w+lam),1,o.phi,'c','acc','目標 u_W 對 z 的散佈圖');
  plot($('#c2-fb'),3*Math.sqrt(1+lam),2,o.kRec,'s','tot','速度 u 對 z 的散佈圖');
  rot($('#c2-ro'),[['Φ*（目標預測係數）',fmt(o.phi)],['還原係數（經 Eq. 5）',fmt(o.kRec),1],['還原係數（閉式）',fmt(o.kCF)],['|差|',Math.abs(o.kRec-o.kCF).toExponential(1)],['最小速度誤差',fmt(o.vel),1],['目標空間最小誤差',fmt(o.tgt)]]);
  $('#c2-tpl').textContent=`目標空間的最小誤差為 ${fmt(o.tgt)}，換回速度後為 ${fmt(o.vel)}；後者在任何 w 下都相同。`;
  announce('c2',`λ = ${lam}，t = ${t.toFixed(2)}，w = ${w}，還原係數 ${fmt(o.kRec)}`)}

/* ---------- C3 frozen-error single timestep ---------- */
function c3calc(t,A,C,rho){const a=1-t,s=t,G=rho*Math.sqrt(A*C),f=w=>(A*w*w-2*G*w+C)/(a*w+s)**2,P=A*s+a*G,Q=G*s+a*C;let kind,ws,fmin;
  if(P>0&&Q>0){kind='interior';ws=Q/P;fmin=f(ws)}else if(P>0){kind='w=0';ws=0;fmin=f(0)}else{kind='inf';ws=Infinity;fmin=A/a/a}
  return{a,s,G,f,P,Q,kind,ws,fmin,eq19:a/s*C/A,f0:C/s/s,finf:A/a/a,f1:f(1),fstar:1/(a*a/A+s*s/C)}}
const KIND={interior:'內部最小','w=0':'最小值在邊界 w = 0',inf:'下確界在邊界 w → ∞'};
function initC3(){const c3=initC3.s={};c3.gt=bindRange($('#c3-t'),$('#c3-to'),renderC3,v=>v.toFixed(2));c3.gA=bindList($('#c3-A'),POS_VALUES,1,$('#c3-Ao'),renderC3);c3.gC=bindList($('#c3-C'),POS_VALUES,0.1,$('#c3-Co'),renderC3);
  c3.gr=bindRange($('#c3-r'),$('#c3-ro2'),renderC3,v=>v.toFixed(1));
  $('#c3-adv').addEventListener('change',e=>{$('#c3-rw').classList.toggle('hide',!e.target.checked);renderC3()});renderC3()}
function renderC3(){const S=initC3.s,adv=$('#c3-adv').checked,t=S.gt(),A=S.gA(),C=S.gC(),rho=adv?S.gr():0,o=c3calc(t,A,C,rho);
  const box=$('#c3-fig'),svg=$('svg',box),W=widthOf(box,280,560),H=280;
  const ws=[];for(let i=0;i<=240;i++)ws.push(10**(-3+i*0.025));
  const ys=ws.map(o.f).concat([o.f0,o.finf]);let lo=Math.min(...ys),hi=Math.max(...ys);lo/=1.2;hi*=1.2;
  const lt=Math.floor(Math.log10(lo)),ht=Math.ceil(Math.log10(hi)),yt=[];for(let k=lt;k<=ht;k++)if(10**k>=lo&&10**k<=hi)yt.push([10**k,k>=0&&k<4?String(10**k):'1e'+k]);
  const F=frame(svg,{W,H,m:{l:62,r:14,t:30,b:40},x:[1e-3,1e3],xlog:true,y:[lo,hi],ylog:true,xt:[[1e-3,'10⁻³'],[1e-1,'0.1'],[1,'1'],[10,'10'],[1e3,'10³']],yt,xl:'雜訊權重 w（對數）',yl:'f(w)'});
  el('path',{d:lp(ws.map(w=>[F.X(w),F.Y(o.f(w))])),class:'tot'},F.d);
  const yi=F.Y(o.finf);el('line',{x1:F.x0,x2:F.x1,y1:yi,y2:yi,class:'ref'},F.d);tx(F.g,F.x1-4,yi-6,'w → ∞：A/α²',{class:'tk','text-anchor':'end'});
  const x0p=F.x0-14;marker(svg,'c',x0p,F.Y(o.f0),'hollow',4);tx(svg,x0p-8,F.Y(o.f0)+4,'w=0',{class:'tk','text-anchor':'end'});
  if(o.kind==='interior'){const inR=o.ws>=1e-3&&o.ws<=1e3,xx=F.X(Math.min(1e3,Math.max(1e-3,o.ws)));
    if(inR){el('line',{x1:xx,x2:xx,y1:F.y0,y2:F.Y(o.fmin),class:'acc dash','stroke-width':1.5},F.d);marker(F.d,'c',xx,F.Y(o.fmin),'acc-f',5)}
    else el('path',{d:o.ws>1e3?`M${F.x1} ${F.y0-12}l-9 -5v10z`:`M${F.x0} ${F.y0-12}l9 -5v10z`,class:'acc-f'},F.g);
    tx(F.g,Math.min(Math.max(xx,F.x0+40),F.x1-40),Math.max(F.y1+12,F.Y(o.fmin)-14),'w* = '+sig(o.ws),{class:'lab','text-anchor':'middle'})}
  else if(o.kind==='w=0'){el('circle',{cx:x0p,cy:F.Y(o.f0),r:8,class:'acc',fill:'none','stroke-width':2},svg);tx(F.g,F.x0+8,F.y1+12,'最小值在邊界 w = 0',{class:'lab'})}
  else{el('circle',{cx:F.x1,cy:yi,r:8,class:'acc',fill:'none','stroke-width':2},F.g);tx(F.g,F.x1-8,F.y1+12,'下確界在 w → ∞',{class:'lab','text-anchor':'end'})}
  if(adv&&rho!==0&&o.eq19>=1e-3&&o.eq19<=1e3)marker(F.d,'d',F.X(o.eq19),F.Y(o.f(o.eq19)),'hollow',5);
  const rows=[['情形',KIND[o.kind]],['w*（Eq. 18）',fmt(o.ws),1],['Eq. 19 近似 αC/(σA)',fmt(o.eq19)],['f(w*)',fmt(o.fmin)],['f(0) = C/σ²',fmt(o.f0)],['f(1)（速度）',fmt(o.f1)],['A/α²（w → ∞）',fmt(o.finf)]];
  if(rho===0)rows.push(['f* = 1/(α²/A + σ²/C)',fmt(o.fstar)]);if(!adv)rows.shift();
  rot($('#c3-ro'),rows);
  $('#c3-tpl').textContent=o.kind==='interior'?`內部最小：w* = ${fmt(o.ws)}，損失比速度目標（w = 1）低 ${(100*(1-o.fmin/o.f1)).toFixed(1)}%。`:o.kind==='w=0'?`ρ = ${rho.toFixed(1)} 時 Q = Γσ + αC ≤ 0，f 在 w ≥ 0 遞增，最小值在 w = 0。`:`ρ = ${rho.toFixed(1)} 時 P = Aσ + αΓ ≤ 0，f 一路遞減，下確界是 w → ∞ 的 ${fmt(o.finf)}。`;
  announce('c3',`t = ${t.toFixed(2)}，A = ${A}，C = ${C}，${KIND[o.kind]}，w* = ${fmt(o.ws)}`);setAria(svg,`f(w) 曲線，${KIND[o.kind]}，w* = ${fmt(o.ws)}`)}

/* ---------- C4 time cancellation vs no-information limit ---------- */
const M=1024,TG=Float64Array.from({length:M},(_,j)=>(j+.5)/M);
function dens(smp,mu){const p=new Float64Array(M);let S=0;for(let j=0;j<M;j++){const t=TG[j];p[j]=smp==='uniform'?1:Math.exp(-((Math.log(t/(1-t))-mu)**2)/2)/(Math.sqrt(2*Math.PI)*t*(1-t));S+=p[j]}for(let j=0;j<M;j++)p[j]/=S;return p}
const ft=(mode,par,t,w)=>{const a=1-t,s=t;return(mode==='sep'?a*w*w+s*par:w*w+par)/(a*w+s)**2};
const wstarT=(mode,par,t)=>mode==='sep'?par:(1-t)/t*par;
const NG=4001;
function c4avg(mode,par,p,i){const w=10**(-4+0.002*i);let s=0;for(let j=0;j<M;j++)s+=p[j]*ft(mode,par,TG[j],w);return s}
function c4final(L){let i=0;for(let k=1;k<NG;k++)if(L[k]<L[i])i=k;if(i===0)return{w:1e-4,edge:-1};if(i===NG-1)return{w:1e4,edge:1};
  let x=-4+0.002*i;const dd=L[i-1]-2*L[i]+L[i+1];if(dd>0)x+=0.002*0.5*(L[i-1]-L[i+1])/dd;return{w:10**x,edge:0}}
function c4Search(mode,par,smp,mu){const p=dens(smp,mu),L=new Float64Array(NG);for(let i=0;i<NG;i++)L[i]=c4avg(mode,par,p,i);return c4final(L)}
const c4={mode:'sep',par:{sep:0.1,zs:1},smp:'uniform',mu:0,cache:new Map(),last:null};
const TS=[0.1,0.3,0.5,0.7,0.9],DASH=['','8 4','3 3','1 4','8 3 1 3'],SHP=['c','s','t','d','x'];
function initC4(){const pi=$('#c4-p');pi.min=0;pi.max=POS_VALUES.length-1;pi.step=1;
  const syncP=()=>{pi.value=POS_VALUES.indexOf(c4.par[c4.mode]);const v=c4.par[c4.mode];$('#c4-po').textContent=v;pi.setAttribute('aria-valuetext',String(v));$('#c4-pn').textContent=c4.mode==='sep'?'C̄/Ā':'資料變異數 λ'};syncP();
  pi.addEventListener('input',()=>{c4.par[c4.mode]=POS_VALUES[+pi.value];syncP();renderC4()});
  seg($('#c4-mode'),v=>{c4.mode=v;syncP();renderC4()});
  seg($('#c4-smp'),v=>{c4.smp=v;$('#c4-mw').classList.toggle('hide',v!=='logitn');$('#c4-mnote').classList.toggle('hide',v!=='logitn');renderC4()});
  bindRange($('#c4-mu'),$('#c4-muo'),()=>{c4.mu=+$('#c4-mu').value;renderC4()},v=>v.toFixed(1));
  const lg=$('#c4-lg');lg.innerHTML=TS.map((t,k)=>`<span class="lg-s"><svg width="34" height="12" aria-hidden="true"><line x1="1" y1="6" x2="33" y2="6" stroke="currentColor" stroke-width="2" stroke-dasharray="${DASH[k]}"/></svg>${['●','■','▲','◆','✕'][k]} t = ${t}</span>`).join('')+'<span class="lg acc">時間平均最佳 ŵ</span>';
  renderC4()}
function renderC4(){const {mode,smp}=c4,par=c4.par[mode],mu=smp==='logitn'?c4.mu:0,key=`${mode}|${par}|${smp}|${mu}`;
  $('#c4-lead').textContent=mode==='sep'?'假設：Eq. 20 的可分離誤差，取 Ā = 1、q(t) ≡ 1、Γ = 0。這是論文的充分條件，不是從網路推得；玩具只算「若成立，最小點在哪」，不能學出 w ∝ λ。':'假設：網路對這個方向沒有資訊，目標預測恆為 0（Eq. 25，論文稱 zero-skill limit）。玩具只算這個極限下的最佳值，不能學出 w ∝ λ。';
  if(c4.cache.has(key)){$('#c4-busy').classList.add('hide');drawC4(c4.cache.get(key));return}
  if(c4.last)drawC4(c4.last,true);$('#c4-busy').classList.remove('hide');
  const p=dens(smp,mu),L=new Float64Array(NG);
  chunkRun('c4',NG,400,(i0,i1)=>{for(let i=i0;i<i1;i++)L[i]=c4avg(mode,par,p,i)},()=>{const r=c4final(L),res={mode,par,smp,mu,p,opt:r};c4.cache.set(key,res);$('#c4-busy').classList.add('hide');drawC4(res)})}
function drawC4(res,stale){const {mode,par,smp,opt}=res;if(!stale)c4.last=res;
  const cur=stale?{mode:c4.mode,par:c4.par[c4.mode]}:res;
  const box=$('#c4-fig'),svg=$('svg',box),W=widthOf(box,280,620),H=300;
  const F=frame(svg,{W,H,m:{l:46,r:14,t:30,b:40},x:[1e-3,1e3],xlog:true,y:[1,100],ylog:true,xt:[[1e-3,'10⁻³'],[1e-1,'0.1'],[1,'1'],[10,'10'],[1e3,'10³']],yt:[[1,'1'],[3,'3'],[10,'10'],[30,'30'],[100,'100']],xl:'雜訊權重 w（對數）',yl:'f_t(w) ÷ 最小值'});
  const ws=[];for(let i=0;i<=240;i++)ws.push(10**(-3+i*0.025));
  TS.forEach((t,k)=>{const wt=wstarT(cur.mode,cur.par,t),fm=ft(cur.mode,cur.par,t,wt);const pts=ws.map(w=>[F.X(w),F.Y(Math.min(1e3,ft(cur.mode,cur.par,t,w)/fm))]);
    el('path',{d:lp(pts),class:'ink','stroke-width':1.8,'stroke-dasharray':DASH[k]},F.d);
    for(let i=1;i<pts.length;i++){const r=ft(cur.mode,cur.par,t,ws[i])/fm,r0=ft(cur.mode,cur.par,t,ws[i-1])/fm;if((r>100)!==(r0>100)){el('path',{d:`M${pts[i][0]} ${F.y1+1}l4 7h-8z`,class:'ink-f'},F.g)}}
    if(wt>=1e-3&&wt<=1e3)marker(F.d,SHP[k],F.X(wt),F.Y(1)-2-k*0,'hollow',5)});
  if(!stale||res.mode===cur.mode){const w=opt.w,inR=w>=1e-3&&w<=1e3,xx=F.X(Math.min(1e3,Math.max(1e-3,w)));
    if(inR)el('line',{x1:xx,x2:xx,y1:F.y0,y2:F.y1,class:'acc','stroke-width':2.5,opacity:stale?.35:1},F.g);else el('path',{d:w>1e3?`M${F.x1} ${F.y0-14}l-9 -5v10z`:`M${F.x0} ${F.y0-14}l9 -5v10z`,class:'acc-f'},F.g)}
  const pb=$('#c4-pfig'),ps=$('svg',pb);
  if(smp==='logitn'&&!stale){pb.classList.remove('hide');const Wp=widthOf(pb,240,620),G=frame(ps,{W:Wp,H:96,m:{l:46,r:14,t:10,b:30},x:[0,1],y:[0,1.05],xt:[0,.5,1].map(v=>[v,String(v)]),yt:[],xl:'p(t)（正規化到最大值 1）'});
    let mx=0;res.p.forEach(v=>mx=Math.max(mx,v));el('path',{d:lp(Array.from(res.p,(v,j)=>[G.X(TG[j]),G.Y(v/mx)])),class:'perp'},G.d)}else if(!stale)pb.classList.add('hide');
  if(stale)return;
  const owTxt=opt.edge>0?'≥ 10⁴（超出搜尋範圍）':opt.edge<0?'≤ 10⁻⁴':sig(opt.w);
  const rows=TS.map(t=>[`t = ${t} 的最小點`,sig(wstarT(mode,par,t))]);rows.push(['時間平均最佳 ŵ',owTxt,1]);
  if(mode==='sep')rows.push(['C̄/Ā',String(par)]);else{const st=1,cs=par/(par+st*st);rows.push(['EDM（t = 0.5）：s = σ/α',fmt(st)],['EDM：c_skip = λ/(λ + s²)',fmt(cs)],['EDM：w = c_skip·s/(1 − c_skip)',fmt(cs*st/(1-cs))])}
  rot($('#c4-ro'),rows);
  $('#c4-tpl').textContent=mode==='sep'?`五個時刻的最小點都是 ${par}；時間平均最佳 ŵ = ${owTxt}。`:`逐時刻最小點由 ${sig(wstarT(mode,par,0.1))} 變到 ${sig(wstarT(mode,par,0.9))}；時間平均最佳 ŵ = ${owTxt}，換時間分佈後會改變。`;
  announce('c4',`${mode==='sep'?'可分離':'無資訊極限'}，參數 ${par}，時間平均最佳 ŵ = ${owTxt}`);setAria(svg,`五個時刻的正規化損失曲線；時間平均最佳 ŵ = ${owTxt}`)}

/* ---------- C5 first spectral update ---------- */
const L2S=[0.05,0.1,0.25,0.5,0.8,0.95],NS5=[8,32,128,512,2048];
const c5={seed:239,l2:0.25,n:32};
function c5draw(r,lam2,th,n,keep){const c=Math.cos(th),s=Math.sin(th),l1=Math.sqrt(1),l2=Math.sqrt(lam2);let g11=0,g12=0,g22=0;const pts=keep?[]:null;
  for(let i=0;i<n;i++){const ga=gauss(r),gb=gauss(r),e1=gauss(r),e2=gauss(r),p=l1*ga,q=l2*gb,x1=c*p-s*q,x2=s*p+c*q;
    g11+=x1*x1-2*x1*e1;g22+=x2*x2-2*x2*e2;g12+=x1*x2-(x1*e2+e1*x2);if(keep&&i<512)pts.push([x1,x2])}
  g11/=n;g12/=n;g22/=n;const tie=g12===0&&g11===g22,ph=0.5*Math.atan2(2*g12,g11-g22);let d=(ph-th)*180/Math.PI;d=((d+90)%180+180)%180-90;
  const m=(g11+g22)/2,h=Math.hypot((g11-g22)/2,g12);return{err:d,ph,tie,ev:[m-h,m+h],pts}}
function c5median(a){const s=[...a].sort((x,y)=>x-y),n=s.length;return n%2?s[(n-1)/2]:0.5*(s[n/2-1]+s[n/2])}
function c5Stats(lam2,thdeg,n,seed){const th=thdeg*Math.PI/180,r=mulberry32(seed),first=c5draw(r,lam2,th,n,true),errs=[Math.abs(first.err)];for(let k=1;k<200;k++)errs.push(Math.abs(c5draw(r,lam2,th,n,false).err));return{first,errs,med:c5median(errs)}}
function initC5(){const mk=(node,vals,key)=>{node.innerHTML=vals.map(v=>`<button type="button" data-v="${v}" aria-pressed="${v===c5[key]}">${v}</button>`).join('');seg(node,v=>{c5[key]=+v;renderC5()})};
  mk($('#c5-l2'),L2S,'l2');mk($('#c5-n'),NS5,'n');c5.gth=bindRange($('#c5-th'),$('#c5-tho'),renderC5,v=>v+'°');
  $('#c5-re').addEventListener('click',()=>{c5.seed++;renderC5()});renderC5()}
function renderC5(){const lam2=c5.l2,n=c5.n,thd=c5.gth(),th=thd*Math.PI/180,seed=c5.seed;
  const r=mulberry32(seed),first=c5draw(r,lam2,th,n,true),errs=[Math.abs(first.err)];
  drawC5a(first,th,n);drawC5b(null);
  rot($('#c5-ro'),[['本次角度誤差',first.tie?'重根：方向不確定':first.err.toFixed(2)+'°',1],['200 次中位數 |誤差|','計算中…'],['Ĝ 特徵值（rank 1、2，升冪）',first.ev.map(v=>v.toFixed(4)).join('、')]]);
  $('#c5-busy').classList.remove('hide');
  chunkRun('c5',199,10,(i0,i1)=>{for(let k=i0;k<i1;k++)errs.push(Math.abs(c5draw(r,lam2,th,n,false).err))},()=>{const med=c5median(errs);$('#c5-busy').classList.add('hide');drawC5b(errs,med);
    rot($('#c5-ro'),[['本次角度誤差',first.tie?'重根：方向不確定':first.err.toFixed(2)+'°',1],['200 次中位數 |誤差|',med.toFixed(2)+'°',1],['Ĝ 特徵值（rank 1、2，升冪）',first.ev.map(v=>v.toFixed(4)).join('、')]]);
    $('#c5-tpl').textContent=`n = ${n}、λ₂ = ${lam2} 時，第一次更新的主方向與 PCA 主軸的角度誤差中位數為 ${med.toFixed(2)}°；主軸對應升冪排列的 rank 2。`;
    announce('c5',`n = ${n}，λ2 = ${lam2}，中位數角度誤差 ${med.toFixed(2)} 度`)})}
function drawC5a(o,th,n){const box=$('#c5-fa'),svg=$('svg',box),S=widthOf(box,220,420);
  const F=frame(svg,{W:S,H:S,m:{l:30,r:10,t:10,b:30},x:[-3,3],y:[-3,3],xt:[-2,0,2],yt:[-2,0,2]});
  o.pts.forEach(p=>el('circle',{cx:F.X(p[0]),cy:F.Y(p[1]),r:2,class:'pt'},F.d));
  const ln=(a,cls,w)=>el('path',{d:lp([[F.X(-3.2*Math.cos(a)),F.Y(-3.2*Math.sin(a))],[F.X(3.2*Math.cos(a)),F.Y(3.2*Math.sin(a))]]),class:cls,'stroke-width':w},F.d);
  ln(th,'par dash',2.5);if(!o.tie){ln(o.ph,'acc',3);ln(o.ph+Math.PI/2,'acc',1)}
  if(n>512)tx(F.g,F.x1-4,F.y1+12,'只畫前 512 點',{class:'tk','text-anchor':'end'});
  setAria(svg,`資料點與方向；本次角度誤差 ${o.err.toFixed(2)} 度`)}
function drawC5b(errs,med){const box=$('#c5-fb'),svg=$('svg',box),S=widthOf(box,220,420);
  const bins=new Array(18).fill(0);if(errs)errs.forEach(e=>bins[Math.min(17,Math.floor(e/5))]++);const mx=Math.max(10,...bins);
  const F=frame(svg,{W:S,H:S,m:{l:36,r:10,t:10,b:36},x:[0,90],y:[0,mx*1.1],xt:[0,30,60,90].map(v=>[v,v+'°']),yt:[0,Math.round(mx/2),mx],xl:'|角度誤差|'});
  if(!errs){tx(F.g,(F.x0+F.x1)/2,(F.y0+F.y1)/2,'計算中…',{class:'al','text-anchor':'middle'});return}
  bins.forEach((b,i)=>{if(b)el('rect',{x:F.X(i*5)+1,y:F.Y(b),width:F.X(5)-F.X(0)-2,height:F.y0-F.Y(b),class:'par-f',opacity:.75},F.d)});
  const xm=F.X(Math.min(90,med));el('line',{x1:xm,x2:xm,y1:F.y0,y2:F.y1,class:'acc','stroke-width':2},F.d);tx(F.g,Math.min(xm+4,F.x1-60),F.y1+12,'中位數 '+med.toFixed(2)+'°',{class:'lab'});
  setAria(svg,`200 個 batch 的角度誤差直方圖，中位數 ${med.toFixed(2)} 度`)}

/* ---------- timeline ---------- */
const TL_GROUP={'2006.11239':['base','prediction-targets-overview'],'2011.13456':['base','prediction-targets-overview'],
 '2210.02747':['vel','flow-matching-velocity-target'],'2209.03003':['vel','flow-matching-velocity-target'],
 '2202.00512':['tgt','aniso-interpolated-targets-edm-kdiff-asymflow'],'2206.00364':['tgt','aniso-interpolated-targets-edm-kdiff-asymflow'],
 '2212.09748':['pix','aniso-pixel-patchified-setting'],'2511.13720':['pix','aniso-pixel-patchified-setting'],
 '2601.21419':['int','aniso-interpolated-targets-edm-kdiff-asymflow'],'2605.12964':['int','aniso-interpolated-targets-edm-kdiff-asymflow'],
 '2404.00456':['cond','aniso-quantization-conditioning'],'2603.02337':['cond','aniso-spectral-weight-analysis'],
 '2611.17767':['me','aniso-target-normal-form-and-recovery']};
const TL_NAME={base:'擴散基礎',vel:'速度回歸 · w = 1',tgt:'目標選擇',pix:'像素骨幹 · w = 0',int:'內插前驅',cond:'條件數',me:'逐方向 w · 本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  if(!data.some(e=>e.arxiv==='2611.17767'))data.push({year:2026,title:'Learning Anisotropic Prediction Targets for Flow Models',authors_short:'本論文',arxiv:'2611.17767',role:'每個方向學習獨立的非負雜訊權重與基底。'});
  data=data.map((e,i)=>({e,i})).sort((a,b)=>(+a.e.year||0)-(+b.e.year||0)||a.i-b.i).map(o=>o.e);
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];used.add(g[0]);const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    const extra=e.arxiv==='2601.21419'?'<a href="prediction-targets-k-diff.html">教學</a>':'';
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${g[1]?`<a href="${g[1]}.html">筆記</a>`:''}${extra}</div>`;track.appendChild(a)});
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
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;$$('.toc a,.tocbar a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+id));const ab=$(`.tocbar a[href="#${id}"]`);if(ab&&bar.offsetParent)bar.scrollTo({left:ab.offsetLeft-16,behavior:RM?'auto':'smooth'})}})},{rootMargin:'-40% 0px -55% 0px'});
  secs.forEach(s=>io.observe(s));
  addEventListener('scroll',()=>$('#appbar').classList.toggle('raised',scrollY>8),{passive:true})}
function initTheme(){let saved=null;try{saved=localStorage.getItem('pm-theme')}catch(e){}if(saved)document.documentElement.dataset.theme=saved;
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){}})}
function initScrollRegions(){$$('.eq-m').forEach((e,i)=>{const n=e.nextElementSibling;e.setAttribute('tabindex','0');e.setAttribute('role','region');e.setAttribute('aria-label',(n?n.textContent:'公式')+'（可左右捲動）')})}

/* ---------- self test (synchronous cores only) ---------- */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t)=>Math.abs(a-b)<=t+1e-12,rel=(a,b,t)=>Math.abs(a/b-1)<=t;
  const r=mulberry32(239);ok(near(r(),0.105665,1e-6)&&near(r(),0.373234,1e-6),'PRNG');
  let o=c1calc(0.01,0,0.1,239,false);ok(near(o.err,10,1e-4),'C1 err unclamped '+o.err);
  o=c1calc(0.01,0,0.1,239,true);ok(near(o.uh,2.3525,1e-4)&&near(o.bias,0.4424,1e-4)&&near(o.noise,2.5,1e-4)&&near(o.err,2.9424,1e-4),'C1 clamped '+o.uh);
  o=c1calc(0.3,1,0.1,239,false);ok(near(o.pe,0.7,1e-9)&&near(o.err,0.1,1e-4),'C1 t=.3 w=1');
  o=c1calc(0.05,10,0.1,239,false);ok(near(o.uh,-0.5794,1e-4)&&near(o.amp,0.1047,1e-4),'C1 t=.05 w=10');
  ok(W_VALUES.every(w=>Math.abs(c2calc(4,0.5,w).kRec+1.2)<1e-12),'C2 invariance');
  o=c2calc(4,0.5,10);ok(near(o.phi,2.4,1e-9)&&near(o.tgt,96.8,1e-9)&&near(o.vel,3.2,1e-9),'C2 w=10');
  let q=c3calc(0.5,1,0.1,0);ok(q.kind==='interior'&&near(q.ws,0.1,1e-4)&&near(q.fmin,0.3636,1e-4)&&near(q.fstar,q.fmin,1e-12)&&near(q.f1,1.1,1e-4),'C3 row1');
  q=c3calc(0.5,1,0.1,0.5);ok(q.kind==='interior'&&near(q.ws,0.2229,1e-4)&&near(q.fmin,0.2118,1e-4)&&near(q.f1,0.7838,1e-4),'C3 row2');
  q=c3calc(0.5,1,10,-0.9);ok(q.kind==='inf'&&near(q.fmin,4,1e-4)&&near(q.f1,16.6921,1e-4),'C3 row6');
  q=c3calc(0.8,1,0.1,0);ok(fmt(q.f0)==='0.1563','C3 f0 fmt');q=c3calc(0.5,0.01,10,0.9);ok(fmt(q.eq19)==='1000.0000'&&near(q.ws,34.9098,1e-4),'C3 row7');
  const s4=[['zs',1,'uniform',0,1],['zs',1,'logitn',1,0.5668],['zs',1,'logitn',-1.5,2.457],['zs',100,'logitn',-1.5,108.6],['zs',10,'uniform',0,3.162],['sep',1.5,'logitn',-1,1.5],['sep',0.1,'uniform',0,0.1]];
  s4.forEach(([m,p,s,mu,v])=>{const res=c4Search(m,p,s,mu);ok(res.edge===0&&rel(res.w,v,0.005),`C4 ${m} ${p} ${s} ${mu}: ${res.w}`)});
  [[0.25,32,12.49,11.16],[0.25,128,0.25,5.68],[0.8,2048,-5.27,6.65]].forEach(([l2,n,e,md])=>{const st=c5Stats(l2,30,n,239);ok(near(st.first.err,e,0.01)&&near(st.med,md,0.01),`C5 l2=${l2} n=${n}: ${st.first.err.toFixed(3)} ${st.med.toFixed(3)}`)});
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  ok(!!document.querySelector('.katex')||typeof katex==='undefined','KaTeX rendered');
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ---------- boot ---------- */
initTheme();initToc();initScrollRegions();$$('.waxis').forEach(waxis);renderFam();initC1();initC2();initC3();initC4();initC5();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderFam();renderC1();renderC2();renderC3();if(c4.last)drawC4(c4.last);renderC5()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
function boot2(){renderMath();if(TEST)selfTest()}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',boot2);else boot2();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
})();
