
/* ---------- toy (design §4.0; same algorithm as verify/toy.py) ---------- */
const TOY={MU_B:-0.3,SD_B:0.3,N:40,SD_R:0.05,RIDGE:1e-3,A_STAR:0.2,W_Q:0.3,B0:4,G:2001};
const GRID=Float64Array.from({length:TOY.G},(_,j)=>-1+2*j/(TOY.G-1));
function rng(seed){let a=seed>>>0;return()=>{a=(a+0x6D2B79F5)>>>0;let t=Math.imul(a^(a>>>15),a|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296}}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}
const qTrue=a=>TOY.B0+Math.exp(-((a-TOY.A_STAR)**2)/(2*TOY.W_Q*TOY.W_Q));
function toyData(seed){const r=rng(seed),A=[],R=[];for(let i=0;i<TOY.N;i++){const a=Math.min(1,Math.max(-1,TOY.MU_B+TOY.SD_B*gauss(r)));A.push(a);R.push(qTrue(a)+TOY.SD_R*gauss(r))}return{A,R}}
function solve3(M,b){const n=3,X=M.map((row,i)=>row.concat([b[i]]));
  for(let c=0;c<n;c++){let p=c;for(let i=c+1;i<n;i++)if(Math.abs(X[i][c])>Math.abs(X[p][c]))p=i;[X[c],X[p]]=[X[p],X[c]];
    for(let i=c+1;i<n;i++){const f=X[i][c]/X[c][c];for(let j=c;j<=n;j++)X[i][j]-=f*X[c][j]}}
  const x=[0,0,0];for(let i=n-1;i>=0;i--){let s=X[i][n];for(let j=i+1;j<n;j++)s-=X[i][j]*x[j];x[i]=s/X[i][i]}return x}
function fitQ(A,R){const N=A.length,abar=A.reduce((s,a)=>s+a,0)/N,v=A.reduce((s,a)=>s+(a-abar)**2,0)/N,sa=Math.sqrt(v);
  if(sa<1e-6)return null;
  const phi=a=>{const u=(a-abar)/sa;return[1,u,u*u]},P=A.map(phi);
  const H=[0,1,2].map(i=>[0,1,2].map(j=>P.reduce((s,p)=>s+p[i]*p[j],0)/N+(i===j&&i>0?TOY.RIDGE:0)));
  const b=[0,1,2].map(i=>P.reduce((s,p,k)=>s+p[i]*R[k],0)/N),c=solve3(H,b);
  const Qh=a=>{const f=phi(a);return c[0]*f[0]+c[1]*f[1]+c[2]*f[2]};
  return{A,c,abar,v,sa,Qh,QG:Float64Array.from(GRID,Qh),meanAbs:A.reduce((s,a)=>s+Math.abs(Qh(a)),0)/N}}
function argmaxArr(f){let j0=0,b=-Infinity;for(let j=0;j<TOY.G;j++){const v=f(j);if(v>b){b=v;j0=j}}return j0}
// alpha = Infinity -> argmax Q-hat; opts {scale, off, norm, dropV}
function policyAt(fit,alpha,o){o=o||{};const sc=o.scale??1,off=o.off??0,norm=o.norm??true;
  const m=Math.max(Math.abs(sc)*meanAbsShift(fit,off),1e-12);
  if(alpha===Infinity){const j=argmaxArr(j=>sc*(fit.QG[j]+off));return{pi:GRID[j],lam:null,m}}
  const lam=norm?alpha/m:alpha,ab=fit.abar,v=o.dropV?0:fit.v;
  const j=argmaxArr(j=>lam*sc*(fit.QG[j]+off)-(GRID[j]-ab)**2-v);return{pi:GRID[j],lam,m}}
function meanAbsShift(fit,off){if(off===0)return fit.meanAbs;const A=fit.A;return A.reduce((s,a)=>s+Math.abs(fit.Qh(a)+off),0)/A.length}
function piInterior(fit,lam){const den=2*fit.sa*fit.sa-2*lam*fit.c[2];return den>0?fit.abar+lam*fit.c[1]*fit.sa/den:null}
function alphaAtBound(fit){const c1=fit.c[1],c2=fit.c[2],sa=fit.sa,d=(c1>=0?1:-1)-fit.abar,den=c1*sa+2*c2*d;if(den===0)return null;
  const lam=2*sa*sa*d/den;if(!(lam>0)||2*sa*sa-2*lam*c2<=0)return null;return lam*fit.meanAbs}

const toyState={seed:7,data:null,fit:null,sweep:null};
function toyReset(seed){toyState.seed=seed;toyState.data=toyData(seed);toyState.fit=fitQ(toyState.data.A,toyState.data.R);
  toyState.sweep=toyState.fit?Float64Array.from({length:101},(_,k)=>qTrue(policyAt(toyState.fit,k/10).pi)):null}

/* ---------- C1 (two instances) ---------- */
const c1s={alpha:2.5};
const RS=(n,s)=>{const c=n.querySelector(`[data-r="${s}"]`);return c};
function setR(box,k,v){const c=RS(box,k);if(c)c.textContent=v}
function drawC1(box,full){const F=toyState.fit,D=toyState.data,svg=$('.c1-main svg',box);
  $$('.c1-seed',box).forEach(n=>n.textContent=toyState.seed);
  if(!F){svg.textContent='';svg.setAttribute('viewBox','0 0 300 40');tx(svg,4,24,'資料動作相同，無法擬合',{class:'lab'});return}
  const W=chartW($('.c1-main',box)),H=Math.round(Math.min(320,W*0.66));
  let lo=Infinity,hi=-Infinity;for(let j=0;j<TOY.G;j+=4){const a=GRID[j];lo=Math.min(lo,F.QG[j],qTrue(a));hi=Math.max(hi,F.QG[j],qTrue(a))}
  const pad=(hi-lo)*0.05;lo-=pad;hi+=pad;
  const yt=[lo+pad,(lo+hi)/2,hi-pad].map(v=>[v,v.toFixed(1)]);
  const G=frame(svg,{W,H,m:{l:44,r:14,t:30,b:52},x:[-1,1],y:[lo,hi],xt:[-1,-0.5,0,0.5,1].map(v=>[v,String(v)]),yt,xl:'動作 a',yl:'期望報酬'});
  const amin=Math.min(...D.A),amax=Math.max(...D.A);
  el('rect',{x:G.X(amin),y:G.y1,width:G.X(amax)-G.X(amin),height:G.y0-G.y1,class:'band'},G.g);
  tx(G.g,(G.X(amin)+G.X(amax))/2,G.y1+14,'資料範圍',{class:'tk','text-anchor':'middle'});
  D.A.forEach(a=>el('line',{x1:G.X(a),x2:G.X(a),y1:G.y0+22,y2:G.y0+30,class:'rug'},G.g));
  const qp=[],hp=[];for(let j=0;j<TOY.G;j+=8){qp.push([G.X(GRID[j]),G.Y(qTrue(GRID[j]))]);hp.push([G.X(GRID[j]),G.Y(F.QG[j])])}
  qp.push([G.X(1),G.Y(qTrue(1))]);hp.push([G.X(1),G.Y(F.QG[TOY.G-1])]);
  el('path',{d:lp(qp),class:'ref'},G.g);el('path',{d:lp(hp),class:'par'},G.g);
  tx(G.g,G.X(1)-4,G.Y(F.QG[TOY.G-1])+16,'擬合的 Q',{class:'lab','text-anchor':'end'});
  tx(G.g,G.X(1)-4,G.Y(qTrue(1))-8,'真實值',{class:'lab','text-anchor':'end'});
  const pBC=policyAt(F,0).pi,pQ=policyAt(F,Infinity).pi,rows=[[pBC,'sq','bc'],[pQ,'tri','q']];let p2=null,lam=null;
  if(full){const r=policyAt(F,c1s.alpha===Infinity?Infinity:c1s.alpha);p2=r.pi;lam=r.lam;
    if(c1s.alpha!==Infinity){const ab=F.abar,Jv=j=>lam*F.QG[j]-(GRID[j]-ab)**2-F.v,jm=GRID.indexOf(p2),Jm=Jv(jm),shift=F.QG[jm]-Jm,pts=[];
      for(let j=0;j<TOY.G;j+=8){const y=Jv(j)+shift;if(y>=lo)pts.push([G.X(GRID[j]),G.Y(y)])}
      if(pts.length>1)el('path',{d:lp(pts),class:'acc-thin'},G.g)}
    rows.push([p2,'circ','bc2'])}
  rows.forEach(([p,shape,cls])=>{const x=G.X(p);el('line',{x1:x,x2:x,y1:G.Y(qTrue(p)),y2:G.y0,class:'cur'},G.g);marker(G,x,G.Y(qTrue(p)),shape,cls,false)});
  const fmt=v=>f3(v);
  [[pBC,0],[pQ,1]].concat(full?[[p2,2]]:[]).forEach(([p,i])=>{const h=F.Qh(p),q=qTrue(p);setR(box,'a'+i,fmt(p));setR(box,'h'+i,fmt(h));setR(box,'q'+i,fmt(q));setR(box,'e'+i,fmt(h-q))});
  if(full)setR(box,'lam',lam==null?'∞':(+lam.toPrecision(4)).toString());
  svg.setAttribute('aria-label',`玩具：資料平均動作 ${f3(pBC)}，只最大化擬合值選 ${f3(pQ)}`+(full?`，TD3+BC 選 ${f3(p2)}，真實值 ${f3(qTrue(p2))}`:''));
  if(full)drawSweep(box,F);
  announce(full?'c1b':'c1a',full?`α = ${c1s.alpha===Infinity?'無限大':c1s.alpha.toFixed(1)}，TD3+BC 動作 ${f3(p2)}，真實值 ${f3(qTrue(p2))}`:`只最大化擬合值選 ${f3(pQ)}，真實值 ${f3(qTrue(pQ))}`)}
function drawSweep(box,F){const sv=$('.c1-sweep svg',box),S=toyState.sweep,W=chartW($('.c1-sweep',box)),H=Math.round(Math.max(170,Math.min(210,W*0.4)));
  let lo=Math.min(...S),hi=Math.max(...S);const pad=(hi-lo)*0.1||0.05;lo-=pad;hi+=pad;
  const G=frame(sv,{W,H,m:{l:44,r:14,t:28,b:40},x:[0,10],y:[lo,hi],xt:[0,2.5,5,7.5,10].map(v=>[v,String(v)]),yt:[[lo+pad,(lo+pad).toFixed(2)],[hi-pad,(hi-pad).toFixed(2)]],xl:'α',yl:'TD3+BC 動作的真實值'});
  const qBC=qTrue(policyAt(F,0).pi),qQ=qTrue(policyAt(F,Infinity).pi);
  [[qBC,'純 BC',qBC>=qQ?-5:13],[qQ,'只最大化擬合值',qQ>qBC?-5:13]].forEach(([v,l,dy])=>{el('line',{x1:G.x0,x2:G.x1,y1:G.Y(v),y2:G.Y(v),class:'ref'},G.g);tx(G.g,G.x1-2,G.Y(v)+dy,l,{class:'tk','text-anchor':'end'})});
  el('path',{d:lp(Array.from(S,(v,k)=>[G.X(k/10),G.Y(v)])),class:'acc-thin'},G.g);
  const ab=alphaAtBound(F);if(ab!=null&&ab<=10){const x=G.X(ab);el('line',{x1:x,x2:x,y1:G.y0,y2:G.y0-10,class:'ink'},G.g);tx(G.g,x+4,G.y1+12,'到達動作上界 α ≈ '+ab.toFixed(2),{class:'tk'});el('line',{x1:x,x2:x,y1:G.y1+16,y2:G.y0,class:'cur'},G.g)}
  if(c1s.alpha!==Infinity)vline(G,G.X(c1s.alpha));
  sv.setAttribute('aria-label','TD3+BC 動作的真實值隨 α 變化：先升後降，之後停在動作上界')}
let setChip=null;
function renderC1(){drawC1($('#c1a'),false);drawC1($('#c1b'),true)}
function initC1(){toyReset(7);const R=sched(renderC1);
  $$('[data-act="seed"]').forEach(b=>b.addEventListener('click',()=>{toyReset(toyState.seed+1);renderC1();renderC2()}));
  const sl=$('#c1-al'),out=$('#c1-alo');
  const chipVal=()=>c1s.alpha===Infinity?'inf':(c1s.alpha===0?'0':(c1s.alpha===2.5?'2.5':''));
  setChip=radiogroup($('#c1-chip'),'2.5',v=>{c1s.alpha=v==='inf'?Infinity:+v;if(v!=='inf'){sl.value=v}out.textContent=v==='inf'?'∞':(+v).toFixed(1);sl.setAttribute('aria-valuetext',v==='inf'?'α 無限大':'α = '+(+v).toFixed(1));R()});
  sl.addEventListener('input',()=>{c1s.alpha=+sl.value;out.textContent=c1s.alpha.toFixed(1);sl.setAttribute('aria-valuetext','α = '+c1s.alpha.toFixed(1));setChip(chipVal());if(!chipVal())$('#c1-chip button').tabIndex=0;R()});
  renderC1()}

/* ---------- C2 ---------- */
const c2s={logc:0,b:0,mode:'norm'};
function c2Pol(fit,lc,b,norm){return policyAt(fit,2.5,{scale:10**lc,off:b,norm,dropV:true})}
function renderC2(){const F=toyState.fit,box=$('#c2-fig'),svg=$('svg',box);if(!F){svg.textContent='';return}
  const W=chartW(box),H=Math.round(Math.min(300,W*0.62));
  const G=frame(svg,{W,H,m:{l:44,r:14,t:30,b:42},x:[-2,2],y:[-1,1],xt:[-2,-1,0,1,2].map(v=>[v,'10'+['⁻²','⁻¹','⁰','¹','²'][v+2]]),yt:[[-1,'−1'],[0,'0'],[1,'1']],xl:'尺度 c（對數）',yl:'TD3+BC 動作 π*'});
  const A=toyState.data.A,amin=Math.min(...A),amax=Math.max(...A);
  el('rect',{x:G.x0,y:G.Y(amax),width:G.x1-G.x0,height:G.Y(amin)-G.Y(amax),class:'band'},G.g);
  tx(G.g,G.x0+4,G.Y(amax)+14,'資料範圍',{class:'tk'});
  [['norm',true,'acc-thin'],['fixed',false,'ref']].forEach(([k,n,cls])=>{const pts=[];for(let i=0;i<=400;i++){const lc=-2+i*0.01;pts.push([G.X(lc),G.Y(c2Pol(F,lc,c2s.b,n).pi)])}el('path',{d:lp(pts),class:cls+(k===c2s.mode?' strong':'')},G.g)});
  vline(G,G.X(c2s.logc));
  const n=c2s.mode==='norm',r=c2Pol(F,c2s.logc,c2s.b,n),c=10**c2s.logc,g=r.lam*c*F.c[1]/F.sa;
  marker(G,G.X(c2s.logc),G.Y(r.pi),'circ','bc2',false);
  const box2=$('#c2');setR(box2,'m',fe(r.m));setR(box2,'lam',fe(r.lam));setR(box2,'pi',f3(r.pi));setR(box2,'q',(+qTrue(r.pi).toPrecision(5)).toString());setR(box2,'g',fe(g));
  const mb=meanAbsShift(F,c2s.b);$('#c2-warn').classList.toggle('hide',!(n&&mb<0.5));
  svg.setAttribute('aria-label',`尺度 c = ${fe(c)}，平移 b = ${c2s.b.toFixed(1)}，${n?'λ 正規化':'固定 λ'}，TD3+BC 動作 ${f3(r.pi)}`);
  announce('c2',`${n?'λ 正規化':'固定 λ'}：c = ${fe(c)}，b = ${c2s.b.toFixed(1)}，動作 ${f3(r.pi)}，λ = ${fe(r.lam)}`)}
function initC2(){const R=sched(renderC2);
  const c=$('#c2-c'),b=$('#c2-b');
  c.addEventListener('input',()=>{c2s.logc=+c.value;const v=fe(10**c2s.logc);$('#c2-co').textContent=v;c.setAttribute('aria-valuetext','c = '+v);R()});
  b.addEventListener('input',()=>{c2s.b=+b.value;$('#c2-bo').textContent=c2s.b.toFixed(1);b.setAttribute('aria-valuetext','b = '+c2s.b.toFixed(1));R()});
  radiogroup($('#c2-mode'),c2s.mode,v=>{c2s.mode=v;R()});renderC2()}

/* ---------- C4 (data: design §6.3) ---------- */
const C4D={"T2":{"algs":["BC","BRAC-p","AWAC","CQL","Fisher-BRC","TD3+BC"],
 "rows":[
  ["random","HalfCheetah",[2.0,0.1],[23.5],[2.2],[21.7,0.9],[32.2,2.2],[10.2,1.3]],
  ["random","Hopper",[9.5,0.1],[11.1],[9.6],[10.7,0.1],[11.4,0.2],[11.0,0.1]],
  ["random","Walker2d",[1.2,0.2],[0.8],[5.1],[2.7,1.2],[0.6,0.6],[1.4,1.6]],
  ["medium","HalfCheetah",[36.6,0.6],[44.0],[37.4],[37.2,0.3],[41.3,0.5],[42.8,0.3]],
  ["medium","Hopper",[30.0,0.5],[31.2],[72.0],[44.2,10.8],[99.4,0.4],[99.5,1.0]],
  ["medium","Walker2d",[11.4,6.3],[72.7],[30.1],[57.5,8.3],[79.5,1.0],[79.7,1.8]],
  ["medium-replay","HalfCheetah",[34.7,1.8],[45.6],null,[41.9,1.1],[43.3,0.9],[43.3,0.5]],
  ["medium-replay","Hopper",[19.7,5.9],[0.7],null,[28.6,0.9],[35.6,2.5],[31.4,3.0]],
  ["medium-replay","Walker2d",[8.3,1.5],[-0.3],null,[15.8,2.6],[42.6,7.0],[25.2,5.1]],
  ["medium-expert","HalfCheetah",[67.6,13.2],[43.8],[36.8],[27.1,3.9],[96.1,9.5],[97.9,4.4]],
  ["medium-expert","Hopper",[89.6,27.6],[1.1],[80.9],[111.4,1.2],[90.6,43.3],[112.2,0.2]],
  ["medium-expert","Walker2d",[12.0,5.8],[-0.3],[42.7],[68.1,13.1],[103.6,4.6],[101.1,9.3]],
  ["expert","HalfCheetah",[105.2,1.7],[3.8],[78.5],[82.4,7.4],[106.8,3.0],[105.7,1.9]],
  ["expert","Hopper",[111.5,1.3],[6.6],[85.2],[111.2,2.1],[112.3,0.2],[112.2,0.2]],
  ["expert","Walker2d",[56.0,24.9],[-0.2],[57.0],[103.8,7.6],[79.9,32.4],[105.7,2.7]]],
 "total":[[595.3,91.5],[284.1],null,[764.3,61.5],[974.6,108.3],[979.3,33.4]]},
 "T3":[["CQL（GitHub）",25,89,251],["Fisher-BRC（GitHub，TensorFlow）",39,33,132],["Fisher-BRC（PyTorch 重寫）",15,58,128],["TD3+BC",0,0,39]],
 "F8":[["DT",981],["DT（100k 步）",98],["TD3+BC",39]],
 "T9":{"rows":[["medium","HalfCheetah",[42.6,0.1],[48.3,0.3]],["medium","Hopper",[67.6,1],[59.3,4.2]],["medium","Walker2d",[74.0,1.4],[83.7,2.1]],
  ["medium-replay","HalfCheetah",[36.6,0.8],[44.6,0.5]],["medium-replay","Hopper",[82.7,7],[60.9,18.8]],["medium-replay","Walker2d",[66.6,3],[81.8,5.5]],
  ["medium-expert","HalfCheetah",[86.8,1.3],[90.7,4.3]],["medium-expert","Hopper",[107.6,1.8],[98.0,9.4]],["medium-expert","Walker2d",[108.1,0.2],[110.1,0.5]]],
  "totalDT":[[672.6,16.6],[677.4,45.6]],"totalAll":[1013.2,57.4]}};
const ALG_CLS=['bcA','brac','awac','cql','fbrc','td3bc'];
const DSETS=['random','medium','medium-replay','medium-expert','expert'];
const c4s={tab:'t2',ds:'all',on:[true,false,false,true,true,true]};
const hm=m=>m<1?'< 1 分':(m>=60?`${Math.floor(m/60)}h ${m%60}m`:`${m}m`);
function hbar(G,x0v,v,y,h,cls){const x0=G.X(Math.max(0,x0v)),x1=G.X(Math.max(0,v));el('rect',{x:Math.min(x0,x1),y,width:Math.max(1,Math.abs(x1-x0)),height:h,class:'b-'+cls},G.g)}
function renderC4(){const fig=$('#c4-fig'),svg=$('svg',fig),W=widthOf(fig,280,900),narrow=W<600,note=$('#c4-note');
  $('#c4-ds').classList.toggle('hide',c4s.tab!=='t2');$('#c4-alg').classList.toggle('hide',c4s.tab!=='t2');
  if(c4s.tab==='t2'){const T=C4D.T2,algs=[0,1,2,3,4,5].filter(i=>c4s.on[i]),rows=T.rows.filter(r=>c4s.ds==='all'||r[0]===c4s.ds);
    const bh=11,gap=12,gh=algs.length*bh+gap,head=c4s.ds==='all'?18:0;let H=34+40;let groups=[];let y=30;let lastDs=null;
    rows.forEach(r=>{if(c4s.ds==='all'&&r[0]!==lastDs){y+=head;lastDs=r[0];groups.push({ds:r[0],y:y-6})}groups.push({r,y});y+=gh});H=y+36;
    const lab=narrow?96:130;
    const G=frame(svg,{W,H,m:{l:lab,r:30,t:30,b:36},x:[0,140],y:[0,1],xt:[0,20,40,60,80,100,120,140].map(v=>[v,String(v)]),yt:[],xl:'正規化分數',yl:''});
    groups.forEach(o=>{if(!o.r){tx(G.g,4,o.y,o.ds,{class:'al'});return}const r=o.r;tx(G.g,lab-8,o.y+algs.length*bh/2+4,r[1],{class:'lab','text-anchor':'end'});
      algs.forEach((i,k)=>{const d=r[2+i],yy=o.y+k*bh;if(!d){tx(G.g,G.X(0)+4,yy+bh-2,T.algs[i]+' 無資料',{class:'tk'});return}
        hbar(G,0,d[0],yy+1,bh-2,ALG_CLS[i]);if(d[1]!=null){const a=G.X(Math.max(0,d[0]-d[1])),b=G.X(d[0]+d[1]);el('line',{x1:a,x2:b,y1:yy+bh/2,y2:yy+bh/2,class:'whisk'},G.g)}
        if(!narrow||algs.length<=4)tx(G.g,G.X(Math.max(0,d[0]+(d[1]||0)))+4,yy+bh-2,String(d[0]),{class:'tk'})})});
    $('#c4-lg2')&&0;
    note.innerHTML=`總分（論文 Total）：${T.algs.map((a,i)=>T.total[i]?`${a} ${T.total[i][0]}${T.total[i][1]!=null?' ± '+T.total[i][1]:''}`:`${a} -`).join('、')}。Total 的 ± 等於各列 ± 之和（本頁核算），不是總分的標準差。<span class="tag">本頁推導</span>Fisher-BRC 各列加總為 975.2，論文 Total 為 974.6。圖例：${algs.map(i=>`<svg class="sw" viewBox="0 0 14 10" aria-hidden="true"><rect class="b-${ALG_CLS[i]}" x=".5" y=".5" width="13" height="9"/></svg>${T.algs[i]}`).join(' ')}`;
    svg.setAttribute('aria-label',`Table 2 正規化分數，資料集：${c4s.ds==='all'?'全部':c4s.ds}；`+rows.map(r=>r[0]+' '+r[1]+'：'+algs.map(i=>T.algs[i]+' '+(r[2+i]?r[2+i][0]:'無資料')).join('、')).join('；'))}
  else if(c4s.tab==='time'){const rows=C4D.T3.map(r=>({n:r[0],imp:r[1],alg:r[2],tot:r[3],base:r[3]-r[1]-r[2]})).concat(C4D.F8.slice(0,2).map(r=>({n:r[0],tot:r[1],only:true})));
    const H=40+rows.length*34+40,lab=narrow?120:220;
    const G=frame(svg,{W,H,m:{l:lab,r:60,t:30,b:40},x:[0,1000],y:[0,1],xt:[0,250,500,750,1000].map(v=>[v,String(v)]),yt:[],xl:'分鐘',yl:''});
    rows.forEach((r,i)=>{const y=34+i*34;tx(G.g,lab-8,y+15,narrow?r.n.replace('（GitHub，TensorFlow）','（TF）').replace('（PyTorch 重寫）','（PyTorch）').replace('（GitHub）',''):r.n,{class:'lab','text-anchor':'end'});
      if(r.only){hbar(G,0,r.tot,y,20,'only')}else{hbar(G,0,r.base,y,20,'base');if(r.imp)hbar(G,r.base,r.base+r.imp,y,20,'imp');if(r.alg)hbar(G,r.base+r.imp,r.tot,y,20,'alg')}
      tx(G.g,G.X(Math.min(r.tot,1000))+4,y+15,hm(r.tot)+(r.tot>1000?' →':''),{class:'tk'})});
    note.innerHTML='堆疊：<svg class="sw" viewBox="0 0 14 10" aria-hidden="true"><rect class="b-base" x=".5" y=".5" width="13" height="9"/></svg>底層演算法（本頁推導：總計減兩項調整） <svg class="sw" viewBox="0 0 14 10" aria-hidden="true"><rect class="b-imp" x=".5" y=".5" width="13" height="9"/></svg>實作調整 <svg class="sw" viewBox="0 0 14 10" aria-hidden="true"><rect class="b-alg" x=".5" y=".5" width="13" height="9"/></svg>演算法調整 <svg class="sw" viewBox="0 0 14 10" aria-hidden="true"><rect class="b-only" x=".5" y=".5" width="13" height="9"/></svg>只有總計（Fig. 8）。TD3+BC 的兩項調整分別 &lt; 1s、&lt; 5s。DT 16h 21m 超出軸範圍。';
    svg.setAttribute('aria-label','訓練時間：'+rows.map(r=>r.n+' '+hm(r.tot)).join('、'))}
  else{const T=C4D.T9,bh=12,gh=2*bh+12,H=34+T.rows.length*gh+40,lab=narrow?110:190;
    const G=frame(svg,{W,H,m:{l:lab,r:30,t:30,b:40},x:[0,120],y:[0,1],xt:[0,40,80,120].map(v=>[v,String(v)]),yt:[],xl:'正規化分數（D4RL v2）',yl:''});
    T.rows.forEach((r,i)=>{const y=30+i*gh;tx(G.g,lab-8,y+bh+4,(narrow?r[0].replace('medium-','m-'):r[0])+' '+r[1],{class:'lab','text-anchor':'end'});
      [[r[2],'dt'],[r[3],'td3bc']].forEach(([d,c],k)=>{hbar(G,0,d[0],y+k*bh+1,bh-2,c);tx(G.g,G.X(d[0])+4,y+k*bh+bh-2,String(d[0]),{class:'tk'})})});
    note.innerHTML=`<svg class="sw" viewBox="0 0 14 10" aria-hidden="true"><rect class="b-dt" x=".5" y=".5" width="13" height="9"/></svg>DT <svg class="sw" viewBox="0 0 14 10" aria-hidden="true"><rect class="b-td3bc" x=".5" y=".5" width="13" height="9"/></svg>TD3+BC。Total(DT)：DT ${T.totalDT[0][0]}、TD3+BC ${T.totalDT[1][0]}；TD3+BC 全部 15 個任務 ${T.totalAll[0]}。DT 取自其論文、3 個 seed；TD3+BC 5 個 seed。Total 的 ± 同樣是各列 ± 之和。`;
    svg.setAttribute('aria-label','Table 9：'+T.rows.map(r=>r[0]+' '+r[1]+' DT '+r[2][0]+'、TD3+BC '+r[3][0]).join('；'))}}
function c4Table(){const T=C4D.T2,c=d=>d?d[0]+(d[1]!=null?' ± '+d[1]:''):'-';
  $('#c4-table').innerHTML=`<div class="tscroll"><table><caption>Table 2：D4RL v0 正規化分數（最後 10 次評估、5 個 seed）</caption><thead><tr><th>資料集</th><th>環境</th>${T.algs.map(a=>`<th class="num">${a}</th>`).join('')}</tr></thead><tbody>${T.rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td>${r.slice(2).map(d=>`<td class="num">${c(d)}</td>`).join('')}</tr>`).join('')}<tr class="hl"><td colspan="2">Total</td>${T.total.map(d=>`<td class="num">${c(d)}</td>`).join('')}</tr></tbody></table></div>
  <div class="tscroll"><table><caption>Table 3：訓練時間</caption><thead><tr><th>方法</th><th class="num">實作調整</th><th class="num">演算法調整</th><th class="num">總計</th></tr></thead><tbody>${C4D.T3.map(r=>`<tr><td>${r[0]}</td><td class="num">${r[0]==='TD3+BC'?'&lt; 1s':hm(r[1])}</td><td class="num">${r[0]==='TD3+BC'?'&lt; 5s':hm(r[2])}</td><td class="num">${hm(r[3])}</td></tr>`).join('')}</tbody></table></div>
  <div class="tscroll"><table><caption>Table 9：D4RL v2（DT 有結果的 9 個任務）</caption><thead><tr><th>資料集</th><th>環境</th><th class="num">DT</th><th class="num">TD3+BC</th></tr></thead><tbody>${C4D.T9.rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td class="num">${c(r[2])}</td><td class="num">${c(r[3])}</td></tr>`).join('')}<tr class="hl"><td colspan="2">Total(DT)</td><td class="num">${c(C4D.T9.totalDT[0])}</td><td class="num">${c(C4D.T9.totalDT[1])}</td></tr></tbody></table></div>`}
function initC4(){const narrow=innerWidth<600;if(narrow)c4s.ds='random';
  const ds=$('#c4-ds');(narrow?DSETS:['all'].concat(DSETS)).forEach(d=>{const b=document.createElement('button');b.className='chip';b.dataset.v=d;b.textContent=d==='all'?'全部':d;ds.appendChild(b)});
  radiogroup(ds,c4s.ds,v=>{c4s.ds=v;renderC4()});
  const al=$('#c4-alg');C4D.T2.algs.forEach((a,i)=>{const l=document.createElement('label');l.className='chk';l.innerHTML=`<input type="checkbox"${c4s.on[i]?' checked':''}><svg class="sw" viewBox="0 0 14 10" aria-hidden="true"><rect class="b-${ALG_CLS[i]}" x=".5" y=".5" width="13" height="9"/></svg>${a}`;
    $('input',l).addEventListener('change',e=>{c4s.on[i]=e.target.checked;if(!c4s.on.some(Boolean)){c4s.on[i]=true;e.target.checked=true}renderC4()});al.appendChild(l)});
  radiogroup($('#c4-tab'),c4s.tab,v=>{c4s.tab=v;renderC4()});c4Table();renderC4()}

/* ---------- self test (design §5 / §6.3) ---------- */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const chk=(x,ref,m,tol)=>ok(Math.abs(x-ref)<=(tol??1e-3*Math.abs(ref)),`${m}: ${x} vs ${ref}`);
  const d=toyData(7),F=fitQ(d.A,d.R);
  chk(F.abar,-0.2678,'abar',6e-5);chk(F.v,0.08509,'v',6e-6);chk(F.sa,0.2917,'sa',6e-5);chk(F.meanAbs,4.39708,'mean|Qh|',6e-6);
  [[0,4.32324],[1,0.33739],[2,0.07384]].forEach(([i,v])=>chk(F.c[i],v,'c'+i,6e-6));
  chk(Math.min(...d.A),-0.829,'amin',6e-4);chk(Math.max(...d.A),0.254,'amax',6e-4);
  [[0,-0.268,4.2962,4.3231],[1,-0.104,4.5984,4.5360],[2,0.167,4.9940,4.9903],[2.5,0.381,4.8336,5.4391],[3,0.699,4.2507,6.2528],[3.2,0.874],[3.3,0.977],[3.4,1,4.0286,7.1846],[10,1,4.0286,7.1846],[Infinity,1,4.0286,7.1846]].forEach(([al,p,q,h])=>{
    const r=policyAt(F,al);chk(r.pi,p,'pi alpha='+al,1.1e-3);if(q!=null){chk(qTrue(r.pi),q,'q alpha='+al,6e-5);chk(F.Qh(r.pi),h,'Qh alpha='+al,6e-5)}});
  chk(policyAt(F,2.5).lam,0.5686,'lam 2.5',6e-5);
  chk(piInterior(F,policyAt(F,3.3).lam),0.9768,'pi_int 3.3',6e-5);chk(piInterior(F,policyAt(F,3).lam),0.6994,'pi_int 3',6e-5);
  ok(piInterior(F,policyAt(F,5.1).lam)===null,'J convex at 5.1');chk(alphaAtBound(F),3.3212,'alpha at bound',6e-4);
  let mx=0,prev=policyAt(F,0).pi;for(let k=1;k<=33;k++){const p=policyAt(F,k/10).pi;mx=Math.max(mx,Math.abs(p-prev));prev=p}ok(mx<=0.13,'C1 continuity max step '+mx);
  const d1=toyData(1),F1=fitQ(d1.A,d1.R);chk(policyAt(F1,2.5).pi,0.264,'seed1 pi',1.1e-3);
  const c2=(lc,b,n)=>c2Pol(F,lc,b,n);
  chk(c2(-2,0,true).pi,0.381,'C2 norm c=0.01',1.1e-3);chk(c2(2,0,true).pi,0.381,'C2 norm c=100',1.1e-3);chk(c2(-2,0,false).pi,-0.253,'C2 fixed c=0.01',1.1e-3);chk(c2(-1,0,false).pi,-0.083,'C2 fixed c=0.1',1.1e-3);
  chk(c2(0,0,false).pi,1,'C2 fixed c=1',1.1e-3);chk(c2(0,4,true).pi,-0.036,'C2 norm b=4',1.1e-3);chk(c2(0,8,true).pi,-0.126,'C2 norm b=8',1.1e-3);chk(c2(0,-2,true).pi,1,'C2 norm b=-2',1.1e-3);
  chk(c2(0,4,true).lam,0.2977,'C2 lam b=4',6e-5);chk(c2(0,-4,true).lam,6.162,'C2 lam b=-4',6e-4);
  const g=r=>r.lam*F.c[1]/F.sa;chk(g(c2(0,0,true)),0.6576,'C2 g norm',6e-5);chk(c2(0,0,false).lam*F.c[1]/F.sa,2.892,'C2 g fixed',6e-4);
  let mmin=Infinity;for(let k=0;k<=160;k++)mmin=Math.min(mmin,meanAbsShift(F,-8+0.1*k));chk(mmin,0.2810,'C2 min m(b)',6e-5);
  const T=C4D.T2,sum=i=>Math.round(T.rows.reduce((s,r)=>s+r[2+i][0],0)*10)/10,psum=i=>Math.round(T.rows.reduce((s,r)=>s+(r[2+i][1]||0),0)*10)/10;
  ok(sum(5)===979.3&&sum(0)===595.3&&sum(3)===764.3&&sum(1)===284.1,'T2 sums');ok(sum(4)===975.2,'Fisher row sum 975.2');
  ok(psum(5)===33.4&&psum(4)===108.3&&psum(3)===61.5&&psum(0)===91.5,'T2 pm sums');
  ok(T.rows.every(r=>r[7][0]>=r[2][0]),'TD3+BC >= BC on all rows');
  ok(Math.round(C4D.T9.rows.reduce((s,r)=>s+r[3][0],0)*10)/10===677.4,'T9 subset sum');ok(Math.round((677.4+11.0+8.5+1.6+96.7+107.8+110.2)*10)/10===1013.2,'T9 all sum');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  ok(!!document.querySelector('.katex')||typeof katex==='undefined','KaTeX rendered (or not loaded)');
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}
