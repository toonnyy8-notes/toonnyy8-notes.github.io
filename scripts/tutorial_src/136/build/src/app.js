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

/* ================= number formats (design.md 4, one definition) ================= */
const SUP={'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'};
function fmtSci(x){if(x===0)return'0';const e=Math.floor(Math.log10(Math.abs(x))+1e-12);let m=x/10**e;if(Math.abs(m)>=9.995){m/=10;return fmtSci(m*10**(e+1))}return m.toFixed(2)+'×10'+String(e).split('').map(c=>SUP[c]).join('')}
function fmtNum(x,d){const s=x.toFixed(d);return Number(s)===0?(0).toFixed(d):s.replace('-','−')}
const fmtProb=x=>x<1e-3?fmtSci(x):x.toPrecision(3);
const fmtGrad=x=>Math.abs(x)<5e-5?'≈ 0':fmtNum(x,2);

/* ================= 2x2 design grid ================= */
function renderGrid22(){$$('.grid22').forEach(g=>{const hl=g.dataset.hl,acc=k=>FIG1.ACC[k];
  const cell=(r,k)=>{const a=hl==='fig1'?`<span>1e-5：<b>${acc(r+'|'+k+'|1')}</b></span><span>5e-5：<b>${acc(r+'|'+k+'|5')}</b></span>`:'<span>1e-5</span><span>5e-5</span>';
    return `<div class="g-cell${(r==='On')===(k==='Rev')?' conv':''}"><div class="g-n">${r}PD·${k==='F'?'F':'Rev'}</div><div class="g-lr">${a}</div></div>`};
  g.innerHTML=`<div class="g-h"></div><div class="g-h">forward（F）</div><div class="g-h">reverse（Rev）</div><div class="g-r">OnPD</div>${cell('On','F')}${cell('On','Rev')}<div class="g-r">OffPD</div>${cell('Off','F')}${cell('Off','Rev')}`})}

/* ================= C1: paired differences on Fig. 1 (check.py C1) ================= */
const FIG1={ACC:{'On|F|1':72,'On|F|5':72,'On|Rev|1':72,'On|Rev|5':35,'Off|F|1':71,'Off|F|5':73,'Off|Rev|1':35,'Off|Rev|5':49},
            SPA:{'On|F|1':85.3,'On|F|5':51.9,'On|Rev|1':88.5,'On|Rev|5':58.5,'Off|F|1':88.8,'Off|F|5':59.5,'Off|Rev|1':89.6,'Off|Rev|5':60.0}};
const C1F=['rollout','KL','LR'],C1LEV=[['On','Off'],['F','Rev'],['1','5']];
const C1LN={rollout:{On:'OnPD',Off:'OffPD'},KL:{F:'forward',Rev:'reverse'},LR:{'1':'LR 1e-5','5':'LR 5e-5'}};
const C1SH=['圓','方','三角','菱形'],C1SY=['●','■','▲','◆'],C1TT={rollout:'改 rollout（On→Off）',KL:'改 KL 方向（F→Rev）',LR:'改學習率（1e-5→5e-5）'};
function pairDiffs(metric,factor){const f=C1F.indexOf(factor),o=[0,1,2].filter(i=>i!==f),M=FIG1[metric],out=[];
  for(const a of C1LEV[o[0]])for(const b of C1LEV[o[1]]){const k1=[],k2=[];k1[o[0]]=k2[o[0]]=a;k1[o[1]]=k2[o[1]]=b;k1[f]=C1LEV[f][0];k2[f]=C1LEV[f][1];
    const v1=M[k1.join('|')],v2=M[k2.join('|')];
    const fx=[a,b].map((v,j)=>{const nm=C1F[o[j]];return nm==='LR'?(v==='1'?'1e-5':'5e-5'):v}).join('·');
    out.push({fixed:fx,n1:C1LN[factor][C1LEV[f][0]],n2:C1LN[factor][C1LEV[f][1]],v1,v2,d:Math.round(Math.abs(v1-v2)*10)/10})}
  return out}
const c1s={metric:'SPA',sel:null};
function c1RowText(f,r,i){return `${C1TT[f]}，固定 ${r.fixed}（${C1SH[i]}）：${r.n1} ${r.v1}，${r.n2} ${r.v2}，差 ${r.d} 個百分點`}
function shape(svg,i,cx,cy,cls){const s=6;if(i===0)return el('circle',{cx,cy,r:s,class:cls},svg);if(i===1)return el('rect',{x:cx-s,y:cy-s,width:2*s,height:2*s,class:cls},svg);
  if(i===2)return el('path',{d:`M${cx} ${cy-s-1}L${cx+s+1} ${cy+s}L${cx-s-1} ${cy+s}Z`,class:cls},svg);return el('path',{d:`M${cx} ${cy-s-2}L${cx+s+2} ${cy}L${cx} ${cy+s+2}L${cx-s-2} ${cy}Z`,class:cls},svg)}
function renderC1(){const box=$('#c1-fig');box.innerHTML='';const tb=$('#c1-table tbody');tb.innerHTML='';
  C1F.forEach(f=>{const rows=pairDiffs(c1s.metric,f);
    const wrap=document.createElement('div');wrap.className='c1row';
    wrap.innerHTML=`<div class="c1h"><b>${C1TT[f]}</b><span class="c1lg">${rows.map((r,i)=>`${C1SY[i]} ${r.fixed}`).join('　')}</span></div><svg role="group" aria-label="${C1TT[f]}的四組配對差，左右鍵切換"></svg>`;box.appendChild(wrap);
    const svg=$('svg',wrap),W=Math.max(260,Math.floor(box.clientWidth||600)),H=66,L=14,R=W-18,x=lin(0,40,L,R),yc=30;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
    el('line',{x1:L,x2:R,y1:yc,y2:yc,class:'grid'},svg);
    (W<480?[0,20,40]:[0,10,20,30,40]).forEach(t=>{el('line',{x1:x(t),x2:x(t),y1:yc+12,y2:yc+16,class:'axis'},svg);tx(svg,x(t),H-4,t+(t===40?' pp':''),{class:'lab','text-anchor':'middle'})});
    const ord=rows.map((r,i)=>i).sort((a,b)=>rows[a].d-rows[b].d),dy={};let prev=-1e9,k=0;ord.forEach(i=>{const px=x(rows[i].d);k=(px-prev<14)?k+1:0;dy[i]=[0,-11,11,-22][k%4];prev=px});
    rows.forEach((r,i)=>{const g=el('g',{class:'c1pt',tabindex:i===0?'0':'-1',role:'button','aria-label':c1RowText(f,r,i),'data-f':f,'data-i':i},svg);
      const on=c1s.sel&&c1s.sel.f===f&&c1s.sel.i===i;shape(g,i,x(r.d),yc+dy[i],on?'mk-sel':'mk');
      g.addEventListener('click',()=>c1Select(f,i,true))});
    svg.addEventListener('keydown',e=>{const pts=$$('.c1pt',svg),cur=pts.indexOf(document.activeElement);if(cur<0)return;let j=cur;
      if(e.key==='ArrowRight'||e.key==='ArrowDown')j=(cur+1)%4;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')j=(cur+3)%4;else if(e.key==='Enter'||e.key===' '){c1Select(f,cur,true);e.preventDefault();return}else return;
      e.preventDefault();c1Select(f,j,true)});
    tb.insertAdjacentHTML('beforeend',`<tr class="grp"><td colspan="4"><b>${C1TT[f]}</b></td></tr>`);
    rows.forEach((r,i)=>{const on=c1s.sel&&c1s.sel.f===f&&c1s.sel.i===i;
      tb.insertAdjacentHTML('beforeend',`<tr${on?' class="hl" aria-current="true"':''}><td data-h="固定組合">固定 ${r.fixed}（${C1SY[i]} ${C1SH[i]}）<span class="sr">：</span></td><td data-h="方法 1">${r.n1}：${fmtNum(r.v1,c1s.metric==='ACC'?0:1)}<span class="sr">；</span></td><td data-h="方法 2">${r.n2}：${fmtNum(r.v2,c1s.metric==='ACC'?0:1)}<span class="sr">；</span></td><td data-h="差" class="num">差 ${r.d.toFixed(c1s.metric==='ACC'?0:1)} pp</td></tr>`)})})}
function c1Select(f,i,focus){c1s.sel={f,i};renderC1();const g=$(`#c1-fig .c1pt[data-f="${f}"][data-i="${i}"]`);
  $$(`#c1-fig .c1pt[data-f="${f}"]`).forEach(p=>p.setAttribute('tabindex',p===g?'0':'-1'));if(focus&&g)g.focus();
  announce('c1',c1RowText(f,pairDiffs(c1s.metric,f)[i],i))}
function initC1(){segInit($('#c1-metric'),['ACC','SPA'],c1s.metric,m=>m==='ACC'?'ID 準確率':'更新稀疏度',m=>{c1s.metric=m;const s=c1s.sel;renderC1();if(s){const g=$(`#c1-fig .c1pt[data-f="${s.f}"][data-i="${s.i}"]`);$$(`#c1-fig .c1pt[data-f="${s.f}"]`).forEach(p=>p.setAttribute('tabindex',p===g?'0':'-1'))}announce('c1','指標：'+(m==='ACC'?'ID 準確率':'更新稀疏度'))});renderC1()}

/* ================= C2: token KL and logit gradients (check.py C2) ================= */
const SHAPE=[.4,.3,.2,.1],TOK5=['A','B','C','D','E'];
const dist5=a=>[a,...SHAPE.map(s=>(1-a)*s)];
function klParts(aS,aT){const S=dist5(aS),T=dist5(aT),r=S.map((s,i)=>Math.log(s)-Math.log(T[i]));
  const DF=T.reduce((a,t,i)=>a+t*(Math.log(t)-Math.log(S[i])),0),DR=S.reduce((a,s,i)=>a+s*r[i],0);
  const gF=S.map((s,i)=>s-T[i]),gR=S.map((s,i)=>s*(r[i]-DR)),n2=g=>Math.sqrt(g.reduce((a,x)=>a+x*x,0));
  return{S,T,r,DF,DR,gF,gR,nF:n2(gF),nR:n2(gR),dr:Math.max(...r)-Math.min(...r),maxF:Math.max(...gF.map(Math.abs)),sumF:gF.reduce((a,b)=>a+b,0),sumR:gR.reduce((a,b)=>a+b,0)}}
const c2s={aS:.5,aT:.01};
const C2P=[['預設',.5,.01],['兩者相同',.3,.3],['教師給 A 的機率趨近 0',.5,1e-6],['學生給 A 的機率趨近 0',1e-6,.6]];
function gradBars(boxId,g,ymax,hatch){const {svg,W,H}=frame($('#'+boxId),240,460,200),L=40,R=W-8,T=14,B=H-26,x=lin(0,5,L,R),y=lin(-ymax,ymax,B,T),bw=(R-L)/5*.56;
  el('line',{x1:L,x2:R,y1:y(0),y2:y(0),class:'axis'},svg);[-ymax,0,ymax].forEach(t=>tx(svg,L-6,y(t)+4,fmtNum(t,ymax>=10?0:1),{class:'lab','text-anchor':'end'}));
  if(!hatch){[-1,1].forEach(t=>el('line',{x1:L,x2:R,y1:y(t),y2:y(t),class:'ref-s'},svg));tx(svg,R,y(1)-4,'界 ±1',{class:'lab','text-anchor':'end'})}
  g.forEach((v,i)=>{const cx=x(i+.5),y0=y(0),y1=y(v);el('rect',{x:cx-bw/2,y:Math.min(y0,y1),width:bw,height:Math.max(1,Math.abs(y1-y0)),class:hatch?'bar-h':'bar-s'},svg);
    tx(svg,cx,v>=0?y1-5:y1+14,fmtGrad(v),{class:'lab','text-anchor':'middle'});tx(svg,cx,H-6,TOK5[i],{class:'lab13','text-anchor':'middle'})});return svg}
function renderC2(){const o=klParts(c2s.aS,c2s.aT);$('#c2-so').textContent=fmtProb(c2s.aS);$('#c2-to').textContent=fmtProb(c2s.aT);
  $('#c2-s').setAttribute('aria-valuetext','π_S(A) = '+fmtProb(c2s.aS));$('#c2-t').setAttribute('aria-valuetext','π_T(A) = '+fmtProb(c2s.aT));
  const sf=gradBars('c2-f',o.gF,1,false),ym=Math.max(1,Math.max(...o.gR.map(Math.abs)))*1.1,sr=gradBars('c2-r',o.gR,ym,true);
  dl($('#c2-read'),[['forward KL',fmtNum(o.DF,4)+' nats'],['reverse KL',fmtNum(o.DR,4)+' nats'],['‖g_F‖₂（界 √2）',fmtNum(o.nF,4)],['‖g_R‖₂（界 Δr，不緊）',fmtNum(o.nR,4)+' ≤ '+fmtNum(o.dr,4)],['A 的梯度 forward',fmtGrad(o.gF[0])],['A 的梯度 reverse',fmtGrad(o.gR[0])]]);
  const s=`π_S(A) = ${fmtProb(c2s.aS)}，π_T(A) = ${fmtProb(c2s.aT)}；forward KL = ${fmtNum(o.DF,4)}，reverse KL = ${fmtNum(o.DR,4)}；A 的 logit 梯度 forward ${fmtGrad(o.gF[0])}、reverse ${fmtGrad(o.gR[0])}`;
  announce('c2',s);setLabel(sf,'c2f','forward 梯度：'+o.gF.map((v,i)=>TOK5[i]+' '+fmtGrad(v)).join('，'));setLabel(sr,'c2r','reverse 梯度：'+o.gR.map((v,i)=>TOK5[i]+' '+fmtGrad(v)).join('，'))}
function initC2(){const box=$('#c2-presets');C2P.forEach(([n,a,b])=>{const bt=document.createElement('button');bt.type='button';bt.className='chip';bt.textContent=n;
    bt.addEventListener('click',()=>{c2s.aS=a;c2s.aT=b;$('#c2-s').value=Math.log10(a).toFixed(2);$('#c2-t').value=Math.log10(b).toFixed(2);renderC2()});box.appendChild(bt)});
  $('#c2-s').addEventListener('input',e=>{c2s.aS=10**Number(e.target.value);renderC2()});$('#c2-t').addEventListener('input',e=>{c2s.aT=10**Number(e.target.value);renderC2()});
  $('#c2-s').value=Math.log10(c2s.aS).toFixed(2);$('#c2-t').value=Math.log10(c2s.aT).toFixed(2);renderC2()}

/* ================= C3: rollout-stability gap (check.py C3) ================= */
function boundParts(delta,eps){const dr=Math.log1p(-delta)-Math.log(delta);
  return{dr,rev:eps*Math.SQRT2/8*dr,fwd:eps*Math.SQRT2/2*(.5-delta),fbound:2*Math.SQRT2*eps,rbound:2*dr*eps}}
const c3s={d:.01,e:.1},C3X=Math.log10(1/(1+Math.exp(16)));
function renderC3(){const {d,e}=c3s,o=boundParts(d,e);$('#c3-do').textContent=fmtSci(d);$('#c3-eo').textContent=e.toFixed(2);
  $('#c3-d').setAttribute('aria-valuetext','δ = '+fmtSci(d));$('#c3-e').setAttribute('aria-valuetext','ε = '+e.toFixed(2));
  const {svg,W,H}=frame($('#c3-fig'),300,860,280),L=52,R=W-14,T=14,B=H-34,x=lin(-12,-.31,L,R),y=lin(-5,2,B,T),cl=v=>Math.max(-5,Math.min(2,v)),narrow=W<520;
  for(let t=-5;t<=2;t++){el('line',{x1:L,x2:R,y1:y(t),y2:y(t),class:'grid'},svg);if(!narrow||t%2===0)tx(svg,L-6,y(t)+4,'10'+String(t).split('').map(c=>SUP[c]).join(''),{class:'lab','text-anchor':'end'})}
  for(let t=-12;t<=0;t+=narrow?4:2){if(t>-.31)continue;el('line',{x1:x(t),x2:x(t),y1:B,y2:B+4,class:'axis'},svg);tx(svg,x(t),B+18,'10'+String(t).split('').map(c=>SUP[c]).join(''),{class:'lab','text-anchor':'middle'})}
  tx(svg,R,H-2,'δ（對數）',{class:'lab','text-anchor':'end'});tx(svg,L,T-2,'更新差（對數）',{class:'lab'});
  const pts=f=>{const a=[];for(let i=0;i<=200;i++){const lx=-12+(11.69)*i/200;a.push([x(lx),y(cl(Math.log10(f(10**lx))))])}return a};
  el('path',{d:lp([[L,y(Math.log10(o.fbound))],[R,y(Math.log10(o.fbound))]]),class:'dot3-s'},svg);
  el('path',{d:lp(pts(dd=>boundParts(dd,e).fwd)),class:'npar-s'},svg);
  el('path',{d:lp(pts(dd=>boundParts(dd,e).rev)),class:'acc-s'},svg);
  el('circle',{cx:x(C3X),cy:y(Math.log10(o.fbound)),r:6,class:'hollow'},svg);tx(svg,x(C3X)+9,y(Math.log10(o.fbound))-8,'交點 δ ≈ 1.1×10⁻⁷',{class:'lab13 halo'});
  const ld=Math.log10(d);el('line',{x1:x(ld),x2:x(ld),y1:T,y2:B,class:'cur'},svg);
  el('circle',{cx:x(ld),cy:y(cl(Math.log10(o.rev))),r:4.5,class:'acc-f'},svg);el('rect',{x:x(ld)-4,y:y(cl(Math.log10(o.fwd)))-4,width:8,height:8,class:'par-f'},svg);
  dl($('#c3-read'),[['Δr = log((1−δ)/δ)',fmtNum(o.dr,4)],['reverse 更新差（Eq. 57）',fmtNum(o.rev,4)],['reverse 條件界 2Δr·ε（Eq. 59）',fmtNum(o.rbound,4)],['forward 更新差（同一反例）',fmtNum(o.fwd,4)],['forward 的界 2√2·ε（Eq. 27）',fmtNum(o.fbound,4)]]);
  const s=`δ = ${fmtSci(d)}，ε = ${e.toFixed(2)}；reverse 更新差 ${fmtNum(o.rev,4)}，forward 更新差 ${fmtNum(o.fwd,4)}（界 ${fmtNum(o.fbound,4)}）`;announce('c3',s);setLabel(svg,'c3',s)}
function initC3(){$('#c3-d').addEventListener('input',e=>{c3s.d=10**Number(e.target.value);renderC3()});$('#c3-e').addEventListener('input',e=>{c3s.e=Number(e.target.value);renderC3()});renderC3()}

/* ================= C4: rollout-policy spectrum (check.py C4) ================= */
const PS=[.45,.25,.15,.08,.05,.02],PT=[.03,.57,.25,.10,.0499,.0001],ALPHA=.2,CC=2*Math.log(3),TOK6=['A','B','C','D','E','F'];
function softmax(z){const m=Math.max(...z),e=z.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s)}
function spectrum(lam,safe){lam=Math.round(lam*100)/100;const lS=PS.map(Math.log),lT=PT.map(Math.log),r=lS.map((a,i)=>a-lT[i]);
  const zh=lS.map((a,i)=>(1+lam)/2*a+(1-lam)/2*lT[i]);let z;
  if(!safe||Math.abs(lam)<=1)z=zh;
  else if(lam>1){const mx=Math.max(...lS);z=lS.map((a,i)=>a+(lam-1)/2*((a>=mx+Math.log(ALPHA)?1:0)*Math.min(Math.max(r[i],0),CC)))}
  else{const mx=Math.max(...lT);z=lT.map((a,i)=>a+(-lam-1)/2*((a>=mx+Math.log(ALPHA)?1:0)*Math.min(Math.max(-r[i],0),CC)))}
  const p=softmax(z),anc=lam>1?lS:lam<-1?lT:null;let amp=null;
  if(anc){const d=z.map((v,i)=>v-anc[i]);amp=Math.exp(Math.max(...d)-Math.min(...d))}return{p,amp,lam}}
const c4s={lam:0,safe:true};
const fb=v=>v<.01?'<0.01':v.toFixed(2);
function renderC4(){const o=spectrum(c4s.lam,c4s.safe),lam=o.lam;$('#c4-lo').textContent=lam.toFixed(2);$('#c4-l').setAttribute('aria-valuetext','λ = '+lam.toFixed(2));
  const box=$('#c4-fig'),narrow=(box.clientWidth||600)<520,ser=[[PS,'bS-r'],[PT,'bT-r'],[o.p,'bL-r']];let svg;
  if(!narrow){const f=frame(box,300,860,250);svg=f.svg;const {W,H}=f,L=40,R=W-8,T=12,B=H-28,x=lin(0,6,L,R),y=lin(0,.8,B,T),gw=(R-L)/6,bw=gw*.24;
    [0,.2,.4,.6,.8].forEach(t=>{el('line',{x1:L,x2:R,y1:y(t),y2:y(t),class:'grid'},svg);tx(svg,L-6,y(t)+4,t.toFixed(1),{class:'lab','text-anchor':'end'})});
    TOK6.forEach((tk,i)=>{ser.forEach(([a,c],j)=>{const cx=L+gw*i+gw/2+(j-1)*(bw+3),v=a[i];el('rect',{x:cx-bw/2,y:y(v),width:bw,height:Math.max(1,B-y(v)),class:c},svg);tx(svg,cx,y(v)-4-(v<.05?(j%2)*13:0),fb(v),{class:'lab9','text-anchor':'middle'})});
      tx(svg,L+gw*i+gw/2,H-8,tk,{class:'lab13','text-anchor':'middle'})})}
  else{const f=frame(box,280,520,6*66+20);svg=f.svg;const {W}=f,L=22,R=W-44,x=lin(0,.8,L,R),bh=14;
    TOK6.forEach((tk,i)=>{const y0=10+i*66;tx(svg,4,y0+28,tk,{class:'lab13'});ser.forEach(([a,c],j)=>{const yy=y0+j*(bh+4),v=a[i];el('rect',{x:L,y:yy,width:Math.max(1,x(v)-L),height:bh,class:c},svg);tx(svg,x(v)+4,yy+11,fb(v),{class:'lab'})})})}
  dl($('#c4-read'),[['λ',lam.toFixed(2)],['防護',c4s.safe?'開':'關'],['最大勝算放大',o.amp===null?'—（內插區）':fmtNum(o.amp,2)+' 倍'],['π_λ(F)',fmtNum(o.p[5],4)]]);
  const s=`λ = ${lam.toFixed(2)}，防護 ${c4s.safe?'開':'關'}；π_λ = ${o.p.map((v,i)=>TOK6[i]+' '+v.toFixed(3)).join('，')}；最大勝算放大 ${o.amp===null?'—':fmtNum(o.amp,2)}`;announce('c4',s);setLabel(svg,'c4',s)}
function initC4(){const box=$('#c4-btns');[[-1,'λ = −1（OffPD）'],[0,'λ = 0'],[1,'λ = 1（OnPD）'],[2,'λ = 2']].forEach(([v,n])=>{const b=document.createElement('button');b.type='button';b.className='chip';b.textContent=n;
    b.addEventListener('click',()=>{c4s.lam=v;$('#c4-l').value=v;renderC4()});box.appendChild(b)});
  $('#c4-l').addEventListener('input',e=>{c4s.lam=Number(e.target.value);renderC4()});$('#c4-safe').addEventListener('change',e=>{c4s.safe=e.target.checked;renderC4()});renderC4()}

/* ================= timeline groups ================= */
const TL_GROUP={'1503.02531':['kd','knowledge-distillation-foundations'],'1606.07947':['kd','knowledge-distillation-foundations'],
  '2306.13649':['opd','odd-onpd-vs-offpd-setup'],'2410.11325':['opd','odd-onpd-vs-offpd-setup'],
  '2509.04259':['forget','odd-on-policy-forgetting-sparsity-claims'],'2510.18874':['forget','odd-on-policy-forgetting-sparsity-claims'],'2505.11711':['forget','odd-on-policy-forgetting-sparsity-claims'],'2601.19897':['forget','odd-on-policy-forgetting-sparsity-claims'],
  '2501.17161':['gen','odd-sft-vs-rl-generalization-claims'],
  '1412.6980':['opt','odd-learning-rate-adam-clipping-rlvr'],'2604.13627':['opt','odd-learning-rate-adam-clipping-rlvr'],'2506.09477':['opt','odd-token-kl-direction-gradients'],
  '2604.13016':['recent','odd-recent-opd-studies-2026'],'2609.35259':['me','odd-lineage-hub']};
const TL_NAME={kd:'蒸餾基礎',opd:'on-policy 蒸餾',forget:'被檢驗的說法：遺忘與稀疏',gen:'被檢驗的說法：泛化',opt:'最佳化與 KL 估計',recent:'同期 on-policy 蒸餾研究',me:'本文',other:'其他'};
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
  {const eq=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]),dd=(m,f)=>pairDiffs(m,f).map(r=>r.d);
   ok(eq(dd('SPA','LR'),[33.4,30,29.3,29.6]),'C1 SPA LR');ok(eq(dd('SPA','rollout'),[3.5,7.6,1.1,1.5]),'C1 SPA rollout');ok(eq(dd('SPA','KL'),[3.2,6.6,.8,.5]),'C1 SPA KL');
   ok(eq(dd('ACC','rollout'),[1,1,37,14]),'C1 ACC rollout');ok(eq(dd('ACC','KL'),[0,37,36,24]),'C1 ACC KL');ok(eq(dd('ACC','LR'),[0,37,2,14]),'C1 ACC LR');
   ok($$('#c1-fig svg').every(s=>$$('.c1pt[tabindex="0"]',s).length===1),'C1 roving tabindex')}
  // S5 two-token example
  {const S=softmax([1,0]),T=[.5,.5],DF=T.reduce((a,t,i)=>a+t*Math.log(t/S[i]),0),DR=S.reduce((a,s,i)=>a+s*Math.log(s/T[i]),0);
   ok(near(S[0],.7311,1e-4)&&near(DF,.1201,1e-4)&&near(DR,.1109,1e-4),'S5 two-token');
   const p=$('#grad .prose').textContent;ok(['0.7311','0.1201','0.1109','0.2311','0.1966'].every(v=>p.includes(v)),'S5 numbers in prose')}
  // C2
  {const T2=[[.5,.01,[.6371,1.6145,.49,1.1488,.5587,1.3098,4.5951]],[.3,.3,[0,0,0,0,0,0,0]],[.5,1e-6,[.6931,6.2146,.5,3.4539,.5701,3.938,13.8155]],[1e-6,.6,[7.6163,.9163,-.6,0,.6841,0,14.221]],[.9,1e-6,[2.3026,12.1089,.9,1.4411,1.0262,1.6432,16.0127]]];
   T2.forEach(([a,b,v])=>{const o=klParts(a,b),got=[o.DF,o.DR,o.gF[0],o.gR[0],o.nF,o.nR,o.dr];ok(got.every((g,i)=>near(g,v[i])),`C2 table ${a},${b}`);
     ok(Math.abs(o.sumF)<1e-12&&Math.abs(o.sumR)<1e-9&&o.nR<=o.dr+1e-12&&o.maxF<=1,`C2 invariants ${a},${b}`)});
   ok(fmtNum(-1.42e-5,2)==='0.00'&&fmtSci(1e-6)==='1.00×10⁻⁶'&&fmtGrad(-1.42e-5)==='≈ 0','C2 formats')}
  // C3
  {const T3=[[.1,.25,[1.0986,.0194,.0177,.2828,.2197]],[.1,.01,[4.5951,.0812,.0346,.2828,.919]],[.1,1e-4,[9.2102,.1628,.0353,.2828,1.842]],[.1,1e-8,[18.4207,.3256,.0354,.2828,3.6841]],[.1,1e-12,[27.631,.4885,.0354,.2828,5.5262]],[1,.01,[4.5951,.8123,.3465,2.8284,9.1902]],[1,1e-12,[27.631,4.8845,.3536,2.8284,55.262]]];
   T3.forEach(([e,d,v])=>{const o=boundParts(d,e),got=[o.dr,o.rev,o.fwd,o.fbound,o.rbound];ok(got.every((g,i)=>near(g,v[i],1e-4)),`C3 table eps=${e} delta=${d}`)});
   ok(near(boundParts(1/(1+Math.exp(16)),1).rev,2.828427,1e-6),'C3 crossing')}
  // C4
  {const T4=[[-2,1,[.022,.6313,.2367,.0733,.0366,.0001],1.51],[-2,0,[.0057,.6362,.2386,.0826,.0368,0],21.3542],[-1.5,1,[.0258,.6015,.2439,.0859,.0429,.0001],1.2288],[-1.5,0,[.0132,.6062,.2459,.0915,.0432,0],4.6211],
     [-1,1,[.03,.57,.25,.1,.0499,.0001],null],[0,1,[.1403,.4558,.2338,.108,.0603,.0017],null],[1,1,[.45,.25,.15,.08,.05,.02],null],[1.5,1,[.5863,.1881,.1128,.0602,.0376,.015],1.7321],[1.5,0,[.6228,.1431,.0928,.0532,.0352,.0529],4.6211],[2,1,[.7105,.1316,.0789,.0421,.0263,.0105],3],[2,0,[.7175,.0682,.0478,.0295,.0206,.1164],21.3542]];
   T4.forEach(([l,s,p,a])=>{const o=spectrum(l,!!s);ok(o.p.every((v,i)=>near(v,p[i]))&&(a===null?o.amp===null:near(o.amp,a)),`C4 table lam=${l} safe=${s}`)});
   ok(spectrum(1,true).p.every((v,i)=>Math.abs(v-PS[i])<1e-12)&&spectrum(-1,false).p.every((v,i)=>Math.abs(v-PT[i])<1e-12),'C4 anchors');
   let g=true;for(let i=0;i<=80;i++){const l=-2+.05*i,o=spectrum(l,true),lr=Math.round(l*100)/100;if(Math.abs(lr)<=1?o.amp!==null:!(o.amp<=3+1e-12))g=false}ok(g,'C4 grid amp<=3 and interior null')}
  // LO prose checks (word-boundary regex + keyword)
  {const P=id=>$$('#'+id+' .prose p').map(p=>p.textContent).join('\n'),num=(t,n)=>new RegExp('(?<![\\d.])'+n.replace('.','\\.')+'(?![\\d])').test(t);
   const LO=[['setup',['stop-gradient','半梯度'],[]],['kl',['mode-seeking','全詞彙','Prop. C.4'],[]],['design',['稀疏度','150 步'],['7']],['main-results',['每一組配對'],['72','73','1.3','11.2','14.0','85.3','89.6','51.9','60.0']],
     ['grad',['趨近 0','可任意大'],['1']],['bound-rev',['Eq. 26','完整輸出序列','一步的期望半梯度'],['2']],['spectrum',['勝算','幾何中點'],['0.2','3']],['spectrum-results',['學習率'],['80','5.2']],
     ['beyond',['RLVR','可能有助於抑制','百分點或相對比例'],['10','300']],['robust',['偏樂觀','OffPD 尤其明顯','沒有一致的 OnPD–OffPD 差距'],['622','59.0']]];
   LO.forEach(([id,kw,ns])=>{const t=P(id);ok(kw.every(k=>t.includes(k))&&ns.every(n=>num(t,n)),'LO prose '+id+' missing: '+[...kw.filter(k=>!t.includes(k)),...ns.filter(n=>!num(t,n))].join(','))})}
  // DOM
  ok($$('input[type=range]').every(r=>(r.getAttribute('aria-valuetext')||'').length>0),'aria-valuetext on all sliders');
  ok($$('details.check').length===10,'check boxes: '+$$('details.check').length);
  ok(['c1','c2','c3','c4'].every(id=>!!$('#'+id)),'plates present');
  {const tl=JSON.parse($('#timeline-data').textContent);ok(tl.length===14&&tl.every(e=>TL_GROUP[e.arxiv]),'timeline groups')}
  ok(!!document.querySelector('.katex')||typeof katex==='undefined','KaTeX rendered');
  ok(!document.querySelector('.katex-error'),'no KaTeX errors');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ================= boot ================= */
document.body.insertAdjacentHTML('afterbegin','<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs><pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" fill="var(--surface)"/><line x1="0" y1="0" x2="0" y2="5" style="stroke:var(--ink-2)" stroke-width="2"/></pattern><pattern id="hatchA" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" fill="var(--surface)"/><line x1="0" y1="0" x2="0" y2="5" style="stroke:var(--accent)" stroke-width="2.4"/></pattern></defs></svg>');
initTheme();initToc();renderGrid22();initC1();initC2();initC3();initC4();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;clearTimeout(rz);rz=setTimeout(()=>{renderC1();renderC2();renderC3();renderC4()},150)});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))addEventListener('load',()=>setTimeout(selfTest,50));
})();
