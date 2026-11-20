
/* ================= shared math (mirrors verify/check.py) ================= */
const log1mPow=(p,k)=>p>=1?(k>0?0:1):Math.exp(k*Math.log1p(-p));
const passK=(p,k)=>p>=1?1:-Math.expm1(k*Math.log1p(-p));
const nonzeroG=(p,G)=>p>=1?0:passK(p,G)-Math.pow(p,G);
function softmax(th){const m=Math.max(...th),e=th.map(t=>Math.exp(t-m)),Z=e.reduce((a,b)=>a+b,0);return e.map(v=>v/Z)}
function erfcc(x){ // Numerical Recipes; fractional error < 1.2e-7
  const z=Math.abs(x),t=1/(1+0.5*z);
  const r=t*Math.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+
    t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))));
  return x>=0?r:2-r}
const sgn=v=>v<0?'−'+String(-v):'+'+String(v);
function fmtSig(x,n=4){if(x===0)return'0';if(!Number.isFinite(x))return'—';const a=Math.abs(x);if(a<1e-3||a>=1e5)return x.toExponential(n-1).replace('-','−');return x.toPrecision(n).replace('-','−')}
const fx=(v,n)=>{if(Math.abs(v)<0.5*10**-n)v=0;return v.toFixed(n).replace('-','−')};
function svgBox(id,min,max,H){const box=$('#'+id),W=widthOf(box,min,max),svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);return{svg,W,H}}

/* ---------------- C1 ---------------- */
const c1={lp:Math.log10(.002),G:32,k:8};
function c1Vals(s){const p=10**s.lp;return{p,F:log1mPow(p,128),pk:passK(p,s.k),nz:nonzeroG(p,s.G),allw:100*log1mPow(p,s.G)}}
function c1Draw(){const {svg,W,H}=svgBox('c1-fig',300,620,230),L=40,R=W-10,T=12,B=H-30;const X=lin(-4,0,L,R),Y=lin(0,1,B,T);const s=c1,v=c1Vals(s);
  [0,.25,.5,.75,1].forEach(y=>{el('line',{x1:L,x2:R,y1:Y(y),y2:Y(y),class:'grid'},svg);tx(svg,L-6,Y(y)+4,String(y),{class:'lbl12','text-anchor':'end'})});
  (W<480?[-4,-2,0]:[-4,-3,-2,-1,0]).forEach(e=>tx(svg,X(e),B+18,'1e'+(e===0?'0':'−'+(-e)),{class:'lbl12','text-anchor':'middle'}));
  tx(svg,R,B+18+0,'',{});
  el('line',{x1:L,x2:R,y1:Y(.075),y2:Y(.075),class:'ref2'},svg);tx(svg,R-4,Y(.075)-5,'Table 4 基礎模型 pass@32 = 7.5%（MATH 平均）',{class:'lbl12 halo','text-anchor':'end'});
  const pts=f=>{const a=[];for(let i=0;i<=200;i++){const e=-4+4*i/200;a.push([X(e),Y(f(10**e))])}return lp(a)};
  el('path',{d:pts(p=>log1mPow(p,128)),class:'ref',style:'stroke-width:2'},svg);
  el('path',{d:pts(p=>passK(p,s.k)),class:'acc-s'},svg);
  el('path',{d:pts(p=>nonzeroG(p,s.G)),class:'cb'},svg);
  const xp=X(s.lp);el('line',{x1:xp,x2:xp,y1:T,y2:B,class:'axis0'},svg);
  el('circle',{cx:xp,cy:Y(v.F),r:5,class:'fill-ink3'},svg);el('rect',{x:xp-4.5,y:Y(v.pk)-4.5,width:9,height:9,class:'fill-acc'},svg);
  const ty=Y(v.nz);el('path',{d:`M${xp} ${ty-6}L${xp+6} ${ty+5}L${xp-6} ${ty+5}Z`,class:'fill-b'},svg);
  tx(svg,(L+R)/2,H-2,'',{});
  $('#c1-po').textContent=`${fx(s.lp,2)}（p = ${fmtSig(v.p,4)}）`;$('#c1-p').setAttribute('aria-valuetext',`p = ${fmtSig(v.p,4)}`);
  dl($('#c1-read'),[['p',fmtSig(v.p,4)],['通過 0/128 篩選',fx(v.F,4)],[`pass@${s.k}`,fx(v.pk,4),1],[`群組有梯度（G = ${s.G}）`,fx(v.nz,4)],['每 100 題全錯',fx(v.allw,1)+' 題']]);
  $('#c1-note').textContent=v.p>=1?'全部答對時優勢也全為 0，群組同樣沒有梯度。':(s.k===s.G?`此時 pass@k 與群組有梯度的曲線幾乎重合（只差 p^${s.G}）。`:'');
  svg.setAttribute('aria-label',`p = ${fmtSig(v.p,4)}：通過篩選 ${fx(v.F,4)}，pass@${s.k} ${fx(v.pk,4)}，群組有梯度 ${fx(v.nz,4)}`);
  announce('c1',`p = ${fmtSig(v.p,3)}，pass@${s.k} = ${fx(v.pk,4)}，群組有梯度 ${fx(v.nz,4)}`)}
function initC1(){const sl=$('#c1-p');sl.value=c1.lp;sl.addEventListener('input',()=>{c1.lp=+sl.value;c1Draw()});
  segInit($('#c1-G'),[32,128],c1.G,null,g=>{c1.G=g;c1Draw()});segInit($('#c1-k'),[1,4,8,16,32],c1.k,null,k=>{c1.k=k;c1Draw()});c1Draw()}

/* ---------------- C2 ---------------- */
const TH2=[1.0,0.5,0.0,-0.5,0.8,0.2];
const C2DEF={S:[1,1,1,1,0,0],z:[0,1,1,3],R:[.02,-.01,0,.03],est:'rloo'};
const c2=JSON.parse(JSON.stringify(C2DEF));
function computeC2(s){const pi0=softmax(TH2),pS=pi0.reduce((a,p,i)=>a+(s.S[i]?p:0),0),pic=pi0.map((p,i)=>s.S[i]?p/pS:0),g=s.z.length;
  const sumR=s.R.reduce((a,b)=>a+b,0),A=s.est==='rloo'?s.R.map(r=>r-(sumR-r)/(g-1)):s.R.slice();
  const L=[0,1,2,3,4,5].map(j=>A.reduce((a,ai,i)=>a+ai*((s.z[i]===j?1:0)-pic[j]),0)),Q=[0,1,2,3,4,5].map(j=>A.reduce((a,ai,i)=>a+ai*((s.z[i]===j?1:0)-pi0[j]),0));
  const sumA=A.reduce((a,b)=>a+b,0),diff=Math.max(...L.map((l,j)=>Math.abs(l-Q[j])));return{pi0,pS,pic,A,L,Q,sumA,diff}}
function c2Verdict(s,o){const rej=[0,1,2,3,4,5].filter(j=>!s.S[j]).map(j=>j+1);const same=o.diff<1e-12?'兩式相同。':`兩式不同，最大差 ${fmtSig(o.diff,4)}。`;
  if(!rej.length)return same+'目前沒有被拒絕的輸出。';
  const sA=Math.abs(o.sumA)<=1e-12?'ΣA = 0，沒有偏差。':(o.sumA>0?'ΣA > 0，這些輸出被壓低。':'ΣA < 0，這些輸出被抬高。');
  return `${same}被拒絕的輸出 ${rej.join('、')} 在 Q 中的分量是 −(ΣA)π₀(j)；${sA}`}
function c2Draw(){const s=c2,o=computeC2(s),{svg,W,H}=svgBox('c2-fig',300,900,316);const L=36,R=W-6,gw=(R-L)/6;
  const top={T:20,B:110},bot={T:150,B:280};const Yt=lin(0,.5,top.B,top.T);const m=Math.max(.02,...o.L.map(Math.abs),...o.Q.map(Math.abs))*1.15,Yb=lin(-m,m,bot.B,bot.T);
  tx(svg,L,12,'機率 π₀ 與 π_c',{class:'lbl12'});tx(svg,L,bot.T-8,'對 θ_j 的梯度分量（上方＝提高 θ_j）',{class:'lbl12'});
  el('line',{x1:L,x2:R,y1:top.B,y2:top.B,class:'axis0'},svg);el('line',{x1:L,x2:R,y1:Yb(0),y2:Yb(0),class:'axis0'},svg);
  const bw=Math.min(18,gw*.32);
  for(let j=0;j<6;j++){const cx=L+gw*(j+.5);
    el('rect',{x:cx-bw-1,y:Yt(o.pi0[j]),width:bw,height:top.B-Yt(o.pi0[j]),class:'fill-a',opacity:.35},svg);
    if(s.S[j])el('rect',{x:cx+1,y:Yt(o.pic[j]),width:bw,height:top.B-Yt(o.pic[j]),class:'fill-a'},svg);
    else tx(svg,cx+1+bw/2,top.B-4,'×',{class:'lbl12','text-anchor':'middle'});
    const bar=(v,x,cls)=>{const y0=Yb(0),y1=Yb(v);return el('rect',Object.assign({x,y:Math.min(y0,y1),width:bw,height:Math.max(.5,Math.abs(y1-y0))},cls),svg)};
    bar(o.L[j],cx-bw-1,{class:'fill-b'});bar(o.Q[j],cx+1,{fill:'none',stroke:'var(--ink-2)','stroke-width':1.6});
    tx(svg,cx,bot.B+16,`z=${j+1}`,{class:'lbl12','text-anchor':'middle'});if(!s.S[j])tx(svg,cx,bot.B+30,'拒絕',{class:'lbl12','text-anchor':'middle'})}
  dl($('#c2-read'),[['π₀',o.pi0.map(v=>fx(v,4)).join(', ')],['接受率 π₀(S)',fx(o.pS,4)],['平均抽幾次',fx(1/o.pS,3)],['優勢 A',o.A.map(v=>fx(v,5)).join(', ')],['ΣA',fx(o.sumA,5),1],['L（對 π）',o.L.map(v=>fx(v,5)).join(', ')],['Q（對 π₀）',o.Q.map(v=>fx(v,5)).join(', ')]]);
  const vt=c2Verdict(s,o);$('#c2-verdict').textContent=vt;$('#c2-verdict').classList.toggle('on',o.diff>=1e-12);svg.setAttribute('aria-label',vt);announce('c2',vt)}
function c2BuildCustom(){const S=$('#c2-S');S.textContent='';for(let j=0;j<6;j++){const l=document.createElement('label');l.className='chk';l.innerHTML=`<input type="checkbox" ${c2.S[j]?'checked':''} data-j="${j}">輸出 ${j+1} 格式正確`;S.appendChild(l)}
  S.onchange=e=>{const j=+e.target.dataset.j;const nx=c2.S.slice();nx[j]=e.target.checked?1:0;if(!nx.some(Boolean)){e.target.checked=true;announce('c2','至少要保留一個格式正確的輸出');return}c2.S=nx;
    const first=nx.findIndex(Boolean);c2.z=c2.z.map(z=>nx[z]?z:first);c2Rows();c2Draw()};c2Rows()}
function c2Rows(){const box=$('#c2-rows');box.textContent='';c2.z.forEach((z,i)=>{const d=document.createElement('div');d.className='srow';
  d.innerHTML=`<span class="lbl">樣本 ${i+1}</span><select aria-label="樣本 ${i+1} 的輸出 z">${[0,1,2,3,4,5].filter(j=>c2.S[j]).map(j=>`<option value="${j}" ${j===z?'selected':''}>z = ${j+1}</option>`).join('')}</select>`+slider(`c2-r${i}`,'獎勵 R',-0.04,0.04,0.001,c2.R[i]);
  box.appendChild(d);d.querySelector('select').addEventListener('change',e=>{c2.z[i]=+e.target.value;c2Draw()});
  const sl=d.querySelector('input[type=range]'),out=d.querySelector('output');const upd=()=>{out.textContent=fx(c2.R[i],3);sl.setAttribute('aria-valuetext','R = '+fx(c2.R[i],3))};upd();
  sl.addEventListener('input',()=>{c2.R[i]=+sl.value;upd();c2Draw()})})}
function initC2(){const syncEst=segInit($('#c2-est'),['rloo','reinforce'],c2.est,k=>k==='rloo'?'RLOO':'REINFORCE（A = R）',k=>{c2.est=k;c2Draw()});
  $('#c2-zero').addEventListener('click',()=>{c2.R=[0,0,0,0];c2Rows();c2Draw()});$('#c2-flip').addEventListener('click',()=>{c2.R=c2.R.map(r=>-r||0);c2Rows();c2Draw()});
  c2BuildCustom();c2Draw()}

/* ---------------- C3 ---------------- */
const C3DEF=[[1,2,0,1],[-1,0,1,-1],[0,1,-1,0],[2,1,3,0]];
const c3={C:C3DEF.map(r=>r.slice()),prev:[.010,.014],tau:.01,mu:0,sd:.02,sc:.005,r:4,m:3};
function computeC3a(C,prev,tau){const Rdj=C.map(r=>r.map(c=>c/64)),Rd=Rdj.map(r=>r.reduce((a,b)=>a+b,0)/r.length),Rhat=Rd.reduce((a,b)=>a+b,0)/Rd.length;
  const MA=(prev.reduce((a,b)=>a+b,0)+Rhat)/(prev.length+1),promote=MA>tau;let ds=0;Rd.forEach((v,d)=>{if(v>Rd[ds])ds=d});
  const js=[0,1,2,3].sort((a,b)=>Rdj[ds][a]-Rdj[ds][b]||a-b)[1];const sum=Rd.reduce((a,b)=>a+b,0),A=Rd.map(v=>v-(sum-v)/(Rd.length-1));return{Rdj,Rd,Rhat,MA,promote,ds,js,A}}
function computeC3b(mu,sd,sc,r,m,g=4,tau=.01){const sMA=Math.sqrt((sc*sc+sd*sd/(g*r))/m),P=.5*erfcc((tau-mu)/(sMA*Math.SQRT2));return{sMA,P,E:198*P}}
function c3Grid(){const t=$('#c3-grid');t.innerHTML=t.querySelector('caption').outerHTML+'<thead><tr><th>d</th>'+[1,2,3,4].map(j=>`<th>學生 ${j}</th>`).join('')+'<th class="num">R_d</th></tr></thead><tbody>'+
  c3.C.map((row,d)=>`<tr><th>${d+1}</th>`+row.map((c,j)=>`<td><input type="number" inputmode="numeric" min="-64" max="64" step="1" value="${c}" data-d="${d}" data-j="${j}" aria-label="資料集 ${d+1}，學生 ${j+1} 的淨增答對題數" aria-describedby="c3-err"></td>`).join('')+`<td class="num" id="c3-rd${d}"></td></tr>`).join('')+'</tbody>';
  t.oninput=e=>{const i=e.target;if(i.tagName!=='INPUT')return;const v=Number(i.value),okv=i.value.trim()!==''&&Number.isInteger(v)&&v>=-64&&v<=64;i.setAttribute('aria-invalid',String(!okv));
    c3.C[+i.dataset.d][+i.dataset.j]=okv?v:0;$('#c3-err').textContent=$$('input[aria-invalid=true]',t).length?'請輸入 −64 到 64 的整數（無效的格子以 0 計算）。':'';c3aDraw()}}
function c3aDraw(){const o=computeC3a(c3.C,c3.prev,c3.tau);o.Rd.forEach((v,d)=>{$('#c3-rd'+d).textContent=fx(v,6)});
  $$('#c3-grid tbody tr').forEach((tr,d)=>{tr.classList.toggle('sel',o.promote&&d===o.ds);$$('td',tr).forEach((td,j)=>td.classList.toggle('sel',o.promote&&d===o.ds&&j===o.js))});
  $('#c3-p1o').textContent=fx(c3.prev[0],3);$('#c3-p2o').textContent=fx(c3.prev[1],3);
  $('#c3-steps').innerHTML=[`第 13 行：R_d = (${o.Rd.map(v=>fx(v,6)).join(', ')})`,`第 16 行：步平均 = ${fx(o.Rhat,6)}`,`第 17 行：移動平均 = (${fx(c3.prev[0],3)} + ${fx(c3.prev[1],3)} + ${fx(o.Rhat,6)}) / 3 = ${fx(o.MA,6)}`,
    `第 18 行：${fx(o.MA,6)} ${o.MA>c3.tau?'>':'≤'} τ = ${c3.tau}`+(o.promote?`；第 19–20 行：d* = ${o.ds+1}，j* = ${o.js+1}（c = ${c3.C[o.ds][o.js]}）`:''),`第 25 行：老師優勢 A_d = (${o.A.map(v=>fx(v,6)).join(', ')})，ΣA = ${fx(o.A.reduce((a,b)=>a+b,0),6)}`].map(s=>`<li>${s}</li>`).join('');
  dl($('#c3-read'),[['步平均',fx(o.Rhat,6)],['移動平均',fx(o.MA,6),1],['τ',String(c3.tau)]]);
  const v=$('#c3-verdict');v.classList.toggle('on',o.promote);v.textContent=o.promote?`✓ 晉升：資料集 ${o.ds+1}、學生 ${o.js+1}；D_best 增加 64 題。`:'✗ 不晉升：學生回到基準。';
  const {svg,W,H}=svgBox('c3-adv',260,560,120),L=60,R=W-50,mid=(L+R)/2,mx=Math.max(.005,...o.A.map(Math.abs)),X=lin(-mx,mx,L,R);el('line',{x1:mid,x2:mid,y1:4,y2:H-4,class:'axis0'},svg);
  o.A.forEach((a,d)=>{const y=8+d*27;el('rect',{x:Math.min(mid,X(a)),y,width:Math.max(1,Math.abs(X(a)-mid)),height:16,class:a>=0?'fill-acc':'fill-b'},svg);tx(svg,4,y+12,`A_${d+1}`,{class:'lbl12'});tx(svg,W-4,y+12,fx(a,4),{class:'lbl12','text-anchor':'end'})});
  svg.setAttribute('aria-label','老師的 RLOO 優勢：'+o.A.map((a,d)=>`資料集 ${d+1} ${fx(a,4)}`).join('，'));
  announce('c3',v.textContent+` 移動平均 ${fx(o.MA,6)}。`);return o}
function c3bDraw(){const s=c3,o=computeC3b(s.mu,s.sd,s.sc,s.r,s.m);['mu','sd','sc'].forEach(k=>{$(`#c3-${k}o`).textContent=fx(s[k],3);$(`#c3-${k}`).setAttribute('aria-valuetext',`${k} = ${fx(s[k],3)}`)});
  const {svg,W,H}=svgBox('c3b-fig',300,620,200),L=14,R=W-14,T=14,B=H-30,lo=Math.min(s.mu-4*o.sMA,-.01),hi=Math.max(s.mu+4*o.sMA,.03),X=lin(lo,hi,L,R),pk=1/(o.sMA*Math.sqrt(2*Math.PI)),Y=lin(0,pk*1.08,B,T);
  const pdf=x=>Math.exp(-.5*((x-s.mu)/o.sMA)**2)*pk;const pts=[],sh=[[X(Math.max(.01,lo)),B]];for(let i=0;i<=240;i++){const x=lo+(hi-lo)*i/240;pts.push([X(x),Y(pdf(x))]);if(x>=.01)sh.push([X(x),Y(pdf(x))])}sh.push([X(hi),B]);
  el('path',{d:lp(sh)+'Z',class:'shade'},svg);el('path',{d:lp(pts),class:'acc-s'},svg);el('line',{x1:L,x2:R,y1:B,y2:B,class:'axis0'},svg);
  el('line',{x1:X(.01),x2:X(.01),y1:T,y2:B,class:'ref'},svg);tx(svg,X(.01)+4,T+10,'τ = 0.01',{class:'lbl12'});
  [lo,s.mu,hi].forEach((v,i)=>tx(svg,X(v),B+18,fx(v,3),{class:'lbl12','text-anchor':['start','middle','end'][i]}));
  dl($('#c3b-read'),[['移動平均標準差',fx(o.sMA,5)],['每步誤晉升機率',o.P<1e-12?'< 1e-12':fmtSig(o.P,4),1],['200 步中期望次數',fx(o.E,3)]]);
  const combos=[[1,1],[4,1],[1,3],[4,3]];$('#c3b-tab').innerHTML=$('#c3b-tab caption').outerHTML+'<thead><tr><th>r</th><th>m</th><th class="num">P（目前 σ_c）</th><th class="num">P（σ_c = 0，下界）</th></tr></thead><tbody>'+
    combos.map(([r,m])=>{const a=computeC3b(s.mu,s.sd,s.sc,r,m).P,b=computeC3b(s.mu,s.sd,0,r,m).P;return`<tr${r===s.r&&m===s.m?' class="hl"':''}><td>${r}</td><td>${m}</td><td class="num">${fmtSig(a,3)}</td><td class="num">${fmtSig(b,3)}</td></tr>`}).join('')+'</tbody>';
  svg.setAttribute('aria-label',`移動平均的分布，τ 右側面積 = 每步誤晉升機率 ${fmtSig(o.P,3)}`);announce('c3b',`每步誤晉升機率 ${fmtSig(o.P,3)}，200 步期望 ${fx(o.E,2)} 次`)}
function initC3(){c3Grid();c3aDraw();
  [['c3-p1',0],['c3-p2',1]].forEach(([id,i])=>{const sl=$('#'+id);sl.addEventListener('input',()=>{c3.prev[i]=+sl.value;sl.setAttribute('aria-valuetext',fx(c3.prev[i],3));c3aDraw()})});
  $('#c3-reset').addEventListener('click',()=>{c3.C=C3DEF.map(r=>r.slice());c3.prev=[.010,.014];$('#c3-p1').value=.01;$('#c3-p2').value=.014;$('#c3-err').textContent='';c3Grid();c3aDraw()});
  ['mu','sd','sc'].forEach(k=>{const sl=$('#c3-'+k);sl.addEventListener('input',()=>{c3[k]=+sl.value;c3bDraw()})});
  segInit($('#c3-r'),[1,2,4,8],c3.r,null,v=>{c3.r=v;c3bDraw()});segInit($('#c3-m'),[1,3],c3.m,null,v=>{c3.m=v;c3bDraw()});
  const tabs=$$('#c3 [role=tab]');const pick=i=>{tabs.forEach((t,j)=>{t.setAttribute('aria-selected',String(i===j));t.tabIndex=i===j?0:-1;$('#'+t.getAttribute('aria-controls')).hidden=i!==j});if(i===1)c3bDraw()};
  tabs.forEach((t,i)=>{t.addEventListener('click',()=>pick(i));t.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();const j=(i+1)%2;pick(j);tabs[j].focus()}})});c3bDraw()}

/* ---------------- C4 ---------------- */
const RES=JSON.parse($('#results-data').textContent);
const DSN={MATH:'MATH',HARP:'HARP',OLYMP:'OlympiadBench'};
const DEFROW={MATH:'SOAR-PQ (Ours)',HARP:'SOAR-PQ (Ours)',OLYMP:'SOAR-PQ (HARP) (Ours)'};
const REAL=new Set(['HARP train (128)','MATH train (128)','MATH train (Full)']);
const grpOf=n=>REAL.has(n)?2:(/Ours|Intrinsic|Inference with/.test(n)?1:0);
const GRPN=['推論與直接訓練','SOAR 與老師取樣','真實題目參照'];
const disp=n=>n.replace('Hard-Only (g = 128)','Hard-Only（G = 128；原文 g = 128）');
const c4={ds:'MATH',ki:4,mode:'abs',sel:'SOAR-PQ (Ours)'};
const KS=[1,4,8,16,32];
function rowOf(ds,n){return RES[ds].find(r=>r.name===n)}
function deriveC4(ds,ki,name){const [m,sd]=rowOf(ds,name).vals[ki],[ho,hsd]=rowOf(ds,'Hard-Only').vals[ki],full=rowOf(ds,'MATH train (Full)').vals[ki][0];
  const den=full-ho,rec=den>0?(m-ho)/den:null;return{m,sd,ho,hsd,dpp:m-ho,ratio:m/ho,rec,overlap:(m-sd)<=(ho+hsd)&&(ho-hsd)<=(m+sd),
    s:{dpp:(m-ho>=0?'+':'−')+Math.abs(m-ho).toFixed(1),ratio:(m/ho).toFixed(2)+'×',rec:rec===null?'—':rec.toFixed(2).replace('-','−')}}}
function c4Draw(){const s=c4,rows=RES[s.ds],box=$('#c4-fig');const W=widthOf(box,280,640),L=8,R=W-8;const hoV=rowOf(s.ds,'Hard-Only').vals[s.ki][0];
  const val=r=>{const [m,sd]=r.vals[s.ki];return s.mode==='abs'?[m,sd]:[m-hoV,sd]};let lo=s.mode==='abs'?0:Math.min(0,...rows.map(r=>val(r)[0]-val(r)[1])),hi=Math.max(...rows.map(r=>val(r)[0]+val(r)[1]))*1.05;
  const X=lin(lo,hi,L,R);box.textContent='';let g=-1;
  rows.forEach(r=>{const gr=grpOf(r.name);if(gr!==g){g=gr;const h=document.createElement('div');h.className='c4-grp';h.textContent=GRPN[gr];box.appendChild(h);
      if(gr===0){const nn=document.createElement('p');nn.className='c4-gnote';nn.textContent='Hard-Only (g = 128)：論文在這裡用 g 表示學生群組大小，與本頁的 g（資料集數）衝突，所以本頁寫 G。';box.appendChild(nn)}
      if(gr===2){const nn=document.createElement('p');nn.className='c4-gnote';nn.textContent='論文稱 MATH train (Full) 為上界參考，但它不是最大值：MATH 上 HARP train (128) 23.0 > 22.0；OlympiadBench 上 MATH train (128) 14.6 > 13.1。';box.appendChild(nn)}}
    const b=document.createElement('button');b.type='button';b.className='c4-row';b.setAttribute('aria-pressed',String(r.name===s.sel));const [m,sd]=val(r),raw=r.vals[s.ki];
    b.setAttribute('aria-label',`${disp(r.name)}：${raw[0]} ± ${raw[1]}`);b.innerHTML=`<span class="lbl" style="padding:4px 8px 0">${disp(r.name)}<output>${raw[0].toFixed(1)} ± ${raw[1].toFixed(1)}</output></span>`;
    const svg=el('svg',{width:W,height:22,viewBox:`0 0 ${W} 22`,'aria-hidden':'true'});b.appendChild(svg);
    el('line',{x1:X(s.mode==='abs'?hoV:0),x2:X(s.mode==='abs'?hoV:0),y1:0,y2:22,class:'ref'},svg);
    el('line',{x1:X(Math.max(lo,m-sd)),x2:X(m+sd),y1:11,y2:11,stroke:'var(--ink-2)','stroke-width':2},svg);
    const cx=X(m);if(gr===2)el('rect',{x:cx-5,y:6,width:10,height:10,transform:`rotate(45 ${cx} 11)`,class:'fill-a'},svg);
    else if(/Ours/.test(r.name))el('circle',{cx,cy:11,r:5.5,class:'fill-acc'},svg);else el('circle',{cx,cy:11,r:5,fill:'var(--surface)',stroke:'var(--ink-2)','stroke-width':2},svg);
    b.addEventListener('click',()=>{s.sel=r.name;c4Draw()});box.appendChild(b)});
  const ax=el('svg',{width:W,height:20,viewBox:`0 0 ${W} 20`,'aria-hidden':'true'});[lo,(lo+hi)/2,hi].forEach((v,i)=>tx(ax,X(v),14,fx(v,1)+(s.mode==='abs'?'%':' pp'),{class:'lbl12','text-anchor':['start','middle','end'][i]}));box.appendChild(ax);
  const o=deriveC4(s.ds,s.ki,s.sel);$('#c4-sel').textContent=`${disp(s.sel)}｜${DSN[s.ds]}｜pass@${KS[s.ki]}`;
  dl($('#c4-read'),[['平均 ± SD',`${o.m.toFixed(1)} ± ${o.sd.toFixed(1)}`],['Hard-Only',`${o.ho.toFixed(1)} ± ${o.hsd.toFixed(1)}`],['差（百分點）',o.s.dpp,1],['倍數',o.s.ratio],['回收比例'+(s.ds==='OLYMP'?'（本頁延伸）':''),o.s.rec],['± SD 區間重疊',o.overlap?'是':'否']]);
  $('#c4-note').textContent=(s.ds==='OLYMP'?'OlympiadBench 的回收比例是本頁延伸，分母 MATH train (Full) 不是最大值。':'回收比例 = (方法 − Hard-Only) / (MATH train (Full) − Hard-Only)。')+'區間重疊只是描述，不是顯著性檢定。';
  $('#c4-table').innerHTML=`<caption>${DSN[s.ds]} pass@k（%）</caption><thead><tr><th>方法</th>${KS.map(k=>`<th class="num">k=${k}</th>`).join('')}</tr></thead><tbody>`+rows.map(r=>`<tr${r.name===s.sel?' class="hl"':''}><td>${disp(r.name)}</td>${r.vals.map(([m,sd])=>`<td class="num">${m.toFixed(1)} ± ${sd.toFixed(1)}</td>`).join('')}</tr>`).join('')+'</tbody>';
  announce('c4',`${disp(s.sel)}，pass@${KS[s.ki]}：差 ${o.s.dpp} 百分點，倍數 ${o.s.ratio}，回收比例 ${o.s.rec}`);return o}
function initC4(){segInit($('#c4-ds'),['MATH','HARP','OLYMP'],c4.ds,k=>DSN[k],k=>{c4.ds=k;c4.sel=DEFROW[k];c4Draw()});
  segInit($('#c4-k'),[0,1,2,3,4],c4.ki,i=>'k='+KS[i],i=>{c4.ki=i;c4Draw()});segInit($('#c4-mode'),['abs','pp'],c4.mode,m=>m==='abs'?'絕對值':'相對 Hard-Only',m=>{c4.mode=m;c4Draw()});c4Draw()}

/* ---------------- static figures ---------------- */
function boxT(svg,x,y,w,h,lines,cls){el('rect',{x,y,width:w,height:h,rx:10,class:cls||'box'},svg);lines.forEach((s,i)=>tx(svg,x+w/2,y+h/2+(i-(lines.length-1)/2)*17+5,s,{class:'lbl13','text-anchor':'middle'}))}
function arrow(svg,x1,y1,x2,y2){el('path',{d:`M${x1} ${y1}L${x2} ${y2}`,class:'arr','marker-end':'url(#ah)'},svg)}
function defs(svg){const d=el('defs',{},svg),m=el('marker',{id:'ah',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'},d);el('path',{d:'M0 0L10 5L0 10Z',fill:'var(--ink-2)'},m)}
function drawF0(){const box=$('#f0'),W=widthOf(box,300,860),svg=box.querySelector('svg');svg.textContent='';defs(svg);
  if(W>=600){const H=250;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);const bw=(W-60)/5,y=70,h=64;
    const labs=[['老師'],['g 個合成資料集','（各 n 題）'],['每個資料集訓練','r 個學生（10 步）'],['在 64 道獎勵題上','測正確率'],['資料集獎勵','R_d']];
    labs.forEach((l,i)=>{const x=6+i*(bw+12);boxT(svg,x,y,bw,h,l,i===0?'box':'box');if(i<4)arrow(svg,x+bw+1,y+h/2,x+bw+11,y+h/2)});
    el('rect',{x:6+2*(bw+12)-6,y:y-22,width:2*bw+24,height:h+44,rx:12,class:'dash'},svg);tx(svg,6+2*(bw+12),y-28,'內層：學生 RLOO',{class:'lbl12'});
    const xr=6+4*(bw+12)+bw/2,xl=6+bw/2;el('path',{d:`M${xr} ${y+h}C${xr} ${y+h+70} ${xl} ${y+h+70} ${xl} ${y+h+2}`,class:'arr','marker-end':'url(#ah)'},svg);tx(svg,(xl+xr)/2,y+h+66,'外層：RLOO 更新老師',{class:'lbl13','text-anchor':'middle'});
    tx(svg,6,24,'老師看不到難題訓練集（只收到資料集獎勵這個數字）',{class:'lbl12'})}
  else{const bw=Math.min(W-40,300),x=(W-bw)/2,h=52,gap=26,H=6*(h+gap)+30;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
    const labs=[['老師'],['g 個合成資料集（各 n 題）'],['每個資料集訓練 r 個學生','（10 步，內層 RLOO）'],['在 64 道獎勵題上測正確率'],['資料集獎勵 R_d'],['外層：RLOO 更新老師']];
    labs.forEach((l,i)=>{const y=24+i*(h+gap);boxT(svg,x,y,bw,h,l,i===2||i===3?'box':'box');if(i<5)arrow(svg,W/2,y+h+1,W/2,y+h+gap-2)});
    tx(svg,x,14,'老師看不到難題訓練集',{class:'lbl12'})}}
function drawF1(){const {svg,W,H}=svgBox('f1',300,640,210),L=40,R=W-10,T=14,B=H-24,X=lin(0,1,L,R),Y=lin(0,1,B,T);const Er=s=>s>=1?0:(1-s)-log1mPow(s,32);
  [0,.5,1].forEach(y=>{el('line',{x1:L,x2:R,y1:Y(y),y2:Y(y),class:'grid'},svg);tx(svg,L-6,Y(y)+4,String(y),{class:'lbl12','text-anchor':'end'})});
  [0,.25,.5,.75,1].forEach(x=>tx(svg,X(x),B+16,String(x),{class:'lbl12','text-anchor':'middle'}));
  const pts=[];for(let i=0;i<=400;i++){const s=i/400;pts.push([X(s),Y(Er(s))])}el('path',{d:lp(pts),class:'acc-s'},svg);
  const ss=1-Math.pow(32,-1/31);el('circle',{cx:X(ss),cy:Y(Er(ss)),r:5,class:'fill-acc'},svg);tx(svg,X(ss)+8,Y(Er(ss))-6,`s* = ${fx(ss,3)}，E[u] = ${fx(Er(ss),3)}`,{class:'lbl13'})}
function drawF2(){const box=$('#f2'),W=widthOf(box,300,860),svg=box.querySelector('svg');svg.textContent='';defs(svg);
  const P=[['PS','SOAR 訓練中的晉升學生','直接推論'],['PQ','PQ＋fail@128 訓練集','從頭訓練新學生'],['Hard-Only','fail@128 訓練集','新學生（G = 32 或 128）'],['老師取樣','老師取樣 128 題＋fail@128 訓練集','新學生']];
  if(W>=600){const rh=50,H=P.length*rh+40;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);const c=[90,(W-90)*.42,(W-90)*.33];
    P.forEach((p,i)=>{const y=10+i*rh;tx(svg,4,y+24,p[0],{class:'lbl13',style:'font-weight:600'});let x=90;boxT(svg,x,y,c[1]-12,38,[p[1]]);arrow(svg,x+c[1]-11,y+19,x+c[1]-1,y+19);x+=c[1];boxT(svg,x,y,c[2]-12,38,[p[2]]);arrow(svg,x+c[2]-11,y+19,x+c[2]-1,y+19);x+=c[2];boxT(svg,x,y,W-x-4,38,['測試集 pass@k'],'box')});
    tx(svg,4,H-6,'所有路徑都在 fail@128 測試集上，以每題 32 個樣本計算 pass@k。',{class:'lbl12'})}
  else{const bw=W-8,bh=34,H=P.length*(3*bh+44)+10;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
    P.forEach((p,i)=>{let y=6+i*(3*bh+44);tx(svg,4,y+12,p[0],{class:'lbl13',style:'font-weight:600'});y+=18;[p[1],p[2],'測試集 pass@k'].forEach((s,j)=>{boxT(svg,4,y,bw,bh-8,[s]);if(j<2)arrow(svg,W/2,y+bh-7,W/2,y+bh-1);y+=bh})})}}

/* ================= timeline ================= */
const TL_GROUP={'Curriculum learning':['cl','soar-curriculum-and-environment-design'],'2012.02096':['cl','soar-curriculum-and-environment-design'],
 '1502.03492':['bi','soar-bilevel-meta-learning-dataset-distillation'],'1611.02779':['bi','soar-bilevel-meta-learning-dataset-distillation'],'1703.03400':['bi','soar-bilevel-meta-learning-dataset-distillation'],'1811.10959':['bi','soar-bilevel-meta-learning-dataset-distillation'],
 '1703.05407':['sp','soar-asymmetric-self-play-origins'],'1712.01815':['sp','soar-asymmetric-self-play-origins'],'2101.04882':['sp','soar-asymmetric-self-play-origins'],
 '2402.14740':['rl','soar-rlvr-sparse-reward-plateau'],'2501.12948':['rl','soar-rlvr-sparse-reward-plateau'],
 '2505.03335':['llm','soar-llm-data-free-self-play'],'2508.05004':['llm','soar-llm-data-free-self-play'],'2505.21444':['llm','soar-llm-data-free-self-play'],
 '2601.18778':['me','soar-teacher-student-rloo-method']};
const TL_NAME={cl:'課程學習',bi:'雙層最佳化',sp:'非對稱自我對弈',rl:'RLVR 與 RLOO',llm:'LLM 自我對弈',me:'本文',other:'其他'};
const tlKey=e=>e.arxiv||e.title;
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.map((e,i)=>[e,i]).sort((a,b)=>((Number(a[0].year)||0)-(Number(b[0].year)||0))||a[1]-b[1]).map(x=>x[0]);
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[tlKey(e)]||['other',null];if(!TL_GROUP[tlKey(e)])console.warn('timeline: ungrouped',tlKey(e));used.add(g[0]);const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${g[1]?`<a href="${g[1]}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
  const on=new Set(used);
  Object.keys(TL_NAME).filter(k=>used.has(k)).forEach(k=>{const b=document.createElement('button');b.type='button';b.className='chip';b.setAttribute('aria-pressed','true');b.textContent=TL_NAME[k];b.dataset.g=k;
    b.addEventListener('click',()=>{if(on.has(k)&&on.size===used.size){on.clear();on.add(k)}else if(on.has(k)){on.delete(k);if(!on.size)used.forEach(x=>on.add(x))}else on.add(k);
      $$('.chip',chips).forEach(c=>c.setAttribute('aria-pressed',String(on.has(c.dataset.g))));$$('.ev',track).forEach(ev=>ev.classList.toggle('dim',!on.has(ev.dataset.g)))});chips.appendChild(b)});
  const rail=$('#rail'),by=dir=>rail.scrollBy({left:dir*Math.min(rail.clientWidth*.8,560),behavior:RM?'auto':'smooth'});
  $('#tl-prev').addEventListener('click',()=>by(-1));$('#tl-next').addEventListener('click',()=>by(1));
  rail.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){by(1);e.preventDefault()}if(e.key==='ArrowLeft'){by(-1);e.preventDefault()}});return data}

