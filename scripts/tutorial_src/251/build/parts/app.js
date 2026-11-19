
/* ---------- page helpers ---------- */
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const loc=v=>Math.round(v).toLocaleString('zh-TW');
const SUP={'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'};
const sup=n=>String(n).split('').map(c=>SUP[c]||c).join('');
const minus=s=>String(s).replace(/^-/,'−');
function pct(p){if(p>=0.01)return Number(p.toPrecision(2))+' %';const e=Math.floor(Math.log10(p)+1e-9),m=p/10**e;return Number(m.toPrecision(2))+'×10'+sup(e)+' %'}
function hatch(svg,id,cls){const d=el('defs',null,svg),p=el('pattern',{id,patternUnits:'userSpaceOnUse',width:5,height:5,patternTransform:'rotate(45)'},d);el('line',{x1:0,y1:0,x2:0,y2:5,class:cls},p)}

/* ---------- C1: three-axis map of Table I ---------- */
const T1=JSON.parse($('#t1-data').textContent),ROWS=T1.rows;
const AX={I:'隱式',E:'顯式',on:'線上',off:'離線',S:'短期',L:'長期'};
const RULE_NAME={opt:'最佳化式寫入',st:'狀態轉移',sig:'訊號閘控',adm:'准入／淘汰／整併',obj:'目標誘導或結構式'};
const XLINK={'Titans':['../index.html#paper-243','在關聯圖中查看 No. 243'],'Engram':['../index.html#paper-90','在關聯圖中查看 No. 090'],'TTT-E2E':['ttt-e2e.html','研究筆記：TTT-E2E'],'In-Place TTT':['in-place-ttt.html','研究筆記：In-Place TTT']};
const XNOTE={'Titans':'Table I 年份為 2025，arXiv 編號 2501.00663。§IV-D a 稱其為「短時間尺度」，指的是更新頻率；Table I 的「長期」指影響計算的視野。',
  'LM2':'Table I 標為離線；§IV-A e 把它與 MEMORYLLM 並列為槽導向或前向更新變體。本頁照錄 Table I。',
  'Samba':'混合模型依主要機制標註，歸在隱式（Table I 標題）。','HAM':'混合模型依主要機制標註，歸在隱式（Table I 標題）。','AMOR':'混合模型依主要機制標註，歸在隱式（Table I 標題）。','Jamba / Jamba-1.5':'混合模型依主要機制標註，歸在隱式（Table I 標題）。'};
const c1={rep:'all',upd:'all',per:'all',rules:new Set(),yr:'all',sel:'Titans',step:0};
const yrOk=(y,v)=>y==='all'||(y==='le2023'?v<=2023:y==='2024'?v===2024:v>=2025);
const c1filter=st=>r=>(st.rep==='all'||r[3]===st.rep)&&(st.upd==='all'||r[4]===st.upd)&&(st.per==='all'||r[5]===st.per)
  &&(!st.rules.size||r[8].some(x=>st.rules.has(x[0]))||(st.rules.has('none')&&!r[8].length))&&yrOk(st.yr,r[2]);
const axes3=r=>`<span class="axes3" aria-hidden="true"><i${r[3]==='E'?' class="f"':''}>${r[3]}</i><i${r[4]==='on'?' class="f"':''}>${r[4]==='on'?'On':'Off'}</i><i${r[5]==='L'?' class="f"':''}>${r[5]}</i></span>`;
function renderC1(say){
  const f=c1filter(c1),vis=ROWS.filter(f),cc=cellCounts(ROWS,f),all=cellCounts(ROWS);let h='';
  for(const rep of['I','E'])for(const upd of['on','off']){
    h+=`<div class="c1-big"><h4 class="mobile-hide">${AX[rep]} · ${AX[upd]}</h4><div class="c1-in">`;
    for(const per of['S','L']){const k=rep+'|'+upd+'|'+per,n=cc[k]||0,rows=vis.filter(r=>r[3]===rep&&r[4]===upd&&r[5]===per);
      h+=`<div class="c1-cell${all[k]?'':' zero'}"><h5><span class="full">${AX[rep]} · ${AX[upd]} · </span>${AX[per]}：<span class="n">${n}</span> 列</h5>`;
      if(!all[k])h+='<p class="z">0（Table I 無此組合）</p>';else if(!n)h+='<p class="z">篩選後沒有列</p>';
      else h+='<ul>'+rows.map(r=>`<li><button type="button" class="pill" data-n="${esc(r[0])}" aria-pressed="${r[0]===c1.sel}" aria-label="${esc(r[0])}，${AX[r[3]]}、${AX[r[4]]}、${AX[r[5]]}，${r[2]}">${esc(r[0])}</button></li>`).join('')+'</ul>';
      h+='</div>'}
    h+='</div></div>'}
  $('#c1-map').innerHTML=h;
  $$('#c1 .seg').forEach(s=>$$('button',s).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===c1[s.dataset.k]))));
  $$('#c1-rules .chip').forEach(b=>b.setAttribute('aria-pressed',String(c1.rules.has(b.dataset.r))));
  $('#c1-yr').value=c1.yr;
  $$('#c1-steps button[data-step]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.step===c1.step)));
  renderHist(vis);renderCard();
  if(say)announce('c1',`顯示 ${vis.length} 列`);
  return vis.length}
function renderCard(){const r=ROWS.find(x=>x[0]===c1.sel),card=$('#c1-card');
  if(!r){card.innerHTML='<p class="nt">點選上方任一系統名稱，查看它的標註與判讀路徑。</p>';return}
  const rules=r[8].length?r[8].map(([c,s])=>`${RULE_NAME[c]}（${s==='T2'?'Table II':'§'+s+' 內文'}）`).join('、')
    :(/Conditional Parameter/.test(r[6])?'Table II 只以「MoE models [12],[41]」泛稱，未逐一列出此系統；本頁不擅自歸列':'Table II 未列');
  const yn=b=>b?'是':'否',lk=XLINK[r[0]];
  card.innerHTML=`<div class="dh"><b>${esc(r[0])}</b><span class="mono">[${r[1]}] · Table I ${r[2]}</span>${axes3(r)}</div>
  <dl><dt>三軸</dt><dd>${AX[r[3]]}／${AX[r[4]]}／${AX[r[5]]}</dd><dt>Main Paradigm</dt><dd>${esc(r[6])}</dd><dt>主要章節</dt><dd>§${esc(r[7])}</dd><dt>更新規則</dt><dd>${rules}</dd></dl>
  <div class="lbl">判讀路徑</div><ol><li>有獨立讀寫介面？→ ${yn(r[3]==='E')}（§II-A）</li><li>推論期更新？→ ${yn(r[4]==='on')}（§II-B）</li><li>有效視野超出局部視窗？→ ${yn(r[5]==='L')}（§II-C）</li></ol>
  ${XNOTE[r[0]]?`<p class="nt">${XNOTE[r[0]]}</p>`:''}${lk?`<p class="nt"><a href="${lk[0]}">${lk[1]}</a></p>`:''}`}
function renderHist(vis){const box=$('#c1-hist').parentNode,svg=$('#c1-hist'),W=widthOf(box,260,440),H=170,yrs=[];for(let y=2017;y<=2026;y++)yrs.push(y);
  const cnt=yrs.map(y=>({I:vis.filter(r=>r[2]===y&&r[3]==='I').length,E:vis.filter(r=>r[2]===y&&r[3]==='E').length}));
  const mx=Math.max(4,...cnt.map(c=>c.I+c.E));
  const fr=frame(svg,{W,H,m:{l:26,r:6,t:16,b:24},x:[2016.5,2026.5],y:[0,mx],yt:[0,Math.ceil(mx/2),mx].map(v=>[v,String(v)])});
  hatch(svg,'c1-hat','hat');const bw=Math.max(6,(fr.x1-fr.x0)/10*.62);
  yrs.forEach((y,i)=>{const x=fr.X(y)-bw/2,c=cnt[i];let yb=fr.Y(0);
    if(c.I){const yt=fr.Y(c.I);el('rect',{x,y:yt,width:bw,height:yb-yt,class:'ihat'},fr.g);yb=yt}
    if(c.E){const yt=fr.Y(c.I+c.E);el('rect',{x,y:yt,width:bw,height:yb-yt,class:'esol'},fr.g);yb=yt}
    if(c.I+c.E)tx(fr.g,fr.X(y),yb-4,String(c.I+c.E),{class:'tk','text-anchor':'middle'});
    if(y%3===2||W>=380)tx(fr.g,fr.X(y),fr.y0+16,W>=380?String(y).slice(2):"'"+String(y).slice(2),{class:'tk','text-anchor':'middle'})});
  svg.setAttribute('aria-label','Table I 每年列數（目前篩選）：'+yrs.map((y,i)=>`${y} 年隱式 ${cnt[i].I}、顯式 ${cnt[i].E}`).filter((s,i)=>cnt[i].I+cnt[i].E).join('；'))}
function c1reset(){c1.rep='all';c1.upd='all';c1.per='all';c1.rules=new Set();c1.yr='all';c1.step=0}
function c1step(s){c1reset();c1.step=s;if(s===2)c1.per='S';if(s===3)c1.rules.add('opt');if(s===4)c1.yr='le2023';if(s>=3)$('#c1-adv').open=true;return renderC1(true)}
function initC1(){
  $('#c1-bnd').innerHTML=T1.boundary.map(b=>`<article><b>${esc(b[0])}</b><span class="v">${esc(b[3])} · §${esc(b[2])}${b[1]?' · ['+b[1]+']':''}</span><br>${esc(b[4])}</article>`).join('');
  $$('#c1-steps button[data-step]').forEach(b=>b.addEventListener('click',()=>{c1step(+b.dataset.step);$('#c1-map').scrollIntoView({behavior:RM?'auto':'smooth',block:'nearest'})}));
  $('#c1-clear').addEventListener('click',()=>{c1reset();renderC1(true)});
  $$('#c1 .seg').forEach(s=>s.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;c1[s.dataset.k]=b.dataset.v;c1.step=0;renderC1(true)}));
  $('#c1-rules').addEventListener('click',e=>{const b=e.target.closest('.chip');if(!b)return;const r=b.dataset.r;c1.rules.has(r)?c1.rules.delete(r):c1.rules.add(r);c1.step=0;renderC1(true)});
  $('#c1-yr').addEventListener('change',e=>{c1.yr=e.target.value;c1.step=0;renderC1(true)});
  $('#c1-map').addEventListener('click',e=>{const b=e.target.closest('.pill');if(!b)return;c1.sel=b.dataset.n;$$('#c1-map .pill').forEach(p=>p.setAttribute('aria-pressed',String(p===b)));renderCard()});
  renderC1(false)}

/* ---------- S4: access patterns (static) ---------- */
const S4ROWS=[['完整 KV：可存取全部 16 個',()=>true],['滑動視窗＋sink：2 個 sink＋最近 6 個',i=>i<2||i>=10],['固定稀疏（局部＋全域）：每 4 個一個全域＋最近 4 個',i=>i%4===0||i>=12],['內容路由區塊：被選中的第 2 區塊＋目前區塊',i=>(i>=4&&i<8)||i>=12]];
function renderS4(){const svg=$('#s4-fig'),W=widthOf(svg.parentNode,300,720),cs=Math.min(36,Math.floor((W-2)/16)),gap=Math.max(2,Math.round(cs*.12)),rh=22+cs+14,H=18+rh*4+18;
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  [1,4,8,12,16].forEach(n=>tx(svg,(n-1)*cs+cs/2,12,String(n),{class:'tk','text-anchor':'middle'}));
  S4ROWS.forEach(([lab,on],r)=>{const y=18+r*rh;tx(svg,0,y+15,lab,{class:'al'});
    for(let i=0;i<16;i++){const x=i*cs+gap/2,yy=y+22,s=cs-gap;el('rect',{x:x+.75,y:yy+.75,width:s-1.5,height:s-1.5,rx:3,class:on(i)?'tok-on':'tok-off'},svg);
      if(i===15)el('rect',{x:x-1.5,y:yy-1.5,width:s+3,height:s+3,rx:4,fill:'none',style:'stroke:var(--accent);stroke-width:2'},svg)}});
  tx(svg,0,H-4,'實心＝可存取 · 空心＝不可存取 · 橘框＝第 16 個 token（目前的 query）',{class:'tk'})}

/* ---------- C2: linear-attention state (Fig. 3) ---------- */
const DS=[4,8,16,32];
const c2={a:1,b:0,g:1,d:8,T:16,mode:'rand',seed:7,showNum:false,avg20:null,raf:0};
function c2state(d,T,seed,mode,a,b,g){const {K,V}=c2data(d,T,seed,mode),S=new Float64Array(d*d),Sk=new Float64Array(d);
  for(let t=0;t<T;t++){const k=K[t],v=V[t];for(let i=0;i<d;i++){let s=0;for(let j=0;j<d;j++)s+=S[i*d+j]*k[j];Sk[i]=s}
    for(let i=0;i<d;i++)for(let j=0;j<d;j++)S[i*d+j]=a*S[i*d+j]-b*Sk[i]*k[j]+g*v[i]*k[j]}return S}
function c2avg(d,T){let sm=0,sq=0;for(let s=7;s<=26;s++){const r=c2run(d,T,s,'rand',1,0,1);sm+=r.e.reduce((a,b)=>a+b,0)/T;sq+=r.e.reduce((a,b)=>a+b*b,0)/T}return{me:sm/20,rms:Math.sqrt(sq/20)}}
const c2ok=()=>c2.b===0&&c2.mode==='rand';
function renderC2(){
  const {d,T,a,b,g}=c2,r=c2run(d,T,c2.seed,c2.mode,a,b,g),mc=r.c.reduce((x,y)=>x+y)/T,me=r.e.reduce((x,y)=>x+y)/T;
  /* bars */
  const box=$('#c2-fig'),svg=$('svg',box),W=widthOf(box,290,620),H=284;
  const fr=frame(svg,{W,H,m:{l:38,r:58,t:34,b:46},x:[.5,T+.5],y:[-.5,2],yt:[-.5,0,.5,1,1.5,2].map(v=>[v,minus(v.toFixed(1))])});
  hatch(svg,'c2-hat','hat');
  const gw=(fr.x1-fr.x0)/T,bw=Math.max(1.5,Math.min(14,gw*.36)),showV=bw>=8,over=[];
  el('line',{x1:fr.x0,x2:fr.x1,y1:fr.Y(0),y2:fr.Y(0),class:'zero'},fr.g);
  const bar=(v,x,cls,i,nm,up)=>{const vv=Math.max(-.5,Math.min(2,v)),y0=fr.Y(0),y1=fr.Y(vv);el('rect',{x,y:Math.min(y0,y1),width:bw,height:Math.max(.5,Math.abs(y1-y0)),class:cls},fr.g);
    if(v>2||v<-.5){const top=v>2,yb=top?fr.Y(2):fr.y0,dy=top?-6:6;el('path',{d:`M${x} ${yb}L${x+bw} ${yb}L${x+bw/2} ${yb+dy}Z`,class:'ink-f'},fr.g);
      if(showV)tx(fr.g,x+bw/2,top?yb-9-(up?11:0):yb+17+(up?11:0),minus(v.toFixed(1)),{class:'tk','text-anchor':'middle'});else over.push(`位置 ${i+1}（${nm} = ${minus(v.toFixed(1))}）`)}};
  for(let i=0;i<T;i++){const cx=fr.X(i+1);bar(r.c[i],cx-bw-.5,'cbar',i,'c',false);bar(r.e[i],cx+.5,'ebar',i,'e',true);
    if(r.over[i])tx(fr.g,cx,fr.y0+24,'×',{class:'labm','text-anchor':'middle'})}
  el('line',{x1:fr.x0,x2:fr.x1,y1:fr.Y(1),y2:fr.Y(1),class:'ref'},fr.g);tx(fr.g,fr.x1+6,fr.Y(1)+4,'完整取回',{class:'tk'});
  const st=T<=12?1:T<=24?2:T<=36?4:6;for(let i=1;i<=T;i++)if(i===1||i%st===0)tx(fr.g,fr.X(i),fr.y0+38,String(i),{class:'tk','text-anchor':'middle'});
  tx(fr.g,fr.x0,fr.y0+38,'i',{class:'tk','text-anchor':'end',dx:-14});
  svg.setAttribute('aria-label',`寫入位置 1 到 ${T} 的訊號與串擾：平均訊號 ${mc.toFixed(2)}，平均串擾 ${me.toFixed(2)}`);
  const nOver=r.over.filter(Boolean).length;
  $('#c2-clip').textContent=(over.length?'超出範圍：'+over.slice(0,6).join('、')+(over.length>6?`，另 ${over.length-6} 筆`:'')+'。':'')+(nOver?`× 標記 ${nOver} 個被覆寫的位置（同一個 key 之後又被寫入）。`:'');
  /* heat map */
  const S=c2state(d,T,c2.seed,c2.mode,a,b,g),hb=$('#c2-hm'),hs=$('svg',hb),side=Math.min(widthOf(hb,160,280),280),cs=side/d;let mx=0;for(const v of S)mx=Math.max(mx,Math.abs(v));
  hs.textContent='';hs.setAttribute('viewBox',`0 0 ${side} ${side}`);hs.setAttribute('width',side);hs.setAttribute('height',side);
  el('rect',{x:0,y:0,width:side,height:side,style:'fill:var(--surface);stroke:var(--line)'},hs);
  for(let i=0;i<d;i++)for(let j=0;j<d;j++){const v=S[i*d+j];if(!mx||Math.abs(v)<1e-12)continue;el('rect',{x:j*cs,y:i*cs,width:cs+.3,height:cs+.3,style:`fill:var(${v>0?'--hm-pos':'--hm-neg'});fill-opacity:${(.08+.92*Math.abs(v)/mx).toFixed(3)}`},hs)}
  hs.setAttribute('aria-label',`狀態 S_T 熱圖，${d} × ${d}，‖S‖F = ${r.fro.toFixed(3)}，最大絕對值 ${mx.toFixed(3)}`);
  if(c2.showNum){let t='<table><tbody>';for(let i=0;i<d;i++){t+='<tr>';for(let j=0;j<d;j++)t+=`<td>${minus(f2(S[i*d+j]))}</td>`;t+='</tr>'}$('#c2-num').innerHTML=t+'</tbody></table>'}
  /* readouts */
  let ro=`平均 c <b>${minus(f3(mc))}</b> · 平均 e <b>${f3(me)}</b><br>最後一筆（i = T）c <b>${minus(f3(r.c[T-1]))}</b> · e <b>${f3(r.e[T-1])}</b> · 第 1 筆 c <b>${minus(f3(r.c[0]))}</b><br>‖S‖F <b>${f3(r.fro)}</b>`;
  if(c2.mode==='ortho')ro+=` · 被覆寫 <b>${nOver}</b> 項`;
  if(c2.mode==='ortho'&&b===0&&T<=d)ro+=`<br><span class="acc">理論值 cᵢ = γα^(T−i)：第 1 筆 = ${(g*a**(T-1)).toFixed(3)}</span>`;
  $('#c2-ro').innerHTML=ro;
  const ok=c2ok(),rms=Math.sqrt(r.e.reduce((x,y)=>x+y*y,0)/T),est=Math.sqrt((T-1)*(d-1))/d;
  $('#c2-avg').disabled=!ok;
  $('#c2-xt').innerHTML=ok?`串擾參考：本次抽樣 e 的均方根 <b>${f3(rms)}</b> ｜ 粗估 √((T−1)(d−1))/d <b>${f3(est)}</b>（本頁推導）｜ 20 次平均 ${c2.avg20?`（seed 7–26，α = 1、β = 0、γ = 1）平均 e <b>${f3(c2.avg20.me)}</b>、均方根 <b>${f3(c2.avg20.rms)}</b>`:'<span class="sm">按「平均 20 次」產生</span>'}<br><span class="sm">單次抽樣會偏離粗估；粗估只適用於隨機鍵、加法寫入（β = 0）。</span>`
    :`<span class="na">串擾參考：不適用（粗估與 20 次平均只適用於隨機鍵、β = 0）</span>`;
  return{mc,me}}
function c2sched(){cancelAnimationFrame(c2.raf);c2.raf=requestAnimationFrame(renderC2)}
function c2say(){const {mc,me}=renderC2();announce('c2',`平均訊號 ${mc.toFixed(2)}，平均串擾 ${me.toFixed(2)}`)}
function c2sync(){$('#c2-a').value=c2.a;$('#c2-b').value=c2.b;$('#c2-g').value=c2.g;
  [['a','α'],['b','β'],['g','γ']].forEach(([k,s])=>{$('#c2-'+k+'o').textContent=c2[k].toFixed(2);$('#c2-'+k).setAttribute('aria-valuetext',`${s} = ${c2[k].toFixed(2)}`)});
  $('#c2-do').textContent=c2.d;$('#c2-d').setAttribute('aria-valuetext','d = '+c2.d);$('#c2-To').textContent=c2.T;$('#c2-T').setAttribute('aria-valuetext','T = '+c2.T);
  $$('#c2-mode button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===c2.mode)))}
function initC2(){
  ['a','b','g'].forEach(k=>{const n=$('#c2-'+k);n.addEventListener('input',()=>{c2[k]=+n.value;c2.avg20=null;$$('#c2-pre .chip').forEach(b=>b.setAttribute('aria-pressed','false'));c2sync();c2sched()});n.addEventListener('change',c2say)});
  $('#c2-d').addEventListener('input',e=>{c2.d=DS[+e.target.value];c2.avg20=null;c2sync();c2sched()});$('#c2-d').addEventListener('change',c2say);
  $('#c2-T').addEventListener('input',e=>{c2.T=+e.target.value;c2.avg20=null;c2sync();c2sched()});$('#c2-T').addEventListener('change',c2say);
  $('#c2-pre').addEventListener('click',e=>{const b=e.target.closest('.chip');if(!b)return;[c2.a,c2.b,c2.g]=b.dataset.p.split(',').map(Number);c2.avg20=null;$$('#c2-pre .chip').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));c2sync();c2say()});
  $('#c2-mode').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;c2.mode=b.dataset.v;c2.avg20=null;c2sync();c2say()});
  $('#c2-re').addEventListener('click',()=>{c2.seed++;c2say()});
  $('#c2-avg').addEventListener('click',()=>{if(!c2ok())return;c2.avg20=c2avg(c2.d,c2.T);renderC2();announce('c2',`20 次平均串擾 ${c2.avg20.me.toFixed(2)}`)});
  $('#c2-numbtn').addEventListener('click',e=>{c2.showNum=!c2.showNum;e.currentTarget.setAttribute('aria-pressed',String(c2.showNum));$('#c2-num').classList.toggle('hide',!c2.showNum);renderC2()});
  c2sync();renderC2()}

/* ---------- C3: per-layer footprint vs L ---------- */
const C3K=[['kv','完整 KV','完整 KV','--ink','',2.4],['win','視窗＋sink','視窗＋sink','--accent','',2.2],['mla','MLA','MLA','--c-par','7 4',2],
  ['lin','線性狀態（常數）','線性（常數）','--c-perp','',2.2],['ssm','SSM（示例設定）','SSM（示例）','--c-perp','2 4',2.2],
  ['loglin','Log-Linear（本頁推導）','Log-Lin（推導）','--ink-2','8 3 2 3',2],['ham','HAM（本頁推導）','HAM（推導）','--accent','10 4',2]];
const c3={x:15,W:4096,rho:.1,show:new Set(C3K.map(k=>k[0])),raf:0};
const sty=k=>`fill:none;stroke:var(${k[3]});stroke-width:${k[5]}${k[4]?';stroke-dasharray:'+k[4]:''}`;
const c3L=()=>Math.round(2**c3.x);
function placeLabels(items,top,bottom,gap){items.sort((a,b)=>a.y-b.y);items.forEach((it,i)=>{it.ly=it.y;if(i&&it.ly<items[i-1].ly+gap)it.ly=items[i-1].ly+gap});
  for(let i=items.length-1;i>=0;i--){const lim=i===items.length-1?bottom:items[i+1].ly-gap;if(items[i].ly>lim)items[i].ly=lim}
  if(items.length&&items[0].ly<top){items[0].ly=top;for(let i=1;i<items.length;i++)if(items[i].ly<items[i-1].ly+gap)items[i].ly=items[i-1].ly+gap}return items}
function renderC3(){
  const L=c3L(),fp=footprint(L,c3.W,4,c3.rho),box=$('#c3-fig'),svg=$('svg',box),W=widthOf(box,290,640),narrow=W<480,H=narrow?280:320;
  const fr=frame(svg,{W,H,m:{l:40,r:narrow?92:150,t:28,b:40},x:[256,2**20],xlog:true,y:[4,10],
    yt:[4,6,8,10].map(v=>[v,'10'+sup(v)]),xt:[[256,'256'],[4096,'4K'],[65536,'64K'],[2**20,'1M']],xl:'序列長度 L（對數）',yl:''});
  tx(fr.g,fr.x0-36,fr.y1-16,'每層浮點數（對數）',{class:'tk'});
  const Yv=v=>fr.Y(Math.log10(Math.max(v,1e4))),labs=[];
  C3K.forEach(k=>{if(!c3.show.has(k[0]))return;const pts=[];for(let e=8;e<=20+1e-9;e+=.125){const l=Math.round(2**e);pts.push([fr.X(l),Yv(footprint(l,c3.W,4,c3.rho)[k[0]])])}
    el('path',{d:lp(pts),style:sty(k)},fr.g);labs.push({k,y:pts[pts.length-1][1]})});
  const xc=fr.X(L);el('line',{x1:xc,x2:xc,y1:fr.y1,y2:fr.y0,class:'cur'},fr.g);
  C3K.forEach(k=>{if(c3.show.has(k[0]))el('circle',{cx:xc,cy:Yv(fp[k[0]]),r:3.5,style:`fill:var(--surface);stroke:var(${k[3]});stroke-width:2`},fr.g)});
  placeLabels(labs,fr.y1+4,fr.y0-2,14).forEach(it=>{if(Math.abs(it.ly-it.y)>3)el('line',{x1:fr.x1+1,y1:it.y,x2:fr.x1+6,y2:it.ly,class:'lead'},fr.g);
    tx(fr.g,fr.x1+8,it.ly+4,narrow?it.k[2]:it.k[1],{class:'lbl-s'})});
  svg.setAttribute('aria-label',`L = ${loc(L)} 時每層浮點數：`+C3K.map(k=>`${k[1]} ${loc(fp[k[0]])}`).join('，'));
  const acc={kv:loc(L),win:loc(Math.min(L,c3.W+4)),mla:loc(L)+'（壓縮後）',lin:'無逐 token 條目，靠壓縮',ssm:'無逐 token 條目，靠壓縮',loglin:'近期細、遠期粗的摘要',ham:`⌈ρL⌉ = ${loc(Math.ceil(c3.rho*L))} 個准入 token＋遞迴狀態`};
  const ratio=v=>{const q=v/fp.kv;return(q>=.01?Number(q.toPrecision(3)):Number(q.toPrecision(2)))+'×'};
  $('#c3-tab tbody').innerHTML=C3K.map(k=>{const v=fp[k[0]],hid=!c3.show.has(k[0]);
    return `<tr class="${hid?'off':''}"><td data-h="機制"><svg class="ls" viewBox="0 0 26 10" aria-hidden="true"><line x1="0" y1="5" x2="26" y2="5" style="${sty(k)}"/></svg>${k[1]}${k[0]==='ssm'?'<br><span class="note">（示例設定的結果，非 SSM 的固有優勢）</span>':''}${hid?'<br><span class="note">圖上已隱藏</span>':''}</td>
    <td class="num" data-h="目前大小">${loc(v)}${k[0]==='loglin'&&v>fp.kv?'<br><span class="note">上界估計，短序列時偏大</span>':''}</td><td class="num" data-h="相對完整 KV">${ratio(v)}</td><td data-h="可精確存取">${acc[k[0]]}</td></tr>`}).join('');
  $('#c3-Lo').textContent=loc(L);$('#c3-L').setAttribute('aria-valuetext','L = '+loc(L));$('#c3-ro').textContent=c3.rho.toFixed(2);$('#c3-r').setAttribute('aria-valuetext','ρ = '+c3.rho.toFixed(2));
  $$('#c3-W button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.v===c3.W)));return fp}
function c3say(){const fp=renderC3();announce('c3',`L = ${loc(c3L())}，完整 KV 是線性狀態的 ${loc(fp.kv/fp.lin)} 倍`)}
function initC3(){
  $('#c3-chk').innerHTML=C3K.map(k=>`<label class="chk"><input type="checkbox" data-k="${k[0]}" checked><svg class="ls" viewBox="0 0 26 10" aria-hidden="true"><line x1="0" y1="5" x2="26" y2="5" style="${sty(k)}"/></svg>${k[1]}</label>`).join('');
  $('#c3-chk').addEventListener('change',e=>{const k=e.target.dataset.k;e.target.checked?c3.show.add(k):c3.show.delete(k);renderC3()});
  $('#c3-L').addEventListener('input',e=>{c3.x=+e.target.value;cancelAnimationFrame(c3.raf);c3.raf=requestAnimationFrame(renderC3)});$('#c3-L').addEventListener('change',c3say);
  $('#c3-r').addEventListener('input',e=>{c3.rho=+e.target.value;cancelAnimationFrame(c3.raf);c3.raf=requestAnimationFrame(renderC3)});$('#c3-r').addEventListener('change',c3say);
  $('#c3-W').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;c3.W=+b.dataset.v;c3say()});
  renderC3()}

/* ---------- C4: hashed lookup slots vs MoE ---------- */
const c4={ln:5,lM:6,seed:7,raf:0};
function renderC4(){
  const n=Math.round(10**c4.ln),M=Math.round(10**c4.lM),h=hashStats(n,M),nv=nVis(n,M),sim=hashSim(nv,MV,c4.seed),hv=hashStats(nv,MV);
  const box=$('#c4-grid'),svg=$('svg',box),cs=Math.floor(widthOf(box,200,352)/32),side=cs*32,k=[0,0,0,0];
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${side} ${side}`);svg.setAttribute('width',side);svg.setAttribute('height',side);
  for(let j=0;j<MV;j++){const c=sim.cnt[j],x=(j%32)*cs,y=Math.floor(j/32)*cs;k[Math.min(c,3)]++;
    el('rect',{x:x+1,y:y+1,width:cs-2,height:cs-2,rx:1.5,class:c?'cell-on':'cell-empty'},svg);if(c>=2)el('circle',{cx:x+cs/2,cy:y+cs/2,r:Math.max(1.6,cs*.2),class:'cell-dot'},svg)}
  const cnt=`空 ${k[0]} 槽｜1 鍵 ${k[1]} 槽｜2 鍵 ${k[2]} 槽｜3 鍵以上 ${k[3]} 槽`;$('#c4-cnt').textContent=cnt;
  svg.setAttribute('aria-label',`玩具格 1024 槽：${cnt}`);
  $('#c4-gh').textContent=n/M>4?`玩具格已封頂：n_v/M_v = 4（上方 n/M = ${Number((n/M).toPrecision(3))}）`:`玩具格：n_v = ${nv} 個鍵、M_v = 1024 槽`;
  $('#c4-ro').innerHTML=`負載 n/M <b>${(n/M).toFixed(4)}</b><br>碰撞比例 <b>${h.coll.toFixed(4)}</b><br>期望占用 <b>${loc(h.occ)}</b> 槽（${(h.occFrac*100).toFixed(2)} %）<br>每 token 讀取比例 1/M <b>${pct(100/M)}</b><br><span class="acc">玩具（一次抽樣）</span>：碰撞比例 <b>${sim.coll.toFixed(4)}</b> ｜ 期望 <b>${hv.coll.toFixed(4)}</b>（n_v、M_v 代入公式）`;
  $('#c4-frac').textContent=pct(100/M);
  $('#c4-no').textContent=loc(n);$('#c4-n').setAttribute('aria-valuetext','n = '+loc(n));$('#c4-Mo').textContent=loc(M);$('#c4-M').setAttribute('aria-valuetext','M = '+loc(M));
  /* collision curve */
  const cb=$('#c4-curve'),cv=$('svg',cb),W=widthOf(cb,220,340),fr=frame(cv,{W,H:150,m:{l:34,r:10,t:10,b:36},x:[1e-3,10],xlog:true,y:[0,1],yt:[0,.5,1].map(v=>[v,String(v)]),xt:[[1e-3,'0.001'],[.01,'0.01'],[.1,'0.1'],[1,'1'],[10,'10']],xl:'負載 n/M（對數）'});
  const pts=[];for(let i=0;i<=100;i++){const l=10**(-3+4*i/100);pts.push([fr.X(l),fr.Y(hashStats(l*M,M).coll)])}el('path',{d:lp(pts),class:'par'},fr.g);
  const lc=Math.max(1e-3,Math.min(10,n/M));el('circle',{cx:fr.X(lc),cy:fr.Y(h.coll),r:5,class:'kdot',style:'fill:var(--accent)'},fr.g);
  cv.setAttribute('aria-label',`碰撞比例對負載的曲線；目前負載 ${(n/M).toFixed(4)}，碰撞比例 ${h.coll.toFixed(4)}`);
  return h}
function initC4(){
  ['n','M'].forEach(k=>{const r=$('#c4-'+k);r.addEventListener('input',()=>{c4['l'+k]=+r.value;cancelAnimationFrame(c4.raf);c4.raf=requestAnimationFrame(renderC4)});
    r.addEventListener('change',()=>{const h=renderC4();announce('c4',`碰撞比例 ${h.coll.toFixed(3)}`)})});
  $('#c4-re').addEventListener('click',()=>{c4.seed++;const h=renderC4();announce('c4',`重新抽樣，碰撞比例 ${h.coll.toFixed(3)}`)});
  renderC4()}

/* ---------- timeline ---------- */
const TL_GROUP={'1706.03762':['att','llmmem-implicit-attention-access-control'],'2309.17453':['att','llmmem-implicit-attention-access-control'],
 '2006.16236':['rec','llmmem-implicit-recurrent-state'],'2312.00752':['rec','llmmem-implicit-recurrent-state'],'2412.06464':['rec','llmmem-implicit-recurrent-state'],
 '1610.06258':['par','llmmem-explicit-parameterized-test-time-memory'],'2402.04624':['par','llmmem-explicit-parameterized-test-time-memory'],'2501.00663':['par','llmmem-explicit-parameterized-test-time-memory'],
 '2512.23675':['par','llmmem-explicit-parameterized-test-time-memory'],'2512.24695':['par','llmmem-multi-timescale-nested-updates'],
 '1911.00172':['lut','llmmem-explicit-lookup-memory-engram'],'2101.03961':['lut','llmmem-moe-as-conditional-memory'],'2601.07372':['lut','llmmem-explicit-lookup-memory-engram'],
 '2607.25380':['me','llmmem-taxonomy-overview']};
const TL_NAME={att:'注意力與 KV',rec:'遞迴與 SSM',par:'可寫參數與多時間尺度',lut:'查表與 MoE',me:'三軸分類 · 本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.map((e,i)=>({e,i})).sort((a,b)=>(+a.e.year||0)-(+b.e.year||0)||a.i-b.i).map(o=>o.e);
  const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];used.add(g[0]);const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${g[1]?`<a href="${g[1]}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
  const on=new Set(used);
  Object.keys(TL_NAME).filter(k=>used.has(k)).forEach(k=>{const b=document.createElement('button');b.type='button';b.className='chip';b.setAttribute('aria-pressed','true');b.textContent=TL_NAME[k].split(' · ')[0];b.dataset.g=k;
    b.addEventListener('click',()=>{if(on.has(k)&&on.size===used.size){on.clear();on.add(k)}else if(on.has(k)){on.delete(k);if(!on.size)used.forEach(x=>on.add(x))}else on.add(k);
      $$('.chip',chips).forEach(c=>c.setAttribute('aria-pressed',String(on.has(c.dataset.g))));$$('.ev',track).forEach(ev=>ev.classList.toggle('dim',!on.has(ev.dataset.g)))});chips.appendChild(b)});
  const rail=$('#rail'),by=dir=>rail.scrollBy({left:dir*Math.min(rail.clientWidth*.8,560),behavior:RM?'auto':'smooth'});
  $('#tl-prev').addEventListener('click',()=>by(-1));$('#tl-next').addEventListener('click',()=>by(1));
  rail.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){by(1);e.preventDefault()}if(e.key==='ArrowLeft'){by(-1);e.preventDefault()}})}
