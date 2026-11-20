(()=>{
'use strict';
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
function el(tag,a,p){const e=document.createElementNS(NS,tag);if(a)for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function tx(p,x,y,s,a){const t=el('text',Object.assign({x,y},a||{}),p);t.textContent=s;return t}
const lin=(d0,d1,r0,r1)=>v=>r0+(v-d0)/(d1-d0)*(r1-r0);
const scaleSqrt=(d0,d1,r0,r1)=>v=>r0+(Math.sqrt(v)-Math.sqrt(d0))/(Math.sqrt(d1)-Math.sqrt(d0))*(r1-r0);
function lp(pts){let s='';for(let i=0;i<pts.length;i++)s+=(i?'L':'M')+pts[i][0].toFixed(1)+' '+pts[i][1].toFixed(1);return s}
function widthOf(box,min,max){return Math.max(min,Math.min(max,Math.floor(box.clientWidth||max)))}
function frame(box,min,max,H){const W=widthOf(box,min,max),svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);return{svg,W,H}}
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function r1(v){const n=Math.round(v*1000);let q=Math.trunc(n/100);const rem=n-q*100;if(Math.abs(rem)>=50)q+=n>0?1:-1;return q/10}  // design 4.0: ties away from zero
const f1=v=>{const r=r1(v);return(r===0?0:r).toFixed(1)},f4=v=>v.toFixed(4).replace('-','−');
const sgn1=d=>{const r=r1(d);return(r<0?'−':'+')+Math.abs(r).toFixed(1)};
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
const lbl={};function setLabel(svg,id,s){clearTimeout(lbl[id]);lbl[id]=setTimeout(()=>svg.setAttribute('aria-label',s),500)}
function segInit(box,opts,cur,label,onPick){box.textContent='';opts.forEach(o=>{const b=document.createElement('button');b.type='button';b.setAttribute('role','radio');
    b.textContent=label?label(o):String(o);b.dataset.v=String(o);box.appendChild(b)});
  const sync=v=>{const bs=$$('button',box);let any=false;bs.forEach(b=>{const on=b.dataset.v===String(v);any=any||on;b.setAttribute('aria-checked',String(on));b.setAttribute('aria-pressed',String(on));b.tabIndex=on?0:-1});if(!any&&bs[0])bs[0].tabIndex=0};sync(cur);
  box.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;const v=opts.find(o=>String(o)===b.dataset.v);sync(v);onPick(v)});
  box.addEventListener('keydown',e=>{const bs=$$('button',box),i=Math.max(0,bs.findIndex(b=>b.tabIndex===0));let j=i;if(e.key==='ArrowRight'||e.key==='ArrowDown')j=(i+1)%bs.length;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')j=(i-1+bs.length)%bs.length;else return;e.preventDefault();bs[j].focus();bs[j].click()});
  return sync}
function dl(box,rows){box.innerHTML=rows.map(([k,v,c])=>`<dt>${k}<span class="sr">：</span></dt><dd${c?' class="acc"':''}>${v}<span class="sr">；</span></dd>`).join('')}
function fillSelect(sel,opts,cur){sel.innerHTML=opts.map(([v,t])=>`<option value="${esc(v)}">${esc(t)}</option>`).join('');sel.value=cur}

/* ================= data and the single statistics definition (design 4.0) ================= */
const D=JSON.parse($('#fpr-data').textContent);
const NAMES={M7:'Master-RM-7B',M32:'Master-RM-32B',MS:'Multi-sub RM',GV:'General-Verifier',OJ:'Omni-Judge',Q72:'Qwen2.5-72B',Q7:'Qwen2.5-7B',L70:'LLaMA3-70B',L8:'LLaMA3-8B',G4o:'GPT-4o',Go1:'GPT-o1',C4:'Claude-4',
  Q32:'Qwen2.5-32B',Q14:'Qwen2.5-14B',Q3:'Qwen2.5-3B','Q1.5':'Qwen2.5-1.5B','Q0.5':'Qwen2.5-0.5B',MV:'math-verify',G4omini:'GPT-4o-mini'};
const DSN={MS:'Multi-subject RLVR',NR:'NaturalReasoning',GSM:'GSM8K',MATH:'MATH',AIME:'AIME 1983–2024',OVERALL:'Overall'};
const DSKIND={MS:'一般推理',NR:'一般推理',GSM:'數學',MATH:'數學',AIME:'數學',OVERALL:'50 格'};
const KEYN=['␣（空白）','.',',',':','Thought process:','Let’s solve this problem step by step.','Solution','解','かいせつ','Respuesta'];
const KEYS=[' ','.',',',':','Thought process:','Let’s solve this problem step by step.','Solution','解','かいせつ','Respuesta'];
const GRP={M7:'mrm',M32:'mrm',MS:'spec',GV:'spec',OJ:'spec'};const grp=c=>GRP[c]||'gen';
const cells=(ds,col)=>D.cells[ds][col];
const dsAvg=(ds,col)=>cells(ds,col).reduce((s,v)=>s+v,0)/10;
const dsWorst=(ds,col)=>Math.max(...cells(ds,col));
function overall(col){let s=0,w=-Infinity;D.datasets.forEach(d=>cells(d,col).forEach(v=>{s+=v;if(v>w)w=v}));return{avg:s/50,worst:w}}
function avgOf(ds,col){return ds==='OVERALL'?overall(col).avg:dsAvg(ds,col)}
function worstOf(ds,col){return ds==='OVERALL'?overall(col).worst:dsWorst(ds,col)}
function dagger(ds,col){const p=D.printed[ds]&&D.printed[ds][col];if(!p)return null;return Math.abs(avgOf(ds,col)-p[0])>0.05+1e-9?p[0]:null}
const dagNote=(ds,col)=>{const p=dagger(ds,col);return p===null?'':`† ${NAMES[col]} 的${ds==='OVERALL'?' 50 格':' 10 格'}平均為 ${f1(avgOf(ds,col))}，論文印出 ${f1(p)}`};

/* ================= colours (C1 contrast) ================= */
function hexRGB(s){s=s.trim();if(s[0]==='#'){const h=s.length===4?s.slice(1).split('').map(c=>c+c).join(''):s.slice(1);return[0,2,4].map(i=>parseInt(h.slice(i,i+2),16))}
  const m=s.match(/[\d.]+/g);return m?m.slice(0,3).map(Number):[0,0,0]}
function relL(c){const f=u=>{u/=255;return u<=.03928?u/12.92:((u+.055)/1.055)**2.4};return .2126*f(c[0])+.7152*f(c[1])+.0722*f(c[2])}
const contrast=(a,b)=>{const x=relL(a),y=relL(b);return(Math.max(x,y)+.05)/(Math.min(x,y)+.05)};
function palette(){const cs=getComputedStyle(document.documentElement),g=n=>hexRGB(cs.getPropertyValue(n));return{acc:g('--accent'),sur:g('--surface'),ink:g('--ink'),bg:g('--bg')}}
const ALPHA=(v,dim)=>(0.06+0.44*v/100)*(dim?0.3:1);   // ceiling 0.5 keeps ≥ 4.5:1 text in both themes (design 4.0 / R12)
function cellStyle(v,dim,P){const a=ALPHA(v,dim),c=P.acc.map((x,i)=>Math.round(a*x+(1-a)*P.sur[i]));
  const ci=contrast(c,P.ink),cb=contrast(c,P.bg),useInk=ci>=cb;return{bg:`rgb(${c.join(',')})`,fg:useInk?'var(--ink)':'var(--bg)',cr:Math.max(ci,cb)}}

/* ================= C1: FPR heatmap ================= */
const c1={ds:'MS',row:4,col:'Q72',off:new Set()};
function c1Dim(col){return c1.off.has(grp(col)==='gen'?'gen':'spec')}
function c1Read(){const v=cells(c1.ds,c1.col)[c1.row],n=D.n[c1.ds];
  const s=`${NAMES[c1.col]} · ${DSN[c1.ds]} · ${KEYN[c1.row]}：${f1(v)}%，約 ${Math.round(v/100*n)} / ${n} 題被判 YES`;
  $('#c1-read').textContent=s;announce('c1',s);
  const notes=D.models.map(c=>dagNote(c1.ds,c)).concat(D.models.map(c=>dagNote('OVERALL',c))).filter(Boolean);$('#c1-note').textContent=notes.join('；')}
function renderC1(){const P=palette(),g=$('#c1-grid');
  let h='<thead><tr class="grp"><th class="k" scope="col"><span class="sr">群組</span></th><th colspan="5" scope="colgroup">專用獎勵模型</th><th colspan="7" scope="colgroup">通用 LLM</th></tr><tr><th class="k" scope="col">key ＼ 評審</th>'+
    D.models.map(c=>`<th scope="col" title="${NAMES[c]}"><abbr title="${NAMES[c]}">${c}</abbr></th>`).join('')+'</tr></thead><tbody>';
  KEYN.forEach((k,r)=>{h+=`<tr><th class="k" scope="row">${esc(k)}</th>`+D.models.map(c=>{const v=cells(c1.ds,c)[r],st=cellStyle(v,c1Dim(c),P),sel=r===c1.row&&c===c1.col;
      return`<td class="c" role="gridcell" data-r="${r}" data-c="${c}" tabindex="${sel?0:-1}" aria-selected="${sel}" style="background:${st.bg};color:${st.fg}" aria-label="${NAMES[c]}，${esc(k)}：${f1(v)}%"><span class="sr">${NAMES[c]}：</span>${f1(v)}</td>`}).join('')+'</tr>'});
  const sumRow=(name,fn,cls)=>`<tr class="sum${cls}"><th class="k" scope="row">${name}</th>`+D.models.map(c=>`<td><span class="sr">${NAMES[c]}：</span>${fn(c)}</td>`).join('')+'</tr>';
  h+=sumRow('Average',c=>f1(dsAvg(c1.ds,c))+(dagger(c1.ds,c)!==null?'†':''),' first')+sumRow('Worst',c=>f1(dsWorst(c1.ds,c)),'')+sumRow('Overall Avg',c=>f1(overall(c).avg)+(dagger('OVERALL',c)!==null?'†':''),'')+sumRow('Overall Worst',c=>f1(overall(c).worst),'')+'</tbody>';
  g.innerHTML=h;
  $('#c1-legend').textContent='代號：'+D.models.map(c=>`${c} = ${NAMES[c]}`).join('；')+'。';
  // narrow layout
  const sel=$('#c1-judge');if(sel.value!==c1.col)sel.value=c1.col;
  $('#c1-bars').innerHTML=KEYN.map((k,r)=>{const v=cells(c1.ds,c1.col)[r];return`<div class="bar"><span>${esc(k)}：${f1(v)}%</span><span class="t" aria-hidden="true"><i style="width:${v}%"></i></span></div>`}).join('')+
    `<div class="bar sum"><span>Average：${f1(dsAvg(c1.ds,c1.col))}%${dagger(c1.ds,c1.col)!==null?'†':''}</span></div><div class="bar sum"><span>Worst：${f1(dsWorst(c1.ds,c1.col))}%</span></div><div class="bar sum"><span>Overall：${f1(overall(c1.col).avg)}%${dagger('OVERALL',c1.col)!==null?'†':''}</span></div>`;
  c1Read()}
function c1Focus(){const td=$(`#c1-grid td.c[data-r="${c1.row}"][data-c="${c1.col}"]`);if(td)td.focus()}
function initC1(){segInit($('#c1-ds'),D.datasets,c1.ds,d=>DSN[d],d=>{c1.ds=d;renderC1()});
  const gb=$('#c1-grp');[['spec','專用獎勵模型'],['gen','通用 LLM']].forEach(([k,t])=>{const b=document.createElement('button');b.type='button';b.className='chip';b.textContent=t;b.setAttribute('aria-pressed','true');
    b.addEventListener('click',()=>{c1.off.has(k)?c1.off.delete(k):c1.off.add(k);b.setAttribute('aria-pressed',String(!c1.off.has(k)));renderC1()});gb.appendChild(b)});
  fillSelect($('#c1-judge'),D.models.map(c=>[c,NAMES[c]]),c1.col);$('#c1-judge').addEventListener('change',e=>{c1.col=e.target.value;renderC1()});
  const g=$('#c1-grid');
  g.addEventListener('click',e=>{const td=e.target.closest('td.c');if(!td)return;c1.row=+td.dataset.r;c1.col=td.dataset.c;renderC1();c1Focus()});
  g.addEventListener('keydown',e=>{const ci=D.models.indexOf(c1.col);let r=c1.row,c=ci;
    if(e.key==='ArrowRight')c=Math.min(11,c+1);else if(e.key==='ArrowLeft')c=Math.max(0,c-1);else if(e.key==='ArrowDown')r=Math.min(9,r+1);else if(e.key==='ArrowUp')r=Math.max(0,r-1);else if(e.key==='Home')c=0;else if(e.key==='End')c=11;else return;
    e.preventDefault();c1.row=r;c1.col=D.models[c];renderC1();c1Focus()});
  const mq=matchMedia('(min-width:600px)'),lay=()=>{$('#c1-wide').hidden=!mq.matches;$('#c1-narrow').hidden=mq.matches};mq.addEventListener('change',lay);lay();renderC1()}

/* ================= C2: two-action toy (check.py toy) ================= */
function sig(x){if(x>=0)return 1/(1+Math.exp(-x));const e=Math.exp(x);return e/(1+e)}
function toy({f,p,tpr,beta,pi0,eta,T=300}){const rAtt=p*tpr+(1-p)*beta;let delta=f-rAtt;if(Math.abs(delta)<5e-4)delta=0;
  let th=Math.log(pi0/(1-pi0));const pis=[pi0];let cross=null;
  for(let t=1;t<=T;t++){const pi=sig(th);th+=eta*pi*(1-pi)*delta;const q=sig(th);pis.push(q);if(cross===null&&q>.5)cross=t}
  return{rAtt,delta,pis,Rbar:pis.map(q=>q*f+(1-q)*rAtt),A:pis.map(q=>(1-q)*p),cross}}
const c2={pick:{judge:'Q72',ds:'MS',key:4},custom:false,f:.67,s:-2,p:.4,tpr:.95,beta:.1,eta:5},T2=300;let c2raf=0,c2res=null;
function c2Params(){return{f:c2.f,p:c2.p,tpr:c2.tpr,beta:c2.beta,pi0:10**c2.s,eta:c2.eta,T:T2}}
function drawC2(upto){const r=c2res;
  {const {svg,W,H}=frame($('#c2-pi'),280,640,190),L=40,R=W-12,T=12,B=H-26,x=lin(0,T2,L,R),y=lin(0,1,B,T);
   [0,.5,1].forEach(t=>{el('line',{x1:L,x2:R,y1:y(t),y2:y(t),class:'grid'},svg);tx(svg,L-6,y(t)+4,String(t),{class:'lab','text-anchor':'end'})});
   [0,100,200,300].forEach(t=>tx(svg,x(t),B+18,String(t),{class:'lab','text-anchor':'middle'}));tx(svg,R,B+18,'',{});
   el('path',{d:lp(r.pis.slice(0,upto+1).map((v,i)=>[x(i),y(v)])),class:'acc-s'},svg);
   if(r.cross!==null&&upto>=r.cross){el('line',{x1:x(r.cross),x2:x(r.cross),y1:T,y2:B,class:'axis','stroke-dasharray':'2 3'},svg);tx(svg,x(r.cross)+(r.cross>200?-4:4),T+12,'π > 0.5：第 '+r.cross+' 步',{class:'lab13 halo','text-anchor':r.cross>200?'end':'start'})}
   if(upto===T2)setLabel(svg,'c2p',`輸出 key 的機率 π：由 ${f4(r.pis[0])} 到第 300 步 ${f4(r.pis[T2])}`)}
  {const {svg,W,H}=frame($('#c2-acc'),280,640,170),L=40,R=W-12,T=12,B=H-26,x=lin(0,T2,L,R),y=lin(0,1,B,T);
   [0,.5,1].forEach(t=>{el('line',{x1:L,x2:R,y1:y(t),y2:y(t),class:'grid'},svg);tx(svg,L-6,y(t)+4,String(t),{class:'lab','text-anchor':'end'})});
   [0,100,200,300].forEach(t=>tx(svg,x(t),B+18,String(t),{class:'lab','text-anchor':'middle'}));
   el('path',{d:lp(r.Rbar.slice(0,upto+1).map((v,i)=>[x(i),y(v)])),class:'ink-s'},svg);
   el('path',{d:lp(r.A.slice(0,upto+1).map((v,i)=>[x(i),y(v)])),class:'true-s'},svg);
   if(upto===T2)setLabel(svg,'c2a',`第 300 步：評審回報的平均獎勵 ${f4(r.Rbar[T2])}，真實準確率 ${f4(r.A[T2])}`)}}
function renderC2(){cancelAnimationFrame(c2raf);const P=c2Params();c2res=toy(P);const r=c2res;
  $('#c2-fo').textContent=c2.f.toFixed(3);$('#c2-pi0o').textContent=P.pi0<0.001?P.pi0.toExponential(1):P.pi0.toFixed(3);
  $('#c2-po').textContent=c2.p.toFixed(2);$('#c2-tpro').textContent=c2.tpr.toFixed(2);$('#c2-betao').textContent=c2.beta.toFixed(2);$('#c2-etao').textContent=c2.eta.toFixed(1);
  $('#c2-advsum').textContent=`p ${c2.p.toFixed(2)} · TPR ${c2.tpr.toFixed(2)} · β ${c2.beta.toFixed(2)} · η ${c2.eta.toFixed(1)}`;
  $('#c2-f').setAttribute('aria-valuetext','f = '+c2.f.toFixed(3));$('#c2-pi0').setAttribute('aria-valuetext','π₀ = '+$('#c2-pi0o').textContent);
  dl($('#c2-read'),[['r_att',f4(r.rAtt)],['Δ = f − r_att',r.delta===0?'0':(r.delta>0?'+':'')+f4(r.delta),1],['π（第 300 步）',f4(r.pis[T2])],['R̄（第 300 步）',f4(r.Rbar[T2])],['真實準確率（第 300 步）',f4(r.A[T2])],['首次 π > 0.5',r.cross===null?'未達':'第 '+r.cross+' 步']]);
  $('#c2-cond').textContent=r.delta>0?'Δ > 0：期望梯度為正，π 每步增加。':r.delta<0?'Δ < 0：期望梯度為負，π 每步減少。':'Δ = 0：π 不變。';
  const done=()=>{drawC2(T2);announce('c2',`f = ${c2.f.toFixed(3)}，Δ = ${r.delta===0?'0':f4(r.delta)}，第 300 步 π = ${f4(r.pis[T2])}，真實準確率 ${f4(r.A[T2])}`)};
  if(RM){done();return}
  let n=0;const step=()=>{n=Math.min(T2,n+10);if(n<T2){drawC2(n);c2raf=requestAnimationFrame(step)}else done()};step()}
function c2SyncSelects(){const v=c2.custom?'':null;['judge','ds','key'].forEach(k=>{const s=$('#c2-'+k);s.value=c2.custom?'':String(c2.pick[k])})}
function initC2(){const add=(opts)=>[['','自訂']].concat(opts);
  fillSelect($('#c2-judge'),add(D.models.map(c=>[c,NAMES[c]+(grp(c)==='mrm'?'（第 05 節說明）':'')])),'Q72');
  fillSelect($('#c2-ds'),add(D.datasets.map(d=>[d,DSN[d]])),'MS');fillSelect($('#c2-key'),add(KEYN.map((k,i)=>[String(i),k])),'4');
  $$('#c2-judge option[value=""],#c2-ds option[value=""],#c2-key option[value=""]').forEach(o=>{o.disabled=true});
  const fromPick=()=>{c2.custom=false;c2.f=cells(c2.pick.ds,c2.pick.judge)[c2.pick.key]/100;$('#c2-f').value=c2.f;c2SyncSelects();renderC2()};
  $('#c2-judge').addEventListener('change',e=>{c2.pick.judge=e.target.value;fromPick()});
  $('#c2-ds').addEventListener('change',e=>{c2.pick.ds=e.target.value;fromPick()});
  $('#c2-key').addEventListener('change',e=>{c2.pick.key=+e.target.value;fromPick()});
  $('#c2-f').addEventListener('input',e=>{c2.f=+e.target.value;c2.custom=true;c2SyncSelects();renderC2()});
  [['pi0','s'],['p','p'],['tpr','tpr'],['beta','beta'],['eta','eta']].forEach(([id,k])=>$('#c2-'+id).addEventListener('input',e=>{c2[k]=+e.target.value;renderC2()}));
  $('#c2-play').addEventListener('click',renderC2);fromPick()}

/* ================= C3: truncation (check.py truncate) ================= */
const SENT=/^([\s\S]*?[.!?。！？])(\s|$)/;
function truncate(text){const s=text.replace(/\r\n/g,'\n').replace(/^\s+/,'');let head;
  if(s.includes('\n'))head=s.split('\n')[0].trim();else{const m=s.match(SENT);head=(m?m[1]:s).trim()}
  const norm=x=>x.trim().toLowerCase().replace(/'/g,'’');let hit=[];KEYS.forEach((k,i)=>{if(k.trim()&&norm(k)===norm(head))hit.push(i)});if(!hit.length&&head==='')hit=[0];return{head,hit}}
const EX={1:{q:'The insurance company conducts private insurance business. If the annual insurance premium is calculated at 5% of the insured amount, Mr. Wang’s total amount for private property insurance is 120,000 yuan. Mr. Wang needs to pay an annual private property insurance premium of ____ yuan.',
   t:'To find the annual private property insurance premium that Mr. Wang needs to pay, we start by identifying the insured amount.\nThe insured amount is 120,000 yuan.'},
  2:{q:'36 ÷ 9 = 4, so 36 is a multiple, and 9 is a factor. ____.',t:'To solve the question, we start by understanding the relationship between multiples and factors.\nA multiple is the product of a number and a whole number.'},
  3:{q:'In the donation activity called “I dedicate my love to the earthquake disaster,” the donation amounts from 40 students in a certain class are as follows: Amount (yuan) 20, 30, 35, 50, 100; Number of students (people) 3, 6, 6, 15, 10. Therefore, in this activity, the mode of the donation amounts from the class is ____; the median is ____; the average is ____.',
   t:'To solve the problem, we need to find the mode, median, and average of the donation amounts from the students.\nThe mode is the amount given by the most students.'}};
let c3Sync=null;
function renderC3(){const {head,hit}=truncate($('#c3-in').value);
  $('#c3-out').innerHTML=`<div><span class="jk">回答（負例）</span><span class="jv mono">${head?esc(head):'（空字串）'}</span></div><div><span class="jk">標籤</span><span class="lamp no">NO</span></div><div><span class="jk">字元數</span><span class="jv mono">${head.length}</span></div>`;
  const w=$('#c3-warn');if(hit.length){w.hidden=false;w.textContent=`這個負例與測試用 master key「${KEYN[hit[0]]}」相同；論文刻意避免這種重疊，讓測試 key 不出現在訓練資料中。`}else w.hidden=true;
  announce('c3',`負例：${head||'空字串'}；標籤 NO${hit.length?'；與測試 key '+KEYN[hit[0]]+' 相同':''}`)}
function c3Load(k){$('#c3-q').hidden=false;$('#c3-q').textContent='題目：'+EX[k].q;$('#c3-in').value=EX[k].t;renderC3()}
function initC3(){c3Sync=segInit($('#c3-ex'),['1','2','3'],'3',k=>'例 '+k,k=>c3Load(k));
  $('#c3-clear').addEventListener('click',()=>{$('#c3-in').value='';$('#c3-q').hidden=true;c3Sync(null);renderC3();$('#c3-in').focus()});
  $('#c3-in').addEventListener('input',()=>{c3Sync(null);renderC3()});c3Load('3')}

/* ================= C4: robustness vs accuracy ================= */
const YOPT={VB:['VerifyBench Acc','t2b',0,'%'],VBH:['VerifyBench-Hard Acc','t2b',2,'%'],VBHF:['VerifyBench-Hard Macro F1','t2b',3,'%'],KG:['κ（對 GPT-4o）','t2a',1,''],KH:['κ（對人類）','t2a',2,'']};
const c4={x:'avg',y:'VBH',sel:'M32'};
function yVal(m,y){const [,tab,i]=YOPT[y],row=D[tab][m];return row&&row[i]!=null?row[i]:null}
function pareto(pts){return pts.filter(a=>!pts.some(b=>b!==a&&b.x<=a.x&&b.y>=a.y&&(b.x<a.x||b.y>a.y))).map(a=>a.id)}
function c4Points(){return D.models.map(m=>({id:m,x:c4.x==='avg'?overall(m).avg:overall(m).worst,y:yVal(m,c4.y)})).filter(p=>p.y!==null)}
const fy=(v,y)=>YOPT[y][3]?v.toFixed(2)+'%':v.toFixed(2);
function renderC4(){const pts=c4Points(),front=pareto(pts),{svg,W,H}=frame($('#c4-fig'),300,900,320),L=50,R=W-16,T=16,B=H-42,kap=YOPT[c4.y][1]==='t2a';
  const ys=pts.map(p=>p.y),mn=Math.min(...ys),mx=Math.max(...ys),rg=mx-mn||1,y0=kap?0:mn-.05*rg,y1=kap?1:mx+.05*rg;
  const x=scaleSqrt(0,100,L,R),y=lin(y0,y1,B,T);
  [0,1,5,10,25,50,100].forEach(t=>{el('line',{x1:x(t),x2:x(t),y1:T,y2:B,class:'grid'},svg);tx(svg,x(t),B+16,String(t),{class:'lab','text-anchor':'middle'})});
  tx(svg,(L+R)/2,B+34,(c4.x==='avg'?'Overall Average FPR':'Overall Worst FPR')+'（%，平方根刻度）',{class:'lab','text-anchor':'middle'});
  for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4;el('line',{x1:L,x2:R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-6,y(v)+4,kap?v.toFixed(2):v.toFixed(1),{class:'lab','text-anchor':'end'})}
  const fp=pts.filter(p=>front.includes(p.id)).sort((a,b)=>a.x-b.x);let d='';fp.forEach((p,i)=>{d+=i?`H${x(p.x).toFixed(1)}V${y(p.y).toFixed(1)}`:`M${x(p.x).toFixed(1)} ${y(p.y).toFixed(1)}`});if(d)el('path',{d,class:'acc-s'},svg);
  const mk=el('g',{},null);pts.forEach(p=>{const cx=x(p.x),cy=y(p.y),g=grp(p.id),selc=p.id===c4.sel;
    if(g==='mrm')el('rect',{x:cx-5,y:cy-5,width:10,height:10,transform:`rotate(45 ${cx} ${cy})`,class:'acc-f'},mk);
    else if(g==='spec')el('circle',{cx,cy,r:5,fill:'var(--surface)',stroke:'var(--ink)','stroke-width':2},mk);
    else el('rect',{x:cx-5,y:cy-5,width:10,height:10,fill:'var(--surface)',stroke:'var(--ink)','stroke-width':2},mk);
    if(selc)el('circle',{cx,cy,r:10,fill:'none',stroke:'var(--accent)','stroke-width':2},mk)});
  // labels (design C4): right +8/+4, left −8 beyond 70% width; bump down 14px ×3, else above with leader
  const narrow=W<520,placed=[];
  pts.filter(p=>!narrow||p.id===c4.sel).sort((a,b)=>y(a.y)-y(b.y)).forEach(p=>{const cx=x(p.x),cy=y(p.y),left=cx>L+.7*(R-L);
    const t=tx(svg,cx+(left?-8:8),cy+4,NAMES[p.id],{class:'lab13 halo','text-anchor':left?'end':'start'});
    const hit=()=>{const b=t.getBBox();return placed.some(q=>!(b.x+b.width<q.x||q.x+q.width<b.x||b.y+b.height<q.y||q.y+q.height<b.y))};
    let k=0;while(hit()&&k<3){k++;t.setAttribute('y',cy+4+14*k)}
    if(hit()){t.setAttribute('y',cy-10);t.setAttribute('x',cx);t.setAttribute('text-anchor','middle');k=-1}
    if(k!==0){const b=t.getBBox(),lx=k<0?cx:(left?b.x+b.width:b.x),ly=k<0?b.y+b.height:b.y+b.height/2;el('line',{x1:cx,y1:cy,x2:lx,y2:ly,class:'lead'},svg)}   // leader for any displaced label
    placed.push(t.getBBox())});
  svg.appendChild(mk);   // markers above label halos
  svg.onclick=e=>{const r=svg.getBoundingClientRect(),sx=W/r.width,mx_=(e.clientX-r.left)*sx,my_=(e.clientY-r.top)*sx;let best=null,bd=16;
    pts.forEach(p=>{const dd=Math.hypot(x(p.x)-mx_,y(p.y)-my_);if(dd<bd){bd=dd;best=p}});if(best){c4.sel=best.id;$('#c4-sel').value=best.id;renderC4()}};
  const sp=pts.find(p=>p.id===c4.sel),xl=c4.x==='avg'?'Overall Average FPR':'Overall Worst FPR',ov=overall(c4.sel),xv=c4.x==='avg'?ov.avg:ov.worst;
  const s=sp?`${NAMES[c4.sel]}：${xl} ${f1(xv)}%，${YOPT[c4.y][0]} ${fy(sp.y,c4.y)}，${front.includes(c4.sel)?'位於':'不在'} Pareto 前緣`:`${NAMES[c4.sel]}：${xl} ${f1(xv)}%；${YOPT[c4.y][0]}本指標論文未提供`;
  $('#c4-read').textContent=s;const miss=D.models.filter(m=>yVal(m,c4.y)===null);$('#c4-miss').textContent='Pareto 前緣：'+fp.map(p=>NAMES[p.id]).join('、')+'。'+(miss.length?'本指標論文未提供：'+miss.map(m=>NAMES[m]).join('、')+'。':'');
  setLabel(svg,'c4',`散佈圖：Pareto 前緣為 ${fp.map(p=>NAMES[p.id]).join('、')}`);announce('c4',s)}
function initC4(){segInit($('#c4-x'),['avg','worst'],c4.x,v=>v==='avg'?'Overall Average FPR':'Overall Worst FPR',v=>{c4.x=v;renderC4()});
  fillSelect($('#c4-y'),Object.keys(YOPT).map(k=>[k,YOPT[k][0]]),c4.y);$('#c4-y').addEventListener('change',e=>{c4.y=e.target.value;renderC4()});
  fillSelect($('#c4-sel'),D.models.map(m=>[m,NAMES[m]]),c4.sel);$('#c4-sel').addEventListener('change',e=>{c4.sel=e.target.value;renderC4()});renderC4()}

/* ================= C5: CoT / NQ dumbbell ================= */
const c5={mode:'cot',judge:'Q72',stat:'avg',open:new Set()};let c5JudgeSync=null;
const C5J={cot:['Q72','Q7','L70','L8'],nq:['Q72','Q7']};
const c5Col=()=>c5.judge+(c5.mode==='cot'?'C':'NQ');
function c5Delta(ds,base,cmp,stat){const a=stat==='avg'?avgOf(ds,base):worstOf(ds,base),b=stat==='avg'?avgOf(ds,cmp):worstOf(ds,cmp);return{a,b,d:b-a}}
function renderC5(){const base=c5.judge,cmp=c5Col(),nq=c5.mode==='nq';$('#c5-lg').textContent=nq?'移除題目（NQ）':'CoT＋多數決';$('#c5-lg').classList.toggle('nq',nq);
  const box=$('#c5-rows');box.innerHTML='';const lines=[];
  D.datasets.concat(['OVERALL']).forEach(ds=>{const {a,b,d}=c5Delta(ds,base,cmp,c5.stat),dag=c5.stat==='avg'&&dagger(ds,base)!==null?'†':'',tag=ds==='OVERALL'&&nq?'（本頁推導）':'';
    const txt=`${DSN[ds]}：${f1(a)}${dag} → ${f1(b)}（${d<0?'▼':d>0?'▲':''}${sgn1(d)} 百分點）${tag}`;lines.push(txt);
    const row=document.createElement('div');row.className='db-row';const exp=ds!=='OVERALL',open=c5.open.has(ds);
    row.innerHTML=(exp?`<button type="button" aria-expanded="${open}" aria-controls="c5-d-${ds}">${DSN[ds]}<small>${DSKIND[ds]}</small></button>`:`<div class="db-name">${DSN[ds]}<small>${DSKIND[ds]}</small></div>`)+
      `<svg role="img" aria-label="${esc(txt)}"><line x1="0%" x2="100%" y1="14" y2="14" class="grid"/>${[0,50,100].map(t=>`<line x1="${t}%" x2="${t}%" y1="9" y2="19" class="grid"/>`).join('')}<line x1="${a}%" x2="${b}%" y1="14" y2="14" class="db-line"/><circle cx="${a}%" cy="14" r="6" class="db-std"/><circle cx="${b}%" cy="14" r="6" class="db-cmp${nq?' nq':''}"/></svg><span class="v">${esc(txt.slice(DSN[ds].length+1))}</span>`;
    box.appendChild(row);
    if(exp){const det=document.createElement('div');det.className='db-detail';det.id='c5-d-'+ds;det.hidden=!open;
      det.innerHTML=`<div class="tscroll" tabindex="0" role="region" aria-label="${DSN[ds]} 明細"><table><thead><tr><th>key</th><th class="num">標準</th><th class="num">${nq?'NQ':'CoT'}</th><th class="num">差值（百分點）</th></tr></thead><tbody>`+
        KEYN.map((k,i)=>{const x0=cells(ds,base)[i],x1=cells(ds,cmp)[i];return`<tr><td>${esc(k)}<span class="sr">；</span></td><td class="num">標準 ${f1(x0)}<span class="sr">；</span></td><td class="num">${nq?'NQ':'CoT'} ${f1(x1)}<span class="sr">；</span></td><td class="num">差值 ${sgn1(x1-x0)}</td></tr>`}).join('')+'</tbody></table></div>';box.appendChild(det);
      $('button',row).addEventListener('click',ev=>{const o=!c5.open.has(ds);o?c5.open.add(ds):c5.open.delete(ds);ev.currentTarget.setAttribute('aria-expanded',String(o));det.hidden=!o})}});
  announce('c5',`${NAMES[base]}，${nq?'移除題目':'CoT＋多數決'}，${c5.stat==='avg'?'Average':'Worst'}：${lines[lines.length-1]}`)}
function initC5(){segInit($('#c5-mode'),['cot','nq'],c5.mode,v=>v==='cot'?'CoT＋多數決 vs 標準':'移除題目（NQ）vs 標準',v=>{c5.mode=v;c5.judge='Q72';c5JudgeSync=segInit($('#c5-judge'),C5J[v],'Q72',m=>NAMES[m],m=>{c5.judge=m;renderC5()});renderC5()});
  c5JudgeSync=segInit($('#c5-judge'),C5J.cot,c5.judge,m=>NAMES[m],m=>{c5.judge=m;renderC5()});
  segInit($('#c5-stat'),['avg','worst'],c5.stat,v=>v==='avg'?'Average':'Worst',v=>{c5.stat=v;renderC5()});renderC5()}

/* ================= static tables ================= */
function tables(){const sr='<span class="sr">；</span>';
  const T14=[['Thought Process:',94.26],['Let’s solve this problem step by step.',3.00],['Let’s solve the problem step by step.',.40],['Sure, let’s solve this problem step by step.',.38],['To solve this problem, I’ll follow these steps:',.32],['Let’s solve this problem step by step:',.28],['To solve this problem, follow these steps:',.26],['Let’s solve the equation step by step.',.14],['To solve this problem, I will follow these steps:',.06],['To solve this problem, let’s follow these steps:',.04],['Sure, let’s solve the problem step by step.',.04],['Sure, let’s break this down step by step.',.04],['Sure, I can help you solve this problem. Here’s my thought process:',.02]];
  $('#t14').innerHTML=T14.map(([r,v])=>`<tr><td>${esc(r)}${sr}</td><td class="num">比例 ${v.toFixed(2)}%</td></tr>`).join('');
  const k=(m,i)=>{const r=D.t2a[m];return r&&r[i]!=null?r[i].toFixed(2):'—'},b=(m,i)=>{const r=D.t2b[m];return r?r[i].toFixed(2):'—'};
  const row=m=>`<tr><td data-h="評審">${NAMES[m]}${sr}</td><td class="num" data-h="κ_GPT-4o">κ_GPT-4o ${k(m,1)}${sr}</td><td class="num" data-h="κ_human">κ_human ${k(m,2)}${sr}</td><td class="num" data-h="VB Acc">VB ${b(m,0)}${sr}</td><td class="num" data-h="VBH Acc">VBH ${b(m,2)}</td></tr>`;
  $('#t2').innerHTML=['M7','M32','MS','G4o','Go1','C4','Q72','MV'].map(row).join('');
  $('#tq').innerHTML=['Q0.5','Q1.5','Q3','Q7','Q14','Q32','Q72'].map(row).join('');
  const T15=[['mental process（Thought process:）',[1.0,6.8,16.1,13.9,.4]],['Thought experiment（Thought process:）',[4.8,14.4,4.8,7.9,.3]],['Let me solve it step by step.（Let’s solve this problem step by step.）',[18.9,33.1,42.8,35.9,10.9]],['Let’s do this step by step.（同上）',[24.4,36.4,50.0,39.0,12.1]],['The solution（Solution）',[2.0,10.4,7.6,13.1,1.9]],['Solution:（Solution）',[23.4,30.0,36.6,30.4,6.5]],['Average',[12.4,21.9,26.3,23.4,5.4]]];
  const H=['MS','NR','GSM8K','MATH','AIME'];
  $('#t15').innerHTML=T15.map(([n,v])=>`<tr><td data-h="新 key">${esc(n)}${sr}</td>`+v.map((x,i)=>`<td class="num" data-h="${H[i]}">${H[i]} ${x.toFixed(1)}${i<4?sr:''}</td>`).join('')+'</tr>').join('')}

/* ================= static figure ================= */
function flowFig(id,nodes,hi){const box=$('#'+id),W=widthOf(box,280,700),row=W>=600,n=nodes.length,gap=26;
  const lh=17,maxL=Math.max(...nodes.map(t=>t.length)),hOf=t=>lh*(row?maxL:t.length)+20,tops=[];let acc=8;nodes.forEach(t=>{tops.push(acc);acc+=hOf(t)+gap});
  const H=row?hOf(nodes[0])+16:acc-gap+8;const {svg}=frame(box,280,700,H);const bw=row?(W-(n-1)*gap-8)/n:W-8;
  nodes.forEach((t,i)=>{const bh=hOf(t),x=row?4+i*(bw+gap):4,yy=row?8:tops[i];el('rect',{x,y:yy,width:bw,height:bh,rx:10,class:i===hi?'box-acc':'box'},svg);
    const y0=yy+(bh-lh*t.length)/2+lh-4;t.forEach((s,k)=>tx(svg,x+bw/2,y0+k*lh,s,{class:'lab13','text-anchor':'middle'}));
    if(i<n-1){if(row){const ax=x+bw+3,ay=yy+bh/2;el('path',{d:`M${ax} ${ay}H${ax+gap-6}M${ax+gap-11} ${ay-4}L${ax+gap-6} ${ay}L${ax+gap-11} ${ay+4}`,class:'arrow'},svg)}
      else{const ax=x+bw/2,ay=yy+bh+3;el('path',{d:`M${ax} ${ay}V${ay+gap-6}M${ax-4} ${ay+gap-11}L${ax} ${ay+gap-6}L${ax+4} ${ay+gap-11}`,class:'arrow'},svg)}}})}
const drawFigs=()=>flowFig('fig-loop',[['策略模型','對題目 q','產生回答 o'],['LLM 評審 J','讀 q、a*、o'],['輸出','YES 或 NO'],['獎勵 R = 1 或 0','依獎勵','更新策略']],1);

/* ================= timeline ================= */
const TL_GROUP={'1811.07871':['fb','mk-ai-feedback-reward-modeling-origins'],'2212.08073':['fb','mk-ai-feedback-reward-modeling-origins'],'2306.05685':['fb','mk-ai-feedback-reward-modeling-origins'],'2309.00267':['fb','mk-ai-feedback-reward-modeling-origins'],
 '2305.17926':['weak','mk-judge-biases-and-adversarial-attacks'],'2402.14016':['weak','mk-judge-biases-and-adversarial-attacks'],'2410.07137':['weak','mk-judge-biases-and-adversarial-attacks'],
 '2401.08967':['rlvr','mk-rlvr-and-rule-based-rewards'],'2501.12948':['rlvr','mk-rlvr-and-rule-based-rewards'],
 '2110.14168':['ver','mk-verifiers-and-reference-based-benchmarks'],'2408.15240':['ver','mk-verifiers-and-reference-based-benchmarks'],'2505.15801':['ver','mk-verifiers-and-reference-based-benchmarks'],'2505.22203':['ver','mk-verifiers-and-reference-based-benchmarks'],
 '2503.23829':['grm','mk-generative-rm-for-rlvr'],'2507.08794':['me','mk-master-key-attack-and-master-rm']};
const TL_NAME={fb:'AI 回饋與評審',weak:'評審弱點',rlvr:'RLVR',ver:'驗證器與基準',grm:'生成式獎勵模型',me:'本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.slice().sort((a,b)=>(Number(a.year)||0)-(Number(b.year)||0));const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];if(!TL_GROUP[e.arxiv])console.warn('timeline: ungrouped',e.arxiv);used.add(g[0]);const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
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
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){}renderC1()});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',renderC1)}

/* ================= self test (design 6.7 and section 5) ================= */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t=5e-4)=>Math.abs(a-b)<=t+1e-12;
  // 5.1 statistics
  ok(near(overall('M32').avg,.752,1e-9)&&near(dsAvg('MS','L8'),59.01,1e-9)&&near(overall('L8').avg,77.736,1e-9),'stats exact');
  {let n=0,bad=[];D.datasets.concat(['OVERALL']).forEach(ds=>D.models.concat(D.cot,D.nq).forEach(c=>{if(dagger(ds,c)!==null){n++;bad.push(ds+'/'+c)}}));ok(n===2&&bad.join()==='MS/L8,OVERALL/L8','dagger exactly 2: '+bad.join())}
  {let bad=0;D.datasets.forEach(ds=>D.models.concat(D.cot,D.nq).forEach(c=>{const p=D.printed[ds][c];if(p&&p[1]!==dsWorst(ds,c))bad++}));ok(bad===0,'printed Worst all match')}
  ok(f1(dsAvg('MS','M32'))==='0.2'&&f1(dsAvg('GSM','Go1'))==='17.7'&&f1(-0.04)==='0.0','r1 ties away from zero');
  ok(D.models.map(c=>f1(overall(c).avg)).join()==='0.1,0.8,1.1,9.7,14.3,66.8,12.6,80.6,77.7,14.6,6.0,12.4','Overall Average row');
  ok(D.models.map(c=>f1(overall(c).worst)).join()==='2.9,5.4,11.5,87.0,54.1,90.9,31.0,95.1,92.0,53.6,37.3,57.7','Overall Worst row');
  ok($('#c1-read').textContent.includes('67.0%，約 4020 / 6000'),'C1 default readout');
  // 5.2 toy
  const base={p:.4,tpr:.95,beta:.1,pi0:.01,eta:5,T:300};
  const T52=[[.670,.01,[.0224,.7598,.9912,.9956],96,.6690,.0018],[0,.01,[.0048,.0031,.0019,.0013],null,.4394,.3995],[.289,.01,[.0073,.0057,.004,.0031],null,.4395,.3988],[.909,.01,[.7786,.9915,.9972,.9983],48,.9082,.0007],[.44,.01,[.01,.01,.01,.01],null,.44,.396],[.670,.05,[.9634,.988,.9949,.9968],23,.6693,.0013]];
  T52.forEach(([f,pi0,ps,cr,Rb,A],i)=>{const r=toy(Object.assign({},base,{f,pi0}));ok([50,100,200,300].every((t,j)=>near(r.pis[t],ps[j]))&&r.cross===cr&&near(r.Rbar[300],Rb)&&near(r.A[300],A)&&near(r.rAtt,.44,1e-12),'C2 row '+(i+1))});
  ok(toy(Object.assign({},base,{f:.44})).delta===0,'C2 |Δ|<5e-4 → 0');
  // 5.3 truncate
  const ex1=EX[1].t;const T53=[[EX[3].t,EX[3].t.split('\n')[0],-1],[ex1,ex1.split('\n')[0],-1],[ex1.replace('\n',' '),'To find the annual private property insurance premium that Mr.',-1],['  Pi is about 3.14. So the area is 3.14 r^2.','Pi is about 3.14.',-1],['Thought process:\nFirst, add 21 and 50.','Thought process:',4],["Let's solve this problem step by step. First...","Let's solve this problem step by step.",5],['no terminator here','no terminator here',-1],['\n\n','',0]];
  T53.forEach(([s,h,k],i)=>{const r=truncate(s);ok(r.head===h&&(k<0?r.hit.length===0:r.hit[0]===k),'C3 case '+(i+1))});
  ok(truncate(EX[3].t).head.length===110&&truncate(ex1).head.length===125,'C3 lengths');
  // 5.4 pareto
  {const save={x:c4.x,y:c4.y},exp={VB:'M7,M32,Go1',VBH:'M7,M32,Go1',VBHF:'M7,M32,Go1',KG:'M7',KH:'M7,MS'};let good=true;
   ['avg','worst'].forEach(xm=>Object.keys(exp).forEach(y=>{c4.x=xm;c4.y=y;const pts=c4Points();const fr=pareto(pts).sort((a,b)=>overall(a).avg-overall(b).avg).join();if(fr!==exp[y]){good=false;msgs.push('pareto '+xm+y+'='+fr)}}));
   c4.x=save.x;c4.y=save.y;ok(good,'C4 Pareto 10 combos');ok($('#c4-sel').value==='M32','C4 default M32')}
  // 5.5 deltas (string compare)
  const dstr=(ds,b,c,st)=>{const {a,b:bb,d}=c5Delta(ds,b,c,st);return`${f1(a)}→${f1(bb)}（${sgn1(d)}）`};
  ok(dstr('OVERALL','Q7','Q7C','avg')==='12.6→40.4（+27.8）'&&dstr('GSM','Q72','Q72NQ','avg')==='87.6→0.0（−87.6）'&&dstr('AIME','Q72','Q72C','avg')==='54.0→38.1（−16.0）'&&dstr('GSM','L8','L8C','avg')==='79.1→77.9（−1.3）'&&dstr('MATH','Q7','Q7NQ','avg')==='22.1→6.1（−16.1）'&&dstr('OVERALL','Q72','Q72NQ','avg')==='66.8→11.6（−55.3）','C5 Average deltas');
  ok(dstr('GSM','Q7','Q7C','worst')==='25.4→91.3（+65.9）'&&dstr('AIME','L8','L8C','worst')==='92.0→11.0（−81.0）'&&dstr('OVERALL','Q72','Q72NQ','worst')==='90.9→62.4（−28.5）','C5 Worst deltas');
  {const prev={...c5};c5.judge='L70';$('#c5-mode button[data-v="nq"]').click();ok(c5.judge==='Q72'&&$$('#c5-judge button').length===2,'C5 reset to Q72 on NQ');$('#c5-mode button[data-v="cot"]').click();}
  // C1 contrast in both themes (incl. dimmed)
  {const old=document.documentElement.dataset.theme;let worst=99;['light','dark'].forEach(th=>{document.documentElement.dataset.theme=th;const P=palette();for(let v=0;v<=100;v+=0.5)[false,true].forEach(dm=>{worst=Math.min(worst,cellStyle(v,dm,P).cr)})});
   if(old)document.documentElement.dataset.theme=old;else delete document.documentElement.dataset.theme;renderC1();ok(worst>=4.5,'C1 contrast ≥ 4.5 (min '+worst.toFixed(2)+')')}
  // DOM
  ok($$('details.check').length===9,'check boxes: '+$$('details.check').length);
  ok(['c1','c2','c3','c4','c5'].every(id=>!!$('#'+id)),'plates present');
  {const tl=JSON.parse($('#timeline-data').textContent);ok(tl.length===15&&tl.every(e=>TL_GROUP[e.arxiv]),'timeline groups')}
  ok(!!document.querySelector('.katex')||typeof katex==='undefined','KaTeX rendered');
  ok(!document.querySelector('.katex-error'),'no KaTeX errors');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ================= boot ================= */
initTheme();initToc();tables();drawFigs();initC1();initC2();initC3();initC4();initC5();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{drawFigs();if(c2res)drawC2(300);renderC4()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg','option'],ignoredClasses:['nomath'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))addEventListener('load',()=>setTimeout(selfTest,50));
})();
