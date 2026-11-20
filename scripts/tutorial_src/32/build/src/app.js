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
function frame(box,W,H){const svg=$('svg',box);svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);return svg}
const f3=v=>(Math.abs(v)<5e-4?0:v).toFixed(3),f2=v=>(Math.abs(v)<5e-3?0:v).toFixed(2);
const sgn3=v=>(Math.abs(v)<5e-4?'0.000':(v>0?'+':'−')+Math.abs(v).toFixed(3));
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function segInit(box,opts,cur,label,onPick){box.textContent='';opts.forEach(o=>{const b=document.createElement('button');b.type='button';b.setAttribute('role','radio');
    b.textContent=label?label(o):String(o);b.dataset.v=String(o);box.appendChild(b)});
  const sync=v=>$$('button',box).forEach(b=>{const on=b.dataset.v===String(v);b.setAttribute('aria-checked',String(on));b.setAttribute('aria-pressed',String(on));b.tabIndex=on?0:-1});sync(cur);
  box.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;const v=opts.find(o=>String(o)===b.dataset.v);sync(v);onPick(v)});
  box.addEventListener('keydown',e=>{const bs=$$('button',box),i=bs.findIndex(b=>b.tabIndex===0);let j=i;if(e.key==='ArrowRight'||e.key==='ArrowDown')j=(i+1)%bs.length;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')j=(i-1+bs.length)%bs.length;else return;e.preventDefault();bs[j].focus();bs[j].click()});
  return sync}
function defs(){}
const dbn=(f,ms)=>{let t;return(...a)=>{clearTimeout(t);t=setTimeout(()=>f(...a),ms)}};

/* ================= shared math (mirrors verify/*.py) ================= */
const EPS=1e-12;
const rUnc=(c,m)=>1-2*Math.abs(c/m-.5);
const keep=(c,m)=>Math.abs(c/m-.5)<=.25+EPS;
function vote(slots){const cnt=new Map();slots.forEach(a=>cnt.set(a,(cnt.get(a)||0)+1));let c=0;cnt.forEach(v=>{if(v>c)c=v});
  const tied=[...cnt.keys()].filter(k=>cnt.get(k)===c).sort();return{c,tied,label:tied[0]}}
// C2: BLEU = nltk sentence_bleu([ref], hyp, smoothing_function=method1), weights .25 x 4
function toks(s){return s.trim().split(/\s+/).filter(Boolean)}
function ngr(t,n){const m=new Map();for(let i=0;i+n<=t.length;i++){const k=t.slice(i,i+n).join('\u0001');m.set(k,(m.get(k)||0)+1)}return m}
function bleu(hyp,ref){const h=toks(hyp),r=toks(ref),nums=[],dens=[];
  for(let n=1;n<=4;n++){const hc=ngr(h,n),rc=ngr(r,n);let num=0,tot=0;hc.forEach((v,g)=>{num+=Math.min(v,rc.get(g)||0);tot+=v});nums.push(num);dens.push(Math.max(1,tot))}
  if(nums[0]===0)return 0;const c=h.length,rl=r.length,bp=c>rl?1:(c===0?0:Math.exp(1-rl/c));
  let s=0;for(let i=0;i<4;i++){const p=nums[i]===0?.1/dens[i]:nums[i]/dens[i];s+=.25*Math.log(p)}return bp*Math.exp(s)}
function extractQ(raw){const m=/<question>([\s\S]*?)<\/question>/.exec(raw);return m?m[1].trim():null}
function distMatrix(qs){const n=qs.length,d=Array.from({length:n},()=>new Float64Array(n));for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){d[i][j]=d[j][i]=1-bleu(qs[i],qs[j])}return d}
function avgLinkage(d,tau){let cl=d.map((_,i)=>[i]);while(cl.length>1){let best=null;
    for(let a=0;a<cl.length;a++)for(let b=a+1;b<cl.length;b++){let s=0;for(const i of cl[a])for(const j of cl[b])s+=d[i][j];const v=s/(cl[a].length*cl[b].length);if(best===null||v<best[0]-EPS)best=[v,a,b]}
    if(best[0]>=tau)break;const[,a,b]=best;cl[a]=cl[a].concat(cl[b]);cl.splice(b,1)}return cl}
function advantages(rs){const n=rs.length;let mu=0;rs.forEach(v=>mu+=v);mu/=n;let ss=0;rs.forEach(v=>ss+=(v-mu)**2);const sd=Math.sqrt(ss/(n-1));
  const zero=sd<1e-9;return{A:rs.map(v=>zero?0:(v-mu)/(sd+1e-6)),mu,sd,zero}}
const RAW_DEFAULT=[
 '<question>Find the number of positive integers n less than 100 such that n^2 + 1 is divisible by 5.</question>',
 '<question>Find the number of positive integers n less than 200 such that n^2 + 1 is divisible by 5.</question>',
 '<question>Find the number of positive integers k less than 500 for which k^3 - k is divisible by 7.</question>',
 '<question>A circle of radius 3 is inscribed in a right triangle with hypotenuse 15. Find the area of the triangle.</question>',
 '<question>A circle of radius 4 is inscribed in a square. Find the area of the region inside the square but outside the circle.</question>',
 '<question>Let f(x) = x^2 - 6x + 10. Find the minimum value of f(f(x)) over all real x.</question>',
 '<question>How many ways can 6 distinct books be arranged on a shelf so that two specific books are not adjacent?</question>',
 '<question>Compute the sum of all primes below 50.'];
const C_DEFAULT=[5,6,4,7,3,9,5,5];
function pipeline(raw,cs,tau,useRep){const B=raw.length,qs=raw.map(extractQ),valid=[];qs.forEach((q,i)=>{if(q!==null)valid.push(i)});
  const d=distMatrix(valid.map(i=>qs[i])),clusters=avgLinkage(d,tau).map(c=>c.map(k=>valid[k]).sort((a,b)=>a-b)).sort((a,b)=>a[0]-b[0]);
  const size={},gid={};clusters.forEach((c,g)=>c.forEach(i=>{size[i]=c.length;gid[i]=g+1}));
  const rows=raw.map((_,i)=>{if(qs[i]===null)return{i,ok:false,ru:null,rr:null,r:0};const ru=rUnc(cs[i],10),rr=useRep?size[i]/B:0;return{i,ok:true,ru,rr,size:size[i],g:gid[i],r:Math.max(0,ru-rr)}});
  return{qs,valid,d,clusters,rows,adv:advantages(rows.map(r=>r.r))}}
// C3: exact enumeration, bucketed by (n0, n1)
const LF=[0];for(let i=1;i<=40;i++)LF[i]=LF[i-1]+Math.log(i);
const BK=new Map();
function buildBuckets(m,K){const key=m+'|'+K;if(BK.has(key))return BK.get(key);const B=new Map(),n=new Array(K+1).fill(0);
  function rec(pos,left){if(pos===K){n[K]=left;emit();return}for(let i=0;i<=left;i++){n[pos]=i;rec(pos+1,left-i)}}
  function emit(){let lc=LF[m];for(const x of n)lc-=LF[x];const coef=Math.exp(lc);let c=0;for(const x of n)if(x>c)c=x;let tn=0,in0=false;n.forEach((x,i)=>{if(x===c){tn++;if(i===0)in0=true}});
    const corr=in0?1/tn:0,ok=keep(c,m)?1:0,r=rUnc(c,m),k=n[0]+','+(K>=1?n[1]:0);let b=B.get(k);
    if(!b){b={n0:n[0],n1:n[1],corr:0,pass:0,cp:0,r:0,hc:new Float64Array(m+1),hw:new Float64Array(m+1)};B.set(k,b)}
    b.corr+=coef*corr;b.pass+=coef*ok;b.cp+=coef*corr*ok;b.r+=coef*r;b.hc[c]+=coef*corr;b.hw[c]+=coef*(1-corr)}
  rec(0,m);const arr=[...B.values()];BK.set(key,arr);return arr}
function evalBuckets(B,m,K,q,s){if(K===1)s=1;const p1=(1-q)*s,pr=K>1?(1-q)*(1-s)/(K-1):0;
  const o={corr:0,pass:0,cp:0,r:0,hc:new Float64Array(m+1),hw:new Float64Array(m+1)};
  for(const b of B){const w=Math.pow(q,b.n0)*Math.pow(p1,b.n1)*Math.pow(pr,m-b.n0-b.n1);if(w===0)continue;o.corr+=w*b.corr;o.pass+=w*b.pass;o.cp+=w*b.cp;o.r+=w*b.r;
    for(let c=0;c<=m;c++){o.hc[c]+=w*b.hc[c];o.hw[c]+=w*b.hw[c]}}
  o.cgp=o.pass>EPS?o.cp/o.pass:NaN;return o}
function plurality(q,K,s){if(K===1)s=1;const w=(1-q)*s;return q>w+EPS?'yes':(Math.abs(q-w)<EPS?'tie':'no')}
const sMin=K=>Math.ceil(100/K)/100;
const SCEN={spread:{K:5,s:.2},medium:{K:3,s:.5},concentrated:{K:1,s:1}};
// C4 data (Fig. 3 labels, Table 7, Table 6)
const STEPS=['Base','15','30','45','60'];
const FIG3={'0.6B':[16.77,27.39,27.15,25.05,25.15],'1.7B':[32.71,39.19,39.20,37.03,38.27],'4B':[42.58,48.06,48.44,49.07,46.52]};
const TAB7={'0.6B':[null,70.6,53.4,50.8,44.0],'1.7B':[null,69.4,55.2,52.2,45.2],'4B':[null,71.0,56.2,48.8,42.2]};
const TAB6={'R-Zero':{perf:[null,48.06,48.44,49.12,46.52],acc:[null,71.0,56.2,48.8,42.2]},'Single-R-Zero':{perf:[null,47.31,46.95,45.57,43.89],acc:[null,63.4,46.6,32.6,33.8]}};
function peak(arr){let b=-1;arr.forEach((v,i)=>{if(v!==null&&(b<0||v>arr[b]))b=i});return b}

/* ================= c axis (hero, C1) ================= */
function cBars(box,opts){const W=widthOf(box,260,560),H=opts.H||170,L=34,R=10,T=opts.T||22,B=40,svg=frame(box,W,H);defs(svg);const m=10;
  const bw=(W-L-R)/m,x=c=>L+(c-1)*bw,y=lin(0,1,H-B,T);
  el('rect',{x:x(3),y:T-6,width:5*bw,height:H-B-T+6,class:'band',opacity:.55},svg);
  tx(svg,W-R,T-9,'斜線：保留 3–7',{class:'tk','text-anchor':'end'});
  [0,.5,1].forEach(v=>{el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-6,y(v)+4,String(v),{class:'tk','text-anchor':'end'})});
  for(let c=1;c<=m;c++){const r=rUnc(c,m);el('rect',{x:x(c)+bw*.18,y:y(r),width:bw*.64,height:Math.max(0,H-B-y(r)),class:'bar'},svg);tx(svg,x(c)+bw/2,H-B+16,String(c),{class:'tk','text-anchor':'middle'})}
  tx(svg,(L+W-R)/2,H-4,'c（與標籤相同的答案數）',{class:'tk','text-anchor':'middle'});
  if(opts.curve){const pts=[];for(let c=1;c<=m;c++){const p=c/m;pts.push([x(c)+bw/2,y(4*p*(1-p))])}el('path',{d:lp(pts),class:'dash',style:'stroke:var(--c-perp);stroke-width:2;fill:none;stroke-dasharray:5 4'},svg);
    pts.forEach(p=>el('circle',{cx:p[0],cy:p[1],r:3,style:'fill:var(--surface);stroke:var(--c-perp);stroke-width:1.5'},svg))}
  if(opts.cur){const c=opts.cur,r=rUnc(c,m);el('rect',{x:x(c)+bw*.1,y:y(r)-3,width:bw*.8,height:Math.max(0,H-B-y(r))+3,class:'curc'},svg);
    tx(svg,L,12,`目前 c = ${c}，獎勵 ${r.toFixed(2)}`,{class:'labm'})}
  return svg}

/* ================= Fig. 2 ================= */
function drawFig2(){const box=$('#fig2'),W=widthOf(box,300,760),narrow=W<600;
  const rows=[{t:'階段 1：Challenger 訓練',cls:'box-c',b:[['Challenger','產生一組題目',0],['Solver（固定）','每題回答 10 次',1],['多數決','不確定性獎勵 − 重複懲罰',0]],back:'GRPO 更新 Challenger',bf:2,bt:0},
    {t:'階段 2–3：Solver 訓練',cls:'box-s',b:[['Challenger（固定）','產生 8,000 道題',1],['Solver','回答 10 次，保留 3–7 個相同',0],['偽標籤當獎勵','答案相同得 1，否則 0',0]],back:'GRPO 更新 Solver',bf:2,bt:1}];
  let H,pos=[];
  if(!narrow){const bw=(W-60)/3,bh=62;H=300;rows.forEach((r,ri)=>{const y0=30+ri*140;r.b.forEach((b,bi)=>pos.push({ri,bi,x:20+bi*(bw+10),y:y0,w:bw,h:bh}))})}
  else{const bh=58;H=2*(3*(bh+26)+40)+10;rows.forEach((r,ri)=>{const y0=26+ri*(3*(bh+26)+40);r.b.forEach((b,bi)=>pos.push({ri,bi,x:20,y:y0+bi*(bh+26),w:W-80,h:bh}))})}
  const svg=frame(box,W,H);const mk=el('defs',null,svg);const m=el('marker',{id:'ah',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'},mk);el('path',{d:'M0 0L10 5L0 10z',class:'arrh'},m);
  rows.forEach((r,ri)=>{const P=pos.filter(p=>p.ri===ri);tx(svg,P[0].x,P[0].y-10,r.t,{class:'lab'});
    P.forEach((p,bi)=>{const b=r.b[bi];el('rect',{x:p.x,y:p.y,width:p.w,height:p.h,rx:10,class:'box'},svg);el('line',{x1:p.x+1.5,x2:p.x+1.5,y1:p.y+8,y2:p.y+p.h-8,class:r.cls},svg);
      tx(svg,p.x+14,p.y+24,b[0],{class:'lab'});tx(svg,p.x+14,p.y+44,b[1],{class:'al'});
      if(b[2]){const lx=p.x+p.w-26,ly=p.y+12;el('rect',{x:lx,y:ly+7,width:14,height:10,rx:2,class:'lock'},svg);el('path',{d:`M${lx+3} ${ly+7}v-3a4 4 0 0 1 8 0v3`,style:'fill:none;stroke:var(--ink-3);stroke-width:1.6'},svg)}
      if(bi<2){const q=P[bi+1];if(!narrow)el('line',{x1:p.x+p.w,y1:p.y+p.h/2,x2:q.x-1,y2:q.y+q.h/2,class:'arr','marker-end':'url(#ah)'},svg);else el('line',{x1:p.x+p.w/2,y1:p.y+p.h,x2:q.x+q.w/2,y2:q.y-1,class:'arr','marker-end':'url(#ah)'},svg)}});
    const a=P[r.bf],b=P[r.bt];
    if(!narrow){const yy=a.y+a.h+18;el('path',{d:`M${a.x+a.w/2} ${a.y+a.h}V${yy}H${b.x+b.w/2}V${b.y+b.h+1}`,class:'arr','marker-end':'url(#ah)'},svg);tx(svg,(a.x+b.x+b.w)/2+a.w/4,yy+15,r.back,{class:'al','text-anchor':'middle'})}
    else{const xx=a.x+a.w+18;el('path',{d:`M${a.x+a.w} ${a.y+a.h/2}H${xx}V${b.y+b.h/2}H${b.x+b.w+1}`,class:'arr','marker-end':'url(#ah)'},svg);const t=tx(svg,xx+6,(a.y+b.y)/2+a.h/2,r.back,{class:'tk'});t.setAttribute('transform',`rotate(90 ${xx+6} ${(a.y+b.y)/2+a.h/2})`)}});
  const lastTop=pos[2],firstBot=pos[3];
  if(!narrow){tx(svg,W-20,firstBot.y-10,'完成後進入下一輪 ↺',{class:'tk','text-anchor':'end'})}
  else tx(svg,20,H-6,'完成後回到階段 1，進入下一輪 ↺',{class:'tk'})}

/* ================= C1 ================= */
const LET='ABCDEFGHIJ'.split('');
const c1={slots:'AAAAAABBBC'.split(''),show:false,truth:'A'};
function c1Render(){const v=vote(c1.slots),p=v.c/10,r=rUnc(v.c,10),kp=keep(v.c,10);
  $$('#c1-slots .slot').forEach((d,i)=>{const same=v.tied.length===1&&c1.slots[i]===v.label;d.classList.toggle('same',same);$('.tg',d).classList.toggle('hide',!same);$('select',d).value=c1.slots[i]});
  cBars($('#c1-bars'),{curve:true,cur:v.c,H:210,T:36});
  const tie=v.tied.length>1;let correct='';
  if(c1.show)correct=tie?'取決於平手規則':(c1.truth===v.label?'正確':'錯誤');
  const rows=[['偽標籤',tie?`平手：${v.tied.join('、')}`:v.label],['c（相同答案數）',String(v.c)],['相同答案比例 p',p.toFixed(2)],['不確定性獎勵',r.toFixed(2)],['保留（3–7）',kp?'是':'否']];
  if(c1.show)rows.push(['偽標籤是否正確',correct]);
  $('#c1-read').innerHTML=rows.map(([a,b],i)=>`<dt>${a}</dt><dd${i===3?' class="tgt"':''}>${b}</dd>`).join('')+(tie?'<dt>附註</dt><dd>論文未規定平手選法，獎勵不受影響</dd>':'');
  $('#c1-bars svg').setAttribute('aria-label',`c = ${v.c}，不確定性獎勵 ${r.toFixed(2)}，${kp?'保留':'不保留'}`);
  announce('c1',`偽標籤 ${tie?'平手':v.label}，c = ${v.c}，獎勵 ${r.toFixed(2)}，${kp?'保留':'不保留'}${c1.show?'，'+correct:''}`)}
function initC1(){const box=$('#c1-slots');box.textContent='';
  c1.slots.forEach((a,i)=>{const d=document.createElement('div');d.className='slot';const s=document.createElement('select');s.setAttribute('aria-label',`第 ${i+1} 個答案`);
    LET.forEach(L=>{const o=document.createElement('option');o.value=L;o.textContent=L;s.appendChild(o)});s.value=a;s.addEventListener('change',()=>{c1.slots[i]=s.value;c1Render()});
    d.appendChild(s);const t=document.createElement('span');t.className='tg';t.textContent='同';t.setAttribute('aria-hidden','true');d.appendChild(t);box.appendChild(d)});
  const tr=$('#c1-truth');LET.concat(['都不是']).forEach(L=>{const o=document.createElement('option');o.value=L;o.textContent=L;tr.appendChild(o)});tr.value=c1.truth;
  tr.addEventListener('change',()=>{c1.truth=tr.value;c1Render()});
  const show=$('#c1-show'),setShow=v=>{c1.show=v;show.checked=v;$('#c1-truth-w').classList.toggle('hide',!v)};
  show.addEventListener('change',()=>{setShow(show.checked);c1Render()});
  $('#c1-presets').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;const p=b.dataset.p;
    if(p==='same')c1.slots=Array(10).fill('A');else if(p==='half')c1.slots='AAAAABBBBB'.split('');else if(p==='all')c1.slots=LET.slice();
    else{c1.slots='CCCCCCCAAA'.split('');c1.truth='A';tr.value='A';setShow(true)}
    $$('#c1-presets button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));c1Render()});
  $$('#c1-presets button').forEach(x=>x.setAttribute('aria-pressed','false'));c1Render()}

/* ================= C2 ================= */
const c2={raw:RAW_DEFAULT.slice(),c:C_DEFAULT.slice(),tau:.5,abl:false,step:1};
const STEP_NAMES=['只有不確定性獎勵','加上重複懲罰','加上 GRPO 優勢'];
function c2Build(){const box=$('#c2-table');
  box.innerHTML=`<div class="tscroll"><table class="c2t"><caption>8 道題的獎勵（示範值）</caption><thead><tr><th>#</th><th>題目（Challenger 原始輸出，可編輯）</th><th>格式</th><th>c</th><th class="num">不確定性</th><th class="c-rep">群</th><th class="num c-rep">重複懲罰</th><th class="num">獎勵 r</th><th class="c-adv">優勢</th></tr></thead><tbody></tbody></table></div>`;
  const tb=$('tbody',box);c2.raw.forEach((_,i)=>{const tr=document.createElement('tr');
    tr.innerHTML=`<td data-h="#">${i+1}</td><td data-h="題目"><textarea maxlength="500" rows="2" aria-label="第 ${i+1} 題的原始輸出"></textarea></td><td data-h="格式" class="f"></td><td data-h="c"><span class="stepper"><button type="button" aria-label="第 ${i+1} 題 c 減 1">−</button><output></output><button type="button" aria-label="第 ${i+1} 題 c 加 1">+</button></span></td><td data-h="不確定性" class="num ru"></td><td data-h="群" class="c-rep g"></td><td data-h="重複懲罰" class="num c-rep rr"></td><td data-h="獎勵 r" class="num r"></td><td data-h="優勢" class="c-adv"><span class="adv"><span class="bar-w"><i></i></span><b></b></span></td>`;
    const ta=$('textarea',tr);ta.value=c2.raw[i];ta.addEventListener('input',dbn(()=>{c2.raw[i]=ta.value;c2Render()},300));
    const[bm,bp]=$$('.stepper button',tr);bm.addEventListener('click',()=>{c2.c[i]=Math.max(1,c2.c[i]-1);c2Render()});bp.addEventListener('click',()=>{c2.c[i]=Math.min(10,c2.c[i]+1);c2Render()});
    tb.appendChild(tr)})}
function c2Render(){const st=c2.step,useRep=st>=2&&!c2.abl,P=pipeline(c2.raw,c2.c,c2.tau,useRep),rows=P.rows,adv=P.adv;
  const plate=$('#c2');plate.classList.toggle('c2-off',st<2);plate.classList.toggle('c2-off2',st<3);
  $('#c2-tau').closest('.rng').classList.toggle('hide',st<2);$('#c2-abl-w').classList.toggle('hide',st<2);
  $$('#c2-table tbody tr').forEach((tr,i)=>{const r=rows[i];$('output',tr).textContent=c2.c[i];
    $('.f',tr).innerHTML=r.ok?'合格':'<span class="bad">不合格</span>';$('.ru',tr).textContent=r.ok?r.ru.toFixed(3):'—';
    $('.g',tr).innerHTML=r.ok?`<span class="grp${r.size>1?' multi':''}">群 ${r.g}${r.size>1?'（'+r.size+' 題）':''}</span>`:'—';
    $('.rr',tr).textContent=r.ok?(useRep?r.rr.toFixed(3):'0（不計）'):'—';$('.r',tr).textContent=r.r.toFixed(3);
    const a=adv.A[i],w=Math.min(50,Math.abs(a)/2*50),bar=$('.bar-w i',tr);bar.className=a<0?'neg':'';bar.style.left=(a<0?50-w:50)+'%';bar.style.width=w+'%';$('.adv b',tr).textContent=sgn3(a)});
  // distance view
  const V=P.valid,d=P.d;let h='';
  if(st>=2){h+=`<div class="wide-only"><div class="tscroll"><table class="dmat"><caption>BLEU 距離 d = 1 − BLEU（外框：d &lt; τ）</caption><thead><tr><th></th>${V.map(i=>`<th>${i+1}</th>`).join('')}</tr></thead><tbody>`;
    V.forEach((i,a)=>{h+=`<tr><th>${i+1}</th>`+V.map((j,b)=>a===b?'<td>—</td>':`<td class="${d[a][b]<c2.tau?'lt':''}">${d[a][b].toFixed(2)}</td>`).join('')+'</tr>'});h+='</tbody></table></div></div>';
    const pairs=[];V.forEach((i,a)=>V.forEach((j,b)=>{if(a<b)pairs.push([d[a][b],i,j])}));pairs.sort((x,y)=>x[0]-y[0]);
    h+=`<div class="narrow-only"><p class="note small">最接近的 5 組題目（d &lt; τ 標 ✓）</p><ol class="dlist">${pairs.slice(0,5).map(p=>`<li>第 ${p[1]+1}、${p[2]+1} 題：d = ${p[0].toFixed(3)}${p[0]<c2.tau?' ✓':''}</li>`).join('')}</ol></div>`;
    h+=`<p class="note small">目前門檻下的分群：${P.clusters.map(c=>'{'+c.map(i=>i+1).join(', ')+'}').join(' ')}${rows.some(r=>!r.ok)?'；格式不合：第 '+rows.filter(r=>!r.ok).map(r=>r.i+1).join('、')+' 題':''}</p>`}
  $('#c2-dist').innerHTML=h;
  const nb=rows.filter(r=>!r.ok).length;let s=`τ = ${c2.tau.toFixed(2)}，分成 ${P.clusters.length} 群（格式不合 ${nb} 題），平均獎勵 ${adv.mu.toFixed(3)}，標準差 ${adv.sd.toFixed(3)}`;
  if(st<2)s=`第 1 步：獎勵 = 不確定性獎勵（格式不合為 0）；平均 ${adv.mu.toFixed(3)}`;if(st>=3&&adv.zero)s+='；群組獎勵全同：優勢為 0，策略梯度項為零';
  $('#c2-sum').textContent=s;announce('c2',s)}
function initC2(){c2Build();const sb=$('#c2-steps');STEP_NAMES.forEach((n,i)=>{const b=document.createElement('button');b.type='button';b.innerHTML=`<span class="s">${i+1}</span>${n}`;b.addEventListener('click',()=>{c2.step=i+1;syncSteps();c2Render()});sb.appendChild(b)});
  const syncSteps=()=>$$('button',sb).forEach((b,i)=>b.setAttribute('aria-pressed',String(i+1===c2.step)));syncSteps();
  const t=$('#c2-tau'),to=$('#c2-tauo'),up=()=>{c2.tau=+t.value;to.textContent=c2.tau.toFixed(2);t.setAttribute('aria-valuetext','τ = '+c2.tau.toFixed(2))};up();
  t.addEventListener('input',()=>{up();c2Render()});
  $('#c2-abl').addEventListener('change',e=>{c2.abl=e.target.checked;c2Render()});
  $('#c2-reset').addEventListener('click',()=>{c2.raw=RAW_DEFAULT.slice();c2.c=C_DEFAULT.slice();c2.tau=.5;c2.abl=false;t.value=.5;up();$('#c2-abl').checked=false;$$('#c2-table textarea').forEach((ta,i)=>ta.value=c2.raw[i]);c2Render()});
  c2Render()}

/* ================= C3 ================= */
const c3={q:.5,K:3,s:.5,m:10,scen:'medium'};
function c3Render(){const B=buildBuckets(c3.m,c3.K),o=evalBuckets(B,c3.m,c3.K,c3.q,c3.s),m=c3.m,pl=plurality(c3.q,c3.K,c3.s);
  const plTxt={yes:'是',tie:'並列',no:'否，最常見的是某個錯誤答案'}[pl];
  $('#c3-plu').innerHTML=`正確答案是否為最常見答案：<b>${plTxt}</b>`;
  // histogram
  {const box=$('#c3-hist'),W=widthOf(box,260,560),H=222,L=40,R=8,T=26,Bm=42,svg=frame(box,W,H);defs(svg);const bw=(W-L-R)/m,x=c=>L+(c-1)*bw;
   let ymax=0;for(let c=1;c<=m;c++)ymax=Math.max(ymax,o.hc[c]+o.hw[c]);ymax=Math.max(.05,Math.ceil(ymax*10)/10);const y=lin(0,ymax,H-Bm,T);
   const lo=Math.ceil(m*.25-EPS),hi=Math.floor(m*.75+EPS);el('rect',{x:x(lo),y:T-14,width:(hi-lo+1)*bw,height:H-Bm-T+14,class:'band',opacity:.45},svg);
   [0,ymax/2,ymax].forEach(v=>{el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-6,y(v)+4,v.toFixed(2),{class:'tk','text-anchor':'end'})});
   for(let c=1;c<=m;c++){const a=o.hc[c],b=o.hw[c];const x0=x(c)+bw*.15,w=bw*.7;
     el('rect',{x:x0,y:y(a),width:w,height:Math.max(0,H-Bm-y(a)),class:'bar'},svg);el('rect',{x:x0,y:y(a+b),width:w,height:Math.max(0,y(a)-y(a+b)),class:'barw'},svg);
     if(m<=10||c%2===0)tx(svg,x(c)+bw/2,H-Bm+15,String(c),{class:'tk','text-anchor':'middle'});
     if(m<=10){const t=a+b>EPS?(a/(a+b)).toFixed(2).replace(/^0/,''):'—';tx(svg,x(c)+bw/2,Math.max(12,y(a+b)-4),t,{class:'tk','text-anchor':'middle'})}}
   tx(svg,(L+W-R)/2,H-4,'c（與偽標籤相同的答案數）',{class:'tk','text-anchor':'middle'});
   svg.setAttribute('aria-label',`c 的分布；偽標籤正確率 ${f3(o.corr)}`)}
  // sweep
  {const box=$('#c3-sweep'),W=widthOf(box,260,560),narrow=W<420,H=212,L=40,R=narrow?10:92,T=10,Bm=40,svg=frame(box,W,H),x=lin(0,1,L,W-R),y=lin(0,1,H-Bm,T);
   [0,.5,1].forEach(v=>{el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-6,y(v)+4,String(v),{class:'tk','text-anchor':'end'});tx(svg,x(v),H-Bm+15,String(v),{class:'tk','text-anchor':'middle'})});
   tx(svg,(L+W-R)/2,H-4,'答對機率 q',{class:'tk','text-anchor':'middle'});
   const S=[[],[],[],[]];for(let i=0;i<=100;i++){const q=i/100,e=evalBuckets(B,m,c3.K,q,c3.s);S[0].push([x(q),y(e.corr)]);S[1].push([x(q),y(e.pass)]);if(e.pass>EPS)S[2].push([x(q),y(e.cgp)]);S[3].push([x(q),y(e.r)])}
   const st=['stroke:var(--c-par);stroke-width:2.5;fill:none','stroke:var(--ink);stroke-width:2;fill:none;stroke-dasharray:6 4','stroke:var(--accent);stroke-width:2.5;fill:none;stroke-dasharray:1.5 3.5;stroke-linecap:round','stroke:var(--c-perp);stroke-width:2;fill:none;stroke-dasharray:9 3 2 3'];
   const nm=['偽標籤正確率','保留率','保留者正確率','平均獎勵'];
   S.forEach((p,i)=>{if(p.length)el('path',{d:lp(p),style:st[i]},svg)});
   el('line',{x1:x(c3.q),x2:x(c3.q),y1:T,y2:H-Bm,style:'stroke:var(--ink-3);stroke-width:1'},svg);
   if(!narrow){const ends=S.map((p,i)=>({i,y:p.length?p[p.length-1][1]:y(0)})).sort((a,b)=>a.y-b.y);for(let k=1;k<ends.length;k++)if(ends[k].y-ends[k-1].y<14)ends[k].y=ends[k-1].y+14;
     ends.forEach(e=>tx(svg,W-R+6,e.y+4,nm[e.i],{class:'al'}))}
   svg.setAttribute('aria-label','四條曲線對答對機率 q：'+nm.join('、'))}
  const rows=[['偽標籤正確率',f3(o.corr)],['保留率（3–7）',f3(o.pass)],['保留者中的正確率',o.pass>EPS?f3(o.cgp):'—（沒有題目被保留）'],['平均不確定性獎勵',f3(o.r)],['最常見答案',plTxt]];
  $('#c3-read').innerHTML=rows.map(([a,b],i)=>`<dt>${a}</dt><dd${i===2?' class="tgt"':''}>${b}</dd>`).join('');
  announce('c3',`q = ${c3.q.toFixed(2)}，偽標籤正確率 ${f3(o.corr)}，保留率 ${f3(o.pass)}，保留者正確率 ${o.pass>EPS?f3(o.cgp):'無'}`)}
let c3SyncK,c3SyncM;
function c3SetS(){const s=$('#c3-s'),mn=sMin(c3.K);s.min=mn;s.disabled=c3.K===1;if(c3.K===1)c3.s=1;if(c3.s<mn)c3.s=mn;s.value=c3.s;$('#c3-so').textContent=c3.s.toFixed(2);
  $('#c3-smin').textContent=c3.K===1?'只有一個錯誤答案（s 固定為 1）':`下限 ${mn.toFixed(2)}（= ⌈100/K⌉/100）`;s.setAttribute('aria-valuetext','s = '+c3.s.toFixed(2))}
function c3SyncScen(){$$('#c3-scen button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.s===c3.scen)))}
function initC3(){const q=$('#c3-q');const upQ=()=>{c3.q=+q.value;$('#c3-qo').textContent=c3.q.toFixed(2);q.setAttribute('aria-valuetext','q = '+c3.q.toFixed(2))};upQ();
  q.addEventListener('input',()=>{upQ();c3Render()});
  c3SyncK=segInit($('#c3-K'),[1,2,3,4,5],c3.K,null,k=>{c3.K=k;c3.scen=null;c3SyncScen();c3SetS();c3Render()});
  c3SyncM=segInit($('#c3-m'),[5,10,20],c3.m,null,v=>{c3.m=v;c3Render()});
  $('#c3-s').addEventListener('input',e=>{c3.s=+e.target.value;c3.scen=null;c3SyncScen();c3SetS();c3Render()});
  $('#c3-scen').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;const S=SCEN[b.dataset.s];c3.scen=b.dataset.s;c3.K=S.K;c3.s=S.s;c3SyncK(c3.K);c3SyncScen();c3SetS();c3Render()});
  c3SyncScen();c3SetS();c3Render()}

/* ================= C4 ================= */
const c4={view:'size',sizes:new Set(['0.6B','1.7B','4B']),sel:{si:0,xi:1}};
const SHP={'0.6B':'circle','1.7B':'square','4B':'tri','R-Zero':'circle','Single-R-Zero':'square'},SCOL={'0.6B':'--c-par','1.7B':'--c-perp','4B':'--accent','R-Zero':'--accent','Single-R-Zero':'--ink-2'};
function c4Series(){if(c4.view==='size')return['0.6B','1.7B','4B'].filter(k=>c4.sizes.has(k)).map(k=>({k,top:FIG3[k],bot:TAB7[k],dash:false}));
  return['R-Zero','Single-R-Zero'].map(k=>({k,top:TAB6[k].perf,bot:TAB6[k].acc,dash:k!=='R-Zero'}))}
function mark(svg,s,x,y,col,r=4.5){const st=`fill:var(${col});stroke:var(--surface);stroke-width:1`;if(s==='circle')el('circle',{cx:x,cy:y,r,style:st},svg);
  else if(s==='square')el('rect',{x:x-r,y:y-r,width:2*r,height:2*r,style:st},svg);else el('path',{d:`M${x} ${y-r-1}L${x+r+1} ${y+r}L${x-r-1} ${y+r}Z`,style:st},svg)}
function star(svg,x,y,r=8){let d='';for(let i=0;i<10;i++){const a=-Math.PI/2+i*Math.PI/5,rr=i%2?r*.45:r;d+=(i?'L':'M')+(x+rr*Math.cos(a)).toFixed(1)+' '+(y+rr*Math.sin(a)).toFixed(1)}el('path',{d:d+'Z',class:'star'},svg)}
function c4Chart(box,key,lo,hi,unit){const ser=c4Series(),W=widthOf(box,280,640),narrow=W<480,H=key==='top'?220:200,L=44,R=narrow?12:104,T=16,B=key==='bot'?40:22,svg=frame(box,W,H);
  const x=lin(0,4,L+10,W-R-10),y=lin(lo,hi,H-B,T);
  const ticks=[];for(let v=Math.ceil(lo/10)*10;v<=hi;v+=10)ticks.push(v);ticks.forEach(v=>{el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),class:'grid'},svg);tx(svg,L-6,y(v)+4,v+unit,{class:'tk','text-anchor':'end'})});
  if(key==='bot')STEPS.forEach((s,i)=>{tx(svg,x(i),H-B+15,narrow?(i?s:'B'):(i?'Step '+s:'Base'),{class:'tk','text-anchor':'middle'});if(i)tx(svg,x(i),H-B+30,`第 ${i} 輪`,{class:'tk','text-anchor':'middle'})});
  tx(svg,L,T-4,key==='top'?'數學平均':'偽標籤正確率（%）',{class:'al'});
  const ends=[];
  ser.forEach((s,si)=>{const a=s[key],col=SCOL[s.k],pts=[];a.forEach((v,i)=>{if(v!==null)pts.push([x(i),y(v),i,v])});
    el('path',{d:lp(pts),style:`stroke:var(${col});stroke-width:2.2;fill:none${s.dash?';stroke-dasharray:6 4':''}`},svg);
    const pk=peak(s.top);pts.forEach(p=>{if(p[2]===pk)star(svg,p[0],p[1]);else mark(svg,SHP[s.k],p[0],p[1],col);
      const h=el('circle',{cx:p[0],cy:p[1],r:12,class:'hit'},svg);h.addEventListener('click',()=>{c4.sel={si,xi:p[2]};c4Render()})});
    if(key==='bot'&&pk>0&&a[pk]!==null)tx(svg,x(pk)+(pk<4?10:-10),y(a[pk])-8,`峰值時 ${a[pk].toFixed(1)}%`,{class:'tk halo','text-anchor':pk<4?'start':'end'});
    const last=pts[pts.length-1];ends.push({y:last[1],t:s.k+' '+last[3].toFixed(key==='top'?2:1),col})});
  // peak guide lines across both charts
  ser.forEach(s=>{const pk=peak(s.top);el('line',{x1:x(pk),x2:x(pk),y1:T,y2:H-B,style:`stroke:var(${SCOL[s.k]});stroke-width:1;stroke-dasharray:3 4;opacity:.7`},svg)});
  if(!narrow){ends.sort((a,b)=>a.y-b.y);for(let k=1;k<ends.length;k++)if(ends[k].y-ends[k-1].y<14)ends[k].y=ends[k-1].y+14;
    const over=ends.length?ends[ends.length-1].y-(H-B):0;if(over>0)ends.forEach(e=>e.y-=over);
    ends.forEach(e=>tx(svg,W-R+8,e.y+4,e.t,{class:'al'}))}
  const sel=c4.sel,ss=ser[sel.si];if(ss){const v=ss[key][sel.xi];if(v!==null&&v!==undefined)el('circle',{cx:x(sel.xi),cy:y(v),r:9,class:'sel-ring'},svg)}
  return svg}
function c4Render(){const ser=c4Series();if(c4.sel.si>=ser.length)c4.sel.si=0;
  if(c4.view==='size'){c4Chart($('#c4-top'),'top',10,55,'');c4Chart($('#c4-bot'),'bot',30,80,'')}else{c4Chart($('#c4-top'),'top',40,55,'');c4Chart($('#c4-bot'),'bot',30,80,'')}
  $('#c4-legend').innerHTML=ser.map(s=>`<span class="lg shape"><span class="mk" style="color:var(${SCOL[s.k]})">${{circle:'●',square:'■',tri:'▲'}[SHP[s.k]]}</span>${s.k}${s.dash?'（虛線）':''}</span>`).join('')+'<span class="lg shape"><span class="mk" style="color:var(--accent)">★</span>峰值</span>';
  const s=ser[c4.sel.si],i=c4.sel.xi;let txt='';if(s){const a=s.top[i],b=s.bot[i];txt=`${s.k}，${i?'Step '+STEPS[i]+'（第 '+i+' 輪）':'Base'}：數學平均 ${a===null?'—':a.toFixed(2)}；偽標籤正確率 ${b===null?'—':b.toFixed(1)+'%'}`}
  $('#c4-read').textContent=txt||'點選或以方向鍵選擇資料點';
  $('#c4-tab').innerHTML=`<div class="tscroll"><table class="stack"><caption>峰值（由資料即時求出）</caption><thead><tr><th>系列</th><th>峰值步數</th><th class="num">峰值分數</th><th class="num">峰值時偽標籤正確率</th></tr></thead><tbody>${ser.map(s=>{const p=peak(s.top);return`<tr><td data-h="系列">${s.k}</td><td data-h="峰值步數">${p?'Step '+STEPS[p]+'（第 '+p+' 輪）':'Base'}</td><td class="num" data-h="峰值分數">${s.top[p].toFixed(2)}</td><td class="num" data-h="峰值時正確率">${s.bot[p]===null?'—':s.bot[p].toFixed(1)+'%'}</td></tr>`}).join('')}</tbody></table></div>`;
  announce('c4',txt)}
function initC4(){segInit($('#c4-view'),['size','share'],'size',v=>v==='size'?'模型大小（Fig. 3、Table 7）':'兩個模型 vs 共用參數（Table 6）',v=>{c4.view=v;c4.sel={si:0,xi:1};$('#c4-sizes').classList.toggle('hide',v!=='size');c4Render()});
  const sz=$('#c4-sizes');['0.6B','1.7B','4B'].forEach(k=>{const b=document.createElement('button');b.type='button';b.className='chip';b.textContent=k;b.setAttribute('aria-pressed','true');
    b.addEventListener('click',()=>{if(c4.sizes.has(k)&&c4.sizes.size>1)c4.sizes.delete(k);else c4.sizes.add(k);b.setAttribute('aria-pressed',String(c4.sizes.has(k)));c4.sel={si:0,xi:1};c4Render()});sz.appendChild(b)});
  $('#c4-fig').addEventListener('keydown',e=>{const n=c4Series().length,s=c4.sel;if(e.key==='ArrowRight')s.xi=Math.min(4,s.xi+1);else if(e.key==='ArrowLeft')s.xi=Math.max(c4.view==='size'?0:1,s.xi-1);
    else if(e.key==='ArrowDown')s.si=(s.si+1)%n;else if(e.key==='ArrowUp')s.si=(s.si-1+n)%n;else return;e.preventDefault();c4Render()});
  c4Render()}

/* ================= timeline / toc / theme ================= */
const TL_GROUP={'2402.03300':['rlvr','rzero-grpo-rlvr-foundation'],'2501.12948':['rlvr','rzero-grpo-rlvr-foundation'],
 '2505.15134':['lf','rzero-label-free-intrinsic-rewards'],'2505.19590':['lf','rzero-label-free-intrinsic-rewards'],
 '2411.04109':['mv','rzero-majority-vote-pseudo-labels'],'2504.16084':['mv','rzero-majority-vote-pseudo-labels'],'2505.20347':['mv','rzero-majority-vote-pseudo-labels'],
 '2401.01335':['sp','rzero-self-play-task-generation'],'2505.03335':['sp','rzero-self-play-task-generation'],'2506.01716':['sp','rzero-self-play-task-generation'],
 '2504.05520':['cur','rzero-difficulty-curriculum'],'2504.03380':['cur','rzero-difficulty-curriculum'],
 '2404.05090':['col','rzero-self-training-collapse'],'2505.21444':['col','rzero-self-training-collapse'],
 '2508.05004':['me','rzero-challenger-solver-pipeline']};
const TL_NAME={rlvr:'RLVR 與 GRPO',lf:'無標籤 RL',mv:'多數決偽標籤',sp:'自我對弈與自產任務',cur:'難度課程',col:'崩潰風險',me:'本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.map((e,i)=>[e,i]).sort((a,b)=>(Number(a[0].year)||0)-(Number(b[0].year)||0)||a[1]-b[1]).map(p=>p[0]);
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];if(!TL_GROUP[e.arxiv])console.warn('timeline: ungrouped',e.arxiv);used.add(g[0]);const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${g[1]?`<a href="${g[1]}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
  const on=new Set(used);
  Object.keys(TL_NAME).filter(k=>used.has(k)).forEach(k=>{const b=document.createElement('button');b.type='button';b.className='chip';b.setAttribute('aria-pressed','true');b.textContent=TL_NAME[k];b.dataset.g=k;
    b.addEventListener('click',()=>{if(on.has(k)&&on.size===used.size){on.clear();on.add(k)}else if(on.has(k)){on.delete(k);if(!on.size)used.forEach(x=>on.add(x))}else on.add(k);
      $$('.chip',chips).forEach(c=>c.setAttribute('aria-pressed',String(on.has(c.dataset.g))));$$('.ev',track).forEach(ev=>ev.classList.toggle('dim',!on.has(ev.dataset.g)))});chips.appendChild(b)});
  const rail=$('#rail'),by=dir=>rail.scrollBy({left:dir*Math.min(rail.clientWidth*.8,560),behavior:RM?'auto':'smooth'});
  $('#tl-prev').addEventListener('click',()=>by(-1));$('#tl-next').addEventListener('click',()=>by(1));
  rail.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){by(1);e.preventDefault()}if(e.key==='ArrowLeft'){by(-1);e.preventDefault()}})}
function initToc(){const secs=$$('[data-toc]'),ol=$('#toc'),bar=$('#tocbar');
  secs.forEach(s=>{ol.insertAdjacentHTML('beforeend',`<li><a href="#${s.id}"><span class="n">${s.dataset.toc}</span><span>${s.dataset.tt}</span></a></li>`);bar.insertAdjacentHTML('beforeend',`<a href="#${s.id}">${s.dataset.toc} ${s.dataset.tt}</a>`)});
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){const id=e.target.id;$$('.toc a,.tocbar a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+id));const ab=$(`.tocbar a[href="#${id}"]`);if(ab&&bar.offsetParent)bar.scrollTo({left:ab.offsetLeft-16,behavior:RM?'auto':'smooth'})}})},{rootMargin:'-40% 0px -55% 0px'});
  secs.forEach(s=>io.observe(s));addEventListener('scroll',()=>$('#appbar').classList.toggle('raised',scrollY>8),{passive:true})}
function initTheme(){let saved=null;try{saved=localStorage.getItem('pm-theme')}catch(e){}if(saved)document.documentElement.dataset.theme=saved;
  $('#theme').addEventListener('click',()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');const nx=cur==='dark'?'light':'dark';document.documentElement.dataset.theme=nx;try{localStorage.setItem('pm-theme',nx)}catch(e){}})}

/* ================= self test (design.md section 5.5) ================= */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t=1e-4)=>Math.abs(a-b)<=t+1e-12;
  // C1 (5.1)
  {const R=[.2,.4,.6,.8,1,.8,.6,.4,.2,0],K=[0,0,1,1,1,1,1,0,0,0];ok(R.every((r,i)=>near(rUnc(i+1,10),r,1e-9))&&K.every((k,i)=>keep(i+1,10)===!!k),'C1 table');
   const v=vote('AAAAAABBBC'.split(''));ok(v.c===6&&v.label==='A'&&near(rUnc(v.c,10),.8,1e-9),'C1 default');
   const w=vote('CCCCCCCAAA'.split(''));ok(w.c===7&&w.label==='C'&&keep(7,10)&&near(rUnc(7,10),.6,1e-9),'C1 consistent wrong');
   ok(vote('AAAAABBBBB'.split('')).tied.length===2,'C1 tie')}
  // C2 (5.2)
  {const q=RAW_DEFAULT.map(extractQ);ok(q[7]===null&&near(bleu(q[0],q[1]),.8492),'C2 bleu(q1,q2)='+bleu(q[0],q[1]).toFixed(4));
   const T=[[.5,true,'[[0,1],[2],[3],[4],[5],[6]]',[.861,.213,.618,-.030,-.030,-1.327,1.266,-1.570],.4844,.3085],
            [.65,true,'[[0,1],[2],[3,4],[5],[6]]',[.943,.308,.705,-.327,-.327,-1.201,1.340,-1.439],.4531,.3149],
            [.7,true,'[[0,1,2],[3,4],[5],[6]]',[.838,.123,.123,-.145,-.145,-1.129,1.732,-1.397],.3906,.2796],
            [.5,false,'[[0,1],[2],[3],[4],[5],[6]]',[1.037,.484,.484,-.069,-.069,-1.176,1.037,-1.729],.6250,.3615]];
   T.forEach(([tau,rep,cl,A,mu,sd])=>{const P=pipeline(RAW_DEFAULT,C_DEFAULT,tau,rep);ok(JSON.stringify(P.clusters)===cl&&P.adv.A.every((a,i)=>near(a,A[i],6e-4))&&near(P.adv.mu,mu)&&near(P.adv.sd,sd),`C2 tau=${tau} rep=${rep}`)});
   const P=pipeline(RAW_DEFAULT,C_DEFAULT,.2,true);ok(JSON.stringify(P.clusters)==='[[0,1],[2],[3],[4],[5],[6]]','C2 tau=0.20');
   ok(advantages(Array(8).fill(.375)).A.every(a=>a===0),'C2 equal rewards -> 0')}
  // C3 (5.3)
  {const T=[[10,3,.5,.5,.7869,.9449,.7750,.8024],[10,3,.7,.5,.9792,.6172,.9663,.5790],[10,1,.3,1,.0988,.6156,.1579,.5761],[10,3,0,.5,0,.9445,NaN,.8202]];
   T.forEach(([m,K,q,s,a,b,c,r])=>{const o=evalBuckets(buildBuckets(m,K),m,K,q,s);ok(near(o.corr,a,5e-4)&&near(o.pass,b,5e-4)&&(isNaN(c)?(o.cgp===0||isNaN(o.cgp)):near(o.cgp,c,5e-4))&&near(o.r,r,5e-4),`C3 m=${m} K=${K} q=${q}`)});
   const pc=(o,c)=>o.hc[c]/(o.hc[c]+o.hw[c]);
   const o1=evalBuckets(buildBuckets(10,1),10,1,.3,1);ok(near(pc(o1,6),.155,5e-4)&&near(pc(o1,8),.006,5e-4),'C3 concentrated P(ok|c)');
   const o2=evalBuckets(buildBuckets(10,3),10,3,.3,.5);ok(near(pc(o2,5),.356,5e-4)&&near(pc(o2,10),.176,5e-4)&&plurality(.3,3,.5)==='no','C3 medium q=0.3 P(ok|c)');
   let tot=0;const o3=evalBuckets(buildBuckets(20,5),20,5,.37,.6);for(let c=0;c<=20;c++)tot+=o3.hc[c]+o3.hw[c];ok(Math.abs(tot-1)<1e-9,'C3 total prob');
   ok([1,2,3,4,5].map(sMin).join()==='1,0.5,0.34,0.25,0.2','C3 s min')}
  // C4 (5.4)
  ok(peak(FIG3['0.6B'])===1&&peak(FIG3['1.7B'])===2&&peak(FIG3['4B'])===3&&TAB7['0.6B'][1]===70.6&&TAB7['4B'][3]===48.8,'C4 peaks');
  ok(peak(TAB6['R-Zero'].perf)===3&&peak(TAB6['Single-R-Zero'].perf)===1,'C4 Table 6 peaks');
  // DOM
  ok($$('details.check').length===13,'check boxes: '+$$('details.check').length);
  {const tl=JSON.parse($('#timeline-data').textContent);ok(tl.length===15&&tl.every(e=>TL_GROUP[e.arxiv]),'timeline groups')}
  ok(!!document.querySelector('.katex')||document.documentElement.classList.contains('nokatex'),'KaTeX rendered');
  ok(!document.querySelector('.katex-error'),'no KaTeX errors');
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'')}

/* ================= boot ================= */
initTheme();initToc();cBars($('#hero-c'),{H:150,T:22});drawFig2();initC1();initC2();initC3();initC4();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{cBars($('#hero-c'),{H:150,T:22});drawFig2();c1Render();c3Render();c4Render()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath();if(!document.querySelector('.katex')&&typeof katex==='undefined')document.documentElement.classList.add('nokatex');if(/[?&]test\b/.test(location.search))setTimeout(selfTest,50)});
})();
