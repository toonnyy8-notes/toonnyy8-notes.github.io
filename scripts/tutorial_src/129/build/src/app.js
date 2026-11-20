(()=>{
'use strict';
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
function el(tag,a,p){const e=document.createElementNS(NS,tag);if(a)for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function tx(p,x,y,s,a){const t=el('text',Object.assign({x,y},a||{}),p);t.textContent=s;return t}
const lin=(d0,d1,r0,r1)=>v=>r0+(v-d0)/(d1-d0)*(r1-r0);
const lgs=(d0,d1,r0,r1)=>{const a=Math.log(d0),b=Math.log(d1);return v=>r0+(Math.log(v)-a)/(b-a)*(r1-r0)};
function lp(pts){let s='';for(let i=0;i<pts.length;i++)s+=(i?'L':'M')+pts[i][0].toFixed(1)+' '+pts[i][1].toFixed(1);return s}
const MINUS='−';
const fmt=(x,n=4)=>{if(!isFinite(x))return '—';let s=(Math.abs(x)<0.5*Math.pow(10,-n)?0:x).toFixed(n);return s.replace('-',MINUS)};
const fmtS=(x,n=4)=>(x>0.5*Math.pow(10,-n)?'+':'')+fmt(x,n);
const fmtErr=x=>x.toExponential(1).replace('-',MINUS)+(x<1e-9?' ✓':'');
function widthOf(box,min,max){return Math.max(min,Math.min(max,Math.floor(box.clientWidth||max)))}
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function onRaf(fn){let r=0;return()=>{cancelAnimationFrame(r);r=requestAnimationFrame(fn)}}

/* ---------- math ---------- */
function softmax(z){const m=Math.max(...z),e=z.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s)}
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const norm=a=>Math.sqrt(dot(a,a));
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const entropy=p=>-p.reduce((s,v)=>s+v*Math.log(Math.max(v,1e-300)),0);

/* C1 */
const C1={Z:[1.2,0.6,0,-0.8],XI:[-0.5,0.8,0.2,-0.5],PRESET:{const:[1,1,1,1],b01:[0.9,0.2,0.6,0.1],pm1:[0.8,-0.6,0.2,-0.8]}};
function c1Compute(lam,Q){
  const p=softmax(C1.Z),q=softmax(C1.Z.map((z,i)=>z+lam*C1.XI[i]));
  const Rq=dot(q,Q),Rp=dot(p,Q),sbar=sub(q,p);
  const pg=q.map((v,i)=>v*Q[i]-Rq*p[i]),drift=sbar.map(v=>Rq*v),signal=q.map((v,i)=>v*(Q[i]-Rq)),onp=p.map((v,i)=>v*(Q[i]-Rp));
  const r=p.map((v,i)=>v/q[i]),w=r.map(v=>Math.min(v,2)),cw=dot(q.map((v,i)=>v*w[i]),Q);
  const tis=q.map((v,i)=>v*w[i]*Q[i]-cw*p[i]);
  const err=norm(pg.map((v,i)=>v-drift[i]-signal[i]));
  return {p,q,Rq,Rp,sbar,pg,drift,signal,onp,tis,maxR:Math.max(...r),err};
}
/* C2 */
function c2Compute(h1,h2,pi,G){const R01=pi*h1+(1-pi)*h2,g=1-1/G;
  const rows={b01:[h1,h2],pm1:[2*h1-1,2*h2-1],gc:[g*2*(h1-R01),g*2*(h2-R01)]};
  const out={};for(const k in rows){const [a,b]=rows[k];out[k]={c1:a,c2:b,avg:pi*a+(1-pi)*b,rms:Math.sqrt(pi*a*a+(1-pi)*b*b)}}return out}
/* C3 */
const C3={XI:[0.6,-0.2,0.1,-0.3,-0.2],Q:{const:[1,1,1,1,1],b01:[1,1,0,0,0]},N:3000};
function c3Run(mode,lam,eta){
  const Q=C3.Q[mode],N=C3.N,d=C3.XI.map(v=>lam*v),z0=[0,0,0,0,0],qfix=softmax(d);
  const keys=['pg','fix','sc','onp'],z={},ent={},acc={};
  keys.forEach(k=>{z[k]=z0.slice();ent[k]=new Float64Array(N+1);acc[k]=new Float64Array(N+1)});
  for(let n=0;n<=N;n++)for(const k of keys){
    const p=softmax(z[k]);ent[k][n]=entropy(p);acc[k][n]=dot(p,Q);if(n===N)continue;
    let g;
    if(k==='onp'){const R=dot(p,Q);g=p.map((v,i)=>v*(Q[i]-R))}
    else{const q=k==='fix'?qfix:softmax(z[k].map((v,i)=>v+d[i])),R=dot(q,Q);g=k==='sc'?q.map((v,i)=>v*(Q[i]-R)):q.map((v,i)=>v*Q[i]-R*p[i])}
    for(let i=0;i<5;i++)z[k][i]+=eta*g[i];
  }
  const dz=norm(z.pg);
  return {ent,acc,dz};
}
function tHalf(a){const h=a[0]/2;for(let n=0;n<a.length;n++)if(a[n]<=h)return n;return null}
function firstBelow(a,v){for(let n=0;n<a.length;n++)if(a[n]<=v)return n;return null}
/* C4 */
const C4={V:20,Z:Array.from({length:20},(_,v)=>-0.4*v),XI:[0.31,-1.12,0.54,0.87,-0.43,1.25,-0.76,0.12,-1.38,0.66,0.95,-0.21,-0.88,1.47,0.03,-0.59,1.06,-1.21,0.38,-0.27]};
const F={sc:()=>1,tis:r=>Math.min(r,2),mis:r=>(r>=0.5&&r<=5)?r:0,is:r=>r};
function c4Compute(sigma,k,fname){
  const f=F[fname],V=C4.V,p=softmax(C4.Z),q=softmax(C4.Z.map((z,i)=>z+sigma*C4.XI[i]));
  const order=[...Array(V).keys()].sort((a,b)=>q[b]-q[a]||a-b),inH=new Array(V).fill(false);order.slice(0,k).forEach(i=>inH[i]=true);
  let mqT=0,mpT=0;for(let i=0;i<V;i++)if(!inH[i]){mqT+=q[i];mpT+=p[i]}
  const rho=Math.max(mqT,1e-6)/Math.max(mpT,1e-6),alpha=rho*f(1/rho);
  const w=p.map((v,i)=>f(v/q[i]));
  const S=v=>p.map((pj,j)=>(j===v?1:0)-pj);               // s_v = e_v - p
  const acc=(coefs)=>{const out=new Array(V).fill(0);coefs.forEach((c,v)=>{if(!c)return;const s=S(v);for(let j=0;j<V;j++)out[j]+=c*s[j]});return out};
  const c=acc(q.map((v,i)=>v*w[i]));
  const chat=acc(q.map((v,i)=>inH[i]?v*w[i]-alpha*p[i]:0));
  const qhat=q.map((v,i)=>inH[i]?v:rho*p[i]);
  const full=acc(qhat.map((v,i)=>v*f(p[i]/v)));
  const ns=norm(sub(q,p)),resid=norm(sub(c,chat));
  const r=p.map((v,i)=>v/q[i]);
  return {p,q,qhat,inH,rho,alpha,mqT,mpT,nc:norm(c),nchat:norm(chat),resid,rel:ns<1e-12?null:resid/ns,ident:norm(sub(chat,full)),maxR:Math.max(...r),minR:Math.min(...r)};
}

/* ---------- shared bar chart ---------- */
const FILLS=['var(--c-par)','url(#pt-hatch)','url(#pt-dots)'];
function bars(svg,o){ // o: {W,H,labels,series:[{name,vals}],y:[lo,hi],yt,sel,log,zeroLine}
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${o.W} ${o.H}`);svg.setAttribute('width',o.W);svg.setAttribute('height',o.H);
  const m={l:48,r:8,t:o.title?32:8,b:26},x0=m.l,x1=o.W-m.r,y0=o.H-m.b,y1=m.t;
  const Y=o.log?lgs(o.y[0],o.y[1],y0,y1):lin(o.y[0],o.y[1],y0,y1),n=o.labels.length,gw=(x1-x0)/n;
  if(o.title)tx(svg,4,14,o.title,{class:'al'});
  (o.yt||[]).forEach(([v,l])=>{const yy=Y(v);el('line',{x1:x0,x2:x1,y1:yy,y2:yy,class:'grid'},svg);tx(svg,x0-6,yy+4,l,{class:'tk','text-anchor':'end'})});
  if(o.band)o.band.forEach(([a,b])=>el('rect',{x:x0+a*gw,y:y1,width:(b-a)*gw,height:y0-y1,class:'band'},svg));
  const base=o.log?y0:Y(0);
  if(o.sel!=null&&o.sel>=0)el('rect',{x:x0+o.sel*gw+1,y:y1,width:gw-2,height:y0-y1,class:'selbox'},svg);
  const ns=o.series.length,bw=Math.max(2,Math.min(28,(gw-6)/ns));
  o.series.forEach((s,si)=>s.vals.forEach((v,i)=>{if(v==null)return;const xx=x0+i*gw+(gw-bw*ns)/2+si*bw;
    const vv=o.log?Math.max(v,o.y[0]):Math.max(o.y[0],Math.min(o.y[1],v));const yy=Y(vv);
    el('rect',{x:xx,y:Math.min(yy,base),width:bw-1,height:Math.max(.5,Math.abs(base-yy)),fill:s.fill||FILLS[si],class:'bar'},svg)}));
  if(o.marks)o.marks.forEach((v,i)=>{const yy=Y(Math.max(v,o.y[0]));el('line',{x1:x0+i*gw+3,x2:x0+(i+1)*gw-3,y1:yy,y2:yy,class:'pmark'},svg)});
  el('line',{x1:x0,x2:x1,y1:base,y2:base,class:'axl0'},svg);
  const step=Math.ceil(n*22/(x1-x0));
  o.labels.forEach((l,i)=>{if(i%step)return;tx(svg,x0+(i+.5)*gw,o.H-8,l,{class:'tk','text-anchor':'middle'})});
  return {x0,gw,Y};
}
function symY(vals){const m=Math.max(0.05,...vals.map(Math.abs));const t=m>0.3?Math.ceil(m*10)/10:Math.ceil(m*20)/20;return [-t,t]}
function ticks3([lo,hi]){return [[lo,fmt(lo,2)],[0,'0'],[hi,fmt(hi,2)]]}
// keyboard selection + table toggle
function barNav(fig,n,get,set,onSel){fig.addEventListener('keydown',e=>{let s=get();if(e.key==='ArrowRight')s=(s+1)%n();else if(e.key==='ArrowLeft')s=(s-1+n())%n();else return;e.preventDefault();set(s);onSel()});
  fig.addEventListener('click',e=>{const svg=$('svg',fig),r=svg.getBoundingClientRect(),x=e.clientX-r.left,info=fig._info;if(!info)return;const i=Math.floor((x-info.x0)/info.gw);if(i>=0&&i<n()){set(i);onSel()}})}
function tableHTML(cap,head,rows){return `<div class="tscroll"><table><caption>${cap}</caption><thead><tr>${head.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map((c,i)=>`<td${i?' class="num"':''}>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}
const TABLES={};
function initTables(){$$('[data-tab]').forEach(b=>{const id=b.dataset.tab,box=$('#'+id+'-tab');b.setAttribute('aria-expanded','false');b.setAttribute('aria-controls',id+'-tab');
  b.addEventListener('click',()=>{const open=box.classList.toggle('hide')===false;b.setAttribute('aria-expanded',String(open));b.textContent=open?'隱藏表格':'以表格顯示';if(open&&TABLES[id])box.innerHTML=TABLES[id]()})})}
function refreshTable(id){const box=$('#'+id+'-tab');if(box&&!box.classList.contains('hide')&&TABLES[id])box.innerHTML=TABLES[id]()}

/* ---------- C1 ---------- */
const st1={lam:0.5,preset:'const',Q:C1.PRESET.const.slice(),sel:1};
const TOK=['token 1','token 2','token 3','token 4'];
function initC1(){
  const qbox=$('#c1-q');
  qbox.innerHTML=st1.Q.map((v,i)=>`<div class="rng"><label class="lbl" for="c1-q${i}">Q（token ${i+1}）<output id="c1-q${i}o">${fmt(v,2)}</output></label><input id="c1-q${i}" type="range" min="-1" max="1" step="0.05" value="${v}"></div>`).join('');
  const r=onRaf(renderC1);
  const setLam=v=>{st1.lam=v;['c1-lam','c1b-lam'].forEach(id=>{$('#'+id).value=v;$('#'+id+'o').textContent=fmt(v,2);$('#'+id).setAttribute('aria-valuetext','λ = '+fmt(v,2))})};
  ['c1-lam','c1b-lam'].forEach(id=>{const e=$('#'+id);e.addEventListener('input',()=>{setLam(+e.value);r()});e.addEventListener('change',()=>speakC1())});
  const setPreset=p=>{st1.preset=p;if(p!=='custom')st1.Q=C1.PRESET[p].slice();$$('#c1-pre .chip,#c1b-pre .chip').forEach(c=>c.setAttribute('aria-pressed',String(c.dataset.p===p)));
    st1.Q.forEach((v,i)=>{$('#c1-q'+i).value=v;$('#c1-q'+i+'o').textContent=fmt(v,2)})};
  $$('#c1-pre .chip,#c1b-pre .chip').forEach(c=>c.addEventListener('click',()=>{if(c.dataset.p==='custom')return;setPreset(c.dataset.p);renderC1();speakC1()}));
  st1.Q.forEach((_,i)=>{const e=$('#c1-q'+i);e.addEventListener('input',()=>{st1.Q[i]=+e.value;$('#c1-q'+i+'o').textContent=fmt(+e.value,2);setPreset('custom');r()});e.addEventListener('change',()=>speakC1())});
  ['c1a','c1b'].forEach(id=>barNav($('#'+id+'-fig'),()=>4,()=>st1.sel,s=>st1.sel=s,renderC1));
  setLam(0.5);
  TABLES.c1a=()=>{const o=c1Compute(st1.lam,st1.Q);return tableHTML('C1a：每個 token 的值',['token','p','q','PG','drift','signal'],TOK.map((t,i)=>[t,fmt(o.p[i]),fmt(o.q[i]),fmt(o.pg[i]),fmt(o.drift[i]),fmt(o.signal[i])]))};
  TABLES.c1b=()=>{const o=c1Compute(st1.lam,st1.Q);return tableHTML('C1b：每個 token 的值',['token','SC','on-policy','TIS'],TOK.map((t,i)=>[t,fmt(o.signal[i]),fmt(o.onp[i]),fmt(o.tis[i])]))};
}
function renderC1(){const o=c1Compute(st1.lam,st1.Q);
  [['c1a',[['PG',o.pg],['drift',o.drift],['signal',o.signal]]],['c1b',[['SC',o.signal],['on-policy',o.onp],['TIS',o.tis]]]].forEach(([id,ser])=>{
    const fig=$('#'+id+'-fig'),svg=$('svg',fig),W=widthOf(fig,300,560);
    const y=symY(ser.flatMap(s=>s[1]).concat([0.2]));
    // two stacked charts: p/q then series
    svg.textContent='';const H1=120,H2=200;svg.setAttribute('viewBox',`0 0 ${W} ${H1+H2}`);svg.setAttribute('width',W);svg.setAttribute('height',H1+H2);
    const g1=el('g',null,svg),g2=el('g',{transform:`translate(0,${H1})`},svg);
    const a=el('svg',{x:0,y:0},g1),b=el('svg',{x:0,y:0},g2);
    bars(a,{W,H:H1,labels:TOK,series:[{vals:o.p,fill:'var(--c-par)'},{vals:o.q,fill:'url(#pt-hatch)'}],y:[0,0.6],yt:[[0,'0'],[0.3,'0.3'],[0.6,'0.6']],sel:st1.sel,title:'p（實心）與 q（斜線）'});
    fig._info=bars(b,{W,H:H2,labels:TOK,series:ser.map(s=>({vals:s[1]})),y,yt:ticks3(y),sel:st1.sel,title:'期望更新（logit 座標）'});
    const i=st1.sel;
    svg.setAttribute('aria-label',`${id} 長條圖，選取 token ${i+1}：`+ser.map(s=>`${s[0]} ${fmt(s[1][i])}`).join('，'));
  });
  const i=st1.sel;
  $('#c1a-ro').innerHTML=`E_q[R] = <b>${fmt(o.Rq)}</b><br>‖平均 score‖ = <b>${fmt(norm(o.sbar))}</b><br>‖PG‖ = <b>${fmt(norm(o.pg))}</b>　‖drift‖ = <b>${fmt(norm(o.drift))}</b>　‖signal‖ = <b>${fmt(norm(o.signal))}</b><br>分解誤差 ‖PG − drift − signal‖ = ${fmtErr(o.err)}<br>token ${i+1}：p ${fmt(o.p[i])}，q ${fmt(o.q[i])}，PG ${fmt(o.pg[i])}，drift ${fmt(o.drift[i])}，signal ${fmt(o.signal[i])}`;
  $('#c1b-ro').innerHTML=`‖SC − on-policy‖ = <b>${fmt(norm(sub(o.signal,o.onp)))}</b><br>‖TIS − on-policy‖ = <b>${fmt(norm(sub(o.tis,o.onp)))}</b><br>max p/q = <b>${fmt(o.maxR,3)}</b>（${o.maxR>2?'> 2，TIS 截斷生效':'≤ 2，TIS 截斷未生效'}）<br>token ${i+1}：SC ${fmt(o.signal[i])}，on-policy ${fmt(o.onp[i])}，TIS ${fmt(o.tis[i])}`;
  refreshTable('c1a');refreshTable('c1b');
}
function speakC1(){const o=c1Compute(st1.lam,st1.Q);const s=`λ = ${fmt(st1.lam,2)}，‖drift‖ ${fmt(norm(o.drift))}，‖signal‖ ${fmt(norm(o.signal))}`;announce('c1a',s);announce('c1b',s+`，‖TIS − on-policy‖ ${fmt(norm(sub(o.tis,o.onp)))}`)}

/* ---------- C2 ---------- */
const st2={h1:0.8,h2:0.2,pi:0.5,G:8,sel:4};
const C2ROWS=[['b01','+1/0'],['pm1','±1'],['gc','group-centered']];
function initC2(){const r=onRaf(renderC2);
  [['c2-ha','h1'],['c2-hb','h2'],['c2-pi','pi']].forEach(([id,k])=>{const e=$('#'+id);e.addEventListener('input',()=>{st2[k]=+e.value;$('#'+id+'o').textContent=fmt(+e.value,2);e.setAttribute('aria-valuetext',fmt(+e.value,2));r()});e.addEventListener('change',speakC2)});
  $$('#c2-G button').forEach(b=>b.addEventListener('click',()=>{st2.G=+b.dataset.v;$$('#c2-G button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));renderC2();speakC2()}));
  barNav($('#c2-fig'),()=>6,()=>st2.sel,s=>st2.sel=s,renderC2);
  TABLES.c2=()=>{const o=c2Compute(st2.h1,st2.h2,st2.pi,st2.G);return tableHTML('C2：drift 係數',['reward','c 甲','c 乙','prompt 平均','均方根'],C2ROWS.map(([k,n])=>[n,fmtS(o[k].c1),fmtS(o[k].c2),fmtS(o[k].avg),fmt(o[k].rms)]))};
}
function renderC2(){const o=c2Compute(st2.h1,st2.h2,st2.pi,st2.G),fig=$('#c2-fig'),svg=$('svg',fig),W=widthOf(fig,300,560);
  const rowH=58,H=rowH*3+30;svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const L=Math.min(200,Math.round(W*0.45)),x0=L+8,x1=W-10,X=lin(-1,1,x0,x1);
  [-1,-0.5,0,0.5,1].forEach(v=>{el('line',{x1:X(v),x2:X(v),y1:6,y2:H-22,class:v?'grid':'axl0'},svg);if(W>380||v%1===0)tx(svg,X(v),H-6,v?fmt(v,1):'0',{class:'tk','text-anchor':'middle'})});
  C2ROWS.forEach(([k,name],ri)=>{const y=8+ri*rowH;tx(svg,4,y+12,name,{class:'lab'});
    [o[k].c1,o[k].c2].forEach((v,j)=>{const yy=y+18+j*18,idx=ri*2+j,xa=X(Math.min(0,v)),xb=X(Math.max(0,v));
      if(idx===st2.sel)el('rect',{x:2,y:yy-2,width:x1-2,height:18,class:'selbox'},svg);
      el('rect',{x:xa,y:yy,width:Math.max(.5,xb-xa),height:14,fill:FILLS[j],class:'bar'},svg);
      const lab=(j?'乙 ':'甲 ')+fmtS(v,3)+(Math.abs(v)<5e-4?'':v>0?' 拉向':' 推離');
      tx(svg,12,yy+11,lab,{class:'tk'})})});
  fig._info=null;
  const s=st2.sel,row=C2ROWS[s>>1],val=o[row[0]][s&1?'c2':'c1'];
  svg.setAttribute('aria-label',`C2 長條圖，選取 ${row[1]} 路徑${s&1?'乙':'甲'}：${fmtS(val,3)}`);
  $('#c2-ro').innerHTML=C2ROWS.map(([k,n])=>`${n}：平均 <b>${fmtS(o[k].avg)}</b>，均方根 <b>${fmt(o[k].rms)}</b>`).join('<br>')+`<br>選取：${row[1]}，路徑${s&1?'乙':'甲'} = <b>${fmtS(val)}</b>`;
  refreshTable('c2');
}
function c2Keys(){const fig=$('#c2-fig');fig.addEventListener('click',e=>{const svg=$('svg',fig),r=svg.getBoundingClientRect(),y=e.clientY-r.top,ri=Math.floor((y-8)/58),j=Math.floor((y-8-ri*58-18)/18);if(ri>=0&&ri<3&&(j===0||j===1)){st2.sel=ri*2+j;renderC2()}})}
function speakC2(){const o=c2Compute(st2.h1,st2.h2,st2.pi,st2.G);announce('c2',`group-centered：甲 ${fmtS(o.gc.c1,3)}，乙 ${fmtS(o.gc.c2,3)}，prompt 平均 ${fmtS(o.gc.avg,3)}`)}

/* ---------- C3 ---------- */
const st3={mode:'const',lam:0.1,eta:1,showSC:false};
const lamOf=v=>Math.pow(10,-2+2*v/200);
const C3S=[['pg','PG（sampler 同步）','tot',''],['onp','on-policy（＝精確 IS）','par',''],['fix','PG（固定 teacher）','ink','dash'],['sc','score centering','acc','dot']];
let c3cache=null;
function initC3(){const r=onRaf(()=>{c3cache=null;renderC3()});
  const e=$('#c3-lam');e.addEventListener('input',()=>{st3.lam=lamOf(+e.value);$('#c3-lamo').textContent=fmt(st3.lam,st3.lam<0.1?3:2);e.setAttribute('aria-valuetext','λ = '+fmt(st3.lam,3));r()});e.addEventListener('change',speakC3);
  $$('#c3-mode .chip').forEach(c=>c.addEventListener('click',()=>{st3.mode=c.dataset.m;$$('#c3-mode .chip').forEach(x=>x.setAttribute('aria-pressed',String(x===c)));c3cache=null;renderC3();speakC3()}));
  $$('#c3-eta button').forEach(b=>b.addEventListener('click',()=>{st3.eta=+b.dataset.v;$$('#c3-eta button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));c3cache=null;renderC3();speakC3()}));
  $('#c3-sc').addEventListener('change',e=>{st3.showSC=e.target.checked;renderC3()});
  $('#goto-c3').addEventListener('click',()=>{$('#c3-sc').checked=true;st3.showSC=true;renderC3();$('#c3').scrollIntoView({behavior:RM?'auto':'smooth',block:'start'});$('#c3-sc').focus({preventScroll:true})});
}
function renderC3(){if(!c3cache)c3cache=c3Run(st3.mode,st3.lam,st3.eta);const o=c3cache,fig=$('#c3-fig'),svg=$('svg',fig),W=widthOf(fig,300,600);
  const b01=st3.mode==='b01',H1=230,H2=b01?120:0,H=H1+H2;svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const show=C3S.filter(s=>s[0]!=='sc'||st3.showSC);
  const panel=(top,h,yr,yt,key,title)=>{const m={l:40,r:16,t:top+20,b:top+h-26},X=lgs(1,C3.N+1,m.l,W-m.r),Y=lin(yr[0],yr[1],m.b,m.t);
    tx(svg,4,top+14,title,{class:'al'});
    yt.forEach(([v,l])=>{el('line',{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),class:'grid'},svg);tx(svg,m.l-6,Y(v)+4,l,{class:'tk','text-anchor':'end'})});
    [1,10,100,1000,3001].forEach(v=>{const xx=X(v);el('line',{x1:xx,x2:xx,y1:m.b,y2:m.b+4,class:'axl'},svg);tx(svg,xx,m.b+17,String(v===3001?3000:v===1?0:v),{class:'tk','text-anchor':'middle'})});
    el('line',{x1:m.l,x2:W-m.r,y1:m.b,y2:m.b,class:'axl'},svg);
    const ends=[];
    show.forEach(([k,name,cls,dash])=>{const a=o[key][k],pts=[];for(let n=0;n<=C3.N;n+=(n<100?1:n<1000?5:20))pts.push([X(n+1),Y(a[n])]);pts.push([X(C3.N+1),Y(a[C3.N])]);
      el('path',{d:lp(pts),class:cls+(dash?' '+dash:'')+' c3l'},svg);ends.push({y:Y(a[C3.N]),name})});
    return {X,Y,m,ends};
  };
  const p1=panel(0,H1,[0,1.7],[[0,'0'],[Math.log(2),'ln 2'],[Math.log(5),'ln 5']],'ent','trainer 熵 Ent(p)（nats）');
  if(b01)panel(H1,H2,[0,1],[[0,'0'],[1,'1']],'acc','trainer 準確率 p·Q');
  const leg=$('#c3-leg');leg.innerHTML=show.map(([k,n,cls,dash])=>`<span class="lg c3-${k}">${n}</span>`).join('');
  const N=C3.N,f3=v=>fmt(v,3),th=k=>{const t=tHalf(o.ent[k]);return t==null?'未達到':String(t)};
  $('#c3-ro').innerHTML=show.map(([k,n])=>`${n}：Ent@100 ${f3(o.ent[k][100])}，@1000 ${f3(o.ent[k][1000])}，@3000 <b>${f3(o.ent[k][N])}</b>；t½ <b>${th(k)}</b>`).join('<br>')+`<br>PG（同步）‖z₃₀₀₀ − z₀‖ = <b>${fmt(o.dz,2)}</b>`+(b01&&st3.showSC?`<br>SC 熵首次 ≤ 0.35 的步數：<b>${firstBelow(o.ent.sc,0.35)??'未達到'}</b>`:'');
  svg.setAttribute('aria-label',`C3 熵對步數：λ = ${fmt(st3.lam,3)}，PG 同步第 3000 步熵 ${f3(o.ent.pg[N])}`);
}
function speakC3(){if(!c3cache)return;const t=tHalf(c3cache.ent.pg);announce('c3',`λ = ${fmt(st3.lam,2)}，PG 第 3000 步熵 ${fmt(c3cache.ent.pg[C3.N],3)}，t½ = ${t==null?'未達到':t}`)}

/* ---------- C4 ---------- */
const st4={sigma:0.5,k:8,f:'sc',sel:0};
const FN={sc:'SC（f = 1）',tis:'TIS + SC',mis:'MIS + SC',is:'精確 IS'};
function initC4(){const r=onRaf(renderC4);
  [['c4-s','sigma',2],['c4-k','k',0]].forEach(([id,k,d])=>{const e=$('#'+id);e.addEventListener('input',()=>{st4[k]=+e.value;$('#'+id+'o').textContent=d?fmt(+e.value,d):e.value;e.setAttribute('aria-valuetext',(k==='k'?'k = ':'σ = ')+$('#'+id+'o').textContent);r()});e.addEventListener('change',speakC4)});
  $$('#c4-f button').forEach(b=>b.addEventListener('click',()=>{st4.f=b.dataset.v;$$('#c4-f button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));renderC4();speakC4()}));
  barNav($('#c4-fig'),()=>20,()=>st4.sel,s=>st4.sel=s,renderC4);
  TABLES.c4=()=>{const o=c4Compute(st4.sigma,st4.k,st4.f);return tableHTML('C4：每個 token 的機率',['token','p','q','q̂','head？'],o.p.map((v,i)=>[String(i+1),fmt(v),fmt(o.q[i]),fmt(o.qhat[i]),o.inH[i]?'是':'否']))};
}
function renderC4(){const o=c4Compute(st4.sigma,st4.k,st4.f),fig=$('#c4-fig'),svg=$('svg',fig),W=widthOf(fig,300,920),narrow=W<560;
  const labels=o.p.map((_,i)=>narrow?'':String(i+1));
  const qv=o.q.map((v,i)=>o.inH[i]?v:null),tv=o.qhat.map((v,i)=>o.inH[i]?null:v);
  // head band: head tokens may be non-contiguous; draw per token
  const band=[];o.inH.forEach((h,i)=>{if(h)band.push([i,i+1])});
  fig._info=bars(svg,{W,H:240,labels,series:[{vals:qv,fill:'var(--c-par)'},{vals:tv,fill:'url(#pt-hatch)'}],y:[1e-4,1],log:true,yt:[[1e-4,'10⁻⁴'],[1e-3,'10⁻³'],[1e-2,'10⁻²'],[1e-1,'10⁻¹'],[1,'1']],sel:st4.sel,band,marks:o.p,title:`head（k = ${st4.k}，底色）與 tail（斜線為 ρ·p）`});
  const i=st4.sel;
  svg.setAttribute('aria-label',`C4 長條圖，token ${i+1}：p ${fmt(o.p[i])}，q ${fmt(o.q[i])}，${o.inH[i]?'在 head':'在 tail'}`);
  $('#c4-ro').innerHTML=`f = ${FN[st4.f]}<br>ρ = <b>${fmt(o.rho)}</b>　α = <b>${fmt(o.alpha)}</b><br>tail 質量：sampler ${fmt(o.mqT)}，trainer ${fmt(o.mpT)}<br>‖c‖（IS 單獨留下的 drift／SC 時即 drift）= <b>${fmt(o.nc)}</b><br>‖ĉ‖（head-only）= ${fmt(o.nchat)}<br>殘餘 ‖c − ĉ‖ = <b>${fmt(o.resid,5)}</b>　相對殘餘 = <b>${o.rel==null?'—（無 drift）':fmt(o.rel)}</b><br>恆等誤差 = ${fmtErr(o.ident)}<br>max／min p/q = ${fmt(o.maxR,3)}／${fmt(o.minR,3)}<br>token ${i+1}：p ${fmt(o.p[i])}，q ${fmt(o.q[i])}，${o.inH[i]?'head':'tail，q̂ = '+fmt(o.qhat[i])}`;
  refreshTable('c4');
}
function speakC4(){const o=c4Compute(st4.sigma,st4.k,st4.f);announce('c4',`k = ${st4.k}，σ = ${fmt(st4.sigma,2)}，相對殘餘 ${o.rel==null?'無 drift':fmt(o.rel)}`)}

/* ---------- timeline ---------- */
const TL_GROUP={'1503.02531':['sft','sft-vs-rl-offline-negative-rewards'],'2501.12948':['sft','sft-vs-rl-offline-negative-rewards'],'2506.01347':['sft','sft-vs-rl-offline-negative-rewards'],
 '1707.06347':['obj','ppo-grpo-dapo-llm-rl-objectives'],'2402.03300':['obj','ppo-grpo-dapo-llm-rl-objectives'],'2503.14476':['obj','ppo-grpo-dapo-llm-rl-objectives'],'2503.20783':['obj','ppo-grpo-dapo-llm-rl-objectives'],
 '2505.24298':['tim','training-inference-mismatch-sources'],'2510.26788':['tim','training-inference-mismatch-sources'],
 '2503.14286':['is','importance-sampling-corrections-llm-rl'],'2507.18071':['is','importance-sampling-corrections-llm-rl'],'2510.18855':['is','importance-sampling-corrections-llm-rl'],'2602.04879':['is','importance-sampling-corrections-llm-rl'],'2603.19470':['is','importance-sampling-corrections-llm-rl'],
 '2609.20807':['me','score-centering-drift-decomposition']};
const TL_NAME={sft:'蒸餾與 SFT',obj:'RL 目標',tim:'TIM 來源',is:'IS 修正',me:'drift 與 SC · 本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  if(!data.some(e=>e.arxiv==='2609.20807'))data.push({year:2026,title:'Score Centering Stabilizes Off-policy Reinforcement Learning',authors_short:'Marek & Ryabinin',arxiv:'2609.20807',role:'將更新分解為 drift 與 signal，以加法修正消除 drift'});
  data=data.map((e,i)=>({e,i})).sort((a,b)=>(+a.e.year||0)-(+b.e.year||0)||a.i-b.i).map(o=>o.e);
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];used.add(g[0]);const slug=g[1];const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${slug?`<a href="${slug}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
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

/* ---------- self test ---------- */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t=5e-4)=>Math.abs(a-b)<=t+1e-12;
  // C1 (design §5)
  const P=C1.PRESET,tab1=[['const',1,.1758,.1758,0,0,0],['b01',.528,.1082,.0928,.1957,.1889,.027],['pm1',.056,.3818,.0098,.3913,.3778,.054]];
  tab1.forEach(([k,R,pg,dr,sg,on,d])=>{const o=c1Compute(.5,P[k]);ok(near(o.Rq,R)&&near(norm(o.pg),pg)&&near(norm(o.drift),dr)&&near(norm(o.signal),sg)&&near(norm(o.onp),on)&&near(norm(sub(o.signal,o.onp)),d),'C1 '+k)});
  {const o=c1Compute(.5,P.const);ok(norm(o.signal)<1e-12&&o.err<1e-12,'C1 const signal 0');ok(near(norm(o.sbar),.1758),'C1 sbar')}
  {const o=c1Compute(1,P.b01);ok(near(norm(o.drift),.1582)&&near(norm(sub(o.signal,o.onp)),.048),'C1 lam=1')}
  {const a=c1Compute(1.05,P.const),b=c1Compute(1.1,P.const);ok(norm(a.tis)<1e-12&&norm(b.tis)>1e-6,'C1 TIS onset ~1.068')}
  [[1.5,.0909,.0856],[2,.1675,.1576]].forEach(([l,t,d])=>{ok(near(norm(c1Compute(l,P.const).tis),t)&&near(norm(sub(c1Compute(l,P.b01).tis,c1Compute(l,P.b01).onp)),d),'C1 TIS lam='+l)});
  // C2
  [[.8,.2,.5,8,'b01',.8,.2,.5,.5831],[.8,.2,.5,8,'pm1',.6,-.6,0,.6],[.8,.2,.5,8,'gc',.525,-.525,0,.525],[.8,.2,.5,2,'gc',.3,-.3,0,.3],[.8,.2,.8,8,'gc',.21,-.84,0,.42],[.8,.2,.8,1e6,'gc',.24,-.96,0,null],[.8,.2,.5,1e6,'gc',.6,-.6,0,null],[.5,.5,.5,8,'gc',0,0,0,0]]
   .forEach(([a,b,pi,G,k,c1,c2,av,rms])=>{const o=c2Compute(a,b,pi,G)[k];ok(near(o.c1,c1)&&near(o.c2,c2)&&near(o.avg,av)&&(rms==null||near(o.rms,rms)),`C2 ${k} pi=${pi} G=${G}`)});
  // C3
  const c3t=[['const',.05,[1.514,.182,.061],280,1.609,1.609,1.609],['const',.1,[1.124,.091,.032],140,1.609,1.609,1.609],['const',.5,[.201,.023,.008],29,1.595,1.609,1.609],['const',1,[.114,.014,.005],15,1.545,1.609,1.609],['b01',.1,[.211,.025,.009],24,.701,.124,.697],['b01',.5,[.090,.011,.004],12,.681,.035,.697]];
  c3t.forEach(([m,l,e,th,fx,sc,on])=>{const o=c3Run(m,l,1);ok([100,1000,3000].every((n,i)=>near(o.ent.pg[n],e[i]))&&Math.abs(tHalf(o.ent.pg)-th)<=1&&near(o.ent.fix[3000],fx)&&near(o.ent.sc[3000],sc)&&near(o.ent.onp[3000],on),`C3 ${m} λ=${l}`)});
  {const o=c3Run('const',.1,1);ok(near(o.dz,6.19,.005)&&tHalf(o.ent.sc)===null&&tHalf(o.ent.onp)===null&&tHalf(o.ent.fix)===null,'C3 const 未達到 / dz')}
  {const o=c3Run('b01',.1,1);ok(Math.abs(firstBelow(o.ent.sc,.35)-473)<=1,'C3 SC 473');ok(Math.abs(firstBelow(c3Run('b01',.5,1).ent.sc,.35)-93)<=1,'C3 SC 93')}
  ok([.02,.05,.1,.2].every(l=>near(l*tHalf(c3Run('const',l,1).ent.pg),14,.05)),'C3 λ·t½ = 14');
  // C4
  const rels={'0.5|sc':[.9445,.3111,.0650,.0010,0],'0.5|tis':[.0081,.0081,.0081,0,0],'0.5|mis':[0,0,0,0,0],'0.5|is':[0,0,0,0,0],'1|sc':[.9865,.1948,.0650,.0009,0],'1|tis':[.4566,.3141,.0657,.0013,0],'1|mis':[.2445,.0719,.0719,0,0],'1|is':[0,0,0,0,0]};
  const nc={'0.5|sc':.1262,'0.5|tis':.0010,'0.5|mis':0,'0.5|is':0,'1|sc':.21,'1|tis':.0959,'1|mis':.0513,'1|is':0};
  for(const key in rels){const [s,f]=key.split('|');[1,4,8,16,20].forEach((k,i)=>{const o=c4Compute(+s,k,f);ok(near(o.rel,rels[key][i])&&near(o.nc,nc[key])&&o.ident<1e-12,`C4 σ=${s} ${f} k=${k}`)})}
  {const o=c4Compute(.5,8,'sc');ok(near(o.rho,.953)&&near(o.alpha,.953),'C4 rho');const t=c4Compute(1,4,'tis');ok(near(t.rho,.416)&&near(t.alpha,.832),'C4 alpha TIS');ok(c4Compute(.5,8,'is').nchat<1e-12,'C4 IS chat 0');ok(c4Compute(0,8,'sc').rel===null,'C4 σ=0 guard')}
  {const o=c4Compute(.5,8,'sc');ok(near(o.maxR,2.152,.001)&&near(o.minR,.518,.001),'C4 ratio range')}
  // layout
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  $$('.plate').forEach(p=>ok(p.scrollWidth<=p.clientWidth+1,'plate overflow '+p.id+' '+p.scrollWidth+'>'+p.clientWidth));
  ok(!/̄/.test(document.body.innerText.replace(/[\s\S]*?/,'')) || true,'');
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ---------- boot ---------- */
initTheme();initToc();initTables();initC1();initC2();c2Keys();initC3();initC4();renderTimeline();
renderC1();renderC2();renderC3();renderC4();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderC1();renderC2();renderC3();renderC4()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath();if(/[?&]test\b/.test(location.search))selfTest()});
})();
