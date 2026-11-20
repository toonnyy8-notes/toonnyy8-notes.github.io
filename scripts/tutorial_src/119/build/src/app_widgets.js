/* ================= shared math (design.md 4, verify/check.py) ================= */
function softmax(u){const m=Math.max(...u),e=u.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s)}
function logSoftmax(u){const m=Math.max(...u),l=m+Math.log(u.reduce((a,v)=>a+Math.exp(v-m),0));return u.map(v=>v-l)}
function kl(a,b){let s=0;for(let i=0;i<a.length;i++)if(a[i]>=1e-15)s+=a[i]*Math.log(a[i]/b[i]);return s}
function fmt(v,d){return(Math.abs(v)<.5*10**-d?0:v).toFixed(d).replace('-','−')}
const tdc=(h,v,last)=>`<td data-h="${h}">${v}${last?'':'<span class="sr">；</span>'}</td>`;

/* ================= C1 top-M + tail ================= */
const U1=[2.0,1.2,0.6,0.1,-0.3,-0.8,-1.2,-1.5],Z1=[0.2,1.5,-0.5,1.0,0.4,-1.0,0.8,-0.6];
function c1(s,M,perm){const pT=softmax(U1);let pS=softmax(U1.map((u,i)=>u+s*(Z1[i]-u)));
  const order=[...U1.keys()].sort((a,b)=>U1[b]-U1[a]||a-b),sup=order.slice(0,M).sort((a,b)=>a-b),tail=[...U1.keys()].filter(i=>!sup.includes(i));
  if(perm&&tail.length>=2){const v=tail.map(i=>pS[i]).reverse();pS=pS.slice();tail.forEach((i,k)=>{pS[i]=v[k]})}
  let tailS=0,tailT=0;tail.forEach(i=>{tailS+=pS[i];tailT+=pT[i]});
  let revTr=0;sup.forEach(i=>{revTr+=pS[i]*Math.log(pS[i]/pT[i])});if(tailS>=1e-15)revTr+=tailS*Math.log(tailS/tailT);
  return{pS,pT,sup,tail,tailS,tailT,rev:kl(pS,pT),fwd:kl(pT,pS),revTr}}
const c1s={s:20,M:3,perm:false};
function renderC1(){const s=c1s.s/20,M=c1s.M,ps=$('#c1-perm'),tailN=8-M;ps.disabled=tailN<2;if(ps.disabled){ps.checked=false;c1s.perm=false}
  $('#c1-perm-l').textContent=ps.disabled?'反轉學生在尾部的機率分配（尾部只有 0–1 個 token，無法重新分配）':'反轉學生在尾部的機率分配';
  const o=c1(s,M,c1s.perm);$('#c1-so').textContent=s.toFixed(2);$('#c1-mo').textContent=String(M);
  $('#c1-s').setAttribute('aria-valuetext','s = '+s.toFixed(2));$('#c1-m').setAttribute('aria-valuetext','M = '+M);
  const rowH=30,H=(8+1)*rowH+28,{svg,W}=frame($('#c1-fig'),280,640,H),L=72,R=W-58,x=lin(0,.55,L,R);
  const ord=[...o.sup,...o.tail];let y=10;
  ord.forEach((i,k)=>{const inS=k<o.sup.length,g=el('g',{opacity:inS?1:.45},svg);
    tx(g,4,y+15,'v'+(i+1),{class:'lab13'});if(inS)tx(g,30,y+15,'前M',{class:'lab'});
    el('rect',{x:L,y:y+2,width:Math.max(1,x(Math.min(o.pT[i],.55))-L),height:11,class:'t-f'},g);
    el('rect',{x:L,y:y+14,width:Math.max(1,x(Math.min(o.pS[i],.55))-L),height:11,class:'s-f'},g);
    tx(g,W-4,y+12,fmt(o.pT[i],3),{class:'lab','text-anchor':'end'});tx(g,W-4,y+25,fmt(o.pS[i],3),{class:'lab','text-anchor':'end'});
    if(k===o.sup.length-1&&o.tail.length)el('line',{x1:4,x2:W-4,y1:y+rowH-1,y2:y+rowH-1,class:'grid'},svg);y+=rowH});
  if(o.tail.length){tx(svg,4,y+15,'⊥ 合併尾部',{class:'lab13'});y+=4;
    el('rect',{x:L+20,y:y+16,width:Math.max(1,x(Math.min(o.tailT,.55))-L),height:11,class:'t-f'},svg);
    el('rect',{x:L+20,y:y+28,width:Math.max(1,x(Math.min(o.tailS,.55))-L),height:11,class:'s-f'},svg);
    tx(svg,W-4,y+26,fmt(o.tailT,3),{class:'lab','text-anchor':'end'});tx(svg,W-4,y+39,fmt(o.tailS,3),{class:'lab','text-anchor':'end'})}
  else tx(svg,4,y+15,'⊥ 尾部為空（M = 8）',{class:'lab'});
  dl($('#c1-read'),[['完整 reverse KL',fmt(o.rev,4)],['完整 forward KL',fmt(o.fwd,4)],['截斷 reverse KL',fmt(o.revTr,4),1],['截斷 − 完整',fmt(o.revTr-o.rev,4)],['尾部質量（學生）',fmt(o.tailS,4)],['尾部質量（老師）',fmt(o.tailT,4)]]);
  $('#c1-tab').innerHTML='<thead><tr><th>token</th><th>老師</th><th>學生</th><th>位置</th></tr></thead><tbody>'+[...U1.keys()].map(i=>'<tr>'+tdc('token','v'+(i+1))+tdc('老師',fmt(o.pT[i],4))+tdc('學生',fmt(o.pS[i],4))+tdc('位置',o.sup.includes(i)?'前 M':'尾部',1)+'</tr>').join('')+'</tbody>';
  const msg=`M = ${M}，截斷 reverse KL ${o.revTr.toFixed(3)}，完整 ${o.rev.toFixed(3)}`;announce('c1',msg);setLabel(svg,'c1','8 個 token 的老師與學生機率；'+msg)}
function initC1(){$('#c1-s').addEventListener('input',e=>{c1s.s=+e.target.value;renderC1()});$('#c1-m').addEventListener('input',e=>{c1s.M=+e.target.value;renderC1()});
  $('#c1-perm').addEventListener('change',e=>{c1s.perm=e.target.checked;renderC1()});renderC1()}

/* ================= C2 delta ledger ================= */
const P=[.40,.30,.20,.10],Q0=[.55,.15,.25,.05];
function c2(s,rb){const q=softmax(P.map((p,i)=>(1-s)*Math.log(p)+s*Math.log(Q0[i]))),Qa=rb.map(r=>2*r-1),d=P.map((p,i)=>Math.log(q[i]/p));
  const Ed=P.reduce((a,p,i)=>a+p*d[i],0),K=kl(P,q),Qb=P.reduce((a,p,i)=>a+p*Qa[i],0),c=P.reduce((a,p,i)=>a+p*Qa[i]*d[i],0),cov=P.reduce((a,p,i)=>a+p*(Qa[i]-Qb)*d[i],0);
  return{q,Qa,d,Ed,K,Qb,c,cov,first:-Qb*K}}
const c2s={s:20,r:[16,6,12,2]};let c2PreSync=null;
const C2PRE={'預設':[16,6,12,2],'全部 0.5':[10,10,10,10],'全部 0.9':[18,18,18,18]};
function renderC2(){const s=c2s.s/20,rb=c2s.r.map(v=>v/20),o=c2(s,rb);$('#c2-so').textContent=s.toFixed(2);
  rb.forEach((r,i)=>{$('#c2-r'+i+'o').textContent=r.toFixed(2);$('#c2-r'+i).setAttribute('aria-valuetext','期望獎勵 '+r.toFixed(2))});
  const pre=Object.keys(C2PRE).find(k=>C2PRE[k].every((v,i)=>v===c2s.r[i]));c2PreSync(pre||null);
  $('#c2-tab').innerHTML='<thead><tr><th>token</th><th>p</th><th>q</th><th>Q = 2r − 1</th><th>δ</th><th>p·δ</th><th>p·Q·δ</th></tr></thead><tbody>'+P.map((p,i)=>'<tr>'+tdc('token','v'+(i+1))+tdc('p',fmt(p,3))+tdc('q',fmt(o.q[i],4))+tdc('Q',fmt(o.Qa[i],2))+tdc('δ',(o.d[i]>5e-5?'＋':o.d[i]<-5e-5?'−':'')+fmt(Math.abs(o.d[i]),4))+tdc('p·δ',fmt(p*o.d[i],4))+tdc('p·Q·δ',fmt(p*o.Qa[i]*o.d[i],4),1)+'</tr>').join('')+'</tbody>';
  const H=4*30+30,{svg,W}=frame($('#c2-fig'),280,640,H),mid=W/2,x=lin(-1,1,mid-(W/2-60),mid+(W/2-60));
  el('line',{x1:mid,x2:mid,y1:4,y2:H-22,class:'axis'},svg);tx(svg,mid,H-6,'δ = log q − log p（向右為正）',{class:'lab','text-anchor':'middle'});
  o.d.forEach((d,i)=>{const y=8+i*30,dd=Math.max(-1,Math.min(1,d));el('rect',{x:Math.min(mid,x(dd)),y,width:Math.max(1,Math.abs(x(dd)-mid)),height:16,class:d>=0?'t-f':'s-f'},svg);
    tx(svg,4,y+13,'v'+(i+1),{class:'lab13'});tx(svg,d>=0?Math.min(W-4,x(dd)+4):Math.max(30,x(dd)-4),y+13,(d>=0?'＋':'−')+fmt(Math.abs(d),3),{class:'lab','text-anchor':d>=0?'start':'end'})});
  dl($('#c2-read'),[['E[δ]',fmt(o.Ed,4)],['−KL',fmt(-o.K,4)],['Q̄',fmt(o.Qb,4)],['c = Σ p·Q·δ',fmt(o.c,4),1],['第一項 −Q̄·KL',fmt(o.first,4)],['共變項',fmt(o.cov,4)],['第一項＋共變項',fmt(o.first+o.cov,4)]]);
  const msg=`s = ${s.toFixed(2)}，E[δ] ${o.Ed.toFixed(3)}，c ${o.c.toFixed(3)}，共變項 ${o.cov.toFixed(3)}`;announce('c2',msg);setLabel(svg,'c2','每個 token 的 δ；'+msg)}
function initC2(){$('#c2-s').addEventListener('input',e=>{c2s.s=+e.target.value;renderC2()});[0,1,2,3].forEach(i=>$('#c2-r'+i).addEventListener('input',e=>{c2s.r[i]=+e.target.value;renderC2()}));
  c2PreSync=segInit($('#c2-pre'),Object.keys(C2PRE),'預設',null,k=>{c2s.r=C2PRE[k].slice();c2s.r.forEach((v,i)=>{$('#c2-r'+i).value=v});renderC2()});renderC2()}

/* ================= C3 teacher dynamics ================= */
const PRESETS={A:[.8,.3,.6,.1],B:[.5,.5,.5,.5],C:[.7,.7,.7,0]},BMAX=5,T3=600,c3Cache=new Map();
function c3(preset,mode,beta,m){const key=`${preset}|${mode}|${beta}|${m}`;if(c3Cache.has(key))return c3Cache.get(key);
  const Qa=PRESETS[preset].map(r=>2*r-1),Qb=P.reduce((a,p,i)=>a+p*Qa[i],0),lp=P.map(Math.log);let u=Q0.map(Math.log),b=mode==='fixed'?beta:0,capped=false;const hist=[];
  for(let i=0;i<=T3;i++){const lq=logSoftmax(u),q=lq.map(Math.exp);let K=0,D=0;for(let k=0;k<4;k++){K+=P[k]*(lp[k]-lq[k]);D+=P[k]*Qa[k]*(lq[k]-lp[k])}
    hist.push({q,KL:K,D,b});if(i===T3)break;
    const g=q.map((qk,k)=>(qk-P[k])-b*(P[k]*Qa[k]-qk*Qb));u=u.map((v,k)=>v-g[k]);const mx=Math.max(...u);u=u.map(v=>Math.max(-30,v-mx));
    if(mode==='dual'){const nb=Math.max(0,b+.5*(m-D));if(nb>=BMAX)capped=true;b=Math.min(BMAX,nb)}}
  let w=P.map((p,k)=>Math.max(0,p*(1+hist[T3].b*Qa[k])));const sw=w.reduce((a,v)=>a+v,0);const qstar=sw>0?w.map(v=>v/sw):P.slice();
  const zero=w.map((v,k)=>v===0?k:-1).filter(k=>k>=0);const r={hist,qstar,capped,zero};c3Cache.set(key,r);return r}
function barsC3(box,st,qstar){const H=150,{svg,W}=frame(box,280,640,H),L=30,R=W-8,gw=(R-L)/4,y=lin(0,.7,H-24,8);
  [0,.2,.4,.6].forEach(t=>{el('line',{x1:L,x2:R,y1:y(t),y2:y(t),class:'grid'},svg);tx(svg,L-4,y(t)+4,t.toFixed(1),{class:'lab','text-anchor':'end'})});
  for(let k=0;k<4;k++){const x0=L+k*gw+gw*.2,bw=gw*.6;el('rect',{x:x0+bw*.2,y:y(st.q[k]),width:bw*.6,height:y(0)-y(st.q[k]),class:'t-f'},svg);
    el('rect',{x:x0,y:y(P[k]),width:bw,height:y(0)-y(P[k]),class:'hollow-r'},svg);
    if(qstar)el('line',{x1:x0-4,x2:x0+bw+4,y1:y(qstar[k]),y2:y(qstar[k]),class:'qs-l'},svg);
    tx(svg,x0+bw/2,H-6,'v'+(k+1),{class:'lab','text-anchor':'middle'})}return svg}
function lineC3(box,arr,cur,{ymin,ymax,label,ref,fmtv}){const H=120,{svg,W}=frame(box,280,640,H),L=44,R=W-8,x=lin(0,T3,L,R),y=lin(ymin,ymax,H-22,10);
  [ymin,(ymin+ymax)/2,ymax].forEach(t=>{el('line',{x1:L,x2:R,y1:y(t),y2:y(t),class:'grid'},svg);tx(svg,L-4,y(t)+4,fmtv(t),{class:'lab','text-anchor':'end'})});
  if(ref!=null){el('line',{x1:L,x2:R,y1:y(ref),y2:y(ref),class:'ref-s'},svg);tx(svg,R,y(ref)-4,'m',{class:'lab','text-anchor':'end'})}
  el('path',{d:lp(arr.map((v,i)=>[x(i),y(Math.max(ymin,Math.min(ymax,v)))])),class:'acc-s'},svg);
  el('line',{x1:x(cur),x2:x(cur),y1:8,y2:H-20,class:'axis'},svg);tx(svg,L,H-6,label,{class:'lab'});tx(svg,R,H-6,'第 '+T3+' 步',{class:'lab','text-anchor':'end'});return svg}
const c3s={pre:'A',mode:'fixed',beta:0,m:5,k:600};let c3PreSync,c3ModeSync;
function setStepC3(n){if(n===1){c3s.mode='fixed';c3s.beta=0}else if(n===2){c3s.mode='fixed';c3s.beta=10}else{c3s.mode='dual';c3s.m=5}c3s.pre='A';c3PreSync('A');c3ModeSync(c3s.mode);$('#c3-beta').value=c3s.beta;$('#c3-m').value=c3s.m;renderC3()}
function renderC3(){const beta=c3s.beta/20,m=c3s.m/100,r=c3(c3s.pre,c3s.mode,beta,m),st=r.hist[c3s.k];
  $('#c3-bwrap').classList.toggle('hide',c3s.mode!=='fixed');$('#c3-mwrap').classList.toggle('hide',c3s.mode!=='dual');
  $('#c3-betao').textContent=beta.toFixed(2);$('#c3-mo').textContent=m.toFixed(2);$('#c3-ko').textContent=String(c3s.k);
  $('#c3-beta').setAttribute('aria-valuetext','β = '+beta.toFixed(2));$('#c3-m').setAttribute('aria-valuetext','m = '+m.toFixed(2));$('#c3-k').setAttribute('aria-valuetext','第 '+c3s.k+' 步');
  const step=c3s.mode==='fixed'?(beta===0?1:beta===.5&&c3s.pre==='A'?2:0):(m===.05&&c3s.pre==='A'?3:0);$$('#c3-steps button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.s===step)));
  const svg=barsC3($('#c3-bars'),st,r.qstar);const Ds=r.hist.map(h=>h.D),Bs=r.hist.map(h=>h.b);
  lineC3($('#c3-d'),Ds,c3s.k,{ymin:Math.min(-.05,...Ds),ymax:Math.max(.35,...Ds),label:'Δ',ref:c3s.mode==='dual'?m:null,fmtv:v=>fmt(v,2)});
  const bmax=Math.max(1,...Bs);lineC3($('#c3-b'),Bs,c3s.k,{ymin:0,ymax:bmax,label:'β',ref:null,fmtv:v=>fmt(v,1)});
  const flags=[];if(r.capped)flags.push('β 已達上限 5（本頁保護）。');r.zero.forEach(k=>flags.push(`q* 在 v${k+1} 為 0：老師 logit 只能漸近地下降，600 步內不會到 0。`));$('#c3-flag').textContent=flags.join(' ');
  dl($('#c3-read'),[['步數',String(c3s.k)],['KL',fmt(st.KL,4)],['Δ',fmt(st.D,4),1],['β',fmt(st.b,4)+(r.capped&&st.b>=BMAX?'（上限）':'')],['q',st.q.map(v=>fmt(v,4)).join(', ')],['q*',r.qstar.map(v=>fmt(v,4)).join(', ')]]);
  const msg=`第 ${c3s.k} 步，Δ = ${st.D.toFixed(3)}，β = ${st.b.toFixed(3)}`;announce('c3',msg);setLabel(svg,'c3','四個 token 的 p、q、q*；'+msg)}
function initC3(){c3PreSync=segInit($('#c3-pre'),['A','B','C'],'A',k=>'設定 '+k,k=>{c3s.pre=k;renderC3()});
  c3ModeSync=segInit($('#c3-mode'),['fixed','dual'],'fixed',k=>k==='fixed'?'固定':'對偶更新',k=>{c3s.mode=k;renderC3()});
  $('#c3-beta').addEventListener('input',e=>{c3s.beta=+e.target.value;renderC3()});$('#c3-m').addEventListener('input',e=>{c3s.m=+e.target.value;renderC3()});$('#c3-k').addEventListener('input',e=>{c3s.k=+e.target.value;renderC3()});
  $$('#c3-steps button').forEach(b=>b.addEventListener('click',()=>setStepC3(+b.dataset.s)));renderC3()}
const c3ls={k:600};
function renderC3lite(){const r=c3('A','fixed',0,0),st=r.hist[c3ls.k];$('#c3l-ko').textContent=String(c3ls.k);$('#c3l-k').setAttribute('aria-valuetext','第 '+c3ls.k+' 步');
  const svg=barsC3($('#c3l-bars'),st,null);const Ks=r.hist.map(h=>h.KL);lineC3($('#c3l-kl'),Ks,c3ls.k,{ymin:0,ymax:.12,label:'KL(p‖q)',ref:null,fmtv:v=>fmt(v,2)});
  dl($('#c3l-read'),[['步數',String(c3ls.k)],['KL',fmt(st.KL,4),1],['q',st.q.map(v=>fmt(v,4)).join(', ')],['p',P.map(v=>fmt(v,2)).join(', ')]]);
  const msg=`第 ${c3ls.k} 步，KL = ${st.KL.toFixed(4)}`;announce('c3l',msg);setLabel(svg,'c3l','學生與老師的機率；'+msg)}
function initC3lite(){$('#c3l-k').addEventListener('input',e=>{c3ls.k=+e.target.value;renderC3lite()});renderC3lite()}

/* ================= C4 score -> counts ================= */
const ROWS=[['q4-bfcl','Qwen3-4B','BFCL-v3 Avg',800,27.38,'GRPO',25.25],['q8-bfcl','Qwen3-8B','BFCL-v3 Avg',800,29.88,'GRPO',29.00],
 ['q4-lcb5','Qwen3-4B','LiveCodeBench v5',279,50.54,'GRPO',50.18],['q4-lcb6','Qwen3-4B','LiveCodeBench v6',131,45.04,'GRPO',44.27],['q4-lcb','Qwen3-4B','LiveCodeBench Avg',410,48.78,'GRPO',48.29],
 ['q4-he','Qwen3-4B','HumanEval+',164,87.80,'SDFT',87.20],['q4-mbpp','Qwen3-4B','MBPP+',378,78.57,'Skill-SD／SDFT',76.98],['q4-ep','Qwen3-4B','EvalPlus Avg',542,81.36,'SDFT',80.07],
 ['o7-lcb5','Olmo3-7B','LiveCodeBench v5',279,53.41,'SDPO',49.82],['o7-lcb6','Olmo3-7B','LiveCodeBench v6',131,45.80,'GRPO',45.80],['o7-lcb','Olmo3-7B','LiveCodeBench Avg',410,50.98,'GRPO',48.29],
 ['o7-he','Olmo3-7B','HumanEval+',164,90.24,'GRPO／SDFT',89.02],['o7-mbpp','Olmo3-7B','MBPP+',378,73.28,'SDFT',73.02],['o7-ep','Olmo3-7B','EvalPlus Avg',542,78.41,'SDFT',77.86]];
function c4(row){const[, , ,N,a, ,b]=row,na=Math.round(a*N/100),nb=Math.round(b*N/100),ph=a/100;return{na,nb,dn:na-nb,dpp:a-b,se:100*Math.sqrt(ph*(1-ph)/N)}}
const c4s={id:'q4-lcb5'};
function renderC4(){const row=ROWS.find(r=>r[0]===c4s.id),[, ,bench,N,a,bn,b]=row,o=c4(row);
  dl($('#c4-read'),[['LOPD',`${a.toFixed(2)}% ＝ ${o.na} / ${N} 題`],[`最強基線（${bn}）`,`${b.toFixed(2)}% ＝ ${o.nb} / ${N} 題`],['差距',`${o.dn} 題（${o.dpp.toFixed(2)} 個百分點）`,1],['LOPD 分數的單一分數 SE（題目取樣，非配對檢定）',`${o.se.toFixed(2)} 個百分點`]]);
  const H=92,{svg,W}=frame($("#c4-fig"),280,640,H),lo=Math.min(a-3*o.se,b-.5*o.se),hi=Math.max(a+3*o.se,b+.5*o.se),L=14,R=W-14,x=lin(lo,hi,L,R),y0=48;
  el('line',{x1:L,x2:R,y1:y0+18,y2:y0+18,class:'axis'},svg);[lo,(lo+hi)/2,hi].forEach(t=>{el('line',{x1:x(t),x2:x(t),y1:y0+18,y2:y0+22,class:'axis'},svg);tx(svg,x(t),y0+36,t.toFixed(1)+'%',{class:'lab','text-anchor':t===lo?'start':t===hi?'end':'middle'})});
  el('line',{x1:x(a-1.96*o.se),x2:x(a+1.96*o.se),y1:y0,y2:y0,class:'ink-s'},svg);[a-1.96*o.se,a+1.96*o.se].forEach(t=>el('line',{x1:x(t),x2:x(t),y1:y0-6,y2:y0+6,class:'ink-s'},svg));
  el('circle',{cx:x(a),cy:y0,r:6,class:'ink-f'},svg);el('rect',{x:x(b)-5,y:y0-19,width:10,height:10,transform:`rotate(45 ${x(b)} ${y0-14})`,class:'dia-r'},svg);
  tx(svg,L,12,'LOPD 單一分數的題目取樣範圍（±1.96 SE）',{class:'lab'});
  const msg=`${bench}：差距 ${o.dn} 題，單一分數 SE ${o.se.toFixed(2)} 個百分點`;announce('c4',msg);setLabel(svg,'c4',msg)}
function initC4(){const s=$('#c4-row');s.innerHTML=ROWS.map(r=>`<option value="${r[0]}"${r[0]===c4s.id?' selected':''}>${r[1]} · ${r[2]}（N = ${r[3]}）</option>`).join('');s.addEventListener('change',e=>{c4s.id=e.target.value;renderC4()});renderC4()}

/* ================= full tables ================= */
const T1F=[['Qwen3-4B · Vanilla',48.6,'28.50','20.50','20.50','22.00','22.88',48.3,52.2,50.6],['Qwen3-4B · GRPO',61.8,'33.00','20.50','24.50','23.00','25.25',58.3,54.4,'56.0'],['Qwen3-4B · Skill-SD',59.1,'33.00','19.00','24.00','22.50','24.63',56.6,55.6,'56.0'],['Qwen3-4B · OPSD',51.2,'31.50','19.50','20.50','29.00','25.13',41.6,53.3,48.6],['Qwen3-4B · SDPO',50.6,'21.50','14.00','13.50','14.00','15.75','45.0',33.3,'38.0'],['Qwen3-4B · SDFT',50.3,'31.50','17.00','17.00','21.50','21.75',53.3,47.8,'50.0'],['Qwen3-4B · LOPD',63.7,'32.50','26.50','24.50','26.00','27.38',56.6,63.3,60.6],
 ['Qwen3-8B · Vanilla',49.2,'34.00','28.50','25.50','25.50','28.38',61.6,'50.0',54.6],['Qwen3-8B · GRPO',57.3,'36.00','28.00','27.00','25.00','29.00','65.0',53.3,'58.0'],['Qwen3-8B · Skill-SD',60.2,'33.00','27.00','25.00','24.50','27.38',63.3,51.1,'56.0'],['Qwen3-8B · OPSD','52.0','30.50','25.00','23.50','24.00','25.75','60.0',47.8,52.7],['Qwen3-8B · SDPO',55.3,'29.00','24.50','23.00','23.50','25.00',56.7,48.9,'52.0'],['Qwen3-8B · SDFT',56.2,'31.50','26.00','24.50','25.50','26.88','65.0',47.8,54.7],['Qwen3-8B · LOPD',66.4,'37.00','29.00','27.50','26.00','29.88',73.3,55.6,62.7]];
const T2F=[['Qwen3-4B · Vanilla','47.67','41.22','45.61','85.37','75.93','78.79'],['Qwen3-4B · GRPO','50.18','44.27','48.29','86.59','76.19','79.34'],['Qwen3-4B · Skill-SD','49.82','41.98','47.32','85.98','76.98','79.70'],['Qwen3-4B · OPSD','42.65','35.11','40.24','85.36','76.72','79.33'],['Qwen3-4B · SDPO','40.86','34.35','38.78','84.76','74.87','77.86'],['Qwen3-4B · SDFT','48.75','43.51','47.07','87.20','76.98','80.07'],['Qwen3-4B · LOPD','50.54','45.04','48.78','87.80','78.57','81.36'],
 ['Olmo3-7B · Vanilla','47.31','44.27','46.34','86.59','70.37','75.28'],['Olmo3-7B · GRPO','49.46','45.80','48.29','89.02','72.75','77.67'],['Olmo3-7B · Skill-SD','49.10','45.03','47.80','87.20','72.22','76.75'],['Olmo3-7B · OPSD','44.09','45.04','44.39','87.80','71.96','76.75'],['Olmo3-7B · SDPO','49.82','43.51','47.80','88.41','72.75','77.49'],['Olmo3-7B · SDFT','47.67','44.27','46.58','89.02','73.02','77.86'],['Olmo3-7B · LOPD','53.41','45.80','50.98','90.24','73.28','78.41']];
function fillTables(){const h1=['方法','Env','Base','Miss Func','Miss Param','Long Ctx','BFCL Avg','M-Step','M-Turn','ACE Avg'],h2=['方法','v5','v6','LCB Avg','HE+','MBPP+','EvalPlus Avg'];
  const rows=(T,h)=>T.map(r=>`<tr${/LOPD/.test(r[0])?' class="hl"':''}>`+r.map((v,i)=>tdc(h[i],String(v),i===r.length-1)).join('')+'</tr>').join('');
  $('#t1-full').innerHTML=rows(T1F,h1);$('#t2-full').innerHTML=rows(T2F,h2)}

/* ================= static figures (4.F) ================= */
function figHero(){const box=$('#fig-hero'),W=widthOf(box,280,760),wide=W>=480,H=wide?190:360,svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const bars=(g,x,y,w,cls,vals)=>{const bw=w/vals.length;vals.forEach((v,i)=>el('rect',{x:x+i*bw+2,y:y+40-v*40,width:bw-4,height:v*40,class:cls},g))};
  const sv=[.15,.5,.2,.1,.05],tv=[.6,.2,.1,.06,.04];
  if(wide){const bx=W*.36;[[16,'學生','任務＋互動歷史',sv,'s-f',false],[106,'老師（權重不更新）','任務＋互動歷史',tv,'t-f',true]].forEach(([y,who,lab,v,cls,lat])=>{
      tx(svg,0,y+10,who,{class:'lab13'});el('rect',{x:0,y:y+18,width:bx-20,height:40,rx:8,class:'box'},svg);tx(svg,10,y+43,lab,{class:'lab'});
      if(lat){const cw=(bx-20)*.36,x0=bx-20-cw-6;for(let j=0;j<3;j++)el('rect',{x:x0+j*(cw/3),y:y+24,width:cw/3-3,height:28,rx:3,class:'box-acc'},svg);tx(svg,x0+cw/2,y+72,'3 筆紀錄 × 32 組數字',{class:'lab','text-anchor':'middle'})}
      el('path',{d:`M${bx-14} ${y+38}H${bx+6}`,class:'arrow'},svg);bars(svg,bx+14,y+18,W*.3,cls,v);tx(svg,bx+14,y+72,'下一個詞的機率',{class:'lab'})});
    const xr=bx+14+W*.3+14;el('path',{d:`M${xr} 56V146`,class:'arrow'},svg);tx(svg,xr+8,96,'兩組機率的差異',{class:'lab'});tx(svg,xr+8,112,'（詳見 04）',{class:'lab'})}
  else{const w=W;[[0,'學生','任務＋互動歷史',sv,'s-f',false],[205,'老師（權重不更新）','任務＋互動歷史',tv,'t-f',true]].forEach(([y,who,lab,v,cls,lat])=>{
      tx(svg,0,y+12,who,{class:'lab13'});el('rect',{x:0,y:y+20,width:w*(lat?.55:.95),height:36,rx:8,class:'box'},svg);tx(svg,10,y+43,lab,{class:'lab'});
      if(lat){for(let j=0;j<3;j++)el('rect',{x:w*.58+j*(w*.12),y:y+24,width:w*.11,height:28,rx:3,class:'box-acc'},svg);tx(svg,w-2,y+70,'3 筆紀錄 × 32 組數字',{class:'lab','text-anchor':'end'})}
      bars(svg,0,y+(lat?76:62),w*.6,cls,v);tx(svg,w*.62,y+(lat?110:96),'下一個詞的機率',{class:'lab'})});
    el('path',{d:`M${W*.2} 132V188`,class:'arrow'},svg);tx(svg,W*.2+8,164,'兩組機率的差異（詳見 04）',{class:'lab'})}}
function figOPSD(){const box=$('#fig-opsd'),W=widthOf(box,280,860),wide=W>=480,colW=wide?(W-24)/2:W,colH=200,H=wide?colH:colH*2+20,svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const col=(x0,y0,title,teacher,note,acc)=>{tx(svg,x0,y0+14,title,{class:'lab13'});
    el('rect',{x:x0,y:y0+26,width:colW,height:44,rx:8,class:'box'},svg);tx(svg,x0+10,y0+44,'學生 π^S_θ：只看 s_t',{class:'lab'});tx(svg,x0+10,y0+62,'自己取樣軌跡 τ',{class:'lab'});
    el('rect',{x:x0,y:y0+86,width:colW,height:44,rx:8,class:acc?'box-acc':'box'},svg);tx(svg,x0+10,y0+104,teacher,{class:'lab'});tx(svg,x0+10,y0+122,'在同一個 prefix 重新評分',{class:'lab'});
    tx(svg,x0,y0+152,'兩個分布 → token 層級 divergence',{class:'lab'});tx(svg,x0,y0+172,note,{class:'lab'})};
  col(0,0,'OPSD（Eq. 2–3）','老師＝同權重 θ＋固定脈絡 c_fix','老師一側 stop-gradient',false);
  if(wide)col(colW+24,0,'LOPD（Eq. 7、10）','老師＝凍結 θ̄＋latent 脈絡 c_φ','無 stop-gradient：梯度回到 φ',true);else col(0,colH+20,'LOPD（Eq. 7、10）','老師＝凍結 θ̄＋latent 脈絡 c_φ','無 stop-gradient：梯度回到 φ',true)}
function figSlots(){const box=$('#fig-slots'),W=widthOf(box,280,860),wide=W>=480,svg=box.querySelector('svg');svg.textContent='';
  if(wide){const H=120,tw=Math.min(90,W*.14),gx=tw+8,gw=W-2*gx,cw=gw/96;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
    el('rect',{x:0,y:30,width:tw,height:26,rx:4,class:'box'},svg);tx(svg,tw/2,47,'t₁…tₐ',{class:'lab','text-anchor':'middle'});
    el('rect',{x:W-tw,y:30,width:tw,height:26,rx:4,class:'box'},svg);tx(svg,W-tw/2,47,'tₐ₊₁…',{class:'lab','text-anchor':'middle'});
    for(let i=0;i<96;i++)el('rect',{x:gx+i*cw,y:30,width:Math.max(.6,cw-.6),height:26,class:'box-acc'},svg);
    for(let j=0;j<3;j++)tx(svg,gx+(j+.5)*32*cw,22,'經驗 '+(j+1)+'（32 格）',{class:'lab','text-anchor':'middle'});
    tx(svg,W/2,82,'96 個 pad 佔位（Eq. 17）',{class:'lab','text-anchor':'middle'});tx(svg,W/2,104,'embedding 後以 latent 向量 ⟨e⟩ 替換（Eq. 18，torch.cat）',{class:'lab','text-anchor':'middle'})}
  else{const cw=(W-48)/32,H=34+3*36+58;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
    el('rect',{x:0,y:4,width:W,height:22,rx:4,class:'box'},svg);tx(svg,8,19,'前段文字 t₁…tₐ',{class:'lab'});
    for(let j=0;j<3;j++){const y=34+j*36;tx(svg,0,y+12,'經驗 '+(j+1),{class:'lab'});for(let i=0;i<32;i++)el('rect',{x:48+i*cw,y:y+2,width:Math.max(.6,cw-.6),height:24,class:'box-acc'},svg)}
    const y=34+3*36;el('rect',{x:0,y,width:W,height:22,rx:4,class:'box'},svg);tx(svg,8,y+15,'後段文字 tₐ₊₁ 之後',{class:'lab'});
    tx(svg,0,y+40,'96 個 pad 佔位，embedding 後以 latent 向量',{class:'lab'});tx(svg,0,y+56,'替換（Eq. 17–18，torch.cat）',{class:'lab'})}}
const F3=[['Frozen φ₀',.573],['m = 0',.551],['m = 0.01',.566],['m = 0.02',.603],['m = 0.05',.637],['m = 0.10',.626],['m = 0.20',.613]];
function figAblation(){const box=$('#fig-abl'),W=widthOf(box,280,760),wide=W>=480,rh=wide?28:44,H=F3.length*rh+40,svg=box.querySelector('svg');svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const L=wide?160:10,R=W-12,x=lin(.54,.65,L,R),yb=F3.length*rh+8;
  [.54,.57,.6,.63,.65].forEach(t=>{el('line',{x1:x(t),x2:x(t),y1:4,y2:yb,class:'grid'},svg);tx(svg,x(t),yb+18,t.toFixed(2),{class:'lab','text-anchor':t===.54?'start':t===.65?'end':'middle'})});
  el('line',{x1:x(.573),x2:x(.573),y1:4,y2:yb,class:'ref-s'},svg);
  F3.forEach(([k,v],i)=>{const y=i*rh+(wide?18:30);tx(svg,wide?0:L,wide?y+4:y-12,k+'　'+v.toFixed(3),{class:'lab13'});el('circle',{cx:x(v),cy:y,r:5,class:i===0?'hollow-c':'ink-f'},svg)});
  tx(svg,x(.573)+4,yb-4,'以凍結 composer 訓練的學生',{class:'lab'})}
function drawFigs(){figHero();figOPSD();figSlots();figAblation()}
function observeWidths(){const map=[['#fig-hero',figHero],['#fig-opsd',figOPSD],['#fig-slots',figSlots],['#fig-abl',figAblation],['#c1-fig',renderC1],['#c2-fig',renderC2],['#c3l-bars',renderC3lite],['#c3-bars',renderC3],['#c4-fig',renderC4]];
  const last=new Map(),tm=new Map();const ro=new ResizeObserver(es=>es.forEach(e=>{const w=Math.round(e.contentRect.width);if(last.get(e.target)===w)return;last.set(e.target,w);const f=map.find(m=>$(m[0])===e.target)[1];clearTimeout(tm.get(f));tm.set(f,setTimeout(f,150))}));
  map.forEach(([s])=>{const n=$(s);last.set(n,Math.round(n.getBoundingClientRect().width));ro.observe(n)})}
