<script>
(()=>{
'use strict';
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
function el(tag,a,p){const e=document.createElementNS(NS,tag);if(a)for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function tx(p,x,y,s,a){const t=el('text',Object.assign({x,y},a||{}),p);t.textContent=s;return t}
const lin=(d0,d1,r0,r1)=>v=>r0+(v-d0)/(d1-d0)*(r1-r0);
function lp(pts){let s='';for(let i=0;i<pts.length;i++)s+=(i?'L':'M')+pts[i][0].toFixed(1)+' '+pts[i][1].toFixed(1);return s}
const f3=v=>(Math.abs(v)<0.0005?0:v).toFixed(3),f4=v=>(Math.abs(v)<0.00005?0:v).toFixed(4);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function widthOf(box,min,max){return Math.max(min,Math.min(max,Math.floor(box.clientWidth||max)))}
function frame(svg,o){
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${o.W} ${o.H}`);svg.setAttribute('width',o.W);svg.setAttribute('height',o.H);
  const m=o.m,x0=m.l,x1=o.W-m.r,y0=o.H-m.b,y1=m.t;
  const X=lin(o.x[0],o.x[1],x0,x1),Y=lin(o.y[0],o.y[1],y0,y1);
  const g=el('g',null,svg);
  (o.yt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:(o.yf?o.yf(v):String(v)),yy=Y(val);el('line',{x1:x0,x2:x1,y1:yy,y2:yy,class:'grid'},g);tx(g,x0-8,yy+4,lab,{class:'tk','text-anchor':'end'})});
  (o.xt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:(o.xf?o.xf(v):String(v)),xx=X(val);el('line',{x1:xx,x2:xx,y1:y0,y2:y0+4,class:'axl'},g);tx(g,xx,y0+18,lab,{class:'tk','text-anchor':'middle'})});
  el('line',{x1:x0,x2:x1,y1:y0,y2:y0,class:'axl'},g);
  if(o.xl)tx(g,(x0+x1)/2,o.H-4,o.xl,{class:'al','text-anchor':'middle'});
  if(o.yl)tx(g,4,12,o.yl,{class:'al'});
  return {X,Y,g,x0,x1,y0,y1};
}
const live={};function announce(id,s,now){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);if(now){n.textContent=s;return}live[id]=setTimeout(()=>{n.textContent=s},500)}
function segInit(sel,cb){$$(sel+' button').forEach(b=>b.addEventListener('click',()=>{$$(sel+' button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));cb(b.dataset.v)}))}
function segSet(sel,v){$$(sel+' button').forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.v===String(v))))}
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}

/* ================= C1 · corruption and supervised positions (port of verify/c1.js) ================= */
const TOK=['for','k','in','range','(','2',',','n','-','1',')',':','if','n','%','k','==','0',':','return','False'];
const VOC=['while','x','+','7','def','*','len','3','<','None','total','j','/','or','5','pass','[',']','>=','is'];
const LAMN=0.1,CTW=0.1;
function corrupt(seed,lam,alpha,mode){const r=mulberry32(seed),out=[];
  for(let i=0;i<TOK.length;i++){const u1=r(),u2=r(),u3=r();
    if(u1<lam){out.push({role:'M',tok:'[M]'});continue}
    if(mode!=='mdlm'&&u2<alpha){let j=Math.floor(u3*VOC.length);if(VOC[j]===TOK[i])j=(j+1)%VOC.length;out.push({role:'N',tok:VOC[j]});continue}
    out.push({role:'C',tok:TOK[i]})}
  return out}
function weights(seq,mode){const cnt={M:0,N:0,C:0};seq.forEach(s=>cnt[s.role]++);
  const w=seq.map(s=>s.role==='M'?1/cnt.M:s.role==='N'?(mode==='mdlm'?0:LAMN/cnt.N):(mode==='ctw'?CTW/cnt.C:0));
  return {w,cnt,total:{M:cnt.M?1:0,N:(mode!=='mdlm'&&cnt.N)?LAMN:0,C:(mode==='ctw'&&cnt.C)?CTW:0}}}
const pEmptyAt=(lam,a,n)=>Math.pow(1-(1-lam)*a,n);
const pEmptyAvg=(a,n)=>a===0?1:-Math.expm1((n+1)*Math.log1p(-a))/(a*(n+1));
const c1={mode:'cdlm',lam:.3,alpha:.1,seed:1};
const ROLE={M:'遮罩',N:'被替換',C:'乾淨可見'};
function renderC1(){const {mode,lam,alpha,seed}=c1,seq=corrupt(seed,lam,alpha,mode),W=weights(seq,mode),mx=Math.max(...W.w);
  $('#c1-lamo').textContent=lam.toFixed(2);$('#c1-ao').textContent=mode==='mdlm'?'0（MDLM 組 α = 0）':alpha.toFixed(2);$('#c1-a').disabled=mode==='mdlm';$('#c1-seedo').textContent='seed = '+seed;
  $('#c1-lam').setAttribute('aria-valuetext','λ = '+lam.toFixed(2));$('#c1-a').setAttribute('aria-valuetext',mode==='mdlm'?'MDLM 組 α = 0':'α = '+alpha.toFixed(2));
  const cell=(s,i)=>{const w=W.w[i],h=w>0?Math.max(3,Math.sqrt(w/mx)*40):0,orig=TOK[i];
    const lab=`位置 ${i}：${ROLE[s.role]}${s.role==='N'?`，原字 ${orig}，現為 ${s.tok}`:s.role==='M'?`，原字 ${orig}`:`，${orig}`}，權重 ${w.toFixed(3)}`;
    return `<span class="tkc ${s.role.toLowerCase()}"><span class="tok ${s.role.toLowerCase()}" role="img" aria-label="${esc(lab)}" title="${esc(lab)}">${esc(s.tok)}</span><span class="wb">${h?`<i style="height:${h.toFixed(1)}px"></i>`:''}</span><span class="wv">${w>0?w.toFixed(3):'0'}</span></span>`};
  $('#c1-toks').innerHTML=`<div class="tokrow">${seq.slice(0,12).map((s,i)=>cell(s,i)).join('')}</div><div class="tokrow">${seq.slice(12).map((s,i)=>cell(s,i+12)).join('')}</div>`;
  const rep=seq.map((s,i)=>s.role==='N'?`位置 ${i} ${TOK[i]} → ${s.tok}`:null).filter(Boolean);
  $('#c1-repl').textContent=rep.length?`被替換：${rep.join('；')}（損失目標是原字）`:'這次抽樣沒有被替換的字。';
  const per=(k,v)=>W.cnt[k]?v.toFixed(4):'—（集合為空）';
  $('#c1-sum').innerHTML=`<b>|M|, |N|, |C∖N|</b> = ${W.cnt.M}, ${W.cnt.N}, ${W.cnt.C}<br>每字權重<br>· 遮罩 ${per('M',1/W.cnt.M)}<br>· 被替換 ${mode==='mdlm'?'0（MDLM 不監督）':per('N',LAMN/W.cnt.N)}<br>· 乾淨可見 ${mode==='ctw'?per('C',CTW/W.cnt.C):'0（不監督）'}<br>三組總權重 <b>${W.total.M} / ${W.total.N} / ${W.total.C}</b>`;
  const a=mode==='mdlm'?0:alpha;
  $('#c1-ana').innerHTML=`<span class="acc">解析值〔本頁推導〕</span> 目前 λ：遮罩／被替換／乾淨 = ${lam.toFixed(3)} / ${((1-lam)*a).toFixed(3)} / ${((1-lam)*(1-a)).toFixed(3)}；對 λ ~ U[0,1] 平均 = 0.500 / ${(a/2).toFixed(3)} / ${((1-a)/2).toFixed(3)}<br>P(N = ∅)：目前 λ、n = 21 → <b>${f4(pEmptyAt(lam,a,21))}</b>；平均、n = 21 → <b>${f4(pEmptyAvg(a,21))}</b>；平均、n = 4096 → <b>${pEmptyAvg(a,4096).toFixed(5)}</b>`;
  announce('c1',`|M| ${W.cnt.M}，|N| ${W.cnt.N}，乾淨可見 ${W.cnt.C}`)}
function initC1(){segInit('#c1-mode',v=>{c1.mode=v;renderC1()});
  $('#c1-lam').addEventListener('input',e=>{c1.lam=+e.target.value;renderC1()});
  $('#c1-a').addEventListener('input',e=>{c1.alpha=+e.target.value;renderC1()});
  $('#c1-seed').addEventListener('click',()=>{c1.seed++;renderC1()});renderC1()}

/* ================= C2 · Algorithm 1 step by step (logic of verify/c2.js) ================= */
const CANON=['for','i','in','range','(','n',')',':','sum','+=','arr','[','i',']'];
const START=CANON.slice();START[12]='j';
const BASEC=[0.99,0.99,0.99,0.88,0.99,0.97,0.99,0.99,0.95,0.98,0.99,0.99,0.97,0.99];
const ARMS={
  mdlm:{conf:(p,t)=>t===CANON[p]?BASEC[p]:(p===12&&t==='j')?0.98:(p===8&&t==='total')?0.60:0.05,argmax:p=>p===8?'total':CANON[p]},
  cdlm:{conf:(p,t)=>t===CANON[p]?BASEC[p]:(p===12&&t==='j')?0.19:0.05,argmax:p=>CANON[p]}};
function c2new(arm,tauPct,T){return{arm,tauPct,T,z:START.slice(),t:0,passes:0,log:[],done:false,ok:false,c:null,fresh:[]}}
function c2step(st){if(st.done)return st;const A=ARMS[st.arm];st.passes++;
  const zt=st.z.map((tok,p)=>tok==='[M]'?A.argmax(p):tok);st.fresh=st.z.map((tok,p)=>tok==='[M]');
  const c=zt.map((tok,p)=>A.conf(p,tok));st.c=c;
  const fin=()=>{st.z=zt;st.done=true;st.ok=zt.every((x,p)=>x===CANON[p])};
  if(st.t===st.T){st.log.push({t:st.t,r:null,passes:st.passes});fin();return st}
  const r=[];c.forEach((v,p)=>{if(Math.round(v*100)<=st.tauPct)r.push(p)});
  st.log.push({t:st.t,r:r.map(p=>zt[p]),passes:st.passes});
  if(!r.length){fin();return st}
  st.zt=zt;st.z=zt.map((tok,p)=>r.includes(p)?'[M]':tok);st.t++;return st}
function c2run(arm,tauPct,T){const st=c2new(arm,tauPct,T);while(!st.done)c2step(st);return st}
let c2=c2new('cdlm',90,4);
const hC=c=>Math.min(1,-Math.log10(1-c)/2);
function renderC2(flash){const st=c2,tau=st.tauPct/100,ht=hC(Math.min(tau,0.99));
  $('#c2-tauo').textContent=tau.toFixed(2);$('#c2-tau').setAttribute('aria-valuetext','τ = '+tau.toFixed(2));
  const shown=st.z,c=st.c;
  const cell=p=>{const tok=shown[p],cv=c?c[p]:null,m=tok==='[M]',low=cv!==null&&Math.round(cv*100)<=st.tauPct;
    const disp=m?'[M]':tok,lab=`位置 ${p}：${m?'已重遮罩（原為 '+(st.zt?st.zt[p]:START[p])+'）':tok}${cv!==null?'，信心 '+cv.toFixed(2):''}`;
    return `<span class="c2t${low?' low':''}">${cv!==null?`<span class="cb"><i style="height:${(hC(cv)*56).toFixed(1)}px"></i><span class="tau" style="bottom:${(ht*56).toFixed(1)}px"></span></span><span class="wv">${cv.toFixed(2)}</span>`:''}<span class="tok ${m?'m':'c'}${flash&&st.fresh[p]&&!m?' flash':''}" role="img" aria-label="${esc(lab)}">${esc(disp)}</span></span>`};
  const idx=[...Array(14).keys()];
  $('#c2-fig').innerHTML=`<p class="c2axis">長條：信心（1−信心 的對數刻度；刻度 0.5／0.9／0.99 對應高度 15％／50％／100％）；虛線：τ = ${tau.toFixed(2)}</p><div class="c2line">${idx.slice(0,8).map(cell).join('')}</div><div class="c2line">${idx.slice(8).map(cell).join('')}</div>`;
  const last=st.log[st.log.length-1];
  const rs=last?(last.r===null?'（最後一次只填遮罩）':last.r.length?'{'+last.r.join(', ')+'}':'∅'):'—';
  let s=st.passes?`第 ${last.t} 步｜已用前向傳遞 <b>${st.passes}</b> 次（上限 ${st.T+1}）｜重遮罩集合 ${esc(rs)}`:`尚未執行｜上限 ${st.T+1} 次前向傳遞`;
  if(st.done)s+=`<br><span class="c2out acc">結束：${st.ok?'與正確程式相同':'仍有錯'}</span>`;
  $('#c2-status').innerHTML=s;
  $('#c2-log tbody').innerHTML=st.log.map(l=>`<tr><td class="num">${l.t}</td><td>${l.r===null?'（回傳）':l.r.length?esc('{'+l.r.join(', ')+'}'):'∅（提早停止）'}</td><td class="num">${l.passes}</td></tr>`).join('')||'<tr><td colspan="3" class="note">尚未執行</td></tr>';
  $('#c2-step').disabled=st.done;$('#c2-run').disabled=st.done}
function c2reset(){c2=c2new(c2.arm,c2.tauPct,c2.T);renderC2()}
function initC2(){segInit('#c2-arm',v=>{c2.arm=v;c2reset()});segInit('#c2-T',v=>{c2.T=+v;c2reset()});
  $('#c2-tau').addEventListener('input',e=>{c2.tauPct=+e.target.value;c2reset()});
  $('#c2-step').addEventListener('click',()=>{c2step(c2);renderC2(!RM);const l=c2.log[c2.log.length-1];
    announce('c2',c2.done?`共 ${c2.log.length} 步、${c2.passes} 次傳遞，結果：${c2.ok?'與正確程式相同':'仍有錯'}`:`第 ${l.t} 步，重遮罩 ${l.r.length} 個位置，已用 ${c2.passes} 次傳遞`,true)});
  $('#c2-run').addEventListener('click',()=>{while(!c2.done)c2step(c2);renderC2(false);announce('c2',`共 ${c2.log.length} 步、${c2.passes} 次傳遞，結果：${c2.ok?'與正確程式相同':'仍有錯'}`,true)});
  $('#c2-reset').addEventListener('click',c2reset);renderC2()}

/* ================= C3 · localization metrics model (port of verify/c3.js) ================= */
const U0=-30,U1=30,NU=6001,DU=(U1-U0)/(NU-1);
const UG=Float64Array.from({length:NU},(_,i)=>U0+i*DU);
function erf(x){const s=x<0?-1:1;x=Math.abs(x);const t=1/(1+0.3275911*x);
  return s*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x))}
const Phi=z=>0.5*(1+erf(z/Math.SQRT2)),sig=u=>1/(1+Math.exp(-u)),logit=c=>Math.log(c/(1-c));
const npdf=(u,m,s)=>Math.exp(-0.5*((u-m)/s)**2)/(s*Math.sqrt(2*Math.PI));
const wq=i=>(i===0||i===NU-1)?0.5*DU:DU;
const M0=3.815,S0=0.86,MH=0,SH=1;
function binCdfs(N,p,KMAX,out){
  if(p<=0){out.fill(1);return}if(p>=1){for(let k=0;k<KMAX;k++)out[k]=k>=N?1:0;return}
  let lt=N*Math.log1p(-p),acc=0;const r=Math.log(p)-Math.log1p(-p);
  for(let k=0;k<KMAX;k++){if(k>0){if(k>N){out[k]=1;continue}lt+=Math.log(N-k+1)-Math.log(k)+r}acc+=Math.exp(lt);out[k]=Math.min(1,acc)}}
function model(P){const me=logit(P.med),KMAX=10,hit=new Float64Array(KMAX),tmp=new Float64Array(KMAX);
  let cl=0,co=0,mass=0;const fcA=new Float64Array(NU),feA=new Float64Array(NU);
  for(let i=0;i<NU;i++){const u=UG[i],w=wq(i),fe=npdf(u,me,P.se);
    const fc=(1-P.pi)*npdf(u,M0,S0)+P.pi*npdf(u,MH,SH);fcA[i]=fc;feA[i]=fe;
    cl+=w*fc*sig(u);co+=w*fe*sig(u);mass+=w*fe;
    const Fc=(1-P.pi)*Phi((u-M0)/S0)+P.pi*Phi((u-MH)/SH);
    binCdfs(P.n-1,Fc,KMAX,tmp);for(let k=0;k<KMAX;k++)hit[k]+=w*fe*tmp[k]}
  const lt=logit(P.tau),Fct=(1-P.pi)*Phi((lt-M0)/S0)+P.pi*Phi((lt-MH)/SH),Fet=Phi((lt-me)/P.se);
  return{clean:cl,corr:co,gap:cl-co,hit:Array.from(hit),hitK:hit[P.K-1],fp5:1-hit[4]/5,
    remaskPct:100*((P.n-1)*Fct+Fet)/P.n,errRemasked:Fet,cleanRemasked:(P.n-1)*Fct,noneBelow:Math.pow(1-Fct,P.n-1)*(1-Fet),mass,fcA,feA}}
const c3={med:0.772,se:1.5,pi:0.05,n:56,tau:0.9,K:5};let c3raf=0;
const CT=[[0.01,'0.01'],[0.1,'0.1'],[0.5,'0.5'],[0.9,'0.9'],[0.99,'0.99'],[0.999,'0.999']];
function renderC3(){c3raf=0;const P=c3,r=model(P);
  $('#c3-tauo').textContent=P.tau.toFixed(2);$('#c3-Ko').textContent=P.K;$('#c3-pio').textContent=P.pi.toFixed(3);$('#c3-medo').textContent=P.med.toFixed(3);$('#c3-seo').textContent=P.se.toFixed(1);$('#c3-no').textContent=P.n;
  /* density */
  const box=$('#c3-dens').parentNode,W=widthOf(box,280,520),H=Math.round(W*0.62),UA=-6,UB=8.5;
  let ymax=0;const pc=[],pe=[];for(let i=0;i<NU;i+=15){const u=UG[i];if(u<UA||u>UB)continue;pc.push([u,r.fcA[i]]);pe.push([u,r.feA[i]]);ymax=Math.max(ymax,r.fcA[i],r.feA[i])}
  const F=frame($('#c3-dens'),{W,H,m:{l:14,r:12,t:28,b:40},x:[UA,UB],y:[0,ymax*1.1],xt:CT.map(([c,l])=>[logit(c),l]),xl:'信心（橫軸為 ln(c/(1−c))）'});
  const svg=$('#c3-dens'),defs=el('defs',null,svg),pat=el('pattern',{id:'c3h',width:7,height:7,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},defs);el('line',{x1:0,y1:0,x2:0,y2:7,class:'hatchl'},pat);
  const ut=logit(P.tau),xt=F.X(ut);
  el('rect',{x:F.x0,y:F.y1,width:Math.max(0,xt-F.x0),height:F.y0-F.y1,fill:'url(#c3h)',opacity:.55},F.g);
  const P2=a=>a.map(([u,v])=>[F.X(u),F.Y(v)]);
  el('path',{d:lp(P2(pc))+`L${F.X(pc[pc.length-1][0]).toFixed(1)} ${F.y0}L${F.X(pc[0][0]).toFixed(1)} ${F.y0}Z`,class:'cl-f'},F.g);
  el('path',{d:lp(P2(pc)),class:'cl'},F.g);el('path',{d:lp(P2(pe)),class:'er'},F.g);
  el('line',{x1:xt,x2:xt,y1:F.y1-6,y2:F.y0,class:'tau'},F.g);tx(F.g,Math.min(xt+4,F.x1-60),F.y1-10,'τ = '+P.tau.toFixed(2),{class:'labm'});
  // direct labels at each curve's peak
  const pk=a=>a.reduce((b,q)=>q[1]>b[1]?q:b,a[0]);const kc=pk(pc),ke=pk(pe);
  tx(F.g,F.X(kc[0]),F.Y(kc[1])-8,'清潔字',{class:'lab','text-anchor':'middle'});
  tx(F.g,Math.max(F.x0+20,Math.min(F.x1-20,F.X(ke[0]))),F.Y(ke[1])-8,'錯字',{class:'lab','text-anchor':'middle'});
  $('#c3-dens').setAttribute('aria-label',`信心分布：清潔平均 ${f3(r.clean)}，錯字平均 ${f3(r.corr)}，Gap ${f3(r.gap)}`);
  /* Hit@K */
  const bx2=$('#c3-hit').parentNode,W2=widthOf(bx2,280,520),H2=Math.round(W2*0.62);
  const G=frame($('#c3-hit'),{W:W2,H:H2,m:{l:44,r:14,t:28,b:40},x:[0.5,10.5],y:[0,1],xt:[1,2,3,4,5,6,7,8,9,10],yt:[0,.25,.5,.75,1],yf:v=>v.toFixed(2),xl:'K',yl:'Hit@K'});
  const hp=r.hit.map((v,k)=>[G.X(k+1),G.Y(v)]);el('path',{d:lp(hp),class:'ink'},G.g);
  hp.forEach(([x,y],k)=>el('circle',{cx:x,cy:y,r:k+1===P.K?6.5:3.5,class:k+1===P.K?'acc-f':'ink-f'},G.g));
  tx(G.g,G.X(P.K),G.Y(r.hitK)+(r.hitK>0.85?20:-12),'Hit@'+P.K+' = '+f3(r.hitK),{class:'labm','text-anchor':P.K>7?'end':P.K<3?'start':'middle'});
  $('#c3-hit').setAttribute('aria-label',`Hit@K 曲線，Hit@1 ${f3(r.hit[0])}，Hit@${P.K} ${f3(r.hitK)}`);
  $('#c3-ro1').innerHTML=`清潔平均 <b>${f4(r.clean)}</b>　錯字平均 <b>${f4(r.corr)}</b><br>Gap <b class="acc">${f4(r.gap)}</b><br>Hit@1 <b class="acc">${f4(r.hit[0])}</b>　Hit@${P.K} <b>${f4(r.hitK)}</b><br>Top-5 誤報率 <b>${f4(r.fp5)}</b>〔本頁推導 1 − Hit@5/5〕`;
  $('#c3-ro2').innerHTML=`τ = ${P.tau.toFixed(2)} 時<br>錯字被重遮罩的機率 <b>${f4(r.errRemasked)}</b><br>每支程式被重遮罩的清潔字 <b>${r.cleanRemasked.toFixed(2)}</b> 個<br>第一步重遮罩比例 <b>${r.remaskPct.toFixed(2)}%</b><br>沒有任何位置 ≤ τ 的機率 <b>${f4(r.noneBelow)}</b>`;
  announce('c3',`Gap ${f3(r.gap)}，Hit@1 ${f3(r.hit[0])}，Hit@${P.K} ${f3(r.hitK)}`)}
const c3q=()=>{if(!c3raf)c3raf=requestAnimationFrame(renderC3)};
function initC3(){segInit('#c3-pre',v=>{c3.med=+v;$('#c3-med').value=v;c3q()});
  [['tau','tau'],['K','K'],['pi','pi'],['med','med'],['se','se'],['n','n']].forEach(([id,k])=>$('#c3-'+id).addEventListener('input',e=>{c3[k]=+e.target.value;if(k==='med')segSet('#c3-pre',['0.772','0.594','0.121'].includes(e.target.value)?e.target.value:'');c3q()}));
  renderC3()}

/* ================= C4 · paper data (verify/c4.js) ================= */
const A34=[0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9];
const T34={gap:[.152,.146,.189,.194,.216,.320,.333,.355,.367,.481,.522,.572],
 hit5:[.811,.779,.811,.807,.837,.852,.883,.863,.869,.913,.902,.874],
 fp5:[.716,.740,.708,.710,.699,.659,.650,.663,.656,.617,.630,.635],
 p1:[.094,.091,.115,.110,.119,.201,.151,.183,.166,.207,.159,.140],
 p4:[.257,.258,.249,.240,.234,.292,.275,.324,.286,.278,.165,.163]};
const YR={gap:[0,.6],hit5:[.75,.95],fp5:[.6,.75],p1:[0,.35]};
const MNAME={gap:'Gap',hit5:'Top-5 命中',fp5:'Top-5 誤報',p1:'Pass@1'};
const DEPTH=[
 {k:'cdlm',name:'CDLM',src:'Table 2，2000 步排程',v:[.197,.213,.210,.205],on:1,main:1,cls:'cl'},
 {k:'ctw',name:'CDLM+ctw',src:'Table 2，2000 步排程',v:[.204,.192,.184,.182],on:1,main:1,cls:'acc'},
 {k:'main',name:'CDLM-0.5B 主檢查點',src:'Table 2，較長排程',v:[.188,.279,.281,.286]},
 {k:'base',name:'Base',src:'Table 35',v:[.145,.240,.255,.263]},
 {k:'mdlm',name:'MDLM-0.5B',src:'Table 35',v:[.138,.236,.246,.245]},
 {k:'moci',name:'MDLM-OCI',src:'Table 35',v:[.140,.223,.235,.239]},
 {k:'coci',name:'CDLM-OCI',src:'Table 35',v:[.220,.302,.303,.307]}];
const SHAPES=['circle','square','tri','diamond','itri','cross','star'];
function marker(g,shape,x,y,cls,s=5){const P=pts=>el('path',{d:'M'+pts.map(p=>p.map(v=>v.toFixed(1)).join(' ')).join('L')+'Z',class:cls},g);
  if(shape==='circle')return el('circle',{cx:x,cy:y,r:s,class:cls},g);
  if(shape==='square')return el('rect',{x:x-s,y:y-s,width:2*s,height:2*s,class:cls},g);
  if(shape==='tri')return P([[x,y-s*1.2],[x+s*1.1,y+s*.8],[x-s*1.1,y+s*.8]]);
  if(shape==='itri')return P([[x,y+s*1.2],[x+s*1.1,y-s*.8],[x-s*1.1,y-s*.8]]);
  if(shape==='diamond')return P([[x,y-s*1.3],[x+s*1.1,y],[x,y+s*1.3],[x-s*1.1,y]]);
  if(shape==='cross')return P([[x-s*.35,y-s],[x+s*.35,y-s],[x+s*.35,y-s*.35],[x+s,y-s*.35],[x+s,y+s*.35],[x+s*.35,y+s*.35],[x+s*.35,y+s],[x-s*.35,y+s],[x-s*.35,y+s*.35],[x-s,y+s*.35],[x-s,y-s*.35],[x-s*.35,y-s*.35]]);
  const q=[];for(let i=0;i<10;i++){const a=-Math.PI/2+i*Math.PI/5,rr=i%2?s*.5:s*1.25;q.push([x+rr*Math.cos(a),y+rr*Math.sin(a)])}return P(q)}
const c4={tab:'alpha',met:'p1',ai:3,on:new Set(DEPTH.filter(d=>d.on).map(d=>d.k))};
function renderC4(){const svg=$('#c4-fig'),box=svg.parentNode,W=widthOf(box,280,640),H=Math.round(Math.max(240,W*0.56));
  $('#c4-actl').classList.toggle('hide',c4.tab!=='alpha');$('#c4-dctl').classList.toggle('hide',c4.tab!=='depth');
  if(c4.tab==='alpha'){const m=c4.met,yr=YR[m],ai=c4.ai;$('#c4-ao').textContent=A34[ai];$('#c4-a').setAttribute('aria-valuetext','α = '+A34[ai]);
    const yt=[];const st=m==='gap'?.1:m==='p1'?.05:m==='hit5'?.05:.05;for(let v=yr[0];v<=yr[1]+1e-9;v+=st)yt.push(+v.toFixed(2));
    const F=frame(svg,{W,H,m:{l:46,r:16,t:30,b:42},x:[-0.5,11.5],y:yr,xt:A34.map((a,i)=>[i,W<420&&i%2?'':String(a)]),yt,yf:v=>v.toFixed(2),xl:'α',yl:MNAME[m]});
    const bg=el('g',null,null);F.g.insertBefore(bg,F.g.firstChild);
    [[0,4,'0.04–0.2'],[5,9,'0.3–0.7'],[10,11,'0.8–0.9']].forEach(([a,b,l],j)=>{const x0=F.X(a-.5),x1=F.X(b+.5);if(j!==1)el('rect',{x:x0,y:F.y1,width:x1-x0,height:F.y0-F.y1,class:'band'},bg);tx(F.g,(x0+x1)/2,F.y1-12,l,{class:'tk','text-anchor':'middle'})});
    el('line',{x1:F.X(3),x2:F.X(3),y1:F.y1,y2:F.y0,class:'cur'},F.g);
    el('line',{x1:F.X(ai),x2:F.X(ai),y1:F.y1,y2:F.y0,class:'ref'},F.g);
    const series=m==='p1'?[['p1','T = 1','circle',''],['p4','T = 4','square',' dash']]:[[m,MNAME[m],'circle','']];
    series.forEach(([k,lab,sh,d])=>{const p=T34[k].map((v,i)=>[F.X(i),F.Y(v)]);el('path',{d:lp(p),class:'ink'+d},F.g);p.forEach(([x,y],i)=>marker(F.g,sh,x,y,i===ai?'acc-f':'ink-f',i===ai?6:4));
      const [lx,ly]=p[0];if(m==='p1')tx(F.g,lx+8,ly+(k==='p1'?22:-12),lab,{class:'lab'})});
    svg.setAttribute('aria-label',`α 掃描：${MNAME[m]}，Table 34`);
    $('#c4-ro').innerHTML=`<b>α = ${A34[ai]}</b>${ai===3?'（主要設定）':''}<br>Gap ${T34.gap[ai].toFixed(3)}<br>Top-5 命中 ${T34.hit5[ai].toFixed(3)}<br>Top-5 誤報 ${T34.fp5[ai].toFixed(3)}<br>Pass@1 T=1 ${T34.p1[ai].toFixed(3)}<br>Pass@1 T=4 ${T34.p4[ai].toFixed(3)}<br>T=1→4 ${(T34.p4[ai]-T34.p1[ai]>=0?'+':'')}${(T34.p4[ai]-T34.p1[ai]).toFixed(3)}`;
  }else{const F=frame(svg,{W,H,m:{l:46,r:W<420?96:150,t:24,b:42},x:[0.8,4.2],y:[.12,.32],xt:[[1,'T = 1'],[2,'2'],[3,'3'],[4,'4']],yt:[.12,.16,.20,.24,.28,.32],yf:v=>v.toFixed(2),xl:'精細化步數 T',yl:'Pass@1'});
    const labs=[];
    DEPTH.forEach((d,j)=>{if(!c4.on.has(d.k))return;const cls=d.cls||'ink';const p=d.v.map((v,i)=>[F.X(i+1),F.Y(v)]);
      el('path',{d:lp(p),class:(d.cls==='cl'?'cl':d.cls==='acc'?'acc':'ink')+(d.main?'':' thin')},F.g);
      p.forEach(([x,y])=>marker(F.g,SHAPES[j],x,y,(d.cls==='cl'?'clm':d.cls==='acc'?'acc-f':'ink-f'),d.main?5:4));
      labs.push({y:p[3][1],y0:p[3][1],x:p[3][0],t:d.name+' '+d.v[3].toFixed(3)})});
    labs.sort((a,b)=>a.y-b.y);for(let i=1;i<labs.length;i++)if(labs[i].y-labs[i-1].y<14)labs[i].y=labs[i-1].y+14;
    const over=labs.length?labs[labs.length-1].y-(F.y0-4):0;if(over>0)labs.forEach(l=>l.y-=over);
    labs.forEach(l=>{if(Math.abs(l.y-l.y0)>2)el('line',{x1:l.x+6,y1:l.y0,x2:l.x+14,y2:l.y,class:'proj'},F.g);tx(F.g,l.x+16,l.y+4,l.t,{class:'lab',style:'font-size:12px'})});
    svg.setAttribute('aria-label','精細化深度：Pass@1 對 T，'+labs.map(l=>l.t).join('、'));
    const on=DEPTH.filter(d=>c4.on.has(d.k));
    $('#c4-ro').innerHTML=on.map(d=>`<b>${d.name}</b>（${d.src}）<br>T=1→4 ${(d.v[3]-d.v[0]>=0?'+':'')}${(d.v[3]-d.v[0]).toFixed(3)}`).join('<br>')||'未選擇曲線';}
}
function initC4(){segInit('#c4-tab',v=>{c4.tab=v;renderC4()});segInit('#c4-met',v=>{c4.met=v;renderC4()});
  $('#c4-a').addEventListener('input',e=>{c4.ai=+e.target.value;renderC4();announce('c4',`α = ${A34[c4.ai]}：Gap ${T34.gap[c4.ai]}，Pass@1 T=1 ${T34.p1[c4.ai]}，T=4 ${T34.p4[c4.ai]}`)});
  const box=$('#c4-dctl');DEPTH.forEach((d,j)=>{const lab=document.createElement('label');lab.className='chk';
    lab.innerHTML=`<input type="checkbox" ${c4.on.has(d.k)?'checked':''}><svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"></svg><span>${d.name}<span class="note">（${d.src}）</span></span>`;
    marker(lab.querySelector('svg'),SHAPES[j],8,8,d.cls==='cl'?'clm':d.cls==='acc'?'acc-f':'ink-f',5);
    lab.querySelector('input').addEventListener('change',e=>{e.target.checked?c4.on.add(d.k):c4.on.delete(d.k);renderC4()});box.appendChild(lab)});
  $('#c4-tbl').innerHTML=`<table><caption>Table 34：α 掃描</caption><thead><tr><th class="num">α</th><th class="num">Gap</th><th class="num">Top-5 命中</th><th class="num">Top-5 誤報</th><th class="num">Pass@1 T=1</th><th class="num">T=4</th></tr></thead><tbody>${A34.map((a,i)=>`<tr${i===3?' class="hl"':''}><td class="num">${a}</td>${['gap','hit5','fp5','p1','p4'].map(k=>`<td class="num">${T34[k][i].toFixed(3)}</td>`).join('')}</tr>`).join('')}</tbody></table>
  <table><caption>Pass@1 隨 T（Table 2、Table 35）</caption><thead><tr><th>設定</th><th class="num">T=1</th><th class="num">T=2</th><th class="num">T=3</th><th class="num">T=4</th></tr></thead><tbody>${DEPTH.map(d=>`<tr><td>${d.name}（${d.src}）</td>${d.v.map(v=>`<td class="num">${v.toFixed(3)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  renderC4()}

/* ================= timeline ================= */
const TL_GROUP={'1810.04805':['base','uniform-noise-gidd-clean-supervision'],'2107.03006':['base','discrete-diffusion-foundations-d3pm'],'2205.14217':['base','discrete-diffusion-foundations-d3pm'],'2211.15029':['base','discrete-diffusion-foundations-d3pm'],
 '2406.07524':['mdlm','mdlm-masked-diffusion-objective'],'2502.09992':['mdlm','large-masked-dlms-llada-dream'],'2508.15487':['mdlm','large-masked-dlms-llada-dream'],
 '2502.03540':['remask','dlm-remasking-self-correction'],'2503.00307':['remask','dlm-remasking-self-correction'],'2510.01384':['remask','dlm-remasking-self-correction'],
 '2503.04482':['mix','uniform-noise-gidd-clean-supervision'],'2107.03374':['bench','code-benchmarks-for-revision'],'2305.01210':['bench','code-benchmarks-for-revision'],
 '2512.15596':['me','corrective-dlm-mixture-noise-posttraining']};
const TL_NAME={base:'離散擴散基礎',mdlm:'遮罩擴散與大型 DLM',remask:'重遮罩解碼',mix:'混合雜訊',bench:'程式基準',me:'監督位置 · 本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  if(!data.some(e=>e.arxiv==='2512.15596'))data.push({year:2025,title:'Corrective Diffusion Language Models',authors_short:'本論文',arxiv:'2512.15596',role:'混合雜訊後訓練與 Code Revision Benchmark。'});
  data=data.map((e,i)=>({e,i})).sort((a,b)=>(+a.e.year||0)-(+b.e.year||0)||a.i-b.i).map(o=>o.e);
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

/* ---------- self test (values from design.md §5) ---------- */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t)=>Math.abs(a-b)<=t+1e-9;
  /* C1 §5.1 */
  [[.3,.1,'cdlm',[7,1,13],[.1429,.1,null]],[.3,.1,'mdlm',[7,0,14],[.1429,null,null]],[.3,.1,'ctw',[7,1,13],[.1429,.1,.0077]],[0,.1,'cdlm',[0,2,19],[null,.05,null]],[1,.1,'cdlm',[21,0,0],[.0476,null,null]],[.3,.5,'cdlm',[7,4,10],[.1429,.025,null]]].forEach(([l,a,m,cnt,w])=>{
    const s=corrupt(1,l,a,m),W=weights(s,m),pw=r=>{const i=s.findIndex(x=>x.role===r);return i<0||W.w[i]===0?null:W.w[i]};
    ok(W.cnt.M===cnt[0]&&W.cnt.N===cnt[1]&&W.cnt.C===cnt[2],`C1 counts λ=${l} α=${a} ${m}: ${W.cnt.M},${W.cnt.N},${W.cnt.C}`);
    ['M','N','C'].forEach((r,j)=>ok(w[j]===null?pw(r)===null:near(pw(r),w[j],1e-4),`C1 weight ${r} λ=${l} α=${a} ${m}`))});
  ok(corrupt(1,.3,.1,'cdlm')[0].tok==='total','C1 pos0 → total');ok(corrupt(1,0,.1,'cdlm')[7].tok==='<','C1 pos7 → <');
  ok(near(pEmptyAt(.3,.1,21),.2178,1e-4),'C1 P∅ λ=.3');ok(near(pEmptyAvg(.1,21),.4098,1e-4),'C1 P∅ avg 21');ok(near(pEmptyAvg(.1,4096),.00244,1e-5),'C1 P∅ avg 4096');
  ok(near(pEmptyAvg(.04,21),.6735,1e-4),'C1 P∅ α=.04');ok(near(pEmptyAvg(.5,21),.0909,1e-4),'C1 P∅ α=.5');ok(near(pEmptyAvg(1,21),1/22,1e-9)&&pEmptyAvg(0,21)===1,'C1 P∅ guards');
  /* C2 §5.2 */
  [['mdlm',80,4,1,false],['mdlm',90,1,2,false],['mdlm',90,4,5,false],['mdlm',96,4,5,false],['mdlm',99,1,2,false],['cdlm',80,4,2,true],['cdlm',90,1,2,true],['cdlm',90,4,5,true],['cdlm',96,4,5,true]].forEach(([a,t,T,p,c])=>{const st=c2run(a,t,T);ok(st.passes===p&&st.ok===c,`C2 ${a} τ=${t/100} T=${T}: passes ${st.passes} ok ${st.ok}`)});
  ok(c2run('mdlm',96,4).z[8]==='total','C2 mdlm τ=.96 sum→total');ok(c2run('mdlm',99,1).log[0].r.length===14,'C2 mdlm τ=.99 remask all 14');
  ok(JSON.stringify(c2run('cdlm',90,4).log.map(l=>l.r))==='[["range","j"],["range"],["range"],["range"],null]','C2 cdlm τ=.9 sets');
  /* C3 §5.3 */
  const D={se:1.5,pi:.05,n:56,tau:.9,K:5},R=o=>model(Object.assign({},D,o));
  const cols=['clean','corr','gap','h1','hitK','fp5','remaskPct','errRemasked','cleanRemasked','noneBelow'];
  const get=(r,c)=>c==='h1'?r.hit[0]:r[c],tol=c=>c==='cleanRemasked'?.02:c==='remaskPct'?.02:.002;
  [[{med:.772},[.9465,.7049,.2416,.1654,.7104,.8579,8.965,.7427,4.278,.0030],'Base'],[{med:.594},[.9465,.5669,.3796,.2984,.8547,.8291,9.223,.8871,4.278,.0013],'MDLM'],[{med:.121},[.9465,.1920,.7544,.7813,.9920,.8016,9.420,.9973,4.278,0],'CDLM'],
   [{med:.121,pi:0},[.97,.192,.7779,.9931,.9988,.8002,4.725,.9973,1.649,.0005],'CDLM π=0'],[{med:.121,pi:.1},[.923,.192,.7309,.6918,.9586,.8083,14.115,.9973,6.907,0],'CDLM π=.1'],[{med:.772,pi:0},[.97,.7049,.2651,.6581,.8234,.8353,4.27,.7427,1.649,.0482],'Base π=0']]
   .forEach(([o,v,lab])=>{const r=R(o);cols.forEach((c,i)=>ok(near(get(r,c),v[i],tol(c)),`C3 ${lab} ${c}: ${get(r,c).toFixed(4)} vs ${v[i]}`))});
  [[{med:.121,tau:.8},{remaskPct:6.489,errRemasked:.9877,cleanRemasked:2.646,noneBelow:.0008}],[{med:.121,tau:.95},{remaskPct:21.215,errRemasked:.9995,cleanRemasked:10.881,noneBelow:0}],
   [{med:.772,tau:.8},{remaskPct:5.697,errRemasked:.5442,cleanRemasked:2.646,noneBelow:.0303}],[{med:.772,tau:.95},{remaskPct:20.993,errRemasked:.8749,cleanRemasked:10.881,noneBelow:0}],
   [{med:.121,n:20},{h1:.894,hitK:.9994,fp5:.8001,remaskPct:12.376,cleanRemasked:1.478,noneBelow:.0006}],[{med:.121,n:200},{h1:.6169,hitK:.8909,fp5:.8218,remaskPct:8.238,cleanRemasked:15.478,noneBelow:0}],
   [{med:.772,n:5},{h1:.7411,hitK:1,fp5:.8,remaskPct:21.076,cleanRemasked:.311,noneBelow:.1861}],[{med:.121,n:5},{h1:.9728,hitK:1,fp5:.8,remaskPct:26.169,cleanRemasked:.311,noneBelow:.0019}],
   [{med:.594,K:10},{h1:.2984,hitK:.9471}],
   [{med:.01,se:4,pi:.2,n:200,tau:.99,K:10},{clean:.876,corr:.1471,gap:.7289,h1:.7277,hitK:.8348,fp5:.8396,remaskPct:85.494,errRemasked:.9892,cleanRemasked:170,noneBelow:0}],
   [{med:.99,se:.3,pi:0,n:20,tau:.5,K:1},{clean:.97,corr:.9896,gap:-.0196,h1:0,hitK:0,fp5:1,remaskPct:.0004,errRemasked:0,cleanRemasked:.0001,noneBelow:.9999}]]
   .forEach(([o,v])=>{const r=R(o);for(const c in v)ok(near(get(r,c),v[c],tol(c)),`C3 ${JSON.stringify(o)} ${c}: ${get(r,c).toFixed(4)} vs ${v[c]}`)});
  ok(near(R({med:.01,se:4,pi:.2,n:200}).mass,1,1e-4),'C3 mass');
  /* C4 §5.4 */
  const rg=(k,lo,hi)=>{const v=T34[k].filter((_,i)=>A34[i]>=lo-1e-9&&A34[i]<=hi+1e-9);return[Math.min(...v),Math.max(...v)]};
  ok(JSON.stringify(rg('p1',.04,.2))==='[0.091,0.119]'&&JSON.stringify(rg('p1',.3,.7))==='[0.151,0.207]','C4 p1 ranges');
  ok(JSON.stringify(rg('p4',.04,.2))==='[0.234,0.258]'&&JSON.stringify(rg('p4',.3,.7))==='[0.275,0.324]','C4 p4 ranges');
  const am=(k,f)=>A34[T34[k].indexOf(f(...T34[k]))];ok(am('p1',Math.max)===.7&&am('p4',Math.max)===.5&&am('hit5',Math.max)===.7&&am('fp5',Math.min)===.7,'C4 argmax');
  ok(T34.gap.every((v,i)=>i<2||v>=T34.gap[i-1]),'C4 gap monotone from 0.06');
  const g=A34.map((_,i)=>+(T34.p4[i]-T34.p1[i]).toFixed(3));ok(g[10]===.006&&g[11]===.023&&Math.min(...g.slice(0,10))===.071&&Math.max(...g.slice(0,10))===.167,'C4 T1→T4 gains');
  ok(near(DEPTH[0].v[3]-DEPTH[0].v[0],.008,1e-9)&&near(DEPTH[1].v[3]-DEPTH[1].v[0],-.022,1e-9)&&near(DEPTH[2].v[3]-DEPTH[2].v[0],.098,1e-9),'C4 depth deltas');
  let inr=0;for(const k in YR)T34[k].forEach(v=>{if(v>=YR[k][0]&&v<=YR[k][1])inr++});T34.p4.forEach(v=>{if(v>=YR.p1[0]&&v<=YR.p1[1])inr++});ok(inr===60,'C4 60 values in fixed axes: '+inr);
  /* layout */
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  $$('.tscroll,.mscroll').forEach((n,i)=>{const p=n.parentElement;if(!p.offsetWidth)return;ok(n.getBoundingClientRect().width<=p.getBoundingClientRect().width+1,'container '+i+' wider than parent')});
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ---------- boot ---------- */
initTheme();initToc();initC1();initC2();initC3();initC4();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderC3();renderC4()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))selfTest();
/* ?cap=<id>: screenshot harness, shows that element at the top of the page by hiding what precedes it */
const capId=new URLSearchParams(location.search).get('cap'),capN=capId&&document.getElementById(capId);
if(capN){const sec=capN.closest('section');$$('main>section').forEach(s=>{if(s!==sec)s.style.display='none'});for(let p=capN;p&&p!==sec;p=p.parentElement)for(let q=p.previousElementSibling;q;q=q.previousElementSibling)q.style.display='none';renderC3();renderC4();const ns=+new URLSearchParams(location.search).get('steps');for(let i=0;i<ns;i++)c2step(c2);if(ns)renderC2()}
})();
</script>
</body>
</html>
