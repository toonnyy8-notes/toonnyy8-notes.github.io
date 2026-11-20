const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
const mq=matchMedia('(prefers-reduced-motion: reduce)');let RM=mq.matches;mq.addEventListener&&mq.addEventListener('change',e=>{RM=e.matches});
const fmt4=v=>Math.abs(v)<1e-4&&v!==0?v.toExponential(2):v.toFixed(4);
const sci=v=>v===0?'0':(Math.abs(v)<1e-3||Math.abs(v)>=1e5?v.toExponential(3):v.toFixed(6));

/* ---------- shared math ---------- */
function sig(u){if(u>=0)return 1/(1+Math.exp(-u));const e=Math.exp(u);return e/(1+e)}
function lse(a){let m=-Infinity;for(const x of a)if(x>m)m=x;if(m===-Infinity)return-Infinity;let s=0;for(const x of a)s+=Math.exp(x-m);return m+Math.log(s)}
function profile(N,beta){const x=[];for(let h=0;h<N;h++)x.push(.05+.9*Math.exp(-beta*((7*h+3)%N)/(N-1)));return x}
function cvStats(x){const n=x.length;if(n<=1||x.every(a=>a===x[0]))return{mu:n?x[0]:0,sd:0,cv:0,ncv2:0,S:0};let mu=0;for(const a of x)mu+=a;mu/=n;let S=0;for(const a of x)S+=(a-mu)**2;const sd=Math.sqrt(S/n);return{mu,sd,cv:sd/mu,ncv2:S/(mu*mu),S}}
function gradNcv2(x){const n=x.length;if(x.every(a=>a===x[0]))return x.map(()=>0);let mu=0;for(const a of x)mu+=a;mu/=n;let S=0;for(const a of x)S+=(a-mu)**2;return x.map(a=>2*(a-mu)/(mu*mu)-2*S/(n*mu**3))}
function topM(x,m){return new Set(x.map((v,h)=>[v,h]).sort((a,b)=>b[0]-a[0]||a[1]-b[1]).slice(0,m).map(p=>p[1]))}
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}

/* head row: bars with optional shared hatching */
function headRow(svg,imp,shared,o){o=o||{};const W=o.W,H=o.H||150,m={l:o.axis?34:4,r:4,t:o.top||16,b:22},n=imp.length;
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('width',W);svg.setAttribute('height',H);
  const x0=m.l,x1=W-m.r,y0=H-m.b,y1=m.t,Y=lin(0,1,y0,y1),bw=(x1-x0)/n,g=el('g',null,svg);
  if(o.axis){[0,.5,1].forEach(v=>{el('line',{x1:x0,x2:x1,y1:Y(v),y2:Y(v),class:'grid'},g);tx(g,x0-6,Y(v)+4,String(v),{class:'tk','text-anchor':'end'})})}
  el('line',{x1:x0,x2:x1,y1:y0,y2:y0,class:'axl'},g);
  const rects=[];
  imp.forEach((v,h)=>{const sh=shared&&shared.has(h),x=x0+h*bw+bw*.12,w=bw*.76;
    if(o.ghost)el('rect',{x,y:Y(o.ghost[h]),width:w,height:y0-Y(o.ghost[h]),class:'ref',style:'stroke-dasharray:3 3'},g);
    const r=el('rect',{x,y:Y(v),width:w,height:Math.max(0,y0-Y(v)),class:sh?'bar-h':'bar'},g);rects.push(r);
    if(sh)tx(g,x+w/2,Y(v)-4,'S',{class:'tk','text-anchor':'middle',style:'fill:var(--ink)'});
    if(o.labels!==false&&(n<=13||h%2===0))tx(g,x+w/2,y0+15,String(h),{class:'tk','text-anchor':'middle'})});
  return{X:h=>x0+h*bw+bw/2,Y,g,bw,x0,x1,y0,rects};
}

/* ---------- hero ---------- */
function renderHero(){const b=$('#hero-fig');headRow($('svg',b),profile(16,4),null,{W:widthOf(b,280,520),H:120,labels:false})}

/* ---------- Fig. 1 ---------- */
function renderFig1(){const box=$('#fig1');box.textContent='';
  const cols=[['Vanilla Attention','隱式門控','softmax',null,'Qwen2、3，LLaMA2、3'],['Sink Attention','隱式門控','softmax，分母 + exp(sink)',null,'GPT-OSS、DeepSeek-V4'],['Gated Attention','顯式門控','softmax','× σ(x W_θ)','Qwen3-Next、Qwen3.5、3.6']];
  cols.forEach(c=>{const svg=el('svg',{viewBox:'0 0 240 350',role:'img','aria-label':`${c[0]}（${c[1]}）：q、k、v 線性映射 → MatMul q·k → ${c[2]} → MatMul × v${c[3]?' → '+c[3]:''}`});box.appendChild(svg);
    tx(svg,120,22,c[0],{class:'ttl','text-anchor':'middle'});tx(svg,120,42,c[1],{class:'boxs','text-anchor':'middle',style:c[3]?'fill:var(--accent)':''});
    const boxes=['q、k、v 線性映射','MatMul q·k',c[2],'MatMul × v'];if(c[3])boxes.push(c[3]);
    const hgt=c[3]?30:36,ys=c[3]?[290,236,182,128,74]:[290,224,158,92];
    boxes.forEach((t,i)=>{const y=ys[i],acc=i===4;el('rect',{x:20,y:y-hgt/2,width:200,height:hgt,rx:8,class:acc?'box-acc':'box'},svg);
      const tt=tx(svg,120,y+5,'',{class:'boxt','text-anchor':'middle'});
      if(t.includes('W_θ')){tt.textContent='× σ(x W';const s=el('tspan',{'baseline-shift':'sub','font-size':'75%'},tt);s.textContent='θ';const e=el('tspan',null,tt);e.textContent=')'}else tt.textContent=t;
      if(i>0){const yp=ys[i-1]-hgt/2,yc=y+hgt/2;el('line',{x1:120,x2:120,y1:yp,y2:yc+6,class:'flow'},svg);el('path',{d:`M120 ${yc}l-5 7h10z`,class:'flow-f'},svg)}});
    tx(svg,120,336,c[4],{class:'boxs','text-anchor':'middle'})})}

/* ---------- S6 geometry ---------- */
function renderGeo(){const box=$('#geo');box.textContent='';const r=mulberry32(584);const U=(a,b)=>a+(b-a)*r();
  const panels=[['Vanilla：query 聚在 k₀ 旁，key 在對側',100,[-20,20],[150,210],[60,110]],['Sink／Gated：query 與 key 分散',60,[0,360],[0,360],[30,110]]];
  panels.forEach(([ttl,k0r,qa,ka,rr])=>{const svg=el('svg',{viewBox:'0 0 300 330',role:'img','aria-label':ttl+'（示意）'});box.appendChild(svg);
    const P=(ang,rad)=>[150+rad*Math.cos(ang*Math.PI/180),170-rad*Math.sin(ang*Math.PI/180)];
    el('circle',{cx:150,cy:170,r:120,class:'ring'},svg);
    const tt=tx(svg,150,22,'',{class:'boxt','text-anchor':'middle'});
    const parts=ttl.split('k₀');tt.textContent=parts[0];if(parts.length>1){const a=el('tspan',null,tt);a.textContent='k';const s=el('tspan',{'baseline-shift':'sub','font-size':'75%'},tt);s.textContent='0';const b=el('tspan',null,tt);b.textContent=parts[1]}
    for(let i=0;i<40;i++){const[x,y]=P(U(qa[0],qa[1]),U(rr[0],rr[1]));el('circle',{cx:x,cy:y,r:3.5,class:'q-pt'},svg)}
    for(let i=0;i<40;i++){const[x,y]=P(U(ka[0],ka[1]),U(rr[0],rr[1]));el('rect',{x:x-3,y:y-3,width:6,height:6,class:'k-pt'},svg)}
    const[sx,sy]=P(0,k0r);tx(svg,sx,sy+6,'★',{'text-anchor':'middle',style:'fill:var(--accent);font-size:18px'})})}

/* ---------- C1 ---------- */
const C1Z=[1.0,.5,2.0,-.5,0],C1V=[[1,0],[0,1],[-1,.5],[.5,-1],[.8,.8]],V0D=[Math.SQRT1_2,Math.SQRT1_2];
const c1={mode:'vanilla',z0:4,s:0,sink:4,rho:0};
function c1calc(st){const z=[st.z0,...C1Z.map(c=>c+st.s)],v=[[st.rho*V0D[0],st.rho*V0D[1]],...C1V];
  const wsum=(w,off)=>[0,1].map(i=>w.reduce((a,x,j)=>a+x*v[j+off][i],0));
  if(st.mode==='vanilla'){const L=lse(z),A=z.map(x=>Math.exp(x-L)),Ln=lse(z.slice(1)),G=sig(Ln-st.z0),At=z.slice(1).map(x=>Math.exp(x-Ln));
    const Od=wsum(A,0),Og=wsum(At,1).map(x=>G*x);return{z,A,As:A[0],G,Gsub:1-A[0],LSE:Ln,ref:Ln,Od,Og,err:Math.hypot(Od[0]-Og[0],Od[1]-Og[1])}}
  const Lt=lse(z),La=lse([...z,st.sink]),A=z.map(x=>Math.exp(x-La)),As=Math.exp(st.sink-La),G=sig(Lt-st.sink),At=z.map(x=>Math.exp(x-Lt));
  const Od=wsum(A,0),Og=wsum(At,0).map(x=>G*x);return{z,A,As,G,Gsub:1-As,LSE:Lt,ref:Lt,Od,Og,err:Math.hypot(Od[0]-Og[0],Od[1]-Og[1])}}
function renderC1(){const st=c1,r=c1calc(st),van=st.mode==='vanilla';
  $$('#c1-mode button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===st.mode)));
  $('#c1-sinkrow').classList.toggle('hide',van);
  [['z0',1],['s',1],['sink',1],['rho',2]].forEach(([k,d])=>{const i=$('#c1-'+k);i.value=st[k];$('#c1-'+k+'o').textContent=st[k].toFixed(d);i.setAttribute('aria-valuetext',`${k} = ${st[k].toFixed(d)}`)});
  // bars
  const bb=$('#c1-bars'),W=widthOf(bb,260,420),labs=van?['0（首）','1','2','3','4','5']:['sink','0','1','2','3','4','5'],vals=van?r.A:[r.As,...r.A],hat=van?[0]:[0];
  const F=frame($('svg',bb),{W,H:230,m:{l:36,r:8,t:24,b:36},x:[0,labs.length],y:[0,1],yt:[0,.5,1],yf:v=>String(v),xl:'token',yl:'注意力權重 A'});
  const bw=(F.x1-F.x0)/labs.length;
  vals.forEach((v,i)=>{const x=F.x0+i*bw+bw*.15,w=bw*.7;el('rect',{x,y:F.Y(v),width:w,height:F.y0-F.Y(v),class:hat.includes(i)?'bar-h':'bar'},F.g);
    tx(F.g,x+w/2,F.Y(v)-5,v.toFixed(2),{class:'tk','text-anchor':'middle',style:'fill:var(--ink)'});tx(F.g,x+w/2,F.y0+15,labs[i],{class:'tk','text-anchor':'middle'})});
  // sigmoid
  const sb=$('#c1-sig'),Ws=widthOf(sb,240,340),p=van?st.z0:st.sink;
  const S=frame($('svg',sb),{W:Ws,H:230,m:{l:36,r:10,t:24,b:36},x:[-4,12],y:[0,1],xt:[-4,0,4,8,12],yt:[0,.5,1],yf:v=>String(v),xl:van?'第一個 token 的 logit':'sink 參數',yl:'門控 G'});
  const pts=[];for(let i=0;i<=80;i++){const q=-4+16*i/80;pts.push([S.X(q),S.Y(sig(r.ref-q))])}el('path',{d:lp(pts),class:'acc',style:'stroke-width:2.5'},S.g);
  if(r.ref>=-4&&r.ref<=12){el('line',{x1:S.X(r.ref),x2:S.X(r.ref),y1:S.y0,y2:S.y1,class:'ref'},S.g);tx(S.g,S.X(r.ref)+4,S.y1+10,'G = 0.5',{class:'tk'})}
  el('circle',{cx:S.X(p),cy:S.Y(r.G),r:5.5,class:'kdot'},S.g);
  // vectors
  const vb=$('#c1-vec'),Wv=widthOf(vb,200,260),Hv=Wv,R0=Math.max(.15,1.25*Math.max(...r.Od.map(Math.abs),...r.Og.map(Math.abs))),X=lin(-R0,R0,10,Wv-10),Yv=lin(-R0,R0,Hv-10,24),svg=$('svg',vb);
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${Wv} ${Hv}`);svg.setAttribute('width',Wv);svg.setAttribute('height',Hv);
  const g=el('g',null,svg);el('line',{x1:X(-R0),x2:X(R0),y1:Yv(0),y2:Yv(0),class:'faint'},g);el('line',{x1:X(0),x2:X(0),y1:Yv(-R0),y2:Yv(R0),class:'faint'},g);tx(g,Wv-10,Yv(0)-6,'±'+R0.toFixed(2),{class:'tk','text-anchor':'end'});
  arrow(g,X(0),Yv(0),X(r.Od[0]),Yv(r.Od[1]),'ink');arrow(g,X(0),Yv(0),X(r.Og[0]),Yv(r.Og[1]),'acc',true);
  if(r.err>1e-9)el('line',{x1:X(r.Og[0]),y1:Yv(r.Og[1]),x2:X(r.Od[0]),y2:Yv(r.Od[1]),class:'ink dot',style:'stroke-width:1.5'},g);
  {const t=tx(g,8,16,r.err<1e-9?'兩種算法相同':'兩者之差 = A',{class:'al'});if(r.err>=1e-9){const s=el('tspan',{'baseline-shift':'sub','font-size':'75%'},t);s.textContent='0';const e=el('tspan',null,t);e.textContent='·ρ'}}
  // table
  $('#c1-As').textContent=fmt4(r.As);$('#c1-Gs').textContent=fmt4(r.Gsub);$('#c1-G').textContent=fmt4(r.G);$('#c1-L').textContent=r.LSE.toFixed(4);
  const vec=o=>`(${o[0].toFixed(4)}, ${o[1].toFixed(4)})`;$('#c1-Od').textContent=vec(r.Od);$('#c1-Og').textContent=vec(r.Og);$('#c1-err').textContent=r.err<1e-12?'0':r.err.toFixed(4);
  $('#c1-note').innerHTML=!van&&r.A[0]>.2?`第一個 token 也收到 A<sub>0</sub> = ${r.A[0].toFixed(4)} 的權重（Sec. 6.2 的 DeepSeek-V4 情形）。若第一個 token 的 value 小（ρ 接近 0），論文建議 A<sub>sink</sub> 應同時計入它；這裡的 G 只計入合成 sink。`:(van?'Vanilla：sink 成分是第一個 token；ρ &gt; 0 時兩種輸出的差恰為 A<sub>0</sub>·ρ。':'Sink：合成 sink 沒有 value，任何 ρ 下兩種輸出都相同。');
  $('#c1-bars svg').setAttribute('aria-label',`A_sink = ${r.As.toFixed(2)}，G = ${r.G.toFixed(2)}`);
  announce('c1',`A_sink = ${r.As.toFixed(4)}，G = ${r.G.toFixed(4)}，兩種輸出之差 ${r.err<1e-12?0:r.err.toFixed(4)}`);
}
function c1setLabels(){const van=c1.mode==='vanilla';const L=$('#c1-sigl'),M=$('#c1-lsel');
  const tex=van?['\\sigma(\\mathrm{LSE}_{\\neg0}-z_0)','\\mathrm{LSE}_{\\neg0}']:['\\sigma(\\mathrm{LSE}_{\\text{tok}}-\\mathrm{sink})','\\mathrm{LSE}_{\\text{tok}}'];
  if(window.katex){katex.render(tex[0],L,{throwOnError:false});katex.render(tex[1],M,{throwOnError:false})}else{L.textContent='$'+tex[0]+'$';M.textContent='$'+tex[1]+'$'}}
function initC1(){$$('#c1-mode button').forEach(b=>b.addEventListener('click',()=>{c1.mode=b.dataset.v;Object.assign(c1,c1.mode==='vanilla'?{z0:4,s:0,rho:0}:{z0:.5,s:0,sink:4,rho:0});c1setLabels();renderC1()}));
  ['z0','s','sink','rho'].forEach(k=>$('#c1-'+k).addEventListener('input',e=>{c1[k]=+e.target.value;renderC1()}));renderC1()}

/* ---------- C2 ---------- */
const c2={N:16,beta:4,m:3,showGrad:false,sel:0};
function c2calc(N,beta,m){const x=profile(N,beta),st=cvStats(x),T=topM(x,m),xr=x.filter((_,h)=>!T.has(h)),sr=cvStats(xr),g=gradNcv2(x);
  return{x,st,T,sr,g,L7:1e-4*st.ncv2,L8:1e-2*(N-m)*sr.cv**2,r8:(N-m)*sr.cv**2,id:x.reduce((a,v)=>a+(v/st.mu-1)**2,0),xg:x.reduce((a,v,i)=>a+v*g[i],0)}}
function renderC2(){const{N,beta,m}=c2,r=c2calc(N,beta,m);
  $$('#c2-N button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.v===N)));
  $('#c2-bo').textContent=beta.toFixed(1);$('#c2-mo').textContent=m;$('#c2-m').max=N-2;$('#c2-m').value=m;
  $('#c2-b').setAttribute('aria-valuetext','β = '+beta.toFixed(1));$('#c2-m').setAttribute('aria-valuetext','m = '+m);
  const box=$('#c2-fig'),W=widthOf(box,300,640),H=c2.showGrad?250:200;
  const R=headRow($('svg',box),r.x,r.T,{W,H:H,axis:true,top:20});
  if(c2.showGrad){/* shrink plot: arrows drawn under axis */}
  const mu=r.st.mu,mur=r.sr.mu;
  el('line',{x1:R.x0,x2:R.x1,y1:R.Y(mu),y2:R.Y(mu),class:'ref'},R.g);tx(R.g,R.x1,R.Y(mu)-4,'μ',{class:'labm','text-anchor':'end'});
  if(m>0){el('line',{x1:R.x0,x2:R.x1,y1:R.Y(mur),y2:R.Y(mur),class:'par dash',style:'stroke-width:1.5'},R.g);{const t=tx(R.g,R.x1,R.Y(mur)+14,'μ',{class:'labm','text-anchor':'end',style:'fill:var(--c-par)'});const s=el('tspan',{'baseline-shift':'sub','font-size':'75%'},t);s.textContent='r'}}
  el('rect',{x:R.X(c2.sel)-R.bw/2,y:R.Y(1)-2,width:R.bw,height:R.y0-R.Y(1)+4,class:'bar-sel',id:'c2-selbox',style:document.activeElement===box?'':'display:none'},R.g);
  const gmax=Math.max(...r.g.map(Math.abs));
  if(c2.showGrad){if(gmax<1e-12){$('#c2-gnote').textContent='所有 head 相同，梯度為 0。'}else{$('#c2-gnote').textContent='箭頭為梯度下降方向 −∂(N·CV²)/∂Imp，長度依最大值正規化。';
    r.g.forEach((gv,h)=>{const L=Math.abs(gv)/gmax*26,cx=R.X(h),cy=R.Y(r.x[h]),dir=gv>0?1:-1;if(L<2)return;arrow(R.g,cx,cy,cx,cy+dir*L,'acc')})}}else $('#c2-gnote').textContent='';
  const rows=[['μ',r.st.mu.toFixed(4)],['std（母體）',r.st.sd.toFixed(4)],['CV',r.st.cv.toFixed(4)],['N·CV²',r.st.ncv2.toFixed(4)],['Σ(x/μ − 1)²',r.id.toFixed(4)],['Eq. 7 損失（λ = 10⁻⁴）',r.L7.toFixed(6)],
    ['shared head',m?[...r.T].sort((a,b)=>a-b).join('、'):'無'],['μ<sub>r</sub>（routed 平均）',r.sr.mu.toFixed(4)],['CV<sub>r</sub>',r.sr.cv.toFixed(4)],['(N − m)·CV<sub>r</sub>²',r.r8.toFixed(4)],['Eq. 8 損失（λ = 10⁻²）',r.L8.toFixed(6)],['Σ x·梯度',Math.abs(r.xg)<1e-12?'≈ 0':r.xg.toExponential(2)]];
  $('#c2-tab tbody').innerHTML=rows.map(([a,b])=>`<tr><td>${a}</td><td class="num">${b}</td></tr>`).join('');
  $('#c2-heads tbody').innerHTML=r.x.map((v,h)=>`<tr><td>${h}</td><td class="num">${v.toFixed(3)}</td><td>${r.T.has(h)?'shared':'routed'}</td><td class="num">${r.g[h].toFixed(3)}</td></tr>`).join('');
  announce('c2',`N = ${N}，β = ${beta.toFixed(1)}，CV = ${r.st.cv.toFixed(4)}，N·CV² = ${r.st.ncv2.toFixed(4)}`);
  c2.last=r;
}
function initC2(){const box=$('#c2-fig');box.tabIndex=0;
  $$('#c2-N button').forEach(b=>b.addEventListener('click',()=>{c2.N=+b.dataset.v;c2.m=Math.min(c2.m,c2.N-2);c2.sel=Math.min(c2.sel,c2.N-1);renderC2()}));
  $('#c2-b').addEventListener('input',e=>{c2.beta=+e.target.value;renderC2()});$('#c2-m').addEventListener('input',e=>{c2.m=+e.target.value;renderC2()});
  $('#c2-g').addEventListener('change',e=>{c2.showGrad=e.target.checked;renderC2()});
  const say=()=>{const r=c2.last,h=c2.sel;const n=$('#c2-live');n.textContent=`head ${h}，Imp ${r.x[h].toFixed(2)}，${r.T.has(h)?'shared':'routed'}`};
  box.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){c2.sel=(c2.sel+(e.key==='ArrowRight'?1:-1)+c2.N)%c2.N;renderC2();$('#c2-selbox').style.display='';say();e.preventDefault()}});
  box.addEventListener('focus',()=>{const s=$('#c2-selbox');if(s)s.style.display='';say()});box.addEventListener('blur',()=>{const s=$('#c2-selbox');if(s)s.style.display='none'});
  renderC2()}

/* ---------- C3 ---------- */
const C3N=16,C3STEPS=3000,C3ETA=.05;
function c3new(mode,pin,m){const x0=profile(C3N,4),T=topM(x0,m);return{mode,pin,m,x0,T,th:x0.map(a=>Math.log(a/(1-a))),step:0,hist:[],running:false}}
function c3loss(st,x){if(st.mode==='eq7')return cvStats(x).ncv2;const xr=x.filter((_,h)=>!st.T.has(h));return(C3N-st.m)*cvStats(xr).cv**2}
function c3rec(st){const x=st.th.map(sig),xr=x.filter((_,h)=>!st.T.has(h));let mn=Infinity,mx=-Infinity,s=0;for(const a of xr){mn=Math.min(mn,a);mx=Math.max(mx,a);s+=a}
  st.hist.push({s:st.step,mn,mx,mean:s/xr.length,sh:[...st.T].sort((a,b)=>a-b).map(h=>x[h]),L:c3loss(st,x),cv:cvStats(x).cv})}
function stepC3(st){const x=st.th.map(sig);let gi;
  if(st.mode==='eq7'){const g=gradNcv2(x);gi=g}else{const idx=[];for(let h=0;h<C3N;h++)if(!st.T.has(h))idx.push(h);const g=gradNcv2(idx.map(h=>x[h]));gi=new Array(C3N).fill(0);idx.forEach((h,i)=>gi[h]=g[i])}
  for(let h=0;h<C3N;h++){if(st.T.has(h)&&(st.pin||st.mode==='eq8'))continue;st.th[h]-=C3ETA*gi[h]*x[h]*(1-x[h])}
  st.step++;if(st.step%10===0)c3rec(st)}
function c3runAll(st){while(st.step<C3STEPS)stepC3(st);return st}
let c3=c3new('eq7',true,3),c3raf=0;c3rec(c3);
function renderC3(){const st=c3,x=st.th.map(sig),last=st.hist[st.hist.length-1];
  $$('#c3-loss button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===st.mode)));
  const pin=$('#c3-pin');pin.disabled=st.mode==='eq8';pin.checked=st.mode==='eq8'?true:st.pin;
  $('#c3-pinnote').textContent=st.mode==='eq8'?'Eq. 8 中 shared head 不在損失內，梯度本來是 0。':'';
  $('#c3-mo').textContent=st.m;$('#c3-m').value=st.m;$('#c3-m').setAttribute('aria-valuetext','m = '+st.m);
  const rb=$('#c3-row');headRow($('svg',rb),x,st.T,{W:widthOf(rb,300,900),H:150,axis:true,ghost:st.x0,top:18});
  const ib=$('#c3-imp'),W=widthOf(ib,260,460);
  const F=frame($('svg',ib),{W,H:230,m:{l:40,r:44,t:24,b:36},x:[0,C3STEPS],y:[0,1],xt:[0,1000,2000,3000],xf:v=>v.toLocaleString('en-US'),yt:[0,.5,1],yf:v=>String(v),xl:'梯度下降步數',yl:'Imp'});
  const H=st.hist;if(H.length){const up=H.map(h=>[F.X(h.s),F.Y(h.mx)]),dn=H.map(h=>[F.X(h.s),F.Y(h.mn)]).reverse();el('path',{d:lp(up.concat(dn))+'Z',class:'band'},F.g);
    el('path',{d:lp(H.map(h=>[F.X(h.s),F.Y(h.mean)])),class:'par'},F.g);
    const labs=[[F.Y(last.mean),last.mean,'--c-par']];
    H[0].sh.forEach((_,k)=>{el('path',{d:lp(H.map(h=>[F.X(h.s),F.Y(h.sh[k])])),class:'perp dash',style:'stroke-width:2'},F.g);labs.push([F.Y(last.sh[k]),last.sh[k],'--c-perp'])});
    labs.sort((a,b)=>a[0]-b[0]);for(let i=1;i<labs.length;i++)labs[i][0]=Math.max(labs[i][0],labs[i-1][0]+13);
    labs.forEach(([y,v,c])=>tx(F.g,F.x1+4,y+4,v.toFixed(2),{class:'tk',style:`fill:var(${c})`}))}
  const lb=$('#c3-loss-fig'),W2=widthOf(lb,260,460),L0=Math.max(H[0].L,1);
  const G=frame($('svg',lb),{W:W2,H:230,m:{l:48,r:12,t:24,b:36},x:[0,C3STEPS],xlog:false,y:[-6,Math.log10(L0)],xt:[0,1000,2000,3000],xf:v=>v.toLocaleString('en-US'),yt:[[-6,'10⁻⁶'],[-4,'10⁻⁴'],[-2,'10⁻²'],[0,'1']].filter(t=>t[0]<=Math.log10(L0)),xl:'梯度下降步數',yl:'損失（對數）'});
  el('path',{d:lp(H.map(h=>[G.X(h.s),G.Y(Math.max(-6,Math.log10(Math.max(h.L,1e-300))))])),class:'tot'},G.g);
  $('#c3-ro').innerHTML=`步數 <b>${st.step}</b>　損失 <b>${last.L<1e-4?last.L.toExponential(2):last.L.toFixed(4)}</b>　routed 平均 <b class="acc">${last.mean.toFixed(4)}</b>　routed 最小／最大 <b>${last.mn.toFixed(4)}</b>／<b>${last.mx.toFixed(4)}</b>　全部 CV <b>${last.cv.toFixed(4)}</b>`;
  $('#c3-run').textContent=st.running?'暫停':(st.step>=C3STEPS?'已完成':'執行');$('#c3-run').disabled=st.step>=C3STEPS&&!st.running;
}
function c3done(){c3.running=false;renderC3();const l=c3.hist[c3.hist.length-1];announce('c3',`第 ${c3.step} 步，routed 平均 ${l.mean.toFixed(2)}`)}
function runC3(){if(c3.running){c3.running=false;cancelAnimationFrame(c3raf);c3done();return}if(c3.step>=C3STEPS)return;c3.running=true;renderC3();
  if(RM){const chunk=()=>{if(!c3.running)return;for(let i=0;i<500&&c3.step<C3STEPS;i++)stepC3(c3);if(c3.step<C3STEPS)setTimeout(chunk,0);else c3done()};setTimeout(chunk,0);return}
  const tick=()=>{if(!c3.running)return;for(let i=0;i<30&&c3.step<C3STEPS;i++)stepC3(c3);renderC3();if(c3.step<C3STEPS)c3raf=requestAnimationFrame(tick);else c3done()};c3raf=requestAnimationFrame(tick)}
function resetC3(o){c3.running=false;cancelAnimationFrame(c3raf);const n=Object.assign({mode:c3.mode,pin:c3.pin,m:c3.m},o||{});c3=c3new(n.mode,n.pin,n.m);c3rec(c3);renderC3()}
function initC3(){$$('#c3-loss button').forEach(b=>b.addEventListener('click',()=>resetC3({mode:b.dataset.v})));
  $('#c3-pin').addEventListener('change',e=>resetC3({pin:e.target.checked}));$('#c3-m').addEventListener('input',e=>resetC3({m:+e.target.value}));
  $('#c3-run').addEventListener('click',runC3);$('#c3-reset').addEventListener('click',()=>resetC3());renderC3()}

/* ---------- C4 ---------- */
const Z4=[5.0,1.2,-0.3,2.1,0.7,-1.5,3.0,0.2,1.8,-0.8,2.6,0.9];
const c4={B:4,c:1,mode:'sink',done:0,m:-Infinity,l:0,trace:[]};
function c4z(){return Z4.map(a=>a*c4.c)}
function c4step(st){const z=Z4.map(a=>a*st.c);if(st.done>=z.length)return;const blk=z.slice(st.done,st.done+st.B);let mn=Math.max(st.m,...blk);
  st.l=(st.m===-Infinity?0:st.l*Math.exp(st.m-mn))+blk.reduce((a,v)=>a+Math.exp(v-mn),0);st.m=mn;st.done+=blk.length;st.trace.push([st.m,st.l,st.m+Math.log(st.l)])}
function c4result(st){const z=Z4.map(a=>a*st.c),L=st.m+Math.log(st.l),direct=lse(z),mx=Math.max(...z);
  const naive=mx>709.78?Infinity:Math.log(z.reduce((a,v)=>a+Math.exp(v),0));const sink=3*st.c;
  const em=-Math.expm1(z[0]-L);return{L,direct,naive,Gs:sig(L-sink),sink,v1:1-Math.exp(z[0]-L),v2:em,v3:sig(lse(z.slice(1))-z[0])}}
function renderC4(){const z=c4z();
  $$('#c4-B button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.v===c4.B)));$$('#c4-c button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.v===c4.c)));
  $$('#c4-mode button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===c4.mode)));
  const prev=c4.done-(c4.trace.length?Math.min(c4.B,c4.done-(c4.trace.length-1)*c4.B):0);
  $('#c4-blocks').innerHTML=z.map((v,i)=>`<span class="${i<c4.done?'done':''}${i<c4.done&&i>=prev?' cur':''}">${i<c4.done?'✓ ':''}${v.toFixed(c4.c>=10?0:1)}<i>${i===0?'首':'z'+i}</i></span>`).join('');
  $('#c4-blocks').setAttribute('aria-label',`已處理 ${c4.done} / 12 個 logit`);
  $('#c4-tab tbody').innerHTML=c4.trace.map((t,i)=>`<tr><td>${i+1}</td><td class="num">${t[0].toFixed(4)}</td><td class="num">${t[1].toFixed(4)}</td><td class="num">${t[2].toFixed(4)}</td></tr>`).join('')||'<tr><td colspan="4">尚未處理</td></tr>';
  let rows=[];if(c4.done>=12){const r=c4result(c4);rows=[['分塊 LSE（m + ln ℓ）',r.L.toFixed(6)],['直接 LSE（一次減最大值）',r.direct.toFixed(6)],['直接加總 ln Σ e^z',r.naive===Infinity?'溢位（Infinity）':r.naive.toFixed(6)]];
    if(c4.mode==='sink')rows.push([`Sink：G = σ(LSE − sink)，sink = ${r.sink}`,r.Gs.toFixed(6)]);
    else rows.push(['Vanilla：1 − exp(z₀ − LSE)（只用 kernel LSE）',sci(r.v1)],['Vanilla：−expm1(z₀ − LSE)（只用 kernel LSE）',sci(r.v2)],['Vanilla：σ(LSE¬0 − z₀)，需另行取得 LSE¬0（論文未說明）',sci(r.v3)]);
    c4.res=r}else c4.res=null;
  $('#c4-res tbody').innerHTML=rows.map(([a,b])=>`<tr><td>${a}</td><td class="num">${b}</td></tr>`).join('')||'<tr><td colspan="2">處理完全部 12 個 logit 後顯示</td></tr>';
  $('#c4-next').disabled=c4.done>=12;
}
function resetC4(o){Object.assign(c4,o||{},{done:0,m:-Infinity,l:0,trace:[]});renderC4()}
function initC4(){$('#c4-B').innerHTML=[1,2,3,4,6,12].map(b=>`<button type="button" data-v="${b}" aria-pressed="false">${b}</button>`).join('');
  $$('#c4-B button').forEach(b=>b.addEventListener('click',()=>resetC4({B:+b.dataset.v})));$$('#c4-c button').forEach(b=>b.addEventListener('click',()=>resetC4({c:+b.dataset.v})));
  $$('#c4-mode button').forEach(b=>b.addEventListener('click',()=>{c4.mode=b.dataset.v;renderC4()}));
  const say=()=>{const r=c4.res;announce('c4',r?`LSE = ${r.L.toFixed(4)}`:`已處理 ${c4.done} 個 logit`)};
  $('#c4-next').addEventListener('click',()=>{c4step(c4);renderC4();say()});$('#c4-all').addEventListener('click',()=>{while(c4.done<12)c4step(c4);renderC4();say()});
  $('#c4-reset').addEventListener('click',()=>resetC4());renderC4()}

/* ---------- timeline ---------- */
const TL_GROUP={'1706.03762':['base','asink-softmax-normalization-and-first-token'],'2309.17453':['app','asink-sink-applications-kv-cache-and-streaming'],'2410.10819':['app','asink-sink-applications-kv-cache-and-streaming'],
 '2410.10781':['cause','asink-mechanism-studies-value-drain-over-mixing'],'2410.13835':['cause','asink-mechanism-studies-value-drain-over-mixing'],'2504.02732':['cause','asink-mechanism-studies-value-drain-over-mixing'],'2510.06477':['cause','asink-mechanism-studies-value-drain-over-mixing'],
 '2504.20966':['soft','asink-eliminating-sink-softmax-modification'],'2508.10925':['soft','asink-eliminating-sink-softmax-modification'],'2306.12929':['gate','asink-gated-attention-explicit-gating'],'2505.06708':['gate','asink-gated-attention-explicit-gating'],
 '1701.06538':['moe','asink-expert-collapse-and-load-balancing'],'2101.03961':['moe','asink-expert-collapse-and-load-balancing'],'2401.06066':['moe','asink-expert-collapse-and-load-balancing'],'2602.01203':['me','asink-native-moe-and-head-collapse']};
const TL_NAME={base:'基礎：softmax 注意力',app:'sink 現象與應用',cause:'sink 成因',soft:'改 softmax',gate:'門控',moe:'MoE 負載平衡',me:'本文',other:'其他'};
function renderTimeline(){const track=$('#rail-track'),chips=$('#tl-chips');let data;
  try{data=JSON.parse($('#timeline-data').textContent);if(!Array.isArray(data))throw 0}catch(e){track.innerHTML='<p class="note">時間線資料尚未提供</p>';$('.tl-ctl').classList.add('hide');return}
  data=data.map((e,i)=>({e,i})).sort((a,b)=>(+a.e.year||0)-(+b.e.year||0)||a.i-b.i).map(o=>o.e);
  const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const used=new Set();
  data.forEach(e=>{const g=TL_GROUP[e.arxiv]||['other',null];used.add(g[0]);const a=document.createElement('article');a.className='ev'+(g[0]==='me'?' me':'');a.dataset.g=g[0];
    a.innerHTML=`<span class="yr">${esc(e.year)}</span><span class="grp">${TL_NAME[g[0]]}</span><h3>${esc(e.title)}</h3><div class="who">${esc(e.authors_short)}</div>${e.role?`<p title="${esc(e.role)}">${esc(e.role)}</p>`:''}<div class="lk">${e.arxiv?`<a href="https://arxiv.org/abs/${esc(e.arxiv)}">arXiv ↗</a>`:''}${g[1]?`<a href="${g[1]}.html">筆記</a>`:''}</div>`;track.appendChild(a)});
  const on=new Set(used);
  Object.keys(TL_NAME).filter(k=>used.has(k)).forEach(k=>{const b=document.createElement('button');b.type='button';b.className='chip';b.setAttribute('aria-pressed','true');b.textContent=TL_NAME[k];b.dataset.g=k;
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
function initEqScroll(){$$('.eq-m').forEach(e=>{if(e.scrollWidth>e.clientWidth+1){e.tabIndex=0;e.setAttribute('role','region');e.setAttribute('aria-label','公式，可橫向捲動')}})}

/* ---------- self test ---------- */
function selfTest(){let pass=0,fail=0;const msgs=[];const ok=(c,m)=>{console.assert(c,m);if(c)pass++;else{fail++;msgs.push(m)}};
  const near=(a,b,t)=>Math.abs(a-b)<=t;const rel=(a,b,t)=>Math.abs(a-b)<=t*Math.abs(b);
  // C1
  const C1T=[[{mode:'vanilla',z0:4,s:0,rho:0},.8034,.1966,2.5925,[-.0525,.0815],0],[{mode:'vanilla',z0:4,s:0,rho:1},.8034,.1966,2.5925,[.5156,.6495],.8034],
    [{mode:'vanilla',z0:8,s:0,rho:0},.9955,.0045,2.5925,[-.0012,.0018],0],[{mode:'vanilla',z0:-2,s:0,rho:0},.0100,.9900,2.5925,[-.2643,.4102],0],[{mode:'vanilla',z0:4,s:2,rho:0},.3561,.6439,4.5925,[-.1719,.2668],0],
    [{mode:'sink',z0:.5,s:0,sink:4,rho:0},.7843,.2157,2.7088,[-.0513,.0795],0],[{mode:'sink',z0:.5,s:0,sink:4,rho:1},.7843,.2157,2.7088,[-.0345,.0963],0],
    [{mode:'sink',z0:.5,s:0,sink:-4,rho:0},.0012,.9988,2.7088,[-.2374,.3684],0],[{mode:'sink',z0:4,s:0,sink:4,rho:0},.4455,.5545,4.2189,[-.0291,.0452],0]];
  C1T.forEach(([st,As,G,L,Od,err],i)=>{const r=c1calc(st);ok(near(r.As,As,1e-4)&&near(r.G,G,1e-4)&&near(r.LSE,L,1e-4)&&near(r.Od[0],Od[0],1e-4)&&near(r.Od[1],Od[1],1e-4)&&(err?near(r.err,err,1e-4):r.err<1e-12)&&Math.abs(r.G-r.Gsub)<1e-12,`C1 row ${i}`)});
  {const r=c1calc({mode:'vanilla',z0:4,s:0,rho:1});ok(Math.abs(r.err-r.As)<1e-9,'C1 err = A0·rho')}
  {const r=c1calc({mode:'vanilla',z0:12,s:-4,rho:0});ok(rel(r.G,1.504e-6,1e-3),'C1 z0=12 s=-4: '+r.G)}
  // C2
  {const r=cvStats([.9,.1,.1,.1]);ok(near(r.cv,1.1547005,1e-6)&&near(r.ncv2,5.333333,1e-6),'cvStats example')}
  const C2T=[[16,4,3,.2869,.2575,.8974,12.8858,'2,9,11',.1787,.6579,5.6261,.001289,.056261],[16,0,3,.95,0,0,0,'0,1,2',.95,0,0,0,0],[16,8,3,.1861,.2422,1.3020,27.1220,'2,9,11',.0838,.6238,5.0585,.002712,.050585],
    [10,4,2,.2979,.2758,.9260,8.5749,'1,4',.1752,.6596,3.4807,.000857,.034807],[13,4,3,.2910,.2645,.9087,10.7350,'7,9,11',.1626,.6091,3.7104,.001074,.037104]];
  C2T.forEach(([N,b,m,mu,sd,cv,nc,sh,mur,cvr,r8,L7,L8])=>{const r=c2calc(N,b,m);const z=(a,e)=>e===0?Math.abs(a)<1e-12:(Math.abs(e)<.01?rel(a,e,1e-3):near(a,e,1e-4));
    ok(z(r.st.mu,mu)&&z(r.st.sd,sd)&&z(r.st.cv,cv)&&z(r.st.ncv2,nc)&&[...r.T].sort((a,c)=>a-c).join(',')===sh&&z(r.sr.mu,mur)&&z(r.sr.cv,cvr)&&z(r.r8,r8)&&z(r.L7,L7)&&z(r.L8,L8)&&Math.abs(r.xg)<1e-12,`C2 N=${N} β=${b} m=${m}`)});
  {const r=c2calc(16,0,3);ok(r.st.cv===0,'C2 β=0 CV exactly 0')}
  // C3
  const C3T=[['eq7',true,3,[.5033,.6460,.7424,.7805],.1181],['eq7',false,3,[.4287,.4415,.4416,.4416],0],['eq8',true,3,[.1614,.1614,.1614,.1614],0],['eq8',true,1,[.2337,.2337,.2337,.2337],0],['eq7',true,1,[.4835,.6233,.7520,.8366],.0169]];
  C3T.forEach(([mode,pin,m,means,L])=>{const st=c3new(mode,pin,m);c3rec(st);c3runAll(st);const at=s=>st.hist.find(h=>h.s===s);
    ok([100,300,1000,3000].every((s,i)=>near(at(s).mean,means[i],5e-4))&&near(at(3000).L,L,5e-4),`C3 ${mode} pin=${pin} m=${m}: ${[100,300,1000,3000].map(s=>at(s).mean.toFixed(4))}`)});
  {const st=c3new('eq7',true,3);c3runAll(st);const x=st.th.map(sig);ok(near(x[2],.739,5e-4)&&near(x[9],.578,5e-4)&&near(x[11],.95,5e-4),'C3 shared unchanged')}
  // C4
  const run4=(B,c)=>{const st={B,c,done:0,m:-Infinity,l:0,trace:[]};while(st.done<12)c4step(st);return st};
  {const st=run4(4,1);ok(st.trace.map(t=>t[2].toFixed(4)).join()==='5.0792,5.2159,5.3308','C4 B=4 trace');ok(near(c4result(st).L,5.330816,1e-6),'C4 LSE c=1 B=4')}
  {const st=run4(1,1);ok(near(st.trace[11][2],5.330816,1e-6)&&st.trace.map(t=>t[2].toFixed(4)).slice(0,4).join()==='5.0000,5.0221,5.0270,5.0792','C4 B=1')}
  {const st=run4(12,1);ok(near(c4result(st).L,5.330816,1e-6),'C4 B=12');const r=c4result(st);ok(near(r.Gs,.911397,1e-6)&&rel(r.v1,.28166,1e-3)&&rel(r.v2,.28166,1e-3)&&rel(r.v3,.28166,1e-3),'C4 gates c=1')}
  {const r=c4result(run4(4,10));ok(rel(r.v2,2.0992e-9,1e-3)&&rel(r.v3,2.0992e-9,1e-3),'C4 c=10 vanilla')}
  {const r=c4result(run4(4,200));ok(r.naive===Infinity&&near(r.L,1000,1e-9)&&r.v1===0&&r.v2===0&&r.v3>0&&Math.abs(Math.log10(r.v3)-Math.log10(1.9152e-174))<=1e-3&&near(r.Gs,1,1e-6),'C4 c=200')}
  ok(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow: scrollWidth='+document.documentElement.scrollWidth+' innerWidth='+innerWidth);
  const line=`selfTest: ${pass} passed, ${fail} failed`;console.log(line,msgs);document.title=line;$('#selftest').textContent=line+(msgs.length?' · '+msgs.join(' | '):'');}

/* ---------- boot ---------- */
initTheme();initToc();renderHero();renderFig1();renderGeo();initC1();initC2();initC3();initC4();renderTimeline();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderHero();renderC1();renderC2();renderC3();renderC4();initEqScroll()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false});initEqScroll()}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))addEventListener('load',()=>setTimeout(()=>{renderMath();selfTest()},0));
