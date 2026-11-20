/* C3 / C3a toy model and exact Algorithm 1 propagation (mirrors verify/check_toy.py).
   Self-contained: serialized with toString() into a Blob Worker; the main thread calls toyCore(null). Do not minify. */
function toyCore(scope){
  'use strict';
  function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}}
  function lse(a){let m=-Infinity;for(const v of a)if(v>m)m=v;if(!Number.isFinite(m))return m;let s=0;for(const v of a)s+=Math.exp(v-m);return m+Math.log(s)}
  const BETA=3.0;
  // model: next-token log-probs for every generated prefix (prompt token x_0 is not an input)
  function makeModel(V,seed,alpha,tau,T){T=T||6;const offs=[];let tot=0;for(let l=0;l<=T;l++){offs.push(tot);tot+=V**l}
    const lp=[],lq=[],lz=[],lw=[];
    for(let l=0;l<T;l++)for(let v=0;v<V**l;v++){const c=offs[l]+v,r=mulberry32(seed*100003+c),g=[];for(let j=0;j<V;j++)g.push(BETA*r());
      const z=lse(g),p=g.map(x=>x-z),zq=lse(p.map(x=>x/tau)),q=p.map(x=>x/tau-zq);
      lp[c]=p;lq[c]=q;lz[c]=lse(p.map(x=>alpha*x));lw[c]=p.map((x,j)=>alpha*x-q[j])}
    return{V,T,seed,alpha,tau,offs,lp,lq,lz,lw}}
  const code=(M,x,len,t)=>M.offs[t]+Math.floor(x/M.V**(len-t));        // prefix of length t of a length-len sequence x (base V, first token most significant)
  const tok=(M,x,len,t)=>Math.floor(x/M.V**(len-1-t))%M.V;              // token at 0-based position t
  function seqStr(M,x,len){let s='';for(let t=0;t<len;t++)s+='abc'[tok(M,x,len,t)];return s}
  function dists(M){const n=M.V**M.T,lb=new Float64Array(n),lt=new Float64Array(n);
    for(let x=0;x<n;x++){let b=0,tt=0;for(let t=0;t<M.T;t++){const c=code(M,x,M.T,t),v=tok(M,x,M.T,t);b+=M.lp[c][v];tt+=M.alpha*M.lp[c][v]-M.lz[c]}lb[x]=b;lt[x]=tt}
    const la=Array.from(lb,v=>M.alpha*v),z=lse(la);
    return{base:Float64Array.from(lb,Math.exp),temp:Float64Array.from(lt,Math.exp),pow:Float64Array.from(la,v=>Math.exp(v-z))}}
  function stageKernel(M,L){const V=M.V,n=V**L,S=L+1,Sw=new Float64Array(n*S),Sq=new Float64Array(n*S),K=new Float64Array(n*n);
    for(let x=0;x<n;x++)for(let t=L-1;t>=0;t--){const c=code(M,x,L,t),v=tok(M,x,L,t);Sw[x*S+t]=Sw[x*S+t+1]+M.lw[c][v];Sq[x*S+t]=Sq[x*S+t+1]+M.lq[c][v]}
    for(let x=0;x<n;x++)for(let j=0;j<L;j++){const span=V**(L-j),base=Math.floor(x/span)*span,old=Sw[x*S+j];
      for(let s=0;s<span;s++){const y=base+s,d=Sw[y*S+j]-old,A=d>=0?1:Math.exp(d),pr=Math.exp(Sq[y*S+j])/L;K[x*n+y]+=pr*A;K[x*n+x]+=pr*(1-A)}}
    return{L,n,K}}
  function kernels(M,B){const ks=[];for(let k=0;k*B<M.T;k++)ks.push(stageKernel(M,Math.min((k+1)*B,M.T)));return ks}
  function alg1(M,ks,N){let mu=new Float64Array([1]),len=0;
    for(const st of ks){while(len<st.L){const nx=new Float64Array(mu.length*M.V);for(let x=0;x<mu.length;x++){if(!mu[x])continue;const q=M.lq[M.offs[len]+x];for(let v=0;v<M.V;v++)nx[x*M.V+v]+=mu[x]*Math.exp(q[v])}mu=nx;len++}
      for(let i=0;i<N;i++){const nx=new Float64Array(st.n);for(let x=0;x<st.n;x++){const w=mu[x];if(!w)continue;const o=x*st.n;for(let y=0;y<st.n;y++)nx[y]+=w*st.K[o+y]}mu=nx}}
    return mu}
  function tv(p,q){let s=0;for(let i=0;i<p.length;i++)s+=Math.abs(p[i]-q[i]);return s/2}
  // Z_t identity check: with tau=1/alpha, MH log-ratio == sum of log Z_t differences (all pairs)
  function zIdentity(V,T,alpha){const M=makeModel(V,1,alpha,1/alpha,T),n=V**T;let worst=0;
    for(let x=0;x<n;x++)for(let y=0;y<n;y++){let m=0;while(m<T&&tok(M,x,T,m)===tok(M,y,T,m))m++;let r=0,z=0;
      for(let t=m;t<T;t++){const cy=code(M,y,T,t),cx=code(M,x,T,t);r+=M.lw[cy][tok(M,y,T,t)]-M.lw[cx][tok(M,x,T,t)];z+=M.lz[cy]-M.lz[cx]}worst=Math.max(worst,Math.abs(r-z))}
    return worst}
  // C3a: PRNG-driven single-stage MH (L = T = 6)
  function chainStart(M,chainSeed){const r=mulberry32(chainSeed),draw=pre=>{const u=r(),q=M.lq[pre];let c=0;for(let v=0;v<M.V;v++){c+=Math.exp(q[v]);if(u<c)return v}return M.V-1};
    const x=[];for(let t=0;t<M.T;t++)x.push(draw(M.offs[t]+pref(M,x,t)));return{M,r,draw,x,n:0}}
  function pref(M,arr,t){let v=0;for(let i=0;i<t;i++)v=v*M.V+arr[i];return v}
  function chainStep(st){const M=st.M,T=M.T,m=1+Math.floor(st.r()*T),y=st.x.slice(0,m-1);
    for(let t=m-1;t<T;t++)y.push(st.draw(M.offs[t]+pref(M,y,t)));
    const rows=[];let sw=0,sz=0;for(let t=m-1;t<T;t++){const cy=M.offs[t]+pref(M,y,t),cx=M.offs[t]+pref(M,st.x,t),dw=M.lw[cy][y[t]]-M.lw[cx][st.x[t]],dz=M.lz[cy]-M.lz[cx];sw+=dw;sz+=dz;rows.push({t:t+1,dw,dz})}
    const A=sw>=0?1:Math.exp(sw),u=st.r(),acc=u<=A,old=st.x.slice();if(acc)st.x=y;st.n++;
    return{n:st.n,m,old,y,rows,sigma:sw,sigmaZ:sz,A,u,acc}}
  // worker protocol with key cache (LRU 8)
  const cache=new Map();const NLIST=[0,1,2,3,5,10,20,50];
  function entry(msg,progress){let e=cache.get(msg.key);if(e){cache.delete(msg.key);cache.set(msg.key,e);return e}
    if(msg.V===undefined)return null;const t0=Date.now(),M=makeModel(msg.V,msg.seed,msg.alpha,msg.tau),D=dists(M),ks=[];
    for(let k=0;k*msg.B<M.T;k++){ks.push(stageKernel(M,Math.min((k+1)*msg.B,M.T)));if(progress)progress(k+1,Math.ceil(M.T/msg.B))}
    const curve=NLIST.map(N=>({N,tv:tv(alg1(M,ks,N),D.pow)}));
    e={M,D,ks,curve,tvBase:tv(D.base,D.pow),tvTemp:tv(D.temp,D.pow),ms:Date.now()-t0};cache.set(msg.key,e);if(cache.size>8)cache.delete(cache.keys().next().value);return e}
  function handle(msg,progress){const e=entry(msg,progress);if(!e)return{reqId:msg.reqId,miss:true};const N=msg.nList[0],mu=alg1(e.M,e.ks,N),D=e.D,n=D.pow.length;
    const idx=Array.from({length:n},(_,i)=>i).sort((a,b)=>D.pow[b]-D.pow[a]),top=idx.slice(0,10).map(i=>({s:seqStr(e.M,i,e.M.T),base:D.base[i],temp:D.temp[i],pow:D.pow[i],alg:mu[i]}));
    const other={base:1,temp:1,pow:1,alg:1};top.forEach(r=>{other.base-=r.base;other.temp-=r.temp;other.pow-=r.pow;other.alg-=r.alg});
    return{reqId:msg.reqId,key:msg.key,top,other,tvBase:e.tvBase,tvTemp:e.tvTemp,tvAlg:tv(mu,D.pow),curve:e.curve,ms:e.ms}}
  const api={mulberry32,lse,makeModel,dists,kernels,alg1,tv,zIdentity,chainStart,chainStep,handle,seqStr};
  if(scope){scope.onmessage=ev=>{const m=ev.data;scope.postMessage(handle(m,(k,K)=>scope.postMessage({reqId:m.reqId,progress:[k,K]})))}}
  return api}
