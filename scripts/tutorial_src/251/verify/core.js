// Shared algorithms: the page inlines these functions verbatim (design.md §6.4).
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}

// C1: cell counts. rows = [name,ref,year,rep,upd,per,paradigm,sec,rules]
function cellCounts(rows,f){const c={};for(const r of rows){if(f&&!f(r))continue;const k=r[3]+'|'+r[4]+'|'+r[5];c[k]=(c[k]||0)+1}return c}

// C2: S_t = a S_{t-1} - b (S_{t-1} k_t) k_t^T + g v_t k_t^T  (Fig. 3), S is d_v x d_k, here d x d.
function c2data(d,T,seed,mode){const r=mulberry32(seed),K=[],V=[];
  const unit=()=>{let x=new Float64Array(d),n=0;for(let i=0;i<d;i++){x[i]=gauss(r);n+=x[i]*x[i]}n=Math.sqrt(n);if(n<1e-12){x.fill(0);x[0]=1;return x}for(let i=0;i<d;i++)x[i]/=n;return x};
  for(let t=0;t<T;t++){let k=unit();const v=unit();if(mode==='ortho'){k=new Float64Array(d);k[t%d]=1}K.push(k);V.push(v)}return{K,V}}
function c2run(d,T,seed,mode,a,b,g){const {K,V}=c2data(d,T,seed,mode),S=new Float64Array(d*d),Sk=new Float64Array(d);
  for(let t=0;t<T;t++){const k=K[t],v=V[t];
    for(let i=0;i<d;i++){let s=0;for(let j=0;j<d;j++)s+=S[i*d+j]*k[j];Sk[i]=s}
    for(let i=0;i<d;i++)for(let j=0;j<d;j++)S[i*d+j]=a*S[i*d+j]-b*Sk[i]*k[j]+g*v[i]*k[j]}
  const c=[],e=[],over=[];let fro=0;for(let x=0;x<d*d;x++)fro+=S[x]*S[x];
  for(let t=0;t<T;t++){const k=K[t],v=V[t];let ci=0;const rr=new Float64Array(d);
    for(let i=0;i<d;i++){let s=0;for(let j=0;j<d;j++)s+=S[i*d+j]*k[j];rr[i]=s;ci+=s*v[i]}
    let ee=0;for(let i=0;i<d;i++){const q=rr[i]-ci*v[i];ee+=q*q}c.push(ci);e.push(Math.sqrt(ee));
    over.push(mode==='ortho'&&t+d<T)}
  return{c,e,over,fro:Math.sqrt(fro)}}

// C3: floats per layer (example config, not from the paper).
const CFG={nh:8,dh:128,dc:512,N:16};
function footprint(L,W,s,rho,cfg){cfg=cfg||CFG;const kv=cfg.nh*cfg.dh,st=cfg.nh*cfg.dh*cfg.dh;
  return{kv:2*L*kv,win:2*Math.min(L,W+s)*kv,mla:L*cfg.dc,lin:st,ssm:kv*cfg.N,loglin:st*(Math.floor(Math.log2(L))+1),ham:st+2*Math.ceil(rho*L)*kv}}

// C4: n keys hashed uniformly into M slots.
function hashStats(n,M){const lq=Math.log1p(-1/M);const occ=-M*Math.expm1(n*lq);const coll=n<=1?0:-Math.expm1((n-1)*lq);return{load:n/M,occ,occFrac:occ/M,coll}}
function hashSim(nv,Mv,seed){const r=mulberry32(seed),cnt=new Int32Array(Mv),pos=[];for(let i=0;i<nv;i++){const p=Math.floor(r()*Mv);cnt[p]++;pos.push(p)}let col=0;for(const p of pos)if(cnt[p]>=2)col++;let occ=0;for(let j=0;j<Mv;j++)if(cnt[j])occ++;return{cnt,coll:nv?col/nv:0,occ}}
const MV=1024;function nVis(n,M){return Math.min(4096,Math.max(1,Math.round(MV*n/M)))}

if(typeof module!=='undefined')module.exports={mulberry32,gauss,cellCounts,c2run,footprint,CFG,hashStats,hashSim,nVis,MV};
