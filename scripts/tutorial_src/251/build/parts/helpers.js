const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const NS='http://www.w3.org/2000/svg';
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
function el(tag,a,p){const e=document.createElementNS(NS,tag);if(a)for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function tx(p,x,y,s,a){const t=el('text',Object.assign({x,y},a||{}),p);t.textContent=s;return t}
const lin=(d0,d1,r0,r1)=>v=>r0+(v-d0)/(d1-d0)*(r1-r0);
const lgs=(d0,d1,r0,r1)=>{const a=Math.log(d0),b=Math.log(d1);return v=>r0+(Math.log(v)-a)/(b-a)*(r1-r0)};
function lp(pts){let s='';for(let i=0;i<pts.length;i++)s+=(i?'L':'M')+pts[i][0].toFixed(1)+' '+pts[i][1].toFixed(1);return s}
const f2=v=>(Math.abs(v)<0.005?0:v).toFixed(2),f3=v=>(Math.abs(v)<0.0005?0:v).toFixed(3);
const fmtNum=x=>x.toLocaleString('zh-TW',{minimumFractionDigits:3,maximumFractionDigits:3});
function fmtK(k){if(k<=0.999)return k.toFixed(3);const s=k.toFixed(8);let i=s.indexOf('.')+1;while(i<s.length&&s[i]==='9')i++;return s.slice(0,Math.min(i+1,s.indexOf('.')+7))}
function widthOf(box,min,max){return Math.max(min,Math.min(max,Math.floor(box.clientWidth||max)))}
function frame(svg,o){
  svg.textContent='';svg.setAttribute('viewBox',`0 0 ${o.W} ${o.H}`);svg.setAttribute('width',o.W);svg.setAttribute('height',o.H);
  const m=o.m,x0=m.l,x1=o.W-m.r,y0=o.H-m.b,y1=m.t;
  const X=(o.xlog?lgs:lin)(o.x[0],o.x[1],x0,x1),Y=lin(o.y[0],o.y[1],y0,y1);
  const g=el('g',null,svg);
  (o.yt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:(o.yf?o.yf(v):String(v)),yy=Y(val);el('line',{x1:x0,x2:x1,y1:yy,y2:yy,class:'grid'},g);tx(g,x0-8,yy+4,lab,{class:'tk','text-anchor':'end'})});
  (o.xt||[]).forEach(v=>{const val=Array.isArray(v)?v[0]:v,lab=Array.isArray(v)?v[1]:(o.xf?o.xf(v):String(v)),xx=X(val);el('line',{x1:xx,x2:xx,y1:y0,y2:y0+4,class:'axl'},g);tx(g,xx,y0+18,lab,{class:'tk','text-anchor':'middle'})});
  el('line',{x1:x0,x2:x1,y1:y0,y2:y0,class:'axl'},g);
  if(o.xl)tx(g,(x0+x1)/2,o.H-4,o.xl,{class:'al','text-anchor':'middle'});
  if(o.yl)tx(g,4,12,o.yl,{class:'al'});
  return {X,Y,g,x0,x1,y0,y1};
}
function arrow(g,x0,y0,x1,y1,cls,dash){const dx=x1-x0,dy=y1-y0,L=Math.hypot(dx,dy);if(L<2)return;const ux=dx/L,uy=dy/L,h=Math.min(10,L*.6);
  el('line',{x1:x0,y1:y0,x2:x1-ux*h*.7,y2:y1-uy*h*.7,class:cls+(dash?' dash':'')},g);
  el('path',{d:`M${x1} ${y1}L${x1-ux*h-uy*h*.45} ${y1-uy*h+ux*h*.45}L${x1-ux*h+uy*h*.45} ${y1-uy*h-ux*h*.45}Z`,class:cls+'-f'},g)}
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
