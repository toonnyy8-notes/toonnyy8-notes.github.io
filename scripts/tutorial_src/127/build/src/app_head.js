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
const UNDEF='undef';
function fmt(v,n=4){if(v===UNDEF)return'未定義';if(v===Infinity)return'∞';if(v===-Infinity)return'−∞';if(!Number.isFinite(v))return'—';if(Math.abs(v)<5*10**-(n+1))v=0;return v.toFixed(n).replace('-','−')}
const norm=(v,n)=>Number((+v).toFixed(n));
const live={};function announce(id,s){const n=$('#'+id+'-live');if(!n)return;clearTimeout(live[id]);live[id]=setTimeout(()=>{n.textContent=s},500)}
function segInit(box,opts,cur,label,onPick){box.textContent='';opts.forEach(o=>{const b=document.createElement('button');b.type='button';b.setAttribute('role','radio');
    b.textContent=label?label(o):String(o);b.dataset.v=String(o);box.appendChild(b)});
  const sync=v=>$$('button',box).forEach(b=>{const on=b.dataset.v===String(v);b.setAttribute('aria-checked',String(on));b.setAttribute('aria-pressed',String(on));b.tabIndex=on?0:-1});sync(cur);
  box.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;const v=opts.find(o=>String(o)===b.dataset.v);sync(v);onPick(v)});
  box.addEventListener('keydown',e=>{const bs=$$('button',box),i=bs.findIndex(b=>b.tabIndex===0);let j=i;if(e.key==='ArrowRight'||e.key==='ArrowDown')j=(i+1)%bs.length;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')j=(i-1+bs.length)%bs.length;else return;e.preventDefault();bs[j].focus();bs[j].click()});
  return sync}
function dl(box,rows){box.innerHTML=rows.map(([k,v,c])=>`<dt>${k}</dt><dd${c?' class="acc"':''}>${v}</dd>`).join('')}
function slider(id,lab,min,max,step,val,ndec){return `<div class="rng"><label class="lbl" for="${id}">${lab} <output id="${id}o"></output></label><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></div>`}

