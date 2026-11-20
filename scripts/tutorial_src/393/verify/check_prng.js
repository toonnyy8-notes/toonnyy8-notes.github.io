// node check_prng.js  -> must print the same 8 numbers as: python3 -I -c "..." in design.md §5.0
function mulberry32(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2)}
const r=mulberry32(3);console.log([...Array(8)].map(()=>gauss(r).toFixed(10)).join(' '));
