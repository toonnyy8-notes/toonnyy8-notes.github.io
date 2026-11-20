// Exact PRNG code the page must use (copied verbatim into design.md 4.0).
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const r=mulberry32(1000);console.log([r(),r(),r()].map(x=>x.toFixed(12)).join(' '));
