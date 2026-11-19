// C4: paper data embedded in the page (Table 34, Table 2, Table 35). Checks the region statements of Sec. 6.5 / App. I.
const A=[0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9];
const T34={gap:[.152,.146,.189,.194,.216,.320,.333,.355,.367,.481,.522,.572],
 hit5:[.811,.779,.811,.807,.837,.852,.883,.863,.869,.913,.902,.874],
 fp5:[.716,.740,.708,.710,.699,.659,.650,.663,.656,.617,.630,.635],
 p1:[.094,.091,.115,.110,.119,.201,.151,.183,.166,.207,.159,.140],
 p4:[.257,.258,.249,.240,.234,.292,.275,.324,.286,.278,.165,.163]};
const rng=(arr,lo,hi)=>{const v=arr.filter((_,i)=>A[i]>=lo-1e-9&&A[i]<=hi+1e-9);return [Math.min(...v),Math.max(...v)];};
for(const k of ['p1','p4']) console.log(k,'[0.04,0.2]',rng(T34[k],0.04,0.2),'[0.3,0.7]',rng(T34[k],0.3,0.7),'[0.8,0.9]',rng(T34[k],0.8,0.9));
console.log('argmax p1 at',A[T34.p1.indexOf(Math.max(...T34.p1))],'argmax p4 at',A[T34.p4.indexOf(Math.max(...T34.p4))],'argmax hit5',A[T34.hit5.indexOf(Math.max(...T34.hit5))],'argmin fp5',A[T34.fp5.indexOf(Math.min(...T34.fp5))]);
console.log('gain T1->T4:',A.map((a,i)=>a+':'+(T34.p4[i]-T34.p1[i]).toFixed(3)).join(' '));
let mono=true;for(let i=2;i<A.length;i++) if(T34.gap[i]<T34.gap[i-1]) mono=false; console.log('gap monotone from 0.06 up:',mono);
const DEPTH={ // Pass@1 at T=1..4, n_replace=1, macro over 12 cells
 'CDLM (2000-step horizon, Table 2)':[.197,.213,.210,.205],'CDLM+ctw (Table 2)':[.204,.192,.184,.182],'CDLM-0.5B main (Table 2)':[.188,.279,.281,.286],
 'Base (Table 35)':[.145,.240,.255,.263],'MDLM-0.5B (Table 35)':[.138,.236,.246,.245],'CDLM-0.5B (Table 35)':[.188,.279,.281,.285],
 'MDLM-OCI (Table 35)':[.140,.223,.235,.239],'CDLM-OCI (Table 35)':[.220,.302,.303,.307]};
for(const [k,v] of Object.entries(DEPTH)) console.log(k.padEnd(36),v.join(' '),' T4-T1',(v[3]-v[0]).toFixed(3));
