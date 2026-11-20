const {runToy}=require('./toy2.js');
for(const eta of [0.1,0.2,0.3])for(const nu of [0.2,0.4])for(const lrT of [0.03,0.05])for(const tau of [0.01]){
  const out=[1,2,3,4].map(seed=>{const R=runToy({eta,nu,lrT,seed,tau});return R.promos.length+'['+R.promos.slice(0,6).join(',')+'] a'+R.final.acc.toFixed(3)+' mu'+R.final.mu.toFixed(1)});
  console.log(`eta ${eta} nu ${nu} lrT ${lrT}:`,out.join('  '));
}
