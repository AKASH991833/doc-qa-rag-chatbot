import puppeteer from 'puppeteer-core';
import fs from 'fs';
const out='/tmp/shots59';fs.mkdirSync(out,{recursive:true});
const browser=await puppeteer.launch({executablePath:'/usr/bin/google-chrome',headless:'new',args:['--no-sandbox','--disable-dev-shm-usage','--hide-scrollbars']});
let fails=[];
const check=(name,ok,extra='')=>{console.log((ok?'PASS':'FAIL')+' '+name+(extra?' — '+extra:''));if(!ok)fails.push(name);};
for(const v of [{name:'desktop',w:1440,h:900},{name:'mobile',w:390,h:844}]){
 for(const theme of ['light','dark']){
  const page=await browser.newPage();
  await page.setViewport({width:v.w,height:v.h,deviceScaleFactor:1});
  const errs=[];page.on('pageerror',e=>errs.push(String(e)));page.on('console',m=>{if(m.type()==='error')errs.push(m.text());});
  await page.evaluateOnNewDocument(t=>{localStorage.clear();localStorage.setItem('theme',t);},theme);
  await page.goto('http://localhost:8059',{waitUntil:'networkidle2'});
  // chunking unit tests
  const units=await page.evaluate(()=>{
    const t=window._test;
    const words=Array.from({length:300},(_,i)=>'w'+i).join(' ');
    const chunks=t.chunkText(words,120,30);
    const orth=t.cosine([1,0,0],[0,1,0]);
    const same=t.cosine([1,2,3],[1,2,3]);
    const chunks2=t.chunkText('one two three');
    return {n:chunks.length,first:chunks[0].split(' ').length,overlap:chunks[0].split(' ')[119]===chunks[1].split(' ')[29],orth,same,short:chunks2.length};
  });
  check(`${v.name}/${theme} chunker splits 300 words`,units.n===3,units.n+' chunks');
  check(`${v.name}/${theme} chunk size 120`,units.first===120,'');
  check(`${v.name}/${theme} 30-word overlap`,units.overlap,'');
  check(`${v.name}/${theme} short text skipped`,units.short===0,'');
  check(`${v.name}/${theme} cosine math`,Math.abs(units.orth)<1e-9&&Math.abs(units.same-1)<1e-9,'');
  // sample docs load
  await page.click('#btnSample');
  await page.waitForFunction(()=>window._test.DOCS.length===3,{timeout:5000});
  const chunkCount=await page.evaluate(()=>window._test.CHUNKS.length);
  check(`${v.name}/${theme} 3 sample docs indexed`,chunkCount>=4,chunkCount+' chunks');
  // fake models for a deterministic end-to-end ask
  await page.evaluate(()=>window._test.enableFake());
  await page.type('#qInput','What is the refund window for annual plans?');
  await page.evaluate(()=>document.getElementById('btnAsk').click());
  await page.waitForFunction(()=>[...document.querySelectorAll('.msg.bot')].some(m=>m.textContent.includes('30 days')),{timeout:8000});
  const botTxt=await page.$eval('#chatLog',e=>e.textContent);
  check(`${v.name}/${theme} answer extracted`,botTxt.includes('30 days'),'');
  check(`${v.name}/${theme} source cited`,botTxt.includes('Refund policy'),'');
  check(`${v.name}/${theme} confidence shown`,/Confidence \d+%/.test(botTxt),'');
  // second question about leave
  await page.type('#qInput','How many leave days per year?');
  await page.evaluate(()=>document.getElementById('btnAsk').click());
  await page.waitForFunction(()=>[...document.querySelectorAll('.msg.bot')].some(m=>m.textContent.includes('24 paid leave days')),{timeout:8000});
  check(`${v.name}/${theme} second answer routed to leave doc`,true,'');
  // remove a doc
  const before=await page.evaluate(()=>window._test.CHUNKS.length);
  await page.evaluate(()=>document.querySelector('[data-del]').click());
  const after=await page.evaluate(()=>window._test.CHUNKS.length);
  check(`${v.name}/${theme} doc removal drops its chunks`,after<before,`${before} -> ${after}`);
  await page.evaluate(()=>document.getElementById('chat').scrollIntoView({block:'center'}));
  await new Promise(s=>setTimeout(s,300));
  await page.screenshot({path:`${out}/chat-${v.name}-${theme}.png`});
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.screenshot({path:`${out}/top-${v.name}-${theme}.png`});
  check(`${v.name}/${theme} no console errors`,errs.length===0,errs.slice(0,3).join('|'));
  await page.close();
 }
}
await browser.close();
if(fails.length){console.log('\nFAILURES: '+fails.join(', '));process.exit(1);}
console.log('\nALL PASS');
