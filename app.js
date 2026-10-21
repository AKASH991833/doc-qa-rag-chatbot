/* Doc QA - retrieval-augmented QA fully in the browser (transformers.js) */
import {pipeline, env} from './vendor/transformers.min.js';
env.allowLocalModels=false;
env.backends.onnx.wasm.wasmPaths='./vendor/';

/* ---------- chunking + vectors ---------- */
function chunkText(text,size=120,overlap=30){
  const words=String(text).replace(/\s+/g,' ').trim().split(' ').filter(Boolean);
  const out=[];
  for(let i=0;i<words.length;i+=size-overlap){
    const c=words.slice(i,i+size).join(' ');
    if(c.split(' ').length>=8)out.push(c);
    if(i+size>=words.length)break;
  }
  return out;
}
function cosine(a,b){let d=0,na=0,nb=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}return na&&nb?d/(Math.sqrt(na)*Math.sqrt(nb)):0;}
function topK(qvec,chunks,k=3){return chunks.map(c=>({c,s:cosine(qvec,c.vec)})).sort((a,b)=>b.s-a.s).slice(0,k);}

/* ---------- state ---------- */
let DOCS=[], CHUNKS=[], EMB=null, QAM=null, LOADING=false, READY=false;
const uid=()=>Math.random().toString(36).slice(2,9);

/* ---------- models ---------- */
function setLoad(pct,text){
  const bar=document.getElementById('loadBar'),fill=document.getElementById('loadFill'),t=document.getElementById('loadText');
  if(pct==null){bar.classList.add('hidden');t.classList.add('hidden');return;}
  bar.classList.remove('hidden');t.classList.remove('hidden');
  fill.style.width=pct+'%';t.textContent=text;
}
async function ensureModels(){
  if(READY)return;
  if(LOADING){while(LOADING)await new Promise(s=>setTimeout(s,300));return;}
  LOADING=true;
  const status=document.getElementById('modelStatus');
  const prog=p=>{
    if(p.status==='progress'&&p.total){const pct=Math.round(p.loaded/p.total*100);setLoad(pct,'Downloading '+p.file+' - '+pct+'%');}
    else if(p.status==='done')setLoad(99,'Almost ready...');
    else if(p.status==='ready')setLoad(null);
  };
  try{
    status.textContent='Loading the embedding model (23 MB, one time)...';
    EMB=await pipeline('feature-extraction','Xenova/all-MiniLM-L6-v2',{quantized:true,progress_callback:prog});
    status.textContent='Loading the QA model (65 MB, one time)...';
    QAM=await pipeline('question-answering','Xenova/distilbert-base-cased-distilled-squad',{quantized:true,progress_callback:prog});
    READY=true;
    status.textContent='Models ready - everything now runs offline on your device.';
    setLoad(null);
    // embed any waiting chunks
    for(const c of CHUNKS)if(!c.vec)c.vec=await embedOne(c.text);
    toast('Models ready');
  }catch(e){
    status.textContent='Could not load the models ('+(e.message||e)+'). Check your connection and retry.';
    setLoad(null);
  }
  LOADING=false;
}
async function embedOne(text){
  const out=await EMB(text,{pooling:'mean',normalize:true});
  return Array.from(out.data);
}

/* ---------- documents ---------- */
const SAMPLE=[
 {name:'Aurora Labs - Leave policy',text:`Aurora Labs Leave Policy. Every full-time employee receives 24 paid leave days per calendar year, accrued at 2 days per month. Unused leave may be carried over, up to a maximum of 6 days, into the first quarter of the following year. Sick leave is separate: 12 days per year with a medical note required after 3 consecutive days. Parental leave is 26 weeks for the primary caregiver and 4 weeks for the secondary caregiver, fully paid. Leave requests under 3 days need manager approval; longer requests also need HR sign-off. Public holidays do not count against the leave balance. During the probation period of 3 months, employees may take up to 5 leave days. Leave encashment is possible only at exit, at the basic salary rate, for a maximum of 15 days.`},
 {name:'Aurora Labs - Refund policy',text:`Aurora Labs Refund Policy. Customers may request a full refund within 30 days of purchase for any annual plan, and within 14 days for monthly plans. Refunds are processed to the original payment method within 5 to 7 business days after approval. Add-on services and usage-based charges are non-refundable once consumed. To request a refund, email billing@auroralabs.example with your invoice number. Enterprise contracts follow the refund terms in their signed order form. If an account is terminated for a policy violation, no refund is issued. Duplicate charges are always refunded in full, immediately upon verification. Refund requests after the window closes are reviewed case by case and may receive account credit instead of money.`},
 {name:'CloudSync - Pricing sheet',text:`CloudSync Pricing. The Free tier includes 5 GB of storage and sync across 3 devices. The Plus plan costs 4.99 dollars per month and includes 200 GB, unlimited devices, and version history for 30 days. The Family plan costs 9.99 dollars per month, covers up to 6 members, and shares 2 TB of storage. The Team plan costs 12 dollars per user per month with 1 TB per user, admin controls, and SSO. Annual billing gives a 17 percent discount on all paid plans. Storage overages on Plus and Family cost 1.50 dollars per 50 GB per month. All plans include end-to-end encryption for private folders. Educational institutions receive 50 percent off Team plans with verification.`}
];
function renderDocs(){
  const el=document.getElementById('docList');
  document.getElementById('docCount').textContent=DOCS.length+' document'+(DOCS.length===1?'':'s');
  el.innerHTML=DOCS.length?DOCS.map(d=>`<div class="card card-pad rise"><div class="row row-wrap spread mb16"><b>${esc(d.name)}</b><button class="btn btn-ghost btn-sm" data-del="${d.id}">Remove</button></div><p class="small muted">${d.chunks.length} chunks · ${d.text.split(' ').length} words</p></div>`).join(''):'<div class="card empty"><h3>No documents yet</h3><p>Load the sample workspace or paste your own text below.</p></div>';
  el.querySelectorAll('[data-del]').forEach(b=>b.addEventListener('click',()=>{
    DOCS=DOCS.filter(d=>d.id!==b.dataset.del);CHUNKS=CHUNKS.filter(c=>c.docId!==b.dataset.del);renderDocs();toast('Document removed');
  }));
}
async function addDoc(name,text){
  name=name.trim()||'Untitled document';text=text.trim();
  if(text.split(' ').length<12){toast('That text is too short to index');return;}
  const chunks=chunkText(text).map(t=>({docId:null,text:t,vec:null}));
  const d={id:uid(),name,text,chunks};
  chunks.forEach(c=>c.docId=d.id);
  DOCS.push(d);CHUNKS.push(...chunks);
  renderDocs();
  if(READY){for(const c of chunks)c.vec=await embedOne(c.text);toast('Document embedded and ready');}
  else toast('Document added - it will be embedded when the models load');
}
document.getElementById('btnAddDoc').addEventListener('click',()=>{
  const n=document.getElementById('docName').value,t=document.getElementById('docText').value;
  addDoc(n,t).then(()=>{document.getElementById('docName').value='';document.getElementById('docText').value='';});
});
document.getElementById('fileInput').addEventListener('change',e=>{
  const f=e.target.files[0];if(!f)return;
  const r=new FileReader();r.onload=()=>addDoc(f.name.replace(/\.(txt|md|text|markdown)$/i,''),String(r.result));r.readAsText(f);
  e.target.value='';
});
document.getElementById('btnSample').addEventListener('click',async()=>{
  for(const d of SAMPLE)await addDoc(d.name,d.text);
  toast('Sample workspace loaded - ask a question below');
  document.getElementById('chat').scrollIntoView({behavior:'smooth'});
});

/* ---------- chat ---------- */
function addMsg(role,html){
  document.getElementById('chatEmpty')?.remove();
  const log=document.getElementById('chatLog');
  const div=document.createElement('div');div.className='msg '+(role==='user'?'user':'bot');
  div.innerHTML=html;log.appendChild(div);log.scrollTop=log.scrollHeight;
  return div;
}
async function ask(q){
  if(!CHUNKS.length){toast('Add or load a document first');document.getElementById('docs').scrollIntoView({behavior:'smooth'});return;}
  addMsg('user',`<div class="msg-bubble">${esc(q)}</div>`);
  const thinking=addMsg('bot','<div class="msg-bubble muted">Thinking...</div>');
  await ensureModels();
  if(!READY){thinking.querySelector('.msg-bubble').textContent='The models could not be loaded, so I cannot answer yet.';return;}
  for(const c of CHUNKS)if(!c.vec)c.vec=await embedOne(c.text);
  const qvec=await embedOne(q);
  const best=topK(qvec,CHUNKS,3);
  let answer=null;
  try{answer=await QAM({question:q,context:best[0].c.text});}catch(e){answer=null;}
  const conf=answer?answer.score:0;
  let body;
  if(!answer||conf<0.05){
    body='<div class="msg-bubble">I could not find a confident answer in your documents. Try rephrasing, or add the document that covers this topic.</div>';
  }else{
    body=`<div class="msg-bubble">${esc(answer.answer)}</div>`+
      `<div class="msg-srcs">`+best.map((b,i)=>{
        const doc=DOCS.find(d=>d.id===b.c.docId);
        return `<div class="msg-src"><b>${i===0?'Answer from':'Also see'} ${esc(doc?doc.name:'document')}</b> (match ${(b.s*100).toFixed(0)}%)<br>${esc(b.c.text.slice(0,110))}...</div>`;
      }).join('')+`</div><div class="small muted">Confidence ${(conf*100).toFixed(0)}% - extractive span, verbatim from your text.</div>`;
  }
  thinking.innerHTML=body;
}
document.getElementById('btnAsk').addEventListener('click',()=>{
  const el=document.getElementById('qInput'),q=el.value.trim();
  if(!q)return;el.value='';ask(q);
});
document.getElementById('qInput').addEventListener('keydown',e=>{if(e.key==='Enter')document.getElementById('btnAsk').click();});

renderDocs();
window._test={
  chunkText,cosine,topK,
  get DOCS(){return DOCS;},get CHUNKS(){return CHUNKS;},
  enableFake(){
    // deterministic stand-ins so the suite runs without a 90 MB download
    const h=s=>{const v=new Array(16).fill(0);String(s).toLowerCase().replace(/[^a-z ]/g,'').split(' ').forEach(w=>{for(const ch of w)v[ch.charCodeAt(0)%16]+=1;});const n=Math.hypot(...v)||1;return v.map(x=>x/n);};
    EMB=async t=>({data:h(t)});
    QAM=async ({question,context})=>{
      const qw=question.toLowerCase().replace(/[^a-z ]/g,'').split(' ').filter(w=>w.length>3);
      const sents=context.split(/(?<=[.!?]) /);
      let bestS=sents[0]||context,bs=-1;
      for(const s of sents){const sc=qw.reduce((a,w)=>a+(s.toLowerCase().includes(w)?1:0),0);if(sc>bs){bs=sc;bestS=s;}}
      return {answer:bestS.trim(),score:bs>0?0.8:0.01};
    };
    READY=true;
    document.getElementById('modelStatus').textContent='Test mode: deterministic stand-in models active.';
  }
};
