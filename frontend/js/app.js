/* Shared helpers + the "view" builder that turns the REAL backend response
   (analysisResult.documents / .comparison / .risk) into what the pages draw.
   The risk score and level are NEVER calculated here - they come from risk.risk_score / risk.risk_level. */
const $=(s,r=document)=>r.querySelector(s);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const NA='N/A';
// Safe display of any backend value: null / undefined / empty -> "N/A", everything else HTML-escaped.
const val=v=>(v===null||v===undefined||String(v).trim()==='')?NA:esc(v);
const pct=v=>Number.isFinite(v)?`${Math.round(v*10)/10}%`:NA;
const badge=(c,t)=>`<span class="b ${c}">${t}</span>`;
function toast(msg,type=''){let w=$('#toasts');if(!w){w=document.createElement('div');w.id='toasts';document.body.append(w)}
 const t=document.createElement('div');t.className='toast '+type;t.textContent=msg;w.append(t);setTimeout(()=>t.remove(),3000)}

/* ---- Mapping from backend values to the existing badge colours (ok / warn / bad) ---- */
const LEVEL_UI={LOW:{c:'ok',label:'LOW RISK'},MEDIUM:{c:'warn',label:'MEDIUM RISK'},HIGH:{c:'bad',label:'HIGH RISK'}};
const STATUS_UI={MATCH:{c:'ok',text:'✓ Match'},PARTIAL_MATCH:{c:'warn',text:'⚠ Partial Match'},MISMATCH:{c:'bad',text:'✕ Mismatch'},NOT_AVAILABLE:{c:'na',text:'– Not Available'}};
const statusUi=s=>STATUS_UI[s]||{c:'na',text:val(s)};
const SEVERITY_UI={HIGH:'bad',MEDIUM:'warn',LOW:'warn'};
// Per-document status is derived from the comparison pairs the document appears in.
const LBL={ok:'✓ Consistent',warn:'⚠ Review',bad:'✕ Inconsistent',na:'– Not compared'};

// Extracted-document fields we know how to label (only non-empty ones are shown).
const DOC_FIELD_LABELS={name:'Name',father_name:"Father's Name",dob:'DOB',gender:'Gender',address:'Address',pincode:'Pincode',state:'State',district:'District',phone:'Phone',email:'Email',
 aadhaar_number:'Aadhaar Number',pan_number:'PAN Number',voter_id:'Voter ID',passport_number:'Passport Number',driving_licence_number:'Driving Licence Number',
 income:'Income',certificate_number:'Certificate Number',application_number:'Application Number',issue_date:'Issue Date',board:'Board',institute:'Institute',
 exam_name:'Exam Name',exam_year:'Exam Year',roll_number:'Roll Number',enrollment_number:'Enrollment Number',percentage:'Percentage',cgpa:'CGPA',result:'Result'};
const ALWAYS_SHOW=['name','dob']; // shown as N/A when missing; other fields are hidden when missing
const fieldLabel=f=>DOC_FIELD_LABELS[f]||String(f||'Field').replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
const hasValue=v=>v!==null&&v!==undefined&&String(v).trim()!=='';

function documentInfo(raw){const o={};
 Object.keys(DOC_FIELD_LABELS).forEach(k=>{if(hasValue(raw[k])||ALWAYS_SHOW.includes(k))o[DOC_FIELD_LABELS[k]]=hasValue(raw[k])?raw[k]:null});
 return o}

/* Build everything the pages need from the stored backend response. Returns null if nothing is stored. */
function buildAnalysisView(stored){
 const result=stored.analysisResult,risk=result.risk||{},comparison=result.comparison||{};
 const pairs=Array.isArray(comparison.pair_comparisons)?comparison.pair_comparisons:[];
 const lvl=LEVEL_UI[risk.risk_level]||{c:'warn',label:`${esc(risk.risk_level||'UNKNOWN')} RISK`};
 const docs=(Array.isArray(result.documents)?result.documents:[]).map((raw,i)=>{
  const filename=raw.filename||`Document ${i+1}`;
  const mine=raw.filename?pairs.filter(p=>p.filename_1===raw.filename||p.filename_2===raw.filename):[];
  let st='na';if(mine.length)st=mine.some(p=>p.mismatch_count>0)?'bad':mine.some(p=>p.partial_match_count>0)?'warn':'ok';
  const cmp=[];
  mine.forEach(p=>{const first=p.filename_1===raw.filename;const other=first?(p.filename_2||p.document_2):(p.filename_1||p.document_1);
   (p.field_comparisons||[]).forEach(fc=>{const s=statusUi(fc.status);cmp.push({label:`${fieldLabel(fc.field)} vs ${other||'other document'}`,text:`${s.text} · ${pct(fc.similarity)}`,c:s.c})})});
  return {id:i+1,raw,filename,type:raw.document_type||'Unknown type',st,info:documentInfo(raw),cmp}});
 const firstWith=k=>{const d=docs.find(x=>hasValue(x.raw[k]));return d?{value:d.raw[k],type:d.type}:null};
 const when=new Date(stored.analyzedAt);
 return {
  docs,pairs,comparison,risk,
  date:isNaN(when)?NA:when.toLocaleDateString('en-GB',{day:'2-digit',month:'long',year:'numeric'}),
  score:risk.risk_score,level:lvl.c,label:lvl.label,
  stats:risk.statistics||{},
  indicators:Array.isArray(risk.risk_indicators)?risk.risk_indicators:[],
  reasons:Array.isArray(risk.reasons)?risk.reasons:[],
  rec:risk.recommendation,disclaimer:risk.disclaimer,
  warnings:Array.isArray(risk.warnings)?risk.warnings:[],
  applicantName:firstWith('name'),applicantDob:firstWith('dob')};
}
const getAnalysisView=()=>{const s=loadAnalysisResult();return s?buildAnalysisView(s):null};
const getDocumentDetails=id=>{const v=getAnalysisView();return v?v.docs.find(d=>d.id==id):null};

/* Inject sidebar: <aside id="side" data-on="Dashboard"> */
(function(){const s=$('#side');if(!s)return;
 const L=[['Dashboard','dashboard.html'],['Documents','upload.html'],['Verification','upload.html'],['Reports','report.html'],['History','history.html'],['Settings','settings.html']];
 s.className='side';s.innerHTML=`<a class="logo" href="index.html">Verify<span>AI</span></a>`+L.map(l=>`<a href="${l[1]}" class="${l[0]==s.dataset.on?'on':''}">${l[0]}</a>`).join('')+`<div class="prof">👤 Admin User<div class="mut">admin@verifyai.demo</div></div>`;
 const h=$('.ham');if(h)h.onclick=()=>s.classList.toggle('open')})();
