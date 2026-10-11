/* Drives dashboard.html, document.html and report.html (body[data-page]).
   All data comes from the REAL backend response saved by upload.js (see api.js).
   Nothing in this file is mock data. */
const RC={ok:'#16a34a',warn:'#d97706',bad:'#dc2626'};
// key/value rows; every value goes through val() so null/undefined show as "N/A"
const kv=o=>Object.entries(o).map(([k,v])=>`<div class="kv"><span>${esc(k)}</span><b>${val(v)}</b></div>`).join('');
const noAnalysis=()=>`<div class="card empty">⚠ No analysis results found.<p style="margin:8px 0 16px">Upload documents and run the verification first.</p><a class="btn" href="upload.html">Upload Documents</a></div>`;

/* ---- HTML builders (shared by dashboard and report) ---- */

// One table per compared pair: Field | Document 1 value | Document 2 value | Similarity | Status
function pairTable(p,n){
 const rows=(Array.isArray(p.field_comparisons)?p.field_comparisons:[]).map(fc=>{const s=statusUi(fc.status),d=fc.status==='MISMATCH'?'diff':'';
  return `<tr><td><b>${esc(fieldLabel(fc.field))}</b></td><td class="${d}">${val(fc.value_1)}</td><td class="${d}">${val(fc.value_2)}</td><td>${pct(fc.similarity)}</td><td>${badge(s.c,s.text)}</td></tr>`}).join('');
 const head=(label,doc,file)=>`${label}: ${val(doc)}<div class="mut" style="text-transform:none">${val(file)}</div>`;
 return `<h4 style="margin:18px 0 4px">Pair ${n}: ${val(p.document_1)} vs ${val(p.document_2)}</h4>
 <p class="mut" style="margin-bottom:8px">Consistency ${pct(p.consistency_score)} · ${val(p.matched_field_count)} matched · ${val(p.partial_match_count)} partial · ${val(p.mismatch_count)} mismatched · ${val(p.not_available_count)} not available</p>
 <div class="tw"><table><tr><th>Field</th><th>${head('Document 1',p.document_1,p.filename_1)}</th><th>${head('Document 2',p.document_2,p.filename_2)}</th><th>Similarity</th><th>Status</th></tr>${rows||'<tr><td colspan="5" class="mut">No fields were compared for this pair.</td></tr>'}</table></div>`}

const comparisonSection=d=>d.pairs.length
 ?`<p class="mut">${val(d.comparison.documents_compared)} documents compared · Overall consistency: ${pct(d.comparison.overall_consistency_score)}</p>`+d.pairs.map((p,i)=>pairTable(p,i+1)).join('')
 :'<p class="mut">No document pairs were compared. Upload at least two documents to see a comparison.</p>';

// Risk indicators: "Name — MISMATCH — 30 points"
const indicatorsSection=d=>d.indicators.length?d.indicators.map(i=>{const c=SEVERITY_UI[i.severity]||'warn';
 const docs=i.document_1||i.document_2?`<p>${val(i.document_1)} vs ${val(i.document_2)}</p>`:'';
 return `<div class="ind ${c}"><b>${c=='ok'?'🟢':c=='warn'?'🟡':'🔴'} ${esc(fieldLabel(i.field))} — ${esc(String(i.status||NA).replace(/_/g,' '))} — ${val(i.points)} points</b> ${badge(c,val(i.severity))}<p>${val(i.message)}</p>${docs}</div>`}).join('')
 :'<div class="ind ok"><b>🟢 No risk indicators</b><p>The rule-based engine found no mismatches or partial matches.</p></div>';

// Reasons + recommendation + disclaimer, all straight from risk.*
const summarySection=d=>`${d.reasons.length?`<h4 style="margin-bottom:6px">Why this score</h4><ul style="margin:0 0 12px 20px;line-height:1.7">${d.reasons.map(r=>`<li>${val(r)}</li>`).join('')}</ul>`:''}
 <p style="line-height:1.7"><b>Recommendation:</b> ${val(d.rec)}</p>
 ${d.warnings.length?`<p class="mut" style="margin-top:8px">Notes: ${d.warnings.map(val).join(' ')}</p>`:''}
 ${d.disclaimer?`<p class="mut" style="margin-top:10px">${val(d.disclaimer)}</p>`:''}`;

// Statistics cards from risk.statistics
function statsCards(d){const s=d.stats,n=k=>Number.isFinite(s[k])?s[k]:null,total=['matches','partial_matches','mismatches','not_available'].reduce((a,k)=>a+(n(k)||0),0);
 const none='✓ None';
 return [['Matches','matches','ok','✓ Fields matched'],['Partial Matches','partial_matches','warn','⚠ Needs review'],['Mismatches','mismatches','bad','✕ Inconsistent'],['Not Available','not_available','na','– Not compared']]
 .map(([label,k,c,text])=>{const v=n(k);const on=v>0||k==='matches';
  return `<div class="card" style="margin:0"><div class="mut">${label}</div><div class="big">${val(v)}</div><div class="bar"><i style="width:${total&&v?Math.round(v/total*100):0}%"></i></div>${on?badge(c,text):badge('ok',none)}</div>`}).join('')}

/* ---- Dashboard ---- */
function renderAnalysisResult(d){
 const a=d.applicantName;
 $('#head').innerHTML=`<h2>Document Verification</h2><div class="sub">${d.docs.length} Documents · ${esc(d.date)}${a?` · Applicant (from ${esc(a.type)}): <b>${val(a.value)}</b>`:''}</div>`;
 $('#risk').innerHTML=`<div class="mut" style="text-align:center;margin-bottom:10px">VERIFICATION RISK SCORE</div><div class="ring" id="ring" style="--c:${RC[d.level]}"><div><span><span id="n">0</span><small>/ 100</small></span></div></div><p style="text-align:center;margin:14px 0 6px">${badge(d.level,d.label)}</p><p class="mut" style="text-align:center">Risk indicators: ${d.indicators.length}</p>`;
 // animate the ring up to the score that came from the risk engine
 const target=Math.max(0,Math.min(100,d.score));let n=0;
 if(target>0){const t=setInterval(()=>{n++;$('#ring').style.setProperty('--p',n);$('#n').textContent=n;if(n>=target)clearInterval(t)},18)}
 $('#metrics').innerHTML=statsCards(d);
 $('#cmp').innerHTML=comparisonSection(d);$('#inds').innerHTML=indicatorsSection(d);$('#sum').innerHTML=summarySection(d);
 const draw=()=>{const q=$('#q').value.toLowerCase(),s=$('#f').value;
  const L=d.docs.filter(x=>`${x.filename} ${x.type}`.toLowerCase().includes(q)&&(!s||x.st===s));
  $('#docs').innerHTML=L.length?L.map(x=>`<div class="dl"><div><b>${esc(x.filename)}</b><div class="mut">${esc(x.type)}</div></div>${badge('ok','OCR Complete')}${badge(x.st,LBL[x.st])}<button class="btn sm" onclick="quick(${x.id})">View</button></div>`).join(''):'<div class="empty">🔍 No documents match your search.</div>'};
 $('#q').oninput=$('#f').onchange=draw;draw();
 window.quick=id=>{const x=d.docs.find(v=>v.id==id);if(!x)return;
  $('#mb').innerHTML=`<h3>${esc(x.filename)}</h3><p class="mut">${esc(x.type)}</p>${kv(x.info)}<div style="margin-top:16px;display:flex;gap:8px"><a class="btn sm" href="document.html?id=${id}">Open full details</a><button class="btn ghost sm" style="color:#0f172a" onclick="$('#modal').classList.remove('show')">Close</button></div>`;$('#modal').classList.add('show')}}
function dashboard(){const d=getAnalysisView();
 if(!d){document.querySelectorAll('.main > .row, .main > .card, .main > a.btn').forEach(e=>e.style.display='none');$('#head').innerHTML=noAnalysis();return}
 renderAnalysisResult(d)}

/* ---- Document details page ---- */
function docPage(){const d=getAnalysisView();
 if(!d)return $('#doc').innerHTML=noAnalysis();
 const x=d.docs.find(v=>v.id==(new URLSearchParams(location.search).get('id')||1));
 if(!x)return $('#doc').innerHTML='<div class="card empty">⚠ Document not found.</div>';
 const ic={ok:'✓',warn:'⚠',bad:'✕',na:'–'};
 $('#doc').innerHTML=`<h2>${esc(x.filename)}</h2><div class="sub">${esc(x.type)}</div><div class="row r2" style="margin-top:20px"><div class="card"><h3>Document Preview</h3><div class="preview"><b style="color:#475569">${esc(x.type.toUpperCase())}</b><i></i><i style="width:70%"></i><i></i><i style="width:50%"></i><i style="width:80%"></i></div></div><div><div class="card"><h3>Document Information</h3><div class="kv"><span>Document Type</span><b>${esc(x.type)}</b></div><div class="kv"><span>OCR Status</span><b>Complete</b></div><h3 style="margin-top:16px">Extracted Information</h3>${kv(x.info)}</div><div class="card"><h3>Cross-Document Comparison</h3>${x.cmp.length?x.cmp.map(c=>`<div class="dl"><span>${esc(c.label)}</span><span>${c.text} ${badge(c.c,ic[c.c])}</span></div>`).join(''):'<p class="mut">This document was not compared with another document.</p>'}</div><a class="btn" href="dashboard.html">Back to Dashboard</a></div></div>`}

/* ---- Report page ---- */
function report(){const d=getAnalysisView();
 if(!d)return $('#rep').innerHTML=noAnalysis();
 const nm=d.applicantName,dob=d.applicantDob;
 $('#rep').innerHTML=`<h2 style="color:#4f46e5">VERIFYAI</h2><h3>DOCUMENT VERIFICATION REPORT</h3><p class="mut">Generated ${esc(d.date)}</p>
 <h4>1. Applicant Information</h4>${kv({Applicant:nm?nm.value:null,DOB:dob?dob.value:null,'Total Documents':d.docs.length,'Risk Score':d.score+'/100',Status:d.label})}<p class="mut">Name and DOB are taken from the first document that contains them.</p>
 <h4>2. Documents Analyzed</h4>${d.docs.map(x=>`<div class="dl"><span>${esc(x.filename)} <span class="mut">(${esc(x.type)})</span></span>${badge(x.st,LBL[x.st])}</div>`).join('')}
 <h4>3. Extracted Information</h4>${d.docs.map(x=>`<p><b>${esc(x.filename)}</b></p>${kv(x.info)}`).join('')}
 <h4>4. Cross-Document Comparison</h4>${comparisonSection(d)}<h4>5. Risk Indicators</h4>${indicatorsSection(d)}
 <h4>6. Verification Summary</h4>${summarySection(d)}<h4>7. Final Recommendation</h4><p>${val(d.rec)}</p>`;
 $('#dl').onclick=()=>{const h=`<!doctype html><meta charset=utf-8><title>VerifyAI Report</title><style>body{font-family:sans-serif;max-width:800px;margin:30px auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px;text-align:left}.kv{display:flex;gap:20px}.kv span{width:140px;color:#666}.b{font-weight:700}</style>${$('#rep').innerHTML}`;
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([h],{type:'text/html'}));a.download='VerifyAI_Report.html';a.click();toast('Report downloaded')};
 $('#pr').onclick=()=>window.print()}
({dashboard,document:docPage,report})[document.body.dataset.page]();
