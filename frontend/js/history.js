function historyPage(){
 const list=$('#historyList'), search=$('#historySearch'), filter=$('#historyFilter');
 const fmt=d=>{const x=new Date(d);return isNaN(x)?'Unknown date':x.toLocaleString('en-IN',{dateStyle:'medium',timeStyle:'short'});};
 function draw(){
  const q=(search.value||'').toLowerCase(), f=filter.value;
  const data=loadVerificationHistory().filter(x=>{
   const text=[x.risk?.level,...(x.documents||[]).map(d=>d.filename),...(x.reasons||[])].join(' ').toLowerCase();
   return (!q||text.includes(q))&&(!f||x.risk?.level===f);
  });
  list.innerHTML=data.length?data.map(x=>{
   const c=x.risk?.level==='HIGH'?'bad':x.risk?.level==='MEDIUM'?'warn':'ok';
   const docs=(x.documents||[]).map(d=>esc(d.filename)).join(', ');
   const stats=x.statistics||{};
   const mismatches=stats.mismatches??'N/A';
   return `<div class="history-item"><div><b>${fmt(x.analyzedAt)}</b><div class="mut" style="margin-top:5px">${docs||'No document names'} </div><div class="mut">${(x.documents||[]).length} document(s) · ${mismatches} mismatch(es)</div></div><div class="history-actions">${badge(c,esc(x.risk?.level||'UNKNOWN')+' RISK')}<b>${val(x.risk?.score)}/100</b><button class="btn sm" onclick="viewHistory('${x.id}')">View</button><button class="btn ghost sm" onclick="removeHistory('${x.id}')">Delete</button></div></div>`;
  }).join(''):'<div class="empty">No verification history found.</div>';
 }
 window.viewHistory=id=>{const x=loadVerificationHistory().find(v=>v.id===id);if(!x)return toast('History entry not found.','bad');sessionStorage.setItem(STORAGE_KEY,JSON.stringify({analysisResult:x.result,analyzedAt:x.analyzedAt}));location.href='dashboard.html';};
 window.removeHistory=id=>{if(!confirm('Delete this verification history entry?'))return;deleteVerificationHistory(id);draw();toast('History entry deleted');};
 $('#clearHistory').onclick=()=>{if(!loadVerificationHistory().length)return toast('History is already empty');if(confirm('Clear all verification history?')){clearVerificationHistory();draw();toast('History cleared');}};
 search.oninput=filter.onchange=draw; draw();
}
if(document.body.dataset.page==='history')historyPage();
