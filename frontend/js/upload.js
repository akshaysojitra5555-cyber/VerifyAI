/* Upload page: sends the selected files to the real backend (see api.js) and opens the dashboard. */
const MAX_FILES=5,MAX_BYTES=10*1024*1024,EXT=/\.(pdf|jpe?g|png)$/i;
let files=[]; // real File objects
const mb=b=>(b/1048576).toFixed(2)+' MB';
function render(){$('#go').disabled=!files.length;
 $('#list').innerHTML=files.length?files.map((f,i)=>`<div class="card dl" style="margin-bottom:10px"><div>📄 <b>${esc(f.name)}</b><div class="mut">${mb(f.size)}</div></div>${badge('ok','✓ Ready')}<button class="btn ghost sm" style="color:#dc2626" onclick="rm(${i})">Remove</button></div>`).join(''):'<div class="card empty">No documents selected yet. Drop files above to begin.</div>'}
function rm(i){files.splice(i,1);render();toast('Document removed')}
function add(list){for(const f of list){
 if(files.length>=MAX_FILES){toast('Maximum 5 documents allowed.','bad');break}
 if(!EXT.test(f.name)){toast('Only PDF, JPG, JPEG and PNG files are supported.','bad');continue}
 if(f.size>=MAX_BYTES){toast('Each file must be smaller than 10 MB.','bad');continue}
 files.push(f)}
 $('#pick').value='';render()}
const dz=$('#drop');
dz.onclick=()=>$('#pick').click();$('#pick').onchange=e=>add(e.target.files);
['dragover','dragenter'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.add('over')}));
['dragleave','drop'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.remove('over')}));
dz.addEventListener('drop',e=>add(e.dataTransfer.files));
const GO_LABEL=$('#go').textContent;let isAnalyzing=false;
const closeOverlay="$('#ov').classList.remove('show')";
// Validates the selection, calls the backend, stores the real result, then opens the dashboard.
async function handleAnalysis(){
 if(isAnalyzing)return;
 if(!files.length)return toast('Please select at least one document.','bad');
 if(files.length>MAX_FILES)return toast('Maximum 5 documents allowed.','bad');
 if(files.some(f=>!EXT.test(f.name)))return toast('Only PDF, JPG, JPEG and PNG files are supported.','bad');
 if(files.some(f=>f.size>=MAX_BYTES))return toast('Each file must be smaller than 10 MB.','bad');
 isAnalyzing=true;$('#go').disabled=true;$('#go').textContent='Analyzing documents...';
 $('#ob').innerHTML=`<h3><span class="spin">⟳</span> Analyzing documents...</h3><p class="mut">Running OCR, extraction, comparison and risk analysis on ${files.length} file(s). This can take a little while.</p>`;
 $('#ov').classList.add('show');
 try{
  const result=await uploadAndAnalyzeDocuments(files); // api.js -> POST /api/compare-documents
  saveAnalysisResult(result);                          // full JSON, replaces any older result
  $('#ob').innerHTML='<h3>✅ Analysis complete</h3><p class="mut">Opening the dashboard...</p>';
  window.location.href='dashboard.html';
 }catch(e){
  console.error('Analysis failed:',e);
  $('#ob').innerHTML=`<h3>⚠ Unable to analyze documents</h3><p class="mut" style="margin-top:8px">${esc(e.message)}</p><button class="btn" style="margin-top:14px" onclick="${closeOverlay}">Close</button>`;
 }finally{isAnalyzing=false;$('#go').textContent=GO_LABEL;render()}}
$('#go').onclick=handleAnalysis;
render();
