function settingsPage(){
 const s=loadVerifySettings(); const dark=$('#darkMode'), max=$('#maxDocuments'), url=$('#apiUrl'), status=$('#apiStatus');
 dark.checked=!!s.darkMode; max.value=String(s.maxDocuments||5); url.value=s.apiBaseUrl||API_BASE_URL;
 function apply(){document.body.classList.toggle('settings-dark',dark.checked); localStorage.setItem('vai_dark_mode',dark.checked?'1':'0');}
 apply(); dark.onchange=apply;
 async function check(){status.textContent='Checking backend...';status.className='setting-status';try{const r=await fetch((url.value||API_BASE_URL).replace(/\/$/,'')+'/api/health');if(!r.ok)throw new Error();status.textContent='● Backend connected';status.className='setting-status connected';}catch(e){status.textContent='● Backend unavailable';status.className='setting-status disconnected';}}
 $('#checkApi').onclick=check;
 $('#saveSettings').onclick=()=>{saveVerifySettings({darkMode:dark.checked,maxDocuments:Number(max.value),apiBaseUrl:(url.value||API_BASE_URL).replace(/\/$/,'')});$('#savedMsg').textContent='Saved';setTimeout(()=>$('#savedMsg').textContent='',1800);toast('Settings saved');check();};
 $('#clearSettingsHistory').onclick=()=>{if(confirm('Clear all verification history?')){clearVerificationHistory();toast('Verification history cleared');}};
 check();
}
if(document.body.dataset.page==='settings')settingsPage();
