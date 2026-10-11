/* Mock data – replace with Flask/FastAPI JSON responses later (see app.js). */
const MOCK={cases:{
low:{id:'case-low',applicant:'Rahul Patel',dob:'12/04/2005',income:'₹1,80,000',date:'01 October 2026',score:28,level:'ok',label:'LOW RISK',confidence:94,
metrics:[['Name Consistency',96,'ok','✓ Consistent'],['DOB Consistency',100,'ok','✓ Consistent'],['Address Consistency',91,'ok','✓ Mostly Consistent'],['Document Validation',94,'ok','✓ Valid']],
indicators:[{c:'warn',sev:'LOW',title:'Minor Name Variation',detail:'"Rahul Patel" vs "Rahul R. Patel"',why:'The difference appears to be a middle initial. The entities are likely to represent the same person.'},{c:'ok',sev:'NONE',title:'No DOB inconsistency detected',detail:'DOB is identical in all documents.',why:''}],
fields:[['Name',['Rahul Patel','Rahul R. Patel','Rahul Patel','Rahul Patel'],'warn'],['DOB',['12/04/2005','12/04/2005','12/04/2005','12/04/2005'],'ok'],['Address',['Ahmedabad','Ahmedabad','Ahmedabad','Ahmedabad'],'ok'],['Income',['—','₹1,80,000','—','₹1,80,000'],'ok']],
summary:'All submitted documents appear broadly consistent. The applicant name contains a minor variation in the income certificate, but the DOB and other identifying information match. No major cross-document inconsistencies were detected.',
rec:'Documents show no major inconsistencies. Manual verification may be performed according to organizational policy.',
docs:[
{id:1,name:'Aadhaar.pdf',type:'ID Proof',size:'2.4 MB',st:'ok',x:{Name:'Rahul Patel',DOB:'12/04/2005',Address:'Ahmedabad','ID Number':'XXXX-XXXX-4821'},cmp:[['Name','Exact match','ok'],['DOB','Exact match','ok']]},
{id:2,name:'Income_Certificate.pdf',type:'Income Certificate',size:'1.1 MB',st:'warn',x:{Name:'Rahul R. Patel',DOB:'12/04/2005','Annual Income':'₹1,80,000','Certificate Number':'INC12345','Issue Date':'15/08/2026'},cmp:[['Name','Minor variation','warn'],['DOB','Exact match','ok'],['Certificate Number','Unique','ok']]},
{id:3,name:'Marksheet.pdf',type:'Academic',size:'1.8 MB',st:'ok',x:{Name:'Rahul Patel',DOB:'12/04/2005',Board:'GSEB'},cmp:[['Name','Exact match','ok'],['DOB','Exact match','ok']]},
{id:4,name:'Caste_Certificate.pdf',type:'Caste Certificate',size:'0.9 MB',st:'ok',x:{Name:'Rahul Patel','Certificate Number':'CC88231'},cmp:[['Name','Exact match','ok'],['Certificate Number','Unique','ok']]},
{id:5,name:'Scholarship_Application.pdf',type:'Application Form',size:'0.6 MB',st:'ok',x:{Name:'Rahul Patel',DOB:'12/04/2005','Annual Income':'₹1,80,000','Application Date':'20/08/2026'},cmp:[['Name','Exact match','ok'],['Income','Exact match','ok']]}]},
high:{id:'case-high',applicant:'Rahul Patel',dob:'12/04/2005',income:'₹1,80,000',date:'01 October 2026',score:78,level:'bad',label:'HIGH RISK',confidence:89,
metrics:[['Name Consistency',62,'bad','✕ Mismatch'],['DOB Consistency',100,'ok','✓ Consistent'],['Address Consistency',88,'warn','⚠ Mostly Consistent'],['Document Validation',61,'bad','✕ Review Needed']],
indicators:[{c:'bad',sev:'HIGH',title:'Name Mismatch',detail:'"Rahul Patel" vs "Rohan Patel"',why:'The first name differs in the income certificate and is unlikely to be an OCR or formatting variation. Manual verification recommended.'},{c:'warn',sev:'MEDIUM',title:'Date Inconsistency',detail:'Certificate issued 15/08/2026, application dated 10/08/2026',why:'The supporting certificate appears to be issued after the application date.'}],
fields:[['Name',['Rahul Patel','Rohan Patel','Rahul Patel','Rahul Patel'],'bad'],['DOB',['12/04/2005','12/04/2005','12/04/2005','12/04/2005'],'ok'],['Address',['Ahmedabad','Ahmedabad','Ahmedabad','Ahmedabad'],'ok'],['Income',['—','₹1,80,000','—','₹1,80,000'],'ok']],
summary:'Potential fraud indicators were found. The name in the income certificate differs from other documents and the certificate issue date is later than the application date. These signals do not prove forgery but raise the risk score.',
rec:'Manual verification is recommended before any decision. Results are risk indicators only, not proof of fraud.',
docs:[
{id:1,name:'Aadhaar.pdf',type:'ID Proof',size:'2.4 MB',st:'ok',x:{Name:'Rahul Patel',DOB:'12/04/2005'},cmp:[['Name','Exact match','ok']]},
{id:2,name:'Income_Certificate.pdf',type:'Income Certificate',size:'1.1 MB',st:'bad',x:{Name:'Rohan Patel',DOB:'12/04/2005','Annual Income':'₹1,80,000','Certificate Number':'INC12345','Issue Date':'15/08/2026'},cmp:[['Name','Mismatch','bad'],['Issue Date','After application date','warn'],['DOB','Exact match','ok']]},
{id:3,name:'Marksheet.pdf',type:'Academic',size:'1.8 MB',st:'ok',x:{Name:'Rahul Patel',DOB:'12/04/2005'},cmp:[['Name','Exact match','ok']]},
{id:4,name:'Caste_Certificate.pdf',type:'Caste Certificate',size:'0.9 MB',st:'ok',x:{Name:'Rahul Patel'},cmp:[['Name','Exact match','ok']]},
{id:5,name:'Scholarship_Application.pdf',type:'Application Form',size:'0.6 MB',st:'warn',x:{Name:'Rahul Patel','Application Date':'10/08/2026'},cmp:[['Application Date','Before certificate date','warn']]}]}
}};
