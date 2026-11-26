#!/usr/bin/env python3
"""Build six blinded low-SNR listening trials; contains no fabricated ratings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import zipfile
from run_evaluation import csv_output

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['noise_cleanliness','speech_naturalness','intelligibility','overall_quality']
COLUMNS=['listener_id','trial_id','sample_code',*FIELDS,'notes']
METHODS=['identity','fir','spectral_subtraction','wiener']

PAGE='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Speech denoising — blind listening</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f9;color:#142337;font:16px/1.55 system-ui,sans-serif}
main{max-width:1100px;margin:auto;padding:28px}h1,h2,h3{line-height:1.2}section{background:white;border:1px solid #dae1eb;border-radius:12px;padding:24px;margin:22px 0}
.hint{color:#526479}.sticky{position:sticky;top:0;background:#f4f6f9ee;padding:14px 0;z-index:2;backdrop-filter:blur(8px)}
audio{width:100%;max-width:450px}input,select,textarea,button{font:inherit;padding:8px;border:1px solid #aebdce;border-radius:6px}button{background:#194f8a;color:white;cursor:pointer}
.sample{border-top:1px solid #dae1eb;padding:18px 0}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0}label{display:block}select{width:100%}textarea{width:100%;min-height:50px}
@media(max-width:700px){main{padding:14px}.grid{grid-template-columns:1fr 1fr}section{padding:16px}}
</style></head><body><main><h1>Speech denoising: blind listening</h1>
<p>Six low-SNR trials · four anonymous versions per trial · about 10–15 minutes.</p>
<section><h2>How to rate</h2><ol><li>Use headphones and a comfortable volume. Keep the playback volume unchanged between versions in each trial.</li>
<li>You may replay each version. The clean reference is optional and is not a method to rate.</li>
<li>Rate each sample from 1 to 5. Use notes for muffling, missing consonants, isolated tones, or other artifacts.</li>
<li>Enter a listener code such as P01 before rating. Each code has a separate saved session; changing it starts or resumes that listener. Export the CSV when all 24 sample rows are complete. Partial exports keep empty cells; they are not completed ratings.</li></ol>
<p><strong>Noise cleanliness:</strong> 1=much background noise; 5=little background noise.<br>
<strong>Speech naturalness:</strong> 1=strongly distorted; 5=natural.<br>
<strong>Intelligibility:</strong> 1=hard to understand; 5=easy.<br>
<strong>Overall quality:</strong> 1=poor; 5=excellent.</p>
<p class="hint">All versions of an utterance use the same excerpt and export gain. Sample labels are randomized for each trial. These six trials cover bus, office and café at nominal 2.5 dB; they are a pilot, not a population-wide listening study.</p></section>
<div class="sticky"><label>Listener code <input id="listener" maxlength="32" placeholder="P01" pattern="[A-Za-z0-9_-]+"></label>
<p id="progress" aria-live="polite"></p><button id="download">Export ratings CSV</button></div><div id="trials"></div>
<footer><p>Audio: Cassia Valentini-Botinhao, University of Edinburgh (2017), doi:10.7488/ds/2117, CC BY 4.0.
Speech: CSTR VCTK; background contributions: DEMAND (Thiemann, Ito, Vincent), doi:10.5281/zenodo.1227121, CC BY-SA 3.0.
Modified for this comparison: 48→16 kHz resampling, common cropping/gain and frozen denoising filters.</p></footer></main>
<script>
const trials=__TRIALS__, fields=__FIELDS__, columns=__COLUMNS__, storageKey=__STORAGE__;
const labels={noise_cleanliness:'Noise cleanliness',speech_naturalness:'Speech naturalness',intelligibility:'Intelligibility',overall_quality:'Overall quality'};
let state={listener:'',ratings:{}};try{state=JSON.parse(localStorage.getItem(storageKey))||state}catch(e){}
const host=document.getElementById('trials'),listener=document.getElementById('listener');listener.value=state.listener||'';
for(const trial of trials){const section=document.createElement('section');
const heading=document.createElement('h2');heading.textContent=trial.id+' · '+trial.noise+' · nominal 2.5 dB';section.append(heading);
const refLabel=document.createElement('p');refLabel.textContent='Optional clean reference';section.append(refLabel);
const ref=document.createElement('audio');ref.controls=true;ref.preload='none';ref.src=trial.reference;section.append(ref);
for(const code of ['A','B','C','D']){const key=trial.id+'_'+code, sample=document.createElement('div');sample.className='sample';
const title=document.createElement('h3');title.textContent='Sample '+code;sample.append(title);
const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=trial.audio[code];sample.append(audio);
const grid=document.createElement('div');grid.className='grid';
for(const field of fields){const label=document.createElement('label');label.textContent=labels[field];const select=document.createElement('select');
select.dataset.key=key;select.dataset.field=field;select.setAttribute('aria-label',trial.id+' sample '+code+' '+labels[field]);
for(const value of ['','1','2','3','4','5']){const option=document.createElement('option');option.value=value;option.textContent=value||'Choose';select.append(option)}
select.value=(state.ratings[key]||{})[field]||'';select.addEventListener('change',save);label.append(select);grid.append(label)}sample.append(grid);
const notes=document.createElement('textarea');notes.placeholder='Optional artifact notes';notes.dataset.key=key;notes.dataset.field='notes';
notes.setAttribute('aria-label',trial.id+' sample '+code+' notes');notes.value=(state.ratings[key]||{}).notes||'';notes.addEventListener('input',save);sample.append(notes);section.append(sample)}host.append(section)}
function save(){const id=listener.value.trim();if(id!==state.listener){
try{if(state.listener)localStorage.setItem(storageKey+'_'+state.listener,JSON.stringify(state))}catch(e){}
let next=null;try{if(id)next=JSON.parse(localStorage.getItem(storageKey+'_'+id))}catch(e){}
state=next||{listener:id,ratings:{}};state.listener=id;
for(const el of document.querySelectorAll('[data-key]'))el.value=(state.ratings[el.dataset.key]||{})[el.dataset.field]||'';}
for(const el of document.querySelectorAll('[data-key]')){state.ratings[el.dataset.key]??={};state.ratings[el.dataset.key][el.dataset.field]=el.value}
try{localStorage.setItem(storageKey,JSON.stringify(state));if(state.listener)localStorage.setItem(storageKey+'_'+state.listener,JSON.stringify(state))}catch(e){}let complete=0;
for(const t of trials)for(const c of ['A','B','C','D'])if(fields.every(f=>(state.ratings[t.id+'_'+c]||{})[f]))complete++;
document.getElementById('progress').textContent=complete+' / 24 sample rows complete. Progress is kept in this browser when available.'}
listener.addEventListener('input',save);save();
const escapeCSV=value=>'"'+String(value??'').replaceAll('"','""')+'"';
document.getElementById('download').addEventListener('click',()=>{save();if(!/^[A-Za-z0-9_-]{1,32}$/.test(state.listener)){document.getElementById('progress').textContent='Enter a listener code using letters, numbers, _ or -.';listener.focus();return}
const rows=[columns];for(const t of trials)for(const c of ['A','B','C','D']){const r=state.ratings[t.id+'_'+c]||{};rows.push([state.listener,t.id,c,...fields.map(f=>r[f]||''),r.notes||''])}
const blob=new Blob([rows.map(r=>r.map(escapeCSV).join(',')).join('\\n')+'\\n'],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='ratings_'+state.listener+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
</script></body></html>'''


def build(output):
    selection=json.loads((output/'case_selection.json').read_text())
    cases=[c for c in selection['cases'] if c['purpose']=='predeclared_low_snr_listening']
    if len(cases)!=6:raise ValueError('Expected six predeclared trials')
    plan=json.loads((output/'plan.json').read_text())
    pack=output/'listening_pack';pack.mkdir(exist_ok=True)
    organizer=output/'organizer';organizer.mkdir(exist_ok=True)
    rng=random.Random(351);key=[];public=[];template=[]
    for i,case in enumerate(cases,1):
        trial=f'T{i:02d}';folder=pack/'audio'/trial;folder.mkdir(parents=True,exist_ok=True)
        source=output/'examples'/case['file'];shutil.copyfile(source/'clean.wav',folder/'reference.wav')
        order=METHODS.copy();rng.shuffle(order)
        trial_data=dict(id=trial,noise=case['noise'],reference=f'audio/{trial}/reference.wav',audio={})
        for code,method in zip('ABCD',order):
            filename='noisy.wav' if method=='identity' else method+'.wav'
            target=folder/(code+'.wav');shutil.copyfile(source/filename,target)
            trial_data['audio'][code]=f'audio/{trial}/{code}.wav'
            key.append(dict(trial_id=trial,sample_code=code,file=case['file'],method=method,
                            variant=plan['primary'][method],audio_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
            template.append(dict(listener_id='',trial_id=trial,sample_code=code,**{field:'' for field in FIELDS},notes=''))
        public.append(trial_data)
    csv_output(organizer/'listening_key.csv',key);csv_output(pack/'ratings_template.csv',template)
    storage=json.dumps('eecs351-listening-'+plan['freeze_sha256'][:16])
    page=PAGE.replace('__TRIALS__',json.dumps(public)).replace('__FIELDS__',json.dumps(FIELDS)) \
             .replace('__COLUMNS__',json.dumps(COLUMNS)).replace('__STORAGE__',storage)
    (pack/'index.html').write_text(page)
    (pack/'README.txt').write_text('Extract this entire ZIP, open index.html in a browser, and keep the audio folder beside it.\n'
        'Use headphones, keep volume fixed within each trial, rate all 24 samples, and export your CSV.\n'
        'Use anonymous listener codes. Do not consult the organizer mapping before rating.\n'
        'Scale anchors and dataset attribution appear in index.html. No human ratings are prefilled.\n')
    with zipfile.ZipFile(output/'listening_pack.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(pack.rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(pack))
    status=dict(status='PENDING_HUMAN_RATINGS',trials=6,samples=24,submitted_listeners=0,
                freeze_sha256=plan['freeze_sha256'],organizer_key_in_package=False)
    (output/'listening_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(f'Blind listening pack: {output/"listening_pack.zip"} (24 unrated samples; mapping kept separately)')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/evaluation')
    build(parser.parse_args().output)
