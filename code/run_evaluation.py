#!/usr/bin/env python3
"""Evaluate the frozen plan once on all 240 held-out pairs; never retunes."""
import os
for _key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from scipy.io import loadmat
from prepare_data import read_manifests,validate_logs
from run_tuning import write_json
from speech_metrics import aligned_signals,baseline_metrics,enhancement_metrics
from evaluation_analysis import sha256,verify_freeze,evaluation_plan,aggregate_results

ROOT=Path(__file__).resolve().parents[1]


def csv_output(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def create_protocol(data_root):
    frozen_path=ROOT/'config/frozen_denoisers.json'
    frozen=json.loads(frozen_path.read_text());verify_freeze(ROOT,frozen)
    manifests=read_manifests(ROOT);validate_logs(manifests,data_root)
    records=manifests['evaluation']
    plan=evaluation_plan(frozen,sha256(frozen_path))
    inventory_path=ROOT/'outputs/setup/download_inventory.json';expected={}
    if inventory_path.exists():
        inventory=json.loads(inventory_path.read_text())
        if inventory.get('status')=='PASS': expected={x['path']:x['sha256'] for x in inventory['files']}
    hashes={}
    for row in records:
        for group in ['clean','noisy']:
            path=f'{group}_testset_wav/{row["File"]}.wav'
            hashes[path]=sha256(data_root/path)
            if path in expected and hashes[path]!=expected[path]: raise ValueError(f'Audio changed: {path}')
    sources=['code/run_evaluation.py','code/prepare_evaluation.m','code/evaluation_analysis.py']
    protocol=dict(schema_version=1,split='evaluation',pairs=240,fs=16000,
        freeze_sha256=sha256(frozen_path),evaluation_manifest_sha256=sha256(ROOT/'data/evaluation_manifest.csv'),
        evaluation_audio_digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
        source_sha256={p:sha256(ROOT/p) for p in sources},
        common_crop_samples_each_end=64,created_utc=datetime.now(timezone.utc).isoformat())
    # Creation time is informative, not part of the reproducible resume signature.
    stable={k:v for k,v in protocol.items() if k!='created_utc'}
    protocol['signature']=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest()
    return protocol,plan,records


def score_file(path,protocol,plan,record):
    raw=loadmat(path,simplify_cells=True)
    if raw['signature']!=protocol['signature'] or raw['split']!='evaluation': raise ValueError('Wrong split/provenance')
    if raw['file']!=record['File'] or raw['noise']!=record['Noise'] or float(raw['nominalSNR'])!=record['NominalSNR']:
        raise ValueError('Held-out labels disagree with the fixed manifest')
    if int(raw['fs'])!=16000 or list(raw['variantIds'])!=[v['id'] for v in plan['variants']]:
        raise ValueError('Unexpected evaluation variant order/sample rate')
    s,y,outputs,interval=aligned_signals(raw['reference'],raw['noisy'],raw['processed'],raw['validRanges'])
    if interval!=(65,len(raw['reference'])-64): raise ValueError('Unexpected scoring interval')
    baseline=baseline_metrics(s,y,16000);rows=[]
    for k,v in enumerate(plan['variants']):
        if v['method']=='identity':
            if not np.array_equal(outputs[:,k],y): raise ValueError('Identity changed')
            metrics=dict(baseline,snr_out_db=baseline['snr_in_db'],stoi_out=baseline['stoi_in'],delta_snr_db=0.,delta_stoi=0.)
        else: metrics=enhancement_metrics(s,outputs[:,k],16000,baseline)
        rows.append(dict(file=record['File'],noise=record['Noise'],nominal_snr_db=record['NominalSNR'],
            speaker=record['File'].split('_')[0],variant=v['id'],method=v['method'],role=v['role'],
            valid_start=interval[0],valid_end=interval[1],samples=len(s),**metrics,
            update_frames=int(raw['updateCounts'][k]),peak_abs=float(np.max(np.abs(raw['processed'][:,k])))))
    return dict(signature=protocol['signature'],file=record['File'],rows=rows)


def worker(path,target,protocol,plan,record):
    result=score_file(path,protocol,plan,record);write_json(target,result);Path(path).unlink()


def choose_examples(records,rows,plan):
    selected=[]
    for noise in ['bus','office','cafe']:
        for speaker in ['p232','p257']:
            row=next(r for r in records if r['Noise']==noise and r['NominalSNR']==2.5 and r['File'].startswith(speaker))
            selected.append(dict(file=row['File'],purpose='predeclared_low_snr_listening',noise=noise,speaker=speaker,nominal_snr_db=2.5))
    primary=[r for r in rows if r['role']=='primary' and r['method']!='identity']
    worst=min(primary,key=lambda r:(r['delta_stoi'],r['file'],r['method']))
    tradeoff=max((r for r in primary if r['delta_stoi']<0 and r['delta_snr_db']>0),
                 key=lambda r:(r['delta_snr_db'],r['file']),default=worst)
    for purpose,row in [('posthoc_largest_stoi_loss',worst),('posthoc_snr_gain_with_stoi_loss',tradeoff)]:
        selected.append(dict(file=row['file'],purpose=purpose,noise=row['noise'],speaker=row['speaker'],
            nominal_snr_db=row['nominal_snr_db'],method=row['method'],delta_stoi=row['delta_stoi'],delta_snr_db=row['delta_snr_db']))
    return dict(status='LISTENING_RATINGS_PENDING',cases=selected,
                note='Six predeclared low-SNR pilot trials plus two explicitly posthoc diagnostic selections; no retuning.')


def run(args):
    start=time.monotonic();out=args.output.resolve()
    protocol,plan,records=create_protocol(Path(os.environ.get('EECS351_DATA_ROOT') or ROOT/'data'))
    if out.exists() and any(out.iterdir()):
        previous=json.loads((out/'protocol.json').read_text())
        if not args.resume or previous['signature']!=protocol['signature']:
            raise ValueError('Use a new output directory or --resume with unchanged frozen inputs/source')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'protocol.json',protocol);write_json(out/'plan.json',plan)
    write_json(out/'status.json',dict(status='RUNNING',stage='frozen output generation'))
    matlab=args.matlab or shutil.which('matlab') or '/Applications/MATLAB_R2024b.app/bin/matlab'
    quote=lambda p:str(p).replace("'","''")
    command=f"addpath('{quote(ROOT/'code')}'); prepare_evaluation('{quote(out)}');"
    print(f'Evaluating {len(plan["variants"])} frozen variants x 240 held-out pairs',flush=True)
    subprocess.run([matlab,'-batch',command],cwd=ROOT,check=True)
    scores=out/'scores';scores.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[]
        for record in records:
            target=scores/(record['File']+'.json')
            if target.exists() and json.loads(target.read_text())['signature']==protocol['signature']:continue
            jobs.append(pool.submit(worker,out/'bundles'/(record['File']+'.mat'),target,protocol,plan,record))
        for i,future in enumerate(as_completed(jobs),1):
            future.result()
            if i%40==0 or i==len(jobs):print(f'Scored held-out files: {i}/{len(jobs)}',flush=True)
    rows=[]
    for record in records:
        cached=json.loads((scores/(record['File']+'.json')).read_text())
        if cached['signature']!=protocol['signature']:raise ValueError('Stale held-out checkpoint')
        rows.extend(cached['rows'])
    summaries=aggregate_results(rows,plan,records)
    csv_output(out/'per_file_metrics.csv',rows)
    for name,table in summaries.items():csv_output(out/(name+'.csv'),table)
    write_json(out/'case_selection.json',choose_examples(records,rows,plan))
    if sha256(ROOT/'config/frozen_denoisers.json')!=protocol['freeze_sha256']:
        raise ValueError('Frozen configuration changed during evaluation')
    provenance=dict(protocol=protocol,plan=plan,artifacts={p.name:sha256(p) for p in out.glob('*.csv')},
                    interpretation='File-resampling intervals conditional on two speakers; no population-wide speaker claim',
                    subjective_listening_status='PENDING_HUMAN_RATINGS')
    write_json(out/'provenance.json',provenance)
    write_json(out/'status.json',dict(status='PASS',pairs=240,variants=len(plan['variants']),scores=len(rows),
                                     elapsed_seconds=round(time.monotonic()-start,2)))
    for row in summaries['overall']:
        if row['role']=='primary':print(f'{row["method"]}: STOI {row["mean_stoi"]:.6f}, '
            f'dSTOI {row["mean_delta_stoi"]:+.6f}, dSNR {row["mean_delta_snr_db"]:+.3f} dB')
    print(f'PASS: complete held-out results in {out}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/evaluation')
    parser.add_argument('--matlab');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,choices=range(1,9),default=4)
    args=parser.parse_args()
    try:run(args)
    except Exception as error:
        if (args.output/'status.json').exists():write_json(args.output/'status.json',dict(status='FAIL',error=str(error)))
        raise
