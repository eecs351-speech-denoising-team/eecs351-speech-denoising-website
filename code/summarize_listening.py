#!/usr/bin/env python3
"""Decode complete real listener CSVs; blank templates never become ratings."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from build_listening_pack import FIELDS
from run_evaluation import csv_output

ROOT=Path(__file__).resolve().parents[1]


def decode_ratings(key_rows,rating_rows):
    key={(r['trial_id'],r['sample_code']):r for r in key_rows}
    if len(key)!=24 or len(key_rows)!=24:raise ValueError('Expected a unique 24-sample organizer key')
    decoded=[];seen=set()
    for row in rating_rows:
        if not any(str(row.get(field,'')).strip() for field in FIELDS):continue
        listener=row.get('listener_id','').strip();sample=(row.get('trial_id'),row.get('sample_code'))
        if not listener or sample not in key:raise ValueError('Missing listener code or unknown blind sample')
        identity=(listener,*sample)
        if identity in seen:raise ValueError('Duplicate listener/sample rating')
        seen.add(identity)
        values={}
        for field in FIELDS:
            text=str(row.get(field,'')).strip()
            if text not in {'1','2','3','4','5'}:raise ValueError('Every rated sample requires four integer scores from 1 to 5')
            values[field]=int(text)
        decoded.append(dict(listener_id=listener,trial_id=sample[0],sample_code=sample[1],
                            method=key[sample]['method'],file=key[sample]['file'],**values,notes=row.get('notes','')))
    for listener in {r['listener_id'] for r in decoded}:
        actual={(r['trial_id'],r['sample_code']) for r in decoded if r['listener_id']==listener}
        if actual!=set(key):raise ValueError(f'{listener}: incomplete trial set; all 24 samples are required')
    return decoded


def summarize(decoded):
    if not decoded:return []
    result=[]
    for method in ['identity','fir','spectral_subtraction','wiener']:
        rows=[r for r in decoded if r['method']==method]
        listeners=sorted({r['listener_id'] for r in rows})
        record=dict(method=method,listeners=len(listeners),sample_ratings=len(rows))
        for field in FIELDS:
            per_listener=[np.mean([r[field] for r in rows if r['listener_id']==listener]) for listener in listeners]
            record['mean_'+field]=float(np.mean(per_listener))
            record['between_listener_sd_'+field]=float(np.std(per_listener,ddof=1)) if len(listeners)>1 else ''
        result.append(record)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key',type=Path,default=ROOT/'outputs/evaluation/organizer/listening_key.csv')
    parser.add_argument('--ratings',type=Path,nargs='*',default=[])
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/evaluation/listening_results')
    args=parser.parse_args()
    with args.key.open() as f:key_rows=list(csv.DictReader(f))
    ratings=[]
    for path in args.ratings:
        with path.open() as f:ratings.extend(csv.DictReader(f))
    decoded=decode_ratings(key_rows,ratings);table=summarize(decoded)
    args.output.mkdir(parents=True,exist_ok=True)
    status=dict(status='COMPLETE_PILOT' if decoded else 'PENDING_HUMAN_RATINGS',
                listeners=len({r['listener_id'] for r in decoded}),sample_ratings=len(decoded))
    if decoded:
        csv_output(args.output/'decoded_ratings.csv',decoded);csv_output(args.output/'summary.csv',table)
    (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status))
