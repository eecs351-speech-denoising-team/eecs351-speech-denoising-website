"""Frozen-plan verification and paired summaries for the 240-file held-out set."""
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_freeze(root, frozen):
    if frozen.get('status') != 'FROZEN':
        raise ValueError('Evaluation requires a frozen training-only selection')
    for path, expected in frozen['protocol']['source_sha256'].items():
        if sha256(root/path) != expected:
            raise ValueError(f'Source changed since freezing: {path}')
    for name, version in frozen['protocol']['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Package changed since freezing: {name}')
    if sha256(root/'data/tuning_manifest.csv') != frozen['protocol']['tuning_manifest_sha256']:
        raise ValueError('Tuning manifest changed since freezing')
    if sha256(root/'config/tuning_candidates.json') != frozen['protocol']['candidate_sha256']:
        raise ValueError('Tuning candidate grid changed since freezing')


def evaluation_plan(frozen, freeze_hash):
    variants=[]
    primary_ids=set(frozen['primary'].values())
    for c in frozen['selected']:
        variants.append(dict(id=c['id'],method=c['method'],options=c['options'],
                             role='primary' if c['id'] in primary_ids else 'independent_noise_mode'))
    for pair in frozen['matched_noise_mode_pairs']:
        id=pair['counterpart_score_candidate']
        if not any(c['id']==id for c in variants):
            variants.append(dict(id=id,method=pair['method'],options=pair['counterpart_options'],role='matched_noise_mode'))
    return dict(schema_version=1,freeze_sha256=freeze_hash,variants=variants,
                primary=frozen['primary'],matched_noise_mode_pairs=frozen['matched_noise_mode_pairs'],
                listening_selection='First manifest file per noise and speaker at nominal 2.5 dB: six low-SNR trials',
                bootstrap=dict(seed=351,replicates=2000,strata=['noise','nominal_snr_db','speaker'],
                               interpretation='Descriptive file-resampling intervals conditional on these two speakers and conditions'))


def validate_results(rows, variants, manifest):
    expected={r['File']:(r['Noise'],float(r['NominalSNR'])) for r in manifest}
    ids={v['id'] for v in variants}
    variant_lookup={v['id']:v for v in variants}
    if len(expected)!=240 or len(rows)!=240*len(ids):
        raise ValueError('Incomplete held-out table; no cases may be omitted')
    keys=[(r['file'],r['variant']) for r in rows]
    if len(set(keys))!=len(keys) or set(keys)!={(file,id) for file in expected for id in ids}:
        raise ValueError('Duplicate, missing or unexpected held-out file/variant')
    for r in rows:
        if r['method']!=variant_lookup[r['variant']]['method'] or r['role']!=variant_lookup[r['variant']]['role']:
            raise ValueError('Held-out method/role differs from the frozen plan')
        if (r['noise'],float(r['nominal_snr_db']))!=expected[r['file']] or r['speaker']!=r['file'].split('_')[0]:
            raise ValueError('Held-out condition/speaker labels changed')
        values=[r[k] for k in ['stoi_in','stoi_out','delta_stoi','snr_in_db','snr_out_db','delta_snr_db']]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite held-out score')
        if abs((r['stoi_out']-r['stoi_in'])-r['delta_stoi'])>1e-12 or \
                abs((r['snr_out_db']-r['snr_in_db'])-r['delta_snr_db'])>1e-12:
            raise ValueError('Inconsistent paired metric changes')
    cells=Counter((r['Noise'],float(r['NominalSNR'])) for r in manifest)
    if len(cells)!=12 or set(cells.values())!={20}:
        raise ValueError('Expected 12 held-out cells with 20 files each')
    if Counter(file.split('_')[0] for file in expected)!={'p232':120,'p257':120}:
        raise ValueError('Expected 120 files per held-out speaker')


def bootstrap_indices(records, replicates=2000, seed=351):
    rng=np.random.default_rng(seed)
    groups={}
    for i,r in enumerate(records):
        groups.setdefault((r['noise'],float(r['nominal_snr_db']),r['speaker']),[]).append(i)
    return np.concatenate([rng.choice(indices,size=(replicates,len(indices)),replace=True)
                           for _,indices in sorted(groups.items())],axis=1)


def summary(rows, indices=None):
    result=dict(pairs=len(rows),mean_stoi=float(np.mean([r['stoi_out'] for r in rows])),
                mean_stoi_in=float(np.mean([r['stoi_in'] for r in rows])),
                mean_delta_stoi=float(np.mean([r['delta_stoi'] for r in rows])),
                mean_snr_in_db=float(np.mean([r['snr_in_db'] for r in rows])),
                mean_snr_out_db=float(np.mean([r['snr_out_db'] for r in rows])),
                mean_delta_snr_db=float(np.mean([r['delta_snr_db'] for r in rows])),
                median_delta_stoi=float(np.median([r['delta_stoi'] for r in rows])),
                stoi_improved_fraction=float(np.mean([r['delta_stoi']>0 for r in rows])),
                snr_improved_fraction=float(np.mean([r['delta_snr_db']>0 for r in rows])))
    if indices is not None:
        for key in ['delta_stoi','delta_snr_db']:
            means=np.asarray([r[key] for r in rows])[indices].mean(axis=1)
            lo,hi=np.percentile(means,[2.5,97.5])
            result[key+'_ci_low']=float(lo);result[key+'_ci_high']=float(hi)
    return result


def aggregate_results(rows, plan, manifest):
    validate_results(rows,plan['variants'],manifest)
    files=[r['File'] for r in manifest]
    lookup={(r['file'],r['variant']):r for r in rows}
    baseline=[lookup[(f,plan['primary']['identity'])] for f in files]
    boot=bootstrap_indices(baseline,**{k:plan['bootstrap'][k] for k in ['replicates','seed']})
    overall=[];conditions=[];speakers=[]
    for variant in plan['variants']:
        selected=[lookup[(f,variant['id'])] for f in files]
        common=dict(variant=variant['id'],method=variant['method'],role=variant['role'])
        overall.append(dict(common,**summary(selected,boot)))
        for noise in ['bus','office','cafe']:
            for snr in [2.5,7.5,12.5,17.5]:
                subset=[r for r in selected if r['noise']==noise and float(r['nominal_snr_db'])==snr]
                conditions.append(dict(common,noise=noise,nominal_snr_db=snr,**summary(subset)))
        for speaker in ['p232','p257']:
            speakers.append(dict(common,speaker=speaker,**summary([r for r in selected if r['speaker']==speaker])))
    comparisons=[]
    for pair in plan['matched_noise_mode_pairs']:
        fixed_id=pair['anchor_candidate'];updated_id=pair['counterpart_score_candidate']
        # These primary anchors were fixed by tuning; do not select from test scores.
        anchor=next(v for v in plan['variants'] if v['id']==fixed_id)
        if anchor['options']['noiseMode']!='fixed':
            fixed_id,updated_id=updated_id,fixed_id
        differences=[dict(delta_stoi=lookup[(f,updated_id)]['stoi_out']-lookup[(f,fixed_id)]['stoi_out'],
                          delta_snr_db=lookup[(f,updated_id)]['snr_out_db']-lookup[(f,fixed_id)]['snr_out_db']) for f in files]
        row=dict(method=pair['method'],comparison='matched_updated_minus_fixed',pairs=240,
                 fixed_variant=fixed_id,updated_variant=updated_id)
        for key in ['delta_stoi','delta_snr_db']:
            values=np.asarray([r[key] for r in differences]);means=values[boot].mean(axis=1)
            row['mean_'+key]=float(values.mean())
            row[key+'_ci_low'],row[key+'_ci_high']=map(float,np.percentile(means,[2.5,97.5]))
        comparisons.append(row)
    return dict(overall=overall,by_condition=conditions,by_speaker=speakers,noise_mode_comparisons=comparisons)
