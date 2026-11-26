#!/usr/bin/env python3
"""Copy verified held-out tables/figures and an unrated listening template into the repo."""
import json
from pathlib import Path
import shutil
from evaluation_analysis import sha256,verify_freeze
from run_tuning import write_json

ROOT=Path(__file__).resolve().parents[1]


def publish(output):
    status=json.loads((output/'status.json').read_text())
    if status.get('status')!='PASS' or status.get('scores')!=1920:raise ValueError('Incomplete evaluation')
    frozen=json.loads((ROOT/'config/frozen_denoisers.json').read_text());verify_freeze(ROOT,frozen)
    provenance=json.loads((output/'provenance.json').read_text())
    if sha256(ROOT/'config/frozen_denoisers.json')!=provenance['protocol']['freeze_sha256']:
        raise ValueError('Frozen settings changed')
    for path,expected in provenance['protocol']['source_sha256'].items():
        if sha256(ROOT/path)!=expected:raise ValueError(f'Evaluation source changed: {path}')
    for filename,expected in provenance['artifacts'].items():
        if sha256(output/filename)!=expected:raise ValueError(f'Metric artifact changed: {filename}')
    reports=ROOT/'reports/evaluation';reports.mkdir(parents=True,exist_ok=True)
    figures=ROOT/'assets/figures/results';figures.mkdir(parents=True,exist_ok=True)
    public=[]
    for name in ['per_file_metrics.csv','overall.csv','by_condition.csv','by_speaker.csv','noise_mode_comparisons.csv',
                 'case_selection.json','listening_status.json']:
        target=reports/name;shutil.copyfile(output/name,target);public.append(target)
    target=reports/'listening_ratings_template.csv'
    shutil.copyfile(output/'listening_pack/ratings_template.csv',target);public.append(target)
    shutil.copyfile(output/'plan.json',ROOT/'config/evaluation_plan.json')
    captions={
        'heldout-overall.png':'240 held-out pairs; four frozen primary systems; descriptive 95% stratified file-bootstrap intervals.',
        'heldout-conditions.png':'Primary changes in all 12 noise/SNR cells; 20 paired files per cell; STOI labels are scaled by 1000.',
        'heldout-speakers.png':'Primary paired changes for each of the two held-out speakers; 120 files per speaker.',
        'tuning-strength-tradeoff.png':'400 tuning pairs only: subtraction alpha=1,2,4, beta=0.01, noise fraction=0.2; not a new test-set search.'}
    for name in captions:
        target=figures/name;shutil.copyfile(output/'figures'/name,target);public.append(target)
    selection=json.loads((output/'case_selection.json').read_text())
    for noise in ['bus','office','cafe']:
        case=next(c for c in selection['cases'] if c['purpose']=='predeclared_low_snr_listening' and c['noise']==noise)
        name=f'example-{noise}.png';target=figures/name
        shutil.copyfile(output/'examples'/case['file']/'spectrograms.png',target);public.append(target)
        captions[name]=f'Predeclared {noise} example {case["file"]}, nominal 2.5 dB; common crop, original common spectral scale.'
    worst=next(c for c in selection['cases'] if c['purpose']=='posthoc_largest_stoi_loss')
    name='posthoc-stoi-loss.png';target=figures/name
    shutil.copyfile(output/'examples'/worst['file']/'spectrograms.png',target);public.append(target)
    captions[name]=f'Explicit posthoc diagnostic: {worst["file"]}, {worst["method"]}, delta STOI {worst["delta_stoi"]:.6f}. Not a representative sample.'
    write_json(reports/'figure_captions.json',captions);public.append(reports/'figure_captions.json')
    listening=dict(status='PENDING_HUMAN_RATINGS',trials=6,samples=24,submitted_listeners=0,
        selection='Predeclared first file per noise/speaker at nominal 2.5 dB; pilot only',
        systems='Four primary systems; noise-update variants are not in this listening pilot',
        scale='Four ordinal 1-to-5 ratings: noise cleanliness, speech naturalness, intelligibility, overall quality',
        randomization='Fixed seed 351; independent method-label permutation per trial, common order across listeners',
        playback='Same valid samples and one export gain per utterance; no independent normalization',
        aggregation='Require all 24 samples per listener; decode after collection; report actual listener count',
        zip_sha256=sha256(output/'listening_pack.zip'),
        organizer_key_in_package=False)
    write_json(reports/'listening_protocol.json',listening);public.append(reports/'listening_protocol.json')
    sources=['code/plot_evaluation.m','code/export_evaluation_examples.m','code/build_listening_pack.py',
             'code/summarize_listening.py','code/publish_evaluation.py']
    provenance['reporting_source_sha256']={p:sha256(ROOT/p) for p in sources}
    provenance['published_artifacts_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in public}
    write_json(reports/'provenance.json',provenance)
    print(f'Published {len(public)} verified tables/figures/templates; human ratings remain pending')


if __name__=='__main__':publish(ROOT/'outputs/evaluation')
