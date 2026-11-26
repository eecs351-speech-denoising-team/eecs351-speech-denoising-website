#!/usr/bin/env python3
"""Generate MATLAB outputs, score all 400 tuning pairs, and freeze a bounded search.

Example: .venv/bin/python code/run_tuning.py --workers 4
Never reads held-out audio. Checkpoints permit --resume with unchanged inputs/code.
"""
import os
for _name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[_name] = '1'  # Avoid nested BLAS threads in independent scoring workers.

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.io import loadmat
from prepare_data import read_manifests, validate_logs
from speech_metrics import aligned_signals, baseline_metrics, enhancement_metrics
from tuning_candidates import build_candidates

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['code/denoise_speech.m', 'code/load_speech_pair.m', 'code/project_config.m',
           'code/prepare_tuning.m', 'code/score_stoi.py', 'code/speech_metrics.py',
           'code/tuning_candidates.py', 'code/run_tuning.py', 'code/prepare_data.py',
           'requirements.txt', 'tests/test_metrics.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def make_protocol(data_root):
    manifests = read_manifests(ROOT)
    validate_logs(manifests, data_root)
    # Verify only training audio; keep an aggregate content digest in the freeze.
    entries = {}
    inventory_path = ROOT/'outputs/setup/download_inventory.json'
    expected = {}
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
        if inventory.get('status') == 'PASS':
            expected = {x['path']: x['sha256'] for x in inventory['files']}
    for row in manifests['tuning']:
        for group in ['clean', 'noisy']:
            relative = f'{group}_trainset_28spk_wav/{row["File"]}.wav'
            actual = digest(data_root/relative)
            if relative in expected and expected[relative] != actual:
                raise ValueError(f'Audio changed since official CRC validation: {relative}')
            entries[relative] = actual
    plan = json.loads((ROOT/'config/tuning_candidates.json').read_text())
    if plan['candidates'] != build_candidates():
        raise ValueError('Candidate JSON differs from the predeclared grid builder')
    protocol = dict(schema_version=1, split='tuning', pairs=400, candidate_count=41,
                    fs=16000, common_crop_samples_each_end=64,
                    alignment='Use denoiser-aligned outputs; no additional shift or gain fit',
                    metrics='Classic pystoi, whole-valid-interval SNR; per-utterance arithmetic means',
                    selection='Descending mean STOI, then mean delta SNR, then candidate ID',
                    search='Core suppression grid plus one-factor-at-a-time noise-estimator variations',
                    tuning_manifest_sha256=digest(ROOT/'data/tuning_manifest.csv'),
                    candidate_sha256=digest(ROOT/'config/tuning_candidates.json'),
                    tuning_audio_digest=hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
                    source_sha256={name: digest(ROOT/name) for name in SOURCES},
                    packages={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'pystoi']})
    protocol['signature'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol, plan, manifests['tuning']


def score_bundle(path, protocol, candidates, expected):
    raw = loadmat(path, simplify_cells=True)
    if raw['signature'] != protocol['signature'] or raw['split'] != 'tuning':
        raise ValueError(f'{path}: wrong split or provenance signature')
    if raw['file'] != expected['File'] or raw['noise'] != expected['Noise'] or float(raw['nominalSNR']) != expected['NominalSNR']:
        raise ValueError(f'{path}: bundle condition labels disagree with manifest')
    if int(raw['fs']) != protocol['fs'] or list(raw['candidateIds']) != [c['id'] for c in candidates]:
        raise ValueError(f'{path}: sample rate or candidate ordering changed')
    reference, noisy, processed, interval = aligned_signals(
        raw['reference'], raw['noisy'], raw['processed'], raw['validRanges'])
    n = len(raw['reference'])
    if interval != (65, n-64):
        raise ValueError(f'{path}: unexpected common interval {interval}')
    baseline = baseline_metrics(reference, noisy, int(raw['fs']))
    rows = []
    for k, candidate in enumerate(candidates):
        if candidate['method'] == 'identity':
            if not np.array_equal(processed[:, k], noisy):
                raise ValueError('Identity output differs from the noisy baseline')
            metrics = dict(baseline, snr_out_db=baseline['snr_in_db'], delta_snr_db=0.0,
                           stoi_out=baseline['stoi_in'], delta_stoi=0.0)
        else:
            metrics = enhancement_metrics(reference, processed[:, k], int(raw['fs']), baseline)
        row = dict(file=raw['file'], noise=raw['noise'], nominal_snr_db=float(raw['nominalSNR']),
                   speaker=raw['file'].split('_')[0], candidate=candidate['id'], method=candidate['method'],
                   family=candidate['family'], valid_start=interval[0], valid_end=interval[1],
                   samples=len(reference), **metrics,
                   update_frames=int(raw['updateCounts'][k]), fallback=str(raw['fallbackReasons'][k]),
                   peak_abs=float(np.max(np.abs(raw['processed'][:, k]))))
        if not all(np.isfinite(row[key]) for key in ['snr_in_db','snr_out_db','delta_snr_db','stoi_in','stoi_out','delta_stoi']):
            raise ValueError(f'Nonfinite metric: {raw["file"]}/{candidate["id"]}')
        rows.append(row)
    return dict(signature=protocol['signature'], file=raw['file'], matlab_version=raw['matlabVersion'], rows=rows)


def worker(path, output, protocol, candidates, record, keep_cache):
    result = score_bundle(path, protocol, candidates, record)
    write_json(output, result)
    if not keep_cache:
        Path(path).unlink()
    return record['File']


def summarize(rows, candidates, expected_files):
    if len(rows) != len(candidates)*len(expected_files):
        raise ValueError('Incomplete metric table: refusing to select or freeze')
    summaries = []
    for candidate in candidates:
        selected = [r for r in rows if r['candidate'] == candidate['id']]
        if len(selected) != 400 or {r['file'] for r in selected} != set(expected_files):
            raise ValueError(f'{candidate["id"]}: expected 400 unique tuning results')
        summaries.append(dict(candidate=candidate['id'], method=candidate['method'], family=candidate['family'], pairs=400,
            mean_stoi=float(np.mean([r['stoi_out'] for r in selected])),
            mean_delta_stoi=float(np.mean([r['delta_stoi'] for r in selected])),
            mean_delta_snr_db=float(np.mean([r['delta_snr_db'] for r in selected])),
            mean_snr_out_db=float(np.mean([r['snr_out_db'] for r in selected])),
            stoi_improved_fraction=float(np.mean([r['delta_stoi'] > 0 for r in selected])),
            mean_update_frames=float(np.mean([r['update_frames'] for r in selected])),
            peak_abs=max(r['peak_abs'] for r in selected)))
    return sorted(summaries, key=lambda r: (-r['mean_stoi'], -r['mean_delta_snr_db'], r['candidate']))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze_selection(protocol, candidates, summaries):
    lookup = {c['id']: c for c in candidates}
    selected = []
    for family in ['baseline','fir','ss_fixed','ss_pause_updated','wiener_fixed','wiener_pause_updated']:
        winner = next(row for row in summaries if row['family'] == family)
        selected.append(dict(lookup[winner['candidate']], tuning_metrics=winner))
    primary = {method: next(row['candidate'] for row in summaries if row['method'] == method)
               for method in ['identity','fir','spectral_subtraction','wiener']}
    # Anchored mode comparisons change only noiseMode. Fixed-mode detector knobs
    # are unused in its gain; find the numerically equivalent scored counterpart.
    pairs = []
    for method in ['spectral_subtraction','wiener']:
        anchor = lookup[primary[method]]
        counterpart = dict(anchor['options'])
        counterpart['noiseMode'] = 'pause_updated' if counterpart['noiseMode'] == 'fixed' else 'fixed'
        effective_keys = ['noiseFraction','alpha','beta'] if method == 'spectral_subtraction' else ['noiseFraction','gainFloor']
        match = next(c for c in candidates if c['method'] == method and c['options']['noiseMode'] == counterpart['noiseMode']
                     and all(c['options'][key] == counterpart[key] for key in effective_keys)
                     and (counterpart['noiseMode'] == 'fixed' or c['options'] == counterpart))
        pairs.append(dict(method=method, anchor_candidate=anchor['id'],
                          counterpart_score_candidate=match['id'], counterpart_options=counterpart))
    return dict(schema_version=1, status='FROZEN', created_utc=datetime.now(timezone.utc).isoformat(),
                protocol=protocol, selected=selected, primary=primary, matched_noise_mode_pairs=pairs,
                note='Training-only selection. No held-out performance claimed; mode-family winners are independently tuned.')


def run(args):
    start = time.monotonic()
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_metrics.py'],
                   cwd=ROOT,check=True)
    output = args.output.resolve()
    protocol, plan, manifest = make_protocol(Path(os.environ.get('EECS351_DATA_ROOT') or ROOT/'data'))
    if output.exists() and any(output.iterdir()):
        previous = json.loads((output/'protocol.json').read_text())
        if not args.resume or previous['signature'] != protocol['signature']:
            raise ValueError('Output already exists: use a new --output, or --resume with identical inputs and code')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'protocol.json', protocol)
    write_json(output/'candidates.json', plan)
    write_json(output/'status.json', dict(status='RUNNING', stage='MATLAB generation'))
    matlab = args.matlab or shutil.which('matlab') or '/Applications/MATLAB_R2024b.app/bin/matlab'
    quote = lambda p: str(p).replace("'", "''")
    command = f"addpath('{quote(ROOT/'code')}'); prepare_tuning('{quote(output)}');"
    print(f'Preparing {len(plan["candidates"])} candidates x {len(manifest)} tuning pairs; no test audio', flush=True)
    subprocess.run([matlab, '-batch', command], cwd=ROOT, check=True)
    write_json(output/'status.json', dict(status='RUNNING', stage='batch scoring'))
    scores = output/'scores'; scores.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for record in manifest:
            target = scores/(record['File']+'.json')
            if target.exists() and json.loads(target.read_text())['signature'] == protocol['signature']:
                continue
            futures.append(pool.submit(worker, output/'bundles'/(record['File']+'.mat'), target,
                                       protocol, plan['candidates'], record, args.keep_cache))
        for k, future in enumerate(as_completed(futures), 1):
            future.result()
            if k % 20 == 0 or k == len(futures):
                print(f'Scored tuning pairs: {k}/{len(futures)}', flush=True)
    rows = []
    for record in manifest:
        cached = json.loads((scores/(record['File']+'.json')).read_text())
        if cached['signature'] != protocol['signature']:
            raise ValueError('Stale score cache')
        rows.extend(cached['rows'])
    summaries = summarize(rows, plan['candidates'], [r['File'] for r in manifest])
    write_csv(output/'per_file_metrics.csv', rows)
    write_csv(output/'candidate_summary.csv', summaries)
    frozen = freeze_selection(protocol, plan['candidates'], summaries)
    write_json(output/'frozen_denoisers.json', frozen)
    write_json(output/'status.json', dict(status='PASS', pairs=400, candidates=41, scores=len(rows),
                                        elapsed_seconds=round(time.monotonic()-start, 2)))
    for winner in frozen['selected']:
        metric = winner['tuning_metrics']
        print(f'{winner["family"]}: {winner["id"]} STOI={metric["mean_stoi"]:.6f} '
              f'dSTOI={metric["mean_delta_stoi"]:+.6f} dSNR={metric["mean_delta_snr_db"]:+.3f} dB')
    print(f'PASS: all {len(rows)} tuning measurements; frozen selection: {output/"frozen_denoisers.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/tuning')
    parser.add_argument('--matlab', help='MATLAB executable; detected automatically on the setup Mac')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--keep-cache', action='store_true', help='Keep large intermediate MAT bundles after scoring')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if (args.output/'status.json').exists():
            write_json(args.output/'status.json', dict(status='FAIL', error=str(error)))
        raise
