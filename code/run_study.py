#!/usr/bin/env python3
"""Learned-alpha study: alpha tuning on training data and final test scoring.

Systems (all at 16 kHz on the data/cache16k files):
  noisy            unprocessed input
  fir              ITU-T G.722 wideband band-pass, 50-7000 Hz, order 512
  dd:<alpha>       decision-directed Wiener, xi_min = -25 dB, noise from quietest 10% frames
  learned:<ckpt>   the same Wiener with a per time-frequency alpha from the trained network

Every system is scored on samples [256, n-256), the interval the order-512 FIR fully supports.

  tune-alpha  400 tuning pairs (data/tuning_manifest.csv, training speakers only).
              Rule, fixed before tuning: highest mean PESQ among alphas whose mean STOI is
              at most 0.005 below the unprocessed input. If no alpha passes, the highest-PESQ
              alpha is reported and flagged as failing the STOI guard.
  evaluate    every system on the full 824-pair test set, once, after all choices are frozen.
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsp
import quality_metrics as qm

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'data'/'cache16k'
MARGIN = 256
ALPHA_GRID = [0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998]
STOI_GUARD = 0.005
_LEARNED = {}


def load_pair(split, name):
    with np.load(CACHE/split/f'{name}.npz') as d:
        return d['clean'].astype(np.float64), d['noisy'].astype(np.float64)


def process(system, noisy):
    if system == 'noisy':
        return noisy
    if system == 'fir':
        return dsp.fir_bandpass(noisy)
    kind, _, arg = system.partition(':')
    if kind == 'dd':
        return dsp.wiener_dd(noisy, alpha=float(arg))
    if kind == 'learned':
        import learned_alpha
        if arg not in _LEARNED:
            _LEARNED[arg] = learned_alpha.load(arg)
        return learned_alpha.denoise(_LEARNED[arg], noisy)
    raise ValueError(f'Unknown system {system!r}')


def score_file(job):
    split, name, systems = job
    clean, noisy = load_pair(split, name)
    keep = slice(MARGIN, len(noisy)-MARGIN)
    rows = []
    for system in systems:
        out = process(system, noisy)
        rows.append(dict(name=name, system=system, **qm.all_metrics(clean[keep], out[keep])))
    return rows


def run(split, names, systems, workers):
    jobs = [(split, n, systems) for n in names]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for k, result in enumerate(pool.map(score_file, jobs, chunksize=4), 1):
            rows.extend(result)
            if k % 100 == 0 or k == len(jobs):
                print(f'scored {k}/{len(jobs)} files', flush=True)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(rows, keys=('system',)):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, members in groups.items():
        out.append(dict(zip(keys, key), files=len(members),
                        **{m: float(np.mean([r[m] for r in members])) for m in ['pesq', 'stoi', 'ssnr', 'si_sdr']}))
    return out


def tune_alpha(args):
    with (ROOT/'data'/'tuning_manifest.csv').open() as f:
        names = [r['File'] for r in csv.DictReader(f)]
    systems = ['noisy'] + [f'dd:{a}' for a in ALPHA_GRID]
    rows = run('train', names, systems, args.workers)
    out = ROOT/'reports'/'learned_alpha'
    write_csv(out/'tuning_alpha_per_file.csv', rows)
    summary = summarize(rows)
    write_csv(out/'tuning_alpha_summary.csv', summary)
    noisy = next(s for s in summary if s['system'] == 'noisy')
    candidates = [s for s in summary if s['system'] != 'noisy']
    passing = [s for s in candidates if s['stoi'] >= noisy['stoi']-STOI_GUARD]
    best = max(passing or candidates, key=lambda s: s['pesq'])
    choice = dict(alpha=float(best['system'].split(':')[1]), passes_stoi_guard=bool(passing),
                  rule=f'max mean PESQ with mean STOI >= noisy - {STOI_GUARD}', grid=ALPHA_GRID,
                  tuning_pairs=len(names), selected=best, noisy=noisy)
    (out/'tuned_alpha.json').write_text(json.dumps(choice, indent=2)+'\n')
    for s in summary:
        print(f"{s['system']:10s} PESQ {s['pesq']:.3f}  STOI {s['stoi']:.4f}  SSNR {s['ssnr']:6.2f}  SI-SDR {s['si_sdr']:6.2f}")
    print('selected', json.dumps({k: choice[k] for k in ['alpha', 'passes_stoi_guard']}))


def evaluate(args):
    with (CACHE/'test_index.csv').open() as f:
        index = {r['name']: r for r in csv.DictReader(f)}
    rows = run('test', sorted(index), args.systems, args.workers)
    for r in rows:
        r['noise'], r['snr_db'] = index[r['name']]['noise'], index[r['name']]['snr_db']
    out = ROOT/'reports'/'learned_alpha'
    write_csv(out/f'{args.tag}_per_file.csv', rows)
    write_csv(out/f'{args.tag}_summary.csv', summarize(rows))
    write_csv(out/f'{args.tag}_by_condition.csv', summarize(rows, ('system', 'noise', 'snr_db')))
    for s in summarize(rows):
        print(f"{s['system']:28s} PESQ {s['pesq']:.3f}  STOI {s['stoi']:.4f}  SSNR {s['ssnr']:6.2f}  SI-SDR {s['si_sdr']:6.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('tune-alpha')
    t.add_argument('--workers', type=int, default=8)
    e = sub.add_parser('evaluate')
    e.add_argument('--systems', nargs='+', required=True)
    e.add_argument('--tag', default='test')
    e.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    tune_alpha(args) if args.command == 'tune-alpha' else evaluate(args)
