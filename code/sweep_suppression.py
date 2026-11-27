#!/usr/bin/env python3
"""How much cleaner can the learned-alpha Wiener get? Inference-time suppression knobs.

Knobs (no retraining): xi_min_db (SNR floor) and noise_scale (noise over-estimation, N -> c*N).
Scored on held-out TRAINING speakers (p226, p287; the model's validation speakers), never the
test set. Also reports residual noise in pauses: output energy in frames where the clean
speech is silent, relative to the noisy input (dB; more negative = cleaner pauses).
  .venv/bin/python code/sweep_suppression.py --files 200
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsp
import learned_alpha
import quality_metrics as qm

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'data'/'cache16k'
XI_MIN = [-25, -30, -35, -40]
NOISE_SCALE = [1.0, 1.5, 2.0]
_MODEL = None


def enhance(y, alpha, xi_min_db, noise_scale):
    Y, layout = dsp.analysis(y)
    N = noise_scale*dsp.noise_psd(y, learned_alpha.NOISE_FRACTION)
    g = dsp.dd_gains(np.abs(Y)**2, N, alpha, xi_min_db)
    return dsp.synthesis(g*Y, layout)


def pause_residual_db(clean, noisy, out):
    L, hop = 400, 160
    starts = np.arange(0, len(clean)-L+1, hop)
    e = np.array([np.sum(clean[s:s+L]**2) for s in starts])
    quiet = starts[e < np.percentile(e, 20)]
    num = sum(np.sum(out[s:s+L]**2) for s in quiet)
    den = sum(np.sum(noisy[s:s+L]**2) for s in quiet)
    return float(10*np.log10(num/den+1e-12))


def score(job):
    name, xi_mins, scales = job
    global _MODEL
    if _MODEL is None:
        _MODEL = learned_alpha.load(learned_alpha.default_checkpoint())
    with np.load(CACHE/'train'/f'{name}.npz') as d:
        clean, noisy = d['clean'].astype(float), d['noisy'].astype(float)
    k = slice(256, len(noisy)-256)
    alpha = learned_alpha.alpha_map(_MODEL, noisy)
    rows = []
    systems = [('noisy', None), ('dd0.98', (0.98, -25, 1.0)), ('dd0.9', (0.9, -25, 1.0))]
    systems += [(f'learned xi{x} N{c:g}', ('A', x, c)) for x in xi_mins for c in scales]
    for label, cfg in systems:
        if cfg is None:
            out = noisy
        else:
            a, x, c = cfg
            out = enhance(noisy, alpha if a == 'A' else a, x, c)
        m = qm.all_metrics(clean[k], out[k])
        m['pause_db'] = pause_residual_db(clean[k], noisy[k], out[k])
        rows.append(dict(name=name, system=label, **m))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--files', type=int, default=200)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--xi-min', type=float, nargs='+', default=XI_MIN)
    parser.add_argument('--noise-scale', type=float, nargs='+', default=NOISE_SCALE)
    parser.add_argument('--tag', default='suppression_sweep_validation')
    args = parser.parse_args()
    with (CACHE/'train_index.csv').open() as f:
        names = sorted(r['name'] for r in csv.DictReader(f) if r['speaker'] in ('p226', 'p287'))
    names = random.Random(351).sample(names, args.files)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [(n, args.xi_min, args.noise_scale) for n in names]
        for k, r in enumerate(pool.map(score, jobs, chunksize=4), 1):
            rows.extend(r)
            if k % 50 == 0:
                print(f'{k}/{len(names)}', flush=True)
    out = ROOT/'reports'/'learned_alpha'
    summary = {}
    for r in rows:
        summary.setdefault(r['system'], []).append(r)
    table = [dict(system=s, **{m: float(np.mean([r[m] for r in v])) for m in ['pesq', 'stoi', 'ssnr', 'si_sdr', 'pause_db']})
             for s, v in summary.items()]
    (out/f'{args.tag}.json').write_text(json.dumps(table, indent=2))
    for t in table:
        print(f"{t['system']:24s} PESQ {t['pesq']:.3f}  STOI {t['stoi']:.4f}  SSNR {t['ssnr']:5.2f}  "
              f"SI-SDR {t['si_sdr']:5.2f}  pause residual {t['pause_db']:6.1f} dB")


if __name__ == '__main__':
    main()
