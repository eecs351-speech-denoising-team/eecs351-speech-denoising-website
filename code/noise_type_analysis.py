#!/usr/bin/env python3
"""Which noises does the learned-alpha Wiener fail to remove, and does tracking the noise help?

On held-out TRAINING speakers (p226, p287; never the test set), per training noise type:
  nonstationarity  std over time of the true noise frame energy (dB): high = bursts / babble
  pause residual   noise left in speech pauses relative to the noisy input (dB)
  burst residual   the same, but only in the loudest 10% of noise frames inside pauses
Compares the fixed noise estimate (quietest 10% of frames, whole recording) with a simple
minimum-statistics tracker (Martin, 2001): smoothed power, per-bin minimum over a 1.5 s
window, times a bias factor, so N follows the background over time.
  .venv/bin/python code/noise_type_analysis.py
"""
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import minimum_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsp
import learned_alpha
import quality_metrics as qm

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'data'/'cache16k'
_MODEL = None


def tracked_noise(P, n, smooth=0.85, window=150, bias=1.5):
    # Frames that overlap the STFT's zero padding would look like silence and drive the
    # minimum to 0; replace them with the nearest frame that lies fully inside the signal.
    first = int(np.ceil(dsp.WINDOW/dsp.HOP))
    last = (n // dsp.HOP)
    P = P.copy()
    P[:first] = P[first]
    P[last+1:] = P[last]
    S = np.empty_like(P)
    acc = P[0]
    for m in range(P.shape[0]):
        acc = smooth*acc+(1-smooth)*P[m]
        S[m] = acc
    return bias*minimum_filter1d(S, size=window, axis=0, mode='nearest')


def dd_gains_tv(P, N, alpha, xi_min_db=-25.0):
    """Decision-directed Wiener with a time-varying noise PSD N[T,F]."""
    xi_min = 10**(xi_min_db/10)
    alpha = np.broadcast_to(alpha, P.shape)
    N = np.maximum(N, 1e-14)
    g = np.empty_like(P)
    prev = np.zeros(P.shape[1])
    for m in range(P.shape[0]):
        xi = np.maximum(alpha[m]*prev/N[m]+(1-alpha[m])*np.maximum(P[m]/N[m]-1, 0), xi_min)
        g[m] = xi/(1+xi)
        prev = g[m]**2*P[m]
    return g


def frame_energy(x, L=400, hop=160):
    starts = np.arange(0, len(x)-L+1, hop)
    return starts, np.array([np.sum(x[s:s+L]**2) for s in starts])


def residuals(clean, noisy, out):
    starts, es = frame_energy(clean)
    _, ev = frame_energy(noisy-clean)
    _, ey = frame_energy(noisy)
    _, eo = frame_energy(out)
    pause = es < np.percentile(es, 20)
    burst = pause & (ev >= np.percentile(ev[pause], 90))
    db = lambda m: float(10*np.log10(eo[m].sum()/ey[m].sum()+1e-12))
    nonstat = float(np.std(10*np.log10(ev+1e-12)))
    return db(pause), db(burst), nonstat


def run(job):
    global _MODEL
    name, noise = job
    if _MODEL is None:
        _MODEL = learned_alpha.load(learned_alpha.default_checkpoint())
    with np.load(CACHE/'train'/f'{name}.npz') as d:
        clean, noisy = d['clean'].astype(float), d['noisy'].astype(float)
    Y, layout = dsp.analysis(noisy)
    P = np.abs(Y)**2
    alpha = learned_alpha.alpha_map(_MODEL, noisy)
    N_fixed = np.broadcast_to(dsp.noise_psd(noisy, learned_alpha.NOISE_FRACTION), P.shape)
    outs = {'fixed N': dsp.synthesis(dd_gains_tv(P, N_fixed, alpha)*Y, layout),
            'tracked N': dsp.synthesis(dd_gains_tv(P, tracked_noise(P, len(noisy)), alpha)*Y, layout)}
    k = slice(256, len(noisy)-256)
    rows = []
    for label, out in outs.items():
        pause, burst, nonstat = residuals(clean[k], noisy[k], out[k])
        rows.append(dict(name=name, noise=noise, system=label, pesq=qm.pesq_wb(clean[k], out[k]),
                         stoi=qm.stoi(clean[k], out[k]), pause_db=pause, burst_db=burst, nonstat_db=nonstat))
    rows.append(dict(name=name, noise=noise, system='noisy', pesq=qm.pesq_wb(clean[k], noisy[k]),
                     stoi=qm.stoi(clean[k], noisy[k]), pause_db=0.0, burst_db=0.0, nonstat_db=nonstat))
    return rows


def main():
    with (CACHE/'train_index.csv').open() as f:
        jobs = [(r['name'], r['noise']) for r in csv.DictReader(f) if r['speaker'] in ('p226', 'p287')]
    rows = []
    with ProcessPoolExecutor(8) as pool:
        for r in pool.map(run, jobs, chunksize=4):
            rows.extend(r)
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r['noise']][r['system']].append(r)
    table = []
    for noise, sys_rows in agg.items():
        m = lambda s, key: float(np.mean([r[key] for r in sys_rows[s]]))
        table.append(dict(noise=noise, files=len(sys_rows['noisy']), nonstat_db=m('noisy', 'nonstat_db'),
                          **{f'{s}_{k}': m(s, k) for s in ['noisy', 'fixed N', 'tracked N'] for k in ['pesq', 'stoi', 'pause_db', 'burst_db']}))
    table.sort(key=lambda t: t['nonstat_db'])
    (ROOT/'reports'/'learned_alpha'/'noise_type_analysis_validation.json').write_text(json.dumps(table, indent=2))
    print(f"{'noise':11s} {'nonstat':>7s} | {'fixed N: pause':>14s} {'burst':>6s} {'ΔPESQ':>6s} | {'tracked N: pause':>16s} {'burst':>6s} {'ΔPESQ':>6s} {'ΔSTOI':>7s}")
    for t in table:
        print(f"{t['noise']:11s} {t['nonstat_db']:7.1f} | {t['fixed N_pause_db']:14.1f} {t['fixed N_burst_db']:6.1f} "
              f"{t['fixed N_pesq']-t['noisy_pesq']:+6.2f} | {t['tracked N_pause_db']:16.1f} {t['tracked N_burst_db']:6.1f} "
              f"{t['tracked N_pesq']-t['noisy_pesq']:+6.2f} {t['tracked N_stoi']-t['fixed N_stoi']:+7.3f}")
    allr = lambda s, k: np.mean([r[k] for r in rows if r['system'] == s])
    print('ALL: PESQ noisy %.3f fixed %.3f tracked %.3f | STOI noisy %.4f fixed %.4f tracked %.4f' % (
        allr('noisy', 'pesq'), allr('fixed N', 'pesq'), allr('tracked N', 'pesq'),
        allr('noisy', 'stoi'), allr('fixed N', 'stoi'), allr('tracked N', 'stoi')))


if __name__ == '__main__':
    main()
