#!/usr/bin/env python3
"""Decode the full VoiceBank-DEMAND archives into a 16 kHz cache for the learned-alpha study.

Reads data/archives/{clean,noisy}_{trainset_28spk,testset}_wav.zip and logfiles.zip,
pairs clean/noisy WAVs by name, resamples 48 kHz -> 16 kHz with scipy.signal.resample_poly
(the same filter for every system in this study), and writes
  data/cache16k/<split>/<name>.npz   (clean, noisy as float32)
  data/cache16k/<split>_index.csv    (name, speaker, noise, snr_db, samples)
Usage: .venv/bin/python code/prepare_corpus.py --split test|train
"""
import argparse
import csv
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import zipfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT/'data'/'archives'
CACHE = ROOT/'data'/'cache16k'
SPLITS = {'train': ('trainset_28spk', 'log_trainset_28spk.txt', 11572),
          'test': ('testset', 'log_testset.txt', 824)}


def wav_members(path):
    with zipfile.ZipFile(path) as z:
        return {Path(n).stem: n for n in z.namelist() if n.endswith('.wav') and '__MACOSX' not in n}


def convert(job):
    split, name, clean_zip, clean_member, noisy_zip, noisy_member = job
    target = CACHE/split/f'{name}.npz'
    if target.exists():
        with np.load(target) as d:
            return name, len(d['clean'])
    signals = []
    for archive, member in [(clean_zip, clean_member), (noisy_zip, noisy_member)]:
        with zipfile.ZipFile(archive) as z:
            x, fs = sf.read(io.BytesIO(z.read(member)), dtype='float64')
        if fs != 48000 or x.ndim != 1:
            raise ValueError(f'{member}: expected 48 kHz mono, got {fs} Hz shape {x.shape}')
        signals.append(x)
    if len(signals[0]) != len(signals[1]):
        raise ValueError(f'{name}: clean/noisy length mismatch')
    clean, noisy = (resample_poly(x, 1, 3).astype(np.float32) for x in signals)
    temporary = target.with_suffix('.part.npz')
    np.savez(temporary, clean=clean, noisy=noisy)
    temporary.rename(target)
    return name, len(clean)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--workers', type=int, default=8)
    split = parser.parse_args().split
    tag, log_name, expected = SPLITS[split]
    clean_zip, noisy_zip = ARCHIVES/f'clean_{tag}_wav.zip', ARCHIVES/f'noisy_{tag}_wav.zip'
    clean, noisy = wav_members(clean_zip), wav_members(noisy_zip)
    if set(clean) != set(noisy) or len(clean) != expected:
        raise SystemExit(f'{split}: expected {expected} matched pairs, got {len(clean)}/{len(noisy)}')
    with zipfile.ZipFile(ARCHIVES/'logfiles.zip') as z:
        log = {}
        for line in z.read(log_name).decode('latin-1').splitlines():
            parts = line.split()
            if len(parts) == 3:
                log[parts[0]] = (parts[1], float(parts[2]))
    if set(log) != set(clean):
        raise SystemExit(f'{split}: log entries do not match the archive')
    (CACHE/split).mkdir(parents=True, exist_ok=True)
    jobs = [(split, n, clean_zip, clean[n], noisy_zip, noisy[n]) for n in sorted(clean)]
    lengths = {}
    with ProcessPoolExecutor(parser.parse_args().workers) as pool:
        for k, (name, samples) in enumerate(pool.map(convert, jobs, chunksize=16), 1):
            lengths[name] = samples
            if k % 1000 == 0 or k == len(jobs):
                print(f'{split}: {k}/{len(jobs)}', flush=True)
    with (CACHE/f'{split}_index.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'speaker', 'noise', 'snr_db', 'samples'])
        for name in sorted(lengths):
            w.writerow([name, name.split('_')[0], log[name][0], log[name][1], lengths[name]])
    print(f'PASS: {len(lengths)} {split} pairs cached in {CACHE/split}')


if __name__ == '__main__':
    main()
