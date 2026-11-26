#!/usr/bin/env python3
"""Classic STOI backend for MATLAB; no Audio Toolbox or WAV quantization needed.

Input MAT file: processed, reference (equal 1-D arrays), fs (scalar).
Output: JSON with score/backend/version. Never normalizes or aligns signals.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.io import loadmat
from pystoi import stoi


def score(processed, reference, fs):
    processed = np.asarray(processed, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if processed.shape != reference.shape or not processed.size:
        raise ValueError('STOI inputs must be nonempty and have identical lengths')
    if not np.isfinite(processed).all() or not np.isfinite(reference).all():
        raise ValueError('STOI inputs contain NaN or Inf')
    if not np.isfinite(fs) or fs <= 0 or int(fs) != fs:
        raise ValueError('STOI sample rate must be a positive integer')
    if processed.size < 0.4*fs:
        raise ValueError('STOI inputs are too short (at least 0.4 seconds required)')
    if not np.any(reference):
        raise ValueError('STOI reference is silent')
    # pystoi warns and returns a sentinel for insufficient active speech.
    # Treat this as a failed score, not a usable intelligibility measurement.
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        value = float(stoi(reference, processed, int(fs), extended=False))
    if not np.isfinite(value):
        raise ValueError('STOI returned a nonfinite value')
    return {'score': value, 'backend': 'pystoi',
            'version': importlib.metadata.version('pystoi'), 'extended': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-mat', required=True, type=Path)
    parser.add_argument('--output-json', required=True, type=Path)
    args = parser.parse_args()
    try:
        arrays = loadmat(args.input_mat)
        result = score(arrays['processed'], arrays['reference'], float(arrays['fs'].item()))
        args.output_json.write_text(json.dumps(result)+'\n')
    except (ValueError, KeyError, RuntimeWarning, OSError) as error:
        parser.exit(1, f'STOI FAILED: {error}\n')
