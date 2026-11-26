"""Aligned SNR/STOI scoring. No delay search, gain fitting or normalization."""
import numpy as np
from score_stoi import score


def signal(value):
    value = np.asarray(value)
    if value.ndim != 1 or not value.size or np.iscomplexobj(value) or not np.isfinite(value).all():
        raise ValueError('Expected a nonempty finite real mono vector')
    return value.astype(np.float64, copy=False)


def common_valid_interval(sample_count, ranges):
    """Return the one-based inclusive intersection, preserving MATLAB indexing."""
    ranges = np.asarray(ranges, dtype=float)
    if ranges.ndim != 2 or ranges.shape[1] != 2 or not len(ranges):
        raise ValueError('Expected one [start,end] range per method')
    if not np.isfinite(ranges).all() or not np.equal(ranges, np.floor(ranges)).all():
        raise ValueError('Valid intervals must contain finite integer sample indices')
    if np.any(ranges[:, 0] < 1) or np.any(ranges[:, 1] > sample_count) or np.any(ranges[:, 0] > ranges[:, 1]):
        raise ValueError('Invalid or empty comparison interval')
    start, end = int(ranges[:, 0].max()), int(ranges[:, 1].min())
    if start > end:
        raise ValueError('Methods have no common valid samples')
    return start, end


def aligned_signals(reference, noisy, processed, ranges):
    reference, noisy = signal(reference), signal(noisy)
    processed = np.asarray(processed)
    if processed.ndim != 2 or processed.shape[0] != len(reference) or noisy.shape != reference.shape:
        raise ValueError('All signals must have the same original sample count')
    if processed.shape[1] != len(ranges) or np.iscomplexobj(processed) or not np.isfinite(processed).all():
        raise ValueError('Invalid processed matrix or method intervals')
    interval = common_valid_interval(len(reference), ranges)
    keep = slice(interval[0]-1, interval[1])
    return reference[keep], noisy[keep], processed[keep, :], interval


def snr_db(reference, processed):
    reference, processed = signal(reference), signal(processed)
    if reference.shape != processed.shape:
        raise ValueError('SNR inputs must have equal lengths')
    power = np.dot(reference, reference)
    if power <= 0:
        raise ValueError('SNR reference has zero energy')
    error = reference-processed
    error_power = np.dot(error, error)
    return float(10*np.log10(power/error_power)) if error_power else float('inf')


def baseline_metrics(reference, noisy, fs):
    return dict(snr_in_db=snr_db(reference, noisy), stoi_in=score(noisy, reference, fs)['score'])


def enhancement_metrics(reference, processed, fs, baseline):
    output_snr = snr_db(reference, processed)
    output_stoi = score(processed, reference, fs)['score']
    return dict(baseline, snr_out_db=output_snr, delta_snr_db=output_snr-baseline['snr_in_db'],
                stoi_out=output_stoi, delta_stoi=output_stoi-baseline['stoi_in'])
