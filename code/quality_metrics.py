"""Speech-quality metrics for the learned-alpha study.

All take (reference, processed) at 16 kHz on the same samples; no alignment,
gain fitting or normalization (SI-SDR is scale-invariant by definition).
  pesq_wb  ITU-T P.862.2 wideband PESQ (python 'pesq' package), MOS-LQO
  stoi     classic STOI via the project's pinned pystoi backend (score_stoi.py)
  ssnr     segmental SNR: 30 ms Hann frames, 75% overlap, frames clipped to
           [-10, 35] dB (the convention of Hu & Loizou's composite measures)
  si_sdr   scale-invariant signal-to-distortion ratio (Le Roux et al. 2019)
"""
import numpy as np
from pesq import pesq
from score_stoi import score as stoi_score

FS = 16000


def pesq_wb(reference, processed, fs=FS):
    return float(pesq(fs, np.asarray(reference, float), np.asarray(processed, float), 'wb'))


def stoi(reference, processed, fs=FS):
    # NumPy 2.x on Apple Accelerate raises spurious divide/overflow flags in matmul even
    # for finite inputs; score_stoi turns warnings into errors, so silence NumPy's FP
    # flags only. pystoi's own insufficient-speech warning is still raised as an error.
    with np.errstate(all='ignore'):
        return float(stoi_score(np.asarray(processed, float), np.asarray(reference, float), fs)['score'])


def ssnr(reference, processed, fs=FS):
    s, x = np.asarray(reference, float), np.asarray(processed, float)
    L = round(0.03*fs)
    hop = L//4
    w = np.hanning(L)
    starts = np.arange(0, len(s)-L+1, hop)
    idx = starts[:, None] + np.arange(L)[None, :]
    signal = np.sum((w*s[idx])**2, axis=1)
    noise = np.sum((w*(s[idx]-x[idx]))**2, axis=1)
    eps = np.finfo(float).eps
    seg = 10*np.log10(signal/(noise+eps)+eps)
    return float(np.mean(np.clip(seg, -10, 35)))


def si_sdr(reference, processed):
    s = np.asarray(reference, float)
    x = np.asarray(processed, float)
    s, x = s-s.mean(), x-x.mean()
    target = (np.dot(x, s)/np.dot(s, s))*s
    error = x-target
    return float(10*np.log10(np.dot(target, target)/np.dot(error, error)))


def all_metrics(reference, processed, fs=FS):
    return dict(pesq=pesq_wb(reference, processed, fs), stoi=stoi(reference, processed, fs),
                ssnr=ssnr(reference, processed, fs), si_sdr=si_sdr(reference, processed))
