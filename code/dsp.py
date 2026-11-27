"""NumPy reference implementations matching the MATLAB denoisers.

fir_bandpass   <-> denoise_speech(y, fs, 'fir', struct('order',N,'cutoffHz',lo,'upperCutoffHz',hi))
wiener_dd      <-> denoise_wiener_dd(y, fs, options)

Framing is identical: periodic Hann window, L zeros of padding before the signal,
normalized weighted overlap-add. Only the noisy signal is used.
"""
import math
import numpy as np
from scipy.signal import firwin

FS = 16000
WINDOW, HOP, NFFT = 400, 160, 512


def hann_periodic(L):
    return 0.5 - 0.5*np.cos(2*np.pi*np.arange(L)/L)


def fir_bandpass(y, order=512, low_hz=50.0, high_hz=7000.0, fs=FS):
    """Linear-phase Hamming-window band-pass, delay compensated, same length as y."""
    y = np.asarray(y, dtype=np.float64)
    b = firwin(order+1, [low_hz, high_hz], pass_zero=False, window='hamming', scale=True, fs=fs)
    delay = order//2
    return np.convolve(y, b)[delay:delay+len(y)]


def frame_layout(n, L=WINDOW, hop=HOP):
    count = math.ceil((n+L)/hop)+1
    padded_length = (count-1)*hop+L
    return count, padded_length


def analysis(y, L=WINDOW, hop=HOP, nfft=NFFT):
    """Padded STFT: frame k starts at original sample k*hop - L. Returns (spectrum[T,F], layout)."""
    n = len(y)
    count, padded_length = frame_layout(n, L, hop)
    padded = np.zeros(padded_length)
    padded[L:L+n] = y
    idx = np.arange(L)[None, :] + hop*np.arange(count)[:, None]
    w = hann_periodic(L)
    return np.fft.rfft(padded[idx]*w, nfft, axis=1), (n, L, hop, nfft, padded_length, idx, w)


def synthesis(spectrum, layout):
    n, L, hop, nfft, padded_length, idx, w = layout
    frames = np.fft.irfft(spectrum, nfft, axis=1)[:, :L]*w
    out = np.zeros(padded_length)
    weights = np.zeros(padded_length)
    np.add.at(out, idx, frames)
    np.add.at(weights, idx, np.broadcast_to(w**2, frames.shape))
    return out[L:L+n]/weights[L:L+n]


def noise_psd(y, noise_fraction=0.1, L=WINDOW, hop=HOP, nfft=NFFT):
    """Mean periodogram of the lowest-energy original full frames (as denoise_speech)."""
    y = np.asarray(y, dtype=np.float64)
    starts = np.arange(0, len(y)-L+1, hop)
    frames = y[starts[:, None] + np.arange(L)[None, :]]
    energy = np.mean(frames**2, axis=1)
    rank = np.argsort(energy, kind='stable')
    chosen = rank[:max(1, math.ceil(noise_fraction*len(rank)))]
    spec = np.fft.rfft(frames[chosen]*hann_periodic(L), nfft, axis=1)
    return np.mean(np.abs(spec)**2, axis=0)


def dd_gains(power, noise, alpha=0.98, xi_min_db=-25.0):
    """Decision-directed a priori SNR and Wiener gain. alpha: scalar or array [T,F]."""
    T, F = power.shape
    alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (T, F))
    N = np.maximum(noise, np.finfo(float).tiny)
    xi_min = 10**(xi_min_db/10)
    gains = np.empty((T, F))
    previous = np.zeros(F)
    for m in range(T):
        gamma = power[m]/N
        xi = np.maximum(alpha[m]*previous/N + (1-alpha[m])*np.maximum(gamma-1, 0), xi_min)
        g = xi/(1+xi)
        gains[m] = g
        previous = g*g*power[m]
    return gains


def wiener_dd(y, alpha=0.98, xi_min_db=-25.0, noise_fraction=0.1, return_gains=False):
    y = np.asarray(y, dtype=np.float64)
    spectrum, layout = analysis(y)
    gains = dd_gains(np.abs(spectrum)**2, noise_psd(y, noise_fraction), alpha, xi_min_db)
    out = synthesis(gains*spectrum, layout)
    return (out, gains) if return_gains else out
