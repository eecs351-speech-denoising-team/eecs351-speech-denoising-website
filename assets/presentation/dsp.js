// In-browser port of code/denoise_speech.m for the presentation demo.
// Same framing (25 ms periodic Hann, 10 ms hop, 512-point FFT at 16 kHz),
// noise initializer, pause-update rule, gains, and overlap-add normalization.
// Indices are 0-based here; validSamples is an inclusive 0-based interval.
(function (root) {
  'use strict';

  function fft(re, im, inverse) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (inverse ? 2 : -2) * Math.PI / len;
      const wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2;
          const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
          const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
        }
      }
    }
    if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }

  function hannPeriodic(L) {
    const w = new Float64Array(L);
    for (let i = 0; i < L; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / L);
    return w;
  }

  // fir1(order, wn, 'high'|'bandpass', hamming(order+1), 'scale'). Scaling gives
  // unit gain at Nyquist (high-pass) or at the passband center (band-pass).
  function firDesign(order, wn, wnUpper) {
    const b = new Float64Array(order + 1), mid = order / 2;
    const lowpass = (c, m) => (m === 0 ? c : Math.sin(Math.PI * c * m) / (Math.PI * m));
    const center = wnUpper ? Math.PI * (wn + wnUpper) / 2 : Math.PI;
    let re = 0, im = 0;
    for (let i = 0; i <= order; i++) {
      const m = i - mid;
      const ideal = wnUpper ? lowpass(wnUpper, m) - lowpass(wn, m) : (m === 0 ? 1 : 0) - lowpass(wn, m);
      b[i] = ideal * (0.54 - 0.46 * Math.cos(2 * Math.PI * i / order));
      re += b[i] * Math.cos(center * i); im -= b[i] * Math.sin(center * i);
    }
    const g = Math.hypot(re, im);
    for (let i = 0; i <= order; i++) b[i] /= g;
    return b;
  }

  function magnitudeResponse(b, K) {
    const re = new Float64Array(K), im = new Float64Array(K);
    // Fold taps beyond K (time aliasing): the K-point DFT is still exact at its K bins.
    for (let i = 0; i < b.length; i++) re[i % K] += b[i];
    fft(re, im, false);
    const out = new Float64Array(K / 2 + 1);
    for (let k = 0; k <= K / 2; k++) out[k] = Math.hypot(re[k], im[k]);
    return out;
  }

  // Padded STFT used by the spectral denoisers: L zeros before the signal,
  // so frame k starts at original sample k*hop - L.
  function stft(y, fs, opts) {
    opts = opts || {};
    const L = opts.windowLength || Math.round(0.025 * fs);
    const hop = opts.hop || Math.round(0.010 * fs);
    const K = opts.nfft || Math.pow(2, Math.ceil(Math.log2(L)));
    const n = y.length, B = K / 2 + 1, w = hannPeriodic(L);
    const F = Math.ceil((n + L) / hop) + 1;
    const paddedLength = (F - 1) * hop + L;
    const padded = new Float64Array(paddedLength);
    padded.set(y, L);
    const re = new Float64Array(F * B), im = new Float64Array(F * B);
    const power = new Float64Array(F * B), frameEnergy = new Float64Array(F);
    const fr = new Float64Array(K), fi = new Float64Array(K);
    for (let k = 0; k < F; k++) {
      fr.fill(0); fi.fill(0);
      let e = 0;
      for (let i = 0; i < L; i++) {
        const v = padded[k * hop + i];
        e += v * v; fr[i] = v * w[i];
      }
      frameEnergy[k] = e / L;
      fft(fr, fi, false);
      for (let b = 0; b < B; b++) {
        re[k * B + b] = fr[b]; im[k * B + b] = fi[b];
        power[k * B + b] = fr[b] * fr[b] + fi[b] * fi[b];
      }
    }
    return { L, hop, K, B, F, n, fs, w, paddedLength, re, im, power, frameEnergy };
  }

  function denoise(y, fs, method, options) {
    options = options || {};
    const n = y.length;
    if (method === 'identity') {
      return { output: Float64Array.from(y), info: { method, validSamples: [0, n - 1] } };
    }
    if (method === 'fir') {
      const order = options.order || 128, cutoffHz = options.cutoffHz || 200;
      const upper = options.upperCutoffHz ? options.upperCutoffHz / (fs / 2) : 0;
      const b = firDesign(order, cutoffHz / (fs / 2), upper), delay = order / 2;
      const output = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const lo = Math.max(0, i + delay - (n - 1)), hi = Math.min(order, i + delay);
        let acc = 0;
        for (let k = lo; k <= hi; k++) acc += b[k] * y[i + delay - k];
        output[i] = acc; // delay already compensated
      }
      return { output, info: { method, coefficients: b, validSamples: [delay, n - 1 - delay],
        response: magnitudeResponse(b, 512) } };
    }

    const o = Object.assign({ noiseFraction: 0.2, noiseMode: 'fixed', noiseLambda: 0.9,
      pauseThresholdDb: 3, minPauseFrames: 3, minEnergyContrastDb: 3,
      alpha: 1, beta: 0.01, gainFloor: 0.05, ddAlpha: 0.98, xiMinDb: -25 }, options);
    // Decision-directed Wiener (code/denoise_wiener_dd.m) always uses the fixed noise estimate.
    const dd = method === 'wiener_dd';
    if (dd) o.noiseMode = 'fixed';
    const S = stft(y, fs, o);
    const { L, hop, K, B, F, w } = S;

    // Initial noise spectrum: lowest-energy unpadded frames of the noisy input.
    const count = Math.floor((n - L) / hop) + 1;
    const energy = new Float64Array(count);
    for (let k = 0; k < count; k++) {
      let e = 0;
      for (let i = 0; i < L; i++) { const v = y[k * hop + i]; e += v * v; }
      energy[k] = e / L;
    }
    const rank = Array.from(energy.keys()).sort((a, b) => energy[a] - energy[b] || a - b);
    const chosen = rank.slice(0, Math.max(1, Math.ceil(o.noiseFraction * count)));
    const noisePower = new Float64Array(B);
    const fr = new Float64Array(K), fi = new Float64Array(K);
    for (const k of chosen) {
      fr.fill(0); fi.fill(0);
      for (let i = 0; i < L; i++) fr[i] = y[k * hop + i] * w[i];
      fft(fr, fi, false);
      for (let b = 0; b < B; b++) noisePower[b] += (fr[b] * fr[b] + fi[b] * fi[b]) / chosen.length;
    }

    // Pause candidates: consecutive full frames within a margin of E20.
    const baseEnergy = energy[chosen[chosen.length - 1]];
    const threshold = baseEnergy * Math.pow(10, o.pauseThresholdDb / 10);
    const highEnergy = energy[rank[Math.max(1, Math.ceil(0.9 * count)) - 1]];
    const contrast = 10 * Math.log10((highEnergy + Number.EPSILON) / (baseEnergy + Number.EPSILON));
    const fullFrames = new Uint8Array(F), candidates = new Uint8Array(F), updates = new Uint8Array(F);
    for (let k = 0; k < F; k++) {
      const start = k * hop - L;
      fullFrames[k] = start >= 0 && start + L <= n ? 1 : 0;
      candidates[k] = fullFrames[k] && S.frameEnergy[k] <= threshold ? 1 : 0;
    }
    let reason = 'fixed-mode', updateCount = 0;
    if (o.noiseMode === 'pause_updated') {
      if (contrast < o.minEnergyContrastDb) {
        reason = 'insufficient-energy-contrast';
      } else {
        let run = 0;
        for (let k = 0; k < F; k++) {
          run = candidates[k] ? run + 1 : 0;
          if (run >= o.minPauseFrames) { updates[k] = 1; updateCount++; }
        }
        reason = updateCount ? 'none' : 'no-confirmed-pause';
      }
    }

    const gain = new Float32Array(F * B);
    const state = Float64Array.from(noisePower);
    const paddedOut = new Float64Array(S.paddedLength), weights = new Float64Array(S.paddedLength);
    const subtraction = method === 'spectral_subtraction';
    const xiMin = Math.pow(10, o.xiMinDb / 10), previous = new Float64Array(B); // |S_hat(m-1)|^2
    for (let k = 0; k < F; k++) {
      if (updates[k]) { // update before applying this frame's gain
        for (let b = 0; b < B; b++) state[b] = o.noiseLambda * state[b] + (1 - o.noiseLambda) * S.power[k * B + b];
      }
      fr.fill(0); fi.fill(0);
      for (let b = 0; b < B; b++) {
        const P = S.power[k * B + b], N = state[b];
        let re = S.re[k * B + b], im = S.im[k * B + b], g;
        if (subtraction) {
          const speech = Math.max(P - o.alpha * N, o.beta * N);
          if (P > 0) { g = Math.sqrt(speech / P); re *= g; im *= g; }
          else { g = 0; re = Math.sqrt(speech); im = 0; }
        } else if (dd) {
          const Ns = Math.max(N, Number.MIN_VALUE), gamma = P / Ns;
          const xi = Math.max(o.ddAlpha * previous[b] / Ns + (1 - o.ddAlpha) * Math.max(gamma - 1, 0), xiMin);
          g = xi / (1 + xi);
          previous[b] = g * g * P;
          re *= g; im *= g;
        } else {
          const speech = Math.max(P - N, 0), den = speech + N;
          g = Math.max(o.gainFloor, speech / (den === 0 ? 1 : den));
          re *= g; im *= g;
        }
        gain[k * B + b] = g;
        fr[b] = re; fi[b] = im;
        if (b > 0 && b < K / 2) { fr[K - b] = re; fi[K - b] = -im; }
      }
      fi[0] = 0; fi[K / 2] = 0; // real(ifft) of the conjugate-symmetric spectrum
      fft(fr, fi, true);
      for (let i = 0; i < L; i++) {
        paddedOut[k * hop + i] += fr[i] * w[i];
        weights[k * hop + i] += w[i] * w[i];
      }
    }
    const output = new Float64Array(n);
    for (let i = 0; i < n; i++) output[i] = paddedOut[L + i] / weights[L + i];
    return { output, info: { method, options: o, validSamples: [0, n - 1], gain, noisePower,
      stft: { L, hop, K, B, F },
      noiseTracking: { frameEnergy: S.frameEnergy, fullFrames, pauseEnergyThreshold: threshold,
        energyContrastDb: contrast, pauseCandidates: candidates, updateMask: updates,
        updateCount, fallbackReason: reason, finalNoisePower: state } } };
  }

  const api = { fft, stft, denoise, firDesign, magnitudeResponse, hannPeriodic };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.DSP = api;
})(typeof window !== 'undefined' ? window : globalThis);
