# Learned-α Wiener filter: study report

EECS 351 Group 11 · Fall 2026

## Summary

We compared an FIR band-pass filter, the standard decision-directed (DD) Wiener filter, and
our modification of it in which a small recurrent network chooses the Wiener filter's
smoothing parameter α for every time–frequency bin. On the 824-pair VoiceBank–DEMAND test
set, the learned α gives the best PESQ (2.28, noisy input 1.97) and is the only Wiener
variant that does not lower STOI. Its main limitation is background speech: like every
energy-based method here, it cannot separate the target voice from other talkers.

## Methods

| System | Description | How its setting was chosen |
|---|---|---|
| Noisy input | Unprocessed | — |
| FIR | Band-pass 50–7,000 Hz, order 512, Hamming window, linear phase, delay compensated | ITU-T G.722 wideband speech band; nothing tuned |
| Wiener, α = 0.98 | DD a priori SNR, G = ξ/(1+ξ), ξ ≥ −25 dB | Literature default (Ephraim & Malah 1984; Scalart & Vieira Filho 1996) |
| Wiener, α = 0.9 | Same, one global α | Best of a 9-value grid on the 400 tuning pairs |
| Wiener, learned α (ours) | Same filter; α(k, m) from a 2-layer GRU (128 units, 0.3 M parameters) | Best epoch by validation PESQ |

All Wiener variants share the STFT (25 ms periodic Hann window, 10 ms hop, 512-point FFT,
16 kHz), the noise estimate (mean periodogram of the quietest 10% of frames of each
recording), the SNR floor (−25 dB), the gain function and the noisy phase.

**Learned α.** Inputs per frame: log noisy power and log a posteriori SNR (514 features).
The output layer starts at zero with α = 0.98 everywhere, so training starts from the
standard filter. Target: the ideal Wiener gain |S|²/(|S|²+|V|²) from each clean/noisy
pair; the mean-squared gain error is back-propagated through the DD recursion. Training:
10,802 pairs from 26 speakers, 20 epochs, Adam (lr 10⁻³), batch 16, CPU. Validation: 770
pairs from speakers p226 and p287; epoch 18 had the best validation PESQ (1.888).

**Fixed selection rules.** Tuned α: highest mean PESQ among values whose mean STOI is at most
0.005 below the noisy input. Learned α: highest validation PESQ. Both were fixed before the
test set was scored, and the test set was scored once.

## Data and metrics

VoiceBank–DEMAND (Valentini-Botinhao, University of Edinburgh, 2017), resampled from 48 kHz
to 16 kHz with `scipy.signal.resample_poly`. Training: 11,572 pairs, 28 speakers, 10 noise
types. Test: 824 pairs, 2 unseen speakers, 5 unseen noise types (bus, café, living room,
office, public square) at 2.5–17.5 dB SNR.

Metrics: wideband PESQ (ITU-T P.862.2), classic STOI (pystoi 0.4.1), segmental SNR (30 ms
frames, clipped to [−10, 35] dB) and SI-SDR. Every system is scored on samples
[256, n − 256), the interval the order-512 FIR filter fully supports. Our noisy-input scores
(PESQ 1.974, STOI 0.925) match the values commonly reported for this test set.

## Results (824 test pairs)

| System | PESQ | STOI | SSNR (dB) | SI-SDR (dB) |
|---|---:|---:|---:|---:|
| Noisy input | 1.974 | 0.925 | 1.79 | 8.51 |
| FIR | 2.045 | 0.925 | 6.37 | **15.46** |
| Wiener, α = 0.98 | 2.159 | 0.904 | 6.13 | 12.27 |
| Wiener, α = 0.9 | 2.251 | 0.918 | 5.00 | 11.19 |
| **Wiener, learned α** | **2.280** | **0.927** | **8.04** | 14.72 |

Paired differences over files, with 95% bootstrap intervals:

- Learned α vs α = 0.98: PESQ +0.12 [+0.10, +0.14], STOI +0.023; better PESQ on 73% of files and better STOI on 94%.
- Learned α vs α = 0.9: PESQ +0.03 [+0.01, +0.04], STOI +0.009, SSNR +3.0 dB. The PESQ gain is small; the
  single α buys quality by giving up noise removal.
- By SNR, the gain is largest at 2.5 dB (PESQ 1.70 vs 1.62 for α = 0.9); at 12.5–17.5 dB the two are similar.
- By noise type, the learned α leads on café, living room, office and public square; on bus the tuned α is
  slightly better on PESQ (2.678 vs 2.652).

The intervals describe this test set. With two test speakers they do not show how the
results generalize to other speakers.

**What the network learned.** On the three demo clips, α is high in speech pauses and low
where speech starts (café clip: about 0.95 in pauses and 0.71 at word onsets). This is the
behaviour a single α cannot provide: smooth in pauses, fast at onsets.

## Limitations

**Background speech.** We measured the noise left in speech pauses for each training noise
type on the 770 validation recordings (our method, relative to the noisy input):

| Noise | Removed in pauses | Type |
|---|---:|---|
| station, ssn, traffic | 24–27 dB | steady |
| car, kitchen, metro | 9–17 dB | changing |
| cafeteria, restaurant | about 14 dB | background talkers |
| babble, meeting | 6–8 dB | background talkers |

Energy-based methods, which include every method in this study, decide what to keep from
how much power a time–frequency bin has relative to an estimated noise level. Background
talkers have the same harmonic structure and timing as the target voice, so these methods
cannot tell them apart. The café test recordings are of this type.

**Changing noise.** The noise estimate is computed once per recording, so noise that changes
over time (metro, car, kitchen) is only partly removed. A minimum-statistics tracker
(Martin, 2001; per-bin minimum of smoothed power over 1.5 s, bias 1.5) made the pauses of
these noises 3–7 dB cleaner at inference, but lowered overall validation PESQ from 1.867 to
1.788 because the network was trained with the fixed estimate. It did not help background
talkers.

**Other limits.** Two test speakers; objective metrics only, no listening test; all methods
keep the noisy phase. Raising the noise estimate by 2× makes pauses about 7 dB cleaner
(validation PESQ 1.888 → 1.842); the reported results use the unscaled estimate.

## Future work

- **Learn the noise estimate** N, so it follows changing noise over time (for example as in
  DeepMMSE, Zhang et al. 2020), and retrain the α network with it.
- **Handle background speech**: separate the target voice from other talkers, which needs
  information beyond energy, such as a learned speech model or several microphones.
- Run a listening test with real listeners.

## Reproduce

```sh
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt pesq soundfile torch
./code/run_learned_alpha.sh                      # data cache, α tuning, training, test scoring
.venv/bin/python code/sweep_suppression.py       # suppression settings (validation)
.venv/bin/python code/noise_type_analysis.py     # per-noise-type analysis (validation)
.venv/bin/python code/make_slide_figures.py      # presentation figures
```

The VoiceBank–DEMAND archives go in `data/archives/` (see `code/prepare_corpus.py`). Code:
`code/dsp.py` (NumPy FIR and DD Wiener, checked against the MATLAB versions to about 1e-15),
`code/learned_alpha.py`, `code/run_study.py`, `code/quality_metrics.py`.

## Files in this folder

| File | Contents |
|---|---|
| `test_summary.csv`, `test_by_condition.csv`, `test_per_file.csv` | Final test results |
| `tuning_alpha_summary.csv`, `tuning_alpha_per_file.csv`, `tuned_alpha.json` | α tuning on 400 pairs |
| `suppression_sweep_validation*.json` | SNR floor and noise over-estimation sweep |
| `noise_type_analysis_validation.json` | Residual noise by noise type, fixed vs tracked noise estimate |
| `sanity_noisy_*.csv` | Noisy-input scores used to check the metric pipeline |
