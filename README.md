# EECS 351 — Speech Denoising

Speech Denoising Using Time–Frequency Filtering  
University of Michigan · Fall 2026

Members: Dayong Fang, Leyu Jiang, Jingyu Liu, Siqi Gong.

## Local experiment setup (step 1)

Requirements: MATLAB R2024a or later with Signal Processing Toolbox, Python
3.10–3.13, and `curl`. Audio Toolbox is not required: every machine uses the
same pinned classic-STOI implementation, [`pystoi==0.4.1`](https://github.com/mpariente/pystoi), via MAT-file interchange.
The MATLAB/Python bridge uses a separate process and preserves double samples.

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3 code/prepare_data.py
```

On Windows, use `.venv\Scripts\python.exe` for the environment's Python.
The downloader reads the committed manifests, validates their condition/speaker
balance and official log labels, and retrieves only the selected original 48 kHz
WAVs using HTTP ranges. It checks every WAV against the source ZIP's size and CRC,
including existing files on rerun, and records SHA-256 hashes. Interrupted runs
can be resumed with the same command. Data and environment files stay local.

Then in MATLAB, from the repository root:

```matlab
addpath('code');
report = check_project_setup;
```

The command must finish with **PASS: 643/643 pairs validated**. It checks all
400 tuning, 240 evaluation, and 3 exploratory pairs for matched mono 48 kHz
audio, length, finite samples, and matched 16 kHz resampling. It also validates
the logs and manifests and tests STOI on an exploratory clip. Missing/corrupt
files, invalid logs, and unavailable dependencies cause a nonzero batch exit.
No final-evaluation intelligibility scores are calculated during setup.

Local evidence is written to:

- `outputs/setup/download_inventory.json`: source URLs, ZIP CRCs, SHA-256 hashes,
  manifest hashes, and download status for 1,286 original WAVs.
- `outputs/setup/validation_report.json`: MATLAB version, dependency/metric
  checks, validated pair count, and all setup errors.
- `outputs/setup/validated_pairs.csv`: per-pair sample rates and sample counts.

Use the shared loader in subsequent algorithms:

```matlab
cfg = project_config;
file = cfg.manifests.tuning.File(1);
[s, y, fs, meta] = load_speech_pair(file, 'tuning', cfg);
% y is the only denoiser input; s is for scoring/diagnostics.
% After producing and aligning output:
% score = project_stoi(output, s, fs, cfg);
```

The loader resamples clean and noisy columns together with MATLAB `resample`;
it does not normalize, fit gains, or crop the pair. The fixed experiment settings
are 16 kHz, 400-sample Hann windows, 160-sample hop, and a 512-point FFT.
`project_stoi` uses `(processed, reference, fs)` argument order, classic STOI,
and rejects unusable signals rather than accepting insufficient-speech sentinels.
It starts one Python process per call; later batch evaluation can use the Python
backend directly to reduce startup overhead.

For an existing data directory, offline ZIP archives, and attribution, see
[`data/README.md`](data/README.md). The local MATLAB executable on the setup
machine is `/Applications/MATLAB_R2024b.app/bin/matlab`; if `matlab` is on PATH,
the same validation can run unattended:

```sh
matlab -batch "addpath('code'); check_project_setup;"
```

Setup regression checks:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
matlab -batch "addpath('code'); r=runtests('tests/test_data_setup.m'); assertSuccess(r);"
```

## Denoising implementations (step 2)

The three baselines share one interface and accept only the noisy signal:

```matlab
addpath('code');
cfg = project_config;
[s,y,fs] = load_speech_pair('p257_004','exploratory',cfg);
[filtered,info] = denoise_speech(y,fs,'fir',struct('cutoffHz',200));
subtracted = denoise_speech(y,fs,'spectral_subtraction',struct('alpha',2,'beta',0.01));
wiener = denoise_speech(y,fs,'wiener',struct('gainFloor',0.05));
run_denoising_demo;
```

| Method | Candidate defaults / behavior |
| --- | --- |
| `identity` | Original noisy samples, unchanged |
| `fir` | Order 128 Hamming-window high-pass at 200 Hz; optional `upperCutoffHz` for band-pass filtering |
| `spectral_subtraction` | Power subtraction with `alpha=1`, noise-power floor `beta=0.01`, and original noisy phase |
| `wiener` | Gain from clipped speech-power estimate, with amplitude `gainFloor=0.05` |

Spectral methods use 25 ms periodic Hann windows, 10 ms hops, and a 512-point
FFT at 16 kHz. A fixed noise spectrum is estimated from the lowest-energy 20%
of full noisy frames (`noiseFraction=0.2`). Padded boundary frames are excluded
from that estimate. This is an **offline, untuned** initializer; pause-based
updates are available as described below. Low-energy speech can
contaminate the estimate, so these defaults do not establish performance.

All methods return the same length without normalization. FIR delay is already
compensated; `info.validSamples` excludes its boundary samples that depend on
zero extension (default `[65, N-64]`). Use the intersection of valid intervals
for every signal in later scoring; do not shift the FIR output again. Spectral
reconstruction pads both ends, restores conjugate symmetry, and normalizes
overlap-add by the actual squared-window weights, preserving boundary samples
when gains are unity. Spectral inputs need at least one full analysis window.

`run_denoising_demo` writes three exploratory clip folders under
`outputs/denoising_demo/`. Each has `clean.wav`, `identity.wav` (noisy),
`fir.wav`, `spectral_subtraction.wav`, `wiener.wav`,
`spectral_subtraction_updated.wav`, `wiener_updated.wav`, and `signals.mat`.
The MAT file retains unscaled double outputs; the seven 24-bit WAVs share one
export gain per utterance to prevent clipping without independent normalization.
`demo_manifest.json` records parameters, gain, and alignment metadata. Running
the demo again replaces its generated files. It does not run the held-out set.

Verify reconstruction, FIR alignment/frequency response, silence, and synthetic
stationary-noise suppression with:

```sh
matlab -batch "addpath('code'); r=runtests('tests/test_denoisers.m'); assertSuccess(r);"
```

## Fixed versus pause-updated noise estimates (step 3)

Both spectral methods accept `noiseMode='fixed'` (the unchanged default) or
`noiseMode='pause_updated'`. They share the same starting noise spectrum and
filter parameters, so their difference isolates the noise-update policy.

```matlab
[fixed, fixedInfo] = denoise_speech(y,fs,'wiener');
[updated, updatedInfo] = denoise_speech(y,fs,'wiener', ...
    struct('noiseMode','pause_updated','noiseLambda',0.9));
run_denoising_demo;
```

For each recording, let E20 be the largest energy among the lowest-energy 20%
of original full frames (controlled by `noiseFraction`). A candidate pause
frame has energy at most `E20 * 10^(pauseThresholdDb/10)`; the default margin
is 3 dB. Only full, unpadded frames can qualify. Updates start on the third
consecutive qualifying frame (`minPauseFrames=3`); the first two frames are
not updated retroactively. A nonqualifying frame resets that counter.

During confirmed candidate pauses, the noise spectrum is updated before applying
that frame's gain: `Pnoise = lambda*Pnoise + (1-lambda)*Pobserved`, with default
`noiseLambda=0.9`. Otherwise the state is held, including at padded boundaries.
The update is applied to the observed noisy spectrum, never a clean signal or
reference-derived residual. The detector and initializer inspect the whole
noisy recording, so this remains an offline experiment, not a streaming VAD.

As a conservative fallback, updates are disabled when the 90th-percentile
frame energy is less than 3 dB above E20 (`minEnergyContrastDb=3`). If no
candidate run is long enough, the initial fixed estimate is also retained.
`info.noiseTracking` records frame starts, energies, candidate and update masks,
threshold, energy contrast, final spectrum, and fallback reason. Set
`storeNoiseHistory=true` to retain the per-frame noise spectrum for diagnostics;
this is off by default to keep metadata small. `info.noisePower` continues to
mean the initial spectrum for compatibility.

The demo compares fixed and updated variants on the three exploratory clips
using matched export gain. `outputs/denoising_demo/noise_update_summary.csv`
records candidate/update counts and fallback reasons for all 12 spectral runs.
The existing `spectral_subtraction.wav` and `wiener.wav` names denote fixed
estimates; files ending in `_updated.wav` use pause updates.

These are candidate settings to tune on the 400-pair tuning manifest, not
measured optimal settings. Low-energy speech may be mistaken for noise; a
fixed energy threshold may miss pauses after the background grows louder.
Neither this gate nor its fallback proves that a frame contains no speech.
Final performance comparisons remain on the held-out set after tuning.

```sh
matlab -batch "addpath('code'); r=runtests('tests/test_noise_updates.m'); assertSuccess(r);"
```

## Verified scoring and frozen tuning (steps 4–5)

Scoring uses the intersection of every candidate's valid interval. For this
search, all FIRs have order 128, so **every signal and baseline is scored on
samples 65 through N−64**. FIR outputs have already been delay-compensated;
there is no second shift, correlation-based alignment, gain fit, or loudness
normalization. SNR includes residual noise and speech distortion. Classic
STOI uses `pystoi==0.4.1`; results are arithmetic means of per-utterance metrics.
`speech_metrics.py` rejects invalid intervals, mismatched shapes, silent
references, and unusable STOI inputs. Failed cases cannot be silently dropped
before selecting a configuration.

The predeclared [`config/tuning_candidates.json`](config/tuning_candidates.json)
contains 41 candidates: identity, six FIRs (100/200/300 Hz lower cutoff, with
or without a 7 kHz upper cutoff), 20 power-subtraction and 14 Wiener variants.
The spectral search covers the proposal's strengths/floors plus one-factor
variations around the default noise fraction, update speed, and pause threshold.
It is a bounded search, not an exhaustive Cartesian parameter grid.

Run the complete search from the repository root:

```sh
.venv/bin/python code/run_tuning.py --workers 4
# Resume an existing run only when inputs and scoring code are unchanged:
.venv/bin/python code/run_tuning.py --workers 4 --resume
```

The runner verifies the training audio hashes, runs metric calibration, calls
MATLAB once to generate unscaled outputs, and batch-scores with persistent
Python workers. All 400 tuning pairs are used for every candidate; no held-out
audio is loaded. Select by descending mean STOI, then mean ΔSNR, then ID for
deterministic ties. Intermediate MAT bundles are deleted after successful
scoring; `--keep-cache` retains them. Protocol/signature checks protect resume
from changes to source, packages, candidates, manifests, or input audio.

The completed run has **16,400 measurements**, no omitted pairs, and these
primary training selections:

| Method | Frozen candidate settings | Mean STOI | Mean ΔSNR (dB) |
| --- | --- | ---: | ---: |
| Noisy baseline | Identity | 0.838221 | 0.000 |
| FIR | Order 128, 100–7000 Hz band-pass | 0.838170 | +2.308 |
| Power subtraction | Fixed, lowest 10% frames, α=1, β=0.01 | 0.838401 | +1.105 |
| Wiener | Fixed, lowest 10% frames, gain floor 0.05 | 0.838369 | +1.679 |

STOI differences are very small; they do not establish significant improvements
or held-out performance. Pause-updated family winners scored below the fixed
family winners on this tuning set. Their separately selected configurations
are retained for comparison, alongside matched mode comparisons that change
only the noise-update policy. The six family winners and provenance are saved
in [`config/frozen_denoisers.json`](config/frozen_denoisers.json); all 41 aggregate
rows are in [`reports/tuning_summary.csv`](reports/tuning_summary.csv).

Apply the frozen primary selections explicitly; prototype defaults are unchanged:

```matlab
addpath('code');
[output,info] = denoise_frozen(y,16000,'wiener');
% info.comparisonSamples is the common scoring interval.
run_frozen_demo;
```

`denoise_frozen` checks the denoiser's SHA-256 against the recorded source and
rejects changed implementations or a different sample rate. The freeze records
the complete scoring source hashes, package versions, candidate/manifest/audio
digests, selection rule, and metric-table hashes. Reproduce a new tuning run
in a new `--output` directory and review it before replacing a published freeze.

Local detailed evidence lives in `outputs/tuning/`: protocol, 400 score
checkpoints, `per_file_metrics.csv`, candidate summary, and run status. The
`listening/` directory contains three predetermined training cells (babble/0,
traffic/5, kitchen/15 dB), matched-level audio, unscaled MAT outputs, and a
common-scale spectrogram comparison. Frozen outputs were independently rescored
against their original tuning rows. These exports support later listening
review; subjective listening results are not claimed here.

Calibration and freeze regression checks:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
matlab -batch "addpath('code'); r=runtests('tests'); assertSuccess(r);"
```

## Website

- `index.html` — project overview and report links
- `proposal.html` — report text, equations, tables, and figures corresponding to the report PDF
- `progress.html` — November progress report with data, three diagnostic plots, current challenges, a three-week plan, and a technical takeaway
- `presentation.html` — project presentation slides with an in-browser denoising demo (arrow keys to navigate, `F` fullscreen, `N` speaker notes, `T` 10-minute timer)
- `assets/presentation/` — slide styles and scripts; `dsp.js` is a JavaScript port of `code/denoise_speech.m`, and `clips.js` holds the three exploratory pairs at 16 kHz (regenerate with `python3 code/build_presentation_clips.py`; needs NumPy and SciPy)
- `methods.html` — Technical Details; reserved for future content; hidden from navigation
- `results.html` — Results; reserved for future content; hidden from navigation
- `references.html` — project references
- `assets/style.css` — shared styling
- `assets/progress.css` — layout and figure styles for the progress report
- `assets/figures/` — three initial analysis figures and three progress figures in `progress/`
- `assets/audio/` — unchanged clean and noisy WAVs for the three exploratory examples
- `data/` — fixed tuning and evaluation manifests, official download links, and data setup instructions
- `reports/project_topic_finalized_revised.pdf` — project report, including MATLAB appendix
- `code/analyze_project_revised.m` — initial analysis adapted from the PDF appendix;
  uses the shared loader and frozen manifests and writes to `outputs/initial_analysis/`
- `code/generate_progress_figures.m` — reproducible MATLAB analysis for the three progress plots
- `code/prepare_data.py` — resumable official-data download and ZIP integrity checks
- `code/project_config.m`, `code/load_speech_pair.m` — shared paths and paired resampling
- `code/check_project_setup.m` — full data/environment validation
- `code/project_stoi.m`, `code/score_stoi.py` — pinned classic-STOI interface
- `code/denoise_speech.m` — identity, FIR, power subtraction, and Wiener baselines
- `code/run_denoising_demo.m` — matched-gain exploratory audio comparisons
- `code/speech_metrics.py` — common-interval, amplitude-sensitive SNR/STOI scoring
- `code/run_tuning.py`, `code/prepare_tuning.m` — complete training-only parameter search
- `code/denoise_frozen.m`, `code/run_frozen_demo.m` — apply and inspect frozen primary selections

The archived proposal/PDF documents the original selection procedure. The live
analysis script now **reads** the committed 400/240 manifests and never replaces
them or writes newly selected manifests under `code/`.

## Preview and editing

Open `index.html` in a browser. Edit the HTML files directly and refresh to see changes. No installation or build step is needed.

## Hosting

This repository is public and owned by the eecs351-speech-denoising-team organization. GitHub Pages publishes the website publicly from `main / (root)`.

Website: https://eecs351-speech-denoising-team.github.io/eecs351-speech-denoising-website/

Progress report: https://eecs351-speech-denoising-team.github.io/eecs351-speech-denoising-website/progress.html

Proposal: https://eecs351-speech-denoising-team.github.io/eecs351-speech-denoising-website/proposal.html
