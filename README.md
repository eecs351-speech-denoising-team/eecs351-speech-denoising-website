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
from that estimate. This is an **offline, untuned** estimator; pause-based
updates and parameter selection remain later steps. Low-energy speech can
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
`fir.wav`, `spectral_subtraction.wav`, `wiener.wav`, and `signals.mat`.
The MAT file retains unscaled double outputs; the five 24-bit WAVs share one
export gain per utterance to prevent clipping without independent normalization.
`demo_manifest.json` records parameters, gain, and alignment metadata. Running
the demo again replaces its generated files. It does not run the held-out set.

Verify reconstruction, FIR alignment/frequency response, silence, and synthetic
stationary-noise suppression with:

```sh
matlab -batch "addpath('code'); r=runtests('tests/test_denoisers.m'); assertSuccess(r);"
```

## Website

- `index.html` — project overview and report links
- `proposal.html` — report text, equations, tables, and figures corresponding to the report PDF
- `progress.html` — November progress report with data, three diagnostic plots, current challenges, a three-week plan, and a technical takeaway
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
