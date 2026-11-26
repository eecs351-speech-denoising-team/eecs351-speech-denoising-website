# VoiceBank–DEMAND data

The [University of Edinburgh dataset record](https://datashare.ed.ac.uk/items/6ed35425-bf14-4d2b-93a1-0a4984952757) is the source for full-resolution 48 kHz clean/noisy pairs. The website includes only the three exploratory pairs, unchanged, in [`assets/audio/`](../assets/audio/). The full archives are larger than standard GitHub file limits.

## Prepare the frozen experiment data

Run `python3 code/prepare_data.py` from the repository root. Python's standard
library and `curl` are sufficient for this command. It checks the committed
manifests, downloads the official condition logs, validates every selected
noise/SNR label, and fetches only 643 pairs (400 tuning, 240 evaluation, and
3 exploratory) from the four official 48 kHz archives. HTTP ranges avoid
downloading the rest of the corpus. Each original WAV is checked against the
archive's uncompressed size and CRC; its SHA-256 is recorded in
`outputs/setup/download_inventory.json`. No waveform is modified.

Default local layout (ignored by Git except for this README and the CSVs):

```text
data/
  tuning_manifest.csv
  evaluation_manifest.csv
  exploratory_manifest.csv
  log_readme.txt
  log_trainset_28spk.txt
  log_testset.txt
  clean_trainset_28spk_wav/  # 400 selected WAVs
  noisy_trainset_28spk_wav/  # 400 selected WAVs
  clean_testset_wav/         # 243 selected WAVs
  noisy_testset_wav/         # 243 selected WAVs
```

Interrupted downloads can be resumed with the same command. Existing WAVs are
CRC-checked before reuse; a corrupt existing file is reported with its path
and is not silently overwritten. Inspect/remove the reported file and rerun.
The inventory is marked FAIL if a rerun fails, rather than retaining a stale PASS.
The download command checks ZIP integrity; MATLAB performs the separate audio
decode, metadata, sample, and resampling checks described in the root README.

To keep data elsewhere:

```sh
python3 code/prepare_data.py --data-root /path/to/voicebank-data
```

```matlab
addpath('code');
check_project_setup('/path/to/voicebank-data');
cfg = project_config('/path/to/voicebank-data');
```

Alternatively set `EECS351_DATA_ROOT` before launching MATLAB. Manifest paths
always point to the committed repository CSVs, even when raw WAVs live elsewhere.
`python3 code/prepare_data.py --check-manifests` checks only selection/balance,
without network access or raw WAVs.

## Offline or manual downloads

If HTTP ranges are unavailable, download the five official ZIPs listed below,
then extract just the selected pairs with the same CRC/log checks:

```sh
python3 code/prepare_data.py --archive-dir /path/to/official-zips
```

The directory must contain `logfiles.zip`, `clean_trainset_28spk_wav.zip`,
`noisy_trainset_28spk_wav.zip`, `clean_testset_wav.zip`, and `noisy_testset_wav.zip`.
For an already extracted corpus, arrange it as above and run MATLAB validation;
run the downloader against the ZIPs to also obtain source-integrity evidence.

The original selection procedure is documented in the proposal. The committed
CSV manifests are the experiment inputs; do not regenerate them during setup.

1. Download the four WAV archives and the condition logs from the official dataset record:
   - [clean_testset_wav.zip](https://datashare.ed.ac.uk/bitstreams/dec213d3-bf57-4777-9663-c24bdce92d5e/download)
   - [noisy_testset_wav.zip](https://datashare.ed.ac.uk/bitstreams/13c1bfbf-14a6-41db-9b41-8f7310f01ad5/download)
   - [clean_trainset_28spk_wav.zip](https://datashare.ed.ac.uk/bitstreams/245452b6-6235-44b6-a6f9-e7eb19797769/download)
   - [noisy_trainset_28spk_wav.zip](https://datashare.ed.ac.uk/bitstreams/ecb5a102-bb00-46d3-8af5-40c79823b837/download)
   - [logfiles.zip](https://datashare.ed.ac.uk/bitstreams/11185dc8-9cf1-405b-b858-35bd6a04aedd/download)
2. Use the downloader's `--archive-dir` mode, or extract into the layout above. Within each split, match by filename stem and use the labels and nominal SNRs in the official logs.
3. Use [`tuning_manifest.csv`](tuning_manifest.csv) for parameter selection (400 training pairs) and [`evaluation_manifest.csv`](evaluation_manifest.csv) after settings are fixed (240 test pairs). The exploratory clips `p257_004`, `p257_017`, and `p257_009` are not part of the evaluation manifest.

To reproduce the three progress figures, run `addpath('code'); generate_progress_figures` from the repository root in MATLAB with Signal Processing Toolbox. It reads the included sample WAVs and [exploratory manifest](exploratory_manifest.csv), so the full corpus is not required for the figures.

## Attribution

Cassia Valentini-Botinhao, *Noisy speech database for training speech enhancement algorithms and TTS models*, University of Edinburgh (2017), [doi:10.7488/ds/2117](https://doi.org/10.7488/ds/2117), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The included test WAVs are unchanged; figures were generated from resampled and analyzed copies. Speech originates from the [CSTR VCTK Corpus](https://doi.org/10.7488/ds/1994), and background noise contributions come from Joachim Thiemann, Nobutaka Ito, and Emmanuel Vincent's [DEMAND dataset](https://doi.org/10.5281/zenodo.1227121), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
