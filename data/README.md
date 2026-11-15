# VoiceBank–DEMAND data

The [University of Edinburgh dataset record](https://datashare.ed.ac.uk/items/6ed35425-bf14-4d2b-93a1-0a4984952757) is the source for full-resolution 48 kHz clean/noisy pairs. The website includes only the three exploratory pairs, unchanged, in [`assets/audio/`](../assets/audio/). The full archives are larger than standard GitHub file limits.

## Reproduce the dataset splits

1. Download the four WAV archives and the condition logs from the official dataset record:
   - [clean_testset_wav.zip](https://datashare.ed.ac.uk/bitstreams/dec213d3-bf57-4777-9663-c24bdce92d5e/download)
   - [noisy_testset_wav.zip](https://datashare.ed.ac.uk/bitstreams/13c1bfbf-14a6-41db-9b41-8f7310f01ad5/download)
   - [clean_trainset_28spk_wav.zip](https://datashare.ed.ac.uk/bitstreams/245452b6-6235-44b6-a6f9-e7eb19797769/download)
   - [noisy_trainset_28spk_wav.zip](https://datashare.ed.ac.uk/bitstreams/ecb5a102-bb00-46d3-8af5-40c79823b837/download)
   - [logfiles.zip](https://datashare.ed.ac.uk/bitstreams/11185dc8-9cf1-405b-b858-35bd6a04aedd/download)
2. Extract the clean and noisy WAVs into separate folders. Within each split, match by filename stem and use the labels and nominal SNRs in the official logs.
3. Use [`tuning_manifest.csv`](tuning_manifest.csv) for parameter selection (400 training pairs) and [`evaluation_manifest.csv`](evaluation_manifest.csv) after settings are fixed (240 test pairs). The exploratory clips `p257_004`, `p257_017`, and `p257_009` are not part of the evaluation manifest.

To reproduce the three progress figures, run `addpath('code'); generate_progress_figures` from the repository root in MATLAB with Signal Processing Toolbox. It reads the included sample WAVs and [exploratory manifest](exploratory_manifest.csv), so the full corpus is not required for the figures.

## Attribution

Cassia Valentini-Botinhao, *Noisy speech database for training speech enhancement algorithms and TTS models*, University of Edinburgh (2017), [doi:10.7488/ds/2117](https://doi.org/10.7488/ds/2117), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The included test WAVs are unchanged; figures were generated from resampled and analyzed copies. Speech originates from the [CSTR VCTK Corpus](https://doi.org/10.7488/ds/1994), and background noise contributions come from Joachim Thiemann, Nobutaka Ito, and Emmanuel Vincent's [DEMAND dataset](https://doi.org/10.5281/zenodo.1227121), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
