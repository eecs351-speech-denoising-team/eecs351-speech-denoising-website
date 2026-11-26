function [s, y, fs, meta] = load_speech_pair(file, split, cfg)
% Read one frozen-manifest pair and identically resample 48 -> 16 kHz.
% No normalization, gain fitting, trimming, or per-signal scaling is applied.
% Example: cfg=project_config; [s,y,fs]=load_speech_pair('p257_004','exploratory',cfg);
if nargin < 3, cfg = project_config; end
split = char(split); file = char(file);
assert(isfield(cfg.manifests, split), 'EECS351:Split', 'Unknown split: %s', split);
assert(~isempty(regexp(file, '^p\d{3}_\d{3}$', 'once')), ...
    'EECS351:File', 'Expected a filename stem such as p257_004.');
manifest = cfg.manifests.(split);
row = manifest(manifest.File==string(file),:);
assert(height(row)==1, 'EECS351:File', '%s is not in the %s manifest.', file, split);
if strcmp(split,'tuning'), suffix='trainset_28spk_wav'; else, suffix='testset_wav'; end
cleanPath = fullfile(cfg.dataRoot, ['clean_' suffix], [file '.wav']);
noisyPath = fullfile(cfg.dataRoot, ['noisy_' suffix], [file '.wav']);
assert(isfile(cleanPath) && isfile(noisyPath), 'EECS351:MissingAudio', ...
    'Missing pair %s. Run python3 code/prepare_data.py. Expected:\n%s\n%s', ...
    file, cleanPath, noisyPath);
[s0, fsClean] = audioread(cleanPath);
[y0, fsNoisy] = audioread(noisyPath);
assert(fsClean==cfg.sourceFs && fsNoisy==fsClean, ...
    'EECS351:SampleRate', '%s: expected matching 48 kHz recordings.', file);
assert(iscolumn(s0) && isequal(size(s0),size(y0)) && ~isempty(s0), ...
    'EECS351:Shape', '%s: expected equal-length nonempty mono recordings.', file);
assert(all(isfinite(s0)) && all(isfinite(y0)), ...
    'EECS351:Samples', '%s: audio contains NaN or Inf.', file);
pair = resample([s0 y0], 1, 3);
s = pair(:,1); y = pair(:,2); fs = cfg.fs;
assert(size(pair,1)==ceil(numel(s0)/3) && all(isfinite(pair),'all'), ...
    'EECS351:Resample', '%s: unexpected resampling result.', file);
meta = struct('file',file,'split',split,'noise',char(row.Noise), ...
    'nominalSNR',row.NominalSNR,'sourceFs',fsClean,'sourceSamples',numel(s0), ...
    'samples',numel(s),'cleanPath',cleanPath,'noisyPath',noisyPath);
end
