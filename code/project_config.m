function cfg = project_config(dataRoot)
% Shared paths and analysis settings. Manifest CSVs are never regenerated.
% Override raw data location with project_config('/path/to/data') or
% the EECS351_DATA_ROOT environment variable. Default: repository/data.
cfg.root = fileparts(fileparts(mfilename('fullpath')));
if nargin < 1 || isempty(dataRoot), dataRoot = getenv('EECS351_DATA_ROOT'); end
if isempty(dataRoot), dataRoot = fullfile(cfg.root, 'data'); end
cfg.dataRoot = char(dataRoot);
cfg.outputRoot = fullfile(cfg.root, 'outputs');
cfg.fs = 16000;
cfg.sourceFs = 48000;
cfg.windowLength = 400;
cfg.hop = 160;
cfg.nfft = 512;
if ispc
    cfg.python = fullfile(cfg.root, '.venv', 'Scripts', 'python.exe');
else
    cfg.python = fullfile(cfg.root, '.venv', 'bin', 'python');
end
splits = ["tuning", "evaluation", "exploratory"];
counts = [400 240 3];
for k = 1:numel(splits)
    split = splits(k);
    filename = fullfile(cfg.root, 'data', split+'_manifest.csv');
    table = readtable(filename, 'TextType', 'string');
    assert(isequal(table.Properties.VariableNames, {'File','Noise','NominalSNR'}), ...
        'EECS351:Manifest', 'Unexpected manifest columns: %s', filename);
    assert(height(table)==counts(k) && numel(unique(table.File))==counts(k), ...
        'EECS351:Manifest', 'Expected %d distinct entries in %s.', counts(k), filename);
    cfg.manifests.(split) = table;
end
assert(isempty(intersect(cfg.manifests.tuning.File,cfg.manifests.evaluation.File)) && ...
    isempty(intersect(cfg.manifests.tuning.File,cfg.manifests.exploratory.File)) && ...
    isempty(intersect(cfg.manifests.evaluation.File,cfg.manifests.exploratory.File)), ...
    'EECS351:Manifest', 'Manifest splits must not overlap.');
end
