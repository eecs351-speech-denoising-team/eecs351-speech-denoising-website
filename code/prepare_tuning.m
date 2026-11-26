function prepare_tuning(outputRoot, limit)
% Generate unscaled tuning outputs for batch Python scoring. No test audio read.
% run_tuning.py writes protocol.json before invoking this function.
cfg=project_config;
if nargin<1 || isempty(outputRoot), outputRoot=fullfile(cfg.outputRoot,'tuning'); end
manifest=cfg.manifests.tuning;
if nargin<2, limit=height(manifest); end
validateattributes(limit,{'numeric'},{'scalar','integer','positive','<=',height(manifest)});
plan=jsondecode(fileread(fullfile(outputRoot,'candidates.json')));
protocol=jsondecode(fileread(fullfile(outputRoot,'protocol.json')));
candidates=plan.candidates;
folder=fullfile(outputRoot,'bundles');
if ~isfolder(folder), mkdir(folder); end
for k=1:limit
    file=char(manifest.File(k));
    target=fullfile(folder,[file '.mat']);
    scored=fullfile(outputRoot,'scores',[file '.json']);
    if isfile(scored)
        cached=jsondecode(fileread(scored));
        if strcmp(cached.signature,protocol.signature), continue; end
    end
    if isfile(target)
        cached=load(target,'signature');
        if strcmp(cached.signature,protocol.signature), continue; end
    end
    [reference,noisy,fs]=load_speech_pair(file,'tuning',cfg);
    processed=zeros(numel(noisy),numel(candidates));
    validRanges=zeros(numel(candidates),2);
    candidateIds=cell(1,numel(candidates));
    updateCounts=zeros(1,numel(candidates));
    fallbackReasons=cell(1,numel(candidates));
    for j=1:numel(candidates)
        c=candidates(j);
        [processed(:,j),info]=denoise_speech(noisy,fs,c.method,c.options);
        validRanges(j,:)=info.validSamples;
        candidateIds{j}=c.id;
        fallbackReasons{j}='not-applicable';
        if isfield(info,'noiseTracking')
            updateCounts(j)=info.noiseTracking.updateCount;
            fallbackReasons{j}=info.noiseTracking.fallbackReason;
        end
    end
    signature=protocol.signature;
    noise=char(manifest.Noise(k)); nominalSNR=manifest.NominalSNR(k);
    split='tuning'; matlabVersion=version;
    temporary=[target '.part'];
    save(temporary,'reference','noisy','processed','fs','validRanges','candidateIds', ...
        'updateCounts','fallbackReasons','signature','noise','nominalSNR','file', ...
        'split','matlabVersion','-v6');
    movefile(temporary,target,'f');
    if mod(k,20)==0 || k==limit, fprintf('Prepared tuning outputs: %d/%d\n',k,limit); end
end
end
