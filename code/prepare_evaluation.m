function prepare_evaluation(outputRoot)
% Apply only the predeclared frozen variants to the fixed held-out manifest.
cfg=project_config;
if nargin<1 || isempty(outputRoot), outputRoot=fullfile(cfg.outputRoot,'evaluation'); end
plan=jsondecode(fileread(fullfile(outputRoot,'plan.json')));
protocol=jsondecode(fileread(fullfile(outputRoot,'protocol.json')));
variants=plan.variants; manifest=cfg.manifests.evaluation;
folder=fullfile(outputRoot,'bundles'); if ~isfolder(folder), mkdir(folder); end
for k=1:height(manifest)
    file=char(manifest.File(k)); target=fullfile(folder,[file '.mat']);
    scored=fullfile(outputRoot,'scores',[file '.json']);
    if isfile(scored)
        cached=jsondecode(fileread(scored));
        if strcmp(cached.signature,protocol.signature), continue; end
    end
    if isfile(target)
        cached=load(target,'signature');
        if strcmp(cached.signature,protocol.signature), continue; end
    end
    [reference,noisy,fs]=load_speech_pair(file,'evaluation',cfg);
    processed=zeros(numel(noisy),numel(variants)); validRanges=zeros(numel(variants),2);
    variantIds=cell(1,numel(variants)); updateCounts=zeros(1,numel(variants));
    for j=1:numel(variants)
        v=variants(j);
        [processed(:,j),info]=denoise_speech(noisy,fs,v.method,v.options);
        validRanges(j,:)=info.validSamples; variantIds{j}=v.id;
        if isfield(info,'noiseTracking'), updateCounts(j)=info.noiseTracking.updateCount; end
    end
    signature=protocol.signature;split='evaluation';
    noise=char(manifest.Noise(k));nominalSNR=manifest.NominalSNR(k);matlabVersion=version;
    temporary=[target '.part'];
    save(temporary,'reference','noisy','processed','fs','validRanges','variantIds','updateCounts', ...
        'signature','split','noise','nominalSNR','file','matlabVersion','-v6');
    movefile(temporary,target,'f');
    if mod(k,20)==0 || k==height(manifest), fprintf('Prepared held-out files: %d/240\n',k); end
end
end
