function manifest = run_denoising_demo(outputRoot)
% Generate three exploratory comparisons with candidate default parameters.
% No tuning or held-out evaluation is performed. All five WAVs per utterance
% use the same export gain; unscaled double outputs are retained in signals.mat.
cfg=project_config;
if nargin<1 || isempty(outputRoot), outputRoot=fullfile(cfg.outputRoot,'denoising_demo'); end
if ~isfolder(outputRoot), mkdir(outputRoot); end
methods={'identity','fir','spectral_subtraction','wiener'};
manifest=struct('stage','untuned exploratory demo','fs',cfg.fs,'clips',struct([]));
for k=1:height(cfg.manifests.exploratory)
    file=cfg.manifests.exploratory.File(k);
    [clean,noisy,fs]=load_speech_pair(file,'exploratory',cfg);
    outputs=struct; metadata=struct;
    for j=1:numel(methods)
        method=methods{j};
        [outputs.(method),metadata.(method)]=denoise_speech(noisy,fs,method);
    end
    folder=fullfile(outputRoot,file);
    if ~isfolder(folder), mkdir(folder); end
    peak=max(abs(clean));
    for j=1:numel(methods), peak=max(peak,max(abs(outputs.(methods{j})))); end
    playbackGain=min(1,0.99/max(peak,eps));
    audiowrite(fullfile(folder,'clean.wav'),clean*playbackGain,fs,'BitsPerSample',24);
    for j=1:numel(methods)
        method=methods{j};
        audiowrite(fullfile(folder,[method '.wav']),outputs.(method)*playbackGain,fs,'BitsPerSample',24);
    end
    save(fullfile(folder,'signals.mat'),'clean','noisy','outputs','metadata','fs','playbackGain','-v7');
    clip=struct('file',char(file),'noise',char(cfg.manifests.exploratory.Noise(k)), ...
        'playbackGain',playbackGain,'methods',metadata);
    if k==1, manifest.clips=clip; else, manifest.clips(k)=clip; end
    fprintf('%s: wrote clean + four baseline outputs to %s\n',file,folder);
end
fid=fopen(fullfile(outputRoot,'demo_manifest.json'),'w');
assert(fid>0,'Could not create demo manifest.');
fprintf(fid,'%s\n',jsonencode(manifest,PrettyPrint=true)); fclose(fid);
end
