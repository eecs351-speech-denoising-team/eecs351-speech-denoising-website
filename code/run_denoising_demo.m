function manifest = run_denoising_demo(outputRoot)
% Generate three exploratory comparisons with candidate default parameters.
% No tuning or held-out evaluation is performed. All seven WAVs per utterance
% use the same export gain; unscaled double outputs are retained in signals.mat.
cfg=project_config;
if nargin<1 || isempty(outputRoot), outputRoot=fullfile(cfg.outputRoot,'denoising_demo'); end
if ~isfolder(outputRoot), mkdir(outputRoot); end
methods={'identity','fir','spectral_subtraction','wiener', ...
    'spectral_subtraction_updated','wiener_updated'};
baseMethods={'identity','fir','spectral_subtraction','wiener','spectral_subtraction','wiener'};
manifest=struct('stage','untuned exploratory demo','fs',cfg.fs,'clips',struct([]));
summary=struct([]);
for k=1:height(cfg.manifests.exploratory)
    file=cfg.manifests.exploratory.File(k);
    [clean,noisy,fs]=load_speech_pair(file,'exploratory',cfg);
    outputs=struct; metadata=struct;
    for j=1:numel(methods)
        method=methods{j};
        options=struct;
        if j>4, options.noiseMode='pause_updated'; end
        [outputs.(method),metadata.(method)]=denoise_speech(noisy,fs,baseMethods{j},options);
        if j>=3
            tracking=metadata.(method).noiseTracking;
            row=struct('File',char(file),'Method',baseMethods{j}, ...
                'NoiseMode',metadata.(method).options.noiseMode, ...
                'PauseCandidateFrames',sum(tracking.pauseCandidates), ...
                'UpdateFrames',tracking.updateCount,'EnergyContrastDb',tracking.energyContrastDb, ...
                'FallbackReason',tracking.fallbackReason);
            if isempty(summary), summary=row; else, summary(end+1)=row; end %#ok<AGROW>
        end
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
    fprintf('%s: wrote clean + six comparison outputs to %s\n',file,folder);
end
fid=fopen(fullfile(outputRoot,'demo_manifest.json'),'w');
assert(fid>0,'Could not create demo manifest.');
fprintf(fid,'%s\n',jsonencode(manifest,PrettyPrint=true)); fclose(fid);
writetable(struct2table(summary),fullfile(outputRoot,'noise_update_summary.csv'));
end
