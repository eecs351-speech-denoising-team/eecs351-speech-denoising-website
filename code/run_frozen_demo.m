function run_frozen_demo(configPath, outputRoot)
% Training-only artifact review: preselected babble/0, traffic/5, kitchen/15 cells.
% All WAVs use the common comparison interval and one gain per utterance.
cfg=project_config;
if nargin<1 || isempty(configPath), configPath=fullfile(cfg.root,'config','frozen_denoisers.json'); end
if nargin<2 || isempty(outputRoot), outputRoot=fullfile(cfg.outputRoot,'tuning','listening'); end
if ~isfolder(outputRoot), mkdir(outputRoot); end
noises=["babble","traffic","kitchen"]; snrs=[0 5 15];
methods={'identity','fir','spectral_subtraction','wiener'};
labels={'Clean','Noisy','FIR','Spectral subtraction','Wiener'};
fig=figure('Visible','off','Color','w','Position',[50 50 1800 950]);
cleanup=onCleanup(@() close(fig));
layout=tiledlayout(fig,3,5,'TileSpacing','compact','Padding','compact');
review=struct([]);
for k=1:3
    manifest=cfg.manifests.tuning;
    row=find(manifest.Noise==noises(k) & manifest.NominalSNR==snrs(k),1);
    file=manifest.File(row);
    [clean,noisy,fs]=load_speech_pair(file,'tuning',cfg);
    outputs=struct; metadata=struct;
    for j=1:numel(methods)
        name=methods{j};
        [outputs.(name),metadata.(name)]=denoise_frozen(noisy,fs,name,configPath);
    end
    interval=metadata.identity.comparisonSamples;
    keep=interval(1):interval(2);
    signals=[clean(keep),outputs.identity(keep),outputs.fir(keep), ...
        outputs.spectral_subtraction(keep),outputs.wiener(keep)];
    assert(all(isfinite(signals),'all'),'Invalid demo samples.');
    gain=min(1,0.99/max(max(abs(signals),[],'all'),eps));
    folder=fullfile(outputRoot,file);
    if ~isfolder(folder), mkdir(folder); end
    names={'clean','noisy','fir','spectral_subtraction','wiener'};
    for j=1:5
        audiowrite(fullfile(folder,[names{j} '.wav']),signals(:,j)*gain,fs,'BitsPerSample',24);
        ax=nexttile(layout);
        starts=1:160:size(signals,1)-400+1;
        indices=(0:399)'+starts;
        w=hann(400,'periodic');
        signal=signals(:,j);
        spectrum=fft(signal(indices).*w,512,1);
        magnitude=abs(spectrum(1:257,:))/sum(w);
        magnitude(2:end-1,:)=2*magnitude(2:end-1,:);
        imagesc(ax,((starts-1)+199.5)/fs,(0:256)*fs/512/1000,20*log10(max(magnitude,1e-6)));
        axis(ax,'xy'); clim(ax,[-90 -10]); ylim(ax,[0 8]);
        title(ax,labels{j}); xlabel(ax,'Time (s)');
        if j==1, ylabel(ax,sprintf('%s / %g dB\nFrequency (kHz)',noises(k),snrs(k))); end
    end
    save(fullfile(folder,'signals.mat'),'clean','noisy','outputs','metadata','fs','interval','gain','-v7');
    entry=struct('file',char(file),'noise',char(noises(k)),'nominalSNR',snrs(k), ...
        'comparisonSamples',interval,'playbackGain',gain);
    if k==1, review=entry; else, review(k)=entry; end
    fprintf('Frozen review audio: %s\n',folder);
end
title(layout,'Training examples: frozen configurations (common scale and valid interval)');
colormap(fig,parula(256));
exportgraphics(fig,fullfile(outputRoot,'tuning_diagnostics.png'),'Resolution',140);
fid=fopen(fullfile(outputRoot,'review_manifest.json'),'w'); assert(fid>0);
fprintf(fid,'%s\n',jsonencode(review,PrettyPrint=true)); fclose(fid);
end
