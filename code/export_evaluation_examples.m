function export_evaluation_examples(outputRoot)
% Export frozen primary comparisons for predeclared trials and labeled diagnostics.
cfg=project_config;
if nargin<1 || isempty(outputRoot),outputRoot=fullfile(cfg.outputRoot,'evaluation');end
selection=jsondecode(fileread(fullfile(outputRoot,'case_selection.json')));
cases=selection.cases;files=strings(1,numel(cases));
for k=1:numel(cases)
    if iscell(cases),entry=cases{k};else,entry=cases(k);end
    files(k)=entry.file;
end
files=unique(files,'stable');methods={'identity','fir','spectral_subtraction','wiener'};
names={'clean','noisy','fir','spectral_subtraction','wiener'};
labels={'Clean','Noisy','FIR','Spectral subtraction','Wiener'};
exampleRoot=fullfile(outputRoot,'examples');if ~isfolder(exampleRoot),mkdir(exampleRoot);end
for k=1:numel(files)
    [clean,noisy,fs,meta]=load_speech_pair(files(k),'evaluation',cfg);
    outputs=struct;metadata=struct;
    for j=1:4
        method=methods{j};[outputs.(method),metadata.(method)]=denoise_frozen(noisy,fs,method);
    end
    interval=metadata.identity.comparisonSamples;keep=interval(1):interval(2);
    signals=[clean(keep),outputs.identity(keep),outputs.fir(keep),outputs.spectral_subtraction(keep),outputs.wiener(keep)];
    gain=min(1,0.99/max(max(abs(signals),[],'all'),eps));
    folder=fullfile(exampleRoot,files(k));if ~isfolder(folder),mkdir(folder);end
    for j=1:5,audiowrite(fullfile(folder,[names{j} '.wav']),signals(:,j)*gain,fs,'BitsPerSample',24);end
    save(fullfile(folder,'signals.mat'),'clean','noisy','outputs','metadata','meta','fs','interval','gain','-v7');
    % Every case has its own common-scale five-panel spectrogram for review.
    fig=figure('Visible','off','Color','w','Position',[50 50 1800 450]);
    layout=tiledlayout(fig,1,5,'TileSpacing','compact');
    starts=1:160:size(signals,1)-400+1;ix=(0:399)'+starts;w=hann(400,'periodic');
    for j=1:5
        signal=signals(:,j);spec=fft(signal(ix).*w,512,1);a=abs(spec(1:257,:))/sum(w);a(2:end-1,:)=2*a(2:end-1,:);
        ax=nexttile(layout);imagesc(ax,((starts-1)+199.5)/fs,(0:256)*fs/512/1000,20*log10(max(a,1e-6)));
        axis(ax,'xy');clim(ax,[-90 -10]);ylim(ax,[0 8]);xlabel(ax,'Time (s)');title(ax,labels{j});
        if j==1,ylabel(ax,'Frequency (kHz)');end
    end
    colormap(fig,parula(256));
    title(layout,sprintf('%s | %s | nominal %.1f dB | frozen configurations',files(k),meta.noise,meta.nominalSNR),'Interpreter','none');
    exportgraphics(fig,fullfile(folder,'spectrograms.png'),'Resolution',160);close(fig);
    fprintf('Exported held-out example: %s\n',files(k));
end
end
