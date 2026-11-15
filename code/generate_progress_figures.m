function summary = generate_progress_figures(siteRoot, outputRoot)
% Reproduce the progress-report figures from the three included audio pairs.
% MATLAB with Signal Processing Toolbox is required (resample and hann).
% From the repository root, addpath('code'); then run either:
%   generate_progress_figures
%   generate_progress_figures([], 'C:/path/to/empty/output-folder')
% With no output folder, figures are written to a new temporary directory.
% Existing files are never replaced. The output path is printed at the end.
here = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(siteRoot), siteRoot = fileparts(here); end
if nargin < 2 || isempty(outputRoot), outputRoot = tempname; end
if ~isfolder(outputRoot), mkdir(outputRoot); end
outputNames = ["short-time-rms.png","spectral-overlap.png", ...
    "noise-floor-variation.png","plot-observations.json"];
assert(~any(isfile(fullfile(outputRoot,outputNames))), ...
    'Choose a new output directory: one or more output files already exist.');

dataRoot = fullfile(siteRoot,'data');
audioRoot = fullfile(siteRoot,'assets','audio');
exploratory = readtable(fullfile(dataRoot,'exploratory_manifest.csv'),'TextType','string');
evaluation = readtable(fullfile(dataRoot,'evaluation_manifest.csv'),'TextType','string');
assert(height(exploratory)==3,'Expected three exploratory entries.');
assert(height(evaluation)==240 && numel(unique(evaluation.File))==240, ...
    'Expected 240 distinct held-out pairs.');
files = ["p257_004","p257_017","p257_009"];
labels = ["Bus","Office","Café"];
sourceLabels = ["bus","office","cafe"];
assert(~any(ismember(files,evaluation.File)), ...
    'The exploratory examples must not occur in the held-out manifest.');
fs = 16000;
period = 400;
hop = 160;
nfft = 512;
w = hann(period,'periodic');
freq = (0:nfft/2)'*fs/nfft;
df = fs/nfft;
fw = [1;2*ones(nfft/2-1,1);1];
bandEdges = [0 300 1000 4000];
threshold = 1e-12;
signalNames = ["Clean reference","Noisy mixture","Residual: noisy − clean"];
signalColors = [0.03 0.35 0.58;0.33 0.39 0.48;0.78 0.24 0.11];
bandColors = [0.57 0.19 0.61;0.08 0.49 0.43;0.82 0.36 0.11];

for c = 1:numel(files)
    row = exploratory(exploratory.File==files(c),:);
    assert(height(row)==1 && row.Noise==sourceLabels(c) && row.NominalSNR==2.5, ...
        'Unexpected test condition for %s.',files(c));
    filename = char(files(c)+".wav");
    [s,fsClean] = audioread(fullfile(audioRoot,'clean',filename));
    [y,fsNoisy] = audioread(fullfile(audioRoot,'noisy',filename));
    assert(fsClean==48000 && fsNoisy==fsClean && iscolumn(s) && isequal(size(s),size(y)), ...
        'Expected equally sized mono 48 kHz recordings.');
    s = resample(s,1,3);
    y = resample(y,1,3);
    v = y-s;
    signals = [s,y,v];
    assert(all(isfinite(signals),'all') && max(abs(s+v-y))<1e-14);
    n = size(signals,1);
    starts = 1:hop:n-period+1;
    idx = (0:period-1)' + starts;
    tt = ((starts-1)+(period-1)/2)/fs;

    psdBySignal = zeros(numel(freq),3);
    rmsDb = zeros(numel(tt),3);
    for k = 1:3
        signal = signals(:,k);
        rmsDb(:,k) = (10*log10(max(mean(signal(idx).^2,1),threshold)))';
        frames = signal(idx).*w;
        spec = fft(frames,nfft,1);
        framePSD = fw.*abs(spec(1:nfft/2+1,:)).^2/(fs*sum(w.^2));
        powerFrequency = sum(framePSD,1)*df;
        powerTime = sum(frames.^2,1)/sum(w.^2);
        assert(max(abs(powerFrequency-powerTime))<1e-12,'PSD energy verification failed.');
        psdBySignal(:,k) = mean(framePSD,2);
    end

    % framePSD is now the one-sided estimate for the residual (signal 3).
    powers = zeros(3,numel(tt));
    for b = 1:3
        inBand = freq>=bandEdges(b) & freq<bandEdges(b+1);
        powers(b,:) = sum(framePSD(inBand,:),1)*df;
    end
    assert(all(sum(powers,1)<=powerFrequency+1e-12,'all'));
    bandDbRaw = 10*log10(max(powers,threshold));
    bandDb = movmedian(bandDbRaw,11,2,'Endpoints','shrink');
    cleanFrames = s(idx);
    residualFrames = v(idx);
    ratio = 10*log10((mean(cleanFrames.^2,1)+eps)./(mean(residualFrames.^2,1)+eps));
    edges = diff([false ratio<=-10 false]);
    startsPause = find(edges==1);
    endsPause = find(edges==-1)-1;
    assert(~isempty(startsPause));
    [~,j] = max(endsPause-startsPause);
    markedPause = [(starts(startsPause(j))-1)/fs, (starts(endsPause(j))-1+period)/fs];

    result(c).psd = psdBySignal; %#ok<AGROW>
    result(c).frameTime = tt;
    result(c).rmsDb = rmsDb;
    result(c).bandDb = bandDb;
    result(c).pause = markedPause;
    summary(c).condition = char(sourceLabels(c)); %#ok<AGROW>
    summary(c).file = char(files(c));
    summary(c).nominal_snr_db = row.NominalSNR;
    summary(c).duration_s = n/fs;
    summary(c).pause_s = markedPause;
    summary(c).whole_clip_snr_db = 10*log10(sum(s.^2)/sum(v.^2));
    summary(c).residual_welch_fraction_below_300_hz = ...
        sum(psdBySignal(freq<300,3))/sum(psdBySignal(:,3));
    summary(c).clean_welch_fraction_below_300_hz = ...
        sum(psdBySignal(freq<300,1))/sum(psdBySignal(:,1));
    summary(c).residual_smoothed_band_p10_p90_db = ...
        [pctRows(bandDb,0.1)',pctRows(bandDb,0.9)'];
    summary(c).residual_smoothed_band_span_db = ...
        (pctRows(bandDb,0.9)-pctRows(bandDb,0.1))';
end

% Figure 1: inspect the short-time RMS of every signal type and noise category.
f = newFigure([1180 690]);
layout = tiledlayout(f,3,3,'TileSpacing','compact','Padding','compact');
levels = [-85 -5];
for c=1:3
    for k=1:3
        ax = nexttile(layout);
        hold(ax,'on');
        markPause(ax,result(c).pause,levels);
        plot(ax,result(c).frameTime,result(c).rmsDb(:,k),'Color',signalColors(k,:),'LineWidth',1.2);
        xlim(ax,[0 3.75]); ylim(ax,levels);
        xticks(ax,0:1:3); yticks(ax,[-80 -60 -40 -20]);
        if c==1, title(ax,signalNames(k)); end
        if k==1, ylabel(ax,labels(c)+" RMS (dBFS)"); else, yticklabels(ax,[]); end
        if c==3, xlabel(ax,'Time (s)'); else, xticklabels(ax,[]); end
        formatAxes(ax);
    end
end
title(layout,'What stays loud between words?','FontSize',19,'Color',[.04 .12 .22]);
subtitle(layout,'Short-time RMS  ·  25 ms frames, 10 ms hop  ·  shaded region: weak clean speech relative to residual');
exportFigure(f,outputRoot,outputNames(1));

% Figure 2: compare absolute spectral density on the same scale.
f = newFigure([1180 450]);
layout = tiledlayout(f,1,3,'TileSpacing','compact','Padding','compact');
for c=1:3
    ax = nexttile(layout);
    ax.XScale = 'log';
    hold(ax,'on');
    markPause(ax,[40 300],[-105 -20]);
    handles = gobjects(1,3);
    for k=1:3
        p = 10*log10(max(result(c).psd(:,k),1e-14));
        widths = [1.5,1.05,1.5];
        styles = {'-','--','-'};
        handles(k) = plot(ax,freq(2:end),p(2:end),styles{k}, ...
            'Color',signalColors(k,:),'LineWidth',widths(k));
    end
    xlim(ax,[40 8000]); ylim(ax,[-105 -20]);
    xticks(ax,[50 100 300 1000 3000 8000]);
    xticklabels(ax,{'50','100','300','1k','3k','8k'});
    yticks(ax,[-100 -80 -60 -40 -20]);
    title(ax,labels(c)); xlabel(ax,'Frequency (Hz)');
    if c==1, ylabel(ax,'PSD (dB FS²/Hz)'); else, yticklabels(ax,[]); end
    formatAxes(ax);
end
lg = legend(ax,handles,{'Clean reference','Noisy mixture','Residual'}, ...
    'Orientation','horizontal','Box','off','FontSize',11);
lg.Layout.Tile = 'south';
title(layout,'Where do speech and noise overlap?','FontSize',19,'Color',[.04 .12 .22]);
subtitle(layout,'One-sided Welch PSD  ·  Hann 25 ms, 10 ms hop, 512-point FFT  ·  shaded region: frequencies below 300 Hz');
exportFigure(f,outputRoot,outputNames(2));

% Figure 3: time-varying amplitude of the residual in three frequency bands.
f = newFigure([1180 480]);
layout = tiledlayout(f,1,3,'TileSpacing','compact','Padding','compact');
yBounds = [-95 -10];
for c=1:3
    ax = nexttile(layout);
    hold(ax,'on');
    markPause(ax,result(c).pause,yBounds);
    handles = gobjects(1,3);
    for b=1:3
        handles(b) = plot(ax,result(c).frameTime,result(c).bandDb(b,:), ...
            'Color',bandColors(b,:),'LineWidth',1.6);
    end
    xlim(ax,[0 3.75]); ylim(ax,yBounds); xticks(ax,0:1:3);
    title(ax,labels(c)); xlabel(ax,'Time (s)');
    if c==1, ylabel(ax,'Residual band power (dB FS²)'); else, yticklabels(ax,[]); end
    formatAxes(ax);
end
lg = legend(ax,handles,{'0–300 Hz','300–1,000 Hz','1,000–4,000 Hz'}, ...
    'Orientation','horizontal','Box','off','FontSize',11);
lg.Layout.Tile = 'south';
title(layout,'How much does the background change over time?','FontSize',19,'Color',[.04 .12 .22]);
subtitle(layout,'Band energy from residual STFT frames  ·  11-frame running median  ·  shading as in the RMS figure');
exportFigure(f,outputRoot,outputNames(3));

target = fullfile(outputRoot,outputNames(4));
fh = fopen(target,'w');
assert(fh~=-1,'Could not open %s',target);
cleaner = onCleanup(@()fclose(fh));
fprintf(fh,'%s\n',jsonencode(summary,PrettyPrint=true));
disp(struct2table(summary));
fprintf('Three plots exported to %s. Held-out exclusion and PSD energy checks passed.\n',outputRoot);
end

function f = newFigure(sizeXY)
f = figure('Visible','off','Color','w','Position',[50 50 sizeXY], ...
    'DefaultAxesFontName','Arial','DefaultTextFontName','Arial');
end

function formatAxes(ax)
grid(ax,'on');
ax.Layer = 'top';
ax.TickDir = 'out';
ax.FontSize = 10;
ax.XColor = [.25 .32 .4];
ax.YColor = [.25 .32 .4];
ax.GridAlpha = .12;
ax.Title.Color = [.06 .16 .28];
ax.Title.FontSize = 12;
end

function markPause(ax,span,bounds)
patch(ax,span([1 2 2 1]),bounds([1 1 2 2]),[.89 .94 .99], ...
    'EdgeColor','none','HandleVisibility','off');
end

function exportFigure(f,outputRoot,name)
exportgraphics(f,fullfile(outputRoot,name),'Resolution',150,'BackgroundColor','white');
close(f);
end

function out = pctRows(x,p)
% Linear interpolation of each sorted row; no Statistics Toolbox required.
x = sort(x,2);
n = size(x,2);
i = 1+(n-1)*p;
w = i-floor(i);
out = (1-w)*x(:,floor(i)) + w*x(:,ceil(i));
end
