% EECS 351: initial VoiceBank-DEMAND analysis (no denoiser evaluated).
% MATLAB R2024a + Signal Processing Toolbox.
% From the repository root: addpath('code'); analyze_project_revised
% Run prepare_data.py first. Frozen manifests are read, never regenerated.
clear; close all; clc;
cfg = project_config;
out = fullfile(cfg.outputRoot, 'initial_analysis');
if ~exist(out, 'dir'), mkdir(out); end
tuning = cfg.manifests.tuning;
evaluation = cfg.manifests.evaluation;
examples = cfg.manifests.exploratory.File;
conditions = cfg.manifests.exploratory.Noise;

% 3. Extract comparable STFTs from three illustrative test recordings.
fs = cfg.fs; L = cfg.windowLength; hop = cfg.hop; K = cfg.nfft;
w = hann(L, 'periodic');
summary = struct([]);
for c = 1:numel(examples)
    name = examples(c);
    [s,y] = load_speech_pair(name,'exploratory',cfg);
    v = y-s; % Diagnostic residual only; NEVER a denoiser input.
    starts = 1:hop:(numel(s)-L+1);
    idx = (0:L-1)' + starts;
    S = fft(s(idx).*w, K, 1);
    Y = fft(y(idx).*w, K, 1);
    V = fft(v(idx).*w, K, 1);
    S = S(1:K/2+1,:); Y = Y(1:K/2+1,:);
    V = V(1:K/2+1,:);
    tt = ((starts-1)+(L-1)/2)/fs;
    f = (0:K/2)'*fs/K;

    % Reference-derived annotation, for analysis only.
    es = mean(s(idx).^2,1); ev = mean(v(idx).^2,1);
    ratio = 10*log10((es+eps)./(ev+eps));
    mask = ratio <= -10;
    edges = diff([false mask false]);
    a = find(edges==1); b = find(edges==-1)-1;
    assert(~isempty(a), 'No noise-dominant interval found. ');
    [~,j] = max(b-a+1);
    lo = (starts(a(j))-1)/fs;
    hi = (starts(b(j))-1+L)/fs;
    selection = floor(lo*fs)+1:min(numel(s),floor(hi*fs));
    localRatio = 10*log10(sum(s(selection).^2)/sum(v(selection).^2));

    % Unweighted nonnegative-frequency-bin share below 300 Hz.
    pRaw = sum(abs(V).^2,2);
    oldBinShare = sum(pRaw(f<300))/sum(pRaw);
    % Parseval-consistent one-sided spectral power fraction.
    pPower = pRaw;
    pPower(2:end-1) = 2*pPower(2:end-1);
    correctedPowerFraction = sum(pPower(f<300))/sum(pPower);

    fig = figure( 'Visible','off','Color','w','Position',[50 50 1100 880]);
    layout = tiledlayout(4,1, 'TileSpacing','compact','Padding','compact');
    nexttile;
    time = (0:numel(s)-1)'/fs;
    plot(time,y,'Color',[.65 .68 .72]); hold on;
    plot(time,s,'Color',[.03 .28 .55]);
    yl = max(abs([s;y]))*1.08;
    ylim([-yl yl]); xlim([0 time(end)]);
    xline(lo,'--','Color',[.8 .2 .1]);
    xline(hi,'--','Color',[.8 .2 .1]);
    ylabel('Amplitude'); legend( 'Noisy','Clean','Location','northeast');
    title(sprintf('%s | %s | nominal 2.5 dB | %.3f-%.3f s ', ...
    name,conditions(c),lo,hi),'Interpreter','none');
    matrices = {S,Y,V};
    labels = { 'Clean reference ','Original noisy recording ', ...
    'Residual y - s (diagnostic only) '};
    for k = 1:3
        nexttile;
        A = abs(matrices{k})/sum(w);
        A(2:end-1,:) = 2*A(2:end-1,:); % Do not double DC/Nyquist.
        db = 20*log10(max(A,1e-6));
        imagesc(tt,f/1000,db); axis xy;
        ylim([0 8]); xlim([0 time(end)]);
        clim([-90 -10]); colormap(fig,parula(256));
        cb = colorbar; cb.Label.String = 'STFT magnitude (dB re FS) ';
        hold on; xline(lo, 'w--'); xline(hi, 'w--');
        ylabel('Frequency (kHz) '); title(labels{k});
    end
    xlabel('Time (s) '); title(layout, 'Initial spectrogram analysis ');
    exportgraphics(fig,fullfile(out,conditions(c)+'_analysis.png'), ...
    'Resolution',180);
    exportgraphics(fig,fullfile(out,conditions(c)+'_analysis.pdf'), ...
    'ContentType','vector');
    close(fig);

    summary(c).file = char(name);
    summary(c).noise = char(conditions(c));
    summary(c).duration_s = numel(s)/fs;
    summary(c).pause_start_s = lo;
    summary(c).pause_end_s = hi;
    summary(c).pause_speech_to_residual_dB = localRatio;
    summary(c).whole_utterance_SNR_dB = ...
    10*log10(sum(s.^2)/sum(v.^2));
    summary(c).original_unweighted_bin_share_lt300Hz = oldBinShare;
    summary(c).corrected_one_sided_power_fraction_lt300Hz = ...
    correctedPowerFraction;
end
fid = fopen(fullfile(out, 'analysis_summary.json'),'w');
assert(fid>0, 'Could not write summary JSON. ');
fprintf(fid,'%s',jsonencode(summary,PrettyPrint=true));
fclose(fid);
save(fullfile(out,'analysis_results.mat'), ...
'summary','tuning','evaluation');
disp(struct2table(summary));
fprintf('Read %d frozen tuning and %d evaluation entries.\n ', ...
height(tuning),height(evaluation));
