function plot_evaluation(outputRoot)
% Four standalone result figures, using held-out tables and existing tuning data.
cfg=project_config;
if nargin<1 || isempty(outputRoot), outputRoot=fullfile(cfg.outputRoot,'evaluation'); end
figures=fullfile(outputRoot,'figures'); if ~isfolder(figures), mkdir(figures); end
overall=readtable(fullfile(outputRoot,'overall.csv'),'TextType','string');
conditions=readtable(fullfile(outputRoot,'by_condition.csv'),'TextType','string');
speakers=readtable(fullfile(outputRoot,'by_speaker.csv'),'TextType','string');
methods=["identity","fir","spectral_subtraction","wiener"];
labels={'Noisy','FIR','Spectral subtraction','Wiener'};
selected=overall([],:);
for method=methods, selected=[selected;overall(overall.method==method & overall.role=="primary",:)]; end %#ok<AGROW>
colors=[.55 .59 .64;.12 .38 .67;.87 .46 .12;.08 .55 .43];
fig=figure('Visible','off','Color','w','Position',[50 50 1400 600]);
layout=tiledlayout(fig,1,2,'TileSpacing','compact');
for j=1:2
    ax=nexttile(layout);
    if j==1
        values=selected.mean_delta_snr_db;lo=selected.delta_snr_db_ci_low;hi=selected.delta_snr_db_ci_high;
        label='Mean SNR change (dB)';
    else
        values=1000*selected.mean_delta_stoi;lo=1000*selected.delta_stoi_ci_low;hi=1000*selected.delta_stoi_ci_high;
        label='Mean STOI change (x 10^{-3})';
    end
    bars=bar(ax,1:4,values,'FaceColor','flat');bars.CData=colors;hold(ax,'on');
    errorbar(ax,1:4,values,max(0,values-lo),max(0,hi-values),'k.','LineWidth',1.5);
    yline(ax,0,'k-');xticks(ax,1:4);xticklabels(ax,labels);xtickangle(ax,15);
    ylabel(ax,label);grid(ax,'on');set(ax,'FontSize',12);
end
title(layout,'Held-out primary methods: 240 paired files');
subtitle(layout,'95% descriptive file-bootstrap intervals; fixed two-speaker/condition mix');
exportgraphics(fig,fullfile(figures,'heldout-overall.png'),'Resolution',180);close(fig);

fig=figure('Visible','off','Color','w','Position',[50 50 1600 900]);
layout=tiledlayout(fig,2,3,'TileSpacing','compact');
noises=["bus","office","cafe"];snrs=[2.5 7.5 12.5 17.5];
matrices=zeros(3,4,3,2);
for m=1:3
    for a=1:3
        for b=1:4
            row=conditions(conditions.method==methods(m+1) & conditions.role=="primary" & ...
                conditions.noise==noises(a) & conditions.nominal_snr_db==snrs(b),:);
            assert(height(row)==1 && row.pairs==20,'Incomplete condition table.');
            matrices(a,b,m,1)=row.mean_delta_snr_db;
            matrices(a,b,m,2)=1000*row.mean_delta_stoi;
        end
    end
end
palette=interp1([0 .5 1],[.8 .2 .12;.98 .98 .98;.12 .37 .7],linspace(0,1,256));
for metric=1:2
    limit=max(abs(matrices(:,:,:,metric)),[],'all');limit=max(limit,eps);
    for m=1:3
        ax=nexttile(layout);values=matrices(:,:,m,metric);
        imagesc(ax,values);clim(ax,[-limit limit]);colormap(ax,palette);colorbar(ax);
        xticks(ax,1:4);xticklabels(ax,string(snrs));yticks(ax,1:3);yticklabels(ax,noises);
        xlabel(ax,'Nominal input SNR (dB)');
        if metric==1, name='SNR change (dB)';else,name='STOI change (x 10^{-3})';end
        title(ax,[labels{m+1} ': ' name]);
        for a=1:3
            for b=1:4
                color='k';if abs(values(a,b))>.65*limit,color='w';end
                text(ax,b,a,sprintf('%+.2f',values(a,b)),'HorizontalAlignment','center','Color',color,'FontSize',11);
            end
        end
        set(ax,'FontSize',11);
    end
end
title(layout,'Held-out changes by noise and SNR: 20 paired files per cell');
exportgraphics(fig,fullfile(figures,'heldout-conditions.png'),'Resolution',180);close(fig);

fig=figure('Visible','off','Color','w','Position',[50 50 1300 550]);
layout=tiledlayout(fig,1,2,'TileSpacing','compact');
for metric=1:2
    values=zeros(2,3);
    for s=1:2
        names=["p232","p257"];
        for m=1:3
            row=speakers(speakers.speaker==names(s) & speakers.method==methods(m+1) & speakers.role=="primary",:);
            assert(height(row)==1 && row.pairs==120,'Incomplete speaker table.');
            if metric==1,values(s,m)=row.mean_delta_snr_db;else,values(s,m)=1000*row.mean_delta_stoi;end
        end
    end
    ax=nexttile(layout);bars=bar(ax,values);
    for m=1:3,bars(m).FaceColor=colors(m+1,:);end
    xticks(ax,1:2);xticklabels(ax,{'p232 (120 files)','p257 (120 files)'});yline(ax,0,'k-');grid(ax,'on');
    if metric==1,ylabel(ax,'Mean SNR change (dB)');else,ylabel(ax,'Mean STOI change (x 10^{-3})');end
    legend(ax,labels(2:4),'Location','best');set(ax,'FontSize',12);
end
title(layout,'Results for each held-out speaker');
exportgraphics(fig,fullfile(figures,'heldout-speakers.png'),'Resolution',180);close(fig);

tuning=readtable(fullfile(cfg.root,'reports','tuning_summary.csv'),'TextType','string');
fig=figure('Visible','off','Color','w','Position',[50 50 1000 650]);ax=axes(fig);hold(ax,'on');
for mode=1:2
    if mode==1,prefix="ss_fixed_strength";else,prefix="ss_pause_updated_strength";end
    x=zeros(1,3);y=x;strengths=[1 2 4];indices=[1 3 5];
    for j=1:3
        row=tuning(tuning.candidate==prefix+string(indices(j)),:);
        x(j)=row.mean_delta_snr_db;y(j)=1000*row.mean_delta_stoi;
    end
    plot(ax,x,y,'-o','LineWidth',2,'MarkerSize',8,'DisplayName',char(replace(prefix,"_strength","")));
    for j=1:3,text(ax,x(j),y(j),sprintf('  alpha=%g',strengths(j)),'FontSize',11);end
end
yline(ax,0,'k:','HandleVisibility','off');xlabel(ax,'Mean SNR change (dB)');ylabel(ax,'Mean STOI change (x 10^{-3})');grid(ax,'on');
legend(ax,'Location','southwest','Interpreter','none');set(ax,'FontSize',12);
title(ax,{'Suppression-strength trade-off (400 tuning files, not held-out scores)', ...
    'Power subtraction: beta=0.01, noise fraction=0.2; other parameters held fixed'});
exportgraphics(fig,fullfile(figures,'tuning-strength-tradeoff.png'),'Resolution',180);close(fig);
end
