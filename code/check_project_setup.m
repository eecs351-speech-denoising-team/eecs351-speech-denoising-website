function report = check_project_setup(dataRoot)
% Validate frozen manifests, official logs, all 643 pairs, and runtime.
% From repository root: addpath('code'); check_project_setup
% Writes outputs/setup/{validation_report.json,validated_pairs.csv}.
% This validates setup, not denoiser performance; evaluation scores stay sealed.
if nargin < 1, dataRoot=[]; end
cfg = project_config(dataRoot);
output = fullfile(cfg.outputRoot,'setup');
if ~isfolder(output), mkdir(output); end
reportPath = fullfile(output,'validation_report.json');
report = struct('status','FAIL','matlab',version,'dataRoot',cfg.dataRoot, ...
    'createdUTC',char(datetime('now','TimeZone','UTC','Format',"yyyy-MM-dd'T'HH:mm:ss'Z'")), ...
    'issues',{{}},'validatedPairs',0);
if ~isfile(cfg.python), report.issues{end+1}='Missing .venv Python: install requirements.txt.'; end
for name = {'resample','hann'}
    if isempty(which(name{1})), report.issues{end+1}=['Missing Signal Processing Toolbox function: ' name{1}]; end
end
report.nativeStoiAvailable = ~isempty(which('stoi'));
rows = struct([]);
for split = ["tuning","evaluation","exploratory"]
    manifest = cfg.manifests.(split);
    if split=="tuning", logName='log_trainset_28spk.txt'; else, logName='log_testset.txt'; end
    try
        log = readtable(fullfile(cfg.dataRoot,logName),'FileType','text', ...
            'Delimiter',' ','MultipleDelimsAsOne',true,'ReadVariableNames',false,'TextType','string');
        assert(width(log)==3,'Unexpected condition-log layout.');
        log.Properties.VariableNames={'File','Noise','NominalSNR'};
        assert(numel(unique(log.File))==height(log),'Duplicate file in official log.');
        [found, location] = ismember(manifest.File,log.File);
        assert(all(found),'Manifest includes a file absent from the official log.');
        assert(all(log.Noise(location)==manifest.Noise) && ...
            all(log.NominalSNR(location)==manifest.NominalSNR),'Manifest labels/SNRs disagree with official log.');
    catch error
        report.issues{end+1}=sprintf('%s log: %s',split,error.message);
    end
    for k = 1:height(manifest)
        file = manifest.File(k);
        try
            [s,y,fs,meta] = load_speech_pair(file,split,cfg);
            % One linear antialiasing operator must preserve the relative pair.
            % Check against separately resampled diagnostic residual for one
            % pair per split; the residual never enters a denoiser.
            if k==1
                originalS=audioread(meta.cleanPath); originalY=audioread(meta.noisyPath);
                residual=resample(originalY-originalS,1,3);
                assert(max(abs((y-s)-residual))<1e-10,'Pair resampling is not linear.');
            end
            row=struct('Split',char(split),'File',char(file),'Noise',meta.noise, ...
                'NominalSNR',meta.nominalSNR,'SourceFs',meta.sourceFs, ...
                'SourceSamples',meta.sourceSamples,'Fs',fs,'Samples',numel(s));
            if isempty(rows), rows=row; else, rows(end+1)=row; end %#ok<AGROW>
        catch error
            report.issues{end+1}=sprintf('%s/%s: %s',split,file,error.message);
        end
    end
    fprintf('%s: checked %d pairs\n',split,height(manifest));
end
report.validatedPairs=numel(rows);
try
    % Python validates the complete predeclared condition/speaker balance.
    % Calling through a temporary Python runner avoids shell path escaping.
    runner = fullfile(output,'check_manifests.py');
    fid=fopen(runner,'w'); assert(fid>0,'Could not create manifest check runner.');
    fprintf(fid,'import runpy, sys\nsys.argv = ["prepare_data.py", "--check-manifests"]\n');
    fprintf(fid,'runpy.run_path(%s, run_name="__main__")\n', ...
        jsonencode(fullfile(cfg.root,'code','prepare_data.py')));
    fclose(fid);
    if ispc
        command=sprintf('"%s" "%s"',cfg.python,runner);
    else
        quote=@(s) [char(39),strrep(s,char(39),[char(39),char(34),char(39),char(34),char(39)]),char(39)];
        command=[quote(cfg.python) ' ' quote(runner)];
    end
    [status,message]=system([command ' 2>&1']);
    assert(status==0,'%s',message);
    report.manifestBalance='PASS';
catch error
    report.issues{end+1}=['Manifest balance: ' error.message];
end
try
    % Use only an exploratory clip for the metric smoke test. No held-out
    % intelligibility scores are generated or exposed during setup.
    [s,~,fs]=load_speech_pair('p257_004','exploratory',cfg);
    [identityScore,backend]=project_stoi(s,s,fs,cfg);
    assert(abs(identityScore-1)<1e-8,'Identity STOI must be approximately one.');
    zeroScore=project_stoi(zeros(size(s)),s,fs,cfg);
    assert(abs(zeroScore)<1e-8,'Zero-output STOI must be approximately zero.');
    report.stoiBackend=backend;
    report.stoiIdentity=identityScore;
    report.stoiZeroOutput=zeroScore;
catch error
    report.issues{end+1}=['STOI runtime: ' error.message];
end
if ~isempty(rows), writetable(struct2table(rows),fullfile(output,'validated_pairs.csv')); end
if isempty(report.issues) && report.validatedPairs==643, report.status='PASS'; end
fid=fopen(reportPath,'w'); assert(fid>0,'Could not write setup report.');
fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true)); fclose(fid);
fprintf('%s: %d/643 pairs validated. Report: %s\n',report.status,report.validatedPairs,reportPath);
if ~strcmp(report.status,'PASS')
    fprintf(2,'%s\n',report.issues{:});
    error('EECS351:SetupFailed','Data/environment validation failed; see validation_report.json.');
end
end
