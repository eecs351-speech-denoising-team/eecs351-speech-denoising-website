function [value, backend] = project_stoi(processed, reference, fs, cfg)
% Classic STOI using the pinned pystoi backend on every machine.
% Argument order matches MATLAB stoi(processed,reference,fs).
% MAT interchange preserves double samples; no WAV export/normalization.
% Uses an external process, avoiding MATLAB/Python ABI version constraints.
if nargin < 4, cfg = project_config; end
assert(iscolumn(processed) && isequal(size(processed),size(reference)), ...
    'EECS351:STOIShape', 'STOI inputs must be matching column vectors.');
assert(isfile(cfg.python), 'EECS351:Python', ...
    'Missing project Python. Run python3 -m venv .venv, then install requirements.txt.');
folder = tempname; mkdir(folder);
cleanup = onCleanup(@() rmdir(folder,'s'));
inputPath = fullfile(folder,'input.mat');
outputPath = fullfile(folder,'output.json');
save(inputPath,'processed','reference','fs','-v7');
% A script avoids shell quoting issues with paths, spaces, and Python arguments.
runner = fullfile(folder,'run_stoi.py');
fid = fopen(runner,'w');
assert(fid>0,'EECS351:STOIIO','Could not create the temporary STOI runner.');
fprintf(fid,'import runpy, sys\n');
script = fullfile(cfg.root,'code','score_stoi.py');
fprintf(fid,'sys.path.insert(0, %s)\n',jsonencode(fullfile(cfg.root,'code')));
fprintf(fid,'sys.argv = [%s, "--input-mat", %s, "--output-json", %s]\n', ...
    jsonencode(script), jsonencode(inputPath), jsonencode(outputPath));
fprintf(fid,'runpy.run_path(%s, run_name="__main__")\n',jsonencode(script));
fclose(fid);
% Unix single quotes protect shell metacharacters; Windows uses double quotes.
if ispc
    assert(~contains(cfg.python,'"') && ~contains(runner,'"'), 'Unsupported quote in path.');
    command = sprintf('"%s" "%s"',cfg.python,runner);
else
    quote = @(s) [char(39),strrep(s,char(39),[char(39),char(34),char(39),char(34),char(39)]),char(39)];
    command = [quote(cfg.python) ' ' quote(runner)];
end
[status, message] = system([command ' 2>&1']);
assert(status==0 && isfile(outputPath), 'EECS351:STOIBackend', ...
    'pystoi backend failed. Install requirements.txt in .venv.\n%s',message);
backend = jsondecode(fileread(outputPath));
value = backend.score;
end
