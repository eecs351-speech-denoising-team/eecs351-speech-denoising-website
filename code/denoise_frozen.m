function [output, info] = denoise_frozen(y, fs, method, configPath)
% Apply the selected training-only configuration. Never retunes from input.
% Defaults to config/frozen_denoisers.json; method uses denoise_speech names.
root=fileparts(fileparts(mfilename('fullpath')));
if nargin<4 || isempty(configPath), configPath=fullfile(root,'config','frozen_denoisers.json'); end
frozen=jsondecode(fileread(configPath));
assert(strcmp(frozen.status,'FROZEN'),'EECS351:NotFrozen','Configuration is not frozen.');
assert(fs==frozen.protocol.fs,'EECS351:FrozenFs','Frozen settings require %d Hz input.',frozen.protocol.fs);
method=validatestring(method,{'identity','fir','spectral_subtraction','wiener'});
% Refuse to silently use the frozen selection with a changed denoiser.
key=matlab.lang.makeValidName('code/denoise_speech.m');
assert(strcmp(fileHash(fullfile(root,'code','denoise_speech.m')),frozen.protocol.source_sha256.(key)), ...
    'EECS351:FrozenSource','Denoiser changed since tuning; verify or retune before applying this freeze.');
id=frozen.primary.(method);
candidate=frozen.selected(strcmp({frozen.selected.id},id));
assert(isscalar(candidate),'EECS351:FrozenCandidate','Frozen candidate is missing or ambiguous.');
[output,info]=denoise_speech(y,fs,method,candidate.options);
margin=frozen.protocol.common_crop_samples_each_end;
assert(numel(y)>2*margin,'EECS351:FrozenInterval','No common valid samples.');
info.comparisonSamples=[margin+1 numel(y)-margin];
info.frozenCandidate=id;
info.protocolSignature=frozen.protocol.signature;
end

function hash=fileHash(path)
fid=fopen(path,'rb'); assert(fid>0,'Could not read denoiser source.');
cleanup=onCleanup(@() fclose(fid));
bytes=fread(fid,Inf,'*uint8');
md=java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(bytes,'int8'));
hash=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
