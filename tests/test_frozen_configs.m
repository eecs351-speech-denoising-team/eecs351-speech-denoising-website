function tests=test_frozen_configs
tests=functiontests(localfunctions);
end

function testSelectedOutputAndComparisonInterval(testCase)
root=fileparts(fileparts(mfilename('fullpath')));
frozen=jsondecode(fileread(fullfile(root,'config','frozen_denoisers.json')));
y=0.1*sin((1:4301)'*0.07)+0.03*cos((1:4301)'*0.3);
for method={'identity','fir','spectral_subtraction','wiener'}
    name=method{1}; id=frozen.primary.(name);
    candidate=frozen.selected(strcmp({frozen.selected.id},id));
    expected=denoise_speech(y,16000,name,candidate.options);
    [actual,info]=denoise_frozen(y,16000,name);
    verifyEqual(testCase,actual,expected,'AbsTol',1e-14);
    verifyEqual(testCase,info.comparisonSamples,[65 4237]);
    verifyEqual(testCase,info.frozenCandidate,id);
end
end

function testWrongSampleRateRejected(testCase)
verifyError(testCase,@() denoise_frozen(ones(1000,1),48000,'identity'),'EECS351:FrozenFs');
end

function testChangedSourceHashRejected(testCase)
root=fileparts(fileparts(mfilename('fullpath')));
frozen=jsondecode(fileread(fullfile(root,'config','frozen_denoisers.json')));
key=matlab.lang.makeValidName('code/denoise_speech.m');
frozen.protocol.source_sha256.(key)=repmat('0',1,64);
path=temporaryConfig(testCase,frozen);
verifyError(testCase,@() denoise_frozen(ones(1000,1),16000,'identity',path),'EECS351:FrozenSource');
end

function testUnfrozenConfigRejected(testCase)
path=temporaryConfig(testCase,struct('status','DRAFT'));
verifyError(testCase,@() denoise_frozen(ones(1000,1),16000,'identity',path),'EECS351:NotFrozen');
end

function path=temporaryConfig(testCase,value)
path=[tempname '.json'];
fid=fopen(path,'w'); assert(fid>0); fprintf(fid,'%s',jsonencode(value)); fclose(fid);
testCase.addTeardown(@() delete(path));
end
