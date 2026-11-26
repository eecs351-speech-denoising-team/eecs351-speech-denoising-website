function tests = test_data_setup
tests = functiontests(localfunctions);
end

function setup(testCase)
cfg=project_config;
folder=tempname; mkdir(folder);
cfg.dataRoot=folder;
mkdir(fullfile(folder,'clean_testset_wav'));
mkdir(fullfile(folder,'noisy_testset_wav'));
testCase.TestData.cfg=cfg;
testCase.addTeardown(@() rmdir(folder,'s'));
end

function testPairResamplingPreservesLevels(testCase)
cfg=testCase.TestData.cfg;
t=(0:48000)'/48000;
s=0.05*sin(2*pi*180*t);
y=s+0.15*sin(2*pi*37*t);
clean=fullfile(cfg.dataRoot,'clean_testset_wav','p257_004.wav');
noisy=fullfile(cfg.dataRoot,'noisy_testset_wav','p257_004.wav');
audiowrite(clean,s,48000,'BitsPerSample',24);
audiowrite(noisy,y,48000,'BitsPerSample',24);
[actualS,actualY,fs]=load_speech_pair('p257_004','exploratory',cfg);
expected=resample([audioread(clean) audioread(noisy)],1,3);
verifyEqual(testCase,fs,16000);
verifyEqual(testCase,[actualS actualY],expected,'AbsTol',1e-12);
verifyEqual(testCase,numel(actualS),16001);
end

function testMissingPairRejected(testCase)
cfg=testCase.TestData.cfg;
verifyError(testCase,@() load_speech_pair('p257_004','exploratory',cfg),'EECS351:MissingAudio');
end

function testRateAndShapeRejected(testCase)
cfg=testCase.TestData.cfg;
clean=fullfile(cfg.dataRoot,'clean_testset_wav','p257_004.wav');
noisy=fullfile(cfg.dataRoot,'noisy_testset_wav','p257_004.wav');
audiowrite(clean,zeros(4800,1),48000);
audiowrite(noisy,zeros(4800,1),16000);
verifyError(testCase,@() load_speech_pair('p257_004','exploratory',cfg),'EECS351:SampleRate');
audiowrite(noisy,zeros(4800,2),48000);
verifyError(testCase,@() load_speech_pair('p257_004','exploratory',cfg),'EECS351:Shape');
end

function testWrongSplitRejected(testCase)
cfg=testCase.TestData.cfg;
verifyError(testCase,@() load_speech_pair('p257_004','tuning',cfg),'EECS351:File');
end
